// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tv-core`'s sumcheck checks against `spec/vectors/sumcheck/sumcheck.json`, honest proofs
//! written by an independent Python prover (RFC-0010): every proof verifies, and every value the
//! prover recorded is what the verifier derives, its batching challenge, each round's challenge
//! and claim, each instance's point, each tree level's line challenge and next claims, and the
//! transcript's state after every stage and level.
//!
//! The rounds' claims are reproduced through [`reduce`] on a transcript replayed to the stage's
//! batching challenge, one prefix of the rounds at a time.

mod support;

use support::{Session, TreeOutcome, extension, run_trees, sessions, tree_batches, verify};
use tv_core::sumcheck::{Prefix, reduce};
use tv_core::tree::verify_product_level;
use tv_field::{Fp, Fp2};
use tv_hash::transcript::{Ascii, Transcript};

/// Squeezes `<prefix>/gamma` with the transcript itself: the batching challenge a stage or a
/// level starts with.
fn squeeze_gamma(transcript: &mut Transcript, prefix: Prefix) -> Fp2 {
    let label = format!("{prefix}/gamma");
    transcript
        .challenge_fp2(Ascii::new(&label).unwrap())
        .unwrap()
}

#[test]
fn every_stage_verifies_and_reproduces_every_recorded_value() {
    let sessions = sessions();
    assert_eq!(sessions.len(), 9);
    let mut stages = 0;
    for session in &sessions {
        let mut transcript = session.context.transcript();
        for (index, stage) in session.stages.iter().enumerate() {
            let what = format!("{}, {}", session.name, stage.prefix);
            let batch = verify(&mut transcript, stage, &stage.instances, &stage.proof)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(batch.gamma, stage.gamma, "{what}: γ");
            assert_eq!(batch.point, stage.challenges, "{what}: the challenges");
            for (i, instance) in stage.instances.iter().enumerate() {
                assert_eq!(
                    batch.point_of(instance.vars),
                    Some(&stage.points[i][..]),
                    "{what}: instance {i}'s point"
                );
            }
            assert_eq!(
                *transcript.state(),
                stage.state,
                "{what}: the state after it"
            );
            assert_eq!(transcript.counter(), stage.counter, "{what}: the squeezes");
            reproduce_claims(session, index);
            stages += 1;
        }
    }
    assert_eq!(stages, 62);
}

/// The claims of stage `index` of `session`: the batched claim, computed here from its
/// definition, then the claim after each prefix of the rounds, through [`reduce`]; the last is
/// the relations' batched value.
fn reproduce_claims(session: &Session, index: usize) {
    let stage = &session.stages[index];
    let what = format!("{}, {}", session.name, stage.prefix);
    let n_max = stage.rounds();
    let mut weight = Fp2::ONE;
    let mut batched = Fp2::ZERO;
    let mut last = Fp2::ZERO;
    let mut evals = stage.proof.final_evals.iter();
    for (i, instance) in stage.instances.iter().enumerate() {
        let scale = Fp::reduce(2).pow((n_max - instance.vars) as u64);
        batched += weight * instance.claim * scale;
        let value = evals
            .by_ref()
            .take(instance.evals)
            .fold(Fp2::ONE, |p, e| p * *e);
        assert_eq!(value, stage.values[i], "{what}: instance {i}'s value");
        last += weight * value;
        weight *= stage.gamma;
    }
    assert_eq!(stage.claims[0], batched, "{what}: claim_1");
    assert_eq!(stage.claims[n_max], last, "{what}: the last claim");
    for rounds in 0..=n_max {
        let mut transcript = session.transcript_before(index);
        assert_eq!(squeeze_gamma(&mut transcript, stage.prefix), stage.gamma);
        let reduction = reduce(
            &mut transcript,
            stage.prefix,
            stage.degree,
            rounds,
            batched,
            &stage.proof.rounds[..rounds],
        )
        .unwrap_or_else(|e| panic!("{what}, {rounds} rounds: {e}"));
        assert_eq!(reduction.point, stage.challenges[..rounds], "{what}");
        assert_eq!(
            reduction.claim, stage.claims[rounds],
            "{what}: claim after {rounds} rounds"
        );
    }
}

#[test]
fn every_tree_batch_verifies_and_reproduces_every_recorded_value() {
    let batches = tree_batches();
    assert_eq!(batches.len(), 10);
    let mut levels = 0;
    for batch in &batches {
        let mut transcript = batch.context.transcript();
        let run = run_trees(&mut transcript, &batch.trees, &batch.proofs(), |_, _| {});
        assert_eq!(run.outcome, TreeOutcome::Accepted, "{}", batch.name);
        assert_eq!(run.levels.len(), batch.levels.len());
        for (index, (ran, recorded)) in run.levels.iter().zip(&batch.levels).enumerate() {
            let what = format!("{}, {}", batch.name, recorded.prefix);
            assert_eq!(recorded.prefix, Prefix::Tree(index as u32));
            assert_eq!(
                ran.entering, recorded.entering,
                "{what}: the claims entering it"
            );
            assert_eq!(ran.outcome.gamma, recorded.gamma, "{what}: γ");
            assert_eq!(
                ran.outcome.point, recorded.challenges,
                "{what}: the challenges"
            );
            assert_eq!(ran.outcome.rho, recorded.rho, "{what}: ρ");
            let next: Vec<_> = recorded
                .next
                .iter()
                .map(|(_, claim)| claim.clone())
                .collect();
            assert_eq!(ran.outcome.claims, next, "{what}: the next claims");
            assert_eq!(ran.state, recorded.state, "{what}: the state after it");
            assert_eq!(ran.counter, recorded.counter, "{what}: the squeezes");
            reproduce_level_claims(batch, index);
            levels += 1;
        }
        // The leaf claims are the last level's, and the leaves' extensions there.
        let last = &batch.levels.last().unwrap().next;
        let mut leaves = run.leaves.clone();
        leaves.sort_by_key(|(tree, _)| *tree);
        let mut recorded = last.clone();
        recorded.sort_by_key(|(tree, _)| *tree);
        assert_eq!(leaves, recorded, "{}: the leaf claims", batch.name);
        for (tree, claim) in &leaves {
            assert_eq!(
                extension(&batch.trees[*tree].leaves, &claim.point),
                claim.value
            );
            assert_eq!(claim.point.len(), batch.trees[*tree].height);
        }
    }
    assert_eq!(levels, 39);
}

/// The claims of level `index` of `batch`, round by round, as for a stage.
fn reproduce_level_claims(batch: &support::TreeBatch, index: usize) {
    let level = &batch.levels[index];
    let n_max = level.rounds();
    let mut weight = Fp2::ONE;
    let mut batched = Fp2::ZERO;
    for (_, claim) in &level.entering {
        let scale = Fp::reduce(2).pow((n_max - claim.point.len()) as u64);
        batched += weight * claim.value * scale;
        weight *= level.gamma;
    }
    assert_eq!(
        level.claims[0], batched,
        "{}, level {index}: claim_1",
        batch.name
    );
    for rounds in 0..=n_max {
        let mut transcript = batch.context.transcript();
        for (earlier, recorded) in batch.levels[..index].iter().enumerate() {
            let claims: Vec<_> = recorded.entering.iter().map(|(_, c)| c.clone()).collect();
            // Only the transcript an earlier level leaves matters here.
            let _ = verify_product_level(&mut transcript, earlier as u32, &claims, &recorded.proof)
                .unwrap();
        }
        assert_eq!(squeeze_gamma(&mut transcript, level.prefix), level.gamma);
        let reduction = reduce(
            &mut transcript,
            level.prefix,
            3,
            rounds,
            batched,
            &level.proof.rounds[..rounds],
        )
        .unwrap();
        assert_eq!(reduction.point, level.challenges[..rounds]);
        assert_eq!(
            reduction.claim, level.claims[rounds],
            "{}, level {index}",
            batch.name
        );
    }
}

/// The trees' roots are their leaves' products, and exactly the trees with a zero leaf have
/// root 0, as SoftmaxRow's do (RFC-0010 'Tree levels').
#[test]
fn every_root_is_the_product_of_its_leaves() {
    let mut zero_roots = 0;
    for batch in tree_batches() {
        for tree in &batch.trees {
            assert_eq!(tree.leaves.len(), 1 << tree.height);
            let product = tree.leaves.iter().fold(Fp::ONE, |p, leaf| p * *leaf);
            assert_eq!(tree.root, product, "{}", batch.name);
            let has_zero = tree.leaves.contains(&Fp::ZERO);
            assert_eq!(tree.root == Fp::ZERO, has_zero);
            zero_roots += usize::from(has_zero);
        }
    }
    assert_eq!(zero_roots, 3);
}

/// The vectors hold what RFC-0010's testing strategy and #59 ask for: single products of 1 to
/// 4 tables on 0 to 12 variables; batches mixing 3, 7 and 12 variables; a batch of `n_max = 0`;
/// trees of heights 1 to 6, joining at the level where their heights match.
#[test]
fn the_vectors_cover_the_shapes_the_issue_asks_for() {
    let sessions = sessions();
    let mut singles = std::collections::BTreeSet::new();
    let mut mixed = false;
    let mut constants = false;
    for stage in sessions.iter().flat_map(|s| &s.stages) {
        let vars: Vec<usize> = stage.instances.iter().map(|i| i.vars).collect();
        if let [instance] = &stage.instances[..] {
            singles.insert((instance.evals, instance.vars));
        }
        mixed |= [3, 7, 12].iter().all(|n| vars.contains(n));
        constants |= stage.instances.len() > 1 && vars.iter().all(|&n| n == 0);
    }
    for factors in 1..=4 {
        for vars in 0..=12 {
            assert!(
                singles.contains(&(factors, vars)),
                "{factors} tables, {vars} variables"
            );
        }
    }
    assert!(mixed && constants);
    let batches = tree_batches();
    let mut heights = std::collections::BTreeSet::new();
    let mut joins = false;
    for batch in &batches {
        let tallest = batch.trees.iter().map(|t| t.height).max().unwrap();
        heights.extend(batch.trees.iter().map(|t| t.height));
        joins |= batch.trees.iter().any(|t| t.height < tallest);
        assert_eq!(batch.levels.len(), tallest);
    }
    assert_eq!(heights, (1..=6).collect());
    assert!(joins);
}
