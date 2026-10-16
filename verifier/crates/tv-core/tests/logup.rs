// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tv-core`'s LogUp-GKR checks on the `logup_batches` of `spec/vectors/sumcheck/`, written by an
//! independent Python prover from RFC-0008 'Fraction tree' and RFC-0010 'Tree levels' (#62).
//!
//! Each batch is one leaf stage on one transcript: `α` and `β`, the roots of every instance's
//! lookup tree and table tree, the levels where those trees and the product forests run
//! together, the values, and the leaf checks. `support::logup::run_logup` plays the verifier's
//! driver, written from the RFC text: it squeezes `α` and `β` and checks the roots through
//! [`check_roots`](tv_core::logup::check_roots), joins each tree where its height says, with the
//! forests' root challenges and root claims, checks each level through
//! [`verify_level`](tv_core::tree::verify_level), computes `t̃(ρ_t)` from the table's rows, and
//! runs [`check_leaves`](tv_core::logup::check_leaves).
//!
//! - Every honest batch verifies and reproduces every value the prover recorded: `α`, `β`, each
//!   level's root challenges, entering claims, `γ`, `λ`, its rounds' challenges and claims, `ρ`,
//!   next claims and transcript, the leaf checks' claims and table evaluations, and the
//!   transcript after the values. Every claim of every level is its tree level's extension at its
//!   point, the trees built here from the rows, lookups and multiplicities, and every claim the
//!   leaf checks leave is true.
//! - Every cheating batch is rejected at exactly the check its expectation names.
//! - The batches cover what #62 asks for, read from the data.

mod support;

use support::Forest;
use support::logup::{
    Expected, Kind, LogUpBatch, LogUpOutcome, Lookup, TreeRef, extension2, fraction_levels,
    logup_batches,
};
use tv_core::logup::LookupError;
use tv_core::sumcheck::{Prefix, reduce};
use tv_core::tree::{LevelClaim, verify_level};
use tv_field::{Fp, Fp2};
use tv_hash::transcript::{Ascii, Transcript};

fn batches() -> Vec<LogUpBatch> {
    logup_batches().expect("`logup_batches` in spec/vectors/sumcheck/sumcheck.json")
}

fn honest() -> Vec<LogUpBatch> {
    batches()
        .into_iter()
        .filter(|batch| batch.expected == Expected::Accept)
        .collect()
}

fn challenge(transcript: &mut Transcript, label: &str) -> Fp2 {
    transcript
        .challenge_fp2(Ascii::new(label).unwrap())
        .unwrap()
}

/// `eq(a, b)` from its definition.
fn eq(a: &[Fp2], b: &[Fp2]) -> Fp2 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(Fp2::ONE, |product, (&x, &y)| {
        product * (x * y + (Fp2::ONE - x) * (Fp2::ONE - y))
    })
}

/// The point of a level claim.
fn point_of(claim: &LevelClaim) -> &[Fp2] {
    match claim {
        LevelClaim::Product(claim) => &claim.point,
        LevelClaim::Fraction(claim) => &claim.point,
    }
}

#[test]
fn every_honest_logup_batch_verifies_and_reproduces_every_recorded_value() {
    let batches = honest();
    assert_eq!(batches.len(), 10);
    let mut levels = 0;
    for batch in &batches {
        let name = &batch.name;
        let run = batch.run();
        assert_eq!(run.outcome, LogUpOutcome::Accepted, "{name}");
        let roots = run.roots.as_ref().unwrap();
        assert_eq!(roots.challenges.alpha, batch.alpha, "{name}: α");
        assert_eq!(roots.challenges.beta, batch.beta, "{name}: β");
        assert_eq!(run.levels.len(), batch.levels.len(), "{name}");
        for (index, (ran, recorded)) in run.levels.iter().zip(&batch.levels).enumerate() {
            let what = format!("{name}, {}", recorded.prefix);
            assert_eq!(recorded.prefix, Prefix::Tree(index as u32), "{what}");
            assert_eq!(ran.root_challenges, recorded.root_challenges, "{what}");
            assert_eq!(
                ran.entering, recorded.entering,
                "{what}: the claims entering it"
            );
            assert_eq!(ran.outcome.gamma, recorded.gamma, "{what}: γ");
            assert_eq!(ran.outcome.lambda, recorded.lambda, "{what}: λ");
            assert_eq!(
                ran.outcome.point, recorded.challenges,
                "{what}: the challenges"
            );
            assert_eq!(ran.outcome.rho, recorded.rho, "{what}: ρ");
            let next: Vec<_> = ran
                .entering
                .iter()
                .map(|(tree, _)| *tree)
                .zip(ran.outcome.claims.iter().cloned())
                .collect();
            assert_eq!(next, recorded.next, "{what}: the next claims");
            assert_eq!(ran.state, recorded.state, "{what}: the state after it");
            assert_eq!(ran.counter, recorded.counter, "{what}: the squeezes");
            reproduce_level_claims(batch, index);
            levels += 1;
        }
        // The leaf checks: each instance's leaf claims, its value, its key and multiplicity
        // claims, and the table evaluation computed here from the rows.
        let claims = run.claims.as_ref().unwrap();
        assert_eq!(batch.leaf_checks.len(), batch.instances.len(), "{name}");
        let mut values = batch.values.iter();
        for (i, (check, instance)) in batch.leaf_checks.iter().zip(&batch.instances).enumerate() {
            let what = format!("{name}, instance {i}");
            let leaf = |kind: Kind| {
                let t = batch
                    .trees
                    .iter()
                    .position(|tree| *tree == TreeRef { kind, index: i })
                    .unwrap();
                match &run.leaves[t] {
                    LevelClaim::Fraction(claim) => claim.clone(),
                    LevelClaim::Product(_) => panic!("{what}: a fraction tree"),
                }
            };
            let (lookup, table) = (leaf(Kind::Lookup), leaf(Kind::Table));
            assert_eq!(lookup.point, check.lookup.point, "{what}");
            assert_eq!(lookup.numerator, check.lookup.numerator, "{what}");
            assert_eq!(lookup.numerator, Fp2::ONE, "{what}");
            assert_eq!(lookup.denominator, check.lookup.denominator, "{what}");
            let sent = if instance.has_values {
                values.next().copied()
            } else {
                None
            };
            assert_eq!(check.lookup.value, sent, "{what}: v");
            let v = sent.unwrap_or(Fp2::ZERO);
            assert_eq!(
                check.lookup.key,
                batch.alpha - lookup.denominator - batch.beta * v,
                "{what}: the key claim"
            );
            assert_eq!(table.point, check.table.point, "{what}");
            assert_eq!(table.numerator, check.table.numerator, "{what}");
            assert_eq!(check.table.multiplicity, -table.numerator, "{what}");
            assert_eq!(table.denominator, check.table.denominator, "{what}");
            let evaluation = instance.table_eval(batch.beta, &table.point);
            assert_eq!(evaluation, check.table.table_eval, "{what}: t̃(ρ_t)");
            assert_eq!(run.table_evals[i], evaluation, "{what}");
            assert_eq!(
                table.denominator,
                batch.alpha - evaluation,
                "{what}: D̃(ρ_t) = α − t̃(ρ_t)"
            );
            let left = &claims.instances[i];
            assert_eq!(left.key.value, check.lookup.key, "{what}");
            assert_eq!(left.key.point, check.lookup.point, "{what}");
            assert_eq!(left.value.as_ref().map(|claim| claim.value), sent, "{what}");
            assert_eq!(left.multiplicity.value, check.table.multiplicity, "{what}");
            assert_eq!(left.multiplicity.point, check.table.point, "{what}");
        }
        assert!(values.next().is_none(), "{name}: every value taken");
        assert_eq!(run.state, batch.state, "{name}: the state after the values");
        assert_eq!(run.counter, batch.counter, "{name}: the squeezes");
    }
    // 3 + 0 + 2 + 3 + 4 + 10 + 2 + 2 + 2 + 1.
    assert_eq!(levels, 29);
}

/// The levels `run` passed are the ones `batch` recorded, value for value.
fn assert_levels_reproduced(batch: &LogUpBatch, run: &support::logup::LogUpRun) {
    for (index, (ran, recorded)) in run.levels.iter().zip(&batch.levels).enumerate() {
        let what = format!("{}, level {index}", batch.name);
        assert_eq!(ran.root_challenges, recorded.root_challenges, "{what}");
        assert_eq!(ran.entering, recorded.entering, "{what}");
        assert_eq!(ran.outcome.gamma, recorded.gamma, "{what}");
        assert_eq!(ran.outcome.lambda, recorded.lambda, "{what}");
        assert_eq!(ran.outcome.point, recorded.challenges, "{what}");
        assert_eq!(ran.outcome.rho, recorded.rho, "{what}");
        let next: Vec<(usize, LevelClaim)> = ran
            .entering
            .iter()
            .map(|(tree, _)| *tree)
            .zip(ran.outcome.claims.iter().cloned())
            .collect();
        assert_eq!(next, recorded.next, "{what}");
        assert_eq!(
            (ran.state, ran.counter),
            (recorded.state, recorded.counter),
            "{what}"
        );
    }
}

/// The first root check that fails, computed here from RFC-0008 'Fraction tree': instance by
/// instance, `d_a ≠ 0`, `d_t ≠ 0`, then `n_a·d_t + n_t·d_a = 0`.
fn root_failure(roots: &[Fp2]) -> Option<LookupError> {
    roots
        .chunks_exact(4)
        .enumerate()
        .find_map(|(instance, block)| {
            let (n_a, d_a, n_t, d_t) = (block[0], block[1], block[2], block[3]);
            if d_a == Fp2::ZERO {
                Some(LookupError::LookupDenominator { instance })
            } else if d_t == Fp2::ZERO {
                Some(LookupError::TableDenominator { instance })
            } else if n_a * d_t + n_t * d_a != Fp2::ZERO {
                Some(LookupError::Unbalanced { instance })
            } else {
                None
            }
        })
}

/// A transcript that has run `batch`'s roots and its levels before `index`, each verified with
/// its recorded claims, then squeezed level `index`'s root challenges, `γ` and, when it has one,
/// `λ`: the transcript its rounds start from.
fn transcript_before_rounds(batch: &LogUpBatch, index: usize) -> Transcript {
    let mut transcript = batch.context.transcript();
    let roots = tv_core::logup::check_roots(&mut transcript, batch.instances.len(), &batch.roots);
    assert!(roots.is_ok(), "{}: {roots:?}", batch.name);
    for (level, recorded) in batch.levels.iter().take(index + 1).enumerate() {
        for c in 0..recorded.root_challenges.len() {
            let squeezed = challenge(&mut transcript, &format!("tree/{level}/root/{c}"));
            assert_eq!(squeezed, recorded.root_challenges[c]);
        }
        if level < index {
            let claims: Vec<LevelClaim> = recorded
                .entering
                .iter()
                .map(|(_, claim)| claim.clone())
                .collect();
            let passed = verify_level(&mut transcript, level as u32, &claims, &recorded.proof);
            assert!(passed.is_ok(), "{}, level {level}: {passed:?}", batch.name);
        }
    }
    let prefix = batch.levels[index].prefix;
    assert_eq!(
        challenge(&mut transcript, &format!("{prefix}/gamma")),
        batch.levels[index].gamma
    );
    if let Some(lambda) = batch.levels[index].lambda {
        assert_eq!(
            challenge(&mut transcript, &format!("{prefix}/lambda")),
            lambda
        );
    }
    transcript
}

/// The claims of level `index`: the batched claim computed here, `Σ_i γ^i·2^{n_max − n_i}·c_i`
/// with `c_i = v_i` or `p_i + λ·q_i`; the claim after each prefix of the rounds, through
/// [`reduce`]; and the last claim from the final evaluations, `Σ_i γ^i·eq(r_i, r'_i)·g_i`, with
/// `g_i = u_i·w_i` or `N_0·D_1 + N_1·D_0 + λ·D_0·D_1`, `eq` computed here.
fn reproduce_level_claims(batch: &LogUpBatch, index: usize) {
    let level = &batch.levels[index];
    let what = format!("{}, {}", batch.name, level.prefix);
    let n_max = level.rounds();
    let lambda = level.lambda.unwrap_or(Fp2::ZERO);
    let mut weight = Fp2::ONE;
    let mut batched = Fp2::ZERO;
    let mut last = Fp2::ZERO;
    let mut evals = level.proof.final_evals.iter().copied();
    for (_, claim) in &level.entering {
        let n_i = point_of(claim).len();
        assert!(n_i <= n_max, "{what}");
        let own = &level.challenges[n_max - n_i..];
        let scale = Fp::reduce(2).pow((n_max - n_i) as u64);
        let (entering, g) = match claim {
            LevelClaim::Product(claim) => {
                let (u, w) = (evals.next().unwrap(), evals.next().unwrap());
                (claim.value, u * w)
            }
            LevelClaim::Fraction(claim) => {
                assert!(level.lambda.is_some(), "{what}: λ with a fraction tree");
                let [n0, n1, d0, d1] = [(); 4].map(|_| evals.next().unwrap());
                (
                    claim.numerator + lambda * claim.denominator,
                    n0 * d1 + n1 * d0 + lambda * d0 * d1,
                )
            }
        };
        batched += weight * entering * scale;
        last += weight * eq(point_of(claim), own) * g;
        weight *= level.gamma;
    }
    assert!(evals.next().is_none(), "{what}: every evaluation used");
    assert_eq!(level.claims.len(), n_max + 1, "{what}");
    assert_eq!(level.claims[0], batched, "{what}: claim_1");
    assert_eq!(level.claims[n_max], last, "{what}: the last claim");
    for rounds in 0..=n_max {
        let mut transcript = transcript_before_rounds(batch, index);
        let reduction = reduce(
            &mut transcript,
            level.prefix,
            3,
            rounds,
            batched,
            &level.proof.rounds[..rounds],
        )
        .map(|reduction| (reduction.point, reduction.claim));
        assert_eq!(
            reduction,
            Ok((level.challenges[..rounds].to_vec(), level.claims[rounds])),
            "{what}: after {rounds} rounds"
        );
    }
}

/// A batch's trees, built here: each instance's fraction trees from its rows, lookups and
/// multiplicities, and each forest's levels from its stacked leaves; level `ℓ` of a tree is
/// `levels[ℓ]`, as `(numerator, denominator)` pairs or as a forest's values.
enum Built {
    Fraction(Vec<Vec<(Fp2, Fp2)>>),
    Forest(Vec<Vec<Fp>>),
}

fn build(batch: &LogUpBatch) -> Vec<Built> {
    batch
        .trees
        .iter()
        .map(|tree| match tree.kind {
            Kind::Lookup => Built::Fraction(fraction_levels(
                batch.instances[tree.index].lookup_leaves(batch.alpha, batch.beta),
            )),
            Kind::Table => Built::Fraction(fraction_levels(
                batch.instances[tree.index].table_leaves(batch.alpha, batch.beta),
            )),
            Kind::Forest => Built::Forest(forest_levels(&batch.forests[tree.index])),
        })
        .collect()
}

/// A forest's levels from its stacked leaves, the root level first: `F_ℓ[k] = F_{ℓ+1}[2k]·F_{ℓ+1}[2k + 1]`.
fn forest_levels(forest: &Forest) -> Vec<Vec<Fp>> {
    let mut levels = vec![forest.leaves.clone()];
    for _ in 0..forest.height {
        let above: Vec<Fp> = levels[0]
            .chunks_exact(2)
            .map(|pair| pair[0] * pair[1])
            .collect();
        levels.insert(0, above);
    }
    levels
}

/// Whether `claim` is level `own` of `built` at its point.
fn holds(built: &Built, own: usize, claim: &LevelClaim) -> bool {
    match (built, claim) {
        (Built::Fraction(levels), LevelClaim::Fraction(claim)) => {
            let n: Vec<Fp2> = levels[own].iter().map(|&(n, _)| n).collect();
            let d: Vec<Fp2> = levels[own].iter().map(|&(_, d)| d).collect();
            claim.point.len() == own
                && extension2(&n, &claim.point) == claim.numerator
                && extension2(&d, &claim.point) == claim.denominator
        }
        (Built::Forest(levels), LevelClaim::Product(claim)) => {
            levels[own].len() == 1 << claim.point.len()
                && support::extension(&levels[own], &claim.point) == claim.value
        }
        _ => false,
    }
}

/// Every claim of the honest vectors is true: the roots message is the trees' roots, built here
/// from the instances; each claim entering a level and each next claim is its tree level's
/// extension at its point; and every leaf claim is the leaves'. The instances are honest: every
/// lookup is a row, and the multiplicities count them, row 0's with the padding rows', which are
/// its copies, also counting the lookup tree's padding.
#[test]
fn every_claim_is_its_trees_level_extension_at_its_point() {
    let mut claims = 0;
    for batch in honest() {
        let name = &batch.name;
        for (i, instance) in batch.instances.iter().enumerate() {
            let rows = instance.rows.len();
            let mut counts = vec![0u64; rows];
            for lookup in &instance.lookups {
                let Lookup::Row(j) = *lookup else {
                    panic!("{name}, instance {i}: an honest lookup is a row");
                };
                assert!(j < rows, "{name}, instance {i}: a row");
                counts[j] += 1;
            }
            counts[0] += ((1 << instance.lookup_height()) - instance.lookups.len()) as u64;
            // Row 0 and its copies share row 0's count.
            let row_zero = instance.multiplicities[rows..]
                .iter()
                .fold(instance.multiplicities[0], |sum, &m| sum + m);
            assert_eq!(
                row_zero,
                Fp::reduce(counts[0]),
                "{name}, instance {i}: row 0"
            );
            for (j, (&m, &count)) in instance
                .multiplicities
                .iter()
                .zip(&counts)
                .enumerate()
                .skip(1)
            {
                assert_eq!(m, Fp::reduce(count), "{name}, instance {i}: row {j}");
            }
            if !instance.has_values {
                assert!(instance.rows.iter().all(|&(_, value)| value == Fp::ZERO));
            }
        }
        let built = build(&batch);
        let heights = batch.heights();
        let tallest = heights.iter().copied().max().unwrap_or(0);
        for (t, (tree, built)) in batch.trees.iter().zip(&built).enumerate() {
            if let (Kind::Lookup | Kind::Table, Built::Fraction(levels)) = (tree.kind, built) {
                let side = if tree.kind == Kind::Lookup { 0 } else { 2 };
                let (n, d) = levels[0][0];
                assert_eq!(
                    batch.roots[4 * tree.index + side..4 * tree.index + side + 2],
                    [n, d],
                    "{name}, tree {t}: its root"
                );
            }
        }
        for (index, level) in batch.levels.iter().enumerate() {
            for (below, list) in [(0, &level.entering), (1, &level.next)] {
                for (t, claim) in list {
                    // The tree's own level: a forest's counts from its root level of 2^s
                    // entries.
                    let own = index - (tallest - heights[*t]) + below;
                    assert!(
                        holds(&built[*t], own, claim),
                        "{name}, level {index}, tree {t}"
                    );
                    claims += 1;
                }
            }
        }
        let run = batch.run();
        for (t, leaf) in run.leaves.iter().enumerate() {
            assert!(
                holds(&built[t], heights[t], leaf),
                "{name}, tree {t}'s leaves"
            );
        }
    }
    assert!(claims > 0);
}

/// The outcome a cheating batch's expectation names.
fn matches_expectation(outcome: &LogUpOutcome, reason: &str, check: &str, instance: usize) -> bool {
    match (reason, check, outcome) {
        ("Lookup", "roots", LogUpOutcome::Roots(error)) => {
            matches!(
                error,
                LookupError::LookupDenominator { .. }
                    | LookupError::TableDenominator { .. }
                    | LookupError::Unbalanced { .. }
            ) && error.rejected_instance() == Some(instance)
        }
        ("Lookup", "leaf numerator", LogUpOutcome::Leaves(error)) => {
            *error == LookupError::LeafNumerator { instance }
        }
        ("Lookup", "table evaluation", LogUpOutcome::Leaves(error)) => {
            *error == LookupError::TableEvaluation { instance }
        }
        ("Sumcheck", check, LogUpOutcome::Level { level, error }) => {
            check == format!("level {level}") && error.rejected_round().is_some()
        }
        _ => false,
    }
}

/// Every cheating batch is rejected at exactly the check its expectation names, with the reason
/// and the instance it gives, and nothing after that check runs.
#[test]
fn every_cheating_batch_is_rejected_at_its_expected_check() {
    let mut cheating = 0;
    for batch in batches() {
        let Expected::Reject {
            reason,
            check,
            instance,
        } = &batch.expected
        else {
            continue;
        };
        let name = &batch.name;
        let run = batch.run();
        assert!(
            matches_expectation(&run.outcome, reason, check, *instance),
            "{name}: expected {reason}, {check}, instance {instance}; got {:?}",
            run.outcome
        );
        // What a verifier derives before the check that fails is what the batch records.
        let mut transcript = batch.context.transcript();
        assert_eq!(
            challenge(&mut transcript, "logup/alpha"),
            batch.alpha,
            "{name}"
        );
        assert_eq!(
            challenge(&mut transcript, "logup/beta"),
            batch.beta,
            "{name}"
        );
        assert_levels_reproduced(&batch, &run);
        match &run.outcome {
            LogUpOutcome::Roots(error) => {
                // The exact root check that fails, from the message.
                assert_eq!(Some(*error), root_failure(&batch.roots), "{name}");
                assert!(run.levels.is_empty());
            }
            LogUpOutcome::Level { level, .. } => {
                assert_eq!(root_failure(&batch.roots), None, "{name}: the roots pass");
                assert_eq!(run.levels.len(), *level as usize);
                assert!(run.claims.is_none());
                // The claims entering the rejected level are the recorded ones, and the one at
                // the expectation's place is a root the message falsified: on no variable, and
                // not the root of the tree built here from the prover's instance.
                let recorded = &batch.levels[*level as usize];
                assert_eq!(run.rejected, recorded.entering, "{name}");
                let (tree, LevelClaim::Fraction(claim)) = &recorded.entering[*instance] else {
                    panic!("{name}: a fraction tree's root");
                };
                assert!(claim.point.is_empty(), "{name}: a root");
                let Built::Fraction(levels) = &build(&batch)[*tree] else {
                    panic!("{name}: a fraction tree");
                };
                assert_ne!((claim.numerator, claim.denominator), levels[0][0], "{name}");
            }
            LogUpOutcome::Leaves(error) => {
                assert_eq!(root_failure(&batch.roots), None, "{name}: the roots pass");
                let tallest = batch.heights().iter().copied().max().unwrap_or(0);
                assert_eq!(run.levels.len(), tallest, "{name}: every level passes");
                // The leaf checks run up to the failing instance, whose table the verifier
                // evaluates from the rows, as the batch records.
                let failing = error.rejected_instance().unwrap();
                for (i, evaluation) in run.table_evals.iter().enumerate() {
                    assert_eq!(*evaluation, batch.leaf_checks[i].table.table_eval, "{name}");
                }
                let reached = if matches!(error, LookupError::TableEvaluation { .. }) {
                    failing + 1
                } else {
                    failing
                };
                assert_eq!(run.table_evals.len(), reached, "{name}");
            }
            other => panic!("{name}: {other:?}"),
        }
        cheating += 1;
    }
    // A lookup outside the table, a wrong value, a multiplicity off by one, a zero root
    // denominator on each side and the phantom row, at the roots; false balanced roots at level 0
    // and at level 1; a zero leaf numerator; a changed table, and the phantom row under zero
    // padding, at the table evaluation.
    assert_eq!(cheating, 11);
}

/// The vectors hold what #62 asks for, read from the data: (1) honest batches with an instance
/// whose table has values and one whose table has none; (2) a lookup tree and a table tree of
/// different heights in one instance; (3) a level with fraction trees and forests together; (4) a
/// level of forests alone, which squeezes no `λ`; (5) a tree joining after level 0, front-padded;
/// (6) a fraction tree of height 0, which joins no level; (7) a leaf stage without LogUp instance,
/// whose roots and values are empty; (8) an empty values message beside an instance; (9) an
/// honest multiplicity on a padding row; and the cheating batches of every check: (10) the roots,
/// (11) a level, (12) the leaf numerator and (13) the table evaluation, and (14) a phantom row,
/// a lookup of `(0, 0)` that is no row, at the roots and at the table evaluation.
#[test]
fn the_logup_vectors_cover_the_shapes_the_issue_asks_for() {
    let mut covered = [false; 14];
    let mut phantoms = 0;
    for batch in batches() {
        let heights = batch.heights();
        if batch.expected == Expected::Accept {
            covered[0] |= batch.instances.iter().any(|i| i.has_values)
                && batch.instances.iter().any(|i| !i.has_values);
            covered[1] |= batch
                .instances
                .iter()
                .any(|i| i.lookup_height() != i.table_height());
            for level in &batch.levels {
                let kinds: Vec<Kind> = level
                    .entering
                    .iter()
                    .map(|(t, _)| batch.trees[*t].kind)
                    .collect();
                covered[2] |=
                    kinds.contains(&Kind::Forest) && kinds.iter().any(|k| *k != Kind::Forest);
                covered[3] |= !kinds.contains(&Kind::Lookup)
                    && !kinds.contains(&Kind::Table)
                    && level.lambda.is_none();
                covered[4] |= level.prefix != Prefix::Tree(0)
                    && level
                        .entering
                        .iter()
                        .any(|(_, claim)| point_of(claim).len() < level.rounds());
            }
            covered[5] |= batch
                .trees
                .iter()
                .zip(&heights)
                .any(|(tree, h)| *h == 0 && tree.kind != Kind::Forest);
            covered[6] |= batch.instances.is_empty() && batch.roots.is_empty();
            covered[7] |= !batch.instances.is_empty() && batch.values.is_empty();
            covered[8] |= batch.instances.iter().any(|i| {
                i.multiplicities[i.rows.len()..]
                    .iter()
                    .any(|&m| m != Fp::ZERO)
            });
        }
        if let Expected::Reject { check, .. } = &batch.expected {
            covered[9] |= check == "roots";
            covered[10] |= check.starts_with("level");
            covered[11] |= check == "leaf numerator";
            covered[12] |= check == "table evaluation";
            // A lookup of (0, 0) into a table without that row, paid on a padding row.
            let phantom = batch.instances.iter().any(|i| {
                i.lookups.contains(&Lookup::Pair(Fp::ZERO, Fp::ZERO))
                    && !i.rows.contains(&(Fp::ZERO, Fp::ZERO))
                    && i.multiplicities[i.rows.len()..]
                        .iter()
                        .any(|&m| m != Fp::ZERO)
            });
            if phantom {
                phantoms += 1;
                covered[13] |= phantoms == 2;
            }
        }
    }
    assert_eq!(covered, [true; 14]);
}

/// The batches' instances are the vectors' own: `h_a` and `h_t` are `⌈log2 N⌉` and `⌈log2 T⌉`
/// (checked as the file is read), and every tree of the schedule is listed once.
#[test]
fn every_tree_is_scheduled_once() {
    for batch in batches() {
        let mut expected: Vec<TreeRef> = (0..batch.instances.len())
            .flat_map(|index| [Kind::Lookup, Kind::Table].map(|kind| TreeRef { kind, index }))
            .chain((0..batch.forests.len()).map(|index| TreeRef {
                kind: Kind::Forest,
                index,
            }))
            .collect();
        let mut listed = batch.trees.clone();
        let key = |tree: &TreeRef| (tree.kind as u8, tree.index);
        expected.sort_by_key(key);
        listed.sort_by_key(key);
        assert_eq!(listed, expected, "{}", batch.name);
    }
}

/// A level of product forests alone runs through [`verify_level`] exactly as through
/// [`verify_product_level`](tv_core::tree::verify_product_level): on every level of the tree and
/// forest batches, honest and with each element of its proof changed by one, both give the same
/// `γ`, point, `ρ` and next claims, or the same error, no `λ`, and the same transcript after.
#[test]
fn a_level_of_forests_alone_runs_as_a_product_tree_level() {
    use support::{forest_batches, tree_batches};
    use tv_core::sumcheck::SumcheckProof;
    use tv_core::tree::{TreeClaim, verify_product_level};

    // Each level as (context, root challenges and claims of every level before it and of it).
    struct Recorded {
        transcript: Box<dyn Fn() -> Transcript>,
        levels: Vec<(usize, Vec<TreeClaim>, SumcheckProof)>,
    }
    let mut recorded: Vec<Recorded> = Vec::new();
    for batch in tree_batches() {
        let context = std::rc::Rc::new(batch.context);
        recorded.push(Recorded {
            transcript: Box::new(move || context.transcript()),
            levels: batch
                .levels
                .iter()
                .map(|level| {
                    let claims = level.entering.iter().map(|(_, c)| c.clone()).collect();
                    (0, claims, level.proof.clone())
                })
                .collect(),
        });
    }
    for batch in forest_batches() {
        let context = std::rc::Rc::new(batch.context);
        recorded.push(Recorded {
            transcript: Box::new(move || context.transcript()),
            levels: batch
                .levels
                .iter()
                .map(|level| {
                    let claims = level
                        .level
                        .entering
                        .iter()
                        .map(|(_, c)| c.clone())
                        .collect();
                    (
                        level.root_challenges.len(),
                        claims,
                        level.level.proof.clone(),
                    )
                })
                .collect(),
        });
    }
    let mut compared = 0;
    for batch in &recorded {
        for index in 0..batch.levels.len() {
            // A transcript through the levels before `index` and the root challenges of `index`.
            let replay = || {
                let mut transcript = (batch.transcript)();
                for (level, (roots, claims, proof)) in batch.levels[..=index].iter().enumerate() {
                    for c in 0..*roots {
                        let _ = challenge(&mut transcript, &format!("tree/{level}/root/{c}"));
                    }
                    if level < index {
                        let _ = verify_product_level(&mut transcript, level as u32, claims, proof)
                            .unwrap();
                    }
                }
                transcript
            };
            let (_, claims, honest) = &batch.levels[index];
            let wrapped: Vec<LevelClaim> =
                claims.iter().cloned().map(LevelClaim::Product).collect();
            let mut proofs = vec![honest.clone()];
            for j in 0..honest.rounds.len() {
                for k in 0..honest.rounds[j].len() {
                    let mut changed = honest.clone();
                    changed.rounds[j][k] += Fp2::ONE;
                    proofs.push(changed);
                }
            }
            for k in 0..honest.final_evals.len() {
                let mut changed = honest.clone();
                changed.final_evals[k] += Fp2::ONE;
                proofs.push(changed);
            }
            for proof in &proofs {
                let mut product = replay();
                let mut general = replay();
                let by_product = verify_product_level(&mut product, index as u32, claims, proof);
                let by_level = verify_level(&mut general, index as u32, &wrapped, proof);
                match (by_product, by_level) {
                    (Ok(a), Ok(b)) => {
                        assert_eq!(b.lambda, None);
                        assert_eq!((a.gamma, &a.point, a.rho), (b.gamma, &b.point, b.rho));
                        let claims: Vec<LevelClaim> =
                            a.claims.into_iter().map(LevelClaim::Product).collect();
                        assert_eq!(claims, b.claims);
                    }
                    (Err(a), Err(b)) => assert_eq!(a, b),
                    (a, b) => panic!("{a:?} and {b:?}"),
                }
                assert_eq!(product.state(), general.state());
                assert_eq!(product.counter(), general.counter());
                compared += 1;
            }
        }
    }
    // The 39 tree levels and their 337 single-element changes (`mutations.rs`), and the 11
    // forest levels and their 162 (`forests.rs`).
    assert_eq!(compared, 39 + 337 + 11 + 162);
}
