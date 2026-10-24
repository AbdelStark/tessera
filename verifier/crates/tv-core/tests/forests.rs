// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tv-core`'s product-tree levels on the forest batches of `spec/vectors/sumcheck/`, honest
//! proofs written by an independent Python prover (RFC-0010 'Tree levels', #58).
//!
//! A forest stacks `2^s` product trees of one height: its level `ℓ` has `2^{s+ℓ}` entries, the
//! tree index in the high `s` variables. It joins the batch with a claim on its root level at a
//! random point `r`, the last `s` of the challenges its level squeezes under `tree/{ℓ}/root/{c}`
//! before `γ`, and the claim `Σ_y v_y·eq(r, y)` comes from its public root values `v_y`, the
//! trees' products. `tv-core` checks each level with [`verify_product_level`] as for single
//! trees; the root challenges and root claims are the caller's, which `support::run_forests`
//! plays, written from the RFC text.
//!
//! - Every batch verifies, and every value the prover recorded is what the driver and the
//!   verifier derive: each level's root challenges, its entering claims, `γ`, the challenges and
//!   the claims of its rounds, `ρ`, its next claims and the transcript after it. Every claim is
//!   its forest level's extension at its point, and every leaf claim the stacked leaves', both
//!   computed here.
//! - Every single-element change of every level's proof is rejected with its exact error, as for
//!   trees in `mutations.rs`; exchanged `u_i` and `w_i` pass their level and fail the next, or
//!   the leaf check, as in `checks.rs`.
//! - A non-zero product claimed as zero, and a root point of the first `s` challenges, are
//!   rejected at the level where their forest joins.
//! - The batches cover the shapes #58 asks for, read from the data.

mod support;

use support::{
    Forest, ForestBatch, ForestOutcome, RootPoint, extension, forest_batches, root_claim,
    run_forests,
};
use tv_core::sumcheck::{Prefix, SumcheckError, SumcheckProof, reduce};
use tv_core::tree::verify_product_level;
use tv_field::{Fp, Fp2};
use tv_hash::transcript::{Ascii, Transcript};

/// The height of `batch`'s tallest forest, its number of levels.
fn height_of(batch: &ForestBatch) -> usize {
    batch
        .forests
        .iter()
        .map(|forest| forest.height)
        .max()
        .unwrap_or(0)
}

/// The level at which `forest` joins a batch whose tallest forest has height `tallest`.
fn joins_at(forest: &Forest, tallest: usize) -> usize {
    tallest - forest.height
}

/// `challenge_fp2(label)`, or `None` when the transcript finds no value.
fn challenge(transcript: &mut Transcript, label: &str) -> Option<Fp2> {
    Ascii::new(label)
        .ok()
        .and_then(|label| transcript.challenge_fp2(label).ok())
}

/// `eq(a, b) = Π_i (a_i·b_i + (1 − a_i)·(1 − b_i))`, from its definition, or `None` for points of
/// different lengths.
fn eq(a: &[Fp2], b: &[Fp2]) -> Option<Fp2> {
    (a.len() == b.len()).then(|| {
        a.iter().zip(b).fold(Fp2::ONE, |product, (&x, &y)| {
            product * (x * y + (Fp2::ONE - x) * (Fp2::ONE - y))
        })
    })
}

/// A forest's levels from its stacked leaves, the root level first, by the rule of a single
/// tree: `F_ℓ[k] = F_{ℓ+1}[2k]·F_{ℓ+1}[2k + 1]`.
fn levels_of(forest: &Forest) -> Vec<Vec<Fp>> {
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

#[test]
fn every_forest_batch_verifies_and_reproduces_every_recorded_value() {
    let batches = forest_batches();
    assert_eq!(batches.len(), 4);
    let mut levels = 0;
    for batch in &batches {
        let mut transcript = batch.context.transcript();
        let run = run_forests(
            &mut transcript,
            &batch.forests,
            &batch.proofs(),
            RootPoint::Last,
        );
        assert_eq!(run.outcome, ForestOutcome::Accepted, "{}", batch.name);
        assert_eq!(run.levels.len(), batch.levels.len(), "{}", batch.name);
        for (index, (ran, recorded)) in run.levels.iter().zip(&batch.levels).enumerate() {
            let level = &recorded.level;
            let what = format!("{}, {}", batch.name, level.prefix);
            assert_eq!(level.prefix, Prefix::Tree(index as u32), "{what}");
            assert_eq!(
                ran.root_challenges, recorded.root_challenges,
                "{what}: the root challenges"
            );
            assert_eq!(
                ran.entering, level.entering,
                "{what}: the claims entering it"
            );
            assert_eq!(ran.outcome.gamma, level.gamma, "{what}: γ");
            assert_eq!(
                ran.outcome.point, level.challenges,
                "{what}: the challenges"
            );
            assert_eq!(ran.outcome.rho, level.rho, "{what}: ρ");
            let next: Vec<_> = ran
                .entering
                .iter()
                .map(|(forest, _)| *forest)
                .zip(ran.outcome.claims.iter().cloned())
                .collect();
            assert_eq!(next, level.next, "{what}: the next claims");
            assert_eq!(ran.state, level.state, "{what}: the state after it");
            assert_eq!(ran.counter, level.counter, "{what}: the squeezes");
            reproduce_level_claims(batch, index);
            levels += 1;
        }
        // The leaf claims are the last level's next claims, on the stacked leaves.
        assert_eq!(
            Some(&run.leaves),
            batch.levels.last().map(|last| &last.level.next),
            "{}: the leaf claims",
            batch.name
        );
        for (forest, claim) in &run.leaves {
            let forest = &batch.forests[*forest];
            assert_eq!(claim.point.len(), forest.rows_log + forest.height);
            assert_eq!(extension(&forest.leaves, &claim.point), claim.value);
        }
    }
    assert_eq!(levels, 11);
}

/// A transcript that has run `batch`'s levels before `index`, each verified with its recorded
/// claims, and squeezed the root challenges of level `index`: the transcript its `γ` comes from.
fn transcript_at_gamma(batch: &ForestBatch, index: usize) -> Transcript {
    let mut transcript = batch.context.transcript();
    for (level, recorded) in batch.levels.iter().take(index + 1).enumerate() {
        let squeezed: Vec<Option<Fp2>> = (0..recorded.root_challenges.len())
            .map(|c| challenge(&mut transcript, &format!("tree/{level}/root/{c}")))
            .collect();
        let recorded_challenges: Vec<Option<Fp2>> =
            recorded.root_challenges.iter().copied().map(Some).collect();
        assert_eq!(squeezed, recorded_challenges);
        if level < index {
            let claims: Vec<_> = recorded
                .level
                .entering
                .iter()
                .map(|(_, claim)| claim.clone())
                .collect();
            let passed = verify_product_level(
                &mut transcript,
                level as u32,
                &claims,
                &recorded.level.proof,
            );
            assert!(passed.is_ok(), "{}, level {level}: {passed:?}", batch.name);
        }
    }
    transcript
}

/// The claims of level `index` of `batch`: the batched claim, computed here from its definition,
/// then the claim after each prefix of the rounds, through [`reduce`]; and the last claim from the
/// final evaluations, `Σ_i γ^i · eq(r_i, r'_i) · u_i · w_i`, with `eq` computed here.
fn reproduce_level_claims(batch: &ForestBatch, index: usize) {
    let level = &batch.levels[index].level;
    let what = format!("{}, {}", batch.name, level.prefix);
    let n_max = level.rounds();
    let mut weight = Fp2::ONE;
    let mut batched = Fp2::ZERO;
    let mut last = Fp2::ZERO;
    let mut evals = level.proof.final_evals.chunks_exact(2);
    for (_, claim) in &level.entering {
        let n_i = claim.point.len();
        assert!(n_i <= n_max, "{what}: an instance of {n_i} variables");
        let scale = Fp::reduce(2).pow((n_max - n_i) as u64);
        batched += weight * claim.value * scale;
        let own = &level.challenges[n_max - n_i..];
        let pair = evals.next();
        assert!(pair.is_some(), "{what}: two evaluations an instance");
        if let (Some([u, w]), Some(eq_i)) = (pair, eq(&claim.point, own)) {
            last += weight * eq_i * *u * *w;
        }
        weight *= level.gamma;
    }
    assert_eq!(
        level.claims.len(),
        n_max + 1,
        "{what}: a claim before each round and after the last"
    );
    assert_eq!(level.claims[0], batched, "{what}: claim_1");
    assert_eq!(level.claims[n_max], last, "{what}: the last claim");
    for rounds in 0..=n_max {
        let mut transcript = transcript_at_gamma(batch, index);
        let gamma = challenge(&mut transcript, &format!("{}/gamma", level.prefix));
        assert_eq!(gamma, Some(level.gamma), "{what}");
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

/// Every claim of the vectors is true: each claim entering a level, and each next claim, is the
/// extension of its forest's level at its point, the levels computed here from the stacked
/// leaves; and each forest's public roots are its root level.
#[test]
fn every_claim_is_its_forest_levels_extension_at_its_point() {
    let mut claims = 0;
    for batch in forest_batches() {
        let tallest = height_of(&batch);
        let tables: Vec<Vec<Vec<Fp>>> = batch.forests.iter().map(levels_of).collect();
        for (forest, levels) in batch.forests.iter().zip(&tables) {
            assert_eq!(levels[0], forest.roots, "{}", batch.name);
        }
        for (index, recorded) in batch.levels.iter().enumerate() {
            for (below, list) in [(0, &recorded.level.entering), (1, &recorded.level.next)] {
                for (f, claim) in list {
                    let forest = &batch.forests[*f];
                    let own = index - joins_at(forest, tallest) + below;
                    assert_eq!(claim.point.len(), forest.rows_log + own);
                    assert_eq!(
                        extension(&tables[*f][own], &claim.point),
                        claim.value,
                        "{}, level {index}, forest {f}",
                        batch.name
                    );
                    claims += 1;
                }
            }
        }
    }
    // Instances entering and leaving the levels: 1 + 1; 3 + 3 for the SoftmaxRow forest; 16 + 16
    // for the six forests; 7 + 7 for the last batch.
    assert_eq!(claims, 2 + 6 + 32 + 14);
}

/// Each public root is its tree's product, and exactly the trees with a zero leaf have root 0, as
/// every tree of a SoftmaxRow forest does.
#[test]
fn every_root_is_its_trees_product_and_zero_exactly_with_a_zero_leaf() {
    let mut zero_roots = 0;
    for batch in forest_batches() {
        for forest in &batch.forests {
            let trees = forest.leaves.chunks_exact(1 << forest.height);
            assert_eq!(trees.len(), forest.roots.len());
            for (tree, &root) in trees.zip(&forest.roots) {
                let product = tree.iter().fold(Fp::ONE, |p, leaf| p * *leaf);
                assert_eq!(root, product, "{}", batch.name);
                assert_eq!(root == Fp::ZERO, tree.contains(&Fp::ZERO));
                zero_roots += usize::from(root == Fp::ZERO);
            }
        }
    }
    // The 8 and 16 trees of the SoftmaxRow forests, and one tree of a forest of 2.
    assert_eq!(zero_roots, 8 + 16 + 1);
}

/// #58's acceptance criterion, as `mutations.rs` has it for trees: every field element of every
/// level's proof, each round value and each final evaluation, changed by one in turn, is rejected
/// with the exact error. A changed round fails its level's final check. A changed `u_i` adds
/// `γ^i·eq(r_i, r'_i)·w_i` to that check, so it fails the level unless `w_i` is zero, and the
/// same for `w_i`: the 16 SoftmaxRow trees whose first leaves are zero have `u` zero where their
/// forest joins. Such a change passes its level and changes the transcript, and the next level
/// rejects it, or after the last level the stage that consumes the leaf claims.
#[test]
fn every_single_element_change_of_every_forest_level_is_rejected() {
    let mut changed = 0;
    let mut passed_on = 0;
    for batch in forest_batches() {
        let last = batch.levels.len() - 1;
        let rejected_at = |level: usize| ForestOutcome::Rejected {
            level: level as u32,
            error: SumcheckError::FinalMismatch {
                round: batch.levels[level].level.rounds() as u32,
            },
        };
        for (level, recorded) in batch.levels.iter().enumerate() {
            let recorded = &recorded.level;
            let mut check = |proofs: Vec<SumcheckProof>, expected: &ForestOutcome, what: String| {
                let mut transcript = batch.context.transcript();
                let run = run_forests(&mut transcript, &batch.forests, &proofs, RootPoint::Last);
                assert_eq!(
                    run.outcome, *expected,
                    "{}, level {level}: {what}",
                    batch.name
                );
                if let ForestOutcome::Rejected { error, .. } = run.outcome {
                    // The driver's Reject(TreeLevel { group, level, round }) reads the round here.
                    assert!(error.rejected_round().is_some(), "{error}");
                }
                changed += 1;
            };
            for j in 0..recorded.rounds() {
                for k in 0..3 {
                    let mut proofs = batch.proofs();
                    proofs[level].rounds[j][k] += Fp2::ONE;
                    check(proofs, &rejected_at(level), format!("round {j}, value {k}"));
                }
            }
            for k in 0..recorded.proof.final_evals.len() {
                let mut proofs = batch.proofs();
                proofs[level].final_evals[k] += Fp2::ONE;
                // The other value of the pair: u for w, w for u.
                let partner = recorded.proof.final_evals[k ^ 1];
                let forest = recorded.entering[k / 2].0;
                let expected = if partner != Fp2::ZERO {
                    rejected_at(level)
                } else if level == last {
                    ForestOutcome::FalseLeaf { forest }
                } else {
                    rejected_at(level + 1)
                };
                passed_on += usize::from(partner == Fp2::ZERO);
                check(proofs, &expected, format!("evaluation {k}"));
            }
        }
    }
    // Σ (3·n_max + 2·m) over the levels: 5 for the forest of 2 trees; 11 + 14 + 17 for the
    // SoftmaxRow forest of 8 trees; 12 + 15 + 20 + 27 for the six forests; 2 + 18 + 21 for the
    // last batch.
    assert_eq!(changed, 5 + 42 + 74 + 41);
    // The w of the SoftmaxRow forest of 16 trees, whose u is zero where it joins.
    assert_eq!(passed_on, 1);
}

/// The level check `Σ_i γ^i · eq · u_i · w_i` cannot tell `u_i` from `w_i`, but the next claim
/// can, as `checks.rs` has it for trees: exchanged, they pass their level, and the next claim is
/// `Ṽ(1 − ρ ‖ r'_i)` where `Ṽ(ρ ‖ r'_i)` is claimed, so the next level rejects it, or after the
/// last level the leaf check does.
#[test]
fn exchanged_u_and_w_pass_their_level_and_fail_the_next() {
    let mut cases = 0;
    for batch in forest_batches() {
        let last = batch.levels.len() - 1;
        for (level, recorded) in batch.levels.iter().enumerate() {
            for (i, (forest, _)) in recorded.level.entering.iter().enumerate() {
                let mut proofs = batch.proofs();
                proofs[level].final_evals.swap(2 * i, 2 * i + 1);
                let mut transcript = batch.context.transcript();
                let run = run_forests(&mut transcript, &batch.forests, &proofs, RootPoint::Last);
                let expected = if level == last {
                    ForestOutcome::FalseLeaf { forest: *forest }
                } else {
                    ForestOutcome::Rejected {
                        level: level as u32 + 1,
                        error: SumcheckError::FinalMismatch {
                            round: batch.levels[level + 1].level.rounds() as u32,
                        },
                    }
                };
                assert_eq!(
                    run.outcome, expected,
                    "{}, level {level}, instance {i}",
                    batch.name
                );
                assert_eq!(run.levels.len(), (level + 1).min(batch.levels.len()));
                cases += 1;
            }
        }
    }
    // One case per instance of every level: 1, 3, 16 and 7.
    assert_eq!(cases, 1 + 3 + 16 + 7);
}

/// RFC-0010 'Testing Strategy': a forest with a tree whose product differs from its public root
/// value is rejected. Claiming a non-zero product as zero, for each such tree alone and for all of
/// a forest's at once, changes the forest's root claim, and the level where it joins rejects the
/// honest proof at its final check, the levels before it untouched.
#[test]
fn a_nonzero_product_claimed_as_zero_is_rejected_where_its_forest_joins() {
    let mut cases = 0;
    for batch in forest_batches() {
        let tallest = height_of(&batch);
        for (f, forest) in batch.forests.iter().enumerate() {
            let join = joins_at(forest, tallest);
            let expected = ForestOutcome::Rejected {
                level: join as u32,
                error: SumcheckError::FinalMismatch {
                    round: batch.levels[join].level.rounds() as u32,
                },
            };
            let nonzero: Vec<usize> = (0..forest.roots.len())
                .filter(|&y| forest.roots[y] != Fp::ZERO)
                .collect();
            let mut claimed: Vec<Vec<usize>> = nonzero.iter().map(|&y| vec![y]).collect();
            if nonzero.len() > 1 {
                claimed.push(nonzero.clone());
            }
            for zeros in claimed {
                let mut forests = batch.forests.clone();
                for &y in &zeros {
                    forests[f].roots[y] = Fp::ZERO;
                }
                let mut transcript = batch.context.transcript();
                let run = run_forests(&mut transcript, &forests, &batch.proofs(), RootPoint::Last);
                assert_eq!(
                    run.outcome, expected,
                    "{}, forest {f}, roots {zeros:?} claimed zero",
                    batch.name
                );
                assert_eq!(run.levels.len(), join);
                cases += 1;
            }
        }
    }
    // The forest of 2 trees: 2 + 1. The six forests: 4 + 1, 2 + 1, the single tree, the one
    // non-zero root of the forest of 2 trees with a zero leaf, and the two single trees. The last
    // batch: its single tree, and 4 + 1 for the forest of 4 trees.
    assert_eq!(cases, 3 + 12 + 6);
}

/// RFC-0010 'Tree levels': a forest's root point is the last `s` of its level's root challenges,
/// in the order squeezed. A driver that takes the first `s` agrees wherever every joining forest
/// has `s = s_max` or `s = 0`, and is rejected at the first level where a forest joins with a
/// larger one, at that level's final check.
#[test]
fn a_root_point_of_the_first_s_challenges_is_rejected() {
    let mut rejected = 0;
    for batch in forest_batches() {
        let tallest = height_of(&batch);
        let differs = (0..tallest).find(|&level| {
            let joining: Vec<usize> = batch
                .forests
                .iter()
                .filter(|forest| joins_at(forest, tallest) == level)
                .map(|forest| forest.rows_log)
                .collect();
            let s_max = joining.iter().copied().max().unwrap_or(0);
            joining.iter().any(|&s| 0 < s && s < s_max)
        });
        let expected = differs.map_or(ForestOutcome::Accepted, |level| ForestOutcome::Rejected {
            level: level as u32,
            error: SumcheckError::FinalMismatch {
                round: batch.levels[level].level.rounds() as u32,
            },
        });
        let mut transcript = batch.context.transcript();
        let run = run_forests(
            &mut transcript,
            &batch.forests,
            &batch.proofs(),
            RootPoint::First,
        );
        assert_eq!(run.outcome, expected, "{}", batch.name);
        rejected += usize::from(differs.is_some());
    }
    // Level 0 of the six forests, where s = 2, 1 and 0 join; level 1 of the last batch, s = 4
    // and 2.
    assert_eq!(rejected, 2);
}

/// The vectors hold what #58 asks for, read from the data: (1) a SoftmaxRow forest of at least 8
/// trees, every one with a zero leaf; (2) a forest of at least 2 trees without a zero leaf, whose
/// root claim is not zero; (3) forests of different `s ≥ 1` joining at one level; (4) a single
/// tree joining with a forest of `s ≥ 1`; (5) a forest joining after level 0, front-padded;
/// (6) a level where only single trees join while forests are active, with no root challenge;
/// (7) a level whose `s_max` is at least 3. At every level, as many root challenges as the
/// largest `s` among the joining forests.
#[test]
fn the_forest_vectors_cover_the_shapes_the_issue_asks_for() {
    let mut covered = [false; 7];
    for batch in forest_batches() {
        let tallest = height_of(&batch);
        assert_eq!(batch.levels.len(), tallest);
        for (index, recorded) in batch.levels.iter().enumerate() {
            let level = &recorded.level;
            let joining: Vec<(usize, &Forest)> = batch
                .forests
                .iter()
                .enumerate()
                .filter(|(_, forest)| joins_at(forest, tallest) == index)
                .collect();
            let s_max = joining
                .iter()
                .map(|(_, forest)| forest.rows_log)
                .max()
                .unwrap_or(0);
            assert_eq!(recorded.root_challenges.len(), s_max, "{}", batch.name);
            let root_claim_of = |f: usize| {
                level
                    .entering
                    .iter()
                    .find(|(owner, _)| *owner == f)
                    .map(|(_, claim)| claim.value)
            };
            for &(f, forest) in &joining {
                let zero_leaf_in_every_tree = forest
                    .leaves
                    .chunks_exact(1 << forest.height)
                    .all(|tree| tree.contains(&Fp::ZERO));
                if forest.rows_log >= 3 && zero_leaf_in_every_tree {
                    covered[0] |= root_claim_of(f) == Some(Fp2::ZERO);
                }
                if forest.rows_log >= 1 && !forest.leaves.contains(&Fp::ZERO) {
                    let claim = root_claim_of(f);
                    covered[1] |= claim.is_some() && claim != Some(Fp2::ZERO);
                }
                covered[4] |= index > 0 && forest.rows_log >= 1 && forest.rows_log < level.rounds();
            }
            let s: Vec<usize> = joining.iter().map(|(_, forest)| forest.rows_log).collect();
            covered[2] |= s
                .iter()
                .any(|&a| s.iter().any(|&b| a >= 1 && b >= 1 && a != b));
            covered[3] |= s.contains(&0) && s_max >= 1;
            let forest_active = batch
                .forests
                .iter()
                .any(|forest| forest.rows_log >= 1 && joins_at(forest, tallest) < index);
            covered[5] |= !s.is_empty() && s_max == 0 && forest_active;
            covered[6] |= s_max >= 3;
        }
    }
    assert_eq!(covered, [true; 7]);
}

/// The root claims the driver forms are the vectors' and the root levels' extensions: the
/// formula `Σ_y v_y·eq(r, y)` of [`root_claim`] agrees with the multilinear extension of the
/// public roots at every root point the vectors use, and is not the claim at the first `s`
/// challenges where those differ from the last.
#[test]
fn every_root_claim_is_the_root_levels_extension_at_the_last_s_challenges() {
    let mut joined = 0;
    for batch in forest_batches() {
        let tallest = height_of(&batch);
        for (f, forest) in batch.forests.iter().enumerate() {
            let recorded = &batch.levels[joins_at(forest, tallest)];
            let entered = recorded
                .level
                .entering
                .iter()
                .find(|(owner, _)| *owner == f);
            let s = forest.rows_log;
            let last = &recorded.root_challenges[recorded.root_challenges.len() - s..];
            let first = &recorded.root_challenges[..s];
            let value = root_claim(&forest.roots, last);
            assert_eq!(value, Some(extension(&forest.roots, last)));
            assert_eq!(
                entered.map(|(_, claim)| (&claim.point[..], claim.value)),
                value.map(|value| (last, value)),
                "{}, forest {f}",
                batch.name
            );
            if first != last && forest.roots.iter().any(|&root| root != forest.roots[0]) {
                assert_ne!(root_claim(&forest.roots, first), value);
            }
            joined += 1;
        }
    }
    assert_eq!(joined, 1 + 1 + 6 + 3);
}
