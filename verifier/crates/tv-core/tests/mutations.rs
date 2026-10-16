// SPDX-License-Identifier: Apache-2.0 OR MIT

//! #59's acceptance criterion: every single-element mutation of every honest proof of
//! `spec/vectors/sumcheck/` is rejected. Each field element of each message, every round
//! evaluation and every final evaluation, is changed by one in turn, and the exact error is
//! asserted.
//!
//! - A stage rejects every change at its final check, `FinalMismatch { round: n_max }`: a round
//!   value changes the transcript and the claim from that round on; a final evaluation changes
//!   its instance's product by the product of the others, which is not zero.
//! - A tree level rejects every change to its rounds at its final check. A change to `u_i` adds
//!   `γ^i·eq(r_i, r'_i)·w_i` to the level's combination, so it fails the level unless `w_i` is
//!   zero, as it is for a tree with a zero leaf at the level it joins, where its two values are
//!   its own level-1 entries. Then the level passes, the transcript and the next claim change,
//!   and the next level rejects the change, or after the last level the stage that consumes the
//!   leaf claims does.

mod support;

use support::{TreeOutcome, run_trees, sessions, tree_batches, verify};
use tv_core::sumcheck::SumcheckError;
use tv_field::Fp2;

#[test]
fn every_single_element_change_of_every_stage_is_rejected() {
    let mut changed = 0;
    let mut expected_total = 0;
    for session in sessions() {
        for (index, stage) in session.stages.iter().enumerate() {
            let expected = Err(SumcheckError::FinalMismatch {
                round: stage.final_round(),
            });
            expected_total += stage.rounds() * stage.degree + stage.proof.final_evals.len();
            let mut check = |proof, what: String| {
                let mut transcript = session.transcript_before(index);
                let outcome = verify(&mut transcript, stage, &stage.instances, &proof).map(|_| ());
                assert_eq!(
                    outcome, expected,
                    "{}, {}: {what}",
                    session.name, stage.prefix
                );
                // The driver's Reject(Sumcheck { stage, round }) reads the round here.
                let round = outcome.unwrap_err().rejected_round();
                assert_eq!(round, Some(stage.final_round()));
                changed += 1;
            };
            for j in 0..stage.rounds() {
                for k in 0..stage.degree {
                    let mut proof = stage.proof.clone();
                    proof.rounds[j][k] += Fp2::ONE;
                    check(proof, format!("round {j}, value {k}"));
                }
            }
            for k in 0..stage.proof.final_evals.len() {
                let mut proof = stage.proof.clone();
                proof.final_evals[k] += Fp2::ONE;
                check(proof, format!("evaluation {k}"));
            }
        }
    }
    assert_eq!(changed, expected_total);
    // The single instances: 91·d values for products of d tables (13 stages of n·d round values
    // and d evaluations); then the batches: 36 + 8, 48 + 10, 10 + 5, and 6 + 14 + 1; then
    // degrees 5 to 7: 15 + 5, 18 + 6, 21 + 7 and 28 + 11.
    assert_eq!(
        changed,
        91 * (1 + 2 + 3 + 4) + 44 + 58 + 15 + 21 + 20 + 24 + 28 + 39
    );
}

#[test]
fn every_single_element_change_of_every_tree_level_is_rejected() {
    let mut changed = 0;
    let mut passed_on = 0;
    for batch in tree_batches() {
        let last = batch.levels.len() - 1;
        for (level, recorded) in batch.levels.iter().enumerate() {
            let at_this_level = TreeOutcome::Rejected {
                level: level as u32,
                error: SumcheckError::FinalMismatch {
                    round: recorded.rounds() as u32,
                },
            };
            let mut check = |proofs: Vec<_>, expected: &TreeOutcome, what: String| {
                let mut transcript = batch.context.transcript();
                let run = run_trees(&mut transcript, &batch.trees, &proofs, |_, _| {});
                assert_eq!(
                    run.outcome, *expected,
                    "{}, level {level}: {what}",
                    batch.name
                );
                if let TreeOutcome::Rejected { error, .. } = run.outcome {
                    let SumcheckError::FinalMismatch { round } = error else {
                        panic!("{error}");
                    };
                    assert_eq!(error.rejected_round(), Some(round));
                }
                changed += 1;
            };
            for j in 0..recorded.rounds() {
                for k in 0..3 {
                    let mut proofs = batch.proofs();
                    proofs[level].rounds[j][k] += Fp2::ONE;
                    check(proofs, &at_this_level, format!("round {j}, value {k}"));
                }
            }
            for k in 0..recorded.proof.final_evals.len() {
                let mut proofs = batch.proofs();
                proofs[level].final_evals[k] += Fp2::ONE;
                // The other value of the pair: u for w, w for u.
                let partner = recorded.proof.final_evals[k ^ 1];
                let tree = recorded.entering[k / 2].0;
                let expected = if partner != Fp2::ZERO {
                    at_this_level.clone()
                } else if level == last {
                    TreeOutcome::FalseLeaf { tree }
                } else {
                    TreeOutcome::Rejected {
                        level: level as u32 + 1,
                        error: SumcheckError::FinalMismatch {
                            round: level as u32 + 1,
                        },
                    }
                };
                passed_on += usize::from(partner == Fp2::ZERO);
                check(proofs, &expected, format!("evaluation {k}"));
            }
        }
    }
    // Single trees of heights 1 to 6: Σ_ℓ (3ℓ + 2) each, 147 together; the six trees, 45 round
    // values and 44 evaluations; the zero tree alone, 15; three of height 4, 18 + 24; heights 2
    // and 5, 30 + 14.
    assert_eq!(changed, 147 + 89 + 15 + 42 + 44);
    // The three trees with a zero leaf, each at the level it joins.
    assert_eq!(passed_on, 3);
}
