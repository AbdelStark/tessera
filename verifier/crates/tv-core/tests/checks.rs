// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Every check of `tv-core`'s sumcheck specification, violated exactly once, with the exact
//! error asserted. Each case starts from an honest proof of `spec/vectors/sumcheck/` and changes
//! one thing: a round's length, the number of rounds or of final evaluations, the batched claim's
//! scale or order, a claim's point or value, a tree instance's two evaluations. The single
//! element changes of every message are in `mutations.rs`.
//!
//! A tree level's check is symmetric in `u_i` and `w_i`, and blind to a claim the next level
//! receives: those changes pass their level and are caught by the next, or by the stage that
//! consumes the leaf claims, which `support::run_trees` stands in for.

mod support;

use support::{Session, Stage, TreeOutcome, product, run_trees, sessions, tree_batches, verify};
use tv_core::sumcheck::{
    Instance, MAX_DEGREE, Prefix, SumcheckError, SumcheckProof, reduce, verify_batch,
};
use tv_core::tree::{TreeClaim, verify_product_level};
use tv_field::{Fp, Fp2};

/// Runs `check` on every stage of every session, with the transcript the stage starts from.
fn each_stage(mut check: impl FnMut(&Session, usize, &Stage)) {
    for session in sessions() {
        for (index, stage) in session.stages.iter().enumerate() {
            check(&session, index, stage);
        }
    }
}

/// `proof` verified as `stage` of `session`, with `instances`.
fn outcome(
    session: &Session,
    index: usize,
    instances: &[Instance],
    proof: &SumcheckProof,
) -> Result<(), SumcheckError> {
    let mut transcript = session.transcript_before(index);
    verify(&mut transcript, &session.stages[index], instances, proof).map(|_| ())
}

#[test]
fn each_round_too_short_or_too_long_is_rejected_at_that_round() {
    let mut cases = 0;
    each_stage(|session, index, stage| {
        let d = stage.degree;
        for j in 0..stage.rounds() {
            let round = j as u32;
            let mut short = stage.proof.clone();
            short.rounds[j].pop();
            let mut long = stage.proof.clone();
            long.rounds[j].push(Fp2::ZERO);
            let mut empty = stage.proof.clone();
            empty.rounds[j].clear();
            for (proof, found) in [(short, d - 1), (long, d + 1), (empty, 0)] {
                assert_eq!(
                    outcome(session, index, &stage.instances, &proof),
                    Err(SumcheckError::RoundLength {
                        round,
                        expected: d,
                        found
                    }),
                    "{}, {}",
                    session.name,
                    stage.prefix
                );
                cases += 1;
            }
        }
    });
    // Three cases a round: 78 rounds in each session of single instances of 1 to 4 tables,
    // then the rounds of the batches and of degrees 5 to 7.
    assert_eq!(
        cases,
        3 * 4 * 78 + 3 * (12 + 12 + 5 + 6) + 3 * (3 + 3 + 3 + 4)
    );
}

#[test]
fn a_missing_round_and_an_extra_round_are_rejected() {
    let mut missing = 0;
    each_stage(|session, index, stage| {
        let n = stage.rounds();
        let what = format!("{}, {}", session.name, stage.prefix);
        if n > 0 {
            // Whichever round is dropped, the proof runs out at the last one.
            for dropped in [0, n - 1] {
                let mut proof = stage.proof.clone();
                proof.rounds.remove(dropped);
                assert_eq!(
                    outcome(session, index, &stage.instances, &proof),
                    Err(SumcheckError::MissingRound {
                        round: n as u32 - 1
                    }),
                    "{what}"
                );
            }
            let mut none = stage.proof.clone();
            none.rounds.clear();
            assert_eq!(
                outcome(session, index, &stage.instances, &none),
                Err(SumcheckError::MissingRound { round: 0 }),
                "{what}"
            );
            missing += 1;
        }
        // A round past the last is rejected whatever it holds: it has no length to check.
        for extra in [
            vec![Fp2::ONE; stage.degree],
            vec![],
            vec![Fp2::ZERO; MAX_DEGREE + 3],
        ] {
            let mut proof = stage.proof.clone();
            proof.rounds.push(extra);
            assert_eq!(
                outcome(session, index, &stage.instances, &proof),
                Err(SumcheckError::ExtraRound { round: n as u32 }),
                "{what}"
            );
        }
    });
    // The single instances on 1 to 12 variables, the four batches with rounds, and the four
    // stages of degrees 5 to 7.
    assert_eq!(missing, 4 * 12 + 4 + 4);
}

#[test]
fn a_missing_or_an_extra_final_evaluation_is_rejected() {
    each_stage(|session, index, stage| {
        let expected = stage.proof.final_evals.len();
        let round = stage.final_round();
        let mut fewer = stage.proof.clone();
        fewer.final_evals.pop();
        let mut more = stage.proof.clone();
        more.final_evals.push(Fp2::ONE);
        let mut none = stage.proof.clone();
        none.final_evals.clear();
        for (proof, found) in [(fewer, expected - 1), (more, expected + 1), (none, 0)] {
            assert_eq!(
                outcome(session, index, &stage.instances, &proof),
                Err(SumcheckError::EvaluationCount {
                    round,
                    expected,
                    found
                }),
                "{}, {}",
                session.name,
                stage.prefix
            );
        }
    });
}

/// RFC-0010 'Testing Strategy': dropping the `2^{n_max − n_i}` scaling is rejected. The honest
/// proof of a front-padded batch does not verify against the unscaled claim `Σ_i γ^i · c_i`,
/// presented as instances whose claims are `c_i / 2^{n_max − n_i}`; and the rounds reduce the
/// unscaled claim to a value that is not the relations' at the point.
#[test]
fn a_batched_claim_without_the_padding_scale_is_rejected() {
    let mut padded = 0;
    each_stage(|session, index, stage| {
        let n_max = stage.rounds();
        if stage
            .instances
            .iter()
            .all(|instance| instance.vars == n_max)
        {
            return;
        }
        let half = Fp::reduce(2).inverse().unwrap();
        let unscaled: Vec<Instance> = stage
            .instances
            .iter()
            .map(|instance| Instance {
                claim: instance.claim * half.pow((n_max - instance.vars) as u64),
                ..*instance
            })
            .collect();
        assert_eq!(
            outcome(session, index, &unscaled, &stage.proof),
            Err(SumcheckError::FinalMismatch {
                round: n_max as u32
            }),
            "{}",
            session.name
        );
        // The rounds from Σ_i γ^i · c_i: the last claim is not Σ_i γ^i · g_i.
        let mut weight = Fp2::ONE;
        let mut claim = Fp2::ZERO;
        for instance in &stage.instances {
            claim += weight * instance.claim;
            weight *= stage.gamma;
        }
        let mut transcript = session.transcript_before(index);
        let label = format!("{}/gamma", stage.prefix);
        transcript
            .challenge_fp2(tv_hash::transcript::Ascii::new(&label).unwrap())
            .unwrap();
        let reduction = reduce(
            &mut transcript,
            stage.prefix,
            stage.degree,
            n_max,
            claim,
            &stage.proof.rounds,
        )
        .unwrap();
        assert_ne!(reduction.claim, stage.claims[n_max]);
        padded += 1;
    });
    assert_eq!(padded, 3);
}

/// The batching powers follow schedule order: the honest proof does not verify with two
/// instances exchanged, nor with two instances' evaluations exchanged.
#[test]
fn instances_out_of_schedule_order_are_rejected() {
    let mut cases = 0;
    each_stage(|session, index, stage| {
        let count = stage.instances.len();
        for a in 0..count {
            for b in a + 1..count {
                let mut swapped = stage.instances.clone();
                swapped.swap(a, b);
                assert_eq!(
                    outcome(session, index, &swapped, &stage.proof),
                    Err(SumcheckError::FinalMismatch {
                        round: stage.final_round()
                    }),
                    "{}: instances {a} and {b}",
                    session.name
                );
                cases += 1;
                if stage.instances[a].evals == stage.instances[b].evals {
                    let start = |i: usize| -> usize {
                        stage.instances[..i]
                            .iter()
                            .map(|instance| instance.evals)
                            .sum()
                    };
                    let k = stage.instances[a].evals;
                    let mut proof = stage.proof.clone();
                    for e in 0..k {
                        proof.final_evals.swap(start(a) + e, start(b) + e);
                    }
                    assert_eq!(
                        outcome(session, index, &stage.instances, &proof),
                        Err(SumcheckError::FinalMismatch {
                            round: stage.final_round()
                        }),
                        "{}: evaluations of {a} and {b}",
                        session.name
                    );
                    cases += 1;
                }
            }
        }
    });
    // Pairs of instances, plus the pairs with as many evaluations: 3 + 1 for the batch of 3,
    // 7 and 12 variables, 6 for the batch of four, 3 + 1 for the equal sizes, 3 constants, and
    // 3 for the batch of degree 7.
    assert_eq!(cases, 3 + 1 + 6 + 3 + 1 + 3 + 3);
}

/// RFC-0010 'Tree levels', step 4: a level's next claim is `u_i + ρ·(w_i − u_i)` at `ρ ‖ r'_i`.
/// A claim with another value, or with `ρ` elsewhere than the lowest variable, fails the level
/// it enters.
#[test]
fn a_wrong_line_value_or_point_is_rejected_at_the_next_level() {
    let mut values = 0;
    let mut points = 0;
    for batch in tree_batches() {
        for level in 1..batch.levels.len() as u32 {
            let entering = &batch.levels[level as usize].entering;
            for (position, (tree, claim)) in entering.iter().enumerate() {
                if claim.point.is_empty() {
                    continue; // a root joining here, not a line's value
                }
                let what = format!("{}, level {level}, tree {tree}", batch.name);
                let expected = TreeOutcome::Rejected {
                    level,
                    error: SumcheckError::FinalMismatch { round: level },
                };
                let mut transcript = batch.context.transcript();
                let run = run_trees(
                    &mut transcript,
                    &batch.trees,
                    &batch.proofs(),
                    |at, claims| {
                        if at == level {
                            claims[position].value += Fp2::ONE;
                        }
                    },
                );
                assert_eq!(run.outcome, expected, "{what}: value");
                values += 1;
                if claim.point.len() >= 2 {
                    let mut transcript = batch.context.transcript();
                    let run = run_trees(
                        &mut transcript,
                        &batch.trees,
                        &batch.proofs(),
                        |at, claims| {
                            if at == level {
                                claims[position].point.rotate_left(1);
                            }
                        },
                    );
                    assert_eq!(run.outcome, expected, "{what}: ρ moved to the top");
                    points += 1;
                }
            }
        }
    }
    assert!(
        values >= 40 && points >= 30,
        "{values} values, {points} points"
    );
}

/// `eq(r_i, r'_i)` is computed from the claim's point, coordinate by coordinate: a claim moved
/// to another point by one coordinate, or with its coordinates reversed, fails its level, and so
/// does a root claim of another value.
#[test]
fn a_claim_at_another_point_is_rejected_by_its_eq_factor() {
    let mut cases = 0;
    for batch in tree_batches() {
        for (level, recorded) in batch.levels.iter().enumerate() {
            let level = level as u32;
            for (position, (_, claim)) in recorded.entering.iter().enumerate() {
                let mut replacements: Vec<TreeClaim> = Vec::new();
                for k in 0..claim.point.len() {
                    let mut moved = claim.clone();
                    moved.point[k] += Fp2::ONE;
                    replacements.push(moved);
                }
                if claim.point.len() >= 2 {
                    let mut reversed = claim.clone();
                    reversed.point.reverse();
                    replacements.push(reversed);
                }
                if claim.point.is_empty() {
                    let mut root = claim.clone();
                    root.value += Fp2::ONE;
                    replacements.push(root);
                }
                for replacement in &replacements {
                    let mut transcript = batch.context.transcript();
                    let run = run_trees(
                        &mut transcript,
                        &batch.trees,
                        &batch.proofs(),
                        |at, claims| {
                            if at == level {
                                claims[position] = replacement.clone();
                            }
                        },
                    );
                    assert_eq!(
                        run.outcome,
                        TreeOutcome::Rejected {
                            level,
                            error: SumcheckError::FinalMismatch {
                                round: recorded.rounds() as u32
                            }
                        },
                        "{}, level {level}, instance {position}",
                        batch.name
                    );
                    cases += 1;
                }
            }
        }
    }
    assert!(cases >= 150, "{cases}");
}

/// The level check `Σ_i γ^i · eq · u_i · w_i` cannot tell `u_i` from `w_i`, but the next
/// claim can: with them exchanged, the level passes, its next claim is
/// `Ṽ(1 − ρ ‖ r'_i)` where `Ṽ(ρ ‖ r'_i)` is claimed, and the next level, or the stage that
/// consumes the leaf claims, rejects it.
#[test]
fn exchanged_u_and_w_pass_their_level_and_fail_the_next() {
    let mut cases = 0;
    for batch in tree_batches() {
        let last = batch.levels.len() - 1;
        for (level, recorded) in batch.levels.iter().enumerate() {
            for (i, (tree, _)) in recorded.entering.iter().enumerate() {
                let mut proofs = batch.proofs();
                proofs[level].final_evals.swap(2 * i, 2 * i + 1);
                let mut transcript = batch.context.transcript();
                let run = run_trees(&mut transcript, &batch.trees, &proofs, |_, _| {});
                let expected = if level == last {
                    TreeOutcome::FalseLeaf { tree: *tree }
                } else {
                    TreeOutcome::Rejected {
                        level: level as u32 + 1,
                        error: SumcheckError::FinalMismatch {
                            round: level as u32 + 1,
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
    assert!(cases >= 60, "{cases}");
}

/// Faults of the caller, the schedule's degree and instances and its relation, are reported as
/// such, the first three before the transcript moves: a proof is never judged on them.
#[test]
fn faults_of_the_caller_are_not_verdicts_on_the_proof() {
    let session = sessions().remove(4);
    let stage = &session.stages[0];
    let untouched = *session.context.transcript().state();
    let attempt = |degree: usize, instances: &[Instance]| {
        let mut transcript = session.context.transcript();
        let outcome = verify_batch(
            &mut transcript,
            stage.prefix,
            degree,
            instances,
            &stage.proof,
            product,
        )
        .map(|_| ());
        assert_eq!(*transcript.state(), untouched, "the transcript moved");
        outcome
    };
    for degree in [0, MAX_DEGREE + 1, usize::MAX] {
        let expected = Err(SumcheckError::Degree { degree });
        assert_eq!(attempt(degree, &stage.instances), expected);
        let mut transcript = session.context.transcript();
        assert_eq!(
            reduce(&mut transcript, stage.prefix, degree, 0, Fp2::ZERO, &[]).map(|_| ()),
            expected
        );
    }
    assert_eq!(attempt(stage.degree, &[]), Err(SumcheckError::NoInstances));
    let mut overflowing = stage.instances.clone();
    overflowing[0].evals = usize::MAX;
    assert_eq!(
        attempt(stage.degree, &overflowing),
        Err(SumcheckError::EvaluationTotal)
    );
    // A relation the caller cannot evaluate, after the rounds and the evaluations.
    let mut transcript = session.context.transcript();
    let outcome = verify_batch(
        &mut transcript,
        stage.prefix,
        stage.degree,
        &stage.instances,
        &stage.proof,
        |i, point, evals| {
            if i == 1 {
                None
            } else {
                product(i, point, evals)
            }
        },
    );
    assert_eq!(outcome, Err(SumcheckError::Relation { instance: 1 }));
    // A tree level with no instance.
    let mut transcript = session.context.transcript();
    assert_eq!(
        verify_product_level(&mut transcript, 0, &[], &SumcheckProof::default()),
        Err(SumcheckError::NoInstances)
    );
    assert_eq!(*transcript.state(), untouched);
}

/// The degree bound is RFC-0014's: every degree from 1 to 7 is accepted for a batch of no
/// rounds, whose single evaluation is its claim.
#[test]
fn every_degree_from_one_to_seven_is_accepted() {
    let session = sessions().remove(0);
    let claim = Fp2::new(Fp::reduce(5), Fp::reduce(9));
    for degree in 1..=MAX_DEGREE {
        let mut transcript = session.context.transcript();
        let proof = SumcheckProof {
            rounds: vec![],
            final_evals: vec![claim],
        };
        let instance = Instance {
            claim,
            vars: 0,
            evals: 1,
        };
        assert!(
            verify_batch(
                &mut transcript,
                Prefix::Stage(0),
                degree,
                &[instance],
                &proof,
                product
            )
            .is_ok()
        );
    }
}
