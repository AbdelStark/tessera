// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Production shared-leaf driver against the independent Python vectors and test-kit prover.
//! The older in-test driver and explicit polynomial evaluations remain independent oracles.
mod support;

use support::logup::{self, Kind, LogUpOutcome, LogUpProof};
use support::{Context, Forest, root_claim};
use tv_core::leaf::{Error, Proof, Schedule, Tree};
use tv_core::tree::LevelClaim;
use tv_field::Fp2;

fn schedule(
    instances: &[logup::Instance],
    forests: &[Forest],
    order: &[logup::TreeRef],
) -> Schedule {
    let trees: Vec<_> = order
        .iter()
        .map(|tree| match tree.kind {
            Kind::Lookup => Tree::Lookup {
                instance: tree.index,
                height: instances[tree.index].lookup_height() as u32,
                has_values: instances[tree.index].has_values,
            },
            Kind::Table => Tree::Table {
                instance: tree.index,
                height: instances[tree.index].table_height() as u32,
            },
            Kind::Forest => Tree::Forest {
                height: forests[tree.index].height as u32,
                trees_log: forests[tree.index].rows_log as u32,
            },
        })
        .collect();
    Schedule::new(&trees).unwrap()
}

fn compare(
    context: &Context,
    instances: &[logup::Instance],
    forests: &[Forest],
    order: &[logup::TreeRef],
    proof: &LogUpProof,
) {
    let mut reference_transcript = context.transcript();
    let reference = logup::run_logup(
        &mut reference_transcript,
        instances,
        forests,
        order,
        proof,
        |_, _| {},
    );
    let plan = schedule(instances, forests, order);
    let mut transcript = context.transcript();
    let actual = plan.verify(
        &mut transcript,
        Proof {
            roots: &proof.roots,
            levels: &proof.levels,
            values: &proof.values,
        },
        |position, point| root_claim(&forests[order[position].index].roots, point),
        |instance, beta, point| instances.get(instance).map(|i| i.table_eval(beta, point)),
    );
    match reference.outcome {
        LogUpOutcome::Accepted | LogUpOutcome::False(_) => {
            let checked = actual.unwrap();
            assert_eq!(checked.lookups, reference.claims.unwrap());
            let expected: Vec<_> = reference
                .leaves
                .iter()
                .enumerate()
                .filter_map(|(tree, claim)| match claim {
                    LevelClaim::Product(claim) => Some(tv_core::leaf::ForestClaim {
                        tree,
                        claim: claim.clone(),
                    }),
                    _ => None,
                })
                .collect();
            assert_eq!(
                checked.forests, expected,
                "every forest debt is retained in schedule order"
            );
            let point: Vec<_> = reference.levels.last().map_or_else(Vec::new, |last| {
                core::iter::once(last.outcome.rho)
                    .chain(last.outcome.point.iter().rev().copied())
                    .collect()
            });
            assert_eq!(
                checked.point, point,
                "RFC-0045 point includes forest-expanded rounds"
            );
        }
        LogUpOutcome::Roots(error) | LogUpOutcome::Leaves(error) => {
            assert_eq!(actual, Err(Error::Lookup(error)))
        }
        LogUpOutcome::Level { level, error } => {
            assert_eq!(actual, Err(Error::Level { level, error }))
        }
        LogUpOutcome::Inconclusive { .. } => {
            panic!("ordinary hashing cannot reach a forced challenge failure")
        }
    }
    assert_eq!(
        transcript.state(),
        reference_transcript.state(),
        "same stopping transcript"
    );
    assert_eq!(transcript.counter(), reference_transcript.counter());
}

#[test]
fn every_python_batch_preserves_the_exact_error_transcript_and_outstanding_claims() {
    let batches = logup::logup_batches().unwrap();
    assert!(batches.len() > 10);
    for batch in batches {
        compare(
            &batch.context,
            &batch.instances,
            &batch.forests,
            &batch.trees,
            &batch.proof(),
        );
    }
}

#[test]
fn seeded_batches_and_single_element_proof_mutations_match_the_independent_driver() {
    let subjects = logup::subjects(24);
    assert_eq!(subjects.len(), 34);
    let mut mutations = 0;
    for subject in subjects {
        let run = |proof: &LogUpProof| {
            compare(
                &subject.context,
                &subject.instances,
                &subject.forests,
                &subject.trees,
                proof,
            )
        };
        run(&subject.proof);
        for index in 0..subject.proof.roots.len() {
            let mut bad = subject.proof.clone();
            bad.roots[index] += Fp2::ONE;
            run(&bad);
            mutations += 1;
        }
        for index in 0..subject.proof.values.len() {
            let mut bad = subject.proof.clone();
            bad.values[index] += Fp2::ONE;
            run(&bad);
            mutations += 1;
        }
        for (level, message) in subject.proof.levels.iter().enumerate() {
            for (round, values) in message.rounds.iter().enumerate() {
                for entry in 0..values.len() {
                    let mut bad = subject.proof.clone();
                    bad.levels[level].rounds[round][entry] += Fp2::ONE;
                    run(&bad);
                    mutations += 1;
                }
            }
            for entry in 0..message.final_evals.len() {
                let mut bad = subject.proof.clone();
                bad.levels[level].final_evals[entry] += Fp2::ONE;
                run(&bad);
                mutations += 1;
            }
        }
    }
    assert!(
        mutations > 1000,
        "actual single-element adversarial coverage: {mutations}"
    );
}

#[test]
fn invalid_own_schedules_are_internal_errors() {
    let lookup = Tree::Lookup {
        instance: 0,
        height: 0,
        has_values: false,
    };
    let table = Tree::Table {
        instance: 0,
        height: 0,
    };
    for trees in [
        vec![lookup],
        vec![table],
        vec![lookup, lookup, table],
        vec![lookup, table, table],
        vec![
            Tree::Lookup {
                instance: usize::MAX,
                height: 1,
                has_values: false,
            },
            table,
        ],
        vec![
            lookup,
            Tree::Table {
                instance: 1,
                height: 0,
            },
        ],
        vec![
            Tree::Lookup {
                instance: 0,
                height: 64,
                has_values: false,
            },
            table,
        ],
        vec![Tree::Forest {
            height: 0,
            trees_log: 1,
        }],
        vec![Tree::Forest {
            height: 63,
            trees_log: 1,
        }],
        vec![Tree::Forest {
            height: 1,
            trees_log: u32::MAX,
        }],
    ] {
        assert_eq!(Schedule::new(&trees).unwrap_err(), Error::Internal);
    }
    assert_eq!(Schedule::new(&[lookup, table]).unwrap().height(), 0);
    assert_eq!(Schedule::new(&[lookup, table]).unwrap().instances(), 1);
    assert_eq!(
        Schedule::new(&[Tree::Forest {
            height: 63,
            trees_log: 0
        }])
        .unwrap()
        .height(),
        63
    );
}

#[test]
fn wrong_level_counts_are_malformed_before_any_challenge_or_callback() {
    let subject = logup::subjects(0)
        .into_iter()
        .find(|s| !s.proof.levels.is_empty())
        .unwrap();
    let plan = schedule(&subject.instances, &subject.forests, &subject.trees);
    for count in [
        0,
        subject.proof.levels.len() - 1,
        subject.proof.levels.len() + 1,
    ] {
        let mut proofs = subject.proof.levels.clone();
        proofs.resize_with(count, || tv_core::sumcheck::SumcheckProof {
            rounds: vec![],
            final_evals: vec![],
        });
        let mut t = subject.context.transcript();
        let state = *t.state();
        assert_eq!(
            plan.verify(
                &mut t,
                Proof {
                    roots: &[],
                    levels: &proofs,
                    values: &[]
                },
                |_, _| panic!("no root callback"),
                |_, _, _| panic!("no table callback")
            ),
            Err(Error::LevelCount {
                expected: plan.height(),
                found: count
            })
        );
        assert_eq!(*t.state(), state);
        assert_eq!(t.counter(), 0);
    }
}

#[test]
fn empty_leaf_stage_absorbs_empty_messages_and_preserves_no_debts() {
    let plan = Schedule::new(&[]).unwrap();
    let t = logup::seeded_context();
    // The context is public and creates a real verifier transcript.
    let mut transcript = tv_hash::transcript::Transcript::new(&t);
    let checked = plan
        .verify(
            &mut transcript,
            Proof {
                roots: &[],
                levels: &[],
                values: &[],
            },
            |_, _| panic!("no forest"),
            |_, _, _| panic!("no table"),
        )
        .unwrap();
    assert!(
        checked.lookups.instances.is_empty()
            && checked.forests.is_empty()
            && checked.point.is_empty()
    );
    let mut expected = tv_hash::transcript::Transcript::new(&t);
    let _roots = tv_core::logup::check_roots(&mut expected, 0, &[]).unwrap();
    let _leaves = tv_core::logup::check_leaves(
        &mut expected,
        &tv_core::logup::Challenges {
            alpha: Fp2::ZERO,
            beta: Fp2::ZERO,
        },
        &[],
        &[],
        |_, _| None,
    )
    .unwrap();
    assert_eq!(transcript.state(), expected.state());
    assert_eq!(transcript.counter(), expected.counter());
}

#[test]
fn missing_public_forest_or_table_data_blames_the_verifier() {
    let subject = logup::subjects(0)
        .into_iter()
        .find(|s| !s.forests.is_empty() && !s.instances.is_empty())
        .unwrap();
    let plan = schedule(&subject.instances, &subject.forests, &subject.trees);
    let proof = Proof {
        roots: &subject.proof.roots,
        levels: &subject.proof.levels,
        values: &subject.proof.values,
    };
    assert_eq!(
        plan.verify(
            &mut subject.context.transcript(),
            proof,
            |_, _| None,
            |_, _, _| None
        ),
        Err(Error::Internal)
    );
    assert!(matches!(
        plan.verify(
            &mut subject.context.transcript(),
            proof,
            |position, point| root_claim(
                &subject.forests[subject.trees[position].index].roots,
                point
            ),
            |_, _, _| None
        ),
        Err(Error::Lookup(
            tv_core::logup::LookupError::TableUnavailable { .. }
        ))
    ));
}
