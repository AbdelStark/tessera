// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Seeded random and mutated inputs (RFC-0019 'Robustness'): no input panics, every structural
//! defect gets exactly the error the crate's order of checks predicts, and no changed proof is
//! accepted with true claims.
//!
//! The oracles judge each result without `tv-core`'s code:
//!
//! - [`structural`] predicts the first failing check of a proof's shape, from the crate
//!   documentation's order alone.
//! - A proof whose shape is right and whose rounds differ from the honest proof's leaves the
//!   honest final evaluations at another point, so it fails the final check, except with
//!   probability `(n·d + m)/|Fp2|`.
//! - A proof that changes only final evaluations keeps the honest point, so the oracle computes
//!   the final check itself, from the honest prover's `γ` and last claim. When the check holds,
//!   as it does with two evaluations of one product exchanged, the proof is accepted, and its
//!   changed evaluation is a false claim on a table, which the stage consuming it rejects.
//! - A batch of trees runs whole, and its leaf claims are checked against the leaves.

mod support;

use support::{
    TreeOutcome, extension, from_reference, instances_of, product, proof_of, run_trees, sessions,
    tree_batches,
};
use tv_core::sumcheck::{Instance, MAX_DEGREE, Prefix, SumcheckError, SumcheckProof, verify_batch};
use tv_core::tree::TreeClaim;
use tv_field::{Fp, Fp2};
use tv_hash::transcript::{Ascii, Mode, ParamSetId, SessionContext, Transcript};
use tv_testkit::SplitMix64;
use tv_testkit::sumcheck::{self as reference, Fe, Product, ProductTree};

fn context() -> SessionContext<'static> {
    SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::Live,
        params: ParamSetId(Ascii::new("live-64/1").unwrap()),
        statement_digest: [0xa5; 32],
        session_id: [0x3c; 16],
    }
}

fn element(rng: &mut SplitMix64) -> Fp2 {
    Fp2::new(Fp::reduce(rng.next_u64()), Fp::reduce(rng.next_u64()))
}

fn below(rng: &mut SplitMix64, bound: usize) -> usize {
    (rng.next_u64() % bound as u64) as usize
}

/// The first check of the crate's order that a proof's shape fails, or `None` when its shape is
/// right: the caller's inputs, then round by round its presence and length, then rounds past
/// the last, then the number of final evaluations.
fn structural(
    degree: usize,
    instances: &[Instance],
    proof: &SumcheckProof,
) -> Option<SumcheckError> {
    if degree == 0 || degree > MAX_DEGREE {
        return Some(SumcheckError::Degree { degree });
    }
    let Some(n_max) = instances.iter().map(|i| i.vars).max() else {
        return Some(SumcheckError::NoInstances);
    };
    let Some(expected) = instances
        .iter()
        .try_fold(0usize, |t, i| t.checked_add(i.evals))
    else {
        return Some(SumcheckError::EvaluationTotal);
    };
    for j in 0..n_max {
        let round = j as u32;
        match proof.rounds.get(j) {
            None => return Some(SumcheckError::MissingRound { round }),
            Some(message) if message.len() != degree => {
                return Some(SumcheckError::RoundLength {
                    round,
                    expected: degree,
                    found: message.len(),
                });
            }
            Some(_) => {}
        }
    }
    if proof.rounds.len() > n_max {
        return Some(SumcheckError::ExtraRound {
            round: n_max as u32,
        });
    }
    if proof.final_evals.len() != expected {
        return Some(SumcheckError::EvaluationCount {
            round: n_max as u32,
            expected,
            found: proof.final_evals.len(),
        });
    }
    None
}

/// Changes `proof` by one to four random edits: a value changed, replaced, inserted or removed,
/// in a round or in the final evaluations; a round inserted, removed, duplicated or two
/// exchanged; two values exchanged; the rounds or the evaluations cut short.
fn mutate(proof: &mut SumcheckProof, rng: &mut SplitMix64) {
    for _ in 0..1 + below(rng, 4) {
        let rounds = proof.rounds.len();
        let evals = proof.final_evals.len();
        match below(rng, 14) {
            0 | 1 if rounds > 0 => {
                let j = below(rng, rounds);
                if !proof.rounds[j].is_empty() {
                    let k = below(rng, proof.rounds[j].len());
                    proof.rounds[j][k] += element(rng);
                }
            }
            2 if rounds > 0 => {
                let j = below(rng, rounds);
                let k = below(rng, proof.rounds[j].len() + 1);
                proof.rounds[j].insert(k, element(rng));
            }
            3 if rounds > 0 => {
                let j = below(rng, rounds);
                if !proof.rounds[j].is_empty() {
                    let k = below(rng, proof.rounds[j].len());
                    proof.rounds[j].remove(k);
                }
            }
            4 => {
                let length = below(rng, MAX_DEGREE + 3);
                let round = (0..length).map(|_| element(rng)).collect();
                proof.rounds.insert(below(rng, rounds + 1), round);
            }
            5 if rounds > 0 => {
                proof.rounds.remove(below(rng, rounds));
            }
            6 if rounds > 0 => {
                let copy = proof.rounds[below(rng, rounds)].clone();
                proof.rounds.insert(below(rng, rounds + 1), copy);
            }
            7 if rounds > 1 => {
                let (a, b) = (below(rng, rounds), below(rng, rounds));
                proof.rounds.swap(a, b);
            }
            8 if rounds > 0 => {
                let j = below(rng, rounds);
                let length = proof.rounds[j].len();
                if length > 1 {
                    proof.rounds[j].swap(below(rng, length), below(rng, length));
                }
            }
            9 | 10 if evals > 0 => {
                let k = below(rng, evals);
                proof.final_evals[k] += element(rng);
            }
            11 => {
                let k = below(rng, evals + 1);
                proof.final_evals.insert(k, element(rng));
            }
            12 if evals > 0 => {
                proof.final_evals.remove(below(rng, evals));
            }
            13 if evals > 1 => {
                let (a, b) = (below(rng, evals), below(rng, evals));
                proof.final_evals.swap(a, b);
            }
            _ => {
                if rng.next_u64() & 1 == 0 {
                    proof.rounds.truncate(below(rng, rounds + 1));
                } else {
                    proof.final_evals.truncate(below(rng, evals + 1));
                }
            }
        }
    }
}

/// `Σ_i γ^i · Π_k e_ik`: the final check's left side for products, computed here.
fn combination(gamma: Fp2, instances: &[Instance], evals: &[Fp2]) -> Fp2 {
    let mut rest = evals.iter();
    let mut weight = Fp2::ONE;
    let mut sum = Fp2::ZERO;
    for instance in instances {
        let value = rest
            .by_ref()
            .take(instance.evals)
            .fold(Fp2::ONE, |p, e| p * *e);
        sum += weight * value;
        weight *= gamma;
    }
    sum
}

/// What verifying `changed`, a change of the honest `proof` of a product batch, must give.
fn expected(
    degree: usize,
    instances: &[Instance],
    honest: &SumcheckProof,
    gamma: Fp2,
    last: Fp2,
    changed: &SumcheckProof,
) -> Result<(), SumcheckError> {
    if let Some(error) = structural(degree, instances, changed) {
        return Err(error);
    }
    let round = instances.iter().map(|i| i.vars).max().unwrap() as u32;
    if changed.rounds != honest.rounds
        || combination(gamma, instances, &changed.final_evals) != last
    {
        return Err(SumcheckError::FinalMismatch { round });
    }
    Ok(())
}

/// Random shapes, random values: no panic, and exactly the error [`structural`] predicts, or a
/// failed final check when the shape is right. Half the cases have the right number of rounds,
/// lengths and evaluations, so the final check is reached.
#[test]
fn random_inputs_get_the_predicted_error() {
    let mut rng = SplitMix64::new(0x5eed_0059_0003);
    let mut reached_final = 0;
    for case in 0..20_000 {
        let degree = below(&mut rng, MAX_DEGREE + 3);
        let instances: Vec<Instance> = (0..below(&mut rng, 5))
            .map(|_| Instance {
                claim: element(&mut rng),
                vars: below(&mut rng, 7),
                evals: below(&mut rng, 5),
            })
            .collect();
        let n_max = instances.iter().map(|i| i.vars).max().unwrap_or(0);
        let well_formed = rng.next_u64() & 1 == 0;
        let rounds = if well_formed {
            n_max
        } else {
            below(&mut rng, n_max + 3)
        };
        let proof = SumcheckProof {
            rounds: (0..rounds)
                .map(|_| {
                    let length = if well_formed && rng.next_u64() % 8 != 0 {
                        degree
                    } else {
                        below(&mut rng, MAX_DEGREE + 3)
                    };
                    (0..length).map(|_| element(&mut rng)).collect()
                })
                .collect(),
            final_evals: {
                let count = if well_formed {
                    instances.iter().map(|i| i.evals).sum()
                } else {
                    below(&mut rng, 12)
                };
                (0..count).map(|_| element(&mut rng)).collect()
            },
        };
        let mut transcript = Transcript::new(&context());
        let prefix = Prefix::Stage(below(&mut rng, 3) as u32);
        let outcome =
            verify_batch(&mut transcript, prefix, degree, &instances, &proof, product).map(|_| ());
        match structural(degree, &instances, &proof) {
            Some(error) => assert_eq!(outcome, Err(error), "case {case}"),
            None => {
                assert_eq!(
                    outcome,
                    Err(SumcheckError::FinalMismatch {
                        round: n_max as u32
                    }),
                    "case {case}"
                );
                reached_final += 1;
            }
        }
    }
    assert!(reached_final > 2_000, "{reached_final}");
}

/// Seeded instances of every shape, proven by the reference prover: each verifies, at the
/// prover's own challenges, and every random change of it gets exactly the predicted result.
#[test]
fn seeded_instances_verify_and_their_changes_do_not() {
    let mut rng = SplitMix64::new(0x5eed_0059_0004);
    let mut accepted_changes = 0;
    let mut changes = 0;
    for case in 0..300 {
        let degree = 1 + below(&mut rng, 4);
        let products: Vec<Product> = (0..1 + below(&mut rng, 4))
            .map(|_| {
                let vars = below(&mut rng, 7);
                let factors = 1 + below(&mut rng, degree);
                Product::new(
                    (0..factors)
                        .map(|_| {
                            (0..1 << vars)
                                .map(|_| Fe::new(rng.next_u64(), rng.next_u64()))
                                .collect()
                        })
                        .collect(),
                )
            })
            .collect();
        let context = context();
        let mut prover = reference::Transcript::new(&context.to_tce());
        let prefix = Prefix::Stage(case);
        let honest = reference::prove(&mut prover, &prefix.to_string(), degree, &products, None);
        let instances = instances_of(&products);
        let proof = proof_of(&honest);
        let mut transcript = Transcript::new(&context);
        let batch = verify_batch(&mut transcript, prefix, degree, &instances, &proof, product)
            .unwrap_or_else(|e| panic!("case {case}: {e}"));
        let challenges: Vec<Fp2> = honest
            .challenges
            .iter()
            .copied()
            .map(from_reference)
            .collect();
        assert_eq!(batch.point, challenges, "case {case}");
        assert_eq!(batch.gamma, from_reference(honest.gamma));
        assert_eq!(
            *transcript.state(),
            prover.state(),
            "case {case}: the transcripts"
        );
        let gamma = from_reference(honest.gamma);
        let last = from_reference(*honest.claims.last().unwrap());
        for _ in 0..40 {
            let mut changed = proof.clone();
            mutate(&mut changed, &mut rng);
            let mut transcript = Transcript::new(&context);
            let outcome = verify_batch(
                &mut transcript,
                prefix,
                degree,
                &instances,
                &changed,
                product,
            )
            .map(|_| ());
            if changed == proof {
                assert_eq!(outcome, Ok(()));
                continue;
            }
            assert_eq!(
                outcome,
                expected(degree, &instances, &proof, gamma, last, &changed),
                "case {case}"
            );
            if outcome.is_ok() {
                // Only evaluations changed, and the check still holds: the claim is false.
                assert_eq!(changed.rounds, proof.rounds);
                assert_ne!(changed.final_evals, proof.final_evals);
                accepted_changes += 1;
            }
            changes += 1;
        }
    }
    assert!(changes > 10_000, "{changes}");
    // Two evaluations of one product exchanged keep its value.
    assert!(accepted_changes > 0);
}

/// Every random change of every tree batch, from the vectors and from the reference prover on
/// seeded trees, is rejected by a level or leaves a false leaf claim; the first changed level
/// rejects a structural defect with the predicted error, and a changed round at its final check.
#[test]
fn changed_tree_batches_are_never_accepted() {
    let mut rng = SplitMix64::new(0x5eed_0059_0005);
    let mut changes = 0;
    for batch in tree_batches() {
        let (context, trees, proofs) = (&batch.context, &batch.trees, batch.proofs());
        let entering: Vec<Vec<TreeClaim>> = batch
            .levels
            .iter()
            .map(|level| level.entering.iter().map(|(_, c)| c.clone()).collect())
            .collect();
        for _ in 0..300 {
            let mut changed = proofs.clone();
            let level = below(&mut rng, changed.len());
            mutate(&mut changed[level], &mut rng);
            let mut transcript = context.transcript();
            let run = run_trees(&mut transcript, trees, &changed, |_, _| {});
            let Some(first) = (0..proofs.len()).find(|&l| changed[l] != proofs[l]) else {
                assert_eq!(run.outcome, TreeOutcome::Accepted);
                continue;
            };
            assert_ne!(run.outcome, TreeOutcome::Accepted);
            let instances: Vec<Instance> = entering[first]
                .iter()
                .map(|c| Instance {
                    claim: c.value,
                    vars: c.point.len(),
                    evals: 2,
                })
                .collect();
            if let Some(error) = structural(3, &instances, &changed[first]) {
                assert_eq!(
                    run.outcome,
                    TreeOutcome::Rejected {
                        level: first as u32,
                        error
                    }
                );
            } else if changed[first].rounds != proofs[first].rounds {
                assert_eq!(
                    run.outcome,
                    TreeOutcome::Rejected {
                        level: first as u32,
                        error: SumcheckError::FinalMismatch {
                            round: first as u32
                        }
                    }
                );
            }
            changes += 1;
        }
    }
    assert!(changes > 2_500, "{changes}");

    // Seeded batches of up to five trees of heights 1 to 5, proven by the reference prover.
    for case in 0..40 {
        let trees: Vec<ProductTree> = (0..1 + below(&mut rng, 5))
            .map(|_| {
                let height = 1 + below(&mut rng, 5);
                ProductTree::new((0..1 << height).map(|_| Fe::base(rng.next_u64())).collect())
            })
            .collect();
        let context = context();
        let mut prover = reference::Transcript::new(&context.to_tce());
        let levels = reference::prove_trees(&mut prover, &trees);
        let proofs: Vec<SumcheckProof> = levels.iter().map(|l| proof_of(&l.proof)).collect();
        let ours: Vec<support::Tree> = trees
            .iter()
            .map(|tree| support::Tree {
                height: tree.height(),
                root: from_reference(tree.root()).c0,
                leaves: tree
                    .leaves()
                    .iter()
                    .map(|&leaf| from_reference(leaf).c0)
                    .collect(),
            })
            .collect();
        let mut transcript = Transcript::new(&context);
        let run = run_trees(&mut transcript, &ours, &proofs, |_, _| {});
        assert_eq!(run.outcome, TreeOutcome::Accepted, "case {case}");
        assert_eq!(*transcript.state(), prover.state(), "case {case}");
        for (tree, claim) in &run.leaves {
            assert_eq!(extension(&ours[*tree].leaves, &claim.point), claim.value);
        }
        for _ in 0..40 {
            let mut changed = proofs.clone();
            let level = below(&mut rng, changed.len());
            mutate(&mut changed[level], &mut rng);
            let mut transcript = Transcript::new(&context);
            let run = run_trees(&mut transcript, &ours, &changed, |_, _| {});
            if changed == proofs {
                assert_eq!(run.outcome, TreeOutcome::Accepted);
            } else {
                assert_ne!(run.outcome, TreeOutcome::Accepted, "case {case}");
            }
        }
    }
}

/// The stages of the vectors, changed at random: the same oracle as for the reference prover's.
#[test]
fn changed_vector_stages_get_the_predicted_result() {
    let mut rng = SplitMix64::new(0x5eed_0059_0006);
    let mut changes = 0;
    for session in sessions() {
        for (index, stage) in session.stages.iter().enumerate() {
            for _ in 0..60 {
                let mut changed = stage.proof.clone();
                mutate(&mut changed, &mut rng);
                let mut transcript = session.transcript_before(index);
                let outcome =
                    support::verify(&mut transcript, stage, &stage.instances, &changed).map(|_| ());
                let last = stage.claims[stage.rounds()];
                let predicted = if changed == stage.proof {
                    Ok(())
                } else {
                    expected(
                        stage.degree,
                        &stage.instances,
                        &stage.proof,
                        stage.gamma,
                        last,
                        &changed,
                    )
                };
                assert_eq!(outcome, predicted, "{}, {}", session.name, stage.prefix);
                changes += 1;
            }
        }
    }
    assert_eq!(changes, 62 * 60);
}
