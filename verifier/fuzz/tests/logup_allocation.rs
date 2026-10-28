// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The memory `tv-core`'s LogUp checks allocate, counted by the `cap` allocator (RFC-0019
//! 'Robustness'). The roots check allocates the TCE bytes of the roots message and the root
//! claims, the leaf checks the TCE bytes of the values message and the claims they leave, and a
//! tree level what a product-forest level does; each only after the message's count is checked
//! against the caller's instances. So a hostile proof, however long its messages, makes a check
//! allocate no more than the honest proof of the same leaf stage does.
//!
//! The honest leaf stage comes from the test kit's reference prover. The allocator's counters are
//! global, so the cases run one after another in a single test.

use std::alloc::System;

use cap::Cap;
use tv_core::logup::{
    Challenges, InstanceClaims, Leaves, LookupError, RootClaims, check_leaves, check_roots,
};
use tv_core::sumcheck::{SumcheckError, SumcheckProof};
use tv_core::tree::{FractionClaim, LevelClaim, TreeClaim, verify_level};
use tv_field::{Fp, Fp2};
use tv_fuzz::sumcheck::context;
use tv_hash::transcript::{Ascii, Transcript, TranscriptError};
use tv_testkit::SplitMix64;
use tv_testkit::logup::{Forest, LogUp, StageClaim, StageTree, prove_leaf_stage};
use tv_testkit::sumcheck::{self as kit, Fe};

#[global_allocator]
static ALLOCATOR: Cap<System> = Cap::new(System, usize::MAX);

/// The bytes `f` allocates in all, and its result.
fn allocated_by<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let before = ALLOCATOR.total_allocated();
    let result = f();
    (result, ALLOCATOR.total_allocated() - before)
}

/// A million values: far more than any honest message.
const HOSTILE: usize = 1 << 20;

fn fp2(value: Fe) -> Fp2 {
    Fp2::new(
        Fp::from_canonical(value.0).unwrap(),
        Fp::from_canonical(value.1).unwrap(),
    )
}

fn fp2s(values: &[Fe]) -> Vec<Fp2> {
    values.iter().copied().map(fp2).collect()
}

/// A reference claim as the verifier holds it.
fn claim(claim: &StageClaim) -> LevelClaim {
    let point = fp2s(&claim.point);
    match claim.values[..] {
        [value] => LevelClaim::Product(TreeClaim {
            point,
            value: fp2(value),
        }),
        [numerator, denominator] => LevelClaim::Fraction(FractionClaim {
            point,
            numerator: fp2(numerator),
            denominator: fp2(denominator),
        }),
        _ => panic!("a claim holds one value or two"),
    }
}

fn fraction(claim: LevelClaim) -> FractionClaim {
    match claim {
        LevelClaim::Fraction(claim) => claim,
        LevelClaim::Product(_) => panic!("a fraction tree's claims"),
    }
}

fn hostile_logup_messages_allocate_no_more_than_honest_ones() {
    // Two instances, the second without values, and a forest of 2 trees of height 2.
    let mut rng = SplitMix64::new(0x5eed_f062_0001);
    let mut row = || (Fe::base(rng.next_u64()), Fe::base(rng.next_u64()));
    let table: Vec<(Fe, Fe)> = (0..6).map(|_| row()).collect();
    let range: Vec<(Fe, Fe)> = (0..4u64).map(|j| (Fe::base(j), Fe::ZERO)).collect();
    let logups = vec![
        LogUp::honest(table, true, &[0, 5, 5, 2, 1]),
        LogUp::honest(range, false, &[3, 1]),
    ];
    let forest = Forest::new(1, (1..=8).map(Fe::base).collect());
    let trees = [
        StageTree::Lookup(0),
        StageTree::Table(0),
        StageTree::Forest(0),
        StageTree::Lookup(1),
        StageTree::Table(1),
    ];
    let mut prover = kit::Transcript::new(&context().to_tce());
    let stage = prove_leaf_stage(&mut prover, &logups, &[forest], &trees, |_| {});
    let roots = fp2s(&stage.roots);
    let values = fp2s(&stage.values);
    assert_eq!(stage.levels.len(), 3);

    // The roots.
    let roots_check = |message: &[Fp2]| {
        let mut transcript = Transcript::new(&context());
        allocated_by(|| check_roots(&mut transcript, 2, message).map(|roots| roots.instances.len()))
    };
    let (outcome, honest) = roots_check(&roots);
    assert_eq!(outcome, Ok(2));
    assert!(honest < 1024, "the honest roots: {honest} bytes");
    let (outcome, bytes) = roots_check(&vec![Fp2::ONE; HOSTILE]);
    assert_eq!(
        outcome,
        Err(LookupError::RootsLength {
            instance: 1,
            expected: 8,
            found: HOSTILE
        })
    );
    assert!(
        bytes <= honest,
        "a million roots: {bytes} bytes, {honest} for the honest"
    );

    // A transcript through the roots and the levels before `upto`, with the claims entering each.
    let replay = |upto: usize| {
        let mut transcript = Transcript::new(&context());
        let _ = check_roots(&mut transcript, 2, &roots).unwrap();
        for (l, level) in stage.levels[..upto].iter().enumerate() {
            // The forests that join here take their root points first.
            for c in 0..level.root_challenges.len() {
                let label = format!("tree/{l}/root/{c}");
                let _ = transcript
                    .challenge_fp2(Ascii::new(&label).unwrap())
                    .unwrap();
            }
            let claims: Vec<LevelClaim> = level.entering.iter().map(claim).collect();
            let proof = SumcheckProof {
                rounds: level.proof.rounds.iter().map(|round| fp2s(round)).collect(),
                final_evals: fp2s(&level.proof.evals),
            };
            let _ = verify_level(&mut transcript, l as u32, &claims, &proof).unwrap();
        }
        transcript
    };

    // The last level, with every kind of instance: a million evaluations.
    let last = stage.levels.len() - 1;
    let level = &stage.levels[last];
    let claims: Vec<LevelClaim> = level.entering.iter().map(claim).collect();
    let honest_proof = SumcheckProof {
        rounds: level.proof.rounds.iter().map(|round| fp2s(round)).collect(),
        final_evals: fp2s(&level.proof.evals),
    };
    let level_check = |proof: &SumcheckProof| {
        let mut transcript = replay(last);
        allocated_by(|| {
            verify_level(&mut transcript, last as u32, &claims, proof)
                .map(|level| level.claims.len())
        })
    };
    let (outcome, honest) = level_check(&honest_proof);
    assert_eq!(outcome, Ok(claims.len()));
    let expected = honest_proof.final_evals.len();
    let mut hostile = honest_proof.clone();
    hostile.final_evals = vec![Fp2::ONE; HOSTILE];
    let (outcome, bytes) = level_check(&hostile);
    assert_eq!(
        outcome,
        Err(SumcheckError::EvaluationCount {
            round: level.proof.rounds.len() as u32,
            expected,
            found: HOSTILE
        })
    );
    assert!(
        bytes <= honest,
        "a million evaluations: {bytes} bytes, {honest} for the honest"
    );

    // The leaf checks: a million values.
    let leaves: Vec<Leaves> = (0..2)
        .map(|i| Leaves {
            lookup: fraction(claim(&stage.leaves[if i == 0 { 0 } else { 3 }])),
            table: fraction(claim(&stage.leaves[if i == 0 { 1 } else { 4 }])),
            has_values: i == 0,
        })
        .collect();
    let beta = fp2(stage.beta);
    let tables: Vec<Vec<Fp2>> = logups
        .iter()
        .map(|logup| fp2s(&logup.table_poly(stage.beta)))
        .collect();
    let leaf_check = |message: &[Fp2]| {
        let mut transcript = replay(stage.levels.len());
        let challenges = tv_core::logup::Challenges {
            alpha: fp2(stage.alpha),
            beta,
        };
        allocated_by(|| {
            check_leaves(
                &mut transcript,
                &challenges,
                &leaves,
                message,
                |i, point| {
                    // t̃(point) by binding one variable at a time, from the lowest; allocation is the
                    // caller's and counted, so the honest run's total includes it.
                    let mut table = tables.get(i)?.clone();
                    for &r in point {
                        table = table
                            .chunks_exact(2)
                            .map(|pair| pair[0] + r * (pair[1] - pair[0]))
                            .collect();
                    }
                    table.first().copied()
                },
            )
            .map(|claims| claims.instances.len())
        })
    };
    let (outcome, honest) = leaf_check(&values);
    assert_eq!(outcome, Ok(2));
    let (outcome, bytes) = leaf_check(&vec![Fp2::ONE; HOSTILE]);
    assert_eq!(
        outcome,
        Err(LookupError::ValuesLength {
            instance: 1,
            expected: 1,
            found: HOSTILE
        })
    );
    assert!(
        bytes <= honest,
        "a million values: {bytes} bytes, {honest} for the honest"
    );
}

/// Runs on the process's only thread (`harness = false`): libtest would keep a thread of its own,
/// whose allocations the process-wide cap counts against this test's budget (#501).
fn transcript_message_shortages_are_operational_and_do_not_absorb_partial_bytes() {
    for budget in [0usize, 7, 8] {
        let mut transcript = Transcript::new(&context());
        let mut expected = Transcript::new(&context());
        expected
            .challenge_fp2(Ascii::new("logup/alpha").unwrap())
            .unwrap();
        expected
            .challenge_fp2(Ascii::new("logup/beta").unwrap())
            .unwrap();
        let before = ALLOCATOR.total_allocated();
        ALLOCATOR
            .set_limit(ALLOCATOR.allocated().checked_add(budget).unwrap())
            .unwrap();
        let result = check_roots(&mut transcript, 0, &[]);
        ALLOCATOR.set_limit(usize::MAX).unwrap();
        let allocated = ALLOCATOR.total_allocated() - before;
        if budget < 8 {
            assert!(matches!(
                result,
                Err(LookupError::Transcript(TranscriptError::Memory))
            ));
            assert_eq!(allocated, 0);
        } else {
            assert!(result.unwrap().instances.is_empty());
            assert_eq!(allocated, 8);
            expected.absorb(Ascii::new("logup/roots").unwrap(), &0u64.to_le_bytes());
        }
        assert_eq!(transcript, expected);
    }
}

fn limited<T>(budget: usize, f: impl FnOnce() -> T) -> (T, usize, usize) {
    let before = ALLOCATOR.allocated();
    let total = ALLOCATOR.total_allocated();
    ALLOCATOR
        .set_limit(before.checked_add(budget).unwrap())
        .unwrap();
    let result = f();
    ALLOCATOR.set_limit(usize::MAX).unwrap();
    (
        result,
        ALLOCATOR.total_allocated() - total,
        ALLOCATOR.allocated() - before,
    )
}

fn root_claim_reservation_is_fallible_after_the_roots_message() {
    let roots = [Fp2::ZERO, Fp2::ONE, Fp2::ZERO, Fp2::ONE];
    let size = core::mem::size_of::<RootClaims>();
    assert!(size > 72);
    let mut payload = 4u64.to_le_bytes().to_vec();
    for root in roots {
        payload.extend_from_slice(&root.to_le_bytes());
    }
    for budget in [72, size - 1, size] {
        let mut transcript = Transcript::new(&context());
        let mut expected = Transcript::new(&context());
        expected
            .challenge_fp2(Ascii::new("logup/alpha").unwrap())
            .unwrap();
        expected
            .challenge_fp2(Ascii::new("logup/beta").unwrap())
            .unwrap();
        expected.absorb(Ascii::new("logup/roots").unwrap(), &payload);
        let (result, allocated, retained) =
            limited(budget, || check_roots(&mut transcript, 1, &roots));
        assert_eq!(result.is_ok(), budget >= size, "root budget {budget}");
        if budget < size {
            assert_eq!(result.unwrap_err(), LookupError::Memory);
            assert_eq!((allocated, retained), (72, 0));
        } else {
            let result = result.unwrap();
            assert_eq!(result.instances.len(), 1);
            assert_eq!((allocated, retained), (72 + size, size));
            assert!(result.instances[0].lookup.point.is_empty());
            assert_eq!(result.instances[0].lookup.denominator, Fp2::ONE);
        }
        assert_eq!(transcript, expected);
    }
    let mut transcript = Transcript::new(&context());
    let (result, allocated, retained) = limited(0, || check_roots(&mut transcript, 1, &[]));
    assert_eq!(
        result.unwrap_err(),
        LookupError::RootsLength {
            instance: 0,
            expected: 4,
            found: 0
        }
    );
    assert_eq!((allocated, retained), (0, 0));
    let (result, allocated, retained) = limited(0, || check_roots(&mut transcript, 0, &roots));
    assert_eq!(
        result.unwrap_err(),
        LookupError::RootsWithoutInstance { found: 4 }
    );
    assert_eq!((allocated, retained), (0, 0));
}

fn leaf_claim_and_each_point_reservation_are_fallible() {
    let challenges = Challenges {
        alpha: Fp2::ONE,
        beta: Fp2::ONE,
    };
    let size = core::mem::size_of::<InstanceClaims>();
    assert!(size > 24);
    for has_values in [true, false] {
        let leaves = [Leaves {
            has_values,
            lookup: FractionClaim {
                point: vec![Fp2::ONE, Fp2::ZERO],
                numerator: Fp2::ONE,
                denominator: Fp2::ZERO,
            },
            table: FractionClaim {
                point: vec![Fp2::ONE; 3],
                numerator: Fp2::ONE,
                denominator: Fp2::ONE,
            },
        }];
        let values = if has_values {
            vec![Fp2::ONE]
        } else {
            Vec::new()
        };
        let mut payload = u64::from(has_values).to_le_bytes().to_vec();
        for value in &values {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        let encoded = payload.len();
        let total = size + 32 + if has_values { 32 } else { 0 } + 48;
        let mut budgets = vec![
            encoded,
            size - 1,
            size,
            size + 31,
            size + 32,
            total - 1,
            total,
        ];
        if has_values {
            budgets.extend([size + 63, size + 64]);
        }
        for budget in budgets {
            let mut transcript = Transcript::new(&context());
            let mut expected = Transcript::new(&context());
            expected.absorb(Ascii::new("logup/values").unwrap(), &payload);
            let (result, allocated, retained) = limited(budget, || {
                check_leaves(&mut transcript, &challenges, &leaves, &values, |_, _| {
                    Some(Fp2::ZERO)
                })
            });
            assert_eq!(
                result.is_ok(),
                budget >= total,
                "leaf budget {budget}, values {has_values}"
            );
            if budget < total {
                assert_eq!(
                    result.unwrap_err(),
                    LookupError::Memory,
                    "budget {budget}, values {has_values}"
                );
                assert_eq!(retained, 0);
                assert!(allocated <= encoded + total);
            } else {
                let claims = result.unwrap();
                let claim = &claims.instances[0];
                assert_eq!(claim.key.point, leaves[0].lookup.point);
                assert_eq!(
                    claim.key.value,
                    if has_values { Fp2::ZERO } else { Fp2::ONE }
                );
                assert_eq!(
                    claim.value.as_ref().map(|v| (&v.point, v.value)),
                    has_values.then_some((&leaves[0].lookup.point, Fp2::ONE))
                );
                assert_eq!(claim.multiplicity.point, leaves[0].table.point);
                assert_eq!(claim.multiplicity.value, -Fp2::ONE);
                assert_eq!((allocated, retained), (encoded + total, total));
            }
            assert_eq!(transcript, expected);
        }
        for bad_table in [false, true] {
            let mut bad = leaves[0].clone();
            let error = if bad_table {
                bad.table.denominator = Fp2::ZERO;
                LookupError::TableEvaluation { instance: 0 }
            } else {
                bad.lookup.numerator = Fp2::ZERO;
                LookupError::LeafNumerator { instance: 0 }
            };
            let mut transcript = Transcript::new(&context());
            let (result, allocated, retained) = limited(size, || {
                check_leaves(
                    &mut transcript,
                    &challenges,
                    core::slice::from_ref(&bad),
                    &values,
                    |_, _| Some(Fp2::ZERO),
                )
            });
            assert_eq!(result.unwrap_err(), error);
            assert_eq!((allocated, retained), (encoded + size, 0));
        }
        let wrong = if has_values {
            Vec::new()
        } else {
            vec![Fp2::ONE]
        };
        let mut transcript = Transcript::new(&context());
        let expected = Transcript::new(&context());
        let (result, allocated, retained) = limited(0, || {
            check_leaves(&mut transcript, &challenges, &leaves, &wrong, |_, _| {
                Some(Fp2::ZERO)
            })
        });
        assert_eq!(
            result.unwrap_err(),
            LookupError::ValuesLength {
                instance: 0,
                expected: usize::from(has_values),
                found: wrong.len()
            }
        );
        assert_eq!((allocated, retained), (0, 0));
        assert_eq!(transcript, expected);
    }
}

fn main() {
    transcript_message_shortages_are_operational_and_do_not_absorb_partial_bytes();
    root_claim_reservation_is_fallible_after_the_roots_message();
    leaf_claim_and_each_point_reservation_are_fallible();
    hostile_logup_messages_allocate_no_more_than_honest_ones();
    println!("hostile_logup_messages_allocate_no_more_than_honest_ones: ok");
}
