// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A prover that lies in one round is caught unless it also forges the final evaluations
//! (RFC-0010 'Soundness of the schedule'), and then the lie has moved into an evaluation claim
//! that is false, which the stage consuming it or the opening checks.
//!
//! The provers are the test kit's reference prover, which shares no code with `tv-core`. In
//! round `j` the liar adds a nonzero constant to every value it sends, `p_j(0), p_j(2), …,
//! p_j(d)`, then continues honestly with the challenges the transcript gives it. With `p(1)`
//! derived, every message is consistent with its claim, so nothing fails before the final check:
//! the claim after round `j` is `p'_j(r_j)` instead of the true `p_j(r_j)`, a polynomial
//! identity that fails except with probability `d/|Fp2|`, and the error carries to the end.
//!
//! Over `Fp2` the rate of a lucky lie cannot be measured; the calibration of RFC-0010 runs over
//! `F65537` on the prover side (#104). These tests check every round of each shape instead.

mod support;

use support::{from_reference, instances_of, product, proof_of};
use tv_core::sumcheck::{Prefix, SumcheckError, verify_batch};
use tv_hash::transcript::{Ascii, Mode, ParamSetId, SessionContext, Transcript};
use tv_testkit::SplitMix64;
use tv_testkit::sumcheck::{self as reference, Fe, Lie, Product};

fn context() -> SessionContext<'static> {
    SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("test-small/1").unwrap()),
        statement_digest: [0x59; 32],
        session_id: [0x10; 16],
    }
}

fn random(rng: &mut SplitMix64) -> Fe {
    Fe::new(rng.next_u64(), rng.next_u64())
}

/// Products of random tables, one `(vars, factors)` each.
fn products(shape: &[(usize, usize)], rng: &mut SplitMix64) -> Vec<Product> {
    shape
        .iter()
        .map(|&(vars, factors)| {
            Product::new(
                (0..factors)
                    .map(|_| (0..1 << vars).map(|_| random(rng)).collect())
                    .collect(),
            )
        })
        .collect()
}

/// The reference prover's proof of `products` under stage 7, with `lie`, and the verifier's
/// result on a transcript of the same session.
fn prove_and_verify(
    degree: usize,
    products: &[Product],
    lie: Option<Lie>,
) -> (
    reference::Proof,
    Result<tv_core::sumcheck::Batch, SumcheckError>,
) {
    let context = context();
    let mut prover = reference::Transcript::new(&context.to_tce());
    let proof = reference::prove(&mut prover, "stage/7", degree, products, lie);
    let mut transcript = Transcript::new(&context);
    let outcome = verify_batch(
        &mut transcript,
        Prefix::Stage(7),
        degree,
        &instances_of(products),
        &proof_of(&proof),
        product,
    );
    (proof, outcome)
}

#[test]
fn a_lie_in_any_round_is_caught_unless_the_final_evaluations_are_forged() {
    let mut rng = SplitMix64::new(0x5eed_0059_0001);
    // A single instance; a front-padded batch whose rounds 0 to 3 bind padding of the second
    // instance and every round padding of the third; degree 1, whose message is p(0) alone; and
    // two instances of one size.
    let shapes: [(usize, &[(usize, usize)]); 4] = [
        (3, &[(6, 3)]),
        (3, &[(6, 2), (2, 3), (0, 1)]),
        (1, &[(5, 1)]),
        (4, &[(4, 4), (4, 2)]),
    ];
    let mut lies = 0;
    for (degree, shape) in shapes {
        let products = products(shape, &mut rng);
        let n_max = shape.iter().map(|&(vars, _)| vars).max().unwrap();
        let (honest, outcome) = prove_and_verify(degree, &products, None);
        let batch = outcome.expect("the honest proof verifies");
        let challenges: Vec<_> = honest
            .challenges
            .iter()
            .copied()
            .map(from_reference)
            .collect();
        assert_eq!(batch.point, challenges);
        for round in 0..n_max {
            for offset in [Fe::ONE, random(&mut rng)] {
                assert_ne!(offset, Fe::ZERO);
                let lie = Lie {
                    round,
                    offset,
                    value: None,
                    forge: false,
                };
                let (lying, outcome) = prove_and_verify(degree, &products, Some(lie));
                assert_ne!(lying.rounds[round], honest.rounds[round]);
                assert_eq!(
                    outcome,
                    Err(SumcheckError::FinalMismatch {
                        round: n_max as u32
                    }),
                    "degree {degree}, {shape:?}: a lie in round {round}"
                );
                // The liar's honest evaluations do not give the claim its lie left.
                assert_eq!(lying.evals, lying.true_evals);

                let (forged, outcome) =
                    prove_and_verify(degree, &products, Some(Lie { forge: true, ..lie }));
                let batch = outcome.expect("forged evaluations pass the final check");
                let challenges: Vec<_> = forged
                    .challenges
                    .iter()
                    .copied()
                    .map(from_reference)
                    .collect();
                assert_eq!(batch.point, challenges);
                // The lie has moved into instance 0's first evaluation, the only one changed:
                // a false claim on its table at the point.
                assert_ne!(forged.evals[0], forged.true_evals[0]);
                assert_eq!(forged.evals[1..], forged.true_evals[1..]);
                let point = &forged.challenges[n_max - products[0].vars()..];
                assert_eq!(
                    reference::mle(&products[0].tables[0], point),
                    forged.true_evals[0]
                );
                lies += 1;
            }
        }
    }
    assert_eq!(lies, 2 * (6 + 6 + 5 + 4));
}

/// A liar who changes one value of one round's message, and answers every later challenge
/// honestly, is caught at the final check too, for every value of every round; forging the final
/// evaluations again moves the lie into a false evaluation claim.
#[test]
fn a_lie_in_one_value_of_one_round_is_caught() {
    let mut rng = SplitMix64::new(0x5eed_0059_0002);
    let degree = 3;
    let products = products(&[(5, 3), (3, 2)], &mut rng);
    let (honest, _) = prove_and_verify(degree, &products, None);
    let mut lies = 0;
    for round in 0..5 {
        for value in 0..degree {
            let lie = Lie {
                round,
                offset: random(&mut rng),
                value: Some(value),
                forge: false,
            };
            let (lying, outcome) = prove_and_verify(degree, &products, Some(lie));
            assert_eq!(lying.rounds[..round], honest.rounds[..round]);
            let changed: Vec<usize> = (0..degree)
                .filter(|&k| lying.rounds[round][k] != honest.rounds[round][k])
                .collect();
            assert_eq!(changed, [value]);
            assert_eq!(
                outcome,
                Err(SumcheckError::FinalMismatch { round: 5 }),
                "round {round}, value {value}"
            );
            let (forged, outcome) =
                prove_and_verify(degree, &products, Some(Lie { forge: true, ..lie }));
            assert!(outcome.is_ok(), "round {round}, value {value}");
            assert_ne!(forged.evals[0], forged.true_evals[0]);
            lies += 1;
        }
    }
    assert_eq!(lies, 15);
}
