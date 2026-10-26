// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0013 'Degree': bounds by protocol and sumcheck kind; Newton interpolation through ten.
mod support;

use support::{Context, from_reference, instances_of, product, proof_of};
use tv_core::sumcheck::{self, Instance, Prefix, Protocol, SumcheckError, SumcheckProof};
use tv_field::{Fp, Fp2};
use tv_hash::transcript::{Ascii, Mode, ParamSetId, SessionContext};
use tv_testkit::SplitMix64;
use tv_testkit::sumcheck::{self as reference, Fe, Product};

fn context() -> Context {
    Context::of(&SessionContext {
        protocol: Ascii::new("tessera-protocol/2").unwrap(),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("test-small/1").unwrap()),
        statement_digest: [0x65; 32],
        session_id: [0x10; 16],
    })
}

fn batch(degree: usize, prefix: Prefix) -> (Context, Vec<Instance>, SumcheckProof) {
    let context = context();
    let mut rng = SplitMix64::new(0x0005_6510);
    let products: Vec<Product> = [3, 1, 0]
        .into_iter()
        .map(|vars| {
            Product::new(
                (0..degree)
                    .map(|_| {
                        (0..1 << vars)
                            .map(|_| Fe::new(rng.next_u64(), rng.next_u64()))
                            .collect()
                    })
                    .collect(),
            )
        })
        .collect();
    let proof = reference::prove(
        &mut context.reference_transcript(),
        &prefix.to_string(),
        degree,
        &products,
        None,
    );
    (context, instances_of(&products), proof_of(&proof))
}

#[test]
fn degree_eight_through_ten_verify_with_the_independent_reference_prover() {
    for degree in 8..=10 {
        for prefix in [Prefix::Stage(1), Prefix::Stage(2)] {
            let (context, instances, proof) = batch(degree, prefix);
            assert!(
                sumcheck::verify_batch_in(
                    &mut context.transcript(),
                    Protocol::Two,
                    prefix,
                    degree,
                    &instances,
                    &proof,
                    product,
                )
                .is_ok(),
                "degree {degree}, {prefix}"
            );
            let mut changed = proof.clone();
            changed.final_evals[0] += Fp2::ONE;
            assert_eq!(
                sumcheck::verify_batch_in(
                    &mut context.transcript(),
                    Protocol::Two,
                    prefix,
                    degree,
                    &instances,
                    &changed,
                    product,
                ),
                Err(SumcheckError::FinalMismatch { round: 3 })
            );
        }
    }
}

#[test]
fn unsupported_degrees_fail_before_any_transcript_operation() {
    for protocol in [Protocol::One, Protocol::Two] {
        for prefix in [
            Prefix::Stage(0),
            Prefix::Stage(1),
            Prefix::Tree(1),
            Prefix::GroupTree { group: 0, level: 1 },
            Prefix::TableTree(1),
            Prefix::Merge,
        ] {
            let max = if protocol == Protocol::Two && prefix == Prefix::Stage(1) {
                10
            } else {
                7
            };
            assert_eq!(protocol.max_degree(prefix), max);
            for degree in [0, max + 1, usize::MAX] {
                let mut transcript = context().transcript();
                let before = *transcript.state();
                let expected = SumcheckError::Degree { degree, max };
                assert_eq!(
                    sumcheck::verify_batch_in(
                        &mut transcript,
                        protocol,
                        prefix,
                        degree,
                        &[],
                        &SumcheckProof::default(),
                        product,
                    ),
                    Err(expected)
                );
                assert_eq!(
                    sumcheck::reduce_in(
                        &mut transcript,
                        protocol,
                        prefix,
                        degree,
                        0,
                        Fp2::ZERO,
                        &[],
                    ),
                    Err(expected)
                );
                assert_eq!(*transcript.state(), before);
                assert_eq!(expected.rejected_round(), None);
            }
        }
    }
}

#[test]
fn each_bad_length_is_rejected_at_its_exact_round() {
    let prefix = Prefix::Stage(1);
    let (context, instances, proof) = batch(10, prefix);
    for round in 0..3 {
        for found in [0, 7, 8, 9, 11] {
            let mut changed = proof.clone();
            changed.rounds[round].resize(found, Fp2::ONE);
            let error = SumcheckError::RoundLength {
                round: round as u32,
                expected: 10,
                found,
            };
            assert_eq!(
                sumcheck::verify_batch_in(
                    &mut context.transcript(),
                    Protocol::Two,
                    prefix,
                    10,
                    &instances,
                    &changed,
                    product,
                ),
                Err(error)
            );
            assert_eq!(error.rejected_round(), Some(round as u32));
        }
    }
    // Refuse the same degree-10 proof under the legacy entry point, even for zero variables.
    assert_eq!(
        sumcheck::verify_batch(
            &mut context.transcript(),
            prefix,
            10,
            &instances,
            &proof,
            product,
        ),
        Err(SumcheckError::Degree { degree: 10, max: 7 })
    );
}

#[test]
fn protocol_one_entry_points_keep_the_same_messages_and_challenges() {
    for degree in 1..=7 {
        let prefix = Prefix::Stage(2);
        let (context, instances, proof) = batch(degree, prefix);
        let mut one = context.transcript();
        let mut two = context.transcript();
        assert_eq!(
            sumcheck::verify_batch(&mut one, prefix, degree, &instances, &proof, product),
            sumcheck::verify_batch_in(
                &mut two,
                Protocol::Two,
                prefix,
                degree,
                &instances,
                &proof,
                product
            )
        );
        assert_eq!(one.state(), two.state());
    }
}

#[test]
fn degree_ten_reduction_interpolates_instead_of_truncating_high_coefficients() {
    let prefix = Prefix::Stage(1);
    let at = |x: u64| Fe::new(x, 0).pow(10);
    let context = context();
    let mut reference = context.reference_transcript();
    let message: Vec<_> = [0].into_iter().chain(2..=10).map(at).collect();
    reference.absorb("stage/1/round/0", &reference::tce(&message));
    let r = reference.challenge_fp2("stage/1/r/0");
    let expected = from_reference(r.pow(10));
    let reduced = sumcheck::reduce_in(
        &mut context.transcript(),
        Protocol::Two,
        prefix,
        10,
        1,
        Fp2::from(Fp::ONE),
        &[message.into_iter().map(from_reference).collect()],
    )
    .unwrap();
    assert_eq!(reduced.claim, expected);
}
