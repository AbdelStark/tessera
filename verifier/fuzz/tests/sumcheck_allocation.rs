// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The memory `tv-core`'s sumcheck checks allocate, counted by the `cap` allocator (RFC-0019
//! 'Robustness'). A check allocates its labels, the challenges of the rounds it reads, and the
//! TCE bytes of a message only after the message's length or count is checked; a tree level also
//! its instances and their next claims, which the caller's claims bound. So a hostile proof,
//! however long its messages and however many its rounds or evaluations, makes a check allocate
//! no more than the honest proof of the same batch does.
//!
//! The allocator's counters are global, so the cases run one after another in a single test.

use std::alloc::System;

use cap::Cap;
use tv_core::sumcheck::{SumcheckError, SumcheckProof, verify_batch};
use tv_core::tree::{TreeClaim, verify_product_level};
use tv_field::Fp2;
use tv_fuzz::sumcheck::{Case, case, context};
use tv_hash::transcript::Transcript;

#[global_allocator]
static ALLOCATOR: Cap<System> = Cap::new(System, usize::MAX);

/// The bytes `f` allocates in all, and its result.
fn allocated_by<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let before = ALLOCATOR.total_allocated();
    let result = f();
    (result, ALLOCATOR.total_allocated() - before)
}

fn product(_: usize, _: &[Fp2], evals: &[Fp2]) -> Option<Fp2> {
    Some(evals.iter().fold(Fp2::ONE, |p, e| p * *e))
}

/// A million values: far more than any honest message.
const HOSTILE: usize = 1 << 20;

#[test]
fn hostile_proofs_allocate_no_more_than_honest_ones() {
    // A stage of degree 3: instances on 6, 2 and 0 variables, of 2, 3 and 1 tables.
    let Case::Stage(stage) = case(&[0, 2, 2, 6, 1, 2, 2, 0, 0, 5, 1, 2, 3, 4, 5, 6, 7, 8]) else {
        panic!("a stage case")
    };
    assert_eq!(stage.challenges.len(), 6);
    let verify = |proof: &SumcheckProof| {
        let mut transcript = Transcript::new(&context());
        verify_batch(
            &mut transcript,
            stage.prefix,
            stage.degree,
            &stage.instances,
            proof,
            product,
        )
        .map(|batch| batch.point.len())
    };
    let (outcome, honest) = allocated_by(|| verify(&stage.honest));
    assert_eq!(outcome, Ok(6));
    assert!(honest < 8 * 1024, "the honest proof: {honest} bytes");

    let mut long_round = stage.honest.clone();
    long_round.rounds[3] = vec![Fp2::ONE; HOSTILE];
    let mut many_rounds = stage.honest.clone();
    many_rounds
        .rounds
        .extend(std::iter::repeat_n(vec![Fp2::ONE; 3], HOSTILE / 8));
    let mut many_evals = stage.honest.clone();
    many_evals.final_evals = vec![Fp2::ONE; HOSTILE];
    for (proof, expected, what) in [
        (
            long_round,
            SumcheckError::RoundLength {
                round: 3,
                expected: 3,
                found: HOSTILE,
            },
            "a round of a million values",
        ),
        (
            many_rounds,
            SumcheckError::ExtraRound { round: 6 },
            "131,072 rounds past the last",
        ),
        (
            many_evals,
            SumcheckError::EvaluationCount {
                round: 6,
                expected: 6,
                found: HOSTILE,
            },
            "a million evaluations",
        ),
    ] {
        let (outcome, bytes) = allocated_by(|| verify(&proof));
        assert_eq!(outcome, Err(expected), "{what}");
        assert!(
            bytes <= honest,
            "{what}: {bytes} bytes, {honest} for the honest proof"
        );
    }

    // A tree batch: a million evaluations at its last level.
    let Case::Trees(trees) = case(&[1, 1, 4, 200, 2, 200, 9, 9, 9, 9, 9, 9, 9, 9]) else {
        panic!("a tree case")
    };
    let last = trees.honest.len() - 1;
    let level = |proof: &SumcheckProof, claims: &[TreeClaim]| {
        let mut transcript = Transcript::new(&context());
        for (l, earlier) in trees.honest[..last].iter().enumerate() {
            // Only the transcript an earlier level leaves matters here.
            let _ = verify_product_level(&mut transcript, l as u32, &trees.entering[l], earlier)
                .unwrap();
        }
        allocated_by(|| {
            verify_product_level(&mut transcript, last as u32, claims, proof)
                .map(|level| level.claims.len())
        })
    };
    let (outcome, honest) = level(&trees.honest[last], &trees.entering[last]);
    assert_eq!(outcome, Ok(2));
    let mut hostile = trees.honest[last].clone();
    hostile.final_evals = vec![Fp2::ONE; HOSTILE];
    let (outcome, bytes) = level(&hostile, &trees.entering[last]);
    assert_eq!(
        outcome,
        Err(SumcheckError::EvaluationCount {
            round: last as u32,
            expected: 4,
            found: HOSTILE
        })
    );
    assert!(
        bytes <= honest,
        "{bytes} bytes, {honest} for the honest level"
    );
}
