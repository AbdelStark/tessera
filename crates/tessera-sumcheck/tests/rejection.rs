// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Every change to an honest proof is rejected (RFC-0010 'Testing Strategy'): each round value
//! and each final evaluation of several honest proofs, padded batches among them, is changed in
//! turn, and the independent verifier rejects every changed proof at its final check,
//! `FinalMismatch { round: n_max }`, the only check a well-formed proof can fail.

mod common;

use common::{
    Fp2, Session, SplitMix64, TestInstance, TestTable, prove, tv_instances, tv_proof, verify,
};
use tessera_field::Field;
use tessera_sumcheck::{Prefix, Protocol};

/// `tessera-protocol/1`'s bound, RFC-0014's `d ≤ 7`, which `tv-core`'s checks hold.
const MAX_DEGREE: usize = Protocol::One.max_degree(Prefix::Stage(1));
use tv_core::sumcheck::SumcheckError;

/// Proves `instances`, checks that the verifier accepts, then changes each round value and each
/// final evaluation in turn by `δ` and checks that every change is rejected at round `n_max`.
fn every_change_is_rejected(
    rng: &mut SplitMix64,
    prefix: Prefix,
    degree: usize,
    instances: &[TestInstance],
) -> usize {
    let session = Session::random(rng);
    let (proven, _) = prove(&session, prefix, degree, instances);
    let claims = tv_instances(instances);
    let proof = proven.proof();
    let n_max = proof.rounds.len();
    let rejected = Err(SumcheckError::FinalMismatch {
        round: n_max as u32,
    });
    let check = |changed: &tessera_sumcheck::SumcheckProof<Fp2>| {
        verify(
            &session,
            prefix,
            degree,
            instances,
            &claims,
            &tv_proof(changed),
        )
        .0
    };
    assert!(check(proof).is_ok(), "the honest proof verifies");
    let mut changes = 0;
    for j in 0..n_max {
        for m in 0..degree {
            let mut changed = proof.clone();
            changed.rounds[j][m] += delta(rng);
            assert_eq!(check(&changed), rejected, "round {j}, value {m}");
            changes += 1;
        }
    }
    for e in 0..proof.final_evals.len() {
        let mut changed = proof.clone();
        changed.final_evals[e] += delta(rng);
        assert_eq!(check(&changed), rejected, "final evaluation {e}");
        changes += 1;
    }
    changes
}

/// A non-zero change: one, or a random element.
fn delta(rng: &mut SplitMix64) -> Fp2 {
    loop {
        let x = if rng.coin() { Fp2::ONE } else { rng.fp2() };
        if x != Fp2::ZERO {
            return x;
        }
    }
}

#[test]
fn every_change_to_a_single_product_is_rejected() {
    let mut rng = SplitMix64::new(0x53_0101);
    for degree in 1..=4 {
        for vars in [1, 5, 9] {
            let instance = TestInstance::random(&mut rng, vars, degree, |t| t % 2 == 0);
            let changes = every_change_is_rejected(&mut rng, Prefix::Stage(0), degree, &[instance]);
            assert_eq!(changes, vars * degree + degree);
        }
    }
}

#[test]
fn every_change_to_a_padded_batch_is_rejected() {
    let mut rng = SplitMix64::new(0x53_0102);
    // RFC-0010's batch of 3, 7 and 12 variables.
    let batch = [
        TestInstance::random(&mut rng, 3, 3, |_| true),
        TestInstance::random(&mut rng, 7, 2, |_| false),
        TestInstance::random(&mut rng, 12, 3, |t| t != 1),
    ];
    assert_eq!(
        every_change_is_rejected(&mut rng, Prefix::Stage(4), 3, &batch),
        12 * 3 + 8
    );
    // Degree 7, an instance on no variable, a product-tree-shaped instance with a public `eq`.
    let batch = [
        TestInstance::random(&mut rng, 2, 7, |t| t < 4),
        TestInstance::random(&mut rng, 0, 3, |_| true),
        TestInstance::new(vec![
            TestTable::eq(&mut rng, 5),
            TestTable::random(&mut rng, 5, true),
            TestTable::random(&mut rng, 5, true),
        ]),
        TestInstance::random(&mut rng, 4, 1, |_| false),
    ];
    assert_eq!(
        every_change_is_rejected(&mut rng, Prefix::Tree(2), MAX_DEGREE, &batch),
        5 * 7 + 7 + 3 + 2 + 1
    );
}

/// A batch with `n_max = 0` has no round: changing a final evaluation is caught by the claim
/// check at round 0.
#[test]
fn every_change_to_a_batch_without_rounds_is_rejected() {
    let mut rng = SplitMix64::new(0x53_0103);
    let batch = [
        TestInstance::random(&mut rng, 0, 2, |_| true),
        TestInstance::random(&mut rng, 0, 3, |_| false),
    ];
    assert_eq!(
        every_change_is_rejected(&mut rng, Prefix::Stage(1), 3, &batch),
        5
    );
}
