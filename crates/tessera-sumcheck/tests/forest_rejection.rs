// SPDX-License-Identifier: Apache-2.0 OR MIT

//! False forests and changed proofs are rejected (RFC-0010 'Tree levels', 'Testing Strategy'),
//! each at the exact level and check where the verifier driver of `common::forests` stops:
//!
//! - a row whose product is not zero, claimed zero as `SoftmaxRow` claims, fails the final check
//!   of the level where its forest joins;
//! - each round value and each evaluation of every level, changed in turn, fails that level's
//!   final check, or, when the evaluation's partner is zero and the check cannot see it, the next
//!   level's, or the leaf claim;
//! - a root challenge other than the transcript's fails the final check of its level.

mod common;

use common::forests::{
    Rejected, TestForest, mle, prove, shapes, true_roots, tv_proofs, verify_forests,
};
use common::{Fp, Fp2, Session, SplitMix64, TvFp2, tv_fp2, tv_proof};
use tessera_field::Field;
use tessera_hash::Transcript;
use tessera_sumcheck::{Forest, SumcheckTranscript, prove_forests};
use tv_core::sumcheck::SumcheckError;

/// The public root values of `SoftmaxRow`: zero for every row.
fn zero_roots(forests: &[TestForest]) -> Vec<Vec<TvFp2>> {
    forests
        .iter()
        .map(|forest| vec![TvFp2::ZERO; 1 << forest.trees_log])
        .collect()
}

/// The verifier's verdict on the honest proof of `forests` against the public roots `roots`.
fn verdict(
    session: &Session,
    forests: &[TestForest],
    roots: &[Vec<TvFp2>],
) -> Result<(), Rejected> {
    let (proven, _) = prove(session, forests);
    verify_forests(session, &shapes(forests), roots, &tv_proofs(&proven), None).map(|_| ())
}

fn final_mismatch(level: usize, rounds: usize) -> Result<(), Rejected> {
    Err(Rejected {
        level,
        error: SumcheckError::FinalMismatch {
            round: rounds as u32,
        },
    })
}

#[test]
fn a_nonzero_row_claimed_zero_is_rejected_where_its_forest_joins() {
    let mut rng = SplitMix64::new(0x58_0101);
    let session = Session::random(&mut rng);
    // A forest of 2^3 rows of 2^3 keys whose row 5 has no zero key, joining at level 3 beside a
    // forest of 2^2 zero rows of 2^6 keys: level 3 has 2 + 3 and 3 variables.
    let tall = TestForest::random(&mut rng, 2, 6, true);
    let mut late = TestForest::random(&mut rng, 3, 3, true);
    for leaf in &mut late.leaves[5 << 3..6 << 3] {
        *leaf = rng.fp() + Fp::ONE;
    }
    let forests = [tall, late];
    assert_ne!(forests[1].roots()[5], TvFp2::ZERO);
    assert_eq!(
        verdict(&session, &forests, &true_roots(&forests)),
        Ok(()),
        "the true roots verify"
    );
    assert_eq!(
        verdict(&session, &forests, &zero_roots(&forests)),
        final_mismatch(3, 5)
    );

    // A single tree with no zero leaf, alone: its root level has no variable, so the level has
    // no round and fails its claim check directly.
    let single = [TestForest::random(&mut rng, 0, 4, false)];
    assert_eq!(
        verdict(&session, &single, &zero_roots(&single)),
        final_mismatch(0, 0)
    );
    // The same tree joining at level 4 beside 2^2 zero rows of 2^6 keys: level 4 has 2 + 4 and
    // 0 variables.
    let forests = [
        TestForest::random(&mut rng, 2, 6, true),
        TestForest::random(&mut rng, 0, 2, false),
    ];
    assert_eq!(
        verdict(&session, &forests, &zero_roots(&forests)),
        final_mismatch(4, 6)
    );
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

/// Changes each round value and each evaluation of each level of the honest proof of `forests`
/// in turn, and checks that the verifier rejects each change at that level's final check.
fn every_change_is_rejected(rng: &mut SplitMix64, forests: &[TestForest]) -> usize {
    let session = Session::random(rng);
    let (proven, _) = prove(&session, forests);
    let roots = true_roots(forests);
    let proofs = tv_proofs(&proven);
    let check = |proofs: &[tv_core::sumcheck::SumcheckProof]| {
        verify_forests(&session, &shapes(forests), &roots, proofs, None).map(|_| ())
    };
    assert_eq!(check(&proofs), Ok(()), "the honest proof verifies");
    let mut changes = 0;
    for (level, proof) in proven.levels().iter().enumerate() {
        let rounds = proof.proof().rounds.len();
        for j in 0..rounds {
            for m in 0..3 {
                let mut changed = proof.proof().clone();
                changed.rounds[j][m] += delta(rng);
                let mut all = proofs.clone();
                all[level] = tv_proof(&changed);
                assert_eq!(
                    check(&all),
                    final_mismatch(level, rounds),
                    "level {level}, round {j}, value {m}"
                );
                changes += 1;
            }
        }
        for e in 0..proof.proof().final_evals.len() {
            let mut changed = proof.proof().clone();
            changed.final_evals[e] += delta(rng);
            let mut all = proofs.clone();
            all[level] = tv_proof(&changed);
            assert_eq!(
                check(&all),
                final_mismatch(level, rounds),
                "level {level}, evaluation {e}"
            );
            changes += 1;
        }
    }
    changes
}

#[test]
fn every_change_to_a_level_is_rejected() {
    let mut rng = SplitMix64::new(0x58_0102);
    // Heights 4, 6 and 2 join at levels 2, 0 and 4; the levels have 0 to 5 variables, and level 0
    // none: 15 rounds of 3 values and 24 evaluations.
    let forests = [
        TestForest::random(&mut rng, 2, 4, false),
        TestForest::random(&mut rng, 0, 6, false),
        TestForest::random(&mut rng, 3, 2, false),
    ];
    assert_eq!(every_change_is_rejected(&mut rng, &forests), 15 * 3 + 24);
    // 2^4 trees of 2^5 leaves alone, with zero roots.
    let forests = [TestForest::random(&mut rng, 4, 5, true)];
    assert_eq!(
        every_change_is_rejected(&mut rng, &forests),
        (4 + 5 + 6 + 7 + 8) * 3 + 2 * 5
    );
}

/// A tree whose left half multiplies to zero has `u = 0` at the level where it joins, so the
/// level's check `γ^i·eq(r_i, r'_i)·u_i·w_i` cannot see a change to `w`. The change still moves
/// the next claim, `u + ρ·(w − u)`, which the next level rejects, or, after the last level, the
/// stage that checks the leaf claim against the leaves.
#[test]
fn a_change_the_level_cannot_see_is_rejected_below_it() {
    let mut rng = SplitMix64::new(0x58_0103);
    let session = Session::random(&mut rng);
    for height in [3, 1] {
        let mut forest = TestForest::random(&mut rng, 0, height, false);
        forest.leaves[0] = Fp::ZERO;
        let forests = [forest];
        let (proven, _) = prove(&session, &forests);
        let proofs = tv_proofs(&proven);
        let roots = true_roots(&forests);
        let first = proven.levels()[0].proof();
        assert_eq!(first.final_evals[0], Fp2::ZERO, "u = 0 at the root level");
        let mut changed = first.clone();
        changed.final_evals[1] += delta(&mut rng);
        let mut all = proofs.clone();
        all[0] = tv_proof(&changed);
        let verdict = verify_forests(&session, &shapes(&forests), &roots, &all, None);
        if height > 1 {
            assert_eq!(verdict.map(|_| ()), final_mismatch(1, 1), "height {height}");
        } else {
            let verified = verdict.expect("the last level cannot see the change");
            let claim = &verified.leaf_claims[0];
            assert_ne!(
                claim.value,
                mle(&forests[0].leaves, &claim.point),
                "the leaf claim is false"
            );
        }
        // Changing u instead is seen at once: w is not zero.
        let mut changed = first.clone();
        changed.final_evals[0] += delta(&mut rng);
        let mut all = proofs;
        all[0] = tv_proof(&changed);
        assert_eq!(
            verify_forests(&session, &shapes(&forests), &roots, &all, None).map(|_| ()),
            final_mismatch(0, 0),
            "height {height}"
        );
    }
}

/// The transcript of a prover that uses another challenge than the transcript's under one label:
/// the squeeze is the honest one, so the transcript stays in step, and the value it hands the
/// prover is the challenge plus one.
struct Tampered {
    inner: Transcript,
    label: String,
}

impl SumcheckTranscript<Fp2> for Tampered {
    fn absorb_message(&mut self, label: &str, message: &[Fp2]) {
        self.inner.absorb_message(label, message);
    }

    fn challenge(&mut self, label: &str) -> Fp2 {
        let challenge = self.inner.challenge_fp2(label);
        if label == self.label {
            challenge + Fp2::ONE
        } else {
            challenge
        }
    }
}

#[test]
fn a_changed_root_challenge_is_rejected() {
    let mut rng = SplitMix64::new(0x58_0104);
    let session = Session::random(&mut rng);
    // Heights 4, 2 and 4 with 2^3, 2^5 and 2^0 trees: three root challenges at level 0, five at
    // level 2. Random roots, then zero roots, whose root claims are zero at every point.
    for zero_rows in [false, true] {
        let forests = [
            TestForest::random(&mut rng, 3, 4, zero_rows),
            TestForest::random(&mut rng, 5, 2, zero_rows),
            TestForest::random(&mut rng, 0, 4, zero_rows),
        ];
        let (proven, _) = prove(&session, &forests);
        let proofs = tv_proofs(&proven);
        let roots = true_roots(&forests);
        let mut changes = 0;
        for (level, proven_level) in proven.levels().iter().enumerate() {
            let rounds = proven_level.proof().rounds.len();
            for c in 0..proven_level.root_challenges().len() {
                // The verifier forms its root claims at the changed point.
                assert_eq!(
                    verify_forests(
                        &session,
                        &shapes(&forests),
                        &roots,
                        &proofs,
                        Some((level, c))
                    )
                    .map(|_| ()),
                    final_mismatch(level, rounds),
                    "level {level}, root challenge {c}"
                );
                // A prover that proves at the changed point.
                let mut tampered = Tampered {
                    inner: session.transcript(),
                    label: format!("tree/{level}/root/{c}"),
                };
                let borrowed: Vec<Forest<'_, Fp2>> =
                    forests.iter().map(TestForest::forest).collect();
                let cheat = prove_forests(&mut tampered, &borrowed).expect("well-formed forests");
                assert_eq!(
                    verify_forests(
                        &session,
                        &shapes(&forests),
                        &roots,
                        &tv_proofs(&cheat),
                        None
                    )
                    .map(|_| ()),
                    final_mismatch(level, rounds),
                    "a prover at the changed point, level {level}, root challenge {c}"
                );
                assert_eq!(
                    tv_fp2(cheat.levels()[level].root_challenges()[c]),
                    tv_fp2(proven.levels()[level].root_challenges()[c] + Fp2::ONE)
                );
                changes += 1;
            }
        }
        assert_eq!(changes, 3 + 5);
    }
}
