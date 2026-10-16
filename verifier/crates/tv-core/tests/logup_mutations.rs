// SPDX-License-Identifier: Apache-2.0 OR MIT

//! #62's acceptance criterion on single-element changes: every field element of every honest
//! leaf stage's proof, changed by one in turn, is rejected with the error the check order gives
//! (RFC-0019 'Check order' step 7), or, where the check it reaches cannot see the change, leaves a
//! false claim that the check after it rejects. The proofs are the vectors' honest LogUp batches,
//! written by an independent Python prover, and seeded batches of the test kit's reference prover.
//!
//! - A root value changed by one unbalances its instance's roots: with `n_a`, `n_t` and the
//!   denominators not zero, `n_a + 1` adds `d_t` to `n_a·d_t + n_t·d_a`, `d_a + 1` adds `n_t`,
//!   `n_t + 1` adds `d_a` and `d_t + 1` adds `n_a`. `Reject(Lookup)` at the roots.
//! - A round value changed moves the transcript and the claim from that round on, so the level
//!   fails its final check, `FinalMismatch { round: n_max }`.
//! - A final evaluation changed by `δ` adds `γ^i·eq(r_i, r'_i)·δ` times its cofactor to the final
//!   check: `w_i` for `u_i`, `u_i` for `w_i`, `D_1` for `N_0`, `D_0` for `N_1`, `N_1 + λ·D_1` for
//!   `D_0` and `N_0 + λ·D_0` for `D_1`. The oracle computes the check itself: when the cofactor is
//!   zero, as `u_i` is where a forest's tree has a zero leaf, the level passes, the transcript
//!   moves, and the next level rejects the change, or after the last level a claim left for later
//!   is false.
//! - A value `v` changed passes the leaf checks, which do not read it, and leaves the key and
//!   value claims of its instance false, which the stages and openings that take them reject.

mod support;

use support::logup::{
    FalseClaim, LogUpOutcome, LogUpProof, LogUpRun, Subject, logup_batches, seeded_subjects,
};
use tv_core::logup::LookupError;
use tv_core::sumcheck::SumcheckError;
use tv_core::tree::LevelClaim;
use tv_field::Fp2;

/// `eq(a, b)` from its definition.
fn eq(a: &[Fp2], b: &[Fp2]) -> Fp2 {
    a.iter().zip(b).fold(Fp2::ONE, |product, (&x, &y)| {
        product * (x * y + (Fp2::ONE - x) * (Fp2::ONE - y))
    })
}

/// `Σ_i γ^i·eq(r_i, r'_i)·g_i` over a level's claims and `evals`, the final check's left side,
/// computed here from RFC-0010 'Tree levels'.
fn combination(
    claims: &[LevelClaim],
    gamma: Fp2,
    lambda: Fp2,
    point: &[Fp2],
    evals: &[Fp2],
) -> Fp2 {
    let mut rest = evals.iter().copied();
    let mut weight = Fp2::ONE;
    let mut sum = Fp2::ZERO;
    for claim in claims {
        let (own, g) = match claim {
            LevelClaim::Product(claim) => {
                let (u, w) = (rest.next().unwrap(), rest.next().unwrap());
                (&claim.point, u * w)
            }
            LevelClaim::Fraction(claim) => {
                let [n0, n1, d0, d1] = [(); 4].map(|_| rest.next().unwrap());
                (&claim.point, n0 * d1 + n1 * d0 + lambda * d0 * d1)
            }
        };
        let r = &point[point.len() - own.len()..];
        sum += weight * eq(own, r) * g;
        weight *= gamma;
    }
    sum
}

/// What a subject's changes came to.
#[derive(Default, Debug)]
struct Tally {
    changed: usize,
    /// Final evaluations whose change passed their level.
    passed_level: usize,
    /// Values whose change passed the leaf checks.
    passed_leaves: usize,
}

/// Changes every element of `subject`'s proof by one, in turn, and checks each result.
fn change_every_element(subject: &Subject, tally: &mut Tally) {
    let name = &subject.name;
    let honest: LogUpRun = subject.run(&subject.proof);
    assert_eq!(honest.outcome, LogUpOutcome::Accepted, "{name}");
    let last = subject.proof.levels.len().checked_sub(1);
    let rejected_at = |level: usize| LogUpOutcome::Level {
        level: level as u32,
        error: SumcheckError::FinalMismatch {
            round: subject.proof.levels[level].rounds.len() as u32,
        },
    };
    let mut check = |proof: LogUpProof, expected: &dyn Fn(&LogUpOutcome) -> bool, what: String| {
        let run = subject.run(&proof);
        assert!(expected(&run.outcome), "{name}, {what}: {:?}", run.outcome);
        tally.changed += 1;
    };

    // The roots.
    for (k, &value) in subject.proof.roots.iter().enumerate() {
        let instance = k / 4;
        let block = &subject.proof.roots[4 * instance..4 * instance + 4];
        // The cofactor of the changed value in n_a·d_t + n_t·d_a, and no denominator made zero.
        let cofactor = [block[3], block[2], block[1], block[0]][k % 4];
        assert_ne!(cofactor, Fp2::ZERO, "{name}: root {k}'s cofactor");
        if k % 2 == 1 {
            assert_ne!(value + Fp2::ONE, Fp2::ZERO, "{name}: root {k}");
        }
        let mut proof = subject.proof.clone();
        proof.roots[k] += Fp2::ONE;
        let expected = LogUpOutcome::Roots(LookupError::Unbalanced { instance });
        check(proof, &|outcome| *outcome == expected, format!("root {k}"));
    }

    // The levels.
    for (level, (proof_level, ran)) in subject.proof.levels.iter().zip(&honest.levels).enumerate() {
        for j in 0..proof_level.rounds.len() {
            for k in 0..proof_level.rounds[j].len() {
                let mut proof = subject.proof.clone();
                proof.levels[level].rounds[j][k] += Fp2::ONE;
                let expected = rejected_at(level);
                check(
                    proof,
                    &|outcome| *outcome == expected,
                    format!("level {level}, round {j}, value {k}"),
                );
            }
        }
        let claims: Vec<LevelClaim> = ran
            .entering
            .iter()
            .map(|(_, claim)| claim.clone())
            .collect();
        let lambda = ran.outcome.lambda.unwrap_or(Fp2::ZERO);
        let honest_value = combination(
            &claims,
            ran.outcome.gamma,
            lambda,
            &ran.outcome.point,
            &proof_level.final_evals,
        );
        for k in 0..proof_level.final_evals.len() {
            let mut proof = subject.proof.clone();
            proof.levels[level].final_evals[k] += Fp2::ONE;
            let value = combination(
                &claims,
                ran.outcome.gamma,
                lambda,
                &ran.outcome.point,
                &proof.levels[level].final_evals,
            );
            let what = format!("level {level}, evaluation {k}");
            if value != honest_value {
                let expected = rejected_at(level);
                check(proof, &|outcome| *outcome == expected, what);
            } else if Some(level) == last {
                // The level passes and moves ρ: the claims left for later are false.
                tally.passed_level += 1;
                check(
                    proof,
                    &|outcome| matches!(outcome, LogUpOutcome::False(_)),
                    what,
                );
            } else {
                // The level passes and moves the transcript: the next level rejects it.
                tally.passed_level += 1;
                let expected = rejected_at(level + 1);
                check(proof, &|outcome| *outcome == expected, what);
            }
        }
    }

    // The values: instance by instance, those with values take one each.
    let receivers: Vec<usize> = subject
        .instances
        .iter()
        .enumerate()
        .filter(|(_, instance)| instance.has_values)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(receivers.len(), subject.proof.values.len(), "{name}");
    for (k, &instance) in receivers.iter().enumerate() {
        let mut proof = subject.proof.clone();
        proof.values[k] += Fp2::ONE;
        let run = subject.run(&proof);
        // The key claim α − q − β·v is checked first; the value claim v is false too.
        assert_eq!(
            run.outcome,
            LogUpOutcome::False(FalseClaim::Key(instance)),
            "{name}, value {k}"
        );
        let left = &run.claims.as_ref().unwrap().instances[instance];
        let honest_left = &honest.claims.as_ref().unwrap().instances[instance];
        assert_ne!(left.value, honest_left.value, "{name}, value {k}");
        assert_ne!(run.state, honest.state, "{name}, value {k}: the transcript");
        tally.passed_leaves += 1;
        tally.changed += 1;
    }
}

/// The size of a proof, element by element: every value the changes reach.
fn elements(proof: &LogUpProof) -> usize {
    proof.roots.len()
        + proof
            .levels
            .iter()
            .map(|level| level.rounds.iter().map(Vec::len).sum::<usize>() + level.final_evals.len())
            .sum::<usize>()
        + proof.values.len()
}

#[test]
fn every_single_element_change_of_every_honest_vector_batch_is_rejected() {
    let batches = logup_batches().expect("`logup_batches` in spec/vectors/sumcheck/sumcheck.json");
    let mut tally = Tally::default();
    let mut total = 0;
    for batch in batches
        .iter()
        .filter(|batch| batch.expected == support::logup::Expected::Accept)
    {
        let subject = Subject::of_batch(batch);
        total += elements(&subject.proof);
        change_every_element(&subject, &mut tally);
    }
    assert_eq!(tally.changed, total);
    // The 10 honest batches: 38, 5, 49, 88, 98, 200, 22, 24, 19 and 5 elements. No evaluation of
    // theirs has a zero cofactor, and each of their 12 values passes the leaf checks.
    assert_eq!(tally.changed, 548);
    assert_eq!(tally.passed_level, 0);
    assert_eq!(tally.passed_leaves, 12);
}

#[test]
fn every_single_element_change_of_seeded_batches_is_rejected() {
    let mut tally = Tally::default();
    let mut total = 0;
    for subject in seeded_subjects(0x5eed_0062_0100, 48) {
        total += elements(&subject.proof);
        change_every_element(&subject, &mut tally);
    }
    assert_eq!(tally.changed, total);
    assert!(tally.changed > 2_000, "{tally:?}");
    // SoftmaxRow forests' w where they join, and every value.
    assert!(tally.passed_level > 0, "{tally:?}");
    assert!(tally.passed_leaves > 20, "{tally:?}");
}
