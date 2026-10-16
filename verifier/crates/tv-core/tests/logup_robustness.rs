// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0019 'Robustness' for the LogUp checks: every count and length the proof gives, and every
//! instance the caller says is present, is checked before anything is sized by it, and each
//! defect gets the typed error the crate's order of checks predicts, never a panic. The subjects
//! are the vectors' honest LogUp batches and seeded batches of the test kit's reference prover.
//!
//! - The roots message, at every length but `4m` and at a million values: `RootsLength` at the
//!   first instance whose four values it lacks, or at the last when it holds more, after `α` and
//!   `β` and before it is absorbed.
//! - The values message, at every length but `k` and at a million values: `ValuesLength` at the
//!   first instance with values whose value it lacks, or at the last instance, before it is
//!   absorbed.
//! - Each level's evaluations and rounds, too few, too many, too long or too short: the errors of
//!   RFC-0010 'Sumcheck messages', at their round.
//! - Instances present or absent: the caller's count of instances off by one, a table's values
//!   said absent or present, and at each level an instance dropped, repeated or of the other
//!   kind. An oracle written from the crate documentation predicts the first structural defect;
//!   without one, the level's final check fails.
//! - Huge declared sizes: a count of instances no message can hold, a claim on a million
//!   variables. Allocation itself is measured in `verifier/fuzz/tests/logup_allocation.rs`.

mod support;

use support::logup::{
    LogUpOutcome, LogUpProof, Subject, logup_batches, run_replacing, seeded_subjects,
};
use tv_core::logup::{LookupError, check_roots};
use tv_core::sumcheck::{SumcheckError, SumcheckProof};
use tv_core::tree::{FractionClaim, LevelClaim, TreeClaim};
use tv_field::Fp2;
use tv_hash::transcript::Ascii;

/// A million values: far more than any honest message.
const HOSTILE: usize = 1 << 20;

/// The vectors' honest batches and seeded ones.
fn subjects() -> Vec<Subject> {
    let batches = logup_batches().expect("`logup_batches` in spec/vectors/sumcheck/sumcheck.json");
    batches
        .iter()
        .filter(|batch| batch.expected == support::logup::Expected::Accept)
        .map(Subject::of_batch)
        .chain(seeded_subjects(0x5eed_0062_0200, 24))
        .collect()
}

/// The seeded subjects alone.
fn seeded() -> Vec<Subject> {
    seeded_subjects(0x5eed_0062_0200, 24)
}

/// `values` taken cyclically from `from`, `count` of them, or zeros when `from` is empty.
fn cycled(from: &[Fp2], count: usize) -> Vec<Fp2> {
    if from.is_empty() {
        return vec![Fp2::ZERO; count];
    }
    from.iter().copied().cycle().take(count).collect()
}

/// The roots message at every other length.
fn roots_lengths(subjects: &[Subject]) {
    for subject in subjects {
        let m = subject.instances.len();
        let honest = &subject.proof.roots;
        assert_eq!(honest.len(), 4 * m);
        for found in (0..=4 * m + 5).chain([HOSTILE]).filter(|&n| n != 4 * m) {
            let roots = cycled(honest, found);
            let expected = match m {
                // With no instance, the message must be empty, and names none.
                0 => LookupError::RootsWithoutInstance { found },
                _ => LookupError::RootsLength {
                    instance: if found < 4 * m { found / 4 } else { m - 1 },
                    expected: 4 * m,
                    found,
                },
            };
            // Directly: the transcript has squeezed α and β, and absorbed nothing.
            let mut transcript = subject.context.transcript();
            assert_eq!(
                check_roots(&mut transcript, m, &roots),
                Err(expected),
                "{}: {found} roots",
                subject.name
            );
            let mut after_challenges = subject.context.transcript();
            for label in ["logup/alpha", "logup/beta"] {
                after_challenges
                    .challenge_fp2(Ascii::new(label).unwrap())
                    .unwrap();
            }
            assert_eq!(transcript.state(), after_challenges.state());
            assert!(expected.is_rejection());
            // Through the driver.
            let proof = LogUpProof {
                roots,
                ..subject.proof.clone()
            };
            assert_eq!(subject.run(&proof).outcome, LogUpOutcome::Roots(expected));
        }
    }
}

#[test]
fn a_roots_message_of_any_other_length_is_rejected_at_the_predicted_instance() {
    roots_lengths(&subjects());
}

#[test]
fn a_roots_message_of_any_other_length_is_rejected_at_the_predicted_instance_seeded() {
    roots_lengths(&seeded());
}

/// The values message at every other length.
fn values_lengths(subjects: &[Subject]) {
    for subject in subjects {
        let with_values: Vec<usize> = subject
            .instances
            .iter()
            .enumerate()
            .filter(|(_, instance)| instance.has_values)
            .map(|(i, _)| i)
            .collect();
        let k = with_values.len();
        let m = subject.instances.len();
        let honest = subject.run(&subject.proof);
        for found in (0..=k + 3).chain([HOSTILE]).filter(|&n| n != k) {
            let expected = match m {
                0 => LookupError::ValuesWithoutInstance { found },
                _ => LookupError::ValuesLength {
                    instance: with_values.get(found).copied().unwrap_or(m - 1),
                    expected: k,
                    found,
                },
            };
            let proof = LogUpProof {
                values: cycled(&subject.proof.values, found),
                ..subject.proof.clone()
            };
            let run = subject.run(&proof);
            assert_eq!(
                run.outcome,
                LogUpOutcome::Leaves(expected),
                "{}: {found} values",
                subject.name
            );
            // Nothing was absorbed: the transcript is where the last level left it.
            let before = run
                .levels
                .last()
                .map_or_else(|| honest_roots_state(subject), |level| level.state);
            assert_eq!(run.state, Some(before), "{}", subject.name);
            assert_eq!(honest.outcome, LogUpOutcome::Accepted);
        }
    }
}

/// The transcript after a subject's honest roots.
fn honest_roots_state(subject: &Subject) -> [u8; 32] {
    let mut transcript = subject.context.transcript();
    let roots = check_roots(
        &mut transcript,
        subject.instances.len(),
        &subject.proof.roots,
    );
    assert!(roots.is_ok());
    *transcript.state()
}

#[test]
fn a_values_message_of_any_other_length_is_rejected_at_the_predicted_instance() {
    values_lengths(&subjects());
}

#[test]
fn a_values_message_of_any_other_length_is_rejected_at_the_predicted_instance_seeded() {
    values_lengths(&seeded());
}

/// The first structural defect of a level's proof for `claims`, in the crate's order of checks
/// ('Tree levels' and 'Rounds'), or `None` when its shape is right.
fn structural(claims: &[LevelClaim], proof: &SumcheckProof) -> Option<SumcheckError> {
    let sizes: Vec<(usize, usize)> = claims
        .iter()
        .map(|claim| match claim {
            LevelClaim::Product(claim) => (claim.point.len(), 2),
            LevelClaim::Fraction(claim) => (claim.point.len(), 4),
        })
        .collect();
    let n_max = sizes.iter().map(|&(vars, _)| vars).max()?;
    let expected: usize = sizes.iter().map(|&(_, evals)| evals).sum();
    for j in 0..n_max {
        let round = j as u32;
        match proof.rounds.get(j) {
            None => return Some(SumcheckError::MissingRound { round }),
            Some(message) if message.len() != 3 => {
                return Some(SumcheckError::RoundLength {
                    round,
                    expected: 3,
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

/// Each level's messages with a wrong count or length.
fn level_shapes(subjects: &[Subject]) {
    let mut cases = 0;
    for subject in subjects {
        let honest = subject.run(&subject.proof);
        for (level, ran) in honest.levels.iter().enumerate() {
            let claims: Vec<LevelClaim> = ran
                .entering
                .iter()
                .map(|(_, claim)| claim.clone())
                .collect();
            let original = &subject.proof.levels[level];
            let n_max = original.rounds.len();
            let total = original.final_evals.len();
            let mut changes: Vec<SumcheckProof> = Vec::new();
            for count in [0, total - 1, total + 1, HOSTILE] {
                let mut changed = original.clone();
                changed.final_evals = cycled(&original.final_evals, count);
                changes.push(changed);
            }
            if n_max > 0 {
                let mut missing = original.clone();
                missing.rounds.pop();
                changes.push(missing);
                for j in [0, n_max - 1] {
                    for length in [0, 2, 4, HOSTILE] {
                        let mut changed = original.clone();
                        changed.rounds[j] = cycled(&original.rounds[j], length);
                        changes.push(changed);
                    }
                }
            }
            for extra in [vec![Fp2::ONE; 3], vec![], vec![Fp2::ZERO; HOSTILE]] {
                let mut changed = original.clone();
                changed.rounds.push(extra);
                changes.push(changed);
            }
            for changed in changes {
                let error = structural(&claims, &changed).expect("a structural defect");
                let mut proof = subject.proof.clone();
                proof.levels[level] = changed;
                assert_eq!(
                    subject.run(&proof).outcome,
                    LogUpOutcome::Level {
                        level: level as u32,
                        error
                    },
                    "{}, level {level}",
                    subject.name
                );
                assert!(error.rejected_round().is_some());
                cases += 1;
            }
        }
    }
    assert!(cases > 100, "{cases}");
}

#[test]
fn wrong_counts_and_lengths_of_a_levels_messages_get_their_errors() {
    level_shapes(&subjects());
}

#[test]
fn wrong_counts_and_lengths_of_a_levels_messages_get_their_errors_seeded() {
    level_shapes(&seeded());
}

/// The claims of another kind: a fraction tree's as a forest's, its numerator as the value, and
/// a forest's as a fraction tree's, its value as the numerator over a denominator of 1.
fn other_kind(claim: &LevelClaim) -> LevelClaim {
    match claim {
        LevelClaim::Fraction(claim) => LevelClaim::Product(TreeClaim {
            point: claim.point.clone(),
            value: claim.numerator,
        }),
        LevelClaim::Product(claim) => LevelClaim::Fraction(FractionClaim {
            point: claim.point.clone(),
            numerator: claim.value,
            denominator: Fp2::ONE,
        }),
    }
}

/// Whether two claims are of one kind, so that a level lists as many evaluations for either.
fn same_kind(a: &LevelClaim, b: &LevelClaim) -> bool {
    matches!(
        (a, b),
        (LevelClaim::Product(_), LevelClaim::Product(_))
            | (LevelClaim::Fraction(_), LevelClaim::Fraction(_))
    )
}

/// At each level, the caller's instances changed: one dropped, one repeated at the end, one of
/// the other kind, or two of one kind exchanged. The level rejects the honest proof with the
/// first structural defect the oracle predicts, or, when the shape is right, at its final check,
/// whose powers of `γ` follow schedule order; a level left with no instance is a fault of the
/// caller.
fn instances_present_or_absent(subjects: &[Subject]) {
    let mut structural_cases = 0;
    let mut final_cases = 0;
    for subject in subjects {
        let honest = subject.run(&subject.proof);
        for (level, ran) in honest.levels.iter().enumerate() {
            let at = level as u32;
            let claims: Vec<LevelClaim> = ran
                .entering
                .iter()
                .map(|(_, claim)| claim.clone())
                .collect();
            let mut variants: Vec<Vec<LevelClaim>> = Vec::new();
            for i in 0..claims.len() {
                let mut dropped = claims.clone();
                dropped.remove(i);
                variants.push(dropped);
                let mut repeated = claims.clone();
                repeated.push(claims[i].clone());
                variants.push(repeated);
                let mut converted = claims.clone();
                converted[i] = other_kind(&claims[i]);
                variants.push(converted);
                for j in i + 1..claims.len() {
                    if same_kind(&claims[i], &claims[j]) && claims[i] != claims[j] {
                        let mut exchanged = claims.clone();
                        exchanged.swap(i, j);
                        variants.push(exchanged);
                    }
                }
            }
            for variant in variants {
                let expected = if variant.is_empty() {
                    SumcheckError::NoInstances
                } else if let Some(error) = structural(&variant, &subject.proof.levels[level]) {
                    structural_cases += 1;
                    error
                } else {
                    final_cases += 1;
                    SumcheckError::FinalMismatch {
                        round: subject.proof.levels[level].rounds.len() as u32,
                    }
                };
                let run = run_replacing(subject, at, variant);
                assert_eq!(
                    run.outcome,
                    LogUpOutcome::Level {
                        level: at,
                        error: expected
                    },
                    "{}, level {level}",
                    subject.name
                );
            }
        }
    }
    assert!(
        structural_cases > 0 && final_cases > 0,
        "{structural_cases}, {final_cases}"
    );
}

#[test]
fn instances_present_or_absent_at_a_level_get_the_predicted_error() {
    instances_present_or_absent(&subjects());
}

#[test]
fn instances_present_or_absent_at_a_level_get_the_predicted_error_seeded() {
    instances_present_or_absent(&seeded());
}

/// The caller's LogUp instances off by one, and a table's values said absent or present: the
/// roots message, or the values message, is then of another length than the caller's
/// instances need.
#[test]
fn the_callers_instances_off_by_one_are_length_errors() {
    for subject in seeded().iter().chain(&subjects()) {
        let m = subject.instances.len();
        let mut transcript = subject.context.transcript();
        assert_eq!(
            check_roots(&mut transcript, m + 1, &subject.proof.roots),
            Err(LookupError::RootsLength {
                instance: m,
                expected: 4 * (m + 1),
                found: 4 * m
            }),
            "{}",
            subject.name
        );
        if m >= 2 {
            let mut transcript = subject.context.transcript();
            assert_eq!(
                check_roots(&mut transcript, m - 1, &subject.proof.roots),
                Err(LookupError::RootsLength {
                    instance: m - 2,
                    expected: 4 * (m - 1),
                    found: 4 * m
                }),
                "{}",
                subject.name
            );
        }
        for i in 0..m {
            let mut flipped = subject.clone();
            flipped.instances[i].has_values = !flipped.instances[i].has_values;
            let with_values: Vec<usize> = flipped
                .instances
                .iter()
                .enumerate()
                .filter(|(_, instance)| instance.has_values)
                .map(|(j, _)| j)
                .collect();
            let found = subject.proof.values.len();
            let instance = with_values.get(found).copied().unwrap_or(m - 1);
            assert_eq!(
                flipped.run(&flipped.proof).outcome,
                LogUpOutcome::Leaves(LookupError::ValuesLength {
                    instance,
                    expected: with_values.len(),
                    found
                }),
                "{}, instance {i}",
                subject.name
            );
        }
    }
}

/// Huge declared sizes are typed errors: a count of instances no message can hold, its expected
/// length saturated; and a claim on a million variables, which asks for rounds the proof does not
/// have, so the level stops at the first missing one.
#[test]
fn huge_declared_sizes_are_typed_errors() {
    for subject in seeded() {
        let m = subject.instances.len();
        let mut transcript = subject.context.transcript();
        assert_eq!(
            check_roots(&mut transcript, usize::MAX, &subject.proof.roots),
            Err(LookupError::RootsLength {
                instance: m,
                expected: usize::MAX,
                found: 4 * m
            })
        );
        let honest = subject.run(&subject.proof);
        for (level, ran) in honest.levels.iter().enumerate() {
            let at = level as u32;
            let rounds = subject.proof.levels[level].rounds.len() as u32;
            let mut claims: Vec<LevelClaim> = ran
                .entering
                .iter()
                .map(|(_, claim)| claim.clone())
                .collect();
            let huge = vec![Fp2::ONE; HOSTILE];
            claims[0] = match &claims[0] {
                LevelClaim::Product(claim) => LevelClaim::Product(TreeClaim {
                    point: huge,
                    value: claim.value,
                }),
                LevelClaim::Fraction(claim) => LevelClaim::Fraction(FractionClaim {
                    point: huge,
                    ..claim.clone()
                }),
            };
            let run = run_replacing(&subject, at, claims);
            assert_eq!(
                run.outcome,
                LogUpOutcome::Level {
                    level: at,
                    error: SumcheckError::MissingRound { round: rounds }
                },
                "{}, level {level}",
                subject.name
            );
        }
    }
}
