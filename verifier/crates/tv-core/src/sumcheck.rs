// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The rounds of a sumcheck and batched sumchecks with front-padding (RFC-0010 'Batching within
//! a stage', 'Sumcheck messages'). The crate documentation is their specification: what is
//! checked, in which order, and what each error means.
//!
//! ```
//! use tv_core::sumcheck::{Instance, Prefix, SumcheckError, SumcheckProof, verify_batch};
//! use tv_field::{Fp, Fp2};
//! use tv_hash::transcript::{Ascii, Mode, ParamSetId, SessionContext, Transcript};
//!
//! let context = SessionContext {
//!     protocol: Ascii::new("tessera-protocol/1")?,
//!     mode: Mode::NonInteractive,
//!     params: ParamSetId(Ascii::new("test-small/1")?),
//!     statement_digest: [0; 32],
//!     session_id: [0; 16],
//! };
//! // A batch of one instance on no variable: a product of two constants, 6 · 7 = 42. It has no
//! // round, and its claim is checked against the relation at once.
//! let instance = Instance { claim: Fp2::from(Fp::reduce(42)), vars: 0, evals: 2 };
//! let proof = SumcheckProof {
//!     rounds: vec![],
//!     final_evals: vec![Fp2::from(Fp::reduce(6)), Fp2::from(Fp::reduce(7))],
//! };
//! let product = |_: usize, _: &[Fp2], evals: &[Fp2]| Some(evals.iter().fold(Fp2::ONE, |p, e| p * *e));
//!
//! let mut transcript = Transcript::new(&context);
//! let batch = verify_batch(&mut transcript, Prefix::Stage(0), 2, &[instance], &proof, product)?;
//! assert!(batch.point.is_empty());
//!
//! // 41 is not 6 · 7.
//! let false_claim = Instance { claim: Fp2::from(Fp::reduce(41)), ..instance };
//! let mut transcript = Transcript::new(&context);
//! assert_eq!(
//!     verify_batch(&mut transcript, Prefix::Stage(0), 2, &[false_claim], &proof, product),
//!     Err(SumcheckError::FinalMismatch { round: 0 })
//! );
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use core::fmt;

use tv_field::{Fp, Fp2};
use tv_hash::transcript::{NotAscii, Transcript, TranscriptError};

use crate::fiat_shamir::{self, FiatShamir, Step};
use crate::interpolation;

/// The largest degree of a round polynomial: 7 (RFC-0014 'Soundness budget', `d ≤ 7`). The
/// smallest is 1, whose message is `p(0)` alone.
pub const MAX_DEGREE: usize = 7;

/// Which sumcheck of the protocol a check runs: the prefix of its labels (RFC-0014 'Transcript
/// schedule').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Prefix {
    /// Stage `k`, labels `stage/{k}/…`.
    Stage(u32),
    /// Level `ℓ` of the leaf stage's trees, labels `tree/{ℓ}/…`.
    Tree(u32),
}

impl fmt::Display for Prefix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Prefix::Stage(k) => write!(f, "stage/{k}"),
            Prefix::Tree(level) => write!(f, "tree/{level}"),
        }
    }
}

/// A sumcheck's messages (RFC-0010 'Sumcheck messages').
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct SumcheckProof {
    /// Round `j`'s message, `j` from 0: the round polynomial's values `p(0), p(2), p(3), …, p(d)`.
    pub rounds: Vec<Vec<Fp2>>,
    /// The final evaluations, instance by instance in schedule order, each instance's in the
    /// order its relation lists its inputs.
    pub final_evals: Vec<Fp2>,
}

/// One instance of a batched sumcheck, as the schedule gives it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Instance {
    /// `c_i`: the claimed sum of the instance's relation over its hypercube.
    pub claim: Fp2,
    /// `n_i`: its number of variables.
    pub vars: usize,
    /// How many final evaluations its relation lists.
    pub evals: usize,
}

/// What the rounds of a sumcheck leave: the point and the claim there.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[must_use = "the rounds verify nothing until the last claim is checked against the relation at the point"]
pub struct Reduction {
    /// The challenges, in round order: the point, its low variables first.
    pub point: Vec<Fp2>,
    /// The last claim, the value the relation must take at `point`.
    pub claim: Fp2,
}

/// A batch that passed every check of [`verify_batch`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[must_use = "the instances' final evaluations hold at the batch's point, where the next claims continue"]
pub struct Batch {
    /// The batching challenge `γ`.
    pub gamma: Fp2,
    /// The challenges, one per round in round order: the stage's point, its low variables first.
    pub point: Vec<Fp2>,
}

impl Batch {
    /// The point of an instance of `vars` variables: the last `vars` challenges (front-padding),
    /// or `None` when the batch has fewer than `vars` rounds.
    pub fn point_of(&self, vars: usize) -> Option<&[Fp2]> {
        suffix(&self.point, vars)
    }
}

/// Why a sumcheck was rejected or could not be checked: the first check that failed, in the
/// order of the crate documentation. Rounds are counted from 0, as their labels count them.
///
/// `MissingRound`, `RoundLength`, `ExtraRound`, `EvaluationCount` and `FinalMismatch` judge the
/// proof: `Reject(Sumcheck { stage, round })` in 04 §2 for a reduction stage, and
/// `Reject(TreeLevel { group, level, round })` for a tree level (RFC-0033, RFC-0054), with `round` from
/// [`rejected_round`](Self::rejected_round). The others judge what the verifier passes in, the
/// schedule's degree, instances and relations, or the verifier itself: from a correct verifier
/// they never come, and they mean a fault of the verifier (`Inconclusive(ERR-INTERNAL)`), not of
/// the proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SumcheckError {
    /// The proof stops before round `round`: it has `round` rounds, and the sumcheck more.
    MissingRound {
        /// The first round missing.
        round: u32,
    },
    /// Round `round`'s message holds `found` evaluations where the degree requires `expected`.
    RoundLength {
        /// The round.
        round: u32,
        /// The degree `d`: the message is `p(0), p(2), …, p(d)`.
        expected: usize,
        /// The evaluations the message holds.
        found: usize,
    },
    /// The proof has a round past the last: `round`, the sumcheck's number of rounds, is the
    /// index of the first one too many.
    ExtraRound {
        /// The first round too many.
        round: u32,
    },
    /// The proof holds `found` final evaluations where the instances' relations list
    /// `expected`.
    EvaluationCount {
        /// The number of rounds: the evaluations follow the last.
        round: u32,
        /// The evaluations the relations list.
        expected: usize,
        /// The evaluations the proof holds.
        found: usize,
    },
    /// The relations' batched value at the final point is not the last claim.
    FinalMismatch {
        /// The number of rounds: the check follows the last.
        round: u32,
    },
    /// The degree is not between 1 and [`MAX_DEGREE`].
    Degree {
        /// The degree asked for.
        degree: usize,
    },
    /// The batch has no instance.
    NoInstances,
    /// The instances' evaluation counts sum past `usize::MAX`.
    EvaluationTotal,
    /// The caller's relation could not give instance `instance`'s value at its point.
    Relation {
        /// The instance, in schedule order from 0.
        instance: usize,
    },
    /// A label built here is not printable ASCII.
    Label(NotAscii),
    /// A challenge found no value in the squeezes RFC-0004 allows ('Failure modes').
    Transcript(TranscriptError),
}

impl SumcheckError {
    /// The round of `Reject(Sumcheck { stage, round })` or `Reject(TreeLevel { group, level, round })`
    /// (04 §2) for a fault of the proof, or
    /// `None` for a fault of the verifier, which is `Inconclusive(ERR-INTERNAL)`.
    pub fn rejected_round(&self) -> Option<u32> {
        match *self {
            SumcheckError::MissingRound { round }
            | SumcheckError::RoundLength { round, .. }
            | SumcheckError::ExtraRound { round }
            | SumcheckError::EvaluationCount { round, .. }
            | SumcheckError::FinalMismatch { round } => Some(round),
            SumcheckError::Degree { .. }
            | SumcheckError::NoInstances
            | SumcheckError::EvaluationTotal
            | SumcheckError::Relation { .. }
            | SumcheckError::Label(_)
            | SumcheckError::Transcript(_) => None,
        }
    }
}

impl fmt::Display for SumcheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SumcheckError::MissingRound { round } => {
                write!(f, "the proof stops before round {round}")
            }
            SumcheckError::RoundLength {
                round,
                expected,
                found,
            } => write!(
                f,
                "round {round} holds {found} evaluations where the degree requires {expected}"
            ),
            SumcheckError::ExtraRound { round } => {
                write!(f, "the proof has a round {round}, past the last")
            }
            SumcheckError::EvaluationCount {
                expected, found, ..
            } => write!(
                f,
                "the proof holds {found} final evaluations where the relations list {expected}"
            ),
            SumcheckError::FinalMismatch { .. } => {
                f.write_str("the final evaluations do not give the last claim")
            }
            SumcheckError::Degree { degree } => write!(
                f,
                "degree {degree} asked for; round polynomials have degree 1 to {MAX_DEGREE}"
            ),
            SumcheckError::NoInstances => f.write_str("the batch has no instance"),
            SumcheckError::EvaluationTotal => {
                f.write_str("the instances' evaluation counts overflow")
            }
            SumcheckError::Relation { instance } => {
                write!(
                    f,
                    "the relation of instance {instance} could not be evaluated"
                )
            }
            SumcheckError::Label(error) => write!(f, "a sumcheck label is not ASCII: {error}"),
            SumcheckError::Transcript(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for SumcheckError {}

/// Checks the rounds of one sumcheck of degree `degree` over `vars` variables that starts from
/// `claim`, under `prefix`, and returns the point and the last claim. The checks, in order, are
/// those of the crate documentation ('Rounds'): the degree is between 1 and [`MAX_DEGREE`];
/// round by round, the round is present and holds `degree` evaluations; no round follows the
/// last.
///
/// `Ok` says only that the rounds were well formed. The caller must check the last claim against
/// the relation at the point, as [`verify_batch`] does: until then nothing about the proof is
/// verified.
pub fn reduce(
    transcript: &mut Transcript,
    prefix: Prefix,
    degree: usize,
    vars: usize,
    claim: Fp2,
    rounds: &[Vec<Fp2>],
) -> Result<Reduction, SumcheckError> {
    check_degree(degree)?;
    reduce_with(transcript, prefix, degree, vars, claim, rounds)
}

/// Checks a batched sumcheck of degree `degree` under `prefix` and returns its batching
/// challenge and point; the checks, in order, are those of the crate documentation ('Batches').
///
/// `instances` come in schedule order. `relation(i, point, evaluations)` gives instance `i`'s
/// relation at its point, the last `n_i` challenges, from its evaluations, the next `k_i` of
/// `proof.final_evals`: public factors such as `eq` included, without `γ^i`, which this function
/// applies, and without `2^{n_max − n_i}`, which belongs to the claim only. It returns `None`
/// when it cannot evaluate, which is a fault of the verifier.
///
/// The batching challenge is squeezed first: the instances' claims must already be fixed by the
/// transcript, as RFC-0010 requires, since they are not absorbed here.
pub fn verify_batch<F>(
    transcript: &mut Transcript,
    prefix: Prefix,
    degree: usize,
    instances: &[Instance],
    proof: &SumcheckProof,
    relation: F,
) -> Result<Batch, SumcheckError>
where
    F: FnMut(usize, &[Fp2], &[Fp2]) -> Option<Fp2>,
{
    verify_batch_with(transcript, prefix, degree, instances, proof, relation)
}

/// `eq(a, b) = Π_i (a_i·b_i + (1 − a_i)·(1 − b_i))` (RFC-0010 'Multilinear extensions'), or
/// `None` when the points have different lengths. Each factor is computed as
/// `1 − a_i − b_i + 2·a_i·b_i`; the empty product is 1.
pub fn eq(a: &[Fp2], b: &[Fp2]) -> Option<Fp2> {
    if a.len() != b.len() {
        return None;
    }
    Some(a.iter().zip(b).fold(Fp2::ONE, |product, (&x, &y)| {
        let xy = x * y;
        product * (Fp2::ONE - x - y + xy + xy)
    }))
}

/// [`reduce`] on any transcript, the degree already checked.
pub(crate) fn reduce_with(
    transcript: &mut impl FiatShamir,
    prefix: Prefix,
    degree: usize,
    vars: usize,
    mut claim: Fp2,
    rounds: &[Vec<Fp2>],
) -> Result<Reduction, SumcheckError> {
    let mut point = Vec::new();
    // The loop stops at the first missing round, so it runs at most once more than the proof
    // has rounds, whatever `vars` is.
    for j in 0..vars {
        let round = round_index(j);
        let message = rounds.get(j).ok_or(SumcheckError::MissingRound { round })?;
        // A degree of at least 1 makes a message of the right length non-empty.
        let Some((&p0, rest)) = message.split_first().filter(|_| message.len() == degree) else {
            return Err(SumcheckError::RoundLength {
                round,
                expected: degree,
                found: message.len(),
            });
        };
        fiat_shamir::absorb(transcript, prefix, Step::Round(j), message)?;
        let r = fiat_shamir::challenge(transcript, prefix, Step::Challenge(j))?;
        claim = interpolation::next_claim(p0, rest, claim, r);
        point.push(r);
    }
    if rounds.len() > vars {
        return Err(SumcheckError::ExtraRound {
            round: round_index(vars),
        });
    }
    Ok(Reduction { point, claim })
}

/// [`verify_batch`] on any transcript.
pub(crate) fn verify_batch_with<F>(
    transcript: &mut impl FiatShamir,
    prefix: Prefix,
    degree: usize,
    instances: &[Instance],
    proof: &SumcheckProof,
    relation: F,
) -> Result<Batch, SumcheckError>
where
    F: FnMut(usize, &[Fp2], &[Fp2]) -> Option<Fp2>,
{
    // 1. The caller's inputs, before the transcript moves.
    let shape = Shape::of(
        degree,
        instances
            .iter()
            .map(|instance| (instance.vars, instance.evals)),
    )?;
    // 2. The batching challenge.
    let gamma = fiat_shamir::challenge(transcript, prefix, Step::Gamma)?;
    finish_batch(transcript, prefix, shape, gamma, instances, proof, relation)
}

/// What a batch's instances fix before the transcript moves (step 1 of 'Batches'): the degree,
/// checked, `n_max` and `Σ k_i`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Shape {
    degree: usize,
    n_max: usize,
    evals: usize,
}

impl Shape {
    /// The shape of a batch of degree `degree` whose instances have the `(n_i, k_i)` of `sizes`,
    /// in schedule order, or the first of `Degree`, `NoInstances` and `EvaluationTotal` that
    /// they fail.
    pub(crate) fn of(
        degree: usize,
        sizes: impl IntoIterator<Item = (usize, usize)>,
    ) -> Result<Self, SumcheckError> {
        check_degree(degree)?;
        let mut n_max: Option<usize> = None;
        let mut evals: usize = 0;
        for (vars, count) in sizes {
            n_max = Some(n_max.map_or(vars, |most| most.max(vars)));
            evals = evals
                .checked_add(count)
                .ok_or(SumcheckError::EvaluationTotal)?;
        }
        Ok(Self {
            degree,
            n_max: n_max.ok_or(SumcheckError::NoInstances)?,
            evals,
        })
    }
}

/// Steps 3 to 8 of 'Batches', after `γ`: `instances` are those whose sizes gave `shape`, in the
/// same order, with their claims.
pub(crate) fn finish_batch<F>(
    transcript: &mut impl FiatShamir,
    prefix: Prefix,
    shape: Shape,
    gamma: Fp2,
    instances: &[Instance],
    proof: &SumcheckProof,
    mut relation: F,
) -> Result<Batch, SumcheckError>
where
    F: FnMut(usize, &[Fp2], &[Fp2]) -> Option<Fp2>,
{
    let Shape {
        degree,
        n_max,
        evals: expected,
    } = shape;
    // 3. Σ_i γ^i · 2^{n_max − n_i} · c_i.
    let mut claim = Fp2::ZERO;
    let mut weight = Fp2::ONE;
    for instance in instances {
        // n_max is the largest n_i, so nothing saturates.
        let padding = n_max.saturating_sub(instance.vars);
        claim += weight * instance.claim * two_to(padding);
        weight *= gamma;
    }

    // 4. The rounds.
    let Reduction { point, claim: last } =
        reduce_with(transcript, prefix, degree, n_max, claim, &proof.rounds)?;

    // 5. and 6. The final evaluations: their number, then the transcript.
    let round = round_index(n_max);
    if proof.final_evals.len() != expected {
        return Err(SumcheckError::EvaluationCount {
            round,
            expected,
            found: proof.final_evals.len(),
        });
    }
    fiat_shamir::absorb(transcript, prefix, Step::Evals, &proof.final_evals)?;

    // 7. and 8. Σ_i γ^i · g_i against the last claim, instance i at the last n_i challenges.
    let mut value = Fp2::ZERO;
    let mut weight = Fp2::ONE;
    let mut rest: &[Fp2] = &proof.final_evals;
    for (index, instance) in instances.iter().enumerate() {
        // The count check above and n_max ≥ n_i make both slices exist.
        let fault = SumcheckError::Relation { instance: index };
        let own = suffix(&point, instance.vars).ok_or(fault)?;
        let (evals, tail) = rest.split_at_checked(instance.evals).ok_or(fault)?;
        rest = tail;
        value += weight * relation(index, own, evals).ok_or(fault)?;
        weight *= gamma;
    }
    if value != last {
        return Err(SumcheckError::FinalMismatch { round });
    }
    Ok(Batch { gamma, point })
}

/// The degree check of every sumcheck: `1 ≤ degree ≤ MAX_DEGREE`.
fn check_degree(degree: usize) -> Result<(), SumcheckError> {
    if (1..=MAX_DEGREE).contains(&degree) {
        Ok(())
    } else {
        Err(SumcheckError::Degree { degree })
    }
}

/// The last `vars` coordinates of `point`.
fn suffix(point: &[Fp2], vars: usize) -> Option<&[Fp2]> {
    point
        .len()
        .checked_sub(vars)
        .and_then(|start| point.get(start..))
}

/// `2^exponent` in `Fp`, one squaring per bit of the exponent.
fn two_to(exponent: usize) -> Fp {
    let mut result = Fp::ONE;
    let mut square = Fp::reduce(2);
    let mut rest = exponent;
    while rest != 0 {
        if rest & 1 == 1 {
            result *= square;
        }
        square *= square;
        rest >>= 1;
    }
    result
}

/// A round index as errors report it. Rounds past `u32::MAX` would take a proof of more than
/// 96 GiB, beyond any decoder's limits, and read as `u32::MAX`.
fn round_index(j: usize) -> u32 {
    u32::try_from(j).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fiat_shamir::stub::{Op, Stub};
    use crate::fiat_shamir::tce;

    fn fp2(c0: u64, c1: u64) -> Fp2 {
        Fp2::new(Fp::reduce(c0), Fp::reduce(c1))
    }

    fn product(_: usize, _: &[Fp2], evals: &[Fp2]) -> Option<Fp2> {
        Some(evals.iter().fold(Fp2::ONE, |p, e| p * *e))
    }

    #[test]
    fn two_to_is_the_power_of_two() {
        let mut expected = Fp::ONE;
        for exponent in 0..200 {
            assert_eq!(two_to(exponent), expected, "2^{exponent}");
            expected = expected + expected;
        }
        // 2 has order 192 in Fp^* (2^96 = −1), so the large exponents are checked through it.
        assert_eq!(two_to(192), Fp::ONE);
        assert_eq!(two_to(usize::MAX), two_to(usize::MAX % 192));
    }

    #[test]
    fn eq_follows_its_definition() {
        let (zero, one) = (Fp2::ZERO, Fp2::ONE);
        assert_eq!(eq(&[], &[]), Some(one));
        assert_eq!(eq(&[one, zero], &[one, zero]), Some(one));
        assert_eq!(eq(&[one, zero], &[zero, zero]), Some(zero));
        let (a, b) = (fp2(3, 5), fp2(11, 2));
        let factor = |x: Fp2, y: Fp2| x * y + (one - x) * (one - y);
        assert_eq!(eq(&[a, b], &[b, a]), Some(factor(a, b) * factor(b, a)));
        assert_eq!(eq(&[a], &[a, b]), None);
    }

    /// The labels and absorbed bytes of a batch, in order (RFC-0014 'Transcript schedule'):
    /// `γ`, then each round's message and challenge, then the evaluations.
    #[test]
    fn a_batch_runs_the_transcript_schedule_in_order() {
        // Instances on 2 and 0 variables, products of 2 and 1 tables: the second is constant 9.
        let instances = [
            Instance {
                claim: fp2(0, 0),
                vars: 2,
                evals: 2,
            },
            Instance {
                claim: fp2(9, 0),
                vars: 0,
                evals: 1,
            },
        ];
        let rounds = vec![vec![fp2(1, 2), fp2(3, 4)], vec![fp2(5, 6), fp2(7, 8)]];
        let evals = vec![fp2(10, 0), fp2(11, 0), fp2(9, 0)];
        let proof = SumcheckProof {
            rounds: rounds.clone(),
            final_evals: evals.clone(),
        };
        let mut stub = Stub::default();
        let outcome =
            verify_batch_with(&mut stub, Prefix::Stage(4), 2, &instances, &proof, product);
        assert_eq!(outcome, Err(SumcheckError::FinalMismatch { round: 2 }));
        assert_eq!(
            stub.script(),
            [
                "?stage/4/gamma",
                "+stage/4/round/0",
                "?stage/4/r/0",
                "+stage/4/round/1",
                "?stage/4/r/1",
                "+stage/4/evals"
            ]
        );
        assert_eq!(
            stub.ops[1],
            Op::Absorb("stage/4/round/0".into(), tce(&rounds[0]))
        );
        assert_eq!(
            stub.ops[3],
            Op::Absorb("stage/4/round/1".into(), tce(&rounds[1]))
        );
        assert_eq!(stub.ops[5], Op::Absorb("stage/4/evals".into(), tce(&evals)));
    }

    /// A wrong number of final evaluations is rejected before they are encoded or absorbed, so
    /// a hostile count costs nothing: the transcript ends at the last round's challenge.
    #[test]
    fn a_wrong_evaluation_count_is_rejected_before_the_evaluations_are_absorbed() {
        let instance = Instance {
            claim: fp2(3, 0),
            vars: 2,
            evals: 2,
        };
        for found in [0, 1, 3, 1000] {
            let proof = SumcheckProof {
                rounds: vec![vec![fp2(1, 0), fp2(2, 0)]; 2],
                final_evals: vec![fp2(4, 0); found],
            };
            let mut stub = Stub::default();
            let outcome =
                verify_batch_with(&mut stub, Prefix::Stage(2), 2, &[instance], &proof, product);
            assert_eq!(
                outcome,
                Err(SumcheckError::EvaluationCount {
                    round: 2,
                    expected: 2,
                    found
                })
            );
            assert_eq!(
                stub.script(),
                [
                    "?stage/2/gamma",
                    "+stage/2/round/0",
                    "?stage/2/r/0",
                    "+stage/2/round/1",
                    "?stage/2/r/1"
                ]
            );
        }
    }

    #[test]
    fn tce_is_the_length_then_c0_and_c1_of_each_element() {
        let bytes = tce(&[fp2(1, 2), fp2(Fp::MODULUS - 1, 0)]);
        let mut expected = 2u64.to_le_bytes().to_vec();
        for word in [1, 2, Fp::MODULUS - 1, 0u64] {
            expected.extend_from_slice(&word.to_le_bytes());
        }
        assert_eq!(bytes, expected);
        assert_eq!(tce(&[]), 0u64.to_le_bytes());
    }

    /// RFC-0004 'Failure modes': a challenge that exhausts its squeezes stops the check with
    /// the transcript's error, a fault of the verifier, at each challenge a batch draws: `γ`,
    /// each round's, and nothing after it is done. BLAKE3 reaches this with probability below
    /// 2^-496, so a stub forces it.
    #[test]
    fn an_exhausted_challenge_stops_the_batch() {
        let instance = Instance {
            claim: fp2(5, 0),
            vars: 3,
            evals: 1,
        };
        let proof = SumcheckProof {
            rounds: vec![vec![fp2(1, 0)]; 3],
            final_evals: vec![fp2(2, 0)],
        };
        for fail_at in 0..4 {
            let mut stub = Stub::failing_at(fail_at);
            let outcome =
                verify_batch_with(&mut stub, Prefix::Stage(1), 1, &[instance], &proof, product);
            assert_eq!(
                outcome,
                Err(SumcheckError::Transcript(TranscriptError::Exhausted)),
                "challenge {fail_at}"
            );
            assert_eq!(outcome.unwrap_err().rejected_round(), None);
            // The failed challenge is the last operation.
            assert!(
                matches!(stub.ops.last(), Some(Op::Challenge(_))),
                "{:?}",
                stub.ops
            );
            let challenges = stub
                .ops
                .iter()
                .filter(|op| matches!(op, Op::Challenge(_)))
                .count();
            assert_eq!(challenges, fail_at + 1);
        }
    }

    #[test]
    fn faults_of_the_proof_have_a_round_and_faults_of_the_verifier_none() {
        let proof_faults = [
            SumcheckError::MissingRound { round: 3 },
            SumcheckError::RoundLength {
                round: 3,
                expected: 2,
                found: 1,
            },
            SumcheckError::ExtraRound { round: 3 },
            SumcheckError::EvaluationCount {
                round: 3,
                expected: 1,
                found: 0,
            },
            SumcheckError::FinalMismatch { round: 3 },
        ];
        for error in proof_faults {
            assert_eq!(error.rejected_round(), Some(3), "{error}");
        }
        let verifier_faults = [
            SumcheckError::Degree { degree: 8 },
            SumcheckError::NoInstances,
            SumcheckError::EvaluationTotal,
            SumcheckError::Relation { instance: 0 },
            SumcheckError::Label(tv_hash::transcript::Ascii::new("\n").unwrap_err()),
            SumcheckError::Transcript(TranscriptError::Exhausted),
        ];
        for error in verifier_faults {
            assert_eq!(error.rejected_round(), None, "{error}");
        }
    }
}
