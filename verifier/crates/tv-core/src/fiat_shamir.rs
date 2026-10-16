// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The transcript as the sumcheck and LogUp checks use it: labels built from RFC-0014's
//! patterns, messages absorbed as TCE `Vec<Fp2>`, challenges in `Fp2`.
//!
//! The checks run on any [`FiatShamir`] transcript. The verifier's is `tv-hash`'s
//! [`Transcript`]; the unit tests put a recording stub in its place, which sees every label and
//! message in order and can make any challenge fail as RFC-0004's bound on squeezes does, a
//! branch BLAKE3 reaches with probability below `2^-496`.

use tv_field::Fp2;
use tv_hash::transcript::{Ascii, NotAscii, Transcript, TranscriptError};

use crate::sumcheck::{Prefix, SumcheckError};

/// What the checks need from a Fiat–Shamir transcript (RFC-0004 'Transcript').
pub(crate) trait FiatShamir {
    /// Absorbs the TCE bytes of a prover message under `label`.
    fn absorb(&mut self, label: Ascii<'_>, tce: &[u8]);
    /// Squeezes a challenge exactly uniform in `Fp2` under `label`.
    fn challenge_fp2(&mut self, label: Ascii<'_>) -> Result<Fp2, TranscriptError>;
}

impl FiatShamir for Transcript {
    fn absorb(&mut self, label: Ascii<'_>, tce: &[u8]) {
        Transcript::absorb(self, label, tce);
    }

    fn challenge_fp2(&mut self, label: Ascii<'_>) -> Result<Fp2, TranscriptError> {
        Transcript::challenge_fp2(self, label)
    }
}

/// A transcript operation of a sumcheck, which names its label under the sumcheck's prefix
/// (RFC-0014 'Transcript schedule'; indices count from 0).
#[derive(Clone, Copy, Debug)]
pub(crate) enum Step {
    /// `<prefix>/gamma`, the batching challenge.
    Gamma,
    /// `<prefix>/lambda`, a tree level's challenge combining a fraction tree's two claims.
    Lambda,
    /// `<prefix>/round/{j}`, the message of round `j`.
    Round(usize),
    /// `<prefix>/r/{j}`, the challenge of round `j`.
    Challenge(usize),
    /// `<prefix>/evals`, the final evaluations.
    Evals,
    /// `<prefix>/line`, a tree level's line challenge.
    Line,
}

/// The label of `step` under `prefix`.
fn label(prefix: Prefix, step: Step) -> String {
    match step {
        Step::Gamma => format!("{prefix}/gamma"),
        Step::Lambda => format!("{prefix}/lambda"),
        Step::Round(j) => format!("{prefix}/round/{j}"),
        Step::Challenge(j) => format!("{prefix}/r/{j}"),
        Step::Evals => format!("{prefix}/evals"),
        Step::Line => format!("{prefix}/line"),
    }
}

/// A transcript operation that could not run: its label is not printable ASCII, or its
/// challenge found no value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Failure {
    /// The label.
    Label(NotAscii),
    /// The challenge.
    Transcript(TranscriptError),
}

impl From<Failure> for SumcheckError {
    fn from(failure: Failure) -> Self {
        match failure {
            Failure::Label(error) => SumcheckError::Label(error),
            Failure::Transcript(error) => SumcheckError::Transcript(error),
        }
    }
}

/// Absorbs `values`, a prover message, as TCE `Vec<Fp2>` under `label`.
pub(crate) fn absorb_under(
    transcript: &mut impl FiatShamir,
    label: &str,
    values: &[Fp2],
) -> Result<(), Failure> {
    let label = Ascii::new(label).map_err(Failure::Label)?;
    transcript.absorb(label, &tce(values));
    Ok(())
}

/// Squeezes a challenge under `label`.
pub(crate) fn challenge_under(
    transcript: &mut impl FiatShamir,
    label: &str,
) -> Result<Fp2, Failure> {
    let label = Ascii::new(label).map_err(Failure::Label)?;
    transcript.challenge_fp2(label).map_err(Failure::Transcript)
}

/// The TCE encoding of a `Vec<Fp2>`: the length as a little-endian `u64`, then each element as
/// `c0 ‖ c1`, 8 little-endian bytes each (RFC-0003 'Primitive types').
pub(crate) fn tce(values: &[Fp2]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len().saturating_mul(16).saturating_add(8));
    // A slice in memory is shorter than 2^64 elements.
    let length = u64::try_from(values.len()).unwrap_or(u64::MAX);
    out.extend_from_slice(&length.to_le_bytes());
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

/// Absorbs `values`, a prover message, as TCE `Vec<Fp2>` under the label of `step`.
pub(crate) fn absorb(
    transcript: &mut impl FiatShamir,
    prefix: Prefix,
    step: Step,
    values: &[Fp2],
) -> Result<(), SumcheckError> {
    absorb_under(transcript, &label(prefix, step), values).map_err(SumcheckError::from)
}

/// Squeezes the challenge of `step`.
pub(crate) fn challenge(
    transcript: &mut impl FiatShamir,
    prefix: Prefix,
    step: Step,
) -> Result<Fp2, SumcheckError> {
    challenge_under(transcript, &label(prefix, step)).map_err(SumcheckError::from)
}

#[cfg(test)]
pub(crate) mod stub {
    //! A transcript stub for unit tests: it records every operation, and answers challenges
    //! with values it derives from the operation count, or with an error at a chosen challenge.

    use super::*;
    use tv_field::Fp;

    /// One recorded operation.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub(crate) enum Op {
        /// An absorb: the label and the TCE bytes.
        Absorb(String, Vec<u8>),
        /// A challenge: the label.
        Challenge(String),
    }

    /// Records operations; the challenge numbered `fail_at` (from 0) fails as an exhausted
    /// transcript does.
    #[derive(Debug, Default)]
    pub(crate) struct Stub {
        pub(crate) ops: Vec<Op>,
        pub(crate) fail_at: Option<usize>,
        challenges: usize,
    }

    impl Stub {
        pub(crate) fn failing_at(challenge: usize) -> Self {
            Self {
                fail_at: Some(challenge),
                ..Self::default()
            }
        }

        /// The labels of the recorded operations, absorbs marked `+` and challenges `?`.
        pub(crate) fn script(&self) -> Vec<String> {
            self.ops
                .iter()
                .map(|op| match op {
                    Op::Absorb(label, _) => format!("+{label}"),
                    Op::Challenge(label) => format!("?{label}"),
                })
                .collect()
        }
    }

    impl FiatShamir for Stub {
        fn absorb(&mut self, label: Ascii<'_>, tce: &[u8]) {
            self.ops
                .push(Op::Absorb(label.as_str().to_owned(), tce.to_vec()));
        }

        fn challenge_fp2(&mut self, label: Ascii<'_>) -> Result<Fp2, TranscriptError> {
            self.ops.push(Op::Challenge(label.as_str().to_owned()));
            let n = self.challenges;
            self.challenges += 1;
            if self.fail_at == Some(n) {
                return Err(TranscriptError::Exhausted);
            }
            // Distinct challenges, far from the interpolation nodes.
            Ok(Fp2::new(
                Fp::reduce(1000 + n as u64),
                Fp::reduce(77 * n as u64 + 5),
            ))
        }
    }
}
