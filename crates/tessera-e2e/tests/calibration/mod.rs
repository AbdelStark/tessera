// SPDX-License-Identifier: Apache-2.0 OR MIT

//! What the soundness-calibration harness shares (07 §4): the test field, a transcript that
//! draws challenges in it, a scripted transcript that sets a challenge, the polynomial arithmetic
//! of the verifiers written here, and a parallel trial runner.
//!
//! The verifiers' arithmetic is written from RFC-0010 'Multilinear extensions' and 'Sumcheck
//! messages', not taken from `tessera-poly`, so that a check here does not share a mistake with
//! the code whose proofs it checks.

pub mod interval;
pub mod logup;
pub mod sumcheck;

use std::panic;
use std::sync::OnceLock;
use std::thread;

use tessera_codec::Ascii;
use tessera_field::{F65537, Field, PrimeField};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_sumcheck::{MAX_DEGREE, SumcheckTranscript};

/// The test field, `P = 2^16 + 1` (RFC-0002 'Traits'): every challenge is drawn in it.
pub type F = F65537;

/// `|F|`.
pub const FIELD: u64 = F65537::MODULUS;

/// A verifier's verdict on one trial, with the challenge the cheating prover bets on.
#[derive(Clone, Copy, Debug)]
pub struct Verdict {
    /// Whether every check passed.
    pub accepted: bool,
    /// The challenge drawn: `r_0` for a sumcheck, `α` for LogUp.
    pub challenge: F,
}

/// RFC-0004's transcript, BLAKE3 over TCE messages, with challenges in `F65537`, drawn as
/// [`Transcript::challenge_fp`] draws in `Fp` but from 17-bit words (`2^16 < P < 2^17`): of a
/// squeeze's four little-endian words, each masked to its low 17 bits, the challenge is the first
/// below `P`; if none is, the transcript squeezes again under the same label. Every element of
/// the field is then drawn with probability exactly `1/P`, if the hash's output is uniform.
/// Messages are absorbed as TCE `Vec<F65537>`.
pub struct SmallTranscript(Transcript);

impl SmallTranscript {
    /// The transcript of trial `trial` of the setting `setting`: a session of its own, with the
    /// setting as its statement digest and the trial as its session id, fresh per session
    /// (INV-PROTO-4). Live, as a session whose prover gets one attempt at its challenges is.
    pub fn session(setting: u64, trial: u64) -> Self {
        let mut statement_digest = [0; 32];
        statement_digest[..8].copy_from_slice(&setting.to_le_bytes());
        let mut session_id = [0; 16];
        session_id[..8].copy_from_slice(&trial.to_le_bytes());
        let context = SessionContext {
            protocol: Ascii::new("tessera-protocol/1").expect("ASCII"),
            mode: Mode::Live,
            params: ParamSetId(Ascii::new("test-small/1").expect("ASCII")),
            statement_digest,
            session_id,
        };
        Self(Transcript::new(&context))
    }
}

impl SumcheckTranscript<F> for SmallTranscript {
    fn absorb_message(&mut self, label: &str, message: &[F]) {
        self.0.absorb(label, &message.to_vec());
    }

    fn challenge(&mut self, label: &str) -> F {
        loop {
            let out = self.0.squeeze(label);
            let found = out
                .chunks_exact(8)
                .map(|word| u64::from_le_bytes(word.try_into().expect("8 bytes")) & 0x1_FFFF)
                .find_map(F::from_canonical_u64);
            if let Some(x) = found {
                return x;
            }
        }
    }
}

/// A transcript that absorbs nothing and answers each challenge from a script: its value for a
/// label the script names, and for any other an element derived from the label and a seed, so
/// that a prover and a verifier given the same script draw the same challenges. It shows which
/// values of a challenge a check accepts.
pub struct Scripted {
    script: Vec<(String, F)>,
    seed: u64,
}

impl Scripted {
    /// The script `script`, other labels answered from `seed`.
    pub fn new(script: &[(&str, F)], seed: u64) -> Self {
        Self {
            script: script
                .iter()
                .map(|&(label, x)| (label.to_owned(), x))
                .collect(),
            seed,
        }
    }
}

impl SumcheckTranscript<F> for Scripted {
    fn absorb_message(&mut self, _label: &str, _message: &[F]) {}

    fn challenge(&mut self, label: &str) -> F {
        if let Some((_, x)) = self.script.iter().find(|(named, _)| named == label) {
            return *x;
        }
        // FNV-1a of the label, then SplitMix64.
        let hash = label
            .bytes()
            .fold(0xCBF2_9CE4_8422_2325 ^ self.seed, |h, b| {
                (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3)
            });
        F::from_u64_reduced(SplitMix64::new(hash).next_u64())
    }
}

/// SplitMix64: reproducible instances.
pub struct SplitMix64(u64);

impl SplitMix64 {
    /// The generator from `seed`.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next 64 bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// An element of the field, near uniform.
    pub fn element(&mut self) -> F {
        F::from_u64_reduced(self.next_u64())
    }

    /// An integer in `0..n`, near uniform.
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

/// `f̃(point)` for a table of `2^n` entries at `n` coordinates (RFC-0010 'Multilinear
/// extensions'): the first coordinate binds the lowest bit of the index, each pair
/// `(f[2k], f[2k + 1])` becoming `f[2k] + r·(f[2k + 1] − f[2k])`.
pub fn mle(table: &[F], point: &[F]) -> F {
    assert_eq!(
        table.len(),
        1 << point.len(),
        "2^n entries at n coordinates"
    );
    let mut values = table.to_vec();
    for &r in point {
        values = values
            .chunks_exact(2)
            .map(|pair| pair[0] + r * (pair[1] - pair[0]))
            .collect();
    }
    values[0]
}

/// `eq(a, b) = Π_i (a_i·b_i + (1 − a_i)·(1 − b_i))`.
pub fn eq(a: &[F], b: &[F]) -> F {
    assert_eq!(a.len(), b.len(), "points of one dimension");
    a.iter().zip(b).fold(F::ONE, |product, (&a, &b)| {
        product * (a * b + (F::ONE - a) * (F::ONE - b))
    })
}

/// The integer `k` in the field.
pub fn int(k: usize) -> F {
    F::from_u64_reduced(k as u64)
}

/// `w[d][k] = 1/Π_{m ≠ k} (k − m)` over the nodes `0, …, d`.
fn weights() -> &'static [[F; MAX_DEGREE + 1]; MAX_DEGREE + 1] {
    static WEIGHTS: OnceLock<[[F; MAX_DEGREE + 1]; MAX_DEGREE + 1]> = OnceLock::new();
    WEIGHTS.get_or_init(|| {
        let mut w = [[F::ZERO; MAX_DEGREE + 1]; MAX_DEGREE + 1];
        for (d, row) in w.iter_mut().enumerate() {
            for (k, weight) in row.iter_mut().enumerate().take(d + 1) {
                let product = (0..=d)
                    .filter(|&m| m != k)
                    .fold(F::ONE, |p, m| p * (int(k) - int(m)));
                *weight = product.inverse().expect("distinct nodes");
            }
        }
        w
    })
}

/// `p(r)` for the polynomial of degree at most `d` with `p(k) = values[k]` at the nodes
/// `k = 0, …, d`, by Lagrange's formula.
pub fn interpolate(values: &[F], r: F) -> F {
    let d = values.len() - 1;
    let weights = &weights()[d];
    (0..=d).fold(F::ZERO, |sum, k| {
        let basis = (0..=d)
            .filter(|&m| m != k)
            .fold(weights[k], |p, m| p * (r - int(m)));
        sum + values[k] * basis
    })
}

/// `p(r)` from a round message `p(0), p(2), …, p(d)` and the round's claim `p(0) + p(1)`
/// (RFC-0010 'Sumcheck messages'): the receiver derives `p(1) = claim − p(0)`.
pub fn round_value(claim: F, message: &[F], r: F) -> F {
    let mut values = Vec::with_capacity(message.len() + 1);
    values.push(message[0]);
    values.push(claim - message[0]);
    values.extend_from_slice(&message[1..]);
    interpolate(&values, r)
}

/// Runs `trial(i)` for every `i < trials` on all the cores and counts those it accepts. Trial `i`
/// depends on `i` alone, so the count does not depend on how many cores there are. A panic in a
/// trial, a failed check, is raised again here.
pub fn count_accepted(trials: u64, trial: impl Fn(u64) -> bool + Sync) -> u64 {
    let threads = thread::available_parallelism().map_or(1, |n| n.get());
    let trial = &trial;
    thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|t| {
                scope.spawn(move || {
                    (t as u64..trials)
                        .step_by(threads)
                        .filter(|&i| trial(i))
                        .count() as u64
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| worker.join().unwrap_or_else(|e| panic::resume_unwind(e)))
            .sum()
    })
}
