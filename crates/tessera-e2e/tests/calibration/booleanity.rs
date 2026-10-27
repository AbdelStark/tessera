// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A witness with one non-Boolean bit against stage 1's booleanity zero-check (RFC-0072 'Testing
//! Strategy' 'Calibration', RFC-0066 'BitDecompose', RFC-0035 'Zero-check points'), and a
//! verifier from RFC-0010 'Sumcheck messages' and 'Batching within a stage'.
//!
//! The prover commits a bit table `b` on `n` variables whose entries are bits but one, `v = 2`
//! at the vertex `x_0`: two bits of a decomposition that still recomposes, `2·2^j = 2^{j+1}`,
//! as RFC-0066's "carry made up by a non-Boolean bit" is. Over F65537 such a decomposition must
//! keep `low + high ≤ 16` (RFC-0066 Lemma 1); the booleanity check reads the bits alone.
//!
//! - **The zero-check.** After the table is bound, the transcript draws the zero-check point `z`
//!   under `stage/1/zero/{c}`, and the instance `Σ_x eq(z, x)·b̃(x)·(1 − b̃(x))` is proven against
//!   the claim 0 by `tessera_sumcheck::prove_batch`, as RFC-0066's booleanity
//!   ([`Elementwise::booleanity`]). Its true sum is `S(z) = eq(z, x_0)·v·(1 − v)`, so it is 0
//!   exactly when `eq(z, x_0) = 0`: when some coordinate `z_c` is `1 − x_{0,c}`, the one value of
//!   `z_c` at which the factor `z_c·x_{0,c} + (1 − z_c)·(1 − x_{0,c})` vanishes.
//! - **Honest otherwise.** The prover then sends the engine's messages, which prove `S(z)`: when
//!   `S(z) = 0` the proof verifies. Otherwise the verifier's claim exceeds the honest one by
//!   `Δ = −S(z) ≠ 0`, and each round sends the honest values plus `Δ/2`, which the verifier reads
//!   as the polynomial `p_j + Δ/2`: the excess halves each round and never vanishes, so the final
//!   check fails, as the round lie's does after a miss ([`super::sumcheck`]).
//!
//! So the proof is accepted exactly when `z` zeroes `eq(z, x_0)`, an event of probability
//! `1 − (1 − 1/|F|)^n ≤ n/|F|` for a uniform `z`: RFC-0014's `n_i/|F|` for a zero-check of `n_i`
//! variables, attained up to `n²/(2|F|²)`.

use tessera_field::Field;
use tessera_sumcheck::{
    Elementwise, Input, Prefix, Protocol, SumcheckProof, SumcheckTranscript, prove_batch,
    zero_check_point,
};

use super::{F, FIELD, SplitMix64, eq, int, mle, round_value};

/// Stage 1, whose zero-check point and sumcheck the labels name.
const PREFIX: Prefix = Prefix::Stage(1);

/// The degree of stage 1's booleanity: `eq·b·b`.
const DEGREE: usize = 3;

/// A booleanity zero-check's verdict: whether every check passed, and the zero-check point the
/// cheating prover bets on.
#[derive(Clone, Debug)]
pub struct ZeroVerdict {
    /// Whether every check passed.
    pub accepted: bool,
    /// The zero-check point `z`.
    pub point: Vec<F>,
}

/// One non-Boolean bit among `2^n` bits: the table, and the vertex `x_0` that holds `v = 2`.
pub struct NonBoolean {
    vars: usize,
    bits: Vec<F>,
    at: usize,
}

impl NonBoolean {
    /// The bits of `rng` on `vars ≥ 1` variables, with `2` at a vertex of `rng`.
    pub fn new(vars: usize, rng: &mut SplitMix64) -> Self {
        assert!(vars >= 1);
        let len = 1usize << vars;
        let at = usize::try_from(rng.below(len as u64)).expect("small");
        let mut bits: Vec<F> = (0..len)
            .map(|_| int((rng.next_u64() & 1) as usize))
            .collect();
        bits[at] = int(2);
        Self { vars, bits, at }
    }

    /// Bits alone: an honest table, whose zero-check holds at every point.
    pub fn honest(vars: usize, rng: &mut SplitMix64) -> Self {
        let mut table = Self::new(vars, rng);
        table.bits[table.at] = int((rng.next_u64() & 1) as usize);
        table
    }

    /// For each coordinate `c`, the value of `z_c` at which `eq(z, x_0)` vanishes: `1 − x_{0,c}`.
    pub fn roots(&self) -> Vec<F> {
        (0..self.vars)
            .map(|c| int(1 - ((self.at >> c) & 1)))
            .collect()
    }

    /// Whether the proof at `z` must be accepted: `eq(z, x_0) = 0`, or the bits are all bits.
    pub fn accepts(&self, z: &[F]) -> bool {
        let boolean = self.bits.iter().all(|&b| b == F::ZERO || b == F::ONE);
        boolean || z.iter().zip(self.roots()).any(|(&z, root)| z == root)
    }

    /// The true sum `Σ_x eq(z, x)·b(x)·(1 − b(x))` at `z`.
    fn sum(&self, z: &[F]) -> F {
        (0..self.bits.len()).fold(F::ZERO, |sum, x| {
            let vertex: Vec<F> = (0..self.vars).map(|c| int((x >> c) & 1)).collect();
            let b = self.bits[x];
            sum + eq(z, &vertex) * b * (F::ONE - b)
        })
    }

    /// One trial: the table bound, the zero-check proven on `prover`'s transcript by a prover
    /// honest but for the table, and the proof checked on `verifier`'s.
    pub fn trial(
        &self,
        prover: &mut impl SumcheckTranscript<F>,
        verifier: &mut impl SumcheckTranscript<F>,
    ) -> ZeroVerdict {
        // The statement: the table, bound before any challenge, as its commitment is.
        prover.absorb_message("round1", &self.bits);
        let mut lying = Lying {
            inner: prover,
            excess: F::ZERO,
            sent: Vec::with_capacity(self.vars),
        };
        let z: Vec<F> = zero_check_point(&mut lying, 1, self.vars);
        // The verifier's claim is 0; the honest one is the true sum.
        lying.excess = -self.sum(&z);
        let bits: Vec<F> = self.bits.clone();
        let relation = Elementwise::<F>::booleanity(Input::Operand(&bits), self.vars)
            .expect("a booleanity relation");
        let tables = relation.tables(&z).expect("the zero-check point");
        let proven = prove_batch(
            &mut lying,
            Protocol::One,
            PREFIX,
            DEGREE,
            &[tables.instance()],
        )
        .expect("a well-formed instance");
        let proof = SumcheckProof {
            rounds: lying.sent,
            final_evals: proven.proof().final_evals.clone(),
        };
        verifier.absorb_message("round1", &self.bits);
        verify(verifier, &self.bits, &proof)
    }
}

/// `1/2 = (P + 1)/2`.
fn half() -> F {
    int(FIELD.div_ceil(2) as usize)
}

/// The prover's transcript: it passes every message and challenge through, but adds half the
/// verifier's excess to each round's message, and keeps the messages it sent.
struct Lying<'a, T> {
    inner: &'a mut T,
    /// By how much the verifier's claim exceeds the honest claim at the current round.
    excess: F,
    sent: Vec<Vec<F>>,
}

impl<T: SumcheckTranscript<F>> SumcheckTranscript<F> for Lying<'_, T> {
    fn absorb_message(&mut self, label: &str, message: &[F]) {
        if !label.starts_with("stage/1/round/") {
            return self.inner.absorb_message(label, message);
        }
        let shift = self.excess * half();
        let sent: Vec<F> = message.iter().map(|&m| m + shift).collect();
        self.inner.absorb_message(label, &sent);
        self.sent.push(sent);
    }

    fn challenge(&mut self, label: &str) -> F {
        let r = self.inner.challenge(label);
        if label.starts_with("stage/1/r/") {
            self.excess *= half();
        }
        r
    }
}

/// Checks a proof of the booleanity zero-check `0 = Σ_x eq(z, x)·b̃(x)·(1 − b̃(x))` as RFC-0010
/// 'Sumcheck messages' and RFC-0035 'Zero-check points' say a verifier does, with the table at
/// hand in place of its opening: `z` under `stage/1/zero/{c}`, `γ`, then for each of the `n`
/// rounds `d = 3` values, absorbed before `r_j`, and `claim_{j+1} = p_j(r_j)` from the claim 0;
/// then one evaluation, absorbed, the table's extension at the point `r`, and the last claim
/// `eq(z, r)·e·(1 − e)`. The verdict carries `z`.
pub fn verify(
    t: &mut impl SumcheckTranscript<F>,
    bits: &[F],
    proof: &SumcheckProof<F>,
) -> ZeroVerdict {
    let vars = bits.len().trailing_zeros() as usize;
    let z: Vec<F> = (0..vars)
        .map(|c| t.challenge(&format!("stage/1/zero/{c}")))
        .collect();
    // One instance: its weight is γ^0 and its scale 2^0, so γ enters nothing.
    let _gamma = t.challenge("stage/1/gamma");
    let mut verdict = ZeroVerdict {
        accepted: false,
        point: z.clone(),
    };
    if proof.rounds.len() != vars {
        return verdict;
    }
    let mut claim = F::ZERO;
    let mut point = Vec::with_capacity(vars);
    for (j, round) in proof.rounds.iter().enumerate() {
        if round.len() != DEGREE {
            return verdict;
        }
        t.absorb_message(&format!("stage/1/round/{j}"), round);
        let r = t.challenge(&format!("stage/1/r/{j}"));
        claim = round_value(claim, round, r);
        point.push(r);
    }
    let [e] = proof.final_evals[..] else {
        return verdict;
    };
    t.absorb_message("stage/1/evals", &proof.final_evals);
    let opened = mle(bits, &point) == e;
    verdict.accepted = opened && eq(&z, &point) * e * (F::ONE - e) == claim;
    verdict
}
