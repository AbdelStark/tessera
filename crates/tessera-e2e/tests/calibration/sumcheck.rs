// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A prover that lies in one sumcheck round, and a verifier from RFC-0010 'Sumcheck messages'
//! (RFC-0010 'Testing Strategy': "a prover lying about one round succeeds at the rate `d/|F|`").
//!
//! The instance is a product of `d` random tables over `n` variables, whose sum `S` the engine
//! proves. The prover claims `S + δ(0) + δ(1)` instead, `δ` a polynomial of degree `d` with `d`
//! roots in the field, and runs `tessera_sumcheck::prove_batch` on a transcript that alters what
//! the engine absorbs:
//!
//! - **Round 0** sends `p_0 + δ`, the honest round polynomial plus `δ`, consistent with the false
//!   claim. The verifier's next claim is `p_0(r_0) + δ(r_0)`: the honest one exactly when `r_0`
//!   is a root of `δ`, and then every later message is the engine's honest one.
//! - **After a miss**, the verifier's claim exceeds the honest one by some `Δ ≠ 0`. Each later
//!   round sends the honest values plus `Δ/2` at every node, which the verifier reads as the
//!   polynomial `p_j + Δ/2`, since it derives `p(1) = claim − p(0)`. Its next claim exceeds the
//!   honest one by `Δ/2`: the excess halves each round and never vanishes, a constant having no
//!   root, so the final check fails.
//!
//! So the proof is accepted exactly when `r_0` is one of `δ`'s `d` roots, whatever the other
//! challenges: a uniform `r_0` gives the rate `d/|F|` exactly, RFC-0014's bound for one round of
//! degree `d`, and no message of `d` values can do better, `p_0' − p_0` having at most `d` roots.

use tessera_field::{Field, PrimeField};
use tessera_sumcheck::{Instance, Prefix, SumcheckProof, SumcheckTranscript, Table, prove_batch};

use super::{F, FIELD, SplitMix64, Verdict, int, mle, round_value};

/// The stage of the lie: its labels are `stage/0/…`.
const PREFIX: Prefix = Prefix::Stage(0);

/// A lie in round 0 of a sumcheck of degree `d` over `n` variables: its difference polynomial
/// `δ = Π_k (X − (P − k))`, `k = 1, …, d`.
pub struct RoundLie {
    degree: usize,
    vars: usize,
    roots: Vec<F>,
    /// `δ(0), δ(2), …, δ(d)`: what round 0 adds to the honest message.
    offsets: Vec<F>,
}

impl RoundLie {
    /// The lie of degree `degree`, `1 ≤ d ≤ 7`, on `vars ≥ 1` variables.
    pub fn new(degree: usize, vars: usize) -> Self {
        assert!((1..=tessera_sumcheck::MAX_DEGREE).contains(&degree) && vars >= 1);
        let roots: Vec<F> = (1..=degree as u64)
            .map(|k| F::from_u64_reduced(FIELD - k))
            .collect();
        let mut lie = Self {
            degree,
            vars,
            roots,
            offsets: Vec::new(),
        };
        lie.offsets = (0..=degree)
            .filter(|&k| k != 1)
            .map(|k| lie.delta(int(k)))
            .collect();
        lie
    }

    /// `δ`'s roots: the values of `r_0` at which the lie is accepted.
    pub fn roots(&self) -> &[F] {
        &self.roots
    }

    /// `δ(x) = Π_z (x − z)` over the roots.
    pub fn delta(&self, x: F) -> F {
        self.roots.iter().fold(F::ONE, |p, &z| p * (x - z))
    }

    /// `δ(0) + δ(1)`: by how much the claim exceeds the true sum.
    pub fn lie(&self) -> F {
        self.delta(F::ZERO) + self.delta(F::ONE)
    }

    /// One trial: a random instance from `rng`, the lie proven on `prover`'s transcript, and the
    /// proof checked on `verifier`'s.
    pub fn trial(
        &self,
        rng: &mut SplitMix64,
        prover: &mut impl SumcheckTranscript<F>,
        verifier: &mut impl SumcheckTranscript<F>,
    ) -> Verdict {
        run(self.degree, self.vars, Some(self), rng, prover, verifier)
    }

    /// One honest trial of degree `degree` on `vars` variables: the true sum, proven by the
    /// engine as it is.
    pub fn honest(
        degree: usize,
        vars: usize,
        rng: &mut SplitMix64,
        prover: &mut impl SumcheckTranscript<F>,
        verifier: &mut impl SumcheckTranscript<F>,
    ) -> Verdict {
        run(degree, vars, None, rng, prover, verifier)
    }
}

/// The instance, the proof of `lie`'s false claim (or of the true sum), and the verdict.
fn run(
    degree: usize,
    vars: usize,
    lie: Option<&RoundLie>,
    rng: &mut SplitMix64,
    prover: &mut impl SumcheckTranscript<F>,
    verifier: &mut impl SumcheckTranscript<F>,
) -> Verdict {
    let tables: Vec<Vec<F>> = (0..degree)
        .map(|_| (0..1 << vars).map(|_| rng.element()).collect())
        .collect();
    let sum = (0..1 << vars).fold(F::ZERO, |s, x| {
        s + tables.iter().fold(F::ONE, |p, table| p * table[x])
    });
    let claim = sum + lie.map_or(F::ZERO, RoundLie::lie);
    // The statement: the claim and the tables, which the transcript binds before `γ` as a stage's
    // input claims and commitments are (RFC-0010 'Batching within a stage').
    let statement: Vec<F> = std::iter::once(claim)
        .chain(tables.iter().flatten().copied())
        .collect();

    prover.absorb_message("round1", &statement);
    let instance = Instance::new(tables.iter().map(|t| Table::base(t)).collect());
    let mut lying = Lying {
        inner: prover,
        lie,
        excess: F::ZERO,
        sent: Vec::with_capacity(vars),
    };
    let proven =
        prove_batch(&mut lying, PREFIX, degree, &[instance]).expect("a well-formed instance");
    let proof = SumcheckProof {
        rounds: lying.sent,
        final_evals: proven.proof().final_evals.clone(),
    };

    verifier.absorb_message("round1", &statement);
    verify(verifier, degree, claim, &tables, &proof)
}

/// `1/2 = (P + 1)/2`.
fn half() -> F {
    F::from_u64_reduced(FIELD.div_ceil(2))
}

/// The lying prover's transcript: it passes every message and challenge through, but adds `δ` to
/// round 0's message and half the verifier's excess to each later round's, and keeps the
/// messages it sent. With no lie, it sends the engine's messages.
struct Lying<'a, T> {
    inner: &'a mut T,
    lie: Option<&'a RoundLie>,
    /// By how much the verifier's claim exceeds the honest claim at the current round, after
    /// round 0.
    excess: F,
    sent: Vec<Vec<F>>,
}

impl<T: SumcheckTranscript<F>> SumcheckTranscript<F> for Lying<'_, T> {
    fn absorb_message(&mut self, label: &str, message: &[F]) {
        let Some(round) = label.strip_prefix("stage/0/round/") else {
            return self.inner.absorb_message(label, message);
        };
        let shifts: Vec<F> = match (self.lie, round) {
            (Some(lie), "0") => lie.offsets.clone(),
            _ => vec![self.excess * half(); message.len()],
        };
        let sent: Vec<F> = message.iter().zip(&shifts).map(|(&m, &s)| m + s).collect();
        self.inner.absorb_message(label, &sent);
        self.sent.push(sent);
    }

    fn challenge(&mut self, label: &str) -> F {
        let r = self.inner.challenge(label);
        if let Some(round) = label.strip_prefix("stage/0/r/") {
            self.excess = match (self.lie, round) {
                (Some(lie), "0") => lie.delta(r),
                _ => self.excess * half(),
            };
        }
        r
    }
}

/// Checks a proof of `Σ_x Π_k T_k(x) = claim` as RFC-0010 'Sumcheck messages' says a verifier
/// does, with the tables at hand in place of their openings: `γ`, then for each of the `n`
/// rounds `d` values, absorbed before `r_j`, and `claim_{j+1} = p_j(r_j)`; then one evaluation
/// per table, absorbed, each the table's extension at the point, whose product is the last
/// claim. The verdict carries `r_0`.
pub fn verify(
    t: &mut impl SumcheckTranscript<F>,
    degree: usize,
    claim: F,
    tables: &[Vec<F>],
    proof: &SumcheckProof<F>,
) -> Verdict {
    let vars = tables[0].len().trailing_zeros() as usize;
    // One instance: its weight is γ^0 and its scale 2^0, so γ enters nothing.
    let _gamma = t.challenge("stage/0/gamma");
    let mut verdict = Verdict {
        accepted: false,
        challenge: F::ZERO,
    };
    if proof.rounds.len() != vars {
        return verdict;
    }
    let mut claim = claim;
    let mut point = Vec::with_capacity(vars);
    for (j, round) in proof.rounds.iter().enumerate() {
        if round.len() != degree {
            return verdict;
        }
        t.absorb_message(&format!("stage/0/round/{j}"), round);
        let r = t.challenge(&format!("stage/0/r/{j}"));
        if j == 0 {
            verdict.challenge = r;
        }
        claim = round_value(claim, round, r);
        point.push(r);
    }
    if proof.final_evals.len() != tables.len() {
        return verdict;
    }
    t.absorb_message("stage/0/evals", &proof.final_evals);
    let opened = tables
        .iter()
        .zip(&proof.final_evals)
        .all(|(table, &eval)| mle(table, &point) == eval);
    let product = proof.final_evals.iter().fold(F::ONE, |p, &e| p * e);
    verdict.accepted = opened && product == claim;
    verdict
}
