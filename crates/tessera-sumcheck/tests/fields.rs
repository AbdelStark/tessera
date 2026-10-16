// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The engine over the calibration field (07 §4): the same code proves batches over
//! `F65537Ext2`, with tables over `F65537` and over the extension, and over `F65537` itself,
//! whose challenges then lie in the prime field, on a transcript the test supplies. `tv-core`
//! checks Goldilocks proofs only, so each proof is checked here as RFC-0010 'Sumcheck messages'
//! says a verifier does, round by round from the batched claim with `p(1) = claim − p(0)` and
//! `tessera_poly::RoundPoly`, which the prover does not use, and each final evaluation against
//! the table's multilinear extension at the instance's point.

mod common;

use common::SplitMix64;
use tessera_codec::{Ascii, Encode};
use tessera_field::{F65537, F65537Ext2, PrimeField, QuadraticExtension};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_poly::{PolyField, RoundPoly, evaluate, evaluate_base};
use tessera_sumcheck::{
    Instance, MAX_DEGREE, Prefix, SumcheckProof, SumcheckTranscript, Table, prove_batch,
};

/// A transcript over `F65537` and its extension: messages absorbed as TCE `Vec`, challenges
/// drawn exactly uniformly by rejecting 17-bit words at or above 65537.
struct SmallTranscript {
    inner: Transcript,
    /// The 17-bit words of the challenge's last squeeze not yet used.
    words: Vec<u64>,
}

impl SmallTranscript {
    fn new(seed: u8) -> Self {
        let context = SessionContext {
            protocol: Ascii::new("tessera-protocol/1").expect("ASCII"),
            mode: Mode::NonInteractive,
            params: ParamSetId(Ascii::new("test-small/1").expect("ASCII")),
            statement_digest: [seed; 32],
            session_id: [seed; 16],
        };
        Self {
            inner: Transcript::new(&context),
            words: Vec::new(),
        }
    }

    /// The next element of the challenge being drawn under `label`.
    fn element(&mut self, label: &str) -> F65537 {
        loop {
            if self.words.is_empty() {
                let out = self.inner.squeeze(label);
                self.words = out
                    .chunks_exact(8)
                    .map(|w| u64::from_le_bytes(w.try_into().expect("8 bytes")) & 0x1_FFFF)
                    .collect();
            }
            if let Some(x) = F65537::from_canonical_u64(self.words.remove(0)) {
                return x;
            }
        }
    }

    fn absorb<T: Encode + Clone>(&mut self, label: &str, message: &[T]) {
        self.inner.absorb(label, &message.to_vec());
    }
}

impl SumcheckTranscript<F65537Ext2> for SmallTranscript {
    fn absorb_message(&mut self, label: &str, message: &[F65537Ext2]) {
        self.absorb(label, message);
    }

    fn challenge(&mut self, label: &str) -> F65537Ext2 {
        self.words.clear();
        let c0 = self.element(label);
        F65537Ext2::new(c0, self.element(label))
    }
}

impl SumcheckTranscript<F65537> for SmallTranscript {
    fn absorb_message(&mut self, label: &str, message: &[F65537]) {
        self.absorb(label, message);
    }

    fn challenge(&mut self, label: &str) -> F65537 {
        self.words.clear();
        self.element(label)
    }
}

/// A table over the prime field or over `F`.
enum Values<F: PolyField> {
    Base(Vec<F::Base>),
    Ext(Vec<F>),
}

impl<F: PolyField> Values<F> {
    fn table(&self) -> Table<'_, F> {
        match self {
            Self::Base(v) => Table::base(v),
            Self::Ext(v) => Table::ext(v),
        }
    }

    fn at(&self, x: usize) -> F {
        match self {
            Self::Base(v) => F::from(v[x]),
            Self::Ext(v) => v[x],
        }
    }

    fn mle(&self, point: &[F]) -> F {
        match self {
            Self::Base(v) => evaluate_base(v, point),
            Self::Ext(v) => evaluate(v, point),
        }
        .expect("a table of 2^n entries at n coordinates")
    }
}

/// Checks a proof as a verifier would, from the instances' sums over their hypercubes.
fn check<F: PolyField>(
    t: &mut SmallTranscript,
    prefix: &str,
    degree: usize,
    instances: &[(usize, Vec<Values<F>>)],
    proof: &SumcheckProof<F>,
) where
    SmallTranscript: SumcheckTranscript<F>,
{
    let n_max = instances
        .iter()
        .map(|(n, _)| *n)
        .max()
        .expect("an instance");
    let gamma = t.challenge(&format!("{prefix}/gamma"));
    let mut claim = F::ZERO;
    let mut weight = F::ONE;
    for (vars, tables) in instances {
        let sum = (0..1 << vars).fold(F::ZERO, |s, x| {
            s + tables.iter().fold(F::ONE, |p, table| p * table.at(x))
        });
        let scale = F::from(F::Base::from_u64_reduced(1 << (n_max - vars)));
        claim += weight * scale * sum;
        weight *= gamma;
    }
    assert_eq!(proof.rounds.len(), n_max);
    let mut point = Vec::new();
    for (j, round) in proof.rounds.iter().enumerate() {
        assert_eq!(round.len(), degree, "round {j} holds d values");
        t.absorb_message(&format!("{prefix}/round/{j}"), round);
        let r = t.challenge(&format!("{prefix}/r/{j}"));
        claim = RoundPoly::new(round).expect("a degree").evaluate(claim, r);
        point.push(r);
    }
    t.absorb_message(&format!("{prefix}/evals"), &proof.final_evals);
    let mut evals = proof.final_evals.iter();
    let mut batched = F::ZERO;
    let mut weight = F::ONE;
    for (vars, tables) in instances {
        let own = &point[n_max - vars..];
        let mut value = F::ONE;
        for table in tables {
            let eval = *evals.next().expect("an evaluation per table");
            assert_eq!(
                eval,
                table.mle(own),
                "the table's extension at the instance's point"
            );
            value *= eval;
        }
        batched += weight * value;
        weight *= gamma;
    }
    assert!(evals.next().is_none());
    assert_eq!(batched, claim, "the final check");
}

trait Draw: PolyField {
    fn draw(rng: &mut SplitMix64) -> Self;
}

impl Draw for F65537 {
    fn draw(rng: &mut SplitMix64) -> Self {
        Self::from_u64_reduced(rng.next_u64())
    }
}

impl Draw for F65537Ext2 {
    fn draw(rng: &mut SplitMix64) -> Self {
        let c0 = F65537::draw(rng);
        Self::new(c0, F65537::draw(rng))
    }
}

/// Random batches over `F`: 1 to 4 instances on 0 to 8 variables, degree 1 to 7, tables of
/// random kinds.
fn random_batches<F: Draw<Base = F65537>>(seed: u64)
where
    SmallTranscript: SumcheckTranscript<F>,
{
    let mut rng = SplitMix64::new(seed);
    for case in 0..100u8 {
        let degree = 1 + rng.below(MAX_DEGREE as u64) as usize;
        let count = 1 + rng.below(4) as usize;
        let instances: Vec<(usize, Vec<Values<F>>)> = (0..count)
            .map(|_| {
                let vars = rng.below(9) as usize;
                let k = 1 + rng.below(degree as u64) as usize;
                let tables = (0..k)
                    .map(|_| match rng.coin() {
                        true => {
                            Values::Base((0..1 << vars).map(|_| F65537::draw(&mut rng)).collect())
                        }
                        false => Values::Ext((0..1 << vars).map(|_| F::draw(&mut rng)).collect()),
                    })
                    .collect();
                (vars, tables)
            })
            .collect();
        let borrowed: Vec<Instance<'_, F>> = instances
            .iter()
            .map(|(_, tables)| Instance::new(tables.iter().map(Values::table).collect()))
            .collect();
        let mut prover = SmallTranscript::new(case);
        let proven = prove_batch(&mut prover, Prefix::Stage(5), degree, &borrowed)
            .expect("a well-formed batch");
        let mut verifier = SmallTranscript::new(case);
        check(&mut verifier, "stage/5", degree, &instances, proven.proof());
        assert_eq!(
            verifier.inner.state(),
            prover.inner.state(),
            "the transcripts end in the same state"
        );
    }
}

#[test]
fn batches_over_the_extension_of_f65537_verify() {
    random_batches::<F65537Ext2>(0x53_0401);
}

#[test]
fn batches_with_challenges_in_f65537_verify() {
    random_batches::<F65537>(0x53_0402);
}
