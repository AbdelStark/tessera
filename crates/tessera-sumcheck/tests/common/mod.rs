// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared test support: seeded inputs, a session's context on both sides, and the in-crate
//! helper that checks every proof with the verifier's independent sumcheck checks,
//! `tv_core::sumcheck::verify_batch`, on a `tv-hash` transcript of the same session (DEP-2).
//!
//! The claims the verifier receives are computed here from the tables, in `tv-field`'s
//! arithmetic, by summing each product over the hypercube; public factors are evaluated by the
//! verifier's own `eq` or by summing the table against `eq` at each vertex. None of it uses the
//! prover's algorithms. [`forests`] does the same for batches of product forests, around
//! `tv_core::tree::verify_product_level`.

#![allow(dead_code)] // each test binary uses part of it

pub mod forests;

use tessera_codec::Ascii;
use tessera_field::{Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_sumcheck::{Instance, Prefix, ProvenBatch, SumcheckProof, Table, prove_batch};
use tv_core::sumcheck as tv;

pub type Fp = Goldilocks;
pub type Fp2 = GoldilocksExt2;
pub type TvFp = tv_field::Fp;
pub type TvFp2 = tv_field::Fp2;

/// SplitMix64 (Steele, Lea and Flood, 2014): reproducible inputs, nothing secret.
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    pub fn coin(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    pub fn fp(&mut self) -> Fp {
        Fp::from_u64_reduced(self.next_u64())
    }

    pub fn fp2(&mut self) -> Fp2 {
        let c0 = self.fp();
        Fp2::new(c0, self.fp())
    }

    pub fn fps(&mut self, n: usize) -> Vec<Fp> {
        (0..n).map(|_| self.fp()).collect()
    }

    pub fn fp2s(&mut self, n: usize) -> Vec<Fp2> {
        (0..n).map(|_| self.fp2()).collect()
    }

    pub fn bytes<const N: usize>(&mut self) -> [u8; N] {
        core::array::from_fn(|_| self.next_u64() as u8)
    }
}

// ----- Conversions to the verifier's field ---------------------------------------------------

pub fn tv_fp(x: Fp) -> TvFp {
    TvFp::from_canonical(x.to_canonical_u64()).expect("a canonical element")
}

pub fn tv_fp2(x: Fp2) -> TvFp2 {
    let [c0, c1] = x.coeffs();
    TvFp2::new(tv_fp(c0), tv_fp(c1))
}

pub fn tv_fp2s(xs: &[Fp2]) -> Vec<TvFp2> {
    xs.iter().copied().map(tv_fp2).collect()
}

pub fn tv_proof(proof: &SumcheckProof<Fp2>) -> tv::SumcheckProof {
    tv::SumcheckProof {
        rounds: proof.rounds.iter().map(|round| tv_fp2s(round)).collect(),
        final_evals: tv_fp2s(&proof.final_evals),
    }
}

pub fn tv_prefix(prefix: Prefix) -> tv::Prefix {
    match prefix {
        Prefix::Stage(k) => tv::Prefix::Stage(k),
        Prefix::Tree(level) => tv::Prefix::Tree(level),
    }
}

// ----- Sessions --------------------------------------------------------------------------------

/// A session's context, built for each side from the same fields.
pub struct Session {
    pub mode: Mode,
    pub params: &'static str,
    pub statement_digest: [u8; 32],
    pub session_id: [u8; 16],
}

pub const PROTOCOL: &str = "tessera-protocol/1";

impl Session {
    pub fn random(rng: &mut SplitMix64) -> Self {
        Self {
            mode: if rng.coin() {
                Mode::Live
            } else {
                Mode::NonInteractive
            },
            params: "test-small/1",
            statement_digest: rng.bytes(),
            session_id: rng.bytes(),
        }
    }

    pub fn context(&self) -> SessionContext {
        SessionContext {
            protocol: Ascii::new(PROTOCOL).expect("ASCII"),
            mode: self.mode,
            params: ParamSetId(Ascii::new(self.params).expect("ASCII")),
            statement_digest: self.statement_digest,
            session_id: self.session_id,
        }
    }

    pub fn transcript(&self) -> Transcript {
        Transcript::new(&self.context())
    }

    pub fn tv_transcript(&self) -> tv_hash::transcript::Transcript {
        use tv_hash::transcript as t;
        let context = t::SessionContext {
            protocol: t::Ascii::new(PROTOCOL).expect("ASCII"),
            mode: match self.mode {
                Mode::Live => t::Mode::Live,
                Mode::NonInteractive => t::Mode::NonInteractive,
            },
            params: t::ParamSetId(t::Ascii::new(self.params).expect("ASCII")),
            statement_digest: self.statement_digest,
            session_id: self.session_id,
        };
        t::Transcript::new(&context)
    }
}

// ----- Test instances ------------------------------------------------------------------------

/// A table's entries.
#[derive(Clone)]
pub enum Values {
    Base(Vec<Fp>),
    Ext(Vec<Fp2>),
}

impl Values {
    pub fn len(&self) -> usize {
        match self {
            Self::Base(v) => v.len(),
            Self::Ext(v) => v.len(),
        }
    }

    /// Entry `x` in the verifier's field.
    pub fn tv_at(&self, x: usize) -> TvFp2 {
        match self {
            Self::Base(v) => TvFp2::from(tv_fp(v[x])),
            Self::Ext(v) => tv_fp2(v[x]),
        }
    }
}

/// Whether a table's evaluation is sent, and how the verifier evaluates it when it is not.
#[derive(Clone)]
pub enum Kind {
    /// An input polynomial: its evaluation is a final evaluation.
    Sent,
    /// A public factor the verifier evaluates by summing the table against `eq` at each vertex.
    Public,
    /// The public factor `eq(r, ·)`, which the verifier evaluates with its own `eq`.
    Eq(Vec<Fp2>),
}

#[derive(Clone)]
pub struct TestTable {
    pub values: Values,
    pub kind: Kind,
}

impl TestTable {
    pub fn base(values: Vec<Fp>) -> Self {
        Self {
            values: Values::Base(values),
            kind: Kind::Sent,
        }
    }

    pub fn ext(values: Vec<Fp2>) -> Self {
        Self {
            values: Values::Ext(values),
            kind: Kind::Sent,
        }
    }

    /// A random table on `vars` variables, over the prime field when `base`.
    pub fn random(rng: &mut SplitMix64, vars: usize, base: bool) -> Self {
        if base {
            Self::base(rng.fps(1 << vars))
        } else {
            Self::ext(rng.fp2s(1 << vars))
        }
    }

    /// `eq(r, ·)` at a random point of `vars` coordinates, as a public factor.
    pub fn eq(rng: &mut SplitMix64, vars: usize) -> Self {
        let r = rng.fp2s(vars);
        Self {
            values: Values::Ext(tessera_poly::eq_table(&r).expect("a small table")),
            kind: Kind::Eq(r),
        }
    }

    /// The same table as a public factor the verifier evaluates by brute force.
    pub fn public(self) -> Self {
        Self {
            kind: Kind::Public,
            ..self
        }
    }

    pub fn table(&self) -> Table<'_, Fp2> {
        let table = match &self.values {
            Values::Base(v) => Table::base(v),
            Values::Ext(v) => Table::ext(v),
        };
        match self.kind {
            Kind::Sent => table,
            Kind::Public | Kind::Eq(_) => table.public(),
        }
    }

    pub fn is_sent(&self) -> bool {
        matches!(self.kind, Kind::Sent)
    }

    /// The verifier's value of a public factor at `point`.
    fn tv_public(&self, point: &[TvFp2]) -> Option<TvFp2> {
        match &self.kind {
            Kind::Sent => None,
            Kind::Eq(r) => tv::eq(&tv_fp2s(r), point),
            Kind::Public => Some(tv_mle(&self.values, point)),
        }
    }
}

/// `Σ_x T[x]·eq(point, x)` by enumeration of the hypercube, in the verifier's field.
pub fn tv_mle(values: &Values, point: &[TvFp2]) -> TvFp2 {
    assert_eq!(values.len(), 1 << point.len());
    (0..values.len()).fold(TvFp2::ZERO, |sum, x| {
        let weight = point.iter().enumerate().fold(TvFp2::ONE, |w, (i, &r)| {
            w * if (x >> i) & 1 == 1 { r } else { TvFp2::ONE - r }
        });
        sum + values.tv_at(x) * weight
    })
}

/// An instance: the product of its tables.
#[derive(Clone)]
pub struct TestInstance {
    pub tables: Vec<TestTable>,
}

impl TestInstance {
    pub fn new(tables: Vec<TestTable>) -> Self {
        Self { tables }
    }

    /// `k` random tables on `vars` variables, each over the prime field when `base(table)`.
    pub fn random(
        rng: &mut SplitMix64,
        vars: usize,
        k: usize,
        mut base: impl FnMut(usize) -> bool,
    ) -> Self {
        Self::new(
            (0..k)
                .map(|t| TestTable::random(rng, vars, base(t)))
                .collect(),
        )
    }

    pub fn instance(&self) -> Instance<'_, Fp2> {
        Instance::new(self.tables.iter().map(TestTable::table).collect())
    }

    pub fn vars(&self) -> usize {
        self.tables[0].values.len().trailing_zeros() as usize
    }

    /// The number of final evaluations it sends.
    pub fn sent(&self) -> usize {
        self.tables.iter().filter(|t| t.is_sent()).count()
    }

    /// `c_i = Σ_x Π_k T_k[x]`, by enumeration, in the verifier's field.
    pub fn claim(&self) -> TvFp2 {
        (0..1 << self.vars()).fold(TvFp2::ZERO, |sum, x| {
            sum + self
                .tables
                .iter()
                .fold(TvFp2::ONE, |p, t| p * t.values.tv_at(x))
        })
    }

    /// The relation at `point` from the sent `evals`, public factors evaluated by the verifier.
    pub fn relation(&self, point: &[TvFp2], evals: &[TvFp2]) -> Option<TvFp2> {
        let mut evals = evals.iter();
        let mut value = TvFp2::ONE;
        for table in &self.tables {
            value *= match table.is_sent() {
                true => *evals.next()?,
                false => table.tv_public(point)?,
            };
        }
        evals.next().is_none().then_some(value)
    }

    pub fn tv_instance(&self) -> tv::Instance {
        tv::Instance {
            claim: self.claim(),
            vars: self.vars(),
            evals: self.sent(),
        }
    }
}

/// The verifier's instances of a batch, claims computed once.
pub fn tv_instances(instances: &[TestInstance]) -> Vec<tv::Instance> {
    instances.iter().map(TestInstance::tv_instance).collect()
}

/// Runs `tv_core::sumcheck::verify_batch` on `proof` with the product relation, on a fresh
/// verifier transcript of `session`, and returns its result and the transcript's final state.
pub fn verify(
    session: &Session,
    prefix: Prefix,
    degree: usize,
    instances: &[TestInstance],
    claims: &[tv::Instance],
    proof: &tv::SumcheckProof,
) -> (Result<tv::Batch, tv::SumcheckError>, [u8; 32]) {
    let mut transcript = session.tv_transcript();
    let result = tv::verify_batch(
        &mut transcript,
        tv_prefix(prefix),
        degree,
        claims,
        proof,
        |i, point, evals| instances.get(i)?.relation(point, evals),
    );
    (result, *transcript.state())
}

/// Proves the batch on a fresh transcript of `session`.
pub fn prove(
    session: &Session,
    prefix: Prefix,
    degree: usize,
    instances: &[TestInstance],
) -> (ProvenBatch<Fp2>, [u8; 32]) {
    let borrowed: Vec<Instance<'_, Fp2>> = instances.iter().map(TestInstance::instance).collect();
    let mut transcript = session.transcript();
    let proven =
        prove_batch(&mut transcript, prefix, degree, &borrowed).expect("a well-formed batch");
    (proven, transcript.state())
}

/// Proves the batch and checks the proof with the independent verifier: it accepts, its
/// challenges and points are the prover's, its transcript ends in the prover's state, and each
/// instance's evaluations are the ones the verifier's relation read.
pub fn prove_and_verify(
    session: &Session,
    prefix: Prefix,
    degree: usize,
    instances: &[TestInstance],
) -> ProvenBatch<Fp2> {
    let (proven, state) = prove(session, prefix, degree, instances);
    let proof = proven.proof();
    let n_max = instances.iter().map(TestInstance::vars).max().unwrap_or(0);
    assert_eq!(proof.rounds.len(), n_max, "one message per round");
    assert!(
        proof.rounds.iter().all(|round| round.len() == degree),
        "d values per round"
    );
    let claims = tv_instances(instances);
    let (result, tv_state) = verify(
        session,
        prefix,
        degree,
        instances,
        &claims,
        &tv_proof(proof),
    );
    let batch = result.expect("the verifier accepts an honest proof");
    assert_eq!(batch.gamma, tv_fp2(*proven.gamma()), "γ");
    assert_eq!(batch.point, tv_fp2s(proven.point()), "the point");
    assert_eq!(tv_state, state, "the transcripts end in the same state");
    assert_eq!(proven.instances(), instances.len());
    let mut sent = 0;
    for (i, instance) in instances.iter().enumerate() {
        let point = proven.instance_point(i).expect("an instance");
        assert_eq!(
            Some(tv_fp2s(point).as_slice()),
            batch.point_of(instance.vars()),
            "instance {i}'s point"
        );
        let evals = proven.instance_evals(i).expect("an instance");
        assert_eq!(evals, &proof.final_evals[sent..sent + instance.sent()]);
        sent += instance.sent();
    }
    assert_eq!(sent, proof.final_evals.len());
    assert_eq!(proven.instance_point(instances.len()), None);
    assert_eq!(proven.instance_evals(instances.len()), None);
    proven
}
