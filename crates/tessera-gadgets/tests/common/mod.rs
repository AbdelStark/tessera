// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared test support, written from the RFCs and sharing no code with the crate under test:
//!
//! - seeded inputs, and gadgets the compiler selects for one `Requant` node (RFC-0006 'Gadget
//!   selection');
//! - RFC-0005's `rq`, the value every requantization gadget must return;
//! - [`rejected`]: which lookups of an assignment are no rows of their tables, over any prime
//!   field. LogUp accepts a set of lookups exactly when every tuple is a row, except with
//!   probability `(N + T)/|Fp2|` (RFC-0008 'Relation'), so this is the check a gadget's lemma
//!   is about;
//! - a LogUp verifier driver around the verifier's independent checks (DEP-2):
//!   `tv_core::logup::check_roots`, `tv_core::tree::verify_level` at each level and
//!   `tv_core::logup::check_leaves`, with each table evaluated by `tv-tables` from the prover's
//!   specification bytes;
//! - multilinear extensions in `tv-field`, for the claims the verifier leaves.

#![allow(dead_code)] // each test binary uses part of it

use tessera_codec::{Ascii, encode};
use tessera_compiler::{Gadget, analyze_fragment, select};
use tessera_field::{Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_gadgets::{Constraints, Lookups, TableColumns, Term};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_ir::{
    CommitmentClass, GadgetPolicy, GraphBuilder, IntRange, IntType, NodeKind, Op, TensorDecl,
};
use tessera_logup::{LogUpInstance, ProvenLogUp, TreeId, prove_logup};
use tessera_poly::PolyField;
use tessera_tables::{GeneratedTable, Table, TableKind, TableSpec};
use tv_codec::{DecodeLimits, decode};
use tv_core::logup::{self as tvl, LookupError};
use tv_core::sumcheck as tv;
use tv_core::tree as tvt;

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

    pub fn bytes<const N: usize>(&mut self) -> [u8; N] {
        core::array::from_fn(|_| self.next_u64() as u8)
    }

    /// A key of `b` signed bits, its magnitude's width spread over `0..b` so that small and
    /// large keys, and both signs, come up.
    pub fn key(&mut self, b: u32) -> i64 {
        let bits = self.below(u64::from(b)) as u32;
        let magnitude = (self.next_u64() >> 2) as i64 & ((1i64 << bits) - 1);
        match self.next_u64() & 1 {
            0 => magnitude,
            _ => -magnitude - 1,
        }
    }
}

/// `count` keys of `b` signed bits: both ends of `[−2^{b−1}, 2^{b−1})`, the values around 0, then
/// seeded keys.
pub fn keys(rng: &mut SplitMix64, b: u32, count: usize) -> Vec<i64> {
    let half = 1i64 << (b - 1);
    let mut keys = vec![-half, -half + 1, half - 2, half - 1, -1, 0, 1];
    keys.truncate(count);
    while keys.len() < count {
        keys.push(rng.key(b));
    }
    keys
}

/// `count` classes below `n`.
pub fn classes(rng: &mut SplitMix64, n: usize, count: usize) -> Vec<u8> {
    (0..count).map(|_| rng.below(n as u64) as u8).collect()
}

// ----- Selection -------------------------------------------------------------------------------

/// The compiler's gadget for one `Requant` of keys whose signed width is exactly `b`, with
/// shift classes `shifts`, saturated to `[lo, hi]`, under `B_max = b_max`, with the tables it
/// names.
pub fn select_requant(
    b: u32,
    shifts: &[u8],
    lo: i64,
    hi: i64,
    b_max: u8,
) -> (Gadget, Vec<TableSpec>) {
    let decl = |range: IntRange, kind| TensorDecl {
        shape: vec![1],
        ty: IntType {
            bits: 64,
            signed: true,
        },
        range,
        kind,
    };
    let advice = NodeKind::Committed {
        commitment: CommitmentClass::Advice,
    };
    let half = 1i128 << (b - 1);
    let mut g = GraphBuilder::new();
    let key = g
        .input(decl(
            IntRange {
                lo: -half,
                hi: half - 1,
            },
            advice,
        ))
        .expect("a key");
    let mut inputs = vec![key];
    if shifts.len() > 1 {
        let weight = NodeKind::Committed {
            commitment: CommitmentClass::Weight,
        };
        let last = shifts.len() as i128 - 1;
        inputs.push(
            g.input(decl(IntRange { lo: 0, hi: last }, weight))
                .expect("classes"),
        );
    }
    let shift = g.shift_classes(shifts.to_vec()).expect("shifts");
    g.node(
        Op::Requant { shift, lo, hi },
        &inputs,
        vec![decl(IntRange { lo: 0, hi: 0 }, advice)],
    )
    .expect("a Requant node");
    let graph = g.finish();
    // A fragment: the committed key stands for a value a larger graph computes and proves
    // (RFC-0006 'Committed graph inputs').
    let ranges = analyze_fragment(&graph, &[]).expect("the graph analyzes");
    let policy = GadgetPolicy {
        max_log2_rows: b_max,
        ..GadgetPolicy::default()
    };
    let selection = select(&graph, &ranges, &[], &policy).expect("a gadget is selected");
    (selection.gadgets[0].1, selection.tables)
}

/// RFC-0005's `rq(z, r, lo, hi) = clamp(⌊(z + 2^{r−1}) / 2^r⌋, lo, hi)`, through 128 bits.
pub fn rq(z: i64, r: u8, lo: i64, hi: i64) -> i64 {
    let shifted = (i128::from(z) + (1i128 << (r - 1))) >> r;
    shifted.clamp(i128::from(lo), i128::from(hi)) as i64
}

// ----- What LogUp accepts ----------------------------------------------------------------------

/// Whether `(key, value)` is a row of `table` over `F`: some `j < T` with `j ≡ key` and the
/// row's value `≡ value`, the table's value column mapped to `F` in the centered representation.
/// A `Range` table has no values.
pub fn in_table<F: PrimeField>(table: &GeneratedTable, key: F, value: Option<F>) -> bool {
    let has_values = !matches!(table.spec().kind, TableKind::Range { .. });
    assert_eq!(
        value.is_some(),
        has_values,
        "a lookup has values exactly when its table does"
    );
    let rows = table.row_count();
    let mut j = Some(key.to_canonical_u64());
    while let Some(row) = j.filter(|&row| row < rows) {
        let (_, v) = table.row(row).expect("a row below T");
        let v = F::from_i64(i64::try_from(v.to_centered_i128()).expect("a true integer"));
        if value.is_none_or(|value| value == v) {
            return true;
        }
        j = row.checked_add(F::MODULUS);
    }
    false
}

/// The lookups of `constraints` whose tuple, at the assignment `value`, is no row of its table,
/// in order. Empty exactly when LogUp would accept every lookup.
pub fn rejected<F: PrimeField + PolyField<Base = F>>(
    constraints: &Constraints<F>,
    tables: &[&GeneratedTable],
    mut value: impl FnMut(Term) -> F,
) -> Vec<usize> {
    constraints
        .lookups
        .iter()
        .enumerate()
        .filter(|(_, lookup)| {
            let table = tables
                .iter()
                .find(|t| t.spec().id == lookup.table)
                .expect("a table of the gadget");
            let key = lookup.key.evaluate(&mut value);
            let v = lookup.value.as_ref().map(|form| form.evaluate(&mut value));
            !in_table(table, key, v)
        })
        .map(|(i, _)| i)
        .collect()
}

/// The first lookup of `constraints` whose tuple is no row, or `None`: [`rejected`]'s first
/// entry, without evaluating the lookups after it.
pub fn first_rejected<F: PrimeField + PolyField<Base = F>>(
    constraints: &Constraints<F>,
    tables: &[&GeneratedTable],
    mut value: impl FnMut(Term) -> F,
) -> Option<usize> {
    constraints.lookups.iter().position(|lookup| {
        let table = tables
            .iter()
            .find(|t| t.spec().id == lookup.table)
            .expect("a table of the gadget");
        let key = lookup.key.evaluate(&mut value);
        let v = lookup.value.as_ref().map(|form| form.evaluate(&mut value));
        !in_table(table, key, v)
    })
}

// ----- Conversions to the verifier's types -----------------------------------------------------

pub fn tv_fp(x: Fp) -> TvFp {
    TvFp::from_canonical(x.to_canonical_u64()).expect("a canonical element")
}

pub fn tv_fp2(x: Fp2) -> TvFp2 {
    let [c0, c1] = x.coeffs();
    TvFp2::new(tv_fp(c0), tv_fp(c1))
}

pub fn fp2(x: TvFp2) -> Fp2 {
    let base = |c: TvFp| Fp::from_canonical_u64(c.to_canonical()).expect("canonical");
    Fp2::new(base(x.c0), base(x.c1))
}

pub fn tv_fp2s(xs: &[Fp2]) -> Vec<TvFp2> {
    xs.iter().copied().map(tv_fp2).collect()
}

/// The verifier's table for a prover specification: its TCE bytes, decoded by `tv-codec` and
/// regenerated by `tv-tables`.
pub fn tv_table(spec: &TableSpec) -> tv_tables::Table {
    let bytes = encode(spec);
    let decoded = decode(&bytes, &DecodeLimits::MESSAGE, tv_tables::TableSpec::decode)
        .unwrap_or_else(|e| panic!("{spec:?}: {e}"));
    tv_tables::Table::from_spec(&decoded).unwrap_or_else(|e| panic!("{spec:?}: {e}"))
}

// ----- Extensions in the verifier's field ------------------------------------------------------

/// The `2^n` weights `eq(point, x)`, built by doubling.
pub fn eq_weights(point: &[TvFp2]) -> Vec<TvFp2> {
    let mut weights = vec![TvFp2::ONE];
    for &r in point {
        let high: Vec<TvFp2> = weights.iter().map(|&e| e * r).collect();
        for (e, h) in weights.iter_mut().zip(&high) {
            *e -= *h;
        }
        weights.extend(high);
    }
    weights
}

/// `f̃(point)` of `2^{|point|}` integers, each mapped to the field in the centered representation.
pub fn mle(values: &[i64], point: &[TvFp2]) -> TvFp2 {
    assert_eq!(
        values.len(),
        1 << point.len(),
        "2^n entries at n coordinates"
    );
    values
        .iter()
        .zip(eq_weights(point))
        .fold(TvFp2::ZERO, |sum, (&v, e)| {
            sum + e * TvFp2::from(tv_fp(Fp::from_i64(v)))
        })
}

/// `f̃(point)` of `2^{|point|}` field elements.
pub fn mle_fp(values: &[Fp], point: &[TvFp2]) -> TvFp2 {
    assert_eq!(
        values.len(),
        1 << point.len(),
        "2^n entries at n coordinates"
    );
    values
        .iter()
        .zip(eq_weights(point))
        .fold(TvFp2::ZERO, |sum, (&v, e)| sum + e * TvFp2::from(tv_fp(v)))
}

/// `⌈log2 n⌉`, 0 for `n ≤ 1`.
pub fn height(n: usize) -> usize {
    n.next_power_of_two().trailing_zeros() as usize
}

// ----- The LogUp verifier driver ---------------------------------------------------------------

pub const PROTOCOL: &str = "tessera-protocol/1";

/// A session's context, built for each side from the same fields.
pub struct Session {
    pub statement_digest: [u8; 32],
    pub session_id: [u8; 16],
}

impl Session {
    pub fn random(rng: &mut SplitMix64) -> Self {
        Self {
            statement_digest: rng.bytes(),
            session_id: rng.bytes(),
        }
    }

    pub fn transcript(&self) -> Transcript {
        Transcript::new(&SessionContext {
            protocol: Ascii::new(PROTOCOL).expect("ASCII"),
            mode: Mode::NonInteractive,
            params: ParamSetId(Ascii::new("test-small/1").expect("ASCII")),
            statement_digest: self.statement_digest,
            session_id: self.session_id,
        })
    }

    pub fn tv_transcript(&self) -> tv_hash::transcript::Transcript {
        use tv_hash::transcript as t;
        t::Transcript::new(&t::SessionContext {
            protocol: t::Ascii::new(PROTOCOL).expect("ASCII"),
            mode: t::Mode::NonInteractive,
            params: t::ParamSetId(t::Ascii::new("test-small/1").expect("ASCII")),
            statement_digest: self.statement_digest,
            session_id: self.session_id,
        })
    }
}

/// What the verifier knows of an instance before the proof: its shape and its table, from the
/// claim list.
pub struct Shape {
    pub lookups: usize,
    pub table: tv_tables::Table,
    pub has_values: bool,
}

impl Shape {
    pub fn of(instance: &LogUpInstance<'_, Fp>, spec: &TableSpec) -> Self {
        Self {
            lookups: instance.lookups.len(),
            table: tv_table(spec),
            has_values: instance.table.has_values(),
        }
    }
}

/// The first check that failed.
#[derive(Debug, PartialEq, Eq)]
pub enum Rejected {
    /// `check_roots` or `check_leaves`.
    Lookup(LookupError),
    /// The tree level `level`.
    Level(usize, tv::SumcheckError),
}

/// The messages a verifier reads, in its types.
pub struct Messages {
    pub roots: Vec<TvFp2>,
    pub levels: Vec<tv::SumcheckProof>,
    pub values: Vec<TvFp2>,
}

impl Messages {
    pub fn of(proven: &ProvenLogUp<Fp2>) -> Self {
        Self {
            roots: tv_fp2s(proven.roots()),
            levels: proven
                .levels()
                .iter()
                .map(|level| tv::SumcheckProof {
                    rounds: level.proof().rounds.iter().map(|r| tv_fp2s(r)).collect(),
                    final_evals: tv_fp2s(&level.proof().final_evals),
                })
                .collect(),
            values: tv_fp2s(proven.values()),
        }
    }
}

/// Proves the instances, alone in the leaf stage and scheduled in pairs, on a fresh transcript
/// of `session`.
pub fn prove(session: &Session, instances: &[LogUpInstance<'_, Fp>]) -> ProvenLogUp<Fp2> {
    let mut transcript = session.transcript();
    prove_logup(
        &mut transcript,
        instances,
        &[],
        &TreeId::paired(instances.len()),
    )
    .expect("well-formed instances")
}

/// Checks `messages` as a verifier that knows the instances' shapes and tables, on a fresh
/// `tv-hash` transcript of `session`: the roots, every level, then the leaf checks against
/// `t̃(ρ_t)` from `tv-tables`. The schedule pairs each instance's lookup tree with its table
/// tree, as [`prove`] does.
pub fn verify(
    session: &Session,
    shapes: &[Shape],
    messages: &Messages,
) -> Result<tvl::LeafClaims, Rejected> {
    let mut transcript = session.tv_transcript();
    let roots = tvl::check_roots(&mut transcript, shapes.len(), &messages.roots)
        .map_err(Rejected::Lookup)?;
    let heights: Vec<usize> = shapes
        .iter()
        .flat_map(|s| [height(s.lookups), usize::from(s.table.log2_rows())])
        .collect();
    let tallest = heights.iter().copied().max().unwrap_or(0);
    assert_eq!(messages.levels.len(), tallest, "one proof per level");
    let mut claims: Vec<tvt::LevelClaim> = roots
        .instances
        .iter()
        .flat_map(|r| {
            [
                tvt::LevelClaim::Fraction(r.lookup.clone()),
                tvt::LevelClaim::Fraction(r.table.clone()),
            ]
        })
        .collect();
    for (level, proof) in messages.levels.iter().enumerate() {
        let present: Vec<usize> = (0..heights.len())
            .filter(|&p| tallest - heights[p] <= level)
            .collect();
        let entering: Vec<tvt::LevelClaim> = present.iter().map(|&p| claims[p].clone()).collect();
        let checked = tvt::verify_level(&mut transcript, level as u32, &entering, proof)
            .map_err(|error| Rejected::Level(level, error))?;
        for (&p, next) in present.iter().zip(&checked.claims) {
            claims[p] = next.clone();
        }
    }
    let fraction = |p: usize| match &claims[p] {
        tvt::LevelClaim::Fraction(claim) => claim.clone(),
        tvt::LevelClaim::Product(_) => panic!("a fraction tree's claims"),
    };
    let leaves: Vec<tvl::Leaves> = shapes
        .iter()
        .enumerate()
        .map(|(k, shape)| tvl::Leaves {
            lookup: fraction(2 * k),
            table: fraction(2 * k + 1),
            has_values: shape.has_values,
        })
        .collect();
    let beta = roots.challenges.beta;
    tvl::check_leaves(
        &mut transcript,
        &roots.challenges,
        &leaves,
        &messages.values,
        |i, rho| shapes.get(i)?.table.eval_mle(rho, beta).ok(),
    )
    .map_err(Rejected::Lookup)
}

/// The columns of every table `lookups` reads, from the gadget's `tables`.
pub fn table_columns(tables: &[&GeneratedTable]) -> Vec<TableColumns<Fp>> {
    tables
        .iter()
        .map(|table| TableColumns::new(table).expect("a table of fewer rows than P"))
        .collect()
}

/// The specification of the table `group` reads.
pub fn spec_of<'a>(
    tables: &[&'a GeneratedTable],
    lookups: &Lookups<Fp>,
    k: usize,
) -> &'a TableSpec {
    let id = lookups.groups()[k].table();
    tables
        .iter()
        .find(|t| t.spec().id == id)
        .expect("a table of the gadget")
        .spec()
}
