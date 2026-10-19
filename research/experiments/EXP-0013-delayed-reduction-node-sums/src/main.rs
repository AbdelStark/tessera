// SPDX-License-Identifier: Apache-2.0 OR MIT

//! EXP-0013: safe delayed reduction versus the EXP-0008 split-eq arithmetic.
//! Const arm 0 preserves that arithmetic, arm 1 scales first, arm 2 reduces the relation first.
//! Inputs are synthetic; every arm reproduces production messages and independent acceptance.
//! The only production-API adaptation is folding synthetic Vec tables locally instead of SecretBuf.

#![forbid(unsafe_code)]
#![allow(
    clippy::print_stdout,
    clippy::cast_precision_loss,
    clippy::too_many_lines
)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use tessera_codec::Ascii;
use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_hash::{Digest, Domain, Mode, ParamSetId, SessionContext, Transcript, hash};
use tessera_poly::{PolyField, eq_table};
use tessera_sumcheck::{
    Claim, Forest, Instance, Prefix, ProvenForests, SumcheckProof, SumcheckTranscript, Table,
    prove_batch, prove_forests,
};
use tv_core::sumcheck as tv;
use tv_core::tree::{TreeClaim, verify_product_level};
use tv_hash::transcript as tvt;

type Fp = Goldilocks;
type Fp2 = GoldilocksExt2;
type TvFp = tv_field::Fp;
type TvFp2 = tv_field::Fp2;

/// Leaves: the seed of `crates/tessera-sumcheck/benches/throughput.rs` `forests` and EXP-0006.
const LEAF_SEED: u64 = 0x7E55_E7A0_0058_0001;
/// The level kernel's point.
const ENGINE_SEED: u64 = 0x7E55_E7A0_0058_0002;
/// The fraction-tree proxy's keys, values, `α`, `β`, point and generic tables.
const FRACTION_SEED: u64 = 0x7E55_E7A0_0058_0003;
/// The small-size equivalence sweep, the two-forest batch and the degenerate cases.
const CHECK_SEED: u64 = 0x7E55_E7A0_0058_0005;

const LEVEL_DEGREE: usize = 3;

const WARMUPS: usize = 3;
const ITERATIONS: usize = 10;
const CV_LIMIT: f64 = 0.03;
/// Every arm repeats its kernel until one iteration lasts at least this long.
const MIN_ITERATION: Duration = Duration::from_millis(20);

/// SplitMix64: reproducible inputs.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn fp(&mut self) -> Fp {
        Fp::from_u64_reduced(self.next_u64())
    }

    fn fp2(&mut self) -> Fp2 {
        Fp2::new(self.fp(), self.fp())
    }

    fn fps(&mut self, n: usize) -> Vec<Fp> {
        (0..n).map(|_| self.fp()).collect()
    }

    fn fp2s(&mut self, n: usize) -> Vec<Fp2> {
        (0..n).map(|_| self.fp2()).collect()
    }
}

// ---------------------------------------------------------------------------------------------
// Sessions: the bench's context on both sides
// ---------------------------------------------------------------------------------------------

const PROTOCOL: &str = "tessera-protocol/1";
const PARAMS: &str = "test-small/1";
const STATEMENT_DIGEST: [u8; 32] = [0x5a; 32];
const SESSION_ID: [u8; 16] = [7; 16];

fn context() -> SessionContext {
    SessionContext {
        protocol: Ascii::new(PROTOCOL).expect("ASCII"),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new(PARAMS).expect("ASCII")),
        statement_digest: STATEMENT_DIGEST,
        session_id: SESSION_ID,
    }
}

fn tv_transcript() -> tvt::Transcript {
    let context = tvt::SessionContext {
        protocol: tvt::Ascii::new(PROTOCOL).expect("ASCII"),
        mode: tvt::Mode::NonInteractive,
        params: tvt::ParamSetId(tvt::Ascii::new(PARAMS).expect("ASCII")),
        statement_digest: STATEMENT_DIGEST,
        session_id: SESSION_ID,
    };
    tvt::Transcript::new(&context)
}

// ---------------------------------------------------------------------------------------------
// Conversions to the verifier's field, and digests
// ---------------------------------------------------------------------------------------------

fn tv_fp(x: Fp) -> TvFp {
    TvFp::from_canonical(x.to_canonical_u64()).expect("a canonical element")
}

fn tv_fp2(x: Fp2) -> TvFp2 {
    let [c0, c1] = x.coeffs();
    TvFp2::new(tv_fp(c0), tv_fp(c1))
}

fn tv_fp2s(xs: &[Fp2]) -> Vec<TvFp2> {
    xs.iter().copied().map(tv_fp2).collect()
}

fn tv_proof(proof: &SumcheckProof<Fp2>) -> tv::SumcheckProof {
    tv::SumcheckProof {
        rounds: proof.rounds.iter().map(|round| tv_fp2s(round)).collect(),
        final_evals: tv_fp2s(&proof.final_evals),
    }
}

/// The canonical little-endian bytes of a proof's messages: each round's length and values, then
/// the final evaluations' length and values, each `Fp2` as `c0 ‖ c1`.
fn proof_bytes(proof: &SumcheckProof<Fp2>, out: &mut Vec<u8>) {
    fn push(out: &mut Vec<u8>, xs: &[Fp2]) {
        out.extend_from_slice(&(xs.len() as u64).to_le_bytes());
        for x in xs {
            for c in x.coeffs() {
                out.extend_from_slice(&c.to_canonical_u64().to_le_bytes());
            }
        }
    }
    out.extend_from_slice(&(proof.rounds.len() as u64).to_le_bytes());
    for round in &proof.rounds {
        push(out, round);
    }
    push(out, &proof.final_evals);
}

/// The BLAKE3 digest of every level's messages.
fn proofs_digest(proofs: &[SumcheckProof<Fp2>]) -> Digest {
    let mut bytes = Vec::new();
    for proof in proofs {
        proof_bytes(proof, &mut bytes);
    }
    hash(Domain::ColumnMerge, &bytes)
}

fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------------------------
// Measurement protocol (RFC-0020 'Measurement protocol')
// ---------------------------------------------------------------------------------------------

/// Seconds per iteration over the measured iterations.
#[derive(Clone, Copy, Debug)]
struct Stats {
    median: f64,
    min: f64,
    max: f64,
    cv: f64,
    /// The case was rerun because its first coefficient of variation exceeded 3%.
    repeated: bool,
    samples: [f64; 10],
}

fn sample(iteration: &mut impl FnMut() -> Duration) -> Stats {
    for _ in 0..WARMUPS {
        black_box(iteration());
    }
    let mut s: Vec<f64> = (0..ITERATIONS).map(|_| iteration().as_secs_f64()).collect();
    let samples: [f64; 10] = s.clone().try_into().expect("ten samples");
    s.sort_by(f64::total_cmp);
    let n = s.len();
    let median = (s[n / 2 - 1] + s[n / 2]) / 2.0;
    let mean = s.iter().sum::<f64>() / n as f64;
    let var = s.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    Stats {
        median,
        min: s[0],
        max: s[n - 1],
        cv: var.sqrt() / mean,
        repeated: false,
        samples,
    }
}

/// 3 warm-ups and 10 measured iterations; a coefficient of variation above 3% reruns the case
/// once, and the rerun is what is reported (flagged if it is still above 3%).
fn measure(mut iteration: impl FnMut() -> Duration) -> Stats {
    let first = sample(&mut iteration);
    if first.cv <= CV_LIMIT {
        return first;
    }
    let mut second = sample(&mut iteration);
    second.repeated = true;
    second
}

/// How many repetitions make one iteration last at least `MIN_ITERATION`.
fn reps_for(mut once: impl FnMut()) -> u32 {
    let start = Instant::now();
    once();
    let one = start.elapsed().max(Duration::from_nanos(1)).as_secs_f64();
    (2.0 * MIN_ITERATION.as_secs_f64() / one)
        .ceil()
        .clamp(1.0, 1e7) as u32
}

/// Times `once` under the protocol, repeating it per iteration until an iteration lasts at least
/// `MIN_ITERATION`; the statistics are per single execution.
fn timed(mut once: impl FnMut()) -> (Stats, u32) {
    let reps = reps_for(&mut once);
    let stats = measure(|| {
        let start = Instant::now();
        for _ in 0..reps {
            once();
        }
        let mut actual = reps;
        while start.elapsed() < MIN_ITERATION {
            once();
            actual += 1;
        }
        let elapsed = start.elapsed();
        assert!(elapsed >= MIN_ITERATION);
        println!(
            "RAW batch_seconds={} executions={}",
            elapsed.as_secs_f64(),
            actual
        );
        elapsed / actual
    });
    (stats, reps)
}

/// One reported case, for the JSON record.
struct Row {
    name: String,
    per: &'static str,
    elements: f64,
    reps: u32,
    stats: Stats,
}

/// Prints a case and returns its median in nanoseconds per unit.
fn report(
    rows: &mut Vec<Row>,
    name: &str,
    elements: f64,
    per: &'static str,
    stats: Stats,
    reps: u32,
) -> f64 {
    let ns = |t: f64| t * 1e9 / elements;
    let flag = if stats.cv > CV_LIMIT {
        "  FLAG: CV > 3% after a rerun"
    } else if stats.repeated {
        "  (rerun: first CV > 3%)"
    } else {
        ""
    };
    println!(
        "{name:<62} {:>9.3} ns/{per:<6} min {:>9.3}  max {:>9.3}  CV {:>5.2}%  ({:>9.3} ms x{reps}){flag}",
        ns(stats.median),
        ns(stats.min),
        ns(stats.max),
        stats.cv * 100.0,
        stats.median * 1e3
    );
    rows.push(Row {
        name: name.to_owned(),
        per,
        elements,
        reps,
        stats,
    });
    ns(stats.median)
}

// ---------------------------------------------------------------------------------------------
// The candidate: split-eq, Gruen-corrected batched sumcheck (IDEA-0021 Mechanism parts 1–3)
// ---------------------------------------------------------------------------------------------

/// `Σ_{x_hi} eq_hi[x_hi] · Σ_{x_lo} eq_lo[x_lo] · V_L(X, x)·V_R(X, x)` at the nodes `X = 0, 2`
/// (and `1` when `WITH_ONE`, the degenerate-challenge fallback), over the pairs
/// `x = x_hi·|eq_lo| + x_lo` of two tables over `Fp`: the first own round. On a pair `(a, b)` a
/// table is `a + X·(b − a)`, its value at node 2 formed as `kernel.rs` forms it, `b + (b − a)`.
/// The products are in `Fp`, the scaling by `eq_lo` an `Fp2`-by-`Fp` product.
fn product_base_sums<const WITH_ONE: bool, const ARM: u8>(
    even: &[Fp],
    odd: &[Fp],
    eq_lo: &[Fp2],
    eq_hi: &[Fp2],
) -> [Fp2; 3] {
    if ARM != 0 {
        return deferred_product_base_sums::<WITH_ONE, ARM>(even, odd, eq_lo, eq_hi);
    }
    let group = 2 * eq_lo.len();
    let mut sums = [Fp2::ZERO; 3];
    for ((h, e), o) in eq_hi
        .iter()
        .zip(even.chunks_exact(group))
        .zip(odd.chunks_exact(group))
    {
        let mut acc = [Fp2::ZERO; 3];
        for ((w, l), r) in eq_lo.iter().zip(e.chunks_exact(2)).zip(o.chunks_exact(2)) {
            let (dl, dr) = (l[1] - l[0], r[1] - r[0]);
            let p0 = l[0] * r[0];
            let p2 = (l[1] + dl) * (r[1] + dr);
            acc[0] += w.mul_by_base(p0);
            acc[2] += w.mul_by_base(p2);
            if WITH_ONE {
                acc[1] += w.mul_by_base(l[1] * r[1]);
            }
        }
        sums[0] += *h * acc[0];
        sums[2] += *h * acc[2];
        if WITH_ONE {
            sums[1] += *h * acc[1];
        }
    }
    sums
}

/// As [`product_base_sums`] over two bound tables over `Fp2`: the rounds after the first.
fn product_ext_sums<const WITH_ONE: bool, const ARM: u8>(
    left: &[Fp2],
    right: &[Fp2],
    eq_lo: &[Fp2],
    eq_hi: &[Fp2],
) -> [Fp2; 3] {
    if ARM != 0 {
        return deferred_product_ext_sums::<WITH_ONE, ARM>(left, right, eq_lo, eq_hi);
    }
    let group = 2 * eq_lo.len();
    let mut sums = [Fp2::ZERO; 3];
    for ((h, l), r) in eq_hi
        .iter()
        .zip(left.chunks_exact(group))
        .zip(right.chunks_exact(group))
    {
        let mut acc = [Fp2::ZERO; 3];
        for ((w, l), r) in eq_lo.iter().zip(l.chunks_exact(2)).zip(r.chunks_exact(2)) {
            let (dl, dr) = (l[1] - l[0], r[1] - r[0]);
            let p0 = l[0] * r[0];
            let p2 = (l[1] + dl) * (r[1] + dr);
            acc[0] += *w * p0;
            acc[2] += *w * p2;
            if WITH_ONE {
                acc[1] += *w * (l[1] * r[1]);
            }
        }
        sums[0] += *h * acc[0];
        sums[2] += *h * acc[2];
        if WITH_ONE {
            sums[1] += *h * acc[1];
        }
    }
    sums
}

/// The fraction proxy's leaf level (numerators 1) as one relation per pair:
/// `d_e(X)·(d_o(X) + γ) + γ²·d_o(X)`, the three instances `[eq, d_e, d_o]`, `[eq, d_e]`,
/// `[eq, d_o]` with their weights `1, γ, γ²` (IDEA-0018 mechanism 3).
fn fraction_leaf_sums<const WITH_ONE: bool, const ARM: u8>(
    d_e: &[Fp2],
    d_o: &[Fp2],
    eq_lo: &[Fp2],
    eq_hi: &[Fp2],
    gamma: Fp2,
    gamma2: Fp2,
) -> [Fp2; 3] {
    if ARM != 0 {
        return deferred_fraction_leaf_sums::<WITH_ONE, ARM>(d_e, d_o, eq_lo, eq_hi, gamma, gamma2);
    }
    let group = 2 * eq_lo.len();
    let mut sums = [Fp2::ZERO; 3];
    let relation = |e: Fp2, o: Fp2| e * (o + gamma) + gamma2 * o;
    for ((h, e), o) in eq_hi
        .iter()
        .zip(d_e.chunks_exact(group))
        .zip(d_o.chunks_exact(group))
    {
        let mut acc = [Fp2::ZERO; 3];
        for ((w, e), o) in eq_lo.iter().zip(e.chunks_exact(2)).zip(o.chunks_exact(2)) {
            let (de, dd) = (e[1] - e[0], o[1] - o[0]);
            acc[0] += *w * relation(e[0], o[0]);
            acc[2] += *w * relation(e[1] + de, o[1] + dd);
            if WITH_ONE {
                acc[1] += *w * relation(e[1], o[1]);
            }
        }
        sums[0] += *h * acc[0];
        sums[2] += *h * acc[2];
        if WITH_ONE {
            sums[1] += *h * acc[1];
        }
    }
    sums
}

/// The fraction proxy's generic level as one relation per pair:
/// `d_o(X)·(n_e(X) + γ²·d_e(X)) + γ·(n_o(X)·d_e(X))`, the three instances `[eq, n_e, d_o]`,
/// `[eq, n_o, d_e]`, `[eq, d_e, d_o]` with their weights `1, γ, γ²`.
fn fraction_generic_sums<const WITH_ONE: bool, const ARM: u8>(
    tables: [&[Fp2]; 4],
    eq_lo: &[Fp2],
    eq_hi: &[Fp2],
    gamma: Fp2,
    gamma2: Fp2,
) -> [Fp2; 3] {
    if ARM != 0 {
        return deferred_fraction_generic_sums::<WITH_ONE, ARM>(
            tables, eq_lo, eq_hi, gamma, gamma2,
        );
    }
    let [n_e, n_o, d_e, d_o] = tables;
    let group = 2 * eq_lo.len();
    let mut sums = [Fp2::ZERO; 3];
    let relation = |ne: Fp2, no: Fp2, de: Fp2, dd: Fp2| dd * (ne + gamma2 * de) + gamma * (no * de);
    for ((((h, ne), no), de), dd) in eq_hi
        .iter()
        .zip(n_e.chunks_exact(group))
        .zip(n_o.chunks_exact(group))
        .zip(d_e.chunks_exact(group))
        .zip(d_o.chunks_exact(group))
    {
        let mut acc = [Fp2::ZERO; 3];
        for ((((w, ne), no), de), dd) in eq_lo
            .iter()
            .zip(ne.chunks_exact(2))
            .zip(no.chunks_exact(2))
            .zip(de.chunks_exact(2))
            .zip(dd.chunks_exact(2))
        {
            acc[0] += *w * relation(ne[0], no[0], de[0], dd[0]);
            let at2 = |p: &[Fp2]| p[1] + (p[1] - p[0]);
            acc[2] += *w * relation(at2(ne), at2(no), at2(de), at2(dd));
            if WITH_ONE {
                acc[1] += *w * relation(ne[1], no[1], de[1], dd[1]);
            }
        }
        sums[0] += *h * acc[0];
        sums[2] += *h * acc[2];
        if WITH_ONE {
            sums[1] += *h * acc[1];
        }
    }
    sums
}

/// `out[t] = a + r·(b − a)` for each pair of a table over `Fp2`, into a new table: the engine's
/// `bind_ext`.
fn bind_ext(table: &[Fp2], r: Fp2) -> Vec<Fp2> {
    table
        .chunks_exact(2)
        .map(|pair| pair[0] + r * (pair[1] - pair[0]))
        .collect()
}

/// The value tables of one instance of the candidate, unbound then bound; the `eq` factor is
/// never a table here.
enum Relation<'a> {
    /// A product-tree level: `Ṽ(0 ‖ x)·Ṽ(1 ‖ x)`, the even and odd halves over `Fp`.
    Product { even: &'a [Fp], odd: &'a [Fp] },
    /// The fraction proxy's leaf level, numerators 1: `d_e`, `d_o` over `Fp2`.
    FractionLeaf { d_e: &'a [Fp2], d_o: &'a [Fp2] },
    /// The fraction proxy's generic level: `n_e, n_o, d_e, d_o` over `Fp2`.
    FractionGeneric { tables: [&'a [Fp2]; 4] },
}

impl Relation<'_> {
    /// The tables' length, `2^{n_i}`.
    fn len(&self) -> usize {
        match self {
            Self::Product { even, .. } => even.len(),
            Self::FractionLeaf { d_e, .. } => d_e.len(),
            Self::FractionGeneric { tables } => tables[0].len(),
        }
    }

    /// The instance's claim from the schedule's claims and `γ`: the one claim of a product level,
    /// or the three fraction instances' claims combined with `1, γ, γ²`.
    fn claim(&self, claims: &[Fp2], gamma: Fp2) -> Fp2 {
        match self {
            Self::Product { .. } => claims[0],
            Self::FractionLeaf { .. } | Self::FractionGeneric { .. } => {
                claims[0] + gamma * claims[1] + gamma * gamma * claims[2]
            }
        }
    }

    /// The tables' single entries, lifted: the bound state of an instance on no variable.
    fn singles(&self) -> Vec<Vec<Fp2>> {
        match self {
            Self::Product { even, odd } => vec![vec![Fp2::from(even[0])], vec![Fp2::from(odd[0])]],
            Self::FractionLeaf { d_e, d_o } => vec![vec![d_e[0]], vec![d_o[0]]],
            Self::FractionGeneric { tables } => tables.iter().map(|t| vec![t[0]]).collect(),
        }
    }

    /// The node sums of the first own round, over the unbound tables.
    fn first_sums<const ARM: u8>(
        &self,
        eq_lo: &[Fp2],
        eq_hi: &[Fp2],
        gamma: Fp2,
        with_one: bool,
    ) -> [Fp2; 3] {
        let g2 = gamma * gamma;
        match (self, with_one) {
            (Self::Product { even, odd }, false) => {
                product_base_sums::<false, ARM>(even, odd, eq_lo, eq_hi)
            }
            (Self::Product { even, odd }, true) => {
                product_base_sums::<true, ARM>(even, odd, eq_lo, eq_hi)
            }
            (Self::FractionLeaf { d_e, d_o }, false) => {
                fraction_leaf_sums::<false, ARM>(d_e, d_o, eq_lo, eq_hi, gamma, g2)
            }
            (Self::FractionLeaf { d_e, d_o }, true) => {
                fraction_leaf_sums::<true, ARM>(d_e, d_o, eq_lo, eq_hi, gamma, g2)
            }
            (Self::FractionGeneric { tables }, false) => {
                fraction_generic_sums::<false, ARM>(*tables, eq_lo, eq_hi, gamma, g2)
            }
            (Self::FractionGeneric { tables }, true) => {
                fraction_generic_sums::<true, ARM>(*tables, eq_lo, eq_hi, gamma, g2)
            }
        }
    }

    /// The node sums of a later own round, over the bound tables.
    fn bound_sums<const ARM: u8>(
        &self,
        bound: &[Vec<Fp2>],
        eq_lo: &[Fp2],
        eq_hi: &[Fp2],
        gamma: Fp2,
        with_one: bool,
    ) -> [Fp2; 3] {
        let g2 = gamma * gamma;
        match (self, with_one) {
            (Self::Product { .. }, false) => {
                product_ext_sums::<false, ARM>(&bound[0], &bound[1], eq_lo, eq_hi)
            }
            (Self::Product { .. }, true) => {
                product_ext_sums::<true, ARM>(&bound[0], &bound[1], eq_lo, eq_hi)
            }
            (Self::FractionLeaf { .. }, false) => {
                fraction_leaf_sums::<false, ARM>(&bound[0], &bound[1], eq_lo, eq_hi, gamma, g2)
            }
            (Self::FractionLeaf { .. }, true) => {
                fraction_leaf_sums::<true, ARM>(&bound[0], &bound[1], eq_lo, eq_hi, gamma, g2)
            }
            (Self::FractionGeneric { .. }, false) => fraction_generic_sums::<false, ARM>(
                [&bound[0], &bound[1], &bound[2], &bound[3]],
                eq_lo,
                eq_hi,
                gamma,
                g2,
            ),
            (Self::FractionGeneric { .. }, true) => fraction_generic_sums::<true, ARM>(
                [&bound[0], &bound[1], &bound[2], &bound[3]],
                eq_lo,
                eq_hi,
                gamma,
                g2,
            ),
        }
    }

    /// The first binding, from the borrowed tables into tables over `Fp2`: `bind_lowest` for
    /// tables over `Fp` (the engine's `bind_base`), `bind_ext` for tables over `Fp2`.
    fn bind_first(&self, r: Fp2) -> Vec<Vec<Fp2>> {
        match self {
            Self::Product { even, odd } => vec![bind_base(even, r), bind_base(odd, r)],
            Self::FractionLeaf { d_e, d_o } => vec![bind_ext(d_e, r), bind_ext(d_o, r)],
            Self::FractionGeneric { tables } => tables.iter().map(|t| bind_ext(t, r)).collect(),
        }
    }

    /// Appends the tables' values at the instance's point in the baseline's order: `u, w` for a
    /// product level; `d_e, d_o, d_e, d_o` for the fraction leaf level's three instances;
    /// `n_e, d_o, n_o, d_e, d_e, d_o` for the generic level's.
    fn final_evals(&self, bound: &[Vec<Fp2>], out: &mut Vec<Fp2>) {
        let at = |k: usize| bound[k][0];
        match self {
            Self::Product { .. } => out.extend([at(0), at(1)]),
            Self::FractionLeaf { .. } => out.extend([at(0), at(1), at(0), at(1)]),
            Self::FractionGeneric { .. } => {
                out.extend([at(0), at(3), at(1), at(2), at(2), at(3)]);
            }
        }
    }
}

/// One instance of the candidate's batch: its relation, the `eq` point `r_i` (lowest variable
/// first, one coordinate per variable) and the schedule's claims on it.
struct SplitInstance<'a> {
    relation: Relation<'a>,
    point: Vec<Fp2>,
    claims: Vec<Fp2>,
}

/// What the candidate's batch returns: the messages, `γ`, the point, and how often the
/// degenerate-challenge fallback ran.
struct SplitBatch {
    proof: SumcheckProof<Fp2>,
    gamma: Fp2,
    point: Vec<Fp2>,
    fallbacks: usize,
}

/// An instance while the rounds run.
struct SplitState {
    vars: usize,
    pad: usize,
    /// `γ^i`.
    weight: Fp2,
    /// The instance's running claim `claim_{i,j}`: its hypercube sum before its first own
    /// round, then `p_{i,j}(c_j)` after each own round.
    claim: Fp2,
    /// `s_i`: `eq` of the bound coordinates at their challenges.
    s: Fp2,
    /// The last own round's line `ℓ(0), ℓ(1)` and node values `v(0), v(1), v(2)`, for the claim
    /// update.
    last: Option<([Fp2; 2], [Fp2; 3])>,
    /// The bound value tables, empty until the first own round; an instance on no variable holds
    /// its tables' single entries.
    bound: Vec<Vec<Fp2>>,
}

/// Proves the batch as `prove_batch` does on the transcript, with the candidate's kernel: the
/// same labels, the same message layout, `p(0), p(2), p(3)` per round and the final evaluations
/// in instance order, so that an honest run produces the engine's bytes.
fn prove_split_batch<T: SumcheckTranscript<Fp2> + ?Sized, const ARM: u8>(
    transcript: &mut T,
    prefix: Prefix,
    instances: &[SplitInstance<'_>],
) -> SplitBatch {
    let inv2 = Fp::from_u64_reduced(2).inverse().expect("2 is invertible");
    let vars: Vec<usize> = instances
        .iter()
        .map(|instance| {
            let len = instance.relation.len();
            assert!(len.is_power_of_two(), "a table of 2^n entries");
            let n = len.trailing_zeros() as usize;
            assert_eq!(instance.point.len(), n, "one coordinate per variable");
            n
        })
        .collect();
    let n_max = vars.iter().copied().max().expect("an instance");
    let gamma = transcript.challenge(&format!("{prefix}/gamma"));
    let mut weight = Fp2::ONE;
    let mut states: Vec<SplitState> = instances
        .iter()
        .zip(&vars)
        .map(|(instance, &n)| {
            let state = SplitState {
                vars: n,
                pad: n_max - n,
                weight,
                claim: instance.relation.claim(&instance.claims, gamma),
                s: Fp2::ONE,
                last: None,
                bound: if n == 0 {
                    instance.relation.singles()
                } else {
                    Vec::new()
                },
            };
            weight *= gamma;
            state
        })
        .collect();

    let mut fallbacks = 0;
    let mut rounds = Vec::with_capacity(n_max);
    let mut point = Vec::with_capacity(n_max);
    for j in 0..n_max {
        let mut message = [Fp2::ZERO; 3];
        for (instance, state) in instances.iter().zip(&mut states) {
            if j < state.pad {
                // Constant in the padding variables: 2^{pad − j − 1}·S_i at every node, S_i the
                // instance's claim (prove_batch computes S_i from the tables).
                let scale = Fp::from_u64_reduced(1 << (state.pad - 1 - j));
                let value = (state.weight * state.claim).mul_by_base(scale);
                for entry in &mut message {
                    *entry += value;
                }
                continue;
            }
            let k = j - state.pad;
            let m = state.vars - k - 1;
            let lo = m.div_ceil(2);
            let eq_lo = eq_table(&instance.point[k + 1..k + 1 + lo]).expect("a small table");
            let eq_hi = eq_table(&instance.point[k + 1 + lo..state.vars]).expect("a small table");
            let r_k = instance.point[k];
            let l0 = state.s * (Fp2::ONE - r_k);
            let l1 = state.s * r_k;
            let with_one = l1 == Fp2::ZERO;
            if with_one {
                fallbacks += 1;
            }
            let sums = if k == 0 {
                instance
                    .relation
                    .first_sums::<ARM>(&eq_lo, &eq_hi, gamma, with_one)
            } else {
                instance
                    .relation
                    .bound_sums::<ARM>(&state.bound, &eq_lo, &eq_hi, gamma, with_one)
            };
            let v0 = sums[0];
            let v2 = sums[2];
            // Gruen's recovery: ℓ(0)·v(0) + ℓ(1)·v(1) = claim; the fallback summed node 1.
            let v1 = if with_one {
                sums[1]
            } else {
                (state.claim - l0 * v0) * l1.inverse().expect("ℓ(1) is non-zero here")
            };
            let d = v2 - v1;
            let v3 = v0 + d + d + d;
            let dl = l1 - l0;
            let l2 = l1 + dl;
            let l3 = l2 + dl;
            message[0] += state.weight * (l0 * v0);
            message[1] += state.weight * (l2 * v2);
            message[2] += state.weight * (l3 * v3);
            state.last = Some(([l0, l1], [v0, v1, v2]));
        }
        transcript.absorb_message(&format!("{prefix}/round/{j}"), &message);
        let r = transcript.challenge(&format!("{prefix}/r/{j}"));
        for (instance, state) in instances.iter().zip(&mut states) {
            if j < state.pad {
                continue;
            }
            let ([l0, l1], [v0, v1, v2]) = state.last.take().expect("an own round");
            let lr = l0 + r * (l1 - l0);
            // v(r) by Newton's form of the quadratic through v(0), v(1), v(2).
            let second = v2 - v1 - v1 + v0;
            let vr = v0 + r * (v1 - v0) + (r * (r - Fp2::ONE)).mul_by_base(inv2) * second;
            state.claim = lr * vr;
            state.s = lr;
            if j == state.pad {
                state.bound = instance.relation.bind_first(r);
            } else {
                for table in &mut state.bound {
                    bind_vec(table, r);
                }
            }
        }
        rounds.push(message.to_vec());
        point.push(r);
    }

    let mut final_evals = Vec::new();
    for (instance, state) in instances.iter().zip(&states) {
        instance
            .relation
            .final_evals(&state.bound, &mut final_evals);
    }
    transcript.absorb_message(&format!("{prefix}/evals"), &final_evals);
    SplitBatch {
        proof: SumcheckProof {
            rounds,
            final_evals,
        },
        gamma,
        point,
        fallbacks,
    }
}

// ---------------------------------------------------------------------------------------------
// The candidate's forest prover (RFC-0010 'Tree levels' on the candidate's kernel)
// ---------------------------------------------------------------------------------------------

/// The harness's copy of `forest.rs` `build_levels`' arithmetic (the production function is
/// private to the crate): the leaves de-interleaved into their even and odd halves, then each
/// level above as `even_t[z] = even_{t+1}[2z]·odd_{t+1}[2z]`,
/// `odd_t[z] = even_{t+1}[2z+1]·odd_{t+1}[2z+1]`. Entry `0` holds the leaf level's halves and
/// entry `h − 1` level 1's; one multiplication per entry produced.
fn build_levels(leaves: &[Fp], height: usize) -> Vec<(Vec<Fp>, Vec<Fp>)> {
    let mut levels: Vec<(Vec<Fp>, Vec<Fp>)> = Vec::with_capacity(height);
    let (even, odd): (Vec<Fp>, Vec<Fp>) = leaves.chunks_exact(2).map(|p| (p[0], p[1])).unzip();
    levels.push((even, odd));
    for _ in 1..height {
        let (e, o) = levels.last().expect("a level");
        let next: (Vec<Fp>, Vec<Fp>) = e
            .chunks_exact(2)
            .zip(o.chunks_exact(2))
            .map(|(e, o)| (e[0] * o[0], e[1] * o[1]))
            .unzip();
        levels.push(next);
    }
    levels
}

/// A forest while its levels are proven by the candidate.
struct ForestState {
    trees_log: usize,
    height: usize,
    join: usize,
    levels: Vec<(Vec<Fp>, Vec<Fp>)>,
    point: Vec<Fp2>,
    claim: Fp2,
}

/// What the candidate's forest prover returns: each level's messages and each forest's leaf
/// claim, as `prove_forests` returns them, plus the fallback count.
struct SplitForests {
    proofs: Vec<SumcheckProof<Fp2>>,
    leaf_claims: Vec<Claim<Fp2>>,
    fallbacks: usize,
}

/// Proves forests `(leaves, height)` from the roots to the leaves on the candidate's kernel,
/// following `prove_forests` step by step (root challenges, the level's batch, `ρ`, the next
/// claims), with each instance's claim derived from the public root values at its root level
/// and carried as `u + ρ·(w − u)` after.
fn split_prove_forests<T: SumcheckTranscript<Fp2> + ?Sized, const ARM: u8>(
    transcript: &mut T,
    forests: &[(&[Fp], usize)],
) -> SplitForests {
    let tallest = forests.iter().map(|&(_, h)| h).max().expect("a forest");
    let mut states: Vec<ForestState> = forests
        .iter()
        .map(|&(leaves, height)| {
            let vars = leaves.len().trailing_zeros() as usize;
            ForestState {
                trees_log: vars - height,
                height,
                join: tallest - height,
                levels: build_levels(leaves, height),
                point: Vec::new(),
                claim: Fp2::ZERO,
            }
        })
        .collect();
    let mut fallbacks = 0;
    let mut proofs = Vec::with_capacity(tallest);
    let mut leaf_claims = Vec::new();
    for level in 0..tallest {
        let s_max = states
            .iter()
            .filter(|state| state.join == level)
            .map(|state| state.trees_log)
            .max()
            .unwrap_or(0);
        let root_challenges: Vec<Fp2> = (0..s_max)
            .map(|c| transcript.challenge(&format!("tree/{level}/root/{c}")))
            .collect();
        for state in states.iter_mut().filter(|state| state.join == level) {
            // The root point and the root claim Σ_y v_y·eq(r, y) from the public root values
            // v_y = even_1[y]·odd_1[y].
            state.point = root_challenges[s_max - state.trees_log..].to_vec();
            let (even, odd) = &state.levels[state.height - 1];
            let weights = eq_table(&state.point).expect("a small table");
            state.claim = even
                .iter()
                .zip(odd)
                .zip(&weights)
                .fold(Fp2::ZERO, |sum, ((&e, &o), &w)| sum + w.mul_by_base(e * o));
        }
        let active: Vec<usize> = (0..states.len())
            .filter(|&i| states[i].join <= level)
            .collect();
        let instances: Vec<SplitInstance<'_>> = active
            .iter()
            .map(|&i| {
                let state = &states[i];
                let t = level - state.join;
                let (even, odd) = &state.levels[state.height - 1 - t];
                SplitInstance {
                    relation: Relation::Product { even, odd },
                    point: state.point.clone(),
                    claims: vec![state.claim],
                }
            })
            .collect();
        let batch = prove_split_batch::<_, ARM>(transcript, Prefix::Tree(level as u32), &instances);
        drop(instances);
        fallbacks += batch.fallbacks;
        let rho = transcript.challenge(&format!("tree/{level}/line"));
        let mut next_claims = Vec::with_capacity(active.len());
        for (&i, evals) in active.iter().zip(batch.proof.final_evals.chunks_exact(2)) {
            let state = &mut states[i];
            let n = state.point.len();
            let own = &batch.point[batch.point.len() - n..];
            let mut point = Vec::with_capacity(n + 1);
            point.push(rho);
            point.extend_from_slice(own);
            let (u, w) = (evals[0], evals[1]);
            let value = u + rho * (w - u);
            let t = level - state.join;
            state.levels[state.height - 1 - t] = (Vec::new(), Vec::new());
            state.point.clone_from(&point);
            state.claim = value;
            next_claims.push(Claim { point, value });
        }
        proofs.push(batch.proof);
        leaf_claims = next_claims;
    }
    SplitForests {
        proofs,
        leaf_claims,
        fallbacks,
    }
}

/// A transcript that returns scripted challenges, for the degenerate-challenge case (iii): both
/// provers are driven by the same script, and their messages must agree.
struct Scripted {
    challenges: Vec<Fp2>,
    next: usize,
    absorbed: Vec<u8>,
}

impl Scripted {
    fn new(challenges: Vec<Fp2>) -> Self {
        Self {
            challenges,
            next: 0,
            absorbed: Vec::new(),
        }
    }
}

impl SumcheckTranscript<Fp2> for Scripted {
    fn absorb_message(&mut self, label: &str, message: &[Fp2]) {
        self.absorbed.extend_from_slice(label.as_bytes());
        for x in message {
            for c in x.coeffs() {
                self.absorbed
                    .extend_from_slice(&c.to_canonical_u64().to_le_bytes());
            }
        }
    }

    fn challenge(&mut self, _label: &str) -> Fp2 {
        let c = self.challenges[self.next];
        self.next += 1;
        c
    }
}

// ---------------------------------------------------------------------------------------------
// Verifier drivers (tv-core), from EXP-0006
// ---------------------------------------------------------------------------------------------

/// `eq(r, y)` at the vertex `y`, from the definition, in the verifier's field.
fn eq_at_vertex(r: &[TvFp2], y: usize) -> TvFp2 {
    r.iter().enumerate().fold(TvFp2::ONE, |product, (i, &ri)| {
        product
            * if (y >> i) & 1 == 1 {
                ri
            } else {
                TvFp2::ONE - ri
            }
    })
}

/// A root claim's value, `Σ_y v_y·eq(r, y)`, from public root values (RFC-0010 'Tree levels').
fn root_claim(roots: &[TvFp2], r: &[TvFp2]) -> TvFp2 {
    assert_eq!(
        roots.len(),
        1 << r.len(),
        "one root per vertex of the point"
    );
    roots
        .iter()
        .enumerate()
        .fold(TvFp2::ZERO, |sum, (y, &v)| sum + v * eq_at_vertex(r, y))
}

/// `f̃(point) = Σ_x f[x]·eq(point, x)` of a table over `Fp`, in the verifier's field.
fn mle(table: &[Fp], point: &[TvFp2]) -> TvFp2 {
    assert_eq!(
        table.len(),
        1 << point.len(),
        "2^n entries at n coordinates"
    );
    let mut weights = vec![TvFp2::ONE];
    for &r in point {
        let high: Vec<TvFp2> = weights.iter().map(|&e| e * r).collect();
        for (e, h) in weights.iter_mut().zip(&high) {
            *e -= *h;
        }
        weights.extend(high);
    }
    table
        .iter()
        .zip(&weights)
        .fold(TvFp2::ZERO, |sum, (&f, &e)| sum + e * tv_fp(f))
}

/// The trees' products, in the verifier's field: the forest's public root values.
fn true_roots(leaves: &[Fp], height: usize) -> Vec<TvFp2> {
    leaves
        .chunks_exact(1 << height)
        .map(|tree| TvFp2::from(tree.iter().fold(TvFp::ONE, |p, &leaf| p * tv_fp(leaf))))
        .collect()
}

/// What a verifier that accepted every level holds.
struct Verified {
    leaf_claims: Vec<TreeClaim>,
    state: [u8; 32],
}

/// Checks the levels' `proofs` as a verifier holding the schedule does (RFC-0010 'Tree levels'),
/// on a fresh `tv-hash` transcript: forest `i` has `2^{s_i}` trees of height `h_i`,
/// `shapes[i] = (s_i, h_i)`, with the public root values `public_roots[i]`. Written from the
/// specification around `tv_core::tree::verify_product_level`; none of it calls a prover.
fn verify_forests(
    shapes: &[(usize, usize)],
    public_roots: &[Vec<TvFp2>],
    proofs: &[tv::SumcheckProof],
) -> Result<Verified, String> {
    let tallest = shapes.iter().map(|&(_, h)| h).max().expect("a forest");
    assert_eq!(proofs.len(), tallest, "one proof per level");
    let mut transcript = tv_transcript();
    let mut claims: Vec<Option<TreeClaim>> = vec![None; shapes.len()];
    for (level, proof) in proofs.iter().enumerate() {
        let joining: Vec<usize> = (0..shapes.len())
            .filter(|&i| tallest - shapes[i].1 == level)
            .collect();
        let s_max = joining.iter().map(|&i| shapes[i].0).max().unwrap_or(0);
        let root_challenges: Vec<TvFp2> = (0..s_max)
            .map(|c| {
                let label = format!("tree/{level}/root/{c}");
                transcript
                    .challenge_fp2(tvt::Ascii::new(&label).expect("ASCII"))
                    .expect("a challenge")
            })
            .collect();
        for &i in &joining {
            let r = root_challenges[s_max - shapes[i].0..].to_vec();
            let value = root_claim(&public_roots[i], &r);
            claims[i] = Some(TreeClaim { point: r, value });
        }
        let present: Vec<usize> = (0..shapes.len())
            .filter(|&i| tallest - shapes[i].1 <= level)
            .collect();
        let entering: Vec<TreeClaim> = present
            .iter()
            .map(|&i| claims[i].clone().expect("a forest that has joined"))
            .collect();
        let checked = verify_product_level(&mut transcript, level as u32, &entering, proof)
            .map_err(|error| format!("level {level}: {error:?}"))?;
        for (&i, next) in present.iter().zip(&checked.claims) {
            claims[i] = Some(next.clone());
        }
    }
    Ok(Verified {
        leaf_claims: claims
            .into_iter()
            .map(|claim| claim.expect("every forest joins"))
            .collect(),
        state: *transcript.state(),
    })
}

/// A table's entries, for the verifier-side claim.
enum Values<'a> {
    Base(&'a [Fp]),
    Ext(&'a [Fp2]),
}

impl Values<'_> {
    fn tv_at(&self, x: usize) -> TvFp2 {
        match self {
            Self::Base(v) => TvFp2::from(tv_fp(v[x])),
            Self::Ext(v) => tv_fp2(v[x]),
        }
    }

    fn at(&self, x: usize) -> Fp2 {
        match self {
            Self::Base(v) => Fp2::from(v[x]),
            Self::Ext(v) => v[x],
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Base(v) => v.len(),
            Self::Ext(v) => v.len(),
        }
    }
}

/// One instance as the verifier sees it: the public `eq` point and the value tables; the claim is
/// the sum over the hypercube.
struct TvInstance<'a> {
    eq_point: &'a [Fp2],
    eq: &'a [Fp2],
    tables: Vec<Values<'a>>,
}

impl TvInstance<'_> {
    /// `Σ_x eq[x]·Π_k T_k[x]` over the hypercube, in the verifier's field.
    fn claim(&self) -> TvFp2 {
        let len = self.tables[0].len();
        (0..len).fold(TvFp2::ZERO, |sum, x| {
            sum + self
                .tables
                .iter()
                .fold(tv_fp2(self.eq[x]), |p, t| p * t.tv_at(x))
        })
    }

    /// The same sum in the prover's field: the candidate's input claim.
    fn prover_claim(&self) -> Fp2 {
        let len = self.tables[0].len();
        (0..len).fold(Fp2::ZERO, |sum, x| {
            sum + self.tables.iter().fold(self.eq[x], |p, t| p * t.at(x))
        })
    }

    fn vars(&self) -> usize {
        self.tables[0].len().trailing_zeros() as usize
    }

    fn evals(&self) -> usize {
        self.tables.len()
    }
}

/// Checks `proof` with the verifier's `verify_batch` against claims summed over the hypercube in
/// `tv-field`, the public `eq` factor evaluated by the verifier's own `eq`; returns the point and
/// the transcript's end state.
fn verify_views(
    name: &str,
    degree: usize,
    views: &[TvInstance<'_>],
    proof: &SumcheckProof<Fp2>,
) -> (Vec<TvFp2>, [u8; 32]) {
    let proof = tv_proof(proof);
    let claims: Vec<tv::Instance> = views
        .iter()
        .map(|view| tv::Instance {
            claim: view.claim(),
            vars: view.vars(),
            evals: view.evals(),
        })
        .collect();
    let eq_points: Vec<Vec<TvFp2>> = views.iter().map(|view| tv_fp2s(view.eq_point)).collect();
    let relation = |index: usize, point: &[TvFp2], evals: &[TvFp2]| -> Option<TvFp2> {
        let product = evals.iter().fold(TvFp2::ONE, |p, e| p * *e);
        Some(tv::eq(&eq_points[index], point)? * product)
    };
    let mut verifier = tv_transcript();
    let batch = tv::verify_batch(
        &mut verifier,
        tv::Prefix::Stage(0),
        degree,
        &claims,
        &proof,
        relation,
    )
    .unwrap_or_else(|error| panic!("{name}: the verifier rejects an honest batch: {error:?}"));
    (batch.point, *verifier.state())
}

// ---------------------------------------------------------------------------------------------
// Correctness checks
// ---------------------------------------------------------------------------------------------

/// Three products of `spec/vectors/field/goldilocks.json` (edge values).
fn check_field() {
    let cases: [(u64, u64, u64); 3] = [
        (
            0xffff_ffff_0000_0000,
            0xffff_ffff_0000_0000,
            0x0000_0000_0000_0001,
        ),
        (
            0x8000_0000_0000_0000,
            0x8000_0000_0000_0000,
            0xffff_fffe_c000_0001,
        ),
        (
            0x0000_0000_ffff_ffff,
            0x0000_0001_0000_0000,
            0xffff_ffff_0000_0000,
        ),
    ];
    for (a, b, product) in cases {
        let got = black_box(Fp::from_u64_reduced(a)) * black_box(Fp::from_u64_reduced(b));
        assert_eq!(
            got.to_canonical_u64(),
            product,
            "field vector {a:#x} * {b:#x}"
        );
    }
    println!("check field:     3 edge products of spec/vectors/field/goldilocks.json reproduced");
}

/// Stacked leaves of `2^s` trees of height `h`, one zero leaf per tree at a seeded position, as
/// `SoftmaxRow` trees have.
fn forest_leaves(rng: &mut SplitMix64, trees_log: usize, height: usize) -> Vec<Fp> {
    let mut leaves = rng.fps(1 << (trees_log + height));
    for y in 0..1usize << trees_log {
        let x = (rng.next_u64() % (1u64 << height)) as usize;
        leaves[(y << height) + x] = Fp::ZERO;
    }
    leaves
}

/// Proves `forests` with the production prover on a fresh transcript.
fn baseline_forests(forests: &[(&[Fp], usize)]) -> (ProvenForests<Fp2>, [u8; 32]) {
    let batch: Vec<Forest<'_, Fp2>> = forests
        .iter()
        .map(|&(leaves, h)| Forest::new(leaves, h))
        .collect();
    let mut transcript = Transcript::new(&context());
    let proven = prove_forests(&mut transcript, &batch).expect("well-formed forests");
    (proven, transcript.state())
}

/// Proves `forests` with the candidate on a fresh transcript.
fn candidate_forests<const ARM: u8>(forests: &[(&[Fp], usize)]) -> (SplitForests, [u8; 32]) {
    let mut transcript = Transcript::new(&context());
    let proven = split_prove_forests::<_, ARM>(&mut transcript, forests);
    (proven, transcript.state())
}

/// The baseline and the candidate on `forests`: the candidate's proof of every level equals the
/// baseline's, the transcripts end in the same state, the candidate's proofs are accepted by the
/// verifier driver from the trees' true products, and the leaf claims agree with the baseline's,
/// the verifier's and the leaves' multilinear extension. Returns the shared digest.
fn check_forests<const ARM: u8>(name: &str, forests: &[(&[Fp], usize)]) -> Digest {
    let (base, base_state) = baseline_forests(forests);
    let (cand, cand_state) = candidate_forests::<ARM>(forests);
    let base_proofs: Vec<SumcheckProof<Fp2>> =
        base.levels().iter().map(|l| l.proof().clone()).collect();
    assert_eq!(
        base_proofs.len(),
        cand.proofs.len(),
        "{name}: the same number of levels"
    );
    for (level, (b, c)) in base_proofs.iter().zip(&cand.proofs).enumerate() {
        assert!(
            b == c,
            "{name}: level {level}: the candidate's messages differ from the baseline's"
        );
    }
    assert_eq!(
        base_state, cand_state,
        "{name}: the transcripts end in the same state"
    );
    assert!(
        base.leaf_claims() == cand.leaf_claims.as_slice(),
        "{name}: the leaf claims agree"
    );
    let shapes: Vec<(usize, usize)> = forests
        .iter()
        .map(|&(leaves, h)| (leaves.len().trailing_zeros() as usize - h, h))
        .collect();
    let roots: Vec<Vec<TvFp2>> = forests
        .iter()
        .map(|&(leaves, h)| true_roots(leaves, h))
        .collect();
    let tv_proofs: Vec<tv::SumcheckProof> = cand.proofs.iter().map(tv_proof).collect();
    let verified = verify_forests(&shapes, &roots, &tv_proofs).unwrap_or_else(|rejected| {
        panic!("{name}: the verifier rejects the candidate's honest proof: {rejected}")
    });
    assert_eq!(
        verified.state, cand_state,
        "{name}: the verifier's transcript state"
    );
    for (i, ((mine, theirs), &(leaves, _))) in cand
        .leaf_claims
        .iter()
        .zip(&verified.leaf_claims)
        .zip(forests)
        .enumerate()
    {
        let point = tv_fp2s(&mine.point);
        assert_eq!(point, theirs.point, "{name}: forest {i}: the leaf point");
        assert_eq!(
            tv_fp2(mine.value),
            theirs.value,
            "{name}: forest {i}: the leaf value"
        );
        assert_eq!(
            theirs.value,
            mle(leaves, &point),
            "{name}: forest {i}: the leaves' extension at the leaf point"
        );
    }
    let digest = proofs_digest(&cand.proofs);
    assert_eq!(digest, proofs_digest(&base_proofs), "{name}: digests");
    println!(
        "check forests:   {name}: {} level(s) bit-identical to prove_forests; transcripts agree; verifier accepts from the true roots; {} leaf claim(s) equal the leaves' MLE; fallbacks {}; digest {}",
        cand.proofs.len(),
        forests.len(),
        cand.fallbacks,
        &hex(&digest)[..16]
    );
    digest
}

/// The hand case: one tree of the leaves `[2, 3, 5, 7]`, root 210. The candidate's proof is
/// accepted against 210 and rejected against 211, so the check can fail.
fn check_hand_case<const ARM: u8>() {
    let fp = Fp::from_u64_reduced;
    let leaves: Vec<Fp> = [2u64, 3, 5, 7].into_iter().map(fp).collect();
    check_forests::<ARM>("tree [2, 3, 5, 7]", &[(&leaves, 2)]);
    let (cand, _) = candidate_forests::<ARM>(&[(&leaves, 2)]);
    let proofs: Vec<tv::SumcheckProof> = cand.proofs.iter().map(tv_proof).collect();
    assert_eq!(
        true_roots(&leaves, 2),
        vec![TvFp2::from(tv_fp(fp(210)))],
        "the root is 210"
    );
    let wrong = vec![vec![TvFp2::from(tv_fp(fp(211)))]];
    let rejected = verify_forests(&[(0, 2)], &wrong, &proofs);
    assert!(
        rejected.is_err(),
        "the verifier accepts the candidate's proof against the public root 211"
    );
    println!(
        "check hand case: tree [2, 3, 5, 7]: candidate's proof accepted from root 210; against 211 rejected ({})",
        rejected.err().unwrap_or_default()
    );
}

/// A product instance of `2^n` entries at a random point, as the baseline's `Instance` and the
/// candidate's `SplitInstance`, with the verifier's view.
struct ProductCase {
    even: Vec<Fp>,
    odd: Vec<Fp>,
    point: Vec<Fp2>,
    eq: Vec<Fp2>,
}

impl ProductCase {
    fn random(rng: &mut SplitMix64, n: usize) -> Self {
        let point = rng.fp2s(n);
        Self {
            even: rng.fps(1 << n),
            odd: rng.fps(1 << n),
            eq: eq_table(&point).expect("a table"),
            point,
        }
    }

    fn baseline(&self) -> Instance<'_, Fp2> {
        Instance::new(vec![
            Table::ext(&self.eq).public(),
            Table::base(&self.even),
            Table::base(&self.odd),
        ])
    }

    fn view(&self) -> TvInstance<'_> {
        TvInstance {
            eq_point: &self.point,
            eq: &self.eq,
            tables: vec![Values::Base(&self.even), Values::Base(&self.odd)],
        }
    }

    fn candidate(&self) -> SplitInstance<'_> {
        SplitInstance {
            relation: Relation::Product {
                even: &self.even,
                odd: &self.odd,
            },
            point: self.point.clone(),
            claims: vec![self.view().prover_claim()],
        }
    }
}

/// Baseline and candidate on one batch of product cases under `stage/0` on fresh transcripts:
/// equal proofs, equal transcript states, verifier acceptance of the candidate's proof. Returns
/// the fallback count.
fn check_product_batch<const ARM: u8>(name: &str, cases: &[ProductCase]) -> usize {
    let instances: Vec<Instance<'_, Fp2>> = cases.iter().map(ProductCase::baseline).collect();
    let mut t = Transcript::new(&context());
    let base = prove_batch(&mut t, Prefix::Stage(0), LEVEL_DEGREE, &instances)
        .expect("a well-formed batch");
    let base_state = t.state();
    let split: Vec<SplitInstance<'_>> = cases.iter().map(ProductCase::candidate).collect();
    let mut t = Transcript::new(&context());
    let cand = prove_split_batch::<_, ARM>(&mut t, Prefix::Stage(0), &split);
    assert!(
        base.proof() == &cand.proof,
        "{name}: the candidate's messages differ from prove_batch's"
    );
    assert_eq!(base.point(), cand.point.as_slice(), "{name}: the point");
    assert_eq!(*base.gamma(), cand.gamma, "{name}: gamma");
    assert_eq!(
        base_state,
        t.state(),
        "{name}: the transcripts end in the same state"
    );
    let views: Vec<TvInstance<'_>> = cases.iter().map(ProductCase::view).collect();
    let (point, state) = verify_views(name, LEVEL_DEGREE, &views, &cand.proof);
    assert_eq!(tv_fp2s(&cand.point), point, "{name}: the verifier's point");
    assert_eq!(state, t.state(), "{name}: the verifier's transcript state");
    cand.fallbacks
}

/// The small-size equivalence sweep: every `n` from 1 to 12 alone, and two-instance batches with
/// front-padding (including an instance on no variable).
fn check_small_sweep<const ARM: u8>(rng: &mut SplitMix64) {
    let mut fallbacks = 0;
    for n in 1..=12 {
        let case = ProductCase::random(rng, n);
        fallbacks += check_product_batch::<ARM>(&format!("sweep n = {n}"), &[case]);
    }
    for (a, b) in [(1usize, 3usize), (4, 2), (0, 5), (7, 7), (3, 9), (9, 1)] {
        let cases = [ProductCase::random(rng, a), ProductCase::random(rng, b)];
        fallbacks += check_product_batch::<ARM>(&format!("sweep batch ({a}, {b})"), &cases);
    }
    assert_eq!(fallbacks, 0, "no fallback on random points");
    println!(
        "check sweep:     n = 1..=12 alone and 6 two-instance batches with front-padding (one instance on no variable): bit-identical to prove_batch, verifier accepts; no fallback"
    );
}

/// The two-forest batch of different heights and leaf counts: 2^2 trees of 2^8 leaves with 2^5
/// trees of 2^6 leaves; the second joins at level 2 with one variable more, so the first is
/// front-padded by one round from then on.
fn check_two_forests<const ARM: u8>(rng: &mut SplitMix64) {
    let a = forest_leaves(rng, 2, 8);
    let b = forest_leaves(rng, 5, 6);
    check_forests::<ARM>(
        "two forests (2^2 x 2^8 leaves, 2^5 x 2^6 leaves)",
        &[(&a, 8), (&b, 6)],
    );
}

/// The degenerate-challenge cases: (i) a zero coordinate of the `eq` point (`ℓ_j(1) = 0`: the
/// fallback sums node 1), (ii) a coordinate equal to one (`ℓ_j(0) = 0`: no fallback), (iii) a
/// scripted challenge `c_k = (r_k − 1)/(2r_k − 1)` that makes `ℓ_k(c_k) = 0`, so `s_i = 0` and
/// every later message is zero while the fallback runs in every later round. Each must agree
/// with `prove_batch` bit for bit.
fn check_degenerate<const ARM: u8>(rng: &mut SplitMix64) {
    let n = 12;
    // (i) r_3 = 0.
    let mut case = ProductCase::random(rng, n);
    case.point[3] = Fp2::ZERO;
    case.eq = eq_table(&case.point).expect("a table");
    let fallbacks = check_product_batch::<ARM>("degenerate (i) r_3 = 0", &[case]);
    assert_eq!(fallbacks, 1, "(i): the fallback ran once, in round 3");
    // (ii) r_5 = 1.
    let mut case = ProductCase::random(rng, n);
    case.point[5] = Fp2::ONE;
    case.eq = eq_table(&case.point).expect("a table");
    let fallbacks = check_product_batch::<ARM>("degenerate (ii) r_5 = 1", &[case]);
    assert_eq!(fallbacks, 0, "(ii): no fallback");
    // (iii) a scripted c_2 with ℓ_2(c_2) = 0.
    let case = ProductCase::random(rng, n);
    let r2 = case.point[2];
    let c2 = (r2 - Fp2::ONE) * (r2.double() - Fp2::ONE).inverse().expect("2r − 1 ≠ 0");
    let mut script = rng.fp2s(n + 1);
    script[3] = c2; // γ, r_0, r_1, then c_2.
    let instances = [case.baseline()];
    let mut t = Scripted::new(script.clone());
    let base = prove_batch(&mut t, Prefix::Stage(0), LEVEL_DEGREE, &instances)
        .expect("a well-formed batch");
    let base_absorbed = t.absorbed;
    let split = [case.candidate()];
    let mut t = Scripted::new(script);
    let cand = prove_split_batch::<_, ARM>(&mut t, Prefix::Stage(0), &split);
    assert!(
        base.proof() == &cand.proof,
        "(iii): the candidate's messages differ from prove_batch's under the scripted challenges"
    );
    assert_eq!(base_absorbed, t.absorbed, "(iii): the absorbed bytes agree");
    assert_eq!(
        cand.fallbacks,
        n - 3,
        "(iii): the fallback ran in every round after round 2"
    );
    for (j, round) in cand.proof.rounds.iter().enumerate().skip(3) {
        assert!(
            round.iter().all(|&x| x == Fp2::ZERO),
            "(iii): round {j} is zero once s_i = 0"
        );
    }
    println!(
        "check fallback:  (i) r_3 = 0: fallback once, bit-identical, verifier accepts; (ii) r_5 = 1: no fallback, bit-identical, verifier accepts; (iii) scripted c_2 with l_2(c_2) = 0: fallback in rounds 3..{n}, every later message zero, bit-identical to prove_batch under the same script"
    );
}

// ---------------------------------------------------------------------------------------------
// SPDX-License-Identifier: Apache-2.0 OR MIT

fn bind_base(t: &[Fp], r: Fp2) -> Vec<Fp2> {
    t.chunks_exact(2)
        .map(|p| Fp2::from(p[0]) + r.mul_by_base(p[1] - p[0]))
        .collect()
}

fn bind_vec(t: &mut Vec<Fp2>, r: Fp2) {
    for i in 0..t.len() / 2 {
        t[i] = t[2 * i] + r * (t[2 * i + 1] - t[2 * i]);
    }
    t.truncate(t.len() / 2);
}

const MAX_BLOCK_LOG: u32 = 20;
const _: () = assert!(130 + MAX_BLOCK_LOG <= 192);

#[derive(Clone, Copy, Default)]
struct Acc192 {
    lo: u64,
    mid: u64,
    hi: u64,
}

impl Acc192 {
    fn add(&mut self, p: u128) {
        let (lo, carry) = self.lo.overflowing_add(p as u64);
        let (mid, carry1) = self.mid.overflowing_add((p >> 64) as u64);
        let (mid, carry2) = mid.overflowing_add(u64::from(carry));
        self.lo = lo;
        self.mid = mid;
        self.hi += u64::from(carry1) + u64::from(carry2);
    }
    fn reduce(self) -> Fp {
        // 2^64 = 2^32-1 and 2^128 = -2^32 modulo Goldilocks.
        Fp::from_u64_reduced(self.lo) + Fp::from_u64_reduced(self.mid).mul_small(u32::MAX)
            - Fp::from_u64_reduced(self.hi) * Fp::from_u64_reduced(1u64 << 32)
    }
}

#[derive(Clone, Copy, Default)]
struct AccExt2 {
    c0: Acc192,
    c1: Acc192,
    c7: Acc192,
}

impl AccExt2 {
    fn product(&mut self, a: Fp2, b: Fp2) {
        let [a0, a1] = a.coeffs().map(|x| x.to_canonical_u64() as u128);
        let [b0, b1] = b.coeffs().map(|x| x.to_canonical_u64() as u128);
        self.c0.add(a0 * b0);
        self.c1.add(a0 * b1);
        self.c1.add(a1 * b0);
        self.c7.add(a1 * b1);
    }
    fn base_product(&mut self, a: Fp2, b: Fp) {
        let [a0, a1] = a.coeffs().map(|x| x.to_canonical_u64() as u128);
        let b = b.to_canonical_u64() as u128;
        self.c0.add(a0 * b);
        self.c1.add(a1 * b);
    }
    fn reduce(self) -> Fp2 {
        Fp2::new(
            self.c0.reduce() + self.c7.reduce().mul_small(7),
            self.c1.reduce(),
        )
    }
}

fn check_block(eq_lo: &[Fp2]) {
    assert!(eq_lo.len() <= 1usize << MAX_BLOCK_LOG);
}

fn deferred_product_base_sums<const ONE: bool, const ARM: u8>(
    even: &[Fp],
    odd: &[Fp],
    eq_lo: &[Fp2],
    eq_hi: &[Fp2],
) -> [Fp2; 3] {
    check_block(eq_lo);
    let mut sums = [Fp2::ZERO; 3];
    let group = 2 * eq_lo.len();
    for ((h, e), o) in eq_hi
        .iter()
        .zip(even.chunks_exact(group))
        .zip(odd.chunks_exact(group))
    {
        let mut acc = [AccExt2::default(); 3];
        for ((w, l), r) in eq_lo.iter().zip(e.chunks_exact(2)).zip(o.chunks_exact(2)) {
            acc[0].base_product(*w, l[0] * r[0]);
            acc[2].base_product(*w, (l[1] + (l[1] - l[0])) * (r[1] + (r[1] - r[0])));
            if ONE {
                acc[1].base_product(*w, l[1] * r[1]);
            }
        }
        sums[0] += *h * acc[0].reduce();
        sums[2] += *h * acc[2].reduce();
        if ONE {
            sums[1] += *h * acc[1].reduce();
        }
    }
    sums
}

fn deferred_product_ext_sums<const ONE: bool, const ARM: u8>(
    left: &[Fp2],
    right: &[Fp2],
    eq_lo: &[Fp2],
    eq_hi: &[Fp2],
) -> [Fp2; 3] {
    check_block(eq_lo);
    let mut sums = [Fp2::ZERO; 3];
    let group = 2 * eq_lo.len();
    for ((h, e), o) in eq_hi
        .iter()
        .zip(left.chunks_exact(group))
        .zip(right.chunks_exact(group))
    {
        let mut acc = [AccExt2::default(); 3];
        for ((w, l), r) in eq_lo.iter().zip(e.chunks_exact(2)).zip(o.chunks_exact(2)) {
            acc[0].product(*w, l[0] * r[0]);
            acc[2].product(*w, (l[1] + (l[1] - l[0])) * (r[1] + (r[1] - r[0])));
            if ONE {
                acc[1].product(*w, l[1] * r[1]);
            }
        }
        sums[0] += *h * acc[0].reduce();
        sums[2] += *h * acc[2].reduce();
        if ONE {
            sums[1] += *h * acc[1].reduce();
        }
    }
    sums
}

fn deferred_fraction_leaf_sums<const ONE: bool, const ARM: u8>(
    d_e: &[Fp2],
    d_o: &[Fp2],
    eq_lo: &[Fp2],
    eq_hi: &[Fp2],
    gamma: Fp2,
    gamma2: Fp2,
) -> [Fp2; 3] {
    check_block(eq_lo);
    let scaled: Vec<Fp2> = if ARM == 1 {
        eq_lo.iter().map(|w| *w * gamma2).collect()
    } else {
        Vec::new()
    };
    let mut sums = [Fp2::ZERO; 3];
    let group = 2 * eq_lo.len();
    for ((h, e), o) in eq_hi
        .iter()
        .zip(d_e.chunks_exact(group))
        .zip(d_o.chunks_exact(group))
    {
        let mut acc = [AccExt2::default(); 3];
        for (i, ((w, e), o)) in eq_lo
            .iter()
            .zip(e.chunks_exact(2))
            .zip(o.chunks_exact(2))
            .enumerate()
        {
            let mut add = |z: usize, e: Fp2, o: Fp2| {
                if ARM == 1 {
                    acc[z].product(*w * e, o + gamma);
                    acc[z].product(scaled[i], o);
                } else {
                    acc[z].product(*w, e * (o + gamma) + gamma2 * o);
                }
            };
            add(0, e[0], o[0]);
            add(2, e[1] + (e[1] - e[0]), o[1] + (o[1] - o[0]));
            if ONE {
                add(1, e[1], o[1]);
            }
        }
        sums[0] += *h * acc[0].reduce();
        sums[2] += *h * acc[2].reduce();
        if ONE {
            sums[1] += *h * acc[1].reduce();
        }
    }
    sums
}

fn deferred_fraction_generic_sums<const ONE: bool, const ARM: u8>(
    tables: [&[Fp2]; 4],
    eq_lo: &[Fp2],
    eq_hi: &[Fp2],
    gamma: Fp2,
    gamma2: Fp2,
) -> [Fp2; 3] {
    check_block(eq_lo);
    let scaled: Vec<Fp2> = if ARM == 1 {
        eq_lo.iter().map(|w| *w * gamma).collect()
    } else {
        Vec::new()
    };
    let [n_e, n_o, d_e, d_o] = tables;
    let mut sums = [Fp2::ZERO; 3];
    let group = 2 * eq_lo.len();
    for ((((h, ne), no), de), dd) in eq_hi
        .iter()
        .zip(n_e.chunks_exact(group))
        .zip(n_o.chunks_exact(group))
        .zip(d_e.chunks_exact(group))
        .zip(d_o.chunks_exact(group))
    {
        let mut acc = [AccExt2::default(); 3];
        for (i, ((((w, ne), no), de), dd)) in eq_lo
            .iter()
            .zip(ne.chunks_exact(2))
            .zip(no.chunks_exact(2))
            .zip(de.chunks_exact(2))
            .zip(dd.chunks_exact(2))
            .enumerate()
        {
            let mut add = |z: usize, ne: Fp2, no: Fp2, de: Fp2, dd: Fp2| {
                if ARM == 1 {
                    acc[z].product(*w * dd, ne + gamma2 * de);
                    acc[z].product(scaled[i] * no, de);
                } else {
                    acc[z].product(*w, dd * (ne + gamma2 * de) + gamma * (no * de));
                }
            };
            let at2 = |t: &[Fp2]| t[1] + (t[1] - t[0]);
            add(0, ne[0], no[0], de[0], dd[0]);
            add(2, at2(ne), at2(no), at2(de), at2(dd));
            if ONE {
                add(1, ne[1], no[1], de[1], dd[1]);
            }
        }
        sums[0] += *h * acc[0].reduce();
        sums[2] += *h * acc[2].reduce();
        if ONE {
            sums[1] += *h * acc[1].reduce();
        }
    }
    sums
}

struct Pair {
    name: String,
    base: f64,
    cand: f64,
    base_cv: f64,
    cand_cv: f64,
}

fn verify_level_arm<const ARM: u8>(
    name: &str,
    split: &SplitInstance<'_>,
    instances: &[Instance<'_, Fp2>],
    views: &[TvInstance<'_>],
) {
    let mut t = Transcript::new(&context());
    let base = prove_batch(&mut t, Prefix::Stage(0), 3, instances).expect("batch");
    let expected_state = t.state();
    let mut t = Transcript::new(&context());
    let got = prove_split_batch::<_, ARM>(&mut t, Prefix::Stage(0), std::slice::from_ref(split));
    assert!(
        base.proof() == &got.proof,
        "{name} arm {ARM}: message identity"
    );
    assert_eq!(t.state(), expected_state, "transcript identity");
    let (point, state) = verify_views(name, 3, views, &got.proof);
    assert_eq!(point, tv_fp2s(&got.point));
    assert_eq!(state, expected_state);
    println!(
        "check {name} arm {ARM}: production proof/transcript identical, tv-core accepts; digest {}",
        hex(&proofs_digest(&[got.proof]))
    );
}

fn bench_split<const ARM: u8>(split: &SplitInstance<'_>) -> (Stats, u32) {
    timed(|| {
        let mut t = Transcript::new(&context());
        black_box(
            prove_split_batch::<_, ARM>(
                &mut t,
                Prefix::Stage(0),
                black_box(std::slice::from_ref(split)),
            )
            .proof,
        );
    })
}

#[allow(clippy::too_many_arguments)]
fn level_arms(
    rows: &mut Vec<Row>,
    pairs: &mut Vec<Pair>,
    name: &str,
    instances: &[Instance<'_, Fp2>],
    views: &[TvInstance<'_>],
    relation: Relation<'_>,
    point: &[Fp2],
    fraction: bool,
    timing: bool,
) {
    let split = SplitInstance {
        relation,
        point: point.to_vec(),
        claims: views.iter().map(TvInstance::prover_claim).collect(),
    };
    verify_level_arm::<0>(name, &split, instances, views);
    verify_level_arm::<1>(name, &split, instances, views);
    verify_level_arm::<2>(name, &split, instances, views);
    if !timing {
        return;
    }
    let m = split.relation.len() as f64;
    let (stats, reps) = timed(|| {
        let mut t = Transcript::new(&context());
        black_box(
            prove_batch(&mut t, Prefix::Stage(0), 3, black_box(instances))
                .expect("batch")
                .into_proof(),
        );
    });
    report(
        rows,
        &format!("{name}: production context"),
        m,
        "entry",
        stats,
        reps,
    );
    let (bs, br) = bench_split::<0>(&split);
    let base = report(rows, &format!("{name}: split"), m, "entry", bs, br);
    if fraction {
        let (cs, cr) = bench_split::<1>(&split);
        let cand = report(rows, &format!("{name}: scale-first"), m, "entry", cs, cr);
        pairs.push(Pair {
            name: format!("{name}: scale-first"),
            base,
            cand,
            base_cv: bs.cv,
            cand_cv: cs.cv,
        });
    }
    let (cs, cr) = bench_split::<2>(&split);
    let arm = if fraction {
        "relation-first"
    } else {
        "deferred"
    };
    let cand = report(rows, &format!("{name}: {arm}"), m, "entry", cs, cr);
    pairs.push(Pair {
        name: format!("{name}: {arm}"),
        base,
        cand,
        base_cv: bs.cv,
        cand_cv: cs.cv,
    });
}

fn fraction_cases(
    rows: &mut Vec<Row>,
    pairs: &mut Vec<Pair>,
    n: usize,
    timing: bool,
    point_case: u8,
) {
    let mut rng = SplitMix64(FRACTION_SEED);
    let m = 1usize << n;
    let keys = rng.fps(2 * m);
    let values = rng.fps(2 * m);
    let (alpha, beta) = (rng.fp2(), rng.fp2());
    let d = fraction_leaves(&keys, &values, alpha, beta);
    let (de, dd): (Vec<Fp2>, Vec<Fp2>) = d.chunks_exact(2).map(|p| (p[0], p[1])).unzip();
    drop((keys, values, d));
    let mut point = rng.fp2s(n);
    if point_case == 1 {
        point[n / 2] = Fp2::ZERO;
    }
    if point_case == 2 {
        point[n / 2] = Fp2::ONE;
    }
    let eq = eq_table(&point).expect("eq");
    let instances = [
        Instance::new(vec![
            Table::ext(&eq).public(),
            Table::ext(&de),
            Table::ext(&dd),
        ]),
        Instance::new(vec![Table::ext(&eq).public(), Table::ext(&de)]),
        Instance::new(vec![Table::ext(&eq).public(), Table::ext(&dd)]),
    ];
    let views = [
        TvInstance {
            eq_point: &point,
            eq: &eq,
            tables: vec![Values::Ext(&de), Values::Ext(&dd)],
        },
        TvInstance {
            eq_point: &point,
            eq: &eq,
            tables: vec![Values::Ext(&de)],
        },
        TvInstance {
            eq_point: &point,
            eq: &eq,
            tables: vec![Values::Ext(&dd)],
        },
    ];
    level_arms(
        rows,
        pairs,
        &format!("fraction leaf 2^{n} point{point_case}"),
        &instances,
        &views,
        Relation::FractionLeaf { d_e: &de, d_o: &dd },
        &point,
        true,
        timing,
    );
    let tables: Vec<Vec<Fp2>> = (0..4).map(|_| rng.fp2s(m)).collect();
    let [ne, no, de, dd] = [&tables[0], &tables[1], &tables[2], &tables[3]];
    let instances = [
        Instance::new(vec![
            Table::ext(&eq).public(),
            Table::ext(ne),
            Table::ext(dd),
        ]),
        Instance::new(vec![
            Table::ext(&eq).public(),
            Table::ext(no),
            Table::ext(de),
        ]),
        Instance::new(vec![
            Table::ext(&eq).public(),
            Table::ext(de),
            Table::ext(dd),
        ]),
    ];
    let views = [
        TvInstance {
            eq_point: &point,
            eq: &eq,
            tables: vec![Values::Ext(ne), Values::Ext(dd)],
        },
        TvInstance {
            eq_point: &point,
            eq: &eq,
            tables: vec![Values::Ext(no), Values::Ext(de)],
        },
        TvInstance {
            eq_point: &point,
            eq: &eq,
            tables: vec![Values::Ext(de), Values::Ext(dd)],
        },
    ];
    level_arms(
        rows,
        pairs,
        &format!("fraction generic 2^{n} point{point_case}"),
        &instances,
        &views,
        Relation::FractionGeneric {
            tables: [ne, no, de, dd],
        },
        &point,
        true,
        timing,
    );
}

fn fraction_leaves(keys: &[Fp], values: &[Fp], alpha: Fp2, beta: Fp2) -> Vec<Fp2> {
    keys.iter()
        .zip(values)
        .map(|(&k, &v)| alpha - Fp2::from(k) - beta.mul_by_base(v))
        .collect()
}

fn product_level(rows: &mut Vec<Row>, pairs: &mut Vec<Pair>, timing: bool) {
    let mut rng = SplitMix64(LEAF_SEED);
    let leaves = forest_leaves(&mut rng, 8, 14);
    let levels = build_levels(&leaves, 14);
    let (even, odd) = &levels[0];
    let mut rng = SplitMix64(ENGINE_SEED);
    let point = rng.fp2s(21);
    let eq = eq_table(&point).expect("eq");
    let instance = Instance::new(vec![
        Table::ext(&eq).public(),
        Table::base(even),
        Table::base(odd),
    ]);
    let view = TvInstance {
        eq_point: &point,
        eq: &eq,
        tables: vec![Values::Base(even), Values::Base(odd)],
    };
    level_arms(
        rows,
        pairs,
        "product level 2^21",
        &[instance],
        &[view],
        Relation::Product { even, odd },
        &point,
        false,
        timing,
    );
}

fn forest_arms(rows: &mut Vec<Row>, pairs: &mut Vec<Pair>, timing: bool) {
    let mut rng = SplitMix64(LEAF_SEED);
    for (trees, height, judged) in [
        (8, 14, true),
        (10, 12, true),
        (0, 22, true),
        (6, 14, false),
        (7, 14, false),
    ] {
        let leaves = forest_leaves(&mut rng, trees, height);
        let forest = [(leaves.as_slice(), height)];
        let name = format!("forest 2^{trees}x2^{height}");
        let a = check_forests::<0>(&name, &forest);
        let b = check_forests::<2>(&name, &forest);
        assert_eq!(a, b, "split vs deferred digest");
        if !timing {
            continue;
        }
        let (stats, reps) = timed(|| {
            let _ = black_box(baseline_forests(black_box(&forest)));
        });
        report(
            rows,
            &format!("{name}: production context"),
            leaves.len() as f64,
            "leaf",
            stats,
            reps,
        );
        let (bs, br) = timed(|| {
            black_box(candidate_forests::<0>(black_box(&forest)));
        });
        let base = report(
            rows,
            &format!("{name}: split"),
            leaves.len() as f64,
            "leaf",
            bs,
            br,
        );
        let (cs, cr) = timed(|| {
            black_box(candidate_forests::<2>(black_box(&forest)));
        });
        let cand = report(
            rows,
            &format!("{name}: deferred"),
            leaves.len() as f64,
            "leaf",
            cs,
            cr,
        );
        if judged {
            pairs.push(Pair {
                name,
                base,
                cand,
                base_cv: bs.cv,
                cand_cv: cs.cv,
            });
        }
    }
}

fn micro_arms(rows: &mut Vec<Row>, timing: bool) {
    let mut rng = SplitMix64(0x7E55_E7A0_0058_0006);
    let a = rng.fp2s(1 << 14);
    let b = rng.fp2s(1 << 14);
    let c = rng.fps(1 << 14);
    let reduced = || {
        black_box(a.as_slice())
            .iter()
            .zip(black_box(b.as_slice()))
            .fold(Fp2::ZERO, |s, (&x, &y)| s + x * y)
    };
    let deferred = || {
        let mut sum = Fp2::ZERO;
        for (a, b) in black_box(a.as_slice())
            .chunks(1 << 10)
            .zip(black_box(b.as_slice()).chunks(1 << 10))
        {
            let mut acc = AccExt2::default();
            for (&x, &y) in a.iter().zip(b) {
                acc.product(x, y);
            }
            sum += acc.reduce();
        }
        sum
    };
    let reduced_base = || {
        a.iter()
            .zip(&c)
            .fold(Fp2::ZERO, |s, (&x, &y)| s + x.mul_by_base(y))
    };
    let deferred_base = || {
        let mut sum = Fp2::ZERO;
        for (a, c) in black_box(a.as_slice())
            .chunks(1 << 10)
            .zip(black_box(c.as_slice()).chunks(1 << 10))
        {
            let mut acc = AccExt2::default();
            for (&x, &y) in a.iter().zip(c) {
                acc.base_product(x, y);
            }
            sum += acc.reduce();
        }
        sum
    };
    assert_eq!(reduced(), deferred());
    assert_eq!(reduced_base(), deferred_base());
    if !timing {
        return;
    }
    for (name, f) in [
        ("dot ext reduced", &reduced as &dyn Fn() -> Fp2),
        ("dot ext deferred", &deferred),
        ("dot base reduced", &reduced_base),
        ("dot base deferred", &deferred_base),
    ] {
        let (stats, reps) = timed(|| {
            black_box(f());
        });
        report(rows, name, a.len() as f64, "product", stats, reps);
    }
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}
fn write_json(path: &str, rows: &[Row], pairs: &[Pair]) {
    let mut out = String::from(
        "{\"experiment\":\"EXP-0013\",\"protocol\":{\"warmups\":3,\"iterations\":10,\"minimum_batch_ms\":20,\"cv_limit\":0.03},\"cases\":[\n",
    );
    for (i, r) in rows.iter().enumerate() {
        out.push_str(&format!("{{\"name\":\"{}\",\"per\":\"{}\",\"elements\":{},\"reps\":{},\"median_ns\":{},\"min_ns\":{},\"max_ns\":{},\"cv\":{},\"repeated\":{},\"samples_seconds_per_execution\":{:?}}}{}\n",json_escape(&r.name),r.per,r.elements,r.reps,r.stats.median*1e9/r.elements,r.stats.min*1e9/r.elements,r.stats.max*1e9/r.elements,r.stats.cv,r.stats.repeated,r.stats.samples,if i+1<rows.len(){","}else{""}));
    }
    out.push_str("],\"pairs\":[\n");
    for (i, p) in pairs.iter().enumerate() {
        let sigma = 1.0 - p.cand / p.base;
        println!(
            "PAIR {} sigma {:.4}% CV {:.3}%/{:.3}%",
            p.name,
            100.0 * sigma,
            100.0 * p.base_cv,
            100.0 * p.cand_cv
        );
        out.push_str(&format!("{{\"name\":\"{}\",\"base_ns\":{},\"cand_ns\":{},\"sigma\":{},\"base_cv\":{},\"cand_cv\":{}}}{}\n",json_escape(&p.name),p.base,p.cand,sigma,p.base_cv,p.cand_cv,if i+1<pairs.len(){","}else{""}));
    }
    out.push_str("]}\n");
    std::fs::write(path, out).expect("JSON");
}

fn run_checks<const ARM: u8>() {
    check_hand_case::<ARM>();
    let mut rng = SplitMix64(CHECK_SEED);
    check_small_sweep::<ARM>(&mut rng);
    check_two_forests::<ARM>(&mut rng);
    check_degenerate::<ARM>(&mut rng);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map_or("check", String::as_str);
    assert!(["all", "check"].contains(&cmd));
    let timing = cmd == "all";
    println!(
        "EXP-0013 delayed reduction; arm 0=split, 1=scale-first, 2=relation-first/deferred; all inputs synthetic"
    );
    check_field();
    run_checks::<0>();
    run_checks::<1>();
    run_checks::<2>();
    let mut rows = Vec::new();
    let mut pairs = Vec::new();
    for n in 1..=12 {
        fraction_cases(&mut rows, &mut pairs, n, false, 0);
    }
    for p in [1, 2] {
        fraction_cases(&mut rows, &mut pairs, 8, false, p);
    }
    if timing {
        forest_arms(&mut rows, &mut pairs, false);
        product_level(&mut rows, &mut pairs, false);
        fraction_cases(&mut rows, &mut pairs, 21, false, 0);
        micro_arms(&mut rows, false);
    }
    forest_arms(&mut rows, &mut pairs, timing);
    product_level(&mut rows, &mut pairs, timing);
    fraction_cases(&mut rows, &mut pairs, 21, timing, 0);
    micro_arms(&mut rows, timing);
    if let Some(path) = args.get(2) {
        write_json(path, &rows, &pairs);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn accumulator_boundaries() {
        let p = 0xffff_ffff_0000_0001u128;
        let mut acc = Acc192::default();
        for _ in 0..(4u64 << MAX_BLOCK_LOG) {
            acc.add((p - 1) * (p - 1));
        }
        assert_eq!(acc.reduce(), Fp::from_u64_reduced(4u64 << MAX_BLOCK_LOG));
        let mut rng = SplitMix64(0x7E55_E7A0_0058_0007);
        let mut acc = Acc192::default();
        let mut sum = Fp::ZERO;
        let mut ext = AccExt2::default();
        let mut ext_sum = Fp2::ZERO;
        let mut base = AccExt2::default();
        let mut base_sum = Fp2::ZERO;
        for _ in 0..1 << 16 {
            let (a, b) = (rng.fp(), rng.fp());
            acc.add(a.to_canonical_u64() as u128 * b.to_canonical_u64() as u128);
            sum += a * b;
            let (a, b) = (rng.fp2(), rng.fp2());
            ext.product(a, b);
            ext_sum += a * b;
            let b = rng.fp();
            base.base_product(a, b);
            base_sum += a.mul_by_base(b);
        }
        assert_eq!(acc.reduce(), sum);
        assert_eq!(ext.reduce(), ext_sum);
        assert_eq!(base.reduce(), base_sum);
        // Independent modular Horner reduction checks all carry words, including a full 192-bit input.
        for words in [[u64::MAX; 3], [0, u64::MAX, 0], [0, 0, 1], [u64::MAX, 0, 1]] {
            let epsilon = Fp::from_u64_reduced(u32::MAX as u64);
            let expected = (Fp::from_u64_reduced(words[2]) * epsilon
                + Fp::from_u64_reduced(words[1]))
                * epsilon
                + Fp::from_u64_reduced(words[0]);
            assert_eq!(
                Acc192 {
                    lo: words[0],
                    mid: words[1],
                    hi: words[2]
                }
                .reduce(),
                expected
            );
        }
    }
}
// SPDX-License-Identifier: Apache-2.0 OR MIT
