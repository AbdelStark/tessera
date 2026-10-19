// SPDX-License-Identifier: Apache-2.0 OR MIT

//! EXP-0008: the split-`eq`, Gruen-corrected level kernel of IDEA-0021 against the engine's
//! `eq`-table kernel (README.md in this directory).
//!
//! Baseline and candidate run in this one program on the same seeded inputs, under RFC-0020's
//! protocol (3 warm-ups, 10 iterations, median, minimum, maximum, coefficient of variation; a
//! coefficient above 3% is flagged and the case rerun once; every iteration lasts at least 20 ms
//! by repetition). The baseline is the production `prove_forests` / `prove_batch` of
//! `tessera-sumcheck`, unmodified. The candidate, written here, never materializes or binds an
//! `eq` table, evaluates two node sums per round, and reconstructs the degree-3 message from the
//! instance's running claim ([Gru24] §3.2, [DT24], [JoltOpt26]). Every candidate proof must equal
//! the baseline's bit for bit and pass the independent verifier (`tv-core`) before anything is
//! timed, so that a fast but wrong path cannot enter a number.
//!
//! Research code: it prints, it panics on a failed check, and nothing depends on it.
//!
//! ```sh
//! cargo run --release --locked -- all [results/run.json]
//! ```

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
use tessera_poly::{PolyField, bind_lowest, bind_lowest_in_place, eq_table};
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

/// `N = 2^22` leaves: RFC-0010's largest test case and the bench's size.
const LOG_N: usize = 22;
/// RFC-0020's `c_π` row shape: rows of 2^14 leaves.
const ROW_HEIGHT: usize = 14;
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
}

fn sample(iteration: &mut impl FnMut() -> Duration) -> Stats {
    for _ in 0..WARMUPS {
        black_box(iteration());
    }
    let mut s: Vec<f64> = (0..ITERATIONS).map(|_| iteration().as_secs_f64()).collect();
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
    (MIN_ITERATION.as_secs_f64() / one).ceil().clamp(1.0, 1e7) as u32
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
        start.elapsed() / reps
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
fn product_base_sums<const WITH_ONE: bool>(
    even: &[Fp],
    odd: &[Fp],
    eq_lo: &[Fp2],
    eq_hi: &[Fp2],
) -> [Fp2; 3] {
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
fn product_ext_sums<const WITH_ONE: bool>(
    left: &[Fp2],
    right: &[Fp2],
    eq_lo: &[Fp2],
    eq_hi: &[Fp2],
) -> [Fp2; 3] {
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
fn fraction_leaf_sums<const WITH_ONE: bool>(
    d_e: &[Fp2],
    d_o: &[Fp2],
    eq_lo: &[Fp2],
    eq_hi: &[Fp2],
    gamma: Fp2,
    gamma2: Fp2,
) -> [Fp2; 3] {
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
fn fraction_generic_sums<const WITH_ONE: bool>(
    tables: [&[Fp2]; 4],
    eq_lo: &[Fp2],
    eq_hi: &[Fp2],
    gamma: Fp2,
    gamma2: Fp2,
) -> [Fp2; 3] {
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
    fn first_sums(&self, eq_lo: &[Fp2], eq_hi: &[Fp2], gamma: Fp2, with_one: bool) -> [Fp2; 3] {
        let g2 = gamma * gamma;
        match (self, with_one) {
            (Self::Product { even, odd }, false) => {
                product_base_sums::<false>(even, odd, eq_lo, eq_hi)
            }
            (Self::Product { even, odd }, true) => {
                product_base_sums::<true>(even, odd, eq_lo, eq_hi)
            }
            (Self::FractionLeaf { d_e, d_o }, false) => {
                fraction_leaf_sums::<false>(d_e, d_o, eq_lo, eq_hi, gamma, g2)
            }
            (Self::FractionLeaf { d_e, d_o }, true) => {
                fraction_leaf_sums::<true>(d_e, d_o, eq_lo, eq_hi, gamma, g2)
            }
            (Self::FractionGeneric { tables }, false) => {
                fraction_generic_sums::<false>(*tables, eq_lo, eq_hi, gamma, g2)
            }
            (Self::FractionGeneric { tables }, true) => {
                fraction_generic_sums::<true>(*tables, eq_lo, eq_hi, gamma, g2)
            }
        }
    }

    /// The node sums of a later own round, over the bound tables.
    fn bound_sums(
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
                product_ext_sums::<false>(&bound[0], &bound[1], eq_lo, eq_hi)
            }
            (Self::Product { .. }, true) => {
                product_ext_sums::<true>(&bound[0], &bound[1], eq_lo, eq_hi)
            }
            (Self::FractionLeaf { .. }, false) => {
                fraction_leaf_sums::<false>(&bound[0], &bound[1], eq_lo, eq_hi, gamma, g2)
            }
            (Self::FractionLeaf { .. }, true) => {
                fraction_leaf_sums::<true>(&bound[0], &bound[1], eq_lo, eq_hi, gamma, g2)
            }
            (Self::FractionGeneric { .. }, false) => fraction_generic_sums::<false>(
                [&bound[0], &bound[1], &bound[2], &bound[3]],
                eq_lo,
                eq_hi,
                gamma,
                g2,
            ),
            (Self::FractionGeneric { .. }, true) => fraction_generic_sums::<true>(
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
            Self::Product { even, odd } => vec![
                bind_lowest::<Fp2>(even, r).expect("a variable"),
                bind_lowest::<Fp2>(odd, r).expect("a variable"),
            ],
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
fn prove_split_batch<T: SumcheckTranscript<Fp2> + ?Sized>(
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
                    .first_sums(&eq_lo, &eq_hi, gamma, with_one)
            } else {
                instance
                    .relation
                    .bound_sums(&state.bound, &eq_lo, &eq_hi, gamma, with_one)
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
                    bind_lowest_in_place(table, r).expect("a variable");
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
fn split_prove_forests<T: SumcheckTranscript<Fp2> + ?Sized>(
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
        let batch = prove_split_batch(transcript, Prefix::Tree(level as u32), &instances);
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
fn candidate_forests(forests: &[(&[Fp], usize)]) -> (SplitForests, [u8; 32]) {
    let mut transcript = Transcript::new(&context());
    let proven = split_prove_forests(&mut transcript, forests);
    (proven, transcript.state())
}

/// The baseline and the candidate on `forests`: the candidate's proof of every level equals the
/// baseline's, the transcripts end in the same state, the candidate's proofs are accepted by the
/// verifier driver from the trees' true products, and the leaf claims agree with the baseline's,
/// the verifier's and the leaves' multilinear extension. Returns the shared digest.
fn check_forests(name: &str, forests: &[(&[Fp], usize)]) -> Digest {
    let (base, base_state) = baseline_forests(forests);
    let (cand, cand_state) = candidate_forests(forests);
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
fn check_hand_case() {
    let fp = Fp::from_u64_reduced;
    let leaves: Vec<Fp> = [2u64, 3, 5, 7].into_iter().map(fp).collect();
    check_forests("tree [2, 3, 5, 7]", &[(&leaves, 2)]);
    let (cand, _) = candidate_forests(&[(&leaves, 2)]);
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
fn check_product_batch(name: &str, cases: &[ProductCase]) -> usize {
    let instances: Vec<Instance<'_, Fp2>> = cases.iter().map(ProductCase::baseline).collect();
    let mut t = Transcript::new(&context());
    let base = prove_batch(&mut t, Prefix::Stage(0), LEVEL_DEGREE, &instances)
        .expect("a well-formed batch");
    let base_state = t.state();
    let split: Vec<SplitInstance<'_>> = cases.iter().map(ProductCase::candidate).collect();
    let mut t = Transcript::new(&context());
    let cand = prove_split_batch(&mut t, Prefix::Stage(0), &split);
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
fn check_small_sweep(rng: &mut SplitMix64) {
    let mut fallbacks = 0;
    for n in 1..=12 {
        let case = ProductCase::random(rng, n);
        fallbacks += check_product_batch(&format!("sweep n = {n}"), &[case]);
    }
    for (a, b) in [(1usize, 3usize), (4, 2), (0, 5), (7, 7), (3, 9), (9, 1)] {
        let cases = [ProductCase::random(rng, a), ProductCase::random(rng, b)];
        fallbacks += check_product_batch(&format!("sweep batch ({a}, {b})"), &cases);
    }
    assert_eq!(fallbacks, 0, "no fallback on random points");
    println!(
        "check sweep:     n = 1..=12 alone and 6 two-instance batches with front-padding (one instance on no variable): bit-identical to prove_batch, verifier accepts; no fallback"
    );
}

/// The two-forest batch of different heights and leaf counts: 2^2 trees of 2^8 leaves with 2^5
/// trees of 2^6 leaves; the second joins at level 2 with one variable more, so the first is
/// front-padded by one round from then on.
fn check_two_forests(rng: &mut SplitMix64) {
    let a = forest_leaves(rng, 2, 8);
    let b = forest_leaves(rng, 5, 6);
    check_forests(
        "two forests (2^2 x 2^8 leaves, 2^5 x 2^6 leaves)",
        &[(&a, 8), (&b, 6)],
    );
}

/// The degenerate-challenge cases: (i) a zero coordinate of the `eq` point (`ℓ_j(1) = 0`: the
/// fallback sums node 1), (ii) a coordinate equal to one (`ℓ_j(0) = 0`: no fallback), (iii) a
/// scripted challenge `c_k = (r_k − 1)/(2r_k − 1)` that makes `ℓ_k(c_k) = 0`, so `s_i = 0` and
/// every later message is zero while the fallback runs in every later round. Each must agree
/// with `prove_batch` bit for bit.
fn check_degenerate(rng: &mut SplitMix64) {
    let n = 12;
    // (i) r_3 = 0.
    let mut case = ProductCase::random(rng, n);
    case.point[3] = Fp2::ZERO;
    case.eq = eq_table(&case.point).expect("a table");
    let fallbacks = check_product_batch("degenerate (i) r_3 = 0", &[case]);
    assert_eq!(fallbacks, 1, "(i): the fallback ran once, in round 3");
    // (ii) r_5 = 1.
    let mut case = ProductCase::random(rng, n);
    case.point[5] = Fp2::ONE;
    case.eq = eq_table(&case.point).expect("a table");
    let fallbacks = check_product_batch("degenerate (ii) r_5 = 1", &[case]);
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
    let cand = prove_split_batch(&mut t, Prefix::Stage(0), &split);
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
// Arms
// ---------------------------------------------------------------------------------------------

/// A verdict-relevant pair of medians.
struct Pair {
    name: String,
    base: f64,
    cand: f64,
    base_cv: f64,
    cand_cv: f64,
}

fn forest_arms(rows: &mut Vec<Row>, pairs: &mut Vec<Pair>) {
    println!(
        "\n== forests: prove_forests (baseline) against split_forest (candidate), one thread, per leaf =="
    );
    let mut rng = SplitMix64(LEAF_SEED);
    let shapes: [(usize, usize, bool); 5] = [
        (LOG_N - ROW_HEIGHT, ROW_HEIGHT, true),
        (10, 12, true),
        (0, LOG_N, true),
        (4, ROW_HEIGHT, false),
        (6, ROW_HEIGHT, false),
    ];
    for (trees_log, height, in_verdict) in shapes {
        let leaves = forest_leaves(&mut rng, trees_log, height);
        let n = leaves.len() as f64;
        let shape = format!(
            "2^{trees_log} trees of 2^{height} leaves (2^{} leaves)",
            trees_log + height
        );
        let forests = [(leaves.as_slice(), height)];
        check_forests(&shape, &forests);
        let (stats, reps) = timed(|| {
            let (proven, _) = baseline_forests(black_box(&forests));
            black_box(proven.into_parts());
        });
        let base = report(
            rows,
            &format!("prove_forests: {shape}"),
            n,
            "leaf",
            stats,
            reps,
        );
        let (stats2, reps2) = timed(|| {
            let (proven, _) = candidate_forests(black_box(&forests));
            black_box((proven.proofs, proven.leaf_claims));
        });
        let cand = report(
            rows,
            &format!("split_forest:  {shape}"),
            n,
            "leaf",
            stats2,
            reps2,
        );
        println!(
            "    ratio T_base / T_cand = {:.3}; speed-up sigma = {:.1}%{}",
            base / cand,
            100.0 * (1.0 - cand / base),
            if in_verdict {
                "  (verdict shape)"
            } else {
                "  (context)"
            }
        );
        if in_verdict {
            pairs.push(Pair {
                name: shape,
                base,
                cand,
                base_cv: stats.cv,
                cand_cv: stats2.cv,
            });
        }
    }
}

fn kernel_arms(rows: &mut Vec<Row>) {
    println!(
        "\n== the level kernel alone: prove_batch on [eq (Fp2, public), even, odd (Fp)] against the split-eq kernel, 2^21 entries, per entry =="
    );
    let mut leaf_rng = SplitMix64(LEAF_SEED);
    let leaves = forest_leaves(&mut leaf_rng, LOG_N - ROW_HEIGHT, ROW_HEIGHT);
    let levels = build_levels(&leaves, ROW_HEIGHT);
    let (even, odd) = &levels[0];
    let m = even.len();
    let mut rng = SplitMix64(ENGINE_SEED);
    let point = rng.fp2s(LOG_N - 1);
    let eq = eq_table(&point).expect("a table");
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
    let claim = view.prover_claim();
    let split = SplitInstance {
        relation: Relation::Product { even, odd },
        point: point.clone(),
        claims: vec![claim],
    };
    // Check: equal proofs and states, verifier acceptance.
    let mut t = Transcript::new(&context());
    let base = prove_batch(
        &mut t,
        Prefix::Stage(0),
        LEVEL_DEGREE,
        std::slice::from_ref(&instance),
    )
    .expect("a well-formed batch");
    let base_state = t.state();
    let mut t = Transcript::new(&context());
    let cand = prove_split_batch(&mut t, Prefix::Stage(0), std::slice::from_ref(&split));
    assert!(
        base.proof() == &cand.proof,
        "level kernel: the messages differ"
    );
    assert_eq!(base_state, t.state(), "level kernel: the transcripts agree");
    let (vpoint, vstate) = verify_views(
        "level kernel",
        LEVEL_DEGREE,
        std::slice::from_ref(&view),
        &cand.proof,
    );
    assert_eq!(
        tv_fp2s(&cand.point),
        vpoint,
        "level kernel: the verifier's point"
    );
    assert_eq!(
        vstate,
        t.state(),
        "level kernel: the verifier's transcript state"
    );
    println!(
        "check kernel:    leaf-level instance (eq x even x odd, 2^21): bit-identical to prove_batch; verify_batch accepts from the hypercube sum; fallbacks {}; digest {}",
        cand.fallbacks,
        &hex(&proofs_digest(std::slice::from_ref(&cand.proof)))[..16]
    );
    let (stats, reps) = timed(|| {
        let mut t = Transcript::new(&context());
        let proven = prove_batch(
            &mut t,
            Prefix::Stage(0),
            LEVEL_DEGREE,
            black_box(std::slice::from_ref(&instance)),
        )
        .expect("a well-formed batch");
        black_box(proven.into_proof());
    });
    let base = report(
        rows,
        "engine: eq public (Fp2) x even x odd (Fp), degree 3, 2^21",
        m as f64,
        "entry",
        stats,
        reps,
    );
    let (stats, reps) = timed(|| {
        let mut t = Transcript::new(&context());
        let proven = prove_split_batch(
            &mut t,
            Prefix::Stage(0),
            black_box(std::slice::from_ref(&split)),
        );
        black_box(proven.proof);
    });
    let cand = report(
        rows,
        "split-eq kernel: even x odd (Fp) at the eq point, 2^21",
        m as f64,
        "entry",
        stats,
        reps,
    );
    println!(
        "    ratio T_base / T_cand = {:.3}; speed-up sigma = {:.1}% (observation a)",
        base / cand,
        100.0 * (1.0 - cand / base)
    );
    // Context: the eq tables of every level alone, the component the candidate removes.
    let trees_log = LOG_N - ROW_HEIGHT;
    let points: Vec<Vec<Fp2>> = (0..ROW_HEIGHT)
        .map(|level| rng.fp2s(trees_log + level))
        .collect();
    let (stats, reps) = timed(|| {
        for point in &points {
            black_box(eq_table(black_box(point)).expect("a table"));
        }
    });
    report(
        rows,
        "context: eq tables of every level (2^8 x 2^14), per leaf",
        (1usize << LOG_N) as f64,
        "leaf",
        stats,
        reps,
    );
}

/// The fraction-tree proxy's values: leaves `d = α − key − β·value`.
fn fraction_leaves(keys: &[Fp], values: &[Fp], alpha: Fp2, beta: Fp2) -> Vec<Fp2> {
    keys.iter()
        .zip(values)
        .map(|(&k, &v)| alpha - Fp2::from(k) - beta.mul_by_base(v))
        .collect()
}

/// One fraction-proxy level: the baseline's three instances and the candidate's one, checked
/// and timed.
#[allow(clippy::too_many_arguments)]
fn fraction_level(
    rows: &mut Vec<Row>,
    pairs: &mut Vec<Pair>,
    name: &str,
    short: &str,
    m: usize,
    instances: &[Instance<'_, Fp2>],
    views: &[TvInstance<'_>],
    relation: Relation<'_>,
    point: &[Fp2],
) {
    let claims: Vec<Fp2> = views.iter().map(TvInstance::prover_claim).collect();
    let split = SplitInstance {
        relation,
        point: point.to_vec(),
        claims,
    };
    let mut t = Transcript::new(&context());
    let base = prove_batch(&mut t, Prefix::Stage(0), LEVEL_DEGREE, instances)
        .expect("a well-formed batch");
    let base_state = t.state();
    let mut t = Transcript::new(&context());
    let cand = prove_split_batch(&mut t, Prefix::Stage(0), std::slice::from_ref(&split));
    assert!(base.proof() == &cand.proof, "{name}: the messages differ");
    assert_eq!(base_state, t.state(), "{name}: the transcripts agree");
    let (vpoint, vstate) = verify_views(name, LEVEL_DEGREE, views, &cand.proof);
    assert_eq!(tv_fp2s(&cand.point), vpoint, "{name}: the verifier's point");
    assert_eq!(vstate, t.state(), "{name}: the verifier's transcript state");
    println!(
        "check fraction:  {name}: bit-identical to prove_batch (3 instances); verify_batch accepts from the hypercube sums; fallbacks {}; digest {}",
        cand.fallbacks,
        &hex(&proofs_digest(std::slice::from_ref(&cand.proof)))[..16]
    );
    let (stats, reps) = timed(|| {
        let mut t = Transcript::new(&context());
        let proven = prove_batch(&mut t, Prefix::Stage(0), LEVEL_DEGREE, black_box(instances))
            .expect("a well-formed batch");
        black_box(proven.into_proof());
    });
    let b = report(
        rows,
        &format!("fraction proxy: {short}, prove_batch (3 instances), 2^21"),
        m as f64,
        "entry",
        stats,
        reps,
    );
    let (stats2, reps2) = timed(|| {
        let mut t = Transcript::new(&context());
        let proven = prove_split_batch(
            &mut t,
            Prefix::Stage(0),
            black_box(std::slice::from_ref(&split)),
        );
        black_box(proven.proof);
    });
    let c = report(
        rows,
        &format!("fraction proxy: {short}, split-eq one-relation kernel, 2^21"),
        m as f64,
        "entry",
        stats2,
        reps2,
    );
    println!(
        "    ratio T_base / T_cand = {:.3}; speed-up sigma = {:.1}% (IDEA-0018 case)",
        b / c,
        100.0 * (1.0 - c / b)
    );
    pairs.push(Pair {
        name: format!("fraction proxy: {short}"),
        base: b,
        cand: c,
        base_cv: stats.cv,
        cand_cv: stats2.cv,
    });
}

fn fraction_arms(rows: &mut Vec<Row>, pairs: &mut Vec<Pair>) {
    println!(
        "\n== fraction-tree proxy (IDEA-0018 case): prove_batch's three instances against the one-relation split-eq kernel, 2^21 entries over Fp2, per entry =="
    );
    let mut rng = SplitMix64(FRACTION_SEED);
    let n = 1usize << LOG_N;
    let m = n / 2;
    let keys = rng.fps(n);
    let values = rng.fps(n);
    let (alpha, beta) = (rng.fp2(), rng.fp2());
    let d = fraction_leaves(&keys, &values, alpha, beta);
    drop((keys, values));
    let (d_e, d_o): (Vec<Fp2>, Vec<Fp2>) = d.chunks_exact(2).map(|p| (p[0], p[1])).unzip();
    drop(d);
    let point = rng.fp2s(LOG_N - 1);
    let eq = eq_table(&point).expect("a table");

    // Leaf level, numerators 1: [eq, d_e, d_o], [eq, d_e], [eq, d_o].
    let instances = [
        Instance::new(vec![
            Table::ext(&eq).public(),
            Table::ext(&d_e),
            Table::ext(&d_o),
        ]),
        Instance::new(vec![Table::ext(&eq).public(), Table::ext(&d_e)]),
        Instance::new(vec![Table::ext(&eq).public(), Table::ext(&d_o)]),
    ];
    let views = [
        TvInstance {
            eq_point: &point,
            eq: &eq,
            tables: vec![Values::Ext(&d_e), Values::Ext(&d_o)],
        },
        TvInstance {
            eq_point: &point,
            eq: &eq,
            tables: vec![Values::Ext(&d_e)],
        },
        TvInstance {
            eq_point: &point,
            eq: &eq,
            tables: vec![Values::Ext(&d_o)],
        },
    ];
    fraction_level(
        rows,
        pairs,
        "leaf level (numerators 1): [eq, d_e, d_o] + [eq, d_e] + [eq, d_o]",
        "leaf level",
        m,
        &instances,
        &views,
        Relation::FractionLeaf {
            d_e: &d_e,
            d_o: &d_o,
        },
        &point,
    );
    drop(instances);
    drop(views);
    drop((d_e, d_o));

    // Generic level: [eq, n_e, d_o], [eq, n_o, d_e], [eq, d_e, d_o] on random Fp2 tables.
    let tables: Vec<Vec<Fp2>> = (0..4).map(|_| rng.fp2s(m)).collect();
    let (n_e, n_o, g_e, g_o) = (&tables[0], &tables[1], &tables[2], &tables[3]);
    let instances = [
        Instance::new(vec![
            Table::ext(&eq).public(),
            Table::ext(n_e),
            Table::ext(g_o),
        ]),
        Instance::new(vec![
            Table::ext(&eq).public(),
            Table::ext(n_o),
            Table::ext(g_e),
        ]),
        Instance::new(vec![
            Table::ext(&eq).public(),
            Table::ext(g_e),
            Table::ext(g_o),
        ]),
    ];
    let views = [
        TvInstance {
            eq_point: &point,
            eq: &eq,
            tables: vec![Values::Ext(n_e), Values::Ext(g_o)],
        },
        TvInstance {
            eq_point: &point,
            eq: &eq,
            tables: vec![Values::Ext(n_o), Values::Ext(g_e)],
        },
        TvInstance {
            eq_point: &point,
            eq: &eq,
            tables: vec![Values::Ext(g_e), Values::Ext(g_o)],
        },
    ];
    fraction_level(
        rows,
        pairs,
        "generic level: [eq, n_e, d_o] + [eq, n_o, d_e] + [eq, d_e, d_o]",
        "generic level",
        m,
        &instances,
        &views,
        Relation::FractionGeneric {
            tables: [n_e, n_o, g_e, g_o],
        },
        &point,
    );
}

// ---------------------------------------------------------------------------------------------
// The verdict rule, printed, and the JSON record
// ---------------------------------------------------------------------------------------------

fn summary(forest_pairs: &[Pair], fraction_pairs: &[Pair]) {
    println!("\n== pre-registered rule (README 'Success criterion'), evaluated ==");
    let mut all_20 = true;
    let mut all_below_15 = true;
    let mut flagged = false;
    for p in forest_pairs {
        let sigma = 1.0 - p.cand / p.base;
        all_20 &= sigma >= 0.20;
        all_below_15 &= sigma < 0.15;
        flagged |= p.base_cv > CV_LIMIT || p.cand_cv > CV_LIMIT;
        println!(
            "  {:<48} base {:>8.3}  cand {:>8.3} ns/leaf  ratio {:.3}  sigma {:>5.1}%  CVs {:.2}% / {:.2}%",
            p.name,
            p.base,
            p.cand,
            p.base / p.cand,
            100.0 * sigma,
            100.0 * p.base_cv,
            100.0 * p.cand_cv
        );
    }
    let word = if flagged {
        "inconclusive (a coefficient of variation above 3% on a verdict arm)"
    } else if all_20 {
        "validated (sigma >= 20% on every verdict shape)"
    } else if all_below_15 {
        "refuted (sigma < 15% on every verdict shape)"
    } else {
        "inconclusive"
    };
    println!("  IDEA-0021 (the experiment's verdict): {word}");
    let mut all_30 = true;
    let mut all_below_10 = true;
    let mut flagged = false;
    for p in fraction_pairs {
        let sigma = 1.0 - p.cand / p.base;
        all_30 &= sigma >= 0.30;
        all_below_10 &= sigma < 0.10;
        flagged |= p.base_cv > CV_LIMIT || p.cand_cv > CV_LIMIT;
        println!(
            "  {:<48} base {:>8.3}  cand {:>8.3} ns/entry ratio {:.3}  sigma {:>5.1}%  CVs {:.2}% / {:.2}%",
            p.name,
            p.base,
            p.cand,
            p.base / p.cand,
            100.0 * sigma,
            100.0 * p.base_cv,
            100.0 * p.cand_cv
        );
    }
    let word = if flagged {
        "inconclusive (a coefficient of variation above 3%)"
    } else if all_30 {
        "validated (sigma >= 30% on both levels)"
    } else if all_below_10 {
        "refuted (sigma < 10% on both levels)"
    } else {
        "inconclusive"
    };
    println!("  IDEA-0018 (the second case's status word): {word}");
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn write_json(path: &str, rows: &[Row], forest_pairs: &[Pair], fraction_pairs: &[Pair]) {
    let mut out = String::from(
        "{\n  \"experiment\": \"EXP-0008\",\n  \"protocol\": {\"warmups\": 3, \"iterations\": 10, \"cv_limit\": 0.03, \"min_iteration_ms\": 20},\n  \"cases\": [\n",
    );
    for (i, row) in rows.iter().enumerate() {
        let ns = |t: f64| t * 1e9 / row.elements;
        out.push_str(&format!(
            "    {{\"name\": \"{}\", \"per\": \"{}\", \"elements\": {}, \"reps\": {}, \"median_ns\": {}, \"min_ns\": {}, \"max_ns\": {}, \"cv\": {}, \"median_ms\": {}, \"repeated\": {}}}{}\n",
            json_escape(&row.name),
            row.per,
            row.elements,
            row.reps,
            ns(row.stats.median),
            ns(row.stats.min),
            ns(row.stats.max),
            row.stats.cv,
            row.stats.median * 1e3,
            row.stats.repeated,
            if i + 1 < rows.len() { "," } else { "" }
        ));
    }
    out.push_str("  ],\n  \"pairs\": [\n");
    let all: Vec<(&str, &Pair)> = forest_pairs
        .iter()
        .map(|p| ("forest", p))
        .chain(fraction_pairs.iter().map(|p| ("fraction", p)))
        .collect();
    for (i, (kind, p)) in all.iter().enumerate() {
        out.push_str(&format!(
            "    {{\"kind\": \"{kind}\", \"name\": \"{}\", \"base_ns\": {}, \"cand_ns\": {}, \"ratio\": {}, \"sigma\": {}, \"base_cv\": {}, \"cand_cv\": {}}}{}\n",
            json_escape(&p.name),
            p.base,
            p.cand,
            p.base / p.cand,
            1.0 - p.cand / p.base,
            p.base_cv,
            p.cand_cv,
            if i + 1 < all.len() { "," } else { "" }
        ));
    }
    out.push_str("  ]\n}\n");
    std::fs::write(path, out).expect("write the JSON record");
    println!("\nJSON record written to {path}");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let command = args.get(1).map_or("all", String::as_str);
    if command != "all" {
        eprintln!("usage: spliteq all [results/run.json]");
        std::process::exit(2);
    }
    println!(
        "EXP-0008 harness: split-eq Gruen level kernel (candidate, this program) against the production prove_forests / prove_batch (baseline), one thread; RFC-0020 protocol (3 warm-ups, 10 iterations, median/min/max/CV, rerun once above 3%, iterations of at least 20 ms by repetition)"
    );
    println!("\n== correctness checks ==");
    check_field();
    check_hand_case();
    let mut rng = SplitMix64(CHECK_SEED);
    check_small_sweep(&mut rng);
    check_two_forests(&mut rng);
    check_degenerate(&mut rng);

    let mut rows = Vec::new();
    let mut forest_pairs = Vec::new();
    let mut fraction_pairs = Vec::new();
    forest_arms(&mut rows, &mut forest_pairs);
    kernel_arms(&mut rows);
    fraction_arms(&mut rows, &mut fraction_pairs);
    summary(&forest_pairs, &fraction_pairs);
    if let Some(path) = args.get(2) {
        write_json(path, &rows, &forest_pairs, &fraction_pairs);
    }
}
