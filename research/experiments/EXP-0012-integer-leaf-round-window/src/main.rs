// SPDX-License-Identifier: Apache-2.0 OR MIT

//! EXP-0012: integer first-round windows against EXP-0008's split-eq kernel,
//! with the current production forest prover as context (README.md).
//!
//! Baseline and candidate run in this one program on the same seeded inputs, under RFC-0020's
//! protocol (3 warm-ups, 10 iterations, median, minimum, maximum, coefficient of variation; a
//! coefficient above 3% is flagged and the case rerun once; every iteration lasts at least 20 ms
//! by repetition). The reference split-eq kernel is ported from EXP-0008. The integer
//! candidate changes only the leaf level's first window, preserving every message.
//! Every candidate proof must equal
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
use tessera_poly::{PolyField, bind_lowest, eq_table};
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
        (0..n)
            .map(|_| Fp::from_u64_reduced(self.next_u64() & ((1 << 22) - 1)))
            .collect()
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
    println!("raw_samples_seconds {:?}", s);
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
    println!("rerun: first CV {}", first.cv);
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
        let elapsed = start.elapsed();
        assert!(
            elapsed >= MIN_ITERATION,
            "sample shorter than 20ms: {elapsed:?}"
        );
        elapsed / reps
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

/// The value tables of one instance of the candidate, unbound then bound; the `eq` factor is
/// never a table here.
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
}

impl Relation<'_> {
    /// The tables' length, `2^{n_i}`.
    fn len(&self) -> usize {
        match self {
            Self::Product { even, .. } => even.len(),
            Self::FractionLeaf { d_e, .. } => d_e.len(),
        }
    }

    /// The instance's claim from the schedule's claims and `γ`: the one claim of a product level,
    /// or the three fraction instances' claims combined with `1, γ, γ²`.
    fn claim(&self, claims: &[Fp2], gamma: Fp2) -> Fp2 {
        match self {
            Self::Product { .. } => claims[0],
            Self::FractionLeaf { .. } => claims[0] + gamma * claims[1] + gamma * gamma * claims[2],
        }
    }

    /// The tables' single entries, lifted: the bound state of an instance on no variable.
    fn singles(&self) -> Vec<Vec<Fp2>> {
        match self {
            Self::Product { even, odd } => vec![vec![Fp2::from(even[0])], vec![Fp2::from(odd[0])]],
            Self::FractionLeaf { d_e, d_o } => vec![vec![d_e[0]], vec![d_o[0]]],
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
        }
    }

    /// The first binding, from the borrowed tables into tables over `Fp2`: `bind_lowest` for
    /// tables over `Fp` (the engine's `bind_base`), `bind_ext` for tables over `Fp2`.
    fn bind_first(&self, r: Fp2) -> Vec<Vec<Fp2>> {
        match self {
            Self::Product { even, odd } => vec![
                bind_lowest::<Fp2>(even, r).expect("a variable").to_vec(),
                bind_lowest::<Fp2>(odd, r).expect("a variable").to_vec(),
            ],
            Self::FractionLeaf { d_e, d_o } => vec![bind_ext(d_e, r), bind_ext(d_o, r)],
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
    window: Option<Window>,
}

// Synthetic-only Vec port of the production SecretBuf fold; no production input enters here.
fn bind_vec(table: &mut Vec<Fp2>, r: Fp2) {
    for i in 0..table.len() / 2 {
        table[i] = table[2 * i] + r * (table[2 * i + 1] - table[2 * i]);
    }
    table.truncate(table.len() / 2);
}

const KEY_MAX: u64 = (1 << 22) - 1;
const WINDOWS: [usize; 4] = [1, 3, 4, 5];

fn cadence(v: usize) -> usize {
    if v == 5 { 8 } else { 32 }
}

// The sign correction subtracts 2^128 mod P from a negative two's-complement word.
fn reduce_signed(x: i128) -> Fp {
    let u = x as u128;
    let hi = (u >> 64) as u64;
    let eps = 0xffff_ffffu64;
    Fp::from_u64_reduced(u as u64) - Fp::from_u64_reduced(hi >> 32)
        + Fp::from_u64_reduced((hi & eps) * eps)
        - Fp::from_u64_reduced(((x >> 127) as u64) & (eps * eps))
}

fn integer_grid(input: &[Fp], width: usize, a: &mut [i64]) {
    for (out, x) in a.iter_mut().zip(input) {
        *out = x.to_canonical_u64() as i64;
    }
    let mut stride = 1;
    for k in 0..width {
        let groups = 1 << (width - k - 1);
        // Expand backwards so the larger ternary grid cannot overwrite unread binary entries.
        for g in (0..groups).rev() {
            for j in (0..stride).rev() {
                let x = a[g * 2 * stride + j];
                let y = a[g * 2 * stride + stride + j];
                a[g * 3 * stride + j] = x;
                a[g * 3 * stride + stride + j] = y;
                a[g * 3 * stride + 2 * stride + j] = 2 * y - x;
            }
        }
        stride *= 3;
    }
}

struct Window {
    width: usize,
    grid: Vec<Fp2>,
    challenges: Vec<Fp2>,
}

impl Window {
    fn new(even: &[Fp], odd: &[Fp], point: &[Fp2], width: usize) -> Self {
        let m = point.len() - width;
        let lo = m.div_ceil(2);
        let eq_lo = eq_table(&point[width..width + lo]).expect("eq low");
        let eq_hi = eq_table(&point[width + lo..]).expect("eq high");
        let cells = 3usize.pow(width as u32);
        let block = 1 << width;
        let mut grid = vec![Fp2::ZERO; cells];
        let mut a = vec![0i64; cells];
        let mut b = vec![0i64; cells];
        let mut inner = vec![Fp2::ZERO; cells];
        let mut sums = vec![[0i128; 2]; cells];
        for (h, wh) in eq_hi.iter().enumerate() {
            inner.fill(Fp2::ZERO);
            for start in (0..eq_lo.len()).step_by(cadence(width)) {
                sums.fill([0; 2]);
                for (l, wl) in eq_lo.iter().enumerate().skip(start).take(cadence(width)) {
                    let index = (h * eq_lo.len() + l) * block;
                    integer_grid(&even[index..index + block], width, &mut a);
                    integer_grid(&odd[index..index + block], width, &mut b);
                    let weights = wl.coeffs().map(|x| x.to_canonical_u64() as i128);
                    for z in 0..cells {
                        let product = (a[z] * b[z]) as i128;
                        sums[z][0] += product * weights[0];
                        sums[z][1] += product * weights[1];
                    }
                }
                for z in 0..cells {
                    inner[z] += Fp2::new(reduce_signed(sums[z][0]), reduce_signed(sums[z][1]));
                }
            }
            for z in 0..cells {
                grid[z] += inner[z] * *wh;
            }
        }
        Self {
            width,
            grid,
            challenges: Vec::with_capacity(width),
        }
    }

    fn sums(&self, future: &[Fp2]) -> [Fp2; 3] {
        let weights = eq_table(future).expect("window eq");
        let mut sums = [Fp2::ZERO; 3];
        for (bits, w) in weights.iter().enumerate() {
            let mut ternary = 0;
            let mut stride = 3;
            for j in 0..future.len() {
                ternary += ((bits >> j) & 1) * stride;
                stride *= 3;
            }
            for (z, sum) in sums.iter_mut().enumerate() {
                *sum += self.grid[ternary + z] * *w;
            }
        }
        sums
    }

    fn collapse(&mut self, r: Fp2) {
        let half = Fp::from_u64_reduced(2).inverse().expect("two");
        let quadratic_weight = (r * (r - Fp2::ONE)).mul_by_base(half);
        for i in 0..self.grid.len() / 3 {
            let a = self.grid[3 * i];
            let b = self.grid[3 * i + 1];
            let c = self.grid[3 * i + 2];
            self.grid[i] = a + r * (b - a) + quadratic_weight * (c - b - b + a);
        }
        self.grid.truncate(self.grid.len() / 3);
        self.challenges.push(r);
    }

    fn bind(&self, table: &[Fp]) -> Vec<Fp2> {
        let w = eq_table(&self.challenges).expect("window binding weights");
        table
            .chunks_exact(w.len())
            .map(|chunk| {
                chunk
                    .iter()
                    .zip(&w)
                    .fold(Fp2::ZERO, |s, (&x, &w)| s + w.mul_by_base(x))
            })
            .collect()
    }
}

fn prove_split_batch_window<T: SumcheckTranscript<Fp2> + ?Sized>(
    transcript: &mut T,
    prefix: Prefix,
    instances: &[SplitInstance<'_>],
    window: usize,
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
                window: if window > 0 && n >= window {
                    match instance.relation {
                        Relation::Product { even, odd } => {
                            Some(Window::new(even, odd, &instance.point, window))
                        }
                        Relation::FractionLeaf { d_e, d_o } => Some(Window::new_fraction(
                            d_e,
                            d_o,
                            &instance.point,
                            window,
                            gamma,
                        )),
                    }
                } else {
                    None
                },
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
            let (eq_lo, eq_hi) = if state.window.as_ref().is_some_and(|w| k < w.width) {
                (Vec::new(), Vec::new())
            } else {
                (
                    eq_table(&instance.point[k + 1..k + 1 + lo]).expect("a small table"),
                    eq_table(&instance.point[k + 1 + lo..state.vars]).expect("a small table"),
                )
            };
            let r_k = instance.point[k];
            let l0 = state.s * (Fp2::ONE - r_k);
            let l1 = state.s * r_k;
            let with_one = l1 == Fp2::ZERO;
            if with_one {
                fallbacks += 1;
            }
            let sums = if let Some(w) = state.window.as_ref().filter(|w| k < w.width) {
                w.sums(&instance.point[k + 1..w.width])
            } else if k == 0 {
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
            if let Some(w) = state.window.as_mut().filter(|w| j - state.pad < w.width) {
                w.collapse(r);
                if j - state.pad + 1 == w.width {
                    state.bound = match instance.relation {
                        Relation::Product { even, odd } => vec![w.bind(even), w.bind(odd)],
                        Relation::FractionLeaf { d_e, d_o } => {
                            vec![w.bind_extension(d_e), w.bind_extension(d_o)]
                        }
                    };
                }
            } else if j == state.pad {
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
fn split_prove_forests<T: SumcheckTranscript<Fp2> + ?Sized>(
    transcript: &mut T,
    forests: &[(&[Fp], usize)],
    window: usize,
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
        let batch = prove_split_batch_window(
            transcript,
            Prefix::Tree(level as u32),
            &instances,
            if level + 1 == tallest { window } else { 0 },
        );
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
    let proven = split_prove_forests(&mut transcript, forests, 0);
    (proven, transcript.state())
}

/// The baseline and the candidate on `forests`: the candidate's proof of every level equals the
/// baseline's, the transcripts end in the same state, the candidate's proofs are accepted by the
/// verifier driver from the trees' true products, and the leaf claims agree with the baseline's,
/// the verifier's and the leaves' multilinear extension. Returns the shared digest.
fn check_forests(name: &str, forests: &[(&[Fp], usize)]) -> Digest {
    check_window_forests(name, forests, 0)
}
fn check_window_forests(name: &str, forests: &[(&[Fp], usize)], window: usize) -> Digest {
    let (base, base_state) = baseline_forests(forests);
    let mut t = Transcript::new(&context());
    let cand = split_prove_forests(&mut t, forests, window);
    let cand_state = t.state();
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
    assert_eq!(
        rejected.as_ref().err().map(String::as_str),
        Some("level 0: FinalMismatch { round: 0 }")
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
    check_product_window(name, cases, 0)
}
fn check_product_window(name: &str, cases: &[ProductCase], window: usize) -> usize {
    let instances: Vec<Instance<'_, Fp2>> = cases.iter().map(ProductCase::baseline).collect();
    let mut t = Transcript::new(&context());
    let base = prove_batch(&mut t, Prefix::Stage(0), LEVEL_DEGREE, &instances)
        .expect("a well-formed batch");
    let base_state = t.state();
    let split: Vec<SplitInstance<'_>> = cases.iter().map(ProductCase::candidate).collect();
    let mut t = Transcript::new(&context());
    let cand = prove_split_batch_window(&mut t, Prefix::Stage(0), &split, window);
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
fn check_degenerate(rng: &mut SplitMix64, window: usize) {
    let n = 12;
    // (i) r_3 = 0.
    let mut case = ProductCase::random(rng, n);
    case.point[3] = Fp2::ZERO;
    case.eq = eq_table(&case.point).expect("a table");
    let fallbacks = check_product_window("degenerate (i) r_3 = 0", &[case], window);
    assert_eq!(fallbacks, 1, "(i): the fallback ran once, in round 3");
    // (ii) r_5 = 1.
    let mut case = ProductCase::random(rng, n);
    case.point[5] = Fp2::ONE;
    case.eq = eq_table(&case.point).expect("a table");
    let fallbacks = check_product_window("degenerate (ii) r_5 = 1", &[case], window);
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
    let cand = prove_split_batch_window(&mut t, Prefix::Stage(0), &split, window);
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

fn checks() {
    const P: u128 = 0xffff_ffff_0000_0001;
    for v in WINDOWS {
        let bound =
            cadence(v) as u128 * u64::MAX as u128 * (KEY_MAX as u128).pow(2) * 9u128.pow(v as u32);
        assert!(bound < (1u128 << 127));
        println!(
            "overflow v={v} cadence={} bound={bound} < 2^127",
            cadence(v)
        );
    }
    for x in [
        0,
        1,
        -1,
        i128::MAX,
        i128::MIN + 1,
        1i128 << 126,
        -(1i128 << 126),
    ] {
        let rem = x.unsigned_abs() % P;
        let expected = if x < 0 { (P - rem) % P } else { rem };
        assert_eq!(reduce_signed(x).to_canonical_u64() as u128, expected);
    }
    let mut rng = SplitMix64(CHECK_SEED);
    for _ in 0..10000 {
        let x = ((rng.next_u64() as u128) << 64 | rng.next_u64() as u128) as i128;
        let rem = x.unsigned_abs() % P;
        let expected = if x < 0 { (P - rem) % P } else { rem };
        assert_eq!(reduce_signed(x).to_canonical_u64() as u128, expected);
    }
    check_field();
    check_hand_case();
    check_small_sweep(&mut rng);
    check_two_forests(&mut rng);
    for v in WINDOWS {
        for n in 0..=12 {
            let case = ProductCase::random(&mut rng, n);
            check_product_window(&format!("window {v} n {n}"), &[case], v);
        }
        for (a, b) in [(1, 3), (4, 2), (0, 5), (7, 7), (3, 9), (9, 1)] {
            check_product_window(
                "front-padding",
                &[
                    ProductCase::random(&mut rng, a),
                    ProductCase::random(&mut rng, b),
                ],
                v,
            );
        }
        let mut maximum = ProductCase::random(&mut rng, 10);
        maximum.even.fill(Fp::from_u64_reduced(KEY_MAX));
        maximum.odd.fill(Fp::from_u64_reduced(KEY_MAX));
        check_product_window("maximum leaves", &[maximum], v);
        check_degenerate(&mut rng, v);
        let a = forest_leaves(&mut rng, 2, 8);
        let b = forest_leaves(&mut rng, 5, 6);
        check_window_forests("two forest windows", &[(&a, 8), (&b, 6)], v);
        let hand = [2, 3, 5, 7].map(Fp::from_u64_reduced);
        check_window_forests("hand windows", &[(&hand, 2)], v);
    }
    println!(
        "all correctness checks passed; exact bounds, signed reduction, windows, padding, degeneracies, tv-core acceptance"
    );
}

fn window_forests(forests: &[(&[Fp], usize)], v: usize) -> SplitForests {
    split_prove_forests(&mut Transcript::new(&context()), forests, v)
}

fn json_rows(path: &str, rows: &[Row]) {
    let mut out = String::from(
        "{\"experiment\":\"EXP-0012\",\"warmups\":3,\"samples\":10,\"minimum_sample_ms\":20,\"rows\":[\n",
    );
    for (i, r) in rows.iter().enumerate() {
        out.push_str(&format!("{{\"name\":\"{}\",\"elements\":{},\"per\":\"{}\",\"median_ns\":{},\"min_ns\":{},\"max_ns\":{},\"cv\":{},\"reps\":{},\"median_iteration_ms\":{},\"rerun\":{}}}{}\n",r.name,r.elements,r.per,r.stats.median*1e9/r.elements,r.stats.min*1e9/r.elements,r.stats.max*1e9/r.elements,r.stats.cv,r.reps,r.stats.median*r.reps as f64*1e3,r.stats.repeated,if i+1<rows.len(){","}else{""}));
    }
    out.push_str("]}\n");
    std::fs::write(path, out).expect("save JSON");
}

fn full_arms(rows: &mut Vec<Row>) {
    let mut rng = SplitMix64(LEAF_SEED);
    for (s, h) in [(8, 14), (10, 12), (0, 22), (6, 14), (7, 14)] {
        let leaves = forest_leaves(&mut rng, s, h);
        let forests = [(leaves.as_slice(), h)];
        let shape = format!("s{s}-h{h}");
        let n = leaves.len() as f64;
        for v in [0, 1, 3, 4, 5] {
            check_window_forests(&format!("{shape} v{v}"), &forests, v);
        }
        for v in [0, 1, 3, 4, 5, 99] {
            let name = if v == 99 {
                format!("forest {shape} production")
            } else {
                format!("forest {shape} v{v}")
            };
            let (stats, reps) = timed(|| {
                if v == 99 {
                    let _ = black_box(baseline_forests(black_box(&forests)));
                } else {
                    black_box(window_forests(black_box(&forests), v));
                }
            });
            report(rows, &name, n, "leaf", stats, reps);
        }
    }
    let mut rng = SplitMix64(ENGINE_SEED);
    let case = ProductCase::random(&mut rng, 21);
    for v in [0, 1, 3, 4, 5] {
        check_product_window(
            &format!("leaf diagnostic v{v}"),
            std::slice::from_ref(&case),
            v,
        );
        let instance = case.candidate();
        let (stats, reps) = timed(|| {
            black_box(prove_split_batch_window(
                &mut Transcript::new(&context()),
                Prefix::Stage(0),
                black_box(std::slice::from_ref(&instance)),
                v,
            ));
        });
        report(
            rows,
            &format!("leaf-level v{v}"),
            case.even.len() as f64,
            "pair",
            stats,
            reps,
        );
    }
    for v in WINDOWS {
        let (stats, reps) = timed(|| {
            black_box(Window::new(
                black_box(&case.even),
                black_box(&case.odd),
                black_box(&case.point),
                v,
            ));
        });
        report(
            rows,
            &format!("grid-formation-and-accumulation v{v}"),
            case.even.len() as f64,
            "pair",
            stats,
            reps,
        );
        let products: Vec<i64> = (0..4096)
            .map(|_| (rng.next_u64() & ((1 << 58) - 1)) as i64)
            .collect();
        let weights: Vec<[i128; 2]> = (0..4096)
            .map(|_| [rng.next_u64() as i128, rng.next_u64() as i128])
            .collect();
        let (stats, reps) = timed(|| {
            let mut total = Fp2::ZERO;
            for (ps, ws) in products.chunks(cadence(v)).zip(weights.chunks(cadence(v))) {
                let mut s = [0i128; 2];
                for (&p, w) in black_box(ps).iter().zip(black_box(ws)) {
                    s[0] += (p as i128) * w[0];
                    s[1] += (p as i128) * w[1];
                }
                total += Fp2::new(reduce_signed(s[0]), reduce_signed(s[1]));
            }
            black_box(total);
        });
        report(
            rows,
            &format!("i128-accumulation-microkernel v{v}"),
            4096.,
            "product",
            stats,
            reps,
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let command = args.get(1).map(String::as_str).unwrap_or("check");
    checks();
    fraction_checks();
    if command == "check" {
        return;
    }
    assert_eq!(
        command, "all",
        "usage: integer-window check | all results/run.json"
    );
    let mut rows = Vec::new();
    full_arms(&mut rows);
    fraction_arms(&mut rows);
    json_rows(args.get(2).expect("JSON path"), &rows);
}

fn field_grid(input: &[Fp2], width: usize, out: &mut [Fp2]) {
    out[..input.len()].copy_from_slice(input);
    let mut stride = 1;
    for k in 0..width {
        for g in (0..1usize << (width - k - 1)).rev() {
            for j in (0..stride).rev() {
                let a = out[g * 2 * stride + j];
                let b = out[g * 2 * stride + stride + j];
                out[g * 3 * stride + j] = a;
                out[g * 3 * stride + stride + j] = b;
                out[g * 3 * stride + 2 * stride + j] = b + b - a;
            }
        }
        stride *= 3;
    }
}

impl Window {
    fn new_fraction(even: &[Fp2], odd: &[Fp2], point: &[Fp2], width: usize, gamma: Fp2) -> Self {
        let m = point.len() - width;
        let lo = m.div_ceil(2);
        let eq_lo = eq_table(&point[width..width + lo]).expect("eq low");
        let eq_hi = eq_table(&point[width + lo..]).expect("eq high");
        let cells = 3usize.pow(width as u32);
        let block = 1 << width;
        let mut grid = vec![Fp2::ZERO; cells];
        let mut inner = grid.clone();
        let mut a = grid.clone();
        let mut b = grid.clone();
        let g2 = gamma * gamma;
        for (h, &wh) in eq_hi.iter().enumerate() {
            inner.fill(Fp2::ZERO);
            for (l, &wl) in eq_lo.iter().enumerate() {
                let index = (h * eq_lo.len() + l) * block;
                field_grid(&even[index..index + block], width, &mut a);
                field_grid(&odd[index..index + block], width, &mut b);
                for z in 0..cells {
                    inner[z] += wl * (a[z] * (b[z] + gamma) + g2 * b[z]);
                }
            }
            for z in 0..cells {
                grid[z] += wh * inner[z];
            }
        }
        Self {
            width,
            grid,
            challenges: Vec::with_capacity(width),
        }
    }

    fn bind_extension(&self, table: &[Fp2]) -> Vec<Fp2> {
        let w = eq_table(&self.challenges).expect("window bind");
        table
            .chunks_exact(w.len())
            .map(|chunk| {
                chunk
                    .iter()
                    .zip(&w)
                    .fold(Fp2::ZERO, |s, (&x, &w)| s + x * w)
            })
            .collect()
    }
}

struct FractionCase {
    even: Vec<Fp2>,
    odd: Vec<Fp2>,
    point: Vec<Fp2>,
    eq: Vec<Fp2>,
}
impl FractionCase {
    fn new(rng: &mut SplitMix64, n: usize) -> Self {
        let alpha = rng.fp2();
        let beta = rng.fp2();
        let mut denominator = || {
            alpha
                - Fp2::from(Fp::from_u64_reduced(rng.next_u64() & 0xffff_ffff))
                - beta.mul_by_base(Fp::from_u64_reduced(rng.next_u64() & 0xffff))
        };
        let even = (0..1 << n).map(|_| denominator()).collect();
        let odd = (0..1 << n).map(|_| denominator()).collect();
        let point = rng.fp2s(n);
        let eq = eq_table(&point).expect("eq");
        Self {
            even,
            odd,
            point,
            eq,
        }
    }
    fn instances(&self) -> [Instance<'_, Fp2>; 3] {
        [
            Instance::new(vec![
                Table::ext(&self.eq).public(),
                Table::ext(&self.even),
                Table::ext(&self.odd),
            ]),
            Instance::new(vec![Table::ext(&self.eq).public(), Table::ext(&self.even)]),
            Instance::new(vec![Table::ext(&self.eq).public(), Table::ext(&self.odd)]),
        ]
    }
    fn views(&self) -> [TvInstance<'_>; 3] {
        [
            TvInstance {
                eq_point: &self.point,
                eq: &self.eq,
                tables: vec![Values::Ext(&self.even), Values::Ext(&self.odd)],
            },
            TvInstance {
                eq_point: &self.point,
                eq: &self.eq,
                tables: vec![Values::Ext(&self.even)],
            },
            TvInstance {
                eq_point: &self.point,
                eq: &self.eq,
                tables: vec![Values::Ext(&self.odd)],
            },
        ]
    }
    fn candidate(&self) -> SplitInstance<'_> {
        SplitInstance {
            relation: Relation::FractionLeaf {
                d_e: &self.even,
                d_o: &self.odd,
            },
            point: self.point.clone(),
            claims: self.views().iter().map(TvInstance::prover_claim).collect(),
        }
    }
    fn check(&self, v: usize) {
        let mut t = Transcript::new(&context());
        let base =
            prove_batch(&mut t, Prefix::Stage(0), 3, &self.instances()).expect("fraction baseline");
        let bs = t.state();
        let instance = self.candidate();
        let mut t = Transcript::new(&context());
        let cand = prove_split_batch_window(&mut t, Prefix::Stage(0), &[instance], v);
        assert_eq!(base.proof(), &cand.proof);
        assert_eq!(bs, t.state());
        let (point, state) = verify_views("fraction window", 3, &self.views(), &cand.proof);
        assert_eq!(point, tv_fp2s(&cand.point));
        assert_eq!(state, t.state());
        println!(
            "check fraction v{v} pairs={} bit-identical production, independent verifier accepts, digest {}",
            self.even.len(),
            hex(&proofs_digest(&[cand.proof]))
        );
    }
}

fn fraction_checks() {
    let mut rng = SplitMix64(0x7E55_E7A0_0058_0003);
    for v in [2, 3, 4, 5] {
        for n in 0..=8 {
            FractionCase::new(&mut rng, n).check(v);
        }
        for x in [Fp2::ZERO, Fp2::ONE] {
            let mut case = FractionCase::new(&mut rng, 8);
            case.point[0] = x;
            case.eq = eq_table(&case.point).expect("eq");
            case.check(v);
        }
        let case = FractionCase::new(&mut rng, 8);
        let r = case.point[0];
        let c = (r - Fp2::ONE) * (r.double() - Fp2::ONE).inverse().expect("nonzero");
        let mut script = rng.fp2s(9);
        script[1] = c;
        let mut t = Scripted::new(script.clone());
        let base = prove_batch(&mut t, Prefix::Stage(0), 3, &case.instances()).expect("baseline");
        let absorbed = t.absorbed;
        let mut t = Scripted::new(script);
        let cand = prove_split_batch_window(&mut t, Prefix::Stage(0), &[case.candidate()], v);
        assert_eq!(base.proof(), &cand.proof);
        assert_eq!(absorbed, t.absorbed);
        assert_eq!(cand.fallbacks, 7);
        println!("check fraction v{v}: zero/one eq coordinates and scripted prior-eq-zero pass");
    }
}

fn fraction_arms(rows: &mut Vec<Row>) {
    let mut rng = SplitMix64(0x7E55_E7A0_0058_0003);
    for leaves_log in 20..=24 {
        let case = FractionCase::new(&mut rng, leaves_log - 1);
        for v in [0, 2, 3, 4, 5] {
            case.check(v);
            let instance = case.candidate();
            let (stats, reps) = timed(|| {
                black_box(prove_split_batch_window(
                    &mut Transcript::new(&context()),
                    Prefix::Stage(0),
                    black_box(std::slice::from_ref(&instance)),
                    v,
                ));
            });
            report(
                rows,
                &format!("fraction N2^{leaves_log} v{v}"),
                case.even.len() as f64,
                "pair",
                stats,
                reps,
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Arms
// ---------------------------------------------------------------------------------------------
