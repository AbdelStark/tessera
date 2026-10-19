// SPDX-License-Identifier: Apache-2.0 OR MIT

//! EXP-0006: the calibration harness for `c_π`, the cost per leaf of the product-tree leaf stage
//! (README.md in this directory).
//!
//! It measures the production kernels unmodified, by path dependency on `tessera-sumcheck` and
//! `tessera-poly`, as RFC-0020's gate measures them (3 warm-ups, 10 iterations, median, minimum,
//! maximum, coefficient of variation; a coefficient above 3% is flagged and the case rerun once),
//! and the in-cache unit costs that the report's fit `F` multiplies by the counts of `count.py`.
//! Every timed proof first passes the independent verifier's checks (`tv-core`), so that a fast
//! but wrong path cannot enter a number.
//!
//! Research code: it prints, it panics on a failed check, and nothing depends on it.
//!
//! ```sh
//! cargo run --release --locked -- all [results/run.json]
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::print_stdout, clippy::cast_precision_loss)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use tessera_codec::Ascii;
use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_poly::{PolyField, bind_lowest, bind_lowest_in_place, eq_table, evaluate_base};
use tessera_sumcheck::{
    Forest, Instance, Prefix, ProvenForests, SumcheckProof, Table, prove_batch, prove_forests,
};
use tv_core::sumcheck as tv;
use tv_core::tree::{TreeClaim, verify_product_level};
use tv_hash::transcript as tvt;

type Fp = Goldilocks;
type Fp2 = GoldilocksExt2;
type TvFp = tv_field::Fp;
type TvFp2 = tv_field::Fp2;

/// Leaves: the seed of `crates/tessera-sumcheck/benches/throughput.rs` `forests`.
const LEAF_SEED: u64 = 0x7E55_E7A0_0058_0001;
/// Engine tables and points.
const ENGINE_SEED: u64 = 0x7E55_E7A0_0058_0002;
/// The fraction-tree proxy's keys, values, `α` and `β`.
const FRACTION_SEED: u64 = 0x7E55_E7A0_0058_0003;
/// Unit microbenchmarks.
const UNIT_SEED: u64 = 0x7E55_E7A0_0058_0004;

/// `N = 2^22` leaves: RFC-0010's largest test case and the bench's size.
const LOG_N: usize = 22;
/// RFC-0020's `c_π` row shape: rows of 2^14 leaves.
const ROW_HEIGHT: usize = 14;
const LEVEL_DEGREE: usize = 3;

const WARMUPS: usize = 3;
const ITERATIONS: usize = 10;
const CV_LIMIT: f64 = 0.03;
/// In-cache microbenchmarks repeat their kernel until one iteration lasts at least this long.
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
// Conversions to the verifier's field
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

/// One reported case, for the JSON record.
struct Row {
    name: String,
    per: &'static str,
    elements: f64,
    stats: Stats,
}

/// Prints a case and returns its median in nanoseconds per unit.
fn report(rows: &mut Vec<Row>, name: &str, elements: f64, per: &'static str, stats: Stats) -> f64 {
    let ns = |t: f64| t * 1e9 / elements;
    let flag = if stats.cv > CV_LIMIT {
        "  FLAG: CV > 3% after a rerun"
    } else if stats.repeated {
        "  (rerun: first CV > 3%)"
    } else {
        ""
    };
    println!(
        "{name:<58} {:>9.3} ns/{per:<6} min {:>9.3}  max {:>9.3}  CV {:>5.2}%  ({:>9.3} ms){flag}",
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
        stats,
    });
    ns(stats.median)
}

// ---------------------------------------------------------------------------------------------
// Operation counts: the rules of count.py, computed from the sizes
// ---------------------------------------------------------------------------------------------

/// Operation counts in the units the harness measures.
#[derive(Clone, Copy, Debug, Default)]
struct Ops {
    fp_mul: f64,
    fp_add: f64,
    fp_sub: f64,
    fp2_mul: f64,
    fp2_mulb: f64,
    fp2_add: f64,
    fp2_sub: f64,
}

impl Ops {
    const fn new(
        fp_mul: f64,
        fp_add: f64,
        fp_sub: f64,
        fp2_mul: f64,
        fp2_mulb: f64,
        fp2_add: f64,
        fp2_sub: f64,
    ) -> Self {
        Self {
            fp_mul,
            fp_add,
            fp_sub,
            fp2_mul,
            fp2_mulb,
            fp2_add,
            fp2_sub,
        }
    }

    fn scale(self, k: f64) -> Self {
        Self::new(
            self.fp_mul * k,
            self.fp_add * k,
            self.fp_sub * k,
            self.fp2_mul * k,
            self.fp2_mulb * k,
            self.fp2_add * k,
            self.fp2_sub * k,
        )
    }

    fn plus(self, o: Self) -> Self {
        Self::new(
            self.fp_mul + o.fp_mul,
            self.fp_add + o.fp_add,
            self.fp_sub + o.fp_sub,
            self.fp2_mul + o.fp2_mul,
            self.fp2_mulb + o.fp2_mulb,
            self.fp2_add + o.fp2_add,
            self.fp2_sub + o.fp2_sub,
        )
    }

    /// Predicted nanoseconds from the measured unit costs.
    fn predict(&self, u: &Units) -> f64 {
        self.fp_mul * u.fp_mul
            + self.fp_add * u.fp_add
            + self.fp_sub * u.fp_sub
            + self.fp2_mul * u.fp2_mul
            + self.fp2_mulb * u.fp2_mulb
            + self.fp2_add * u.fp2_add
            + self.fp2_sub * u.fp2_sub
    }

    /// IDEA-0008's unit: Goldilocks multiplications, an `Fp2` multiplication counting 3 and a
    /// `mul_by_base` 2.
    fn mul_equivalents(&self) -> f64 {
        self.fp_mul + 3.0 * self.fp2_mul + 2.0 * self.fp2_mulb
    }
}

// Per-pair rules (count.py `RULES`): three nodes per round message.
/// `product_sums`, mixed path: eq over `Fp2` public, even and odd over `Fp`.
const ROUND1_MIXED: Ops = Ops::new(3.0, 4.0, 2.0, 0.0, 3.0, 5.0, 1.0);
/// `zip3` over `Fp2`.
const ZIP3_EXT: Ops = Ops::new(0.0, 0.0, 0.0, 6.0, 0.0, 9.0, 3.0);
/// `zip2` over `Fp2`.
const ZIP2_EXT: Ops = Ops::new(0.0, 0.0, 0.0, 3.0, 0.0, 7.0, 2.0);
/// `zip3` over `Fp`.
const ZIP3_BASE: Ops = Ops::new(6.0, 9.0, 3.0, 0.0, 0.0, 0.0, 0.0);
/// `bind_base`, per pair.
const BIND_BASE: Ops = Ops::new(0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0);
/// `bind_ext` / `bind_in_place` / `fold_in_place`, per pair.
const BIND_EXT: Ops = Ops::new(0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0);
/// `eq_table`, per entry.
const EQ_ENTRY: Ops = Ops::new(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0);
/// `level_above`, per entry produced.
const LEVEL_ENTRY: Ops = Ops::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

/// One level's sumcheck on tables of `m` entries (eq, even, odd).
fn count_level_instance(m: usize) -> Ops {
    let half = (m / 2) as f64;
    let rest = half - 1.0;
    ROUND1_MIXED
        .scale(half)
        .plus(BIND_BASE.scale(2.0 * half))
        .plus(BIND_EXT.scale(half))
        .plus(ZIP3_EXT.scale(rest))
        .plus(BIND_EXT.scale(3.0 * rest))
}

/// The table sizes of a forest's levels.
fn level_sizes(trees_log: usize, height: usize) -> Vec<usize> {
    (0..height)
        .map(|level| 1usize << (trees_log + level))
        .collect()
}

fn count_eq_tables(trees_log: usize, height: usize) -> Ops {
    level_sizes(trees_log, height)
        .iter()
        .fold(Ops::default(), |acc, &m| {
            acc.plus(EQ_ENTRY.scale((m - 1) as f64))
        })
}

fn count_build(trees_log: usize, height: usize) -> Ops {
    let n = 1usize << (trees_log + height);
    LEVEL_ENTRY.scale((n - (1usize << (trees_log + 1))) as f64)
}

fn count_forest(trees_log: usize, height: usize) -> Ops {
    level_sizes(trees_log, height)
        .iter()
        .fold(Ops::default(), |acc, &m| acc.plus(count_level_instance(m)))
        .plus(count_eq_tables(trees_log, height))
        .plus(count_build(trees_log, height))
}

fn count_ext3(m: usize) -> Ops {
    let pairs = (m - 1) as f64;
    ZIP3_EXT.scale(pairs).plus(BIND_EXT.scale(3.0 * pairs))
}

fn count_base3(m: usize) -> Ops {
    let half = (m / 2) as f64;
    ZIP3_BASE
        .scale(half)
        .plus(BIND_BASE.scale(3.0 * half))
        .plus(ZIP3_EXT.scale(half - 1.0))
        .plus(BIND_EXT.scale(3.0 * (half - 1.0)))
}

fn count_bind_pass(m: usize) -> Ops {
    BIND_BASE.scale((m / 2) as f64)
}

fn count_bind_every_round(m: usize) -> Ops {
    count_bind_pass(m).plus(BIND_EXT.scale((m / 2 - 1) as f64))
}

fn count_fraction_leaf_level(m: usize) -> Ops {
    let pairs = (m - 1) as f64;
    ZIP3_EXT
        .plus(ZIP2_EXT.scale(2.0))
        .scale(pairs)
        .plus(BIND_EXT.scale(7.0 * pairs))
}

fn count_fraction_generic_level(m: usize) -> Ops {
    let pairs = (m - 1) as f64;
    ZIP3_EXT
        .scale(3.0 * pairs)
        .plus(BIND_EXT.scale(9.0 * pairs))
}

fn count_fraction_build(n: usize) -> Ops {
    Ops::new(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0)
        .scale(n as f64)
        .plus(Ops::new(0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0).scale((n / 2) as f64))
        .plus(Ops::new(0.0, 0.0, 0.0, 3.0, 0.0, 1.0, 0.0).scale((n / 2 - 1) as f64))
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

/// `f̃(point) = Σ_x f[x]·eq(point, x)` of a table over `Fp`, in the verifier's field, with the
/// `eq` weights built by doubling from the definition.
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
/// specification around `tv_core::tree::verify_product_level`, as the sumcheck crate's tests do;
/// none of it calls the prover.
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

/// Proves `forests` on a fresh transcript and returns the result with the transcript's state.
fn prove_forest_batch(forests: &[Forest<'_, Fp2>]) -> (ProvenForests<Fp2>, [u8; 32]) {
    let mut transcript = Transcript::new(&context());
    let proven = prove_forests(&mut transcript, forests).expect("well-formed forests");
    (proven, transcript.state())
}

/// Proves the forests and checks the proof with the verifier driver, from the trees' true
/// products: the verifier accepts, the transcripts end in the same state, and every leaf claim
/// is the verifier's and the stacked leaves' multilinear extension at its point.
fn check_forests(name: &str, shapes: &[(usize, usize)], leaves: &[Vec<Fp>]) {
    let forests: Vec<Forest<'_, Fp2>> = leaves
        .iter()
        .zip(shapes)
        .map(|(leaves, &(_, h))| Forest::new(leaves, h))
        .collect();
    let (proven, state) = prove_forest_batch(&forests);
    let proofs: Vec<tv::SumcheckProof> = proven
        .levels()
        .iter()
        .map(|level| tv_proof(level.proof()))
        .collect();
    let roots: Vec<Vec<TvFp2>> = leaves
        .iter()
        .zip(shapes)
        .map(|(leaves, &(_, h))| true_roots(leaves, h))
        .collect();
    let verified = verify_forests(shapes, &roots, &proofs).unwrap_or_else(|rejected| {
        panic!("{name}: the verifier rejects an honest proof: {rejected}")
    });
    assert_eq!(
        verified.state, state,
        "{name}: the transcripts end in the same state"
    );
    assert_eq!(
        verified.leaf_claims.len(),
        leaves.len(),
        "{name}: a leaf claim per forest"
    );
    for (i, ((mine, theirs), leaves)) in proven
        .leaf_claims()
        .iter()
        .zip(&verified.leaf_claims)
        .zip(leaves)
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
    println!(
        "check forests:   {name}: verifier accepts from the true roots; transcripts agree; {} leaf claim(s) equal the leaves' MLE",
        leaves.len()
    );
}

/// The hand case: one tree of the leaves `[2, 3, 5, 7]`, root 210. Accepted against 210,
/// rejected against 211, so the check can fail.
fn check_hand_case() {
    let fp = Fp::from_u64_reduced;
    let leaves: Vec<Fp> = [2u64, 3, 5, 7].into_iter().map(fp).collect();
    check_forests(
        "tree [2, 3, 5, 7]",
        &[(0, 2)],
        std::slice::from_ref(&leaves),
    );
    let forests = [Forest::<Fp2>::new(&leaves, 2)];
    let (proven, _) = prove_forest_batch(&forests);
    let proofs: Vec<tv::SumcheckProof> = proven
        .levels()
        .iter()
        .map(|level| tv_proof(level.proof()))
        .collect();
    let roots = true_roots(&leaves, 2);
    assert_eq!(roots, vec![TvFp2::from(tv_fp(fp(210)))], "the root is 210");
    let wrong = vec![vec![TvFp2::from(tv_fp(fp(211)))]];
    let rejected = verify_forests(&[(0, 2)], &wrong, &proofs);
    assert!(
        rejected.is_err(),
        "the verifier accepts the proof against the public root 211"
    );
    println!(
        "check hand case: tree [2, 3, 5, 7]: root 210 accepted; against 211 rejected ({})",
        rejected.err().unwrap_or_default()
    );
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

    fn len(&self) -> usize {
        match self {
            Self::Base(v) => v.len(),
            Self::Ext(v) => v.len(),
        }
    }
}

/// One instance as the verifier sees it: its tables (the public `eq` first, when present, with
/// its point) and how many evaluations the proof sends for it.
struct TvInstance<'a> {
    eq_point: Option<&'a [Fp2]>,
    tables: Vec<Values<'a>>,
}

impl TvInstance<'_> {
    /// `Σ_x Π_k T_k[x]` over the hypercube, in the verifier's field.
    fn claim(&self) -> TvFp2 {
        let len = self.tables[0].len();
        (0..len).fold(TvFp2::ZERO, |sum, x| {
            sum + self.tables.iter().fold(TvFp2::ONE, |p, t| p * t.tv_at(x))
        })
    }

    fn vars(&self) -> usize {
        self.tables[0].len().trailing_zeros() as usize
    }

    /// Evaluations sent: every table but the public `eq`.
    fn evals(&self) -> usize {
        self.tables.len() - usize::from(self.eq_point.is_some())
    }
}

/// Proves `instances` on a fresh transcript under `stage/0` and checks the proof with the
/// verifier's `verify_batch` against claims summed over the hypercube in `tv-field`, the public
/// `eq` factor evaluated by the verifier's own `eq`.
fn check_batch(
    name: &str,
    degree: usize,
    instances: &[Instance<'_, Fp2>],
    views: &[TvInstance<'_>],
) {
    let mut transcript = Transcript::new(&context());
    let proven = prove_batch(&mut transcript, Prefix::Stage(0), degree, instances)
        .expect("a well-formed batch");
    let proof = tv_proof(proven.proof());
    let claims: Vec<tv::Instance> = views
        .iter()
        .map(|view| tv::Instance {
            claim: view.claim(),
            vars: view.vars(),
            evals: view.evals(),
        })
        .collect();
    let eq_points: Vec<Option<Vec<TvFp2>>> = views
        .iter()
        .map(|view| view.eq_point.map(tv_fp2s))
        .collect();
    let relation = |index: usize, point: &[TvFp2], evals: &[TvFp2]| -> Option<TvFp2> {
        let product = evals.iter().fold(TvFp2::ONE, |p, e| p * *e);
        match &eq_points[index] {
            Some(r) => Some(tv::eq(r, point)? * product),
            None => Some(product),
        }
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
    assert_eq!(tv_fp2s(proven.point()), batch.point, "{name}: the point");
    assert_eq!(
        *verifier.state(),
        transcript.state(),
        "{name}: the transcripts end in the same state"
    );
    println!(
        "check engine:    {name}: verify_batch accepts {} instance(s) of {} variables from hypercube sums",
        claims.len(),
        claims[0].vars
    );
}

/// `eq_table` against the definition at sampled vertices.
fn check_eq_table(point: &[Fp2], table: &[Fp2]) {
    let n = table.len();
    for s in 0..17usize {
        let x = if s == 16 { n - 1 } else { s * (n / 16) };
        let expected = point.iter().enumerate().fold(Fp2::ONE, |p, (i, &r)| {
            p * if (x >> i) & 1 == 1 { r } else { Fp2::ONE - r }
        });
        assert_eq!(table[x], expected, "eq_table entry {x}");
    }
}

// ---------------------------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------------------------

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

/// The harness's copy of `forest.rs` `build_levels`' arithmetic: the leaves de-interleaved into
/// their even and odd halves, then each level above as `even_t[z] = even_{t+1}[2z]·odd_{t+1}[2z]`,
/// `odd_t[z] = even_{t+1}[2z+1]·odd_{t+1}[2z+1]`, levels `h` down to 1 (the production function
/// is private to the crate). One multiplication per entry produced.
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

/// The fraction-tree proxy's values: leaves `d = α − key − β·value`, the first layer with
/// numerators 1 (`n = d_e + d_o`, `d = d_e·d_o`), and the layers above
/// (`n_L·d_R + n_R·d_L`, `d_L·d_R`) up to the root. Returns the leaves `d` and the root.
fn build_fraction_tree(
    keys: &[Fp],
    values: &[Fp],
    alpha: Fp2,
    beta: Fp2,
) -> (Vec<Fp2>, (Fp2, Fp2)) {
    let d: Vec<Fp2> = keys
        .iter()
        .zip(values)
        .map(|(&k, &v)| alpha - Fp2::from(k) - beta.mul_by_base(v))
        .collect();
    let (mut n, mut dd): (Vec<Fp2>, Vec<Fp2>) = d
        .chunks_exact(2)
        .map(|p| (p[0] + p[1], p[0] * p[1]))
        .unzip();
    while n.len() > 1 {
        let (nn, nd): (Vec<Fp2>, Vec<Fp2>) = n
            .chunks_exact(2)
            .zip(dd.chunks_exact(2))
            .map(|(n, d)| (n[0] * d[1] + n[1] * d[0], d[0] * d[1]))
            .unzip();
        n = nn;
        dd = nd;
    }
    (d, (n[0], dd[0]))
}

/// The fraction tree's root against the definition on a small instance: `n/d = Σ_i 1/d_i`.
fn check_fraction_tree(rng: &mut SplitMix64) {
    let keys = rng.fps(1 << 8);
    let values = rng.fps(1 << 8);
    let (alpha, beta) = (rng.fp2(), rng.fp2());
    let (d, (n, dd)) = build_fraction_tree(&keys, &values, alpha, beta);
    let sum = d.iter().fold(Fp2::ZERO, |s, x| {
        s + x.inverse().expect("a non-zero denominator")
    });
    assert_eq!(
        n,
        dd * sum,
        "fraction tree root n/d = Σ 1/d_i on 2^8 leaves"
    );
    println!("check fraction:  2^8-leaf fraction tree: root n/d equals Σ 1/d_i");
}

// ---------------------------------------------------------------------------------------------
// Unit costs (in cache)
// ---------------------------------------------------------------------------------------------

/// Nanoseconds per unit operation, measured on in-cache inputs (2^12 elements in place).
#[derive(Clone, Copy, Debug)]
struct Units {
    fp_mul: f64,
    fp_add: f64,
    fp_sub: f64,
    fp2_mul: f64,
    fp2_mulb: f64,
    fp2_add: f64,
    fp2_sub: f64,
}

fn unit_case<T: Copy>(
    rows: &mut Vec<Row>,
    name: &str,
    values: &mut [T],
    mut op: impl FnMut(&mut T),
) -> f64 {
    let reps = reps_for(|| {
        for x in black_box(&mut *values).iter_mut() {
            op(x);
        }
    });
    let stats = measure(|| {
        let start = Instant::now();
        for _ in 0..reps {
            for x in black_box(&mut *values).iter_mut() {
                op(x);
            }
        }
        start.elapsed() / reps
    });
    report(
        rows,
        &format!("unit: {name} (x{reps})"),
        values.len() as f64,
        "op",
        stats,
    )
}

fn units(rows: &mut Vec<Row>) -> Units {
    println!("\n== unit costs, in cache (2^12 elements in place by a constant; L1D 128 KiB) ==");
    let mut rng = SplitMix64(UNIT_SEED);
    let c = black_box(rng.fp());
    let c2 = black_box(rng.fp2());
    let mut fps = rng.fps(1 << 12);
    let mut fp2s = rng.fp2s(1 << 12);
    let fp_mul = unit_case(rows, "Fp mul", &mut fps, |x| *x *= c);
    let fp_add = unit_case(rows, "Fp add", &mut fps, |x| *x += c);
    let fp_sub = unit_case(rows, "Fp sub", &mut fps, |x| *x -= c);
    let fp2_mul = unit_case(rows, "Fp2 mul (Karatsuba, mul_small(7))", &mut fp2s, |x| {
        *x *= c2
    });
    let fp2_mulb = unit_case(rows, "Fp2 mul_by_base", &mut fp2s, |x| {
        *x = x.mul_by_base(c)
    });
    let fp2_add = unit_case(rows, "Fp2 add", &mut fp2s, |x| *x += c2);
    let fp2_sub = unit_case(rows, "Fp2 sub", &mut fp2s, |x| *x -= c2);
    black_box((&fps, &fp2s));

    // Memory bandwidth context: a streaming copy of 2^22 Fp2 elements (64 MiB).
    let a = rng.fp2s(1 << LOG_N);
    let mut b = vec![Fp2::ZERO; 1 << LOG_N];
    let stats = measure(|| {
        let start = Instant::now();
        black_box(&mut b).copy_from_slice(black_box(&a));
        start.elapsed()
    });
    report(
        rows,
        "stream copy 2^22 Fp2 (64 MiB read + 64 MiB write)",
        (1u64 << LOG_N) as f64,
        "elem",
        stats,
    );
    println!(
        "    {:.1} GB/s (read + write), one thread",
        2.0 * 64.0 * 1024.0 * 1024.0 / stats.median / 1e9
    );

    Units {
        fp_mul,
        fp_add,
        fp_sub,
        fp2_mul,
        fp2_mulb,
        fp2_add,
        fp2_sub,
    }
}

// ---------------------------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------------------------

/// A measured kernel with the count its prediction uses.
struct Kernel {
    name: String,
    /// Median nanoseconds per unit (leaf, entry or lookup).
    measured: f64,
    /// Operations per unit.
    ops: Ops,
    /// Whether the fit `F` is judged on it.
    in_fit: bool,
}

/// A batch of degree `degree` over `instances`, on a fresh transcript.
fn prove(degree: usize, instances: &[Instance<'_, Fp2>]) {
    let mut transcript = Transcript::new(&context());
    let proven = prove_batch(&mut transcript, Prefix::Stage(0), degree, instances)
        .expect("a well-formed batch");
    black_box(proven.into_proof());
}

fn forest_cases(rows: &mut Vec<Row>, kernels: &mut Vec<Kernel>) {
    println!("\n== prove_forests, one thread, per leaf ==");
    let mut rng = SplitMix64(LEAF_SEED);
    let shapes: [(usize, usize, bool); 5] = [
        (LOG_N - ROW_HEIGHT, ROW_HEIGHT, true),
        (10, 12, true),
        (0, LOG_N, true),
        (4, ROW_HEIGHT, false),
        (6, ROW_HEIGHT, false),
    ];
    for (trees_log, height, in_fit) in shapes {
        let leaves = forest_leaves(&mut rng, trees_log, height);
        let n = leaves.len();
        let name = format!(
            "prove_forests: 2^{trees_log} trees of 2^{height} leaves (2^{} leaves)",
            trees_log + height
        );
        check_forests(&name, &[(trees_log, height)], std::slice::from_ref(&leaves));
        let batch = [Forest::<Fp2>::new(&leaves, height)];
        let stats = measure(|| {
            let start = Instant::now();
            let (proven, _) = prove_forest_batch(black_box(&batch));
            black_box(proven.into_parts());
            start.elapsed()
        });
        let measured = report(rows, &name, n as f64, "leaf", stats);
        kernels.push(Kernel {
            name,
            measured,
            ops: count_forest(trees_log, height).scale(1.0 / n as f64),
            in_fit,
        });
    }
}

fn component_cases(rows: &mut Vec<Row>, kernels: &mut Vec<Kernel>) {
    println!("\n== the leaf stage's components, one thread (2^8 trees of 2^14 leaves) ==");
    let trees_log = LOG_N - ROW_HEIGHT;
    let n = 1usize << LOG_N;
    let mut rng = SplitMix64(ENGINE_SEED);

    // The eq tables of every level: at level ℓ the forest's instance has s + ℓ variables.
    let points: Vec<Vec<Fp2>> = (0..ROW_HEIGHT)
        .map(|level| rng.fp2s(trees_log + level))
        .collect();
    for point in &points {
        check_eq_table(point, &eq_table(point).expect("a table"));
    }
    println!(
        "check eq:        {} eq tables (2^{} to 2^{} entries) equal the definition at 17 sampled vertices each",
        points.len(),
        trees_log,
        LOG_N - 1
    );
    let stats = measure(|| {
        let start = Instant::now();
        for point in &points {
            black_box(eq_table(black_box(point)).expect("a table"));
        }
        start.elapsed()
    });
    let measured = report(rows, "eq tables of every level", n as f64, "leaf", stats);
    kernels.push(Kernel {
        name: "eq tables of every level".to_owned(),
        measured,
        ops: count_eq_tables(trees_log, ROW_HEIGHT).scale(1.0 / n as f64),
        in_fit: true,
    });

    // The level build (the harness's copy of `build_levels`), checked against the trees' products.
    let mut leaf_rng = SplitMix64(LEAF_SEED);
    let leaves = forest_leaves(&mut leaf_rng, trees_log, ROW_HEIGHT);
    let levels = build_levels(&leaves, ROW_HEIGHT);
    let (top_e, top_o) = levels.last().expect("a level");
    for (y, tree) in leaves.chunks_exact(1 << ROW_HEIGHT).enumerate() {
        let product = tree.iter().fold(Fp::ONE, |p, &x| p * x);
        assert_eq!(top_e[y] * top_o[y], product, "tree {y}: the built root");
    }
    println!(
        "check build:     levels of 2^{trees_log} trees of 2^{ROW_HEIGHT} leaves: every built root equals its tree's product (all zero here)"
    );
    let stats = measure(|| {
        let start = Instant::now();
        black_box(build_levels(black_box(&leaves), ROW_HEIGHT));
        start.elapsed()
    });
    let measured = report(
        rows,
        "level build (harness copy of build_levels)",
        n as f64,
        "leaf",
        stats,
    );
    kernels.push(Kernel {
        name: "level build".to_owned(),
        measured,
        ops: count_build(trees_log, ROW_HEIGHT).scale(1.0 / n as f64),
        in_fit: true,
    });

    // The leaf level's instance alone: eq public over Fp2, even and odd over Fp, 2^21 entries.
    let m = n / 2;
    let point = rng.fp2s(LOG_N - 1);
    let eq = eq_table(&point).expect("a table");
    let (even, odd) = &levels[0];
    let instance = Instance::new(vec![
        Table::ext(&eq).public(),
        Table::base(even),
        Table::base(odd),
    ]);
    let view = TvInstance {
        eq_point: Some(&point),
        tables: vec![Values::Ext(&eq), Values::Base(even), Values::Base(odd)],
    };
    check_batch(
        "leaf-level instance (eq x even x odd)",
        LEVEL_DEGREE,
        std::slice::from_ref(&instance),
        std::slice::from_ref(&view),
    );
    let stats = measure(|| {
        let start = Instant::now();
        prove(LEVEL_DEGREE, std::slice::from_ref(&instance));
        start.elapsed()
    });
    let measured = report(
        rows,
        "engine: eq public (Fp2) x even x odd (Fp), degree 3, 2^21",
        m as f64,
        "entry",
        stats,
    );
    kernels.push(Kernel {
        name: "engine: leaf-level instance, per entry".to_owned(),
        measured,
        ops: count_level_instance(m).scale(1.0 / m as f64),
        in_fit: true,
    });

    // Three Fp2 tables, degree 3.
    let ext: Vec<Vec<Fp2>> = (0..3).map(|_| rng.fp2s(m)).collect();
    let instance = Instance::new(ext.iter().map(|t| Table::ext(t)).collect());
    let view = TvInstance {
        eq_point: None,
        tables: ext.iter().map(|t| Values::Ext(t)).collect(),
    };
    check_batch(
        "three Fp2 tables",
        LEVEL_DEGREE,
        std::slice::from_ref(&instance),
        std::slice::from_ref(&view),
    );
    let stats = measure(|| {
        let start = Instant::now();
        prove(LEVEL_DEGREE, std::slice::from_ref(&instance));
        start.elapsed()
    });
    let measured = report(
        rows,
        "engine: product of 3 Fp2 tables, degree 3, 2^21",
        m as f64,
        "entry",
        stats,
    );
    kernels.push(Kernel {
        name: "engine: three Fp2 tables, per entry".to_owned(),
        measured,
        ops: count_ext3(m).scale(1.0 / m as f64),
        in_fit: true,
    });
    drop(ext);

    // Three Fp tables, degree 3.
    let base: Vec<Vec<Fp>> = (0..3).map(|_| rng.fps(m)).collect();
    let instance = Instance::new(base.iter().map(|t| Table::base(t)).collect());
    let view = TvInstance {
        eq_point: None,
        tables: base.iter().map(|t| Values::Base(t)).collect(),
    };
    check_batch(
        "three Fp tables",
        LEVEL_DEGREE,
        std::slice::from_ref(&instance),
        std::slice::from_ref(&view),
    );
    let stats = measure(|| {
        let start = Instant::now();
        prove(LEVEL_DEGREE, std::slice::from_ref(&instance));
        start.elapsed()
    });
    let measured = report(
        rows,
        "engine: product of 3 Fp tables, degree 3, 2^21",
        m as f64,
        "entry",
        stats,
    );
    kernels.push(Kernel {
        name: "engine: three Fp tables, per entry".to_owned(),
        measured,
        ops: count_base3(m).scale(1.0 / m as f64),
        in_fit: true,
    });
    drop(base);

    // Binding: one bind_lowest pass, and binding every round, over 2^22 Fp entries.
    let table = rng.fps(n);
    let r = rng.fp2s(LOG_N);
    let mut bound = bind_lowest(&table, r[0]).expect("a variable");
    for &rj in &r[1..] {
        bind_lowest_in_place(&mut bound, rj).expect("a variable");
    }
    assert_eq!(
        bound,
        vec![evaluate_base(&table, &r).expect("a point")],
        "binding every round equals evaluate_base"
    );
    println!(
        "check binding:   binding 2^{LOG_N} Fp entries round by round equals evaluate_base at the same point"
    );
    let stats = measure(|| {
        let start = Instant::now();
        black_box(bind_lowest(black_box(&table), r[0]).expect("a variable"));
        start.elapsed()
    });
    let measured = report(
        rows,
        "one bind_lowest pass, Fp into Fp2, 2^22",
        n as f64,
        "entry",
        stats,
    );
    kernels.push(Kernel {
        name: "bind_lowest pass, per entry".to_owned(),
        measured,
        ops: count_bind_pass(n).scale(1.0 / n as f64),
        in_fit: true,
    });
    let stats = measure(|| {
        let start = Instant::now();
        let mut bound = bind_lowest(black_box(&table), r[0]).expect("a variable");
        while bound.len() > 1 {
            bind_lowest_in_place(&mut bound, r[0]).expect("a variable");
        }
        black_box(bound);
        start.elapsed()
    });
    let measured = report(rows, "binding every round, 2^22", n as f64, "entry", stats);
    kernels.push(Kernel {
        name: "binding every round, per entry".to_owned(),
        measured,
        ops: count_bind_every_round(n).scale(1.0 / n as f64),
        in_fit: true,
    });
}

/// The fraction-tree proxy: build, leaf level (numerators 1), generic level; per lookup.
fn fraction_cases(rows: &mut Vec<Row>, kernels: &mut Vec<Kernel>) -> f64 {
    println!(
        "\n== fraction-tree proxy (RFC-0008 'Fraction tree' level relation on the engine), one thread =="
    );
    let mut rng = SplitMix64(FRACTION_SEED);
    check_fraction_tree(&mut rng);
    let n = 1usize << LOG_N;
    let m = n / 2;
    let keys = rng.fps(n);
    let values = rng.fps(n);
    let (alpha, beta) = (rng.fp2(), rng.fp2());
    let (d, _) = build_fraction_tree(&keys, &values, alpha, beta);
    let stats = measure(|| {
        let start = Instant::now();
        black_box(build_fraction_tree(
            black_box(&keys),
            black_box(&values),
            alpha,
            beta,
        ));
        start.elapsed()
    });
    let build = report(
        rows,
        "fraction proxy: build (leaves d, first layer, upper layers)",
        n as f64,
        "lookup",
        stats,
    );
    kernels.push(Kernel {
        name: "fraction proxy: build, per lookup".to_owned(),
        measured: build,
        ops: count_fraction_build(n).scale(1.0 / n as f64),
        in_fit: true,
    });

    // Leaf level, numerators 1: [eq, d_e, d_o], [eq, d_e], [eq, d_o].
    let (d_e, d_o): (Vec<Fp2>, Vec<Fp2>) = d.chunks_exact(2).map(|p| (p[0], p[1])).unzip();
    drop(d);
    let point = rng.fp2s(LOG_N - 1);
    let eq = eq_table(&point).expect("a table");
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
            eq_point: Some(&point),
            tables: vec![Values::Ext(&eq), Values::Ext(&d_e), Values::Ext(&d_o)],
        },
        TvInstance {
            eq_point: Some(&point),
            tables: vec![Values::Ext(&eq), Values::Ext(&d_e)],
        },
        TvInstance {
            eq_point: Some(&point),
            tables: vec![Values::Ext(&eq), Values::Ext(&d_o)],
        },
    ];
    check_batch(
        "fraction proxy: leaf level (numerators 1)",
        LEVEL_DEGREE,
        &instances,
        &views,
    );
    let stats = measure(|| {
        let start = Instant::now();
        prove(LEVEL_DEGREE, &instances);
        start.elapsed()
    });
    let leaf = report(
        rows,
        "fraction proxy: leaf level [eq,d_e,d_o] + [eq,d_e] + [eq,d_o], 2^21",
        m as f64,
        "entry",
        stats,
    );
    kernels.push(Kernel {
        name: "fraction proxy: leaf level, per entry".to_owned(),
        measured: leaf,
        ops: count_fraction_leaf_level(m).scale(1.0 / m as f64),
        in_fit: true,
    });
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
            eq_point: Some(&point),
            tables: vec![Values::Ext(&eq), Values::Ext(n_e), Values::Ext(g_o)],
        },
        TvInstance {
            eq_point: Some(&point),
            tables: vec![Values::Ext(&eq), Values::Ext(n_o), Values::Ext(g_e)],
        },
        TvInstance {
            eq_point: Some(&point),
            tables: vec![Values::Ext(&eq), Values::Ext(g_e), Values::Ext(g_o)],
        },
    ];
    check_batch(
        "fraction proxy: generic level",
        LEVEL_DEGREE,
        &instances,
        &views,
    );
    let stats = measure(|| {
        let start = Instant::now();
        prove(LEVEL_DEGREE, &instances);
        start.elapsed()
    });
    let generic = report(
        rows,
        "fraction proxy: generic level, 3 x [eq, Fp2, Fp2], 2^21",
        m as f64,
        "entry",
        stats,
    );
    kernels.push(Kernel {
        name: "fraction proxy: generic level, per entry".to_owned(),
        measured: generic,
        ops: count_fraction_generic_level(m).scale(1.0 / m as f64),
        in_fit: true,
    });
    let per_lookup = build + (leaf + generic) * (m as f64) / (n as f64);
    println!(
        "    fraction tree per lookup: build {build:.3} + (leaf level {leaf:.3} + generic level {generic:.3}) x 2^21 / 2^22 = {per_lookup:.3} ns"
    );
    per_lookup
}

// ---------------------------------------------------------------------------------------------
// The fit F and the pre-registered observations
// ---------------------------------------------------------------------------------------------

fn fit(units: &Units, kernels: &[Kernel]) -> (bool, Vec<(String, f64, f64, f64)>) {
    println!("\n== fit F: predicted = counts x unit costs ==");
    println!(
        "units (ns): Fp mul {:.3} add {:.3} sub {:.3} | Fp2 mul {:.3} mul_by_base {:.3} add {:.3} sub {:.3}",
        units.fp_mul,
        units.fp_add,
        units.fp_sub,
        units.fp2_mul,
        units.fp2_mulb,
        units.fp2_add,
        units.fp2_sub
    );
    let mut pass = true;
    let mut table = Vec::new();
    for k in kernels {
        let predicted = k.ops.predict(units);
        let ratio = predicted / k.measured;
        let ok = (0.5..=2.0).contains(&ratio);
        if k.in_fit && !ok {
            pass = false;
        }
        println!(
            "{:<58} measured {:>9.3}  predicted {:>9.3}  ratio {:>6.3}  {}{}",
            k.name,
            k.measured,
            predicted,
            ratio,
            if ok {
                "in [0.5, 2]"
            } else {
                "OUTSIDE [0.5, 2]"
            },
            if k.in_fit {
                ""
            } else {
                "  (context, not in F)"
            }
        );
        table.push((k.name.clone(), k.measured, predicted, ratio));
    }
    println!("F {}", if pass { "passes" } else { "FAILS" });
    (pass, table)
}

fn observations(kernels: &[Kernel], fraction_per_lookup: f64) {
    println!("\n== pre-registered observations (not part of the verdict) ==");
    let bind = kernels
        .iter()
        .find(|k| k.name.starts_with("binding every round"))
        .expect("the binding kernel");
    let kappa_bind = bind.measured / bind.ops.mul_equivalents();
    println!(
        "(a) kappa from binding every round: {:.3} ns per entry / {:.2} Fp-mul equivalents = {kappa_bind:.3} ns per equivalent",
        bind.measured,
        bind.ops.mul_equivalents()
    );
    for k in kernels
        .iter()
        .filter(|k| k.name.starts_with("prove_forests"))
    {
        let equivalents = k.ops.mul_equivalents();
        let kappa = k.measured / equivalents;
        println!(
            "    {:<58} {:.2} equivalents per leaf; {:.3} ns per leaf = {:.3} ns per equivalent; kappa_bind x count = {:.3} ns ({:+.1}% against measured){}",
            k.name,
            equivalents,
            k.measured,
            kappa,
            kappa_bind * equivalents,
            100.0 * (kappa_bind * equivalents / k.measured - 1.0),
            if (kappa_bind * equivalents / k.measured - 1.0).abs() <= 0.30 {
                "  within +-30%"
            } else {
                "  OUTSIDE +-30%"
            }
        );
    }
    let row = kernels
        .iter()
        .find(|k| k.name.starts_with("prove_forests: 2^8 trees"))
        .expect("the row-shaped forest");
    println!(
        "(b) c_pi / c_l shape proxy: product tree {:.3} ns per leaf / fraction tree {:.3} ns per lookup = {:.3} (IDEA-0008 expected 0.4-0.8)",
        row.measured,
        fraction_per_lookup,
        row.measured / fraction_per_lookup
    );
}

// ---------------------------------------------------------------------------------------------
// JSON record
// ---------------------------------------------------------------------------------------------

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn write_json(
    path: &str,
    rows: &[Row],
    fit: &[(String, f64, f64, f64)],
    pass: bool,
    units: &Units,
) {
    let mut out = String::from(
        "{\n  \"experiment\": \"EXP-0006\",\n  \"protocol\": {\"warmups\": 3, \"iterations\": 10, \"cv_limit\": 0.03},\n",
    );
    out.push_str(&format!(
        "  \"units_ns\": {{\"fp_mul\": {}, \"fp_add\": {}, \"fp_sub\": {}, \"fp2_mul\": {}, \"fp2_mulb\": {}, \"fp2_add\": {}, \"fp2_sub\": {}}},\n",
        units.fp_mul, units.fp_add, units.fp_sub, units.fp2_mul, units.fp2_mulb, units.fp2_add, units.fp2_sub
    ));
    out.push_str("  \"cases\": [\n");
    for (i, row) in rows.iter().enumerate() {
        let ns = |t: f64| t * 1e9 / row.elements;
        out.push_str(&format!(
            "    {{\"name\": \"{}\", \"per\": \"{}\", \"elements\": {}, \"median_ns\": {}, \"min_ns\": {}, \"max_ns\": {}, \"cv\": {}, \"median_ms\": {}, \"repeated\": {}}}{}\n",
            json_escape(&row.name),
            row.per,
            row.elements,
            ns(row.stats.median),
            ns(row.stats.min),
            ns(row.stats.max),
            row.stats.cv,
            row.stats.median * 1e3,
            row.stats.repeated,
            if i + 1 < rows.len() { "," } else { "" }
        ));
    }
    out.push_str("  ],\n  \"fit\": [\n");
    for (i, (name, measured, predicted, ratio)) in fit.iter().enumerate() {
        out.push_str(&format!(
            "    {{\"name\": \"{}\", \"measured_ns\": {measured}, \"predicted_ns\": {predicted}, \"ratio\": {ratio}}}{}\n",
            json_escape(name),
            if i + 1 < fit.len() { "," } else { "" }
        ));
    }
    out.push_str(&format!("  ],\n  \"fit_passes\": {pass}\n}}\n"));
    std::fs::write(path, out).expect("write the JSON record");
    println!("\nJSON record written to {path}");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let command = args.get(1).map_or("all", String::as_str);
    if command != "all" {
        eprintln!("usage: leafcost all [results/run.json]");
        std::process::exit(2);
    }
    println!(
        "EXP-0006 harness: production kernels of tessera-sumcheck / tessera-poly, one thread; RFC-0020 protocol (3 warm-ups, 10 iterations, median/min/max/CV, rerun once above 3%)"
    );
    println!("\n== correctness checks ==");
    check_field();
    check_hand_case();

    let mut rows = Vec::new();
    let mut kernels = Vec::new();
    let units = units(&mut rows);
    forest_cases(&mut rows, &mut kernels);
    component_cases(&mut rows, &mut kernels);
    let fraction_per_lookup = fraction_cases(&mut rows, &mut kernels);
    let (pass, table) = fit(&units, &kernels);
    observations(&kernels, fraction_per_lookup);
    if let Some(path) = args.get(2) {
        write_json(path, &rows, &table, pass, &units);
    }
}
