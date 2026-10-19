// SPDX-License-Identifier: Apache-2.0 OR MIT

//! EXP-0009: analytic padding blocks against the materialized power-of-two padding of product
//! forests (README.md in this directory; IDEA-0019).
//!
//! Five arms prove the same batches of product forests on the production transcript of
//! `tessera-hash`, under the production labels of `tessera-sumcheck`:
//!
//! - `production`: `tessera_sumcheck::prove_forests` on the materialized power-of-two leaf slice,
//!   today's prover, the anchor and the correctness reference;
//! - `dense`: this harness's prover with every leaf declared real, the production algorithm
//!   written with a loop per tree (the baseline of the verdict);
//! - `analytic`: the same prover told which leaves are real: only the real prefix of every tree's
//!   level is built, materialized, bound and folded, and every padded block contributes to each
//!   round message in closed form (the candidate);
//! - `dense-shared`, `analytic-shared`: the same two with the `eq` table read from the level's
//!   shared position table instead of being materialized per tree (context, IDEA-0010).
//!
//! Every harness arm must reproduce the production output bit for bit (round messages, final
//! evaluations, leaf claims and the transcript's final state), and the production proof of every
//! configuration passes the independent verifier's checks (`tv-core`), so that a fast but wrong
//! path cannot enter a number. Timing follows RFC-0020's protocol: 3 warm-ups, 10 iterations,
//! median, minimum, maximum, coefficient of variation; above 3% the arm is rerun once.
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
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::struct_excessive_bools
)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use tessera_codec::Ascii;
use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_poly::{PolyField, eq_table};
use tessera_sumcheck::{Claim, Forest, Prefix, SumcheckProof, SumcheckTranscript, prove_forests};
use tv_core::sumcheck as tv;
use tv_core::tree::{TreeClaim, verify_product_level};
use tv_hash::transcript as tvt;

type Fp = Goldilocks;
type Fp2 = GoldilocksExt2;
type TvFp = tv_field::Fp;
type TvFp2 = tv_field::Fp2;

/// The real leaves of every tree (and the zero leaf's position).
const LEAF_SEED: u64 = 0x7E55_E7A0_0009_0001;
/// The hand case and the `TAIL` check.
const CHECK_SEED: u64 = 0x7E55_E7A0_0009_0002;

/// The padded leaf of a product tree: the public constant 1 (issue #247: `leaf(q, i) =
/// C(q, i)·k_qi + (1 − C(q, i))`). Every padded node above it is 1 as well.
const KAPPA: Fp = Fp::ONE;

/// The degree of a product-tree level's sumcheck (RFC-0010 'Relations').
const LEVEL_DEGREE: usize = 3;

const WARMUPS: usize = 3;
const ITERATIONS: usize = 10;
const CV_LIMIT: f64 = 0.03;
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
#[derive(Clone, Debug)]
struct Stats {
    median: f64,
    min: f64,
    max: f64,
    cv: f64,
    /// The case was rerun because its first coefficient of variation exceeded 3%.
    repeated: bool,
    samples: Vec<(f64, usize)>,
    first_samples: Vec<(f64, usize)>,
}

fn sample_iteration(iteration: &mut impl FnMut() -> Duration) -> (f64, usize) {
    let mut elapsed = Duration::ZERO;
    let mut repetitions = 0;
    while elapsed < MIN_ITERATION {
        elapsed += iteration();
        repetitions += 1;
    }
    (elapsed.as_secs_f64(), repetitions)
}

fn sample(iteration: &mut impl FnMut() -> Duration) -> Stats {
    for _ in 0..WARMUPS {
        black_box(sample_iteration(iteration));
    }
    let samples: Vec<(f64, usize)> = (0..ITERATIONS)
        .map(|_| sample_iteration(iteration))
        .collect();
    let mut s: Vec<f64> = samples.iter().map(|(t, n)| t / *n as f64).collect();
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
        first_samples: Vec::new(),
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
    second.first_samples = first.samples;
    second
}

/// One timed case, for the report and the JSON record.
struct Row {
    config: String,
    arm: &'static str,
    real: usize,
    padded: usize,
    stats: Stats,
}

impl Row {
    fn ms(&self) -> f64 {
        self.stats.median * 1e3
    }

    fn ns_per_real(&self) -> f64 {
        self.stats.median * 1e9 / self.real as f64
    }

    fn ns_per_padded(&self) -> f64 {
        self.stats.median * 1e9 / self.padded as f64
    }
}

fn report(
    rows: &mut Vec<Row>,
    config: &str,
    arm: &'static str,
    real: usize,
    padded: usize,
    stats: Stats,
) {
    let flag = if stats.cv > CV_LIMIT {
        "  FLAG: CV > 3% after a rerun"
    } else if stats.repeated {
        "  (rerun: first CV > 3%)"
    } else {
        ""
    };
    let row = Row {
        config: config.to_owned(),
        arm,
        real,
        padded,
        stats: stats.clone(),
    };
    println!(
        "  {arm:<16} {:>9.3} ms  {:>8.3} ns/real  {:>8.3} ns/padded  min {:>9.3}  max {:>9.3} ms  CV {:>5.2}%{flag}",
        row.ms(),
        row.ns_per_real(),
        row.ns_per_padded(),
        stats.min * 1e3,
        stats.max * 1e3,
        stats.cv * 100.0
    );
    println!(
        "    samples (total seconds, complete proofs): {:?}; first attempt: {:?}",
        stats.samples, stats.first_samples
    );
    rows.push(row);
}

// ---------------------------------------------------------------------------------------------
// Shapes and inputs
// ---------------------------------------------------------------------------------------------

/// A forest's shape: `2^s` trees of height `h` after padding, the first `real_trees` of them
/// real with `real_leaves[y]` real leaves each (`1 ≤ real_leaves[y] ≤ 2^h`, leaves `0 … R − 1`
/// of the tree), the rest trees of ones.
#[derive(Clone)]
struct Shape {
    height: usize,
    trees_log: usize,
    real_trees: usize,
    real_leaves: Vec<usize>,
}

impl Shape {
    fn padded_len(&self) -> usize {
        1 << (self.trees_log + self.height)
    }

    fn real_len(&self) -> usize {
        self.real_leaves.iter().sum()
    }
}

/// A forest with its materialized power-of-two leaf slice: the real leaves of every real tree,
/// then ones.
struct ForestInput {
    shape: Shape,
    padded: Vec<Fp>,
}

/// A batch of forests proven in one `prove_forests` call.
struct Config {
    name: String,
    /// Whether the configuration is one of the four the verdict judges.
    judged: bool,
    forests: Vec<ForestInput>,
}

impl Config {
    fn real(&self) -> usize {
        self.forests.iter().map(|f| f.shape.real_len()).sum()
    }

    fn padded(&self) -> usize {
        self.forests.iter().map(|f| f.shape.padded_len()).sum()
    }

    fn factor(&self) -> f64 {
        self.padded() as f64 / self.real() as f64
    }
}

/// `2^⌈log2 n⌉`.
fn next_pow2(n: usize) -> usize {
    n.next_power_of_two()
}

/// The leaves of a shape from the seeded stream: every real tree's real leaves, with one zero
/// at a seeded real position as `SoftmaxRow` rows have; padded leaves and padded trees are
/// `KAPPA`.
fn make_forest(rng: &mut SplitMix64, shape: Shape) -> ForestInput {
    let h = shape.height;
    let mut padded = vec![KAPPA; shape.padded_len()];
    for (y, &r) in shape.real_leaves.iter().enumerate() {
        assert!(
            r >= 1 && r <= 1 << h,
            "a real tree has 1 to 2^h real leaves"
        );
        let base = y << h;
        for leaf in &mut padded[base..base + r] {
            *leaf = rng.fp();
        }
        let zero = (rng.next_u64() % r as u64) as usize;
        padded[base + zero] = Fp::ZERO;
    }
    ForestInput { shape, padded }
}

/// Causal rows `q = 0 … T − 1`, row `q` with `q + 1` real leaves, in one forest of height `log2 T`.
fn config_one_height(rng: &mut SplitMix64, name: &str, t: usize, judged: bool) -> Config {
    let n = t.trailing_zeros() as usize;
    assert_eq!(1 << n, t, "T is a power of two");
    let shape = Shape {
        height: n,
        trees_log: n,
        real_trees: t,
        real_leaves: (1..=t).collect(),
    };
    Config {
        name: name.to_owned(),
        judged,
        forests: vec![make_forest(rng, shape)],
    }
}

/// Causal rows grouped by height for `heads` head-layer pairs: height 1 holds rows 0 and 1,
/// height `h ≥ 2` the rows with `2^{h−1} < q + 1 ≤ 2^h`; each height's `heads × rows` trees are
/// padded to the next power of two. Tallest forest first.
fn config_grouped(
    rng: &mut SplitMix64,
    name: &str,
    t: usize,
    heads: usize,
    judged: bool,
) -> Config {
    let n = t.trailing_zeros() as usize;
    assert_eq!(1 << n, t, "T is a power of two");
    let mut forests = Vec::new();
    for h in (1..=n).rev() {
        let rows: Vec<usize> = if h == 1 {
            vec![1, 2]
        } else {
            ((1 << (h - 1)) + 1..=1 << h).collect()
        };
        let mut real_leaves = Vec::with_capacity(heads * rows.len());
        for _ in 0..heads {
            real_leaves.extend_from_slice(&rows);
        }
        let real_trees = real_leaves.len();
        let shape = Shape {
            height: h,
            trees_log: next_pow2(real_trees).trailing_zeros() as usize,
            real_trees,
            real_leaves,
        };
        forests.push(make_forest(rng, shape));
    }
    Config {
        name: name.to_owned(),
        judged,
        forests,
    }
}

/// `real_trees` full trees of `2^h` leaves padded to `2^s` trees of ones.
fn config_forest(
    rng: &mut SplitMix64,
    name: &str,
    trees_log: usize,
    real_trees: usize,
    height: usize,
    judged: bool,
) -> Config {
    assert!(real_trees <= 1 << trees_log);
    let shape = Shape {
        height,
        trees_log,
        real_trees,
        real_leaves: vec![1 << height; real_trees],
    };
    Config {
        name: name.to_owned(),
        judged,
        forests: vec![make_forest(rng, shape)],
    }
}

// ---------------------------------------------------------------------------------------------
// The independent checks (the tv-core driver of EXP-0006)
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

// ---------------------------------------------------------------------------------------------
// The output every arm must agree on
// ---------------------------------------------------------------------------------------------

/// Each level's messages, the leaf claims and the transcript's final state.
#[derive(PartialEq, Eq)]
struct Output {
    proofs: Vec<SumcheckProof<Fp2>>,
    claims: Vec<Claim<Fp2>>,
    state: [u8; 32],
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Today's prover on the materialized slices.
fn prove_production(config: &Config) -> Output {
    let forests: Vec<Forest<'_, Fp2>> = config
        .forests
        .iter()
        .map(|f| Forest::new(&f.padded, f.shape.height))
        .collect();
    let mut transcript = Transcript::new(&context());
    let proven = prove_forests(&mut transcript, &forests).expect("well-formed forests");
    let state = transcript.state();
    let (proofs, claims) = proven.into_parts();
    Output {
        proofs,
        claims,
        state,
    }
}

/// Proves `config` with the production prover and checks the proof with the verifier driver
/// from the true roots: the verifier accepts, the transcripts end in the same state, and every
/// leaf claim is the verifier's and the padded leaves' multilinear extension at its point.
fn check_production(config: &Config) -> Output {
    let out = prove_production(config);
    let shapes: Vec<(usize, usize)> = config
        .forests
        .iter()
        .map(|f| (f.shape.trees_log, f.shape.height))
        .collect();
    let roots: Vec<Vec<TvFp2>> = config
        .forests
        .iter()
        .map(|f| true_roots(&f.padded, f.shape.height))
        .collect();
    for (f, forest_roots) in config.forests.iter().zip(&roots) {
        for (y, &root) in forest_roots.iter().enumerate() {
            let expected = if y < f.shape.real_trees {
                TvFp2::ZERO
            } else {
                TvFp2::ONE
            };
            assert_eq!(
                root, expected,
                "{}: tree {y}: a real root is 0 and a padded root is 1",
                config.name
            );
        }
    }
    let proofs: Vec<tv::SumcheckProof> = out.proofs.iter().map(tv_proof).collect();
    let verified = verify_forests(&shapes, &roots, &proofs).unwrap_or_else(|rejected| {
        panic!(
            "{}: the verifier rejects an honest proof: {rejected}",
            config.name
        )
    });
    assert_eq!(
        verified.state, out.state,
        "{}: the transcripts end in the same state",
        config.name
    );
    assert_eq!(
        verified.leaf_claims.len(),
        out.claims.len(),
        "{}: a leaf claim per forest",
        config.name
    );
    for (i, ((mine, theirs), f)) in out
        .claims
        .iter()
        .zip(&verified.leaf_claims)
        .zip(&config.forests)
        .enumerate()
    {
        let point = tv_fp2s(&mine.point);
        assert_eq!(
            point, theirs.point,
            "{}: forest {i}: the leaf point",
            config.name
        );
        assert_eq!(
            tv_fp2(mine.value),
            theirs.value,
            "{}: forest {i}: the leaf value",
            config.name
        );
        assert_eq!(
            theirs.value,
            mle(&f.padded, &point),
            "{}: forest {i}: the padded leaves' extension at the leaf point",
            config.name
        );
    }
    println!(
        "  check: production proof accepted by the tv-core driver from the true roots (0 real, 1 padded); transcripts agree; {} leaf claim(s) equal the padded leaves' MLE; digest {}",
        out.claims.len(),
        hex(&out.state)
    );
    out
}

/// Asserts that `out` equals `reference`, naming the first difference.
fn assert_same(name: &str, arm: &str, out: &Output, reference: &Output) {
    assert_eq!(
        out.proofs.len(),
        reference.proofs.len(),
        "{name}/{arm}: the number of levels"
    );
    for (level, (mine, theirs)) in out.proofs.iter().zip(&reference.proofs).enumerate() {
        assert_eq!(
            mine.rounds.len(),
            theirs.rounds.len(),
            "{name}/{arm}: level {level}: the number of rounds"
        );
        for (j, (a, b)) in mine.rounds.iter().zip(&theirs.rounds).enumerate() {
            assert!(a == b, "{name}/{arm}: level {level}: round {j} differs");
        }
        assert!(
            mine.final_evals == theirs.final_evals,
            "{name}/{arm}: level {level}: the final evaluations differ"
        );
    }
    assert!(
        out.claims == reference.claims,
        "{name}/{arm}: the leaf claims differ"
    );
    assert_eq!(
        out.state, reference.state,
        "{name}/{arm}: the transcripts end in different states"
    );
}

// ---------------------------------------------------------------------------------------------
// The harness's forest prover: ragged tables with closed-form padding blocks
// ---------------------------------------------------------------------------------------------

/// Which prover an arm runs.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Arm {
    /// Only the real prefixes are built, bound and folded; padded blocks contribute in closed
    /// form. Off: every leaf is declared real (today's algorithm).
    analytic: bool,
    /// The `eq` table is read from the level's shared position table scaled per tree, instead of
    /// being materialized over each tree's real prefix.
    shared_eq: bool,
    amortized: bool,
}

const ARMS: [(&str, Arm); 5] = [
    (
        "dense",
        Arm {
            analytic: false,
            shared_eq: false,
            amortized: false,
        },
    ),
    (
        "analytic",
        Arm {
            analytic: true,
            shared_eq: false,
            amortized: false,
        },
    ),
    (
        "dense-shared",
        Arm {
            analytic: false,
            shared_eq: true,
            amortized: false,
        },
    ),
    (
        "analytic-shared",
        Arm {
            analytic: true,
            shared_eq: true,
            amortized: false,
        },
    ),
    (
        "analytic-amortized",
        Arm {
            analytic: true,
            shared_eq: false,
            amortized: true,
        },
    ),
];

/// The values of `a + z·(b − a)` at the `N` nodes, in entries `0..N`: `0, 2, 3, …, N` when
/// `SKIP_ONE` (a round message, which never carries `p(1)`), `0, 1, …, N − 1` otherwise. As
/// `kernel.rs` `line`: every value after the first two comes by adding `Δ = b − a`.
#[inline(always)]
fn line<E: Field, const N: usize, const SKIP_ONE: bool>(a: E, b: E) -> [E; N] {
    let delta = b - a;
    let mut values = [a; N];
    let mut value = if SKIP_ONE { b + delta } else { b };
    for slot in values.iter_mut().skip(1) {
        *slot = value;
        value += delta;
    }
    values
}

/// `TAIL(ρ, P) = Σ_{x ≥ P} eq(ρ, x)` over `|ρ|` coordinates, `0 ≤ P ≤ 2^{|ρ|}`: one minus the bit
/// recursion `LT(ρ, P) = Σ_{i: P_i = 1} (1 − ρ_i)·Π_{l > i} eq(ρ_l, P_l)` of RFC-0008 'Structured
/// evaluation' (`x < P` exactly when, at the highest bit where they differ, `x` has 0 and `P` has
/// 1, the lower bits free and summing to one). `O(|ρ|)` operations.
fn tail(rho: &[Fp2], p: usize) -> Fp2 {
    let n = rho.len();
    if p == 0 {
        return Fp2::ONE;
    }
    if p >= 1 << n {
        return Fp2::ZERO;
    }
    let mut lt = Fp2::ZERO;
    let mut suffix = Fp2::ONE;
    for i in (0..n).rev() {
        let one_minus = Fp2::ONE - rho[i];
        if (p >> i) & 1 == 1 {
            lt += one_minus * suffix;
            suffix *= rho[i];
        } else {
            suffix *= one_minus;
        }
    }
    Fp2::ONE - lt
}

/// The halves of one level of a forest, packed over its real trees: tree `y`'s even entries are
/// `even[e_off[y]..e_off[y + 1]]` and its odd entries `odd[o_off[y]..o_off[y + 1]]`; the entries
/// beyond are the constant `KAPPA`. Declared-dense forests fill every slot, so the layout is the
/// production's.
struct Level {
    even: Vec<Fp>,
    odd: Vec<Fp>,
    e_off: Vec<usize>,
    o_off: Vec<usize>,
}

impl Level {
    fn e_len(&self, y: usize) -> usize {
        self.e_off[y + 1] - self.e_off[y]
    }

    fn o_len(&self, y: usize) -> usize {
        self.o_off[y + 1] - self.o_off[y]
    }
}

/// Levels `1 … h` of a forest from its padded slice, at entries `1..=h` (entry 0 stays empty):
/// tree `y` has `declared[y]` real leaves, and the real count of a level is `⌈R/2⌉` of the level
/// below. The leaf level is de-interleaved in one pass; above it,
/// `V_t[z] = even_{t+1}[z]·odd_{t+1}[z]` for `z < R_t`, the one node with a padded child
/// multiplying by `KAPPA`, split into even and odd entries by the parity of `z`
/// (`forest.rs` `build_levels`, `leaf_level`, `level_above`).
fn build_levels(padded: &[Fp], height: usize, declared: &[usize]) -> Vec<Level> {
    let trees = declared.len();
    let mut levels: Vec<Level> = Vec::with_capacity(height + 1);
    // Entry 0: the roots, never built.
    levels.push(Level {
        even: Vec::new(),
        odd: Vec::new(),
        e_off: vec![0; trees + 1],
        o_off: vec![0; trees + 1],
    });
    for _ in 1..=height {
        levels.push(Level {
            even: Vec::new(),
            odd: Vec::new(),
            e_off: Vec::with_capacity(trees + 1),
            o_off: Vec::with_capacity(trees + 1),
        });
    }
    // The leaf level.
    {
        let total: usize = declared.iter().sum();
        let leaf = &mut levels[height];
        leaf.even.reserve_exact(total.div_ceil(2));
        leaf.odd.reserve_exact(total / 2);
        leaf.e_off.push(0);
        leaf.o_off.push(0);
        for (y, &r) in declared.iter().enumerate() {
            let slice = &padded[y << height..(y << height) + r];
            for pair in slice.chunks_exact(2) {
                leaf.even.push(pair[0]);
                leaf.odd.push(pair[1]);
            }
            if r % 2 == 1 {
                leaf.even.push(slice[r - 1]);
            }
            leaf.e_off.push(leaf.even.len());
            leaf.o_off.push(leaf.odd.len());
        }
    }
    // The levels above, from the one below each.
    for t in (1..height).rev() {
        let (upper, lower) = levels.split_at_mut(t + 1);
        let below = &lower[0];
        let level = &mut upper[t];
        let total: usize = (0..trees)
            .map(|y| (below.e_len(y) + below.o_len(y)).div_ceil(2))
            .sum();
        level.even.reserve_exact(total.div_ceil(2));
        level.odd.reserve_exact(total / 2);
        level.e_off.push(0);
        level.o_off.push(0);
        for y in 0..trees {
            let e = &below.even[below.e_off[y]..below.e_off[y + 1]];
            let o = &below.odd[below.o_off[y]..below.o_off[y + 1]];
            let full = e.len().min(o.len());
            for (ep, op) in e[..full].chunks_exact(2).zip(o[..full].chunks_exact(2)) {
                level.even.push(ep[0] * op[0]);
                level.odd.push(ep[1] * op[1]);
            }
            let mut z = full - full % 2;
            if z < full {
                level.even.push(e[z] * o[z]);
                z += 1;
            }
            if e.len() > o.len() {
                let v = e[full] * KAPPA;
                if z % 2 == 0 {
                    level.even.push(v);
                } else {
                    level.odd.push(v);
                }
            }
            level.e_off.push(level.even.len());
            level.o_off.push(level.odd.len());
        }
    }
    levels
}

/// The tables an instance reads in its first own round, over the prime subfield.
enum Source<'a> {
    /// A level's packed halves.
    Level(&'a Level),
    /// The tree-level tables of a forest at its root level (`m = 0`), one entry per real tree.
    Owned { even: Vec<Fp>, odd: Vec<Fp> },
}

/// A sub-table of an instance over `vars` free coordinates: a real tree over its position
/// coordinates (phase 1) or the whole forest over its tree coordinates (phase 2). Its even, odd
/// and `eq` tables are real prefixes into the instance's buffers; the entries beyond are the
/// constant `KAPPA` (even, odd) or `scale × shared[x]` (`eq`).
#[derive(Clone, Copy)]
struct Group {
    tree: usize,
    e_start: usize,
    e_len: usize,
    o_start: usize,
    o_len: usize,
    q_start: usize,
    q_len: usize,
    /// The group's `eq` factor over the coordinates the shared table does not carry: `eq_high(y)`
    /// in phase 1, `C_m` in phase 2.
    scale: Fp2,
    vars: usize,
}

/// Public lengths determine the traversal; the original tree index remains attached to a group.
fn equal_runs(groups: &[Group]) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut start = 0;
    while start < groups.len() {
        let g = groups[start];
        let mut end = start + 1;
        while end < groups.len()
            && (groups[end].e_len, groups[end].o_len, groups[end].q_len)
                == (g.e_len, g.o_len, g.q_len)
        {
            end += 1;
        }
        runs.push((start, end));
        start = end;
    }
    runs
}

/// Four independent tail folds expose lanes to LLVM without architecture-specific unsafe code.
fn odd_tails(values: [Fp2; 4], padding: [Fp2; 4], r: Fp2) -> [Fp2; 4] {
    std::array::from_fn(|lane| values[lane] + r * (padding[lane] - values[lane]))
}

/// Fold one of the three prefixes, with a run's odd boundary handled in four-tree batches.
fn fold_run_table(
    base: Option<&[Fp]>,
    bound: &[Fp2],
    groups: &[Group],
    runs: &[(usize, usize)],
    column: usize,
    shared: &[Fp2],
    r: Fp2,
) -> (Vec<Fp2>, Vec<(usize, usize)>) {
    let location = |g: &Group| match column {
        0 => (g.e_start, g.e_len),
        1 => (g.o_start, g.o_len),
        _ => (g.q_start, g.q_len),
    };
    let mut offsets = Vec::with_capacity(groups.len());
    let mut total = 0;
    for g in groups {
        let len = location(g).1.div_ceil(2);
        offsets.push((total, len));
        total += len;
    }
    let mut result = vec![Fp2::ZERO; total];
    for &(start, end) in runs {
        let len = location(&groups[start]).1;
        for (gi, g) in groups.iter().enumerate().take(end).skip(start) {
            let old = location(g).0;
            let target = offsets[gi].0;
            if let Some(base) = base {
                for (slot, pair) in result[target..target + len / 2]
                    .iter_mut()
                    .zip(base[old..old + len - len % 2].chunks_exact(2))
                {
                    *slot = r.mul_by_base(pair[1] - pair[0]).add_base(pair[0]);
                }
            } else {
                for (slot, pair) in result[target..target + len / 2]
                    .iter_mut()
                    .zip(bound[old..old + len - len % 2].chunks_exact(2))
                {
                    *slot = pair[0] + r * (pair[1] - pair[0]);
                }
            }
        }
        if len % 2 == 1 {
            for first in (start..end).step_by(4) {
                let lanes = (end - first).min(4);
                let mut values = [Fp2::ZERO; 4];
                let mut padding = [Fp2::from(KAPPA); 4];
                for lane in 0..lanes {
                    let g = groups[first + lane];
                    let old = location(&g).0 + len - 1;
                    values[lane] = base.map_or_else(|| bound[old], |b| Fp2::from(b[old]));
                    if column == 2 {
                        padding[lane] = g.scale * shared[len];
                    }
                }
                let folded = odd_tails(values, padding, r);
                for lane in 0..lanes {
                    result[offsets[first + lane].0 + len / 2] = folded[lane];
                }
            }
        }
    }
    (result, offsets)
}

fn fold_bound_run_table(
    table: &mut [Fp2],
    groups: &[Group],
    runs: &[(usize, usize)],
    column: usize,
    shared: &[Fp2],
    r: Fp2,
) {
    let location = |g: &Group| match column {
        0 => (g.e_start, g.e_len),
        1 => (g.o_start, g.o_len),
        _ => (g.q_start, g.q_len),
    };
    for &(start, end) in runs {
        let len = location(&groups[start]).1;
        for g in &groups[start..end] {
            let offset = location(g).0;
            for x in 0..len / 2 {
                let a = table[offset + 2 * x];
                let b = table[offset + 2 * x + 1];
                table[offset + x] = a + r * (b - a);
            }
        }
        if len % 2 == 1 {
            for first in (start..end).step_by(4) {
                let lanes = (end - first).min(4);
                let mut values = [Fp2::ZERO; 4];
                let mut padding = [Fp2::from(KAPPA); 4];
                for lane in 0..lanes {
                    let g = groups[first + lane];
                    values[lane] = table[location(&g).0 + len - 1];
                    if column == 2 {
                        padding[lane] = g.scale * shared[len];
                    }
                }
                let folded = odd_tails(values, padding, r);
                for lane in 0..lanes {
                    table[location(&groups[first + lane]).0 + len / 2] = folded[lane];
                }
            }
        }
    }
}

/// One forest's instance at one level of the batch, while the rounds run.
struct Inst<'a> {
    arm: Arm,
    /// Position coordinates (`t − 1` at tree-level `t`) and tree coordinates (`s`).
    m: usize,
    s: usize,
    /// The claim's point, lowest coordinate first: `m` position coordinates, then `s` tree ones.
    rho: Vec<Fp2>,
    real_trees: usize,
    base: Source<'a>,
    /// The bound tables after the first own round, packed over the groups.
    be: Vec<Fp2>,
    bo: Vec<Fp2>,
    /// The materialized `eq` prefixes, packed over the groups (empty in shared mode).
    eqb: Vec<Fp2>,
    groups: Vec<Group>,
    /// The shared `eq` tables over the position and the tree coordinates, folded as the rounds
    /// bind them, with the scalar each fold multiplies them by.
    eq_low: Vec<Fp2>,
    eq_low_scale: Fp2,
    eq_high: Vec<Fp2>,
    eq_high_scale: Fp2,
    phase2: bool,
    bound: bool,
    /// Own rounds done.
    own: usize,
    /// Batch bookkeeping, as `prover.rs` `State`: `n_max − n_i`, `γ^i`, `γ^i·S_i` and the first
    /// own round at every node for a padded instance.
    pad: usize,
    weight: Fp2,
    weighted_sum: Fp2,
    first: [Fp2; 4],
    memo: Vec<Option<Fp2>>,
}

impl<'a> Inst<'a> {
    fn bind_amortized(&mut self, r: Fp2) {
        let runs = equal_runs(&self.groups);
        if self.bound {
            fold_bound_run_table(&mut self.be, &self.groups, &runs, 0, &self.eq_low, r);
            fold_bound_run_table(&mut self.bo, &self.groups, &runs, 1, &self.eq_low, r);
            fold_bound_run_table(&mut self.eqb, &self.groups, &runs, 2, &self.eq_low, r);
            for g in &mut self.groups {
                g.e_len = g.e_len.div_ceil(2);
                g.o_len = g.o_len.div_ceil(2);
                g.q_len = g.q_len.div_ceil(2);
            }
            return;
        }
        let (be, e_offsets) = fold_run_table(
            if self.bound {
                None
            } else {
                Some(self.base_even())
            },
            &self.be,
            &self.groups,
            &runs,
            0,
            &self.eq_low,
            r,
        );
        let (bo, o_offsets) = fold_run_table(
            if self.bound {
                None
            } else {
                Some(self.base_odd())
            },
            &self.bo,
            &self.groups,
            &runs,
            1,
            &self.eq_low,
            r,
        );
        let (eqb, q_offsets) =
            fold_run_table(None, &self.eqb, &self.groups, &runs, 2, &self.eq_low, r);
        self.be = be;
        self.bo = bo;
        self.eqb = eqb;
        self.bound = true;
        for (i, g) in self.groups.iter_mut().enumerate() {
            (g.e_start, g.e_len) = e_offsets[i];
            (g.o_start, g.o_len) = o_offsets[i];
            (g.q_start, g.q_len) = q_offsets[i];
        }
    }
    /// The instance of a forest at tree-level `t = m + 1`, on the level's tables, with the claim's
    /// point `rho`.
    fn new(
        arm: Arm,
        level: &'a Level,
        m: usize,
        s: usize,
        real_trees: usize,
        rho: Vec<Fp2>,
    ) -> Self {
        assert_eq!(rho.len(), m + s);
        let eq_low = eq_table(&rho[..m]).expect("a table");
        let eq_high = eq_table(&rho[m..]).expect("a table");
        let mut eqb = Vec::new();
        let mut groups = Vec::with_capacity(real_trees);
        let (base, phase2) = if m > 0 {
            if !arm.shared_eq {
                let total: usize = (0..real_trees)
                    .map(|y| level.e_len(y).min(level.o_len(y)))
                    .sum();
                eqb.reserve_exact(total);
            }
            for (y, &scale) in eq_high.iter().enumerate().take(real_trees) {
                let q_start = eqb.len();
                let q_len = if arm.shared_eq {
                    0
                } else {
                    let len = level.e_len(y).min(level.o_len(y));
                    eqb.extend(eq_low[..len].iter().map(|&e| e * scale));
                    len
                };
                groups.push(Group {
                    tree: y,
                    e_start: level.e_off[y],
                    e_len: level.e_len(y),
                    o_start: level.o_off[y],
                    o_len: level.o_len(y),
                    q_start,
                    q_len,
                    scale,
                    vars: m,
                });
            }
            (Source::Level(level), false)
        } else {
            // The root level of the forest: one entry per real tree, the tree-level tables.
            let even: Vec<Fp> = (0..real_trees)
                .map(|y| {
                    if level.e_len(y) > 0 {
                        level.even[level.e_off[y]]
                    } else {
                        KAPPA
                    }
                })
                .collect();
            let odd: Vec<Fp> = (0..real_trees)
                .map(|y| {
                    if level.o_len(y) > 0 {
                        level.odd[level.o_off[y]]
                    } else {
                        KAPPA
                    }
                })
                .collect();
            let q_len = if arm.shared_eq {
                0
            } else {
                eqb.extend_from_slice(&eq_high[..real_trees]);
                real_trees
            };
            groups.push(Group {
                tree: 0,
                e_start: 0,
                e_len: real_trees,
                o_start: 0,
                o_len: real_trees,
                q_start: 0,
                q_len,
                scale: Fp2::ONE,
                vars: s,
            });
            (Source::Owned { even, odd }, true)
        };
        if arm.amortized && !phase2 {
            groups.sort_by_key(|g| (g.e_len, g.o_len, g.q_len));
        }
        Self {
            arm,
            m,
            s,
            rho,
            real_trees,
            base,
            be: Vec::new(),
            bo: Vec::new(),
            eqb,
            groups,
            eq_low,
            eq_low_scale: Fp2::ONE,
            eq_high,
            eq_high_scale: Fp2::ONE,
            phase2,
            bound: false,
            own: 0,
            pad: 0,
            weight: Fp2::ONE,
            weighted_sum: Fp2::ZERO,
            first: [Fp2::ZERO; 4],
            memo: Vec::new(),
        }
    }

    fn vars(&self) -> usize {
        self.m + self.s
    }

    fn base_even(&self) -> &[Fp] {
        match &self.base {
            Source::Level(level) => &level.even,
            Source::Owned { even, .. } => even,
        }
    }

    fn base_odd(&self) -> &[Fp] {
        match &self.base {
            Source::Level(level) => &level.odd,
            Source::Owned { odd, .. } => odd,
        }
    }

    fn base_e(&self, g: &Group, x: usize) -> Fp {
        if x < g.e_len {
            self.base_even()[g.e_start + x]
        } else {
            KAPPA
        }
    }

    fn base_o(&self, g: &Group, x: usize) -> Fp {
        if x < g.o_len {
            self.base_odd()[g.o_start + x]
        } else {
            KAPPA
        }
    }

    fn ext_e(&self, g: &Group, x: usize) -> Fp2 {
        if x < g.e_len {
            self.be[g.e_start + x]
        } else {
            Fp2::from(KAPPA)
        }
    }

    fn ext_o(&self, g: &Group, x: usize) -> Fp2 {
        if x < g.o_len {
            self.bo[g.o_start + x]
        } else {
            Fp2::from(KAPPA)
        }
    }

    /// The group's `eq` value at `x`: the materialized prefix, or `scale × shared[x]` beyond it;
    /// in shared mode `shared[x]` alone, the scale being applied to the group's sum.
    fn eq_at(&self, g: &Group, shared: &[Fp2], x: usize) -> Fp2 {
        if self.arm.shared_eq {
            shared[x]
        } else if x < g.q_len {
            self.eqb[g.q_start + x]
        } else {
            g.scale * shared[x]
        }
    }

    /// The shared `eq` table and its scalar for the current phase.
    fn shared(&self) -> (&[Fp2], Fp2) {
        if self.phase2 {
            (&self.eq_high, self.eq_high_scale)
        } else {
            (&self.eq_low, self.eq_low_scale)
        }
    }

    /// `Σ` over the group's pairs of `eq·even·odd` on the line through each pair, at the `N`
    /// nodes: the fully real pairs by the zipped loop, the one pair with a padded entry by the
    /// accessors, and the fully padded tail in closed form,
    /// `κ²·scale·shared_scale·eq(ρ_j, z)·TAIL(ρ_rest, P)`.
    fn group_sums<const N: usize, const SKIP_ONE: bool>(
        &mut self,
        gi: usize,
        rho_j: Fp2,
        rest_from: usize,
        run_shape: Option<(usize, usize)>,
    ) -> [Fp2; 4] {
        let g = self.groups[gi];
        let pairs = 1usize << (g.vars - 1);
        let q_pairs = if self.arm.shared_eq {
            pairs
        } else {
            g.q_len / 2
        };
        let (p_full, p_pad) = run_shape.unwrap_or_else(|| {
            (
                (g.e_len / 2).min(g.o_len / 2).min(q_pairs),
                g.e_len.div_ceil(2).max(g.o_len.div_ceil(2)),
            )
        });
        let (shared, shared_scale) = if self.phase2 {
            (&self.eq_high[..], self.eq_high_scale)
        } else {
            (&self.eq_low[..], self.eq_low_scale)
        };
        let mut sums = [Fp2::ZERO; 4];
        let eq_src: &[Fp2] = if self.arm.shared_eq {
            &shared[..2 * p_full]
        } else {
            &self.eqb[g.q_start..g.q_start + 2 * p_full]
        };
        if self.bound {
            let ev = &self.be[g.e_start..g.e_start + 2 * p_full];
            let od = &self.bo[g.o_start..g.o_start + 2 * p_full];
            for ((q, e), o) in eq_src
                .chunks_exact(2)
                .zip(ev.chunks_exact(2))
                .zip(od.chunks_exact(2))
            {
                let ql = line::<Fp2, N, SKIP_ONE>(q[0], q[1]);
                let el = line::<Fp2, N, SKIP_ONE>(e[0], e[1]);
                let ol = line::<Fp2, N, SKIP_ONE>(o[0], o[1]);
                for n in 0..N {
                    sums[n] += ql[n] * el[n] * ol[n];
                }
            }
        } else {
            let ev = &self.base_even()[g.e_start..g.e_start + 2 * p_full];
            let od = &self.base_odd()[g.o_start..g.o_start + 2 * p_full];
            for ((q, e), o) in eq_src
                .chunks_exact(2)
                .zip(ev.chunks_exact(2))
                .zip(od.chunks_exact(2))
            {
                let ql = line::<Fp2, N, SKIP_ONE>(q[0], q[1]);
                let el = line::<Fp, N, SKIP_ONE>(e[0], e[1]);
                let ol = line::<Fp, N, SKIP_ONE>(o[0], o[1]);
                for n in 0..N {
                    sums[n] += ql[n].mul_by_base(el[n] * ol[n]);
                }
            }
        }
        // The pairs with a padded entry (at most two).
        for x in p_full..p_pad {
            let ql = line::<Fp2, N, SKIP_ONE>(
                self.eq_at(&g, shared, 2 * x),
                self.eq_at(&g, shared, 2 * x + 1),
            );
            if self.bound {
                let el = line::<Fp2, N, SKIP_ONE>(self.ext_e(&g, 2 * x), self.ext_e(&g, 2 * x + 1));
                let ol = line::<Fp2, N, SKIP_ONE>(self.ext_o(&g, 2 * x), self.ext_o(&g, 2 * x + 1));
                for n in 0..N {
                    sums[n] += ql[n] * el[n] * ol[n];
                }
            } else {
                let el =
                    line::<Fp, N, SKIP_ONE>(self.base_e(&g, 2 * x), self.base_e(&g, 2 * x + 1));
                let ol =
                    line::<Fp, N, SKIP_ONE>(self.base_o(&g, 2 * x), self.base_o(&g, 2 * x + 1));
                for n in 0..N {
                    sums[n] += ql[n].mul_by_base(el[n] * ol[n]);
                }
            }
        }
        // The fully padded tail, in closed form.
        if p_pad < pairs && run_shape.is_none() {
            let t = match self.memo[p_pad] {
                Some(t) => t,
                None => {
                    let t = tail(&self.rho[rest_from..rest_from + (g.vars - 1)], p_pad);
                    self.memo[p_pad] = Some(t);
                    t
                }
            };
            let ql = line::<Fp2, N, SKIP_ONE>(Fp2::ONE - rho_j, rho_j);
            let mut factor = Fp2::from(KAPPA * KAPPA) * shared_scale * t;
            if !self.arm.shared_eq {
                factor *= g.scale;
            }
            for (sum, q) in sums.iter_mut().zip(ql) {
                *sum += factor * q;
            }
        }
        if self.arm.shared_eq {
            for sum in sums.iter_mut().take(N) {
                *sum *= g.scale;
            }
        }
        sums
    }

    /// The instance's sums at the `N` nodes of its current own round: every group, and in phase 1
    /// the padded trees, `κ²·C_j·eq(ρ_j, z)·TAIL(ρ_tree, N_trees)`.
    fn round_sums<const N: usize, const SKIP_ONE: bool>(&mut self) -> [Fp2; 4] {
        let j = self.own;
        let rho_j = self.rho[j];
        let vars = self.groups[0].vars;
        if !self.arm.amortized || self.phase2 {
            self.memo.clear();
            self.memo.resize(1 << (vars - 1), None);
            self.memo.push(None);
        }
        let mut sums = [Fp2::ZERO; 4];
        if self.arm.amortized && !self.phase2 {
            for (start, end) in equal_runs(&self.groups) {
                let g = self.groups[start];
                let full = (g.e_len / 2).min(g.o_len / 2).min(g.q_len / 2);
                let padded = g.e_len.div_ceil(2).max(g.o_len.div_ceil(2));
                for gi in start..end {
                    let s = self.group_sums::<N, SKIP_ONE>(gi, rho_j, j + 1, Some((full, padded)));
                    for n in 0..N {
                        sums[n] += s[n];
                    }
                }
                if padded < 1 << (g.vars - 1) {
                    let scale_sum = self.groups[start..end]
                        .iter()
                        .fold(Fp2::ZERO, |sum, tree| sum + tree.scale);
                    let coefficient = Fp2::from(KAPPA * KAPPA)
                        * self.eq_low_scale
                        * tail(&self.rho[j + 1..j + g.vars], padded)
                        * scale_sum;
                    let ql = line::<Fp2, N, SKIP_ONE>(Fp2::ONE - rho_j, rho_j);
                    for n in 0..N {
                        sums[n] += coefficient * ql[n];
                    }
                }
            }
        } else {
            for gi in 0..self.groups.len() {
                let s = self.group_sums::<N, SKIP_ONE>(gi, rho_j, j + 1, None);
                for n in 0..N {
                    sums[n] += s[n];
                }
            }
        }
        if !self.phase2 && self.real_trees < 1 << self.s {
            let t = tail(&self.rho[self.m..], self.real_trees);
            let ql = line::<Fp2, N, SKIP_ONE>(Fp2::ONE - rho_j, rho_j);
            let factor = Fp2::from(KAPPA * KAPPA) * self.eq_low_scale * t;
            for n in 0..N {
                sums[n] += factor * ql[n];
            }
        }
        sums
    }

    /// Prepares the instance for a batch of `n_max` variables with weight `γ^i`: for a padded
    /// instance, its sum over its own hypercube from its tables at every node, as `State::new`.
    fn prepare(&mut self, n_max: usize, weight: Fp2) {
        self.pad = n_max - self.vars();
        self.weight = weight;
        if self.pad > 0 {
            self.first = if self.vars() == 0 {
                let (u, w) = self.final_evals();
                let mut first = [Fp2::ZERO; 4];
                let (q, _) = self.shared();
                let product = q[0] * u * w;
                first[0] = product;
                first
            } else {
                self.round_sums::<4, false>()
            };
            self.weighted_sum = weight * (self.first[0] + self.first[1]);
        }
    }

    /// Adds `γ^i` times the instance's round polynomial at `0, 2, 3` into `message`, for round
    /// `j` of the batch (`State::add_round`).
    fn add_round(&mut self, j: usize, message: &mut [Fp2; 3]) {
        if j < self.pad {
            let value = self
                .weighted_sum
                .mul_by_base(Fp::from_u64_reduced(1 << (self.pad - 1 - j)));
            for entry in message.iter_mut() {
                *entry += value;
            }
            return;
        }
        let sums = if j == self.pad && self.pad > 0 {
            [self.first[0], self.first[2], self.first[3]]
        } else {
            let s = self.round_sums::<3, true>();
            [s[0], s[1], s[2]]
        };
        for (entry, sum) in message.iter_mut().zip(sums) {
            *entry += self.weight * sum;
        }
    }

    /// Binds the instance's lowest unbound variable to `r` after batch round `j`, if the round
    /// was one of its own: every group's prefixes fold (the first own round from the prime
    /// subfield into the bound buffers, as `bind_base`; later rounds in place), the materialized
    /// `eq` prefixes fold with `scale × shared[x]` past their end, then the shared table folds.
    fn bind(&mut self, j: usize, r: Fp2) {
        if j < self.pad {
            return;
        }
        if self.arm.amortized && !self.phase2 {
            self.bind_amortized(r);
        } else if !self.bound {
            let mut be = Vec::with_capacity(self.groups.iter().map(|g| g.e_len.div_ceil(2)).sum());
            let mut bo = Vec::with_capacity(self.groups.iter().map(|g| g.o_len.div_ceil(2)).sum());
            for gi in 0..self.groups.len() {
                let g = self.groups[gi];
                let e_start = be.len();
                for x in 0..g.e_len / 2 {
                    let (a, b) = (self.base_e(&g, 2 * x), self.base_e(&g, 2 * x + 1));
                    be.push(r.mul_by_base(b - a).add_base(a));
                }
                if g.e_len % 2 == 1 {
                    let a = self.base_e(&g, g.e_len - 1);
                    be.push(r.mul_by_base(KAPPA - a).add_base(a));
                }
                let o_start = bo.len();
                for x in 0..g.o_len / 2 {
                    let (a, b) = (self.base_o(&g, 2 * x), self.base_o(&g, 2 * x + 1));
                    bo.push(r.mul_by_base(b - a).add_base(a));
                }
                if g.o_len % 2 == 1 {
                    let a = self.base_o(&g, g.o_len - 1);
                    bo.push(r.mul_by_base(KAPPA - a).add_base(a));
                }
                let g = &mut self.groups[gi];
                g.e_start = e_start;
                g.e_len = g.e_len.div_ceil(2);
                g.o_start = o_start;
                g.o_len = g.o_len.div_ceil(2);
            }
            self.be = be;
            self.bo = bo;
            self.bound = true;
        } else {
            for gi in 0..self.groups.len() {
                let g = self.groups[gi];
                let half = g.e_len / 2;
                for x in 0..half {
                    let (a, b) = (self.be[g.e_start + 2 * x], self.be[g.e_start + 2 * x + 1]);
                    self.be[g.e_start + x] = a + r * (b - a);
                }
                if g.e_len % 2 == 1 {
                    let a = self.be[g.e_start + g.e_len - 1];
                    self.be[g.e_start + half] = a + r * (Fp2::from(KAPPA) - a);
                }
                let half = g.o_len / 2;
                for x in 0..half {
                    let (a, b) = (self.bo[g.o_start + 2 * x], self.bo[g.o_start + 2 * x + 1]);
                    self.bo[g.o_start + x] = a + r * (b - a);
                }
                if g.o_len % 2 == 1 {
                    let a = self.bo[g.o_start + g.o_len - 1];
                    self.bo[g.o_start + half] = a + r * (Fp2::from(KAPPA) - a);
                }
                let g = &mut self.groups[gi];
                g.e_len = g.e_len.div_ceil(2);
                g.o_len = g.o_len.div_ceil(2);
            }
        }
        // The materialized eq prefixes, with the shared table (not yet folded) past their end.
        if !self.arm.shared_eq && (!self.arm.amortized || self.phase2) {
            for gi in 0..self.groups.len() {
                let g = self.groups[gi];
                let half = g.q_len / 2;
                for x in 0..half {
                    let (a, b) = (self.eqb[g.q_start + 2 * x], self.eqb[g.q_start + 2 * x + 1]);
                    self.eqb[g.q_start + x] = a + r * (b - a);
                }
                if g.q_len % 2 == 1 {
                    let a = self.eqb[g.q_start + g.q_len - 1];
                    let shared: &[Fp2] = if self.phase2 {
                        &self.eq_high
                    } else {
                        &self.eq_low
                    };
                    let b = g.scale * shared[g.q_len];
                    self.eqb[g.q_start + half] = a + r * (b - a);
                }
                self.groups[gi].q_len = g.q_len.div_ceil(2);
            }
        }
        // The shared table folds, and its scalar picks up eq(ρ_j, r).
        let rho_j = self.rho[self.own];
        let factor = (Fp2::ONE - rho_j) * (Fp2::ONE - r) + rho_j * r;
        let table = if self.phase2 {
            &mut self.eq_high
        } else {
            &mut self.eq_low
        };
        let half = table.len() / 2;
        for x in 0..half {
            let (a, b) = (table[2 * x], table[2 * x + 1]);
            table[x] = a + r * (b - a);
        }
        table.truncate(half);
        if self.phase2 {
            self.eq_high_scale *= factor;
        } else {
            self.eq_low_scale *= factor;
        }
        for g in &mut self.groups {
            g.vars -= 1;
        }
        self.own += 1;
        if !self.phase2 && self.own == self.m {
            self.enter_tree_rounds();
        }
    }

    /// After the position rounds every tree is one entry: the tree-level tables, one group over
    /// the tree coordinates with the padded trees as its constant tail.
    fn enter_tree_rounds(&mut self) {
        let n = self.real_trees;
        if self.arm.amortized {
            self.groups.sort_unstable_by_key(|g| g.tree);
        }
        let mut be = Vec::with_capacity(n);
        let mut bo = Vec::with_capacity(n);
        let mut eqb = Vec::with_capacity(if self.arm.shared_eq { 0 } else { n });
        for g in &self.groups {
            be.push(self.ext_e(g, 0));
            bo.push(self.ext_o(g, 0));
            if !self.arm.shared_eq {
                eqb.push(self.eq_at(g, &self.eq_low, 0));
            }
        }
        self.be = be;
        self.bo = bo;
        self.eqb = eqb;
        self.groups = vec![Group {
            tree: 0,
            e_start: 0,
            e_len: n,
            o_start: 0,
            o_len: n,
            q_start: 0,
            q_len: if self.arm.shared_eq { 0 } else { n },
            scale: self.eq_low_scale,
            vars: self.s,
        }];
        self.phase2 = true;
    }

    /// `(u, w)`: the even and odd tables at the instance's point, after its last own round.
    fn final_evals(&self) -> (Fp2, Fp2) {
        assert!(self.phase2, "the tree rounds have run");
        let g = &self.groups[0];
        if self.bound {
            (self.ext_e(g, 0), self.ext_o(g, 0))
        } else {
            (Fp2::from(self.base_e(g, 0)), Fp2::from(self.base_o(g, 0)))
        }
    }
}

/// One level's batched sumcheck of degree 3 over `instances` under `tree/{tag}`, as
/// `prove_batch`: `γ`, the rounds, the final evaluations; returns the messages and the point.
fn prove_level(
    transcript: &mut Transcript,
    tag: u32,
    instances: &mut [Inst<'_>],
) -> (SumcheckProof<Fp2>, Vec<Fp2>) {
    let prefix = Prefix::Tree(tag);
    let gamma = transcript.challenge(&format!("{prefix}/gamma"));
    let n_max = instances.iter().map(Inst::vars).max().unwrap_or(0);
    let mut weight = Fp2::ONE;
    for inst in instances.iter_mut() {
        inst.prepare(n_max, weight);
        weight *= gamma;
    }
    let mut rounds = Vec::with_capacity(n_max);
    let mut point = Vec::with_capacity(n_max);
    for j in 0..n_max {
        let mut message = [Fp2::ZERO; LEVEL_DEGREE];
        for inst in instances.iter_mut() {
            inst.add_round(j, &mut message);
        }
        transcript.absorb_message(&format!("{prefix}/round/{j}"), &message);
        let r = transcript.challenge(&format!("{prefix}/r/{j}"));
        for inst in instances.iter_mut() {
            inst.bind(j, r);
        }
        rounds.push(message.to_vec());
        point.push(r);
    }
    let mut final_evals = Vec::with_capacity(2 * instances.len());
    for inst in instances.iter() {
        let (u, w) = inst.final_evals();
        final_evals.push(u);
        final_evals.push(w);
    }
    transcript.absorb_message(&format!("{prefix}/evals"), &final_evals);
    (
        SumcheckProof {
            rounds,
            final_evals,
        },
        point,
    )
}

/// A forest while its levels are proven.
struct ForestState {
    trees_log: usize,
    join: usize,
    real_trees: usize,
    levels: Vec<Level>,
    point: Vec<Fp2>,
}

/// The harness's `prove_forests`: the production schedule (root challenges, instances in forest
/// order, `γ`, the rounds, `ρ`, the next claims) over ragged tables. In the dense arms every leaf
/// is declared real.
fn prove_harness(config: &Config, arm: Arm) -> Output {
    let tallest = config
        .forests
        .iter()
        .map(|f| f.shape.height)
        .max()
        .expect("a forest");
    let mut states: Vec<ForestState> = config
        .forests
        .iter()
        .map(|f| {
            let shape = &f.shape;
            let (declared, real_trees) = if arm.analytic {
                (shape.real_leaves.clone(), shape.real_trees)
            } else {
                (
                    vec![1 << shape.height; 1 << shape.trees_log],
                    1 << shape.trees_log,
                )
            };
            ForestState {
                trees_log: shape.trees_log,
                join: tallest - shape.height,
                real_trees,
                levels: build_levels(&f.padded, shape.height, &declared),
                point: Vec::new(),
            }
        })
        .collect();
    let mut transcript = Transcript::new(&context());
    let mut proofs = Vec::with_capacity(tallest);
    let mut claims: Vec<Claim<Fp2>> = Vec::new();
    for level in 0..tallest {
        let tag = level as u32;
        let prefix = Prefix::Tree(tag);
        let s_max = states
            .iter()
            .filter(|state| state.join == level)
            .map(|state| state.trees_log)
            .max()
            .unwrap_or(0);
        let root_challenges: Vec<Fp2> = (0..s_max)
            .map(|c| transcript.challenge(&format!("{prefix}/root/{c}")))
            .collect();
        for state in states.iter_mut().filter(|state| state.join == level) {
            state.point = root_challenges[s_max - state.trees_log..].to_vec();
        }
        let active: Vec<usize> = (0..states.len())
            .filter(|&i| states[i].join <= level)
            .collect();
        let mut instances: Vec<Inst<'_>> = active
            .iter()
            .map(|&i| {
                let state = &states[i];
                let t = level - state.join + 1;
                Inst::new(
                    arm,
                    &state.levels[t],
                    t - 1,
                    state.trees_log,
                    state.real_trees,
                    state.point.clone(),
                )
            })
            .collect();
        let (proof, point) = prove_level(&mut transcript, tag, &mut instances);
        let vars: Vec<usize> = instances.iter().map(Inst::vars).collect();
        drop(instances);
        let rho = transcript.challenge(&format!("{prefix}/line"));
        let n_max = point.len();
        let mut next = Vec::with_capacity(active.len());
        for (k, &i) in active.iter().enumerate() {
            let own = &point[n_max - vars[k]..];
            let mut claim_point = Vec::with_capacity(own.len() + 1);
            claim_point.push(rho);
            claim_point.extend_from_slice(own);
            let (u, w) = (proof.final_evals[2 * k], proof.final_evals[2 * k + 1]);
            states[i].point.clone_from(&claim_point);
            let used = level - states[i].join + 1;
            states[i].levels[used] = Level {
                even: Vec::new(),
                odd: Vec::new(),
                e_off: Vec::new(),
                o_off: Vec::new(),
            };
            next.push(Claim {
                point: claim_point,
                value: u + rho * (w - u),
            });
        }
        claims = next;
        proofs.push(proof);
    }
    Output {
        proofs,
        claims,
        state: transcript.state(),
    }
}

// ---------------------------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------------------------

/// `TAIL` against the brute-force suffix sum of `eq_table` at 6 coordinates, for every `P`.
fn check_tail(rng: &mut SplitMix64) {
    let rho: Vec<Fp2> = (0..6).map(|_| rng.fp2()).collect();
    let table = eq_table(&rho).expect("a table");
    for p in 0..=table.len() {
        let brute = table[p..].iter().fold(Fp2::ZERO, |s, &e| s + e);
        assert_eq!(tail(&rho, p), brute, "TAIL(rho, {p})");
    }
    println!(
        "  check: TAIL(rho, P) equals the suffix sum of eq_table(rho) for every P at 6 coordinates"
    );
}

/// The hand case: rows `[2, 3]` and `[5, 7, 11]` in trees of height 2, padded with ones to
/// `[2, 3, 1, 1, 5, 7, 11, 1]`: accepted from the roots `(6, 385)`, rejected against `(6, 386)`;
/// every harness arm equals the production; and a padding of 0 would make every root 0.
fn check_hand_case() {
    let fp = Fp::from_u64_reduced;
    let shape = Shape {
        height: 2,
        trees_log: 1,
        real_trees: 2,
        real_leaves: vec![2, 3],
    };
    let padded: Vec<Fp> = [2u64, 3, 1, 1, 5, 7, 11, 1].into_iter().map(fp).collect();
    let config = Config {
        name: "hand".to_owned(),
        judged: false,
        forests: vec![ForestInput {
            shape: shape.clone(),
            padded: padded.clone(),
        }],
    };
    let out = prove_production(&config);
    let proofs: Vec<tv::SumcheckProof> = out.proofs.iter().map(tv_proof).collect();
    let roots = true_roots(&padded, 2);
    let six = TvFp2::from(tv_fp(fp(6)));
    let v385 = TvFp2::from(tv_fp(fp(385)));
    assert_eq!(
        roots,
        vec![six, v385],
        "the roots are the products of the real leaves, 6 and 385"
    );
    let verified =
        verify_forests(&[(1, 2)], &[roots], &proofs).expect("the honest proof is accepted");
    assert_eq!(verified.state, out.state);
    assert_eq!(
        verified.leaf_claims[0].value,
        mle(&padded, &verified.leaf_claims[0].point)
    );
    let wrong = vec![vec![six, TvFp2::from(tv_fp(fp(386)))]];
    let rejected = verify_forests(&[(1, 2)], &wrong, &proofs);
    assert_eq!(
        rejected.as_ref().err().map(String::as_str),
        Some("level 0: FinalMismatch { round: 1 }"),
        "the wrong root must fail at the independent sumcheck final check"
    );
    for (name, arm) in ARMS {
        let mine = prove_harness(&config, arm);
        assert_same("hand", name, &mine, &out);
    }
    let zero_padded: Vec<Fp> = [2u64, 3, 0, 0, 5, 7, 11, 0].into_iter().map(fp).collect();
    let zero_roots = true_roots(&zero_padded, 2);
    assert_eq!(zero_roots, vec![TvFp2::ZERO, TvFp2::ZERO]);
    println!(
        "  check: hand case [2, 3 | 1, 1] [5, 7, 11 | 1]: roots (6, 385) accepted, (6, 386) rejected ({}); all harness arms equal the production; a padding of 0 gives roots (0, 0)",
        rejected.err().unwrap_or_default()
    );
}

/// Small ragged shapes with every edge (rows of one leaf, trees with a single real node at a
/// level, padded trees, several forests joining at different levels): every arm equals the
/// production.
fn check_small_shapes(rng: &mut SplitMix64) {
    let configs = [
        config_one_height(rng, "small one-height T=16", 16, false),
        config_grouped(rng, "small grouped T=16", 16, 1, false),
        config_grouped(rng, "small 70B grouped T=8 x5", 8, 5, false),
        config_forest(rng, "small 70B forest 5 of 8 x 2^3", 3, 5, 3, false),
        config_forest(rng, "single tree 2^4", 0, 1, 4, false),
    ];
    for config in &configs {
        let out = check_production(config);
        for (name, arm) in ARMS {
            let mine = prove_harness(config, arm);
            assert_same(&config.name, name, &mine, &out);
        }
        println!(
            "  check: {}: all harness arms equal the production",
            config.name
        );
    }
}

fn check_amortized_shapes(rng: &mut SplitMix64) {
    for lengths in [
        vec![13, 1, 13, 7, 2, 7, 16],
        vec![3; 7],
        vec![1, 2, 3, 4, 5, 6, 7, 8],
    ] {
        let shape = Shape {
            height: 4,
            trees_log: 3,
            real_trees: lengths.len(),
            real_leaves: lengths,
        };
        let config = Config {
            name: "unsorted/repeated-length boundary case".to_owned(),
            judged: false,
            forests: vec![make_forest(rng, shape)],
        };
        let reference = check_production(&config);
        for (name, arm) in ARMS {
            assert_same(&config.name, name, &prove_harness(&config, arm), &reference);
        }
    }
    println!("  check: unsorted/repeated-length groups preserve original tree-coordinate order");
}

// ---------------------------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------------------------

fn time_config(rows: &mut Vec<Row>, config: &Config, amendment: bool) -> Output {
    let real = config.real();
    let padded = config.padded();
    println!(
        "\n== {}: {} forest(s), {} padded leaves, {} real, f = {:.5}{} ==",
        config.name,
        config.forests.len(),
        padded,
        real,
        config.factor(),
        if config.judged {
            " (judged)"
        } else {
            " (context)"
        }
    );
    let reference = check_production(config);
    for (name, arm) in ARMS {
        let mine = prove_harness(config, arm);
        assert_same(&config.name, name, &mine, &reference);
    }
    println!(
        "  check: all harness arms equal the production (proofs, leaf claims, transcript state)"
    );
    let stats = measure(|| {
        let start = Instant::now();
        black_box(prove_production(black_box(config)));
        start.elapsed()
    });
    report(rows, &config.name, "production", real, padded, stats);
    for (name, arm) in ARMS.into_iter().take(if amendment { 5 } else { 4 }) {
        let stats = measure(|| {
            let start = Instant::now();
            black_box(prove_harness(black_box(config), arm));
            start.elapsed()
        });
        report(rows, &config.name, name, real, padded, stats);
    }
    reference
}

fn find<'r>(rows: &'r [Row], config: &str, arm: &str) -> &'r Row {
    rows.iter()
        .find(|r| r.config == config && r.arm == arm)
        .expect("a timed row")
}

/// One configuration's ratios: name, `f`, `R_same`, `R_prod`, production / dense, the shared-`eq`
/// ratio, `a(c) / u` and the candidate's nanoseconds per real leaf.
type Ratios = (String, f64, f64, f64, f64, f64, f64, f64);

/// The pre-registered quantities and the rule's outcome, from the rows.
fn judge(rows: &[Row], configs: &[Config], amendment: bool) -> Vec<Ratios> {
    println!("\n== ratios (medians) ==");
    let u = find(rows, "U23", "dense").ns_per_real();
    println!("  u = dense ns per leaf on U23 = {u:.3}");
    let mut table = Vec::new();
    let mut validated = true;
    let mut refuted = false;
    let mut cv_flag = false;
    for config in configs {
        let name = &config.name;
        let f = config.factor();
        let prod = find(rows, name, "production");
        let dense = find(rows, name, "dense");
        let analytic = find(
            rows,
            name,
            if amendment {
                "analytic-amortized"
            } else {
                "analytic"
            },
        );
        let dense_sh = find(rows, name, "dense-shared");
        let analytic_sh = find(rows, name, "analytic-shared");
        let r_same = dense.stats.median / analytic.stats.median;
        let r_prod = prod.stats.median / analytic.stats.median;
        let loop_q = prod.stats.median / dense.stats.median;
        let r_shared = dense_sh.stats.median / analytic_sh.stats.median;
        let a_over_u = analytic.ns_per_real() / u;
        let band = if config.judged {
            let within = (0.90..=1.10).contains(&(r_same / f));
            let half = 1.0 + (f - 1.0) / 2.0;
            if r_same <= half {
                refuted = true;
                "REFUTED band (<= half recovered)"
            } else if within && a_over_u <= 1.05 {
                "validated band"
            } else {
                validated = false;
                "inconclusive band"
            }
        } else {
            "context"
        };
        if config.judged {
            for row in [dense, analytic] {
                if row.stats.cv > CV_LIMIT {
                    cv_flag = true;
                }
            }
        }
        println!(
            "  {name:<10} f {f:.4}  R_same {r_same:.4} (R_same/f {:.3})  R_prod {r_prod:.4}  prod/dense {loop_q:.3}  shared: dense/analytic {r_shared:.4}  a(c)/u {a_over_u:.3}  {band}",
            r_same / f
        );
        table.push((
            name.clone(),
            f,
            r_same,
            r_prod,
            loop_q,
            r_shared,
            a_over_u,
            analytic.ns_per_real(),
        ));
    }
    if find(rows, "U23", "dense").stats.cv > CV_LIMIT {
        cv_flag = true;
    }
    let outcome = if refuted {
        "refuted"
    } else if validated && !cv_flag {
        "validated"
    } else {
        "inconclusive"
    };
    println!(
        "  pre-registered rule over the judged configurations: {outcome}{}",
        if cv_flag {
            " (a CV above 3% on an arm a ratio needs)"
        } else {
            ""
        }
    );
    table
}

// ---------------------------------------------------------------------------------------------
// JSON record
// ---------------------------------------------------------------------------------------------

fn sample_json(samples: &[(f64, usize)]) -> String {
    format!(
        "[{}]",
        samples
            .iter()
            .map(|(seconds, repetitions)| format!("[{seconds}, {repetitions}]"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn write_json(path: &str, rows: &[Row], ratios: &[Ratios], digests: &[(String, String)]) {
    let mut out = String::from(
        "{\n  \"experiment\": \"EXP-0009\",\n  \"protocol\": {\"warmups\": 3, \"iterations\": 10, \"cv_limit\": 0.03, \"minimum_sample_ms\": 20},\n  \"cases\": [\n",
    );
    for (i, row) in rows.iter().enumerate() {
        out.push_str(&format!(
            "    {{\"config\": \"{}\", \"arm\": \"{}\", \"real_leaves\": {}, \"padded_leaves\": {}, \"median_ms\": {}, \"min_ms\": {}, \"max_ms\": {}, \"cv\": {}, \"ns_per_real_leaf\": {}, \"ns_per_padded_leaf\": {}, \"repeated\": {}, \"samples_total_seconds_and_repetitions\": {}, \"first_attempt_total_seconds_and_repetitions\": {}}}{}\n",
            row.config,
            row.arm,
            row.real,
            row.padded,
            row.ms(),
            row.stats.min * 1e3,
            row.stats.max * 1e3,
            row.stats.cv,
            row.ns_per_real(),
            row.ns_per_padded(),
            row.stats.repeated,
            sample_json(&row.stats.samples),
            sample_json(&row.stats.first_samples),
            if i + 1 < rows.len() { "," } else { "" }
        ));
    }
    out.push_str("  ],\n  \"ratios\": [\n");
    for (i, (name, f, r_same, r_prod, loop_q, r_shared, a_over_u, a)) in ratios.iter().enumerate() {
        out.push_str(&format!(
            "    {{\"config\": \"{name}\", \"f\": {f}, \"R_same\": {r_same}, \"R_prod\": {r_prod}, \"production_over_dense\": {loop_q}, \"R_shared\": {r_shared}, \"a_over_u\": {a_over_u}, \"analytic_ns_per_real_leaf\": {a}}}{}\n",
            if i + 1 < ratios.len() { "," } else { "" }
        ));
    }
    out.push_str("  ],\n  \"digests\": [\n");
    for (i, (name, digest)) in digests.iter().enumerate() {
        out.push_str(&format!(
            "    {{\"config\": \"{name}\", \"transcript_state\": \"{digest}\"}}{}\n",
            if i + 1 < digests.len() { "," } else { "" }
        ));
    }
    out.push_str("  ]\n}\n");
    std::fs::write(path, out).expect("write the JSON record");
    println!("\nJSON record written to {path}");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let command = args.get(1).map_or("all", String::as_str);
    let amendment = command == "amendment";
    if !["all", "amendment", "check"].contains(&command) {
        eprintln!("usage: padblocks all|amendment|check [results/run.json]");
        std::process::exit(2);
    }
    println!(
        "EXP-0009 harness: production prove_forests against the harness's dense and analytic forest provers, one thread; RFC-0020 protocol (3 warm-ups, 10 iterations, median/min/max/CV, rerun once above 3%)"
    );
    println!("\n== correctness checks ==");
    let mut check_rng = SplitMix64(CHECK_SEED);
    check_tail(&mut check_rng);
    check_hand_case();
    check_small_shapes(&mut check_rng);
    check_amortized_shapes(&mut check_rng);
    if command == "check" {
        return;
    }

    let mut rng = SplitMix64(LEAF_SEED);
    let configs = vec![
        config_one_height(&mut rng, "S1", 4096, true),
        config_grouped(&mut rng, "S2", 4096, 1, true),
        config_forest(&mut rng, "S3", 10, 640, 14, true),
        config_grouped(&mut rng, "S4", 2048, 5, true),
        config_forest(&mut rng, "U23", 9, 512, 14, false),
        config_forest(&mut rng, "U24", 10, 1024, 14, false),
        config_one_height(&mut rng, "S1-1K", 1024, false),
        config_grouped(&mut rng, "S2-1K", 1024, 1, false),
    ];
    let mut rows = Vec::new();
    let mut digests = Vec::new();
    for config in &configs {
        let reference = time_config(&mut rows, config, amendment);
        digests.push((config.name.clone(), hex(&reference.state)));
    }
    let ratios = judge(&rows, &configs, amendment);
    if let Some(path) = args.get(2) {
        write_json(path, &rows, &ratios, &digests);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amendment_preserves_transcripts_and_tree_order() {
        let mut rng = SplitMix64(CHECK_SEED);
        check_tail(&mut rng);
        check_hand_case();
        check_small_shapes(&mut rng);
        check_amortized_shapes(&mut rng);
    }
}
