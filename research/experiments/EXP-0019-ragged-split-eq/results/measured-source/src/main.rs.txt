// SPDX-License-Identifier: Apache-2.0 OR MIT

//! EXP-0019: committed preregistration in README.md; synthetic inputs only.
//! Input generation and independent verification driver retained from pinned EXP-0009.

#![forbid(unsafe_code)]
#![allow(
    clippy::print_stdout,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::struct_excessive_bools
)]

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::hint::black_box;
use std::time::{Duration, Instant};

use tessera_codec::{Ascii, SecretBuf, Zeroize};
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
        1usize
            .checked_shl(
                u32::try_from(self.trees_log.checked_add(self.height).expect("shape bits"))
                    .expect("u32 bits"),
            )
            .expect("shape shift")
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
        assert!(r <= 1 << h, "a real tree has 1 to 2^h real leaves");
        let base = y << h;
        for leaf in &mut padded[base..base + r] {
            *leaf = rng.fp();
        }
        if r > 0 {
            let zero = (rng.next_u64() % r as u64) as usize;
            padded[base + zero] = Fp::ZERO;
        }
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
        "  check: production proof accepted by the tv-core driver from the true roots (independent products); transcripts agree; {} leaf claim(s) equal the padded leaves' MLE; digest {}",
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

mod bench;
mod kernel;
use kernel::tail;

fn prove_candidate(config: &Config) -> Output {
    let mut transcript = Transcript::new(&context());
    let (proofs, claims, _) = kernel::prove(config, &mut transcript);
    Output {
        proofs,
        claims,
        state: transcript.state(),
    }
}

fn check_output(config: &Config, out: &Output) {
    let shapes = config
        .forests
        .iter()
        .map(|f| (f.shape.trees_log, f.shape.height))
        .collect::<Vec<_>>();
    let roots = config
        .forests
        .iter()
        .map(|f| true_roots(&f.padded, f.shape.height))
        .collect::<Vec<_>>();
    let proofs = out.proofs.iter().map(tv_proof).collect::<Vec<_>>();
    let accepted =
        verify_forests(&shapes, &roots, &proofs).expect("independent product-level acceptance");
    assert_eq!(accepted.state, out.state);
    for ((claim, checked), f) in out
        .claims
        .iter()
        .zip(accepted.leaf_claims)
        .zip(&config.forests)
    {
        assert_eq!(tv_fp2s(&claim.point), checked.point);
        assert_eq!(tv_fp2(claim.value), checked.value);
        assert_eq!(checked.value, mle(&f.padded, &checked.point));
    }
}
fn check_pair(config: &Config) -> String {
    let a = check_production(config);
    let b = prove_candidate(config);
    assert_same(&config.name, "ragged-split-eq", &b, &a);
    check_output(config, &b);
    hex(&a.state)
}
fn configs() -> Vec<Config> {
    let mut rng = SplitMix64(LEAF_SEED);
    vec![
        config_one_height(&mut rng, "S1", 4096, true),
        config_grouped(&mut rng, "S2", 4096, 1, true),
        config_forest(&mut rng, "S3", 10, 640, 14, true),
        config_grouped(&mut rng, "S4", 2048, 5, true),
        config_forest(&mut rng, "U23", 9, 512, 14, false),
        config_forest(&mut rng, "U24", 10, 1024, 14, false),
        config_one_height(&mut rng, "S1-1K", 1024, false),
        config_grouped(&mut rng, "S2-1K", 1024, 1, false),
    ]
}

/// Public scripted challenges exercise events too rare for random tests. They are NOT FS proofs.
struct Script {
    inner: Transcript,
    mode: u8,
    labels: Vec<String>,
}
impl Script {
    fn new(mode: u8) -> Self {
        Self {
            inner: Transcript::new(&context()),
            mode,
            labels: Vec::new(),
        }
    }
}
impl SumcheckTranscript<Fp2> for Script {
    fn absorb_message(&mut self, label: &str, message: &[Fp2]) {
        self.inner.absorb_message(label, message);
    }
    fn challenge(&mut self, label: &str) -> Fp2 {
        assert!(
            !label.contains("lambda"),
            "product-only schedule must not squeeze lambda"
        );
        self.labels.push(label.to_owned());
        let raw = self.inner.challenge(label);
        if self.mode == 0 {
            return raw;
        }
        let two = Fp2::from(Fp::from_u64_reduced(2));
        match self.mode {
            1 => {
                if label.contains("/root/") || label.ends_with("/line") {
                    Fp2::ZERO
                } else {
                    two
                }
            }
            2 => Fp2::ONE,
            3 => {
                if label.contains("/root/") || label.ends_with("/line") {
                    two
                } else if label.ends_with("/r/0") {
                    Fp2::from(Fp::from_u64_reduced(3))
                        .inverse()
                        .expect("3 invertible")
                } else {
                    two
                }
            }
            4 => {
                if label.ends_with("/line") {
                    Fp2::ZERO
                } else {
                    two
                }
            }
            5 => {
                if label.ends_with("/line") {
                    Fp2::ONE
                } else {
                    two
                }
            }
            _ => panic!("script mode"),
        }
    }
}
/// Independently defined script in verifier arithmetic: no prover field, fold, eq or interpolation.
fn tv_script(mode: u8, label: &str) -> TvFp2 {
    let n = if mode == 1 && (label.contains("/root/") || label.ends_with("/line"))
        || mode == 4 && label.ends_with("/line")
    {
        0
    } else if mode == 2 || mode == 5 && label.ends_with("/line") {
        1
    } else {
        2
    };
    if mode == 3 && label.ends_with("/r/0") {
        TvFp2::from(TvFp::from_canonical(3).expect("3"))
            .inverse()
            .expect("3 inverse")
    } else {
        TvFp2::from(TvFp::from_canonical(n).expect("small"))
    }
}
fn tv_interpolate(values: &[TvFp2; 4], r: TvFp2) -> TvFp2 {
    let nodes = [0u64, 1, 2, 3].map(|v| TvFp2::from(TvFp::from_canonical(v).expect("node")));
    let mut out = TvFp2::ZERO;
    for i in 0..4 {
        let mut n = TvFp2::ONE;
        let mut d = TvFp2::ONE;
        for j in 0..4 {
            if i != j {
                n *= r - nodes[j];
                d *= nodes[i] - nodes[j];
            }
        }
        out += values[i] * n * d.inverse().expect("distinct nodes");
    }
    out
}
/// Independent scripted verifier because tv-core's injectable transcript entrypoint is private.
/// Normal proofs of BOTH arms still use public tv-core::verify_product_level, above.
fn check_scripted_independent(config: &Config, out: &Output, mode: u8) {
    let tallest = config
        .forests
        .iter()
        .map(|f| f.shape.height)
        .max()
        .expect("height");
    let mut claims: Vec<Option<TreeClaim>> = vec![None; config.forests.len()];
    for (level, proof) in out.proofs.iter().enumerate() {
        let joining = (0..config.forests.len())
            .filter(|&i| tallest - config.forests[i].shape.height == level)
            .collect::<Vec<_>>();
        let smax = joining
            .iter()
            .map(|&i| config.forests[i].shape.trees_log)
            .max()
            .unwrap_or(0);
        let root = (0..smax)
            .map(|c| tv_script(mode, &format!("tree/{level}/root/{c}")))
            .collect::<Vec<_>>();
        for &i in &joining {
            let f = &config.forests[i];
            let point = root[smax - f.shape.trees_log..].to_vec();
            let value = root_claim(&true_roots(&f.padded, f.shape.height), &point);
            claims[i] = Some(TreeClaim { point, value });
        }
        let active = (0..config.forests.len())
            .filter(|&i| tallest - config.forests[i].shape.height <= level)
            .collect::<Vec<_>>();
        let max = active
            .iter()
            .map(|&i| claims[i].as_ref().expect("claim").point.len())
            .max()
            .unwrap_or(0);
        assert_eq!(proof.rounds.len(), max);
        let gamma = tv_script(mode, &format!("tree/{level}/gamma"));
        let mut weight = TvFp2::ONE;
        let mut claim = TvFp2::ZERO;
        for &i in &active {
            let c = claims[i].as_ref().expect("claim");
            claim += weight
                * c.value
                * TvFp2::from(
                    TvFp::from_canonical(1u64 << (max - c.point.len())).expect("padding"),
                );
            weight *= gamma;
        }
        let mut challenges = Vec::new();
        for (j, message) in proof.rounds.iter().enumerate() {
            assert_eq!(message.len(), 3);
            let p0 = tv_fp2(message[0]);
            let vals = [p0, claim - p0, tv_fp2(message[1]), tv_fp2(message[2])];
            let c = tv_script(mode, &format!("tree/{level}/r/{j}"));
            claim = tv_interpolate(&vals, c);
            challenges.push(c);
        }
        assert_eq!(proof.final_evals.len(), 2 * active.len());
        let mut expected = TvFp2::ZERO;
        weight = TvFp2::ONE;
        let rho = tv_script(mode, &format!("tree/{level}/line"));
        for (k, &i) in active.iter().enumerate() {
            let before = claims[i].as_ref().expect("claim");
            let own = &challenges[max - before.point.len()..];
            let eq = before
                .point
                .iter()
                .zip(own)
                .fold(TvFp2::ONE, |p, (&a, &b)| {
                    p * ((TvFp2::ONE - a) * (TvFp2::ONE - b) + a * b)
                });
            let (u, w) = (
                tv_fp2(proof.final_evals[2 * k]),
                tv_fp2(proof.final_evals[2 * k + 1]),
            );
            expected += weight * eq * u * w;
            weight *= gamma;
            let mut point = vec![rho];
            point.extend_from_slice(own);
            claims[i] = Some(TreeClaim {
                point,
                value: u + rho * (w - u),
            });
        }
        assert_eq!(
            claim, expected,
            "script {mode}, level {level}: independent final relation"
        );
    }
    for ((claim, expected), f) in claims.into_iter().zip(&out.claims).zip(&config.forests) {
        let c = claim.expect("joined");
        assert_eq!(c.point, tv_fp2s(&expected.point));
        assert_eq!(c.value, tv_fp2(expected.value));
        assert_eq!(c.value, mle(&f.padded, &c.point));
    }
}
fn check_scripts(config: &Config) {
    for mode in 0..=5 {
        let mut ta = Script::new(mode);
        let forests = config
            .forests
            .iter()
            .map(|f| Forest::new(&f.padded, f.shape.height))
            .collect::<Vec<_>>();
        let a = prove_forests(&mut ta, &forests).expect("shape");
        let (pa, ca) = a.into_parts();
        let a = Output {
            proofs: pa,
            claims: ca,
            state: ta.inner.state(),
        };
        let mut tb = Script::new(mode);
        let (pb, cb, fallbacks) = kernel::prove(config, &mut tb);
        let b = Output {
            proofs: pb,
            claims: cb,
            state: tb.inner.state(),
        };
        assert_same(&config.name, &format!("script-{mode}"), &b, &a);
        assert_eq!(ta.labels, tb.labels);
        if mode > 0 {
            check_scripted_independent(config, &a, mode);
            check_scripted_independent(config, &b, mode);
        } else {
            check_output(config, &a);
            check_output(config, &b);
        }
        println!(
            "  script {mode}: full outputs, no-lambda schedule and independent arithmetic agree; fallback rounds {fallbacks}, digest {}",
            hex(&b.state)
        );
    }
}
fn small_checks() {
    let mut rng = SplitMix64(CHECK_SEED);
    let rho = (0..6).map(|_| rng.fp2()).collect::<Vec<_>>();
    let weights = eq_table(&rho).expect("eq");
    for p in 0..=64 {
        assert_eq!(
            tail(&rho, p),
            weights[p..].iter().copied().fold(Fp2::ZERO, |a, b| a + b)
        );
    }
    let hand = Config {
        name: "hand".into(),
        judged: false,
        forests: vec![ForestInput {
            shape: Shape {
                height: 2,
                trees_log: 1,
                real_trees: 2,
                real_leaves: vec![2, 3],
            },
            padded: [2, 3, 1, 1, 5, 7, 11, 1].map(Fp::from_u64_reduced).to_vec(),
        }],
    };
    check_pair(&hand);
    let a = prove_production(&hand);
    let roots = true_roots(&hand.forests[0].padded, 2);
    assert_eq!(
        roots,
        vec![
            TvFp2::from(TvFp::from_canonical(6).expect("6")),
            TvFp2::from(TvFp::from_canonical(385).expect("385"))
        ]
    );
    let wrong = vec![vec![
        roots[0],
        TvFp2::from(TvFp::from_canonical(386).expect("386")),
    ]];
    let proof = a.proofs.iter().map(tv_proof).collect::<Vec<_>>();
    assert_eq!(
        verify_forests(&[(1, 2)], &wrong, &proof).err().as_deref(),
        Some("level 0: FinalMismatch { round: 1 }")
    );
    println!("  wrong root: level 0: FinalMismatch {{ round: 1 }}");
    let zero = [2, 3, 0, 0, 5, 7, 11, 0].map(Fp::from_u64_reduced);
    assert_eq!(true_roots(&zero, 2), vec![TvFp2::ZERO; 2]);
    for c in [
        config_one_height(&mut rng, "small one-height T=16", 16, false),
        config_grouped(&mut rng, "small grouped T=16", 16, 1, false),
        config_grouped(&mut rng, "small 70B grouped T=8 x5", 8, 5, false),
        config_forest(&mut rng, "small 70B forest5of8", 3, 5, 3, false),
        config_forest(&mut rng, "single tree2^4", 0, 1, 4, false),
    ] {
        check_pair(&c);
    }
    for lengths in [
        vec![13, 1, 13, 7, 2, 7, 16],
        vec![3; 7],
        vec![1, 2, 3, 4, 5, 6, 7, 8],
    ] {
        let c = Config {
            name: "unsorted/repeated".into(),
            judged: false,
            forests: vec![make_forest(
                &mut rng,
                Shape {
                    height: 4,
                    trees_log: 3,
                    real_trees: lengths.len(),
                    real_leaves: lengths,
                },
            )],
        };
        check_pair(&c);
    }
    // New controls appended AFTER the historical check RNG stream.
    let extremes = [
        0,
        1,
        0xffff_ffff,
        0x1_0000_0000,
        0xffff_ffff_0000_0000,
        0xffff_fffe_ffff_ffff,
    ]
    .map(Fp::from_u64_reduced);
    for lengths in [vec![0, 1, 2, 3, 4, 7, 8, 15, 16], vec![0; 7], vec![1, 0]] {
        let h = if lengths.len() == 2 { 1 } else { 4 };
        let mut f = make_forest(
            &mut rng,
            Shape {
                height: h,
                trees_log: lengths.len().next_power_of_two().trailing_zeros() as usize,
                real_trees: lengths.len(),
                real_leaves: lengths,
            },
        );
        for (y, &r) in f.shape.real_leaves.iter().enumerate() {
            for x in 0..r {
                f.padded[(y << h) + x] = extremes[(x + y) % extremes.len()];
            }
        }
        let c = Config {
            name: "new boundary/extreme".into(),
            judged: false,
            forests: vec![f],
        };
        check_pair(&c);
        check_scripts(&c);
    }
    let c = config_grouped(&mut rng, "scripted mixed heights", 16, 3, false);
    check_scripts(&c);
    println!("SMALL_CHECKS_PASS");
}
fn full_checks() -> Vec<Value> {
    small_checks();
    let old: Value = serde_json::from_str(include_str!(
        "../../EXP-0009-analytic-padding-blocks/results/amendment-run.json"
    ))
    .expect("historical JSON");
    let mut digests = Vec::new();
    for c in configs() {
        let digest = check_pair(&c);
        let expected = old["digests"]
            .as_array()
            .expect("digests")
            .iter()
            .find(|v| v["config"] == c.name)
            .expect("historical shape")["transcript_state"]
            .as_str()
            .expect("state");
        assert_eq!(digest, expected, "historical {} digest", c.name);
        digests.push(json!({"config":c.name,"transcript_state":digest,"independent_both":true,"real_leaves":c.real(),"padded_leaves":c.padded()}));
    }
    println!("FULL_CHECKS_PASS: all eight historical transcript states unchanged");
    digests
}
fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    match args.get(1).map_or("check", String::as_str) {
        "small" => small_checks(),
        "check" => {
            let d = full_checks();
            std::fs::write(
                "results/check.json",
                serde_json::to_vec_pretty(
                    &json!({"experiment":"EXP-0019","correctness":"pass","digests":d}),
                )
                .expect("JSON"),
            )
            .expect("write checks");
        }
        "diagnose" => {
            let mut result = Vec::new();
            for c in configs() {
                let mut t = Transcript::new(&context());
                let (_, _, fallbacks) = kernel::prove(&c, &mut t);
                result.push(json!({"config":c.name,"fallback_rounds":fallbacks,"transcript_state":hex(&t.state())}));
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&result).expect("diagnostics JSON")
            );
        }
        "paired" => bench::run(args.get(2).map_or("results/run.json", String::as_str)),
        _ => {
            eprintln!("usage: ragged-split-eq small|check|paired [results/run.json]");
            std::process::exit(2);
        }
    }
}
#[cfg(test)]
mod tests {
    #[test]
    fn exact_boundaries_and_scripted_zero_challenges() {
        super::small_checks();
    }
}
