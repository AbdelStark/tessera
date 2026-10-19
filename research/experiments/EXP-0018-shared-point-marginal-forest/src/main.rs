// SPDX-License-Identifier: Apache-2.0 OR MIT
//! EXP-0018: actual production S/F/B and a separately judged public-eq proxy.
//! Synthetic input only. See the committed preregistration and README.
#![forbid(unsafe_code)]
#![allow(
    clippy::print_stdout,
    clippy::cast_precision_loss,
    clippy::too_many_lines
)]
use std::hint::black_box;
use std::io::Write;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tessera_codec::Ascii;
use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_hash::{Domain, Mode, ParamSetId, SessionContext, Transcript, hash};
use tessera_poly::eq_table;
use tessera_sumcheck::{
    Forest, FractionTree, Instance, LeafNumerators, Prefix, SumcheckProof, SumcheckTranscript,
    Table, Tree, prove_batch, prove_forests, prove_trees,
};
use tv_core::sumcheck as tv;
use tv_core::tree::{FractionClaim, LevelClaim as TvClaim, TreeClaim, verify_level};
use tv_hash::transcript as tvt;
type Fp = Goldilocks;
type Fp2 = GoldilocksExt2;
type TvFp = tv_field::Fp;
type TvFp2 = tv_field::Fp2;
const LEAF_SEED: u64 = 0x7E55_E7A0_0018_0001;
const ENGINE_SEED: u64 = 0x7E55_E7A0_0018_0002;
const CHECK_SEED: u64 = 0x7E55_E7A0_0018_0003;
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

fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

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

/// One instance as the verifier sees it: the public `eq` point and the value tables; the claim is
/// the sum over the hypercube.
struct TvInstance<'a> {
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

    fn vars(&self) -> usize {
        self.tables[0].len().trailing_zeros() as usize
    }

    fn evals(&self) -> usize {
        self.tables.len()
    }
}

#[derive(Clone, Copy)]
struct Shape {
    s: usize,
    h: usize,
}
impl Shape {
    fn log_n(self) -> usize {
        self.s + self.h
    }
    fn name(self) -> String {
        format!("s{}h{}", self.s, self.h)
    }
}
struct Inputs {
    shape: Shape,
    leaves: Vec<Fp>,
    denoms: Vec<Fp2>,
    points: Vec<Vec<Fp2>>,
}
impl Inputs {
    fn new(shape: Shape) -> Self {
        let mut leaves_rng = SplitMix64(LEAF_SEED);
        let mut rng = SplitMix64(ENGINE_SEED);
        let mut leaves = leaves_rng.fps(1 << shape.log_n());
        for tree in leaves.chunks_exact_mut(1 << shape.h) {
            let i = leaves_rng.next_u64() as usize % tree.len();
            tree[i] = Fp::ZERO;
        }
        let alpha = rng.fp2();
        let beta = rng.fp2();
        let denoms = (0..leaves.len())
            .map(|_| {
                let key = Fp::from_u64_reduced(rng.next_u64() & ((1 << 22) - 1));
                let value = Fp::from_u64_reduced(rng.next_u64() & 65535);
                alpha - Fp2::from(key) - beta * Fp2::from(value)
            })
            .collect::<Vec<_>>();
        assert!(denoms.iter().all(|&d| d != Fp2::ZERO));
        let points = (0..shape.h).map(|l| rng.fp2s(shape.s + l)).collect();
        Self {
            shape,
            leaves,
            denoms,
            points,
        }
    }
}
// Materialized original synthetic tables are inputs, not work hidden inside a measured arm.
// Constructors, all derived levels, proofs, leaf claims and their clearing/drop are timed.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Output {
    proofs: Vec<SumcheckProof<Fp2>>,
    claims: Vec<tessera_sumcheck::LevelClaim<Fp2>>,
    state: [u8; 32],
}
fn primary(input: &Inputs, arm: usize) -> Output {
    let mut t = Transcript::new(&context());
    let (proofs, claims) = if arm == 0 {
        let forest = Forest::new(&input.leaves, input.shape.h);
        let (p, c) = prove_forests(&mut t, &[forest])
            .expect("forest proof")
            .into_parts();
        (
            p,
            c.into_iter()
                .map(tessera_sumcheck::LevelClaim::Product)
                .collect(),
        )
    } else {
        let frac = FractionTree::new(input.shape.log_n(), LeafNumerators::Unit, |i| {
            input.denoms[i]
        })
        .expect("fraction");
        let mut trees = vec![Tree::Fraction(frac)];
        if arm == 2 {
            trees.push(Tree::Forest(Forest::new(&input.leaves, input.shape.h)));
        }
        prove_trees(&mut t, trees)
            .expect("trees proof")
            .into_parts()
    };
    Output {
        proofs,
        claims,
        state: t.state(),
    }
}
fn tv_claim(c: &tessera_sumcheck::LevelClaim<Fp2>) -> TvClaim {
    match c {
        tessera_sumcheck::LevelClaim::Product(c) => TvClaim::Product(TreeClaim {
            point: tv_fp2s(&c.point),
            value: tv_fp2(c.value),
        }),
        tessera_sumcheck::LevelClaim::Fraction(c) => TvClaim::Fraction(FractionClaim {
            point: tv_fp2s(&c.point),
            numerator: tv_fp2(c.numerator),
            denominator: tv_fp2(c.denominator),
        }),
    }
}
fn claim_point(c: &TvClaim) -> &[TvFp2] {
    match c {
        TvClaim::Product(c) => &c.point,
        TvClaim::Fraction(c) => &c.point,
    }
}
fn tv_fraction_root(denoms: &[Fp2]) -> (TvFp2, TvFp2) {
    // Binary-carry aggregation is independent from the production level representation.
    let mut stack: Vec<Option<(TvFp2, TvFp2)>> =
        vec![None; denoms.len().trailing_zeros() as usize + 1];
    for &d in denoms {
        let mut node = (TvFp2::ONE, tv_fp2(d));
        let mut j = 0;
        while let Some(a) = stack[j].take() {
            node = (a.0 * node.1 + node.0 * a.1, a.1 * node.1);
            j += 1;
        }
        stack[j] = Some(node);
    }
    stack.last().expect("root slot").expect("one root")
}
fn tv_ext_mle(table: &[Fp2], point: &[TvFp2]) -> TvFp2 {
    assert_eq!(table.len(), 1 << point.len());
    let mut v: Vec<_> = table.iter().copied().map(tv_fp2).collect();
    for &r in point {
        let n = v.len() / 2;
        for i in 0..n {
            v[i] = v[2 * i] + r * (v[2 * i + 1] - v[2 * i]);
        }
        v.truncate(n);
    }
    v[0]
}
#[derive(Clone)]
struct Root {
    s: usize,
    h: usize,
    product: Vec<TvFp2>,
    fraction: Option<(TvFp2, TvFp2)>,
}
fn roots(input: &Inputs, arm: usize) -> Vec<Root> {
    let forest = || Root {
        s: input.shape.s,
        h: input.shape.h,
        product: true_roots(&input.leaves, input.shape.h),
        fraction: None,
    };
    if arm == 0 {
        vec![forest()]
    } else {
        let mut r = vec![Root {
            s: 0,
            h: input.shape.log_n(),
            product: vec![],
            fraction: Some(tv_fraction_root(&input.denoms)),
        }];
        if arm == 2 {
            r.push(forest());
        }
        r
    }
}
fn trace_json(level: usize, entering: &[TvClaim], next: &[TvClaim], join: bool) -> String {
    let before = entering.len() == 2 && claim_point(&entering[0]) == claim_point(&entering[1]);
    let after = next.len() == 2 && claim_point(&next[0]) == claim_point(&next[1]);
    let lengths = entering
        .iter()
        .map(|c| claim_point(c).len().to_string())
        .collect::<Vec<_>>()
        .join(",");
    let digests = entering
        .iter()
        .map(|c| {
            let mut bytes = vec![];
            for x in claim_point(c) {
                bytes.extend(x.to_le_bytes());
            }
            format!("\"{}\"", hex(&hash(Domain::ColumnMerge, &bytes)))
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"level\":{level},\"forest_join\":{join},\"entering_vars\":[{lengths}],\"entering_point_digests\":[{digests}],\"entering_equal\":{before},\"next_equal\":{after}}}"
    )
}
fn verify_tree(roots: &[Root], output: &Output) -> Result<(Vec<TvClaim>, Vec<String>), String> {
    let tallest = roots.iter().map(|r| r.h).max().expect("trees");
    assert_eq!(output.proofs.len(), tallest);
    let mut t = tv_transcript();
    let mut claims: Vec<Option<TvClaim>> = vec![None; roots.len()];
    let mut trace = vec![];
    for (i, r) in roots.iter().enumerate() {
        if r.h == 0 {
            let (n, d) = r.fraction.expect("zero height fraction");
            claims[i] = Some(TvClaim::Fraction(FractionClaim {
                point: vec![],
                numerator: n,
                denominator: d,
            }));
        }
    }
    for (level, proof) in output.proofs.iter().enumerate() {
        let joining = (0..roots.len())
            .filter(|&i| tallest - roots[i].h == level)
            .collect::<Vec<_>>();
        let smax = joining.iter().map(|&i| roots[i].s).max().unwrap_or(0);
        let r = (0..smax)
            .map(|c| {
                t.challenge_fp2(tvt::Ascii::new(&format!("tree/{level}/root/{c}")).expect("label"))
                    .expect("root challenge")
            })
            .collect::<Vec<_>>();
        for &i in &joining {
            claims[i] = Some(if let Some((n, d)) = roots[i].fraction {
                TvClaim::Fraction(FractionClaim {
                    point: vec![],
                    numerator: n,
                    denominator: d,
                })
            } else {
                let point = r[smax - roots[i].s..].to_vec();
                let value = root_claim(&roots[i].product, &point);
                TvClaim::Product(TreeClaim { point, value })
            });
        }
        let present = (0..roots.len())
            .filter(|&i| tallest - roots[i].h <= level)
            .collect::<Vec<_>>();
        let entering = present
            .iter()
            .map(|&i| claims[i].clone().expect("joined"))
            .collect::<Vec<_>>();
        let checked = verify_level(&mut t, level as u32, &entering, &tv_proof(proof))
            .map_err(|e| format!("level {level}: {e:?}"))?;
        let forest_join = roots.len() == 2 && joining.contains(&1);
        if roots.len() == 2 && roots[0].fraction.is_some() && entering.len() == 2 {
            assert_eq!(
                claim_point(&entering[0]).len(),
                claim_point(&entering[1]).len()
            );
            if !forest_join {
                assert_eq!(
                    claim_point(&entering[0]),
                    claim_point(&entering[1]),
                    "equal points after common level"
                );
            }
            assert_eq!(
                claim_point(&checked.claims[0]),
                claim_point(&checked.claims[1]),
                "shared leaf/next point"
            );
        }
        trace.push(trace_json(level, &entering, &checked.claims, forest_join));
        for (&i, c) in present.iter().zip(checked.claims) {
            claims[i] = Some(c);
        }
    }
    let claims = claims
        .into_iter()
        .map(|c| c.expect("all joined"))
        .collect::<Vec<_>>();
    assert_eq!(
        claims,
        output.claims.iter().map(tv_claim).collect::<Vec<_>>()
    );
    assert_eq!(*t.state(), output.state);
    Ok((claims, trace))
}
fn digest_output(output: &Output) -> String {
    let mut bytes = vec![];
    for p in &output.proofs {
        proof_bytes(p, &mut bytes);
    }
    bytes.extend(output.state);
    // Full same-workload equality additionally checks every claim structurally.
    for claim in &output.claims {
        for x in claim.point() {
            for c in x.coeffs() {
                bytes.extend(c.to_canonical_u64().to_le_bytes());
            }
        }
        match claim {
            tessera_sumcheck::LevelClaim::Product(c) => {
                for a in c.value.coeffs() {
                    bytes.extend(a.to_canonical_u64().to_le_bytes());
                }
            }
            tessera_sumcheck::LevelClaim::Fraction(c) => {
                for x in [c.numerator, c.denominator] {
                    for a in x.coeffs() {
                        bytes.extend(a.to_canonical_u64().to_le_bytes());
                    }
                }
            }
        }
    }
    hex(&hash(Domain::ColumnMerge, &bytes))
}
fn check_primary(input: &Inputs, arm: usize) -> String {
    let out = primary(input, arm);
    assert_eq!(out, primary(input, arm), "same-workload replay");
    let r = roots(input, arm);
    let (claims, trace) = verify_tree(&r, &out).expect("independent tree acceptance");
    for claim in &claims {
        match claim {
            TvClaim::Product(c) => assert_eq!(c.value, mle(&input.leaves, &c.point)),
            TvClaim::Fraction(c) => {
                assert_eq!(c.numerator, TvFp2::ONE);
                assert_eq!(c.denominator, tv_ext_mle(&input.denoms, &c.point));
            }
        }
    }
    format!(
        "{{\"case\":\"production\",\"shape\":\"{}\",\"arm\":\"{}\",\"independent_acceptance\":true,\"leaf_claims_checked\":{},\"deterministic_replay\":true,\"digest\":\"{}\",\"trace\":[{}]}}",
        input.shape.name(),
        ["S", "F", "B"][arm],
        claims.len(),
        digest_output(&out),
        trace.join(",")
    )
}
// Proxy levels deliberately use the public generic API, not the private production kernel.
fn product_levels<E: Field>(leaves: &[E], height: usize) -> Vec<(Vec<E>, Vec<E>)> {
    let mut levels = Vec::with_capacity(height);
    let (even, odd) = leaves.chunks_exact(2).map(|p| (p[0], p[1])).unzip();
    levels.push((even, odd));
    for _ in 1..height {
        let (e, o): &(Vec<E>, Vec<E>) = levels.last().expect("level");
        let next = e
            .chunks_exact(2)
            .zip(o.chunks_exact(2))
            .map(|(e, o)| (e[0] * o[0], e[1] * o[1]))
            .unzip();
        levels.push(next);
    }
    levels
}
fn proxy(input: &Inputs, arm: usize) -> Output {
    let p = if arm != 1 {
        product_levels(&input.leaves, input.shape.h)
    } else {
        vec![]
    };
    let r = if arm != 0 {
        product_levels(&input.denoms, input.shape.h)
    } else {
        vec![]
    };
    let mut t = Transcript::new(&context());
    let mut proofs = vec![];
    for level in 0..input.shape.h {
        let eq = eq_table(&input.points[level]).expect("public eq");
        let idx = input.shape.h - 1 - level;
        let mut instances = vec![];
        if arm != 0 {
            instances.push(Instance::new(vec![
                Table::ext(&eq).public(),
                Table::ext(&r[idx].0),
                Table::ext(&r[idx].1),
            ]));
        }
        if arm != 1 {
            instances.push(Instance::new(vec![
                Table::ext(&eq).public(),
                Table::base(&p[idx].0),
                Table::base(&p[idx].1),
            ]));
        }
        proofs.push(
            prove_batch(&mut t, Prefix::Tree(level as u32), 3, &instances)
                .expect("proxy batch")
                .into_proof(),
        );
    }
    Output {
        proofs,
        claims: vec![],
        state: t.state(),
    }
}
// Independent level arithmetic over tv-field; never a prover field multiplication.
fn tv_product_levels(mut nodes: Vec<TvFp2>, height: usize) -> Vec<(Vec<TvFp2>, Vec<TvFp2>)> {
    let mut levels = vec![];
    for _ in 0..height {
        let halves = nodes.chunks_exact(2).map(|p| (p[0], p[1])).unzip();
        levels.push(halves);
        nodes = nodes.chunks_exact(2).map(|p| p[0] * p[1]).collect();
    }
    levels
}
fn check_proxy(input: &Inputs, arm: usize) -> String {
    let out = proxy(input, arm);
    assert_eq!(out, proxy(input, arm), "proxy replay");
    let p = product_levels(&input.leaves, input.shape.h);
    let r = product_levels(&input.denoms, input.shape.h);
    let expected_p = tv_product_levels(
        input
            .leaves
            .iter()
            .copied()
            .map(|v| TvFp2::from(tv_fp(v)))
            .collect(),
        input.shape.h,
    );
    let expected_r = tv_product_levels(tv_fp2s(&input.denoms), input.shape.h);
    for ((a, b), (c, d)) in p.iter().zip(&expected_p) {
        assert_eq!(
            a.iter()
                .copied()
                .map(|v| TvFp2::from(tv_fp(v)))
                .collect::<Vec<_>>(),
            *c
        );
        assert_eq!(
            b.iter()
                .copied()
                .map(|v| TvFp2::from(tv_fp(v)))
                .collect::<Vec<_>>(),
            *d
        );
    }
    for ((a, b), (c, d)) in r.iter().zip(&expected_r) {
        assert_eq!(tv_fp2s(a), *c);
        assert_eq!(tv_fp2s(b), *d);
    }
    let mut t = tv_transcript();
    for level in 0..input.shape.h {
        let eq = eq_table(&input.points[level]).expect("eq");
        let idx = input.shape.h - 1 - level;
        let mut views = vec![];
        if arm != 0 {
            views.push(TvInstance {
                eq: &eq,
                tables: vec![Values::Ext(&r[idx].0), Values::Ext(&r[idx].1)],
            });
        }
        if arm != 1 {
            views.push(TvInstance {
                eq: &eq,
                tables: vec![Values::Base(&p[idx].0), Values::Base(&p[idx].1)],
            });
        }
        let specs = views
            .iter()
            .map(|v| tv::Instance {
                claim: v.claim(),
                vars: v.vars(),
                evals: v.evals(),
            })
            .collect::<Vec<_>>();
        // Derive the public factor from its coordinates, rather than trusting the prover eq table.
        let mut independent_eq = vec![TvFp2::ONE];
        for r in tv_fp2s(&input.points[level]) {
            let high = independent_eq.iter().map(|&e| e * r).collect::<Vec<_>>();
            for (e, h) in independent_eq.iter_mut().zip(&high) {
                *e -= *h;
            }
            independent_eq.extend(high);
        }
        for v in &views {
            assert_eq!(tv_fp2s(v.eq), independent_eq);
        }
        let point = tv_fp2s(&input.points[level]);
        let checked = tv::verify_batch(
            &mut t,
            tv::Prefix::Tree(level as u32),
            3,
            &specs,
            &tv_proof(&out.proofs[level]),
            |_, p, e| Some(tv::eq(&point, p)? * e.iter().copied().fold(TvFp2::ONE, |a, b| a * b)),
        )
        .expect("independent proxy acceptance");
        // All owed evaluations checked independently at their own suffix point.
        let proof = &out.proofs[level];
        let mut cursor = 0;
        for v in &views {
            let n = v.vars();
            let point = &checked.point[checked.point.len() - n..];
            for table in &v.tables {
                let value = match table {
                    Values::Base(a) => mle(a, point),
                    Values::Ext(a) => tv_ext_mle(a, point),
                };
                assert_eq!(value, tv_fp2(proof.final_evals[cursor]));
                cursor += 1;
            }
        }
        assert_eq!(cursor, proof.final_evals.len());
    }
    assert_eq!(*t.state(), out.state);
    format!(
        "{{\"case\":\"proxy\",\"shape\":\"{}\",\"arm\":\"{}\",\"independent_acceptance\":true,\"deterministic_replay\":true,\"all_final_evaluations_checked\":true,\"digest\":\"{}\"}}",
        input.shape.name(),
        ["P", "R", "T"][arm],
        digest_output(&out)
    )
}

fn controls() {
    let fp = Fp::from_u64_reduced;
    let mut rng = SplitMix64(CHECK_SEED);
    let leaves = [2, 3, 5, 7].map(fp).to_vec();
    let mut input = Inputs::new(Shape { s: 0, h: 2 });
    input.leaves = leaves;
    assert_eq!(
        true_roots(&input.leaves, 2),
        vec![TvFp2::from(tv_fp(fp(210)))]
    );
    for a in 0..3 {
        check_primary(&input, a);
        check_proxy(&input, a);
    }
    let out = primary(&input, 0);
    let mut wrong = roots(&input, 0);
    wrong[0].product[0] = TvFp2::from(tv_fp(fp(211)));
    assert_eq!(
        verify_tree(&wrong, &out).err().as_deref(),
        Some("level 0: FinalMismatch { round: 0 }")
    );
    let mut bad = out.clone();
    bad.proofs[0].final_evals[0] += Fp2::ONE;
    assert_eq!(
        verify_tree(&roots(&input, 0), &bad).err().as_deref(),
        Some("level 0: FinalMismatch { round: 0 }")
    );
    // A changed owed leaf claim fails comparison with the independently carried leaf statement.
    let mut bad = out.clone();
    if let tessera_sumcheck::LevelClaim::Product(c) = &mut bad.claims[0] {
        c.value += Fp2::ONE;
    }
    assert_ne!(
        bad.claims.iter().map(tv_claim).collect::<Vec<_>>(),
        verify_tree(&roots(&input, 0), &out).expect("accept").0
    );
    println!(
        "control root210 accepted; wrong211/final-proof exact FinalMismatch round0; altered owed leaf mismatch caught"
    );
    for (s, h) in [(0, 1), (2, 2), (1, 3), (3, 1)] {
        let mut input = Inputs::new(Shape { s, h });
        for (l, p) in input.points.iter_mut().enumerate() {
            for (j, x) in p.iter_mut().enumerate() {
                *x = if (l + j) % 2 == 0 {
                    Fp2::ZERO
                } else {
                    Fp2::ONE
                };
            }
        }
        for a in 0..3 {
            check_primary(&input, a);
            check_proxy(&input, a);
        }
    }
    let zero_vars = Inputs::new(Shape { s: 0, h: 0 });
    check_primary(&zero_vars, 1);
    let mut t = Transcript::new(&context());
    assert!(matches!(
        prove_forests::<Fp2, _>(&mut t, &[Forest::new(&zero_vars.leaves, 0)]),
        Err(tessera_sumcheck::SumcheckError::Height {
            forest: 0,
            height: 0,
            vars: 0
        })
    ));
    // Two genuine different-N forests exercise front padding and non-equal/nested points.
    for shapes in [
        [Shape { s: 1, h: 3 }, Shape { s: 0, h: 2 }],
        [Shape { s: 0, h: 2 }, Shape { s: 2, h: 1 }],
    ] {
        let inputs = shapes.map(Inputs::new);
        let mut t = Transcript::new(&context());
        let forests: Vec<_> = inputs
            .iter()
            .map(|i| Forest::new(&i.leaves, i.shape.h))
            .collect();
        let (proofs, claims) = prove_forests(&mut t, &forests)
            .expect("padding")
            .into_parts();
        let output = Output {
            proofs,
            claims: claims
                .into_iter()
                .map(tessera_sumcheck::LevelClaim::Product)
                .collect(),
            state: t.state(),
        };
        let r = inputs
            .iter()
            .map(|i| roots(i, 0).remove(0))
            .collect::<Vec<_>>();
        let (claims, _) = verify_tree(&r, &output).expect("front padding accepted");
        for (input, c) in inputs.iter().zip(claims) {
            if let TvClaim::Product(c) = c {
                assert_eq!(c.value, mle(&input.leaves, &c.point));
            } else {
                panic!("product");
            }
        }
    }
    // Current split-eq production versus literal full-eq generic proof under a scripted prior=0.
    let s = 4;
    let leaves = rng.fps(2 << s);
    let mut point = rng.fp2s(s);
    point[0] = Fp2::ZERO;
    point[2] = Fp2::ONE;
    let mut challenges = rng.fp2s(s);
    challenges[0] = Fp2::ONE;
    let gamma = rng.fp2();
    let rho = rng.fp2();
    let mut script = point.clone();
    script.push(gamma);
    script.extend(&challenges);
    script.push(rho);
    let mut transcript = Scripted::new(script);
    let product = prove_forests(&mut transcript, &[Forest::new(&leaves, 1)]).expect("scripted");
    let (even, odd): (Vec<_>, Vec<_>) = leaves.chunks_exact(2).map(|a| (a[0], a[1])).unzip();
    let eq = eq_table(&point).expect("eq");
    let views = [Instance::new(vec![
        Table::ext(&eq).public(),
        Table::base(&even),
        Table::base(&odd),
    ])];
    let mut script = vec![gamma];
    script.extend(challenges);
    let mut transcript = Scripted::new(script);
    let generic =
        prove_batch(&mut transcript, Prefix::Tree(0), 3, &views).expect("scripted generic");
    assert_eq!(product.levels()[0].proof(), generic.proof());
    assert!(
        generic
            .proof()
            .rounds
            .iter()
            .skip(1)
            .all(|r| r.iter().all(|&x| x == Fp2::ZERO))
    );
    // Independent field evaluation of the final public factor is exactly zero.
    assert_eq!(
        tv::eq(&tv_fp2s(&point), &tv_fp2s(generic.point())),
        Some(TvFp2::ZERO)
    );
    let mut zero = Inputs::new(Shape { s: 1, h: 2 });
    zero.denoms[0] = Fp2::ZERO;
    for a in 1..3 {
        check_primary(&zero, a);
    }
    println!(
        "controls: zero-variable, mixed heights/front padding, zero/one eq, scripted prior-eq-zero (split/full-eq exact), unit numerator, zero denominator passed"
    );
}
fn checks(large: bool, path: &str) {
    controls();
    let shapes = if large {
        vec![
            Shape { s: 8, h: 14 },
            Shape { s: 10, h: 12 },
            Shape { s: 0, h: 22 },
            Shape { s: 6, h: 14 },
            Shape { s: 7, h: 14 },
        ]
    } else {
        vec![
            Shape { s: 3, h: 4 },
            Shape { s: 0, h: 7 },
            Shape { s: 2, h: 3 },
        ]
    };
    let mut records = vec![];
    for shape in shapes {
        let input = Inputs::new(shape);
        for arm in 0..3 {
            println!("checking production {} {}", shape.name(), arm);
            records.push(check_primary(&input, arm));
        }
        for arm in 0..3 {
            println!("checking proxy {} {}", shape.name(), arm);
            records.push(check_proxy(&input, arm));
        }
    }
    std::fs::write(
        path,
        format!(
            "{{\"controls_passed\":true,\"large\":{large},\"records\":[{}]}}\n",
            records.join(",")
        ),
    )
    .expect("check output");
}

const ORDERS: [[usize; 3]; 10] = [
    [0, 1, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
    [1, 0, 2],
    [0, 2, 1],
    [0, 1, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];
fn now_ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos()
}
fn execute(input: &Inputs, proxy_case: bool, arm: usize) {
    black_box(if proxy_case {
        proxy(black_box(input), arm)
    } else {
        primary(black_box(input), arm)
    });
}
fn duration(input: &Inputs, proxy_case: bool, arm: usize, reps: u32) -> Duration {
    let start = Instant::now();
    for _ in 0..reps {
        execute(input, proxy_case, arm);
    }
    start.elapsed()
}
fn cv(xs: &[f64]) -> f64 {
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let v = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (xs.len() - 1) as f64;
    v.sqrt() / mean.abs()
}
fn median(xs: &[f64]) -> f64 {
    let mut x = xs.to_vec();
    x.sort_by(f64::total_cmp);
    (x[4] + x[5]) / 2.0
}
fn stats(xs: &[f64]) -> String {
    format!(
        "{{\"median\":{},\"min\":{},\"max\":{},\"cv\":{}}}",
        median(xs),
        xs.iter().copied().fold(f64::INFINITY, f64::min),
        xs.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        cv(xs)
    )
}
fn attempt(
    input: &Inputs,
    proxy_case: bool,
    index: usize,
    reps: &[u32; 3],
    journal: &mut std::fs::File,
) -> (String, bool) {
    let mut raw = vec![];
    let mut values = [vec![], vec![], vec![]];
    let mut marginals = vec![];
    let mut ratios = vec![];
    let mut durations_ok = true;
    for (block, order) in ORDERS.iter().enumerate() {
        let mut times = [0.; 3];
        let mut samples = vec![];
        for &arm in order {
            let started = now_ns();
            let d = duration(input, proxy_case, arm, reps[arm]);
            let ended = now_ns();
            durations_ok &= d >= Duration::from_millis(20);
            let normalized = d.as_secs_f64() / f64::from(reps[arm]);
            times[arm] = normalized;
            values[arm].push(normalized);
            writeln!(journal,"{{\"case\":\"{}\",\"shape\":\"{}\",\"attempt\":{index},\"block\":{block},\"arm\":{arm},\"start_unix_ns\":{started},\"end_unix_ns\":{ended},\"reps\":{},\"elapsed_ns\":{}}}",if proxy_case{"proxy"}else{"production"},input.shape.name(),reps[arm],d.as_nanos()).expect("journal sample");
            journal.flush().expect("journal flush");
            samples.push(format!("{{\"arm\":{},\"start_unix_ns\":{started},\"end_unix_ns\":{ended},\"reps\":{},\"elapsed_ns\":{},\"seconds_per_execution\":{normalized}}}",arm,reps[arm],d.as_nanos()));
        }
        let m = times[2] - times[1];
        let r = m / times[0];
        marginals.push(m);
        ratios.push(r);
        raw.push(format!("{{\"block\":{block},\"samples_in_order\":[{}],\"S_or_P\":{},\"F_or_R\":{},\"B_or_T\":{},\"marginal\":{m},\"ratio\":{r}}}",samples.join(","),times[0],times[1],times[2]));
        println!(
            "sample case={} shape={} attempt={} block={} marginal_ratio={:.6}",
            if proxy_case { "proxy" } else { "production" },
            input.shape.name(),
            index,
            block,
            r
        );
    }
    let positive = marginals.iter().all(|&x| x > 0.);
    let base_ok = values.iter().all(|v| cv(v) <= 0.03);
    let marginal_ok = cv(&marginals) <= 0.03;
    let noise_ok = base_ok && marginal_ok && durations_ok;
    let verdict = if !noise_ok || !positive {
        "inconclusive"
    } else if ratios.iter().all(|&r| r < 0.90) {
        "validated"
    } else if ratios.iter().all(|&r| r >= 0.90) {
        "refuted"
    } else {
        "inconclusive"
    };
    (
        format!(
            "{{\"attempt\":{index},\"blocks\":[{}],\"arm_statistics\":[{},{},{}],\"marginal_statistics\":{},\"ratio_statistics\":{},\"duration_floor_passed\":{durations_ok},\"all_marginals_positive\":{positive},\"noise_passed\":{noise_ok},\"verdict\":\"{verdict}\"}}",
            raw.join(","),
            stats(&values[0]),
            stats(&values[1]),
            stats(&values[2]),
            stats(&marginals),
            stats(&ratios)
        ),
        noise_ok,
    )
}
fn timings(path: &str) {
    let mut journal =
        std::fs::File::create(format!("{path}.samples.ndjson")).expect("raw sample journal");
    let mut rows = vec![];
    for shape in [
        Shape { s: 8, h: 14 },
        Shape { s: 10, h: 12 },
        Shape { s: 0, h: 22 },
        Shape { s: 6, h: 14 },
        Shape { s: 7, h: 14 },
    ] {
        let input = Inputs::new(shape);
        for proxy_case in [false, true] {
            let mut reps = [1; 3];
            let mut calibrations = vec![];
            for (arm, r) in reps.iter_mut().enumerate() {
                for warmup in 0..3 {
                    let d = duration(&input, proxy_case, arm, 1);
                    calibrations.push(format!(
                        "{{\"arm\":{arm},\"warmup\":{warmup},\"elapsed_ns\":{}}}",
                        d.as_nanos()
                    ));
                    if warmup == 2 {
                        *r = (0.060 / d.as_secs_f64()).ceil().max(1.) as u32;
                    }
                }
            }
            let (first, ok) = attempt(&input, proxy_case, 0, &reps, &mut journal);
            let mut attempts = vec![first];
            if !ok {
                attempts.push(attempt(&input, proxy_case, 1, &reps, &mut journal).0);
            }
            rows.push(format!("{{\"case\":\"{}\",\"shape\":\"{}\",\"total_log_leaves\":{},\"forest_height\":{},\"judged\":{},\"warmups\":[{}],\"attempts\":[{}]}}",if proxy_case{"proxy"}else{"production"},shape.name(),shape.log_n(),shape.h,shape.log_n()==22,calibrations.join(","),attempts.join(",")));
            // Persist after every complete comparison; interruption never discards completed raw blocks.
            std::fs::write(path, format!("{{\"rows\":[{}]}}\n", rows.join(",")))
                .expect("timing output");
        }
    }
}
fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let mode = args.get(1).map_or("check", String::as_str);
    let default = if mode == "all" {
        "results/run.json"
    } else {
        "results/check.json"
    };
    let path = args.get(2).map_or(default, String::as_str);
    match mode {
        "check" => checks(false, path),
        "check-large" => checks(true, path),
        "all" => timings(path),
        _ => panic!("check | check-large | all [output.json]"),
    }
}
