// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The engine over the calibration field (07 §4): the same code proves batches over
//! `F65537Ext2`, with tables over `F65537` and over the extension, and over `F65537` itself,
//! whose challenges then lie in the prime field, on a transcript the test supplies. `tv-core`
//! checks Goldilocks proofs only, so each proof is checked here as RFC-0010 'Sumcheck messages'
//! says a verifier does, round by round from the batched claim with `p(1) = claim − p(0)` and
//! `tessera_poly::RoundPoly`, which the prover does not use, and each final evaluation against
//! the table's multilinear extension at the instance's point.
//!
//! Batches of product forests are checked the same way, level by level as RFC-0010 'Tree levels'
//! says: root challenges, root claims from the trees' products, the level's sumcheck and its
//! final check `Σ_i γ^i·eq(r_i, r'_i)·u_i·w_i`, `ρ` and the next claims, and the leaf claims
//! against the leaves' multilinear extension.
//!
//! Stages of matmul relations are checked as RFC-0010 'Relations' says: each instance's claim is
//! its true product's extension at the input point, `Y = X·Wᵀ` with a row per token for the
//! weights, the final check is `Σ_i γ^i·eq(ρ_i, ρ'_i)·u_i·v_i` with `eq` left out of a single
//! product, and each claim on an operand is that operand's extension at the point the RFC
//! assembles.

mod common;

use common::SplitMix64;
use tessera_codec::{Ascii, Encode};
use tessera_field::{F65537, F65537Ext2, Field, PrimeField, QuadraticExtension};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_poly::{PolyField, RoundPoly, evaluate, evaluate_base};
use tessera_sumcheck::{
    Forest, Instance, MAX_DEGREE, MatMulAct, MatMulActShape, MatMulWeight, MatMulWeightShape,
    Prefix, Protocol, RightLayout, SumcheckProof, SumcheckTranscript, Table, prove_batch,
    prove_forests,
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

/// Random batches over `F`: 1 to 4 instances on 0 to 8 variables, degree 1 to 10, the bound of a
/// reduction stage of `tessera-protocol/2` (RFC-0013 'Masked extensions' 'Degree'), tables of
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
        // A reduction stage of tessera-protocol/2, whose bound is MAX_DEGREE (RFC-0013 'Degree').
        let proven = prove_batch(
            &mut prover,
            Protocol::Two,
            Prefix::Stage(5),
            degree,
            &borrowed,
        )
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

/// `eq(r, y)` at the vertex `y`, from the definition: `Π_i (r_i·y_i + (1 − r_i)·(1 − y_i))`,
/// `y_i` bit `i − 1` of `y`.
fn eq_at_vertex<F: PolyField>(r: &[F], y: usize) -> F {
    r.iter().enumerate().fold(F::ONE, |product, (i, &ri)| {
        product * if (y >> i) & 1 == 1 { ri } else { F::ONE - ri }
    })
}

/// `eq(a, b) = Π_i (a_i·b_i + (1 − a_i)·(1 − b_i))`.
fn eq_points<F: PolyField>(a: &[F], b: &[F]) -> F {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(F::ONE, |product, (&a, &b)| {
        product * (a * b + (F::ONE - a) * (F::ONE - b))
    })
}

/// A batch of forests over `F`, `(s, h, zero rows)` each: heights 4, 6, 2, 3 and 1 join at
/// levels 2, 0, 4, 3 and 5, the forests of 2^2, 2^3 and 2^5 trees under root challenges, the
/// single trees under none. Proven, then checked level by level as a verifier would.
fn forest_batch<F: Draw<Base = F65537>>(seed: u8)
where
    SmallTranscript: SumcheckTranscript<F>,
{
    let mut rng = SplitMix64::new(0x58_0400 + u64::from(seed));
    let shapes = [
        (2usize, 4usize, true),
        (0, 6, false),
        (3, 2, false),
        (5, 3, true),
        (0, 1, true),
    ];
    let leaves: Vec<Vec<F65537>> = shapes
        .iter()
        .map(|&(s, h, zero_rows)| {
            let mut leaves: Vec<F65537> =
                (0..1 << (s + h)).map(|_| F65537::draw(&mut rng)).collect();
            if zero_rows {
                for y in 0..1 << s {
                    leaves[(y << h) + rng.below(1 << h) as usize] = F65537::ZERO;
                }
            }
            leaves
        })
        .collect();
    let forests: Vec<Forest<'_, F>> = leaves
        .iter()
        .zip(&shapes)
        .map(|(leaves, &(_, h, _))| Forest::new(leaves, h))
        .collect();
    let mut prover = SmallTranscript::new(seed);
    let proven = prove_forests(&mut prover, &forests).expect("well-formed forests");

    // The verifier, from RFC-0010 'Tree levels', with the trees' true products as public roots.
    let roots: Vec<Vec<F>> = leaves
        .iter()
        .zip(&shapes)
        .map(|(leaves, &(_, h, _))| {
            leaves
                .chunks_exact(1 << h)
                .map(|tree| F::from(tree.iter().fold(F65537::ONE, |p, &x| p * x)))
                .collect()
        })
        .collect();
    let tallest = shapes.iter().map(|&(_, h, _)| h).max().expect("a forest");
    assert_eq!(proven.levels().len(), tallest);
    let mut t = SmallTranscript::new(seed);
    let mut claims: Vec<Option<(Vec<F>, F)>> = vec![None; shapes.len()];
    for (level, record) in proven.levels().iter().enumerate() {
        let prefix = format!("tree/{level}");
        let joins = |i: usize| tallest - shapes[i].1;
        let s_max = (0..shapes.len())
            .filter(|&i| joins(i) == level)
            .map(|i| shapes[i].0)
            .max()
            .unwrap_or(0);
        let root: Vec<F> = (0..s_max)
            .map(|c| t.challenge(&format!("{prefix}/root/{c}")))
            .collect();
        assert!(
            record.root_challenges() == root,
            "level {level}: the root challenges"
        );
        for i in (0..shapes.len()).filter(|&i| joins(i) == level) {
            let r = root[s_max - shapes[i].0..].to_vec();
            let value = roots[i]
                .iter()
                .enumerate()
                .fold(F::ZERO, |sum, (y, &v)| sum + v * eq_at_vertex(&r, y));
            claims[i] = Some((r, value));
        }
        let present: Vec<usize> = (0..shapes.len()).filter(|&i| joins(i) <= level).collect();
        assert_eq!(record.forests(), present, "level {level}: the instances");
        let entering: Vec<(Vec<F>, F)> = present
            .iter()
            .map(|&i| claims[i].clone().expect("joined"))
            .collect();

        let gamma = t.challenge(&format!("{prefix}/gamma"));
        assert_eq!(*record.gamma(), gamma, "level {level}: γ");
        let n_max = entering
            .iter()
            .map(|(r, _)| r.len())
            .max()
            .expect("an instance");
        let mut claim = F::ZERO;
        let mut weight = F::ONE;
        for (r, value) in &entering {
            let scale = F::from(F65537::from_u64_reduced(1 << (n_max - r.len())));
            claim += weight * scale * *value;
            weight *= gamma;
        }
        let proof = record.proof();
        assert_eq!(
            proof.rounds.len(),
            n_max,
            "level {level}: one round per variable"
        );
        let mut point = Vec::new();
        for (j, round) in proof.rounds.iter().enumerate() {
            assert_eq!(round.len(), 3, "level {level}: round {j} holds 3 values");
            t.absorb_message(&format!("{prefix}/round/{j}"), round);
            let r = t.challenge(&format!("{prefix}/r/{j}"));
            claim = RoundPoly::new(round).expect("degree 3").evaluate(claim, r);
            point.push(r);
        }
        assert!(record.point() == point, "level {level}: the point");
        assert_eq!(proof.final_evals.len(), 2 * entering.len());
        t.absorb_message(&format!("{prefix}/evals"), &proof.final_evals);
        let rho = t.challenge(&format!("{prefix}/line"));
        assert_eq!(*record.rho(), rho, "level {level}: ρ");
        let mut batched = F::ZERO;
        let mut weight = F::ONE;
        for (k, ((r, _), evals)) in entering
            .iter()
            .zip(proof.final_evals.chunks_exact(2))
            .enumerate()
        {
            let own = &point[n_max - r.len()..];
            let (u, w) = (evals[0], evals[1]);
            batched += weight * eq_points(r, own) * u * w;
            weight *= gamma;
            let mut next = vec![rho];
            next.extend_from_slice(own);
            let value = u + rho * (w - u);
            let mine = &record.next_claims()[k];
            assert!(
                mine.point == next,
                "level {level}: instance {k}'s next point"
            );
            assert_eq!(
                mine.value, value,
                "level {level}: instance {k}'s next value"
            );
            claims[present[k]] = Some((next, value));
        }
        assert_eq!(batched, claim, "level {level}: the final check");
    }
    for (i, (claim, leaves)) in proven.leaf_claims().iter().zip(&leaves).enumerate() {
        assert_eq!(claim.point.len(), shapes[i].0 + shapes[i].1);
        assert_eq!(
            claim.value,
            evaluate_base(leaves, &claim.point).expect("2^n leaves at n coordinates"),
            "forest {i}: the leaves' extension at the leaf point"
        );
    }
    assert_eq!(
        t.inner.state(),
        prover.inner.state(),
        "the transcripts end in the same state"
    );
}

#[test]
fn forests_over_the_extension_of_f65537_verify() {
    for seed in 0..20 {
        forest_batch::<F65537Ext2>(seed);
    }
}

#[test]
fn forests_with_challenges_in_f65537_verify() {
    for seed in 0..20 {
        forest_batch::<F65537>(seed);
    }
}

/// Random `F65537` entries in `blocks` blocks of `block`, the blocks from `live` on zero: the zero
/// layers or products that pad a count to a power of two.
fn padded(rng: &mut SplitMix64, blocks: usize, block: usize, live: usize) -> Vec<F65537> {
    (0..blocks * block)
        .map(|k| match k / block < live {
            true => F65537::draw(rng),
            false => F65537::ZERO,
        })
        .collect()
}

/// A matmul over `F65537`, its operands row-major with the stacking index highest, as RFC-0010
/// 'Relations' and 'Multilinear extensions' define them:
///
/// - `MatMulWeight` of `(n_ℓ, n_t, n_o, n_i)`: `W` as `(ℓ, o, i)` and `X` as `(ℓ, t, i)`, one row
///   per token, `Y[ℓ, t, o] = Σ_i W[ℓ, o, i]·X[ℓ, t, i]`, the input point `r_o ‖ r_t ‖ ρ`, the
///   evaluations `[w, x]`, and the claims on `W` at `r_i ‖ r_o ‖ ρ_ℓ` and on `X` at
///   `r_i ‖ r_t ‖ ρ_ℓ`;
/// - `MatMulAct` of `(n_b, n_a, n_j, n_c)`: `A` as `(b, a, j)`, and `B` as `(b, j, c)` or `K` as
///   `(b, c, j)`, `Z[b, a, c] = Σ_j A[b, a, j]·B[b, j, c]`, the input point `r_c ‖ r_a ‖ ρ`, the
///   evaluations `[a, b]`, and the claims on `A` at `r_j ‖ r_a ‖ ρ_b` and on `B` at
///   `r_c ‖ r_j ‖ ρ_b`, or on `K` at `r_j ‖ r_c ‖ ρ_b`.
enum TestMatmul {
    Weight {
        shape: MatMulWeightShape,
        w: Vec<F65537>,
        x: Vec<F65537>,
    },
    Act {
        shape: MatMulActShape,
        layout: RightLayout,
        a: Vec<F65537>,
        b: Vec<F65537>,
    },
}

impl TestMatmul {
    /// `2^{n_ℓ}` layers of `(n_t, n_o, n_i)`, `live` of them nonzero.
    fn weight(
        rng: &mut SplitMix64,
        (layer_vars, token_vars, out_vars, in_vars): (usize, usize, usize, usize),
        live: usize,
    ) -> Self {
        let w = padded(rng, 1 << layer_vars, 1 << (out_vars + in_vars), live);
        let x = padded(rng, 1 << layer_vars, 1 << (token_vars + in_vars), live);
        let shape = MatMulWeightShape {
            layer_vars,
            token_vars,
            out_vars,
            in_vars,
        };
        Self::Weight { shape, w, x }
    }

    /// `2^{n_b}` products of `(n_a, n_j, n_c)`, `live` of them nonzero.
    fn act(
        rng: &mut SplitMix64,
        (stack_vars, row_vars, inner_vars, col_vars): (usize, usize, usize, usize),
        live: usize,
        layout: RightLayout,
    ) -> Self {
        let a = padded(rng, 1 << stack_vars, 1 << (row_vars + inner_vars), live);
        let b = padded(rng, 1 << stack_vars, 1 << (inner_vars + col_vars), live);
        let shape = MatMulActShape {
            stack_vars,
            row_vars,
            inner_vars,
            col_vars,
        };
        Self::Act {
            shape,
            layout,
            a,
            b,
        }
    }

    /// The operands, in the order of their evaluations.
    fn operands(&self) -> [&[F65537]; 2] {
        match self {
            Self::Weight { w, x, .. } => [w, x],
            Self::Act { a, b, .. } => [a, b],
        }
    }

    /// The stacking variables, the inner variables, and the input point's other coordinates,
    /// which precede `ρ`.
    fn dims(&self) -> (usize, usize, usize) {
        match self {
            Self::Weight { shape, .. } => (
                shape.layer_vars,
                shape.in_vars,
                shape.out_vars + shape.token_vars,
            ),
            Self::Act { shape, .. } => (
                shape.stack_vars,
                shape.inner_vars,
                shape.col_vars + shape.row_vars,
            ),
        }
    }

    /// The product, row-major as the input point reads it: `Y` as `(ℓ, t, o)`, `Z` as
    /// `(b, a, c)`.
    fn product(&self) -> Vec<F65537> {
        match self {
            Self::Weight { shape, w, x } => {
                let (tokens, outs, ins) = (
                    1usize << shape.token_vars,
                    1usize << shape.out_vars,
                    1usize << shape.in_vars,
                );
                (0..(1usize << shape.layer_vars) * tokens * outs)
                    .map(|k| {
                        let (l, t, o) = (k / (tokens * outs), (k / outs) % tokens, k % outs);
                        (0..ins).fold(F65537::ZERO, |sum, i| {
                            sum + w[(l * outs + o) * ins + i] * x[(l * tokens + t) * ins + i]
                        })
                    })
                    .collect()
            }
            Self::Act {
                shape,
                layout,
                a,
                b,
            } => {
                let (rows, inner, cols) = (
                    1usize << shape.row_vars,
                    1usize << shape.inner_vars,
                    1usize << shape.col_vars,
                );
                let b_at = |s: usize, j: usize, c: usize| match layout {
                    RightLayout::Direct => b[(s * inner + j) * cols + c],
                    RightLayout::Transposed => b[(s * cols + c) * inner + j],
                };
                (0..(1usize << shape.stack_vars) * rows * cols)
                    .map(|k| {
                        let (s, r, c) = (k / (rows * cols), (k / cols) % rows, k % cols);
                        (0..inner).fold(F65537::ZERO, |sum, j| {
                            sum + a[(s * rows + r) * inner + j] * b_at(s, j, c)
                        })
                    })
                    .collect()
            }
        }
    }

    /// The points of the claims on the operands, in the order of their evaluations, from the
    /// input point and the instance's point `r_i ‖ ρ_ℓ` or `r_j ‖ ρ_b`.
    fn claim_points<F: Copy>(&self, input: &[F], point: &[F]) -> [Vec<F>; 2] {
        match self {
            Self::Weight { shape, .. } => {
                let (r_i, rho) = point.split_at(shape.in_vars);
                let (r_o, rest) = input.split_at(shape.out_vars);
                let r_t = &rest[..shape.token_vars];
                [[r_i, r_o, rho].concat(), [r_i, r_t, rho].concat()]
            }
            Self::Act { shape, layout, .. } => {
                let (r_j, rho) = point.split_at(shape.inner_vars);
                let (r_c, rest) = input.split_at(shape.col_vars);
                let r_a = &rest[..shape.row_vars];
                let b = match layout {
                    RightLayout::Direct => [r_c, r_j, rho].concat(),
                    RightLayout::Transposed => [r_j, r_c, rho].concat(),
                };
                [[r_j, r_a, rho].concat(), b]
            }
        }
    }
}

/// A stage of matmuls over `F`: weights over 2^2 layers with 3 live for 2^2 tokens, over one
/// layer for one token, and of no output variable over 2^1 layers for 2^3 tokens; products with
/// `B` stored either way over 2^1 and 2^2 with 3 live, and one with no column variable. Proven,
/// then checked as a verifier would from the true products.
fn matmul_stage<F: Draw<Base = F65537>>(seed: u8)
where
    SmallTranscript: SumcheckTranscript<F>,
{
    let mut rng = SplitMix64::new(0x57_0400 + u64::from(seed));
    let matmuls = [
        TestMatmul::weight(&mut rng, (2, 2, 3, 4), 3),
        TestMatmul::weight(&mut rng, (0, 0, 2, 5), 1),
        TestMatmul::weight(&mut rng, (1, 3, 0, 2), 2),
        TestMatmul::act(&mut rng, (1, 2, 3, 2), 2, RightLayout::Direct),
        TestMatmul::act(&mut rng, (2, 1, 2, 3), 3, RightLayout::Transposed),
        TestMatmul::act(&mut rng, (0, 3, 2, 0), 1, RightLayout::Direct),
    ];
    let inputs: Vec<Vec<F>> = matmuls
        .iter()
        .map(|m| {
            let (stack, _, other) = m.dims();
            (0..other + stack).map(|_| F::draw(&mut rng)).collect()
        })
        .collect();

    // The prover.
    let weights: Vec<Option<MatMulWeight<'_, F>>> = matmuls
        .iter()
        .map(|m| match m {
            TestMatmul::Weight { shape, w, x } => {
                Some(MatMulWeight::new(w, x, *shape).expect("a shape"))
            }
            TestMatmul::Act { .. } => None,
        })
        .collect();
    let acts: Vec<Option<MatMulAct<'_, F>>> = matmuls
        .iter()
        .map(|m| match m {
            TestMatmul::Act {
                shape,
                layout,
                a,
                b,
            } => Some(MatMulAct::new(a, b, *layout, *shape).expect("a shape")),
            TestMatmul::Weight { .. } => None,
        })
        .collect();
    let weight_tables: Vec<_> = weights
        .iter()
        .zip(&inputs)
        .map(|(w, input)| w.map(|w| w.tables(input).expect("a point")))
        .collect();
    let act_tables: Vec<_> = acts
        .iter()
        .zip(&inputs)
        .map(|(a, input)| a.map(|a| a.tables(input).expect("a point")))
        .collect();
    let instances: Vec<Instance<'_, F>> = weight_tables
        .iter()
        .zip(&act_tables)
        .map(|(w, a)| match (w, a) {
            (Some(w), _) => w.instance(),
            (_, Some(a)) => a.instance(),
            (None, None) => unreachable!("a matmul of either kind"),
        })
        .collect();
    let mut prover = SmallTranscript::new(seed);
    let proven =
        prove_batch(&mut prover, Protocol::One, Prefix::Stage(3), 3, &instances).expect("a stage");
    let claims: Vec<[(Vec<F>, F); 2]> = weight_tables
        .iter()
        .zip(&act_tables)
        .enumerate()
        .map(|(k, (w, a))| {
            let point = proven.instance_point(k).expect("an instance");
            let evals = proven.instance_evals(k).expect("an instance");
            match (w, a) {
                (Some(w), _) => {
                    let c = w.claims(point, evals).expect("the results");
                    [(c.w.point, c.w.value), (c.x.point, c.x.value)]
                }
                (_, Some(a)) => {
                    let c = a.claims(point, evals).expect("the results");
                    [(c.a.point, c.a.value), (c.b.point, c.b.value)]
                }
                (None, None) => unreachable!("a matmul of either kind"),
            }
        })
        .collect();

    // The verifier, from RFC-0010 'Relations', with the true products' extensions as claims.
    let proof = proven.proof();
    let vars: Vec<usize> = matmuls
        .iter()
        .map(|m| {
            let (stack, inner, _) = m.dims();
            inner + stack
        })
        .collect();
    let n_max = vars.iter().copied().max().expect("an instance");
    let mut t = SmallTranscript::new(seed);
    let gamma = t.challenge("stage/3/gamma");
    let mut claim = F::ZERO;
    let mut weight = F::ONE;
    for ((m, input), &n) in matmuls.iter().zip(&inputs).zip(&vars) {
        let value = evaluate_base(&m.product(), input).expect("2^n entries at n coordinates");
        claim += weight * F::from(F65537::from_u64_reduced(1 << (n_max - n))) * value;
        weight *= gamma;
    }
    assert_eq!(proof.rounds.len(), n_max);
    let mut point = Vec::new();
    for (j, round) in proof.rounds.iter().enumerate() {
        assert_eq!(round.len(), 3, "round {j} holds 3 values");
        t.absorb_message(&format!("stage/3/round/{j}"), round);
        let r = t.challenge(&format!("stage/3/r/{j}"));
        claim = RoundPoly::new(round).expect("degree 3").evaluate(claim, r);
        point.push(r);
    }
    assert!(proven.point() == point, "the point");
    assert_eq!(proof.final_evals.len(), 2 * matmuls.len());
    t.absorb_message("stage/3/evals", &proof.final_evals);
    let mut batched = F::ZERO;
    let mut weight = F::ONE;
    for (k, (((m, input), &n), evals)) in matmuls
        .iter()
        .zip(&inputs)
        .zip(&vars)
        .zip(proof.final_evals.chunks_exact(2))
        .enumerate()
    {
        let (stack, inner, other) = m.dims();
        let own = &point[n_max - n..];
        let eq = match stack {
            0 => F::ONE,
            _ => eq_points(&input[other..], &own[inner..]),
        };
        batched += weight * eq * evals[0] * evals[1];
        weight *= gamma;
        for (operand, ((expected, eval), (mine, value))) in m.operands().into_iter().zip(
            m.claim_points(input, own)
                .into_iter()
                .zip(evals)
                .zip(&claims[k]),
        ) {
            assert!(*mine == expected, "instance {k}: a claim's point");
            assert_eq!(
                value, eval,
                "instance {k}: a claim's value is the evaluation sent"
            );
            assert_eq!(
                *value,
                evaluate_base(operand, &expected).expect("2^n entries at n coordinates"),
                "instance {k}: the operand's extension at the claim's point"
            );
        }
    }
    assert_eq!(batched, claim, "the final check");
    assert_eq!(
        t.inner.state(),
        prover.inner.state(),
        "the transcripts end in the same state"
    );
}

#[test]
fn matmul_stages_over_the_extension_of_f65537_verify() {
    for seed in 0..10 {
        matmul_stage::<F65537Ext2>(seed);
    }
}

#[test]
fn matmul_stages_with_challenges_in_f65537_verify() {
    for seed in 0..10 {
        matmul_stage::<F65537>(seed);
    }
}
