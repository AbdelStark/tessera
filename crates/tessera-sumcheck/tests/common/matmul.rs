// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Matmul relations for the tests, and the verifier's side of them, written from RFC-0010
//! 'Relations', 'Multilinear extensions' and 'Batching within a stage' around the verifier's
//! independent batch check, `tv_core::sumcheck::verify_batch` (DEP-2).
//!
//! A case holds random operands over `Fp`, row-major with the stacking index highest, zero past
//! the products the schedule has, and a random input point. Its verifier's side uses `tv-field`
//! and `tv_core::sumcheck::eq` alone:
//!
//! - the product, entry by entry from the operands: `Y[ℓ, t, o] = Σ_i W[ℓ, o, i]·X[ℓ, t, i]`,
//!   one row of `X` and `Y` per token, and `Z[b, a, c] = Σ_j A[b, a, j]·B[b, j, c]`; and the input
//!   claim, the product's multilinear extension at the input point, summed over the hypercube;
//! - the relation at the instance's point: `eq(ρ, ρ_ℓ)·w·x` over stacked layers, `w·x` for a
//!   single one, and the same for `A` and `B`;
//! - the points of the claims on the operands, assembled from the verifier's point and the input
//!   point: `W` at `r_i ‖ r_o ‖ ρ_ℓ` and `X` at `r_i ‖ r_t ‖ ρ_ℓ`; `A` at `r_j ‖ r_a ‖ ρ_b`, and
//!   `B` at `r_c ‖ r_j ‖ ρ_b`, or `K` at `r_j ‖ r_c ‖ ρ_b` when stored transposed.
//!
//! The prover's side runs the relations of `tessera-sumcheck` and `prove_batch`. None of the
//! verifier's side uses the prover's code.

use tessera_field::Field;
use tessera_sumcheck::{
    Claim, Instance, MatMulAct, MatMulActShape, MatMulActTables, MatMulWeight, MatMulWeightShape,
    MatMulWeightTables, Prefix, ProvenBatch, RightLayout, Table, prove_batch,
};
use tv_core::sumcheck as tv;

use super::forests::mle;
use super::{
    Fp, Fp2, Session, SplitMix64, TestInstance, TvFp, TvFp2, tv_fp, tv_fp2, tv_fp2s, tv_prefix,
    tv_proof,
};

/// `Σ_x values[x]·eq(point, x)` over the hypercube, in `tv-field`, the `eq` weights built by
/// doubling as `forests::mle` builds them.
pub fn extension(values: &[TvFp], point: &[TvFp2]) -> TvFp2 {
    assert_eq!(
        values.len(),
        1 << point.len(),
        "2^n values at n coordinates"
    );
    let mut weights = vec![TvFp2::ONE];
    for &r in point {
        let high: Vec<TvFp2> = weights.iter().map(|&e| e * r).collect();
        for (e, h) in weights.iter_mut().zip(&high) {
            *e -= *h;
        }
        weights.extend(high);
    }
    values
        .iter()
        .zip(&weights)
        .fold(TvFp2::ZERO, |sum, (&v, &e)| sum + e * TvFp2::from(v))
}

/// Random `Fp` entries, the first `live` blocks of `block` entries random and the rest zero: the
/// zero layers or products that pad the schedule's count to a power of two.
fn padded(rng: &mut SplitMix64, blocks: usize, block: usize, live: usize) -> Vec<Fp> {
    assert!(live <= blocks, "at most as many live blocks as blocks");
    let mut values = rng.fps(blocks * block);
    for value in &mut values[live * block..] {
        *value = Fp::ZERO;
    }
    values
}

/// A `MatMulWeight` case: `2^{n_ℓ}` layers, each the activations of `2^{n_t}` tokens by `2^{n_i}`
/// inputs and a weight matrix of `2^{n_o}` outputs by `2^{n_i}` inputs, layers from `layers` on
/// zero.
#[derive(Clone)]
pub struct WeightCase {
    pub shape: MatMulWeightShape,
    /// `W`, row-major `(ℓ, o, i)`.
    pub w: Vec<Fp>,
    /// `X`, row-major `(ℓ, t, i)`.
    pub x: Vec<Fp>,
    /// The input claim's point on `Y`, `r_o ‖ r_t ‖ ρ`.
    pub point: Vec<Fp2>,
}

impl WeightCase {
    pub fn random(rng: &mut SplitMix64, shape: MatMulWeightShape, layers: usize) -> Self {
        let MatMulWeightShape {
            layer_vars,
            token_vars,
            out_vars,
            in_vars,
        } = shape;
        let w = padded(rng, 1 << layer_vars, 1 << (out_vars + in_vars), layers);
        let x = padded(rng, 1 << layer_vars, 1 << (token_vars + in_vars), layers);
        let point = rng.fp2s(out_vars + token_vars + layer_vars);
        Self { shape, w, x, point }
    }

    /// `(n_ℓ, n_t, n_o, n_i)`, all layers live.
    pub fn full(
        rng: &mut SplitMix64,
        (layer_vars, token_vars, out_vars, in_vars): (usize, usize, usize, usize),
    ) -> Self {
        let shape = MatMulWeightShape {
            layer_vars,
            token_vars,
            out_vars,
            in_vars,
        };
        Self::random(rng, shape, 1 << layer_vars)
    }

    pub fn relation(&self) -> MatMulWeight<'_, Fp2> {
        MatMulWeight::new(&self.w, &self.x, self.shape).expect("operands of the shape")
    }

    pub fn vars(&self) -> usize {
        self.shape.in_vars + self.shape.layer_vars
    }

    /// `Y[ℓ, t, o] = Σ_i W[ℓ, o, i]·X[ℓ, t, i]`, row-major `(ℓ, t, o)`, in `tv-field`.
    pub fn product(&self) -> Vec<TvFp> {
        let MatMulWeightShape {
            layer_vars,
            token_vars,
            out_vars,
            in_vars,
        } = self.shape;
        let (tokens, outs, ins) = (1 << token_vars, 1 << out_vars, 1 << in_vars);
        (0..1 << (layer_vars + token_vars + out_vars))
            .map(|k| {
                let (layer, token, out) = (k / (tokens * outs), (k / outs) % tokens, k % outs);
                (0..ins).fold(TvFp::ZERO, |sum, i| {
                    let w = self.w[(layer * outs + out) * ins + i];
                    let x = self.x[(layer * tokens + token) * ins + i];
                    sum + tv_fp(w) * tv_fp(x)
                })
            })
            .collect()
    }

    /// The input claim: `Ỹ(r_o ‖ r_t ‖ ρ)`.
    pub fn claim(&self) -> TvFp2 {
        extension(&self.product(), &tv_fp2s(&self.point))
    }

    /// `ρ`, the input point's layer coordinate.
    pub fn rho(&self) -> Vec<TvFp2> {
        tv_fp2s(&self.point[self.shape.out_vars + self.shape.token_vars..])
    }

    /// The relation at the instance's point `r_i ‖ ρ_ℓ` from the evaluations `[w, x]`:
    /// `eq(ρ, ρ_ℓ)·w·x`, or `w·x` for a single layer.
    pub fn relation_at(&self, point: &[TvFp2], evals: &[TvFp2]) -> Option<TvFp2> {
        let [w, x] = evals else { return None };
        let rho_l = point.get(self.shape.in_vars..)?;
        if self.shape.layer_vars == 0 {
            Some(*w * *x)
        } else {
            Some(tv::eq(&self.rho(), rho_l)? * *w * *x)
        }
    }

    /// The points of the claims on `W` and `X` from the instance's point `r_i ‖ ρ_ℓ`:
    /// `r_i ‖ r_o ‖ ρ_ℓ` and `r_i ‖ r_t ‖ ρ_ℓ`.
    pub fn claim_points(&self, point: &[TvFp2]) -> [Vec<TvFp2>; 2] {
        let (r_i, rho_l) = point.split_at(self.shape.in_vars);
        let input = tv_fp2s(&self.point);
        let (r_o, rest) = input.split_at(self.shape.out_vars);
        let r_t = &rest[..self.shape.token_vars];
        [[r_i, r_o, rho_l].concat(), [r_i, r_t, rho_l].concat()]
    }
}

/// A `MatMulAct` case: `2^{n_b}` products of a `2^{n_a} × 2^{n_j}` matrix `A` by a
/// `2^{n_j} × 2^{n_c}` matrix `B`, products from `count` on zero.
#[derive(Clone)]
pub struct ActCase {
    pub shape: MatMulActShape,
    pub layout: RightLayout,
    pub a: Vec<Fp>,
    /// `B` as `layout` stores it.
    pub b: Vec<Fp>,
    /// The input claim's point on `Z`, `r_c ‖ r_a ‖ ρ`.
    pub point: Vec<Fp2>,
}

impl ActCase {
    pub fn random(
        rng: &mut SplitMix64,
        shape: MatMulActShape,
        layout: RightLayout,
        count: usize,
    ) -> Self {
        let MatMulActShape {
            stack_vars,
            row_vars,
            inner_vars,
            col_vars,
        } = shape;
        let a = padded(rng, 1 << stack_vars, 1 << (row_vars + inner_vars), count);
        let b = padded(rng, 1 << stack_vars, 1 << (inner_vars + col_vars), count);
        let point = rng.fp2s(col_vars + row_vars + stack_vars);
        Self {
            shape,
            layout,
            a,
            b,
            point,
        }
    }

    /// `(n_b, n_a, n_j, n_c)`, all products live.
    pub fn full(
        rng: &mut SplitMix64,
        (stack_vars, row_vars, inner_vars, col_vars): (usize, usize, usize, usize),
        layout: RightLayout,
    ) -> Self {
        let shape = MatMulActShape {
            stack_vars,
            row_vars,
            inner_vars,
            col_vars,
        };
        Self::random(rng, shape, layout, 1 << stack_vars)
    }

    pub fn relation(&self) -> MatMulAct<'_, Fp2> {
        MatMulAct::new(&self.a, &self.b, self.layout, self.shape).expect("operands of the shape")
    }

    pub fn vars(&self) -> usize {
        self.shape.inner_vars + self.shape.stack_vars
    }

    /// `B[b, j, c]`, read from the operand as stored.
    fn b_at(&self, b: usize, j: usize, c: usize) -> Fp {
        let (inner, cols) = (1 << self.shape.inner_vars, 1 << self.shape.col_vars);
        match self.layout {
            RightLayout::Direct => self.b[(b * inner + j) * cols + c],
            RightLayout::Transposed => self.b[(b * cols + c) * inner + j],
        }
    }

    /// The same products with `B` stored the other way.
    pub fn relaid(&self) -> Self {
        let (inner, cols) = (1 << self.shape.inner_vars, 1 << self.shape.col_vars);
        let layout = match self.layout {
            RightLayout::Direct => RightLayout::Transposed,
            RightLayout::Transposed => RightLayout::Direct,
        };
        let b = (0..self.b.len())
            .map(|k| {
                let (stack, rest) = (k / (inner * cols), k % (inner * cols));
                match layout {
                    // Entry (b·J + j)·C + c of B.
                    RightLayout::Direct => self.b_at(stack, rest / cols, rest % cols),
                    // Entry (b·C + c)·J + j of K.
                    RightLayout::Transposed => self.b_at(stack, rest % inner, rest / inner),
                }
            })
            .collect();
        Self {
            layout,
            b,
            ..self.clone()
        }
    }

    /// `Z[b, a, c] = Σ_j A[b, a, j]·B[b, j, c]`, row-major `(b, a, c)`, in `tv-field`.
    pub fn product(&self) -> Vec<TvFp> {
        let MatMulActShape {
            stack_vars,
            row_vars,
            inner_vars,
            col_vars,
        } = self.shape;
        let (rows, inner, cols) = (1 << row_vars, 1 << inner_vars, 1 << col_vars);
        (0..1 << (stack_vars + row_vars + col_vars))
            .map(|k| {
                let (b, a, c) = (k / (rows * cols), (k / cols) % rows, k % cols);
                (0..inner).fold(TvFp::ZERO, |sum, j| {
                    sum + tv_fp(self.a[(b * rows + a) * inner + j]) * tv_fp(self.b_at(b, j, c))
                })
            })
            .collect()
    }

    /// The input claim: `Z̃(r_c ‖ r_a ‖ ρ)`.
    pub fn claim(&self) -> TvFp2 {
        extension(&self.product(), &tv_fp2s(&self.point))
    }

    /// `ρ`, the input point's stacking coordinate.
    pub fn rho(&self) -> Vec<TvFp2> {
        tv_fp2s(&self.point[self.shape.col_vars + self.shape.row_vars..])
    }

    /// The relation at the instance's point `r_j ‖ ρ_b` from the evaluations `[a, b]`:
    /// `eq(ρ, ρ_b)·a·b`, or `a·b` for a single product.
    pub fn relation_at(&self, point: &[TvFp2], evals: &[TvFp2]) -> Option<TvFp2> {
        let [a, b] = evals else { return None };
        let rho_b = point.get(self.shape.inner_vars..)?;
        if self.shape.stack_vars == 0 {
            Some(*a * *b)
        } else {
            Some(tv::eq(&self.rho(), rho_b)? * *a * *b)
        }
    }

    /// The points of the claims on `A` and on `B` as stored, from the instance's point
    /// `r_j ‖ ρ_b`: `r_j ‖ r_a ‖ ρ_b`, and `r_c ‖ r_j ‖ ρ_b`, or `r_j ‖ r_c ‖ ρ_b` for `K`.
    pub fn claim_points(&self, point: &[TvFp2]) -> [Vec<TvFp2>; 2] {
        let (r_j, rho_b) = point.split_at(self.shape.inner_vars);
        let input = tv_fp2s(&self.point);
        let (r_c, rest) = input.split_at(self.shape.col_vars);
        let r_a = &rest[..self.shape.row_vars];
        let b = match self.layout {
            RightLayout::Direct => [r_c, r_j, rho_b].concat(),
            RightLayout::Transposed => [r_j, r_c, rho_b].concat(),
        };
        [[r_j, r_a, rho_b].concat(), b]
    }
}

/// One instance of a stage.
#[derive(Clone)]
pub enum Member {
    Weight(WeightCase),
    Act(ActCase),
    /// A product of tables, as `verifier.rs` batches them.
    Product(TestInstance),
}

impl Member {
    pub fn vars(&self) -> usize {
        match self {
            Self::Weight(case) => case.vars(),
            Self::Act(case) => case.vars(),
            Self::Product(instance) => instance.vars(),
        }
    }

    /// The final evaluations it sends.
    pub fn sent(&self) -> usize {
        match self {
            Self::Weight(_) | Self::Act(_) => 2,
            Self::Product(instance) => instance.sent(),
        }
    }

    pub fn claim(&self) -> TvFp2 {
        match self {
            Self::Weight(case) => case.claim(),
            Self::Act(case) => case.claim(),
            Self::Product(instance) => instance.claim(),
        }
    }

    pub fn tv_instance(&self) -> tv::Instance {
        tv::Instance {
            claim: self.claim(),
            vars: self.vars(),
            evals: self.sent(),
        }
    }

    pub fn relation_at(&self, point: &[TvFp2], evals: &[TvFp2]) -> Option<TvFp2> {
        match self {
            Self::Weight(case) => case.relation_at(point, evals),
            Self::Act(case) => case.relation_at(point, evals),
            Self::Product(instance) => instance.relation(point, evals),
        }
    }

    /// A matmul's operands as stored, in the order of its claims: `[W, X]` or `[A, B]`.
    pub fn operands(&self) -> Option<[&[Fp]; 2]> {
        match self {
            Self::Weight(case) => Some([&case.w, &case.x]),
            Self::Act(case) => Some([&case.a, &case.b]),
            Self::Product(_) => None,
        }
    }

    /// The verifier's points of a matmul's claims, from the instance's point.
    pub fn claim_points(&self, point: &[TvFp2]) -> Option<[Vec<TvFp2>; 2]> {
        match self {
            Self::Weight(case) => Some(case.claim_points(point)),
            Self::Act(case) => Some(case.claim_points(point)),
            Self::Product(_) => None,
        }
    }
}

/// The prover's tables of a member.
enum Built<'a> {
    Weight(MatMulWeightTables<'a, Fp2>),
    Act(MatMulActTables<'a, Fp2>),
    Product(Instance<'a, Fp2>),
}

/// A proven stage: the batch, each matmul's claims on its operands in the order of its
/// relation, and the transcript's final state.
pub struct ProvenStage {
    pub proven: ProvenBatch<Fp2>,
    pub claims: Vec<Option<[Claim<Fp2>; 2]>>,
    pub state: [u8; 32],
}

/// Proves the stage on a fresh transcript of `session`, each matmul through its relation's
/// tables and claims, with the input points of the cases.
pub fn prove_stage(
    session: &Session,
    prefix: Prefix,
    degree: usize,
    members: &[Member],
) -> ProvenStage {
    let built: Vec<Built<'_>> = members
        .iter()
        .map(|member| match member {
            Member::Weight(case) => Built::Weight(
                case.relation()
                    .tables(&case.point)
                    .expect("a point of n_o + n_ℓ coordinates"),
            ),
            Member::Act(case) => Built::Act(
                case.relation()
                    .tables(&case.point)
                    .expect("a point of n_c + n_a + n_b coordinates"),
            ),
            Member::Product(instance) => Built::Product(instance.instance()),
        })
        .collect();
    let instances: Vec<Instance<'_, Fp2>> = built
        .iter()
        .map(|built| match built {
            Built::Weight(tables) => tables.instance(),
            Built::Act(tables) => tables.instance(),
            Built::Product(instance) => instance.clone(),
        })
        .collect();
    let mut transcript = session.transcript();
    let proven =
        prove_batch(&mut transcript, prefix, degree, &instances).expect("a well-formed stage");
    let claims = built
        .iter()
        .enumerate()
        .map(|(k, built)| {
            let point = proven.instance_point(k).expect("an instance");
            let evals = proven.instance_evals(k).expect("an instance");
            match built {
                Built::Weight(tables) => {
                    let claims = tables.claims(point, evals).expect("the instance's results");
                    Some([claims.w, claims.x])
                }
                Built::Act(tables) => {
                    let claims = tables.claims(point, evals).expect("the instance's results");
                    Some([claims.a, claims.b])
                }
                Built::Product(_) => None,
            }
        })
        .collect();
    ProvenStage {
        proven,
        claims,
        state: transcript.state(),
    }
}

/// Runs `tv_core::sumcheck::verify_batch` on `proof` with the members' relations, from the
/// claims `claims`, on a fresh verifier transcript of `session`.
pub fn verify_stage(
    session: &Session,
    prefix: Prefix,
    degree: usize,
    members: &[Member],
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
        |i, point, evals| members.get(i)?.relation_at(point, evals),
    );
    (result, *transcript.state())
}

/// Proves the stage and checks it with the independent verifier from the true products: it
/// accepts; its `γ`, point and final transcript state are the prover's; and every claim on an
/// operand is at the point the verifier derives, with the evaluation the verifier read, which is
/// the operand's multilinear extension at that point.
pub fn prove_and_verify_stage(
    session: &Session,
    prefix: Prefix,
    degree: usize,
    members: &[Member],
) -> ProvenStage {
    let stage = prove_stage(session, prefix, degree, members);
    let proof = stage.proven.proof();
    let n_max = members.iter().map(Member::vars).max().unwrap_or(0);
    assert_eq!(proof.rounds.len(), n_max, "one message per round");
    let claims: Vec<tv::Instance> = members.iter().map(Member::tv_instance).collect();
    let (result, tv_state) =
        verify_stage(session, prefix, degree, members, &claims, &tv_proof(proof));
    let batch = result.expect("the verifier accepts an honest stage");
    assert_eq!(batch.gamma, tv_fp2(*stage.proven.gamma()), "γ");
    assert_eq!(batch.point, tv_fp2s(stage.proven.point()), "the point");
    assert_eq!(
        tv_state, stage.state,
        "the transcripts end in the same state"
    );
    let mut offset = 0;
    for (k, (member, claims)) in members.iter().zip(&stage.claims).enumerate() {
        let evals = tv_fp2s(&proof.final_evals[offset..offset + member.sent()]);
        offset += member.sent();
        let point = batch.point_of(member.vars()).expect("an instance's point");
        let (Some(expected), Some(operands)) = (member.claim_points(point), member.operands())
        else {
            assert!(claims.is_none(), "instance {k}: claims only for a matmul");
            continue;
        };
        let claims = claims.as_ref().expect("a matmul's claims");
        for (n, ((claim, expected), operand)) in
            claims.iter().zip(expected).zip(operands).enumerate()
        {
            assert_eq!(
                tv_fp2s(&claim.point),
                expected,
                "instance {k}, operand {n}: the point the verifier derives"
            );
            assert_eq!(
                tv_fp2(claim.value),
                evals[n],
                "instance {k}, operand {n}: the evaluation the verifier read"
            );
            assert_eq!(
                tv_fp2(claim.value),
                mle(operand, &expected),
                "instance {k}, operand {n}: the operand's extension at the point"
            );
        }
    }
    assert_eq!(offset, proof.final_evals.len(), "every evaluation read");
    stage
}

/// Checks claims as the next stage checks a claim it consumes: claim `k`, on the table
/// `tables[k]`, enters as the instance `Σ_x eq(point, x)·T[x]` with the claim's value, `eq` a
/// public factor. The prover's engine proves the instances of degree 2 on a fresh transcript of
/// `session`; the verifier checks them from the claims' values, then each final evaluation
/// against its table's extension at the instance's point, as an opening would.
pub fn next_stage(
    session: &Session,
    prefix: Prefix,
    claims: &[(&[Fp], &Claim<Fp2>)],
) -> Result<(), tv::SumcheckError> {
    let eqs: Vec<Vec<Fp2>> = claims
        .iter()
        .map(|(_, claim)| tessera_poly::eq_table(&claim.point).expect("a small table"))
        .collect();
    let instances: Vec<Instance<'_, Fp2>> = claims
        .iter()
        .zip(&eqs)
        .map(|(&(table, _), eq)| Instance::new(vec![Table::ext(eq).public(), Table::base(table)]))
        .collect();
    let proven =
        prove_batch(&mut session.transcript(), prefix, 2, &instances).expect("a well-formed stage");
    let tv_claims: Vec<tv::Instance> = claims
        .iter()
        .map(|(_, claim)| tv::Instance {
            claim: tv_fp2(claim.value),
            vars: claim.point.len(),
            evals: 1,
        })
        .collect();
    let points: Vec<Vec<TvFp2>> = claims
        .iter()
        .map(|(_, claim)| tv_fp2s(&claim.point))
        .collect();
    let batch = tv::verify_batch(
        &mut session.tv_transcript(),
        tv_prefix(prefix),
        2,
        &tv_claims,
        &tv_proof(proven.proof()),
        |i, point, evals| Some(tv::eq(points.get(i)?, point)? * *evals.first()?),
    )?;
    for (k, ((table, claim), eval)) in claims.iter().zip(&proven.proof().final_evals).enumerate() {
        let point = batch
            .point_of(claim.point.len())
            .expect("an instance's point");
        assert_eq!(
            tv_fp2(*eval),
            mle(table, point),
            "claim {k}: the opening at the next stage's point"
        );
    }
    Ok(())
}
