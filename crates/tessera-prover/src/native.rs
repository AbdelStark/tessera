// SPDX-License-Identifier: Apache-2.0 OR MIT

//! CPU native model relations in the graph's single stage batch (RFC-0037/0050).

use tessera_codec::SecretBuf;
use tessera_compiler::schedule::{LayerMap, Native, NativeOutput, NodeRef, StageSchedule};
use tessera_compiler::{AuxColumn, CompareColumn, Lowering, RuntimePlan, Selection};
use tessera_field::{Field, Goldilocks as G, GoldilocksExt2 as F, PrimeField};
use tessera_gadgets::{RsqrtRow, Slack};
use tessera_ir::{ClaimList, NodeId, Op, TensorId};
use tessera_poly::{eq_table, evaluate_base, less_than};
use tessera_sumcheck::{Claim, Instance, Table, Term};

use crate::ProvenLeaf;
use crate::leaf::{LeafLayout, ModelColumn, ModelWitness};
use crate::stages::{NativeContext, NativeProver, StageError, WitnessSource};

mod runtime;
use runtime::Runtime;

fn bad() -> StageError {
    StageError::Internal {
        rule: "native model sources have their canonical identities and domains",
    }
}
fn vars(shape: &[u32]) -> u32 {
    shape
        .iter()
        .map(|n| n.next_power_of_two().trailing_zeros())
        .sum()
}
fn zeros(n: usize) -> Result<SecretBuf<G>, StageError> {
    let mut v = SecretBuf::try_with_capacity(n).map_err(|_| StageError::Memory)?;
    v.resize(n, G::ZERO);
    Ok(v)
}
fn fp(n: u64) -> F {
    F::from(G::from_u64_reduced(n))
}
fn evaluate(v: &[G], p: &[F]) -> Result<F, StageError> {
    evaluate_base(v, p).map_err(|_| bad())
}
fn selector(shape: &[u32], point: &[F]) -> Result<F, StageError> {
    let mut at = 0;
    let mut out = F::ONE;
    for &n in shape.iter().rev() {
        let end = at + vars(&[n]) as usize;
        out *= less_than(point.get(at..end).ok_or_else(bad)?, u64::from(n)).map_err(|_| bad())?;
        at = end;
    }
    if at != point.len() {
        return Err(bad());
    }
    Ok(out)
}
#[derive(Clone, Copy)]
enum Relation {
    Product,
    RootPad,
    ResidualPad,
    Key,
    Forest,
    Reciprocal,
    MaxPad,
    ReciprocalPad,
    Runtime,
}
struct Family {
    site: usize,
    nodes: Vec<NodeId>,
    shape: Vec<u32>,
    lambda: u32,
    kind: Kind,
}
enum Kind {
    Root {
        digits: usize,
        coefficients: usize,
        epsilon: u64,
        residual: TensorId,
    },
    Softmax {
        sigma: u8,
        forest: Option<usize>,
    },
    Runtime(Runtime),
}
struct Slot {
    family: usize,
    relation: Relation,
}

/// Public native descriptors and complete graph schedule for the canonical CPU model.
/// This is metadata, with no proof or acceptance verdict.
pub struct ModelNativePlan {
    families: Vec<Family>,
    slots: Vec<Slot>,
    native: Vec<Native>,
    schedule: StageSchedule,
}
impl ModelNativePlan {
    /// Derive every root coefficient, Softmax relation and padding check from public metadata.
    /// The canonical `layout` must belong to the same lowering, selection and claims.
    ///
    /// # Errors
    /// `Internal` for inconsistent metadata, or `Memory` when the geometry cannot fit.
    pub fn derive(
        lowering: &Lowering,
        selection: &Selection,
        claims: &ClaimList,
        layout: &LeafLayout<ModelColumn>,
    ) -> Result<Self, StageError> {
        Self::derive_inner(lowering, selection, claims, layout, None)
    }

    /// Derive the complete runtime schedule, replacing only the legacy weighted token
    /// boundary and appending exact-greedy relations in draft RFC-0053 order.
    ///
    /// # Errors
    /// `Internal` for inconsistent public metadata; `Memory` for unsupported geometry.
    pub fn derive_runtime(
        lowering: &Lowering,
        runtime: &RuntimePlan,
        claims: &ClaimList,
        layout: &LeafLayout<ModelColumn>,
    ) -> Result<Self, StageError> {
        Self::derive_inner(lowering, runtime.selection(), claims, layout, Some(runtime))
    }

    fn derive_inner(
        lowering: &Lowering,
        selection: &Selection,
        claims: &ClaimList,
        layout: &LeafLayout<ModelColumn>,
        runtime: Option<&RuntimePlan>,
    ) -> Result<Self, StageError> {
        let layers = LayerMap::new(lowering).map_err(|_| bad())?;
        if (runtime.is_none() && selection.tables != claims.tables)
            || !selection.tables.starts_with(&claims.tables)
        {
            return Err(bad());
        }
        let mut families = Vec::new();
        let mut forest = 0;
        for (site, s) in layout.sites().iter().enumerate() {
            let nodes: Vec<_> = (0..s.layers.len())
                .map(|l| match s.node {
                    NodeRef::Node(n) => n,
                    NodeRef::Position(p) => layers.node(p, l as u32),
                })
                .collect();
            let node = &lowering.graph.nodes[nodes.first().ok_or_else(bad)?.0 as usize];
            let kind = match node.op {
                Op::RsqrtRow {
                    dim,
                    kappa,
                    eps_int,
                } => {
                    let root =
                        RsqrtRow::new(dim, kappa, eps_int, &selection.tables).map_err(|_| bad())?;
                    let sum = lowering.graph.producer(node.inputs[0]).ok_or_else(bad)?;
                    let sum = &lowering.graph.nodes[sum.0 as usize];
                    if !matches!(sum.op, Op::SumSquares { .. }) {
                        return Err(bad());
                    }
                    let w = usize::from(root.root_bits());
                    Kind::Root {
                        digits: w.div_ceil(8),
                        coefficients: (2 * w + 55).div_ceil(8),
                        epsilon: eps_int,
                        residual: sum.inputs[0],
                    }
                }
                Op::SoftmaxRow { sigma, .. } => {
                    let shape = &lowering.graph.tensors[node.inputs[0].0 as usize].shape;
                    let own = (shape.last().copied().ok_or_else(bad)? > 1).then_some(forest);
                    forest += usize::from(own.is_some());
                    Kind::Softmax { sigma, forest: own }
                }
                _ => continue,
            };
            let tensor = match kind {
                Kind::Root { .. } => node.outputs[0],
                Kind::Softmax { .. } => node.inputs[0],
                Kind::Runtime(_) => return Err(bad()),
            };
            families.push(Family {
                site,
                nodes,
                shape: lowering.graph.tensors[tensor.0 as usize].shape.clone(),
                lambda: if matches!(s.node, NodeRef::Position(_)) {
                    layers.vars
                } else {
                    0
                },
                kind,
            });
        }
        let mut native = Vec::new();
        let mut slots = Vec::new();
        let mut deferred = Vec::new();
        for (i, f) in families.iter().enumerate() {
            let node = &lowering.graph.nodes[f.nodes[0].0 as usize];
            let local = vars(&f.shape);
            let mut add = |relation, local, degree, evaluations, zero, outputs| {
                native.push(Native {
                    tag: (slots.len() + 1) as u32,
                    node: node.id,
                    vars: local + f.lambda,
                    degree,
                    evaluations,
                    zero,
                    outputs,
                });
                slots.push(Slot {
                    family: i,
                    relation,
                });
            };
            let output = |tensor, from, vars| NativeOutput { tensor, from, vars };
            match f.kind {
                Kind::Root {
                    digits,
                    coefficients,
                    residual,
                    ..
                } => {
                    for _ in 0..2 * coefficients {
                        add(
                            Relation::Product,
                            local,
                            4,
                            digits + 4,
                            false,
                            vec![
                                output(node.inputs[0], 0, local),
                                output(node.outputs[0], 0, local),
                            ],
                        );
                    }
                    add(
                        Relation::RootPad,
                        local,
                        3,
                        1,
                        true,
                        vec![output(node.outputs[0], 0, local)],
                    );
                    let rvars = vars(&lowering.graph.tensors[residual.0 as usize].shape);
                    add(
                        Relation::ResidualPad,
                        rvars,
                        3,
                        1,
                        true,
                        vec![output(residual, 0, rvars)],
                    );
                }
                Kind::Softmax { .. } => {
                    for relation in [Relation::Key, Relation::Forest] {
                        add(
                            relation,
                            local,
                            3,
                            2,
                            false,
                            vec![output(node.inputs[0], 0, local)],
                        );
                    }
                    let row = vars(&lowering.graph.tensors[node.outputs[1].0 as usize].shape);
                    add(
                        Relation::Reciprocal,
                        row,
                        3,
                        2,
                        false,
                        vec![output(node.outputs[1], 0, row)],
                    );
                    add(Relation::MaxPad, row, 3, 1, true, Vec::new());
                    add(
                        Relation::ReciprocalPad,
                        row,
                        3,
                        1,
                        true,
                        vec![output(node.outputs[1], 0, row)],
                    );
                    deferred.push((node.id, Some(0)));
                }
                Kind::Runtime(_) => return Err(bad()),
            }
        }
        if let Some(runtime) = runtime {
            runtime::append(
                lowering,
                runtime,
                layout,
                &mut families,
                &mut slots,
                &mut native,
                &mut deferred,
            )?;
        }
        let schedule =
            tessera_compiler::schedule::derive_composed_schedule(lowering, &deferred, &native)
                .map_err(|_| bad())?;
        Ok(Self {
            families,
            slots,
            native,
            schedule,
        })
    }
    /// Complete single-batch graph/native stage schedule.
    pub fn schedule(&self) -> &StageSchedule {
        &self.schedule
    }
    /// Native descriptors in stable tag order.
    pub fn native(&self) -> &[Native] {
        &self.native
    }
}

fn product_terms(digits: usize, j: usize, upper: bool) -> Vec<Term<F>> {
    let eq = digits + 4;
    let mut terms = Vec::new();
    // Literal coefficient expansion preserves the constant-one upper increment.
    for a in 0..digits {
        for b in 0..digits {
            for k in 0..4 {
                if a + b + 2 * k != j {
                    continue;
                }
                terms.push(Term::new(F::ONE, vec![a, b, digits + k, eq]));
                if upper && a == 0 {
                    terms.push(Term::new(F::ONE, vec![b, digits + k, eq]));
                }
                if upper && b == 0 {
                    terms.push(Term::new(F::ONE, vec![a, digits + k, eq]));
                }
                if upper && a == 0 && b == 0 {
                    terms.push(Term::new(F::ONE, vec![digits + k, eq]));
                }
            }
        }
    }
    if terms.is_empty() {
        // RFC-0050 retains every high coefficient, including identically zero ones.
        terms.push(Term::new(F::ZERO, vec![0, 0, digits, eq]));
    }

    terms
}

struct Data {
    raw: Vec<usize>,
    mask: Option<Vec<G>>,
    terms: Vec<Term<F>>,
    fixed: Vec<F>,
    point: Vec<F>,
    full_point: bool,
    equality: Vec<F>,
    target: F,
    public: Vec<Vec<F>>,
}
/// Clearing resident CPU tables for every native relation in [`ModelNativePlan`].
/// All returned final evaluations still owe their original graph/auxiliary openings.
pub struct ModelNativeProver<'a> {
    plan: &'a ModelNativePlan,
    tables: Vec<SecretBuf<G>>,
    data: Vec<Data>,
}
impl<'a> ModelNativeProver<'a> {
    /// Build native tables from the same original witness and canonical leaf proof.
    /// The leaf proof fixes all incoming targets before the mixed stage draws gamma.
    ///
    /// # Errors
    /// `Internal` for inconsistent own sources/points, `Memory` for failed reservations.
    pub fn new(
        plan: &'a ModelNativePlan,
        lowering: &Lowering,
        witness: &ModelWitness,
        graph: &impl WitnessSource,
        leaf: &ProvenLeaf,
    ) -> Result<Self, StageError> {
        let mut out = Self {
            plan,
            tables: Vec::new(),
            data: Vec::new(),
        };
        for (index, family) in plan.families.iter().enumerate() {
            out.family(index, family, lowering, witness, graph, leaf)?;
        }
        if out.data.len() != plan.slots.len() {
            return Err(bad());
        }
        Ok(out)
    }
    fn table(&mut self, v: SecretBuf<G>) -> usize {
        self.tables.push(v);
        self.tables.len() - 1
    }
    fn source(
        &mut self,
        f: &Family,
        witness: &ModelWitness,
        c: ModelColumn,
    ) -> Result<usize, StageError> {
        let block = witness.values(f.site, 0, c).ok_or_else(bad)?.len();
        let len = block.checked_shl(f.lambda).ok_or(StageError::Memory)?;
        let mut v = SecretBuf::try_with_capacity(len).map_err(|_| StageError::Memory)?;
        for l in 0..f.nodes.len() {
            let s = witness.values(f.site, l, c).ok_or_else(bad)?;
            if s.len() != block {
                return Err(bad());
            }
            v.extend(s.iter().map(|&x| G::from_i64(x)));
        }
        v.resize(len, G::ZERO);
        Ok(self.table(v))
    }
    fn fixed(
        &self,
        f: &Family,
        witness: &ModelWitness,
        column: ModelColumn,
        point: &[F],
    ) -> Result<Vec<F>, StageError> {
        (0..f.nodes.len())
            .map(|l| {
                let values = witness.values(f.site, l, column).ok_or_else(bad)?;
                let mut buffer =
                    SecretBuf::try_with_capacity(values.len()).map_err(|_| StageError::Memory)?;
                buffer.extend(values.iter().map(|&v| G::from_i64(v)));
                evaluate(&buffer, point)
            })
            .collect()
    }
    fn add(
        &mut self,
        raw: Vec<usize>,
        mask: Option<Vec<G>>,
        terms: Vec<Term<F>>,
        fixed: Vec<F>,
        point: Vec<F>,
        full_point: bool,
    ) {
        self.data.push(Data {
            raw,
            mask,
            terms,
            fixed,
            point,
            full_point,
            equality: Vec::new(),
            target: F::ZERO,
            public: Vec::new(),
        });
    }
    fn padding(&mut self, raw: usize, shape: &[u32]) -> Result<(), StageError> {
        let n = self.tables[raw].len();
        let local = vars(shape);
        let block = 1usize.checked_shl(local).ok_or(StageError::Memory)?;
        let layers = self.plan.families[self.plan.slots[self.data.len()].family]
            .nodes
            .len();
        let mask = (0..n)
            .map(|i| {
                let mut at = i % block;
                let mut real = i / block < layers;
                for &d in shape.iter().rev() {
                    let padded = d.next_power_of_two() as usize;
                    real &= at % padded < d as usize;
                    at /= padded;
                }
                if real { G::ZERO } else { G::ONE }
            })
            .collect();
        self.add(
            vec![raw],
            Some(mask),
            vec![Term::new(F::ONE, vec![0, 1, 2])],
            Vec::new(),
            Vec::new(),
            false,
        );
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn family(
        &mut self,
        index: usize,
        f: &Family,
        lowering: &Lowering,
        witness: &ModelWitness,
        graph: &impl WitnessSource,
        leaf: &ProvenLeaf,
    ) -> Result<(), StageError> {
        let local = vars(&f.shape) as usize;
        let p = leaf.point().get(..local).ok_or_else(bad)?;
        let node = &lowering.graph.nodes[f.nodes[0].0 as usize];
        match f.kind {
            Kind::Runtime(kind) => {
                return runtime::build(self, f, kind, lowering, witness, graph, leaf);
            }
            Kind::Root {
                digits,
                coefficients,
                epsilon: _,
                residual,
            } => {
                let rho = self.source(f, witness, ModelColumn::Output(0))?;
                let u = self.source(f, witness, ModelColumn::RootInput)?;
                let mut raw = Vec::new();
                for (len, base, whole, column) in [
                    (
                        digits,
                        8u32,
                        rho,
                        AuxColumn::RootByte as fn(u32) -> AuxColumn,
                    ),
                    (4, 16, u, AuxColumn::InputLimb),
                ] {
                    let mut top = zeros(self.tables[whole].len())?;
                    top.copy_from_slice(&self.tables[whole]);
                    for j in 0..len - 1 {
                        let low =
                            self.source(f, witness, ModelColumn::Auxiliary(column(j as u32)))?;
                        let scale = G::from_u64_reduced(1u64 << (base * j as u32));
                        for (v, &x) in top.iter_mut().zip(self.tables[low].iter()) {
                            *v -= scale * x;
                        }
                        raw.push(low);
                    }
                    let inverse = G::from_u64_reduced(1u64 << (base * (len - 1) as u32))
                        .inverse()
                        .ok_or_else(bad)?;
                    for v in top.iter_mut() {
                        *v *= inverse;
                    }
                    raw.push(self.table(top));
                }
                for upper in [false, true] {
                    for j in 0..coefficients {
                        let column = if upper {
                            CompareColumn::RightByte
                        } else {
                            CompareColumn::LeftByte
                        };
                        let carry = if upper {
                            CompareColumn::RightCarry
                        } else {
                            CompareColumn::LeftCarry
                        };
                        let mut fixed = vec![F::ZERO; f.nodes.len()];
                        for (role, limb, coefficient) in [(column, j, fp(1)), (carry, j, fp(256))]
                            .into_iter()
                            .chain(j.checked_sub(1).map(|k| (carry, k, -F::ONE)))
                        {
                            let values = self.fixed(
                                f,
                                witness,
                                ModelColumn::Auxiliary(AuxColumn::Comparison {
                                    upper,
                                    column: role,
                                    limb: limb as u32,
                                }),
                                p,
                            )?;
                            for (v, x) in fixed.iter_mut().zip(values) {
                                *v += coefficient * x;
                            }
                        }
                        let terms = product_terms(digits, j, upper);
                        self.add(raw.clone(), None, terms, fixed, p.to_vec(), false);
                    }
                }
                self.padding(rho, &f.shape)?;
                let residual_shape = &lowering.graph.tensors[residual.0 as usize].shape;
                let block = 1usize
                    .checked_shl(vars(residual_shape))
                    .ok_or(StageError::Memory)?;
                let mut values = zeros(block << f.lambda)?;
                for (l, &id) in f.nodes.iter().enumerate() {
                    let root = &lowering.graph.nodes[id.0 as usize];
                    let sum = lowering.graph.producer(root.inputs[0]).ok_or_else(bad)?;
                    let t = lowering.graph.nodes[sum.0 as usize].inputs[0];
                    pad(
                        graph.values(t).ok_or_else(bad)?,
                        residual_shape,
                        &mut values[l * block..(l + 1) * block],
                    )?;
                }
                let id = self.table(values);
                self.padding(id, residual_shape)?;
            }
            Kind::Softmax { sigma, forest } => {
                let &[heads, n, keys] = f.shape.as_slice() else {
                    return Err(bad());
                };
                if n != keys {
                    return Err(bad());
                }
                let np = n.next_power_of_two() as usize;
                let rows = heads.next_power_of_two() as usize * np;
                let max = self.source(f, witness, ModelColumn::Auxiliary(AuxColumn::Maximum))?;
                let mut broadcast = zeros(self.tables[max].len() * np)?;
                for (r, &v) in self.tables[max].iter().enumerate() {
                    broadcast[r * np..(r + 1) * np].fill(v);
                }
                let maximum = self.table(broadcast);
                let mut scores = zeros(self.tables[maximum].len())?;
                let block = rows * np;
                for (l, &id) in f.nodes.iter().enumerate() {
                    let input = lowering.graph.nodes[id.0 as usize].inputs[0];
                    pad(
                        graph.values(input).ok_or_else(bad)?,
                        &f.shape,
                        &mut scores[l * block..(l + 1) * block],
                    )?;
                }
                let score = self.table(scores);
                let mask: Vec<_> = (0..self.tables[score].len())
                    .map(|i| {
                        let (l, h, q, k) = (i / block, i % block / (np * np), i / np % np, i % np);
                        if l < f.nodes.len() && h < heads as usize && q < n as usize && k <= q {
                            G::ONE
                        } else {
                            G::ZERO
                        }
                    })
                    .collect();
                let mut fixed = self.fixed(f, witness, ModelColumn::ExpKey, p)?;
                let causal = evaluate(&mask[..block], p)?;
                for (v, &id) in fixed.iter_mut().zip(&f.nodes) {
                    let Op::SoftmaxRow { exp_table, .. } = lowering.graph.nodes[id.0 as usize].op
                    else {
                        return Err(bad());
                    };
                    let spec = lowering
                        .tables
                        .iter()
                        .find(|t| t.id == exp_table)
                        .ok_or_else(bad)?;
                    *v -= fp((1u64 << spec.log2_rows) - 1) * (F::ONE - causal);
                }
                let terms = vec![
                    Term::new(F::ONE, vec![0, 2, 3]),
                    Term::new(-F::ONE, vec![1, 2, 3]),
                ];
                self.add(
                    vec![maximum, score],
                    Some(mask.clone()),
                    terms.clone(),
                    fixed,
                    p.to_vec(),
                    false,
                );
                let (point, target) = if let Some(k) = forest {
                    let c = leaf.logup().forest_claims().get(k).ok_or_else(bad)?;
                    (
                        c.point.clone(),
                        c.value - F::ONE + evaluate(&mask, &c.point)?,
                    )
                } else {
                    (p.to_vec(), F::ZERO)
                };
                self.add(
                    vec![maximum, score],
                    Some(mask),
                    terms,
                    vec![target],
                    point,
                    forest.is_some(),
                );
                let row_shape = &lowering.graph.tensors[node.outputs[1].0 as usize].shape;
                let rowvars = vars(row_shape) as usize;
                let rowp = leaf.point().get(..rowvars).ok_or_else(bad)?;
                let reciprocal = self.source(f, witness, ModelColumn::Output(1))?;
                let exp = self.source(f, witness, ModelColumn::Output(0))?;
                let mut average = zeros(self.tables[reciprocal].len())?;
                let inverse = G::from_u64_reduced(np as u64).inverse().ok_or_else(bad)?;
                for (v, chunk) in average.iter_mut().zip(self.tables[exp].chunks_exact(np)) {
                    *v = chunk.iter().copied().fold(G::ZERO, |a, x| a + x) * inverse;
                }
                let average = self.table(average);
                let constant = fp(1u64 << (2 * u32::from(sigma))) * selector(row_shape, rowp)?;
                let fixed = self
                    .fixed(f, witness, ModelColumn::Slack(Slack::A), rowp)?
                    .into_iter()
                    .map(|a| constant - a)
                    .collect();
                self.add(
                    vec![reciprocal, average],
                    None,
                    vec![Term::new(fp(np as u64), vec![0, 1, 2])],
                    fixed,
                    rowp.to_vec(),
                    false,
                );
                self.padding(max, row_shape)?;
                self.padding(reciprocal, row_shape)?;
            }
        }
        if self
            .plan
            .slots
            .get(self.data.len().saturating_sub(1))
            .is_none_or(|s| s.family != index)
        {
            return Err(bad());
        }
        Ok(())
    }
}
fn pad(values: &[i64], shape: &[u32], out: &mut [G]) -> Result<(), StageError> {
    if values.len() != shape.iter().map(|&n| n as usize).product::<usize>()
        || out.len() != 1usize.checked_shl(vars(shape)).ok_or(StageError::Memory)?
    {
        return Err(bad());
    }
    for (flat, &v) in values.iter().enumerate() {
        let (mut rest, mut index, mut shift) = (flat, 0, 0);
        for &n in shape.iter().rev() {
            index |= (rest % n as usize) << shift;
            rest /= n as usize;
            shift += vars(&[n]);
        }
        out[index] = G::from_i64(v);
    }
    Ok(())
}
impl NativeProver for ModelNativeProver<'_> {
    fn prepare(&mut self, ctx: NativeContext<'_>) -> Result<(), StageError> {
        if ctx.stage != 1
            || ctx.layer.len() != self.plan.schedule.layers.vars as usize
            || ctx.zero.len() != self.plan.schedule.stages[1].zero_vars as usize
        {
            return Err(bad());
        }
        for ((data, meta), slot) in self
            .data
            .iter_mut()
            .zip(&self.plan.native)
            .zip(&self.plan.slots)
        {
            let family = &self.plan.families[slot.family];
            if let Kind::Runtime(kind) = family.kind {
                runtime::prepare(data, family, kind, &ctx)?;
                continue;
            }
            let layer = if family.lambda == 0 {
                &[][..]
            } else {
                ctx.layer
            };
            let target = if meta.zero {
                ctx.zero
                    .get(
                        ctx.zero
                            .len()
                            .checked_sub(meta.vars as usize)
                            .ok_or_else(bad)?..,
                    )
                    .ok_or_else(bad)?
                    .to_vec()
            } else {
                let mut p = data.point.clone();
                if !data.full_point {
                    p.extend_from_slice(layer);
                }
                p
            };
            if target.len() != meta.vars as usize {
                return Err(bad());
            }
            data.equality = eq_table(&target).map_err(|_| bad())?;
            data.target = if meta.zero {
                F::ZERO
            } else if data.full_point {
                *data.fixed.first().ok_or_else(bad)?
            } else {
                let weights = eq_table(layer).map_err(|_| bad())?;
                data.fixed
                    .iter()
                    .zip(weights)
                    .fold(F::ZERO, |sum, (&x, w)| sum + x * w)
            };
        }
        Ok(())
    }
    fn claim(&self, tag: u32) -> Result<F, StageError> {
        Ok(self
            .data
            .get(tag.checked_sub(1).ok_or_else(bad)? as usize)
            .ok_or_else(bad)?
            .target)
    }
    fn instance(&self, tag: u32) -> Result<Instance<'_, F>, StageError> {
        let d = self
            .data
            .get(tag.checked_sub(1).ok_or_else(bad)? as usize)
            .ok_or_else(bad)?;
        let mut tables: Vec<_> = d
            .raw
            .iter()
            .map(|&i| Table::base(&self.tables[i]))
            .collect();
        if !d.equality.is_empty() {
            tables.push(Table::ext(&d.equality).public());
        }
        if let Some(m) = &d.mask {
            tables.push(Table::base(m).public());
        }
        tables.extend(d.public.iter().map(|v| Table::ext(v).public()));
        Ok(Instance::sum_of_products(tables, d.terms.clone()))
    }
    fn claims(&self, tag: u32, point: &[F], values: &[F]) -> Result<Vec<Claim<F>>, StageError> {
        let i = tag.checked_sub(1).ok_or_else(bad)? as usize;
        let meta = self.plan.native.get(i).ok_or_else(bad)?;
        if point.len() != meta.vars as usize || values.len() != meta.evaluations {
            return Err(bad());
        }
        let slot = &self.plan.slots[i];
        let f = &self.plan.families[slot.family];
        let output = |v| Claim {
            point: point.to_vec(),
            value: v,
        };
        Ok(match (slot.relation, &f.kind) {
            (Relation::Runtime, Kind::Runtime(Runtime::Tokens)) => vec![
                output(values[0]),
                Claim {
                    point: point[vars(&[f.shape[1]]) as usize..].to_vec(),
                    value: values[1],
                },
            ],
            (Relation::Runtime, Kind::Runtime(_)) => vec![output(values[0])],
            (
                Relation::Product,
                Kind::Root {
                    digits, epsilon, ..
                },
            ) => {
                let rho = values[..*digits]
                    .iter()
                    .enumerate()
                    .fold(F::ZERO, |s, (j, &v)| s + fp(1u64 << (8 * j)) * v);
                let u = values[*digits..]
                    .iter()
                    .enumerate()
                    .fold(F::ZERO, |s, (j, &v)| s + fp(1u64 << (16 * j)) * v);
                let local = point.len() - f.lambda as usize;
                let real = selector(&f.shape, &point[..local])?
                    * less_than(&point[local..], f.nodes.len() as u64).map_err(|_| bad())?;
                vec![output(u - fp(*epsilon) * real), output(rho)]
            }
            (Relation::Key | Relation::Forest, _) => vec![output(values[1])],
            (
                Relation::Reciprocal
                | Relation::RootPad
                | Relation::ResidualPad
                | Relation::ReciprocalPad,
                _,
            ) => vec![output(values[0])],
            (Relation::MaxPad, _) => Vec::new(),
            _ => return Err(bad()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tessera_field::QuadraticExtension;
    #[test]
    fn every_quartic_coefficient_matches_the_frozen_integer_oracle() {
        let document: serde_json::Value = serde_json::from_str(include_str!(
            "../../../spec/vectors/gadget-finals/roots.json"
        ))
        .unwrap();
        let field = |v: &serde_json::Value| G::from_canonical_u64(v.as_u64().unwrap()).unwrap();
        let ext = |v: &serde_json::Value| F::new(field(&v[0]), field(&v[1]));
        for case in document["cases"].as_array().unwrap() {
            for root in case["roots"].as_array().unwrap() {
                let digits = root["root_digits"].as_array().unwrap();
                let inputs = root["u_digits"].as_array().unwrap();
                let point: Vec<_> = root["point"].as_array().unwrap().iter().map(ext).collect();
                let equality = eq_table(&point).unwrap();
                let targets = root["product_targets"].as_array().unwrap();
                let count = targets.len() / 2;
                for (i, target) in targets.iter().enumerate() {
                    let terms = product_terms(digits.len(), i % count, i >= count);
                    let mut sum = F::ZERO;
                    for (row, &weight) in equality.iter().enumerate() {
                        let values: Vec<_> = digits
                            .iter()
                            .chain(inputs)
                            .map(|v| F::from(field(&v[row])))
                            .chain([weight])
                            .collect();
                        sum += terms.iter().fold(F::ZERO, |sum, t| {
                            sum + *t.weight()
                                * t.factors()
                                    .iter()
                                    .fold(F::ONE, |product, &j| product * values[j])
                        });
                    }
                    assert_eq!(sum, ext(target), "{} coefficient {i}", case["name"]);
                }
            }
        }
    }
}
