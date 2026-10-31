// SPDX-License-Identifier: Apache-2.0 OR MIT

//! CPU native model relations in the graph's single stage batch (RFC-0037/0050).

use std::collections::HashMap;
use std::sync::Arc;

use tessera_codec::SecretBuf;
use tessera_compiler::schedule::{
    LayerMap, Native, NativeKind, NativeOutput, NodeRef, StageSchedule,
};
use tessera_compiler::{AuxColumn, CompareColumn, Lowering, RuntimePlan, Selection};
use tessera_field::{Field, Goldilocks as G, GoldilocksExt2 as F, PrimeField};
use tessera_gadgets::{RsqrtRow, Slack};
use tessera_ir::{ClaimList, NodeId, Op, TensorId};
use tessera_poly::{PolyField, eq_table, evaluate_base, less_than};
use tessera_sumcheck::{Claim, Instance, Table, Term};

use crate::leaf::{LeafClaims, LeafLayout, ModelColumn, ModelWitness, token, tokens};
use crate::stages::{NativeContext, NativeProver, StageError, WitnessSource};

mod openings;
mod runtime;
mod score;
use runtime::Runtime;
pub use score::inputs::ScoreInputMemory;

fn bad() -> StageError {
    StageError::Internal {
        rule: "native model sources have their canonical identities and domains",
    }
}
fn gadget_error(error: tessera_gadgets::GadgetError) -> StageError {
    if error.code() == tessera_codec::error::ErrorCode::ResMemory {
        StageError::Memory
    } else {
        bad()
    }
}
fn vars(shape: &[u32]) -> u32 {
    shape
        .iter()
        .map(|n| n.next_power_of_two().trailing_zeros())
        .sum()
}

/// Evaluate a borrowed integer column without converting it to a full field table. The two
/// public equality halves use O(sqrt(entries)) storage even for score-sized leaf columns.
fn evaluate_integers(values: &[i64], point: &[F]) -> Result<F, StageError> {
    let variables = u32::try_from(point.len()).map_err(|_| bad())?;
    if 1usize.checked_shl(variables) != Some(values.len()) {
        return Err(bad());
    }
    let split = point.len() / 2;
    let low = eq_table(&point[..split]).map_err(|_| bad())?;
    let high = eq_table(&point[split..]).map_err(|_| bad())?;
    Ok(values
        .chunks_exact(low.len())
        .zip(high)
        .fold(F::ZERO, |sum, (chunk, weight)| {
            let inner = chunk
                .iter()
                .zip(&low)
                .fold(F::ZERO, |sum, (&v, &w)| sum + w.mul_by_base(G::from_i64(v)));
            sum + weight * inner
        }))
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

impl Relation {
    /// The compiler's kind of a model relation, which the descriptor carries (RFC-0013 'Degree'
    /// keys the instance by it); `None` for a request's, whose kind its [`Runtime`] gives
    /// (`runtime::append`).
    fn kind(self) -> Option<NativeKind> {
        Some(match self {
            Self::Product => NativeKind::Product,
            Self::RootPad => NativeKind::RootPad,
            Self::ResidualPad => NativeKind::ResidualPad,
            Self::Key => NativeKind::Key,
            Self::Forest => NativeKind::Forest,
            Self::Reciprocal => NativeKind::Reciprocal,
            Self::MaxPad => NativeKind::MaxPad,
            Self::ReciprocalPad => NativeKind::ReciprocalPad,
            Self::Runtime => return None,
        })
    }
}
struct Family {
    windows: Vec<tessera_compiler::GreedyKeyColumn>,
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
    batch: Option<tessera_ir::BatchSlots>,
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
                    let root = RsqrtRow::new(dim, kappa, eps_int, &selection.tables)
                        .map_err(gadget_error)?;
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
                windows: Vec::new(),
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
            let mut add = |relation: Relation, local, degree, evaluations, zero, outputs| {
                native.push(Native {
                    tag: (slots.len() + 1) as u32,
                    kind: relation.kind().ok_or_else(bad)?,
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
                Ok::<_, StageError>(())
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
                        )?;
                    }
                    add(
                        Relation::RootPad,
                        local,
                        3,
                        1,
                        true,
                        vec![output(node.outputs[0], 0, local)],
                    )?;
                    let rvars = vars(&lowering.graph.tensors[residual.0 as usize].shape);
                    add(
                        Relation::ResidualPad,
                        rvars,
                        3,
                        1,
                        true,
                        vec![output(residual, 0, rvars)],
                    )?;
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
                        )?;
                    }
                    let row = vars(&lowering.graph.tensors[node.outputs[1].0 as usize].shape);
                    add(
                        Relation::Reciprocal,
                        row,
                        3,
                        2,
                        false,
                        vec![output(node.outputs[1], 0, row)],
                    )?;
                    add(Relation::MaxPad, row, 3, 1, true, Vec::new())?;
                    add(
                        Relation::ReciprocalPad,
                        row,
                        3,
                        1,
                        true,
                        vec![output(node.outputs[1], 0, row)],
                    )?;
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
            batch: lowering.graph.slots.clone(),
            families,
            slots,
            native,
            schedule,
        })
    }
    /// Public resident native table geometry, with each raw evaluation and public factor charged.
    pub(crate) fn window_shape(
        &self,
        tag: u32,
    ) -> Result<tessera_sumcheck::WindowShape, StageError> {
        let index = tag.checked_sub(1).ok_or_else(bad)? as usize;
        let meta = self.native.get(index).ok_or_else(bad)?;
        let slot = self.slots.get(index).ok_or_else(bad)?;
        let family = &self.families[slot.family];
        let public = match (&family.kind, slot.relation) {
            (Kind::Runtime(Runtime::Tokens), _) => 2,
            (Kind::Runtime(Runtime::Observe), _) => 0,
            (Kind::Runtime(Runtime::Key(..)), _) => 4 + vars(&[family.shape[0]]) as usize,
            (Kind::Runtime(Runtime::Selected(_)), _) => 3 + vars(&[family.shape[0]]) as usize,
            (_, Relation::Product | Relation::Reciprocal) => 1,
            _ => 2,
        };
        Ok(tessera_sumcheck::WindowShape {
            vars: meta.vars as usize,
            tables: meta.evaluations + public,
            delta: meta.degree,
            windowable: false,
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
    score: Option<score::Score>,
    raw: Vec<usize>,
    mask: Option<Vec<G>>,
    terms: Vec<Term<F>>,
    fixed: Vec<F>,
    point: Vec<F>,
    full_point: bool,
    equality: Arc<Vec<F>>,
    target: F,
    public: Vec<Vec<F>>,
}
/// Clearing resident CPU tables for every native relation in [`ModelNativePlan`].
/// All returned final evaluations still owe their original graph/auxiliary openings.
pub struct ModelNativeProver<'a> {
    plan: &'a ModelNativePlan,
    tables: Vec<SecretBuf<G>>,
    data: Vec<Data>,
    literal_scores: bool,
}
impl<'a> ModelNativeProver<'a> {
    /// Build native tables from the same original witness and canonical leaf proof.
    /// The leaf proof fixes all incoming targets before the mixed stage draws gamma. Each
    /// family's targets lie where the leaf stage left its tensors' claims ([`LeafClaims`]):
    /// prefixes of its point, or the merged points of a grouped stage (RFC-0054 'The stage's
    /// outputs').
    ///
    /// Single-request score relations use computed RFC-0056 rounds. Multi-request
    /// RFC-0071 slots retain the slot-aware literal path; no computed working-set
    /// reduction is claimed for those batches.
    ///
    /// # Errors
    /// `Internal` for inconsistent own sources/points, `Memory` for failed reservations.
    pub fn new(
        plan: &'a ModelNativePlan,
        lowering: &Lowering,
        witness: &ModelWitness,
        graph: &impl WitnessSource,
        leaf: &dyn LeafClaims,
    ) -> Result<Self, StageError> {
        Self::build(plan, lowering, witness, graph, leaf, false)
    }

    /// Literal RFC-0010 oracle: materialize the score, broadcast maximum, equality and mask
    /// tensors. This sends the same messages as [`Self::new`] but uses quadratic memory.
    ///
    /// # Errors
    /// The same source/geometry and reservation failures as [`Self::new`].
    pub fn new_literal(
        plan: &'a ModelNativePlan,
        lowering: &Lowering,
        witness: &ModelWitness,
        graph: &impl WitnessSource,
        leaf: &dyn LeafClaims,
    ) -> Result<Self, StageError> {
        Self::build(plan, lowering, witness, graph, leaf, true)
    }

    fn build(
        plan: &'a ModelNativePlan,
        lowering: &Lowering,
        witness: &ModelWitness,
        graph: &impl WitnessSource,
        leaf: &dyn LeafClaims,
        literal_scores: bool,
    ) -> Result<Self, StageError> {
        let mut out = Self {
            plan,
            tables: Vec::new(),
            data: Vec::new(),
            literal_scores,
        };
        // A claim's point reads its tensor's token axis only in token groups (RFC-0054 'The
        // stage's outputs'), whose axes RFC-0054 derives; an ungrouped stage needs none, and a
        // graph holding RFC-0066's operations has none (RFC-0072 'Non-Goals' 'Token groups').
        let axes = match leaf.token_bits() {
            0 => Vec::new(),
            _ => tokens(lowering).map_err(|_| bad())?,
        };
        for (index, family) in plan.families.iter().enumerate() {
            out.family(index, family, lowering, witness, graph, leaf, &axes)?;
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
                evaluate_integers(values, point)
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
            score: None,
            raw,
            mask,
            terms,
            fixed,
            point,
            full_point,
            equality: Arc::default(),
            target: F::ZERO,
            public: Vec::new(),
        });
    }
    fn padding(&mut self, raw: usize, shape: &[u32], token_axis: usize) -> Result<(), StageError> {
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
                for (axis, &d) in shape.iter().enumerate().rev() {
                    let padded = d.next_power_of_two() as usize;
                    real &= at % padded < d as usize;
                    if axis == token_axis
                        && let Some(batch) = &self.plan.batch
                    {
                        real &= batch.is_real_row(at % padded);
                    }
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
        leaf: &dyn LeafClaims,
        axes: &[Option<u8>],
    ) -> Result<(), StageError> {
        let node = &lowering.graph.nodes[f.nodes[0].0 as usize];
        // The family's leaf point: its tensor's, the root's output or the softmax's scores, of
        // the family's shape (RFC-0054 'The stage's outputs').
        let tensor = match f.kind {
            Kind::Root { .. } => node.outputs.first().copied(),
            Kind::Softmax { .. } => node.inputs.first().copied(),
            Kind::Runtime(_) => None,
        };
        let point = match tensor {
            Some(t) => leaf
                .tensor_point(&f.shape, token(axes, t))
                .ok_or_else(bad)?,
            None => Vec::new(),
        };
        let p = point.as_slice();
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
                self.padding(rho, &f.shape, 0)?;
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
                self.padding(id, residual_shape, 0)?;
            }
            Kind::Softmax { sigma, forest } => {
                let &[heads, n, keys] = f.shape.as_slice() else {
                    return Err(bad());
                };
                let np = keys.next_power_of_two() as usize;
                let qp = n.next_power_of_two() as usize;
                let rows = heads.next_power_of_two() as usize * qp;
                let max = self.source(f, witness, ModelColumn::Auxiliary(AuxColumn::Maximum))?;
                // RFC-0071 batches have a slot-local key axis and causal mask.
                // RFC-0056's computed input geometry below is single-request only.
                // Preserve the independently checked batch path until a slot-aware
                // computed algorithm has its own equivalence and memory evidence.
                if self.literal_scores || lowering.graph.slots.is_some() {
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
                            let (l, h, q, k) =
                                (i / block, i % block / (np * qp), i / np % qp, i % np);
                            let (real, position) = match &lowering.graph.slots {
                                Some(slots) => (slots.is_real_row(q), q % slots.slot_len as usize),
                                None => (q < n as usize, q),
                            };
                            if l < f.nodes.len() && h < heads as usize && real && k <= position {
                                G::ONE
                            } else {
                                G::ZERO
                            }
                        })
                        .collect();
                    let mut fixed = self.fixed(f, witness, ModelColumn::ExpKey, p)?;
                    let causal = evaluate(&mask[..block], p)?;
                    for (v, &id) in fixed.iter_mut().zip(&f.nodes) {
                        let Op::SoftmaxRow { exp_table, .. } =
                            lowering.graph.nodes[id.0 as usize].op
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
                        let c = leaf.forest_claim(k).ok_or_else(bad)?;
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
                } else {
                    let inputs = Arc::new(score_inputs(f, lowering, graph, &self.tables[max])?);
                    let mut fixed = self.fixed(f, witness, ModelColumn::ExpKey, p)?;
                    let causal = score::mask(p, heads as usize, n as usize, 1);
                    for (v, &id) in fixed.iter_mut().zip(&f.nodes) {
                        let Op::SoftmaxRow { exp_table, .. } =
                            lowering.graph.nodes[id.0 as usize].op
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
                    self.add(Vec::new(), None, Vec::new(), fixed, p.to_vec(), false);
                    self.data.last_mut().ok_or_else(bad)?.score = Some(score::Score {
                        inputs: Arc::clone(&inputs),
                        point: Vec::new(),
                    });
                    let (point, target) = if let Some(k) = forest {
                        let c = leaf.forest_claim(k).ok_or_else(bad)?;
                        (
                            c.point.clone(),
                            c.value - F::ONE
                                + score::mask(&c.point, heads as usize, n as usize, f.nodes.len()),
                        )
                    } else {
                        (p.to_vec(), F::ZERO)
                    };
                    self.add(
                        Vec::new(),
                        None,
                        Vec::new(),
                        vec![target],
                        point,
                        forest.is_some(),
                    );
                    self.data.last_mut().ok_or_else(bad)?.score = Some(score::Score {
                        inputs,
                        point: Vec::new(),
                    });
                }
                let row_shape = &lowering.graph.tensors[node.outputs[1].0 as usize].shape;
                let rows = leaf
                    .tensor_point(row_shape, token(axes, node.outputs[1]))
                    .ok_or_else(bad)?;
                let rowp = rows.as_slice();
                let reciprocal = self.source(f, witness, ModelColumn::Output(1))?;
                let mut average = zeros(self.tables[reciprocal].len())?;
                let inverse = G::from_u64_reduced(np as u64).inverse().ok_or_else(bad)?;
                let rows = average.len() >> f.lambda;
                for l in 0..f.nodes.len() {
                    let exp = witness
                        .values(f.site, l, ModelColumn::Output(0))
                        .ok_or_else(bad)?;
                    if exp.len() != rows * np {
                        return Err(bad());
                    }
                    for (v, chunk) in average[l * rows..(l + 1) * rows]
                        .iter_mut()
                        .zip(exp.chunks_exact(np))
                    {
                        *v = chunk.iter().fold(G::ZERO, |sum, &x| sum + G::from_i64(x)) * inverse;
                    }
                }
                let average = self.table(average);
                let psi = match &lowering.graph.slots {
                    Some(slots) => {
                        let time = &rowp[..vars(&[n]) as usize];
                        let head = &rowp[time.len()..];
                        tessera_poly::slot_selector(
                            time,
                            slots.slot_len.trailing_zeros() as usize,
                            &slots
                                .lengths
                                .iter()
                                .map(|&n| u64::from(n))
                                .collect::<Vec<_>>(),
                        )
                        .map_err(|_| bad())?
                            * selector(&[heads], head)?
                    }
                    None => selector(row_shape, rowp)?,
                };
                let constant = fp(1u64 << (2 * u32::from(sigma))) * psi;
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
                self.padding(max, row_shape, 1)?;
                self.padding(reciprocal, row_shape, 1)?;
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
fn score_inputs(
    f: &Family,
    lowering: &Lowering,
    graph: &impl WitnessSource,
    maximum: &[G],
) -> Result<score::Inputs, StageError> {
    let shape = score::inputs::Shape::derive(f, lowering)?;
    if maximum.len() != shape.maxima {
        return Err(bad());
    }
    let mut raw_queries =
        SecretBuf::try_with_capacity(shape.raw_queries).map_err(|_| StageError::Memory)?;
    let mut raw_keys =
        SecretBuf::try_with_capacity(shape.raw_keys).map_err(|_| StageError::Memory)?;
    for &id in &f.nodes {
        let (q, k) = score::inputs::ids(lowering, id)?;
        for (id, out, size) in [
            (q, &mut raw_queries, shape.raw_queries / f.nodes.len()),
            (k, &mut raw_keys, shape.raw_keys / f.nodes.len()),
        ] {
            let values = graph.values(id).ok_or_else(bad)?;
            if values.len() != size {
                return Err(bad());
            }
            // The public range guarantees INT8; keep the conversion branch-free.
            out.extend(values.iter().map(|&x| x as i8));
        }
    }
    let (heads, key_heads, positions, dim) =
        (shape.heads, shape.key_heads, shape.positions, shape.dim);
    let layers = 1usize.checked_shl(f.lambda).ok_or(StageError::Memory)?;
    let np = positions.next_power_of_two();
    let padded =
        |source: &SecretBuf<i8>, h: usize, hp: usize| -> Result<SecretBuf<i8>, StageError> {
            let len = layers
                .checked_mul(hp)
                .and_then(|v| v.checked_mul(np))
                .and_then(|v| v.checked_mul(dim))
                .ok_or(StageError::Memory)?;
            let mut out = SecretBuf::try_with_capacity(len).map_err(|_| StageError::Memory)?;
            out.resize(len, 0);
            for l in 0..f.nodes.len() {
                for head in 0..h {
                    for q in 0..positions {
                        let from = ((l * h + head) * positions + q) * dim;
                        let to = ((l * hp + head) * np + q) * dim;
                        out[to..to + dim].copy_from_slice(&source[from..from + dim]);
                    }
                }
            }
            Ok(out)
        };
    let queries = padded(&raw_queries, heads, heads.next_power_of_two())?;
    drop(raw_queries);
    let keys = padded(&raw_keys, key_heads, key_heads)?;
    drop(raw_keys);
    let mut maxima = zeros(maximum.len())?;
    maxima.copy_from_slice(maximum);
    Ok(score::Inputs {
        layers: f.nodes.len(),
        heads,
        key_heads,
        positions,
        dim,
        layer_vars: f.lambda as usize,
        queries,
        keys,
        maxima,
    })
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
        let mut equalities = HashMap::new();
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
            if let Some(score) = &mut data.score {
                score.point = target;
            } else {
                if let Some(equality) = equalities.get(&target) {
                    data.equality = Arc::clone(equality);
                } else {
                    let equality = Arc::new(eq_table(&target).map_err(|_| bad())?);
                    equalities.insert(target, Arc::clone(&equality));
                    data.equality = equality;
                }
            }
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
        if let Some(score) = &d.score {
            return Ok(Instance::computed(score));
        }
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
        // RFC-0055: the pre-P1 allocation plan charges every resident native column, including
        // the public equality, masks and runtime coefficients. Keep that public projection
        // pinned to the actual builder rather than silently undercharging a new column.
        let shape = self.plan.window_shape(tag)?;
        let width = 1usize
            .checked_shl(shape.vars as u32)
            .ok_or(StageError::Memory)?;
        if tables.len() != shape.tables || tables.iter().any(|t| t.len() != width) {
            return Err(bad());
        }
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
                let rows = match &self.plan.batch {
                    Some(batch) => {
                        let time_vars = vars(&[*f.shape.last().ok_or_else(bad)?]) as usize;
                        let lengths: Vec<_> = batch.lengths.iter().map(|&n| u64::from(n)).collect();
                        tessera_poly::slot_selector(
                            &point[..time_vars],
                            batch.slot_bits() as usize,
                            &lengths,
                        )
                        .map_err(|_| bad())?
                            * selector(&f.shape[..f.shape.len() - 1], &point[time_vars..local])?
                    }
                    None => selector(&f.shape, &point[..local])?,
                };
                let real =
                    rows * less_than(&point[local..], f.nodes.len() as u64).map_err(|_| bad())?;
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
    fn native_memory_projection_rejects_an_extra_public_column_and_a_short_private_column() {
        let plan = ModelNativePlan {
            batch: None,
            families: vec![Family {
                windows: Vec::new(),
                site: 0,
                nodes: vec![NodeId(0)],
                shape: vec![4],
                lambda: 0,
                kind: Kind::Root {
                    digits: 1,
                    coefficients: 1,
                    epsilon: 1,
                    residual: TensorId(0),
                },
            }],
            slots: vec![Slot {
                family: 0,
                relation: Relation::Product,
            }],
            native: vec![Native {
                tag: 1,
                kind: NativeKind::Product,
                node: NodeId(0),
                vars: 2,
                degree: 4,
                evaluations: 1,
                zero: false,
                outputs: Vec::new(),
            }],
            schedule: StageSchedule {
                layers: LayerMap::none(),
                sites: Vec::new(),
                stages: Vec::new(),
                pool: Vec::new(),
            },
        };
        let mut prover = ModelNativeProver {
            plan: &plan,
            tables: vec![SecretBuf::filled(G::ONE, 4)],
            literal_scores: true,
            data: vec![Data {
                score: None,
                raw: vec![0],
                mask: None,
                terms: vec![Term::new(F::ONE, vec![0, 1])],
                fixed: Vec::new(),
                point: Vec::new(),
                full_point: false,
                equality: Arc::new(vec![F::ONE; 4]),
                target: F::ZERO,
                public: Vec::new(),
            }],
        };
        assert!(prover.instance(1).is_ok());
        prover.data[0].public.push(vec![F::ZERO; 4]);
        assert!(matches!(
            prover.instance(1),
            Err(StageError::Internal {
                rule: "native model sources have their canonical identities and domains"
            })
        ));
        prover.data[0].public.pop();
        prover.tables[0].truncate(3);
        assert!(matches!(
            prover.instance(1),
            Err(StageError::Internal {
                rule: "native model sources have their canonical identities and domains"
            })
        ));
    }

    #[test]
    fn borrowed_integer_columns_use_bounded_storage_and_the_original_extension() {
        for variables in [0, 1, 2, 3, 8, 9, 18] {
            let values: Vec<i64> = (0..1i64 << variables).map(|x| -x * x + 3 * x - 7).collect();
            let point: Vec<_> = (0..variables)
                .map(|j| F::new(G::from_u64_reduced(3 + j), G::from_u64_reduced(5 + j)))
                .collect();
            let mut sum = F::ZERO;
            let mut diagonal = F::ZERO;
            for (j, &r) in point.iter().enumerate() {
                let a = fp(1 << j);
                sum += a * r;
                diagonal += a * a * r * (F::ONE - r);
            }
            // On Boolean x, x^2 = Σ a_j^2 x_j + 2Σ_{j<k} a_j a_k x_j x_k.
            // Multilinearizing replaces x_j^2 by x_j, giving this closed form.
            let expected = -(sum * sum + diagonal) + fp(3) * sum - fp(7);
            let info = allocation_counter::measure(|| {
                assert_eq!(evaluate_integers(&values, &point).unwrap(), expected);
            });
            assert!(info.bytes_max <= 32 * 1024, "{variables}: {info:?}");
        }
        assert!(evaluate_integers(&[1, 2, 3], &[F::ONE]).is_err());
    }
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

#[cfg(test)]
mod memory_error_tests {
    use super::*;
    #[test]
    fn native_table_core_resources_preserve_the_resource_refusal() {
        let error = |error| tessera_gadgets::GadgetError::Spec {
            table: tessera_ir::TableId(11),
            error,
        };
        assert_eq!(
            gadget_error(error(tessera_tables::TableError::Memory)),
            StageError::Memory
        );
        assert_eq!(
            gadget_error(error(tessera_tables::TableError::Digest)),
            bad()
        );
    }
}
