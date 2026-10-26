// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A reference prover of the stages, for tests only: a synthetic witness of a lowered graph and
//! honest proofs of its schedule's stages.
//!
//! - **The witness.** Every graph input and every committed output of a gadget takes random values
//!   in the test kit's field; every other node's outputs are computed by the node's semantics
//!   (RFC-0006 'Operation set'): `MatMulWeight`, `MatMulAct`, `Mul`, `SumSquares`, RoPE's
//!   rotation by its constants (RFC-0005), the linear maps, and `c_0 = 2^{s_e}·x_0`. The stages'
//!   relations are identities of the field, so random values satisfy them as a model's would.
//! - **The prover.** Each instance's tables are built from its relation's identity and the
//!   witness, contracted at its claim's point as RFC-0010 'Relations' writes them, never from the
//!   verifier's evaluation points; the proof is the test kit's batched sumcheck, on its own field
//!   and transcript. A claim the verifier routes wrongly therefore meets a sumcheck whose claim is
//!   the witness's true value, and fails.
//! - **The checks of the result.** Every pool claim the verifier returns is evaluated against
//!   the witness, term by term.
//!
//! The schedule tells the prover which instances to prove, in which order, and which values to
//! send for a split claim, each the witness's true evaluation.

#![allow(dead_code)]

use tv_core::lowering::{Broadcast, Graph, NodeId, Op, Role, TensorId};
use tv_core::schedule::{
    Claim, Coefficient, Layers, Part, Point, Relation, Route, Schedule, Source, TensorRef,
};
use tv_core::stages::{LeafValues, PoolClaim, PoolTensor, StageProof};
use tv_core::sumcheck::SumcheckProof;
use tv_field::{Fp, Fp2};
use tv_tables::rope::Rope;
use tv_testkit::SplitMix64;
use tv_testkit::sumcheck::{self as reference, Fe, Form, Product};

// ----- Field conversions -------------------------------------------------------------------------

pub fn to_fp2(value: Fe) -> Fp2 {
    Fp2::new(
        Fp::from_canonical(value.0).unwrap(),
        Fp::from_canonical(value.1).unwrap(),
    )
}

pub fn to_fe(value: Fp2) -> Fe {
    let bytes = value.to_le_bytes();
    Fe(
        u64::from_le_bytes(bytes[..8].try_into().unwrap()),
        u64::from_le_bytes(bytes[8..].try_into().unwrap()),
    )
}

fn fe_of_fp(value: Fp) -> Fe {
    Fe(value.to_canonical(), 0)
}

pub fn signed(v: i64) -> Fe {
    let magnitude = Fe::base(v.unsigned_abs());
    if v < 0 {
        Fe::ZERO - magnitude
    } else {
        magnitude
    }
}

fn random(rng: &mut SplitMix64) -> Fe {
    Fe::new(rng.next_u64(), rng.next_u64())
}

// ----- Shapes and padded tables ------------------------------------------------------------------

/// `⌈log2 d⌉`.
pub fn vars(d: u32) -> usize {
    (u32::BITS - (d - 1).leading_zeros()) as usize
}

pub fn shape_vars(dims: &[u32]) -> usize {
    dims.iter().map(|&d| vars(d)).sum()
}

/// The row-major multi-index of `flat`.
fn unravel(mut flat: usize, dims: &[u32]) -> Vec<usize> {
    let mut idx = vec![0; dims.len()];
    for a in (0..dims.len()).rev() {
        idx[a] = flat % dims[a] as usize;
        flat /= dims[a] as usize;
    }
    idx
}

fn ravel(idx: &[usize], dims: &[u32]) -> usize {
    idx.iter()
        .zip(dims)
        .fold(0, |flat, (&i, &d)| flat * d as usize + i)
}

/// The entry of a multi-index in the per-dimension padded table: the last axis lowest.
fn padded_index(idx: &[usize], dims: &[u32]) -> usize {
    let mut at = 0;
    let mut shift = 0;
    for a in (0..dims.len()).rev() {
        at |= idx[a] << shift;
        shift += vars(dims[a]);
    }
    at
}

/// A tensor's per-dimension padded table (RFC-0011 rule 1).
pub fn padded(values: &[Fe], dims: &[u32]) -> Vec<Fe> {
    let mut table = vec![Fe::ZERO; 1 << shape_vars(dims)];
    for (flat, &v) in values.iter().enumerate() {
        table[padded_index(&unravel(flat, dims), dims)] = v;
    }
    table
}

/// `eq(r, x)` for the index `x`.
pub fn eq_at(r: &[Fe], x: usize) -> Fe {
    r.iter().enumerate().fold(Fe::ONE, |p, (i, &c)| {
        p * if (x >> i) & 1 == 1 { c } else { Fe::ONE - c }
    })
}

/// `LT(r, a) = Σ_{h<a} eq(r, h)`, by its definition.
pub fn lt(r: &[Fe], a: u64) -> Fe {
    (0..a).fold(Fe::ZERO, |sum, h| sum + eq_at(r, h as usize))
}

/// The table with its variables `low..low + mid` bound to `point`: entry `l + h·2^low` is
/// `Σ_m eq(point, m)·table[l + m·2^low + h·2^{low + mid}]`.
fn bind(table: &[Fe], low: usize, point: &[Fe]) -> Vec<Fe> {
    let mid = point.len();
    let high = table.len().trailing_zeros() as usize - low - mid;
    let mut out = vec![Fe::ZERO; 1 << (low + high)];
    for (index, &v) in table.iter().enumerate() {
        let l = index & ((1 << low) - 1);
        let m = (index >> low) & ((1 << mid) - 1);
        let h = index >> (low + mid);
        out[l | (h << low)] = out[l | (h << low)] + eq_at(point, m) * v;
    }
    out
}

// ----- The witness ---------------------------------------------------------------------------------

/// Every tensor's values, row-major in its real shape, by id.
#[derive(Clone, Copy, Debug)]
pub enum IndicatorFault {
    Entry(Fe),
    NonBoolean,
    Missing,
    Extra,
    Index,
    ScalarToken,
    Padding,
}

pub struct Witness {
    pub values: Vec<Vec<Fe>>,
    indicator_padding: Option<(TensorId, Vec<Fe>)>,
}

impl Witness {
    /// Random graph inputs and gadget outputs, every other output computed.
    pub fn new(graph: &Graph, rope: &Rope, seed: u64) -> Self {
        Self::with_indicator_entry(graph, rope, seed, None)
    }

    /// Change one indicator entry before computing all downstream graph values.
    pub fn with_indicator_entry(graph: &Graph, rope: &Rope, seed: u64, entry: Option<Fe>) -> Self {
        Self::with_onehot_fault(graph, rope, seed, entry.map(IndicatorFault::Entry))
    }

    pub fn with_onehot_fault(
        graph: &Graph,
        rope: &Rope,
        seed: u64,
        fault: Option<IndicatorFault>,
    ) -> Self {
        let mut indicator_padding = None;
        let vocab = graph
            .nodes()
            .iter()
            .find_map(|node| match node.op() {
                Op::OneHot { vocab } => Some(u64::from(*vocab)),
                _ => None,
            })
            .unwrap();
        let mut rng = SplitMix64::new(seed);
        let mut values: Vec<Vec<Fe>> = vec![Vec::new(); graph.tensors().len()];
        let size = |t: TensorId| -> usize {
            graph
                .tensor(t)
                .unwrap()
                .shape
                .dims()
                .iter()
                .map(|&d| d as usize)
                .product()
        };
        for input in graph.inputs() {
            let t = input.tensor;
            values[t.0 as usize] = match input.role {
                Role::Positions => (0..size(t) as u64).map(Fe::base).collect(),
                Role::Tokens => (0..size(t))
                    .map(|_| Fe::base(rng.next_u64() % vocab))
                    .collect(),
                _ => (0..size(t)).map(|_| random(&mut rng)).collect(),
            };
        }
        if let Some(fault) = fault {
            let token = graph
                .inputs()
                .iter()
                .find(|i| i.role == Role::Tokens)
                .unwrap()
                .tensor;
            match fault {
                IndicatorFault::Entry(_) => {}
                IndicatorFault::Extra | IndicatorFault::ScalarToken => {
                    values[token.0 as usize][0] = Fe::ONE
                }
                IndicatorFault::Padding => values[token.0 as usize][0] = Fe::base(vocab),
                _ => values[token.0 as usize][0] = Fe::ZERO,
            }
        }
        for node in graph.nodes() {
            let dims = |t: TensorId| graph.tensor(t).unwrap().shape.dims().to_vec();
            let input = |i: usize| values[node.inputs()[i].0 as usize].clone();
            let outs: Vec<Vec<Fe>> = match *node.op() {
                Op::OneHot { vocab } => {
                    let mut h = vec![Fe::ZERO; size(node.outputs()[0])];
                    for (row, token) in input(0).iter().enumerate() {
                        if token.0 < u64::from(vocab) {
                            h[row * vocab as usize + token.0 as usize] = Fe::ONE;
                        }
                    }
                    if let Some(fault) = fault {
                        match fault {
                            IndicatorFault::Entry(value) => h[0] = value,
                            IndicatorFault::NonBoolean => {
                                h[..vocab as usize].fill(Fe::ZERO);
                                h[0] = Fe::base(2);
                                h[1] = Fe::ZERO - Fe::base(2);
                                h[2] = Fe::ONE;
                            }
                            IndicatorFault::Missing => h[..vocab as usize].fill(Fe::ZERO),
                            IndicatorFault::Extra => {
                                h[0] = Fe::ONE;
                                h[1] = Fe::ONE;
                            }
                            IndicatorFault::Index => {
                                h[0] = Fe::ZERO;
                                h[1] = Fe::ONE;
                            }
                            IndicatorFault::ScalarToken => {
                                h[..vocab as usize].fill(Fe::ZERO);
                                h[0] = Fe::ONE;
                            }
                            IndicatorFault::Padding => {
                                assert!(!vocab.is_power_of_two());
                                let mut table = padded(&h, &dims(node.outputs()[0]));
                                table[vocab as usize] = Fe::ONE;
                                indicator_padding = Some((node.outputs()[0], table));
                            }
                        }
                    }
                    vec![h]
                }
                Op::Requant { .. }
                | Op::RsqrtRow { .. }
                | Op::SoftmaxRow { .. }
                | Op::TableMap { .. } => node
                    .outputs()
                    .iter()
                    .map(|&o| (0..size(o)).map(|_| random(&mut rng)).collect())
                    .collect(),
                Op::MatMulWeight { .. } => {
                    let (x, w) = (input(0), input(1));
                    let wd = dims(node.inputs()[1]);
                    let (out, inner) = (wd[0] as usize, wd[1] as usize);
                    let rows = x.len() / inner;
                    let mut y = vec![Fe::ZERO; rows * out];
                    for t in 0..rows {
                        for o in 0..out {
                            let mut sum = Fe::ZERO;
                            for i in 0..inner {
                                sum = sum + w[o * inner + i] * x[t * inner + i];
                            }
                            y[t * out + o] = sum;
                        }
                    }
                    vec![y]
                }
                Op::ConstMul { c } => {
                    let c = signed(graph.constants()[c as usize]);
                    vec![input(0).iter().map(|&v| c * v).collect()]
                }
                Op::Add => {
                    let (a, b) = (input(0), input(1));
                    vec![a.iter().zip(&b).map(|(&x, &y)| x + y).collect()]
                }
                Op::Reshape => vec![input(0)],
                Op::SumSquares { .. } => {
                    let x = input(0);
                    let d = *dims(node.inputs()[0]).last().unwrap() as usize;
                    vec![
                        x.chunks(d)
                            .map(|row| row.iter().fold(Fe::ZERO, |s, &v| s + v * v))
                            .collect(),
                    ]
                }
                Op::Mul { broadcast } => {
                    let (a, b) = (input(0), input(1));
                    let y = a
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| {
                            let j = match broadcast {
                                Broadcast::Trailing => i / (a.len() / b.len()),
                                Broadcast::Leading => i % b.len(),
                            };
                            v * b[j]
                        })
                        .collect();
                    vec![y]
                }
                Op::Rotate { .. } => {
                    let x = input(0);
                    let xd = dims(node.inputs()[0]);
                    let width = *xd.last().unwrap() as usize;
                    let half = width / 2;
                    let per_position = x.len() / xd[0] as usize;
                    let mut y = vec![Fe::ZERO; x.len()];
                    for (flat, out) in y.iter_mut().enumerate() {
                        let position = (flat / per_position) as u32;
                        let j = flat % width;
                        let base = flat - j;
                        let i = j % half;
                        let (c, s) = rope.pair(position, i).unwrap();
                        let (c, s) = (signed(c), signed(s));
                        let (a, b) = (x[base + i], x[base + i + half]);
                        *out = if j < half {
                            a * c - b * s
                        } else {
                            a * s + b * c
                        };
                    }
                    vec![y]
                }
                Op::Transpose { perm } => {
                    let x = input(0);
                    let xd = dims(node.inputs()[0]);
                    let yd = dims(node.outputs()[0]);
                    let mut y = vec![Fe::ZERO; x.len()];
                    for (flat, out) in y.iter_mut().enumerate() {
                        let iy = unravel(flat, &yd);
                        let mut ix = vec![0; xd.len()];
                        for (i, &p) in perm.iter().enumerate() {
                            ix[p as usize] = iy[i];
                        }
                        *out = x[ravel(&ix, &xd)];
                    }
                    vec![y]
                }
                Op::Expand { axis, .. } => {
                    let x = input(0);
                    let xd = dims(node.inputs()[0]);
                    let yd = dims(node.outputs()[0]);
                    let mut y = vec![Fe::ZERO; size(node.outputs()[0])];
                    for (flat, out) in y.iter_mut().enumerate() {
                        let mut iy = unravel(flat, &yd);
                        iy.remove(axis as usize);
                        *out = x[ravel(&iy, &xd)];
                    }
                    vec![y]
                }
                Op::MatMulAct { transpose_rhs } => {
                    let (a, b) = (input(0), input(1));
                    let ad = dims(node.inputs()[0]);
                    let yd = dims(node.outputs()[0]);
                    let rank = ad.len();
                    let (m, k) = (ad[rank - 2] as usize, ad[rank - 1] as usize);
                    let n = yd[rank - 1] as usize;
                    let lead = a.len() / (m * k);
                    let mut y = vec![Fe::ZERO; lead * m * n];
                    for l in 0..lead {
                        for i in 0..m {
                            for c in 0..n {
                                let mut sum = Fe::ZERO;
                                for j in 0..k {
                                    let bv = if transpose_rhs {
                                        b[(l * n + c) * k + j]
                                    } else {
                                        b[(l * k + j) * n + c]
                                    };
                                    sum = sum + a[(l * m + i) * k + j] * bv;
                                }
                                y[(l * m + i) * n + c] = sum;
                            }
                        }
                    }
                    vec![y]
                }
                // The anchor circuits' operations: no lowered graph holds one.
                op @ (Op::Sub
                | Op::ConstAdd { .. }
                | Op::Sum { .. }
                | Op::Slice { .. }
                | Op::GatherPublic { .. }
                | Op::Bitwise { .. }
                | Op::BitDecompose { .. }
                | Op::BitFn { .. }
                | Op::AssertEqual) => {
                    panic!("the reference evaluates the lowering's operations, not {op:?}")
                }
            };
            for (&o, v) in node.outputs().iter().zip(outs) {
                assert_eq!(v.len(), size(o), "{:?}", node.op());
                values[o.0 as usize] = v;
            }
        }
        Self {
            values,
            indicator_padding,
        }
    }

    pub fn table(&self, tensor: TensorId, dims: &[u32]) -> Vec<Fe> {
        match &self.indicator_padding {
            Some((id, table)) if *id == tensor => table.clone(),
            _ => padded(&self.values[tensor.0 as usize], dims),
        }
    }
}

// ----- The prover --------------------------------------------------------------------------------

/// The graph's layers: where layer ℓ's node and tensors are.
pub struct Layout<'g> {
    pub graph: &'g Graph,
    pub layers: usize,
    pub lambda: usize,
    starts: Vec<u32>,
}

impl<'g> Layout<'g> {
    pub fn new(graph: &'g Graph) -> Self {
        let producer = |c: TensorId| graph.tensor(c).unwrap().producer.unwrap().0;
        let checkpoints = graph.checkpoints();
        let layers = checkpoints.len() - 1;
        let starts = checkpoints[..layers]
            .iter()
            .map(|&c| producer(c) + 1)
            .collect();
        Self {
            graph,
            layers,
            lambda: if layers <= 1 { 0 } else { vars(layers as u32) },
            starts,
        }
    }

    /// Layer `layer`'s copy of layer 0's node `node`.
    pub fn node_at(&self, node: NodeId, layer: usize) -> NodeId {
        NodeId(node.0 - self.starts[0] + self.starts[layer])
    }

    /// Layer `layer`'s tensor of the stacked kind `rep`, layer 0's.
    pub fn tensor_at(&self, rep: TensorId, layer: usize) -> TensorId {
        let graph = self.graph;
        if let Some(p) = graph.tensor(rep).unwrap().producer {
            let node = graph.node(p).unwrap();
            let o = node.outputs().iter().position(|&t| t == rep).unwrap();
            return graph.node(self.node_at(p, layer)).unwrap().outputs()[o];
        }
        let role = graph
            .inputs()
            .iter()
            .find(|i| i.tensor == rep)
            .unwrap()
            .role;
        let same = |r: Role| match (role, r) {
            (Role::Weight(a), Role::Weight(b))
            | (Role::Classes(a), Role::Classes(b))
            | (Role::Multipliers(a), Role::Multipliers(b)) => {
                a.ty == b.ty && b.layer as usize == layer
            }
            _ => false,
        };
        graph.inputs().iter().find(|i| same(i.role)).unwrap().tensor
    }

    /// The per-layer tensors of a schedule tensor: one for a tensor outside the layers.
    pub fn tensors(&self, tensor: TensorRef) -> Vec<TensorId> {
        match tensor {
            TensorRef::Single(t) => vec![t],
            TensorRef::LayerInput => self.graph.checkpoints()[..self.layers].to_vec(),
            TensorRef::Stacked(rep) => (0..self.layers).map(|l| self.tensor_at(rep, l)).collect(),
        }
    }

    pub fn dims(&self, t: TensorId) -> Vec<u32> {
        self.graph.tensor(t).unwrap().shape.dims().to_vec()
    }
}

/// The coordinates the transcript gave so far, as the schedule's sources name them.
#[derive(Default)]
pub struct Sources {
    pub leaf: Vec<Vec<Fe>>,
    pub layer: Vec<Fe>,
    pub rounds: Vec<Vec<Fe>>,
    pub zero: Vec<Vec<Fe>>,
}

impl Sources {
    pub fn resolve(&self, point: &Point) -> Vec<Fe> {
        point
            .runs()
            .iter()
            .flat_map(|run| {
                let source: &[Fe] = match run.source {
                    Source::Leaf(c) => &self.leaf[c as usize],
                    Source::Layer => &self.layer,
                    Source::Rounds(k) => &self.rounds[k as usize - 1],
                    Source::Zero(k) => &self.zero[k as usize - 1],
                };
                source[run.start..run.start + run.len].to_vec()
            })
            .collect()
    }
}

/// An honest proof of a schedule over a witness, with what the verifier needs to check it.
pub struct Proven {
    pub leaf: Vec<LeafValues>,
    pub leaf_splits: Vec<Fp2>,
    pub stages: Vec<StageProof>,
}

/// Independent extra relation oracle used only by mixed-stage tests.
pub trait NativeOracle {
    fn product(&self, prover: &Prover<'_>, stage: usize, instance: usize, zero: &[Fe]) -> Product;
}

pub struct Prover<'a> {
    pub layout: Layout<'a>,
    pub witness: &'a Witness,
    pub schedule: &'a Schedule,
    pub rope: &'a Rope,
    pub sources: Sources,
    /// An adapted malicious proof replaces just this OneHot relation by zero.
    pub omit_onehot: Option<usize>,
    pub native: Option<&'a dyn NativeOracle>,
}

impl Prover<'_> {
    /// A tensor's per-layer value at `point`: the true extension of its padded table.
    pub fn value_at(&self, t: TensorId, point: &[Fe]) -> Fe {
        let dims = self.layout.dims(t);
        reference::mle(&self.witness.table(t, &dims), point)
    }

    /// The table of a schedule tensor over its variables, then the layers' when stacked; padded
    /// layers zero.
    pub fn stacked_table(&self, tensor: TensorRef) -> Vec<Fe> {
        let tensors = self.layout.tensors(tensor);
        let dims = self.layout.dims(tensors[0]);
        let own = 1usize << shape_vars(&dims);
        let mut table = Vec::new();
        for t in &tensors {
            table.extend(self.witness.table(*t, &dims));
        }
        if matches!(tensor, TensorRef::Single(_)) {
            return table;
        }
        table.resize(own << self.layout.lambda, Fe::ZERO);
        table
    }

    /// An atom's tensor at its point with its layer weights, without its coefficient: at layer
    /// `layer` for a leaf claim's per-layer value.
    fn atom_tensor(&self, claim: &Claim, a: usize, layer: Option<usize>) -> Fe {
        let atom = &claim.atoms[a];
        let point = self.sources.resolve(&atom.point);
        match (&atom.layers, layer) {
            (_, Some(l)) => {
                let tensors = self.layout.tensors(atom.tensor);
                let t = if tensors.len() == 1 {
                    tensors[0]
                } else {
                    tensors[l]
                };
                self.value_at(t, &point)
            }
            (Layers::None, None) => self.value_at(self.layout.tensors(atom.tensor)[0], &point),
            (Layers::One(l), None) => {
                self.value_at(self.layout.tensors(atom.tensor)[*l as usize], &point)
            }
            (Layers::Point(rho), None) => {
                let rho = self.sources.resolve(rho);
                reference::mle(&self.stacked_table(atom.tensor), &[point, rho].concat())
            }
        }
    }

    fn coefficient(&self, coefficient: &Coefficient, layer: usize) -> Fe {
        coefficient
            .terms()
            .iter()
            .map(|term| {
                let c = if term.constant.len() == 1 {
                    term.constant[0]
                } else {
                    term.constant[layer]
                };
                term.lts.iter().fold(fe_of_fp(c), |p, (point, bound)| {
                    p * lt(&self.sources.resolve(point), *bound)
                })
            })
            .fold(Fe::ZERO, |s, v| s + v)
    }

    /// The values a split claim sends, in part order.
    fn split_values(&self, claim: &Claim, layer: Option<usize>) -> Vec<Fe> {
        let Route::Split(parts) = &claim.route else {
            return Vec::new();
        };
        parts
            .iter()
            .map(|part| match part {
                // A virtual atom's value is its tensor's evaluation, without its coefficient; a
                // group's is its combination, coefficients included.
                Part::Virtual { atom, .. } => self.atom_tensor(claim, *atom, layer),
                Part::Group { atoms } => atoms.iter().fold(Fe::ZERO, |s, &a| {
                    let c = self.coefficient(&claim.atoms[a].coefficient, layer.unwrap_or(0));
                    s + c * self.atom_tensor(claim, a, layer)
                }),
            })
            .collect()
    }

    /// The leaf claims at random points, their true values, and the leaf stage's split values,
    /// site by site, layer by layer, claim by claim.
    pub fn leaf(&mut self, rng: &mut SplitMix64) -> (Vec<LeafValues>, Vec<Fe>) {
        let claims = self.schedule.leaf();
        let mut leaf = Vec::new();
        for claim in claims {
            let point: Vec<Fe> = (0..claim.claim.vars).map(|_| random(rng)).collect();
            self.sources.leaf.push(point.clone());
            let node = self.layout.graph.node(claim.site).unwrap();
            let values = match claim.input {
                Some(i) if claim.stacked => (0..self.layout.layers)
                    .map(|l| {
                        let site = self
                            .layout
                            .graph
                            .node(self.layout.node_at(claim.site, l))
                            .unwrap();
                        self.value_at(site.inputs()[i], &point)
                    })
                    .collect(),
                Some(i) => vec![self.value_at(node.inputs()[i], &point)],
                None => vec![self.value_at(node.outputs()[0], &point)],
            };
            leaf.push(LeafValues {
                point: point.iter().map(|&v| to_fp2(v)).collect(),
                values: values.iter().map(|&v| to_fp2(v)).collect(),
            });
        }
        let mut splits = Vec::new();
        let mut start = 0;
        while start < claims.len() {
            let end = (start..claims.len())
                .find(|&c| claims[c].site != claims[start].site)
                .unwrap_or(claims.len());
            let layers = if claims[start].stacked {
                self.layout.layers
            } else {
                1
            };
            for l in 0..layers {
                for claim in &claims[start..end] {
                    let layer = claim.stacked.then_some(l);
                    splits.extend(self.split_values(&claim.claim, layer));
                }
            }
            start = end;
        }
        (leaf, splits)
    }

    /// Instance `instance`'s tables, from its relation's identity and the witness.
    fn product(&self, k: usize, i: usize, zero: &[Fe]) -> Product {
        let mut product = self.exact_product(k, i, zero);
        let instance = &self.schedule.stages()[k].instances[i];
        let obligation = match instance.relation {
            Relation::Booleanity => Some(0),
            Relation::Vocabulary { .. } => Some(1),
            Relation::OneHotCount { .. } => Some(2),
            Relation::OneHotIndex { .. } => Some(3),
            _ => None,
        };
        if obligation.is_some() && self.omit_onehot == obligation {
            product.form = Form::Product;
            product.public = vec![vec![Fe::ZERO; 1usize << instance.vars]];
            product.lambda = None;
        }
        product
    }

    fn exact_product(&self, k: usize, i: usize, zero: &[Fe]) -> Product {
        let instance = &self.schedule.stages()[k].instances[i];
        let graph = self.layout.graph;
        let node = graph.node(instance.node).unwrap();
        let r = self.sources.resolve(&instance.point);
        let rho = match &instance.layers {
            Layers::Point(p) => Some(self.sources.resolve(p)),
            _ => None,
        };
        // The per-layer nodes whose relation the instance stacks.
        let nodes: Vec<NodeId> = if instance.stacked {
            (0..self.layout.layers)
                .map(|l| self.layout.node_at(instance.node, l))
                .collect()
        } else {
            vec![instance.node]
        };
        let layers = if instance.stacked {
            1 << self.layout.lambda
        } else {
            1
        };
        // Concatenates a per-layer table over the layers, padded layers zero.
        let stack = |per_layer: &dyn Fn(&tv_core::lowering::Node) -> Vec<Fe>| -> Vec<Fe> {
            let mut table = Vec::new();
            let mut size = 0;
            for &n in &nodes {
                let t = per_layer(graph.node(n).unwrap());
                size = t.len();
                table.extend(t);
            }
            table.resize(size * layers, Fe::ZERO);
            table
        };
        let table_of = |t: TensorId| self.witness.table(t, &self.layout.dims(t));
        // eq(ρ, ℓ) times a per-layer public factor.
        let public = |own: Vec<Fe>| -> Vec<Fe> {
            match &rho {
                None => own,
                Some(rho) => (0..layers)
                    .flat_map(|l| {
                        let e = eq_at(rho, l);
                        own.iter().map(move |&v| v * e).collect::<Vec<_>>()
                    })
                    .collect(),
            }
        };
        let (tables, public_table, form) = match instance.relation {
            Relation::Native { .. } => {
                return self
                    .native
                    .expect("mixed reference oracle")
                    .product(self, k, i, zero);
            }
            Relation::MatMulWeight { in_vars, out_vars } => {
                let (r_o, r_t) = r.split_at(out_vars);
                let w = stack(&|n| bind(&table_of(n.inputs()[1]), in_vars, r_o));
                let x = stack(&|n| bind(&table_of(n.inputs()[0]), in_vars, r_t));
                // No stacking variable, no factor: the relation is then of degree 2.
                let p = rho
                    .as_ref()
                    .filter(|rho| !rho.is_empty())
                    .map(|_| public(vec![Fe::ONE; 1 << in_vars]));
                (vec![w, x], p, Form::Product)
            }
            Relation::MatMulAct {
                transpose_rhs,
                k_vars,
                m_vars,
                n_vars,
                lead_vars,
            } => {
                let r_c = &r[..n_vars];
                let r_a = &r[n_vars..n_vars + m_vars];
                let r_h = &r[n_vars + m_vars..];
                let a = stack(&|n| bind(&table_of(n.inputs()[0]), k_vars, r_a));
                let b = stack(&|n| {
                    let table = table_of(n.inputs()[1]);
                    if transpose_rhs {
                        bind(&table, k_vars, r_c)
                    } else {
                        bind(&table, 0, r_c)
                    }
                });
                let heads: Vec<Fe> = (0..1usize << (k_vars + lead_vars))
                    .map(|x| eq_at(r_h, x >> k_vars))
                    .collect();
                (vec![a, b], Some(public(heads)), Form::Product)
            }
            Relation::Elementwise {
                broadcast, b_vars, ..
            } => {
                let own = shape_vars(&self.layout.dims(node.outputs()[0]));
                let a = stack(&|n| table_of(n.inputs()[0]));
                let b = stack(&|n| {
                    let b = table_of(n.inputs()[1]);
                    (0..1usize << own)
                        .map(|x| match broadcast {
                            Broadcast::Trailing => b[x >> (own - b_vars)],
                            Broadcast::Leading => b[x & ((1 << b_vars) - 1)],
                        })
                        .collect()
                });
                let eqs: Vec<Fe> = (0..1usize << own).map(|x| eq_at(&r, x)).collect();
                (vec![a, b], Some(public(eqs)), Form::Product)
            }
            Relation::SumSquares { col_vars } => {
                let own = shape_vars(&self.layout.dims(node.inputs()[0]));
                let x = stack(&|n| table_of(n.inputs()[0]));
                let eqs: Vec<Fe> = (0..1usize << own)
                    .map(|x| eq_at(&r, x >> col_vars))
                    .collect();
                (vec![x], Some(public(eqs)), Form::Square)
            }
            Relation::Rotate { positions, .. } => {
                let xd = self.layout.dims(node.inputs()[0]);
                let own = shape_vars(&xd);
                let x = stack(&|n| table_of(n.inputs()[0]));
                let dh_vars = vars(*xd.last().unwrap());
                let b = dh_vars - 1;
                let position_shift = own - vars(xd[0]);
                let kernel: Vec<Fe> = (0..1usize << own)
                    .map(|x| {
                        let position = x >> position_shift;
                        let frequency = x & ((1 << b) - 1);
                        let bit = (x >> b) & 1;
                        let (c, s) = if (position as u32) < positions {
                            let (c, s) = self.rope.pair(position as u32, frequency).unwrap();
                            (signed(c), signed(s))
                        } else {
                            (Fe::ZERO, Fe::ZERO)
                        };
                        let mut rest = Fe::ONE;
                        for (i, &ri) in r.iter().enumerate() {
                            if i != b {
                                rest = rest * if (x >> i) & 1 == 1 { ri } else { Fe::ONE - ri };
                            }
                        }
                        let rb = r[b];
                        let xb = Fe::base(bit as u64);
                        let eq_b = rb * xb + (Fe::ONE - rb) * (Fe::ONE - xb);
                        rest * (c * eq_b + s * (rb - xb))
                    })
                    .collect();
                (vec![x], Some(public(kernel)), Form::Product)
            }
            Relation::Vocabulary { vocab, vocab_vars } => {
                let h = table_of(node.outputs()[0]);
                let mask = (1usize << vocab_vars) - 1;
                let eqs = (0..h.len()).map(|v| eq_at(zero, v)).collect();
                let outside = (0..h.len())
                    .map(|v| {
                        if v & mask < vocab as usize {
                            Fe::ZERO
                        } else {
                            Fe::ONE
                        }
                    })
                    .collect();
                return Product {
                    tables: vec![h],
                    public: vec![eqs, outside],
                    lambda: None,
                    form: Form::Product,
                };
            }
            Relation::OneHotCount {
                vocab_vars,
                positions,
            } => {
                let h = table_of(node.outputs()[0]);
                let width = 1usize << vocab_vars;
                let eqs = (0..h.len())
                    .map(|v| eq_at(&zero[vocab_vars..], v / width))
                    .collect();
                let selector = (0..h.len())
                    .map(|v| {
                        if v / width < positions as usize {
                            Fe::ONE
                        } else {
                            Fe::ZERO
                        }
                    })
                    .collect();
                return Product {
                    tables: vec![h],
                    public: vec![eqs, selector],
                    lambda: None,
                    form: Form::OneHotCount(Fe::base(width as u64).inverse().unwrap()),
                };
            }
            Relation::OneHotIndex { vocab_vars } => {
                let h = table_of(node.outputs()[0]);
                let token = table_of(node.inputs()[0]);
                let width = 1usize << vocab_vars;
                let eqs = (0..h.len())
                    .map(|v| eq_at(&zero[vocab_vars..], v / width))
                    .collect();
                let indices = (0..h.len()).map(|v| Fe::base((v % width) as u64)).collect();
                let tokens = (0..h.len()).map(|v| token[v / width]).collect();
                return Product {
                    tables: vec![h, tokens],
                    public: vec![eqs, indices],
                    lambda: None,
                    form: Form::OneHotIndex(Fe::base(width as u64).inverse().unwrap()),
                };
            }
            Relation::Booleanity => {
                let h = table_of(node.outputs()[0]);
                let eqs = (0..h.len()).map(|v| eq_at(zero, v)).collect();
                (vec![h], Some(eqs), Form::Booleanity)
            }
            Relation::Equality { .. } => {
                let Op::ConstMul { c } = *node.op() else {
                    panic!("an equality zero-check of a ConstMul")
                };
                let constant = signed(graph.constants()[c as usize]);
                let y = table_of(node.outputs()[0]);
                let x = table_of(node.inputs()[0]);
                let eqs: Vec<Fe> = (0..y.len()).map(|v| eq_at(zero, v)).collect();
                (vec![y, x], Some(eqs), Form::Difference(constant))
            }
        };
        // Each public table is a product of factors over disjoint variables, itself multilinear.
        Product {
            tables,
            public: public_table.into_iter().collect(),
            lambda: None,
            form,
        }
    }

    /// Proves every stage after the leaf stage's splits, on `transcript`.
    pub fn stages(
        &mut self,
        transcript: &mut reference::Transcript,
        lie: Option<(usize, reference::Lie)>,
    ) -> Vec<StageProof> {
        let mut proofs = Vec::new();
        for (k, stage) in self.schedule.stages().iter().enumerate() {
            let prefix = format!("stage/{}", stage.index);
            if stage.index == 1 {
                self.sources.layer = (0..stage.layer_vars)
                    .map(|c| transcript.challenge_fp2(&format!("{prefix}/layer/{c}")))
                    .collect();
            }
            let zero: Vec<Fe> = (0..stage.zero_vars)
                .map(|c| transcript.challenge_fp2(&format!("{prefix}/zero/{c}")))
                .collect();
            self.sources.zero.push(zero.clone());
            let products: Vec<Product> = (0..stage.instances.len())
                .map(|i| {
                    let instance = &stage.instances[i];
                    let own_zero = &zero[zero.len().saturating_sub(instance.vars)..];
                    let own_zero = if instance.zero.is_empty() {
                        &[][..]
                    } else {
                        own_zero
                    };
                    self.product(k, i, own_zero)
                })
                .collect();
            let lie = lie.filter(|(at, _)| *at == k).map(|(_, lie)| lie);
            let proof = reference::prove(transcript, &prefix, stage.degree, &products, lie);
            self.sources.rounds.push(proof.challenges.clone());
            let mut splits = Vec::new();
            for instance in &stage.instances {
                for claim in &instance.evals {
                    splits.extend(self.split_values(claim, None));
                }
            }
            transcript.absorb(&format!("{prefix}/split"), &reference::tce(&splits));
            proofs.push(StageProof {
                stage: stage.index,
                sumcheck: SumcheckProof {
                    rounds: proof
                        .rounds
                        .iter()
                        .map(|r| r.iter().map(|&v| to_fp2(v)).collect())
                        .collect(),
                    final_evals: proof.evals.iter().map(|&v| to_fp2(v)).collect(),
                },
                splits: splits.iter().map(|&v| to_fp2(v)).collect(),
            });
        }
        proofs
    }

    /// A pool claim's combination evaluated from the witness: `Σ_t Σ_ℓ w_t(ℓ)·P̃_{t,ℓ}(point)`.
    pub fn pool_value(&self, claim: &PoolClaim) -> Fe {
        let point: Vec<Fe> = claim.point.iter().map(|&v| to_fe(v)).collect();
        claim
            .terms
            .iter()
            .map(|term| match term.tensor {
                PoolTensor::Single(t) => to_fe(term.weights[0]) * self.value_at(t, &point),
                PoolTensor::Stacked(rep) => term
                    .weights
                    .iter()
                    .enumerate()
                    .map(|(l, &w)| to_fe(w) * self.value_at(self.layout.tensor_at(rep, l), &point))
                    .fold(Fe::ZERO, |s, v| s + v),
            })
            .fold(Fe::ZERO, |s, v| s + v)
    }
}
