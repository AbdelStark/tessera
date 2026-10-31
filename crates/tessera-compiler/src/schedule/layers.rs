// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The layers of a lowered graph and their template (RFC-0037 'Layers and kinds').

use tessera_field::{Goldilocks, PrimeField};
use tessera_ir::{CommitmentClass, Graph, NodeId, NodeKind, Op, TensorId, TensorSpec};
use tessera_tables::TableSpec;

use super::padding::nonzero_at_padding;
use super::public::{producers, public_tensors};
use super::{Kind, Layers, axis_vars, refuse};
use crate::{CompileError, Lowering};

/// The layers of a lowered graph: the checkpoints `c_0, …, c_L` bound them, and every layer is one
/// template, node for node (RFC-0037 'Layers and kinds').
///
/// Layer `ℓ` holds the nodes after `c_ℓ`'s producer, up to and including `c_{ℓ+1}`'s. The nodes up
/// to `c_0`'s producer are the prologue, and those after `c_L`'s the epilogue. [`LayerMap::of`]
/// checks the template:
///
/// - every layer has as many nodes, the last producing the layer's checkpoint;
/// - the nodes at one position have one operation, up to their layer's own weights, constants,
///   shift classes and tables, and the same inputs and outputs, of one shape and kind;
/// - no position holds a `ConstAdd`, a selection (`Slice`, `GatherPublic`), a `ConstMul` by a
///   constant per channel, a `BitDecompose`, an `AssertEqual`, a lookup non-zero at padding
///   ([`nonzero_at_padding`](super::nonzero_at_padding)), or a node whose output is public
///   ([`public_tensors`](super::public_tensors)), whose stacked forms or padded layers RFC-0072
///   leaves unspecified ('Graphs without layers');
/// - each input is an earlier output of its own layer, its layer's checkpoint, or a graph input:
///   one every layer shares, or one of its own in each layer;
/// - each output of a layer is read in its layer only, but for its checkpoint, which the next
///   layer, or the epilogue, reads.
///
/// A graph with no checkpoint has no layers, `L = 0` and `λ = 0` ([`LayerMap::none`]): every node
/// is outside them, as the anchor's circuits are (RFC-0072 'Graphs without layers').
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LayerMap {
    /// The layers `L`.
    pub count: u32,
    /// The layer variables `λ = ⌈log2 L⌉`, the highest of every stacked kind.
    pub vars: u32,
    /// The first node of layer 0: the prologue is the nodes before it.
    pub first: u32,
    /// The nodes of a layer, `T`.
    pub width: u32,
    /// The kind of each input of the node at each position.
    inputs: Vec<Vec<Kind>>,
    /// `c_0, …, c_L`.
    checkpoints: Vec<TensorId>,
    /// The output of the last position that is the layer's checkpoint.
    last_output: u32,
    /// Whether each position's `ConstMul`, if it is one, multiplies by the same `Fp` constant in
    /// every layer.
    uniform: Vec<bool>,
}

impl LayerMap {
    /// Requested heap bytes owned by the layer map, including nested spare
    /// capacity (RFC-0037/0018). Excludes the inline object, construction scratch
    /// and allocator overhead. Does not allocate or validate; returns `None`
    /// if a checked byte count or sum overflows.
    pub fn heap_bytes(&self) -> Option<u64> {
        let Self {
            count: _,
            vars: _,
            first: _,
            width: _,
            inputs,
            checkpoints,
            last_output: _,
            uniform,
        } = self;
        let array = |n: usize, size: usize| (n as u64).checked_mul(size as u64);
        let mut bytes = array(inputs.capacity(), size_of::<Vec<Kind>>())?
            .checked_add(array(checkpoints.capacity(), size_of::<TensorId>())?)?
            .checked_add(array(uniform.capacity(), size_of::<bool>())?)?;
        for values in inputs {
            bytes = bytes.checked_add(array(values.capacity(), size_of::<Kind>())?)?;
        }
        Some(bytes)
    }

    /// The layers of `lowering`'s graph, from its checkpoints: [`LayerMap::of`].
    ///
    /// # Errors
    ///
    /// [`CompileError::Schedule`] for the first rule of the template broken, as [`LayerMap`] lists
    /// them.
    pub fn new(lowering: &Lowering) -> Result<Self, CompileError> {
        Self::of(&lowering.graph, &lowering.tables, &lowering.checkpoints)
    }

    /// The layers of a graph with no checkpoint: none, `L = 0` and `λ = 0`, every node outside them
    /// and every kind a tensor's (RFC-0037 'Layers and kinds' 'No checkpoint').
    pub fn none() -> Self {
        Self {
            count: 0,
            vars: 0,
            first: 0,
            width: 0,
            inputs: Vec::new(),
            checkpoints: Vec::new(),
            last_output: 0,
            uniform: Vec::new(),
        }
    }

    /// The layers of `graph`, whose tables are `tables`, that the checkpoints `c_0, …, c_L` bound;
    /// with no checkpoint, [`LayerMap::none`]. One checkpoint bounds no layer and is refused.
    ///
    /// # Errors
    ///
    /// [`CompileError::Schedule`] for the first rule of the template broken, as [`LayerMap`] lists
    /// them, or for a lookup at a template position whose table is not among `tables`.
    pub fn of(
        graph: &Graph,
        tables: &[TableSpec],
        checkpoints: &[TensorId],
    ) -> Result<Self, CompileError> {
        if checkpoints.is_empty() {
            return Ok(Self::none());
        }
        let producers = producers(graph);
        let producer = |t: TensorId| producers.get(t.0 as usize).copied().flatten();
        let count = u32::try_from(checkpoints.len() - 1).unwrap_or(u32::MAX);
        if count == 0 {
            return refuse(None, "the lowering has a layer");
        }
        let mut ends = Vec::with_capacity(checkpoints.len());
        for &c in checkpoints {
            match producer(c) {
                Some((node, _)) => ends.push(node.0),
                None => return refuse(None, "every checkpoint is a node's output"),
            }
            let advice = NodeKind::Committed {
                commitment: CommitmentClass::Advice,
            };
            if graph.tensors.get(c.0 as usize).map(|t| t.kind) != Some(advice) {
                return refuse(None, "every checkpoint is committed advice");
            }
        }
        let first = ends[0] + 1;
        let width = ends[1].saturating_sub(ends[0]);
        if width == 0
            || ends
                .windows(2)
                .any(|pair| pair[1].checked_sub(pair[0]) != Some(width))
        {
            return refuse(None, "every layer holds as many nodes");
        }
        let last_output = producer(checkpoints[1]).map_or(0, |(_, output)| output);
        if checkpoints[1..]
            .iter()
            .any(|&c| producer(c).map(|(_, output)| output) != Some(last_output))
        {
            return refuse(
                None,
                "every layer's checkpoint is one output of its last node",
            );
        }
        let vars = axis_vars(count);
        let mut map = Self {
            count,
            vars,
            first,
            width,
            inputs: Vec::with_capacity(width as usize),
            checkpoints: checkpoints.to_vec(),
            last_output,
            uniform: Vec::with_capacity(width as usize),
        };
        let public = public_tensors(graph);
        for position in 0..width {
            map.check_position(graph, tables, &producers, &public, position)?;
        }
        map.check_readers(graph, &producers)?;
        Ok(map)
    }

    /// The node at `position` in layer `layer`.
    pub fn node(&self, position: u32, layer: u32) -> NodeId {
        NodeId(self.first + layer * self.width + position)
    }

    /// The first node of the epilogue.
    pub fn end(&self) -> u32 {
        self.first + self.count * self.width
    }

    /// The kind of input `input` of the node at `position`.
    pub fn input(&self, position: u32, input: u32) -> Kind {
        self.inputs[position as usize][input as usize]
    }

    /// Whether the `ConstMul` at `position` multiplies by one constant in every layer.
    pub fn uniform(&self, position: u32) -> bool {
        self.uniform[position as usize]
    }

    /// The layer and position of `node`, when it is in a layer: never with no layers.
    pub fn position(&self, node: NodeId) -> Option<(u32, u32)> {
        let offset = node.0.checked_sub(self.first)?;
        let layer = offset.checked_div(self.width)?;
        (layer < self.count).then_some((layer, offset % self.width))
    }

    /// The kind of `tensor` as a node outside the layers reads it, with its layer weights: the last
    /// checkpoint is the last position's output in the last layer; any other tensor is itself.
    pub fn outside(&self, tensor: TensorId) -> (Kind, Layers) {
        if self.checkpoints.last() == Some(&tensor) {
            let kind = Kind::Layer {
                position: self.width - 1,
                output: self.last_output,
            };
            return (kind, Layers::One(self.count - 1));
        }
        (Kind::Tensor(tensor), Layers::None)
    }

    /// The tensor of `kind` in layer `layer`; for a [`Kind::Tensor`], the tensor itself. `None`
    /// for a layer that does not exist.
    pub fn tensor(&self, graph: &Graph, kind: Kind, layer: u32) -> Option<TensorId> {
        if layer >= self.count && !matches!(kind, Kind::Tensor(_)) {
            return None;
        }
        let node = |position: u32| graph.nodes.get(self.node(position, layer).0 as usize);
        match kind {
            Kind::Layer { position, output } => {
                node(position)?.outputs.get(output as usize).copied()
            }
            Kind::LayerInput => self.checkpoints.get(layer as usize).copied(),
            Kind::LayerParameter { position, input } => {
                node(position)?.inputs.get(input as usize).copied()
            }
            Kind::Tensor(tensor) => Some(tensor),
        }
    }

    /// The tensor that stands for `kind` in shapes and classes: layer 0's.
    pub(super) fn spec<'g>(&self, graph: &'g Graph, kind: Kind) -> &'g TensorSpec {
        let tensor = self.tensor(graph, kind, 0).unwrap_or(TensorId(0));
        &graph.tensors[tensor.0 as usize]
    }

    /// Checks the nodes at `position` in every layer against layer 0's, and records its inputs'
    /// kinds.
    fn check_position(
        &mut self,
        graph: &Graph,
        tables: &[TableSpec],
        producers: &[Option<(NodeId, u32)>],
        public: &[bool],
        position: u32,
    ) -> Result<(), CompileError> {
        let base_id = self.node(position, 0);
        let base = &graph.nodes[base_id.0 as usize];
        // Neither a public tensor's padded layers nor a lookup's non-zero at padding are specified;
        // each layer's node is checked, as a lookup may read a table of its layer's own.
        let mut unspecified = !stacks(graph, &base.op);
        for layer in 0..self.count {
            let Some(node) = graph.nodes.get(self.node(position, layer).0 as usize) else {
                continue;
            };
            unspecified |= node
                .outputs
                .iter()
                .any(|t| public.get(t.0 as usize).copied().unwrap_or(false));
            unspecified |= nonzero_at_padding(graph, tables, node)?;
        }
        if unspecified {
            return refuse(
                Some(base_id),
                "a layer template holds no ConstAdd, selection, ConstMul per channel, \
                 BitDecompose, AssertEqual, lookup non-zero at padding or node whose output is \
                 public",
            );
        }
        // Where each input comes from, in every layer.
        let mut sources: Vec<Vec<Source>> = vec![Vec::new(); base.inputs.len()];
        for layer in 0..self.count {
            let id = self.node(position, layer);
            let node = &graph.nodes[id.0 as usize];
            if !same_op(graph, &base.op, &node.op, layer) {
                return refuse(Some(id), "the nodes at one position have one operation");
            }
            if node.inputs.len() != base.inputs.len() || node.outputs.len() != base.outputs.len() {
                return refuse(
                    Some(id),
                    "the nodes at one position have as many inputs and outputs",
                );
            }
            let tensors = |a: TensorId| &graph.tensors[a.0 as usize];
            let outputs = base.outputs.iter().zip(&node.outputs);
            let inputs = base.inputs.iter().zip(&node.inputs);
            if outputs
                .chain(inputs)
                .any(|(&a, &b)| !same_tensor(tensors(a), tensors(b)))
            {
                return refuse(
                    Some(id),
                    "the tensors at one position have one shape and kind",
                );
            }
            for (i, &t) in node.inputs.iter().enumerate() {
                let source = if t == self.checkpoints[layer as usize] {
                    Source::Checkpoint
                } else {
                    match producers.get(t.0 as usize).copied().flatten() {
                        None => Source::Graph(t),
                        Some((m, output)) => match self.position(m) {
                            Some((l, p)) if l == layer && p < position => Source::Own(p, output),
                            _ => {
                                return refuse(
                                    Some(id),
                                    "a layer reads its own outputs, its checkpoint and graph \
                                     inputs only",
                                );
                            }
                        },
                    }
                };
                sources[i].push(source);
            }
        }
        let mut kinds = Vec::with_capacity(sources.len());
        for (i, sources) in (0u32..).zip(&sources) {
            let kind = match sources[0] {
                Source::Checkpoint if sources.iter().all(|&s| s == Source::Checkpoint) => {
                    Kind::LayerInput
                }
                Source::Own(p, output) if sources.iter().all(|&s| s == sources[0]) => Kind::Layer {
                    position: p,
                    output,
                },
                Source::Graph(t) => {
                    let mut tensors: Vec<TensorId> = Vec::with_capacity(sources.len());
                    for &source in sources {
                        match source {
                            Source::Graph(u) => tensors.push(u),
                            _ => {
                                return refuse(
                                    Some(base_id),
                                    "the inputs at one position come from one place in every \
                                     layer",
                                );
                            }
                        }
                    }
                    tensors.sort_unstable();
                    tensors.dedup();
                    match tensors.len() {
                        // RFC-0037 'Layers and kinds': a persistent layer parameter stays
                        // stacked at L=1 (lambda=0). Cardinality alone cannot distinguish
                        // it from a shared input; public inputs and genuinely shared
                        // inputs across multiple layers retain the case below (#598).
                        1 if self.count == 1
                            && matches!(
                                graph.tensors[t.0 as usize].kind,
                                NodeKind::Committed {
                                    commitment: CommitmentClass::Weight
                                }
                            ) =>
                        {
                            Kind::LayerParameter { position, input: i }
                        }
                        1 => Kind::Tensor(t),
                        n if n == sources.len() => Kind::LayerParameter { position, input: i },
                        _ => {
                            return refuse(
                                Some(base_id),
                                "a graph input is every layer's, or one of its own in each",
                            );
                        }
                    }
                }
                _ => {
                    return refuse(
                        Some(base_id),
                        "the inputs at one position come from one place in every layer",
                    );
                }
            };
            kinds.push(kind);
        }
        self.inputs.push(kinds);
        let constant = |layer: u32| match graph.nodes[self.node(position, layer).0 as usize].op {
            Op::ConstMul { c } => graph.constants.get(c.0 as usize).map(|values| {
                values
                    .iter()
                    .copied()
                    .map(Goldilocks::from_i64)
                    .collect::<Vec<_>>()
            }),
            _ => None,
        };
        let first = constant(0);
        self.uniform
            .push((1..self.count).all(|layer| constant(layer) == first));
        Ok(())
    }

    /// Checks that every output of a layer is read in its layer only, but for its checkpoint,
    /// which the next layer or the epilogue reads, and that the prologue's tensors other than the
    /// first checkpoint are read outside the layers.
    fn check_readers(
        &self,
        graph: &Graph,
        producers: &[Option<(NodeId, u32)>],
    ) -> Result<(), CompileError> {
        for node in &graph.nodes {
            let reader = self.position(node.id).map(|(layer, _)| layer);
            for &t in &node.inputs {
                let Some((m, _)) = producers.get(t.0 as usize).copied().flatten() else {
                    continue;
                };
                let written = self.position(m).map(|(layer, _)| layer);
                let fine = match (written, reader) {
                    (Some(w), Some(r)) => {
                        w == r || (r == w + 1 && t == self.checkpoints[r as usize])
                    }
                    (Some(w), None) => {
                        node.id.0 >= self.end()
                            && w + 1 == self.count
                            && Some(&t) == self.checkpoints.last()
                    }
                    (None, Some(r)) => r == 0 && t == self.checkpoints[0],
                    (None, None) => true,
                };
                if !fine {
                    return refuse(
                        Some(node.id),
                        "a layer's outputs are read in the layer, but for its checkpoint",
                    );
                }
            }
        }
        Ok(())
    }
}

/// Where an input of a layer's node comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Source {
    /// The layer's checkpoint `c_ℓ`.
    Checkpoint,
    /// Output `output` of the layer's node at position `p`.
    Own(u32, u32),
    /// A graph input.
    Graph(TensorId),
}

/// Whether a node of `op` may sit at a template's position (RFC-0072 'Graphs without layers' 'Not
/// inside a layer'): not a `ConstAdd`, a selection, a `ConstMul` by a constant of more than one
/// value, a `BitDecompose` or an `AssertEqual`, whose stacked forms (a per-layer constant on a
/// padded layer, stacked bit zero-checks) are not specified. A lookup non-zero at padding and a
/// node whose output is public are refused too, by [`LayerMap::of`], their padded layers being
/// unspecified. A constant of one value is a scalar,
/// whatever the last axis's length ('Scalar or per channel'). `BitFn` and `Bitwise` stack as any
/// relation and lookup do.
fn stacks(graph: &Graph, op: &Op) -> bool {
    match op {
        Op::ConstAdd { .. }
        | Op::Slice { .. }
        | Op::GatherPublic { .. }
        | Op::BitDecompose { .. }
        | Op::AssertEqual => false,
        Op::ConstMul { c } => graph
            .constants
            .get(c.0 as usize)
            .is_some_and(|v| v.len() == 1),
        _ => true,
    }
}

/// Whether two tensors at one position have one shape and kind: a lookup output's table may be
/// its layer's own.
fn same_tensor(a: &TensorSpec, b: &TensorSpec) -> bool {
    let kinds = match (a.kind, b.kind) {
        (NodeKind::LookupOutput { .. }, NodeKind::LookupOutput { .. }) => true,
        (x, y) => x == y,
    };
    a.shape == b.shape && kinds
}

/// Whether `op`, in layer `layer`, is `base` up to the layer's own weights, constants, shift
/// classes and tables.
fn same_op(graph: &Graph, base: &Op, op: &Op, layer: u32) -> bool {
    let constant = |c: tessera_ir::ConstRef| graph.constants.get(c.0 as usize).map(Vec::len);
    let classes =
        |s: tessera_ir::ShiftClassRef| graph.shift_classes.get(s.0 as usize).map(Vec::len);
    match (base, op) {
        (Op::MatMulWeight { weight: a }, Op::MatMulWeight { weight: b }) => {
            a.ty == b.ty && b.layer == layer
        }
        (Op::ConstMul { c: a }, Op::ConstMul { c: b })
        | (Op::ConstAdd { c: a }, Op::ConstAdd { c: b }) => constant(*a) == constant(*b),
        (
            Op::Requant { shift: a, lo, hi },
            Op::Requant {
                shift: b,
                lo: lo_b,
                hi: hi_b,
            },
        ) => {
            // One class or several: their number is the table's, and the nodes are alike.
            let several = |s| classes(s).map(|n| n > 1);
            several(*a) == several(*b) && lo == lo_b && hi == hi_b
        }
        (Op::Rotate { .. }, Op::Rotate { .. })
        | (Op::TableMap { .. }, Op::TableMap { .. })
        | (Op::Bitwise { .. }, Op::Bitwise { .. }) => true,
        (Op::SoftmaxRow { sigma: a, .. }, Op::SoftmaxRow { sigma: b, .. }) => a == b,
        (Op::GatherPublic { index: a }, Op::GatherPublic { index: b }) => {
            graph.constants.get(a.0 as usize) == graph.constants.get(b.0 as usize)
        }
        (a, b) => a == b,
    }
}
