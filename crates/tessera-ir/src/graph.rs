// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The dataflow graph (RFC-0006 'Operation set'): tensors, nodes, the public declarations they
//! name, and the rules a node must keep.

use tessera_tables::TableId;

use crate::{
    Broadcast, CommitmentClass, ConstRef, IntRange, IntType, IrError, NodeId, NodeKind, Op,
    OpClass, ShiftClassRef, TensorDecl, TensorId, TensorSpec, WeightRef,
};

/// The most elements one tensor may hold: `2^48`, beyond any v1 tile.
const MAX_ELEMENTS: u64 = 1 << 48;

/// A node: an operation applied to input tensors, producing output tensors.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Node {
    /// The node's index.
    pub id: NodeId,
    /// The operation.
    pub op: Op,
    /// The input tensors, in the operation's order.
    pub inputs: Vec<TensorId>,
    /// The output tensors, in the operation's order.
    pub outputs: Vec<TensorId>,
}

/// A dataflow graph, the constraint system the proof schedule derives its stages from (RFC-0010).
/// Nodes are in topological order: each input is a graph input, a tensor no node produces, or
/// the output of an earlier node. Ids are dense in position order, so two builds from the same
/// calls are equal (INV-DM-5).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Graph {
    /// The nodes, in topological order.
    pub nodes: Vec<Node>,
    /// The tensors, by id.
    pub tensors: Vec<TensorSpec>,
    /// The public constants, by [`ConstRef`]: each a scalar or one value per channel.
    pub constants: Vec<Vec<i64>>,
    /// The public shift-class sets, by [`ShiftClassRef`]: the shift `r_c` of each class.
    pub shift_classes: Vec<Vec<u8>>,
    /// The weight tensors, each with the matrix it holds, in declaration order.
    pub weights: Vec<(WeightRef, TensorId)>,
}

impl Graph {
    /// The node that produces `tensor`, or `None` for a graph input.
    pub fn producer(&self, tensor: TensorId) -> Option<NodeId> {
        self.nodes
            .iter()
            .find(|node| node.outputs.contains(&tensor))
            .map(|node| node.id)
    }

    /// Checks every rule a [`GraphBuilder`] keeps, for a graph assembled otherwise: dense ids,
    /// well-formed tensors and declarations, one producer per tensor, topological order, inputs
    /// that are public or committed, and each node's operation rules.
    ///
    /// # Errors
    ///
    /// The first rule broken, as an [`IrError`], checking tensors and declarations first and
    /// then the nodes in order.
    pub fn validate(&self) -> Result<(), IrError> {
        for (position, tensor) in (0u32..).zip(&self.tensors) {
            if tensor.id != TensorId(position) {
                return Err(IrError::Id {
                    rule: "tensor ids are their positions",
                });
            }
            check_tensor(tensor.id, &tensor.shape, tensor.ty, tensor.range)?;
        }
        for (position, node) in (0u32..).zip(&self.nodes) {
            if node.id != NodeId(position) {
                return Err(IrError::Id {
                    rule: "node ids are their positions",
                });
            }
        }
        self.constants.iter().try_for_each(|c| check_constant(c))?;
        self.shift_classes
            .iter()
            .try_for_each(|s| check_shift_classes(s))?;
        let tensor_count = self.tensors.len();
        let mut producer = vec![None; tensor_count];
        for node in &self.nodes {
            for &output in &node.outputs {
                let slot = producer.get_mut(index(output)).ok_or(IrError::Id {
                    rule: "every output names a tensor",
                })?;
                if slot.replace(node.id).is_some() {
                    return Err(IrError::ProducedTwice { tensor: output });
                }
            }
        }
        for (position, &(weight, tensor)) in self.weights.iter().enumerate() {
            if self.weights[..position].iter().any(|&(w, _)| w == weight) {
                return Err(IrError::DuplicateWeight { weight });
            }
            let spec = self.tensors.get(index(tensor)).ok_or(IrError::Id {
                rule: "every weight names a tensor",
            })?;
            check_weight_tensor(spec)?;
            if producer[index(tensor)].is_some() {
                return Err(IrError::InputKind { tensor });
            }
        }
        for tensor in &self.tensors {
            if producer[index(tensor.id)].is_none() {
                check_input_kind(tensor.id, tensor.kind)?;
            }
        }
        for node in &self.nodes {
            for (input, &tensor) in node.inputs.iter().enumerate() {
                if index(tensor) >= tensor_count {
                    return Err(IrError::UnknownTensor {
                        node: node.id,
                        input,
                    });
                }
                if producer[index(tensor)].is_some_and(|p| p >= node.id) {
                    return Err(IrError::Order {
                        node: node.id,
                        input,
                    });
                }
            }
            let inputs: Vec<View<'_>> = node
                .inputs
                .iter()
                .map(|&t| View::of(&self.tensors[index(t)]))
                .collect();
            let outputs: Vec<View<'_>> = node
                .outputs
                .iter()
                .map(|&t| View::of(&self.tensors[index(t)]))
                .collect();
            self.check_op(node.id, &node.op, &node.inputs, &inputs, &outputs)?;
        }
        Ok(())
    }

    /// The rules of `op` for a node `node` with these inputs and outputs (RFC-0006 'Operation
    /// set').
    fn check_op(
        &self,
        node: NodeId,
        op: &Op,
        input_ids: &[TensorId],
        inputs: &[View<'_>],
        outputs: &[View<'_>],
    ) -> Result<(), IrError> {
        let fail = |rule| {
            Err(IrError::Op {
                node,
                op: op.name(),
                rule,
            })
        };
        let unknown = |what| Err(IrError::UnknownReference { node, what });
        let arity = |n_in: usize, n_out: usize, rule| {
            if inputs.len() == n_in && outputs.len() == n_out {
                Ok(())
            } else {
                fail(rule)
            }
        };
        // Linear and degree-2 operations produce virtual tensors (RFC-0006).
        if op.class() != OpClass::Gadget && outputs.iter().any(|o| o.kind != NodeKind::Virtual) {
            return fail("a linear or degree-2 operation produces a virtual tensor");
        }
        match op {
            Op::MatMulWeight { weight } => {
                arity(
                    2,
                    1,
                    "takes the input and the weight, and produces one tensor",
                )?;
                let Some(&(_, w)) = self.weights.iter().find(|(r, _)| r == weight) else {
                    return unknown("weight");
                };
                if input_ids[1] != w {
                    return fail("the second input is the weight tensor its WeightRef declares");
                }
                let (x, wt, y) = (inputs[0].shape, inputs[1].shape, outputs[0].shape);
                let [rows, cols] = wt else {
                    return fail("the weight is a matrix");
                };
                if last(x) != *cols {
                    return fail("the weight's columns match the input's last axis");
                }
                if y.len() != x.len() || y[..y.len() - 1] != x[..x.len() - 1] || last(y) != *rows {
                    return fail("the output is the input with its last axis the weight's rows");
                }
            }
            Op::Add | Op::Sub => {
                arity(2, 1, "takes two tensors and produces one")?;
                if inputs[0].shape != inputs[1].shape || outputs[0].shape != inputs[0].shape {
                    return fail("the operands and the output have one shape");
                }
            }
            Op::ConstMul { c } | Op::ConstAdd { c } => {
                arity(1, 1, "takes one tensor and produces one")?;
                let Some(values) = self.constants.get(index_of(c.0)) else {
                    return unknown("constant");
                };
                let x = inputs[0].shape;
                if values.len() != 1 && values.len() != usize::try_from(last(x)).unwrap_or(0) {
                    return fail("the constant is a scalar or one value per channel");
                }
                if outputs[0].shape != x {
                    return fail("the output has the input's shape");
                }
            }
            Op::Rotate { .. } => {
                arity(
                    2,
                    1,
                    "takes the tensor and its rows' positions, and produces one",
                )?;
                let (x, positions) = (inputs[0].shape, inputs[1]);
                if positions.kind != NodeKind::Public {
                    return fail("the positions are public");
                }
                if positions.shape != [x[0]] {
                    return fail("one position per row of the outermost axis");
                }
                if !last(x).is_multiple_of(2) {
                    return fail("the last axis holds pairs");
                }
                if outputs[0].shape != x {
                    return fail("the output has the input's shape");
                }
            }
            Op::Sum { axis } | Op::SumSquares { axis } => {
                arity(1, 1, "takes one tensor and produces one")?;
                let x = inputs[0].shape;
                if usize::from(*axis) >= x.len() {
                    return fail("the axis is one of the input's");
                }
                if outputs[0].shape != without_axis(x, usize::from(*axis)) {
                    return fail("the output is the input without the axis");
                }
            }
            Op::Reshape => {
                arity(1, 1, "takes one tensor and produces one")?;
                if elements(outputs[0].shape) != elements(inputs[0].shape) {
                    return fail("the output has as many elements as the input");
                }
            }
            Op::Slice { axis, start, len } => {
                arity(1, 1, "takes one tensor and produces one")?;
                let (x, axis) = (inputs[0].shape, usize::from(*axis));
                if axis >= x.len() {
                    return fail("the axis is one of the input's");
                }
                if *len == 0 || u64::from(*start) + u64::from(*len) > u64::from(x[axis]) {
                    return fail("the slice is non-empty and inside the axis");
                }
                let mut expected = x.to_vec();
                expected[axis] = *len;
                if outputs[0].shape != expected {
                    return fail("the output is the input with the axis cut to the slice");
                }
            }
            Op::GatherPublic { index } => {
                arity(1, 1, "takes one tensor and produces one")?;
                let Some(rows) = self.constants.get(index_of(index.0)) else {
                    return unknown("constant");
                };
                let x = inputs[0].shape;
                if rows.iter().any(|&r| r < 0 || r >= i64::from(x[0])) {
                    return fail("every index is a row of the outermost axis");
                }
                let mut expected = vec![u32::try_from(rows.len()).unwrap_or(u32::MAX)];
                expected.extend_from_slice(&x[1..]);
                if outputs[0].shape != expected {
                    return fail("the output has one row per index");
                }
            }
            Op::MatMulAct { transpose_rhs } => {
                arity(2, 1, "takes two tensors and produces one")?;
                let (a, b, y) = (inputs[0].shape, inputs[1].shape, outputs[0].shape);
                let r = a.len();
                if r < 2 || b.len() != r || a[..r - 2] != b[..r - 2] {
                    return fail("the operands are matrices with equal leading axes");
                }
                // `[…, m, k] × […, k, n]`, or `[…, m, k] × […, n, k]` transposed.
                let (inner, outer) = if *transpose_rhs {
                    (b[r - 1], b[r - 2])
                } else {
                    (b[r - 2], b[r - 1])
                };
                if a[r - 1] != inner {
                    return fail("the inner dimensions agree");
                }
                let mut expected = a[..r - 1].to_vec();
                expected.push(outer);
                if y != expected {
                    return fail("the output is the operands' product shape");
                }
            }
            Op::Mul { broadcast } => {
                arity(2, 1, "takes two tensors and produces one")?;
                let (a, b) = (inputs[0].shape, inputs[1].shape);
                match broadcast {
                    Broadcast::Trailing if b.len() > a.len() || a[..b.len()] != *b => {
                        return fail("the second operand has the first's shape or a prefix of it");
                    }
                    Broadcast::Leading if b.len() != 1 || a.last() != b.first() => {
                        return fail("the second operand has the first's last axis");
                    }
                    _ => {}
                }
                if outputs[0].shape != a {
                    return fail("the output has the first operand's shape");
                }
            }
            Op::OneHot { vocab } => {
                arity(1, 1, "takes the tokens and produces their one-hot vectors")?;
                if *vocab == 0 {
                    return fail("the vocabulary is not empty");
                }
                let mut expected = inputs[0].shape.to_vec();
                expected.push(*vocab);
                if outputs[0].shape != expected {
                    return fail("the output adds an axis of the vocabulary's size");
                }
                if outputs[0].kind != advice() {
                    return fail("the one-hot vectors are committed advice");
                }
            }
            Op::Requant { shift, lo, hi } => {
                let Some(classes) = self.shift_classes.get(index_of(shift.0)) else {
                    return unknown("set of shift classes");
                };
                let n_in = if classes.len() > 1 { 2 } else { 1 };
                arity(
                    n_in,
                    1,
                    "takes the tensor, with the channel classes when there are several, and \
                     produces one",
                )?;
                if lo > hi {
                    return fail("lo ≤ hi");
                }
                let x = inputs[0].shape;
                if n_in == 2 {
                    let c = inputs[1];
                    if c.kind != weight() || c.shape != [last(x)] {
                        return fail("the channel classes are committed weights, one per channel");
                    }
                }
                if outputs[0].shape != x {
                    return fail("the output has the input's shape");
                }
                if !matches!(outputs[0].kind, NodeKind::LookupOutput { .. })
                    && outputs[0].kind != advice()
                {
                    return fail("the output is committed, as a lookup output or as advice");
                }
            }
            Op::TableMap { table } => {
                if !(1..=2).contains(&inputs.len()) || outputs.len() != 1 {
                    return fail(
                        "takes the tensor, with the channel classes of a classed table, and \
                         produces one",
                    );
                }
                let x = inputs[0].shape;
                if let Some(c) = inputs.get(1)
                    && (c.kind != weight() || c.shape != [last(x)])
                {
                    return fail("the channel classes are committed weights, one per channel");
                }
                if outputs[0].shape != x {
                    return fail("the output has the input's shape");
                }
                if outputs[0].kind != lookup_output(*table) {
                    return fail("the output is the table's lookup output");
                }
            }
            Op::Bitwise { table } => {
                arity(2, 1, "takes its operands and produces one tensor")?;
                if inputs.iter().any(|i| i.shape != inputs[0].shape)
                    || outputs[0].shape != inputs[0].shape
                {
                    return fail("the operands and the output have one shape");
                }
                if outputs[0].kind != lookup_output(*table) {
                    return fail("the output is the table's lookup output");
                }
            }
            Op::SoftmaxRow { exp_table, .. } => {
                arity(
                    1,
                    2,
                    "takes the scores and produces the exponentials and reciprocals",
                )?;
                let x = inputs[0].shape;
                let r = x.len();
                if r < 2 || x[r - 1] != x[r - 2] {
                    return fail(
                        "the scores' last two axes are the query and key positions of one sequence",
                    );
                }
                if outputs[0].shape != x || outputs[1].shape != without_axis(x, r - 1) {
                    return fail("one exponential per score and one reciprocal per row");
                }
                if outputs[0].kind != lookup_output(*exp_table) || outputs[1].kind != advice() {
                    return fail("the exponentials are Exp lookup outputs, the reciprocals advice");
                }
            }
            Op::RsqrtRow { kappa, dim, .. } => {
                arity(
                    1,
                    1,
                    "takes the sums of squares and produces one root per row",
                )?;
                if *dim == 0 {
                    return fail("the rows have at least one element");
                }
                // Keeps `N = dim · 2^{2κ}` below `2^94`, an exact `u128` in the analysis.
                if *kappa > 31 {
                    return fail("κ ≤ 31");
                }
                if outputs[0].shape != inputs[0].shape {
                    return fail("one root per sum");
                }
                if outputs[0].kind != advice() {
                    return fail("the roots are committed advice");
                }
            }
            Op::Argmax | Op::InverseCdf { .. } => {
                arity(
                    2,
                    0,
                    "takes the logits and the one-hot tokens, and produces nothing",
                )?;
                if inputs[1].shape != inputs[0].shape {
                    return fail("one one-hot entry per logit");
                }
            }
        }
        Ok(())
    }
}

/// Builds a [`Graph`] in topological order, checking each declaration and node as it is added,
/// so that a graph it finishes passes [`Graph::validate`]. A refused call leaves the builder
/// as it was.
#[derive(Clone, Debug, Default)]
pub struct GraphBuilder {
    graph: Graph,
}

impl GraphBuilder {
    /// An empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declares a graph input: a public or committed tensor no node produces.
    ///
    /// # Errors
    ///
    /// [`IrError::Tensor`] for a malformed description, [`IrError::InputKind`] for a virtual
    /// tensor or a lookup output.
    pub fn input(&mut self, decl: TensorDecl) -> Result<TensorId, IrError> {
        let id = self.next_tensor(0);
        check_tensor(id, &decl.shape, decl.ty, decl.range)?;
        check_input_kind(id, decl.kind)?;
        Ok(self.push_tensor(decl))
    }

    /// Declares the weight matrix `weight`, of `rows × cols` INT8 values in `[−127, 127]`
    /// (RFC-0006 'Range analysis'), committed in the weight commitment.
    ///
    /// # Errors
    ///
    /// [`IrError::DuplicateWeight`] for a matrix declared before, [`IrError::Tensor`] for an
    /// empty or oversized one.
    pub fn weight(&mut self, weight: WeightRef, rows: u32, cols: u32) -> Result<TensorId, IrError> {
        if self.graph.weights.iter().any(|&(w, _)| w == weight) {
            return Err(IrError::DuplicateWeight { weight });
        }
        let decl = weight_decl(rows, cols);
        check_tensor(self.next_tensor(0), &decl.shape, decl.ty, decl.range)?;
        let id = self.push_tensor(decl);
        self.graph.weights.push((weight, id));
        Ok(id)
    }

    /// Declares a public constant: a scalar or one value per channel.
    ///
    /// # Errors
    ///
    /// [`IrError::Declaration`] for an empty one.
    pub fn constant(&mut self, values: Vec<i64>) -> Result<ConstRef, IrError> {
        check_constant(&values)?;
        let id = ConstRef(u32::try_from(self.graph.constants.len()).unwrap_or(u32::MAX));
        self.graph.constants.push(values);
        Ok(id)
    }

    /// Declares a public set of shift classes, `r_c` for each class `c`.
    ///
    /// # Errors
    ///
    /// [`IrError::Declaration`] unless it has one to four classes, each with `1 ≤ r_c ≤ 63`
    /// (RFC-0007, RFC-0008).
    pub fn shift_classes(&mut self, r_by_class: Vec<u8>) -> Result<ShiftClassRef, IrError> {
        check_shift_classes(&r_by_class)?;
        let id = ShiftClassRef(u32::try_from(self.graph.shift_classes.len()).unwrap_or(u32::MAX));
        self.graph.shift_classes.push(r_by_class);
        Ok(id)
    }

    /// Adds a node applying `op` to `inputs`, producing a new tensor for each of `outputs`.
    ///
    /// # Errors
    ///
    /// [`IrError::Tensor`] for a malformed output, [`IrError::UnknownTensor`] for an input that
    /// does not exist yet, [`IrError::UnknownReference`] for an undeclared constant, set of shift
    /// classes or weight, and [`IrError::Op`] for a rule of the operation.
    pub fn node(
        &mut self,
        op: Op,
        inputs: &[TensorId],
        outputs: Vec<TensorDecl>,
    ) -> Result<Vec<TensorId>, IrError> {
        let node = NodeId(u32::try_from(self.graph.nodes.len()).unwrap_or(u32::MAX));
        for (k, decl) in outputs.iter().enumerate() {
            check_tensor(self.next_tensor(k), &decl.shape, decl.ty, decl.range)?;
        }
        let mut input_views = Vec::with_capacity(inputs.len());
        for (input, &tensor) in inputs.iter().enumerate() {
            let spec = self
                .graph
                .tensors
                .get(index(tensor))
                .ok_or(IrError::UnknownTensor { node, input })?;
            input_views.push(View::of(spec));
        }
        let output_views: Vec<View<'_>> = outputs
            .iter()
            .map(|decl| View {
                shape: &decl.shape,
                kind: decl.kind,
            })
            .collect();
        self.graph
            .check_op(node, &op, inputs, &input_views, &output_views)?;
        let ids: Vec<TensorId> = outputs
            .into_iter()
            .map(|decl| self.push_tensor(decl))
            .collect();
        self.graph.nodes.push(Node {
            id: node,
            op,
            inputs: inputs.to_vec(),
            outputs: ids.clone(),
        });
        Ok(ids)
    }

    /// The graph built so far.
    pub fn finish(self) -> Graph {
        self.graph
    }

    fn next_tensor(&self, offset: usize) -> TensorId {
        TensorId(u32::try_from(self.graph.tensors.len() + offset).unwrap_or(u32::MAX))
    }

    fn push_tensor(&mut self, decl: TensorDecl) -> TensorId {
        let id = self.next_tensor(0);
        self.graph.tensors.push(TensorSpec {
            id,
            shape: decl.shape,
            ty: decl.ty,
            range: decl.range,
            kind: decl.kind,
        });
        id
    }
}

/// What the rules read of a tensor.
#[derive(Clone, Copy)]
struct View<'a> {
    shape: &'a [u32],
    kind: NodeKind,
}

impl<'a> View<'a> {
    fn of(spec: &'a TensorSpec) -> Self {
        Self {
            shape: &spec.shape,
            kind: spec.kind,
        }
    }
}

fn index(tensor: TensorId) -> usize {
    index_of(tensor.0)
}

fn index_of(id: u32) -> usize {
    usize::try_from(id).unwrap_or(usize::MAX)
}

fn advice() -> NodeKind {
    NodeKind::Committed {
        commitment: CommitmentClass::Advice,
    }
}

fn weight() -> NodeKind {
    NodeKind::Committed {
        commitment: CommitmentClass::Weight,
    }
}

fn lookup_output(table: TableId) -> NodeKind {
    NodeKind::LookupOutput { table }
}

fn weight_decl(rows: u32, cols: u32) -> TensorDecl {
    TensorDecl {
        shape: vec![rows, cols],
        ty: IntType::I8,
        range: IntRange { lo: -127, hi: 127 },
        kind: weight(),
    }
}

/// The last axis of a non-empty shape.
fn last(shape: &[u32]) -> u32 {
    shape.last().copied().unwrap_or(0)
}

/// `shape` without `axis`, or `[1]` when nothing would be left.
fn without_axis(shape: &[u32], axis: usize) -> Vec<u32> {
    let rest: Vec<u32> = shape
        .iter()
        .enumerate()
        .filter(|&(k, _)| k != axis)
        .map(|(_, &d)| d)
        .collect();
    if rest.is_empty() { vec![1] } else { rest }
}

/// The number of elements of a checked shape.
fn elements(shape: &[u32]) -> u64 {
    shape.iter().map(|&d| u64::from(d)).product()
}

fn check_tensor(id: TensorId, shape: &[u32], ty: IntType, range: IntRange) -> Result<(), IrError> {
    let rule = |rule| Err(IrError::Tensor { tensor: id, rule });
    if shape.is_empty() || shape.contains(&0) {
        return rule("a shape has at least one axis, each of length at least 1");
    }
    let count = shape
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(u64::from(d)));
    if count.is_none_or(|n| n > MAX_ELEMENTS) {
        return rule("a tensor holds at most 2^48 elements");
    }
    if !(1..=64).contains(&ty.bits) {
        return rule("an integer type has 1 to 64 bits");
    }
    if range.lo > range.hi {
        return rule("a range has lo ≤ hi");
    }
    Ok(())
}

fn check_input_kind(tensor: TensorId, kind: NodeKind) -> Result<(), IrError> {
    match kind {
        NodeKind::Public | NodeKind::Committed { .. } => Ok(()),
        NodeKind::Virtual | NodeKind::LookupOutput { .. } => Err(IrError::InputKind { tensor }),
    }
}

fn check_weight_tensor(spec: &TensorSpec) -> Result<(), IrError> {
    let ok = spec.shape.len() == 2 && spec.kind == weight();
    if ok {
        Ok(())
    } else {
        Err(IrError::Declaration {
            rule: "a weight tensor is a committed weight matrix",
        })
    }
}

fn check_constant(values: &[i64]) -> Result<(), IrError> {
    if values.is_empty() {
        return Err(IrError::Declaration {
            rule: "a constant holds at least one value",
        });
    }
    Ok(())
}

fn check_shift_classes(r_by_class: &[u8]) -> Result<(), IrError> {
    if !(1..=4).contains(&r_by_class.len()) || r_by_class.iter().any(|r| !(1..=63).contains(r)) {
        return Err(IrError::Declaration {
            rule: "a set of shift classes has one to four, each with 1 ≤ r_c ≤ 63",
        });
    }
    Ok(())
}
