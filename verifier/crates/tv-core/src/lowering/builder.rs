// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A graph without a lowering, built call by call: [`GraphBuilder`] checks RFC-0006's rules
//! ('Operation set', the table of shapes, as RFC-0066 and RFC-0072 amend it) for every node it
//! adds, and refuses the first rule a call breaks with a [`BuildError`] naming it. It is the only
//! way a graph other than [`lower`](super::lower)'s is made: the anchor circuits' builders and
//! the tests that hand-build graphs of RFC-0072's relations use it, and [`GraphBuilder::extend`]
//! appends nodes to a lowered graph.
//!
//! The graph is a pure function of the calls (INV-DM-5): ids are dense in creation order, a
//! node's outputs are created with it in output order, and a table of a kind declared before is
//! that table again. Nodes are in topological order by construction, since a node reads only
//! tensors that exist, and no tensor has two producers.
//!
//! # What it does not check
//!
//! The builder checks shapes, kinds and parameters' domains: what makes a graph one RFC-0006
//! admits, not what makes it the graph the specification defines. A wrong rotation's indices, a
//! swapped `Ch` operand or a wrong round constant pass every rule. Two preconditions follow:
//!
//! - **Its calls are the verifier's own.** Only the verifier's circuit builders, written from
//!   the specification over public sizes, and tests may drive it. It must never be fed a graph
//!   that the proof or the statement describes, node by node or in any other form: such a graph
//!   would be the prover's word for the relation it is checked against (SP-7, RFC-0019
//!   Alternative 2).
//! - **Its capacity is already bounded.** The caller computes the [`Capacity`] from sizes that
//!   the statement's maximum and the local [`ResourceLimits`](crate::ResourceLimits) have
//!   already bounded, refusing larger sizes before building (`StatementMismatch { Batch }` or
//!   `ERR-RES-MEMORY`): [`GraphBuilder::new`] reserves the whole capacity at once.
//!
//! # Declarations
//!
//! Each call checks, in this order, and the first rule broken is the error,
//! [`BuildError::Declaration`]:
//!
//! - [`input`](GraphBuilder::input)`(dims, kind, ty, lo, hi)`: the capacity; a shape of one to
//!   four axes, each of length at least 1 ([`Rule::Shape`]); `kind` public or committed advice
//!   ([`Rule::InputKind`]); `ty` of 1 to 64 bits, `lo ≤ hi`, both values of `ty`
//!   ([`Rule::Range`]). Its role is [`Role::Input`]`(k)`, the `k`-th input the builder declares.
//! - [`weight`](GraphBuilder::weight)`(role, dims, ty, lo, hi)`: the same, for an input committed
//!   in the weight commitment, whose role is a weight matrix, a class vector or a multiplier
//!   vector ([`Rule::InputKind`]), each declared once ([`Rule::Weight`]). A weight matrix joins
//!   the weight list.
//! - [`constant`](GraphBuilder::constant)`(values)`: the capacity; at least one value
//!   ([`Rule::Constant`]).
//! - [`shift_classes`](GraphBuilder::shift_classes)`(shifts)`: the capacity; one to four shifts,
//!   each from 1 to 63 ([`Rule::ShiftClasses`]).
//! - [`table`](GraphBuilder::table)`(kind)`: `AesSbox`, `Gf256Mul { c }` or `Bitwise8 { op }`,
//!   the anchor's tables, whose rows are `2^8`, `2^8` and `2^16` (RFC-0008 'Table kinds'); the
//!   lowering derives every other kind ([`Rule::Table`]); then the capacity. A kind equal to an
//!   earlier table's is that table, as RFC-0006 'Gadget selection' reuses the lowest id of an
//!   equal kind; otherwise it takes the next id.
//! - [`checkpoint`](GraphBuilder::checkpoint)`(tensor)`: the capacity; a tensor of the graph
//!   ([`Rule::Checkpoint`]). The schedule judges the layers checkpoints bound (RFC-0037).
//!
//! # Nodes
//!
//! [`node`](GraphBuilder::node)`(op, inputs, outputs)` declares each output's shape and kind and
//! checks, in this order, [`BuildError::Node`] naming the node the call would have made:
//!
//! 1. The capacity: one more node, and its outputs' tensors.
//! 2. The number of inputs ([`Rule::Inputs`]): one, except two for `MatMulWeight`, `Add`, `Sub`,
//!    `Rotate`, `MatMulAct`, `Mul`, `Bitwise` and `AssertEqual`, one or two for `Requant` and
//!    `TableMap`, and `f`'s arity for `BitFn { f }`. Then the number of outputs
//!    ([`Rule::Outputs`]): one, except two for `SoftmaxRow`, none for `AssertEqual`, and for
//!    `BitDecompose` two, or one when `high = 0`.
//! 3. Each input is a tensor of the graph ([`Rule::Dangling`]).
//! 4. Each output's shape has one to four axes, each of length at least 1 ([`Rule::Shape`]).
//! 5. Each output's kind is one its operation produces ([`Rule::OutputKind`]). A linear or
//!    degree-2 node's is virtual or committed advice, except a matmul's, virtual (RFC-0006: the
//!    schedule proves which committed outputs it can, RFC-0072); `BitFn`'s is virtual;
//!    `OneHot`'s, `Requant`'s, `RsqrtRow`'s and `BitDecompose`'s are advice; `TableMap`'s and
//!    `Bitwise`'s are the lookup output of the node's table; `SoftmaxRow`'s are the lookup output
//!    of its `Exp` table, then advice. No node's output is declared public: RFC-0072 'What is
//!    public' derives which are.
//! 6. The operation's own rules, in the order of this table:
//!
//! | Operation | Rules, in order |
//! |---|---|
//! | `OneHot { vocab }` | `vocab ≥ 1` ([`Rule::Parameter`]); `h` is `t` with an innermost axis of `vocab` ([`Rule::Output`]) |
//! | `MatMulWeight { weight }` | `W` is the matrix the weight list names for `weight` ([`Rule::Weight`]); `W [out, in]` and `x […, in]` ([`Rule::MatMul`]); `y […, out]` ([`Rule::Output`]) |
//! | `ConstMul { c }`, `ConstAdd { c }` | constant `c` exists ([`Rule::Constant`]); it has one value or as many as `x`'s last axis ([`Rule::ConstantLength`]); `y = x` ([`Rule::Output`]) |
//! | `Add`, `Sub` | `a` and `b` of one shape ([`Rule::Equal`]); `y = a` ([`Rule::Output`]) |
//! | `Rotate { rope }` | `rope` is a `Rope` table ([`Rule::Table`]); `pos` is public ([`Rule::InputKind`]); `x [n, …, 2m]` with `m ≥ 1` and `pos [n]` ([`Rule::Rotate`]); `y = x` ([`Rule::Output`]) |
//! | `Sum { axis }`, `SumSquares { axis }` | `axis` is one of `x`'s ([`Rule::Axis`]); `y` is `x` without it, `[1]` for `x`'s only axis ([`Rule::Output`]) |
//! | `Reshape` | as many elements in `y` as in `x` ([`Rule::Elements`]) |
//! | `Slice { axis, start, len }` | `axis` is one of `x`'s ([`Rule::Axis`]); `len ≥ 1` and `start + len` within the axis ([`Rule::Selection`]); `y` is `x` with that axis of length `len` ([`Rule::Output`]) |
//! | `GatherPublic { index }` | constant `index` exists ([`Rule::Constant`]); each of its values is a row of `x`'s outermost axis ([`Rule::Selection`]); `y` is `x` with one outermost row per value ([`Rule::Output`]) |
//! | `Transpose { perm }` | `perm` is a permutation of `x`'s axes ([`Rule::Permutation`]); `y`'s axis `i` is `x`'s axis `perm[i]` ([`Rule::Output`]) |
//! | `Expand { axis, len }` | `axis ≤` `x`'s rank ([`Rule::Axis`]); `len ≥ 1` ([`Rule::Parameter`]); `y` is `x` with an axis of `len` inserted before `axis` ([`Rule::Output`]) |
//! | `MatMulAct { transpose_rhs }` | `a […, m, k]` and `b […, k, n]`, or `[…, n, k]` transposed, of one rank and equal leading axes ([`Rule::MatMul`]); `y […, m, n]` ([`Rule::Output`]) |
//! | `Mul { broadcast }` | `b` is `a`'s shape or a prefix of it for `Trailing`, `a`'s last axis for `Leading` ([`Rule::Broadcast`]); `y = a` ([`Rule::Output`]) |
//! | `Requant { shift, lo, hi }` | set `shift` exists ([`Rule::ShiftClasses`]); `lo ≤ hi` ([`Rule::Parameter`]); the classes ([`Rule::Classes`], [`Rule::InputKind`]); `y = x` ([`Rule::Output`]) |
//! | `TableMap { table }` | `table` is a `Silu`, `AesSbox` or `Gf256Mul` table ([`Rule::Table`]); the classes, a `Silu` table's of several ([`Rule::Classes`], [`Rule::InputKind`]); `y = x` ([`Rule::Output`]) |
//! | `SoftmaxRow { exp_table, σ }` | `exp_table` is an `Exp` table ([`Rule::Table`]); `s […, n, n]` ([`Rule::Softmax`]); `e = s` and `R` is `s` without its last axis ([`Rule::Output`]) |
//! | `RsqrtRow { κ, e, dim }` | `dim ≥ 1` and `κ ≤ 31` ([`Rule::Parameter`]); `ρ = u` ([`Rule::Output`]) |
//! | `Bitwise { table }` | `table` is a `Bitwise8` table ([`Rule::Table`]); `a` and `b` of one shape ([`Rule::Equal`]); `y = a` ([`Rule::Output`]) |
//! | `BitDecompose { low, high }` | `1 ≤ low` and `low + high ≤ 62` ([`Rule::Parameter`]); `b` is `x` with an innermost axis of `low`, and `h` with one of `high` ([`Rule::Output`]) |
//! | `BitFn { f }` | its inputs of one shape ([`Rule::Equal`]); `y` of that shape ([`Rule::Output`]) |
//! | `AssertEqual` | `a` and `b` of one shape ([`Rule::Equal`]) |
//!
//! The classes: when the shift-class set or the table has more than one class, a second input,
//! the channel classes `c [x's last axis]`, committed in the weight commitment; with one class,
//! no second input. A second input where none belongs, none where one does, or one of another
//! shape is [`Rule::Classes`]; one of another kind [`Rule::InputKind`].
//!
//! Four readings of RFC-0006 are this module's. A `Requant`'s output is advice, as the
//! lowering's are, not a lookup output, whose table gadget selection numbers later. `OneHot`'s
//! vocabulary axis is the innermost, as in the lowering and RFC-0037. A `MatMulAct` reads its
//! operands as one sequence's, RFC-0071's slots being a batch lowering's, not a built graph's.
//! `Rotate`'s positions are a graph input declared public, as the lowering's are: a tensor
//! RFC-0072 'What is public' would derive public from public inputs is refused, until the
//! schedule derives public tensors.
//!
//! Two rules of the circuits are not the builder's. RFC-0066 'The graph' gives each
//! `GatherPublic`, `ConstMul` and `ConstAdd` of the SHA-256 circuit a constant of its own, never
//! shared, while RFC-0006's constant references may be shared: the circuit's builder keeps that
//! rule, and its tests check it. And a table's id is its declaration's: a circuit's builder that
//! declares each table just before its first node numbers them in node order of first use.
//!
//! # Capacity
//!
//! Before anything is built, the caller states a [`Capacity`], counted in closed form from the
//! public sizes it builds for (RFC-0019 'Robustness'), and the builder reserves its tensors,
//! nodes and constant values at once, fallibly ([`BuildError::Memory`]). No call then lets a
//! collection pass its bound ([`BuildError::Capacity`]): the tensors, and with them the graph
//! inputs and checkpoints, at most `tensors`; the nodes, and with them the shift-class sets and
//! the tables, at most `nodes`; the constants' values, and so the constants, at most
//! `constant_values`. A node holds at most three inputs and two outputs, inline, so the graph's
//! memory is linear in the capacity, and no call allocates in proportion to a shape's elements:
//! element counts are compared in `u128`, where four axes of `u32` cannot overflow.
//!
//! # Errors
//!
//! A graph is built by the verifier's own code from public data, so a broken rule is a fault of
//! the verifier, `Inconclusive(ERR-INTERNAL)`, as is a capacity counted wrong; memory that could
//! not be reserved is `Inconclusive(ERR-RES-MEMORY)` ([`BuildError::code`]).

use core::fmt;

use tv_tables::{TableId, TableKind};

use super::{
    Broadcast, Graph, Ids, Input, IntType, Kind, LoweredTable, Node, NodeId, Op, Role, Shape,
    ShiftClasses, Span, Tensor, TensorId, WeightRef,
};

/// The most a [`GraphBuilder`] may hold, counted by its caller before building (module
/// documentation, 'Capacity').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Capacity {
    /// Tensors, graph inputs and node outputs together; also the most graph inputs and
    /// checkpoints.
    pub tensors: u32,
    /// Nodes; also the most shift-class sets and tables.
    pub nodes: u32,
    /// The constants' values, all constants together; each constant has one at least, so this
    /// also bounds the constants.
    pub constant_values: u32,
}

/// A rule of RFC-0006 a call breaks; the module documentation says which call checks which.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Rule {
    /// A shape of no axis, of more than four, or with an axis of length 0.
    Shape,
    /// A type of no bit or of more than 64, `lo > hi`, or a bound the type does not hold.
    Range,
    /// A graph input neither public nor committed advice; a weight-committed input whose role is
    /// not a weight matrix, class vector or multiplier vector; or a node's input of a kind its
    /// operation does not read: `Rotate`'s positions not public, channel classes not committed
    /// in the weight commitment.
    InputKind,
    /// A weight-committed input declared twice, or a `MatMulWeight` whose matrix is not the one
    /// the weight list names for its `WeightRef`.
    Weight,
    /// A constant of no value, or a node naming a constant that does not exist.
    Constant,
    /// A shift-class set of no shift, of more than four, or with a shift outside 1 to 63; or a
    /// node naming a set that does not exist.
    ShiftClasses,
    /// A table kind the builder does not declare, or a node naming a table that does not exist
    /// or whose kind its operation does not read.
    Table,
    /// A checkpoint that is no tensor of the graph.
    Checkpoint,
    /// A node of another number of inputs than its operation's.
    Inputs,
    /// A node of another number of outputs than its operation's.
    Outputs,
    /// A node input that is no tensor of the graph.
    Dangling,
    /// A node output of a kind its operation does not produce.
    OutputKind,
    /// An operation's parameter outside its domain: `vocab`, `Expand`'s `len` or `dim` of 0,
    /// `κ > 31`, `lo > hi`, `low = 0` or `low + high > 62`.
    Parameter,
    /// An axis that is not one of the input's, or past its rank for `Expand`.
    Axis,
    /// A `Transpose` whose `perm` is not a permutation of its input's axes.
    Permutation,
    /// Inputs of one operation that must have one shape and do not.
    Equal,
    /// A `ConstMul` or `ConstAdd` constant of neither one value nor one per channel of the
    /// last axis.
    ConstantLength,
    /// A `Slice` of no row or past its axis, or a gather index outside the outermost axis.
    Selection,
    /// A `Mul` whose second operand does not fit its broadcast.
    Broadcast,
    /// Matmul operands whose shapes do not multiply.
    MatMul,
    /// A `Rotate` whose input is not `[n, …, 2m]` or whose positions are not `[n]`.
    Rotate,
    /// A `SoftmaxRow` whose scores are not `[…, n, n]`.
    Softmax,
    /// Channel classes where the set or table has one class, none where it has several, or not
    /// one per channel of the input's last axis.
    Classes,
    /// A `Reshape` that changes the number of elements.
    Elements,
    /// An output whose shape is not the one its operation gives its inputs.
    Output,
}

/// Why a [`GraphBuilder`] call failed (module documentation, 'Errors').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BuildError {
    /// A declaration breaks `rule`: a graph input, a weight-committed input, a constant, a
    /// shift-class set, a table or a checkpoint. Nothing was added.
    Declaration {
        /// The rule broken.
        rule: Rule,
    },
    /// The node `node`, the id the call would have given it, breaks `rule`. Nothing was added.
    Node {
        /// The node's id.
        node: NodeId,
        /// The rule broken.
        rule: Rule,
    },
    /// The call would pass the builder's [`Capacity`]: a count of the verifier's own that was
    /// wrong.
    Capacity,
    /// Memory for the graph could not be reserved.
    Memory,
}

impl BuildError {
    /// The identifier of 04 §3: `ERR-RES-MEMORY` for [`BuildError::Memory`], `ERR-INTERNAL`
    /// otherwise.
    pub const fn code(&self) -> &'static str {
        match self {
            BuildError::Memory => "ERR-RES-MEMORY",
            BuildError::Declaration { .. } | BuildError::Node { .. } | BuildError::Capacity => {
                "ERR-INTERNAL"
            }
        }
    }
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuildError::Declaration { rule } => write!(f, "a declaration breaks {rule:?}"),
            BuildError::Node { node, rule } => write!(f, "node {} breaks {rule:?}", node.0),
            BuildError::Capacity => f.write_str("the graph would pass its capacity"),
            BuildError::Memory => f.write_str("memory for the graph could not be reserved"),
        }
    }
}

impl std::error::Error for BuildError {}

/// A graph built call by call, under RFC-0006's rules (module documentation).
///
/// It checks shapes and kinds, not meaning: only the verifier's own circuit builders and tests
/// drive it, never a graph the proof or the statement describes, and its [`Capacity`] comes from
/// sizes the statement's maximum and the local resource limits already bound (module
/// documentation, 'What it does not check').
#[derive(Debug)]
#[must_use = "a builder does nothing until its graph is taken with `finish`"]
pub struct GraphBuilder {
    graph: Graph,
    capacity: Capacity,
    /// The number of [`Role::Input`] inputs so far.
    declared: u32,
}

impl GraphBuilder {
    /// A builder of an empty graph: no sequence, checkpoint or residual route.
    ///
    /// # Errors
    /// [`BuildError::Memory`] when the capacity's tensors, nodes and constant values cannot be
    /// reserved.
    pub fn new(capacity: Capacity) -> Result<Self, BuildError> {
        let graph = Graph {
            n: 0,
            tensors: Vec::new(),
            inputs: Vec::new(),
            nodes: Vec::new(),
            constant_values: Vec::new(),
            constant_spans: Vec::new(),
            shift_classes: Vec::new(),
            weights: Vec::new(),
            tables: Vec::new(),
            checkpoints: Vec::new(),
            routes: None,
        };
        Self::extend(graph, capacity)
    }

    /// A builder that appends to `graph`, a lowered graph for one: its sequence, checkpoints,
    /// routes and tables stay, and new ids follow its own. The capacity counts the whole graph.
    ///
    /// # Errors
    /// [`BuildError::Capacity`] when `graph` already holds more than `capacity`;
    /// [`BuildError::Memory`] when the rest cannot be reserved.
    pub fn extend(mut graph: Graph, capacity: Capacity) -> Result<Self, BuildError> {
        let tensors = bound(capacity.tensors)?;
        let nodes = bound(capacity.nodes)?;
        let values = bound(capacity.constant_values)?;
        if graph.tensors.len() > tensors
            || graph.inputs.len() > tensors
            || graph.checkpoints.len() > tensors
            || graph.nodes.len() > nodes
            || graph.shift_classes.len() > nodes
            || graph.tables.len() > nodes
            || graph.constant_values.len() > values
            || graph.constant_spans.len() > values
        {
            return Err(BuildError::Capacity);
        }
        reserve_to(&mut graph.tensors, tensors)?;
        reserve_to(&mut graph.nodes, nodes)?;
        reserve_to(&mut graph.constant_values, values)?;
        let declared = graph
            .inputs
            .iter()
            .filter(|i| matches!(i.role, Role::Input(_)))
            .count();
        Ok(Self {
            graph,
            capacity,
            declared: u32::try_from(declared).map_err(|_| BuildError::Capacity)?,
        })
    }

    /// The graph so far.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// The graph built.
    pub fn finish(self) -> Graph {
        self.graph
    }

    /// Declares a graph input of shape `dims`, public or committed advice, of type `ty` and
    /// declared range `[lo, hi]`: the module documentation's rules. Its role is
    /// [`Role::Input`], numbered among the builder's inputs.
    ///
    /// # Errors
    /// [`BuildError::Capacity`]; [`BuildError::Declaration`] with [`Rule::Shape`],
    /// [`Rule::InputKind`] or [`Rule::Range`]; [`BuildError::Memory`].
    pub fn input(
        &mut self,
        dims: &[u32],
        kind: Kind,
        ty: IntType,
        lo: i64,
        hi: i64,
    ) -> Result<TensorId, BuildError> {
        self.room_for_tensors(1)?;
        let shape = shape_of(dims).ok_or(declaration(Rule::Shape))?;
        if !matches!(kind, Kind::Public | Kind::Advice) {
            return Err(declaration(Rule::InputKind));
        }
        if !holds(ty, lo, hi) {
            return Err(declaration(Rule::Range));
        }
        let role = Role::Input(self.declared);
        let next = self.declared.checked_add(1).ok_or(BuildError::Capacity)?;
        let tensor = self.add_input(shape, kind, role, ty, lo, hi)?;
        self.declared = next;
        Ok(tensor)
    }

    /// Declares an input committed in the weight commitment: a weight matrix, which joins the
    /// weight list, or a weight's class or multiplier vector, each once.
    ///
    /// # Errors
    /// [`BuildError::Capacity`]; [`BuildError::Declaration`] with [`Rule::Shape`],
    /// [`Rule::InputKind`], [`Rule::Range`] or [`Rule::Weight`]; [`BuildError::Memory`].
    pub fn weight(
        &mut self,
        role: Role,
        dims: &[u32],
        ty: IntType,
        lo: i64,
        hi: i64,
    ) -> Result<TensorId, BuildError> {
        self.room_for_tensors(1)?;
        let shape = shape_of(dims).ok_or(declaration(Rule::Shape))?;
        if !matches!(
            role,
            Role::Weight(_) | Role::Classes(_) | Role::Multipliers(_)
        ) {
            return Err(declaration(Rule::InputKind));
        }
        if !holds(ty, lo, hi) {
            return Err(declaration(Rule::Range));
        }
        // A scan of every input, quadratic in the weight declarations: a lowering extended by
        // a builder holds a few hundred inputs, and the anchor's graphs declare no weight.
        if self.graph.inputs.iter().any(|i| i.role == role) {
            return Err(declaration(Rule::Weight));
        }
        // Reserved first, so that a failure leaves the graph unchanged.
        self.graph
            .weights
            .try_reserve(1)
            .map_err(|_| BuildError::Memory)?;
        let tensor = self.add_input(shape, Kind::Weight, role, ty, lo, hi)?;
        if let Role::Weight(weight) = role {
            self.graph.weights.push((weight, tensor));
        }
        Ok(tensor)
    }

    /// Declares a constant of `values`: one, one per channel, or a gather's indices. Returns
    /// its index.
    ///
    /// # Errors
    /// [`BuildError::Capacity`]; [`BuildError::Declaration`] with [`Rule::Constant`];
    /// [`BuildError::Memory`].
    pub fn constant(&mut self, values: &[i64]) -> Result<u32, BuildError> {
        let start = self.graph.constant_values.len();
        let end = start
            .checked_add(values.len())
            .ok_or(BuildError::Capacity)?;
        if end > bound(self.capacity.constant_values)? {
            return Err(BuildError::Capacity);
        }
        if values.is_empty() {
            return Err(declaration(Rule::Constant));
        }
        let index =
            u32::try_from(self.graph.constant_spans.len()).map_err(|_| BuildError::Capacity)?;
        let span = Span {
            start: u32::try_from(start).map_err(|_| BuildError::Capacity)?,
            len: u32::try_from(values.len()).map_err(|_| BuildError::Capacity)?,
        };
        // Both reserved before either grows, so that a failure leaves the graph unchanged; the
        // values were reserved to the capacity, which `end` is within.
        self.graph
            .constant_values
            .try_reserve(values.len())
            .map_err(|_| BuildError::Memory)?;
        self.graph
            .constant_spans
            .try_reserve(1)
            .map_err(|_| BuildError::Memory)?;
        self.graph.constant_spans.push(span);
        self.graph.constant_values.extend_from_slice(values);
        Ok(index)
    }

    /// Declares a shift-class set, `r_c` for each class. Returns its index.
    ///
    /// # Errors
    /// [`BuildError::Capacity`]; [`BuildError::Declaration`] with [`Rule::ShiftClasses`];
    /// [`BuildError::Memory`].
    pub fn shift_classes(&mut self, shifts: &[u8]) -> Result<u32, BuildError> {
        if self.graph.shift_classes.len() >= bound(self.capacity.nodes)? {
            return Err(BuildError::Capacity);
        }
        let mut set = ShiftClasses {
            shifts: [0; 4],
            len: u8::try_from(shifts.len()).map_err(|_| declaration(Rule::ShiftClasses))?,
        };
        let slots = set
            .shifts
            .get_mut(..shifts.len())
            .filter(|_| !shifts.is_empty())
            .ok_or(declaration(Rule::ShiftClasses))?;
        if shifts.iter().any(|r| !(1..=63).contains(r)) {
            return Err(declaration(Rule::ShiftClasses));
        }
        slots.copy_from_slice(shifts);
        let index =
            u32::try_from(self.graph.shift_classes.len()).map_err(|_| BuildError::Capacity)?;
        push(&mut self.graph.shift_classes, set)?;
        Ok(index)
    }

    /// Declares one of the anchor's tables, `AesSbox`, `Gf256Mul { c }` or `Bitwise8 { op }`:
    /// the earlier table of an equal kind, or a new one under the next id.
    ///
    /// # Errors
    /// [`BuildError::Declaration`] with [`Rule::Table`]; [`BuildError::Capacity`];
    /// [`BuildError::Memory`].
    pub fn table(&mut self, kind: TableKind) -> Result<TableId, BuildError> {
        let log2_rows = match kind {
            TableKind::AesSbox | TableKind::Gf256Mul { .. } => 8,
            TableKind::Bitwise8 { .. } => 16,
            TableKind::Range { .. }
            | TableKind::Requant { .. }
            | TableKind::Pow2 { .. }
            | TableKind::Exp { .. }
            | TableKind::Exp2Frac { .. }
            | TableKind::Silu { .. }
            | TableKind::Rope { .. } => return Err(declaration(Rule::Table)),
        };
        if let Some(earlier) = self.graph.tables.iter().find(|t| t.kind == kind) {
            return Ok(earlier.id);
        }
        if self.graph.tables.len() >= bound(self.capacity.nodes)? {
            return Err(BuildError::Capacity);
        }
        let id = TableId(u32::try_from(self.graph.tables.len()).map_err(|_| BuildError::Capacity)?);
        push(
            &mut self.graph.tables,
            LoweredTable {
                id,
                kind,
                log2_rows,
            },
        )?;
        Ok(id)
    }

    /// Declares `tensor` the next checkpoint.
    ///
    /// # Errors
    /// [`BuildError::Capacity`]; [`BuildError::Declaration`] with [`Rule::Checkpoint`];
    /// [`BuildError::Memory`].
    pub fn checkpoint(&mut self, tensor: TensorId) -> Result<(), BuildError> {
        if self.graph.checkpoints.len() >= bound(self.capacity.tensors)? {
            return Err(BuildError::Capacity);
        }
        if self.graph.tensor(tensor).is_none() {
            return Err(declaration(Rule::Checkpoint));
        }
        push(&mut self.graph.checkpoints, tensor)
    }

    /// Adds a node of `op` reading `inputs`, its outputs of the shapes and kinds `outputs`
    /// declares, in order, after the checks of the module documentation. Returns the node, whose
    /// [`outputs`](Node::outputs) are the new tensors.
    ///
    /// # Errors
    /// [`BuildError::Capacity`]; [`BuildError::Node`] with the first rule broken;
    /// [`BuildError::Memory`].
    pub fn node(
        &mut self,
        op: Op,
        inputs: &[TensorId],
        outputs: &[(&[u32], Kind)],
    ) -> Result<Node, BuildError> {
        // 1. The capacity.
        let count = self.graph.nodes.len();
        if count >= bound(self.capacity.nodes)? {
            return Err(BuildError::Capacity);
        }
        self.room_for_tensors(outputs.len())?;
        let id = NodeId(u32::try_from(count).map_err(|_| BuildError::Capacity)?);
        let fail = |rule| BuildError::Node { node: id, rule };
        // 2. The numbers of inputs and outputs.
        let (least, most) = input_arity(&op);
        if inputs.len() < least || inputs.len() > most {
            return Err(fail(Rule::Inputs));
        }
        if outputs.len() != output_arity(&op) {
            return Err(fail(Rule::Outputs));
        }
        // 3. Every input exists.
        if inputs.iter().any(|&t| self.graph.tensor(t).is_none()) {
            return Err(fail(Rule::Dangling));
        }
        // 4. The outputs' shapes.
        let mut shapes = [Shape {
            dims: [0; 4],
            rank: 0,
        }; 2];
        for (slot, (dims, _)) in shapes.iter_mut().zip(outputs) {
            *slot = shape_of(dims).ok_or(fail(Rule::Shape))?;
        }
        // 5. The outputs' kinds.
        for (index, &(_, kind)) in outputs.iter().enumerate() {
            if !produces(&op, index, kind) {
                return Err(fail(Rule::OutputKind));
            }
        }
        // 6. The operation's own rules.
        let shapes = shapes.get(..outputs.len()).ok_or(fail(Rule::Outputs))?;
        self.check(&op, inputs, shapes).map_err(fail)?;

        // The node, whole, before the graph changes: its outputs' ids are the next tensors'.
        let first = self.graph.tensors.len();
        let mut created = [TensorId(0); 2];
        let slots = created
            .get_mut(..outputs.len())
            .ok_or(fail(Rule::Outputs))?;
        for (offset, slot) in slots.iter_mut().enumerate() {
            let tensor = first.checked_add(offset).ok_or(BuildError::Capacity)?;
            *slot = TensorId(u32::try_from(tensor).map_err(|_| BuildError::Capacity)?);
        }
        let node = Node {
            op,
            inputs: Ids::of(inputs).map_err(|_| fail(Rule::Inputs))?,
            outputs: Ids::of(created.get(..outputs.len()).ok_or(fail(Rule::Outputs))?)
                .map_err(|_| fail(Rule::Outputs))?,
        };
        // Room for both, then the pushes, which cannot fail: nothing is added unless all is.
        self.graph
            .tensors
            .try_reserve(outputs.len())
            .map_err(|_| BuildError::Memory)?;
        self.graph
            .nodes
            .try_reserve(1)
            .map_err(|_| BuildError::Memory)?;
        for (&shape, &(_, kind)) in shapes.iter().zip(outputs) {
            self.graph.tensors.push(Tensor {
                shape,
                kind,
                producer: Some(id),
            });
        }
        self.graph.nodes.push(node);
        Ok(node)
    }

    /// [`node`](GraphBuilder::node) for an operation of one output, of shape `dims` and kind
    /// `kind`: that output.
    ///
    /// # Errors
    /// Those of [`node`](GraphBuilder::node).
    pub fn node1(
        &mut self,
        op: Op,
        inputs: &[TensorId],
        dims: &[u32],
        kind: Kind,
    ) -> Result<TensorId, BuildError> {
        let id = NodeId(u32::try_from(self.graph.nodes.len()).map_err(|_| BuildError::Capacity)?);
        let node = self.node(op, inputs, &[(dims, kind)])?;
        node.outputs().first().copied().ok_or(BuildError::Node {
            node: id,
            rule: Rule::Outputs,
        })
    }

    // ----- Checks ------------------------------------------------------------------------------

    /// Room for `count` more tensors.
    fn room_for_tensors(&self, count: usize) -> Result<(), BuildError> {
        let total = self
            .graph
            .tensors
            .len()
            .checked_add(count)
            .ok_or(BuildError::Capacity)?;
        if total > bound(self.capacity.tensors)? {
            return Err(BuildError::Capacity);
        }
        Ok(())
    }

    /// The rules of `op` itself, row by row of the module documentation's table; `outputs` holds
    /// the declared shapes, of the right number.
    fn check(&self, op: &Op, inputs: &[TensorId], outputs: &[Shape]) -> Result<(), Rule> {
        let x = self.dims(inputs, 0)?;
        let y = |index: usize| -> Result<&[u32], Rule> {
            outputs.get(index).map(Shape::dims).ok_or(Rule::Outputs)
        };
        let expect = |index: usize, dims: &[u32]| -> Result<(), Rule> {
            if y(index)? == dims {
                Ok(())
            } else {
                Err(Rule::Output)
            }
        };
        match *op {
            Op::OneHot { vocab } => {
                if vocab == 0 {
                    return Err(Rule::Parameter);
                }
                expect(0, appended(x, vocab)?.as_slice())
            }
            Op::MatMulWeight { weight } => {
                let w = *inputs.get(1).ok_or(Rule::Inputs)?;
                if self.weight_matrix(weight) != Some(w) {
                    return Err(Rule::Weight);
                }
                let (&[out, inner], Some(&last)) = (self.dims(inputs, 1)?, x.last()) else {
                    return Err(Rule::MatMul);
                };
                if last != inner {
                    return Err(Rule::MatMul);
                }
                expect(0, replaced_last(x, out)?.as_slice())
            }
            Op::ConstMul { c } | Op::ConstAdd { c } => {
                let values = self.graph.constant(c).ok_or(Rule::Constant)?;
                let channels = x.last().map(|&d| usize::try_from(d).unwrap_or(usize::MAX));
                if values.len() != 1 && Some(values.len()) != channels {
                    return Err(Rule::ConstantLength);
                }
                expect(0, x)
            }
            Op::Add | Op::Sub | Op::Bitwise { .. } | Op::AssertEqual => {
                if let Op::Bitwise { table } = *op
                    && !self.table_is(table, |k| matches!(k, TableKind::Bitwise8 { .. }))
                {
                    return Err(Rule::Table);
                }
                if self.dims(inputs, 1)? != x {
                    return Err(Rule::Equal);
                }
                match op {
                    Op::AssertEqual => Ok(()),
                    _ => expect(0, x),
                }
            }
            Op::Rotate { rope } => {
                if !self.table_is(rope, |k| matches!(k, TableKind::Rope { .. })) {
                    return Err(Rule::Table);
                }
                let positions = *inputs.get(1).ok_or(Rule::Inputs)?;
                if self.kind(positions)? != Kind::Public {
                    return Err(Rule::InputKind);
                }
                let (Some(&n), Some(&width)) = (x.first(), x.last()) else {
                    return Err(Rule::Rotate);
                };
                if x.len() < 2
                    || width < 2
                    || !width.is_multiple_of(2)
                    || self.dims(inputs, 1)? != [n]
                {
                    return Err(Rule::Rotate);
                }
                expect(0, x)
            }
            Op::Sum { axis } | Op::SumSquares { axis } => {
                let axis = usize::from(axis);
                if axis >= x.len() {
                    return Err(Rule::Axis);
                }
                let kept = removed(x, axis)?;
                if kept.as_slice().is_empty() {
                    expect(0, &[1])
                } else {
                    expect(0, kept.as_slice())
                }
            }
            Op::Reshape => {
                if elements(x) != elements(y(0)?) {
                    return Err(Rule::Elements);
                }
                Ok(())
            }
            Op::Slice { axis, start, len } => {
                let axis = usize::from(axis);
                let length = *x.get(axis).ok_or(Rule::Axis)?;
                let end = start.checked_add(len);
                if len == 0 || end.is_none_or(|end| end > length) {
                    return Err(Rule::Selection);
                }
                let mut sliced = Shape::from_dims(x)?;
                *sliced.dims.get_mut(axis).ok_or(Rule::Axis)? = len;
                expect(0, sliced.dims())
            }
            Op::GatherPublic { index } => {
                let values = self.graph.constant(index).ok_or(Rule::Constant)?;
                let rows = i64::from(*x.first().ok_or(Rule::Selection)?);
                if values.iter().any(|&v| v < 0 || v >= rows) {
                    return Err(Rule::Selection);
                }
                let mut gathered = Shape::from_dims(x)?;
                *gathered.dims.first_mut().ok_or(Rule::Selection)? =
                    u32::try_from(values.len()).map_err(|_| Rule::Selection)?;
                expect(0, gathered.dims())
            }
            Op::Transpose { perm } => {
                let mut seen = [false; 4];
                if perm.len() != x.len() {
                    return Err(Rule::Permutation);
                }
                for &a in perm.iter() {
                    let slot = seen
                        .get_mut(usize::from(a))
                        .filter(|_| usize::from(a) < x.len())
                        .ok_or(Rule::Permutation)?;
                    if *slot {
                        return Err(Rule::Permutation);
                    }
                    *slot = true;
                }
                let mut moved = Shape::from_dims(x)?;
                for (slot, &a) in moved.dims.iter_mut().zip(perm.iter()) {
                    *slot = *x.get(usize::from(a)).ok_or(Rule::Permutation)?;
                }
                expect(0, moved.dims())
            }
            Op::Expand { axis, len } => {
                let axis = usize::from(axis);
                if axis > x.len() {
                    return Err(Rule::Axis);
                }
                if len == 0 {
                    return Err(Rule::Parameter);
                }
                expect(0, inserted(x, axis, len)?.as_slice())
            }
            Op::MatMulAct { transpose_rhs } => {
                let b = self.dims(inputs, 1)?;
                let rank = x.len();
                let lead = rank.checked_sub(2).ok_or(Rule::MatMul)?;
                if b.len() != rank || x.get(..lead) != b.get(..lead) {
                    return Err(Rule::MatMul);
                }
                // `a […, m, k]`; `b […, k, n]`, or `[…, n, k]` transposed.
                let k = *x.get(lead.saturating_add(1)).ok_or(Rule::MatMul)?;
                let (Some(&b0), Some(&b1)) = (b.get(lead), b.get(lead.saturating_add(1))) else {
                    return Err(Rule::MatMul);
                };
                let (inner, n) = if transpose_rhs { (b1, b0) } else { (b0, b1) };
                if inner != k {
                    return Err(Rule::MatMul);
                }
                expect(0, replaced_last(x, n)?.as_slice())
            }
            Op::Mul { broadcast } => {
                let b = self.dims(inputs, 1)?;
                let fits = match broadcast {
                    Broadcast::Trailing => x.get(..b.len()) == Some(b),
                    Broadcast::Leading => x.last().is_some_and(|&last| b == [last]),
                };
                if !fits {
                    return Err(Rule::Broadcast);
                }
                expect(0, x)
            }
            Op::Requant { shift, lo, hi } => {
                let classes = self
                    .graph
                    .shift_classes
                    .get(usize::try_from(shift).map_err(|_| Rule::ShiftClasses)?)
                    .ok_or(Rule::ShiftClasses)?
                    .as_slice()
                    .len();
                if lo > hi {
                    return Err(Rule::Parameter);
                }
                self.classes(inputs, x, classes)?;
                expect(0, x)
            }
            Op::TableMap { table } => {
                let kind = &self.graph.table(table).ok_or(Rule::Table)?.kind;
                let classes = match kind {
                    TableKind::Silu { r_by_class, .. } => r_by_class.len(),
                    TableKind::AesSbox | TableKind::Gf256Mul { .. } => 1,
                    TableKind::Range { .. }
                    | TableKind::Requant { .. }
                    | TableKind::Pow2 { .. }
                    | TableKind::Bitwise8 { .. }
                    | TableKind::Exp { .. }
                    | TableKind::Exp2Frac { .. }
                    | TableKind::Rope { .. } => return Err(Rule::Table),
                };
                self.classes(inputs, x, classes)?;
                expect(0, x)
            }
            Op::SoftmaxRow { exp_table, .. } => {
                if !self.table_is(exp_table, |k| matches!(k, TableKind::Exp { .. })) {
                    return Err(Rule::Table);
                }
                let rank = x.len();
                let (Some(&rows), Some(&keys)) = (
                    rank.checked_sub(2).and_then(|r| x.get(r)),
                    rank.checked_sub(1).and_then(|r| x.get(r)),
                ) else {
                    return Err(Rule::Softmax);
                };
                if rows != keys {
                    return Err(Rule::Softmax);
                }
                expect(0, x)?;
                expect(1, x.get(..rank.saturating_sub(1)).ok_or(Rule::Softmax)?)
            }
            Op::RsqrtRow { kappa, dim, .. } => {
                if dim == 0 || kappa > 31 {
                    return Err(Rule::Parameter);
                }
                expect(0, x)
            }
            Op::BitDecompose { low, high } => {
                if low == 0 || u16::from(low).saturating_add(u16::from(high)) > 62 {
                    return Err(Rule::Parameter);
                }
                expect(0, appended(x, u32::from(low))?.as_slice())?;
                if high > 0 {
                    expect(1, appended(x, u32::from(high))?.as_slice())?;
                }
                Ok(())
            }
            Op::BitFn { .. } => {
                for index in 1..inputs.len() {
                    if self.dims(inputs, index)? != x {
                        return Err(Rule::Equal);
                    }
                }
                expect(0, x)
            }
        }
    }

    /// The classes rule: a second input `c [x's last axis]`, weight-committed, exactly when
    /// there are several classes.
    fn classes(&self, inputs: &[TensorId], x: &[u32], classes: usize) -> Result<(), Rule> {
        match (inputs.get(1), classes > 1) {
            (None, false) => Ok(()),
            (Some(_), false) | (None, true) => Err(Rule::Classes),
            (Some(&c), true) => {
                if self.dims(inputs, 1)? != x.last().map(core::slice::from_ref).unwrap_or(&[]) {
                    return Err(Rule::Classes);
                }
                if self.kind(c)? != Kind::Weight {
                    return Err(Rule::InputKind);
                }
                Ok(())
            }
        }
    }

    /// The shape of input `index`, which exists by step 3.
    fn dims(&self, inputs: &[TensorId], index: usize) -> Result<&[u32], Rule> {
        let id = *inputs.get(index).ok_or(Rule::Inputs)?;
        Ok(self.graph.tensor(id).ok_or(Rule::Dangling)?.shape.dims())
    }

    /// The kind of `tensor`, which exists by step 3.
    fn kind(&self, tensor: TensorId) -> Result<Kind, Rule> {
        Ok(self.graph.tensor(tensor).ok_or(Rule::Dangling)?.kind)
    }

    /// Whether table `id` exists and its kind passes `is`.
    fn table_is(&self, id: TableId, is: impl Fn(&TableKind) -> bool) -> bool {
        self.graph.table(id).is_some_and(|t| is(&t.kind))
    }

    /// The tensor the weight list names for `weight`.
    fn weight_matrix(&self, weight: WeightRef) -> Option<TensorId> {
        self.graph
            .weights
            .iter()
            .find(|(w, _)| *w == weight)
            .map(|&(_, t)| t)
    }

    /// Appends a graph input, its tensor first.
    fn add_input(
        &mut self,
        shape: Shape,
        kind: Kind,
        role: Role,
        ty: IntType,
        lo: i64,
        hi: i64,
    ) -> Result<TensorId, BuildError> {
        if self.graph.inputs.len() >= bound(self.capacity.tensors)? {
            return Err(BuildError::Capacity);
        }
        let tensor =
            TensorId(u32::try_from(self.graph.tensors.len()).map_err(|_| BuildError::Capacity)?);
        // Reserve both before pushing either, so that a failure leaves the graph unchanged.
        self.graph
            .inputs
            .try_reserve(1)
            .map_err(|_| BuildError::Memory)?;
        push(
            &mut self.graph.tensors,
            Tensor {
                shape,
                kind,
                producer: None,
            },
        )?;
        self.graph.inputs.push(Input {
            tensor,
            role,
            ty,
            lo,
            hi,
        });
        Ok(tensor)
    }
}

impl Shape {
    /// The shape of `dims`, an existing tensor's, to edit.
    fn from_dims(dims: &[u32]) -> Result<Self, Rule> {
        shape_of(dims).ok_or(Rule::Shape)
    }
}

/// The least and most inputs of `op`.
fn input_arity(op: &Op) -> (usize, usize) {
    match op {
        Op::OneHot { .. }
        | Op::ConstMul { .. }
        | Op::ConstAdd { .. }
        | Op::Reshape
        | Op::Sum { .. }
        | Op::SumSquares { .. }
        | Op::Slice { .. }
        | Op::GatherPublic { .. }
        | Op::Transpose { .. }
        | Op::Expand { .. }
        | Op::RsqrtRow { .. }
        | Op::SoftmaxRow { .. }
        | Op::BitDecompose { .. } => (1, 1),
        Op::Requant { .. } | Op::TableMap { .. } => (1, 2),
        Op::MatMulWeight { .. }
        | Op::Add
        | Op::Sub
        | Op::Rotate { .. }
        | Op::MatMulAct { .. }
        | Op::Mul { .. }
        | Op::Bitwise { .. }
        | Op::AssertEqual => (2, 2),
        Op::BitFn { f } => (f.arity(), f.arity()),
    }
}

/// The number of outputs of `op`.
fn output_arity(op: &Op) -> usize {
    match *op {
        Op::AssertEqual => 0,
        Op::SoftmaxRow { .. } => 2,
        Op::BitDecompose { high, .. } => {
            if high == 0 {
                1
            } else {
                2
            }
        }
        Op::OneHot { .. }
        | Op::MatMulWeight { .. }
        | Op::ConstMul { .. }
        | Op::Add
        | Op::Reshape
        | Op::SumSquares { .. }
        | Op::RsqrtRow { .. }
        | Op::Mul { .. }
        | Op::Requant { .. }
        | Op::Rotate { .. }
        | Op::Transpose { .. }
        | Op::Expand { .. }
        | Op::MatMulAct { .. }
        | Op::TableMap { .. }
        | Op::Sub
        | Op::ConstAdd { .. }
        | Op::Sum { .. }
        | Op::Slice { .. }
        | Op::GatherPublic { .. }
        | Op::Bitwise { .. }
        | Op::BitFn { .. } => 1,
    }
}

/// Whether `op` produces output `index` of kind `kind` (step 5).
fn produces(op: &Op, index: usize, kind: Kind) -> bool {
    match *op {
        // Matmuls and `BitFn`: virtual.
        Op::MatMulWeight { .. } | Op::MatMulAct { .. } | Op::BitFn { .. } => kind == Kind::Virtual,
        // The other linear and degree-2 nodes: virtual or committed advice.
        Op::ConstMul { .. }
        | Op::Add
        | Op::Reshape
        | Op::SumSquares { .. }
        | Op::Mul { .. }
        | Op::Rotate { .. }
        | Op::Transpose { .. }
        | Op::Expand { .. }
        | Op::Sub
        | Op::ConstAdd { .. }
        | Op::Sum { .. }
        | Op::Slice { .. }
        | Op::GatherPublic { .. } => matches!(kind, Kind::Virtual | Kind::Advice),
        // Gadgets that commit advice.
        Op::OneHot { .. } | Op::Requant { .. } | Op::RsqrtRow { .. } | Op::BitDecompose { .. } => {
            kind == Kind::Advice
        }
        // Lookups: the output of the node's own table.
        Op::TableMap { table } | Op::Bitwise { table } => kind == Kind::Lookup { table },
        Op::SoftmaxRow { exp_table, .. } => match index {
            0 => kind == Kind::Lookup { table: exp_table },
            _ => kind == Kind::Advice,
        },
        Op::AssertEqual => false,
    }
}

/// The shape of `dims` when RFC-0006 admits it and the verifier holds it: one to four axes, each
/// of length at least 1.
fn shape_of(dims: &[u32]) -> Option<Shape> {
    if dims.is_empty() || dims.contains(&0) {
        return None;
    }
    Shape::of(dims).ok()
}

/// Whether `ty` holds `lo` and `hi`, `lo ≤ hi`, and has 1 to 64 bits.
fn holds(ty: IntType, lo: i64, hi: i64) -> bool {
    let bits = u32::from(ty.bits);
    if !(1..=64).contains(&bits) || lo > hi {
        return false;
    }
    // At most 64 bits, so `2^bits` is exact in `i128`.
    let Some(power) = 1i128.checked_shl(if ty.signed {
        bits.saturating_sub(1)
    } else {
        bits
    }) else {
        return false;
    };
    let (least, greatest) = if ty.signed {
        (power.saturating_neg(), power.saturating_sub(1))
    } else {
        (0, power.saturating_sub(1))
    };
    least <= i128::from(lo) && i128::from(hi) <= greatest
}

/// A shape derived from an input's, of up to five axes, so that one with an axis more than a
/// tensor holds can still be compared, and differ.
struct Derived {
    dims: [u32; 5],
    len: usize,
}

impl Derived {
    fn as_slice(&self) -> &[u32] {
        self.dims.get(..self.len).unwrap_or(&[])
    }

    /// `parts` one after the other, at most five axes in all.
    fn of(parts: &[&[u32]]) -> Result<Self, Rule> {
        let mut derived = Self {
            dims: [0; 5],
            len: 0,
        };
        for part in parts {
            let end = derived.len.checked_add(part.len()).ok_or(Rule::Output)?;
            derived
                .dims
                .get_mut(derived.len..end)
                .ok_or(Rule::Output)?
                .copy_from_slice(part);
            derived.len = end;
        }
        Ok(derived)
    }
}

/// `dims` with an innermost axis of `len`.
fn appended(dims: &[u32], len: u32) -> Result<Derived, Rule> {
    Derived::of(&[dims, &[len]])
}

/// `dims` with an axis of `len` before axis `axis`.
fn inserted(dims: &[u32], axis: usize, len: u32) -> Result<Derived, Rule> {
    let (before, after) = (
        dims.get(..axis).ok_or(Rule::Axis)?,
        dims.get(axis..).ok_or(Rule::Axis)?,
    );
    Derived::of(&[before, &[len], after])
}

/// `dims` without axis `axis`.
fn removed(dims: &[u32], axis: usize) -> Result<Derived, Rule> {
    let (before, after) = (
        dims.get(..axis).ok_or(Rule::Axis)?,
        dims.get(axis.saturating_add(1)..).ok_or(Rule::Axis)?,
    );
    Derived::of(&[before, after])
}

/// `dims` with its last axis of length `len`.
fn replaced_last(dims: &[u32], len: u32) -> Result<Derived, Rule> {
    let lead = dims.len().checked_sub(1).ok_or(Rule::Output)?;
    Derived::of(&[dims.get(..lead).ok_or(Rule::Output)?, &[len]])
}

/// The number of elements of a shape of at most four `u32` axes, exact in `u128`.
fn elements(dims: &[u32]) -> u128 {
    dims.iter()
        .fold(1u128, |n, &d| n.saturating_mul(u128::from(d)))
}

fn declaration(rule: Rule) -> BuildError {
    BuildError::Declaration { rule }
}

/// A capacity's bound as a length.
fn bound(capacity: u32) -> Result<usize, BuildError> {
    usize::try_from(capacity).map_err(|_| BuildError::Memory)
}

/// Reserves `values` to hold `total` entries in all.
fn reserve_to<T>(values: &mut Vec<T>, total: usize) -> Result<(), BuildError> {
    let more = total.saturating_sub(values.len());
    values
        .try_reserve_exact(more)
        .map_err(|_| BuildError::Memory)
}

/// Pushes `value`, its memory reserved fallibly.
fn push<T>(values: &mut Vec<T>, value: T) -> Result<(), BuildError> {
    values.try_reserve(1).map_err(|_| BuildError::Memory)?;
    values.push(value);
    Ok(())
}
