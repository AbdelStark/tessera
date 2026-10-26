// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The Llama lowering, node by node (RFC-0044, which supersedes RFC-0006 'Llama lowering' for
//! the graph's ids): [`lower`] builds, from a claim list's public part alone, the graph the
//! compiler builds from `D`, for a sequence of `n` tokens. The claim list's fields it reads are
//! the architecture, the profile, `PublicQuantSpec` and `silu_in_exponents`, as RFC-0042 says;
//! it never reads the claim list's tables, which validation compares with the tables it derives
//! (RFC-0007 step 5.3).
//!
//! # Ids
//!
//! Every tensor has a dense id in creation order: a graph input when the procedure that names
//! it creates it, a node's outputs with the node, in output order. Nodes, constants and
//! shift-class sets have dense ids of their own, in creation order, and each mention of a
//! constant or a set creates a new entry. Table 0 is RoPE's; each `Exp` and `Silu` table takes the
//! next id when its definition is new and the earlier table's otherwise. Gadget selection's tables
//! follow, and are not the lowering's (RFC-0006 'Gadget selection').
//!
//! # The procedures
//!
//! With `n` the length, `d = d_model`, `h = n_heads`, `kv = n_kv_heads`, `g = h / kv`,
//! `dh = d_head`, `ff = d_ff`, `V = vocab`, `b = residual_bits` and `σ = exp_sigma`, each
//! creates, in order (outputs written `[shape] kind`; RFC-0044 'Procedures'):
//!
//! - `matmul(T, ℓ, x, rows, cols)`: the input `W_{T,ℓ}` `[rows, cols]`, appended to the weight
//!   list; `MatMulWeight { (T, ℓ) }(x, W)` → `[n, rows]` virtual.
//! - `key(x, p, c)`: at an affine site of shift `r`, the multiplier vector of the weight before
//!   `p`, `[c]`, then `Mul { Leading }(x, m)` → `[n, c]` virtual, with the shifts `[r]`; at a `pot`
//!   site of one class, nothing; at a `pot` site of `k ≥ 2` classes, the class vector `κ` of the
//!   weight before `p`, `[c]`.
//! - `requant(z, shifts, κ, c, lo, hi)`: a shift-class set, then `Requant`(`z` or `z, κ`) →
//!   `[n, c]` advice.
//! - `rescale(x, p, c)`: `key`, then `requant` with `lo = −128`, `hi = 127`.
//! - `norm(res, p)`: `SumSquares { 1 }` → `[n]` virtual; `RsqrtRow { κ, e, d }` → `[n]` advice;
//!   `Mul { Trailing }(res, ρ)` → `[n, d]` virtual; `rescale(·, p, d)`.
//! - `rope(x, k)`: `Reshape` → `[n, k, dh]`, `Rotate { 0 }(·, positions)`, `Reshape` → `[n, k·dh]`,
//!   all virtual.
//! - `heads(x, k)`: `Reshape` → `[n, k, dh]`, `Transpose { [1, 0, 2] }` → `[k, n, dh]`; when
//!   `k ≠ h`, `Expand { 1, g }` → `[k, g, n, dh]` and `Reshape` → `[h, n, dh]`, all virtual.
//! - `residual(acc, res, p)`: `key(acc, p, d)`, then the site's route, the same in every layer
//!   (RFC-0038 'The residual's route'). The fused key, when every layer's site has one shift
//!   `r ≤ 62`, of one class or affine: a constant `[2^r]`, `ConstMul(res)`, `Add(z, ·)`, a set
//!   `[r]`, `Requant { −2^{b−1}, 2^{b−1} − 1 }` → advice. Two steps otherwise:
//!   `requant(z, shifts, κ, d, 1 − 2^b, 2^b − 1)`, `Add(res, u)`, a constant `[2]`, `ConstMul`, a
//!   set `[1]`, `Requant { −2^{b−1}, 2^{b−1} − 1 }` → advice.
//!
//! The graph: the inputs `tokens` `[n]` advice and `positions` `[n]` public; the prologue
//! (`OneHot`, `matmul(Embed)`, a constant `[2^{s_e}]` and the `ConstMul` that commits the first
//! checkpoint); each layer as RFC-0044 'A layer' lists it, the attention's `Exp` table named
//! between the scores and `SoftmaxRow`, SiLU's table and class vector before its `TableMap`;
//! and the epilogue, the final norm and the LM head. An input's type is the narrowest unsigned
//! one that holds its values, of at least one bit, except a weight's, `i8`.
//!
//! # The tables
//!
//! - `Exp`: with `w = bitlen(128 · 255 · dh)`, for `key_bits` from `w` up to 32, the definition
//!   `Exp { key_bits, σ, scale }`: an earlier table of that definition is the layer's; otherwise a
//!   definition that maps its last key `2^{key_bits} − 1` to 0 is a new table; the first width
//!   that gives a table ends the search. The value is read, not the zero tail
//!   ([`exp_maps_to_zero`], RFC-0030). RFC-0007 step 2 guarantees that width exists and that
//!   RFC-0008 allows its table (RFC-0044, as amended), so the search never fails on a public
//!   part that [`check_public`](crate::validation::check_public) accepted: a search that did
//!   would be [`LowerError::Internal`].
//! - `Silu { bitlen(127 · 128 · d) + 1, the shifts of p(Silu), the layer's silu_in_exponent, the
//!   exponent of p(Silu) }`, an earlier table of the same definition reused.
//!
//! A derived table's `log2_rows` is `⌈log2 T⌉` of RFC-0008's rows: `max_context · d_head / 2`
//! for `Rope`, `2^{key_bits}` for `Exp`, `|r_by_class| · 2^b` for `Silu`. The lowering does not
//! judge its tables against RFC-0008's parameter rules: validation does, when it compares them
//! with the claim list's tables, each of which it has regenerated (RFC-0007 steps 4 and 5.3).
//!
//! # Tables RFC-0008 refuses
//!
//! RFC-0007 step 2 guarantees that RFC-0008 allows every `Exp` table the lowering derives. A
//! `Rope` or `Silu` table may still be one RFC-0008 refuses, such as RoPE's pairs past `2^24` or
//! a SiLU core past `2^24` values: the lowering derives it as it derives any other, and
//! validation's comparison of the derived tables with the claim list's, each of which it has
//! regenerated, refuses it (RFC-0007 steps 4 and 5.3).
//!
//! # Cost
//!
//! The graph holds about seventy tensors and sixty nodes per layer, each in a few words: shapes,
//! operands and shift-class sets are held inline. Every collection is reserved fallibly, so a
//! claim list of more layers than memory holds is [`LowerError::Memory`], never an abort. The
//! work is linear in the layers: an `Exp` search tries at most 17 widths, each one key, and
//! table definitions are found again by hashing. The map is only looked up, never iterated, so
//! no id depends on its order, and its keys are random per process, so a claim list cannot
//! choose definitions that collide.
//!
//! # Graphs without a lowering
//!
//! The IR also holds the operations RFC-0066 and RFC-0072 add for the anchor circuits: `Sub`,
//! `ConstAdd`, `Sum`, `Slice`, `GatherPublic`, `Bitwise`, `BitDecompose`, `BitFn` and
//! `AssertEqual`, a `ConstMul` by a constant per channel, and lookups into `AesSbox`, `Gf256Mul`
//! and `Bitwise8`. The lowering creates none of them. A [`GraphBuilder`] makes a graph of them,
//! with no checkpoint, no sequence length and no residual route, and checks RFC-0006's shape and
//! arity rules node by node; its module documentation lists them.
//!
//! The checks that read a graph after the lowering, range analysis, gadget selection, the
//! schedule, the advice layout, the token axes and the gadgets' relations, prove the Llama
//! lowering's operations only. Each refuses a node of any other operation, the first in node
//! order, with its own typed error, until #150 adds that operation's relations to it.

use std::collections::HashMap;

use tv_codec::Rational;
use tv_tables::{TableId, TableKind, exp_maps_to_zero};

use crate::claims::{Profile, QuantPoint, Site, WeightType};
use crate::validation::{PublicPart, Rescale};

mod builder;
pub use builder::{BuildError, Capacity, GraphBuilder, Rule};

/// A tensor's dense id, in creation order (03 §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(pub u32);

/// A node's dense id, in creation order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

/// A weight matrix: its type and layer (RFC-0006's `WeightRef`). `Embed` and `LmHead` are at
/// layer 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WeightRef {
    /// The weight's type.
    pub ty: WeightType,
    /// The weight's layer.
    pub layer: u32,
}

/// What a tensor is (03 §3's `NodeKind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// `Committed { Advice }`.
    Advice,
    /// `Committed { Weight }`.
    Weight,
    /// `Public`: known to the verifier.
    Public,
    /// `Virtual`: never committed.
    Virtual,
    /// `LookupOutput { table }`: the committed value of a lookup into `table`.
    Lookup {
        /// The table looked up.
        table: TableId,
    },
}

/// An integer type (03 §1's `IntType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IntType {
    /// The bits, from 1 to 64.
    pub bits: u8,
    /// Whether it is signed.
    pub signed: bool,
}

/// What a graph input holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Role {
    /// The sequence's tokens.
    Tokens,
    /// The sequence's positions, `0` to `n − 1`.
    Positions,
    /// A weight matrix.
    Weight(WeightRef),
    /// The shift-class index of each output channel of a weight, committed with it.
    Classes(WeightRef),
    /// The affine multiplier of each output channel of a weight, committed with it.
    Multipliers(WeightRef),
    /// A public or advice-committed input a [`GraphBuilder`] declares, numbered from 0 among
    /// the builder's inputs in declaration order: a circuit's message bits, initial value, masks
    /// or committed bytes (RFC-0066 'The graph'), for example.
    Input(u32),
}

/// A graph input: a tensor no node produces, with its declared type and range `[lo, hi]`
/// (RFC-0044 'Graph inputs').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Input {
    /// The input's tensor.
    pub tensor: TensorId,
    /// What it holds.
    pub role: Role,
    /// Its type.
    pub ty: IntType,
    /// The least value it may hold.
    pub lo: i64,
    /// The greatest value it may hold.
    pub hi: i64,
}

/// A tensor's shape: one to four axes, held inline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: [u32; 4],
    rank: u8,
}

impl Shape {
    /// The shape of `dims`, at most four axes.
    fn of(dims: &[u32]) -> Result<Self, LowerError> {
        let mut shape = Self {
            dims: [0; 4],
            rank: u8::try_from(dims.len()).map_err(|_| LowerError::Internal)?,
        };
        let slots = shape
            .dims
            .get_mut(..dims.len())
            .ok_or(LowerError::Internal)?;
        slots.copy_from_slice(dims);
        Ok(shape)
    }

    /// The axes' lengths, outermost first.
    pub fn dims(&self) -> &[u32] {
        self.dims.get(..usize::from(self.rank)).unwrap_or(&[])
    }
}

/// A permutation of a tensor's axes, one to four, held inline: `Transpose`'s output axis `i` is
/// its input axis `perm[i]`. It reads as the slice of those axes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Perm {
    axes: [u8; 4],
    rank: u8,
}

impl Perm {
    /// The permutation listing `axes`; `None` past four. Whether they permute a tensor's axes is
    /// [`Rule::Permutation`]'s question, which a [`GraphBuilder`] asks of each `Transpose`.
    pub fn new(axes: &[u8]) -> Option<Self> {
        let mut perm = Self {
            axes: [0; 4],
            rank: u8::try_from(axes.len()).ok()?,
        };
        perm.axes.get_mut(..axes.len())?.copy_from_slice(axes);
        Some(perm)
    }

    /// The axes, output axis by output axis.
    pub fn as_slice(&self) -> &[u8] {
        self.axes.get(..usize::from(self.rank)).unwrap_or(&[])
    }
}

impl core::ops::Deref for Perm {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl TryFrom<Vec<u8>> for Perm {
    /// The axes given back, more than four of them.
    type Error = Vec<u8>;

    fn try_from(axes: Vec<u8>) -> Result<Self, Vec<u8>> {
        Self::new(&axes).ok_or(axes)
    }
}

/// A Boolean function of bits, as its multilinear polynomial (RFC-0066 '`BitFn { f }`').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BoolFn {
    /// `a ⊕ b`: `a + b − 2ab`.
    Xor,
    /// `a ⊕ b ⊕ c`: `a + b + c − 2ab − 2ac − 2bc + 4abc`.
    Xor3,
    /// `a ∧ b`: `ab`.
    And,
    /// FIPS 180-4's `(e ∧ f) ⊕ (¬e ∧ g)`: `ef + g − eg`.
    Ch,
    /// The majority of three bits: `ab + ac + bc − 2abc`.
    Maj,
}

impl BoolFn {
    /// Its number of inputs: two for `Xor` and `And`, three for `Xor3`, `Ch` and `Maj`.
    pub const fn arity(self) -> usize {
        match self {
            BoolFn::Xor | BoolFn::And => 2,
            BoolFn::Xor3 | BoolFn::Ch | BoolFn::Maj => 3,
        }
    }
}

/// A tensor of the graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tensor {
    /// Its shape.
    pub shape: Shape,
    /// Its kind.
    pub kind: Kind,
    /// The node that produces it, `None` for a graph input.
    pub producer: Option<NodeId>,
}

/// How `Mul`'s second operand broadcasts (RFC-0025 'Lowering').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Broadcast {
    /// One value per row: `b`'s shape is a prefix of `a`'s.
    Trailing,
    /// One value per channel: `b`'s shape is `a`'s last axis.
    Leading,
}

/// The operations of the verifier's IR, with RFC-0006's names and parameters: the Llama
/// lowering's, then those RFC-0066 and RFC-0072 add for the anchor circuits, which only a
/// [`GraphBuilder`] creates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    /// The committed one-hot token vectors.
    OneHot {
        /// The vocabulary's size.
        vocab: u32,
    },
    /// `W x`, `W` committed in the weight commitment.
    MatMulWeight {
        /// The weight.
        weight: WeightRef,
    },
    /// Multiplication by a public constant: one value, or one per channel of the last axis
    /// (RFC-0072 'Linear operations that are not coordinate maps'). The lowering's are all one
    /// value.
    ConstMul {
        /// The constant's index.
        c: u32,
    },
    /// Elementwise sum.
    Add,
    /// A change of shape.
    Reshape,
    /// The sum of squares along an axis.
    SumSquares {
        /// The axis summed.
        axis: u8,
    },
    /// RMSNorm's root, a multi-limb bracket.
    RsqrtRow {
        /// `κ`.
        kappa: u8,
        /// The integer epsilon `e`.
        eps_int: u64,
        /// The row's width.
        dim: u32,
    },
    /// Elementwise product.
    Mul {
        /// How the second operand broadcasts.
        broadcast: Broadcast,
    },
    /// A requantization (RFC-0005's `rq`).
    Requant {
        /// The shift-class set's index.
        shift: u32,
        /// The clamp's lower bound.
        lo: i64,
        /// The clamp's upper bound.
        hi: i64,
    },
    /// RoPE's pair rotation by the constants of a `Rope` table.
    Rotate {
        /// The `Rope` table.
        rope: TableId,
    },
    /// A permutation of the axes.
    Transpose {
        /// Output axis `i` is input axis `perm[i]`.
        perm: Perm,
    },
    /// An axis inserted and the input repeated along it.
    Expand {
        /// The axis inserted.
        axis: u8,
        /// Its length.
        len: u32,
    },
    /// Activation by activation: `QK^T` (transposed) or `PV`.
    MatMulAct {
        /// Whether the right operand is transposed.
        transpose_rhs: bool,
    },
    /// Causal softmax rows: the exponentials, looked up, and the reciprocals.
    SoftmaxRow {
        /// The `Exp` table.
        exp_table: TableId,
        /// `σ`.
        sigma: u8,
    },
    /// A single-input lookup: SiLU's in the lowering; `AesSbox` and `Gf256Mul` in AES's
    /// (RFC-0072 'Gadgets and leaf sites').
    TableMap {
        /// The table.
        table: TableId,
    },
    /// Elementwise difference, `a − b`.
    Sub,
    /// Addition of a public constant, one value or one per channel of the last axis, at real
    /// positions only (RFC-0072 'Padded positions').
    ConstAdd {
        /// The constant's index.
        c: u32,
    },
    /// The sum along an axis.
    Sum {
        /// The axis summed.
        axis: u8,
    },
    /// Rows `start` to `start + len − 1` of an axis (RFC-0072 'The selection relation').
    Slice {
        /// The axis cut.
        axis: u8,
        /// The first row kept.
        start: u32,
        /// The rows kept.
        len: u32,
    },
    /// The rows of the outermost axis that a public constant lists, in its order (RFC-0072 'The
    /// selection relation').
    GatherPublic {
        /// The index constant's index.
        index: u32,
    },
    /// A byte operation of two operands, by lookup into a `Bitwise8` table: AES's alone
    /// (RFC-0066).
    Bitwise {
        /// The `Bitwise8` table.
        table: TableId,
    },
    /// The committed bits of a true integer in two parts: `low` result bits and `high` carry
    /// bits, each output's bit axis innermost (RFC-0066 '`BitDecompose { low, high }`').
    BitDecompose {
        /// The result bits.
        low: u8,
        /// The carry bits; 0 for none, and no second output.
        high: u8,
    },
    /// A Boolean function of two or three bit tensors (RFC-0066 '`BitFn { f }`').
    BitFn {
        /// The function.
        f: BoolFn,
    },
    /// The relation `a = b` elementwise, with no output (RFC-0066 '`AssertEqual`').
    AssertEqual,
}

/// At most three tensor ids, held inline: a node's inputs, up to `BitFn`'s three, or its
/// outputs, up to two.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Ids {
    ids: [TensorId; 3],
    len: u8,
}

impl Ids {
    fn of(ids: &[TensorId]) -> Result<Self, LowerError> {
        let mut held = Self {
            ids: [TensorId(0); 3],
            len: u8::try_from(ids.len()).map_err(|_| LowerError::Internal)?,
        };
        let slots = held.ids.get_mut(..ids.len()).ok_or(LowerError::Internal)?;
        slots.copy_from_slice(ids);
        Ok(held)
    }

    fn as_slice(&self) -> &[TensorId] {
        self.ids.get(..usize::from(self.len)).unwrap_or(&[])
    }
}

/// A node: an operation, its inputs and its outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Node {
    op: Op,
    inputs: Ids,
    outputs: Ids,
}

impl Node {
    /// The operation.
    pub fn op(&self) -> &Op {
        &self.op
    }

    /// The input tensors, in order.
    pub fn inputs(&self) -> &[TensorId] {
        self.inputs.as_slice()
    }

    /// The output tensors, in order.
    pub fn outputs(&self) -> &[TensorId] {
        self.outputs.as_slice()
    }
}

/// A shift-class set: one to four shifts, held inline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShiftClasses {
    shifts: [u8; 4],
    len: u8,
}

impl ShiftClasses {
    /// The set of `shifts`, each from 1 to 63 and at most four, as validation's step 2 leaves
    /// them.
    fn of(shifts: &[i8]) -> Result<Self, LowerError> {
        let mut set = Self {
            shifts: [0; 4],
            len: u8::try_from(shifts.len()).map_err(|_| LowerError::Internal)?,
        };
        let slots = set
            .shifts
            .get_mut(..shifts.len())
            .ok_or(LowerError::Internal)?;
        for (slot, &shift) in slots.iter_mut().zip(shifts) {
            *slot = u8::try_from(shift).map_err(|_| LowerError::Internal)?;
        }
        Ok(set)
    }

    /// The shifts, one per class, in order.
    pub fn as_slice(&self) -> &[u8] {
        self.shifts.get(..usize::from(self.len)).unwrap_or(&[])
    }
}

/// A table the lowering derives: its id, its definition and `⌈log2 T⌉` of its rows.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LoweredTable {
    /// The table's id.
    pub id: TableId,
    /// Its definition.
    pub kind: TableKind,
    /// `⌈log2 T⌉`.
    pub log2_rows: u8,
}

/// A residual site's route, the same in every layer (RFC-0038 'The residual's route').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Route {
    /// One `Requant` of `acc + 2^r · res`.
    FusedKey,
    /// The update rescaled within `±(2^b − 1)`, then `2(res + u)` with the shift 1.
    TwoSteps,
}

/// Where a constant's values lie among all constants' values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Span {
    start: u32,
    len: u32,
}

/// The graph of one sequence, as [`lower`] gives it, or a graph a [`GraphBuilder`] made.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Graph {
    /// The sequence's length; 0 for a graph without a lowering.
    n: u32,
    tensors: Vec<Tensor>,
    inputs: Vec<Input>,
    nodes: Vec<Node>,
    /// Every constant's values, constant after constant.
    constant_values: Vec<i64>,
    /// Each constant's values among them, by index.
    constant_spans: Vec<Span>,
    shift_classes: Vec<ShiftClasses>,
    weights: Vec<(WeightRef, TensorId)>,
    tables: Vec<LoweredTable>,
    checkpoints: Vec<TensorId>,
    /// The residual sites' routes; `None` for a graph without a lowering.
    routes: Option<[Route; 2]>,
}

impl Graph {
    /// The sequence's length, from 1 to `max_context` for a lowered graph; 0 for a graph a
    /// [`GraphBuilder`] made from nothing, which has no sequence.
    pub fn length(&self) -> u32 {
        self.n
    }

    /// Every tensor, by id: the graph inputs and the nodes' outputs, in creation order.
    pub fn tensors(&self) -> &[Tensor] {
        &self.tensors
    }

    /// The tensor `id`.
    pub fn tensor(&self, id: TensorId) -> Option<&Tensor> {
        self.tensors.get(usize::try_from(id.0).ok()?)
    }

    /// The graph inputs, in creation order.
    pub fn inputs(&self) -> &[Input] {
        &self.inputs
    }

    /// The nodes, by id, in topological order.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// The node `id`.
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(usize::try_from(id.0).ok()?)
    }

    /// Every constant's values, constant after constant in index order. Every constant the Llama
    /// lowering creates is one value, so in a lowered graph entry `c` is constant `c`. A graph a
    /// [`GraphBuilder`] made may hold constants of several values: read one by its index with
    /// [`Graph::constant`].
    pub fn constants(&self) -> &[i64] {
        &self.constant_values
    }

    /// Constant `index`'s values: one, or one per channel, or a gather's indices.
    pub fn constant(&self, index: u32) -> Option<&[i64]> {
        let span = self.constant_spans.get(usize::try_from(index).ok()?)?;
        let start = usize::try_from(span.start).ok()?;
        let end = start.checked_add(usize::try_from(span.len).ok()?)?;
        self.constant_values.get(start..end)
    }

    /// The number of constants.
    pub fn constant_count(&self) -> usize {
        self.constant_spans.len()
    }

    /// The shift-class sets, by index.
    pub fn shift_classes(&self) -> &[ShiftClasses] {
        &self.shift_classes
    }

    /// The weight list, in creation order: each weight and its tensor.
    pub fn weights(&self) -> &[(WeightRef, TensorId)] {
        &self.weights
    }

    /// The graph's own tables, by id: the lowering's, RoPE's and then each `Exp` and `Silu` table
    /// in order of first use; then those a [`GraphBuilder`] declared, in order of declaration.
    /// Gadget selection's tables follow them and are not here.
    pub fn tables(&self) -> &[LoweredTable] {
        &self.tables
    }

    /// The table `id`, one of the graph's own.
    pub fn table(&self, id: TableId) -> Option<&LoweredTable> {
        self.tables.get(usize::try_from(id.0).ok()?)
    }

    /// The checkpoints `c_0, …, c_L`: the residual after the embedding and after each layer
    /// (RFC-0037 'Layers and kinds'). A graph a [`GraphBuilder`] made has none unless declared.
    pub fn checkpoints(&self) -> &[TensorId] {
        &self.checkpoints
    }

    /// The route of a residual site, `AttnResidual` or `MlpResidual`; `None` for another site,
    /// and for a graph a [`GraphBuilder`] made from nothing.
    pub fn route(&self, site: Site) -> Option<Route> {
        let routes = self.routes.as_ref()?;
        match site {
            Site::AttnResidual => routes.first().copied(),
            Site::MlpResidual => routes.get(1).copied(),
            _ => None,
        }
    }

    /// The first node, in node order, of an operation the Llama lowering never creates: `Sub`,
    /// `ConstAdd`, `Sum`, `Slice`, `GatherPublic`, `Bitwise`, `BitDecompose`, `BitFn`,
    /// `AssertEqual`, a `ConstMul` whose constant is not one value, or a `TableMap` into a table
    /// other than `Silu`. The checks after the lowering prove the lowering's operations only,
    /// and each refuses such a node until #150 gives it that operation's relations. A node past
    /// `u32` ids, which no graph holds, is named as `u32::MAX`.
    pub(crate) fn first_non_llama(&self) -> Option<NodeId> {
        let index = self
            .nodes
            .iter()
            .position(|node| !self.is_llama(node.op()))?;
        Some(NodeId(u32::try_from(index).unwrap_or(u32::MAX)))
    }

    /// Whether `op` is one of the Llama lowering's operations, with the parameters it gives
    /// them (RFC-0044).
    fn is_llama(&self, op: &Op) -> bool {
        match *op {
            Op::OneHot { .. }
            | Op::MatMulWeight { .. }
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
            | Op::SoftmaxRow { .. } => true,
            Op::ConstMul { c } => self.constant(c).is_some_and(|values| values.len() == 1),
            Op::TableMap { table } => self
                .table(table)
                .is_some_and(|t| matches!(t.kind, TableKind::Silu { .. })),
            Op::Sub
            | Op::ConstAdd { .. }
            | Op::Sum { .. }
            | Op::Slice { .. }
            | Op::GatherPublic { .. }
            | Op::Bitwise { .. }
            | Op::BitDecompose { .. }
            | Op::BitFn { .. }
            | Op::AssertEqual => false,
        }
    }
}

/// Edits for unit tests that break a rule the lowering keeps, such as the schedule's template.
#[cfg(test)]
impl Graph {
    /// Replaces node `node`'s operation.
    pub(crate) fn set_op(&mut self, node: NodeId, op: Op) {
        self.nodes[node.0 as usize].op = op;
    }

    /// Replaces node `node`'s input `index` with `tensor`.
    pub(crate) fn set_input(&mut self, node: NodeId, index: usize, tensor: TensorId) {
        let node = &mut self.nodes[node.0 as usize];
        let mut inputs = node.inputs.as_slice().to_vec();
        inputs[index] = tensor;
        node.inputs = Ids::of(&inputs).unwrap();
    }

    /// Replaces tensor `tensor`'s shape.
    pub(crate) fn set_shape(&mut self, tensor: TensorId, dims: &[u32]) {
        self.tensors[tensor.0 as usize].shape = Shape::of(dims).unwrap();
    }

    /// Replaces tensor `tensor`'s kind.
    pub(crate) fn set_kind(&mut self, tensor: TensorId, kind: Kind) {
        self.tensors[tensor.0 as usize].kind = kind;
    }

    /// Replaces constant `index`, of one value.
    pub(crate) fn set_constant(&mut self, index: u32, value: i64) {
        let span = self.constant_spans[index as usize];
        assert_eq!(span.len, 1, "a constant of one value");
        self.constant_values[span.start as usize] = value;
    }
}

/// Why a public part could not be lowered. A public part that [`check_public`] accepted always
/// lowers, given the memory and a length in range.
///
/// [`check_public`]: crate::validation::check_public
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LowerError {
    /// The length is outside `1` to `max_context`: the caller's.
    Length {
        /// The length asked for.
        n: u32,
    },
    /// Memory for the graph could not be reserved.
    Memory,
    /// A step that the public part's checks make infallible failed: a bug of this crate. An
    /// `Exp` search that finds no width is one, since RFC-0007 step 2 admits only score scales
    /// whose width exists.
    Internal,
}

/// Lowers `public`'s claim list for a sequence of `n` tokens, `1 ≤ n ≤ max_context`, in the
/// order of the module documentation (RFC-0044).
pub fn lower(public: &PublicPart<'_>, n: u32) -> Result<Graph, LowerError> {
    let arch = public.arch();
    if n == 0 || n > arch.max_context {
        return Err(LowerError::Length { n });
    }
    let routes = [
        route(public, Site::AttnResidual),
        route(public, Site::MlpResidual),
    ];
    let mut lowering = Lowering {
        public,
        n,
        d: arch.d_model,
        h: arch.n_heads,
        kv: arch.n_kv_heads,
        dh: arch.d_head,
        ff: arch.d_ff,
        vocab: arch.vocab,
        graph: Graph {
            n,
            tensors: Vec::new(),
            inputs: Vec::new(),
            nodes: Vec::new(),
            constant_values: Vec::new(),
            constant_spans: Vec::new(),
            shift_classes: Vec::new(),
            weights: Vec::new(),
            tables: Vec::new(),
            checkpoints: Vec::new(),
            routes: Some(routes),
        },
        known: HashMap::new(),
        positions: TensorId(0),
    };
    lowering.build()?;
    Ok(lowering.graph)
}

/// A residual site's route: the fused key when every layer's site has one shift of at most 62,
/// as a `pot` site of one class or an affine site; two steps otherwise.
fn route(public: &PublicPart<'_>, site: Site) -> Route {
    let fused = (0..public.layers()).all(|layer| match public.rescale(layer, site) {
        Some(Rescale::Affine(r)) => r <= 62,
        Some(Rescale::Pot(&[r])) => r <= 62,
        _ => false,
    });
    if fused {
        Route::FusedKey
    } else {
        Route::TwoSteps
    }
}

/// A table definition the lowering may meet again, keyed without allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Definition {
    Exp {
        key_bits: u8,
        sigma: u8,
        scale: Rational,
    },
    Silu {
        b: u8,
        shifts: ShiftClasses,
        in_exp: i8,
        out_exp: i8,
    },
}

/// What `key` gives: the key, its shifts and its class vector.
type Key = (TensorId, ShiftClasses, Option<TensorId>);

/// The lowering's state: the graph so far, the table definitions met, and the dimensions.
struct Lowering<'p, 'a> {
    public: &'p PublicPart<'a>,
    n: u32,
    d: u32,
    h: u32,
    kv: u32,
    dh: u32,
    ff: u32,
    vocab: u32,
    graph: Graph,
    known: HashMap<Definition, TableId>,
    positions: TensorId,
}

impl Lowering<'_, '_> {
    fn build(&mut self) -> Result<(), LowerError> {
        let (n, d, vocab) = (self.n, self.d, self.vocab);
        // The graph inputs, then RoPE's table, which is table 0.
        let last = |count: u32| i64::from(count).checked_sub(1).ok_or(LowerError::Internal);
        let tokens = self.input(&[n], Kind::Advice, Role::Tokens, 0, last(vocab)?)?;
        self.positions = self.input(&[n], Kind::Public, Role::Positions, 0, last(n)?)?;
        self.rope_table()?;
        // The prologue.
        let one_hot = self.node1(Op::OneHot { vocab }, &[tokens], &[n, vocab], Kind::Advice)?;
        let embedded = self.matmul(WeightType::Embed, 0, one_hot, d, vocab)?;
        let s_e = self.public.embed_shift();
        let c = self.constant(pow2(s_e)?)?;
        let mut res = self.node1(Op::ConstMul { c }, &[embedded], &[n, d], Kind::Advice)?;
        self.checkpoint(res)?;
        for layer in 0..self.public.layers() {
            res = self.layer(layer, res)?;
            self.checkpoint(res)?;
        }
        // The epilogue: the final norm and the LM head, whose output is the logits.
        let y = self.norm(res, QuantPoint::FinalNorm)?;
        self.matmul(WeightType::LmHead, 0, y, vocab, d)?;
        Ok(())
    }

    /// RFC-0044 'A layer', from the residual `res`; returns the layer's output, `c_{ℓ+1}`.
    fn layer(&mut self, layer: u32, res: TensorId) -> Result<TensorId, LowerError> {
        let (n, d, h, kv, dh, ff) = (self.n, self.d, self.h, self.kv, self.dh, self.ff);
        let point = |site| QuantPoint::Layer { layer, site };
        let q_width = h.checked_mul(dh).ok_or(LowerError::Internal)?;
        let kv_width = kv.checked_mul(dh).ok_or(LowerError::Internal)?;
        let servable = self.public.profile() == Profile::Servable;

        // 1. The attention norm.
        let y = self.norm(res, point(Site::AttnNorm))?;
        // 2. The projections, all reading `y`.
        let q = self.matmul(WeightType::Q, layer, y, q_width, d)?;
        let k = self.matmul(WeightType::K, layer, y, kv_width, d)?;
        let v = self.matmul(WeightType::V, layer, y, kv_width, d)?;
        // 3. RoPE.
        let q = self.rope(q, h)?;
        let k = self.rope(k, kv)?;
        // 4. The rescales.
        let q = self.rescale(q, point(Site::RopeQ), q_width)?;
        let k = self.rescale(k, point(Site::RopeK), kv_width)?;
        let v = self.rescale(v, point(Site::V), kv_width)?;
        // 5. The attention.
        let q = self.heads(q, h)?;
        let k = self.heads(k, kv)?;
        let v = self.heads(v, kv)?;
        let scores = self.node1(
            Op::MatMulAct {
                transpose_rhs: true,
            },
            &[q, k],
            &[h, n, n],
            Kind::Virtual,
        )?;
        let exp_table = self.exp_table(layer)?;
        let sigma = self.public.quant().exp_sigma;
        let softmax = self.node(
            Op::SoftmaxRow { exp_table, sigma },
            &[scores],
            &[
                (Shape::of(&[h, n, n])?, Kind::Lookup { table: exp_table }),
                (Shape::of(&[h, n])?, Kind::Advice),
            ],
        )?;
        let (e, reciprocals) = match softmax.as_slice() {
            &[e, r] => (e, r),
            _ => return Err(LowerError::Internal),
        };
        let pv = self.node1(
            Op::MatMulAct {
                transpose_rhs: false,
            },
            &[e, v],
            &[h, n, dh],
            Kind::Virtual,
        )?;
        let scaled = self.node1(
            Op::Mul {
                broadcast: Broadcast::Trailing,
            },
            &[pv, reciprocals],
            &[h, n, dh],
            Kind::Virtual,
        )?;
        let swapped = self.node1(
            Op::Transpose {
                perm: swap_outer()?,
            },
            &[scaled],
            &[n, h, dh],
            Kind::Virtual,
        )?;
        let mut o = self.node1(Op::Reshape, &[swapped], &[n, q_width], Kind::Virtual)?;
        // 6. `servable` rescales the attention output.
        if servable {
            o = self.rescale(o, point(Site::AttnOut), q_width)?;
        }
        // 7. The O projection and the residual.
        let acc = self.matmul(WeightType::O, layer, o, d, q_width)?;
        let res = self.residual(acc, res, layer, Site::AttnResidual)?;
        // 8. The MLP norm.
        let y = self.norm(res, point(Site::MlpNorm))?;
        // 9. The gate and up projections.
        let gate = self.matmul(WeightType::Gate, layer, y, ff, d)?;
        let up = self.matmul(WeightType::Up, layer, y, ff, d)?;
        // 10. SiLU: its table, then the gate's classes when the site has several.
        let shifts = match self.public.rescale(layer, Site::Silu) {
            Some(Rescale::Pot(shifts)) => shifts,
            _ => return Err(LowerError::Internal),
        };
        let table = self.silu_table(layer, shifts)?;
        let kind = Kind::Lookup { table };
        let a = if shifts.len() >= 2 {
            let classes = self.input(
                &[ff],
                Kind::Weight,
                Role::Classes(WeightRef {
                    ty: WeightType::Gate,
                    layer,
                }),
                0,
                class_bound(shifts.len())?,
            )?;
            self.node1(Op::TableMap { table }, &[gate, classes], &[n, ff], kind)?
        } else {
            self.node1(Op::TableMap { table }, &[gate], &[n, ff], kind)?
        };
        // 11. The product.
        let product = if servable {
            let u = self.rescale(up, point(Site::Up), ff)?;
            let product = self.node1(
                Op::Mul {
                    broadcast: Broadcast::Trailing,
                },
                &[a, u],
                &[n, ff],
                Kind::Virtual,
            )?;
            self.rescale(product, point(Site::Product), ff)?
        } else {
            self.node1(
                Op::Mul {
                    broadcast: Broadcast::Trailing,
                },
                &[a, up],
                &[n, ff],
                Kind::Virtual,
            )?
        };
        // 12. The down projection and the residual: the layer's output.
        let acc = self.matmul(WeightType::Down, layer, product, d, ff)?;
        self.residual(acc, res, layer, Site::MlpResidual)
    }

    // ----- Procedures --------------------------------------------------------------------------

    /// `matmul(T, ℓ, x, rows, cols)`.
    fn matmul(
        &mut self,
        ty: WeightType,
        layer: u32,
        x: TensorId,
        rows: u32,
        cols: u32,
    ) -> Result<TensorId, LowerError> {
        let weight = WeightRef { ty, layer };
        let w = self.input(&[rows, cols], Kind::Weight, Role::Weight(weight), -127, 127)?;
        push(&mut self.graph.weights, (weight, w))?;
        self.node1(
            Op::MatMulWeight { weight },
            &[x, w],
            &[self.n, rows],
            Kind::Virtual,
        )
    }

    /// `key(x, p, c)`: what the site `p` rescales of `x`, its shifts and its class vector.
    fn key(&mut self, x: TensorId, point: QuantPoint, c: u32) -> Result<Key, LowerError> {
        let rescale = match point {
            QuantPoint::Layer { layer, site } => self.public.rescale(layer, site),
            QuantPoint::FinalNorm => Some(self.public.final_norm()),
            QuantPoint::Embed => None,
        }
        .ok_or(LowerError::Internal)?;
        match rescale {
            Rescale::Affine(r) => {
                let weight = weight_before(point)?;
                let m = self.input(&[c], Kind::Weight, Role::Multipliers(weight), 0, 65_535)?;
                let z = self.node1(
                    Op::Mul {
                        broadcast: Broadcast::Leading,
                    },
                    &[x, m],
                    &[self.n, c],
                    Kind::Virtual,
                )?;
                Ok((z, ShiftClasses::of(&[r])?, None))
            }
            Rescale::Pot(shifts) if shifts.len() == 1 => Ok((x, ShiftClasses::of(shifts)?, None)),
            Rescale::Pot(shifts) => {
                let weight = weight_before(point)?;
                let classes = self.input(
                    &[c],
                    Kind::Weight,
                    Role::Classes(weight),
                    0,
                    class_bound(shifts.len())?,
                )?;
                Ok((x, ShiftClasses::of(shifts)?, Some(classes)))
            }
        }
    }

    /// `requant(z, shifts, κ, c, lo, hi)`.
    fn requant(
        &mut self,
        (z, shifts, classes): Key,
        c: u32,
        lo: i64,
        hi: i64,
    ) -> Result<TensorId, LowerError> {
        let shift = self.shift_set(shifts)?;
        let op = Op::Requant { shift, lo, hi };
        let shape = [self.n, c];
        match classes {
            Some(classes) => self.node1(op, &[z, classes], &shape, Kind::Advice),
            None => self.node1(op, &[z], &shape, Kind::Advice),
        }
    }

    /// `rescale(x, p, c)`: the site's key, requantized to INT8.
    fn rescale(&mut self, x: TensorId, point: QuantPoint, c: u32) -> Result<TensorId, LowerError> {
        let key = self.key(x, point, c)?;
        self.requant(key, c, -128, 127)
    }

    /// `norm(res, p)`.
    fn norm(&mut self, res: TensorId, point: QuantPoint) -> Result<TensorId, LowerError> {
        let (n, d) = (self.n, self.d);
        let quant = self.public.quant();
        let squares = self.node1(Op::SumSquares { axis: 1 }, &[res], &[n], Kind::Virtual)?;
        let root = self.node1(
            Op::RsqrtRow {
                kappa: quant.rsqrt_kappa,
                eps_int: quant.rms_eps_int,
                dim: d,
            },
            &[squares],
            &[n],
            Kind::Advice,
        )?;
        let scaled = self.node1(
            Op::Mul {
                broadcast: Broadcast::Trailing,
            },
            &[res, root],
            &[n, d],
            Kind::Virtual,
        )?;
        self.rescale(scaled, point, d)
    }

    /// `rope(x, k)`, for `k` heads.
    fn rope(&mut self, x: TensorId, k: u32) -> Result<TensorId, LowerError> {
        let (n, dh) = (self.n, self.dh);
        let width = k.checked_mul(dh).ok_or(LowerError::Internal)?;
        let heads = self.node1(Op::Reshape, &[x], &[n, k, dh], Kind::Virtual)?;
        let positions = self.positions;
        let rotated = self.node1(
            Op::Rotate { rope: TableId(0) },
            &[heads, positions],
            &[n, k, dh],
            Kind::Virtual,
        )?;
        self.node1(Op::Reshape, &[rotated], &[n, width], Kind::Virtual)
    }

    /// `heads(x, k)`, for `k` heads: head-major, then expanded to the query heads.
    fn heads(&mut self, x: TensorId, k: u32) -> Result<TensorId, LowerError> {
        let (n, h, dh) = (self.n, self.h, self.dh);
        let split = self.node1(Op::Reshape, &[x], &[n, k, dh], Kind::Virtual)?;
        let head_major = self.node1(
            Op::Transpose {
                perm: swap_outer()?,
            },
            &[split],
            &[k, n, dh],
            Kind::Virtual,
        )?;
        if k == h {
            return Ok(head_major);
        }
        let g = h.checked_div(k).ok_or(LowerError::Internal)?;
        let expanded = self.node1(
            Op::Expand { axis: 1, len: g },
            &[head_major],
            &[k, g, n, dh],
            Kind::Virtual,
        )?;
        self.node1(Op::Reshape, &[expanded], &[h, n, dh], Kind::Virtual)
    }

    /// `residual(acc, res, p)`, by the site's route.
    fn residual(
        &mut self,
        acc: TensorId,
        res: TensorId,
        layer: u32,
        site: Site,
    ) -> Result<TensorId, LowerError> {
        let (n, d) = (self.n, self.d);
        let route = self.graph.route(site).ok_or(LowerError::Internal)?;
        let key = self.key(acc, QuantPoint::Layer { layer, site }, d)?;
        // The residual's range, `[−2^{b−1}, 2^{b−1} − 1]`, and the update's, `±(2^b − 1)`.
        let bits = u32::from(self.public.quant().residual_bits);
        let half = pow2(bits.checked_sub(1).ok_or(LowerError::Internal)?)?;
        let lo = half.checked_neg().ok_or(LowerError::Internal)?;
        let hi = half.checked_sub(1).ok_or(LowerError::Internal)?;
        let (shift, sum) = match route {
            Route::FusedKey => {
                let (z, shifts, _) = key;
                let r = match shifts.as_slice() {
                    &[r] => r,
                    _ => return Err(LowerError::Internal),
                };
                let c = self.constant(pow2(u32::from(r))?)?;
                let scaled = self.node1(Op::ConstMul { c }, &[res], &[n, d], Kind::Virtual)?;
                let sum = self.node1(Op::Add, &[z, scaled], &[n, d], Kind::Virtual)?;
                // A new set `[r]`: each mention creates an entry.
                (self.shift_set(shifts)?, sum)
            }
            Route::TwoSteps => {
                let bound = pow2(bits)?.checked_sub(1).ok_or(LowerError::Internal)?;
                let floor = bound.checked_neg().ok_or(LowerError::Internal)?;
                let update = self.requant(key, d, floor, bound)?;
                let sum = self.node1(Op::Add, &[res, update], &[n, d], Kind::Virtual)?;
                let c = self.constant(2)?;
                let doubled = self.node1(Op::ConstMul { c }, &[sum], &[n, d], Kind::Virtual)?;
                (self.shift_set(ShiftClasses::of(&[1])?)?, doubled)
            }
        };
        self.node1(Op::Requant { shift, lo, hi }, &[sum], &[n, d], Kind::Advice)
    }

    // ----- Tables ------------------------------------------------------------------------------

    /// Table 0: `Rope` with the architecture's `RopeSpec`, `max_context` and `d_head`.
    fn rope_table(&mut self) -> Result<(), LowerError> {
        let arch = self.public.arch();
        let rope = arch.rope;
        let d_head = u16::try_from(arch.d_head).map_err(|_| LowerError::Internal)?;
        let pairs = u64::from(arch.max_context)
            .checked_mul(u64::from(arch.d_head))
            .and_then(|count| count.checked_div(2))
            .ok_or(LowerError::Internal)?;
        let kind = TableKind::Rope {
            theta: rope.theta,
            factor: rope.factor,
            low_freq_factor: rope.low_freq_factor,
            high_freq_factor: rope.high_freq_factor,
            original_max: rope.original_max,
            bits: rope.bits,
            max_context: arch.max_context,
            d_head,
        };
        self.new_table(kind, ceil_log2(pairs)?).map(|_| ())
    }

    /// The layer's `Exp` table: the narrowest width from the spread's up whose last key maps to
    /// 0, or an earlier table of that width's definition (RFC-0044 'The tables'). RFC-0007 step 2
    /// guarantees the width exists, so the search always ends.
    fn exp_table(&mut self, layer: u32) -> Result<TableId, LowerError> {
        let scale = self.public.score_scale(layer).ok_or(LowerError::Internal)?;
        let sigma = self.public.quant().exp_sigma;
        let from = spread_bits(self.dh).ok_or(LowerError::Internal)?;
        for key_bits in from..=32 {
            let definition = Definition::Exp {
                key_bits,
                sigma,
                scale,
            };
            if let Some(&id) = self.known.get(&definition) {
                return Ok(id);
            }
            let last = 1u64
                .checked_shl(u32::from(key_bits))
                .and_then(|rows| rows.checked_sub(1))
                .ok_or(LowerError::Internal)?;
            // Validation has made every score scale positive, so the test cannot fail.
            if exp_maps_to_zero(last, sigma, scale).map_err(|_| LowerError::Internal)? {
                let kind = TableKind::Exp {
                    key_bits,
                    sigma,
                    scale,
                };
                let id = self.new_table(kind, key_bits)?;
                self.remember(definition, id)?;
                return Ok(id);
            }
        }
        // RFC-0007 step 2 refused every score scale that no width serves.
        Err(LowerError::Internal)
    }

    /// The layer's `Silu` table, or an earlier table of the same definition.
    fn silu_table(&mut self, layer: u32, shifts: &[i8]) -> Result<TableId, LowerError> {
        let gate_bound = 127u64
            .checked_mul(128)
            .and_then(|bound| bound.checked_mul(u64::from(self.d)))
            .ok_or(LowerError::Internal)?;
        let b = bitlen(gate_bound)
            .checked_add(1)
            .ok_or(LowerError::Internal)?;
        let in_exp = self
            .public
            .silu_in_exponent(layer)
            .ok_or(LowerError::Internal)?;
        let out_exp = self
            .public
            .exponent(layer, Site::Silu)
            .ok_or(LowerError::Internal)?;
        let set = ShiftClasses::of(shifts)?;
        let definition = Definition::Silu {
            b,
            shifts: set,
            in_exp,
            out_exp,
        };
        if let Some(&id) = self.known.get(&definition) {
            return Ok(id);
        }
        let mut r_by_class = Vec::new();
        r_by_class
            .try_reserve_exact(set.as_slice().len())
            .map_err(|_| LowerError::Memory)?;
        r_by_class.extend_from_slice(set.as_slice());
        let classes = u64::try_from(r_by_class.len()).map_err(|_| LowerError::Internal)?;
        let log2_rows = b
            .checked_add(ceil_log2(classes)?)
            .ok_or(LowerError::Internal)?;
        let kind = TableKind::Silu {
            b,
            r_by_class,
            in_exp,
            out_exp,
        };
        let id = self.new_table(kind, log2_rows)?;
        self.remember(definition, id)?;
        Ok(id)
    }

    /// Appends a table under the next id.
    fn new_table(&mut self, kind: TableKind, log2_rows: u8) -> Result<TableId, LowerError> {
        let id = TableId(next_id(self.graph.tables.len())?);
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

    /// Records that `definition` is table `id`, for later layers.
    fn remember(&mut self, definition: Definition, id: TableId) -> Result<(), LowerError> {
        self.known.try_reserve(1).map_err(|_| LowerError::Memory)?;
        self.known.insert(definition, id);
        Ok(())
    }

    // ----- The graph ---------------------------------------------------------------------------

    /// A new tensor, with the next id.
    fn tensor(
        &mut self,
        shape: Shape,
        kind: Kind,
        producer: Option<NodeId>,
    ) -> Result<TensorId, LowerError> {
        let id = TensorId(next_id(self.graph.tensors.len())?);
        push(
            &mut self.graph.tensors,
            Tensor {
                shape,
                kind,
                producer,
            },
        )?;
        Ok(id)
    }

    /// A graph input of values in `[lo, hi]`: a weight, of type `i8`, or an unsigned input of the
    /// narrowest type that holds `hi`, of at least one bit.
    fn input(
        &mut self,
        dims: &[u32],
        kind: Kind,
        role: Role,
        lo: i64,
        hi: i64,
    ) -> Result<TensorId, LowerError> {
        let tensor = self.tensor(Shape::of(dims)?, kind, None)?;
        let ty = match role {
            Role::Weight(_) => IntType {
                bits: 8,
                signed: true,
            },
            _ => {
                let hi = u64::try_from(hi).map_err(|_| LowerError::Internal)?;
                IntType {
                    bits: bitlen(hi).max(1),
                    signed: false,
                }
            }
        };
        push(
            &mut self.graph.inputs,
            Input {
                tensor,
                role,
                ty,
                lo,
                hi,
            },
        )?;
        Ok(tensor)
    }

    /// A node of `op` over `inputs`, whose outputs are created with it, in order.
    fn node(
        &mut self,
        op: Op,
        inputs: &[TensorId],
        outputs: &[(Shape, Kind)],
    ) -> Result<Ids, LowerError> {
        let node = NodeId(next_id(self.graph.nodes.len())?);
        let mut created = [TensorId(0); 2];
        let slots = created
            .get_mut(..outputs.len())
            .ok_or(LowerError::Internal)?;
        for (slot, &(shape, kind)) in slots.iter_mut().zip(outputs) {
            *slot = self.tensor(shape, kind, Some(node))?;
        }
        let outputs = Ids::of(created.get(..outputs.len()).ok_or(LowerError::Internal)?)?;
        push(
            &mut self.graph.nodes,
            Node {
                op,
                inputs: Ids::of(inputs)?,
                outputs,
            },
        )?;
        Ok(outputs)
    }

    /// A node of one output, `[dims] kind`.
    fn node1(
        &mut self,
        op: Op,
        inputs: &[TensorId],
        dims: &[u32],
        kind: Kind,
    ) -> Result<TensorId, LowerError> {
        let outputs = self.node(op, inputs, &[(Shape::of(dims)?, kind)])?;
        outputs
            .as_slice()
            .first()
            .copied()
            .ok_or(LowerError::Internal)
    }

    /// A new constant of one value; its index.
    fn constant(&mut self, value: i64) -> Result<u32, LowerError> {
        let index = next_id(self.graph.constant_spans.len())?;
        let start = next_id(self.graph.constant_values.len())?;
        push(&mut self.graph.constant_values, value)?;
        push(&mut self.graph.constant_spans, Span { start, len: 1 })?;
        Ok(index)
    }

    /// A new shift-class set; its index.
    fn shift_set(&mut self, set: ShiftClasses) -> Result<u32, LowerError> {
        let index = next_id(self.graph.shift_classes.len())?;
        push(&mut self.graph.shift_classes, set)?;
        Ok(index)
    }

    /// Records a checkpoint.
    fn checkpoint(&mut self, tensor: TensorId) -> Result<(), LowerError> {
        push(&mut self.graph.checkpoints, tensor)
    }
}

/// The weight before a site (RFC-0044 'Notation'): the weight whose channels a classed or affine
/// site's vector follows.
fn weight_before(point: QuantPoint) -> Result<WeightRef, LowerError> {
    match point {
        QuantPoint::Layer { layer, site } => site
            .weight_before()
            .map(|ty| WeightRef { ty, layer })
            .ok_or(LowerError::Internal),
        QuantPoint::Embed | QuantPoint::FinalNorm => Err(LowerError::Internal),
    }
}

/// `[1, 0, 2]`: the permutation that swaps a rank-3 tensor's two outer axes, the heads' layouts'.
fn swap_outer() -> Result<Perm, LowerError> {
    Perm::new(&[1, 0, 2]).ok_or(LowerError::Internal)
}

/// The greatest class index of a site of `classes` classes.
fn class_bound(classes: usize) -> Result<i64, LowerError> {
    i64::try_from(classes)
        .ok()
        .and_then(|k| k.checked_sub(1))
        .ok_or(LowerError::Internal)
}

/// `2^e` for `e ≤ 62`.
fn pow2(e: u32) -> Result<i64, LowerError> {
    1i64.checked_shl(e)
        .filter(|&v| v > 0)
        .ok_or(LowerError::Internal)
}

/// `bitlen(128 · 255 · d_head)`: the width of the scores' spread, where the `Exp` search starts
/// (RFC-0044 'The tables', RFC-0030 'Range analysis and validation').
pub(crate) fn spread_bits(d_head: u32) -> Option<u8> {
    128u64
        .checked_mul(255)?
        .checked_mul(u64::from(d_head))
        .map(bitlen)
}

/// The number of bits of `x`: `bitlen(1) = 1`, `bitlen(0) = 0`.
fn bitlen(x: u64) -> u8 {
    // At most 64, so the conversion is exact.
    u8::try_from(u64::BITS.saturating_sub(x.leading_zeros())).unwrap_or(u8::MAX)
}

/// `⌈log2 x⌉` for `x ≥ 1`: the least `k` with `2^k ≥ x`.
fn ceil_log2(x: u64) -> Result<u8, LowerError> {
    let below = x.checked_sub(1).ok_or(LowerError::Internal)?;
    Ok(bitlen(below))
}

/// The id after `len` entries.
fn next_id(len: usize) -> Result<u32, LowerError> {
    u32::try_from(len).map_err(|_| LowerError::Memory)
}

/// Pushes `value`, its memory reserved fallibly.
fn push<T>(values: &mut Vec<T>, value: T) -> Result<(), LowerError> {
    values.try_reserve(1).map_err(|_| LowerError::Memory)?;
    values.push(value);
    Ok(())
}
