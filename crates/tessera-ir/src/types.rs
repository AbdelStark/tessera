// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The IR's identifiers and tensor descriptions (03 §1, §3).

use tessera_tables::TableId;

/// A tensor's index in its graph: dense, in creation order (INV-DM-5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(pub u32);

/// A node's index in its graph: dense, in creation order, which is a topological order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

/// A reduction stage of the proof schedule (RFC-0010), assigned when the schedule is derived.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StageId(pub u16);

/// A gadget instance (RFC-0009), assigned at gadget selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GadgetId(pub u32);

/// An integer type: `bits` from 1 to 64, signed or not (03 §1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IntType {
    /// The width in bits.
    pub bits: u8,
    /// Whether the type holds negative values.
    pub signed: bool,
}

impl IntType {
    /// `INT8`, the type of requantized activations and of weights.
    pub const I8: Self = Self {
        bits: 8,
        signed: true,
    };

    /// `INT32`, the type of linear-layer accumulators.
    pub const I32: Self = Self {
        bits: 32,
        signed: true,
    };
}

/// An inclusive integer interval `[lo, hi]`, `lo ≤ hi` (03 §1). Range analysis (RFC-0006
/// 'Range analysis') computes it; it over-approximates every value the tensor can take
/// (INV-DM-2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IntRange {
    /// The least value.
    pub lo: i128,
    /// The greatest value.
    pub hi: i128,
}

/// The weight matrices of `D`, one per type and layer (03 §2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum WeightType {
    /// The transposed embedding.
    Embed,
    /// The query projection.
    Q,
    /// The key projection.
    K,
    /// The value projection.
    V,
    /// The output projection.
    O,
    /// The MLP gate projection.
    Gate,
    /// The MLP up projection.
    Up,
    /// The MLP down projection.
    Down,
    /// The vocabulary head.
    LmHead,
}

/// The weight matrix a `MatMulWeight` multiplies by: its type and layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WeightRef {
    /// The matrix type.
    pub ty: WeightType,
    /// The layer, 0 for `Embed` and `LmHead`.
    pub layer: u32,
}

/// A public integer constant of the graph: a scalar or one value per channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConstRef(pub u32);

/// A public set of shift classes, `r_c` per class (RFC-0007 'Public and hidden quantization
/// data'); which class an output channel takes is hidden.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ShiftClassRef(pub u32);

/// The sampler PRF an `InverseCdf` draws from (RFC-0017).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PrfRef(pub u32);

/// Which commitment holds a committed tensor (03 §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CommitmentClass {
    /// The per-proof advice commitment (RFC-0011).
    Advice,
    /// The persistent weight commitment (RFC-0012).
    Weight,
}

/// How a tensor's values reach the proof (03 §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NodeKind {
    /// Never committed: claims on it reduce through sumchecks.
    Virtual,
    /// Committed in the advice or the weight commitment.
    Committed {
        /// The commitment.
        commitment: CommitmentClass,
    },
    /// The committed value of a lookup in `table`.
    LookupOutput {
        /// The table.
        table: TableId,
    },
    /// Known to the verifier: constants, positions, shapes.
    Public,
}

/// A tensor of the graph (03 §3).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorSpec {
    /// The tensor's index.
    pub id: TensorId,
    /// Its dimensions, outermost first: at least one, each at least 1.
    pub shape: Vec<u32>,
    /// Its element type.
    pub ty: IntType,
    /// The interval its values lie in.
    pub range: IntRange,
    /// How its values reach the proof.
    pub kind: NodeKind,
}

/// A tensor about to be created: everything but its id, which the graph assigns.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorDecl {
    /// Its dimensions, outermost first: at least one, each at least 1.
    pub shape: Vec<u32>,
    /// Its element type.
    pub ty: IntType,
    /// The interval its values lie in.
    pub range: IntRange,
    /// How its values reach the proof.
    pub kind: NodeKind,
}
