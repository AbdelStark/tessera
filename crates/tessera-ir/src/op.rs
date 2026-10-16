// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The closed operation set of RFC-0006 'Operation set'.

use tessera_tables::TableId;

use crate::{ConstRef, PrfRef, ShiftClassRef, WeightRef};

/// An operation of the IR. Each takes a fixed number of input tensors and produces a fixed
/// number of output tensors, under the shape and kind rules of RFC-0006 'Operation set', which
/// [`GraphBuilder::node`](crate::GraphBuilder::node) checks.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    /// `y = W x` over the last axis of `x`: inputs `x` and the weight tensor named by `weight`.
    MatMulWeight {
        /// The weight matrix.
        weight: WeightRef,
    },
    /// `y = a + b`, elementwise.
    Add,
    /// `y = a − b`, elementwise.
    Sub,
    /// `y = c · x`, the constant a scalar or one value per channel of the last axis.
    ConstMul {
        /// The constant.
        c: ConstRef,
    },
    /// `y = x + c`, the constant a scalar or one value per channel of the last axis.
    ConstAdd {
        /// The constant.
        c: ConstRef,
    },
    /// RoPE: each pair of the last axis rotated by public constants of the row's position;
    /// inputs `x` and the public positions of its rows.
    Rotate {
        /// The `Rope` table of the constants.
        rope: TableId,
    },
    /// The sum over one axis.
    Sum {
        /// The axis, counted from the outermost.
        axis: u8,
    },
    /// The same elements in another shape.
    Reshape,
    /// `len` consecutive positions of one axis, from `start`.
    Slice {
        /// The axis, counted from the outermost.
        axis: u8,
        /// The first position kept.
        start: u32,
        /// The number of positions kept.
        len: u32,
    },
    /// The rows of the outermost axis at public indices.
    GatherPublic {
        /// The indices, in output order.
        index: ConstRef,
    },
    /// The product of two activation matrices over their last two axes, the leading axes
    /// batched: `PV`, or `QK^T` with the right operand transposed.
    MatMulAct {
        /// Whether the right operand is `[…, n, k]` and enters transposed, as `K` does in
        /// `QK^T`, rather than `[…, k, n]`.
        transpose_rhs: bool,
    },
    /// The elementwise product. The second operand may be broadcast over trailing axes.
    Mul,
    /// The sum of squares over one axis.
    SumSquares {
        /// The axis, counted from the outermost.
        axis: u8,
    },
    /// The committed one-hot vector of each token.
    OneHot {
        /// The vocabulary size.
        vocab: u32,
    },
    /// `rq(x, r_c, lo, hi)` of RFC-0005 for each element, `c` the output channel's hidden class;
    /// with more than one class, the second input is the channel-class vector.
    Requant {
        /// The public shifts of the classes.
        shift: ShiftClassRef,
        /// The saturation floor.
        lo: i64,
        /// The saturation ceiling.
        hi: i64,
    },
    /// A single-input lookup, elementwise: SiLU and other nonlinearities.
    TableMap {
        /// The table.
        table: TableId,
    },
    /// Softmax over the last axis by the max certificate and the reciprocal bracket (RFC-0009):
    /// outputs the exponentials and each row's reciprocal `R_q`.
    SoftmaxRow {
        /// The `Exp` table.
        exp_table: TableId,
        /// The exponentials' precision `σ`.
        sigma: u8,
    },
    /// `ρ = max{ρ : ρ^2 (u + e) ≤ d · 2^{2κ}}` for each row's sum of squares (RFC-0009).
    RsqrtRow {
        /// The precision `κ`.
        kappa: u8,
        /// The exact epsilon term `e`.
        eps_int: u64,
    },
    /// Certifies that each row's one-hot token is the argmax of its logits (RFC-0017).
    Argmax,
    /// Certifies that each row's one-hot token is the inverse-CDF sample of its logits
    /// (RFC-0017).
    InverseCdf {
        /// The sampler PRF.
        prf: PrfRef,
    },
    /// A byte operation of the anchor circuits by lookup, elementwise (RFC-0015).
    Bitwise {
        /// The table.
        table: TableId,
    },
}

/// How a node's relation enters the proof (RFC-0006, RFC-0010): linear and degree-2 operations
/// produce virtual tensors whose claims reduce through sumchecks; gadget-backed ones produce
/// committed tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpClass {
    /// Linear in its non-constant inputs.
    Linear,
    /// Of degree 2 in its inputs.
    Degree2,
    /// Backed by a gadget of RFC-0009, which commits its outputs and internal advice.
    Gadget,
}

impl Op {
    /// The operation's name, as RFC-0006 writes it.
    pub fn name(&self) -> &'static str {
        match self {
            Self::MatMulWeight { .. } => "MatMulWeight",
            Self::Add => "Add",
            Self::Sub => "Sub",
            Self::ConstMul { .. } => "ConstMul",
            Self::ConstAdd { .. } => "ConstAdd",
            Self::Rotate { .. } => "Rotate",
            Self::Sum { .. } => "Sum",
            Self::Reshape => "Reshape",
            Self::Slice { .. } => "Slice",
            Self::GatherPublic { .. } => "GatherPublic",
            Self::MatMulAct { .. } => "MatMulAct",
            Self::Mul => "Mul",
            Self::SumSquares { .. } => "SumSquares",
            Self::OneHot { .. } => "OneHot",
            Self::Requant { .. } => "Requant",
            Self::TableMap { .. } => "TableMap",
            Self::SoftmaxRow { .. } => "SoftmaxRow",
            Self::RsqrtRow { .. } => "RsqrtRow",
            Self::Argmax => "Argmax",
            Self::InverseCdf { .. } => "InverseCdf",
            Self::Bitwise { .. } => "Bitwise",
        }
    }

    /// Whether the operation is linear, of degree 2, or gadget-backed.
    pub fn class(&self) -> OpClass {
        match self {
            Self::MatMulWeight { .. }
            | Self::Add
            | Self::Sub
            | Self::ConstMul { .. }
            | Self::ConstAdd { .. }
            | Self::Rotate { .. }
            | Self::Sum { .. }
            | Self::Reshape
            | Self::Slice { .. }
            | Self::GatherPublic { .. } => OpClass::Linear,
            Self::MatMulAct { .. } | Self::Mul | Self::SumSquares { .. } => OpClass::Degree2,
            Self::OneHot { .. }
            | Self::Requant { .. }
            | Self::TableMap { .. }
            | Self::SoftmaxRow { .. }
            | Self::RsqrtRow { .. }
            | Self::Argmax
            | Self::InverseCdf { .. }
            | Self::Bitwise { .. } => OpClass::Gadget,
        }
    }
}
