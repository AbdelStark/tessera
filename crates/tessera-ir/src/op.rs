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
    /// RoPE: inputs `x` and the public positions of its rows. With a last axis of `2m`, the pair
    /// `(a, b) = (x_i, x_{i+m})` becomes `(a·C − b·S, a·S + b·C)`, where `(C, S)` are the constants
    /// of the row's position and frequency `i`: Llama's `rotate_half` layout, the channel order of
    /// Hugging Face checkpoints.
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
    /// The axes in another order: the output's axis `i` is the input's axis `perm[i]`
    /// (RFC-0037 'Head layouts').
    Transpose {
        /// The permutation of the input's axes, outermost first.
        perm: Vec<u8>,
    },
    /// The input repeated along a new axis of `len` positions, inserted before axis `axis`
    /// (RFC-0037 'Head layouts').
    Expand {
        /// The new axis's place, counted from the outermost: from 0 to the input's rank.
        axis: u8,
        /// Its length, at least 1.
        len: u32,
    },
    /// The product of two activation matrices over their last two axes, the leading axes
    /// batched: `PV`, or `QK^T` with the right operand transposed.
    MatMulAct {
        /// Whether the right operand is `[…, n, k]` and enters transposed, as `K` does in
        /// `QK^T`, rather than `[…, k, n]`.
        transpose_rhs: bool,
    },
    /// The elementwise product `a · b`, with `b` broadcast as `broadcast` says.
    Mul {
        /// How `b`'s shape meets `a`'s.
        broadcast: Broadcast,
    },
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
    /// A single-key lookup, elementwise: SiLU and other nonlinearities. With a table of several
    /// classes, the second input is the channel-class vector, as for `Requant`.
    TableMap {
        /// The table.
        table: TableId,
    },
    /// Softmax over the last axis by the max certificate and the reciprocal bracket (RFC-0009):
    /// outputs the exponentials and each row's reciprocal `R_q`. The last two axes are the query
    /// and key positions of one sequence, and row `q` sees keys `i ≤ q`: the exponentials of the
    /// others are zero.
    SoftmaxRow {
        /// The `Exp` table.
        exp_table: TableId,
        /// The exponentials' precision `σ`.
        sigma: u8,
    },
    /// `ρ = max{ρ : ρ^2 (u + e) ≤ d · 2^{2κ}}` for each row's sum of squares `u` (RFC-0009).
    RsqrtRow {
        /// The precision `κ`.
        kappa: u8,
        /// The exact epsilon term `e`.
        eps_int: u64,
        /// The row length `d` the sums ran over.
        dim: u32,
    },
    /// Certifies that each row's one-hot token is the argmax of its logits (RFC-0017).
    Argmax,
    /// Certifies that each row's one-hot token is the inverse-CDF sample of its logits
    /// (RFC-0017).
    InverseCdf {
        /// The sampler PRF.
        prf: PrfRef,
    },
    /// A byte operation of the anchor circuits by lookup, elementwise (RFC-0015): AES's.
    Bitwise {
        /// The table.
        table: TableId,
    },
    /// The bits of a true integer `x ∈ [0, 2^{low + high})`, in two committed outputs: `b`, the
    /// `low` bits of weight `2^0` to `2^{low−1}`, and `h`, the `high` bits above them, none when
    /// `high` is 0. Each output is `x`'s shape with the bit index as a new innermost axis. Its
    /// zero-checks are booleanity and recomposition, which make the bits `x`'s binary digits and
    /// range-check a committed `x` (RFC-0066 Lemma 1).
    BitDecompose {
        /// The result bits, at least 1.
        low: u8,
        /// The bits above them, the carry of a sum; `low + high ≤ 62`.
        high: u8,
    },
    /// A Boolean function of two or three bit tensors, elementwise, as its multilinear polynomial
    /// (RFC-0066): virtual, its range `[0, 1]`.
    BitFn {
        /// The function.
        f: BoolFn,
    },
    /// `a = b`, elementwise: a relation with no output, checked at a zero-check point
    /// (RFC-0066).
    AssertEqual,
}

/// The Boolean functions of [`Op::BitFn`], each the multilinear polynomial that agrees with it on
/// `{0, 1}` (RFC-0066 'BitFn').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BoolFn {
    /// `a ⊕ b = a + b − 2ab`.
    Xor,
    /// `a ⊕ b ⊕ c = a + b + c − 2ab − 2ac − 2bc + 4abc`.
    Xor3,
    /// `a ∧ b = ab`.
    And,
    /// FIPS 180-4's `Ch(e, f, g) = (e ∧ f) ⊕ (¬e ∧ g) = ef + g − eg`.
    Ch,
    /// FIPS 180-4's `Maj(a, b, c) = ab + ac + bc − 2abc`.
    Maj,
}

impl BoolFn {
    /// The number of inputs: 2 for `Xor` and `And`, 3 for the others.
    pub fn arity(self) -> usize {
        match self {
            Self::Xor | Self::And => 2,
            Self::Xor3 | Self::Ch | Self::Maj => 3,
        }
    }

    /// The function's value on its [`arity`](Self::arity) bits, each 0 or 1, in `x`'s first
    /// entries. Callers pass bits: on other integers the polynomial's `i64` arithmetic can
    /// overflow, and a missing input reads as 0.
    pub fn eval(self, x: &[i64]) -> i64 {
        let at = |i: usize| x.get(i).copied().unwrap_or(0);
        let (a, b, c) = (at(0), at(1), at(2));
        match self {
            Self::Xor => a + b - 2 * a * b,
            Self::Xor3 => a + b + c - 2 * a * b - 2 * a * c - 2 * b * c + 4 * a * b * c,
            Self::And => a * b,
            Self::Ch => a * b + c - a * c,
            Self::Maj => a * b + a * c + b * c - 2 * a * b * c,
        }
    }
}

/// How `Mul`'s second operand `b` meets its first `a` (RFC-0006 'Operation set'). Shapes alone
/// cannot say: when `a`'s first and last axes have equal lengths, `b` of that length fits both.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Broadcast {
    /// `b`'s shape is `a`'s, or a prefix of it, broadcast over the trailing axes: one value per
    /// row, such as a norm's root or a softmax row's reciprocal.
    Trailing,
    /// `b`'s shape is `a`'s last axis, broadcast over the leading axes: one value per channel,
    /// such as an affine site's multipliers (RFC-0025).
    Leading,
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
    /// Of degree at most 3 in its inputs: [`Op::BitFn`]'s polynomials.
    Degree3,
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
            Self::Transpose { .. } => "Transpose",
            Self::Expand { .. } => "Expand",
            Self::MatMulAct { .. } => "MatMulAct",
            Self::Mul { .. } => "Mul",
            Self::SumSquares { .. } => "SumSquares",
            Self::OneHot { .. } => "OneHot",
            Self::Requant { .. } => "Requant",
            Self::TableMap { .. } => "TableMap",
            Self::SoftmaxRow { .. } => "SoftmaxRow",
            Self::RsqrtRow { .. } => "RsqrtRow",
            Self::Argmax => "Argmax",
            Self::InverseCdf { .. } => "InverseCdf",
            Self::Bitwise { .. } => "Bitwise",
            Self::BitDecompose { .. } => "BitDecompose",
            Self::BitFn { .. } => "BitFn",
            Self::AssertEqual => "AssertEqual",
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
            | Self::GatherPublic { .. }
            | Self::Transpose { .. }
            | Self::Expand { .. } => OpClass::Linear,
            Self::MatMulAct { .. } | Self::Mul { .. } | Self::SumSquares { .. } => OpClass::Degree2,
            Self::BitFn { .. } => OpClass::Degree3,
            Self::OneHot { .. }
            | Self::Requant { .. }
            | Self::TableMap { .. }
            | Self::SoftmaxRow { .. }
            | Self::RsqrtRow { .. }
            | Self::Argmax
            | Self::InverseCdf { .. }
            | Self::Bitwise { .. }
            | Self::BitDecompose { .. }
            | Self::AssertEqual => OpClass::Gadget,
        }
    }
}
