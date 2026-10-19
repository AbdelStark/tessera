// SPDX-License-Identifier: Apache-2.0 OR MIT

//! What a proof's transcript schedule depends on (RFC-0014 'Transcript schedule'): the leaf
//! stage's trees, the reduction stages, the layer point, the commitments' proximity points and
//! the opened matrices. Every size is public: the claim list, the batch and the parameter set
//! give it (INV-SCH-1, INV-PROTO-5).

use crate::{MatrixCodeword, ParamSet, ProtocolError, QueryShape, WeightMatrix};

/// The most variables of any instance, tree level or point: its table of `2^n` entries is
/// indexed by a `u64`.
pub(crate) const MAX_VARS: u32 = 64;

/// The most layer variables: `λ = ⌈log2 L⌉` for a `u32` count of layers.
const MAX_LAYER_VARS: u32 = 32;

/// The longest codeword a `u32` column index reaches (RFC-0036 '`ColumnQueries`').
const MAX_CODEWORD: u64 = 1 << 32;

/// A tree of the leaf stage (RFC-0010 'Tree levels', RFC-0008 'Fraction tree').
///
/// The trees come in the levels' schedule order: each LogUp instance's lookup tree, then its
/// table tree, the instances in ascending `TableId` (RFC-0045 'The check'), beside the product
/// forests. The order fixes the instances' numbering within a level and the layout of
/// `tree/{ℓ}/evals`; the labels depend only on the heights and the forests' sizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LeafTree {
    /// A LogUp fraction tree of `2^height` leaves: a lookup tree, `height = ⌈log2 N⌉`, or a table
    /// tree, `height = ⌈log2 T⌉`. A tree of height 0 joins no level: its root is its leaf.
    Fraction {
        /// `h`.
        height: u32,
    },
    /// A product forest of `2^trees_log` trees of `2^height` leaves each, `height ≥ 1`; a single
    /// tree when `trees_log = 0`.
    Forest {
        /// `s`.
        trees_log: u32,
        /// `h`.
        height: u32,
    },
}

impl LeafTree {
    /// Its height `h`: it joins the batch at level `H − h`, with `H` the tallest tree's.
    pub const fn height(self) -> u32 {
        match self {
            Self::Fraction { height } | Self::Forest { height, .. } => height,
        }
    }

    /// `s`, the variables of its root level: 0 for a fraction tree and a single product tree.
    pub const fn trees_log(self) -> u32 {
        match self {
            Self::Fraction { .. } => 0,
            Self::Forest { trees_log, .. } => trees_log,
        }
    }
}

/// A reduction stage (RFC-0010 'Batching within a stage', RFC-0037 'Stages and their
/// instances').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StageSizes {
    /// `n_max`, its rounds: the most variables among its instances.
    pub rounds: u32,
    /// The coordinates of its zero-check point, the most variables among its `ZeroCheck`
    /// instances: 0 for a stage that holds none (RFC-0035 'Zero-check points').
    pub zero_vars: u32,
}

/// A weight matrix with committed rows, and the coordinates of its proximity point `ξ*_T`:
/// `⌈log2 m_T⌉` for its `m_T` committed rows in `tessera-protocol/1` (RFC-0049 'The proximity
/// row', RFC-0040 'The matrices').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatrixPoint {
    /// The matrix.
    pub matrix: WeightMatrix,
    /// The coordinates of `ξ*_T`.
    pub vars: u32,
}

/// The sizes one proof unit's transcript schedule is expanded from.
///
/// The stages and the layer point come from the stage schedule (RFC-0037), the trees from the
/// leaf layout (RFC-0045) and the product forests (RFC-0010), the points and codewords from the
/// commitments' layouts (RFC-0011, RFC-0040). The column queries' counts and the advice
/// codeword's length come from the parameter set.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ScheduleSizes {
    /// The leaf stage's trees, in the levels' schedule order.
    pub trees: Vec<LeafTree>,
    /// `λ = ⌈log2 L⌉`: the coordinates of stage 1's layer point (RFC-0037 'The layer point').
    pub layer_vars: u32,
    /// The reduction stages, `stages[i]` being stage `i + 1`.
    pub stages: Vec<StageSizes>,
    /// `ϑ_A`, the coordinates of the advice's proximity point `ξ*`: the bits of its row index,
    /// `ν − log2 k` (RFC-0011 'Opening protocol' step 3).
    pub advice_vars: u32,
    /// The weight matrices with committed rows, in RFC-0040's matrix order, strictly increasing,
    /// with their proximity points' coordinates.
    pub proximity: Vec<MatrixPoint>,
    /// The matrices the proof opens, those with claims, in `weight_combined`'s order: ids
    /// strictly increasing, each with its codeword's length `n_T`.
    pub opened: Vec<MatrixCodeword>,
}

impl ScheduleSizes {
    /// The column queries' shape these sizes give at `params` (RFC-0036 '`ColumnQueries`'): `τ_A`
    /// indices below `n_A = 2^{log2 k + log2 inv_rate}`, then `τ_W` below each opened matrix's
    /// `n_T`.
    pub fn query_shape(&self, params: &ParamSet) -> QueryShape {
        let advice = params.layout.advice;
        let log_n = u32::from(advice.log2_k) + u32::from(advice.log2_inv_rate);
        QueryShape {
            advice_count: params.advice_queries,
            advice_len: 1u64.checked_shl(log_n).unwrap_or(0),
            matrix_count: params.weight_queries,
            matrices: self.opened.clone(),
        }
    }

    /// Checks that a schedule can be expanded from these sizes at `params`; their source, the
    /// party's own derivation from public data, is at fault otherwise (`ERR-INTERNAL`).
    pub(crate) fn validate(&self, params: &ParamSet) -> Result<(), ProtocolError> {
        for tree in &self.trees {
            let fits = match *tree {
                LeafTree::Fraction { height } => height <= MAX_VARS,
                LeafTree::Forest { trees_log, height } => {
                    height >= 1 && trees_log.checked_add(height).is_some_and(|v| v <= MAX_VARS)
                }
            };
            if !fits {
                return internal(
                    "a leaf tree has at most 2^64 leaves, and a product forest a height of 1 at least",
                );
            }
        }
        if self.layer_vars > MAX_LAYER_VARS {
            return internal("the layer point has at most 32 coordinates");
        }
        if u16::try_from(self.stages.len()).is_err() {
            return internal("the reduction stages are numbered by a u16");
        }
        for stage in &self.stages {
            if stage.rounds > MAX_VARS || stage.zero_vars > stage.rounds {
                return internal(
                    "a stage has at most 64 rounds, and its zero-check point no more coordinates",
                );
            }
        }
        if self.advice_vars > MAX_VARS || self.proximity.iter().any(|p| p.vars > MAX_VARS) {
            return internal("a proximity point has at most 64 coordinates");
        }
        if !self.proximity.windows(2).all(|w| w[0].matrix < w[1].matrix) {
            return internal(
                "the matrices with committed rows come in RFC-0040's order, once each",
            );
        }
        let mut previous = None;
        for codeword in &self.opened {
            let Some(matrix) = WeightMatrix::from_id(codeword.matrix) else {
                return internal("an opened matrix is a weight matrix");
            };
            if previous.is_some_and(|p| matrix <= p) {
                return internal("the opened matrices come in id order, once each");
            }
            previous = Some(matrix);
            if !self.proximity.iter().any(|p| p.matrix == matrix) {
                return internal("an opened matrix has committed rows");
            }
            if !codeword.len.is_power_of_two() || codeword.len > MAX_CODEWORD {
                return internal(
                    "a matrix's codeword holds a power of two of symbols, at most 2^32",
                );
            }
        }
        let advice = params.layout.advice;
        if u32::from(advice.log2_k) + u32::from(advice.log2_inv_rate) > 32 {
            return internal("the advice codeword holds at most 2^32 symbols");
        }
        self.query_shape(params).validate()
    }
}

fn internal(rule: &'static str) -> Result<(), ProtocolError> {
    Err(ProtocolError::Internal { rule })
}
