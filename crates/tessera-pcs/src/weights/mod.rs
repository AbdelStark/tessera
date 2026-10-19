// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The persistent weight commitment of RFC-0012 as `tessera-protocol/1` commits it (RFC-0040):
//! the ten matrices' layout, their roots, `R_W` and the two artifacts.
//!
//! - [`WeightLayout`] places every weight and channel parameter of `D` from public data
//!   ([`WeightShapes`]), and gives each matrix's message width `k_T`, row index and committed
//!   rows ([`MatrixLayout`]).
//! - [`WeightCommitment`] commits `D`'s weights ([`ModelWeights`]) on that layout: rate
//!   `1/2^log2_inv_rate`, no mask, one chain per column from the matrix's `commitment_id`, a
//!   Merkle root per matrix, and [`WeightCommitmentPublic`], which `R_W` hashes.
//! - [`WeightCommitmentState`] holds the public part, the masks' seed and the budget, written as
//!   a secret artifact.
//! - [`WeightFold`] opens a matrix by the folding pass (RFC-0012, RFC-0049): one read of its
//!   committed rows ([`MatrixRows`], [`WeightRows`]) gives the sumcheck input and both combined
//!   rows ([`WeightCombinedRows`]).

mod commit;
pub(crate) mod layout;
mod opening;
mod public;
mod state;

pub use commit::WeightCommitment;
pub use layout::{
    ChannelParamsRef, ChannelValues, MatrixKind, MatrixLayout, ModelWeights, WeightLayout,
    WeightRows, WeightShape, WeightShapes, WeightTensorRef,
};
pub use opening::{MatrixRows, WeightCombinedRows, WeightFold};
pub use public::WeightCommitmentPublic;
pub use state::{OpenedColumns, TypeBudget, WeightCommitmentState};
