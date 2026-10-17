// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The one error type of the crate (04 §1). An error names tensors, tables and rules, all public,
//! and never holds a value the engine computes: those are secret-class (05 §3).

use core::fmt;

use tessera_codec::error::ErrorCode;
use tessera_ir::{TableId, TensorId};
use tessera_tables::TableError;

/// Why the engine cannot compute a witness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WitnessError {
    /// The request is malformed, such as a sequence of another length than the lowering's or a
    /// token outside the vocabulary (`ERR-USAGE`).
    Usage {
        /// What the request breaks.
        rule: &'static str,
    },
    /// The model does not fit the lowering: a weight or a channel-class vector it lacks, of
    /// another size, or a class its site does not have (`ERR-QUANT-ARCH`).
    Model {
        /// What the model breaks.
        rule: &'static str,
    },
    /// A table the lowering names does not regenerate from its specification (`ERR-INTERNAL`).
    Table {
        /// The table.
        table: TableId,
        /// Why its generator refuses it.
        error: TableError,
    },
    /// A computed value lies outside the range analysis gave its tensor, which INV-DM-2 rules
    /// out: a bug in the analysis or the engine (`ERR-INTERNAL`). Checked in debug builds.
    Range {
        /// The tensor.
        tensor: TensorId,
    },
    /// A lookup's key is not in its table, or another invariant of the lowered graph fails
    /// (`ERR-INTERNAL`).
    Internal {
        /// The invariant.
        rule: &'static str,
    },
    /// A tensor's memory cannot be allocated (`ERR-RES-MEMORY`).
    Memory {
        /// The tensor.
        tensor: TensorId,
    },
}

impl WitnessError {
    /// The stable error code (04 §3).
    pub fn code(&self) -> ErrorCode {
        match self {
            Self::Usage { .. } => ErrorCode::Usage,
            Self::Model { .. } => ErrorCode::QuantArch,
            Self::Table { .. } | Self::Range { .. } | Self::Internal { .. } => ErrorCode::Internal,
            Self::Memory { .. } => ErrorCode::ResMemory,
        }
    }
}

impl fmt::Display for WitnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage { rule } => write!(f, "the request breaks a rule: {rule}"),
            Self::Model { rule } => write!(f, "the model does not fit the lowering: {rule}"),
            Self::Internal { rule } => write!(f, "an invariant fails: {rule}"),
            Self::Table { table, error } => {
                write!(f, "table {} does not regenerate: {error}", table.0)
            }
            Self::Range { tensor } => write!(f, "tensor {} leaves its analyzed range", tensor.0),
            Self::Memory { tensor } => write!(f, "tensor {} cannot be allocated", tensor.0),
        }
    }
}

impl std::error::Error for WitnessError {}
