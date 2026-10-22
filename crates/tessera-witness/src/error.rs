// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The one error type of the crate (04 §1). An error names tensors, tables and rules, all public,
//! and never holds a value the engine computes: those are secret-class (05 §3).

use core::fmt;

use tessera_codec::error::ErrorCode;
use tessera_gadgets::GadgetError;
use tessera_ir::{TableId, TensorId};
use tessera_tables::TableError;

/// Why the engine cannot compute a witness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WitnessError {
    /// A recorded output is not its preceding logit row's lowest-index maximum
    /// (`ERR-WITNESS-EXACT-MISMATCH`). No token, value or mismatch vector is disclosed.
    ExactMismatch,
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
    /// A table's multiplicity buffer cannot be allocated (`ERR-RES-MEMORY`).
    MemoryTable {
        /// The table.
        table: TableId,
    },
    /// A gadget of the selection refuses its tables or the values a tile gives it, which the
    /// selection and range analysis rule out (`ERR-INTERNAL`). Like every variant, it carries
    /// rules, identifiers and lengths, never a value.
    Gadget(GadgetError),
}

impl WitnessError {
    /// The stable error code (04 §3).
    pub fn code(&self) -> ErrorCode {
        match self {
            Self::ExactMismatch => ErrorCode::WitnessExactMismatch,
            Self::Usage { .. } => ErrorCode::Usage,
            Self::Model { .. } => ErrorCode::QuantArch,
            Self::Table { .. } | Self::Range { .. } | Self::Internal { .. } | Self::Gadget(_) => {
                ErrorCode::Internal
            }
            Self::Memory { .. } | Self::MemoryTable { .. } => ErrorCode::ResMemory,
        }
    }
}

impl fmt::Display for WitnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExactMismatch => {
                f.write_str("the recorded outputs do not satisfy exact greedy mode")
            }
            Self::Usage { rule } => write!(f, "the request breaks a rule: {rule}"),
            Self::Model { rule } => write!(f, "the model does not fit the lowering: {rule}"),
            Self::Internal { rule } => write!(f, "an invariant fails: {rule}"),
            Self::Table { table, error } => {
                write!(f, "table {} does not regenerate: {error}", table.0)
            }
            Self::Range { tensor } => write!(f, "tensor {} leaves its analyzed range", tensor.0),
            Self::Memory { tensor } => write!(f, "tensor {} cannot be allocated", tensor.0),
            Self::MemoryTable { table } => {
                write!(f, "multiplicities of table {} cannot be allocated", table.0)
            }
            Self::Gadget(error) => write!(f, "a gadget refuses its witness: {error}"),
        }
    }
}

impl std::error::Error for WitnessError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Gadget(error) => Some(error),
            _ => None,
        }
    }
}

impl From<GadgetError> for WitnessError {
    fn from(error: GadgetError) -> Self {
        Self::Gadget(error)
    }
}
