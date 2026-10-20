// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Independent final relations of RFC-0009. A local relation check is not proof
//! acceptance: its returned obligations must be bound to the graph reductions,
//! canonical advice roles and prescribed range lookups before the opening checks.

pub mod comparison;
pub mod decomposition;
pub mod root_products;

use crate::validation::GadgetId;
use core::fmt;

/// A failed gadget relation, or a failure of the verifier itself (04 §2/§3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GadgetError {
    /// `Reject(Gadget { gadget })`: the proof's evaluations do not satisfy the relation.
    Rejected {
        /// The verifier-derived gadget's index, never an identifier from the proof.
        gadget: GadgetId,
    },
    /// `Inconclusive(ERR-RES-MEMORY)`: public metadata could not be reserved.
    Memory,
    /// `Inconclusive(ERR-INTERNAL)`: inconsistent verifier-owned parameters or geometry.
    Internal,
}

impl GadgetError {
    /// The gadget payload of `REJ-GADGET`, or `None` for a verifier failure.
    pub const fn rejected_gadget(self) -> Option<GadgetId> {
        match self {
            Self::Rejected { gadget } => Some(gadget),
            Self::Memory | Self::Internal => None,
        }
    }

    /// The stable identifier of 04 §2/§3.
    pub const fn code(self) -> &'static str {
        match self {
            Self::Rejected { .. } => "REJ-GADGET",
            Self::Memory => "ERR-RES-MEMORY",
            Self::Internal => "ERR-INTERNAL",
        }
    }
}

impl fmt::Display for GadgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Rejected { gadget } => write!(f, "gadget {} failed its final relation", gadget.0),
            Self::Memory => f.write_str("memory for gadget checks ran out"),
            Self::Internal => f.write_str("gadget checks failed internally"),
        }
    }
}
impl std::error::Error for GadgetError {}
