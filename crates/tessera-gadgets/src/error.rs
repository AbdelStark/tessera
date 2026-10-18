// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The one error type of the crate (04 §1). A gadget's parameters and tables are public, so an
//! error may name them. Its inputs, classes and committed values are secret-class (05 §3), and an
//! error carries none of them: no value, and not the position of the element that failed.

use core::fmt;

use tessera_codec::error::ErrorCode;
use tessera_tables::{TableError, TableId};

use crate::Committed;

/// A column that a gadget's witness or its LogUp instances read, besides its input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Column {
    /// The hidden class column `c`, one class per element.
    Classes,
    /// A committed tensor of the gadget.
    Tensor(Committed),
    /// The multiplicities, one buffer per LogUp instance.
    Multiplicities,
}

/// `classes`, `tensor ZLo` or `multiplicities`.
impl fmt::Display for Column {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Classes => f.write_str("classes"),
            Self::Tensor(tensor) => write!(f, "tensor {tensor:?}"),
            Self::Multiplicities => f.write_str("multiplicities"),
        }
    }
}

/// Why a gadget, its witness or its LogUp instances could not be built.
///
/// Every variant is `ERR-INTERNAL` (04 §3). The compiler's selection names the gadget and its
/// tables from public data, and range analysis bounds every input it reads (INV-IR-2), so a
/// gadget that does not match its tables, a column of the wrong length or an input outside the
/// gadget's domain is a bug upstream: RFC-0008 'Failure modes' makes an honest key outside its
/// table `ERR-INTERNAL`.
///
/// The variants carry table identifiers, column names, lengths and rules, never a value: an
/// error cannot carry a key, a class or a committed value (05 §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GadgetError {
    /// The gadget is not one this type takes, or its parameters break a precondition of its
    /// lemma (RFC-0009, RFC-0024).
    Shape {
        /// The rule the gadget breaks.
        rule: &'static str,
    },
    /// A table the gadget names is not among the specifications or is not the table the gadget
    /// reads there; or, for its LogUp instances, a table without columns among those given, or
    /// a lookup's key that names no row of it.
    Table {
        /// The table.
        table: TableId,
        /// The rule it breaks.
        rule: &'static str,
    },
    /// A table's specification does not regenerate (RFC-0008 'Table specifications').
    Spec {
        /// The table.
        table: TableId,
        /// Why its generator refuses it.
        error: TableError,
    },
    /// A column the gadget reads was not given.
    Missing {
        /// The column.
        column: Column,
    },
    /// A column has another length than the one its shape fixes: one entry per element of the
    /// input, or one buffer of multiplicities per LogUp instance.
    Length {
        /// The column.
        column: Column,
        /// Its length.
        len: usize,
        /// The length its shape fixes.
        expected: usize,
    },
    /// An input outside the gadget's domain: a key outside its signed width, a class outside
    /// its classes, or a `NonNeg` value outside `[0, 2^k)`. Range analysis rules it out
    /// (INV-IR-2, INV-DM-2).
    Input {
        /// The domain.
        rule: &'static str,
    },
}

impl GadgetError {
    /// The stable error code (04 §3): `ERR-INTERNAL` for every variant.
    pub fn code(&self) -> ErrorCode {
        ErrorCode::Internal
    }
}

/// For example, `table 3: RequantSplit's low part reads Range { s }`.
impl fmt::Display for GadgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape { rule } => write!(f, "the gadget breaks a rule: {rule}"),
            Self::Table { table, rule } => write!(f, "table {}: {rule}", table.0),
            Self::Spec { table, error } => {
                write!(f, "table {} does not regenerate: {error}", table.0)
            }
            Self::Missing { column } => write!(f, "the {column} the gadget reads were not given"),
            Self::Length {
                column,
                len,
                expected,
            } => write!(
                f,
                "the {column} are {len} where the gadget's shape gives {expected}"
            ),
            Self::Input { rule } => write!(f, "an input lies outside the gadget's domain: {rule}"),
        }
    }
}

impl std::error::Error for GadgetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Spec { error, .. } => Some(error),
            _ => None,
        }
    }
}
