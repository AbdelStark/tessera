// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The one error type of the crate (04 §1). A table's parameters are public, so its errors may
//! name them; a lookup's key is a witness value and never reaches an error (05 §3).

use core::fmt;

/// Why a table could not be built from its kind, or a claim list's table differs from the one
/// its kind generates (RFC-0008 'Table specifications').
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableError {
    /// A parameter outside what RFC-0008 allows for the kind.
    Parameter {
        /// The kind, as RFC-0008 names it.
        kind: &'static str,
        /// The rule the parameters break.
        rule: &'static str,
    },
    /// The specification's `log2_rows` is not `⌈log2 T⌉`.
    Log2Rows {
        /// What the specification states.
        stated: u8,
        /// What the kind generates.
        expected: u8,
    },
    /// The specification's digest is not the one its kind generates.
    Digest,
}

/// For example, `a Requant table needs 1 ≤ r_c ≤ 63 for every class`.
impl fmt::Display for TableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parameter { kind, rule } => write!(f, "a {kind} table needs {rule}"),
            Self::Log2Rows { stated, expected } => {
                write!(
                    f,
                    "log2_rows is {stated}, but the kind's rows need {expected}"
                )
            }
            Self::Digest => f.write_str("the digest is not the one the table's kind generates"),
        }
    }
}

impl std::error::Error for TableError {}
