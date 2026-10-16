// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The one error type of the crate. Every input-dependent condition is one of its variants, never
//! a panic (04 §1).

use core::fmt;

use crate::MAX_DEGREE;

/// Why a table, a point or a round polynomial was refused.
///
/// The variants carry lengths, degrees and the public bound of [`less_than`](crate::less_than),
/// never a field element, so an error cannot carry a secret table value (05 §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolyError {
    /// A table's length is not a power of two (zero included).
    TableLength {
        /// The length given.
        len: usize,
    },
    /// A table of one entry has no variable left to bind.
    NoVariable,
    /// A point has the wrong number of coordinates: a table of `2^n` entries takes `n`, and
    /// two points compared coordinate by coordinate take the same number.
    PointLength {
        /// The number of coordinates needed.
        expected: usize,
        /// The number given.
        found: usize,
    },
    /// `LT(r, a)` is defined for `0 ≤ a ≤ 2^n`, where `n` is the number of coordinates of `r`.
    BoundTooLarge {
        /// The bound `a` given.
        bound: u64,
        /// The number of coordinates `n`.
        vars: usize,
    },
    /// A table of `2^vars` entries cannot be allocated.
    TooLarge {
        /// The number of variables of the table.
        vars: usize,
    },
    /// A round polynomial's degree is outside `1..=MAX_DEGREE`.
    Degree {
        /// The degree given: the number of evaluations sent.
        degree: usize,
    },
}

impl fmt::Display for PolyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TableLength { len } => {
                write!(
                    f,
                    "a multilinear table needs a power-of-two length, not {len}"
                )
            }
            Self::NoVariable => write!(f, "a table of one entry has no variable to bind"),
            Self::PointLength { expected, found } => {
                write!(f, "a point needs {expected} coordinates here, not {found}")
            }
            Self::BoundTooLarge { bound, vars } => {
                write!(f, "LT(r, a) needs a ≤ 2^{vars}, not a = {bound}")
            }
            Self::TooLarge { vars } => {
                write!(f, "a table of 2^{vars} entries cannot be allocated")
            }
            Self::Degree { degree } => {
                write!(
                    f,
                    "a round polynomial has degree 1 to {MAX_DEGREE}, not {degree}"
                )
            }
        }
    }
}

impl std::error::Error for PolyError {}
