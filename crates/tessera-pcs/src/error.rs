// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The one error type of the crate. Every input-dependent condition is one of its variants, never
//! a panic (04 §1).

use core::fmt;

/// Why a transform or an encoding was refused.
///
/// The variants carry lengths, counts and exponents, never a field element, so an error cannot
/// carry a row value, a mask coefficient or a codeword symbol (05 §3). Every check runs before
/// anything is written or drawn: on error, the values, the codeword buffer and the mask source are
/// untouched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcsError {
    /// A transform length or a message length that is not a power of two (zero included).
    NotPowerOfTwo {
        /// The length given.
        len: usize,
    },
    /// A transform of more than `2^32` points: `Fp^*` has no larger power-of-two subgroup
    /// (`P − 1 = 2^32·(2^32 − 1)`). For a message of `k` values, the transform is the codeword's,
    /// of `2k` points.
    TooLong {
        /// `log2` of the number of points asked for.
        log_len: u32,
    },
    /// A slice handed to a transform does not have the plan's length.
    TransformLength {
        /// The plan's length.
        expected: usize,
        /// The length given.
        found: usize,
    },
    /// A row does not hold the encoder's `k` values.
    RowLength {
        /// The message length `k`.
        expected: usize,
        /// The length given.
        found: usize,
    },
    /// A batch of rows is not a whole number of rows of `k` values.
    PartialRow {
        /// The number of values given.
        len: usize,
        /// The message length `k`.
        row_len: usize,
    },
    /// A codeword buffer does not hold `n = 2k` symbols per row.
    CodewordLength {
        /// The number of symbols needed.
        expected: usize,
        /// The number of symbols given.
        found: usize,
    },
    /// More mask coefficients than values in a row: the encoding needs `P ≤ k`, so that the
    /// masked polynomial has degree below `n = 2k` and the codeword determines it.
    MaskTooLarge {
        /// The number of mask coefficients `P` asked for.
        mask_coeffs: usize,
        /// The message length `k`.
        message_len: usize,
    },
    /// A coset shift of zero: `0·L` is not a coset.
    ZeroShift,
}

impl fmt::Display for PcsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotPowerOfTwo { len } => {
                write!(
                    f,
                    "a transform or row length must be a power of two, not {len}"
                )
            }
            Self::TooLong { log_len } => write!(
                f,
                "a transform of 2^{log_len} points needs a subgroup of order 2^{log_len}; the largest is 2^32"
            ),
            Self::TransformLength { expected, found } => {
                write!(f, "the plan transforms {expected} values, not {found}")
            }
            Self::RowLength { expected, found } => {
                write!(f, "a row holds {expected} values, not {found}")
            }
            Self::PartialRow { len, row_len } => {
                write!(f, "{len} values do not fill rows of {row_len}")
            }
            Self::CodewordLength { expected, found } => {
                write!(
                    f,
                    "the codeword buffer needs {expected} symbols, not {found}"
                )
            }
            Self::MaskTooLarge {
                mask_coeffs,
                message_len,
            } => write!(
                f,
                "a row of {message_len} values takes at most {message_len} mask coefficients, not {mask_coeffs}"
            ),
            Self::ZeroShift => write!(f, "a coset shift must be non-zero"),
        }
    }
}

impl std::error::Error for PcsError {}
