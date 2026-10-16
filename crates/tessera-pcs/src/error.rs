// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The one error type of the crate. Every input-dependent condition is one of its variants, never
//! a panic (04 §1).

use core::fmt;

/// Why a transform, an encoding, a commitment, an opening or an answer was refused.
///
/// The variants carry lengths, counts, exponents, positions and public indices (a layout's
/// segments, a query), never a field element, so an error cannot carry a row value, a mask
/// coefficient, a codeword symbol or a point (05 §3). Every check runs before anything is written,
/// drawn or computed: on error, the values, the codeword buffer and the mask source are
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
    /// A commitment with no mask coefficient. RFC-0011 'Encoding' masks every row with `P_A`
    /// coefficients, at least the parameter set's `τ_A`, and an opening's verifier refuses
    /// `P_A = 0`.
    NoMask,
    /// A row index of more than 64 bits: a row is a `u64`.
    RowVars {
        /// The bit count given.
        row_vars: u32,
    },
    /// A layout with no segment. A matrix has at least one segment, which may be empty.
    NoSegments,
    /// A row that is not above the row before it in layout order, within its segment or across
    /// segments (INV-PCS-2).
    RowOrder {
        /// The row's segment.
        segment: usize,
        /// The row's position in its segment.
        position: usize,
    },
    /// A row index at or above `2^row_vars`.
    RowRange {
        /// The row's segment.
        segment: usize,
        /// The row's position in its segment.
        position: usize,
        /// The layout's bit count: rows lie below `2^row_vars`.
        row_vars: u32,
    },
    /// The values of the non-zero rows are not `N` rows of `k` values, `N` the layout's count.
    MatrixLength {
        /// The number of non-zero rows `N`.
        rows: usize,
        /// The number of values `k` in a row.
        row_len: usize,
        /// The number of values given.
        found: usize,
    },
    /// A group's column point does not have `log2 k` coordinates.
    ColumnPoint {
        /// The group, in group order.
        group: usize,
        /// `log2 k`.
        expected: usize,
        /// The coordinates given.
        found: usize,
    },
    /// A claim's row point does not have `row_vars` coordinates.
    RowPoint {
        /// The claim's group.
        group: usize,
        /// The claim's position in its group.
        claim: usize,
        /// The layout's `row_vars`.
        expected: usize,
        /// The coordinates given.
        found: usize,
    },
    /// The proximity point `ξ*` does not have `row_vars` coordinates.
    XiPoint {
        /// The layout's `row_vars`.
        expected: usize,
        /// The coordinates given.
        found: usize,
    },
    /// An answer to no column query: a multiproof opens at least one leaf (RFC-0004 'Merkle
    /// trees').
    NoQueries,
    /// A queried column index that is not a column: `n` or more.
    QueryRange {
        /// The query's position in draw order: the first that is out of range.
        position: usize,
        /// The queried index.
        query: u64,
        /// The number of columns `n`.
        columns: usize,
    },
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
            Self::NoMask => write!(
                f,
                "a commitment masks every row with at least 1 mask coefficient, not 0"
            ),
            Self::RowVars { row_vars } => {
                write!(f, "a row index has at most 64 bits, not {row_vars}")
            }
            Self::NoSegments => write!(f, "a layout has at least 1 segment, not 0"),
            Self::RowOrder { segment, position } => write!(
                f,
                "row {position} of segment {segment} is not above the row before it in layout order"
            ),
            Self::RowRange {
                segment,
                position,
                row_vars,
            } => write!(
                f,
                "row {position} of segment {segment} is not below 2^{row_vars}"
            ),
            Self::MatrixLength {
                rows,
                row_len,
                found,
            } => write!(
                f,
                "{rows} non-zero rows of {row_len} values each are not {found} values"
            ),
            Self::ColumnPoint {
                group,
                expected,
                found,
            } => write!(
                f,
                "the column point of group {group} has {expected} coordinates, not {found}"
            ),
            Self::RowPoint {
                group,
                claim,
                expected,
                found,
            } => write!(
                f,
                "the row point of claim {claim} of group {group} has {expected} coordinates, not {found}"
            ),
            Self::XiPoint { expected, found } => {
                write!(
                    f,
                    "the proximity point has {expected} coordinates, not {found}"
                )
            }
            Self::NoQueries => write!(f, "an answer needs at least 1 column query, not 0"),
            Self::QueryRange {
                position,
                query,
                columns,
            } => write!(
                f,
                "query {position} asks for column {query} of a commitment of {columns} columns"
            ),
        }
    }
}

impl std::error::Error for PcsError {}
