// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Why a table specification was refused, or a table could not be generated or evaluated.

use core::fmt;

/// The first rule a table specification breaks, in the order of the crate documentation, or why
/// a table could not be generated or evaluated.
///
/// [`is_rejection`](Self::is_rejection) says whose fault an error is. A parameter outside
/// RFC-0008 'Table specifications', a wrong `log2_rows` and another digest judge the claim list:
/// `Reject(StatementMismatch)` at its validation (RFC-0007 'Verifier validation' step 4, 04 §2).
/// `NotLookupTable`, `PointLength`, `Memory` and `Internal` judge what the verifier passes in, or
/// the verifier itself: `Inconclusive(ERR-RES-MEMORY)` for `Memory`, `Inconclusive(ERR-INTERNAL)`
/// for the others.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TableError {
    /// `Range`: `bits` outside `1..=32`.
    RangeBits {
        /// The key width.
        bits: u8,
    },
    /// `Requant`, `Silu`: `b = 0`, a key of no bit.
    KeyBits,
    /// `Requant`, `Silu`: the number of classes `n = |r_by_class|` outside `1..=4`.
    Classes {
        /// The number of classes.
        classes: usize,
    },
    /// `Requant`, `Silu`: `b + ⌈log2 n⌉ > 32`, more rows than a key of `B_max = 32` bits packs.
    RowBits {
        /// The key width.
        b: u8,
        /// The number of classes.
        classes: usize,
    },
    /// `Requant`: a class's shift outside `1..=63`.
    RequantShift {
        /// The class, an index into `r_by_class`.
        class: usize,
        /// Its shift.
        shift: u8,
    },
    /// `Requant`: `lo > hi`.
    RequantClamp {
        /// The clamp's lower bound.
        lo: i32,
        /// The clamp's upper bound.
        hi: i32,
    },
    /// `Pow2`: `max_exp` outside `1..=63`, so that some value would leave the centered range or
    /// the table would be empty.
    Pow2Exponent {
        /// The number of rows.
        max_exp: u8,
    },
    /// `Exp`: `key_bits` outside `1..=32`.
    ExpKeyBits {
        /// The key width.
        key_bits: u8,
    },
    /// `Exp`: `σ > 32`.
    ExpSigma {
        /// `σ`.
        sigma: u8,
    },
    /// `Exp`: `scale ≤ 0`.
    ExpScale,
    /// `Exp`: the zero tail starts after key `2^24`, so the dense core would hold more than
    /// `2^24` values.
    ExpTail {
        /// `z`, the first key of the zero tail, or `T` when the table has none.
        tail: u64,
    },
    /// `Exp2Frac`: `frac_bits` outside `1..=16`.
    Exp2FracBits {
        /// The key width.
        frac_bits: u8,
    },
    /// `Exp2Frac`: `out_bits` outside `1..=56`.
    Exp2FracOutBits {
        /// The output's fractional bits.
        out_bits: u8,
    },
    /// `Silu`: a class's shift not below `b`.
    SiluShift {
        /// The class, an index into `r_by_class`.
        class: usize,
        /// Its shift.
        shift: u8,
    },
    /// `Silu`: `Σ_c 2^{b − r_c} > 2^24`, a dense core of more than `2^24` values.
    SiluCore {
        /// `Σ_c 2^{b − r_c}`.
        values: u64,
    },
    /// `Silu`: `in_exp` outside `−64..=0`.
    SiluInExp {
        /// The input's exponent.
        in_exp: i8,
    },
    /// `Silu`: the output shift `63 − in_exp + out_exp` outside `1..=126`.
    SiluOutShift {
        /// `63 − in_exp + out_exp`.
        shift: i16,
    },
    /// `Rope`: `θ < 2`.
    RopeTheta {
        /// The base.
        theta: u32,
    },
    /// `Rope`: `factor = 0`.
    RopeFactor,
    /// `Rope`: not `1 ≤ low_freq_factor < high_freq_factor`.
    RopeFrequencyFactors {
        /// `low_freq_factor`.
        low: u16,
        /// `high_freq_factor`.
        high: u16,
    },
    /// `Rope`: `original_max = 0`.
    RopeOriginalMax,
    /// `Rope`: `bits` outside `1..=30`.
    RopeBits {
        /// The constants' fractional bits.
        bits: u8,
    },
    /// `Rope`: `d_head` odd or 0.
    RopeHeadDim {
        /// The head dimension.
        d_head: u16,
    },
    /// `Rope`: `max_context = 0`.
    RopeContext,
    /// `Rope`: more than `2^24` pairs, `max_context · d_head / 2`.
    RopePairs {
        /// The number of pairs.
        pairs: u64,
    },
    /// The specification's `log2_rows` is not `⌈log2 T⌉` for the kind's `T` rows.
    Log2Rows {
        /// `⌈log2 T⌉`.
        expected: u8,
        /// The specification's.
        found: u8,
    },
    /// The specification's digest is not the regenerated table's.
    Digest,
    /// Structured evaluation of a `Rope` table, which holds constants and is not a lookup table.
    ///
    /// The verifier's fault, under a precondition on the caller: a claim list that looks up a
    /// `Rope` table must already have been refused with `Reject(StatementMismatch)` at
    /// claim-list validation (RFC-0007 'Verifier validation' step 5), before any evaluation.
    NotLookupTable,
    /// An evaluation point of `found` coordinates for a table of `2^expected` padded rows.
    PointLength {
        /// The table's `log2_rows`.
        expected: u8,
        /// The point's coordinates.
        found: usize,
    },
    /// Memory for a table's dense core could not be reserved.
    Memory,
    /// A step that the parameter checks make infallible failed, a bound of RFC-0008
    /// 'Fixed-point generators' among them: a bug of this crate that no input reaches.
    Internal,
}

impl TableError {
    /// Whether the error is a fault of the claim list, a rejection
    /// (`Reject(StatementMismatch)`), rather than of the verifier.
    pub fn is_rejection(&self) -> bool {
        match self {
            TableError::RangeBits { .. }
            | TableError::KeyBits
            | TableError::Classes { .. }
            | TableError::RowBits { .. }
            | TableError::RequantShift { .. }
            | TableError::RequantClamp { .. }
            | TableError::Pow2Exponent { .. }
            | TableError::ExpKeyBits { .. }
            | TableError::ExpSigma { .. }
            | TableError::ExpScale
            | TableError::ExpTail { .. }
            | TableError::Exp2FracBits { .. }
            | TableError::Exp2FracOutBits { .. }
            | TableError::SiluShift { .. }
            | TableError::SiluCore { .. }
            | TableError::SiluInExp { .. }
            | TableError::SiluOutShift { .. }
            | TableError::RopeTheta { .. }
            | TableError::RopeFactor
            | TableError::RopeFrequencyFactors { .. }
            | TableError::RopeOriginalMax
            | TableError::RopeBits { .. }
            | TableError::RopeHeadDim { .. }
            | TableError::RopeContext
            | TableError::RopePairs { .. }
            | TableError::Log2Rows { .. }
            | TableError::Digest => true,
            TableError::NotLookupTable
            | TableError::PointLength { .. }
            | TableError::Memory
            | TableError::Internal => false,
        }
    }
}

impl fmt::Display for TableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TableError::RangeBits { bits } => write!(f, "Range: bits = {bits} is outside 1..=32"),
            TableError::KeyBits => write!(f, "the key width b is 0"),
            TableError::Classes { classes } => {
                write!(f, "{classes} classes, outside 1..=4")
            }
            TableError::RowBits { b, classes } => write!(
                f,
                "b = {b} with {classes} classes packs keys of more than 32 bits"
            ),
            TableError::RequantShift { class, shift } => write!(
                f,
                "Requant: class {class} has the shift {shift}, outside 1..=63"
            ),
            TableError::RequantClamp { lo, hi } => {
                write!(f, "Requant: lo = {lo} is above hi = {hi}")
            }
            TableError::Pow2Exponent { max_exp } => {
                write!(f, "Pow2: max_exp = {max_exp} is outside 1..=63")
            }
            TableError::ExpKeyBits { key_bits } => {
                write!(f, "Exp: key_bits = {key_bits} is outside 1..=32")
            }
            TableError::ExpSigma { sigma } => write!(f, "Exp: σ = {sigma} is above 32"),
            TableError::ExpScale => write!(f, "Exp: the scale is not positive"),
            TableError::ExpTail { tail } => {
                write!(f, "Exp: the zero tail starts at key {tail}, after key 2^24")
            }
            TableError::Exp2FracBits { frac_bits } => {
                write!(f, "Exp2Frac: frac_bits = {frac_bits} is outside 1..=16")
            }
            TableError::Exp2FracOutBits { out_bits } => {
                write!(f, "Exp2Frac: out_bits = {out_bits} is outside 1..=56")
            }
            TableError::SiluShift { class, shift } => {
                write!(f, "Silu: class {class} has the shift {shift}, not below b")
            }
            TableError::SiluCore { values } => write!(
                f,
                "Silu: the dense core holds {values} values, more than 2^24"
            ),
            TableError::SiluInExp { in_exp } => {
                write!(f, "Silu: in_exp = {in_exp} is outside -64..=0")
            }
            TableError::SiluOutShift { shift } => write!(
                f,
                "Silu: the output shift 63 - in_exp + out_exp = {shift} is outside 1..=126"
            ),
            TableError::RopeTheta { theta } => write!(f, "Rope: θ = {theta} is below 2"),
            TableError::RopeFactor => write!(f, "Rope: factor is 0"),
            TableError::RopeFrequencyFactors { low, high } => write!(
                f,
                "Rope: low_freq_factor = {low} and high_freq_factor = {high} break 1 ≤ low < high"
            ),
            TableError::RopeOriginalMax => write!(f, "Rope: original_max is 0"),
            TableError::RopeBits { bits } => write!(f, "Rope: bits = {bits} is outside 1..=30"),
            TableError::RopeHeadDim { d_head } => {
                write!(
                    f,
                    "Rope: d_head = {d_head} is not an even number of at least 2"
                )
            }
            TableError::RopeContext => write!(f, "Rope: max_context is 0"),
            TableError::RopePairs { pairs } => {
                write!(f, "Rope: {pairs} pairs, more than 2^24")
            }
            TableError::Log2Rows { expected, found } => {
                write!(f, "log2_rows is {found} where the rows need {expected}")
            }
            TableError::Digest => write!(f, "the digest is not the regenerated table's"),
            TableError::NotLookupTable => {
                write!(f, "a Rope table is not a lookup table and has no extension")
            }
            TableError::PointLength { expected, found } => write!(
                f,
                "an evaluation point of {found} coordinates for a table of 2^{expected} rows"
            ),
            TableError::Memory => write!(f, "memory for a table's core could not be reserved"),
            TableError::Internal => write!(f, "a table generator broke a bound it guarantees"),
        }
    }
}

impl std::error::Error for TableError {}
