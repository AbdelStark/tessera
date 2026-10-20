// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The one error type of the crate. Every input-dependent condition is one of its variants, never
//! a panic (04 §1).

use core::fmt;

use crate::WeightMatrix;

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
    /// of `n = 2^r·k` points at rate `1/2^r`.
    TooLong {
        /// `log2` of the number of points asked for.
        log_len: u32,
    },
    /// A code of rate 1, `n = k`: an encoding needs `log2` of its inverse rate to be at least 1,
    /// so that a codeword holds more symbols than its message and the masked polynomial, of degree
    /// below `k + P ≤ 2k`, fits it.
    InverseRate {
        /// The `log2` of the inverse rate given.
        log2_inv_rate: u32,
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
    /// A codeword buffer does not hold `n = 2^r·k` symbols per row.
    CodewordLength {
        /// The number of symbols needed.
        expected: usize,
        /// The number of symbols given.
        found: usize,
    },
    /// More mask coefficients than values in a row: the encoding needs `P ≤ k`, so that the
    /// masked polynomial has degree below `2k ≤ n` and the codeword determines it.
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
    /// Mask coefficients for a commitment that masks no row: a weight matrix in
    /// `tessera-protocol/1` (RFC-0040 'Encoding, leaves and roots').
    UnexpectedMask {
        /// The number of mask coefficients `P` given.
        mask_coeffs: usize,
    },
    /// A commitment of one chain per column whose layout does not have exactly one segment: a
    /// weight matrix has no segments and no `column-merge` (RFC-0040).
    Segments {
        /// The number of segments given.
        segments: usize,
    },
    /// A row index of more than 64 bits: a row is a `u64`.
    RowVars {
        /// The bit count given.
        row_vars: u32,
    },
    /// A layout with no segment. A matrix has at least one segment, which may be empty.
    NoSegments,
    /// A streaming segment index outside the canonical layout.
    SegmentIndex {
        /// The index given.
        segment: u32,
        /// The layout's segment count.
        segments: u32,
    },
    /// A merge with a different number of parts from its layout.
    SegmentCount {
        /// The layout's segment count.
        expected: usize,
        /// The number of parts given.
        found: usize,
    },
    /// A merge's part does not have the index of its position.
    SegmentOrder {
        /// Its expected position.
        position: usize,
        /// The index of the part given.
        segment: u32,
    },
    /// A part belongs to different commitment parameters or a different public layout.
    SegmentContext {
        /// The part's index.
        segment: u32,
    },
    /// A push or finish does not supply the segment's next non-zero row (INV-PCS-2).
    StreamRow {
        /// The segment.
        segment: u32,
        /// The next row, or none when the segment is complete.
        expected: Option<u64>,
        /// The row given, or none for a premature finish.
        found: Option<u64>,
    },
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
    /// The proximity point `ξ*` does not have the coordinates its commitment takes: the advice's
    /// `row_vars`, every bit of the row index (RFC-0011 step 3), and a weight matrix's
    /// `⌈log2 m_T⌉`, the bits of a committed row's position (RFC-0049).
    XiPoint {
        /// The coordinates it takes.
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
    /// A weight type with no output channel or no input channel: its matrix would hold no
    /// entry (RFC-0040 'The matrices').
    EmptyWeight {
        /// The weight type.
        matrix: WeightMatrix,
    },
    /// A weight tensor, or a weight's channel parameters, that the layout has no place for:
    /// the channel parameters' matrix, a layer the type does not have (only layer 0 for
    /// `Embed` and `LmHead`, `n_layers` layers for the others), or channel parameters for a
    /// weight the shapes do not list.
    WeightPlace {
        /// The weight type.
        matrix: WeightMatrix,
        /// The layer.
        layer: u32,
        /// Whether the input is a weight's channel parameters rather than its tensor.
        channels: bool,
    },
    /// A second weight tensor, or a second set of channel parameters, for one weight.
    WeightTwice {
        /// The weight type.
        matrix: WeightMatrix,
        /// The layer.
        layer: u32,
        /// Whether the input is a weight's channel parameters rather than its tensor.
        channels: bool,
    },
    /// No weight tensor, or no channel parameters, for a weight the shapes have: the first in
    /// `(type, layer)` order.
    WeightMissing {
        /// The weight type.
        matrix: WeightMatrix,
        /// The layer.
        layer: u32,
        /// Whether the input is a weight's channel parameters rather than its tensor.
        channels: bool,
    },
    /// A weight tensor that does not hold `out · in` entries, or channel parameters that do not
    /// hold one value per output channel.
    WeightLength {
        /// The weight type.
        matrix: WeightMatrix,
        /// The layer.
        layer: u32,
        /// Whether the input is a weight's channel parameters rather than its tensor.
        channels: bool,
        /// The entries or values its shape gives it.
        expected: usize,
        /// The entries or values given.
        found: usize,
    },
    /// A matrix whose committed rows hold more values than a buffer can.
    TooLarge {
        /// The committed rows.
        rows: u64,
        /// The values in a row.
        row_len: usize,
    },
    /// An opened column that is not above the one before it: a budget's opened columns are a
    /// set, strictly increasing (RFC-0040 'WeightCommitmentState').
    ColumnOrder {
        /// The column's position in the list.
        position: usize,
    },
    /// A fold of a matrix with no committed row: it has no claim, no combined rows and no
    /// proximity point (RFC-0049), as `ChannelParams` when no weight has channel parameters.
    NoRows {
        /// The matrix.
        matrix: WeightMatrix,
    },
    /// A fold of `ChannelParams` with no claim (RFC-0052).
    NoClaims,
    /// A fold point that does not have the coordinates the matrix's fold takes: `log2 out_pad`
    /// for a weight type, `r_o`, and the row index's bits for `ChannelParams` (RFC-0012
    /// 'Opening by the folding pass').
    FoldPoint {
        /// The matrix.
        matrix: WeightMatrix,
        /// The coordinates it takes.
        expected: usize,
        /// The coordinates given.
        found: usize,
    },
    /// A layer point `ρ_ℓ` that does not have the matrix's `log2 L_pad` coordinates.
    LayerPoint {
        /// The matrix.
        matrix: WeightMatrix,
        /// The coordinates it takes.
        expected: usize,
        /// The coordinates given.
        found: usize,
    },
    /// The rows of one matrix given to the fold of another.
    OtherMatrix {
        /// The matrix the fold opens.
        expected: WeightMatrix,
        /// The matrix whose rows were given.
        found: WeightMatrix,
    },
    /// A committed row asked for past the matrix's `m_T`.
    RowIndex {
        /// The matrix.
        matrix: WeightMatrix,
        /// The position asked for.
        index: usize,
        /// `m_T`, its committed rows.
        rows: u64,
    },
    /// No advice layout derives from the claim list: RFC-0007 step 6 refuses it, which the
    /// verifier reports as `Reject(StatementMismatch { field: Layout })` (RFC-0039).
    Layout {
        /// The rule the claim list breaks.
        rule: &'static str,
    },
    /// The data a layout is derived from breaks a rule of its form: a model without a layer, a
    /// sequence of no position or past the context, a layer without a tile. The caller derives
    /// it from the claim list and the batch, so this is the caller's error.
    LayoutData {
        /// The rule the data breaks.
        rule: &'static str,
    },
    /// A tile given to the layout breaks a rule of its form.
    Tile {
        /// The tile, counted from 0 in production order.
        tile: usize,
        /// The rule it breaks.
        rule: &'static str,
    },
    /// A claim's point does not have one coordinate per variable of its tensor's domain.
    ClaimPoint {
        /// The tensor's variables `n_X`.
        expected: usize,
        /// The coordinates given.
        found: usize,
    },
    /// An element's coordinates are not one per axis of its tensor.
    ElementAxes {
        /// The tensor's axes.
        expected: usize,
        /// The coordinates given.
        found: usize,
    },
    /// An element's coordinate lies past its axis's padded length.
    ElementRange {
        /// The axis, counted from 0, outermost first.
        axis: usize,
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
            Self::InverseRate { log2_inv_rate } => write!(
                f,
                "a code's inverse rate is 2^1 or more, not 2^{log2_inv_rate}"
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
            Self::UnexpectedMask { mask_coeffs } => write!(
                f,
                "an unmasked commitment takes 0 mask coefficients, not {mask_coeffs}"
            ),
            Self::Segments { segments } => write!(
                f,
                "a commitment of one chain per column has 1 segment, not {segments}"
            ),
            Self::RowVars { row_vars } => {
                write!(f, "a row index has at most 64 bits, not {row_vars}")
            }
            Self::NoSegments => write!(f, "a layout has at least 1 segment, not 0"),
            Self::SegmentIndex { segment, segments } => {
                write!(f, "segment {segment} is outside {segments} segments")
            }
            Self::SegmentCount { expected, found } => {
                write!(f, "a merge needs {expected} segments, not {found}")
            }
            Self::SegmentOrder { position, segment } => write!(
                f,
                "merge position {position} needs segment {position}, not {segment}"
            ),
            Self::SegmentContext { segment } => write!(
                f,
                "segment {segment} has another commitment's parameters or layout"
            ),
            Self::StreamRow {
                segment,
                expected,
                found,
            } => match (expected, found) {
                (Some(expected), Some(found)) => write!(
                    f,
                    "segment {segment} requires row {expected} next, not {found}"
                ),
                (Some(expected), None) => {
                    write!(f, "segment {segment} ended before row {expected}")
                }
                (None, Some(found)) => write!(
                    f,
                    "segment {segment} is complete, so cannot accept row {found}"
                ),
                (None, None) => write!(f, "segment {segment} has no next row"),
            },
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
            Self::EmptyWeight { matrix } => write!(
                f,
                "the {matrix:?} weight type has at least 1 output and 1 input channel"
            ),
            Self::WeightPlace {
                matrix,
                layer,
                channels,
            } => write!(
                f,
                "the layout has no place for {}",
                what(*matrix, *layer, *channels)
            ),
            Self::WeightTwice {
                matrix,
                layer,
                channels,
            } => write!(f, "{} is given twice", what(*matrix, *layer, *channels)),
            Self::WeightMissing {
                matrix,
                layer,
                channels,
            } => write!(f, "{} is missing", what(*matrix, *layer, *channels)),
            Self::WeightLength {
                matrix,
                layer,
                channels,
                expected,
                found,
            } => write!(
                f,
                "{} holds {expected} values, not {found}",
                what(*matrix, *layer, *channels)
            ),
            Self::TooLarge { rows, row_len } => write!(
                f,
                "{rows} committed rows of {row_len} values each do not fit in memory"
            ),
            Self::ColumnOrder { position } => {
                write!(f, "opened column {position} is not above the one before it")
            }
            Self::NoRows { matrix } => write!(
                f,
                "the {matrix:?} matrix has no committed row: it has no combined rows and no proximity point"
            ),
            Self::NoClaims => write!(
                f,
                "the fold of the ChannelParams matrix needs at least 1 claim, not 0"
            ),
            Self::FoldPoint {
                matrix,
                expected,
                found,
            } => write!(
                f,
                "the fold of the {matrix:?} matrix takes a point of {expected} coordinates, not {found}"
            ),
            Self::LayerPoint {
                matrix,
                expected,
                found,
            } => write!(
                f,
                "the {matrix:?} matrix's layer point has {expected} coordinates, not {found}"
            ),
            Self::OtherMatrix { expected, found } => write!(
                f,
                "the fold of the {expected:?} matrix cannot read the rows of the {found:?} matrix"
            ),
            Self::RowIndex {
                matrix,
                index,
                rows,
            } => write!(
                f,
                "the {matrix:?} matrix has {rows} committed rows, so no row {index}"
            ),
            Self::Layout { rule } => {
                write!(f, "the claim list has no advice layout: {rule}")
            }
            Self::LayoutData { rule } => write!(f, "the layout's data breaks a rule: {rule}"),
            Self::Tile { tile, rule } => write!(f, "tile {tile} breaks a rule: {rule}"),
            Self::ClaimPoint { expected, found } => write!(
                f,
                "a claim on a tensor of {expected} variables has {expected} coordinates, not {found}"
            ),
            Self::ElementAxes { expected, found } => write!(
                f,
                "an element of a tensor of {expected} axes has {expected} coordinates, not {found}"
            ),
            Self::ElementRange { axis } => write!(
                f,
                "an element's coordinate on axis {axis} lies past the axis's padded length"
            ),
        }
    }
}

/// `the Q weight of layer 3`, or `the channel parameters of the Q weight of layer 3`.
fn what(matrix: WeightMatrix, layer: u32, channels: bool) -> String {
    let weight = format!("the {matrix:?} weight of layer {layer}");
    match channels {
        true => format!("the channel parameters of {weight}"),
        false => weight,
    }
}

impl std::error::Error for PcsError {}
