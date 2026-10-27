// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Independent weight layouts (RFC-0012, RFC-0040, RFC-0049 and RFC-0052).
//!
//! The caller has validated the claim list. The layout reads only its public architecture,
//! rescaling sites and commitment parameters, never a weight or a prover-declared row list.
//! It retains intervals of live rows rather than expanding a potentially enormous matrix.
//! Structural row indices address the weight polynomial; committed-row positions address
//! proximity coefficients (RFC-0049). They are deliberately separate.
//!
//! # Caller and check order
//!
//! The caller first validates its own claim list and applies its resource limits to decoded proof
//! messages (RFC-0019 steps2–4). [`check_public`] binds all ten roots, the layout and epoch to the
//! application's trusted `R_W`; [`derive()`] reconstructs the layouts and descriptor hashes from
//! public inputs. A supplied root or row map from the proof never replaces these values.
//!
//! In step9, [`check_evaluations`] checks a matrix's unmasked combined rows (`1` evaluation row
//! for `Embed`..`LmHead`, `m ≥ 1` evaluation rows in pool order for `ChannelParams`, followed by
//! the proximity row) and reduces its schedule-derived claims. Its must-use [`Evaluated`] still
//! owes step11: after transcript-derived column queries, [`Evaluated::check_columns`]
//! authenticates direct chain leaves and the canonical multiproof, then checks all re-encoded rows
//! at every distinct queried column. Structural row coordinates bind the evaluation rows; dense
//! committed-row positions bind the proximity row. The smallest failing column across all rows
//! determines the proximity rejection.
//!
//! An empty matrix is public zero: its claims must be zero, with no combined rows, queries or
//! answers. Protocol2 masks/budgets and the full verifier driver (#128) remain separate. No method
//! returns a whole-proof Accept verdict.
//!
//! All retained layouts are public and compact. Messages and points are public/protocol data;
//! nothing is logged. Invalid verifier-derived sizes/points/queries return an inconclusive
//! internal error, failed reservations a memory error; proof faults return the exact evaluation,
//! Merkle or proximity rejection. Resource admission remains the caller's responsibility, as for
//! [`crate::opening`]; this component neither decodes proof bytes nor runs an unbounded search.

use core::{fmt, num::NonZeroU32, ops::Range};

use crate::claims::{ClaimList, QuantPoint, WeightLayoutParams, WeightType};
use crate::opening::{CombinedRow, OpeningError, multilinear};
use tv_codec::WeightCommitmentPublic;
use tv_field::{Fp, Fp2};
use tv_hash::column::column_digest;
use tv_hash::hash;
use tv_hash::merkle::{MultiproofError, verify_multiproof};
use tv_hash::{Digest, Domain, DomainHasher};

/// A matrix in RFC-0012's declaration order, including the hidden channel parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Matrix {
    /// The transposed embedding.
    Embed,
    /// The query projection.
    Q,
    /// The key projection.
    K,
    /// The value projection.
    V,
    /// The attention output projection.
    O,
    /// The MLP gate.
    Gate,
    /// The MLP up projection.
    Up,
    /// The MLP down projection.
    Down,
    /// The language-model head.
    LmHead,
    /// The class indices or affine multipliers.
    ChannelParams,
}

impl Matrix {
    /// Every matrix, in the order of the ten roots in `WeightCommitmentPublic`.
    pub const ALL: [Self; 10] = [
        Self::Embed,
        Self::Q,
        Self::K,
        Self::V,
        Self::O,
        Self::Gate,
        Self::Up,
        Self::Down,
        Self::LmHead,
        Self::ChannelParams,
    ];

    /// The matrix's TCE tag (RFC-0031).
    pub const fn tag(self) -> u16 {
        match self {
            Self::Embed => 0,
            Self::Q => 1,
            Self::K => 2,
            Self::V => 3,
            Self::O => 4,
            Self::Gate => 5,
            Self::Up => 6,
            Self::Down => 7,
            Self::LmHead => 8,
            Self::ChannelParams => 9,
        }
    }

    /// The matrix's canonical name in RFC-0012 and `tessera.soundness/1` reports.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Embed => "Embed",
            Self::Q => "Q",
            Self::K => "K",
            Self::V => "V",
            Self::O => "O",
            Self::Gate => "Gate",
            Self::Up => "Up",
            Self::Down => "Down",
            Self::LmHead => "LmHead",
            Self::ChannelParams => "ChannelParams",
        }
    }

    /// Parses a matrix from its canonical name.
    pub fn by_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.name() == name)
    }

    pub(crate) fn of_weight(ty: WeightType) -> Self {
        match ty {
            WeightType::Embed => Self::Embed,
            WeightType::Q => Self::Q,
            WeightType::K => Self::K,
            WeightType::V => Self::V,
            WeightType::O => Self::O,
            WeightType::Gate => Self::Gate,
            WeightType::Up => Self::Up,
            WeightType::Down => Self::Down,
            WeightType::LmHead => Self::LmHead,
        }
    }
}

/// A failure to derive the verifier's own weight layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutError {
    /// Invalid public sizes or metadata after claim-list validation: `ERR-INTERNAL`.
    Internal,
    /// A public layout buffer could not be reserved: `ERR-RES-MEMORY`.
    Memory,
}

impl LayoutError {
    /// The inconclusive error's stable identifier; neither error rejects a proof.
    pub const fn code(self) -> &'static str {
        match self {
            Self::Internal => "ERR-INTERNAL",
            Self::Memory => "ERR-RES-MEMORY",
        }
    }
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}
impl std::error::Error for LayoutError {}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Rows {
    Layers { live: u64, stride: u64 },
    Intervals(Vec<Range<u64>>),
}

/// One matrix's public layout in the unmasked `tessera-protocol/1` commitment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    matrix: Matrix,
    log2_k: u32,
    row_vars: u32,
    count: u64,
    rows: Rows,
    commitment_id: Digest,
    row_block: NonZeroU32,
}

impl Layout {
    /// The weight matrix.
    pub fn matrix(&self) -> Matrix {
        self.matrix
    }
    /// `log2(k_T)`, the message width, before rate-1/4 encoding.
    pub fn log2_k(&self) -> u32 {
        self.log2_k
    }
    /// The structural polynomial's row-index bit count, including skipped zero rows.
    pub fn row_vars(&self) -> u32 {
        self.row_vars
    }
    /// `m_T`, the number of committed rows, with no masks or blinding rows in protocol1.
    pub fn committed_rows(&self) -> u64 {
        self.count
    }
    /// The number of proximity-point coordinates: `ceil(log2(m_T))`, zero for an empty matrix.
    pub fn proximity_vars(&self) -> u32 {
        u64::BITS.saturating_sub(self.count.saturating_sub(1).leading_zeros())
    }
    /// The hash of RFC-0031's independently encoded matrix descriptor.
    pub fn commitment_id(&self) -> Digest {
        self.commitment_id
    }
    /// Map a committed-row position to its structural index, or `None` past `m_T`.
    pub fn row_at(&self, position: u64) -> Option<u64> {
        if position >= self.count {
            return None;
        }
        match &self.rows {
            Rows::Layers { live, stride } => position
                .checked_div(*live)?
                .checked_mul(*stride)?
                .checked_add(position.checked_rem(*live)?),
            Rows::Intervals(intervals) => {
                let mut rest = position;
                for span in intervals {
                    let len = span.end.checked_sub(span.start)?;
                    if rest < len {
                        return span.start.checked_add(rest);
                    }
                    rest = rest.checked_sub(len)?;
                }
                None
            }
        }
    }
    /// Map a reading site's claim on weight type `ty` at `channel_point` (`log2(out_pad)`
    /// coordinates) and `layer_point` (`log2(L_pad)` coordinates) to its `(column_point,
    /// row_point)` on `ChannelParams` (RFC-0052 'Point on `Ã_CP`').
    ///
    /// # Errors
    /// `Internal` if this layout is not `ChannelParams`, if `ty` has no `ChannelParams` site in
    /// `claims`, or if `channel_point` or `layer_point` has the wrong length; `Memory` if a
    /// point buffer cannot be reserved.
    pub fn channel_point(
        &self,
        claims: &ClaimList,
        ty: WeightType,
        channel_point: &[Fp2],
        layer_point: &[Fp2],
    ) -> Result<(Vec<Fp2>, Vec<Fp2>), Error> {
        if self.matrix != Matrix::ChannelParams {
            return Err(Error::Internal);
        }
        let has_site = claims
            .quant
            .shift_class_sets
            .iter()
            .filter(|(_, shifts)| shifts.len() > 1)
            .map(|(point, _)| point)
            .chain(claims.quant.affine_sites.iter().map(|(point, _)| point))
            .any(|point| {
                matches!(
                    *point,
                    QuantPoint::Layer { site, .. } if site.weight_before() == Some(ty)
                )
            });
        if !has_site {
            return Err(Error::Internal);
        }
        let (out, _, _) = shape(claims, Matrix::of_weight(ty)).map_err(|_| Error::Internal)?;
        let c_i = usize::try_from(padded(out).map_err(|_| Error::Internal)?.trailing_zeros())
            .map_err(|_| Error::Internal)?;
        let log2_l_pad = usize::try_from(
            padded(u64::from(claims.arch.n_layers))
                .map_err(|_| Error::Internal)?
                .trailing_zeros(),
        )
        .map_err(|_| Error::Internal)?;
        let expected_layers = if matches!(ty, WeightType::Embed | WeightType::LmHead) {
            0
        } else {
            log2_l_pad
        };
        if channel_point.len() != c_i || layer_point.len() != expected_layers {
            return Err(Error::Internal);
        }
        let cv = usize::try_from(self.log2_k).map_err(|_| Error::Internal)?;
        let rv = usize::try_from(self.row_vars).map_err(|_| Error::Internal)?;
        let total_vars = cv.checked_add(rv).ok_or(Error::Internal)?;
        let high_vars = log2_l_pad.checked_add(4).ok_or(Error::Internal)?;
        let gamma = total_vars.checked_sub(high_vars).ok_or(Error::Internal)?;
        if c_i > gamma {
            return Err(Error::Internal);
        }
        let mut full = reserve(total_vars)?;
        full.extend_from_slice(channel_point);
        for _ in c_i..gamma {
            full.push(Fp2::ZERO);
        }
        full.extend_from_slice(layer_point);
        for _ in expected_layers..log2_l_pad {
            full.push(Fp2::ZERO);
        }
        let tag = Matrix::of_weight(ty).tag();
        for bit in 0..4u32 {
            full.push(if (tag >> bit) & 1 == 1 {
                Fp2::ONE
            } else {
                Fp2::ZERO
            });
        }
        let (col_slice, row_slice) = full.split_at_checked(cv).ok_or(Error::Internal)?;
        let mut col_pt = reserve(cv)?;
        col_pt.extend_from_slice(col_slice);
        let mut row_pt = reserve(rv)?;
        row_pt.extend_from_slice(row_slice);
        Ok((col_pt, row_pt))
    }
}

fn padded(value: u64) -> Result<u64, LayoutError> {
    if value == 0 {
        return Err(LayoutError::Internal);
    }
    value
        .checked_next_power_of_two()
        .ok_or(LayoutError::Internal)
}

pub(crate) fn shape(claims: &ClaimList, matrix: Matrix) -> Result<(u64, u64, u64), LayoutError> {
    let a = claims.arch;
    let d = u64::from(a.d_model);
    let ff = u64::from(a.d_ff);
    let v = u64::from(a.vocab);
    let q = u64::from(a.n_heads)
        .checked_mul(u64::from(a.d_head))
        .ok_or(LayoutError::Internal)?;
    let kv = u64::from(a.n_kv_heads)
        .checked_mul(u64::from(a.d_head))
        .ok_or(LayoutError::Internal)?;
    let layers = u64::from(a.n_layers);
    match matrix {
        Matrix::Embed => Ok((d, v, 1)),
        Matrix::Q => Ok((q, d, layers)),
        Matrix::K | Matrix::V => Ok((kv, d, layers)),
        Matrix::O => Ok((d, q, layers)),
        Matrix::Gate | Matrix::Up => Ok((ff, d, layers)),
        Matrix::Down => Ok((d, ff, layers)),
        Matrix::LmHead => Ok((v, d, 1)),
        Matrix::ChannelParams => Err(LayoutError::Internal),
    }
}

fn weight(claims: &ClaimList, matrix: Matrix, k_w: u64) -> Result<Layout, LayoutError> {
    let (out, input, layers) = shape(claims, matrix)?;
    let op = padded(out)?;
    let ip = padded(input)?;
    let full = op.checked_mul(ip).ok_or(LayoutError::Internal)?;
    let k = ip.max(k_w.min(full));
    let outputs = k.checked_div(ip).ok_or(LayoutError::Internal)?;
    let stride = op.checked_div(outputs).ok_or(LayoutError::Internal)?;
    let live = out.div_ceil(outputs);
    let capacity = stride
        .checked_mul(padded(layers)?)
        .ok_or(LayoutError::Internal)?;
    make(
        matrix,
        k,
        capacity,
        live.checked_mul(layers).ok_or(LayoutError::Internal)?,
        Rows::Layers { live, stride },
        claims.layout.advice.row_block,
    )
}

fn channel(claims: &ClaimList, k_w: u64) -> Result<Layout, LayoutError> {
    let layers = padded(u64::from(claims.arch.n_layers))?;
    let sites = claims
        .quant
        .shift_class_sets
        .iter()
        .filter(|(_, shifts)| shifts.len() > 1)
        .map(|(point, _)| point)
        .chain(claims.quant.affine_sites.iter().map(|(point, _)| point));
    let mut live = Vec::new();
    let total_sites = claims
        .quant
        .shift_class_sets
        .len()
        .checked_add(claims.quant.affine_sites.len())
        .ok_or(LayoutError::Internal)?;
    live.try_reserve_exact(total_sites)
        .map_err(|_| LayoutError::Memory)?;
    let mut c = 1;
    for point in sites {
        let QuantPoint::Layer { layer, site } = *point else {
            continue;
        };
        let Some(ty) = site.weight_before() else {
            continue;
        };
        let matrix = Matrix::of_weight(ty);
        let (out, _, n) = shape(claims, matrix)?;
        if u64::from(layer) >= n {
            return Err(LayoutError::Internal);
        }
        c = c.max(padded(out)?);
        live.push((matrix.tag(), u64::from(layer), out));
    }
    let total = layers
        .checked_mul(16)
        .and_then(|x| x.checked_mul(c))
        .ok_or(LayoutError::Internal)?;
    let k = k_w.min(total);
    let mut intervals: Vec<Range<u64>> = Vec::new();
    intervals
        .try_reserve_exact(live.len())
        .map_err(|_| LayoutError::Memory)?;
    for (ty, layer, out) in live {
        let first = u64::from(ty)
            .checked_mul(layers)
            .and_then(|x| x.checked_add(layer))
            .and_then(|x| x.checked_mul(c))
            .ok_or(LayoutError::Internal)?;
        let end = first.checked_add(out).ok_or(LayoutError::Internal)?;
        intervals.push(first.checked_div(k).ok_or(LayoutError::Internal)?..end.div_ceil(k));
    }
    intervals.sort_unstable_by_key(|span| span.start);
    let mut merged: Vec<Range<u64>> = Vec::new();
    merged
        .try_reserve_exact(intervals.len())
        .map_err(|_| LayoutError::Memory)?;
    for span in intervals {
        match merged.last_mut() {
            Some(last) if span.start <= last.end => last.end = last.end.max(span.end),
            _ => merged.push(span),
        }
    }
    let count = merged.iter().try_fold(0u64, |sum, span| {
        sum.checked_add(span.end.checked_sub(span.start)?)
    });
    make(
        Matrix::ChannelParams,
        k,
        total.checked_div(k).ok_or(LayoutError::Internal)?,
        count.ok_or(LayoutError::Internal)?,
        Rows::Intervals(merged),
        claims.layout.advice.row_block,
    )
}

fn make(
    matrix: Matrix,
    k: u64,
    capacity: u64,
    count: u64,
    rows: Rows,
    block: u32,
) -> Result<Layout, LayoutError> {
    if !k.is_power_of_two() || k.trailing_zeros() > 30 || !capacity.is_power_of_two() {
        return Err(LayoutError::Internal);
    }
    Ok(Layout {
        matrix,
        log2_k: k.trailing_zeros(),
        row_vars: capacity.trailing_zeros(),
        count,
        rows,
        commitment_id: [0; 32],
        row_block: NonZeroU32::new(block).ok_or(LayoutError::Internal)?,
    })
}

fn descriptor(
    layout: &Layout,
    params: WeightLayoutParams,
    epoch: u32,
) -> Result<Digest, LayoutError> {
    let mut hash = DomainHasher::new(Domain::CommitmentId);
    hash.update(&1u16.to_le_bytes()); // RFC-0031: WeightMatrix is descriptor variant1.
    hash.update(&[params.log2_k, params.log2_inv_rate]);
    hash.update(&params.mask_coeffs.to_le_bytes());
    hash.update(&params.blinding_rows.to_le_bytes());
    hash.update(&layout.matrix.tag().to_le_bytes());
    hash.update(&[u8::try_from(layout.log2_k).map_err(|_| LayoutError::Internal)?]);
    hash.update(&layout.row_block.get().to_le_bytes());
    hash.update(&layout.count.to_le_bytes());
    hash.update(&epoch.to_le_bytes());
    Ok(hash.finalize())
}

/// Derive every matrix's layout and commitment id from an already validated claim list.
///
/// `epoch` comes from the trusted public commitment. No rows are added for the layout's
/// blinding budget in protocol1. The returned array has RFC-0012's matrix order.
///
/// # Errors
/// `Internal` for invalid sizes/metadata in the verifier's own validated input; `Memory` for a
/// layout buffer that could not be reserved. No proof input is read here.
pub fn derive(claims: &ClaimList, epoch: u32) -> Result<[Layout; 10], LayoutError> {
    let params = claims.layout.weights;
    if params.log2_inv_rate != 2 {
        return Err(LayoutError::Internal);
    }
    let k_w = 1u64
        .checked_shl(u32::from(params.log2_k))
        .ok_or(LayoutError::Internal)?;
    let mut layouts = Vec::new();
    layouts
        .try_reserve_exact(10)
        .map_err(|_| LayoutError::Memory)?;
    for matrix in Matrix::ALL {
        let mut layout = if matrix == Matrix::ChannelParams {
            channel(claims, k_w)?
        } else {
            weight(claims, matrix, k_w)?
        };
        layout.commitment_id = descriptor(&layout, params, epoch)?;
        layouts.push(layout);
    }
    layouts.try_into().map_err(|_| LayoutError::Internal)
}

/// A weight proof's rejection, or an error in the verifier's inputs/computation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// The verifier's artifact differs from its own trusted root/epoch: `ERR-USAGE`.
    Usage,
    /// Public artifact and claim list differ: `StatementMismatch { field: Layout }`.
    Layout,
    /// A combined row fails evaluation or shape: `Reject(Evaluation)`.
    Evaluation {
        /// The weight matrix being opened.
        matrix: Matrix,
    },
    /// Column authentication fails: `Reject(MerklePath)`.
    MerklePath {
        /// The weight matrix being opened.
        matrix: Matrix,
        /// The smallest queried column; zero for extra data on an empty matrix.
        column: u32,
    },
    /// A combined codeword disagrees with a column: `Reject(Proximity)`.
    Proximity {
        /// The weight matrix being opened.
        matrix: Matrix,
        /// The smallest failing queried column, across both combined rows.
        column: u32,
    },
    /// A buffer could not be reserved: `ERR-RES-MEMORY`.
    Memory,
    /// The verification deadline expired: `ERR-RES-TIME`.
    Time,
    /// Invalid verifier-derived parameters/points/queries: `ERR-INTERNAL`.
    Internal,
}
impl Error {
    /// Whether a proof was rejected, rather than the verifier being unable to judge it.
    pub const fn is_rejection(self) -> bool {
        matches!(
            self,
            Self::Layout
                | Self::Evaluation { .. }
                | Self::MerklePath { .. }
                | Self::Proximity { .. }
        )
    }
    /// The stable error/rejection code (04 sections2–3).
    pub const fn code(self) -> &'static str {
        match self {
            Self::Usage => "ERR-USAGE",
            Self::Layout => "REJ-STATEMENT",
            Self::Evaluation { .. } => "REJ-PCS-EVAL",
            Self::MerklePath { .. } => "REJ-PCS-MERKLE",
            Self::Proximity { .. } => "REJ-PCS-PROXIMITY",
            Self::Memory => "ERR-RES-MEMORY",
            Self::Time => "ERR-RES-TIME",
            Self::Internal => "ERR-INTERNAL",
        }
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {self:?}", self.code())
    }
}
impl std::error::Error for Error {}

/// Bind the public weight artifact before proof checks (RFC-0040 step4).
///
/// # Errors
/// `Usage` for a root/epoch disagreement with the verifier's own trusted inputs; otherwise
/// `Layout` for a layout differing from the claim list's. The driver compares statement fields.
pub fn check_public(
    claims: &ClaimList,
    public: &WeightCommitmentPublic,
    trusted_root: &Digest,
    trusted_epoch: u32,
) -> Result<(), Error> {
    if &hash(Domain::WeightsRoot, &public.encode()) != trusted_root || public.epoch != trusted_epoch
    {
        return Err(Error::Usage);
    }
    let a = public.layout;
    let b = claims.layout.weights;
    if a.log2_k != b.log2_k
        || a.log2_inv_rate != b.log2_inv_rate
        || a.mask_coeffs != b.mask_coeffs
        || a.blinding_rows != b.blinding_rows
    {
        return Err(Error::Layout);
    }
    Ok(())
}

/// A schedule-derived evaluation at structural row coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Claim<'a> {
    /// The `log2(k_T)` column coordinates.
    pub column_point: &'a [Fp2],
    /// Structural row coordinates, including skipped public zero rows.
    pub row_point: &'a [Fp2],
    /// The transcript-derived coefficient (one for an unbatched claim).
    pub gamma: Fp2,
    /// The schedule's claimed value.
    pub value: Fp2,
}
/// Trusted context for one matrix's opening; no field is taken from its proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Opening<'a> {
    /// The verifier-derived layout.
    pub layout: &'a Layout,
    /// The matrix root from the bound public weight artifact.
    pub root: Digest,
    /// The evaluation claims from the schedule: one for `Embed`..`LmHead`, and `m ≥ 1` in pool
    /// order for `ChannelParams` (RFC-0052).
    pub claims: &'a [Claim<'a>],
    /// The seed-time point indexing committed-row positions (RFC-0049).
    pub xi_point: &'a [Fp2],
}
/// Evaluation passed; column authentication and consistency remain owed.
#[derive(Debug)]
#[must_use = "check the columns to bind these rows to the trusted weight commitment"]
pub struct Evaluated<'a> {
    opening: Opening<'a>,
    combined: &'a [CombinedRow],
    k: usize,
    n: usize,
}

/// Check step9 before queries: `m + 1` unmasked rows (`1 + 1` for `Embed`..`LmHead`, `m + 1` for
/// `ChannelParams`) for a nonempty matrix, no row and all claim values 0 for an empty one. Each
/// evaluation row `i` must have MLE `gamma * value_i` at `column_point_i` (RFC-0052).
/// Weight-type aliases at the same point/gamma each check evaluation row zero (draft RFC-0053).
///
/// # Errors
/// `Internal` for empty claims, distinct point/gamma aliases on `Embed`..`LmHead`, or incorrect
/// verifier-derived point lengths; `Evaluation` for proof shape or evaluation failure. Remaining
/// column checks are carried by the must-use result.
pub fn check_evaluations<'a>(
    opening: Opening<'a>,
    combined: &'a [CombinedRow],
) -> Result<Evaluated<'a>, Error> {
    let layout = opening.layout;
    let first = opening.claims.first().ok_or(Error::Internal)?;
    if layout.matrix != Matrix::ChannelParams
        && opening.claims.iter().any(|c| {
            c.row_point != first.row_point
                || c.column_point != first.column_point
                || c.gamma != first.gamma
        })
    {
        return Err(Error::Internal);
    }
    let cv = usize::try_from(layout.log2_k).map_err(|_| Error::Internal)?;
    let rv = usize::try_from(layout.row_vars).map_err(|_| Error::Internal)?;
    let xv = usize::try_from(layout.proximity_vars()).map_err(|_| Error::Internal)?;
    if opening
        .claims
        .iter()
        .any(|c| c.column_point.len() != cv || c.row_point.len() != rv)
        || opening.xi_point.len() != xv
    {
        return Err(Error::Internal);
    }
    let k = 1usize.checked_shl(layout.log2_k).ok_or(Error::Internal)?;
    let n = k.checked_mul(4).ok_or(Error::Internal)?;
    let failure = Error::Evaluation {
        matrix: layout.matrix,
    };
    if layout.count == 0 {
        if !combined.is_empty() || opening.claims.iter().any(|c| c.value != Fp2::ZERO) {
            return Err(failure);
        }
    } else {
        let evaluations = if layout.matrix == Matrix::ChannelParams {
            opening.claims.len()
        } else {
            1
        };
        let expected_rows = evaluations.checked_add(1).ok_or(Error::Internal)?;
        if combined.len() != expected_rows
            || combined
                .iter()
                .any(|r| r.u.len() != k || !r.mask.is_empty())
        {
            return Err(failure);
        }
        for (index, claim) in opening.claims.iter().enumerate() {
            let row = combined
                .get(if layout.matrix == Matrix::ChannelParams {
                    index
                } else {
                    0
                })
                .ok_or(Error::Internal)?;
            if multilinear(&row.u, claim.column_point).ok_or(Error::Internal)?
                != claim.gamma * claim.value
            {
                return Err(failure);
            }
        }
    }
    Ok(Evaluated {
        opening,
        combined,
        k,
        n,
    })
}
fn eq_index(point: &[Fp2], index: u64) -> Fp2 {
    point.iter().zip(0u32..).fold(Fp2::ONE, |value, (&r, bit)| {
        let one = index.checked_shr(bit).is_some_and(|x| x & 1 == 1);
        value * if one { r } else { Fp2::ONE - r }
    })
}
fn reserve<T>(size: usize) -> Result<Vec<T>, Error> {
    let mut result = Vec::new();
    result.try_reserve_exact(size).map_err(|_| Error::Memory)?;
    Ok(result)
}
impl Evaluated<'_> {
    /// Check step11: shape, direct chain leaves and canonical multiproof, then consistency of
    /// all encoded combined rows at every distinct query. Queries are verifier-derived draws
    /// with replacement; proof columns occur once per distinct index in increasing order.
    ///
    /// # Errors
    /// Invalid own queries give `Internal`; allocation failure `Memory`. Proof shape or
    /// authentication gives `MerklePath` at the smallest query, consistency `Proximity` at the
    /// smallest failing query across all rows.
    pub fn check_columns(
        self,
        queries: &[u64],
        columns: &[Vec<Fp>],
        siblings: &[Digest],
    ) -> Result<(), Error> {
        self.check_columns_with_cancel(queries, columns, siblings, &mut || false)
    }

    pub(crate) fn check_columns_with_cancel(
        self,
        queries: &[u64],
        columns: &[Vec<Fp>],
        siblings: &[Digest],
        cancel: &mut impl FnMut() -> bool,
    ) -> Result<(), Error> {
        let o = self.opening;
        let l = o.layout;
        let matrix = l.matrix;
        if l.count == 0 {
            if !queries.is_empty() {
                return Err(Error::Internal);
            }
            return if columns.is_empty() && siblings.is_empty() {
                Ok(())
            } else {
                Err(Error::MerklePath { matrix, column: 0 })
            };
        }
        let n = u64::try_from(self.n).map_err(|_| Error::Internal)?;
        if queries.is_empty() || queries.iter().any(|&j| j >= n) {
            return Err(Error::Internal);
        }
        let mut distinct = reserve(queries.len())?;
        distinct.extend_from_slice(queries);
        distinct.sort_unstable();
        distinct.dedup();
        let first = u32::try_from(*distinct.first().ok_or(Error::Internal)?)
            .map_err(|_| Error::Internal)?;
        let merkle = Error::MerklePath {
            matrix,
            column: first,
        };
        let count = usize::try_from(l.count).map_err(|_| Error::Internal)?;
        if columns.len() != distinct.len() || columns.iter().any(|col| col.len() != count) {
            return Err(merkle);
        }
        let mut leaves = reserve(distinct.len())?;
        for (&j, col) in distinct.iter().zip(columns) {
            leaves.push((j, column_digest(&l.commitment_id, j, l.row_block, col)));
        }
        verify_multiproof(&o.root, n, &leaves, siblings).map_err(|error| match error {
            MultiproofError::MissingSiblings { .. }
            | MultiproofError::ExtraSiblings { .. }
            | MultiproofError::RootMismatch => merkle,
            _ => Error::Internal,
        })?;
        let mut coefficients = reserve(count)?;
        let mut codeword = reserve(self.n)?;
        let mut agreed = distinct.len();
        for (kind, row) in self.combined.iter().enumerate() {
            coefficients.clear();
            // The final row is proximity even when several claims alias evaluation row zero.
            let claim = if kind.checked_add(1) == Some(self.combined.len()) {
                None
            } else {
                o.claims.get(kind)
            };
            for position in 0..l.count {
                // Structural coordinates for evaluation rows 0..m, dense answer positions for xi.
                let c = match claim {
                    Some(claim) => {
                        claim.gamma
                            * eq_index(claim.row_point, l.row_at(position).ok_or(Error::Internal)?)
                    }
                    None => eq_index(o.xi_point, position),
                };
                coefficients.push(c);
            }
            crate::opening::encode_with_cancel(row, self.k, self.n, &mut codeword, cancel)
                .map_err(|e| match e {
                    OpeningError::Memory => Error::Memory,
                    OpeningError::Time => Error::Time,
                    _ => Error::Internal,
                })?;
            for (position, (&j, col)) in distinct.iter().zip(columns).enumerate().take(agreed) {
                let expected = coefficients
                    .iter()
                    .zip(col)
                    .fold(Fp2::ZERO, |v, (&c, &x)| v + c * x);
                let at = usize::try_from(j).map_err(|_| Error::Internal)?;
                if codeword.get(at).copied().ok_or(Error::Internal)? != expected {
                    agreed = position;
                    break;
                }
            }
            if agreed == 0 {
                break;
            }
        }
        match distinct.get(agreed) {
            Some(&j) => Err(Error::Proximity {
                matrix,
                column: u32::try_from(j).map_err(|_| Error::Internal)?,
            }),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod cancellation_tests {
    use super::*;
    #[test]
    fn authenticated_weight_columns_propagate_transform_cancellation() {
        let layout = Layout {
            matrix: Matrix::Embed,
            log2_k: 0,
            row_vars: 0,
            count: 1,
            rows: Rows::Layers { live: 1, stride: 1 },
            commitment_id: [42; 32],
            row_block: NonZeroU32::new(1).unwrap(),
        };
        let columns = vec![vec![Fp::ZERO]; 4];
        let leaves: Vec<_> = (0..4)
            .map(|j| {
                column_digest(
                    &layout.commitment_id,
                    j,
                    layout.row_block,
                    &columns[j as usize],
                )
            })
            .collect();
        let root = tv_hash::merkle::node(
            &tv_hash::merkle::node(&leaves[0], &leaves[1]),
            &tv_hash::merkle::node(&leaves[2], &leaves[3]),
        );
        let claims = [Claim {
            column_point: &[],
            row_point: &[],
            gamma: Fp2::ONE,
            value: Fp2::ZERO,
        }];
        let opening = Opening {
            layout: &layout,
            root,
            claims: &claims,
            xi_point: &[],
        };
        let combined = vec![
            CombinedRow {
                u: vec![Fp2::ZERO],
                mask: vec![]
            };
            2
        ];
        let queries = [0, 1, 2, 3];
        assert_eq!(
            check_evaluations(opening, &combined)
                .unwrap()
                .check_columns(&queries, &columns, &[]),
            Ok(())
        );
        for stop in [1, 3] {
            let mut polls = 0;
            assert_eq!(
                check_evaluations(opening, &combined)
                    .unwrap()
                    .check_columns_with_cancel(&queries, &columns, &[], &mut || {
                        polls += 1;
                        polls == stop
                    }),
                Err(Error::Time)
            );
            assert_eq!(polls, stop);
        }
        assert_eq!(Error::Time.code(), "ERR-RES-TIME");
        assert!(!Error::Time.is_rejection());
    }
}
