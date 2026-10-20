// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The opening checks of the advice commitment (RFC-0011 'Opening protocol' steps 5 to 8,
//! RFC-0019 'Check order' steps 9 and 11). The crate documentation ('Openings') is their
//! specification: what is checked, in which order, and what each error means.
//!
//! The verifier runs [`check_evaluations`] when the combined rows arrive, before it draws the
//! column queries, and [`Evaluated::check_columns`] on the answer to them. The second takes what
//! the first returns, so the columns are only ever checked against combined rows that passed the
//! evaluation check, for the same opening.
//! [`crate::advice::opening`] binds canonical model geometry and original combination claims
//! to this shared checking engine; this module's public primitive takes explicit row maps.
//!
//! ```
//! use tv_core::opening::{
//!     Claim, CombinedRow, Group, Layout, Opening, OpeningError, Params, check_evaluations,
//! };
//! use tv_field::{Fp, Fp2};
//!
//! let value = |x: u64| Fp2::from(Fp::reduce(x));
//! // Rows of k = 2 values, encoded at rate 1/2 with one mask coefficient. Rows 0 and 1, of the
//! // only segment, are the non-zero rows.
//! let params = Params {
//!     log2_k: 1,
//!     log2_inv_rate: 1,
//!     mask_coeffs: 1,
//!     row_block: 32,
//!     commitment_id: [0; 32],
//! };
//! let segments = vec![vec![0, 1]];
//! // One claim, Ã(0 ‖ 0) = 5: row 0 at column 0. Its group's combined row is row 0 itself.
//! let claim = Claim { row_point: vec![value(0)], gamma: value(1), value: value(5) };
//! let groups = vec![Group { column_point: vec![value(0)], claims: vec![claim] }];
//! let opening = Opening {
//!     params,
//!     layout: Layout { row_vars: 1, segments: &segments },
//!     root: [0; 32],
//!     groups: &groups,
//!     xi_point: &[value(3)],
//! };
//! let combined = vec![
//!     CombinedRow { u: vec![value(5), value(9)], mask: vec![value(4)] },
//!     // The proximity row.
//!     CombinedRow { u: vec![value(7), value(2)], mask: vec![value(1)] },
//! ];
//! let evaluated = check_evaluations(opening, &combined)?;
//! // The column check follows on the answer to the queries, here one without its columns:
//! // rejected at the smallest queried column.
//! assert_eq!(
//!     evaluated.check_columns(&[3, 1, 3], &[], &[]),
//!     Err(OpeningError::MerklePath { column: 1 })
//! );
//!
//! // Without its proximity row, the proof has the wrong shape.
//! assert_eq!(
//!     check_evaluations(opening, &combined[..1]).err(),
//!     Some(OpeningError::Evaluation { group: None })
//! );
//! // A combined row that takes the value 9 at column 0 does not give the claim.
//! let swapped = vec![
//!     CombinedRow { u: vec![value(9), value(5)], mask: vec![value(4)] },
//!     combined[1].clone(),
//! ];
//! assert_eq!(
//!     check_evaluations(opening, &swapped).err(),
//!     Some(OpeningError::Evaluation { group: Some(0) })
//! );
//! # Ok::<(), OpeningError>(())
//! ```

use core::fmt;
use core::num::NonZeroU32;

use tv_field::{Fp, Fp2, TWO_ADICITY, coset_ntt, intt};
use tv_hash::column::column_digest;
use tv_hash::merkle::{MultiproofError, verify_multiproof};
use tv_hash::{Digest, Domain, DomainHasher};

/// The largest bit count of a row index: a row is a `u64`.
pub const MAX_ROW_VARS: u32 = 64;

/// The shift of the coset `7·L` on which codewords are evaluated (RFC-0011 'Encoding').
const COSET_SHIFT: Fp = Fp::SEVEN;

/// The values a fold keeps, one per level: a row of `k ≤ 2^32` values has at most 33 levels.
const FOLD_LEVELS: usize = 33;

/// The bits of a row index that one table of `eq` covers.
const TABLE_BITS: usize = 8;

/// The public parameters of an advice commitment (RFC-0011 'Encoding', 'Column digests and
/// root'), from the parameter set and the commitment's descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Params {
    /// `log2 k`: each row holds `k = 2^log2_k` values, the evaluations of a polynomial of degree
    /// below `k` on the subgroup `H` of order `k`.
    pub log2_k: u32,
    /// `log2` of the inverse rate: a row's codeword has `n = k·2^log2_inv_rate` symbols, its
    /// values on the coset `7·L` of the subgroup of order `n`.
    pub log2_inv_rate: u32,
    /// `P_A`, the coefficients of each row's mask, from 1 to `k`.
    pub mask_coeffs: u32,
    /// `B`, the rows a column digest absorbs per step (RFC-0004 'Column digests').
    pub row_block: u32,
    /// The 32 bytes bound into the first digest of every column.
    pub commitment_id: Digest,
}

/// Which rows of the advice matrix are committed, from the public layout (RFC-0011 'Canonical
/// advice layout').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Layout<'a> {
    /// The row index's bit count: the matrix has rows `0` to `2^row_vars − 1`, at most 64 bits.
    pub row_vars: u32,
    /// The non-zero rows of each segment, segments in layout order and rows increasing within
    /// and across them. A segment may be empty. Every row not listed is zero: it is neither
    /// committed nor opened.
    pub segments: &'a [Vec<u64>],
}

/// A claim `Ã(column_point ‖ row_point) = value` on the advice polynomial, the column point being
/// its group's.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Claim {
    /// The row part of the point, `row_vars` coordinates, lowest variable first.
    pub row_point: Vec<Fp2>,
    /// `γ`, the claim's batching challenge.
    pub gamma: Fp2,
    /// The claimed value.
    pub value: Fp2,
}

/// The claims that share a column sub-point (RFC-0011 step 1).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Group {
    /// The column part of the claims' points, `log2_k` coordinates, lowest variable first.
    pub column_point: Vec<Fp2>,
    /// The group's claims, in the order the caller batches them.
    pub claims: Vec<Claim>,
}

/// A combined row of the proof (RFC-0011 step 5): `u_g` and `M_g` for a group, or `u_ξ` and
/// `M_ξ` for the proximity row.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CombinedRow {
    /// `u`: the combined row's `k` values on `H`.
    pub u: Vec<Fp2>,
    /// `M`: the combined mask's `P_A` coefficients, constant first.
    pub mask: Vec<Fp2>,
}

/// What the verifier knows of an advice opening before the combined rows arrive: the
/// commitment, the layout, and the claims and challenges from the transcript.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Opening<'a> {
    /// The commitment's parameters.
    pub params: Params,
    /// The non-zero rows.
    pub layout: Layout<'a>,
    /// The commitment's Merkle root over its `n` leaves.
    pub root: Digest,
    /// The groups of claims, in group order.
    pub groups: &'a [Group],
    /// `ξ*`, `row_vars` coordinates: the proximity row's coefficients are `ξ_row = eq(ξ*, row)`.
    pub xi_point: &'a [Fp2],
}

/// Combined rows that passed [`check_evaluations`], with the opening they were checked for.
///
/// They are bound to the claims, and to the commitment only once
/// [`check_columns`](Self::check_columns) has checked them against the queried columns.
#[derive(Clone, Debug)]
#[must_use = "the combined rows are bound to the commitment only once the queried columns are checked against them"]
pub struct Evaluated<'a> {
    opening: Opening<'a>,
    combined: &'a [CombinedRow],
    sizes: Sizes,
}

/// Why an opening was rejected or could not be checked: the first check that failed, in the
/// order of the crate documentation ('Openings').
///
/// `Evaluation`, `MerklePath` and `Proximity` judge the proof: `Reject(Evaluation | MerklePath |
/// Proximity)` in 04 §2, for the commitment the caller checks. Every other variant judges what
/// the verifier passes in, or the verifier itself, and means `Inconclusive`, never a verdict on
/// the proof: [`code`](Self::code) gives which.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpeningError {
    /// The combined rows do not have the opening's shape (`group` is `None`), or group `group`'s
    /// claims do not give what its combined row opens to at its column point.
    Evaluation {
        /// The first group whose sums differ, or `None` for a shape that is wrong.
        group: Option<usize>,
    },
    /// The answer does not hold one column of `N` symbols per distinct query, or the columns'
    /// leaves and the siblings do not lead to the root. A multiproof fails as a whole, so the
    /// column is the smallest queried index.
    MerklePath {
        /// The smallest queried index.
        column: u32,
    },
    /// A combined row disagrees with a queried column.
    Proximity {
        /// The smallest queried index at which some combined row disagrees.
        column: u32,
    },
    /// The inverse rate is 1, or the codeword is longer than `2^32` or than memory can address.
    CodeLength {
        /// `log2 k`.
        log2_k: u32,
        /// `log2` of the inverse rate.
        log2_inv_rate: u32,
    },
    /// `P_A` is 0 or above `k`.
    MaskCoefficients {
        /// `P_A`.
        mask_coeffs: u32,
    },
    /// `B` is 0.
    RowBlock,
    /// The row index has more than [`MAX_ROW_VARS`] bits.
    RowVars {
        /// The bit count given.
        row_vars: u32,
    },
    /// The layout has no segment.
    NoSegments,
    /// A row is not above the one before it in layout order.
    RowOrder {
        /// The row's segment.
        segment: usize,
        /// The row's position in its segment.
        position: usize,
    },
    /// A row is `2^row_vars` or more.
    RowRange {
        /// The row's segment.
        segment: usize,
        /// The row's position in its segment.
        position: usize,
    },
    /// A group's column point does not have `log2_k` coordinates.
    ColumnPoint {
        /// The group.
        group: usize,
        /// The coordinates it has.
        length: usize,
    },
    /// A claim's row point does not have `row_vars` coordinates.
    RowPoint {
        /// The claim's group.
        group: usize,
        /// The claim's position in its group.
        claim: usize,
        /// The coordinates it has.
        length: usize,
    },
    /// `ξ*` does not have `row_vars` coordinates.
    XiPoint {
        /// The coordinates it has.
        length: usize,
    },
    /// No column is queried.
    NoQueries,
    /// A query is `n` or more.
    QueryRange {
        /// The query's position in draw order.
        position: usize,
    },
    /// Memory for a buffer the parameters size could not be reserved.
    Memory,
    /// A step that the checks before it make infallible failed: a bug of this crate.
    Internal,
}

impl OpeningError {
    /// Whether the proof is at fault: `Reject(reason)` in 04 §2. Otherwise the verifier could
    /// not judge the proof: `Inconclusive(error)`.
    pub const fn is_rejection(&self) -> bool {
        matches!(
            self,
            Self::Evaluation { .. } | Self::MerklePath { .. } | Self::Proximity { .. }
        )
    }

    /// The error's identifier in 04 §2 and §3: `REJ-PCS-EVAL`, `REJ-PCS-MERKLE` and
    /// `REJ-PCS-PROXIMITY` for the proof's faults, `ERR-RES-MEMORY` for memory that could not be
    /// reserved, `ERR-INTERNAL` for every other fault of the verifier.
    pub const fn code(&self) -> &'static str {
        match self {
            Self::Evaluation { .. } => "REJ-PCS-EVAL",
            Self::MerklePath { .. } => "REJ-PCS-MERKLE",
            Self::Proximity { .. } => "REJ-PCS-PROXIMITY",
            Self::Memory => "ERR-RES-MEMORY",
            Self::CodeLength { .. }
            | Self::MaskCoefficients { .. }
            | Self::RowBlock
            | Self::RowVars { .. }
            | Self::NoSegments
            | Self::RowOrder { .. }
            | Self::RowRange { .. }
            | Self::ColumnPoint { .. }
            | Self::RowPoint { .. }
            | Self::XiPoint { .. }
            | Self::NoQueries
            | Self::QueryRange { .. }
            | Self::Internal => "ERR-INTERNAL",
        }
    }
}

impl fmt::Display for OpeningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Evaluation { group: None } => {
                f.write_str("the combined rows do not have the opening's shape")
            }
            Self::Evaluation { group: Some(group) } => write!(
                f,
                "the claims of group {group} do not give what its combined row opens to"
            ),
            Self::MerklePath { column } => write!(
                f,
                "the answer does not open the queried columns, the smallest of which is {column}"
            ),
            Self::Proximity { column } => {
                write!(f, "a combined row disagrees with column {column}")
            }
            Self::CodeLength {
                log2_k,
                log2_inv_rate,
            } => write!(
                f,
                "rows of 2^{log2_k} values at inverse rate 2^{log2_inv_rate} have no supported \
                 code: the inverse rate is at least 2 and the codeword at most 2^{TWO_ADICITY} \
                 symbols"
            ),
            Self::MaskCoefficients { mask_coeffs } => write!(
                f,
                "{mask_coeffs} mask coefficients asked for; a row's mask has 1 to k"
            ),
            Self::RowBlock => f.write_str("column digests cannot chain blocks of 0 rows"),
            Self::RowVars { row_vars } => write!(
                f,
                "row indices of {row_vars} bits asked for; they have at most {MAX_ROW_VARS}"
            ),
            Self::NoSegments => f.write_str("the layout has no segment"),
            Self::RowOrder { segment, position } => write!(
                f,
                "row {position} of segment {segment} is not above the row before it"
            ),
            Self::RowRange { segment, position } => write!(
                f,
                "row {position} of segment {segment} is outside the row index's range"
            ),
            Self::ColumnPoint { group, length } => write!(
                f,
                "the column point of group {group} has {length} coordinates, not log2 k"
            ),
            Self::RowPoint {
                group,
                claim,
                length,
            } => write!(
                f,
                "the row point of claim {claim} of group {group} has {length} coordinates, not \
                 the row index's bit count"
            ),
            Self::XiPoint { length } => write!(
                f,
                "ξ* has {length} coordinates, not the row index's bit count"
            ),
            Self::NoQueries => f.write_str("no column is queried"),
            Self::QueryRange { position } => {
                write!(f, "query {position} is outside the codeword")
            }
            Self::Memory => f.write_str("memory for the column check could not be reserved"),
            Self::Internal => {
                f.write_str("an opening check failed a step its earlier checks make infallible")
            }
        }
    }
}

impl std::error::Error for OpeningError {}

/// Checks the verifier's inputs, then the shape of the combined rows and each group's
/// evaluation claim (RFC-0011 steps 5 and 6, RFC-0019 step 9), in the order of the crate
/// documentation ('The verifier's inputs', 'The evaluation check'):
///
/// - `combined` holds `G + 1` rows, the groups' in group order and then the proximity row, each
///   of `k` values and `P_A` mask coefficients, else `Evaluation { group: None }`;
/// - for each group `g` in order, `Σ γ·value` over its claims equals
///   `Σ_{i < k} u_g[i]·eq(column_point, i)`, else `Evaluation { group: Some(g) }`.
///
/// It reads nothing of the proof before the opening's own values pass their checks, which are
/// faults of the verifier, and allocates nothing.
///
/// `Ok` binds the combined rows to the claims only: the caller must check the answer to the
/// column queries with [`Evaluated::check_columns`], which binds them to the commitment.
pub fn check_evaluations<'a>(
    opening: Opening<'a>,
    combined: &'a [CombinedRow],
) -> Result<Evaluated<'a>, OpeningError> {
    let sizes = Sizes::of(&opening)?;

    check_combined(
        sizes,
        combined,
        opening.groups.iter().map(|group| {
            let claimed = group
                .claims
                .iter()
                .fold(Fp2::ZERO, |sum, claim| sum + claim.gamma * claim.value);
            (group.column_point.as_slice(), claimed)
        }),
    )?;
    Ok(Evaluated {
        opening,
        combined,
        sizes,
    })
}

/// Shared shape and evaluation checks; each caller derives its own claim sums.
pub(crate) fn check_combined<'a>(
    sizes: Sizes,
    combined: &[CombinedRow],
    groups: impl ExactSizeIterator<Item = (&'a [Fp2], Fp2)>,
) -> Result<(), OpeningError> {
    // RFC-0011 step 5: G + 1 rows of k values and P_A mask coefficients.
    let rows = groups.len().saturating_add(1);
    let shaped = combined.len() == rows
        && combined
            .iter()
            .all(|row| row.u.len() == sizes.k && row.mask.len() == sizes.mask);
    if !shaped {
        return Err(OpeningError::Evaluation { group: None });
    }

    // Step 6: Σ γ·v = ⟨u_g, eq(column point)⟩, group by group.
    for (index, (group, row)) in groups.zip(combined).enumerate() {
        let (point, claimed) = group;
        let opened = multilinear(&row.u, point).ok_or(OpeningError::Internal)?;
        if claimed != opened {
            return Err(OpeningError::Evaluation { group: Some(index) });
        }
    }
    Ok(())
}

impl Evaluated<'_> {
    /// Checks the answer to the column queries (RFC-0011 steps 7 and 8, RFC-0019 step 11), in the
    /// order of the crate documentation ('The column check'). `queries` are the drawn column
    /// indices in draw order, possibly repeated; `columns` hold one column per distinct index, in
    /// increasing index order, each with one symbol per non-zero row in layout order; `siblings`
    /// are the canonical multiproof of the distinct indices (RFC-0004 'Merkle trees').
    ///
    /// - The queries are at least one and each below `n`: else `NoQueries` or `QueryRange`,
    ///   faults of the verifier.
    /// - One column of `N` symbols per distinct query, whose leaves and the siblings lead to the
    ///   root: else `MerklePath` at the smallest queried index.
    /// - Every combined row, groups then the proximity row, agrees with every queried column:
    ///   else `Proximity` at the smallest queried index where one does not.
    ///
    /// `Ok` is the end of the opening: the claims hold on the committed advice, up to the
    /// soundness error of RFC-0011 'Opening protocol'.
    pub fn check_columns(
        self,
        queries: &[u64],
        columns: &[Vec<Fp>],
        siblings: &[Digest],
    ) -> Result<(), OpeningError> {
        let Self {
            opening,
            combined,
            sizes,
        } = self;

        ColumnCheck {
            root: &opening.root,
            combined,
            sizes,
        }
        .check(
            queries,
            columns,
            siblings,
            |j, column| leaf(&opening, sizes.block, j, column),
            |index, out| row_coefficients(&opening, index, out, sizes.rows),
        )
    }
}

/// Authentication and consistency shared by explicit and canonical advice layouts.
pub(crate) struct ColumnCheck<'a> {
    pub(crate) root: &'a Digest,
    pub(crate) combined: &'a [CombinedRow],
    pub(crate) sizes: Sizes,
}

impl ColumnCheck<'_> {
    pub(crate) fn check(
        self,
        queries: &[u64],
        columns: &[Vec<Fp>],
        siblings: &[Digest],
        make_leaf: impl Fn(u64, &[Fp]) -> Result<Digest, OpeningError>,
        fill_coefficients: impl Fn(usize, &mut Vec<Fp2>) -> Result<(), OpeningError>,
    ) -> Result<(), OpeningError> {
        let Self {
            root,
            combined,
            sizes,
        } = self;
        // 1. The queries, the verifier's own; D, their sorted distinct set.
        if queries.is_empty() {
            return Err(OpeningError::NoQueries);
        }
        if let Some(position) = queries.iter().position(|&j| j >= sizes.leaves) {
            return Err(OpeningError::QueryRange { position });
        }
        let mut distinct = Vec::new();
        distinct
            .try_reserve_exact(queries.len())
            .map_err(|_| OpeningError::Memory)?;
        distinct.extend_from_slice(queries);
        distinct.sort_unstable();
        distinct.dedup();
        let smallest = distinct.first().copied().ok_or(OpeningError::Internal)?;
        let merkle_path = OpeningError::MerklePath {
            column: column_number(smallest)?,
        };

        // 2. Step 8.1: one column per distinct index, each of N symbols.
        if columns.len() != distinct.len()
            || columns.iter().any(|column| column.len() != sizes.rows)
        {
            return Err(merkle_path);
        }

        // 3. Step 8.2: the leaves, column i being column D_i, and the multiproof.
        let mut leaves = Vec::new();
        leaves
            .try_reserve_exact(distinct.len())
            .map_err(|_| OpeningError::Memory)?;
        for (&j, column) in distinct.iter().zip(columns) {
            leaves.push((j, make_leaf(j, column)?));
        }
        verify_multiproof(root, sizes.leaves, &leaves, siblings).map_err(|error| {
            match error {
                MultiproofError::MissingSiblings { .. }
                | MultiproofError::ExtraSiblings { .. }
                | MultiproofError::RootMismatch => merkle_path,
                // n is a power of two and D sorted, distinct, non-empty and below n.
                MultiproofError::TreeSize
                | MultiproofError::NoLeaves
                | MultiproofError::Unordered { .. }
                | MultiproofError::OutOfRange { .. } => OpeningError::Internal,
            }
        })?;

        // 4. Step 8.3: each combined row against each queried column, one codeword at a time.
        let mut coefficients = Vec::new();
        coefficients
            .try_reserve_exact(sizes.rows)
            .map_err(|_| OpeningError::Memory)?;
        let mut codeword = Vec::new();
        codeword
            .try_reserve_exact(sizes.n)
            .map_err(|_| OpeningError::Memory)?;
        // Every row checked so far agrees with the columns at the positions of D before this
        // one. A row is checked only there: a failure further on would not be the first.
        let mut agreed = distinct.len();
        for (index, row) in combined.iter().enumerate() {
            if agreed == 0 {
                break;
            }
            fill_coefficients(index, &mut coefficients)?;
            if coefficients.len() != sizes.rows {
                return Err(OpeningError::Internal);
            }
            encode(row, sizes.k, sizes.n, &mut codeword)?;
            for (position, (&j, column)) in distinct.iter().zip(columns).enumerate().take(agreed) {
                let at = usize::try_from(j).map_err(|_| OpeningError::Internal)?;
                let expected = *codeword.get(at).ok_or(OpeningError::Internal)?;
                if combine(&coefficients, column) != expected {
                    agreed = position;
                    break;
                }
            }
        }
        match distinct.get(agreed) {
            Some(&j) => Err(OpeningError::Proximity {
                column: column_number(j)?,
            }),
            None => Ok(()),
        }
    }
}

/// The sizes the verifier's inputs fix, once checked.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Sizes {
    /// `k`, the values of a row.
    pub(crate) k: usize,
    /// `n`, the symbols of a codeword.
    pub(crate) n: usize,
    /// `n`, the leaves of the Merkle tree.
    pub(crate) leaves: u64,
    /// `P_A`, the coefficients of a mask.
    pub(crate) mask: usize,
    /// `B`, the rows of a column-digest step.
    pub(crate) block: NonZeroU32,
    /// `N`, the non-zero rows.
    pub(crate) rows: usize,
}

impl Sizes {
    /// Checks the verifier's inputs in the order of the crate documentation ('The verifier's
    /// inputs') and returns the sizes they fix.
    pub(crate) fn parameters(params: Params) -> Result<Self, OpeningError> {
        let Params {
            log2_k,
            log2_inv_rate,
            mask_coeffs,
            row_block,
            ..
        } = params;

        // 1. 2k ≤ n ≤ 2^32, and n addressable.
        let code_length = OpeningError::CodeLength {
            log2_k,
            log2_inv_rate,
        };
        let log2_n = log2_k
            .checked_add(log2_inv_rate)
            .filter(|&log2_n| log2_inv_rate >= 1 && log2_n <= TWO_ADICITY)
            .ok_or(code_length)?;
        let leaves = 1u64.checked_shl(log2_n).ok_or(code_length)?;
        let n = usize::try_from(leaves).map_err(|_| code_length)?;
        let k = n.checked_shr(log2_inv_rate).ok_or(code_length)?;

        // 2. 1 ≤ P_A ≤ k.
        let mask = usize::try_from(mask_coeffs)
            .ok()
            .filter(|mask| (1..=k).contains(mask))
            .ok_or(OpeningError::MaskCoefficients { mask_coeffs })?;

        // 3. B ≥ 1.
        let block = NonZeroU32::new(row_block).ok_or(OpeningError::RowBlock)?;

        Ok(Self {
            k,
            n,
            leaves,
            mask,
            block,
            rows: 0,
        })
    }

    fn of(opening: &Opening<'_>) -> Result<Self, OpeningError> {
        let mut sizes = Self::parameters(opening.params)?;
        let log2_k = opening.params.log2_k;
        let code_length = OpeningError::CodeLength {
            log2_k,
            log2_inv_rate: opening.params.log2_inv_rate,
        };
        // 4. to 6. The layout.
        sizes.rows = check_layout(opening.layout)?;

        // 7. and 8. The points.
        let column_vars = usize::try_from(log2_k).map_err(|_| code_length)?;
        let row_vars =
            usize::try_from(opening.layout.row_vars).map_err(|_| OpeningError::RowVars {
                row_vars: opening.layout.row_vars,
            })?;
        for (group, entry) in opening.groups.iter().enumerate() {
            if entry.column_point.len() != column_vars {
                return Err(OpeningError::ColumnPoint {
                    group,
                    length: entry.column_point.len(),
                });
            }
            for (claim, entry) in entry.claims.iter().enumerate() {
                if entry.row_point.len() != row_vars {
                    return Err(OpeningError::RowPoint {
                        group,
                        claim,
                        length: entry.row_point.len(),
                    });
                }
            }
        }
        if opening.xi_point.len() != row_vars {
            return Err(OpeningError::XiPoint {
                length: opening.xi_point.len(),
            });
        }

        Ok(sizes)
    }
}

/// Checks the layout: the row index's bit count, at least one segment, and every row above the
/// one before it and below `2^row_vars`. Returns `N`, the number of non-zero rows.
fn check_layout(layout: Layout<'_>) -> Result<usize, OpeningError> {
    let Layout { row_vars, segments } = layout;
    if row_vars > MAX_ROW_VARS {
        return Err(OpeningError::RowVars { row_vars });
    }
    if segments.is_empty() {
        return Err(OpeningError::NoSegments);
    }
    let mut previous: Option<u64> = None;
    let mut total: usize = 0;
    for (segment, rows) in segments.iter().enumerate() {
        for (position, &row) in rows.iter().enumerate() {
            if previous.is_some_and(|previous| row <= previous) {
                return Err(OpeningError::RowOrder { segment, position });
            }
            // The bits from row_vars up are zero; with row_vars = 64 there are none.
            if row.checked_shr(row_vars).is_some_and(|high| high != 0) {
                return Err(OpeningError::RowRange { segment, position });
            }
            previous = Some(row);
        }
        // The rows are in memory, so their number fits a usize.
        total = total.saturating_add(rows.len());
    }
    Ok(total)
}

/// `Σ_x values[x]·eq(point, x)` over the `2^m` values of a table on `m = point.len()` variables,
/// or `None` when there are not `2^m` values or `m` exceeds 32.
///
/// One pass, merging each pair of values as soon as both are known: values `x` and `x + 2^t` of
/// level `t` differ in bit `t`, and merge into `a + r_t·(b − a)`, the value of level `t + 1`.
/// Value `x` closes as many levels as `x` has trailing ones, so `pending[t]` holds the open value
/// of level `t`.
pub(crate) fn multilinear(values: &[Fp2], point: &[Fp2]) -> Option<Fp2> {
    let vars = u32::try_from(point.len()).ok()?;
    let size = 1usize.checked_shl(vars)?;
    if values.len() != size || point.len() >= FOLD_LEVELS {
        return None;
    }
    let mut pending = [Fp2::ZERO; FOLD_LEVELS];
    for (index, &value) in values.iter().enumerate() {
        let closed = usize::try_from(index.trailing_ones()).ok()?;
        let mut carry = value;
        for (&low, &r) in pending.iter().zip(point).take(closed) {
            carry = low + r * (carry - low);
        }
        *pending.get_mut(closed)? = carry;
    }
    pending.get(point.len()).copied()
}

/// `weight·eq(point, x)` for `x` of `point.len()` bits, as `weight` times one entry per 8 bits
/// of `x`: table `c` holds `eq` of the coordinates `8c, …, 8c + 7` at every value of those bits.
pub(crate) struct EqTables {
    weight: Fp2,
    vars: u32,
    tables: Vec<Vec<Fp2>>,
}

impl EqTables {
    /// The tables of `point`, at most [`MAX_ROW_VARS`] coordinates, or `None` for a longer one.
    pub(crate) fn new(point: &[Fp2], weight: Fp2) -> Option<Self> {
        let vars = u32::try_from(point.len())
            .ok()
            .filter(|&vars| vars <= MAX_ROW_VARS)?;
        let tables = point
            .chunks(TABLE_BITS)
            .map(|chunk| {
                // Each coordinate doubles the table: the new bit is the highest, 0 in the low
                // half and 1 in the high half.
                let mut table = vec![Fp2::ONE];
                for &r in chunk {
                    let high: Vec<Fp2> = table.iter().map(|&entry| entry * r).collect();
                    let one_minus_r = Fp2::ONE - r;
                    for entry in &mut table {
                        *entry *= one_minus_r;
                    }
                    table.extend(high);
                }
                table
            })
            .collect();
        Some(Self {
            weight,
            vars,
            tables,
        })
    }

    /// `weight·eq(point, x)`, or `None` when `x ≥ 2^point.len()`.
    pub(crate) fn at(&self, x: u64) -> Option<Fp2> {
        if x.checked_shr(self.vars).is_some_and(|high| high != 0) {
            return None;
        }
        // At most 64 coordinates: one table per byte of x at most.
        let mut value = self.weight;
        for (table, byte) in self.tables.iter().zip(x.to_le_bytes()) {
            value *= *table.get(usize::from(byte))?;
        }
        Some(value)
    }
}

/// Writes the coefficients of combined row `index` for every non-zero row in layout order into
/// `out`, which holds `rows` values after: `c_g(row) = Σ γ·eq(row_point, row)` for group
/// `g = index`, and `ξ_row = eq(ξ*, row)` for the proximity row, which follows the last group.
fn row_coefficients(
    opening: &Opening<'_>,
    index: usize,
    out: &mut Vec<Fp2>,
    rows: usize,
) -> Result<(), OpeningError> {
    out.clear();
    out.resize(rows, Fp2::ZERO);
    let mut add = |point: &[Fp2], weight: Fp2| -> Result<(), OpeningError> {
        let tables = EqTables::new(point, weight).ok_or(OpeningError::Internal)?;
        let layout_rows = opening.layout.segments.iter().flatten();
        for (slot, &row) in out.iter_mut().zip(layout_rows) {
            *slot += tables.at(row).ok_or(OpeningError::Internal)?;
        }
        Ok(())
    };
    match opening.groups.get(index) {
        Some(group) => {
            for claim in &group.claims {
                add(&claim.row_point, claim.gamma)?;
            }
        }
        None => add(opening.xi_point, Fp2::ONE)?,
    }
    Ok(())
}

/// `Σ_row coefficients[row]·column[row]`: a combined row's coefficients against a column's
/// symbols.
fn combine(coefficients: &[Fp2], column: &[Fp]) -> Fp2 {
    coefficients
        .iter()
        .zip(column)
        .fold(Fp2::ZERO, |sum, (&coefficient, &symbol)| {
            sum + coefficient * symbol
        })
}

/// The codeword of a combined row, its `n` values `enc(u)(x_j) + Z_H(x_j)·M(x_j)` at
/// `x_j = 7·ω_n^j`, written into `codeword`: the inverse NTT on `H` gives `f = enc(u)`'s
/// coefficients, `g = f + (X^k − 1)·M` subtracts `M` from the low coefficients and adds it from
/// `X^k` up, and the coset NTT evaluates `g` on `7·L`.
pub(crate) fn encode(
    row: &CombinedRow,
    k: usize,
    n: usize,
    codeword: &mut Vec<Fp2>,
) -> Result<(), OpeningError> {
    codeword.clear();
    codeword.extend_from_slice(&row.u);
    codeword.resize(n, Fp2::ZERO);
    let (low, high) = codeword
        .split_at_mut_checked(k)
        .ok_or(OpeningError::Internal)?;
    // The shape check gives P_A ≤ k ≤ n − k, so no mask coefficient falls outside the zips.
    if row.mask.len() > low.len() || row.mask.len() > high.len() {
        return Err(OpeningError::Internal);
    }
    intt(low).map_err(|_| OpeningError::Internal)?;
    for ((low, high), &m) in low.iter_mut().zip(high.iter_mut()).zip(&row.mask) {
        *low -= m;
        *high += m;
    }
    coset_ntt(codeword, COSET_SHIFT).map_err(|_| OpeningError::Internal)
}

/// `leaf_j = H("column-merge", d_j^(1) ‖ … ‖ d_j^(S))`, each segment's digest chaining its share
/// of the column's symbols, which the shape check sized to the layout.
fn leaf(
    opening: &Opening<'_>,
    block: NonZeroU32,
    column: u64,
    symbols: &[Fp],
) -> Result<Digest, OpeningError> {
    merged_leaf(
        &opening.params.commitment_id,
        block,
        column,
        symbols,
        opening.layout.segments.iter().map(|part| Ok(part.len())),
    )
}

/// Every segment participates, including empty segments and S=1 (INV-PCS-3).
pub(crate) fn merged_leaf(
    commitment_id: &Digest,
    block: NonZeroU32,
    column: u64,
    symbols: &[Fp],
    segments: impl Iterator<Item = Result<usize, OpeningError>>,
) -> Result<Digest, OpeningError> {
    let mut merge = DomainHasher::new(Domain::ColumnMerge);
    let mut rest = symbols;
    for length in segments {
        let (own, tail) = rest
            .split_at_checked(length?)
            .ok_or(OpeningError::Internal)?;
        merge.update(&column_digest(commitment_id, column, block, own));
        rest = tail;
    }
    if !rest.is_empty() {
        return Err(OpeningError::Internal);
    }
    Ok(merge.finalize())
}

/// A queried index as a rejection reports it: below `n ≤ 2^32`, so a `u32`.
fn column_number(j: u64) -> Result<u32, OpeningError> {
    u32::try_from(j).map_err(|_| OpeningError::Internal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sumcheck::eq;
    use tv_field::root_of_unity;
    use tv_testkit::{Json, SplitMix64, read_vectors};

    fn member<'a>(json: &'a Json, key: &str) -> &'a Json {
        json.get(key).unwrap_or_else(|| panic!("no `{key}`"))
    }

    fn list<'a>(json: &'a Json, key: &str) -> &'a [Json] {
        member(json, key).as_array().unwrap()
    }

    fn fp(json: &Json) -> Fp {
        Fp::from_canonical(json.as_hex_u64().unwrap()).unwrap()
    }

    /// An `Fp2` written `[c0, c1]`, or an `Fp` written alone, embedded.
    fn fp2(json: &Json) -> Fp2 {
        match json.as_array() {
            Some([c0, c1]) => Fp2::new(fp(c0), fp(c1)),
            Some(_) => panic!("an Fp2 element has two coefficients"),
            None => Fp2::from(fp(json)),
        }
    }

    fn fp2s(json: &Json) -> Vec<Fp2> {
        json.as_array().unwrap().iter().map(fp2).collect()
    }

    fn random(rng: &mut SplitMix64) -> Fp2 {
        Fp2::new(Fp::reduce(rng.next_u64()), Fp::reduce(rng.next_u64()))
    }

    /// `x`'s low `vars` bits as a Boolean point, lowest first.
    fn bits(x: u64, vars: usize) -> Vec<Fp2> {
        (0..vars)
            .map(|t| {
                if (x >> t) & 1 == 1 {
                    Fp2::ONE
                } else {
                    Fp2::ZERO
                }
            })
            .collect()
    }

    /// `spec/vectors/poly/poly.json` 'multilinear', from an independent oracle: tables of `Fp`
    /// and `Fp2` values on up to 12 variables, and their extensions at a point.
    #[test]
    fn the_fold_gives_the_poly_vectors_extensions() {
        let file = read_vectors("poly/poly.json");
        let cases = list(&file, "multilinear");
        assert_eq!(cases.len(), 22);
        for case in cases {
            let table = fp2s(member(case, "table"));
            let point = fp2s(member(case, "point"));
            assert_eq!(
                multilinear(&table, &point),
                Some(fp2(member(case, "value"))),
                "n = {}",
                point.len()
            );
        }
    }

    /// The fold is `Σ_x values[x]·eq(point, x)`, and takes exactly `2^m` values for `m ≤ 32`.
    #[test]
    fn the_fold_is_the_sum_against_eq_and_takes_only_whole_tables() {
        let mut rng = SplitMix64::new(0x6f70_0061_0001);
        for vars in 0..=9 {
            let values: Vec<Fp2> = (0..1 << vars).map(|_| random(&mut rng)).collect();
            let point: Vec<Fp2> = (0..vars).map(|_| random(&mut rng)).collect();
            let expected = values.iter().enumerate().fold(Fp2::ZERO, |sum, (x, &v)| {
                sum + v * eq(&point, &bits(x as u64, vars)).unwrap()
            });
            assert_eq!(
                multilinear(&values, &point),
                Some(expected),
                "{vars} variables"
            );
            if vars > 0 {
                assert_eq!(multilinear(&values[1..], &point), None);
                assert_eq!(multilinear(&values, &point[1..]), None);
            }
        }
        assert_eq!(multilinear(&[Fp2::ONE], &[Fp2::ONE; 33]), None);
    }

    /// `spec/vectors/poly/poly.json` 'eq_tables': `eq(point, x)` for every `x` of up to 10 bits,
    /// with weight 1 and with a random weight.
    #[test]
    fn the_eq_tables_give_the_poly_vectors_eq_tables() {
        let file = read_vectors("poly/poly.json");
        let cases = list(&file, "eq_tables");
        assert_eq!(cases.len(), 11);
        let weight = Fp2::new(Fp::reduce(5), Fp::reduce(11));
        for case in cases {
            let point = fp2s(member(case, "point"));
            let table = fp2s(member(case, "table"));
            assert_eq!(table.len(), 1 << point.len());
            let tables = EqTables::new(&point, Fp2::ONE).unwrap();
            let weighted = EqTables::new(&point, weight).unwrap();
            for (x, &expected) in table.iter().enumerate() {
                assert_eq!(tables.at(x as u64), Some(expected), "x = {x}");
                assert_eq!(weighted.at(x as u64), Some(weight * expected), "x = {x}");
            }
            assert_eq!(tables.at(table.len() as u64), None);
        }
    }

    /// The tables give `eq`'s product formula for points of 0 to 64 coordinates, at the smallest
    /// and largest rows and random ones, and refuse a row with a bit beyond the point.
    #[test]
    fn the_eq_tables_follow_the_product_formula_up_to_64_bits() {
        let mut rng = SplitMix64::new(0x6f70_0061_0002);
        for vars in 0..=64usize {
            let point: Vec<Fp2> = (0..vars).map(|_| random(&mut rng)).collect();
            let weight = random(&mut rng);
            let tables = EqTables::new(&point, weight).unwrap();
            let top = if vars == 64 {
                u64::MAX
            } else {
                (1u64 << vars) - 1
            };
            for x in [0, top, rng.next_u64() & top, rng.next_u64() & top] {
                assert_eq!(
                    tables.at(x),
                    Some(weight * eq(&point, &bits(x, vars)).unwrap()),
                    "{vars} coordinates, x = {x}"
                );
            }
            if vars < 64 {
                assert_eq!(tables.at(top + 1), None);
                assert_eq!(tables.at(u64::MAX), None);
            }
        }
        assert!(EqTables::new(&[Fp2::ONE; 65], Fp2::ONE).is_none());
    }

    /// `spec/vectors/encoding/encoding.json`, from an independent oracle: rows of `k = 1` to 64
    /// values with masks of 0 to `k` coefficients, and their codewords of `2k` symbols on `7·L`.
    /// Both coefficients of `Fp2` are checked, as the encoding acts on each alone.
    #[test]
    fn the_encoding_gives_the_encoding_vectors_codewords() {
        let file = read_vectors("encoding/encoding.json");
        assert_eq!(member(&file, "coset_shift").as_u64(), Some(7));
        let cases = list(&file, "encoding");
        assert_eq!(cases.len(), 25);
        let lanes: [fn(Fp) -> Fp2; 2] = [Fp2::from, |x| Fp2::new(Fp::ZERO, x)];
        for case in cases {
            let k = usize::try_from(member(case, "k").as_u64().unwrap()).unwrap();
            let row: Vec<Fp> = list(case, "row").iter().map(fp).collect();
            let mask: Vec<Fp> = list(case, "mask").iter().map(fp).collect();
            let codeword: Vec<Fp> = list(case, "codeword").iter().map(fp).collect();
            assert_eq!(codeword.len(), 2 * k);
            for lane in lanes {
                let combined = CombinedRow {
                    u: row.iter().copied().map(lane).collect(),
                    mask: mask.iter().copied().map(lane).collect(),
                };
                let mut out = Vec::new();
                encode(&combined, k, 2 * k, &mut out).unwrap();
                let expected: Vec<Fp2> = codeword.iter().copied().map(lane).collect();
                assert_eq!(out, expected, "k = {k}, P = {}", mask.len());
            }
        }
    }

    /// The values `spec/vectors/pcs/opening.json` records for each honest opening are the checks'
    /// own: each combined row's coefficients, the leaves, each combined row's codeword at the
    /// queries, and the queried points `7·ω_n^j`.
    #[test]
    fn the_opening_vectors_recorded_values_are_the_checks_own() {
        let file = read_vectors("pcs/opening.json");
        for case in list(&file, "openings") {
            let name = member(case, "name").as_str().unwrap();
            let number = |key: &str| u32::try_from(member(case, key).as_u64().unwrap()).unwrap();
            let digest = |json: &Json| -> Digest {
                tv_testkit::hex(json.as_str().unwrap()).try_into().unwrap()
            };
            let segments: Vec<Vec<u64>> = list(case, "segments")
                .iter()
                .map(|s| {
                    s.as_array()
                        .unwrap()
                        .iter()
                        .map(|r| r.as_u64().unwrap())
                        .collect()
                })
                .collect();
            let groups: Vec<Group> = list(case, "groups")
                .iter()
                .map(|g| Group {
                    column_point: fp2s(member(g, "column_point")),
                    claims: list(g, "claims")
                        .iter()
                        .map(|c| Claim {
                            row_point: fp2s(member(c, "row_point")),
                            gamma: fp2(member(c, "gamma")),
                            value: fp2(member(c, "value")),
                        })
                        .collect(),
                })
                .collect();
            let xi = fp2s(member(case, "xi_point"));
            let opening = Opening {
                params: Params {
                    log2_k: number("log2_k"),
                    log2_inv_rate: number("log2_inv_rate"),
                    mask_coeffs: number("mask_coeffs"),
                    row_block: number("row_block"),
                    commitment_id: digest(member(case, "commitment_id")),
                },
                layout: Layout {
                    row_vars: number("row_vars"),
                    segments: &segments,
                },
                root: digest(member(case, "root")),
                groups: &groups,
                xi_point: &xi,
            };
            let sizes = Sizes::of(&opening).unwrap();
            let combined: Vec<CombinedRow> = list(case, "combined")
                .iter()
                .map(|row| CombinedRow {
                    u: fp2s(member(row, "u")),
                    mask: fp2s(member(row, "mask")),
                })
                .collect();
            let mut queries: Vec<u64> = list(case, "queries")
                .iter()
                .map(|q| q.as_u64().unwrap())
                .collect();
            queries.sort_unstable();
            queries.dedup();
            let columns: Vec<Vec<Fp>> = list(case, "columns")
                .iter()
                .map(|c| c.as_array().unwrap().iter().map(fp).collect())
                .collect();
            let recorded = member(case, "intermediate");

            let siblings: Vec<_> = list(case, "siblings").iter().map(digest).collect();
            let reached = std::cell::Cell::new(false);
            let fault = ColumnCheck {
                root: &opening.root,
                combined: &combined,
                sizes,
            }
            .check(
                &queries,
                &columns,
                &siblings,
                |j, column| leaf(&opening, sizes.block, j, column),
                |index, out| {
                    reached.set(true);
                    row_coefficients(&opening, index, out, sizes.rows)?;
                    out.pop();
                    Ok(())
                },
            );
            assert!(
                reached.get(),
                "the authenticated fixture reaches coefficient construction"
            );
            assert_eq!(
                fault,
                Err(OpeningError::Internal),
                "a short internal row buffer is never zipped away"
            );

            let mut coefficients = Vec::new();
            let mut codeword = Vec::new();
            for (index, row) in combined.iter().enumerate() {
                row_coefficients(&opening, index, &mut coefficients, sizes.rows).unwrap();
                assert_eq!(
                    coefficients,
                    fp2s(&list(recorded, "row_coefficients")[index]),
                    "{name}: row {index}'s coefficients"
                );
                encode(row, sizes.k, sizes.n, &mut codeword).unwrap();
                let at_queries: Vec<Fp2> = queries.iter().map(|&j| codeword[j as usize]).collect();
                assert_eq!(
                    at_queries,
                    fp2s(&list(recorded, "combined_at_queries")[index]),
                    "{name}: row {index} at the queries"
                );
            }
            for ((&j, column), expected) in
                queries.iter().zip(&columns).zip(list(recorded, "leaves"))
            {
                assert_eq!(
                    leaf(&opening, sizes.block, j, column).unwrap(),
                    digest(expected),
                    "{name}: leaf {j}"
                );
            }
            let omega = root_of_unity(sizes.leaves.trailing_zeros()).unwrap();
            let points: Vec<Fp> = queries.iter().map(|&j| Fp::SEVEN * omega.pow(j)).collect();
            let recorded_points: Vec<Fp> = list(recorded, "points").iter().map(fp).collect();
            assert_eq!(points, recorded_points, "{name}: the points");
        }
    }

    #[test]
    fn faults_of_the_proof_are_rejections_and_the_others_inconclusive() {
        let rejections = [
            (OpeningError::Evaluation { group: None }, "REJ-PCS-EVAL"),
            (OpeningError::Evaluation { group: Some(3) }, "REJ-PCS-EVAL"),
            (OpeningError::MerklePath { column: 7 }, "REJ-PCS-MERKLE"),
            (OpeningError::Proximity { column: 7 }, "REJ-PCS-PROXIMITY"),
        ];
        for (error, code) in rejections {
            assert!(error.is_rejection(), "{error}");
            assert_eq!(error.code(), code);
        }
        assert!(!OpeningError::Memory.is_rejection());
        assert_eq!(OpeningError::Memory.code(), "ERR-RES-MEMORY");
        let faults = [
            OpeningError::CodeLength {
                log2_k: 40,
                log2_inv_rate: 1,
            },
            OpeningError::MaskCoefficients { mask_coeffs: 0 },
            OpeningError::RowBlock,
            OpeningError::RowVars { row_vars: 65 },
            OpeningError::NoSegments,
            OpeningError::RowOrder {
                segment: 1,
                position: 0,
            },
            OpeningError::RowRange {
                segment: 0,
                position: 2,
            },
            OpeningError::ColumnPoint {
                group: 0,
                length: 3,
            },
            OpeningError::RowPoint {
                group: 0,
                claim: 1,
                length: 3,
            },
            OpeningError::XiPoint { length: 0 },
            OpeningError::NoQueries,
            OpeningError::QueryRange { position: 4 },
            OpeningError::Internal,
        ];
        for error in faults {
            assert!(!error.is_rejection(), "{error}");
            assert_eq!(error.code(), "ERR-INTERNAL", "{error}");
        }
    }
}
