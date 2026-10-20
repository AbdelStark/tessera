// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Canonical advice openings, from verifier-derived public geometry and claim obligations.
//!
//! [`Commitment::bind`] implements RFC-0019 step 4's layout binding. After stages and gadget
//! checks, [`Commitment::prepare`] groups the verifier's original combination claims, not
//! prover-supplied row maps or individual image values (RFC-0011/0037). An original claim
//! takes one gamma power across all its weighted pieces, layers and attention tiles.
//!
//! The caller validates its claim list and admits decoded messages under its resource/time
//! limits, as for [`crate::weights`]. It supplies every obligation from its stages/gadgets;
//! this component neither decodes bytes nor returns a whole-proof Accept verdict. Protocol/2
//! blinding remains separate. Prepared pieces borrow lazy public images rather than expanding
//! all tiles. Column checks retain one coefficient vector and one re-encoded row at a time.
//! Production callers obtain their geometry from [`crate::validation::Validated::advice_layout`]
//! and their encoding parameters from that same validated claim list.
//! The caller runs RFC-0014's transcript schedule: gamma follows the fixed original claims,
//! and column queries follow absorption of every combined row. These methods do not draw
//! challenges or authenticate transcript timing.

use super::{ClaimImages, Layout, PointError, Source};
use crate::claims::AdviceLayoutParams;
use crate::opening::{
    ColumnCheck, CombinedRow, EqTables, OpeningError, Params, Sizes, check_combined, merged_leaf,
};
use crate::validation::{ClaimsError, StatementField};
use tv_field::{Fp, Fp2};
use tv_hash::{Digest, Domain, DomainHasher};

/// One public piece of an original combination claim (RFC-0037 'The claim pool').
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Piece {
    /// Actual tensor, auxiliary or multiplicity column, independently located by the verifier.
    pub source: Source,
    /// Its public coefficient, including a stacked claim's weight at this actual layer.
    pub coefficient: Fp2,
}

/// One verifier-derived obligation: `sum(piece.coefficient * source(point)) = value`.
/// A combination has one claimed value and takes one batching power, not one per piece/image.
#[derive(Clone, Copy, Debug)]
pub struct Claim<'a> {
    /// Natural LSB-first coordinates shared by the combination's pieces.
    pub point: &'a [Fp2],
    /// Public pieces, in the order the verifier's obligation lists them; at least one.
    pub pieces: &'a [Piece],
    /// The original claim's value, already bound by the stage/gadget that produced it.
    pub value: Fp2,
}

/// A Round1 root bound to the verifier's canonical geometry and descriptor (RFC-0031).
/// It is immutable across the subsequent evaluation and column checks.
pub struct Commitment<'a> {
    layout: &'a Layout<'a>,
    params: Params,
    sizes: Sizes,
    root: Digest,
}

impl<'a> Commitment<'a> {
    /// Derives the advice identifier from public inputs and compares Round1's `layout_digest`.
    /// `params` comes from the validated claim list; `root` and `layout_digest` from Round1.
    /// The caller has bound the rest of Round1's statement at RFC-0019 step 4.
    ///
    /// # Errors
    /// `Internal` for incompatible verifier-derived parameters/geometry, `Memory` when the
    /// row count is not addressable, or `StatementMismatch(Layout)` for a different identifier.
    pub fn bind(
        layout: &'a Layout<'a>,
        params: AdviceLayoutParams,
        statement_digest: &Digest,
        unit: u32,
        root: Digest,
        layout_digest: Digest,
    ) -> Result<Self, ClaimsError> {
        if params.log2_k != layout.config.log_k
            || params.segments != layout.config.segments
            || params.log2_inv_rate != 1
        {
            return Err(ClaimsError::Internal);
        }
        let mut encoding = Params {
            log2_k: u32::from(params.log2_k),
            log2_inv_rate: u32::from(params.log2_inv_rate),
            mask_coeffs: params.mask_coeffs,
            row_block: params.row_block,
            commitment_id: [0; 32],
        };
        let mut sizes = Sizes::parameters(encoding).map_err(|_| ClaimsError::Internal)?;
        let rows = layout.committed_rows()?;
        sizes.rows = usize::try_from(rows).map_err(|_| ClaimsError::Memory)?;
        let mut descriptor = DomainHasher::new(Domain::CommitmentId);
        descriptor.update(&0u16.to_le_bytes());
        descriptor.update(statement_digest);
        descriptor.update(&unit.to_le_bytes());
        descriptor.update(&[params.log2_k, params.log2_inv_rate]);
        for value in [params.mask_coeffs, params.row_block, params.segments] {
            descriptor.update(&value.to_le_bytes());
        }
        descriptor.update(&rows.to_le_bytes());
        encoding.commitment_id = descriptor.finalize();
        if layout_digest != encoding.commitment_id {
            return Err(ClaimsError::StatementMismatch {
                field: StatementField::Layout,
            });
        }
        Ok(Self {
            layout,
            params: encoding,
            sizes,
            root,
        })
    }

    /// Groups original claims by column sub-point in stable first-appearance order.
    /// `gamma` is `pcs/advice/gamma`; `xi` is the row-index point from `pcs/advice/xi`.
    /// Powers restart at gamma^1 for each group, preserving original claim order (RFC-0011).
    ///
    /// # Errors
    /// `XiPoint` for the verifier's wrong proximity arity; `Internal` for an empty combination,
    /// absent source, wrong natural arity or pieces with different column points; `Memory`
    /// for a failed public-metadata reservation. No proof messages are read here.
    pub fn prepare<'s>(
        &'s self,
        claims: &[Claim<'s>],
        gamma: Fp2,
        xi: &'s [Fp2],
    ) -> Result<Opening<'s>, OpeningError> {
        let row_vars = self
            .layout
            .nu()
            .checked_sub(self.params.log2_k)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or(OpeningError::Internal)?;
        if xi.len() != row_vars {
            return Err(OpeningError::XiPoint { length: xi.len() });
        }
        let mut groups: Vec<Group<'s>> = Vec::new();
        groups
            .try_reserve_exact(claims.len())
            .map_err(|_| OpeningError::Memory)?;
        let column_vars = usize::from(self.layout.config.log_k);
        for claim in claims {
            let mut column_point: Option<Vec<Fp2>> = None;
            let mut pieces = Vec::new();
            pieces
                .try_reserve_exact(claim.pieces.len())
                .map_err(|_| OpeningError::Memory)?;
            for piece in claim.pieces {
                let mut images = self
                    .layout
                    .claim(piece.source, claim.point)
                    .map_err(point_error)?;
                let first = images
                    .next()
                    .transpose()
                    .map_err(point_error)?
                    .ok_or(OpeningError::Internal)?;
                let point = first
                    .point
                    .get(..column_vars)
                    .ok_or(OpeningError::Internal)?;
                if let Some(previous) = &column_point {
                    if previous != point {
                        return Err(OpeningError::Internal);
                    }
                } else {
                    let mut point = first.point;
                    point.truncate(column_vars);
                    column_point = Some(point);
                }
                // Whole-row allocation makes the column point identical for all repeats.
                // row_coefficient includes the first consumed image as well (INV-PCS-1).
                pieces.push(WeightedPiece {
                    images,
                    weight: piece.coefficient,
                });
            }
            let column_point = column_point.ok_or(OpeningError::Internal)?;
            let index = match groups.iter().position(|g| g.point == column_point) {
                Some(index) => index,
                None => {
                    let index = groups.len();
                    groups.push(Group {
                        point: column_point,
                        power: Fp2::ONE,
                        expected: Fp2::ZERO,
                        pieces: Vec::new(),
                    });
                    index
                }
            };
            let group = groups.get_mut(index).ok_or(OpeningError::Internal)?;
            group.power *= gamma;
            group.expected += group.power * claim.value;
            group
                .pieces
                .try_reserve_exact(pieces.len())
                .map_err(|_| OpeningError::Memory)?;
            for mut piece in pieces {
                piece.weight *= group.power;
                group.pieces.push(piece);
            }
        }
        Ok(Opening {
            commitment: self,
            groups,
            xi,
        })
    }
}

struct WeightedPiece<'a> {
    images: ClaimImages<'a, 'a, 'a>,
    weight: Fp2,
}

struct Group<'a> {
    point: Vec<Fp2>,
    power: Fp2,
    expected: Fp2,
    pieces: Vec<WeightedPiece<'a>>,
}

/// Verifier-derived groups bound to one canonical advice commitment and proximity point.
#[must_use = "the original advice claims still require evaluation and authenticated column checks"]
pub struct Opening<'a> {
    commitment: &'a Commitment<'a>,
    groups: Vec<Group<'a>>,
    xi: &'a [Fp2],
}

impl Opening<'_> {
    /// The derived number of column-point groups; the proof sends this many rows plus proximity.
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Checks combined-row shape and each original combination's batched value (RFC-0019 step 9).
    ///
    /// # Errors
    /// `Evaluation(None)` for shape, `Evaluation(Some(g))` for the first bad group, or `Internal`
    /// for inconsistent verifier metadata. Success still owes authenticated column checks.
    pub fn check_evaluations<'a>(
        &'a self,
        combined: &'a [CombinedRow],
    ) -> Result<Evaluated<'a>, OpeningError> {
        check_combined(
            self.commitment.sizes,
            combined,
            self.groups.iter().map(|g| (g.point.as_slice(), g.expected)),
        )?;
        Ok(Evaluated {
            opening: self,
            combined,
        })
    }

    fn coefficients(&self, index: usize, out: &mut Vec<Fp2>) -> Result<(), OpeningError> {
        out.clear();
        let proximity = if index == self.groups.len() {
            Some(EqTables::new(self.xi, Fp2::ONE).ok_or(OpeningError::Internal)?)
        } else {
            None
        };
        let group = if proximity.is_none() {
            Some(self.groups.get(index).ok_or(OpeningError::Internal)?)
        } else {
            None
        };
        for tile in self.commitment.layout.tiles() {
            for column in 0..tile.columns() {
                let location = tile.location(column).ok_or(OpeningError::Internal)?;
                let end = location
                    .base
                    .checked_add(location.zero_at)
                    .ok_or(OpeningError::Internal)?;
                for row in location.base..end {
                    if out.len() >= self.commitment.sizes.rows {
                        return Err(OpeningError::Internal);
                    }
                    let value = if let Some(tables) = &proximity {
                        tables.at(row).ok_or(OpeningError::Internal)?
                    } else {
                        let group = group.ok_or(OpeningError::Internal)?;
                        let mut value = Fp2::ZERO;
                        for piece in &group.pieces {
                            value += piece.weight
                                * piece.images.row_coefficient(row).map_err(point_error)?;
                        }
                        value
                    };
                    out.push(value);
                }
            }
        }
        Ok(())
    }
}

/// Combined rows that passed the canonical groups' evaluation checks.
#[must_use = "the advice claims bind to their commitment only after authenticated column checks"]
pub struct Evaluated<'a> {
    opening: &'a Opening<'a>,
    combined: &'a [CombinedRow],
}

impl Evaluated<'_> {
    /// Authenticates canonical segment-merged leaves and multiproof, then checks every combined
    /// row at each distinct query (RFC-0019 step 11). Queries are the verifier's draws.
    ///
    /// # Errors
    /// `NoQueries`/`QueryRange` for the verifier's invalid draws; `MerklePath` at the smallest
    /// queried index for shape/authentication; `Proximity` at the smallest inconsistent column;
    /// `Memory` for failed reservations or `Internal` for inconsistent verifier metadata.
    pub fn check_columns(
        self,
        queries: &[u64],
        columns: &[Vec<Fp>],
        siblings: &[Digest],
    ) -> Result<(), OpeningError> {
        let context = self.opening.commitment;
        ColumnCheck {
            root: &context.root,
            combined: self.combined,
            sizes: context.sizes,
        }
        .check(
            queries,
            columns,
            siblings,
            |j, symbols| {
                merged_leaf(
                    &context.params.commitment_id,
                    context.sizes.block,
                    j,
                    symbols,
                    (0..context.layout.config.segments).map(|segment| {
                        let rows = context
                            .layout
                            .committed_rows_in_segment(segment)
                            .map_err(|_| OpeningError::Internal)?;
                        usize::try_from(rows).map_err(|_| OpeningError::Internal)
                    }),
                )
            },
            |index, out| self.opening.coefficients(index, out),
        )
    }
}

fn point_error(error: PointError) -> OpeningError {
    match error {
        PointError::Memory => OpeningError::Memory,
        _ => OpeningError::Internal,
    }
}
