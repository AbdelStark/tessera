// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Original complete-model claims for the canonical PCS checks (RFC-0011/0037/0052).
//! Packing claims discharges no opening or external token/greedy obligation.

use crate::advice::{self, Source, opening as canonical};
use crate::claims::ClaimList;
use crate::gadgets::model::{External, Reduced};
use crate::lowering::{Graph, Kind, Role};
use crate::schedule::{Commitment, Layout as Template, ScheduleError};
use crate::stages::{PoolClaim, PoolTensor};
use crate::weights::{self, Matrix};
use tv_field::Fp2;

/// A fault in verifier-derived opening metadata, never a proof rejection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// Inconsistent original source, point, layer weights or matrix (`ERR-INTERNAL`).
    Internal,
    /// A metadata reservation failed (`ERR-RES-MEMORY`).
    Memory,
}
type Result<T> = core::result::Result<T, Error>;
fn reserve<T>(n: usize) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| Error::Memory)?;
    Ok(v)
}
fn copy<T: Copy>(p: &[T]) -> Result<Vec<T>> {
    let mut v = reserve(p.len())?;
    v.extend_from_slice(p);
    Ok(v)
}
fn push<T>(v: &mut Vec<T>, value: T) -> Result<()> {
    v.try_reserve(1).map_err(|_| Error::Memory)?;
    v.push(value);
    Ok(())
}
fn template_error(e: ScheduleError) -> Error {
    match e {
        ScheduleError::Memory => Error::Memory,
        _ => Error::Internal,
    }
}
fn weight_error(e: weights::Error) -> Error {
    match e {
        weights::Error::Memory => Error::Memory,
        _ => Error::Internal,
    }
}
fn basis(point: &[Fp2], l: usize) -> Fp2 {
    point.iter().enumerate().fold(Fp2::ONE, |w, (j, &r)| {
        w * if (l >> j) & 1 == 0 { Fp2::ONE - r } else { r }
    })
}
fn bits(n: usize) -> Result<usize> {
    usize::try_from(usize::BITS.saturating_sub(n.saturating_sub(1).leading_zeros()))
        .map_err(|_| Error::Internal)
}
struct Advice {
    point: Vec<Fp2>,
    pieces: Vec<canonical::Piece>,
    value: Fp2,
}
struct Weight {
    column: Vec<Fp2>,
    row: Vec<Fp2>,
    value: Fp2,
}
/// Every original complete-model opening claim, with duplicate obligations retained.
/// Advice combinations precede the native/fixed sources and multiplicities. Matrix claims
/// keep their pool order, including all ChannelParams reading sites. No result is Accept.
#[must_use = "check every canonical advice/weight opening and the retained external bindings"]
pub struct Claims<'a> {
    advice: Vec<Advice>,
    weights: [Vec<Weight>; 10],
    external: Option<External<'a>>,
}
impl<'a> Claims<'a> {
    /// Pack the EXACT complete reducer output, using the same validated graph/claim list and
    /// canonical layouts as the bound commitments. No proof supplies source positions.
    /// Arbitrary advice weights are retained; a weight claim must retain its original equality
    /// point and unit-coefficient weight tensor, as RFC-0012/0052 prescribe. Draft RFC-0053's
    /// same-point weight aliases remain separate value obligations for the evaluation check.
    ///
    /// # Errors
    /// `Internal` for inconsistent own metadata, `Memory` for failed reservations.
    pub fn derive(
        graph: &Graph,
        claims: &ClaimList,
        advice: &advice::Layout<'_>,
        layouts: &[weights::Layout; 10],
        reduced: Reduced<'a>,
    ) -> Result<Self> {
        let template = Template::of(graph).map_err(template_error)?;
        for (layout, matrix) in layouts.iter().zip(Matrix::ALL) {
            if layout.matrix() != matrix {
                return Err(Error::Internal);
            }
        }
        let mut out = Self {
            advice: Vec::new(),
            weights: core::array::from_fn(|_| Vec::new()),
            external: reduced.external,
        };
        for claim in reduced.graph.pool {
            if claim.terms.is_empty() {
                return Err(Error::Internal);
            }
            match claim.commitment {
                Commitment::Advice => {
                    let mut pieces = Vec::new();
                    for term in claim.terms {
                        let tensors = match term.tensor {
                            PoolTensor::Single(t) => {
                                let mut v = reserve(1)?;
                                v.push(t);
                                v
                            }
                            PoolTensor::Stacked(t) => template.family(t).map_err(template_error)?,
                        };
                        if tensors.len() != term.weights.len() {
                            return Err(Error::Internal);
                        }
                        for (t, coefficient) in tensors.into_iter().zip(term.weights) {
                            push(
                                &mut pieces,
                                canonical::Piece {
                                    source: Source::Tensor(t),
                                    coefficient,
                                },
                            )?;
                        }
                    }
                    out.advice(advice, claim.point, pieces, claim.value)?;
                }
                Commitment::Weight(_) | Commitment::ChannelParams => {
                    out.weight(graph, claims, layouts, &template, claim)?
                }
            }
        }
        for claim in reduced.sources {
            let count = claim.sources.len();
            if (claim.layer_point.is_empty() && count != 1)
                || (!claim.layer_point.is_empty()
                    && (count
                        != usize::try_from(claims.arch.n_layers).map_err(|_| Error::Internal)?
                        || claim.layer_point.len() != bits(count)?))
            {
                return Err(Error::Internal);
            }
            let mut pieces = reserve(count)?;
            for (l, source) in claim.sources.into_iter().enumerate() {
                pieces.push(canonical::Piece {
                    source,
                    coefficient: basis(&claim.layer_point, l),
                });
            }
            out.advice(advice, claim.point, pieces, claim.value)?;
        }
        for &(table, ref claim) in reduced.multiplicities {
            let mut pieces = reserve(1)?;
            pieces.push(canonical::Piece {
                source: Source::Multiplicity(table),
                coefficient: Fp2::ONE,
            });
            out.advice(advice, copy(&claim.point)?, pieces, claim.value)?;
        }
        Ok(out)
    }
    fn advice(
        &mut self,
        layout: &advice::Layout<'_>,
        point: Vec<Fp2>,
        pieces: Vec<canonical::Piece>,
        value: Fp2,
    ) -> Result<()> {
        for p in &pieces {
            // Only validate source/arity here; the PCS checks later consume ALL its images.
            let _ = layout.claim(p.source, &point).map_err(|e| match e {
                advice::PointError::Memory => Error::Memory,
                _ => Error::Internal,
            })?;
        }
        push(
            &mut self.advice,
            Advice {
                point,
                pieces,
                value,
            },
        )
    }
    fn weight(
        &mut self,
        graph: &Graph,
        claims: &ClaimList,
        layouts: &[weights::Layout; 10],
        template: &Template,
        claim: PoolClaim,
    ) -> Result<()> {
        let [term] = claim.terms.as_slice() else {
            return Err(Error::Internal);
        };
        let (tensor, layer) = match term.tensor {
            PoolTensor::Single(t) => {
                if term.weights.as_slice() != [Fp2::ONE] || term.layer_point.is_some() {
                    return Err(Error::Internal);
                }
                (t, &[][..])
            }
            PoolTensor::Stacked(t) => {
                let family = template.family(t).map_err(template_error)?;
                let layer = term.layer_point.as_deref().ok_or(Error::Internal)?;
                if layer.len() != bits(family.len())?
                    || term.weights.len() != family.len()
                    || term
                        .weights
                        .iter()
                        .enumerate()
                        .any(|(l, &w)| w != basis(layer, l))
                {
                    return Err(Error::Internal);
                }
                (t, layer)
            }
        };
        if graph.tensor(tensor).is_none_or(|t| t.kind != Kind::Weight) {
            return Err(Error::Internal);
        }
        let role = graph
            .inputs()
            .iter()
            .find(|i| i.tensor == tensor)
            .map(|i| i.role)
            .ok_or(Error::Internal)?;
        let matrix = match (claim.commitment, role) {
            (Commitment::Weight(ty), Role::Weight(w)) if w.ty == ty => Matrix::of_weight(ty),
            (Commitment::ChannelParams, Role::Classes(_) | Role::Multipliers(_)) => {
                Matrix::ChannelParams
            }
            _ => return Err(Error::Internal),
        };
        let index = usize::from(matrix.tag());
        let layout = layouts.get(index).ok_or(Error::Internal)?;
        let (column, row) = if let Role::Classes(w) | Role::Multipliers(w) = role {
            layout
                .channel_point(claims, w.ty, &claim.point, layer)
                .map_err(weight_error)?
        } else {
            let mut full = copy(&claim.point)?;
            full.try_reserve(layer.len()).map_err(|_| Error::Memory)?;
            full.extend_from_slice(layer);
            let cv = usize::try_from(layout.log2_k()).map_err(|_| Error::Internal)?;
            let rv = usize::try_from(layout.row_vars()).map_err(|_| Error::Internal)?;
            if full.len() != cv.checked_add(rv).ok_or(Error::Internal)? {
                return Err(Error::Internal);
            }
            (
                copy(full.get(..cv).ok_or(Error::Internal)?)?,
                copy(full.get(cv..).ok_or(Error::Internal)?)?,
            )
        };
        let bucket = self.weights.get_mut(index).ok_or(Error::Internal)?;
        if matrix != Matrix::ChannelParams
            && bucket
                .first()
                .is_some_and(|prior| prior.column != column || prior.row != row)
        {
            return Err(Error::Internal);
        }
        push(
            bucket,
            Weight {
                column,
                row,
                value: claim.value,
            },
        )
    }
    /// Original advice claims for the canonical evaluation and column checks, in fixed order.
    ///
    /// # Errors
    /// `Memory` if the borrowed claim descriptors cannot be reserved.
    pub fn advice_claims(&self) -> Result<Vec<canonical::Claim<'_>>> {
        let mut out = reserve(self.advice.len())?;
        for c in &self.advice {
            out.push(canonical::Claim {
                point: &c.point,
                pieces: &c.pieces,
                value: c.value,
            });
        }
        Ok(out)
    }
    /// One matrix's original claims in pool order, scaled by its already drawn opening gamma.
    ///
    /// # Errors
    /// `Memory` if the borrowed claim descriptors cannot be reserved.
    pub fn weight_claims(&self, matrix: Matrix, gamma: Fp2) -> Result<Vec<weights::Claim<'_>>> {
        let bucket = self
            .weights
            .get(usize::from(matrix.tag()))
            .ok_or(Error::Internal)?;
        let mut out = reserve(bucket.len())?;
        for c in bucket {
            out.push(weights::Claim {
                column_point: &c.column,
                row_point: &c.row,
                gamma,
                value: c.value,
            });
        }
        Ok(out)
    }
    /// `e_T`, the claims `matrix` has in the pool, aliases at one point included (RFC-0065 'The
    /// count'); 0 for a matrix no claim opens.
    pub fn claim_count(&self, matrix: Matrix) -> usize {
        self.weights
            .get(usize::from(matrix.tag()))
            .map_or(0, Vec::len)
    }
    /// Matrices required by the original reduction claims, in canonical order.
    /// A proof's matrix list never determines which commitments must be opened.
    pub fn matrices(&self) -> impl Iterator<Item = Matrix> + '_ {
        Matrix::ALL
            .into_iter()
            .zip(&self.weights)
            .filter_map(|(matrix, claims)| (!claims.is_empty()).then_some(matrix))
    }
    /// Legacy token/logit debts, when the plan used external boundaries. Runtime plans return
    /// `None` only after routing those sources through native and graph reductions; all
    /// canonical openings in this plan remain owed.
    pub fn external(&self) -> Option<External<'a>> {
        self.external
    }
}
