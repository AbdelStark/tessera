// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Canonical CPU advice rows lent to P1, P4 and P5 (RFC-0011/0047/0050).

use core::fmt;
use tessera_codec::SecretBuf;
use tessera_compiler::{AdviceSource, Lowering, ModelAdviceLayout};
use tessera_field::{Field, Goldilocks, PrimeField};
use tessera_pcs::{PcsError, Placement};
use tessera_witness::{TileWitness, WitnessError, WitnessSource};

/// Why canonical witness rows could not be supplied to a PCS pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// Inconsistent own tile, source or domain (`ERR-INTERNAL`).
    Internal,
    /// A clearing row/column buffer could not be reserved (`ERR-RES-MEMORY`).
    Memory,
    /// The canonical tile source failed, preserving its public error.
    Witness(WitnessError),
    /// The PCS pass or placement failed, preserving its public error.
    Pcs(PcsError),
    /// The opening or column pass failed, preserving its public error.
    Opening(tessera_pcs::AdviceOpeningError),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Internal => {
                f.write_str("canonical advice rows disagree with their public metadata")
            }
            Self::Memory => f.write_str("canonical advice rows do not fit in memory"),
            Self::Witness(e) => e.fmt(f),
            Self::Pcs(e) => e.fmt(f),
            Self::Opening(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for Error {}
impl From<PcsError> for Error {
    fn from(e: PcsError) -> Self {
        Self::Pcs(e)
    }
}
impl From<tessera_pcs::AdviceOpeningError> for Error {
    fn from(e: tessera_pcs::AdviceOpeningError) -> Self {
        Self::Opening(e)
    }
}
fn zeros<T: Copy>(n: usize, value: T) -> Result<SecretBuf<T>, Error> {
    let mut v = SecretBuf::try_with_capacity(n).map_err(|_| Error::Memory)?;
    v.resize(n, value);
    Ok(v)
}

/// Lend every committed row of `segment`, in canonical order, to an existing streaming
/// commitment/opening/column pass. The same layout and witness source supply all three passes.
/// The callback follows the secret borrower rules: it retains no borrowed value after the call.
///
/// Table multiplicities use bounded clearing chunks; ordinary columns pad one tile's column
/// in clearing storage. No secret sparsity changes which rows are emitted.
///
/// # Errors
/// `Internal` for inconsistent own metadata, `Memory` for failed reservations, or the exact
/// `Witness`/`Pcs`/`Opening` error. Discard a pass after it returns an error.
pub fn visit_rows(
    lowering: &Lowering,
    layout: &ModelAdviceLayout<'_>,
    source: &(impl WitnessSource + ?Sized),
    segment: u32,
    mut sink: impl FnMut(u64, &[Goldilocks]) -> Result<(), Error>,
) -> Result<(), Error> {
    if layout.layout().segments().get(segment as usize).is_none() {
        return Err(Error::Internal);
    }
    let width = 1usize
        .checked_shl(layout.layout().params().log2_k)
        .ok_or(Error::Memory)?;
    for (index, tile) in layout.tiles().iter().enumerate() {
        let placement = layout.layout().tiles().get(index).ok_or(Error::Internal)?;
        if placement.segment != segment {
            continue;
        }
        let witness = source.tile(tile).map_err(Error::Witness)?;
        for &name in layout.sources(index).ok_or(Error::Internal)? {
            let place = layout.location(index, name).ok_or(Error::Internal)?;
            if let AdviceSource::Multiplicity(_) = name {
                table_rows(
                    place,
                    witness.multiplicities().ok_or(Error::Internal)?,
                    width,
                    &mut sink,
                )?;
                continue;
            }
            let (values, domain) = column(lowering, tile, &witness, name)?;
            let expected = domain
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d as usize))
                .ok_or(Error::Memory)?;
            if values.len() != expected {
                return Err(Error::Internal);
            }
            // shortcut: materialize one padded tile column; stream ordinary columns under #134.
            let count = usize::try_from(place.row_count())
                .ok()
                .and_then(|n| n.checked_mul(width))
                .ok_or(Error::Memory)?;
            let mut rows = zeros(count, Goldilocks::ZERO)?;
            let mut coords = vec![0u64; domain.len()];
            for (flat, &value) in values.iter().enumerate() {
                let mut rest = flat;
                for (coordinate, &len) in coords.iter_mut().zip(&domain).rev() {
                    *coordinate = (rest % len as usize) as u64;
                    rest /= len as usize;
                }
                let (row, column) = place.locate(&coords).map_err(Error::Pcs)?;
                let offset = row
                    .checked_sub(place.first_row())
                    .and_then(|r| r.checked_mul(width as u64))
                    .and_then(|r| r.checked_add(column))
                    .and_then(|r| usize::try_from(r).ok())
                    .ok_or(Error::Internal)?;
                *rows.get_mut(offset).ok_or(Error::Internal)? = Goldilocks::from_i64(value);
            }
            for row in place.nonzero().iter().cloned().flatten() {
                let start = usize::try_from(row - place.first_row())
                    .ok()
                    .and_then(|r| r.checked_mul(width))
                    .ok_or(Error::Internal)?;
                sink(row, rows.get(start..start + width).ok_or(Error::Internal)?)?;
            }
        }
    }
    Ok(())
}
fn table_rows(
    place: &Placement,
    column: &tessera_witness::MultiplicityColumn,
    width: usize,
    sink: &mut impl FnMut(u64, &[Goldilocks]) -> Result<(), Error>,
) -> Result<(), Error> {
    if Some(column.rows()) != 1u64.checked_shl(place.vars()) {
        return Err(Error::Internal);
    }
    let capacity = column.rows().min(width.max(1 << 16) as u64) as usize;
    let mut counts = zeros(capacity, 0u64)?;
    let mut row = zeros(width, Goldilocks::ZERO)?;
    // RFC-0011 rule 4 omits whole rows of public table-tree padding.
    for run in place.nonzero() {
        let mut start = (run.start - place.first_row()) * width as u64;
        let end = ((run.end - place.first_row()) * width as u64).min(column.rows());
        while start < end {
            let count =
                usize::try_from((end - start).min(capacity as u64)).map_err(|_| Error::Memory)?;
            counts.resize(count, 0);
            column
                .read_rows(start, &mut counts)
                .map_err(Error::Witness)?;
            for (i, chunk) in counts.chunks(width).enumerate() {
                row.fill(Goldilocks::ZERO);
                for (out, &v) in row.iter_mut().zip(chunk) {
                    *out = Goldilocks::from_u64_reduced(v);
                }
                let (at, col) = place
                    .locate(&[start + (i * width) as u64])
                    .map_err(Error::Pcs)?;
                if col != 0 {
                    return Err(Error::Internal);
                }
                sink(at, &row)?;
            }
            start += count as u64;
        }
    }
    Ok(())
}
fn column<'a>(
    lowering: &Lowering,
    tile: &tessera_compiler::SequenceTile,
    values: &'a TileWitness,
    source: AdviceSource,
) -> Result<(&'a [i64], Vec<u32>), Error> {
    if let AdviceSource::GreedyMaximum(id) = source {
        let (actual, data) = values.greedy_maximum().ok_or(Error::Internal)?;
        if actual != id || tile.tile != (tessera_compiler::Tile::GreedyMaximum { gadget: id.0 }) {
            return Err(Error::Internal);
        }
        let domain = lowering
            .graph
            .tensors
            .get(lowering.tokens.0 as usize)
            .ok_or(Error::Internal)?
            .shape
            .clone();
        return Ok((data, domain));
    }
    let (data, tensor) = match source {
        AdviceSource::Tensor(t) => {
            let i = tile
                .shares
                .iter()
                .position(|s| s.tensor == t)
                .ok_or(Error::Internal)?;
            (
                values
                    .shares()
                    .get(i)
                    .map(|v| &v[..])
                    .ok_or(Error::Internal)?,
                t,
            )
        }
        AdviceSource::Auxiliary(node, role) => {
            let node = lowering
                .graph
                .nodes
                .get(node.0 as usize)
                .ok_or(Error::Internal)?;
            let output = usize::from(matches!(node.op, tessera_ir::Op::SoftmaxRow { .. }));
            (
                values
                    .gadget()
                    .iter()
                    .find(|(r, _)| *r == role)
                    .map(|(_, v)| &v[..])
                    .ok_or(Error::Internal)?,
                *node.outputs.get(output).ok_or(Error::Internal)?,
            )
        }
        AdviceSource::Multiplicity(_) | AdviceSource::GreedyMaximum(_) => {
            return Err(Error::Internal);
        }
    };
    let share = tile
        .shares
        .iter()
        .find(|s| s.tensor == tensor)
        .ok_or(Error::Internal)?;
    let domain = share.ranges.iter().map(|r| r.end - r.start).collect();
    Ok((data, domain))
}
