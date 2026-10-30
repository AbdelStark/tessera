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
    /// A clearing row buffer could not be reserved (`ERR-RES-MEMORY`).
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
        match e {
            PcsError::LayoutMemory => Self::Memory,
            _ => Self::Pcs(e),
        }
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
/// Table multiplicities use bounded clearing chunks; ordinary columns reuse one clearing
/// field row, with metadata proportional to the source rank. No secret sparsity changes
/// which rows are emitted.
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
            column_rows(place, values, &domain, width, &mut sink)?;
        }
    }
    Ok(())
}
fn column_rows(
    place: &Placement,
    values: &[i64],
    domain: &[u32],
    width: usize,
    sink: &mut impl FnMut(u64, &[Goldilocks]) -> Result<(), Error>,
) -> Result<(), Error> {
    let expected = domain
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d as usize))
        .ok_or(Error::Memory)?;
    if values.len() != expected {
        return Err(Error::Internal);
    }
    // RFC-0011/0047: only one clearing row is resident, including for permuted axes.
    // The public nonzero schedule is unchanged across P1, P4 and P5 (INV-PCS-1/2).
    let mut fields = zeros(width, Goldilocks::ZERO)?;
    for row in place.nonzero().iter().cloned().flatten() {
        fields.fill(Goldilocks::ZERO);
        for (column, source) in place
            .row_indices(row, domain)
            .map_err(Error::from)?
            .enumerate()
        {
            if let Some(index) = source {
                *fields.get_mut(column).ok_or(Error::Internal)? =
                    Goldilocks::from_i64(*values.get(index).ok_or(Error::Internal)?);
            }
        }
        sink(row, &fields)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use tessera_pcs::{
        AdviceLayout, Axis, DenseShape, LayoutParams, Padding, TensorShape, TileShape,
    };

    fn layout(domain: &[u32], log2_k: u32) -> AdviceLayout {
        AdviceLayout::new(
            LayoutParams {
                log2_k,
                layers: 1,
                segments: 1,
                group: 1,
                query_block: 1,
                max_context: 1,
            },
            1,
            &[TileShape {
                layer: Some(0),
                values: vec![
                    TensorShape::Dense(DenseShape::new(vec![], Padding::Zero)),
                    TensorShape::Dense(DenseShape::new(
                        domain.iter().map(|&d| Axis::new(u64::from(d))).collect(),
                        Padding::Zero,
                    )),
                ],
            }],
        )
        .unwrap()
    }

    // Frozen pre-#630 scalar scatter, kept only as a differential oracle.
    fn scatter_rows(
        place: &Placement,
        values: &[i64],
        domain: &[u32],
        width: usize,
        sink: &mut impl FnMut(u64, &[Goldilocks]) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let expected = domain
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d as usize))
            .ok_or(Error::Memory)?;
        if values.len() != expected {
            return Err(Error::Internal);
        }
        // Original eager allocation, deliberately retained only for the test oracle.
        let count = usize::try_from(place.row_count())
            .ok()
            .and_then(|n| n.checked_mul(width))
            .ok_or(Error::Memory)?;
        let mut rows = zeros(count, Goldilocks::ZERO)?;
        let mut coords = vec![0u64; domain.len()];
        for (flat, &value) in values.iter().enumerate() {
            let mut rest = flat;
            for (coordinate, &len) in coords.iter_mut().zip(domain).rev() {
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
        Ok(())
    }

    #[test]
    fn source_metadata_reservation_failure_keeps_the_memory_error_code() {
        let error = Error::from(PcsError::LayoutMemory);
        assert_eq!(error, Error::Memory);
        assert_eq!(
            crate::ProverError::from(error).code(),
            tessera_codec::error::ErrorCode::ResMemory
        );
        assert_eq!(
            crate::ProverError::from(PcsError::LayoutMemory).code(),
            tessera_codec::error::ErrorCode::ResMemory
        );
    }

    #[test]
    fn ordinary_rows_match_scatter_at_signed_boundaries_and_padding() {
        let boundaries = [
            0,
            1,
            -1,
            i64::MIN,
            i64::MAX,
            (1 << 32) - 1,
            1 << 32,
            -(1 << 32),
        ];
        for domain in [
            vec![],
            vec![1],
            vec![3],
            vec![1, 5, 1],
            vec![4, 5],
            vec![3, 5],
            vec![3, 4, 5],
            vec![17, 3],
        ] {
            for log2_k in [0, 2, 5, 8] {
                let layout = layout(&domain, log2_k);
                let place = &layout.tensors()[1];
                let width = 1 << log2_k;
                let size = domain.iter().map(|&d| d as usize).product();
                let mut values = zeros(size, 0i64).unwrap();
                for (i, v) in values.iter_mut().enumerate() {
                    *v = boundaries[i % boundaries.len()];
                }
                let mut expected =
                    zeros(place.row_count() as usize * width, Goldilocks::ZERO).unwrap();
                scatter_rows(place, &values, &domain, width, &mut |row, fields| {
                    let start = (row - place.first_row()) as usize * width;
                    expected[start..start + width].copy_from_slice(fields);
                    Ok(())
                })
                .unwrap();
                let mut ordered = place.nonzero().iter().cloned().flatten();
                column_rows(place, &values, &domain, width, &mut |row, fields| {
                    assert_eq!(Some(row), ordered.next());
                    let start = (row - place.first_row()) as usize * width;
                    assert_eq!(fields, &expected[start..start + width]);
                    Ok(())
                })
                .unwrap();
                assert_eq!(ordered.next(), None);
                assert_eq!(
                    column_rows(place, &[], &domain, width, &mut |_, _| panic!(
                        "invalid source emitted a row"
                    )),
                    Err(Error::Internal)
                );
                let mut calls = 0;
                assert_eq!(
                    column_rows(place, &values, &domain, width, &mut |_, _| {
                        calls += 1;
                        Err(Error::Memory)
                    }),
                    Err(Error::Memory)
                );
                assert_eq!(calls, 1);
            }
        }
    }

    #[test]
    fn ordinary_row_peak_is_bounded_and_storage_is_released_on_sink_failure() {
        for domain in [[32, 33], [256, 257]] {
            let layout = layout(&domain, 5);
            let place = &layout.tensors()[1];
            let values = zeros(domain[0] as usize * domain[1] as usize, 7i64).unwrap();
            for fail in [false, true] {
                let mut reused = true;
                let measured = allocation_counter::measure(|| {
                    let mut row_storage = None;
                    let result = column_rows(place, &values, &domain, 32, &mut |_, fields| {
                        let address = fields.as_ptr() as usize;
                        reused &= *row_storage.get_or_insert(address) == address;
                        if fail { Err(Error::Memory) } else { Ok(()) }
                    });
                    assert_eq!(result, if fail { Err(Error::Memory) } else { Ok(()) });
                });
                assert_eq!(
                    measured.bytes_current, 0,
                    "OBS-6: release on success and error"
                );
                assert!(
                    measured.bytes_max <= 32 * 8 + 1024,
                    "adapter peak {} exceeds one row plus bounded metadata",
                    measured.bytes_max
                );
                let emitted = if fail {
                    1
                } else {
                    place
                        .nonzero()
                        .iter()
                        .map(|r| (r.end - r.start) as usize)
                        .sum()
                };
                // Equal addresses alone can hide allocate/free cycles at the same address.
                // Allow one field allocation and one public metadata allocation per row.
                assert!(measured.count_total as usize <= emitted + 1);
                assert!(reused, "one reused clearing row");
            }
        }
    }
}
