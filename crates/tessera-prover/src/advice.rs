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

/// Batches of built rows queued ahead of the sink (P1/P4/P5 row pipeline).
const BATCHES_AHEAD: usize = 8;
/// Field elements per batch: one hand-over between threads amortized over short rows.
const BATCH_ELEMENTS: usize = 1 << 18;
/// Clearing rows `column_rows` fills at once with several threads.
const COLUMN_WORKERS: usize = 32;

/// Rows of `width` elements per queued batch.
fn batch_rows(width: usize) -> usize {
    (BATCH_ELEMENTS / width.max(1)).max(1)
}

/// Clearing field rows of `width` elements a pass holds at most: the column workers', the
/// queue's batches with the one being built and the one in the sink, and the producer's
/// working row. The memory plan charges them (`cpu::physical`).
pub(crate) fn rows_resident(width: u64) -> u64 {
    let batch = (BATCH_ELEMENTS as u64 / width.max(1)).max(1);
    (COLUMN_WORKERS as u64 + 1).saturating_add(batch.saturating_mul(BATCHES_AHEAD as u64 + 2))
}
/// Regenerated tiles a pass holds at most: the one being emitted and the next, regenerated
/// ahead. The memory plan charges them (`cpu::physical`).
pub(crate) const TILES_RESIDENT: u64 = 2;

type Sink<'a> = dyn FnMut(u64, &[Goldilocks]) -> Result<(), Error> + 'a;
/// Takes a built row; it may swap the buffer for another of the same length, whose contents
/// the producer overwrites.
type Emit<'a> = dyn FnMut(u64, &mut SecretBuf<Goldilocks>) -> Result<(), Error> + 'a;

/// Lend every committed row of `segment`, in canonical order, to an existing streaming
/// commitment/opening/column pass. The same layout and witness source supply all three passes.
/// The callback follows the secret borrower rules: it retains no borrowed value after the call.
///
/// With several threads ([`tessera_sumcheck::parallelism`]) one thread regenerates the next
/// tile while another builds rows into a bounded queue, and the caller's thread only runs the
/// sink: rows reach it in exactly the serial order, so commitments, chains and transcripts are
/// unchanged. Table multiplicities use bounded clearing chunks; ordinary columns use clearing
/// rows, at most `rows_resident` of them, with metadata proportional to the source rank. No
/// secret sparsity changes which rows are emitted.
///
/// # Errors
/// `Internal` for inconsistent own metadata, `Memory` for failed reservations, or the exact
/// `Witness`/`Pcs`/`Opening` error, the first in row order. Discard a pass after it returns an
/// error.
pub fn visit_rows(
    lowering: &Lowering,
    layout: &ModelAdviceLayout<'_>,
    source: &(impl WitnessSource + Sync + ?Sized),
    segment: u32,
    mut sink: impl FnMut(u64, &[Goldilocks]) -> Result<(), Error>,
) -> Result<(), Error> {
    if layout.layout().segments().get(segment as usize).is_none() {
        return Err(Error::Internal);
    }
    let width = 1usize
        .checked_shl(layout.layout().params().log2_k)
        .ok_or(Error::Memory)?;
    let threads = tessera_sumcheck::parallelism();
    // Public per-pass timings for the stage profile (#744): tile regeneration and row building
    // (on one thread less the sink; on several, with the producer's waits for the sink), the
    // sink alone, and the time outside it.
    let mut sunk = 0f64;
    let mut timed = |row: u64, values: &[Goldilocks]| -> Result<(), Error> {
        let at = std::time::Instant::now();
        let r = sink(row, values);
        sunk += at.elapsed().as_secs_f64();
        r
    };
    let started = std::time::Instant::now();
    let (regenerate, build) = pipeline(
        threads,
        batch_rows(width),
        |emit| tile_rows(lowering, layout, source, segment, width, threads > 1, emit),
        &mut timed,
    )?;
    let millis = |s: f64| (s * 1e3) as u64;
    let waited = (started.elapsed().as_secs_f64() - sunk).max(0.0);
    tessera_codec::event!(
        DEBUG,
        "prover.rows_visited",
        segment = segment,
        regenerate_ms = millis(regenerate),
        rows_ms = millis(if threads > 1 { build } else { build - sunk }),
        sink_ms = millis(sunk),
        waited_ms = millis(waited);
        "segment {segment}: tiles {regenerate_ms} ms, rows {rows_ms} ms, sink {sink_ms} ms, outside the sink {waited_ms} ms"
    );
    Ok(())
}

/// Runs `produce` on its own thread, feeding `sink` on this one through a queue of
/// `BATCHES_AHEAD` batches of `batch` clearing rows, in `produce`'s order; one thread runs both
/// in turn. The first error in row order wins, as on one thread: a producer error travels the
/// queue behind the rows before it, and a sink error closes the queue, which stops the producer.
fn pipeline<T: Send>(
    threads: usize,
    batch: usize,
    produce: impl FnOnce(&mut Emit<'_>) -> Result<T, Error> + Send,
    sink: &mut Sink<'_>,
) -> Result<T, Error> {
    type Batch = Vec<(u64, SecretBuf<Goldilocks>)>;
    if threads <= 1 {
        return produce(&mut |row, buffer| sink(row, buffer));
    }
    let batch = batch.max(1);
    let (batches, ready) = std::sync::mpsc::sync_channel::<Result<Batch, Error>>(BATCHES_AHEAD);
    let (spent, recycled) = std::sync::mpsc::channel::<SecretBuf<Goldilocks>>();
    std::thread::scope(|scope| {
        let producer = scope.spawn(move || {
            let mut built: Batch = Vec::with_capacity(batch);
            // The built row itself is queued, and the producer continues in a buffer the sink
            // gave back, or a new one only when none came back, so the rows live are bounded
            // by the queue (`rows_resident`).
            let mut emit = |row: u64, buffer: &mut SecretBuf<Goldilocks>| -> Result<(), Error> {
                let mut next = match recycled.try_recv() {
                    Ok(next) => next,
                    Err(_) => zeros(buffer.len(), Goldilocks::ZERO)?,
                };
                next.resize(buffer.len(), Goldilocks::ZERO);
                built.push((row, core::mem::replace(buffer, next)));
                if built.len() == batch {
                    // A closed queue means the sink failed; its error is the pass's.
                    let full = core::mem::replace(&mut built, Vec::with_capacity(batch));
                    batches.send(Ok(full)).map_err(|_| Error::Internal)?;
                }
                Ok(())
            };
            let produced = produce(&mut emit);
            if !built.is_empty() {
                let _ = batches.send(Ok(built));
            }
            if let Err(e) = produced {
                let _ = batches.send(Err(e));
            }
            produced
        });
        let mut consumed = Ok(());
        'queue: for rows in &ready {
            match rows {
                Ok(rows) => {
                    for (row, buffer) in rows {
                        let sunk = sink(row, &buffer);
                        let _ = spent.send(buffer);
                        if let Err(e) = sunk {
                            consumed = Err(e);
                            break 'queue;
                        }
                    }
                }
                Err(e) => {
                    consumed = Err(e);
                    break;
                }
            }
        }
        drop(ready);
        let produced = producer.join().unwrap_or(Err(Error::Internal));
        consumed?;
        produced
    })
}

/// Emits every row of `segment`'s tiles in canonical order, returning the seconds spent
/// waiting for tiles and building rows. With `ahead`, the next tile regenerates on its own
/// thread while this one's rows are built.
fn tile_rows(
    lowering: &Lowering,
    layout: &ModelAdviceLayout<'_>,
    source: &(impl WitnessSource + Sync + ?Sized),
    segment: u32,
    width: usize,
    ahead: bool,
    emit: &mut Emit<'_>,
) -> Result<(f64, f64), Error> {
    let mut tiles = Vec::new();
    for index in 0..layout.tiles().len() {
        let placement = layout.layout().tiles().get(index).ok_or(Error::Internal)?;
        if placement.segment == segment {
            tiles.push(index);
        }
    }
    let fetch = |index: usize| -> Result<TileWitness, Error> {
        let tile = layout.tiles().get(index).ok_or(Error::Internal)?;
        source.tile(tile).map_err(Error::Witness)
    };
    let (mut regenerate, mut build) = (0f64, 0f64);
    // Clearing row buffers shared by every placement of the pass, grown on demand.
    let mut pool: Vec<SecretBuf<Goldilocks>> = Vec::new();
    std::thread::scope(|scope| {
        let mut next: Option<std::thread::ScopedJoinHandle<'_, _>> = None;
        for (position, &index) in tiles.iter().enumerate() {
            let at = std::time::Instant::now();
            let witness = match next.take() {
                Some(job) => job.join().unwrap_or(Err(Error::Internal))?,
                None => fetch(index)?,
            };
            if let Some(&following) = tiles.get(position + 1).filter(|_| ahead) {
                next = Some(scope.spawn(move || fetch(following)));
            }
            let tile_regenerate = at.elapsed().as_secs_f64();
            regenerate += tile_regenerate;
            let at = std::time::Instant::now();
            let tile = layout.tiles().get(index).ok_or(Error::Internal)?;
            for &name in layout.sources(index).ok_or(Error::Internal)? {
                let place = layout.location(index, name).ok_or(Error::Internal)?;
                if let AdviceSource::Multiplicity(_) = name {
                    table_rows(
                        place,
                        witness.multiplicities().ok_or(Error::Internal)?,
                        width,
                        &mut &mut *emit,
                    )?;
                    continue;
                }
                let (values, domain) = column(lowering, tile, &witness, name)?;
                column_rows(place, values, &domain, width, &mut pool, &mut &mut *emit)?;
            }
            let tile_build = at.elapsed().as_secs_f64();
            build += tile_build;
            if tile_regenerate + tile_build > 1.0 {
                let millis = |s: f64| (s * 1e3) as u64;
                let (index, regenerate_ms, rows_ms) =
                    (index as u64, millis(tile_regenerate), millis(tile_build));
                tessera_codec::event!(DEBUG, "prover.slow_tile", index = index, regenerate_ms = regenerate_ms, rows_ms = rows_ms; "tile {index}: regenerate {regenerate_ms} ms, rows {rows_ms} ms");
            }
        }
        Ok((regenerate, build))
    })
}
fn column_rows(
    place: &Placement,
    values: &[i64],
    domain: &[u32],
    width: usize,
    pool: &mut Vec<SecretBuf<Goldilocks>>,
    sink: &mut impl FnMut(u64, &mut SecretBuf<Goldilocks>) -> Result<(), Error>,
) -> Result<(), Error> {
    let expected = domain
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d as usize))
        .ok_or(Error::Memory)?;
    if values.len() != expected {
        return Err(Error::Internal);
    }
    // RFC-0011/0047: one clearing row per worker is resident, including for permuted axes.
    // The public nonzero schedule is unchanged across P1, P4 and P5 (INV-PCS-1/2), and rows
    // reach the sink in that order whatever the worker count.
    // Up to COLUMN_WORKERS rows are resident, within the plan's ROWS_RESIDENT.
    let fill = |row: u64, fields: &mut [Goldilocks]| -> Result<(), Error> {
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
        Ok(())
    };
    let workers = tessera_sumcheck::parallelism().clamp(1, COLUMN_WORKERS);
    if workers == 1 {
        // One clearing row, as the memory plan charges.
        let mut fields = zeros(width, Goldilocks::ZERO)?;
        for row in place.nonzero().iter().cloned().flatten() {
            fill(row, &mut fields)?;
            sink(row, &mut fields)?;
        }
        return Ok(());
    }
    let order: Vec<u64> = place.nonzero().iter().cloned().flatten().collect();
    let workers = workers.min(order.len().max(1));
    while pool.len() < workers {
        pool.push(zeros(width, Goldilocks::ZERO)?);
    }
    let rows = &mut pool[..workers];
    for batch in order.chunks(workers) {
        if workers == 1 {
            fill(batch[0], &mut rows[0])?;
        } else {
            std::thread::scope(|scope| {
                let jobs: Vec<_> = batch
                    .iter()
                    .zip(rows.iter_mut())
                    .map(|(&row, fields)| {
                        let fill = &fill;
                        scope.spawn(move || fill(row, fields))
                    })
                    .collect();
                jobs.into_iter()
                    .try_for_each(|job| job.join().unwrap_or(Err(Error::Internal)))
            })?;
        }
        for (&row, fields) in batch.iter().zip(rows.iter_mut()) {
            sink(row, fields)?;
        }
    }
    Ok(())
}

fn table_rows(
    place: &Placement,
    column: &tessera_witness::MultiplicityColumn,
    width: usize,
    sink: &mut impl FnMut(u64, &mut SecretBuf<Goldilocks>) -> Result<(), Error>,
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
                sink(at, &mut row)?;
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
    fn row_mapping_memory_counts_retained_axis_descriptors() {
        for domain in [
            vec![],
            vec![1],
            vec![3],
            vec![1, 5, 1],
            vec![3, 4, 5],
            vec![1; 257],
        ] {
            let layout = layout(&domain, 2);
            let place = &layout.tensors()[1];
            let mut mapping = None;
            let allocations = allocation_counter::measure(|| {
                mapping = Some(place.row_indices(place.first_row(), &domain).unwrap());
            });
            let mut bytes = None;
            let sizing = allocation_counter::measure(|| bytes = place.row_mapping_memory_bytes());
            assert_eq!(sizing.count_total, 0);
            assert_eq!(
                allocations.bytes_current,
                i64::try_from(bytes.unwrap()).unwrap()
            );
            drop(mapping);
        }
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
                column_rows(
                    place,
                    &values,
                    &domain,
                    width,
                    &mut Vec::new(),
                    &mut |row, fields| {
                        assert_eq!(Some(row), ordered.next());
                        let start = (row - place.first_row()) as usize * width;
                        assert_eq!(&fields[..], &expected[start..start + width]);
                        Ok(())
                    },
                )
                .unwrap();
                assert_eq!(ordered.next(), None);
                assert_eq!(
                    column_rows(
                        place,
                        &[],
                        &domain,
                        width,
                        &mut Vec::new(),
                        &mut |_, _| panic!("invalid source emitted a row")
                    ),
                    Err(Error::Internal)
                );
                let mut calls = 0;
                assert_eq!(
                    column_rows(
                        place,
                        &values,
                        &domain,
                        width,
                        &mut Vec::new(),
                        &mut |_, _| {
                            calls += 1;
                            Err(Error::Memory)
                        }
                    ),
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
                    let result = column_rows(
                        place,
                        &values,
                        &domain,
                        32,
                        &mut Vec::new(),
                        &mut |_, fields| {
                            let address = fields.as_ptr() as usize;
                            reused &= *row_storage.get_or_insert(address) == address;
                            if fail { Err(Error::Memory) } else { Ok(()) }
                        },
                    );
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

    /// The row pipeline delivers the producer's rows in its order, with their values, whatever
    /// the thread count, and the first error in row order wins: a producer error after the rows
    /// before it, a sink error before any later row.
    #[test]
    fn pipeline_keeps_row_order_and_the_first_error_at_every_thread_count() {
        const ROWS: u64 = 200;
        let planted = Error::Witness(WitnessError::Usage { rule: "planted" });
        let produce = |fail: Option<u64>| {
            move |emit: &mut Emit<'_>| -> Result<u64, Error> {
                let mut row = zeros(5, Goldilocks::ZERO)?;
                for r in 0..ROWS {
                    if fail == Some(r) {
                        return Err(planted);
                    }
                    for (i, v) in row.iter_mut().enumerate() {
                        *v = Goldilocks::from_u64_reduced(r * 7 + i as u64);
                    }
                    emit(r * 3 + 1, &mut row)?;
                }
                Ok(ROWS)
            }
        };
        for (threads, batch) in [1, 2, 8].into_iter().flat_map(|t| [(t, 1), (t, 7)]) {
            let mut seen = Vec::new();
            let mut sink = |row: u64, values: &[Goldilocks]| {
                let r = (row - 1) / 3;
                let expected: Vec<_> = (0..5)
                    .map(|i| Goldilocks::from_u64_reduced(r * 7 + i))
                    .collect();
                assert_eq!(values, &expected[..], "row {row} at {threads} threads");
                seen.push(row);
                Ok(())
            };
            assert_eq!(pipeline(threads, batch, produce(None), &mut sink), Ok(ROWS));
            assert_eq!(seen, (0..ROWS).map(|r| r * 3 + 1).collect::<Vec<_>>());

            let mut seen = 0u64;
            let got = pipeline(threads, batch, produce(Some(150)), &mut |_, _| {
                seen += 1;
                Ok(())
            });
            assert_eq!((got, seen), (Err(planted), 150));

            let mut seen = 0u64;
            let got = pipeline(threads, batch, produce(None), &mut |row, _| {
                seen += 1;
                if row == 40 * 3 + 1 {
                    return Err(Error::Pcs(PcsError::LayoutMemory));
                }
                Ok(())
            });
            assert_eq!((got, seen), (Err(Error::Pcs(PcsError::LayoutMemory)), 41));
        }
    }
}
