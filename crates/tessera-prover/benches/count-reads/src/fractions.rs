// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Public table/count fixture around the production fraction-row regeneration kernel.

use criterion::{BenchmarkId, Criterion, Throughput};
use std::{
    error::Error,
    hint::black_box,
    num::NonZeroUsize,
    path::Path,
    time::{Duration, Instant},
};
use tessera_codec::SecretBuf;
use tessera_field::{Field, Goldilocks as B, GoldilocksExt2 as F, PrimeField, QuadraticExtension};
use tessera_poly::PolyField;
use tessera_sumcheck::{FractionTreeRows, FractionTreeSource, WindowError};
use tessera_tables::{GeneratedTable, Table, TableKind};
use tessera_witness::TileWitness;

fn buffer<T: Copy + Send>(len: usize, value: T) -> Result<SecretBuf<T>, WindowError> {
    let mut out = SecretBuf::try_with_capacity(len).map_err(|_| WindowError::Allocation)?;
    out.resize(len, value);
    Ok(out)
}

struct Fractions {
    tile: TileWitness,
    table: GeneratedTable,
    dense_counts: Option<SecretBuf<u64>>,
    alpha: F,
    beta: F,
}

impl FractionTreeSource<F> for Fractions {
    fn height(&self) -> usize {
        self.table.log2_rows() as usize
    }

    fn read_scratch_bytes(&self, rows: usize) -> Option<u64> {
        (rows as u64).checked_mul(32)
    }

    fn read(&self, first: usize, n: &mut [F], d: &mut [F]) -> Result<(), WindowError> {
        if n.len() != d.len()
            || first
                .checked_add(n.len())
                .is_none_or(|end| end > 1usize << self.height())
        {
            return Err(WindowError::Source);
        }
        // Match the production table adapter's base buffers and raw-count lifetime.
        let mut keys = buffer(n.len(), B::ZERO)?;
        let mut values = buffer(n.len(), B::ZERO)?;
        let mut counts = buffer(n.len(), B::ZERO)?;
        let valued = !matches!(self.table.spec().kind, TableKind::Range { .. });
        {
            let mut raw = buffer(n.len(), 0u64)?;
            if let Some(dense) = &self.dense_counts {
                raw.copy_from_slice(&dense[first..first + n.len()]);
            } else {
                self.tile
                    .multiplicities()
                    .ok_or(WindowError::Source)?
                    .read_rows(first as u64, &mut raw)
                    .map_err(|_| WindowError::Source)?;
            }
            for j in 0..n.len() {
                let (key, value) = self
                    .table
                    .row((first + j) as u64)
                    .ok_or(WindowError::Source)?;
                keys[j] = key;
                values[j] = if valued { value } else { B::ZERO };
                counts[j] = B::from_u64_reduced(raw[j]);
            }
        }
        for j in 0..n.len() {
            n[j] = -F::from(counts[j]);
            let compressed = if valued {
                self.beta.mul_by_base(values[j]).add_base(keys[j])
            } else {
                F::from(keys[j])
            };
            d[j] = self.alpha - compressed;
        }
        Ok(())
    }
}

/// Alternate each baseline/candidate pair and rotate the candidate order (RFC-0020).
pub(super) fn paired(
    kind: &str,
    case: &str,
    mut run: impl FnMut(usize) -> Result<(), Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    let candidates = &super::CHUNKS[3..];
    // Pair order depends on sample alone, independently of candidate rotation (#679).
    let schedule: Vec<_> = (0..20usize)
        .flat_map(|sample| {
            (0..candidates.len()).map(move |slot| {
                (
                    sample,
                    (sample + slot) % candidates.len(),
                    sample.is_multiple_of(2),
                )
            })
        })
        .collect();
    let mut orders = vec![[0usize; 2]; candidates.len()];
    for &(_, candidate, baseline_first) in &schedule {
        orders[candidate][usize::from(baseline_first)] += 1;
    }
    assert!(
        orders.iter().all(|order| *order == [10, 10]),
        "each candidate must appear first ten times and second ten times"
    );
    for &chunk in &super::CHUNKS[2..] {
        run(chunk)?;
    }
    for (sample, candidate, baseline_first) in schedule {
        let chunk = candidates[candidate];
        let mut measured = [0u128; 2];
        let order = if baseline_first { [0, 1] } else { [1, 0] };
        for at in order {
            let start = Instant::now();
            run(if at == 0 { super::CHUNKS[2] } else { chunk })?;
            measured[at] = start.elapsed().as_nanos();
        }
        eprintln!(
            "pair kind={kind} case={case} chunk={chunk} sample={sample} before_ns={} after_ns={} baseline_first={}",
            measured[0],
            measured[1],
            order[0] == 0
        );
    }
    Ok(())
}

pub(super) fn run(root: &Path) -> Result<(), Box<dyn Error>> {
    let (tile, updates, table) = super::prepare(root, 1)?;
    let mut source = Fractions {
        tile,
        table,
        dense_counts: None,
        alpha: F::new(B::from_u64_reduced(19), B::ONE),
        beta: F::from(B::from_u64_reduced(7)),
    };
    let height = source.height();
    if source.tile.multiplicities().ok_or("table counts")?.rows() != 1u64 << height {
        return Err("generated table and count domains disagree".into());
    }
    if std::env::var_os("COUNT_BENCH_MATERIALIZED_DENSE").is_some() {
        let mut counts = buffer(1usize << height, 0u64)?;
        source
            .tile
            .multiplicities()
            .ok_or("table counts")?
            .read_rows(0, &mut counts)?;
        source.dense_counts = Some(counts);
        eprintln!(
            "storage_control=materialized-canonical-dense rows={} retained_weighted_oracle=true materialization_outside_timing=true",
            1usize << height
        );
    }
    let depth = height.saturating_sub(10).min(12);
    let nodes = 1usize << depth;
    let mut n = buffer(nodes, F::ZERO)?;
    let mut d = buffer(nodes, F::ZERO)?;
    let mut expected_n = buffer(nodes, F::ZERO)?;
    let mut expected_d = buffer(nodes, F::ZERO)?;
    let nonzero_alpha = source.alpha;
    // Nonzero extension denominators pin node order. Probe zero denominators separately.
    for zero in [false, true] {
        source.alpha = if zero {
            F::from(source.table.row(0).ok_or("row zero")?.0)
        } else {
            nonzero_alpha
        };
        source.beta = if zero {
            F::ZERO
        } else {
            F::from(B::from_u64_reduced(7))
        };
        // Compare dense-copy controls to the canonical count reader, outside timing.
        let dense = source.dense_counts.take();
        let reference =
            FractionTreeRows::new(&source, NonZeroUsize::new(65536).ok_or("chunk")?, u64::MAX)?;
        let root = reference.root()?;
        reference.read_level(depth, 0, &mut expected_n, &mut expected_d)?;
        source.dense_counts = dense;
        for chunk in super::CHUNKS {
            let rows =
                FractionTreeRows::new(&source, NonZeroUsize::new(chunk).ok_or("chunk")?, u64::MAX)?;
            assert!(
                rows.root()? == root,
                "public chunking changes a fraction root"
            );
            rows.read_level(depth, 0, &mut n, &mut d)?;
            assert!(
                n[..] == expected_n[..] && d[..] == expected_d[..],
                "public chunking changes ordered fraction nodes"
            );
        }
    }
    source.alpha = nonzero_alpha;
    source.beta = F::from(B::from_u64_reduced(7));
    for &chunk in &super::CHUNKS[2..] {
        let rows =
            FractionTreeRows::new(&source, NonZeroUsize::new(chunk).ok_or("chunk")?, u64::MAX)?;
        eprintln!(
            "fraction_chunk_rows={chunk} effective_chunk_rows={} scratch_bytes={}",
            rows.chunk_rows(),
            rows.scratch_bytes()
        );
    }
    eprintln!(
        "scope=fraction-regeneration-only height={height} depth={depth} updates={updates} node_count={nodes} adapter=public-fixture-production-buffer-lifetimes"
    );
    for case in ["root", "level"] {
        paired("fractions", case, |chunk| {
            let rows =
                FractionTreeRows::new(&source, NonZeroUsize::new(chunk).ok_or("chunk")?, u64::MAX)?;
            if case == "root" {
                black_box(rows.root()?);
            } else {
                rows.read_level(depth, 0, &mut n, &mut d)?;
                black_box((&n[..], &d[..]));
            }
            Ok(())
        })?;
    }
    let mut criterion = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .configure_from_args();
    let mut group = criterion.benchmark_group("weighted-fraction-regeneration");
    group.throughput(Throughput::Elements(1u64 << height));
    for case in ["root", "level"] {
        for &chunk in &super::CHUNKS[2..] {
            let rows =
                FractionTreeRows::new(&source, NonZeroUsize::new(chunk).ok_or("chunk")?, u64::MAX)?;
            group.bench_with_input(BenchmarkId::new(case, chunk), &chunk, |b, _| {
                b.iter(|| {
                    if case == "root" {
                        black_box(rows.root().expect("validated public source"));
                    } else {
                        rows.read_level(depth, 0, &mut n, &mut d)
                            .expect("validated public source");
                        black_box((&n[..], &d[..]));
                    }
                })
            });
        }
    }
    group.finish();
    criterion.final_summary();
    Ok(())
}
