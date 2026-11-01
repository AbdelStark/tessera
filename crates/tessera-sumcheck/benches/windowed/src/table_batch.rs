// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Complete RFC-0054 table batch, including roots, regeneration, all levels and clearing.
//! Synthetic public rows use non-certifying test parameters; no model proof is measured.

use std::{cell::Cell, num::NonZeroUsize, time::Duration};

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use tessera_codec::Ascii;
use tessera_field::{Goldilocks as B, GoldilocksExt2 as F, PrimeField, QuadraticExtension};
use tessera_hash::{Domain, Mode, ParamSetId, SessionContext, Transcript, hash};
use tessera_logup::{ProvenTables, TableBatchBudget, TableSource, prove_table_batch_bounded};
use tessera_sumcheck::WindowError;

struct Source {
    rows: usize,
    reads: Cell<u64>,
    cache_depth: Option<usize>,
    chunk: usize,
}

impl TableSource<B> for Source {
    fn rows(&self) -> usize {
        self.rows
    }
    fn fraction_cache_depth(&self) -> Option<usize> {
        self.cache_depth
    }
    fn has_values(&self) -> bool {
        true
    }
    fn read(
        &self,
        first: usize,
        keys: &mut [B],
        values: &mut [B],
        counts: &mut [B],
    ) -> Result<(), WindowError> {
        if keys.len() != values.len()
            || keys.len() != counts.len()
            || first
                .checked_add(keys.len())
                .is_none_or(|end| end > self.rows)
        {
            return Err(WindowError::Source);
        }
        self.reads.set(self.reads.get() + 1);
        for j in 0..keys.len() {
            let row = (first + j) as u64;
            keys[j] = B::from_u64_reduced(row);
            values[j] = -B::from_u64_reduced(3 * row + 7);
            counts[j] = B::from_u64_reduced(5 * row + 3);
        }
        Ok(())
    }
}

fn run(source: &Source) -> ProvenTables<F> {
    let context = SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("test-small/1").unwrap()),
        statement_digest: [47; 32],
        session_id: [61; 16],
    };
    prove_table_batch_bounded(
        &mut Transcript::new(&context),
        F::from(B::from_u64_reduced(1_000_003)),
        F::from(B::from_u64_reduced(53)),
        &[source],
        TableBatchBudget {
            allowance: u64::MAX,
            source_bytes: size_of::<Source>() as u64,
            chunk_rows: NonZeroUsize::new(source.chunk).unwrap(),
        },
    )
    .unwrap()
}

fn digest(proof: &ProvenTables<F>) -> String {
    let mut bytes = Vec::new();
    let mut append = |x: &F| {
        for c in x.coeffs() {
            bytes.extend_from_slice(&c.to_canonical_u64().to_le_bytes());
        }
    };
    for value in proof.roots() {
        append(value);
    }
    for level in proof.levels() {
        for value in level.proof().rounds.iter().flatten() {
            append(value);
        }
        for value in &level.proof().final_evals {
            append(value);
        }
    }
    for claim in proof.multiplicities() {
        for value in &claim.point {
            append(value);
        }
        append(&claim.value);
    }
    hash(Domain::Table, &bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("bounded-table-batch");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    for height in [14, 16] {
        let source = Source {
            rows: 1 << height,
            reads: Cell::new(0),
            cache_depth: None,
            chunk: 256,
        };
        let proof = run(&source);
        println!(
            "fixture height={height} chunk=256 reads={} digest={}",
            source.reads.get(),
            digest(&proof)
        );
        group.bench_with_input(
            BenchmarkId::new("rows", source.rows),
            &source,
            |b, source| {
                b.iter(|| {
                    source.reads.set(0);
                    let _ = black_box(run(source));
                });
            },
        );
    }
    group.finish();
    let mut group = c.benchmark_group("bounded-table-cache");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    for height in [14, 18] {
        let mut expected = None;
        for cache_depth in [None, Some(height - 4)] {
            let source = Source {
                rows: 1 << height,
                reads: Cell::new(0),
                cache_depth,
                chunk: 65_536,
            };
            let actual = digest(&run(&source));
            if let Some(ref expected) = expected {
                assert_eq!(&actual, expected);
            } else {
                expected = Some(actual.clone());
            }
            println!(
                "cache fixture height={height} depth={} chunk=65536 reads={} digest={actual}",
                cache_depth.unwrap_or(0),
                source.reads.get()
            );
            group.bench_with_input(
                BenchmarkId::new(format!("h{height}"), cache_depth.unwrap_or(0)),
                &source,
                |b, source| {
                    b.iter(|| {
                        source.reads.set(0);
                        let _ = black_box(run(source));
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
