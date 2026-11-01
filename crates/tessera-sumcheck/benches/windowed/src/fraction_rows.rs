// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Synthetic bounded fraction regeneration; no model or certifying proof is timed.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use std::num::NonZeroUsize;
use tessera_field::{Goldilocks as B, GoldilocksExt2 as F, PrimeField, QuadraticExtension};
use tessera_hash::{Domain, hash};
use tessera_sumcheck::{FractionTreeRows, FractionTreeSource, WindowError};

struct Source;
impl FractionTreeSource<F> for Source {
    fn height(&self) -> usize {
        20
    }
    fn read(&self, first: usize, n: &mut [F], d: &mut [F]) -> Result<(), WindowError> {
        for (i, (n, d)) in n.iter_mut().zip(d).enumerate() {
            let x = (first + i) as u64;
            *n = F::new(
                B::from_u64_reduced(x.wrapping_mul(0x9e37_79b9_7f4a_7c15)),
                B::from_u64_reduced(x ^ 0xabcd_ef01),
            );
            *d = F::new(
                B::from_u64_reduced(x.wrapping_mul(0xbf58_476d_1ce4_e5b9)),
                B::from_u64_reduced(x ^ 0x9876_5432),
            );
        }
        Ok(())
    }
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("bounded-fraction-root");
    group.sample_size(20);
    for chunk in [1, 17, 256, 4096] {
        let rows =
            FractionTreeRows::new(&Source, NonZeroUsize::new(chunk).unwrap(), u64::MAX).unwrap();
        let root = rows.root().unwrap();
        let bytes: Vec<_> = [root.0, root.1]
            .iter()
            .flat_map(|f| f.coeffs())
            .flat_map(|x| x.to_canonical_u64().to_le_bytes())
            .collect();
        let digest: String = hash(Domain::Table, &bytes)
            .iter()
            .map(|x| format!("{x:02x}"))
            .collect();
        println!("public fixture height=20 chunk={chunk} root_digest={digest}");
        group.bench_with_input(BenchmarkId::new("chunk", chunk), &rows, |b, r| {
            b.iter(|| black_box(r.root().unwrap()));
        });
    }
    group.finish();
}
criterion_group!(benches, bench);
criterion_main!(benches);
