// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0055 CPU grids, including P3 preparation, P3b folds and clearing. Source regeneration
//! is a no-op over resident synthetic inputs: this measures arithmetic, not GEMM regeneration.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use tessera_codec::Ascii;
use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_sumcheck::{
    Instance, Prefix, PreparedWindow, Protocol, Table, Term, WindowError, WindowInstance,
    WindowPass, WindowSource, WindowTable, prove_batch,
};

type F = GoldilocksExt2;
struct Source(Vec<Vec<F>>);
impl WindowSource<F> for Source {
    fn regenerate(&mut self, _: WindowPass) -> Result<(), WindowError> {
        Ok(())
    }
    fn read(&self, id: usize, start: usize, out: &mut [F]) -> Result<(), WindowError> {
        out.copy_from_slice(&self.0[id][start..start + out.len()]);
        Ok(())
    }
}
fn fp(n: usize) -> F {
    F::from(Goldilocks::from_u64_reduced(n as u64))
}
fn bench(c: &mut Criterion) {
    let vars = 16;
    let mut source = Source(
        (0..2)
            .map(|k| {
                (0..1 << vars)
                    .map(|i| fp((3 * i + 7 * k + 1) % 127))
                    .collect()
            })
            .collect(),
    );
    let point: Vec<_> = (0..vars).map(|j| fp(5 * j + 3)).collect();
    let eq: Vec<_> = (0..1 << vars)
        .map(|i| {
            point.iter().enumerate().fold(F::ONE, |v, (j, &r)| {
                v * if i >> j & 1 == 0 { F::ONE - r } else { r }
            })
        })
        .collect();
    let spec = WindowInstance {
        vars,
        tables: vec![
            WindowTable {
                id: 0,
                public: false,
            },
            WindowTable {
                id: 1,
                public: false,
            },
        ],
        terms: vec![Term::new(F::ONE, vec![0, 1])],
        eq: point.into_iter().map(Some).collect(),
        windowed: true,
    };
    let context = SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("test-small/1").unwrap()),
        statement_digest: [45; 32],
        session_id: [59; 16],
    };
    let mut group = c.benchmark_group("65536-entry-product");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    group.bench_function("resident", |b| {
        b.iter(|| {
            let instance = Instance::new(vec![
                Table::ext(&source.0[0]),
                Table::ext(&source.0[1]),
                Table::ext(&eq).public(),
            ]);
            let _ = black_box(
                prove_batch(
                    &mut Transcript::new(&context),
                    Protocol::One,
                    Prefix::Stage(1),
                    3,
                    &[instance],
                )
                .unwrap(),
            );
        })
    });
    for cut in 3..=8 {
        group.bench_with_input(BenchmarkId::new("grid", cut), &cut, |b, &cut| {
            b.iter(|| {
                let prepared =
                    PreparedWindow::new(&mut source, Protocol::One, 3, vec![spec.clone()], cut)
                        .unwrap();
                let _ = black_box(
                    prepared
                        .prove(
                            &mut Transcript::new(&context),
                            Prefix::Stage(1),
                            &mut source,
                        )
                        .unwrap(),
                );
            })
        });
    }
    group.finish();
}
criterion_group!(benches, bench);
criterion_main!(benches);
