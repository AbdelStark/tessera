// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0054 P2b, including regeneration copies and clearing, on public synthetic inputs.

use std::time::Duration;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use tessera_codec::{Ascii, Secret};
use tessera_field::{Field, Goldilocks as B, GoldilocksExt2 as F, PrimeField};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_sumcheck::{Merge, MergeClaim, MergePolynomial, MergeTables, prove_merge_two_pass};

fn fp(value: usize) -> F {
    F::from(B::from_u64_reduced(value as u64))
}

fn evaluate(values: &[B], point: &[F]) -> F {
    values.iter().enumerate().fold(F::ZERO, |sum, (i, &v)| {
        let weight = point.iter().enumerate().fold(F::ONE, |e, (j, &r)| {
            e * if i >> j & 1 == 0 { F::ONE - r } else { r }
        });
        sum + weight * F::from(v)
    })
}

fn bench(c: &mut Criterion) {
    let context = SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("test-small/1").unwrap()),
        statement_digest: [66; 32],
        session_id: [69; 16],
    };
    let mut group = c.benchmark_group("merge-p2b");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    for (name, heights, cut) in [
        ("uniform-cut0", &[14][..], 0),
        ("uniform-cut4", &[14][..], 4),
        ("mixed-cut4", &[14, 13, 12][..], 4),
    ] {
        let shapes: Vec<_> = (0..12)
            .map(|i| {
                let vars = heights[i % heights.len()];
                MergePolynomial {
                    vars,
                    group_vars: vec![vars - 1],
                }
            })
            .collect();
        let values: Vec<Vec<_>> = shapes
            .iter()
            .enumerate()
            .map(|(k, p)| {
                (0..1 << p.vars)
                    .map(|i| B::from_u64_reduced(((13 * i + 5 * k) % 127) as u64))
                    .collect()
            })
            .collect();
        let claims = (0..2)
            .map(|h| {
                shapes
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        let point: Vec<_> = (0..p.free_vars()).map(|j| fp(3 + 5 * j)).collect();
                        let mut natural = point.clone();
                        natural.push(fp(h));
                        MergeClaim {
                            polynomial: i,
                            point,
                            group_bits: h as u64,
                            value: evaluate(&values[i], &natural),
                        }
                    })
                    .collect()
            })
            .collect();
        let merge = Merge::new(shapes, claims, vec![fp(3), fp(7)], fp(11)).unwrap();
        let mut tables = MergeTables::new(merge.polynomials(), 2, cut).unwrap();
        for h in 0..2 {
            let blocks: Vec<_> = values
                .iter()
                .map(|v| Secret::new(v[h * (v.len() / 2)..(h + 1) * (v.len() / 2)].to_vec()))
                .collect();
            tables
                .add_group(h, &merge.claims()[h], merge.gammas()[h], |i, _| {
                    Some(&blocks[i])
                })
                .unwrap();
        }
        group.bench_function(name, |b| {
            b.iter(|| {
                black_box(
                    prove_merge_two_pass(&mut Transcript::new(&context), &merge, &tables, |i| {
                        Secret::new(values[i].clone())
                    })
                    .unwrap(),
                )
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
