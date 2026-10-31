// SPDX-License-Identifier: Apache-2.0 OR MIT

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use std::time::Duration;
use tessera_hash::MerkleTree;

fn bench(c: &mut Criterion) {
    let n = 1usize << 16;
    let leaves = (0..n)
        .map(|i| {
            let mut digest = [0xA5u8; 32];
            digest[..8].copy_from_slice(&(i as u64).to_le_bytes());
            digest
        })
        .collect();
    let tree = MerkleTree::new(leaves).unwrap();
    let cases = [
        ("one", vec![n as u64 - 1]),
        (
            "sparse8",
            (0..8).map(|i| (i * 337 + 17) % n as u64).collect(),
        ),
        (
            "sparse193",
            (0..193).map(|i| (i * 337 + 17) % n as u64).collect(),
        ),
        ("cluster193", (0..193).collect()),
        ("all", (0..n as u64).collect()),
    ];
    let mut group = c.benchmark_group("merkle_open");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    for (name, indices) in &cases {
        let proof = tree.open(indices).unwrap();
        eprintln!(
            "PUBLIC merkle case={name} leaves={n} draws={} siblings={} capacity={}",
            indices.len(),
            proof.siblings.len(),
            proof.siblings.capacity()
        );
        group.bench_function(*name, |b| {
            b.iter(|| black_box(tree.open(black_box(indices)).unwrap()))
        });
    }
    group.finish();
}
criterion_group!(benches, bench);
criterion_main!(benches);
