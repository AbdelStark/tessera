// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Release-mode throughput of `tv-hash` against the test kit's reference BLAKE3, which follows the
//! specification's structure without aiming for speed: `cargo bench -p tv-hash`.
//!
//! It reports the best of several runs, in GB/s (10^9 bytes per second), for one 64 MiB input
//! and for many 64-byte inputs (a Merkle node's size), and the other shapes the verifier hashes:
//! column steps, SHA-256 and RFC 6962 nodes. `cargo bench -p tv-hash -- quick` runs fewer
//! repetitions.

#![allow(clippy::print_stdout)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use tv_hash::{Domain, DomainHasher, hash, merkle, rfc6962, sha256};
use tv_testkit::{SplitMix64, blake3};

/// The best time of `runs` runs of `work`.
fn best(runs: u32, mut work: impl FnMut()) -> Duration {
    (0..runs)
        .map(|_| {
            let start = Instant::now();
            work();
            start.elapsed()
        })
        .min()
        .unwrap_or_default()
}

fn report(name: &str, bytes: usize, time: Duration) {
    let gbps = bytes as f64 / time.as_secs_f64() / 1e9;
    println!(
        "{name:<44} {gbps:>8.3} GB/s  ({:.3} ms)",
        time.as_secs_f64() * 1e3
    );
}

fn random_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut rng = SplitMix64::new(seed);
    (0..len).map(|_| rng.next_u64() as u8).collect()
}

fn main() {
    let quick = std::env::args().any(|arg| arg == "quick");
    let runs = if quick { 3 } else { 10 };

    // One 64 MiB input.
    let big = random_bytes(64 << 20, 1);
    let context = Domain::Envelope.context();
    assert_eq!(
        hash(Domain::Envelope, &big),
        blake3::derive_key(context, &big)
    );
    let time = best(runs, || {
        black_box(hash(Domain::Envelope, black_box(&big)));
    });
    report("tv-hash BLAKE3, one 64 MiB input", big.len(), time);
    let time = best(runs.min(3), || {
        black_box(blake3::derive_key(context, black_box(&big)));
    });
    report("tv-testkit reference, one 64 MiB input", big.len(), time);
    let time = best(runs, || {
        let mut hasher = DomainHasher::new(Domain::Envelope);
        for piece in big.chunks(4096) {
            hasher.update(piece);
        }
        black_box(hasher.finalize());
    });
    report("tv-hash BLAKE3, 64 MiB in 4 KiB updates", big.len(), time);

    // Many 64-byte inputs, the size of a Merkle node.
    let count = 1 << 20;
    let small = random_bytes(64 * count, 2);
    let time = best(runs, || {
        for input in small.chunks_exact(64) {
            black_box(hash(Domain::MerkleNode, black_box(input)));
        }
    });
    report("tv-hash BLAKE3, 2^20 inputs of 64 bytes", small.len(), time);
    let time = best(runs.min(3), || {
        for input in small.chunks_exact(64) {
            black_box(blake3::derive_key(
                Domain::MerkleNode.context(),
                black_box(input),
            ));
        }
    });
    report(
        "tv-testkit reference, 2^20 inputs of 64 bytes",
        small.len(),
        time,
    );
    let digests: Vec<[u8; 32]> = small
        .chunks_exact(32)
        .map(|d| d.try_into().expect("32 bytes"))
        .collect();
    let time = best(runs, || {
        for pair in digests.chunks_exact(2) {
            black_box(merkle::node(black_box(&pair[0]), black_box(&pair[1])));
        }
    });
    report("tv-hash merkle::node, 2^20 nodes", small.len(), time);

    // Column steps of B = 32 rows: 32 + 8 · 32 = 288 bytes each.
    let steps = random_bytes(288 * (1 << 18), 3);
    let time = best(runs, || {
        for input in steps.chunks_exact(288) {
            black_box(hash(Domain::ColumnStep, black_box(input)));
        }
    });
    report(
        "tv-hash BLAKE3, 2^18 inputs of 288 bytes",
        steps.len(),
        time,
    );

    // SHA-256 and the RFC 6962 tree.
    let time = best(runs, || {
        black_box(sha256::sha256(black_box(&big[..16 << 20])));
    });
    report("tv-hash SHA-256, one 16 MiB input", 16 << 20, time);
    let time = best(runs, || {
        for pair in digests.chunks_exact(2).take(1 << 18) {
            black_box(rfc6962::node_hash(black_box(&pair[0]), black_box(&pair[1])));
        }
    });
    report("tv-hash rfc6962::node_hash, 2^18 nodes", 64 << 18, time);
}
