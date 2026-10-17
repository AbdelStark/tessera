// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Throughput of the reference commitment and opening, single-threaded over Goldilocks:
//!
//! - `commit`: encoding, chaining, merging and the Merkle tree, in nanoseconds per committed
//!   element (each of the `N·k` values of the non-zero rows);
//! - `open`: the combined rows of 16 groups of 40 claims in all and the proximity row, in
//!   nanoseconds per opened element (each of the `N·k` values combined into all `G + 1` rows)
//!   and per product of an `Fp2` coefficient by an `Fp` value, `(G + 1)·N·(k + P_A)` of them;
//! - `answer`: 155 queries, `τ_A` of `live-64`, in nanoseconds per answered symbol;
//! - `drop`: dropping a commitment, which clears its rows, masks and codewords (05 §3 OBS-6), in
//!   nanoseconds per committed element.
//!
//! The rows lie in 8 segments of consecutive rows, with `P_A = min(200, k)` and `B = 32`.
//! `cargo bench -p tessera-pcs --bench commitment` runs it and reports the fastest of several
//! runs. Without `--bench`, as under `cargo test --benches`, it runs once at small sizes as a
//! smoke test.

#![forbid(unsafe_code)]
// A benchmark reports on stdout.
#![allow(clippy::print_stdout)]

use std::hint::black_box;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use tessera_codec::Secret;
use tessera_field::{Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_pcs::{ClaimGroup, CommitmentParams, ReferenceCommitment, RowClaim, RowLayout};

type Fp = Goldilocks;
type Fp2 = GoldilocksExt2;

/// SplitMix64: reproducible rows, mask coefficients, points and queries.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn fp(&mut self) -> Fp {
        Fp::from_u64_reduced(self.next_u64())
    }

    fn fp2(&mut self) -> Fp2 {
        let c0 = self.fp();
        Fp2::new(c0, self.fp())
    }

    fn fp2s(&mut self, n: usize) -> Vec<Fp2> {
        (0..n).map(|_| self.fp2()).collect()
    }
}

/// The fastest of `runs` runs of `run`, each of which times what it measures.
fn fastest(runs: usize, mut run: impl FnMut() -> Duration) -> Duration {
    (0..runs).map(|_| run()).min().unwrap_or_default()
}

fn report(what: &str, log_k: u32, rows: usize, count: usize, unit: &str, time: Duration) {
    let ns = time.as_secs_f64() * 1e9 / count as f64;
    let ms = time.as_secs_f64() * 1e3;
    println!("{what:<8} k = 2^{log_k:<2} N = {rows:<5} {ns:>8.3} ns per {unit:<17} {ms:>9.2} ms");
}

/// `rows` consecutive rows in 8 segments.
fn layout(rows: usize) -> RowLayout {
    let row_vars = (rows.max(2) as u64).next_power_of_two().trailing_zeros();
    let per_segment = rows.div_ceil(8);
    let segments = (0..8)
        .map(|s| (s * per_segment).min(rows)..((s + 1) * per_segment).min(rows))
        .map(|range| range.map(|r| r as u64).collect())
        .collect();
    RowLayout::new(row_vars, segments).expect("a valid layout")
}

fn run(rng: &mut SplitMix64, log_k: u32, rows: usize, runs: usize) {
    let k = 1usize << log_k;
    let params = CommitmentParams {
        log2_k: log_k,
        mask_coeffs: k.min(200),
        row_block: NonZeroUsize::new(32).expect("non-zero"),
        commitment_id: [0x54; 32],
    };
    let values: Vec<Fp> = (0..rows * k).map(|_| rng.fp()).collect();
    let mut commitment = None;
    let time = fastest(runs, || {
        let input = values.clone();
        let layout = layout(rows);
        let mut masks = SplitMix64(0x7E55_E7A0_0054_0B01);
        let start = Instant::now();
        let committed =
            ReferenceCommitment::commit(params, layout, Secret::new(input), &mut || masks.fp())
                .expect("valid inputs");
        let elapsed = start.elapsed();
        commitment = Some(committed);
        elapsed
    });
    report("commit", log_k, rows, rows * k, "committed element", time);
    let commitment = commitment.expect("committed at least once");

    // 16 groups of 40 claims in all, then the proximity row.
    let row_vars = commitment.layout().row_vars() as usize;
    let groups: Vec<ClaimGroup> = (0..16)
        .map(|g| ClaimGroup {
            column_point: rng.fp2s(log_k as usize),
            claims: (0..if g < 8 { 3 } else { 2 })
                .map(|_| RowClaim {
                    row_point: rng.fp2s(row_vars),
                    gamma: rng.fp2(),
                })
                .collect(),
        })
        .collect();
    let xi = rng.fp2s(row_vars);
    let time = fastest(runs, || {
        let start = Instant::now();
        black_box(
            commitment
                .open(black_box(&groups), &xi)
                .expect("valid points"),
        );
        start.elapsed()
    });
    report("open", log_k, rows, rows * k, "opened element", time);
    let products = 17 * rows * (k + params.mask_coeffs);
    report("  same", log_k, rows, products, "Fp2·Fp product", time);

    let queries: Vec<u64> = (0..155).map(|_| rng.next_u64() % (2 * k as u64)).collect();
    let mut distinct = queries.clone();
    distinct.sort_unstable();
    distinct.dedup();
    let time = fastest(runs, || {
        let start = Instant::now();
        black_box(
            commitment
                .answer(black_box(&queries))
                .expect("valid queries"),
        );
        start.elapsed()
    });
    report(
        "answer",
        log_k,
        rows,
        distinct.len() * rows,
        "answered symbol",
        time,
    );

    drop(commitment);
    let time = fastest(runs, || {
        let mut masks = SplitMix64(0x7E55_E7A0_0054_0B02);
        let committed = ReferenceCommitment::commit(
            params,
            layout(rows),
            Secret::new(values.clone()),
            &mut || masks.fp(),
        )
        .expect("valid inputs");
        let start = Instant::now();
        drop(black_box(committed));
        start.elapsed()
    });
    report("drop", log_k, rows, rows * k, "committed element", time);
}

fn main() {
    let mut rng = SplitMix64(0x7E55_E7A0_0054_0B00);
    if std::env::args().any(|arg| arg == "--bench") {
        run(&mut rng, 10, 1024, 5);
        run(&mut rng, 14, 256, 5);
        run(&mut rng, 18, 32, 3);
    } else {
        run(&mut rng, 6, 20, 1);
    }
}
