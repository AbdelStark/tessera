// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Throughput of the CPU encoding, in nanoseconds per element, single-threaded over Goldilocks:
//!
//! - `ntt` and `intt` at 2^16, 2^20 and 2^23 points, per point;
//! - `coset_ntt` and `coset_intt` at 2^23 points, per point;
//! - `encode_row` at `k = 2^22` with `P = 200` mask coefficients, per row value (the committed
//!   element of RFC-0020's `c_c`) and per codeword symbol;
//! - the one-time precomputation of an `NttPlan` of 2^23 points and a `RowEncoder` for
//!   `k = 2^22`, per twiddle-table entry.
//!
//! `cargo bench -p tessera-pcs` runs it and reports the fastest of several runs. Without
//! `--bench`, as under `cargo test --benches`, it runs once at small sizes as a smoke test.

#![forbid(unsafe_code)]
// A benchmark reports on stdout.
#![allow(clippy::print_stdout)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use tessera_field::{Goldilocks, PrimeField};
use tessera_pcs::{COSET_SHIFT, NttPlan, RowEncoder};

type Fp = Goldilocks;

/// SplitMix64: reproducible inputs and mask coefficients.
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

    fn fps(&mut self, n: usize) -> Vec<Fp> {
        (0..n).map(|_| self.fp()).collect()
    }
}

/// The fastest of `runs` runs of `run`, each of which times what it measures.
fn fastest(runs: usize, mut run: impl FnMut() -> Duration) -> Duration {
    (0..runs).map(|_| run()).min().unwrap_or_default()
}

fn report(what: &str, log_n: u32, elements: usize, unit: &str, time: Duration) {
    let ns = time.as_secs_f64() * 1e9 / elements as f64;
    let ms = time.as_secs_f64() * 1e3;
    println!("{what:<30} 2^{log_n:<3} {ns:>7.3} ns per {unit:<15} {ms:>9.2} ms");
}

/// A transform of `values`, timed on a fresh copy of `input` each run.
fn transform(
    runs: usize,
    input: &[Fp],
    mut f: impl FnMut(&mut [Fp]) -> Result<(), tessera_pcs::PcsError>,
) -> Duration {
    let mut values = input.to_vec();
    fastest(runs, || {
        values.copy_from_slice(input);
        let start = Instant::now();
        f(black_box(&mut values)).expect("matching length");
        start.elapsed()
    })
}

fn transforms(rng: &mut SplitMix64, log_n: u32, runs: usize, coset: bool) {
    let n = 1usize << log_n;
    let plan = NttPlan::new(n).expect("a supported size");
    let input = rng.fps(n);
    let time = transform(runs, &input, |v| plan.ntt(v));
    report("ntt", log_n, n, "point", time);
    let time = transform(runs, &input, |v| plan.intt(v));
    report("intt", log_n, n, "point", time);
    if coset {
        let time = transform(runs, &input, |v| plan.coset_ntt(v, COSET_SHIFT));
        report("coset_ntt, shift 7", log_n, n, "point", time);
        let time = transform(runs, &input, |v| plan.coset_intt(v, COSET_SHIFT));
        report("coset_intt, shift 7", log_n, n, "point", time);
    }
}

fn encoding(rng: &mut SplitMix64, log_k: u32, mask_coeffs: usize, runs: usize) {
    let k = 1usize << log_k;
    let encoder = RowEncoder::new(k, mask_coeffs).expect("valid parameters");
    let row = rng.fps(k);
    let mut codeword = vec![Fp::default(); 2 * k];
    let mut masks = SplitMix64(0x7E55_E7A0_0047_0402);
    let time = fastest(runs, || {
        let start = Instant::now();
        encoder
            .encode_row(black_box(&row), &mut || masks.fp(), &mut codeword)
            .expect("matching lengths");
        start.elapsed()
    });
    let what = format!("encode_row, P = {mask_coeffs}");
    report(&what, log_k, k, "row value", time);
    report("  the same", log_k + 1, 2 * k, "codeword symbol", time);
}

fn precomputation(log_n: u32, runs: usize) {
    let n = 1usize << log_n;
    let time = fastest(runs, || {
        let start = Instant::now();
        black_box(NttPlan::new(n).expect("a supported size"));
        start.elapsed()
    });
    // Two tables of n/2 entries.
    report("NttPlan::new", log_n, n, "table entry", time);
    let k = n / 2;
    let time = fastest(runs, || {
        let start = Instant::now();
        black_box(RowEncoder::new(k, 200).expect("valid parameters"));
        start.elapsed()
    });
    // Tables of k/2, k and k entries.
    report(
        "RowEncoder::new, P = 200",
        log_n - 1,
        k * 5 / 2,
        "table entry",
        time,
    );
}

/// Keeps one core busy for `duration` before anything is timed, so that the first measurement
/// does not run on a core that is still raising its clock.
fn warm_up(duration: Duration) {
    let mut rng = SplitMix64(0x7E55_E7A0_0047_0403);
    let plan = NttPlan::new(1 << 16).expect("a supported size");
    let mut values = rng.fps(1 << 16);
    let start = Instant::now();
    while start.elapsed() < duration {
        plan.ntt(black_box(&mut values)).expect("matching length");
    }
}

fn main() {
    let mut rng = SplitMix64(0x7E55_E7A0_0047_0401);
    if std::env::args().any(|arg| arg == "--bench") {
        warm_up(Duration::from_millis(500));
        transforms(&mut rng, 16, 50, false);
        transforms(&mut rng, 20, 10, false);
        transforms(&mut rng, 23, 5, true);
        encoding(&mut rng, 22, 200, 5);
        precomputation(23, 3);
    } else {
        transforms(&mut rng, 10, 1, true);
        encoding(&mut rng, 9, 200, 1);
        precomputation(10, 1);
    }
}
