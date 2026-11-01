// SPDX-License-Identifier: Apache-2.0 OR MIT

//! CPU throughput in nanoseconds per element over Goldilocks. Transforms are scalar;
//! row encoding accepts `--threads=N` (default 1):
//!
//! - `ntt` and `intt` at 2^16, 2^20 and 2^23 points, per point;
//! - `coset_ntt` and `coset_intt` at 2^23 points, per point;
//! - `encode_row` at `k = 2^22` with `P = 200` mask coefficients, per row value (the committed
//!   element of RFC-0020's `c_c`) and per codeword symbol;
//! - the one-time precomputation of an `NttPlan` of 2^23 points and a `RowEncoder` for
//!   `k = 2^22`, per twiddle-table entry.
//!
//! `cargo bench -p tessera-pcs` reports RFC-0020 statistics (3 warm-ups and 10 samples),
//! repeating once when the coefficient of variation exceeds 3%. It prints raw samples and
//! retains any discarded first run. Without
//! `--bench`, as under `cargo test --benches`, it runs once at small sizes as a smoke test.

#![forbid(unsafe_code)]
// A benchmark reports on stdout.
#![allow(clippy::print_stdout)]

use core::num::NonZeroUsize;
use std::hint::black_box;
use std::time::{Duration, Instant};

use tessera_bench::Statistics;
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

/// Nanoseconds per run, with a noisy first run retained when it was repeated.
struct Measurement {
    samples: Vec<f64>,
    rejected: Option<Vec<f64>>,
}

fn measure(runs: usize, mut run: impl FnMut() -> Duration) -> Measurement {
    let mut sample = || {
        for _ in 0..if runs > 1 { 3 } else { 0 } {
            run();
        }
        (0..runs)
            .map(|_| run().as_secs_f64() * 1e9)
            .collect::<Vec<_>>()
    };
    let first = sample();
    if runs > 1 && Statistics::of(&first).expect("finite timings").noisy() {
        Measurement {
            samples: sample(),
            rejected: Some(first),
        }
    } else {
        Measurement {
            samples: first,
            rejected: None,
        }
    }
}

fn report(what: &str, log_n: u32, elements: usize, unit: &str, time: &Measurement) {
    let stats = Statistics::of(&time.samples).expect("finite timings");
    let ns = stats.median / elements as f64;
    let ms = stats.median / 1e6;
    println!("{what:<30} 2^{log_n:<3} {ns:>7.3} ns per {unit:<15} {ms:>9.2} ms median");
    println!(
        "  min={:.2} ms max={:.2} ms CV={:.2}%{} samples_ns={:?}",
        stats.min / 1e6,
        stats.max / 1e6,
        stats.cv * 100.,
        if stats.noisy() {
            " NOISY: repeat on an idle host"
        } else {
            ""
        },
        time.samples
    );
    if let Some(first) = &time.rejected {
        println!("  discarded noisy first run, samples_ns={first:?}");
    }
}

/// A transform of `values`, timed on a fresh copy of `input` each run.
fn transform(
    runs: usize,
    input: &[Fp],
    mut f: impl FnMut(&mut [Fp]) -> Result<(), tessera_pcs::PcsError>,
) -> Measurement {
    let mut values = input.to_vec();
    measure(runs, || {
        values.copy_from_slice(input);
        let start = Instant::now();
        f(black_box(&mut values)).expect("matching length");
        black_box(&values);
        start.elapsed()
    })
}

fn transforms(rng: &mut SplitMix64, log_n: u32, runs: usize, coset: bool) {
    let n = 1usize << log_n;
    let plan = NttPlan::new(n).expect("a supported size");
    let input = rng.fps(n);
    let time = transform(runs, &input, |v| plan.ntt(v));
    report("ntt", log_n, n, "point", &time);
    let time = transform(runs, &input, |v| plan.intt(v));
    report("intt", log_n, n, "point", &time);
    if coset {
        let time = transform(runs, &input, |v| plan.coset_ntt(v, COSET_SHIFT));
        report("coset_ntt, shift 7", log_n, n, "point", &time);
        let time = transform(runs, &input, |v| plan.coset_intt(v, COSET_SHIFT));
        report("coset_intt, shift 7", log_n, n, "point", &time);
    }
}

fn paired_rows(
    scalar: &RowEncoder,
    parallel: &RowEncoder,
    row: &[Fp],
    runs: usize,
) -> (Measurement, Measurement) {
    let mut a = vec![Fp::default(); scalar.codeword_len()];
    let mut b = a.clone();
    let mut rejected = None;
    for attempt in 0..2 {
        let mut times = [Vec::with_capacity(runs), Vec::with_capacity(runs)];
        let warm = if runs > 1 { 3 } else { 0 };
        for pair in 0..warm + runs {
            let mut elapsed = [0.; 2];
            for which in if pair % 2 == 0 { [0, 1] } else { [1, 0] } {
                let mut masks = SplitMix64(0x7e55_e7a0_0047_0402);
                let (encoder, out) = if which == 0 {
                    (scalar, &mut a)
                } else {
                    (parallel, &mut b)
                };
                let start = Instant::now();
                encoder
                    .encode_row(black_box(row), &mut || masks.fp(), black_box(&mut *out))
                    .expect("matching lengths");
                black_box(&*out);
                elapsed[which] = start.elapsed().as_secs_f64() * 1e9;
            }
            assert_eq!(a, b, "complete paired codewords");
            if pair >= warm {
                times[0].push(elapsed[0]);
                times[1].push(elapsed[1]);
            }
        }
        if attempt == 0
            && runs > 1
            && times
                .iter()
                .any(|x| Statistics::of(x).expect("finite times").noisy())
        {
            rejected = Some(times);
            continue;
        }
        let [before, after] = times;
        let (old_before, old_after) = rejected.map_or((None, None), |[a, b]| (Some(a), Some(b)));
        return (
            Measurement {
                samples: before,
                rejected: old_before,
            },
            Measurement {
                samples: after,
                rejected: old_after,
            },
        );
    }
    unreachable!("the second paired sample is always reported")
}

fn encoding(
    rng: &mut SplitMix64,
    log_k: u32,
    mask_coeffs: usize,
    runs: usize,
    threads: NonZeroUsize,
) {
    let k = 1usize << log_k;
    let encoder = RowEncoder::new(k, 1, mask_coeffs)
        .expect("valid parameters")
        .threads(threads);
    let row = rng.fps(k);
    if threads.get() > 1 {
        let scalar = encoder.clone().threads(NonZeroUsize::MIN);
        let (before, after) = paired_rows(&scalar, &encoder, &row, runs);
        report("encode_row scalar paired", log_k, k, "row value", &before);
        report(
            &format!("encode_row threads={threads} paired"),
            log_k,
            k,
            "row value",
            &after,
        );
        report("  parallel codeword", log_k + 1, 2 * k, "symbol", &after);
        return;
    }
    let mut codeword = vec![Fp::default(); 2 * k];
    let mut masks = SplitMix64(0x7E55_E7A0_0047_0402);
    let time = measure(runs, || {
        let start = Instant::now();
        encoder
            .encode_row(black_box(&row), &mut || masks.fp(), &mut codeword)
            .expect("matching lengths");
        black_box(&codeword);
        start.elapsed()
    });
    let what = format!("encode_row, P = {mask_coeffs}, threads={threads}");
    report(&what, log_k, k, "row value", &time);
    report("  the same", log_k + 1, 2 * k, "codeword symbol", &time);
}

fn precomputation(log_n: u32, runs: usize) {
    let n = 1usize << log_n;
    let time = measure(runs, || {
        let start = Instant::now();
        black_box(NttPlan::new(n).expect("a supported size"));
        start.elapsed()
    });
    // Two tables of n/2 entries.
    report("NttPlan::new", log_n, n, "table entry", &time);
    let k = n / 2;
    let time = measure(runs, || {
        let start = Instant::now();
        black_box(RowEncoder::new(k, 1, 200).expect("valid parameters"));
        start.elapsed()
    });
    // Tables of k/2, k and k entries.
    report(
        "RowEncoder::new, P = 200",
        log_n - 1,
        k * 5 / 2,
        "table entry",
        &time,
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
    let threads = std::env::args()
        .find_map(|a| a.strip_prefix("--threads=").map(str::to_owned))
        .map_or(NonZeroUsize::MIN, |v| {
            v.parse().expect("positive --threads=N")
        });
    let mut rng = SplitMix64(0x7E55_E7A0_0047_0401);
    if std::env::args().any(|arg| arg == "--bench") {
        warm_up(Duration::from_millis(500));
        transforms(&mut rng, 16, 10, false);
        transforms(&mut rng, 20, 10, false);
        transforms(&mut rng, 23, 10, true);
        encoding(&mut rng, 22, 200, 10, threads);
        precomputation(23, 10);
    } else {
        transforms(&mut rng, 10, 1, true);
        encoding(&mut rng, 9, 200, 1, threads);
        precomputation(10, 1);
    }
}
