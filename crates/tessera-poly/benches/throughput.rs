// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Throughput of the CPU prover's table operations, in nanoseconds per element, single-threaded
//! over Goldilocks and its quadratic extension:
//!
//! - binding the lowest variable of an `Fp` table into an `Fp2` table, and of an `Fp2` table in
//!   place, per input entry;
//! - building an `eq` table, per output entry;
//! - evaluating `Fp` and `Fp2` tables at an `Fp2` point, per table entry, next to the successive
//!   binding the evaluation replaces;
//! - evaluating a round polynomial, per call.
//!
//! `cargo bench -p tessera-poly` runs it at 2^20 and 2^24 entries and reports the fastest of
//! several runs. Without `--bench`, as under `cargo test --benches`, it runs once at 2^10 as a
//! smoke test.

#![forbid(unsafe_code)]
// A benchmark reports on stdout.
#![allow(clippy::print_stdout)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use tessera_codec::SecretBuf;
use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_poly::{
    RoundPoly, bind_lowest, bind_lowest_in_place, eq_table, evaluate, evaluate_base,
};

type Fp = Goldilocks;
type Fp2 = GoldilocksExt2;

/// SplitMix64: reproducible inputs.
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
        Fp2::new(self.fp(), self.fp())
    }
}

/// The time `f` takes.
fn timed<T>(f: impl FnOnce() -> T) -> Duration {
    let start = Instant::now();
    black_box(f());
    start.elapsed()
}

/// The fastest of `runs` runs of `run`, each of which times what it measures.
fn fastest(runs: usize, mut run: impl FnMut() -> Duration) -> Duration {
    (0..runs).map(|_| run()).min().unwrap_or_default()
}

fn report(what: &str, log_n: u32, elements: usize, time: Duration) {
    let ns = time.as_secs_f64() * 1e9 / elements as f64;
    println!("{what:<44} 2^{log_n:<3} {ns:>7.3} ns per entry");
}

fn run(log_n: u32, runs: usize) {
    let n = 1usize << log_n;
    let mut rng = SplitMix64(0x7E55_E7A0_0046_0001 ^ u64::from(log_n));
    let base: Vec<Fp> = (0..n).map(|_| rng.fp()).collect();
    let ext: Vec<Fp2> = (0..n).map(|_| rng.fp2()).collect();
    let point: Vec<Fp2> = (0..log_n).map(|_| rng.fp2()).collect();
    let r = point[0];
    let mut work: SecretBuf<Fp2> = SecretBuf::with_capacity(n);

    let time = fastest(runs, || timed(|| bind_lowest(black_box(&base), r)));
    report("bind_lowest, Fp table into Fp2", log_n, n, time);

    let time = fastest(runs, || {
        work.clear();
        work.extend(ext.iter().copied());
        timed(|| bind_lowest_in_place(black_box(&mut work), r))
    });
    report("bind_lowest_in_place, Fp2 table", log_n, n, time);

    let time = fastest(runs, || timed(|| eq_table(black_box(&point))));
    report("eq_table", log_n, n, time);

    let time = fastest(runs, || timed(|| evaluate_base(black_box(&base), &point)));
    report("evaluate_base, Fp table at an Fp2 point", log_n, n, time);
    // The method evaluation replaces: one binding per variable.
    let time = fastest(runs, || {
        timed(|| {
            let mut bound = bind_lowest(black_box(&base), point[0]).expect("a variable");
            for &r in &point[1..] {
                bind_lowest_in_place(&mut bound, r).expect("a variable");
            }
            bound[0]
        })
    });
    report("  by successive binding, for comparison", log_n, n, time);

    let time = fastest(runs, || timed(|| evaluate(black_box(&ext), &point)));
    report("evaluate, Fp2 table at an Fp2 point", log_n, n, time);
    let time = fastest(runs, || {
        work.clear();
        work.extend(ext.iter().copied());
        timed(|| {
            for &r in &point {
                bind_lowest_in_place(black_box(&mut work), r).expect("a variable");
            }
            work[0]
        })
    });
    report("  by successive binding, for comparison", log_n, n, time);
}

fn round_polys(runs: usize) {
    let mut rng = SplitMix64(0x7E55_E7A0_0046_0002);
    for degree in [3, 7] {
        let polys: Vec<RoundPoly<Fp2>> = (0..1024)
            .map(|_| {
                let evaluations: Vec<Fp2> = (0..degree).map(|_| rng.fp2()).collect();
                RoundPoly::new(&evaluations).expect("a supported degree")
            })
            .collect();
        let (claim, r) = (rng.fp2(), rng.fp2());
        let time = fastest(runs, || {
            timed(|| {
                black_box(&polys)
                    .iter()
                    .fold(Fp2::ZERO, |sum, p| sum + p.evaluate(claim, r))
            })
        });
        let ns = time.as_secs_f64() * 1e9 / polys.len() as f64;
        let what = format!("RoundPoly::evaluate, degree {degree}");
        println!("{what:<44}       {ns:>7.3} ns per call");
    }
}

/// Keeps one core busy for `duration` before anything is timed, so that the first measurement
/// does not run on a core that is still raising its clock.
fn warm_up(duration: Duration) {
    let mut rng = SplitMix64(0x7E55_E7A0_0046_0003);
    let table: Vec<Fp2> = (0..1 << 16).map(|_| rng.fp2()).collect();
    let start = Instant::now();
    while start.elapsed() < duration {
        black_box(evaluate(black_box(&table), &[rng.fp2(); 16]).expect("matching lengths"));
    }
}

fn main() {
    if std::env::args().any(|arg| arg == "--bench") {
        warm_up(Duration::from_millis(500));
        run(20, 20);
        run(24, 5);
        round_polys(50);
    } else {
        run(10, 1);
        round_polys(1);
    }
}
