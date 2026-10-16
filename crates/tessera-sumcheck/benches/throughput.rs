// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Throughput of the batched sumcheck prover, in nanoseconds per table entry, single-threaded
//! over Goldilocks and its quadratic extension:
//!
//! - single products of `k = 1, 2, 3, 4` tables over `Fp`, of degree `k`, at `2^20` and `2^22`
//!   entries, per entry of all `k` tables; at `2^20` the same products over `Fp2` show what the
//!   prime-subfield first round saves;
//! - the batch of 3, 7 and 12 variables of RFC-0010's testing strategy, of degree 3, products of
//!   3, 2 and 3 tables over `Fp`, per entry of all its tables;
//! - one `tessera_poly::bind_lowest` pass over an `Fp` table, per entry, and binding every round
//!   (that pass, then `bind_lowest_in_place` down to one entry): what binding alone costs, a lower
//!   bound of the prover's cost per entry of each table.
//!
//! `cargo bench -p tessera-sumcheck` reports the fastest of several runs. Without `--bench`, as
//! under `cargo test --benches`, it runs once at small sizes as a smoke test.

#![forbid(unsafe_code)]
// A benchmark reports on stdout.
#![allow(clippy::print_stdout)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use tessera_codec::Ascii;
use tessera_field::{Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_poly::{bind_lowest, bind_lowest_in_place};
use tessera_sumcheck::{Instance, Prefix, Table, prove_batch};

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

fn context() -> SessionContext {
    SessionContext {
        protocol: Ascii::new("tessera-protocol/1").expect("ASCII"),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("test-small/1").expect("ASCII")),
        statement_digest: [0x5a; 32],
        session_id: [7; 16],
    }
}

/// The fastest of `runs` runs of `f`.
fn fastest<T>(runs: usize, mut f: impl FnMut() -> T) -> Duration {
    (0..runs)
        .map(|_| {
            let start = Instant::now();
            black_box(f());
            start.elapsed()
        })
        .min()
        .unwrap_or_default()
}

fn report(what: &str, entries: usize, time: Duration) {
    let ns = time.as_secs_f64() * 1e9 / entries as f64;
    let ms = time.as_secs_f64() * 1e3;
    println!("{what:<52} {ns:>7.3} ns per entry  ({ms:>8.2} ms)");
}

/// A batch of degree `degree` over `instances`, on a fresh transcript.
fn prove(degree: usize, instances: &[Instance<'_, Fp2>]) {
    let mut transcript = Transcript::new(&context());
    let proven = prove_batch(&mut transcript, Prefix::Stage(0), degree, instances)
        .expect("a well-formed batch");
    black_box(proven.into_proof());
}

fn single_products(log_n: u32, runs: usize, ext_too: bool) {
    let n = 1usize << log_n;
    let mut rng = SplitMix64(0x7E55_E7A0_0053_0001 ^ u64::from(log_n));
    let base: Vec<Vec<Fp>> = (0..4).map(|_| (0..n).map(|_| rng.fp()).collect()).collect();
    for k in 1..=4 {
        let instance = Instance::new(base[..k].iter().map(|t| Table::base(t)).collect());
        let time = fastest(runs, || prove(k, std::slice::from_ref(&instance)));
        report(&format!("product of {k} Fp tables, 2^{log_n}"), k * n, time);
    }
    if ext_too {
        let ext: Vec<Vec<Fp2>> = (0..4)
            .map(|_| (0..n).map(|_| rng.fp2()).collect())
            .collect();
        for k in 1..=4 {
            let instance = Instance::new(ext[..k].iter().map(|t| Table::ext(t)).collect());
            let time = fastest(runs, || prove(k, std::slice::from_ref(&instance)));
            report(
                &format!("product of {k} Fp2 tables, 2^{log_n}"),
                k * n,
                time,
            );
        }
    }
}

fn mixed_batch(runs: usize) {
    let mut rng = SplitMix64(0x7E55_E7A0_0053_0002);
    let shapes = [(3usize, 3usize), (7, 2), (12, 3)];
    let tables: Vec<Vec<Vec<Fp>>> = shapes
        .iter()
        .map(|&(vars, k)| {
            (0..k)
                .map(|_| (0..1 << vars).map(|_| rng.fp()).collect())
                .collect()
        })
        .collect();
    let instances: Vec<Instance<'_, Fp2>> = tables
        .iter()
        .map(|factors| Instance::new(factors.iter().map(|t| Table::base(t)).collect()))
        .collect();
    let entries = shapes.iter().map(|&(vars, k)| k << vars).sum();
    let time = fastest(runs, || prove(3, &instances));
    report(
        "batch of 3, 7 and 12 variables, degree 3, Fp",
        entries,
        time,
    );
}

fn binding(log_n: u32, runs: usize) {
    let n = 1usize << log_n;
    let mut rng = SplitMix64(0x7E55_E7A0_0053_0003 ^ u64::from(log_n));
    let table: Vec<Fp> = (0..n).map(|_| rng.fp()).collect();
    let r = rng.fp2();
    let time = fastest(runs, || bind_lowest(black_box(&table), r));
    report(
        &format!("one bind_lowest pass, Fp into Fp2, 2^{log_n}"),
        n,
        time,
    );
    // Binding alone over every round: the first pass, then in place down to one entry.
    let time = fastest(runs, || {
        let mut bound = bind_lowest(black_box(&table), r).expect("a variable");
        while bound.len() > 1 {
            bind_lowest_in_place(&mut bound, r).expect("a variable");
        }
        bound
    });
    report(&format!("  binding every round, 2^{log_n}"), n, time);
}

/// Keeps one core busy before anything is timed, so that the first measurement does not run on
/// a core that is still raising its clock.
fn warm_up(duration: Duration) {
    let mut rng = SplitMix64(0x7E55_E7A0_0053_0004);
    let table: Vec<Fp> = (0..1 << 14).map(|_| rng.fp()).collect();
    let start = Instant::now();
    while start.elapsed() < duration {
        let instance = Instance::new(vec![Table::base(&table), Table::base(&table)]);
        prove(2, &[instance]);
    }
}

fn main() {
    if std::env::args().any(|arg| arg == "--bench") {
        warm_up(Duration::from_millis(500));
        binding(20, 20);
        single_products(20, 10, true);
        binding(22, 5);
        single_products(22, 3, false);
        mixed_batch(200);
    } else {
        binding(10, 1);
        single_products(10, 1, true);
        mixed_batch(1);
    }
}
