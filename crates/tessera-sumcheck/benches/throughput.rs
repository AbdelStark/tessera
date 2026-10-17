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
//!   bound of the prover's cost per entry of each table. Every timing includes clearing what the
//!   prover derives as it drops (05 §3 OBS-6); beside the pass, the bound table's clearing alone
//!   shows what that costs;
//! - `prove_forests` on a forest of 2^10 trees of 2^12 leaves, on one tree of 2^22 leaves and on a
//!   batch of forests of several heights and tree counts, 2^22 leaves each, per leaf. Beside each
//!   come the `eq` tables the batch prover would build for its levels, alone, which the levels'
//!   split-`eq` kernel never builds, and the engine's degree-3 sumcheck of a product
//!   of three tables of as many entries as leaves, per entry: three tables over `Fp`, and a tree
//!   level's shape, one over `Fp2` and two over `Fp`.
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
use tessera_poly::{bind_lowest, bind_lowest_in_place, eq_table};
use tessera_sumcheck::{Forest, Instance, Prefix, Table, prove_batch, prove_forests};

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

/// The fastest of `runs` runs of `f`, dropping its result included.
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

/// The fastest of `runs` drops of what `make` builds, untimed.
fn fastest_drop<T>(runs: usize, mut make: impl FnMut() -> T) -> Duration {
    (0..runs)
        .map(|_| {
            let value = black_box(make());
            let start = Instant::now();
            drop(value);
            start.elapsed()
        })
        .min()
        .unwrap_or_default()
}

fn report(what: &str, entries: usize, time: Duration) {
    report_per(what, entries, "entry", time);
}

fn report_per(what: &str, count: usize, unit: &str, time: Duration) {
    let ns = time.as_secs_f64() * 1e9 / count as f64;
    let ms = time.as_secs_f64() * 1e3;
    println!("{what:<52} {ns:>7.3} ns per {unit:<5} ({ms:>8.2} ms)");
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
    // Its result is secret-class and clears its allocation as it drops: what that costs alone.
    let time = fastest_drop(runs, || bind_lowest(&table, r).expect("a variable"));
    report("  of which clearing the bound table", n, time);
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

/// `prove_forests` over forests of `(s, h)` shapes, `2^s` trees of height `h`, per leaf; the `eq`
/// tables the batch prover would build for its levels, alone, per leaf, which the split-`eq`
/// kernel saves; and the engine's degree-3 sumcheck of three tables of
/// as many entries as the batch has leaves, per entry, three over `Fp` and a tree level's shape,
/// one over `Fp2` and two over `Fp`.
fn forests(what: &str, shapes: &[(usize, usize)], runs: usize) {
    let mut rng = SplitMix64(0x7E55_E7A0_0058_0001);
    let leaves: Vec<Vec<Fp>> = shapes
        .iter()
        .map(|&(s, h)| (0..1usize << (s + h)).map(|_| rng.fp()).collect())
        .collect();
    let batch: Vec<Forest<'_, Fp2>> = leaves
        .iter()
        .zip(shapes)
        .map(|(leaves, &(_, h))| Forest::new(leaves, h))
        .collect();
    let total: usize = leaves.iter().map(Vec::len).sum();
    assert!(
        total.is_power_of_two(),
        "the engine's tables have as many entries"
    );
    let time = fastest(runs, || {
        let mut transcript = Transcript::new(&context());
        let proven = prove_forests(&mut transcript, &batch).expect("well-formed forests");
        proven.into_parts()
    });
    report_per(&format!("forests: {what}"), total, "leaf", time);

    // The eq tables the batch prover would build at every level: at level ℓ, one per forest that
    // has joined, on its s + ℓ − (H − h) coordinates. The split-eq kernel builds none.
    let tallest = shapes.iter().map(|&(_, h)| h).max().unwrap_or(0);
    let points: Vec<Vec<Fp2>> = (0..tallest)
        .flat_map(|level| {
            shapes
                .iter()
                .filter(move |&&(_, h)| tallest - h <= level)
                .map(move |&(s, h)| s + level - (tallest - h))
        })
        .map(|vars| (0..vars).map(|_| rng.fp2()).collect())
        .collect();
    let time = fastest(runs, || {
        for point in &points {
            black_box(eq_table(black_box(point)).expect("a table"));
        }
    });
    report_per(
        "  the eq tables a batch prover would build",
        total,
        "leaf",
        time,
    );

    let a: Vec<Fp> = (0..total).map(|_| rng.fp()).collect();
    let b: Vec<Fp> = (0..total).map(|_| rng.fp()).collect();
    let c: Vec<Fp> = (0..total).map(|_| rng.fp()).collect();
    let instance = Instance::new(vec![Table::base(&a), Table::base(&b), Table::base(&c)]);
    let time = fastest(runs, || prove(3, std::slice::from_ref(&instance)));
    report("  engine: product of 3 Fp tables, degree 3", total, time);
    let eq: Vec<Fp2> = (0..total).map(|_| rng.fp2()).collect();
    let instance = Instance::new(vec![
        Table::ext(&eq).public(),
        Table::base(&b),
        Table::base(&c),
    ]);
    let time = fastest(runs, || prove(3, std::slice::from_ref(&instance)));
    report("  engine: Fp2 public x 2 Fp tables, degree 3", total, time);
}

/// The three forest batches of 2^22 leaves, or of 2^10 as a smoke test.
fn forest_cases(large: bool, runs: usize) {
    if large {
        forests("2^10 trees of 2^12 leaves", &[(10, 12)], runs);
        forests("one tree of 2^22 leaves", &[(0, 22)], runs);
        forests(
            "2^9 x 2^12, 2^11 x 2^9 and 1 x 2^20 leaves",
            &[(9, 12), (11, 9), (0, 20)],
            runs,
        );
    } else {
        forests("2^4 trees of 2^6 leaves", &[(4, 6)], runs);
        forests(
            "2^3 x 2^5, 2^5 x 2^2 and 1 x 2^7 leaves",
            &[(3, 5), (5, 2), (0, 7)],
            runs,
        );
    }
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
        forest_cases(true, 5);
    } else {
        binding(10, 1);
        single_products(10, 1, true);
        mixed_batch(1);
        forest_cases(false, 1);
    }
}
