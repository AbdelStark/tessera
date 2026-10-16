// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Throughput of the LogUp prover, in nanoseconds per lookup, single-threaded over Goldilocks and
//! its quadratic extension: one instance of `2^16` to `2^22` random lookups into a dense table of
//! `2^16` rows with values, the RFC-0008 v0.1 shape.
//!
//! - `prove_logup` as a whole: challenges, both trees, roots, every level, values and claims;
//! - its parts, each through the public API: the lookup tree alone (`FractionTree::new` with
//!   numerators of 1, the leaves `α − key − β·value` computed as the prover does), the table tree
//!   alone, the levels of both trees (`prove_trees`), and the evaluation of the value column at
//!   the lookup tree's leaf point;
//! - `multiplicities`, which an honest witness pass runs before `α`.
//!
//! `cargo bench -p tessera-logup` reports the fastest of several runs. Without `--bench`, as under
//! `cargo test --benches`, it runs once at small sizes as a smoke test.

#![forbid(unsafe_code)]
// A benchmark reports on stdout.
#![allow(clippy::print_stdout)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use tessera_codec::Ascii;
use tessera_field::{Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_logup::{LogUpInstance, TreeId, Tuples, multiplicities, prove_logup};
use tessera_poly::{PolyField, evaluate_base};
use tessera_sumcheck::{FractionTree, LeafNumerators, Tree, prove_trees};

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
        statement_digest: [0x60; 32],
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

fn report(what: &str, lookups: usize, time: Duration) {
    let ns = time.as_secs_f64() * 1e9 / lookups as f64;
    let ms = time.as_secs_f64() * 1e3;
    println!("{what:<46} {ns:>8.2} ns per lookup ({ms:>9.2} ms)");
}

/// The instance's columns: a table of `2^rows_log` random rows with values, and `2^lookups_log`
/// lookups of random rows.
struct Columns {
    keys: Vec<Fp>,
    values: Vec<Fp>,
    rows: Vec<u32>,
    lookup_keys: Vec<Fp>,
    lookup_values: Vec<Fp>,
    multiplicities: Vec<Fp>,
}

impl Columns {
    fn random(rows_log: u32, lookups_log: u32) -> Self {
        let mut rng = SplitMix64(0x60_5001 ^ u64::from(lookups_log));
        let rows = 1usize << rows_log;
        let keys: Vec<Fp> = (0..rows).map(|_| rng.fp()).collect();
        let values: Vec<Fp> = (0..rows).map(|_| rng.fp()).collect();
        let lookup_rows: Vec<u32> = (0..1usize << lookups_log)
            .map(|_| (rng.next_u64() % rows as u64) as u32)
            .collect();
        let lookup_keys = lookup_rows.iter().map(|&r| keys[r as usize]).collect();
        let lookup_values = lookup_rows.iter().map(|&r| values[r as usize]).collect();
        let multiplicities =
            multiplicities(Tuples::new(&keys, Some(&values)), &lookup_rows).expect("rows");
        Self {
            keys,
            values,
            rows: lookup_rows,
            lookup_keys,
            lookup_values,
            multiplicities,
        }
    }

    fn instance(&self) -> LogUpInstance<'_, Fp> {
        LogUpInstance::new(
            Tuples::new(&self.keys, Some(&self.values)),
            Tuples::new(&self.lookup_keys, Some(&self.lookup_values)),
            &self.multiplicities,
        )
    }
}

fn logup(rows_log: u32, lookups_log: u32, runs: usize) {
    let columns = Columns::random(rows_log, lookups_log);
    let lookups = 1usize << lookups_log;
    let instance = columns.instance();
    let at = format!("2^{lookups_log} lookups, 2^{rows_log} rows");
    let time = fastest(runs, || {
        let mut transcript = Transcript::new(&context());
        let proven = prove_logup::<Fp2, _>(&mut transcript, &[instance], &[], &TreeId::paired(1))
            .expect("a well-formed instance");
        proven.into_claims()
    });
    report(&format!("prove_logup: {at}"), lookups, time);

    // The parts, at random challenges.
    let mut rng = SplitMix64(0x60_5002);
    let (alpha, beta) = (rng.fp2(), rng.fp2());
    let leaf = |key: Fp, value: Fp| alpha - beta.mul_by_base(value).add_base(key);
    let lookup_tree = || {
        FractionTree::<Fp2>::new(lookups_log as usize, LeafNumerators::Unit, |x| {
            leaf(columns.lookup_keys[x], columns.lookup_values[x])
        })
        .expect("a tree")
    };
    let negated: Vec<Fp> = columns.multiplicities.iter().map(|&m| -m).collect();
    let table_tree = || {
        FractionTree::<Fp2>::new(rows_log as usize, LeafNumerators::Base(&negated), |x| {
            leaf(columns.keys[x], columns.values[x])
        })
        .expect("a tree")
    };
    report("  the lookup tree", lookups, fastest(runs, lookup_tree));
    report("  the table tree", lookups, fastest(runs, table_tree));
    // The levels, the trees built outside the timing.
    let time = (0..runs)
        .map(|_| {
            let trees = vec![Tree::Fraction(lookup_tree()), Tree::Fraction(table_tree())];
            let mut transcript = Transcript::new(&context());
            let start = Instant::now();
            let proven = prove_trees(&mut transcript, trees).expect("fraction trees");
            black_box(proven.into_parts());
            start.elapsed()
        })
        .min()
        .unwrap_or_default();
    report("  the levels of both trees", lookups, time);
    let point: Vec<Fp2> = (0..lookups_log).map(|_| rng.fp2()).collect();
    let time = fastest(runs, || evaluate_base(&columns.lookup_values, &point));
    report("  the value column at ρ_a", lookups, time);
    let table = Tuples::new(&columns.keys, Some(&columns.values));
    let time = fastest(runs, || multiplicities::<Fp>(table, &columns.rows));
    report("  multiplicities, before α", lookups, time);
}

/// Keeps one core busy before anything is timed, so that the first measurement does not run on
/// a core that is still raising its clock.
fn warm_up(duration: Duration) {
    let columns = Columns::random(8, 12);
    let instance = columns.instance();
    let start = Instant::now();
    while start.elapsed() < duration {
        let mut transcript = Transcript::new(&context());
        let proven = prove_logup::<Fp2, _>(&mut transcript, &[instance], &[], &TreeId::paired(1))
            .expect("a well-formed instance");
        black_box(proven.into_claims());
    }
}

fn main() {
    if std::env::args().any(|arg| arg == "--bench") {
        warm_up(Duration::from_millis(500));
        for (lookups_log, runs) in [(16, 20), (18, 10), (20, 5), (22, 3)] {
            logup(16, lookups_log, runs);
        }
    } else {
        logup(8, 10, 1);
    }
}
