// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The structured evaluation of tables' multilinear extensions (RFC-0008 'Structured
//! evaluation'): the time of one `eval_mle` at a random `Fp2` point, per kind, at the
//! production sizes:
//!
//! - `Range` and `Requant` of `2^32` rows, whose closed forms cost `O(n)` per class: INT8
//!   outputs of four classes, and 20-bit outputs of three classes, padded;
//! - `Exp` of `2^32` rows with a core of 12.5 million keys, and `Silu` with the largest core,
//!   `2^24` values, both `O(core)`;
//! - `Bitwise8` and `AesSbox`, enumerated.
//!
//! `cargo bench -p tessera-tables` reports the fastest of several runs, single-threaded. Without
//! `--bench`, as under `cargo test --benches`, it runs once at small parameters as a smoke test.
//! Generating a table's core is not timed: a claim list's tables are generated once.

#![forbid(unsafe_code)]
// A benchmark reports on stdout.
#![allow(clippy::print_stdout)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use tessera_codec::Rational;
use tessera_field::{Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_tables::{BitOp, GeneratedTable, Table, TableId, TableKind};

type Fp2 = GoldilocksExt2;

/// SplitMix64: reproducible points.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn fp2(&mut self) -> Fp2 {
        Fp2::new(
            Goldilocks::from_u64_reduced(self.next_u64()),
            Goldilocks::from_u64_reduced(self.next_u64()),
        )
    }
}

/// The fastest of `runs` evaluations of `table` at random points.
fn fastest(table: &GeneratedTable, runs: usize, rng: &mut SplitMix64) -> Duration {
    let n = usize::from(table.log2_rows());
    (0..runs)
        .map(|_| {
            let rho: Vec<Fp2> = (0..n).map(|_| rng.fp2()).collect();
            let beta = rng.fp2();
            let start = Instant::now();
            black_box(table.eval_mle(black_box(&rho), beta)).expect("a point of the table");
            start.elapsed()
        })
        .min()
        .unwrap_or_default()
}

fn requant(b: u8, r_by_class: &[u8], lo: i32, hi: i32) -> TableKind {
    TableKind::Requant {
        b,
        r_by_class: r_by_class.to_vec(),
        lo,
        hi,
    }
}

fn run(production: bool, runs: usize) {
    let mut rng = SplitMix64(0x7AB1_E5E7_A100_0075);
    let (b, exp_bits, silu_b) = if production { (30, 32, 26) } else { (8, 10, 8) };
    let cases = [
        ("Range", TableKind::Range { bits: exp_bits }),
        (
            "Requant, INT8, 4 classes",
            requant(b, &[3, 7, 12, 20], -128, 127),
        ),
        (
            "Requant, 20 bits, 3 classes",
            requant(b, &[1, 9, 16], -(1 << 19), (1 << 19) - 1),
        ),
        (
            "Exp, scale 10^-6",
            TableKind::Exp {
                key_bits: exp_bits,
                sigma: 16,
                scale: Rational::new(1, 1_000_000).expect("a scale"),
            },
        ),
        (
            "Silu, one class",
            TableKind::Silu {
                b: silu_b,
                r_by_class: vec![2],
                in_exp: -16,
                out_exp: -6,
            },
        ),
        ("Bitwise8, xor", TableKind::Bitwise8 { op: BitOp::Xor }),
        ("AesSbox", TableKind::AesSbox),
    ];
    for (what, kind) in cases {
        let table = GeneratedTable::new(TableId(0), kind).expect("valid parameters");
        let time = fastest(&table, runs, &mut rng);
        println!(
            "{what:<28} 2^{:<2} rows  {:>10.1} µs per evaluation",
            table.log2_rows(),
            time.as_secs_f64() * 1e6,
        );
    }
}

fn main() {
    if std::env::args().any(|arg| arg == "--bench") {
        run(true, 20);
    } else {
        run(false, 1);
    }
}
