// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The CPU microbenchmarks of 08 §6: the prover's primitives at fixed sizes, single-threaded,
//! each measured with RFC-0020's protocol (3 warm-up runs, 10 measured runs; median, extremes and
//! coefficient of variation, flagged above 3%).
//!
//! | Benchmark | Work per run | Unit |
//! |---|---|---|
//! | `field/fp-mul`, `field/fp2-mul` | `2^20` independent Goldilocks or `Fp2` products | product |
//! | `field/fp-batch-inverse` | one batch inversion of `2^16` elements | element |
//! | `ntt/fp` | one forward NTT of `2^20` points, its plan built once | point |
//! | `hash/column-chains` | one block of `B = 32` rows of `2^14` columns into their chains | symbol |
//! | `merkle/multiproof` | 64 canonical multiproofs of 155 indices in a tree of `2^20` leaves | opening |
//! | `sumcheck/product-3` | a degree-3 sumcheck of three `Fp` tables of `2^20` entries | entry |
//! | `logup/prove` | the LogUp stage of `2^18` lookups into `2^16` rows with values | lookup |
//!
//! `cargo bench -p tessera-bench --bench primitives` prints them. With `-- --runner-class NAME
//! --out FILE` it also writes a `tessera.microbenchmarks/1` baseline. Without `--bench`, as under
//! `cargo test --benches`, every benchmark runs once at small sizes as a smoke test, and nothing
//! is written.

#![forbid(unsafe_code)]
// A benchmark reports on stdout and its errors on stderr.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::hint::black_box;
use std::num::NonZeroUsize;
use std::process::ExitCode;

use tessera_bench::{Measurement, MicrobenchmarkReport, Protocol, measure};
use tessera_codec::Ascii;
use tessera_field::{
    Field, Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension, batch_inverse,
};
use tessera_hash::{
    ColumnChains, Digest, MerkleTree, Mode, ParamSetId, SessionContext, Transcript,
};
use tessera_logup::{LogUpInstance, TreeId, Tuples, multiplicities, prove_logup};
use tessera_pcs::NttPlan;
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

    fn fps(&mut self, n: usize) -> Vec<Fp> {
        (0..n).map(|_| self.fp()).collect()
    }

    fn fp2s(&mut self, n: usize) -> Vec<Fp2> {
        (0..n).map(|_| Fp2::new(self.fp(), self.fp())).collect()
    }
}

fn transcript() -> Transcript {
    Transcript::new(&SessionContext {
        protocol: Ascii::new("tessera-protocol/1").expect("ASCII"),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("test-small/1").expect("ASCII")),
        statement_digest: [0x65; 32],
        session_id: [0x65; 16],
    })
}

/// The sizes of one run: the full ones, or a smoke test's.
struct Sizes {
    field_log: u32,
    inverse_log: u32,
    ntt_log: u32,
    columns_log: u32,
    merkle_log: u32,
    openings: usize,
    sumcheck_log: u32,
    lookups_log: u32,
    rows_log: u32,
}

const FULL: Sizes = Sizes {
    field_log: 20,
    inverse_log: 16,
    ntt_log: 20,
    columns_log: 14,
    merkle_log: 20,
    openings: 64,
    sumcheck_log: 20,
    lookups_log: 18,
    rows_log: 16,
};

const SMOKE: Sizes = Sizes {
    field_log: 10,
    inverse_log: 8,
    ntt_log: 10,
    columns_log: 6,
    merkle_log: 10,
    openings: 2,
    sumcheck_log: 10,
    lookups_log: 8,
    rows_log: 6,
};

/// A benchmark: its name, its unit of work, its units per run and the run.
type Bench = (&'static str, &'static str, usize, Box<dyn FnMut()>);

fn benchmarks(s: &Sizes) -> Vec<Bench> {
    let mut rng = SplitMix64(0x0065_0001);
    let mut list: Vec<Bench> = Vec::new();

    let n = 1usize << s.field_log;
    let (a, b) = (rng.fps(n), rng.fps(n));
    let mut out = vec![Fp::ZERO; n];
    list.push((
        "field/fp-mul",
        "product",
        n,
        Box::new(move || {
            for ((o, &x), &y) in out.iter_mut().zip(&a).zip(&b) {
                *o = x * y;
            }
            black_box(&out);
        }),
    ));
    let (a, b) = (rng.fp2s(n), rng.fp2s(n));
    let mut out = vec![Fp2::ZERO; n];
    list.push((
        "field/fp2-mul",
        "product",
        n,
        Box::new(move || {
            for ((o, &x), &y) in out.iter_mut().zip(&a).zip(&b) {
                *o = x * y;
            }
            black_box(&out);
        }),
    ));
    let n = 1usize << s.inverse_log;
    let source = rng.fps(n);
    let mut work = source.clone();
    list.push((
        "field/fp-batch-inverse",
        "element",
        n,
        Box::new(move || {
            work.copy_from_slice(&source);
            batch_inverse(&mut work).expect("no zero among random elements");
            black_box(&work);
        }),
    ));

    let n = 1usize << s.ntt_log;
    let plan = NttPlan::new(n).expect("a power-of-two NTT");
    let source = rng.fps(n);
    let mut work = source.clone();
    list.push((
        "ntt/fp",
        "point",
        n,
        Box::new(move || {
            work.copy_from_slice(&source);
            plan.ntt(&mut work).expect("a plan of this size");
            black_box(&work);
        }),
    ));

    let columns = 1usize << s.columns_log;
    let block = 32;
    let rows = rng.fps(columns * block);
    let id: Digest = [0x65; 32];
    list.push((
        "hash/column-chains",
        "symbol",
        columns * block,
        Box::new(move || {
            let mut chains = ColumnChains::new(
                &id,
                NonZeroUsize::new(columns).expect("columns"),
                NonZeroUsize::new(block).expect("a block"),
            );
            chains.absorb_block(&rows).expect("one whole block");
            black_box(chains.finish());
        }),
    ));

    let leaves = 1usize << s.merkle_log;
    let digests: Vec<Digest> = (0..leaves)
        .map(|_| {
            let mut d = [0u8; 32];
            for chunk in d.chunks_exact_mut(8) {
                chunk.copy_from_slice(&rng.next_u64().to_le_bytes());
            }
            d
        })
        .collect();
    let tree = MerkleTree::new(digests).expect("a power-of-two tree");
    let queries: Vec<Vec<u64>> = (0..s.openings)
        .map(|_| {
            let mut indices: Vec<u64> = (0..155).map(|_| rng.next_u64() % leaves as u64).collect();
            indices.sort_unstable();
            indices.dedup();
            indices
        })
        .collect();
    list.push((
        "merkle/multiproof",
        "opening",
        s.openings,
        Box::new(move || {
            for indices in &queries {
                black_box(tree.open(indices).expect("indices in range"));
            }
        }),
    ));

    let n = 1usize << s.sumcheck_log;
    let tables = [rng.fps(n), rng.fps(n), rng.fps(n)];
    list.push((
        "sumcheck/product-3",
        "entry",
        n,
        Box::new(move || {
            let instance = Instance::<Fp2>::new(tables.iter().map(|t| Table::base(t)).collect());
            let proven = prove_batch(&mut transcript(), Prefix::Stage(0), 3, &[instance])
                .expect("a well-formed batch");
            black_box(proven.into_proof());
        }),
    ));

    let (rows, lookups) = (1usize << s.rows_log, 1usize << s.lookups_log);
    let keys: Vec<Fp> = (0..rows as u64).map(Fp::from_u64_reduced).collect();
    let values = rng.fps(rows);
    let picks: Vec<u32> = (0..lookups)
        .map(|_| (rng.next_u64() % rows as u64) as u32)
        .collect();
    let lookup_keys: Vec<Fp> = picks.iter().map(|&r| keys[r as usize]).collect();
    let lookup_values: Vec<Fp> = picks.iter().map(|&r| values[r as usize]).collect();
    let counts =
        multiplicities(Tuples::new(&keys, Some(&values)), &picks).expect("rows of the table");
    list.push((
        "logup/prove",
        "lookup",
        lookups,
        Box::new(move || {
            let instance = LogUpInstance::new(
                Tuples::new(&keys, Some(&values)),
                Tuples::new(&lookup_keys, Some(&lookup_values)),
                &counts,
            );
            let proven =
                prove_logup::<Fp2, _>(&mut transcript(), &[instance], &[], &TreeId::paired(1))
                    .expect("a well-formed stage");
            black_box(proven.into_claims());
        }),
    ));
    list
}

/// `--runner-class NAME` and `--out FILE`, after the arguments cargo passes.
struct Args {
    full: bool,
    runner_class: Option<String>,
    out: Option<String>,
}

fn args() -> Result<Args, String> {
    let mut parsed = Args {
        full: false,
        runner_class: None,
        out: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bench" => parsed.full = true,
            "--runner-class" => parsed.runner_class = args.next(),
            "--out" => parsed.out = args.next(),
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    match (&parsed.runner_class, &parsed.out) {
        (Some(_), Some(_)) | (None, None) => Ok(parsed),
        _ => Err("--runner-class and --out go together".to_owned()),
    }
}

fn main() -> ExitCode {
    let args = match args() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("error: {message}");
            eprintln!(
                "usage: cargo bench -p tessera-bench --bench primitives [-- --runner-class NAME --out FILE]"
            );
            return ExitCode::from(64);
        }
    };
    let (sizes, protocol) = match args.full {
        true => (&FULL, Protocol::RFC_0020),
        false => (
            &SMOKE,
            Protocol {
                warmups: 0,
                iterations: 1,
            },
        ),
    };
    let mut measurements = Vec::new();
    println!(
        "{:<24} {:>9} {:>12} {:>12} {:>12} {:>7}  unit",
        "benchmark", "units", "median ns", "min ns", "max ns", "cv"
    );
    for (name, unit, units, mut run) in benchmarks(sizes) {
        let statistics = match measure(protocol, units as f64, &mut run) {
            Ok(statistics) => statistics,
            Err(error) => {
                eprintln!("error: {name}: {error}");
                return ExitCode::from(70);
            }
        };
        println!(
            "{name:<24} {units:>9} {:>12.3} {:>12.3} {:>12.3} {:>6.2}%  per {unit}{}",
            statistics.median,
            statistics.min,
            statistics.max,
            100.0 * statistics.cv,
            if statistics.noisy() {
                "  (noisy: above 3%)"
            } else {
                ""
            }
        );
        measurements.push(Measurement {
            name: name.to_owned(),
            unit: unit.to_owned(),
            units: units as u64,
            statistics,
        });
    }
    let (Some(runner_class), Some(out), true) = (args.runner_class, args.out, args.full) else {
        return ExitCode::SUCCESS;
    };
    let written = MicrobenchmarkReport::new(&runner_class, protocol, measurements)
        .and_then(|report| report.to_json())
        .map_err(|error| error.to_string())
        .and_then(|json| {
            std::fs::write(&out, json + "\n").map_err(|error| format!("{out}: {error}"))
        });
    match written {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::from(70)
        }
    }
}
