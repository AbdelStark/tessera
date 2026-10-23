// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The tree levels' kernels under RFC-0020's measurement protocol, single-threaded over
//! Goldilocks: product forests through `prove_forests`, per leaf, and fraction trees through
//! `prove_trees`, per leaf, both public entry points, so that one program times any revision.
//!
//! - Forests of 2^22 leaves in three shapes: 2^8 trees of 2^14 leaves, 2^10 of 2^12, one of 2^22.
//! - Fraction trees of 2^21 leaves: a lookup tree (numerators 1) and a table tree (numerators in
//!   `Fp`), each alone; the trees are built outside the timing.
//! - Protocol: 3 warm-ups and 10 measured iterations per case; an iteration repeats the case
//!   until it lasts at least 20 ms; median, minimum, maximum and coefficient of variation; a
//!   coefficient above 3% reruns the case once. Every raw sample is written.
//!
//! ```sh
//! cargo bench -p tessera-sumcheck --bench levels -- OUT.json [--small]
//! ```
//!
//! The digest of every case's proof is printed: two revisions that agree on it prove the same.
//! `--small` runs `2^12`-leaf forests and `2^11`-leaf trees; without `--bench`, as under
//! `cargo test --benches`, so does the program, as a smoke test.

#![forbid(unsafe_code)]
// A benchmark reports on stdout.
#![allow(clippy::print_stdout)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use tessera_codec::Ascii;
use tessera_field::{Goldilocks as Fp, GoldilocksExt2 as Fp2, PrimeField, QuadraticExtension};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_sumcheck::{
    Forest, FractionTree, LeafNumerators, SumcheckProof, Tree, prove_forests, prove_trees,
};

const WARMUPS: usize = 3;
const ITERATIONS: usize = 10;
const CV_LIMIT: f64 = 0.03;
const MIN_ITERATION: Duration = Duration::from_millis(20);

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

/// The canonical bytes of a proof's messages, hashed.
fn digest(proofs: &[&SumcheckProof<Fp2>]) -> String {
    let mut bytes = Vec::new();
    for proof in proofs {
        for round in &proof.rounds {
            for x in round {
                for c in x.coeffs() {
                    bytes.extend_from_slice(&c.to_canonical_u64().to_le_bytes());
                }
            }
        }
        for x in &proof.final_evals {
            for c in x.coeffs() {
                bytes.extend_from_slice(&c.to_canonical_u64().to_le_bytes());
            }
        }
    }
    tessera_hash::hash(tessera_hash::Domain::Table, &bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

struct Stats {
    samples: Vec<f64>,
    median: f64,
    min: f64,
    max: f64,
    cv: f64,
}

fn stats(samples: Vec<f64>) -> Stats {
    let mut sorted = samples.clone();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len() as f64;
    let mean = sorted.iter().sum::<f64>() / n;
    let var = sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    Stats {
        median: (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0,
        min: sorted[0],
        max: sorted[sorted.len() - 1],
        cv: var.sqrt() / mean,
        samples,
    }
}

/// `case` returns the time of one execution, its setup excluded; an iteration repeats it until
/// it lasts `MIN_ITERATION`, and a sample is the iteration's time per unit.
fn measure(units: usize, mut case: impl FnMut() -> Duration) -> (Stats, Vec<f64>) {
    let mut iteration = || {
        let (mut total, mut reps) = (Duration::ZERO, 0u32);
        while total < MIN_ITERATION {
            total += case();
            reps += 1;
        }
        total.as_secs_f64() * 1e9 / f64::from(reps) / units as f64
    };
    for _ in 0..WARMUPS {
        iteration();
    }
    let first: Vec<f64> = (0..ITERATIONS).map(|_| iteration()).collect();
    let first = stats(first);
    if first.cv <= CV_LIMIT {
        return (first, Vec::new());
    }
    let rejected = first.samples;
    (
        stats((0..ITERATIONS).map(|_| iteration()).collect()),
        rejected,
    )
}

fn forest_case(name: &str, s: usize, h: usize, rows: &mut Vec<serde_json::Value>) {
    let mut rng = SplitMix64(0x7E55_E7A0_0391_0001 ^ ((s as u64) << 8 | h as u64));
    let leaves: Vec<Fp> = (0..1usize << (s + h)).map(|_| rng.fp()).collect();
    let batch = [Forest::<Fp2>::new(&leaves, h)];
    let mut transcript = Transcript::new(&context());
    let proven = prove_forests(&mut transcript, &batch).expect("a forest");
    let (proofs, _) = proven.into_parts();
    let refs: Vec<&SumcheckProof<Fp2>> = proofs.iter().collect();
    let digest = digest(&refs);
    let (stats, rejected) = measure(leaves.len(), || {
        let mut transcript = Transcript::new(&context());
        let start = Instant::now();
        let proven = prove_forests(&mut transcript, black_box(&batch)).expect("a forest");
        let elapsed = start.elapsed();
        drop(black_box(proven));
        elapsed
    });
    report(name, "ns/leaf", &digest, stats, rejected, rows);
}

fn fraction_case(name: &str, log: usize, base: bool, rows: &mut Vec<serde_json::Value>) {
    let mut rng = SplitMix64(0x7E55_E7A0_0391_0002 ^ (log as u64) << 1 ^ u64::from(base));
    let denominators: Vec<Fp2> = (0..1usize << log).map(|_| rng.fp2()).collect();
    let numerators: Vec<Fp> = (0..1usize << log).map(|_| rng.fp()).collect();
    let tree = || {
        let numerators = match base {
            true => LeafNumerators::Base(&numerators),
            false => LeafNumerators::Unit,
        };
        FractionTree::<Fp2>::new(log, numerators, |x| denominators[x]).expect("a tree")
    };
    let mut transcript = Transcript::new(&context());
    let proven = prove_trees(&mut transcript, vec![Tree::Fraction(tree())]).expect("a tree");
    let (proofs, _) = proven.into_parts();
    let refs: Vec<&SumcheckProof<Fp2>> = proofs.iter().collect();
    let digest = digest(&refs);
    let (stats, rejected) = measure(denominators.len(), || {
        let trees = vec![Tree::Fraction(tree())];
        let mut transcript = Transcript::new(&context());
        let start = Instant::now();
        let proven = prove_trees(&mut transcript, black_box(trees)).expect("a tree");
        let elapsed = start.elapsed();
        drop(black_box(proven));
        elapsed
    });
    report(name, "ns/leaf", &digest, stats, rejected, rows);
}

fn report(
    name: &str,
    unit: &str,
    digest: &str,
    stats: Stats,
    rejected: Vec<f64>,
    rows: &mut Vec<serde_json::Value>,
) {
    println!(
        "{name:<34} {:>9.3} {unit} (min {:.3}, max {:.3}, CV {:.4}%{}) digest {}",
        stats.median,
        stats.min,
        stats.max,
        100.0 * stats.cv,
        if rejected.is_empty() { "" } else { ", rerun" },
        &digest[..16],
    );
    rows.push(serde_json::json!({
        "case": name, "unit": unit, "digest": digest, "median": stats.median, "min": stats.min,
        "max": stats.max, "cv": stats.cv, "samples": stats.samples, "rejected_first_run": rejected,
    }));
}

fn main() {
    let bench = std::env::args().any(|a| a == "--bench");
    let args: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| a != "--bench")
        .collect();
    let out = args.iter().find(|a| !a.starts_with("--")).cloned();
    let small = !bench || args.iter().any(|a| a == "--small");
    let (forest_log, fraction_log) = if small { (12, 11) } else { (22, 21) };
    let mut rows = Vec::new();
    let shapes: &[usize] = if small { &[4, 0] } else { &[8, 10, 0] };
    for &s in shapes {
        let h = forest_log - s;
        forest_case(&format!("forest 2^{s} x 2^{h}"), s, h, &mut rows);
    }
    let lookup = format!("fraction lookup tree 2^{fraction_log}");
    fraction_case(&lookup, fraction_log, false, &mut rows);
    let table = format!("fraction table tree 2^{fraction_log}");
    fraction_case(&table, fraction_log, true, &mut rows);
    if let Some(path) = out {
        let report = serde_json::json!({"protocol": "RFC-0020: 3 warm-ups, 10 samples, >=20 ms, CV rerun at 3%", "cases": rows});
        std::fs::write(&path, format!("{report:#}\n")).expect("the report");
    }
}
