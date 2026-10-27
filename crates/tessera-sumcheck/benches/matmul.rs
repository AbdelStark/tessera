// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Throughput of the matmul relations, single-threaded over Goldilocks and its quadratic
//! extension, at several sizes, to show that the prover's cost is linear in the operands' sizes:
//! `O(|W| + |X|)` and `O(|A| + |B|)` field work beyond the product, never `O(m·k·n)`.
//!
//! For each shape it reports, the fastest of several runs:
//!
//! - **contraction**: `tables`, the `eq` tables and the contractions, per entry of the operands it
//!   contracts, `|W| + |X|` for `MatMulWeight` and `|A| + |B|` for `MatMulAct` when both have a
//!   variable to contract;
//! - **sumcheck**: `prove_batch` on the instance alone, of the relation's degree, per entry of the
//!   instance, `2^{n_i + n_ℓ}` or `2^{n_j + n_b}`;
//! - **whole**: `tables`, the sumcheck and `claims`, per entry of both operands.
//!
//! The weight shapes are the issue's, then grow one dimension at a time from `2^3` layers of
//! `2^10` tokens' activations of `2^10` inputs by `2^10 × 2^10` weights: tokens, outputs, inputs
//! and layers. The activation shapes are the issue's, then attention's, `QKᵀ` with `K` stored
//! transposed and `PV` with `V` as it is, over `2^4` heads.
//!
//! `cargo bench -p tessera-sumcheck --bench matmul` runs them. Without `--bench`, as under
//! `cargo test --benches`, it runs once at small sizes as a smoke test.

#![forbid(unsafe_code)]
// A benchmark reports on stdout.
#![allow(clippy::print_stdout)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use tessera_codec::Ascii;
use tessera_field::{Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_sumcheck::{
    Instance, MatMulAct, MatMulActShape, MatMulWeight, MatMulWeightShape, Prefix, Protocol,
    ProvenBatch, RightLayout, prove_batch,
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

    fn fps(&mut self, n: usize) -> Vec<Fp> {
        (0..n)
            .map(|_| Fp::from_u64_reduced(self.next_u64()))
            .collect()
    }

    fn fp2s(&mut self, n: usize) -> Vec<Fp2> {
        (0..n)
            .map(|_| {
                let c0 = Fp::from_u64_reduced(self.next_u64());
                Fp2::new(c0, Fp::from_u64_reduced(self.next_u64()))
            })
            .collect()
    }
}

fn context() -> SessionContext {
    SessionContext {
        protocol: Ascii::new("tessera-protocol/1").expect("ASCII"),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("test-small/1").expect("ASCII")),
        statement_digest: [0x57; 32],
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

fn ns_per(time: Duration, count: usize) -> f64 {
    time.as_secs_f64() * 1e9 / count as f64
}

/// The instance alone in a stage of degree `degree`, on a fresh transcript.
fn prove(degree: usize, instance: Instance<'_, Fp2>) -> ProvenBatch<Fp2> {
    let mut transcript = Transcript::new(&context());
    prove_batch(
        &mut transcript,
        Protocol::One,
        Prefix::Stage(0),
        degree,
        &[instance],
    )
    .expect("a valid stage")
}

/// One line of the report.
struct Line {
    what: String,
    /// The entries the contractions read.
    contracted: usize,
    /// The entries of both operands.
    operands: usize,
    /// The entries of the instance.
    instance: usize,
    contraction: Duration,
    sumcheck: Duration,
    whole: Duration,
}

fn header() {
    println!(
        "{:<42} {:>9} {:>9} {:>12} {:>12} {:>12} {:>10}",
        "shape", "operands", "instance", "contraction", "sumcheck", "whole", "whole"
    );
    println!(
        "{:<42} {:>9} {:>9} {:>12} {:>12} {:>12} {:>10}",
        "", "log2", "log2", "ns/read", "ns/instance", "ns/operand", "ms"
    );
}

fn report(line: &Line) {
    println!(
        "{:<42} {:>9.2} {:>9} {:>12.3} {:>12.3} {:>12.3} {:>10.2}",
        line.what,
        (line.operands as f64).log2(),
        line.instance.trailing_zeros(),
        ns_per(line.contraction, line.contracted),
        ns_per(line.sumcheck, line.instance),
        ns_per(line.whole, line.operands),
        line.whole.as_secs_f64() * 1e3,
    );
}

/// `MatMulWeight` over `2^{n_ℓ}` layers of `2^{n_t} × 2^{n_i}` activations by `2^{n_o} × 2^{n_i}`
/// weights.
fn weight(
    rng: &mut SplitMix64,
    what: &str,
    (layer_vars, token_vars, out_vars, in_vars): (usize, usize, usize, usize),
    runs: usize,
) {
    let shape = MatMulWeightShape {
        layer_vars,
        token_vars,
        out_vars,
        in_vars,
    };
    let w = rng.fps(1 << (layer_vars + out_vars + in_vars));
    let x = rng.fps(1 << (layer_vars + token_vars + in_vars));
    let point = rng.fp2s(out_vars + token_vars + layer_vars);
    let relation = MatMulWeight::<Fp2>::new(&w, &x, shape).expect("a valid shape");
    let degree = relation.degree();
    // An operand with no variable to contract is used as it is.
    let contracted = usize::from(out_vars > 0) * w.len() + usize::from(token_vars > 0) * x.len();
    assert!(contracted > 0, "every shape here contracts an operand");
    let contraction = fastest(runs, || relation.tables(&point).expect("a point"));
    let tables = relation.tables(&point).expect("a point");
    let sumcheck = fastest(runs, || prove(degree, tables.instance()).into_proof());
    let whole = fastest(runs, || {
        let tables = relation.tables(&point).expect("a point");
        let proven = prove(degree, tables.instance());
        let at = proven.instance_point(0).expect("an instance");
        tables
            .claims(at, proven.instance_evals(0).expect("an instance"))
            .expect("the instance's results")
    });
    report(&Line {
        what: format!(
            "{what} 2^{layer_vars} x 2^{token_vars}x2^{in_vars} by W 2^{out_vars}x2^{in_vars}"
        ),
        contracted,
        operands: w.len() + x.len(),
        instance: 1 << (in_vars + layer_vars),
        contraction,
        sumcheck,
        whole,
    });
}

/// `MatMulAct` over `2^{n_b}` products of `2^{n_a} × 2^{n_j}` by `2^{n_j} × 2^{n_c}`.
fn act(
    rng: &mut SplitMix64,
    what: &str,
    (stack_vars, row_vars, inner_vars, col_vars): (usize, usize, usize, usize),
    layout: RightLayout,
    runs: usize,
) {
    let shape = MatMulActShape {
        stack_vars,
        row_vars,
        inner_vars,
        col_vars,
    };
    let a = rng.fps(1 << (stack_vars + row_vars + inner_vars));
    let b = rng.fps(1 << (stack_vars + inner_vars + col_vars));
    let point = rng.fp2s(col_vars + row_vars + stack_vars);
    let relation = MatMulAct::<Fp2>::new(&a, &b, layout, shape).expect("a valid shape");
    let degree = relation.degree();
    // An operand with no variable to contract is used as it is.
    let contracted = usize::from(row_vars > 0) * a.len() + usize::from(col_vars > 0) * b.len();
    assert!(contracted > 0, "every shape here contracts an operand");
    let contraction = fastest(runs, || relation.tables(&point).expect("a point"));
    let tables = relation.tables(&point).expect("a point");
    let sumcheck = fastest(runs, || prove(degree, tables.instance()).into_proof());
    let whole = fastest(runs, || {
        let tables = relation.tables(&point).expect("a point");
        let proven = prove(degree, tables.instance());
        let at = proven.instance_point(0).expect("an instance");
        tables
            .claims(at, proven.instance_evals(0).expect("an instance"))
            .expect("the instance's results")
    });
    report(&Line {
        what: format!(
            "{what} 2^{stack_vars} x 2^{row_vars}x2^{inner_vars} by 2^{inner_vars}x2^{col_vars}"
        ),
        contracted,
        operands: a.len() + b.len(),
        instance: 1 << (inner_vars + stack_vars),
        contraction,
        sumcheck,
        whole,
    });
}

/// Keeps one core busy before anything is timed, so that the first measurement does not run on
/// a core that is still raising its clock.
fn warm_up(rng: &mut SplitMix64, duration: Duration) {
    let shape = MatMulWeightShape {
        layer_vars: 1,
        token_vars: 4,
        out_vars: 4,
        in_vars: 10,
    };
    let w = rng.fps(1 << 15);
    let x = rng.fps(1 << 15);
    let point = rng.fp2s(9);
    let relation = MatMulWeight::<Fp2>::new(&w, &x, shape).expect("a valid shape");
    let start = Instant::now();
    while start.elapsed() < duration {
        black_box(relation.tables(&point).expect("a point"));
    }
}

/// A `MatMulWeight`: a name and `(n_ℓ, n_t, n_o, n_i)`.
type WeightCase = (&'static str, (usize, usize, usize, usize));

/// A `MatMulAct`: a name, `(n_b, n_a, n_j, n_c)` and how `B` is stored.
type ActCase = (&'static str, (usize, usize, usize, usize), RightLayout);

/// The weight shapes: the issue's, `2^6` tokens by `2^8` inputs times `(2^5 × 2^8)ᵀ`, then from
/// `2^3` layers of `2^10` tokens by `2^10` inputs times `(2^10 × 2^10)ᵀ`, more tokens, outputs,
/// inputs and layers, one at a time.
const WEIGHTS: [WeightCase; 10] = [
    ("issue X", (0, 6, 5, 8)),
    ("X", (3, 10, 10, 10)),
    ("X", (3, 11, 10, 10)),
    ("X", (3, 12, 10, 10)),
    ("X", (3, 10, 11, 10)),
    ("X", (3, 10, 12, 10)),
    ("X", (3, 10, 10, 11)),
    ("X", (3, 10, 10, 12)),
    ("X", (4, 10, 10, 10)),
    ("X", (5, 10, 10, 10)),
];

/// The issue's shape, then attention's over `2^4` heads: `QKᵀ` with a head dimension of `2^7`
/// and `2^9` to `2^11` positions, `K` stored transposed; `PV` over as many positions, `V` as it
/// is.
const ACTS: [ActCase; 7] = [
    ("issue", (0, 6, 8, 5), RightLayout::Direct),
    ("QK^T", (4, 9, 7, 9), RightLayout::Transposed),
    ("QK^T", (4, 10, 7, 10), RightLayout::Transposed),
    ("QK^T", (4, 11, 7, 11), RightLayout::Transposed),
    ("PV", (4, 9, 9, 7), RightLayout::Direct),
    ("PV", (4, 10, 10, 7), RightLayout::Direct),
    ("PV", (4, 11, 11, 7), RightLayout::Direct),
];

/// The smoke test's shapes.
const SMOKE_WEIGHTS: [WeightCase; 2] = [("X", (2, 3, 4, 5)), ("X", (0, 0, 3, 4))];
const SMOKE_ACTS: [ActCase; 2] = [
    ("QK^T", (2, 3, 3, 3), RightLayout::Transposed),
    ("PV", (1, 3, 3, 2), RightLayout::Direct),
];

fn main() {
    let mut rng = SplitMix64(0x7E55_E7A0_0057_0001);
    let (weights, acts, runs): (&[WeightCase], &[ActCase], usize) =
        if std::env::args().any(|arg| arg == "--bench") {
            warm_up(&mut rng, Duration::from_millis(500));
            (&WEIGHTS, &ACTS, 5)
        } else {
            (&SMOKE_WEIGHTS, &SMOKE_ACTS, 1)
        };
    header();
    for &(what, shape) in weights {
        weight(&mut rng, what, shape, runs);
    }
    for &(what, shape, layout) in acts {
        act(&mut rng, what, shape, layout, runs);
    }
}
