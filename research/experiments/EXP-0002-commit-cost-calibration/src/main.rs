// SPDX-License-Identifier: Apache-2.0 OR MIT

//! EXP-0002: the calibration harness for `c_c`, the cost per committed 64-bit element of the
//! advice commitment (README.md in this directory).
//!
//! It measures the production kernels unmodified, by path dependency on `tessera-pcs` and
//! `tessera-hash`, as RFC-0020's gate measures them (3 warm-ups, 10 iterations, median, minimum,
//! maximum, coefficient of variation; a coefficient above 3% is flagged and the case rerun once),
//! and the in-cache unit costs that the report's fit `F` multiplies by the counts of `count.py`.
//! Every timed kernel first passes a correctness check, so that a fast but wrong path cannot
//! enter a number.
//!
//! Research code: it prints, it panics on a failed check, and nothing depends on it.
//!
//! ```sh
//! cargo run --release --locked -- all [results/run.json]
//! cargo run --release --locked -- check | units | kernels | bsweep | threads
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::print_stdout, clippy::cast_precision_loss)]

use std::hint::black_box;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use tessera_field::{Field, Goldilocks, PrimeField};
use tessera_hash::{ColumnChains, Digest, Domain, Hasher, hash};
use tessera_pcs::{COSET_SHIFT, NttPlan, RowEncoder};

type Fp = Goldilocks;

/// Rows and transform inputs: the seed of `crates/tessera-pcs/benches/throughput.rs`.
const ROW_SEED: u64 = 0x7E55_E7A0_0047_0401;
/// Mask coefficients, as in that bench.
const MASK_SEED: u64 = 0x7E55_E7A0_0047_0402;
/// The hashed block of rows.
const BLOCK_SEED: u64 = 0x7E55_E7A0_0047_0404;

/// `k = 2^22`: the row width of RFC-0011.
const LOG_K: u32 = 22;
/// `P_A = 200`: the `ni-100` mask length, as the `tessera-pcs` bench uses.
const MASK_COEFFS: usize = 200;
/// `B = 32`: the row block of RFC-0004 'Column digests'.
const ROW_BLOCK: usize = 32;
/// `n = 2^23` columns: the codeword length.
const LOG_COLUMNS: u32 = LOG_K + 1;

const WARMUPS: usize = 3;
const ITERATIONS: usize = 10;
const CV_LIMIT: f64 = 0.03;
/// In-cache microbenchmarks repeat their kernel until one iteration lasts at least this long.
const MIN_ITERATION: Duration = Duration::from_millis(20);

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
}

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).expect("non-zero")
}

// ---------------------------------------------------------------------------------------------
// Measurement protocol (RFC-0020 'Measurement protocol')
// ---------------------------------------------------------------------------------------------

/// Seconds per iteration over the measured iterations.
#[derive(Clone, Copy, Debug)]
struct Stats {
    median: f64,
    min: f64,
    max: f64,
    cv: f64,
    /// The case was rerun because its first coefficient of variation exceeded 3%.
    repeated: bool,
}

fn sample(iteration: &mut impl FnMut() -> Duration) -> Stats {
    for _ in 0..WARMUPS {
        black_box(iteration());
    }
    let mut s: Vec<f64> = (0..ITERATIONS)
        .map(|_| iteration().as_secs_f64())
        .collect();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    let median = (s[n / 2 - 1] + s[n / 2]) / 2.0;
    let mean = s.iter().sum::<f64>() / n as f64;
    let var = s.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    Stats {
        median,
        min: s[0],
        max: s[n - 1],
        cv: var.sqrt() / mean,
        repeated: false,
    }
}

/// 3 warm-ups and 10 measured iterations; a coefficient of variation above 3% reruns the case
/// once, and the rerun is what is reported (flagged if it is still above 3%).
fn measure(mut iteration: impl FnMut() -> Duration) -> Stats {
    let first = sample(&mut iteration);
    if first.cv <= CV_LIMIT {
        return first;
    }
    let mut second = sample(&mut iteration);
    second.repeated = true;
    second
}

/// How many repetitions make one iteration last at least `MIN_ITERATION`.
fn reps_for(mut once: impl FnMut()) -> u32 {
    let start = Instant::now();
    once();
    let one = start.elapsed().max(Duration::from_nanos(1)).as_secs_f64();
    (MIN_ITERATION.as_secs_f64() / one).ceil().clamp(1.0, 1e7) as u32
}

/// One reported case, for the JSON record.
struct Row {
    name: String,
    per: &'static str,
    elements: f64,
    stats: Stats,
}

/// Prints a case and returns its median in nanoseconds per unit.
fn report(rows: &mut Vec<Row>, name: &str, elements: f64, per: &'static str, stats: Stats) -> f64 {
    let ns = |t: f64| t * 1e9 / elements;
    let flag = if stats.cv > CV_LIMIT {
        "  FLAG: CV > 3% after a rerun"
    } else if stats.repeated {
        "  (rerun: first CV > 3%)"
    } else {
        ""
    };
    println!(
        "{name:<52} {:>9.3} ns/{per:<8} min {:>9.3}  max {:>9.3}  CV {:>5.2}%  ({:>9.3} ms){flag}",
        ns(stats.median),
        ns(stats.min),
        ns(stats.max),
        stats.cv * 100.0,
        stats.median * 1e3
    );
    rows.push(Row {
        name: name.to_owned(),
        per,
        elements,
        stats,
    });
    ns(stats.median)
}

// ---------------------------------------------------------------------------------------------
// Assembly anchors: `asm_count.py` counts the ARM64 instructions of these two functions.
// ---------------------------------------------------------------------------------------------

/// One Goldilocks multiplication, `reduce128` included.
#[inline(never)]
fn mul_anchor(a: Fp, b: Fp) -> Fp {
    a * b
}

/// One Cooley–Tukey butterfly `(u, v) ↦ (u + c·v, u − c·v)`, as `ct_butterflies` computes it.
#[inline(never)]
fn ct_butterfly_anchor(u: Fp, v: Fp, c: Fp) -> (Fp, Fp) {
    let t = v * c;
    (u + t, u - t)
}

// ---------------------------------------------------------------------------------------------
// Correctness checks
// ---------------------------------------------------------------------------------------------

fn hex_bytes(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex"))
        .collect()
}

fn digest_hex(d: &Digest) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

fn fp_hex(s: &str) -> Fp {
    Fp::from_u64_reduced(u64::from_str_radix(s.trim_start_matches("0x"), 16).expect("hex"))
}

/// Three products of `spec/vectors/field/goldilocks.json` (edge values), through the anchor.
fn check_field() {
    let cases: [(u64, u64, u64); 3] = [
        (0xffff_ffff_0000_0000, 0xffff_ffff_0000_0000, 0x0000_0000_0000_0001),
        (0x8000_0000_0000_0000, 0x8000_0000_0000_0000, 0xffff_fffe_c000_0001),
        (0x0000_0000_ffff_ffff, 0x0000_0001_0000_0000, 0xffff_ffff_0000_0000),
    ];
    for (a, b, product) in cases {
        let got = mul_anchor(
            black_box(Fp::from_u64_reduced(a)),
            black_box(Fp::from_u64_reduced(b)),
        );
        assert_eq!(got.to_canonical_u64(), product, "field vector {a:#x} * {b:#x}");
    }
    let (u, v, c) = (
        Fp::from_u64_reduced(5),
        Fp::from_u64_reduced(7),
        Fp::from_u64_reduced(11),
    );
    let (x, y) = ct_butterfly_anchor(black_box(u), black_box(v), black_box(c));
    assert_eq!((x, y), (u + v * c, u - v * c), "butterfly anchor");
    println!("check field:     3 edge products of spec/vectors/field/goldilocks.json reproduced through mul_anchor");
}

/// The hand case of `crates/tessera-pcs/src/lib.rs`: `k = 8`, `P = 2`, row `1..=8`, masks 11, 12.
fn check_encoder_hand_case() {
    let fp = Fp::from_u64_reduced;
    let row: Vec<Fp> = (1..=8).map(fp).collect();
    let encoder = RowEncoder::new(8, 2).expect("valid");
    let mut codeword = vec![Fp::ZERO; 16];
    let mut masks = [fp(11), fp(12)].into_iter();
    encoder
        .encode_row(&row, &mut || masks.next().unwrap_or_default(), &mut codeword)
        .expect("lengths");
    let mut g = codeword.clone();
    NttPlan::new(16)
        .expect("size")
        .coset_intt(&mut g, COSET_SHIFT)
        .expect("length");
    let expected_top = [fp(11), fp(12), fp(0), fp(0), fp(0), fp(0), fp(0), fp(0)];
    assert_eq!(&g[8..], &expected_top, "mask coefficients in the top half");
    let mut f: Vec<Fp> = (0..8).map(|i| g[i] + g[i + 8]).collect();
    NttPlan::new(8).expect("size").ntt(&mut f).expect("length");
    assert_eq!(f, row, "the fold modulo X^8 - 1 evaluates to the row");
    println!("check encoder:   hand case k = 8, P = 2 (row 1..=8, masks 11, 12): coefficients [11, 12, 0, ...] and the row recovered");
}

/// At full size: `encode_row` equals `NttPlan::intt` → mask → `NttPlan::coset_ntt` symbol for
/// symbol (an independent path: bit-reversal permutation and the `Coset` shift instead of the
/// spread table), and the codeword's coefficients vanish from `k + P` upward.
fn check_encoder_full(encoder: &RowEncoder, row: &[Fp], codeword: &[Fp]) {
    let k = row.len();
    let n = 2 * k;
    let plan_k = NttPlan::new(k).expect("size");
    let plan_n = NttPlan::new(n).expect("size");
    let mut f = row.to_vec();
    plan_k.intt(&mut f).expect("length");
    let mut g = vec![Fp::ZERO; n];
    g[..k].copy_from_slice(&f);
    let mut masks = SplitMix64(MASK_SEED);
    for i in 0..encoder.mask_coeffs() {
        let m = masks.fp();
        g[i] -= m;
        g[k + i] = m;
    }
    plan_n.coset_ntt(&mut g, COSET_SHIFT).expect("length");
    assert!(
        g == codeword,
        "encode_row differs from intt -> mask -> coset_ntt"
    );
    let mut c = codeword.to_vec();
    plan_n.coset_intt(&mut c, COSET_SHIFT).expect("length");
    assert!(
        c[k + encoder.mask_coeffs()..].iter().all(|x| *x == Fp::ZERO),
        "codeword degree exceeds k + P - 1"
    );
    println!(
        "check encoder:   k = 2^{LOG_K}, P = {}: encode_row == intt -> mask -> coset_ntt on all {n} symbols; degree < k + P",
        encoder.mask_coeffs()
    );
}

/// RFC-0004 'Column digests', assembled here from the incremental `Hasher`: `d_j[0] =
/// H(column-init, id ‖ u64(j))`, `d_j[i+1] = H(column-step, d_j[i] ‖ symbols of block i at j)`.
fn reference_leaf(id: &Digest, j: usize, blocks: &[&[Fp]], columns: usize) -> Digest {
    let mut d = Hasher::new(Domain::ColumnInit)
        .update(id)
        .update(&(j as u64).to_le_bytes())
        .finalize();
    for block in blocks {
        let mut h = Hasher::new(Domain::ColumnStep);
        h.update(&d);
        for row in block.chunks_exact(columns) {
            h.update(&row[j].to_canonical_u64().to_le_bytes());
        }
        d = h.finalize();
    }
    d
}

/// The same block absorbed `repeats` times.
fn reference_leaf_repeated(id: &Digest, j: usize, block: &[Fp], columns: usize, repeats: usize) -> Digest {
    let blocks: Vec<&[Fp]> = (0..repeats).map(|_| block).collect();
    reference_leaf(id, j, &blocks, columns)
}

/// The case `3d405d23…` of `spec/vectors/hash/hash.json`: 5 rows, 3 columns, `B = 2`, a partial
/// last block; reproduced by `ColumnChains` and by `reference_leaf`.
fn check_chain_vector() {
    let id: Digest = hex_bytes("3d405d23706e8a023f3fa025db2e6f48599a104983555fbb95d61eb703bba406")
        .try_into()
        .expect("32 bytes");
    let rows_hex = [
        ["0x8eb74278b3c8e984", "0x6f8653dbc20365bf", "0xc7b8fe2cfeef3ec1"],
        ["0x97ad026e360c2411", "0x9c31b4e4ea0b1d7a", "0x1c14696411cf0bc3"],
        ["0x83ce97f12f7c6431", "0x2e4b86f55d495bc1", "0x695f73cff282779d"],
        ["0x5f92ae0f8b134827", "0xe1c4a28ff6d99ef0", "0x4b3eefea430c38bb"],
        ["0xc7a36480509a74a4", "0xa530a65a24b79d74", "0x93b8b4084e6f6614"],
    ];
    let expected = [
        "de0f902e501cb4b2387cad14f5e52739e70a1e5480ffe4744834dea22c2e20c0",
        "c340a23015e9657002f7e55c084d3a6f5b67594027aeec4408a141b959d55e67",
        "59a6453fcd8aa11fcad03ea4745b98c357ca360a4ca0ea16a5a203bce6aec09e",
    ];
    let rows: Vec<Fp> = rows_hex.iter().flatten().map(|s| fp_hex(s)).collect();
    let blocks: [&[Fp]; 3] = [&rows[0..6], &rows[6..12], &rows[12..15]];
    let mut chains = ColumnChains::new(&id, nz(3), nz(2));
    for block in blocks {
        chains.absorb_block(block).expect("block");
    }
    let leaves = chains.finish();
    for (j, (leaf, exp)) in leaves.iter().zip(expected).enumerate() {
        assert_eq!(digest_hex(leaf), exp, "ColumnChains leaf {j} against the vector");
        assert_eq!(
            reference_leaf(&id, j, &blocks, 3),
            *leaf,
            "reference derivation of leaf {j}"
        );
    }
    println!("check chains:    hash.json case 3d405d23… (5 rows, 3 columns, B = 2) reproduced by ColumnChains and by the reference derivation");
}

// ---------------------------------------------------------------------------------------------
// Unit costs (in cache)
// ---------------------------------------------------------------------------------------------

/// Nanoseconds per unit operation, measured on in-cache inputs.
#[derive(Clone, Copy, Debug)]
struct Units {
    /// Per radix-2 butterfly (multiplication, addition, subtraction, loads and stores), from
    /// `encode_row` with `P = 0` at small `k`, which has no bit-reversal pass.
    butterfly: f64,
    /// Per point of `bit_reverse_scaled`, from `ntt` at small `n` given `butterfly`.
    bitrev: f64,
    /// Per Goldilocks multiplication of an array element by a constant, in place (throughput).
    mul: f64,
    /// Per `hash(ColumnStep, ·)` call, excluding compressions.
    hash_call: f64,
    /// Per BLAKE3 compression.
    compression: f64,
}

/// Least squares of `y = a·x1 + b·x2` through the origin (two regressors).
fn least_squares_2(xs: &[(f64, f64, f64)]) -> (f64, f64) {
    let (mut s11, mut s12, mut s22, mut s1y, mut s2y) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &(x1, x2, y) in xs {
        s11 += x1 * x1;
        s12 += x1 * x2;
        s22 += x2 * x2;
        s1y += x1 * y;
        s2y += x2 * y;
    }
    let det = s11 * s22 - s12 * s12;
    ((s1y * s22 - s2y * s12) / det, (s11 * s2y - s12 * s1y) / det)
}

/// Least squares of `y = a + b·x`.
fn least_squares_affine(xs: &[(f64, f64)]) -> (f64, f64) {
    let n = xs.len() as f64;
    let (sx, sy) = xs.iter().fold((0.0, 0.0), |(a, b), &(x, y)| (a + x, b + y));
    let (sxx, sxy) = xs
        .iter()
        .fold((0.0, 0.0), |(a, b), &(x, y)| (a + x * x, b + x * y));
    let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    ((sy - slope * sx) / n, slope)
}

fn units(rows: &mut Vec<Row>, rng: &mut SplitMix64) -> Units {
    println!("\n== unit costs, in cache (L1D 128 KiB per performance core) ==");

    // Multiplication throughput: 2^12 elements in place by a constant.
    let mut values = rng.fps(1 << 12);
    let c = rng.fp();
    let reps = reps_for(|| {
        for x in black_box(&mut values).iter_mut() {
            *x *= black_box(c);
        }
    });
    let stats = measure(|| {
        let start = Instant::now();
        for _ in 0..reps {
            for x in black_box(&mut values).iter_mut() {
                *x *= black_box(c);
            }
        }
        start.elapsed() / reps
    });
    let mul = report(rows, &format!("unit: mul, 2^12 in place by a constant (x{reps})"), 4096.0, "mul", stats);

    // Multiplication latency: a dependent chain.
    let mut acc = rng.fp();
    let steps = 1u32 << 16;
    let reps = reps_for(|| {
        for _ in 0..steps {
            acc *= black_box(c);
        }
    });
    let stats = measure(|| {
        let start = Instant::now();
        for _ in 0..reps {
            for _ in 0..steps {
                acc *= black_box(c);
            }
        }
        start.elapsed() / reps
    });
    black_box(acc);
    report(rows, &format!("unit: mul, dependent chain of 2^16 (x{reps})"), f64::from(steps), "mul", stats);

    // Butterflies: encode_row with P = 0 at k = 2^9 .. 2^12 (no bit reversal; codeword ≤ 64 KiB).
    let mut fits = Vec::new();
    for log_k in 9..=12u32 {
        let k = 1usize << log_k;
        let encoder = RowEncoder::new(k, 0).expect("valid");
        let row = rng.fps(k);
        let mut codeword = vec![Fp::ZERO; 2 * k];
        let mut none = || Fp::ZERO;
        let reps = reps_for(|| {
            encoder
                .encode_row(black_box(&row), &mut none, black_box(&mut codeword))
                .expect("lengths");
        });
        let stats = measure(|| {
            let start = Instant::now();
            for _ in 0..reps {
                encoder
                    .encode_row(black_box(&row), &mut none, black_box(&mut codeword))
                    .expect("lengths");
            }
            start.elapsed() / reps
        });
        // (k/2)·log k + k·(log k + 1) butterflies, k spread multiplications.
        let butterflies = (k / 2) as f64 * f64::from(log_k) + k as f64 * f64::from(log_k + 1);
        let per_bf = (stats.median * 1e9 - k as f64 * mul) / butterflies;
        report(rows, &format!("unit: encode_row 2^{log_k}, P = 0 (x{reps})"), k as f64, "value", stats);
        println!("    {butterflies:.0} butterflies: {per_bf:.4} ns per butterfly after k multiplications at {mul:.4} ns");
        fits.push((butterflies, k as f64, stats.median * 1e9));
    }
    // Fit T = b·butterflies + m'·k over the four sizes as a cross-check on the subtraction above.
    let (b_fit, m_fit) = least_squares_2(&fits);
    let butterfly = fits
        .iter()
        .map(|&(bf, k, t)| (t - k * mul) / bf)
        .sum::<f64>()
        / fits.len() as f64;
    println!(
        "    per butterfly: {butterfly:.4} ns (mean of the four); two-regressor fit gives {b_fit:.4} ns per butterfly and {m_fit:.4} ns per spread value"
    );

    // Bit reversal: ntt at n = 2^10 .. 2^13 given the butterfly cost.
    let mut bitrevs = Vec::new();
    for log_n in 10..=13u32 {
        let n = 1usize << log_n;
        let plan = NttPlan::new(n).expect("size");
        let mut values = rng.fps(n);
        let reps = reps_for(|| plan.ntt(black_box(&mut values)).expect("length"));
        let stats = measure(|| {
            let start = Instant::now();
            for _ in 0..reps {
                plan.ntt(black_box(&mut values)).expect("length");
            }
            start.elapsed() / reps
        });
        let butterflies = (n / 2) as f64 * f64::from(log_n);
        let per_point = (stats.median * 1e9 - butterflies * butterfly) / n as f64;
        report(rows, &format!("unit: ntt 2^{log_n} (x{reps})"), n as f64, "point", stats);
        println!("    {butterflies:.0} butterflies: {per_point:.4} ns per point of bit reversal after them");
        bitrevs.push(per_point);
    }
    let bitrev = bitrevs.iter().sum::<f64>() / bitrevs.len() as f64;
    println!("    per point of bit reversal: {bitrev:.4} ns (mean of the four)");

    // BLAKE3: hash(ColumnStep, 64·c bytes) for c compressions.
    let mut hashes = Vec::new();
    for c in [1usize, 2, 3, 4, 5, 8, 16] {
        let input: Vec<u8> = (0..64 * c).map(|i| (i * 31 + 7) as u8).collect();
        let reps = reps_for(|| {
            black_box(hash(Domain::ColumnStep, black_box(&input)));
        });
        let stats = measure(|| {
            let start = Instant::now();
            for _ in 0..reps {
                black_box(hash(Domain::ColumnStep, black_box(&input)));
            }
            start.elapsed() / reps
        });
        let ns = report(rows, &format!("unit: hash(column-step, {} bytes = {c} compressions) (x{reps})", 64 * c), 1.0, "call", stats);
        hashes.push((c as f64, ns));
    }
    let (hash_call, compression) = least_squares_affine(&hashes);
    println!("    fit: {hash_call:.3} ns per call + {compression:.3} ns per compression");
    for &(c, ns) in &hashes {
        println!("      c = {c:>2}: measured {ns:>8.3} ns, fit {:>8.3} ns", hash_call + compression * c);
    }
    // The chain's own input length: 32 + 8·32 = 288 bytes, 5 compressions.
    let input: Vec<u8> = (0..288).map(|i| (i * 31 + 7) as u8).collect();
    let reps = reps_for(|| {
        black_box(hash(Domain::ColumnStep, black_box(&input)));
    });
    let stats = measure(|| {
        let start = Instant::now();
        for _ in 0..reps {
            black_box(hash(Domain::ColumnStep, black_box(&input)));
        }
        start.elapsed() / reps
    });
    let ns288 = report(rows, &format!("unit: hash(column-step, 288 bytes = 5 compressions) (x{reps})"), 1.0, "call", stats);
    println!("    288 bytes: measured {ns288:.3} ns, fit {:.3} ns", hash_call + 5.0 * compression);

    // Memory bandwidth context: a streaming copy of 2^23 elements.
    let a = rng.fps(1 << 23);
    let mut b = vec![Fp::ZERO; 1 << 23];
    let stats = measure(|| {
        let start = Instant::now();
        black_box(&mut b).copy_from_slice(black_box(&a));
        start.elapsed()
    });
    report(rows, "stream copy 2^23 elements (64 MiB read + 64 MiB write)", (1u64 << 23) as f64, "point", stats);
    println!("    {:.1} GB/s (read + write), one thread", 2.0 * 64.0 * 1024.0 * 1024.0 / stats.median / 1e9);

    Units {
        butterfly,
        bitrev,
        mul,
        hash_call,
        compression,
    }
}

// ---------------------------------------------------------------------------------------------
// The four kernels of the fit
// ---------------------------------------------------------------------------------------------

/// Median nanoseconds per committed element of each kernel.
#[derive(Clone, Copy, Debug)]
struct Kernels {
    intt: f64,
    coset_ntt: f64,
    encode_row: f64,
    chain: f64,
}

/// The commitment id of the chains: the ASCII bytes `EXP-0002`, zero-padded.
fn commitment_id() -> Digest {
    let mut id = [0u8; 32];
    id[..8].copy_from_slice(b"EXP-0002");
    id
}

fn transform_case(
    rows: &mut Vec<Row>,
    name: &str,
    input: &[Fp],
    elements: f64,
    mut f: impl FnMut(&mut [Fp]),
) -> f64 {
    let mut values = input.to_vec();
    let stats = measure(|| {
        values.copy_from_slice(input);
        let start = Instant::now();
        f(black_box(&mut values));
        start.elapsed()
    });
    report(rows, name, elements, "element", stats)
}

fn kernels(rows: &mut Vec<Row>, rng: &mut SplitMix64) -> Kernels {
    println!("\n== kernels, one thread, per committed element (k = 2^{LOG_K} elements per row) ==");
    let k = 1usize << LOG_K;
    let n = 2 * k;

    let plan_k = NttPlan::new(k).expect("size");
    let plan_n = NttPlan::new(n).expect("size");
    let input_k = rng.fps(k);
    let input_n = rng.fps(n);
    let intt = transform_case(rows, "intt 2^22 (1 point per element)", &input_k, k as f64, |v| {
        plan_k.intt(v).expect("length");
    });
    let coset_ntt = transform_case(
        rows,
        "coset_ntt 2^23, shift 7 (2 points per element)",
        &input_n,
        k as f64,
        |v| plan_n.coset_ntt(v, COSET_SHIFT).expect("length"),
    );
    println!("    coset_ntt 2^23: {:.3} ns per point", coset_ntt / 2.0);

    // The encoder, checked at full size before it is timed.
    let encoder = RowEncoder::new(k, MASK_COEFFS).expect("valid");
    let row = rng.fps(k);
    let mut codeword = vec![Fp::ZERO; n];
    let mut masks = SplitMix64(MASK_SEED);
    encoder
        .encode_row(&row, &mut || masks.fp(), &mut codeword)
        .expect("lengths");
    check_encoder_full(&encoder, &row, &codeword);
    let reference = codeword.clone();
    let stats = measure(|| {
        let start = Instant::now();
        encoder
            .encode_row(black_box(&row), &mut || masks.fp(), &mut codeword)
            .expect("lengths");
        start.elapsed()
    });
    let encode_row = report(rows, &format!("encode_row 2^22, P = {MASK_COEFFS}"), k as f64, "element", stats);
    println!("    encode_row: {:.3} ns per codeword symbol", encode_row / 2.0);

    // The chains over one block of B = 32 rows and 2^23 columns (2 GiB), checked after timing
    // against the reference derivation on 64 sampled columns.
    let columns = 1usize << LOG_COLUMNS;
    let id = commitment_id();
    let mut block_rng = SplitMix64(BLOCK_SEED);
    let block = block_rng.fps(ROW_BLOCK * columns);
    let mut chains = ColumnChains::new(&id, nz(columns), nz(ROW_BLOCK));
    let mut absorbed = 0usize;
    let stats = measure(|| {
        let start = Instant::now();
        chains.absorb_block(black_box(&block)).expect("block");
        absorbed += 1;
        start.elapsed()
    });
    let chain = report(
        rows,
        &format!("absorb_block 2^{LOG_COLUMNS} columns, B = {ROW_BLOCK} (2 symbols per element)"),
        (ROW_BLOCK * columns / 2) as f64,
        "element",
        stats,
    );
    println!("    chains: {:.3} ns per symbol, {:.3} GB/s of symbols", chain / 2.0, 8.0 / (chain / 2.0));
    let leaves = chains.finish();
    for s in 0..64usize {
        let j = if s == 63 { columns - 1 } else { s * (columns / 63) };
        assert_eq!(
            leaves[j],
            reference_leaf_repeated(&id, j, &block, columns, absorbed),
            "leaf {j} after {absorbed} blocks against the reference derivation"
        );
    }
    let mut record = Hasher::new(Domain::ColumnMerge);
    for leaf in &leaves {
        record.update(leaf);
    }
    println!(
        "check chains:    64 sampled leaves of the 2^{LOG_COLUMNS}-column block after {absorbed} absorptions equal the reference derivation; leaf_0 {}; record digest of all leaves {}",
        digest_hex(&leaves[0]),
        digest_hex(&record.finalize())
    );

    let _ = reference;
    Kernels {
        intt,
        coset_ntt,
        encode_row,
        chain,
    }
}

// ---------------------------------------------------------------------------------------------
// The fit F
// ---------------------------------------------------------------------------------------------

fn fit(units: &Units, kernels: &Kernels) -> bool {
    println!("\n== fit F: predicted = counts × unit costs, per committed element ==");
    let Units {
        butterfly: b,
        bitrev: r,
        mul: m,
        hash_call: o,
        compression: u,
    } = *units;
    // Counts per committed element (count.py): intt 11 butterflies + 1 scaled bit-reversal point;
    // coset_ntt 23 butterflies + 2 bit-reversal points + about 1 group twiddle multiplication;
    // encode_row 34 butterflies + 1 spread multiplication; chain (call + 5 compressions) / 16.
    let cases = [
        ("intt 2^22", kernels.intt, 11.0 * b + r + m),
        ("coset_ntt 2^23", kernels.coset_ntt, 23.0 * b + 2.0 * r + m),
        ("encode_row 2^22, P = 200", kernels.encode_row, 34.0 * b + m),
        ("absorb_block 2^23, B = 32", kernels.chain, (o + 5.0 * u) / 16.0),
    ];
    println!("{:<28} {:>12} {:>12} {:>9}  within [0.5, 2]", "kernel", "measured ns", "predicted ns", "ratio");
    let mut pass = true;
    for (name, measured, predicted) in cases {
        let ratio = predicted / measured;
        let ok = (0.5..=2.0).contains(&ratio);
        pass &= ok;
        println!(
            "{name:<28} {measured:>12.3} {predicted:>12.3} {ratio:>9.3}  {}",
            if ok { "yes" } else { "NO" }
        );
    }
    println!("F {}", if pass { "passes" } else { "FAILS" });
    pass
}

// ---------------------------------------------------------------------------------------------
// The B sweep and thread scaling
// ---------------------------------------------------------------------------------------------

fn b_sweep(rows: &mut Vec<Row>, rng: &mut SplitMix64) {
    println!("\n== B sweep, 2^20 columns, one thread ==");
    let columns = 1usize << 20;
    let max_b = 128usize;
    let block = rng.fps(max_b * columns);
    let id = commitment_id();
    for b in [8usize, 16, 28, 32, 60, 64, 124, 128] {
        let mut chains = ColumnChains::new(&id, nz(columns), nz(b));
        let slice = &block[..b * columns];
        let stats = measure(|| {
            let start = Instant::now();
            chains.absorb_block(black_box(slice)).expect("block");
            start.elapsed()
        });
        let per_element = report(rows, &format!("absorb_block 2^20 columns, B = {b}"), (b * columns / 2) as f64, "element", stats);
        let compressions = (32 + 8 * b).div_ceil(64);
        println!(
            "    {compressions} compressions per column-step of {} bytes: {:.4} per symbol; {:.3} ns per symbol",
            32 + 8 * b,
            compressions as f64 / b as f64,
            per_element / 2.0
        );
    }
}

fn threads(rows: &mut Vec<Row>, rng: &mut SplitMix64) {
    println!("\n== thread scaling (aggregate ns per committed element) ==");
    let k = 1usize << LOG_K;
    let encoder = RowEncoder::new(k, MASK_COEFFS).expect("valid");
    let row = rng.fps(k);
    let mut reference = vec![Fp::ZERO; 2 * k];
    let mut masks = SplitMix64(MASK_SEED);
    encoder
        .encode_row(&row, &mut || masks.fp(), &mut reference)
        .expect("lengths");
    check_encoder_full(&encoder, &row, &reference);
    // The threads share the encoder's tables by reference, as a GPU would share its twiddles.
    let encoder = &encoder;

    for t in [1usize, 2, 4, 8, 12] {
        let rows_t: Vec<Vec<Fp>> = (0..t).map(|_| row.clone()).collect();
        let mut codewords: Vec<Vec<Fp>> = (0..t).map(|_| vec![Fp::ZERO; 2 * k]).collect();
        let mut mask_rngs: Vec<SplitMix64> = (0..t).map(|_| SplitMix64(MASK_SEED)).collect();
        let stats = measure(|| {
            for m in &mut mask_rngs {
                *m = SplitMix64(MASK_SEED);
            }
            let start = Instant::now();
            std::thread::scope(|s| {
                for ((r, cw), m) in rows_t.iter().zip(codewords.iter_mut()).zip(mask_rngs.iter_mut()) {
                    s.spawn(move || {
                        encoder
                            .encode_row(black_box(r), &mut || m.fp(), black_box(cw))
                            .expect("lengths");
                    });
                }
            });
            start.elapsed()
        });
        report(rows, &format!("encode_row 2^22, P = {MASK_COEFFS}, {t} threads"), (t * k) as f64, "element", stats);
        for cw in &codewords {
            assert!(*cw == reference, "a thread's codeword differs from the reference");
        }
    }
    println!("check encoder:   every thread's codeword equals the single-thread reference, at every thread count");

    let columns = 1usize << LOG_COLUMNS;
    let id = commitment_id();
    let mut block_rng = SplitMix64(BLOCK_SEED);
    let block = block_rng.fps(ROW_BLOCK * columns);
    for t in [1usize, 2, 4, 8, 12] {
        let bounds: Vec<usize> = (0..=t).map(|i| columns * i / t).collect();
        let shards: Vec<Vec<Fp>> = (0..t)
            .map(|i| {
                let (lo, hi) = (bounds[i], bounds[i + 1]);
                let mut shard = Vec::with_capacity(ROW_BLOCK * (hi - lo));
                for r in 0..ROW_BLOCK {
                    shard.extend_from_slice(&block[r * columns + lo..r * columns + hi]);
                }
                shard
            })
            .collect();
        let mut chains: Vec<ColumnChains> = (0..t)
            .map(|i| ColumnChains::new(&id, nz(bounds[i + 1] - bounds[i]), nz(ROW_BLOCK)))
            .collect();
        let mut absorbed = 0usize;
        let stats = measure(|| {
            let start = Instant::now();
            std::thread::scope(|s| {
                for (c, shard) in chains.iter_mut().zip(shards.iter()) {
                    s.spawn(move || c.absorb_block(black_box(shard)).expect("block"));
                }
            });
            absorbed += 1;
            start.elapsed()
        });
        report(
            rows,
            &format!("absorb_block 2^{LOG_COLUMNS} columns, B = {ROW_BLOCK}, {t} threads"),
            (ROW_BLOCK * columns / 2) as f64,
            "element",
            stats,
        );
        for (i, c) in chains.into_iter().enumerate() {
            let leaves = c.finish();
            let w = leaves.len();
            for j in [0, w / 3, 2 * w / 3, w - 1] {
                assert_eq!(
                    leaves[j],
                    reference_leaf_repeated(&id, j, &shards[i], w, absorbed),
                    "shard {i} leaf {j} after {absorbed} blocks"
                );
            }
        }
    }
    println!("check chains:    4 sampled leaves per shard equal the reference derivation, at every thread count");
}

// ---------------------------------------------------------------------------------------------

fn json(rows: &[Row], units: Option<&Units>, kernels: Option<&Kernels>, fit_pass: Option<bool>) -> String {
    let mut out = String::from("{\n  \"protocol\": {\"warmups\": 3, \"iterations\": 10, \"cv_limit\": 0.03},\n  \"rows\": [\n");
    for (i, row) in rows.iter().enumerate() {
        let ns = |t: f64| t * 1e9 / row.elements;
        out.push_str(&format!(
            "    {{\"name\": {:?}, \"per\": {:?}, \"elements\": {}, \"median_ns\": {:.4}, \"min_ns\": {:.4}, \"max_ns\": {:.4}, \"cv\": {:.5}, \"repeated\": {}}}{}\n",
            row.name,
            row.per,
            row.elements,
            ns(row.stats.median),
            ns(row.stats.min),
            ns(row.stats.max),
            row.stats.cv,
            row.stats.repeated,
            if i + 1 < rows.len() { "," } else { "" }
        ));
    }
    out.push_str("  ]");
    if let Some(u) = units {
        out.push_str(&format!(
            ",\n  \"units_ns\": {{\"butterfly\": {:.4}, \"bitrev_point\": {:.4}, \"mul\": {:.4}, \"hash_call\": {:.3}, \"compression\": {:.3}}}",
            u.butterfly, u.bitrev, u.mul, u.hash_call, u.compression
        ));
    }
    if let Some(k) = kernels {
        out.push_str(&format!(
            ",\n  \"kernels_ns_per_element\": {{\"intt\": {:.4}, \"coset_ntt\": {:.4}, \"encode_row\": {:.4}, \"chain\": {:.4}}}",
            k.intt, k.coset_ntt, k.encode_row, k.chain
        ));
    }
    if let Some(p) = fit_pass {
        out.push_str(&format!(",\n  \"fit_passes\": {p}"));
    }
    out.push_str("\n}\n");
    out
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().map_or("all", String::as_str);
    let json_path = args.get(1).cloned();
    println!("EXP-0002 calibration harness: mode {mode}; protocol {WARMUPS} warm-ups, {ITERATIONS} iterations, CV limit {:.0}%", CV_LIMIT * 100.0);

    println!("\n== correctness checks ==");
    check_field();
    check_encoder_hand_case();
    check_chain_vector();

    let mut rows = Vec::new();
    let mut rng = SplitMix64(ROW_SEED);
    let (mut u, mut k, mut p) = (None, None, None);
    match mode {
        "check" => {}
        "units" => {
            u = Some(units(&mut rows, &mut rng));
        }
        "kernels" => {
            let un = units(&mut rows, &mut rng);
            let ke = kernels(&mut rows, &mut rng);
            p = Some(fit(&un, &ke));
            u = Some(un);
            k = Some(ke);
        }
        "bsweep" => b_sweep(&mut rows, &mut rng),
        "threads" => threads(&mut rows, &mut rng),
        "all" => {
            let un = units(&mut rows, &mut rng);
            let ke = kernels(&mut rows, &mut rng);
            p = Some(fit(&un, &ke));
            u = Some(un);
            k = Some(ke);
            b_sweep(&mut rows, &mut rng);
            threads(&mut rows, &mut rng);
        }
        other => {
            eprintln!("unknown mode {other}: all | check | units | kernels | bsweep | threads");
            std::process::exit(2);
        }
    }
    if let Some(path) = json_path {
        std::fs::write(&path, json(&rows, u.as_ref(), k.as_ref(), p)).expect("write json");
        println!("\nwrote {path}");
    }
}
