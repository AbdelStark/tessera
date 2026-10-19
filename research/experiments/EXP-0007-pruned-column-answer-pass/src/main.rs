// SPDX-License-Identifier: Apache-2.0 OR MIT

//! EXP-0007: the output-pruned DIF coset NTT of the column-answer pass (P5 of IDEA-0016) against
//! a full coset NTT plus a gather, in one harness (README.md in this directory).
//!
//! Six arms per `τ` and query set, on one thread:
//!
//! - `full_coef`: the harness's radix-2 DIF kernel over all `log2 n` levels, then a gather;
//! - `pruned_coef`: the same kernel pruned by output (skip / one-sided descent / full level);
//! - `fixed8_coef`: eight full levels, then one descent per query (the idea note's accounting);
//! - `coset_ntt_coef`: the production `NttPlan::coset_ntt`, then a gather (context, reference);
//! - `encode_row_eval`: the production `RowEncoder::encode_row`, then a gather (reference);
//! - `pruned_eval`: the production `NttPlan::intt`, the RFC-0011 mask, then the pruned kernel.
//!
//! Every candidate's symbols are compared bit for bit with the production reference before and
//! after it is timed, so a fast but wrong path cannot enter a number. Protocol: 3 warm-ups, 10
//! measured iterations, median, minimum, maximum, coefficient of variation; above 3% the arm is
//! rerun once (RFC-0020 'Measurement protocol').
//!
//! Research code: it prints, it panics on a failed check, and nothing depends on it.
//!
//! ```sh
//! cargo run --release --locked | tee results/run.txt
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::print_stdout, clippy::cast_precision_loss, clippy::too_many_arguments)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use tessera_field::{Field, Goldilocks, PrimeField};
use tessera_hash::{Digest, Domain, hash};
use tessera_pcs::{COSET_SHIFT, NttPlan, RowEncoder};

type Fp = Goldilocks;

/// The row: the seed of `crates/tessera-pcs/benches/throughput.rs` and EXP-0002.
const ROW_SEED: u64 = 0x7E55_E7A0_0047_0401;
/// The `P` mask coefficients, as in that bench.
const MASK_SEED: u64 = 0x7E55_E7A0_0047_0402;
/// The queried column indices.
const QUERY_SEED: u64 = 0x7E55_E7A0_0047_0407;

/// `k = 2^22`: the row width of RFC-0011.
const LOG_K: u32 = 22;
/// `P_A = 200`: the `ni-100` mask length.
const MASK_COEFFS: usize = 200;
/// The two `τ` of the pre-registration.
const TAUS: [usize; 2] = [130, 163];
/// The idea note's fixed depth: `⌈log2 163⌉ = 8` full levels.
const FIXED_LEVELS: u32 = 8;

const WARMUPS: usize = 3;
const ITERATIONS: usize = 10;
const CV_LIMIT: f64 = 0.03;
/// Groups of at most this many entries are finished level by level: 32 KiB of values in the
/// first-level cache, as `crates/tessera-pcs/src/ntt.rs` does.
const LEAF: usize = 1 << 12;

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

// ---------------------------------------------------------------------------------------------
// Twiddles and the DIF kernel (the remainder tree of crates/tessera-pcs/src/ntt.rs, subgroup)
// ---------------------------------------------------------------------------------------------

/// `rev(i)` over `bits` bits.
#[inline]
fn reverse_bits(i: usize, bits: u32) -> usize {
    i.reverse_bits()
        .checked_shr(usize::BITS - bits)
        .unwrap_or(0)
}

/// `out[t] = scale·base^{rev(t)}` for `t < 2^bits`.
fn bit_reversed_powers(base: Fp, bits: u32, scale: Fp) -> Vec<Fp> {
    let mut squares = Vec::with_capacity(bits as usize);
    let mut square = base;
    for _ in 0..bits {
        squares.push(square);
        square = square.square();
    }
    let mut out = Vec::with_capacity(1 << bits);
    out.push(scale);
    for factor in squares.iter().rev() {
        let half = out.len();
        out.extend_from_within(..);
        for x in &mut out[half..] {
            *x *= *factor;
        }
    }
    out
}

/// `Z[t] = ω_n^{rev(t)}` for `t < n/2`, `rev` over `log2 n − 1` bits, `ω_n = 7^((P − 1)/n)`.
fn twiddles(log_n: u32) -> Vec<Fp> {
    bit_reversed_powers(Fp::two_adic_generator(log_n), log_n - 1, Fp::ONE)
}

/// `7^i` for `i < len`.
fn powers_of_shift(len: usize) -> Vec<Fp> {
    let mut out = Vec::with_capacity(len);
    let mut p = Fp::ONE;
    for _ in 0..len {
        out.push(p);
        p *= COSET_SHIFT;
    }
    out
}

/// Exact operation counts of one arm (E0): radix-2 butterflies and multiply-adds.
#[derive(Clone, Copy, Debug, Default)]
struct Ops {
    butterflies: u64,
    muladds: u64,
}

/// `(u, v) ↦ (u + c·v, u − c·v)` between the halves of a group: `f mod (X^m − c)` and
/// `f mod (X^m + c)` from `f mod (X^{2m} − c²)`.
#[inline(always)]
fn butterflies(lo: &mut [Fp], hi: &mut [Fp], c: Fp) {
    for (u, v) in lo.iter_mut().zip(hi.iter_mut()) {
        let t = *v * c;
        *v = *u - t;
        *u += t;
    }
}

/// [`butterflies`] with `c = 1`: the first group of every level on the subgroup.
#[inline(always)]
fn unit_butterflies(lo: &mut [Fp], hi: &mut [Fp]) {
    for (u, v) in lo.iter_mut().zip(hi.iter_mut()) {
        let t = *v;
        *v = *u - t;
        *u += t;
    }
}

/// One level over the group `group` of its level, split into `lo` and `hi`.
#[inline(always)]
fn level(lo: &mut [Fp], hi: &mut [Fp], tw: &[Fp], group: usize, ops: &mut Ops) {
    ops.butterflies += lo.len() as u64;
    if group == 0 {
        unit_butterflies(lo, hi);
    } else {
        butterflies(lo, hi, tw[group]);
    }
}

/// Every level of the group `values`, the `group`-th of its level: natural-order coefficients
/// in, evaluations at `ω_n^{rev(p)}` in position `p` out.
fn full_group(values: &mut [Fp], tw: &[Fp], group: usize, ops: &mut Ops) {
    if values.len() <= LEAF {
        full_leaf(values, tw, group, ops);
        return;
    }
    let half = values.len() / 2;
    let (lo, hi) = values.split_at_mut(half);
    level(lo, hi, tw, group, ops);
    full_group(lo, tw, 2 * group, ops);
    full_group(hi, tw, 2 * group + 1, ops);
}

/// The levels of a cache-resident group, one level at a time; below group `g`, the groups of
/// the `d`-th level are `g·2^d, …, g·2^d + 2^d − 1`.
fn full_leaf(values: &mut [Fp], tw: &[Fp], group: usize, ops: &mut Ops) {
    let mut half = values.len() / 2;
    let mut first = group;
    let mut count = 1;
    while half > 0 {
        ops.butterflies += (half * count) as u64;
        let mut groups = values
            .chunks_exact_mut(2 * half)
            .zip(&tw[first..first + count]);
        if first == 0 {
            if let Some((chunk, _)) = groups.next() {
                let (lo, hi) = chunk.split_at_mut(half);
                unit_butterflies(lo, hi);
            }
        }
        for (chunk, &z) in groups {
            let (lo, hi) = chunk.split_at_mut(half);
            butterflies(lo, hi, z);
        }
        half /= 2;
        first *= 2;
        count *= 2;
    }
}

// ---------------------------------------------------------------------------------------------
// Output pruning
// ---------------------------------------------------------------------------------------------

/// A queried symbol: its leaf position `rev_n(j)` in the kernel's output order, and the slot of
/// the answer vector it fills.
#[derive(Clone, Copy, Debug)]
struct Query {
    leaf: usize,
    slot: usize,
}

/// The one needed half of a level, `lo ± c·hi`, from `lo`/`hi` into `dst`.
#[inline(always)]
fn half_level(dst: &mut [Fp], lo: &[Fp], hi: &[Fp], c: Option<Fp>, upper: bool) {
    match (c, upper) {
        (Some(c), false) => {
            for ((d, &u), &v) in dst.iter_mut().zip(lo).zip(hi) {
                *d = u + v * c;
            }
        }
        (Some(c), true) => {
            for ((d, &u), &v) in dst.iter_mut().zip(lo).zip(hi) {
                *d = u - v * c;
            }
        }
        (None, false) => {
            for ((d, &u), &v) in dst.iter_mut().zip(lo).zip(hi) {
                *d = u + v;
            }
        }
        (None, true) => {
            for ((d, &u), &v) in dst.iter_mut().zip(lo).zip(hi) {
                *d = u - v;
            }
        }
    }
}

/// The twiddle of group `group`, `None` for the unit twiddle of group 0.
#[inline(always)]
fn twiddle(tw: &[Fp], group: usize) -> Option<Fp> {
    (group != 0).then(|| tw[group])
}

/// The value at leaf `leaf` of the group `values` (the `group`-th of its level, leaves
/// `[base, base + len)`), by the one-sided descent: at each level only the half that holds the
/// leaf is computed, `m/2` multiply-adds, into `scratch` and then in place. `m − 1` multiply-adds
/// in all: the direct evaluation of one output of the sub-transform, without a power table.
fn descend(
    values: &[Fp],
    tw: &[Fp],
    mut group: usize,
    mut base: usize,
    leaf: usize,
    scratch: &mut [Fp],
    ops: &mut Ops,
) -> Fp {
    let mut m = values.len();
    if m == 1 {
        return values[0];
    }
    let mut half = m / 2;
    {
        let (lo, hi) = values.split_at(half);
        let upper = leaf >= base + half;
        ops.muladds += half as u64;
        half_level(&mut scratch[..half], lo, hi, twiddle(tw, group), upper);
        group = 2 * group + usize::from(upper);
        if upper {
            base += half;
        }
    }
    m = half;
    while m > 1 {
        half = m / 2;
        let upper = leaf >= base + half;
        ops.muladds += half as u64;
        let (lo, hi) = scratch[..m].split_at_mut(half);
        let c = twiddle(tw, group);
        // In place: `lo[i] = lo[i] ± c·hi[i]`.
        match (c, upper) {
            (Some(c), false) => {
                for (u, &v) in lo.iter_mut().zip(hi.iter()) {
                    *u += v * c;
                }
            }
            (Some(c), true) => {
                for (u, &v) in lo.iter_mut().zip(hi.iter()) {
                    *u -= v * c;
                }
            }
            (None, false) => {
                for (u, &v) in lo.iter_mut().zip(hi.iter()) {
                    *u += v;
                }
            }
            (None, true) => {
                for (u, &v) in lo.iter_mut().zip(hi.iter()) {
                    *u -= v;
                }
            }
        }
        group = 2 * group + usize::from(upper);
        if upper {
            base += half;
        }
        m = half;
    }
    scratch[0]
}

/// The output-pruned levels of the group `values` (leaves `[base, base + len)`): no query, skip;
/// one query, descend; two or more, a full level and both halves. `queries` is sorted by leaf.
fn pruned_group(
    values: &mut [Fp],
    tw: &[Fp],
    group: usize,
    base: usize,
    queries: &[Query],
    out: &mut [Fp],
    scratch: &mut [Fp],
    ops: &mut Ops,
) {
    match queries {
        [] => return,
        [q] => {
            out[q.slot] = descend(values, tw, group, base, q.leaf, scratch, ops);
            return;
        }
        _ => {}
    }
    assert!(values.len() > 1, "two queries at one leaf");
    let half = values.len() / 2;
    let (lo, hi) = values.split_at_mut(half);
    level(lo, hi, tw, group, ops);
    let split = queries.partition_point(|q| q.leaf < base + half);
    let (ql, qh) = queries.split_at(split);
    pruned_group(lo, tw, 2 * group, base, ql, out, scratch, ops);
    pruned_group(hi, tw, 2 * group + 1, base + half, qh, out, scratch, ops);
}

/// The idea note's accounting: `FIXED_LEVELS` full levels over every group, then one descent
/// per query from its group.
fn fixed_group(
    values: &mut [Fp],
    tw: &[Fp],
    group: usize,
    base: usize,
    depth: u32,
    queries: &[Query],
    out: &mut [Fp],
    scratch: &mut [Fp],
    ops: &mut Ops,
) {
    if depth == FIXED_LEVELS || values.len() == 1 {
        for q in queries {
            out[q.slot] = descend(values, tw, group, base, q.leaf, scratch, ops);
        }
        return;
    }
    let half = values.len() / 2;
    let (lo, hi) = values.split_at_mut(half);
    level(lo, hi, tw, group, ops);
    let split = queries.partition_point(|q| q.leaf < base + half);
    let (ql, qh) = queries.split_at(split);
    fixed_group(lo, tw, 2 * group, base, depth + 1, ql, out, scratch, ops);
    fixed_group(hi, tw, 2 * group + 1, base + half, depth + 1, qh, out, scratch, ops);
}

// ---------------------------------------------------------------------------------------------
// The buffers of the two forms
// ---------------------------------------------------------------------------------------------

/// Coefficient form (IDEA-0002): `a_i = f_i·7^i` for `i < k`, `a_{k+i} = m_i·7^{k+i}` for
/// `i < P`, zero above. `k` multiplications, `P` multiplications, a zero fill.
fn build_coef(a: &mut [Fp], f: &[Fp], masks: &[Fp], pow7: &[Fp]) {
    let k = f.len();
    for ((x, &c), &p) in a[..k].iter_mut().zip(f).zip(pow7) {
        *x = c * p;
    }
    let (masked, rest) = a[k..].split_at_mut(masks.len());
    for ((x, &m), &p) in masked.iter_mut().zip(masks).zip(&pow7[k..]) {
        *x = m * p;
    }
    rest.fill(Fp::ZERO);
}

/// Evaluation form (RFC-0011): from the natural-order coefficients `f` of the row's polynomial,
/// `g = f + (X^k − 1)·m`, then `a_i = g_i·7^i`.
fn build_eval(a: &mut [Fp], f: &[Fp], masks: &[Fp], pow7: &[Fp]) {
    let k = f.len();
    for ((x, &c), &p) in a[..k].iter_mut().zip(f).zip(pow7) {
        *x = c * p;
    }
    for (i, &m) in masks.iter().enumerate() {
        a[i] -= m * pow7[i];
    }
    let (masked, rest) = a[k..].split_at_mut(masks.len());
    for ((x, &m), &p) in masked.iter_mut().zip(masks).zip(&pow7[k..]) {
        *x = m * p;
    }
    rest.fill(Fp::ZERO);
}

/// The queries of a sorted index list, sorted by leaf.
fn queries_of(indices: &[usize], log_n: u32) -> Vec<Query> {
    let mut qs: Vec<Query> = indices
        .iter()
        .enumerate()
        .map(|(slot, &j)| Query {
            leaf: reverse_bits(j, log_n),
            slot,
        })
        .collect();
    qs.sort_by_key(|q| q.leaf);
    qs
}

// ---------------------------------------------------------------------------------------------
// The arms
// ---------------------------------------------------------------------------------------------

/// Everything the arms share for one size.
struct Setup {
    log_n: u32,
    tw: Vec<Fp>,
    pow7: Vec<Fp>,
    plan_n: NttPlan,
    plan_k: NttPlan,
    encoder: RowEncoder,
}

impl Setup {
    fn new(log_k: u32, mask_coeffs: usize) -> Self {
        let k = 1usize << log_k;
        let log_n = log_k + 1;
        Self {
            log_n,
            tw: twiddles(log_n),
            pow7: powers_of_shift(k + mask_coeffs),
            plan_n: NttPlan::new(2 * k).expect("size"),
            plan_k: NttPlan::new(k).expect("size"),
            encoder: RowEncoder::new(k, mask_coeffs).expect("valid"),
        }
    }
}

/// Working memory for the arms: the size-`n` buffer, a size-`k` buffer, the descent scratch.
struct Buffers {
    a: Vec<Fp>,
    kbuf: Vec<Fp>,
    scratch: Vec<Fp>,
    out: Vec<Fp>,
}

impl Buffers {
    fn new(k: usize, tau: usize) -> Self {
        Self {
            a: vec![Fp::ZERO; 2 * k],
            kbuf: vec![Fp::ZERO; k],
            scratch: vec![Fp::ZERO; k],
            out: vec![Fp::ZERO; tau],
        }
    }
}

/// `full_coef`: build, every level, gather.
fn arm_full_coef(s: &Setup, b: &mut Buffers, f: &[Fp], masks: &[Fp], indices: &[usize], ops: &mut Ops) {
    build_coef(&mut b.a, f, masks, &s.pow7);
    full_group(&mut b.a, &s.tw, 0, ops);
    for (o, &j) in b.out.iter_mut().zip(indices) {
        *o = b.a[reverse_bits(j, s.log_n)];
    }
}

/// `pruned_coef`: build, the output-pruned levels.
fn arm_pruned_coef(s: &Setup, b: &mut Buffers, f: &[Fp], masks: &[Fp], queries: &[Query], ops: &mut Ops) {
    build_coef(&mut b.a, f, masks, &s.pow7);
    pruned_group(&mut b.a, &s.tw, 0, 0, queries, &mut b.out, &mut b.scratch, ops);
}

/// `fixed8_coef`: build, eight full levels, one descent per query.
fn arm_fixed_coef(s: &Setup, b: &mut Buffers, f: &[Fp], masks: &[Fp], queries: &[Query], ops: &mut Ops) {
    build_coef(&mut b.a, f, masks, &s.pow7);
    fixed_group(&mut b.a, &s.tw, 0, 0, 0, queries, &mut b.out, &mut b.scratch, ops);
}

/// `coset_ntt_coef`: the production kernel on the natural-order `g`, then a gather. The copy of
/// `g` into the buffer is the caller's, outside the timed region.
fn arm_coset_ntt_coef(s: &Setup, b: &mut Buffers, indices: &[usize]) {
    s.plan_n.coset_ntt(&mut b.a, COSET_SHIFT).expect("length");
    for (o, &j) in b.out.iter_mut().zip(indices) {
        *o = b.a[j];
    }
}

/// `encode_row_eval`: the production encoder, then a gather.
fn arm_encode_row_eval(s: &Setup, b: &mut Buffers, row: &[Fp], indices: &[usize]) {
    let mut rng = SplitMix64(MASK_SEED);
    s.encoder
        .encode_row(row, &mut || rng.fp(), &mut b.a)
        .expect("lengths");
    for (o, &j) in b.out.iter_mut().zip(indices) {
        *o = b.a[j];
    }
}

/// `pruned_eval`: the production inverse NTT of the row, the RFC-0011 mask, the pruned levels.
fn arm_pruned_eval(s: &Setup, b: &mut Buffers, row: &[Fp], masks: &[Fp], queries: &[Query], ops: &mut Ops) {
    b.kbuf.copy_from_slice(row);
    s.plan_k.intt(&mut b.kbuf).expect("length");
    build_eval(&mut b.a, &b.kbuf, masks, &s.pow7);
    pruned_group(&mut b.a, &s.tw, 0, 0, queries, &mut b.out, &mut b.scratch, ops);
}

// ---------------------------------------------------------------------------------------------
// Correctness checks
// ---------------------------------------------------------------------------------------------

fn digest_hex(d: &Digest) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// The record digest of a symbol vector: `column-merge` over the canonical little-endian bytes.
fn record_digest(symbols: &[Fp]) -> String {
    let bytes: Vec<u8> = symbols
        .iter()
        .flat_map(|x| x.to_canonical_u64().to_le_bytes())
        .collect();
    digest_hex(&hash(Domain::ColumnMerge, &bytes))
}

fn assert_symbols(name: &str, got: &[Fp], reference: &[Fp], indices: &[usize]) {
    for ((g, &j), r) in got.iter().zip(indices).zip(reference) {
        assert!(
            g == r,
            "{name}: symbol at column {j} differs from the reference ({:#x} against {:#x})",
            g.to_canonical_u64(),
            r.to_canonical_u64()
        );
    }
}

/// The hand case of `crates/tessera-pcs/src/lib.rs`: `k = 8`, `P = 2`, row `1..=8`, masks 11, 12;
/// the pruned pass in both forms against the production references at every pre-registered
/// subset of positions.
fn check_hand_case() {
    let fp = Fp::from_u64_reduced;
    let s = Setup::new(3, 2);
    let row: Vec<Fp> = (1..=8).map(fp).collect();
    let masks = [fp(11), fp(12)];
    let subsets: [&[usize]; 5] = [
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        &[0],
        &[3, 11],
        &[1, 5, 9, 13],
        &[2, 10],
    ];

    // Evaluation form: encode_row with the masks 11, 12 (the doc example), the pruned pass on
    // intt(row) with the same masks.
    let mut reference = vec![Fp::ZERO; 16];
    let mut it = masks.into_iter();
    s.encoder
        .encode_row(&row, &mut || it.next().unwrap_or_default(), &mut reference)
        .expect("lengths");
    {
        // The doc's own assertion, so that the reference is the documented codeword.
        let mut g = reference.clone();
        s.plan_n.coset_intt(&mut g, COSET_SHIFT).expect("length");
        assert_eq!(&g[8..], &[fp(11), fp(12), fp(0), fp(0), fp(0), fp(0), fp(0), fp(0)]);
    }
    for indices in subsets {
        let mut b = Buffers::new(8, indices.len());
        let queries = queries_of(indices, s.log_n);
        let mut ops = Ops::default();
        b.kbuf.copy_from_slice(&row);
        s.plan_k.intt(&mut b.kbuf).expect("length");
        build_eval(&mut b.a, &b.kbuf, &masks, &s.pow7);
        pruned_group(&mut b.a, &s.tw, 0, 0, &queries, &mut b.out, &mut b.scratch, &mut ops);
        let gathered: Vec<Fp> = indices.iter().map(|&j| reference[j]).collect();
        assert_symbols("hand case, evaluation form", &b.out, &gathered, indices);
        // The fixed-depth variant at depth 8 exceeds log2 16, so it descends from leaves: a
        // degenerate but valid run.
        let mut ops = Ops::default();
        build_eval(&mut b.a, &b.kbuf, &masks, &s.pow7);
        fixed_group(&mut b.a, &s.tw, 0, 0, 0, &queries, &mut b.out, &mut b.scratch, &mut ops);
        assert_symbols("hand case, evaluation form, fixed depth", &b.out, &gathered, indices);
    }

    // Coefficient form: g = (1, …, 8, 11, 12, 0, …), coset_ntt(g, 7) as the reference.
    let mut g = vec![Fp::ZERO; 16];
    g[..8].copy_from_slice(&row);
    g[8] = masks[0];
    g[9] = masks[1];
    let mut reference = g.clone();
    s.plan_n.coset_ntt(&mut reference, COSET_SHIFT).expect("length");
    for indices in subsets {
        let mut b = Buffers::new(8, indices.len());
        let queries = queries_of(indices, s.log_n);
        let mut ops = Ops::default();
        arm_pruned_coef(&s, &mut b, &row, &masks, &queries, &mut ops);
        let gathered: Vec<Fp> = indices.iter().map(|&j| reference[j]).collect();
        assert_symbols("hand case, coefficient form", &b.out, &gathered, indices);
        let mut ops = Ops::default();
        arm_full_coef(&s, &mut b, &row, &masks, indices, &mut ops);
        assert_symbols("hand case, coefficient form, full kernel", &b.out, &gathered, indices);
    }
    println!(
        "check hand case: k = 8, P = 2 (row 1..=8, masks 11, 12): pruned, fixed-depth and full kernels equal encode_row and coset_ntt at all 16 positions and at {{0}}, {{3, 11}}, {{1, 5, 9, 13}}, {{2, 10}}"
    );
}

// ---------------------------------------------------------------------------------------------
// Query sets
// ---------------------------------------------------------------------------------------------

/// `τ` distinct indices in `[0, n)`, repeats rejected, sorted.
fn uniform_queries(rng: &mut SplitMix64, n: usize, tau: usize) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::with_capacity(tau);
    while out.len() < tau {
        let j = (rng.next_u64() as usize) & (n - 1);
        if !out.contains(&j) {
            out.push(j);
        }
    }
    out.sort_unstable();
    out
}

/// `r + 256·t` for `t < τ`: one residue class modulo 256, every query in one group at depth 8.
fn clustered_queries(rng: &mut SplitMix64, tau: usize) -> Vec<usize> {
    let r = (rng.next_u64() as usize) & 255;
    (0..tau).map(|t| r + 256 * t).collect()
}

/// `⌊τ/2⌋` pairs `(j, j + n/2)` with distinct `j < n/2`, one unpaired draw when `τ` is odd.
fn paired_queries(rng: &mut SplitMix64, n: usize, tau: usize) -> Vec<usize> {
    let mut lows: Vec<usize> = Vec::with_capacity(tau);
    while lows.len() < tau.div_ceil(2) {
        let j = (rng.next_u64() as usize) & (n / 2 - 1);
        if !lows.contains(&j) {
            lows.push(j);
        }
    }
    let mut out: Vec<usize> = Vec::with_capacity(tau);
    for (t, &j) in lows.iter().enumerate() {
        out.push(j);
        if out.len() < tau && (2 * t + 1) < tau {
            out.push(j + n / 2);
        }
    }
    out.truncate(tau);
    out.sort_unstable();
    out
}

// ---------------------------------------------------------------------------------------------

/// One reported arm.
struct Row {
    arm: &'static str,
    set: &'static str,
    tau: usize,
    stats: Stats,
    ops: Ops,
    digest: String,
}

fn report(rows: &mut Vec<Row>, arm: &'static str, set: &'static str, tau: usize, k: usize, stats: Stats, ops: Ops, digest: String) {
    let flag = if stats.cv > CV_LIMIT {
        "  FLAG: CV > 3% after a rerun"
    } else if stats.repeated {
        "  (rerun: first CV > 3%)"
    } else {
        ""
    };
    let ms = |t: f64| t * 1e3;
    println!(
        "{arm:<16} {set:<9} τ={tau:<4} {:>9.3} ms/row  {:>8.3} ns/element  min {:>9.3}  max {:>9.3}  CV {:>5.2}%   bf/el {:>6.3}  ma/el {:>6.3}  digest {}{flag}",
        ms(stats.median),
        stats.median * 1e9 / k as f64,
        ms(stats.min),
        ms(stats.max),
        stats.cv * 100.0,
        ops.butterflies as f64 / k as f64,
        ops.muladds as f64 / k as f64,
        &digest[..16],
    );
    rows.push(Row {
        arm,
        set,
        tau,
        stats,
        ops,
        digest,
    });
}

fn find_row<'a>(rows: &'a [Row], arm: &str, set: &str, tau: usize) -> &'a Row {
    rows.iter()
        .find(|r| r.arm == arm && r.set == set && r.tau == tau)
        .expect("row")
}

fn main() {
    println!(
        "EXP-0007 harness: k = 2^{LOG_K}, P = {MASK_COEFFS}, τ ∈ {TAUS:?}; protocol {WARMUPS} warm-ups, {ITERATIONS} iterations, CV limit {:.0}%",
        CV_LIMIT * 100.0
    );
    println!("\n== correctness checks ==");
    check_hand_case();

    let k = 1usize << LOG_K;
    let n = 2 * k;
    let t0 = Instant::now();
    let s = Setup::new(LOG_K, MASK_COEFFS);
    println!("setup: twiddles, 7^i table, NttPlan 2^{}, NttPlan 2^{LOG_K}, RowEncoder in {:.2} s", LOG_K + 1, t0.elapsed().as_secs_f64());

    let row = SplitMix64(ROW_SEED).fps(k);
    let masks = SplitMix64(MASK_SEED).fps(MASK_COEFFS);

    // References at full size.
    // Coefficient form: g = (f, m, 0), f the row read as coefficients.
    let mut g_coef = vec![Fp::ZERO; n];
    g_coef[..k].copy_from_slice(&row);
    g_coef[k..k + MASK_COEFFS].copy_from_slice(&masks);
    let mut ref_coef = g_coef.clone();
    s.plan_n.coset_ntt(&mut ref_coef, COSET_SHIFT).expect("length");
    // Evaluation form: encode_row with the same mask stream.
    let mut ref_eval = vec![Fp::ZERO; n];
    {
        let mut rng = SplitMix64(MASK_SEED);
        s.encoder
            .encode_row(&row, &mut || rng.fp(), &mut ref_eval)
            .expect("lengths");
    }
    // The harness kernel against the production one on every leaf.
    {
        let mut b = Buffers::new(k, 1);
        let mut ops = Ops::default();
        build_coef(&mut b.a, &row, &masks, &s.pow7);
        full_group(&mut b.a, &s.tw, 0, &mut ops);
        for (j, r) in ref_coef.iter().enumerate() {
            assert!(
                b.a[reverse_bits(j, s.log_n)] == *r,
                "full kernel differs from coset_ntt at column {j}"
            );
        }
        println!(
            "check kernel:    full_coef equals coset_ntt(g, 7) on all 2^{} leaves ({} butterflies = {:.1} per committed element); reference digests coef {} eval {}",
            s.log_n,
            ops.butterflies,
            ops.butterflies as f64 / k as f64,
            &record_digest(&ref_coef)[..16],
            &record_digest(&ref_eval)[..16]
        );
    }

    // Context: the coefficient buffer alone (k + P multiplications and a zero fill), which every
    // coefficient-form arm but `coset_ntt_coef` pays inside its timed region.
    {
        let mut b = Buffers::new(k, 1);
        let stats = measure(|| {
            let start = Instant::now();
            build_coef(black_box(&mut b.a), black_box(&row), &masks, &s.pow7);
            start.elapsed()
        });
        println!(
            "\ncontext: build_coef alone {:.3} ms/row ({:.3} ns/element; min {:.3}, max {:.3}, CV {:.2}%)",
            stats.median * 1e3,
            stats.median * 1e9 / k as f64,
            stats.min * 1e3,
            stats.max * 1e3,
            stats.cv * 100.0
        );
    }

    let mut rows: Vec<Row> = Vec::new();
    for &tau in &TAUS {
        let mut qrng = SplitMix64(QUERY_SEED);
        let sets: [(&'static str, Vec<usize>); 3] = [
            ("uniform", uniform_queries(&mut qrng, n, tau)),
            ("clustered", clustered_queries(&mut qrng, tau)),
            ("paired", paired_queries(&mut qrng, n, tau)),
        ];
        for (set, indices) in &sets {
            let set: &'static str = set;
            assert_eq!(indices.len(), tau);
            let distinct: std::collections::BTreeSet<usize> = indices.iter().copied().collect();
            assert_eq!(distinct.len(), tau, "{set}: repeated index");
            let queries = queries_of(indices, s.log_n);
            let gathered_coef: Vec<Fp> = indices.iter().map(|&j| ref_coef[j]).collect();
            let gathered_eval: Vec<Fp> = indices.iter().map(|&j| ref_eval[j]).collect();
            let groups_at_8: std::collections::BTreeSet<usize> =
                queries.iter().map(|q| q.leaf >> (s.log_n - FIXED_LEVELS)).collect();
            println!(
                "\n== τ = {tau}, {set} queries: first {:?}, {} distinct groups at depth {FIXED_LEVELS} ==",
                &indices[..4.min(tau)],
                groups_at_8.len()
            );
            let mut b = Buffers::new(k, tau);

            // full_coef (baseline, coefficient form)
            let mut ops = Ops::default();
            arm_full_coef(&s, &mut b, &row, &masks, indices, &mut ops);
            assert_symbols("full_coef", &b.out, &gathered_coef, indices);
            let stats = measure(|| {
                let mut o = Ops::default();
                let start = Instant::now();
                arm_full_coef(&s, black_box(&mut b), black_box(&row), &masks, indices, &mut o);
                start.elapsed()
            });
            assert_symbols("full_coef after timing", &b.out, &gathered_coef, indices);
            report(&mut rows, "full_coef", set, tau, k, stats, ops, record_digest(&b.out));

            // pruned_coef (candidate, coefficient form)
            let mut ops = Ops::default();
            arm_pruned_coef(&s, &mut b, &row, &masks, &queries, &mut ops);
            assert_symbols("pruned_coef", &b.out, &gathered_coef, indices);
            let stats = measure(|| {
                let mut o = Ops::default();
                let start = Instant::now();
                arm_pruned_coef(&s, black_box(&mut b), black_box(&row), &masks, &queries, &mut o);
                start.elapsed()
            });
            assert_symbols("pruned_coef after timing", &b.out, &gathered_coef, indices);
            report(&mut rows, "pruned_coef", set, tau, k, stats, ops, record_digest(&b.out));

            // fixed8_coef (the note's accounting)
            let mut ops = Ops::default();
            arm_fixed_coef(&s, &mut b, &row, &masks, &queries, &mut ops);
            assert_symbols("fixed8_coef", &b.out, &gathered_coef, indices);
            let stats = measure(|| {
                let mut o = Ops::default();
                let start = Instant::now();
                arm_fixed_coef(&s, black_box(&mut b), black_box(&row), &masks, &queries, &mut o);
                start.elapsed()
            });
            assert_symbols("fixed8_coef after timing", &b.out, &gathered_coef, indices);
            report(&mut rows, "fixed8_coef", set, tau, k, stats, ops, record_digest(&b.out));

            // coset_ntt_coef (context: today's production kernel)
            let stats = measure(|| {
                b.a.copy_from_slice(&g_coef);
                let start = Instant::now();
                arm_coset_ntt_coef(&s, black_box(&mut b), indices);
                start.elapsed()
            });
            assert_symbols("coset_ntt_coef", &b.out, &gathered_coef, indices);
            report(&mut rows, "coset_ntt_coef", set, tau, k, stats, Ops::default(), record_digest(&b.out));

            // encode_row_eval (baseline, evaluation form)
            let stats = measure(|| {
                let start = Instant::now();
                arm_encode_row_eval(&s, black_box(&mut b), black_box(&row), indices);
                start.elapsed()
            });
            assert_symbols("encode_row_eval", &b.out, &gathered_eval, indices);
            report(&mut rows, "encode_row_eval", set, tau, k, stats, Ops::default(), record_digest(&b.out));

            // pruned_eval (candidate, evaluation form)
            let mut ops = Ops::default();
            arm_pruned_eval(&s, &mut b, &row, &masks, &queries, &mut ops);
            assert_symbols("pruned_eval", &b.out, &gathered_eval, indices);
            let stats = measure(|| {
                let mut o = Ops::default();
                let start = Instant::now();
                arm_pruned_eval(&s, black_box(&mut b), black_box(&row), &masks, &queries, &mut o);
                start.elapsed()
            });
            assert_symbols("pruned_eval after timing", &b.out, &gathered_eval, indices);
            report(&mut rows, "pruned_eval", set, tau, k, stats, ops, record_digest(&b.out));
        }
    }

    println!("\n== ratios (medians) ==");
    println!("{:<9} {:<5} {:>10} {:>10} {:>12} {:>12} {:>12}", "set", "τ", "R_coef", "R_eval", "R_fixed8", "R_today", "R_eval_coef");
    let find = |arm: &str, set: &str, tau: usize| find_row(&rows, arm, set, tau);
    for &tau in &TAUS {
        for set in ["uniform", "clustered", "paired"] {
            let full = find("full_coef", set, tau).stats.median;
            let pruned = find("pruned_coef", set, tau).stats.median;
            let fixed = find("fixed8_coef", set, tau).stats.median;
            let today = find("coset_ntt_coef", set, tau).stats.median;
            let enc = find("encode_row_eval", set, tau).stats.median;
            let pe = find("pruned_eval", set, tau).stats.median;
            println!(
                "{set:<9} {tau:<5} {:>10.3} {:>10.3} {:>12.3} {:>12.3} {:>12.3}",
                full / pruned,
                enc / pe,
                full / fixed,
                today / pruned,
                enc / pruned
            );
        }
    }
    println!(
        "R_coef = full_coef / pruned_coef; R_eval = encode_row_eval / pruned_eval; R_fixed8 = full_coef / fixed8_coef; R_today = coset_ntt_coef / pruned_coef; R_eval_coef = encode_row_eval / pruned_coef"
    );
    println!("\nrecord (per arm: median ms, butterflies and multiply-adds per committed element, column-merge digest of the τ symbols)");
    for r in &rows {
        println!(
            "  {:<16} {:<9} τ={:<4} {:>9.3} ms  bf/el {:>6.3}  ma/el {:>6.3}  {}",
            r.arm,
            r.set,
            r.tau,
            r.stats.median * 1e3,
            r.ops.butterflies as f64 / k as f64,
            r.ops.muladds as f64 / k as f64,
            r.digest
        );
    }
}
