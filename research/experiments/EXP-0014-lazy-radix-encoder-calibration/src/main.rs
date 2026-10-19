// SPDX-License-Identifier: Apache-2.0 OR MIT
//! EXP-0014: arithmetic-preserving radix calibration on synthetic rows only.
#![forbid(unsafe_code)]

use serde_json::{Value, json};
use std::hint::{black_box, select_unpredictable};
use std::time::{Duration, Instant};
use tessera_field::{Field, Goldilocks as Fp, PrimeField};
use tessera_hash::{Domain, hash};
use tessera_pcs::{NttPlan, RowEncoder};

const P: u64 = 0xffff_ffff_0000_0001;
const EPS: u64 = 0xffff_ffff;
const LEAF: usize = 1 << 12;

#[inline(always)]
fn pick(b: bool, x: u64) -> u64 {
    select_unpredictable(b, x, 0)
}
#[inline(always)]
fn canonical(x: u64) -> u64 {
    let (r, b) = x.overflowing_sub(P);
    r.wrapping_add(pick(b, P))
}
#[inline(always)]
fn add<const LAZY: bool>(a: u64, b: u64) -> u64 {
    let (v, c) = a.overflowing_add(b);
    let (v, c2) = v.overflowing_add(pick(c, EPS));
    if LAZY {
        v.wrapping_add(pick(c2, EPS))
    } else {
        canonical(v)
    }
}
#[inline(always)]
fn sub<const LAZY: bool>(a: u64, b: u64) -> u64 {
    let (v, c) = a.overflowing_sub(b);
    let (v, c2) = v.overflowing_sub(pick(c, EPS));
    if LAZY {
        v.wrapping_sub(pick(c2, EPS))
    } else {
        v
    }
}
#[inline(always)]
fn reduce<const LAZY: bool>(x: u128) -> u64 {
    let lo = x as u64;
    let hi = (x >> 64) as u64;
    let (v, b) = lo.overflowing_sub(hi >> 32);
    let v = v.wrapping_sub(pick(b, EPS));
    let (v, c) = v.overflowing_add((hi & EPS) * EPS);
    let v = v.wrapping_add(pick(c, EPS));
    if LAZY { v } else { canonical(v) }
}
#[inline(always)]
fn mul<const LAZY: bool>(a: u64, b: u64) -> u64 {
    reduce::<LAZY>(a as u128 * b as u128)
}
#[inline(always)]
fn cheap<const LAZY: bool>(x: u64, exp: u32) -> u64 {
    let exp = exp % 192;
    let e = exp % 96;
    let product = if e < 64 {
        (x as u128) << e
    } else {
        ((x as u128) << (e - 32)) - ((x as u128) << (e - 64))
    };
    let y = reduce::<LAZY>(product);
    if exp >= 96 { sub::<LAZY>(0, y) } else { y }
}

struct Rng(u64);
impl Rng {
    fn word(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn fp(&mut self) -> Fp {
        Fp::from_u64_reduced(self.word())
    }
    fn values(&mut self, n: usize) -> Vec<Fp> {
        (0..n).map(|_| self.fp()).collect()
    }
}
fn rev(i: usize, bits: u32) -> usize {
    i.reverse_bits()
        .checked_shr(usize::BITS - bits)
        .unwrap_or(0)
}
fn twiddles(n: usize, inverse: bool) -> Vec<u64> {
    if n == 1 {
        return Vec::new();
    }
    let root = Fp::two_adic_generator(n.trailing_zeros());
    let root = if inverse {
        root.inverse().unwrap()
    } else {
        root
    };
    let mut powers = vec![Fp::ONE; n / 2];
    for i in 1..n / 2 {
        powers[i] = powers[i - 1] * root;
    }
    (0..n / 2)
        .map(|i| powers[rev(i, n.trailing_zeros() - 1)].to_canonical_u64())
        .collect()
}

struct Roots {
    tw: Vec<u64>,
    powers: [Vec<u64>; 3],
}
impl Roots {
    fn new(n: usize, inverse: bool) -> Self {
        let tw = twiddles(n, inverse);
        let powers = std::array::from_fn(|i| {
            let r = 1usize << (i + 2);
            let mut out = Vec::with_capacity((n / r) * (r - 1));
            for g in 0..n / r {
                let c = tw[g * r / 2];
                let mut power = c;
                for _ in 1..r {
                    out.push(power);
                    power = mul::<false>(power, c);
                }
            }
            out
        });
        Self { tw, powers }
    }
}

// Generated fixed-size butterflies below use public constant exponents, not table lookups.
mod small;
use small::{lazy_radix16_anchor, radix2_anchor, small2, small4, small8, small16};

fn block<const R: usize, const LAZY: bool, const GS: bool>(
    values: &mut [u64],
    tw: &Roots,
    g: usize,
    inverse: bool,
) {
    let m = values.len() / R;
    let powers = if R == 2 {
        &tw.tw[g..g + 1]
    } else {
        &tw.powers[R.trailing_zeros() as usize - 2][g * (R - 1)..(g + 1) * (R - 1)]
    };
    for j in 0..m {
        let mut x = [0u64; 16];
        for q in 0..R {
            x[q] = values[q * m + j];
        }
        if !GS && g != 0 {
            for q in 1..R {
                x[q] = mul::<LAZY>(x[q], powers[q - 1]);
            }
        }
        match R {
            2 => small2::<LAZY, GS>(&mut x, inverse),
            4 => small4::<LAZY, GS>(&mut x, inverse),
            8 => small8::<LAZY, GS>(&mut x, inverse),
            16 => small16::<LAZY, GS>(&mut x, inverse),
            _ => unreachable!(),
        }
        if GS && g != 0 {
            for q in 1..R {
                x[q] = mul::<LAZY>(x[q], powers[q - 1]);
            }
        }
        for q in 0..R {
            values[q * m + j] = x[q];
        }
    }
}

fn stage<const LAZY: bool, const GS: bool>(
    v: &mut [u64],
    tw: &Roots,
    g: usize,
    bits: u32,
    inverse: bool,
) {
    match bits {
        1 => block::<2, LAZY, GS>(v, tw, g, inverse),
        2 => block::<4, LAZY, GS>(v, tw, g, inverse),
        3 => block::<8, LAZY, GS>(v, tw, g, inverse),
        4 => block::<16, LAZY, GS>(v, tw, g, inverse),
        _ => unreachable!(),
    }
}

fn group<const BITS: u32, const LAZY: bool, const GS: bool>(
    v: &mut [u64],
    tw: &Roots,
    g: usize,
    inverse: bool,
) {
    let log = v.len().trailing_zeros();
    if log == 0 {
        return;
    }
    if v.len() > LEAF {
        let b = BITS.min(log);
        let radix = 1 << b;
        let chunk = v.len() / radix;
        if !GS {
            stage::<LAZY, GS>(v, tw, g, b, inverse);
        }
        for (i, c) in v.chunks_exact_mut(chunk).enumerate() {
            group::<BITS, LAZY, GS>(c, tw, g * radix + i, inverse);
        }
        if GS {
            stage::<LAZY, GS>(v, tw, g, b, inverse);
        }
        return;
    }
    if GS {
        let tail = if log.is_multiple_of(BITS) {
            BITS
        } else {
            log % BITS
        };
        let mut size = 1usize << tail;
        loop {
            let count = v.len() / size;
            let b = if size == 1usize << tail { tail } else { BITS };
            for (i, c) in v.chunks_exact_mut(size).enumerate() {
                stage::<LAZY, GS>(c, tw, g * count + i, b, inverse);
            }
            if size == v.len() {
                break;
            }
            size <<= BITS;
        }
    } else {
        let mut size = v.len();
        while size > 1 {
            let b = BITS.min(size.trailing_zeros());
            let count = v.len() / size;
            for (i, c) in v.chunks_exact_mut(size).enumerate() {
                stage::<LAZY, GS>(c, tw, g * count + i, b, inverse);
            }
            size >>= b;
        }
    }
}

mod reverse;
use reverse::reverse;

struct Plan {
    forward: Roots,
    inverse: Roots,
    spread: Vec<u64>,
    inv: u64,
    shift_k: u64,
    coset: Vec<u64>,
}
impl Plan {
    fn new(n: usize, k: usize) -> Self {
        let inv = Fp::from_u64_reduced(n as u64)
            .inverse()
            .unwrap()
            .to_canonical_u64();
        let ik = Fp::from_u64_reduced(k as u64).inverse().unwrap();
        let shift = Fp::from_u64_reduced(7);
        let mut powers = vec![ik; k];
        for i in 1..k {
            powers[i] = powers[i - 1] * shift;
        }
        let spread = (0..k)
            .map(|i| powers[rev(i, k.trailing_zeros())].to_canonical_u64())
            .collect();
        let mut coset = vec![1u64; n];
        for i in 1..n {
            coset[i] = mul::<false>(coset[i - 1], 7);
        }
        Self {
            coset,
            forward: Roots::new(n, false),
            inverse: Roots::new(k, true),
            spread,
            inv,
            shift_k: shift.pow(k as u64).to_canonical_u64(),
        }
    }
    fn transform<const BITS: u32, const LAZY: bool>(&self, v: &mut [u64], kind: &str) {
        match kind {
            "intt" => {
                reverse(v, Some(self.inv));
                group::<BITS, LAZY, true>(v, &self.inverse, 0, true);
            }
            "ntt" | "coset" => {
                if kind == "coset" {
                    for (x, power) in v.iter_mut().zip(&self.coset) {
                        *x = mul::<LAZY>(*x, *power);
                    }
                }
                group::<BITS, LAZY, false>(v, &self.forward, 0, false);
                reverse(v, None);
            }
            _ => unreachable!(),
        }
        for x in v {
            *x = canonical(*x);
        }
    }
    fn encode<const BITS: u32, const LAZY: bool>(
        &self,
        row: &[u64],
        mask: &[u64],
        out: &mut [u64],
    ) {
        let k = row.len();
        out[..k].copy_from_slice(row);
        group::<BITS, LAZY, false>(&mut out[..k], &self.inverse, 0, true);
        spread::<LAZY>(out, &self.spread);
        let mut power = 1;
        for (i, &m) in mask.iter().enumerate() {
            let v = mul::<LAZY>(m, power);
            let j = 2 * rev(i, k.trailing_zeros());
            out[j] = sub::<LAZY>(out[j], v);
            out[j + 1] = mul::<LAZY>(v, self.shift_k);
            power = mul::<false>(power, 7);
        }
        group::<BITS, LAZY, true>(out, &self.forward, 0, false);
        for x in out {
            *x = canonical(*x);
        }
    }
}

fn run_transform(p: &Plan, a: &mut [u64], arm: usize, kind: &str) {
    match arm {
        0 => p.transform::<2, false>(a, kind),
        1 => p.transform::<3, false>(a, kind),
        2 => p.transform::<4, false>(a, kind),
        3 => p.transform::<2, true>(a, kind),
        4 => p.transform::<3, true>(a, kind),
        5 => p.transform::<4, true>(a, kind),
        _ => unreachable!(),
    }
}
fn run_encode(p: &Plan, row: &[u64], mask: &[u64], out: &mut [u64], arm: usize) {
    match arm {
        0 => p.encode::<2, false>(row, mask, out),
        1 => p.encode::<3, false>(row, mask, out),
        2 => p.encode::<4, false>(row, mask, out),
        3 => p.encode::<2, true>(row, mask, out),
        4 => p.encode::<3, true>(row, mask, out),
        5 => p.encode::<4, true>(row, mask, out),
        _ => unreachable!(),
    }
}

// Preserve the production encoder spread traversal on synthetic words.
fn spread<const LAZY: bool>(codeword: &mut [u64], scale: &[u64]) {
    let stride = codeword.len() / scale.len();
    let mut m = scale.len();
    while m >= 2 {
        let (from, to) = codeword[..stride * m].split_at_mut(stride * m / 2);
        for ((run, &y), &s) in to
            .chunks_exact_mut(stride)
            .zip(&from[m / 2..m])
            .zip(&scale[m / 2..m])
        {
            run[0] = mul::<LAZY>(y, s);
            run[1..].fill(0);
        }
        m /= 2;
    }
    codeword[0] = mul::<LAZY>(codeword[0], scale[0]);
    codeword[1..stride].fill(0);
}

fn words(a: &[Fp]) -> Vec<u64> {
    a.iter().map(Fp::to_canonical_u64).collect()
}
fn parse_words(v: &Value) -> Vec<u64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| u64::from_str_radix(v.as_str().unwrap().trim_start_matches("0x"), 16).unwrap())
        .collect()
}
fn check_arithmetic() {
    let edge = [0, 1, P - 1, P, (1u64 << 32) - 1, 1u64 << 32, u64::MAX];
    let check = |a: u64, b: u64| {
        assert_eq!(
            canonical(add::<true>(a, b)),
            ((a as u128 + b as u128) % P as u128) as u64
        );
        assert_eq!(
            canonical(sub::<true>(a, b)),
            ((a as u128 + 2 * P as u128 - b as u128) % P as u128) as u64
        );
        assert_eq!(
            canonical(mul::<true>(a, b)),
            ((a as u128 * b as u128) % P as u128) as u64
        );
        let x = Fp::from_u64_reduced(a);
        for e in (0..192).step_by(12) {
            assert_eq!(
                canonical(cheap::<true>(a, e)),
                (x * Fp::from_u64_reduced(2).pow(e as u64)).to_canonical_u64()
            );
        }
    };
    for &a in &edge {
        for &b in &edge {
            check(a, b);
        }
    }
    let mut rng = Rng(0x7e55_e7a0_0047_0414);
    for _ in 0..4096 {
        check(rng.word(), rng.word());
    }
    // Both second corrections are reached: MAX+MAX and 0-MAX.
    assert_eq!(canonical(add::<true>(u64::MAX, u64::MAX)), 2 * EPS - 2);
    assert_eq!(canonical(sub::<true>(0, u64::MAX)), P - EPS + 1);
    assert_eq!(
        Fp::two_adic_generator(3).to_canonical_u64(),
        canonical(cheap::<true>(1, 120))
    );
    assert_eq!(
        Fp::two_adic_generator(4).to_canonical_u64(),
        canonical(cheap::<true>(1, 156))
    );
    let c = rng.fp();
    let mut powers = [1; 15];
    let mut power = c;
    for v in &mut powers {
        *v = power.to_canonical_u64();
        power *= c;
    }
    let mut tw = [1; 15];
    let root = Fp::two_adic_generator(4);
    let mut at = 0;
    for level in 0..4 {
        let half = 16 >> (level + 1);
        for g in 0..1 << level {
            tw[at] = (c.pow(half as u64) * root.pow(rev(g, 3) as u64)).to_canonical_u64();
            at += 1;
        }
    }
    let mut a = std::array::from_fn(|_| rng.fp().to_canonical_u64());
    let mut b = a;
    radix2_anchor(&mut a, &tw);
    lazy_radix16_anchor(&mut b, &powers);
    assert_eq!(a, b.map(canonical));
}
fn check_transform(n: usize, input: &[u64], expected: &[u64], kind: &str) {
    let p = Plan::new(n, n);
    for arm in 0..6 {
        let mut a = input.to_vec();
        run_transform(&p, &mut a, arm, kind);
        assert_eq!(a, expected, "transform {kind} n{n} arm{arm}");
        assert!(a.iter().all(|x| *x < P));
    }
}
fn check_row(row: &[u64], mask: &[u64], expected: Option<&[u64]>) {
    let k = row.len();
    let p = Plan::new(2 * k, k);
    let base = RowEncoder::new(k, 1, mask.len()).unwrap();
    let input: Vec<Fp> = row.iter().map(|&x| Fp::from_u64_reduced(x)).collect();
    let mut out = vec![Fp::ZERO; 2 * k];
    let mut i = 0;
    base.encode_row(
        &input,
        &mut || {
            let x = Fp::from_u64_reduced(mask[i]);
            i += 1;
            x
        },
        &mut out,
    )
    .unwrap();
    assert_eq!(i, mask.len());
    let expected_base = words(&out);
    if let Some(expected) = expected {
        assert_eq!(expected_base, expected);
    }
    for arm in 0..6 {
        let mut out = vec![u64::MAX; 2 * k];
        run_encode(&p, row, mask, &mut out, arm);
        assert_eq!(out, expected_base, "encode k{k} P{} arm{arm}", mask.len());
    }
}
fn check_all() {
    check_arithmetic();
    let vectors: Value = serde_json::from_str(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../spec/vectors/encoding/encoding.json"
        ))
        .unwrap(),
    )
    .unwrap();
    for case in vectors["ntt"].as_array().unwrap() {
        let a = parse_words(&case["coefficients"]);
        let b = parse_words(&case["values"]);
        let c = parse_words(&case["coset_values"]);
        check_transform(a.len(), &a, &b, "ntt");
        check_transform(a.len(), &b, &a, "intt");
        check_transform(a.len(), &a, &c, "coset");
    }
    for case in vectors["encoding"].as_array().unwrap() {
        check_row(
            &parse_words(&case["row"]),
            &parse_words(&case["mask"]),
            Some(&parse_words(&case["codeword"])),
        );
    }
    let mut rng = Rng(0x7e55_e7a0_0047_0401);
    for log in 0..=12 {
        let n = 1 << log;
        let input = rng.values(n);
        let base = NttPlan::new(n).unwrap();
        for kind in ["ntt", "intt", "coset"] {
            let mut out = input.clone();
            production_transform(&base, &mut out, kind);
            check_transform(n, &words(&input), &words(&out), kind);
        }
        for masks in [0, 1, n / 2, n] {
            check_row(&words(&input), &words(&rng.values(masks)), None);
        }
    }
    println!(
        "correctness: every encoding vector, small transform/row sweep, raw-word boundaries, double corrections and equivalent anchors pass"
    );
}
fn production_transform(p: &NttPlan, a: &mut [Fp], kind: &str) {
    match kind {
        "ntt" => p.ntt(a).unwrap(),
        "intt" => p.intt(a).unwrap(),
        "coset" => p.coset_ntt(a, Fp::from_u64_reduced(7)).unwrap(),
        _ => unreachable!(),
    }
}

fn stats(samples: &[f64]) -> (f64, f64, f64, f64) {
    let mut s = samples.to_vec();
    s.sort_by(f64::total_cmp);
    let mean = s.iter().sum::<f64>() / s.len() as f64;
    let cv =
        (s.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (s.len() - 1) as f64).sqrt() / mean;
    ((s[4] + s[5]) / 2.0, s[0], s[9], cv)
}
fn timed(name: &str, elements: usize, mut once: impl FnMut()) -> Value {
    let mut attempts = Vec::new();
    for attempt in 0..2 {
        let mut samples = Vec::new();
        let mut batches = Vec::new();
        for i in 0..13 {
            let start = Instant::now();
            let mut reps = 0;
            while start.elapsed() < Duration::from_millis(20) {
                once();
                reps += 1;
            }
            let elapsed = start.elapsed().as_secs_f64();
            batches.push(json!({"seconds":elapsed,"executions":reps,"warmup":i<3}));
            if i >= 3 {
                samples.push(elapsed / reps as f64);
            }
        }
        let (median, min, max, cv) = stats(&samples);
        attempts.push(json!({"samples_seconds_per_execution":samples,"batches":batches,"median_seconds":median,"min_seconds":min,"max_seconds":max,"cv":cv}));
        if cv <= 0.03 || attempt == 1 {
            break;
        }
    }
    let retained = attempts.last().unwrap();
    println!(
        "{name}: {:.3} ns/element CV {:.4}%",
        retained["median_seconds"].as_f64().unwrap() * 1e9 / elements as f64,
        retained["cv"].as_f64().unwrap() * 100.0
    );
    json!({"name":name,"elements":elements,"attempts":attempts,"retained_attempt":attempts.len()-1})
}
const ARMS: [&str; 7] = [
    "production",
    "canonical4",
    "canonical8",
    "canonical16",
    "lazy4",
    "lazy8",
    "lazy16",
];

fn digest(a: &[u64]) -> String {
    let bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
    hash(Domain::ColumnMerge, &bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn big_cases(timing: bool) -> Vec<Value> {
    let mut rows = Vec::new();
    let mut rng = Rng(0x7e55_e7a0_0047_0401);
    for log in [20, 22, 23] {
        let n = 1usize << log;
        let input = rng.values(n);
        let raw = words(&input);
        let base = NttPlan::new(n).unwrap();
        let p = Plan::new(n, n);
        for kind in ["ntt", "intt", "coset"] {
            let mut expected = input.clone();
            production_transform(&base, &mut expected, kind);
            let expected = words(&expected);
            for arm in 1..7 {
                let mut got = raw.clone();
                run_transform(&p, &mut got, arm - 1, kind);
                assert_eq!(got, expected, "full {log} {kind} arm{arm}");
            }
            println!(
                "check {kind} 2^{log}: all arms identical, digest {}",
                digest(&expected)
            );
            if timing {
                for (arm, label) in ARMS.iter().enumerate() {
                    let name = format!("{kind} 2^{log} {label}");
                    if arm == 0 {
                        let mut out = input.clone();
                        rows.push(timed(&name, n, || {
                            out.copy_from_slice(black_box(&input));
                            production_transform(&base, &mut out, kind);
                            black_box(&out);
                        }));
                    } else {
                        let mut out = raw.clone();
                        rows.push(timed(&name, n, || {
                            out.copy_from_slice(black_box(&raw));
                            run_transform(&p, &mut out, arm - 1, kind);
                            black_box(&out);
                        }));
                    }
                }
            }
        }
    }
    for log in [20, 22] {
        let k = 1usize << log;
        let row = rng.values(k);
        let raw = words(&row);
        let mut mr = Rng(0x7e55_e7a0_0047_0402);
        let masks = mr.values(200);
        let mask = words(&masks);
        let p = Plan::new(2 * k, k);
        let base = RowEncoder::new(k, 1, 200).unwrap();
        let mut expected = vec![Fp::ZERO; 2 * k];
        let mut i = 0;
        base.encode_row(
            &row,
            &mut || {
                let x = masks[i];
                i += 1;
                x
            },
            &mut expected,
        )
        .unwrap();
        let expected = words(&expected);
        for arm in 0..6 {
            let mut out = vec![0; 2 * k];
            run_encode(&p, &raw, &mask, &mut out, arm);
            assert_eq!(out, expected, "full encode {log} arm{arm}");
        }
        println!(
            "check encode 2^{log}: all arms identical, digest {}",
            digest(&expected)
        );
        if timing {
            for (arm, label) in ARMS.iter().enumerate() {
                let name = format!("encode 2^{log} {label}");
                if arm == 0 {
                    let mut out = vec![Fp::ZERO; 2 * k];
                    rows.push(timed(&name, k, || {
                        let mut i = 0;
                        base.encode_row(
                            black_box(&row),
                            &mut || {
                                let x = masks[i];
                                i += 1;
                                x
                            },
                            &mut out,
                        )
                        .unwrap();
                        black_box(&out);
                    }));
                } else {
                    let mut out = vec![0; 2 * k];
                    rows.push(timed(&name, k, || {
                        run_encode(&p, black_box(&raw), black_box(&mask), &mut out, arm - 1);
                        black_box(&out);
                    }));
                }
            }
        }
    }
    rows
}
fn main() {
    let command = std::env::args().nth(1).unwrap_or_else(|| "check".into());
    assert!(["check", "all"].contains(&command.as_str()));
    check_all();
    let timing = command == "all";
    if timing {
        big_cases(false);
    }
    let rows = big_cases(timing);
    if timing {
        std::fs::write(concat!(env!("CARGO_MANIFEST_DIR"),"/results/run.json"),serde_json::to_string_pretty(&json!({"experiment":"EXP-0014","warmups":3,"samples":10,"min_batch_ms":20,"cases":rows})).unwrap()).unwrap();
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn arithmetic_and_vectors() {
        check_all();
    }
}
