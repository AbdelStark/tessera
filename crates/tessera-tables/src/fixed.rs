// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The fixed-point arithmetic of the transcendental generators (RFC-0008 'Fixed-point
//! generators'). A Q64 number `v` stands for `v / 2^64`. Every intermediate stays below `2^128`,
//! every division is floor division, and the results are the definition of the tables: an
//! implementation computes exactly these integers.

/// `⌊ln 2 · 2^64⌋`, pinned in `spec/vectors/tables/constants.json`.
pub(crate) const LN2: u128 = 0xB172_17F7_D1CF_79AB;

/// 1 in Q64.
const ONE: u128 = 1 << 64;

/// 1 in Q62, the format of [`exp_pos`]'s result, which reaches 2.
const ONE_62: u128 = 1 << 62;

/// The Taylor terms of both series: the remainder after the 20th is below `2^-70` on `[0, ln 2)`.
const TERMS: u128 = 20;

/// `e^{−x}` for a Q64 `x`: `(y, n)` with `e^{−x/2^64} ≈ y · 2^{−64−n}` and `0 < y ≤ 2^64`.
///
/// With `n = ⌊x / LN2⌋` and `r = x − n·LN2 < LN2`, Horner's rule over the Taylor series of
/// `e^{−r}` gives `y`: from `y = 2^64`, for `i` from 20 down to 1, `y ← 2^64 − ⌊r·y / (i·2^64)⌋`.
/// `r·y < 2^128` throughout.
pub(crate) fn exp_neg(x: u128) -> (u128, u128) {
    let n = x / LN2;
    let r = x - n * LN2;
    let mut y = ONE;
    for i in (1..=TERMS).rev() {
        y = ONE - r * y / (i * ONE);
    }
    (y, n)
}

/// `e^{x}` for a Q64 `x` with `0 ≤ x ≤ LN2`, in Q62: from `y = 2^62`, for `i` from 20 down to 1,
/// `y ← 2^62 + ⌊x·y / (i·2^64)⌋`. `y ≤ 2^63` and `x·y < 2^127` throughout. Every step rounds
/// down and the series' dropped terms are positive, so `y / 2^62` never exceeds `e^{x/2^64}`.
pub(crate) fn exp_pos(x: u128) -> u128 {
    let mut y = ONE_62;
    for i in (1..=TERMS).rev() {
        y = ONE_62 + x * y / (i * ONE);
    }
    y
}

/// `⌊2π · 2^64⌋`, pinned in `spec/vectors/tables/constants.json`.
pub(crate) const TAU: u128 = 0x6_487E_D511_0B46_11A6;

/// `⌊π/2 · 2^64⌋`, pinned in `spec/vectors/tables/constants.json`.
pub(crate) const HALF_PI: u128 = 0x1_921F_B544_42D1_8469;

/// The terms of the series of `atanh`, whose ratio is at most `1/9`: the remainder after the
/// 22nd is below `2^-69`.
const ATANH_TERMS: u128 = 22;

/// The pairs of Taylor terms of `cos` and `sin`: on `[0, π/2)` the remainder after the 13th
/// pair is below `2^-75`.
const TRIG_TERMS: i128 = 13;

/// `ln n` in Q64 for an integer `n ≥ 1`. With `n = 2^m · u` and `u ∈ [1, 2)`,
/// `ln u = 2 · atanh(z)` for `z = (u − 1)/(u + 1) < 1/3`, summed by Horner's rule over `w = z²`:
/// from `a = ⌊2^64 / 45⌋`, for `k` from 21 down to 0, `a ← ⌊2^64 / (2k + 1)⌋ + ⌊w · a / 2^64⌋`;
/// then `ln u = ⌊z · a / 2^63⌋`, and `ln n = m · LN2 + ln u`.
pub(crate) fn ln(n: u64) -> u128 {
    let m = 63 - n.leading_zeros();
    let (n, top) = (u128::from(n), 1u128 << m);
    let z = ((n - top) << 64) / (n + top);
    let w = (z * z) >> 64;
    let mut a = ONE / (2 * ATANH_TERMS + 1);
    for k in (0..ATANH_TERMS).rev() {
        a = ONE / (2 * k + 1) + ((w * a) >> 64);
    }
    u128::from(m) * LN2 + ((z * a) >> 63)
}

/// `(cos θ, sin θ)` in Q62, signed, for a Q64 angle `θ < 2^100`. The angle is reduced modulo
/// `TAU`, then to `r ∈ [0, HALF_PI)` and its quadrant; Horner's rule over `s = r²` gives
/// `cos r`, and `sin r` from its series divided by `r`, both from `2^62`:
/// `c ← 2^62 − ⌊s·c / ((2k−1)·2k·2^62)⌋` and `t ← 2^62 − ⌊s·t / (2k·(2k+1)·2^62)⌋` for `k` from
/// 13 down to 1, then `sin r = ⌊r·t / 2^62⌋`.
pub(crate) fn cos_sin(angle: u128) -> (i128, i128) {
    let turn = angle % TAU;
    let quadrant = turn / HALF_PI;
    // In Q62, `r < 2^62.66`, so `r² < 2^126`; `s < 2^64` and `|c|, |t| ≤ 2^62` keep products
    // below `2^127`.
    let r = i128::try_from((turn - quadrant * HALF_PI) >> 2).unwrap_or(0);
    let one = 1i128 << 62;
    let s = (r * r) >> 62;
    let (mut c, mut t) = (one, one);
    for k in (1..=TRIG_TERMS).rev() {
        c = one - (s * c).div_euclid((2 * k - 1) * (2 * k) * one);
        t = one - (s * t).div_euclid((2 * k) * (2 * k + 1) * one);
    }
    let sin = (r * t) >> 62;
    match quadrant % 4 {
        0 => (c, sin),
        1 => (-sin, c),
        2 => (-c, -sin),
        _ => (sin, -c),
    }
}
