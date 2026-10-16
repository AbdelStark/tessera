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
