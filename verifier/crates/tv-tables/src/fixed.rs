// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The fixed-point arithmetic of RFC-0008 'Fixed-point generators': the pinned constants and
//! the five functions that define the transcendental kinds.
//!
//! A Q64 number `v` stands for `v / 2^64`, and Q62 and Q63 likewise. Division is floor division
//! and `>>` the arithmetic shift. The RFC keeps every intermediate below `2^128`, so `u128` and
//! `i128` hold them exactly. These integers are the definition of the tables, not an
//! approximation of it: a function that rounds differently defines other tables.
//!
//! Where the RFC divides a product by `d·2^64` (or `d·2^62`), the code computes
//! `⌊⌊product / 2^64⌋ / d⌋`. For an integer `N` and positive integers `a` and `b`,
//! `⌊⌊N / a⌋ / b⌋ = ⌊N / (a·b)⌋`, negative `N` included, so the result is the RFC's. The
//! arithmetic is checked: each function returns `None` when an argument is outside its range,
//! or when an intermediate leaves the range the RFC states, which no argument in range reaches.

/// `LN2 = ⌊ln 2 · 2^64⌋`, pinned in `spec/vectors/tables/constants.json`.
pub const LN2: u128 = 0xB172_17F7_D1CF_79AB;

/// `TAU = ⌊2π · 2^64⌋`, pinned in `spec/vectors/tables/constants.json`.
pub const TAU: u128 = 0x6_487E_D511_0B46_11A6;

/// `HALF_PI = ⌊π/2 · 2^64⌋`, pinned in `spec/vectors/tables/constants.json`.
pub const HALF_PI: u128 = 0x1_921F_B544_42D1_8469;

/// 1 in Q64.
const ONE_Q64: u128 = 1 << 64;

/// 1 in Q62.
const ONE_Q62: u128 = 1 << 62;

/// 1 in Q62, signed.
const ONE_Q62_SIGNED: i128 = 1 << 62;

/// The angles `cos_sin` accepts are below `2^100` in Q64.
const ANGLE_BOUND: u128 = 1 << 100;

/// `⌊a·b / 2^shift⌋` for unsigned `a` and `b`, or `None` when `a·b ≥ 2^128`.
fn mul_shr(a: u128, b: u128, shift: u32) -> Option<u128> {
    a.checked_mul(b)?.checked_shr(shift)
}

/// `exp_neg(x)` for a Q64 `x ≥ 0`: `(y, n)` with `e^{−x/2^64} ≈ y · 2^{−64−n}` and
/// `0 < y ≤ 2^64`.
///
/// 1. `n = ⌊x / LN2⌋` and `r = x − n·LN2`.
/// 2. `y = 2^64`, then for `i` from 20 down to 1, `y ← 2^64 − ⌊r·y / (i·2^64)⌋`.
///
/// `r < 2^64` and `y ≤ 2^64`, so `r·y < 2^128`.
pub fn exp_neg(x: u128) -> Option<(u128, u128)> {
    let n = x.checked_div(LN2)?;
    let r = x.checked_rem(LN2)?;
    let mut y = ONE_Q64;
    for i in (1..=20u128).rev() {
        let term = mul_shr(r, y, 64)?.checked_div(i)?;
        y = ONE_Q64.checked_sub(term)?;
    }
    Some((y, n))
}

/// `exp_pos(x)` for a Q64 `x` with `0 ≤ x ≤ LN2`: `e^{x/2^64}` in Q62, never above it.
///
/// `y = 2^62`, then for `i` from 20 down to 1, `y ← 2^62 + ⌊x·y / (i·2^64)⌋`. `y ≤ 2^63`
/// throughout. `None` for `x > LN2`.
pub fn exp_pos(x: u128) -> Option<u128> {
    if x > LN2 {
        return None;
    }
    let mut y = ONE_Q62;
    for i in (1..=20u128).rev() {
        let term = mul_shr(x, y, 64)?.checked_div(i)?;
        y = ONE_Q62.checked_add(term)?;
    }
    Some(y)
}

/// `ln(n)` for an integer `1 ≤ n < 2^64`: `ln n` in Q64. `None` for `n = 0`.
///
/// With `n = 2^m·u`, `u ∈ [1, 2)`, `ln n = m·LN2 + 2·atanh(z)` for `z = (u − 1)/(u + 1) < 1/3`:
///
/// 1. `z = ⌊(n − 2^m)·2^64 / (n + 2^m)⌋` and `w = ⌊z² / 2^64⌋`.
/// 2. `a = ⌊2^64 / 45⌋`, then for `k` from 21 down to 0, `a ← ⌊2^64 / (2k + 1)⌋ + ⌊w·a / 2^64⌋`.
/// 3. The result is `m·LN2 + ⌊z·a / 2^63⌋`.
pub fn ln(n: u64) -> Option<u128> {
    let m = n.checked_ilog2()?;
    let power = 1u128.checked_shl(m)?;
    let n = u128::from(n);
    let z = n
        .checked_sub(power)?
        .checked_mul(ONE_Q64)?
        .checked_div(n.checked_add(power)?)?;
    let w = mul_shr(z, z, 64)?;
    let mut a = ONE_Q64.checked_div(45)?;
    for k in (0..=21u128).rev() {
        let odd = k.checked_mul(2)?.checked_add(1)?;
        a = ONE_Q64.checked_div(odd)?.checked_add(mul_shr(w, a, 64)?)?;
    }
    u128::from(m)
        .checked_mul(LN2)?
        .checked_add(mul_shr(z, a, 63)?)
}

/// `cos_sin(θ)` for a Q64 angle `θ < 2^100`: `(cos θ, sin θ)` in Q62, signed. `None` for a
/// larger angle.
///
/// 1. `t = θ mod TAU`, `q = ⌊t / HALF_PI⌋` and `r = (t − q·HALF_PI) >> 2`, the angle within
///    its quadrant in Q62.
/// 2. `s = ⌊r² / 2^62⌋` and `c = t' = 2^62`, then for `k` from 13 down to 1,
///    `c ← 2^62 − ⌊s·c / ((2k−1)·2k·2^62)⌋` and `t' ← 2^62 − ⌊s·t' / (2k·(2k+1)·2^62)⌋`, with
///    signed floor division.
/// 3. With `sin r = ⌊r·t' / 2^62⌋`, quadrant `q mod 4` gives `(c, sin r)`, `(−sin r, c)`,
///    `(−c, −sin r)` or `(sin r, −c)`.
///
/// `TAU − 4·HALF_PI = 2`, so `q` is 4 for the two largest `t`, whose `r` is 0: `q mod 4` puts
/// them in the first quadrant.
pub fn cos_sin(theta: u128) -> Option<(i128, i128)> {
    if theta >= ANGLE_BOUND {
        return None;
    }
    let t = theta.checked_rem(TAU)?;
    let q = t.checked_div(HALF_PI)?;
    let within = t.checked_sub(q.checked_mul(HALF_PI)?)?.checked_shr(2)?;
    let r = i128::try_from(within).ok()?;
    let s = r.checked_mul(r)?.checked_shr(62)?;
    let (mut c, mut t_prime) = (ONE_Q62_SIGNED, ONE_Q62_SIGNED);
    for k in (1..=13i128).rev() {
        let two_k = k.checked_mul(2)?;
        let cos_divisor = two_k.checked_sub(1)?.checked_mul(two_k)?;
        let sin_divisor = two_k.checked_mul(two_k.checked_add(1)?)?;
        // `>> 62` on `i128` is the arithmetic shift, floor division by `2^62`, and
        // `div_euclid` by a positive divisor is floor division.
        let cos_term = s
            .checked_mul(c)?
            .checked_shr(62)?
            .checked_div_euclid(cos_divisor)?;
        let sin_term = s
            .checked_mul(t_prime)?
            .checked_shr(62)?
            .checked_div_euclid(sin_divisor)?;
        c = ONE_Q62_SIGNED.checked_sub(cos_term)?;
        t_prime = ONE_Q62_SIGNED.checked_sub(sin_term)?;
    }
    let sin_r = r.checked_mul(t_prime)?.checked_shr(62)?;
    Some(match q.checked_rem(4)? {
        0 => (c, sin_r),
        1 => (sin_r.checked_neg()?, c),
        2 => (c.checked_neg()?, sin_r.checked_neg()?),
        _ => (sin_r, c.checked_neg()?),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_neg_is_exact_at_zero_and_reduces_by_ln2() {
        assert_eq!(exp_neg(0), Some((ONE_Q64, 0)));
        // x = LN2 reduces to r = 0: y = 2^64 and n = 1, e^{-ln 2} = 2^{-1}.
        assert_eq!(exp_neg(LN2), Some((ONE_Q64, 1)));
        assert_eq!(exp_neg(5 * LN2 + 7).map(|(_, n)| n), Some(5));
        // The bound of the RFC: 0 < y ≤ 2^64 for every r below LN2.
        for x in [1, LN2 - 1, LN2 / 2, 1 << 63] {
            let (y, n) = exp_neg(x).unwrap();
            assert!(y > 0 && y <= ONE_Q64, "{x}");
            assert_eq!(n, 0);
        }
    }

    #[test]
    fn exp_pos_is_exact_at_zero_and_refuses_arguments_above_ln2() {
        assert_eq!(exp_pos(0), Some(ONE_Q62));
        let top = exp_pos(LN2).unwrap();
        // e^{ln 2} = 2, from below, and within four units of 2^-62.
        assert!(top < 2 * ONE_Q62 && top + 4 >= 2 * ONE_Q62, "{top}");
        assert_eq!(exp_pos(LN2 + 1), None);
    }

    #[test]
    fn ln_of_powers_of_two_is_a_multiple_of_ln2() {
        assert_eq!(ln(0), None);
        assert_eq!(ln(1), Some(0));
        for m in 0..64u32 {
            assert_eq!(ln(1 << m), Some(u128::from(m) * LN2), "2^{m}");
        }
        // ln 3 = 1.0986122886681098 · 2^64 ≈ 0x1193EA7AAD030A976, within a few units.
        let ln3 = ln(3).unwrap();
        assert!(ln3.abs_diff(0x1_193E_A7AA_D030_A976) < 8, "{ln3:#x}");
    }

    /// Exact outputs at arguments where one series term fewer changes the result, from a
    /// big-integer transcription of RFC-0008's steps (Python integers, each step's floor taken
    /// directly as `⌊a·b / (d·2^64)⌋`). The number of terms is part of the definition even where
    /// it moves a result by less than one unit, and a RoPE frequency is used exactly at every
    /// position.
    #[test]
    fn every_series_term_counts() {
        for (x, y, n) in [
            (0x2_C3DB_86C7_8CEB_0CD3, 0x80F7_5A59_2A3B_CD36, 3),
            (0x16_1DF6_C3B3_52B2_A144, 0x8869_F3BA_C294_B0E3, 31),
            (0x1_5CAB_FB99_E197_7962, 0x8325_DA09_8512_1BD8, 1),
            (0xD_1763_57B7_EF34_F27C, 0x8A71_B4BA_A2A1_40E0, 18),
        ] {
            assert_eq!(exp_neg(x), Some((y, n)), "exp_neg({x:#x})");
        }
        for (x, y) in [
            (0xAB4C_E0F9_F3F5_FB9C, 0x7CF6_C246_BC17_4A1C),
            (0xAF1B_2C74_FF5F_A6C6, 0x7ED5_E77A_50BE_C3C7),
            (0xA951_4AE0_AAA8_90E5, 0x7BFF_F150_2FEA_58AB),
        ] {
            assert_eq!(exp_pos(x), Some(y), "exp_pos({x:#x})");
        }
        for (n, expected) in [
            (4_583_742_181_530_100_207, 0x2A_F813_7DB0_37EF_9C6E),
            (17_361_478_931_665_825_463, 0x2C_4D00_4880_5370_E7B4),
            (8_983_506_198_948_470_377, 0x2B_A454_FECB_D5E9_257A),
        ] {
            assert_eq!(ln(n), Some(expected), "ln({n})");
        }
        for (theta, cos, sin) in [
            (
                0x3_72F8_279C_7DB0_5CEE_3485_58EF,
                11_452_494_408_838_716,
                4_611_671_798_050_081_262,
            ),
            (
                0xA_A758_9B31_B67E_A24A_70A8_8053,
                -4_570_296_597_474_265_658,
                616_471_364_844_957_025,
            ),
            (
                0x9_CE3E_A091_951C_48E4_0123_D0FA,
                -4_606_783_743_167_973_764,
                212_582_869_116_790_077,
            ),
        ] {
            assert_eq!(cos_sin(theta), Some((cos, sin)), "cos_sin({theta:#x})");
        }
    }

    #[test]
    fn cos_sin_is_exact_at_zero_and_refuses_large_angles() {
        assert_eq!(cos_sin(0), Some((ONE_Q62_SIGNED, 0)));
        assert_eq!(cos_sin(ANGLE_BOUND), None);
        // The two angles whose quadrant index is 4 are in the first quadrant, at r = 0.
        assert_eq!(cos_sin(TAU - 1), Some((ONE_Q62_SIGNED, 0)));
        assert_eq!(cos_sin(TAU - 2), Some((ONE_Q62_SIGNED, 0)));
        // A quarter turn: q = 1 and r = 0, so (−sin 0, cos 0) = (0, 1).
        assert_eq!(cos_sin(HALF_PI), Some((0, ONE_Q62_SIGNED)));
        assert_eq!(cos_sin(2 * HALF_PI), Some((-ONE_Q62_SIGNED, 0)));
        assert_eq!(cos_sin(3 * HALF_PI), Some((0, -ONE_Q62_SIGNED)));
        // A full turn and the angle itself agree.
        assert_eq!(cos_sin(TAU + 12345), cos_sin(12345));
    }
}
