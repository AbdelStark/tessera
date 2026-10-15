// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The Goldilocks field `Fp`, `P = 2^64 − 2^32 + 1` (RFC-0002).
//!
//! Elements are held canonically, in `[0, P)`, after every operation. Arithmetic is branch-free in
//! the values (INV-ARCH-2): conditional corrections are computed with masks, never with `if` on
//! data. Every operation is a `const fn`, so the constants below (the two-adic roots of unity) are
//! computed at compile time from their definitions rather than copied.

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::{Field, PrimeField};

/// `P = 2^64 − 2^32 + 1`.
const P: u64 = 0xFFFF_FFFF_0000_0001;
/// `2^64 − P = 2^32 − 1`, so `2^64 ≡ EPSILON (mod P)`.
const EPSILON: u64 = 0xFFFF_FFFF;

/// An all-ones mask when `bit` is set, zero otherwise.
#[inline(always)]
const fn mask(bit: bool) -> u64 {
    (bit as u64).wrapping_neg()
}

/// Reduces `x < 2^64` to `[0, P)`: `x < 2P`, so one conditional subtraction suffices.
#[inline(always)]
const fn canonicalize(x: u64) -> u64 {
    // Subtract P; on borrow (x < P) add it back. Three instructions on aarch64 and x86-64.
    let (reduced, borrow) = x.overflowing_sub(P);
    reduced.wrapping_add(P & mask(borrow))
}

#[inline(always)]
const fn add(a: u64, b: u64) -> u64 {
    // a, b < P. If a + b wraps, the true sum S lies in [2^64, 2P), and S − P = wrapped + EPSILON
    // is below P, so the second addition cannot wrap and the final correction is a no-op.
    let (sum, carry) = a.overflowing_add(b);
    canonicalize(sum.wrapping_add(EPSILON & mask(carry)))
}

#[inline(always)]
const fn sub(a: u64, b: u64) -> u64 {
    // On borrow the wrapped difference is a − b + 2^64; a − b + P = wrapped − EPSILON lies in
    // (0, P), so the result is canonical without a further correction.
    let (diff, borrow) = a.overflowing_sub(b);
    diff.wrapping_sub(EPSILON & mask(borrow))
}

/// Reduces a 128-bit product modulo `P` (RFC-0002 'Representation and reduction').
///
/// With `x = lo + 2^64·(hi_lo + 2^32·hi_hi)`: `2^64 ≡ 2^32 − 1` and `2^96 ≡ −1`, so
/// `x ≡ lo − hi_hi + hi_lo·(2^32 − 1) (mod P)`.
#[inline(always)]
const fn reduce128(x: u128) -> u64 {
    let lo = x as u64;
    let hi = (x >> 64) as u64;
    let hi_hi = hi >> 32;
    let hi_lo = hi & EPSILON;
    // lo − hi_hi. On borrow (lo < hi_hi < 2^32) the wrapped value exceeds 2^64 − 2^32, so
    // subtracting EPSILON (that is, adding P − 2^64) cannot underflow.
    let (t0, borrow) = lo.overflowing_sub(hi_hi);
    let t0 = t0.wrapping_sub(EPSILON & mask(borrow));
    // hi_lo·(2^32 − 1) ≤ (2^32 − 1)^2 < 2^64.
    let t1 = hi_lo * EPSILON;
    // t0 + t1 < 2^65 − 2^33; on carry the wrapped sum is at most 2^64 − 2^33, so adding EPSILON
    // (that is, 2^64 mod P) cannot wrap again.
    let (t2, carry) = t0.overflowing_add(t1);
    canonicalize(t2.wrapping_add(EPSILON & mask(carry)))
}

#[inline(always)]
const fn mul(a: u64, b: u64) -> u64 {
    reduce128(a as u128 * b as u128)
}

/// `base^exp` by square-and-multiply. The loop depends on `exp` only, which is public everywhere
/// this is used (RFC-0002 'Timing').
const fn pow(base: u64, mut exp: u64) -> u64 {
    let mut acc = 1;
    let mut square = base;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = mul(acc, square);
        }
        square = mul(square, square);
        exp >>= 1;
    }
    acc
}

/// `x^(2^k)`.
const fn square_n(mut x: u64, k: u32) -> u64 {
    let mut i = 0;
    while i < k {
        x = mul(x, x);
        i += 1;
    }
    x
}

/// A primitive `2^32`-th root of unity: `7^((P − 1) / 2^32)` (RFC-0002 'Constants').
const ROOT_2_32: u64 = pow(7, (P - 1) >> 32);

/// `ROOTS[k]` is a primitive `2^k`-th root of unity, `ROOT_2_32^(2^(32 − k))`.
const ROOTS: [u64; 33] = {
    let mut roots = [0u64; 33];
    let mut k = 0;
    while k <= 32 {
        roots[k] = square_n(ROOT_2_32, 32 - k as u32);
        k += 1;
    }
    roots
};

/// An element of the Goldilocks field, `P = 2^64 − 2^32 + 1`, held in canonical form.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Goldilocks(u64);

impl Goldilocks {
    /// The modulus `P`.
    pub const ORDER: u64 = P;
    /// The generator of the multiplicative group used by RFC-0002 (`7`).
    pub const MULTIPLICATIVE_GENERATOR: Self = Self(7);
    /// The primitive `2^32`-th root of unity `7^((P − 1) / 2^32)`.
    pub const ROOT_2_32: Self = Self(ROOT_2_32);
}

impl fmt::Debug for Goldilocks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Add for Goldilocks {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(add(self.0, rhs.0))
    }
}

impl Sub for Goldilocks {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(sub(self.0, rhs.0))
    }
}

impl Mul for Goldilocks {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self(mul(self.0, rhs.0))
    }
}

impl Neg for Goldilocks {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(sub(0, self.0))
    }
}

impl AddAssign for Goldilocks {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for Goldilocks {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for Goldilocks {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Field for Goldilocks {
    const ZERO: Self = Self(0);
    const ONE: Self = Self(1);

    #[inline]
    fn square(&self) -> Self {
        Self(mul(self.0, self.0))
    }

    #[inline]
    fn double(&self) -> Self {
        Self(add(self.0, self.0))
    }

    /// `x^(P − 2)` by a fixed addition chain of 63 squarings and 9 multiplications (RFC-0002
    /// 'Representation and reduction'). With `a_k = x^(2^k − 1)`:
    /// `P − 2 = (2^32 − 2)·2^32 + (2^32 − 1)`, so `x^(P − 2) = b^(2^32)·b·x` with `b = a_31^2`.
    fn inverse(&self) -> Option<Self> {
        let x = self.0;
        let a2 = mul(mul(x, x), x); // x^(2^2 - 1)
        let a3 = mul(mul(a2, a2), x); // x^(2^3 - 1)
        let a6 = mul(square_n(a3, 3), a3);
        let a12 = mul(square_n(a6, 6), a6);
        let a24 = mul(square_n(a12, 12), a12);
        let a30 = mul(square_n(a24, 6), a6);
        let a31 = mul(mul(a30, a30), x);
        let b = mul(a31, a31); // x^(2^32 - 2)
        let inv = mul(mul(square_n(b, 32), b), x);
        // Zero has no inverse; the chain maps it to zero.
        (x != 0).then_some(Self(inv))
    }

    #[inline]
    fn pow(&self, exp: u64) -> Self {
        Self(pow(self.0, exp))
    }
}

impl PrimeField for Goldilocks {
    const MODULUS: u64 = P;
    const TWO_ADICITY: u32 = 32;

    #[inline]
    fn from_u64_reduced(x: u64) -> Self {
        Self(canonicalize(x))
    }

    /// The signed embedding: `x ≥ 0` maps to `x`, `x < 0` to `P − |x|`. Branch-free, because the
    /// witness engine embeds secret values.
    #[inline]
    fn from_i64(x: i64) -> Self {
        let magnitude = x.unsigned_abs(); // ≤ 2^63 < P, so already canonical
        let negated = sub(0, magnitude);
        let negative = mask(x < 0);
        Self((negated & negative) | (magnitude & !negative))
    }

    #[inline]
    fn from_canonical_u64(x: u64) -> Option<Self> {
        (x < P).then_some(Self(x))
    }

    #[inline]
    fn to_canonical_u64(&self) -> u64 {
        self.0
    }

    #[inline]
    fn to_centered_i128(&self) -> i128 {
        let x = self.0 as i128;
        let high = (self.0 > (P - 1) / 2) as i128;
        x - high * P as i128
    }

    fn two_adic_generator(log_n: u32) -> Self {
        assert!(
            log_n <= 32,
            "Goldilocks has no element of order 2^{log_n}; the two-adicity is 32"
        );
        Self(ROOTS[log_n as usize])
    }
}
