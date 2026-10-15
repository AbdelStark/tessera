// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `Fp`, the Goldilocks field `P = 2^64 − 2^32 + 1` (RFC-0002 'Constants').
//!
//! Elements are held in Montgomery form, `a·R mod P` with `R = 2^64`, and multiplied with
//! Montgomery reduction (REDC). The verifier sees only public and protocol data, so it needs no
//! constant-time arithmetic (INV-ARCH-2 constrains the prover); this code favors plain,
//! checkable steps.

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// The modulus.
const P: u64 = 0xFFFF_FFFF_0000_0001;

/// `−P^{-1} mod 2^64`, by Newton's iteration for the inverse modulo `2^64`.
const P_PRIME: u64 = {
    // inverse ← inverse·(2 − P·inverse) doubles the number of correct low bits; P is odd, so
    // 1 is correct to one bit and six steps reach 64.
    let mut inverse: u64 = 1;
    let mut i = 0;
    while i < 6 {
        inverse = inverse.wrapping_mul(2u64.wrapping_sub(P.wrapping_mul(inverse)));
        i += 1;
    }
    inverse.wrapping_neg()
};

/// `R mod P = 2^64 mod P`, the Montgomery form of 1.
const R_MOD_P: u64 = ((1u128 << 64) % P as u128) as u64;

/// `R^2 mod P`: multiplying by it in Montgomery form maps `x` to `x·R mod P`.
const R2_MOD_P: u64 = ((R_MOD_P as u128 * R_MOD_P as u128) % P as u128) as u64;

/// Montgomery reduction: `t·R^{-1} mod P` for `t < P·R`.
const fn redc(t: u128) -> u64 {
    let t_lo = t as u64;
    let t_hi = (t >> 64) as u64;
    // m·P ≡ −t (mod 2^64), so t + m·P is divisible by 2^64.
    let m = t_lo.wrapping_mul(P_PRIME);
    let mp = m as u128 * P as u128;
    let mp_hi = (mp >> 64) as u64;
    // The low words t_lo and (m·P)_lo sum to 0 or 2^64: they carry exactly when t_lo ≠ 0.
    let carry = (t_lo != 0) as u64;
    // (t + m·P) / 2^64 = t_hi + mp_hi + carry, with t_hi < P (t < P·2^64) and mp_hi < P
    // (m < 2^64), so the true sum is below 2P and one subtraction of P reduces it.
    let (sum, overflow) = t_hi.overflowing_add(mp_hi + carry);
    if overflow || sum >= P {
        sum.wrapping_sub(P)
    } else {
        sum
    }
}

/// An element of `Fp`, in Montgomery form.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Fp(u64);

impl Fp {
    /// The modulus `P = 2^64 − 2^32 + 1`.
    pub const MODULUS: u64 = P;
    /// 0.
    pub const ZERO: Self = Self(0);
    /// 1.
    pub const ONE: Self = Self(R_MOD_P);
    /// 7: the generator of `Fp^*` and the extension's non-residue (RFC-0002 'Constants').
    pub const SEVEN: Self = Self::reduce(7);

    /// `x mod P`.
    pub const fn reduce(x: u64) -> Self {
        // x·R^2 < 2^64·P = P·R for every 64-bit x, so REDC takes it whole: (x·R^2)·R^{-1} = x·R.
        Self(redc(x as u128 * R2_MOD_P as u128))
    }

    /// The element `x`, or `None` when `x ≥ P` (RFC-0002 'Canonical encoding').
    pub const fn from_canonical(x: u64) -> Option<Self> {
        if x < P { Some(Self::reduce(x)) } else { None }
    }

    /// The canonical representative in `[0, P)`.
    pub const fn to_canonical(self) -> u64 {
        redc(self.0 as u128)
    }

    /// Decodes 8 little-endian bytes, rejecting values `≥ P`.
    pub fn from_le_bytes(bytes: [u8; 8]) -> Option<Self> {
        Self::from_canonical(u64::from_le_bytes(bytes))
    }

    /// The canonical 8-byte little-endian encoding.
    pub fn to_le_bytes(self) -> [u8; 8] {
        self.to_canonical().to_le_bytes()
    }

    /// `self^exp` by square-and-multiply.
    pub fn pow(self, mut exp: u64) -> Self {
        let (mut acc, mut square) = (Self::ONE, self);
        while exp > 0 {
            if exp & 1 == 1 {
                acc *= square;
            }
            square *= square;
            exp >>= 1;
        }
        acc
    }

    /// `self^(P − 2) = self^{-1}`, or `None` for zero.
    pub fn inverse(self) -> Option<Self> {
        if self == Self::ZERO {
            None
        } else {
            Some(self.pow(P - 2))
        }
    }
}

impl fmt::Debug for Fp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_canonical())
    }
}

impl Add for Fp {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        // Both operands are below P, so the true sum is below 2P.
        let (sum, overflow) = self.0.overflowing_add(rhs.0);
        Self(if overflow || sum >= P {
            sum.wrapping_sub(P)
        } else {
            sum
        })
    }
}

impl Sub for Fp {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(if self.0 >= rhs.0 {
            self.0 - rhs.0
        } else {
            self.0.wrapping_sub(rhs.0).wrapping_add(P)
        })
    }
}

impl Mul for Fp {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        // (aR)(bR)R^{-1} = abR: the product stays in Montgomery form.
        Self(redc(self.0 as u128 * rhs.0 as u128))
    }
}

impl Neg for Fp {
    type Output = Self;
    fn neg(self) -> Self {
        Self::ZERO - self
    }
}

impl AddAssign for Fp {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for Fp {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for Fp {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn montgomery_constants_are_what_their_definitions_say() {
        // P^{-1} mod 2^64 = 1 + 2^32, because (1 − 2^32)(1 + 2^32) = 1 − 2^64.
        assert_eq!(P.wrapping_mul(P_PRIME.wrapping_neg()), 1);
        assert_eq!(P_PRIME, 0xFFFF_FFFE_FFFF_FFFF);
        assert_eq!(R_MOD_P, (1 << 32) - 1);
        assert_eq!(Fp::ONE.to_canonical(), 1);
        assert_eq!(Fp::reduce(P).to_canonical(), 0);
    }
}
