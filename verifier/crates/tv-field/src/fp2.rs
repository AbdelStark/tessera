// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `Fp2 = Fp[X]/(X^2 − 7)` (RFC-0002 'Quadratic extension'), the field every challenge lives in.

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::Fp;

/// `c0 + c1·X` with `X^2 = 7`.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Fp2 {
    /// The constant coefficient.
    pub c0: Fp,
    /// The coefficient of `X`.
    pub c1: Fp,
}

impl Fp2 {
    /// 0.
    pub const ZERO: Self = Self {
        c0: Fp::ZERO,
        c1: Fp::ZERO,
    };
    /// 1.
    pub const ONE: Self = Self {
        c0: Fp::ONE,
        c1: Fp::ZERO,
    };

    /// `c0 + c1·X`.
    pub const fn new(c0: Fp, c1: Fp) -> Self {
        Self { c0, c1 }
    }

    /// Decodes `c0` then `c1`, 8 little-endian bytes each, rejecting non-canonical coefficients.
    pub fn from_le_bytes(bytes: [u8; 16]) -> Option<Self> {
        let (c0, c1) = bytes.split_at(8);
        Some(Self {
            c0: Fp::from_le_bytes(c0.try_into().ok()?)?,
            c1: Fp::from_le_bytes(c1.try_into().ok()?)?,
        })
    }

    /// The canonical 16-byte encoding: `c0` then `c1`.
    pub fn to_le_bytes(self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&self.c0.to_le_bytes());
        out[8..].copy_from_slice(&self.c1.to_le_bytes());
        out
    }

    /// `c0 − c1·X`.
    pub fn conjugate(self) -> Self {
        Self {
            c0: self.c0,
            c1: -self.c1,
        }
    }

    /// `c0^2 − 7·c1^2`, zero only for zero because 7 is not a square in `Fp`.
    pub fn norm(self) -> Fp {
        self.c0 * self.c0 - Fp::SEVEN * self.c1 * self.c1
    }

    /// `conjugate / norm`, or `None` for zero: `self · conjugate = norm`, an element of `Fp`.
    pub fn inverse(self) -> Option<Self> {
        Some(self.conjugate() * self.norm().inverse()?)
    }
}

impl From<Fp> for Fp2 {
    fn from(c0: Fp) -> Self {
        Self { c0, c1: Fp::ZERO }
    }
}

impl fmt::Debug for Fp2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:?} + {:?}·X)", self.c0, self.c1)
    }
}

impl Add for Fp2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            c0: self.c0 + rhs.c0,
            c1: self.c1 + rhs.c1,
        }
    }
}

impl Sub for Fp2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            c0: self.c0 - rhs.c0,
            c1: self.c1 - rhs.c1,
        }
    }
}

impl Mul for Fp2 {
    type Output = Self;
    /// The schoolbook product reduced by `X^2 = 7`: four base multiplications.
    fn mul(self, rhs: Self) -> Self {
        Self {
            c0: self.c0 * rhs.c0 + Fp::SEVEN * (self.c1 * rhs.c1),
            c1: self.c0 * rhs.c1 + self.c1 * rhs.c0,
        }
    }
}

impl Mul<Fp> for Fp2 {
    type Output = Self;
    fn mul(self, rhs: Fp) -> Self {
        Self {
            c0: self.c0 * rhs,
            c1: self.c1 * rhs,
        }
    }
}

impl Neg for Fp2 {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            c0: -self.c0,
            c1: -self.c1,
        }
    }
}

impl AddAssign for Fp2 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for Fp2 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for Fp2 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
