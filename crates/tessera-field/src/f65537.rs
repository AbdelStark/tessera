// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The test field `F65537`, `P = 2^16 + 1` (RFC-0002 'Traits'; 07 §4).
//!
//! Small enough that a cheating prover who guesses challenges succeeds often enough to measure,
//! so soundness-calibration tests can compare empirical false-accept rates with the analytic
//! bounds. Never used for proofs.

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::{Field, PrimeField};

const P: u64 = 65_537;

/// An element of `F65537`, held in `[0, P)`.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct F65537(u64);

impl F65537 {
    /// A generator of the multiplicative group (`3`).
    pub const MULTIPLICATIVE_GENERATOR: Self = Self(3);
}

impl fmt::Debug for F65537 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Add for F65537 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self((self.0 + rhs.0) % P)
    }
}

impl Sub for F65537 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self((self.0 + P - rhs.0) % P)
    }
}

impl Mul for F65537 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0 % P)
    }
}

impl Neg for F65537 {
    type Output = Self;
    fn neg(self) -> Self {
        Self((P - self.0) % P)
    }
}

impl AddAssign for F65537 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for F65537 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for F65537 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Field for F65537 {
    const ZERO: Self = Self(0);
    const ONE: Self = Self(1);

    fn square(&self) -> Self {
        *self * *self
    }

    fn double(&self) -> Self {
        *self + *self
    }

    fn inverse(&self) -> Option<Self> {
        (self.0 != 0).then(|| self.pow(P - 2))
    }

    fn pow(&self, mut exp: u64) -> Self {
        let (mut acc, mut square) = (Self::ONE, *self);
        while exp > 0 {
            if exp & 1 == 1 {
                acc *= square;
            }
            square = square.square();
            exp >>= 1;
        }
        acc
    }
}

impl PrimeField for F65537 {
    const MODULUS: u64 = P;
    const TWO_ADICITY: u32 = 16;

    fn from_u64_reduced(x: u64) -> Self {
        Self(x % P)
    }

    fn from_i64(x: i64) -> Self {
        Self(x.rem_euclid(P as i64) as u64)
    }

    fn from_canonical_u64(x: u64) -> Option<Self> {
        (x < P).then_some(Self(x))
    }

    fn to_canonical_u64(&self) -> u64 {
        self.0
    }

    fn to_centered_i128(&self) -> i128 {
        let x = self.0 as i128;
        if self.0 > (P - 1) / 2 {
            x - P as i128
        } else {
            x
        }
    }

    fn two_adic_generator(log_n: u32) -> Self {
        assert!(
            log_n <= 16,
            "F65537 has no element of order 2^{log_n}; the two-adicity is 16"
        );
        // P − 1 = 2^16, so 3^(2^(16 − log_n)) has order exactly 2^log_n.
        Self::MULTIPLICATIVE_GENERATOR.pow(1 << (16 - log_n))
    }
}
