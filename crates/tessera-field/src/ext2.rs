// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The quadratic extension `Fp2 = Fp[X]/(X^2 − W)` of RFC-0002 'Quadratic extension', in which
//! every challenge lives: about 2^128 elements for Goldilocks (`W = 7`).

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use tessera_codec::Zeroize;

use crate::{F65537, Field, Goldilocks, PrimeField};

/// A prime field with a fixed quadratic non-residue `W`, over which [`QuadraticExt`] is built.
pub trait ExtensionBase: PrimeField {
    /// The non-residue `W`: `X^2 − W` is irreducible.
    const NON_RESIDUE: Self;

    /// `W · x`. Fields override this when multiplication by `W` has a cheaper form.
    #[inline]
    fn mul_by_non_residue(x: Self) -> Self {
        Self::NON_RESIDUE * x
    }
}

impl ExtensionBase for Goldilocks {
    /// 7, the smallest quadratic non-residue modulo the Goldilocks prime (RFC-0002 'Constants').
    const NON_RESIDUE: Self = Goldilocks::MULTIPLICATIVE_GENERATOR;

    #[inline]
    fn mul_by_non_residue(x: Self) -> Self {
        x.mul_small(7)
    }
}

impl ExtensionBase for F65537 {
    /// 3, a generator of `F65537^*` and hence a non-residue.
    const NON_RESIDUE: Self = F65537::MULTIPLICATIVE_GENERATOR;
}

/// The operations of a quadratic extension of `B` (RFC-0002 'Traits').
pub trait QuadraticExtension<B: PrimeField>: Field + From<B> {
    /// The non-residue `W`.
    const NON_RESIDUE: B;

    /// `c0 + c1·X`.
    fn new(c0: B, c1: B) -> Self;

    /// `[c0, c1]`.
    fn coeffs(&self) -> [B; 2];

    /// Multiplication by a base-field element: two base multiplications.
    fn mul_base(&self, b: B) -> Self;

    /// `c0 − c1·X`, the image under the non-trivial automorphism.
    fn conjugate(&self) -> Self;

    /// `c0^2 − W·c1^2 = self · conjugate(self)`, an element of the base field.
    fn norm(&self) -> B;
}

/// `c0 + c1·X` with `X^2 = W`.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct QuadraticExt<B> {
    c0: B,
    c1: B,
}

/// Clears both coefficients (05 §3 OBS-2, OBS-6).
impl<B: Zeroize> Zeroize for QuadraticExt<B> {
    fn zeroize(&mut self) {
        self.c0.zeroize();
        self.c1.zeroize();
    }
}

/// `Fp2` over Goldilocks: `Fp[X]/(X^2 − 7)`.
pub type GoldilocksExt2 = QuadraticExt<Goldilocks>;
/// The extension of the test field: `F65537[X]/(X^2 − 3)`.
pub type F65537Ext2 = QuadraticExt<F65537>;

impl<B: ExtensionBase> fmt::Debug for QuadraticExt<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:?} + {:?}·X)", self.c0, self.c1)
    }
}

impl<B: ExtensionBase> From<B> for QuadraticExt<B> {
    #[inline]
    fn from(b: B) -> Self {
        Self { c0: b, c1: B::ZERO }
    }
}

impl<B: ExtensionBase> Add for QuadraticExt<B> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            c0: self.c0 + rhs.c0,
            c1: self.c1 + rhs.c1,
        }
    }
}

impl<B: ExtensionBase> Sub for QuadraticExt<B> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            c0: self.c0 - rhs.c0,
            c1: self.c1 - rhs.c1,
        }
    }
}

impl<B: ExtensionBase> Mul for QuadraticExt<B> {
    type Output = Self;
    /// Karatsuba: three base multiplications and one multiplication by `W` (RFC-0002).
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let v0 = self.c0 * rhs.c0;
        let v1 = self.c1 * rhs.c1;
        let cross = (self.c0 + self.c1) * (rhs.c0 + rhs.c1);
        Self {
            c0: v0 + B::mul_by_non_residue(v1),
            c1: cross - v0 - v1,
        }
    }
}

impl<B: ExtensionBase> Neg for QuadraticExt<B> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            c0: -self.c0,
            c1: -self.c1,
        }
    }
}

impl<B: ExtensionBase> AddAssign for QuadraticExt<B> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<B: ExtensionBase> SubAssign for QuadraticExt<B> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<B: ExtensionBase> MulAssign for QuadraticExt<B> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<B: ExtensionBase> Field for QuadraticExt<B> {
    const ZERO: Self = Self {
        c0: B::ZERO,
        c1: B::ZERO,
    };
    const ONE: Self = Self {
        c0: B::ONE,
        c1: B::ZERO,
    };

    /// `(c0 + c1·X)^2 = (c0^2 + W·c1^2) + 2·c0·c1·X`.
    #[inline]
    fn square(&self) -> Self {
        let cross = self.c0 * self.c1;
        Self {
            c0: self.c0.square() + B::mul_by_non_residue(self.c1.square()),
            c1: cross.double(),
        }
    }

    #[inline]
    fn double(&self) -> Self {
        Self {
            c0: self.c0.double(),
            c1: self.c1.double(),
        }
    }

    /// `conjugate · norm^{-1}`. The norm vanishes only at zero because `W` is a non-residue.
    fn inverse(&self) -> Option<Self> {
        let norm_inverse = self.norm().inverse()?;
        Some(self.conjugate().mul_base(norm_inverse))
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

impl<B: ExtensionBase> QuadraticExtension<B> for QuadraticExt<B> {
    const NON_RESIDUE: B = <B as ExtensionBase>::NON_RESIDUE;

    #[inline]
    fn new(c0: B, c1: B) -> Self {
        Self { c0, c1 }
    }

    #[inline]
    fn coeffs(&self) -> [B; 2] {
        [self.c0, self.c1]
    }

    #[inline]
    fn mul_base(&self, b: B) -> Self {
        Self {
            c0: self.c0 * b,
            c1: self.c1 * b,
        }
    }

    #[inline]
    fn conjugate(&self) -> Self {
        Self {
            c0: self.c0,
            c1: -self.c1,
        }
    }

    #[inline]
    fn norm(&self) -> B {
        self.c0.square() - B::mul_by_non_residue(self.c1.square())
    }
}
