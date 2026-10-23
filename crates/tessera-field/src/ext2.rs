// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The quadratic extension `Fp2 = Fp[X]/(X^2 − W)` of RFC-0002 'Quadratic extension', in which
//! every challenge lives: about 2^128 elements for Goldilocks (`W = 7`).

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use tessera_codec::Zeroize;

use crate::goldilocks::Wide192;
use crate::{F65537, Field, Goldilocks, PrimeField};

/// A prime field with a fixed quadratic non-residue `W`, over which [`QuadraticExt`] is built.
pub trait ExtensionBase: PrimeField {
    /// The non-residue `W`: `X^2 − W` is irreducible.
    const NON_RESIDUE: Self;

    /// A sum of products, reduced once when the sum is read (#391): a wide integer where the
    /// field is large, the element itself where reducing each product costs nothing worth saving.
    type Wide: Copy + Default + Zeroize;

    /// `log2` of the most products one [`Self::Wide`] sums before its bound would fail.
    const WIDE_LOG_TERMS: u32;

    /// `W · x`. Fields override this when multiplication by `W` has a cheaper form.
    #[inline]
    fn mul_by_non_residue(x: Self) -> Self {
        Self::NON_RESIDUE * x
    }

    /// Adds `a·b` to `wide`.
    fn wide_add_product(wide: &mut Self::Wide, a: Self, b: Self);

    /// The sum `wide` holds, reduced.
    fn wide_reduce(wide: &Self::Wide) -> Self;
}

impl ExtensionBase for Goldilocks {
    /// 7, the smallest quadratic non-residue modulo the Goldilocks prime (RFC-0002 'Constants').
    const NON_RESIDUE: Self = Goldilocks::MULTIPLICATIVE_GENERATOR;

    type Wide = Wide192;

    const WIDE_LOG_TERMS: u32 = Wide192::LOG_TERMS;

    #[inline]
    fn mul_by_non_residue(x: Self) -> Self {
        x.mul_small(7)
    }

    #[inline(always)]
    fn wide_add_product(wide: &mut Wide192, a: Self, b: Self) {
        wide.add_product(a, b);
    }

    #[inline(always)]
    fn wide_reduce(wide: &Wide192) -> Self {
        wide.reduce()
    }
}

impl ExtensionBase for F65537 {
    /// 3, a generator of `F65537^*` and hence a non-residue.
    const NON_RESIDUE: Self = F65537::MULTIPLICATIVE_GENERATOR;

    /// The test field reduces every product: its sums serve calibration, not speed.
    type Wide = F65537;

    const WIDE_LOG_TERMS: u32 = 64;

    #[inline]
    fn wide_add_product(wide: &mut F65537, a: Self, b: Self) {
        *wide += a * b;
    }

    #[inline]
    fn wide_reduce(wide: &F65537) -> Self {
        *wide
    }
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

/// A sum of products in the quadratic extension, reduced once (#391). Each product
/// `(a_0 + a_1·X)(b_0 + b_1·X)` is summed schoolbook as its four partial products,
/// `c_0 += a_0·b_0`, `c_1 += a_0·b_1 + a_1·b_0` and `c_W += a_1·b_1`, every one a wide sum of the
/// base field; the reduction is `(c_0 + W·c_W) + c_1·X`. No partial product is reduced, and no
/// Karatsuba pre-sum widens an operand.
pub struct WideExt2<B: ExtensionBase> {
    c0: B::Wide,
    c1: B::Wide,
    cw: B::Wide,
}

impl<B: ExtensionBase> Clone for WideExt2<B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B: ExtensionBase> Copy for WideExt2<B> {}

impl<B: ExtensionBase> Default for WideExt2<B> {
    fn default() -> Self {
        Self {
            c0: B::Wide::default(),
            c1: B::Wide::default(),
            cw: B::Wide::default(),
        }
    }
}

impl<B: ExtensionBase> WideExt2<B> {
    /// `log2` of the most products one sum holds: `c_1` takes two partial products per product.
    pub const LOG_TERMS: u32 = B::WIDE_LOG_TERMS - 1;

    /// Adds `a·b`.
    #[inline(always)]
    pub fn add_product(&mut self, a: QuadraticExt<B>, b: QuadraticExt<B>) {
        B::wide_add_product(&mut self.c0, a.c0, b.c0);
        B::wide_add_product(&mut self.c1, a.c0, b.c1);
        B::wide_add_product(&mut self.c1, a.c1, b.c0);
        B::wide_add_product(&mut self.cw, a.c1, b.c1);
    }

    /// Adds `a·b` for `b` in the base field: two partial products.
    #[inline(always)]
    pub fn add_base_product(&mut self, a: QuadraticExt<B>, b: B) {
        B::wide_add_product(&mut self.c0, a.c0, b);
        B::wide_add_product(&mut self.c1, a.c1, b);
    }

    /// The sum, reduced.
    #[inline(always)]
    pub fn reduce(&self) -> QuadraticExt<B> {
        QuadraticExt {
            c0: B::wide_reduce(&self.c0) + B::mul_by_non_residue(B::wide_reduce(&self.cw)),
            c1: B::wide_reduce(&self.c1),
        }
    }
}

/// Clears the three partial sums (05 §3 OBS-6).
impl<B: ExtensionBase> Zeroize for WideExt2<B> {
    fn zeroize(&mut self) {
        self.c0.zeroize();
        self.c1.zeroize();
        self.cw.zeroize();
    }
}

impl<B: ExtensionBase> fmt::Debug for WideExt2<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("WideExt2")
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

#[cfg(test)]
mod wide_tests {
    //! Schoolbook extension sums against the field's own multiplication (#391).
    use super::*;

    fn elements() -> Vec<GoldilocksExt2> {
        let edges = [
            0,
            1,
            2,
            7,
            0xFFFF_FFFF,
            1 << 32,
            1 << 63,
            0xFFFF_FFFF_0000_0000,
        ];
        let mut out: Vec<GoldilocksExt2> = edges
            .iter()
            .flat_map(|&a| edges.iter().map(move |&b| (a, b)))
            .map(|(a, b)| {
                QuadraticExt::new(
                    Goldilocks::from_u64_reduced(a),
                    Goldilocks::from_u64_reduced(b),
                )
            })
            .collect();
        let mut x = 0x7E55_E7A0_0391_0008u64;
        for _ in 0..512 {
            let mut next = || {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                Goldilocks::from_u64_reduced(x)
            };
            out.push(QuadraticExt::new(next(), next()));
        }
        out
    }

    #[test]
    fn extension_sums_equal_their_reduced_products() {
        let xs = elements();
        let (mut wide, mut base) = (WideExt2::<Goldilocks>::default(), WideExt2::default());
        let (mut sum, mut base_sum) = (GoldilocksExt2::ZERO, GoldilocksExt2::ZERO);
        for (i, &a) in xs.iter().enumerate() {
            let b = xs[(7 * i + 3) % xs.len()];
            wide.add_product(a, b);
            sum += a * b;
            let c = b.coeffs()[1];
            base.add_base_product(a, c);
            base_sum += a.mul_base(c);
            assert_eq!(wide.reduce(), sum);
            assert_eq!(base.reduce(), base_sum);
        }
        assert_eq!(WideExt2::<Goldilocks>::LOG_TERMS, 63);
    }

    #[test]
    fn the_test_field_reduces_every_product() {
        let (a, b) = (
            F65537Ext2::new(
                F65537::from_u64_reduced(5),
                F65537::from_u64_reduced(65_536),
            ),
            F65537Ext2::new(F65537::from_u64_reduced(3), F65537::from_u64_reduced(2)),
        );
        let mut wide = WideExt2::<F65537>::default();
        wide.add_product(a, b);
        wide.add_product(b, a);
        assert_eq!(wide.reduce(), a * b + b * a);
    }
}
