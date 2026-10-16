// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The fields the crate computes in, each with the prime field it contains (RFC-0002 'Traits').

use tessera_field::{
    ExtensionBase, F65537, Field, Goldilocks, PrimeField, QuadraticExt, QuadraticExtension,
};

/// A field together with its prime subfield [`Base`](PolyField::Base): `GoldilocksExt2` over
/// `Goldilocks`, `F65537Ext2` over `F65537`, and each prime field over itself.
///
/// Tables of committed values hold `Base` elements, challenges and points are elements of `Self`,
/// and the constants of round-polynomial interpolation are integers of `Base`. With a prime field
/// as `Self`, the same generic code runs a sumcheck whose challenges lie in the prime field, as the
/// soundness-calibration tests over `F65537` need (07 §4).
///
/// `tessera-field`'s traits name no prime subfield, so this trait supplies one. It is implemented
/// for every field of `tessera-field`: the two prime fields one by one, since a blanket
/// implementation over `PrimeField` would conflict with the one over `QuadraticExt`.
pub trait PolyField: Field + From<Self::Base> {
    /// The prime subfield.
    type Base: PrimeField;

    /// `self · b` for `b` in the prime subfield: in a quadratic extension, two base
    /// multiplications ([`QuadraticExtension::mul_base`]) instead of a full product.
    fn mul_by_base(&self, b: Self::Base) -> Self;

    /// `self + b` for `b` in the prime subfield: in a quadratic extension, one base addition.
    fn add_base(&self, b: Self::Base) -> Self;
}

macro_rules! prime_fields {
    ($($field:ty),*) => {$(
        impl PolyField for $field {
            type Base = Self;

            #[inline]
            fn mul_by_base(&self, b: Self) -> Self {
                *self * b
            }

            #[inline]
            fn add_base(&self, b: Self) -> Self {
                *self + b
            }
        }
    )*};
}

prime_fields!(Goldilocks, F65537);

impl<B: ExtensionBase> PolyField for QuadraticExt<B> {
    type Base = B;

    #[inline]
    fn mul_by_base(&self, b: B) -> Self {
        QuadraticExtension::mul_base(self, b)
    }

    #[inline]
    fn add_base(&self, b: B) -> Self {
        let [c0, c1] = self.coeffs();
        Self::new(c0 + b, c1)
    }
}
