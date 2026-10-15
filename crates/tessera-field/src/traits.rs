// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The field traits of RFC-0002 'Traits'. Compiler, sumcheck, lookup and commitment code is
//! generic over them, so the soundness-calibration tests can run the same code over the small
//! test field [`F65537`](crate::F65537).

use core::fmt;
use core::hash::Hash;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A finite field.
///
/// `Debug` is required in addition to the RFC's list so that generic code and tests can report
/// values; a secret value is wrapped in `Secret` before it reaches anything that prints.
pub trait Field:
    Copy
    + Eq
    + Hash
    + fmt::Debug
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
{
    /// The additive identity.
    const ZERO: Self;
    /// The multiplicative identity.
    const ONE: Self;

    /// `self · self`.
    fn square(&self) -> Self;

    /// `self + self`.
    fn double(&self) -> Self;

    /// The multiplicative inverse, or `None` for zero.
    fn inverse(&self) -> Option<Self>;

    /// `self^exp`. Variable-time in `exp`, which must be public (RFC-0002 'Timing').
    fn pow(&self, exp: u64) -> Self;
}

/// A prime field whose modulus fits in 64 bits.
pub trait PrimeField: Field + Ord {
    /// The modulus `P`.
    const MODULUS: u64;
    /// The largest `k` such that `2^k` divides `P − 1`.
    const TWO_ADICITY: u32;

    /// `x mod P`.
    fn from_u64_reduced(x: u64) -> Self;

    /// The signed embedding: `x ≥ 0` maps to `x`, `x < 0` to `P − |x|` (03 §1).
    fn from_i64(x: i64) -> Self;

    /// The element `x`, or `None` when `x ≥ P`: canonical decoding never reduces silently.
    fn from_canonical_u64(x: u64) -> Option<Self>;

    /// The canonical representative in `[0, P)`.
    fn to_canonical_u64(&self) -> u64;

    /// The centered representative: `x` if `x ≤ (P − 1)/2`, else `x − P`. Meaningful only for
    /// values known to be true integers (RFC-0002 'Signed integers').
    fn to_centered_i128(&self) -> i128;

    /// A primitive `2^log_n`-th root of unity.
    ///
    /// # Panics
    ///
    /// If `log_n > TWO_ADICITY`: no such root exists, and asking for one is a programming error.
    fn two_adic_generator(log_n: u32) -> Self;
}

/// `batch_inverse` found a zero at `index`; the slice was left unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZeroElement {
    /// Position of the first zero.
    pub index: usize,
}

impl fmt::Display for ZeroElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cannot invert zero at index {}", self.index)
    }
}

impl std::error::Error for ZeroElement {}

/// Replaces every element of `xs` by its inverse with Montgomery's trick: one inversion and
/// `3(n − 1)` multiplications. On a zero, returns the index of the first one and leaves `xs`
/// unchanged.
pub fn batch_inverse<F: Field>(xs: &mut [F]) -> Result<(), ZeroElement> {
    if let Some(index) = xs.iter().position(|x| *x == F::ZERO) {
        return Err(ZeroElement { index });
    }
    // prefix[i] = x_0 · … · x_{i−1}.
    let mut prefix = Vec::with_capacity(xs.len());
    let mut product = F::ONE;
    for x in xs.iter() {
        prefix.push(product);
        product *= *x;
    }
    // A field has no zero divisors, so a product of non-zero elements is non-zero.
    let mut inverse = product
        .inverse()
        .expect("a product of non-zero field elements is non-zero");
    for (x, before) in xs.iter_mut().zip(prefix).rev() {
        // inverse = (x_0 · … · x_i)^{-1}; it becomes (x_0 · … · x_{i−1})^{-1} for the next step.
        let next = inverse * *x;
        *x = inverse * before;
        inverse = next;
    }
    Ok(())
}
