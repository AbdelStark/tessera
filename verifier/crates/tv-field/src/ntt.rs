// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Roots of unity and the number-theoretic transform over the power-of-two subgroups of `Fp^*`,
//! with the coset variants that re-encode combined rows on `7·L` (RFC-0011 'Encoding').
//!
//! Evaluations are in natural order: index `i` holds the value at `ω^i`, or at `shift·ω^i` on a
//! coset. A transform of length `n ≥ 2` allocates `n/2` twiddle factors (8 bytes each) of
//! scratch, 32 MiB at `n = 2^23`.

use core::fmt;
use core::ops::{Add, Mul, Sub};

use crate::{Fp, Fp2};

/// `log2` of the largest power-of-two subgroup of `Fp^*`: `P − 1 = 2^32·(2^32 − 1)`.
pub const TWO_ADICITY: u32 = 32;

/// `7^((P − 1) / 2^log_n)` for `log_n ≤ TWO_ADICITY`. 7 generates `Fp^*`, so this has order
/// exactly `2^log_n`.
fn primitive_root(log_n: u32) -> Fp {
    Fp::SEVEN.pow((Fp::MODULUS - 1) >> log_n)
}

/// A primitive `2^log_n`-th root of unity, `7^((P − 1) / 2^log_n)` (RFC-0002 'Constants'), or
/// `None` when `log_n` exceeds [`TWO_ADICITY`].
pub fn root_of_unity(log_n: u32) -> Option<Fp> {
    (log_n <= TWO_ADICITY).then(|| primitive_root(log_n))
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for crate::Fp {}
    impl Sealed for crate::Fp2 {}
}

/// The values a transform applies to: [`Fp`], and [`Fp2`] coefficient-wise. Twiddles stay in
/// `Fp`.
pub trait NttValue:
    sealed::Sealed + Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Fp, Output = Self>
{
}

impl NttValue for Fp {}
impl NttValue for Fp2 {}

/// Why a transform cannot run. Every transform checks its arguments before touching the
/// values, so on error they are unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NttError {
    /// The length is not a power of two.
    NotPowerOfTwo {
        /// The length given.
        len: usize,
    },
    /// The length is a power of two above `2^TWO_ADICITY`: `Fp^*` has no subgroup that large.
    TooLong {
        /// `log2` of the length given.
        log_len: u32,
    },
    /// A coset shift of zero: `0·L` is not a coset.
    ZeroShift,
    /// Scratch for the twiddle factors could not be reserved. Values are unchanged.
    Memory,
}

impl fmt::Display for NttError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotPowerOfTwo { len } => write!(f, "NTT length {len} is not a power of two"),
            Self::TooLong { log_len } => {
                write!(f, "NTT length 2^{log_len} exceeds 2^{TWO_ADICITY}")
            }
            Self::ZeroShift => write!(f, "coset shift is zero"),
            Self::Memory => f.write_str("cannot reserve NTT twiddle factors"),
        }
    }
}

impl std::error::Error for NttError {}

/// `log2` of a transform length: a power of two no larger than `2^TWO_ADICITY`.
fn log_len(len: usize) -> Result<u32, NttError> {
    if !len.is_power_of_two() {
        return Err(NttError::NotPowerOfTwo { len });
    }
    match len.trailing_zeros() {
        log_n if log_n > TWO_ADICITY => Err(NttError::TooLong { log_len: log_n }),
        log_n => Ok(log_n),
    }
}

/// The radix-2 decimation-in-time DFT at `root`, an element of order `values.len()`: the
/// values in bit-reversed order, then `log2 n` layers of butterflies
/// `(u, v) ↦ (u + w·v, u − w·v)`.
fn transform<T: NttValue>(values: &mut [T], twiddles: &[Fp]) {
    let n = values.len();
    if n < 2 {
        return;
    }
    let log_n = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - log_n);
        if i < j {
            values.swap(i, j);
        }
    }
    let mut half = 1;
    while half < n {
        let stride = n / (2 * half);
        for block in values.chunks_exact_mut(2 * half) {
            let (lo, hi) = block.split_at_mut(half);
            for ((u, v), w) in lo.iter_mut().zip(hi).zip(twiddles.iter().step_by(stride)) {
                let t = *v * *w;
                (*u, *v) = (*u + t, *u - t);
            }
        }
        half *= 2;
    }
}

/// Reserve before any permutation or coset scaling: a memory error leaves values unchanged.
fn twiddles(n: usize, root: Fp) -> Result<Vec<Fp>, NttError> {
    let mut powers = Vec::new();
    powers
        .try_reserve_exact(n / 2)
        .map_err(|_| NttError::Memory)?;
    // Every n/(2·half)-th root power is the twiddle of a layer of width 2·half.
    let mut w = Fp::ONE;
    for _ in 0..n / 2 {
        powers.push(w);
        w *= root;
    }
    Ok(powers)
}

/// The inverse DFT on a subgroup of order `n = 2^log_n`, then `values[i] *= ratio^i`. The
/// inverse DFT is the DFT at `ω^{-1} = ω^{n−1}` scaled by `n^{-1}`; and `n^{-1} = −(P − 1)/n`
/// because `n·(P − 1)/n = P − 1 = −1`.
fn inverse_transform<T: NttValue>(values: &mut [T], log_n: u32, ratio: Fp) -> Result<(), NttError> {
    let powers = twiddles(values.len(), primitive_root(log_n).pow((1 << log_n) - 1))?;
    transform(values, &powers);
    let mut factor = -Fp::reduce((Fp::MODULUS - 1) >> log_n);
    for value in values.iter_mut() {
        *value = *value * factor;
        factor *= ratio;
    }
    Ok(())
}

/// Evaluates `f(X) = Σ_j values[j]·X^j` at `ω^0, …, ω^{n−1}` in place, where `n` is
/// `values.len()` and `ω` is [`root_of_unity`]`(log2 n)`.
pub fn ntt<T: NttValue>(values: &mut [T]) -> Result<(), NttError> {
    let log_n = log_len(values.len())?;
    let powers = twiddles(values.len(), primitive_root(log_n))?;
    transform(values, &powers);
    Ok(())
}

/// The inverse of [`ntt`]: the coefficients of the polynomial of degree below `n` that takes
/// the given values at `ω^0, …, ω^{n−1}`.
pub fn intt<T: NttValue>(values: &mut [T]) -> Result<(), NttError> {
    let log_n = log_len(values.len())?;
    inverse_transform(values, log_n, Fp::ONE)
}

/// Evaluates `f(X) = Σ_j values[j]·X^j` on the coset `shift·L` in place: at
/// `shift·ω^0, …, shift·ω^{n−1}`, with `L` the subgroup of order `n = values.len()`.
pub fn coset_ntt<T: NttValue>(values: &mut [T], shift: Fp) -> Result<(), NttError> {
    let log_n = log_len(values.len())?;
    if shift == Fp::ZERO {
        return Err(NttError::ZeroShift);
    }
    let powers = twiddles(values.len(), primitive_root(log_n))?;
    // f(shift·X) has coefficients values[j]·shift^j.
    let mut power = Fp::ONE;
    for value in values.iter_mut() {
        *value = *value * power;
        power *= shift;
    }
    transform(values, &powers);
    Ok(())
}

/// The inverse of [`coset_ntt`]: the coefficients of the polynomial of degree below `n` that
/// takes the given values at `shift·ω^0, …, shift·ω^{n−1}`.
pub fn coset_intt<T: NttValue>(values: &mut [T], shift: Fp) -> Result<(), NttError> {
    let log_n = log_len(values.len())?;
    let shift_inverse = shift.inverse().ok_or(NttError::ZeroShift)?;
    inverse_transform(values, log_n, shift_inverse)
}

#[cfg(all(test, target_pointer_width = "64"))]
mod tests {
    use super::*;

    /// A zero-sized value, so that a slice longer than `2^TWO_ADICITY` costs no memory.
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Unit;

    impl sealed::Sealed for Unit {}
    impl NttValue for Unit {}

    impl Add for Unit {
        type Output = Self;
        fn add(self, _: Self) -> Self {
            Self
        }
    }

    impl Sub for Unit {
        type Output = Self;
        fn sub(self, _: Self) -> Self {
            Self
        }
    }

    impl Mul<Fp> for Unit {
        type Output = Self;
        fn mul(self, _: Fp) -> Self {
            Self
        }
    }

    #[test]
    fn the_length_limit_is_two_to_the_two_adicity() {
        assert_eq!(log_len(1 << 32), Ok(32));
        // Rejected before any work: no twiddle table, no permutation.
        let mut values = vec![Unit; 1 << 33];
        let expected = Err(NttError::TooLong { log_len: 33 });
        assert_eq!(ntt(&mut values), expected);
        assert_eq!(intt(&mut values), expected);
        assert_eq!(coset_ntt(&mut values, Fp::SEVEN), expected);
        assert_eq!(coset_intt(&mut values, Fp::SEVEN), expected);
    }
}
