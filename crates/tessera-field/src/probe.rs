// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Code-generation probes for `cargo xtask check-branchless` (INV-ARCH-2, #242): one loop per
//! operation on secret values, each the loop a kernel inlines the operation into.
//!
//! The gate compiles this crate for x86_64 with the `codegen-probe` feature and reads the
//! assembly of each function: beyond the loop's own entry check and back-edge, a conditional
//! jump in one is a branch on the data. x86_64's code generation turns selects inside loops into
//! such jumps unless they are marked unpredictable, so a loop is where the probe must look. No
//! crate enables the feature; it exists for the gate.

use crate::{Goldilocks, GoldilocksExt2, PrimeField};

/// A loop applying `op` pairwise over `a` and `b` into `out`.
macro_rules! binary {
    ($(#[$doc:meta] $name:ident, $t:ty, $op:tt;)*) => {$(
        #[$doc]
        #[inline(never)]
        pub fn $name(a: &[$t], b: &[$t], out: &mut [$t]) {
            for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
                *o = x $op y;
            }
        }
    )*};
}

binary! {
    /// `Fp` sums.
    fp_add, Goldilocks, +;
    /// `Fp` differences.
    fp_sub, Goldilocks, -;
    /// `Fp` products.
    fp_mul, Goldilocks, *;
    /// `Fp2` sums.
    fp2_add, GoldilocksExt2, +;
    /// `Fp2` differences.
    fp2_sub, GoldilocksExt2, -;
    /// `Fp2` products.
    fp2_mul, GoldilocksExt2, *;
}

/// `Fp` negations.
#[inline(never)]
pub fn fp_neg(a: &[Goldilocks], out: &mut [Goldilocks]) {
    for (o, &x) in out.iter_mut().zip(a) {
        *o = -x;
    }
}

/// Products by the small constant 7, `Fp2`'s non-residue.
#[inline(never)]
pub fn fp_mul_small(a: &[Goldilocks], out: &mut [Goldilocks]) {
    for (o, &x) in out.iter_mut().zip(a) {
        *o = x.mul_small(7);
    }
}

/// Reductions of 64-bit words, as committed values enter the field.
#[inline(never)]
pub fn fp_from_u64(a: &[u64], out: &mut [Goldilocks]) {
    for (o, &x) in out.iter_mut().zip(a) {
        *o = Goldilocks::from_u64_reduced(x);
    }
}

/// Signed embeddings, as the witness engine's integers enter the field.
#[inline(never)]
pub fn fp_from_i64(a: &[i64], out: &mut [Goldilocks]) {
    for (o, &x) in out.iter_mut().zip(a) {
        *o = Goldilocks::from_i64(x);
    }
}

/// Centered readings, as the witness engine reads values back.
#[inline(never)]
pub fn fp_to_centered(a: &[Goldilocks], out: &mut [i128]) {
    for (o, x) in out.iter_mut().zip(a) {
        *o = x.to_centered_i128();
    }
}

/// Radix-2 butterflies, `(a + w·b, a − w·b)`, the loop of an NTT.
#[inline(never)]
pub fn fp_butterfly(
    a: &[Goldilocks],
    b: &[Goldilocks],
    w: &[Goldilocks],
    low: &mut [Goldilocks],
    high: &mut [Goldilocks],
) {
    let pairs = a
        .iter()
        .zip(b)
        .zip(w)
        .zip(low.iter_mut().zip(high.iter_mut()));
    for (((&x, &y), &twiddle), (l, h)) in pairs {
        let t = twiddle * y;
        *l = x + t;
        *h = x - t;
    }
}

/// An `Fp2` inner product, the accumulation of a sumcheck round.
#[inline(never)]
pub fn fp2_dot(a: &[GoldilocksExt2], b: &[GoldilocksExt2]) -> GoldilocksExt2 {
    a.iter()
        .zip(b)
        .fold(GoldilocksExt2::default(), |acc, (&x, &y)| acc + x * y)
}
