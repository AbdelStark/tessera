// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared test support: a seeded generator over every field of `tessera-field`, and the
//! definitions the tests compare the crate against, written from RFC-0010 'Multilinear
//! extensions' without the crate's algorithms.

#![allow(dead_code)] // each test binary uses part of it

use tessera_field::{
    ExtensionBase, F65537, Field, Goldilocks, PrimeField, QuadraticExt, QuadraticExtension,
};
use tessera_poly::PolyField;

/// SplitMix64 (Steele, Lea and Flood, 2014): reproducible inputs, nothing secret.
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    pub fn element<F: Random>(&mut self) -> F {
        F::random(self)
    }

    pub fn elements<F: Random>(&mut self, n: usize) -> Vec<F> {
        (0..n).map(|_| F::random(self)).collect()
    }
}

/// Values the tests draw at random.
pub trait Random: Copy {
    fn random(rng: &mut SplitMix64) -> Self;
}

/// A field the generic tests run over: they draw elements of it and of its prime subfield.
pub trait TestField: Random + PolyField<Base: Random> {}

impl<F: Random + PolyField<Base: Random>> TestField for F {}

impl Random for Goldilocks {
    fn random(rng: &mut SplitMix64) -> Self {
        Self::from_u64_reduced(rng.next_u64())
    }
}

impl Random for F65537 {
    fn random(rng: &mut SplitMix64) -> Self {
        Self::from_u64_reduced(rng.next_u64())
    }
}

impl<B: ExtensionBase + Random> Random for QuadraticExt<B> {
    fn random(rng: &mut SplitMix64) -> Self {
        let c0 = B::random(rng);
        Self::new(c0, B::random(rng))
    }
}

/// The integer `k` as a field element.
pub fn small<F: Field>(k: u64) -> F {
    (0..k).fold(F::ZERO, |x, _| x + F::ONE)
}

/// `eq(r, x)` at the vertex `x` of the hypercube, `x_1` its lowest bit, by its definition: the
/// product of `r_i` where `x_i = 1` and of `1 − r_i` where `x_i = 0`.
pub fn eq_vertex<F: Field>(r: &[F], x: usize) -> F {
    r.iter().enumerate().fold(F::ONE, |product, (i, &ri)| {
        product * if (x >> i) & 1 == 1 { ri } else { F::ONE - ri }
    })
}

/// `Σ_x f[x]·eq(r, x)` by enumeration of the hypercube, with each entry lifted to `F`.
pub fn brute_force<T: Copy, F: Field + From<T>>(table: &[T], r: &[F]) -> F {
    assert_eq!(
        table.len(),
        1 << r.len(),
        "a table of 2^n entries takes n coordinates"
    );
    table.iter().enumerate().fold(F::ZERO, |sum, (x, &value)| {
        sum + F::from(value) * eq_vertex(r, x)
    })
}

/// The point with coordinate `j` replaced by `t`.
pub fn with<F: Copy>(point: &[F], j: usize, t: F) -> Vec<F> {
    let mut point = point.to_vec();
    point[j] = t;
    point
}

/// Whether `f` is affine in coordinate `j` of `point` at the probe `s`:
/// `f(…, s, …) = f(…, 0, …) + s·(f(…, 1, …) − f(…, 0, …))`.
pub fn affine_in<F: Field>(f: impl Fn(&[F]) -> F, point: &[F], j: usize, s: F) -> bool {
    let at = |t| f(&with(point, j, t));
    let (zero, one) = (at(F::ZERO), at(F::ONE));
    at(s) == zero + s * (one - zero)
}

/// The Boolean point `x_1, …, x_n` of the integer `x`, `x_1` its lowest bit.
pub fn vertex<F: Field>(x: usize, n: usize) -> Vec<F> {
    (0..n)
        .map(|i| if (x >> i) & 1 == 1 { F::ONE } else { F::ZERO })
        .collect()
}
