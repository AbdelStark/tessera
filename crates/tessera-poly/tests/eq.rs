// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `eq` tables and `eq` of two points (RFC-0010 'Multilinear extensions'): the weights of a point
//! sum to 1, each entry is its definition, and `eq(a, b)` is the multilinear extension of
//! equality, over Goldilocks and `F65537` and their extensions.

mod common;

use common::{SplitMix64, TestField, affine_in, eq_vertex, vertex};
use tessera_field::{F65537, F65537Ext2, Goldilocks, GoldilocksExt2};
use tessera_poly::{eq, eq_table};

macro_rules! over_every_field {
    ($check:ident) => {
        $check::<GoldilocksExt2>(11);
        $check::<F65537Ext2>(12);
        $check::<Goldilocks>(13);
        $check::<F65537>(14);
    };
}

fn weights_sum_to_one<F: TestField>(seed: u64) {
    let mut rng = SplitMix64::new(seed);
    for n in 0..=16 {
        let point: Vec<F> = rng.elements(n);
        let table = eq_table(&point).expect("a small table");
        assert_eq!(table.len(), 1 << n);
        let sum = table.iter().fold(F::ZERO, |sum, &w| sum + w);
        assert_eq!(sum, F::ONE, "Σ_x eq(r, x) over 2^{n} vertices");
    }
}

#[test]
fn the_eq_weights_of_a_point_sum_to_one() {
    over_every_field!(weights_sum_to_one);
}

fn entries_are_their_definition<F: TestField>(seed: u64) {
    let mut rng = SplitMix64::new(seed);
    for n in 0..=10 {
        let point: Vec<F> = rng.elements(n);
        let table = eq_table(&point).expect("a small table");
        for (x, &entry) in table.iter().enumerate() {
            assert_eq!(entry, eq_vertex(&point, x), "entry {x} of 2^{n}");
        }
    }
}

#[test]
fn each_eq_table_entry_is_the_product_of_its_factors() {
    over_every_field!(entries_are_their_definition);
}

fn eq_of_two_points_is_an_extension_of_equality<F: TestField>(seed: u64) {
    let mut rng = SplitMix64::new(seed);
    // On the hypercube, eq is the equality predicate.
    for n in 0..=4 {
        for x in 0..1 << n {
            for y in 0..1 << n {
                let expected = if x == y { F::ONE } else { F::ZERO };
                assert_eq!(eq(&vertex::<F>(x, n), &vertex(y, n)), Ok(expected));
            }
        }
    }
    for n in 0..=8 {
        let (a, b): (Vec<F>, Vec<F>) = (rng.elements(n), rng.elements(n));
        // eq(a, ·) is multilinear, so its extension at b is Σ_x eq(a, x)·eq(b, x).
        let sum = (0..1 << n).fold(F::ZERO, |sum, x| sum + eq_vertex(&a, x) * eq_vertex(&b, x));
        assert_eq!(eq(&a, &b), Ok(sum), "{n} coordinates");
        // Multilinear in every coordinate of either point.
        for j in 0..n {
            let s = rng.element();
            assert!(affine_in(|p| eq(p, &b).expect("lengths"), &a, j, s));
            assert!(affine_in(|p| eq(&a, p).expect("lengths"), &b, j, s));
        }
    }
}

#[test]
fn eq_of_two_points_is_the_multilinear_extension_of_equality() {
    over_every_field!(eq_of_two_points_is_an_extension_of_equality);
}
