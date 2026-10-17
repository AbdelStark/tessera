// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Dense multilinear tables (RFC-0010 'Multilinear extensions'): evaluation against brute force on
//! random tables up to 2^12 entries, binding against its definition, and the consistency of the
//! two, over Goldilocks and `F65537`, with challenges in their extensions and in the prime fields
//! themselves.

mod common;

use common::{SplitMix64, TestField, brute_force, vertex};
use tessera_codec::SecretBuf;
use tessera_field::{F65537, F65537Ext2, Goldilocks, GoldilocksExt2};
use tessera_poly::{bind_lowest, bind_lowest_in_place, evaluate, evaluate_base, num_vars};

/// Tables up to 2^12 entries, the bound of the acceptance criterion.
const MAX_VARS: usize = 12;

/// Runs a generic check over every field: the two extensions, and the prime fields used as
/// challenge fields, as soundness calibration does.
macro_rules! over_every_field {
    ($check:ident) => {
        $check::<GoldilocksExt2>(1);
        $check::<F65537Ext2>(2);
        $check::<Goldilocks>(3);
        $check::<F65537>(4);
    };
}

fn evaluation_is_the_sum_over_the_hypercube<F: TestField>(seed: u64) {
    let mut rng = SplitMix64::new(seed);
    for n in 0..=MAX_VARS {
        let base: Vec<F::Base> = rng.elements(1 << n);
        let table: Vec<F> = rng.elements(1 << n);
        let point: Vec<F> = rng.elements(n);
        assert_eq!(
            evaluate_base(&base, &point),
            Ok(brute_force(&base, &point)),
            "base table of 2^{n} entries"
        );
        assert_eq!(
            evaluate(&table, &point),
            Ok(brute_force(&table, &point)),
            "table of 2^{n} entries"
        );
    }
}

#[test]
fn evaluation_equals_brute_force_up_to_2_12_entries() {
    over_every_field!(evaluation_is_the_sum_over_the_hypercube);
}

fn evaluation_at_a_vertex_is_the_entry<F: TestField>(seed: u64) {
    let mut rng = SplitMix64::new(seed);
    for n in 0..=6 {
        let base: Vec<F::Base> = rng.elements(1 << n);
        let table: Vec<F> = rng.elements(1 << n);
        for x in 0..1 << n {
            // x_1 is the least significant bit of the index.
            let point = vertex::<F>(x, n);
            assert_eq!(evaluate_base(&base, &point), Ok(F::from(base[x])));
            assert_eq!(evaluate(&table, &point), Ok(table[x]), "index {x} of 2^{n}");
        }
    }
}

#[test]
fn evaluation_at_a_vertex_reads_the_table_lowest_bit_first() {
    over_every_field!(evaluation_at_a_vertex_is_the_entry);
}

fn binding_follows_its_definition<F: TestField>(seed: u64) {
    let mut rng = SplitMix64::new(seed);
    for n in 1..=8 {
        let base: Vec<F::Base> = rng.elements(1 << n);
        let table: Vec<F> = rng.elements(1 << n);
        let r: F = rng.element();
        let bound = bind_lowest(&base, r).expect("a table with a variable");
        let mut in_place: SecretBuf<F> = table.iter().copied().collect();
        bind_lowest_in_place(&mut in_place, r).expect("a table with a variable");
        assert_eq!(bound.len(), 1 << (n - 1));
        assert_eq!(in_place.len(), 1 << (n - 1));
        for k in 0..1 << (n - 1) {
            let (lo, hi) = (F::from(base[2 * k]), F::from(base[2 * k + 1]));
            assert_eq!(bound[k], lo + r * (hi - lo), "pair {k} of 2^{n}");
            let (lo, hi) = (table[2 * k], table[2 * k + 1]);
            assert_eq!(in_place[k], lo + r * (hi - lo), "pair {k} of 2^{n}");
        }
        // x_1 = 0 keeps the even entries and x_1 = 1 the odd ones: the binding is of the lowest
        // variable, in its direction.
        for (x1, parity) in [(F::ZERO, 0), (F::ONE, 1)] {
            let lifted: Vec<F> = base
                .iter()
                .skip(parity)
                .step_by(2)
                .map(|&b| F::from(b))
                .collect();
            assert_eq!(
                bind_lowest(&base, x1).map(|bound| bound.to_vec()),
                Ok(lifted)
            );
            let mut in_place: SecretBuf<F> = table.iter().copied().collect();
            bind_lowest_in_place(&mut in_place, x1).expect("a table with a variable");
            let kept: Vec<F> = table.iter().skip(parity).step_by(2).copied().collect();
            assert_eq!(in_place[..], kept[..]);
        }
    }
}

#[test]
fn binding_maps_each_pair_to_its_line_at_the_challenge() {
    over_every_field!(binding_follows_its_definition);
}

fn binding_then_evaluating_is_evaluating<F: TestField>(seed: u64) {
    let mut rng = SplitMix64::new(seed);
    for n in 1..=MAX_VARS {
        let base: Vec<F::Base> = rng.elements(1 << n);
        let table: Vec<F> = rng.elements(1 << n);
        let point: Vec<F> = rng.elements(n);
        let bound = bind_lowest(&base, point[0]).expect("a table with a variable");
        assert_eq!(evaluate(&bound, &point[1..]), evaluate_base(&base, &point));
        let mut in_place: SecretBuf<F> = table.iter().copied().collect();
        bind_lowest_in_place(&mut in_place, point[0]).expect("a table with a variable");
        assert_eq!(evaluate(&in_place, &point[1..]), evaluate(&table, &point));
        // Round j of a sumcheck binds x_j: n bindings in round order leave f̃(point).
        for &r in &point[1..] {
            bind_lowest_in_place(&mut in_place, r).expect("a table with a variable");
        }
        assert_eq!(in_place.len(), 1);
        assert_eq!(Ok(in_place[0]), evaluate(&table, &point), "2^{n} entries");
    }
}

#[test]
fn binding_the_lowest_variable_then_evaluating_the_rest_is_evaluating() {
    over_every_field!(binding_then_evaluating_is_evaluating);
}

#[test]
fn binding_in_place_keeps_the_allocation() {
    let mut rng = SplitMix64::new(5);
    let mut table: SecretBuf<GoldilocksExt2> = rng
        .elements::<GoldilocksExt2>(1 << 10)
        .into_iter()
        .collect();
    let (start, capacity) = (table.as_ptr(), table.capacity());
    while table.len() > 1 {
        bind_lowest_in_place(&mut table, rng.element()).expect("a table with a variable");
        assert_eq!((table.as_ptr(), table.capacity()), (start, capacity));
    }
}

#[test]
fn a_table_of_two_to_the_n_entries_has_n_variables() {
    for n in 0..usize::BITS as usize {
        assert_eq!(num_vars(1 << n), Ok(n));
    }
}
