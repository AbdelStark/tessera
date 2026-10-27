// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The new refusals of sum-of-products instances and fraction trees are typed errors, never
//! panics (04 §1): each test violates one condition and asserts the exact error, the checks come
//! in the documented order, and a refused batch leaves the transcript where it was, before `γ`.

mod common;

use common::{Fp, Fp2, Session, SplitMix64};
use tessera_field::Field;
use tessera_hash::Transcript;
use tessera_sumcheck::{
    FractionTree, Instance, LeafNumerators, Prefix, Protocol, SumcheckError, Table, Term,
    prove_batch,
};

/// Proves on a fresh transcript and checks that a refusal left it untouched.
fn refused(degree: usize, instances: &[Instance<'_, Fp2>]) -> SumcheckError {
    let session = Session::random(&mut SplitMix64::new(0x60_0301));
    let mut transcript = session.transcript();
    let error = prove_batch(
        &mut transcript,
        Protocol::One,
        Prefix::Tree(0),
        degree,
        instances,
    )
    .expect_err("the batch is refused");
    let fresh: Transcript = session.transcript();
    assert_eq!(
        (transcript.state(), transcript.counter()),
        (fresh.state(), fresh.counter()),
        "a refused batch does not move the transcript"
    );
    error
}

#[test]
fn a_sum_of_no_term_is_refused() {
    let table = [Fp::ONE; 4];
    let instances = [
        Instance::new(vec![Table::base(&table)]),
        Instance::sum_of_products(vec![Table::base(&table)], vec![]),
    ];
    assert_eq!(
        refused(3, &instances),
        SumcheckError::NoTerms { instance: 1 }
    );
    // A sum of products with no table is refused for its tables first.
    let none = [Instance::<Fp2>::sum_of_products(vec![], vec![])];
    assert_eq!(refused(3, &none), SumcheckError::NoTables { instance: 0 });
}

#[test]
fn a_term_of_no_factor_or_of_more_than_the_degree_is_refused() {
    let table = [Fp2::ONE; 8];
    for degree in 1..=Protocol::One.max_degree(Prefix::Tree(0)) {
        let tables = || vec![Table::ext(&table); 2];
        for factors in [0, degree + 1, degree + 4] {
            let terms = vec![
                Term::new(Fp2::ONE, vec![0]),
                Term::new(Fp2::ONE, vec![1; factors]),
            ];
            let instances = [
                Instance::new(vec![Table::ext(&table)]),
                Instance::sum_of_products(tables(), terms),
            ];
            assert_eq!(
                refused(degree, &instances),
                SumcheckError::TermFactors {
                    instance: 1,
                    term: 1,
                    factors,
                    degree
                }
            );
        }
        // More tables than the degree are a product's error, not a sum's.
        let many = vec![Table::ext(&table); degree + 3];
        let instances = [Instance::sum_of_products(
            many,
            vec![Term::new(Fp2::ONE, vec![degree + 2])],
        )];
        let session = Session::random(&mut SplitMix64::new(0x60_0302));
        assert!(
            prove_batch(
                &mut session.transcript(),
                Protocol::One,
                Prefix::Tree(0),
                degree,
                &instances
            )
            .is_ok()
        );
    }
}

#[test]
fn a_factor_that_names_no_table_is_refused() {
    let table = [Fp::ONE; 2];
    let terms = vec![
        Term::new(Fp2::ONE, vec![0, 1]),
        Term::new(Fp2::ONE, vec![2, 1, 3, 5]),
    ];
    let instances = [Instance::sum_of_products(
        vec![
            Table::base(&table),
            Table::base(&table),
            Table::base(&table),
        ],
        terms,
    )];
    assert_eq!(
        refused(4, &instances),
        SumcheckError::TermFactor {
            instance: 0,
            term: 1,
            factor: 3,
            tables: 3
        }
    );
    // The factor count is checked before the factors' names.
    let instances = [Instance::sum_of_products(
        vec![Table::base(&table)],
        vec![Term::new(Fp2::ONE, vec![7, 7, 7])],
    )];
    assert_eq!(
        refused(2, &instances),
        SumcheckError::TermFactors {
            instance: 0,
            term: 0,
            factors: 3,
            degree: 2
        }
    );
    // Terms are checked before the tables' lengths.
    let odd = [Fp::ONE; 3];
    let instances = [Instance::sum_of_products(
        vec![Table::base(&odd)],
        vec![Term::new(Fp2::ONE, vec![1])],
    )];
    assert_eq!(
        refused(2, &instances),
        SumcheckError::TermFactor {
            instance: 0,
            term: 0,
            factor: 1,
            tables: 1
        }
    );
}

/// A sum of products checks its tables' lengths as a product does.
#[test]
fn a_sum_of_tables_of_different_sizes_is_refused() {
    let (four, eight) = ([Fp::ONE; 4], [Fp2::ONE; 8]);
    let instances = [Instance::sum_of_products(
        vec![Table::base(&four), Table::ext(&eight)],
        vec![Term::new(Fp2::ONE, vec![0])],
    )];
    assert_eq!(
        refused(2, &instances),
        SumcheckError::TableSizes {
            instance: 0,
            table: 1,
            len: 8,
            expected: 4
        }
    );
}

#[test]
fn a_fraction_tree_too_tall_to_allocate_is_refused() {
    for height in [60, 63, 64, usize::MAX] {
        let mut called = false;
        let tree = FractionTree::<Fp2>::new(height, LeafNumerators::Unit, |_| {
            called = true;
            Fp2::ONE
        });
        assert_eq!(
            tree.map(|tree| tree.height()),
            Err(SumcheckError::FractionHeight { height })
        );
        assert!(!called, "refused before any leaf is asked for");
    }
}

#[test]
fn base_numerators_not_one_per_leaf_are_refused() {
    let numerators = [Fp::ONE; 6];
    for (height, len) in [(3, 6), (2, 6), (0, 6), (3, 0)] {
        let tree =
            FractionTree::<Fp2>::new(height, LeafNumerators::Base(&numerators[..len]), |_| {
                Fp2::ONE
            });
        assert_eq!(
            tree.map(|tree| tree.height()),
            Err(SumcheckError::FractionNumerators {
                len,
                expected: 1 << height
            })
        );
    }
    // The height is checked first.
    let tree = FractionTree::<Fp2>::new(64, LeafNumerators::Base(&numerators), |_| Fp2::ONE);
    assert_eq!(
        tree.map(|tree| tree.height()),
        Err(SumcheckError::FractionHeight { height: 64 })
    );
}
