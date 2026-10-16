// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Every input-dependent failure is a typed error, never a panic (04 §1): each test violates one
//! condition and asserts the exact error, the checks come in the documented order, and a refused
//! batch leaves the transcript where it was, before `γ`.

mod common;

use common::{Fp, Fp2, Session, SplitMix64};
use tessera_field::Field;
use tessera_hash::Transcript;
use tessera_sumcheck::{Instance, MAX_DEGREE, Prefix, SumcheckError, Table, prove_batch};

/// Proves on a fresh transcript and checks that a refusal left it untouched.
fn refused(degree: usize, instances: &[Instance<'_, Fp2>]) -> SumcheckError {
    let session = Session::random(&mut SplitMix64::new(0x53_0301));
    let mut transcript = session.transcript();
    let error = prove_batch(&mut transcript, Prefix::Stage(0), degree, instances)
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
fn a_degree_outside_1_to_7_is_refused() {
    let table = [Fp::ONE; 4];
    let instances = [Instance::new(vec![Table::base(&table)])];
    for degree in [0, MAX_DEGREE + 1, 64, usize::MAX] {
        assert_eq!(
            refused(degree, &instances),
            SumcheckError::Degree { degree }
        );
        // The degree is checked first, before the instances.
        assert_eq!(refused(degree, &[]), SumcheckError::Degree { degree });
    }
}

#[test]
fn a_batch_without_instances_is_refused() {
    for degree in 1..=MAX_DEGREE {
        assert_eq!(refused(degree, &[]), SumcheckError::NoInstances);
    }
}

#[test]
fn an_instance_without_tables_is_refused() {
    let table = [Fp2::ONE; 8];
    let instances = [
        Instance::new(vec![Table::ext(&table)]),
        Instance::new(vec![]),
        Instance::new(vec![]),
    ];
    assert_eq!(
        refused(2, &instances),
        SumcheckError::NoTables { instance: 1 }
    );
}

#[test]
fn an_instance_with_more_tables_than_the_degree_is_refused() {
    let table = [Fp::ONE; 2];
    for degree in 1..MAX_DEGREE {
        let tables = vec![Table::base(&table); degree + 1];
        let instances = [
            Instance::new(vec![Table::base(&table)]),
            Instance::new(tables),
        ];
        assert_eq!(
            refused(degree, &instances),
            SumcheckError::TooManyTables {
                instance: 1,
                tables: degree + 1,
                degree
            }
        );
    }
    // The count is checked before the lengths.
    let odd = [Fp::ONE; 3];
    let instances = [Instance::new(vec![Table::base(&odd), Table::base(&odd)])];
    assert_eq!(
        refused(1, &instances),
        SumcheckError::TooManyTables {
            instance: 0,
            tables: 2,
            degree: 1
        }
    );
}

#[test]
fn a_table_length_that_is_not_a_power_of_two_is_refused() {
    let good = [Fp::ONE; 4];
    for len in [0, 3, 5, 6, 7, 12, 1000] {
        let bad = vec![Fp2::ONE; len];
        let instances = [
            Instance::new(vec![Table::base(&good)]),
            Instance::new(vec![Table::base(&good), Table::ext(&bad)]),
        ];
        assert_eq!(
            refused(3, &instances),
            SumcheckError::TableLength {
                instance: 1,
                table: 1,
                len
            }
        );
        let first = [Instance::new(vec![Table::ext(&bad), Table::base(&good)])];
        assert_eq!(
            refused(3, &first),
            SumcheckError::TableLength {
                instance: 0,
                table: 0,
                len
            }
        );
    }
}

#[test]
fn tables_of_one_instance_of_different_sizes_are_refused() {
    let four = [Fp::ONE; 4];
    let eight = [Fp2::ONE; 8];
    let one = [Fp::ONE; 1];
    let cases = [
        (vec![Table::base(&four), Table::ext(&eight)], 1, 8, 4),
        (
            vec![Table::ext(&eight), Table::ext(&eight), Table::base(&four)],
            2,
            4,
            8,
        ),
        (
            vec![Table::base(&one), Table::base(&four).public()],
            1,
            4,
            1,
        ),
    ];
    for (tables, table, len, expected) in cases {
        assert_eq!(
            refused(3, &[Instance::new(tables)]),
            SumcheckError::TableSizes {
                instance: 0,
                table,
                len,
                expected
            }
        );
    }
    // Within a table, the length is checked before the size: a table of 3 entries beside one of
    // 4 is refused for its length.
    let three = [Fp::ONE; 3];
    let instances = [Instance::new(vec![Table::base(&four), Table::base(&three)])];
    assert_eq!(
        refused(2, &instances),
        SumcheckError::TableLength {
            instance: 0,
            table: 1,
            len: 3
        }
    );
}

/// The messages name shapes only.
#[test]
fn errors_display_shapes() {
    let cases = [
        (
            SumcheckError::Degree { degree: 9 },
            "a sumcheck has degree 1 to 7, not 9",
        ),
        (
            SumcheckError::NoInstances,
            "a batched sumcheck needs at least one instance",
        ),
        (
            SumcheckError::NoTables { instance: 2 },
            "instance 2 is a product of no table",
        ),
        (
            SumcheckError::TooManyTables {
                instance: 1,
                tables: 4,
                degree: 3,
            },
            "instance 1 is a product of 4 tables, more than the degree 3",
        ),
        (
            SumcheckError::TableLength {
                instance: 0,
                table: 2,
                len: 6,
            },
            "table 2 of instance 0 has 6 entries, not a power of two",
        ),
        (
            SumcheckError::TableSizes {
                instance: 3,
                table: 1,
                len: 8,
                expected: 4,
            },
            "table 1 of instance 3 has 8 entries where its first table has 4",
        ),
    ];
    for (error, message) in cases {
        assert_eq!(error.to_string(), message);
    }
}
