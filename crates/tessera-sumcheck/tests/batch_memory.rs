// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Independent invalid public metadata cases for the ordinary storage planner.
use allocation_counter::measure;
use tessera_field::GoldilocksExt2 as F;
use tessera_sumcheck::{BatchMemory, BatchShape, MAX_DEGREE, WindowError};

#[test]
fn each_invalid_batch_shape_is_refused_without_allocating() {
    let literal = BatchShape {
        vars: 3,
        tables: 2,
        terms: 1,
        evaluations: 1,
        computed: false,
    };
    let computed = BatchShape {
        tables: 0,
        terms: 0,
        computed: true,
        ..literal
    };
    let invalid = [
        BatchShape {
            vars: usize::BITS as usize,
            ..literal
        },
        BatchShape {
            tables: 0,
            evaluations: 0,
            ..literal
        },
        BatchShape {
            evaluations: 3,
            ..literal
        },
        BatchShape {
            tables: 4,
            terms: 0,
            ..literal
        },
        BatchShape {
            tables: 1,
            ..computed
        },
        BatchShape {
            terms: 1,
            ..computed
        },
        BatchShape {
            vars: 63,
            ..computed
        },
    ];
    let measured = measure(|| {
        for degree in [0, MAX_DEGREE + 1] {
            assert_eq!(
                BatchMemory::new::<F>(degree, &[literal]),
                Err(WindowError::Shape(
                    "a nonempty batch needs a supported degree"
                ))
            );
        }
        assert_eq!(
            BatchMemory::new::<F>(3, &[]),
            Err(WindowError::Shape(
                "a nonempty batch needs a supported degree"
            ))
        );
        for shape in invalid {
            assert_eq!(
                BatchMemory::new::<F>(3, &[shape]),
                Err(WindowError::Shape(
                    "batch counts cover the canonical instance"
                ))
            );
        }
        assert!(BatchMemory::new::<F>(3, &[literal, computed]).is_ok());
    });
    assert_eq!(measured.count_total, 0);
}

#[test]
fn unaddressable_declared_counts_are_refused_without_allocating() {
    let literal = BatchShape {
        vars: 1,
        tables: 1,
        terms: 1,
        evaluations: 1,
        computed: false,
    };
    let computed = BatchShape {
        tables: 0,
        terms: 0,
        computed: true,
        ..literal
    };
    let measured = measure(|| {
        for shape in [
            BatchShape {
                tables: usize::MAX,
                ..literal
            },
            BatchShape {
                terms: usize::MAX,
                ..literal
            },
            BatchShape {
                evaluations: usize::MAX,
                ..computed
            },
        ] {
            assert_eq!(
                BatchMemory::new::<F>(3, &[shape]),
                Err(WindowError::Allocation)
            );
        }
    });
    assert_eq!(measured.count_total, 0);
}
