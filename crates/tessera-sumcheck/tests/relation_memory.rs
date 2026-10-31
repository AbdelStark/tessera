// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Real relation builders are the allocation oracle; inputs are outside the meter.
use tessera_field::{Field, Goldilocks as G, GoldilocksExt2 as F};
use tessera_sumcheck::{
    Elementwise, ElementwiseShape, ElementwiseTables, Input, Instance, MatMulAct, MatMulActShape,
    MatMulActTables, MatMulWeight, MatMulWeightShape, MatMulWeightTables, Recomposition,
    RecompositionTables, RelationMemory, RightLayout, Rotate, RotateTables, Selected, Selection,
    SelectionTables, Term, WindowError,
};
trait Fixture {
    fn instance(&self) -> Instance<'_, F>;
}
impl Fixture for MatMulActTables<'_, F> {
    fn instance(&self) -> Instance<'_, F> {
        self.instance()
    }
}
impl Fixture for MatMulWeightTables<'_, F> {
    fn instance(&self) -> Instance<'_, F> {
        self.instance()
    }
}
impl Fixture for ElementwiseTables<'_, F> {
    fn instance(&self) -> Instance<'_, F> {
        self.instance()
    }
}
impl Fixture for RotateTables<'_, F> {
    fn instance(&self) -> Instance<'_, F> {
        self.instance()
    }
}
impl Fixture for SelectionTables<F> {
    fn instance(&self) -> Instance<'_, F> {
        self.instance()
    }
}
impl Fixture for RecompositionTables<F> {
    fn instance(&self) -> Instance<'_, F> {
        self.instance()
    }
}
fn check<T: Fixture>(memory: RelationMemory, build: impl FnOnce() -> T) {
    let mut held = None;
    let measured = allocation_counter::measure(|| {
        held = Some(build());
    });
    assert!(
        measured.bytes_max <= memory.construction_peak_bytes(),
        "constructor {measured:?}, {memory:?}"
    );
    assert!(
        u64::try_from(measured.bytes_current).unwrap() <= memory.retained_bytes(),
        "tables {measured:?}, {memory:?}"
    );
    let tables = held.unwrap();
    let mut instance = None;
    let measured = allocation_counter::measure(|| {
        instance = Some(tables.instance());
    });
    assert!(
        measured.bytes_max <= memory.instance_bytes(),
        "instance {measured:?}, {memory:?}"
    );
}
#[test]
fn both_matrix_layouts_and_weight_contractions_fit() {
    for stack_vars in [0, 2] {
        for row_vars in [0, 3] {
            for inner_vars in [0, 5] {
                for col_vars in [0, 2] {
                    let shape = MatMulActShape {
                        stack_vars,
                        row_vars,
                        inner_vars,
                        col_vars,
                    };
                    let a = vec![G::ONE; 1 << (stack_vars + row_vars + inner_vars)];
                    let b = vec![G::ONE; 1 << (stack_vars + col_vars + inner_vars)];
                    let point = vec![F::ONE; stack_vars + row_vars + col_vars];
                    let memory = RelationMemory::matmul::<F>(shape).unwrap();
                    for layout in [RightLayout::Direct, RightLayout::Transposed] {
                        check(memory, || {
                            MatMulAct::new(&a, &b, layout, shape)
                                .unwrap()
                                .tables(&point)
                                .unwrap()
                        });
                    }
                    let shape = MatMulWeightShape {
                        layer_vars: stack_vars,
                        token_vars: row_vars,
                        out_vars: col_vars,
                        in_vars: inner_vars,
                    };
                    check(memory, || {
                        MatMulWeight::new(&b, &a, shape)
                            .unwrap()
                            .tables(&point)
                            .unwrap()
                    });
                }
            }
        }
    }
}
#[test]
fn elementwise_clones_expansions_eq_and_terms_fit() {
    for vars in [0usize, 1, 5, 11] {
        for summed_vars in [0, vars / 2, vars] {
            let a = vec![G::ONE; 1 << vars];
            let scalar = [G::ONE];
            let point = vec![F::ONE; vars - summed_vars];
            let shape = ElementwiseShape { vars, summed_vars };
            let count = 3 + usize::from(vars > 0);
            let memory =
                RelationMemory::elementwise::<F>(shape, count, 1 + usize::from(vars > 0), &[2, 3])
                    .unwrap();
            check(memory, || {
                let mut inputs = vec![
                    Input::Operand(&a),
                    Input::Broadcast {
                        operand: &scalar,
                        low_vars: 0,
                    },
                    Input::Public(&a),
                ];
                if vars > 0 {
                    inputs.push(Input::Partner {
                        operand: &a,
                        bit: 0,
                    });
                }
                Elementwise::new(
                    shape,
                    inputs,
                    vec![
                        Term::new(F::ONE, vec![0, 1]),
                        Term::new(F::ONE, vec![0, 0, 2]),
                    ],
                )
                .unwrap()
                .tables(&point)
                .unwrap()
            });
        }
    }
}
#[test]
fn rotation_and_selection_kernels_and_contraction_scratch_fit() {
    for vars in [1usize, 3, 11] {
        for bit in [0, vars - 1] {
            let x = vec![G::ONE; 1 << vars];
            let trig = vec![G::ONE; 1 << (vars - 1)];
            let point = vec![F::ONE; vars];
            check(RelationMemory::rotate::<F>(vars).unwrap(), || {
                Rotate::new(&x, &trig, &trig, bit, vars)
                    .unwrap()
                    .tables(&point)
                    .unwrap()
            });
        }
    }
    for (low, axis, high, rows) in [(0usize, 0, 0, 0), (1, 3, 2, 2), (3, 1, 0, 5)] {
        for members in [0usize, 1, 5] {
            let x = vec![G::ONE; 1 << (low + axis + high)];
            let point = vec![F::ONE; low + rows + high];
            let indices = (0..1usize << rows)
                .map(|i| i % (1usize << axis))
                .collect::<Vec<_>>();
            let kernel = (0..members)
                .map(|_| Selected {
                    coefficient: F::ONE,
                    index: &indices,
                })
                .collect::<Vec<_>>();
            check(
                RelationMemory::selection::<F>(low, axis, high, rows, members).unwrap(),
                || {
                    Selection::new(&x, low, axis, high)
                        .unwrap()
                        .tables(&point, rows, &kernel)
                        .unwrap()
                },
            );
        }
    }
}
#[test]
fn ragged_bit_recomposition_and_public_x_clone_overlap_fit() {
    let bits = |n: u8| u32::from(n).next_power_of_two().trailing_zeros() as usize;
    for (low, high) in [(1u8, 0u8), (3, 2), (2, 5), (31, 31)] {
        for rows in [0usize, 4] {
            for public in [false, true] {
                let x = vec![G::ONE; 1 << rows];
                let b = vec![G::ONE; 1 << (rows + bits(low))];
                let h = (high > 0).then(|| vec![G::ONE; 1 << (rows + bits(high))]);
                let point = vec![F::ONE; rows];
                check(
                    RelationMemory::recomposition::<F>(low, high, rows).unwrap(),
                    || {
                        Recomposition::new(
                            if public {
                                Input::Public(&x)
                            } else {
                                Input::Operand(&x)
                            },
                            &b,
                            h.as_deref(),
                            low,
                            high,
                            rows,
                        )
                        .unwrap()
                        .tables(&point)
                        .unwrap()
                    },
                );
            }
        }
    }
}
#[test]
fn invalid_relation_geometry_and_address_limits_fail_without_allocation() {
    let measured = allocation_counter::measure(|| {
        let shape = WindowError::Shape("invalid canonical relation memory geometry");
        assert_eq!(RelationMemory::rotate::<F>(0), Err(shape));
        assert_eq!(
            RelationMemory::elementwise::<F>(
                ElementwiseShape {
                    vars: 1,
                    summed_vars: 2
                },
                1,
                0,
                &[1]
            ),
            Err(shape)
        );
        assert_eq!(
            RelationMemory::elementwise::<F>(
                ElementwiseShape {
                    vars: 1,
                    summed_vars: 0
                },
                0,
                0,
                &[1]
            ),
            Err(shape)
        );
        assert_eq!(
            RelationMemory::elementwise::<F>(
                ElementwiseShape {
                    vars: 1,
                    summed_vars: 0
                },
                1,
                2,
                &[1]
            ),
            Err(shape)
        );
        assert_eq!(
            RelationMemory::elementwise::<F>(
                ElementwiseShape {
                    vars: 1,
                    summed_vars: 0
                },
                1,
                0,
                &[]
            ),
            Err(shape)
        );
        assert_eq!(RelationMemory::recomposition::<F>(0, 1, 0), Err(shape));
        assert_eq!(RelationMemory::recomposition::<F>(32, 31, 0), Err(shape));
        assert_eq!(
            RelationMemory::rotate::<F>(usize::BITS as usize),
            Err(shape)
        );
        assert_eq!(
            RelationMemory::rotate::<F>(usize::BITS as usize - 2),
            Err(WindowError::Allocation)
        );
        assert!(RelationMemory::matmul::<F>(MatMulActShape::default()).is_ok());
    });
    assert_eq!(measured.count_total, 0);
}
