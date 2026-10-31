// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The oracle is the unchanged literal engine with independently materialized full EQ tables.

use tessera_codec::Ascii;
use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_poly::evaluate;

use super::*;
use crate::{Instance, Table, prove_batch};

type F = GoldilocksExt2;

#[test]
fn one_round_windows_and_supplied_gamma_preserve_the_literal_transcript() {
    for width in [1, 2, MAX_WINDOW_VARS] {
        let (mut source, relations) = specs(&[(7, 2, true, 2), (3, 1, true, 1), (0, 1, false, 0)]);
        let eqs = eq_tables(&relations);
        let mut literal_transcript = Transcript::new(&context());
        let expected = prove_batch(
            &mut literal_transcript,
            Protocol::One,
            Prefix::Tree(5),
            3,
            &literal(&source, &relations, &eqs),
        )
        .unwrap();
        let prepared =
            PreparedWindow::with_window_vars(&mut source, Protocol::One, 3, relations, 4, width)
                .unwrap();
        let mut actual_transcript = Transcript::new(&context());
        // A tree caller owns gamma, then lambda, before it builds dependent relations.
        // This API must never squeeze gamma twice.
        let gamma = actual_transcript.challenge_fp2("tree/5/gamma");
        let actual = prepared
            .prove_with_gamma(&mut actual_transcript, Prefix::Tree(5), &mut source, gamma)
            .unwrap();
        assert_eq!(actual.proof().rounds, expected.proof().rounds);
        assert_eq!(actual.proof().final_evals, expected.proof().final_evals);
        assert_eq!(actual.point(), expected.point());
        assert_eq!(
            actual_transcript.challenge_fp2("after"),
            literal_transcript.challenge_fp2("after")
        );
        let folded: Vec<_> = source
            .passes
            .iter()
            .filter_map(|p| match p {
                WindowPass::Fold(at) => Some(*at),
                WindowPass::Prepare => None,
            })
            .collect();
        let expected_folds: Vec<_> = (width..4).step_by(width).chain([4]).collect();
        assert_eq!(folded, expected_folds);
        let shapes = [WindowShape {
            vars: 7,
            tables: 3,
            delta: 2,
            windowable: true,
        }];
        let charged = WindowMemory::with_window_vars(&shapes, 4, size_of::<F>(), width).unwrap();
        assert_eq!(charged.fold_passes, 4usize.div_ceil(width));
    }
}

#[test]
fn invalid_round_window_width_is_refused_before_a_source_pass() {
    for width in [0, MAX_WINDOW_VARS + 1, usize::MAX] {
        let (mut source, relations) = specs(&[(3, 2, true, 2)]);
        let result =
            PreparedWindow::with_window_vars(&mut source, Protocol::One, 3, relations, 1, width);
        assert!(matches!(
            result,
            Err(WindowError::Shape(
                "a grid covers one through eight own variables"
            ))
        ));
        assert!(source.passes.is_empty());
        assert_eq!(
            WindowMemory::with_window_vars(
                &[WindowShape {
                    vars: 3,
                    tables: 3,
                    delta: 2,
                    windowable: true
                }],
                1,
                size_of::<F>(),
                width
            ),
            Err(WindowError::Shape(
                "a grid covers one through eight own variables"
            ))
        );
    }
}

fn fp(n: u64) -> F {
    F::from(Goldilocks::from_u64_reduced(n))
}

fn context() -> SessionContext {
    SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::Live,
        params: ParamSetId(Ascii::new("test-small/1").unwrap()),
        statement_digest: [45; 32],
        session_id: [59; 16],
    }
}

struct Source {
    tables: Vec<Vec<F>>,
    passes: Vec<WindowPass>,
}

impl WindowSource<F> for Source {
    fn regenerate(&mut self, pass: WindowPass) -> Result<(), WindowError> {
        self.passes.push(pass);
        Ok(())
    }

    fn read(&self, id: usize, start: usize, out: &mut [F]) -> Result<(), WindowError> {
        let input = self
            .tables
            .get(id)
            .and_then(|t| t.get(start..start + out.len()))
            .ok_or(WindowError::Source)?;
        out.copy_from_slice(input);
        Ok(())
    }
}

fn specs(shapes: &[(usize, usize, bool, usize)]) -> (Source, Vec<WindowInstance<F>>) {
    let mut source = Source {
        tables: Vec::new(),
        passes: Vec::new(),
    };
    let specs = shapes
        .iter()
        .map(|&(vars, factors, windowed, eq_kind)| {
            // An unused private table still owes a final evaluation; public operands are interleaved.
            let tables = (0..factors + 1)
                .map(|k| {
                    let id = source.tables.len();
                    source.tables.push(
                        (0..1 << vars)
                            .map(|index| {
                                let edges = [
                                    0,
                                    1,
                                    u64::MAX - (1 << 32),
                                    (1 << 32) - 1,
                                    1 << 32,
                                    u64::MAX - 1,
                                ];
                                F::new(
                                    Goldilocks::from_u64_reduced(edges[(index + k) % edges.len()]),
                                    Goldilocks::from_u64_reduced((3 * index + k + 1) as u64),
                                )
                            })
                            .collect(),
                    );
                    WindowTable { id, public: k == 1 }
                })
                .collect();
            let terms = vec![
                Term::new(fp(7), (0..factors).collect()),
                Term::new(fp(11), vec![0; factors]),
            ];
            let eq = (0..vars)
                .map(|j| {
                    if eq_kind == 0 || eq_kind == 1 && j % 2 == 0 {
                        None
                    } else {
                        Some(F::new(
                            Goldilocks::from_u64_reduced((j + 3) as u64),
                            Goldilocks::from_u64_reduced((2 * j + 1) as u64),
                        ))
                    }
                })
                .collect();
            WindowInstance {
                vars,
                tables,
                terms,
                eq,
                windowed,
            }
        })
        .collect();
    (source, specs)
}

fn literal<'a>(
    source: &'a Source,
    specs: &'a [WindowInstance<F>],
    eqs: &'a [Vec<F>],
) -> Vec<Instance<'a, F>> {
    specs
        .iter()
        .zip(eqs)
        .map(|(s, eq)| {
            let mut tables: Vec<_> = s
                .tables
                .iter()
                .map(|table| {
                    let t = Table::ext(&source.tables[table.id]);
                    if table.public { t.public() } else { t }
                })
                .collect();
            let at = tables.len();
            tables.push(Table::ext(eq).public());
            let terms = s
                .terms
                .iter()
                .map(|term| {
                    let mut factors = term.factors().to_vec();
                    if s.eq.iter().any(Option::is_some) {
                        factors.push(at);
                    }
                    Term::new(*term.weight(), factors)
                })
                .collect();
            Instance::sum_of_products(tables, terms)
        })
        .collect()
}

fn eq_tables(specs: &[WindowInstance<F>]) -> Vec<Vec<F>> {
    specs
        .iter()
        .map(|s| {
            (0..1 << s.vars)
                .map(|x| {
                    // Direct definition, no grid or split-weight helpers from the optimized path.
                    s.eq.iter().enumerate().fold(F::ONE, |v, (bit, z)| {
                        v * z.map_or(F::ONE, |z| if x >> bit & 1 == 0 { F::ONE - z } else { z })
                    })
                })
                .collect()
        })
        .collect()
}

fn compare(shapes: &[(usize, usize, bool, usize)], cut: usize) {
    let (mut source, specs) = specs(shapes);
    let eqs = eq_tables(&specs);
    let degree = shapes
        .iter()
        .map(|&(n, d, _, eq)| d + usize::from(eq != 0 && n != 0))
        .max()
        .unwrap();
    let mut a = Transcript::new(&context());
    let reference = prove_batch(
        &mut a,
        Protocol::One,
        Prefix::Stage(3),
        degree,
        &literal(&source, &specs, &eqs),
    )
    .unwrap();
    let prepared =
        PreparedWindow::new(&mut source, Protocol::One, degree, specs.clone(), cut).unwrap();
    for (index, spec) in specs.iter().enumerate() {
        let expected = (0..1 << spec.vars).fold(F::ZERO, |sum, x| {
            sum + eqs[index][x]
                * spec.terms.iter().fold(F::ZERO, |sum, term| {
                    sum + term.factors().iter().fold(*term.weight(), |p, &k| {
                        p * source.tables[spec.tables[k].id][x]
                    })
                })
        });
        assert_eq!(prepared.sum(index), Some(expected));
    }
    let mut b = Transcript::new(&context());
    let actual = prepared
        .prove(&mut b, Prefix::Stage(3), &mut source)
        .unwrap();
    assert_eq!(
        actual.proof().rounds,
        reference.proof().rounds,
        "rounds at cut {cut}"
    );
    assert_eq!(
        actual.proof().final_evals,
        reference.proof().final_evals,
        "evaluations at cut {cut}"
    );
    assert_eq!(actual.point(), reference.point());
    assert_eq!(a.challenge_fp2("after"), b.challenge_fp2("after"));
    for (index, spec) in specs.iter().enumerate() {
        let point = actual.instance_point(index).unwrap();
        let mut evals = actual.instance_evals(index).unwrap().iter();
        for table in &spec.tables {
            if !table.public {
                assert_eq!(
                    *evals.next().unwrap(),
                    evaluate(&source.tables[table.id], point).unwrap()
                );
            }
        }
    }
    let padding = specs
        .iter()
        .filter(|s| s.windowed && cut > actual.point().len() - s.vars)
        .map(|s| actual.point().len() - s.vars)
        .min();
    let folds: Vec<_> = padding.map_or_else(Vec::new, |p| {
        (p + MAX_WINDOW_VARS..cut)
            .step_by(MAX_WINDOW_VARS)
            .chain([cut])
            .map(WindowPass::Fold)
            .collect()
    });
    assert_eq!(source.passes, [vec![WindowPass::Prepare], folds].concat());
}

#[test]
fn every_cut_product_degree_eq_subset_and_front_padding_matches_literal() {
    for factors in 1..=3 {
        for eq in 0..=2 {
            for small in 0..=5 {
                for cut in 0..=4 {
                    compare(
                        &[
                            (5, factors, true, eq),
                            (small, factors, true, eq),
                            (3, 2, false, 2),
                        ],
                        cut,
                    );
                }
            }
        }
    }
}

#[test]
fn multiple_windows_include_short_instances_and_resident_companions() {
    for cut in [7, 8, 9, 10] {
        compare(
            &[
                (11, 2, true, 2),
                (8, 3, true, 1),
                (2, 2, true, 2),
                (4, 3, false, 2),
            ],
            cut,
        );
    }
}

#[test]
fn scalar_batch_has_no_round_or_fold_pass() {
    compare(&[(0, 1, true, 0)], 0);
}

#[test]
fn invalid_shapes_are_refused_before_a_source_pass() {
    let (mut source, valid) = specs(&[(4, 2, true, 2)]);
    for (degree, cut, change, expected) in [
        (0, 1, 0, "a nonempty batch needs degree 1 through 7"),
        (8, 1, 0, "a nonempty batch needs degree 1 through 7"),
        (3, 4, 0, "the cut must leave the final round resident"),
        (
            3,
            1,
            1,
            "a relation needs tables, terms and one EQ option per variable",
        ),
        (2, 1, 0, "a term must fit the table list and degree"),
        (3, 1, 2, "a term must fit the table list and degree"),
        (3, 1, 3, "a term must fit the table list and degree"),
    ] {
        let mut specs = valid.clone();
        match change {
            1 => {
                specs[0].eq.pop();
            }
            2 => specs[0].terms[0] = Term::new(fp(1), vec![9]),
            3 => specs[0].terms[0] = Term::new(fp(1), vec![]),
            _ => {}
        }
        assert!(
            matches!(PreparedWindow::new(&mut source, Protocol::One, degree, specs, cut), Err(WindowError::Shape(rule)) if rule == expected)
        );
        assert!(source.passes.is_empty());
    }
    assert!(matches!(
        PreparedWindow::new(&mut source, Protocol::Two, 3, valid, 1),
        Err(WindowError::Shape("masked windowing is not implemented"))
    ));
    assert!(source.passes.is_empty());
}

#[test]
fn lagrange_basis_handles_all_nodes_and_zero_one_challenges() {
    for radix in 2..=8 {
        for node in 0..radix {
            let weights = basis::<F>(radix, fp(node as u64)).unwrap();
            assert_eq!(
                weights,
                (0..radix)
                    .map(|i| if i == node { F::ONE } else { F::ZERO })
                    .collect::<Vec<_>>()
            );
        }
    }
}

fn tv_value(value: F) -> tv_field::Fp2 {
    let [a, b] = value.coeffs();
    tv_field::Fp2::new(
        tv_field::Fp::from_canonical(a.to_canonical_u64()).unwrap(),
        tv_field::Fp::from_canonical(b.to_canonical_u64()).unwrap(),
    )
}

fn independent_verify(
    corrupt: bool,
) -> Result<tv_core::sumcheck::Batch, tv_core::sumcheck::SumcheckError> {
    use tv_core::sumcheck as tv;
    use tv_hash::transcript as th;
    let (mut source, specs) = specs(&[(5, 2, true, 2), (3, 2, true, 1), (4, 2, false, 0)]);
    let mut prepared =
        PreparedWindow::new(&mut source, Protocol::One, 3, specs.clone(), 3).unwrap();
    let claims: Vec<_> = specs
        .iter()
        .enumerate()
        .map(|(i, s)| tv::Instance {
            claim: tv_value(prepared.sum(i).unwrap()),
            vars: s.vars,
            evals: s.tables.iter().filter(|t| !t.public).count(),
        })
        .collect();
    if corrupt {
        // Change an algebraic grid entry, retaining the original true claim and source.
        // Omitted p(1) makes this a final relation mismatch, not a round-length failure.
        prepared.states[0].grid.as_mut().unwrap().values[0] += fp(1);
    }
    let proof = prepared
        .prove(
            &mut Transcript::new(&context()),
            Prefix::Stage(3),
            &mut source,
        )
        .unwrap()
        .into_proof();
    let proof = tv::SumcheckProof {
        rounds: proof
            .rounds
            .iter()
            .map(|r| r.iter().copied().map(tv_value).collect())
            .collect(),
        final_evals: proof.final_evals.iter().copied().map(tv_value).collect(),
    };
    let context = th::SessionContext {
        protocol: th::Ascii::new("tessera-protocol/1").unwrap(),
        mode: th::Mode::Live,
        params: th::ParamSetId(th::Ascii::new("test-small/1").unwrap()),
        statement_digest: [45; 32],
        session_id: [59; 16],
    };
    // The relation check uses verifier field arithmetic and the definition of an MLE. No
    // optimized grid, EQ, prefix-weight or interpolation helper is shared with this oracle.
    let relation = |i: usize, point: &[tv_field::Fp2], evals: &[tv_field::Fp2]| {
        let s = &specs[i];
        let mut sent = evals.iter();
        let values: Vec<_> = s
            .tables
            .iter()
            .map(|table| {
                if table.public {
                    source.tables[table.id].iter().enumerate().fold(
                        tv_field::Fp2::ZERO,
                        |sum, (index, &value)| {
                            let weight = point.iter().enumerate().fold(
                                tv_field::Fp2::ONE,
                                |weight, (bit, &r)| {
                                    weight
                                        * if index >> bit & 1 == 0 {
                                            tv_field::Fp2::ONE - r
                                        } else {
                                            r
                                        }
                                },
                            );
                            sum + tv_value(value) * weight
                        },
                    )
                } else {
                    *sent.next().unwrap()
                }
            })
            .collect();
        let eq =
            s.eq.iter()
                .zip(point)
                .fold(tv_field::Fp2::ONE, |p, (&z, &r)| {
                    p * z.map_or(tv_field::Fp2::ONE, |z| {
                        let z = tv_value(z);
                        (tv_field::Fp2::ONE - z) * (tv_field::Fp2::ONE - r) + z * r
                    })
                });
        Some(
            eq * s.terms.iter().fold(tv_field::Fp2::ZERO, |sum, term| {
                sum + term
                    .factors()
                    .iter()
                    .fold(tv_value(*term.weight()), |p, &k| p * values[k])
            }),
        )
    };
    tv::verify_batch(
        &mut th::Transcript::new(&context),
        tv::Prefix::Stage(3),
        3,
        &claims,
        &proof,
        relation,
    )
}

#[test]
fn independent_verifier_accepts_honest_grids_and_rejects_algebraic_corruption() {
    assert!(independent_verify(false).is_ok());
    assert!(matches!(
        independent_verify(true),
        Err(tv_core::sumcheck::SumcheckError::FinalMismatch { round: 5 })
    ));
}

struct BoundaryTranscript {
    count: usize,
}

impl SumcheckTranscript<F> for BoundaryTranscript {
    fn absorb_message(&mut self, _: &str, _: &[F]) {}
    fn challenge(&mut self, _: &str) -> F {
        let out = fp((self.count % 4) as u64);
        self.count += 1;
        out
    }
}

#[test]
fn zero_one_and_other_grid_node_challenges_match_literal_rounds() {
    let (mut source, specs) = specs(&[(11, 3, true, 2), (3, 2, true, 1)]);
    let eqs = eq_tables(&specs);
    let reference = prove_batch(
        &mut BoundaryTranscript { count: 0 },
        Protocol::One,
        Prefix::Stage(3),
        4,
        &literal(&source, &specs, &eqs),
    )
    .unwrap();
    let prepared = PreparedWindow::new(&mut source, Protocol::One, 4, specs, 9).unwrap();
    let actual = prepared
        .prove(
            &mut BoundaryTranscript { count: 0 },
            Prefix::Stage(3),
            &mut source,
        )
        .unwrap();
    assert_eq!(actual.proof().rounds, reference.proof().rounds);
    assert_eq!(actual.proof().final_evals, reference.proof().final_evals);
}

#[test]
fn source_failures_propagate_from_preparation_and_fold() {
    struct Failing {
        inner: Source,
        pass: Option<WindowPass>,
        read: bool,
    }
    impl WindowSource<F> for Failing {
        fn regenerate(&mut self, pass: WindowPass) -> Result<(), WindowError> {
            if self.pass == Some(pass) {
                Err(WindowError::Source)
            } else {
                self.inner.regenerate(pass)
            }
        }
        fn read(&self, id: usize, start: usize, out: &mut [F]) -> Result<(), WindowError> {
            if self.read {
                Err(WindowError::Source)
            } else {
                self.inner.read(id, start, out)
            }
        }
    }
    for (pass, read) in [(Some(WindowPass::Prepare), false), (None, true)] {
        let (inner, specs) = specs(&[(4, 2, true, 2)]);
        let mut source = Failing { inner, pass, read };
        assert!(matches!(
            PreparedWindow::new(&mut source, Protocol::One, 3, specs, 2),
            Err(WindowError::Source)
        ));
    }
    let (inner, specs) = specs(&[(4, 2, true, 2)]);
    let mut source = Failing {
        inner,
        pass: Some(WindowPass::Fold(2)),
        read: false,
    };
    let prepared = PreparedWindow::new(&mut source, Protocol::One, 3, specs, 2).unwrap();
    assert!(matches!(
        prepared.prove(
            &mut Transcript::new(&context()),
            Prefix::Stage(3),
            &mut source
        ),
        Err(WindowError::Source)
    ));
}

#[test]
fn grids_reduce_measured_engine_peak_and_do_not_retain_full_tables() {
    let (mut source, specs) = specs(&[(16, 2, true, 2)]);
    let eqs = eq_tables(&specs);
    let mut reference = None;
    let resident = allocation_counter::measure(|| {
        reference = Some(
            prove_batch(
                &mut Transcript::new(&context()),
                Protocol::One,
                Prefix::Stage(3),
                3,
                &literal(&source, &specs, &eqs),
            )
            .unwrap()
            .into_proof(),
        );
    });
    let window = allocation_counter::measure(|| {
        let prepared =
            PreparedWindow::new(&mut source, Protocol::One, 3, specs.clone(), 8).unwrap();
        assert!(prepared.states.iter().all(|s| s.folded.is_empty()));
        let actual = prepared
            .prove(
                &mut Transcript::new(&context()),
                Prefix::Stage(3),
                &mut source,
            )
            .unwrap();
        let reference = reference.as_ref().unwrap();
        assert_eq!(actual.proof().rounds, reference.rounds);
        assert_eq!(actual.proof().final_evals, reference.final_evals);
    });
    assert!(
        window.bytes_max * 2 < resident.bytes_max,
        "window peak {}, resident peak {}",
        window.bytes_max,
        resident.bytes_max
    );
}

#[test]
fn public_memory_model_bounds_every_measured_cut_and_selects_the_first_that_fits() {
    let (mut source, specs) = specs(&[
        (12, 2, true, 2),
        (9, 3, true, 1),
        (3, 3, false, 2),
        (0, 2, true, 0),
    ]);
    let shapes: Vec<_> = specs
        .iter()
        .map(|s| WindowShape {
            vars: s.vars,
            tables: s.tables.len(),
            delta: s.terms.iter().map(|t| t.factors().len()).max().unwrap(),
            windowable: s.windowed,
        })
        .collect();
    let mut plans = Vec::new();
    for cut in 0..12 {
        let plan = WindowMemory::for_cut(&shapes, cut, size_of::<F>()).unwrap();
        let measured = allocation_counter::measure(|| {
            let prepared =
                PreparedWindow::new(&mut source, Protocol::One, 4, specs.clone(), cut).unwrap();
            let _ = prepared
                .prove(
                    &mut Transcript::new(&context()),
                    Prefix::Stage(3),
                    &mut source,
                )
                .unwrap();
        });
        assert!(
            measured.bytes_max <= plan.working_bytes,
            "cut {cut}: peak {}, plan {}",
            measured.bytes_max,
            plan.working_bytes
        );
        assert_eq!(
            plan.fold_passes,
            source
                .passes
                .iter()
                .filter(|p| matches!(p, WindowPass::Fold(_)))
                .count()
        );
        source.passes.clear();
        plans.push(plan);
    }
    for budget in plans.iter().map(|p| p.working_bytes) {
        let selected = WindowMemory::select(&shapes, budget, size_of::<F>()).unwrap();
        assert_eq!(
            Some(&selected),
            plans.iter().find(|p| p.working_bytes <= budget)
        );
    }
    let minimum = plans.iter().map(|p| p.working_bytes).min().unwrap();
    assert_eq!(
        WindowMemory::select(&shapes, minimum - 1, 16),
        Err(WindowError::Allocation)
    );
    assert!(matches!(
        WindowMemory::for_cut(
            &[WindowShape {
                vars: 62,
                tables: usize::MAX,
                delta: 3,
                windowable: true
            }],
            0,
            16
        ),
        Err(WindowError::Allocation)
    ));
    assert!(matches!(
        WindowMemory::for_cut(
            &[WindowShape {
                vars: 64,
                tables: 1,
                delta: 3,
                windowable: true
            }],
            0,
            16
        ),
        Err(WindowError::Shape(_))
    ));
}

#[test]
fn resident_largest_instance_does_not_charge_padding_as_an_own_window() {
    compare(&[(12, 2, false, 2), (9, 2, true, 2), (2, 3, true, 1)], 10);
    compare(&[(12, 2, false, 2), (2, 3, true, 1)], 9);
    let shapes = [
        WindowShape {
            vars: 12,
            tables: 2,
            delta: 2,
            windowable: false,
        },
        WindowShape {
            vars: 9,
            tables: 2,
            delta: 2,
            windowable: true,
        },
    ];
    assert_eq!(
        WindowMemory::for_cut(&shapes, 10, 16).unwrap().fold_passes,
        1
    );
}

#[test]
fn overflowing_resident_charge_does_not_hide_a_representable_folded_plan() {
    let shapes = [WindowShape {
        vars: 62,
        tables: 1,
        delta: 1,
        windowable: true,
    }];
    assert_eq!(
        WindowMemory::for_cut(&shapes, 0, 16),
        Err(WindowError::Allocation)
    );
    assert_eq!(WindowMemory::select(&shapes, u64::MAX, 16).unwrap().cut, 3);
}

#[test]
fn every_graph_relation_kind_matches_literal_in_one_mixed_batch() {
    use super::super::Values;
    use crate::{
        Elementwise, ElementwiseShape, Input, MatMulAct, MatMulActShape, MatMulWeight,
        MatMulWeightShape, Recomposition, RightLayout, Rotate, Selected, Selection,
    };
    let base: Vec<_> = (0..64)
        .map(|i| Goldilocks::from_u64_reduced((3 * i + 7) as u64))
        .collect();
    let other: Vec<_> = (0..64)
        .map(|i| Goldilocks::from_u64_reduced((i * i + 1) as u64))
        .collect();
    let point: Vec<_> = (0..6).map(|i| fp((i + 3) as u64)).collect();
    let mul = Elementwise::mul(
        Input::Operand(&base),
        Input::Broadcast {
            operand: &other[..4],
            low_vars: 2,
        },
        6,
    )
    .unwrap()
    .tables(&point)
    .unwrap();
    let squares = Elementwise::sum_squares(
        Input::Operand(&base),
        ElementwiseShape {
            vars: 6,
            summed_vars: 2,
        },
    )
    .unwrap()
    .tables(&point[..4])
    .unwrap();
    let zero = Elementwise::new(
        ElementwiseShape::elementwise(6),
        vec![Input::Operand(&base), Input::Public(&other)],
        vec![Term::new(F::ONE, vec![0]), Term::new(-F::ONE, vec![1])],
    )
    .unwrap()
    .tables(&point)
    .unwrap();
    let rotate = Rotate::new(&base, &other[..32], &base[..32], 2, 6).and_then(|r| r.tables(&point));
    let rotate = rotate.unwrap();
    let selection = Selection::new(&base, 1, 4, 1)
        .unwrap()
        .tables(
            &point[..4],
            2,
            &[Selected {
                coefficient: fp(3),
                index: &[1, 4, 7],
            }],
        )
        .unwrap();
    let recomposition = Recomposition::new(
        Input::Operand(&base[..8]),
        &other[..32],
        Some(&base[..16]),
        3,
        2,
        3,
    )
    .unwrap()
    .tables(&point[..3])
    .unwrap();
    let mw = MatMulWeight::new(
        &base[..16],
        &other[..16],
        MatMulWeightShape {
            layer_vars: 1,
            token_vars: 1,
            in_vars: 2,
            out_vars: 1,
        },
    )
    .unwrap()
    .tables(&point[..3])
    .unwrap();
    let ma = MatMulAct::new(
        &base[..16],
        &other[..16],
        RightLayout::Direct,
        MatMulActShape {
            stack_vars: 1,
            row_vars: 1,
            inner_vars: 2,
            col_vars: 1,
        },
    )
    .unwrap()
    .tables(&point[..3])
    .unwrap();
    let instances = [
        mul.instance(),
        squares.instance(),
        zero.instance(),
        rotate.instance(),
        selection.instance(),
        recomposition.instance(),
        mw.instance(),
        ma.instance(),
    ];
    let mut source = Source {
        tables: Vec::new(),
        passes: Vec::new(),
    };
    let specs: Vec<_> = instances
        .iter()
        .enumerate()
        .map(|(i, instance)| {
            let vars = instance.tables()[0].len().trailing_zeros() as usize;
            let tables = instance
                .tables()
                .iter()
                .map(|table| {
                    let id = source.tables.len();
                    source.tables.push(match table.values {
                        Values::Base(values) => values.iter().copied().map(F::from).collect(),
                        Values::Ext(values) => values.to_vec(),
                    });
                    WindowTable {
                        id,
                        public: table.is_public(),
                    }
                })
                .collect();
            let terms = instance.terms().map_or_else(
                || vec![Term::new(F::ONE, (0..instance.tables().len()).collect())],
                <[_]>::to_vec,
            );
            // Public relation kernels remain their original multilinear factors. In particular a
            // Rotate's eq-weighted trig kernel must not be replaced by a product off the cube.
            WindowInstance {
                vars,
                tables,
                terms,
                eq: vec![None; vars],
                windowed: i < 6,
            }
        })
        .collect();
    let mut tr = Transcript::new(&context());
    let reference = prove_batch(&mut tr, Protocol::One, Prefix::Stage(3), 3, &instances).unwrap();
    let after = tr.challenge_fp2("after");
    for cut in 0..6 {
        let prepared =
            PreparedWindow::new(&mut source, Protocol::One, 3, specs.clone(), cut).unwrap();
        let mut tr = Transcript::new(&context());
        let actual = prepared
            .prove(&mut tr, Prefix::Stage(3), &mut source)
            .unwrap();
        assert_eq!(actual.proof().rounds, reference.proof().rounds);
        assert_eq!(actual.proof().final_evals, reference.proof().final_evals);
        assert_eq!(tr.challenge_fp2("after"), after);
    }
}

#[test]
fn prefix_weights_do_not_overlap_a_consumed_grid_allocation() {
    // Recreate the last boundary of a 32-round cut. The contracted grid has one live entry
    // but retains its original 3^8 capacity. Refuse the first source read so the test measures
    // the real allocation boundary without visiting a 2^33-entry virtual tensor.
    let shape = WindowShape {
        vars: 33,
        tables: 2,
        delta: 2,
        windowable: true,
    };
    let plan = WindowMemory::for_cut(&[shape], 32, size_of::<F>()).unwrap();
    let measured = allocation_counter::measure(|| {
        let mut values = SecretBuf::filled(F::ZERO, 3usize.pow(8));
        values.truncate(1);
        let mut state = State {
            spec: WindowInstance {
                vars: 33,
                tables: vec![
                    WindowTable {
                        id: 0,
                        public: false,
                    },
                    WindowTable {
                        id: 1,
                        public: false,
                    },
                ],
                terms: vec![Term::new(F::ONE, vec![0, 1])],
                eq: vec![None; 33],
                windowed: false,
            },
            grid: Some(Grid {
                values,
                radix: 3,
                end: 32,
            }),
            folded: Vec::new(),
            bound: 32,
            sum: SecretBuf::filled(F::ZERO, 1),
        };
        let source = Source {
            tables: Vec::new(),
            passes: Vec::new(),
        };
        assert_eq!(
            state.prepare(&source, &vec![fp(7); 32], 32, false),
            Err(WindowError::Source)
        );
    });
    assert!(
        measured.bytes_max <= plan.working_bytes,
        "prefix/grid overlap: measured {}, plan {}",
        measured.bytes_max,
        plan.working_bytes
    );
}

#[test]
fn window_metadata_covers_consumed_specs_messages_and_returned_storage() {
    fn measure(source: &mut Source, specs: &[WindowInstance<F>], degree: usize, cuts: &[usize]) {
        let shapes = specs
            .iter()
            .map(|s| WindowShape {
                vars: s.vars,
                tables: s.tables.len(),
                delta: s.terms.iter().map(|t| t.factors().len()).max().unwrap(),
                windowable: s.windowed,
            })
            .collect::<Vec<_>>();
        let terms = specs
            .iter()
            .map(|s| s.terms.len() as u64)
            .collect::<Vec<_>>();
        let mut metadata = None;
        let planning = allocation_counter::measure(|| {
            metadata = Some(WindowMetadataMemory::new::<F>(&shapes, &terms, degree).unwrap());
        });
        assert_eq!(planning.count_total, 0);
        let metadata = metadata.unwrap();
        // Source-owned callback history is outside the engine charge.
        source.passes.reserve(128);
        for width in [1, 2, MAX_WINDOW_VARS] {
            for &cut in cuts {
                let fields =
                    WindowMemory::with_window_vars(&shapes, cut, size_of::<F>(), width).unwrap();
                let mut result = None;
                let measured = allocation_counter::measure(|| {
                    let prepared = PreparedWindow::with_window_vars(
                        source,
                        Protocol::One,
                        degree,
                        specs.to_vec(),
                        cut,
                        width,
                    )
                    .unwrap();
                    result = Some(
                        prepared
                            .prove(
                                &mut BoundaryTranscript { count: 0 },
                                Prefix::GroupTree(u32::MAX, u32::MAX),
                                source,
                            )
                            .unwrap(),
                    );
                });
                assert!(
                    measured.bytes_max <= fields.working_bytes + metadata.peak_bytes(),
                    "cut {cut}, width {width}: {measured:?}, fields {fields:?}, metadata {metadata:?}"
                );
                assert!(
                    u64::try_from(measured.bytes_current).unwrap() <= metadata.output_bytes(),
                    "output {measured:?}, metadata {metadata:?}"
                );
                drop(result);
                source.passes.clear();
            }
        }
    }
    let (mut source, specs) = specs(&[(8, 3, true, 2), (5, 2, false, 1), (0, 1, true, 0)]);
    measure(&mut source, &specs, 4, &[0, 1, 4, 7]);

    // Metadata dominates the singleton payload; retain each unused column and
    // each distinct term descriptor even though all values happen to be zero.
    let mut source = Source {
        tables: vec![vec![F::ZERO]; 512],
        passes: Vec::new(),
    };
    let specs = vec![WindowInstance {
        vars: 0,
        tables: (0..512)
            .map(|id| WindowTable {
                id,
                public: id % 3 == 0,
            })
            .collect(),
        terms: (0..8192)
            .map(|i| Term::new(fp(i + 1), vec![i as usize % 512]))
            .collect(),
        eq: Vec::new(),
        windowed: true,
    }];
    measure(&mut source, &specs, 1, &[0]);
}
