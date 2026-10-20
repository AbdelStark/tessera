// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Actual root-product sumchecks from the frozen direct-vertex oracle.
mod support;
use support::{Context, digest, extension};
use tv_core::gadgets::{GadgetError, root_products::Products};
use tv_core::sumcheck::{self, Instance, Prefix, SumcheckError, SumcheckProof};
use tv_field::{Fp, Fp2};
use tv_hash::transcript::{Ascii, Transcript};
use tv_testkit::{Json, read_vectors};

fn array(v: &Json) -> &[Json] {
    v.as_array().unwrap()
}
fn get<'a>(v: &'a Json, k: &str) -> &'a Json {
    v.get(k).unwrap()
}
fn field(v: &Json) -> Fp2 {
    let a = array(v);
    Fp2::new(
        Fp::from_canonical(a[0].as_u64().unwrap()).unwrap(),
        Fp::from_canonical(a[1].as_u64().unwrap()).unwrap(),
    )
}
fn fields(v: &Json) -> Vec<Fp2> {
    array(v).iter().map(field).collect()
}
fn plan(case: &Json) -> Products {
    Products::derive(get(case, "width").as_u64().unwrap() as u8).unwrap()
}
fn before(case: &Json) -> Transcript {
    let mut t = Context::from_json(get(case, "context")).transcript();
    let claims = fields(get(case, "claims"));
    let mut bytes = (claims.len() as u64).to_le_bytes().to_vec();
    for value in claims {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    t.absorb(Ascii::new("stage/0/split").unwrap(), &bytes);
    t
}
fn proof(case: &Json) -> SumcheckProof {
    SumcheckProof {
        rounds: array(get(case, "rounds")).iter().map(fields).collect(),
        final_evals: fields(get(case, "final_evals")),
    }
}
fn run(
    case: &Json,
    proof: &SumcheckProof,
    replace_one_by_selector: bool,
) -> Result<(sumcheck::Batch, Transcript), SumcheckError> {
    let plan = plan(case);
    let target = fields(get(case, "target"));
    let instances: Vec<_> = fields(get(case, "claims"))
        .into_iter()
        .map(|claim| Instance {
            claim,
            vars: 3,
            evals: plan.evaluations(),
        })
        .collect();
    let mut t = before(case);
    let batch = sumcheck::verify_batch(
        &mut t,
        Prefix::Stage(1),
        4,
        &instances,
        proof,
        |index, point, evals| {
            let (increment, j) = (index >= plan.coefficients(), index % plan.coefficients());
            let value = if increment && replace_one_by_selector {
                let mut changed = evals.to_vec();
                // Four real rows followed by four padded ones: psi=1-x_2.
                changed[0] -= point[2];
                plan.evaluate(j, increment, &changed)?
            } else {
                plan.evaluate(j, increment, evals)?
            };
            Some(sumcheck::eq(&target, point)? * value)
        },
    )?;
    Ok((batch, t))
}

#[test]
fn all_root_products_verify_and_bind_to_complete_comparison_coefficients() {
    let fixture = read_vectors("gadget-finals/root-products.json");
    let comparisons = read_vectors("gadget-finals/comparison.json");
    let cases = array(get(&fixture, "cases"));
    assert_eq!(cases.len(), 35);
    let mut instance_count = 0;
    for case in cases {
        assert_eq!(get(case, "degree").as_u64(), Some(4));
        let plan = plan(case);
        let proof = proof(case);
        let (batch, t) = run(case, &proof, false).unwrap();
        assert_eq!(batch.gamma, field(get(case, "gamma")));
        assert_eq!(batch.point, fields(get(case, "round_point")));
        assert_eq!(*t.state(), digest(get(case, "state").as_str().unwrap()));
        assert_eq!(t.counter(), get(case, "counter").as_u64().unwrap());
        let target = fields(get(case, "target"));
        let claims = fields(get(case, "claims"));
        assert_eq!(claims.len(), 2 * plan.coefficients());
        let name = get(case, "name").as_str().unwrap();
        for (side, key) in [("lower", "left"), ("upper", "right")] {
            let wanted = format!("{name}-{side}");
            let candidates = case
                .get("alternative_comparisons")
                .unwrap_or_else(|| get(&comparisons, "cases"));
            let comparison = array(candidates)
                .iter()
                .find(|c| get(c, "name").as_str() == Some(&wanted))
                .unwrap();
            let rows = array(get(comparison, "rows"));
            for j in 0..plan.coefficients() {
                let values: Vec<_> = rows
                    .iter()
                    .map(|r| Fp::from_canonical(array(get(r, key))[j].as_u64().unwrap()).unwrap())
                    .collect();
                let expected = extension(&values, &target);
                let i = if side == "upper" {
                    plan.coefficients() + j
                } else {
                    j
                };
                assert_eq!(claims[i], expected);
            }
            use tv_core::gadgets::comparison::{Comparison, Values};
            let bounds = array(get(comparison, "bounds"));
            let integers = |v: &Json| {
                array(v)
                    .iter()
                    .map(|v| v.as_u64().unwrap())
                    .collect::<Vec<_>>()
            };
            let comparison_plan =
                Comparison::derive(8, &integers(&bounds[0]), &integers(&bounds[1])).unwrap();
            let at = |values: Vec<Fp>| extension(&values, &target);
            let input = |key: &str| {
                (0..plan.coefficients())
                    .map(|j| {
                        at(rows
                            .iter()
                            .map(|r| {
                                Fp::from_canonical(array(get(r, key))[j].as_u64().unwrap()).unwrap()
                            })
                            .collect())
                    })
                    .collect::<Vec<_>>()
            };
            let left = input("left");
            let right = input("right");
            let arrays: [Vec<_>; 6] = std::array::from_fn(|a| {
                (0..plan.coefficients())
                    .map(|j| {
                        at(rows
                            .iter()
                            .map(|r| {
                                Fp::from_canonical(
                                    array(&array(get(r, "arrays"))[a])[j].as_u64().unwrap(),
                                )
                                .unwrap()
                            })
                            .collect())
                    })
                    .collect()
            });
            let checked = comparison_plan.check(
                tv_core::validation::GadgetId(99),
                Values {
                    point: &target,
                    left: &left,
                    right: &right,
                    arrays: arrays.each_ref().map(Vec::as_slice),
                },
            );
            if case.get("alternative_comparisons").is_some() && side == "lower" {
                assert_eq!(
                    checked.err(),
                    Some(GadgetError::Rejected {
                        gadget: tv_core::validation::GadgetId(99)
                    })
                );
            } else {
                assert_eq!(checked.unwrap().claims().count(), 8 * plan.coefficients());
            }
        }
        let expected: Vec<_> = array(get(case, "columns"))
            .iter()
            .map(|column| {
                let values: Vec<_> = array(column)
                    .iter()
                    .map(|v| {
                        let pair = array(v);
                        assert_eq!(pair[1].as_u64(), Some(0));
                        Fp::from_canonical(pair[0].as_u64().unwrap()).unwrap()
                    })
                    .collect();
                extension(&values, &batch.point)
            })
            .collect();
        for actual in proof.final_evals.chunks_exact(plan.evaluations()) {
            assert_eq!(actual, &expected);
        }
        let mut t = before(case);
        let gamma = t
            .challenge_fp2(Ascii::new("stage/1/gamma").unwrap())
            .unwrap();
        let (mut weight, mut entering) = (Fp2::ONE, Fp2::ZERO);
        for claim in &claims {
            entering += weight * *claim;
            weight *= gamma;
        }
        let reduced =
            sumcheck::reduce(&mut t, Prefix::Stage(1), 4, 3, entering, &proof.rounds).unwrap();
        assert_eq!(reduced.claim, field(get(case, "final_claim")));
        instance_count += claims.len();
    }
    assert_eq!(instance_count, 826);
}

#[test]
fn zero_factors_and_complete_119_bit_integer_products_agree() {
    let plan = Products::derive(32).unwrap();
    let mut values = [Fp2::ZERO; 8];
    values[4] = Fp2::ONE;
    assert_eq!(plan.evaluate(0, false, &values), Some(Fp2::ZERO));
    assert_eq!(plan.evaluate(0, true, &values), Some(Fp2::ONE));
    for j in 1..plan.coefficients() {
        assert_eq!(plan.evaluate(j, true, &values), Some(Fp2::ZERO));
    }
    let values: Vec<_> = [255u64, 255, 255, 255, 65535, 65535, 65535, 127]
        .into_iter()
        .map(|v| Fp2::from(Fp::from_canonical(v).unwrap()))
        .collect();
    for increment in [false, true] {
        let mut full = 0u128;
        for j in 0..plan.coefficients() {
            let bytes = plan.evaluate(j, increment, &values).unwrap().to_le_bytes();
            assert_eq!(&bytes[8..], &[0; 8]);
            let coefficient = u64::from_le_bytes(bytes[..8].try_into().unwrap());
            full += u128::from(coefficient) << (8 * j);
        }
        let root = (1u128 << 32) - 1 + u128::from(increment);
        assert_eq!(full, root * root * ((1u128 << 55) - 1));
        assert_eq!(128 - full.leading_zeros(), 119);
    }
}

#[test]
fn field_point_identity_tampering_and_wrong_shapes_get_exact_errors() {
    let fixture = read_vectors("gadget-finals/root-products.json");
    for case in array(get(&fixture, "cases")) {
        let original = proof(case);
        assert_eq!(
            run(case, &original, true).err(),
            Some(SumcheckError::FinalMismatch { round: 3 })
        );
        // R_0 and U_0 both enter coefficient zero: altered final evaluations fail.
        for index in [0, plan(case).root_limbs()] {
            let mut changed = original.clone();
            changed.final_evals[index] += Fp2::ONE;
            assert_eq!(
                run(case, &changed, false).err(),
                Some(SumcheckError::FinalMismatch { round: 3 })
            );
        }
        let mut changed = original.clone();
        changed.final_evals.pop();
        assert!(matches!(
            run(case, &changed, false),
            Err(SumcheckError::EvaluationCount { round: 3, .. })
        ));
        let plan = plan(case);
        assert_eq!(
            plan.evaluate(
                plan.coefficients(),
                false,
                &original.final_evals[..plan.evaluations()]
            ),
            None
        );
        assert_eq!(
            plan.evaluate(0, false, &original.final_evals[..plan.evaluations() - 1]),
            None
        );
    }
    for width in [0, 33, 255] {
        assert_eq!(Products::derive(width).err(), Some(GadgetError::Internal));
    }
}
