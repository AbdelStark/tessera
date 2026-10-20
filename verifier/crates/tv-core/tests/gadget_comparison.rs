// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Comparison relations against the frozen pre-implementation integer-prefix oracle.
use tv_core::advice::ComparisonRole;
use tv_core::gadgets::{
    GadgetError,
    comparison::{Comparison, Input, Values},
};
use tv_core::validation::GadgetId;
use tv_field::{Fp, Fp2};
use tv_testkit::{Json, read_vectors};

fn array(v: &Json) -> &[Json] {
    v.as_array().unwrap()
}
fn member<'a>(v: &'a Json, k: &str) -> &'a Json {
    v.get(k).unwrap()
}
fn integers(v: &Json) -> Vec<u64> {
    array(v).iter().map(|v| v.as_u64().unwrap()).collect()
}
fn field(v: &Json) -> Fp2 {
    let a = integers(v);
    Fp2::new(
        Fp::from_canonical(a[0]).unwrap(),
        Fp::from_canonical(a[1]).unwrap(),
    )
}
fn fields(v: &Json) -> Vec<Fp2> {
    array(v).iter().map(field).collect()
}
fn scalar(v: &Json) -> Vec<Fp2> {
    integers(v)
        .into_iter()
        .map(|v| Fp2::from(Fp::from_canonical(v).unwrap()))
        .collect()
}
fn fixture() -> Json {
    read_vectors("gadget-finals/comparison.json")
}
const ID: GadgetId = GadgetId(19);
fn rejected() -> GadgetError {
    GadgetError::Rejected { gadget: ID }
}

fn check_claims(
    plan: &Comparison,
    bits: u8,
    point: &[Fp2],
    left: &[Fp2],
    right: &[Fp2],
    arrays: &[Vec<Fp2>; 6],
    widths: &[Json],
) {
    let values = Values {
        point,
        left,
        right,
        arrays: arrays.each_ref().map(Vec::as_slice),
    };
    let checked = plan.check(ID, values).unwrap();
    let claims: Vec<_> = checked.claims().collect();
    assert_eq!(claims.len(), plan.limbs() * 8);
    let roles = [
        Input::Left,
        Input::Right,
        Input::Witness(ComparisonRole::LeftDigit),
        Input::Witness(ComparisonRole::LeftCarry),
        Input::Witness(ComparisonRole::RightDigit),
        Input::Witness(ComparisonRole::RightCarry),
        Input::Witness(ComparisonRole::Difference),
        Input::Witness(ComparisonRole::Borrow),
    ];
    for (input, values) in [left, right]
        .into_iter()
        .chain(arrays.iter().map(Vec::as_slice))
        .enumerate()
    {
        for (limb, &value) in values.iter().enumerate() {
            let claim = &claims[input * plan.limbs() + limb];
            assert_eq!(claim.input, roles[input]);
            assert_eq!(claim.limb, limb);
            assert_eq!(claim.point, point);
            assert_eq!(claim.value, value);
            let expected = match input {
                0 | 1 => None,
                3 => Some(integers(&widths[0])[limb] as u8),
                5 => Some(integers(&widths[1])[limb] as u8),
                7 => Some(1),
                _ => Some(bits),
            };
            assert_eq!(claim.range_bits, expected);
        }
    }
}

#[test]
fn every_integer_and_non_boolean_comparison_matches_the_frozen_oracle() {
    let fixture = fixture();
    assert_eq!(member(&fixture, "schema_version").as_u64(), Some(1));
    let cases = array(member(&fixture, "cases"));
    assert_eq!(cases.len(), 75);
    let mut witnesses = 0;
    for case in cases {
        let bits = member(case, "bits").as_u64().unwrap() as u8;
        let bounds = array(member(case, "bounds"));
        let plan = Comparison::derive(bits, &integers(&bounds[0]), &integers(&bounds[1])).unwrap();
        let widths = array(member(case, "widths"));
        for row in array(member(case, "rows")) {
            let arrays: [Vec<_>; 6] = array(member(row, "arrays"))
                .iter()
                .map(scalar)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            check_claims(
                &plan,
                bits,
                &[],
                &scalar(member(row, "left")),
                &scalar(member(row, "right")),
                &arrays,
                widths,
            );
            witnesses += 1;
        }
        let evals = member(case, "evaluations");
        let left = fields(member(evals, "left"));
        let right = fields(member(evals, "right"));
        let arrays: [Vec<_>; 6] = array(member(evals, "arrays"))
            .iter()
            .map(fields)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let point = fields(member(case, "point"));
        check_claims(&plan, bits, &point, &left, &right, &arrays, widths);
        // Each altered witness column is rejected at this same non-Boolean point.
        for a in 0..6 {
            for limb in 0..plan.limbs() {
                let mut changed = arrays.clone();
                changed[a][limb] += Fp2::ONE;
                assert_eq!(
                    plan.check(
                        ID,
                        Values {
                            point: &point,
                            left: &left,
                            right: &right,
                            arrays: changed.each_ref().map(Vec::as_slice)
                        }
                    )
                    .err(),
                    Some(rejected()),
                    "{} array {a} limb {limb}",
                    member(case, "name").as_str().unwrap()
                );
            }
        }
    }
    assert_eq!(witnesses, 572);
}

fn f(v: u64) -> Fp2 {
    Fp2::from(Fp::reduce(v))
}

#[test]
fn final_borrow_and_both_final_carries_are_checked_even_when_the_equations_hold() {
    let plan = Comparison::derive(8, &[255, 0], &[255, 0]).unwrap();
    let z = [Fp2::ZERO; 2];
    // X=1, Y=0: all digits/borrows lie in their prescribed boxes and every
    // subtraction equation holds, but Y-X has an outgoing borrow.
    let one = [Fp2::ONE, Fp2::ZERO];
    let diff = [f(255); 2];
    let borrow = [Fp2::ONE; 2];
    let values = Values {
        point: &[],
        left: &one,
        right: &z,
        arrays: [&one, &z, &z, &z, &diff, &borrow],
    };
    assert_eq!(plan.check(ID, values).err(), Some(rejected()));
    // The local terminal-carry checks also reject truncated raw inputs. Binding
    // the public coefficient bounds upstream is a separate required obligation.
    let overflow = [Fp2::ZERO, f(256)];
    let carry = [Fp2::ZERO, Fp2::ONE];
    for arrays in [[&z, &carry, &z, &z, &z, &z], [&z, &z, &z, &carry, &z, &z]] {
        let left = if arrays[1] == &carry { &overflow } else { &z };
        let right = if arrays[3] == &carry { &overflow } else { &z };
        assert_eq!(
            plan.check(
                ID,
                Values {
                    point: &[],
                    left,
                    right,
                    arrays: arrays.map(|a| &a[..])
                }
            )
            .err(),
            Some(rejected())
        );
    }
}

#[test]
fn invalid_public_parameters_are_internal_and_proof_shapes_name_the_gadget() {
    for (bits, l, r) in [
        (0, vec![0], vec![0]),
        (33, vec![0], vec![0]),
        (8, vec![], vec![]),
        (8, vec![0], vec![0, 0]),
        (8, vec![1 << 40, 0], vec![0, 0]),
        (8, vec![0, 0], vec![1 << 40, 0]),
        (8, vec![256], vec![0]),
        (8, vec![0], vec![256]),
        (16, vec![1 << 40, 0, 0], vec![0, 0, 0]),
        (16, vec![0, 0, 0], vec![1 << 40, 0, 0]),
    ] {
        assert_eq!(
            Comparison::derive(bits, &l, &r).err(),
            Some(GadgetError::Internal)
        );
    }
    // Enough zero high limbs make the final maximum carry zero; only the
    // unsupported lookup width rejects these public boxes.
    let mut wide_carry = vec![0; 41];
    wide_carry[0] = 1 << 39;
    for (left, right) in [
        (&wide_carry[..], &[0; 41][..]),
        (&[0; 41][..], &wide_carry[..]),
    ] {
        assert_eq!(
            Comparison::derive(1, left, right).err(),
            Some(GadgetError::Internal)
        );
    }
    let plan = Comparison::derive(8, &[0, 0], &[0, 0]).unwrap();
    let z = [Fp2::ZERO; 2];
    for which in 0..8 {
        for bad in [&z[..1], &[Fp2::ZERO; 3][..]] {
            let mut slices = [&z[..]; 8];
            slices[which] = bad;
            let result = plan.check(
                ID,
                Values {
                    point: &[],
                    left: slices[0],
                    right: slices[1],
                    arrays: slices[2..].try_into().unwrap(),
                },
            );
            let error = result.err().unwrap();
            assert_eq!(error, rejected());
            assert_eq!(error.rejected_gadget(), Some(ID));
            assert_eq!(error.code(), "REJ-GADGET");
            assert_eq!(error.to_string(), "gadget 19 failed its final relation");
        }
    }
    assert_eq!(GadgetError::Internal.code(), "ERR-INTERNAL");
    assert_eq!(GadgetError::Memory.code(), "ERR-RES-MEMORY");
    assert_eq!(GadgetError::Internal.rejected_gadget(), None);
    assert_eq!(GadgetError::Memory.rejected_gadget(), None);
}
