// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Narrowed limb boxes against the frozen direct-cube oracle (RFC-0009/0050).
use tv_core::gadgets::{
    GadgetError,
    decomposition::{Decomposition, Values},
};
use tv_core::validation::GadgetId;
use tv_field::{Fp, Fp2};
use tv_testkit::{Json, read_vectors};

fn array(v: &Json) -> &[Json] {
    v.as_array().unwrap()
}
fn get<'a>(v: &'a Json, k: &str) -> &'a Json {
    v.get(k).unwrap()
}
fn scalar(v: &Json) -> Fp2 {
    Fp2::from(Fp::from_canonical(v.as_u64().unwrap()).unwrap())
}
fn field(v: &Json) -> Fp2 {
    let a = array(v);
    Fp2::new(
        Fp::from_canonical(a[0].as_u64().unwrap()).unwrap(),
        Fp::from_canonical(a[1].as_u64().unwrap()).unwrap(),
    )
}
const ID: GadgetId = GadgetId(23);
fn rejected() -> GadgetError {
    GadgetError::Rejected { gadget: ID }
}

fn verify(plan: Decomposition, point: &[Fp2], value: Fp2, keys: &[Fp2], widths: &[Json]) {
    let low = &keys[..keys.len() - 1];
    let checked = plan
        .check(
            ID,
            Values {
                point,
                value,
                low,
                keys,
            },
        )
        .unwrap();
    let claims: Vec<_> = checked.claims().collect();
    assert_eq!(claims.len(), keys.len() + 1);
    assert_eq!(claims[0].limb, None);
    assert_eq!(claims[0].range_bits, None);
    assert_eq!(claims[0].value, value);
    assert_eq!(claims[0].point, point);
    for (i, (&key, width)) in keys.iter().zip(widths).enumerate() {
        assert_eq!(claims[i + 1].limb, Some(i));
        assert_eq!(
            claims[i + 1].range_bits,
            Some(width.as_u64().unwrap() as u8)
        );
        assert_eq!(claims[i + 1].point, point);
        assert_eq!(claims[i + 1].value, key);
    }
    for i in 0..keys.len() {
        let mut wrong = keys.to_vec();
        wrong[i] += Fp2::ONE;
        assert_eq!(
            plan.check(
                ID,
                Values {
                    point,
                    value,
                    low,
                    keys: &wrong
                }
            )
            .err(),
            Some(rejected())
        );
    }
    for i in 0..low.len() {
        let mut wrong = low.to_vec();
        wrong[i] += Fp2::ONE;
        assert_eq!(
            plan.check(
                ID,
                Values {
                    point,
                    value,
                    low: &wrong,
                    keys
                }
            )
            .err(),
            Some(rejected())
        );
    }
    assert_eq!(
        plan.check(
            ID,
            Values {
                point,
                value: value + Fp2::ONE,
                low,
                keys
            }
        )
        .err(),
        Some(rejected())
    );
}

#[test]
fn all_62_widths_and_every_narrowed_top_match_the_independent_oracle() {
    let fixture = read_vectors("gadget-finals/decomposition.json");
    assert_eq!(get(&fixture, "schema_version").as_u64(), Some(1));
    let cases = array(get(&fixture, "cases"));
    assert_eq!(cases.len(), 248);
    let mut rows = 0;
    for case in cases {
        let bits = get(case, "bits").as_u64().unwrap() as u8;
        let limb = get(case, "limb_bits").as_u64().unwrap() as u8;
        let plan = Decomposition::derive(bits, limb).unwrap();
        let widths = array(get(case, "widths"));
        assert_eq!(plan.limbs(), widths.len());
        for row in array(get(case, "rows")) {
            let keys: Vec<_> = array(get(row, "limbs")).iter().map(scalar).collect();
            verify(plan, &[], scalar(get(row, "value")), &keys, widths);
            rows += 1;
        }
        let point: Vec<_> = array(get(case, "point")).iter().map(field).collect();
        let keys: Vec<_> = array(get(case, "limbs")).iter().map(field).collect();
        verify(plan, &point, field(get(case, "value")), &keys, widths);
    }
    assert_eq!(rows, 992);
}

#[test]
fn invalid_boxes_are_internal_and_wrong_proof_shapes_reject_the_gadget() {
    for (bits, limb) in [
        (0, 16),
        (63, 16),
        (64, 16),
        (255, 8),
        (55, 0),
        (55, 33),
        (32, 255),
    ] {
        assert_eq!(
            Decomposition::derive(bits, limb).err(),
            Some(GadgetError::Internal)
        );
    }
    let plan = Decomposition::derive(55, 16).unwrap();
    for (low, keys) in [(2, 4), (4, 4), (3, 3), (3, 5)] {
        let low = vec![Fp2::ZERO; low];
        let keys = vec![Fp2::ZERO; keys];
        assert_eq!(
            plan.check(
                ID,
                Values {
                    point: &[],
                    value: Fp2::ZERO,
                    low: &low,
                    keys: &keys
                }
            )
            .err(),
            Some(rejected())
        );
    }
    // Full 16-bit top limbs would admit u+P as a second decomposition.
    let u = Fp2::ZERO;
    let digits: Vec<_> = (0..4)
        .map(|j| Fp2::from(Fp::reduce((Fp::MODULUS >> (16 * j)) & 65535)))
        .collect();
    assert_eq!(
        plan.check(
            ID,
            Values {
                point: &[],
                value: u,
                low: &digits[..3],
                keys: &digits
            }
        )
        .err(),
        None
    );
    let checked = plan
        .check(
            ID,
            Values {
                point: &[],
                value: u,
                low: &digits[..3],
                keys: &digits,
            },
        )
        .unwrap();
    let top = checked.claims().last().unwrap();
    assert_eq!(top.range_bits, Some(7));
    assert_eq!(top.value, digits[3]);
    let narrow = tv_tables::Table::generate(
        tv_tables::TableId(0),
        tv_tables::TableKind::Range { bits: 7 },
    )
    .unwrap();
    let wide = tv_tables::Table::generate(
        tv_tables::TableId(1),
        tv_tables::TableKind::Range { bits: 16 },
    )
    .unwrap();
    assert_eq!(narrow.row(Fp::MODULUS >> 48), None);
    assert!(wide.row(Fp::MODULUS >> 48).is_some());
    // The affine field identity alone holds: the returned top range obligation
    // is what rules this attack out. It must never be omitted or widened.
}
