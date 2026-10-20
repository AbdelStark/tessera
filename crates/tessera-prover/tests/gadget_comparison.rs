// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The independent comparison oracle consumed by the producer and verifier.
//! The verifier passed its frozen cases before this producer adapter was written (DEP-2).
use serde_json::Value;
use tessera_field::{Goldilocks, PrimeField};
use tessera_gadgets::{BigIntCompare, Committed, CompareColumn as C, NonNeg, Side, Term};
use tessera_tables::{GeneratedTable, Table, TableId, TableKind, TableSpec};
use tv_core::gadgets::comparison::{Comparison, Values};
use tv_core::validation::GadgetId;
use tv_field::{Fp, Fp2};

fn uints(v: &Value) -> Vec<u64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect()
}
fn fp2(v: &Value) -> Fp2 {
    let v = uints(v);
    Fp2::new(
        Fp::from_canonical(v[0]).unwrap(),
        Fp::from_canonical(v[1]).unwrap(),
    )
}
fn field(v: u64) -> Fp2 {
    Fp2::from(Fp::from_canonical(v).unwrap())
}
fn eq(point: &[Fp2], index: usize) -> Fp2 {
    point.iter().enumerate().fold(Fp2::ONE, |out, (i, &r)| {
        out * if index >> i & 1 == 0 { Fp2::ONE - r } else { r }
    })
}

fn range_tables() -> Vec<TableSpec> {
    (1..=32)
        .map(|bits| {
            GeneratedTable::new(TableId(u32::from(bits)), TableKind::Range { bits })
                .unwrap()
                .spec()
                .clone()
        })
        .collect()
}

#[test]
fn generated_columns_and_their_evaluations_match_all_frozen_comparisons() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../spec/vectors/gadget-finals/comparison.json"
    ))
    .unwrap();
    assert_eq!(fixture["schema_version"], 1);
    let tables = range_tables();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 75);
    let mut row_count = 0;
    for case in cases {
        let bits = case["bits"].as_u64().unwrap() as u8;
        let (bounds_l, bounds_r) = (uints(&case["bounds"][0]), uints(&case["bounds"][1]));
        let producer = BigIntCompare::new(bits, &bounds_l, &bounds_r, &tables).unwrap();
        let independent = Comparison::derive(bits, &bounds_l, &bounds_r).unwrap();
        let m = producer.limbs();
        assert_eq!(m, independent.limbs());
        let rows = case["rows"].as_array().unwrap();
        let left: Vec<Vec<i64>> = (0..m)
            .map(|j| {
                rows.iter()
                    .map(|row| row["left"][j].as_i64().unwrap())
                    .collect()
            })
            .collect();
        let right: Vec<Vec<i64>> = (0..m)
            .map(|j| {
                rows.iter()
                    .map(|row| row["right"][j].as_i64().unwrap())
                    .collect()
            })
            .collect();
        let witness = producer
            .witness(
                &left.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                &right.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            )
            .unwrap();
        let point: Vec<_> = case["point"].as_array().unwrap().iter().map(fp2).collect();
        let columns: [Vec<_>; 6] = std::array::from_fn(|a| {
            (0..m)
                .map(|j| match a {
                    0 => C::Digit(Side::Left, j),
                    1 => C::Carry(Side::Left, j),
                    2 => C::Digit(Side::Right, j),
                    3 => C::Carry(Side::Right, j),
                    4 => C::Difference(j),
                    5 => C::Borrow(j),
                    _ => unreachable!(),
                })
                .collect()
        });
        let arrays: [Vec<_>; 6] = std::array::from_fn(|a| {
            columns[a]
                .iter()
                .enumerate()
                .map(|(j, &column)| {
                    let actual = witness.tensor(column).unwrap();
                    assert_eq!(actual.len(), rows.len());
                    for (row, &value) in rows.iter().zip(actual) {
                        assert_eq!(row["arrays"][a][j].as_i64(), Some(value));
                    }
                    let evaluation = actual.iter().enumerate().fold(Fp2::ZERO, |out, (r, &v)| {
                        out + eq(&point, r) * field(v as u64)
                    });
                    assert_eq!(evaluation, fp2(&case["evaluations"]["arrays"][a][j]));
                    evaluation
                })
                .collect()
        });
        let eval = |input: &[Vec<i64>]| {
            input
                .iter()
                .map(|column| {
                    column.iter().enumerate().fold(Fp2::ZERO, |out, (r, &v)| {
                        out + eq(&point, r) * field(v as u64)
                    })
                })
                .collect::<Vec<_>>()
        };
        let left = eval(&left);
        let right = eval(&right);
        for (actual, expected) in [
            (&left, &case["evaluations"]["left"]),
            (&right, &case["evaluations"]["right"]),
        ] {
            for (&actual, expected) in actual.iter().zip(expected.as_array().unwrap()) {
                assert_eq!(actual, fp2(expected));
            }
        }
        let checked = independent
            .check(
                GadgetId(19),
                Values {
                    point: &point,
                    left: &left,
                    right: &right,
                    arrays: arrays.each_ref().map(Vec::as_slice),
                },
            )
            .unwrap();
        assert_eq!(checked.claims().count(), 8 * m);
        row_count += rows.len();
    }
    assert_eq!(row_count, 572);
}

#[test]
fn producer_nonneg_forms_and_witnesses_match_all_frozen_decompositions() {
    use tv_core::gadgets::decomposition::{Decomposition, Values};
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../spec/vectors/gadget-finals/decomposition.json"
    ))
    .unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 248);
    let tables = range_tables();
    let mut count = 0;
    for case in cases {
        let bits = case["bits"].as_u64().unwrap() as u8;
        let limb = case["limb_bits"].as_u64().unwrap() as u8;
        let producer = NonNeg::new(bits, limb, &tables).unwrap();
        let independent = Decomposition::derive(bits, limb).unwrap();
        let rows = case["rows"].as_array().unwrap();
        let input: Vec<_> = rows
            .iter()
            .map(|row| row["value"].as_i64().unwrap())
            .collect();
        let witness = producer.witness(&input).unwrap();
        let point: Vec<_> = case["point"].as_array().unwrap().iter().map(fp2).collect();
        let low: Vec<_> = (0..producer.limbs() - 1)
            .map(|j| {
                let values = witness.tensor(Committed::Limb(j as u8)).unwrap();
                for (row, &value) in rows.iter().zip(values) {
                    assert_eq!(row["limbs"][j as usize].as_i64(), Some(value));
                }
                values.iter().enumerate().fold(Fp2::ZERO, |out, (r, &v)| {
                    out + eq(&point, r) * field(v as u64)
                })
            })
            .collect();
        let constraints = producer.constraints::<Goldilocks>();
        let keys: Vec<_> = constraints
            .lookups
            .iter()
            .enumerate()
            .map(|(j, lookup)| {
                let values: Vec<_> = input
                    .iter()
                    .enumerate()
                    .map(|(r, &v)| {
                        let key = lookup.key.evaluate::<Goldilocks>(|term| match term {
                            Term::Input => Goldilocks::from_i64(v),
                            Term::Committed(c) => {
                                Goldilocks::from_i64(witness.tensor(c).unwrap()[r])
                            }
                            Term::Class => unreachable!(),
                        });
                        let key = key.to_canonical_u64();
                        assert_eq!(rows[r]["limbs"][j].as_u64(), Some(key));
                        key
                    })
                    .collect();
                let key = values
                    .iter()
                    .enumerate()
                    .fold(Fp2::ZERO, |out, (r, &v)| out + eq(&point, r) * field(v));
                assert_eq!(key, fp2(&case["limbs"][j]));
                key
            })
            .collect();
        let value = input.iter().enumerate().fold(Fp2::ZERO, |out, (r, &v)| {
            out + eq(&point, r) * field(v as u64)
        });
        assert_eq!(value, fp2(&case["value"]));
        let checked = independent
            .check(
                GadgetId(23),
                Values {
                    point: &point,
                    value,
                    low: &low,
                    keys: &keys,
                },
            )
            .unwrap();
        assert_eq!(checked.claims().count(), keys.len() + 1);
        count += input.len();
    }
    assert_eq!(count, 992);
}
