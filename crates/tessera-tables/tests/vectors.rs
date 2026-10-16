// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The arithmetic kinds against `spec/vectors/tables/` (RFC-0008 'Table specifications'). Its
//! oracle is independent and checks the AES tables against FIPS-197 before writing anything.
//!
//! Each specification decodes, encodes back to its own bytes and regenerates with its digest;
//! its rows, padding included, are the oracle's; and a lookup at either edge of its key range,
//! or just past one, hits the oracle's row or none. A specification wrong in any one field is
//! refused with the error for that field, and so are parameters just outside each rule.

use std::fs;
use std::path::PathBuf;

use serde_json::Value;
use tessera_codec::{DecodeLimits, Rational, decode, encode};
use tessera_field::{Goldilocks, PrimeField};
use tessera_tables::{BitOp, GeneratedTable, Table, TableError, TableId, TableKind, TableSpec};

fn cases() -> Vec<Value> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/tables/tables.json");
    let document: Value =
        serde_json::from_str(&fs::read_to_string(path).expect("the vectors")).expect("JSON");
    document["cases"].as_array().expect("cases").clone()
}

fn bytes(hex: &Value) -> Vec<u8> {
    let hex = hex.as_str().expect("a hex string");
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex"))
        .collect()
}

fn spec_of(case: &Value) -> TableSpec {
    decode(&bytes(&case["spec"]), DecodeLimits::MESSAGE).expect("the specification decodes")
}

fn integer(value: &Value) -> i64 {
    value.as_i64().expect("an integer")
}

#[test]
fn every_vector_regenerates_with_its_digest_rows_and_lookups() {
    let cases = cases();
    assert_eq!(cases.len(), 33);
    for case in &cases {
        let name = case["name"].as_str().expect("a name");
        let spec = spec_of(case);
        assert_eq!(encode(&spec), bytes(&case["spec"]), "{name}: canonical");
        assert_eq!(spec.digest.to_vec(), bytes(&case["digest"]), "{name}");
        let table = GeneratedTable::from_spec(&spec).expect(name);
        assert_eq!(table.spec(), &spec, "{name}");
        let rows = case["rows"].as_u64().expect("rows");
        assert_eq!(table.row_count(), rows, "{name}");
        assert_eq!(
            u64::from(table.log2_rows()),
            case["log2_rows"].as_u64().expect("log2")
        );
        if let Some(values) = case["values"].as_array() {
            assert_eq!(values.len() as u64, rows, "{name}");
            for (j, value) in (0u64..).zip(values) {
                let expected = (
                    Goldilocks::from_u64_reduced(j),
                    Goldilocks::from_i64(integer(value)),
                );
                assert_eq!(table.row(j), Some(expected), "{name}: row {j}");
            }
        }
        // Rows T to 2^log2_rows − 1 repeat row 0, and nothing lies past them.
        let padded = 1u64 << table.log2_rows();
        for index in [rows, (rows + padded) / 2, padded - 1] {
            if index >= rows && index < padded {
                assert_eq!(
                    table.row(index),
                    table.row(0),
                    "{name}: padding row {index}"
                );
            }
        }
        assert_eq!(table.row(padded), None, "{name}: past the padding");
        for lookup in case["lookups"].as_array().expect("lookups") {
            let key = integer(&lookup["key"]);
            let class = u8::try_from(integer(&lookup["class"])).expect("a class");
            let row = lookup["row"].as_u64();
            let value = lookup["value"].as_i64();
            assert_eq!(
                table.index(key, class),
                row,
                "{name}: index of {key} in {class}"
            );
            assert_eq!(
                table.lookup(key, class),
                value,
                "{name}: lookup of {key} in {class}"
            );
            if let (Some(row), Some(value)) = (row, value) {
                assert_eq!(
                    table.row(row),
                    Some((
                        Goldilocks::from_u64_reduced(row),
                        Goldilocks::from_i64(value)
                    )),
                    "{name}: the row a lookup hits holds its key and value"
                );
            }
        }
    }
}

/// Claim-list validation regenerates a specification from its kind and id (RFC-0008 'Table
/// specifications'): changing any one field of a pinned one is refused, the parameters and id
/// through the digest they enter.
#[test]
fn a_specification_wrong_in_one_field_is_refused() {
    for case in cases() {
        let spec = spec_of(&case);
        let mut digest = spec.clone();
        digest.digest[31] ^= 1;
        assert_eq!(GeneratedTable::from_spec(&digest), Err(TableError::Digest));
        for stated in [spec.log2_rows.wrapping_sub(1), spec.log2_rows + 1] {
            let wrong = TableSpec {
                log2_rows: stated,
                ..spec.clone()
            };
            assert_eq!(
                GeneratedTable::from_spec(&wrong),
                Err(TableError::Log2Rows {
                    stated,
                    expected: spec.log2_rows,
                })
            );
        }
        let id = TableSpec {
            id: TableId(spec.id.0 + 1),
            ..spec.clone()
        };
        assert_eq!(GeneratedTable::from_spec(&id), Err(TableError::Digest));
    }
    // A parameter changed within what the kind allows leaves the digest of the old one.
    let requant = cases()
        .iter()
        .map(spec_of)
        .find(|s| matches!(s.kind, TableKind::Requant { .. }))
        .expect("a Requant vector");
    let mut changed = requant.clone();
    if let TableKind::Requant { lo, .. } = &mut changed.kind {
        *lo -= 1;
    }
    assert_eq!(GeneratedTable::from_spec(&changed), Err(TableError::Digest));
}

/// Each rule of RFC-0008 'Table specifications', at its edges: the last parameters inside are
/// built, the first outside refused with the rule.
#[test]
fn parameters_just_outside_each_rule_are_refused() {
    let build = |kind| GeneratedTable::new(TableId(0), kind).map(|t| t.row_count());
    let refused = |kind, rule| Err(TableError::Parameter { kind, rule });
    let requant = |b, r_by_class: &[u8], lo, hi| TableKind::Requant {
        b,
        r_by_class: r_by_class.to_vec(),
        lo,
        hi,
    };
    let range_rule = "1 ≤ bits ≤ 32";
    for bits in [0, 33, u8::MAX] {
        assert_eq!(
            build(TableKind::Range { bits }),
            refused("Range", range_rule)
        );
    }
    assert_eq!(build(TableKind::Range { bits: 1 }), Ok(2));
    assert_eq!(build(TableKind::Range { bits: 32 }), Ok(1 << 32));

    let classes = "one to four classes";
    assert_eq!(build(requant(4, &[], -1, 1)), refused("Requant", classes));
    assert_eq!(
        build(requant(4, &[1; 5], -1, 1)),
        refused("Requant", classes)
    );
    let shape = "b ≥ 1 and b + ⌈log2 n⌉ ≤ 32 for n classes";
    assert_eq!(build(requant(0, &[1], -1, 1)), refused("Requant", shape));
    assert_eq!(build(requant(33, &[1], -1, 1)), refused("Requant", shape));
    assert_eq!(
        build(requant(32, &[1, 1], -1, 1)),
        refused("Requant", shape)
    );
    assert_eq!(
        build(requant(31, &[1, 1, 1], -1, 1)),
        refused("Requant", shape)
    );
    assert_eq!(build(requant(32, &[1], -1, 1)), Ok(1 << 32));
    assert_eq!(build(requant(31, &[1, 1], -1, 1)), Ok(1 << 32));
    assert_eq!(build(requant(30, &[1; 4], -1, 1)), Ok(1 << 32));
    let shifts = "1 ≤ r_c ≤ 63 for every class";
    assert_eq!(
        build(requant(4, &[1, 0], -1, 1)),
        refused("Requant", shifts)
    );
    assert_eq!(
        build(requant(4, &[64, 1], -1, 1)),
        refused("Requant", shifts)
    );
    assert_eq!(build(requant(4, &[1, 63], -1, 1)), Ok(32));
    assert_eq!(build(requant(4, &[1], 2, 1)), refused("Requant", "lo ≤ hi"));
    assert_eq!(build(requant(4, &[1], 1, 1)), Ok(16));

    let pow2 = "1 ≤ max_exp ≤ 63";
    for max_exp in [0, 64] {
        assert_eq!(build(TableKind::Pow2 { max_exp }), refused("Pow2", pow2));
    }
    assert_eq!(build(TableKind::Pow2 { max_exp: 63 }), Ok(63));
    assert_eq!(build(TableKind::Bitwise8 { op: BitOp::And }), Ok(1 << 16));

    let scale = |num, den| Rational::new(num, den).expect("a fraction");
    let exp = |key_bits, sigma, scale| TableKind::Exp {
        key_bits,
        sigma,
        scale,
    };
    for key_bits in [0, 33] {
        assert_eq!(
            build(exp(key_bits, 16, scale(1, 4))),
            refused("Exp", "1 ≤ key_bits ≤ 32")
        );
    }
    assert_eq!(build(exp(4, 33, scale(1, 4))), refused("Exp", "σ ≤ 32"));
    for num in [0, -1] {
        assert_eq!(
            build(exp(4, 16, scale(num, 4))),
            refused("Exp", "a positive scale")
        );
    }
    assert_eq!(build(exp(4, 32, scale(1, 4))), Ok(16));
    // The zero tail must start by key 2^24; at σ = 16 it starts near 18·ln 2 / scale. The
    // check comes before the core is generated, so a refused table costs nothing.
    let tail = "a zero tail from key 2^24 or earlier";
    assert_eq!(build(exp(26, 16, scale(1, 1 << 21))), refused("Exp", tail));
    assert_eq!(
        build(exp(26, 16, scale(1, 2896))),
        Ok(1 << 26),
        "the 405B-like tail"
    );

    for frac_bits in [0, 17] {
        assert_eq!(
            build(TableKind::Exp2Frac {
                frac_bits,
                out_bits: 16
            }),
            refused("Exp2Frac", "1 ≤ frac_bits ≤ 16")
        );
    }
    for out_bits in [0, 57] {
        assert_eq!(
            build(TableKind::Exp2Frac {
                frac_bits: 8,
                out_bits
            }),
            refused("Exp2Frac", "1 ≤ out_bits ≤ 56")
        );
    }
    assert_eq!(
        build(TableKind::Exp2Frac {
            frac_bits: 16,
            out_bits: 56
        }),
        Ok(1 << 16)
    );

    let silu = |b, r_by_class: &[u8], in_exp, out_exp| TableKind::Silu {
        b,
        r_by_class: r_by_class.to_vec(),
        in_exp,
        out_exp,
    };
    assert_eq!(
        build(silu(6, &[], -3, -4)),
        refused("Silu", "one to four classes")
    );
    assert_eq!(
        build(silu(6, &[0; 5], -3, -4)),
        refused("Silu", "one to four classes")
    );
    let shape = "b ≥ 1 and b + ⌈log2 n⌉ ≤ 32 for n classes";
    assert_eq!(build(silu(0, &[0], -3, -4)), refused("Silu", shape));
    assert_eq!(build(silu(32, &[20, 20], -3, -4)), refused("Silu", shape));
    assert_eq!(
        build(silu(6, &[1, 6], -3, -4)),
        refused("Silu", "r_c < b for every class")
    );
    assert_eq!(
        build(silu(6, &[5], -3, -4)),
        Ok(64),
        "r_c = b − 1: h is −1 or 0"
    );
    // The values of h are counted before any is generated.
    assert_eq!(
        build(silu(26, &[1], -3, -4)),
        refused("Silu", "at most 2^24 values of h over the classes")
    );
    assert_eq!(build(silu(30, &[14, 15, 16, 17], -10, -4)), Ok(1 << 32));
    for in_exp in [-65, 1] {
        assert_eq!(
            build(silu(6, &[0], in_exp, -4)),
            refused("Silu", "−64 ≤ in_exp ≤ 0")
        );
    }
    let shift = "1 ≤ 63 − in_exp + out_exp ≤ 126";
    assert_eq!(build(silu(6, &[0], 0, -63)), refused("Silu", shift));
    assert_eq!(build(silu(6, &[0], -64, 0)), refused("Silu", shift));
    assert_eq!(build(silu(6, &[0], 0, -62)), Ok(64));
    assert_eq!(build(silu(6, &[0], -64, -1)), Ok(64));
}

/// The witness-side lookup computes what the rows hold for every key of the byte tables and of a
/// four-class `Requant`, and refuses the classes past the last.
#[test]
fn lookups_agree_with_the_rows_for_every_key() {
    let kinds = [
        TableKind::AesSbox,
        TableKind::Gf256Mul { c: 0x1b },
        TableKind::Bitwise8 { op: BitOp::Xor },
        TableKind::Requant {
            b: 6,
            r_by_class: vec![1, 3, 5, 40],
            lo: -9,
            hi: 7,
        },
        TableKind::Exp {
            key_bits: 12,
            sigma: 16,
            scale: Rational::new(1, 100).expect("a fraction"),
        },
        TableKind::Exp2Frac {
            frac_bits: 8,
            out_bits: 16,
        },
        TableKind::Silu {
            b: 7,
            r_by_class: vec![0, 2, 3],
            in_exp: -4,
            out_exp: -3,
        },
    ];
    for kind in kinds {
        let table = GeneratedTable::new(TableId(7), kind.clone()).expect("a table");
        let (offset, classes, width) = match &kind {
            TableKind::Requant { b, r_by_class, .. } | TableKind::Silu { b, r_by_class, .. } => {
                (1i64 << (b - 1), r_by_class.len(), 1u64 << b)
            }
            _ => (0, 1, table.row_count()),
        };
        for class in 0..classes {
            let class = u8::try_from(class).expect("a class");
            for x in 0..width {
                let key = i64::try_from(x).expect("a key") - offset;
                let row = table.index(key, class).expect("a key of the table");
                assert_eq!(row, x + width * u64::from(class), "{kind:?}");
                let value = table.lookup(key, class).map(Goldilocks::from_i64);
                assert_eq!(table.row(row).map(|(_, v)| v), value, "{kind:?}: {key}");
            }
        }
        let past = u8::try_from(classes).expect("a class");
        assert_eq!(table.index(0, past), None, "{kind:?}: class {past}");
        assert_eq!(table.lookup(0, past), None, "{kind:?}: class {past}");
    }
}
