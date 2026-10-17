// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tv-tables` against `spec/vectors/tables/` (RFC-0008 'Testing Strategy'), written by an
//! independent Python oracle that checks every `Exp`, `Exp2Frac` and `Silu` value against a
//! 60-digit reference and the S-box against FIPS-197: the pinned constants, and for every case
//! its TCE specification, digest, row count, `log2_rows`, pieces, the value of every row the
//! file lists, its edge lookups, and RoPE's frequencies and pairs, with the 4,008 samples of
//! Llama-3.1's 131,072-position table.

mod support;

use support::{
    centered, digest_hex, hex_u128, integer, kind, list, member, number, table_case, text,
};
use tv_codec::{DecodeLimits, decode};
use tv_field::Fp;
use tv_tables::{Table, TableKind, TableSpec, fixed, rope};
use tv_testkit::{Json, hex, read_vectors};

/// Every case of `tables.json`.
fn cases() -> Vec<Json> {
    support::table_cases()
}

/// A case's specification, decoded from its TCE bytes.
fn spec(case: &Json) -> TableSpec {
    let bytes = hex(text(case, "spec"));
    decode(&bytes, &DecodeLimits::MESSAGE, TableSpec::decode)
        .unwrap_or_else(|e| panic!("{}: {e}", text(case, "name")))
}

#[test]
fn the_constants_are_the_pinned_ones() {
    let file = read_vectors("tables/constants.json");
    assert_eq!(hex_u128(member(&file, "LN2")), fixed::LN2);
    assert_eq!(hex_u128(member(&file, "TAU")), fixed::TAU);
    assert_eq!(hex_u128(member(&file, "HALF_PI")), fixed::HALF_PI);
}

#[test]
fn the_vectors_cover_every_kind() {
    assert_eq!(cases().len(), 37);
    let mut tags: Vec<u16> = cases().iter().map(|case| spec(case).kind.tag()).collect();
    tags.sort_unstable();
    tags.dedup();
    assert_eq!(tags, (0..10).collect::<Vec<_>>());
}

/// INV-TCE-1 and INV-TCE-2 on every vector: the bytes decode to the case's kind, `log2_rows` and
/// digest, and the value encodes back to the same bytes.
#[test]
fn every_specification_decodes_to_its_kind_and_encodes_back() {
    for case in cases() {
        let name = text(&case, "name");
        let bytes = hex(text(&case, "spec"));
        let spec = spec(&case);
        assert_eq!(spec.kind, kind(member(&case, "kind")), "{name}");
        assert_eq!(
            i128::from(spec.log2_rows),
            number(&case, "log2_rows"),
            "{name}"
        );
        assert_eq!(spec.digest, digest_hex(text(&case, "digest")), "{name}");
        assert_eq!(spec.to_tce(), bytes, "{name}");
        assert!(bytes.len() as u64 >= TableSpec::MIN_SIZE, "{name}");
    }
}

#[test]
fn every_vector_specification_regenerates_to_its_digest() {
    for case in cases() {
        let name = text(&case, "name");
        let spec = spec(&case);
        let table = Table::from_spec(&spec).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(table.spec(), &spec, "{name}");
        assert_eq!(i128::from(table.rows()), number(&case, "rows"), "{name}");
        assert_eq!(
            i128::from(table.log2_rows()),
            number(&case, "log2_rows"),
            "{name}"
        );
        let pieces: Vec<i128> = list(&case, "pieces").iter().map(integer).collect();
        let found: Vec<i128> = table.pieces().into_iter().map(i128::from).collect();
        assert_eq!(found, pieces, "{name}");
        // The table a kind defines under an id is the one the vector names.
        let generated = Table::generate(spec.id, spec.kind.clone()).unwrap();
        assert_eq!(generated.spec(), &spec, "{name}");
    }
}

/// Every value the file lists, and every padded row: rows from `T` on repeat row 0.
#[test]
fn every_listed_value_is_the_regenerated_one() {
    let mut checked = 0;
    for case in cases() {
        let name = text(&case, "name");
        let Some(values) = case.get("values") else {
            continue;
        };
        if matches!(spec(&case).kind, TableKind::Rope { .. }) {
            continue;
        }
        let values: Vec<i128> = values.as_array().unwrap().iter().map(integer).collect();
        let table = Table::from_spec(&spec(&case)).unwrap();
        assert_eq!(values.len() as u64, table.rows(), "{name}");
        for (j, &value) in values.iter().enumerate() {
            let j = j as u64;
            assert_eq!(
                table.value(j).map(i128::from),
                Some(value),
                "{name}, row {j}"
            );
            assert_eq!(
                table.row(j),
                Some((Fp::reduce(j), centered(value))),
                "{name}, row {j}"
            );
        }
        assert_eq!(table.value(table.rows()), None, "{name}");
        let padded = 1u64 << table.log2_rows();
        for j in table.rows()..padded {
            assert_eq!(
                table.row(j),
                Some((Fp::ZERO, centered(values[0]))),
                "{name}, padded row {j}"
            );
        }
        assert_eq!(table.row(padded), None, "{name}");
        checked += values.len();
    }
    assert_eq!(checked, 5089, "the values the file lists");
}

/// `Silu`'s step 1 sets `E = y >> n`, or 0 when `n ≥ 128` (RFC-0008 'Fixed-point generators'):
/// at `in_exp = 0` the argument `|h|·2^64` has `n = 128` exactly at `h = ±89`, where a shift by
/// `n` would leave the 128 bits, and more up to `n = 184` at `h = −128`. The table of every `h`
/// from −128 to 127 is generated and holds the oracle's values there.
#[test]
fn silu_arguments_past_128_ln2_have_their_values() {
    let case = table_case("silu-b8-wide-arguments");
    let spec = spec(&case);
    assert_eq!(
        spec.kind,
        TableKind::Silu {
            b: 8,
            r_by_class: vec![0],
            in_exp: 0,
            out_exp: 0
        }
    );
    assert_eq!(fixed::exp_neg(89 << 64).map(|(_, n)| n), Some(128));
    assert_eq!(fixed::exp_neg(88 << 64).map(|(_, n)| n), Some(126));
    assert_eq!(fixed::exp_neg(128 << 64).map(|(_, n)| n), Some(184));
    let table = Table::from_spec(&spec).unwrap();
    let values: Vec<i128> = list(&case, "values").iter().map(integer).collect();
    for h in [-128i64, -89, -88, 88, 89, 127] {
        let row = u64::try_from(h + 128).unwrap();
        assert_eq!(
            table.value(row).map(i128::from),
            Some(values[row as usize]),
            "h = {h}"
        );
    }
}

/// The row a lookup's key and class hit, by RFC-0008's packing: `(k + 2^{b−1}) + 2^b·c` for
/// `Requant` and `Silu`, the operand itself for the other kinds (`a + 2^8·b` is already one
/// integer for `Bitwise8`), and no row outside the key range or past the classes.
fn packed_row(kind: &TableKind, rows: u64, key: i128, class: i128) -> Option<u64> {
    match kind {
        TableKind::Requant { b, r_by_class, .. } | TableKind::Silu { b, r_by_class, .. } => {
            let half = 1i128 << (b - 1);
            let in_range = (-half..half).contains(&key);
            let in_classes = (0..r_by_class.len() as i128).contains(&class);
            (in_range && in_classes).then(|| ((key + half) + (class << b)) as u64)
        }
        _ => (class == 0 && (0..i128::from(rows)).contains(&key)).then_some(key as u64),
    }
}

#[test]
fn every_lookup_hits_its_row_and_value() {
    let mut checked = 0;
    for case in cases() {
        let name = text(&case, "name");
        let spec = spec(&case);
        let table = Table::from_spec(&spec).unwrap();
        for lookup in list(&case, "lookups") {
            let (key, class) = (number(lookup, "key"), number(lookup, "class"));
            let row = packed_row(&spec.kind, table.rows(), key, class);
            let expected_row = member(lookup, "row");
            let expected_value = member(lookup, "value");
            match row {
                None => {
                    assert!(expected_row.is_null(), "{name}: ({key}, {class})");
                    assert!(expected_value.is_null(), "{name}: ({key}, {class})");
                }
                Some(row) => {
                    assert_eq!(
                        i128::from(row),
                        integer(expected_row),
                        "{name}: ({key}, {class})"
                    );
                    assert_eq!(
                        table.value(row).map(i128::from),
                        Some(integer(expected_value)),
                        "{name}: ({key}, {class})"
                    );
                }
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 292, "the lookups the file lists");
}

#[test]
fn every_rope_table_has_its_frequencies_and_pairs() {
    let mut tables = 0;
    for case in cases() {
        let name = text(&case, "name");
        let spec = spec(&case);
        if !matches!(spec.kind, TableKind::Rope { .. }) {
            continue;
        }
        let table = Table::from_spec(&spec).unwrap();
        let rope = table.rope().unwrap();
        let frequencies: Vec<u128> = list(&case, "frequencies").iter().map(hex_u128).collect();
        assert_eq!(rope.frequencies(), frequencies.as_slice(), "{name}");
        assert_eq!(rope.pairs(), table.rows(), "{name}");
        if let Some(values) = case.get("values") {
            let values = values.as_array().unwrap();
            assert_eq!(values.len() as u64, rope.pairs(), "{name}");
            for (j, pair) in values.iter().enumerate() {
                let pair: Vec<i128> = pair.as_array().unwrap().iter().map(integer).collect();
                let found = rope.row(j as u64).unwrap();
                assert_eq!(
                    vec![i128::from(found.0), i128::from(found.1)],
                    pair,
                    "{name}, row {j}"
                );
            }
            assert_eq!(rope.row(rope.pairs()), None, "{name}");
        }
        // A Rope table holds constants, not lookup rows.
        assert_eq!(table.row(0), None, "{name}");
        assert_eq!(table.value(0), None, "{name}");
        tables += 1;
    }
    assert_eq!(tables, 3);
}

/// Llama-3.1's RoPE at 131,072 positions: 4,008 sampled pairs. The frequencies do not depend on
/// `max_context`, so a one-position table with the same parameters gives them without hashing
/// the 2^23 pairs.
#[test]
fn llama31_samples_at_131072_positions() {
    let file = read_vectors("tables/tables.json");
    let llama = member(&file, "rope_llama31_131072");
    let TableKind::Rope {
        theta,
        factor,
        low_freq_factor,
        high_freq_factor,
        original_max,
        bits,
        max_context,
        d_head,
    } = kind(member(llama, "kind"))
    else {
        panic!("not a Rope kind");
    };
    assert_eq!(max_context, 131_072);
    let one_position = TableKind::Rope {
        theta,
        factor,
        low_freq_factor,
        high_freq_factor,
        original_max,
        bits,
        max_context: 1,
        d_head,
    };
    let table = Table::generate(tv_tables::TableId(0), one_position).unwrap();
    let frequencies = table.rope().unwrap().frequencies().to_vec();
    assert_eq!(frequencies.len(), usize::from(d_head / 2));
    // The 8,192-position table of the vectors has the same frequencies.
    let short = cases()
        .into_iter()
        .find(|case| text(case, "name") == "rope-llama31-8k")
        .unwrap();
    let listed: Vec<u128> = list(&short, "frequencies").iter().map(hex_u128).collect();
    assert_eq!(frequencies, listed);
    let samples = list(llama, "samples");
    assert_eq!(samples.len(), 4008);
    for sample in samples {
        let position = u32::try_from(number(sample, "position")).unwrap();
        let frequency = usize::try_from(number(sample, "frequency")).unwrap();
        let expected: Vec<i128> = list(sample, "constants").iter().map(integer).collect();
        let (c, s) = rope::constants(frequencies[frequency], position, bits).unwrap();
        assert_eq!(
            vec![i128::from(c), i128::from(s)],
            expected,
            "position {position}, frequency {frequency}"
        );
    }
}
