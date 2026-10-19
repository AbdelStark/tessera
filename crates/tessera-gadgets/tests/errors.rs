// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Every check of the crate, each broken alone, refused with its exact error, all
//! `ERR-INTERNAL` (04 §3): a gadget that does not match its tables, a parameter outside its
//! lemma's preconditions, a column missing or of the wrong length, an input outside the gadget's
//! domain, a lookup's key that names no row, and a table of more rows than the field has
//! elements. Each refused case sits beside an accepted one at the boundary, so that the check
//! does not refuse more than it must.

mod common;

use common::Fp;
use tessera_codec::SecretBuf;
use tessera_codec::error::ErrorCode;
use tessera_compiler::Gadget;
use tessera_field::{F65537, PrimeField};
use tessera_gadgets::{
    Column, Committed, GadgetError, Lookups, NonNeg, Requant, TableColumns, Witness,
};
use tessera_tables::{GeneratedTable, Table, TableError, TableId, TableKind, TableSpec};

fn spec(id: u32, kind: TableKind) -> TableSpec {
    GeneratedTable::new(TableId(id), kind.clone())
        .unwrap_or_else(|e| panic!("{kind:?}: {e}"))
        .spec()
        .clone()
}

fn range(id: u32, bits: u8) -> TableSpec {
    spec(id, TableKind::Range { bits })
}

fn rq(id: u32, b: u8, r_by_class: &[u8], lo: i32, hi: i32) -> TableSpec {
    spec(
        id,
        TableKind::Requant {
            b,
            r_by_class: r_by_class.to_vec(),
            lo,
            hi,
        },
    )
}

fn table(id: u32, rule: &'static str) -> GadgetError {
    GadgetError::Table {
        table: TableId(id),
        rule,
    }
}

fn shape(rule: &'static str) -> GadgetError {
    GadgetError::Shape { rule }
}

/// `Requant::new`'s verdict, its error's code checked.
fn requant(gadget: Gadget, specs: &[TableSpec]) -> Result<Requant, GadgetError> {
    Requant::new(&gadget, specs).inspect_err(|e| assert_eq!(e.code(), ErrorCode::Internal))
}

const MISSING: &str = "the specifications hold every table the gadget names";

#[test]
fn requant1_reads_a_requant_table_among_the_specifications() {
    let gadget = Gadget::Requant1 { table: TableId(0) };
    assert!(requant(gadget, &[rq(0, 6, &[3], -4, 3)]).is_ok());
    assert_eq!(
        requant(gadget, &[rq(1, 6, &[3], -4, 3)]).err(),
        Some(table(0, MISSING))
    );
    assert_eq!(
        requant(gadget, &[range(0, 6)]).err(),
        Some(table(0, "Requant1 reads a Requant table"))
    );
    // A specification that does not regenerate: another digest, another row count.
    let mut tampered = rq(0, 6, &[3], -4, 3);
    tampered.digest[0] ^= 1;
    assert_eq!(
        requant(gadget, &[tampered]).err(),
        Some(GadgetError::Spec {
            table: TableId(0),
            error: TableError::Digest
        })
    );
    let mut tampered = rq(0, 6, &[3], -4, 3);
    tampered.log2_rows += 1;
    assert_eq!(
        requant(gadget, &[tampered]).err(),
        Some(GadgetError::Spec {
            table: TableId(0),
            error: TableError::Log2Rows {
                stated: 7,
                expected: 6
            }
        })
    );
}

#[test]
fn only_requantization_gadgets_are_taken() {
    let refused = shape("a requantization gadget: Requant1, RequantSplit or RequantWide");
    for gadget in [
        Gadget::Lookup { table: TableId(0) },
        Gadget::SoftmaxRow {
            exp: TableId(0),
            reciprocal: TableId(0),
            limb: None,
            top: TableId(0),
        },
        Gadget::RsqrtRow,
        Gadget::OneHot,
        Gadget::Argmax,
        Gadget::InverseCdf,
    ] {
        assert_eq!(
            requant(gadget, &[rq(0, 6, &[3], -4, 3)]).err(),
            Some(refused),
            "{gadget:?}"
        );
    }
}

#[test]
fn requant_split_reads_its_two_tables_and_a_key_of_at_most_62_bits() {
    let split = |split| Gadget::RequantSplit {
        split,
        low: TableId(0),
        high: TableId(1),
    };
    let high = rq(1, 6, &[3], -50, 77);
    assert!(requant(split(6), &[range(0, 6), high.clone()]).is_ok());
    assert_eq!(
        requant(split(6), core::slice::from_ref(&high)).err(),
        Some(table(0, MISSING))
    );
    assert_eq!(
        requant(split(6), &[range(0, 6)]).err(),
        Some(table(1, MISSING))
    );
    let low = "RequantSplit's low part reads Range { s }";
    assert_eq!(
        requant(split(6), &[range(0, 5), high.clone()]).err(),
        Some(table(0, low))
    );
    assert_eq!(
        requant(split(6), &[rq(0, 6, &[1], 0, 0), high.clone()]).err(),
        Some(table(0, low))
    );
    assert_eq!(
        requant(split(6), &[range(0, 6), range(1, 6)]).err(),
        Some(table(1, "RequantSplit's high part reads a Requant table"))
    );
    // b = s + (b − s): 30 + 32 = 62 is the most Lemma 2 takes.
    let widest = rq(1, 32, &[1], -128, 127);
    assert!(requant(split(30), &[range(0, 30), widest.clone()]).is_ok());
    assert_eq!(
        requant(split(31), &[range(0, 31), widest]).err(),
        Some(shape("a split key has at most 62 bits (RFC-0009 Lemma 2)"))
    );
}

#[test]
fn requant_split_keeps_n_2_b_below_p() {
    // RFC-0009 Lemma 2 compares `z_lo + 2^s·j` and `z_off + 2^b·c`, both in [0, n·2^b), so it
    // needs n·2^b < P. Four classes at 62 bits reach it, 4·2^62 = 2^64 > P; three classes at 62
    // bits, 3·2^62, and four at 61, 2^63, stay below, at either split point.
    let split = |split| Gadget::RequantSplit {
        split,
        low: TableId(0),
        high: TableId(1),
    };
    let four = [1, 2, 3, 4];
    assert_eq!(
        requant(split(32), &[range(0, 32), rq(1, 30, &four, -128, 127)]).err(),
        Some(shape(
            "a split key of n classes has n·2^b < P (RFC-0009 Lemma 2)"
        ))
    );
    assert!(requant(split(32), &[range(0, 32), rq(1, 30, &four[..3], -128, 127)]).is_ok());
    assert!(requant(split(32), &[range(0, 32), rq(1, 29, &four, -128, 127)]).is_ok());
    assert!(requant(split(31), &[range(0, 31), rq(1, 30, &four, -128, 127)]).is_ok());
}

/// `RequantWide` with s = 2, t = 3, k = 9 for one class of shift 3 and bounds ±3: the reduced
/// shift is 1 and `⌈log2 3⌉ = 2`, so `t = 1 + 2`.
fn wide_specs() -> Vec<TableSpec> {
    vec![
        range(0, 2),
        range(1, 3),
        range(2, 9),
        rq(3, 10, &[1], -2, 1),
        rq(4, 5, &[1], -3, 3),
    ]
}

fn wide(split: u8, middle: u8, low: Option<u32>) -> Gadget {
    Gadget::RequantWide {
        split,
        middle,
        low: low.map(TableId),
        mid: TableId(1),
        top: TableId(2),
        sign: TableId(3),
        high: TableId(4),
    }
}

/// `specs` with entry `index` replaced by `replacement`.
fn replaced(specs: &[TableSpec], index: usize, replacement: TableSpec) -> Vec<TableSpec> {
    let mut specs = specs.to_vec();
    specs[index] = replacement;
    specs
}

#[test]
fn requant_wide_reads_its_five_tables_in_their_roles() {
    let specs = wide_specs();
    assert!(requant(wide(2, 3, Some(0)), &specs).is_ok());
    // The low part's table exactly when s ≥ 1.
    let presence = shape("RequantWide's low part has a Range table exactly when s ≥ 1");
    assert_eq!(requant(wide(2, 3, None), &specs).err(), Some(presence));
    assert_eq!(requant(wide(0, 3, Some(0)), &specs).err(), Some(presence));
    // Each table missing, then each of the wrong kind.
    for (index, id) in [(0usize, 0u32), (1, 1), (2, 2), (3, 3), (4, 4)] {
        let mut without = specs.clone();
        without.remove(index);
        assert_eq!(
            requant(wide(2, 3, Some(0)), &without).err(),
            Some(table(id, MISSING))
        );
    }
    let cases = [
        (0, range(0, 3), "RequantWide's low part reads Range { s }"),
        (
            1,
            range(1, 4),
            "RequantWide's middle part reads Range { t }",
        ),
        (
            2,
            rq(2, 9, &[1], -2, 1),
            "RequantWide's top part reads a Range table",
        ),
        (
            3,
            rq(3, 10, &[1], -2, 2),
            "RequantWide's saturation reads Requant { k + 1, [1], −2, 1 }",
        ),
        (
            3,
            rq(3, 9, &[1], -2, 1),
            "RequantWide's saturation reads Requant { k + 1, [1], −2, 1 }",
        ),
        (
            3,
            rq(3, 10, &[2], -2, 1),
            "RequantWide's saturation reads Requant { k + 1, [1], −2, 1 }",
        ),
        (4, range(4, 5), "RequantWide's output reads a Requant table"),
        (
            4,
            rq(4, 6, &[1], -3, 3),
            "RequantWide's output reads Requant { t + 2, r_c − s, lo, hi }",
        ),
    ];
    for (index, replacement, rule) in cases {
        let id = replacement.id.0;
        assert_eq!(
            requant(wide(2, 3, Some(0)), &replaced(&specs, index, replacement)).err(),
            Some(table(id, rule)),
            "{rule}"
        );
    }
}

#[test]
fn requant_wide_keeps_lemma_11s_preconditions() {
    // b = s + t + k: 30 + 1 + 31 = 62 is the most Lemma 11 takes, with t = 1 + ⌈log2 1⌉.
    let widest = |split: u8| {
        vec![
            range(0, split),
            range(1, 1),
            range(2, 31),
            rq(3, 32, &[1], -2, 1),
            rq(4, 3, &[1], 0, 0),
        ]
    };
    assert!(requant(wide(30, 1, Some(0)), &widest(30)).is_ok());
    assert_eq!(
        requant(wide(31, 1, Some(0)), &widest(31)).err(),
        Some(shape("a wide key has at most 62 bits (RFC-0024 Lemma 11)"))
    );
    // The middle part has exactly RFC-0024's width: one bit less or more is refused.
    let middle = shape("RequantWide's middle part has t = (r_max − s) + ⌈log2 M⌉ bits (RFC-0024)");
    for t in [2u8, 4] {
        let specs = vec![
            range(0, 2),
            range(1, t),
            range(2, 9),
            rq(3, 10, &[1], -2, 1),
            rq(4, t + 2, &[1], -3, 3),
        ];
        assert_eq!(
            requant(wide(2, t, Some(0)), &specs).err(),
            Some(middle),
            "t = {t}"
        );
    }
    // The largest reduced shift and the larger bound set it: shifts [1, 2] need one more bit.
    let classes = |r: &[u8], (lo, hi): (i32, i32)| {
        vec![
            range(0, 2),
            range(1, 3),
            range(2, 9),
            rq(3, 10, &[1], -2, 1),
            rq(4, 5, r, lo, hi),
        ]
    };
    assert!(requant(wide(2, 3, Some(0)), &classes(&[1, 1], (-3, 2))).is_ok());
    assert_eq!(
        requant(wide(2, 3, Some(0)), &classes(&[1, 2], (-3, 3))).err(),
        Some(middle)
    );
    assert_eq!(
        requant(wide(2, 3, Some(0)), &classes(&[1], (-5, 3))).err(),
        Some(middle)
    );
}

#[test]
fn non_neg_takes_its_widths_and_finds_its_two_range_tables() {
    let ranges = [range(0, 4), range(1, 2)];
    assert!(NonNeg::new(10, 4, &ranges).is_ok());
    let bits = shape("NonNeg proves v ∈ [0, 2^k) for 1 ≤ k ≤ 62 (RFC-0009 Lemma 3)");
    assert_eq!(NonNeg::new(0, 4, &ranges).err(), Some(bits));
    assert_eq!(
        NonNeg::new(63, 16, &[range(0, 16), range(1, 15)]).err(),
        Some(bits)
    );
    assert!(NonNeg::new(62, 16, &[range(0, 16), range(1, 14)]).is_ok());
    assert!(NonNeg::new(1, 4, &[range(0, 1)]).is_ok());
    assert_eq!(
        NonNeg::new(10, 0, &ranges).err(),
        Some(shape("a NonNeg limb has at least one bit"))
    );
    let missing = shape("the specifications hold NonNeg's Range { L } and Range { k − (n−1)L }");
    assert_eq!(NonNeg::new(10, 4, &[range(1, 2)]).err(), Some(missing));
    assert_eq!(NonNeg::new(10, 4, &[range(0, 4)]).err(), Some(missing));
    let mut tampered = range(1, 2);
    tampered.digest[31] ^= 1;
    assert_eq!(
        NonNeg::new(10, 4, &[range(0, 4), tampered]).err(),
        Some(GadgetError::Spec {
            table: TableId(1),
            error: TableError::Digest
        })
    );
}

/// The gadgets the witness tests use, with their key widths: Requant1 of four classes,
/// RequantSplit of two and RequantWide of one.
fn gadgets() -> Vec<(Requant, u32, usize)> {
    let one = Requant::new(
        &Gadget::Requant1 { table: TableId(0) },
        &[rq(0, 6, &[1, 2, 3, 4], -20, 20)],
    )
    .expect("Requant1");
    let split = Requant::new(
        &Gadget::RequantSplit {
            split: 6,
            low: TableId(0),
            high: TableId(1),
        },
        &[range(0, 6), rq(1, 6, &[1, 3], -50, 77)],
    )
    .expect("RequantSplit");
    let wide = Requant::new(&wide(2, 3, Some(0)), &wide_specs()).expect("RequantWide");
    vec![(one, 6, 4), (split, 12, 2), (wide, 14, 1)]
}

#[test]
fn a_witness_takes_one_class_per_key_and_keys_inside_their_width() {
    let domain = GadgetError::Input {
        rule: "every key lies in [−2^{b−1}, 2^{b−1}) and every class below n (INV-IR-2)",
    };
    for (gadget, b, classes) in gadgets() {
        let half = 1i64 << (b - 1);
        let ends = [-half, half - 1];
        let zeros = vec![0u8; 2];
        let given = (classes > 1).then_some(&zeros[..]);
        assert!(gadget.witness(&ends, given).is_ok(), "b = {b}");
        if classes > 1 {
            assert_eq!(
                gadget.witness(&ends, None).err(),
                Some(GadgetError::Missing {
                    column: Column::Classes
                })
            );
        } else {
            // One class: the classes may be left out, or be zeros.
            assert!(gadget.witness(&ends, Some(&zeros)).is_ok());
        }
        for (classes, len) in [(&[0][..], 1), (&[0, 0, 0][..], 3)] {
            assert_eq!(
                gadget.witness(&ends, Some(classes)).err(),
                Some(GadgetError::Length {
                    column: Column::Classes,
                    len,
                    expected: 2
                })
            );
        }
        // Keys one past either end, and far outside, which must not overflow.
        for key in [-half - 1, half, i64::MIN, i64::MAX] {
            let error = gadget
                .witness(&[0, key], given)
                .expect_err("outside the width");
            assert_eq!(error, domain, "b = {b}, z = {key}");
            assert_eq!(error.code(), ErrorCode::Internal);
        }
        // A class past the last.
        let last = u8::try_from(classes).expect("small");
        assert_eq!(
            gadget.witness(&ends, Some(&[0, last])).err(),
            Some(domain),
            "b = {b}"
        );
        assert!(gadget.witness(&ends, Some(&[0, last - 1])).is_ok());
    }
}

#[test]
fn a_non_neg_witness_takes_values_in_its_range() {
    let gadget = NonNeg::new(10, 4, &[range(0, 4), range(1, 2)]).expect("NonNeg");
    assert!(gadget.witness(&[0, 1023]).is_ok());
    for v in [-1, 1024, i64::MIN, i64::MAX] {
        assert_eq!(
            gadget.witness(&[5, v]).err(),
            Some(GadgetError::Input {
                rule: "every value lies in [0, 2^k) (RFC-0009 Lemma 3)"
            }),
            "v = {v}"
        );
    }
}

/// A copy of `witness` without `drop`, and with `short` one entry shorter.
fn edited(
    witness: &Witness,
    tensors: &[Committed],
    drop: Option<Committed>,
    short: Option<Committed>,
) -> Witness {
    Witness::from_tensors(
        tensors
            .iter()
            .filter(|&&t| Some(t) != drop)
            .map(|&t| {
                let values = witness.tensor(t).expect("a tensor");
                let len = values.len() - usize::from(Some(t) == short);
                (t, values[..len].iter().copied().collect::<SecretBuf<i64>>())
            })
            .collect(),
    )
}

#[test]
fn lookups_take_every_column_they_read_at_the_keys_length() {
    let split = &gadgets()[1].0;
    let keys = [-2048, -1, 0, 2047];
    let classes = [0u8, 1, 1, 0];
    let witness = split.witness(&keys, Some(&classes)).expect("a witness");
    let constraints = split.constraints::<Fp>();
    let tensors = constraints.committed.clone();
    assert!(Lookups::new(&constraints, &keys, Some(&classes), &witness).is_ok());
    assert_eq!(
        Lookups::new(&constraints, &keys, None, &witness).err(),
        Some(GadgetError::Missing {
            column: Column::Classes
        })
    );
    for (given, len) in [(&classes[..3], 3), (&[0, 1, 1, 0, 1][..], 5)] {
        assert_eq!(
            Lookups::new(&constraints, &keys, Some(given), &witness).err(),
            Some(GadgetError::Length {
                column: Column::Classes,
                len,
                expected: 4
            })
        );
    }
    for tensor in [Committed::ZLo, Committed::Y] {
        let without = edited(&witness, &tensors, Some(tensor), None);
        assert_eq!(
            Lookups::new(&constraints, &keys, Some(&classes), &without).err(),
            Some(GadgetError::Missing {
                column: Column::Tensor(tensor)
            })
        );
        let short = edited(&witness, &tensors, None, Some(tensor));
        assert_eq!(
            Lookups::new(&constraints, &keys, Some(&classes), &short).err(),
            Some(GadgetError::Length {
                column: Column::Tensor(tensor),
                len: 3,
                expected: 4
            })
        );
        // A column of one entry more than the keys.
        let long = edited(&witness, &tensors, None, None);
        let extra = &keys[..3];
        assert_eq!(
            Lookups::new(&constraints, extra, Some(&classes[..3]), &long).err(),
            Some(GadgetError::Length {
                column: Column::Tensor(Committed::ZLo),
                len: 4,
                expected: 3
            })
        );
    }
    // A description whose lookups into one table disagree on values.
    let mut mixed = constraints.clone();
    mixed.lookups[1].table = mixed.lookups[0].table;
    assert_eq!(
        Lookups::new(&mixed, &keys, Some(&classes), &witness).err(),
        Some(shape(
            "the lookups into one table all have values, or none has"
        ))
    );
    // A gadget of one class reads no class column.
    let wide = &gadgets()[2].0;
    let witness = wide.witness(&[0, 1], None).expect("a witness");
    assert!(Lookups::new(&wide.constraints::<Fp>(), &[0, 1], None, &witness).is_ok());
}

#[test]
fn multiplicities_and_instances_take_every_table_and_keys_that_name_rows() {
    let split = &gadgets()[1].0;
    let keys = [-2048, -1, 0, 2047];
    let classes = [0u8, 1, 1, 0];
    let witness = split.witness(&keys, Some(&classes)).expect("a witness");
    let constraints = split.constraints::<Fp>();
    let lookups = Lookups::new(&constraints, &keys, Some(&classes), &witness).expect("columns");
    let columns: Vec<TableColumns<Fp>> = split
        .tables()
        .into_iter()
        .map(|t| TableColumns::new(t).expect("columns"))
        .collect();
    let multiplicities = lookups.multiplicities(&columns).expect("honest rows");
    assert!(lookups.instances(&columns, &multiplicities).is_ok());
    let held = "the caller holds the columns of every table the lookups read";
    assert_eq!(
        lookups.multiplicities(&columns[1..]).err(),
        Some(table(0, held))
    );
    assert_eq!(
        lookups.instances(&columns[..1], &multiplicities).err(),
        Some(table(1, held))
    );
    assert_eq!(
        lookups.instances(&columns, &multiplicities[..1]).err(),
        Some(GadgetError::Length {
            column: Column::Multiplicities,
            len: 1,
            expected: 2
        })
    );
    let mut more = lookups.multiplicities(&columns).expect("honest rows");
    more.push(SecretBuf::filled(Fp::from_u64_reduced(0), 1));
    assert_eq!(
        lookups.instances(&columns, &more).err(),
        Some(GadgetError::Length {
            column: Column::Multiplicities,
            len: 3,
            expected: 2
        })
    );
    // Keys that name no row: `z_lo + 2^s`, one past Range { 6 }; `z_lo + 1`, whose high key is
    // no integer and far past `2^32`.
    let unnamed = "every lookup's key names a row of its table";
    for (delta, refused) in [(64, 0), (1, 1)] {
        let tensors = constraints
            .committed
            .iter()
            .map(|&t| {
                let mut values: SecretBuf<i64> = witness
                    .tensor(t)
                    .expect("a tensor")
                    .iter()
                    .copied()
                    .collect();
                if t == Committed::ZLo {
                    values[2] += delta;
                }
                (t, values)
            })
            .collect();
        let cheat = Witness::from_tensors(tensors);
        let lookups = Lookups::new(&constraints, &keys, Some(&classes), &cheat).expect("columns");
        assert_eq!(
            lookups.multiplicities(&columns).err(),
            Some(table(refused, unnamed)),
            "z_lo + {delta}"
        );
    }
}

#[test]
fn a_tables_columns_need_distinct_keys_in_their_field() {
    // 2^16 rows fit F65537's 65537 elements; 2^17 do not.
    let fits = GeneratedTable::new(TableId(7), TableKind::Range { bits: 16 }).expect("Range");
    assert!(TableColumns::<F65537>::new(&fits).is_ok());
    let wide = GeneratedTable::new(TableId(8), TableKind::Range { bits: 17 }).expect("Range");
    assert_eq!(
        TableColumns::<F65537>::new(&wide).err(),
        Some(table(
            8,
            "a LogUp table has no more rows than its field has elements"
        ))
    );
    assert!(TableColumns::<Fp>::new(&wide).is_ok());
}
