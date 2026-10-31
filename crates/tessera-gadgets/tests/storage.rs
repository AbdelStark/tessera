// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Independent allocator accounting of public gadget templates (RFC-0018, #641).
mod common;

use allocation_counter::measure;
use tessera_codec::Rational;
use tessera_gadgets::{Requant, RsqrtRow, SoftmaxRow};
use tessera_tables::{GeneratedTable, Table, TableId, TableKind, TableSpec};

fn check<T: Clone>(make: impl FnOnce() -> T, heap: impl Fn(&T) -> Option<usize>) {
    let mut value = None;
    let built = measure(|| value = Some(make()));
    let value = value.unwrap();
    let bytes = heap(&value).unwrap();
    assert_eq!(usize::try_from(built.bytes_current).unwrap(), bytes);
    let queried = measure(|| assert_eq!(heap(&value), Some(bytes)));
    assert_eq!(queried.count_total, 0);
    let mut cloned = None;
    let copying = measure(|| cloned = Some(value.clone()));
    let cloned = cloned.unwrap();
    assert_eq!(
        usize::try_from(copying.bytes_current).unwrap(),
        heap(&cloned).unwrap()
    );
    assert!(heap(&cloned).unwrap() <= bytes);
}

#[test]
fn requant_templates_count_wide_boxes_and_all_class_payloads() {
    for (b, shifts, lo, hi, maximum) in [
        (8, &[3u8][..], -16, 15, 8),
        (12, &[5, 7][..], -16, 15, 10),
        (14, &[3][..], -3, 3, 10),
        (14, &[3, 4][..], -5, 2, 10),
        (13, &[2, 3, 4, 5][..], -4, 4, 10),
        (12, &[1, 2][..], -2, 3, 9),
        (62, &[33, 34, 35, 36][..], -128, 127, 24),
    ] {
        let (selected, tables) = common::select_requant(b, shifts, lo, hi, maximum);
        check(
            || Requant::new(&selected, &tables).unwrap(),
            Requant::heap_bytes,
        );
    }
}

#[test]
fn every_root_width_counts_comparison_bounds_carries_and_table_capacities() {
    let tables = (1..=32)
        .map(|bits| {
            GeneratedTable::new(TableId(u32::from(bits)), TableKind::Range { bits })
                .unwrap()
                .spec()
                .clone()
        })
        .collect::<Vec<_>>();
    for line in include_str!("../../../spec/vectors/rsqrt/canonical.txt")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let numbers = line
            .split_whitespace()
            .map(|s| s.parse::<u64>().unwrap())
            .collect::<Vec<_>>();
        check(
            || RsqrtRow::new(numbers[1] as u32, numbers[2] as u8, numbers[3], &tables).unwrap(),
            RsqrtRow::heap_bytes,
        );
    }
}

fn softmax_specs(sigma: u8, limb: u8) -> Vec<TableSpec> {
    let mut tables = vec![
        GeneratedTable::new(
            TableId(0),
            TableKind::Exp {
                key_bits: 6,
                sigma,
                scale: Rational::new(1, 1).unwrap(),
            },
        )
        .unwrap()
        .spec()
        .clone(),
    ];
    for kind in SoftmaxRow::range_kinds(sigma, limb).unwrap() {
        tables.push(
            GeneratedTable::new(TableId(tables.len() as u32), kind)
                .unwrap()
                .spec()
                .clone(),
        );
    }
    tables
}

#[test]
fn softmax_templates_count_exp_cores_without_borrowed_input_specs() {
    for (sigma, limb, context) in [
        (0, 1, 1),
        (0, 32, 1),
        (16, 8, 2048),
        (16, 16, 131072),
        (31, 32, 2),
    ] {
        let tables = softmax_specs(sigma, limb);
        check(
            || SoftmaxRow::new(TableId(0), limb, context, &tables).unwrap(),
            SoftmaxRow::heap_bytes,
        );
    }
}

#[test]
fn root_metadata_traffic_covers_all_32_widths_and_complete_constraint_templates() {
    use tessera_field::Goldilocks;
    let tables = (1..=32)
        .map(|bits| {
            GeneratedTable::new(TableId(u32::from(bits)), TableKind::Range { bits })
                .unwrap()
                .spec()
                .clone()
        })
        .collect::<Vec<_>>();
    let mut plan = None;
    let planning = measure(|| plan = RsqrtRow::memory());
    assert_eq!(planning.count_total, 0);
    let plan = plan.unwrap();
    for bits in 1..=32 {
        let mut root = None;
        let building = measure(|| root = Some(RsqrtRow::new(1, bits - 1, 1, &tables).unwrap()));
        assert!(
            building.bytes_total <= plan.template_bytes(),
            "{bits}: {building:?} {plan:?}"
        );
        let root = root.unwrap();
        assert_eq!(root.root_bits(), bits);
        let mut constraints = None;
        let describing = measure(|| constraints = Some(root.constraints::<Goldilocks>().unwrap()));
        assert!(
            describing.bytes_total <= plan.constraints_bytes(),
            "{bits}: {describing:?} {plan:?}"
        );
        let constraints = constraints.unwrap();
        if bits == 32 {
            assert_eq!(constraints.committed.len(), 187);
            assert_eq!(constraints.lookups.len(), 189);
            assert_eq!(constraints.equalities.len(), 97);
        }
    }
}

#[test]
fn requant_metadata_traffic_covers_all_forms_classes_and_wide_conversion() {
    use tessera_field::Goldilocks;
    let mut plan = None;
    let planning = measure(|| plan = Requant::memory());
    assert_eq!(planning.count_total, 0);
    let plan = plan.unwrap();
    for (b, shifts, lo, hi, maximum) in [
        (8, &[3u8][..], -16, 15, 8),
        (12, &[5, 7][..], -16, 15, 10),
        (14, &[3][..], -3, 3, 10),
        (14, &[3, 4][..], -5, 2, 10),
        (13, &[2, 3, 4, 5][..], -4, 4, 10),
        (12, &[1, 2][..], -2, 3, 9),
        (62, &[33, 34, 35, 36][..], -128, 127, 24),
    ] {
        let (selected, tables) = common::select_requant(b, shifts, lo, hi, maximum);
        let mut gadget = None;
        let building = measure(|| gadget = Some(Requant::new(&selected, &tables).unwrap()));
        assert!(
            building.bytes_total <= plan.template_bytes(),
            "{building:?} {plan:?}"
        );
        let gadget = gadget.unwrap();
        let mut constraints = None;
        let describing = measure(|| constraints = Some(gadget.constraints::<Goldilocks>()));
        assert!(
            describing.bytes_total <= plan.constraints_bytes(),
            "{describing:?} {plan:?}"
        );
        drop(constraints);
    }
}

#[test]
fn softmax_metadata_traffic_covers_dense_cores_and_49_limb_slack_templates() {
    use tessera_field::Goldilocks;
    for (sigma, limb, context) in [
        (0, 1, 1),
        (0, 32, 1),
        (16, 8, 2048),
        (16, 16, 131072),
        (31, 32, 2),
        (31, 1, 1),
    ] {
        let tables = softmax_specs(sigma, limb);
        let mut plan = None;
        let planning = measure(|| plan = Some(SoftmaxRow::memory(TableId(0), &tables).unwrap()));
        assert_eq!(planning.count_total, 0);
        let plan = plan.unwrap();
        let mut gadget = None;
        let building =
            measure(|| gadget = Some(SoftmaxRow::new(TableId(0), limb, context, &tables).unwrap()));
        assert!(
            building.bytes_total <= plan.template_bytes(),
            "{building:?} {plan:?}"
        );
        let gadget = gadget.unwrap();
        let mut constraints = None;
        let describing = measure(|| constraints = Some(gadget.constraints::<Goldilocks>()));
        assert!(
            describing.bytes_total <= plan.constraints_bytes(),
            "{describing:?} {plan:?}"
        );
        let constraints = constraints.unwrap();
        if sigma == 31 && limb == 1 {
            assert_eq!(constraints.committed.len(), 99);
            assert_eq!(constraints.lookups.len(), 100);
            assert_eq!(constraints.relations.len(), 7);
        }
    }
}

#[test]
fn softmax_metadata_planning_refuses_missing_and_nonexp_specs_without_allocation() {
    use tessera_gadgets::GadgetError;
    let ranges = [
        GeneratedTable::new(TableId(0), TableKind::Range { bits: 1 })
            .unwrap()
            .spec()
            .clone(),
    ];
    for (tables, rule) in [
        (&[][..], "SoftmaxRow's Exp specification exists"),
        (&ranges[..], "SoftmaxRow reads Exp"),
    ] {
        let measured = measure(|| {
            assert_eq!(
                SoftmaxRow::memory(TableId(0), tables).err(),
                Some(GadgetError::Table {
                    table: TableId(0),
                    rule
                })
            );
        });
        assert_eq!(measured.count_total, 0);
    }
}

#[test]
fn requant_witness_requests_cover_all_forms_and_wide_transfers() {
    for (b, shifts, lo, hi, maximum) in [
        (8, &[3u8][..], -16, 15, 8),
        (12, &[5, 7][..], -16, 15, 10),
        (14, &[3][..], -3, 3, 10),
        (14, &[3, 4][..], -5, 2, 10),
        (13, &[2, 3, 4, 5][..], -4, 4, 10),
        (12, &[1, 2][..], -2, 3, 9),
        (62, &[33, 34, 35, 36][..], -128, 127, 24),
    ] {
        let (selected, tables) = common::select_requant(b, shifts, lo, hi, maximum);
        let gadget = Requant::new(&selected, &tables).unwrap();
        for rows in [0, 1, 3, 17, 1024] {
            let values = vec![0; rows];
            let classes = (0..rows)
                .map(|i| (i % shifts.len()) as u8)
                .collect::<Vec<_>>();
            let mut bound = None;
            let planning = measure(|| bound = Requant::witness_memory(rows as u64));
            assert_eq!(planning.count_total, 0);
            let bound = bound.unwrap();
            let mut tensors = None;
            let built = measure(|| {
                tensors = Some(
                    gadget
                        .witness(&values, Some(&classes))
                        .unwrap()
                        .into_tensors(),
                )
            });
            assert!(
                built.bytes_total <= bound,
                "{selected:?} rows={rows} {built:?} {bound}"
            );
        }
    }
}

#[test]
fn root_witness_requests_cover_every_width_comparison_and_transferred_column() {
    let tables = (1..=32)
        .map(|bits| {
            GeneratedTable::new(TableId(u32::from(bits)), TableKind::Range { bits })
                .unwrap()
                .spec()
                .clone()
        })
        .collect::<Vec<_>>();
    for bits in 1..=32 {
        let root = RsqrtRow::new(1, bits - 1, 1, &tables).unwrap();
        for rows in [0, 1, 3, 17, 1024] {
            let real = (0..rows).map(|i| i % 3 != 1).collect::<Vec<_>>();
            let values = real
                .iter()
                .enumerate()
                .map(|(i, &r)| {
                    if !r {
                        0
                    } else if i % 2 == 0 {
                        1
                    } else {
                        (1 << 55) - 1
                    }
                })
                .collect::<Vec<_>>();
            let mut bound = None;
            let planning = measure(|| bound = RsqrtRow::witness_memory(rows as u64));
            assert_eq!(planning.count_total, 0);
            let bound = bound.unwrap();
            let mut tensors = None;
            let built =
                measure(|| tensors = Some(root.witness(&values, &real).unwrap().into_tensors()));
            assert!(
                built.bytes_total <= bound,
                "bits={bits} rows={rows} {built:?} {bound}"
            );
            if bits == 32 {
                assert_eq!(tensors.as_ref().unwrap().len(), 187);
            }
        }
    }
}

#[test]
fn softmax_witness_and_evaluation_requests_cover_copies_and_lookup_concatenation() {
    use tessera_field::Goldilocks;
    for (sigma, limb, max_context) in [
        (0, 1, 4096),
        (0, 32, 4096),
        (16, 8, 4096),
        (16, 17, 65536), // Reciprocal Range(17) shares the NonNeg limb group.
        (16, 16, 4096),
        (31, 32, 2),
        (31, 1, 2),
    ] {
        let tables = softmax_specs(sigma, limb);
        for positions in [1usize, 2, 8, 128, 4096, 65536]
            .into_iter()
            .filter(|&n| n <= max_context)
        {
            let gadget = SoftmaxRow::new(TableId(0), limb, max_context, &tables).unwrap();
            let scores = (0..positions)
                .map(|i| (i % 23) as i64 - 11)
                .collect::<Vec<_>>();
            let mut bound = None;
            let planning = measure(|| {
                bound = Some((
                    SoftmaxRow::witness_memory(positions as u64).unwrap(),
                    SoftmaxRow::evaluation_memory(positions as u64).unwrap(),
                ))
            });
            assert_eq!(planning.count_total, 0);
            let (witness_bound, evaluation_bound) = bound.unwrap();
            for valid in [0, 1, positions.div_ceil(2), positions] {
                let mut witness = None;
                let built = measure(|| witness = Some(gadget.witness(&scores, valid).unwrap()));
                assert!(
                    built.bytes_total <= witness_bound,
                    "sigma={sigma} limb={limb} positions={positions} {built:?} {witness_bound}"
                );
                let witness = witness.unwrap();
                let mut evaluation = None;
                let built = measure(|| {
                    evaluation = Some(
                        gadget
                            .evaluate::<Goldilocks>(&scores, valid, &witness)
                            .unwrap(),
                    )
                });
                assert!(
                    built.bytes_total <= evaluation_bound,
                    "sigma={sigma} limb={limb} positions={positions} {built:?} {evaluation_bound}"
                );
            }
        }
    }
}

#[test]
fn witness_request_planning_refuses_unaddressable_rows_without_allocation() {
    let planning = measure(|| {
        assert_eq!(Requant::witness_memory(u64::MAX), None);
        assert_eq!(RsqrtRow::witness_memory(u64::MAX), None);
        for rows in [0, 3, 1 << 63] {
            assert_eq!(SoftmaxRow::witness_memory(rows), None);
            assert_eq!(SoftmaxRow::evaluation_memory(rows), None);
        }
    });
    assert_eq!(planning.count_total, 0);
}
