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
