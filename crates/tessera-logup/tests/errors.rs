// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Every input-dependent failure is a typed error, never a panic (04 §1): each test violates one
//! condition and asserts the exact error; the checks come in the documented order, the instances'
//! columns, then the forests, then the schedule; and a refused stage leaves the transcript where
//! it was, before `α`. Every message names indices and lengths only (05 §3).

mod common;

use common::{Fp, Fp2, Session, SplitMix64, TestForest, TestInstance, TestTable};
use tessera_field::Field;
use tessera_hash::Transcript;
use tessera_logup::{
    Column, LogUpError, LogUpInstance, TreeId, Tuples, multiplicities, prove_logup,
};
use tessera_sumcheck::{Forest, SumcheckError};

/// Proves on a fresh transcript and checks that a refusal left it untouched.
fn refused(
    instances: &[LogUpInstance<'_, Fp>],
    forests: &[Forest<'_, Fp2>],
    schedule: &[TreeId],
) -> LogUpError {
    let session = Session::random(&mut SplitMix64::new(0x60_4001));
    let mut transcript = session.transcript();
    let error = prove_logup(&mut transcript, instances, forests, schedule)
        .expect_err("the stage is refused");
    let fresh: Transcript = session.transcript();
    assert_eq!(
        (transcript.state(), transcript.counter()),
        (fresh.state(), fresh.counter()),
        "a refused stage does not move the transcript"
    );
    error
}

fn honest(rng: &mut SplitMix64, rows: usize, lookups: usize, with_values: bool) -> TestInstance {
    let table = TestTable::random(rng, rows, with_values);
    TestInstance::random(rng, table, lookups)
}

#[test]
fn a_table_without_rows_is_refused() {
    let mut rng = SplitMix64::new(0x60_4002);
    let good = honest(&mut rng, 3, 4, true);
    let keys = [Fp::ONE; 2];
    let empty = LogUpInstance::new(Tuples::new(&[], None), Tuples::new(&keys, None), &[]);
    assert_eq!(
        refused(&[good.instance(), empty], &[], &TreeId::paired(2)),
        LogUpError::NoRows { instance: 1 }
    );
}

#[test]
fn an_instance_without_lookups_is_refused() {
    let mut rng = SplitMix64::new(0x60_4003);
    let good = honest(&mut rng, 3, 4, false);
    let none = LogUpInstance::new(
        good.table.tuples(),
        Tuples::new(&[], None),
        &good.multiplicities,
    );
    assert_eq!(
        refused(&[none], &[], &TreeId::paired(1)),
        LogUpError::NoLookups { instance: 0 }
    );
}

#[test]
fn values_on_one_side_only_are_refused() {
    let mut rng = SplitMix64::new(0x60_4004);
    let with = honest(&mut rng, 4, 4, true);
    let without = honest(&mut rng, 4, 4, false);
    // Lookups without values into a table with values, and the other way round.
    let a = LogUpInstance::new(
        with.table.tuples(),
        Tuples::new(&with.keys, None),
        &with.multiplicities,
    );
    let b = LogUpInstance::new(
        without.table.tuples(),
        Tuples::new(&without.keys, Some(&without.keys)),
        &without.multiplicities,
    );
    for (bad, instance) in [(a, 0), (b, 0)] {
        assert_eq!(
            refused(&[bad], &[], &TreeId::paired(1)),
            LogUpError::ValuesMismatch { instance }
        );
    }
}

#[test]
fn columns_of_the_wrong_length_are_refused() {
    let mut rng = SplitMix64::new(0x60_4005);
    let i = honest(&mut rng, 5, 6, true);
    let values = i.table.values.clone().expect("values");
    let lookup_values = i.values.clone().expect("values");
    let cases = [
        (
            LogUpInstance::new(
                Tuples::new(&i.table.keys, Some(&values[..4])),
                Tuples::new(&i.keys, Some(&lookup_values)),
                &i.multiplicities,
            ),
            Column::TableValues,
            4,
            5,
        ),
        (
            LogUpInstance::new(
                i.table.tuples(),
                Tuples::new(&i.keys, Some(&lookup_values[..5])),
                &i.multiplicities,
            ),
            Column::LookupValues,
            5,
            6,
        ),
        (
            LogUpInstance::new(
                i.table.tuples(),
                Tuples::new(&i.keys, Some(&lookup_values)),
                &i.multiplicities[..5],
            ),
            Column::Multiplicities,
            5,
            8,
        ),
    ];
    for (bad, column, len, expected) in cases {
        let good = i.instance();
        assert_eq!(
            refused(&[good, bad], &[], &TreeId::paired(2)),
            LogUpError::ColumnLength {
                instance: 1,
                column,
                len,
                expected
            }
        );
    }
    // Checked in order: the table's values before the lookups.
    let short = LogUpInstance::new(
        Tuples::new(&i.table.keys, Some(&values[..1])),
        Tuples::new(&[], None),
        &[],
    );
    assert_eq!(
        refused(&[short], &[], &TreeId::paired(1)),
        LogUpError::ColumnLength {
            instance: 0,
            column: Column::TableValues,
            len: 1,
            expected: 5
        }
    );
}

#[test]
fn a_lookup_of_no_row_is_refused_by_the_count() {
    let mut rng = SplitMix64::new(0x60_4006);
    let table = TestTable::random(&mut rng, 5, true);
    // Row 5 is the table tree's padding row, not a row of the table.
    for (rows, bad) in [
        (vec![0u32, 4, 5, 1], 2),
        (vec![7, 0], 0),
        (vec![1, 2, 3, u32::MAX], 3),
    ] {
        assert_eq!(
            multiplicities::<Fp>(table.tuples(), &rows),
            Err(LogUpError::LookupRow {
                lookup: bad,
                rows: 5
            })
        );
    }
    let empty = TestTable {
        keys: Vec::new(),
        values: None,
    };
    assert_eq!(
        multiplicities::<Fp>(empty.tuples(), &[0]),
        Err(LogUpError::LookupRow { lookup: 0, rows: 0 })
    );
    // A table of no row has no multiplicity, and no lookup no padding.
    assert_eq!(multiplicities::<Fp>(empty.tuples(), &[]), Ok(Vec::new()));
    assert_eq!(
        multiplicities::<Fp>(table.tuples(), &[]),
        Ok(vec![Fp::ZERO; 8])
    );
}

#[test]
fn a_schedule_of_the_wrong_length_is_refused() {
    let mut rng = SplitMix64::new(0x60_4007);
    let i = honest(&mut rng, 3, 3, true);
    let forest = TestForest::random(&mut rng, 1, 2);
    let schedule = [TreeId::Lookup(0), TreeId::Forest(0), TreeId::Table(0)];
    for len in [0, 2] {
        assert_eq!(
            refused(&[i.instance()], &[forest.forest()], &schedule[..len]),
            LogUpError::ScheduleLength { len, expected: 3 }
        );
    }
    let longer = [schedule.as_slice(), &[TreeId::Lookup(0)]].concat();
    assert_eq!(
        refused(&[i.instance()], &[forest.forest()], &longer),
        LogUpError::ScheduleLength {
            len: 4,
            expected: 3
        }
    );
}

#[test]
fn a_schedule_entry_naming_no_tree_or_a_named_one_is_refused() {
    let mut rng = SplitMix64::new(0x60_4008);
    let instances = [honest(&mut rng, 3, 3, true), honest(&mut rng, 2, 5, false)];
    let borrowed: Vec<LogUpInstance<'_, Fp>> =
        instances.iter().map(TestInstance::instance).collect();
    let cases = [
        // Instance 2 does not exist.
        (
            vec![
                TreeId::Lookup(0),
                TreeId::Table(0),
                TreeId::Lookup(2),
                TreeId::Table(1),
            ],
            2,
        ),
        // No forest.
        (
            vec![
                TreeId::Lookup(0),
                TreeId::Forest(0),
                TreeId::Lookup(1),
                TreeId::Table(1),
            ],
            1,
        ),
        // The table tree of instance 0 twice, so instance 1's is missing.
        (
            vec![
                TreeId::Table(0),
                TreeId::Lookup(0),
                TreeId::Lookup(1),
                TreeId::Table(0),
            ],
            3,
        ),
    ];
    for (schedule, position) in cases {
        assert_eq!(
            refused(&borrowed, &[], &schedule),
            LogUpError::ScheduleTree { position }
        );
    }
}

#[test]
fn a_forest_of_the_wrong_shape_is_refused_before_the_schedule() {
    let mut rng = SplitMix64::new(0x60_4009);
    let i = honest(&mut rng, 3, 3, true);
    let good = TestForest::random(&mut rng, 1, 2);
    let leaves = [Fp::ONE; 6];
    let odd = Forest::<Fp2>::new(&leaves, 1);
    let flat = Forest::<Fp2>::new(&leaves[..4], 0);
    // Forest 1's leaf count is not a power of two; a schedule that is also wrong is not reached.
    assert_eq!(
        refused(&[i.instance()], &[good.forest(), odd], &[]),
        LogUpError::Sumcheck(SumcheckError::LeafCount { forest: 1, len: 6 })
    );
    assert_eq!(
        refused(&[i.instance()], &[flat], &[]),
        LogUpError::Sumcheck(SumcheckError::Height {
            forest: 0,
            height: 0,
            vars: 2
        })
    );
    // The instances come first.
    let none = LogUpInstance::new(i.table.tuples(), Tuples::new(&[], None), &i.multiplicities);
    assert_eq!(
        refused(&[none], &[odd], &[]),
        LogUpError::NoLookups { instance: 0 }
    );
}

/// Debug forms show shapes, never an entry (05 §3).
#[test]
fn debug_forms_show_shapes_only() {
    let mut rng = SplitMix64::new(0x60_400a);
    let i = honest(&mut rng, 3, 5, true);
    let shown = format!("{:?}", i.instance());
    assert_eq!(
        shown,
        "LogUpInstance { table: Tuples { keys: 3, values: Some(3) }, lookups: Tuples { keys: 5, \
         values: Some(5) }, multiplicities: 4 }"
    );
    let session = Session::random(&mut rng);
    let (proven, _) = common::prove(&session, std::slice::from_ref(&i), &[], &TreeId::paired(1));
    let shown = format!("{proven:?}");
    for value in i.keys.iter().chain(i.table.keys.iter()) {
        let canonical = format!("{}", tessera_field::PrimeField::to_canonical_u64(value));
        assert!(!shown.contains(&canonical), "no key in the debug form");
    }
    assert!(format!("{:?}", proven.claims()[0]).contains("vars"));
    let _ = Fp::ZERO;
}
