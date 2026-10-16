// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Honest LogUp leaf stages through the independent verifier (DEP-2; RFC-0008 'Testing
//! Strategy'): every stage is proven by `prove_logup` and checked by the driver of `common`,
//! `tv_core::logup::check_roots`, `tv_core::tree::verify_level` at each level and
//! `tv_core::logup::check_leaves`, with `t̃(ρ_t)` evaluated from the table's rows. The verifier
//! must accept and draw the prover's challenges, and every claim must be its polynomial's
//! extension at its point.
//!
//! The cases: random dense tables of `2^8` to `2^16` rows with `2^4` to `2^20` lookups, up to
//! `2^20` lookups into `2^16` rows; instance sizes `2^k − 1`, `2^k` and `2^k + 1` on both sides; tables with and without values; several instances of different
//! heights in schedules of every kind, beside product forests; and instances of one lookup and
//! one row, whose trees join no level.

mod common;

use common::{Session, SplitMix64, TestForest, TestInstance, TestTable, prove_and_verify};
use tessera_logup::TreeId;

/// One instance alone, its two trees paired.
fn alone(session: &Session, instance: TestInstance) {
    let proven = prove_and_verify(
        session,
        std::slice::from_ref(&instance),
        &[],
        &TreeId::paired(1),
    );
    let tallest = instance.lookup_height().max(instance.table_height());
    assert_eq!(proven.levels().len(), tallest);
}

/// Dense tables of `2^8` to `2^16` rows, with and without values, and `2^4` to `2^16` random
/// lookups: one shape per pair of sizes, the larger lookup counts at the larger tables.
#[test]
fn random_dense_tables_verify() {
    let mut rng = SplitMix64::new(0x60_1001);
    let session = Session::random(&mut rng);
    for (rows_log, lookups_log) in [
        (8, 4),
        (8, 12),
        (10, 8),
        (12, 16),
        (14, 10),
        (16, 4),
        (16, 14),
    ] {
        for with_values in [true, false] {
            let table = TestTable::random(&mut rng, 1 << rows_log, with_values);
            let instance = TestInstance::random(&mut rng, table, 1 << lookups_log);
            alone(&session, instance);
        }
    }
}

/// RFC-0008's largest case: `2^20` lookups into `2^16` rows with values.
#[test]
fn two_to_the_20_lookups_into_two_to_the_16_rows_verify() {
    let mut rng = SplitMix64::new(0x60_1002);
    let session = Session::random(&mut rng);
    let table = TestTable::random(&mut rng, 1 << 16, true);
    let instance = TestInstance::random(&mut rng, table, 1 << 20);
    alone(&session, instance);
}

/// `2^20` lookups into `2^8` rows, in the default run: the lookup tree's 20 levels.
#[test]
fn two_to_the_20_lookups_into_a_small_table_verify() {
    let mut rng = SplitMix64::new(0x60_1003);
    let session = Session::random(&mut rng);
    let table = TestTable::random(&mut rng, 1 << 8, false);
    let instance = TestInstance::random(&mut rng, table, 1 << 20);
    alone(&session, instance);
}

/// Instance sizes `2^k − 1`, `2^k` and `2^k + 1` on both sides, for `k` from 1 to 6, every
/// combination: padding leaves on the lookup side, padding rows on the table side, both, or
/// neither.
#[test]
fn instance_sizes_around_powers_of_two_verify() {
    let mut rng = SplitMix64::new(0x60_1004);
    let session = Session::random(&mut rng);
    let mut cases = 0;
    for k in 1..=6usize {
        let sizes = [(1usize << k) - 1, 1 << k, (1 << k) + 1];
        for &rows in &sizes {
            for &lookups in &sizes {
                let table = TestTable::random(&mut rng, rows, cases % 2 == 0);
                let instance = TestInstance::random(&mut rng, table, lookups);
                alone(&session, instance);
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 54);
}

/// Every lookup of row 0 and of the last row, the padding's row and the row next to the padding
/// rows, and a lookup of every row.
#[test]
fn lookups_at_the_edges_of_the_table_verify() {
    let mut rng = SplitMix64::new(0x60_1005);
    let session = Session::random(&mut rng);
    for (rows, with_values) in [(5usize, true), (8, false), (9, true)] {
        let table = TestTable::random(&mut rng, rows, with_values);
        for lookups in [
            vec![0; 7],
            vec![rows - 1; 3],
            (0..rows).collect(),
            vec![0, rows - 1],
        ] {
            alone(&session, TestInstance::of_rows(table.clone(), &lookups));
        }
    }
}

/// One lookup into a table of one row: both trees have height 0, no level, and the values
/// message follows the roots.
#[test]
fn one_lookup_into_one_row_verifies() {
    let mut rng = SplitMix64::new(0x60_1006);
    let session = Session::random(&mut rng);
    for with_values in [true, false] {
        let table = TestTable::random(&mut rng, 1, with_values);
        let instance = TestInstance::of_rows(table, &[0]);
        let proven = prove_and_verify(&session, &[instance], &[], &TreeId::paired(1));
        assert!(proven.levels().is_empty());
        assert_eq!(proven.values().len(), usize::from(with_values));
    }
}

/// Several instances of different heights, tables with and without values, in schedules of
/// every kind: paired, all lookup trees first, table trees before their lookup trees, reversed.
#[test]
fn several_instances_in_any_schedule_verify() {
    let mut rng = SplitMix64::new(0x60_1007);
    let session = Session::random(&mut rng);
    let shapes = [
        (5usize, 9usize, true),
        (17, 3, false),
        (1, 1, true),
        (64, 100, true),
        (3, 33, false),
    ];
    let instances: Vec<TestInstance> = shapes
        .iter()
        .map(|&(rows, lookups, with_values)| {
            let table = TestTable::random(&mut rng, rows, with_values);
            TestInstance::random(&mut rng, table, lookups)
        })
        .collect();
    let m = instances.len();
    let paired = TreeId::paired(m);
    let lookups_first: Vec<TreeId> = (0..m)
        .map(TreeId::Lookup)
        .chain((0..m).map(TreeId::Table))
        .collect();
    let tables_first: Vec<TreeId> = (0..m)
        .flat_map(|k| [TreeId::Table(k), TreeId::Lookup(k)])
        .collect();
    let mut reversed = paired.clone();
    reversed.reverse();
    for schedule in [paired, lookups_first, tables_first, reversed] {
        let proven = prove_and_verify(&session, &instances, &[], &schedule);
        assert_eq!(proven.values().len(), 3, "the three instances with values");
    }
}

/// LogUp instances beside product forests, interleaved: levels of forests alone squeeze no
/// `λ`, forests of several trees join under root challenges where fraction trees are present.
#[test]
fn instances_beside_forests_verify() {
    let mut rng = SplitMix64::new(0x60_1008);
    let session = Session::random(&mut rng);
    let instances: Vec<TestInstance> = [(6usize, 12usize, true), (2, 3, false), (1, 1, true)]
        .iter()
        .map(|&(rows, lookups, with_values)| {
            let table = TestTable::random(&mut rng, rows, with_values);
            TestInstance::random(&mut rng, table, lookups)
        })
        .collect();
    let forests = vec![
        TestForest::random(&mut rng, 1, 6),
        TestForest::random(&mut rng, 2, 2),
        TestForest::random(&mut rng, 0, 4),
    ];
    let schedule = [
        TreeId::Table(1),
        TreeId::Forest(2),
        TreeId::Lookup(0),
        TreeId::Forest(0),
        TreeId::Table(0),
        TreeId::Lookup(2),
        TreeId::Forest(1),
        TreeId::Lookup(1),
        TreeId::Table(2),
    ];
    let proven = prove_and_verify(&session, &instances, &forests, &schedule);
    // The forest of height 6 alone at levels 0 and 1, before the lookup tree of height 4.
    let lambdas: Vec<bool> = proven
        .levels()
        .iter()
        .map(|level| level.lambda().is_some())
        .collect();
    assert_eq!(lambdas, [false, false, true, true, true, true]);
    // No instance at all: a forest alone, with the messages absorbed empty.
    let forest = [TestForest::random(&mut rng, 1, 1)];
    let proven = prove_and_verify(&session, &[], &forest, &[TreeId::Forest(0)]);
    assert!(proven.roots().is_empty() && proven.values().is_empty());
    // Nothing at all: α and β, then both messages empty, and no level.
    let proven = prove_and_verify(&session, &[], &[], &[]);
    assert!(proven.levels().is_empty() && proven.roots().is_empty());
}

/// Seeded random stages: 1 to 4 instances of 1 to 300 rows and lookups, with or without values,
/// beside 0 to 2 forests, in a random schedule.
#[test]
fn seeded_random_stages_verify() {
    let mut rng = SplitMix64::new(0x60_1009);
    let session = Session::random(&mut rng);
    for _ in 0..30 {
        let m = 1 + rng.below(4) as usize;
        let instances: Vec<TestInstance> = (0..m)
            .map(|_| {
                let rows = 1 + rng.below(300) as usize;
                let lookups = 1 + rng.below(300) as usize;
                let with_values = rng.coin();
                let table = TestTable::random(&mut rng, rows, with_values);
                TestInstance::random(&mut rng, table, lookups)
            })
            .collect();
        let forests: Vec<TestForest> = (0..rng.below(3))
            .map(|_| {
                let (s, h) = (rng.below(4) as usize, 1 + rng.below(8) as usize);
                TestForest::random(&mut rng, s, h)
            })
            .collect();
        let mut schedule: Vec<TreeId> = TreeId::paired(m)
            .into_iter()
            .chain((0..forests.len()).map(TreeId::Forest))
            .collect();
        for i in (1..schedule.len()).rev() {
            let j = rng.below(i as u64 + 1) as usize;
            schedule.swap(i, j);
        }
        let _ = prove_and_verify(&session, &instances, &forests, &schedule);
    }
}

/// `multiplicities` counts what an honest witness commits: on random shapes with and without
/// padding on either side, the counts of the lookups' rows with the lookup tree's padding on row
/// 0, computed here from the definition; an instance that commits them verifies.
#[test]
fn the_counted_multiplicities_are_the_honest_ones() {
    let mut rng = SplitMix64::new(0x60_100a);
    let session = Session::random(&mut rng);
    for _ in 0..20 {
        let rows = 1 + rng.below(70) as usize;
        let lookups = 1 + rng.below(90) as usize;
        let with_values = rng.coin();
        let table = TestTable::random(&mut rng, rows, with_values);
        let chosen: Vec<usize> = (0..lookups)
            .map(|_| rng.below(rows as u64) as usize)
            .collect();
        let honest = TestInstance::of_rows(table, &chosen);
        let as_u32: Vec<u32> = chosen.iter().map(|&r| r as u32).collect();
        let counted = tessera_logup::multiplicities(honest.table.tuples(), &as_u32)
            .expect("rows of the table");
        assert_eq!(
            counted, honest.multiplicities,
            "{rows} rows, {lookups} lookups"
        );
        let mut instance = honest.clone();
        instance.multiplicities = counted;
        let _ = prove_and_verify(&session, &[instance], &[], &TreeId::paired(1));
    }
}
