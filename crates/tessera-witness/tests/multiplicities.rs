// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0041: a table tile contains its committed lookup multiplicities.

mod common;

use common::{conformance, lowered, plan_of};
use tessera_codec::{Secret, SecretBuf};
use tessera_compiler::Tile;
use tessera_witness::{Checkpoints, Engine, Regenerator, Resident, WitnessSource};

use std::collections::{BTreeMap, BTreeSet};
use tessera_compiler::{Lowering, Selection};
use tessera_field::{Goldilocks, PrimeField};
use tessera_gadgets::{Requant, Term};
use tessera_ir::{Op, TableId};
use tessera_tables::{GeneratedTable, Table, TableKind};
use tessera_witness::{MultiplicityColumn, Witness};

struct Histogram {
    rows: u64,
    counts: BTreeMap<u64, u64>,
}

fn check_column(column: &MultiplicityColumn, want: &Histogram) {
    assert_eq!(column.rows(), want.rows);
    let width = want.rows.min(1_024);
    let mut starts: BTreeSet<u64> = want.counts.keys().map(|key| key / width * width).collect();
    starts.extend([0, want.rows - width]);
    let mut seed = 0x0112_0050_0000_0001u64;
    for _ in 0..64 {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        starts.insert(seed % (want.rows / width) * width);
    }
    let mut chunk = SecretBuf::filled(0, width as usize);
    for start in starts {
        column
            .read_rows(start, &mut chunk)
            .expect("canonical chunk");
        assert!(
            chunk.iter().enumerate().all(|(offset, count)| {
                *count
                    == want
                        .counts
                        .get(&(start + offset as u64))
                        .copied()
                        .unwrap_or(0)
            }),
            "every contributed row and tested zero/padding chunk matches the independent histogram"
        );
    }
}

/// Materialize a per-dimension padded tensor, using coordinates rather than the production
/// counter's aggregate padding formula.
fn padded(values: &[i64], shape: &[u32]) -> Vec<i64> {
    let axes: Vec<usize> = shape
        .iter()
        .map(|d| d.next_power_of_two() as usize)
        .collect();
    (0..axes.iter().product())
        .map(|mut p| {
            let mut coordinates = vec![0; axes.len()];
            for i in (0..axes.len()).rev() {
                coordinates[i] = p % axes[i];
                p /= axes[i];
            }
            if coordinates
                .iter()
                .zip(shape)
                .any(|(&c, &d)| c >= d as usize)
            {
                return 0;
            }
            let i = coordinates
                .iter()
                .zip(shape)
                .fold(0, |i, (&c, &d)| i * d as usize + c);
            values[i]
        })
        .collect()
}

/// Full first-pass lookup columns, independently materialized under RFC-0035/RFC-0045.
fn histogram(
    lowering: &Lowering,
    selection: &Selection,
    witness: &Witness,
) -> BTreeMap<TableId, Histogram> {
    let mut columns: BTreeMap<TableId, Vec<u64>> = BTreeMap::new();
    let tensor = |t: tessera_ir::TensorId| &witness.tensor(t).expect("tensor")[..];
    let shape = |t: tessera_ir::TensorId| lowering.graph.tensors[t.0 as usize].shape.as_slice();
    let mut add = |table, keys: Vec<u64>| columns.entry(table).or_default().extend(keys);
    for node in &lowering.graph.nodes {
        let Some((_, gadget)) = selection.of(node.id) else {
            continue;
        };
        let input = node.inputs[0];
        let z = padded(tensor(input), shape(input));
        let classes = if matches!(node.op, Op::Requant { .. } | Op::TableMap { .. }) {
            node.inputs.get(1).map(|&c| {
                let by_channel = padded(tensor(c), shape(c));
                (0..z.len())
                    .map(|i| u8::try_from(by_channel[i % by_channel.len()]).expect("class"))
                    .collect::<Vec<_>>()
            })
        } else {
            None
        };
        match node.op {
            Op::Requant { .. } => {
                let req = Requant::new(&gadget, &selection.tables).expect("requantization");
                let values = req.witness(&z, classes.as_deref()).expect("padded witness");
                for lookup in req.constraints::<Goldilocks>().lookups {
                    let keys = (0..z.len())
                        .map(|i| {
                            lookup
                                .key
                                .evaluate::<Goldilocks>(|term| match term {
                                    Term::Input => Goldilocks::from_i64(z[i]),
                                    Term::Class => Goldilocks::from_u64_reduced(u64::from(
                                        classes.as_ref().map_or(0, |c| c[i]),
                                    )),
                                    Term::Committed(c) => Goldilocks::from_i64(
                                        values.tensor(c).expect("committed value")[i],
                                    ),
                                })
                                .to_canonical_u64()
                        })
                        .collect();
                    add(lookup.table, keys);
                }
            }
            Op::TableMap { table } | Op::Bitwise { table } => {
                let spec = selection
                    .tables
                    .iter()
                    .find(|s| s.id == table)
                    .expect("specification");
                let t = GeneratedTable::from_spec(spec).expect("table");
                let b = matches!(node.op, Op::Bitwise { .. })
                    .then(|| padded(tensor(node.inputs[1]), shape(node.inputs[1])));
                let keys = z
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        t.index(
                            v + b.as_ref().map_or(0, |b| b[i] << 8),
                            classes.as_ref().map_or(0, |c| c[i]),
                        )
                        .expect("row")
                    })
                    .collect();
                add(table, keys);
            }
            Op::RsqrtRow { .. } => {
                let real: Vec<_> = padded(&vec![1; tensor(input).len()], shape(input))
                    .into_iter()
                    .map(|r| r == 1)
                    .collect();
                let (_, lookups) = common::certificates::root(selection, node, &z, &real);
                for (table, keys, _) in lookups {
                    add(table, keys.into_iter().map(u64::from).collect());
                }
            }
            Op::SoftmaxRow { .. } => {
                let [heads, n, _] = *shape(input) else {
                    panic!("scores");
                };
                let width = n.next_power_of_two() as usize;
                let e = padded(tensor(node.outputs[0]), shape(node.outputs[0]));
                for h in 0..heads.next_power_of_two() {
                    for q in 0..n.next_power_of_two() {
                        let offset = (h * n.next_power_of_two() + q) as usize * width;
                        let valid = if h < heads && q < n {
                            q as usize + 1
                        } else {
                            0
                        };
                        let (_, lookups) = common::certificates::softmax(
                            selection,
                            node,
                            &z[offset..offset + width],
                            valid,
                            &e[offset..offset + width],
                        );
                        for (table, keys, _) in lookups {
                            add(table, keys.into_iter().map(u64::from).collect());
                        }
                    }
                }
            }
            _ => {}
        }
    }
    columns
        .into_iter()
        .map(|(id, mut keys)| {
            keys.resize(keys.len().next_power_of_two(), 0); // RFC-0045 column tail.
            let spec = selection.tables.iter().find(|s| s.id == id).expect("table");
            let rows = GeneratedTable::from_spec(spec)
                .expect("table")
                .row_count()
                .next_power_of_two();
            let mut counts = BTreeMap::<u64, u64>::new();
            for key in keys {
                *counts.entry(key).or_default() += 1;
            }
            (id, Histogram { rows, counts })
        })
        .collect()
}

#[test]
fn every_multiplicity_matches_the_padded_first_pass_column() {
    let mut checked = 0;
    for c in conformance() {
        for n in [3, 5] {
            let tokens = Secret::new((0..n).map(|i| (i * 113 + 7) % c.model.arch.vocab).collect());
            let (lowering, selection) = lowered(&c.model, n);
            let engine = Engine::new(&lowering).expect("tables");
            let witness = engine.run(&c.model, &tokens).expect("first pass");
            let expected = histogram(&lowering, &selection, &witness);
            let store = Checkpoints::new(&engine, &c.model, &tokens).expect("checkpoints");
            let resident =
                Resident::new(&engine, &selection, &c.model, &witness).expect("resident");
            let source =
                Regenerator::new(&engine, &selection, &c.model, &tokens, &store).expect("source");
            let mut tables = 0;
            for tile in plan_of(&c.model, 1).expand(&lowering).expect("tiles") {
                if let Tile::Table { table } = tile.tile {
                    let want = &expected[&TableId(table)];
                    assert!(want.counts.values().sum::<u64>().is_power_of_two());
                    for reader in [
                        &resident as &dyn WitnessSource,
                        &source as &dyn WitnessSource,
                    ] {
                        let values = reader.tile(&tile).expect("multiplicities");
                        check_column(values.multiplicities().expect("a count column"), want);
                        assert!((values.bytes() as u64) <= want.rows * size_of::<u64>() as u64);
                    }
                    checked += 1;
                    tables += 1;
                }
            }
            assert_eq!(
                tables,
                expected.len(),
                "every selected lookup table has a tile"
            );
        }
    }
    assert!(checked > 20);
}

#[test]
fn a_shared_table_includes_root_and_requantization_contributions() {
    let c = conformance().remove(0);
    let (lowering, selection) = lowered(&c.model, 3);
    // The actual RFC-0050 carry family is also a split-key family in this public fixture.
    let shared = selection
        .tables
        .iter()
        .find(|t| matches!(t.kind, TableKind::Range { bits: 25 }))
        .expect("the shared carry/split family")
        .id;
    assert!(selection.gadgets.iter().any(|(node, g)| matches!(
        g,
        tessera_compiler::Gadget::RsqrtRow
    )
        && selection.reads(*node).unwrap().contains(&shared)));
    assert!(selection.gadgets.iter().any(|(node, g)| matches!(
        g,
        tessera_compiler::Gadget::Requant1 { .. }
            | tessera_compiler::Gadget::RequantSplit { .. }
            | tessera_compiler::Gadget::RequantWide { .. }
    )
        && selection.reads(*node).unwrap().contains(&shared)));
    assert!(!selection.gadgets.iter().any(|(node, g)| matches!(
        g,
        tessera_compiler::Gadget::SoftmaxRow { .. }
    )
        && selection.reads(*node).unwrap().contains(&shared)));
    let engine = Engine::new(&lowering).expect("tables");
    let tokens = Secret::new(vec![1, 2, 3]);
    let witness = engine.run(&c.model, &tokens).expect("first pass");
    let store = Checkpoints::new(&engine, &c.model, &tokens).expect("store");
    let resident = Resident::new(&engine, &selection, &c.model, &witness).expect("resident");
    let source = Regenerator::new(&engine, &selection, &c.model, &tokens, &store).expect("source");
    let tile = tessera_compiler::SequenceTile {
        tile: Tile::Table { table: shared.0 },
        block: None,
        shares: Vec::new(),
    };
    let expected = histogram(&lowering, &selection, &witness);
    for reader in [&resident as &dyn WitnessSource, &source] {
        let values = reader.tile(&tile).expect("complete shared-table counts");
        check_column(values.multiplicities().expect("column"), &expected[&shared]);
    }
}
