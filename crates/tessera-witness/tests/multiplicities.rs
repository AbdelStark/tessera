// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0041: a table tile contains its committed lookup multiplicities.

mod common;

use common::{conformance, lowered, plan_of};
use tessera_codec::Secret;
use tessera_compiler::Tile;
use tessera_witness::{Checkpoints, Engine, Regenerator, Resident, WitnessSource};

use std::collections::BTreeMap;
use tessera_compiler::{Lowering, Selection};
use tessera_field::{Goldilocks, PrimeField};
use tessera_gadgets::{Requant, Term};
use tessera_ir::{Op, TableId};
use tessera_tables::{GeneratedTable, Table, TableKind};
use tessera_witness::Witness;

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
) -> BTreeMap<TableId, Vec<u64>> {
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
            Op::SoftmaxRow { exp_table, .. } => {
                let spec = selection
                    .tables
                    .iter()
                    .find(|s| s.id == exp_table)
                    .expect("Exp");
                let TableKind::Exp { key_bits, .. } = spec.kind else {
                    panic!("Exp");
                };
                let [heads, n, _] = *shape(input) else {
                    panic!("scores");
                };
                let scores = tensor(input);
                let mut keys = Vec::new();
                for h in 0..heads.next_power_of_two() {
                    for q in 0..n.next_power_of_two() {
                        for i in 0..n.next_power_of_two() {
                            let key = if h < heads && q < n && i <= q {
                                let row = ((h * n + q) * n) as usize;
                                let max =
                                    *scores[row..=row + q as usize].iter().max().expect("row");
                                u64::try_from(max - scores[row + i as usize]).expect("key")
                            } else {
                                (1u64 << key_bits) - 1
                            };
                            keys.push(key);
                        }
                    }
                }
                add(exp_table, keys);
            }
            _ => {}
        }
    }
    columns
        .into_iter()
        .map(|(id, mut keys)| {
            keys.resize(keys.len().next_power_of_two(), 0); // RFC-0045 column tail.
            let spec = selection.tables.iter().find(|s| s.id == id).expect("table");
            let rows = GeneratedTable::from_spec(spec).expect("table").row_count() as usize;
            let mut counts = vec![0; rows.next_power_of_two()];
            for key in keys {
                counts[key as usize] += 1;
            }
            (id, counts)
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
                    if common::unsupported_table(&selection, &tile) {
                        common::assert_unsupported(resident.tile(&tile));
                        common::assert_unsupported(source.tile(&tile));
                        continue;
                    }
                    let want = &expected[&TableId(table)];
                    assert!(want.iter().sum::<u64>().is_power_of_two());
                    for reader in [
                        &resident as &dyn WitnessSource,
                        &source as &dyn WitnessSource,
                    ] {
                        let values = reader.tile(&tile).expect("multiplicities");
                        assert!(
                            values.multiplicities() == Some(want.as_slice()),
                            "{} n={n}, table {table}",
                            c.name
                        );
                        assert_eq!(values.bytes(), want.len() * size_of::<u64>());
                    }
                    checked += 1;
                    tables += 1;
                }
            }
            let supported = expected
                .keys()
                .filter(|id| {
                    !common::unsupported_table(
                        &selection,
                        &tessera_compiler::SequenceTile {
                            tile: Tile::Table { table: id.0 },
                            block: None,
                            shares: Vec::new(),
                        },
                    )
                })
                .count();
            assert_eq!(tables, supported, "every supported lookup table has a tile");
        }
    }
    assert!(checked > 20);
}

#[test]
fn newly_selected_auxiliary_tables_are_explicitly_refused() {
    let c = conformance().remove(0);
    let (lowering, selection) = lowered(&c.model, 3);
    let engine = Engine::new(&lowering).expect("tables");
    let tokens = Secret::new(vec![1, 2, 3]);
    let witness = engine.run(&c.model, &tokens).expect("first pass");
    let store = Checkpoints::new(&engine, &c.model, &tokens).expect("store");
    let resident = Resident::new(&engine, &selection, &c.model, &witness).expect("resident");
    let source = Regenerator::new(&engine, &selection, &c.model, &tokens, &store).expect("source");
    let mut refused = std::collections::BTreeSet::new();
    for tile in plan_of(&c.model, 1).expand(&lowering).expect("tiles") {
        if common::unsupported_table(&selection, &tile) {
            common::assert_unsupported(resident.tile(&tile));
            common::assert_unsupported(source.tile(&tile));
            if let Tile::Table { table } = tile.tile {
                refused.insert(table);
            }
        }
    }
    assert!(refused.len() >= 2, "reciprocal and bracket ranges");
}

#[test]
fn an_auxiliary_table_shared_with_requantization_is_refused() {
    let c = conformance().remove(0);
    let (lowering, mut selection) = lowered(&c.model, 3);
    // Synthetic identifier alias: it must never expose only the supported contributor's counts.
    let shared = selection
        .gadgets
        .iter()
        .find_map(|(_, g)| match *g {
            tessera_compiler::Gadget::Requant1 { table } => Some(table),
            tessera_compiler::Gadget::RequantSplit { low, .. } => Some(low),
            tessera_compiler::Gadget::RequantWide { mid, .. } => Some(mid),
            _ => None,
        })
        .expect("a supported requantization table");
    // The model-only lowering has no decoding node yet. Inject its public selection metadata
    // without replacing any supported node, to exercise the future shared-table case directly.
    selection.gadgets.push((
        tessera_ir::NodeId(u32::MAX),
        tessera_compiler::Gadget::Argmax { table: shared },
    ));
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
    common::assert_unsupported(resident.tile(&tile));
    common::assert_unsupported(source.tile(&tile));
}
