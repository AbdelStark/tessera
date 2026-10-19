// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Layer-boundary checkpoints and the tiles regenerated from them (RFC-0018 'Witness engine',
//! 'Pass structure', 'Crash consistency'; RFC-0037 'Layers and kinds'; RFC-0041 'Tiles').
//!
//! - **The store.** On every conformance sequence, the checkpoints hold `c_0 … c_L` as INT32, each
//!   the residual the independent Python oracle computed at that layer boundary and the first full
//!   pass's tensor, bit for bit.
//! - **Bit for bit.** On every conformance model, at the vectors' lengths and at drawn ones from one
//!   token to the whole context, powers of two and not, under query blocks that hold the whole
//!   sequence or split it into blocks of four queries or of one, every tile of the plan regenerated
//!   from the checkpoints holds what the first full pass gives it: each share's elements, its
//!   gadget's committed values, and its lookups' tables, keys and values. The first pass's tiles are
//!   cut from its tensors here, independently of the crate: the shares by their ranges, and the
//!   lookups as RFC-0009 makes them, through `tessera-gadgets`' requantizations, the `Silu` table's
//!   packed keys, and the max certificate's `Exp` keys with `K_0` at every masked position. The
//!   resident source, which slices the first pass's witness, gives the same tiles.
//! - **Every kind.** The tiles met cover the tokens, each kind of node tile the lowering has
//!   (`OneHot`, the first checkpoint's `ConstMul`, `RsqrtRow`, `Requant1`, `RequantSplit`,
//!   `RequantWide` and `TableMap`), attention tiles at every query block, those that hold padding
//!   alone among them, and the tables'.
//! - **Rows of their tables.** Every lookup a source gives is a row of its table.
//! - **Determinism.** Engines on one thread or three regenerate the same tiles, and compute the same
//!   checkpoints: a prover that reruns its phases after a crash finds them again (RFC-0018 'Crash
//!   consistency').
//! - **Refusals.** A store of another sequence, an engine without a lowering, tokens of another
//!   length or outside the vocabulary, a first pass of another lowering, and each way a tile can
//!   fail to be one of the plan's are refused with `ERR-USAGE`.

mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroUsize;

use common::{QUERY_BLOCKS, Rng, conformance, indices, lowered, plan_of};
use tessera_codec::Secret;
use tessera_codec::error::ErrorCode;
use tessera_compiler::{Gadget, Lowering, QueryBlock, Selection, SequenceTile, Tile};
use tessera_field::{Goldilocks, PrimeField};
use tessera_gadgets::{Committed, Requant, Term};
use tessera_ir::{IntegerModel, Op, TableId, TensorId};
use tessera_tables::{GeneratedTable, Table, TableKind};
use tessera_witness::{
    Checkpoints, Engine, Regenerator, Resident, TileWitness, Witness, WitnessError, WitnessSource,
};

// ----- The first pass's tiles, cut independently ------------------------------------------------

/// What a tile holds, as plain values: each share's elements, the gadget's committed values beyond
/// the node's outputs, and each lookup's table, keys and values.
struct Expected {
    shares: Vec<Vec<i64>>,
    gadget: Vec<(Committed, Vec<i64>)>,
    lookups: Vec<(TableId, Vec<u32>, Option<Vec<i64>>)>,
}

/// The first full pass of a sequence, with what cutting its tiles reads: the lowering, the gadgets
/// and the tables they read, regenerated from their specifications.
struct FirstPass<'a> {
    lowering: &'a Lowering,
    selection: &'a Selection,
    model: &'a IntegerModel,
    witness: &'a Witness,
    tables: BTreeMap<TableId, GeneratedTable>,
}

impl<'a> FirstPass<'a> {
    fn new(
        lowering: &'a Lowering,
        selection: &'a Selection,
        model: &'a IntegerModel,
        witness: &'a Witness,
    ) -> Self {
        let tables = selection
            .tables
            .iter()
            .filter(|spec| !matches!(spec.kind, TableKind::Rope { .. }))
            .map(|spec| (spec.id, GeneratedTable::from_spec(spec).expect("a table")))
            .collect();
        Self {
            lowering,
            selection,
            model,
            witness,
            tables,
        }
    }

    fn values(&self, tensor: TensorId) -> &[i64] {
        self.witness
            .tensor(tensor)
            .expect("the first pass's tensor")
    }

    fn shape(&self, tensor: TensorId) -> &[u32] {
        &self.lowering.graph.tensors[tensor.0 as usize].shape
    }

    fn table(&self, id: TableId) -> &GeneratedTable {
        self.tables.get(&id).expect("a table the gadgets read")
    }

    /// The tile as the first pass holds it.
    fn tile(&self, tile: &SequenceTile) -> Expected {
        let shares = tile
            .shares
            .iter()
            .map(|share| {
                let values = self.values(share.tensor);
                indices(self.shape(share.tensor), &share.ranges)
                    .into_iter()
                    .map(|i| values[i])
                    .collect()
            })
            .collect();
        let mut expected = Expected {
            shares,
            gadget: Vec::new(),
            lookups: Vec::new(),
        };
        match tile.tile {
            Tile::Tokens | Tile::Table { .. } => {}
            Tile::Node { node } => self.node(node, &mut expected),
            Tile::Attention { node, kv_head } => {
                let block = tile.block.as_ref().expect("an attention tile's block");
                self.attention(node, kv_head, block, &mut expected);
            }
        }
        expected
    }

    /// Each element's class, its channel's, or `None` for a node of one class.
    fn classes(&self, inputs: &[TensorId], len: usize) -> Option<Vec<u8>> {
        let classes = self.values(*inputs.get(1)?);
        Some(
            (0..len)
                .map(|i| u8::try_from(classes[i % classes.len()]).expect("a class"))
                .collect(),
        )
    }

    fn node(&self, node: u32, expected: &mut Expected) {
        let node = &self.lowering.graph.nodes[node as usize];
        match &node.op {
            Op::Requant { .. } => {
                let (_, gadget) = self.selection.of(node.id).expect("a gadget");
                let requant = Requant::new(&gadget, &self.selection.tables).expect("a gadget");
                let z = self.values(node.inputs[0]);
                let classes = self.classes(&node.inputs, z.len());
                let witness = requant
                    .witness(z, classes.as_deref())
                    .expect("the first pass's keys and classes");
                assert!(
                    witness.tensor(Committed::Y) == Some(self.values(node.outputs[0])),
                    "the gadget's y is D's"
                );
                let constraints = requant.constraints::<Goldilocks>();
                expected.gadget = constraints
                    .committed
                    .iter()
                    .filter(|&&c| c != Committed::Y)
                    .map(|&c| (c, witness.tensor(c).expect("a tensor").to_vec()))
                    .collect();
                let term = |t: Term, i: usize| -> Goldilocks {
                    match t {
                        Term::Input => Goldilocks::from_i64(z[i]),
                        Term::Class => Goldilocks::from_u64_reduced(u64::from(
                            classes.as_ref().map_or(0, |c| c[i]),
                        )),
                        Term::Committed(c) => {
                            Goldilocks::from_i64(witness.tensor(c).expect("a tensor")[i])
                        }
                    }
                };
                for lookup in &constraints.lookups {
                    let keys = (0..z.len())
                        .map(|i| {
                            let key = lookup.key.evaluate::<Goldilocks>(|t| term(t, i));
                            u32::try_from(key.to_canonical_u64()).expect("a row")
                        })
                        .collect();
                    let values = lookup.value.as_ref().map(|form| {
                        (0..z.len())
                            .map(|i| {
                                let value = form.evaluate::<Goldilocks>(|t| term(t, i));
                                i64::try_from(value.to_centered_i128()).expect("an i64")
                            })
                            .collect()
                    });
                    expected.lookups.push((lookup.table, keys, values));
                }
            }
            Op::TableMap { table } => {
                let t = self.table(*table);
                let x = self.values(node.inputs[0]);
                let classes = self.classes(&node.inputs, x.len());
                let class = |i: usize| classes.as_ref().map_or(0, |c| c[i]);
                let keys = (0..x.len())
                    .map(|i| {
                        let row = t.index(x[i], class(i)).expect("a key of the table");
                        u32::try_from(row).expect("a row")
                    })
                    .collect();
                let values = (0..x.len())
                    .map(|i| t.lookup(x[i], class(i)).expect("a key of the table"))
                    .collect();
                expected.lookups.push((*table, keys, Some(values)));
            }
            _ => {}
        }
    }

    /// RFC-0009 'SoftmaxRow', relation 1: the key `c_q − x_qi` at a valid position `i ≤ q` and
    /// `K_0` at a masked one, with the exponential its value, for the tile's query heads and held
    /// queries, every key with them.
    fn attention(&self, node: u32, kv_head: u32, block: &QueryBlock, expected: &mut Expected) {
        let node = &self.lowering.graph.nodes[node as usize];
        let Op::SoftmaxRow { exp_table, .. } = node.op else {
            panic!("an attention tile's node is a SoftmaxRow");
        };
        let TableKind::Exp { key_bits, .. } = self.table(exp_table).spec().kind else {
            panic!("the max certificate reads an Exp table");
        };
        let masked = (1u32 << key_bits) - 1;
        let scores = self.values(node.inputs[0]);
        let e = self.values(node.outputs[0]);
        let n = self.shape(node.inputs[0])[1] as usize;
        let arch = &self.model.arch;
        let group = arch.n_heads / arch.n_kv_heads;
        let held = block.queries.start.min(n as u32)..block.queries.end.min(n as u32);
        let (mut keys, mut values) = (Vec::new(), Vec::new());
        for h in kv_head * group..(kv_head + 1) * group {
            for q in held.clone() {
                let row = (h as usize * n + q as usize) * n;
                let top = scores[row..=row + q as usize]
                    .iter()
                    .copied()
                    .max()
                    .expect("a valid key");
                for i in 0..n {
                    keys.push(if i <= q as usize {
                        u32::try_from(top - scores[row + i]).expect("a spread")
                    } else {
                        masked
                    });
                    values.push(e[row + i]);
                }
            }
        }
        expected.lookups.push((exp_table, keys, Some(values)));
    }

    /// What kind of tile `tile` is, for the coverage the test asserts.
    fn kind(&self, tile: &SequenceTile) -> String {
        match tile.tile {
            Tile::Tokens => "Tokens".into(),
            Tile::Table { .. } => "Table".into(),
            Tile::Attention { .. } if tile.shares.iter().all(|s| s.ranges[1].is_empty()) => {
                "Attention, padding alone".into()
            }
            Tile::Attention { .. } => "Attention".into(),
            Tile::Node { node } => {
                let id = self.lowering.graph.nodes[node as usize].id;
                match self.selection.of(id).map(|(_, g)| g) {
                    Some(Gadget::Requant1 { .. }) => "Requant1".into(),
                    Some(Gadget::RequantSplit { .. }) => "RequantSplit".into(),
                    Some(Gadget::RequantWide { .. }) => "RequantWide".into(),
                    _ => self.lowering.graph.nodes[node as usize].op.name().into(),
                }
            }
        }
    }
}

/// `got` holds `want`, element by element.
fn equal<T: PartialEq>(label: &str, got: &[T], want: &[T]) {
    assert_eq!(got.len(), want.len(), "{label}: the length");
    if let Some(i) = got.iter().zip(want).position(|(a, b)| a != b) {
        panic!("{label}: element {i} differs");
    }
}

/// `tile` holds what `expected` says.
fn check(label: &str, tile: &TileWitness, expected: &Expected) {
    assert_eq!(
        tile.shares().len(),
        expected.shares.len(),
        "{label}: the shares"
    );
    for (k, (got, want)) in tile.shares().iter().zip(&expected.shares).enumerate() {
        equal(&format!("{label}, share {k}"), got, want);
    }
    let names = |g: &[(Committed, Vec<i64>)]| g.iter().map(|(c, _)| *c).collect::<Vec<_>>();
    let got: Vec<Committed> = tile.gadget().iter().map(|(c, _)| *c).collect();
    assert_eq!(got, names(&expected.gadget), "{label}: the gadget's values");
    for ((c, got), (_, want)) in tile.gadget().iter().zip(&expected.gadget) {
        equal(&format!("{label}, {c:?}"), got, want);
    }
    let lookups: Vec<_> = tile.lookups().collect();
    assert_eq!(
        lookups.len(),
        expected.lookups.len(),
        "{label}: the lookups"
    );
    for (j, (got, (table, keys, values))) in lookups.iter().zip(&expected.lookups).enumerate() {
        assert_eq!(got.table(), *table, "{label}: lookup {j}'s table");
        equal(&format!("{label}, lookup {j}'s keys"), got.keys(), keys);
        match (got.values(), values) {
            (Some(got), Some(want)) => equal(&format!("{label}, lookup {j}'s values"), got, want),
            (None, None) => {}
            _ => panic!("{label}: lookup {j} has values exactly when its table does"),
        }
    }
}

/// Every lookup of `tile` is a row of its table: `(key, value)` at row `key` (RFC-0008).
fn rows_of_their_tables(label: &str, tile: &TileWitness, first: &FirstPass<'_>) {
    for (j, lookup) in tile.lookups().enumerate() {
        let table = first.table(lookup.table());
        for (x, &key) in lookup.keys().iter().enumerate() {
            assert!(
                u64::from(key) < table.row_count(),
                "{label}: lookup {j}'s key {x} names no row"
            );
            let (row_key, row_value) = table.row(u64::from(key)).expect("a row");
            assert_eq!(row_key, Goldilocks::from_u64_reduced(u64::from(key)));
            if let Some(values) = lookup.values() {
                assert!(
                    row_value == Goldilocks::from_i64(values[x]),
                    "{label}: lookup {j}'s element {x} is not its row's value"
                );
            }
        }
    }
}

// ----- The store --------------------------------------------------------------------------------

/// RFC-0018 'Witness engine': the checkpoints are the layer-boundary residuals, INT32, of the
/// proof unit's tokens; RFC-0037: `c_0` after the embedding, then one after each layer.
#[test]
fn the_checkpoints_are_the_oracles_residuals() {
    let mut checked = 0;
    for c in conformance() {
        let d = c.model.arch.d_model as usize;
        for sequence in &c.sequences {
            let n = sequence.tokens.len();
            let (lowering, _) = lowered(&c.model, u32::try_from(n).expect("small"));
            let engine = Engine::new(&lowering).expect("the tables regenerate");
            let tokens = Secret::new(sequence.tokens.clone());
            let store = Checkpoints::new(&engine, &c.model, &tokens).expect("the checkpoints");
            let first = engine.run(&c.model, &tokens).expect("the first pass");
            let layers = c.model.arch.n_layers as usize;
            assert_eq!(store.len(), layers + 1, "{}: c_0 … c_L", c.name);
            assert_eq!(store.len(), sequence.residuals.len(), "{}", c.name);
            assert_eq!(store.bytes(), (layers + 1) * n * d * 4, "INT32 values");
            for (k, (oracle, &tensor)) in sequence
                .residuals
                .iter()
                .zip(&lowering.checkpoints)
                .enumerate()
            {
                let residual: Vec<i64> = store
                    .residual(k)
                    .expect("a checkpoint")
                    .iter()
                    .map(|&v| i64::from(v))
                    .collect();
                assert_eq!(&residual, oracle, "{}: c_{k} is the oracle's", c.name);
                let pass = first.tensor(tensor).expect("the first pass's tensor");
                assert!(
                    residual[..] == pass[..],
                    "{}: c_{k} is the first pass's",
                    c.name
                );
            }
            assert!(store.residual(layers + 1).is_none());
            checked += 1;
        }
    }
    assert_eq!(checked, 8, "every sequence of every model");
}

// ----- Bit for bit ------------------------------------------------------------------------------

/// RFC-0018 'Pass structure': any tile regenerated from the checkpoints is the first pass's, for
/// every tile kind, query blocks of every size, and lengths that are and are not powers of two.
#[test]
fn every_tile_regenerates_bit_for_bit_from_the_checkpoints() {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut lengths: BTreeSet<usize> = BTreeSet::new();
    let mut tiles = 0usize;
    let mut rng = Rng(0x0112_C4EC_7B0C_0001);
    for c in conformance() {
        let plans: Vec<_> = QUERY_BLOCKS.iter().map(|&q| plan_of(&c.model, q)).collect();
        let vocab = c.model.arch.vocab;
        let mut sequences: Vec<Vec<u32>> = c.sequences.iter().map(|s| s.tokens.clone()).collect();
        for n in [1, 2, 3, 16, 17, c.model.arch.max_context] {
            sequences.push(rng.tokens(vocab, n));
        }
        for tokens in sequences {
            let n = tokens.len();
            let (lowering, selection) = lowered(&c.model, u32::try_from(n).expect("small"));
            let engine = Engine::new(&lowering).expect("the tables regenerate");
            let secret = Secret::new(tokens);
            let witness = engine.run(&c.model, &secret).expect("the first pass");
            let first = FirstPass::new(&lowering, &selection, &c.model, &witness);
            let store = Checkpoints::new(&engine, &c.model, &secret).expect("the checkpoints");
            let resident =
                Resident::new(&engine, &selection, &c.model, &witness).expect("a resident source");
            let regenerator = Regenerator::new(&engine, &selection, &c.model, &secret, &store)
                .expect("a regenerator");
            for (p, plan) in plans.iter().enumerate() {
                let q = plan.claims.gadget_policy.query_block;
                for tile in plan.expand(&lowering).expect("the plan expands") {
                    // A node's tiles do not depend on the query block: the first plan checks them.
                    if p > 0 && !matches!(tile.tile, Tile::Attention { .. }) {
                        continue;
                    }
                    let label = format!(
                        "{}, n = {n}, Q = {q}, {:?} {:?}",
                        c.name, tile.tile, tile.block
                    );
                    if common::unsupported_table(&selection, &tile) {
                        common::assert_unsupported(resident.tile(&tile));
                        common::assert_unsupported(regenerator.tile(&tile));
                        continue;
                    }
                    let expected = first.tile(&tile);
                    let from_first = resident.tile(&tile).expect("the resident source's tile");
                    check(&format!("{label}, resident"), &from_first, &expected);
                    let regenerated = regenerator.tile(&tile).expect("a regenerated tile");
                    check(&format!("{label}, regenerated"), &regenerated, &expected);
                    rows_of_their_tables(&label, &regenerated, &first);
                    seen.insert(first.kind(&tile));
                    tiles += 1;
                }
            }
            lengths.insert(n);
        }
    }
    let expected: BTreeSet<String> = [
        "Tokens",
        "OneHot",
        "ConstMul",
        "RsqrtRow",
        "Requant1",
        "RequantSplit",
        "RequantWide",
        "TableMap",
        "Attention",
        "Attention, padding alone",
        "Table",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    assert_eq!(seen, expected, "every kind of tile");
    assert!(
        lengths.iter().any(|n| n.is_power_of_two()) && lengths.iter().any(|n| !n.is_power_of_two())
    );
    assert!(lengths.contains(&64) && lengths.contains(&1), "{lengths:?}");
    assert!(tiles > 5_000, "{tiles} tiles");
}

// ----- Determinism -------------------------------------------------------------------------------

/// RFC-0018 'Crash consistency' and 'Determinism': a phase rerun after a crash rebuilds its state
/// byte for byte, so neither the checkpoints nor a regenerated tile depends on the thread count.
#[test]
fn the_thread_count_changes_no_checkpoint_and_no_tile() {
    let c = conformance().pop().expect("tiny-4");
    let tokens = c.sequences[1].tokens.clone();
    let n = u32::try_from(tokens.len()).expect("small");
    let (lowering, selection) = lowered(&c.model, n);
    let secret = Secret::new(tokens);
    let threads = |k| NonZeroUsize::new(k).expect("positive");
    let one = Engine::new(&lowering).expect("tables").threads(threads(1));
    let three = Engine::new(&lowering).expect("tables").threads(threads(3));
    let store_one = Checkpoints::new(&one, &c.model, &secret).expect("checkpoints");
    let store_three = Checkpoints::new(&three, &c.model, &secret).expect("checkpoints");
    for k in 0..store_one.len() {
        assert!(store_one.residual(k) == store_three.residual(k), "c_{k}");
    }
    let a = Regenerator::new(&one, &selection, &c.model, &secret, &store_one).expect("a source");
    let b =
        Regenerator::new(&three, &selection, &c.model, &secret, &store_three).expect("a source");
    let plan = plan_of(&c.model, 4);
    for tile in plan.expand(&lowering).expect("the plan expands") {
        if common::unsupported_table(&selection, &tile) {
            common::assert_unsupported(a.tile(&tile));
            common::assert_unsupported(b.tile(&tile));
            continue;
        }
        let (x, y) = (
            a.tile(&tile).expect("a tile"),
            b.tile(&tile).expect("a tile"),
        );
        let label = format!("{:?} {:?}", tile.tile, tile.block);
        assert!(
            x.multiplicities() == y.multiplicities(),
            "{label}: multiplicities"
        );
        assert_eq!(x.shares().len(), y.shares().len());
        for (s, t) in x.shares().iter().zip(y.shares()) {
            equal(&label, s, t);
        }
        for ((_, s), (_, t)) in x.gadget().iter().zip(y.gadget()) {
            equal(&label, s, t);
        }
        for (s, t) in x.lookups().zip(y.lookups()) {
            equal(&label, s.keys(), t.keys());
            assert!(s.values() == t.values(), "{label}");
        }
    }
}

// ----- Refusals ---------------------------------------------------------------------------------

fn usage(error: WitnessError) -> &'static str {
    assert_eq!(error.code(), ErrorCode::Usage, "{error}");
    match error {
        WitnessError::Usage { rule } => rule,
        other => panic!("a usage error, found {other:?}"),
    }
}

/// The tile `tile` with its shares changed by `change`.
fn altered(tile: &SequenceTile, change: impl FnOnce(&mut SequenceTile)) -> SequenceTile {
    let mut tile = tile.clone();
    change(&mut tile);
    tile
}

#[test]
fn sources_refuse_what_is_not_their_sequences() {
    let c = conformance().remove(0);
    let tokens = c.sequences[0].tokens.clone();
    let n = u32::try_from(tokens.len()).expect("small");
    let (lowering, selection) = lowered(&c.model, n);
    let engine = Engine::new(&lowering).expect("tables");
    let secret = Secret::new(tokens.clone());
    let store = Checkpoints::new(&engine, &c.model, &secret).expect("checkpoints");
    let witness = engine.run(&c.model, &secret).expect("the first pass");

    // Tokens of another length, or outside the vocabulary.
    let short = Secret::new(tokens[1..].to_vec());
    let rule = "the sequence has as many tokens as the lowering's positions";
    let error = Checkpoints::new(&engine, &c.model, &short).expect_err("refused");
    assert_eq!(usage(error), rule);
    let error =
        Regenerator::new(&engine, &selection, &c.model, &short, &store).expect_err("refused");
    assert_eq!(usage(error), rule);
    let mut outside = tokens.clone();
    outside[0] = c.model.arch.vocab;
    let outside = Secret::new(outside);
    let rule = "every token lies inside the vocabulary";
    let error = Checkpoints::new(&engine, &c.model, &outside).expect_err("refused");
    assert_eq!(usage(error), rule);
    let error =
        Regenerator::new(&engine, &selection, &c.model, &outside, &store).expect_err("refused");
    assert_eq!(usage(error), rule);

    // A store of another sequence: one position shorter, or another model's of the same shapes.
    let (shorter, _) = lowered(&c.model, n - 1);
    let shorter_engine = Engine::new(&shorter).expect("tables");
    let shorter_tokens = Secret::new(tokens[1..].to_vec());
    let other_store =
        Checkpoints::new(&shorter_engine, &c.model, &shorter_tokens).expect("checkpoints");
    let rule = "the checkpoints are the lowering's, for its sequence";
    let error = Regenerator::new(&engine, &selection, &c.model, &secret, &other_store)
        .expect_err("refused");
    assert_eq!(usage(error), rule);
    let minimal = conformance().remove(1);
    let (minimal_lowering, _) = lowered(&minimal.model, n);
    let minimal_engine = Engine::new(&minimal_lowering).expect("tables");
    let minimal_store =
        Checkpoints::new(&minimal_engine, &minimal.model, &secret).expect("checkpoints");
    assert_eq!(minimal_store.len(), store.len(), "the same shapes");
    let error = Regenerator::new(&engine, &selection, &c.model, &secret, &minimal_store)
        .expect_err("refused");
    assert_eq!(usage(error), rule);

    // An engine without a lowering.
    let bare = Engine::for_graph(&lowering.graph, &lowering.tables).expect("tables");
    let rule = "a model runs on the engine of its lowering";
    assert_eq!(
        usage(Checkpoints::new(&bare, &c.model, &secret).expect_err("refused")),
        rule
    );
    assert_eq!(
        usage(Regenerator::new(&bare, &selection, &c.model, &secret, &store).expect_err("refused")),
        rule
    );
    assert_eq!(
        usage(Resident::new(&bare, &selection, &c.model, &witness).expect_err("refused")),
        rule
    );

    // A first pass of another lowering.
    let shorter_witness = shorter_engine
        .run(&c.model, &shorter_tokens)
        .expect("a first pass");
    assert_eq!(
        usage(Resident::new(&engine, &selection, &c.model, &shorter_witness).expect_err("refused")),
        "the witness is the first pass of the engine's lowering"
    );

    // Tiles that are not the plan's.
    let regenerator =
        Regenerator::new(&engine, &selection, &c.model, &secret, &store).expect("a source");
    let resident = Resident::new(&engine, &selection, &c.model, &witness).expect("a source");
    let plan = plan_of(&c.model, 1);
    let tiles = plan.expand(&lowering).expect("the plan expands");
    let node = tiles
        .iter()
        .find(|t| matches!(t.tile, Tile::Node { .. }))
        .expect("a node's tile");
    let attention = tiles
        .iter()
        .find(|t| matches!(t.tile, Tile::Attention { .. }))
        .expect("an attention tile");
    let Tile::Attention { node: softmax, .. } = attention.tile else {
        unreachable!()
    };
    let nodes = u32::try_from(lowering.graph.nodes.len()).expect("small");
    let kv_heads = c.model.arch.n_kv_heads;
    let cases: Vec<(SequenceTile, &str)> = vec![
        (
            altered(node, |t| t.tile = Tile::Node { node: nodes }),
            "a tile names a node of the lowering's graph",
        ),
        (
            altered(attention, |t| {
                t.tile = Tile::Attention {
                    node: 0,
                    kv_head: 0,
                }
            }),
            "an attention tile names a KV head of a layer's SoftmaxRow",
        ),
        (
            altered(attention, |t| {
                t.tile = Tile::Attention {
                    node: softmax,
                    kv_head: kv_heads,
                }
            }),
            "an attention tile names a KV head of a layer's SoftmaxRow",
        ),
        (
            altered(attention, |t| {
                t.block = Some(QueryBlock {
                    index: 1,
                    queries: 0..1,
                })
            }),
            "an attention tile's query block is one of its sequence's",
        ),
        (
            altered(attention, |t| t.block = None),
            "an attention tile's query block is one of its sequence's",
        ),
        (
            altered(node, |t| {
                t.block = Some(QueryBlock {
                    index: 0,
                    queries: 0..1,
                })
            }),
            "only an attention tile has a query block",
        ),
        (
            altered(node, |t| t.shares[0].ranges[0].end -= 1),
            "a tile's shares are those RFC-0041 gives it",
        ),
        (
            altered(attention, |t| {
                t.shares.pop();
            }),
            "a tile's shares are those RFC-0041 gives it",
        ),
        (
            altered(node, |t| t.tile = Tile::Table { table: 9_999 }),
            "a table tile names a table the gadgets read",
        ),
    ];
    for (tile, rule) in cases {
        let label = format!("{:?} {:?}", tile.tile, tile.block);
        let error = regenerator.tile(&tile).expect_err("refused");
        assert_eq!(usage(error), rule, "{label}, regenerated");
        let error = resident.tile(&tile).expect_err("refused");
        assert_eq!(usage(error), rule, "{label}, resident");
    }
}
