// SPDX-License-Identifier: Apache-2.0 OR MIT

//! What the checkpoints and the regenerated tiles hold (RFC-0018 'Witness engine', 01 §6
//! INV-ARCH-3), as a counting allocator sees it. `allocation-counter` counts the current thread's
//! allocations only, so every engine here runs on one thread. Borrowed engine tables are built
//! before measurement; source-owned public data is measured separately.
//!
//! - **The store** keeps `(L + 1)·n·d_model` INT32 values and nothing else, and computing it holds
//!   less than the first full pass, which keeps every tensor.
//! - **Nothing between tiles.** Regenerating a tile and dropping it frees every byte the
//!   regeneration allocated: a source holds its checkpoints, and the tile it is regenerating.
//! - **Depth.** A tile's peak, and every byte its regeneration allocates, are the same at its
//!   template position in every layer: regeneration reruns the tile's layer from its checkpoint,
//!   never the layers before it.
//! - **No score tensor.** Doubling the sequence at most doubles the peak of every node's tile and of
//!   every attention tile of one query: none of them holds a `[n_heads, n, n]` tensor, whose bytes
//!   would quadruple.
//! - **The figures**, printed for each model at the whole context: the store, the first pass's peak,
//!   the largest tile, and the largest regeneration peak, against the store plus that tile.

#![allow(clippy::print_stderr)] // the measured figures

mod common;

use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use allocation_counter::{AllocationInfo, measure};
use common::{Rng, conformance, lowered, plan_of};
use tessera_codec::Secret;
use tessera_compiler::schedule::LayerMap;
use tessera_compiler::{Lowering, SequenceTile, Tile};
use tessera_ir::{IntegerModel, NodeId};
use tessera_witness::{Checkpoints, Engine, Regenerator, WitnessSource};

/// One thread, so that the counter sees every allocation.
fn one() -> NonZeroUsize {
    NonZeroUsize::new(1).expect("one")
}

/// A tile's identity across lengths and layers: its kind, its node's template position and layer
/// (or its node outside the layers), its KV head and its query block.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    kind: &'static str,
    position: Option<u32>,
    layer: Option<u32>,
    node: u32,
    kv_head: u32,
    block: u32,
}

fn key(layers: &LayerMap, tile: &SequenceTile) -> Key {
    let (kind, node, kv_head) = match tile.tile {
        Tile::Tokens => ("tokens", 0, 0),
        Tile::Node { node } => ("node", node, 0),
        Tile::Attention { node, kv_head } => ("attention", node, kv_head),
        Tile::Table { table } => ("table", table, 0),
    };
    let place = matches!(tile.tile, Tile::Node { .. } | Tile::Attention { .. })
        .then(|| layers.position(NodeId(node)))
        .flatten();
    Key {
        kind,
        position: place.map(|(_, p)| p),
        layer: place.map(|(l, _)| l),
        node: if place.is_some() { 0 } else { node },
        kv_head,
        block: tile.block.as_ref().map_or(0, |b| b.index),
    }
}

/// One tile's regeneration: its peak, the bytes it allocated in all, and the tile's own bytes.
#[derive(Clone, Copy)]
struct Figures {
    peak: u64,
    total: u64,
    bytes: usize,
}

/// A sequence's measurements under one plan: each tile's figures, the store's, and the first
/// pass's peak.
struct Measured {
    tiles: BTreeMap<Key, Figures>,
    store: usize,
    store_retained: i64,
    store_peak: u64,
    first_pass: u64,
    source_retained: u64,
    source_peak: u64,
    visit_metadata: usize,
}

fn measured(model: &IntegerModel, lowering: &Lowering, tokens: &[u32], q: u32) -> Measured {
    let n = u32::try_from(tokens.len()).expect("small");
    let (_, selection) = lowered(model, n);
    let layers = LayerMap::new(lowering).expect("one template");
    let engine = Engine::new(lowering).expect("tables").threads(one());
    let secret = Secret::new(tokens.to_vec());
    let first = measure(|| drop(engine.run(model, &secret).expect("the first pass")));
    let mut store = None;
    let built = measure(|| {
        store = Some(Checkpoints::new(&engine, model, &secret).expect("the checkpoints"));
    });
    let store = store.expect("built");
    let mut source = None;
    let baseline = measure(|| {
        source = Some(
            Regenerator::new(&engine, &selection, model, &secret, &store).expect("a regenerator"),
        );
    });
    let source = source.expect("source constructed");
    let plan = plan_of(model, q);
    let mut tiles = BTreeMap::new();
    let mut visit_metadata = 0;
    for tile in plan.expand(lowering).expect("the plan expands") {
        visit_metadata = visit_metadata.max(
            tile.shares.capacity() * size_of::<tessera_compiler::Share>()
                + tile
                    .shares
                    .iter()
                    .map(|s| s.ranges.capacity() * size_of::<core::ops::Range<u32>>())
                    .sum::<usize>(),
        );
        let mut bytes = 0;
        let info: AllocationInfo = measure(|| {
            let regenerated = source.tile(&tile).expect("a tile");
            bytes = regenerated.bytes();
            drop(regenerated);
        });
        assert_eq!(
            info.bytes_current, 0,
            "{:?} {:?}: a regeneration frees all it allocates",
            tile.tile, tile.block
        );
        assert_eq!(info.count_current, 0);
        let figures = Figures {
            peak: info.bytes_max,
            total: info.bytes_total,
            bytes,
        };
        tiles.insert(key(&layers, &tile), figures);
    }
    Measured {
        tiles,
        store: store.bytes(),
        store_retained: built.bytes_current,
        store_peak: built.bytes_max,
        first_pass: first.bytes_max,
        source_retained: u64::try_from(baseline.bytes_current).expect("public source baseline"),
        source_peak: baseline.bytes_max,
        visit_metadata,
    }
}

#[test]
fn regeneration_holds_the_checkpoints_and_one_tile() {
    let mut rng = Rng(0x0112_3E30_0000_0001);
    eprintln!(
        "{:<16} {:>3} {:>4} {:>12} {:>12} {:>12} {:>12} {:>12} {:>7} {:>7}  largest peak's tile",
        "model",
        "n",
        "Q",
        "checkpoints",
        "first pass",
        "store pass",
        "largest tile",
        "largest peak",
        "peak/T",
        "(C+P)/W",
    );
    for c in conformance() {
        let context = c.model.arch.max_context;
        let long = rng.tokens(c.model.arch.vocab, context);
        let half = long[..context as usize / 2].to_vec();
        let (lowering, _) = lowered(&c.model, context);
        let (half_lowering, _) = lowered(&c.model, context / 2);
        for q in [512, 1] {
            let full = measured(&c.model, &lowering, &long, q);
            // The store keeps its residuals alone, beside a few words of bookkeeping per
            // checkpoint, and building it holds less than the first pass's witness.
            let layers = c.model.arch.n_layers as usize;
            let d = c.model.arch.d_model as usize;
            assert_eq!(full.store, (layers + 1) * context as usize * d * 4);
            let retained = usize::try_from(full.store_retained).expect("retained");
            assert!(
                retained >= full.store && retained <= full.store + 64 * (layers + 1),
                "{}: the store keeps {retained} bytes for {} of residuals",
                c.name,
                full.store
            );
            assert!(full.store_peak < full.first_pass, "{}", c.name);
            // Depth: every layer's tile peaks, and allocates in all, as the first layer's at its
            // position.
            for (k, figures) in &full.tiles {
                if let Some(layer) = k.layer.filter(|&l| l > 0) {
                    let first_layer = Key {
                        layer: Some(0),
                        ..k.clone()
                    };
                    let base = full.tiles[&first_layer];
                    let label = format!("{}, Q = {q}: {k:?} in layer {layer}", c.name);
                    assert_eq!(figures.peak, base.peak, "{label}: the peak");
                    assert_eq!(figures.total, base.total, "{label}: the bytes allocated");
                }
            }
            let (largest_key, largest) = full
                .tiles
                .iter()
                .max_by_key(|(_, f)| f.peak)
                .expect("tiles");
            let largest_tile = full.tiles.values().map(|f| f.bytes).max().expect("tiles");
            // A counting tile holds its one column plus at most one ordinary regeneration's
            // working set. This includes replay scratch, not only returned buffer sizes.
            let replay_peak = full
                .tiles
                .iter()
                .filter(|(k, _)| k.kind != "table")
                .map(|(_, f)| f.peak)
                .max()
                .expect("ordinary tiles");
            for (k, figures) in full.tiles.iter().filter(|(k, _)| k.kind == "table") {
                assert!(
                    figures.peak <= figures.bytes as u64 + replay_peak + full.visit_metadata as u64,
                    "{} Q={q} {k:?}: peak={} column={} replay={replay_peak} plus public visit metadata",
                    c.name,
                    figures.peak,
                    figures.bytes
                );
            }
            // Count the source's public tables/constraints too; never hide them in C or T.
            let whole_peak =
                retained as u64 + (full.source_retained + largest.peak).max(full.source_peak);
            let whole_peak = whole_peak.max(full.store_peak);
            assert!(
                whole_peak * 2 < full.first_pass,
                "{}: full accounted residency is below half the resident witness",
                c.name
            );
            eprintln!(
                "{} Q={q}: checkpoint allocations={retained}, public source retained={}, public construction peak={}, C+source+working peak={whole_peak}",
                c.name, full.source_retained, full.source_peak
            );
            let ratio = largest.peak as f64 / largest_tile as f64;
            let share = (full.store as f64 + largest.peak as f64) / full.first_pass as f64;
            eprintln!(
                "{:<16} {:>3} {:>4} {:>12} {:>12} {:>12} {:>12} {:>12} {:>7.2} {:>7.3}  {largest_key:?}",
                c.name,
                context,
                q,
                full.store,
                full.first_pass,
                full.store_peak,
                largest_tile,
                largest.peak,
                ratio,
                share,
            );
            if q != 1 {
                continue;
            }
            // Linear in the length: a node's tile, or an attention tile of one query.
            let halved = measured(&c.model, &half_lowering, &half, q);
            for (k, figures) in &halved.tiles {
                if matches!(k.kind, "table" | "tokens") {
                    continue;
                }
                let peak = figures.peak;
                let doubled = full
                    .tiles
                    .get(k)
                    .expect("the same tile at twice the length")
                    .peak;
                assert!(
                    doubled <= 2 * peak,
                    "{}: {k:?} peaks at {doubled} bytes for n = {context} and {peak} for n = {}",
                    c.name,
                    context / 2
                );
            }
        }
    }
}
