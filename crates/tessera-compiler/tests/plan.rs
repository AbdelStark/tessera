// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The tiles and the execution plan (RFC-0041) of the conformance models, expanded at sequence
//! lengths from one token to the whole context, powers of two and their neighbours, with query
//! blocks that hold the whole sequence or split it down to one query each.
//!
//! - **Partition.** Every element of every tensor committed in the advice, the tokens among them,
//!   lies in exactly one tile, and no tile holds an element of any other tensor.
//! - **Order.** The tiles follow node order, a layer's attention node's by KV head, then query
//!   block, and the tables come last, by id. A query block holds `Q' = min(Q, N)` queries. A gadget
//!   outside the layers, even a `SoftmaxRow`, takes a node's tile.
//! - **Heads.** An attention tile holds the query heads that read its KV head's keys, as the
//!   conformance oracle's traces show them.
//! - **Length.** Node and tensor ids, and so the tiles, are the same for every length.
//! - **Determinism and secrecy.** Two builds give the same bytes. A plan holds no weight, and a
//!   model whose weights and channel parameters change has the same plan.
//! - **Binding.** A plan with one tile changed, removed, added or moved, or with another claim
//!   list, is refused as `ERR-USAGE`.
//! - **Encoding.** A plan decodes from its encoding and its envelope to itself, a tile's tag is its
//!   declaration index, and the golden plans of `spec/vectors/codec/` decode to the plans the
//!   compiler derives.

use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};

use serde_json::Value;
use tessera_codec::error::ErrorCode;
use tessera_codec::{Ascii, DecodeLimits, decode, encode, envelope, schema};
use tessera_compiler::{
    CompileError, ExecutionPlan, Gadget, Lowering, PublicModel, QueryBlock, Tile, analyze,
    check_plan, claim_list, lower, plan, report, select, tiles,
};
use tessera_ir::{
    AdviceLayoutParams, ClaimList, CommitmentClass, CommitmentLayout, GadgetPolicy, Graph,
    IntRange, IntType, IntegerModel, Node, NodeId, NodeKind, Op, ParamSetId, TableId, TensorId,
    TensorSpec, WeightLayoutParams,
};
use tessera_tables::TableKind;

/// `test-small/1`'s layout (RFC-0014 'Parameter sets').
const TEST_SMALL: CommitmentLayout = CommitmentLayout {
    advice: AdviceLayoutParams {
        log2_k: 10,
        log2_inv_rate: 1,
        mask_coeffs: 16,
        row_block: 4,
        segments: 1,
    },
    weights: WeightLayoutParams {
        log2_k: 8,
        log2_inv_rate: 2,
        mask_coeffs: 64,
        blinding_rows: 32,
    },
};

/// Query blocks: RFC-0006's default, which holds the whole context of every conformance model, and
/// two that split it, into blocks of four queries and of one.
const QUERY_BLOCKS: [u32; 3] = [512, 4, 1];

/// The refusals, as the rules they name.
const QUERY_BLOCK: &str = "query_block is a power of two: RFC-0034's layout takes its logarithm";
const NOT_THE_MODELS: &str =
    "the claim list is the model's, for its own gadget policy, layout and parameter set";
const OTHER_CLAIMS: &str = "an execution plan holds D's claim list";
const OTHER_TILES: &str = "an execution plan holds the tiles D's claim list gives";
const OUTSIDE: &str = "the plan's tiles name nodes of the lowering's graph";
const NOT_ATTENTION: &str = "an attention tile names a KV head of a SoftmaxRow over n_heads heads";

fn vectors() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors")
}

fn json(path: PathBuf) -> Value {
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).expect("JSON")
}

fn hex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).expect("hex"))
        .collect()
}

fn model_file(file: &str) -> IntegerModel {
    let bytes = fs::read(vectors().join("models").join(file)).expect("the artifact");
    envelope::open(&bytes, &schema::INTEGER_MODEL, DecodeLimits::INTEGER_MODEL)
        .expect("an IntegerModel")
}

/// The conformance models (07 §5), each with its name and its artifact's file.
fn models() -> Vec<(String, String, IntegerModel)> {
    let index = json(vectors().join("models/models.json"));
    let models: Vec<_> = index["models"]
        .as_array()
        .expect("models")
        .iter()
        .map(|entry| {
            let file = entry["file"].as_str().expect("a file");
            let name = entry["name"].as_str().expect("a name");
            (name.to_owned(), file.to_owned(), model_file(file))
        })
        .collect();
    assert_eq!(models.len(), 4, "every conformance model");
    models
}

/// `model`'s claim list for `test-small/1`, under the default gadget policy with query blocks of
/// `query_block` positions.
fn claims(model: &IntegerModel, query_block: u32) -> ClaimList {
    let public = PublicModel::of(model);
    let lowering = lower(public, 1).expect("the model lowers");
    let ranges = analyze(&lowering.graph, &lowering.tables).expect("ranges");
    let policy = GadgetPolicy {
        query_block,
        ..GadgetPolicy::default()
    };
    let selection =
        select(&lowering.graph, &ranges, &lowering.tables, &policy).expect("every gadget fits");
    let params = ParamSetId(Ascii::new("test-small/1").expect("ASCII"));
    claim_list(public, &selection, &policy, TEST_SMALL, params)
        .expect("the maximum advice layout fits")
}

/// The lengths a model's tiles are expanded at: powers of two and their neighbours, and the whole
/// context.
fn lengths(model: &IntegerModel) -> Vec<u32> {
    let context = model.arch.max_context;
    let mut lengths: Vec<u32> = [1, 2, 3, 4, 5, 7, 8, 9, 16, 17, context]
        .into_iter()
        .filter(|&n| n <= context)
        .collect();
    lengths.dedup();
    lengths
}

/// Whether a tensor of `kind` is committed in the advice: advice or a lookup output (03 §3).
fn committed(kind: NodeKind) -> bool {
    matches!(
        kind,
        NodeKind::Committed {
            commitment: CommitmentClass::Advice
        } | NodeKind::LookupOutput { .. }
    )
}

/// The row-major index, in a tensor of `shape`, of every element whose coordinate on each axis lies
/// in that axis's range.
fn indices(shape: &[u32], ranges: &[Range<u32>]) -> Vec<usize> {
    assert_eq!(ranges.len(), shape.len(), "one range per axis");
    let mut indices = vec![0usize];
    for (range, &len) in ranges.iter().zip(shape) {
        assert!(
            range.start <= range.end && range.end <= len,
            "{range:?} lies in an axis of {len}"
        );
        indices = indices
            .iter()
            .flat_map(|&outer| {
                range
                    .clone()
                    .map(move |i| outer * len as usize + i as usize)
            })
            .collect();
    }
    indices
}

/// The layer whose nodes hold `node`: those after the producer of its checkpoint, up to and
/// including the producer of the next (RFC-0037 'Layers and kinds').
fn layer_of(lowering: &Lowering, node: u32) -> Option<usize> {
    let producer = |&t| lowering.graph.producer(t).expect("a node's output").0;
    let ends: Vec<u32> = lowering.checkpoints.iter().map(producer).collect();
    ends.windows(2)
        .position(|pair| pair[0] < node && node <= pair[1])
}

// ----- Partition and order ---------------------------------------------------------------------

/// RFC-0041 'Tiles': once the attention tiles are expanded, every element of every tensor committed
/// in the advice, the tokens among them, lies in exactly one tile, and no tile holds an element of
/// any other tensor. Elements are counted one by one.
#[test]
fn every_committed_element_lies_in_exactly_one_tile() {
    for (name, _, model) in models() {
        let public = PublicModel::of(&model);
        let plans: Vec<ExecutionPlan> = QUERY_BLOCKS
            .iter()
            .map(|&q| plan(public, &claims(&model, q)).expect("a plan"))
            .collect();
        for n in lengths(&model) {
            let lowering = lower(public, n).expect("the model lowers");
            let graph = &lowering.graph;
            for plan in &plans {
                let label = format!(
                    "{name}, n = {n}, Q = {}",
                    plan.claims.gadget_policy.query_block
                );
                let mut counts: Vec<Option<Vec<u32>>> = graph
                    .tensors
                    .iter()
                    .map(|t| {
                        let elements = t.shape.iter().map(|&d| d as usize).product();
                        committed(t.kind).then(|| vec![0; elements])
                    })
                    .collect();
                for tile in plan.expand(&lowering).expect("the plan expands") {
                    for share in &tile.shares {
                        let spec = &graph.tensors[share.tensor.0 as usize];
                        let count = counts[share.tensor.0 as usize].as_mut().unwrap_or_else(|| {
                            panic!(
                                "{label}: {:?} holds tensor {}, which is {:?}",
                                tile.tile, share.tensor.0, spec.kind
                            )
                        });
                        for index in indices(&spec.shape, &share.ranges) {
                            count[index] += 1;
                        }
                    }
                }
                let mut elements = 0;
                for (t, count) in counts.iter().enumerate() {
                    let Some(count) = count else { continue };
                    if let Some(i) = count.iter().position(|&k| k != 1) {
                        panic!(
                            "{label}: element {i} of tensor {t} lies in {} tiles",
                            count[i]
                        );
                    }
                    elements += count.len();
                }
                // The tokens and their one-hot vectors at least.
                let vocab = model.arch.vocab as usize;
                assert!(elements > n as usize * (1 + vocab), "{label}: {elements}");
            }
        }
    }
}

/// The tables some lookup of `lowering` reads, by id: those in which the commitment report counts a
/// lookup.
fn read_tables(public: PublicModel<'_>, lowering: &Lowering) -> Vec<u32> {
    let policy = GadgetPolicy::default();
    let ranges = analyze(&lowering.graph, &lowering.tables).expect("ranges");
    let selection =
        select(&lowering.graph, &ranges, &lowering.tables, &policy).expect("every gadget fits");
    let report = report(public, lowering, &selection, &policy).expect("a report");
    let mut read: Vec<u32> = report
        .tables
        .iter()
        .filter(|t| t.lookups_per_token + t.score_lookups > 0)
        .map(|t| t.id.0)
        .collect();
    read.sort_unstable();
    read
}

/// RFC-0041 'Production order', from the graph: the tokens; then, in node order, each layer's
/// `SoftmaxRow` by KV head, then query block, and every other node with a committed output; then
/// the tables `read`. Query block `b` holds the queries `b·Q'` to `(b + 1)·Q' − 1` of the sequence
/// padded to `N = 2^{⌈log2 n⌉}`, with `Q' = min(Q, N)`.
fn production_order(
    model: &IntegerModel,
    lowering: &Lowering,
    read: &[u32],
    query_block: u32,
    n: u32,
) -> Vec<(Tile, Option<QueryBlock>)> {
    let graph = &lowering.graph;
    let padded = n.next_power_of_two();
    let block = query_block.min(padded);
    let mut order = vec![(Tile::Tokens, None)];
    for node in &graph.nodes {
        let id = node.id.0;
        if layer_of(lowering, id).is_some() && matches!(node.op, Op::SoftmaxRow { .. }) {
            for kv_head in 0..model.arch.n_kv_heads {
                for b in 0..padded / block {
                    let queries = b * block..(b + 1) * block;
                    let tile = Tile::Attention { node: id, kv_head };
                    order.push((tile, Some(QueryBlock { index: b, queries })));
                }
            }
        } else if node
            .outputs
            .iter()
            .any(|&t| committed(graph.tensors[t.0 as usize].kind))
        {
            order.push((Tile::Node { node: id }, None));
        }
    }
    order.extend(read.iter().map(|&table| (Tile::Table { table }, None)));
    order
}

/// RFC-0041 'Production order': the expanded tiles come in the order the RFC fixes, from the
/// tokens to the tables, with each query block's queries.
#[test]
fn tiles_follow_node_order_with_the_tables_last() {
    for (name, _, model) in models() {
        let public = PublicModel::of(&model);
        let plans: Vec<ExecutionPlan> = QUERY_BLOCKS
            .iter()
            .map(|&q| plan(public, &claims(&model, q)).expect("a plan"))
            .collect();
        for n in lengths(&model) {
            let lowering = lower(public, n).expect("the model lowers");
            let read = read_tables(public, &lowering);
            for plan in &plans {
                let q = plan.claims.gadget_policy.query_block;
                let expected = production_order(&model, &lowering, &read, q, n);
                let found: Vec<(Tile, Option<QueryBlock>)> = plan
                    .expand(&lowering)
                    .expect("the plan expands")
                    .into_iter()
                    .map(|tile| (tile.tile, tile.block))
                    .collect();
                assert_eq!(found, expected, "{name}, n = {n}, Q = {q}");
            }
        }
    }
}

/// RFC-0041 'Tiles': a layer's attention node is its `SoftmaxRow`, and a node commits a value when
/// it has an output committed in the advice, or when it is a gadget, every one of which commits
/// values of its own (RFC-0009). A `SoftmaxRow` and an `Argmax` added to the epilogue, outside the
/// layers, each take one node's tile: the first holds its exponentials and reciprocals whole, and
/// the second, which has no output, holds its certificate's maximum alone.
#[test]
fn a_gadget_outside_the_layers_takes_a_node_tile() {
    let models = models();
    let (name, _, model) = &models[0];
    let public = PublicModel::of(model);
    let n = 5;
    let mut lowering = lower(public, n).expect("the model lowers");
    let ranges = analyze(&lowering.graph, &lowering.tables).expect("ranges");
    let policy = GadgetPolicy::default();
    let selection =
        select(&lowering.graph, &ranges, &lowering.tables, &policy).expect("every gadget fits");
    let before = tiles(public, &lowering, &selection).expect("tiles");

    // The logits' products with themselves, `[n, n]`, then their softmax and their argmax.
    let graph = &mut lowering.graph;
    let tensor = |graph: &mut Graph, shape: Vec<u32>, kind: NodeKind| {
        let id = TensorId(u32::try_from(graph.tensors.len()).expect("few tensors"));
        graph.tensors.push(TensorSpec {
            id,
            shape,
            ty: IntType::I32,
            range: IntRange { lo: 0, hi: 0 },
            kind,
        });
        id
    };
    let exp = selection
        .tables
        .iter()
        .find(|spec| matches!(spec.kind, TableKind::Exp { .. }))
        .map(|spec| spec.id)
        .expect("an Exp table");
    let scores = tensor(graph, vec![n, n], NodeKind::Virtual);
    let e = tensor(graph, vec![n, n], NodeKind::LookupOutput { table: exp });
    let r = tensor(
        graph,
        vec![n],
        NodeKind::Committed {
            commitment: CommitmentClass::Advice,
        },
    );
    let (logits, one_hot) = (lowering.logits, lowering.one_hot);
    let first = u32::try_from(graph.nodes.len()).expect("few nodes");
    for (k, (op, inputs, outputs)) in [
        (
            Op::MatMulAct {
                transpose_rhs: true,
            },
            vec![logits, logits],
            vec![scores],
        ),
        (
            Op::SoftmaxRow {
                exp_table: exp,
                sigma: 16,
            },
            vec![scores],
            vec![e, r],
        ),
        (Op::Argmax, vec![logits, one_hot], vec![]),
    ]
    .into_iter()
    .enumerate()
    {
        graph.nodes.push(Node {
            id: NodeId(first + u32::try_from(k).expect("three")),
            op,
            inputs,
            outputs,
        });
    }
    graph.validate().expect("a graph RFC-0006 admits");

    let after = tiles(public, &lowering, &selection).expect("tiles");
    let tables = before
        .iter()
        .position(|t| matches!(t, Tile::Table { .. }))
        .expect("a table's tile");
    let mut expected = before.clone();
    expected.splice(
        tables..tables,
        [
            Tile::Node { node: first + 1 },
            Tile::Node { node: first + 2 },
        ],
    );
    assert_eq!(after, expected, "{name}");

    let plan = ExecutionPlan {
        claims: claims(model, 512),
        tiles: after,
    };
    let expanded = plan.expand(&lowering).expect("the plan expands");
    let held = |node: u32| -> Vec<(TensorId, Vec<Range<u32>>)> {
        expanded
            .iter()
            .find(|t| t.tile == Tile::Node { node })
            .expect("the node's tile")
            .shares
            .iter()
            .map(|s| (s.tensor, s.ranges.clone()))
            .collect()
    };
    let whole = |t: TensorId| -> Vec<Range<u32>> {
        let shape = &lowering.graph.tensors[t.0 as usize].shape;
        shape.iter().map(|&len| 0..len).collect()
    };
    assert_eq!(held(first + 1), [(e, whole(e)), (r, whole(r))]);
    assert_eq!(whole(e), [0..n, 0..n]);
    assert!(held(first + 2).is_empty());
}

/// RFC-0041 'Tiles': the attention tile of KV head `g` holds the query heads that read `g`'s keys.
/// The conformance oracle's traces show which: each maximum of a head, its row's greatest score
/// `max_{i ≤ q} q_h · k_i`, is that of one KV head's keys only.
#[test]
fn an_attention_tile_holds_the_query_heads_that_read_its_keys() {
    let models = models();
    let traces = json(vectors().join("models/traces.json"));
    let traced = traces["models"].as_array().expect("models");
    assert_eq!(traced.len(), models.len());
    let mut tiles_checked = 0;
    for ((name, _, model), entry) in models.iter().zip(traced) {
        assert_eq!(entry["name"].as_str(), Some(name.as_str()));
        let arch = &model.arch;
        let (heads, kv, width) = (
            arch.n_heads as usize,
            arch.n_kv_heads as usize,
            arch.d_head as usize,
        );
        let public = PublicModel::of(model);
        let plans = [512, 1].map(|q| plan(public, &claims(model, q)).expect("a plan"));
        for sequence in entry["sequences"].as_array().expect("sequences") {
            let tensors = sequence["tensors"].as_array().expect("tensors");
            let tensor = |wanted: &str| -> Vec<i128> {
                let found = tensors
                    .iter()
                    .find(|t| t["name"].as_str() == Some(wanted))
                    .unwrap_or_else(|| panic!("{name}: no {wanted}"));
                found["values"]
                    .as_array()
                    .expect("values")
                    .iter()
                    .map(|v| i128::from(v.as_i64().expect("an integer")))
                    .collect()
            };
            let n = tensor("one-hot").len() / arch.vocab as usize;
            // The KV head each query head of each layer reads, from the oracle's values alone.
            let reads: Vec<Vec<usize>> = (0..arch.n_layers)
                .map(|layer| {
                    let q = tensor(&format!("layer {layer} RopeQ"));
                    let k = tensor(&format!("layer {layer} RopeK"));
                    let maxima = tensor(&format!("layer {layer} maxima"));
                    let score = |p: usize, h: usize, i: usize, g: usize| -> i128 {
                        (0..width)
                            .map(|d| q[(p * heads + h) * width + d] * k[(i * kv + g) * width + d])
                            .sum()
                    };
                    let matches = |h: usize, g: usize| {
                        (0..n).all(|p| {
                            let greatest = (0..=p).map(|i| score(p, h, i, g)).max();
                            greatest == Some(maxima[h * n + p])
                        })
                    };
                    (0..heads)
                        .map(|h| {
                            let read: Vec<usize> = (0..kv).filter(|&g| matches(h, g)).collect();
                            assert_eq!(read.len(), 1, "{name}, layer {layer}, head {h}: {read:?}");
                            read[0]
                        })
                        .collect()
                })
                .collect();
            let lowering = lower(public, u32::try_from(n).expect("short")).expect("lowers");
            for plan in &plans {
                for tile in plan.expand(&lowering).expect("the plan expands") {
                    let Tile::Attention { node, kv_head } = tile.tile else {
                        continue;
                    };
                    let layer = layer_of(&lowering, node).expect("a layer's node");
                    let expected: Vec<u32> = (0..heads)
                        .filter(|&h| reads[layer][h] == kv_head as usize)
                        .map(|h| u32::try_from(h).expect("few heads"))
                        .collect();
                    assert_eq!(
                        tile.shares.len(),
                        2,
                        "{name}: the exponentials and reciprocals"
                    );
                    for share in &tile.shares {
                        let held: Vec<u32> = share.ranges[0].clone().collect();
                        assert_eq!(
                            held, expected,
                            "{name}, n = {n}, layer {layer}, KV head {kv_head}"
                        );
                    }
                    tiles_checked += 1;
                }
            }
        }
    }
    assert!(tiles_checked >= 100, "{tiles_checked} attention tiles");
}

// ----- Length, determinism, secrecy ------------------------------------------------------------

/// RFC-0041 'The plan': a lowering's node and tensor ids do not depend on the sequence's length,
/// only its shapes do, so the tiles any length's lowering gives are the plan's.
#[test]
fn the_tiles_are_the_same_for_every_length() {
    for (name, _, model) in models() {
        let public = PublicModel::of(&model);
        let claims = claims(&model, 512);
        let plan = plan(public, &claims).expect("a plan");
        let one = lower(public, 1).expect("the model lowers");
        let kinds =
            |l: &Lowering| -> Vec<_> { l.graph.tensors.iter().map(|t| (t.id, t.kind)).collect() };
        let mut all: Vec<u32> = (1..=16).chain(lengths(&model)).collect();
        all.sort_unstable();
        all.dedup();
        for n in all {
            let lowering = lower(public, n).expect("the model lowers");
            let label = format!("{name}, n = {n}");
            assert_eq!(lowering.graph.nodes, one.graph.nodes, "{label}: the nodes");
            assert_eq!(kinds(&lowering), kinds(&one), "{label}: the tensors");
            assert_eq!(lowering.tokens, one.tokens, "{label}: the tokens");
            assert_eq!(
                lowering.checkpoints, one.checkpoints,
                "{label}: the checkpoints"
            );
            let ranges = analyze(&lowering.graph, &lowering.tables).expect("ranges");
            let selection = select(
                &lowering.graph,
                &ranges,
                &lowering.tables,
                &claims.gadget_policy,
            )
            .expect("every gadget fits");
            let tiles = tiles(public, &lowering, &selection).expect("tiles");
            assert_eq!(tiles, plan.tiles, "{label}: the tiles");
        }
    }
}

/// The weights' canary: a run of distinct values no plan holds.
const CANARY: [i8; 12] = [113, -97, 61, -29, 107, -71, 89, -53, 101, -83, 67, -41];

/// RFC-0041 'The plan', determinism: two builds give the same bytes, and the plan holds no secret
/// value. A model whose every weight holds a canary pattern and whose channel parameters change has
/// the same plan, and the canary, which its own encoding holds, is not in the plan's envelope.
#[test]
fn a_plan_is_deterministic_and_holds_no_weight() {
    for (name, _, model) in models() {
        let build = |m: &IntegerModel| plan(PublicModel::of(m), &claims(m, 512)).expect("a plan");
        let bytes = encode(&build(&model));
        assert_eq!(encode(&build(&model)), bytes, "{name}: two builds");

        let mut hidden = model.clone();
        for weight in &mut hidden.weights {
            for (i, w) in weight.data.iter_mut().enumerate() {
                *w = CANARY[i % CANARY.len()];
            }
        }
        assert!(
            !hidden.quant.channels.is_empty(),
            "{name} has channel parameters"
        );
        for params in &mut hidden.quant.channels {
            params.class.reverse();
            params.multiplier.reverse();
        }
        let canary: Vec<u8> = CANARY.iter().map(|&w| w.to_le_bytes()[0]).collect();
        let holds = |bytes: &[u8]| bytes.windows(canary.len()).any(|w| w == canary);
        assert!(
            holds(&encode(&hidden)),
            "{name}: the model holds its canary"
        );
        let plan = build(&hidden);
        assert_eq!(
            encode(&plan),
            bytes,
            "{name}: the plan reads no hidden value"
        );
        assert!(
            !holds(&envelope::seal(&schema::EXECUTION_PLAN, &plan)),
            "{name}: the plan holds a weight"
        );
    }
}

// ----- Binding ---------------------------------------------------------------------------------

/// `tile` changed into another: the next node, KV head or table, or a node for the tokens.
fn changed(tile: Tile) -> Tile {
    match tile {
        Tile::Tokens => Tile::Node { node: 0 },
        Tile::Node { node } => Tile::Node { node: node + 1 },
        Tile::Attention { node, kv_head } => Tile::Attention {
            node,
            kv_head: kv_head + 1,
        },
        Tile::Table { table } => Tile::Table { table: table + 1 },
    }
}

/// The tile lists that differ from `tiles` by one edit, each with what it does: the first and the
/// last tile of each kind changed, removed or added a second time, two tiles moved, and a tile
/// added at the end.
fn edits(tiles: &[Tile]) -> Vec<(String, Vec<Tile>)> {
    let kind = |t: &Tile| match t {
        Tile::Tokens => 0,
        Tile::Node { .. } => 1,
        Tile::Attention { .. } => 2,
        Tile::Table { .. } => 3,
    };
    let mut places = Vec::new();
    for k in 0..4 {
        let of_kind: Vec<usize> = (0..tiles.len()).filter(|&i| kind(&tiles[i]) == k).collect();
        assert!(!of_kind.is_empty(), "a tile of each kind");
        places.extend([of_kind[0], of_kind[of_kind.len() - 1]]);
    }
    places.sort_unstable();
    places.dedup();
    let mut edits = Vec::new();
    for &i in &places {
        let mut other = tiles.to_vec();
        other[i] = changed(other[i]);
        edits.push((format!("tile {i} changed"), other));
        let mut other = tiles.to_vec();
        other.remove(i);
        edits.push((format!("tile {i} removed"), other));
        let mut other = tiles.to_vec();
        other.insert(i, tiles[i]);
        edits.push((format!("tile {i} added twice"), other));
    }
    let first = |k| {
        (0..tiles.len())
            .find(|&i| kind(&tiles[i]) == k)
            .expect("a tile")
    };
    let attention = first(2);
    let table = first(3);
    for (what, (a, b)) in [
        ("two KV heads swapped", (attention, attention + 1)),
        (
            "the last node's and the first table's swapped",
            (table - 1, table),
        ),
        ("the tokens moved after the first node", (0, 1)),
    ] {
        let mut other = tiles.to_vec();
        other.swap(a, b);
        edits.push((what.to_owned(), other));
    }
    let mut other = tiles.to_vec();
    let last = tiles.iter().filter_map(|t| match t {
        Tile::Table { table } => Some(*table),
        _ => None,
    });
    other.push(Tile::Table {
        table: last.max().expect("a table") + 1,
    });
    edits.push(("a table added at the end".to_owned(), other));
    edits
}

/// RFC-0041 'The plan', checking: a plan whose tiles are not those `D`'s claim list gives, by one
/// tile changed, removed, added or moved, is refused as `ERR-USAGE`: it is not `D`'s plan.
#[test]
fn a_plan_with_other_tiles_is_refused() {
    for (name, _, model) in models().into_iter().take(2) {
        let public = PublicModel::of(&model);
        let claims = claims(&model, 512);
        let plan = plan(public, &claims).expect("a plan");
        assert_eq!(check_plan(&plan, public, &claims), Ok(()), "{name}");
        for (what, tiles) in edits(&plan.tiles) {
            assert_ne!(tiles, plan.tiles, "{name}: {what}");
            let other = ExecutionPlan {
                claims: plan.claims.clone(),
                tiles,
            };
            let error = check_plan(&other, public, &claims).expect_err(&what);
            assert_eq!(
                error,
                CompileError::Usage { rule: OTHER_TILES },
                "{name}: {what}"
            );
            assert_eq!(error.code(), ErrorCode::Usage);
        }
    }
}

/// RFC-0041 'The plan', checking: a plan whose claim list is not the one `D` gives for the plan's
/// reader is refused as `ERR-USAGE`: another gadget policy, parameter set, layout, protocol version
/// or table, or another model's claim list. So is a plan checked against `D` under another gadget
/// policy.
#[test]
fn a_plan_with_another_claim_list_is_refused() {
    let models = models();
    let plans: Vec<(ClaimList, ExecutionPlan)> = models
        .iter()
        .map(|(_, _, model)| {
            let claims = claims(model, 512);
            let plan = plan(PublicModel::of(model), &claims).expect("a plan");
            (claims, plan)
        })
        .collect();
    let refusal = Err(CompileError::Usage { rule: OTHER_CLAIMS });
    for (k, (name, _, model)) in models.iter().enumerate() {
        let public = PublicModel::of(model);
        let (claims, plan) = &plans[k];
        let mut edited: Vec<(&str, ClaimList)> = Vec::new();
        let mut other = claims.clone();
        other.gadget_policy.query_block = 256;
        edited.push(("another query block", other));
        let mut other = claims.clone();
        other.params = ParamSetId(Ascii::new("test-small/2").expect("ASCII"));
        edited.push(("another parameter set", other));
        let mut other = claims.clone();
        other.layout.advice.log2_k += 1;
        edited.push(("another advice layout", other));
        let mut other = claims.clone();
        other.protocol_version += 1;
        edited.push(("another protocol version", other));
        let mut other = claims.clone();
        other.tables[3].digest[0] ^= 1;
        edited.push(("a table's digest", other));
        for (j, (other_claims, _)) in plans.iter().enumerate() {
            if j != k {
                edited.push(("another model's claim list", other_claims.clone()));
            }
        }
        for (what, other) in edited {
            assert_ne!(other, *claims, "{name}: {what}");
            let forged = ExecutionPlan {
                claims: other,
                tiles: plan.tiles.clone(),
            };
            assert_eq!(
                check_plan(&forged, public, claims),
                refusal,
                "{name}: {what}"
            );
        }
        // Another model's plan, whole.
        for (j, (_, other_plan)) in plans.iter().enumerate() {
            if j != k {
                assert_eq!(check_plan(other_plan, public, claims), refusal, "{name}");
            }
        }
        // D's plan for another gadget policy, against this one.
        let other = self::claims(model, 4);
        assert_eq!(check_plan(plan, public, &other), refusal, "{name}");
        let error = check_plan(plan, public, &other).expect_err("refused");
        assert_eq!(error.code(), ErrorCode::Usage);
    }
}

/// [`plan`] builds a plan from the claim list `D` gives: one whose protocol version, architecture,
/// quantization data or tables are not those of `D`'s lowering and gadget selection under its own
/// gadget policy is refused as `ERR-USAGE`. The gadget policy, the layout and the parameter set are
/// the claim list's own.
#[test]
fn a_plan_is_built_from_the_models_claim_list_only() {
    let models = models();
    let (name, _, model) = &models[0];
    let public = PublicModel::of(model);
    let claims = claims(model, 512);
    let refusal = Err(CompileError::Usage {
        rule: NOT_THE_MODELS,
    });
    let mut edited: Vec<(&str, ClaimList)> = Vec::new();
    let mut other = claims.clone();
    other.protocol_version += 1;
    edited.push(("another protocol version", other));
    let mut other = claims.clone();
    other.arch.max_context /= 2;
    edited.push(("another architecture", other));
    let mut other = claims.clone();
    other.quant.residual_bits -= 1;
    edited.push(("other quantization data", other));
    let mut other = claims.clone();
    other.tables[1].digest[31] ^= 0x80;
    edited.push(("a table's digest", other));
    let mut other = claims.clone();
    other.tables.pop();
    edited.push(("a table missing", other));
    edited.push(("another model's", self::claims(&models[1].2, 512)));
    for (what, other) in edited {
        assert_eq!(plan(public, &other), refusal, "{name}: {what}");
        if let Err(error) = plan(public, &other) {
            assert_eq!(error.code(), ErrorCode::Usage);
        }
    }
    // The claim list's own policy, layout and parameter set.
    let mut other = claims.clone();
    other.gadget_policy.query_block = 64;
    other.layout.advice.log2_k = 12;
    other.params = ParamSetId(Ascii::new("live-64/1").expect("ASCII"));
    let built = plan(public, &other).expect("a plan");
    assert_eq!(built.claims, other, "{name}");
    assert_eq!(built.tiles, plan(public, &claims).expect("a plan").tiles);
}

/// RFC-0041 'Production order': `Q` is a power of two, as RFC-0034's layout takes its logarithm. A
/// claim list with another is refused by the plan, and a plan that states another does not expand.
#[test]
fn a_query_block_that_is_not_a_power_of_two_is_refused() {
    let models = models();
    let (_, _, model) = &models[0];
    let public = PublicModel::of(model);
    let refusal = CompileError::Usage { rule: QUERY_BLOCK };
    let valid = claims(model, 1);
    for q in [0, 3, 6, 100, 511, 513, (1 << 31) + 1, u32::MAX] {
        let mut invalid = valid.clone();
        invalid.gadget_policy.query_block = q;
        assert_eq!(plan(public, &invalid), Err(refusal), "Q = {q}");
    }
    for q in [1, 2, 64, 1 << 31] {
        assert!(plan(public, &claims(model, q)).is_ok(), "Q = {q}");
    }
    let lowering = lower(public, 5).expect("the model lowers");
    let mut stated = plan(public, &claims(model, 512)).expect("a plan");
    stated.claims.gadget_policy.query_block = 3;
    assert_eq!(stated.expand(&lowering), Err(refusal));
}

/// [`ExecutionPlan::expand`] refuses a plan whose tiles do not fit the lowering, as `ERR-USAGE`: a
/// node past its graph, or an attention tile on a node that is no `SoftmaxRow` of `n_heads` heads
/// over the sequence, or on a KV head the architecture does not have.
#[test]
fn a_plan_that_does_not_fit_the_lowering_does_not_expand() {
    let models = models();
    let (name, _, model) = &models[0];
    let public = PublicModel::of(model);
    let lowering = lower(public, 7).expect("the model lowers");
    let base = plan(public, &claims(model, 4)).expect("a plan");
    assert!(base.expand(&lowering).is_ok(), "{name}");
    let nodes = u32::try_from(lowering.graph.nodes.len()).expect("few nodes");
    let attention = base
        .tiles
        .iter()
        .position(|t| matches!(t, Tile::Attention { .. }))
        .expect("an attention tile");
    let Tile::Attention { node, .. } = base.tiles[attention] else {
        unreachable!("an attention tile")
    };
    let with_tile = |at: usize, tile: Tile| {
        let mut plan = base.clone();
        plan.tiles[at] = tile;
        plan
    };
    let outside = CompileError::Usage { rule: OUTSIDE };
    let not_attention = CompileError::Usage {
        rule: NOT_ATTENTION,
    };
    for (what, plan, refusal) in [
        (
            "a node past the graph",
            with_tile(1, Tile::Node { node: nodes }),
            outside,
        ),
        (
            "an attention tile past the graph",
            with_tile(
                attention,
                Tile::Attention {
                    node: nodes,
                    kv_head: 0,
                },
            ),
            outside,
        ),
        (
            "an attention tile on the one-hot vectors' node",
            with_tile(
                attention,
                Tile::Attention {
                    node: 0,
                    kv_head: 0,
                },
            ),
            not_attention,
        ),
        (
            "a KV head past the architecture's",
            with_tile(
                attention,
                Tile::Attention {
                    node,
                    kv_head: model.arch.n_kv_heads,
                },
            ),
            not_attention,
        ),
    ] {
        assert_eq!(plan.expand(&lowering), Err(refusal), "{name}: {what}");
    }
    let (heads, kv) = (model.arch.n_heads, model.arch.n_kv_heads);
    for (what, n_heads, n_kv_heads) in [
        ("twice the query heads", 2 * heads, kv),
        ("no KV head", heads, 0),
        ("KV heads that do not divide the heads", heads, heads + 1),
    ] {
        let mut plan = base.clone();
        plan.claims.arch.n_heads = n_heads;
        plan.claims.arch.n_kv_heads = n_kv_heads;
        assert_eq!(plan.expand(&lowering), Err(not_attention), "{name}: {what}");
    }
}

/// RFC-0041 'Production order': a table has a tile when a lookup reads it. RoPE's constants are
/// `Rotate`'s, which reads no table, so RoPE's table has none, and neither has a table the
/// selection holds but no gadget reads.
#[test]
fn a_table_no_lookup_reads_has_no_tile() {
    for (name, _, model) in models() {
        let public = PublicModel::of(&model);
        let lowering = lower(public, 3).expect("the model lowers");
        let ranges = analyze(&lowering.graph, &lowering.tables).expect("ranges");
        let policy = GadgetPolicy::default();
        let mut selection =
            select(&lowering.graph, &ranges, &lowering.tables, &policy).expect("every gadget fits");
        let before = tiles(public, &lowering, &selection).expect("tiles");
        let with_tiles: Vec<u32> = before
            .iter()
            .filter_map(|t| match t {
                Tile::Table { table } => Some(*table),
                _ => None,
            })
            .collect();
        let rope: Vec<u32> = selection
            .tables
            .iter()
            .filter(|spec| matches!(spec.kind, TableKind::Rope { .. }))
            .map(|spec| spec.id.0)
            .collect();
        assert_eq!(rope, [0], "{name}: RoPE's table");
        assert!(!with_tiles.contains(&0), "{name}: RoPE's table has a tile");
        let mut unread = selection.tables.last().cloned().expect("a table");
        unread.id = TableId(u32::try_from(selection.tables.len()).expect("few tables"));
        selection.tables.push(unread);
        let after = tiles(public, &lowering, &selection).expect("tiles");
        assert_eq!(after, before, "{name}: a table no gadget reads");
    }
}

// ----- Encoding --------------------------------------------------------------------------------

/// RFC-0041 'The plan', RFC-0003 'Composite types': a tile is a `u16` tag, its variant's
/// declaration index, then its fields as `u32`s; a tag past the last variant is non-canonical.
#[test]
fn a_tile_encodes_its_declaration_index_then_its_fields() {
    for (tile, bytes) in [
        (Tile::Tokens, vec![0, 0]),
        (Tile::Node { node: 0x0102_0304 }, vec![1, 0, 4, 3, 2, 1]),
        (
            Tile::Attention {
                node: 7,
                kv_head: u32::MAX,
            },
            vec![2, 0, 7, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF],
        ),
        (Tile::Table { table: 0 }, vec![3, 0, 0, 0, 0, 0]),
    ] {
        assert_eq!(encode(&tile), bytes, "{tile:?}");
        assert_eq!(decode::<Tile>(&bytes, DecodeLimits::MESSAGE), Ok(tile));
    }
    let error = decode::<Tile>(&[4, 0, 0, 0, 0, 0], DecodeLimits::MESSAGE).expect_err("tag 4");
    assert_eq!(
        (error.code, error.offset),
        (ErrorCode::CodecNoncanonical, 0)
    );
}

/// A plan is its claim list then its tiles (RFC-0041 'The plan', in declaration order), and decodes
/// from its encoding and from its envelope to itself.
#[test]
fn a_plan_decodes_to_itself() {
    for (name, _, model) in models() {
        let plan = plan(PublicModel::of(&model), &claims(&model, 4)).expect("a plan");
        let bytes = encode(&plan);
        assert_eq!(
            bytes,
            [encode(&plan.claims), encode(&plan.tiles)].concat(),
            "{name}"
        );
        let decoded: ExecutionPlan = decode(&bytes, DecodeLimits::MESSAGE).expect("a plan");
        assert_eq!(decoded, plan, "{name}");
        let sealed = envelope::seal(&schema::EXECUTION_PLAN, &plan);
        let opened: ExecutionPlan =
            envelope::open(&sealed, &schema::EXECUTION_PLAN, DecodeLimits::MESSAGE)
                .expect("its envelope");
        assert_eq!(opened, plan, "{name}");
    }
}

/// A tile as the codec vectors write it (`spec/vectors/codec/README.md`, 'Values').
fn shown(value: &Value) -> Tile {
    let object = value.as_object().expect("an enum value");
    assert_eq!(object.len(), 1, "one variant");
    let (variant, fields) = object.iter().next().expect("a variant");
    let field = |name: &str| -> u32 {
        fields[name]
            .as_str()
            .and_then(|n| n.parse().ok())
            .unwrap_or_else(|| panic!("{variant}: {name} in decimal"))
    };
    match variant.as_str() {
        "Tokens" => {
            assert!(fields.is_null(), "a unit variant");
            Tile::Tokens
        }
        "Node" => Tile::Node {
            node: field("node"),
        },
        "Attention" => Tile::Attention {
            node: field("node"),
            kv_head: field("kv_head"),
        },
        "Table" => Tile::Table {
            table: field("table"),
        },
        other => panic!("no tile {other}"),
    }
}

/// The golden plans of `spec/vectors/codec/execution-plan.json`, one per conformance model: each
/// opens, under the message limits, to the plan the compiler derives for its model for
/// `test-small/1`, whose claim list is the golden one of `spec/vectors/claims/` and has the stated
/// digest, and whose tiles are the vector's values. Every proper prefix of its payload is cut
/// short.
#[test]
fn the_golden_plans_decode_to_their_cases() {
    let file = json(vectors().join("codec/execution-plan.json"));
    assert_eq!(file["schema"].as_u64(), Some(0x0202));
    assert_eq!(file["version"].as_u64(), Some(1));
    assert_eq!(file["params"].as_str(), Some("test-small/1"));
    let golden_claims = json(vectors().join("claims/claim-lists.json"));
    let cases = file["cases"].as_array().expect("cases");
    assert_eq!(cases.len(), 4, "every conformance model");
    for case in cases {
        let name = case["name"].as_str().expect("a name");
        let bytes = hex(case["hex"].as_str().expect("hex"));
        let decoded: ExecutionPlan =
            envelope::open(&bytes, &schema::EXECUTION_PLAN, DecodeLimits::MESSAGE)
                .expect("an ExecutionPlan");
        let model = model_file(case["model"].as_str().expect("a model"));
        let derived = plan(PublicModel::of(&model), &claims(&model, 512)).expect("a plan");
        assert_eq!(decoded, derived, "{name}: the compiler's plan");

        let digest: String = decoded
            .claims
            .digest()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(
            case["claim_list_digest"].as_str(),
            Some(digest.as_str()),
            "{name}"
        );
        let shown: Vec<Tile> = case["tiles"]
            .as_array()
            .expect("tiles")
            .iter()
            .map(shown)
            .collect();
        assert_eq!(decoded.tiles, shown, "{name}: the tiles");
        let sealed = golden_claims["cases"]
            .as_array()
            .expect("claim lists")
            .iter()
            .find(|c| c["name"].as_str() == Some(name))
            .and_then(|c| c["hex"].as_str())
            .map(hex)
            .expect("the model's golden claim list");
        let claims: ClaimList = envelope::open(&sealed, &schema::CLAIM_LIST, DecodeLimits::MESSAGE)
            .expect("a claim list");
        assert_eq!(decoded.claims, claims, "{name}: the golden claim list");

        let payload = &bytes[envelope::HEADER_LEN..bytes.len() - envelope::DIGEST_LEN];
        for cut in 0..payload.len() {
            let error = decode::<ExecutionPlan>(&payload[..cut], DecodeLimits::MESSAGE)
                .expect_err("a proper prefix decoded");
            assert_eq!(
                error.code,
                ErrorCode::CodecTruncated,
                "{name}, first {cut} bytes"
            );
        }
    }
}

#[test]
fn argmax_adds_its_range_multiplicities_to_the_plan() {
    let (_, _, model) = models().into_iter().next().unwrap();
    let public = PublicModel::of(&model);
    let mut lowering = lower(public, 3).unwrap();
    let indicator = lowering
        .graph
        .nodes
        .iter()
        .find_map(|node| matches!(node.op, Op::OneHot { .. }).then(|| node.outputs[0]))
        .unwrap();
    let id = NodeId(lowering.graph.nodes.len() as u32);
    lowering.graph.nodes.push(Node {
        id,
        op: Op::Argmax,
        inputs: vec![lowering.logits, indicator],
        outputs: vec![],
    });
    lowering.graph.validate().unwrap();
    let ranges = analyze(&lowering.graph, &lowering.tables).unwrap();
    let selection = select(
        &lowering.graph,
        &ranges,
        &lowering.tables,
        &GadgetPolicy::default(),
    )
    .unwrap();
    let (_, Gadget::Argmax { table }) = selection.of(id).unwrap() else {
        panic!("Argmax");
    };
    let plan = tiles(public, &lowering, &selection).unwrap();
    assert_eq!(
        plan.iter()
            .filter(|t| **t == Tile::Table { table: table.0 })
            .count(),
        1
    );
    assert!(plan.contains(&Tile::Node { node: id.0 }));
}
