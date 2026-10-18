// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tiles and the execution plan (RFC-0041, RFC-0006 'Tiling and the execution plan').
//!
//! - **Tiles.** A sequence's committed values split into tiles, each value in exactly one
//!   (RFC-0041 'Tiles'):
//!   - [`Tile::Tokens`], the tokens;
//!   - [`Tile::Node`] for each node that commits a value, other than a layer's attention node: the
//!     prologue's and the epilogue's nodes, and a layer's linear tiles `(layer, operation)`;
//!   - [`Tile::Attention`] for each KV head of a layer's attention node, its `SoftmaxRow`. A
//!     sequence of `n` tokens expands it into its query blocks of `Q' = min(Q, N)` positions, with
//!     `N = 2^{⌈log2 n⌉}` and `Q = GadgetPolicy.query_block`: RFC-0006's `(layer, kv_head,
//!     query_block)` tiles;
//!   - [`Tile::Table`] for each table some lookup reads: its multiplicities.
//! - **Production order.** The tokens, then the nodes' tiles in node order, an attention node's by
//!   KV head, then query block, then the tables' by id (RFC-0041 'Production order'). The advice
//!   is allocated tile by tile in this order (RFC-0011 rule 2), and the verifier derives it to
//!   locate every committed tensor, so any other order rejects every honest proof.
//! - **The plan.** An [`ExecutionPlan`], schema `0x0202`, holds the claim list it plans and one
//!   sequence's tiles. A lowering's node and tensor ids do not depend on the sequence's length,
//!   only its shapes do, so the tiles serve every length: [`ExecutionPlan::expand`] gives those of
//!   `n` tokens with the elements each holds. [`plan`] builds the plan from the public part of `D`
//!   and its claim list alone, so it holds no secret value, and [`check_plan`] refuses a plan that
//!   is not `D`'s (`ERR-USAGE`). Row ranges, stage membership and devices are not in it: they are
//!   the advice layout's (RFC-0011), the stage schedule's (RFC-0037) and the prover's (RFC-0018).

use core::ops::Range;

use tessera_codec::{Decode, Encode};
use tessera_ir::{
    ClaimList, CommitmentClass, Graph, Node, NodeKind, Op, OpClass, TableId, TensorId,
};

use crate::schedule::LayerMap;
use crate::{
    CompileError, Gadget, Lowering, PublicModel, Selection, analyze, claim_list, lower, select,
};

/// A `query_block` that is not a power of two: RFC-0007 step 6 refuses its claim list as `Layout`
/// (RFC-0039).
const QUERY_BLOCK: &str = "query_block is a power of two: RFC-0034's layout takes its logarithm";

/// A claim list that `claim_list` does not emit for the model.
const NOT_THE_MODELS: &str =
    "the claim list is the model's, for its own gadget policy, layout and parameter set";

/// The two ways a plan is not `D`'s (RFC-0041 'The plan').
const OTHER_CLAIMS: &str = "an execution plan holds D's claim list";
const OTHER_TILES: &str = "an execution plan holds the tiles D's claim list gives";

/// Tiles that do not fit the lowering they are expanded on.
const OUTSIDE: &str = "the plan's tiles name nodes of the lowering's graph";
const NOT_ATTENTION: &str = "an attention tile names a KV head of a SoftmaxRow over n_heads heads";

/// A tile of an [`ExecutionPlan`] (RFC-0041 'The plan'), the same for every sequence length. Its
/// TCE tag is its declaration index (RFC-0003 'Composite types').
///
/// A tile holds elements of graph tensors, which [`ExecutionPlan::expand`] lists, and the values
/// its node's gadget commits beyond the node's outputs (RFC-0041 'What is committed'). Those are no
/// graph tensor: the tile owns them by definition. Within a tile the node's outputs come first, in
/// output order, then the gadget's values, in the order RFC-0009 lists them:
///
/// - `Requant1`, a [`Gadget::Lookup`] (`TableMap`, `Bitwise`) and `OneHot` commit their outputs
///   alone;
/// - `RequantSplit` adds the low parts `z_lo`;
/// - `RequantWide` adds `z_lo` when it has a low part, then the middle parts `h0` and the
///   saturations `g`;
/// - `RsqrtRow` adds the limbs of each row's `u`, then the `BigIntCompare` witnesses of its two
///   comparisons, which the gadgets fix (#118);
/// - `SoftmaxRow` adds, for each row an attention tile holds, the maximum `c_q`, then the limbs of
///   the row's two `NonNeg` instances;
/// - `Argmax` and `InverseCdf` have no output: their tiles hold their certificates' values alone
///   (RFC-0017).
///
/// The masking coefficients of `tessera-protocol/2` are in no tile (RFC-0013).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub enum Tile {
    /// The tokens, a graph input committed as advice. Tag 0.
    Tokens,
    /// A node that commits a value, other than a layer's attention node: its committed outputs,
    /// then its gadget's values. Tag 1.
    Node {
        /// The node's index in the lowering's graph.
        node: u32,
    },
    /// Every query block of one KV head of a layer's attention node, its `SoftmaxRow`: the
    /// exponentials `e[h, q, i]` and reciprocals `R[h, q]` of the KV head's query heads, then
    /// their maxima `c[h, q]` and bracket limbs. Tag 2.
    Attention {
        /// The `SoftmaxRow`'s index in the lowering's graph.
        node: u32,
        /// The KV head `g`, from 0 to `n_kv_heads − 1`.
        kv_head: u32,
    },
    /// The multiplicities of a table some lookup reads, one per row, shared by every lookup into
    /// it (RFC-0008). Tag 3.
    Table {
        /// The table's [`TableId`].
        table: u32,
    },
}

/// The execution plan, schema `0x0202`, version 1, private, `.tep` (RFC-0041 'The plan', 03 §5):
/// the claim list it plans, and one sequence's tiles in production order.
///
/// [`plan`] builds it from the public part of `D` and its claim list: two builds give the same
/// bytes, and it holds no secret value. Its readers take `D` too and refuse a plan that is not
/// `D`'s ([`check_plan`]).
#[derive(Clone, Debug, PartialEq, Eq, Encode, Decode)]
pub struct ExecutionPlan {
    /// The claim list it plans (RFC-0007, RFC-0026). The commands that read a plan take no claim
    /// list (02 §2.2).
    pub claims: ClaimList,
    /// One sequence's tiles, in production order: the tiles of every length.
    pub tiles: Vec<Tile>,
}

/// A tile of a sequence of `n` tokens (RFC-0041 'Tiles'): a plan's tile, an attention tile taken at
/// one of its query blocks, with the elements it holds of the graph's tensors.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SequenceTile {
    /// The plan's tile.
    pub tile: Tile,
    /// An attention tile's query block; `None` for any other tile.
    pub block: Option<QueryBlock>,
    /// The elements it holds of each graph tensor committed in the advice, in the tile's order:
    /// the tokens, or the node's committed outputs in output order. A table's tile holds none: its
    /// multiplicities are no graph tensor.
    pub shares: Vec<Share>,
}

/// A query block of an attention tile (RFC-0041 'Tiles'): the tile of KV head `g` and block `b`
/// holds `g`'s query heads at the queries `b·Q'` to `(b + 1)·Q' − 1` of the sequence padded to `N`
/// positions, every key with them.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct QueryBlock {
    /// `b`, from 0 to `N / Q' − 1`.
    pub index: u32,
    /// Its queries, as positions of the padded sequence: those from `n` on are padding, which
    /// holds what the gadgets compute on zero inputs (RFC-0035).
    pub queries: Range<u32>,
}

/// The elements of one graph tensor that a tile holds: every element whose coordinate on each axis
/// lies in that axis's range. An attention tile's share of a tensor is allocated as a tensor of its
/// own (RFC-0011 rule 2).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Share {
    /// The tensor.
    pub tensor: TensorId,
    /// One range per axis, outermost first, inside the tensor's shape. A query block that holds
    /// padding alone has an empty range of queries.
    pub ranges: Vec<Range<u32>>,
}

impl ExecutionPlan {
    /// The tiles of the sequence `lowering` covers, in production order, with the elements each
    /// holds (RFC-0041 'Tiles'): each attention tile becomes its `N / Q'` query blocks, in order.
    /// `lowering` is `D`'s lowering of any length.
    ///
    /// The tile of KV head `g` and block `b` holds `e[h, q, i]` and `R[h, q]` for the query heads
    /// `h` with `⌊h / (n_heads / n_kv_heads)⌋ = g`, the queries `q` of its block that lie in the
    /// sequence, and every key `i`.
    ///
    /// # Errors
    ///
    /// [`CompileError::Usage`] for a plan whose `query_block` is not a power of two, or whose tiles
    /// do not fit `lowering`: a node past its graph, or an attention tile on a node that is no
    /// `SoftmaxRow` of `n_heads` heads over the sequence, or on a KV head past `n_kv_heads`.
    pub fn expand(&self, lowering: &Lowering) -> Result<Vec<SequenceTile>, CompileError> {
        let graph = &lowering.graph;
        let outside = CompileError::Usage { rule: OUTSIDE };
        let n = graph
            .tensors
            .get(lowering.tokens.0 as usize)
            .and_then(|tokens| tokens.shape.first().copied())
            .ok_or(outside)?;
        let padded = n.next_power_of_two();
        let block = query_block(self.claims.gadget_policy.query_block, padded)?;
        let node_of = |node: u32| graph.nodes.get(node as usize).ok_or(outside);
        let mut expanded = Vec::with_capacity(self.tiles.len());
        for &tile in &self.tiles {
            match tile {
                Tile::Tokens => expanded.push(SequenceTile {
                    tile,
                    block: None,
                    shares: whole(graph, lowering.tokens).into_iter().collect(),
                }),
                Tile::Node { node } => expanded.push(SequenceTile {
                    tile,
                    block: None,
                    shares: node_of(node)?
                        .outputs
                        .iter()
                        .filter_map(|&t| whole(graph, t))
                        .collect(),
                }),
                Tile::Attention { node, kv_head } => {
                    let node = node_of(node)?;
                    let heads = self.query_heads(graph, node, kv_head, n)?;
                    for index in 0..padded / block {
                        let queries = index * block..(index + 1) * block;
                        // The block's queries that lie in the sequence: none past its end.
                        let held = queries.start.min(n)..queries.end.min(n);
                        let shares = node
                            .outputs
                            .iter()
                            .filter_map(|&t| rows(graph, t, &heads, &held))
                            .collect();
                        expanded.push(SequenceTile {
                            tile,
                            block: Some(QueryBlock { index, queries }),
                            shares,
                        });
                    }
                }
                Tile::Table { .. } => expanded.push(SequenceTile {
                    tile,
                    block: None,
                    shares: Vec::new(),
                }),
            }
        }
        Ok(expanded)
    }

    /// The query heads of KV head `kv_head` at `node`, a `SoftmaxRow` whose scores are
    /// `[n_heads, n, n]`: those `h` with `⌊h / (n_heads / n_kv_heads)⌋ = kv_head`, the heads that
    /// read its keys (RFC-0006 'Llama lowering').
    fn query_heads(
        &self,
        graph: &Graph,
        node: &Node,
        kv_head: u32,
        n: u32,
    ) -> Result<Range<u32>, CompileError> {
        let arch = &self.claims.arch;
        let scores = node
            .inputs
            .first()
            .and_then(|&t| graph.tensors.get(t.0 as usize));
        let fits = matches!(node.op, Op::SoftmaxRow { .. })
            && scores.is_some_and(|s| s.shape == [arch.n_heads, n, n])
            && arch.n_kv_heads > 0
            && arch.n_heads.is_multiple_of(arch.n_kv_heads)
            && kv_head < arch.n_kv_heads;
        if !fits {
            return Err(CompileError::Usage {
                rule: NOT_ATTENTION,
            });
        }
        let group = arch.n_heads / arch.n_kv_heads;
        Ok(kv_head * group..(kv_head + 1) * group)
    }
}

/// The tiles of `lowering`, whose gadgets `selection` chose, in production order (RFC-0041
/// 'Production order'): the tokens; then, in node order, each node that commits a value, a layer's
/// attention node once per KV head; then each table some gadget reads, by id. Any lowering of
/// `model` gives the same tiles, whatever its length.
///
/// # Errors
///
/// [`CompileError::Schedule`] (`ERR-INTERNAL`) for a graph whose layers are not one template
/// ([`LayerMap`]): RFC-0037 then defines no layer, and no layer's attention node.
pub fn tiles(
    model: PublicModel<'_>,
    lowering: &Lowering,
    selection: &Selection,
) -> Result<Vec<Tile>, CompileError> {
    let layers = LayerMap::new(lowering)?;
    let graph = &lowering.graph;
    let mut tiles = vec![Tile::Tokens];
    for node in &graph.nodes {
        // A layer's attention node is its `SoftmaxRow` (RFC-0041 'Tiles').
        let attention =
            matches!(node.op, Op::SoftmaxRow { .. }) && layers.position(node.id).is_some();
        if attention {
            tiles.extend((0..model.arch.n_kv_heads).map(|kv_head| Tile::Attention {
                node: node.id.0,
                kv_head,
            }));
        } else if commits(graph, node) {
            tiles.push(Tile::Node { node: node.id.0 });
        }
    }
    let mut read: Vec<TableId> = selection
        .gadgets
        .iter()
        .flat_map(|&(_, gadget)| reads(gadget))
        .collect();
    read.sort_unstable();
    read.dedup();
    tiles.extend(read.into_iter().map(|table| Tile::Table { table: table.0 }));
    Ok(tiles)
}

/// The execution plan of `claims`, `model`'s claim list (RFC-0041 'The plan'): the claim list, and
/// the tiles of `model`'s lowering with the gadgets its gadget policy selects. It reads the public
/// part of `D` alone, so it holds no secret value, and two builds give the same bytes.
///
/// # Errors
///
/// - [`CompileError::Usage`] for a claim list whose `query_block` is not a power of two, or that is
///   not the one [`claim_list`] emits for `model` under the claim list's own gadget policy, layout
///   and parameter set;
/// - the errors of [`lower`] and [`select`], for a model or a gadget policy they refuse;
/// - [`CompileError::Schedule`] for a lowering whose layers are not one template.
pub fn plan(model: PublicModel<'_>, claims: &ClaimList) -> Result<ExecutionPlan, CompileError> {
    let policy = &claims.gadget_policy;
    // Every length's expansion takes `Q`'s logarithm.
    query_block(policy.query_block, 1)?;
    // Ids do not depend on the length: one position lowers the graph of every length.
    let lowering = lower(model, 1)?;
    let ranges = analyze(&lowering.graph, &lowering.tables)?;
    let selection = select(&lowering.graph, &ranges, &lowering.tables, policy)?;
    if claim_list(
        model,
        &selection,
        policy,
        claims.layout,
        claims.params.clone(),
    ) != *claims
    {
        return Err(CompileError::Usage {
            rule: NOT_THE_MODELS,
        });
    }
    Ok(ExecutionPlan {
        claims: claims.clone(),
        tiles: tiles(model, &lowering, &selection)?,
    })
}

/// Checks that `plan` is `D`'s (RFC-0041 'The plan'): its claim list is `claims`, the claim list of
/// `D`, whose public part is `model`, for the plan's reader, and its tiles are those `claims`
/// gives. The readers of a plan, `tessera prove` and `tessera commit-weights`, take `D` too.
///
/// # Errors
///
/// [`CompileError::Usage`] for a plan that is not `D`'s: its claim list or its tiles differ. The
/// errors of [`plan`] for a claim list that is not `model`'s.
pub fn check_plan(
    plan: &ExecutionPlan,
    model: PublicModel<'_>,
    claims: &ClaimList,
) -> Result<(), CompileError> {
    if plan.claims != *claims {
        return Err(CompileError::Usage { rule: OTHER_CLAIMS });
    }
    if plan.tiles != self::plan(model, claims)?.tiles {
        return Err(CompileError::Usage { rule: OTHER_TILES });
    }
    Ok(())
}

/// `Q' = min(Q, N)` for a sequence padded to `padded` positions, after checking that `Q` is a power
/// of two (RFC-0041 'Production order').
fn query_block(q: u32, padded: u32) -> Result<u32, CompileError> {
    if q.is_power_of_two() {
        Ok(q.min(padded))
    } else {
        Err(CompileError::Usage { rule: QUERY_BLOCK })
    }
}

/// Whether a tensor of `kind` is committed in the advice: advice or a lookup output. Weights are
/// the weight commitment's (RFC-0040).
fn committed(kind: NodeKind) -> bool {
    matches!(
        kind,
        NodeKind::Committed {
            commitment: CommitmentClass::Advice
        } | NodeKind::LookupOutput { .. }
    )
}

/// Whether `node` commits a value: a gadget, all of which commit values of their own (RFC-0009), or
/// a node with an output committed in the advice, such as the first checkpoint (RFC-0037).
fn commits(graph: &Graph, node: &Node) -> bool {
    node.op.class() == OpClass::Gadget
        || node.outputs.iter().any(|t| {
            graph
                .tensors
                .get(t.0 as usize)
                .is_some_and(|spec| committed(spec.kind))
        })
}

/// All of `tensor`, when it is committed in the advice.
fn whole(graph: &Graph, tensor: TensorId) -> Option<Share> {
    let spec = graph.tensors.get(tensor.0 as usize)?;
    committed(spec.kind).then(|| Share {
        tensor,
        ranges: spec.shape.iter().map(|&len| 0..len).collect(),
    })
}

/// The rows of `tensor`, an output of an attention node, at the query heads `heads` and the
/// queries `queries`, with every key: `e[h, q, i]` or `R[h, q]`, whose axes are the heads, then the
/// queries, then the keys.
fn rows(
    graph: &Graph,
    tensor: TensorId,
    heads: &Range<u32>,
    queries: &Range<u32>,
) -> Option<Share> {
    let mut share = whole(graph, tensor)?;
    if let [h, q, ..] = share.ranges.as_mut_slice() {
        *h = heads.clone();
        *q = queries.clone();
    }
    Some(share)
}

/// The tables `gadget`'s lookups read. The range checks of `SoftmaxRow` and `RsqrtRow` read tables
/// the selection does not allocate yet (#118, #122).
fn reads(gadget: Gadget) -> Vec<TableId> {
    match gadget {
        Gadget::Requant1 { table }
        | Gadget::Lookup { table }
        | Gadget::SoftmaxRow { exp: table } => {
            vec![table]
        }
        Gadget::RequantSplit { low, high, .. } => vec![low, high],
        Gadget::RequantWide {
            low,
            mid,
            top,
            sign,
            high,
            ..
        } => low.into_iter().chain([mid, top, sign, high]).collect(),
        Gadget::RsqrtRow | Gadget::OneHot | Gadget::Argmax | Gadget::InverseCdf => Vec::new(),
    }
}
