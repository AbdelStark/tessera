// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The commitment report (RFC-0006 'Outputs'): what a lowered model commits and looks up, the
//! exact counts the cost model reads in place of the source's projected ones (RFC-0020, RISK-2).
//!
//! The report counts each gadget of the [`Selection`] by its definition in RFC-0009 and RFC-0024:
//!
//! - **`Requant1`**: one committed lookup output and one lookup per element;
//! - **`RequantSplit`**: the output and its low part, two lookups;
//! - **`RequantWide`**: the output, its low part when it has one, its middle part and its
//!   saturation; one lookup more than it commits;
//! - **`TableMap`, `Bitwise`**: the output and one lookup;
//! - **`SoftmaxRow`**: per valid score, the exponential and its lookup, and a leaf of its row's
//!   product tree; per row, `c_q`, `R_q`, `R_q`'s range check, and the two `NonNeg` of its
//!   reciprocal bracket, each of `⌈(σ + 18)/L⌉` limbs with one fewer committed;
//! - **`RsqrtRow`**: RFC-0050's exact width-derived columns and every range request,
//!   including both virtual tops and all comparison arrays;
//! - **`OneHot`**: `vocab` committed entries per token.
//!
//! Counts are per token: the lowering covers one sequence, and every count above is a multiple of
//! its length. A layer's count comes from the first layer's nodes, between its checkpoints, since
//! every layer lowers alike. Scores grow with the context, so they are counted apart, per score.
//!
//! The tables and their multiplicities follow the complete selected request lists. A table
//! without a lookup has no multiplicities; RoPE is a public kernel, not a lookup.
//!
//! The stage count is RFC-0010's: the leaf stage, then one stage per sumcheck relation on the
//! longest chain from a gadget's input down to committed tensors. `MatMulWeight`, `MatMulAct`,
//! `Mul`, `Rotate`, `SumSquares` and a `ConstMul` by per-channel constants are proved by
//! sumchecks; adds, a `Sum` over an axis, a `ConstMul` by one constant, `ConstAdd`, `Reshape`,
//! `Slice` and `GatherPublic` map a claim onto their inputs at no stage. The opening groups `G`
//! and the masks of RFC-0011 and RFC-0013 follow from the stage schedule's claim points (#113,
//! #149, #154).

use serde_json::{Map, Value, json};
use tessera_ir::{
    CommitmentClass, GadgetPolicy, Graph, NodeId, NodeKind, Op, OpClass, Profile, QuantPoint, Site,
    TensorId,
};
use tessera_tables::{TableId, TableKind};

use crate::schedule::{Coord, Kind, Layers, NativeKind, NodeRef, Relation, Request};
use crate::{CompileError, Gadget, Lowering, PublicModel, Selection};

/// The report's schema identifier (03 §5).
pub const SCHEMA: &str = "tessera.commitment-report/1";

/// Committed elements by class and lookups, over some part of the model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tally {
    /// Gadget outputs that are lookup outputs: requantized and table-mapped values.
    pub lookup_outputs: u64,
    /// The extra parts of split requantizations: `RequantSplit`'s low part, `RequantWide`'s low
    /// part, middle part and saturation.
    pub split_limbs: u64,
    /// Per-row values of row gadgets: `SoftmaxRow`'s maximum, reciprocal and bracket limbs,
    /// `RsqrtRow`'s root and limbs.
    pub row_scalars: u64,
    /// One-hot entries.
    pub one_hot: u64,
    /// Linear outputs committed as advice, which stage 1 proves equal to their relation: the
    /// first checkpoint, the embedding scaled into the residual (RFC-0037).
    pub checkpoints: u64,
    /// Lookups.
    pub lookups: u64,
}

impl Tally {
    /// Every committed element.
    pub fn committed(&self) -> u64 {
        self.lookup_outputs + self.split_limbs + self.row_scalars + self.one_hot + self.checkpoints
    }

    fn per_token(self, n: u64) -> Result<Self, CompileError> {
        Ok(Self {
            lookup_outputs: per_token(self.lookup_outputs, n)?,
            split_limbs: per_token(self.split_limbs, n)?,
            row_scalars: per_token(self.row_scalars, n)?,
            one_hot: per_token(self.one_hot, n)?,
            checkpoints: per_token(self.checkpoints, n)?,
            lookups: per_token(self.lookups, n)?,
        })
    }
}

/// One table: its rows, and the lookups into it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableUse {
    /// The table.
    pub id: TableId,
    /// Its kind, as RFC-0008 names it.
    pub kind: &'static str,
    /// Its actual table rows: one multiplicity each, excluding power-of-two padding.
    pub rows: u64,
    /// Lookups into it per token, outside the scores, including every auxiliary range.
    pub lookups_per_token: u64,
    /// Lookups into it by the scores of the lowering's sequence: one per score of each layer
    /// whose exponentials it holds.
    pub score_lookups: u64,
}

/// How a stage's prover holds a table (RFC-0055 'The window', RFC-0056 'The prover').
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Holding {
    /// An elementwise relation's operand, which a window may fold: a regeneration yields it
    /// block by block.
    Windowable,
    /// A table held whole: a matrix product's folded operand, or an operand RFC-0056's prover
    /// keeps for `SoftmaxRow`'s key and forest instances.
    Resident,
    /// A table the prover never holds: `SoftmaxRow`'s key and forest instances over the score
    /// tensor, whose rounds RFC-0056 computes from the queries and the folded keys.
    Computed,
}

/// How a table's entries grow with a batch of `T` tokens at context `S`, powers of two
/// (RFC-0055 'Working sets', RFC-0056).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Growth {
    /// Not at all: a weight's folded table.
    Fixed,
    /// With the tokens, `T`: an activation's operand.
    Tokens,
    /// With the key positions the batch attends, `max(T, S)`: RFC-0056's resident keys.
    Keys,
    /// With the tokens and the context, `T·S`: the score tensor.
    TokensAndContext,
}

/// Tables of one size in a reduction stage: `count + count_per_bit·b` tables of
/// `2^{log2_entries}` entries of `bytes` bytes, times the batch's growth (RFC-0055 'Working
/// sets', RFC-0056). `b` is the bits of a request's positions, `log2 min(T, S)`: a request's
/// greedy relations hold one image of its one-hot vectors per bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TableClass {
    /// How the entries grow.
    pub growth: Growth,
    /// How the prover holds the tables.
    pub kind: Holding,
    /// `log2` of one table's entries, per unit of its growth.
    pub log2_entries: u32,
    /// The tables of this size, before the per-bit ones.
    pub count: u64,
    /// The tables added per bit of a request's positions.
    pub count_per_bit: u64,
    /// The bytes of one entry: 16, an `Fp2`, or 1 for an INT8 operand held as it is.
    pub bytes: u8,
}

/// A reduction stage's sumcheck tables (RFC-0055 'Working sets'): its instances' operands, the
/// native relations' included (RFC-0056).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageTables {
    /// The stage, from 1.
    pub stage: u16,
    /// Its instances' tables by class: the classes that grow fastest first, then by kind, size
    /// and width.
    pub tables: Vec<TableClass>,
    /// What RFC-0056's prover holds besides for the stage's computed instances, by class.
    pub residents: Vec<TableClass>,
}

/// What a lowered model commits and looks up (RFC-0006 'Outputs').
#[derive(Clone, Debug, PartialEq)]
pub struct CommitmentReport {
    /// The profile.
    pub profile: Profile,
    /// `n_layers`, `d_model`, `n_heads`, `n_kv_heads`, `d_head`, `d_ff` and `vocab`.
    pub shape: [u32; 7],
    /// The weights' entries: `F_lin` counts two operations each.
    pub parameters: u64,
    /// The lowering's sequence length.
    pub positions: u32,
    /// `B_max`, the gadget policy's table bound.
    pub max_log2_rows: u8,
    /// RFC-0010's stage count.
    pub stages: u32,
    /// Each site's requantized elements per layer per token, in the lowering's order.
    pub requantized: Vec<(Site, u64)>,
    /// One layer, per token.
    pub per_layer: Tally,
    /// The whole model per token, outside the scores.
    pub per_token: Tally,
    /// Valid attention scores in the lowering's sequence, all layers and heads.
    pub scores: u64,
    /// Product-tree leaves in the lowering's sequence: each row's tree has `2^h` leaves.
    pub product_tree_leaves: u64,
    /// The tables.
    pub tables: Vec<TableUse>,
    /// Each reduction stage's sumcheck tables, in stage order.
    pub stage_tables: Vec<StageTables>,
}

impl CommitmentReport {
    /// Requantized elements per layer per token, every site's.
    pub fn requantized_per_layer(&self) -> u64 {
        self.requantized.iter().map(|(_, n)| n).sum()
    }

    /// `A`: scores per token per unit of context, `n_heads · n_layers / 2` (08 §1).
    pub fn scores_per_token_per_ctx(&self) -> f64 {
        f64::from(self.shape[2]) * f64::from(self.shape[0]) / 2.0
    }

    /// `b`, the score block of RFC-0034: at context `S`, a token commits `A·(S + min(S, b))`
    /// scores. An attention tile holds the `g` query heads of a KV head and `Q = 512` queries
    /// (RFC-0006), and an advice row of the production parameter sets holds `2^22` entries
    /// (RFC-0014). A row therefore holds `2^w` keys, `w = 22 − log2 Q − log2 g`, and
    /// `b = max(Q, 2^w)`: 512 for Llama-3.1-405B's `g = 16`, 2,048 for 8B's `g = 4`.
    pub fn score_block(&self) -> u64 {
        const LOG2_K: u32 = 22;
        const LOG2_Q: u32 = 9;
        let group = (self.shape[2] / self.shape[3].max(1)).max(1);
        let w = LOG2_K.saturating_sub(LOG2_Q + group.next_power_of_two().ilog2());
        (1u64 << LOG2_Q).max(1 << w)
    }

    /// Multiplicities per proof, one per real row of every table a lookup reads (RFC-0008/0041).
    pub fn multiplicities(&self) -> u64 {
        self.tables
            .iter()
            .filter(|t| t.lookups_per_token > 0 || t.score_lookups > 0)
            .map(|t| t.rows)
            .sum()
    }

    /// The counts of RFC-0020's cost model, under the names of `tessera.cost-model/1`. `F_att`
    /// counts `4 · d_head` operations per score, the products of `QK^T` and `PV`, the table rows
    /// are the [`multiplicities`](Self::multiplicities), and the score block is
    /// [`score_block`](Self::score_block).
    #[allow(clippy::cast_precision_loss)]
    pub fn counts(&self) -> Value {
        let a = self.scores_per_token_per_ctx();
        json!({
            "commits_per_token": self.per_token.committed() as f64,
            "lookups_per_token": self.per_token.lookups as f64,
            "scores_per_token_per_ctx": a,
            "native_ops_lin": 2.0 * self.parameters as f64,
            "native_ops_att_per_ctx": 4.0 * f64::from(self.shape[4]) * a,
            "table_rows": self.multiplicities() as f64,
            "score_block": self.score_block() as f64,
            "reduction_stages": self.stage_tables.iter().map(stage_json).collect::<Vec<_>>(),
        })
    }

    /// The report as `tessera.commitment-report/1`, with the projections of `μ` of `cost_model`:
    /// the `tessera.cost-model/1` report that `tessera-bench` gives for
    /// [`counts`](Self::counts). It keeps the projections and what they assume: the unit costs,
    /// the memory, the scenario and the regenerations.
    pub fn to_json(&self, cost_model: &Value) -> Value {
        let inputs = &cost_model["inputs"];
        let projection_inputs = json!({
            "unit_costs": inputs["unit_costs"],
            "memory": inputs["memory"],
            "scenario": inputs["scenario"],
            "regen_full_equivalents": inputs["counts"]["regen_full_equivalents"],
        });
        let [n_layers, d_model, n_heads, n_kv_heads, d_head, d_ff, vocab] = self.shape;
        let mut requantized = Map::new();
        for (site, n) in &self.requantized {
            requantized.insert(format!("{site:?}"), json!(n));
        }
        requantized.insert("total".into(), json!(self.requantized_per_layer()));
        let tally = |t: &Tally| {
            json!({
                "lookup_outputs": t.lookup_outputs,
                "split_limbs": t.split_limbs,
                "row_scalars": t.row_scalars,
                "one_hot": t.one_hot,
                "checkpoints": t.checkpoints,
                "total": t.committed(),
            })
        };
        json!({
            "schema": SCHEMA,
            "model": {
                "profile": format!("{:?}", self.profile).to_lowercase(),
                "n_layers": n_layers,
                "d_model": d_model,
                "n_heads": n_heads,
                "n_kv_heads": n_kv_heads,
                "d_head": d_head,
                "d_ff": d_ff,
                "vocab": vocab,
                "parameters": self.parameters,
            },
            "lowering": {
                "positions": self.positions,
                "max_log2_rows": self.max_log2_rows,
                "stages": self.stages,
            },
            "per_layer_per_token": {
                "requantized": requantized,
                "committed": tally(&self.per_layer),
                "lookups": self.per_layer.lookups,
            },
            "per_token": {
                "committed": tally(&self.per_token),
                "lookups": self.per_token.lookups,
            },
            "per_score": {
                "commitments": 1,
                "lookups": 1,
                "product_tree_leaves": 1,
                "scores_per_token_per_ctx": self.scores_per_token_per_ctx(),
            },
            "sequence": {
                "scores": self.scores,
                "product_tree_leaves": self.product_tree_leaves,
            },
            "tables": self.tables.iter().map(|t| json!({
                "id": t.id.0,
                "kind": t.kind,
                "rows": t.rows,
                "lookups_per_token": t.lookups_per_token,
                "score_lookups": t.score_lookups,
            })).collect::<Vec<_>>(),
            "multiplicities": {
                "total": self.multiplicities(),
                "excludes": "tables without lookups and power-of-two padding",
            },
            "stage_tables": self.stage_tables.iter().map(|s| {
                let mut stage = stage_json(s);
                stage["stage"] = json!(s.stage);
                stage
            }).collect::<Vec<_>>(),
            "estimates": [],
            "counts": self.counts(),
            "projection_inputs": projection_inputs,
            "projections": cost_model["projections"],
        })
    }
}

/// The commitment report of `lowering`, whose gadgets `selection` chose under `policy`.
///
/// # Errors
///
/// [`CompileError::Usage`] for a graph the report does not count: one with a scoring certificate,
/// a lowering without its checkpoints, a count that is not a multiple of the sequence's length, or
/// a policy with limbs of no bits.
pub fn report(
    model: PublicModel<'_>,
    lowering: &Lowering,
    selection: &Selection,
    policy: &GadgetPolicy,
) -> Result<CommitmentReport, CompileError> {
    let graph = &lowering.graph;
    let arch = model.arch;
    let n = u64::from(
        *graph.tensors[lowering.positions.0 as usize]
            .shape
            .first()
            .unwrap_or(&1),
    );
    // The first layer's nodes lie between the checkpoints before and after it.
    let (first, last) = match lowering.checkpoints.as_slice() {
        [before, after, ..] => (producer(graph, *before)?, producer(graph, *after)?),
        _ => {
            return Err(CompileError::Usage {
                rule: "the lowering names its checkpoints",
            });
        }
    };
    let limb_bits = u64::from(policy.limb_bits);
    if limb_bits == 0 {
        return Err(CompileError::Usage {
            rule: "limbs have at least one bit",
        });
    }
    let mut tables = Vec::with_capacity(selection.tables.len());
    for spec in &selection.tables {
        tables.push(TableUse {
            id: spec.id,
            kind: kind_name(&spec.kind),
            rows: crate::advice::table_rows(spec)?,
            lookups_per_token: 0,
            score_lookups: 0,
        });
    }
    let mut whole = Tally::default();
    let mut layer = Tally::default();
    let (mut scores, mut leaves) = (0u64, 0u64);
    for &(node, gadget) in &selection.gadgets {
        let node_ref = &graph.nodes[node.0 as usize];
        let elements = elements(graph, node_ref.outputs.first().copied());
        let mut tally = Tally::default();
        let mut used: Vec<(TableId, u64)> = Vec::new();
        match gadget {
            Gadget::Requant1 { table } => {
                tally.lookup_outputs = elements;
                tally.lookups = elements;
                used.push((table, elements));
            }
            Gadget::RequantSplit { low, high, .. } => {
                tally.lookup_outputs = elements;
                tally.split_limbs = elements;
                tally.lookups = 2 * elements;
                used.extend([(low, elements), (high, elements)]);
            }
            Gadget::RequantWide {
                low,
                mid,
                top,
                sign,
                high,
                ..
            } => {
                let parts = 2 + u64::from(low.is_some());
                tally.lookup_outputs = elements;
                tally.split_limbs = parts * elements;
                tally.lookups = (parts + 2) * elements;
                used.extend(low.map(|t| (t, elements)));
                used.extend([mid, top, sign, high].map(|t| (t, elements)));
            }
            Gadget::Lookup { table } => {
                tally.lookup_outputs = elements;
                tally.lookups = elements;
                used.push((table, elements));
            }
            Gadget::SoftmaxRow {
                exp,
                reciprocal,
                limb,
                top,
            } => {
                let Op::SoftmaxRow { sigma, .. } = node_ref.op else {
                    return Err(CompileError::Usage {
                        rule: "a SoftmaxRow gadget sits on a SoftmaxRow node",
                    });
                };
                let shape = &graph.tensors[node_ref.inputs[0].0 as usize].shape;
                let width = u64::from(*shape.last().unwrap_or(&1));
                let rows = elements / width.max(1);
                // Row q of a sequence sees keys 0 to q: n(n + 1)/2 valid scores per n rows.
                let blocks = rows / width.max(1);
                let valid = blocks * width * (width + 1) / 2;
                scores += valid;
                leaves += rows * width.next_power_of_two();
                let bracket = u64::from(sigma) + 18;
                let limbs = bracket.div_ceil(limb_bits);
                // c_q, R_q and two NonNeg of `limbs` limbs, one fewer committed.
                tally.row_scalars = rows * (2 + 2 * (limbs - 1));
                // R_q's range check and every limb's.
                tally.lookups = rows * (1 + 2 * limbs);
                used.push((reciprocal, rows));
                if let Some(limb) = limb {
                    used.push((limb, 2 * (limbs - 1) * rows));
                }
                used.push((top, 2 * rows));
                if let Some(t) = tables.iter_mut().find(|t| t.id == exp) {
                    t.score_lookups += valid;
                }
            }
            Gadget::RsqrtRow => {
                let requests = selection.reads(node).ok_or(CompileError::Usage {
                    rule: "every root has its complete canonical range requests",
                })?;
                let lookups = u64::try_from(requests.len()).map_err(|_| CompileError::Usage {
                    rule: "a root's lookup count fits u64",
                })?;
                // Only the input/root tops are virtual; every other request has a committed value.
                let columns = lookups.checked_sub(2).ok_or(CompileError::Usage {
                    rule: "a root includes both virtual top requests",
                })?;
                tally.row_scalars = elements * columns;
                tally.lookups = elements * lookups;
                used.extend(requests.iter().map(|&id| (id, elements)));
            }
            Gadget::OneHot => tally.one_hot = elements,
            Gadget::Argmax { .. } | Gadget::InverseCdf => {
                return Err(CompileError::Usage {
                    rule: "the report counts the forward pass, without scoring certificates",
                });
            }
        }
        for (id, count) in used {
            if let Some(t) = tables.iter_mut().find(|t| t.id == id) {
                t.lookups_per_token += count;
            }
        }
        add(&mut whole, tally);
        if node > first && node <= last {
            add(&mut layer, tally);
        }
    }
    // Linear outputs committed as advice (RFC-0037): none is inside a layer.
    for node in &graph.nodes {
        if node.op.class() == OpClass::Gadget {
            continue;
        }
        for &output in &node.outputs {
            if matches!(
                graph.tensors[output.0 as usize].kind,
                NodeKind::Committed {
                    commitment: CommitmentClass::Advice
                }
            ) {
                whole.checkpoints += elements(graph, Some(output));
            }
        }
    }
    for t in &mut tables {
        t.lookups_per_token = per_token(t.lookups_per_token, n)?;
    }
    let mut requantized = Vec::new();
    for &(point, tensor) in &lowering.sites {
        if let QuantPoint::Layer { layer: 0, site } = point {
            requantized.push((site, per_token(elements(graph, Some(tensor)), n)?));
        }
    }
    let parameters = graph
        .weights
        .iter()
        .map(|&(_, t)| elements(graph, Some(t)))
        .sum();
    let stage_tables = stage_tables(model, lowering, n)?;
    Ok(CommitmentReport {
        profile: model.profile,
        shape: [
            arch.n_layers,
            arch.d_model,
            arch.n_heads,
            arch.n_kv_heads,
            arch.d_head,
            arch.d_ff,
            arch.vocab,
        ],
        parameters,
        positions: u32::try_from(n).unwrap_or(u32::MAX),
        max_log2_rows: policy.max_log2_rows,
        stages: stage_count(graph),
        requantized,
        per_layer: layer.per_token(n)?,
        per_token: whole.per_token(n)?,
        scores,
        product_tree_leaves: leaves,
        tables,
        stage_tables,
    })
}

/// A stage's tables as the cost model reads them: each class's growth, holding, size, counts and
/// width.
fn stage_json(stage: &StageTables) -> Value {
    let classes = |classes: &[TableClass]| {
        classes
            .iter()
            .map(|c| {
                json!({
                    "growth": match c.growth {
                        Growth::Fixed => "fixed",
                        Growth::Tokens => "tokens",
                        Growth::Keys => "keys",
                        Growth::TokensAndContext => "tokens_and_context",
                    },
                    "kind": match c.kind {
                        Holding::Windowable => "windowable",
                        Holding::Resident => "resident",
                        Holding::Computed => "computed",
                    },
                    "log2_entries": c.log2_entries,
                    "count": c.count,
                    "count_per_bit": c.count_per_bit,
                    "bytes": c.bytes,
                })
            })
            .collect::<Vec<_>>()
    };
    json!({
        "tables": classes(&stage.tables),
        "residents": classes(&stage.residents),
    })
}

/// One instance: its relation, its variables, its tables, and how the prover holds them.
type HeldInstance = (Relation, u32, u64, Holding);

/// A reduction stage's instances.
type StageInstances = (u16, Vec<HeldInstance>);

/// What holds one set of tables in a stage (RFC-0056 'Shared instances').
#[derive(PartialEq, Eq, Hash)]
enum Share<'a> {
    /// Instances of one relation of one tensor at one point.
    Graph(Relation, NodeRef, Kind, &'a [Coord], &'a Layers, u32),
    /// A root's coefficient products, which share its digit and limb tables.
    Products(NodeRef),
    /// Any other native relation.
    Native(u32),
}

/// Each reduction stage's instances in `lowering`'s composed schedule, with the native relations
/// of a request with outputs (RFC-0053, RFC-0056).
/// - **Tables.** A graph instance holds one table per evaluation it leaves, a native one its
///   evaluations.
/// - **Holding.** A matrix product's tables are held whole. `SoftmaxRow`'s key and forest
///   instances hold none.
/// - **Sharing.** Instances that share their tables count them once: one relation of one tensor
///   at one point, and a root's coefficient products.
fn instances(lowering: &Lowering) -> Result<Vec<StageInstances>, CompileError> {
    let natives = crate::schedule::model_natives(lowering, Request::Generate)?;
    let schedule =
        crate::schedule::derive_composed_schedule(lowering, &natives.deferred, &natives.native)?;
    let native_kind = |tag: u32| {
        let at = usize::try_from(tag).unwrap_or(usize::MAX).wrapping_sub(1);
        natives.kinds.get(at).copied()
    };
    let kind_of = |relation: &Relation| match relation {
        Relation::MatMulWeight | Relation::MatMulAct { .. } => Holding::Resident,
        Relation::Native { tag, .. } => match native_kind(*tag) {
            Some(NativeKind::Key | NativeKind::Forest) => Holding::Computed,
            _ => Holding::Windowable,
        },
        _ => Holding::Windowable,
    };
    Ok(schedule
        .stages
        .iter()
        .filter(|s| s.index > 0)
        .map(|s| {
            let mut seen = std::collections::HashSet::new();
            let instances = s
                .instances
                .iter()
                .filter(|i| {
                    let share = match i.relation {
                        Relation::Native { tag, .. } => match native_kind(tag) {
                            Some(NativeKind::Product) => Share::Products(i.node),
                            _ => Share::Native(tag),
                        },
                        relation => {
                            Share::Graph(relation, i.node, i.kind, &i.point, &i.layers, i.vars)
                        }
                    };
                    seen.insert(share)
                })
                .map(|i| {
                    let tables = match i.relation {
                        Relation::Native { evaluations, .. } => evaluations as u64,
                        _ => i.claims.len() as u64,
                    };
                    (i.relation, i.vars, tables, kind_of(&i.relation))
                })
                .collect();
            (s.index, instances)
        })
        .collect())
}

/// The reduction stages' tables, from the schedules of `m` and `m/2` positions, `m` the largest
/// power of two at or below `n`: a stage's classes are the model's, whatever the sequence.
///
/// - **Growth.** Each instance's variables grow by one when the sequence doubles, a per-token
///   instance; by two, an instance over the score tensor; or not at all. Its tables grow by one
///   per doubling, a request's greedy relation, or not at all. The report refuses a graph where
///   one does anything else.
/// - **Residents.** Each stage of `SoftmaxRow`'s key and forest instances also holds what
///   RFC-0056's prover keeps for them: the INT8 queries and keys, and two row tables.
/// - **A sequence of one position.** It counts every table per token and position, each count
///   growing by itself per bit: an upper bound at every batch and context.
fn stage_tables(
    model: PublicModel<'_>,
    lowering: &Lowering,
    n: u64,
) -> Result<Vec<StageTables>, CompileError> {
    let linear = || CompileError::Usage {
        rule: "every stage's tables are linear in the sequence's length or in its square",
    };
    let mut stages = if n < 2 {
        instances(lowering)?
            .into_iter()
            .map(|(stage, instances)| {
                let sizes = instances
                    .into_iter()
                    .map(|(_, vars, count, kind)| {
                        (kind, Growth::TokensAndContext, vars, count, count, 16)
                    })
                    .collect();
                classify(stage, sizes)
            })
            .collect::<Vec<_>>()
    } else {
        let bits = n.ilog2();
        let m = 1u32.checked_shl(bits).ok_or_else(linear)?;
        let at_m = match u64::from(m) == n {
            true => instances(lowering)?,
            false => instances(&crate::lower(model, m)?)?,
        };
        let at_half = instances(&crate::lower(model, m / 2)?)?;
        if at_half.len() != at_m.len() {
            return Err(linear());
        }
        at_m.into_iter()
            .zip(at_half)
            .map(|((stage, full), (other, halved))| {
                if stage != other || full.len() != halved.len() {
                    return Err(linear());
                }
                let sizes = full
                    .into_iter()
                    .zip(halved)
                    .map(|((relation, vars, count, kind), (r, v, c, _))| {
                        let (growth, log2_entries, count, per_bit) =
                            table_class((relation, vars, count), (r, v, c), bits)
                                .ok_or_else(linear)?;
                        Ok((kind, growth, log2_entries, count, per_bit, 16))
                    })
                    .collect::<Result<Vec<_>, CompileError>>()?;
                Ok(classify(stage, sizes))
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    // RFC-0056's residents, in each stage that holds a computed instance: per site, its queries
    // and keys as INT8, and the score and row-maximum row tables. The keys' folds are computed
    // round by round from the keys, and not held.
    let arch = model.arch;
    let log2 = |x: u32| x.next_power_of_two().trailing_zeros();
    let lambda = log2(arch.n_layers);
    let (heads, kv) = (log2(arch.n_heads) + lambda, log2(arch.n_kv_heads) + lambda);
    let d = log2(arch.d_head);
    for stage in &mut stages {
        let sites = stage
            .tables
            .iter()
            .filter(|c| c.kind == Holding::Computed)
            .map(|c| c.count)
            .sum::<u64>()
            / 4;
        if sites == 0 {
            continue;
        }
        let resident = |growth, log2_entries, count, bytes| {
            (Holding::Resident, growth, log2_entries, count, 0, bytes)
        };
        stage.residents = classify(
            stage.stage,
            vec![
                resident(Growth::Tokens, heads + d, sites, 1),
                resident(Growth::Keys, kv + d, sites, 1),
                resident(Growth::Tokens, heads, 2 * sites, 16),
            ],
        )
        .tables;
    }
    Ok(stages)
}

/// A stage's tables by class: one class per growth, kind, size, per-bit count and width, the
/// classes that grow fastest first, then by kind, size and width.
fn classify(stage: u16, sizes: Vec<(Holding, Growth, u32, u64, u64, u8)>) -> StageTables {
    let mut tables: Vec<TableClass> = Vec::new();
    for (kind, growth, log2_entries, count, count_per_bit, bytes) in sizes {
        match tables.iter_mut().find(|c| {
            c.growth == growth
                && c.kind == kind
                && c.log2_entries == log2_entries
                && c.bytes == bytes
                && (c.count_per_bit == 0) == (count_per_bit == 0)
        }) {
            Some(class) => {
                class.count += count;
                class.count_per_bit += count_per_bit;
            }
            None => tables.push(TableClass {
                growth,
                kind,
                log2_entries,
                count,
                count_per_bit,
                bytes,
            }),
        }
    }
    tables.sort_by_key(|c| {
        (
            core::cmp::Reverse(c.growth),
            c.kind,
            c.log2_entries,
            c.bytes,
            c.count_per_bit,
        )
    });
    StageTables {
        stage,
        tables,
        residents: Vec::new(),
    }
}

/// One instance's tables, `(growth, log2_entries, count, count_per_bit)`, from its schedules at
/// `2^bits` positions and at half as many.
/// - **Variables.** The variables it gains when the sequence doubles set the growth: 0, 1 or 2.
///   `log2_entries` is its variables less the growth's bits.
/// - **Tables.** Its tables gain none or one when the sequence doubles: `count_per_bit`, and
///   `count` the rest.
///
/// `None` for anything else: another relation, a growth of more than two variables or one table,
/// or fewer variables or tables than the growth's bits.
fn table_class(
    (relation, vars, count): (Relation, u32, u64),
    (r, v, c): (Relation, u32, u64),
    bits: u32,
) -> Option<(Growth, u32, u64, u64)> {
    // A native relation's evaluations are its tables, which may grow; the rest is its identity.
    let identity = |relation: Relation| match relation {
        Relation::Native {
            tag, degree, zero, ..
        } => Relation::Native {
            tag,
            degree,
            evaluations: 0,
            zero,
        },
        other => other,
    };
    if identity(relation) != identity(r) {
        return None;
    }
    let growth = match vars.checked_sub(v)? {
        0 => Growth::Fixed,
        1 => Growth::Tokens,
        2 => Growth::TokensAndContext,
        _ => return None,
    };
    let steps = match growth {
        Growth::Fixed => 0,
        Growth::TokensAndContext => 2,
        Growth::Tokens | Growth::Keys => 1,
    };
    let log2_entries = vars.checked_sub(steps * bits)?;
    let per_bit = count.checked_sub(c).filter(|&g| g <= 1)?;
    let base = count.checked_sub(per_bit * u64::from(bits))?;
    Some((growth, log2_entries, base, per_bit))
}

/// `x` per position of an `n`-position sequence, exactly.
fn per_token(x: u64, n: u64) -> Result<u64, CompileError> {
    match x.checked_rem(n) {
        Some(0) => Ok(x / n),
        _ => Err(CompileError::Usage {
            rule: "every count of the report is a multiple of the sequence's length",
        }),
    }
}

fn add(total: &mut Tally, part: Tally) {
    total.lookup_outputs += part.lookup_outputs;
    total.split_limbs += part.split_limbs;
    total.row_scalars += part.row_scalars;
    total.one_hot += part.one_hot;
    total.checkpoints += part.checkpoints;
    total.lookups += part.lookups;
}

fn elements(graph: &Graph, tensor: Option<TensorId>) -> u64 {
    tensor.map_or(0, |t| {
        graph.tensors[t.0 as usize]
            .shape
            .iter()
            .map(|&d| u64::from(d))
            .product()
    })
}

fn producer(graph: &Graph, tensor: TensorId) -> Result<NodeId, CompileError> {
    graph.producer(tensor).ok_or(CompileError::Usage {
        rule: "a checkpoint is a node's output",
    })
}

/// RFC-0010's stage count of `graph`: the leaf stage, then one per sumcheck relation on the
/// longest chain from a gadget's input, or from a linear output committed as advice, down to
/// committed tensors. Stage 1 proves such an output equal to its relation by a zero-check, whose
/// claims on the node's inputs go on (RFC-0037).
pub fn stage_count(graph: &Graph) -> u32 {
    // The sumcheck relations a claim on each tensor still needs: none on a committed one, which
    // is opened.
    let mut depth = vec![0u32; graph.tensors.len()];
    for node in &graph.nodes {
        let below = node
            .inputs
            .iter()
            .map(|&t| depth[t.0 as usize])
            .max()
            .unwrap_or(0);
        let own = u32::from(sumcheck(graph, &node.op));
        for &t in &node.outputs {
            if matches!(graph.tensors[t.0 as usize].kind, NodeKind::Virtual) {
                depth[t.0 as usize] = below + own;
            }
        }
    }
    let deepest = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.op.class(), OpClass::Gadget))
        .flat_map(|n| n.inputs.iter())
        .map(|&t| depth[t.0 as usize])
        .max()
        .unwrap_or(0);
    let committed_linear = graph
        .nodes
        .iter()
        .filter(|n| n.op.class() != OpClass::Gadget)
        .filter(|n| {
            n.outputs.iter().any(|&t| {
                matches!(
                    graph.tensors[t.0 as usize].kind,
                    NodeKind::Committed {
                        commitment: CommitmentClass::Advice
                    }
                )
            })
        })
        .map(|n| {
            1 + n
                .inputs
                .iter()
                .map(|&t| depth[t.0 as usize])
                .max()
                .unwrap_or(0)
        })
        .max()
        .unwrap_or(0);
    1 + deepest.max(committed_linear)
}

/// Whether RFC-0010 proves `op` by a sumcheck rather than mapping a claim onto its inputs. A
/// sum over an axis maps its claim to `(½, …, ½)` on the axis, a `Linear` relation.
fn sumcheck(graph: &Graph, op: &Op) -> bool {
    match op {
        Op::MatMulWeight { .. }
        | Op::MatMulAct { .. }
        | Op::Mul { .. }
        | Op::Rotate { .. }
        | Op::SumSquares { .. } => true,
        // A product by one constant scales a claim; by one per channel it does not.
        Op::ConstMul { c } => graph
            .constants
            .get(c.0 as usize)
            .is_some_and(|v| v.len() > 1),
        _ => false,
    }
}

/// RFC-0008's name of a table kind.
fn kind_name(kind: &TableKind) -> &'static str {
    match kind {
        TableKind::Range { .. } => "Range",
        TableKind::Requant { .. } => "Requant",
        TableKind::Pow2 { .. } => "Pow2",
        TableKind::Bitwise8 { .. } => "Bitwise8",
        TableKind::AesSbox => "AesSbox",
        TableKind::Gf256Mul { .. } => "Gf256Mul",
        TableKind::Exp { .. } => "Exp",
        TableKind::Exp2Frac { .. } => "Exp2Frac",
        TableKind::Silu { .. } => "Silu",
        TableKind::Rope { .. } => "Rope",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tessera_ir::Broadcast;

    /// RFC-0055 'Working sets', RFC-0056: a table grows by one variable per doubling of the
    /// sequence, by two over the score tensor, or not at all; its count by one per doubling, a
    /// request's greedy relation, or not at all; each other change refuses the graph.
    #[test]
    fn a_table_grows_by_zero_one_or_two_variables_and_nothing_else() {
        use Growth::{Fixed, Tokens, TokensAndContext};
        let mul = Relation::Mul(Broadcast::Trailing);
        // Per token: 2^20 entries at 16 positions are 2^16 per token.
        assert_eq!(
            table_class((mul, 20, 2), (mul, 19, 2), 4),
            Some((Tokens, 16, 2, 0))
        );
        // Fixed: a weight's folded table.
        let weight = Relation::MatMulWeight;
        assert_eq!(
            table_class((weight, 21, 1), (weight, 21, 1), 4),
            Some((Fixed, 21, 1, 0))
        );
        // Per token and position: the score tensor.
        assert_eq!(
            table_class((mul, 22, 2), (mul, 20, 2), 4),
            Some((TokensAndContext, 14, 2, 0))
        );
        // One table more per doubling: 4 + 3 tables at 16 positions, three and one per bit.
        assert_eq!(
            table_class((mul, 20, 7), (mul, 19, 6), 4),
            Some((Tokens, 16, 3, 1))
        );
        // Growing by three variables or two tables, or shrinking, is none of these.
        assert_eq!(table_class((mul, 23, 2), (mul, 20, 2), 4), None);
        assert_eq!(table_class((mul, 20, 8), (mul, 19, 6), 4), None);
        assert_eq!(table_class((mul, 19, 2), (mul, 20, 2), 4), None);
        assert_eq!(table_class((mul, 20, 2), (mul, 19, 3), 4), None);
        // Another relation is another instance.
        assert_eq!(table_class((mul, 20, 2), (weight, 19, 2), 4), None);
        // A table smaller than its growth's bits holds less than one entry per token, and a count
        // smaller than its bits less than one table.
        assert_eq!(table_class((mul, 3, 1), (mul, 2, 1), 4), None);
        assert_eq!(
            table_class((mul, 4, 1), (mul, 3, 1), 4),
            Some((Tokens, 0, 1, 0))
        );
        assert_eq!(table_class((mul, 7, 1), (mul, 5, 1), 4), None);
        assert_eq!(table_class((mul, 20, 3), (mul, 19, 2), 4), None);
    }

    /// Tables of one size share a class only when they are of one growth, kind, width and
    /// per-bit count.
    #[test]
    fn classes_part_tables_of_one_size_by_kind() {
        use Growth::{Fixed, Keys, Tokens, TokensAndContext};
        use Holding::{Computed, Resident, Windowable};
        let stage = classify(
            1,
            vec![
                (Windowable, Tokens, 14, 2, 0, 16),
                (Windowable, Fixed, 14, 3, 0, 16),
                (Windowable, Tokens, 14, 1, 0, 16),
                (Windowable, Tokens, 14, 3, 1, 16),
                (Resident, Tokens, 14, 5, 0, 16),
                (Resident, Tokens, 14, 4, 0, 1),
                (Resident, Keys, 14, 4, 0, 1),
                (Computed, TokensAndContext, 14, 4, 0, 16),
                (Resident, Fixed, 21, 7, 0, 16),
            ],
        );
        let class = |growth, kind, log2_entries, count, count_per_bit, bytes| TableClass {
            growth,
            kind,
            log2_entries,
            count,
            count_per_bit,
            bytes,
        };
        assert_eq!(
            stage.tables,
            [
                class(TokensAndContext, Computed, 14, 4, 0, 16),
                class(Keys, Resident, 14, 4, 0, 1),
                class(Tokens, Windowable, 14, 3, 0, 16),
                class(Tokens, Windowable, 14, 3, 1, 16),
                class(Tokens, Resident, 14, 4, 0, 1),
                class(Tokens, Resident, 14, 5, 0, 16),
                class(Fixed, Windowable, 14, 3, 0, 16),
                class(Fixed, Resident, 21, 7, 0, 16),
            ]
        );
    }
}
