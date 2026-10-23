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

use crate::schedule::Relation;
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

/// Tables of one size in a reduction stage: `count` tables of `2^{log2_entries}` entries, per
/// token of the batch when `per_token` (RFC-0055 'Working sets'). A stage's instances of `n_i`
/// variables each hold one table per evaluation they leave, and a batch of `T` tokens, a power of
/// two, gives a per-token instance `log2_entries + log2(T)` variables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TableClass {
    /// Whether the tables grow with the batch: their entries are per token.
    pub per_token: bool,
    /// Whether a window may fold them: an elementwise relation's operands, which a regeneration
    /// yields block by block. A matrix product's tables are folds over another index, and stay
    /// resident (RFC-0055 'The window').
    pub windowable: bool,
    /// `log2` of the entries of one table, per token when `per_token`.
    pub log2_entries: u32,
    /// The tables of this size.
    pub count: u64,
}

/// A reduction stage's sumcheck tables (RFC-0055 'Working sets'). They do not depend on the
/// context: a stage's tables are its instances' elementwise operands and folded matrix products,
/// each linear in the tokens or independent of them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageTables {
    /// The stage, from 1.
    pub stage: u16,
    /// Its tables by size, per-token classes first, each by increasing size.
    pub tables: Vec<TableClass>,
}

impl StageTables {
    /// The entries of the stage's per-token tables, per token, or of its other tables, saturating
    /// at `u64::MAX`.
    pub fn entries(&self, per_token: bool) -> u64 {
        self.tables
            .iter()
            .filter(|c| c.per_token == per_token)
            .map(|c| {
                let size = 1u64.checked_shl(c.log2_entries).unwrap_or(u64::MAX);
                c.count.saturating_mul(size)
            })
            .fold(0, u64::saturating_add)
    }
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
                stage["table_entries_per_token"] = json!(s.entries(true));
                stage["table_entries_fixed"] = json!(s.entries(false));
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

/// A stage's tables as the cost model reads them: each class's `count`, `log2_entries` and
/// `per_token`.
fn stage_json(stage: &StageTables) -> Value {
    json!({
        "tables": stage.tables.iter().map(|c| json!({
            "per_token": c.per_token,
            "windowable": c.windowable,
            "log2_entries": c.log2_entries,
            "count": c.count,
        })).collect::<Vec<_>>(),
    })
}

/// A reduction stage's instances: each one's relation, variables and tables, one table per
/// evaluation it leaves.
type StageInstances = (u16, Vec<(Relation, u32, u64)>);

/// Each reduction stage's instances in `lowering`.
fn instances(lowering: &Lowering) -> Result<Vec<StageInstances>, CompileError> {
    Ok(crate::schedule::derive_schedule(lowering)?
        .stages
        .iter()
        .filter(|s| s.index > 0)
        .map(|s| {
            let instances = s
                .instances
                .iter()
                .map(|i| (i.relation, i.vars, i.claims.len() as u64))
                .collect();
            (s.index, instances)
        })
        .collect())
}

/// The reduction stages' tables, from the schedules of `n` and `n/2` positions: each instance's
/// variables grow by one when the sequence doubles, a per-token instance, or not at all, and the
/// report refuses a graph where one does anything else. A sequence that is not a power of two of
/// at least 2 counts every table as per token, `2^{n_i − ⌊log2 n⌋}` entries per token: an upper
/// bound at every larger batch.
fn stage_tables(
    model: PublicModel<'_>,
    lowering: &Lowering,
    n: u64,
) -> Result<Vec<StageTables>, CompileError> {
    let linear = || CompileError::Usage {
        rule: "every stage's tables are linear in the sequence's length",
    };
    let at_n = instances(lowering)?;
    let bits = n.max(1).ilog2();
    if n < 2 || !n.is_power_of_two() {
        return Ok(at_n
            .into_iter()
            .map(|(stage, instances)| {
                let sizes = instances
                    .into_iter()
                    .map(|(relation, vars, count)| {
                        (relation, true, vars.saturating_sub(bits), count)
                    })
                    .collect();
                classify(stage, sizes)
            })
            .collect());
    }
    let half = u32::try_from(n / 2).map_err(|_| linear())?;
    let at_half = instances(&crate::lower(model, half)?)?;
    if at_half.len() != at_n.len() {
        return Err(linear());
    }
    at_n.into_iter()
        .zip(at_half)
        .map(|((stage, full), (other, halved))| {
            if stage != other || full.len() != halved.len() {
                return Err(linear());
            }
            let sizes = full
                .into_iter()
                .zip(halved)
                .map(|(full, half)| {
                    let (per_token, log2_entries, count) =
                        table_class(full, half, bits).ok_or_else(linear)?;
                    Ok((full.0, per_token, log2_entries, count))
                })
                .collect::<Result<Vec<_>, CompileError>>()?;
            Ok(classify(stage, sizes))
        })
        .collect()
}

/// A stage's tables by class: one class per kind (per token or fixed, windowable or not) and
/// size, per-token classes first, each kind by increasing size. A matrix product's tables are
/// not windowable.
fn classify(stage: u16, sizes: Vec<(Relation, bool, u32, u64)>) -> StageTables {
    let mut tables: Vec<TableClass> = Vec::new();
    for (relation, per_token, log2_entries, count) in sizes {
        let windowable = !matches!(
            relation,
            Relation::MatMulWeight | Relation::MatMulAct { .. }
        );
        match tables.iter_mut().find(|c| {
            c.per_token == per_token && c.windowable == windowable && c.log2_entries == log2_entries
        }) {
            Some(class) => class.count += count,
            None => tables.push(TableClass {
                per_token,
                windowable,
                log2_entries,
                count,
            }),
        }
    }
    tables.sort_by_key(|c| (!c.per_token, !c.windowable, c.log2_entries));
    StageTables { stage, tables }
}

/// One instance's tables, `(per_token, log2_entries, count)`, from its schedules at `2^bits`
/// positions and at half as many: per token when its variables grow by one, `log2_entries` then
/// being its variables per token; fixed when they do not grow. `None` for anything else: another
/// relation or table count, a growth of more than one variable, or fewer variables than the
/// sequence's bits.
fn table_class(
    (relation, vars, count): (Relation, u32, u64),
    (r, v, c): (Relation, u32, u64),
    bits: u32,
) -> Option<(bool, u32, u64)> {
    if relation != r || count != c {
        return None;
    }
    match vars.checked_sub(v) {
        Some(1) if vars >= bits => Some((true, vars - bits, count)),
        Some(0) => Some((false, vars, count)),
        _ => None,
    }
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

    /// RFC-0055 'Working sets': a table grows by one variable per doubling of the sequence or not
    /// at all, and each other change refuses the graph.
    #[test]
    fn a_table_is_per_token_or_fixed_and_nothing_else() {
        let mul = Relation::Mul(Broadcast::Trailing);
        // Per token: 2^20 entries at 16 positions are 2^16 per token.
        assert_eq!(
            table_class((mul, 20, 2), (mul, 19, 2), 4),
            Some((true, 16, 2))
        );
        // Fixed: a weight's folded table.
        let weight = Relation::MatMulWeight;
        assert_eq!(
            table_class((weight, 21, 1), (weight, 21, 1), 4),
            Some((false, 21, 1))
        );
        // Growing by two variables, or shrinking, is not linear in the tokens.
        assert_eq!(table_class((mul, 20, 2), (mul, 18, 2), 4), None);
        assert_eq!(table_class((mul, 19, 2), (mul, 20, 2), 4), None);
        // Another relation or table count is another instance.
        assert_eq!(table_class((mul, 20, 2), (weight, 19, 2), 4), None);
        assert_eq!(table_class((mul, 20, 2), (mul, 19, 1), 4), None);
        // A per-token table smaller than the sequence holds less than one entry per token.
        assert_eq!(table_class((mul, 3, 1), (mul, 2, 1), 4), None);
        assert_eq!(table_class((mul, 4, 1), (mul, 3, 1), 4), Some((true, 0, 1)));
    }

    /// Tables of one size share a class only when they are of one kind: per token or fixed, and
    /// windowable or a matrix product's.
    #[test]
    fn classes_part_tables_of_one_size_by_kind() {
        let mul = Relation::Mul(Broadcast::Trailing);
        let weight = Relation::MatMulWeight;
        let stage = classify(
            1,
            vec![
                (mul, true, 14, 2),
                (mul, false, 14, 3),
                (Relation::SumSquares, true, 14, 1),
                (weight, true, 14, 5),
                (weight, false, 21, 7),
            ],
        );
        let class = |per_token, windowable, log2_entries, count| TableClass {
            per_token,
            windowable,
            log2_entries,
            count,
        };
        assert_eq!(
            stage.tables,
            [
                class(true, true, 14, 3),
                class(true, false, 14, 5),
                class(false, true, 14, 3),
                class(false, false, 21, 7),
            ]
        );
        assert_eq!(
            (stage.entries(true), stage.entries(false)),
            (8 << 14, (3 << 14) + (7 << 21))
        );
    }
}
