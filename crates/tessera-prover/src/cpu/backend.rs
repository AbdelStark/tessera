// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::*;
use crate::leaf::table_source::GeneratedCounts;
use crate::leaf::{
    GroupedClaims, GroupedLayout, LeafClaims, LeafLayout, MergeMode, ModelColumn, ModelWitness,
    ProvenGroupedLeaf, ProvenLeaf, SPLIT_LABEL, prepare_leaf_bounded, prove_grouped_leaf_bounded,
    token, tokens,
};
use crate::native::{ModelNativePlan, ModelNativeProver};
use compiler::{Lowering, ModelAdviceLayout, RuntimePlan, schedule::LayerMap};
use tessera_field::GoldilocksExt2 as F;
use tessera_ir::{ClaimList, StageId};
use tessera_sumcheck::SumcheckTranscript;
use tessera_tables::{GeneratedTable, Table as _, TableError};
use tessera_witness::WitnessSource as _;
use tessera_witness::{Engine, Resident, Witness};

use super::plan::{Cells, Pass, PassPlan};
use super::table_plan::TablePlan;

pub(super) struct Unit {
    lowering: Lowering,
    layers: LayerMap,
    runtime: RuntimePlan,
    leaf: LeafLayout<ModelColumn>,
    native: ModelNativePlan,
    /// The leaf stage's token groups, and their layouts when it has any (RFC-0054).
    groups: protocol::LeafGroups,
    grouped: Option<GroupedLayout>,
    windows: Vec<usize>,
    table_allowance: u64,
    table_chunk_rows: NonZeroUsize,
    table_retained: u64,
    forest_bytes: u64,
}
/// A proven leaf stage: in one batch, or in token groups with its atom splits.
enum Leaf {
    Stage(ProvenLeaf),
    Grouped(Box<ProvenGroupedLeaf>, Vec<F>),
}
impl Unit {
    /// The source of the witness pass's tiles for this unit: a grouped stage counts every
    /// group's padding at row 0 (RFC-0054 'When groups apply').
    fn source<'w>(
        &self,
        engine: &'w Engine<'w>,
        model: &IntegerModel,
        witness: &'w Witness,
    ) -> Result<Resident<'w>, ProverError> {
        let source = Resident::runtime(engine, &self.runtime, model, witness)?;
        Ok(match &self.grouped {
            Some(grouped) => source.with_grouped_columns(grouped.columns()?)?,
            None => source,
        })
    }
}
/// The leaf stage's grouping under `budget` (RFC-0054 'When groups apply'): none when `plan`,
/// the stage in one batch, fits; otherwise the fewest group bits whose largest group's lookup
/// trees fit. The planner keeps a token subset of at least 256 tokens and splits layers instead
/// (RFC-0054 'Compute-bound groups'), and splits tokens further only past the layer index's
/// bits. A plan that no grouping fits keeps `plan`, which the budget then refuses.
fn plan_groups(
    plan: PassPlan,
    leaf: &LeafLayout<ModelColumn>,
    layers: &LayerMap,
    tokens: u32,
    budget: MemoryBudget,
    selection: &compiler::Selection,
) -> Result<(PassPlan, Option<GroupedLayout>), ProverError> {
    if let Ok(fit) = plan.clone().for_budget(budget) {
        return Ok((fit, None));
    }
    let token_vars = tokens.next_power_of_two().trailing_zeros();
    for g in 1..=u32::from(protocol::MAX_LEAF_GROUP_BITS) {
        let mut token_bits = g.min(token_vars.saturating_sub(8));
        let mut layer_bits = g - token_bits;
        if layer_bits > layers.vars {
            token_bits += layer_bits - layers.vars;
            layer_bits = layers.vars;
        }
        if token_bits > token_vars {
            break;
        }
        let groups = protocol::LeafGroups {
            token_bits: token_bits as u8,
            layer_bits: layer_bits as u8,
        };
        let grouped = GroupedLayout::new(leaf, layers, groups)?;
        let mut lookups = 0u64;
        for h in 0..grouped.count() {
            let group = grouped.group(h)?;
            let mut held = 0u64;
            for instance in group.instances() {
                held = held
                    .checked_add(1u64.checked_shl(instance.height).ok_or_else(memory)?)
                    .ok_or_else(memory)?;
            }
            lookups = lookups.max(held);
        }
        let candidate = plan.clone().grouped(groups, lookups).tables(TablePlan::new(
            leaf,
            selection,
            Some(&grouped),
        )?);
        if let Ok(fit) = candidate.for_budget(budget) {
            return Ok((fit, Some(grouped)));
        }
    }
    Ok((plan, None))
}
struct Data {
    statement: protocol::PublicStatement,
    unit: Unit,
    witness: Witness,
    seed: random::Seed,
    p1: pcs::AdviceOpening<'static>,
    permit: Option<pcs::ColumnPermit>,
    transcript: Option<Transcript>,
}
pub(super) struct CpuBackend<'a> {
    prepared: Prepared<'a>,
    data: Option<Data>,
}

fn memory() -> ProverError {
    ProverError::Resource(ErrorCode::ResMemory)
}
fn generated_table_error(error: TableError) -> ProverError {
    match error {
        TableError::Memory => memory(),
        _ => internal("selected table regenerates"),
    }
}
fn power(n: u64) -> Result<u64, ProverError> {
    n.checked_next_power_of_two().ok_or_else(memory)
}
fn reserve(budget: MemoryBudget, bytes: Option<u64>) -> Result<(), ProverError> {
    if bytes.is_none_or(|n| n > budget.host) {
        return Err(memory());
    }
    Ok(())
}

/// Conservative protocol/1 regenerated-weight working set, excluding the caller's INT8
/// model. This covers all retained public trees, one codeword block/encoder, two copies of
/// every possible queried column, and the folding pass's field buffers. The proof's resident
/// graph/native cells remain separately charged by `PassPlan`; no witness guard is removed.
pub(super) fn check_weight_memory(c: &ClaimList, budget: MemoryBudget) -> Result<(), ProverError> {
    reserve(budget, Some(weight_memory(c)?))
}
fn weight_memory(c: &ClaimList) -> Result<u64, ProverError> {
    let shapes = c
        .arch
        .weight_shapes()
        .map_err(|_| usage("invalid weight shapes"))?;
    let params =
        ParamSet::defined(&c.params).ok_or_else(|| usage("defined parameters required"))?;
    let k_w = 1u64
        .checked_shl(u32::from(c.layout.weights.log2_k))
        .ok_or_else(memory)?;
    let rate = 1u64
        .checked_shl(u32::from(c.layout.weights.log2_inv_rate))
        .ok_or_else(memory)?;
    let mut columns = 0u64;
    let mut max_columns = 0u64;
    let mut rows = 0u64;
    let mut folds = 0u64;
    let mut channels = 1u64;
    for (index, shape) in shapes.iter().enumerate() {
        let inputs = power(u64::from(shape.inputs))?;
        let outputs = power(u64::from(shape.out))?;
        channels = channels.max(outputs);
        let cells = inputs.checked_mul(outputs).ok_or_else(memory)?;
        let k = inputs.max(k_w.min(cells));
        let n = k.checked_mul(rate).ok_or_else(memory)?;
        let layers = if index == 0 || index == 8 {
            1
        } else {
            u64::from(c.arch.n_layers)
        };
        let packed = k / inputs;
        let count = u64::from(shape.out)
            .div_ceil(packed)
            .checked_mul(layers)
            .ok_or_else(memory)?;
        rows = rows.checked_add(count).ok_or_else(memory)?;
        columns = columns.checked_add(n).ok_or_else(memory)?;
        max_columns = max_columns.max(n);
        folds = folds
            .checked_add(k.checked_mul(power(layers)?).ok_or_else(memory)?)
            .ok_or_else(memory)?;
    }
    // All possible ChannelParams positions, even when the actual model commits fewer.
    let cp_cells = 16u64
        .checked_mul(power(u64::from(c.arch.n_layers))?)
        .and_then(|n| n.checked_mul(channels))
        .ok_or_else(memory)?;
    let cp_k = k_w.min(cp_cells);
    let cp_n = cp_k.checked_mul(rate).ok_or_else(memory)?;
    columns = columns.checked_add(cp_n).ok_or_else(memory)?;
    max_columns = max_columns.max(cp_n);
    rows = rows.checked_add(cp_cells / cp_k).ok_or_else(memory)?;
    folds = folds.checked_add(cp_k).ok_or_else(memory)?;
    let trees = columns.checked_mul(64).ok_or_else(memory)?;
    let block = u64::from(c.layout.advice.row_block)
        .checked_mul(8)
        .and_then(|n| n.checked_add(192))
        .and_then(|n| n.checked_mul(max_columns))
        .ok_or_else(memory)?;
    let answers = rows
        .checked_mul(u64::from(params.weight_queries))
        .and_then(|n| n.checked_mul(16))
        .ok_or_else(memory)?;
    let folds = folds.checked_mul(64).ok_or_else(memory)?;
    let metadata = u64::from(c.arch.n_layers)
        .checked_mul(4096)
        .and_then(|n| n.checked_add(1 << 16))
        .ok_or_else(memory)?;
    trees
        .checked_add(block)
        .and_then(|n| n.checked_add(answers))
        .and_then(|n| n.checked_add(folds))
        .and_then(|n| n.checked_add(metadata))
        .ok_or_else(memory)
}

pub(super) fn model_weights<'a>(
    model: &'a IntegerModel,
    lowering: &Lowering,
) -> Result<pcs::ModelWeights<'a>, ProverError> {
    let mut channels = Vec::new();
    for (multipliers, entries) in [(false, &lowering.classes), (true, &lowering.multipliers)] {
        for (w, _) in entries {
            let c = model
                .quant
                .channels
                .iter()
                .find(|c| c.ty == w.ty && c.layer == w.layer)
                .ok_or_else(|| internal("validated channel parameters exist"))?;
            channels.push(pcs::ChannelParamsRef {
                matrix: w.ty.matrix(),
                layer: w.layer,
                values: if multipliers {
                    pcs::ChannelValues::Multipliers(&c.multiplier)
                } else {
                    pcs::ChannelValues::Classes(&c.class)
                },
            });
        }
    }
    Ok(pcs::ModelWeights {
        tensors: model
            .weights
            .iter()
            .map(|w| pcs::WeightTensorRef {
                matrix: w.ty.matrix(),
                layer: w.layer,
                data: &w.data,
            })
            .collect(),
        channels,
    })
}
fn encoder(c: &ClaimList) -> Result<pcs::RowEncoder, ProverError> {
    let a = c.layout.advice;
    Ok(pcs::RowEncoder::new(
        1usize.checked_shl(u32::from(a.log2_k)).ok_or_else(memory)?,
        u32::from(a.log2_inv_rate),
        a.mask_coeffs as usize,
    )?)
}
fn wire(c: pcs::CombinedRow) -> protocol::CombinedRow {
    protocol::CombinedRow {
        u: c.u,
        mask: c.mask,
    }
}

impl<'a> CpuBackend<'a> {
    pub(super) fn new(prepared: Prepared<'a>, batch: &ProofBatch) -> Result<Self, ProverError> {
        let mut backend = Self {
            prepared,
            data: None,
        };
        backend.commit_batch(batch)?;
        Ok(backend)
    }
    pub(super) fn session(&self) -> Result<[u8; 16], ProverError> {
        Ok(self
            .data
            .as_ref()
            .ok_or_else(|| internal("P1 has a fresh seed"))?
            .seed
            .session())
    }
    /// The proof's unit and its pass plan, from public data alone: the plan is checked before
    /// any table or secret witness data is loaded.
    pub(super) fn unit(
        prepared: &Prepared<'_>,
        batch: &ProofBatch,
    ) -> Result<(Unit, PassPlan), ProverError> {
        let c = &prepared.plan.claims;
        let s = &batch.statement;
        if s.claim_list_digest
            != tessera_hash::hash(tessera_hash::Domain::ClaimList, &tessera_codec::encode(c))
            || s.weights_root != prepared.weights.weights_root()
            || s.weights_epoch != prepared.weights.public().epoch
            || s.predictor
                != (protocol::PredictorSpec::Exact {
                    decoding: protocol::Decoding::Greedy,
                })
            || s.batch.requests.is_empty()
        {
            return Err(usage(
                "an exact-greedy batch must bind this claim list and weight epoch",
            ));
        }
        let mut keys = std::collections::HashSet::new();
        let mut lengths = Vec::with_capacity(s.batch.requests.len());
        let mut requests = Vec::with_capacity(s.batch.requests.len());
        for r in &s.batch.requests {
            if !keys.insert((r.session_id, r.request_id)) {
                return Err(usage("a batch names each request key once"));
            }
            let n = r
                .prompt_len
                .checked_add(r.output_len)
                .ok_or_else(|| usage("request length overflow"))?;
            if r.prompt_len == 0 || n > c.arch.max_context {
                return Err(usage(
                    "requests require a nonempty prompt within max_context",
                ));
            }
            lengths.push(n);
            requests.push((r.prompt_len, r.output_len));
        }
        compiler::check_batch_advice_rows(c, &requests)?;
        let lowering = compiler::lower_batch(PublicModel::of(prepared.model), &lengths)?;
        let n = lowering.graph.tensors[lowering.tokens.0 as usize].shape[0];
        let ranges = compiler::analyze(&lowering.graph, &lowering.tables)?;
        let selected =
            compiler::select_model(c.protocol_version, &lowering, &ranges, &c.gadget_policy)?;
        let runtime = RuntimePlan::derive_batch(&lowering, &selected, c, &requests)?;
        let layers = LayerMap::new(&lowering)?;
        let leaf = LeafLayout::runtime(&lowering, &layers, &runtime, c)?;
        let native = ModelNativePlan::derive_runtime(&lowering, &runtime, c, &leaf)?;
        // The resident cells, at their padded sizes (cpu::plan).
        let mut cells = Cells::default();
        for t in &lowering.graph.tensors {
            let padded = t
                .shape
                .iter()
                .try_fold(1u64, |p, &d| p.checked_mul(power(u64::from(d)).ok()?));
            cells.tensors = cells
                .tensors
                .checked_add(padded.ok_or_else(memory)?)
                .ok_or_else(memory)?;
        }
        for i in leaf.instances() {
            let table = runtime
                .selection()
                .tables
                .iter()
                .find(|t| t.id == i.table)
                .ok_or_else(|| internal("leaf table is selected"))?;
            cells.lookups = cells
                .lookups
                .checked_add(power(i.entries)?)
                .ok_or_else(memory)?;
            cells.table_rows = cells
                .table_rows
                .checked_add(
                    1u64.checked_shl(u32::from(table.log2_rows))
                        .ok_or_else(memory)?,
                )
                .ok_or_else(memory)?;
        }
        let advice = ModelAdviceLayout::runtime(&lowering, &runtime, c)?;
        cells.advice = advice
            .layout()
            .rows()
            .checked_mul(1u64 << c.layout.advice.log2_k)
            .ok_or_else(memory)?;
        struct MetadataOnly;
        impl crate::stages::WitnessSource for MetadataOnly {
            fn values(&self, _: tessera_ir::TensorId) -> Option<&[i64]> {
                None
            }
        }
        let metadata = crate::StageProver::new(&lowering, native.schedule(), &MetadataOnly);
        let shapes = (0..native.schedule().stages.len())
            .map(|k| {
                if k == 0 {
                    Ok(Vec::new())
                } else {
                    metadata.window_shapes(k as u16, |tag| native.window_shape(tag))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let plan = PassPlan::new(native.schedule().stages.len(), cells, weight_memory(c)?)
            .staged(shapes)?
            .tables(TablePlan::new(&leaf, runtime.selection(), None)?);
        let (plan, grouped) = plan_groups(
            plan,
            &leaf,
            &layers,
            n,
            prepared.cfg.memory_budget,
            runtime.selection(),
        )?;
        let table = plan.table_plan().ok_or_else(memory)?;
        let (table_allowance, table_retained, forest_bytes) =
            (table.working_bytes, table.retained, table.forests);
        let table_chunk_rows = NonZeroUsize::new(table.chunk_rows).ok_or_else(memory)?;
        Ok((
            Unit {
                lowering,
                layers,
                runtime,
                leaf,
                native,
                groups: plan.leaf_groups(),
                grouped,
                windows: plan.stage_windows().iter().map(|s| s.cut).collect(),
                table_allowance,
                table_chunk_rows,
                table_retained,
                forest_bytes,
            },
            plan,
        ))
    }
    pub(super) fn check_witness(
        prepared: &Prepared<'_>,
        batch: &ProofBatch,
    ) -> Result<(), ProverError> {
        let (unit, plan) = Self::unit(prepared, batch)?;
        plan.check(prepared.cfg.memory_budget)?;
        let engine = Engine::new(&unit.lowering)?.threads(prepared.threads);
        let witness = engine.run_batch(prepared.model, &batch.tokens)?;
        let source = unit.source(&engine, prepared.model, &witness)?;
        let advice =
            ModelAdviceLayout::runtime(&unit.lowering, &unit.runtime, &prepared.plan.claims)?;
        use tessera_witness::WitnessSource as _;
        for tile in advice.tiles() {
            if matches!(tile.tile, compiler::Tile::GreedyMaximum { .. }) {
                source.tile(tile)?;
            }
        }
        Ok(())
    }

    pub(super) fn round2(
        &mut self,
        tr: &mut Transcript,
    ) -> Result<(protocol::Round2Response, protocol::QueryShape), ProverError> {
        let p = &self.prepared;
        let d = self
            .data
            .as_mut()
            .ok_or_else(|| internal("P1 precedes proof computation"))?;
        let u = &d.unit;
        let c = &p.plan.claims;
        let advice = ModelAdviceLayout::runtime(&u.lowering, &u.runtime, c)?;
        let leaf_stage = span!(DEBUG, "stage", id = 0u16).entered();
        let leaf_started = Instant::now();
        let engine = Engine::new(&u.lowering)?.threads(p.threads);
        let source = u.source(&engine, p.model, &d.witness)?;
        let actual = ModelWitness::new(
            &u.leaf,
            &u.lowering,
            &u.layers,
            &d.witness,
            &source,
            &advice,
        )?;
        let mut xi: [Vec<F>; 10] = core::array::from_fn(|_| Vec::new());
        for m in pcs::WeightMatrix::ALL {
            let rows = p.layout.matrix(m).rows();
            if rows > 0 {
                xi[m.index()] = (0..rows.next_power_of_two().trailing_zeros())
                    .map(|_| tr.challenge(&format!("pcs/weights/{m:?}/xi")))
                    .collect();
            }
        }
        let tables: Vec<GeneratedCounts> = u
            .leaf
            .instances()
            .iter()
            .map(|i| {
                let spec = u
                    .runtime
                    .selection()
                    .tables
                    .iter()
                    .find(|t| t.id == i.table)
                    .ok_or_else(|| internal("leaf table exists"))?;
                let generated = GeneratedTable::from_spec(spec).map_err(generated_table_error)?;
                metrics::lookups(i.table.0, i.entries);
                metrics::table_rows(i.table.0, generated.row_count());
                let tile = advice.tiles().iter().find(|tile| matches!(tile.tile, compiler::Tile::Table { table } if table == i.table.0))
                    .ok_or_else(|| internal("the advice plan has each leaf table's canonical count tile"))?;
                GeneratedCounts::new(generated, source.tile(tile)?).map_err(ProverError::Leaf)
            })
            .collect::<Result<_, _>>()?;
        // `bytes()` includes each used descriptor; charge the vector's unused slots too.
        let unused = (tables.capacity() - tables.len())
            .checked_mul(size_of::<GeneratedCounts>())
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(memory)?;
        let source_bytes = tables.iter().try_fold(
            u.table_retained
                .checked_add(u.forest_bytes)
                .and_then(|n| n.checked_add(unused))
                .ok_or_else(memory)?,
            |total, table| {
                total
                    .checked_add(table.bytes()?)
                    .ok_or(crate::LeafError::Memory)
            },
        )?;
        let table_budget = tessera_logup::TableBatchBudget {
            allowance: u.table_allowance,
            source_bytes,
            chunk_rows: u.table_chunk_rows,
        };
        let tensors = actual.tensors();
        let forests = actual.forests();
        metrics::product_leaves(forests.iter().map(|f| f.len() as u64).sum());
        let mut stages = crate::StageProver::new(&u.lowering, u.native.schedule(), &d.witness);
        let schedule = u.native.schedule();
        let leaf = match &u.grouped {
            None => {
                let lookup =
                    prepare_leaf_bounded(tr, &u.leaf, &tables, &tensors, &forests, table_budget)?;
                let points: Vec<_> = schedule.stages[0]
                    .claims
                    .iter()
                    .map(|c| lookup.point()[..c.point.len()].to_vec())
                    .collect();
                let _graph_leaf = stages.leaf(&points)?;
                Leaf::Stage(lookup.finish(&stages.leaf_split_values()?)?)
            }
            Some(grouped) => {
                let proven = prove_grouped_leaf_bounded(
                    tr,
                    &u.leaf,
                    grouped,
                    &tables,
                    &tensors,
                    &actual.stage_forests(),
                    // RFC-0054 'Pass structure': the merge's table rounds, then P2b.
                    MergeMode::TwoPass { j_star: None },
                    table_budget,
                )?;
                // RFC-0037's claims on each gadget input and on the logits, at their tensors'
                // merged points (RFC-0054 'The stage's outputs').
                let claims = GroupedClaims::new(&proven, grouped);
                let axes = tokens(&u.lowering)?;
                let graph = &u.lowering.graph;
                let mut points = Vec::with_capacity(schedule.sites.len());
                for (site, claim) in schedule.sites.iter().zip(&schedule.stages[0].claims) {
                    let tensor = match *site {
                        compiler::schedule::Site::Gadget { node, input } => {
                            let id = match node {
                                compiler::schedule::NodeRef::Position(position) => {
                                    u.layers.node(position, 0)
                                }
                                compiler::schedule::NodeRef::Node(id) => id,
                            };
                            *graph.nodes[id.0 as usize]
                                .inputs
                                .get(input as usize)
                                .ok_or_else(|| internal("a leaf site's input exists"))?
                        }
                        compiler::schedule::Site::Logits => u.lowering.logits,
                    };
                    let shape = &graph.tensors[tensor.0 as usize].shape;
                    let point = claims
                        .tensor_point(shape, token(&axes, tensor))
                        .filter(|p| p.len() == claim.point.len())
                        .ok_or_else(|| internal("the merge's point holds every leaf claim"))?;
                    points.push(point);
                }
                let _graph_leaf = stages.leaf(&points)?;
                let atoms = stages.leaf_split_values()?;
                tr.absorb_message(SPLIT_LABEL, &atoms);
                Leaf::Grouped(Box::new(proven), atoms)
            }
        };
        let grouped_claims = match (&leaf, &u.grouped) {
            (Leaf::Grouped(proven, _), Some(grouped)) => Some(GroupedClaims::new(proven, grouped)),
            _ => None,
        };
        let claims: &dyn LeafClaims = match (&leaf, &grouped_claims) {
            (Leaf::Stage(proven), _) => proven,
            (Leaf::Grouped(..), Some(claims)) => claims,
            (Leaf::Grouped(..), None) => return Err(internal("a grouped stage has its layouts")),
        };
        event!(
            DEBUG,
            "prover.stage_proved",
            id = 0u16,
            groups = u.grouped.as_ref().map_or(1, GroupedLayout::count);
            "stage {id}: in {groups} groups"
        );
        // No generated table core or count tile survives P2 into reduction stages.
        drop(tables);
        metrics::stage(0, leaf_started.elapsed());
        metrics::phase(metrics::Phase::Stage, leaf_started.elapsed());
        metrics::pass(Pass::P2, leaf_started.elapsed());
        drop(leaf_stage);
        let windowed = u.windows.iter().any(|&c| c != 0);
        // The window engine retains native operands as opaque resident tables;
        // its public projection charges that literal representation (RFC-0055).
        // Computed score states are used only by the ordinary resident engine.
        let mut native = if windowed {
            ModelNativeProver::new_literal(&u.native, &u.lowering, &actual, &d.witness, claims)?
        } else {
            ModelNativeProver::new(&u.native, &u.lowering, &actual, &d.witness, claims)?
        };
        // The native prover owns its remaining leaf inputs; reduction stages read the witness.
        drop(actual);
        let mut proofs = Vec::new();
        let mut regenerate = || {
            engine.regenerate(p.model, &d.witness).map_err(|e| match e {
                tessera_witness::WitnessError::Memory { .. }
                | tessera_witness::WitnessError::MemoryTable { .. } => crate::StageError::Memory,
                _ => crate::StageError::Internal {
                    rule: "the canonical stage witness regenerates",
                },
            })
        };
        for k in 1..u.native.schedule().stages.len() {
            let _stage = span!(DEBUG, "stage", id = k as u16).entered();
            let started = Instant::now();
            let result = if windowed {
                stages.prove_stage_windowed_composed(
                    tr,
                    k as u16,
                    u.windows[k],
                    u.windows.get(k + 1).copied().unwrap_or(0),
                    &mut native,
                    &mut regenerate,
                )?
            } else {
                stages.prove_stage_composed(tr, k as u16, &mut native)?
            };
            metrics::stage(k as u16, started.elapsed());
            metrics::phase(metrics::Phase::Stage, started.elapsed());
            if !windowed {
                metrics::pass(Pass::P3, started.elapsed());
            }
            event!(
                DEBUG,
                "prover.stage_proved",
                id = result.stage,
                rounds = result.sumcheck.rounds.len();
                "stage {id}: {rounds} rounds"
            );
            proofs.push(protocol::StageProof {
                stage: StageId(result.stage),
                sumcheck: result.sumcheck,
                splits: result.splits,
            });
        }
        let targets = crate::ModelOpenings::derive(
            &u.lowering,
            &u.leaf,
            claims,
            &u.native,
            &stages,
            &advice,
            &p.layout,
        )?;
        // Original opening debts now live in `targets`; P4/P5 regenerate their source rows.
        // Clear the completed reduction working sets before allocating combined rows (OBS-6).
        drop(stages);
        drop(native);
        let gamma = tr.challenge("pcs/advice/gamma");
        let av: Vec<_> = (0..advice.layout().row_vars())
            .map(|_| tr.challenge("pcs/advice/xi"))
            .collect();
        let mut theta: [Option<F>; 10] = [None; 10];
        for m in pcs::WeightMatrix::ALL {
            if !targets.weight_points(m).is_empty() && p.layout.matrix(m).rows() > 0 {
                theta[m.index()] = Some(tr.challenge(&format!("pcs/weights/{m:?}/gamma")));
            }
        }
        let encoder = encoder(c)?;
        let advice_opening = span!(DEBUG, "opening", commitment = "advice").entered();
        let opening = Instant::now();
        let prepared = d.p1.prepare(targets.advice(), gamma, &av)?;
        let mut masks = d.seed.masks();
        let mut pass = prepared.begin(&encoder, &mut masks)?;
        for segment in 0..c.layout.advice.segments {
            crate::advice::visit_rows(&u.lowering, &advice, &source, segment, |r, v| {
                pass.push_row(r, v).map_err(Into::into)
            })?;
        }
        let (combined, permit) = pass.finish()?.into_parts();
        let advice_combined: Vec<_> = combined.into_iter().map(wire).collect();
        tr.absorb("pcs/advice/combined", &advice_combined);
        event!(
            DEBUG,
            "prover.opening_combined",
            rows = advice_combined.len();
            "{rows} combined rows"
        );
        drop(advice_opening);
        let weights = model_weights(p.model, &u.lowering)?;
        let mut weight_combined = Vec::new();
        let mut matrices = Vec::new();
        for m in pcs::WeightMatrix::ALL {
            let Some(theta) = theta[m.index()] else {
                continue;
            };
            let _opening = span!(DEBUG, "opening", commitment = Public(format!("{m:?}"))).entered();
            let full = targets.weight_points(m);
            let matrix = p.layout.matrix(m);
            let mut reader = p.layout.rows(m, &weights)?;
            let layer;
            let fold = if m == pcs::WeightMatrix::ChannelParams {
                layer = Vec::new();
                let points = full
                    .iter()
                    .map(|p| &p[matrix.log2_k() as usize..])
                    .collect::<Vec<_>>();
                pcs::WeightFold::channel_params(&p.layout, &mut reader, &points, &xi[m.index()])?
            } else {
                if full.iter().any(|point| point != &full[0]) {
                    return Err(internal("ordinary matrix aliases share one point"));
                }
                let pcs::MatrixKind::Weight { shape, a, h, .. } = matrix.kind() else {
                    return Err(internal("ordinary matrix kind"));
                };
                let input = shape.inputs.next_power_of_two().trailing_zeros() as usize;
                let output = (a + h) as usize;
                layer = full[0][input + output..].to_vec();
                pcs::WeightFold::new(
                    &p.layout,
                    m,
                    &mut reader,
                    &full[0][input..input + output],
                    &xi[m.index()],
                )?
            };
            let combined = fold.combined_rows(&layer, theta)?;
            let packet = protocol::WeightCombinedRows {
                matrix: m.index() as u8,
                evaluations: combined.evaluations.into_iter().map(wire).collect(),
                proximity: wire(combined.proximity),
            };
            tr.absorb(&format!("pcs/weights/{m:?}/combined"), &packet);
            event!(
                DEBUG,
                "prover.opening_combined",
                rows = packet.evaluations.len() + 1;
                "{rows} combined rows"
            );
            weight_combined.push(packet);
            matrices.push(protocol::MatrixCodeword {
                matrix: m.index() as u8,
                len: 1u64 << (matrix.log2_k() + u32::from(c.layout.weights.log2_inv_rate)),
            });
        }
        metrics::phase(metrics::Phase::Opening, opening.elapsed());
        metrics::pass(Pass::P4, opening.elapsed());
        let scoring = protocol::ScoringClaims {
            requests: vec![protocol::FixedBits(0); d.statement.batch.requests.len()],
        };
        tr.absorb("scoring", &scoring);
        let round2 = protocol::Round2Response {
            logup: match &leaf {
                Leaf::Stage(leaf) => protocol::LogUpProof {
                    roots: leaf.logup().roots().to_vec(),
                    values: leaf.logup().values().to_vec(),
                },
                // A grouped stage's roots and values are its groups' (RFC-0054 'Messages').
                Leaf::Grouped(..) => protocol::LogUpProof {
                    roots: Vec::new(),
                    values: Vec::new(),
                },
            },
            trees: match &leaf {
                Leaf::Stage(leaf) => protocol::TreeProof::ungrouped(
                    leaf.logup()
                        .levels()
                        .iter()
                        .map(|p| p.proof().clone())
                        .collect(),
                    leaf.splits().to_vec(),
                ),
                Leaf::Grouped(proven, atoms) => protocol::TreeProof {
                    levels: proven
                        .tables()
                        .levels()
                        .iter()
                        .map(|l| l.proof().clone())
                        .collect(),
                    groups: proven
                        .groups()
                        .iter()
                        .zip(proven.splits())
                        .map(|(group, splits)| protocol::GroupProof {
                            roots: group.roots().to_vec(),
                            levels: group.levels().iter().map(|l| l.proof().clone()).collect(),
                            values: group.values().to_vec(),
                            splits: splits.clone(),
                        })
                        .collect(),
                    table_roots: proven.tables().roots().to_vec(),
                    merge: proven.merge().proof().clone(),
                    splits: atoms.clone(),
                },
            },
            stages: proofs,
            advice_combined,
            weight_combined,
            scoring,
        };
        let mut sumcheck_size = EncodedSize::default();
        sumcheck_size.add(&round2.logup);
        sumcheck_size.add(&round2.trees);
        sumcheck_size.add(&round2.stages);
        metrics::proof_bytes(metrics::ProofPart::Sumcheck, sumcheck_size.finish());
        let mut opening_size = EncodedSize::default();
        opening_size.add(&round2.advice_combined);
        opening_size.add(&round2.weight_combined);
        metrics::proof_bytes(metrics::ProofPart::Openings, opening_size.finish());
        let shape = protocol::QueryShape {
            advice_count: p.cfg.params.advice_queries,
            advice_len: encoder.codeword_len() as u64,
            matrix_count: p.cfg.params.weight_queries,
            matrices,
        };
        d.permit = Some(permit);
        Ok((round2, shape))
    }
}

impl CpuBackend<'_> {
    fn commit_batch(&mut self, batch: &ProofBatch) -> Result<(), ProverError> {
        let _round1 = span!(DEBUG, "round1").entered();
        let p1 = Instant::now();
        let (unit, plan) = Self::unit(&self.prepared, batch)?;
        plan.check(self.prepared.cfg.memory_budget)?;
        let c = &self.prepared.plan.claims;
        let engine = Engine::new(&unit.lowering)?.threads(self.prepared.threads);
        // P1 retains one canonical witness; budgeted stage windows regenerate from its tokens.
        let tile = Public(format!("0..{}", c.arch.n_layers));
        let witness = {
            let _pass = span!(DEBUG, "witness_pass", layer_tile = tile).entered();
            let started = Instant::now();
            let witness = engine.run_batch(self.prepared.model, &batch.tokens)?;
            metrics::phase(metrics::Phase::Witness, started.elapsed());
            event!(
                DEBUG,
                "prover.witness_computed",
                layers = c.arch.n_layers;
                "witness of {layers} layers computed"
            );
            witness
        };
        let committing = Instant::now();
        let source = unit.source(&engine, self.prepared.model, &witness)?;
        let advice = ModelAdviceLayout::runtime(&unit.lowering, &unit.runtime, c)?;
        let id = pcs::CommitmentDescriptor::Advice {
            statement_digest: batch.statement.digest(),
            unit: 0,
            layout: c.layout.advice,
            rows: advice.layout().rows(),
        }
        .commitment_id();
        let encoder = encoder(c)?;
        let seed = random::Seed::new(self.prepared.cfg.mask_rng)?;
        let mut masks = seed.masks();
        let a = c.layout.advice;
        let params = pcs::CommitmentParams {
            log2_k: u32::from(a.log2_k),
            log2_inv_rate: u32::from(a.log2_inv_rate),
            mask_coeffs: a.mask_coeffs as usize,
            row_block: NonZeroUsize::new(a.row_block as usize)
                .ok_or_else(|| usage("zero row block"))?,
            commitment_id: id,
        };
        let mut parts = Vec::new();
        for segment in 0..a.segments {
            let mut pass = pcs::StreamingCommitter::new(
                params,
                advice.layout(),
                segment,
                &encoder,
                &mut masks,
            )?;
            crate::advice::visit_rows(&unit.lowering, &advice, &source, segment, |r, v| {
                pass.push_row(r, v).map_err(Into::into)
            })?;
            parts.push(pass.finish()?);
        }
        let opening = pcs::AdviceOpening::new(&parts)?.into_owned();
        metrics::phase(metrics::Phase::Round1, committing.elapsed());
        metrics::pass(Pass::P1, p1.elapsed());
        let rows = advice.layout().rows();
        metrics::committed(metrics::Committed::Advice, rows << a.log2_k);
        metrics::committed(metrics::Committed::Mask, rows * u64::from(a.mask_coeffs));
        event!(
            DEBUG,
            "prover.round1_committed",
            rows = advice.layout().rows(),
            segments = a.segments;
            "advice committed: {rows} rows in {segments} segments"
        );
        self.data = Some(Data {
            statement: batch.statement.clone(),
            unit,
            witness,
            seed,
            p1: opening,
            permit: None,
            transcript: None,
        });
        Ok(())
    }
}

impl ProverEngine for CpuBackend<'_> {
    type Error = ProverError;
    fn commit(
        &mut self,
        _session_id: &[u8; 16],
    ) -> Result<(protocol::PublicStatement, protocol::AdviceCommitment), ProverError> {
        let data = self
            .data
            .as_ref()
            .ok_or_else(|| internal("P1 was prepared for this one-shot session"))?;
        Ok((data.statement.clone(), data.p1.commitment()))
    }
    fn prover_share(&mut self) -> Result<[u8; 32], ProverError> {
        Ok(self
            .data
            .as_ref()
            .ok_or_else(|| internal("P1 has a per-proof seed"))?
            .seed
            .share())
    }
    fn prove(
        &mut self,
        round1: &protocol::Round1,
        seed: &tessera_hash::Digest,
    ) -> Result<(protocol::Round2Response, protocol::QueryShape), ProverError> {
        let mut tr = transcript(Mode::Live, self.prepared.cfg.params, round1)?;
        tr.absorb("seed", seed);
        let answer = self.round2(&mut tr)?;
        self.data
            .as_mut()
            .ok_or_else(|| internal("P4 retains its P1"))?
            .transcript = Some(tr);
        Ok(answer)
    }
    fn answer(
        &mut self,
        queries: &protocol::ColumnQueries,
    ) -> Result<protocol::Round3Response, ProverError> {
        let _round3 = span!(DEBUG, "round3").entered();
        let answering = Instant::now();
        // Taking Data clears every per-proof allocation on both success and error.
        let mut d = self
            .data
            .take()
            .ok_or_else(|| internal("P1 precedes column answers"))?;
        // Live queries are absorbed even though P5 draws no further challenges.
        if self.prepared.cfg.mode == Mode::Live {
            d.transcript
                .as_mut()
                .ok_or_else(|| internal("Live P5 retains its Round2 transcript"))?
                .absorb("queries", queries);
        }
        let p = &self.prepared;
        let c = &p.plan.claims;
        let u = &d.unit;
        let engine = Engine::new(&u.lowering)?.threads(p.threads);
        let source = u.source(&engine, p.model, &d.witness)?;
        let advice = ModelAdviceLayout::runtime(&u.lowering, &u.runtime, c)?;
        let encoder = encoder(c)?;
        let permit = d
            .permit
            .as_ref()
            .ok_or_else(|| internal("P4 precedes column answers"))?;
        let mut masks = d.seed.masks();
        let indices = queries
            .advice
            .iter()
            .map(|&i| u64::from(i))
            .collect::<Vec<_>>();
        let mut pass = d.p1.begin_columns(permit, &encoder, &mut masks, &indices)?;
        for segment in 0..c.layout.advice.segments {
            crate::advice::visit_rows(&u.lowering, &advice, &source, segment, |r, v| {
                pass.push_row(r, v).map_err(Into::into)
            })?;
        }
        let answer = pass.finish()?;
        let mut weight_columns = Vec::new();
        for q in &queries.weights {
            let m = pcs::WeightMatrix::ALL
                .get(q.matrix as usize)
                .ok_or_else(|| internal("checked query matrix"))?;
            let indices = q.indices.iter().map(|&i| u64::from(i)).collect::<Vec<_>>();
            let source = model_weights(p.model, &u.lowering)?;
            let a = p.weights.answer(*m, &source, &indices)?;
            weight_columns.push(protocol::MatrixColumns {
                matrix: q.matrix,
                columns: a.columns,
                proof: protocol::MultiProof {
                    siblings: a.proof.siblings,
                },
            });
        }
        event!(
            DEBUG,
            "prover.round3_answered",
            advice = indices.len(),
            matrices = weight_columns.len();
            "{advice} advice columns, {matrices} matrices answered"
        );
        let round3 = protocol::Round3Response {
            advice_columns: answer.columns,
            advice_proof: protocol::MultiProof {
                siblings: answer.proof.siblings,
            },
            weight_columns,
        };
        metrics::phase(metrics::Phase::Round3, answering.elapsed());
        metrics::pass(Pass::P5, answering.elapsed());
        metrics::proof_bytes(metrics::ProofPart::Columns, EncodedSize::of(&round3));
        Ok(round3)
    }
    fn leaf_groups(&self) -> protocol::LeafGroups {
        self.data
            .as_ref()
            .map_or(protocol::LeafGroups::NONE, |d| d.unit.groups)
    }
    fn discard(&mut self) {
        self.data = None;
    }
}

#[cfg(test)]
mod memory_error_tests {
    use super::*;
    #[test]
    fn core_resource_failures_keep_their_code_through_every_prover_wrapper() {
        for error in [TableError::Memory, TableError::Digest] {
            let expected = if error == TableError::Memory {
                ErrorCode::ResMemory
            } else {
                ErrorCode::Internal
            };
            let table = tessera_ir::TableId(11);
            let gadget = tessera_gadgets::GadgetError::Spec { table, error };
            let errors = [
                generated_table_error(error),
                ProverError::from(tessera_witness::WitnessError::Table { table, error }),
                ProverError::from(tessera_witness::WitnessError::from(gadget)),
                ProverError::from(crate::LeafError::from(gadget)),
                ProverError::from(crate::StageError::from_table(error, "canonical lookup")),
            ];
            for outer in errors {
                assert_eq!(outer.code(), expected, "{outer:?}");
            }
            let model =
                ProverError::from(tessera_compiler::CompileError::ModelTable { table, error });
            assert_eq!(
                model.code(),
                if error == TableError::Memory {
                    ErrorCode::ResMemory
                } else {
                    ErrorCode::QuantArch
                }
            );
        }
        assert!(matches!(
            generated_table_error(TableError::Digest),
            ProverError::Protocol(protocol::ProtocolError::Internal {
                rule: "selected table regenerates"
            })
        ));
    }
}
