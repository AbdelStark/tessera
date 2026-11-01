// SPDX-License-Identifier: Apache-2.0 OR MIT

//! In-progress composition of physical phase bounds for RFC-0018 / #641.
//! This test-only probe does not admit a route: preparation and protocol ownership
//! must be completed before its component sums can replace the resident floor.

use super::*;
use crate::leaf::{GroupedLayout, LeafLayout, ModelColumn, ModelWitness};
use crate::native::ModelNativePlan;
use compiler::schedule::LayerMap;
use compiler::{Lowering, ModelAdviceLayout, RuntimePlan};
use tessera_field::GoldilocksExt2 as F;
use tessera_sumcheck::{WindowMemory, WindowShape};
use tessera_witness::Engine;

fn memory() -> ProverError {
    ProverError::Resource(ErrorCode::ResMemory)
}
fn sum(values: impl IntoIterator<Item = u64>) -> Result<u64, ProverError> {
    values
        .into_iter()
        .try_fold(0u64, |n, x| n.checked_add(x).ok_or_else(memory))
}
fn times(n: u64, x: u64) -> Result<u64, ProverError> {
    n.checked_mul(x).ok_or_else(memory)
}
fn array<T>(n: u64) -> Result<u64, ProverError> {
    let bytes = times(n, size_of::<T>() as u64)?;
    if usize::try_from(n).is_err() || bytes > isize::MAX as u64 {
        return Err(memory());
    }
    Ok(bytes)
}
fn required(n: Option<u64>) -> Result<u64, ProverError> {
    n.ok_or_else(memory)
}

fn growing<T>(n: u64) -> Result<u64, ProverError> {
    if n == 0 {
        return Ok(0);
    }
    let minimum = if size_of::<T>() == 1 {
        8
    } else if size_of::<T>() <= 1024 {
        4
    } else {
        1
    };
    array::<T>(times(4, n.max(minimum))?)
}
fn sumcheck_messages(vars: u64, degree: u64, evaluations: u64) -> Result<u64, ProverError> {
    sum([
        growing::<Vec<F>>(vars)?,
        times(vars, growing::<F>(degree)?)?,
        growing::<F>(evaluations)?,
    ])
}

// An ungrouped LogUp result owns every original claim/level point; Round2 clones
// only its wire messages. Forest counts below deliberately include every site.
fn leaf_messages(input: &Inputs<'_>, graph_splits: u64) -> Result<(u64, u64), ProverError> {
    if input.grouped.is_some() {
        return Err(usage(
            "physical composition still owes grouped message storage",
        ));
    }
    let instances = input.leaf.instances().len() as u64;
    let forests = input.leaf.sites().len() as u64;
    let trees = sum([times(2, instances)?, forests])?;
    let mut vars = 0u64;
    for tensor in &input.lowering.graph.tensors {
        let v = tensor.shape.iter().try_fold(0u64, |v, &n| {
            sum([
                v,
                u64::from(
                    n.checked_next_power_of_two()
                        .ok_or_else(memory)?
                        .trailing_zeros(),
                ),
            ])
        })?;
        vars = vars.max(sum([v, u64::from(input.layers.vars)])?);
    }
    vars = vars.max(
        input
            .leaf
            .instances()
            .iter()
            .map(|i| u64::from(i.height))
            .max()
            .unwrap_or(0),
    );
    vars = vars.max(
        input
            .runtime
            .selection()
            .tables
            .iter()
            .map(|t| u64::from(t.log2_rows))
            .max()
            .unwrap_or(0),
    );
    // Include the pair bit added to each next-level claim's point.
    vars = sum([vars, 1])?;
    let messages = sumcheck_messages(vars, 7, times(4, trees)?)?;
    let level = sum([
        times(2, growing::<F>(vars)?)?,
        messages,
        growing::<usize>(trees)?,
        growing::<tessera_sumcheck::LevelClaim<F>>(trees)?,
        times(trees, growing::<F>(vars)?)?,
    ])?;
    let splits = sum([growing::<F>(input.leaf.split_len() as u64)?, graph_splits])?;
    let original = sum([
        growing::<tessera_sumcheck::ProvenTreeLevel<F>>(vars)?,
        times(vars, level)?,
        growing::<F>(times(5, instances)?)?,
        growing::<tessera_sumcheck::LevelClaim<F>>(trees)?,
        times(trees, growing::<F>(vars)?)?,
        growing::<tessera_logup::InstanceClaims<F>>(instances)?,
        times(times(3, instances)?, growing::<F>(vars)?)?,
        growing::<tessera_sumcheck::Claim<F>>(forests)?,
        times(forests, growing::<F>(vars)?)?,
        growing::<F>(vars)?,
        splits,
    ])?;
    let wire = sum([
        growing::<tessera_sumcheck::SumcheckProof<F>>(vars)?,
        times(vars, messages)?,
        growing::<F>(times(5, instances)?)?,
        splits,
    ])?;
    Ok((original, wire))
}
fn response_containers(
    input: &Inputs<'_>,
    geometry: &crate::OpeningGeometry,
    params: &ParamSet,
) -> Result<u64, ProverError> {
    let matrices = pcs::WeightMatrix::ALL.len() as u64;
    let requests = input.statement.batch.requests.len() as u64;
    let aq = u64::from(params.advice_queries);
    let wq = u64::from(params.weight_queries);
    let indices = sum([aq, times(matrices, wq)?])?;
    let mut heap = sum([
        // Caller statement, backend copy, sent Round1 and transient public clones.
        times(4, growing::<protocol::RequestRef>(requests)?)?,
        growing::<protocol::StageProof>(input.native.schedule().stages.len() as u64)?,
        growing::<protocol::CombinedRow>(sum([geometry.groups() as u64, 1])?)?,
        growing::<protocol::WeightCombinedRows>(matrices)?,
        growing::<protocol::MatrixColumns>(matrices)?,
        growing::<protocol::MatrixQueries>(matrices)?,
        growing::<protocol::MatrixCodeword>(matrices)?,
        growing::<protocol::FixedBits>(requests)?,
        // challenge_indices owns u64 draws while converting the public u32 queries;
        // P5 holds advice and one weight u64 index vector beside those queries.
        growing::<u32>(indices)?,
        growing::<u64>(indices)?,
        growing::<u64>(sum([aq, wq])?)?,
    ])?;
    for matrix in pcs::WeightMatrix::ALL {
        heap = sum([
            heap,
            growing::<protocol::CombinedRow>(sum([geometry.weight_claims(matrix), 1])?)?,
            growing::<Vec<F>>(geometry.weight_claims(matrix))?,
            growing::<F>(geometry.weight_coordinates(matrix))?,
            growing::<F>(u64::from(input.weights.matrix(matrix).row_vars()))?,
        ])?;
    }
    Ok(heap)
}

struct Inputs<'a> {
    claims: &'a tessera_ir::ClaimList,
    statement: &'a protocol::PublicStatement,
    lowering: &'a Lowering,
    layers: &'a LayerMap,
    runtime: &'a RuntimePlan,
    leaf: &'a LeafLayout<ModelColumn>,
    native: &'a ModelNativePlan,
    grouped: Option<&'a GroupedLayout>,
    weights: &'a pcs::WeightLayout,
}
struct MetadataOnly;
impl crate::stages::WitnessSource for MetadataOnly {
    fn values(&self, _: tessera_ir::TensorId) -> Option<&[i64]> {
        None
    }
}

// Keep all component owners explicit. No favorable field-point collision or
// secret sparsity removes an opening, operand, table or count backing.
fn components(input: &Inputs<'_>, host: u64) -> Result<serde_json::Value, ProverError> {
    let Inputs {
        claims: c,
        statement,
        lowering,
        layers,
        runtime,
        leaf,
        native,
        grouped,
        weights,
    } = input;
    let advice = ModelAdviceLayout::runtime(lowering, runtime, c)?;
    let geometry = crate::OpeningGeometry::derive(
        lowering,
        &runtime.selection().tables,
        leaf,
        *grouped,
        native,
        &advice,
        weights,
    )?;
    let targets = geometry.target_memory(lowering, leaf, native)?;
    let params =
        ParamSet::defined(&c.params).ok_or_else(|| usage("defined parameters required"))?;
    let a = c.layout.advice;
    let commitment = pcs::CommitmentParams {
        log2_k: u32::from(a.log2_k),
        log2_inv_rate: u32::from(a.log2_inv_rate),
        mask_coeffs: a.mask_coeffs as usize,
        row_block: NonZeroUsize::new(a.row_block as usize).ok_or_else(memory)?,
        commitment_id: pcs::CommitmentDescriptor::Advice {
            statement_digest: statement.digest(),
            unit: 0,
            layout: a,
            rows: advice.layout().rows(),
        }
        .commitment_id(),
    };
    let p1 = pcs::AdviceCommitmentMemory::new(advice.layout(), commitment)?;
    let p4 = pcs::AdviceOpeningMemory::new(
        advice.layout(),
        commitment,
        geometry.advice_claims(),
        geometry.advice_images(),
        geometry.groups() as u64,
    )?;
    let p5 = pcs::AdviceColumnMemory::new(
        advice.layout(),
        commitment,
        u64::from(params.advice_queries),
    )?;
    let weight = weights.regenerated_work_memory(
        NonZeroU32::new(a.row_block).ok_or_else(memory)?,
        u64::from(params.weight_queries),
        geometry.weight_claims(pcs::WeightMatrix::ChannelParams),
    )?;
    let replay = tessera_witness::CheckpointMemory::new(lowering)?;
    let tables = Engine::table_memory(&lowering.tables)?;
    let unit = sum([
        required(lowering.heap_bytes())?,
        required(layers.heap_bytes())?,
        required(runtime.heap_upper_bound())?,
        required(leaf.heap_bytes())?,
        required(native.heap_bytes())?,
        grouped.map_or(Ok(0), |g| required(g.heap_bytes()))?,
    ])?;
    // WeightLayout's only heap owner is WeightShapes.channels; MatrixLayout is Copy.
    let weight_layout =
        array::<(pcs::WeightMatrix, u32)>(weights.shapes().channels.capacity() as u64)?;
    let base = sum([
        unit,
        required(advice.heap_upper_bound())?,
        weight_layout,
        weight.retained_bytes(),
        tables.retained_bytes(),
        replay.retained_bytes(),
        p1.retained_bytes(),
    ])?;
    // Public-only engine construction is guarded before loading its table cores.
    if sum([unit, weight.retained_bytes(), tables.peak_bytes()])? > host {
        return Err(memory());
    }
    let engine = Engine::new(lowering)?;
    let count_plans = engine.runtime_multiplicity_plans(&c.arch, runtime)?;
    let count_payload = count_plans
        .iter()
        .map(|p| p.payload_bytes())
        .max()
        .unwrap_or(0);
    let count_total = sum(count_plans.iter().map(|p| p.payload_bytes()))?;
    let mut source_metadata = engine.runtime_source_metadata_memory(runtime)?;
    if let Some(groups) = grouped {
        let columns = groups.columns()?;
        source_metadata = source_metadata
            .with_grouped_columns(columns.len() as u64, columns.capacity() as u64)
            .ok_or_else(memory)?;
    }
    let source = sum([
        source_metadata.bytes(),
        engine.source_replay_validation_memory()?,
        engine.runtime_source_tile_work_memory(runtime)?,
        count_payload,
    ])?;
    let width = 1u64.checked_shl(u32::from(a.log2_k)).ok_or_else(memory)?;
    // table_rows keeps one bounded u64 chunk beside one clearing field row.
    let rank = advice
        .tiles()
        .iter()
        .flat_map(|t| &t.shares)
        .map(|s| s.ranges.len())
        .chain(lowering.graph.tensors.iter().map(|t| t.shape.len()))
        .max()
        .unwrap_or(0);
    let mapping = advice.layout().tensors().iter().try_fold(0u64, |n, p| {
        let bytes = p
            .row_mapping_memory_bytes()
            .and_then(|b| u64::try_from(b).ok())
            .ok_or_else(memory)?;
        Ok::<_, ProverError>(n.max(bytes))
    })?;
    let rows = sum([
        array::<u64>(width.max(1 << 16))?,
        array::<tessera_field::Goldilocks>(width)?,
        array::<u32>(times(4, (rank as u64).max(4))?)?,
        mapping,
    ])?;
    let graph_leaf = crate::StageProver::new(lowering, native.schedule(), &MetadataOnly)
        .work_memory(0)?
        .output_bytes();
    let (leaf_original, leaf_wire) = leaf_messages(input, graph_leaf)?;
    let response_containers = response_containers(input, &geometry, params)?;
    let messages = sum([leaf_original, leaf_wire, response_containers])?;
    let active = sum([base, source, rows, random::heap_bytes(), messages])?;
    let leaf_source = ModelWitness::memory(leaf, lowering, layers, &advice)?;
    let stage_state = crate::stages::StageStateMemory::new(&lowering.graph, native.schedule())?;
    let native_memory = native.memory(lowering, true)?;
    let stage_driver = crate::StageProver::new(lowering, native.schedule(), &MetadataOnly);
    let p2_fixed = sum([
        active,
        leaf_source.peak_bytes(),
        stage_state.construction_peak_bytes(),
        stage_driver.work_memory(0)?.peak_bytes()?,
        native_memory.construction_peak_bytes(),
    ])?;
    let table_allowance = host.checked_sub(p2_fixed).ok_or_else(memory)?;
    let table =
        table_plan::TablePlan::new(leaf, runtime.selection(), *grouped, Some(&count_plans))?
            .select(table_allowance)?;
    // The canonical native reservation counts every Term allocation. Its total
    // therefore bounds any individual tag's term count, including one fallback.
    let native_terms =
        (native_memory.instance_bytes() / size_of::<tessera_sumcheck::Term<F>>() as u64).max(1);
    let mut stage_rows = Vec::new();
    let mut stages = Vec::new();
    let mut max_source = 0;
    let mut max_driver = 0;
    let mut max_window_metadata = 0;
    let mut stage_outputs = 0;
    for k in 1..native.schedule().stages.len() {
        let shapes = stage_driver.window_shapes(k as u16, |tag| native.window_shape(tag))?;
        let source = stage_driver
            .relation_source_memory(k as u16, |tag| Ok(native.window_shape(tag)?.tables))?;
        let metadata = stage_driver.window_metadata_memory(
            k as u16,
            |tag| native.window_shape(tag),
            native_terms,
        )?;
        let driver = stage_driver.work_memory(k as u16)?.peak_bytes()?;
        max_source = max_source.max(source.metadata_bytes());
        max_driver = max_driver.max(driver);
        max_window_metadata = max_window_metadata.max(metadata.peak_bytes());
        stage_outputs = sum([stage_outputs, metadata.output_bytes()])?;
        stage_rows.push(
            serde_json::json!({"stage":k,"source_metadata":source.metadata_bytes(),
            "ordinary_source_inputs":source.resident_input_bytes(),"driver":driver,
            "window_metadata":metadata.peak_bytes(),"output":metadata.output_bytes()}),
        );
        stages.push(shapes);
    }
    let stage_fixed = sum([
        active,
        stage_state.construction_peak_bytes(),
        native_memory.prepared_peak_bytes(),
        native_memory.instance_bytes(),
        replay.peak_bytes(),
        times(2, max_source)?,
        times(2, max_window_metadata)?,
        max_driver,
        stage_outputs,
    ])?;
    let window_allowance = host.checked_sub(stage_fixed).ok_or_else(memory)? / 2;
    let mut windows = Vec::new();
    for shapes in &stages {
        windows.push(positive_cut(shapes, window_allowance)?);
    }
    let stage_peak = sum([
        stage_fixed,
        times(
            2,
            windows.iter().map(|p| p.working_bytes).max().unwrap_or(0),
        )?,
    ])?;
    let p1_peak = sum([
        unit,
        weight.retained_bytes(),
        weight_layout,
        tables.peak_bytes(),
        required(advice.heap_upper_bound())?,
        replay.peak_bytes(),
        source,
        rows,
        random::heap_bytes(),
        response_containers,
        p1.peak_bytes(),
    ])?;
    let p2_peak = sum([p2_fixed, table.working_bytes])?;
    let p4_peak = sum([
        active,
        p1.encoder_peak_bytes(),
        targets.construction_peak_bytes(),
        p4.peak_bytes(),
        weight.model_refs_bytes(),
        weight.fold_peak_bytes(),
        weight.combined_output_bytes(),
        stage_outputs,
    ])?;
    let p5_peak = sum([
        active,
        p1.encoder_peak_bytes(),
        targets.retained_bytes(),
        p4.output_bytes(),
        weight.combined_output_bytes(),
        p5.peak_bytes(),
        weight.model_refs_bytes(),
        weight.answer_peak_bytes(),
        weight.answer_output_bytes(),
        stage_outputs,
    ])?;
    Ok(
        serde_json::json!({"complete_bound":false,"route_admitted":false,"proof_generated":false,
        "host":host,"unit_metadata":unit,"base":base,"source_work":source,"source_rows":rows,"message_owners":messages,
        "leaf_original":leaf_original,"leaf_wire_copy":leaf_wire,"response_containers":response_containers,
        "engine_retained":tables.retained_bytes(),"count_backings":count_total,
        "replay_retained":replay.retained_bytes(),"replay_constructor":replay.peak_bytes(),
        "geometry_constructor":geometry.construction_peak_bytes(lowering,leaf,*grouped,native,&advice)?,
        "p1_components":p1_peak,"p2_components":p2_peak,"p3_components":stage_peak,
        "p4_components":p4_peak,"p5_components":p5_peak,
        "stages":stage_rows,"cuts":windows.iter().map(|p|p.cut).collect::<Vec<_>>(),
        "unpriced":["public preparation and sizing constructors",
            "NI and Live packet encodings",
            "session/query containers","caller model and process overhead"]}),
    )
}
fn positive_cut(shapes: &[WindowShape], allowance: u64) -> Result<WindowMemory, ProverError> {
    let rounds = shapes.iter().map(|s| s.vars).max().unwrap_or(0);
    for cut in 1..rounds {
        if let Ok(plan) = WindowMemory::for_cut(shapes, cut, size_of::<F>())
            && plan.working_bytes <= allowance
        {
            return Ok(plan);
        }
    }
    Err(memory())
}

#[test]
fn physical_windows_keep_positive_cuts_and_opaque_operands() {
    let shapes = [
        WindowShape {
            vars: 24,
            tables: 2,
            delta: 2,
            windowable: true,
        },
        WindowShape {
            vars: 10,
            tables: 4,
            delta: 3,
            windowable: false,
        },
    ];
    let plan = positive_cut(&shapes, 2 << 20).unwrap();
    assert_eq!(plan.cut, 9);
    assert!(plan.working_bytes <= 2 << 20);
    assert!(matches!(
        positive_cut(&shapes, 0),
        Err(ProverError::Resource(ErrorCode::ResMemory))
    ));
    let one = [WindowShape {
        vars: 1,
        tables: 1,
        delta: 1,
        windowable: true,
    }];
    assert!(matches!(
        positive_cut(&one, 64 << 20),
        Err(ProverError::Resource(ErrorCode::ResMemory))
    ));
}

#[test]
fn message_reservations_reject_overflow_without_allocating() {
    for (vars, degree, evaluations) in [
        (u64::MAX, 7, 1),
        (1, u64::MAX, 1),
        (1, 7, u64::MAX),
        (1u64 << 62, 7, 1),
        (1, 1u64 << 62, 1),
        (1, 7, 1u64 << 62),
    ] {
        let mut result = None;
        let measured = allocation_counter::measure(|| {
            result = Some(sumcheck_messages(vars, degree, evaluations));
        });
        assert_eq!(measured.count_total, 0);
        assert!(matches!(
            result.unwrap(),
            Err(ProverError::Resource(ErrorCode::ResMemory))
        ));
    }
    assert!(sumcheck_messages(32, 7, 128).unwrap() > 0);
}

#[test]
#[ignore = "explicit public real-model component probe; not a complete memory qualification"]
fn public_component_probe() {
    use tessera_codec::{DecodeLimits, envelope, schema};
    let path = std::env::var_os("TESSERA_PHYSICAL_PLAN").expect("public execution plan path");
    let context =
        std::env::var_os("TESSERA_PHYSICAL_REQUEST").expect("public request metadata path");
    let epoch = std::env::var_os("TESSERA_PHYSICAL_WEIGHTS").expect("public weight epoch path");
    let bytes = std::fs::read(path).unwrap();
    let plan: ExecutionPlan =
        envelope::open(&bytes, &schema::EXECUTION_PLAN, DecodeLimits::INTEGER_MODEL).unwrap();
    let weights: pcs::WeightCommitmentPublic = envelope::open(
        &std::fs::read(epoch).unwrap(),
        &schema::WEIGHT_COMMITMENT_PUBLIC,
        DecodeLimits::INTEGER_MODEL,
    )
    .unwrap();
    let request: serde_json::Value =
        serde_json::from_slice(&std::fs::read(context).unwrap()).unwrap();
    let parse_hex = |text: &str| {
        text.as_bytes()
            .chunks_exact(2)
            .map(|c| u8::from_str_radix(std::str::from_utf8(c).unwrap(), 16).unwrap())
            .collect::<Vec<_>>()
    };
    let c = &plan.claims;
    let prompt = u32::try_from(request["prompt_len"].as_u64().unwrap()).unwrap();
    assert_eq!(prompt, 1, "frozen two-token request");
    let statement = protocol::PublicStatement {
        claim_list_digest: tessera_hash::hash(
            tessera_hash::Domain::ClaimList,
            &tessera_codec::encode(c),
        ),
        weights_root: weights.weights_root(),
        weights_epoch: weights.epoch,
        manifest_digest: parse_hex(request["manifest_digest"].as_str().unwrap())
            .try_into()
            .unwrap(),
        batch: protocol::BatchDescriptor {
            requests: vec![protocol::RequestRef {
                request_id: request["request_id"].as_u64().unwrap(),
                session_id: parse_hex(request["request_session"].as_str().unwrap())
                    .try_into()
                    .unwrap(),
                prompt_len: prompt,
                output_len: 1,
            }],
        },
        predictor: protocol::PredictorSpec::Exact {
            decoding: protocol::Decoding::Greedy,
        },
    };
    let lower = compiler::lower(PublicModel::of_claims(c), 2).unwrap();
    let ranges = compiler::analyze(&lower.graph, &lower.tables).unwrap();
    let selection =
        compiler::select_model(c.protocol_version, &lower, &ranges, &c.gadget_policy).unwrap();
    let runtime = RuntimePlan::derive(&lower, &selection, c, prompt, 0).unwrap();
    let layers = LayerMap::new(&lower).unwrap();
    let leaf = LeafLayout::runtime(&lower, &layers, &runtime, c).unwrap();
    let native = ModelNativePlan::derive_runtime(&lower, &runtime, c, &leaf).unwrap();
    let layout = pcs::WeightLayout::new(
        c.layout.weights,
        pcs::WeightShapes {
            layers: c.arch.n_layers,
            shapes: c.arch.weight_shapes().unwrap(),
            channels: lower
                .classes
                .iter()
                .chain(&lower.multipliers)
                .map(|(w, _)| (w.ty.matrix(), w.layer))
                .collect(),
        },
    )
    .unwrap();
    let input = Inputs {
        claims: c,
        statement: &statement,
        lowering: &lower,
        layers: &layers,
        runtime: &runtime,
        leaf: &leaf,
        native: &native,
        grouped: None,
        weights: &layout,
    };
    let bill = components(&input, request["memory_budget_bytes"].as_u64().unwrap()).unwrap();
    use std::io::Write as _;
    let output = std::env::var_os("TESSERA_PHYSICAL_RECEIPT").expect("fresh public receipt path");
    let mut output = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output)
        .unwrap();
    output
        .write_all(&serde_json::to_vec_pretty(&bill).unwrap())
        .unwrap();
}
