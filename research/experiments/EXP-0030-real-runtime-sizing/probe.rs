// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Metadata-only research adapter; never produces a usable claim artifact or a proof.
#![forbid(unsafe_code)]

use serde_json::{Value, json};
use std::{collections::BTreeMap, error::Error, fs, path::Path};
use tessera_codec::{DecodeLimits, envelope, schema};
use tessera_compiler::schedule::LayerMap;
use tessera_compiler::{
    Gadget, Lowering, ModelAdviceLayout, PublicModel, RuntimePlan, Selection, analyze, lower_batch,
    select,
};
use tessera_ir::{ClaimList, IntegerModel, TableId, TensorId};
use tessera_prover::native::ModelNativePlan;
use tessera_prover::{Cells, LeafLayout, MemoryBudget, PassPlan, StageProver};
use tessera_sumcheck::WindowMemory;
use tessera_tables::{GeneratedTable, Table, TableSpec};

#[path = "../EXP-0029-uniform-requant/layout_probe.rs"]
#[allow(dead_code)]
mod candidate;

const RULE: &str = "a site's real layers read the same tensors over the same axes (RFC-0045 'One claim per input')";

fn power(n: u64) -> u64 {
    n.checked_next_power_of_two().expect("public metadata fits")
}
fn ranges(v: &[std::ops::Range<u64>]) -> Vec<Value> {
    v.iter().map(|r| json!([r.start, r.end])).collect()
}
fn cells(c: Cells) -> Value {
    json!({"tensors":c.tensors,"lookups":c.lookups,"table_rows":c.table_rows,"advice":c.advice})
}
fn intern(tables: &mut Vec<TableSpec>, old: &TableSpec) -> Result<TableId, Box<dyn Error>> {
    if let Some(found) = tables
        .iter()
        .filter(|t| t.kind == old.kind)
        .min_by_key(|t| t.id)
    {
        return Ok(found.id);
    }
    let id = TableId(tables.iter().map(|t| t.id.0 + 1).max().unwrap_or(0));
    tables.push(GeneratedTable::new(id, old.kind.clone())?.spec().clone());
    Ok(id)
}
// Rebuild the complete cache, including virtual/repeated reads, before deriving any layout.
fn finalize(lower: &Lowering, selected: &mut Selection) -> Result<(), Box<dyn Error>> {
    let original = selected.clone();
    let mut tables = lower.tables.clone();
    let mut mapping = BTreeMap::new();
    let mut reads = BTreeMap::new();
    for (id, gadget) in &mut selected.gadgets {
        let mut ordered = Vec::new();
        for old in original.reads(*id).ok_or("complete ordered cache")? {
            let spec = original
                .tables
                .iter()
                .find(|t| t.id == *old)
                .ok_or("old table")?;
            let new = intern(&mut tables, spec)?;
            assert!(mapping.insert(*old, new).is_none_or(|prior| prior == new));
            ordered.push(new);
        }
        let remap = |id: &mut TableId| -> Result<(), Box<dyn Error>> {
            *id = *mapping.get(id).ok_or("every table field has a read")?;
            Ok(())
        };
        match gadget {
            Gadget::Requant1 { table } | Gadget::Lookup { table } | Gadget::Argmax { table } => {
                remap(table)?
            }
            Gadget::RequantSplit { low, high, .. } => {
                remap(low)?;
                remap(high)?;
            }
            Gadget::RequantWide {
                low,
                mid,
                top,
                sign,
                high,
                ..
            } => {
                if let Some(id) = low {
                    remap(id)?;
                }
                remap(mid)?;
                remap(top)?;
                remap(sign)?;
                remap(high)?;
            }
            Gadget::SoftmaxRow {
                exp,
                reciprocal,
                limb,
                top,
            } => {
                remap(exp)?;
                remap(reciprocal)?;
                if let Some(id) = limb {
                    remap(id)?;
                }
                remap(top)?;
            }
            Gadget::RsqrtRow
            | Gadget::OneHot
            | Gadget::InverseCdf
            | Gadget::BitDecompose { .. }
            | Gadget::AssertEqual => (),
        }
        reads.insert(*id, ordered);
    }
    selected.tables = tables;
    selected.reads = reads;
    Ok(())
}

struct MetadataOnly;
impl tessera_prover::stages::WitnessSource for MetadataOnly {
    fn values(&self, _: TensorId) -> Option<&[i64]> {
        None
    }
}

fn measure(
    lower: &Lowering,
    selected: &Selection,
    claims: &ClaimList,
    n: u32,
) -> Result<Value, Box<dyn Error>> {
    let runtime = RuntimePlan::derive_batch(lower, selected, claims, &[(n - 1, 1)])?;
    let layers = LayerMap::new(lower)?;
    let leaf = LeafLayout::runtime(lower, &layers, &runtime, claims)?;
    let native = ModelNativePlan::derive_runtime(lower, &runtime, claims, &leaf)?;
    let advice = ModelAdviceLayout::runtime(lower, &runtime, claims)?;
    let tensors: Vec<_> = lower
        .graph
        .tensors
        .iter()
        .map(|t| json!({"id":t.id.0,"shape":t.shape}))
        .collect();
    let mut c = Cells::default();
    for t in &lower.graph.tensors {
        c.tensors = c
            .tensors
            .checked_add(t.shape.iter().map(|&d| power(u64::from(d))).product())
            .ok_or("tensor overflow")?;
    }
    let instances:Vec<_>=leaf.instances().iter().map(|i| {
        let table=runtime.selection().tables.iter().find(|t|t.id==i.table).expect("used table");
        c.lookups+=power(i.entries); c.table_rows+=1u64<<table.log2_rows;
        json!({"table":i.table.0,"entries":i.entries,"height":i.height,"table_log2_rows":table.log2_rows,"kind":format!("{:?}",table.kind),"blocks":i.blocks.iter().map(|b|json!({"site":b.site,"layer":b.layer,"lookup":b.lookup,"vars":b.vars,"offset":b.offset,"domain":leaf.sites()[b.site].layers[b.layer].lookups[b.lookup].domain})).collect::<Vec<_>>()})
    }).collect();
    let layout = advice.layout();
    c.advice = layout.rows() * (1u64 << claims.layout.advice.log2_k);
    let metadata = StageProver::new(lower, native.schedule(), &MetadataOnly);
    let shapes = (0..native.schedule().stages.len())
        .map(|stage| {
            if stage == 0 {
                Ok(Vec::new())
            } else {
                metadata.window_shapes(stage as u16, |tag| native.window_shape(tag))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut stages = Vec::new();
    for (stage, shape) in native.schedule().stages.iter().zip(&shapes) {
        let windows = if shape.is_empty() {
            vec![json!({"cut":0,"fold_passes":0,"working_bytes":0})]
        } else {
            (0..shape.iter().map(|s|s.vars).max().unwrap_or(0)).map(|cut| {
                match WindowMemory::for_cut(shape,cut,16) {
                    Ok(w)=>json!({"cut":w.cut,"fold_passes":w.fold_passes,"working_bytes":w.working_bytes}),
                    Err(e)=>json!({"cut":cut,"error":format!("{e}")}),
                }
            }).collect()
        };
        stages.push(json!({"index":stage.index,"rounds":stage.rounds,"zero_vars":stage.zero_vars,"instances":stage.instances.iter().map(|i|json!({"vars":i.vars,"claims":[i.claims.start,i.claims.end]})).collect::<Vec<_>>(),"shapes":shape.iter().map(|s|json!({"vars":s.vars,"tables":s.tables,"delta":s.delta,"windowable":s.windowable})).collect::<Vec<_>>(),"windows":windows}));
    }
    let weight_bytes = tessera_prover::cpu::backend::weight_memory(claims)?;
    let plan = PassPlan::new(shapes.len(), c, weight_bytes).staged(shapes)?;
    let budgets=[16u64,64,256,1024].into_iter().map(|gib| {
        let budget=gib<<30;
        match plan.clone().for_budget(MemoryBudget{host:budget}) {
            Ok(fit)=>{assert!(fit.check(MemoryBudget{host:budget}).is_ok());json!({"gib":gib,"fits":true,"resident_bytes":fit.resident_bytes(),"cuts":fit.stage_windows().iter().map(|s|s.cut).collect::<Vec<_>>()})},
            Err(e)=>json!({"gib":gib,"fits":false,"error":format!("{e}")}),
        }
    }).collect::<Vec<_>>();
    let mut zero = c;
    zero.lookups = 0;
    let floor = PassPlan::new(native.schedule().stages.len(), zero, weight_bytes)
        .resident_bytes()
        .ok_or("floor overflow")?;
    let params = tessera_prover::ParamSet::defined(&claims.params).ok_or("defined parameters")?;
    let shapes = claims.arch.weight_shapes()?;
    Ok(
        json!({"vocab":claims.arch.vocab,"greedy_site":leaf.sites().len()-1,"positions":n,"policy":claims.gadget_policy.max_log2_rows,"cells":cells(c),"bytes_per_cell":PassPlan::BYTES_PER_CELL,"weight_bytes":weight_bytes,"resident_bytes":plan.resident_bytes(),"zero_lookup_base_floor":floor,"tensors":tensors,"instances":instances,"split_values":leaf.split_len(),"stage_pool_claims":native.schedule().pool.len(),"stages":stages,"budgets":budgets,"advice":{"rows":layout.rows(),"log2_k":claims.layout.advice.log2_k,"allocated_rows":layout.allocated_rows(),"placements":layout.tensors().iter().map(|p|json!({"first_row":p.first_row(),"vars":p.vars(),"nonzero":ranges(p.nonzero())})).collect::<Vec<_>>(),"segments":layout.segments().iter().map(|s|json!({"rows":[s.rows.start,s.rows.end],"nonzero":ranges(&s.nonzero)})).collect::<Vec<_>>()},"weight_inputs":{"layers":claims.arch.n_layers,"log2_k":claims.layout.weights.log2_k,"log2_inv_rate":claims.layout.weights.log2_inv_rate,"row_block":claims.layout.advice.row_block,"queries":params.weight_queries,"matrices":shapes.iter().map(|s|json!({"inputs":s.inputs,"out":s.out})).collect::<Vec<_>>()}}),
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let parent = root.join("../EXP-0029-uniform-requant");
    let source: ClaimList = envelope::open(
        &fs::read(parent.join("real-1b-claims.tcl"))?,
        &schema::CLAIM_LIST,
        DecodeLimits::MESSAGE,
    )?;
    let control = PassPlan::new(
        1,
        Cells {
            tensors: 1,
            lookups: 2,
            table_rows: 3,
            advice: 4,
        },
        1024,
    );
    assert_eq!(control.resident_bytes(), Some(1664));
    assert!(control.check(MemoryBudget { host: 1664 }).is_ok());
    assert!(control.check(MemoryBudget { host: 1663 }).is_err());
    let mut records = Vec::new();
    let tiny: IntegerModel = envelope::open(
        &fs::read(root.join("../../../spec/vectors/models/tiny-2-servable.d"))?,
        &schema::INTEGER_MODEL,
        DecodeLimits::INTEGER_MODEL,
    )?;
    let lower = lower_batch(PublicModel::of(&tiny), &[2])?;
    let bounds = analyze(&lower.graph, &lower.tables)?;
    let mut selected = select(&lower.graph, &bounds, &lower.tables, &source.gadget_policy)?;
    let original = selected.clone();
    finalize(&lower, &mut selected)?;
    assert_eq!(original, selected);
    let claims = tessera_compiler::claim_list(
        PublicModel::of(&tiny),
        &selected,
        &source.gadget_policy,
        source.layout,
        source.params.clone(),
    )?;
    let mut row = measure(&lower, &selected, &claims, 2)?;
    row["case"] = json!("tiny-control");
    records.push(row);
    let plan_text = fs::read_to_string(parent.join("results/layout-plans.tsv"))?;
    for n in [2u32, 512, 2048] {
        let lower = lower_batch(PublicModel::of_claims(&source), &[n])?;
        let ranges = analyze(&lower.graph, &lower.tables)?;
        let layers = LayerMap::new(&lower)?;
        for limit in [30u8, 31, 32] {
            let mut claims = source.clone();
            claims.gadget_policy.max_log2_rows = limit;
            let mut selected = select(&lower.graph, &ranges, &lower.tables, &claims.gadget_policy)?;
            claims.tables = selected.tables.clone();
            let baseline = LeafLayout::model(&lower, &layers, &selected, &claims);
            assert!(
                matches!(&baseline,Err(tessera_prover::leaf::LeafError::Internal{rule}) if *rule==RULE)
            );
            let mut plans = BTreeMap::new();
            for line in plan_text.lines() {
                let p: Vec<_> = line.split('\t').collect();
                if p.len() == 6 && p[0] == "real" && p[1].parse::<u8>()? == limit {
                    plans.insert(
                        p[2].parse::<u32>()?,
                        (
                            p[3].parse::<u8>()?,
                            p[4].parse::<u8>()?,
                            p[5].parse::<u8>()?,
                        ),
                    );
                }
            }
            assert_eq!(candidate::patch(&lower, &mut selected, &plans)?, 16);
            finalize(&lower, &mut selected)?;
            claims.tables = selected.tables.clone();
            let mut row = measure(&lower, &selected, &claims, n)?;
            row["case"] = json!("real");
            row["baseline"] = json!("exact-leaf-refusal");
            println!(
                "n={n} B={limit} floor={} peak={}",
                row["zero_lookup_base_floor"], row["resident_bytes"]
            );
            records.push(row);
            fs::write(
                root.join("results/counts.json"),
                serde_json::to_vec_pretty(
                    &json!({"scope":"E0 public metadata and unchanged planner; no proof or complete G-4","records":records}),
                )?,
            )?;
        }
    }
    assert_eq!(records.len(), 10);
    println!("complete public runtime records written; no witness or proof created");
    Ok(())
}
