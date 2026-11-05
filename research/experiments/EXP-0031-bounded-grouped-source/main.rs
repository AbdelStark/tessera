// SPDX-License-Identifier: Apache-2.0 OR MIT

//! EXP-0031 partial source harness. Merge/audit/mutation gates are not yet run.
#![forbid(unsafe_code)]

mod adapter;
mod fixture;
mod memory;
mod merge_probe;

use serde_json::json;
use std::path::PathBuf;
use tessera_field::{Goldilocks as G, PrimeField};
use tessera_logup::Tuples;
use tessera_protocol::LeafGroups;

fn column_digest(columns: &adapter::Columns) -> String {
    let mut hash = blake3::Hasher::new();
    for (table, keys, values) in columns {
        hash.update(&table.0.to_le_bytes());
        hash.update(&(keys.len() as u64).to_le_bytes());
        hash.update(&[u8::from(values.is_some())]);
        for key in keys.iter() {
            hash.update(&key.to_canonical_u64().to_le_bytes());
        }
        if let Some(values) = values {
            for value in values.iter() {
                hash.update(&value.to_canonical_u64().to_le_bytes());
            }
        }
    }
    hash.finalize().to_hex().to_string()
}
use tessera_prover::leaf::grouped::{forest_share, group_columns_from};
use tessera_prover::leaf::{GroupedLayout, StageForest};

fn main() {
    let mut args = std::env::args().skip(1);
    let mode = args.next().expect("explicit partial stage");
    let out = PathBuf::from(args.next().expect("public result path"));
    assert!(args.next().is_none(), "no extra arguments");
    if mode == "merge-only" {
        let result = merge_probe::run();
        std::fs::write(
            out,
            serde_json::to_vec(&result).expect("public merge records"),
        )
        .expect("merge result file");
        println!("merge prototype complete; full experiment remains unjudged");
        return;
    }
    assert_eq!(mode, "source-only", "explicit partial source mode");
    assert_eq!(adapter::insert(0, 1, |_| Some(0)), 0);
    assert_eq!(adapter::insert(0, 1, |_| Some(1)), 1);
    let hand = [3, 5];
    assert_eq!(hand[adapter::insert(0, 1, |_| Some(1)) as usize], 5);
    assert_eq!(memory::add(u64::MAX, 1), Err(memory::Refusal::Memory));
    assert_eq!(memory::mul(u64::MAX, 2), Err(memory::Refusal::Memory));
    assert_eq!(
        memory::array::<u8>(isize::MAX as u64 + 1),
        Err(memory::Refusal::Memory)
    );
    let keys: Vec<G> = (0..16).map(G::from_u64_reduced).collect();
    let vals: Vec<G> = (0..16).map(|x| G::from_u64_reduced(2 * x + 3)).collect();
    let mut cases = Vec::new();
    let mut refusals = Vec::new();
    for id in 0..fixture::LENGTHS.len() {
        let f = fixture::Fixture::new(id);
        let mut resident = None;
        let resident_allocation = allocation_counter::measure(|| resident = Some(f.materialize()));
        let resident = resident.expect("materialized reference source");
        assert!(
            resident_allocation.bytes_current >= 0,
            "baseline source retained allocation"
        );
        let views = fixture::views(&resident);
        let baseline_payload: u64 = resident
            .iter()
            .flatten()
            .flatten()
            .map(|(_, v)| (v.len() * size_of::<i64>()) as u64)
            .sum();
        for height in [1, 2] {
            let geometry = fixture::ForestGeometry::new(&f, height);
            let mut forest = None;
            let forest_allocation =
                allocation_counter::measure(|| forest = Some(geometry.materialize(&f)));
            let forest = forest.expect("materialized reference forest");
            assert!(
                forest_allocation.bytes_current >= 0,
                "baseline forest retained allocation"
            );
            let baseline_payload = baseline_payload + (forest.len() * size_of::<G>()) as u64;
            let reference_forest =
                StageForest::new(&forest, height, geometry.queries, geometry.layers);
            for (gt, gl) in fixture::GROUPS {
                let spec = LeafGroups {
                    token_bits: gt,
                    layer_bits: gl,
                };
                let admissible =
                    u32::from(gt) <= fixture::bits(f.tokens) && u32::from(gl) <= f.layers.vars;
                let grouped = match GroupedLayout::new(&f.stage, &f.layers, spec) {
                    Ok(g) => {
                        assert!(admissible, "impossible synthetic split accepted");
                        g
                    }
                    Err(e) => {
                        assert!(!admissible, "registered legal source shape refused");
                        refusals.push(json!({"fixture":id,"height":height,"gt":gt,"gl":gl,"error":format!("{e:?}")}));
                        continue;
                    }
                };
                let mut groups = Vec::with_capacity(grouped.count());
                let mut earlier_bound = 0;
                let mut earlier_actual = 0;
                let mut retained_records = Vec::with_capacity(grouped.count());
                let mut retained_forest_records = Vec::with_capacity(grouped.count());
                for h in 0..grouped.count() {
                    let layout = grouped.group(h).expect("group descriptor");
                    let reference =
                        group_columns_from(&f.stage, &grouped, h, &layout, &views, |table| {
                            Ok(Tuples::new(&keys, (table.0 == 2).then_some(&vals[..])))
                        })
                        .expect("resident group columns");
                    let expected_forest = forest_share(
                        &reference_forest,
                        u32::from(gt),
                        u32::from(gl),
                        layout.tokens(),
                        layout.layers(),
                    )
                    .expect("resident forest share");
                    let bill = memory::SourceBill::new(
                        &layout,
                        &geometry,
                        u32::from(gt),
                        u32::from(gl),
                        earlier_bound,
                    )
                    .expect("public bill");
                    bill.check(bill.peak).expect("exact prototype allowance");
                    let rejected = allocation_counter::measure(|| {
                        assert_eq!(bill.check(bill.peak - 1), Err(memory::Refusal::Memory))
                    });
                    assert_eq!(rejected.count_total, 0, "budget refuses without allocation");
                    let mut candidate = None;
                    let measured = allocation_counter::measure(|| {
                        candidate = Some(
                            adapter::group(&f, &grouped, h, &layout, &geometry)
                                .expect("candidate source"),
                        )
                    });
                    let candidate = candidate.expect("source output retained");
                    let observed = earlier_actual + measured.bytes_max;
                    assert!(observed <= bill.peak, "source peak within public bill");
                    assert_eq!(candidate.columns.len(), reference.len(), "instance count");
                    for (got, want) in candidate.columns.iter().zip(&reference) {
                        assert_eq!(got.0, want.0, "public table order");
                        assert!(got.1[..] == want.1[..], "canonical keys");
                        assert!(
                            match (&got.2, &want.2) {
                                (None, None) => true,
                                (Some(a), Some(b)) => a[..] == b[..],
                                _ => false,
                            },
                            "canonical values/padding"
                        );
                    }
                    assert!(
                        candidate.forest[..] == expected_forest[..],
                        "canonical forest share"
                    );
                    for record in &candidate.indices {
                        let block = layout
                            .instances()
                            .iter()
                            .find(|i| i.table.0 == record.table)
                            .and_then(|i| {
                                i.blocks.iter().find(|b| {
                                    b.site == record.site
                                        && b.layer == record.layer
                                        && b.lookup == record.lookup
                                })
                            })
                            .expect("record block");
                        assert_eq!(
                            Some(record.original),
                            grouped.source(h, block, record.position),
                            "original source index"
                        );
                    }
                    let metadata = layout.instances().iter().map(|i| json!({"table":i.table.0,"height":i.height,"entries":i.entries,"blocks":i.blocks.iter().map(|b| json!({"site":b.site,"layer":b.layer,"lookup":b.lookup,"vars":b.vars,"offset":b.offset})).collect::<Vec<_>>()})).collect::<Vec<_>>();
                    groups.push(json!({"h":h,"bill":bill,"observed_peak":observed,"baseline_source_payload":baseline_payload,"instances":metadata,"column_digest":column_digest(&candidate.columns),"indices":candidate.indices,"forest_indices":candidate.forest_indices}));
                    // Public serialization/report allocations are outside the source counter.
                    // Keep source index records too: they are charged as earlier owners.
                    earlier_actual += (candidate.indices.capacity()
                        * size_of::<adapter::IndexRecord>()
                        + candidate.forest_indices.capacity() * size_of::<adapter::ForestRecord>())
                        as u64;
                    earlier_bound = memory::add(earlier_bound, bill.records)
                        .expect("public earlier-record bound");
                    retained_records.push(candidate.indices);
                    retained_forest_records.push(candidate.forest_indices);
                }
                cases.push(json!({"fixture":id,"lengths":f.lengths,"layers":f.layers.count,"slot":f.slot,"tokens":f.tokens,"gt":gt,"gl":gl,"forest":{"height":geometry.height,"queries":geometry.queries,"layers":geometry.layers,"vars":geometry.vars},"baseline_source_construction_peak":resident_allocation.bytes_max.max(resident_allocation.bytes_current as u64+forest_allocation.bytes_max),"baseline_source_retained":resident_allocation.bytes_current+forest_allocation.bytes_current,"groups":groups}));
            }
        }
    }
    let source_payload_reduction_cases = cases
        .iter()
        .filter(|c| {
            let groups = c["groups"].as_array().expect("public group records");
            groups
                .iter()
                .map(|g| g["bill"]["source_payload"].as_u64().expect("source bytes"))
                .max()
                .expect("group payload")
                < groups[0]["baseline_source_payload"]
                    .as_u64()
                    .expect("baseline payload")
        })
        .count();
    assert!(
        source_payload_reduction_cases > 0,
        "at least one strict source payload reduction"
    );
    let result = json!({"stage":"source-prototype-only","verdict":"not judged","registered_criteria_complete":false,"production_admission":false,"real_model_acceptance":false,"seed":fixture::SEED,"cases":cases,"refusals":refusals,"source_payload_reduction_cases":source_payload_reduction_cases,"measurement_scope":"source allocations and earlier index records; public layouts, resident oracle/lent views, serialized reports and future merge/phase owners remain separate; baseline source constructors measured separately"});
    std::fs::write(
        out,
        serde_json::to_vec(&result).expect("public result encoding"),
    )
    .expect("public result file");
    println!("source prototype complete; full experiment remains unjudged");
}
