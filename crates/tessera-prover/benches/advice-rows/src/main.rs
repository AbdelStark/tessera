// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Measure the actual canonical row visitor on a public synthetic model; no proof claim.
#![forbid(unsafe_code)]

use criterion::Criterion;
use std::{fs, hint::black_box, path::Path, time::Duration};
use tessera_codec::{Ascii, DecodeLimits, Secret, envelope, schema};
use tessera_compiler::{ModelAdviceLayout, PublicModel, analyze, claim_list, lower, select};
use tessera_ir::{GadgetPolicy, IntegerModel, ParamSetId};
use tessera_prover::advice::visit_rows;
use tessera_witness::{Engine, Resident};

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../..");
    let model: IntegerModel = envelope::open(
        &fs::read(root.join("spec/vectors/models/tiny-2-servable.d")).unwrap(),
        &schema::INTEGER_MODEL,
        DecodeLimits::INTEGER_MODEL,
    )
    .unwrap();
    let mut criterion = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2))
        .configure_from_args();
    let allocation = std::env::var_os("ADVICE_ROW_ALLOCATION").is_some();
    for n in [3u32, 8, 16] {
        let policy = GadgetPolicy {
            query_block: 4,
            ..GadgetPolicy::default()
        };
        let lower = lower(PublicModel::of(&model), n).unwrap();
        let ranges = analyze(&lower.graph, &lower.tables).unwrap();
        let selection = select(&lower.graph, &ranges, &lower.tables, &policy).unwrap();
        // A deliberately synthetic benchmark layout, never a named certifying bundle.
        let mut params = tessera_protocol::ParamSet::NI_100.layout;
        params.advice.log2_k = 8;
        params.advice.mask_coeffs = 4;
        params.advice.row_block = 2;
        params.advice.segments = 2;
        let claims = claim_list(
            PublicModel::of(&model),
            &selection,
            &policy,
            params,
            ParamSetId(Ascii::new("bench-rows/1").unwrap()),
        )
        .unwrap();
        let layout = ModelAdviceLayout::new(&lower, &selection, &claims).unwrap();
        let engine = Engine::new(&lower).unwrap();
        let tokens = Secret::new(vec![1; n as usize]);
        let witness = engine.run(&model, &tokens).unwrap();
        let source = Resident::new(&engine, &selection, &model, &witness).unwrap();
        let run = || {
            visit_rows(&lower, &layout, &source, 0, |r, values| {
                black_box((r, values));
                Ok(())
            })
            .unwrap();
        };
        if allocation {
            let measured = allocation_counter::measure(run);
            println!(
                "positions={n} peak_bytes={} allocated_bytes={} allocations={}",
                measured.bytes_max, measured.bytes_total, measured.count_total
            );
        } else {
            criterion.bench_function(&format!("canonical-ordinary-rows/{n}"), |b| b.iter(run));
        }
    }
    if !allocation {
        criterion.final_summary();
    }
}
