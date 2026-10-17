// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The commitment report's committed example, `docs/perf/examples/commitment-report-405b.json`
//! (RFC-0006 'Outputs'): Llama-3.1-405B's shape in the default `servable` profile, lowered for 16
//! positions with the shifts of the compiler's lowering tests, and projected by the cost model at
//! 1K, 4K, 16K and 128K. The projections take the source's unit costs, memory and scenario and
//! five regenerations, as 08 §3 does.
//!
//! Every site has one shift class, so each residual site updates in one step. That is RISK-2's
//! variant with one exponent per tensor at O and down, 32,768 commitments and lookups per layer per
//! token below the quantizer's per-channel classes, which #88 reports.
//!
//! `cargo xtask lint` validates the file against its JSON Schema. This test keeps it current, up
//! to the last bits of numbers that platforms round differently. `TESSERA_WRITE_EXAMPLES=1` makes
//! it rewrite the file.

use std::fs;
use std::path::PathBuf;

use serde_json::Value;
use tessera_bench::{Counts, MemoryModel, ModelReport, Scenario, UnitCosts};
use tessera_codec::Rational;
use tessera_compiler::{CommitmentReport, PublicModel, analyze, lower, report, select};
use tessera_ir::{
    ArchFamily, ArchSpec, GadgetPolicy, Profile, PublicQuantSpec, QuantPoint, RopeSpec, Site,
};

const ROPE: RopeSpec = RopeSpec {
    theta: 500_000,
    factor: 8,
    low_freq_factor: 1,
    high_freq_factor: 4,
    original_max: 8192,
    bits: 14,
};

/// Llama-3.1-405B's public part in `servable`: every site of one class, with the shifts the
/// compiler's lowering tests give the scales of RFC-0005, at a residual scale of `2^{−7}`.
fn llama_405b() -> (ArchSpec, PublicQuantSpec, Vec<(u32, i8)>) {
    let (layers, d) = (126, 16_384);
    let arch = ArchSpec {
        family: ArchFamily::LlamaV3,
        n_layers: layers,
        d_model: d,
        n_heads: 128,
        n_kv_heads: 8,
        d_head: 128,
        d_ff: 53_248,
        vocab: 128_256,
        max_context: 131_072,
        rope: ROPE,
        rms_eps: Rational::new(1, 100_000).expect("a rational"),
    };
    // RMSNorm's e = ⌈d · 10^{−5} · 2^{14}⌉, and RFC-0023's κ.
    let eps = (u64::from(d) * (1 << 14)).div_ceil(100_000);
    let kappa = (0..=31u8)
        .rev()
        .find(|&k| (u128::from(d) << (2 * k)) / u128::from(eps) < 1 << 64)
        .expect("κ = 0 fits");
    let sites = [
        (Site::AttnNorm, i8::try_from(kappa).expect("small") - 3),
        (Site::RopeQ, 22),
        (Site::RopeK, 22),
        (Site::V, 8),
        (Site::AttnOut, 32),
        (Site::AttnResidual, 4),
        (Site::MlpNorm, i8::try_from(kappa).expect("small") - 3),
        (Site::Silu, 20),
        (Site::Up, 8),
        (Site::Product, 5),
        (Site::MlpResidual, 4),
    ];
    let mut act_exponents = vec![(QuantPoint::Embed, -6)];
    let mut shift_class_sets = Vec::new();
    for layer in 0..layers {
        for (site, r) in sites {
            let point = QuantPoint::Layer { layer, site };
            if !matches!(site, Site::AttnResidual | Site::MlpResidual) {
                act_exponents.push((point, -3));
            }
            shift_class_sets.push((point, vec![r]));
        }
    }
    act_exponents.push((QuantPoint::FinalNorm, -3));
    shift_class_sets.push((
        QuantPoint::FinalNorm,
        vec![i8::try_from(kappa).expect("small") - 3],
    ));
    let quant = PublicQuantSpec {
        act_exponents,
        residual_exponent: -7,
        residual_bits: 20,
        rsqrt_kappa: kappa,
        rms_eps_int: eps,
        exp_sigma: 16,
        score_scales: (0..layers)
            .map(|l| (l, Rational::new(1, 1 << 14).expect("a scale")))
            .collect(),
        shift_class_sets,
        rope: ROPE,
        logits_bits: 28,
    };
    let silu = (0..layers).map(|l| (l, -6)).collect();
    (arch, quant, silu)
}

fn the_report() -> Value {
    let (arch, quant, silu) = llama_405b();
    let model = PublicModel {
        arch: &arch,
        profile: Profile::Servable,
        quant: &quant,
        silu_in_exponents: &silu,
    };
    let lowering = lower(model, 16).expect("405B's shape lowers");
    let ranges = analyze(&lowering.graph, &lowering.tables).expect("ranges");
    let policy = GadgetPolicy::default();
    let selection =
        select(&lowering.graph, &ranges, &lowering.tables, &policy).expect("every gadget fits");
    let report: CommitmentReport = report(model, &lowering, &selection, &policy).expect("a report");
    let counts = report.counts();
    let number = |key: &str| counts[key].as_f64().expect("a count");
    let model_counts = Counts {
        commits_per_token: number("commits_per_token"),
        lookups_per_token: number("lookups_per_token"),
        scores_per_token_per_ctx: number("scores_per_token_per_ctx"),
        native_ops_lin: number("native_ops_lin"),
        native_ops_att_per_ctx: number("native_ops_att_per_ctx"),
        regen_full_equivalents: 5.0,
    };
    let cost_model = ModelReport::new(
        false,
        model_counts,
        UnitCosts::SOURCE,
        MemoryModel::NODE_405B,
        Scenario::SOURCE,
        &[1024, 4096, 16_384, 131_072],
    )
    .expect("the cost model prices the counts");
    let cost_model: Value =
        serde_json::from_str(&cost_model.to_json().expect("JSON")).expect("JSON");
    report.to_json(&cost_model)
}

/// Equal JSON, numbers within a relative 10^{−9}: the projections' last bits differ across
/// platforms; every count is an integer and compares exactly.
fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => match (x.as_u64(), y.as_u64()) {
            (Some(x), Some(y)) => x == y,
            _ => {
                let (x, y) = (
                    x.as_f64().unwrap_or(f64::NAN),
                    y.as_f64().unwrap_or(f64::NAN),
                );
                (x - y).abs() <= 1e-9 * x.abs().max(y.abs())
            }
        },
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(x, y)| same(x, y))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, x)| y.get(k).is_some_and(|y| same(x, y)))
        }
        _ => a == b,
    }
}

#[test]
fn the_committed_405b_example_is_the_reports() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/perf/examples/commitment-report-405b.json");
    let report = the_report();
    if std::env::var_os("TESSERA_WRITE_EXAMPLES").is_some() {
        let text = serde_json::to_string_pretty(&report).expect("JSON") + "\n";
        fs::write(&path, text).expect("the example");
    }
    let committed: Value =
        serde_json::from_str(&fs::read_to_string(&path).expect("the committed example"))
            .expect("JSON");
    assert!(
        same(&committed, &report),
        "{} is stale: rerun this test with TESSERA_WRITE_EXAMPLES=1",
        path.display()
    );
    // 405B's linear-layer count is RISK-2's, and the stages RFC-0010's.
    assert_eq!(report["lowering"]["stages"], 3);
    assert_eq!(
        report["per_layer_per_token"]["requantized"]["total"],
        260_096
    );
}
