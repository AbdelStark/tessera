// SPDX-License-Identifier: Apache-2.0 OR MIT

//! EXP-0027 (d): the compiler's counts at Llama-3.1-405B that the batched mask opening reads.
//!
//! It lowers 405B's public part in `servable` with two shift classes at `B_max = 32` exactly as
//! `crates/tessera-bench/tests/commitment_report.rs` does, and prints, as JSON, RFC-0010's stage
//! count, the LogUp tables, the product-tree leaves, the reduction stages' tables (RFC-0055,
//! RFC-0056) and the advice layout's rows at 16,384 positions (RFC-0011). `count.py` reads it.
//!
//! ```sh
//! cargo run --release --locked --manifest-path research/experiments/EXP-0027-batched-mask-opening/schedule/Cargo.toml -- <out.json>
//! ```

#![forbid(unsafe_code)]

use std::fs;

use serde_json::{Value, json};
use tessera_codec::{Ascii, Rational};
use tessera_compiler::{
    Growth, Holding, ModelAdviceLayout, PublicModel, TableClass, analyze, claim_list, lower,
    report, select,
};
use tessera_ir::{
    AdviceLayoutParams, ArchFamily, ArchSpec, CommitmentLayout, GadgetPolicy, ParamSetId, Profile,
    PublicQuantSpec, QuantPoint, RopeSpec, Site, WeightLayoutParams,
};

const ROPE: RopeSpec = RopeSpec {
    theta: 500_000,
    factor: 8,
    low_freq_factor: 1,
    high_freq_factor: 4,
    original_max: 8192,
    bits: 14,
};

const LAYERS: u32 = 126;
const POSITIONS: u32 = 16;
const CLASSES: i8 = 2;

/// Llama-3.1-405B's public part in `servable`, as the bench test builds it: residual scale
/// `2^{−7}`, the compiler's lowering-test shifts, and `CLASSES` shift classes at every site after a
/// weight.
fn llama_405b() -> (ArchSpec, PublicQuantSpec, Vec<(u32, i8)>) {
    let d = 16_384;
    let arch = ArchSpec {
        family: ArchFamily::LlamaV3,
        n_layers: LAYERS,
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
    let eps = (u64::from(d) * (1 << 14)).div_ceil(100_000);
    let kappa = (0..=31u8)
        .rev()
        .find(|&k| (u128::from(d) << (2 * k)) / u128::from(eps) < 1 << 64)
        .expect("κ = 0 fits");
    let norm = i8::try_from(kappa).expect("small") - 3;
    let sites: &[(Site, i8)] = &[
        (Site::AttnNorm, norm),
        (Site::RopeQ, 22),
        (Site::RopeK, 22),
        (Site::V, 8),
        (Site::AttnOut, 32),
        (Site::AttnResidual, 4),
        (Site::MlpNorm, norm),
        (Site::Silu, 20),
        (Site::Up, 8),
        (Site::Product, 5),
        (Site::MlpResidual, 4),
    ];
    let mut act_exponents = vec![(QuantPoint::Embed, -6)];
    let mut shift_class_sets = Vec::new();
    for layer in 0..LAYERS {
        for &(site, r) in sites {
            let point = QuantPoint::Layer { layer, site };
            let residual = matches!(site, Site::AttnResidual | Site::MlpResidual);
            if !residual {
                act_exponents.push((point, -3));
            }
            let after_a_weight = residual
                || matches!(
                    site,
                    Site::RopeQ | Site::RopeK | Site::V | Site::Silu | Site::Up
                );
            let count = if after_a_weight { CLASSES } else { 1 };
            shift_class_sets.push((point, (r..r + count).collect()));
        }
    }
    act_exponents.push((QuantPoint::FinalNorm, -3));
    shift_class_sets.push((QuantPoint::FinalNorm, vec![norm]));
    let quant = PublicQuantSpec {
        act_exponents,
        residual_exponent: -7,
        residual_bits: 20,
        rsqrt_kappa: kappa,
        rms_eps_int: eps,
        exp_sigma: 16,
        score_scales: (0..LAYERS)
            .map(|l| (l, Rational::new(1, 1 << 14).expect("a scale")))
            .collect(),
        shift_class_sets,
        affine_sites: Vec::new(),
        rope: ROPE,
        logits_bits: 28,
        logits_exponent: -11,
    };
    let silu = (0..LAYERS).map(|l| (l, -6)).collect();
    (arch, quant, silu)
}

fn class_json(t: &TableClass) -> Value {
    let growth = match t.growth {
        Growth::Fixed => "fixed",
        Growth::Tokens => "tokens",
        Growth::Keys => "keys",
        Growth::TokensAndContext => "tokens_and_context",
    };
    let kind = match t.kind {
        Holding::Windowable => "windowable",
        Holding::Resident => "resident",
        Holding::Computed => "computed",
    };
    json!({
        "growth": growth,
        "kind": kind,
        "log2_entries": t.log2_entries,
        "count": t.count,
        "count_per_bit": t.count_per_bit,
        "bytes": t.bytes,
    })
}

fn main() {
    let out = std::env::args().nth(1).expect("an output path");
    let (arch, quant, silu) = llama_405b();
    let model = PublicModel {
        arch: &arch,
        profile: Profile::Servable,
        quant: &quant,
        silu_in_exponents: &silu,
    };
    let lowering = lower(model, POSITIONS).expect("405B's shape lowers");
    let ranges = analyze(&lowering.graph, &lowering.tables).expect("ranges");
    let policy = GadgetPolicy {
        max_log2_rows: 32,
        ..GadgetPolicy::default()
    };
    let selection =
        select(&lowering.graph, &ranges, &lowering.tables, &policy).expect("every gadget fits");
    let rep = report(model, &lowering, &selection, &policy).expect("a report");

    // The advice layout at 16,384 positions, as the bench test derives it (RFC-0011 sizes).
    let layout = CommitmentLayout {
        advice: AdviceLayoutParams {
            log2_k: 22,
            log2_inv_rate: 1,
            mask_coeffs: 200,
            row_block: 32,
            segments: 8,
        },
        weights: WeightLayoutParams {
            log2_k: 20,
            log2_inv_rate: 2,
            mask_coeffs: 65_536,
            blinding_rows: 4096,
        },
    };
    let claims = claim_list(
        model,
        &selection,
        &policy,
        layout,
        ParamSetId(Ascii::new("ni-100/1").expect("ASCII")),
    )
    .expect("a claim list");
    let full = lower(model, 16_384).expect("16,384 positions lower");
    let advice = ModelAdviceLayout::new(&full, &selection, &claims).expect("an advice layout");

    let report = json!({
        "source": "tessera-compiler report(), servable, 2 classes, B_max 32, 126 layers, 16 positions; advice layout at 16,384 positions",
        "shape": rep.shape,
        "positions": rep.positions,
        "stages": rep.stages,
        "product_tree_leaves": rep.product_tree_leaves,
        "scores": rep.scores,
        "tables": rep.tables.iter().map(|t| json!({
            "id": t.id.0,
            "kind": t.kind,
            "rows": t.rows,
            "lookups_per_token": t.lookups_per_token,
            "score_lookups": t.score_lookups,
        })).collect::<Vec<_>>(),
        "stage_tables": rep.stage_tables.iter().map(|s| json!({
            "stage": s.stage,
            "tables": s.tables.iter().map(class_json).collect::<Vec<_>>(),
            "residents": s.residents.iter().map(class_json).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "advice": {
            "rows": advice.layout().rows(),
            "allocated_rows": advice.layout().allocated_rows(),
            "variables": advice.layout().variables(),
        },
    });
    fs::write(&out, serde_json::to_string_pretty(&report).expect("JSON") + "\n").expect("written");
    println!("wrote {out}");
}
