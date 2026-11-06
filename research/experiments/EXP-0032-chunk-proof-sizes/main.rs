// SPDX-License-Identifier: Apache-2.0 OR MIT

//! EXP-0032: the exact non-zero advice rows of whole 405B `servable` requests (IDEA-0043).
//! The fixture, policy and layout are `crates/tessera-bench/tests/commitment_report.rs`'s.

use std::time::Instant;
use tessera_codec::{Ascii, Rational};
use tessera_compiler::{ModelAdviceLayout, PublicModel, analyze, claim_list, lower, select};
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

/// Llama-3.1-405B's public part in `profile`, at a residual scale of `2^{−7}`, with the shifts the
/// compiler's lowering tests give the scales of RFC-0005. Every site after a weight takes
/// `classes` shift classes, from its shift up; every other site one.
fn llama_405b(profile: Profile, classes: i8) -> (ArchSpec, PublicQuantSpec, Vec<(u32, i8)>) {
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
    // RMSNorm's e = ⌈d · 10^{−5} · 2^{14}⌉, and RFC-0023's κ.
    let eps = (u64::from(d) * (1 << 14)).div_ceil(100_000);
    let kappa = (0..=31u8)
        .rev()
        .find(|&k| (u128::from(d) << (2 * k)) / u128::from(eps) < 1 << 64)
        .expect("κ = 0 fits");
    let norm = i8::try_from(kappa).expect("small") - 3;
    // `minimal`'s residual sites read the O and down projections' unrequantized inputs.
    let sites: &[(Site, i8)] = match profile {
        Profile::Servable => &[
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
        ],
        Profile::Minimal => &[
            (Site::AttnNorm, norm),
            (Site::RopeQ, 22),
            (Site::RopeK, 22),
            (Site::V, 8),
            (Site::AttnResidual, 37),
            (Site::MlpNorm, norm),
            (Site::Silu, 20),
            (Site::MlpResidual, 17),
        ],
    };
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
            let count = if after_a_weight { classes } else { 1 };
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

fn main() {
    let (arch, quant, silu) = llama_405b(Profile::Servable, 2);
    let model = PublicModel {
        arch: &arch,
        profile: Profile::Servable,
        quant: &quant,
        silu_in_exponents: &silu,
    };
    // As the report test: selection on a 16-position lowering, the layout on the full one.
    let lowering = lower(model, 16).expect("405B lowers");
    let ranges = analyze(&lowering.graph, &lowering.tables).expect("ranges");
    let policy = GadgetPolicy {
        max_log2_rows: 32,
        ..GadgetPolicy::default()
    };
    let selection =
        select(&lowering.graph, &ranges, &lowering.tables, &policy).expect("B_max = 32 selects");
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
        ParamSetId(Ascii::new("ni-100/1").unwrap()),
    )
    .expect("claims");
    let control = [
        (16384u32, 953_580u64),
        (24576, 1_819_563),
        (32768, 2_889_129),
    ];
    println!("[");
    let sizes = [
        4096u32, 8192, 12288, 16384, 20480, 24576, 32768, 49152, 65536,
    ];
    for (k, &n) in sizes.iter().enumerate() {
        let t = Instant::now();
        let full = lower(model, n).expect("lowers");
        let advice = ModelAdviceLayout::new(&full, &selection, &claims).expect("layout");
        let rows = advice.layout().rows();
        if let Some(&(_, want)) = control.iter().find(|(m, _)| *m == n) {
            assert_eq!(rows, want, "control row count at n = {n} (RFC-0057)");
        }
        let sep = if k + 1 < sizes.len() { "," } else { "" };
        println!(
            "  {{\"n\": {n}, \"rows\": {rows}, \"allocated_rows\": {}, \"variables\": {}, \"seconds\": {:.1}}}{sep}",
            advice.layout().allocated_rows(),
            advice.layout().variables(),
            t.elapsed().as_secs_f64()
        );
        eprintln!("n = {n}: {rows} rows in {:.1} s", t.elapsed().as_secs_f64());
    }
    println!("]");
}
