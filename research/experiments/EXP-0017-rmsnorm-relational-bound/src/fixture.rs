// SPDX-License-Identifier: Apache-2.0 OR MIT

// Adapted from crates/tessera-bench/tests/commitment_report.rs at main709c37c.
// Original SHA256 62e2c82a604e5ff7726f64036dc5335bef8d230e75a6ef3c94de034b17bc301c.
// Changes: parameterize d/layers and kappa16/cap31 controls; 8B heads/d_ff.
use tessera_codec::Rational;
use tessera_ir::{ArchFamily, ArchSpec, Profile, PublicQuantSpec, QuantPoint, RopeSpec, Site};

const ROPE: RopeSpec = RopeSpec {
    theta: 500_000,
    factor: 8,
    low_freq_factor: 1,
    high_freq_factor: 4,
    original_max: 8192,
    bits: 14,
};

/// Llama-3.1-405B's public part in `profile`, at a residual scale of `2^{−7}`, with the shifts the
/// compiler's lowering tests give the scales of RFC-0005. Every site after a weight takes
/// `classes` shift classes, from its shift up; every other site one.
pub(super) fn fixture(
    profile: Profile,
    classes: i8,
    d: u32,
    layers: u32,
    mode: u8,
) -> (ArchSpec, PublicQuantSpec, Vec<(u32, i8)>) {
    let arch = ArchSpec {
        family: ArchFamily::LlamaV3,
        n_layers: layers,
        d_model: d,
        n_heads: d / 128,
        n_kv_heads: 8,
        d_head: 128,
        d_ff: if d == 4096 { 14_336 } else { 53_248 },
        vocab: 128_256,
        max_context: 131_072,
        rope: ROPE,
        rms_eps: Rational::new(1, 100_000).expect("a rational"),
    };
    // RMSNorm's e = ⌈d · 10^{−5} · 2^{14}⌉, and RFC-0023's κ.
    let eps = if mode == 31 {
        u64::from(d)
    } else {
        (u64::from(d) * (1 << 14)).div_ceil(100_000)
    };
    let kappa = (0..=31u8)
        .rev()
        .find(|&k| (u128::from(d) << (2 * k)) / u128::from(eps) < 1 << 64)
        .expect("κ = 0 fits");
    let kappa = if mode == 16 { 16 } else { kappa };
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
    for layer in 0..layers {
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
        score_scales: (0..layers)
            .map(|l| (l, Rational::new(1, 1 << 14).expect("a scale")))
            .collect(),
        shift_class_sets,
        affine_sites: Vec::new(),
        rope: ROPE,
        logits_bits: 28,
        logits_exponent: -11,
    };
    let silu = (0..layers).map(|l| (l, -6)).collect();
    (arch, quant, silu)
}
