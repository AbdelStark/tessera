// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The `IntegerModel` vectors (03 §2, `spec/vectors/model/`): two one-layer models sealed in
//! their envelopes, with their values in the language-neutral JSON of the codec vectors. The
//! Python writer and reader of `tessera-quant` check them in both directions: decoding the hex
//! gives the value, and encoding the value gives the hex.
//!
//! Between them the models hold every variant of every enum `IntegerModel` uses, both profiles,
//! both scale policies and a map of several entries, so that each tag and each canonical order
//! is pinned. Their fields hold distinct values; they pin encodings, not a calibrated model.

use serde_json::{Value, json};
use tessera_codec::{Ascii, Map, Rational, envelope, schema};
use tessera_ir::{
    ArchFamily, ArchSpec, CalibrationIdentity, ChannelParams, IntegerModel, Profile, Provenance,
    PublicQuantSpec, QuantPoint, QuantSpec, RopeSpec, Site, WeightTensor, WeightType,
};

fn ascii(text: &str) -> Ascii {
    Ascii::new(text).expect("ASCII")
}

fn rational(num: i64, den: u64) -> Rational {
    Rational::new(num, den).expect("a non-zero denominator")
}

const ROPE: RopeSpec = RopeSpec {
    theta: 500_000,
    factor: 8,
    low_freq_factor: 1,
    high_freq_factor: 4,
    original_max: 8192,
    bits: 14,
};

/// One layer of width 4: two query heads and one key and value head of width 2.
fn arch() -> ArchSpec {
    ArchSpec {
        family: ArchFamily::LlamaV3,
        n_layers: 1,
        d_model: 4,
        n_heads: 2,
        n_kv_heads: 1,
        d_head: 2,
        d_ff: 8,
        vocab: 6,
        max_context: 16,
        rope: ROPE,
        rms_eps: rational(1, 100_000),
    }
}

/// Every weight of `arch()`, ordered by `(ty, layer)`, with entries from `seed` on.
fn weights(seed: i8) -> Vec<WeightTensor> {
    let (d, q, kv, ff, vocab) = (4, 4, 2, 8, 6);
    [
        // The transposed embedding: its rows are its output channels (03 §2).
        (WeightType::Embed, d, vocab),
        (WeightType::Q, q, d),
        (WeightType::K, kv, d),
        (WeightType::V, kv, d),
        (WeightType::O, d, q),
        (WeightType::Gate, ff, d),
        (WeightType::Up, ff, d),
        (WeightType::Down, d, ff),
        (WeightType::LmHead, vocab, d),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, (ty, rows, cols))| {
        let first = i32::from(seed) + 37 * i32::try_from(index).unwrap_or(0);
        WeightTensor {
            ty,
            layer: 0,
            rows,
            cols,
            data: (0..rows * cols)
                .map(|i| {
                    let x = (first + 29 * i32::try_from(i).unwrap_or(0)).rem_euclid(255) - 127;
                    i8::try_from(x).unwrap_or(0)
                })
                .collect(),
        }
    })
    .collect()
}

fn layer(site: Site) -> QuantPoint {
    QuantPoint::Layer { layer: 0, site }
}

/// The servable profile in `pot` mode: every site of a layer, two of them with two classes.
fn servable() -> IntegerModel {
    let int8 = [
        Site::AttnNorm,
        Site::RopeQ,
        Site::RopeK,
        Site::V,
        Site::AttnOut,
        Site::MlpNorm,
        Site::Silu,
        Site::Up,
        Site::Product,
    ];
    let mut act_exponents = vec![(QuantPoint::Embed, -3)];
    act_exponents.extend(int8.iter().zip(-12..).map(|(&site, e)| (layer(site), e)));
    act_exponents.push((QuantPoint::FinalNorm, -4));
    let sites = [
        (Site::AttnNorm, vec![12]),
        (Site::RopeQ, vec![13]),
        (Site::RopeK, vec![14]),
        (Site::V, vec![7, 9]),
        (Site::AttnOut, vec![15]),
        (Site::AttnResidual, vec![16]),
        (Site::MlpNorm, vec![17]),
        (Site::Silu, vec![5, 6]),
        (Site::Up, vec![18]),
        (Site::Product, vec![19]),
        (Site::MlpResidual, vec![20]),
    ];
    let mut shift_class_sets: Vec<(QuantPoint, Vec<i8>)> = sites
        .into_iter()
        .map(|(site, shifts)| (layer(site), shifts))
        .collect();
    shift_class_sets.push((QuantPoint::FinalNorm, vec![11]));
    IntegerModel {
        arch: arch(),
        profile: Profile::Servable,
        quant: QuantSpec {
            public: PublicQuantSpec {
                act_exponents,
                residual_exponent: -10,
                residual_bits: 20,
                rsqrt_kappa: 16,
                rms_eps_int: 3,
                exp_sigma: 16,
                score_scales: vec![(0, rational(3, 1024))],
                shift_class_sets,
                affine_sites: Vec::new(),
                rope: ROPE,
                logits_bits: 28,
                logits_exponent: -11,
            },
            silu_in_exponents: vec![(0, -9)],
            channels: vec![
                ChannelParams {
                    ty: WeightType::V,
                    layer: 0,
                    class: vec![1, 0],
                    multiplier: Vec::new(),
                },
                ChannelParams {
                    ty: WeightType::Gate,
                    layer: 0,
                    class: vec![0, 1, 1, 0, 1, 0, 0, 1],
                    multiplier: Vec::new(),
                },
            ],
        },
        weights: weights(-12),
        provenance: Provenance {
            tessera_quant: ascii("0.1.0"),
            checkpoint: vec![(ascii("model.safetensors"), [0xA7; 32])],
            calibration: CalibrationIdentity {
                dataset: ascii("allenai/c4"),
                config: ascii("en"),
                split: ascii("validation"),
                revision: ascii("1588ec454efa1a09f29cd18ddd04fe05fc8653a2"),
                files: vec![ascii("en/c4-validation.00000-of-00008.json.gz")],
                seed: 0x5EED,
                sequences: 512,
                held_out: 64,
                tokens_per_sequence: 2048,
                selection: [0x3C; 32],
            },
            statistics: [0x11; 32],
            alpha: rational(1, 2),
            libraries: libraries(&[("torch", "2.14.1"), ("numpy", "2.5.4")]),
        },
    }
}

/// The minimal profile with two affine sites (RFC-0025): the sites the servable profile adds are
/// absent, and the V and down projections carry multipliers, with no classes.
fn minimal() -> IntegerModel {
    let int8 = [
        Site::AttnNorm,
        Site::RopeQ,
        Site::RopeK,
        Site::V,
        Site::MlpNorm,
        Site::Silu,
    ];
    let mut act_exponents = vec![(QuantPoint::Embed, -2)];
    act_exponents.extend(int8.iter().zip(-9..).map(|(&site, e)| (layer(site), e)));
    act_exponents.push((QuantPoint::FinalNorm, -5));
    let sites = [
        Site::AttnNorm,
        Site::RopeQ,
        Site::RopeK,
        Site::V,
        Site::AttnResidual,
        Site::MlpNorm,
        Site::Silu,
        Site::MlpResidual,
    ];
    let affine = |site| matches!(site, Site::V | Site::MlpResidual);
    let shifts = sites.into_iter().zip(21..);
    let mut shift_class_sets: Vec<(QuantPoint, Vec<i8>)> = shifts
        .clone()
        .filter(|&(site, _)| !affine(site))
        .map(|(site, r)| (layer(site), vec![r]))
        .collect();
    shift_class_sets.push((QuantPoint::FinalNorm, vec![63]));
    let affine_sites = shifts
        .filter(|&(site, _)| affine(site))
        .map(|(site, r)| (layer(site), r))
        .collect();
    IntegerModel {
        arch: arch(),
        profile: Profile::Minimal,
        quant: QuantSpec {
            public: PublicQuantSpec {
                act_exponents,
                residual_exponent: -11,
                residual_bits: 20,
                rsqrt_kappa: 16,
                rms_eps_int: 1,
                exp_sigma: 16,
                score_scales: vec![(0, rational(-5, 4096))],
                shift_class_sets,
                affine_sites,
                rope: ROPE,
                logits_bits: 28,
                logits_exponent: 7,
            },
            silu_in_exponents: vec![(0, -8)],
            channels: vec![
                ChannelParams {
                    ty: WeightType::V,
                    layer: 0,
                    class: Vec::new(),
                    multiplier: vec![40_000, 33_001],
                },
                ChannelParams {
                    ty: WeightType::Down,
                    layer: 0,
                    class: Vec::new(),
                    multiplier: vec![65_535, 32_768, 1, 50_000],
                },
            ],
        },
        weights: weights(101),
        provenance: Provenance {
            tessera_quant: ascii("0.1.0"),
            checkpoint: vec![
                (ascii("model-00001-of-00002.safetensors"), [0x01; 32]),
                (ascii("model-00002-of-00002.safetensors"), [0xFE; 32]),
            ],
            calibration: CalibrationIdentity {
                dataset: ascii("allenai/c4"),
                config: ascii("en"),
                split: ascii("validation"),
                revision: ascii("1588ec454efa1a09f29cd18ddd04fe05fc8653a2"),
                files: vec![
                    ascii("en/c4-validation.00000-of-00008.json.gz"),
                    ascii("en/c4-validation.00001-of-00008.json.gz"),
                ],
                seed: u64::MAX,
                sequences: 128,
                held_out: 16,
                tokens_per_sequence: 512,
                selection: [0xC3; 32],
            },
            statistics: [0x22; 32],
            alpha: rational(3, 4),
            // Sorted by their encoding, a length prefix before the bytes: by length first.
            libraries: libraries(&[
                ("transformers", "5.20.1"),
                ("python", "3.13.11"),
                ("torch", "2.14.1"),
                ("safetensors", "0.7.0"),
                ("numpy", "2.5.4"),
                ("datasets", "4.4.1"),
                ("blake3", "1.0.11"),
            ]),
        },
    }
}

fn libraries(entries: &[(&str, &str)]) -> Map<Ascii, Ascii> {
    Map::from_entries(
        entries
            .iter()
            .map(|&(name, version)| (ascii(name), ascii(version)))
            .collect(),
    )
    .expect("distinct names")
}

/// The vectors' models, by name.
pub(crate) fn models() -> [(&'static str, IntegerModel); 2] {
    [("servable-pot", servable()), ("minimal-affine", minimal())]
}

fn number(n: impl ToString) -> Value {
    Value::String(n.to_string())
}

fn bytes(data: &[u8]) -> Value {
    Value::Array(data.iter().map(|&b| number(b)).collect())
}

fn ratio(r: &Rational) -> Value {
    json!({ "num": number(r.num()), "den": number(r.den()) })
}

fn text(a: &Ascii) -> Value {
    Value::String(a.as_str().to_owned())
}

/// A unit variant: `{"Variant": null}`.
fn unit(name: &str) -> Value {
    json!({ name: null })
}

fn weight_type(ty: WeightType) -> Value {
    unit(&format!("{ty:?}"))
}

fn point(p: &QuantPoint) -> Value {
    match p {
        QuantPoint::Embed => unit("Embed"),
        QuantPoint::Layer { layer, site } => json!({
            "Layer": { "layer": number(layer), "site": unit(&format!("{site:?}")) }
        }),
        QuantPoint::FinalNorm => unit("FinalNorm"),
    }
}

fn rope(r: &RopeSpec) -> Value {
    json!({
        "theta": number(r.theta),
        "factor": number(r.factor),
        "low_freq_factor": number(r.low_freq_factor),
        "high_freq_factor": number(r.high_freq_factor),
        "original_max": number(r.original_max),
        "bits": number(r.bits),
    })
}

/// The model's value in the codec vectors' JSON (`spec/vectors/codec/README.md`, 'Values').
fn value(m: &IntegerModel) -> Value {
    let a = &m.arch;
    let q = &m.quant.public;
    let p = &m.provenance;
    let c = &p.calibration;
    json!({
        "arch": {
            "family": unit(match a.family {
                ArchFamily::LlamaV3 => "LlamaV3",
            }),
            "n_layers": number(a.n_layers),
            "d_model": number(a.d_model),
            "n_heads": number(a.n_heads),
            "n_kv_heads": number(a.n_kv_heads),
            "d_head": number(a.d_head),
            "d_ff": number(a.d_ff),
            "vocab": number(a.vocab),
            "max_context": number(a.max_context),
            "rope": rope(&a.rope),
            "rms_eps": ratio(&a.rms_eps),
        },
        "profile": unit(match m.profile {
            Profile::Servable => "Servable",
            Profile::Minimal => "Minimal",
        }),
        "quant": {
            "public": {
                "act_exponents": q.act_exponents.iter()
                    .map(|(point_, e)| json!([point(point_), number(e)])).collect::<Vec<_>>(),
                "residual_exponent": number(q.residual_exponent),
                "residual_bits": number(q.residual_bits),
                "rsqrt_kappa": number(q.rsqrt_kappa),
                "rms_eps_int": number(q.rms_eps_int),
                "exp_sigma": number(q.exp_sigma),
                "score_scales": q.score_scales.iter()
                    .map(|(layer, s)| json!([number(layer), ratio(s)])).collect::<Vec<_>>(),
                "shift_class_sets": q.shift_class_sets.iter()
                    .map(|(point_, shifts)| json!([
                        point(point_),
                        shifts.iter().map(|&r| number(r)).collect::<Vec<_>>(),
                    ])).collect::<Vec<_>>(),
                "affine_sites": q.affine_sites.iter()
                    .map(|(point_, r)| json!([point(point_), number(r)])).collect::<Vec<_>>(),
                "rope": rope(&q.rope),
                "logits_bits": number(q.logits_bits),
                "logits_exponent": number(q.logits_exponent),
            },
            "silu_in_exponents": m.quant.silu_in_exponents.iter()
                .map(|(layer, e)| json!([number(layer), number(e)])).collect::<Vec<_>>(),
            "channels": m.quant.channels.iter().map(|ch| json!({
                "ty": weight_type(ch.ty),
                "layer": number(ch.layer),
                "class": bytes(&ch.class),
                "multiplier": ch.multiplier.iter().map(|&x| number(x)).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
        },
        "weights": m.weights.iter().map(|w| json!({
            "ty": weight_type(w.ty),
            "layer": number(w.layer),
            "rows": number(w.rows),
            "cols": number(w.cols),
            "data": w.data.iter().map(|&x| number(x)).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "provenance": {
            "tessera_quant": text(&p.tessera_quant),
            "checkpoint": p.checkpoint.iter()
                .map(|(name, digest)| json!([text(name), bytes(digest)])).collect::<Vec<_>>(),
            "calibration": {
                "dataset": text(&c.dataset),
                "config": text(&c.config),
                "split": text(&c.split),
                "revision": text(&c.revision),
                "files": c.files.iter().map(text).collect::<Vec<_>>(),
                "seed": number(c.seed),
                "sequences": number(c.sequences),
                "held_out": number(c.held_out),
                "tokens_per_sequence": number(c.tokens_per_sequence),
                "selection": bytes(&c.selection),
            },
            "statistics": bytes(&p.statistics),
            "alpha": ratio(&p.alpha),
            "libraries": p.libraries.entries().iter()
                .map(|(name, version)| json!([text(name), text(version)])).collect::<Vec<_>>(),
        },
    })
}

/// `spec/vectors/model/integer-model.json`.
pub(crate) fn render() -> Value {
    json!({
        "description": "IntegerModel envelopes (03 §2, schema 0x0101), written by `cargo xtask gen-vectors` from the Rust types of tessera-ir: for each case, decoding `hex` under the IntegerModel limits gives `value`, and encoding `value` gives `hex`. Values follow spec/vectors/codec/README.md 'Values'.",
        "schema": schema::INTEGER_MODEL.id.0,
        "version": schema::INTEGER_MODEL.version,
        "type": "IntegerModel",
        "limits": "integer-model",
        "cases": models()
            .iter()
            .map(|(name, model)| {
                let sealed = envelope::seal(&schema::INTEGER_MODEL, model);
                json!({
                    "name": name,
                    "hex": sealed.iter().map(|b| format!("{b:02x}")).collect::<String>(),
                    "value": value(model),
                })
            })
            .collect::<Vec<_>>(),
    })
}
