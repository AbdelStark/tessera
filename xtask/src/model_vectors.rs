// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The `IntegerModel` vector (03 §2, `spec/vectors/model/`): a small model sealed in its
//! envelope, with its value in the language-neutral JSON of the codec vectors. The Python writer
//! and reader of `tessera-quant` check it in both directions: decoding the hex gives the value,
//! and encoding the value gives the hex.

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

/// A model of one layer of width 4, every field holding a value distinct from its neighbours'.
pub(crate) fn tiny() -> IntegerModel {
    let rope = RopeSpec {
        theta: 500_000,
        factor: 8,
        low_freq_factor: 1,
        high_freq_factor: 4,
        original_max: 8192,
        bits: 14,
    };
    let layer = |site| QuantPoint::Layer { layer: 0, site };
    let weight = |ty, rows: u32, cols: u32, first: i8| WeightTensor {
        ty,
        layer: 0,
        rows,
        cols,
        data: (0..rows * cols)
            .map(|i| {
                first
                    .wrapping_add(i8::try_from(i % 64).unwrap_or(0))
                    .clamp(-127, 127)
            })
            .collect(),
    };
    IntegerModel {
        arch: ArchSpec {
            family: ArchFamily::LlamaV3,
            n_layers: 1,
            d_model: 4,
            n_heads: 2,
            n_kv_heads: 1,
            d_head: 2,
            d_ff: 8,
            vocab: 6,
            max_context: 16,
            rope,
            rms_eps: rational(1, 100_000),
        },
        profile: Profile::Servable,
        quant: QuantSpec {
            public: PublicQuantSpec {
                act_exponents: vec![
                    (QuantPoint::Embed, -3),
                    (layer(Site::AttnNorm), -5),
                    (layer(Site::V), -6),
                    (QuantPoint::FinalNorm, -4),
                ],
                residual_exponent: -10,
                residual_bits: 20,
                rsqrt_kappa: 16,
                rms_eps_int: 3,
                exp_sigma: 16,
                score_scales: vec![(0, rational(3, 1024))],
                shift_class_sets: vec![
                    (layer(Site::AttnNorm), vec![12]),
                    (layer(Site::V), vec![7, 9]),
                ],
                rope,
                logits_bits: 28,
            },
            channels: vec![ChannelParams {
                ty: WeightType::V,
                layer: 0,
                class: vec![1, 0],
                multiplier: Vec::new(),
            }],
        },
        weights: vec![
            weight(WeightType::Embed, 4, 6, -12),
            weight(WeightType::V, 2, 4, 100),
        ],
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
            libraries: Map::from_entries(
                [("torch", "2.14.1"), ("numpy", "2.5.4")]
                    .into_iter()
                    .map(|(name, version)| (ascii(name), ascii(version)))
                    .collect(),
            )
            .expect("distinct names"),
        },
    }
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
            "family": unit("LlamaV3"),
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
                "rope": rope(&q.rope),
                "logits_bits": number(q.logits_bits),
            },
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
    let model = tiny();
    let sealed = envelope::seal(&schema::INTEGER_MODEL, &model);
    json!({
        "description": "An IntegerModel (03 §2, schema 0x0101), written by `cargo xtask gen-vectors` from the Rust types of tessera-ir: decoding `hex` gives `value`, and encoding `value` gives `hex`. Values follow spec/vectors/codec/README.md 'Values'.",
        "schema": schema::INTEGER_MODEL.id.0,
        "version": schema::INTEGER_MODEL.version,
        "type": "IntegerModel",
        "limits": "integer-model",
        "hex": sealed.iter().map(|b| format!("{b:02x}")).collect::<String>(),
        "value": value(&model),
    })
}
