// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The Llama-3.1 lowering (RFC-0006 'Llama lowering') against RFC-0005 'Integer semantics'.
//!
//! - **The reference.** [`reference`] computes `D`'s logits token by token from the semantics
//!   table, over the model's weights and channel classes, with no code of the lowering. Lowered
//!   graphs of seeded tiny models, in both profiles, with grouped-query attention and sites of one
//!   to four classes, run on the IR's interpreter to the same logits.
//! - **Range soundness.** On random and adversarial tokens and weights, every value of every
//!   lowered tensor lies in the range the lowering gives it.
//! - **Profiles.** The `servable` lowering keeps S1 to S3, `minimal` rescales exactly eight
//!   tensors per layer, and a graph that breaks one constraint is refused with
//!   `ERR-COMPILE-PROFILE`.
//! - **Validation.** A model that breaks one rule is refused with `ERR-QUANT-ARCH`, naming it.
//! - **Real shapes.** Llama-3.1-8B's shape lowers from its public part alone, without weights, and
//!   its scores reach `2^21`.
//! - **Conformance models.** The models of `spec/vectors/models/` (07 §5) lower, and their graphs
//!   give, bit for bit, the logits the independent Python oracle computed, as does the reference.

mod common;

use std::fs;
use std::path::Path;

use common::{Lookup, ROPE, Seeded, check_ranges, evaluate, llama, regenerate, rq, sites};
use tessera_codec::error::ErrorCode;
use tessera_codec::{Ascii, DecodeLimits, Map, Rational, envelope, schema};
use tessera_compiler::{
    CompileError, Lowering, PublicModel, Range, analyze, check_profile, lower, validate,
};
use tessera_ir::{
    ArchFamily, ArchSpec, CalibrationIdentity, ChannelParams, IntRange, IntegerModel, NodeKind, Op,
    Profile, Provenance, PublicQuantSpec, QuantPoint, QuantSpec, Site, TableId, WeightRef,
    WeightTensor, WeightType,
};
use tessera_tables::{GeneratedTable, RopeConstants, Table, TableKind};

// ----- Tiny models -------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct Shape {
    layers: u32,
    heads: u32,
    kv_heads: u32,
    d_head: u32,
    d_ff: u32,
    vocab: u32,
    context: u32,
}

const SHAPES: [Shape; 4] = [
    // Grouped-query attention, two query heads per KV head.
    Shape {
        layers: 2,
        heads: 4,
        kv_heads: 2,
        d_head: 4,
        d_ff: 24,
        vocab: 17,
        context: 16,
    },
    // One KV head for all, an odd head count.
    Shape {
        layers: 1,
        heads: 3,
        kv_heads: 1,
        d_head: 2,
        d_ff: 12,
        vocab: 9,
        context: 8,
    },
    // Multi-head attention, wider heads.
    Shape {
        layers: 1,
        heads: 2,
        kv_heads: 2,
        d_head: 8,
        d_ff: 8,
        vocab: 5,
        context: 8,
    },
    Shape {
        layers: 3,
        heads: 2,
        kv_heads: 1,
        d_head: 4,
        d_ff: 16,
        vocab: 33,
        context: 12,
    },
];

/// The weight before `site`, whose output channels its classes follow.
fn weight_of(site: Site) -> Option<WeightType> {
    match site {
        Site::RopeQ => Some(WeightType::Q),
        Site::RopeK => Some(WeightType::K),
        Site::V => Some(WeightType::V),
        Site::AttnResidual => Some(WeightType::O),
        Site::Silu => Some(WeightType::Gate),
        Site::Up => Some(WeightType::Up),
        Site::MlpResidual => Some(WeightType::Down),
        _ => None,
    }
}

fn ascii(text: &str) -> Ascii {
    Ascii::new(text).expect("ASCII")
}

fn provenance() -> Provenance {
    Provenance {
        tessera_quant: ascii("0.1.0"),
        checkpoint: vec![(ascii("model.safetensors"), [7; 32])],
        calibration: CalibrationIdentity {
            dataset: ascii("allenai/c4"),
            config: ascii("en"),
            split: ascii("validation"),
            revision: ascii("1588ec454efa1a09f29cd18ddd04fe05fc8653a2"),
            files: vec![ascii("en/c4-validation.00000-of-00008.json.gz")],
            seed: 1,
            sequences: 1,
            held_out: 1,
            tokens_per_sequence: 1,
            selection: [3; 32],
        },
        statistics: [5; 32],
        alpha: Rational::new(1, 2).expect("a rational"),
        libraries: Map::from_entries(Vec::new()).expect("no entries"),
    }
}

/// `log2` of the typical magnitude of a sum of `terms` products of uniform INT8 weights, whose
/// standard deviation is about 2^6.2, with inputs of magnitude `2^input`.
fn accumulated(input: f64, terms: u32) -> f64 {
    6.2 + input + 0.5 * f64::from(terms).log2()
}

/// A shift from a value of magnitude `2^from` to one of `2^to`, with some spread per class.
fn shift(seed: &mut Seeded, from: f64, to: f64) -> i8 {
    let jitter = f64::from(u8::try_from(seed.range(0, 2)).expect("small")) - 1.0;
    ((from - to + jitter).round() as i8).clamp(1, 63)
}

/// A model of `shape` with uniform INT8 weights, one to four classes at each site after a
/// weight, and shifts that keep its values away from both zero and saturation, so that every
/// step of the semantics shows in the logits.
fn tiny(seed: u64, profile: Profile, shape: Shape) -> IntegerModel {
    let mut rng = Seeded(seed);
    let d = shape.heads * shape.d_head;
    let arch = ArchSpec {
        family: ArchFamily::LlamaV3,
        n_layers: shape.layers,
        d_model: d,
        n_heads: shape.heads,
        n_kv_heads: shape.kv_heads,
        d_head: shape.d_head,
        d_ff: shape.d_ff,
        vocab: shape.vocab,
        max_context: shape.context,
        rope: ROPE,
        rms_eps: Rational::new(1, 100_000).expect("a rational"),
    };
    // Magnitudes, as log2: INT8 activations near 2^5, the residual near 2^12, RoPE's constants
    // 2^14, the attention output 2^{2σ} = 2^32 times a value.
    let (int8, residual) = (5.0, 12.0);
    let projection = accumulated(int8, d);
    let attention = 32.0 + int8;
    let estimate = |site: Site| -> (f64, f64) {
        match site {
            Site::AttnNorm | Site::MlpNorm => (16.0, int8),
            Site::RopeQ | Site::RopeK => (projection + 14.0, int8),
            Site::V | Site::Up => (projection, int8),
            Site::AttnOut => (attention, int8),
            Site::AttnResidual => match profile {
                Profile::Servable => (accumulated(int8, d), residual),
                Profile::Minimal => (accumulated(attention, d), residual),
            },
            Site::Silu => (projection, 5.0),
            Site::Product => (2.0 * int8, int8),
            Site::MlpResidual => match profile {
                Profile::Servable => (accumulated(int8, shape.d_ff), residual),
                Profile::Minimal => (accumulated(int8 + projection, shape.d_ff), residual),
            },
        }
    };
    let mut shift_class_sets = Vec::new();
    let mut channels = Vec::new();
    for layer in 0..shape.layers {
        for site in sites(profile) {
            let count = if weight_of(site).is_some() {
                rng.range(1, 4)
            } else {
                1
            };
            let (from, to) = estimate(site);
            let shifts: Vec<i8> = (0..count).map(|_| shift(&mut rng, from, to)).collect();
            shift_class_sets.push((QuantPoint::Layer { layer, site }, shifts));
            if let (Some(ty), true) = (weight_of(site), count > 1) {
                let rows = match ty {
                    WeightType::Q => d,
                    WeightType::K | WeightType::V => shape.kv_heads * shape.d_head,
                    WeightType::Gate | WeightType::Up => shape.d_ff,
                    _ => d,
                };
                let class = (0..rows)
                    .map(|_| u8::try_from(rng.range(0, count - 1)).expect("a class"))
                    .collect();
                channels.push(ChannelParams {
                    ty,
                    layer,
                    class,
                    multiplier: Vec::new(),
                });
            }
        }
    }
    shift_class_sets.push((QuantPoint::FinalNorm, vec![11]));
    channels.sort_by_key(|c| (c.ty, c.layer));
    let residual_exponent = -10;
    let embed_shift = i8::try_from(rng.range(0, 5)).expect("small");
    let mut act_exponents = vec![(QuantPoint::Embed, residual_exponent + embed_shift)];
    for layer in 0..shape.layers {
        for site in sites(profile) {
            if !matches!(site, Site::AttnResidual | Site::MlpResidual) {
                let exponent = if site == Site::Silu { -3 } else { -4 };
                act_exponents.push((QuantPoint::Layer { layer, site }, exponent));
            }
        }
    }
    act_exponents.push((QuantPoint::FinalNorm, -4));
    let score_scales = (0..shape.layers)
        .map(|layer| {
            let num = i64::try_from(rng.range(1, 4)).expect("small");
            (layer, Rational::new(num, 4096).expect("a scale"))
        })
        .collect();
    let silu_in_exponents = (0..shape.layers)
        .map(|layer| (layer, i8::try_from(rng.range(-5, -3)).expect("small")))
        .collect();
    let mut weights = Vec::new();
    let (q, kv) = (d, shape.kv_heads * shape.d_head);
    let mut push = |ty, layer, rows: u32, cols: u32, rng: &mut Seeded| {
        let data = (0..rows * cols)
            .map(|_| i8::try_from(rng.range(-127, 127)).expect("INT8"))
            .collect();
        weights.push(WeightTensor {
            ty,
            layer,
            rows,
            cols,
            data,
        });
    };
    push(WeightType::Embed, 0, d, shape.vocab, &mut rng);
    for (ty, rows, cols) in [
        (WeightType::Q, q, d),
        (WeightType::K, kv, d),
        (WeightType::V, kv, d),
        (WeightType::O, d, q),
        (WeightType::Gate, shape.d_ff, d),
        (WeightType::Up, shape.d_ff, d),
        (WeightType::Down, d, shape.d_ff),
    ] {
        for layer in 0..shape.layers {
            push(ty, layer, rows, cols, &mut rng);
        }
    }
    push(WeightType::LmHead, 0, shape.vocab, d, &mut rng);
    IntegerModel {
        arch,
        profile,
        quant: QuantSpec {
            public: PublicQuantSpec {
                act_exponents,
                residual_exponent,
                residual_bits: 20,
                rsqrt_kappa: 16,
                // At least d/4 + 1, so that a row of small values keeps ρ < 2^17.
                rms_eps_int: u64::from(d),
                exp_sigma: 16,
                score_scales,
                shift_class_sets,
                rope: ROPE,
                logits_bits: 28,
            },
            silu_in_exponents,
            channels,
        },
        weights,
        provenance: provenance(),
    }
}

// ----- The reference: RFC-0005 'Integer semantics', token by token ------------------------------

/// The tables the reference reads, built once per model: RoPE's constants, and each layer's `Exp`
/// and `Silu` tables with keys wide enough for any tiny model.
struct ReferenceTables {
    rope: RopeConstants,
    exp: Vec<GeneratedTable>,
    silu: Vec<GeneratedTable>,
}

fn reference_tables(model: &IntegerModel) -> ReferenceTables {
    let (arch, quant) = (&model.arch, &model.quant.public);
    let rope = RopeConstants::new(&TableKind::Rope {
        theta: arch.rope.theta,
        factor: arch.rope.factor,
        low_freq_factor: arch.rope.low_freq_factor,
        high_freq_factor: arch.rope.high_freq_factor,
        original_max: arch.rope.original_max,
        bits: arch.rope.bits,
        max_context: arch.max_context,
        d_head: u16::try_from(arch.d_head).expect("small"),
    })
    .expect("RoPE");
    let exp = quant
        .score_scales
        .iter()
        .map(|&(_, scale)| {
            let kind = TableKind::Exp {
                key_bits: 30,
                sigma: quant.exp_sigma,
                scale,
            };
            GeneratedTable::new(TableId(0), kind).expect("an Exp table")
        })
        .collect();
    let silu = (0..arch.n_layers)
        .map(|layer| {
            let point = QuantPoint::Layer {
                layer,
                site: Site::Silu,
            };
            let (_, shifts) = quant
                .shift_class_sets
                .iter()
                .find(|(p, _)| *p == point)
                .expect("the site");
            let (_, out_exp) = quant
                .act_exponents
                .iter()
                .find(|(p, _)| *p == point)
                .expect("the exponent");
            let kind = TableKind::Silu {
                b: 22,
                r_by_class: shifts.iter().map(|&r| r as u8).collect(),
                in_exp: model.quant.silu_in_exponents[layer as usize].1,
                out_exp: *out_exp,
            };
            GeneratedTable::new(TableId(0), kind).expect("a Silu table")
        })
        .collect();
    ReferenceTables { rope, exp, silu }
}

/// `D`'s logits for `tokens`, `[n][vocab]`, from RFC-0005's semantics table, position by position
/// with plain loops over the model's weights. It reads the tables' values from `tessera-tables`,
/// whose generators have their own oracle, and nothing of the lowering.
fn reference(model: &IntegerModel, tables: &ReferenceTables, tokens: &[u32]) -> Vec<Vec<i128>> {
    let arch = &model.arch;
    let quant = &model.quant.public;
    let (n, d, vocab) = (tokens.len(), arch.d_model as usize, arch.vocab as usize);
    let (heads, dh) = (arch.n_heads as usize, arch.d_head as usize);
    let group = heads / arch.n_kv_heads as usize;
    let weight = |ty: WeightType, layer: u32| {
        model
            .weights
            .iter()
            .find(|w| w.ty == ty && w.layer == layer)
            .expect("a weight")
    };
    // `W x` for a row-major `rows × cols` weight.
    let apply = |w: &WeightTensor, x: &[i128]| -> Vec<i128> {
        let cols = w.cols as usize;
        (0..w.rows as usize)
            .map(|o| {
                (0..cols)
                    .map(|i| i128::from(w.data[o * cols + i]) * x[i])
                    .sum()
            })
            .collect()
    };
    let shifts = |point: QuantPoint| -> Vec<u8> {
        let (_, set) = quant
            .shift_class_sets
            .iter()
            .find(|(p, _)| *p == point)
            .expect("a site");
        set.iter()
            .map(|&r| u8::try_from(r).expect("a shift"))
            .collect()
    };
    let exponent = |point: QuantPoint| {
        quant
            .act_exponents
            .iter()
            .find(|(p, _)| *p == point)
            .expect("an exponent")
            .1
    };
    // Channel c's class at a site after the weight (ty, layer): 0 for a site of one class.
    let class_of = |ty: WeightType, layer: u32, c: usize| -> usize {
        model
            .quant
            .channels
            .iter()
            .find(|p| p.ty == ty && p.layer == layer)
            .map_or(0, |p| usize::from(p.class[c]))
    };
    let half = 1i64 << (quant.residual_bits - 1);
    // A rescale of `x` at a site: rq per channel, with the class of its weight's channel.
    let rescale = |x: &[i128], point: QuantPoint, after: Option<WeightType>, lo: i64, hi: i64| {
        let set = shifts(point);
        let QuantPoint::Layer { layer, .. } = point else {
            return x.iter().map(|&v| rq(v, set[0], lo, hi)).collect::<Vec<_>>();
        };
        x.iter()
            .enumerate()
            .map(|(c, &v)| {
                let class = after.map_or(0, |ty| class_of(ty, layer, c));
                rq(v, set[class], lo, hi)
            })
            .collect()
    };
    // RFC-0022: res ← clamp(res + ⌊(acc + 2^{r−1}) / 2^r⌋), the sum saturated within the
    // residual's bits rather than the update, r the shift of the class of W's channel.
    let saturate = |res: &mut [i128], acc: &[i128], point: QuantPoint, ty: WeightType| {
        let set = shifts(point);
        let QuantPoint::Layer { layer, .. } = point else {
            unreachable!("a residual site is a layer's")
        };
        for (c, (x, &a)) in res.iter_mut().zip(acc).enumerate() {
            let r = set[class_of(ty, layer, c)];
            let update = (a + (1i128 << (r - 1))) >> r;
            *x = (*x + update).clamp(-i128::from(half), i128::from(half) - 1);
        }
    };
    // y = rq(res · ρ, r) with ρ = max{ρ : ρ^2 (Σ res^2 + e) ≤ d · 2^{2κ}}, by bisection: ρ = 0
    // fits, and ρ = d · 2^κ + 1 does not.
    let norm = |res: &[i128], point: QuantPoint| {
        let sum: i128 = res.iter().map(|&x| x * x).sum::<i128>() + i128::from(quant.rms_eps_int);
        let kappa = u32::from(quant.rsqrt_kappa);
        let big = (d as i128) << (2 * kappa);
        let (mut fits, mut fails) = (0i128, ((d as i128) << kappa) + 1);
        while fails - fits > 1 {
            let mid = (fits + fails) / 2;
            if mid * mid * sum <= big {
                fits = mid;
            } else {
                fails = mid;
            }
        }
        let scaled: Vec<i128> = res.iter().map(|&x| x * fits).collect();
        rescale(&scaled, point, None, -128, 127)
    };
    let rope = &tables.rope;
    // (a, b) = (x_i, x_{i + d_head/2}) of each head ↦ (a·C − b·S, a·S + b·C) at position p.
    let rotate = |x: &[i128], p: usize| -> Vec<i128> {
        let mut y = x.to_vec();
        for head in 0..x.len() / dh {
            for i in 0..dh / 2 {
                let (c, s) = rope
                    .at(p as u32, i as u32)
                    .map(|(c, s)| (i128::from(c), i128::from(s)))
                    .expect("a pair");
                let (a, b) = (x[head * dh + i], x[head * dh + i + dh / 2]);
                y[head * dh + i] = a * c - b * s;
                y[head * dh + i + dh / 2] = a * s + b * c;
            }
        }
        y
    };
    let sigma = quant.exp_sigma;

    // Embed: x0 is the token's column of the transposed embedding; res = x0 · 2^{s_e}.
    let embed = weight(WeightType::Embed, 0);
    let s_e = i32::from(exponent(QuantPoint::Embed)) - i32::from(quant.residual_exponent);
    let mut res: Vec<Vec<i128>> = tokens
        .iter()
        .map(|&t| {
            (0..d)
                .map(|c| i128::from(embed.data[c * vocab + t as usize]) << s_e)
                .collect()
        })
        .collect();
    for layer in 0..arch.n_layers {
        let point = |site| QuantPoint::Layer { layer, site };
        let exp = &tables.exp[layer as usize];
        let (mut q8, mut k8, mut v8) = (Vec::new(), Vec::new(), Vec::new());
        for (p, r) in res.iter().enumerate() {
            let y = norm(r, point(Site::AttnNorm));
            let q = rotate(&apply(weight(WeightType::Q, layer), &y), p);
            let k = rotate(&apply(weight(WeightType::K, layer), &y), p);
            let v = apply(weight(WeightType::V, layer), &y);
            q8.push(rescale(
                &q,
                point(Site::RopeQ),
                Some(WeightType::Q),
                -128,
                127,
            ));
            k8.push(rescale(
                &k,
                point(Site::RopeK),
                Some(WeightType::K),
                -128,
                127,
            ));
            v8.push(rescale(&v, point(Site::V), Some(WeightType::V), -128, 127));
        }
        for p in 0..n {
            // Each query head against its group's KV head over positions i ≤ p.
            let mut o = vec![0i128; heads * dh];
            for head in 0..heads {
                let kv = head / group;
                let scores: Vec<i128> = (0..=p)
                    .map(|i| {
                        (0..dh)
                            .map(|j| q8[p][head * dh + j] * k8[i][kv * dh + j])
                            .sum()
                    })
                    .collect();
                let max = *scores.iter().max().expect("a score");
                let e: Vec<i128> = scores
                    .iter()
                    .map(|&x| {
                        let key = i64::try_from(max - x).expect("a key");
                        i128::from(exp.lookup(key, 0).expect("an Exp key"))
                    })
                    .collect();
                let reciprocal = (1i128 << (2 * u32::from(sigma))) / e.iter().sum::<i128>();
                for j in 0..dh {
                    let pv: i128 = (0..=p).map(|i| e[i] * v8[i][kv * dh + j]).sum();
                    o[head * dh + j] = reciprocal * pv;
                }
            }
            let o = match model.profile {
                Profile::Servable => rescale(&o, point(Site::AttnOut), None, -128, 127),
                Profile::Minimal => o,
            };
            let acc = apply(weight(WeightType::O, layer), &o);
            saturate(&mut res[p], &acc, point(Site::AttnResidual), WeightType::O);
        }
        let silu = &tables.silu[layer as usize];
        for r in &mut res {
            let y = norm(r, point(Site::MlpNorm));
            let g = apply(weight(WeightType::Gate, layer), &y);
            let u = apply(weight(WeightType::Up, layer), &y);
            let a: Vec<i128> = g
                .iter()
                .enumerate()
                .map(|(c, &key)| {
                    let class = u8::try_from(class_of(WeightType::Gate, layer, c)).expect("small");
                    let key = i64::try_from(key).expect("a key");
                    i128::from(silu.lookup(key, class).expect("a SiLU key"))
                })
                .collect();
            let h = match model.profile {
                Profile::Servable => {
                    let u8 = rescale(&u, point(Site::Up), Some(WeightType::Up), -128, 127);
                    let product: Vec<i128> = a.iter().zip(&u8).map(|(a, u)| a * u).collect();
                    rescale(&product, point(Site::Product), None, -128, 127)
                }
                Profile::Minimal => a.iter().zip(&u).map(|(a, u)| a * u).collect(),
            };
            let acc = apply(weight(WeightType::Down, layer), &h);
            saturate(r, &acc, point(Site::MlpResidual), WeightType::Down);
        }
    }
    let head = weight(WeightType::LmHead, 0);
    res.iter()
        .map(|r| apply(head, &norm(r, QuantPoint::FinalNorm)))
        .collect()
}

// ----- Running a lowering ------------------------------------------------------------------------

/// Every input value of `lowering` for `tokens`: the tokens, positions `0` to `n − 1`, and the
/// model's weights and channel classes.
fn inputs(
    model: &IntegerModel,
    lowering: &Lowering,
    tokens: &[u32],
) -> Vec<(tessera_ir::TensorId, Vec<i128>)> {
    let mut inputs = vec![
        (
            lowering.tokens,
            tokens.iter().map(|&t| i128::from(t)).collect(),
        ),
        (
            lowering.positions,
            (0..tokens.len()).map(|p| p as i128).collect(),
        ),
    ];
    for &(weight, tensor) in &lowering.graph.weights {
        let w = model
            .weights
            .iter()
            .find(|w| (w.ty, w.layer) == (weight.ty, weight.layer))
            .expect("the weight");
        inputs.push((tensor, w.data.iter().map(|&x| i128::from(x)).collect()));
    }
    for &(weight, tensor) in &lowering.classes {
        let params = model
            .quant
            .channels
            .iter()
            .find(|c| (c.ty, c.layer) == (weight.ty, weight.layer))
            .expect("the classes");
        inputs.push((
            tensor,
            params.class.iter().map(|&c| i128::from(c)).collect(),
        ));
    }
    inputs
}

/// A lowering with its tables regenerated from their specifications, as a verifier would.
struct Lowered {
    lowering: Lowering,
    tables: Vec<Box<dyn Lookup>>,
}

/// The lowered graph's logits for `tokens`, `[n][vocab]`, after checking every value against its
/// tensor's range.
fn run(model: &IntegerModel, lowered: &Lowered, tokens: &[u32]) -> Vec<Vec<i128>> {
    let lowering = &lowered.lowering;
    let values = evaluate(
        &lowering.graph,
        &lowered.tables,
        inputs(model, lowering, tokens),
    );
    let ranges: Vec<Range> = lowering
        .graph
        .tensors
        .iter()
        .map(|t| Range::Bounded(t.range))
        .collect();
    check_ranges(&ranges, &values);
    values[lowering.logits.0 as usize]
        .chunks(model.arch.vocab as usize)
        .map(<[i128]>::to_vec)
        .collect()
}

fn lowered(model: &IntegerModel, n: u32) -> Lowered {
    validate(model).expect("a valid model");
    let lowering = lower(PublicModel::of(model), n).expect("the model lowers");
    let tables = regenerate(&lowering.tables);
    Lowered { lowering, tables }
}

// ----- The lowering computes D ------------------------------------------------------------------

#[test]
fn lowered_graphs_compute_the_reference_logits_in_both_profiles() {
    let mut rng = Seeded(0x10E5_0078);
    let (mut distinct, mut runs) = (0, 0);
    // Residual sites by route: the fused key for one class, two steps for several (RFC-0022).
    let (mut fused, mut two_step) = (0, 0);
    for profile in [Profile::Servable, Profile::Minimal] {
        for (k, &shape) in SHAPES.iter().enumerate() {
            for trial in 0..3u64 {
                let model = tiny(rng.next() ^ trial, profile, shape);
                fused += residual_sites(&model, |count| count == 1);
                two_step += residual_sites(&model, |count| count > 1);
                let tables = reference_tables(&model);
                for n in [1, 2, shape.context / 2 + 1, shape.context] {
                    let lowered = lowered(&model, n);
                    for _ in 0..2 {
                        let tokens: Vec<u32> = (0..n)
                            .map(|_| {
                                u32::try_from(rng.range(0, i128::from(shape.vocab) - 1))
                                    .expect("a token")
                            })
                            .collect();
                        let logits = run(&model, &lowered, &tokens);
                        assert_eq!(
                            logits,
                            reference(&model, &tables, &tokens),
                            "{profile:?}, shape {k}, trial {trial}, n = {n}"
                        );
                        runs += 1;
                        let all = logits.iter().flatten();
                        if all.clone().min() != all.max() {
                            distinct += 1;
                        }
                    }
                }
            }
        }
    }
    // The models are not degenerate: the logits of nearly every run differ across the vocabulary.
    assert!(distinct * 10 >= runs * 9, "{distinct} of {runs} runs");
    assert!(
        fused >= 10 && two_step >= 10,
        "{fused} fused, {two_step} in two steps"
    );
}

/// Each sequence of each conformance model (07 §5): the lowered graph, on the interpreter, gives
/// the logits `spec/vectors/models/generate.py` computed from RFC-0005 in Python, and so does the
/// reference. The models clamp at an INT8 requantization, the Silu table, the residual's INT20 and
/// INT16 and the exponentials, so each clamp is compared too.
#[test]
fn the_conformance_models_lower_to_the_oracles_logits() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/models");
    let text = fs::read_to_string(dir.join("models.json")).expect("the index");
    let index: serde_json::Value = serde_json::from_str(&text).expect("JSON");
    let models = index["models"].as_array().expect("models");
    assert_eq!(models.len(), 3, "every conformance model");
    let mut sequences = 0;
    for entry in models {
        let name = entry["name"].as_str().expect("a name");
        let bytes =
            fs::read(dir.join(entry["file"].as_str().expect("a file"))).expect("the artifact");
        let model: IntegerModel =
            envelope::open(&bytes, &schema::INTEGER_MODEL, DecodeLimits::INTEGER_MODEL)
                .expect("an IntegerModel");
        let tables = reference_tables(&model);
        for sequence in entry["sequences"].as_array().expect("sequences") {
            let tokens: Vec<u32> = sequence["tokens"]
                .as_array()
                .expect("tokens")
                .iter()
                .map(|t| u32::try_from(t.as_u64().expect("a token")).expect("a u32"))
                .collect();
            let expected: Vec<Vec<i128>> = sequence["logits"]
                .as_array()
                .expect("logits")
                .iter()
                .map(|row| {
                    row.as_array()
                        .expect("a row")
                        .iter()
                        .map(|x| i128::from(x.as_i64().expect("a logit")))
                        .collect()
                })
                .collect();
            let n = u32::try_from(tokens.len()).expect("a length");
            let logits = run(&model, &lowered(&model, n), &tokens);
            assert_eq!(logits, expected, "{name}, tokens {tokens:?}: the lowering");
            assert_eq!(
                reference(&model, &tables, &tokens),
                expected,
                "{name}, tokens {tokens:?}: the reference"
            );
            sequences += 1;
        }
    }
    assert_eq!(sequences, 6, "two sequences per model");
}

#[test]
fn every_lowered_value_stays_in_range_on_random_and_adversarial_inputs() {
    let mut rng = Seeded(0xADD5_0078);
    for profile in [Profile::Servable, Profile::Minimal] {
        for (k, &shape) in SHAPES.iter().enumerate() {
            let mut model = tiny(rng.next(), profile, shape);
            let tables = reference_tables(&model);
            let n = shape.context;
            let lowered = lowered(&model, n);
            for trial in 0..60 {
                // Adversarial: weights all at +127 or −127, or alternating; and one token
                // repeated, so that every score of a row ties.
                let adversarial = trial >= 40;
                if adversarial {
                    let sign = |i: usize| match trial % 3 {
                        0 => 127,
                        1 => -127,
                        _ => {
                            if i.is_multiple_of(2) {
                                127
                            } else {
                                -127
                            }
                        }
                    };
                    for w in &mut model.weights {
                        for (i, x) in w.data.iter_mut().enumerate() {
                            *x = sign(i);
                        }
                    }
                }
                let tokens: Vec<u32> = (0..n)
                    .map(|_| {
                        if adversarial {
                            0
                        } else {
                            u32::try_from(rng.range(0, i128::from(shape.vocab) - 1))
                                .expect("a token")
                        }
                    })
                    .collect();
                let logits = run(&model, &lowered, &tokens);
                assert_eq!(
                    logits,
                    reference(&model, &tables, &tokens),
                    "{profile:?}, shape {k}, trial {trial}"
                );
            }
        }
    }
}

// ----- Profiles ----------------------------------------------------------------------------------

/// The residual sites of `model` whose class count `keep` accepts.
fn residual_sites(model: &IntegerModel, keep: fn(usize) -> bool) -> usize {
    model
        .quant
        .public
        .shift_class_sets
        .iter()
        .filter(|(point, shifts)| {
            let residual = matches!(
                point,
                QuantPoint::Layer {
                    site: Site::AttnResidual | Site::MlpResidual,
                    ..
                }
            );
            residual && keep(shifts.len())
        })
        .count()
}

#[test]
fn servable_keeps_s1_to_s3_and_minimal_rescales_eight_tensors_per_layer() {
    for (profile, per_layer) in [(Profile::Servable, 11), (Profile::Minimal, 8)] {
        let shape = SHAPES[0];
        let model = tiny(5, profile, shape);
        let lowering = lowered(&model, 6).lowering;
        let graph = &lowering.graph;
        // Each layer's rescales, then the final norm's: Requant nodes and the SiLU lookup, and the
        // sum's clamp of each residual site of several classes (RFC-0022).
        let rescales = graph
            .nodes
            .iter()
            .filter(|n| matches!(n.op, Op::Requant { .. } | Op::TableMap { .. }))
            .count();
        let classed_residuals = residual_sites(&model, |count| count > 1);
        assert!(classed_residuals > 0 && residual_sites(&model, |count| count == 1) > 0);
        assert_eq!(
            rescales,
            per_layer * shape.layers as usize + 1 + classed_residuals
        );
        // S1, S2: every weight matmul's operands and accumulator, as the lowering keeps them.
        for node in &graph.nodes {
            let Op::MatMulWeight { weight } = node.op else {
                continue;
            };
            let range = |t: tessera_ir::TensorId| graph.tensors[t.0 as usize].range;
            let wide = matches!(weight.ty, WeightType::O | WeightType::Down);
            let operand = range(node.inputs[0]);
            if profile == Profile::Servable || !wide {
                assert!(operand.lo >= -128 && operand.hi <= 127, "{weight:?}");
            } else {
                assert!(operand.hi > 127, "{weight:?} takes the wide operand");
            }
            let acc = range(node.outputs[0]);
            if profile == Profile::Servable {
                assert!(acc.lo >= i128::from(i32::MIN) && acc.hi <= i128::from(i32::MAX));
            }
        }
        // S3: the forward pass's gadgets are RFC-0005's.
        assert!(graph.nodes.iter().all(|n| matches!(
            n.op,
            Op::MatMulWeight { .. }
                | Op::Add
                | Op::ConstMul { .. }
                | Op::Rotate { .. }
                | Op::Reshape
                | Op::GatherPublic { .. }
                | Op::MatMulAct { .. }
                | Op::Mul
                | Op::SumSquares { .. }
                | Op::OneHot { .. }
                | Op::Requant { .. }
                | Op::TableMap { .. }
                | Op::SoftmaxRow { .. }
                | Op::RsqrtRow { .. }
        )));
    }
}

#[test]
fn a_graph_breaking_one_profile_constraint_is_refused() {
    let model = tiny(9, Profile::Servable, SHAPES[1]);
    let lowering = lowered(&model, 4).lowering;
    let ranges = analyze(&lowering.graph, &lowering.tables).expect("analyzes");
    assert_eq!(
        check_profile(
            &lowering.graph,
            &ranges,
            &lowering.tables,
            Profile::Servable
        ),
        Ok(())
    );
    // Minimal's graph against servable's constraints: the O projection takes the attention
    // output unrescaled.
    let minimal = tiny(9, Profile::Minimal, SHAPES[1]);
    let minimal = lowered(&minimal, 4).lowering;
    let ranges = analyze(&minimal.graph, &minimal.tables).expect("analyzes");
    let error = check_profile(&minimal.graph, &ranges, &minimal.tables, Profile::Servable)
        .expect_err("S1 broken");
    assert!(matches!(
        error,
        CompileError::Profile {
            rule: "S1: a weight matmul's operands are INT8",
            ..
        }
    ));
    assert_eq!(error.code(), ErrorCode::CompileProfile);
    assert_eq!(
        check_profile(&minimal.graph, &ranges, &minimal.tables, Profile::Minimal),
        Ok(())
    );

    // S2 alone: a weight matmul of INT8 operands whose accumulator exceeds INT32.
    use tessera_ir::{CommitmentClass, GraphBuilder, IntType, TensorDecl};
    let decl = |shape: &[u32], kind, lo, hi| TensorDecl {
        shape: shape.to_vec(),
        ty: IntType::I32,
        range: IntRange { lo, hi },
        kind,
    };
    let advice = NodeKind::Committed {
        commitment: CommitmentClass::Advice,
    };
    let mut g = GraphBuilder::new();
    let wide = 140_000;
    let x = g.input(decl(&[1, wide], advice, -128, 127)).expect("x");
    let up = WeightRef {
        ty: WeightType::Up,
        layer: 0,
    };
    let w = g.weight(up, 2, wide).expect("a weight");
    g.node(
        Op::MatMulWeight { weight: up },
        &[x, w],
        vec![decl(&[1, 2], NodeKind::Virtual, 0, 0)],
    )
    .expect("a node");
    let graph = g.finish();
    let ranges = analyze(&graph, &[]).expect("analyzes");
    assert!(matches!(
        check_profile(&graph, &ranges, &[], Profile::Servable),
        Err(CompileError::Profile {
            rule: "S2: a weight matmul's accumulator fits INT32",
            ..
        })
    ));
    assert_eq!(
        check_profile(&graph, &ranges, &[], Profile::Minimal),
        Ok(())
    );

    // S3 alone: an Exp table applied elementwise.
    let exp = GeneratedTable::new(
        TableId(0),
        TableKind::Exp {
            key_bits: 8,
            sigma: 16,
            scale: Rational::new(1, 16).expect("a scale"),
        },
    )
    .expect("an Exp table");
    let mut g = GraphBuilder::new();
    let x = g.input(decl(&[4], advice, 0, 255)).expect("x");
    g.node(
        Op::TableMap { table: TableId(0) },
        &[x],
        vec![decl(
            &[4],
            NodeKind::LookupOutput { table: TableId(0) },
            0,
            0,
        )],
    )
    .expect("a node");
    let graph = g.finish();
    let specs = [exp.spec().clone()];
    let ranges = analyze(&graph, &specs).expect("analyzes");
    for profile in [Profile::Servable, Profile::Minimal] {
        assert!(matches!(
            check_profile(&graph, &ranges, &specs, profile),
            Err(CompileError::Profile {
                rule: "S3: an elementwise nonlinearity is the SiLU table",
                ..
            })
        ));
    }
}

#[test]
fn small_rows_meet_the_epsilon_term() {
    // Token 3's embedding column is zero and token 4's holds a few ±1, with no shift into the
    // residual. The first norm's sum is then 0 or 3, below e = d: ρ = ⌊√(d·2^{2κ}/(u + e))⌋
    // depends on e, and stays inside the gadget's range check because e > d/4.
    for profile in [Profile::Servable, Profile::Minimal] {
        let shape = SHAPES[0];
        let mut model = tiny(77, profile, shape);
        let quant = &mut model.quant.public;
        quant.act_exponents[0].1 = quant.residual_exponent;
        let vocab = shape.vocab as usize;
        let embed = model
            .weights
            .iter_mut()
            .find(|w| w.ty == WeightType::Embed)
            .expect("the embedding");
        for c in 0..embed.rows as usize {
            embed.data[c * vocab + 3] = 0;
            embed.data[c * vocab + 4] = [1, 0, -1, 0, 0, 1][c % 6];
        }
        let tables = reference_tables(&model);
        let lowered = lowered(&model, 4);
        for tokens in [[3, 3, 3, 3], [4, 4, 4, 4], [3, 4, 1, 4]] {
            assert_eq!(
                run(&model, &lowered, &tokens),
                reference(&model, &tables, &tokens)
            );
        }
    }
}

#[test]
fn a_model_breaking_one_rule_is_refused_naming_it() {
    let base = tiny(31, Profile::Servable, SHAPES[0]);
    assert_eq!(validate(&base), Ok(()));
    let model_rule = |rule| CompileError::Model { rule };
    let layer = |site| QuantPoint::Layer { layer: 0, site };
    type Edit = Box<dyn Fn(&mut IntegerModel)>;
    let public: Vec<(&str, Edit)> = vec![
        (
            "a model has at least one layer",
            Box::new(|m| m.arch.n_layers = 0),
        ),
        (
            "n_kv_heads divides n_heads",
            Box::new(|m| m.arch.n_kv_heads = 3),
        ),
        (
            "d_head is even, from 2 to 2^16 − 2",
            Box::new(|m| m.arch.d_head = 3),
        ),
        (
            "d_model = n_heads · d_head",
            Box::new(|m| m.arch.d_model += 2),
        ),
        ("d_ff ≥ 1", Box::new(|m| m.arch.d_ff = 0)),
        (
            "1 ≤ vocab ≤ 2^18",
            Box::new(|m| m.arch.vocab = (1 << 18) + 1),
        ),
        (
            "1 ≤ max_context ≤ 2^17",
            Box::new(|m| m.arch.max_context = (1 << 17) + 1),
        ),
        (
            "the quantization data's RoPE is the architecture's",
            Box::new(|m| m.quant.public.rope.bits = 13),
        ),
        (
            "act_exponents holds the embedding, each INT8 site of the profile and the final \
             norm, once each and sorted",
            Box::new(|m| {
                m.quant.public.act_exponents.remove(1);
            }),
        ),
        (
            "act_exponents holds the embedding, each INT8 site of the profile and the final \
             norm, once each and sorted",
            Box::new(move |m| {
                let point = layer(Site::AttnResidual);
                let exponents = &mut m.quant.public.act_exponents;
                let at = exponents.partition_point(|&(p, _)| p < point);
                exponents.insert(at, (point, -4));
            }),
        ),
        (
            "shift_class_sets holds each site of the profile and the final norm, once each and \
             sorted",
            Box::new(|m| m.quant.public.shift_class_sets.swap(0, 1)),
        ),
        (
            "a site has one to four classes, each shift from 1 to 63",
            Box::new(|m| m.quant.public.shift_class_sets[0].1 = vec![0]),
        ),
        (
            "a site has one to four classes, each shift from 1 to 63",
            Box::new(|m| m.quant.public.shift_class_sets[1].1 = vec![9; 5]),
        ),
        (
            "a site after no weight has one class",
            Box::new(|m| m.quant.public.shift_class_sets[0].1 = vec![9, 10]),
        ),
        (
            "score_scales holds one scale per layer, sorted",
            Box::new(|m| {
                m.quant.public.score_scales.pop();
            }),
        ),
        (
            "a score scale is positive",
            Box::new(|m| {
                m.quant.public.score_scales[0].1 = Rational::new(-1, 4096).expect("a rational");
            }),
        ),
        (
            "silu_in_exponents holds one exponent per layer, sorted",
            Box::new(|m| m.quant.silu_in_exponents[0].0 = 5),
        ),
        (
            "the residual has 2 to 31 bits",
            Box::new(|m| m.quant.public.residual_bits = 32),
        ),
        (
            "the embedding's shift into the residual keeps INT8 rows inside it",
            Box::new(|m| {
                let residual = m.quant.public.residual_exponent;
                m.quant.public.act_exponents[0].1 = residual + 13;
            }),
        ),
        (
            "the embedding's shift into the residual keeps INT8 rows inside it",
            Box::new(|m| {
                let residual = m.quant.public.residual_exponent;
                m.quant.public.act_exponents[0].1 = residual - 1;
            }),
        ),
        (
            "RMSNorm's ε term e is at least 1, so that ρ exists",
            Box::new(|m| m.quant.public.rms_eps_int = 0),
        ),
        (
            "κ ≤ 31, so that N = d · 2^{2κ} stays below 2^94",
            Box::new(|m| m.quant.public.rsqrt_kappa = 32),
        ),
    ];
    for (rule, edit) in &public {
        let mut model = base.clone();
        edit(&mut model);
        let error = validate(&model).expect_err(rule);
        assert_eq!(error, model_rule(*rule));
        assert_eq!(error.code(), ErrorCode::QuantArch);
        // The lowering reads only the public part, and refuses it alike.
        assert_eq!(lower(PublicModel::of(&model), 4), Err(model_rule(*rule)));
    }

    let hidden: Vec<(&str, Edit)> = vec![
        (
            "the weights are the architecture's, one per type and layer",
            Box::new(|m| {
                m.weights.pop();
            }),
        ),
        (
            "the weights are the architecture's, ordered by (ty, layer)",
            Box::new(|m| m.weights.swap(1, 2)),
        ),
        (
            "each weight has its architecture's shape",
            Box::new(|m| {
                let w = &mut m.weights[0];
                (w.rows, w.cols) = (w.cols, w.rows);
            }),
        ),
        (
            "a weight holds rows · cols entries",
            Box::new(|m| {
                m.weights[3].data.pop();
            }),
        ),
        (
            "every weight lies in [−127, 127]",
            Box::new(|m| m.weights[5].data[7] = i8::MIN),
        ),
        (
            "an affine rescale needs public data the claim list lacks yet (#274)",
            Box::new(|m| {
                let rows = m.quant.channels[0].class.len();
                m.quant.channels[0].multiplier = vec![40_000; rows];
            }),
        ),
        (
            "a ChannelParams exists for each weight a site of several classes rescales, and \
             only for one, sorted by (ty, layer)",
            Box::new(|m| {
                m.quant.channels.remove(0);
            }),
        ),
        (
            "a ChannelParams exists for each weight a site of several classes rescales, and \
             only for one, sorted by (ty, layer)",
            Box::new(|m| {
                m.quant.channels.insert(
                    0,
                    ChannelParams {
                        ty: WeightType::Embed,
                        layer: 0,
                        class: vec![0; 16],
                        multiplier: Vec::new(),
                    },
                );
            }),
        ),
        (
            "a ChannelParams holds one class per output channel",
            Box::new(|m| {
                m.quant.channels[0].class.push(0);
            }),
        ),
        (
            "a channel class indexes its site's shift classes",
            Box::new(|m| m.quant.channels[0].class[0] = 4),
        ),
    ];
    assert!(
        !base.quant.channels.is_empty(),
        "the base model has a classed site"
    );
    for (rule, edit) in &hidden {
        let mut model = base.clone();
        edit(&mut model);
        assert_eq!(validate(&model), Err(model_rule(*rule)), "{rule}");
        // The public part is intact: the lowering does not read what broke.
        assert!(lower(PublicModel::of(&model), 4).is_ok(), "{rule}");
    }

    // Tables the public constants define, refused by RFC-0008: RoPE's θ, then layer 0's SiLU
    // input exponent.
    let mut model = base.clone();
    model.arch.rope.theta = 1;
    model.quant.public.rope.theta = 1;
    let error = lower(PublicModel::of(&model), 4).expect_err("θ = 1");
    assert!(matches!(
        error,
        CompileError::ModelTable {
            table: TableId(0),
            ..
        }
    ));
    assert_eq!(error.code(), ErrorCode::QuantArch);
    let mut model = base.clone();
    model.quant.silu_in_exponents[0].1 = 1;
    let error = lower(PublicModel::of(&model), 4).expect_err("in_exp = 1");
    assert!(matches!(error, CompileError::ModelTable { .. }), "{error}");
    assert_eq!(error.code(), ErrorCode::QuantArch);

    // The embedding's shift at its edges: 127 · 2^{s_e} < 2^{bits − 1}.
    for (bits, shift, valid) in [
        (20, 12, true),
        (20, 13, false),
        (31, 23, true),
        (31, 24, false),
    ] {
        let mut model = base.clone();
        let quant = &mut model.quant.public;
        quant.residual_bits = bits;
        quant.act_exponents[0].1 = quant.residual_exponent + shift;
        let rule = "the embedding's shift into the residual keeps INT8 rows inside it";
        let expected = if valid { Ok(()) } else { Err(model_rule(rule)) };
        assert_eq!(validate(&model), expected, "{bits} bits, s_e = {shift}");
    }

    // A sequence of no position, or longer than the context.
    for n in [0, SHAPES[0].context + 1] {
        let error = lower(PublicModel::of(&base), n).expect_err("refused");
        assert_eq!(
            error,
            CompileError::Usage {
                rule: "a sequence has 1 to max_context positions"
            }
        );
        assert_eq!(error.code(), ErrorCode::Usage);
    }
}

// ----- Real shapes -------------------------------------------------------------------------------

/// Every norm of `graph`: its sum's range and its root's.
fn norms(graph: &tessera_ir::Graph) -> Vec<(IntRange, IntRange)> {
    let range = |t: tessera_ir::TensorId| graph.tensors[t.0 as usize].range;
    graph
        .nodes
        .iter()
        .filter(|n| matches!(n.op, Op::RsqrtRow { .. }))
        .map(|n| (range(n.inputs[0]), range(n.outputs[0])))
        .collect()
}

#[test]
fn llama_shapes_lower_from_their_public_part_alone() {
    // 8B's shape through all 32 layers: the saturating residual keeps every norm's sum at most
    // d · 2^38 (RFC-0022, #284). With e = 672 < d / 4 and RFC-0023's κ = 30, the largest root
    // takes the full 32 bits of one lookup, and every root lies inside the range check (#285).
    for profile in [Profile::Servable, Profile::Minimal] {
        let (arch, quant, silu) = llama(profile, 32, 4096, 32, 8, 14_336);
        let model = PublicModel {
            arch: &arch,
            profile,
            quant: &quant,
            silu_in_exponents: &silu,
        };
        let lowering = lower(model, 64).expect("8B's shape lowers");
        let graph = &lowering.graph;
        // The first norm reads the embedding's 127 · 2^{s_e}, s_e = 1; every later one the
        // saturated residual.
        let norms = norms(graph);
        assert_eq!(norms.len(), 2 * 32 + 1);
        for (k, (sum, root)) in norms.into_iter().enumerate() {
            let top = if k == 0 { 4096 * 254 * 254 } else { 4096 << 38 };
            assert_eq!(sum, IntRange { lo: 0, hi: top });
            assert_eq!(
                root,
                IntRange {
                    lo: 0,
                    hi: (1 << 32) - 1
                }
            );
        }
        let range = |t: tessera_ir::TensorId| graph.tensors[t.0 as usize].range;
        let find = |op: fn(&Op) -> bool| graph.nodes.iter().find(|n| op(&n.op)).expect("a node");
        // The embedding rows stay INT8 by the one-hot fact.
        let embed = find(|op| {
            matches!(
                op,
                Op::MatMulWeight {
                    weight: WeightRef {
                        ty: WeightType::Embed,
                        ..
                    }
                }
            )
        });
        assert_eq!(range(embed.outputs[0]), IntRange { lo: -127, hi: 127 });
        // Scores of INT8 heads of width 128 reach 2^21, inside a 22-bit Exp table's keys.
        let softmax = find(|op| matches!(op, Op::SoftmaxRow { .. }));
        assert_eq!(
            range(softmax.inputs[0]),
            IntRange {
                lo: -(1 << 21) + (1 << 14),
                hi: 1 << 21
            }
        );
        let kinds: Vec<&TableKind> = lowering.tables.iter().map(|t| &t.kind).collect();
        assert!(matches!(kinds[0], TableKind::Rope { .. }));
        assert!(matches!(kinds[1], TableKind::Exp { key_bits: 22, .. }));
        // The gate accumulator's keys within ±127 · 128 · 4096 < 2^26: b = 27.
        assert!(matches!(kinds[2], TableKind::Silu { b: 27, .. }));
        // Every layer's tables are the same definition, shared.
        assert_eq!(lowering.tables.len(), 3);
        // The attention output by the softmax fact.
        let output = graph
            .nodes
            .iter()
            .find(|n| n.op == Op::Mul && n.inputs[1] == softmax.outputs[1])
            .expect("R · PV");
        assert_eq!(
            range(output.outputs[0]),
            IntRange {
                lo: -(1 << 39),
                hi: 127 << 32
            }
        );
        // Each tensor takes the narrowest type of its range: INT8 rescales, the 20-bit residual
        // updates, 23-bit scores, 1-bit one-hot entries.
        let ty = |t: tessera_ir::TensorId| graph.tensors[t.0 as usize].ty;
        let typed = |bits, signed| tessera_ir::IntType { bits, signed };
        for node in &graph.nodes {
            if let Op::Requant { lo, .. } = node.op {
                let bits = if lo == -128 { 8 } else { 20 };
                assert_eq!(ty(node.outputs[0]), typed(bits, true));
            }
        }
        assert_eq!(ty(softmax.inputs[0]), typed(23, true));
        assert_eq!(ty(softmax.outputs[0]), typed(17, false));
        assert_eq!(ty(lowering.one_hot), typed(1, false));
        assert_eq!(ty(lowering.tokens), typed(17, false));
        // The hooks.
        assert_eq!(
            graph.tensors[lowering.logits.0 as usize].shape,
            [64, 128_256]
        );
        assert_eq!(
            graph.tensors[lowering.one_hot.0 as usize].shape,
            [64, 128_256]
        );
        assert_eq!(
            graph.tensors[lowering.positions.0 as usize].kind,
            NodeKind::Public
        );
        assert!(lowering.classes.is_empty());
    }
    // 405B's shape through all 126 layers. Its widest accumulator, minimal's O projection, fits
    // RequantSplit's 62 bits: 127 · 16384 · 2^39 < 2^60.
    let (arch, quant, silu) = llama(Profile::Minimal, 126, 16_384, 128, 8, 53_248);
    let model = PublicModel {
        arch: &arch,
        profile: Profile::Minimal,
        quant: &quant,
        silu_in_exponents: &silu,
    };
    let lowering = lower(model, 8).expect("405B's shape lowers");
    let graph = &lowering.graph;
    let norms = norms(graph);
    assert_eq!(norms.len(), 2 * 126 + 1);
    for (k, (sum, root)) in norms.into_iter().enumerate() {
        let top = if k == 0 {
            16_384 * 254 * 254
        } else {
            16_384 << 38
        };
        assert_eq!(sum, IntRange { lo: 0, hi: top });
        assert_eq!(
            root,
            IntRange {
                lo: 0,
                hi: (1 << 32) - 1
            }
        );
    }
    let o = graph
        .nodes
        .iter()
        .find(|n| {
            matches!(
                n.op,
                Op::MatMulWeight {
                    weight: WeightRef {
                        ty: WeightType::O,
                        ..
                    }
                }
            )
        })
        .expect("the O projection");
    let acc = graph.tensors[o.outputs[0].0 as usize].range;
    assert_eq!(acc.hi, 127 * 16_384 * (1 << 39));
    assert!(acc.hi < 1 << 60);
}

#[test]
fn the_residual_saturates_as_rfc_0022_defines() {
    // Residual shifts of 1 and 2 make the updates dwarf INT20: the sum saturates at both bounds,
    // through the fused key of a site of one class and the two steps of a site of several, whose
    // first clamp binds too. The lowered graph keeps the reference's logits throughout. κ = 28,
    // with every norm's shift 12 wider, keeps a saturated row's root near 2^9: at κ = 16 it is 0,
    // and every norm after saturation would output 0 whatever the residual held.
    let mut rng = Seeded(0x5A7_0284);
    let (mut low, mut high) = (false, false);
    for profile in [Profile::Servable, Profile::Minimal] {
        for &shape in &SHAPES {
            let mut model = tiny(rng.next(), profile, shape);
            model.quant.public.rsqrt_kappa = 28;
            for (point, shifts) in &mut model.quant.public.shift_class_sets {
                let (residual, norm) = match point {
                    QuantPoint::Layer { site, .. } => (
                        matches!(site, Site::AttnResidual | Site::MlpResidual),
                        matches!(site, Site::AttnNorm | Site::MlpNorm),
                    ),
                    QuantPoint::FinalNorm => (false, true),
                    QuantPoint::Embed => (false, false),
                };
                if residual {
                    for (k, r) in shifts.iter_mut().enumerate() {
                        *r = 1 + i8::try_from(k % 2).expect("small");
                    }
                }
                if norm {
                    shifts[0] += 12;
                }
            }
            let tables = reference_tables(&model);
            let lowered = lowered(&model, shape.context);
            let lowering = &lowered.lowering;
            for _ in 0..4 {
                let tokens: Vec<u32> = (0..shape.context)
                    .map(|_| {
                        u32::try_from(rng.range(0, i128::from(shape.vocab) - 1)).expect("a token")
                    })
                    .collect();
                assert_eq!(
                    run(&model, &lowered, &tokens),
                    reference(&model, &tables, &tokens),
                    "{profile:?}"
                );
                let values = evaluate(
                    &lowering.graph,
                    &lowered.tables,
                    inputs(&model, lowering, &tokens),
                );
                for node in &lowering.graph.nodes {
                    if let Op::Requant {
                        lo: -524_288,
                        hi: 524_287,
                        ..
                    } = node.op
                    {
                        let out = &values[node.outputs[0].0 as usize];
                        low |= out.contains(&-524_288);
                        high |= out.contains(&524_287);
                    }
                }
            }
        }
    }
    assert!(low && high, "the residual reached both of its bounds");
}
