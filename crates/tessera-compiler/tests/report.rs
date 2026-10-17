// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The commitment report (RFC-0006 'Outputs') against counts derived by hand from RFC-0005's
//! sites and RFC-0009's gadgets.
//!
//! - **The source's count.** `minimal` at 405B shapes requantizes 137,216 elements per layer per
//!   token, the source's eight tensors.
//! - **RISK-2's counts.** At 405B shapes, `servable` commits RISK-2's 359,424 elements per layer
//!   per token when its residual sites take two shift classes. With four, the down projection's
//!   update key reaches 33 bits and splits: 16,384 more. `minimal` commits RISK-2's upper bound,
//!   285,696 elements with 302,080 lookups, and 252,928 with 269,312 with one class.
//! - **Classes.** Lookup outputs, split limbs, row scalars and one-hot entries are counted apart,
//!   per layer and per token: `RequantSplit` adds one limb per element, `RequantWide` three, or two
//!   without low bits. A residual site of several classes updates in two steps.
//! - **Scores.** The causal rows hold `n(n + 1)/2` scores per head, whose product trees have
//!   `2^{⌈log2 n⌉}` leaves per row.
//! - **Totals.** What the layers leave over per token is the embedding's one-hot vector and the
//!   final norm; the tables' lookups and the row gadgets' make up every lookup, in both profiles;
//!   the parameters are Llama-3.1-8B's published count without its norm gains, which the quantizer
//!   folds.
//! - **Stages.** RFC-0010's: 3 in `servable` and 4 in `minimal`, at any depth (INV-SCH-4). On a
//!   chain built by hand, each sumcheck relation adds a stage, each claim map none, and a committed
//!   tensor ends the chain.

mod common;

use common::{ADVICE, decl, llama, one, virt};
use tessera_compiler::{
    CommitmentReport, Gadget, PublicModel, Selection, analyze, lower, report, select, stage_count,
};
use tessera_ir::{
    Broadcast, GadgetPolicy, GraphBuilder, NodeKind, Op, Profile, PublicQuantSpec, QuantPoint,
    Site, TableId, TensorId, WeightRef, WeightType,
};

/// `d_model`, `n_heads`, `n_kv_heads` and `d_ff`.
type Shape = (u32, u32, u32, u32);

const B8: Shape = (4096, 32, 8, 14_336);
const B405: Shape = (16_384, 128, 8, 53_248);

/// The report of a Llama-shaped model whose public part `quant` edits and whose selection
/// `selection` edits.
fn report_with(
    profile: Profile,
    layers: u32,
    (d, heads, kv_heads, d_ff): Shape,
    n: u32,
    quant: impl FnOnce(&mut PublicQuantSpec),
    selection: impl FnOnce(&mut Selection),
) -> CommitmentReport {
    let (arch, mut public, silu) = llama(profile, layers, d, heads, kv_heads, d_ff);
    quant(&mut public);
    let model = PublicModel {
        arch: &arch,
        profile,
        quant: &public,
        silu_in_exponents: &silu,
    };
    let lowering = lower(model, n).expect("the shape lowers");
    let ranges = analyze(&lowering.graph, &lowering.tables).expect("ranges");
    let policy = GadgetPolicy::default();
    let mut selected =
        select(&lowering.graph, &ranges, &lowering.tables, &policy).expect("every gadget fits");
    selection(&mut selected);
    report(model, &lowering, &selected, &policy).expect("a report")
}

fn report_of(profile: Profile, layers: u32, shape: Shape, n: u32) -> CommitmentReport {
    report_with(profile, layers, shape, n, |_| {}, |_| {})
}

/// Gives every residual site `n` shift classes, from its own shift up.
fn residual_classes(n: i8) -> impl FnOnce(&mut PublicQuantSpec) {
    move |quant| {
        for (point, shifts) in &mut quant.shift_class_sets {
            if let QuantPoint::Layer {
                site: Site::AttnResidual | Site::MlpResidual,
                ..
            } = point
            {
                *shifts = (shifts[0]..shifts[0] + n).collect();
            }
        }
    }
}

/// Every lookup is a table's, or a row gadget's: `SoftmaxRow`'s 7 per row, `RsqrtRow`'s 30.
fn assert_every_lookup_is_a_tables_or_a_row_gadgets(r: &CommitmentReport, heads: u64) {
    let tables: u64 = r.tables.iter().map(|t| t.lookups_per_token).sum();
    let layers = u64::from(r.shape[0]);
    assert_eq!(
        tables + layers * (heads * 7 + 2 * 30) + 30,
        r.per_token.lookups
    );
}

#[test]
fn minimal_405b_requantizes_the_sources_137216_elements_per_layer_per_token() {
    let r = report_of(Profile::Minimal, 2, B405, 2);
    assert_eq!(
        r.requantized,
        [
            (Site::AttnNorm, 16_384),
            (Site::RopeQ, 16_384),
            (Site::RopeK, 1024),
            (Site::V, 1024),
            (Site::AttnResidual, 16_384),
            (Site::MlpNorm, 16_384),
            (Site::Silu, 53_248),
            (Site::MlpResidual, 16_384),
        ]
    );
    assert_eq!(r.requantized_per_layer(), 137_216);
}

#[test]
fn the_405b_counts_are_risk_2s() {
    // The rescales' committed elements and lookups per layer per token, without the row gadgets'
    // 128 · 7 + 2 · 30 lookups.
    let rescales = |profile, classes| {
        let r = report_with(profile, 2, B405, 2, residual_classes(classes), |_| {});
        let layer = r.per_layer;
        (
            layer.lookup_outputs + layer.split_limbs,
            layer.lookups - (128 * 7 + 2 * 30),
        )
    };
    // servable, one class: 260,096 outputs and 66,560 limbs. Two: each residual site's update
    // adds one output per element, and its key keeps one table with its class bit.
    assert_eq!(rescales(Profile::Servable, 1), (326_656, 326_656));
    assert_eq!(rescales(Profile::Servable, 2), (359_424, 359_424));
    // Four: the down projection's update key, 31 bits of `127 · 128 · 53,248` and two of classes,
    // splits.
    assert_eq!(rescales(Profile::Servable, 4), (375_808, 375_808));
    // minimal: RISK-2's upper bound with classes, two steps at each residual site, at two
    // classes as at four.
    assert_eq!(rescales(Profile::Minimal, 1), (252_928, 269_312));
    assert_eq!(rescales(Profile::Minimal, 2), (285_696, 302_080));
    assert_eq!(rescales(Profile::Minimal, 4), (285_696, 302_080));
}

#[test]
fn committed_elements_are_counted_by_class() {
    // servable 8B: eleven sites; the two norms, RoPE's queries and keys and the attention output
    // split, one limb per element.
    let r = report_of(Profile::Servable, 2, B8, 5);
    assert_eq!(r.requantized_per_layer(), 69_632);
    let layer = r.per_layer;
    assert_eq!(layer.lookup_outputs, 69_632);
    assert_eq!(layer.split_limbs, 4096 + 4096 + 1024 + 4096 + 4096);
    // Per row: SoftmaxRow's c_q, R_q and two NonNeg(·, 34) of two committed limbs each, for 32
    // heads; RsqrtRow's 20 for each of two norms.
    assert_eq!(layer.row_scalars, 32 * 6 + 2 * 20);
    assert_eq!(layer.one_hot, 0);
    // A lookup per output, one more per split element, and the row gadgets' 7 and 30.
    assert_eq!(layer.lookups, 69_632 + 17_408 + 32 * 7 + 2 * 30);
    // minimal 8B: the down projection's residual takes RequantWide, three parts per element.
    let m = report_of(Profile::Minimal, 2, B8, 5);
    let splits = 4096 + 4096 + 1024 + 4096 + 4096;
    assert_eq!(m.per_layer.split_limbs, splits + 3 * 4096);
    assert_eq!(
        m.per_layer.lookups,
        36_864 + splits + 4 * 4096 + 32 * 7 + 2 * 30
    );
    assert_every_lookup_is_a_tables_or_a_row_gadgets(&m, 32);
}

#[test]
fn a_wide_requantization_without_low_bits_commits_one_part_fewer() {
    // RFC-0024: four commitments and five lookups per element, three and four when s = 0.
    let wide = report_of(Profile::Minimal, 2, B8, 2).per_layer;
    let without_low = report_with(
        Profile::Minimal,
        2,
        B8,
        2,
        |_| {},
        |selection| {
            for (_, gadget) in &mut selection.gadgets {
                if let Gadget::RequantWide { low, .. } = gadget {
                    *low = None;
                }
            }
        },
    )
    .per_layer;
    assert_eq!(wide.lookup_outputs, without_low.lookup_outputs);
    assert_eq!(wide.split_limbs - without_low.split_limbs, 4096);
    assert_eq!(wide.lookups - without_low.lookups, 4096);
}

#[test]
fn a_residual_site_of_several_classes_updates_in_two_steps() {
    // RFC-0022: the clamped update, then the saturating sum, each a lookup output; the site is
    // still one site, and at 8B both keys fit one table with their class bit.
    let one = report_of(Profile::Servable, 2, B8, 2);
    let two = report_with(Profile::Servable, 2, B8, 2, residual_classes(2), |_| {});
    assert_eq!(two.requantized, one.requantized);
    assert_eq!(
        two.per_layer.lookup_outputs,
        one.per_layer.lookup_outputs + 2 * 4096
    );
    assert_eq!(two.per_layer.split_limbs, one.per_layer.split_limbs);
    assert_eq!(two.per_layer.lookups, one.per_layer.lookups + 2 * 4096);
}

#[test]
fn the_tokens_totals_are_the_layers_and_the_ends_of_the_model() {
    let r = report_of(Profile::Servable, 2, B8, 5);
    let (token, layer) = (r.per_token, r.per_layer);
    // The embedding's one-hot vector, and the final norm: its output and limb, its root.
    assert_eq!(token.one_hot, 128_256);
    assert_eq!(token.lookup_outputs - 2 * layer.lookup_outputs, 4096);
    assert_eq!(token.split_limbs - 2 * layer.split_limbs, 4096);
    assert_eq!(token.row_scalars - 2 * layer.row_scalars, 20);
    assert_eq!(token.lookups - 2 * layer.lookups, 2 * 4096 + 30);
    assert_every_lookup_is_a_tables_or_a_row_gadgets(&r, 32);
    assert_eq!(
        r.multiplicities(),
        r.tables.iter().map(|t| t.rows).sum::<u64>()
    );
    // Five positions: 15 scores per head and layer, rows of 8 leaves.
    assert_eq!(r.scores, 2 * 32 * 15);
    assert_eq!(r.product_tree_leaves, 2 * 32 * 5 * 8);
    // Every score reads its layer's Exp table.
    assert_eq!(
        r.tables.iter().map(|t| t.score_lookups).sum::<u64>(),
        r.scores
    );
    assert!((r.scores_per_token_per_ctx() - 32.0).abs() < f64::EPSILON);
}

#[test]
fn the_parameters_are_llama_8bs_without_its_norm_gains() {
    // 8,030,261,248 parameters, of which 2 · 32 + 1 gains of 4096 are folded into the weights.
    let r = report_of(Profile::Servable, 32, B8, 2);
    assert_eq!(r.parameters, 8_030_261_248 - 65 * 4096);
    let counts = r.counts();
    assert_eq!(
        counts["native_ops_lin"].as_f64(),
        Some(2.0 * r.parameters as f64)
    );
    assert_eq!(
        counts["scores_per_token_per_ctx"].as_f64(),
        Some(32.0 * 32.0 / 2.0)
    );
    // F_att: four operations per score per head dimension, QK^T's and PV's products.
    assert_eq!(
        counts["native_ops_att_per_ctx"].as_f64(),
        Some(4.0 * 128.0 * 512.0)
    );
}

/// Each layer's `site` affine (RFC-0025), its shift 16 bits above the `pot` one.
fn affine_at(site: Site) -> impl FnOnce(&mut PublicQuantSpec) {
    move |quant| {
        let points: Vec<(QuantPoint, i8)> = quant
            .shift_class_sets
            .iter()
            .filter(|(p, _)| matches!(p, QuantPoint::Layer { site: s, .. } if *s == site))
            .map(|(p, shifts)| (*p, shifts[0] + 16))
            .collect();
        quant
            .shift_class_sets
            .retain(|(p, _)| points.iter().all(|(q, _)| q != p));
        quant.affine_sites.extend(points);
        quant.affine_sites.sort_by_key(|&(p, _)| p);
    }
}

#[test]
fn the_stage_count_is_rfc_0010s_at_every_depth() {
    let stages = |profile, layers, quant: fn(&mut PublicQuantSpec)| {
        report_with(profile, layers, B8, 3, quant, |_| {}).stages
    };
    for layers in [2, 4, 8] {
        assert_eq!(stages(Profile::Servable, layers, |_| {}), 3, "{layers}");
        assert_eq!(stages(Profile::Minimal, layers, |_| {}), 4, "{layers}");
        // RFC-0025: an affine RoPE site's key is `Mul ← Rotate ← MatMulWeight`, three relations,
        // so `servable` takes `minimal`'s four stages; an affine `V`'s two keep it at three.
        let rope = |q: &mut PublicQuantSpec| affine_at(Site::RopeQ)(q);
        let v = |q: &mut PublicQuantSpec| affine_at(Site::V)(q);
        assert_eq!(stages(Profile::Servable, layers, rope), 4, "{layers}");
        assert_eq!(stages(Profile::Servable, layers, v), 3, "{layers}");
        assert_eq!(stages(Profile::Minimal, layers, rope), 4, "{layers}");
    }
}

/// The stages of a graph whose one gadget requantizes the key of `shape` that `chain` builds from
/// two committed `[4, 8]` tensors.
fn stages_of(
    shape: &[u32],
    chain: impl FnOnce(&mut GraphBuilder, TensorId, TensorId) -> TensorId,
) -> u32 {
    let mut g = GraphBuilder::new();
    let x = g.input(decl(&[4, 8], ADVICE, -128, 127)).expect("x");
    let y = g.input(decl(&[4, 8], ADVICE, -128, 127)).expect("y");
    let key = chain(&mut g, x, y);
    let shift = g.shift_classes(vec![4]).expect("a shift");
    let op = Op::Requant {
        shift,
        lo: -128,
        hi: 127,
    };
    one(&mut g, op, &[key], decl(shape, ADVICE, -128, 127));
    stage_count(&g.finish())
}

#[test]
fn each_sumcheck_relation_adds_a_stage_and_each_claim_map_none() {
    let matmul = |g: &mut GraphBuilder, x: TensorId| {
        let weight = WeightRef {
            ty: WeightType::Q,
            layer: 0,
        };
        let w = g.weight(weight, 8, 8).expect("a weight");
        one(g, Op::MatMulWeight { weight }, &[x, w], virt(&[4, 8]))
    };
    let rotate = |g: &mut GraphBuilder, x: TensorId| {
        let positions = g
            .input(decl(&[4], NodeKind::Public, 0, 3))
            .expect("positions");
        let rope = TableId(0);
        one(g, Op::Rotate { rope }, &[x, positions], virt(&[4, 8]))
    };
    // RFC-0010's sumcheck relations.
    assert_eq!(
        stages_of(&[4, 8], |g, x, _| matmul(g, x)),
        2,
        "MatMulWeight"
    );
    assert_eq!(
        stages_of(&[4, 4], |g, x, y| one(
            g,
            Op::MatMulAct {
                transpose_rhs: true
            },
            &[x, y],
            virt(&[4, 4])
        )),
        2,
        "MatMulAct"
    );
    assert_eq!(
        stages_of(&[4, 8], |g, x, y| one(
            g,
            Op::Mul {
                broadcast: Broadcast::Trailing
            },
            &[x, y],
            virt(&[4, 8])
        )),
        2,
        "Mul"
    );
    assert_eq!(stages_of(&[4, 8], |g, x, _| rotate(g, x)), 2, "Rotate");
    assert_eq!(
        stages_of(&[4], |g, x, _| one(
            g,
            Op::SumSquares { axis: 1 },
            &[x],
            virt(&[4])
        )),
        2,
        "SumSquares"
    );
    assert_eq!(
        stages_of(&[4, 8], |g, x, _| {
            let c = g.constant((1..=8).collect()).expect("per channel");
            one(g, Op::ConstMul { c }, &[x], virt(&[4, 8]))
        }),
        2,
        "ConstMul per channel"
    );
    // Claim maps. A sum over an axis is one: its claim lands at `(½, …, ½)` on the axis
    // (RFC-0010 'Relations').
    assert_eq!(
        stages_of(&[4], |g, x, _| one(
            g,
            Op::Sum { axis: 1 },
            &[x],
            virt(&[4])
        )),
        1,
        "Sum"
    );
    assert_eq!(
        stages_of(&[4, 8], |g, x, _| {
            let c = g.constant(vec![3]).expect("one constant");
            one(g, Op::ConstMul { c }, &[x], virt(&[4, 8]))
        }),
        1,
        "ConstMul by one constant"
    );
    assert_eq!(
        stages_of(&[4, 8], |g, x, _| {
            let c = g.constant(vec![3]).expect("one constant");
            one(g, Op::ConstAdd { c }, &[x], virt(&[4, 8]))
        }),
        1,
        "ConstAdd"
    );
    assert_eq!(
        stages_of(&[4, 8], |g, x, y| one(g, Op::Add, &[x, y], virt(&[4, 8]))),
        1,
        "Add"
    );
    assert_eq!(
        stages_of(&[4, 8], |g, x, y| one(g, Op::Sub, &[x, y], virt(&[4, 8]))),
        1,
        "Sub"
    );
    assert_eq!(
        stages_of(&[4, 2, 4], |g, x, _| one(
            g,
            Op::Reshape,
            &[x],
            virt(&[4, 2, 4])
        )),
        1,
        "Reshape"
    );
    // Relations chain through virtual tensors, and a committed tensor ends the chain.
    assert_eq!(
        stages_of(&[4, 8], |g, x, _| {
            let acc = matmul(g, x);
            rotate(g, acc)
        }),
        3,
        "Rotate after MatMulWeight"
    );
    assert_eq!(
        stages_of(&[4, 8], |g, x, y| {
            let acc = matmul(g, x);
            let shift = g.shift_classes(vec![4]).expect("a shift");
            let op = Op::Requant {
                shift,
                lo: -128,
                hi: 127,
            };
            let committed = one(g, op, &[acc], decl(&[4, 8], ADVICE, -128, 127));
            one(
                g,
                Op::Mul {
                    broadcast: Broadcast::Trailing,
                },
                &[committed, y],
                virt(&[4, 8]),
            )
        }),
        2,
        "Mul after a committed requantization"
    );
}
