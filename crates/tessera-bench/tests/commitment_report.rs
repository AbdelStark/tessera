// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Llama-3.1-405B's commitment reports (RFC-0006 'Outputs'), committed in `docs/perf/v0.2/`: both
//! profiles, with one, two or four shift classes at every site after a weight, at `B_max` of 30
//! and 32. Each model is lowered for 16 positions with the shifts of the compiler's lowering
//! tests. The cost model projects it at 1K, 4K, 16K and 128K with the source's unit costs, memory
//! and scenario and five regenerations, as 08 §3 does.
//!
//! - **The reports.** `cargo xtask lint` validates each against its JSON Schema. This test keeps
//!   them current, up to the last bits of numbers that platforms round differently, and
//!   `TESSERA_WRITE_EXAMPLES=1` makes it rewrite them.
//! - **`B_max = 30`** refuses all six configurations: RFC-0050's root needs Range(32).
//! - **`B_max = 34`** lies outside RFC-0008, whose tables hold at most `2^{32}` rows. Under
//!   RFC-0006's selection rule it would change one gadget: with four classes, `servable`'s down
//!   projection update, whose 33-bit key would fit one table.
//!
//! - **The reduction stages' working sets** (RFC-0055). The two-class reports' tables, evaluated at
//!   the projections' batch of 16K tokens, are the schedule's own at 16K positions, and the cost
//!   model's working sets and windows follow from them: 4.08 and 5.50 TB in `minimal`'s stages 1
//!   and 2, windows of 6 rounds each; 9.57 TB in `servable`'s stage 1, a window of 8.
//!
//! `docs/perf/v0.2/README.md` reads these numbers and answers OQ-0006-1.

use std::fs;
use std::path::PathBuf;

use serde_json::Value;
use tessera_bench::{
    Counts, MemoryModel, ModelReport, ReductionStage, STAGE_BYTES_PER_ENTRY, Scenario, StageTable,
    UnitCosts, stage_plan,
};
use tessera_codec::{Ascii, Rational, envelope, schema};
use tessera_compiler::{
    CompileError, Gadget, Lowering, ModelAdviceLayout, PublicModel, Range, Ranges, Selection,
    analyze, claim_list, derive_schedule, lower, report, schedule::Relation, select,
};
use tessera_ir::{
    AdviceLayoutParams, ArchFamily, ArchSpec, CommitmentLayout, GadgetPolicy, Op, ParamSetId,
    Profile, PublicQuantSpec, QuantPoint, RopeSpec, Site, WeightLayoutParams,
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

/// The report of `selection` as `tessera.commitment-report/1`, with its projections.
fn report_json(
    model: PublicModel<'_>,
    lowering: &Lowering,
    selection: &Selection,
    policy: &GadgetPolicy,
) -> Value {
    let report = report(model, lowering, selection, policy).expect("a report");
    let counts = report.counts();
    let number = |key: &str| counts[key].as_f64().expect("a count");
    let model_counts = Counts {
        commits_per_token: number("commits_per_token"),
        lookups_per_token: number("lookups_per_token"),
        scores_per_token_per_ctx: number("scores_per_token_per_ctx"),
        native_ops_lin: number("native_ops_lin"),
        native_ops_att_per_ctx: number("native_ops_att_per_ctx"),
        regen_full_equivalents: 5.0,
        table_rows: number("table_rows"),
        score_block: number("score_block"),
        reduction_stages: counts["reduction_stages"]
            .as_array()
            .expect("the reduction stages")
            .iter()
            .map(|stage| ReductionStage {
                tables: stage["tables"]
                    .as_array()
                    .expect("the tables")
                    .iter()
                    .map(|t| StageTable {
                        per_token: t["per_token"].as_bool().expect("a flag"),
                        windowable: t["windowable"].as_bool().expect("a flag"),
                        log2_entries: u32::try_from(t["log2_entries"].as_u64().expect("a size"))
                            .expect("a size"),
                        count: t["count"].as_f64().expect("a count"),
                    })
                    .collect(),
            })
            .collect(),
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

/// Compares `report` with the committed file, after rewriting it under `TESSERA_WRITE_EXAMPLES`.
fn check_committed(name: &str, report: &Value) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/perf/v0.2")
        .join(name);
    if std::env::var_os("TESSERA_WRITE_EXAMPLES").is_some() {
        let text = serde_json::to_string_pretty(report).expect("JSON") + "\n";
        fs::write(&path, text).expect("the report");
    }
    let committed: Value =
        serde_json::from_str(&fs::read_to_string(&path).expect("the committed report"))
            .expect("JSON");
    assert!(
        same(&committed, report),
        "{} is stale: rerun this test with TESSERA_WRITE_EXAMPLES=1",
        path.display()
    );
}

/// The signed width `b` of a key in `[lo, hi]`: `2^{b−1} > hi` and `2^{b−1} ≥ −lo`, as gadget
/// selection reads it (RFC-0006).
fn signed_width(lo: i128, hi: i128) -> u32 {
    let bits = |x: i128| u128::BITS - x.max(0).unsigned_abs().leading_zeros();
    1 + bits(hi).max(bits(!lo))
}

/// Elements per layer per token whose requantization would change gadget at `B_max = 34`, by
/// RFC-0006's selection rule: a split key of at most 34 bits with its class bits would take
/// `Requant1`, and a wide key whose split `b + ⌈log2 n⌉ − 34` falls below every rounding bit would
/// split. Tables of more than `2^{32}` rows are outside RFC-0008, so this is a hypothesis.
fn changed_at_34(lowering: &Lowering, ranges: &Ranges, selection: &Selection) -> u64 {
    let graph = &lowering.graph;
    let mut changed = 0;
    for &(node, gadget) in &selection.gadgets {
        let node = &graph.nodes[node.0 as usize];
        let Op::Requant { shift, .. } = &node.op else {
            continue;
        };
        let shifts = &graph.shift_classes[shift.0 as usize];
        let class_bits = usize::BITS - (shifts.len().max(1) - 1).leading_zeros();
        let Some(Range::Bounded(key)) = ranges.of(node.inputs[0]) else {
            panic!("every requantized key is bounded");
        };
        let width = signed_width(key.lo, key.hi) + class_bits;
        let least = u32::from(shifts.iter().copied().min().expect("a class"));
        let changes = match gadget {
            Gadget::RequantSplit { .. } => width <= 34,
            Gadget::RequantWide { .. } => width.saturating_sub(34) < least,
            _ => false,
        };
        if changes {
            let elements: u64 = graph.tensors[node.outputs[0].0 as usize]
                .shape
                .iter()
                .map(|&n| u64::from(n))
                .product();
            changed += elements;
        }
    }
    changed / u64::from(POSITIONS) / u64::from(LAYERS)
}

#[test]
fn the_committed_405b_reports_are_the_compilers() {
    for profile in [Profile::Servable, Profile::Minimal] {
        let name = match profile {
            Profile::Servable => "servable",
            Profile::Minimal => "minimal",
        };
        for classes in [1, 2, 4] {
            let (arch, quant, silu) = llama_405b(profile, classes);
            let model = PublicModel {
                arch: &arch,
                profile,
                quant: &quant,
                silu_in_exponents: &silu,
            };
            let lowering = lower(model, POSITIONS).expect("405B's shape lowers");
            let ranges = analyze(&lowering.graph, &lowering.tables).expect("ranges");
            for b_max in [30u8, 32] {
                let policy = GadgetPolicy {
                    max_log2_rows: b_max,
                    ..GadgetPolicy::default()
                };
                let case = format!("{name}, {classes} classes, B_max = {b_max}");
                match select(&lowering.graph, &ranges, &lowering.tables, &policy) {
                    Ok(selection) => {
                        assert_eq!(b_max, 32, "{case}: the root requires Range(32)");
                        if profile == Profile::Servable && classes == 2 {
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
                            .unwrap();
                            let full = lower(model, 16384).unwrap();
                            let actual =
                                ModelAdviceLayout::new(&full, &selection, &claims).unwrap();
                            let bytes = envelope::seal(&schema::CLAIM_LIST, &claims);
                            let (independent, validated) =
                                tv_core::validation::load(&bytes, &tv_codec::DecodeLimits::MESSAGE)
                                    .unwrap();
                            assert_eq!(independent.arch.n_layers, 126);
                            let native = validated.advice_layout(16384).unwrap();
                            assert_eq!(actual.layout().allocated_rows(), native.rows());
                            assert_eq!(actual.layout().variables(), native.nu());
                            assert_eq!(actual.layout().allocated_rows(), 2_328_576);
                            assert_eq!(actual.layout().variables(), 44);
                            assert_eq!(actual.layout().rows(), 953_580);
                        }
                        let report = report_json(model, &lowering, &selection, &policy);
                        let file =
                            format!("commitment-report-405b-{name}-c{classes}-b{b_max}.json");
                        check_committed(&file, &report);
                        // RFC-0010's stages, and RISK-2's requantized elements.
                        let stages = if profile == Profile::Servable { 3 } else { 4 };
                        assert_eq!(report["lowering"]["stages"], stages, "{case}");
                        let requantized = if profile == Profile::Servable {
                            260_096
                        } else {
                            137_216
                        };
                        assert_eq!(
                            report["per_layer_per_token"]["requantized"]["total"], requantized,
                            "{case}"
                        );
                        if b_max == 32 {
                            let changed = changed_at_34(&lowering, &ranges, &selection);
                            let expected = match (profile, classes) {
                                (Profile::Servable, 4) => 16_384,
                                _ => 0,
                            };
                            assert_eq!(changed, expected, "{case}");
                        }
                    }
                    Err(CompileError::Range { node, rule, .. }) => {
                        assert_eq!(b_max, 30, "{case}: refused: {rule}");
                        let root = lowering
                            .graph
                            .nodes
                            .iter()
                            .find(|n| matches!(n.op, Op::RsqrtRow { .. }))
                            .expect("the first norm")
                            .id;
                        assert_eq!(node, root, "{case}: first refusal is the root's Range(32)");
                        assert_eq!(
                            rule, "a lookup's table has at most 2^{B_max} rows",
                            "{case}"
                        );
                    }
                    Err(error) => panic!("{case}: {error}"),
                }
            }
        }
    }
}

/// RFC-0055 'Working sets' at the projections' batch: for each profile's two-class model, the
/// report's tables at `2^14` tokens are the schedule's at 16,384 positions, stage by stage, and
/// the cost model's working sets and windows are what the schedule's own tables give: each
/// instance front-padded by `p = n_max − n_i` folds by `2^{min(n_i, c − p)}` under a window of
/// `c` rounds, and `c` is the fewest after which the folded tables fit in 0.2 TB.
#[test]
fn the_stage_working_sets_at_the_batch_are_the_schedules() {
    const BATCH_BITS: u32 = 14;
    for (profile, windows) in [
        (Profile::Minimal, vec![6, 6, 0]),
        (Profile::Servable, vec![8, 0]),
    ] {
        let (arch, quant, silu) = llama_405b(profile, 2);
        let model = PublicModel {
            arch: &arch,
            profile,
            quant: &quant,
            silu_in_exponents: &silu,
        };
        let lowering = lower(model, POSITIONS).expect("405B's shape lowers");
        let ranges = analyze(&lowering.graph, &lowering.tables).expect("ranges");
        let policy = GadgetPolicy::default();
        let selection = select(&lowering.graph, &ranges, &lowering.tables, &policy).expect("fits");
        let r = report(model, &lowering, &selection, &policy).expect("a report");
        let at_batch = lower(model, 1 << BATCH_BITS).expect("16K positions lower");
        let schedule = derive_schedule(&at_batch).expect("a schedule");
        // Each instance's variables, tables, and whether a window may fold them: a matrix
        // product's stay whole.
        let stages: Vec<Vec<(u32, u64, bool)>> = schedule
            .stages
            .iter()
            .filter(|s| s.index > 0)
            .map(|s| {
                s.instances
                    .iter()
                    .map(|i| {
                        let product = matches!(
                            i.relation,
                            Relation::MatMulWeight | Relation::MatMulAct { .. }
                        );
                        (i.vars, i.claims.len() as u64, !product)
                    })
                    .collect()
            })
            .collect();
        assert_eq!(r.stage_tables.len(), stages.len(), "{profile:?}");
        // The report's classes are the schedule's tables at the batch.
        for (reported, scheduled) in r.stage_tables.iter().zip(&stages) {
            let mut by_vars = std::collections::BTreeMap::<u32, u64>::new();
            for &(vars, tables, _) in scheduled {
                *by_vars.entry(vars).or_default() += tables;
            }
            let mut predicted = std::collections::BTreeMap::<u32, u64>::new();
            for class in &reported.tables {
                let vars = class.log2_entries + if class.per_token { BATCH_BITS } else { 0 };
                *predicted.entry(vars).or_default() += class.count;
            }
            assert_eq!(predicted, by_vars, "{profile:?}, stage {}", reported.stage);
        }
        // The cost model's plan, against the schedule's tables directly.
        let counts = Counts {
            reduction_stages: r
                .stage_tables
                .iter()
                .map(|s| ReductionStage {
                    tables: s
                        .tables
                        .iter()
                        .map(|c| StageTable {
                            per_token: c.per_token,
                            windowable: c.windowable,
                            log2_entries: c.log2_entries,
                            count: c.count as f64,
                        })
                        .collect(),
                })
                .collect(),
            ..Counts::SOURCE_405B
        };
        let memory = MemoryModel::NODE_405B;
        let plan = stage_plan(&counts, &memory, 1 << BATCH_BITS).expect("a plan");
        for ((window, scheduled), expected) in plan.iter().zip(&stages).zip(&windows) {
            let n_max = scheduled.iter().map(|&(vars, ..)| vars).max().unwrap_or(0);
            let folded = |c: u32| -> f64 {
                scheduled
                    .iter()
                    .map(|&(vars, tables, windowable)| {
                        let p = n_max - vars;
                        let bound = match windowable {
                            true => vars.min(c.saturating_sub(p)),
                            false => 0,
                        };
                        tables as f64 * (1u64 << (vars - bound)) as f64 * STAGE_BYTES_PER_ENTRY
                    })
                    .sum()
            };
            assert_eq!(window.bytes, folded(0), "{profile:?}");
            assert_eq!(window.window_rounds, *expected, "{profile:?}");
            assert!(folded(*expected) <= memory.free_bytes, "{profile:?}");
            if *expected > 0 {
                assert!(folded(expected - 1) > memory.free_bytes, "{profile:?}");
            }
        }
    }
}
