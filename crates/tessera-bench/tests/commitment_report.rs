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
//! - **The reduction stages' working sets** (RFC-0055, RFC-0056). The two-class reports' tables,
//!   evaluated at the projections' batch of 16K tokens, are the composed schedule's own at 16K
//!   positions, native relations and a request with outputs included, and the cost model's working
//!   sets and windows follow from them and the residents: 6.41 and 6.60 TB in `minimal`'s stages 1
//!   and 2, windows of 15 and 7 rounds at 16K context; 11.91 and 1.11 TB in `servable`'s, 15
//!   rounds in two windows and 3 in one.
//!
//! `docs/perf/v0.2/README.md` reads these numbers and answers OQ-0006-1.

use std::fs;
use std::path::PathBuf;

use serde_json::Value;
use tessera_bench::{
    Counts, Growth, Holding, MAX_WINDOW_VARS, MemoryModel, ModelReport, ReductionStage,
    STAGE_BYTES_PER_ENTRY, Scenario, StageTable, UnitCosts, stage_plan,
};
use tessera_codec::{Ascii, Rational, envelope, schema};
use tessera_compiler::{
    CompileError, Gadget, Lowering, ModelAdviceLayout, PublicModel, Range, Ranges, RuntimePlan,
    Selection, analyze, claim_list, lower, report,
    schedule::{NativeKind, Relation, Request, Source, derive_composed_schedule, model_natives},
    select,
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
            .map(|stage| {
                let classes = |key: &str| -> Vec<StageTable> {
                    stage[key]
                        .as_array()
                        .expect("the tables")
                        .iter()
                        .map(|t| StageTable {
                            growth: match t["growth"].as_str().expect("a growth") {
                                "fixed" => Growth::Fixed,
                                "tokens" => Growth::Tokens,
                                "keys" => Growth::Keys,
                                "tokens_and_context" => Growth::TokensAndContext,
                                other => panic!("no growth {other}"),
                            },
                            kind: match t["kind"].as_str().expect("a kind") {
                                "windowable" => Holding::Windowable,
                                "resident" => Holding::Resident,
                                "computed" => Holding::Computed,
                                other => panic!("no kind {other}"),
                            },
                            log2_entries: u32::try_from(
                                t["log2_entries"].as_u64().expect("a size"),
                            )
                            .expect("a size"),
                            count: t["count"].as_f64().expect("a count"),
                            count_per_bit: t["count_per_bit"].as_f64().expect("a count"),
                            bytes: t["bytes"].as_f64().expect("a width"),
                        })
                        .collect()
                };
                ReductionStage {
                    tables: classes("tables"),
                    residents: classes("residents"),
                }
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

/// RFC-0055 'Working sets' and RFC-0056 at the projections' batch, one sequence of 16,384
/// tokens: for each profile's two-class model, the report's tables at `2^14` tokens and context
/// are the composed schedule's at 16,384 positions, the native relations and a request with
/// outputs included and shared tables once, stage by stage.
/// The cost model's working sets and windows are what the schedule's own tables and RFC-0056's
/// residents give.
/// - **Folding.** Each windowable instance front-padded by `p = n_max − n_i` folds by
///   `2^{min(n_i, c − p)}` under a window of `c` rounds. A matrix product's tables and the
///   residents stay whole, and the score instances hold nothing.
/// - **The window.** `c` is the fewest rounds after which the held tables fit in 0.2 TB, and the
///   rounds take a window per eight own variables they bind of one instance.
#[test]
fn the_stage_working_sets_at_the_batch_are_the_schedules() {
    const BITS: u32 = 14;
    for (profile, windows) in [
        (Profile::Minimal, vec![(15, 1), (7, 1), (0, 0)]),
        (Profile::Servable, vec![(15, 2), (3, 1)]),
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
        let at_batch = lower(model, 1 << BITS).expect("16K positions lower");
        let natives = model_natives(&at_batch, Request::Generate).expect("the native relations");
        let schedule = derive_composed_schedule(&at_batch, &natives.deferred, &natives.native)
            .expect("a schedule");
        // Each instance's variables, tables, and how the prover holds them.
        let stages: Vec<Vec<(u32, u64, Holding)>> = schedule
            .stages
            .iter()
            .filter(|s| s.index > 0)
            .map(|s| {
                // Instances of one relation of one tensor at one point share their tables, and so
                // do a root's coefficient products (RFC-0056 'Shared instances').
                let mut seen = std::collections::HashSet::new();
                s.instances
                    .iter()
                    .filter(|i| match i.relation {
                        Relation::Native { tag, .. } => match natives.kinds[tag as usize - 1] {
                            NativeKind::Product => seen.insert((None, Some(i.node), None)),
                            _ => seen.insert((None, None, Some(tag))),
                        },
                        _ => seen.insert((
                            Some((i.relation, i.kind, &i.point, &i.layers, i.vars)),
                            Some(i.node),
                            None,
                        )),
                    })
                    .map(|i| match i.relation {
                        Relation::MatMulWeight | Relation::MatMulAct { .. } => {
                            (i.vars, i.claims.len() as u64, Holding::Resident)
                        }
                        Relation::Native {
                            tag, evaluations, ..
                        } => {
                            let kind = match natives.kinds[tag as usize - 1] {
                                NativeKind::Key | NativeKind::Forest => Holding::Computed,
                                _ => Holding::Windowable,
                            };
                            (i.vars, evaluations as u64, kind)
                        }
                        _ => (i.vars, i.claims.len() as u64, Holding::Windowable),
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
                *predicted
                    .entry(class.log2_entries + growth(class.growth) * BITS)
                    .or_default() += class.count + class.count_per_bit * u64::from(BITS);
            }
            assert_eq!(predicted, by_vars, "{profile:?}, stage {}", reported.stage);
        }
        // The cost model's plan, against the schedule's tables and the residents directly.
        let counts = report_counts(&r);
        let memory = MemoryModel::NODE_405B;
        let plan = stage_plan(&counts, &memory, 1 << BITS, 1 << BITS).expect("a plan");
        for (((window, scheduled), expected), reported) in
            plan.iter().zip(&stages).zip(&windows).zip(&r.stage_tables)
        {
            let n_max = scheduled.iter().map(|&(vars, ..)| vars).max().unwrap_or(0);
            let residents: f64 = reported
                .residents
                .iter()
                .map(|c| {
                    let vars = c.log2_entries + growth(c.growth) * BITS;
                    let count = c.count + c.count_per_bit * u64::from(BITS);
                    count as f64 * (1u64 << vars) as f64 * f64::from(c.bytes)
                })
                .sum();
            let bound = |c: u32, vars: u32, kind| match kind {
                Holding::Windowable => vars.min(c.saturating_sub(n_max - vars)),
                Holding::Resident | Holding::Computed => 0,
            };
            let folded = |c: u32| -> f64 {
                residents
                    + scheduled
                        .iter()
                        .filter(|&&(.., kind)| kind != Holding::Computed)
                        .map(|&(vars, tables, kind)| {
                            let entries = (1u64 << (vars - bound(c, vars, kind))) as f64;
                            tables as f64 * entries * STAGE_BYTES_PER_ENTRY
                        })
                        .sum::<f64>()
            };
            assert_eq!(window.bytes, folded(0), "{profile:?}");
            assert_eq!(
                (window.window_rounds, window.windows),
                *expected,
                "{profile:?}"
            );
            let own = scheduled
                .iter()
                .map(|&(vars, _, kind)| bound(window.window_rounds, vars, kind))
                .max()
                .unwrap_or(0);
            assert_eq!(window.own_vars, own, "{profile:?}");
            assert_eq!(window.windows, own.div_ceil(MAX_WINDOW_VARS), "{profile:?}");
            let c = window.window_rounds;
            assert!(folded(c) <= memory.free_bytes, "{profile:?}");
            if c > 0 {
                assert!(folded(c - 1) > memory.free_bytes, "{profile:?}");
            }
        }
    }
}

/// The variables a class gains per bit of a batch of as many tokens as context positions.
fn growth(growth: tessera_compiler::Growth) -> u32 {
    match growth {
        tessera_compiler::Growth::Fixed => 0,
        tessera_compiler::Growth::Tokens | tessera_compiler::Growth::Keys => 1,
        tessera_compiler::Growth::TokensAndContext => 2,
    }
}

/// The cost model's counts of a report's reduction stages: its tables and its residents.
fn report_counts(r: &tessera_compiler::CommitmentReport) -> Counts {
    Counts {
        reduction_stages: r
            .stage_tables
            .iter()
            .map(|s| {
                let classes = |classes: &[tessera_compiler::TableClass]| -> Vec<StageTable> {
                    classes
                        .iter()
                        .map(|c| StageTable {
                            growth: match c.growth {
                                tessera_compiler::Growth::Fixed => Growth::Fixed,
                                tessera_compiler::Growth::Tokens => Growth::Tokens,
                                tessera_compiler::Growth::Keys => Growth::Keys,
                                tessera_compiler::Growth::TokensAndContext => {
                                    Growth::TokensAndContext
                                }
                            },
                            kind: match c.kind {
                                tessera_compiler::Holding::Windowable => Holding::Windowable,
                                tessera_compiler::Holding::Resident => Holding::Resident,
                                tessera_compiler::Holding::Computed => Holding::Computed,
                            },
                            log2_entries: c.log2_entries,
                            count: c.count as f64,
                            count_per_bit: c.count_per_bit as f64,
                            bytes: f64::from(c.bytes),
                        })
                        .collect()
                };
                ReductionStage {
                    tables: classes(&s.tables),
                    residents: classes(&s.residents),
                }
            })
            .collect(),
        ..Counts::SOURCE_405B
    }
}

/// RFC-0014's default statement counts the 405B leaf stage's merge in token groups (RFC-0054
/// 'Soundness'): `P = 53,491` merged polynomials (`53,490` for the model alone), the lowering's
/// split values, the generation request's greedy key column (RFC-0053) and the `SoftmaxRow`
/// forest, and `n_max = 42`, the forest leaf table's variables at 16K tokens. Both workspaces'
/// layouts count the polynomials.
#[test]
fn the_default_statements_merge_sizes_are_the_405b_lowerings() {
    let (arch, quant, silu) = llama_405b(Profile::Servable, 2);
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
    let selection = select(&lowering.graph, &ranges, &lowering.tables, &policy).expect("fits");
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
    // The prover's merge polynomials at the batch's 16K positions: one per split value.
    let full = lower(model, 1 << 14).unwrap();
    let layers = tessera_compiler::schedule::LayerMap::new(&full).unwrap();
    let leaf = tessera_prover::LeafLayout::model(&full, &layers, &selection, &claims).unwrap();
    let one_layer_bit = tessera_protocol::LeafGroups {
        token_bits: 0,
        layer_bits: 1,
    };
    let grouped = tessera_prover::leaf::GroupedLayout::new(&leaf, &layers, one_layer_bit).unwrap();
    let splits = grouped.polynomials();
    assert_eq!(splits.len(), 53_489);
    assert_eq!(splits.iter().map(|p| p.vars).max(), Some(35));
    // The verifier's, from its own lowering: the split values and the one forest.
    let bytes = envelope::seal(&schema::CLAIM_LIST, &claims);
    let (independent, validated) =
        tv_core::validation::load(&bytes, &tv_codec::DecodeLimits::MESSAGE).unwrap();
    let tv_one_layer_bit = tv_core::protocol::LeafGroups {
        token_bits: 0,
        layer_bits: 1,
    };
    let verified = tv_core::leaf::groups::Grouped::derive(
        validated.graph(),
        independent.gadget_policy,
        tv_one_layer_bit,
    )
    .unwrap();
    let forests = verified
        .polynomials()
        .iter()
        .filter(|p| matches!(p.source, tv_core::leaf::groups::Source::Forest { .. }))
        .count();
    assert_eq!((verified.polynomials().len(), forests), (53_490, 1));
    // With a generation request (RFC-0053): GreedyKey adds one 31-variable split polynomial and
    // an implicit Range(29) table (27 tables in all, 2^{31} entries), giving P = 53,491.
    let runtime = RuntimePlan::derive(&full, &selection, &claims, 1, 0).unwrap();
    assert_eq!(runtime.selection().tables.len(), 27);
    let cert = runtime.certificate().expect("generation certificate");
    assert_eq!((cert.range_bits, cert.entries()), (29, 1 << 31));
    let runtime_leaf = tessera_prover::LeafLayout::runtime(&full, &layers, &runtime, &claims)
        .expect("runtime leaf");
    let runtime_grouped =
        tessera_prover::leaf::GroupedLayout::new(&runtime_leaf, &layers, one_layer_bit).unwrap();
    assert_eq!(runtime_grouped.polynomials().len(), 53_490);
    let tv_public = tv_core::validation::check_public(&independent).unwrap();
    let tv_graph = tv_core::lowering::lower(&tv_public, 1 << 14).unwrap();
    let tv_runtime = tv_core::gadgets::runtime::Plan::derive(
        &tv_graph,
        independent.gadget_policy,
        1,
        0,
        validated.tables(),
    )
    .unwrap();
    assert_eq!(tv_runtime.catalog().len(), 27);
    let tv_cert = tv_runtime.certificate().expect("generation certificate");
    assert_eq!(tv_cert.entries(), 1 << 31);
    let tv_runtime_grouped =
        tv_core::leaf::groups::Grouped::derive_runtime(&tv_graph, &tv_runtime, tv_one_layer_bit)
            .unwrap();
    let tv_runtime_forests = tv_runtime_grouped
        .polynomials()
        .iter()
        .filter(|p| matches!(p.source, tv_core::leaf::groups::Source::Forest { .. }))
        .count();
    assert_eq!(
        (tv_runtime_grouped.polynomials().len(), tv_runtime_forests),
        (53_491, 1)
    );
    // The default statement: P = 53,491, and n_max the forest's 2^28 trees of 2^14 leaves.
    let groups = tessera_protocol::StatementSizes::llama_405b_16k().groups;
    assert_eq!(groups.merge_claims, 53_491);
    assert_eq!(groups.merge_rounds, 28 + 14);
}

/// RFC-0014's default statement counts the composed schedule's instances per reduction stage
/// (`derive_composed_schedule` with `model_natives`, #566): `[123, 98]` with a generation request
/// (90 of stage 1's RFC-0050's root products, 90 of stage 2's the norms' `SumSquares`), `[122, 97]`
/// for prefill, and `[122, 96]` for the model alone, which both workspaces' schedules derive. In
/// `tessera-protocol/2` the layer sumcheck batches 36 column claims (RFC-0068) and the masks'
/// batching 2,668 mask claims (RFC-0070).
#[test]
fn the_default_statements_stage_sizes_are_the_405b_schedules() {
    let (arch, quant, silu) = llama_405b(Profile::Servable, 2);
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
    let selection = select(&lowering.graph, &ranges, &lowering.tables, &policy).expect("fits");
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
    let full = lower(model, 1 << 14).unwrap();
    for (request, expected, expected_folds, expected_zeros) in [
        (
            Request::None,
            [122usize, 96],
            [16usize, 96],
            (13usize, 362u32, 35u32),
        ),
        (Request::Prefill, [122, 97], [15, 97], (14, 393, 35)),
        (Request::Generate, [123, 98], [15, 98], (14, 393, 35)),
    ] {
        let natives = model_natives(&full, request).unwrap();
        let schedule = derive_composed_schedule(&full, &natives.deferred, &natives.native).unwrap();
        let counts: Vec<usize> = schedule.stages[1..]
            .iter()
            .map(|s| s.instances.len())
            .collect();
        assert_eq!(counts, expected, "405B servable {request:?}");
        let zeros: Vec<u32> = schedule.stages[1..]
            .iter()
            .flat_map(|s| &s.instances)
            .filter(|i| {
                matches!(i.source, Source::ZeroCheck)
                    || matches!(i.relation, Relation::Native { zero: true, .. })
            })
            .map(|i| i.vars)
            .collect();
        assert_eq!(
            (
                zeros.len(),
                zeros.iter().sum::<u32>(),
                zeros.into_iter().max().unwrap()
            ),
            expected_zeros,
            "405B servable zero-checks {request:?}"
        );
        let products = schedule.stages[1]
            .instances
            .iter()
            .filter(|i| {
                matches!(
                    i.relation,
                    Relation::Native { tag, .. }
                        if natives.kinds[tag as usize - 1] == NativeKind::Product
                )
            })
            .count();
        let sum_squares = schedule.stages[2]
            .instances
            .iter()
            .filter(|i| i.relation == Relation::SumSquares)
            .count();
        assert_eq!((products, sum_squares), (90, 90), "{request:?}");
        let degrees =
            tessera_compiler::schedule::protocol2_stages(&full.graph, &schedule, 22).unwrap();
        let ds: Vec<(u32, usize)> = degrees.iter().map(|s| (s.rounds, s.d)).collect();
        assert_eq!(ds, [(42, 10), (35, 7)], "{request:?}");
        let folds: Vec<usize> = degrees.iter().map(|s| s.folds).collect();
        assert_eq!(folds, expected_folds, "405B servable folds {request:?}");
    }
    // 405B minimal at 16K positions: three reduction stages.
    let (m_arch, m_quant, m_silu) = llama_405b(Profile::Minimal, 2);
    let m_full = lower(
        PublicModel {
            arch: &m_arch,
            profile: Profile::Minimal,
            quant: &m_quant,
            silu_in_exponents: &m_silu,
        },
        1 << 14,
    )
    .unwrap();
    for (request, expected) in [
        (Request::None, [119usize, 97, 2]),
        (Request::Prefill, [119, 98, 2]),
        (Request::Generate, [120, 99, 2]),
    ] {
        let natives = model_natives(&m_full, request).unwrap();
        let schedule =
            derive_composed_schedule(&m_full, &natives.deferred, &natives.native).unwrap();
        let counts: Vec<usize> = schedule.stages[1..]
            .iter()
            .map(|s| s.instances.len())
            .collect();
        assert_eq!(counts, expected, "405B minimal {request:?}");
    }
    // The verifier's independent schedule for the 405B claim list at 16K positions.
    let bytes = envelope::seal(&schema::CLAIM_LIST, &claims);
    let (independent, validated) =
        tv_core::validation::load(&bytes, &tv_codec::DecodeLimits::MESSAGE).unwrap();
    let tv_public = tv_core::validation::check_public(&independent).unwrap();
    let tv_graph = tv_core::lowering::lower(&tv_public, 1 << 14).unwrap();
    let tv_gadgets = tv_core::gadgets::Plan::derive(&tv_graph, independent.gadget_policy).unwrap();
    let tv_plan = tv_core::gadgets::model::Plan::derive(&tv_gadgets).unwrap();
    let tv_counts: Vec<usize> = tv_plan
        .schedule()
        .stages()
        .iter()
        .map(|s| s.instances.len())
        .collect();
    assert_eq!(tv_counts, [122, 96]);
    let tv_runtime = tv_core::gadgets::runtime::Plan::derive(
        &tv_graph,
        independent.gadget_policy,
        1,
        0,
        validated.tables(),
    )
    .unwrap();
    let tv_runtime_plan = tv_core::gadgets::model::Plan::derive_runtime(&tv_runtime).unwrap();
    let tv_runtime_counts: Vec<usize> = tv_runtime_plan
        .schedule()
        .stages()
        .iter()
        .map(|s| s.instances.len())
        .collect();
    assert_eq!(tv_runtime_counts, [123, 98]);
    // tessera-protocol/2's additional batched sumchecks on the 405B model layout (RFC-0068,
    // RFC-0070): 25 active lookup instances (26 tables with RoPE's unused entry), 36 column
    // claims in the layer sumcheck, and 2,668 mask claims in the masks' batching.
    let layers = tessera_compiler::schedule::LayerMap::new(&full).unwrap();
    let leaf = tessera_prover::LeafLayout::model(&full, &layers, &selection, &claims).unwrap();
    assert_eq!((selection.tables.len(), leaf.instances().len()), (26, 25));
    let value_columns = leaf
        .instances()
        .iter()
        .filter(|inst| {
            inst.blocks.first().is_some_and(|b| {
                leaf.sites()[b.site].layers[b.layer].lookups[b.lookup]
                    .lookup
                    .value
                    .is_some()
            })
        })
        .count();
    let layer_claims = leaf.instances().len() + value_columns;
    assert_eq!(layer_claims, 36);
    let fraction_masks: u32 = leaf
        .instances()
        .iter()
        .map(|inst| {
            let log2_rows = u32::from(
                selection
                    .tables
                    .iter()
                    .find(|t| t.id == inst.table)
                    .unwrap()
                    .log2_rows,
            );
            2 * inst.height.max(2) + 2 * log2_rows.max(2)
        })
        .sum();
    let softmax_level_masks = 14 - 1;
    let masked_sumchecks = 1 + 42 + 1 + 2;
    // LeafLayout::model lays out the 126 real layers (table 4 at height 30); StatementSizes'
    // first 26 lookups add SoftmaxRow's two padded layers (table 4 at height 31), giving
    // RFC-0070's 2,621 tree-level mask claims and 2,668 mask claims in all, while its 27th
    // entry is tessera-protocol/1's runtime greedy Range(29) (RFC-0053).
    assert_eq!(fraction_masks + softmax_level_masks, 2_619);
    let statement_lookups = tessera_protocol::StatementSizes::llama_405b_16k().lookups;
    assert_eq!(statement_lookups.len(), 27);
    assert_eq!(
        statement_lookups[26],
        tessera_protocol::LookupSizes {
            lookups: 1 << 31,
            rows: 1 << 29,
        }
    );
    let padded_fraction_masks: u32 = statement_lookups[..26]
        .iter()
        .filter(|l| l.lookups > 0)
        .map(|l| {
            let h_l = (u64::BITS - (l.lookups - 1).leading_zeros()).max(2);
            let h_t = (u64::BITS - (l.rows - 1).leading_zeros()).max(2);
            2 * h_l + 2 * h_t
        })
        .sum();
    assert_eq!(padded_fraction_masks + softmax_level_masks, 2_621);
    assert_eq!(masked_sumchecks + 2_621 + 1, 2_668);
    // The four conformance claim lists at max_context.
    let claim_lists_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../spec/vectors/claims/claim-lists.json");
    let claim_lists: Value =
        serde_json::from_str(&fs::read_to_string(claim_lists_path).unwrap()).unwrap();
    for (name, expected) in [
        (
            "tiny-2-servable",
            [vec![122usize, 96], vec![122, 97], vec![123, 98]],
        ),
        (
            "tiny-4-servable",
            [vec![122, 96], vec![122, 97], vec![123, 98]],
        ),
        (
            "tiny-2-minimal",
            [vec![119, 97, 2], vec![119, 98, 2], vec![120, 99, 2]],
        ),
        (
            "tiny-2-affine",
            [vec![122, 98, 2], vec![122, 99, 2], vec![123, 100, 2]],
        ),
    ] {
        let hex = claim_lists["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == name)
            .unwrap()["hex"]
            .as_str()
            .unwrap();
        let raw: Vec<u8> = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect();
        let list: tessera_ir::ClaimList = envelope::open(
            &raw,
            &schema::CLAIM_LIST,
            tessera_codec::DecodeLimits::MESSAGE,
        )
        .unwrap();
        let c_full = lower(PublicModel::of_claims(&list), list.arch.max_context).unwrap();
        for (request, want) in [Request::None, Request::Prefill, Request::Generate]
            .into_iter()
            .zip(expected)
        {
            let natives = model_natives(&c_full, request).unwrap();
            let schedule =
                derive_composed_schedule(&c_full, &natives.deferred, &natives.native).unwrap();
            let got: Vec<usize> = schedule.stages[1..]
                .iter()
                .map(|s| s.instances.len())
                .collect();
            assert_eq!(got, want, "{name} {request:?}");
        }
    }
    // Both calculators' default 405B statements hold the composed schedule's generation counts.
    let prover_default = tessera_protocol::StatementSizes::llama_405b_16k();
    assert_eq!(prover_default.stages, vec![123, 98]);
    assert_eq!(prover_default.zero_checks, vec![42; 14]);
    assert_eq!(prover_default.tree_levels, vec![55; 42]);
    assert_eq!(prover_default.lookups.len(), 27);
    assert_eq!(prover_default.groups.merge_claims, 53_491);
    let tv_default = tv_core::soundness::StatementSizes::default_405b();
    assert_eq!(tv_default.stages, vec![123, 98]);
    assert_eq!(tv_default.zero_checks, vec![42; 14]);
    assert_eq!(tv_default.tree_levels, vec![55; 42]);
    assert_eq!(tv_default.lookups.len(), 27);
    assert_eq!(tv_default.groups.merge_claims, 53_491);
}
