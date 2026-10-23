// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The cost model's properties (RFC-0020 'Testing Strategy'): `μ` rises with every unit cost and
//! every count of proof work, and with the native rate it is measured against, and falls with
//! native work, each input moving the terms it enters and no other; the streaming term is zero
//! while a batch's leaves fit in memory and steps up one pass per doubling of them beyond it;
//! the column answers and the row combination price every committed element; each table row
//! costs what a token's committed lookup costs, once per batch; a reduction stage larger than
//! memory takes the fewest window rounds after which its tables fit, front padding counted, and
//! one fold pass; the corrected projections the documents quote are the model's; and an input
//! the model cannot price is refused, never turned into a number.

use tessera_bench::{
    Counts, MemoryModel, ModelError, ModelReport, MuBreakdown, Problem, ReductionStage, Scenario,
    StageTable, TENSOR_BYTES_PER_LEAF, UnitCosts, detection_probability, leaf_plan, mu,
    proof_time_s, sampling_rate, shared_streaming_passes, stage_plan,
};

/// A tables class of a reduction stage.
fn tables(per_token: bool, log2_entries: u32, count: f64) -> StageTable {
    StageTable {
        per_token,
        windowable: true,
        log2_entries,
        count,
    }
}

/// A class of a matrix product's tables, which no window folds.
fn resident(per_token: bool, log2_entries: u32, count: f64) -> StageTable {
    StageTable {
        windowable: false,
        ..tables(per_token, log2_entries, count)
    }
}

/// The source's model with the table trees' streamed rounds priced at 7 ps per leaf, the merge
/// at 3 ps per element (RFC-0054) and a window's fold at 1.5 ps per entry (RFC-0055), column
/// answers at 15 ps and the row combination at 2 ps per committed element, five regenerations,
/// 2 · 10^10 table rows, a score block of 512 (RFC-0034) and one reduction stage of two
/// `2^23`-entry tables per token, 4.4 TB at 16K tokens, so that every term is positive: the
/// batch's leaf stage is grouped at every context, its table trees stream, and its reduction
/// stage is windowed.
fn every_term() -> (Counts, UnitCosts, MemoryModel) {
    let counts = Counts {
        regen_full_equivalents: 5.0,
        table_rows: 2e10,
        score_block: 512.0,
        reduction_stages: vec![ReductionStage {
            tables: vec![tables(true, 23, 2.0), tables(false, 21, 4.0)],
        }],
        ..Counts::SOURCE_405B
    };
    let costs = UnitCosts {
        streaming_pass_ps: 7.0,
        merge_ps: 3.0,
        window_ps: 1.5,
        column_answer_ps: 15.0,
        row_combination_ps: 2.0,
        ..UnitCosts::SOURCE
    };
    (counts, costs, MemoryModel::NODE_405B)
}

const CONTEXTS: [u32; 3] = [1024, 16_384, 131_072];
const BATCH: u32 = Scenario::SOURCE.batch_tokens;

fn terms(m: &MuBreakdown) -> [f64; 8] {
    [
        m.linear,
        m.attention,
        m.weights,
        m.regen,
        m.streaming,
        m.column_answers,
        m.row_combination,
        m.tables,
    ]
}

/// Each term after one input moved, against before: up, down or unchanged.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Move {
    Up,
    Down,
    Same,
}

fn moves(before: &MuBreakdown, after: &MuBreakdown) -> [Move; 8] {
    let (b, a) = (terms(before), terms(after));
    core::array::from_fn(|t| match a[t].partial_cmp(&b[t]) {
        Some(core::cmp::Ordering::Greater) => Move::Up,
        Some(core::cmp::Ordering::Less) => Move::Down,
        _ => Move::Same,
    })
}

#[test]
fn mu_rises_with_every_cost_and_count_of_proof_work_and_falls_with_native_work() {
    use Move::{Down, Same, Up};
    let (counts, costs, memory) = every_term();
    // Each input raised by half, with the terms it must move: linear, attention, weights, regen,
    // streaming, column answers, row combination, tables, in that order.
    type Raise = (&'static str, fn(&mut Counts, &mut UnitCosts), [Move; 8]);
    let raises: [Raise; 20] = [
        (
            "commit_ps",
            |_, u| u.commit_ps *= 1.5,
            [Up, Up, Same, Same, Same, Same, Same, Up],
        ),
        (
            "lookup_ps",
            |_, u| u.lookup_ps *= 1.5,
            [Up, Up, Same, Same, Same, Same, Same, Up],
        ),
        (
            "product_leaf_ps",
            |_, u| u.product_leaf_ps *= 1.5,
            [Same, Up, Same, Same, Same, Same, Same, Same],
        ),
        (
            "glue_fraction",
            |_, u| u.glue_fraction *= 1.5,
            [Up, Up, Same, Same, Same, Same, Same, Up],
        ),
        (
            "weight_pass_gpu_s",
            |_, u| u.weight_pass_gpu_s *= 1.5,
            [Same, Same, Up, Same, Same, Same, Same, Same],
        ),
        (
            "streaming_pass_ps",
            |_, u| u.streaming_pass_ps *= 1.5,
            [Same, Same, Same, Same, Up, Same, Same, Same],
        ),
        (
            "merge_ps",
            |_, u| u.merge_ps *= 1.5,
            [Same, Same, Same, Same, Up, Same, Same, Same],
        ),
        (
            "window_ps",
            |_, u| u.window_ps *= 1.5,
            [Same, Same, Same, Same, Up, Same, Same, Same],
        ),
        (
            "column_answer_ps",
            |_, u| u.column_answer_ps *= 1.5,
            [Same, Same, Same, Same, Same, Up, Same, Same],
        ),
        (
            "row_combination_ps",
            |_, u| u.row_combination_ps *= 1.5,
            [Same, Same, Same, Same, Same, Same, Up, Same],
        ),
        // A faster native path makes the same proof a larger multiple of it.
        (
            "native_ops_per_s",
            |_, u| u.native_ops_per_s *= 1.5,
            [Up, Up, Up, Same, Up, Up, Up, Up],
        ),
        (
            "commits_per_token",
            |c, _| c.commits_per_token *= 1.5,
            [Up, Same, Same, Same, Same, Up, Up, Same],
        ),
        (
            "lookups_per_token",
            |c, _| c.lookups_per_token *= 1.5,
            [Up, Same, Same, Same, Up, Same, Same, Same],
        ),
        (
            "scores_per_token_per_ctx",
            |c, _| c.scores_per_token_per_ctx *= 1.5,
            [Same, Up, Same, Same, Up, Up, Up, Same],
        ),
        (
            "native_ops_lin",
            |c, _| c.native_ops_lin *= 1.5,
            [Down, Down, Down, Same, Down, Down, Down, Down],
        ),
        (
            "native_ops_att_per_ctx",
            |c, _| c.native_ops_att_per_ctx *= 1.5,
            [Down, Down, Down, Same, Down, Down, Down, Down],
        ),
        (
            "regen_full_equivalents",
            |c, _| c.regen_full_equivalents *= 1.5,
            [Same, Same, Same, Up, Same, Same, Same, Same],
        ),
        (
            "score_block",
            |c, _| c.score_block *= 1.5,
            [Same, Up, Same, Same, Same, Up, Up, Same],
        ),
        (
            "table_rows",
            |c, _| c.table_rows *= 1.5,
            [Same, Same, Same, Same, Up, Same, Same, Up],
        ),
        // A windowed stage's larger tables: more entries to fold, the same one fold pass.
        (
            "reduction stage tables",
            |c, _| c.reduction_stages[0].tables[0].count *= 1.5,
            [Same, Same, Same, Same, Up, Same, Same, Same],
        ),
    ];
    for (name, raise, expected) in raises {
        for context in CONTEXTS {
            let before = mu(&counts, &costs, &memory, context, BATCH).expect("a projection");
            let (mut c, mut u) = (counts.clone(), costs);
            raise(&mut c, &mut u);
            let after = mu(&c, &u, &memory, context, BATCH).expect("a projection");
            assert_eq!(moves(&before, &after), expected, "{name} at {context}");
            let total_moves = expected.contains(&Up) as i8 - expected.contains(&Down) as i8;
            match total_moves {
                1 => assert!(after.total > before.total, "{name} at {context}: μ rises"),
                -1 => assert!(after.total < before.total, "{name} at {context}: μ falls"),
                _ => unreachable!("every input moves μ one way"),
            }
        }
    }
}

/// RFC-0054's plan: one batch while the lookup trees and product forests fit, and one more
/// doubling of the token groups per doubling of their leaves beyond the memory, whatever doubles
/// them. A grouped stage pays one merge, a regeneration and its folds, however many groups it
/// has; the table trees stream their levels that do not fit, `c(c + 1)/2` rounds, and regenerate
/// nothing.
#[test]
fn the_leaf_plan_doubles_its_groups_per_doubling_of_the_leaves() {
    let (counts, costs, _) = every_term();
    // The leaf stage alone: no table rows, and no reduction stage to window.
    let no_tables = Counts {
        table_rows: 0.0,
        reduction_stages: Vec::new(),
        ..counts
    };
    let context = 4096;
    let leaves = (no_tables.lookups_per_token
        + 2.0 * no_tables.scores_per_token_per_ctx * f64::from(context))
        * f64::from(BATCH);
    // A group keeps its level tables, 32 bytes, and the tensor value each leaf reads.
    let exact = MemoryModel {
        free_bytes: leaves * (32.0 + TENSOR_BYTES_PER_LEAF),
        bytes_per_leaf: 32.0,
    };
    let below = MemoryModel {
        free_bytes: exact.free_bytes * (1.0 - 1e-9),
        ..exact
    };
    let plan =
        |c: &Counts, m: &MemoryModel, batch: u32| leaf_plan(c, m, context, batch).expect("a plan");
    let fits = plan(&no_tables, &exact, BATCH);
    assert_eq!(
        (fits.groups, fits.table_rounds, fits.merge_passes()),
        (1, 0, 0)
    );
    let split = plan(&no_tables, &below, BATCH);
    assert_eq!(
        (split.groups, split.merge_passes()),
        (2, 1),
        "one byte short"
    );
    let terms = |c: &Counts, m: &MemoryModel| {
        let m = mu(c, &costs, m, context, BATCH).expect("a projection");
        (m.regen, m.streaming)
    };
    let regenerations = no_tables.regen_full_equivalents;
    assert_eq!(
        terms(&no_tables, &exact),
        (regenerations, 0.0),
        "nothing to merge or stream"
    );
    // The merge: one more regeneration, and 3 ps per merged element, an element per leaf.
    let native = (no_tables.native_ops_lin + no_tables.native_ops_att_per_ctx * f64::from(context))
        / costs.native_ops_per_s;
    let folds = costs.merge_ps * 1e-12 * leaves / f64::from(BATCH) / native;
    let merged = |m: &MemoryModel| {
        let (regen, streaming) = terms(&no_tables, m);
        regen == regenerations + 1.0 && (streaming - folds).abs() < 1e-12 * folds
    };
    assert!(merged(&below));
    for k in 1..=12u32 {
        let memory = MemoryModel {
            free_bytes: exact.free_bytes / f64::from(1u32 << k),
            ..exact
        };
        assert_eq!(
            plan(&no_tables, &memory, BATCH).groups,
            1 << k,
            "memory / 2^{k}"
        );
        assert_eq!(
            plan(&no_tables, &exact, BATCH << k).groups,
            1 << k,
            "batch 2^{k}·T"
        );
        // However many groups, one merge.
        assert!(merged(&memory), "memory / 2^{k}");
    }
    // The table trees stream c(c + 1)/2 rounds: beside the lookups, which fill half of twice the
    // memory, while those fit; alone once the stage is grouped.
    let rows = |bytes: f64| Counts {
        table_rows: bytes / 32.0,
        ..no_tables.clone()
    };
    let free = exact.free_bytes;
    let ungrouped = MemoryModel {
        free_bytes: 2.0 * free,
        ..exact
    };
    for (tables, rounds) in [(0.5, 0), (1.5, 1), (3.5, 3), (7.5, 6), (15.5, 10)] {
        let beside = plan(&rows(tables * free), &ungrouped, BATCH);
        assert_eq!(
            (beside.groups, beside.table_rounds),
            (1, rounds),
            "{tables}·M beside"
        );
    }
    for (tables, rounds) in [(0.5, 0), (1.5, 1), (3.5, 3), (7.5, 6), (15.5, 10)] {
        let alone = plan(&rows(tables * free), &below, BATCH);
        assert_eq!(
            (alone.groups, alone.table_rounds),
            (2, rounds),
            "{tables}·M alone"
        );
    }
    // RFC-0018 as written: one batch, every level of every tree streaming, one regeneration per
    // round: c(c + 1)/2 with c the leaf level's.
    for k in 0..=12u32 {
        let memory = MemoryModel {
            free_bytes: exact.free_bytes / f64::from(1u32 << k),
            ..exact
        };
        assert_eq!(
            shared_streaming_passes(&no_tables, &memory, context, BATCH),
            Ok(k * (k + 1) / 2),
            "memory / 2^{k}"
        );
    }
}

/// The column answers evaluate every element the witness pass commits, `C` per token in the
/// linear layers and one per score, and nothing else; so does the row combination, which folds
/// each into the opening's combined rows. Priced at the commitment's cost, with lookups,
/// product leaves and glue free, each costs exactly what committing does.
#[test]
fn the_column_answers_and_the_row_combination_price_every_committed_element() {
    let (counts, _, memory) = every_term();
    let free = UnitCosts {
        lookup_ps: 0.0,
        product_leaf_ps: 0.0,
        glue_fraction: 0.0,
        ..UnitCosts::SOURCE
    };
    // A term's name, the unit costs that price it at c_c, and the term.
    type Priced = (&'static str, UnitCosts, fn(&MuBreakdown) -> f64);
    let priced: [Priced; 2] = [
        (
            "column answers",
            UnitCosts {
                column_answer_ps: UnitCosts::SOURCE.commit_ps,
                ..free
            },
            |m| m.column_answers,
        ),
        (
            "row combination",
            UnitCosts {
                row_combination_ps: UnitCosts::SOURCE.commit_ps,
                ..free
            },
            |m| m.row_combination,
        ),
    ];
    for (name, costs, term) in priced {
        for context in CONTEXTS {
            let m = mu(&counts, &costs, &memory, context, BATCH).expect("a projection");
            let committing = m.linear + m.attention;
            assert!(
                (term(&m) - committing).abs() <= 1e-12 * committing,
                "{name} at {context}: {} against {committing}",
                term(&m)
            );
        }
    }
}

/// Each table row costs one committed multiplicity and one fraction-tree leaf, priced as a lookup:
/// with as many lookups as commitments per token, the table term is the linear term times
/// `M / (T · C)`, `T` the batch's tokens, so doubling the batch halves it. Without table rows, as
/// in the source's counts, it is zero.
#[test]
fn each_table_row_costs_a_committed_lookup_once_per_batch() {
    let (counts, costs, memory) = every_term();
    let counts = Counts {
        lookups_per_token: counts.commits_per_token,
        ..counts
    };
    for context in CONTEXTS {
        for batch in [BATCH, 2 * BATCH] {
            let m = mu(&counts, &costs, &memory, context, batch).expect("a projection");
            let ratio = counts.table_rows / (f64::from(batch) * counts.commits_per_token);
            let expected = m.linear * ratio;
            assert!(
                (m.tables - expected).abs() <= 1e-12 * expected,
                "at {context}, batch {batch}: {} against {expected}",
                m.tables
            );
        }
        let none = Counts {
            table_rows: 0.0,
            ..counts.clone()
        };
        let m = mu(&none, &costs, &memory, context, BATCH).expect("a projection");
        assert_eq!(m.tables, 0.0);
    }
}

/// RFC-0055's plan: a reduction stage whose tables fit takes no window; a larger one takes the
/// fewest rounds `c` after which its tables, each folded by `2^{min(n, c − p)}` with `p` its front
/// padding, fit, and one fold pass, a regeneration counted when the model counts any, with its
/// folds at `window_ps` per entry. Per-token tables grow with the batch, padded to a power of two,
/// and fixed ones do not.
#[test]
fn a_stage_takes_the_fewest_window_rounds_after_which_its_tables_fit() {
    let stage = |classes: Vec<StageTable>| Counts {
        regen_full_equivalents: 5.0,
        reduction_stages: vec![ReductionStage { tables: classes }],
        ..Counts::SOURCE_405B
    };
    let memory = |free_bytes: f64| MemoryModel {
        free_bytes,
        ..MemoryModel::NODE_405B
    };
    let window = |counts: &Counts, free: f64, batch: u32| {
        stage_plan(counts, &memory(free), batch).map(|p| (p[0].bytes, p[0].window_rounds))
    };
    // One table of 2^20 entries per token: 2^34 entries, 2^38 bytes, at 2^14 tokens. Each halving
    // of the memory adds a round.
    let one = stage(vec![tables(true, 20, 1.0)]);
    let bytes = f64::from(2u8).powi(38);
    for k in 0..=12 {
        let free = bytes / f64::from(1u32 << k);
        assert_eq!(
            window(&one, free, 1 << 14),
            Ok((bytes, k)),
            "memory / 2^{k}"
        );
    }
    // A batch is padded to a power of two: 3·2^12 tokens hold what 2^14 do.
    assert_eq!(window(&one, bytes / 2.0, 3 << 12), Ok((bytes, 1)));
    // Front padding. Beside it, four tables of 2^18 entries per token are padded by 2: a window
    // of c rounds folds them by 2^{c − 2} only. Half the memory takes three rounds, not one.
    let padded = stage(vec![tables(true, 20, 1.0), tables(true, 18, 4.0)]);
    assert_eq!(window(&padded, bytes, 1 << 14), Ok((2.0 * bytes, 3)));
    assert_eq!(window(&padded, 2.0 * bytes, 1 << 14), Ok((2.0 * bytes, 0)));
    // Fixed tables do not grow with the batch.
    let fixed = stage(vec![tables(false, 34, 1.0)]);
    for batch in [1u32 << 10, 1 << 14, 1 << 17] {
        assert_eq!(
            window(&fixed, bytes / 4.0, batch),
            Ok((bytes, 2)),
            "{batch}"
        );
    }
    // A window stops one round short of the stage's last, so that every windowed table keeps a
    // round in memory (RFC-0013's masked last round): its 33 rounds leave the five tables two
    // entries each, 160 bytes. Less memory has no plan.
    assert_eq!(window(&padded, 160.0, 1 << 14), Ok((2.0 * bytes, 33)));
    let too_small = Err(ModelError::Input {
        name: "free_bytes",
        problem: Problem::TooSmall,
    });
    assert_eq!(window(&padded, 159.0, 1 << 14), too_small);
    // A matrix product's tables stay whole: a stage of them alone has no window, and beside
    // windowed ones they count in full, so the window must fold the others further.
    let product = stage(vec![resident(true, 20, 1.0)]);
    assert_eq!(window(&product, bytes, 1 << 14), Ok((bytes, 0)));
    assert_eq!(window(&product, bytes / 2.0, 1 << 14), too_small);
    let beside = stage(vec![tables(true, 20, 1.0), resident(false, 30, 1.0)]);
    let whole = 16.0 * f64::from(2u8).powi(30);
    assert_eq!(
        window(&beside, whole + bytes / 4.0, 1 << 14),
        Ok((bytes + whole, 2))
    );
    // Only the windowed tables' entries are folded and priced.
    let plan = stage_plan(&beside, &memory(whole + bytes / 4.0), 1 << 14).expect("a plan");
    assert_eq!(plan[0].windowed_entries, f64::from(2u8).powi(34));
    // A batch beyond 2^31 tokens pads beyond u32 and is refused, never wrapped.
    assert_eq!(
        stage_plan(&one, &memory(bytes), (1 << 31) + 1),
        Err(ModelError::Input {
            name: "batch_tokens",
            problem: Problem::TooLarge
        })
    );
    // μ: the fold pass is one more regeneration, and the folds cost window_ps per entry, shared by
    // the batch's tokens. A stage that fits adds neither.
    let costs = UnitCosts {
        window_ps: 2.0,
        ..UnitCosts::SOURCE
    };
    let base = |counts: &Counts, free: f64| {
        mu(counts, &costs, &memory(free), 1024, 1 << 14).expect("a projection")
    };
    let fits = base(&one, bytes);
    let windowed = base(&one, bytes / 2.0);
    assert_eq!(windowed.regen - fits.regen, 1.0);
    let native =
        (one.native_ops_lin + one.native_ops_att_per_ctx * 1024.0) / costs.native_ops_per_s;
    let folds = 2e-12 * f64::from(2u8).powi(34) / f64::from(1u32 << 14) / native;
    assert!(
        (windowed.streaming - fits.streaming - folds).abs() <= 1e-12 * folds,
        "the folds: {} against {folds}",
        windowed.streaming - fits.streaming
    );
    // The source's counts count no regeneration, and no fold pass either.
    let uncounted = Counts {
        regen_full_equivalents: 0.0,
        ..one.clone()
    };
    assert_eq!(base(&uncounted, bytes / 2.0).regen, 0.0);
}

/// The reduction stages' tables of `minimal`'s two-class report at `B_max = 32`
/// (`docs/perf/v0.2/`, RFC-0055): stages 1 to 3, by size, per token or fixed.
fn minimal_stages() -> Vec<ReductionStage> {
    vec![
        ReductionStage {
            tables: vec![
                tables(true, 14, 5.0),
                tables(true, 17, 6.0),
                tables(true, 21, 7.0),
                resident(false, 14, 2.0),
                resident(false, 21, 8.0),
                resident(false, 23, 2.0),
            ],
        },
        ReductionStage {
            tables: vec![
                tables(true, 21, 2.0),
                tables(true, 23, 2.0),
                resident(false, 17, 2.0),
                resident(false, 21, 4.0),
            ],
        },
        ReductionStage {
            tables: vec![resident(true, 14, 2.0), resident(false, 21, 2.0)],
        },
    ]
}

/// The same for `servable`'s two-class report: stages 1 and 2.
fn servable_stages() -> Vec<ReductionStage> {
    vec![
        ReductionStage {
            tables: vec![
                tables(true, 14, 5.0),
                tables(true, 17, 6.0),
                tables(true, 21, 9.0),
                tables(true, 23, 2.0),
                resident(false, 14, 2.0),
                resident(false, 21, 10.0),
                resident(false, 23, 2.0),
            ],
        },
        ReductionStage {
            tables: vec![
                resident(true, 14, 2.0),
                resident(false, 17, 2.0),
                resident(false, 21, 4.0),
            ],
        },
    ]
}

/// The corrected 405B projections that 08 §3, SC-2, RISK-9 and RISK-10 quote, each to the digits
/// quoted. The counts are the source's model with RISK-2's upper bound for `minimal`: 285,696
/// commitments and 302,080 lookups per layer per token over 126 layers (RFC-0024's wide rescale
/// and RFC-0022's two steps at classed residual sites), five regenerations (RFC-0018 P1 to P5),
/// the table rows of the two-class report at `B_max = 32` with RFC-0028's balanced split
/// (`docs/perf/v0.2/`), RFC-0034's score block of 512, and the report's reduction stages. The
/// 16K-token batch's leaf stage is grouped at every context, so its merge adds a sixth
/// regeneration (RFC-0054), and its stages 1 and 2 are windowed, a seventh and an eighth
/// (RFC-0055). The column answers are free or priced as committing is, 30 ps per element; the
/// merge's folds are free or priced at RFC-0054's 2 or 4 ps per element, the table trees'
/// streamed rounds at 6 or 8 ps per leaf, and the windows at RFC-0055's 1 or 8 ps per entry. A change to the model that moves one fails here until the documents follow.
#[test]
fn the_corrected_projections_are_the_ones_the_documents_quote() {
    let corrected = Counts {
        commits_per_token: 285_696.0 * 126.0,
        lookups_per_token: 302_080.0 * 126.0,
        regen_full_equivalents: 5.0,
        table_rows: 6_659_571_712.0,
        score_block: 512.0,
        reduction_stages: minimal_stages(),
        ..Counts::SOURCE_405B
    };
    // The column answers', the merge's, the table rounds' and the windows' unit costs.
    let at =
        |counts: &Counts, context: u32, [column_answer_ps, merge_ps, ps, window_ps]: [f64; 4]| {
            let costs = UnitCosts {
                column_answer_ps,
                merge_ps,
                streaming_pass_ps: ps,
                window_ps,
                ..UnitCosts::SOURCE
            };
            mu(counts, &costs, &MemoryModel::NODE_405B, context, BATCH).expect("a projection")
        };
    let free = [0.0; 4];
    let quoted = |what: &str, value: f64, figure: f64, digits: i32| {
        let half = 0.5 * 10f64.powi(-digits);
        assert!(
            (value - figure).abs() <= half,
            "{what}: the model gives {value}, the documents {figure}"
        );
    };
    let (k1, k4, k16, k128) = (1024, 4096, 16_384, 131_072);
    quoted(
        "source linear term at 1K",
        at(&Counts::SOURCE_405B, k1, free).linear,
        2.0,
        1,
    );
    quoted(
        "corrected linear term at 1K",
        at(&corrected, k1, free).linear,
        4.4,
        1,
    );
    let (base1, base16) = (at(&corrected, k1, free), at(&corrected, k16, free));
    let (mu1, mu16) = (base1.total, base16.total);
    // RISK-9: five regenerations, the merge's sixth, and the two windows' folds (RFC-0055).
    assert_eq!((base1.regen, base16.regen), (8.0, 8.0));
    let windows = stage_plan(&corrected, &MemoryModel::NODE_405B, BATCH).expect("a plan");
    let windows: Vec<(f64, u32)> = windows
        .iter()
        .map(|w| ((w.bytes / 1e10).round() / 100.0, w.window_rounds))
        .collect();
    assert_eq!(windows, [(4.08, 6), (5.5, 6), (0.01, 0)]);
    quoted("μ(1K)", mu1, 13.8, 1);
    quoted("μ(4K)", at(&corrected, k4, free).total, 17.0, 1);
    quoted("μ(16K)", mu16, 29.1, 1);
    quoted("μ(128K)", at(&corrected, k128, free).total, 98.3, 1);
    // RFC-0034: the diagonal blocks' masked halves, which the layout cannot skip, add 0.21 at 1K
    // and 0.19 at 16K.
    let valid_half = Counts {
        score_block: 0.0,
        ..corrected.clone()
    };
    quoted(
        "diagonal blocks at 1K",
        mu1 - at(&valid_half, k1, free).total,
        0.21,
        2,
    );
    quoted(
        "diagonal blocks at 16K",
        mu16 - at(&valid_half, k16, free).total,
        0.19,
        2,
    );
    quoted("1K table term", base1.tables, 0.048, 3);
    quoted("16K table term", base16.tables, 0.044, 3);
    let per_ps = |context| at(&corrected, context, [1.0, 0.0, 0.0, 0.0]).column_answers;
    quoted("1K per ps of c_a", per_ps(k1), 0.070, 3);
    quoted("16K per ps of c_a", per_ps(k16), 0.23, 2);
    // 08 §3: a picosecond of the row combination costs what one of the column answers does,
    // since both price every committed element.
    for context in [k1, k16] {
        let costs = UnitCosts {
            row_combination_ps: 1.0,
            ..UnitCosts::SOURCE
        };
        let m =
            mu(&corrected, &costs, &MemoryModel::NODE_405B, context, BATCH).expect("a projection");
        assert_eq!(m.row_combination, per_ps(context), "at {context}");
    }
    let at_cc = [30.0, 0.0, 0.0, 0.0];
    let (full1, full16) = (at(&corrected, k1, at_cc), at(&corrected, k16, at_cc));
    quoted("1K column answers at c_c", full1.column_answers, 2.1, 1);
    quoted("16K column answers at c_c", full16.column_answers, 7.0, 1);
    quoted("μ(1K) at c_c", full1.total, 15.9, 1);
    quoted("μ(16K) at c_c", full16.total, 36.1, 1);
    // SC-2's ceilings of 12 and 45: the 1K one missed before the column answers, the 16K one's
    // headroom.
    quoted("1K excess", 100.0 * (mu1 / 12.0 - 1.0), 15.0, 0);
    quoted(
        "1K excess at c_c",
        100.0 * (full1.total / 12.0 - 1.0),
        32.0,
        0,
    );
    quoted("16K headroom", 45.0 / mu16, 1.55, 2);
    quoted("16K headroom at c_c", 45.0 / full16.total, 1.25, 2);
    // RISK-10, 08 §3 and SC-2: RFC-0054's plan. The batch takes 256, 512, 1,024 and 8,192 groups
    // and its table trees stream one round. The merge's folds at 2 and 4 ps per element; the
    // table trees' round at 6 or 8 ps per leaf adds under 0.005. RFC-0055's windows at 1 and 8 ps
    // per entry.
    for (context, groups, low, high, (window_low, window_high)) in [
        (k1, 256, 0.16, 0.32, (0.05, 0.42)),
        (k4, 512, 0.30, 0.60, (0.05, 0.42)),
        (k16, 1024, 0.82, 1.63, (0.05, 0.39)),
        (k128, 8192, 3.78, 7.56, (0.03, 0.26)),
    ] {
        let plan = leaf_plan(&corrected, &MemoryModel::NODE_405B, context, BATCH).expect("a plan");
        assert_eq!(
            (plan.groups, plan.table_rounds),
            (groups, 1),
            "at {context}"
        );
        quoted(
            &format!("merge at 2 ps, {context}"),
            at(&corrected, context, [0.0, 2.0, 0.0, 0.0]).streaming,
            low,
            2,
        );
        quoted(
            &format!("merge at 4 ps, {context}"),
            at(&corrected, context, [0.0, 4.0, 0.0, 0.0]).streaming,
            high,
            2,
        );
        let rounds = at(&corrected, context, [0.0, 0.0, 8.0, 0.0]).streaming;
        assert!(
            rounds > 0.0 && rounds < 0.005,
            "table rounds at {context}: {rounds}"
        );
        quoted(
            &format!("windows at 1 ps, {context}"),
            at(&corrected, context, [0.0, 0.0, 0.0, 1.0]).streaming,
            window_low,
            2,
        );
        quoted(
            &format!("windows at 8 ps, {context}"),
            at(&corrected, context, [0.0, 0.0, 0.0, 8.0]).streaming,
            window_high,
            2,
        );
    }
    // RFC-0054 'Motivation': RFC-0018 as written streams 36, 45 and 55 passes, each a
    // regeneration and a fold at 6 to 8 ps per leaf: +53 to +59, +85 to +99 and +190 to +235.
    for (context, passes, low, high) in [
        (k1, 36, 53.0, 59.0),
        (k4, 45, 85.0, 99.0),
        (k16, 55, 190.0, 235.0),
    ] {
        let shared = shared_streaming_passes(&corrected, &MemoryModel::NODE_405B, context, BATCH)
            .expect("a count");
        assert_eq!(shared, passes, "at {context}");
        let leaves = corrected.lookups_per_token
            + 2.0 * corrected.scores_per_token_per_ctx * f64::from(context);
        let native = (corrected.native_ops_lin
            + corrected.native_ops_att_per_ctx * f64::from(context))
            / UnitCosts::SOURCE.native_ops_per_s;
        for (ps, figure) in [(6.0, low), (8.0, high)] {
            let added = f64::from(shared) * (1.0 + ps * 1e-12 * leaves / native);
            quoted(
                &format!("RFC-0018 passes at {ps} ps, {context}"),
                added,
                figure,
                0,
            );
        }
    }
    // 08 §3 and SC-2: with the merge, the table round and the windows at their upper estimates,
    // and c_a = c_c, 16.6 at 1K, 39% above that ceiling, and 38.1 at 16K, 1.18 times below it.
    let upper = [30.0, 4.0, 8.0, 8.0];
    let (all1, all16) = (at(&corrected, k1, upper), at(&corrected, k16, upper));
    quoted("μ(1K) at c_c, upper", all1.total, 16.6, 1);
    quoted(
        "1K excess at c_c, upper",
        100.0 * (all1.total / 12.0 - 1.0),
        39.0,
        0,
    );
    quoted("μ(16K) at c_c, upper", all16.total, 38.1, 1);
    quoted("16K headroom at c_c, upper", 45.0 / all16.total, 1.18, 2);
    // The upper estimates alone take 1K to 14.5, the lower ones to 14.0: no c_a meets that
    // ceiling. At 16K it holds while c_a stays below about 60 ps at the upper estimates.
    quoted(
        "μ(1K), upper",
        at(&corrected, k1, [0.0, 4.0, 8.0, 8.0]).total,
        14.5,
        1,
    );
    quoted(
        "μ(1K), lower",
        at(&corrected, k1, [0.0, 2.0, 6.0, 1.0]).total,
        14.0,
        1,
    );
    quoted(
        "c_a that meets the 16K ceiling, upper",
        (45.0 - at(&corrected, k16, [0.0, 4.0, 8.0, 8.0]).total) / per_ps(k16),
        60.0,
        0,
    );
    // 08 §3's sampling rate for 10% overhead at η = 0.3, and daily detection of 100 violations.
    for (total, rate, detection) in [(mu1, 2.4, 91.0), (full1.total, 2.1, 88.0)] {
        let s = sampling_rate(0.1, total, 0.3).expect("a rate");
        quoted("sampling at 1K", 100.0 * s, rate, 1);
        let d = detection_probability(s, 100).expect("a probability");
        quoted("detection at 1K", 100.0 * d, detection, 0);
    }
}

/// The same projections at the default `servable` profile's count, which 00 §7 SC-2, 00 §9 RISK-2,
/// 00 §9 RISK-10, 08 §3 and `SPEC.md` quote beside `minimal`'s: 359,424 commitments and lookups per
/// layer per token under RFC-0006's selection rule, with RFC-0022's two steps at both classed
/// residual sites, over 126 layers, the two-class report's table rows at `B_max = 32` with the
/// balanced split, RFC-0034's score block of 512, and the report's reduction stages: stage 1 is
/// windowed, a seventh regeneration. Its window binds eight variables of `a·u`, so its folds are
/// priced at RFC-0055's 5 or 17 ps per entry.
#[test]
fn the_servable_projections_are_the_ones_the_documents_quote() {
    let servable = Counts {
        commits_per_token: 359_424.0 * 126.0,
        lookups_per_token: 359_424.0 * 126.0,
        regen_full_equivalents: 5.0,
        table_rows: 7_693_467_648.0,
        score_block: 512.0,
        reduction_stages: servable_stages(),
        ..Counts::SOURCE_405B
    };
    let at = |context: u32, [column_answer_ps, merge_ps, ps, window_ps]: [f64; 4]| {
        let costs = UnitCosts {
            column_answer_ps,
            merge_ps,
            streaming_pass_ps: ps,
            window_ps,
            ..UnitCosts::SOURCE
        };
        mu(&servable, &costs, &MemoryModel::NODE_405B, context, BATCH).expect("a projection")
    };
    let free = [0.0; 4];
    let quoted = |what: &str, value: f64, figure: f64, digits: i32| {
        let half = 0.5 * 10f64.powi(-digits);
        assert!(
            (value - figure).abs() <= half,
            "{what}: the model gives {value}, the documents {figure}"
        );
    };
    let (k1, k16) = (1024, 16_384);
    quoted("linear term at 1K", at(k1, free).linear, 5.3, 1);
    let (mu1, mu16) = (at(k1, free).total, at(k16, free).total);
    assert_eq!(at(k1, free).regen, 7.0);
    let windows = stage_plan(&servable, &MemoryModel::NODE_405B, BATCH).expect("a plan");
    let windows: Vec<(f64, u32)> = windows
        .iter()
        .map(|w| ((w.bytes / 1e10).round() / 100.0, w.window_rounds))
        .collect();
    assert_eq!(windows, [(9.57, 8), (0.01, 0)]);
    quoted("μ(1K)", mu1, 13.7, 1);
    quoted("μ(16K)", mu16, 29.0, 1);
    quoted("1K table term", at(k1, free).tables, 0.055, 3);
    quoted("16K table term", at(k16, free).tables, 0.051, 3);
    quoted(
        "1K per ps of c_a",
        at(k1, [1.0, 0.0, 0.0, 0.0]).column_answers,
        0.084,
        3,
    );
    quoted(
        "16K per ps of c_a",
        at(k16, [1.0, 0.0, 0.0, 0.0]).column_answers,
        0.25,
        2,
    );
    let at_cc = [30.0, 0.0, 0.0, 0.0];
    let (full1, full16) = (at(k1, at_cc), at(k16, at_cc));
    quoted("1K column answers at c_c", full1.column_answers, 2.5, 1);
    quoted("16K column answers at c_c", full16.column_answers, 7.4, 1);
    quoted("μ(1K) at c_c", full1.total, 16.2, 1);
    quoted("μ(16K) at c_c", full16.total, 36.4, 1);
    // SC-2: the 1K ceiling missed by 14% before the column answers, by 35% at c_a = c_c; 16K's
    // headroom.
    quoted("1K excess", 100.0 * (mu1 / 12.0 - 1.0), 14.0, 0);
    quoted(
        "1K excess at c_c",
        100.0 * (full1.total / 12.0 - 1.0),
        35.0,
        0,
    );
    quoted("16K headroom", 45.0 / mu16, 1.55, 2);
    quoted("16K headroom at c_c", 45.0 / full16.total, 1.24, 2);
    // The window at 5 and 17 ps per entry.
    quoted(
        "1K windows at 5 ps",
        at(k1, [0.0, 0.0, 0.0, 5.0]).streaming,
        0.265,
        3,
    );
    quoted(
        "1K windows at 17 ps",
        at(k1, [0.0, 0.0, 0.0, 17.0]).streaming,
        0.90,
        2,
    );
    quoted(
        "16K windows at 5 ps",
        at(k16, [0.0, 0.0, 0.0, 5.0]).streaming,
        0.25,
        2,
    );
    quoted(
        "16K windows at 17 ps",
        at(k16, [0.0, 0.0, 0.0, 17.0]).streaming,
        0.84,
        2,
    );
    // With the merge, the table round and the window at their upper estimates as well: 17.5 at
    // 1K, 46% above the ceiling, and 38.9 at 16K, below it, while c_a stays below about 55 ps.
    let upper = [30.0, 4.0, 8.0, 17.0];
    let (all1, all16) = (at(k1, upper), at(k16, upper));
    quoted("μ(1K) at c_c, upper", all1.total, 17.5, 1);
    quoted(
        "1K excess at c_c, upper",
        100.0 * (all1.total / 12.0 - 1.0),
        46.0,
        0,
    );
    quoted("μ(16K) at c_c, upper", all16.total, 38.9, 1);
    quoted("16K headroom at c_c, upper", 45.0 / all16.total, 1.16, 2);
    quoted(
        "c_a that meets the 16K ceiling, upper",
        (45.0 - at(k16, [0.0, 4.0, 8.0, 17.0]).total)
            / at(k16, [1.0, 0.0, 0.0, 0.0]).column_answers,
        55.0,
        0,
    );
    for (total, rate, detection) in [(mu1, 2.4, 91.0), (full1.total, 2.1, 87.0)] {
        let s = sampling_rate(0.1, total, 0.3).expect("a rate");
        quoted("sampling at 1K", 100.0 * s, rate, 1);
        let d = detection_probability(s, 100).expect("a probability");
        quoted("detection at 1K", 100.0 * d, detection, 0);
    }
}

/// RFC-0034's closed form: at context `S`, a token commits `A·(S + min(S, b))` scores, with
/// `b = max(Q, 2^w)`. Here the layout's rows are counted one by one: in each query block, every
/// key block whose first key does not follow the block's last query. The model's committed
/// scores, priced at one picosecond each, match the count for key blocks as wide as, wider and
/// narrower than the query blocks, and for a sequence shorter than a block.
#[test]
fn the_committed_scores_are_the_layouts_rows() {
    // (log2 Q, w, log2 S).
    for (log2_q, w, log2_s) in [(3u32, 3u32, 7u32), (3, 5, 8), (5, 3, 8), (3, 5, 4)] {
        let (q, s) = (1u64 << log2_q, 1u64 << log2_s);
        let keys = 1u64 << w.min(log2_s);
        let mut committed = 0u64;
        for block in 0..s / q {
            for row in 0..s / keys {
                if row * keys < (block + 1) * q {
                    committed += q * keys;
                }
            }
        }
        // One score per token per unit of context is half a score for each of two (layer, head)
        // pairs, so a token commits 2·committed/S of them.
        let counts = Counts {
            commits_per_token: 0.0,
            lookups_per_token: 0.0,
            scores_per_token_per_ctx: 1.0,
            score_block: q.max(1 << w) as f64,
            ..Counts::SOURCE_405B
        };
        let costs = UnitCosts {
            commit_ps: 1.0,
            lookup_ps: 0.0,
            product_leaf_ps: 0.0,
            glue_fraction: 0.0,
            ..UnitCosts::SOURCE
        };
        let context = u32::try_from(s).expect("a small context");
        let m = mu(&counts, &costs, &MemoryModel::NODE_405B, context, BATCH).expect("a projection");
        let native = (counts.native_ops_lin + counts.native_ops_att_per_ctx * s as f64)
            / costs.native_ops_per_s;
        let modelled = m.attention * native / 1e-12;
        let counted = 2.0 * committed as f64 / s as f64;
        assert!(
            (modelled - counted).abs() <= 1e-9 * counted,
            "Q = 2^{log2_q}, 2^{w} keys, S = 2^{log2_s}: the model commits {modelled}, the layout {counted}"
        );
    }
}

/// The report states every input and every term under its own name. Here every term is
/// positive and no two inputs or terms are equal, so a value written under another's key shows;
/// with the source's defaults the regeneration, streaming and column-answer terms are all zero,
/// and a swap between them would not.
#[test]
fn the_report_states_each_input_and_term_under_its_own_name() {
    let (counts, costs, memory) = every_term();
    let counts = Counts {
        lookups_per_token: 1.5 * counts.commits_per_token,
        ..counts
    };
    let report = ModelReport::new(
        false,
        counts.clone(),
        costs,
        memory,
        Scenario::SOURCE,
        &CONTEXTS,
    )
    .expect("a report");
    let json: serde_json::Value =
        serde_json::from_str(&report.to_json().expect("JSON")).expect("JSON");
    // serde_json parses a float to within an ulp or so, not always to the nearest.
    let states = |object: &serde_json::Value, key: &str, value: f64| {
        let stated = object[key].as_f64().unwrap_or(f64::NAN);
        assert!(
            (stated - value).abs() <= 1e-15 * value.abs(),
            "{key}: {stated} against {value}"
        );
    };
    for (key, value) in [
        ("commits_per_token", counts.commits_per_token),
        ("lookups_per_token", counts.lookups_per_token),
        ("scores_per_token_per_ctx", counts.scores_per_token_per_ctx),
        ("native_ops_lin", counts.native_ops_lin),
        ("native_ops_att_per_ctx", counts.native_ops_att_per_ctx),
        ("regen_full_equivalents", counts.regen_full_equivalents),
        ("table_rows", counts.table_rows),
        ("score_block", counts.score_block),
    ] {
        states(&json["inputs"]["counts"], key, value);
    }
    for (key, value) in [
        ("commit_ps", costs.commit_ps),
        ("lookup_ps", costs.lookup_ps),
        ("product_leaf_ps", costs.product_leaf_ps),
        ("glue_fraction", costs.glue_fraction),
        ("weight_pass_gpu_s", costs.weight_pass_gpu_s),
        ("native_ops_per_s", costs.native_ops_per_s),
        ("streaming_pass_ps", costs.streaming_pass_ps),
        ("merge_ps", costs.merge_ps),
        ("column_answer_ps", costs.column_answer_ps),
        ("row_combination_ps", costs.row_combination_ps),
    ] {
        states(&json["inputs"]["unit_costs"], key, value);
    }
    assert_eq!(report.projections.len(), CONTEXTS.len());
    for (k, p) in report.projections.iter().enumerate() {
        for (key, value) in [
            ("linear", p.mu.linear),
            ("attention", p.mu.attention),
            ("weights", p.mu.weights),
            ("regen", p.mu.regen),
            ("streaming", p.mu.streaming),
            ("column_answers", p.mu.column_answers),
            ("row_combination", p.mu.row_combination),
            ("tables", p.mu.tables),
            ("total", p.mu.total),
        ] {
            states(&json["projections"][k]["mu"], key, value);
        }
        let projection = &json["projections"][k];
        assert_eq!(projection["leaf_groups"], u64::from(p.leaf_groups));
        assert_eq!(projection["table_rounds"], u64::from(p.table_rounds));
    }
}

fn refused<T>(name: &'static str, problem: Problem) -> Result<T, ModelError> {
    Err(ModelError::Input { name, problem })
}

/// Every input the model cannot price is refused with its name and what is wrong with it, and
/// an output that overflows is an error, not an infinity.
#[test]
fn inputs_the_model_cannot_price_are_refused() {
    let (counts, costs, memory) = every_term();
    let with_costs = |change: fn(&mut UnitCosts)| {
        let mut u = costs;
        change(&mut u);
        mu(&counts, &u, &memory, 1024, BATCH).map(|m| m.total)
    };
    assert_eq!(
        with_costs(|u| u.commit_ps = f64::NAN),
        refused("commit_ps", Problem::NotFinite)
    );
    assert_eq!(
        with_costs(|u| u.lookup_ps = -1.0),
        refused("lookup_ps", Problem::Negative)
    );
    assert_eq!(
        with_costs(|u| u.glue_fraction = f64::INFINITY),
        refused("glue_fraction", Problem::NotFinite)
    );
    assert_eq!(
        with_costs(|u| u.native_ops_per_s = 0.0),
        refused("native_ops_per_s", Problem::NotPositive)
    );
    assert_eq!(
        with_costs(|u| u.column_answer_ps = -1.0),
        refused("column_answer_ps", Problem::Negative)
    );
    assert_eq!(
        with_costs(|u| u.row_combination_ps = f64::NAN),
        refused("row_combination_ps", Problem::NotFinite)
    );
    assert_eq!(
        with_costs(|u| u.merge_ps = -1.0),
        refused("merge_ps", Problem::Negative)
    );
    let with_counts = |change: fn(&mut Counts), context: u32| {
        let mut c = counts.clone();
        change(&mut c);
        mu(&c, &costs, &memory, context, BATCH).map(|m| m.total)
    };
    assert_eq!(
        with_counts(|c| c.regen_full_equivalents = -4.0, 1024),
        refused("regen_full_equivalents", Problem::Negative)
    );
    assert_eq!(
        with_counts(|c| c.table_rows = -1.0, 1024),
        refused("table_rows", Problem::Negative)
    );
    assert_eq!(
        with_counts(|c| c.score_block = f64::INFINITY, 1024),
        refused("score_block", Problem::NotFinite)
    );
    assert_eq!(
        with_counts(|c| c.native_ops_lin = 0.0, 0),
        refused("native_ops_lin", Problem::NotPositive),
        "no native work to compare with"
    );
    assert_eq!(
        with_counts(|c| c.commits_per_token = f64::MAX, 1024),
        Err(ModelError::Output { name: "linear" }),
        "an overflowing term"
    );
    let no_memory = MemoryModel {
        free_bytes: 0.0,
        ..memory
    };
    assert_eq!(
        mu(&counts, &costs, &no_memory, 1024, BATCH).map(|m| m.total),
        refused("free_bytes", Problem::NotPositive)
    );
    assert_eq!(
        mu(&counts, &costs, &memory, 1024, 0).map(|m| m.total),
        refused("batch_tokens", Problem::NotPositive)
    );
    // RFC-0054: memory in which one token's leaves do not fit, or that would take more than
    // 2^14 groups, has no plan.
    for free_bytes in [1e6, 1e9] {
        let small = MemoryModel {
            free_bytes,
            ..memory
        };
        assert_eq!(
            mu(&counts, &costs, &small, 1024, BATCH).map(|m| m.total),
            refused("free_bytes", Problem::TooSmall),
            "{free_bytes} bytes"
        );
    }
    assert_eq!(
        proof_time_s(3.0, &counts, &costs, 1024, 0),
        refused("gpus", Problem::NotPositive)
    );
    assert_eq!(
        sampling_rate(0.1, 3.0, 0.0),
        refused("eta", Problem::NotPositive)
    );
    assert_eq!(
        sampling_rate(0.1, 0.0, 0.3),
        refused("mu", Problem::NotPositive)
    );
    assert_eq!(
        detection_probability(1.5, 100),
        refused("rate", Problem::AboveOne)
    );
    assert_eq!(
        ModelReport::new(true, counts, costs, memory, Scenario::SOURCE, &[])
            .map(|r| r.projections.len()),
        refused("contexts", Problem::Empty)
    );
    // Within range, the edges are numbers: a budget for every request samples all of them, and
    // sampling everything detects any violation.
    assert_eq!(sampling_rate(10.0, 3.0, 0.3), Ok(1.0));
    assert_eq!(detection_probability(1.0, 1), Ok(1.0));
    assert_eq!(
        detection_probability(1.0, 0),
        Ok(0.0),
        "no violation, nothing to detect"
    );
    assert_eq!(detection_probability(0.0, 1_000_000), Ok(0.0));
}
