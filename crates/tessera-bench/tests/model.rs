// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The cost model's properties (RFC-0020 'Testing Strategy'): `μ` rises with every unit cost and
//! every count of proof work, and with the native rate it is measured against, and falls with
//! native work, each input moving the terms it enters and no other; the streaming term is zero
//! while a batch's leaves fit in memory and steps up one pass per doubling of them beyond it;
//! the column answers price every committed element; each table row costs what a token's
//! committed lookup costs, once per batch; the corrected projections the documents quote are the
//! model's; and an input the model cannot price is refused, never turned into a number.

use tessera_bench::{
    Counts, MemoryModel, ModelError, ModelReport, MuBreakdown, Problem, Scenario, UnitCosts,
    detection_probability, mu, proof_time_s, sampling_rate, streaming_passes,
};

/// The source's model with streamed passes priced at RFC-0018's 7 ps per score, column answers
/// at 15 ps per committed element, five regenerations and 2 · 10^10 table rows, so that every
/// term is positive.
fn every_term() -> (Counts, UnitCosts, MemoryModel) {
    let counts = Counts {
        regen_full_equivalents: 5.0,
        table_rows: 2e10,
        ..Counts::SOURCE_405B
    };
    let costs = UnitCosts {
        streaming_pass_ps: 7.0,
        column_answer_ps: 15.0,
        ..UnitCosts::SOURCE
    };
    (counts, costs, MemoryModel::NODE_405B)
}

const CONTEXTS: [u32; 3] = [1024, 16_384, 131_072];
const BATCH: u32 = Scenario::SOURCE.batch_tokens;

fn terms(m: &MuBreakdown) -> [f64; 7] {
    [
        m.linear,
        m.attention,
        m.weights,
        m.regen,
        m.streaming,
        m.column_answers,
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

fn moves(before: &MuBreakdown, after: &MuBreakdown) -> [Move; 7] {
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
    // streaming, column answers, tables, in that order.
    type Raise = (&'static str, fn(&mut Counts, &mut UnitCosts), [Move; 7]);
    let raises: [Raise; 15] = [
        (
            "commit_ps",
            |_, u| u.commit_ps *= 1.5,
            [Up, Up, Same, Same, Same, Same, Up],
        ),
        (
            "lookup_ps",
            |_, u| u.lookup_ps *= 1.5,
            [Up, Up, Same, Same, Same, Same, Up],
        ),
        (
            "product_leaf_ps",
            |_, u| u.product_leaf_ps *= 1.5,
            [Same, Up, Same, Same, Same, Same, Same],
        ),
        (
            "glue_fraction",
            |_, u| u.glue_fraction *= 1.5,
            [Up, Up, Same, Same, Same, Same, Up],
        ),
        (
            "weight_pass_gpu_s",
            |_, u| u.weight_pass_gpu_s *= 1.5,
            [Same, Same, Up, Same, Same, Same, Same],
        ),
        (
            "streaming_pass_ps",
            |_, u| u.streaming_pass_ps *= 1.5,
            [Same, Same, Same, Same, Up, Same, Same],
        ),
        (
            "column_answer_ps",
            |_, u| u.column_answer_ps *= 1.5,
            [Same, Same, Same, Same, Same, Up, Same],
        ),
        // A faster native path makes the same proof a larger multiple of it.
        (
            "native_ops_per_s",
            |_, u| u.native_ops_per_s *= 1.5,
            [Up, Up, Up, Same, Up, Up, Up],
        ),
        (
            "commits_per_token",
            |c, _| c.commits_per_token *= 1.5,
            [Up, Same, Same, Same, Same, Up, Same],
        ),
        (
            "lookups_per_token",
            |c, _| c.lookups_per_token *= 1.5,
            [Up, Same, Same, Same, Same, Same, Same],
        ),
        (
            "scores_per_token_per_ctx",
            |c, _| c.scores_per_token_per_ctx *= 1.5,
            [Same, Up, Same, Same, Up, Up, Same],
        ),
        (
            "native_ops_lin",
            |c, _| c.native_ops_lin *= 1.5,
            [Down, Down, Down, Same, Down, Down, Down],
        ),
        (
            "native_ops_att_per_ctx",
            |c, _| c.native_ops_att_per_ctx *= 1.5,
            [Down, Down, Down, Same, Down, Down, Down],
        ),
        (
            "regen_full_equivalents",
            |c, _| c.regen_full_equivalents *= 1.5,
            [Same, Same, Same, Up, Same, Same, Same],
        ),
        (
            "table_rows",
            |c, _| c.table_rows *= 1.5,
            [Same, Same, Same, Same, Same, Same, Up],
        ),
    ];
    for (name, raise, expected) in raises {
        for context in CONTEXTS {
            let before = mu(&counts, &costs, &memory, context, BATCH).expect("a projection");
            let (mut c, mut u) = (counts, costs);
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

/// `c(S)` is 0 while a batch's attention leaves fit in the free memory, and one more per
/// doubling of the leaves beyond it, whatever doubles them; the streaming term is the passes
/// times the cost of one, so it steps with them.
#[test]
fn the_streaming_term_steps_one_pass_per_doubling_of_the_leaves() {
    let (counts, costs, _) = every_term();
    let context = 4096;
    let leaves = counts.scores_per_token_per_ctx * f64::from(context) * f64::from(BATCH);
    let exact = MemoryModel {
        free_bytes: leaves * 32.0,
        bytes_per_leaf: 32.0,
    };
    let below = MemoryModel {
        free_bytes: exact.free_bytes * (1.0 - 1e-9),
        ..exact
    };
    assert_eq!(
        streaming_passes(&counts, &exact, context, BATCH),
        Ok(0),
        "the leaves just fit"
    );
    assert_eq!(
        streaming_passes(&counts, &below, context, BATCH),
        Ok(1),
        "one byte short"
    );
    let one = mu(&counts, &costs, &below, context, BATCH)
        .expect("a projection")
        .streaming;
    assert!(one > 0.0);
    assert_eq!(
        mu(&counts, &costs, &exact, context, BATCH)
            .expect("a projection")
            .streaming,
        0.0
    );
    for k in 1..=12u32 {
        let batch = BATCH << k;
        assert_eq!(
            streaming_passes(&counts, &exact, context, batch),
            Ok(k),
            "batch 2^{k}·T"
        );
        let memory = MemoryModel {
            free_bytes: exact.free_bytes / f64::from(1u32 << k),
            ..exact
        };
        assert_eq!(
            streaming_passes(&counts, &memory, context, BATCH),
            Ok(k),
            "memory / 2^{k}"
        );
        let streaming = mu(&counts, &costs, &memory, context, BATCH)
            .expect("a projection")
            .streaming;
        assert!(
            (streaming / one - f64::from(k)).abs() < 1e-12,
            "memory / 2^{k}: {k} passes' worth"
        );
    }
    // The source prices no pass: its streaming term is zero however many there are.
    let source = mu(
        &Counts::SOURCE_405B,
        &UnitCosts::SOURCE,
        &MemoryModel::NODE_405B,
        131_072,
        BATCH,
    )
    .expect("the source's model");
    assert_eq!(source.streaming, 0.0);
    // The counts RFC-0018 'Leaf levels', RISK-10 and 08 §3 quote for the source's 16K-token batch
    // on 0.2 TB: at 16K, 2.165×10^12 leaves of 32 bytes are 346 times the memory, so 9 passes.
    for (context, passes) in [(1024, 5), (4096, 7), (16_384, 9)] {
        assert_eq!(
            streaming_passes(
                &Counts::SOURCE_405B,
                &MemoryModel::NODE_405B,
                context,
                BATCH
            ),
            Ok(passes),
            "the documents count {passes} passes over the attention leaves at {context}"
        );
    }
}

/// The column answers evaluate every element the witness pass commits, `C` per token in the
/// linear layers and one per score, and nothing else: priced at the commitment's cost, with
/// lookups, product leaves and glue free, they cost exactly what committing does.
#[test]
fn the_column_answers_price_every_committed_element() {
    let (counts, _, memory) = every_term();
    let costs = UnitCosts {
        lookup_ps: 0.0,
        product_leaf_ps: 0.0,
        glue_fraction: 0.0,
        column_answer_ps: UnitCosts::SOURCE.commit_ps,
        ..UnitCosts::SOURCE
    };
    for context in CONTEXTS {
        let m = mu(&counts, &costs, &memory, context, BATCH).expect("a projection");
        let committing = m.linear + m.attention;
        assert!(
            (m.column_answers - committing).abs() <= 1e-12 * committing,
            "at {context}: {} against {committing}",
            m.column_answers
        );
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
            ..counts
        };
        let m = mu(&none, &costs, &memory, context, BATCH).expect("a projection");
        assert_eq!(m.tables, 0.0);
    }
}

/// The corrected 405B projections that 08 §3, SC-2, RISK-9 and RISK-10 quote, each to the digits
/// quoted. The counts are the source's model with RISK-2's upper bound for `minimal`: 285,696
/// commitments and 302,080 lookups per layer per token over 126 layers (RFC-0024's wide rescale
/// and RFC-0022's two steps at classed residual sites), five regenerations (RFC-0018 P1 to P5), and
/// the table rows of the two-class report at `B_max = 32` with RFC-0028's balanced split
/// (`docs/perf/v0.2/`). The column answers are free or priced as committing is, 30 ps per element.
/// The streamed passes over the attention leaves are free or priced at RFC-0018's 6 or 8 ps per
/// score. A change to the model that moves one fails here until the documents follow.
#[test]
fn the_corrected_projections_are_the_ones_the_documents_quote() {
    let corrected = Counts {
        commits_per_token: 285_696.0 * 126.0,
        lookups_per_token: 302_080.0 * 126.0,
        regen_full_equivalents: 5.0,
        table_rows: 6_659_571_712.0,
        ..Counts::SOURCE_405B
    };
    let at = |counts: &Counts, context: u32, column_answer_ps: f64, streaming_pass_ps: f64| {
        let costs = UnitCosts {
            column_answer_ps,
            streaming_pass_ps,
            ..UnitCosts::SOURCE
        };
        mu(counts, &costs, &MemoryModel::NODE_405B, context, BATCH).expect("a projection")
    };
    let quoted = |what: &str, value: f64, figure: f64, digits: i32| {
        let half = 0.5 * 10f64.powi(-digits);
        assert!(
            (value - figure).abs() <= half,
            "{what}: the model gives {value}, the documents {figure}"
        );
    };
    let (k1, k4, k16) = (1024, 4096, 16_384);
    quoted(
        "source linear term at 1K",
        at(&Counts::SOURCE_405B, k1, 0.0, 0.0).linear,
        2.0,
        1,
    );
    quoted(
        "corrected linear term at 1K",
        at(&corrected, k1, 0.0, 0.0).linear,
        4.4,
        1,
    );
    let (mu1, mu16) = (
        at(&corrected, k1, 0.0, 0.0).total,
        at(&corrected, k16, 0.0, 0.0).total,
    );
    quoted("μ(1K)", mu1, 10.6, 1);
    quoted("μ(16K)", mu16, 25.9, 1);
    quoted(
        "1K table term",
        at(&corrected, k1, 0.0, 0.0).tables,
        0.048,
        3,
    );
    quoted(
        "16K table term",
        at(&corrected, k16, 0.0, 0.0).tables,
        0.044,
        3,
    );
    let per_ps = |context| at(&corrected, context, 1.0, 0.0).column_answers;
    quoted("1K per ps of c_a", per_ps(k1), 0.064, 3);
    quoted("16K per ps of c_a", per_ps(k16), 0.23, 2);
    let (full1, full16) = (
        at(&corrected, k1, 30.0, 0.0),
        at(&corrected, k16, 30.0, 0.0),
    );
    quoted("1K column answers at c_c", full1.column_answers, 1.9, 1);
    quoted("16K column answers at c_c", full16.column_answers, 6.8, 1);
    quoted("μ(1K) at c_c", full1.total, 12.5, 1);
    // SC-2's headroom: ceilings of 12 and 45 over the corrected figures.
    quoted("1K headroom", 12.0 / mu1, 1.14, 2);
    quoted("16K headroom", 45.0 / mu16, 1.74, 2);
    quoted("16K headroom at c_c", 45.0 / full16.total, 1.37, 2);
    quoted(
        "1K excess at c_c",
        100.0 * (full1.total / 12.0 - 1.0),
        4.0,
        0,
    );
    // RISK-10, 08 §3 and SC-2: the streamed passes over the attention leaves at 6 and 8 ps per
    // score, and their share of the attention term. They do not depend on the profile.
    for (context, low, high, share_low, share_high, digits) in [
        (k1, 0.36, 0.48, 32.0, 42.0, 2),
        (k4, 2.0, 2.7, 45.0, 59.0, 1),
        (k16, 9.6, 12.8, 57.0, 76.0, 1),
    ] {
        let attention = at(&corrected, context, 0.0, 0.0).attention;
        for (ps, figure, share) in [(6.0, low, share_low), (8.0, high, share_high)] {
            let streaming = at(&corrected, context, 0.0, ps).streaming;
            quoted(
                &format!("streaming at {ps} ps, {context}"),
                streaming,
                figure,
                digits,
            );
            quoted(
                &format!("share of attention at {ps} ps, {context}"),
                100.0 * streaming / attention,
                share,
                0,
            );
        }
    }
    // 08 §3 and SC-2: at c_c with the upper streaming estimate, 8% above the 1K ceiling and 1.3%
    // above the 16K one.
    let (all1, all16) = (
        at(&corrected, k1, 30.0, 8.0),
        at(&corrected, k16, 30.0, 8.0),
    );
    quoted("μ(1K) at c_c with streaming", all1.total, 13.0, 1);
    quoted(
        "1K excess at c_c with streaming",
        100.0 * (all1.total / 12.0 - 1.0),
        8.0,
        0,
    );
    quoted("μ(16K) at c_c with streaming", all16.total, 45.6, 1);
    quoted(
        "16K excess at c_c with streaming",
        100.0 * (all16.total / 45.0 - 1.0),
        1.3,
        1,
    );
    // RISK-10 and 08 §3: with the upper streaming estimate, the ceilings hold while c_a stays below
    // about 15 ps at 1K and 27 ps at 16K.
    for (context, ceiling, room) in [(k1, 12.0, 15.0), (k16, 45.0, 27.0)] {
        let streamed = at(&corrected, context, 0.0, 8.0).total;
        quoted(
            &format!("c_a that meets the ceiling at {context}"),
            (ceiling - streamed) / per_ps(context),
            room,
            0,
        );
    }
    // 08 §3's sampling rate for 10% overhead at η = 0.3, and daily detection of 100 violations.
    for (total, rate, detection) in [(mu1, 3.2, 96.0), (full1.total, 2.7, 93.0)] {
        let s = sampling_rate(0.1, total, 0.3).expect("a rate");
        quoted("sampling at 1K", 100.0 * s, rate, 1);
        let d = detection_probability(s, 100).expect("a probability");
        quoted("detection at 1K", 100.0 * d, detection, 0);
    }
}

/// The same projections at the default `servable` profile's count, which 00 §7 SC-2, 00 §9 RISK-2,
/// 00 §9 RISK-10, 08 §3 and `SPEC.md` quote beside `minimal`'s: 359,424 commitments and lookups per
/// layer per token under RFC-0006's selection rule, with RFC-0022's two steps at both classed
/// residual sites, over 126 layers, and the two-class report's table rows at `B_max = 32` with the
/// balanced split.
#[test]
fn the_servable_projections_are_the_ones_the_documents_quote() {
    let servable = Counts {
        commits_per_token: 359_424.0 * 126.0,
        lookups_per_token: 359_424.0 * 126.0,
        regen_full_equivalents: 5.0,
        table_rows: 7_693_467_648.0,
        ..Counts::SOURCE_405B
    };
    let at = |context: u32, column_answer_ps: f64, streaming_pass_ps: f64| {
        let costs = UnitCosts {
            column_answer_ps,
            streaming_pass_ps,
            ..UnitCosts::SOURCE
        };
        mu(&servable, &costs, &MemoryModel::NODE_405B, context, BATCH).expect("a projection")
    };
    let quoted = |what: &str, value: f64, figure: f64, digits: i32| {
        let half = 0.5 * 10f64.powi(-digits);
        assert!(
            (value - figure).abs() <= half,
            "{what}: the model gives {value}, the documents {figure}"
        );
    };
    let (k1, k16) = (1024, 16_384);
    quoted("linear term at 1K", at(k1, 0.0, 0.0).linear, 5.3, 1);
    let (mu1, mu16) = (at(k1, 0.0, 0.0).total, at(k16, 0.0, 0.0).total);
    quoted("μ(1K)", mu1, 11.5, 1);
    quoted("μ(16K)", mu16, 26.8, 1);
    quoted("1K table term", at(k1, 0.0, 0.0).tables, 0.055, 3);
    quoted("16K table term", at(k16, 0.0, 0.0).tables, 0.051, 3);
    quoted(
        "1K per ps of c_a",
        at(k1, 1.0, 0.0).column_answers,
        0.078,
        3,
    );
    quoted(
        "16K per ps of c_a",
        at(k16, 1.0, 0.0).column_answers,
        0.24,
        2,
    );
    let (full1, full16) = (at(k1, 30.0, 0.0), at(k16, 30.0, 0.0));
    quoted("1K column answers at c_c", full1.column_answers, 2.3, 1);
    quoted("16K column answers at c_c", full16.column_answers, 7.2, 1);
    quoted("μ(1K) at c_c", full1.total, 13.9, 1);
    quoted("μ(16K) at c_c", full16.total, 34.0, 1);
    // SC-2's headroom, and the 1K ceiling missed by 15% at c_a = c_c.
    quoted("1K headroom", 12.0 / mu1, 1.04, 2);
    quoted("16K headroom", 45.0 / mu16, 1.68, 2);
    quoted(
        "1K excess at c_c",
        100.0 * (full1.total / 12.0 - 1.0),
        15.0,
        0,
    );
    // SC-2, RISK-10 and 08 §3: the upper streaming estimate alone brings the 1K projection to
    // about 12.0, the ceiling, and leaves the column answers no room under it.
    let streamed1 = at(k1, 0.0, 8.0).total;
    quoted("μ(1K) with streaming", streamed1, 12.0, 1);
    let room = (12.0 - streamed1) / at(k1, 1.0, 0.0).column_answers;
    assert!(
        (0.0..0.1).contains(&room),
        "servable's 1K ceiling leaves {room} ps per element for the column answers"
    );
    // With the column answers priced as committing as well: 14.3 at 1K and 46.8 at 16K, 19% and
    // 4% above the ceilings; `SPEC.md` rounds them to 14.3 and 47.
    let (all1, all16) = (at(k1, 30.0, 8.0), at(k16, 30.0, 8.0));
    quoted("μ(1K) at c_c with streaming", all1.total, 14.3, 1);
    quoted(
        "1K excess at c_c with streaming",
        100.0 * (all1.total / 12.0 - 1.0),
        19.0,
        0,
    );
    quoted("μ(16K) at c_c with streaming", all16.total, 46.8, 1);
    quoted(
        "μ(16K) at c_c with streaming, SPEC.md",
        all16.total,
        47.0,
        0,
    );
    quoted(
        "16K excess at c_c with streaming",
        100.0 * (all16.total / 45.0 - 1.0),
        4.0,
        0,
    );
    for (total, rate, detection) in [(mu1, 2.9, 95.0), (full1.total, 2.4, 91.0)] {
        let s = sampling_rate(0.1, total, 0.3).expect("a rate");
        quoted("sampling at 1K", 100.0 * s, rate, 1);
        let d = detection_probability(s, 100).expect("a probability");
        quoted("detection at 1K", 100.0 * d, detection, 0);
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
    let report = ModelReport::new(false, counts, costs, memory, Scenario::SOURCE, &CONTEXTS)
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
        ("column_answer_ps", costs.column_answer_ps),
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
            ("tables", p.mu.tables),
            ("total", p.mu.total),
        ] {
            states(&json["projections"][k]["mu"], key, value);
        }
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
    let with_counts = |change: fn(&mut Counts), context: u32| {
        let mut c = counts;
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
