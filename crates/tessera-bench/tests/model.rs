// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The cost model's properties (RFC-0020 'Testing Strategy'): `μ` rises with every unit cost and
//! every count of proof work, and with the native rate it is measured against, and falls with
//! native work, each input moving the terms it enters and no other; the streaming term is zero
//! while a batch's leaves fit in memory and steps up one pass per doubling of them beyond it;
//! and an input the model cannot price is refused, never turned into a number.

use tessera_bench::{
    Counts, MemoryModel, ModelError, ModelReport, MuBreakdown, Problem, Scenario, UnitCosts,
    detection_probability, mu, proof_time_s, sampling_rate, streaming_passes,
};

/// The source's model with streamed passes priced at RFC-0018's 7 ps per score and four
/// regenerations, so that every term is positive.
fn every_term() -> (Counts, UnitCosts, MemoryModel) {
    let counts = Counts {
        regen_full_equivalents: 4.0,
        ..Counts::SOURCE_405B
    };
    let costs = UnitCosts {
        streaming_pass_ps: 7.0,
        ..UnitCosts::SOURCE
    };
    (counts, costs, MemoryModel::NODE_405B)
}

const CONTEXTS: [u32; 3] = [1024, 16_384, 131_072];
const BATCH: u32 = Scenario::SOURCE.batch_tokens;

fn terms(m: &MuBreakdown) -> [f64; 5] {
    [m.linear, m.attention, m.weights, m.regen, m.streaming]
}

/// Each term after one input moved, against before: up, down or unchanged.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Move {
    Up,
    Down,
    Same,
}

fn moves(before: &MuBreakdown, after: &MuBreakdown) -> [Move; 5] {
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
    // streaming, in that order.
    type Raise = (&'static str, fn(&mut Counts, &mut UnitCosts), [Move; 5]);
    let raises: [Raise; 13] = [
        (
            "commit_ps",
            |_, u| u.commit_ps *= 1.5,
            [Up, Up, Same, Same, Same],
        ),
        (
            "lookup_ps",
            |_, u| u.lookup_ps *= 1.5,
            [Up, Up, Same, Same, Same],
        ),
        (
            "product_leaf_ps",
            |_, u| u.product_leaf_ps *= 1.5,
            [Same, Up, Same, Same, Same],
        ),
        (
            "glue_fraction",
            |_, u| u.glue_fraction *= 1.5,
            [Up, Up, Same, Same, Same],
        ),
        (
            "weight_pass_gpu_s",
            |_, u| u.weight_pass_gpu_s *= 1.5,
            [Same, Same, Up, Same, Same],
        ),
        (
            "streaming_pass_ps",
            |_, u| u.streaming_pass_ps *= 1.5,
            [Same, Same, Same, Same, Up],
        ),
        // A faster native path makes the same proof a larger multiple of it.
        (
            "native_ops_per_s",
            |_, u| u.native_ops_per_s *= 1.5,
            [Up, Up, Up, Same, Up],
        ),
        (
            "commits_per_token",
            |c, _| c.commits_per_token *= 1.5,
            [Up, Same, Same, Same, Same],
        ),
        (
            "lookups_per_token",
            |c, _| c.lookups_per_token *= 1.5,
            [Up, Same, Same, Same, Same],
        ),
        (
            "scores_per_token_per_ctx",
            |c, _| c.scores_per_token_per_ctx *= 1.5,
            [Same, Up, Same, Same, Up],
        ),
        (
            "native_ops_lin",
            |c, _| c.native_ops_lin *= 1.5,
            [Down, Down, Down, Same, Down],
        ),
        (
            "native_ops_att_per_ctx",
            |c, _| c.native_ops_att_per_ctx *= 1.5,
            [Down, Down, Down, Same, Down],
        ),
        (
            "regen_full_equivalents",
            |c, _| c.regen_full_equivalents *= 1.5,
            [Same, Same, Same, Up, Same],
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
    assert_eq!(
        streaming_passes(&Counts::SOURCE_405B, &MemoryModel::NODE_405B, 16_384, BATCH),
        Ok(9),
        "about 8 at 16K (RFC-0018 'Leaf levels'): 2.2×10^12 leaves of 32 bytes over 0.2 TB"
    );
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
