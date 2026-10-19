// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The transcript schedule of `tessera-protocol/1` (RFC-0014 'Transcript schedule' and
//! 'Grinding'), checked against models written here from the specification:
//!
//! - the registry: every pattern has its own template, and every label follows its pattern;
//! - a worked expansion, every label derived here by hand from RFC-0014, RFC-0010 'Tree levels',
//!   RFC-0037 'Messages and transcript', RFC-0045 'The split' and RFC-0049 'The schedule';
//! - INV-FS-4: no label twice, on random sizes too;
//! - INV-FS-1: every prover message, in RFC-0036's order, is absorbed before each challenge that
//!   depends on it and after each challenge it is computed from, on random sizes too; and each
//!   violation is caught;
//! - the proofs of work: a nonce one bit short is refused at each phase, and grinding against a
//!   stub hash never stops below its target, whose bits are counted exactly;
//! - reordering two absorbs changes every later challenge;
//! - a transcript held to its schedule refuses every operation out of its place.

use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use proptest::prelude::*;
use tessera_codec::{Ascii, Sink};
use tessera_field::{Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_hash::{
    Digest, Domain, Hasher, Mode, ParamSetId, SessionContext, Transcript, TranscriptHash, hash,
};
use tessera_protocol::{
    ColumnQueries, Label, LabelPattern, LeafTree, MatrixCodeword, MatrixPoint, MatrixQueries, Op,
    ParamSet, PowRejected, ProofOfWork, Protocol, ProtocolError, ScheduleFault, ScheduleSizes,
    ScheduledTranscript, StageSizes, TranscriptSchedule, WEIGHTS_XI, WeightMatrix, WeightsXi,
};

type Fp2 = GoldilocksExt2;

// ----- Sizes and sessions -------------------------------------------------------------------------

/// Sizes that reach every rule of the schedule: forests that join at levels 0 and 1, a single
/// tree, a fraction tree that joins at the last level and one that joins none; a stage of no
/// round and a zero-check outside stage 1; a matrix with committed rows that the proof does not
/// open, and one whose point has no coordinate.
fn rich() -> ScheduleSizes {
    ScheduleSizes {
        trees: vec![
            LeafTree::Forest {
                trees_log: 1,
                height: 3,
            },
            LeafTree::Fraction { height: 1 },
            LeafTree::Fraction { height: 0 },
            LeafTree::Forest {
                trees_log: 2,
                height: 2,
            },
            LeafTree::Forest {
                trees_log: 0,
                height: 3,
            },
        ],
        layer_vars: 2,
        stages: vec![
            StageSizes {
                rounds: 3,
                zero_vars: 2,
            },
            StageSizes {
                rounds: 0,
                zero_vars: 0,
            },
            StageSizes {
                rounds: 2,
                zero_vars: 1,
            },
        ],
        advice_vars: 3,
        proximity: vec![
            MatrixPoint {
                matrix: WeightMatrix::Q,
                vars: 2,
            },
            MatrixPoint {
                matrix: WeightMatrix::Down,
                vars: 0,
            },
            MatrixPoint {
                matrix: WeightMatrix::ChannelParams,
                vars: 1,
            },
        ],
        opened: vec![
            MatrixCodeword {
                matrix: WeightMatrix::Q.id(),
                len: 1 << 10,
            },
            MatrixCodeword {
                matrix: WeightMatrix::ChannelParams.id(),
                len: 1 << 9,
            },
        ],
    }
}

/// The fewest sizes: no tree, no stage, no matrix.
fn bare() -> ScheduleSizes {
    ScheduleSizes {
        trees: Vec::new(),
        layer_vars: 0,
        stages: Vec::new(),
        advice_vars: 0,
        proximity: Vec::new(),
        opened: Vec::new(),
    }
}

/// `test-small/1` with the proofs of work of a hypothetical set: what the soundness calculator
/// calls a set built by hand, which no proof can use.
fn small_with_pow(seed_pow: u32, query_pow: u32) -> ParamSet {
    ParamSet {
        seed_pow,
        query_pow,
        ..ParamSet::TEST_SMALL
    }
}

fn expand(sizes: &ScheduleSizes, params: &ParamSet, mode: Mode) -> TranscriptSchedule {
    TranscriptSchedule::expand(sizes, params, mode, Protocol::One).expect("a schedule")
}

fn context(schedule: &TranscriptSchedule) -> SessionContext {
    SessionContext {
        protocol: Ascii::new("tessera-protocol/1").expect("ASCII"),
        mode: schedule.mode(),
        params: schedule.params().param_set_id(),
        statement_digest: [0x5c; 32],
        session_id: [0x2d; 16],
    }
}

fn labels(schedule: &TranscriptSchedule) -> Vec<String> {
    schedule.labels().map(|l| l.to_string()).collect()
}

fn fp2(a: u64, b: u64) -> Fp2 {
    Fp2::new(
        Goldilocks::from_u64_reduced(a),
        Goldilocks::from_u64_reduced(b),
    )
}

/// The stand-in message absorbed at step `i`: one element, distinct per step.
fn message(i: usize) -> Vec<Fp2> {
    vec![fp2(i as u64 + 1, 7)]
}

/// What a step gave the party that ran it.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Drawn {
    Fp2(Vec<Fp2>),
    Digest(Digest),
    Indices(Vec<u64>),
    Nonce(u64),
}

/// Runs every step of the transcript's schedule in order, absorbing stand-in messages: a prover
/// grinds (`nonces` is `None`); a verifier checks the prover's nonces in order. Stops at the
/// first rejected nonce.
fn play<H: TranscriptHash>(
    t: &mut ScheduledTranscript<'_, H>,
    schedule: &TranscriptSchedule,
    nonces: Option<&[u64]>,
) -> Result<Vec<Drawn>, PowRejected> {
    let mut drawn = Vec::new();
    let mut nonces = nonces.map(|n| n.iter().copied());
    for (i, step) in schedule.steps().iter().enumerate() {
        match step.op {
            Op::Absorb => match step.label {
                Label::Seed => t.absorb(step.label, &[0x5e; 32]),
                Label::Queries => t.absorb(
                    step.label,
                    &ColumnQueries {
                        advice: vec![1, 2, 3],
                        weights: vec![MatrixQueries {
                            matrix: 1,
                            indices: vec![4],
                        }],
                    },
                ),
                _ => t.absorb(step.label, &message(i)),
            },
            Op::Digest => drawn.push(Drawn::Digest(t.squeeze(step.label))),
            Op::Fp2 { .. } => drawn.push(Drawn::Fp2(t.point(step.label))),
            Op::Indices { .. } => drawn.push(Drawn::Indices(t.challenge_indices(step.label))),
            Op::Grind { .. } => match &mut nonces {
                None => drawn.push(Drawn::Nonce(t.grind(step.label))),
                Some(given) => {
                    let nonce = given.next().expect("a nonce per proof of work");
                    t.check_grind(step.label, nonce)?;
                    drawn.push(Drawn::Nonce(nonce));
                }
            },
        }
    }
    Ok(drawn)
}

fn nonces(drawn: &[Drawn]) -> Vec<u64> {
    drawn
        .iter()
        .filter_map(|d| match d {
            Drawn::Nonce(n) => Some(*n),
            _ => None,
        })
        .collect()
}

// ----- The registry -------------------------------------------------------------------------------

/// A label of `pattern` with sample indices: `ℓ = 3`, `j = 5`, `c = 2`, `k = 4`, `T = Gate`.
fn sample(pattern: LabelPattern) -> Label {
    let gate = WeightMatrix::Gate;
    match pattern {
        LabelPattern::Round1 => Label::Round1,
        LabelPattern::Seed => Label::Seed,
        LabelPattern::SeedPow => Label::SeedPow,
        LabelPattern::WeightsXi => Label::WeightsXi(gate),
        LabelPattern::LogUpAlpha => Label::LogUpAlpha,
        LabelPattern::LogUpBeta => Label::LogUpBeta,
        LabelPattern::LogUpRoots => Label::LogUpRoots,
        LabelPattern::TreeRoot => Label::TreeRoot { level: 3, coord: 2 },
        LabelPattern::TreeGamma => Label::TreeGamma { level: 3 },
        LabelPattern::TreeLambda => Label::TreeLambda { level: 3 },
        LabelPattern::TreeRound => Label::TreeRound { level: 3, round: 5 },
        LabelPattern::TreeChallenge => Label::TreeChallenge { level: 3, round: 5 },
        LabelPattern::TreeEvals => Label::TreeEvals { level: 3 },
        LabelPattern::TreeLine => Label::TreeLine { level: 3 },
        LabelPattern::LogUpValues => Label::LogUpValues,
        LabelPattern::StageLayer => Label::StageLayer { coord: 2 },
        LabelPattern::StageZero => Label::StageZero { stage: 4, coord: 2 },
        LabelPattern::StageGamma => Label::StageGamma { stage: 4 },
        LabelPattern::StageRound => Label::StageRound { stage: 4, round: 5 },
        LabelPattern::StageChallenge => Label::StageChallenge { stage: 4, round: 5 },
        LabelPattern::StageEvals => Label::StageEvals { stage: 4 },
        LabelPattern::StageSplit => Label::StageSplit { stage: 4 },
        LabelPattern::AdviceGamma => Label::AdviceGamma,
        LabelPattern::AdviceXi => Label::AdviceXi,
        LabelPattern::WeightsGamma => Label::WeightsGamma(gate),
        LabelPattern::AdviceCombined => Label::AdviceCombined,
        LabelPattern::WeightsCombined => Label::WeightsCombined(gate),
        LabelPattern::Scoring => Label::Scoring,
        LabelPattern::Queries => Label::Queries,
        LabelPattern::QueryPow => Label::QueryPow,
        LabelPattern::AdviceQueries => Label::AdviceQueries,
        LabelPattern::WeightsQueries => Label::WeightsQueries(gate),
    }
}

/// RFC-0014's table, as it writes the patterns of `tessera-protocol/1`, with RFC-0035's,
/// RFC-0037's and the proofs of work's.
const TEMPLATES: [&str; 32] = [
    "round1",
    "seed-pow",
    "seed",
    "logup/alpha",
    "logup/beta",
    "logup/roots",
    "tree/{ℓ}/root/{c}",
    "tree/{ℓ}/gamma",
    "tree/{ℓ}/lambda",
    "tree/{ℓ}/round/{j}",
    "tree/{ℓ}/r/{j}",
    "tree/{ℓ}/evals",
    "tree/{ℓ}/line",
    "logup/values",
    "stage/1/layer/{c}",
    "stage/{k}/zero/{c}",
    "stage/{k}/gamma",
    "stage/{k}/round/{j}",
    "stage/{k}/r/{j}",
    "stage/{k}/evals",
    "stage/{k}/split",
    "pcs/advice/gamma",
    "pcs/advice/xi",
    "pcs/weights/{T}/gamma",
    "pcs/weights/{T}/xi",
    "pcs/advice/combined",
    "pcs/weights/{T}/combined",
    "scoring",
    "queries",
    "query-pow",
    "pcs/advice/queries",
    "pcs/weights/{T}/queries",
];

/// The registry holds RFC-0014's patterns, each once, and every label is its pattern with its
/// indices written in decimal and `{T}` as the matrix's name: printable ASCII, which the
/// transcript requires.
#[test]
fn every_label_is_its_pattern_with_its_indices() {
    let templates: Vec<&str> = LabelPattern::ALL.iter().map(|p| p.template()).collect();
    assert_eq!(templates, TEMPLATES);
    let distinct: HashSet<&str> = templates.iter().copied().collect();
    assert_eq!(distinct.len(), templates.len(), "a template twice");
    for pattern in LabelPattern::ALL {
        let label = sample(pattern);
        assert_eq!(label.pattern(), pattern);
        let written = pattern
            .template()
            .replace("{ℓ}", "3")
            .replace("{j}", "5")
            .replace("{c}", "2")
            .replace("{k}", "4")
            .replace("{T}", "Gate");
        assert_eq!(label.to_string(), written, "{pattern:?}");
        assert!(
            written.bytes().all(|b| (0x20..=0x7E).contains(&b)),
            "{written} is printable ASCII"
        );
    }
    // RFC-0036 'Matrix ids': `{T}` is the variant's name.
    assert_eq!(
        Label::WeightsXi(WeightMatrix::ChannelParams).to_string(),
        "pcs/weights/ChannelParams/xi"
    );
    assert_eq!(
        Label::WeightsGamma(WeightMatrix::Q).to_string(),
        "pcs/weights/Q/gamma"
    );
}

// ----- The expansion ------------------------------------------------------------------------------

/// `rich()`'s schedule in a live session, every label derived by hand:
///
/// - row 3: the seed, then the points of Q, Down (no coordinate) and ChannelParams, RFC-0040's
///   order, all three with committed rows (RFC-0049);
/// - row 5: `H = 3`. Level 0: the forest of 2 trees and the single tree join, so one root
///   coordinate; no fraction tree, no `λ`; instances of 1 and 0 variables, one round. Level 1:
///   the forest of 4 trees joins, two root coordinates; instances of 2, 1 and 2 variables, two
///   rounds. Level 2: the fraction tree of height 1 joins, no root coordinate, `λ`; instances of
///   3, 2, 3 and 0 variables, three rounds. The fraction tree of height 0 joins no level;
/// - row 6: stage 1 draws the layer point, then the zero-check point, then `γ` (RFC-0037), stage
///   2 has no round, stage 3 has a zero-check point;
/// - rows 7 and 8: Q and ChannelParams are opened, Down is not.
const RICH_LIVE: &[&str] = &[
    "round1",
    "seed",
    "pcs/weights/Q/xi",
    "pcs/weights/Down/xi",
    "pcs/weights/ChannelParams/xi",
    "logup/alpha",
    "logup/beta",
    "logup/roots",
    "tree/0/root/0",
    "tree/0/gamma",
    "tree/0/round/0",
    "tree/0/r/0",
    "tree/0/evals",
    "tree/0/line",
    "tree/1/root/0",
    "tree/1/root/1",
    "tree/1/gamma",
    "tree/1/round/0",
    "tree/1/r/0",
    "tree/1/round/1",
    "tree/1/r/1",
    "tree/1/evals",
    "tree/1/line",
    "tree/2/gamma",
    "tree/2/lambda",
    "tree/2/round/0",
    "tree/2/r/0",
    "tree/2/round/1",
    "tree/2/r/1",
    "tree/2/round/2",
    "tree/2/r/2",
    "tree/2/evals",
    "tree/2/line",
    "logup/values",
    "stage/0/split",
    "stage/1/layer/0",
    "stage/1/layer/1",
    "stage/1/zero/0",
    "stage/1/zero/1",
    "stage/1/gamma",
    "stage/1/round/0",
    "stage/1/r/0",
    "stage/1/round/1",
    "stage/1/r/1",
    "stage/1/round/2",
    "stage/1/r/2",
    "stage/1/evals",
    "stage/1/split",
    "stage/2/gamma",
    "stage/2/evals",
    "stage/2/split",
    "stage/3/zero/0",
    "stage/3/gamma",
    "stage/3/round/0",
    "stage/3/r/0",
    "stage/3/round/1",
    "stage/3/r/1",
    "stage/3/evals",
    "stage/3/split",
    "pcs/advice/gamma",
    "pcs/advice/xi",
    "pcs/weights/Q/gamma",
    "pcs/weights/ChannelParams/gamma",
    "pcs/advice/combined",
    "pcs/weights/Q/combined",
    "pcs/weights/ChannelParams/combined",
    "scoring",
    "queries",
];

#[test]
fn a_live_schedule_is_rfc_0014s_table_expanded() {
    let schedule = expand(&rich(), &ParamSet::TEST_SMALL, Mode::Live);
    assert_eq!(labels(&schedule), RICH_LIVE);
    let op = |label: Label| schedule.steps()[schedule.position(label).expect("the label")].op;
    // A proximity point draws one challenge per coordinate; every other challenge is one.
    assert_eq!(op(Label::WeightsXi(WeightMatrix::Q)), Op::Fp2 { count: 2 });
    assert_eq!(
        op(Label::WeightsXi(WeightMatrix::Down)),
        Op::Fp2 { count: 0 }
    );
    assert_eq!(op(Label::AdviceXi), Op::Fp2 { count: 3 });
    assert_eq!(op(Label::TreeLambda { level: 2 }), Op::Fp2 { count: 1 });
    assert_eq!(op(Label::Seed), Op::Absorb);
    assert_eq!(op(Label::Queries), Op::Absorb);
}

/// Rows 3 and 10 of a non-interactive proof: the proofs of work and the squeezed seed and
/// indices, `τ_A` below `n_A = 2^{10 + 1}` and `τ_W` below each opened matrix's `n_T`.
#[test]
fn a_non_interactive_schedule_grinds_for_the_seed_and_the_queries() {
    let params = small_with_pow(7, 9);
    let schedule = expand(&rich(), &params, Mode::NonInteractive);
    let steps = schedule.steps();
    let live = expand(&rich(), &params, Mode::Live);
    // Rows 4 to 9 are the live session's.
    let middle = |s: &TranscriptSchedule| {
        let from = s.position(Label::LogUpAlpha).expect("alpha");
        let to = s.position(Label::Scoring).expect("scoring");
        s.steps()[from..=to].to_vec()
    };
    assert_eq!(middle(&schedule), middle(&live));
    let head: Vec<_> = steps[..3].iter().map(|s| (s.label, s.op)).collect();
    assert_eq!(
        head,
        [
            (Label::Round1, Op::Absorb),
            (
                Label::SeedPow,
                Op::Grind {
                    phase: ProofOfWork::Seed,
                    bits: 7
                }
            ),
            (Label::Seed, Op::Digest),
        ]
    );
    let scoring = schedule.position(Label::Scoring).expect("scoring");
    let tail: Vec<_> = steps[scoring + 1..]
        .iter()
        .map(|s| (s.label, s.op))
        .collect();
    assert_eq!(
        tail,
        [
            (
                Label::QueryPow,
                Op::Grind {
                    phase: ProofOfWork::Query,
                    bits: 9
                }
            ),
            (
                Label::AdviceQueries,
                Op::Indices {
                    log_n: 11,
                    count: 8
                }
            ),
            (
                Label::WeightsQueries(WeightMatrix::Q),
                Op::Indices {
                    log_n: 10,
                    count: 8
                }
            ),
            (
                Label::WeightsQueries(WeightMatrix::ChannelParams),
                Op::Indices { log_n: 9, count: 8 }
            ),
        ]
    );
}

/// RFC-0014 'Grinding': `α` and `β` are squeezed right after the seed, so that a new pair costs
/// a new seed nonce, and the indices right after the query nonce. No prover message comes between
/// a proof of work and the challenges it covers, which the prover could vary for free: between
/// `seed-pow` and `logup/beta`, and from `query-pow` on, the transcript only squeezes.
#[test]
fn the_proofs_of_work_cover_the_challenges_right_after_them() {
    for sizes in [rich(), bare()] {
        let schedule = expand(&sizes, &small_with_pow(3, 4), Mode::NonInteractive);
        let steps = schedule.steps();
        let at = |label: Label| schedule.position(label).expect("the label");
        let covered = |from: usize, to: usize| {
            steps[from + 1..=to]
                .iter()
                .all(|s| matches!(s.op, Op::Digest | Op::Fp2 { .. } | Op::Indices { .. }))
        };
        assert!(covered(at(Label::SeedPow), at(Label::LogUpBeta)));
        assert!(covered(at(Label::QueryPow), steps.len() - 1));
        // The verifier's live draws need no proof of work: a live schedule grinds nothing.
        let live = expand(&sizes, &small_with_pow(3, 4), Mode::Live);
        assert!(
            live.steps()
                .iter()
                .all(|s| !matches!(s.op, Op::Grind { .. }))
        );
    }
}

/// Each matrix's indices are `τ_W`, the advice's `τ_A`: at `ni-100/1`, 193 and 122.
#[test]
fn the_queries_take_the_parameter_sets_counts() {
    let schedule = expand(&rich(), &ParamSet::NI_100, Mode::NonInteractive);
    let counts: Vec<(Label, u32)> = schedule
        .steps()
        .iter()
        .filter_map(|s| match s.op {
            Op::Indices { count, .. } => Some((s.label, count)),
            _ => None,
        })
        .collect();
    assert_eq!(
        counts,
        [
            (Label::AdviceQueries, 193),
            (Label::WeightsQueries(WeightMatrix::Q), 122),
            (Label::WeightsQueries(WeightMatrix::ChannelParams), 122),
        ]
    );
}

/// With no tree, `logup/values` follows `logup/roots`, both absorbed even when empty, and the
/// leaf stage's splits follow them (RFC-0008 'Fraction tree', RFC-0045 'The split').
#[test]
fn a_schedule_without_trees_absorbs_the_empty_leaf_messages() {
    let schedule = expand(&bare(), &ParamSet::TEST_SMALL, Mode::Live);
    assert_eq!(
        labels(&schedule),
        [
            "round1",
            "seed",
            "logup/alpha",
            "logup/beta",
            "logup/roots",
            "logup/values",
            "stage/0/split",
            "pcs/advice/gamma",
            "pcs/advice/xi",
            "pcs/advice/combined",
            "scoring",
            "queries",
        ]
    );
    // Fraction trees of height 0 join no level either.
    let flat = ScheduleSizes {
        trees: vec![LeafTree::Fraction { height: 0 }; 4],
        ..bare()
    };
    assert_eq!(
        labels(&expand(&flat, &ParamSet::TEST_SMALL, Mode::Live)),
        labels(&schedule)
    );
}

/// RFC-0049's testing strategy: each `pcs/weights/{T}/xi` comes right after `seed`, for every
/// matrix with committed rows in RFC-0040's order, and none comes at row 7.
#[test]
fn the_weights_proximity_points_follow_the_seed() {
    // The registry's one placement of the weights' points is RFC-0049's row 3.
    assert_eq!(WEIGHTS_XI, WeightsXi::AfterSeed);
    assert_eq!(LabelPattern::WeightsXi.row(), LabelPattern::Seed.row());
    for mode in [Mode::Live, Mode::NonInteractive] {
        let schedule = expand(&rich(), &ParamSet::TEST_SMALL, mode);
        let seed = schedule.position(Label::Seed).expect("seed");
        let after: Vec<Label> = schedule.labels().skip(seed + 1).take(4).collect();
        assert_eq!(
            after,
            [
                Label::WeightsXi(WeightMatrix::Q),
                Label::WeightsXi(WeightMatrix::Down),
                Label::WeightsXi(WeightMatrix::ChannelParams),
                Label::LogUpAlpha,
            ],
            "{mode:?}"
        );
        let row_7 = schedule.position(Label::AdviceGamma).expect("row 7");
        let row_8 = schedule.position(Label::AdviceCombined).expect("row 8");
        assert!(
            schedule.steps()[row_7..row_8]
                .iter()
                .all(|s| !matches!(s.label, Label::WeightsXi(_))),
            "{mode:?}"
        );
    }
}

/// The schedule is refused, in this order, for `tessera-protocol/2` (#148), a mode the set does
/// not allow, a proof of work past 64 bits, and sizes that break a rule.
#[test]
fn a_schedule_no_proof_can_have_is_refused() {
    let refused = |sizes: &ScheduleSizes, params: &ParamSet, mode: Mode, protocol: Protocol| {
        match TranscriptSchedule::expand(sizes, params, mode, protocol) {
            Err(ProtocolError::Internal { rule }) => rule,
            other => panic!("refused, not {other:?}"),
        }
    };
    let one = Protocol::One;
    let small = ParamSet::TEST_SMALL;
    assert!(refused(&rich(), &small, Mode::Live, Protocol::Two).contains("#148"));
    assert!(refused(&rich(), &ParamSet::NI_100, Mode::Live, one).contains("mode"));
    assert!(refused(&rich(), &ParamSet::LIVE_64, Mode::NonInteractive, one).contains("mode"));
    assert!(refused(&rich(), &small_with_pow(65, 0), Mode::NonInteractive, one).contains("64"));
    assert!(refused(&rich(), &small_with_pow(0, 65), Mode::NonInteractive, one).contains("64"));
    // A live session grinds nothing.
    assert!(TranscriptSchedule::expand(&rich(), &small_with_pow(65, 65), Mode::Live, one).is_ok());

    let broken: Vec<(ScheduleSizes, &str)> = vec![
        (
            ScheduleSizes {
                trees: vec![LeafTree::Forest {
                    trees_log: 1,
                    height: 0,
                }],
                ..rich()
            },
            "a height of 1",
        ),
        (
            ScheduleSizes {
                trees: vec![LeafTree::Fraction { height: 65 }],
                ..rich()
            },
            "2^64 leaves",
        ),
        (
            ScheduleSizes {
                trees: vec![LeafTree::Forest {
                    trees_log: 60,
                    height: 5,
                }],
                ..rich()
            },
            "2^64 leaves",
        ),
        (
            ScheduleSizes {
                layer_vars: 33,
                ..rich()
            },
            "32 coordinates",
        ),
        (
            ScheduleSizes {
                stages: vec![StageSizes {
                    rounds: 2,
                    zero_vars: 3,
                }],
                ..rich()
            },
            "zero-check point",
        ),
        (
            ScheduleSizes {
                stages: vec![StageSizes {
                    rounds: 65,
                    zero_vars: 0,
                }],
                ..rich()
            },
            "64 rounds",
        ),
        (
            ScheduleSizes {
                stages: vec![
                    StageSizes {
                        rounds: 0,
                        zero_vars: 0
                    };
                    1 << 16
                ],
                ..rich()
            },
            "u16",
        ),
        (
            ScheduleSizes {
                advice_vars: 65,
                ..rich()
            },
            "64 coordinates",
        ),
        (
            ScheduleSizes {
                proximity: vec![
                    MatrixPoint {
                        matrix: WeightMatrix::Q,
                        vars: 1,
                    },
                    MatrixPoint {
                        matrix: WeightMatrix::Q,
                        vars: 1,
                    },
                ],
                opened: Vec::new(),
                ..rich()
            },
            "RFC-0040's order",
        ),
        (
            ScheduleSizes {
                opened: vec![MatrixCodeword { matrix: 10, len: 8 }],
                ..rich()
            },
            "is a weight matrix",
        ),
        (
            ScheduleSizes {
                opened: vec![
                    MatrixCodeword {
                        matrix: WeightMatrix::ChannelParams.id(),
                        len: 8,
                    },
                    MatrixCodeword {
                        matrix: WeightMatrix::Q.id(),
                        len: 8,
                    },
                ],
                ..rich()
            },
            "id order",
        ),
        (
            ScheduleSizes {
                opened: vec![
                    MatrixCodeword {
                        matrix: WeightMatrix::Q.id(),
                        len: 8,
                    };
                    2
                ],
                ..rich()
            },
            "id order",
        ),
        (
            ScheduleSizes {
                opened: vec![MatrixCodeword {
                    matrix: WeightMatrix::K.id(),
                    len: 8,
                }],
                ..rich()
            },
            "committed rows",
        ),
        (
            ScheduleSizes {
                opened: vec![MatrixCodeword {
                    matrix: WeightMatrix::Q.id(),
                    len: 12,
                }],
                ..rich()
            },
            "power of two",
        ),
        (
            ScheduleSizes {
                opened: vec![MatrixCodeword {
                    matrix: WeightMatrix::Q.id(),
                    len: 1 << 33,
                }],
                ..rich()
            },
            "power of two",
        ),
    ];
    for (sizes, rule) in broken {
        for mode in [Mode::Live, Mode::NonInteractive] {
            let found = refused(&sizes, &small, mode, one);
            assert!(found.contains(rule), "{rule:?} in {found:?}");
        }
    }
    // A set's counts of queries: the calculator's hypothetical sets reach the same check.
    let no_queries = ParamSet {
        advice_queries: 0,
        ..small
    };
    assert!(refused(&rich(), &no_queries, Mode::Live, one).contains("query counts"));
}

/// RFC-0014's stage labels include the largest accepted u16 stage index.
#[test]
fn the_last_u16_stage_expands_without_overflow() {
    let sizes = ScheduleSizes {
        stages: vec![
            StageSizes {
                rounds: 0,
                zero_vars: 0
            };
            usize::from(u16::MAX)
        ],
        ..bare()
    };
    for mode in [Mode::Live, Mode::NonInteractive] {
        let schedule = expand(&sizes, &ParamSet::TEST_SMALL, mode);
        assert!(
            schedule
                .position(Label::StageGamma { stage: u16::MAX })
                .is_some()
        );
        assert!(
            schedule
                .position(Label::StageSplit { stage: u16::MAX })
                .is_some()
        );
        distinct(&schedule).expect("INV-FS-4 at the stage-index boundary");
    }
}

// ----- INV-FS-4 -----------------------------------------------------------------------------------

/// INV-FS-4: no label twice.
fn distinct(schedule: &TranscriptSchedule) -> Result<(), String> {
    let mut seen = HashSet::new();
    for label in schedule.labels() {
        if !seen.insert(label.to_string()) {
            return Err(format!("{label} twice"));
        }
    }
    Ok(())
}

#[test]
fn no_label_is_used_twice() {
    for sizes in [rich(), bare()] {
        for mode in [Mode::Live, Mode::NonInteractive] {
            let schedule = expand(&sizes, &ParamSet::TEST_SMALL, mode);
            assert_eq!(distinct(&schedule), Ok(()));
        }
    }
}

// ----- INV-FS-1 -----------------------------------------------------------------------------------

/// A prover's message part, as RFC-0036 lays out `Round1`, `Round2Response` and `NiProof`:
/// what an absorb of the schedule carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Part {
    Round1,
    SeedNonce,
    LogUpRoots,
    TreeRound(u32, u32),
    TreeEvals(u32),
    LogUpValues,
    Splits(u16),
    StageRound(u16, u32),
    StageEvals(u16),
    AdviceCombined,
    WeightCombined(WeightMatrix),
    Scoring,
    QueryNonce,
}

/// The rounds of each tree level (RFC-0010 'Tree levels'): a tree of height `h` joins at
/// `H − h`, and at level `ℓ` it is an instance on `s + ℓ − (H − h)` variables. A level has as
/// many rounds as its largest instance has variables.
fn level_rounds(trees: &[LeafTree]) -> Vec<u32> {
    let tallest = trees.iter().map(|t| t.height()).max().unwrap_or(0);
    let mut rounds = vec![0; tallest as usize];
    for tree in trees {
        let join = tallest - tree.height();
        for (level, r) in rounds.iter_mut().enumerate().skip(join as usize) {
            *r = (*r).max(tree.trees_log() + level as u32 - join);
        }
    }
    rounds
}

/// RFC-0036's order of the prover's message parts: `Round1`, `NiProof.seed_nonce`, then
/// `Round2Response` as the verifier reads it, the leaf stage's roots, levels, values and splits,
/// each reduction stage's rounds, evaluations and splits, the combined rows and the scoring
/// claims, then `NiProof.query_nonce`.
fn message_order(sizes: &ScheduleSizes, mode: Mode) -> Vec<Part> {
    let mut parts = vec![Part::Round1];
    if mode == Mode::NonInteractive {
        parts.push(Part::SeedNonce);
    }
    parts.push(Part::LogUpRoots);
    for (level, &rounds) in (0u32..).zip(&level_rounds(&sizes.trees)) {
        parts.extend((0..rounds).map(|j| Part::TreeRound(level, j)));
        parts.push(Part::TreeEvals(level));
    }
    parts.extend([Part::LogUpValues, Part::Splits(0)]);
    for (k, stage) in (1u16..).zip(&sizes.stages) {
        parts.extend((0..stage.rounds).map(|j| Part::StageRound(k, j)));
        parts.extend([Part::StageEvals(k), Part::Splits(k)]);
    }
    parts.push(Part::AdviceCombined);
    for codeword in &sizes.opened {
        let matrix = WeightMatrix::from_id(codeword.matrix).expect("a matrix");
        parts.push(Part::WeightCombined(matrix));
    }
    parts.push(Part::Scoring);
    if mode == Mode::NonInteractive {
        parts.push(Part::QueryNonce);
    }
    parts
}

/// The part an absorb or a proof of work carries. The live seed is the coin toss and the live
/// queries are the verifier's: neither is the prover's.
fn carried(label: Label) -> Option<Part> {
    Some(match label {
        Label::Round1 => Part::Round1,
        Label::SeedPow => Part::SeedNonce,
        Label::LogUpRoots => Part::LogUpRoots,
        Label::TreeRound { level, round } => Part::TreeRound(level, round),
        Label::TreeEvals { level } => Part::TreeEvals(level),
        Label::LogUpValues => Part::LogUpValues,
        Label::StageSplit { stage } => Part::Splits(stage),
        Label::StageRound { stage, round } => Part::StageRound(stage, round),
        Label::StageEvals { stage } => Part::StageEvals(stage),
        Label::AdviceCombined => Part::AdviceCombined,
        Label::WeightsCombined(matrix) => Part::WeightCombined(matrix),
        Label::Scoring => Part::Scoring,
        Label::QueryPow => Part::QueryNonce,
        _ => return None,
    })
}

/// For a challenge, the last part it depends on, which must be absorbed before it is squeezed
/// (INV-FS-1), and the first part computed from it, which must be absorbed after it; `None` for
/// the end of the transcript. From the specification:
///
/// - the seed follows its nonce (RFC-0014 'Grinding'), and the weights' points and `α`, `β`
///   follow `Round1`: the weights are fixed in `R_W` before the session (RFC-0049) and the
///   lookups and multiplicities are committed in `Round1` (RFC-0008 'Batching');
/// - a level's or a stage's root point, `γ`, `λ`, layer and zero-check points follow the claims
///   they combine: the previous level's evaluations, the roots, the leaf stage's splits or the
///   previous stage's (RFC-0010, RFC-0037, RFC-0045 'Soundness');
/// - a round's challenge follows its round's message, and the line challenge the evaluations;
/// - the opening challenges follow every stage (RFC-0014 'Grinding'); the column indices follow
///   the query nonce, after every combined row (RFC-0011 step 7).
fn bounds(label: Label, sizes: &ScheduleSizes) -> (Part, Option<Part>) {
    let levels = level_rounds(&sizes.trees);
    let stages = u16::try_from(sizes.stages.len()).expect("a u16");
    let opened = |matrix: WeightMatrix| sizes.opened.iter().any(|c| c.matrix == matrix.id());
    // The first part of level `ℓ`, or what follows the last level.
    let level_start = |level: u32| match levels.get(level as usize) {
        Some(0) => Part::TreeEvals(level),
        Some(_) => Part::TreeRound(level, 0),
        None => Part::LogUpValues,
    };
    let after_level = |level: u32| match level.checked_sub(1) {
        Some(previous) => Part::TreeEvals(previous),
        None => Part::LogUpRoots,
    };
    let stage_start = |k: u16| match sizes.stages[usize::from(k) - 1].rounds {
        0 => Part::StageEvals(k),
        _ => Part::StageRound(k, 0),
    };
    let round_next = |k: u16, j: u32| match j + 1 < sizes.stages[usize::from(k) - 1].rounds {
        true => Part::StageRound(k, j + 1),
        false => Part::StageEvals(k),
    };
    match label {
        Label::Seed => (Part::SeedNonce, Some(Part::LogUpRoots)),
        // A matrix that is not opened sends no proximity row (RFC-0036 'weight_combined').
        Label::WeightsXi(matrix) => (
            Part::Round1,
            opened(matrix).then_some(Part::WeightCombined(matrix)),
        ),
        Label::LogUpAlpha | Label::LogUpBeta => (Part::Round1, Some(Part::LogUpRoots)),
        Label::TreeRoot { level, .. }
        | Label::TreeGamma { level }
        | Label::TreeLambda { level } => (after_level(level), Some(level_start(level))),
        Label::TreeChallenge { level, round } => {
            let next = match round + 1 < levels[level as usize] {
                true => Part::TreeRound(level, round + 1),
                false => Part::TreeEvals(level),
            };
            (Part::TreeRound(level, round), Some(next))
        }
        Label::TreeLine { level } => (Part::TreeEvals(level), Some(level_start(level + 1))),
        Label::StageLayer { .. } => (Part::Splits(0), Some(stage_start(1))),
        Label::StageZero { stage, .. } | Label::StageGamma { stage } => {
            (Part::Splits(stage - 1), Some(stage_start(stage)))
        }
        Label::StageChallenge { stage, round } => (
            Part::StageRound(stage, round),
            Some(round_next(stage, round)),
        ),
        Label::AdviceGamma | Label::AdviceXi => (Part::Splits(stages), Some(Part::AdviceCombined)),
        Label::WeightsGamma(matrix) => (Part::Splits(stages), Some(Part::WeightCombined(matrix))),
        Label::AdviceQueries | Label::WeightsQueries(_) => (Part::QueryNonce, None),
        other => panic!("{other} is no challenge"),
    }
}

/// INV-FS-1 against RFC-0036's message order: the prover's parts are absorbed in that order,
/// each challenge follows the seed and the last part it depends on, and precedes the first part
/// computed from it.
fn absorbs_precede_challenges(
    steps: &[(Label, Op)],
    sizes: &ScheduleSizes,
    mode: Mode,
) -> Result<(), String> {
    let mut at: HashMap<Part, usize> = HashMap::new();
    let mut order = Vec::new();
    for (i, &(label, op)) in steps.iter().enumerate() {
        if matches!(op, Op::Absorb | Op::Grind { .. })
            && let Some(part) = carried(label)
        {
            at.insert(part, i);
            order.push(part);
        }
    }
    let expected = message_order(sizes, mode);
    if order != expected {
        return Err(format!(
            "the absorbs are not RFC-0036's order: {order:?} against {expected:?}"
        ));
    }
    let seed = steps
        .iter()
        .position(|&(label, _)| label == Label::Seed)
        .ok_or("no seed")?;
    // The seed is fixed only after `Round1` (INV-PROTO-2), and a live session's queries only
    // after the whole `Round2Response` (INV-PROTO-3).
    if seed < at[&Part::Round1] {
        return Err("the seed comes before Round1".to_owned());
    }
    if let Some(queries) = steps.iter().position(|&(label, _)| label == Label::Queries)
        && at.values().any(|&i| i > queries)
    {
        return Err("the live queries come before a prover message".to_owned());
    }
    for (i, &(label, op)) in steps.iter().enumerate() {
        if !matches!(op, Op::Fp2 { .. } | Op::Digest | Op::Indices { .. }) {
            continue;
        }
        if label != Label::Seed && i < seed {
            return Err(format!("{label} is squeezed before the seed"));
        }
        let (after, before) = bounds(label, sizes);
        if at[&after] >= i {
            return Err(format!("{label} is squeezed before {after:?} is absorbed"));
        }
        if let Some(before) = before
            && at[&before] <= i
        {
            return Err(format!(
                "{label} is squeezed after {before:?}, computed from it"
            ));
        }
    }
    Ok(())
}

fn pairs(schedule: &TranscriptSchedule) -> Vec<(Label, Op)> {
    schedule.steps().iter().map(|s| (s.label, s.op)).collect()
}

#[test]
fn every_prover_message_is_absorbed_before_the_challenges_that_depend_on_it() {
    for sizes in [rich(), bare()] {
        for mode in [Mode::Live, Mode::NonInteractive] {
            let schedule = expand(&sizes, &ParamSet::TEST_SMALL, mode);
            assert_eq!(
                absorbs_precede_challenges(&pairs(&schedule), &sizes, mode),
                Ok(())
            );
        }
    }
}

/// The check of INV-FS-1 is live: moving a challenge before the message it depends on, or past
/// the message computed from it, or a message out of RFC-0036's order, is caught.
#[test]
fn each_violation_of_inv_fs_1_is_caught() {
    let sizes = rich();
    for mode in [Mode::Live, Mode::NonInteractive] {
        let schedule = expand(&sizes, &ParamSet::TEST_SMALL, mode);
        let steps = pairs(&schedule);
        let at = |label: Label| schedule.position(label).expect("the label");
        // Swap two steps, or move step `from` to just before `to`.
        let swapped = |a: usize, b: usize| {
            let mut s = steps.clone();
            s.swap(a, b);
            s
        };
        let moved = |from: usize, to: usize| {
            let mut s = steps.clone();
            let step = s.remove(from);
            s.insert(to, step);
            s
        };
        let violations = [
            // A round's challenge before its message.
            swapped(
                at(Label::TreeRound { level: 1, round: 1 }),
                at(Label::TreeChallenge { level: 1, round: 1 }),
            ),
            swapped(
                at(Label::StageRound { stage: 3, round: 0 }),
                at(Label::StageChallenge { stage: 3, round: 0 }),
            ),
            // The line challenge before the level's evaluations.
            swapped(
                at(Label::TreeEvals { level: 0 }),
                at(Label::TreeLine { level: 0 }),
            ),
            // `α` after the roots computed from it.
            moved(at(Label::LogUpAlpha), at(Label::LogUpRoots) + 1),
            // The layer point before the leaf stage's splits (RFC-0045 'Soundness').
            moved(
                at(Label::StageLayer { coord: 0 }),
                at(Label::StageSplit { stage: 0 }),
            ),
            // A stage's `γ` before the previous stage's splits.
            moved(
                at(Label::StageGamma { stage: 2 }),
                at(Label::StageSplit { stage: 1 }),
            ),
            // An opening challenge before the last stage's splits.
            moved(at(Label::AdviceGamma), at(Label::StageSplit { stage: 3 })),
            // `θ_T` after its matrix's combined rows.
            moved(
                at(Label::WeightsGamma(WeightMatrix::Q)),
                at(Label::WeightsCombined(WeightMatrix::Q)) + 1,
            ),
            // A weight point before the seed.
            moved(at(Label::WeightsXi(WeightMatrix::Q)), at(Label::Seed)),
            // Two messages out of RFC-0036's order.
            swapped(at(Label::LogUpValues), at(Label::StageSplit { stage: 0 })),
            swapped(at(Label::AdviceCombined), at(Label::Scoring)),
        ];
        for (n, mutant) in violations.iter().enumerate() {
            assert!(
                absorbs_precede_challenges(mutant, &sizes, mode).is_err(),
                "violation {n} in {mode:?} is not caught"
            );
        }
    }
    // Squeezing the indices before the query nonce.
    let schedule = expand(&sizes, &ParamSet::TEST_SMALL, Mode::NonInteractive);
    let mut steps = pairs(&schedule);
    let pow = schedule.position(Label::QueryPow).expect("query-pow");
    steps.swap(pow, pow + 1);
    assert!(absorbs_precede_challenges(&steps, &sizes, Mode::NonInteractive).is_err());
}

// ----- Proofs of work -----------------------------------------------------------------------------

/// The leading zero bits of `H("pow", state ‖ Ascii(label) ‖ nonce)`, most significant bit of
/// byte 0 first (RFC-0004 `grind`), computed here from the definition.
fn pow_zeros(state: &Digest, label: &str, nonce: u64) -> u32 {
    let mut input = state.to_vec();
    input.extend_from_slice(&(label.len() as u64).to_le_bytes());
    input.extend_from_slice(label.as_bytes());
    input.extend_from_slice(&nonce.to_le_bytes());
    let digest = hash(Domain::Pow, &input);
    let mut zeros = 0;
    for byte in digest {
        zeros += byte.leading_zeros();
        if byte != 0 {
            break;
        }
    }
    zeros
}

/// The first nonce with exactly `bits − 1` leading zero bits: one bit short.
fn one_bit_short(state: &Digest, label: &str, bits: u32) -> u64 {
    (0..)
        .find(|&nonce| pow_zeros(state, label, nonce) == bits - 1)
        .expect("a nonce one bit short")
}

/// A nonce one bit short of `g_seed` or `g_query` is refused at its phase with
/// `Reject(ProofOfWork)`, and is not absorbed; the prover's nonce meets the target, and the
/// verifier then draws the prover's seed and queries. At `ni-100/1`'s 20 bits, and at a small
/// hypothetical set's.
#[test]
fn a_nonce_one_bit_short_is_refused_at_each_phase() {
    for params in [ParamSet::NI_100, small_with_pow(5, 11)] {
        let sizes = ScheduleSizes {
            opened: vec![MatrixCodeword {
                matrix: WeightMatrix::Q.id(),
                len: 1 << 22,
            }],
            proximity: vec![MatrixPoint {
                matrix: WeightMatrix::Q,
                vars: 3,
            }],
            ..bare()
        };
        let schedule = expand(&sizes, &params, Mode::NonInteractive);
        let session = context(&schedule);
        let mut prover = ScheduledTranscript::new(&session, &schedule).expect("a transcript");
        let honest = play(&mut prover, &schedule, None).expect("the prover's run");
        prover.finish().expect("the prover's schedule");
        let honest_nonces = nonces(&honest);
        let [seed_nonce, query_nonce] = honest_nonces[..] else {
            panic!("two proofs of work");
        };

        // The seed's phase.
        let mut verifier = ScheduledTranscript::new(&session, &schedule).expect("a transcript");
        verifier.absorb(Label::Round1, &message(0));
        let state = verifier.transcript().state();
        assert!(pow_zeros(&state, "seed-pow", seed_nonce) >= params.seed_pow);
        let short = one_bit_short(&state, "seed-pow", params.seed_pow);
        assert_eq!(
            verifier.seed_check(short),
            Err(PowRejected {
                phase: ProofOfWork::Seed
            })
        );
        assert_eq!(
            verifier.transcript().state(),
            state,
            "the short nonce is not absorbed"
        );
        assert_eq!(
            verifier.finish().map(|_| ()),
            Err(ScheduleFault::Rejected(PowRejected {
                phase: ProofOfWork::Seed
            }))
        );

        // The queries' phase: everything up to `query-pow` honest.
        let mut verifier = ScheduledTranscript::new(&session, &schedule).expect("a transcript");
        let pow = schedule.position(Label::QueryPow).expect("query-pow");
        for (i, step) in schedule.steps()[..pow].iter().enumerate() {
            match step.op {
                Op::Absorb => verifier.absorb(step.label, &message(i)),
                Op::Grind { .. } => verifier
                    .check_grind(step.label, seed_nonce)
                    .expect("the honest seed nonce"),
                Op::Digest => {
                    verifier.squeeze(step.label);
                }
                Op::Fp2 { .. } => {
                    verifier.point(step.label);
                }
                Op::Indices { .. } => unreachable!("indices follow query-pow"),
            }
        }
        let state = verifier.transcript().state();
        assert!(pow_zeros(&state, "query-pow", query_nonce) >= params.query_pow);
        let short = one_bit_short(&state, "query-pow", params.query_pow);
        assert_eq!(
            verifier.queries_check(short),
            Err(PowRejected {
                phase: ProofOfWork::Query
            })
        );
        assert_eq!(
            verifier.transcript().state(),
            state,
            "the short nonce is not absorbed"
        );

        // With the prover's nonces, the verifier draws what the prover drew.
        let mut verifier = ScheduledTranscript::new(&session, &schedule).expect("a transcript");
        let checked = play(&mut verifier, &schedule, Some(&honest_nonces)).expect("accepted");
        assert_eq!(checked, honest);
        verifier.finish().expect("the verifier's schedule");
    }
}

/// The phases' own calls: a prover's `seed_grind` and `queries_grind` and a verifier's
/// `seed_check` and `queries_check` agree, the queries being the squeezed indices as
/// `ColumnQueries`, and in a live session the seed and the queries are absorbed at their places.
#[test]
fn the_seed_and_queries_phases_of_both_modes_agree() {
    let sizes = rich();
    let params = small_with_pow(6, 6);
    let schedule = expand(&sizes, &params, Mode::NonInteractive);
    let session = context(&schedule);
    let run = |verify: Option<(u64, u64)>| {
        let mut t = ScheduledTranscript::new(&session, &schedule).expect("a transcript");
        t.absorb(Label::Round1, &message(0));
        let (seed_nonce, seed) = match verify {
            None => t.seed_grind(),
            Some((n, _)) => (n, t.seed_check(n).expect("the seed nonce")),
        };
        let row_3 = schedule.position(Label::Seed).expect("seed") + 1;
        let pow = schedule.position(Label::QueryPow).expect("query-pow");
        for (i, step) in schedule.steps().iter().enumerate().take(pow).skip(row_3) {
            match step.op {
                Op::Absorb => t.absorb(step.label, &message(i)),
                _ => {
                    t.point(step.label);
                }
            }
        }
        let (query_nonce, queries) = match verify {
            None => t.queries_grind(),
            Some((_, n)) => (n, t.queries_check(n).expect("the query nonce")),
        };
        t.finish().expect("the schedule");
        (seed_nonce, seed, query_nonce, queries)
    };
    let (seed_nonce, seed, query_nonce, queries) = run(None);
    assert_eq!(
        run(Some((seed_nonce, query_nonce))),
        (seed_nonce, seed, query_nonce, queries.clone())
    );
    // The queries have the parameter set's counts, below each codeword's length.
    assert!(sizes.query_shape(&params).check(&queries).is_ok());
    assert_eq!(queries.advice.len(), 8);
    let matrices: Vec<u8> = queries.weights.iter().map(|m| m.matrix).collect();
    assert_eq!(
        matrices,
        [WeightMatrix::Q.id(), WeightMatrix::ChannelParams.id()]
    );

    // A live session absorbs the coin-tossed seed under `seed` and the queries under `queries`.
    let live = expand(&sizes, &params, Mode::Live);
    let session = context(&live);
    let mut t = ScheduledTranscript::new(&session, &live).expect("a transcript");
    let mut reference = Transcript::new(&session);
    t.absorb(Label::Round1, &message(0));
    reference.absorb("round1", &message(0));
    t.seed_live(&seed);
    reference.absorb("seed", &seed);
    assert_eq!(t.transcript().state(), reference.state());
    let last = live.position(Label::Queries).expect("queries");
    for (i, step) in live.steps().iter().enumerate().take(last).skip(2) {
        match step.op {
            Op::Absorb => t.absorb(step.label, &message(i)),
            _ => {
                t.point(step.label);
            }
        }
    }
    t.queries_live(&queries);
    t.finish().expect("the live schedule");
}

/// A hash whose proof-of-work digests are scripted: nonce `n` has exactly `zeros[n]` leading
/// zero bits, and every other domain is BLAKE3's. It counts the proof-of-work hashes.
struct Scripted {
    zeros: Vec<u32>,
    pows: Rc<Cell<u64>>,
}

struct Input {
    domain: Domain,
    bytes: Vec<u8>,
    hasher: Hasher,
}

impl Sink for Input {
    fn put(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
        self.hasher.update(bytes);
    }
}

/// A digest of exactly `zeros` leading zero bits, most significant bit of byte 0 first: zero
/// bytes, then a byte whose first one is at bit `zeros mod 8`, then ones.
fn with_zeros(zeros: u32) -> Digest {
    let mut digest = [0xFF; 32];
    let full = (zeros / 8) as usize;
    digest[..full].fill(0);
    if full < 32 {
        digest[full] = 0xFF >> (zeros % 8);
    }
    digest
}

impl TranscriptHash for Scripted {
    type Input = Input;

    fn input(&self, domain: Domain) -> Input {
        Input {
            domain,
            bytes: Vec::new(),
            hasher: Hasher::new(domain),
        }
    }

    fn digest(&self, input: Input) -> Digest {
        if input.domain != Domain::Pow {
            return input.hasher.finalize();
        }
        self.pows.set(self.pows.get() + 1);
        let mut nonce = [0; 8];
        nonce.copy_from_slice(&input.bytes[input.bytes.len() - 8..]);
        let n = u64::from_le_bytes(nonce) as usize;
        with_zeros(self.zeros.get(n).copied().unwrap_or(0))
    }
}

/// Grinding against a stub hash, whose nonces `0..40` all fall short of 13 bits, by up to one
/// bit, and whose nonce 40 has exactly 13, measured by the hashes it makes: the prover hashes
/// nonces 0 to 40, each once, and stops at 40, never below the target. The verifier counts the
/// bits exactly with one hash per nonce: 12 is refused, 13 and 16 are accepted, 8 is refused, at
/// every bit position within a byte.
#[test]
fn grinding_never_stops_below_its_target() {
    const TARGET: u32 = 13;
    let mut zeros: Vec<u32> = (0..40).map(|n| n % TARGET).collect();
    zeros.extend([TARGET, 16, 8, TARGET - 1]);
    assert!(zeros[..40].iter().all(|&z| z < TARGET));
    let params = small_with_pow(TARGET, 0);
    let schedule = expand(&bare(), &params, Mode::NonInteractive);
    let session = context(&schedule);
    let pows = Rc::new(Cell::new(0));
    let stub = || Scripted {
        zeros: zeros.clone(),
        pows: Rc::clone(&pows),
    };

    let mut prover =
        ScheduledTranscript::with_hash(&session, stub(), &schedule).expect("a transcript");
    prover.absorb(Label::Round1, &message(0));
    let (nonce, _) = prover.seed_grind();
    assert_eq!(nonce, 40);
    assert_eq!(pows.get(), 41, "nonces 0 to 40, each hashed once");

    let accepted = |nonce: u64| {
        pows.set(0);
        let mut t =
            ScheduledTranscript::with_hash(&session, stub(), &schedule).expect("a transcript");
        t.absorb(Label::Round1, &message(0));
        let outcome = t.seed_check(nonce);
        assert_eq!(pows.get(), 1, "one hash per check");
        match outcome {
            Ok(_) => true,
            Err(rejected) => {
                assert_eq!(rejected.phase, ProofOfWork::Seed);
                false
            }
        }
    };
    for n in 0..40 {
        assert!(!accepted(n), "nonce {n} has {} bits", zeros[n as usize]);
    }
    assert!(accepted(40), "exactly the target");
    assert!(
        accepted(41),
        "more than the target, a nonce not the smallest"
    );
    assert!(!accepted(42), "8 bits");
    assert!(!accepted(43), "one bit short");
    // The bits are counted most significant first: 13 zeros, then a one at bit 5 of byte 1.
    assert_eq!(with_zeros(13)[..2], [0x00, 0x07]);
}

// ----- Reordering ---------------------------------------------------------------------------------

/// Every output a run draws, in order, as raw transcript calls, each challenge in `Fp2` as its
/// two words, each index and each word of the seed: `swap` exchanges the steps at those two
/// positions, labels and messages alike. Proofs of work ask 0 bits, so the nonces are 0.
fn outputs(schedule: &TranscriptSchedule, swap: Option<(usize, usize)>) -> Vec<u64> {
    let mut steps: Vec<(usize, Label, Op)> = schedule
        .steps()
        .iter()
        .enumerate()
        .map(|(i, s)| (i, s.label, s.op))
        .collect();
    if let Some((a, b)) = swap {
        steps.swap(a, b);
    }
    let mut t = Transcript::new(&context(schedule));
    let mut out = Vec::new();
    for (i, label, op) in steps {
        let text = label.to_string();
        match op {
            Op::Absorb => t.absorb(&text, &message(i)),
            Op::Fp2 { count } => {
                for _ in 0..count {
                    let [c0, c1] = t.challenge_fp2(&text).coeffs();
                    out.extend([c0.to_canonical_u64(), c1.to_canonical_u64()]);
                }
            }
            Op::Indices { log_n, count } => out.extend(t.challenge_indices(&text, log_n, count)),
            Op::Digest => out.extend(
                t.squeeze(&text)
                    .chunks_exact(8)
                    .map(|w| u64::from_le_bytes(w.try_into().expect("8 bytes"))),
            ),
            Op::Grind { bits, .. } => {
                assert_eq!(bits, 0, "test-small/1 grinds for no bit");
                assert_eq!(t.grind(&text, 0), 0);
            }
        }
    }
    out
}

/// Exchanging any two adjacent absorbs changes every output drawn after them, and none drawn
/// before: the challenges, the seed and the column indices. In both modes, for every pair of
/// adjacent absorbs of `rich()`'s schedule.
#[test]
fn reordering_two_absorbs_changes_every_later_challenge() {
    let mut pairs = 0;
    for mode in [Mode::Live, Mode::NonInteractive] {
        let schedule = expand(&rich(), &ParamSet::TEST_SMALL, mode);
        let honest = outputs(&schedule, None);
        let steps = schedule.steps();
        for i in 0..steps.len() - 1 {
            if steps[i].op != Op::Absorb || steps[i + 1].op != Op::Absorb {
                continue;
            }
            // The outputs drawn before step `i`.
            let before = honest.len() - drawn_from(&schedule, i);
            let reordered = outputs(&schedule, Some((i, i + 1)));
            assert_eq!(reordered.len(), honest.len());
            assert_eq!(reordered[..before], honest[..before], "{mode:?}, step {i}");
            for (k, (x, y)) in reordered.iter().zip(&honest).enumerate().skip(before) {
                assert_ne!(
                    x,
                    y,
                    "{mode:?}: output {k} after exchanging steps {i} and {}",
                    i + 1
                );
            }
            pairs += usize::from(before < honest.len());
        }
    }
    // Pairs with outputs after them: round1 and seed, the leaf stage's values and splits, each
    // stage's evaluations and splits; in a non-interactive proof the combined rows and scoring too.
    assert!(pairs >= 10, "{pairs} pairs");
}

/// How many outputs the steps from `from` on draw.
fn drawn_from(schedule: &TranscriptSchedule, from: usize) -> usize {
    schedule.steps()[from..]
        .iter()
        .map(|s| match s.op {
            Op::Fp2 { count } => 2 * count as usize,
            Op::Indices { count, .. } => count as usize,
            Op::Digest => 4,
            Op::Absorb | Op::Grind { .. } => 0,
        })
        .sum()
}

// ----- A transcript held to its schedule ----------------------------------------------------------

fn scheduled<'s>(schedule: &'s TranscriptSchedule) -> ScheduledTranscript<'s> {
    ScheduledTranscript::new(&context(schedule), schedule).expect("a transcript")
}

/// A prover and a verifier held to one schedule draw the same challenges and end in the same
/// state, in both modes, and the run's transcript is the raw transcript's.
#[test]
fn both_sides_held_to_one_schedule_agree() {
    for (mode, params) in [
        (Mode::Live, ParamSet::TEST_SMALL),
        (Mode::NonInteractive, small_with_pow(4, 5)),
    ] {
        let schedule = expand(&rich(), &params, mode);
        let mut prover = scheduled(&schedule);
        let drawn = play(&mut prover, &schedule, None).expect("the prover");
        let mut verifier = scheduled(&schedule);
        let checked = play(&mut verifier, &schedule, Some(&nonces(&drawn))).expect("accepted");
        assert_eq!(drawn, checked);
        let (p, v) = (
            prover.finish().expect("the prover's"),
            verifier.finish().expect("the verifier's"),
        );
        assert_eq!(p.state(), v.state());
        assert_eq!(p.counter(), v.counter());
    }
}

/// Every operation out of its place is a fault at its step: another label, another operation,
/// a point drawn in part, a step skipped, an operation past the end; and stopping early leaves
/// the schedule unfinished.
#[test]
fn an_operation_out_of_its_place_is_a_fault() {
    let schedule = expand(&rich(), &ParamSet::TEST_SMALL, Mode::Live);
    let diverged = |step: usize, label: &str| {
        Err(ScheduleFault::Diverged {
            step,
            label: label.to_owned(),
        })
    };

    // Another label: `logup/alpha` where `seed` is next.
    let mut t = scheduled(&schedule);
    t.absorb(Label::Round1, &message(0));
    t.challenge_fp2(Label::LogUpAlpha);
    assert_eq!(t.finish().map(|_| ()), diverged(1, "logup/alpha"));

    // Another operation under the right label: absorbing where a challenge is squeezed.
    let mut t = scheduled(&schedule);
    let alpha = schedule.position(Label::LogUpAlpha).expect("alpha");
    play_until(&mut t, &schedule, alpha);
    t.absorb(Label::LogUpAlpha, &message(0));
    assert_eq!(t.finish().map(|_| ()), diverged(alpha, "logup/alpha"));

    // A point drawn in part: one of Q's two coordinates, then the next point.
    let mut t = scheduled(&schedule);
    let q = schedule
        .position(Label::WeightsXi(WeightMatrix::Q))
        .expect("Q's point");
    play_until(&mut t, &schedule, q);
    t.challenge_fp2(Label::WeightsXi(WeightMatrix::Q));
    t.point(Label::WeightsXi(WeightMatrix::ChannelParams));
    assert_eq!(
        t.finish().map(|_| ()),
        diverged(q, "pcs/weights/ChannelParams/xi")
    );

    // A step skipped: `tree/0/root/0` left out.
    let mut t = scheduled(&schedule);
    let root = schedule
        .position(Label::TreeRoot { level: 0, coord: 0 })
        .expect("a root coordinate");
    play_until(&mut t, &schedule, root);
    t.challenge_fp2(Label::TreeGamma { level: 0 });
    assert_eq!(t.finish().map(|_| ()), diverged(root, "tree/0/gamma"));

    // Stopping early.
    let mut t = scheduled(&schedule);
    play_until(&mut t, &schedule, root);
    assert_eq!(
        t.finish().map(|_| ()),
        Err(ScheduleFault::Unfinished { step: root })
    );

    // Past the end.
    let mut t = scheduled(&schedule);
    play(&mut t, &schedule, None).expect("a run");
    t.absorb(Label::Scoring, &message(0));
    assert_eq!(
        t.finish().map(|_| ()),
        diverged(schedule.steps().len(), "scoring")
    );

    // The first fault is kept, and the operations still run on the transcript.
    let mut t = scheduled(&schedule);
    t.absorb(Label::Seed, &[0u8; 32]);
    t.absorb(Label::Round1, &message(0));
    assert_eq!(
        t.fault(),
        Some(&ScheduleFault::Diverged {
            step: 0,
            label: "seed".to_owned()
        })
    );
    let mut raw = Transcript::new(&context(&schedule));
    raw.absorb("seed", &[0u8; 32]);
    raw.absorb("round1", &message(0));
    assert_eq!(t.transcript().state(), raw.state());
}

/// A verifier never accepts a proof of work out of its place: the check refuses it, and the
/// fault is recorded.
#[test]
fn a_proof_of_work_out_of_its_place_is_refused() {
    let params = small_with_pow(0, 0);
    let schedule = expand(&rich(), &params, Mode::NonInteractive);
    let mut t = scheduled(&schedule);
    // Before `round1`: every nonce meets 0 bits, but the step is `round1`.
    assert_eq!(
        t.check_grind(Label::SeedPow, 0),
        Err(PowRejected {
            phase: ProofOfWork::Seed
        })
    );
    assert!(matches!(
        t.fault(),
        Some(ScheduleFault::Diverged { step: 0, .. })
    ));
    let mut t = scheduled(&schedule);
    t.absorb(Label::Round1, &message(0));
    assert_eq!(
        t.queries_check(0),
        Err(PowRejected {
            phase: ProofOfWork::Query
        })
    );
}

/// A point of no coordinate draws nothing: the transcript passes it, drawing it gives no
/// coordinate, and a challenge drawn under its label is one too many.
#[test]
fn a_point_of_no_coordinate_draws_nothing() {
    let schedule = expand(&rich(), &ParamSet::TEST_SMALL, Mode::Live);
    let down = schedule
        .position(Label::WeightsXi(WeightMatrix::Down))
        .expect("Down's point");
    let mut passed = scheduled(&schedule);
    play_until(&mut passed, &schedule, down);
    let mut drawn = scheduled(&schedule);
    play_until(&mut drawn, &schedule, down);
    assert_eq!(drawn.point(Label::WeightsXi(WeightMatrix::Down)), []);
    for t in [&mut passed, &mut drawn] {
        assert_eq!(
            t.point(Label::WeightsXi(WeightMatrix::ChannelParams)).len(),
            1
        );
        assert_eq!(t.fault(), None);
    }
    assert_eq!(passed.transcript().state(), drawn.transcript().state());

    // Another operation under its label is out of its place too.
    let mut absorbed = scheduled(&schedule);
    play_until(&mut absorbed, &schedule, down);
    absorbed.absorb(Label::WeightsXi(WeightMatrix::Down), &message(down));
    assert_eq!(
        absorbed.fault(),
        Some(&ScheduleFault::Diverged {
            step: down,
            label: "pcs/weights/Down/xi".to_owned()
        })
    );

    let mut extra = scheduled(&schedule);
    play_until(&mut extra, &schedule, down);
    extra.challenge_fp2(Label::WeightsXi(WeightMatrix::Down));
    assert_eq!(
        extra.fault(),
        Some(&ScheduleFault::Diverged {
            step: down,
            label: "pcs/weights/Down/xi".to_owned()
        })
    );
}

/// A session of another mode, parameter set or protocol than the schedule's is refused.
#[test]
fn a_session_that_is_not_the_schedules_is_refused() {
    let schedule = expand(&rich(), &ParamSet::TEST_SMALL, Mode::Live);
    let good = context(&schedule);
    let others = [
        SessionContext {
            mode: Mode::NonInteractive,
            ..good.clone()
        },
        SessionContext {
            params: ParamSetId(Ascii::new("live-64/1").expect("ASCII")),
            ..good.clone()
        },
        SessionContext {
            protocol: Ascii::new("tessera-protocol/2").expect("ASCII"),
            ..good.clone()
        },
    ];
    for other in others {
        assert!(matches!(
            ScheduledTranscript::new(&other, &schedule),
            Err(ProtocolError::Internal { .. })
        ));
    }
    assert!(ScheduledTranscript::new(&good, &schedule).is_ok());
}

/// Plays the first `steps` steps as a prover.
fn play_until(t: &mut ScheduledTranscript<'_>, schedule: &TranscriptSchedule, steps: usize) {
    for (i, step) in schedule.steps()[..steps].iter().enumerate() {
        match step.op {
            Op::Absorb if step.label == Label::Seed => t.absorb(step.label, &[0x5e; 32]),
            Op::Absorb => t.absorb(step.label, &message(i)),
            Op::Digest => {
                t.squeeze(step.label);
            }
            Op::Fp2 { .. } => {
                t.point(step.label);
            }
            Op::Indices { .. } => {
                t.challenge_indices(step.label);
            }
            Op::Grind { .. } => {
                t.grind(step.label);
            }
        }
    }
}

// ----- Random sizes -------------------------------------------------------------------------------

fn trees() -> impl Strategy<Value = Vec<LeafTree>> {
    let tree = prop_oneof![
        (0u32..6).prop_map(|height| LeafTree::Fraction { height }),
        (0u32..4, 1u32..6).prop_map(|(trees_log, height)| LeafTree::Forest { trees_log, height }),
    ];
    prop::collection::vec(tree, 0..6)
}

fn stages() -> impl Strategy<Value = Vec<StageSizes>> {
    prop::collection::vec(
        (0u32..6).prop_flat_map(|rounds| {
            (Just(rounds), 0..=rounds)
                .prop_map(|(rounds, zero_vars)| StageSizes { rounds, zero_vars })
        }),
        0..4,
    )
}

/// Matrices with committed rows, each with its point, and the opened ones among them.
fn matrices() -> impl Strategy<Value = (Vec<MatrixPoint>, Vec<MatrixCodeword>)> {
    prop::collection::vec((any::<bool>(), 0u32..4, any::<bool>(), 1u32..12), 10).prop_map(
        |choices| {
            let mut proximity = Vec::new();
            let mut opened = Vec::new();
            for (matrix, (committed, vars, open, log_n)) in
                WeightMatrix::ALL.into_iter().zip(choices)
            {
                if committed {
                    proximity.push(MatrixPoint { matrix, vars });
                    if open {
                        opened.push(MatrixCodeword {
                            matrix: matrix.id(),
                            len: 1 << log_n,
                        });
                    }
                }
            }
            (proximity, opened)
        },
    )
}

fn sizes() -> impl Strategy<Value = ScheduleSizes> {
    (trees(), 0u32..4, stages(), 0u32..5, matrices()).prop_map(
        |(trees, layer_vars, stages, advice_vars, (proximity, opened))| ScheduleSizes {
            trees,
            layer_vars,
            stages,
            advice_vars,
            proximity,
            opened,
        },
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    /// On random sizes, in both modes: no label twice (INV-FS-4), every prover message absorbed
    /// before the challenges that depend on it (INV-FS-1), and a prover and a verifier held to
    /// the schedule agree.
    #[test]
    fn random_schedules_keep_their_invariants(sizes in sizes(), seed_pow in 0u32..4, query_pow in 0u32..4) {
        let params = small_with_pow(seed_pow, query_pow);
        for mode in [Mode::Live, Mode::NonInteractive] {
            let schedule = expand(&sizes, &params, mode);
            prop_assert_eq!(distinct(&schedule), Ok(()));
            prop_assert_eq!(absorbs_precede_challenges(&pairs(&schedule), &sizes, mode), Ok(()));
            let mut prover = scheduled(&schedule);
            let drawn = play(&mut prover, &schedule, None).expect("the prover");
            let mut verifier = scheduled(&schedule);
            let checked = play(&mut verifier, &schedule, Some(&nonces(&drawn)));
            prop_assert_eq!(checked, Ok(drawn));
            prop_assert!(prover.finish().is_ok());
            prop_assert!(verifier.finish().is_ok());
        }
    }
}
