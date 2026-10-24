// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The soundness calculator (RFC-0014 'Soundness calculator').
//!
//! - **The oracle's reports.** `spec/vectors/soundness/soundness.json`, written by an
//!   independent oracle (`generate.py`), holds `tessera.soundness/1` reports for RFC-0014's
//!   sets at the 405B 16K-token, 16K-context statement, for statements and parameters that miss
//!   one target each, at the edges of the exact comparisons, and at random sizes under random
//!   parameter sets. Every row is the oracle's: its bound, bits, per-query bits, grinding, target,
//!   margin and decision, and so are the minimum and the verdict on the targets.
//! - **Never above the true value.** Every FixedBits `r` satisfies `r ≤ 256·λ < r + 1 + 2^-20`,
//!   `λ = g + max(0, −log2 ε)` computed by the oracle with 60-digit decimals, not by LOG2.
//! - **#103's criteria.** `live-64` and `ni-100` meet their targets at the 405B sizes and
//!   `test-small` is held to none; `N + T = 2^50` fails `ni-100`'s LogUp row; the per-query bits
//!   are `−log2((1 + ρ')/2)` within the precision; RFC-0014's printed figures are checked against
//!   the exact bounds.
//! - **RFC-0049.** `tessera-protocol/1` counts a weight matrix by its committed rows with no mask,
//!   and gives a matrix with no committed row no proximity rows; `tessera-protocol/2` counts its
//!   blinding rows and mask coefficients.
//! - **Adversarial.** For every row, a size or parameter that makes exactly that row miss its
//!   target, which is then the minimum; and decisions at the boundary that rounded bits would get
//!   wrong.
//! - **Refused.** Sizes the calculator cannot bound are `ERR-USAGE`; a parameter set the build
//!   does not accept is refused by id.

mod common;

use std::fs;

use serde_json::Value;
use tessera_codec::Ascii;
use tessera_codec::error::ErrorCode;
use tessera_hash::{Mode, ParamSetId};
use tessera_ir::{AdviceLayoutParams, CommitmentLayout, WeightLayoutParams};
use tessera_protocol::{
    Commitment, ErrorBound, FixedBits, LeafGroupSizes, LookupSizes, MatrixSizes, ParamMode,
    ParamSet, ProofOfWork, ProtocolError, Rounds, RowKind, SoundnessReport, SoundnessRow,
    StatementSizes, Targets, WeightMatrix,
};

fn load() -> Value {
    let path = common::vectors().join("soundness/soundness.json");
    serde_json::from_str(&fs::read_to_string(path).expect("vector file")).expect("JSON")
}

fn text(value: &Value) -> &str {
    value
        .as_str()
        .unwrap_or_else(|| panic!("a string: {value}"))
}

fn int<T: TryFrom<u64>>(value: &Value) -> T {
    let n = value
        .as_u64()
        .unwrap_or_else(|| panic!("an integer: {value}"));
    T::try_from(n).unwrap_or_else(|_| panic!("{n} fits its type"))
}

fn list(value: &Value) -> &Vec<Value> {
    value
        .as_array()
        .unwrap_or_else(|| panic!("a list: {value}"))
}

fn matrix(name: &str) -> WeightMatrix {
    *WeightMatrix::ALL
        .iter()
        .find(|m| m.name() == name)
        .unwrap_or_else(|| panic!("a matrix: {name}"))
}

/// A parameter set as the oracle states it. A random set's id is not RFC-0014's: the calculator
/// reads its values, never its name.
fn params(value: &Value) -> ParamSet {
    let advice = &value["advice"];
    let weights = &value["weights"];
    ParamSet {
        id: Box::leak(text(&value["id"]).to_owned().into_boxed_str()),
        mode: match text(&value["mode"]) {
            "live" => ParamMode::Live,
            "non-interactive" => ParamMode::NonInteractive,
            "either" => ParamMode::Either,
            other => panic!("a mode: {other}"),
        },
        layout: CommitmentLayout {
            advice: AdviceLayoutParams {
                log2_k: int(&advice["log2_k"]),
                log2_inv_rate: int(&advice["log2_inv_rate"]),
                mask_coeffs: int(&advice["mask_coeffs"]),
                row_block: int(&advice["row_block"]),
                segments: int(&advice["segments"]),
            },
            weights: WeightLayoutParams {
                log2_k: int(&weights["log2_k"]),
                log2_inv_rate: int(&weights["log2_inv_rate"]),
                mask_coeffs: int(&weights["mask_coeffs"]),
                blinding_rows: int(&weights["blinding_rows"]),
            },
        },
        advice_queries: int(&value["advice_queries"]),
        weight_queries: int(&value["weight_queries"]),
        seed_pow: int(&value["seed_pow"]),
        query_pow: int(&value["query_pow"]),
        release: value["release"].as_bool().expect("a boolean"),
        targets: match &value["targets"] {
            Value::Null => None,
            t => Some(Targets {
                bits: int(&t["bits"]),
                one_shot: int(&t["one_shot"]),
            }),
        },
    }
}

/// Statement sizes as the oracle states them.
fn sizes(value: &Value) -> StatementSizes {
    let ints = |key: &str| -> Vec<u64> { list(&value[key]).iter().map(int).collect() };
    let small = |key: &str| -> Vec<u32> { list(&value[key]).iter().map(int).collect() };
    StatementSizes {
        protocol: int(&value["protocol"]),
        rounds: list(&value["rounds"])
            .iter()
            .map(|r| Rounds {
                count: int(&r["count"]),
                degree: int(&r["degree"]),
            })
            .collect(),
        stages: ints("stages"),
        tree_levels: ints("tree_levels"),
        zero_checks: small("zero_checks"),
        forests: small("forests"),
        lookups: list(&value["lookups"])
            .iter()
            .map(|l| LookupSizes {
                lookups: int(&l["lookups"]),
                rows: int(&l["rows"]),
            })
            .collect(),
        claim_groups: ints("claim_groups"),
        advice_rows: int(&value["advice_rows"]),
        matrices: list(&value["matrices"])
            .iter()
            .map(|m| MatrixSizes {
                matrix: matrix(text(&m["matrix"])),
                width: int(&m["width"]),
                rows: int(&m["rows"]),
                variables: int(&m["variables"]),
            })
            .collect(),
        groups: LeafGroupSizes {
            count: int(&value["groups"]["count"]),
            level_rounds: list(&value["groups"]["level_rounds"])
                .iter()
                .map(|r| Rounds {
                    count: int(&r["count"]),
                    degree: int(&r["degree"]),
                })
                .collect(),
            merge_rounds: int(&value["groups"]["merge_rounds"]),
            merge_claims: int(&value["groups"]["merge_claims"]),
        },
    }
}

/// The oracle's name of a row and of its commitment.
fn names(row: &SoundnessRow) -> (&'static str, Option<&'static str>) {
    let commitment = |c: Commitment| match c {
        Commitment::Advice => "advice",
        Commitment::Weights(m) => m.name(),
    };
    match row.kind {
        RowKind::Rounds => ("rounds", None),
        RowKind::Union => ("union", None),
        RowKind::Forests => ("forests", None),
        RowKind::LogUp => ("logup", None),
        RowKind::Openings => ("openings", None),
        RowKind::Queries(c) => ("queries", Some(commitment(c))),
        RowKind::Field(c) => ("field", Some(commitment(c))),
        RowKind::Linking => ("linking", None),
        RowKind::Hash => ("hash", None),
    }
}

fn optional(value: &Value) -> Option<u64> {
    value.as_u64()
}

/// The row is the oracle's, field by field.
fn same_row(ours: &SoundnessRow, theirs: &Value, case: &str) {
    let (row, commitment) = names(ours);
    let what = format!("{case}: {row} {commitment:?}");
    assert_eq!(row, text(&theirs["row"]), "{what}");
    assert_eq!(commitment, theirs["commitment"].as_str(), "{what}");
    match ours.bound {
        ErrorBound::Field { numerator } => {
            assert_eq!(numerator.to_string(), text(&theirs["numerator"]), "{what}");
        }
        ErrorBound::Queries {
            codeword,
            message,
            mask,
            queries,
        } => {
            assert_eq!(codeword, int::<u64>(&theirs["codeword"]), "{what}");
            assert_eq!(message, int::<u64>(&theirs["message"]), "{what}");
            assert_eq!(mask, int::<u64>(&theirs["mask"]), "{what}");
            assert_eq!(queries, int::<u32>(&theirs["queries"]), "{what}");
        }
        ErrorBound::Hash => assert!(theirs.get("numerator").is_none(), "{what}"),
    }
    assert_eq!(
        ours.theta,
        theirs["theta"].as_u64().map(|t| t as u32),
        "{what}"
    );
    assert_eq!(
        ours.per_query.map(|b| b.0),
        optional(&theirs["per_query_bits"]),
        "{what}"
    );
    assert_eq!(ours.bits.map(|b| b.0), optional(&theirs["bits"]), "{what}");
    assert_eq!(ours.grinding, int::<u32>(&theirs["grinding"]), "{what}");
    let covered = ours.covered_by.map(|p| match p {
        ProofOfWork::Seed => "seed-pow",
        ProofOfWork::Query => "query-pow",
    });
    assert_eq!(covered, theirs["covered_by"].as_str(), "{what}");
    assert_eq!(
        ours.target,
        theirs["target"].as_u64().map(|t| t as u32),
        "{what}"
    );
    assert_eq!(ours.margin(), theirs["margin"].as_i64(), "{what}");
    assert_eq!(ours.pass, theirs["pass"].as_bool(), "{what}");
}

fn report(params: &ParamSet, sizes: &StatementSizes) -> SoundnessReport {
    SoundnessReport::new(params, sizes).expect("sizes the calculator bounds")
}

fn row(report: &SoundnessReport, kind: RowKind) -> &SoundnessRow {
    report
        .rows
        .iter()
        .find(|r| r.kind == kind)
        .unwrap_or_else(|| panic!("a row {kind:?}"))
}

/// The rows that miss their targets.
fn failures(report: &SoundnessReport) -> Vec<RowKind> {
    report
        .rows
        .iter()
        .filter(|r| r.pass == Some(false))
        .map(|r| r.kind)
        .collect()
}

fn minimum(report: &SoundnessReport) -> RowKind {
    report.rows[report.minimum().expect("a row with bits")].kind
}

#[test]
fn the_parameter_sets_are_the_oracles() {
    let vectors = load();
    let theirs: Vec<ParamSet> = list(&vectors["parameter_sets"])
        .iter()
        .map(params)
        .collect();
    let ours: Vec<ParamSet> = ParamSet::ALL.iter().map(|s| **s).collect();
    assert_eq!(ours, theirs);
    // The row block and the segments are the advice layout's (RFC-0014's table, RFC-0026).
    for set in ParamSet::ALL {
        assert_eq!(set.row_block(), set.layout.advice.row_block);
        assert_eq!(set.segments(), set.layout.advice.segments);
    }
}

#[test]
fn the_405b_statement_is_the_oracles() {
    let vectors = load();
    let case = list(&vectors["cases"])
        .iter()
        .find(|c| text(&c["name"]) == "405b/live-64/1")
        .expect("the 405B case");
    assert_eq!(
        StatementSizes::llama_405b_16k(),
        sizes(&case["report"]["statement"])
    );
}

/// The default's LogUp instances are the compiler's 405B report's at 16K tokens: each table's
/// lookups per token times 16,384, and its rows, but `Exp`'s lookups, every position of the
/// padded score tensor, `2^42` (RFC-0030 'Cost'). The report is the source RFC-0014 names; the
/// oracle states its tables.
#[test]
fn the_405b_lookups_are_the_compilers_reports() {
    let path =
        common::vectors().join("../../docs/perf/v0.2/commitment-report-405b-servable-c2-b32.json");
    let report: Value =
        serde_json::from_str(&fs::read_to_string(path).expect("the report")).expect("JSON");
    assert_eq!(report["model"]["n_layers"], 126);
    assert_eq!(report["model"]["profile"], "servable");
    let tables = list(&report["tables"]);
    let statement = StatementSizes::llama_405b_16k();
    assert_eq!(statement.lookups.len(), tables.len());
    for (instance, table) in statement.lookups.iter().zip(tables) {
        let per_token: u64 = int(&table["lookups_per_token"]);
        let expected = match text(&table["kind"]) {
            "Exp" => {
                assert_eq!(per_token, 0, "Exp is looked up by the scores alone");
                1 << 42
            }
            _ => {
                assert_eq!(int::<u64>(&table["score_lookups"]), 0);
                let per_row = if text(&table["kind"]) == "Range" {
                    match int::<u64>(&table["rows"]) {
                        131_072 => 1,
                        65_536 => 4,
                        4 => 2,
                        _ => 0,
                    }
                } else {
                    0
                };
                assert!(per_token >= 126 * 128 * per_row);
                // A Range family can serve both root/requant and Softmax keys.
                // Only its Softmax contribution extends to the padded layers.
                per_token * (1 << 14) + per_row * (128 - 126) * 128 * (1 << 14)
            }
        };
        assert_eq!(instance.lookups, expected, "{table}");
        assert_eq!(instance.rows, int::<u64>(&table["rows"]), "{table}");
    }
}

#[test]
fn every_report_is_the_oracles() {
    let vectors = load();
    let cases = list(&vectors["cases"]);
    assert!(cases.len() >= 70, "{} cases", cases.len());
    for case in cases {
        let name = text(&case["name"]);
        let theirs = &case["report"];
        let ours = report(&params(&theirs["params"]), &sizes(&theirs["statement"]));
        let rows = list(&theirs["rows"]);
        assert_eq!(ours.rows.len(), rows.len(), "{name}");
        for (ours, theirs) in ours.rows.iter().zip(rows) {
            same_row(ours, theirs, name);
        }
        assert_eq!(
            ours.minimum().map(|i| i as u64),
            theirs["minimum"].as_u64(),
            "{name}"
        );
        assert_eq!(
            ours.meets_targets(),
            theirs["meets_targets"].as_bool(),
            "{name}"
        );
    }
}

/// `10^30·2^-20 = 2^10·5^30`: the stated precision, in the units of the oracle's true values.
const SLACK_E30: u128 = 1024 * 931_322_574_615_478_515_625;
const E30: u128 = 1_000_000_000_000_000_000_000_000_000_000;

/// `r ≤ 256·λ < r + 1 + 2^-20`, from `t = ⌊256·λ·10^30⌋`: `r·10^30 ≤ t` puts `r` below, and
/// `t + 1 ≤ (r + 1)·10^30 + 2^10·5^30` keeps it within the precision.
fn within_precision(r: u64, truth: &Value, what: &str) {
    let t: u128 = text(truth).parse().expect("a decimal");
    let r = u128::from(r);
    assert!(r * E30 <= t, "{what}: {r} lies above the true {t}·10^-30");
    assert!(
        t < (r + 1) * E30 + SLACK_E30,
        "{what}: {r} lies more than 2^-8·(1 + 2^-20) bits below the true {t}·10^-30"
    );
}

#[test]
fn every_bound_lies_below_the_true_value_within_its_precision() {
    let vectors = load();
    let mut checked = 0;
    for case in list(&vectors["cases"]) {
        let name = text(&case["name"]);
        let theirs = &case["report"];
        let ours = report(&params(&theirs["params"]), &sizes(&theirs["statement"]));
        for (row, truth) in ours.rows.iter().zip(list(&case["true"])) {
            let what = format!("{name}: {:?}", row.kind);
            match row.bits {
                Some(bits) => within_precision(bits.0, &truth["bits"], &what),
                None => assert!(truth["bits"].is_null(), "{what}"),
            }
            if let Some(per_query) = row.per_query {
                within_precision(per_query.0, &truth["per_query"], &what);
            }
            checked += 1;
        }
    }
    assert!(checked > 1000, "{checked} rows");
}

#[test]
fn live_64_and_ni_100_meet_their_targets_at_the_405b_statement() {
    let statement = StatementSizes::llama_405b_16k();
    for set in [&ParamSet::LIVE_64, &ParamSet::NI_100] {
        let report = report(set, &statement);
        assert_eq!(report.meets_targets(), Some(true), "{}", set.id);
        assert!(
            report.rows.iter().all(|r| r.pass == Some(true)),
            "{}",
            set.id
        );
    }
    // Under tessera-protocol/1 the weights carry no mask, 65.76 bits for τ_W = 97 (RFC-0049), so
    // both sets' tightest row is the advice queries: 64.32 bits for live-64's 64, 80.09 + 20 for
    // ni-100's 100.
    let live = report(&ParamSet::LIVE_64, &statement);
    assert_eq!(minimum(&live), RowKind::Queries(Commitment::Advice));
    let ni = report(&ParamSet::NI_100, &statement);
    assert_eq!(minimum(&ni), RowKind::Queries(Commitment::Advice));
    // Under tessera-protocol/2, whose weights carry P_W mask coefficients, both sets still meet
    // their targets, and live-64's tightest row is a weight matrix's queries, 64.03 bits.
    let masked = StatementSizes {
        protocol: 2,
        groups: LeafGroupSizes::NONE,
        ..statement.clone()
    };
    for set in [&ParamSet::LIVE_64, &ParamSet::NI_100] {
        assert_eq!(
            report(set, &masked).meets_targets(),
            Some(true),
            "{}",
            set.id
        );
    }
    assert_eq!(
        minimum(&report(&ParamSet::LIVE_64, &masked)),
        RowKind::Queries(Commitment::Weights(WeightMatrix::Embed))
    );
    // test-small certifies nothing: it has no targets, so no row passes or fails.
    let small = report(&ParamSet::TEST_SMALL, &statement);
    assert_eq!(small.meets_targets(), None);
    assert!(
        small
            .rows
            .iter()
            .all(|r| r.pass.is_none() && r.target.is_none())
    );
}

/// RFC-0014 'Testing Strategy': a profile with `N + T = 2^50` makes the LogUp row fail its
/// target, at `ni-100`: `78 + 20 − ε` bits for 100. At `live-64` the same instance gives
/// `78 − ε` one-shot bits, which its 64-bit one-shot target accepts.
#[test]
fn an_instance_of_n_plus_t_2_50_fails_the_logup_row() {
    let mut statement = StatementSizes::llama_405b_16k();
    statement.lookups.push(LookupSizes {
        lookups: (1 << 50) - (1 << 22),
        rows: 1 << 22,
    });
    let ni = report(&ParamSet::NI_100, &statement);
    assert_eq!(failures(&ni), [RowKind::LogUp]);
    assert_eq!(minimum(&ni), RowKind::LogUp);
    let logup = row(&ni, RowKind::LogUp);
    assert_eq!(logup.bound, ErrorBound::Field { numerator: 1 << 50 });
    // 98 − 6.7·10^-10 bits: one unit below 98.
    assert_eq!(logup.bits, Some(FixedBits(98 * 256 - 1)));
    let live = report(&ParamSet::LIVE_64, &statement);
    assert_eq!(row(&live, RowKind::LogUp).pass, Some(true));
    assert_eq!(
        row(&live, RowKind::LogUp).bits,
        Some(FixedBits(78 * 256 - 1))
    );
}

/// The per-query bits of every commitment are `−log2((1 + ρ')/2)`, `ρ' = (k + P)/n`, within
/// the precision and never above, at both release sets; the oracle's 60-digit values, which do
/// not go through LOG2, are the true ones.
#[test]
fn the_per_query_bits_are_minus_log2_of_one_plus_rho_over_two() {
    let vectors = load();
    for name in ["405b/live-64/1", "405b/ni-100/1"] {
        let case = list(&vectors["cases"])
            .iter()
            .find(|c| text(&c["name"]) == name)
            .expect("the case");
        let ours = report(
            &params(&case["report"]["params"]),
            &StatementSizes::llama_405b_16k(),
        );
        let mut commitments = 0;
        for (row, truth) in ours.rows.iter().zip(list(&case["true"])) {
            if let RowKind::Queries(_) = row.kind {
                let per_query = row.per_query.expect("a queries row's per-query bits");
                within_precision(per_query.0, &truth["per_query"], name);
                commitments += 1;
            }
        }
        assert_eq!(commitments, 11, "the advice and ten weight matrices");
    }
}

/// RFC-0014 'Parameter sets' prints `64.3` bits for `τ_A = 155`, `80.1` for 193, `64.0` for
/// `τ_W = 97` and `80.5` for 122. The exact values are 64.328, 80.098, 64.035 and 80.538: three
/// figures are lower bounds, and `80.1` is 80.098 rounded to nearest, 0.002 above the true
/// value. No figure is off by more than its rounding, half its last digit. The weights' figures
/// count their `P_W` mask coefficients, which `tessera-protocol/2` applies: they are checked on
/// the 405B statement under it (`405b-protocol-2/…`), where `tessera-protocol/1` has none
/// (RFC-0049).
///
/// The verdicts are exact. With `t = ⌊256·λ·10^30⌋`, the oracle's, and a figure of `f` tenths,
/// `λ ≥ f/10` exactly when `t ≥ 256·f·10^29`, an integer; the calculator's FixedBits is checked
/// against the same figure, since a report shows it.
#[test]
fn rfc_0014s_figures_are_lower_bounds_but_80_1_which_is_rounded() {
    let vectors = load();
    let figures: [(&str, Commitment, u32, bool); 4] = [
        ("405b/live-64/1", Commitment::Advice, 643, true),
        ("405b/ni-100/1", Commitment::Advice, 801, false),
        (
            "405b-protocol-2/live-64/1",
            Commitment::Weights(WeightMatrix::Down),
            640,
            true,
        ),
        (
            "405b-protocol-2/ni-100/1",
            Commitment::Weights(WeightMatrix::Down),
            805,
            true,
        ),
    ];
    for (name, commitment, tenths, lower_bound) in figures {
        let case = list(&vectors["cases"])
            .iter()
            .find(|c| text(&c["name"]) == name)
            .expect("the case");
        let set = params(&case["report"]["params"]);
        let statement = StatementSizes {
            protocol: int(&case["report"]["statement"]["protocol"]),
            groups: sizes(&case["report"]["statement"]).groups,
            ..StatementSizes::llama_405b_16k()
        };
        let r = report(&set, &statement);
        let at = r
            .rows
            .iter()
            .position(|row| row.kind == RowKind::Queries(commitment))
            .expect("the queries row");
        // The queries' own bits, without the query-pow's 20 at ni-100: τ·b(ρ').
        let grinding = u128::from(set.query_pow) * 256 * E30;
        let truth: u128 = text(&case["true"][at]["bits"]).parse().expect("a decimal");
        let truth = truth - grinding;
        let figure = u128::from(tenths) * 256 * (E30 / 10);
        assert_eq!(truth >= figure, lower_bound, "{name} {commitment:?}");
        // Within rounding: above the figure less 0.05 bits.
        assert!(truth > figure - 128 * (E30 / 10), "{name} {commitment:?}");
        // The FixedBits, a lower bound, shows a figure only when the true value reaches it.
        let bits = u128::from(r.rows[at].bits.expect("bits").0) - u128::from(set.query_pow) * 256;
        assert!(
            bits * 10 < u128::from(tenths) * 256 || lower_bound,
            "{name}"
        );
    }
}

/// For every row of RFC-0014's table, a size or parameter that makes exactly that row miss its
/// target at both release sets, which makes it the minimum. The rounds row cannot fail alone:
/// the union bound counts every term it counts, so it fails too, and is the minimum unless the
/// two tie, when the rounds row, the first, is.
#[test]
fn each_row_misses_its_target_alone() {
    let base = StatementSizes::llama_405b_16k();
    let down = |width: Option<u64>, variables: Option<u32>| {
        let mut s = base.clone();
        let m = s
            .matrices
            .iter_mut()
            .find(|m| m.matrix == WeightMatrix::Down)
            .expect("Down");
        m.width = width.unwrap_or(m.width);
        m.variables = variables.unwrap_or(m.variables);
        s
    };
    let changed = |change: &dyn Fn(&mut StatementSizes)| {
        let mut s = base.clone();
        change(&mut s);
        s
    };
    let down_queries = RowKind::Queries(Commitment::Weights(WeightMatrix::Down));
    let down_field = RowKind::Field(Commitment::Weights(WeightMatrix::Down));
    for set in [ParamSet::LIVE_64, ParamSet::NI_100] {
        // An instance of N + T = 2^50 at ni-100; at live-64's one-shot target of 64 bits it takes
        // N + T = 2^64.
        let logup = match set.mode {
            ParamMode::NonInteractive => LookupSizes {
                lookups: (1 << 50) - (1 << 22),
                rows: 1 << 22,
            },
            _ => LookupSizes {
                lookups: 1 << 63,
                rows: 1 << 63,
            },
        };
        let fewer_queries = ParamSet {
            advice_queries: set.advice_queries - 1,
            ..set
        };
        let cases: Vec<(RowKind, ParamSet, StatementSizes)> = vec![
            (
                RowKind::Rounds,
                set,
                changed(&|s| {
                    s.rounds.push(Rounds {
                        count: 1,
                        degree: 1 << 29,
                    })
                }),
            ),
            (
                RowKind::Union,
                set,
                changed(&|s| {
                    s.rounds.push(Rounds {
                        count: 1 << 28,
                        degree: 1,
                    })
                }),
            ),
            (RowKind::Forests, set, changed(&|s| s.forests.push(1 << 29))),
            (RowKind::LogUp, set, changed(&|s| s.lookups.push(logup))),
            (
                RowKind::Openings,
                set,
                changed(&|s| s.claim_groups.push(1 << 29)),
            ),
            (
                RowKind::Queries(Commitment::Advice),
                fewer_queries,
                base.clone(),
            ),
            // Under tessera-protocol/1 the weights carry no mask, and their rate is 1/4 at every
            // width: a narrower Down misses its queries' target only with its P_W mask
            // coefficients, under tessera-protocol/2 (RFC-0049).
            (
                down_queries,
                set,
                StatementSizes {
                    protocol: 2,
                    groups: LeafGroupSizes::NONE,
                    ..down(Some(1 << 19), None)
                },
            ),
            (
                RowKind::Field(Commitment::Advice),
                set,
                changed(&|s| s.advice_rows = 1 << 33),
            ),
            (down_field, set, down(Some(1 << 24), None)),
            (RowKind::Linking, set, down(None, Some(1 << 29))),
        ];
        for (kind, params, statement) in cases {
            let r = report(&params, &statement);
            let what = format!("{}: {kind:?}", set.id);
            let mut expected = vec![kind];
            if kind == RowKind::Rounds {
                expected.push(RowKind::Union);
            }
            assert_eq!(failures(&r), expected, "{what}");
            assert_eq!(r.meets_targets(), Some(false), "{what}");
            let least = minimum(&r);
            if kind == RowKind::Rounds {
                let union = row(&r, RowKind::Union).bits;
                assert!(union <= row(&r, RowKind::Rounds).bits, "{what}");
                assert!(
                    least == RowKind::Rounds || least == RowKind::Union,
                    "{what}: {least:?}"
                );
            } else {
                assert_eq!(least, kind, "{what}");
            }
        }
    }
}

/// The decisions are exact where rounded bits would err: `|Fp2| = p^2 < 2^128`, so a numerator
/// of `2^28` gives `100 − 6.7·10^-10` bits and fails a 100-bit target, while `⌊q/2^100⌋ =
/// 2^28 − 1` passes it with `4.7·10^-9` to spare. A 64-bit FixedBits shows both as 100.00 or
/// 99.99; the decision compares `c·2^100` with `q`.
#[test]
fn the_decisions_are_exact_at_the_boundary() {
    let mut statement = StatementSizes::llama_405b_16k();
    statement.claim_groups = vec![(1 << 28) - 1];
    let passes = report(&ParamSet::NI_100, &statement);
    let openings = row(&passes, RowKind::Openings);
    assert_eq!(openings.pass, Some(true));
    assert_eq!(openings.bits, Some(FixedBits(100 * 256)));
    statement.claim_groups = vec![1 << 28];
    let fails = report(&ParamSet::NI_100, &statement);
    let openings = row(&fails, RowKind::Openings);
    assert_eq!(openings.pass, Some(false));
    assert_eq!(openings.bits, Some(FixedBits(100 * 256 - 1)));
    assert_eq!(openings.margin(), Some(-1));
}

/// Each row-index bit that a profile's advice rows add grows the advice's field term by
/// `n/|Fp2|` (RFC-0014 'Testing Strategy'): `ϑ = ⌈log2 rows⌉`, and the numerator is `ϑ·n`.
#[test]
fn each_advice_row_index_bit_adds_n_to_the_field_term() {
    let mut statement = StatementSizes::llama_405b_16k();
    let n = 1u128 << 23;
    for (rows, theta) in [
        (1, 0),
        (2, 1),
        (3, 2),
        (1 << 20, 20),
        ((1 << 20) + 1, 21),
        (1 << 21, 21),
        (u64::MAX, 64),
    ] {
        statement.advice_rows = rows;
        let r = report(&ParamSet::LIVE_64, &statement);
        let field = row(&r, RowKind::Field(Commitment::Advice));
        assert_eq!(field.theta, Some(theta), "{rows} rows");
        assert_eq!(
            field.bound,
            ErrorBound::Field {
                numerator: u128::from(theta) * n
            },
            "{rows} rows"
        );
        // One row has no row index: the field term vanishes.
        assert_eq!(field.bits.is_none(), theta == 0, "{rows} rows");
    }
}

/// Sizes the calculator cannot bound are refused with `ERR-USAGE`, each with its rule.
#[test]
fn sizes_it_cannot_bound_are_refused() {
    let base = StatementSizes::llama_405b_16k();
    let set = ParamSet::LIVE_64;
    let refuse = |params: &ParamSet, sizes: &StatementSizes| {
        let error = SoundnessReport::new(params, sizes).expect_err("refused");
        assert!(matches!(error, ProtocolError::Sizes { .. }), "{error:?}");
        assert_eq!(error.code(), ErrorCode::Usage);
    };
    for protocol in [0, 3] {
        refuse(
            &set,
            &StatementSizes {
                protocol,
                ..base.clone()
            },
        );
    }
    for (advice, weights) in [(0, 97), (155, 0), ((1 << 16) + 1, 97), (155, u32::MAX)] {
        let params = ParamSet {
            advice_queries: advice,
            weight_queries: weights,
            ..set
        };
        refuse(&params, &base);
    }
    let mut long = set;
    long.layout.advice.log2_k = 32;
    refuse(&long, &base);
    refuse(
        &set,
        &StatementSizes {
            advice_rows: 0,
            ..base.clone()
        },
    );
    let mut swapped = base.clone();
    swapped.matrices.swap(0, 1);
    refuse(&set, &swapped);
    let mut twice = base.clone();
    twice.matrices.push(twice.matrices[9]);
    refuse(&set, &twice);
    for width in [0, (1 << 30) + 1] {
        let mut wide = base.clone();
        wide.matrices[0].width = width;
        refuse(&set, &wide);
    }
    // RFC-0054 'Soundness': G is a power of two up to 2^14, 1 in tessera-protocol/2, and no
    // grouping has no grouped sizes.
    for count in [0, 3, 1 << 15] {
        let mut groups = base.clone();
        groups.groups.count = count;
        refuse(&set, &groups);
    }
    refuse(
        &set,
        &StatementSizes {
            protocol: 2,
            ..base.clone()
        },
    );
    let none = |change: fn(&mut LeafGroupSizes)| {
        let mut sizes = base.clone();
        sizes.groups = LeafGroupSizes::NONE;
        change(&mut sizes.groups);
        sizes
    };
    assert!(SoundnessReport::new(&set, &none(|_| {})).is_ok());
    refuse(&set, &none(|g| g.merge_claims = 1));
    refuse(&set, &none(|g| g.merge_rounds = 1));
    refuse(
        &set,
        &none(|g| {
            g.level_rounds.push(Rounds {
                count: 0,
                degree: 0,
            })
        }),
    );
    let mut empty = StatementSizes {
        protocol: 2,
        groups: LeafGroupSizes::NONE,
        ..base.clone()
    };
    empty.matrices[0].rows = 0;
    let no_blinding = ParamSet {
        layout: CommitmentLayout {
            weights: WeightLayoutParams {
                blinding_rows: 0,
                ..set.layout.weights
            },
            ..set.layout
        },
        ..set
    };
    refuse(&no_blinding, &empty);
    // Under tessera-protocol/1 the same matrix is bounded: it has no proximity rows (RFC-0049).
    let unopened = StatementSizes {
        protocol: 1,
        ..empty.clone()
    };
    assert!(SoundnessReport::new(&no_blinding, &unopened).is_ok());
    // The edges themselves are bounded: 2^16 queries and a codeword of 2^32 symbols.
    let mut widest = base.clone();
    widest.matrices[0].width = 1 << 30;
    let most = ParamSet {
        advice_queries: 1 << 16,
        weight_queries: 1 << 16,
        ..set
    };
    assert!(SoundnessReport::new(&most, &widest).is_ok());
}

/// RFC-0049 'The calculator': `tessera-protocol/1` commits no blinding row and no mask for the
/// weights, so a matrix's field term counts `ϑ_T = ⌈log2 m_T⌉` and its queries the unmasked rate
/// `k_T/n_T`, `P = 0`; a matrix with no committed row has neither row. `tessera-protocol/2` keeps
/// RFC-0014's counts, `⌈log2(m_T + B_W)⌉` and `P_W`. Every set, both protocols, every 405B
/// matrix, and the empty `ChannelParams` of a model whose weights have no channel parameters.
#[test]
fn protocol_1_counts_a_matrix_by_its_committed_rows_with_no_mask() {
    let base = StatementSizes::llama_405b_16k();
    assert_eq!(
        base.protocol, 1,
        "the 405B statement is tessera-protocol/1's"
    );
    for set in [ParamSet::LIVE_64, ParamSet::NI_100, ParamSet::TEST_SMALL] {
        let weights = set.layout.weights;
        for protocol in [1, 2] {
            // RFC-0054 'Messages': tessera-protocol/2 accepts no grouping.
            let statement = StatementSizes {
                protocol,
                groups: match protocol {
                    1 => base.groups.clone(),
                    _ => LeafGroupSizes::NONE,
                },
                ..base.clone()
            };
            let r = report(&set, &statement);
            for m in &statement.matrices {
                let what = format!("{}, protocol {protocol}: {:?}", set.id, m.matrix);
                let (rows, mask) = match protocol {
                    1 => (m.rows, 0),
                    _ => (
                        m.rows + u64::from(weights.blinding_rows),
                        u64::from(weights.mask_coeffs),
                    ),
                };
                // ⌈log2 rows⌉, from its definition.
                let theta = (0..64).find(|&t| rows <= 1 << t).expect("rows ≤ 2^63");
                let n = m.width << weights.log2_inv_rate;
                let commitment = Commitment::Weights(m.matrix);
                let field = row(&r, RowKind::Field(commitment));
                assert_eq!(field.theta, Some(theta), "{what}");
                assert_eq!(
                    field.bound,
                    ErrorBound::Field {
                        numerator: u128::from(theta) * u128::from(n)
                    },
                    "{what}"
                );
                assert_eq!(
                    row(&r, RowKind::Queries(commitment)).bound,
                    ErrorBound::Queries {
                        codeword: n,
                        message: m.width,
                        mask,
                        queries: set.weight_queries,
                    },
                    "{what}"
                );
            }
        }
    }
    // No ChannelParams: its matrix commits no row. Under tessera-protocol/1 it has no queries row
    // and no field term, and is not refused; under tessera-protocol/2 its B_W blinding rows count.
    let mut empty = base.clone();
    let channels = empty
        .matrices
        .iter_mut()
        .find(|m| m.matrix == WeightMatrix::ChannelParams)
        .expect("ChannelParams");
    channels.rows = 0;
    let channels = Commitment::Weights(WeightMatrix::ChannelParams);
    for set in [ParamSet::LIVE_64, ParamSet::NI_100] {
        let r = report(&set, &empty);
        assert!(
            r.rows.iter().all(|row| !matches!(
                row.kind,
                RowKind::Queries(c) | RowKind::Field(c) if c == channels
            )),
            "{}",
            set.id
        );
        assert_eq!(
            r.rows.len(),
            report(&set, &base).rows.len() - 2,
            "{}",
            set.id
        );
        assert_eq!(r.meets_targets(), Some(true), "{}", set.id);
        let masked = StatementSizes {
            protocol: 2,
            groups: LeafGroupSizes::NONE,
            ..empty.clone()
        };
        assert_eq!(
            row(&report(&set, &masked), RowKind::Field(channels)).theta,
            Some(12),
            "{}: ⌈log2 B_W⌉",
            set.id
        );
    }
}

fn id(text: &str) -> ParamSetId {
    ParamSetId(Ascii::new(text).expect("ASCII"))
}

/// The lookup by id (INV-PROTO-5): a release build accepts `live-64/1` and `ni-100/1` only;
/// `test-small/1` needs the `test-params` feature, which no release build has. Every set is
/// defined, so a report can describe `test-small/1`, but only accepting it lets a proof use it.
#[test]
fn a_parameter_set_is_accepted_by_its_id() {
    assert_eq!(ParamSet::accept(&id("live-64/1")), Ok(&ParamSet::LIVE_64));
    assert_eq!(ParamSet::accept(&id("ni-100/1")), Ok(&ParamSet::NI_100));
    for unknown in [
        "live-64",
        "live-64/2",
        "ni-100/0",
        "LIVE-64/1",
        "",
        "test-small",
    ] {
        assert_eq!(
            ParamSet::accept(&id(unknown)),
            Err(ProtocolError::UnknownParams),
            "{unknown:?}"
        );
        assert_eq!(ParamSet::defined(&id(unknown)), None, "{unknown:?}");
    }
    let small = ParamSet::accept(&id("test-small/1"));
    if cfg!(feature = "test-params") {
        assert_eq!(small, Ok(&ParamSet::TEST_SMALL));
    } else {
        assert_eq!(small, Err(ProtocolError::TestParams));
        assert_eq!(ProtocolError::TestParams.code(), ErrorCode::Usage);
    }
    // RFC-0014's modes: live-64 runs live only, ni-100 non-interactively only, test-small either.
    for (set, live, ni) in [
        (ParamSet::LIVE_64, true, false),
        (ParamSet::NI_100, false, true),
        (ParamSet::TEST_SMALL, true, true),
    ] {
        assert_eq!(set.mode.allows(Mode::Live), live, "{}", set.id);
        assert_eq!(set.mode.allows(Mode::NonInteractive), ni, "{}", set.id);
    }
    for set in ParamSet::ALL {
        assert_eq!(ParamSet::defined(&set.param_set_id()), Some(set));
        assert_eq!(set.param_set_id().0.as_str(), set.id);
        // A release verifier accepts exactly the sets that have targets to meet.
        assert_eq!(set.release, set.targets.is_some(), "{}", set.id);
    }
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        if n <= 1 {
            return 0;
        }
        let bits = 64 - (n - 1).leading_zeros();
        let shift = 64 - bits;
        loop {
            let x = self.next() >> shift;
            if x < n {
                return x;
            }
        }
    }

    fn bits(&mut self, most: u32) -> u64 {
        let width = self.below(u64::from(most) + 1) as u32;
        if width == 0 {
            0
        } else if width == 64 {
            self.next()
        } else {
            self.below(1u64 << width)
        }
    }
}

fn to_tv_params(p: &ParamSet) -> tv_core::soundness::SoundnessParams {
    tv_core::soundness::SoundnessParams {
        id: String::from(p.id),
        mode: match p.mode {
            ParamMode::Live => tv_core::soundness::SessionMode::Live,
            ParamMode::NonInteractive => tv_core::soundness::SessionMode::NonInteractive,
            ParamMode::Either => tv_core::soundness::SessionMode::Either,
        },
        release: p.release,
        advice: tv_core::claims::AdviceLayoutParams {
            log2_k: p.layout.advice.log2_k,
            log2_inv_rate: p.layout.advice.log2_inv_rate,
            mask_coeffs: p.layout.advice.mask_coeffs,
            row_block: p.layout.advice.row_block,
            segments: p.layout.advice.segments,
        },
        weights: tv_core::claims::WeightLayoutParams {
            log2_k: p.layout.weights.log2_k,
            log2_inv_rate: p.layout.weights.log2_inv_rate,
            mask_coeffs: p.layout.weights.mask_coeffs,
            blinding_rows: p.layout.weights.blinding_rows,
        },
        advice_queries: p.advice_queries,
        weight_queries: p.weight_queries,
        seed_pow: p.seed_pow,
        query_pow: p.query_pow,
        targets: p.targets.map(|t| tv_core::soundness::Targets {
            bits: t.bits,
            one_shot: t.one_shot,
        }),
    }
}

fn to_tv_statement(s: &StatementSizes) -> tv_core::soundness::StatementSizes {
    tv_core::soundness::StatementSizes {
        protocol: u8::try_from(s.protocol).expect("protocol"),
        rounds: s
            .rounds
            .iter()
            .map(|r| tv_core::soundness::RoundGroup {
                count: r.count,
                degree: u64::from(r.degree),
            })
            .collect(),
        stages: s.stages.clone(),
        tree_levels: s.tree_levels.clone(),
        zero_checks: s.zero_checks.iter().copied().map(u64::from).collect(),
        forests: s.forests.iter().copied().map(u64::from).collect(),
        lookups: s
            .lookups
            .iter()
            .map(|l| tv_core::soundness::LookupSize {
                lookups: l.lookups,
                rows: l.rows,
            })
            .collect(),
        claim_groups: s.claim_groups.clone(),
        advice_rows: s.advice_rows,
        groups: tv_core::soundness::GroupSizes {
            count: s.groups.count,
            level_rounds: s
                .groups
                .level_rounds
                .iter()
                .map(|r| tv_core::soundness::RoundGroup {
                    count: r.count,
                    degree: u64::from(r.degree),
                })
                .collect(),
            merge_rounds: u64::from(s.groups.merge_rounds),
            merge_claims: s.groups.merge_claims,
        },
        matrices: s
            .matrices
            .iter()
            .map(|m| tv_core::soundness::MatrixSize {
                matrix: tv_core::weights::Matrix::by_name(m.matrix.name()).expect("matrix"),
                width: m.width,
                rows: m.rows,
                variables: u64::from(m.variables),
            })
            .collect(),
    }
}

/// RFC-0014 'Soundness calculator', DEP-2: both workspaces state the same default 405B
/// statement, its grouped sizes included (RFC-0054 'Soundness'), each derived on its own.
#[test]
fn both_calculators_state_the_same_405b_default() {
    assert_eq!(
        to_tv_statement(&StatementSizes::llama_405b_16k()),
        tv_core::soundness::StatementSizes::default_405b()
    );
}

/// #110, RFC-0014 'Testing Strategy': both calculators agree on every row for 1,000 random
/// statement and parameter profiles.
#[test]
fn prover_and_verifier_calculators_agree_on_1000_random_profiles() {
    let mut rng = SplitMix64(0x1100_0014_2026_1020);
    for i in 0..1000 {
        let p = if rng.below(4) == 0 {
            *ParamSet::ALL[rng.below(3) as usize]
        } else {
            let log2_k = rng.below(25) as u8;
            let inv_rate = rng.below(u64::from((33 - log2_k).min(5))) as u8;
            let w_log2_k = rng.below(23) as u8;
            let w_inv_rate = rng.below(5) as u8;
            let targets = if rng.below(5) == 0 {
                None
            } else {
                Some(Targets {
                    bits: 60 + rng.below(70) as u32,
                    one_shot: 40 + rng.below(70) as u32,
                })
            };
            ParamSet {
                id: "random",
                mode: match rng.below(3) {
                    0 => ParamMode::Live,
                    1 => ParamMode::NonInteractive,
                    _ => ParamMode::Either,
                },
                release: rng.below(2) == 1,
                layout: CommitmentLayout {
                    advice: AdviceLayoutParams {
                        log2_k,
                        log2_inv_rate: inv_rate,
                        mask_coeffs: rng.bits(18) as u32,
                        row_block: 1 + rng.below(64) as u32,
                        segments: 1 + rng.below(8) as u32,
                    },
                    weights: WeightLayoutParams {
                        log2_k: w_log2_k,
                        log2_inv_rate: w_inv_rate,
                        mask_coeffs: rng.bits(18) as u32,
                        blinding_rows: rng.bits(13) as u32,
                    },
                },
                advice_queries: 1 + rng.below(300) as u32,
                weight_queries: 1 + rng.below(300) as u32,
                seed_pow: rng.below(33) as u32,
                query_pow: rng.below(33) as u32,
                targets,
            }
        };

        let protocol = 1 + rng.below(2) as u8;
        let w_inv = u32::from(p.layout.weights.log2_inv_rate);
        let blinding = u64::from(p.layout.weights.blinding_rows);
        let mut matrix_indices: Vec<usize> =
            (0..rng.below(6)).map(|_| rng.below(10) as usize).collect();
        matrix_indices.sort_unstable();
        matrix_indices.dedup();
        let matrices = matrix_indices
            .into_iter()
            .map(|idx| {
                let width = if rng.below(2) == 1 {
                    1u64 << rng.below(u64::from(33 - w_inv))
                } else {
                    rng.bits(32 - w_inv).max(1)
                };
                let mut rows = if rng.below(8) == 0 { 0 } else { rng.bits(63) };
                if protocol == 2 && rows.saturating_add(blinding) == 0 {
                    rows = 1;
                }
                MatrixSizes {
                    matrix: WeightMatrix::ALL[idx],
                    width,
                    rows,
                    variables: rng.bits(32) as u32,
                }
            })
            .collect();

        let statement = StatementSizes {
            protocol: u16::from(protocol),
            rounds: (0..rng.below(5))
                .map(|_| Rounds {
                    count: rng.bits(40),
                    degree: rng.bits(12) as u32,
                })
                .collect(),
            stages: (0..rng.below(5)).map(|_| rng.bits(20)).collect(),
            tree_levels: (0..rng.below(7)).map(|_| rng.bits(16)).collect(),
            zero_checks: (0..rng.below(5)).map(|_| rng.bits(7) as u32).collect(),
            forests: (0..rng.below(4)).map(|_| rng.bits(8) as u32).collect(),
            lookups: (0..rng.below(5))
                .map(|_| LookupSizes {
                    lookups: rng.bits(64),
                    rows: rng.bits(64),
                })
                .collect(),
            claim_groups: (0..rng.below(5)).map(|_| rng.bits(32)).collect(),
            advice_rows: rng.bits(64).max(1),
            matrices,
            // RFC-0054 'Soundness': a random grouping of 2^0 to 2^14 groups, none in
            // tessera-protocol/2.
            groups: match 1u64 << rng.below(15) {
                count if count > 1 && protocol == 1 => LeafGroupSizes {
                    count,
                    level_rounds: (0..rng.below(3))
                        .map(|_| Rounds {
                            count: rng.bits(20),
                            degree: rng.bits(4) as u32,
                        })
                        .collect(),
                    merge_rounds: rng.bits(7) as u32,
                    merge_claims: rng.bits(24),
                },
                _ => LeafGroupSizes::NONE,
            },
        };

        let prover_report = SoundnessReport::new(&p, &statement).expect("prover report");
        let verifier_report = tv_core::soundness::SoundnessReport::evaluate(
            to_tv_params(&p),
            to_tv_statement(&statement),
        )
        .expect("verifier report");

        let verifier_json: Value =
            serde_json::from_str(&verifier_report.to_json()).expect("verifier JSON");
        let v_rows = list(&verifier_json["rows"]);
        assert_eq!(
            prover_report.rows.len(),
            v_rows.len(),
            "profile {i}: row count"
        );
        for (p_row, v_row) in prover_report.rows.iter().zip(v_rows) {
            same_row(p_row, v_row, &format!("random-1000/{i}"));
        }
        assert_eq!(
            prover_report.minimum(),
            verifier_report.minimum,
            "profile {i}: minimum"
        );
        assert_eq!(
            prover_report.meets_targets(),
            verifier_report.meets_targets,
            "profile {i}: meets_targets"
        );
    }
}
