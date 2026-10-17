// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The golden test of RFC-0020 'Executable cost model': with the source's counts and unit
//! costs, no regeneration, no streaming, no column answers and no row combination, the model
//! reproduces the source's chart (source §5) within ±0.05, and its calculators reproduce §8
//! within 0.1 percentage point from that table's `μ` column. The figures come from
//! `spec/vectors/cost-model/cost-model.json`, which is checked here against the source document
//! itself, line by line.

use std::fs;
use std::path::PathBuf;

use serde_json::Value;
use tessera_bench::{
    Counts, MemoryModel, SOURCE_CONTEXTS, Scenario, UnitCosts, detection_probability, mu,
    proof_time_s, sampling_rate,
};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn vectors() -> Value {
    let path = root().join("spec/vectors/cost-model/cost-model.json");
    let text = fs::read_to_string(&path).expect("the cost-model vectors");
    serde_json::from_str(&text).expect("JSON")
}

fn numbers(value: &Value) -> Vec<f64> {
    let array = value.as_array().expect("an array");
    array
        .iter()
        .map(|x| x.as_f64().expect("a number"))
        .collect()
}

fn number(value: &Value) -> f64 {
    value.as_f64().expect("a number")
}

/// `1K` → 1024, …, `128K` → 131072.
fn context(label: &str) -> u32 {
    let kilo: u32 = label
        .trim()
        .strip_suffix('K')
        .and_then(|k| k.parse().ok())
        .expect("a context such as 16K");
    kilo * 1024
}

/// The cells of the markdown table rows among `lines`, from a row whose first cell is a context.
fn rows<'a>(lines: &[&'a str]) -> Vec<Vec<&'a str>> {
    lines
        .iter()
        .filter_map(|line| line.strip_prefix("| "))
        .map(|row| row.trim_end_matches(" |").split(" | ").collect::<Vec<_>>())
        .filter(|cells| {
            cells[0].ends_with('K') && cells[0][..cells[0].len() - 1].parse::<u32>().is_ok()
        })
        .collect()
}

fn close(a: f64, b: f64, relative: f64) -> bool {
    (a - b).abs() <= relative * a.abs().max(b.abs())
}

/// The vectors copy the source: each input's text stands on its line, and the chart's series,
/// the chart-data table and the §8 table are the document's, figure for figure.
#[test]
fn the_vectors_copy_the_source_document() {
    let v = vectors();
    let source = v["source"].as_str().expect("the source's path");
    let document = fs::read_to_string(root().join(source)).expect("the source document");
    let lines: Vec<&str> = document.lines().collect();
    let range = |name: &str| {
        let [first, last] = [0, 1].map(|i| v[name]["lines"][i].as_u64().expect("a line") as usize);
        &lines[first - 1..last]
    };

    for (name, input) in v["inputs"].as_object().expect("the inputs") {
        let line = input["line"].as_u64().expect("a line") as usize;
        let text = input["text"].as_str().expect("a quote");
        assert!(
            lines[line - 1].contains(text),
            "{name}: line {line} says {text:?}"
        );
    }

    let chart = range("chart");
    let axis = chart
        .iter()
        .find_map(|line| line.trim_start().strip_prefix("x-axis"))
        .expect("the chart's axis");
    let labels = axis
        .split('[')
        .nth(1)
        .expect("its labels")
        .trim_end_matches(']');
    let contexts: Vec<u32> = labels
        .split(", ")
        .map(|l| context(l.trim_matches('"')))
        .collect();
    let expected_contexts: Vec<u32> = v["chart"]["contexts"]
        .as_array()
        .expect("contexts")
        .iter()
        .map(|c| c.as_u64().expect("a context") as u32)
        .collect();
    assert_eq!(contexts, expected_contexts);
    let series: Vec<Vec<f64>> = chart
        .iter()
        .filter_map(|line| line.trim_start().strip_prefix("line ["))
        .map(|values| {
            let values = values.trim_end_matches(']').split(", ");
            values.map(|x| x.parse().expect("a figure")).collect()
        })
        .collect();
    // The caption's order: total, attention, linear layers.
    assert_eq!(
        series,
        ["total", "attention", "linear"].map(|key| numbers(&v["chart"][key]))
    );
    let table = rows(chart);
    assert_eq!(table.len(), 8, "the chart-data table");
    for (k, cells) in table.iter().enumerate() {
        let figures: Vec<f64> = cells[1..]
            .iter()
            .map(|x| x.parse().expect("a figure"))
            .collect();
        assert_eq!(context(cells[0]), contexts[k]);
        assert_eq!(
            figures[0], series[2][k],
            "the table's linear column is the chart's"
        );
        assert_eq!(
            figures[1], series[1][k],
            "the table's attention column is the chart's"
        );
        assert_eq!(figures[2], numbers(&v["chart"]["weight_pass"])[k]);
        assert_eq!(figures[3], numbers(&v["chart"]["total_one_decimal"])[k]);
    }

    let section_8 = rows(range("section_8"));
    let expected = v["section_8"]["rows"].as_array().expect("the §8 rows");
    assert_eq!(section_8.len(), expected.len());
    for (cells, row) in section_8.iter().zip(expected) {
        assert_eq!(
            u64::from(context(cells[0])),
            row["context"].as_u64().expect("a context")
        );
        assert_eq!(cells[1].parse::<f64>().expect("μ"), number(&row["mu"]));
        let seconds = match cells[2].split_once(' ').expect("a time and its unit") {
            (t, "s") => t.parse::<f64>().expect("seconds"),
            (t, "min") => 60.0 * t.parse::<f64>().expect("minutes"),
            (_, unit) => panic!("unit {unit}"),
        };
        assert_eq!(seconds, number(&row["proof_time_s"]));
        let percent = |cell: &str| {
            cell.trim_start_matches('≈')
                .trim_end_matches('%')
                .parse::<f64>()
        };
        assert_eq!(
            percent(cells[3]).expect("a rate"),
            number(&row["sampling_percent"])
        );
        assert_eq!(
            percent(cells[4]).expect("a probability"),
            number(&row["detection_percent"])
        );
        assert_eq!(
            cells[4].starts_with('≈'),
            row["detection_approximate"].as_bool().expect("a flag")
        );
    }
}

/// The crate's source defaults are the vectors' inputs, exactly.
#[test]
fn the_source_defaults_are_the_vectors_inputs() {
    let v = vectors();
    let input = |name: &str| number(&v["inputs"][name]["value"]);
    let u = UnitCosts::SOURCE;
    let c = Counts::SOURCE_405B;
    let s = Scenario::SOURCE;
    assert_eq!(
        [
            u.commit_ps,
            u.lookup_ps,
            u.product_leaf_ps,
            u.glue_fraction,
            u.weight_pass_gpu_s,
            u.native_ops_per_s
        ],
        [
            "commit_ps",
            "lookup_ps",
            "product_leaf_ps",
            "glue_fraction",
            "weight_pass_gpu_s",
            "native_ops_per_s"
        ]
        .map(input)
    );
    assert_eq!(
        u.streaming_pass_ps, 0.0,
        "the source prices no streamed pass"
    );
    assert_eq!(
        u.column_answer_ps, 0.0,
        "the source prices no column answers"
    );
    assert_eq!(
        u.row_combination_ps, 0.0,
        "the source prices no row combination"
    );
    assert_eq!(
        [
            c.commits_per_token,
            c.lookups_per_token,
            c.scores_per_token_per_ctx,
            c.native_ops_lin,
            c.native_ops_att_per_ctx
        ],
        [
            "commits_per_token",
            "lookups_per_token",
            "scores_per_token_per_ctx",
            "native_ops_lin",
            "native_ops_att_per_ctx"
        ]
        .map(input)
    );
    assert_eq!(
        c.regen_full_equivalents, 0.0,
        "the source counts no regeneration"
    );
    assert_eq!(f64::from(s.batch_tokens), input("batch_tokens"));
    let scenario = &v["section_8"];
    assert_eq!(u64::from(s.gpus), scenario["gpus"].as_u64().expect("G"));
    assert_eq!(s.eta, number(&scenario["eta"]));
    assert_eq!(s.target_overhead, number(&scenario["target_overhead"]));
    assert_eq!(
        u64::from(s.violations),
        scenario["violations"].as_u64().expect("v")
    );
    let contexts: Vec<u64> = v["chart"]["contexts"]
        .as_array()
        .expect("contexts")
        .iter()
        .map(|c| c.as_u64().expect("a context"))
        .collect();
    assert_eq!(contexts, SOURCE_CONTEXTS.map(u64::from));
}

/// `mu` is the oracle's model to 1e-12, term by term: the same formula in the same units.
#[test]
fn the_model_is_the_oracles() {
    let v = vectors();
    for row in v["model"].as_array().expect("the oracle's model") {
        let context = row["context"].as_u64().expect("a context") as u32;
        let m = mu(
            &Counts::SOURCE_405B,
            &UnitCosts::SOURCE,
            &MemoryModel::NODE_405B,
            context,
            Scenario::SOURCE.batch_tokens,
        )
        .expect("the source's model");
        for (term, ours) in [
            ("linear", m.linear),
            ("attention", m.attention),
            ("weights", m.weights),
            ("total", m.total),
        ] {
            let oracle = number(&row[term]);
            assert!(
                close(ours, oracle, 1e-12),
                "{term} at {context}: {ours} against {oracle}"
            );
        }
        assert_eq!(
            (m.regen, m.streaming, m.column_answers, m.row_combination),
            (0.0, 0.0, 0.0, 0.0)
        );
    }
}

/// The golden criterion: the model within ±0.05 of the chart's linear, attention and total
/// columns at all eight contexts, and within ±0.0005 of the weight pass, which the chart-data
/// table gives to three decimals.
#[test]
fn the_model_reproduces_the_source_chart() {
    let v = vectors();
    let chart = &v["chart"];
    let tolerance = number(&v["tolerances"]["chart"]);
    let weight_tolerance = number(&v["tolerances"]["weight_pass"]);
    for (k, &context) in SOURCE_CONTEXTS.iter().enumerate() {
        let m = mu(
            &Counts::SOURCE_405B,
            &UnitCosts::SOURCE,
            &MemoryModel::NODE_405B,
            context,
            Scenario::SOURCE.batch_tokens,
        )
        .expect("the source's model");
        for (term, ours, tol) in [
            ("linear", m.linear, tolerance),
            ("attention", m.attention, tolerance),
            ("total", m.total, tolerance),
            ("weight_pass", m.weights, weight_tolerance),
        ] {
            let source = numbers(&chart[term])[k];
            assert!(
                (ours - source).abs() <= tol,
                "{term} at {context}: {ours:.4} against the source's {source}"
            );
        }
    }
}

/// The calculators, from §8's own `μ`: its sampling and detection columns within 0.1
/// percentage point, its proof times within half the unit of their last printed digit, and the
/// oracle's values to 1e-12.
#[test]
fn the_calculators_reproduce_section_8() {
    let v = vectors();
    let points = number(&v["tolerances"]["percentage_points"]);
    let s = Scenario::SOURCE;
    let rows = v["section_8"]["rows"].as_array().expect("the §8 rows");
    let oracle = v["calculators"]
        .as_array()
        .expect("the oracle's calculators");
    for (row, oracle) in rows.iter().zip(oracle) {
        let context = row["context"].as_u64().expect("a context") as u32;
        let table_mu = number(&row["mu"]);
        let time = proof_time_s(
            table_mu,
            &Counts::SOURCE_405B,
            &UnitCosts::SOURCE,
            context,
            s.gpus,
        )
        .expect("a proof time");
        let rate = sampling_rate(s.target_overhead, table_mu, s.eta).expect("a rate");
        let detection = detection_probability(rate, s.violations).expect("a probability");
        let printed_time = number(&row["proof_time_s"]);
        assert!(
            (time - printed_time).abs() <= number(&row["proof_time_half_unit_s"]),
            "proof time at {context}: {time:.3} s against {printed_time} s"
        );
        assert!(
            (100.0 * rate - number(&row["sampling_percent"])).abs() <= points,
            "sampling at {context}"
        );
        assert!(
            (100.0 * detection - number(&row["detection_percent"])).abs() <= points,
            "detection at {context}"
        );
        for (ours, key) in [
            (time, "proof_time_s"),
            (rate, "sampling_rate"),
            (detection, "detection"),
        ] {
            assert!(
                close(ours, number(&oracle[key]), 1e-12),
                "{key} at {context}"
            );
        }
    }
}
