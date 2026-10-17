// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tessera bench model`, end to end through the built binary (RFC-0020 'Executable cost
//! model', 02 §2): the report it prints and writes, and the exit code of each failure (02 §2.4).

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

use serde_json::Value;

fn tessera(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tessera"))
        .args(args)
        .output()
        .expect("the tessera binary runs")
}

fn stdout(output: &Output) -> &str {
    std::str::from_utf8(&output.stdout).expect("UTF-8")
}

fn stderr(output: &Output) -> &str {
    std::str::from_utf8(&output.stderr).expect("UTF-8")
}

fn vectors() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../spec/vectors/cost-model/cost-model.json");
    serde_json::from_str(&fs::read_to_string(path).expect("the vectors")).expect("JSON")
}

/// A directory of its own for one test.
fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tessera-cli-{test}-{}", std::process::id()));
    fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

/// With the source's defaults, the report is a `tessera.cost-model/1` object whose eight
/// projections reproduce the source's chart within ±0.05, with the source's scenario.
#[test]
fn the_source_defaults_reproduce_the_source_chart() {
    let output = tessera(&["bench", "model", "--source-defaults", "--json"]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    assert_eq!(stderr(&output), "");
    let report: Value = serde_json::from_str(stdout(&output)).expect("one JSON object");
    assert_eq!(report["schema"], "tessera.cost-model/1");
    assert_eq!(report["inputs"]["source_defaults"], true);
    assert_eq!(report["inputs"]["scenario"]["gpus"], 8);
    let v = vectors();
    let chart = &v["chart"];
    let projections = report["projections"].as_array().expect("projections");
    assert_eq!(projections.len(), 8);
    for (k, p) in projections.iter().enumerate() {
        assert_eq!(p["context"], chart["contexts"][k]);
        for (ours, theirs) in [
            ("total", "total"),
            ("linear", "linear"),
            ("attention", "attention"),
        ] {
            let (a, b) = (&p["mu"][ours], &chart[theirs][k]);
            let (a, b) = (a.as_f64().expect("μ"), b.as_f64().expect("the chart"));
            assert!(
                (a - b).abs() <= 0.05,
                "{ours} at {}: {a} against {b}",
                p["context"]
            );
        }
        assert_eq!(p["mu"]["regen"], 0.0);
        assert_eq!(p["mu"]["streaming"], 0.0);
        assert_eq!(p["mu"]["column_answers"], 0.0);
        assert_eq!(p["mu"]["row_combination"], 0.0);
        // The calculators of source §8 from the printed μ, recomputed here from the vectors'
        // inputs: 10% overhead at η = 0.3, 100 violating requests, one request on 8 GPUs.
        let total = p["mu"]["total"].as_f64().expect("μ");
        let context = p["context"].as_f64().expect("a context");
        let input = |name: &str| v["inputs"][name]["value"].as_f64().expect("an input");
        let native = (input("native_ops_lin") + input("native_ops_att_per_ctx") * context)
            / input("native_ops_per_s");
        let rate = (0.1 / (total * 0.3)).min(1.0);
        for (key, expected) in [
            ("sampling_rate", rate),
            ("detection", 1.0 - (1.0 - rate).powi(100)),
            ("proof_time_s", total * context * native / 8.0),
        ] {
            let printed = p[key].as_f64().expect("a number");
            assert!(
                (printed - expected).abs() <= 1e-9 * expected,
                "{key} at {context}: {printed} against {expected}"
            );
        }
    }
}

/// Whether `a` and `b` are the same JSON document up to the last bits of their numbers: the same
/// keys, strings, booleans and array lengths, and numbers within `1e-12` of each other, relative.
/// The model's probabilities go through `ln_1p` and `exp_m1`, which no platform rounds correctly,
/// so the last bits differ between, say, aarch64 macOS and x86_64 Linux.
fn same_report(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (
                x.as_f64().unwrap_or(f64::NAN),
                y.as_f64().unwrap_or(f64::NAN),
            );
            (x - y).abs() <= 1e-12 * x.abs().max(y.abs())
        }
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(x, y)| same_report(x, y))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len()
                && x.iter()
                    .all(|(key, x)| y.get(key).is_some_and(|y| same_report(x, y)))
        }
        _ => a == b,
    }
}

/// `docs/perf/examples/cost-model-source-defaults.json`, the documented example of the report,
/// which `cargo xtask lint` validates against its JSON Schema, is what the command prints, up to
/// the last bits of its numbers: a model that changes changes it, and
/// `--source-defaults --json > FILE` rewrites it.
#[test]
fn the_committed_example_is_what_the_command_prints() {
    let output = tessera(&["bench", "model", "--source-defaults", "--json"]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/perf/examples/cost-model-source-defaults.json");
    let example: Value =
        serde_json::from_str(&fs::read_to_string(&path).expect("the committed example"))
            .expect("JSON");
    let printed: Value = serde_json::from_str(stdout(&output)).expect("JSON");
    assert!(
        same_report(&example, &printed),
        "{} is stale: rerun `tessera bench model --source-defaults --json > {}`",
        path.display(),
        "docs/perf/examples/cost-model-source-defaults.json"
    );
    // The comparison itself: a report off in one number's sixth digit is not the same.
    let mut off = printed.clone();
    off["projections"][0]["mu"]["total"] = serde_json::json!(3.18);
    assert!(!same_report(&printed, &off));
}

/// `--out` writes the report that `--json` prints; without `--json`, stdout is a table with one
/// row per requested context.
#[test]
fn out_writes_the_report_and_stdout_shows_it() {
    let dir = scratch("out");
    let path = dir.join("mu.json");
    let shown = path.to_str().expect("a UTF-8 path");
    let json = tessera(&[
        "bench",
        "model",
        "--source-defaults",
        "--json",
        "--out",
        shown,
    ]);
    assert_eq!(json.status.code(), Some(0), "{}", stderr(&json));
    assert_eq!(
        fs::read_to_string(&path).expect("the report"),
        stdout(&json)
    );

    let table = tessera(&[
        "bench",
        "model",
        "--source-defaults",
        "--context",
        "1024,4096",
        "--out",
        shown,
    ]);
    assert_eq!(table.status.code(), Some(0), "{}", stderr(&table));
    let lines: Vec<&str> = stdout(&table).lines().collect();
    assert_eq!(lines.len(), 3, "a header and one row per context");
    assert!(lines[1].trim_start().starts_with("1024") && lines[2].trim_start().starts_with("4096"));
    let report: Value = serde_json::from_str(&fs::read_to_string(&path).expect("the report"))
        .expect("the file holds the report");
    let contexts: Vec<&Value> = report["projections"]
        .as_array()
        .expect("projections")
        .iter()
        .map(|p| &p["context"])
        .collect();
    assert_eq!(contexts, [1024, 4096]);
    let _ = fs::remove_dir_all(dir);
}

/// Usage errors exit 64 with `ERR-USAGE` and print nothing on stdout; help and version succeed.
#[test]
fn usage_errors_exit_64() {
    for args in [
        &["bench", "model"][..],
        &["bench", "model", "--source-defaults", "--eta", "0"],
        &[
            "bench",
            "model",
            "--source-defaults",
            "--target-overhead=-0.1",
        ],
        &["bench", "model", "--source-defaults", "--gpus", "0"],
        &["bench", "model", "--source-defaults", "--batch-tokens", "0"],
        &["bench", "model", "--source-defaults", "--context", "1K"],
        &["bench", "model", "--source-defaults", "--plan", "plan.tep"],
        &["bench"],
        &[],
    ] {
        let output = tessera(args);
        assert_eq!(
            output.status.code(),
            Some(64),
            "{args:?}: {}",
            stderr(&output)
        );
        assert_eq!(stdout(&output), "", "{args:?}");
        assert!(!stderr(&output).is_empty(), "{args:?}");
    }
    let refused = tessera(&["bench", "model", "--source-defaults", "--eta", "0"]);
    assert_eq!(
        stderr(&refused),
        "error: ERR-USAGE: the input eta is not positive\n"
    );
    for args in [
        &["--help"][..],
        &["--version"],
        &["bench", "model", "--help"],
    ] {
        let output = tessera(args);
        assert_eq!(output.status.code(), Some(0), "{args:?}");
        assert!(!stdout(&output).is_empty(), "{args:?}");
    }
}

/// A report that cannot be written exits 20 with `ERR-RES-IO`, and leaves nothing behind.
#[test]
fn a_report_that_cannot_be_written_exits_20() {
    let dir = scratch("unwritable");
    let path = dir.join("missing").join("mu.json");
    let output = tessera(&[
        "bench",
        "model",
        "--source-defaults",
        "--out",
        path.to_str().expect("UTF-8"),
    ]);
    assert_eq!(output.status.code(), Some(20));
    assert!(
        stderr(&output).starts_with("error: ERR-RES-IO: cannot write "),
        "{}",
        stderr(&output)
    );
    assert_eq!(stdout(&output), "");
    assert_eq!(fs::read_dir(&dir).expect("the directory").count(), 0);
    let _ = fs::remove_dir_all(dir);
}
