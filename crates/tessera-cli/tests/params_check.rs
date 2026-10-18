// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tessera params check`, end to end through the built binary (RFC-0014 'Soundness
//! calculator', 02 §2.2).
//!
//! - **The oracle's reports.** With `--json`, the report of each parameter set at the default
//!   statement, the 405B 16K-token, 16K-context one, is the `tessera.soundness/1` report the
//!   independent oracle writes in `spec/vectors/soundness/soundness.json`, value for value.
//! - **The sizes.** `--lookups N` and `--table-rows T` replace the LogUp instances with one of
//!   `N` lookups and `T` rows, each taking the default's largest instance's when left out:
//!   `N + T = 2^50` fails `ni-100`'s LogUp row and no other.
//! - **The table.** Without `--json`, every row with its bits and target, then the minimum.
//! - **The examples.** `docs/perf/examples/soundness-*.json`, which `cargo xtask lint`
//!   validates against the report's schema, are what the command prints.
//! - **Exit statuses** (02 §2.4). A report exits 0, whether or not every row meets its target:
//!   the report says which do. A malformed argument exits 64.

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

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The oracle's report named `name`.
fn oracle(name: &str) -> Value {
    let path = root().join("spec/vectors/soundness/soundness.json");
    let vectors: Value =
        serde_json::from_str(&fs::read_to_string(path).expect("the vectors")).expect("JSON");
    vectors["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .find(|c| c["name"] == name)
        .unwrap_or_else(|| panic!("the case {name}"))["report"]
        .clone()
}

/// The report `tessera params check --json` prints for `args`, which exits 0.
fn checked(args: &[&str]) -> Value {
    let mut all = vec!["params", "check"];
    all.extend(args);
    all.push("--json");
    let output = tessera(&all);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{args:?}: {}",
        stderr(&output)
    );
    assert_eq!(stderr(&output), "", "{args:?}");
    serde_json::from_str(stdout(&output)).expect("one JSON object")
}

fn rows(report: &Value) -> &Vec<Value> {
    report["rows"].as_array().expect("rows")
}

#[test]
fn the_reports_are_the_oracles_at_the_405b_statement() {
    for (name, id) in [
        ("live-64", "live-64/1"),
        ("ni-100", "ni-100/1"),
        ("test-small", "test-small/1"),
    ] {
        let printed = checked(&["--params", name]);
        assert_eq!(printed["schema"], "tessera.soundness/1");
        assert_eq!(printed, oracle(&format!("405b/{id}")), "{name}");
    }
    assert_eq!(checked(&["--params", "live-64"])["meets_targets"], true);
    assert_eq!(checked(&["--params", "ni-100"])["meets_targets"], true);
    assert_eq!(
        checked(&["--params", "test-small"])["meets_targets"],
        Value::Null
    );
}

#[test]
fn lookups_and_table_rows_replace_the_logup_instances() {
    // N + T = 2^50 with the default's T = 2^22.
    let report = checked(&["--params", "ni-100", "--lookups", "1125899902648320"]);
    let instances = report["statement"]["lookups"].as_array().expect("lookups");
    assert_eq!(instances.len(), 1);
    assert_eq!(instances[0]["lookups"], 1_125_899_902_648_320u64);
    assert_eq!(instances[0]["rows"], 4_194_304);
    let failed: Vec<&Value> = rows(&report)
        .iter()
        .filter(|r| r["pass"] == false)
        .map(|r| &r["row"])
        .collect();
    assert_eq!(failed, ["logup"]);
    assert_eq!(report["meets_targets"], false);
    let logup = &rows(&report)[report["minimum"].as_u64().expect("an index") as usize];
    assert_eq!(logup["row"], "logup");
    assert_eq!(logup["numerator"], "1125899906842624");
    assert_eq!(logup["bits"], 98 * 256 - 1);
    // --table-rows alone keeps the default's N = 2^42.
    let report = checked(&["--params", "live-64", "--table-rows", "1"]);
    let instances = report["statement"]["lookups"].as_array().expect("lookups");
    assert_eq!(instances.len(), 1);
    assert_eq!(instances[0]["lookups"], 1u64 << 42);
    assert_eq!(instances[0]["rows"], 1);
    let logup = rows(&report)
        .iter()
        .find(|r| r["row"] == "logup")
        .expect("the LogUp row");
    assert_eq!(logup["numerator"], ((1u64 << 42) + 1).to_string());
    assert_eq!(report["meets_targets"], true);
}

#[test]
fn the_table_lists_every_row_and_the_minimum() {
    let output = tessera(&["params", "check", "--params", "live-64"]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let text = stdout(&output);
    let report = oracle("405b/live-64/1");
    // A header, a line per row, then the minimum and the verdict.
    let lines: Vec<&str> = text.lines().collect();
    let table: Vec<&str> = lines
        .iter()
        .skip_while(|l| !l.starts_with("row"))
        .skip(1)
        .take_while(|l| !l.is_empty())
        .copied()
        .collect();
    assert_eq!(table.len(), rows(&report).len(), "{text}");
    for (line, row) in table.iter().zip(rows(&report)) {
        // Bits with two decimals, rounded down: the table never shows more than the bound.
        let bits = row["bits"].as_u64().expect("bits");
        let shown = format!("{}.{:02}", bits / 256, bits % 256 * 100 / 256);
        assert!(line.contains(&shown), "{line}: {shown}");
        assert!(line.ends_with("yes"), "{line}");
    }
    assert!(
        table[0].starts_with("sumcheck and tree rounds"),
        "{}",
        table[0]
    );
    assert!(
        table[5].starts_with("proximity queries, advice"),
        "{}",
        table[5]
    );
    assert!(
        text.contains("minimum: proximity queries, Embed: 64.03 bits, target 64"),
        "{text}"
    );
    assert!(text.contains("every row meets its target"), "{text}");
    // A row that misses its target says so, and so does the verdict.
    let output = tessera(&[
        "params",
        "check",
        "--params",
        "ni-100",
        "--lookups",
        "1125899902648320",
    ]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let text = stdout(&output);
    let logup = text
        .lines()
        .find(|l| l.starts_with("LogUp identity"))
        .expect("the LogUp row");
    assert!(logup.contains("97.99") && logup.ends_with("NO"), "{logup}");
    assert!(text.contains("1 row misses its target"), "{text}");
    // test-small is held to none.
    let output = tessera(&["params", "check", "--params", "test-small"]);
    assert!(
        stdout(&output).contains("test-small/1 has no targets: it certifies nothing"),
        "{}",
        stdout(&output)
    );
}

/// `docs/perf/examples/soundness-{live-64,ni-100}.json`, the documented examples of the report,
/// which `cargo xtask lint` validates against its JSON Schema, are what the command prints:
/// `tessera params check --params live-64 --json > FILE` rewrites one.
#[test]
fn the_committed_examples_are_what_the_command_prints() {
    for name in ["live-64", "ni-100"] {
        let path = root().join(format!("docs/perf/examples/soundness-{name}.json"));
        let example: Value =
            serde_json::from_str(&fs::read_to_string(&path).expect("the committed example"))
                .expect("JSON");
        assert_eq!(
            example,
            checked(&["--params", name]),
            "{} is stale: rerun `tessera params check --params {name} --json`",
            path.display()
        );
    }
}

#[test]
fn malformed_arguments_exit_64() {
    for args in [
        &["params", "check"][..],
        &["params", "check", "--params", "live-64/1"],
        &["params", "check", "--params", "live-128"],
        &["params", "check", "--params", "ni-100", "--lookups", "0"],
        &[
            "params",
            "check",
            "--params",
            "ni-100",
            "--table-rows",
            "-1",
        ],
        &["params", "check", "--params", "ni-100", "--lookups", "2^50"],
        &[
            "params",
            "check",
            "--params",
            "ni-100",
            "--lookups",
            "18446744073709551616",
        ],
    ] {
        let output = tessera(args);
        assert_eq!(output.status.code(), Some(64), "{args:?}");
        assert_eq!(stdout(&output), "", "{args:?}");
        assert!(!stderr(&output).is_empty(), "{args:?}");
        // With --json, the one object on stdout names the code.
        let mut json = args.to_vec();
        json.push("--json");
        let output = tessera(&json);
        assert_eq!(output.status.code(), Some(64), "{args:?}");
        let error: Value = serde_json::from_str(stdout(&output)).expect("one JSON object");
        assert_eq!(error["error"]["code"], "ERR-USAGE", "{args:?}");
    }
    // The help documents the default statement and each flag.
    let help = tessera(&["params", "check", "--help"]);
    assert_eq!(help.status.code(), Some(0));
    for text in [
        "405B",
        "16K",
        "2^42",
        "--lookups",
        "--table-rows",
        "RFC-0014",
    ] {
        assert!(stdout(&help).contains(text), "{text}");
    }
}
