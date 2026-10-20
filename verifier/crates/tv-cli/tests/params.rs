// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Integration tests for `tessera-verify params` (RFC-0014 'Soundness calculator', 02 §2.3).

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

use tv_testkit::Json;

fn tessera_verify(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tessera-verify"))
        .args(args)
        .output()
        .expect("the tessera-verify binary runs")
}

fn stdout(output: &Output) -> &str {
    std::str::from_utf8(&output.stdout).expect("UTF-8")
}

fn stderr(output: &Output) -> &str {
    std::str::from_utf8(&output.stderr).expect("UTF-8")
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn canonical_json(j: &Json) -> Json {
    match j {
        Json::Array(items) => Json::Array(items.iter().map(canonical_json).collect()),
        Json::Object(members) => {
            let mut sorted: Vec<(String, Json)> = members
                .iter()
                .map(|(k, v)| (k.clone(), canonical_json(v)))
                .collect();
            sorted.sort_by(|a, b| a.0.cmp(&b.0));
            Json::Object(sorted)
        }
        other => other.clone(),
    }
}

fn oracle(name: &str) -> Json {
    let vectors = tv_testkit::read_vectors("soundness/soundness.json");
    vectors
        .get("cases")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c.get("name").unwrap().as_str() == Some(name))
        .unwrap_or_else(|| panic!("missing oracle case {name}"))
        .get("report")
        .unwrap()
        .clone()
}

fn checked(args: &[&str]) -> Json {
    let mut all = vec!["params"];
    all.extend_from_slice(args);
    all.push("--json");
    let output = tessera_verify(&all);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{args:?}: {}",
        stderr(&output)
    );
    assert_eq!(stderr(&output), "", "{args:?}");
    Json::parse(stdout(&output)).expect("one JSON object")
}

#[test]
fn reports_match_the_oracle_and_committed_examples_at_405b() {
    for (name, id) in [
        ("live-64", "live-64/1"),
        ("ni-100", "ni-100/1"),
        ("test-small", "test-small/1"),
    ] {
        let printed = checked(&["--params", name]);
        assert_eq!(
            printed.get("schema").unwrap().as_str(),
            Some("tessera.soundness/1")
        );
        assert_eq!(
            canonical_json(&printed),
            canonical_json(&oracle(&format!("405b/{id}"))),
            "{name}"
        );
    }

    for name in ["live-64", "ni-100"] {
        let path = root().join(format!("docs/perf/examples/soundness-{name}.json"));
        let committed = fs::read_to_string(&path).expect("committed example");
        let output = tessera_verify(&["params", "--params", name, "--json"]);
        assert_eq!(output.status.code(), Some(0));
        assert_eq!(stdout(&output), committed, "{}", path.display());
    }
}

#[test]
fn lookups_and_table_rows_override_logup_instances() {
    // N + T = 2^50 with default T = 2^22 fails ni-100's LogUp row.
    let report = checked(&["--params", "ni-100", "--lookups", "1125899902648320"]);
    let instances = report
        .get("statement")
        .unwrap()
        .get("lookups")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(instances.len(), 1);
    assert_eq!(
        instances[0].get("lookups").unwrap().as_u64(),
        Some(1_125_899_902_648_320)
    );
    assert_eq!(instances[0].get("rows").unwrap().as_u64(), Some(4_194_304));
    assert_eq!(report.get("meets_targets"), Some(&Json::Bool(false)));

    let rows = report.get("rows").unwrap().as_array().unwrap();
    let failed: Vec<&str> = rows
        .iter()
        .filter(|r| r.get("pass") == Some(&Json::Bool(false)))
        .map(|r| r.get("row").unwrap().as_str().unwrap())
        .collect();
    assert_eq!(failed, ["logup"]);
    let min_idx = usize::try_from(report.get("minimum").unwrap().as_u64().unwrap()).unwrap();
    let logup = &rows[min_idx];
    assert_eq!(logup.get("row").unwrap().as_str(), Some("logup"));
    assert_eq!(
        logup.get("numerator").unwrap().as_str(),
        Some("1125899906842624")
    );
    assert_eq!(logup.get("bits").unwrap().as_u64(), Some(98 * 256 - 1));

    // --table-rows alone keeps default N = 2^42.
    let report = checked(&["--params", "live-64", "--table-rows", "1"]);
    let instances = report
        .get("statement")
        .unwrap()
        .get("lookups")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(instances.len(), 1);
    assert_eq!(
        instances[0].get("lookups").unwrap().as_u64(),
        Some(1u64 << 42)
    );
    assert_eq!(instances[0].get("rows").unwrap().as_u64(), Some(1));
    assert_eq!(report.get("meets_targets"), Some(&Json::Bool(true)));
}

#[test]
fn human_readable_table_lists_every_row_and_minimum() {
    let output = tessera_verify(&["params", "--params", "live-64"]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let text = stdout(&output);
    let report = oracle("405b/live-64/1");
    let rows = report.get("rows").unwrap().as_array().unwrap();
    let lines: Vec<&str> = text.lines().collect();
    let table: Vec<&str> = lines
        .iter()
        .skip_while(|l| !l.starts_with("row"))
        .skip(1)
        .take_while(|l| !l.is_empty())
        .copied()
        .collect();
    assert_eq!(table.len(), rows.len(), "{text}");
    for (line, row) in table.iter().zip(rows) {
        let bits = row.get("bits").unwrap().as_u64().unwrap();
        let shown = format!("{}.{:02}", bits / 256, bits % 256 * 100 / 256);
        assert!(line.contains(&shown), "{line}: {shown}");
        assert!(line.ends_with("yes"), "{line}");
    }
    assert!(table[0].starts_with("sumcheck and tree rounds"));
    assert!(table[5].starts_with("proximity queries, advice"));
    assert!(text.contains("minimum: proximity queries, advice: 64.32 bits, target 64"));
    assert!(text.contains("every row meets its target"));

    let output = tessera_verify(&[
        "params",
        "--params",
        "ni-100",
        "--lookups",
        "1125899902648320",
    ]);
    assert_eq!(output.status.code(), Some(0));
    let text = stdout(&output);
    let logup = text
        .lines()
        .find(|l| l.starts_with("LogUp identity"))
        .expect("LogUp row");
    assert!(logup.contains("97.99") && logup.ends_with("NO"), "{logup}");
    assert!(text.contains("1 row misses its target"), "{text}");

    let output = tessera_verify(&["params", "--params", "test-small"]);
    assert_eq!(output.status.code(), Some(0));
    assert!(
        stdout(&output).contains("test-small/1 has no targets: it certifies nothing"),
        "{}",
        stdout(&output)
    );
}

#[test]
fn malformed_arguments_exit_64() {
    for args in [
        &["params"][..],
        &["params", "--params", "live-64/1"],
        &["params", "--params", "live-128"],
        &["params", "--params", "ni-100", "--lookups", "0"],
        &["params", "--params", "ni-100", "--table-rows", "-1"],
        &["params", "--params", "ni-100", "--lookups", "2^50"],
        &[
            "params",
            "--params",
            "ni-100",
            "--lookups",
            "18446744073709551616",
        ],
    ] {
        let output = tessera_verify(args);
        assert_eq!(output.status.code(), Some(64), "{args:?}");
        assert_eq!(stdout(&output), "", "{args:?}");
        assert!(!stderr(&output).is_empty(), "{args:?}");

        let mut json = args.to_vec();
        json.push("--json");
        let output = tessera_verify(&json);
        assert_eq!(output.status.code(), Some(64), "{args:?}");
        let err = Json::parse(stdout(&output)).expect("JSON error object");
        assert_eq!(
            err.get("error").unwrap().get("code").unwrap().as_str(),
            Some("ERR-USAGE"),
            "{args:?}"
        );
    }

    let help = tessera_verify(&["params", "--help"]);
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
