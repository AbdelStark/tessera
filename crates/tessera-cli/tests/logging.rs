// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `--log-format` and `--log-level` through the built binary (05 §2, 02 §2):
//!
//! - **Unchanged by default.** At the default level and format, `tessera compile` writes exactly
//!   its one progress line, and a failure exactly `error: ERR-...: what happened`.
//! - **Levels.** `--log-level debug` adds the compiler's per-stage event with its span path;
//!   `--log-level error` keeps the failure and drops the progress line.
//! - **JSON lines.** `--log-format json` writes one object per line carrying the same message.
//!   `tessera-e2e`'s `exact_cli` test validates real runs' lines against the schema.

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

fn stderr(output: &Output) -> &str {
    std::str::from_utf8(&output.stderr).expect("UTF-8")
}

fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tessera-logging-{test}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

fn compile(dir: &str, extra: &[&str]) -> Output {
    let model = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../spec/vectors/exact-cli/tiny-3-reference.d");
    let mut args: Vec<&str> = extra.to_vec();
    let model = model.to_str().expect("UTF-8 path");
    args.extend([
        "compile",
        "--model",
        model,
        "--params",
        "test-small",
        "--max-log2-rows",
        "18",
        "--query-block",
        "2",
        "--out-dir",
        dir,
    ]);
    tessera(&args)
}

#[test]
fn default_lines_are_unchanged() {
    let dir = scratch("default");
    let out = dir.join("out");
    let shown = out.to_str().expect("UTF-8 path");
    let output = compile(shown, &[]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    assert_eq!(
        stderr(&output),
        format!("tessera compile: wrote claims.tcl, plan.tep, commitment-report.json to {shown}\n")
    );
    let missing = dir.join("missing.tnp");
    let failed = tessera(&["inspect", missing.to_str().expect("UTF-8 path")]);
    assert_eq!(failed.status.code(), Some(20));
    let line = stderr(&failed);
    assert!(
        line.starts_with("error: ERR-RES-IO: cannot read "),
        "{line}"
    );
    assert_eq!(line.lines().count(), 1, "{line}");
}

#[test]
fn levels_choose_the_lines() {
    let dir = scratch("levels");
    let debug = compile(
        dir.join("debug").to_str().unwrap(),
        &["--log-level", "debug"],
    );
    assert_eq!(debug.status.code(), Some(0), "{}", stderr(&debug));
    let lines: Vec<&str> = stderr(&debug).lines().collect();
    assert_eq!(lines.len(), 2, "{lines:?}");
    assert!(
        lines[0].starts_with("debug compile{layers=3 params=test-small/1}: ")
            && lines[0].ends_with(" gadgets"),
        "{lines:?}"
    );
    let quiet = compile(
        dir.join("quiet").to_str().unwrap(),
        &["--log-level", "error"],
    );
    assert_eq!(quiet.status.code(), Some(0));
    assert_eq!(stderr(&quiet), "");
    let missing = dir.join("missing.tnp");
    let failed = tessera(&["--log-level", "error", "inspect", missing.to_str().unwrap()]);
    assert!(stderr(&failed).starts_with("error: ERR-RES-IO: "));
}

#[test]
fn json_lines_carry_the_message() {
    let dir = scratch("json");
    let out = dir.join("out");
    let shown = out.to_str().unwrap();
    let output = compile(shown, &["--log-format", "json"]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let lines: Vec<Value> = stderr(&output)
        .lines()
        .map(|l| serde_json::from_str(l).expect("one JSON object per line"))
        .collect();
    assert_eq!(lines.len(), 1);
    let line = &lines[0];
    assert_eq!(line["event"], "tessera.cli.compile_written");
    assert_eq!(line["level"], "info");
    assert_eq!(line["target"], "tessera_cli::compile");
    assert_eq!(line["span_path"], Value::Array(Vec::new()));
    assert_eq!(line["fields"]["dir"], shown);
    assert_eq!(
        line["fields"]["message"],
        format!("tessera compile: wrote claims.tcl, plan.tep, commitment-report.json to {shown}")
    );
}
