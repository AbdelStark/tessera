// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tessera audit plan` end to end (RFC-0016 'Calculators'):
//! - `docs/perf/examples/audit-plan.json` is what the command prints for the documented policy
//!   and traffic, up to the last bits of its numbers, and `--out` writes what `--json` prints;
//! - each bucket's `μ` is the cost model's at the bucket's mean length, as `tessera bench model`
//!   projects it;
//! - each traffic profile the command must refuse exits 64, `ERR-USAGE`, naming its fault.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tessera-plan-{test}-{}", std::process::id()));
    fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

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

/// The documented policy and traffic.
fn examples() -> (String, String) {
    let at = |name: &str| {
        root()
            .join("docs/perf/examples")
            .join(name)
            .to_str()
            .expect("a UTF-8 path")
            .to_owned()
    };
    (at("audit-policy.json"), at("audit-traffic.json"))
}

/// Whether `a` and `b` are the same JSON document up to the last bits of their numbers: the
/// probabilities go through `ln_1p` and `exp_m1`, which platforms round differently.
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

/// `docs/perf/examples/audit-plan.json`, which `cargo xtask lint` validates against its JSON
/// Schema, is what the command prints: `tessera --json audit plan --policy
/// docs/perf/examples/audit-policy.json --traffic docs/perf/examples/audit-traffic.json
/// --source-defaults > FILE` rewrites it. `--out` writes the same report, and without `--json`
/// stdout is a table with a row per bucket and the totals.
#[test]
fn the_committed_example_is_what_the_command_prints() {
    let (policy, traffic) = examples();
    let dir = scratch("example");
    let out = dir.join("plan.json");
    let args = [
        "--json",
        "audit",
        "plan",
        "--policy",
        &policy,
        "--traffic",
        &traffic,
        "--source-defaults",
        "--out",
        out.to_str().expect("a UTF-8 path"),
    ];
    let output = tessera(&args);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let printed: Value = serde_json::from_str(stdout(&output)).expect("JSON");
    let path = root().join("docs/perf/examples/audit-plan.json");
    let example: Value =
        serde_json::from_str(&fs::read_to_string(&path).expect("the committed example"))
            .expect("JSON");
    assert!(
        same_report(&example, &printed),
        "{} is stale: rerun the command in this test's documentation",
        path.display()
    );
    let written = fs::read_to_string(&out).expect("--out's report");
    assert_eq!(written, stdout(&output));
    // The comparison catches a change in a number's sixth digit.
    let mut off = printed.clone();
    off["detection"] = serde_json::json!(0.330218);
    assert!(!same_report(&printed, &off));

    let table = tessera(&[
        "audit",
        "plan",
        "--policy",
        &policy,
        "--traffic",
        &traffic,
        "--source-defaults",
    ]);
    assert_eq!(table.status.code(), Some(0), "{}", stderr(&table));
    let lines: Vec<&str> = stdout(&table).lines().collect();
    assert_eq!(lines.len(), 1 + 3 + 1, "{}", stdout(&table));
    assert!(lines[0].contains("detection") && lines[0].contains("in place"));
    assert!(lines[4].starts_with("detection 33.0"), "{}", lines[4]);
    fs::remove_dir_all(&dir).ok();
}

/// Each bucket's `μ` is `tessera bench model`'s at the bucket's mean length, with the same
/// scenario.
#[test]
fn each_buckets_mu_is_the_cost_models_at_its_mean_length() {
    let (policy, traffic) = examples();
    let plan = tessera(&[
        "--json",
        "audit",
        "plan",
        "--policy",
        &policy,
        "--traffic",
        &traffic,
        "--source-defaults",
    ]);
    assert_eq!(plan.status.code(), Some(0), "{}", stderr(&plan));
    let plan: Value = serde_json::from_str(stdout(&plan)).expect("JSON");
    let model = tessera(&[
        "--json",
        "bench",
        "model",
        "--source-defaults",
        "--context",
        "700,6000,60000",
    ]);
    assert_eq!(model.status.code(), Some(0), "{}", stderr(&model));
    let model: Value = serde_json::from_str(stdout(&model)).expect("JSON");
    let buckets = plan["buckets"].as_array().expect("buckets");
    let projections = model["projections"].as_array().expect("projections");
    assert_eq!(buckets.len(), projections.len());
    for (b, p) in buckets.iter().zip(projections) {
        assert_eq!(b["tokens"], p["context"]);
        assert_eq!(b["mu"], p["mu"]["total"], "{b}");
        // At a bucket's own target rate, its traffic alone costs the target overhead: the rate
        // `bench model` reports for the same scenario.
        assert_eq!(b["target_rate"], p["sampling_rate"], "{b}");
    }
}

/// Traffic profiles the command refuses, each `ERR-USAGE`, exit 64, with its fault in the
/// message.
#[test]
fn ill_formed_traffic_is_refused_with_err_usage() {
    let (policy, good) = examples();
    let dir = scratch("refused");
    let good: Value =
        serde_json::from_str(&fs::read_to_string(&good).expect("the example")).expect("JSON");
    let bucket = |b: usize| good["buckets"][b].clone();
    let with_buckets = |buckets: Vec<Value>| {
        let mut v = good.clone();
        v["buckets"] = Value::Array(buckets);
        v
    };
    let set = |b: usize, key: &str, value: Value| {
        let mut v = good.clone();
        v["buckets"][b][key] = value;
        v
    };
    let mut unknown = good.clone();
    unknown["period"] = serde_json::json!(1);
    let mut extra_key = good.clone();
    extra_key["buckets"][0]["mean"] = serde_json::json!(1);
    let mut other_schema = good.clone();
    other_schema["schema"] = serde_json::json!("tessera.audit-traffic/2");
    let cases: Vec<(&str, String, &str)> = vec![
        ("not JSON", "{".to_owned(), "is not JSON"),
        (
            "another schema",
            other_schema.to_string(),
            "is not a tessera.audit-traffic/1 file",
        ),
        ("an unknown key", unknown.to_string(), "unknown key period"),
        (
            "a bucket's fourth key",
            extra_key.to_string(),
            "{requests, tokens, violations}",
        ),
        (
            "a negative count",
            set(0, "requests", serde_json::json!(-1)).to_string(),
            "requests is not an unsigned integer",
        ),
        (
            "two buckets for three",
            with_buckets(vec![bucket(0), bucket(1)]).to_string(),
            "the policy has 3 buckets, and 2 entries were given",
        ),
        (
            "a mean past its bucket",
            set(0, "tokens", serde_json::json!(1_025)).to_string(),
            "bucket 0's mean length does not lie in the bucket",
        ),
        (
            "a mean in the bucket before",
            set(2, "tokens", serde_json::json!(16_384)).to_string(),
            "bucket 2's mean length does not lie in the bucket",
        ),
        (
            "more violations than requests",
            set(1, "violations", serde_json::json!(180_001)).to_string(),
            "bucket 1 has more violating requests than requests",
        ),
        (
            "no request",
            with_buckets(
                (0..3)
                    .map(|b| {
                        let mut x = bucket(b);
                        x["requests"] = serde_json::json!(0);
                        x["violations"] = serde_json::json!(0);
                        x
                    })
                    .collect(),
            )
            .to_string(),
            "the traffic has no request",
        ),
    ];
    for (what, text, fault) in cases {
        let path = dir.join("traffic.json");
        fs::write(&path, text).expect("a traffic file");
        let output = tessera(&[
            "--json",
            "audit",
            "plan",
            "--policy",
            &policy,
            "--traffic",
            path.to_str().expect("a UTF-8 path"),
            "--source-defaults",
        ]);
        assert_eq!(
            output.status.code(),
            Some(64),
            "{what}: {}",
            stderr(&output)
        );
        let error: Value = serde_json::from_str(stdout(&output)).expect("JSON");
        assert_eq!(error["error"]["code"], "ERR-USAGE", "{what}");
        let message = error["error"]["message"].as_str().expect("a message");
        assert!(message.contains(fault), "{what}: {message}");
    }
    // Neither counts given, or both: refused before anything is read.
    for counts in [&[][..], &["--source-defaults", "--plan", "plan.tep"][..]] {
        let mut args = vec!["audit", "plan", "--policy", &policy, "--traffic", &policy];
        args.extend_from_slice(counts);
        let output = tessera(&args);
        assert_eq!(
            output.status.code(),
            Some(64),
            "{counts:?}: {}",
            stderr(&output)
        );
    }
    fs::remove_dir_all(&dir).ok();
}
