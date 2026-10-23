// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The run manifest (05 §6) through the built binary, with `--features test-rng`:
//!
//! - **OBS-5.** Two identical runs of `compile`, `commit-weights` and a pinned-mask NI `prove`
//!   write manifests that differ only in timings and host fields, so their output digests agree.
//! - **The schema.** Every manifest, successful or failed, validates against
//!   `spec/schemas/run-manifest-v1.schema.json`.
//! - **No secret-class value.** A secret-class artifact is named without its digest, and the
//!   digests of the secret inputs and outputs appear nowhere in a manifest.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn tessera(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tessera"))
        .args(args)
        .output()
        .expect("the tessera binary runs")
}

fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tessera-manifest-{test}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The file's envelope digest, its last 32 bytes.
fn digest(path: &Path) -> String {
    let bytes = fs::read(path).expect("the artifact");
    hex(&bytes[bytes.len() - 32..])
}

/// The manifest at `path`, checked against its schema.
fn manifest(path: &Path) -> Value {
    let text = fs::read_to_string(path).expect("a run manifest");
    let value: Value = serde_json::from_str(&text).expect("JSON");
    let schema: Value = serde_json::from_slice(
        &fs::read(root().join("spec/schemas/run-manifest-v1.schema.json")).expect("the schema"),
    )
    .expect("JSON");
    assert!(xtask::reports::schema_problems(&schema).is_empty());
    let errors = xtask::reports::validate(&schema, &schema, &value, "");
    assert!(errors.is_empty(), "{errors:?}\n{text}");
    value
}

/// The manifest without what OBS-5 lets two identical runs differ in: timings, host fields and
/// the unit costs and memory peaks measured from them.
fn reproducible(mut value: Value) -> Value {
    let object = value.as_object_mut().expect("an object");
    for key in ["host", "started_at", "elapsed_seconds"] {
        assert!(object.remove(key).is_some(), "{key}");
    }
    let measured = [
        "tessera_unit_cost_ps",
        "tessera_glue_fraction",
        "tessera_host_memory_peak_bytes",
    ];
    let metrics = object["metrics"].as_array_mut().expect("metrics");
    metrics
        .retain(|m| m["type"] != "histogram" && !measured.contains(&m["name"].as_str().unwrap()));
    value
}

fn run_twice(args: &[&str], path: &Path) -> Value {
    let first = tessera(args);
    assert_eq!(
        first.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    let a = manifest(path);
    let second = tessera(args);
    assert_eq!(
        second.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&second.stderr)
    );
    let b = manifest(path);
    assert_eq!(reproducible(a.clone()), reproducible(b), "OBS-5");
    a
}

fn outputs(value: &Value) -> Vec<(String, String, Value)> {
    value["outputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| {
            (
                o["schema"].as_str().unwrap().to_owned(),
                o["class"].as_str().unwrap().to_owned(),
                o["digest"].clone(),
            )
        })
        .collect()
}

#[test]
fn identical_runs_write_identical_manifests() {
    let dir = scratch("obs5");
    let model = root().join("spec/vectors/exact-cli/tiny-3-reference.d");
    let tokens = root().join("spec/vectors/exact-cli/recorded.ttk");
    let (model, tokens) = (model.to_str().unwrap(), tokens.to_str().unwrap());
    let compiled = dir.join("compiled");
    let compiled_s = compiled.to_str().unwrap();
    let compile = run_twice(
        &[
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
            compiled_s,
        ],
        &compiled.join("run-manifest.json"),
    );
    assert_eq!(compile["command"], "compile");
    assert_eq!(compile["params"], "test-small/1");
    assert_eq!(compile["exit_code"], 0);
    assert_eq!(compile["error_code"], Value::Null);
    assert_eq!(compile["tool"]["features"], serde_json::json!(["test-rng"]));
    assert_eq!(compile["inputs"][0]["schema"], "IntegerModel");
    assert_eq!(compile["inputs"][0]["class"], "secret");
    assert_eq!(compile["inputs"][0]["digest"], Value::Null);
    assert_eq!(
        outputs(&compile),
        vec![
            (
                "ClaimList".into(),
                "public".into(),
                Value::String(digest(&compiled.join("claims.tcl")))
            ),
            (
                "ExecutionPlan".into(),
                "private".into(),
                Value::String(digest(&compiled.join("plan.tep")))
            ),
            (
                "tessera.commitment-report/1".into(),
                "public".into(),
                Value::Null
            ),
        ]
    );

    let keys = dir.join("keys");
    fs::create_dir_all(&keys).unwrap();
    let weights = keys.join("weights.twc");
    let weights_s = weights.to_str().unwrap();
    let plan = compiled.join("plan.tep");
    let plan_s = plan.to_str().unwrap();
    let commit = run_twice(
        &[
            "commit-weights",
            "--model",
            model,
            "--plan",
            plan_s,
            "--out",
            weights_s,
        ],
        &keys.join("run-manifest.json"),
    );
    assert_eq!(
        outputs(&commit),
        vec![
            ("WeightCommitmentState".into(), "secret".into(), Value::Null),
            (
                "WeightCommitmentPublic".into(),
                "public".into(),
                Value::String(digest(&keys.join("weights.twr")))
            ),
        ]
    );
    let committed = commit["metrics"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| {
            m["name"] == "tessera_committed_elements_total" && m["labels"]["class"] == "weight"
        })
        .expect("weight elements counted");
    assert!(committed["value"].as_u64().unwrap() > 0);

    let proof = dir.join("proof.tnp");
    let proof_s = proof.to_str().unwrap();
    let session = "11".repeat(16);
    let manifest_digest = "13".repeat(32);
    let prove: Vec<&str> = vec![
        "prove",
        "--model",
        model,
        "--plan",
        plan_s,
        "--weights",
        weights_s,
        "--reference-tokens",
        tokens,
        "--prompt-len",
        "2",
        "--request-id",
        "23",
        "--request-session",
        &session,
        "--manifest-digest",
        &manifest_digest,
        "--out",
        proof_s,
        "--test-mask-seed",
        "135",
    ];
    // The proof path is fresh for each run (no overwrite): rename the first proof away.
    let first = tessera(&prove);
    assert_eq!(
        first.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    let a = manifest(&keys.join("run-manifest.json"));
    let first_digest = digest(&proof);
    fs::rename(&proof, dir.join("first.tnp")).unwrap();
    let second = tessera(&prove);
    assert_eq!(
        second.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&second.stderr)
    );
    let b = manifest(&keys.join("run-manifest.json"));
    assert_eq!(reproducible(a.clone()), reproducible(b), "OBS-5");
    assert_eq!(
        digest(&proof),
        first_digest,
        "a pinned-mask proof reproduces byte for byte"
    );
    assert_eq!(a["protocol"], "tessera-protocol/1");
    assert!(outputs(&a).contains(&(
        "NiProof".into(),
        "protocol".into(),
        Value::String(first_digest)
    )));
    // Every label 05 §4 gives the metrics the CPU prover records.
    let labels = |name: &str, key: &str| -> std::collections::BTreeSet<String> {
        a["metrics"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|m| m["name"] == name)
            .map(|m| m["labels"][key].as_str().unwrap().to_owned())
            .collect()
    };
    let set = |items: &[&str]| items.iter().map(|s| (*s).to_owned()).collect();
    assert_eq!(
        labels("tessera_phase_seconds", "phase"),
        set(&["opening", "round1", "round3", "stage", "witness"])
    );
    assert_eq!(
        labels("tessera_stage_seconds", "stage"),
        set(&["0", "1", "2"])
    );
    assert_eq!(
        labels("tessera_committed_elements_total", "class"),
        set(&["advice", "mask", "weight"])
    );
    assert_eq!(
        labels("tessera_proof_bytes", "part"),
        set(&["columns", "openings", "round1", "sumcheck"])
    );
    assert_eq!(
        labels("tessera_unit_cost_ps", "kind"),
        set(&["commit", "lookup", "product_leaf"])
    );
    // RFC-0018 'Pass structure': every pass counted and timed, P3 once per reduction stage.
    let passes = labels("tessera_passes_total", "pass");
    assert_eq!(passes, set(&["p1", "p2", "p3", "p4", "p5"]));
    assert_eq!(labels("tessera_pass_seconds", "pass"), passes);
    let count = |pass: &str| {
        a["metrics"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["name"] == "tessera_passes_total" && m["labels"]["pass"] == pass)
            .map(|m| m["value"].as_u64().unwrap())
    };
    for (pass, n) in [("p1", 1), ("p2", 1), ("p3", 2), ("p4", 1), ("p5", 1)] {
        assert_eq!(count(pass), Some(n), "{pass}");
    }
    let commit = a["tool"]["git_commit"]
        .as_str()
        .expect("a build inside a git checkout");
    assert!(
        commit.len() == 40 && commit.bytes().all(|b| b.is_ascii_hexdigit()),
        "{commit}"
    );
    let names: Vec<&str> = a["metrics"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["name"].as_str().unwrap())
        .collect();
    for name in [
        "tessera_phase_seconds",
        "tessera_stage_seconds",
        "tessera_committed_elements_total",
        "tessera_lookups_total",
        "tessera_table_rows_total",
        "tessera_product_leaves_total",
        "tessera_proof_bytes",
        "tessera_unit_cost_ps",
        "tessera_glue_fraction",
        "tessera_passes_total",
        "tessera_pass_seconds",
    ] {
        assert!(names.contains(&name), "{name}: {names:?}");
    }

    // OBS-6: no digest of a secret-class artifact, anywhere in any manifest.
    for secret in [
        digest(Path::new(model)),
        digest(&weights),
        digest(Path::new(tokens)),
    ] {
        for path in [
            compiled.join("run-manifest.json"),
            keys.join("run-manifest.json"),
        ] {
            assert!(!fs::read_to_string(&path).unwrap().contains(&secret));
        }
    }
}

#[test]
fn a_failed_run_records_its_error() {
    let dir = scratch("failed");
    let path = dir.join("elsewhere.json");
    let missing = dir.join("missing.d");
    let out = dir.join("out");
    let output = tessera(&[
        "--run-manifest",
        path.to_str().unwrap(),
        "compile",
        "--model",
        missing.to_str().unwrap(),
        "--params",
        "test-small",
        "--out-dir",
        out.to_str().unwrap(),
    ]);
    assert_eq!(output.status.code(), Some(20));
    let value = manifest(&path);
    assert_eq!(value["exit_code"], 20);
    assert_eq!(value["error_code"], "ERR-RES-IO");
    assert_eq!(value["outputs"], serde_json::json!([]));
    // A command outside 05 §6's list writes none.
    let inspect = tessera(&[
        "--run-manifest",
        dir.join("none.json").to_str().unwrap(),
        "params",
        "check",
        "--params",
        "test-small",
    ]);
    assert!(inspect.status.code().is_some());
    assert!(!dir.join("none.json").exists());
}
