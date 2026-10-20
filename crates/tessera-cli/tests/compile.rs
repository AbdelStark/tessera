// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tessera compile`, end to end through the built binary (02 §2.2, RFC-0006 'Outputs', RFC-0029):
//!
//! - **Determinism and golden conformance artifacts (INV-IR-4).** Compiling each conformance model
//!   under `test-small` writes `claims.tcl`, `plan.tep` and `commitment-report.json` whose sealed
//!   bytes equal the golden vectors in `spec/vectors/claims/claim-lists.json` and
//!   `spec/vectors/codec/execution-plan.json`. Compiling a second time yields identical bytes.
//! - **JSON Schema and `--json`.** `commitment-report.json` carries `"schema":
//!   "tessera.commitment-report/1"` and every required section of
//!   `spec/schemas/commitment-report-v1.schema.json`. With `--json`, stdout holds the same report.
//! - **`tessera bench model --plan`.** Pricing a compiled `plan.tep` produces the cost-model
//!   projections embedded in `commitment-report.json`, and rejects a tampered plan (`ERR-USAGE`).
//! - **Adversarial refusals.** An invalid model weight (`-128`, `i8::MIN`) exits 65
//!   (`ERR-QUANT-ARCH`) without writing any output artifact; an unwritable output directory exits
//!   20 (`ERR-RES-IO`).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;
use tessera_codec::{DecodeLimits, envelope, schema};
use tessera_compiler::ExecutionPlan;
use tessera_ir::IntegerModel;

fn tessera<S: AsRef<std::ffi::OsStr>>(args: &[S]) -> Output {
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

fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tessera-compile-{test}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

fn hex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).expect("hex"))
        .collect()
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&fs::read_to_string(path).expect("file exists")).expect("valid JSON")
}

#[test]
fn compile_emits_deterministic_golden_artifacts_and_valid_report() {
    let dir = scratch("golden");
    let vectors = root().join("spec/vectors");
    let golden_plans = read_json(&vectors.join("codec/execution-plan.json"));
    let golden_claims = read_json(&vectors.join("claims/claim-lists.json"));
    let report_schema = read_json(&root().join("spec/schemas/commitment-report-v1.schema.json"));
    let required_keys: Vec<&str> = report_schema["required"]
        .as_array()
        .expect("required keys")
        .iter()
        .map(|v| v.as_str().expect("string"))
        .collect();

    let cases = golden_plans["cases"].as_array().expect("cases");
    assert_eq!(cases.len(), 4);

    for case in cases {
        let name = case["name"].as_str().expect("name");
        let model_file = case["model"].as_str().expect("model file");
        let model_path = vectors.join("models").join(model_file);
        let out_a = dir.join(format!("{name}-a"));
        let out_b = dir.join(format!("{name}-b"));

        let run_a = tessera(&[
            "compile",
            "--model",
            model_path.to_str().unwrap(),
            "--params",
            "test-small",
            "--out-dir",
            out_a.to_str().unwrap(),
            "--json",
        ]);
        assert_eq!(run_a.status.code(), Some(0), "{name}: {}", stderr(&run_a));
        assert!(stderr(&run_a).contains("wrote claims.tcl, plan.tep, commitment-report.json"));

        let run_b = tessera(&[
            "compile",
            "--model",
            model_path.to_str().unwrap(),
            "--params",
            "test-small",
            "--out-dir",
            out_b.to_str().unwrap(),
        ]);
        assert_eq!(run_b.status.code(), Some(0), "{name}: {}", stderr(&run_b));
        assert_eq!(stdout(&run_b), "");

        // INV-IR-4: running twice yields identical output bytes.
        for file in ["claims.tcl", "plan.tep", "commitment-report.json"] {
            assert_eq!(
                fs::read(out_a.join(file)).expect(file),
                fs::read(out_b.join(file)).expect(file),
                "{name}: {file} must be deterministic"
            );
        }

        // Golden claim list and execution plan bytes match spec/vectors/.
        let expected_plan = hex(case["hex"].as_str().expect("plan hex"));
        assert_eq!(
            fs::read(out_a.join("plan.tep")).unwrap(),
            expected_plan,
            "{name}: plan.tep"
        );
        let expected_claims = golden_claims["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"].as_str() == Some(name))
            .and_then(|c| c["hex"].as_str())
            .map(hex)
            .expect("golden claim list");
        assert_eq!(
            fs::read(out_a.join("claims.tcl")).unwrap(),
            expected_claims,
            "{name}: claims.tcl"
        );

        // --json stdout matches commitment-report.json and satisfies the schema's required structure.
        let report_file_text = fs::read_to_string(out_a.join("commitment-report.json")).unwrap();
        assert_eq!(stdout(&run_a), report_file_text, "{name}: --json output");
        let report: Value = serde_json::from_str(&report_file_text).expect("valid JSON report");
        assert_eq!(report["schema"], "tessera.commitment-report/1");
        let obj = report.as_object().expect("report object");
        for &key in &required_keys {
            assert!(obj.contains_key(key), "{name}: missing required key {key}");
        }
        assert_eq!(obj.len(), required_keys.len(), "{name}: no extra keys");
    }

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn bench_model_reads_compiled_plan_and_refuses_tampered_plan() {
    let dir = scratch("bench-plan");
    let model_path = root().join("spec/vectors/models/tiny-2-servable.d");
    let out_dir = dir.join("compiled");

    let compiled = tessera(&[
        "compile",
        "--model",
        model_path.to_str().unwrap(),
        "--params",
        "live-64",
        "--out-dir",
        out_dir.to_str().unwrap(),
    ]);
    assert_eq!(compiled.status.code(), Some(0), "{}", stderr(&compiled));

    let plan_path = out_dir.join("plan.tep");
    let bench = tessera(&[
        "bench",
        "model",
        "--plan",
        plan_path.to_str().unwrap(),
        "--context",
        "1024,4096,16384,131072",
        "--json",
    ]);
    assert_eq!(bench.status.code(), Some(0), "{}", stderr(&bench));
    let bench_report: Value = serde_json::from_str(stdout(&bench)).expect("JSON");
    let commit_report = read_json(&out_dir.join("commitment-report.json"));
    assert_eq!(bench_report["schema"], "tessera.cost-model/1");
    assert_eq!(bench_report["inputs"]["source_defaults"], false);
    assert_eq!(bench_report["projections"], commit_report["projections"]);

    // Tamper with the plan's tiles and reseal: bench model --plan must refuse with ERR-USAGE (64).
    let plan_bytes = fs::read(&plan_path).unwrap();
    let mut plan: ExecutionPlan =
        envelope::open(&plan_bytes, &schema::EXECUTION_PLAN, DecodeLimits::MESSAGE).unwrap();
    plan.tiles.pop();
    let tampered_path = dir.join("tampered.tep");
    fs::write(
        &tampered_path,
        envelope::seal(&schema::EXECUTION_PLAN, &plan),
    )
    .unwrap();
    let refused = tessera(&[
        "bench",
        "model",
        "--plan",
        tampered_path.to_str().unwrap(),
        "--json",
    ]);
    assert_eq!(refused.status.code(), Some(64));
    let err: Value = serde_json::from_str(stdout(&refused)).expect("JSON error");
    assert_eq!(err["error"]["code"], "ERR-USAGE");

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn invalid_model_weights_exit_65_and_unwritable_out_dir_exits_20() {
    let dir = scratch("errors");
    let model_path = root().join("spec/vectors/models/tiny-2-servable.d");
    let model_bytes = fs::read(&model_path).unwrap();
    let mut model: IntegerModel = envelope::open(
        &model_bytes,
        &schema::INTEGER_MODEL,
        DecodeLimits::INTEGER_MODEL,
    )
    .unwrap();
    // Weight entry -128 (i8::MIN) violates RFC-0005's symmetric [-127, 127] range.
    model.weights[0].data[0] = i8::MIN;
    let bad_model_path = dir.join("bad-weight.d");
    fs::write(
        &bad_model_path,
        envelope::seal(&schema::INTEGER_MODEL, &model),
    )
    .unwrap();

    let out_dir = dir.join("out");
    let refused = tessera(&[
        "compile",
        "--model",
        bad_model_path.to_str().unwrap(),
        "--params",
        "test-small",
        "--out-dir",
        out_dir.to_str().unwrap(),
        "--json",
    ]);
    assert_eq!(refused.status.code(), Some(65));
    let err: Value = serde_json::from_str(stdout(&refused)).expect("JSON error");
    assert_eq!(err["command"], "compile");
    assert_eq!(err["error"]["code"], "ERR-QUANT-ARCH");
    assert!(!out_dir.exists(), "failed compile must not create out-dir");

    // Unwritable out-dir (pointing inside an existing regular file) exits 20 with ERR-RES-IO.
    let blocker = dir.join("blocker");
    fs::write(&blocker, b"not a directory").unwrap();
    let blocked_out = blocker.join("sub");
    let io_fail = tessera(&[
        "compile",
        "--model",
        model_path.to_str().unwrap(),
        "--params",
        "test-small",
        "--out-dir",
        blocked_out.to_str().unwrap(),
        "--json",
    ]);
    assert_eq!(io_fail.status.code(), Some(20));
    let err: Value = serde_json::from_str(stdout(&io_fail)).expect("JSON error");
    assert_eq!(err["command"], "compile");
    assert_eq!(err["error"]["code"], "ERR-RES-IO");

    let _ = fs::remove_dir_all(dir);
}
