// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Integration tests for `tessera commit-weights` (`02 §2.2`, `RFC-0012`, `RFC-0040`, `RFC-0041`,
//! `RFC-0048`).
//!
//! - **Golden vectors (`RFC-0040`).** Every conformance model (`tiny-2-servable`,
//!   `tiny-2-minimal`, `tiny-4-servable`, `tiny-2-affine`) committed under its `test-small` plan
//!   prints the oracle's `R_W` (`spec/vectors/weights/weights.json`), writes `weights.twr` byte
//!   for byte equal to `spec/vectors/weights/<name>.twr` with the default file mode, and writes
//!   `weights.twc` with mode `0600` (on Unix) and an untouched budget.
//! - **Determinism & `--json`.** Re-running yields the same `R_W` and `.twr` bytes, and with the
//!   same mask seed the same `.twc` bytes; `--json` prints one object with `command`,
//!   `weights_root`, `epoch`, `out`, `public` and the ten matrix `roots`.
//! - **Adversarial checks (`RFC-0041`, `RFC-0048`, `02 §2.4`).** A plan from another model or with
//!   tampered claims/tiles/layout, or `--out` ending in `.twr`, exits `64` (`ERR-USAGE`). A model
//!   whose weights (`-128`), channel classes (`>= n`), affine multipliers (`0`) or layer count
//!   (`0`) leave their domains exits `65` (`ERR-QUANT-ARCH`). An unwritable output path exits `20`
//!   (`ERR-RES-IO`).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;
use tessera_codec::{DecodeLimits, envelope, schema};
use tessera_compiler::{ExecutionPlan, PublicModel, analyze, claim_list, lower, plan, select};
use tessera_ir::{GadgetPolicy, IntegerModel};
use tessera_pcs::{WeightCommitmentPublic, WeightCommitmentState};

/// Offset of the 32-byte `masks_seed` inside a `0x0302` envelope: 20-byte header + 334-byte
/// `WeightCommitmentPublic`.
const SEED_AT: usize = 20 + 334;

fn tessera<S: AsRef<std::ffi::OsStr>>(args: &[S]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tessera"))
        .args(args)
        .output()
        .expect("the tessera binary runs")
}

fn text(bytes: &[u8]) -> &str {
    std::str::from_utf8(bytes).expect("UTF-8")
}

fn vectors() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors")
}

fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "tessera-commit-weights-{test}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

fn load_model(name: &str) -> IntegerModel {
    let bytes = fs::read(vectors().join("models").join(name)).expect("model vector");
    envelope::open(&bytes, &schema::INTEGER_MODEL, DecodeLimits::INTEGER_MODEL)
        .expect("valid IntegerModel")
}

fn load_weights_vectors() -> Value {
    let text = fs::read_to_string(vectors().join("weights/weights.json")).expect("weights.json");
    serde_json::from_str(&text).expect("valid JSON")
}

fn build_plan_for(model: &IntegerModel) -> ExecutionPlan {
    let public = PublicModel::of(model);
    let lowering = lower(public, 1).expect("lowers");
    let ranges = analyze(&lowering.graph, &lowering.tables).expect("ranges");
    let policy = GadgetPolicy::default();
    let selection = select(&lowering.graph, &ranges, &lowering.tables, &policy).expect("select");
    let params = tessera_protocol::ParamSet::TEST_SMALL;
    let claims = claim_list(
        public,
        &selection,
        &policy,
        params.layout,
        params.param_set_id(),
    )
    .expect("claims");
    plan(public, &claims).expect("plan")
}

#[cfg(unix)]
fn mode(path: &Path) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    fs::metadata(path).expect("metadata").permissions().mode() & 0o777
}

#[test]
fn commit_weights_reproduces_every_conformance_models_oracle_root_and_writes_0600_state() {
    let dir = scratch("conformance");
    let oracle = load_weights_vectors();
    let models = oracle["models"].as_array().expect("models");

    #[cfg(unix)]
    let default_mode = {
        let probe = dir.join("probe");
        fs::write(&probe, b"x").expect("probe");
        mode(&probe)
    };

    for entry in models {
        let Some(model_rel) = entry["model"].as_str() else {
            // The synthetic model has no IntegerModel artifact on disk.
            continue;
        };
        let name = entry["name"].as_str().expect("name");
        let model_path = vectors().join("weights").join(model_rel);
        let case_dir = dir.join(name);

        let compile_out = tessera(&[
            "compile",
            "--model",
            model_path.to_str().expect("UTF-8"),
            "--params",
            "test-small",
            "--out-dir",
            case_dir.to_str().expect("UTF-8"),
        ]);
        assert_eq!(compile_out.status.code(), Some(0), "{name}: compile");

        let plan_path = case_dir.join("plan.tep");
        let twc_path = case_dir.join("weights.twc");
        let twr_path = case_dir.join("weights.twr");

        let out = tessera(&[
            "commit-weights",
            "--model",
            model_path.to_str().expect("UTF-8"),
            "--plan",
            plan_path.to_str().expect("UTF-8"),
            "--out",
            twc_path.to_str().expect("UTF-8"),
        ]);
        assert_eq!(
            out.status.code(),
            Some(0),
            "{name} stderr: {}",
            text(&out.stderr)
        );
        assert!(
            text(&out.stderr).contains("tessera commit-weights: wrote"),
            "{name} stderr: {}",
            text(&out.stderr)
        );

        let expected_rw = entry["weights_root"].as_str().expect("weights_root");
        assert_eq!(text(&out.stdout), format!("{expected_rw}\n"), "{name}: R_W");

        let golden_twr = fs::read(
            vectors()
                .join("weights")
                .join(entry["file"].as_str().unwrap()),
        )
        .expect("golden .twr");
        let written_twr = fs::read(&twr_path).expect("written .twr");
        assert_eq!(written_twr, golden_twr, "{name}: .twr bytes");

        #[cfg(unix)]
        {
            assert_eq!(mode(&twc_path), 0o600, "{name}: .twc mode 0600");
            assert_eq!(mode(&twr_path), default_mode, "{name}: .twr default mode");
        }

        let public: WeightCommitmentPublic = envelope::open(
            &written_twr,
            &schema::WEIGHT_COMMITMENT_PUBLIC,
            DecodeLimits::MESSAGE,
        )
        .expect("valid .twr");
        let state = WeightCommitmentState::read(&twc_path).expect("valid .twc");
        assert_eq!(state.public, public, "{name}: state.public");
        assert_eq!(state.public.epoch, 0);
        assert!(
            state
                .budget
                .iter()
                .all(|b| b.opened_columns.columns().is_empty() && b.next_blinding_row == 0),
            "{name}: untouched budget"
        );
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn rerunning_yields_the_same_weights_root_and_json_reports_every_root() {
    let dir = scratch("rerun");
    let oracle = load_weights_vectors();
    let entry = oracle["models"]
        .as_array()
        .expect("models")
        .iter()
        .find(|m| m["name"].as_str() == Some("tiny-2-servable"))
        .expect("tiny-2-servable");
    let model_path = vectors().join("models/tiny-2-servable.d");
    let model = load_model("tiny-2-servable.d");
    let compiled_plan = build_plan_for(&model);
    let plan_path = dir.join("plan.tep");
    fs::write(
        &plan_path,
        envelope::seal(&schema::EXECUTION_PLAN, &compiled_plan),
    )
    .expect("plan.tep");

    let first_twc = dir.join("first.twc");
    let second_twc = dir.join("second.twc");
    let first = tessera(&[
        "commit-weights",
        "--model",
        model_path.to_str().expect("UTF-8"),
        "--plan",
        plan_path.to_str().expect("UTF-8"),
        "--out",
        first_twc.to_str().expect("UTF-8"),
    ]);
    let second = tessera(&[
        "--json",
        "commit-weights",
        "--model",
        model_path.to_str().expect("UTF-8"),
        "--plan",
        plan_path.to_str().expect("UTF-8"),
        "--out",
        second_twc.to_str().expect("UTF-8"),
    ]);
    assert_eq!(first.status.code(), Some(0));
    assert_eq!(second.status.code(), Some(0));

    let rw = text(&first.stdout).trim();
    let report: Value = serde_json::from_str(text(&second.stdout)).expect("JSON output");
    assert_eq!(report["command"], "commit-weights");
    assert_eq!(report["weights_root"].as_str(), Some(rw));
    assert_eq!(report["epoch"].as_u64(), Some(0));
    assert_eq!(
        report["out"].as_str(),
        Some(second_twc.to_str().expect("UTF-8"))
    );
    assert_eq!(
        report["public"].as_str(),
        Some(dir.join("second.twr").to_str().expect("UTF-8"))
    );
    let expected_roots: Vec<&str> = entry["matrices"]
        .as_array()
        .expect("matrices")
        .iter()
        .map(|m| m["root"].as_str().expect("root"))
        .collect();
    let reported_roots: Vec<&str> = report["roots"]
        .as_array()
        .expect("roots")
        .iter()
        .map(|r| r.as_str().expect("hex"))
        .collect();
    assert_eq!(reported_roots, expected_roots);

    assert_eq!(
        fs::read(dir.join("first.twr")).expect("first.twr"),
        fs::read(dir.join("second.twr")).expect("second.twr"),
    );

    // Apart from the 32-byte CSPRNG masks_seed (and the envelope digest at the end), both .twc
    // files are identical; and re-saving a state preserves its seed and bytes exactly.
    let a = fs::read(&first_twc).expect("first.twc");
    let b = fs::read(&second_twc).expect("second.twc");
    assert_eq!(a[..SEED_AT], b[..SEED_AT]);
    assert_eq!(a[SEED_AT + 32..a.len() - 32], b[SEED_AT + 32..b.len() - 32]);
    let state = WeightCommitmentState::read(&first_twc).expect("read first.twc");
    let re_saved = dir.join("resaved.twc");
    state.write(&re_saved).expect("write resaved.twc");
    assert_eq!(fs::read(&re_saved).expect("resaved.twc"), a);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn commit_weights_refuses_mismatched_plans_out_of_domain_models_and_bad_paths() {
    let dir = scratch("adversarial");
    let model_path = vectors().join("models/tiny-2-servable.d");
    let servable = load_model("tiny-2-servable.d");
    let minimal = load_model("tiny-2-minimal.d");
    let affine = load_model("tiny-2-affine.d");

    let honest_plan = build_plan_for(&servable);
    let plan_path = dir.join("plan.tep");
    fs::write(
        &plan_path,
        envelope::seal(&schema::EXECUTION_PLAN, &honest_plan),
    )
    .expect("plan.tep");

    // 1. --out ending in .twr is refused with exit 64 (ERR-USAGE).
    let twr_out = dir.join("weights.twr");
    let out = tessera(&[
        "--json",
        "commit-weights",
        "--model",
        model_path.to_str().expect("UTF-8"),
        "--plan",
        plan_path.to_str().expect("UTF-8"),
        "--out",
        twr_out.to_str().expect("UTF-8"),
    ]);
    assert_eq!(out.status.code(), Some(64));
    let err: Value = serde_json::from_str(text(&out.stdout)).expect("JSON");
    assert_eq!(err["error"]["code"], "ERR-USAGE");

    // 2. Plan from another model, or with tampered tiles or layout, exits 64 (ERR-USAGE).
    let mut bad_tiles = honest_plan.clone();
    bad_tiles.tiles.pop();
    let mut bad_layout = honest_plan.clone();
    bad_layout.claims.layout.weights.log2_k += 1;
    for (name, forged) in [
        ("other-model.tep", build_plan_for(&minimal)),
        ("bad-tiles.tep", bad_tiles),
        ("bad-layout.tep", bad_layout),
    ] {
        let p = dir.join(name);
        fs::write(&p, envelope::seal(&schema::EXECUTION_PLAN, &forged)).expect("write plan");
        let twc = dir.join("never.twc");
        let out = tessera(&[
            "--json",
            "commit-weights",
            "--model",
            model_path.to_str().expect("UTF-8"),
            "--plan",
            p.to_str().expect("UTF-8"),
            "--out",
            twc.to_str().expect("UTF-8"),
        ]);
        assert_eq!(out.status.code(), Some(64), "{name}");
        let err: Value = serde_json::from_str(text(&out.stdout)).expect("JSON");
        assert_eq!(err["error"]["code"], "ERR-USAGE", "{name}");
        assert!(!twc.exists(), "{name}: nothing written");
    }

    // 3. RFC-0048 domain checks: weight -128, class >= count, multiplier 0, n_layers == 0 all
    // exit 65 (ERR-QUANT-ARCH).
    let mut weight_min = servable.clone();
    weight_min.weights[0].data[0] = i8::MIN;
    let mut class_oob = servable.clone();
    class_oob.quant.channels[0].class[0] = 99;
    let mut mult_zero = affine.clone();
    mult_zero.quant.channels[0].multiplier[0] = 0;
    let mut zero_layers = servable.clone();
    zero_layers.arch.n_layers = 0;

    let affine_plan_path = dir.join("affine.tep");
    fs::write(
        &affine_plan_path,
        envelope::seal(&schema::EXECUTION_PLAN, &build_plan_for(&affine)),
    )
    .expect("affine.tep");

    for (name, bad_model, plan_file) in [
        ("weight-min.d", weight_min, &plan_path),
        ("class-oob.d", class_oob, &plan_path),
        ("mult-zero.d", mult_zero, &affine_plan_path),
        ("zero-layers.d", zero_layers, &plan_path),
    ] {
        let m = dir.join(name);
        fs::write(&m, envelope::seal(&schema::INTEGER_MODEL, &bad_model)).expect("write model");
        let twc = dir.join("never.twc");
        let out = tessera(&[
            "--json",
            "commit-weights",
            "--model",
            m.to_str().expect("UTF-8"),
            "--plan",
            plan_file.to_str().expect("UTF-8"),
            "--out",
            twc.to_str().expect("UTF-8"),
        ]);
        assert_eq!(out.status.code(), Some(65), "{name}");
        let err: Value = serde_json::from_str(text(&out.stdout)).expect("JSON");
        assert_eq!(err["error"]["code"], "ERR-QUANT-ARCH", "{name}");
        assert!(!twc.exists(), "{name}: nothing written");
    }

    // 4. Unwritable output path exits 20 (ERR-RES-IO).
    let blocker = dir.join("not-a-dir");
    fs::write(&blocker, b"file").expect("blocker");
    let bad_out = blocker.join("weights.twc");
    let out = tessera(&[
        "--json",
        "commit-weights",
        "--model",
        model_path.to_str().expect("UTF-8"),
        "--plan",
        plan_path.to_str().expect("UTF-8"),
        "--out",
        bad_out.to_str().expect("UTF-8"),
    ]);
    assert_eq!(out.status.code(), Some(20));
    let err: Value = serde_json::from_str(text(&out.stdout)).expect("JSON");
    assert_eq!(err["error"]["code"], "ERR-RES-IO");

    let _ = fs::remove_dir_all(&dir);
}
