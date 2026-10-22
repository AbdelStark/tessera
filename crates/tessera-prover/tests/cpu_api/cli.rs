// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Cross-workspace CLI acceptance stays on the producer side (DEP-2).
use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

// Reuse the repository gate's schema engine; no JSON dependency enters the verifier graph.
use xtask::reports as report_schema;

struct Dir(PathBuf);
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn schema(report: &str) -> serde_json::Value {
    let value: serde_json::Value = serde_json::from_str(report).unwrap();
    let schema: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../spec/schemas/verdict-v1.schema.json"
    ))
    .unwrap();
    assert!(report_schema::schema_problems(&schema).is_empty());
    let errors = report_schema::validate(&schema, &schema, &value, "");
    assert!(errors.is_empty(), "{errors:?}");
    value
}
pub(super) fn check(
    plan: &tessera_compiler::ExecutionPlan,
    batch: &ProofBatch,
    weights: &WeightCommitmentPublic,
    proof: &protocol::NiProof,
) {
    check_with(plan, batch, weights, proof, None);
}
fn check_with(
    plan: &tessera_compiler::ExecutionPlan,
    batch: &ProofBatch,
    weights: &WeightCommitmentPublic,
    proof: &protocol::NiProof,
    binary: Option<&Path>,
) {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = Dir(std::env::temp_dir().join(format!(
        "tessera-public-cli-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )));
    fs::create_dir(&dir.0).unwrap();
    let claims = dir.0.join("claims.tcl");
    let epoch = dir.0.join("weights.twr");
    let ni = dir.0.join("proof.tnp");
    let out = dir.0.join("verdict.json");
    fs::write(&claims, envelope::seal(&schema::CLAIM_LIST, &plan.claims)).unwrap();
    fs::write(
        &epoch,
        envelope::seal(&schema::WEIGHT_COMMITMENT_PUBLIC, weights),
    )
    .unwrap();
    let write_proof = |proof: &protocol::NiProof| {
        fs::write(&ni, envelope::seal(&schema::NI_PROOF, proof)).unwrap()
    };
    write_proof(proof);
    let hex = |bytes: [u8; 32]| bytes.iter().map(|b| format!("{b:02x}")).collect::<String>();
    let args = vec![
        "verify".into(),
        "--claims".into(),
        claims.display().to_string(),
        "--weights".into(),
        epoch.display().to_string(),
        "--weights-root".into(),
        hex(batch.statement.weights_root),
        "--statement-digest".into(),
        hex(batch.statement.digest()),
        "--proof".into(),
        ni.display().to_string(),
        "--bound".into(),
        "0".into(),
        "--out".into(),
        out.display().to_string(),
        "--json".into(),
    ];
    let invoke = |args: &[String]| match binary {
        None => tv_cli::run(args),
        Some(binary) => {
            let out = std::process::Command::new(binary)
                .args(args)
                .output()
                .unwrap();
            tv_cli::CommandOutput {
                exit_code: out.status.code().unwrap(),
                stdout: String::from_utf8(out.stdout).unwrap(),
                stderr: String::from_utf8(out.stderr).unwrap(),
            }
        }
    };
    let result = invoke(&args);
    let report = schema(&result.stdout);
    let accepts =
        tv_core::params::ParamSet::accept(&independent(plan, batch, weights).0.params).is_some();
    assert_eq!(
        result.exit_code,
        if accepts { 0 } else { 10 },
        "{}",
        result.stdout
    );
    assert_eq!(fs::read_to_string(&out).unwrap(), result.stdout);
    assert_eq!(report["execution"]["statement_pinned"], true);
    let catalog = invoke(&[
        "tables".into(),
        "--claims".into(),
        claims.display().to_string(),
        "--json".into(),
    ]);
    assert_eq!(catalog.exit_code, 0, "{}", catalog.stdout);
    let value: serde_json::Value = serde_json::from_str(&catalog.stdout).unwrap();
    let catalog_schema: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../spec/schemas/table-digests-v1.schema.json"
    ))
    .unwrap();
    assert!(report_schema::schema_problems(&catalog_schema).is_empty());
    assert!(report_schema::validate(&catalog_schema, &catalog_schema, &value, "").is_empty());
    assert_eq!(
        value["tables"].as_array().unwrap().len(),
        plan.claims.tables.len()
    );
    for (reported, expected) in value["tables"]
        .as_array()
        .unwrap()
        .iter()
        .zip(&plan.claims.tables)
    {
        assert_eq!(reported["id"], expected.id.0);
        assert_eq!(reported["log2_rows"], expected.log2_rows);
        assert_eq!(reported["digest"], hex(expected.digest));
    }
    let mut bad_claims = plan.claims.clone();
    bad_claims.tables[0].digest[0] ^= 1;
    fs::write(&claims, envelope::seal(&schema::CLAIM_LIST, &bad_claims)).unwrap();
    let bad = invoke(&[
        "tables".into(),
        "--claims".into(),
        claims.display().to_string(),
        "--json".into(),
    ]);
    assert_eq!(bad.exit_code, 10);
    let bad = schema(&bad.stdout);
    assert_eq!(bad["verdict"]["reason"]["code"], "REJ-STATEMENT");
    let table = GeneratedTable::new(
        tessera_tables::TableId(99),
        tessera_tables::TableKind::Range { bits: 1 },
    )
    .unwrap();
    bad_claims.tables[0] = tessera_tables::Table::spec(&table).clone();
    fs::write(&claims, envelope::seal(&schema::CLAIM_LIST, &bad_claims)).unwrap();
    let bad = invoke(&[
        "tables".into(),
        "--claims".into(),
        claims.display().to_string(),
        "--json".into(),
    ]);
    assert_eq!(bad.exit_code, 10);
    let bad = schema(&bad.stdout);
    assert_eq!(bad["verdict"]["reason"]["code"], "REJ-STATEMENT");
    assert_eq!(bad["verdict"]["reason"]["field"], "Table { table: 99 }");
    fs::write(&claims, envelope::seal(&schema::CLAIM_LIST, &plan.claims)).unwrap();
    let capped = invoke(&[
        "tables".into(),
        "--claims".into(),
        claims.display().to_string(),
        "--max-table-rows".into(),
        "0".into(),
        "--json".into(),
    ]);
    assert_eq!(capped.exit_code, 20);
    let capped: serde_json::Value = serde_json::from_str(&capped.stdout).unwrap();
    assert_eq!(capped["error"]["code"], "ERR-RES-TIME");
    if !accepts {
        return;
    }
    assert_eq!(report["verdict"]["status"], "accept");
    assert_eq!(report["execution"]["exit_code"], 0);
    let phases = report["execution"]["trace"].as_array().unwrap();
    assert_eq!(phases.len(), 3);
    for (phase, name) in phases.iter().zip(["load", "statement_pin", "verification"]) {
        assert_eq!(phase["phase"], name);
        assert!(phase["duration_ns"].as_u64().is_some());
    }
    assert_eq!(phases[2]["outcome"], "accept");
    assert_eq!(report["test_parameters"], true);
    assert_eq!(report["anchor_verified"], false);
    assert_eq!(report["ledger_updated"], false);
    assert_eq!(report["zero_knowledge"], false);
    assert!(
        report["checks"]
            .as_array()
            .unwrap()
            .iter()
            .all(|c| c["duration_ns"].is_null())
    );
    let mut bad = proof.clone();
    bad.round2.advice_combined[0].u[0] += F::ONE;
    write_proof(&bad);
    let rejected = invoke(&args);
    assert_eq!(rejected.exit_code, 10);
    let rejected = schema(&rejected.stdout);
    assert_eq!(rejected["verdict"]["reason"]["code"], "REJ-PCS-EVAL");
    write_proof(proof);
    for (key, value, code) in [
        ("--statement-digest", "ab".repeat(32), 64),
        ("--weights-root", "ab".repeat(32), 64),
        ("--out", dir.0.display().to_string(), 20),
    ] {
        let mut wrong = args.clone();
        let i = wrong.iter().position(|a| a == key).unwrap();
        wrong[i + 1] = value;
        let result = invoke(&wrong);
        assert_eq!(result.exit_code, code, "{key}: {}", result.stdout);
        let report = schema(&result.stdout);
        assert_eq!(report["execution"]["exit_code"], code);
        assert_eq!(
            report["verdict"]["error"]["code"],
            if code == 64 {
                "ERR-USAGE"
            } else {
                "ERR-RES-IO"
            }
        );
        assert_eq!(report["execution"]["statement_pinned"], code != 64);
    }
}

#[test]
#[ignore = "requires a separately built verifier binary; required CI runs this explicitly"]
fn public_cli_binary_accepts_cpu_proof_and_rejects_tampering() {
    let binary = std::env::var_os("TESSERA_VERIFY_BIN")
        .expect("point to the independent verifier-workspace build");
    let model = bounded_model();
    let (plan, batch, weights) = fixture(&model, 3, 1);
    let mut state = WeightCommitmentState::new(weights).unwrap();
    let mut prover = Prover::new(&plan, &model, &mut state, cfg(Mode::NonInteractive)).unwrap();
    let proof = prover.prove_non_interactive(&batch).unwrap();
    check_with(&plan, &batch, &weights, &proof, Some(Path::new(&binary)));
}
