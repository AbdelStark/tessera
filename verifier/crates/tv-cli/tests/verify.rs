// SPDX-License-Identifier: Apache-2.0 OR MIT

//! CLI argument, resource, syntax and output-preservation boundaries.

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use tv_testkit::Json;

struct Dir(PathBuf);
impl Dir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "tessera-cli-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn args(&self) -> Vec<String> {
        let f = self.0.join("input.bin");
        let weights = self.0.join("weights.bin");
        let proof = self.0.join("proof.bin");
        fs::write(&f, b"public fixture canary").unwrap();
        fs::write(&weights, b"public fixture canary").unwrap();
        fs::write(&proof, b"public fixture canary").unwrap();
        vec![
            "verify".into(),
            "--claims".into(),
            f.display().to_string(),
            "--weights".into(),
            weights.display().to_string(),
            "--proof".into(),
            proof.display().to_string(),
            "--weights-root".into(),
            "00".repeat(32),
            "--statement-digest".into(),
            "00".repeat(32),
            "--bound".into(),
            "0".into(),
            "--json".into(),
        ]
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn binary(args: &[String]) -> (i32, Json) {
    let output = Command::new(env!("CARGO_BIN_EXE_tessera-verify"))
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    (
        output.status.code().unwrap(),
        Json::parse(std::str::from_utf8(&output.stdout).unwrap()).unwrap(),
    )
}
#[test]
fn output_cannot_replace_any_input_even_when_validation_fails() {
    let dir = Dir::new();
    for name in ["input.bin", "weights.bin", "proof.bin"] {
        let mut args = dir.args();
        let input = dir.0.join(name);
        args.extend(["--out".into(), input.display().to_string()]);
        let (code, report) = binary(&args);
        assert_eq!(code, 64);
        assert_eq!(error(&report), "ERR-USAGE");
        assert_eq!(fs::read(&input).unwrap(), b"public fixture canary");
    }
}
fn error(report: &Json) -> &str {
    report
        .get("verdict")
        .and_then(|v| v.get("error"))
        .or_else(|| report.get("error"))
        .unwrap()
        .get("code")
        .unwrap()
        .as_str()
        .unwrap()
}
#[test]
fn syntax_io_policy_and_malformed_inputs_have_distinct_exits() {
    let dir = Dir::new();
    let args = dir.args();
    assert_eq!(binary(&args).0, 65);
    for (name, value, code, reason) in [
        ("--timeout-ms", "0", 20, "ERR-RES-TIME"),
        ("--timeout-ms", "+1", 64, "ERR-USAGE"),
        ("--max-memory-bytes", "0", 20, "ERR-RES-MEMORY"),
        ("--max-artifact-bytes", "1", 65, "ERR-CODEC-LIMIT"),
        ("--unknown", "1", 64, "ERR-USAGE"),
        ("--bound", "0.1", 64, "ERR-USAGE"),
        ("--statement-digest", "ab", 64, "ERR-USAGE"),
    ] {
        let mut test = args.clone();
        if let Some(i) = test.iter().position(|a| a == name) {
            test[i + 1] = value.into();
        } else {
            test.extend([name.into(), value.into()]);
        }
        let (actual, report) = binary(&test);
        assert_eq!(actual, code, "{name}");
        assert_eq!(error(&report), reason, "{name}");
    }
    let mut missing = args.clone();
    let i = missing.iter().position(|a| a == "--claims").unwrap();
    missing[i + 1] = dir.0.join("missing").display().to_string();
    let (code, report) = binary(&missing);
    assert_eq!(code, 20);
    assert_eq!(error(&report), "ERR-RES-IO");
    missing[i + 1] = dir.0.display().to_string();
    let (code, report) = binary(&missing);
    assert_eq!(code, 20);
    assert_eq!(error(&report), "ERR-RES-IO");
}

#[test]
fn report_replaces_only_the_requested_output_and_leaves_no_scratch_file() {
    let dir = Dir::new();
    let mut args = dir.args();
    let output = dir.0.join("verdict.json");
    fs::write(&output, b"previous report").unwrap();
    args.extend(["--out".into(), output.display().to_string()]);
    assert_eq!(binary(&args).0, 65);
    Json::parse(&fs::read_to_string(&output).unwrap()).unwrap();
    assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 5);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            fs::metadata(&output).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
    let blocked = dir.0.join("blocked");
    fs::create_dir(&blocked).unwrap();
    let at = args.iter().position(|a| a == "--out").unwrap();
    args[at + 1] = blocked.display().to_string();
    let (code, report) = binary(&args);
    assert_eq!(code, 20);
    assert_eq!(error(&report), "ERR-RES-IO");
    assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 6);
}

#[test]
fn duplicate_options_and_unimplemented_capabilities_never_accept() {
    let dir = Dir::new();
    let mut args = dir.args();
    args.extend(["--bound".into(), "1".into()]);
    assert_eq!(binary(&args).0, 64);
    let mut repeated_json = dir.args();
    repeated_json.push("--json".into());
    assert_eq!(binary(&repeated_json).0, 64);
    let mut missing_value = dir.args();
    missing_value.extend(["--out".into(), "--json".into()]);
    assert_eq!(binary(&missing_value).0, 64);
    for name in [
        "--claims",
        "--weights",
        "--proof",
        "--weights-root",
        "--statement-digest",
        "--bound",
    ] {
        let mut missing = dir.args();
        let at = missing.iter().position(|a| a == name).unwrap();
        missing.drain(at..at + 2);
        assert_eq!(binary(&missing).0, 64, "{name}");
    }
    for command in [
        vec!["live"],
        vec!["weights", "link"],
        vec!["audit", "plan"],
        vec!["audit", "report"],
    ] {
        let mut args = command.iter().map(|s| String::from(*s)).collect::<Vec<_>>();
        args.push("--json".into());
        assert_eq!(binary(&args).0, 64);
    }
}

#[test]
fn oversized_argument_surfaces_are_refused() {
    for args in [
        vec!["--help".to_string(); 65],
        vec!["--help".into(), "x".repeat(8193)],
    ] {
        assert_eq!(tv_cli::run(args).exit_code, 64);
    }
    for mut args in [vec!["--help".to_string(); 65], vec!["x".repeat(8193)]] {
        args.push("--json".into());
        if args.len() > 65 {
            args[0] = "--json".into();
        }
        let output = tv_cli::run(args);
        assert_eq!(output.exit_code, 64);
        assert!(output.stderr.is_empty());
        let report = Json::parse(&output.stdout).unwrap();
        assert_eq!(error(&report), "ERR-USAGE");
    }
}

#[test]
fn help_in_an_artifact_argument_cannot_bypass_verification() {
    let dir = Dir::new();
    for key in ["--claims", "--weights", "--proof"] {
        for help in ["--help", "-h"] {
            let mut args = dir.args();
            let at = args.iter().position(|a| a == key).unwrap();
            args[at + 1] = help.into();
            args.extend([
                "--run-manifest".into(),
                dir.0.join("run-manifest.json").display().to_string(),
            ]);
            assert_ne!(tv_cli::run(&args).exit_code, 0, "{key} {help}");
        }
    }
    let mut args = dir.args();
    args.push("--help".into());
    assert_eq!(tv_cli::run(args).exit_code, 64);
    assert_eq!(tv_cli::run(["verify", "--help"]).exit_code, 0);
    for args in [
        ["params", "--params", "--help", "--json"],
        ["tables", "--claims", "--help", "--json"],
    ] {
        assert_eq!(tv_cli::run(args).exit_code, 64);
    }
}

#[cfg(unix)]
#[test]
fn fifo_input_is_refused_before_a_blocking_open() {
    let dir = Dir::new();
    let mut args = dir.args();
    let fifo = dir.0.join("fifo");
    assert!(
        Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap()
            .success()
    );
    args[2] = fifo.display().to_string();
    let mut child = Command::new(env!("CARGO_BIN_EXE_tessera-verify"))
        .args(&args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert_eq!(status.code(), Some(20));
            break;
        }
        if std::time::Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("opened a FIFO before rejecting it");
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

#[cfg(unix)]
#[test]
fn binary_handles_non_utf8_and_closed_stdout_without_panicking() {
    use std::os::{
        fd::OwnedFd,
        unix::{ffi::OsStringExt, net::UnixStream},
    };
    let output = Command::new(env!("CARGO_BIN_EXE_tessera-verify"))
        .arg(std::ffi::OsString::from_vec(vec![0xff]))
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(64));
    assert!(!String::from_utf8_lossy(&output.stderr).contains("panic"));
    let (reader, writer) = UnixStream::pair().unwrap();
    drop(reader);
    let writer: OwnedFd = writer.into();
    let output = Command::new(env!("CARGO_BIN_EXE_tessera-verify"))
        .arg("--help")
        .stdout(std::process::Stdio::from(writer))
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(20));
    assert!(output.stderr.is_empty());
}

#[test]
fn panic_boundary_returns_internal_without_payload() {
    let args = std::iter::once_with(|| -> String { panic!("private-panic-canary") });
    let output = tv_cli::run(args);
    assert_eq!(output.exit_code, 70);
    assert!(!output.stdout.contains("private-panic-canary"));
    assert!(output.stderr.is_empty());
    let json = Json::parse(&output.stdout).unwrap();
    assert_eq!(
        json.get("verdict")
            .unwrap()
            .get("error")
            .unwrap()
            .get("code")
            .unwrap()
            .as_str(),
        Some("ERR-INTERNAL")
    );
}

#[test]
fn argument_errors_escape_every_json_control_character() {
    for code in 0..32 {
        let command = format!("unknown-{}", char::from(code));
        let output = tv_cli::run([command.as_str(), "--json"]);
        assert_eq!(output.exit_code, 64);
        assert!(
            output
                .stdout
                .trim_end_matches('\n')
                .chars()
                .all(|c| c >= ' '),
            "unescaped control {code}: {:?}",
            output.stdout
        );
        assert!(
            Json::parse(&output.stdout).is_ok(),
            "control {code}: {:?}",
            output.stdout
        );
    }
}

#[cfg(unix)]
#[test]
fn symlink_output_alias_is_refused_and_inputs_are_preserved() {
    let dir = Dir::new();
    let mut args = dir.args();
    let original = dir.0.join("input.bin");
    let alias = dir.0.join("alias");
    std::os::unix::fs::symlink(&original, &alias).unwrap();
    args.extend(["--out".into(), alias.display().to_string()]);
    assert_eq!(binary(&args).0, 64);
    assert_eq!(fs::read(&original).unwrap(), b"public fixture canary");
    assert!(alias.is_symlink());
}

#[path = "support/schema.rs"]
mod schema;

fn fixture(dir: &Dir) -> Vec<String> {
    let mut args = dir.args();
    args.retain(|a| a != "--json");
    for (key, name) in [
        ("--claims", "claims.tcl"),
        ("--weights", "weights.twr"),
        ("--proof", "ni.tnp"),
    ] {
        let at = args.iter().position(|a| a == key).unwrap();
        fs::copy(
            tv_testkit::vectors_dir().join("tamper").join(name),
            &args[at + 1],
        )
        .unwrap();
    }
    let proof = tv_core::protocol::NiProof::open(
        &fs::read(&args[6]).unwrap(),
        &tv_codec::DecodeLimits::MESSAGE,
    )
    .unwrap();
    let root = args.iter().position(|a| a == "--weights-root").unwrap();
    args[root + 1] = proof
        .round1
        .statement
        .weights_root
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let pin = args.iter().position(|a| a == "--statement-digest").unwrap();
    args[pin + 1] = tv_testkit::read_vectors("tamper/protocol-1.json")
        .get("base")
        .unwrap()
        .get("statement_digest")
        .unwrap()
        .as_str()
        .unwrap()
        .into();
    args
}
fn raw(dir: &Dir, args: &[String]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_tessera-verify"))
        .current_dir(&dir.0)
        .args(args)
        .output()
        .unwrap()
}
fn events(output: &std::process::Output) -> Vec<Json> {
    let lines = std::str::from_utf8(&output.stderr).unwrap();
    assert!(!lines.is_empty());
    lines
        .lines()
        .map(|line| schema::check("log-event", line))
        .collect()
}
fn accepted_build() -> bool {
    tv_core::params::ParamSet::ALL
        .iter()
        .any(|p| p.name() == "test-small/1")
}
#[test]
fn log_formats_preserve_base_human_stdout_and_json_is_exclusive() {
    let dir = Dir::new();
    let args = fixture(&dir);
    let baseline = raw(&dir, &args);
    // Captured from the base binary with the recorded tiny proof, before this change.
    let expected = if accepted_build() {
        "Accept (test-small/1; test parameters certify nothing)\n"
    } else {
        "Reject (REJ-PARAMS)\n"
    };
    assert_eq!(baseline.stdout, expected.as_bytes());
    assert!(
        std::str::from_utf8(&baseline.stderr)
            .unwrap()
            .starts_with("verify: completed in ")
    );
    let mut human = args.clone();
    human.extend(["--log-format".into(), "human".into()]);
    let human = raw(&dir, &human);
    assert_eq!(human.stdout, baseline.stdout);
    assert!(human.stderr.starts_with(b"verify: completed in "));
    assert_eq!(
        std::str::from_utf8(&human.stderr).unwrap().lines().count(),
        1
    );
    assert!(
        human
            .stderr
            .ends_with(b" ns; protocol/1, anchor/ledger/ZK unchecked\n")
    );
    assert!(Json::parse(std::str::from_utf8(&human.stderr).unwrap()).is_err());
    let mut json = args;
    json.extend(["--log-format".into(), "json".into()]);
    let json = raw(&dir, &json);
    assert_eq!(json.stdout, baseline.stdout);
    let lines = events(&json);
    assert_eq!(
        lines.first().unwrap().get("event").unwrap().as_str(),
        Some("tessera.verifier.verify_started")
    );
    assert_eq!(
        lines.last().unwrap().get("event").unwrap().as_str(),
        Some("tessera.verifier.verify_finished")
    );
    let proof_bytes = fs::read(dir.0.join("proof.bin")).unwrap();
    let digest: String = proof_bytes[proof_bytes.len() - 32..]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let mut kinds = Vec::new();
    let mut active = Vec::new();
    for event in &lines {
        let name = event.get("event").unwrap().as_str().unwrap();
        let Json::Object(values) = event.get("fields").unwrap() else {
            panic!()
        };
        let expected_level = match name {
            "tessera.verifier.verify_started" | "tessera.verifier.check_started" => {
                assert!(values.is_empty());
                "info"
            }
            "tessera.verifier.check_finished" => {
                assert_eq!(values.len(), 2);
                assert_eq!(
                    event
                        .get("fields")
                        .unwrap()
                        .get("outcome")
                        .unwrap()
                        .as_str(),
                    Some("completed")
                );
                if event
                    .get("fields")
                    .unwrap()
                    .get("outcome")
                    .unwrap()
                    .as_str()
                    == Some("error")
                {
                    "error"
                } else {
                    "info"
                }
            }
            "tessera.verifier.verify_finished" => {
                assert_eq!(values.len(), 1);
                assert_eq!(
                    event
                        .get("fields")
                        .unwrap()
                        .get("exit_code")
                        .unwrap()
                        .as_u64(),
                    Some(json.status.code().unwrap() as u64)
                );
                if json.status.success() {
                    "info"
                } else {
                    "error"
                }
            }
            _ => panic!("unexpected event {name}"),
        };
        assert_eq!(event.get("level").unwrap().as_str(), Some(expected_level));
        let spans = event.get("span_path").unwrap().as_array().unwrap();
        for check in &spans[1..] {
            let Json::Object(fields) = check.get("fields").unwrap() else {
                panic!()
            };
            assert_eq!(fields.len(), 1);
            assert_eq!(fields[0].0, "kind");
        }
        assert_eq!(spans[0].get("name").unwrap().as_str(), Some("verify"));
        let Json::Object(fields) = spans[0].get("fields").unwrap() else {
            panic!()
        };
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].0, "proof_digest");
        assert_eq!(fields[0].1.as_str(), Some(digest.as_str()));
        if spans.len() >= 2 {
            let check = spans.last().unwrap();
            assert_eq!(check.get("name").unwrap().as_str(), Some("check"));
            let kind = check
                .get("fields")
                .unwrap()
                .get("kind")
                .unwrap()
                .as_str()
                .unwrap();
            if event.get("event").unwrap().as_str() == Some("tessera.verifier.check_started") {
                active.push(kind);
            }
            if event.get("event").unwrap().as_str() == Some("tessera.verifier.check_finished") {
                assert_eq!(active.pop(), Some(kind));
                let kind = check
                    .get("fields")
                    .unwrap()
                    .get("kind")
                    .unwrap()
                    .as_str()
                    .unwrap();
                if !["load", "statement_pin", "verification"].contains(&kind) {
                    assert_eq!(spans.len(), 3);
                    assert_eq!(
                        spans[1]
                            .get("fields")
                            .unwrap()
                            .get("kind")
                            .unwrap()
                            .as_str(),
                        Some("verification")
                    );
                }
                kinds.push(kind);
                assert!(
                    event
                        .get("fields")
                        .unwrap()
                        .get("duration_ns")
                        .unwrap()
                        .as_u64()
                        .unwrap()
                        > 0
                );
            }
        }
    }
    assert!(active.is_empty());
    if accepted_build() {
        assert_eq!(
            kinds,
            [
                "load",
                "statement_pin",
                "input",
                "params",
                "claims",
                "statement",
                "transcript",
                "leaf",
                "stages",
                "evaluations",
                "queries",
                "columns",
                "exact_bound",
                "complete",
                "verification"
            ]
        );
    } else {
        assert_eq!(kinds, ["load", "statement_pin"]);
    }
}
#[test]
fn manifests_match_schema_and_record_artifacts_and_check_histograms() {
    let dir = Dir::new();
    let mut args = fixture(&dir);
    args.extend(["--log-format".into(), "json".into()]);
    args.extend([
        "--json".into(),
        "--out".into(),
        dir.0.join("verdict.json").display().to_string(),
    ]);
    let output = raw(&dir, &args);
    let default_path = dir.0.join("run-manifest.json");
    let manifest = schema::check("run-manifest", &fs::read_to_string(&default_path).unwrap());
    assert_eq!(manifest.get("command").unwrap().as_str(), Some("verify"));
    if output.status.success() {
        assert!(manifest.get("error_code").unwrap().is_null());
    }
    assert_eq!(
        manifest.get("tool").unwrap().get("name").unwrap().as_str(),
        Some("tessera-verify")
    );
    assert!(
        manifest
            .get("toolchain")
            .unwrap()
            .get("rustc")
            .unwrap()
            .as_str()
            .unwrap()
            .starts_with("rustc ")
    );
    assert_eq!(
        manifest.get("exit_code").unwrap().as_u64(),
        Some(output.status.code().unwrap() as u64)
    );
    assert_eq!(manifest.get("inputs").unwrap().as_array().unwrap().len(), 3);
    for input in manifest.get("inputs").unwrap().as_array().unwrap() {
        let bytes = fs::read(input.get("path").unwrap().as_str().unwrap()).unwrap();
        let digest: String = bytes[bytes.len() - 32..]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(input.get("digest").unwrap().as_str(), Some(digest.as_str()));
    }
    assert_eq!(
        manifest.get("outputs").unwrap().as_array().unwrap().len(),
        1
    );
    let report = Json::parse(std::str::from_utf8(&output.stdout).unwrap()).unwrap();
    for phase in report
        .get("execution")
        .unwrap()
        .get("trace")
        .unwrap()
        .as_array()
        .unwrap()
    {
        let kind = phase.get("phase").unwrap().as_str().unwrap();
        let metric = manifest
            .get("metrics")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m.get("labels").unwrap().get("check").unwrap().as_str() == Some(kind))
            .unwrap();
        let Json::Number(seconds) = metric.get("sum").unwrap() else {
            panic!()
        };
        let ns = phase.get("duration_ns").unwrap().as_u64().unwrap();
        assert!((seconds.parse::<f64>().unwrap() - ns as f64 / 1_000_000_000.0).abs() < 1e-9);
    }
    let metrics = manifest.get("metrics").unwrap().as_array().unwrap();
    let lines = events(&output);
    let finished: Vec<_> = lines
        .iter()
        .filter(|e| e.get("event").unwrap().as_str() == Some("tessera.verifier.check_finished"))
        .map(|e| {
            e.get("span_path")
                .unwrap()
                .as_array()
                .unwrap()
                .last()
                .unwrap()
                .get("fields")
                .unwrap()
                .get("kind")
                .unwrap()
                .as_str()
                .unwrap()
        })
        .collect();
    let measured: Vec<_> = metrics
        .iter()
        .map(|m| {
            m.get("labels")
                .unwrap()
                .get("check")
                .unwrap()
                .as_str()
                .unwrap()
        })
        .collect();
    assert_eq!(measured, finished);
    for metric in metrics {
        assert_eq!(
            metric.get("name").unwrap().as_str(),
            Some("tessera_verifier_seconds")
        );
        assert_eq!(metric.get("type").unwrap().as_str(), Some("histogram"));
        assert_eq!(metric.get("count").unwrap().as_u64(), Some(1));
        assert!(
            metric
                .get("labels")
                .unwrap()
                .get("check")
                .unwrap()
                .as_str()
                .is_some()
        );
    }
    assert_eq!(
        Json::parse(std::str::from_utf8(&output.stdout).unwrap()).unwrap(),
        Json::parse(&fs::read_to_string(dir.0.join("verdict.json")).unwrap()).unwrap()
    );
    let override_path = dir.0.join("override-\"\\\n.json");
    args.extend(["--run-manifest".into(), override_path.display().to_string()]);
    let previous = fs::read(&default_path).unwrap();
    assert_eq!(raw(&dir, &args).status.code(), output.status.code());
    schema::check("run-manifest", &fs::read_to_string(override_path).unwrap());
    assert_eq!(fs::read(default_path).unwrap(), previous);
}
#[test]
fn failed_checks_finish_their_span_and_manifest_with_public_errors() {
    let dir = Dir::new();
    let mut args = dir.args();
    args.extend(["--log-format".into(), "json".into()]);
    let output = raw(&dir, &args);
    assert_eq!(output.status.code(), Some(65));
    let lines = events(&output);
    assert!(lines.iter().any(|v| v.get("event").unwrap().as_str()
        == Some("tessera.verifier.check_finished")
        && v.get("fields").unwrap().get("outcome").unwrap().as_str() == Some("error")));
    let text = fs::read_to_string(dir.0.join("run-manifest.json")).unwrap();
    assert!(!text.contains("public fixture canary"));
    let manifest = schema::check("run-manifest", &text);
    assert_eq!(
        manifest.get("error_code").unwrap().as_str(),
        Some("ERR-CODEC-MAGIC")
    );
    assert_eq!(manifest.get("exit_code").unwrap().as_u64(), Some(65));
}
#[test]
fn observability_options_reject_invalid_values_and_preserve_input_and_report_paths() {
    let dir = Dir::new();
    for extra in [
        vec!["--log-format", "xml"],
        vec!["--log-format"],
        vec!["--log-format", "json", "--log-format", "human"],
        vec!["--run-manifest"],
        vec!["--run-manifest", "a", "--run-manifest", "b"],
    ] {
        let mut args = dir.args();
        args.extend(extra.into_iter().map(String::from));
        assert_eq!(raw(&dir, &args).status.code(), Some(64));
        assert!(!dir.0.join("run-manifest.json").exists());
    }
    for name in ["input.bin", "weights.bin", "proof.bin", "verdict.json"] {
        let mut args = dir.args();
        let output = dir.0.join("verdict.json");
        fs::write(&output, b"keep report").unwrap();
        args.extend([
            "--out".into(),
            output.display().to_string(),
            "--run-manifest".into(),
            dir.0.join(name).display().to_string(),
        ]);
        assert_eq!(raw(&dir, &args).status.code(), Some(64));
        assert_eq!(fs::read(&output).unwrap(), b"keep report");
        assert_eq!(
            fs::read(dir.0.join("input.bin")).unwrap(),
            b"public fixture canary"
        );
    }
    let mut args = dir.args();
    args.extend([
        "--run-manifest".into(),
        dir.0.display().to_string(),
        "--log-format".into(),
        "json".into(),
    ]);
    let output = raw(&dir, &args);
    assert_eq!(output.status.code(), Some(20));
    assert_eq!(
        error(&Json::parse(std::str::from_utf8(&output.stdout).unwrap()).unwrap()),
        "ERR-RES-IO"
    );
    events(&output);
}

#[test]
fn invalid_json_logging_requests_still_emit_only_schema_valid_stderr() {
    let dir = Dir::new();
    for json_report in [false, true] {
        let mut args = dir.args();
        if !json_report {
            args.retain(|a| a != "--json");
        }
        args.extend([
            "--log-format".into(),
            "json".into(),
            "--unknown".into(),
            "x".into(),
        ]);
        let output = raw(&dir, &args);
        assert_eq!(output.status.code(), Some(64));
        events(&output);
        assert!(!dir.0.join("run-manifest.json").exists());
    }
}
#[test]
fn a_written_error_report_is_an_output_artifact_but_a_failed_report_is_not() {
    let dir = Dir::new();
    let mut args = fixture(&dir);
    args.extend([
        "--out".into(),
        dir.0.join("verdict.json").display().to_string(),
        "--json".into(),
    ]);
    let at = args.iter().position(|a| a == "--claims").unwrap();
    args[at + 1] = dir.0.join("missing").display().to_string();
    let output = raw(&dir, &args);
    assert_eq!(output.status.code(), Some(20));
    let path = dir.0.join("run-manifest.json");
    let m = schema::check("run-manifest", &fs::read_to_string(&path).unwrap());
    assert_eq!(m.get("outputs").unwrap().as_array().unwrap().len(), 1);
    let blocked = dir.0.join("blocked");
    fs::create_dir(&blocked).unwrap();
    let at = args.iter().position(|a| a == "--out").unwrap();
    args[at + 1] = blocked.display().to_string();
    assert_eq!(raw(&dir, &args).status.code(), Some(20));
    let m = schema::check("run-manifest", &fs::read_to_string(path).unwrap());
    assert!(m.get("outputs").unwrap().as_array().unwrap().is_empty());
}
#[test]
fn help_and_other_commands_do_not_write_verification_manifests() {
    let dir = Dir::new();
    for args in [
        vec!["verify", "--help"],
        vec!["params", "--params", "test-small", "--json"],
    ] {
        let output = raw(
            &dir,
            &args.into_iter().map(String::from).collect::<Vec<_>>(),
        );
        assert_eq!(output.status.code(), Some(0));
        assert!(!dir.0.join("run-manifest.json").exists());
        assert!(output.stderr.is_empty());
    }
}

#[test]
fn rejected_driver_checks_close_in_order_and_stop_before_later_checks() {
    if !accepted_build() {
        return;
    }
    let dir = Dir::new();
    let args = fixture(&dir);
    let good = tv_core::protocol::NiProof::open(
        &fs::read(&args[6]).unwrap(),
        &tv_codec::DecodeLimits::MESSAGE,
    )
    .unwrap();
    for (kind, later) in [
        ("leaf", "stages"),
        ("stages", "evaluations"),
        ("columns", "exact_bound"),
    ] {
        let mut proof = good.clone();
        match kind {
            "leaf" => proof.round2.logup.roots.swap(1, 2),
            "stages" => proof.round2.stages.clear(),
            "columns" => proof.round3.advice_proof.siblings[0][0] ^= 1,
            _ => unreachable!(),
        }
        let payload = proof.to_tce().unwrap();
        let original = fs::read(&args[6]).unwrap();
        let mut bytes = original[..12].to_vec();
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&payload);
        bytes.extend_from_slice(&tv_hash::hash(tv_hash::Domain::Envelope, &bytes));
        fs::write(&args[6], bytes).unwrap();
        let mut args = args.clone();
        args.extend(["--log-format".into(), "json".into(), "--json".into()]);
        let output = raw(&dir, &args);
        assert_eq!(
            output.status.code(),
            Some(10),
            "{kind}: {}",
            String::from_utf8_lossy(&output.stdout)
        );
        let lines = events(&output);
        let finished: Vec<_> = lines
            .iter()
            .filter(|e| e.get("event").unwrap().as_str() == Some("tessera.verifier.check_finished"))
            .collect();
        let kinds: Vec<_> = finished
            .iter()
            .map(|e| {
                e.get("span_path")
                    .unwrap()
                    .as_array()
                    .unwrap()
                    .last()
                    .unwrap()
                    .get("fields")
                    .unwrap()
                    .get("kind")
                    .unwrap()
                    .as_str()
                    .unwrap()
            })
            .collect();
        assert!(kinds.contains(&kind));
        assert!(!kinds.contains(&later));
        let failed_index = kinds.iter().position(|k| *k == kind).unwrap();
        for successful in &finished[..failed_index] {
            assert_eq!(
                successful
                    .get("fields")
                    .unwrap()
                    .get("outcome")
                    .unwrap()
                    .as_str(),
                Some("completed")
            );
        }
        let failed = finished[failed_index];
        assert_eq!(
            failed
                .get("fields")
                .unwrap()
                .get("outcome")
                .unwrap()
                .as_str(),
            Some("error")
        );
        let manifest = schema::check(
            "run-manifest",
            &fs::read_to_string(dir.0.join("run-manifest.json")).unwrap(),
        );
        let metrics = manifest.get("metrics").unwrap().as_array().unwrap();
        assert_eq!(metrics.len(), kinds.len());
        let measured: Vec<_> = metrics
            .iter()
            .map(|m| {
                m.get("labels")
                    .unwrap()
                    .get("check")
                    .unwrap()
                    .as_str()
                    .unwrap()
            })
            .collect();
        assert_eq!(measured, kinds);
        for (metric, event) in metrics.iter().zip(finished) {
            let Json::Number(seconds) = metric.get("sum").unwrap() else {
                panic!()
            };
            let seconds: f64 = seconds.parse().unwrap();
            let ns = event
                .get("fields")
                .unwrap()
                .get("duration_ns")
                .unwrap()
                .as_u64()
                .unwrap();
            assert!((seconds - ns as f64 / 1_000_000_000.0).abs() < 1e-9);
            assert_eq!(metric.get("min"), metric.get("sum"));
            assert_eq!(metric.get("max"), metric.get("sum"));
        }
    }
}

#[test]
fn manifest_default_follows_the_verdict_or_proof_instead_of_working_directory() {
    let dir = Dir::new();
    let args = dir.args();
    let cwd = dir.0.join("cwd");
    fs::create_dir(&cwd).unwrap();
    let invoke = |args: &[String]| {
        Command::new(env!("CARGO_BIN_EXE_tessera-verify"))
            .current_dir(&cwd)
            .args(args)
            .output()
            .unwrap()
    };
    assert_eq!(invoke(&args).status.code(), Some(65));
    let proof_manifest = dir.0.join("run-manifest.json");
    let previous = fs::read(&proof_manifest).unwrap();
    assert!(!cwd.join("run-manifest.json").exists());
    let out_dir = dir.0.join("reports");
    fs::create_dir(&out_dir).unwrap();
    let mut args = args;
    args.extend([
        "--out".into(),
        out_dir.join("verdict.json").display().to_string(),
    ]);
    assert_eq!(invoke(&args).status.code(), Some(65));
    schema::check(
        "run-manifest",
        &fs::read_to_string(out_dir.join("run-manifest.json")).unwrap(),
    );
    assert_eq!(fs::read(proof_manifest).unwrap(), previous);
    assert!(!cwd.join("run-manifest.json").exists());
}

#[test]
fn oversized_verify_requests_honor_json_logging_before_option_parsing() {
    let dir = Dir::new();
    let mut args = vec!["verify".into(), "--log-format".into(), "json".into()];
    args.push("x".repeat(8193));
    let output = raw(&dir, &args);
    assert_eq!(output.status.code(), Some(64));
    events(&output);
    assert!(!dir.0.join("run-manifest.json").exists());
}

#[test]
fn fresh_verdict_and_manifest_aliases_are_refused_before_either_write() {
    let dir = Dir::new();
    fs::create_dir(dir.0.join("nested")).unwrap();
    for (out, manifest) in [
        ("explicit.json", Some("explicit.json")),
        ("run-manifest.json", None),
        ("relative.json", Some("./relative.json")),
        ("./reverse.json", Some("nested/../reverse.json")),
    ] {
        let mut args = dir.args();
        args.extend(["--out".into(), out.into()]);
        if let Some(manifest) = manifest {
            args.extend(["--run-manifest".into(), manifest.into()]);
        }
        assert!(!dir.0.join(out).exists());
        let output = raw(&dir, &args);
        assert_eq!(output.status.code(), Some(64), "{out}");
        assert_eq!(
            error(&Json::parse(std::str::from_utf8(&output.stdout).unwrap()).unwrap()),
            "ERR-USAGE"
        );
        assert!(!dir.0.join(out).exists(), "wrote colliding output {out}");
        if let Some(manifest) = manifest {
            assert!(!dir.0.join(manifest).exists());
        }
        assert_eq!(
            fs::read(dir.0.join("input.bin")).unwrap(),
            b"public fixture canary"
        );
    }
}

#[cfg(unix)]
#[test]
fn non_utf8_arguments_preserve_json_logging_and_verdict_format_intent() {
    use std::os::unix::ffi::OsStringExt;
    let dir = Dir::new();
    for json_report in [false, true] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_tessera-verify"));
        command
            .current_dir(&dir.0)
            .args(["verify", "--log-format", "json"])
            .arg(std::ffi::OsString::from_vec(vec![0xff]));
        if json_report {
            command.arg("--json");
        }
        let output = command.output().unwrap();
        assert_eq!(output.status.code(), Some(64));
        let lines = events(&output);
        assert_eq!(
            lines
                .last()
                .unwrap()
                .get("fields")
                .unwrap()
                .get("exit_code")
                .unwrap()
                .as_u64(),
            Some(64)
        );
        if json_report {
            assert_eq!(
                error(&Json::parse(std::str::from_utf8(&output.stdout).unwrap()).unwrap()),
                "ERR-USAGE"
            );
        } else {
            assert!(output.stdout.is_empty());
        }
        assert!(!dir.0.join("run-manifest.json").exists());
    }
}
