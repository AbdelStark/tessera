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
    assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 4);
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
    assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 5);
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
fn os_argument_collection_stops_at_the_same_limit_as_utf8_arguments() {
    use std::ffi::OsString;
    assert_eq!(
        tv_cli::run_os(std::iter::repeat_n(OsString::from("--help"), 64)).exit_code,
        0
    );
    let read = std::cell::Cell::new(0);
    let args = (0..).map(|i| {
        assert!(i < 65, "argument iterator read past the limit");
        read.set(read.get() + 1);
        OsString::from("--help")
    });
    let output = tv_cli::run_os(args);
    assert_eq!(output.exit_code, 64);
    assert!(output.stderr.contains("too many or oversized arguments"));
    assert_eq!(read.get(), 65);
}

#[test]
fn os_argument_panics_stay_inside_the_cli_boundary() {
    let args =
        std::iter::once_with(|| -> std::ffi::OsString { panic!("private-os-argument-canary") });
    let output = tv_cli::run_os(args);
    assert_eq!(output.exit_code, 70);
    assert!(output.stderr.is_empty());
    assert!(!output.stdout.contains("private-os-argument-canary"));
    assert_eq!(error(&Json::parse(&output.stdout).unwrap()), "ERR-INTERNAL");
}

#[test]
fn oversized_os_arguments_preserve_the_requested_error_format() {
    let args = std::iter::repeat_n(std::ffi::OsString::from("--help"), 64)
        .chain([std::ffi::OsString::from("--json")]);
    let output = tv_cli::run_os(args);
    assert_eq!(output.exit_code, 64);
    assert!(output.stderr.is_empty());
    assert_eq!(error(&Json::parse(&output.stdout).unwrap()), "ERR-USAGE");
    for option in ["--log-format", "--log-level"] {
        let args = [option.to_string(), "x".repeat(8193)];
        let output = tv_cli::run_os(args.iter().map(std::ffi::OsString::from));
        assert_eq!(output, tv_cli::run(&args));
        assert!(
            output
                .stderr
                .contains(&format!("invalid or missing `{option}` value"))
        );
    }
    for (bytes, exit) in [(8192, 0), (8193, 64)] {
        assert_eq!(
            tv_cli::run_os(["--help".into(), std::ffi::OsString::from("x".repeat(bytes))])
                .exit_code,
            exit
        );
    }
    for args in [
        vec!["--log-format".into(), "json".into(), "x".repeat(8193)],
        vec!["--json".into(), "x".repeat(8193)],
    ] {
        let expected = tv_cli::run(&args);
        let actual = tv_cli::run_os(args.into_iter().map(std::ffi::OsString::from));
        assert_eq!(actual.exit_code, expected.exit_code);
        assert_eq!(actual.stdout, expected.stdout);
        if expected.stderr.is_empty() {
            assert!(actual.stderr.is_empty());
        } else {
            let log = Json::parse(&actual.stderr).unwrap();
            let fields = log.get("fields").unwrap();
            assert_eq!(fields.get("code").unwrap().as_str(), Some("ERR-USAGE"));
            assert_eq!(
                fields.get("detail").unwrap().as_str(),
                Some("too many or oversized arguments")
            );
        }
    }
}

#[cfg(unix)]
#[test]
fn oversized_non_utf8_os_arguments_take_the_size_limit_first() {
    use std::os::unix::ffi::OsStringExt;
    let output = tv_cli::run_os([
        std::ffi::OsString::from("--json"),
        std::ffi::OsString::from_vec(vec![0xff; 8193]),
    ]);
    assert_eq!(output.exit_code, 64);
    assert!(output.stderr.is_empty());
    let report = Json::parse(&output.stdout).unwrap();
    assert_eq!(error(&report), "ERR-USAGE");
    assert_eq!(
        report
            .get("error")
            .unwrap()
            .get("message")
            .unwrap()
            .as_str(),
        Some("too many or oversized arguments")
    );
    let mut args = vec![
        std::ffi::OsString::from("--json"),
        std::ffi::OsString::from_vec(vec![0xff]),
    ];
    args.extend(std::iter::repeat_n(std::ffi::OsString::from("--help"), 63));
    let output = tv_cli::run_os(args);
    assert_eq!(output.exit_code, 64);
    assert!(output.stderr.is_empty());
    let report = Json::parse(&output.stdout).unwrap();
    assert_eq!(
        report
            .get("error")
            .unwrap()
            .get("message")
            .unwrap()
            .as_str(),
        Some("too many or oversized arguments")
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

#[test]
fn bound_option_accepts_exact_256ths_and_rejects_inexact_or_overflowing_decimals() {
    let dir = Dir::new();
    let base = dir.args();
    let at = base.iter().position(|a| a == "--bound").unwrap();

    for exact in ["0", "0.00390625", "6.25", "72057594037927935.99609375"] {
        let mut args = base.clone();
        args[at + 1] = exact.into();
        let (code, report) = binary(&args);
        // Passes CLI --bound validation and proceeds to artifact envelope opening (65).
        assert_eq!(code, 65, "exact --bound {exact} should pass CLI parsing");
        assert_eq!(error(&report), "ERR-CODEC-MAGIC", "{exact}");
    }

    for invalid in [
        "0.001",
        "0.3",
        "-0.5",
        "1.",
        "0.000000001",
        "72057594037927936",
    ] {
        let mut args = base.clone();
        args[at + 1] = invalid.into();
        let (code, report) = binary(&args);
        assert_eq!(code, 64, "invalid --bound {invalid} must fail as ERR-USAGE");
        assert_eq!(error(&report), "ERR-USAGE", "{invalid}");
    }
}
