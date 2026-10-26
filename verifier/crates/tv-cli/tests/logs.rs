// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tessera-verify`'s log lines (05 §2 'The verifier's lines', §5 'The verifier's spans'):
//! today's human lines at the default level, `tessera.log-event/1` lines with
//! `--log-format json`, `--log-level` choosing the most detailed level written, and both options
//! read before any command's own.

mod observe_support;
use observe_support::{
    Field, Fixture, accepts_test_small, assert_event, envelope_digest, lines, parse_line,
};
use tv_cli::{CommandOutput, run};

const LEVELS: [&str; 5] = ["error", "warn", "info", "debug", "trace"];
const CHECKS: [&str; 13] = [
    "input",
    "params",
    "claims",
    "statement",
    "weight_reservation",
    "transcript",
    "leaf",
    "stages",
    "evaluations",
    "queries",
    "columns",
    "exact_bound",
    "complete",
];

/// The human `verified` line: `verify: completed in N ns; protocol/1, anchor/ledger/ZK unchecked`.
fn assert_completed(line: &str) {
    let rest = line
        .strip_prefix("verify: completed in ")
        .unwrap_or_else(|| panic!("{line:?}"));
    let (n, tail) = rest.split_once(" ns").unwrap();
    assert!(
        !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()),
        "{line:?}"
    );
    assert_eq!(tail, "; protocol/1, anchor/ledger/ZK unchecked", "{line:?}");
}

/// What the fixture's verdict is in this build: steps 1 and 2 with `REJ-PARAMS` in a plain
/// build, all 13 when test-small is accepted.
fn expected_verdict() -> (&'static str, Option<&'static str>, u64, u8) {
    if accepts_test_small() {
        ("accept", None, 0, 13)
    } else {
        ("reject", Some("REJ-PARAMS"), 10, 2)
    }
}

#[test]
fn human_lines_are_todays_at_the_default_level() {
    let refused = |args: &[&str], line: &str| {
        for extra in [
            &[][..],
            &["--log-format", "human"],
            &["--log-level", "info"],
        ] {
            let mut all: Vec<&str> = args.to_vec();
            all.extend_from_slice(extra);
            let out = run(&all);
            assert_eq!(out.exit_code, 64, "{all:?}");
            assert_eq!(out.stdout, "", "{all:?}");
            assert_eq!(out.stderr, line, "{all:?}");
        }
    };
    refused(&[], "error (ERR-USAGE): missing command\n");
    refused(
        &["frobnicate"],
        "error (ERR-USAGE): unknown command `frobnicate`\n",
    );
    refused(
        &["verify", "--bogus", "1"],
        "error (ERR-USAGE): invalid, missing or duplicate option; see verify --help\n",
    );
    refused(
        &["params", "--params", "nope"],
        "error (ERR-USAGE): unknown parameter set `nope` (expected live-64, ni-100, or test-small)\n",
    );
    refused(
        &["weights", "link"],
        "error (ERR-USAGE): this operation requires the ledger's weight epochs (#158)\n",
    );
    let f = Fixture::new();
    let (status, code, exit, _) = expected_verdict();
    for extra in [&[][..], &["--log-format", "human", "--log-level", "info"]] {
        let out = run(f.with(extra));
        assert_eq!(i64::from(out.exit_code), exit as i64);
        let stdout = match code {
            None => "Accept (test-small/1; test parameters certify nothing)\n".to_string(),
            Some(code) => format!("Reject ({code})\n"),
        };
        assert_eq!(out.stdout, stdout, "{status}");
        let line = out.stderr.strip_suffix('\n').unwrap();
        assert!(!line.contains('\n'), "{:?}", out.stderr);
        assert_completed(line);
    }
    // With --json, stdout carries the result and the human format writes nothing.
    for extra in [&["--json"][..], &["--json", "--log-level", "trace"]] {
        let out = run(f.with(extra));
        assert_eq!(out.stderr, "", "{extra:?}");
        assert!(out.stdout.starts_with("{\"schema\":\"tessera.verdict/1\""));
    }
    let out = run(["verify", "--bogus", "1", "--json"]);
    assert_eq!(out.stderr, "");
}

#[test]
fn json_lines_follow_the_schema_and_carry_the_spans() {
    let f = Fixture::new();
    let (status, code, exit, last) = expected_verdict();
    let digest = envelope_digest(&f.proof);
    for json in [false, true] {
        let mut extra = vec!["--log-format", "json", "--log-level", "debug"];
        if json {
            extra.push("--json");
        }
        let out = run(f.with(&extra));
        assert_eq!(u64::try_from(out.exit_code).unwrap(), exit);
        // The JSON format writes its lines with or without --json; stdout is unchanged.
        assert_eq!(out.stdout.starts_with('{'), json, "{}", out.stdout);
        let parsed = lines(&out.stderr);
        assert_eq!(parsed.len(), usize::from(last) + 1, "{}", out.stderr);
        let verify_span = vec![("proof_digest".to_string(), Field::Text(digest.clone()))];
        // Each line carries the time of its event, so they follow each other in time.
        for pair in parsed.windows(2) {
            assert!(
                pair[0].ts <= pair[1].ts,
                "{} then {}",
                pair[0].ts,
                pair[1].ts
            );
        }
        for (i, line) in parsed.iter().enumerate() {
            assert_event(line);
            if i < usize::from(last) {
                assert_eq!(line.event, "tessera.tv_cli.check_completed");
                assert_eq!(line.field("step").unwrap().int(), i as u64 + 1);
                if i == 4 {
                    // Step 5 does nothing in `tessera-protocol/1`.
                    assert_eq!(line.field("duration_ns").unwrap().int(), 0);
                }
                let outcome = line.field("outcome").unwrap().text();
                let expected = if i + 1 == usize::from(last) && code.is_some() {
                    "reject"
                } else if i == 4 {
                    "not_applicable"
                } else {
                    "pass"
                };
                assert_eq!(outcome, expected, "{line:?}");
                assert_eq!(line.spans.len(), 2, "{line:?}");
                assert_eq!(line.spans[0].name, "verify");
                assert_eq!(line.spans[0].fields, verify_span);
                assert_eq!(line.spans[1].name, "check");
                assert_eq!(
                    line.spans[1].fields,
                    [("kind".to_string(), Field::Text(CHECKS[i].to_string()))]
                );
            } else {
                assert_eq!(line.event, "tessera.tv_cli.verified");
                assert_eq!(line.field("status").unwrap().text(), status);
                assert_eq!(line.field("code").map(Field::text), code);
                assert_eq!(line.field("exit_code").unwrap().int(), exit);
                assert_eq!(line.spans.len(), 1);
                assert_eq!(line.spans[0].name, "verify");
                assert_eq!(line.spans[0].fields, verify_span);
            }
        }
        // The deterministic report keeps its null per-check durations (05 §5).
        if json {
            let report = tv_testkit::Json::parse(&out.stdout).unwrap();
            let checks = report.get("checks").and_then(|c| c.as_array()).unwrap();
            assert_eq!(checks.len(), if status == "accept" { 13 } else { 0 });
            for check in checks {
                assert!(check.get("duration_ns").unwrap().is_null(), "{check:?}");
            }
        }
    }
}

#[test]
fn log_level_selects_the_most_detailed_level_written() {
    let f = Fixture::new();
    let (_, _, _, last) = expected_verdict();
    for (rank, level) in LEVELS.iter().enumerate() {
        // verify writes `verified` at info and one `check_completed` per check at debug.
        let expected = match rank {
            0 | 1 => 0,
            2 => 1,
            _ => usize::from(last) + 1,
        };
        let out = run(f.with(&["--log-format", "json", "--log-level", level]));
        let parsed = lines(&out.stderr);
        assert_eq!(parsed.len(), expected, "{level}: {}", out.stderr);
        for line in &parsed {
            let line_rank = LEVELS.iter().position(|l| *l == line.level).unwrap();
            assert!(line_rank <= rank, "{level}: {line:?}");
        }
        // A refusal is at error, which every level writes.
        let out = run([
            "verify",
            "--bogus",
            "1",
            "--log-format",
            "json",
            "--log-level",
            level,
        ]);
        let parsed = lines(&out.stderr);
        assert_eq!(parsed.len(), 1, "{level}");
        assert_eq!(parsed[0].event, "tessera.tv_cli.command_failed");
        // The human format filters alike, and writes no level prefix below debug.
        let out = run(f.with(&["--log-level", level]));
        let human: Vec<&str> = out.stderr.lines().collect();
        assert_eq!(human.len(), expected, "{level}: {}", out.stderr);
        if let Some(line) = human.last() {
            assert_completed(line);
        }
    }
}

#[test]
fn human_debug_lines_carry_the_level_and_the_span_path() {
    let f = Fixture::new();
    let (_, code, _, last) = expected_verdict();
    let digest = envelope_digest(&f.proof);
    let out = run(f.with(&["--log-level", "debug"]));
    let human: Vec<&str> = out.stderr.lines().collect();
    assert_eq!(human.len(), usize::from(last) + 1, "{}", out.stderr);
    for (i, line) in human[..usize::from(last)].iter().enumerate() {
        let prefix = format!(
            "debug verify{{proof_digest={digest}}}:check{{kind={}}}: step {} ",
            CHECKS[i],
            i + 1
        );
        let rest = line
            .strip_prefix(&prefix)
            .unwrap_or_else(|| panic!("{line:?} lacks {prefix:?}"));
        let outcome = if i + 1 == usize::from(last) && code.is_some() {
            "reject"
        } else if i == 4 {
            "not_applicable"
        } else {
            "pass"
        };
        let ns = rest
            .strip_prefix(&format!("{outcome} in "))
            .and_then(|r| r.strip_suffix(" ns"))
            .unwrap_or_else(|| panic!("{line:?}"));
        assert!(ns.bytes().all(|b| b.is_ascii_digit()) && !ns.is_empty());
    }
    assert_completed(human[usize::from(last)]);
}

#[test]
fn a_proof_that_fails_to_open_records_no_digest() {
    let f = Fixture::new();
    std::fs::write(&f.proof, b"not an envelope").unwrap();
    let out = run(f.with(&["--log-level", "debug"]));
    assert_eq!(out.exit_code, 65);
    let human: Vec<&str> = out.stderr.lines().collect();
    assert_eq!(human.len(), 2, "{}", out.stderr);
    let rest = human[0]
        .strip_prefix("debug verify:check{kind=input}: step 1 inconclusive in ")
        .unwrap_or_else(|| panic!("{:?}", human[0]));
    assert!(rest.ends_with(" ns"));
    assert_completed(human[1]);
    let out = run(f.with(&["--log-format", "json", "--log-level", "debug"]));
    for line in lines(&out.stderr) {
        assert_event(&line);
        assert_eq!(line.spans[0].name, "verify");
        assert!(line.spans[0].fields.is_empty(), "{line:?}");
        if line.event == "tessera.tv_cli.verified" {
            assert_eq!(line.field("status").unwrap().text(), "inconclusive");
            assert_eq!(line.field("code").unwrap().text(), "ERR-CODEC-MAGIC");
            assert_eq!(line.field("exit_code").unwrap().int(), 65);
        }
    }
}

#[test]
fn a_proof_whose_envelope_verifies_records_its_digest_even_if_its_payload_fails() {
    let f = Fixture::new();
    // The proof's envelope resealed over a one-byte payload: sound framing and digest.
    let mut bytes = std::fs::read(&f.proof).unwrap()[..12].to_vec();
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.push(0);
    let digest = tv_hash::hash(tv_hash::Domain::Envelope, &bytes);
    bytes.extend_from_slice(&digest);
    std::fs::write(&f.proof, &bytes).unwrap();
    let out = run(f.with(&["--log-format", "json", "--log-level", "debug"]));
    assert_eq!(out.exit_code, 65);
    let parsed = lines(&out.stderr);
    assert_eq!(parsed.len(), 2, "{}", out.stderr);
    for line in &parsed {
        assert_event(line);
        assert_eq!(
            line.spans[0].fields,
            [(
                "proof_digest".to_string(),
                Field::Text(envelope_digest(&f.proof))
            )]
        );
    }
    assert_eq!(parsed[0].field("outcome").unwrap().text(), "inconclusive");
    assert_eq!(
        parsed[1].field("code").unwrap().text(),
        "ERR-CODEC-TRUNCATED"
    );
}

/// `out` is one `command_failed` line of `command` with `detail`, exit 64.
fn assert_refused_as_json(out: &CommandOutput, command: &str, detail: &str) {
    assert_eq!(out.exit_code, 64);
    let parsed = lines(&out.stderr);
    assert_eq!(parsed.len(), 1, "{}", out.stderr);
    let line = &parsed[0];
    assert_event(line);
    assert_eq!(line.event, "tessera.tv_cli.command_failed");
    assert!(line.spans.is_empty());
    assert_eq!(line.field("command").unwrap().text(), command);
    assert_eq!(line.field("code").unwrap().text(), "ERR-USAGE");
    assert_eq!(line.field("detail").unwrap().text(), detail);
}

#[test]
fn a_refused_option_is_logged_in_the_format_asked_for() {
    let detail = "invalid, missing or duplicate option; see verify --help";
    for args in [
        &["verify", "--bogus", "1", "--log-format", "json"][..],
        &["--log-format", "json", "verify", "--bogus", "1"],
        &[
            "verify",
            "--log-level",
            "debug",
            "--bogus",
            "1",
            "--log-format",
            "json",
        ],
    ] {
        let out = run(args);
        assert_eq!(out.stdout, "");
        assert_refused_as_json(&out, "verify", detail);
    }
    // With --json the result object is still stdout's.
    let out = run(["verify", "--bogus", "1", "--json", "--log-format", "json"]);
    assert_refused_as_json(&out, "verify", detail);
    assert_eq!(
        out.stdout,
        format!(
            "{{\"command\":\"verify\",\"error\":{{\"code\":\"ERR-USAGE\",\"message\":\"{detail}\"}}}}\n"
        )
    );
    for (args, command, detail) in [
        (
            &["--log-format", "json"][..],
            "tessera-verify",
            "missing command",
        ),
        (
            &["--log-format", "json", "nope"],
            "tessera-verify",
            "unknown command `nope`",
        ),
        (
            &["tables", "--log-format", "json", "--nope", "x"],
            "tables",
            "table catalog could not be evaluated",
        ),
        (
            &["live", "--log-format", "json"],
            "live",
            "invalid or missing request policy; see live --help",
        ),
        (
            &["audit", "nope", "--log-format", "json"],
            "audit",
            "unknown audit command `nope`",
        ),
        (
            &["params", "--log-format", "json"],
            "params",
            "missing required `--params <PARAMSET>`",
        ),
    ] {
        assert_refused_as_json(&run(args), command, detail);
    }
    let many = vec!["--help"; 70];
    let mut args: Vec<&str> = vec!["--log-format", "json"];
    args.extend(many);
    assert_refused_as_json(
        &run(args),
        "tessera-verify",
        "too many or oversized arguments",
    );
}

#[test]
fn an_invalid_log_option_is_refused_in_the_human_format() {
    for args in [
        &["--log-format", "xml", "verify"][..],
        &["verify", "--log-format"],
        &["verify", "--log-format", "JSON"],
        &["--log-format", "json", "--log-level", "verbose", "verify"],
        &["--log-level", "json", "--log-format", "json"],
        &["--log-level"],
        &["--log-format", "json", "--log-format", "json", "verify"],
        &["--log-level", "info", "--log-level", "debug"],
    ] {
        let out = run(args);
        assert_eq!(out.exit_code, 64, "{args:?}");
        assert_eq!(out.stdout, "", "{args:?}");
        let line = out.stderr.strip_suffix('\n').unwrap();
        assert!(!line.contains('\n'), "{args:?}: {:?}", out.stderr);
        assert!(line.starts_with("error (ERR-USAGE): "), "{args:?}: {line}");
        assert!(line.contains("--log-"), "{args:?}: {line}");
        // Under --json the result object says so, and the human format writes no line.
        let mut with_json = args.to_vec();
        with_json.push("--json");
        let out = run(&with_json);
        assert_eq!(out.exit_code, 64);
        assert_eq!(out.stderr, "");
        assert!(
            out.stdout
                .starts_with("{\"command\":\"tessera-verify\",\"error\":{\"code\":\"ERR-USAGE\""),
            "{}",
            out.stdout
        );
    }
}

#[test]
fn every_command_takes_both_options() {
    // Help, params and tables succeed with the options, which reach no command parser.
    for args in [
        &["--help", "--log-format", "json", "--log-level", "trace"][..],
        &["verify", "--help", "--log-format", "json"],
        &[
            "--log-level",
            "error",
            "params",
            "--params",
            "ni-100",
            "--json",
        ],
        &["audit", "--log-level", "warn", "--help"],
    ] {
        let out = run(args);
        assert_eq!(out.exit_code, 0, "{args:?}: {}", out.stderr);
        assert_eq!(out.stderr, "", "{args:?}");
    }
    let parsed =
        parse_line(&run(["--log-level", "trace", "--log-format", "json", "frobnicate"]).stderr);
    assert_eq!(parsed.level, "error");
}

/// A non-UTF-8 argument is refused as before, `ERR-USAGE`, with the same human line, but in the
/// format the valid arguments ask for (05 §2), at the library's and the binary's boundary.
#[cfg(unix)]
#[test]
fn a_non_utf8_argument_is_refused_in_the_format_asked_for() {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;
    let bad = || OsString::from_vec(vec![b'x', 0xff]);
    let os = |args: &[&str]| -> Vec<OsString> { args.iter().map(OsString::from).collect() };
    let detail = "unknown command `invalid-utf8-argument`";
    // Today's human line, unchanged.
    let mut args = os(&["verify", "--claims"]);
    args.push(bad());
    let out = tv_cli::run_os(args);
    assert_eq!(out.exit_code, 64);
    assert_eq!(out.stdout, "");
    assert_eq!(out.stderr, format!("error (ERR-USAGE): {detail}\n"));
    // The options read before every other one still apply, wherever they are.
    for (before, after) in [
        (&["--log-format", "json", "verify"][..], &[][..]),
        (
            &["verify"],
            &["--log-level", "error", "--log-format", "json"],
        ),
    ] {
        let mut args = os(before);
        args.push(bad());
        args.extend(os(after));
        let out = tv_cli::run_os(args);
        assert_eq!(out.stdout, "");
        assert_refused_as_json(&out, "tessera-verify", detail);
    }
    // So does --json, whose stdout carries the result.
    let mut args = os(&["--json", "--log-format", "json"]);
    args.push(bad());
    let out = tv_cli::run_os(args);
    assert_refused_as_json(&out, "tessera-verify", detail);
    assert_eq!(
        out.stdout,
        format!(
            "{{\"command\":\"tessera-verify\",\"error\":{{\"code\":\"ERR-USAGE\",\"message\":\"{detail}\"}}}}\n"
        )
    );
    // A log option whose own value is not UTF-8 is an invalid value, refused in human form.
    let mut args = os(&["--log-format"]);
    args.push(bad());
    let out = tv_cli::run_os(args);
    assert_eq!(out.exit_code, 64);
    assert!(
        out.stderr
            .starts_with("error (ERR-USAGE): invalid or missing `--log-format` value"),
        "{}",
        out.stderr
    );
    // Valid arguments run as `run` runs them.
    assert_eq!(
        tv_cli::run_os(os(&["--log-format", "human", "frobnicate"])),
        run(["--log-format", "human", "frobnicate"])
    );
    // At the binary's boundary.
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_tessera-verify"))
        .args(["--log-format", "json", "verify"])
        .arg(bad())
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(64));
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert_refused_as_json(
        &CommandOutput {
            exit_code: 64,
            stdout: String::new(),
            stderr,
        },
        "tessera-verify",
        detail,
    );
}

/// `(kind, step, outcome)` of every `check_completed` line, in order.
fn checks_of(lines: &[observe_support::Line]) -> Vec<(String, u64, String)> {
    lines
        .iter()
        .filter(|l| l.event == "tessera.tv_cli.check_completed")
        .map(|l| {
            (
                l.spans[1].fields[0].1.text().to_string(),
                l.field("step").unwrap().int(),
                l.field("outcome").unwrap().text().to_string(),
            )
        })
        .collect()
}

/// In every build, tv-core's own checks reach the log through the CLI: a claim list under the
/// production set `ni-100/1` passes the CLI's step 1, joined with the driver's, and the driver's
/// steps 2 and 3, then fails in step 3 (its layout is test-small's, `REJ-STATEMENT`) or, with
/// ni-100's layout, in step 4 (the statement names the original list, `ERR-USAGE`).
#[test]
fn the_cores_checks_reach_the_log_in_every_build() {
    let digest = |f: &Fixture| envelope_digest(&f.proof);
    for (layout, last, outcome, status, code, exit) in [
        (false, 3u64, "reject", "reject", "REJ-STATEMENT", 10u64),
        (true, 4, "inconclusive", "inconclusive", "ERR-USAGE", 64),
    ] {
        let f = Fixture::new();
        f.as_ni_100(layout);
        let manifest = f.dir.join("run-manifest.json");
        let out = run(f.with(&[
            "--log-format",
            "json",
            "--log-level",
            "debug",
            "--run-manifest",
            &manifest.display().to_string(),
        ]));
        assert_eq!(
            u64::try_from(out.exit_code).unwrap(),
            exit,
            "{}",
            out.stderr
        );
        let parsed = lines(&out.stderr);
        let expected: Vec<(String, u64, String)> = (1..=last)
            .map(|step| {
                let kind = CHECKS[usize::try_from(step - 1).unwrap()].to_string();
                let outcome = if step == last { outcome } else { "pass" };
                (kind, step, outcome.to_string())
            })
            .collect();
        // One step-1 line: the CLI's reading and pinning joined with the driver's limits.
        assert_eq!(checks_of(&parsed), expected, "{}", out.stderr);
        assert_eq!(parsed.len(), usize::try_from(last).unwrap() + 1);
        for line in &parsed {
            assert_event(line);
            assert_eq!(
                line.spans[0].fields,
                [("proof_digest".to_string(), Field::Text(digest(&f)))]
            );
        }
        let verified = parsed.last().unwrap();
        assert_eq!(verified.event, "tessera.tv_cli.verified");
        assert_eq!(verified.field("status").unwrap().text(), status);
        assert_eq!(verified.field("code").unwrap().text(), code);
        // The manifest's samples are the same checks.
        let m = tv_testkit::Json::parse(&std::fs::read_to_string(&manifest).unwrap()).unwrap();
        let labels: Vec<&str> = m
            .get("metrics")
            .and_then(|v| v.as_array())
            .unwrap()
            .iter()
            .map(|s| {
                s.get("labels")
                    .unwrap()
                    .get("check")
                    .unwrap()
                    .as_str()
                    .unwrap()
            })
            .collect();
        assert_eq!(labels, CHECKS[..usize::try_from(last).unwrap()]);
        assert_eq!(m.get("error_code").unwrap().as_str(), Some(code));
        // The same checks in human form.
        let human = run(f.with(&["--log-level", "debug"])).stderr;
        let prefixes: Vec<&str> = human
            .lines()
            .map(|l| l.split(": ").next().unwrap())
            .collect();
        let mut expected: Vec<String> = CHECKS[..usize::try_from(last).unwrap()]
            .iter()
            .map(|k| {
                format!(
                    "debug verify{{proof_digest={}}}:check{{kind={k}}}",
                    digest(&f)
                )
            })
            .collect();
        expected.push("verify".into());
        assert_eq!(prefixes, expected, "{human}");
    }
}

/// A live session through the CLI, its prover played by a thread from the tamper corpus's
/// recorded messages: Round1 takes the verifier's fresh session id, so steps 1 to 6 pass; the
/// recorded Round2 was proven for another coin toss, so the verifier's own share makes it fail.
/// The log is the CLI's step 1, `start`'s step 2, Round1's 1 to 6 with its two openings joined
/// into step 1, then Round2's checks up to the rejection; step 5 takes no time. A plain build
/// refuses test-small at step 2, before the session starts.
#[test]
fn a_live_session_logs_each_messages_checks() {
    use std::fs;
    use std::time::{Duration, Instant};
    use tv_codec::DecodeLimits;
    use tv_core::protocol::{ProverSeedShare, Round1, SeedCommit};
    let f = Fixture::new();
    let exchange = f.dir.join("exchange");
    fs::create_dir(&exchange).unwrap();
    let corpus = tv_testkit::read_vectors("tamper/protocol-1.json");
    let request = corpus.get("base").and_then(|b| b.get("request")).unwrap();
    let field = |k: &str| match request.get(k).unwrap() {
        tv_testkit::Json::String(s) | tv_testkit::Json::Number(s) => s.clone(),
        other => panic!("{other:?}"),
    };
    let mut args: Vec<String> = f.args();
    args[0] = "live".into();
    let at = args.iter().position(|a| a == "--proof").unwrap();
    args.drain(at..at + 2);
    for (key, value) in [
        ("--connect", exchange.display().to_string()),
        ("--request-id", field("request_id")),
        ("--request-session", field("session_id")),
        ("--prompt-len", field("prompt_len")),
        ("--output-len", field("output_len")),
        ("--manifest-digest", field("manifest_digest")),
        ("--timeout-ms", "30000".into()),
        ("--log-format", "json".into()),
        ("--log-level", "debug".into()),
    ] {
        args.extend([key.to_string(), value]);
    }
    let pass = |kind: &str, step: u64| (kind.to_string(), step, "pass".to_string());
    let through = |last: u64| -> Vec<(String, u64, String)> {
        (1..=last)
            .map(|step| {
                let kind = CHECKS[usize::try_from(step - 1).unwrap()];
                let outcome = if step == 5 { "not_applicable" } else { "pass" };
                (kind.to_string(), step, outcome.to_string())
            })
            .collect()
    };
    if !accepts_test_small() {
        let out = run(&args);
        assert_eq!(out.exit_code, 10);
        let parsed = lines(&out.stderr);
        assert_eq!(
            checks_of(&parsed),
            [
                pass("input", 1),
                ("params".to_string(), 2, "reject".to_string())
            ]
        );
        assert_eq!(fs::read_dir(&exchange).unwrap().count(), 0);
        return;
    }
    let dir = exchange.clone();
    let prover = std::thread::spawn(move || {
        let limits = DecodeLimits::MESSAGE;
        let base = tv_testkit::vectors_dir().join("tamper");
        let wait = |name: &str| {
            let deadline = Instant::now() + Duration::from_secs(30);
            loop {
                if let Ok(bytes) = fs::read(dir.join(name)) {
                    return bytes;
                }
                assert!(Instant::now() < deadline, "no {name}");
                std::thread::sleep(Duration::from_millis(5));
            }
        };
        let publish = |name: &str, bytes: &[u8]| {
            let temporary = dir.join(format!(".{name}.prover"));
            fs::write(&temporary, bytes).unwrap();
            fs::rename(&temporary, dir.join(name)).unwrap();
        };
        let commit = SeedCommit::open(&wait("00-SeedCommit.tce"), &limits).unwrap();
        let mut round1 =
            Round1::open(&fs::read(base.join("live-round1.tce")).unwrap(), &limits).unwrap();
        round1.session_id = commit.session_id;
        publish(
            "01-Round1.tce",
            &observe_support::seal(Round1::SCHEMA, &round1.to_tce().unwrap()),
        );
        let share = ProverSeedShare {
            session_id: commit.session_id,
            share: [7; 32],
        };
        publish(
            "01-ProverSeedShare.tce",
            &observe_support::seal(ProverSeedShare::SCHEMA, &share.to_tce().unwrap()),
        );
        wait("02-SeedReveal.tce");
        publish(
            "03-Round2Response.tce",
            &fs::read(base.join("live-round2.tce")).unwrap(),
        );
    });
    let out = run(&args);
    prover.join().unwrap();
    assert_eq!(out.exit_code, 10, "{}{}", out.stdout, out.stderr);
    let parsed = lines(&out.stderr);
    let mut expected = vec![pass("input", 1), pass("params", 2)];
    expected.extend(through(6));
    expected.extend(through(6));
    expected.push(("leaf".to_string(), 7, "reject".to_string()));
    assert_eq!(checks_of(&parsed), expected, "{}", out.stderr);
    for line in &parsed {
        assert_event(line);
        // A live session has no proof file, so its `verify` span records no digest.
        assert!(line.spans[0].fields.is_empty(), "{line:?}");
        if line.field("step").map(Field::int) == Some(5) {
            assert_eq!(line.field("duration_ns").unwrap().int(), 0);
        }
    }
    let verified = parsed.last().unwrap();
    assert_eq!(verified.field("status").unwrap().text(), "reject");
    // The queries never went out: Round2 failed before them.
    assert!(!exchange.join("04-ColumnQueries.tce").exists());
}

/// Operator text reaches `detail` and stdout's result escaped, whatever it holds: a quote, a
/// backslash and control characters, through the whole CLI.
#[test]
fn operator_text_is_escaped_end_to_end() {
    let command = "a\"b\\c\u{1}d\te\n";
    let detail = format!("unknown command `{command}`");
    for json in [false, true] {
        let mut args = vec!["--log-format", "json", command];
        if json {
            args.push("--json");
        }
        let out = run(&args);
        assert_refused_as_json(&out, "tessera-verify", &detail);
        let raw = out.stderr.strip_suffix('\n').unwrap();
        assert!(raw.chars().all(|c| c >= ' '), "{raw:?}");
        if json {
            let result = tv_testkit::Json::parse(&out.stdout).unwrap();
            let message = result.get("error").and_then(|e| e.get("message")).unwrap();
            assert_eq!(message.as_str(), Some(detail.as_str()));
        }
    }
}
