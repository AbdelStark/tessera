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
        for (i, line) in parsed.iter().enumerate() {
            assert_event(line);
            if i < usize::from(last) {
                assert_eq!(line.event, "tessera.tv_cli.check_completed");
                assert_eq!(line.field("step").unwrap().int(), i as u64 + 1);
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
