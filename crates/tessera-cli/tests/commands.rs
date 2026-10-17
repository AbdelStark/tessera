// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The `tessera` command tree of 02 §2.2, end to end through the built binary.
//!
//! - **The tree.** `tessera --help` lists every command of 02 §2.2, and each has its own help.
//! - **Not yet.** Every command but `bench model` checks the envelope of each TCE artifact it
//!   reads, then exits 64 (`ERR-USAGE`), naming the issue that implements it, and writes nothing.
//! - **Exit statuses** (02 §2.4). A malformed artifact exits 65 with its fault's code: bytes
//!   that are no envelope, another schema's envelope, a truncated one, a corrupted digest,
//!   trailing bytes. An artifact that cannot be read exits 20. A malformed argument exits 64.
//!   With `--json`, stdout holds one error object naming the command and the code.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;
use tessera_codec::envelope;
use tessera_codec::schema::{self, Schema};

/// The commands of 02 §2.2.
const COMMANDS: [&str; 11] = [
    "compile",
    "commit-weights",
    "weights refresh",
    "prove",
    "audit sample",
    "audit batch",
    "audit plan",
    "params check",
    "bench unit-costs",
    "bench model",
    "recorder ref",
];

fn tessera<S: AsRef<std::ffi::OsStr>>(args: &[S]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tessera"))
        .args(args)
        .output()
        .expect("the tessera binary runs")
}

fn text(bytes: &[u8]) -> &str {
    std::str::from_utf8(bytes).expect("UTF-8")
}

/// The exit status, and the error object printed with `--json`.
fn refused<S: AsRef<std::ffi::OsStr>>(args: &[S]) -> (i32, Value) {
    let mut args: Vec<std::ffi::OsString> = args.iter().map(|a| a.as_ref().to_owned()).collect();
    args.push("--json".into());
    let output = tessera(&args);
    let object = serde_json::from_str(text(&output.stdout)).expect("one JSON object");
    (output.status.code().expect("an exit status"), object)
}

/// A directory of its own for one test.
fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tessera-commands-{test}-{}", std::process::id()));
    fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

/// A valid envelope of `schema`, holding a byte: what the envelope checks see of an artifact.
fn sealed(dir: &Path, name: &str, schema: &Schema) -> PathBuf {
    let path = dir.join(name);
    fs::write(&path, envelope::seal(schema, &0u8)).expect("the envelope");
    path
}

/// A conformance model, a valid `IntegerModel` artifact.
fn model() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/models/tiny-2-servable.d")
}

#[test]
fn the_help_lists_every_command_of_02_2() {
    let output = tessera(&["--help"]);
    assert_eq!(output.status.code(), Some(0));
    let help = text(&output.stdout);
    let listed: Vec<String> = help
        .lines()
        .skip_while(|line| *line != "Every command:")
        .skip(1)
        .map(|line| {
            let words: Vec<&str> = line.split_whitespace().collect();
            let end = words.iter().position(|w| w.starts_with(char::is_uppercase));
            words[1..end.unwrap_or(words.len())].join(" ")
        })
        .collect();
    assert_eq!(listed, COMMANDS);
    for command in COMMANDS {
        let mut args: Vec<&str> = command.split(' ').collect();
        args.push("--help");
        let output = tessera(&args);
        assert_eq!(output.status.code(), Some(0), "{command}");
        assert!(text(&output.stdout).contains("Usage: tessera"), "{command}");
    }
}

#[test]
fn every_command_checks_its_artifacts_then_names_its_issue() {
    let dir = scratch("pending");
    let model = model();
    let plan = sealed(&dir, "plan.tep", &schema::EXECUTION_PLAN);
    let weights = sealed(&dir, "weights.twc", &schema::WEIGHT_COMMITMENT_STATE);
    let sample = sealed(&dir, "sample.tas", &schema::AUDIT_SAMPLE);
    let manifest = sealed(&dir, "manifest.tlm", &schema::LOG_MANIFEST);
    let out = dir.join("out");
    let (m, p, w, s, l, d, o) = (
        model.as_os_str(),
        plan.as_os_str(),
        weights.as_os_str(),
        sample.as_os_str(),
        manifest.as_os_str(),
        dir.as_os_str(),
        out.as_os_str(),
    );
    let cases: [(&str, Vec<&std::ffi::OsStr>, u32); 11] = [
        (
            "compile",
            vec![
                "--model".as_ref(),
                m,
                "--params".as_ref(),
                "test-small".as_ref(),
                "--out-dir".as_ref(),
                o,
            ],
            87,
        ),
        (
            "commit-weights",
            vec![
                "--model".as_ref(),
                m,
                "--plan".as_ref(),
                p,
                "--out".as_ref(),
                o,
            ],
            111,
        ),
        (
            "weights refresh",
            vec!["--weights".as_ref(), w, "--out".as_ref(), o],
            153,
        ),
        (
            "prove",
            vec![
                "--plan".as_ref(),
                p,
                "--model".as_ref(),
                m,
                "--weights".as_ref(),
                w,
                "--sample".as_ref(),
                s,
                "--transcripts".as_ref(),
                d,
                "--out".as_ref(),
                o,
            ],
            157,
        ),
        (
            "audit sample",
            vec![
                "--manifest".as_ref(),
                l,
                "--seed".as_ref(),
                o,
                "--rate".as_ref(),
                "0.01".as_ref(),
                "--out".as_ref(),
                o,
            ],
            157,
        ),
        (
            "audit batch",
            vec!["--sample".as_ref(), s, "--out-dir".as_ref(), o],
            157,
        ),
        (
            "audit plan",
            vec![
                "--policy".as_ref(),
                o,
                "--traffic".as_ref(),
                o,
                "--plan".as_ref(),
                p,
            ],
            157,
        ),
        (
            "params check",
            vec!["--params".as_ref(), "ni-100".as_ref()],
            103,
        ),
        (
            "bench unit-costs",
            vec!["--device".as_ref(), "0".as_ref(), "--out".as_ref(), o],
            49,
        ),
        ("bench model", vec!["--plan".as_ref(), p], 87),
        (
            "recorder ref",
            vec![
                "--listen".as_ref(),
                "127.0.0.1:7070".as_ref(),
                "--log-dir".as_ref(),
                d,
            ],
            139,
        ),
    ];
    for (command, flags, issue) in cases {
        let mut args: Vec<&std::ffi::OsStr> = command.split(' ').map(AsRef::as_ref).collect();
        args.extend(flags);
        let output = tessera(&args);
        let what = match command {
            "bench model" => "bench model --plan",
            other => other,
        };
        let message = format!("tessera {what} is not implemented yet (#{issue})");
        assert_eq!(output.status.code(), Some(64), "{command}");
        assert_eq!(
            text(&output.stderr),
            format!("error: ERR-USAGE: {message}\n")
        );
        assert_eq!(text(&output.stdout), "", "{command}");
        let (status, error) = refused(&args);
        assert_eq!(status, 64);
        assert_eq!(error["command"], command);
        assert_eq!(error["error"]["code"], "ERR-USAGE");
        assert_eq!(error["error"]["message"], message.as_str());
    }
    assert!(!out.exists(), "nothing was written");
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn a_malformed_artifact_exits_65_with_its_faults_code_and_an_unreadable_one_20() {
    let dir = scratch("malformed");
    let valid = fs::read(model()).expect("the model");
    let mut corrupted = valid.clone();
    *corrupted.last_mut().expect("a digest") ^= 1;
    let mut trailing = valid.clone();
    trailing.push(0);
    let cases: [(&str, Vec<u8>, &str); 5] = [
        ("garbage.d", b"not an artifact".to_vec(), "ERR-CODEC-MAGIC"),
        (
            "tokens.d",
            envelope::seal(&schema::TOKEN_FILE, &0u8),
            "ERR-CODEC-MAGIC",
        ),
        (
            "short.d",
            valid[..valid.len() - 1].to_vec(),
            "ERR-CODEC-TRUNCATED",
        ),
        ("corrupted.d", corrupted, "ERR-CODEC-DIGEST"),
        ("trailing.d", trailing, "ERR-CODEC-NONCANONICAL"),
    ];
    let compile = |model: &Path| {
        let out = dir.join("out");
        refused(&[
            "compile".as_ref(),
            "--model".as_ref(),
            model.as_os_str(),
            "--params".as_ref(),
            "test-small".as_ref(),
            "--out-dir".as_ref(),
            out.as_os_str(),
        ])
    };
    for (name, bytes, code) in cases {
        let path = dir.join(name);
        fs::write(&path, bytes).expect("the artifact");
        let (status, error) = compile(&path);
        assert_eq!(
            (status, error["error"]["code"].as_str()),
            (65, Some(code)),
            "{name}"
        );
        let message = error["error"]["message"].as_str().expect("a message");
        assert!(
            message.ends_with("is not a valid IntegerModel envelope"),
            "{message}"
        );
    }
    let (status, error) = compile(&dir.join("absent.d"));
    assert_eq!(
        (status, error["error"]["code"].as_str()),
        (20, Some("ERR-RES-IO"))
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn a_malformed_argument_exits_64() {
    let dir = scratch("arguments");
    let model = model();
    let plan = sealed(&dir, "plan.tep", &schema::EXECUTION_PLAN);
    let (m, p, d) = (
        model.to_str().expect("UTF-8"),
        plan.to_str().expect("UTF-8"),
        dir.to_str().expect("UTF-8"),
    );
    let prove = [
        "prove",
        "--plan",
        p,
        "--model",
        m,
        "--weights",
        p,
        "--sample",
        p,
        "--transcripts",
        d,
    ];
    let with = |extra: &[&'static str]| -> Vec<String> {
        prove.iter().chain(extra).map(ToString::to_string).collect()
    };
    let sample = |rate: &str| -> Vec<String> {
        [
            "audit",
            "sample",
            "--manifest",
            p,
            "--seed",
            p,
            "--rate",
            rate,
            "--out",
            d,
        ]
        .map(ToString::to_string)
        .to_vec()
    };
    let strings = |args: &[&str]| -> Vec<String> { args.iter().map(ToString::to_string).collect() };
    for args in [
        strings(&["compile", "--model", m, "--params", "bogus", "--out-dir", d]),
        with(&[]),
        with(&["--live", "localhost:1", "--out", "proof.tnp"]),
        with(&["--out", "proof.tnp", "--memory-budget", "0"]),
        with(&["--out", "proof.tnp", "--backend", "tpu"]),
        sample("0"),
        sample("1.5"),
        strings(&["recorder", "ref", "--listen", "nowhere", "--log-dir", d]),
        strings(&["bench", "model", "--plan", p, "--source-defaults"]),
        strings(&["frobnicate"]),
    ] {
        let output = tessera(&args);
        assert_eq!(output.status.code(), Some(64), "{args:?}");
        assert_eq!(text(&output.stdout), "", "{args:?}");
        let (status, error) = refused(&args);
        assert_eq!(status, 64);
        assert_eq!(error["command"], Value::Null, "{args:?}");
        assert_eq!(error["error"]["code"], "ERR-USAGE", "{args:?}");
        assert!(
            !error["error"]["message"]
                .as_str()
                .expect("a message")
                .is_empty()
        );
    }
    let _ = fs::remove_dir_all(dir);
}
