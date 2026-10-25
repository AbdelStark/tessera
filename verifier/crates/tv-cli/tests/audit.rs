// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tessera-verify audit plan` and `audit report` (02 §2.3, RFC-0016 'Calculators' and 'Audit
//! report').
//!
//! The plan reproduces `docs/perf/examples/audit-plan.json`, the prover's, within `10^{-12}`
//! relative, from `audit-policy.json`, `audit-traffic.json` and the `tessera.cost-model/1` report
//! `tessera bench model --source-defaults --context 700,6000,60000 --json` prints
//! (`tests/data/mu-700-6000-60000.json`); its detection matches without the report. Each refusal
//! of a policy, a profile or a cost model exits 64. The report is built from a ledger and a
//! sample written by `tv-core`, and each failure exits with its code.

use std::fs;
use std::path::{Path, PathBuf};
use tv_codec::DecodeLimits;
use tv_core::audit::{AuditSample, CoinTossTranscript, RequestKey, VerifierLedger};
use tv_core::protocol::{BatchDescriptor, FixedBits, PublicStatement, RequestRef};
use tv_hash::{Domain, hash};
use tv_testkit::Json;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}
fn example(name: &str) -> PathBuf {
    repo().join("docs/perf/examples").join(name)
}
fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/mu-700-6000-60000.json")
}
fn run(args: &[&str]) -> tv_cli::CommandOutput {
    tv_cli::run(args.iter().map(|s| s.to_string()).collect::<Vec<_>>())
}
fn path(p: &Path) -> &str {
    p.to_str().unwrap()
}
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tessera-cli-audit-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}
fn error_code(stdout: &str) -> String {
    let doc = Json::parse(stdout).unwrap();
    doc.get("error")
        .and_then(|e| e.get("code"))
        .and_then(Json::as_str)
        .unwrap()
        .to_string()
}

/// `a` and `b` agree within `10^{-12}` relative.
fn close(a: &Json, b: &Json, what: &str) {
    let (Json::Number(x), Json::Number(y)) = (a, b) else {
        panic!("{what}: {a:?} against {b:?}");
    };
    let (x, y): (f64, f64) = (x.parse().unwrap(), y.parse().unwrap());
    assert!(
        (x - y).abs() <= 1e-12 * x.abs().max(y.abs()),
        "{what}: {x} against {y}"
    );
}

fn plan(mu: bool) -> Json {
    let policy = example("audit-policy.json");
    let traffic = example("audit-traffic.json");
    let fixture = fixture();
    let mut args = vec![
        "audit",
        "plan",
        "--policy",
        path(&policy),
        "--traffic",
        path(&traffic),
        "--json",
    ];
    if mu {
        args.extend(["--mu", path(&fixture)]);
    }
    let out = run(&args);
    assert_eq!(out.exit_code, 0, "{}", out.stdout);
    Json::parse(&out.stdout).unwrap()
}

const DETECTION: [&str; 3] = ["detection", "detection_in_place", "expected_selected"];
const BUCKET_DETECTION: [&str; 5] = [
    "rate",
    "lowest_later_rate",
    "expected_selected",
    "detection",
    "detection_in_place",
];
const BUCKET_INTEGERS: [&str; 4] = ["max_tokens", "requests", "tokens", "violations"];

#[test]
fn the_plan_reproduces_the_provers_example_within_its_tolerance() {
    let expected = Json::parse(&fs::read_to_string(example("audit-plan.json")).unwrap()).unwrap();
    let got = plan(true);
    let Json::Object(members) = &got else {
        panic!("not an object")
    };
    let Json::Object(wanted) = &expected else {
        panic!("not an object")
    };
    let mut keys: Vec<&str> = members.iter().map(|(k, _)| k.as_str()).collect();
    let mut want: Vec<&str> = wanted.iter().map(|(k, _)| k.as_str()).collect();
    keys.sort();
    want.sort();
    assert_eq!(keys, want, "the plan's keys");
    for key in ["schema", "source_defaults"] {
        assert_eq!(got.get(key), expected.get(key), "{key}");
    }
    for key in DETECTION
        .iter()
        .chain(&["eta", "target_overhead", "overhead"])
    {
        close(got.get(key).unwrap(), expected.get(key).unwrap(), key);
    }
    let buckets = got.get("buckets").and_then(Json::as_array).unwrap();
    let wanted = expected.get("buckets").and_then(Json::as_array).unwrap();
    assert_eq!(buckets.len(), wanted.len());
    for (i, (b, w)) in buckets.iter().zip(wanted).enumerate() {
        for key in BUCKET_INTEGERS {
            assert_eq!(b.get(key), w.get(key), "bucket {i}: {key}");
        }
        for key in BUCKET_DETECTION.iter().chain(&["mu", "target_rate"]) {
            close(
                b.get(key).unwrap(),
                w.get(key).unwrap(),
                &format!("bucket {i}: {key}"),
            );
        }
    }
}

#[test]
fn detection_alone_needs_no_cost_model() {
    let expected = Json::parse(&fs::read_to_string(example("audit-plan.json")).unwrap()).unwrap();
    let got = plan(false);
    for key in DETECTION {
        close(got.get(key).unwrap(), expected.get(key).unwrap(), key);
    }
    for key in ["source_defaults", "eta", "target_overhead", "overhead"] {
        assert!(got.get(key).unwrap().is_null(), "{key}");
    }
    let buckets = got.get("buckets").and_then(Json::as_array).unwrap();
    let wanted = expected.get("buckets").and_then(Json::as_array).unwrap();
    for (i, (b, w)) in buckets.iter().zip(wanted).enumerate() {
        for key in BUCKET_INTEGERS {
            assert_eq!(b.get(key), w.get(key), "bucket {i}: {key}");
        }
        for key in BUCKET_DETECTION {
            close(
                b.get(key).unwrap(),
                w.get(key).unwrap(),
                &format!("bucket {i}: {key}"),
            );
        }
        assert!(b.get("mu").unwrap().is_null() && b.get("target_rate").unwrap().is_null());
    }
    // The committed example of the verifier's plan without --mu is this output.
    let committed = fs::read_to_string(example("audit-plan-detection.json")).unwrap();
    assert_eq!(
        Json::parse(&committed).unwrap(),
        got,
        "audit-plan-detection.json is stale"
    );
    // The table names the same figures.
    let policy = example("audit-policy.json");
    let traffic = example("audit-traffic.json");
    let table = run(&[
        "audit",
        "plan",
        "--policy",
        path(&policy),
        "--traffic",
        path(&traffic),
    ]);
    assert_eq!(table.exit_code, 0);
    assert!(
        table
            .stdout
            .contains("detection (padded into the lowest later rate): 0.330217")
    );
    assert!(table.stdout.contains("needs --mu"));
}

/// Runs `audit plan` on the examples, with `policy`, `traffic` or `mu` replaced by `text` when
/// given, and returns the exit code and error code.
fn refused(policy: Option<&str>, traffic: Option<&str>, mu: Option<&str>) -> (i32, String) {
    let dir = scratch("refused");
    let write = |name: &str, text: Option<&str>, default: PathBuf| match text {
        Some(t) => {
            let p = dir.join(name);
            fs::write(&p, t).unwrap();
            p
        }
        None => default,
    };
    let p = write("policy.json", policy, example("audit-policy.json"));
    let t = write("traffic.json", traffic, example("audit-traffic.json"));
    let m = write("mu.json", mu, fixture());
    let out = run(&[
        "audit",
        "plan",
        "--policy",
        path(&p),
        "--traffic",
        path(&t),
        "--mu",
        path(&m),
        "--json",
    ]);
    let code = error_code(&out.stdout);
    fs::remove_dir_all(&dir).unwrap();
    (out.exit_code, code)
}

fn edited(file: &str, from: &str, to: &str) -> String {
    let text = fs::read_to_string(example(file)).unwrap();
    assert!(text.contains(from), "{file}: {from}");
    text.replacen(from, to, 1)
}

#[test]
fn every_policy_traffic_and_cost_model_refusal_is_a_usage_error() {
    let usage = (64, "ERR-USAGE".to_string());
    let policies = [
        // Not the schema, a missing key, an unknown key, a key twice.
        edited(
            "audit-policy.json",
            "tessera.audit-policy/1",
            "tessera.audit-policy/2",
        ),
        edited("audit-policy.json", "\"period_ns\": 86400000000000,", ""),
        edited(
            "audit-policy.json",
            "\"params\"",
            "\"extra\": 1, \"params\"",
        ),
        edited(
            "audit-policy.json",
            "\"params\"",
            "\"period_ns\": 1, \"params\"",
        ),
        // A rate that is no rate, above 1, a number.
        edited("audit-policy.json", "\"0.1\"", "\"0.1e1\""),
        edited("audit-policy.json", "\"0.1\"", "\"1.5\""),
        edited("audit-policy.json", "\"0.1\"", "0.1"),
        // A bucket key missing or unknown.
        edited(
            "audit-policy.json",
            "{\"max_tokens\": 1024, \"rate\": \"0.1\"}",
            "{\"max_tokens\": 1024}",
        ),
        edited(
            "audit-policy.json",
            "\"rate\": \"0.1\"",
            "\"rate\": \"0.1\", \"x\": 0",
        ),
        // Buckets not increasing, the batch bounds broken.
        edited(
            "audit-policy.json",
            "16384, \"rate\": \"0.02\"",
            "1024, \"rate\": \"0.02\"",
        ),
        edited(
            "audit-policy.json",
            "\"max_batch_tokens\": 262144",
            "\"max_batch_tokens\": 32767",
        ),
        edited(
            "audit-policy.json",
            "\"max_batch_tokens\": 262144",
            "\"max_batch_tokens\": 0",
        ),
        // Integers out of range or not integers; a name not printable ASCII or empty.
        edited(
            "audit-policy.json",
            "\"max_batch_tokens\": 262144",
            "\"max_batch_tokens\": 4294967296",
        ),
        edited(
            "audit-policy.json",
            "\"period_ns\": 86400000000000",
            "\"period_ns\": 8.64e13",
        ),
        edited(
            "audit-policy.json",
            "\"period_ns\": 86400000000000",
            "\"period_ns\": -1",
        ),
        edited("audit-policy.json", "\"live-64/1\"", "\"live-64/\\u0007\""),
        edited("audit-policy.json", "\"live-64/1\"", "\"\""),
        // Not JSON.
        "{".to_string(),
    ];
    for p in &policies {
        assert_eq!(refused(Some(p), None, None), usage, "{p}");
    }
    let traffics = [
        // Not one entry per bucket.
        edited("audit-traffic.json", ",\n    {\"requests\": 20000, \"tokens\": 60000, \"violations\": 20}", ""),
        // A mean length above its bucket, at the previous bucket's bound, zero.
        edited("audit-traffic.json", "\"tokens\": 700", "\"tokens\": 1025"),
        edited("audit-traffic.json", "\"tokens\": 6000", "\"tokens\": 1024"),
        edited("audit-traffic.json", "\"tokens\": 700", "\"tokens\": 0"),
        // More violations than requests.
        edited("audit-traffic.json", "\"requests\": 20000", "\"requests\": 19"),
        // No request at all.
        "{\"schema\": \"tessera.audit-traffic/1\", \"buckets\": [{\"requests\": 0, \"tokens\": 700, \"violations\": 0}, {\"requests\": 0, \"tokens\": 6000, \"violations\": 0}, {\"requests\": 0, \"tokens\": 60000, \"violations\": 0}]}".to_string(),
        // An unknown key, a missing one, another schema.
        edited("audit-traffic.json", "\"violations\": 50", "\"violations\": 50, \"x\": 1"),
        edited("audit-traffic.json", ", \"violations\": 50", ""),
        edited("audit-traffic.json", "tessera.audit-traffic/1", "tessera.audit-policy/1"),
    ];
    for t in &traffics {
        assert_eq!(refused(None, Some(t), None), usage, "{t}");
    }
    let fixture = fs::read_to_string(fixture()).unwrap();
    let models = [
        // Contexts that are not the traffic's mean lengths, or not all of them.
        fixture.replacen("\"context\": 700", "\"context\": 701", 1),
        fixture.replacen("\"context\": 60000", "\"context\": 6000", 1),
        // μ, η or the target not positive.
        fixture.replacen("\"total\": 2.826003702051215", "\"total\": 0.0", 1),
        fixture.replacen("\"eta\": 0.3", "\"eta\": -0.3", 1),
        fixture.replacen("\"target_overhead\": 0.1", "\"target_overhead\": 0", 1),
        // Another schema, a missing μ.
        fixture.replacen("tessera.cost-model/1", "tessera.cost-model/2", 1),
        fixture.replacen(
            "\"total\": 8.473179410506082",
            "\"sum\": 8.473179410506082",
            1,
        ),
    ];
    for (i, m) in models.iter().enumerate() {
        assert_ne!(m, &fixture, "model {i} is edited");
        assert_eq!(refused(None, None, Some(m)), usage, "model {i}");
    }
    // A file that cannot be read is the verifier's I/O.
    let out = run(&[
        "audit",
        "plan",
        "--policy",
        "/nonexistent/policy.json",
        "--traffic",
        path(&example("audit-traffic.json")),
        "--json",
    ]);
    assert_eq!(
        (out.exit_code, error_code(&out.stdout).as_str()),
        (20, "ERR-RES-IO")
    );
    // Unknown subcommands and options.
    assert_eq!(run(&["audit", "sample", "--json"]).exit_code, 64);
    assert_eq!(
        run(&[
            "audit",
            "plan",
            "--policy",
            "a",
            "--traffic",
            "b",
            "--eta",
            "1",
            "--json"
        ])
        .exit_code,
        64
    );
    assert_eq!(run(&["audit", "--help"]).exit_code, 0);
    assert_eq!(run(&["audit", "plan", "--help"]).exit_code, 0);
}

// ----- audit report ----------------------------------------------------------------------------

fn toss() -> CoinTossTranscript {
    CoinTossTranscript {
        verifier_commitment: hash(Domain::CoinCommit, &[3; 32]),
        prover_share: [4; 32],
        verifier_share: [3; 32],
    }
}

/// A sample of `keys` over the manifest `[7; 32]`, the policy selecting everything.
fn sample_payload(keys: &[RequestKey], toss: CoinTossTranscript) -> Vec<u8> {
    let mut payload = vec![7; 32];
    payload.extend(86_400u64.to_le_bytes());
    payload.extend(1u64.to_le_bytes());
    payload.extend(u32::MAX.to_le_bytes());
    payload.extend(u64::MAX.to_le_bytes());
    payload.extend(16_384u32.to_le_bytes());
    payload.extend(262_144u32.to_le_bytes());
    payload.extend(604_800u64.to_le_bytes());
    payload.extend(9u64.to_le_bytes());
    payload.extend(b"live-64/1");
    payload.extend(toss.verifier_commitment);
    payload.extend(toss.prover_share);
    payload.extend(toss.verifier_share);
    payload.extend((keys.len() as u64).to_le_bytes());
    for k in keys {
        payload.extend(k.session_id);
        payload.extend(k.request_id.to_le_bytes());
    }
    payload
}

fn keys() -> Vec<RequestKey> {
    (1..=3)
        .map(|r| RequestKey {
            session_id: [1; 16],
            request_id: r,
        })
        .collect()
}

fn statement(requests: &[u64]) -> PublicStatement {
    PublicStatement {
        claim_list_digest: [0; 32],
        weights_root: [0; 32],
        weights_epoch: 0,
        manifest_digest: [7; 32],
        batch: BatchDescriptor {
            requests: requests
                .iter()
                .map(|&r| RequestRef {
                    request_id: r,
                    session_id: [1; 16],
                    prompt_len: 1,
                    output_len: 1,
                })
                .collect(),
        },
        predictor: tv_core::protocol::PredictorSpec::Exact {
            decoding: tv_core::protocol::Decoding::Greedy,
        },
    }
}

/// The ledger of a verifier that derived `samples`, one per manifest, with nothing proven:
/// decoded from its bytes, as a verifier's ledger file holds it after its reveals.
fn deriving(samples: &[&AuditSample]) -> VerifierLedger {
    let mut ids: Vec<([u8; 32], [u8; 32])> = samples
        .iter()
        .map(|s| (*s.manifest_digest(), *s.identity()))
        .collect();
    ids.sort();
    let mut payload = Vec::new();
    for _ in 0..3 {
        payload.extend(0u64.to_le_bytes());
    }
    payload.extend((ids.len() as u64).to_le_bytes());
    for (manifest, id) in &ids {
        payload.extend(manifest);
        payload.extend(id);
    }
    payload.extend(0u64.to_le_bytes());
    VerifierLedger::decode_payload(&payload, &DecodeLimits::MESSAGE).unwrap()
}

/// Writes the sample and a ledger deriving it and accepting `accepted` with `Û` 5 each, and
/// returns the paths.
fn report_inputs(dir: &Path, accepted: &[u64]) -> (PathBuf, PathBuf) {
    let sample =
        AuditSample::decode_payload(&sample_payload(&keys(), toss()), &DecodeLimits::MESSAGE)
            .unwrap();
    let tas = dir.join("sample.tas");
    fs::write(&tas, sample.seal().unwrap()).unwrap();
    let tvl = dir.join("ledger.tvl");
    let mut ledger = deriving(&[&sample]);
    if !accepted.is_empty() {
        ledger
            .record_accepted(
                &sample,
                &statement(accepted),
                &vec![FixedBits(5); accepted.len()],
            )
            .unwrap();
    }
    fs::write(&tvl, ledger.seal().unwrap()).unwrap();
    (tvl, tas)
}

fn report(tvl: &Path, tas: &Path, out: &Path) -> tv_cli::CommandOutput {
    run(&[
        "audit",
        "report",
        "--ledger",
        path(tvl),
        "--sample",
        path(tas),
        "--out",
        path(out),
        "--json",
    ])
}

#[test]
fn the_report_gives_each_selected_request_its_outcome_at_the_deadline() {
    let dir = scratch("report");
    let (tvl, tas) = report_inputs(&dir, &[2]);
    let out = dir.join("report.json");
    let result = report(&tvl, &tas, &out);
    // Two requests are unproven, so the audit did not pass.
    assert_eq!(result.exit_code, 10, "{}", result.stdout);
    assert_eq!(fs::read_to_string(&out).unwrap(), result.stdout);
    let doc = Json::parse(&result.stdout).unwrap();
    assert_eq!(
        doc.get("schema").and_then(Json::as_str),
        Some("tessera.audit-report/1")
    );
    let outcomes = doc.get("outcomes").and_then(Json::as_array).unwrap();
    let kinds: Vec<(u64, &str)> = outcomes
        .iter()
        .map(|o| {
            (
                o.get("request_id").and_then(Json::as_u64).unwrap(),
                o.get("outcome").and_then(Json::as_str).unwrap(),
            )
        })
        .collect();
    assert_eq!(kinds, [(1, "unproven"), (2, "accepted"), (3, "unproven")]);
    assert_eq!(
        outcomes[0].get("unproven").and_then(Json::as_str),
        Some("DeadlineMissed")
    );
    assert_eq!(
        outcomes[1].get("uhat_fixed").and_then(Json::as_u64),
        Some(5)
    );
    let summary = doc.get("summary").unwrap();
    for (key, value) in [
        ("selected", 3),
        ("accepted", 1),
        ("unproven", 2),
        ("max_uhat_fixed", 5),
    ] {
        assert_eq!(
            summary.get(key).and_then(Json::as_u64),
            Some(value),
            "{key}"
        );
    }
    fs::remove_dir_all(&dir).unwrap();

    // Every request accepted: exit 0.
    let dir = scratch("report-all");
    let (tvl, tas) = report_inputs(&dir, &[1, 2, 3]);
    let out = dir.join("report.json");
    let result = report(&tvl, &tas, &out);
    assert_eq!(result.exit_code, 0, "{}", result.stdout);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn every_report_input_failure_exits_with_its_code() {
    let dir = scratch("report-failures");
    let (tvl, tas) = report_inputs(&dir, &[1]);
    let out = dir.join("report.json");
    // A ledger that does not decode.
    let bad = dir.join("bad.tvl");
    let mut bytes = fs::read(&tvl).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    fs::write(&bad, &bytes).unwrap();
    let r = report(&bad, &tas, &out);
    assert_eq!(
        (r.exit_code, error_code(&r.stdout).as_str()),
        (65, "ERR-CODEC-DIGEST")
    );
    // A sample given as the ledger, and a ledger as the sample.
    let r = report(&tas, &tas, &out);
    assert_eq!(
        (r.exit_code, error_code(&r.stdout).as_str()),
        (65, "ERR-CODEC-MAGIC")
    );
    let r = report(&tvl, &tvl, &out);
    assert_eq!(
        (r.exit_code, error_code(&r.stdout).as_str()),
        (65, "ERR-CODEC-MAGIC")
    );
    // A sample whose coin toss does not open.
    let mut broken = toss();
    broken.verifier_share[0] ^= 1;
    let sample =
        AuditSample::decode_payload(&sample_payload(&keys(), broken), &DecodeLimits::MESSAGE)
            .unwrap();
    let bad_toss = dir.join("bad.tas");
    fs::write(&bad_toss, sample.seal().unwrap()).unwrap();
    // Not the verifier's own: the ledger does not derive it (RFC-0067), exit 64.
    let r = report(&tvl, &bad_toss, &out);
    assert_eq!(
        (r.exit_code, error_code(&r.stdout).as_str()),
        (64, "ERR-USAGE")
    );
    // Recorded as derived, as only a hand-made ledger can, since a reveal derives from a toss
    // that opens: its toss is refused, the one path to ERR-PROTO-COINTOSS left.
    let owned = dir.join("owned.tvl");
    fs::write(&owned, deriving(&[&sample]).seal().unwrap()).unwrap();
    let r = report(&owned, &bad_toss, &out);
    assert_eq!(
        (r.exit_code, error_code(&r.stdout).as_str()),
        (20, "ERR-PROTO-COINTOSS")
    );
    // A sample whose toss opens, written by the prover: its own s_V and a policy of rate 0.
    let forged_toss = CoinTossTranscript {
        verifier_commitment: hash(Domain::CoinCommit, &[9; 32]),
        prover_share: [4; 32],
        verifier_share: [9; 32],
    };
    let forged =
        AuditSample::decode_payload(&sample_payload(&[], forged_toss), &DecodeLimits::MESSAGE)
            .unwrap();
    let forged_tas = dir.join("forged.tas");
    fs::write(&forged_tas, forged.seal().unwrap()).unwrap();
    let r = report(&tvl, &forged_tas, &out);
    assert_eq!(
        (r.exit_code, error_code(&r.stdout).as_str()),
        (64, "ERR-USAGE")
    );
    // The ledger as a symbolic link, or under a second hard link: refused, exit 64.
    #[cfg(unix)]
    {
        let link = dir.join("link.tvl");
        std::os::unix::fs::symlink(&tvl, &link).unwrap();
        let r = report(&link, &tas, &out);
        assert_eq!(
            (r.exit_code, error_code(&r.stdout).as_str()),
            (64, "ERR-USAGE")
        );
        let hard = dir.join("hard.tvl");
        fs::hard_link(&tvl, &hard).unwrap();
        let r = report(&tvl, &tas, &out);
        assert_eq!(
            (r.exit_code, error_code(&r.stdout).as_str()),
            (64, "ERR-USAGE")
        );
        fs::remove_file(&hard).unwrap();
    }
    // The output naming an input, and missing options.
    let r = report(&tvl, &tas, &tvl);
    assert_eq!(
        (r.exit_code, error_code(&r.stdout).as_str()),
        (64, "ERR-USAGE")
    );
    let r = run(&[
        "audit",
        "report",
        "--ledger",
        path(&tvl),
        "--sample",
        path(&tas),
        "--json",
    ]);
    assert_eq!(r.exit_code, 64);
    // A missing ledger.
    let r = report(&dir.join("none.tvl"), &tas, &out);
    assert_eq!(
        (r.exit_code, error_code(&r.stdout).as_str()),
        (20, "ERR-RES-IO")
    );
    assert!(!out.exists(), "no failure writes a report");
    fs::remove_dir_all(&dir).unwrap();
}
