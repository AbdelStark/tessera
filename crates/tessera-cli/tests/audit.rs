// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tessera audit share` and `tessera audit sample` end to end (RFC-0016 'Sample derivation',
//! RFC-0063). The reference recorder's core logs three periods, which the test writes as the
//! recorder does. The prover answers a commitment over a manifest, the verifier reveals, and the
//! command's sample must be the one the libraries draw from the same logs and record. Then each
//! input the commands must refuse, with its code:
//! - a period with one before it, but no previous period given, or a previous period that is not
//!   the one before: `ERR-USAGE`;
//! - a log that is not its manifest's: `ERR-CODEC-DIGEST`;
//! - a reveal that does not open the recorded commitment, the shares of another manifest's toss
//!   among them: `ERR-PROTO-COINTOSS`;
//! - a manifest the prover answered no commitment for: `ERR-USAGE`;
//! - another commitment or policy for a manifest already answered: `ERR-PROTO-ORDER`;
//! - a policy file the reader refuses: `ERR-USAGE`;
//! - periods of zero length out of their place: `ERR-USAGE`.

use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;
use tessera_anchor::recorder::{Period, Recorder, WireFrame, encode_log};
use tessera_anchor::{
    Direction, HeapKey, RequestEndFrame, Session, TokenFrame, VerifiedLog, candidates,
};
use tessera_codec::{DecodeLimits, Secret, decode, envelope, schema};
use tessera_protocol::{AuditSample, RateBucket, RequestKey, ShareRecord, coin_commitment};

fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tessera-audit-{test}-{}", std::process::id()));
    fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

fn tessera<S: AsRef<OsStr>>(args: &[S]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tessera"))
        .args(args)
        .output()
        .expect("the tessera binary runs")
}

/// The exit status and the error code of a refused run, from its `--json` error object.
fn refused<S: AsRef<OsStr>>(args: &[S]) -> (i32, String) {
    let mut args: Vec<OsString> = args.iter().map(|a| a.as_ref().to_owned()).collect();
    args.push("--json".into());
    let output = tessera(&args);
    let object: Value =
        serde_json::from_slice(&output.stdout).expect("one JSON error object on stdout");
    (
        output.status.code().expect("an exit status"),
        object["error"]["code"].as_str().expect("a code").to_owned(),
    )
}

/// Three periods of one log: a session whose request 1 ends in period 0, request 2 starts in
/// period 0 and ends in period 1, request 3 starts in period 1 and never ends, and a stray end
/// in period 2. Each period's files are written as the recorder writes them.
fn three_periods(dir: &Path) -> Vec<(PathBuf, PathBuf, Period)> {
    let seed = Secret::new(HeapKey::from_box(Box::new([0x51; 32])));
    let mut recorder = Recorder::new(seed, [4; 16], 1_000);
    let (mut session, open, _) = Session::open().expect("a session");
    let id = session.session_id();
    let mut t = 1_000;
    let mut tick = || {
        t += 10;
        t
    };
    let mut seal = |request: u64, direction: Direction, index: u32, n: usize| {
        let frame = TokenFrame::new(request, direction, index, Secret::new(vec![7u32; n]));
        WireFrame::Sealed(session.seal(&frame).expect("sealed"))
    };
    let end = |request_id: u64| {
        WireFrame::End(RequestEndFrame {
            session_id: id,
            request_id,
        })
    };
    let mut periods = Vec::new();
    recorder.record(&WireFrame::Open(open), tick());
    for frame in [
        seal(1, Direction::In, 0, 5),
        seal(1, Direction::Out, 0, 9),
        end(1),
        seal(2, Direction::In, 0, 40),
    ] {
        recorder.record(&frame, tick());
    }
    periods.push(recorder.rotate(tick()).expect("period 0"));
    for frame in [
        seal(2, Direction::Out, 0, 60),
        end(2),
        seal(3, Direction::In, 0, 3),
    ] {
        recorder.record(&frame, tick());
    }
    periods.push(recorder.rotate(tick()).expect("period 1"));
    recorder.record(&end(99), tick());
    periods.push(recorder.rotate(tick()).expect("period 2"));
    periods
        .into_iter()
        .map(|period| {
            let stem = format!("{:020}", period.manifest.body.period_start_ns);
            let tlm = dir.join(format!("{stem}.tlm"));
            let tdl = dir.join(format!("{stem}.tdl"));
            fs::write(
                &tlm,
                envelope::seal(&schema::LOG_MANIFEST, &period.manifest),
            )
            .expect("the manifest");
            fs::write(&tdl, encode_log(&period.entries)).expect("the log");
            (tlm, tdl, period)
        })
        .collect()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

const POLICY: &str = r#"{
  "schema": "tessera.audit-policy/1",
  "period_ns": 86400000000000,
  "buckets": [
    {"max_tokens": 16, "rate": "0.5"},
    {"max_tokens": 64, "rate": "1"}
  ],
  "min_batch_tokens": 16384,
  "max_batch_tokens": 262144,
  "proof_deadline_ns": 604800000000000,
  "params": "live-64/1"
}"#;

/// Runs a command that must succeed, and returns its stdout, trimmed.
fn ok<S: AsRef<OsStr>>(args: &[S]) -> String {
    let output = tessera(args);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("UTF-8")
        .trim()
        .to_owned()
}

/// `tessera audit share`'s arguments.
fn share_args(manifest: &Path, policy: &Path, commitment: &str, state: &Path) -> Vec<OsString> {
    vec![
        "audit".into(),
        "share".into(),
        "--manifest".into(),
        manifest.into(),
        "--policy".into(),
        policy.into(),
        "--commitment".into(),
        commitment.into(),
        "--state".into(),
        state.into(),
    ]
}

/// `tessera audit sample`'s arguments.
fn sample_args(
    manifest: &Path,
    log: &Path,
    previous: Option<(&Path, &Path)>,
    between: &[&Path],
    state: &Path,
    verifier_share: &str,
    out: &Path,
) -> Vec<OsString> {
    let mut args: Vec<OsString> = vec![
        "audit".into(),
        "sample".into(),
        "--manifest".into(),
        manifest.into(),
        "--log".into(),
        log.into(),
        "--state".into(),
        state.into(),
        "--verifier-share".into(),
        verifier_share.into(),
        "--out".into(),
        out.into(),
    ];
    if let Some((m, l)) = previous {
        args.extend([
            "--previous-manifest".into(),
            m.into(),
            "--previous-log".into(),
            l.into(),
        ]);
    }
    for b in between {
        args.extend(["--between".into(), (*b).into()]);
    }
    args
}

fn read_sample(path: &Path) -> AuditSample {
    envelope::open(
        &fs::read(path).expect("the sample"),
        &schema::AUDIT_SAMPLE,
        DecodeLimits::MESSAGE,
    )
    .expect("an AuditSample envelope")
}

/// The record the prover kept for a manifest.
fn record(state: &Path, manifest: &Period) -> ShareRecord {
    let path = state.join(format!("{}.share", hex(&manifest.manifest.identity())));
    decode(&fs::read(path).expect("the record"), DecodeLimits::MESSAGE).expect("a share record")
}

/// A candidate as the tests list it: its request, its tokens and whether it ended.
type Listed = (u64, u64, bool);

/// The candidates of `current`, after `previous`, as `(request_id, tokens, ended)`, and as
/// `AuditSample::draw` takes them.
fn listed(previous: Option<&Period>, current: &Period) -> (Vec<Listed>, Vec<(RequestKey, u64)>) {
    let verified = |p: &Period| VerifiedLog::check(&p.manifest, p.entries.clone()).expect("a log");
    let previous = previous.map(verified);
    let found = candidates(previous.as_ref(), &verified(current));
    (
        found
            .iter()
            .map(|c| (c.request_id, c.tokens, c.ended))
            .collect(),
        found
            .iter()
            .map(|c| {
                (
                    RequestKey {
                        session_id: c.session_id,
                        request_id: c.request_id,
                    },
                    c.tokens,
                )
            })
            .collect(),
    )
}

#[test]
fn the_sample_is_the_one_the_logs_record_and_reveal_give() {
    let dir = scratch("sample");
    let periods = three_periods(&dir);
    let (t0, d0, p0) = &periods[0];
    let (t1, d1, p1) = &periods[1];
    let (t2, d2, p2) = &periods[2];
    let policy = dir.join("policy.json");
    fs::write(&policy, POLICY).expect("the policy");
    let state = dir.join("state");
    let out = dir.join("sample.tas");
    let verifier_share = [0x0f; 32];
    let (c, v) = (hex(&coin_commitment(&verifier_share)), hex(&verifier_share));

    // Period 1, whose previous period is period 0.
    let s_p = ok(&share_args(t1, &policy, &c, &state));
    ok(&sample_args(t1, d1, Some((t0, d0)), &[], &state, &v, &out));
    let written = read_sample(&out);
    let kept = record(&state, p1);
    assert_eq!(hex(&kept.prover_share), s_p);
    // Request 2 ended here, over 40 + 60 tokens; request 3 is not a candidate yet; request 1
    // ended in period 0.
    let (found, given) = listed(Some(p0), p1);
    assert_eq!(found, [(2, 100, true)]);
    assert_eq!(
        written,
        kept.draw(verifier_share, given)
            .expect("the libraries' sample")
    );
    assert_eq!(hex(&written.coin_toss().prover_share), s_p);
    // 100 tokens is past both buckets: selected whatever the seed.
    assert_eq!(written.selected().len(), 1);
    assert_eq!(
        written.policy().buckets(),
        [
            RateBucket {
                max_tokens: 16,
                threshold: 1 << 63
            },
            RateBucket {
                max_tokens: 64,
                threshold: u64::MAX
            }
        ]
    );

    // Period 2: request 3, framed in period 1 and never ended, is an unterminated candidate.
    ok(&share_args(t2, &policy, &c, &state));
    ok(&sample_args(t2, d2, Some((t1, d1)), &[], &state, &v, &out));
    let (found, given) = listed(Some(p1), p2);
    assert_eq!(found, [(3, 3, false), (99, 0, true)]);
    assert_eq!(
        read_sample(&out),
        record(&state, p2)
            .draw(verifier_share, given)
            .expect("the libraries' sample")
    );

    // Period 0 is the log's first: it needs no previous period. Its toss is under the committed
    // example policy, `docs/perf/examples/audit-policy.json`.
    let example = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/perf/examples/audit-policy.json");
    ok(&share_args(t0, &example, &c, &state));
    ok(&sample_args(t0, d0, None, &[], &state, &v, &out));
    let threshold = |rate| RateBucket::threshold_of(rate).expect("a rate");
    let rates: Vec<(u32, u64)> = read_sample(&out)
        .policy()
        .buckets()
        .iter()
        .map(|b| (b.max_tokens, b.threshold))
        .collect();
    assert_eq!(
        rates,
        [
            (1024, threshold("0.1")),
            (16_384, threshold("0.02")),
            (131_072, threshold("0.004"))
        ]
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn inputs_that_cannot_give_the_sample_are_refused_with_their_code() {
    let dir = scratch("refused");
    let periods = three_periods(&dir);
    let (t0, d0, _) = &periods[0];
    let (t1, d1, _) = &periods[1];
    let (t2, d2, _) = &periods[2];
    let good = dir.join("policy.json");
    fs::write(&good, POLICY).expect("the policy");
    let state = dir.join("state");
    let out = dir.join("sample.tas");
    let (v1, v0) = ([0x0f; 32], [0x1e; 32]);
    let (c1, c0) = (hex(&coin_commitment(&v1)), hex(&coin_commitment(&v0)));
    let (v1, v0) = (hex(&v1), hex(&v0));
    ok(&share_args(t1, &good, &c1, &state));
    let usage = (64, "ERR-USAGE".to_owned());
    // No previous period for period 1.
    assert_eq!(
        refused(&sample_args(t1, d1, None, &[], &state, &v1, &out)),
        usage
    );
    // Period 0's log under period 1's manifest.
    assert_eq!(
        refused(&sample_args(t1, d0, Some((t0, d0)), &[], &state, &v1, &out)),
        (65, "ERR-CODEC-DIGEST".to_owned())
    );
    // A reveal that does not open the recorded commitment.
    let cointoss = (20, "ERR-PROTO-COINTOSS".to_owned());
    let wrong = hex(&[0x0e; 32]);
    assert_eq!(
        refused(&sample_args(
            t1,
            d1,
            Some((t0, d0)),
            &[],
            &state,
            &wrong,
            &out
        )),
        cointoss
    );
    // The shares of another manifest's toss: the verifier knows `s_V` for period 0's commitment,
    // and reveals it for period 1, whose record holds another commitment.
    ok(&share_args(t0, &good, &c0, &state));
    assert_eq!(
        refused(&sample_args(t1, d1, Some((t0, d0)), &[], &state, &v0, &out)),
        cointoss
    );
    // A manifest the prover answered no commitment for, and a record of another manifest kept
    // under its name.
    assert_eq!(
        refused(&sample_args(t2, d2, Some((t1, d1)), &[], &state, &v1, &out)),
        usage
    );
    // Period 0 is not the period before period 2, which now has a record, so only the chain
    // refuses it.
    ok(&share_args(t2, &good, &c1, &state));
    assert_eq!(
        refused(&sample_args(t2, d2, Some((t0, d0)), &[], &state, &v1, &out)),
        usage
    );
    ok(&sample_args(t2, d2, Some((t1, d1)), &[], &state, &v1, &out));
    let _ = fs::remove_file(&out);
    let elsewhere = dir.join("elsewhere");
    fs::create_dir_all(&elsewhere).expect("a state");
    let name = |p: &Period| format!("{}.share", hex(&p.manifest.identity()));
    fs::copy(
        state.join(name(&periods[0].2)),
        elsewhere.join(name(&periods[1].2)),
    )
    .expect("a copied record");
    assert_eq!(
        refused(&sample_args(
            t1,
            d1,
            Some((t0, d0)),
            &[],
            &elsewhere,
            &v0,
            &out
        )),
        usage
    );
    // Policy files the reader refuses, and no record kept for them.
    for (name, text) in [
        (
            "unknown-key",
            POLICY.replace("\"params\"", "\"extra\": 1, \"params\""),
        ),
        ("missing-key", POLICY.replace("\"period_ns\"", "\"period\"")),
        (
            "rate-form",
            POLICY.replace("\"0.5\"", "\"0.50000000000000000000\""),
        ),
        ("rate-number", POLICY.replace("\"0.5\"", "0.5")),
        (
            "buckets-order",
            POLICY.replace("\"max_tokens\": 64", "\"max_tokens\": 16"),
        ),
        ("batch-bounds", POLICY.replace("262144", "32767")),
        ("schema", POLICY.replace("audit-policy/1", "audit-policy/2")),
        ("params-empty", POLICY.replace("\"live-64/1\"", "\"\"")),
    ] {
        let path = dir.join(format!("{name}.json"));
        fs::write(&path, text).expect("a policy");
        assert_eq!(
            refused(&share_args(t2, &path, &c1, &state)),
            usage,
            "{name}"
        );
    }
    assert!(
        fs::read_dir(&state).expect("the state").count() == 3,
        "only the three answers made are kept"
    );
    assert!(!out.exists(), "a refused run writes no sample");
    let _ = fs::remove_dir_all(&dir);
}

/// Periods of zero length that the chain may hold, and the inputs around them the command must
/// refuse: a zero-length previous period, one of non-zero length or with entries between, a
/// zero-length period to sample, and a period that is not the log's first given as though it
/// were. Each would otherwise give another sample than RFC-0063's.
#[test]
fn zero_length_periods_are_passed_through_never_sampled_or_drawn_from() {
    use tessera_anchor::{LogManifest, ManifestBody};
    let dir = scratch("zero");
    let periods = three_periods(&dir);
    let seed = Secret::new(HeapKey::from_box(Box::new([0x51; 32])));
    let (t0, d0, p0) = &periods[0];
    let (_, d1, p1) = &periods[1];
    let base = p0.manifest.body.clone();
    let write = |name: &str, manifest: &LogManifest| {
        let path = dir.join(name);
        fs::write(&path, envelope::seal(&schema::LOG_MANIFEST, manifest)).expect("written");
        path
    };
    let sign = |body: ManifestBody| body.sign(&seed).expect("signed");
    let empty_log = dir.join("empty.tdl");
    fs::write(&empty_log, b"").expect("an empty log");
    // Z: zero length after period 0; P1z: period 1 after Z.
    let z = sign(ManifestBody {
        period_start_ns: base.period_end_ns,
        first_seq: p1.manifest.body.first_seq,
        entry_count: 0,
        entries_root: tessera_anchor::merkle::root(&[]),
        prev_manifest: p0.manifest.identity(),
        ..base.clone()
    });
    let p1z = sign(ManifestBody {
        prev_manifest: z.identity(),
        ..p1.manifest.body.clone()
    });
    // X: an empty period of non-zero length after period 0; P1x: period 1 after it.
    let x = sign(ManifestBody {
        period_end_ns: base.period_end_ns + 5,
        ..z.body.clone()
    });
    let p1x = sign(ManifestBody {
        period_start_ns: x.body.period_end_ns,
        prev_manifest: x.identity(),
        ..p1.manifest.body.clone()
    });
    // W: zero length but one entry; P1w: after it, its entries shifted by one.
    let w = sign(ManifestBody {
        entry_count: 1,
        ..z.body.clone()
    });
    let p1w = sign(ManifestBody {
        first_seq: p1.manifest.body.first_seq + 1,
        prev_manifest: w.identity(),
        ..p1.manifest.body.clone()
    });
    // Late: names no previous manifest but does not start at `seq` 0.
    let late = sign(ManifestBody {
        first_seq: 5,
        entry_count: 0,
        entries_root: tessera_anchor::merkle::root(&[]),
        ..base.clone()
    });
    let (z_tlm, p1z_tlm) = (write("z.tlm", &z), write("p1z.tlm", &p1z));
    let (x_tlm, p1x_tlm) = (write("x.tlm", &x), write("p1x.tlm", &p1x));
    let (w_tlm, p1w_tlm) = (write("w.tlm", &w), write("p1w.tlm", &p1w));
    let late_tlm = write("late.tlm", &late);
    let policy = dir.join("policy.json");
    fs::write(&policy, POLICY).expect("the policy");
    let state = dir.join("state");
    let out = dir.join("sample.tas");
    let verifier_share = [0x0f; 32];
    let (c, v) = (hex(&coin_commitment(&verifier_share)), hex(&verifier_share));
    for manifest in [&p1z_tlm, &p1x_tlm, &p1w_tlm, &z_tlm, &late_tlm] {
        ok(&share_args(manifest, &policy, &c, &state));
    }
    let usage = (64, "ERR-USAGE".to_owned());
    // Through Z, period 1 draws from period 0.
    ok(&sample_args(
        &p1z_tlm,
        d1,
        Some((t0, d0)),
        &[&z_tlm],
        &state,
        &v,
        &out,
    ));
    assert_eq!(*read_sample(&out).manifest_digest(), p1z.identity());
    assert_eq!(
        read_sample(&out).selected().len(),
        1,
        "request 2, past every bucket"
    );
    // Z as the previous period: it has no log to draw from, and period 0's unterminated request
    // would escape.
    assert_eq!(
        refused(&sample_args(
            &p1z_tlm,
            d1,
            Some((&z_tlm, &empty_log)),
            &[],
            &state,
            &v,
            &out
        )),
        usage
    );
    // Z itself: no candidates, so no sample.
    assert_eq!(
        refused(&sample_args(
            &z_tlm,
            &empty_log,
            Some((t0, d0)),
            &[],
            &state,
            &v,
            &out
        )),
        usage
    );
    // X between, which links but has length: period 0 is not the previous period.
    assert_eq!(
        refused(&sample_args(
            &p1x_tlm,
            d1,
            Some((t0, d0)),
            &[&x_tlm],
            &state,
            &v,
            &out
        )),
        usage
    );
    // W between, of zero length but with an entry.
    assert_eq!(
        refused(&sample_args(
            &p1w_tlm,
            d1,
            Some((t0, d0)),
            &[&w_tlm],
            &state,
            &v,
            &out
        )),
        usage
    );
    // Late, given as though it were the log's first.
    assert_eq!(
        refused(&sample_args(
            &late_tlm,
            &empty_log,
            None,
            &[],
            &state,
            &v,
            &out
        )),
        usage
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_prover_answers_one_commitment_and_policy_per_manifest() {
    let dir = scratch("share");
    let periods = three_periods(&dir);
    let (t0, _, _) = &periods[0];
    let (t1, _, p1) = &periods[1];
    let policy = dir.join("policy.json");
    fs::write(&policy, POLICY).expect("the policy");
    let other_policy = dir.join("other.json");
    fs::write(&other_policy, POLICY.replace("\"0.5\"", "\"0.25\"")).expect("a policy");
    let state = dir.join("state");
    let c = hex(&coin_commitment(&[0x2d; 32]));
    let other = hex(&coin_commitment(&[0x2e; 32]));
    let s_p = ok(&share_args(t1, &policy, &c, &state));
    assert_eq!(s_p.len(), 64);
    // Kept before it was printed, under the manifest's digest, with the policy.
    let kept = record(&state, p1);
    assert_eq!(hex(&kept.prover_share), s_p);
    assert_eq!(hex(&kept.verifier_commitment), c);
    let path = state.join(format!("{}.share", hex(&p1.manifest.identity())));
    let bytes = fs::read(&path).expect("the record");
    // The same commitment and policy again: the same share; with --json, one object.
    assert_eq!(ok(&share_args(t1, &policy, &c, &state)), s_p);
    let mut json = share_args(t1, &policy, &c, &state);
    json.push("--json".into());
    let object: Value = serde_json::from_str(&ok(&json)).expect("JSON");
    assert_eq!(object["command"], "audit share");
    assert_eq!(object["prover_share"], s_p.as_str());
    assert_eq!(object["verifier_commitment"], c.as_str());
    // Another commitment, or another policy: refused, and the record is unchanged.
    let order = (20, "ERR-PROTO-ORDER".to_owned());
    assert_eq!(refused(&share_args(t1, &policy, &other, &state)), order);
    assert_eq!(refused(&share_args(t1, &other_policy, &c, &state)), order);
    assert_eq!(fs::read(&path).expect("the record"), bytes);
    // No temporary file is left behind.
    let names: Vec<String> = fs::read_dir(&state)
        .expect("the state")
        .map(|e| {
            e.expect("an entry")
                .file_name()
                .into_string()
                .expect("a name")
        })
        .collect();
    assert_eq!(names.len(), 1, "{names:?}");
    // Another manifest has its own toss.
    assert_ne!(ok(&share_args(t0, &policy, &other, &state)), s_p);
    let _ = fs::remove_dir_all(&dir);
}
