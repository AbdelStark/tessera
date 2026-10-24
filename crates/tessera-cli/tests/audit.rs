// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tessera audit sample` end to end (RFC-0016 'Sample derivation', RFC-0063). The reference
//! recorder's core logs three periods, the test writes them as the recorder does, and the
//! command's sample must be the one the libraries draw from the same logs, coin toss and policy.
//! Then each input the command must refuse, with its code:
//! - a period with one before it, but no previous period given: `ERR-USAGE`;
//! - a previous period that is not the one before: `ERR-USAGE`;
//! - a log that is not its manifest's: `ERR-CODEC-DIGEST`;
//! - a verifier share that does not open its commitment: `ERR-PROTO-COINTOSS`;
//! - a policy file with an unknown key, a rate of another form, or buckets out of order:
//!   `ERR-USAGE`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;
use tessera_anchor::recorder::{Period, Recorder, WireFrame, encode_log};
use tessera_anchor::{
    Direction, HeapKey, RequestEndFrame, Session, TokenFrame, VerifiedLog, candidates,
};
use tessera_codec::{DecodeLimits, Secret, envelope, schema};
use tessera_protocol::{AuditSample, CoinTossTranscript, RateBucket, RequestKey, coin_commitment};

fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tessera-audit-{test}-{}", std::process::id()));
    fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

fn tessera(args: &[&std::ffi::OsStr]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tessera"))
        .args(args)
        .output()
        .expect("the tessera binary runs")
}

/// The exit status and the error code of a refused run, from its `--json` error object.
fn refused(args: &[&std::ffi::OsStr]) -> (i32, String) {
    let mut args = args.to_vec();
    args.push("--json".as_ref());
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

#[test]
fn the_sample_is_the_one_the_logs_coin_toss_and_policy_give() {
    let dir = scratch("sample");
    let periods = three_periods(&dir);
    let policy = dir.join("policy.json");
    fs::write(&policy, POLICY).expect("the policy");
    let verifier_share = [0x0f; 32];
    let coin_toss = CoinTossTranscript {
        verifier_commitment: coin_commitment(&verifier_share),
        prover_share: [0xf0; 32],
        verifier_share,
    };
    let (c, p, v) = (
        hex(&coin_toss.verifier_commitment),
        hex(&coin_toss.prover_share),
        hex(&coin_toss.verifier_share),
    );
    let out = dir.join("sample.tas");
    // Period 1, whose previous period is period 0.
    let (prev_tlm, prev_tdl, prev) = &periods[0];
    let (tlm, tdl, current) = &periods[1];
    let base: Vec<&std::ffi::OsStr> = vec![
        "audit".as_ref(),
        "sample".as_ref(),
        "--manifest".as_ref(),
        tlm.as_os_str(),
        "--log".as_ref(),
        tdl.as_os_str(),
        "--policy".as_ref(),
        policy.as_os_str(),
        "--verifier-commitment".as_ref(),
        c.as_ref(),
        "--prover-share".as_ref(),
        p.as_ref(),
        "--verifier-share".as_ref(),
        v.as_ref(),
        "--out".as_ref(),
        out.as_os_str(),
    ];
    let mut args = base.clone();
    args.extend::<[&std::ffi::OsStr; 4]>([
        "--previous-manifest".as_ref(),
        prev_tlm.as_os_str(),
        "--previous-log".as_ref(),
        prev_tdl.as_os_str(),
    ]);
    let output = tessera(&args);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let written: AuditSample = envelope::open(
        &fs::read(&out).expect("the sample"),
        &schema::AUDIT_SAMPLE,
        DecodeLimits::MESSAGE,
    )
    .expect("an AuditSample envelope");

    // The libraries' sample from the same inputs.
    let previous = VerifiedLog::check(&prev.manifest, prev.entries.clone()).expect("period 0");
    let log = VerifiedLog::check(&current.manifest, current.entries.clone()).expect("period 1");
    let found = candidates(Some(&previous), &log);
    // Request 2 ended here, over 40 + 60 tokens; request 3 is not a candidate yet; request 1
    // ended in period 0.
    let listed: Vec<(u64, u64, bool)> = found
        .iter()
        .map(|c| (c.request_id, c.tokens, c.ended))
        .collect();
    assert_eq!(listed, [(2, 100, true)]);
    let expected = AuditSample::draw(
        current.manifest.identity(),
        written.policy().clone(),
        coin_toss.clone(),
        found.iter().map(|c| {
            (
                RequestKey {
                    session_id: c.session_id,
                    request_id: c.request_id,
                },
                c.tokens,
            )
        }),
    )
    .expect("a sample");
    assert_eq!(written, expected);
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
    let (tlm2, tdl2, p2) = &periods[2];
    let mut args: Vec<&std::ffi::OsStr> = base.clone();
    args[3] = tlm2.as_os_str();
    args[5] = tdl2.as_os_str();
    args.extend::<[&std::ffi::OsStr; 4]>([
        "--previous-manifest".as_ref(),
        tlm.as_os_str(),
        "--previous-log".as_ref(),
        tdl.as_os_str(),
    ]);
    let output = tessera(&args);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let log2 = VerifiedLog::check(&p2.manifest, p2.entries.clone()).expect("period 2");
    let found = candidates(Some(&log), &log2);
    let listed: Vec<(u64, u64, bool)> = found
        .iter()
        .map(|c| (c.request_id, c.tokens, c.ended))
        .collect();
    assert_eq!(listed, [(3, 3, false), (99, 0, true)]);

    // Period 0 is the log's first: it needs no previous period. It is sampled here under the
    // committed example policy, `docs/perf/examples/audit-policy.json`.
    let example = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/perf/examples/audit-policy.json");
    let (tlm0, tdl0, _) = &periods[0];
    let mut args: Vec<&std::ffi::OsStr> = base.clone();
    args[3] = tlm0.as_os_str();
    args[5] = tdl0.as_os_str();
    args[7] = example.as_os_str();
    let output = tessera(&args);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let written: AuditSample = envelope::open(
        &fs::read(&out).expect("the sample"),
        &schema::AUDIT_SAMPLE,
        DecodeLimits::MESSAGE,
    )
    .expect("an AuditSample envelope");
    let rates: Vec<(u32, u64)> = written
        .policy()
        .buckets()
        .iter()
        .map(|b| (b.max_tokens, b.threshold))
        .collect();
    let threshold = |rate| RateBucket::threshold_of(rate).expect("a rate");
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
    let good = dir.join("policy.json");
    fs::write(&good, POLICY).expect("the policy");
    let verifier_share = [0x0f; 32];
    let c = hex(&coin_commitment(&verifier_share));
    let v = hex(&verifier_share);
    let wrong = hex(&[0x0e; 32]);
    let p = hex(&[0xf0; 32]);
    let out = dir.join("sample.tas");
    let (t0, d0, _) = &periods[0];
    let (t1, d1, _) = &periods[1];
    let (t2, d2, _) = &periods[2];
    let run = |manifest: &Path,
               log: &Path,
               previous: Option<(&Path, &Path)>,
               policy: &Path,
               share: &str| {
        let mut args: Vec<&std::ffi::OsStr> = vec![
            "audit".as_ref(),
            "sample".as_ref(),
            "--manifest".as_ref(),
            manifest.as_os_str(),
            "--log".as_ref(),
            log.as_os_str(),
            "--policy".as_ref(),
            policy.as_os_str(),
            "--verifier-commitment".as_ref(),
            c.as_ref(),
            "--prover-share".as_ref(),
            p.as_ref(),
            "--verifier-share".as_ref(),
            share.as_ref(),
            "--out".as_ref(),
            out.as_os_str(),
        ];
        if let Some((m, l)) = previous {
            args.extend::<[&std::ffi::OsStr; 4]>([
                "--previous-manifest".as_ref(),
                m.as_os_str(),
                "--previous-log".as_ref(),
                l.as_os_str(),
            ]);
        }
        refused(&args)
    };
    let usage = (64, "ERR-USAGE".to_owned());
    // No previous period for period 1.
    assert_eq!(run(t1, d1, None, &good, &v), usage);
    // Period 0 is not the period before period 2.
    assert_eq!(run(t2, d2, Some((t0, d0)), &good, &v), usage);
    // Period 0's log under period 1's manifest.
    assert_eq!(
        run(t1, d0, Some((t0, d0)), &good, &v),
        (65, "ERR-CODEC-DIGEST".to_owned())
    );
    // A share that does not open the commitment.
    assert_eq!(
        run(t1, d1, Some((t0, d0)), &good, &wrong),
        (20, "ERR-PROTO-COINTOSS".to_owned())
    );
    // Policy files the reader refuses.
    for (name, text) in [
        ("unknown-key", POLICY.replace("\"period_ns\"", "\"period\"")),
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
    ] {
        let path = dir.join(format!("{name}.json"));
        fs::write(&path, text).expect("a policy");
        assert_eq!(run(t1, d1, Some((t0, d0)), &path, &v), usage, "{name}");
    }
    assert!(!out.exists(), "a refused run writes no sample");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_zero_length_period_between_passes_through_and_nothing_else_does() {
    use tessera_anchor::ManifestBody;
    let dir = scratch("between");
    let periods = three_periods(&dir);
    let seed = Secret::new(HeapKey::from_box(Box::new([0x51; 32])));
    let (t0, d0, p0) = &periods[0];
    let (_, d1, p1) = &periods[1];
    // A zero-length period after period 0, then period 1 re-signed to follow it: its entries,
    // `first_seq` and start are unchanged, since the zero-length period holds no entry.
    let zero = ManifestBody {
        period_start_ns: p0.manifest.body.period_end_ns,
        first_seq: p1.manifest.body.first_seq,
        entry_count: 0,
        entries_root: tessera_anchor::merkle::root(&[]),
        prev_manifest: p0.manifest.identity(),
        ..p0.manifest.body
    }
    .sign(&seed)
    .expect("the zero-length period");
    let after = ManifestBody {
        prev_manifest: zero.identity(),
        ..p1.manifest.body
    }
    .sign(&seed)
    .expect("period 1 after it");
    let zero_tlm = dir.join("zero.tlm");
    let after_tlm = dir.join("after.tlm");
    fs::write(&zero_tlm, envelope::seal(&schema::LOG_MANIFEST, &zero)).expect("written");
    fs::write(&after_tlm, envelope::seal(&schema::LOG_MANIFEST, &after)).expect("written");
    let policy = dir.join("policy.json");
    fs::write(&policy, POLICY).expect("the policy");
    let verifier_share = [0x0f; 32];
    let (c, p, v) = (
        hex(&coin_commitment(&verifier_share)),
        hex(&[0xf0; 32]),
        hex(&verifier_share),
    );
    let out = dir.join("sample.tas");
    let run = |between: &Path| {
        let args: Vec<&std::ffi::OsStr> = vec![
            "audit".as_ref(),
            "sample".as_ref(),
            "--manifest".as_ref(),
            after_tlm.as_os_str(),
            "--log".as_ref(),
            d1.as_os_str(),
            "--previous-manifest".as_ref(),
            t0.as_os_str(),
            "--previous-log".as_ref(),
            d0.as_os_str(),
            "--between".as_ref(),
            between.as_os_str(),
            "--policy".as_ref(),
            policy.as_os_str(),
            "--verifier-commitment".as_ref(),
            c.as_ref(),
            "--prover-share".as_ref(),
            p.as_ref(),
            "--verifier-share".as_ref(),
            v.as_ref(),
            "--out".as_ref(),
            out.as_os_str(),
        ];
        args.iter().map(|a| a.to_os_string()).collect::<Vec<_>>()
    };
    let args = run(&zero_tlm);
    let refs: Vec<&std::ffi::OsStr> = args.iter().map(AsRef::as_ref).collect();
    let output = tessera(&refs);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let written: AuditSample = envelope::open(
        &fs::read(&out).expect("the sample"),
        &schema::AUDIT_SAMPLE,
        DecodeLimits::MESSAGE,
    )
    .expect("an AuditSample envelope");
    assert_eq!(*written.manifest_digest(), after.identity());
    assert_eq!(written.selected().len(), 1, "request 2, past every bucket");
    // A period of non-zero length cannot stand between: period 0 itself, which also breaks the
    // link.
    let args = run(t0);
    let refs: Vec<&std::ffi::OsStr> = args.iter().map(AsRef::as_ref).collect();
    assert_eq!(refused(&refs), (64, "ERR-USAGE".to_owned()));
    let _ = fs::remove_dir_all(&dir);
}

/// Periods of zero length that the chain may hold, and the inputs around them the command must
/// refuse: a zero-length previous period, one of non-zero length or with entries between, and a
/// zero-length period to sample. Each would otherwise give another sample than RFC-0063's.
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
        ..base
    });
    let p1z = sign(ManifestBody {
        prev_manifest: z.identity(),
        ..p1.manifest.body
    });
    // X: an empty period of non-zero length after period 0; P1x: period 1 after it.
    let x = sign(ManifestBody {
        period_end_ns: base.period_end_ns + 5,
        ..z.body
    });
    let p1x = sign(ManifestBody {
        period_start_ns: x.body.period_end_ns,
        prev_manifest: x.identity(),
        ..p1.manifest.body
    });
    // W: zero length but one entry; P1w: after it, its entries shifted by one.
    let w = sign(ManifestBody {
        entry_count: 1,
        ..z.body
    });
    let p1w = sign(ManifestBody {
        first_seq: p1.manifest.body.first_seq + 1,
        prev_manifest: w.identity(),
        ..p1.manifest.body
    });
    let (z_tlm, p1z_tlm) = (write("z.tlm", &z), write("p1z.tlm", &p1z));
    let (x_tlm, p1x_tlm) = (write("x.tlm", &x), write("p1x.tlm", &p1x));
    let (w_tlm, p1w_tlm) = (write("w.tlm", &w), write("p1w.tlm", &p1w));
    let policy = dir.join("policy.json");
    fs::write(&policy, POLICY).expect("the policy");
    let verifier_share = [0x0f; 32];
    let (c, p, v) = (
        hex(&coin_commitment(&verifier_share)),
        hex(&[0xf0; 32]),
        hex(&verifier_share),
    );
    let out = dir.join("sample.tas");
    let run = |manifest: &Path, log: &Path, previous: Option<(&Path, &Path)>, between: &[&Path]| {
        let mut args: Vec<std::ffi::OsString> = [
            "audit".into(),
            "sample".into(),
            "--manifest".into(),
            manifest.as_os_str().to_owned(),
            "--log".into(),
            log.as_os_str().to_owned(),
            "--policy".into(),
            policy.as_os_str().to_owned(),
            "--verifier-commitment".into(),
            c.clone().into(),
            "--prover-share".into(),
            p.clone().into(),
            "--verifier-share".into(),
            v.clone().into(),
            "--out".into(),
            out.as_os_str().to_owned(),
        ]
        .into();
        if let Some((m, l)) = previous {
            args.extend([
                "--previous-manifest".into(),
                m.as_os_str().to_owned(),
                "--previous-log".into(),
                l.as_os_str().to_owned(),
            ]);
        }
        for b in between {
            args.extend(["--between".into(), b.as_os_str().to_owned()]);
        }
        let refs: Vec<&std::ffi::OsStr> = args.iter().map(AsRef::as_ref).collect();
        let output = tessera(&refs);
        let status = output.status.code().expect("an exit status");
        if status == 0 {
            return (0, String::new());
        }
        refused(&refs)
    };
    let usage = (64, "ERR-USAGE".to_owned());
    // Through Z, period 1 draws from period 0.
    assert_eq!(run(&p1z_tlm, d1, Some((t0, d0)), &[&z_tlm]).0, 0);
    // Z as the previous period: it has no log to draw from, and period 0's unterminated request
    // would escape.
    assert_eq!(run(&p1z_tlm, d1, Some((&z_tlm, &empty_log)), &[]), usage);
    // Z itself: no candidates, so no sample.
    assert_eq!(run(&z_tlm, &empty_log, Some((t0, d0)), &[]), usage);
    // X between, which links but has length: period 0 is not the previous period.
    assert_eq!(run(&p1x_tlm, d1, Some((t0, d0)), &[&x_tlm]), usage);
    // W between, of zero length but with an entry.
    assert_eq!(run(&p1w_tlm, d1, Some((t0, d0)), &[&w_tlm]), usage);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_prover_answers_one_commitment_per_manifest_and_the_sample_follows() {
    let dir = scratch("share");
    let periods = three_periods(&dir);
    let (t0, d0, _) = &periods[0];
    let (t1, d1, p1) = &periods[1];
    let state = dir.join("state");
    let verifier_share = [0x2d; 32];
    let c = hex(&coin_commitment(&verifier_share));
    let other = hex(&coin_commitment(&[0x2e; 32]));
    let share = |manifest: &Path, commitment: &str| {
        tessera(&[
            "audit".as_ref(),
            "share".as_ref(),
            "--manifest".as_ref(),
            manifest.as_os_str(),
            "--commitment".as_ref(),
            commitment.as_ref(),
            "--state".as_ref(),
            state.as_os_str(),
        ])
    };
    let first = share(t1, &c);
    assert_eq!(
        first.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    let s_p = String::from_utf8(first.stdout)
        .expect("UTF-8")
        .trim()
        .to_owned();
    assert_eq!(s_p.len(), 64);
    // Kept before it was printed, as 96 bytes under the manifest's digest.
    let record = state.join(format!("{}.share", hex(&p1.manifest.identity())));
    assert_eq!(fs::read(&record).expect("the record").len(), 96);
    // The same commitment again: the same share.
    let again = share(t1, &c);
    assert_eq!(String::from_utf8(again.stdout).expect("UTF-8").trim(), s_p);
    // Another commitment for the manifest: refused, and the record is unchanged.
    let kept = fs::read(&record).expect("the record");
    let (status, code) = refused(&[
        "audit".as_ref(),
        "share".as_ref(),
        "--manifest".as_ref(),
        t1.as_os_str(),
        "--commitment".as_ref(),
        other.as_ref(),
        "--state".as_ref(),
        state.as_os_str(),
    ]);
    assert_eq!((status, code.as_str()), (20, "ERR-PROTO-ORDER"));
    assert_eq!(fs::read(&record).expect("the record"), kept);
    // Another manifest has its own toss.
    let elsewhere = share(t0, &other);
    assert_eq!(elsewhere.status.code(), Some(0));
    assert_ne!(
        String::from_utf8(elsewhere.stdout).expect("UTF-8").trim(),
        s_p
    );

    // The verifier reveals; the sample is drawn from the three values.
    let policy = dir.join("policy.json");
    fs::write(&policy, POLICY).expect("the policy");
    let out = dir.join("sample.tas");
    let v = hex(&verifier_share);
    let output = tessera(&[
        "audit".as_ref(),
        "sample".as_ref(),
        "--manifest".as_ref(),
        t1.as_os_str(),
        "--log".as_ref(),
        d1.as_os_str(),
        "--previous-manifest".as_ref(),
        t0.as_os_str(),
        "--previous-log".as_ref(),
        d0.as_os_str(),
        "--policy".as_ref(),
        policy.as_os_str(),
        "--verifier-commitment".as_ref(),
        c.as_ref(),
        "--prover-share".as_ref(),
        s_p.as_ref(),
        "--verifier-share".as_ref(),
        v.as_ref(),
        "--out".as_ref(),
        out.as_os_str(),
    ]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let written: AuditSample = envelope::open(
        &fs::read(&out).expect("the sample"),
        &schema::AUDIT_SAMPLE,
        DecodeLimits::MESSAGE,
    )
    .expect("an AuditSample envelope");
    assert_eq!(hex(&written.coin_toss().prover_share), s_p);
    let _ = fs::remove_dir_all(&dir);
}
