// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A sampled audit driven from the command line, end to end (02 §2.2, RFC-0016, RFC-0015
//! 'Prover pre-check'), through the real binaries and a separately built `tessera-verify`:
//!
//! 1. `tessera recorder ref` logs one session that a test client and upstream play through it:
//!    request 23, the exact-CLI fixture's sequence (two prompt tokens, then three output tokens in
//!    two frames), and request 24, two tokens. The client keeps the session's transcript.
//! 2. `tessera audit share` and `audit sample` draw the period's sample: request 23 is selected,
//!    request 24 is not (its bucket's rate is 0).
//! 3. `tessera audit batch` reports one batch of request 23, with the lengths its frames give.
//! 4. `tessera prove --sample` proves batch 0, non-interactively and live.
//! 5. `tessera-verify` accepts both against the statement the test builds itself, with the
//!    verifier's own decoders, from what the client sent and the period's manifest.
//!
//! Then what must be refused, with its code:
//! - a stored transcript with one output token changed: `ERR-WITNESS-ANCHOR`, before `Round1`
//!   (no packet in the Live directory) and before the witness, which would have reported
//!   `ERR-WITNESS-EXACT-MISMATCH`; no stored transcript, and a request whose frames give no
//!   lengths (a period the recorder's core writes, since the reference recorder never logs such
//!   a frame): `ERR-WITNESS-ANCHOR` too;
//! - a request not in the sample: no batch past the last, no sample that selects it (a sample
//!   edited to add it and resealed), no reference flags beside `--sample`: `ERR-USAGE`;
//! - a batch of two requests, under a policy that selects both: `ERR-USAGE`, naming #357;
//! - a sample whose policy names another parameter set than the claim list's: `ERR-USAGE`, since
//!   the verifier would refuse the proof (`Reject(Params)`, RFC-0016 'Verifier checks', #530);
//! - malformed inputs: a corrupted sample or transcript, a log that is not its manifest's (65);
//!   a sample of another period, flags missing or malformed (64).
//!
//! Every `--json` output validates against its schema: `tessera.audit-batches/1`,
//! `tessera.proof-run/1` and `tessera.verdict/1`.

use std::ffi::OsString;
use std::fs;
use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tessera_anchor::recorder::{Recorder, WireFrame, decode_log, encode_log};
use tessera_anchor::{
    DigestLogEntry, Direction, HeapKey, RequestEndFrame, Session, SessionTranscript, TokenFrame,
};
use tessera_codec::{DecodeLimits, Secret, decode, encode, envelope, schema};
use tessera_protocol::{AuditSample, coin_commitment};
use tv_codec::{Envelope, WeightCommitmentPublic};
use tv_core::{claims::ClaimList, protocol::*};
use tv_hash::{Domain, hash};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn hex(v: &[u8]) -> String {
    v.iter().map(|b| format!("{b:02x}")).collect()
}

/// A scratch directory, removed with the test.
struct Dir(PathBuf);

impl Dir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "tessera-prove-sample-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).expect("a scratch directory");
        Self(path)
    }

    fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// A child process with a watchdog, killed and reaped when dropped.
struct Process {
    child: Child,
    stdout: PathBuf,
    stderr: PathBuf,
}

impl Process {
    fn start(command: &mut Command, dir: &Path) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let stdout = dir.join(format!("{id}.stdout"));
        let stderr = dir.join(format!("{id}.stderr"));
        let child = command
            .stdout(Stdio::from(fs::File::create(&stdout).expect("stdout")))
            .stderr(Stdio::from(fs::File::create(&stderr).expect("stderr")))
            .spawn()
            .expect("the binary starts");
        Self {
            child,
            stdout,
            stderr,
        }
    }

    fn wait(mut self, expected: i32) -> String {
        let start = Instant::now();
        let status = loop {
            if let Some(s) = self.child.try_wait().expect("a status") {
                break s;
            }
            assert!(
                start.elapsed() < Duration::from_secs(180),
                "CLI watchdog expired: {}",
                fs::read_to_string(&self.stderr).unwrap_or_default()
            );
            thread::sleep(Duration::from_millis(10));
        };
        let stdout = fs::read_to_string(&self.stdout).expect("stdout");
        assert_eq!(
            status.code(),
            Some(expected),
            "stdout: {stdout}\nstderr: {}",
            fs::read_to_string(&self.stderr).unwrap_or_default()
        );
        stdout
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// `tessera-verify`, built with test parameters into its own target directory, which the
/// exact-mode CLI suite shares, unless `TESSERA_VERIFY_BIN` names one (docs/testing/exact-cli.md).
fn verifier_binary() -> &'static Path {
    static BIN: OnceLock<PathBuf> = OnceLock::new();
    BIN.get_or_init(|| {
        if let Some(path) = std::env::var_os("TESSERA_VERIFY_BIN") {
            return PathBuf::from(path);
        }
        let dir = Dir::new();
        let target = root().join("target/exact-cli-binaries/verifier");
        let mut cargo = Command::new("cargo");
        cargo
            .current_dir(root().join("verifier"))
            .args(["build", "--locked", "-p", "tv-cli", "--target-dir"])
            .arg(&target)
            .args(["--features", "tv-core/test-params"]);
        Process::start(&mut cargo, &dir.0).wait(0);
        target
            .join("debug")
            .join(format!("tessera-verify{}", std::env::consts::EXE_SUFFIX))
    })
}

/// A report checked against its schema, `spec/schemas/<name>-v1.schema.json`.
fn report(text: &str, name: &str) -> Value {
    let value: Value = serde_json::from_str(text).unwrap_or_else(|e| panic!("{e}: {text}"));
    let schema: Value = serde_json::from_slice(
        &fs::read(root().join(format!("spec/schemas/{name}-v1.schema.json"))).expect("a schema"),
    )
    .expect("JSON");
    assert!(xtask::reports::schema_problems(&schema).is_empty());
    let errors = xtask::reports::validate(&schema, &schema, &value, "");
    assert!(errors.is_empty(), "{errors:?}: {text}");
    value
}

/// The period's files and its entries.
struct Period {
    tlm: PathBuf,
    tdl: PathBuf,
    entries: Vec<DigestLogEntry>,
}

/// Every period the recorder has written to `dir`, in chain order: its files are named by its
/// start, and the log is renamed into place before the manifest.
fn periods(dir: &Path) -> Vec<Period> {
    let mut stems: Vec<String> = fs::read_dir(dir)
        .expect("the log directory")
        .filter_map(|e| {
            let name = e.ok()?.file_name().into_string().ok()?;
            name.strip_suffix(".tlm").map(str::to_owned)
        })
        .collect();
    stems.sort();
    stems
        .iter()
        .map(|stem| {
            let tdl = dir.join(format!("{stem}.tdl"));
            let entries = decode_log(&fs::read(&tdl).expect("a log")).expect("its entries");
            Period {
                tlm: dir.join(format!("{stem}.tlm")),
                tdl,
                entries,
            }
        })
        .collect()
}

fn ends(period: &Period, request: u64) -> bool {
    period.entries.iter().any(
        |e| matches!(e, DigestLogEntry::RequestEnd { request_id, .. } if *request_id == request),
    )
}

const POLICY_ONE: &str = r#"{
  "schema": "tessera.audit-policy/1",
  "period_ns": 1000000000,
  "buckets": [
    {"max_tokens": 4, "rate": "0"},
    {"max_tokens": 64, "rate": "1"}
  ],
  "min_batch_tokens": 16,
  "max_batch_tokens": 64,
  "proof_deadline_ns": 604800000000000,
  "params": "test-small/1"
}"#;

/// Every request selected, and both in one batch.
const POLICY_BOTH: &str = r#"{
  "schema": "tessera.audit-policy/1",
  "period_ns": 1000000000,
  "buckets": [{"max_tokens": 64, "rate": "1"}],
  "min_batch_tokens": 16,
  "max_batch_tokens": 64,
  "proof_deadline_ns": 604800000000000,
  "params": "test-small/1"
}"#;

/// `POLICY_ONE` under `live-64/1`, where the claim list names `test-small/1`.
const POLICY_OTHER_PARAMS: &str = r#"{
  "schema": "tessera.audit-policy/1",
  "period_ns": 1000000000,
  "buckets": [
    {"max_tokens": 4, "rate": "0"},
    {"max_tokens": 64, "rate": "1"}
  ],
  "min_batch_tokens": 16,
  "max_batch_tokens": 64,
  "proof_deadline_ns": 604800000000000,
  "params": "live-64/1"
}"#;

/// The serving stack's session: the recorded conversation, its transcript and the statement the
/// verifier expects.
struct Fixture {
    dir: Dir,
    model: PathBuf,
    /// `audit sample`'s log flags for the period of request 23's end.
    logs: Vec<OsString>,
    /// The period after it, an empty one, and the period itself, as its previous.
    next_logs: Vec<OsString>,
    session: [u8; 16],
    manifest_digest: String,
    weights_root: String,
    pin: String,
}

fn os(items: &[&str]) -> Vec<OsString> {
    items.iter().map(OsString::from).collect()
}

impl Fixture {
    fn path(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    fn process(&self, verifier: bool, args: &[OsString]) -> Process {
        let binary: &Path = if verifier {
            verifier_binary()
        } else {
            Path::new(env!("CARGO_BIN_EXE_tessera"))
        };
        Process::start(Command::new(binary).args(args), &self.dir.0)
    }

    fn run(&self, verifier: bool, args: &[OsString], expected: i32) -> String {
        self.process(verifier, args).wait(expected)
    }

    /// The exit status and the error object of a refused `tessera --json` run.
    fn refused(&self, args: &[OsString], expected: i32) -> Value {
        let mut args = args.to_vec();
        args.push("--json".into());
        let text = self.run(false, &args, expected);
        let v: Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{e}: {text}"));
        assert!(v["error"]["code"].is_string(), "{v}");
        v
    }

    fn new() -> Self {
        let dir = Dir::new();
        let model = root().join("spec/vectors/exact-cli/tiny-3-reference.d");
        let tessera = |args: &[OsString]| {
            Process::start(
                Command::new(env!("CARGO_BIN_EXE_tessera")).args(args),
                &dir.0,
            )
            .wait(0)
        };
        let mut compile = os(&["--json", "compile", "--model"]);
        compile.push(model.clone().into());
        compile.extend(os(&["--params", "test-small", "--out-dir"]));
        compile.push(dir.join("compiled").into());
        compile.extend(os(&["--max-log2-rows", "18", "--query-block", "2"]));
        report(&tessera(&compile), "commitment-report");
        let mut commit = os(&["commit-weights", "--model"]);
        commit.push(model.clone().into());
        commit.push("--plan".into());
        commit.push(dir.join("compiled/plan.tep").into());
        commit.push("--out".into());
        commit.push(dir.join("weights.twc").into());
        tessera(&commit);

        // 1. The session, sealed by the serving stack, which keeps its transcript.
        let (mut session, open, keys) = Session::open().expect("a session");
        let id = session.session_id();
        let mut transcript = SessionTranscript::new(id, keys);
        let mut seal = |request: u64, direction: Direction, index: u32, tokens: &[u32]| {
            let frame =
                |t: &[u32]| TokenFrame::new(request, direction, index, Secret::new(t.to_vec()));
            let sealed = session.seal(&frame(tokens)).expect("sealed");
            transcript.record(&sealed, frame(tokens)).expect("recorded");
            WireFrame::Sealed(sealed)
        };
        let end = |request_id| {
            WireFrame::End(RequestEndFrame {
                session_id: id,
                request_id,
            })
        };
        // The fixture's tokens: prompt [1, 2], greedy output [2, 6, 0] (spec/vectors/exact-cli).
        let inbound = [
            WireFrame::Open(open),
            seal(23, Direction::In, 0, &[1, 2]),
            seal(24, Direction::In, 0, &[3]),
        ];
        // Both ends together, so that one period holds them.
        let outbound = [
            seal(23, Direction::Out, 0, &[2, 6]),
            seal(23, Direction::Out, 1, &[0]),
            seal(24, Direction::Out, 0, &[1]),
            end(23),
            end(24),
        ];
        let transcripts = dir.join("transcripts");
        fs::create_dir_all(&transcripts).expect("the transcripts");
        let file = transcripts.join(format!("{}.tst", hex(&id)));
        envelope::write_file(&file, &schema::SESSION_TRANSCRIPT, &transcript).expect("written");
        // The tampered copy: request 23's last output token, 0, stored as 1. Frames encode
        // their counter, request, tag, index and token count in 26 bytes, after 56 of header.
        let mut bytes = encode(&transcript);
        let sizes = [2usize, 1, 2, 1, 1];
        let at = 56 + sizes[..3].iter().map(|n| 26 + 4 * n).sum::<usize>() + 26;
        assert_eq!(&bytes[at..at + 4], &0u32.to_le_bytes());
        bytes[at] = 1;
        let tampered: SessionTranscript =
            decode(&bytes, DecodeLimits::MESSAGE).expect("still a transcript");
        let tampered_dir = dir.join("tampered");
        fs::create_dir_all(&tampered_dir).expect("the tampered transcripts");
        envelope::write_file(
            &tampered_dir.join(format!("{}.tst", hex(&id))),
            &schema::SESSION_TRANSCRIPT,
            &tampered,
        )
        .expect("written");

        // The recorder, between the client and the upstream.
        let log_dir = dir.join("log");
        let key = dir.join("recorder.seed");
        fs::write(&key, [0x2a; 32]).expect("the seed");
        let upstream = TcpListener::bind("127.0.0.1:0").expect("an upstream port");
        let upstream_addr = upstream.local_addr().expect("its address");
        let answer: Vec<u8> = outbound.iter().flat_map(WireFrame::to_bytes).collect();
        let server = thread::spawn(move || {
            let (mut conn, _) = upstream.accept().expect("the recorder connects");
            let mut received = Vec::new();
            conn.read_to_end(&mut received).expect("the inbound bytes");
            conn.write_all(&answer).expect("the answer");
            conn.shutdown(Shutdown::Write)
                .expect("the end of the answer");
        });
        let mut recorder = Command::new(env!("CARGO_BIN_EXE_tessera"));
        recorder
            .args(["recorder", "ref", "--listen", "127.0.0.1:0", "--upstream"])
            .arg(upstream_addr.to_string())
            .arg("--log-dir")
            .arg(&log_dir)
            .arg("--key")
            .arg(&key)
            .args(["--period", "1"])
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        let mut running = Running(recorder.spawn().expect("the recorder starts"));
        let mut stderr = running.0.stderr.take().expect("its stderr");
        let mut line = Vec::new();
        let mut byte = [0u8; 1];
        while stderr.read(&mut byte).expect("its stderr") == 1 && byte[0] != b'\n' {
            line.push(byte[0]);
        }
        let line = String::from_utf8(line).expect("UTF-8");
        let address = line
            .split("listening on ")
            .nth(1)
            .and_then(|rest| rest.split(',').next())
            .expect("the listening address")
            .trim()
            .to_owned();
        thread::spawn(move || {
            let mut rest = String::new();
            let _ = stderr.read_to_string(&mut rest);
        });
        let mut client = TcpStream::connect(&address).expect("the client connects");
        let inbound: Vec<u8> = inbound.iter().flat_map(WireFrame::to_bytes).collect();
        client.write_all(&inbound).expect("the prompts");
        client
            .shutdown(Shutdown::Write)
            .expect("the end of the prompts");
        let mut received = Vec::new();
        client.read_to_end(&mut received).expect("the answer");
        drop(client);
        server.join().expect("the upstream");

        // The period of the ends, and the one after it, written.
        let deadline = Instant::now() + Duration::from_secs(30);
        let (written, at) = loop {
            let written = periods(&log_dir);
            if let Some(at) = written.iter().position(|p| ends(p, 23))
                && written.len() > at + 1
            {
                break (written, at);
            }
            assert!(
                Instant::now() < deadline,
                "no period with the requests' ends"
            );
            thread::sleep(Duration::from_millis(100));
        };
        drop(running);
        assert!(ends(&written[at], 24), "both ends in one period");
        let logs_of = |at: usize| {
            let mut logs = os(&["--manifest"]);
            logs.push(written[at].tlm.clone().into());
            logs.push("--log".into());
            logs.push(written[at].tdl.clone().into());
            if at > 0 {
                logs.push("--previous-manifest".into());
                logs.push(written[at - 1].tlm.clone().into());
                logs.push("--previous-log".into());
                logs.push(written[at - 1].tdl.clone().into());
            }
            logs
        };

        // 5. The verifier's statement: from the client's own knowledge, the verifier's decoders
        // of the claim list, the public weight commitment and the manifest.
        let limits = tv_codec::DecodeLimits::MESSAGE;
        let claims = ClaimList::open(
            &fs::read(dir.join("compiled/claims.tcl")).expect("the claims"),
            &limits,
        )
        .expect("a claim list");
        let weights = Envelope::open(
            &fs::read(dir.join("weights.twr")).expect("the public commitment"),
            WeightCommitmentPublic::EXPECTED,
            &limits,
            |b| hash(Domain::Envelope, b),
        )
        .expect("an envelope")
        .decode_payload(|c, _| WeightCommitmentPublic::decode(c))
        .expect("a commitment");
        let weights_root = weights.weights_root(|b| hash(Domain::WeightsRoot, b));
        let manifest = tv_core::anchor::LogManifest::open(
            &fs::read(&written[at].tlm).expect("the manifest"),
            &limits,
        )
        .expect("a manifest");
        let statement = PublicStatement {
            claim_list_digest: claims.digest(),
            weights_root,
            weights_epoch: weights.epoch,
            manifest_digest: *manifest.identity(),
            batch: BatchDescriptor {
                requests: vec![RequestRef {
                    request_id: 23,
                    session_id: id,
                    prompt_len: 2,
                    output_len: 3,
                }],
            },
            predictor: PredictorSpec::Exact {
                decoding: Decoding::Greedy,
            },
        };
        let f = Self {
            logs: logs_of(at),
            next_logs: logs_of(at + 1),
            session: id,
            manifest_digest: hex(manifest.identity()),
            weights_root: hex(&weights_root),
            pin: hex(&statement.digest()),
            model,
            dir,
        };
        // 2. The samples: under the first policy, and under one that selects both requests,
        // each with its own state; the next period's sample too.
        for (name, policy) in [
            ("one", POLICY_ONE),
            ("both", POLICY_BOTH),
            ("other-params", POLICY_OTHER_PARAMS),
        ] {
            f.sample(name, policy, &f.logs.clone());
        }
        f.sample("next", POLICY_ONE, &f.next_logs.clone());
        f
    }

    /// Runs `audit share` and `audit sample` over `logs` into `<name>.tas`.
    fn sample(&self, name: &str, policy: &str, logs: &[OsString]) {
        draw(&self.dir, name, policy, logs);
    }

    fn batch(&self, sample: &Path, out: &str) -> Vec<OsString> {
        let mut args = os(&["--json", "audit", "batch", "--sample"]);
        args.push(sample.into());
        args.extend_from_slice(&self.logs);
        args.push("--out-dir".into());
        args.push(self.path(out).into());
        args
    }

    /// `tessera prove --sample` of batch 0 of `sample`, live into `live` or to `proof.tnp`.
    fn prove(&self, sample: &Path, transcripts: &str, live: Option<&Path>) -> Vec<OsString> {
        let mut args = os(&["--json", "prove", "--model"]);
        args.push(self.model.clone().into());
        args.push("--plan".into());
        args.push(self.path("compiled/plan.tep").into());
        args.push("--weights".into());
        args.push(self.path("weights.twc").into());
        args.push("--sample".into());
        args.push(sample.into());
        args.extend_from_slice(&self.logs);
        args.push("--transcripts".into());
        args.push(self.path(transcripts).into());
        args.extend(os(&["--batch", "0"]));
        match live {
            Some(dir) => {
                args.push("--live".into());
                args.push(dir.into());
            }
            None => {
                args.push("--out".into());
                args.push(self.path("proof.tnp").into());
            }
        }
        args
    }

    fn verify(&self, live: Option<&Path>) -> Vec<OsString> {
        let mut args = os(&[if live.is_some() { "live" } else { "verify" }, "--json"]);
        args.push("--claims".into());
        args.push(self.path("compiled/claims.tcl").into());
        args.push("--weights".into());
        args.push(self.path("weights.twr").into());
        args.extend(os(&[
            "--weights-root",
            &self.weights_root,
            "--statement-digest",
            &self.pin,
            "--bound",
            "0",
        ]));
        match live {
            Some(dir) => {
                args.push("--connect".into());
                args.push(dir.into());
                args.extend(os(&[
                    "--request-id",
                    "23",
                    "--request-session",
                    &hex(&self.session),
                    "--prompt-len",
                    "2",
                    "--output-len",
                    "3",
                    "--manifest-digest",
                    &self.manifest_digest,
                    "--timeout-ms",
                    "30000",
                ]));
            }
            None => {
                args.push("--proof".into());
                args.push(self.path("proof.tnp").into());
            }
        }
        args
    }

    fn accepted(&self, text: &str) {
        let v = report(text, "verdict");
        assert_eq!(v["verdict"]["status"], "accept", "{v}");
        assert_eq!(v["requests"][0]["request_id"], 23);
        assert_eq!(v["digests"]["statement"], self.pin);
        assert_eq!(v["execution"]["statement_pinned"], true);
    }
}

/// `tessera` on `args` in `dir`, which must exit with `expected`; its stdout.
fn tessera(dir: &Dir, args: &[OsString], expected: i32) -> String {
    Process::start(
        Command::new(env!("CARGO_BIN_EXE_tessera")).args(args),
        &dir.0,
    )
    .wait(expected)
}

/// Runs `audit share` and `audit sample` over `logs` into `<dir>/<name>.tas`, with the policy
/// `policy` and a state of the sample's own.
fn draw(dir: &Dir, name: &str, policy: &str, logs: &[OsString]) {
    let policy_file = dir.join(&format!("{name}.json"));
    fs::write(&policy_file, policy).expect("the policy");
    let state = dir.join(&format!("state-{name}"));
    let share = [0x0f; 32];
    let mut args = os(&["audit", "share"]);
    args.extend_from_slice(&logs[..2]);
    args.push("--policy".into());
    args.push(policy_file.into());
    args.push("--commitment".into());
    args.push(hex(&coin_commitment(&share)).into());
    args.push("--state".into());
    args.push(state.clone().into());
    tessera(dir, &args, 0);
    let mut args = os(&["audit", "sample"]);
    args.extend_from_slice(logs);
    args.push("--state".into());
    args.push(state.into());
    args.push("--verifier-share".into());
    args.push(hex(&share).into());
    args.push("--out".into());
    args.push(dir.join(&format!("{name}.tas")).into());
    tessera(dir, &args, 0);
}

/// The recorder's process, killed when dropped.
struct Running(Child);

impl Drop for Running {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// A copy of the file at `from` with its byte `at` flipped.
fn corrupt(from: &Path, to: &Path, at: usize) {
    let mut bytes = fs::read(from).expect("the file");
    bytes[at] ^= 1;
    fs::write(to, bytes).expect("the copy");
}

fn replace(args: &[OsString], key: &str, value: impl Into<OsString>) -> Vec<OsString> {
    let mut args = args.to_vec();
    let i = args.iter().position(|a| a == key).expect("the flag");
    args[i + 1] = value.into();
    args
}

#[test]
fn a_sampled_batch_is_proven_from_the_cli_and_refused_when_it_must_be() {
    let f = Fixture::new();

    // 3. The batches: request 23 alone, with its frames' lengths; the report on disk is stdout's.
    let text = f.run(false, &f.batch(&f.path("one.tas"), "batches-one"), 0);
    let batches = report(&text, "audit-batches");
    assert_eq!(
        fs::read_to_string(f.path("batches-one/batches.json")).expect("the report"),
        text
    );
    assert_eq!(batches["manifest_digest"], f.manifest_digest);
    assert_eq!(batches["selected"], 1);
    assert_eq!(
        batches["batches"],
        json!([{
            "index": 0,
            "group": "greedy",
            "tokens": 5,
            "requests": [{
                "session_id": hex(&f.session),
                "request_id": 23,
                "tokens": 5,
                "prompt_len": 2,
                "output_len": 3,
            }],
        }])
    );
    let both = report(
        &f.run(false, &f.batch(&f.path("both.tas"), "batches-both"), 0),
        "audit-batches",
    );
    assert_eq!(
        both["batches"][0]["requests"].as_array().map(Vec::len),
        Some(2)
    );

    // 4 and 5. Non-interactive, then live; each proof accepted against the test's own statement.
    let produced = report(
        &f.run(false, &f.prove(&f.path("one.tas"), "transcripts", None), 0),
        "proof-run",
    );
    assert_eq!(produced["statement_digest"], f.pin);
    assert_eq!(produced["weights_root"], f.weights_root);
    assert_eq!(produced["mode"], "NonInteractive");
    f.accepted(&f.run(true, &f.verify(None), 0));
    let live = f.path("live");
    fs::create_dir(&live).expect("the session directory");
    let verifier = f.process(true, &f.verify(Some(&live)));
    let producer = f.process(
        false,
        &f.prove(&f.path("one.tas"), "transcripts", Some(&live)),
    );
    let produced = report(&producer.wait(0), "proof-run");
    assert_eq!(produced["mode"], "Live");
    assert_eq!(produced["statement_digest"], f.pin);
    f.accepted(&verifier.wait(0));

    // A tampered stored transcript: refused by the pre-check, before Round1 and the witness.
    let fresh = f.path("live-tampered");
    fs::create_dir(&fresh).expect("a session directory");
    let mut tampered_live = f.prove(&f.path("one.tas"), "tampered", Some(&fresh));
    tampered_live.extend(os(&["--timeout-ms", "5000"]));
    let bad = f.refused(&tampered_live[1..], 20);
    assert_eq!(bad["error"]["code"], "ERR-WITNESS-ANCHOR", "{bad}");
    assert_eq!(
        fs::read_dir(&fresh).expect("the directory").count(),
        0,
        "a packet was sent"
    );
    let ni = f.prove(&f.path("one.tas"), "tampered", None);
    let ni = replace(&ni, "--out", f.path("tampered.tnp"));
    let bad = f.refused(&ni[1..], 20);
    assert_eq!(bad["error"]["code"], "ERR-WITNESS-ANCHOR", "{bad}");
    assert!(!f.path("tampered.tnp").exists());
    // No stored transcript of the session at all.
    fs::create_dir(f.path("none")).expect("an empty directory");
    let missing = replace(&ni, "--transcripts", f.path("none"));
    assert_eq!(
        f.refused(&missing[1..], 20)["error"]["code"],
        "ERR-WITNESS-ANCHOR"
    );

    // A request not in the sample. No batch past the last.
    let out = replace(&ni, "--transcripts", f.path("transcripts"));
    let usage = |args: &[OsString], says: &str| {
        let v = f.refused(args, 64);
        assert_eq!(v["error"]["code"], "ERR-USAGE", "{v}");
        let message = v["error"]["message"].as_str().unwrap_or_default();
        assert!(message.contains(says), "{message}");
    };
    usage(&replace(&out, "--batch", "1")[1..], "there is no batch 1");
    // No sample that selects it: request 24 added to the sample and resealed. The selected keys
    // close the encoding, a count and then each key's 24 bytes.
    let sample: AuditSample = envelope::open(
        &fs::read(f.path("one.tas")).expect("the sample"),
        &schema::AUDIT_SAMPLE,
        DecodeLimits::MESSAGE,
    )
    .expect("a sample");
    let mut payload = encode(&sample);
    payload.truncate(payload.len() - 8 - 24 * sample.selected().len());
    payload.extend_from_slice(&2u64.to_le_bytes());
    for request in [23u64, 24] {
        payload.extend_from_slice(&f.session);
        payload.extend_from_slice(&request.to_le_bytes());
    }
    let forged: AuditSample = decode(&payload, DecodeLimits::MESSAGE).expect("a sample");
    assert_eq!(forged.selected().len(), 2);
    assert_eq!(forged.policy(), sample.policy());
    fs::write(
        f.path("forged.tas"),
        envelope::seal(&schema::AUDIT_SAMPLE, &forged),
    )
    .expect("written");
    let unselected = "does not select what its period's logs and coin toss select";
    usage(
        &replace(&out, "--sample", f.path("forged.tas"))[1..],
        unselected,
    );
    usage(
        &f.batch(&f.path("forged.tas"), "batches-forged")[1..],
        unselected,
    );
    // No reference flags beside the sample.
    let mut flags = out[1..].to_vec();
    flags.extend(os(&["--request-id", "24"]));
    usage(&flags, "required arguments were not provided");

    // A batch of two requests.
    usage(&replace(&out, "--sample", f.path("both.tas"))[1..], "#357");
    // A sample of the same selection under another parameter set: refused before anything is
    // proven, so the proof file the accepted run wrote is left as it was.
    let before = fs::read(f.path("proof.tnp")).ok();
    usage(
        &replace(&out, "--sample", f.path("other-params.tas"))[1..],
        "the claim list's parameter set test-small/1 is not the sample policy's live-64/1",
    );
    assert_eq!(fs::read(f.path("proof.tnp")).ok(), before);

    // Malformed inputs. A corrupted sample and transcript: 65.
    corrupt(&f.path("one.tas"), &f.path("corrupt.tas"), 40);
    for args in [
        replace(&out, "--sample", f.path("corrupt.tas")),
        f.batch(&f.path("corrupt.tas"), "batches-corrupt"),
    ] {
        assert_eq!(
            f.refused(&args[1..], 65)["error"]["code"],
            "ERR-CODEC-DIGEST"
        );
    }
    let name = format!("{}.tst", hex(&f.session));
    fs::create_dir(f.path("corrupt")).expect("a directory");
    corrupt(
        &f.path(&format!("transcripts/{name}")),
        &f.path(&format!("corrupt/{name}")),
        60,
    );
    let corrupt_transcript = replace(&out, "--transcripts", f.path("corrupt"));
    assert_eq!(
        f.refused(&corrupt_transcript[1..], 65)["error"]["code"],
        "ERR-CODEC-DIGEST"
    );
    // A log that is not its manifest's: an empty one.
    fs::write(f.path("empty.tdl"), b"").expect("an empty log");
    for args in [&out, &f.batch(&f.path("one.tas"), "batches-log")] {
        let args = replace(args, "--log", f.path("empty.tdl"));
        assert_eq!(
            f.refused(&args[1..], 65)["error"]["code"],
            "ERR-CODEC-DIGEST"
        );
    }
    // A sample of another period, and malformed or missing flags: 64.
    usage(
        &replace(&out, "--sample", f.path("next.tas"))[1..],
        "is not a sample of --manifest's period",
    );
    usage(&replace(&out, "--batch", "first")[1..], "--batch");
    let i = out
        .iter()
        .position(|a| a == "--transcripts")
        .expect("the flag");
    let mut without = out[1..].to_vec();
    without.drain(i - 1..i + 1);
    usage(&without, "required arguments were not provided");
    assert!(
        !f.path("tampered.tnp").exists(),
        "a refused run wrote a proof"
    );
}

/// RFC-0016 step 4's 'A frame with no length', which the reference recorder never logs: its
/// parser refuses such a header. The recorder's core writes the period instead, request 25's
/// frame 23 bytes long beside request 23. `audit batch` reports request 25 first, alone and
/// without lengths, and `prove --sample` refuses its batch with `ERR-WITNESS-ANCHOR` before it
/// reads a transcript or the model.
#[test]
fn a_request_whose_frames_give_no_lengths_is_batched_alone_then_refused() {
    let dir = Dir::new();
    let seed = Secret::new(HeapKey::from_box(Box::new([0x51; 32])));
    let mut recorder = Recorder::new(seed, [4; 16], 1_000);
    let (mut session, open, _) = Session::open().expect("a session");
    let id = session.session_id();
    let mut seal = |request: u64, direction: Direction, tokens: &[u32]| {
        let frame = TokenFrame::new(request, direction, 0, Secret::new(tokens.to_vec()));
        session.seal(&frame).expect("sealed")
    };
    let mut broken = seal(25, Direction::In, &[]);
    broken.ciphertext.push(0);
    broken.header.length = 23;
    let end = |request_id| {
        WireFrame::End(RequestEndFrame {
            session_id: id,
            request_id,
        })
    };
    let frames = [
        WireFrame::Open(open),
        WireFrame::Sealed(seal(23, Direction::In, &[1, 2])),
        WireFrame::Sealed(broken),
        WireFrame::Sealed(seal(23, Direction::Out, &[2, 6, 0])),
        end(23),
        end(25),
    ];
    for (frame, t) in frames.iter().zip((1_010..).step_by(10)) {
        recorder.record(frame, t);
    }
    let period = recorder.rotate(2_000).expect("a period");
    let (tlm, tdl) = (dir.join("period.tlm"), dir.join("period.tdl"));
    fs::write(
        &tlm,
        envelope::seal(&schema::LOG_MANIFEST, &period.manifest),
    )
    .expect("written");
    fs::write(&tdl, encode_log(&period.entries)).expect("written");
    let mut logs = os(&["--manifest"]);
    logs.push(tlm.into());
    logs.push("--log".into());
    logs.push(tdl.into());
    draw(&dir, "one", POLICY_ONE, &logs);

    let mut batch = os(&["--json", "audit", "batch", "--sample"]);
    batch.push(dir.join("one.tas").into());
    batch.extend_from_slice(&logs);
    batch.push("--out-dir".into());
    batch.push(dir.join("batches").into());
    let report = report(&tessera(&dir, &batch, 0), "audit-batches");
    let session = hex(&id);
    assert_eq!(report["selected"], 2);
    assert_eq!(
        report["batches"],
        json!([
            {
                "index": 0,
                "group": "greedy",
                "tokens": u64::MAX,
                "requests": [{"session_id": session, "request_id": 25, "tokens": u64::MAX}],
            },
            {
                "index": 1,
                "group": "greedy",
                "tokens": 5,
                "requests": [{
                    "session_id": session,
                    "request_id": 23,
                    "tokens": 5,
                    "prompt_len": 2,
                    "output_len": 3,
                }],
            },
        ])
    );

    fs::create_dir(dir.join("transcripts")).expect("a directory");
    let mut prove = os(&["--json", "prove", "--model"]);
    prove.push(
        root()
            .join("spec/vectors/exact-cli/tiny-3-reference.d")
            .into(),
    );
    prove.push("--plan".into());
    prove.push(dir.join("absent.tep").into());
    prove.push("--weights".into());
    prove.push(dir.join("absent.twc").into());
    prove.push("--sample".into());
    prove.push(dir.join("one.tas").into());
    prove.extend_from_slice(&logs);
    prove.push("--transcripts".into());
    prove.push(dir.join("transcripts").into());
    prove.extend(os(&["--batch", "0", "--out"]));
    prove.push(dir.join("proof.tnp").into());
    let text = tessera(&dir, &prove, 20);
    let refused: Value = serde_json::from_str(&text).expect("an error object");
    assert_eq!(refused["error"]["code"], "ERR-WITNESS-ANCHOR");
    assert_eq!(
        refused["error"]["message"],
        "request 25: its lengths are not its frames' token counts"
    );
    assert!(!dir.join("proof.tnp").exists());
}
