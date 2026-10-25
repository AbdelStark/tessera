// SPDX-License-Identifier: Apache-2.0 OR MIT

//! 07 §6: real, separately built CLIs; expected statements never come from proofs.
use serde_json::Value;
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};
use tv_codec::{DecodeLimits, Envelope, WeightCommitmentPublic};
use tv_core::{claims::ClaimList, protocol::*};
use tv_hash::{Domain, hash};

mod canary;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn hex(v: &[u8]) -> String {
    v.iter().map(|b| format!("{b:02x}")).collect()
}
struct Dir(PathBuf);
impl Dir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "tessera-exact-cli-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

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
            .stdout(Stdio::from(fs::File::create(&stdout).unwrap()))
            .stderr(Stdio::from(fs::File::create(&stderr).unwrap()))
            .spawn()
            .unwrap();
        Self {
            child,
            stdout,
            stderr,
        }
    }
    fn wait(mut self, expected: i32) -> String {
        let start = Instant::now();
        let status = loop {
            if let Some(s) = self.child.try_wait().unwrap() {
                break s;
            }
            assert!(
                start.elapsed() < Duration::from_secs(180),
                "CLI watchdog expired: {}",
                fs::read_to_string(&self.stderr).unwrap()
            );
            std::thread::sleep(Duration::from_millis(10));
        };
        let stdout = fs::read_to_string(&self.stdout).unwrap();
        assert_eq!(
            status.code(),
            Some(expected),
            "stdout: {stdout}\nstderr: {}",
            fs::read_to_string(&self.stderr).unwrap()
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

fn binaries() -> &'static (PathBuf, PathBuf) {
    static BINS: OnceLock<(PathBuf, PathBuf)> = OnceLock::new();
    BINS.get_or_init(|| {
        let dir = Dir::new();
        let build = |env: &str, workspace: &str, package: &str, name: &str, features: &[&str]| {
            if let Some(path) = std::env::var_os(env) {
                return PathBuf::from(path);
            }
            // Separate build directories avoid the outer cargo test's build lock.
            let target = root()
                .join("target/exact-cli-binaries")
                .join(if workspace == "." {
                    "prover"
                } else {
                    workspace
                });
            let mut cargo = Command::new("cargo");
            cargo
                .current_dir(root().join(workspace))
                .args(["build", "--locked", "-p", package, "--target-dir"])
                .arg(&target)
                .args(features);
            Process::start(&mut cargo, &dir.0).wait(0);
            target
                .join("debug")
                .join(format!("{name}{}", std::env::consts::EXE_SUFFIX))
        };
        (
            build("TESSERA_BIN", ".", "tessera-cli", "tessera", &[]),
            build(
                "TESSERA_VERIFY_BIN",
                "verifier",
                "tv-cli",
                "tessera-verify",
                &["--features", "tv-core/test-params"],
            ),
        )
    })
}
fn report(text: &str, name: &str) -> Value {
    let value: Value = serde_json::from_str(text).unwrap();
    let schema: Value = serde_json::from_slice(
        &fs::read(root().join(format!("spec/schemas/{name}-v1.schema.json"))).unwrap(),
    )
    .unwrap();
    assert!(xtask::reports::schema_problems(&schema).is_empty());
    let errors = xtask::reports::validate(&schema, &schema, &value, "");
    assert!(errors.is_empty(), "{errors:?}");
    value
}
fn set(args: &mut [String], key: &str, value: impl ToString) {
    let i = args.iter().position(|x| x == key).unwrap();
    args[i + 1] = value.to_string();
}
/// A CLI fixture of `spec/vectors/`: its model, recorded tokens and the request they make.
struct Set {
    dir: &'static str,
    model: &'static str,
    layers: u64,
    prompt_len: u32,
    output_len: u32,
}
const EXACT: Set = Set {
    dir: "exact-cli",
    model: "tiny-3-reference.d",
    layers: 3,
    prompt_len: 2,
    output_len: 3,
};
/// OBS-3's canary model (`spec/vectors/canary/`).
const CANARY: Set = Set {
    dir: "canary",
    model: "canary-reference.d",
    layers: 2,
    prompt_len: 3,
    output_len: 3,
};
struct Fixture {
    dir: Dir,
    set: &'static Set,
    /// Logging flags given to every producer command.
    log: &'static [&'static str],
    model: String,
    tokens: String,
    pin: String,
    weights_root: String,
}
impl Fixture {
    fn new() -> Self {
        Self::with(&EXACT, &[])
    }
    fn with(set: &'static Set, log: &'static [&'static str]) -> Self {
        let vectors = root().join("spec/vectors").join(set.dir);
        let mut f = Self {
            dir: Dir::new(),
            set,
            log,
            model: vectors.join(set.model).display().to_string(),
            tokens: vectors.join("recorded.ttk").display().to_string(),
            pin: String::new(),
            weights_root: String::new(),
        };
        let compile = f.run(
            false,
            &[
                "--json".into(),
                "compile".into(),
                "--model".into(),
                f.model.clone(),
                "--params".into(),
                "test-small".into(),
                "--out-dir".into(),
                f.path("compiled"),
                "--max-log2-rows".into(),
                "18".into(),
                "--query-block".into(),
                "2".into(),
            ],
            0,
        );
        let compile = report(&compile, "commitment-report");
        assert_eq!(compile["model"]["n_layers"], set.layers);
        f.run(
            false,
            &[
                "--json".into(),
                "commit-weights".into(),
                "--model".into(),
                f.model.clone(),
                "--plan".into(),
                f.path("compiled/plan.tep"),
                "--out".into(),
                f.path("weights.twc"),
            ],
            0,
        );
        let limits = DecodeLimits::MESSAGE;
        let claims =
            ClaimList::open(&fs::read(f.path("compiled/claims.tcl")).unwrap(), &limits).unwrap();
        let bytes = fs::read(f.path("weights.twr")).unwrap();
        let weights = Envelope::open(&bytes, WeightCommitmentPublic::EXPECTED, &limits, |b| {
            hash(Domain::Envelope, b)
        })
        .unwrap()
        .decode_payload(|c, _| WeightCommitmentPublic::decode(c))
        .unwrap();
        let weights_root = weights.weights_root(|b| hash(Domain::WeightsRoot, b));
        // Construct the entire expected statement BEFORE either producer mode starts.
        let statement = PublicStatement {
            claim_list_digest: claims.digest(),
            weights_root,
            weights_epoch: weights.epoch,
            manifest_digest: [0x13; 32],
            batch: BatchDescriptor {
                requests: vec![RequestRef {
                    request_id: 23,
                    session_id: [0x11; 16],
                    prompt_len: set.prompt_len,
                    output_len: set.output_len,
                }],
            },
            predictor: PredictorSpec::Exact {
                decoding: Decoding::Greedy,
            },
        };
        f.pin = hex(&statement.digest());
        f.weights_root = hex(&weights_root);
        f
    }
    fn path(&self, s: &str) -> String {
        self.dir.0.join(s).display().to_string()
    }
    fn process(&self, verifier: bool, args: &[String]) -> Process {
        let mut command = if verifier {
            Command::new(&binaries().1)
        } else {
            let mut command = Command::new(&binaries().0);
            command.args(self.log);
            command
        };
        Process::start(command.args(args), &self.dir.0)
    }
    fn run(&self, verifier: bool, args: &[String], expected: i32) -> String {
        self.process(verifier, args).wait(expected)
    }
    fn prove(&self, live: bool) -> Vec<String> {
        let prompt = self.set.prompt_len.to_string();
        [
            "--json",
            "prove",
            "--model",
            &self.model,
            "--plan",
            &self.path("compiled/plan.tep"),
            "--weights",
            &self.path("weights.twc"),
            "--reference-tokens",
            &self.tokens,
            "--prompt-len",
            &prompt,
            "--request-id",
            "23",
            "--request-session",
            &"11".repeat(16),
            "--manifest-digest",
            &"13".repeat(32),
            if live { "--live" } else { "--out" },
            &self.path(if live { "live" } else { "proof.tnp" }),
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    }
    fn verify(&self, live: bool) -> Vec<String> {
        let (prompt, output) = (
            self.set.prompt_len.to_string(),
            self.set.output_len.to_string(),
        );
        let mut args: Vec<String> = [
            if live { "live" } else { "verify" },
            "--json",
            "--claims",
            &self.path("compiled/claims.tcl"),
            "--weights",
            &self.path("weights.twr"),
            "--weights-root",
            &self.weights_root,
            "--statement-digest",
            &self.pin,
            "--bound",
            "0",
            "--out",
            &self.path("verdict.json"),
            if live { "--connect" } else { "--proof" },
            &self.path(if live { "live" } else { "proof.tnp" }),
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        if live {
            args.extend(
                [
                    "--request-id",
                    "23",
                    "--request-session",
                    &"11".repeat(16),
                    "--prompt-len",
                    &prompt,
                    "--output-len",
                    &output,
                    "--manifest-digest",
                    &"13".repeat(32),
                    "--timeout-ms",
                    "30000",
                ]
                .into_iter()
                .map(str::to_owned),
            );
        }
        args
    }
    /// Proves the request, non-interactively then verified or in a directory Live session, and
    /// returns the producer's and the verifier's reports.
    fn proved(&self, live: bool) -> (String, String) {
        if live {
            fs::create_dir(self.path("live")).unwrap();
        }
        let verifier = live.then(|| self.process(true, &self.verify(true)));
        let mut producer = self.process(false, &self.prove(live));
        match verifier {
            Some(mut verifier) => {
                let start = Instant::now();
                loop {
                    let p = producer.child.try_wait().unwrap();
                    let v = verifier.child.try_wait().unwrap();
                    for status in [p, v].into_iter().flatten() {
                        assert!(status.success(), "Live peer failed: {status}");
                    }
                    if p.is_some() && v.is_some() {
                        break;
                    }
                    assert!(
                        start.elapsed() < Duration::from_secs(60),
                        "Live pair watchdog expired"
                    );
                    std::thread::sleep(Duration::from_millis(5));
                }
                (producer.wait(0), verifier.wait(0))
            }
            None => (producer.wait(0), self.run(true, &self.verify(false), 0)),
        }
    }
    fn accepted(&self, text: &str) {
        let v = report(text, "verdict");
        assert_eq!(v["verdict"]["status"], "accept");
        assert_eq!(v["batch_uhat_fixed"], 0);
        assert_eq!(v["requests"][0]["request_id"], 23);
        assert_eq!(v["requests"][0]["uhat_fixed"], 0);
        assert_eq!(v["digests"]["statement"], self.pin);
        assert_eq!(v["execution"]["statement_pinned"], true);
        assert_eq!(v["test_parameters"], true);
        for key in ["anchor_verified", "ledger_updated", "zero_knowledge"] {
            assert_eq!(v[key], false);
        }
        assert_eq!(fs::read_to_string(self.path("verdict.json")).unwrap(), text);
    }
}

/// Runs the producer with `--log-format json --log-level trace` and returns its stderr lines,
/// each checked against `tessera.log-event/1` (05 §2).
fn logged(f: &Fixture, args: &[String], expected: i32) -> Vec<Value> {
    let mut full: Vec<String> = ["--log-format", "json", "--log-level", "trace"]
        .into_iter()
        .map(str::to_owned)
        .collect();
    full.extend_from_slice(args);
    let process = f.process(false, &full);
    let stderr = process.stderr.clone();
    process.wait(expected);
    let schema: Value = serde_json::from_slice(
        &fs::read(root().join("spec/schemas/log-event-v1.schema.json")).unwrap(),
    )
    .unwrap();
    assert!(xtask::reports::schema_problems(&schema).is_empty());
    fs::read_to_string(stderr)
        .unwrap()
        .lines()
        .map(|line| {
            let value: Value = serde_json::from_str(line).unwrap_or_else(|e| panic!("{e}: {line}"));
            let errors = xtask::reports::validate(&schema, &schema, &value, "");
            assert!(errors.is_empty(), "{errors:?}: {line}");
            let event = value["event"].as_str().unwrap();
            let krate = value["target"]
                .as_str()
                .unwrap()
                .split("::")
                .next()
                .unwrap();
            assert_eq!(
                event.split('.').take(2).collect::<Vec<_>>(),
                ["tessera", krate.strip_prefix("tessera_").unwrap()],
                "{line}"
            );
            value
        })
        .collect()
}

/// 05 §2: every stderr line of a JSON-logged compile, weight commitment, NI proof and failed
/// command is a `tessera.log-event/1` object; the failure keeps 02 §2's code and message.
#[test]
fn e2e_json_log_lines_follow_their_schema() {
    let f = Fixture::new();
    let mut lines = logged(
        &f,
        &[
            "compile".into(),
            "--model".into(),
            f.model.clone(),
            "--params".into(),
            "test-small".into(),
            "--out-dir".into(),
            f.path("logged"),
            "--max-log2-rows".into(),
            "18".into(),
            "--query-block".into(),
            "2".into(),
        ],
        0,
    );
    lines.extend(logged(
        &f,
        &[
            "commit-weights".into(),
            "--model".into(),
            f.model.clone(),
            "--plan".into(),
            f.path("logged/plan.tep"),
            "--out".into(),
            f.path("logged.twc"),
        ],
        0,
    ));
    lines.extend(logged(&f, &f.prove(false), 0));
    let failed = logged(&f, &["inspect".into(), f.path("missing.tnp")], 20);
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0]["event"], "tessera.cli.command_failed");
    assert_eq!(failed[0]["level"], "error");
    assert_eq!(failed[0]["fields"]["error"], "ERR-RES-IO");
    assert!(
        failed[0]["fields"]["message"]
            .as_str()
            .unwrap()
            .starts_with("ERR-RES-IO: cannot read ")
    );
    let events: std::collections::BTreeSet<&str> =
        lines.iter().map(|l| l["event"].as_str().unwrap()).collect();
    for event in [
        "tessera.cli.compile_written",
        "tessera.cli.weights_written",
        "tessera.compiler.compiled",
        "tessera.pcs.weights_committed",
        "tessera.pcs.row_block_committed",
        "tessera.prover.stage_proved",
        "tessera.prover.round3_answered",
    ] {
        assert!(events.contains(event), "{event}: {events:?}");
    }
    // The `prove` span names the batch by the statement digest the verifier pins.
    for line in &lines {
        for span in line["span_path"].as_array().unwrap() {
            if span["name"] == "prove" {
                assert_eq!(span["fields"]["batch"], f.pin.as_str(), "{line}");
            }
        }
    }
    let spans: std::collections::BTreeSet<&str> = lines
        .iter()
        .flat_map(|l| l["span_path"].as_array().unwrap())
        .map(|s| s["name"].as_str().unwrap())
        .collect();
    for span in [
        "compile",
        "commit_weights",
        "prove",
        "witness_pass",
        "round1",
        "stage",
        "opening",
        "round3",
    ] {
        assert!(spans.contains(span), "{span}: {spans:?}");
    }
}

#[test]
fn e2e_exact_tiny_ni_and_live_ragged() {
    let f = Fixture::new();
    let expected: Value = serde_json::from_slice(
        &fs::read(root().join("spec/vectors/exact-cli/expected.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(expected["layers"], 3);
    assert_eq!(expected["token_ids"].as_array().unwrap().len(), 5);
    assert!(
        expected["logits"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| row.as_array().unwrap())
            .any(|v| v.as_i64().unwrap() != 0)
    );
    for live in [false, true] {
        let (produced, verdict) = f.proved(live);
        let producer = report(&produced, "proof-run");
        assert_eq!(producer["statement_digest"], f.pin);
        assert_eq!(producer["weights_root"], f.weights_root);
        assert_eq!(
            producer["mode"],
            if live { "Live" } else { "NonInteractive" }
        );
        assert_eq!(producer["verifier_acceptance"], false);
        assert_eq!(producer["test_parameters"], true);
        f.accepted(&verdict);
    }
    let before = fs::read(f.path("live/00-SeedCommit.tce")).unwrap();
    let again = report(&f.run(true, &f.verify(true), 64), "verdict");
    assert_eq!(again["verdict"]["error"]["code"], "ERR-USAGE");
    assert_eq!(fs::read(f.path("live/00-SeedCommit.tce")).unwrap(), before);
    let producer: Value = serde_json::from_str(&f.run(false, &f.prove(true), 64)).unwrap();
    assert_eq!(producer["error"]["code"], "ERR-USAGE");
    let proof = fs::read(f.path("proof.tnp")).unwrap();
    let output: Value = serde_json::from_str(&f.run(false, &f.prove(false), 64)).unwrap();
    assert_eq!(output["error"]["code"], "ERR-USAGE");
    assert_eq!(fs::read(f.path("proof.tnp")).unwrap(), proof);
    for (key, value) in [
        ("--statement-digest", "ab".repeat(32)),
        ("--weights-root", "ab".repeat(32)),
    ] {
        let mut args = f.verify(false);
        set(&mut args, key, value);
        let bad = report(&f.run(true, &args, 64), "verdict");
        assert_eq!(bad["verdict"]["error"]["code"], "ERR-USAGE");
        assert_eq!(bad["execution"]["statement_pinned"], false);
    }
    let mut corrupt = proof.clone();
    corrupt[20] ^= 1;
    fs::write(f.path("proof.tnp"), corrupt).unwrap();
    let bad = report(&f.run(true, &f.verify(false), 65), "verdict");
    assert_eq!(bad["verdict"]["error"]["code"], "ERR-CODEC-DIGEST");
    fs::write(f.path("proof.tnp"), &proof).unwrap();
    let mut alias = f.verify(false);
    set(&mut alias, "--out", f.path("proof.tnp"));
    let bad = report(&f.run(true, &alias, 64), "verdict");
    assert_eq!(bad["verdict"]["error"]["code"], "ERR-USAGE");
    assert_eq!(fs::read(f.path("proof.tnp")).unwrap(), proof);
}

#[test]
fn e2e_exact_mismatch_and_input_boundaries() {
    let f = Fixture::new();
    let mut bytes = fs::read(&f.tokens).unwrap();
    // Third token is recorded output. Reseal to isolate the exactness check from the codec.
    bytes[20 + 16 + 2 * 4] = (bytes[20 + 16 + 2 * 4] + 1) % 8;
    bytes.truncate(bytes.len() - 32);
    bytes.extend_from_slice(&hash(Domain::Envelope, &bytes));
    fs::write(f.path("mismatch.ttk"), bytes).unwrap();
    let mut args = f.prove(false);
    set(&mut args, "--reference-tokens", f.path("mismatch.ttk"));
    let bad: Value = serde_json::from_str(&f.run(false, &args, 20)).unwrap();
    assert_eq!(bad["error"]["code"], "ERR-WITNESS-EXACT-MISMATCH");
    assert!(!Path::new(&f.path("proof.tnp")).exists());
    for (key, value) in [
        ("--request-session", "zz".repeat(16)),
        ("--manifest-digest", "0".repeat(63)),
        ("--prompt-len", "0".into()),
        ("--prompt-len", "5".into()),
        ("--backend", "gpu".into()),
        ("--devices", "1".into()),
        ("--memory-budget", u64::MAX.to_string()),
    ] {
        let mut args = f.prove(false);
        if ["--backend", "--devices", "--memory-budget"].contains(&key) {
            args.extend([key.into(), value]);
        } else {
            set(&mut args, key, value);
        }
        let bad: Value = serde_json::from_str(&f.run(false, &args, 64)).unwrap();
        assert_eq!(bad["error"]["code"], "ERR-USAGE");
        assert!(!Path::new(&f.path("proof.tnp")).exists());
    }
    let original = fs::read(&f.tokens).unwrap();
    let mut args = f.prove(false);
    set(&mut args, "--out", &f.tokens);
    let bad: Value = serde_json::from_str(&f.run(false, &args, 64)).unwrap();
    assert_eq!(bad["error"]["code"], "ERR-USAGE");
    assert_eq!(fs::read(&f.tokens).unwrap(), original);
    #[cfg(unix)]
    {
        let fifo = f.path("tokens.fifo");
        assert!(
            Command::new("mkfifo")
                .arg(&fifo)
                .status()
                .unwrap()
                .success()
        );
        let mut args = f.prove(false);
        set(&mut args, "--reference-tokens", fifo);
        let mut p = f.process(false, &args);
        let start = Instant::now();
        while p.child.try_wait().unwrap().is_none() {
            assert!(
                start.elapsed() < Duration::from_secs(2),
                "nonregular input blocked before refusal"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        let bad: Value = serde_json::from_str(&p.wait(20)).unwrap();
        assert_eq!(bad["error"]["code"], "ERR-RES-IO");
    }
}

fn packet(path: &Path, peer: &mut Process) -> Vec<u8> {
    let start = Instant::now();
    while !path.exists() {
        if let Some(status) = peer.child.try_wait().unwrap() {
            assert!(
                path.exists(),
                "peer stopped before publishing {}: {status}",
                path.display()
            );
        }
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "missing packet: {}",
            path.display()
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    fs::read(path).unwrap()
}
fn reseal(original: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut bytes = original[..12].to_vec();
    bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    bytes.extend_from_slice(payload);
    bytes.extend_from_slice(&hash(Domain::Envelope, &bytes));
    bytes
}

#[test]
fn e2e_live_rejects_each_round_before_releasing_the_next_challenge() {
    let f = Fixture::new();
    for (fault, exit, code) in [
        (1, 20, "ERR-PROTO-VERSION"),
        (4, 10, "REJ-PCS-EVAL"),
        (6, 10, "REJ-PCS-MERKLE"),
    ] {
        let vdir = f.dir.0.join(format!("verifier-{fault}"));
        let pdir = f.dir.0.join(format!("prover-{fault}"));
        fs::create_dir(&vdir).unwrap();
        fs::create_dir(&pdir).unwrap();
        let mut va = f.verify(true);
        set(&mut va, "--connect", vdir.display());
        let mut verifier = f.process(true, &va);
        let mut pa = f.prove(true);
        set(&mut pa, "--live", pdir.display());
        let mut producer = f.process(false, &pa);
        let names = [
            "00-SeedCommit.tce",
            "01-Round1.tce",
            "01-ProverSeedShare.tce",
            "02-SeedReveal.tce",
            "03-Round2Response.tce",
            "04-ColumnQueries.tce",
            "05-Round3Response.tce",
        ];
        for (i, name) in names.iter().enumerate() {
            let (from, to, peer) = if [0, 3, 5].contains(&i) {
                (&vdir, &pdir, &mut verifier)
            } else {
                (&pdir, &vdir, &mut producer)
            };
            let mut bytes = packet(&from.join(name), peer);
            if i == fault {
                let limits = DecodeLimits::MESSAGE;
                let payload = match fault {
                    1 => {
                        let mut r = Round1::open(&bytes, &limits).unwrap();
                        r.protocol_version += 1;
                        r.to_tce().unwrap()
                    }
                    4 => {
                        let mut r = Round2Response::open(&bytes, &limits).unwrap();
                        r.advice_combined[0].u[0] += tv_field::Fp2::ONE;
                        r.to_tce().unwrap()
                    }
                    6 => {
                        let mut r = Round3Response::open(&bytes, &limits).unwrap();
                        r.advice_columns[0][0] += tv_field::Fp::ONE;
                        r.to_tce().unwrap()
                    }
                    _ => unreachable!(),
                };
                bytes = reseal(&bytes, &payload);
            }
            let temp = to.join("relay.tmp");
            fs::write(&temp, bytes).unwrap();
            fs::rename(temp, to.join(name)).unwrap();
            // Round1 is consumed together with the prover seed share.
            if i == fault.max(2) {
                break;
            }
        }
        let bad = report(&verifier.wait(exit), "verdict");
        assert_eq!(
            bad["verdict"][if exit == 10 { "reason" } else { "error" }]["code"],
            code
        );
        if fault <= 2 {
            assert!(!vdir.join(names[3]).exists());
        }
        if fault <= 4 {
            assert!(!vdir.join(names[5]).exists());
        }
        if fault == 6 {
            producer.wait(0);
        } else {
            drop(producer);
        }
    }
}

const JSON_TRACE: &[&str] = &["--log-format", "json", "--log-level", "trace"];
const HUMAN_TRACE: &[&str] = &["--log-format", "human", "--log-level", "trace"];
/// The extensions of the secret-class artifacts of 03 §5.
const SECRET: [&str; 5] = ["d", "twc", "tst", "ttk", "tlg"];

/// Every canary form found under the fixture's directory, outside the secret-class artifacts,
/// and the files scanned, by path relative to it.
fn scan(f: &Fixture, canaries: &canary::Canaries) -> (Vec<String>, Vec<String>) {
    // The paths the test passes are public, and a directory's name can spell a token id.
    let mut public: Vec<String> = [f.dir.0.clone(), root(), std::env::temp_dir()]
        .iter()
        .flat_map(|p| [p.clone(), p.canonicalize().unwrap()])
        .map(|p| p.display().to_string())
        .collect();
    public.sort_by_key(|p| std::cmp::Reverse(p.len()));
    let (mut findings, mut scanned) = (Vec::new(), Vec::new());
    let mut pending = vec![f.dir.0.clone()];
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
                continue;
            }
            let name = path.strip_prefix(&f.dir.0).unwrap().display().to_string();
            if path
                .extension()
                .is_some_and(|e| SECRET.iter().any(|s| e == *s))
            {
                continue;
            }
            let mut bytes = fs::read(&path).unwrap();
            if let Ok(text) = std::str::from_utf8(&bytes) {
                let mut text = text.to_owned();
                for p in &public {
                    text = text.replace(p.as_str(), "<path>");
                }
                bytes = text.into_bytes();
            }
            findings.extend(
                canaries
                    .find(&bytes)
                    .into_iter()
                    .map(|form| format!("{name}: {form}")),
            );
            scanned.push(name);
        }
    }
    scanned.sort();
    (findings, scanned)
}

/// OBS-3 (05 §3), 07 §6's `e2e_redaction`: a full cycle of the canary model, each producer
/// command logged at `trace` in both formats, leaves no canary in a log line, report, protocol
/// artifact or error message, while the secret-class files it reads carry every one; and the scan
/// finds a canary injected into those artifacts.
#[test]
fn e2e_redaction_canary_scan() {
    let vectors = root().join("spec/vectors/canary");
    let manifest: Value =
        serde_json::from_slice(&fs::read(vectors.join("canaries.json")).unwrap()).unwrap();
    let canaries = canary::Canaries::new(&manifest);
    let model = canaries.find(&fs::read(vectors.join(CANARY.model)).unwrap());
    let runs = manifest["weight_runs"].as_array().unwrap().len();
    assert_eq!(
        model
            .iter()
            .filter(|f| f.ends_with(" little-endian bytes"))
            .count(),
        runs,
        "{model:?}"
    );
    for (file, sequences) in [
        ("recorded.ttk", ["prompt", "continuation"]),
        ("mismatch.ttk", ["prompt", "altered continuation"]),
    ] {
        let found = canaries.find(&fs::read(vectors.join(file)).unwrap());
        for s in sequences {
            assert!(
                found.iter().any(|f| f.starts_with(&format!("{s} ["))),
                "{file}: {found:?}"
            );
        }
    }
    for log in [JSON_TRACE, HUMAN_TRACE] {
        let f = Fixture::with(&CANARY, log);
        for live in [false, true] {
            let (produced, verdict) = f.proved(live);
            report(&produced, "proof-run");
            f.accepted(&verdict);
        }
        // The errors the canary's inputs can raise: an exact mismatch, a usage error, a corrupt
        // proof and a wrong pin.
        let mut args = f.prove(false);
        set(
            &mut args,
            "--reference-tokens",
            vectors.join("mismatch.ttk").display(),
        );
        set(&mut args, "--out", f.path("mismatch.tnp"));
        let bad: Value = serde_json::from_str(&f.run(false, &args, 20)).unwrap();
        assert_eq!(bad["error"]["code"], "ERR-WITNESS-EXACT-MISMATCH");
        set(
            &mut args,
            "--prompt-len",
            CANARY.prompt_len + CANARY.output_len,
        );
        let bad: Value = serde_json::from_str(&f.run(false, &args, 64)).unwrap();
        assert_eq!(bad["error"]["code"], "ERR-USAGE");
        let mut corrupt = fs::read(f.path("proof.tnp")).unwrap();
        corrupt[1000] ^= 1;
        fs::write(f.path("corrupt.tnp"), corrupt).unwrap();
        let mut args = f.verify(false);
        set(&mut args, "--proof", f.path("corrupt.tnp"));
        set(&mut args, "--out", f.path("corrupt.json"));
        let bad = report(&f.run(true, &args, 65), "verdict");
        assert_eq!(bad["verdict"]["error"]["code"], "ERR-CODEC-DIGEST");
        let mut args = f.verify(false);
        set(&mut args, "--statement-digest", "ab".repeat(32));
        set(&mut args, "--out", f.path("pin.json"));
        report(&f.run(true, &args, 64), "verdict");
        // Every artifact rendered, the secret-class ones included: a rendering is a report.
        let mut artifacts = vec![
            f.model.clone(),
            f.tokens.clone(),
            f.path("compiled/claims.tcl"),
            f.path("compiled/plan.tep"),
            f.path("weights.twr"),
            f.path("weights.twc"),
            f.path("proof.tnp"),
        ];
        for entry in fs::read_dir(f.path("live")).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "tce") {
                artifacts.push(path.display().to_string());
            }
        }
        for artifact in artifacts {
            f.run(false, &["--json".into(), "inspect".into(), artifact], 0);
        }
        let (findings, scanned) = scan(&f, &canaries);
        assert!(findings.is_empty(), "{findings:#?}");
        for name in [
            "proof.tnp",
            "live/05-Round3Response.tce",
            "compiled/plan.tep",
            "verdict.json",
        ] {
            assert!(scanned.iter().any(|s| s == name), "{name}: {scanned:?}");
        }
        assert!(!scanned.iter().any(|s| s == "weights.twc"));
        // Each producer command logged at `trace`.
        let traced = scanned.iter().filter(|s| s.ends_with(".stderr")).any(|s| {
            fs::read_to_string(f.dir.0.join(s))
                .unwrap()
                .lines()
                .any(|l| l.starts_with("trace ") || l.contains(r#""level":"trace""#))
        });
        assert!(traced, "{log:?}");
        // Every secret-class input and output, and the proof, rendered.
        let rendered: Vec<String> = scanned
            .iter()
            .filter(|s| s.ends_with(".stdout"))
            .map(|s| fs::read_to_string(f.dir.0.join(s)).unwrap())
            .collect();
        for schema in [
            "IntegerModel",
            "TokenFile",
            "WeightCommitmentState",
            "NiProof",
        ] {
            let field = format!(r#""schema": "{schema}""#);
            assert!(rendered.iter().any(|r| r.contains(&field)), "{schema}");
        }
        // A canary injected into a copy of the proof and of a log is found where it was put.
        let mut proof = fs::read(f.path("proof.tnp")).unwrap();
        let run: Vec<u8> = manifest["weight_runs"][0]["weights"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| w.as_i64().unwrap() as u8)
            .collect();
        proof.splice(4096..4096, run);
        let stderr = scanned.iter().find(|s| s.ends_with(".stderr")).unwrap();
        let mut text = fs::read_to_string(f.dir.0.join(stderr)).unwrap();
        let token = &manifest["token_ids"][0];
        text.push_str(&format!("token {token}\n"));
        fs::create_dir(f.path("injected")).unwrap();
        fs::write(f.path("injected/proof.tnp"), proof).unwrap();
        fs::write(f.path("injected/log.stderr"), text).unwrap();
        let (mut findings, _) = scan(&f, &canaries);
        findings.sort();
        let value = manifest["weight_runs"][0]["value"].as_str().unwrap();
        assert_eq!(
            findings,
            [
                format!("injected/log.stderr: token {token} decimal"),
                format!("injected/proof.tnp: Embed 0 weights {value} little-endian bytes"),
            ]
        );
    }
}
