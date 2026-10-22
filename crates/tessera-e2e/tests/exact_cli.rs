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
struct Fixture {
    dir: Dir,
    model: String,
    tokens: String,
    pin: String,
    weights_root: String,
}
impl Fixture {
    fn new() -> Self {
        let mut f = Self {
            dir: Dir::new(),
            model: root()
                .join("spec/vectors/exact-cli/tiny-3-reference.d")
                .display()
                .to_string(),
            tokens: root()
                .join("spec/vectors/exact-cli/recorded.ttk")
                .display()
                .to_string(),
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
        assert_eq!(compile["model"]["n_layers"], 3);
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
                    prompt_len: 2,
                    output_len: 3,
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
        Process::start(
            Command::new(if verifier {
                &binaries().1
            } else {
                &binaries().0
            })
            .args(args),
            &self.dir.0,
        )
    }
    fn run(&self, verifier: bool, args: &[String], expected: i32) -> String {
        self.process(verifier, args).wait(expected)
    }
    fn prove(&self, live: bool) -> Vec<String> {
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
            "2",
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
                    "2",
                    "--output-len",
                    "3",
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
        if live {
            fs::create_dir(f.path("live")).unwrap();
        }
        let verifier = live.then(|| f.process(true, &f.verify(true)));
        let mut producer = f.process(false, &f.prove(live));
        let (produced, verdict) = match verifier {
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
            None => (producer.wait(0), f.run(true, &f.verify(false), 0)),
        };
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
