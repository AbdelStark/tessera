// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tessera-verify audit commit` and `audit reveal` (02 §2.3, RFC-0067 'The pending toss'), on
//! the signed manifests and logs of `spec/vectors/manifest/`: the commitment printed only once
//! the toss is kept, the reveal recording the verifier's own sample before it writes it, each
//! refusal with its exit code and error, a reveal stopped after its record and run again, a
//! ledger created only by `--new-ledger`, a reveal through periods of zero length named by
//! `--between`, and no output that holds the share. The restart across processes is
//! `audit_restart.rs`'s.

mod toss_support;

use std::fs;
use std::path::Path;
use toss_support::{
    Inputs, arg, derive, expected_sample, inputs, policy_json, scratch, share_of, to_hex,
    zero_length_chain,
};
use tv_codec::DecodeLimits;
use tv_core::audit::{AuditSample, CoinTossTranscript, LedgerFile, read_ledger};
use tv_hash::{Domain, hash};
use tv_testkit::Json;
use tv_testkit::blake3::derive_key;

const PROVER: [u8; 32] = [0x5a; 32];

fn run(args: &[String]) -> tv_cli::CommandOutput {
    tv_cli::run(args.to_vec())
}

fn strings(args: &[&str]) -> Vec<String> {
    args.iter().map(|s| s.to_string()).collect()
}

/// `audit commit` over period `period` into `toss`.
fn commit_args(i: &Inputs, period: usize, toss: &Path) -> Vec<String> {
    strings(&[
        "audit",
        "commit",
        "--manifest",
        &arg(&i.manifests[period]),
        "--recorder-key",
        &i.key,
        "--policy",
        &arg(&i.policy),
        "--toss",
        &arg(toss),
    ])
}

/// `audit reveal` of `toss` over the second period, the first named as the previous one.
fn reveal_args(
    i: &Inputs,
    toss: &Path,
    prover: &[u8; 32],
    ledger: &Path,
    out: &Path,
) -> Vec<String> {
    strings(&[
        "audit",
        "reveal",
        "--toss",
        &arg(toss),
        "--prover-share",
        &to_hex(prover),
        "--manifest",
        &arg(&i.manifests[1]),
        "--recorder-key",
        &i.key,
        "--log",
        &arg(&i.logs[1]),
        "--previous-manifest",
        &arg(&i.manifests[0]),
        "--previous-log",
        &arg(&i.logs[0]),
        "--ledger",
        &arg(ledger),
        "--out",
        &arg(out),
    ])
}

fn json(mut args: Vec<String>) -> Vec<String> {
    args.push("--json".into());
    args
}

/// `args` with `--new-ledger`, which creates the ledger.
fn new_ledger(mut args: Vec<String>) -> Vec<String> {
    args.push("--new-ledger".into());
    args
}

/// `args` with `--between path` added for each of `paths`, in order.
fn between(mut args: Vec<String>, paths: &[&Path]) -> Vec<String> {
    for path in paths {
        args.extend(["--between".to_string(), arg(path)]);
    }
    args
}

/// `args` with `flag`'s value replaced by `value`.
fn with(mut args: Vec<String>, flag: &str, value: &str) -> Vec<String> {
    let at = args.iter().position(|a| a == flag).unwrap();
    args[at + 1] = value.to_string();
    args
}

/// `args` without `flag` and its value.
fn without(mut args: Vec<String>, flag: &str) -> Vec<String> {
    let at = args.iter().position(|a| a == flag).unwrap();
    args.drain(at..at + 2);
    args
}

/// The exit code and the JSON error's code of a `--json` run.
fn refusal(out: &tv_cli::CommandOutput) -> (i32, String) {
    let doc = Json::parse(&out.stdout).unwrap_or_else(|e| panic!("{e}: {}", out.stdout));
    let code = doc
        .get("error")
        .and_then(|e| e.get("code"))
        .and_then(Json::as_str)
        .unwrap_or_else(|| panic!("{}", out.stdout))
        .to_string();
    (out.exit_code, code)
}

fn text(doc: &Json, key: &str) -> String {
    doc.get(key).and_then(Json::as_str).unwrap().to_string()
}

/// RFC-0003's envelope of schema `0x0702` version 1 around `payload`, with the test kit's
/// BLAKE3.
fn seal_toss(payload: &[u8]) -> Vec<u8> {
    let mut out = [
        &b"TSRA"[..],
        &1u16.to_le_bytes(),
        &0x0702u16.to_le_bytes(),
        &1u16.to_le_bytes(),
        &0u16.to_le_bytes(),
        &(payload.len() as u64).to_le_bytes(),
        payload,
    ]
    .concat();
    let digest = derive_key("tessera/v1/envelope", &out);
    out.extend_from_slice(&digest);
    out
}

/// Whether `haystack` holds `needle` as consecutive bytes.
fn holds(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

/// No byte of `out` spells the share: neither its raw bytes nor its hexadecimal, of either case.
fn keeps_secret(out: &tv_cli::CommandOutput, share: &[u8; 32], what: &str) {
    let hex = to_hex(share);
    for stream in [&out.stdout, &out.stderr] {
        let bytes = stream.as_bytes();
        assert!(!holds(bytes, share), "{what}: raw share in {stream}");
        assert!(
            !stream.to_lowercase().contains(&hex),
            "{what}: the share in hexadecimal in {stream}"
        );
    }
}

// ----- The commands ----------------------------------------------------------------------------

/// The commitment is printed once the toss is kept, and is `H("coin-commit", share)` of the
/// file's share; the reveal writes RFC-0016's sample over the shares and the logs, which the
/// ledger records as the verifier's own and the report then accepts as such; and the toss is
/// gone. A second toss over the period is refused by the ledger, `ERR-PROTO-ORDER`.
#[test]
fn the_cli_commits_then_reveals_the_verifiers_own_sample() {
    let dir = scratch("flow");
    let i = inputs(&dir);
    let toss = dir.join("toss.tvt");
    let out = run(&json(commit_args(&i, 1, &toss)));
    assert_eq!(out.exit_code, 0, "{}", out.stdout);
    let bytes = fs::read(&toss).unwrap();
    let share = share_of(&bytes);
    let commitment = hash(Domain::CoinCommit, &share);
    assert_eq!(commitment, derive_key("tessera/v1/coin-commit", &share));
    let doc = Json::parse(&out.stdout).unwrap();
    assert_eq!(text(&doc, "command"), "audit commit");
    assert_eq!(text(&doc, "commitment"), to_hex(&commitment));
    assert_eq!(text(&doc, "manifest_digest"), to_hex(&i.identities[1]));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = fs::metadata(&toss).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }
    // Without --json the commitment alone, on a line.
    let plain = dir.join("plain.tvt");
    let out = run(&commit_args(&i, 1, &plain));
    assert_eq!(out.exit_code, 0, "{}", out.stderr);
    let other = hash(Domain::CoinCommit, &share_of(&fs::read(&plain).unwrap()));
    assert_eq!(out.stdout, format!("{}\n", to_hex(&other)));
    assert!(out.stderr.is_empty());
    // The reveal.
    let ledger = dir.join("ledger.tvl");
    let tas = dir.join("sample.tas");
    let out = run(&json(new_ledger(reveal_args(
        &i, &toss, &PROVER, &ledger, &tas,
    ))));
    assert_eq!(out.exit_code, 0, "{}", out.stdout);
    let sample = AuditSample::open(&fs::read(&tas).unwrap(), &DecodeLimits::MESSAGE).unwrap();
    assert_eq!(sample, expected_sample(&i, commitment, PROVER, share));
    let doc = Json::parse(&out.stdout).unwrap();
    assert_eq!(text(&doc, "command"), "audit reveal");
    assert_eq!(text(&doc, "sample"), to_hex(sample.identity()));
    assert_eq!(text(&doc, "manifest_digest"), to_hex(&i.identities[1]));
    assert_eq!(
        doc.get("selected").and_then(Json::as_u64),
        Some(sample.selected().len() as u64)
    );
    assert_eq!(doc.get("ledger_created"), Some(&Json::Bool(true)));
    assert_eq!(
        read_ledger(&ledger).unwrap().derived(),
        [(i.identities[1], *sample.identity())]
    );
    assert!(fs::symlink_metadata(&toss).is_err(), "the toss is cleared");
    // The report takes the sample as the verifier's own: every selected request unproven.
    let report = run(&strings(&[
        "audit",
        "report",
        "--ledger",
        &arg(&ledger),
        "--sample",
        &arg(&tas),
        "--out",
        &arg(&dir.join("report.json")),
        "--json",
    ]));
    let expected = if sample.selected().is_empty() { 0 } else { 10 };
    assert_eq!(report.exit_code, expected, "{}", report.stdout);
    // A second toss over the period: the ledger takes no other sample, and the toss is kept.
    let kept = fs::read(&plain).unwrap();
    let second = dir.join("second.tas");
    let out = run(&json(reveal_args(&i, &plain, &PROVER, &ledger, &second)));
    assert_eq!(refusal(&out), (20, "ERR-PROTO-ORDER".into()));
    assert_eq!(fs::read(&plain).unwrap(), kept);
    assert!(!second.exists());
    assert_eq!(read_ledger(&ledger).unwrap().derived().len(), 1);
    fs::remove_dir_all(&dir).unwrap();
}

/// Each refusal of 02 §2.3 and §2.4, alone: the exit code and the error, the toss left as it
/// was, no ledger created and no sample written.
#[test]
fn every_refusal_of_commit_and_reveal_exits_with_its_code() {
    let dir = scratch("refusals");
    let i = inputs(&dir);
    let toss = dir.join("toss.tvt");
    assert_eq!(run(&commit_args(&i, 1, &toss)).exit_code, 0);
    let bytes = fs::read(&toss).unwrap();
    let ledger = dir.join("ledger.tvl");
    let tas = dir.join("sample.tas");
    let unpinned = "11".repeat(32);

    // ----- commit
    let fresh = dir.join("fresh.tvt");
    let commits = [
        (
            "an existing toss",
            json(commit_args(&i, 1, &toss)),
            (64, "ERR-USAGE"),
        ),
        (
            "an unpinned recorder key",
            with(
                json(commit_args(&i, 1, &fresh)),
                "--recorder-key",
                &unpinned,
            ),
            (10, "REJ-ANCHOR"),
        ),
        (
            "a recorder key that is not hexadecimal",
            with(json(commit_args(&i, 1, &fresh)), "--recorder-key", "zz"),
            (64, "ERR-USAGE"),
        ),
        (
            "a log given as the manifest",
            with(
                json(commit_args(&i, 1, &fresh)),
                "--manifest",
                &arg(&i.logs[1]),
            ),
            (65, "ERR-CODEC-MAGIC"),
        ),
        (
            "a missing option",
            without(json(commit_args(&i, 1, &fresh)), "--policy"),
            (64, "ERR-USAGE"),
        ),
    ];
    for (what, args, (exit, code)) in commits {
        let out = run(&args);
        assert_eq!(refusal(&out), (exit, code.to_string()), "{what}");
        assert!(!out.stdout.contains("commitment\""), "{what}");
        assert!(!fresh.exists(), "{what}");
        assert_eq!(fs::read(&toss).unwrap(), bytes, "{what}");
    }
    let bad_policy = dir.join("bad-policy.json");
    fs::write(&bad_policy, policy_json("2")).unwrap();
    let out = run(&with(
        json(commit_args(&i, 1, &fresh)),
        "--policy",
        &arg(&bad_policy),
    ));
    assert_eq!(refusal(&out), (64, "ERR-USAGE".into()));
    assert!(!fresh.exists());

    // ----- reveal
    let reveal = || json(reveal_args(&i, &toss, &PROVER, &ledger, &tas));
    let other = dir.join("other.tvt");
    assert_eq!(run(&commit_args(&i, 0, &other)).exit_code, 0);
    let mut flipped = bytes.clone();
    let at = flipped.len() - 33;
    flipped[at] ^= 1;
    let broken = dir.join("broken.tvt");
    fs::write(&broken, &flipped).unwrap();
    let resealed = dir.join("resealed.tvt");
    fs::write(&resealed, seal_toss(&flipped[20..flipped.len() - 32])).unwrap();
    #[cfg(unix)]
    let link = {
        let link = dir.join("link.tvt");
        std::os::unix::fs::symlink(&toss, &link).unwrap();
        link
    };
    let reveals = [
        (
            "a toss of another manifest",
            with(reveal(), "--toss", &arg(&other)),
            (64, "ERR-USAGE"),
        ),
        (
            "a share flipped and the envelope resealed",
            with(reveal(), "--toss", &arg(&resealed)),
            (20, "ERR-PROTO-COINTOSS"),
        ),
        (
            "a share flipped, the envelope broken",
            with(reveal(), "--toss", &arg(&broken)),
            (65, "ERR-CODEC-DIGEST"),
        ),
        #[cfg(unix)]
        (
            "a toss behind a symbolic link",
            with(reveal(), "--toss", &arg(&link)),
            (64, "ERR-USAGE"),
        ),
        (
            "a lost toss",
            with(reveal(), "--toss", &arg(&dir.join("lost.tvt"))),
            (20, "ERR-RES-IO"),
        ),
        (
            "a log that is not its manifest's",
            with(reveal(), "--log", &arg(&i.logs[0])),
            (65, "ERR-CODEC-DIGEST"),
        ),
        (
            "an unpinned recorder key",
            with(reveal(), "--recorder-key", &unpinned),
            (10, "REJ-ANCHOR"),
        ),
        (
            "a previous manifest without its log",
            without(reveal(), "--previous-log"),
            (64, "ERR-USAGE"),
        ),
        (
            "no previous period where one exists",
            without(without(reveal(), "--previous-log"), "--previous-manifest"),
            (64, "ERR-USAGE"),
        ),
        (
            "a prover share that is not 64 hexadecimal digits",
            with(reveal(), "--prover-share", "00"),
            (64, "ERR-USAGE"),
        ),
        (
            "an output that names the toss",
            with(reveal(), "--out", &arg(&toss)),
            (64, "ERR-USAGE"),
        ),
        (
            "a missing ledger without --new-ledger",
            reveal(),
            (20, "ERR-RES-IO"),
        ),
        (
            "--new-ledger twice",
            new_ledger(new_ledger(reveal())),
            (64, "ERR-USAGE"),
        ),
        (
            "a --between manifest of non-zero length",
            between(reveal(), &[&i.manifests[0]]),
            (64, "ERR-USAGE"),
        ),
        (
            "a --between that is not a manifest",
            between(reveal(), &[&i.logs[0]]),
            (65, "ERR-CODEC-MAGIC"),
        ),
    ];
    for (what, args, (exit, code)) in reveals {
        let out = run(&args);
        assert_eq!(refusal(&out), (exit, code.to_string()), "{what}");
        assert_eq!(fs::read(&toss).unwrap(), bytes, "{what}");
        assert!(fs::symlink_metadata(&ledger).is_err(), "{what}: no ledger");
        assert!(!tas.exists(), "{what}: no sample");
    }
    // Every refused toss is left as it was.
    assert_eq!(fs::read(&broken).unwrap(), flipped);
    assert!(fs::metadata(&other).is_ok() && fs::metadata(&resealed).is_ok());
    // --new-ledger over anything at the path: another file, a ledger, a link that leads
    // nowhere. Each is refused and left as it was, and the toss is kept.
    let occupied = dir.join("occupied.tvl");
    fs::write(&occupied, b"not a ledger").unwrap();
    let held = dir.join("held");
    fs::create_dir(&held).unwrap();
    let existing = held.join("ledger.tvl");
    drop(LedgerFile::create(&existing).unwrap());
    let ledger_bytes = fs::read(&existing).unwrap();
    let mut taken = vec![
        (occupied.clone(), b"not a ledger".to_vec()),
        (existing, ledger_bytes),
    ];
    #[cfg(unix)]
    {
        let dangling = dir.join("dangling.tvl");
        std::os::unix::fs::symlink(dir.join("nowhere.tvl"), &dangling).unwrap();
        taken.push((dangling, Vec::new()));
    }
    for (path, content) in &taken {
        let out = run(&new_ledger(with(reveal(), "--ledger", &arg(path))));
        assert_eq!(
            refusal(&out),
            (64, "ERR-USAGE".into()),
            "{}",
            path.display()
        );
        if content.is_empty() {
            assert!(fs::read_link(path).is_ok() && !dir.join("nowhere.tvl").exists());
        } else {
            assert_eq!(&fs::read(path).unwrap(), content, "{}", path.display());
        }
        assert_eq!(fs::read(&toss).unwrap(), bytes);
        assert!(!tas.exists());
    }
    // The toss itself, refused by nothing, reveals into a ledger --new-ledger creates.
    let out = run(&new_ledger(reveal()));
    assert_eq!(out.exit_code, 0, "{}", out.stdout);
    let doc = Json::parse(&out.stdout).unwrap();
    assert_eq!(doc.get("ledger_created"), Some(&Json::Bool(true)));
    assert_eq!(read_ledger(&ledger).unwrap().derived().len(), 1);
    fs::remove_dir_all(&dir).unwrap();
}

/// RFC-0063 'The previous period's log': `P − 1` reaches `P` through the periods of zero length
/// between them, which `--between` names in chain order, each signed under the pinned key. The
/// vectors have no such period, so the test recorder re-signs their two periods with two of
/// zero length between them. The reveal through them writes RFC-0016's sample over the logs;
/// one missing, both missing, the two in the wrong order, one named twice, `P − 1` named among
/// them and a forged signature are each refused, and nothing is recorded or written.
#[test]
fn a_reveal_through_periods_of_zero_length_names_them_with_between() {
    let dir = scratch("between");
    let i = inputs(&dir);
    let chain = zero_length_chain(&i);
    let toss = dir.join("toss.tvt");
    let commit = with(
        with(commit_args(&i, 1, &toss), "--manifest", &arg(&chain.period)),
        "--recorder-key",
        &chain.key,
    );
    let out = run(&json(commit));
    assert_eq!(out.exit_code, 0, "{}", out.stdout);
    let doc = Json::parse(&out.stdout).unwrap();
    assert_eq!(text(&doc, "manifest_digest"), to_hex(&chain.identity));
    let bytes = fs::read(&toss).unwrap();
    let ledger = dir.join("ledger.tvl");
    let tas = dir.join("sample.tas");
    let reveal = || {
        let args = reveal_args(&i, &toss, &PROVER, &ledger, &tas);
        let args = with(args, "--manifest", &arg(&chain.period));
        let args = with(args, "--previous-manifest", &arg(&chain.previous));
        new_ledger(json(with(args, "--recorder-key", &chain.key)))
    };
    let [z1, z2] = [chain.zeros[0].as_path(), chain.zeros[1].as_path()];
    let refusals = [
        ("no --between", reveal(), (64, "ERR-USAGE")),
        ("Z1 alone", between(reveal(), &[z1]), (64, "ERR-USAGE")),
        ("Z2 alone", between(reveal(), &[z2]), (64, "ERR-USAGE")),
        (
            "Z2 before Z1",
            between(reveal(), &[z2, z1]),
            (64, "ERR-USAGE"),
        ),
        (
            "Z1 twice",
            between(reveal(), &[z1, z1, z2]),
            (64, "ERR-USAGE"),
        ),
        (
            "P − 1 among them",
            between(reveal(), &[&chain.previous, z1, z2]),
            (64, "ERR-USAGE"),
        ),
        (
            "Z1's signature forged",
            between(reveal(), &[&chain.forged, z2]),
            (10, "REJ-ANCHOR"),
        ),
    ];
    for (what, args, (exit, code)) in refusals {
        let out = run(&args);
        assert_eq!(refusal(&out), (exit, code.to_string()), "{what}");
        assert_eq!(fs::read(&toss).unwrap(), bytes, "{what}");
        assert!(fs::symlink_metadata(&ledger).is_err(), "{what}: no ledger");
        assert!(!tas.exists(), "{what}: no sample");
    }
    // Through both, in chain order: RFC-0016's sample of `P` over the window they bind.
    let out = run(&between(reveal(), &[z1, z2]));
    assert_eq!(out.exit_code, 0, "{}", out.stdout);
    let sample = AuditSample::open(&fs::read(&tas).unwrap(), &DecodeLimits::MESSAGE).unwrap();
    let share = share_of(&bytes);
    let expected = derive(
        &chain.key,
        (&chain.period, &i.logs[1]),
        (&chain.previous, &i.logs[0]),
        &[z1, z2],
        CoinTossTranscript {
            verifier_commitment: hash(Domain::CoinCommit, &share),
            prover_share: PROVER,
            verifier_share: share,
        },
    );
    assert_eq!(sample, expected);
    assert_eq!(sample.manifest_digest(), &chain.identity);
    assert_eq!(
        read_ledger(&ledger).unwrap().derived(),
        [(chain.identity, *sample.identity())]
    );
    assert!(fs::symlink_metadata(&toss).is_err());
    fs::remove_dir_all(&dir).unwrap();
}

/// RFC-0067 'Cleared after the record': a reveal stopped after the ledger's record, here by an
/// output path that cannot be written, keeps the toss. Run again against another prover share
/// it is refused by the ledger; against the same share it writes the same sample, the ledger
/// unchanged and holding one `derived` entry, and the toss is removed.
#[test]
fn a_reveal_stopped_after_its_record_is_run_again_and_changes_nothing() {
    let dir = scratch("stopped");
    let i = inputs(&dir);
    let toss = dir.join("toss.tvt");
    assert_eq!(run(&commit_args(&i, 1, &toss)).exit_code, 0);
    let bytes = fs::read(&toss).unwrap();
    let ledger = dir.join("ledger.tvl");
    // A directory where the sample should go: the rename over it fails after the record.
    let occupied = dir.join("occupied");
    fs::create_dir(&occupied).unwrap();
    let out = run(&json(new_ledger(reveal_args(
        &i, &toss, &PROVER, &ledger, &occupied,
    ))));
    assert_eq!(refusal(&out), (20, "ERR-RES-IO".into()));
    assert!(
        out.stdout
            .contains("reveal again with the same prover share")
    );
    let recorded = fs::read(&ledger).unwrap();
    let derived = read_ledger(&ledger).unwrap().derived().to_vec();
    assert_eq!(derived.len(), 1);
    assert_eq!(fs::read(&toss).unwrap(), bytes, "the toss is kept");
    assert!(fs::read_dir(&occupied).unwrap().next().is_none());
    let tas = dir.join("sample.tas");
    // Run again with --new-ledger: the ledger exists now, and is not replaced.
    let out = run(&json(new_ledger(reveal_args(
        &i, &toss, &PROVER, &ledger, &tas,
    ))));
    assert_eq!(refusal(&out), (64, "ERR-USAGE".into()));
    assert_eq!(fs::read(&ledger).unwrap(), recorded);
    assert_eq!(fs::read(&toss).unwrap(), bytes);
    // Another prover share: the ledger holds the period's sample.
    let out = run(&json(reveal_args(&i, &toss, &[0xee; 32], &ledger, &tas)));
    assert_eq!(refusal(&out), (20, "ERR-PROTO-ORDER".into()));
    assert_eq!(fs::read(&toss).unwrap(), bytes);
    assert!(!tas.exists());
    // The same share: the same sample, the ledger as the first run left it, the toss removed.
    let out = run(&json(reveal_args(&i, &toss, &PROVER, &ledger, &tas)));
    assert_eq!(out.exit_code, 0, "{}", out.stdout);
    let doc = Json::parse(&out.stdout).unwrap();
    assert_eq!(doc.get("ledger_created"), Some(&Json::Bool(false)));
    let sample = AuditSample::open(&fs::read(&tas).unwrap(), &DecodeLimits::MESSAGE).unwrap();
    let share = share_of(&bytes);
    let commitment = hash(Domain::CoinCommit, &share);
    assert_eq!(sample, expected_sample(&i, commitment, PROVER, share));
    assert_eq!(derived, [(i.identities[1], *sample.identity())]);
    assert_eq!(fs::read(&ledger).unwrap(), recorded);
    assert!(fs::symlink_metadata(&toss).is_err());
    let mut names: Vec<String> = fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert!(
        names.iter().all(|n| !n.ends_with(".tmp")),
        "no temporary file left: {names:?}"
    );
    fs::remove_dir_all(&dir).unwrap();
}

/// 05 §3: the share never reaches the CLI's output. Every output of the commands, their
/// successes and their refusals, with and without `--json`, is searched for its raw bytes and
/// its hexadecimal of either case, before the reveal and after it.
#[test]
fn the_share_never_appears_in_the_clis_output() {
    let dir = scratch("secret");
    let i = inputs(&dir);
    let toss = dir.join("toss.tvt");
    let ledger = dir.join("ledger.tvl");
    let tas = dir.join("sample.tas");
    let mut outputs = Vec::new();
    outputs.push(("commit --json", run(&json(commit_args(&i, 1, &toss)))));
    let share = share_of(&fs::read(&toss).unwrap());
    let plain = dir.join("plain.tvt");
    outputs.push(("commit", run(&commit_args(&i, 1, &plain))));
    let unpinned = "11".repeat(32);
    let occupied = dir.join("occupied");
    fs::create_dir(&occupied).unwrap();
    drop(LedgerFile::create(&ledger).unwrap());
    let reveal = |out: &Path| reveal_args(&i, &toss, &PROVER, &ledger, out);
    let refusals = [
        ("a second commit", commit_args(&i, 1, &toss)),
        (
            "an unpinned key",
            with(reveal(&tas), "--recorder-key", &unpinned),
        ),
        (
            "a log not its manifest's",
            with(reveal(&tas), "--log", &arg(&i.logs[0])),
        ),
        (
            "a missing ledger",
            with(reveal(&tas), "--ledger", &arg(&dir.join("missing.tvl"))),
        ),
        ("--new-ledger over the ledger", new_ledger(reveal(&tas))),
        ("a stop after the record", reveal(&occupied)),
        (
            "another prover share",
            with(reveal(&tas), "--prover-share", &"ee".repeat(32)),
        ),
    ];
    for (what, args) in refusals {
        outputs.push((what, run(&json(args.clone()))));
        outputs.push((what, run(&args)));
    }
    outputs.push(("reveal --json", run(&json(reveal(&tas)))));
    outputs.push(("help", run(&strings(&["audit", "--help"]))));
    // The plain toss, revealed without --json into another ledger, prints one line.
    let other = dir.join("other");
    fs::create_dir(&other).unwrap();
    let plain_share = share_of(&fs::read(&plain).unwrap());
    let last = run(&new_ledger(reveal_args(
        &i,
        &plain,
        &PROVER,
        &other.join("ledger.tvl"),
        &other.join("sample.tas"),
    )));
    assert_eq!(last.exit_code, 0, "{}", last.stderr);
    assert!(last.stdout.starts_with("sample ") && last.stdout.ends_with(" selected\n"));
    keeps_secret(&last, &plain_share, "reveal");
    for (what, out) in &outputs {
        keeps_secret(out, &share, what);
        keeps_secret(out, &plain_share, what);
    }
    // The successes and refusals happened as named.
    let codes: Vec<i32> = outputs.iter().map(|(_, o)| o.exit_code).collect();
    assert_eq!(
        codes,
        [
            0, 0, 64, 64, 10, 10, 65, 65, 20, 20, 64, 64, 20, 20, 20, 20, 0, 0
        ]
    );
    // After the reveal the share is public, in the sample, and still not printed.
    let sample = AuditSample::open(&fs::read(&tas).unwrap(), &DecodeLimits::MESSAGE).unwrap();
    assert_eq!(sample.coin_toss().verifier_share, share);
    fs::remove_dir_all(&dir).unwrap();
}
