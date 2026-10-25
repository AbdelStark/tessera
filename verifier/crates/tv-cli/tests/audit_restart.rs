// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Issue #529's criterion across processes: `tessera-verify audit commit` runs in one process,
//! which ends; `audit reveal` runs in another, from the files alone, and writes the sample a
//! reveal without the restart derives, RFC-0016's sample over the toss's shares and the logs of
//! `spec/vectors/manifest/`.
//!
//! The test spawns processes, so it lives in a test binary of its own: a child holds a copy of
//! its parent's open file descriptions until it executes its program, a ledger's directory lock
//! among them, and would hold another test's lock a while longer.

mod toss_support;

use std::fs;
use std::process::Command;
use toss_support::{arg, expected_sample, inputs, scratch, share_of, to_hex};
use tv_codec::DecodeLimits;
use tv_core::audit::{AuditSample, read_ledger};
use tv_testkit::Json;
use tv_testkit::blake3::derive_key;

#[test]
fn a_verifier_restarted_between_commit_and_reveal_derives_the_same_sample() {
    let dir = scratch("restart");
    let i = inputs(&dir);
    let toss = dir.join("toss.tvt");
    let ledger = dir.join("ledger.tvl");
    let tas = dir.join("sample.tas");
    let prover = [0x3c; 32];
    let bin = env!("CARGO_BIN_EXE_tessera-verify");
    // The first process commits and ends.
    let committed = Command::new(bin)
        .args([
            "audit",
            "commit",
            "--manifest",
            &arg(&i.manifests[1]),
            "--recorder-key",
            &i.key,
            "--policy",
            &arg(&i.policy),
            "--toss",
            &arg(&toss),
            "--json",
        ])
        .output()
        .unwrap();
    let stdout = String::from_utf8(committed.stdout).unwrap();
    assert_eq!(committed.status.code(), Some(0), "{stdout}");
    let doc = Json::parse(&stdout).unwrap();
    let commitment = doc
        .get("commitment")
        .and_then(Json::as_str)
        .unwrap()
        .to_string();
    let bytes = fs::read(&toss).unwrap();
    let share = share_of(&bytes);
    assert_eq!(
        commitment,
        to_hex(&derive_key("tessera/v1/coin-commit", &share))
    );
    // The second process holds nothing of the first but its files.
    let revealed = Command::new(bin)
        .args([
            "audit",
            "reveal",
            "--toss",
            &arg(&toss),
            "--prover-share",
            &to_hex(&prover),
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
            &arg(&ledger),
            "--new-ledger",
            "--out",
            &arg(&tas),
        ])
        .output()
        .unwrap();
    let stdout = String::from_utf8(revealed.stdout).unwrap();
    let stderr = String::from_utf8(revealed.stderr).unwrap();
    assert_eq!(revealed.status.code(), Some(0), "{stdout}{stderr}");
    let sample = AuditSample::open(&fs::read(&tas).unwrap(), &DecodeLimits::MESSAGE).unwrap();
    let commitment: [u8; 32] = derive_key("tessera/v1/coin-commit", &share);
    assert_eq!(sample, expected_sample(&i, commitment, prover, share));
    assert_eq!(
        stdout,
        format!(
            "sample {}: {} requests selected\n",
            to_hex(sample.identity()),
            sample.selected().len()
        )
    );
    assert!(!stdout.contains(&to_hex(&share)) && !stderr.contains(&to_hex(&share)));
    // Recorded as the verifier's own, and the toss cleared.
    assert_eq!(
        read_ledger(&ledger).unwrap().derived(),
        [(i.identities[1], *sample.identity())]
    );
    assert!(fs::symlink_metadata(&toss).is_err());
    fs::remove_dir_all(&dir).unwrap();
}
