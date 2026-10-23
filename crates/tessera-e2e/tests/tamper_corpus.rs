// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The SC-4 tamper corpus (07 §4) through the independent verifier's public API: every
//! non-interactive case of `spec/vectors/tamper/protocol-1.json` yields its predeclared verdict.
//! The recorded live cases replay inside `tv-core`, whose public `LiveVerifier` draws its own
//! entropy; `tv-core`'s corpus test judges both sessions.
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use tv_codec::{DecodeLimits, Envelope, WeightCommitmentPublic};
use tv_core::claims::ClaimList;
use tv_core::params::ParamSet;
use tv_core::protocol::{
    BatchDescriptor, Decoding, FixedBits, PredictorSpec, PublicStatement, RequestRef,
};
use tv_core::{ResourceLimits, Verdict, VerifierConfig, verify_non_interactive_bytes};
use tv_hash::{Domain, hash};

const P: u64 = 0xFFFF_FFFF_0000_0001;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/tamper")
}
fn base(name: &str) -> Vec<u8> {
    fs::read(dir().join(name)).unwrap()
}
fn unhex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
        .collect()
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The edited NI envelope of `case` (RFC-0003 'Envelope' for the reseal).
fn mutate(case: &Value) -> Vec<u8> {
    let original = base("ni.tnp");
    let reseal = case["reseal"].as_bool().unwrap();
    let mut bytes = if reseal {
        original[20..original.len() - 32].to_vec()
    } else {
        original.clone()
    };
    for edit in case["edits"].as_array().unwrap() {
        assert_eq!(edit["target"], "ni.tnp");
        let at = usize::try_from(edit["offset"].as_u64().unwrap()).unwrap();
        if let Some(shift) = edit.get("shift_fp2") {
            let count = usize::try_from(shift["count"].as_u64().unwrap()).unwrap();
            let delta: Vec<u64> = shift["delta"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_u64().unwrap())
                .collect();
            for (i, word) in bytes[at..at + 16 * count].chunks_exact_mut(8).enumerate() {
                let x = u64::from_le_bytes(word.try_into().unwrap());
                assert!(x < P && delta[i % 2] < P);
                let sum = (u128::from(x) + u128::from(delta[i % 2])) % u128::from(P);
                word.copy_from_slice(&u64::try_from(sum).unwrap().to_le_bytes());
            }
        } else {
            let before = unhex(edit["before"].as_str().unwrap());
            assert_eq!(&bytes[at..at + before.len()], before, "{}", case["id"]);
            bytes.splice(
                at..at + before.len(),
                unhex(edit["after"].as_str().unwrap()),
            );
        }
    }
    if reseal {
        let mut out = original[..12].to_vec();
        out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&bytes);
        out.extend_from_slice(&hash(Domain::Envelope, &out));
        bytes = out;
    }
    bytes
}

#[test]
fn every_non_interactive_case_yields_its_predeclared_verdict() {
    let corpus: Value = serde_json::from_slice(&base("protocol-1.json")).unwrap();
    assert_eq!(corpus["schema"], "tessera.tamper-corpus/1");
    for (name, digest) in corpus["base_sha256"].as_object().unwrap() {
        assert_eq!(
            hex(&tv_hash::sha256::sha256(&base(name))),
            digest.as_str().unwrap()
        );
    }
    // The application's trusted inputs: claim list, epoch artifact and request metadata.
    let limits = DecodeLimits::MESSAGE;
    let claims = ClaimList::open(&base("claims.tcl"), &limits).unwrap();
    let weights = Envelope::open(
        &base("weights.twr"),
        WeightCommitmentPublic::EXPECTED,
        &limits,
        |b| hash(Domain::Envelope, b),
    )
    .unwrap()
    .decode_payload(|c, _| WeightCommitmentPublic::decode(c))
    .unwrap();
    let request = &corpus["base"]["request"];
    let expected = PublicStatement {
        claim_list_digest: claims.digest(),
        weights_root: weights.weights_root(|b| hash(Domain::WeightsRoot, b)),
        weights_epoch: weights.epoch,
        manifest_digest: unhex(request["manifest_digest"].as_str().unwrap())
            .try_into()
            .unwrap(),
        batch: BatchDescriptor {
            requests: vec![RequestRef {
                request_id: request["request_id"].as_u64().unwrap(),
                session_id: unhex(request["session_id"].as_str().unwrap())
                    .try_into()
                    .unwrap(),
                prompt_len: u32::try_from(request["prompt_len"].as_u64().unwrap()).unwrap(),
                output_len: u32::try_from(request["output_len"].as_u64().unwrap()).unwrap(),
            }],
        },
        predictor: PredictorSpec::Exact {
            decoding: Decoding::Greedy,
        },
    };
    assert_eq!(
        hex(&expected.digest()),
        corpus["base"]["statement_digest"].as_str().unwrap()
    );
    let config = VerifierConfig {
        params: ParamSet::TestSmall,
        limits: ResourceLimits::default(),
        accept_bound: FixedBits(0),
        weights,
    };
    let verify = |bytes: &[u8]| verify_non_interactive_bytes(&claims, &expected, bytes, &config);
    let accepted = verify(&base("ni.tnp"));
    assert!(matches!(accepted, Verdict::Accept(_)), "{accepted:?}");

    let mut judged: BTreeMap<String, usize> = BTreeMap::new();
    let mut wrong = Vec::new();
    for case in corpus["mutations"].as_array().unwrap() {
        let targets: Vec<&str> = case["edits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["target"].as_str().unwrap())
            .collect();
        if targets.iter().all(|t| *t != "ni.tnp") {
            *judged
                .entry("recorded live, in tv-core".into())
                .or_default() += 1;
            continue;
        }
        *judged
            .entry(case["family"].as_str().unwrap().into())
            .or_default() += 1;
        let report: Value =
            serde_json::from_str(&verify(&mutate(case)).to_json().unwrap()).unwrap();
        if report["verdict"] != case["expected"] {
            wrong.push(format!("{}: {}", case["id"], report["verdict"]));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong verdicts:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
    let total: usize = judged.values().sum();
    assert_eq!(total as u64, corpus["coverage"]["cases"].as_u64().unwrap());
    assert!(judged.len() > 10, "{judged:?}");
}
