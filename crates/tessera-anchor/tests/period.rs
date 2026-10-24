// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A period's verified log and its candidates (RFC-0062 'The check', RFC-0063 'Which requests a
//! period draws from'), through the public API:
//! - the check refuses a log of another count, an entry out of sequence and another root, in
//!   that order, each with `ERR-CODEC-DIGEST`;
//! - a frame's tokens are `n` for `22 + 4n` bytes and nothing else;
//! - the candidates and their lengths match `spec/vectors/audit/audit.json`, from an independent
//!   oracle, over random logs of two periods.

use serde_json::Value;
use tessera_anchor::{
    Candidate, DigestLogEntry, Direction, HeapKey, LogManifest, LogMismatch, ManifestBody,
    VerifiedLog, candidates, frame_tokens, key_id, merkle, public_key,
};
use tessera_codec::Secret;
use tessera_codec::error::ErrorCode;

fn seed() -> Secret<HeapKey<32>> {
    Secret::new(HeapKey::from_box(Box::new([0x3c; 32])))
}

/// A signed manifest over `entries`, whose `seq` start at `first_seq`.
fn manifest_over(entries: &[DigestLogEntry], first_seq: u64) -> LogManifest {
    let leaves: Vec<_> = entries.iter().map(merkle::leaf_hash).collect();
    ManifestBody {
        recorder_key_id: key_id(&public_key(&seed())),
        log_id: [2; 16],
        period_start_ns: 1_000,
        period_end_ns: 2_000,
        first_seq,
        entry_count: entries.len() as u64,
        entries_root: merkle::root(&leaves),
        prev_manifest: [0; 32],
    }
    .sign(&seed())
    .expect("a signature")
}

fn frame(seq: u64, session: u8, request: u64, length: u32) -> DigestLogEntry {
    DigestLogEntry::Frame {
        seq,
        t_ns: 1_500,
        session_id: [session; 16],
        request_id: request,
        direction: Direction::In,
        frame_index: 0,
        length,
        frame_seq: 0,
        sha256: [0; 32],
    }
}

#[test]
fn the_check_refuses_a_log_that_is_not_its_manifests_in_order() {
    let entries: Vec<_> = (0..4).map(|i| frame(10 + i, 1, i, 26)).collect();
    let manifest = manifest_over(&entries, 10);
    let log = VerifiedLog::check(&manifest, entries.clone()).expect("its own log");
    assert_eq!(log.entries(), &entries[..]);
    assert_eq!(log.manifest(), &manifest);
    // Another count, before anything else: one entry short, also out of sequence.
    let short = entries[1..].to_vec();
    assert_eq!(
        VerifiedLog::check(&manifest, short),
        Err(LogMismatch::Count {
            expected: 4,
            found: 3
        })
    );
    // An entry out of sequence, before the root.
    let mut swapped = entries.clone();
    swapped.swap(1, 2);
    assert_eq!(
        VerifiedLog::check(&manifest, swapped),
        Err(LogMismatch::Seq { index: 1 })
    );
    // The right count and sequence, another entry: another root.
    let mut changed = entries.clone();
    changed[3] = frame(13, 1, 3, 30);
    assert_eq!(
        VerifiedLog::check(&manifest, changed),
        Err(LogMismatch::Root)
    );
    for mismatch in [
        LogMismatch::Count {
            expected: 1,
            found: 0,
        },
        LogMismatch::Seq { index: 0 },
        LogMismatch::Root,
    ] {
        assert_eq!(mismatch.code(), ErrorCode::CodecDigest);
    }
    // A sequence that would pass `u64::MAX` is out of sequence, not a wrap.
    let late = vec![frame(u64::MAX, 1, 0, 26), frame(0, 1, 1, 26)];
    let manifest = manifest_over(&late, u64::MAX);
    assert_eq!(
        VerifiedLog::check(&manifest, late),
        Err(LogMismatch::Seq { index: 1 })
    );
}

#[test]
fn a_frame_has_tokens_only_at_22_plus_4n_bytes() {
    assert_eq!(frame_tokens(22), Some(0));
    assert_eq!(frame_tokens(26), Some(1));
    assert_eq!(frame_tokens(22 + 4 * 1000), Some(1000));
    for length in [0, 1, 21, 23, 24, 25, 27, u32::MAX] {
        assert_eq!(frame_tokens(length), None, "{length}");
    }
    assert_eq!(
        frame_tokens(u32::MAX - 1),
        Some(u64::from((u32::MAX - 1 - 22) / 4))
    );
}

// ----- The oracle's logs ------------------------------------------------------------------------

fn hex(v: &Value) -> Vec<u8> {
    let s = v.as_str().expect("hex");
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex"))
        .collect()
}

fn session(v: &Value) -> [u8; 16] {
    hex(v).try_into().expect("16 bytes")
}

/// The oracle's entries of one period, numbered from `first_seq`.
fn entries(v: &Value, first_seq: u64) -> Vec<DigestLogEntry> {
    v.as_array()
        .expect("entries")
        .iter()
        .zip(first_seq..)
        .map(|(e, seq)| {
            let session_id = session(&e["session"]);
            let t_ns = 1_500;
            match e["kind"].as_str().expect("a kind") {
                "open" => DigestLogEntry::SessionOpen {
                    seq,
                    t_ns,
                    session_id,
                    key_commitment: [7; 32],
                },
                "carry" => DigestLogEntry::SessionCarry {
                    seq,
                    t_ns,
                    session_id,
                    key_commitment: [7; 32],
                },
                "close" => DigestLogEntry::SessionClose {
                    seq,
                    t_ns,
                    session_id,
                },
                "end" => DigestLogEntry::RequestEnd {
                    seq,
                    t_ns,
                    session_id,
                    request_id: e["request"].as_u64().expect("a request"),
                },
                "frame" => DigestLogEntry::Frame {
                    seq,
                    t_ns,
                    session_id,
                    request_id: e["request"].as_u64().expect("a request"),
                    direction: match e["direction"].as_str().expect("a direction") {
                        "In" => Direction::In,
                        _ => Direction::Out,
                    },
                    frame_index: e["index"].as_u64().expect("an index") as u32,
                    length: e["length"].as_u64().expect("a length") as u32,
                    frame_seq: seq as u32,
                    sha256: [seq as u8; 32],
                },
                other => panic!("an entry of kind {other}"),
            }
        })
        .collect()
}

#[test]
fn the_candidates_are_the_oracles() {
    let v: Value = serde_json::from_str(include_str!("../../../spec/vectors/audit/audit.json"))
        .expect("the vectors");
    let cases = v["candidates"].as_array().expect("candidate cases");
    assert!(cases.len() >= 40);
    for (i, case) in cases.iter().enumerate() {
        let previous = (!case["previous"].is_null()).then(|| {
            let log = entries(&case["previous"], 0);
            VerifiedLog::check(&manifest_over(&log, 0), log).expect("the previous log")
        });
        let first = previous.as_ref().map_or(0, |p| p.entries().len() as u64);
        let log = entries(&case["current"], first);
        let current = VerifiedLog::check(&manifest_over(&log, first), log).expect("the log");
        let expected: Vec<Candidate> = case["candidates"]
            .as_array()
            .expect("candidates")
            .iter()
            .map(|c| Candidate {
                session_id: session(&c["key"]["session_id"]),
                request_id: c["key"]["request_id"].as_u64().expect("a request"),
                tokens: c["tokens"].as_u64().expect("tokens"),
                ended: c["ended"].as_bool().expect("ended"),
            })
            .collect();
        assert_eq!(
            candidates(previous.as_ref(), &current),
            expected,
            "case {i}"
        );
    }
}

#[test]
fn a_period_of_zero_length_has_no_candidates() {
    // The previous period: request 4 framed, not ended, an unterminated candidate of the next
    // period of non-zero length.
    let before = vec![frame(0, 1, 4, 26)];
    let previous = VerifiedLog::check(&manifest_over(&before, 0), before).expect("its log");
    let empty = VerifiedLog::check(&manifest_over(&[], 1), Vec::new()).expect("an empty log");
    let unterminated = candidates(Some(&previous), &empty);
    assert_eq!(unterminated.len(), 1);
    assert!(!unterminated[0].ended);
    // The same empty log over a period of zero length draws nothing.
    let zero = ManifestBody {
        period_end_ns: 1_000,
        ..manifest_over(&[], 1).body
    }
    .sign(&seed())
    .expect("a signature");
    let zero = VerifiedLog::check(&zero, Vec::new()).expect("an empty log");
    assert_eq!(candidates(Some(&previous), &zero), []);
}
