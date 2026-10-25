// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The anchor's log checks against the independent oracles' vectors:
//!
//! - `spec/vectors/manifest/`: three signed manifests of one log. Each opens to its fields and
//!   identity, hands the strict Ed25519 seam exactly the vector's key, signed message and
//!   signature, and verifies its log; the chain links; the window of each period binds its
//!   predecessor; the second period's candidates are its two requests, which pass every anchor
//!   check; and a change to any signed byte is refused.
//! - `spec/vectors/anchor/`: every entry of five sessions decodes to the vector's fields, and the
//!   sessions, laid end to end as one period, are a log whose every request passes.
//! - `spec/vectors/audit/`: the `candidates` of every two-period log, with their lengths.

mod anchor_support;

use anchor_support::{Fields, req};
use tv_codec::{DecodeLimits, ErrorCode};
use tv_core::anchor::{
    Candidate, DigestLogEntry, Direction, LogManifest, ManifestError, RecorderKeys, SamplerPins,
    SamplerPolicy, VerifiedLog, Window, check_batch,
};
use tv_core::protocol::{BatchDescriptor, PublicStatement, RequestRef};
use tv_hash::sha256::sha256;
use tv_testkit::{Json, hex, read_vectors};

fn text<'a>(value: &'a Json, key: &str) -> &'a str {
    value
        .get(key)
        .and_then(Json::as_str)
        .unwrap_or_else(|| panic!("`{key}`"))
}
fn number(value: &Json, key: &str) -> u64 {
    value
        .get(key)
        .and_then(Json::as_u64)
        .unwrap_or_else(|| panic!("`{key}`"))
}
fn list<'a>(value: &'a Json, key: &str) -> &'a [Json] {
    value
        .get(key)
        .and_then(Json::as_array)
        .unwrap_or_else(|| panic!("`{key}`"))
}
fn bytes32(text: &str) -> [u8; 32] {
    hex(text).try_into().unwrap()
}
fn bytes16(text: &str) -> [u8; 16] {
    hex(text).try_into().unwrap()
}

// ----- spec/vectors/manifest/ -------------------------------------------------------------------

struct ManifestVector {
    tlm: Vec<u8>,
    payload: Vec<u8>,
    signed: Vec<u8>,
    signature: [u8; 64],
    identity: [u8; 32],
    log: Vec<u8>,
    entries: Vec<Vec<u8>>,
    fields: Json,
}

fn manifest_vectors() -> (Json, Vec<ManifestVector>) {
    let doc = read_vectors("manifest/manifests.json");
    let manifests = list(&doc, "manifests")
        .iter()
        .map(|m| {
            let entries: Vec<Vec<u8>> = list(m, "entries")
                .iter()
                .map(|e| hex(e.as_str().unwrap()))
                .collect();
            ManifestVector {
                tlm: hex(text(m, "envelope")),
                payload: hex(text(m, "manifest")),
                signed: hex(text(m, "signed_message")),
                signature: hex(text(m, "signature")).try_into().unwrap(),
                identity: bytes32(text(m, "identity")),
                log: entries.concat(),
                entries,
                fields: m.clone(),
            }
        })
        .collect();
    (doc, manifests)
}

/// The strict Ed25519 seam as a verifier that knows one valid triple: it accepts exactly the
/// vector's key, message and signature, as a correct strict verifier does for this key, and
/// refuses everything else.
fn knows<'a>(
    key: &'a [u8; 32],
    message: &'a [u8],
    signature: &'a [u8; 64],
) -> impl FnOnce(&[u8; 32], &[u8], &[u8; 64]) -> bool + 'a {
    move |k, m, s| k == key && m == message && s == signature
}

#[test]
fn every_manifest_opens_to_its_fields_identity_and_signed_message() {
    let (doc, manifests) = manifest_vectors();
    let public_key = bytes32(text(&doc, "public_key"));
    let key_id = bytes32(text(&doc, "recorder_key_id"));
    assert_eq!(
        sha256(&public_key),
        key_id,
        "recorder_key_id = SHA-256(public_key)"
    );
    let keys = RecorderKeys::new(vec![public_key]);
    for (i, v) in manifests.iter().enumerate() {
        let sealed = LogManifest::open(&v.tlm, &DecodeLimits::MESSAGE).unwrap();
        assert_eq!(sealed.identity(), &v.identity, "manifest {i}: identity");
        assert_eq!(
            &v.tlm[v.tlm.len() - 32..],
            &v.identity,
            "the envelope's last 32 bytes"
        );
        let m = sealed.manifest();
        assert_eq!(m.recorder_key_id, key_id);
        assert_eq!(m.log_id, bytes16(text(&doc, "log_id")));
        assert_eq!(m.period_start_ns, number(&v.fields, "period_start_ns"));
        assert_eq!(m.period_end_ns, number(&v.fields, "period_end_ns"));
        assert_eq!(m.first_seq, number(&v.fields, "first_seq"));
        assert_eq!(m.entry_count, v.entries.len() as u64);
        assert_eq!(m.entries_root, bytes32(text(&v.fields, "entries_root")));
        assert_eq!(m.prev_manifest, bytes32(text(&v.fields, "prev_manifest")));
        assert_eq!(m.signature.scheme, 0);
        assert_eq!(m.signature.public_key, public_key);
        assert_eq!(m.signature.signature, v.signature);
        assert_eq!(v.signed.len(), 163);
        assert_eq!(
            m.signed_message().as_slice(),
            v.signed.as_slice(),
            "manifest {i}"
        );
        // The signed message is the label and the payload's first 144 bytes.
        assert_eq!(&v.signed[..19], b"tessera-manifest/v1");
        assert_eq!(&v.signed[19..], &v.payload[..144]);
        let mut seen = None;
        let checked = sealed
            .check(&keys, |k, msg, s| {
                seen = Some((*k, msg.to_vec(), *s));
                true
            })
            .unwrap();
        assert_eq!(
            seen,
            Some((public_key, v.signed.clone(), v.signature)),
            "manifest {i}"
        );
        assert_eq!(checked.identity(), &v.identity);
        // The root is RFC 9162's over the vector's leaves.
        for (entry, leaf) in v.entries.iter().zip(list(&v.fields, "leaves")) {
            assert_eq!(
                sha256(&[&[0][..], entry].concat()),
                bytes32(leaf.as_str().unwrap())
            );
        }
        let log = VerifiedLog::check(&checked, &v.log, &DecodeLimits::MESSAGE).unwrap();
        assert_eq!(log.entries().len(), v.entries.len());
        assert_eq!(log.manifest_digest(), &v.identity);
    }
    // Every entry's RFC 9162 audit path verifies at its own index against the signed root and
    // count, with the verifier's tv-hash (RFC-0060 'Inclusion proofs').
    for v in &manifests {
        let root = bytes32(text(&v.fields, "entries_root"));
        for (index, (entry, proof)) in v.entries.iter().zip(list(&v.fields, "proofs")).enumerate() {
            let proof: Vec<[u8; 32]> = proof
                .as_array()
                .unwrap()
                .iter()
                .map(|d| bytes32(d.as_str().unwrap()))
                .collect();
            let leaf = tv_hash::rfc6962::leaf_hash(entry);
            let size = v.entries.len() as u64;
            assert_eq!(
                tv_hash::rfc6962::verify_inclusion(index as u64, size, &leaf, &proof, &root),
                Ok(())
            );
        }
    }
    // The third period is empty: its root is SHA-256("").
    assert_eq!(manifests[2].entries.len(), 0);
    assert_eq!(
        bytes32(text(&manifests[2].fields, "entries_root")),
        sha256(&[])
    );
}

fn verified(doc: &Json, v: &ManifestVector) -> VerifiedLog {
    let keys = RecorderKeys::new(vec![bytes32(text(doc, "public_key"))]);
    let pk = bytes32(text(doc, "public_key"));
    let checked = LogManifest::open(&v.tlm, &DecodeLimits::MESSAGE)
        .unwrap()
        .check(&keys, knows(&pk, &v.signed, &v.signature))
        .unwrap();
    VerifiedLog::check(&checked, &v.log, &DecodeLimits::MESSAGE).unwrap()
}

#[test]
fn the_manifests_chain_and_each_period_binds_its_predecessor() {
    let (doc, manifests) = manifest_vectors();
    let logs: Vec<VerifiedLog> = manifests.iter().map(|v| verified(&doc, v)).collect();
    assert!(logs[0].manifest().precedes(logs[1].manifest()));
    assert!(logs[1].manifest().precedes(logs[2].manifest()));
    assert!(!logs[0].manifest().precedes(logs[2].manifest()));
    assert!(!logs[1].manifest().precedes(logs[0].manifest()));
    // The genesis has no previous period; each later one binds the one before.
    Window::bind(&logs[0], &[], None).unwrap();
    Window::bind(&logs[1], &[], Some(&logs[0])).unwrap();
    Window::bind(&logs[2], &[], Some(&logs[1])).unwrap();
    // RFC-0063: the second period's candidates are request 7, whose frames lie in the first
    // period and whose end in the second, and request 9.
    let window = Window::bind(&logs[1], &[], Some(&logs[0])).unwrap();
    assert_eq!(
        window.candidates().unwrap(),
        vec![
            Candidate {
                session_id: [0x30; 16],
                request_id: 7,
                ended: true,
                tokens: 15
            },
            Candidate {
                session_id: [0x31; 16],
                request_id: 9,
                ended: true,
                tokens: 4
            },
        ]
    );
    // Read alone, the first period has no candidate: request 7 has no end there.
    assert_eq!(
        Window::bind(&logs[0], &[], None)
            .unwrap()
            .candidates()
            .unwrap(),
        vec![]
    );
    // The empty third period draws nothing: request 9 ended in its own period.
    assert_eq!(
        Window::bind(&logs[2], &[], Some(&logs[1]))
            .unwrap()
            .candidates()
            .unwrap(),
        vec![]
    );

    // Both requests pass the anchor checks: request 7 with three inbound frames of 0, 1 and 2
    // tokens and three outbound of 3, 4 and 5, request 9 with four outbound frames of one token.
    let statement = PublicStatement {
        claim_list_digest: [0; 32],
        weights_root: [0; 32],
        weights_epoch: 0,
        manifest_digest: manifests[1].identity,
        batch: BatchDescriptor {
            requests: vec![
                RequestRef {
                    request_id: 7,
                    session_id: [0x30; 16],
                    prompt_len: 3,
                    output_len: 12,
                },
                RequestRef {
                    request_id: 9,
                    session_id: [0x31; 16],
                    prompt_len: 0,
                    output_len: 4,
                },
            ],
        },
        predictor: anchor_support::greedy(),
    };
    let pins = SamplerPins::load(Vec::new(), SamplerPolicy::default_for(60_000_000_000)).unwrap();
    assert_eq!(check_batch(&statement, &window, &pins), Ok(()));
}

/// A change to any byte the signature covers is refused, whether or not the envelope's digest
/// is recomputed: without, the envelope fails (`Inconclusive(ERR-CODEC-*)`); with, the manifest
/// fails its own checks (`Reject(Anchor { None, ManifestSignature })`), at the key identifier for
/// its bytes and at the signature for the others.
#[test]
fn a_change_to_any_signed_byte_is_refused() {
    let (doc, manifests) = manifest_vectors();
    let pk = bytes32(text(&doc, "public_key"));
    let keys = RecorderKeys::new(vec![pk]);
    for v in &manifests {
        for at in 0..v.payload.len() {
            let mut payload = v.payload.clone();
            payload[at] ^= 0x01;
            // As delivered, with the old digest.
            let mut tlm = v.tlm.clone();
            tlm[20 + at] ^= 0x01;
            let error = LogManifest::open(&tlm, &DecodeLimits::MESSAGE).unwrap_err();
            assert_eq!(error.code(), ErrorCode::Digest, "byte {at}");
            assert_eq!(error.offset(), 20 + v.payload.len() as u64);
            // Resealed: the signature check sees the change.
            let resealed = anchor_support::seal(&payload);
            let outcome = LogManifest::open(&resealed, &DecodeLimits::MESSAGE)
                .map(|sealed| sealed.check(&keys, knows(&pk, &v.signed, &v.signature)));
            let expected = match at {
                0..32 => Err(ManifestError::KeyId),
                32..144 => Err(ManifestError::Signature),
                144..146 => Err(ManifestError::Scheme),
                // The key's length: decoding or the length check refuses it.
                146..154 => {
                    assert!(
                        matches!(outcome, Err(_) | Ok(Err(ManifestError::KeyLength))),
                        "byte {at}"
                    );
                    continue;
                }
                154..186 => Err(ManifestError::KeyId),
                186..194 => {
                    assert!(
                        matches!(outcome, Err(_) | Ok(Err(ManifestError::SignatureLength))),
                        "byte {at}"
                    );
                    continue;
                }
                _ => Err(ManifestError::Signature),
            };
            let outcome = outcome.unwrap_or_else(|e| panic!("byte {at}: {e}"));
            assert_eq!(outcome.map(|_| ()), expected, "byte {at}");
        }
    }
}

// ----- spec/vectors/anchor/ ---------------------------------------------------------------------

fn decode_one(bytes: &[u8]) -> DigestLogEntry {
    tv_codec::decode(bytes, &DecodeLimits::MESSAGE, DigestLogEntry::decode).unwrap()
}

#[test]
fn every_anchor_entry_decodes_to_its_fields() {
    let doc = read_vectors("anchor/frames.json");
    let mut decoded = 0;
    for session in list(&doc, "sessions") {
        let id = bytes16(text(session, "session_id"));
        let key = bytes32(text(session, "key_commitment"));
        for (name, tag) in [("open_entry", 0), ("carry_entry", 3), ("close_entry", 2)] {
            let entry = session.get(name).unwrap();
            let value = decode_one(&hex(text(entry, "tce")));
            assert_eq!(value.tag(), tag);
            assert_eq!(value.seq(), number(entry, "seq"));
            assert_eq!(value.t_ns(), number(entry, "t_ns"));
            assert_eq!(value.session_id(), &id);
            assert_eq!(value.key_commitment(), (tag != 2).then_some(&key));
            decoded += 1;
        }
        for frame in list(session, "frames") {
            let entry = frame.get("entry").unwrap();
            let DigestLogEntry::Frame(f) = decode_one(&hex(text(entry, "tce"))) else {
                panic!("a frame entry");
            };
            let header = hex(text(frame, "header"));
            assert_eq!(header.len(), 41, "RFC-0059: the header is 41 bytes");
            assert_eq!(f.seq, number(entry, "seq"));
            assert_eq!(f.t_ns, number(entry, "t_ns"));
            assert_eq!(f.session_id, id);
            assert_eq!(f.request_id, number(frame, "request_id"));
            assert_eq!(
                f.direction,
                if number(frame, "direction") == 0 {
                    Direction::In
                } else {
                    Direction::Out
                }
            );
            assert_eq!(f.frame_index as u64, number(frame, "frame_index"));
            assert_eq!(f.frame_seq as u64, number(frame, "frame_seq"));
            assert_eq!(f.length as usize, hex(text(frame, "ciphertext")).len());
            assert_eq!(
                &header[33..37],
                &f.length.to_le_bytes(),
                "the header's length"
            );
            assert_eq!(f.tokens(), Some(list(frame, "tokens").len() as u32));
            assert_eq!(f.sha256, bytes32(text(frame, "sha256")));
            decoded += 1;
        }
        for end in list(session, "request_ends") {
            let entry = end.get("entry").unwrap();
            let value = decode_one(&hex(text(entry, "tce")));
            assert_eq!(
                value,
                DigestLogEntry::RequestEnd {
                    seq: number(entry, "seq"),
                    t_ns: number(entry, "t_ns"),
                    session_id: id,
                    request_id: number(end, "request_id"),
                }
            );
            decoded += 1;
        }
    }
    // Five sessions: 15 frames, 3 entries of each session's own, and 7 request ends.
    assert_eq!(decoded, 15 + 5 * 3 + 7);
}

/// The five sessions' opens, frames, ends and closes, in `seq` order, are one period's log from
/// `seq` 0, and every request in it passes the anchor checks with the lengths of its tokens.
#[test]
fn the_anchor_sessions_are_a_log_whose_requests_pass() {
    let doc = read_vectors("anchor/frames.json");
    let mut entries: Vec<(u64, Vec<u8>)> = Vec::new();
    let mut requests: Vec<RequestRef> = Vec::new();
    for session in list(&doc, "sessions") {
        let id = bytes16(text(session, "session_id"));
        for name in ["open_entry", "close_entry"] {
            let entry = session.get(name).unwrap();
            entries.push((number(entry, "seq"), hex(text(entry, "tce"))));
        }
        for frame in list(session, "frames") {
            let entry = frame.get("entry").unwrap();
            entries.push((number(entry, "seq"), hex(text(entry, "tce"))));
            let r = number(frame, "request_id");
            let tokens = list(frame, "tokens").len() as u32;
            let at = match requests
                .iter()
                .position(|q| q.session_id == id && q.request_id == r)
            {
                Some(at) => at,
                None => {
                    requests.push(RequestRef {
                        request_id: r,
                        session_id: id,
                        prompt_len: 0,
                        output_len: 0,
                    });
                    requests.len() - 1
                }
            };
            if number(frame, "direction") == 0 {
                requests[at].prompt_len += tokens;
            } else {
                requests[at].output_len += tokens;
            }
        }
        for end in list(session, "request_ends") {
            let entry = end.get("entry").unwrap();
            entries.push((number(entry, "seq"), hex(text(entry, "tce"))));
        }
    }
    entries.sort();
    assert!(
        entries
            .iter()
            .enumerate()
            .all(|(i, (seq, _))| *seq == i as u64)
    );
    let encodings: Vec<Vec<u8>> = entries.into_iter().map(|(_, e)| e).collect();
    let fields = Fields {
        key_id: sha256(&anchor_support::KEY),
        log_id: anchor_support::LOG_ID,
        start: 0,
        end: 10_000_000,
        first_seq: 0,
        count: encodings.len() as u64,
        root: anchor_support::mth(&encodings),
        prev: [0; 32],
        scheme: 0,
        public_key: anchor_support::KEY.to_vec(),
        signature: anchor_support::SIGNATURE.to_vec(),
    };
    let checked = anchor_support::checked(&fields.seal());
    let log = VerifiedLog::check(&checked, &encodings.concat(), &DecodeLimits::MESSAGE).unwrap();
    let window = Window::bind(&log, &[], None).unwrap();
    assert_eq!(requests.len(), 7);
    let statement = PublicStatement {
        claim_list_digest: [0; 32],
        weights_root: [0; 32],
        weights_epoch: 0,
        manifest_digest: *checked.identity(),
        batch: BatchDescriptor {
            requests: requests.clone(),
        },
        predictor: anchor_support::greedy(),
    };
    assert_eq!(
        check_batch(&statement, &window, &anchor_support::no_pins()),
        Ok(())
    );
    // Every request is an ended candidate of the period, its length its token count.
    let mut expected: Vec<Candidate> = requests
        .iter()
        .map(|r| Candidate {
            session_id: r.session_id,
            request_id: r.request_id,
            ended: true,
            tokens: u64::from(r.prompt_len) + u64::from(r.output_len),
        })
        .collect();
    expected.sort_by_key(|c| (c.session_id, c.request_id));
    assert_eq!(window.candidates().unwrap(), expected);
    // One token short in any request's lengths is refused, naming it.
    for (i, request) in requests.iter().enumerate() {
        let mut batch = requests.clone();
        if batch[i].prompt_len > 0 {
            batch[i].prompt_len -= 1;
        } else {
            batch[i].output_len += 1;
        }
        let statement = PublicStatement {
            batch: BatchDescriptor { requests: batch },
            ..statement.clone()
        };
        assert_eq!(
            check_batch(&statement, &window, &anchor_support::no_pins()),
            Err(tv_core::anchor::Failure::Rejected(
                tv_core::RejectReason::Anchor {
                    request_id: Some(request.request_id),
                    kind: tv_core::verdict::AnchorFailure::Frames,
                }
            ))
        );
    }
}

// ----- spec/vectors/audit/ ----------------------------------------------------------------------

/// The scenario items of one of the vectors' logs.
fn items(log: &[Json]) -> anchor_support::Items {
    use anchor_support::{FrameItem, Item};
    log.iter()
        .map(|e| {
            let s = text(e, "session");
            let key = |s: &str| -> [u8; 32] { sha256(s.as_bytes()) };
            match text(e, "kind") {
                "open" => Item::Open { s: 0, key: key(s) },
                "carry" => Item::Carry { s: 0, key: key(s) },
                "close" => Item::Close { s: 0 },
                "end" => Item::End {
                    s: 0,
                    r: number(e, "request"),
                    t: None,
                },
                "frame" => Item::Frame(FrameItem {
                    s: 0,
                    r: number(e, "request"),
                    direction: if text(e, "direction") == "In" {
                        Direction::In
                    } else {
                        Direction::Out
                    },
                    index: number(e, "index") as u32,
                    length: number(e, "length") as u32,
                    counter: None,
                    t: None,
                }),
                other => panic!("kind {other}"),
            }
        })
        .collect()
}

/// The vectors name sessions by 16 hex bytes; the scenario builder by one byte. Rebuild each
/// entry's encoding with the vector's session instead.
fn with_sessions(built: &mut [anchor_support::Built], logs: &[&[Json]]) {
    for (period, log) in built.iter_mut().zip(logs) {
        for (encoding, e) in period.entries.iter_mut().zip(log.iter()) {
            encoding[18..34].copy_from_slice(&bytes16(text(e, "session")));
        }
        period.tdl = period.entries.concat();
        period.fields.root = anchor_support::mth(&period.entries);
    }
    // Reseal and relink in order.
    let mut prev = [0; 32];
    for period in built.iter_mut() {
        period.fields.prev = prev;
        period.tlm = period.fields.seal();
        period.identity = period.tlm[period.tlm.len() - 32..].try_into().unwrap();
        prev = period.identity;
    }
}

#[test]
fn the_audit_candidates_match_the_oracle() {
    let doc = read_vectors("audit/audit.json");
    let cases = list(&doc, "candidates");
    assert!(!cases.is_empty());
    let (mut with_previous, mut unterminated, mut no_length) = (0, 0, 0);
    for (i, case) in cases.iter().enumerate() {
        let current = list(case, "current");
        let previous = case.get("previous").unwrap().as_array();
        let mut periods = Vec::new();
        let mut logs: Vec<&[Json]> = Vec::new();
        let mut start = anchor_support::GENESIS_NS;
        if let Some(previous) = previous {
            periods.push((start, start + anchor_support::PERIOD_NS, items(previous)));
            logs.push(previous);
            start += anchor_support::PERIOD_NS;
            with_previous += 1;
        }
        periods.push((start, start + anchor_support::PERIOD_NS, items(current)));
        logs.push(current);
        let mut built = anchor_support::chain(&periods);
        with_sessions(&mut built, &logs);
        let verified: Vec<VerifiedLog> = built.iter().map(|b| b.verified()).collect();
        let (period, previous) = match verified.as_slice() {
            [p] => (p, None),
            [q, p] => (p, Some(q)),
            _ => unreachable!(),
        };
        let window = Window::bind(period, &[], previous).unwrap();
        let expected: Vec<Candidate> = list(case, "candidates")
            .iter()
            .map(|c| {
                let key = c.get("key").unwrap();
                Candidate {
                    session_id: bytes16(text(key, "session_id")),
                    request_id: number(key, "request_id"),
                    ended: matches!(c.get("ended"), Some(Json::Bool(true))),
                    tokens: number(c, "tokens"),
                }
            })
            .collect();
        unterminated += expected.iter().filter(|c| !c.ended).count();
        no_length += expected.iter().filter(|c| c.tokens == u64::MAX).count();
        assert_eq!(window.candidates().unwrap(), expected, "case {i}");
    }
    // The cases exercise both periods, unterminated candidates and impossible lengths.
    assert!(
        with_previous > 0 && unterminated > 0 && no_length > 0,
        "{with_previous} {unterminated} {no_length}"
    );
}

#[test]
fn every_request_of_the_vectors_period_is_named_by_the_statement_it_anchors() {
    // The statement must name the window's period: another manifest's identity is the verifier's
    // own inputs disagreeing (RFC-0019 step 4).
    let (doc, manifests) = manifest_vectors();
    let logs: Vec<VerifiedLog> = manifests.iter().map(|v| verified(&doc, v)).collect();
    let window = Window::bind(&logs[1], &[], Some(&logs[0])).unwrap();
    let statement = PublicStatement {
        claim_list_digest: [0; 32],
        weights_root: [0; 32],
        weights_epoch: 0,
        manifest_digest: manifests[0].identity,
        batch: BatchDescriptor {
            requests: vec![req(0x30, 7, 3, 12)],
        },
        predictor: anchor_support::greedy(),
    };
    let pins = anchor_support::no_pins();
    assert_eq!(
        check_batch(&statement, &window, &pins),
        Err(tv_core::anchor::Failure::Inconclusive(
            tv_core::VerifierError {
                code: tv_core::ErrorCode::Usage
            }
        ))
    );
}
