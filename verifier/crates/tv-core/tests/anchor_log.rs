// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The anchor's inputs, each check broken alone: an entry's decoding (RFC-0003 with RFC-0015,
//! RFC-0059, RFC-0062 and RFC-0063's variants), a manifest's own checks (RFC-0060 'Signatures',
//! 'The chain'), a period's log against its manifest (RFC-0062 'The check', 04 §3's order and
//! offsets), and the previous period's binding (RFC-0063 'The previous period's log').

mod anchor_support;

use anchor_support::{
    Built, Fields, GENESIS_NS, KEY, PERIOD_NS, chain, encode_close, encode_end, encode_frame,
    encode_key_entry, end, inbound, keys, open, outbound, seal, sid,
};
use tv_codec::{DecodeError, DecodeLimits, ErrorCode, decode};
use tv_core::anchor::{
    CheckedManifest, DigestLogEntry, Direction, Failure, LogManifest, ManifestError, RecorderKeys,
    VerifiedLog, Window, log_failure,
};
use tv_core::verdict::{AnchorFailure, ErrorCode as Code, RejectReason, Verdict, VerifierError};
use tv_hash::sha256::sha256;

fn decode_entry(bytes: &[u8]) -> Result<DigestLogEntry, DecodeError> {
    decode(bytes, &DecodeLimits::MESSAGE, DigestLogEntry::decode)
}

/// One encoding of each variant, with the offsets at which its fields start.
fn variants() -> Vec<(&'static str, Vec<u8>, Vec<u64>)> {
    let common = vec![0, 2, 10, 18];
    let with = |more: &[u64]| [common.clone(), more.to_vec()].concat();
    vec![
        (
            "SessionOpen",
            encode_key_entry(false, 1, 2, sid(3), [4; 32]),
            with(&[34]),
        ),
        (
            "Frame",
            encode_frame(1, 2, sid(3), 4, Direction::Out, 5, 26, 7, [8; 32]),
            with(&[34, 42, 44, 48, 52, 56]),
        ),
        ("SessionClose", encode_close(1, 2, sid(3)), common.clone()),
        (
            "SessionCarry",
            encode_key_entry(true, 1, 2, sid(3), [4; 32]),
            with(&[34]),
        ),
        ("RequestEnd", encode_end(1, 2, sid(3), 4), with(&[34])),
    ]
}

#[test]
fn each_variant_has_its_size_and_fields() {
    let sizes: Vec<usize> = variants().iter().map(|(_, bytes, _)| bytes.len()).collect();
    assert_eq!(sizes, [66, 88, 34, 66, 42]);
    let decoded: Vec<DigestLogEntry> = variants()
        .iter()
        .map(|(_, bytes, _)| decode_entry(bytes).unwrap())
        .collect();
    assert_eq!(
        decoded[0],
        DigestLogEntry::SessionOpen {
            seq: 1,
            t_ns: 2,
            session_id: sid(3),
            key_commitment: [4; 32]
        }
    );
    let DigestLogEntry::Frame(frame) = decoded[1] else {
        panic!()
    };
    assert_eq!(
        (
            frame.seq,
            frame.t_ns,
            frame.session_id,
            frame.request_id,
            frame.direction
        ),
        (1, 2, sid(3), 4, Direction::Out)
    );
    assert_eq!(
        (
            frame.frame_index,
            frame.length,
            frame.frame_seq,
            frame.sha256
        ),
        (5, 26, 7, [8; 32])
    );
    assert_eq!(frame.tokens(), Some(1));
    assert_eq!(
        decoded[2],
        DigestLogEntry::SessionClose {
            seq: 1,
            t_ns: 2,
            session_id: sid(3)
        }
    );
    assert_eq!(
        decoded[3],
        DigestLogEntry::SessionCarry {
            seq: 1,
            t_ns: 2,
            session_id: sid(3),
            key_commitment: [4; 32]
        }
    );
    assert_eq!(
        decoded[4],
        DigestLogEntry::RequestEnd {
            seq: 1,
            t_ns: 2,
            session_id: sid(3),
            request_id: 4
        }
    );
    for (i, entry) in decoded.iter().enumerate() {
        assert_eq!(entry.tag(), i as u16);
        assert_eq!(
            (entry.seq(), entry.t_ns(), entry.session_id()),
            (1, 2, &sid(3))
        );
    }
}

/// RFC-0059 and RFC-0003: every proper prefix of every variant is `ERR-CODEC-TRUNCATED` at the
/// start of the field the cut falls in; the session, an array, is checked whole.
#[test]
fn every_prefix_is_truncated_at_its_field() {
    for (name, bytes, starts) in variants() {
        for cut in 0..bytes.len() {
            let error = decode_entry(&bytes[..cut]).unwrap_err();
            let field = *starts.iter().filter(|&&s| s <= cut as u64).max().unwrap();
            assert_eq!(
                (error.code(), error.offset()),
                (ErrorCode::Truncated, field),
                "{name} cut at {cut}"
            );
        }
    }
}

#[test]
fn unknown_tags_and_directions_are_noncanonical_at_the_tag() {
    // RFC-0063 took tag 4: the first unknown tag is 5 (RFC-0062 'Testing Strategy').
    for tag in [5u16, 6, 0x0100, u16::MAX] {
        let mut bytes = encode_close(1, 2, sid(3));
        bytes[..2].copy_from_slice(&tag.to_le_bytes());
        let error = decode_entry(&bytes).unwrap_err();
        assert_eq!(
            (error.code(), error.offset()),
            (ErrorCode::Noncanonical, 0),
            "tag {tag}"
        );
    }
    // RFC-0059: a direction of 2, or a tag whose high byte is set.
    for direction in [2u16, 0x0100, 0x0101] {
        let mut bytes = encode_frame(1, 2, sid(3), 4, Direction::In, 5, 26, 7, [8; 32]);
        bytes[42..44].copy_from_slice(&direction.to_le_bytes());
        let error = decode_entry(&bytes).unwrap_err();
        assert_eq!(
            (error.code(), error.offset()),
            (ErrorCode::Noncanonical, 42),
            "{direction}"
        );
    }
    // A frame's length is not the decoder's to judge: 21 and 0 decode, with no token count.
    for length in [0u32, 21, 23, u32::MAX] {
        let bytes = encode_frame(1, 2, sid(3), 4, Direction::In, 5, length, 7, [8; 32]);
        let Ok(DigestLogEntry::Frame(frame)) = decode_entry(&bytes) else {
            panic!()
        };
        assert_eq!(frame.tokens(), None, "length {length}");
    }
    let bytes = encode_frame(1, 2, sid(3), 4, Direction::In, 5, 22, 7, [8; 32]);
    let Ok(DigestLogEntry::Frame(frame)) = decode_entry(&bytes) else {
        panic!()
    };
    assert_eq!(frame.tokens(), Some(0), "an empty frame");
}

#[test]
fn an_entry_takes_two_levels_of_depth() {
    let bytes = encode_close(1, 2, sid(3));
    let limited = |max_depth| DecodeLimits {
        max_depth,
        ..DecodeLimits::MESSAGE
    };
    let error = decode(&bytes, &limited(0), DigestLogEntry::decode).unwrap_err();
    assert_eq!((error.code(), error.offset()), (ErrorCode::Limit, 0));
    let error = decode(&bytes, &limited(1), DigestLogEntry::decode).unwrap_err();
    assert_eq!(
        (error.code(), error.offset()),
        (ErrorCode::Limit, 18),
        "the session's array"
    );
    decode(&bytes, &limited(2), DigestLogEntry::decode).unwrap();
}

// ----- The period's log -------------------------------------------------------------------------

/// A period of seven entries: an open, frames, an end and a close.
fn period() -> Built {
    chain(&[(
        GENESIS_NS,
        GENESIS_NS + PERIOD_NS,
        vec![
            open(1),
            inbound(1, 7, 0, 2).item(),
            outbound(1, 7, 0, 1).item(),
            outbound(1, 7, 1, 0).item(),
            end(1, 7),
            open(2),
            anchor_support::close(1),
        ],
    )])
    .remove(0)
}

fn check(manifest: &CheckedManifest, log: &[u8]) -> Result<VerifiedLog, (ErrorCode, u64)> {
    VerifiedLog::check(manifest, log, &DecodeLimits::MESSAGE).map_err(|e| (e.code(), e.offset()))
}

/// A manifest over `entries`, as `fields` but for its count and root, which `mutate` may change.
fn manifest_over(
    base: &Fields,
    entries: &[Vec<u8>],
    mutate: impl FnOnce(&mut Fields),
) -> CheckedManifest {
    let mut fields = base.clone();
    fields.count = entries.len() as u64;
    fields.root = anchor_support::mth(entries);
    mutate(&mut fields);
    anchor_support::checked(&fields.seal())
}

#[test]
fn an_honest_log_verifies_and_keeps_its_entries() {
    let built = period();
    let log = built.verified();
    assert_eq!(log.entries().len(), 7);
    assert_eq!(log.manifest_digest(), &built.identity);
    for (entry, encoding) in log.entries().iter().zip(&built.entries) {
        assert_eq!(entry, &decode_entry(encoding).unwrap());
    }
    // An empty period: no bytes, `SHA-256("")`.
    let empty = chain(&[(GENESIS_NS, GENESIS_NS + PERIOD_NS, vec![])]).remove(0);
    assert_eq!(empty.fields.root, sha256(&[]));
    assert_eq!(empty.verified().entries(), &[]);
}

/// RFC-0062 'Testing Strategy': one entry missing, one added, two swapped and a wrong count are
/// each refused at load, `Inconclusive(ERR-CODEC-DIGEST)`, in 04 §3's order and at its offsets.
#[test]
fn a_log_that_is_not_its_manifests_is_refused() {
    let built = period();
    let manifest = built.checked();
    let end = built.tdl.len() as u64;
    let starts: Vec<u64> = built
        .entries
        .iter()
        .scan(0u64, |at, e| {
            let start = *at;
            *at += e.len() as u64;
            Some(start)
        })
        .collect();

    // One entry missing: the count, at the log's end, though seq and root are wrong too.
    for missing in 0..built.entries.len() {
        let mut entries = built.entries.clone();
        entries.remove(missing);
        let log = entries.concat();
        assert_eq!(
            check(&manifest, &log).unwrap_err(),
            (ErrorCode::Digest, log.len() as u64)
        );
    }
    // One added, a copy of an entry or a new one.
    let mut log = built.tdl.clone();
    log.extend_from_slice(&encode_close(7, 0, sid(9)));
    assert_eq!(
        check(&manifest, &log).unwrap_err(),
        (ErrorCode::Digest, log.len() as u64)
    );
    // Two swapped: the first wrong seq, at its entry's start.
    for (i, &start) in starts.iter().enumerate().take(built.entries.len() - 1) {
        let mut entries = built.entries.clone();
        entries.swap(i, i + 1);
        assert_eq!(
            check(&manifest, &entries.concat()).unwrap_err(),
            (ErrorCode::Digest, start)
        );
    }
    // A manifest that counts one entry more or less than the log.
    for count in [6, 8] {
        let other = manifest_over(&built.fields, &built.entries, |f| f.count = count);
        assert_eq!(
            check(&other, &built.tdl).unwrap_err(),
            (ErrorCode::Digest, end)
        );
    }
    // A first_seq one off: every seq is wrong, from the first entry.
    let other = manifest_over(&built.fields, &built.entries, |f| f.first_seq = 1);
    assert_eq!(
        check(&other, &built.tdl).unwrap_err(),
        (ErrorCode::Digest, 0)
    );
    // An entry changed in a field no other check reads: the root, at the log's end.
    let mut entries = built.entries.clone();
    entries[3][10] ^= 1; // its t_ns
    assert_eq!(
        check(&manifest, &entries.concat()).unwrap_err(),
        (ErrorCode::Digest, end)
    );
    // Another root signed.
    let other = manifest_over(&built.fields, &built.entries, |f| f.root[0] ^= 1);
    assert_eq!(
        check(&other, &built.tdl).unwrap_err(),
        (ErrorCode::Digest, end)
    );
    // Each is the verifier's own input: Inconclusive, never a rejection (RFC-0062 'Its fault').
    assert!(matches!(
        Verdict::from(log_failure(DecodeError::new(ErrorCode::Digest, end))),
        Verdict::Inconclusive(VerifierError {
            code: Code::CodecDigest
        })
    ));
}

#[test]
fn a_log_that_fails_to_decode_is_inconclusive_with_its_code() {
    let built = period();
    let manifest = built.checked();
    // Cut inside the last entry, a close of 34 bytes: truncated at its session.
    let cut = built.tdl.len() - 1;
    let error =
        VerifiedLog::check(&manifest, &built.tdl[..cut], &DecodeLimits::MESSAGE).unwrap_err();
    assert_eq!(
        (error.code(), error.offset()),
        (ErrorCode::Truncated, cut as u64 - 15)
    );
    assert_eq!(
        log_failure(error),
        Failure::Inconclusive(VerifierError {
            code: Code::CodecTruncated
        })
    );
    // A decoding error comes before the count: a bad tag in an extra entry.
    let mut log = built.tdl.clone();
    log.extend_from_slice(&[9, 0]);
    let error = VerifiedLog::check(&manifest, &log, &DecodeLimits::MESSAGE).unwrap_err();
    assert_eq!(
        (error.code(), error.offset()),
        (ErrorCode::Noncanonical, built.tdl.len() as u64)
    );
    // The whole stream is checked against the limit first.
    let limits = DecodeLimits {
        max_total_bytes: built.tdl.len() as u64 - 1,
        ..DecodeLimits::MESSAGE
    };
    let error = VerifiedLog::check(&manifest, &built.tdl, &limits).unwrap_err();
    assert_eq!((error.code(), error.offset()), (ErrorCode::Limit, 0));
}

// ----- A manifest's own checks ------------------------------------------------------------------

fn base_fields() -> Fields {
    period().fields
}

/// A change to a manifest's fields.
type FieldEdit = dyn Fn(&mut Fields);

/// A change to a manifest's fields, owned.
type Mutation = Box<FieldEdit>;

fn own_check(
    fields: &Fields,
    keys: &RecorderKeys,
    verify: impl FnOnce(&[u8; 32], &[u8], &[u8; 64]) -> bool,
) -> Result<(), ManifestError> {
    LogManifest::open(&fields.seal(), &DecodeLimits::MESSAGE)
        .unwrap()
        .check(keys, verify)
        .map(|_| ())
}

/// A verifier the check must not reach.
fn unreachable_verifier(_: &[u8; 32], _: &[u8], _: &[u8; 64]) -> bool {
    panic!("the strict verifier runs after every other check of the signature")
}

/// RFC-0060 'Testing Strategy': a wrong key identifier, an unpinned key, scheme 1, an unknown
/// scheme, wrong key and signature lengths, a refused signature, an overflowing range and a wrong
/// empty root, each refused alone. All are `Reject(Anchor { None, ManifestSignature })`.
#[test]
fn each_own_check_refuses_alone() {
    let base = base_fields();
    assert_eq!(own_check(&base, &keys(), |_, _, _| true), Ok(()));
    let cases: Vec<(&str, Mutation, ManifestError)> = vec![
        (
            "scheme 1",
            Box::new(|f: &mut Fields| f.scheme = 1),
            ManifestError::Scheme,
        ),
        (
            "scheme 2",
            Box::new(|f: &mut Fields| f.scheme = 2),
            ManifestError::Scheme,
        ),
        (
            "scheme 0xFFFF",
            Box::new(|f: &mut Fields| f.scheme = u16::MAX),
            ManifestError::Scheme,
        ),
        (
            "a 31-byte key",
            Box::new(|f: &mut Fields| f.public_key.pop().map(drop).unwrap()),
            ManifestError::KeyLength,
        ),
        (
            "a 33-byte key",
            Box::new(|f: &mut Fields| f.public_key.push(0)),
            ManifestError::KeyLength,
        ),
        (
            "no key",
            Box::new(|f: &mut Fields| f.public_key.clear()),
            ManifestError::KeyLength,
        ),
        (
            "a 63-byte signature",
            Box::new(|f: &mut Fields| f.signature.pop().map(drop).unwrap()),
            ManifestError::SignatureLength,
        ),
        (
            "a 65-byte signature",
            Box::new(|f: &mut Fields| f.signature.push(0)),
            ManifestError::SignatureLength,
        ),
        (
            "another key id",
            Box::new(|f: &mut Fields| f.key_id[31] ^= 1),
            ManifestError::KeyId,
        ),
        (
            "an unpinned key with its own id",
            Box::new(|f: &mut Fields| {
                f.public_key = vec![0x5b; 32];
                f.key_id = sha256(&f.public_key);
            }),
            ManifestError::Unpinned,
        ),
    ];
    for (name, mutate, expected) in cases {
        let mut fields = base.clone();
        mutate(&mut fields);
        assert_eq!(
            own_check(&fields, &keys(), unreachable_verifier),
            Err(expected),
            "{name}"
        );
        assert_eq!(
            expected.reason(),
            RejectReason::Anchor {
                request_id: None,
                kind: AnchorFailure::ManifestSignature
            }
        );
    }
    // No key pinned at all.
    assert_eq!(
        own_check(&base, &RecorderKeys::new(vec![]), unreachable_verifier),
        Err(ManifestError::Unpinned)
    );
    // A pinned key that the strict verifier refuses, for a bad signature or a key that is no
    // point: the seam's answer is the check's.
    assert_eq!(
        own_check(&base, &keys(), |_, _, _| false),
        Err(ManifestError::Signature)
    );
    // A rotation pins both keys: either signs.
    let both = RecorderKeys::new(vec![[0x5b; 32], KEY]);
    assert_eq!(own_check(&base, &both, |_, _, _| true), Ok(()));

    // After the signature: the period, the range, the empty root, in that order.
    let after: Vec<(&str, Mutation, ManifestError)> = vec![
        (
            "an end before the start",
            Box::new(|f: &mut Fields| f.end = f.start - 1),
            ManifestError::Period,
        ),
        (
            "an overflowing range",
            Box::new(|f: &mut Fields| f.first_seq = u64::MAX - f.count + 1),
            ManifestError::Range,
        ),
        (
            "a wrong empty root",
            Box::new(|f: &mut Fields| {
                f.count = 0;
                f.root = [0; 32];
            }),
            ManifestError::EmptyRoot,
        ),
        (
            "a period and a range both wrong",
            Box::new(|f: &mut Fields| {
                f.end = f.start - 1;
                f.first_seq = u64::MAX;
            }),
            ManifestError::Period,
        ),
    ];
    for (name, mutate, expected) in after {
        let mut fields = base.clone();
        mutate(&mut fields);
        assert_eq!(
            own_check(&fields, &keys(), |_, _, _| true),
            Err(expected),
            "{name}"
        );
        // A signature failure comes first.
        assert_eq!(
            own_check(&fields, &keys(), |_, _, _| false),
            Err(ManifestError::Signature),
            "{name}"
        );
    }
    // The edges pass: a zero-length period, a range ending at 2^64 − 1, an empty period.
    for mutate in [
        Box::new(|f: &mut Fields| f.end = f.start) as Mutation,
        Box::new(|f: &mut Fields| f.first_seq = u64::MAX - f.count),
        Box::new(|f: &mut Fields| {
            f.count = 0;
            f.root = sha256(&[]);
        }),
    ] {
        let mut fields = base.clone();
        mutate(&mut fields);
        assert_eq!(own_check(&fields, &keys(), |_, _, _| true), Ok(()));
    }
}

/// The seam receives the pinned key, the 163-byte message and the signature, and only once.
#[test]
fn the_strict_verifier_is_handed_the_signed_message() {
    let base = base_fields();
    let mut calls = Vec::new();
    own_check(&base, &keys(), |key, message, signature| {
        calls.push((*key, message.to_vec(), *signature));
        true
    })
    .unwrap();
    let message = [&b"tessera-manifest/v1"[..], &base.encode()[..144]].concat();
    assert_eq!(calls, vec![(KEY, message, anchor_support::SIGNATURE)]);
}

#[test]
fn a_manifest_that_fails_to_decode_is_refused_with_its_code() {
    let base = base_fields();
    let tlm = base.seal();
    // Another schema id: an entry's 0x0401 at offset 6.
    let mut other = tlm[..tlm.len() - 32].to_vec();
    other[6..8].copy_from_slice(&0x0401u16.to_le_bytes());
    let digest = tv_testkit::blake3::derive_key("tessera/v1/envelope", &other);
    other.extend_from_slice(&digest);
    let error = LogManifest::open(&other, &DecodeLimits::MESSAGE).unwrap_err();
    assert_eq!((error.code(), error.offset()), (ErrorCode::Magic, 6));
    // A payload with a byte left over.
    let mut payload = base.encode();
    payload.push(0);
    let error = LogManifest::open(&seal(&payload), &DecodeLimits::MESSAGE).unwrap_err();
    assert_eq!(
        (error.code(), error.offset()),
        (ErrorCode::Noncanonical, 20 + payload.len() as u64 - 1)
    );
    // Every proper prefix of the payload, resealed, is truncated.
    let payload = base.encode();
    for cut in 0..payload.len() {
        let error = LogManifest::open(&seal(&payload[..cut]), &DecodeLimits::MESSAGE).unwrap_err();
        assert_eq!(error.code(), ErrorCode::Truncated, "cut at {cut}");
    }
}

// ----- The previous period ----------------------------------------------------------------------

/// A chain of four periods: one with entries, two of zero length, one with entries.
fn four() -> Vec<Built> {
    let p = PERIOD_NS;
    let g = GENESIS_NS;
    chain(&[
        (g, g + p, vec![open(1), inbound(1, 7, 0, 1).item()]),
        (g + p, g + p, vec![]),
        (g + p, g + p, vec![]),
        (g + p, g + 2 * p, vec![carry(1), end(1, 7)]),
    ])
}

fn carry(s: u8) -> anchor_support::Item {
    anchor_support::carry(s)
}

fn usage() -> Failure {
    Failure::Inconclusive(VerifierError { code: Code::Usage })
}

/// RFC-0063 'The previous period's log' and 'Testing Strategy': `P − 1` is reached through the
/// zero-length periods between; one that is not `prev_manifest`'s is `Inconclusive(ERR-USAGE)`;
/// a log's first period needs none.
#[test]
fn the_previous_log_is_bound_through_the_chain() {
    let built = four();
    let logs: Vec<VerifiedLog> = built.iter().map(Built::verified).collect();
    let zero: Vec<CheckedManifest> = built[1..3].iter().map(Built::checked).collect();
    let bind = |period: usize, between: &[CheckedManifest], previous: Option<usize>| {
        Window::bind(&logs[period], between, previous.map(|i| &logs[i])).map(|_| ())
    };
    // Through both zero-length periods, in chain order.
    assert_eq!(bind(3, &zero, Some(0)), Ok(()));
    // A log's first period needs none, and takes none.
    assert_eq!(bind(0, &[], None), Ok(()));
    assert_eq!(bind(0, &[], Some(0)), Err(usage()));
    // Missing: no previous log though P − 1 exists.
    assert_eq!(bind(3, &zero, None), Err(usage()));
    // The zero-length periods missing, out of order, or one of them only.
    assert_eq!(bind(3, &[], Some(0)), Err(usage()));
    assert_eq!(
        bind(3, &[zero[1].clone(), zero[0].clone()], Some(0)),
        Err(usage())
    );
    assert_eq!(bind(3, &zero[1..], Some(0)), Err(usage()));
    // A zero-length period offered as P − 1.
    assert_eq!(bind(3, &[], Some(2)), Err(usage()));
    assert_eq!(bind(3, &zero[1..], Some(1)), Err(usage()));
    // A period of non-zero length offered as a zero-length one between.
    let p0 = built[0].checked();
    assert_eq!(
        bind(3, &[p0, zero[0].clone(), zero[1].clone()], Some(0)),
        Err(usage())
    );
}

#[test]
fn each_link_of_the_chain_is_checked() {
    let p = PERIOD_NS;
    let g = GENESIS_NS;
    let base = chain(&[(g, g + p, vec![open(1)]), (g + p, g + 2 * p, vec![open(2)])]);
    let previous = base[0].verified();
    let relinked = |mutate: &dyn Fn(&mut Fields)| -> Result<(), Failure> {
        let mut fields = base[1].fields.clone();
        mutate(&mut fields);
        let manifest = anchor_support::checked(&fields.seal());
        let log = VerifiedLog::check(&manifest, &base[1].tdl, &DecodeLimits::MESSAGE).unwrap();
        Window::bind(&log, &[], Some(&previous)).map(|_| ())
    };
    assert_eq!(relinked(&|_| {}), Ok(()));
    let breaks: [(&str, &FieldEdit); 6] = [
        ("another log", &|f| f.log_id[0] ^= 1),
        ("a wrong link", &|f| f.prev[0] ^= 1),
        ("a gap of entries", &|f| f.first_seq += 1),
        ("an overlap of entries", &|f| f.first_seq -= 1),
        ("a gap of time", &|f| {
            f.start += 1;
            f.end += 1;
        }),
        ("an overlap of time", &|f| f.start -= 1),
    ];
    for (name, mutate) in breaks {
        // `first_seq` moves the log's seqs too: rebuild the entries to match.
        let mut fields = base[1].fields.clone();
        mutate(&mut fields);
        let entries: Vec<Vec<u8>> = base[1]
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let mut e = e.clone();
                e[2..10].copy_from_slice(&(fields.first_seq + i as u64).to_le_bytes());
                e
            })
            .collect();
        fields.root = anchor_support::mth(&entries);
        let manifest = anchor_support::checked(&fields.seal());
        let log = VerifiedLog::check(&manifest, &entries.concat(), &DecodeLimits::MESSAGE).unwrap();
        assert_eq!(
            Window::bind(&log, &[], Some(&previous)).map(|_| ()),
            Err(usage()),
            "{name}"
        );
    }
    // Without a previous log, P must be a log's first: prev_manifest zero and first_seq 0.
    let first = chain(&[(g, g + p, vec![open(1)])]).remove(0);
    let mut fields = first.fields.clone();
    fields.first_seq = 0;
    fields.prev = [1; 32];
    let manifest = anchor_support::checked(&fields.seal());
    let log = VerifiedLog::check(&manifest, &first.tdl, &DecodeLimits::MESSAGE).unwrap();
    assert_eq!(
        Window::bind(&log, &[], None).map(|_| ()),
        Err(usage()),
        "a link to a period not held"
    );
}

/// A zero-length period between with entries is refused: such a period holds none (RFC-0060,
/// RFC-0063 'The periods'), so its entries would lie in no audited window.
#[test]
fn a_zero_length_period_with_entries_is_refused_between() {
    let p = PERIOD_NS;
    let g = GENESIS_NS;
    let built = chain(&[
        (g, g + p, vec![open(1)]),
        (g + p, g + p, vec![open(2)]),
        (g + p, g + 2 * p, vec![open(3)]),
    ]);
    let logs: Vec<VerifiedLog> = built.iter().map(Built::verified).collect();
    let between = [built[1].checked()];
    assert_eq!(
        Window::bind(&logs[2], &between, Some(&logs[0])).map(|_| ()),
        Err(usage())
    );
    // Its log itself verifies: the recorder's rule, not the manifest's own checks.
    assert_eq!(logs[1].entries().len(), 1);
}

/// RFC-0060 'Testing Strategy', the chain: a first manifest that is not a genesis, and a
/// reordered pair, each refused when bound.
#[test]
fn a_false_genesis_and_a_reordered_pair_are_refused() {
    let p = PERIOD_NS;
    let g = GENESIS_NS;
    let built = chain(&[(g, g + p, vec![open(1)]), (g + p, g + 2 * p, vec![open(2)])]);
    let logs: Vec<VerifiedLog> = built.iter().map(Built::verified).collect();
    // The pair reordered: P − 1 offered as the period, P as its previous.
    assert_eq!(
        Window::bind(&logs[0], &[], Some(&logs[1])).map(|_| ()),
        Err(usage())
    );
    // A first period whose prev_manifest is zero but whose first_seq is not 0: entries before
    // it lie in a period not held.
    let mut fields = built[1].fields.clone();
    fields.prev = [0; 32];
    let manifest = anchor_support::checked(&fields.seal());
    let log = VerifiedLog::check(&manifest, &built[1].tdl, &DecodeLimits::MESSAGE).unwrap();
    assert_eq!(Window::bind(&log, &[], None).map(|_| ()), Err(usage()));
}

/// RFC-0063 'The candidates', with a zero-length previous period: `P − 1` is the last period of
/// non-zero length, reached through it. A period of zero length has no candidates (RFC-0016).
#[test]
fn candidates_reach_through_zero_length_periods() {
    let p = PERIOD_NS;
    let g = GENESIS_NS;
    let built = chain(&[
        (
            g,
            g + p,
            vec![
                open(1),
                inbound(1, 7, 0, 2).item(),
                inbound(1, 8, 0, 5).item(),
            ],
        ),
        (g + p, g + p, vec![]),
        (
            g + p,
            g + 2 * p,
            vec![carry(1), outbound(1, 7, 0, 1).item(), end(1, 7)],
        ),
    ]);
    let logs: Vec<VerifiedLog> = built.iter().map(Built::verified).collect();
    let window = Window::bind(&logs[2], &[built[1].checked()], Some(&logs[0])).unwrap();
    let candidates = window.candidates().unwrap();
    assert_eq!(candidates.len(), 2);
    assert_eq!(
        (
            candidates[0].request_id,
            candidates[0].ended,
            candidates[0].tokens
        ),
        (7, true, 3)
    );
    assert_eq!(
        (
            candidates[1].request_id,
            candidates[1].ended,
            candidates[1].tokens
        ),
        (8, false, 5)
    );
    // The zero-length period, bound to its own predecessor, draws nothing.
    let zero = Window::bind(&logs[1], &[], Some(&logs[0])).unwrap();
    assert_eq!(zero.candidates().unwrap(), vec![]);
}
