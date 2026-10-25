// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The `audit` target's oracle: `tv-core`'s readers of the audit sample (schema `0x0403`), the
//! verifier's ledger (schema `0x0701`) and the verifier's pending toss (schema `0x0702`), payload
//! and envelope, on arbitrary bytes (RFC-0016 'Sample derivation' step 5, RFC-0019 'Ledger',
//! RFC-0067 'The pending toss').
//!
//! The first byte selects the reader, the second the limits; the rest is the input. A rejection
//! must be a fault of the input, never `ERR-RES-MEMORY` or `ERR-INTERNAL`, at an offset inside
//! it. An accepted input must be its value's one encoding: the payload re-encodes, and the
//! envelope reseals, to the input byte for byte (INV-TCE-2), and each proper prefix of a payload
//! is cut short. The value must keep the rules, read here from its accessors: the policy's
//! buckets strictly increasing and its batch bounds; `selected` strictly increasing; `derived`
//! strictly increasing by manifest, and each ledger record of a derived sample, of a non-empty
//! set whose last byte is not zero, with one `Û` per index, by strictly increasing sample
//! (RFC-0067). A
//! report on an accepted sample from an empty ledger, which did not derive it, is refused; from
//! a ledger that derived it, it has every request unproven, or refuses a coin toss that does not
//! open. A toss's reader returns the verifier's error, whose code alone it carries: a rejection
//! must be `ERR-CODEC-*` or `ERR-PROTO-COINTOSS`, the latter only for bytes whose commitment the
//! test kit's BLAKE3 finds the share does not open; an accepted toss re-encodes or reseals to the
//! input, its share opens its commitment, its fields are the input's, and its policy keeps its
//! rules.

use tv_codec::{DecodeLimits, ErrorCode};
use tv_core::anchor::Failure;
use tv_core::audit::{AuditPolicy, AuditReport, AuditSample, VerifierLedger, VerifierToss};
use tv_core::verdict::{ErrorCode as Code, VerifierError};
use tv_testkit::blake3::derive_key;

/// The limits, by the second byte: tight enough to reach each limit with short inputs, and
/// RFC-0003's defaults for messages.
const LIMITS: [DecodeLimits; 2] = [
    DecodeLimits {
        max_total_bytes: 1024,
        max_vec_len: 16,
        max_depth: 4,
    },
    DecodeLimits::MESSAGE,
];

/// Runs one input against the oracle, panicking on any violation.
pub fn check(data: &[u8]) {
    let [selector, config, input @ ..] = data else {
        return;
    };
    let limits = LIMITS[usize::from(config & 1)];
    match selector % 6 {
        0 => match AuditSample::decode_payload(input, &limits) {
            Ok(sample) => {
                assert_eq!(sample.encode_payload().unwrap(), input, "a second encoding");
                sample_rules(&sample);
                prefixes(input, |p| AuditSample::decode_payload(p, &limits).err());
            }
            Err(e) => refused(e, input),
        },
        1 => match VerifierLedger::decode_payload(input, &limits) {
            Ok(ledger) => {
                assert_eq!(ledger.encode_payload().unwrap(), input, "a second encoding");
                ledger_rules(&ledger);
                prefixes(input, |p| VerifierLedger::decode_payload(p, &limits).err());
            }
            Err(e) => refused(e, input),
        },
        2 => match AuditSample::open(input, &limits) {
            Ok(sample) => {
                assert_eq!(sample.seal().unwrap(), input, "a second envelope");
                assert_eq!(sample.identity(), &input[input.len() - 32..]);
                sample_rules(&sample);
            }
            Err(e) => refused(e, input),
        },
        3 => match VerifierLedger::open(input, &limits) {
            Ok(ledger) => {
                assert_eq!(ledger.seal().unwrap(), input, "a second envelope");
                ledger_rules(&ledger);
            }
            Err(e) => refused(e, input),
        },
        4 => match VerifierToss::decode_payload(input, &limits) {
            Ok(toss) => {
                assert_eq!(
                    toss.encode_payload().unwrap().as_slice(),
                    input,
                    "a second encoding"
                );
                toss_rules(&toss, input);
                for cut in 0..input.len() {
                    let e = VerifierToss::decode_payload(&input[..cut], &limits)
                        .expect_err("a proper prefix of an encoding decodes");
                    assert_eq!(e.code, Code::CodecTruncated, "prefix of {cut}");
                }
            }
            Err(e) => toss_refused(e, input),
        },
        _ => match VerifierToss::open(input, &limits) {
            Ok(toss) => {
                assert_eq!(toss.seal().unwrap().as_slice(), input, "a second envelope");
                toss_rules(&toss, &input[20..input.len() - 32]);
            }
            Err(e) => toss_refused(
                e,
                input.get(20..input.len().saturating_sub(32)).unwrap_or(&[]),
            ),
        },
    }
}

/// A toss's rejection: a fault of the input, `ERR-PROTO-COINTOSS` only where the payload's last
/// 32 bytes do not open the 32 before them.
fn toss_refused(e: VerifierError, payload: &[u8]) {
    match e.code {
        Code::CodecMagic
        | Code::CodecVersion
        | Code::CodecNoncanonical
        | Code::CodecTruncated
        | Code::CodecDigest
        | Code::CodecLimit => {}
        Code::ProtocolCoinToss => {
            let n = payload.len();
            assert!(n >= 64, "a commitment refused in {n} bytes");
            assert_ne!(
                derive_key("tessera/v1/coin-commit", &payload[n - 32..]),
                payload[n - 64..n - 32],
                "a share that opens its commitment refused"
            );
        }
        other => panic!("{other:?} on a fuzz input"),
    }
}

/// An accepted toss: its fields are the payload's, its share opens its commitment under the
/// test kit's BLAKE3, and its policy keeps its rules.
fn toss_rules(toss: &VerifierToss, payload: &[u8]) {
    let n = payload.len();
    assert_eq!(toss.manifest_digest(), &payload[..32]);
    assert_eq!(toss.commitment(), &payload[n - 64..n - 32]);
    assert_eq!(
        derive_key("tessera/v1/coin-commit", &payload[n - 32..]),
        payload[n - 64..n - 32],
        "a share that does not open its commitment accepted"
    );
    policy_rules(toss.policy());
}

fn policy_rules(policy: &AuditPolicy) {
    for pair in policy.buckets().windows(2) {
        assert!(
            pair[0].max_tokens < pair[1].max_tokens,
            "buckets not increasing"
        );
    }
    let max = u64::from(policy.max_batch_tokens());
    assert!(max >= 1 && max >= 2 * u64::from(policy.min_batch_tokens()));
}

fn refused(e: tv_codec::DecodeError, input: &[u8]) {
    assert!(
        e.code().is_input_fault() && e.offset() <= input.len() as u64,
        "{e} on a fuzz input"
    );
}

fn prefixes(input: &[u8], decode: impl Fn(&[u8]) -> Option<tv_codec::DecodeError>) {
    for cut in 0..input.len() {
        let e = decode(&input[..cut]).expect("a proper prefix of an encoding decodes");
        assert_eq!(e.code(), ErrorCode::Truncated, "prefix of {cut}: {e}");
    }
}

fn sample_rules(sample: &AuditSample) {
    policy_rules(sample.policy());
    for pair in sample.selected().windows(2) {
        assert!(pair[0] < pair[1], "selected not increasing");
    }
    assert!(matches!(
        AuditReport::from_ledger(sample, &VerifierLedger::new()),
        Err(Failure::Inconclusive(e)) if e.code == Code::Usage
    ));
    match AuditReport::from_ledger(sample, &deriving(sample)) {
        Ok(report) => {
            assert_eq!(report.summary().unproven, sample.selected().len() as u64);
            assert_eq!(report.outcomes().len(), sample.selected().len());
        }
        Err(Failure::Inconclusive(e)) => assert_eq!(e.code, Code::ProtocolCoinToss),
        Err(other) => panic!("{other:?}"),
    }
}

fn ledger_rules(ledger: &VerifierLedger) {
    for pair in ledger.derived().windows(2) {
        assert!(pair[0].0 < pair[1].0, "derived not increasing by manifest");
    }
    for record in ledger.proven_samples() {
        assert!(
            ledger.derived().iter().any(|(_, s)| s == record.sample()),
            "a record of a sample not derived"
        );
    }
    for pair in ledger.proven_samples().windows(2) {
        assert!(
            pair[0].sample() < pair[1].sample(),
            "samples not increasing"
        );
    }
    for record in ledger.proven_samples() {
        let set = record.accepted();
        assert!(!set.is_empty(), "an empty record");
        assert_ne!(set.as_bytes().last(), Some(&0), "a trailing zero byte");
        assert_eq!(set.len(), record.uhat().len() as u64, "one Û per index");
    }
    for rows in ledger.used_blinding_rows() {
        assert_ne!(
            rows.rows.as_bytes().last(),
            Some(&0),
            "a trailing zero byte"
        );
    }
}

/// A ledger that derived `sample` and nothing else, decoded from its bytes.
fn deriving(sample: &AuditSample) -> VerifierLedger {
    let mut payload = vec![0; 24];
    payload.extend(1u64.to_le_bytes());
    payload.extend(sample.manifest_digest());
    payload.extend(sample.identity());
    payload.extend(0u64.to_le_bytes());
    VerifierLedger::decode_payload(&payload, &DecodeLimits::MESSAGE).unwrap()
}
