// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Strict Ed25519 (RFC-0060 'Signatures') against independent data:
//!
//! - RFC 8032 §7.1's five Ed25519 vectors, read from the section's text in
//!   `spec/vectors/ed25519/`, and every one-bit change of their keys, signatures and messages;
//! - every signature of `spec/vectors/manifest/manifests.json`, written by the manifest oracle,
//!   and its two strictness cases;
//! - `spec/vectors/ed25519/strict.json`, the Python oracle's crafted cases, one or more per rule,
//!   each with the exact first rule it fails;
//! - the fourteen encodings of small-order points, written out here from their definition, and
//!   the non-canonical encodings `y ∈ [p, 2^255)`, which must decode as `y − p` does.

mod support;

use std::fs;

use support::{Rng, list, text};
use tv_hash::ed25519::{Ed25519Error, verify_strict};
use tv_testkit::{Json, hex, read_vectors, vectors_dir};

/// `L`, little-endian.
const L: [u8; 32] = [
    0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58, 0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9, 0xde, 0x14,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x10,
];

fn array<const N: usize>(bytes: &[u8]) -> [u8; N] {
    bytes
        .try_into()
        .unwrap_or_else(|_| panic!("{} bytes where {N} are expected", bytes.len()))
}

/// `a + b` for 256-bit little-endian integers, and whether it carried out of 256 bits.
fn add_256(a: &[u8; 32], b: &[u8; 32]) -> ([u8; 32], bool) {
    let mut out = [0u8; 32];
    let mut carry = 0u16;
    for i in 0..32 {
        let sum = u16::from(a[i]) + u16::from(b[i]) + carry;
        out[i] = sum as u8;
        carry = sum >> 8;
    }
    (out, carry == 1)
}

/// Whether the 256-bit little-endian integer `s` is below `L`, compared from the top byte.
fn below_l(s: &[u8; 32]) -> bool {
    for i in (0..32).rev() {
        if s[i] != L[i] {
            return s[i] < L[i];
        }
    }
    false
}

fn with_s(signature: &[u8; 64], s: &[u8; 32]) -> [u8; 64] {
    let mut out = *signature;
    out[32..].copy_from_slice(s);
    out
}

fn with_r(signature: &[u8; 64], r: &[u8; 32]) -> [u8; 64] {
    let mut out = *signature;
    out[..32].copy_from_slice(r);
    out
}

fn s_of(signature: &[u8; 64]) -> [u8; 32] {
    array(&signature[32..])
}

// ----- RFC 8032 §7.1 ------------------------------------------------------------------------------

/// One test of RFC 8032 §7.1.
struct RfcCase {
    name: String,
    secret_key: Vec<u8>,
    public_key: Vec<u8>,
    message_len: usize,
    message: Vec<u8>,
    signature: Vec<u8>,
}

/// The tests of `rfc8032-section-7.1.txt`, the section's text as published, page breaks
/// included: each `-----TEST name` opens a case, a label opens a field, and the lines of
/// lowercase hexadecimal that follow, across page breaks, are its bytes.
fn rfc8032_cases() -> Vec<RfcCase> {
    let path = vectors_dir().join("ed25519/rfc8032-section-7.1.txt");
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut cases: Vec<RfcCase> = Vec::new();
    let mut field: Option<&'static str> = None;
    for line in text.lines().map(str::trim) {
        if let Some(name) = line.strip_prefix("-----TEST ") {
            cases.push(RfcCase {
                name: name.to_owned(),
                secret_key: Vec::new(),
                public_key: Vec::new(),
                message_len: usize::MAX,
                message: Vec::new(),
                signature: Vec::new(),
            });
            field = None;
            continue;
        }
        if let Some(rest) = line.strip_prefix("MESSAGE (length ") {
            let len = rest
                .strip_suffix(" bytes):")
                .or_else(|| rest.strip_suffix(" byte):"))
                .expect("a message length");
            cases.last_mut().expect("a case").message_len = len.parse().unwrap();
            field = Some("message");
            continue;
        }
        match line {
            "-----" => field = None,
            "ALGORITHM:" => field = Some("algorithm"),
            "SECRET KEY:" => field = Some("secret"),
            "PUBLIC KEY:" => field = Some("public"),
            "SIGNATURE:" => field = Some("signature"),
            _ if field == Some("algorithm") && !line.is_empty() => assert_eq!(line, "Ed25519"),
            _ if !line.is_empty()
                && line.len() % 2 == 0
                && line
                    .bytes()
                    .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)) =>
            {
                let case = cases.last_mut().expect("a case");
                let bytes = hex(line);
                match field {
                    Some("secret") => case.secret_key.extend(bytes),
                    Some("public") => case.public_key.extend(bytes),
                    Some("message") => case.message.extend(bytes),
                    Some("signature") => case.signature.extend(bytes),
                    other => panic!("hexadecimal outside a field: {other:?}: {line}"),
                }
            }
            // Prose, page footers and headers, blank lines.
            _ => {}
        }
    }
    cases
}

#[test]
fn rfc_8032_section_7_1() {
    let cases = rfc8032_cases();
    let names: Vec<&str> = cases.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["1", "2", "3", "1024", "SHA(abc)"]);
    for case in &cases {
        assert_eq!(case.secret_key.len(), 32, "TEST {}", case.name);
        assert_eq!(case.message.len(), case.message_len, "TEST {}", case.name);
        let key = array::<32>(&case.public_key);
        let signature = array::<64>(&case.signature);
        assert_eq!(
            verify_strict(&key, &case.message, &signature),
            Ok(()),
            "TEST {}",
            case.name
        );
    }
    let lengths: Vec<usize> = cases.iter().map(|c| c.message.len()).collect();
    assert_eq!(lengths, [0, 1, 2, 1023, 64]);
}

/// Every one-bit change of an RFC 8032 key or signature is refused, by the rule its position
/// allows: a bit of `S` by the range of `S` or by the equation, exactly as `S` then compares with
/// `L`; a bit of `R` or `A` by decoding or by the equation (a flipped bit 255 negates `x`, still a
/// point). A changed message bit fails the equation.
#[test]
fn every_one_bit_change_of_an_rfc_8032_vector_is_refused() {
    for case in rfc8032_cases() {
        let key = array::<32>(&case.public_key);
        let signature = array::<64>(&case.signature);
        for bit in 0..512 {
            let mut changed = signature;
            changed[bit / 8] ^= 1 << (bit % 8);
            let outcome = verify_strict(&key, &case.message, &changed);
            if bit >= 256 {
                let expected = if below_l(&s_of(&changed)) {
                    Ed25519Error::Equation
                } else {
                    Ed25519Error::ScalarOutOfRange
                };
                assert_eq!(outcome, Err(expected), "TEST {} bit {bit}", case.name);
            } else if bit == 255 {
                assert_eq!(outcome, Err(Ed25519Error::Equation), "TEST {}", case.name);
            } else {
                assert!(
                    matches!(
                        outcome,
                        Err(Ed25519Error::RNotAPoint | Ed25519Error::Equation)
                    ),
                    "TEST {} bit {bit}: {outcome:?}",
                    case.name
                );
            }
        }
        for bit in 0..256 {
            let mut changed = key;
            changed[bit / 8] ^= 1 << (bit % 8);
            let outcome = verify_strict(&changed, &case.message, &signature);
            assert!(
                matches!(
                    outcome,
                    Err(Ed25519Error::KeyNotAPoint | Ed25519Error::Equation)
                ),
                "TEST {} key bit {bit}: {outcome:?}",
                case.name
            );
        }
        let bits = case.message.len() * 8;
        let positions = if bits == 0 {
            Vec::new()
        } else {
            vec![0, bits / 2 + 3, bits - 1]
        };
        for bit in positions {
            let mut changed = case.message.clone();
            changed[bit / 8] ^= 1 << (bit % 8);
            assert_eq!(
                verify_strict(&key, &changed, &signature),
                Err(Ed25519Error::Equation),
                "TEST {} message bit {bit}",
                case.name
            );
        }
        // One byte more or less of message.
        let mut longer = case.message.clone();
        longer.push(0);
        assert_eq!(
            verify_strict(&key, &longer, &signature),
            Err(Ed25519Error::Equation)
        );
    }
}

/// `S + L`, `S + 2L` and every larger multiple below `2^256` satisfy the equation modulo `L`,
/// and are refused by the range of `S` alone; so are `S = L` and `S = 2^256 − 1`.
#[test]
fn s_at_or_above_l_is_refused_although_the_equation_holds_modulo_l() {
    for case in rfc8032_cases() {
        let key = array::<32>(&case.public_key);
        let signature = array::<64>(&case.signature);
        let mut s = s_of(&signature);
        let mut multiples = 0;
        loop {
            let (next, carried) = add_256(&s, &L);
            if carried {
                break;
            }
            s = next;
            multiples += 1;
            assert_eq!(
                verify_strict(&key, &case.message, &with_s(&signature, &s)),
                Err(Ed25519Error::ScalarOutOfRange),
                "TEST {} S + {multiples}L",
                case.name
            );
        }
        // S < L < 2^253: at least seven multiples of L fit below 2^256.
        assert!(multiples >= 7, "{multiples}");
        for s in [L, [0xff; 32]] {
            assert_eq!(
                verify_strict(&key, &case.message, &with_s(&signature, &s)),
                Err(Ed25519Error::ScalarOutOfRange)
            );
        }
        let mut l_minus_1 = L;
        l_minus_1[0] -= 1;
        assert_eq!(
            verify_strict(&key, &case.message, &with_s(&signature, &l_minus_1)),
            Err(Ed25519Error::Equation)
        );
    }
}

// ----- Small-order and non-canonical encodings ---------------------------------------------------

/// The eight points of small order, canonically encoded. With the six other encodings below,
/// `y = p` and `y = p + 1` with either sign bit and `y = 1`, `y = p − 1` with the sign bit set,
/// they are the fourteen encodings of small-order points that RFC-0060's decoding meets; the list
/// is the well-known one (Chalkias, Garillot and Nikolaenko, "Taming the many EdDSAs", 2020), and
/// the oracle of `spec/vectors/ed25519/` derives it again from the curve.
const SMALL_ORDER_CANONICAL: [&str; 8] = [
    // The identity (0, 1).
    "0100000000000000000000000000000000000000000000000000000000000000",
    // (0, −1), of order 2.
    "ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
    // (±√−1, 0), of order 4.
    "0000000000000000000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000000000000080",
    // The four points of order 8.
    "c7176a703d4dd84fba3c0b760d10670f2a2053fa2c39ccc64ec7fd7792ac037a",
    "c7176a703d4dd84fba3c0b760d10670f2a2053fa2c39ccc64ec7fd7792ac03fa",
    "26e8958fc2b227b045c3f489f2ef98f0d5dfac05d3c63339b13802886d53fc05",
    "26e8958fc2b227b045c3f489f2ef98f0d5dfac05d3c63339b13802886d53fc85",
];

/// Non-canonical encodings of small-order points that decode: `y = p` (the order-4 points) and
/// `y = p + 1` (the identity) with sign bit 0, and `y = p` with sign bit 1.
const SMALL_ORDER_NON_CANONICAL: [&str; 3] = [
    "edffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
    "edffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    "eeffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
];

/// Encodings with `x = 0` and the sign bit set, which RFC 8032 §5.1.3 refuses and RFC-0060 keeps
/// refusing: the identity with `y = 1` and `y = p + 1`, and `(0, −1)`.
const X_ZERO_SIGN_SET: [&str; 3] = [
    "0100000000000000000000000000000000000000000000000000000000000080",
    "eeffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    "ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
];

fn rfc_test_1() -> ([u8; 32], Vec<u8>, [u8; 64]) {
    let case = rfc8032_cases().into_iter().next().unwrap();
    (
        array(&case.public_key),
        case.message,
        array(&case.signature),
    )
}

#[test]
fn small_order_r_and_keys_are_refused_in_every_encoding() {
    let (key, message, signature) = rfc_test_1();
    let decodable = SMALL_ORDER_CANONICAL
        .iter()
        .chain(&SMALL_ORDER_NON_CANONICAL);
    for encoding in decodable {
        let point = array::<32>(&hex(encoding));
        assert_eq!(
            verify_strict(&key, &message, &with_r(&signature, &point)),
            Err(Ed25519Error::RSmallOrder),
            "R = {encoding}"
        );
        assert_eq!(
            verify_strict(&point, &message, &signature),
            Err(Ed25519Error::KeySmallOrder),
            "A = {encoding}"
        );
    }
    for encoding in X_ZERO_SIGN_SET {
        let point = array::<32>(&hex(encoding));
        assert_eq!(
            verify_strict(&key, &message, &with_r(&signature, &point)),
            Err(Ed25519Error::RNotAPoint),
            "R = {encoding}"
        );
        assert_eq!(
            verify_strict(&point, &message, &signature),
            Err(Ed25519Error::KeyNotAPoint),
            "A = {encoding}"
        );
    }
}

/// `y + p` for `y < 19`, with sign bit `sign`, and `y` itself.
fn y_encodings(y: u8, sign: bool) -> ([u8; 32], [u8; 32]) {
    let mut canonical = [0u8; 32];
    canonical[0] = y;
    let mut p = [0xffu8; 32];
    p[31] = 0x7f;
    p[0] = 0xed;
    let (mut non_canonical, carried) = add_256(&p, &canonical);
    assert!(!carried && non_canonical[31] & 0x80 == 0);
    if sign {
        canonical[31] |= 0x80;
        non_canonical[31] |= 0x80;
    }
    (canonical, non_canonical)
}

/// RFC-0060's decoding accepts `y ∈ [p, 2^255)` as `y − p`: as `R` and as `A`, every
/// non-canonical encoding meets the same rule as the canonical encoding of the same `y` and sign.
/// A non-canonical `R` that decodes to a point of large order still fails the equation, whose
/// encoding is canonical: that is the one place RFC-0060's set differs from RFC 8032 §5.1.3's
/// decoding, and it changes the reason, not the verdict.
#[test]
fn non_canonical_y_decodes_as_its_remainder() {
    let (key, message, signature) = rfc_test_1();
    let mut seen = std::collections::BTreeMap::new();
    for y in 0..19u8 {
        for sign in [false, true] {
            let (canonical, non_canonical) = y_encodings(y, sign);
            let as_r = verify_strict(&key, &message, &with_r(&signature, &non_canonical));
            assert_eq!(
                as_r,
                verify_strict(&key, &message, &with_r(&signature, &canonical)),
                "R, y = p + {y}, sign {sign}"
            );
            let as_key = verify_strict(&non_canonical, &message, &signature);
            // `k` hashes the key's bytes as given, so the two keys' equations differ, but neither
            // can hold without the discrete logarithm of a point whose `y` is below 19.
            assert_eq!(
                as_key,
                verify_strict(&canonical, &message, &signature),
                "A, y = p + {y}, sign {sign}"
            );
            assert_ne!(as_r, Ok(()));
            assert_ne!(as_key, Ok(()));
            *seen.entry(format!("{as_r:?}")).or_insert(0) += 1;
        }
    }
    // Some non-canonical encodings decode to points of large order and reach the equation.
    assert!(seen.contains_key("Err(Equation)"), "{seen:?}");
    assert!(seen.contains_key("Err(RNotAPoint)"), "{seen:?}");
}

// ----- spec/vectors/manifest -----------------------------------------------------------------------

#[test]
fn every_manifest_signature_verifies() {
    let doc = read_vectors("manifest/manifests.json");
    let key = array::<32>(&hex(text(&doc, "public_key")));
    let manifests = list(&doc, "manifests");
    assert_eq!(manifests.len(), 3);
    for (index, manifest) in manifests.iter().enumerate() {
        let message = hex(text(manifest, "signed_message"));
        assert!(message.starts_with(b"tessera-manifest/v1"));
        assert_eq!(message.len(), 163, "RFC-0060 'Signatures': 163 bytes");
        let signature = array::<64>(&hex(text(manifest, "signature")));
        assert_eq!(
            verify_strict(&key, &message, &signature),
            Ok(()),
            "manifest {index}"
        );
        let mut tampered = message.clone();
        tampered[19] ^= 1;
        assert_eq!(
            verify_strict(&key, &tampered, &signature),
            Err(Ed25519Error::Equation)
        );
    }
}

/// The manifest oracle's two strictness cases, which satisfy the cofactorless equation and are
/// refused by one strict rule each: `R` the identity under the honest key, and the identity as the
/// key.
#[test]
fn the_manifest_strictness_cases_are_refused_by_their_rule() {
    let doc = read_vectors("manifest/manifests.json");
    let key = array::<32>(&hex(text(&doc, "public_key")));
    let message = hex(text(&list(&doc, "manifests")[0], "signed_message"));
    let strictness = doc.get("strictness").expect("strictness cases");
    let r_rule = array::<64>(&hex(text(strictness, "r_rule")));
    assert_eq!(
        verify_strict(&key, &message, &r_rule),
        Err(Ed25519Error::RSmallOrder)
    );
    let small_key = array::<32>(&hex(text(strictness, "key_rule_public_key")));
    let key_rule = array::<64>(&hex(text(strictness, "key_rule")));
    assert_eq!(
        verify_strict(&small_key, &message, &key_rule),
        Err(Ed25519Error::KeySmallOrder)
    );
}

// ----- spec/vectors/ed25519/strict.json ----------------------------------------------------------

/// The verifier's error for an oracle outcome name.
fn outcome(name: &str) -> Result<(), Ed25519Error> {
    Err(match name {
        "accept" => return Ok(()),
        "s-out-of-range" => Ed25519Error::ScalarOutOfRange,
        "r-not-a-point" => Ed25519Error::RNotAPoint,
        "key-not-a-point" => Ed25519Error::KeyNotAPoint,
        "r-small-order" => Ed25519Error::RSmallOrder,
        "key-small-order" => Ed25519Error::KeySmallOrder,
        "equation" => Ed25519Error::Equation,
        other => panic!("unknown outcome {other}"),
    })
}

fn strict_cases() -> Vec<Json> {
    let doc = read_vectors("ed25519/strict.json");
    assert_eq!(doc.get("version").and_then(Json::as_u64), Some(1));
    let order: Vec<&str> = list(&doc, "check_order")
        .iter()
        .map(|j| j.as_str().unwrap())
        .collect();
    assert_eq!(
        order,
        [
            "s-out-of-range",
            "r-not-a-point",
            "key-not-a-point",
            "r-small-order",
            "key-small-order",
            "equation"
        ],
        "the oracle's order of rules is the verifier's"
    );
    list(&doc, "cases").to_vec()
}

#[test]
fn every_oracle_case_meets_its_rule() {
    let cases = strict_cases();
    assert_eq!(cases.len(), 154);
    let mut by_outcome = std::collections::BTreeMap::new();
    for case in &cases {
        let name = text(case, "name");
        let key = array::<32>(&hex(text(case, "public_key")));
        let message = hex(text(case, "message"));
        let signature = array::<64>(&hex(text(case, "signature")));
        let expected = text(case, "expected");
        assert_eq!(
            verify_strict(&key, &message, &signature),
            outcome(expected),
            "{name}: {}",
            text(case, "comment")
        );
        *by_outcome.entry(expected.to_owned()).or_insert(0usize) += 1;
    }
    // Every rule is exercised, and so is acceptance.
    for expected in [
        "accept",
        "s-out-of-range",
        "r-not-a-point",
        "key-not-a-point",
        "r-small-order",
        "key-small-order",
        "equation",
    ] {
        assert!(
            by_outcome.contains_key(expected),
            "no case for {expected}: {by_outcome:?}"
        );
    }
}

/// The oracle's cases that only a lax verifier would accept: they satisfy the cofactored
/// equation, and strict verification refuses each.
#[test]
fn cofactored_only_cases_are_refused() {
    let mut count = 0;
    for case in strict_cases() {
        let cofactored = case.get("cofactored") == Some(&Json::Bool(true));
        if cofactored && text(&case, "expected") != "accept" {
            count += 1;
            let key = array::<32>(&hex(text(&case, "public_key")));
            let message = hex(text(&case, "message"));
            let signature = array::<64>(&hex(text(&case, "signature")));
            assert!(verify_strict(&key, &message, &signature).is_err());
        }
    }
    assert!(count > 0);
}

/// The oracle derives the small-order encodings from the curve; this file writes them out from
/// the literature. The two lists are the same fourteen encodings.
#[test]
fn the_oracle_and_this_file_agree_on_the_small_order_encodings() {
    let mut oracle: Vec<String> = strict_cases()
        .iter()
        .filter(|case| text(case, "group") == "small-order-r")
        .map(|case| text(case, "signature")[..64].to_owned())
        .collect();
    let mut here: Vec<String> = SMALL_ORDER_CANONICAL
        .iter()
        .chain(&SMALL_ORDER_NON_CANONICAL)
        .chain(&X_ZERO_SIGN_SET)
        .map(|s| (*s).to_owned())
        .collect();
    oracle.sort();
    here.sort();
    assert_eq!(oracle, here);
}

/// Seeded random keys, signatures and messages, and random signatures under an honest key, never
/// panic and are all refused (RFC-0019 'Robustness'); a third of the signatures have `S < L`, so
/// that decoding and the equation are reached.
#[test]
fn random_inputs_are_refused_without_panicking() {
    let (honest_key, _, _) = rfc_test_1();
    let mut rng = Rng::new(0xed25519);
    for round in 0..600 {
        let key = if round % 2 == 0 {
            honest_key
        } else {
            rng.digest()
        };
        let mut signature = [0u8; 64];
        signature[..32].copy_from_slice(&rng.digest());
        signature[32..].copy_from_slice(&rng.digest());
        if round % 3 == 0 {
            signature[63] &= 0x0f;
        }
        let len = rng.index(300);
        let message = rng.bytes(len);
        let outcome = verify_strict(&key, &message, &signature);
        assert!(outcome.is_err(), "round {round}");
        if !below_l(&s_of(&signature)) {
            assert_eq!(outcome, Err(Ed25519Error::ScalarOutOfRange));
        }
    }
}
