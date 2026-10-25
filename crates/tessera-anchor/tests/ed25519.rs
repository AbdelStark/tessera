// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A differential test against the independent verifier (DEP-2): `tv-hash`'s strict Ed25519
//! verification, written from RFC 8032 and RFC-0060 'Signatures', and `ed25519-dalek` 2.2's
//! `verify_strict`, which the prover signs manifests for. RFC-0060 defines their acceptance set
//! as one, so they must accept and refuse the same signatures:
//! - every crafted case of `spec/vectors/ed25519/strict.json`, from an independent oracle;
//! - random keys, messages and signatures, and each with one bit changed;
//! - each honest signature with `S + k·L` for every `k` that keeps it below `2^256`;
//! - random bytes as keys and signatures.

use std::fs;
use std::path::PathBuf;

use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use serde_json::Value;

/// SplitMix64, the inputs' generator.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn bytes<const N: usize>(&mut self) -> [u8; N] {
        let mut out = [0u8; N];
        for chunk in out.chunks_mut(8) {
            let word = self.next().to_le_bytes();
            chunk.copy_from_slice(&word[..chunk.len()]);
        }
        out
    }
}

fn dalek(key: &[u8; 32], message: &[u8], signature: &[u8; 64]) -> bool {
    VerifyingKey::from_bytes(key)
        .and_then(|k| k.verify_strict(message, &Signature::from_bytes(signature)))
        .is_ok()
}

fn ours(key: &[u8; 32], message: &[u8], signature: &[u8; 64]) -> bool {
    tv_hash::ed25519::verify_strict(key, message, signature).is_ok()
}

/// Both implementations' verdicts on one input, which must agree.
fn agree(key: &[u8; 32], message: &[u8], signature: &[u8; 64], what: &str) -> bool {
    let (d, o) = (
        dalek(key, message, signature),
        ours(key, message, signature),
    );
    assert_eq!(d, o, "{what}: ed25519-dalek {d}, tv-hash {o}");
    d
}

fn hex(v: &Value) -> Vec<u8> {
    let s = v.as_str().expect("hex");
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex"))
        .collect()
}

/// `L`, the order of the base point, little-endian.
const L: [u8; 32] = [
    0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58, 0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9, 0xde, 0x14,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x10,
];

/// `a + b` over 256 bits, little-endian, or `None` past `2^256`.
fn add(a: &[u8; 32], b: &[u8; 32]) -> Option<[u8; 32]> {
    let mut out = [0u8; 32];
    let mut carry = 0u16;
    for i in 0..32 {
        let s = u16::from(a[i]) + u16::from(b[i]) + carry;
        out[i] = s as u8;
        carry = s >> 8;
    }
    (carry == 0).then_some(out)
}

#[test]
fn the_oracles_crafted_cases() {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/ed25519/strict.json");
    let v: Value =
        serde_json::from_str(&fs::read_to_string(path).expect("the vectors")).expect("JSON");
    let cases = v["cases"].as_array().expect("cases");
    assert!(cases.len() >= 150);
    for case in cases {
        let name = case["name"].as_str().expect("a name");
        let key: [u8; 32] = hex(&case["public_key"]).try_into().expect("32 bytes");
        let signature: [u8; 64] = hex(&case["signature"]).try_into().expect("64 bytes");
        let message = hex(&case["message"]);
        let accepted = agree(&key, &message, &signature, name);
        assert_eq!(accepted, case["expected"] == "accept", "{name}");
    }
}

#[test]
fn random_signatures_and_one_bit_changes() {
    let mut rng = Rng(0x0515_0001);
    for i in 0..3_000 {
        let signer = SigningKey::from_bytes(&rng.bytes());
        let key = signer.verifying_key().to_bytes();
        let len = (rng.next() % 200) as usize;
        let message: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
        let signature = signer.sign(&message).to_bytes();
        assert!(agree(&key, &message, &signature, &format!("honest {i}")));
        // One bit of the signature, of the key or of the message.
        let bit = (rng.next() % 8) as u8;
        let mut s = signature;
        s[(rng.next() % 64) as usize] ^= 1 << bit;
        agree(&key, &message, &s, &format!("signature bit, case {i}"));
        let mut k = key;
        k[(rng.next() % 32) as usize] ^= 1 << bit;
        agree(&k, &message, &signature, &format!("key bit, case {i}"));
        if !message.is_empty() {
            let mut m = message.clone();
            let at = (rng.next() as usize) % m.len();
            m[at] ^= 1 << bit;
            assert!(!agree(
                &key,
                &m,
                &signature,
                &format!("message bit, case {i}")
            ));
        }
        // S + k·L, while it stays below 2^256: the same group element, refused by both.
        if i % 50 == 0 {
            let mut scalar: [u8; 32] = signature[32..].try_into().expect("S");
            while let Some(next) = add(&scalar, &L) {
                scalar = next;
                let mut malleated = signature;
                malleated[32..].copy_from_slice(&scalar);
                assert!(!agree(
                    &key,
                    &message,
                    &malleated,
                    &format!("S + kL, case {i}")
                ));
            }
        }
    }
}

#[test]
fn random_bytes_as_keys_and_signatures() {
    let mut rng = Rng(0x0515_0002);
    let mut accepted = 0;
    for i in 0..5_000 {
        let key: [u8; 32] = rng.bytes();
        let signature: [u8; 64] = rng.bytes();
        let message: [u8; 16] = rng.bytes();
        accepted += usize::from(agree(&key, &message, &signature, &format!("random {i}")));
    }
    assert_eq!(accepted, 0, "a random signature verified");
}
