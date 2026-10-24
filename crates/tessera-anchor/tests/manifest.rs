// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0015's manifests with RFC-0060, against `spec/vectors/manifest/`, written by an
//! independent standard-library Python oracle (RFC 8032's reference Ed25519, the BLAKE3 of
//! `spec/vectors/hash/`), and each refusal of a manifest and of a chain.

use ed25519_dalek::Verifier;
use serde_json::Value;
use tessera_anchor::merkle::{inclusion_proof, leaf_hash_bytes, root, verify_inclusion};
use tessera_anchor::{
    ChainError, DigestLogEntry, EntryError, HeapKey, LinkError, LogManifest, ManifestBody,
    ManifestError, check_chain, check_link, key_id, public_key,
};
use tessera_codec::{DecodeLimits, Secret, decode, encode, envelope, schema};

fn hex(v: &Value) -> Vec<u8> {
    let s = v.as_str().unwrap();
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

fn array<const N: usize>(v: &Value) -> [u8; N] {
    hex(v).try_into().unwrap()
}

fn vectors() -> Value {
    serde_json::from_str(include_str!(
        "../../../spec/vectors/manifest/manifests.json"
    ))
    .unwrap()
}

fn seed(v: &Value) -> Secret<HeapKey<32>> {
    Secret::new(HeapKey::from_box(Box::new(array(&v["seed"]))))
}

/// The oracle's chain, rebuilt: each period's entries, root, body, signed message, signature,
/// manifest, envelope, identity and proofs.
fn chain(v: &Value) -> Vec<LogManifest> {
    let seed = seed(v);
    let mut out = Vec::new();
    for m in v["manifests"].as_array().unwrap() {
        let entries: Vec<Vec<u8>> = m["entries"].as_array().unwrap().iter().map(hex).collect();
        // Every entry decodes as a DigestLogEntry and re-encodes to itself.
        for bytes in &entries {
            let entry: DigestLogEntry = decode(bytes, DecodeLimits::MESSAGE).unwrap();
            assert_eq!(&encode(&entry), bytes);
        }
        let leaves: Vec<[u8; 32]> = entries.iter().map(|e| leaf_hash_bytes(e)).collect();
        assert_eq!(
            leaves.iter().map(|l| l.to_vec()).collect::<Vec<_>>(),
            m["leaves"]
                .as_array()
                .unwrap()
                .iter()
                .map(hex)
                .collect::<Vec<_>>()
        );
        let entries_root = root(&leaves);
        assert_eq!(entries_root, array(&m["entries_root"]));
        for (index, proof) in m["proofs"].as_array().unwrap().iter().enumerate() {
            let expected: Vec<[u8; 32]> = proof.as_array().unwrap().iter().map(array).collect();
            assert_eq!(inclusion_proof(&leaves, index).unwrap(), expected);
            let size = leaves.len() as u64;
            assert_eq!(
                verify_inclusion(&leaves[index], index as u64, size, &expected, &entries_root),
                Ok(())
            );
        }
        let body = ManifestBody {
            recorder_key_id: array(&v["recorder_key_id"]),
            log_id: array(&v["log_id"]),
            period_start_ns: m["period_start_ns"].as_u64().unwrap(),
            period_end_ns: m["period_end_ns"].as_u64().unwrap(),
            first_seq: m["first_seq"].as_u64().unwrap(),
            entry_count: entries.len() as u64,
            entries_root,
            prev_manifest: array(&m["prev_manifest"]),
        };
        assert_eq!(body.signed_message(), hex(&m["signed_message"]));
        let manifest = body.sign(&seed).unwrap();
        assert_eq!(manifest.signature.signature, hex(&m["signature"]));
        assert_eq!(encode(&manifest), hex(&m["manifest"]));
        assert_eq!(
            envelope::seal(&schema::LOG_MANIFEST, &manifest),
            hex(&m["envelope"])
        );
        assert_eq!(manifest.identity(), array(&m["identity"]));
        assert_eq!(
            decode::<LogManifest>(&hex(&m["manifest"]), DecodeLimits::MESSAGE),
            Ok(manifest.clone())
        );
        out.push(manifest);
    }
    out
}

#[test]
fn the_chain_matches_the_independent_oracle_and_verifies() {
    let v = vectors();
    let seed = seed(&v);
    let public = public_key(&seed);
    assert_eq!(public, array(&v["public_key"]));
    assert_eq!(key_id(&public), array(&v["recorder_key_id"]));
    let manifests = chain(&v);
    assert_eq!(manifests.len(), 3);
    assert_eq!(manifests[2].body.entry_count, 0, "an empty period");
    assert_eq!(check_chain(&manifests, &[public]), Ok(()));
}

/// RFC 8032 Section 7.1, test 1: the seed's public key, through this crate's API.
#[test]
fn the_public_key_is_rfc_8032s() {
    let seed: [u8; 32] = array(&Value::from(
        "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
    ));
    let s = Secret::new(HeapKey::from_box(Box::new(seed)));
    assert_eq!(
        public_key(&s).to_vec(),
        hex(&Value::from(
            "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
        ))
    );
}

fn honest() -> (Vec<LogManifest>, [u8; 32], Secret<HeapKey<32>>) {
    let v = vectors();
    let s = seed(&v);
    let public = public_key(&s);
    (chain(&v), public, s)
}

#[test]
fn each_refusal_of_a_manifest_alone() {
    let (manifests, public, seed) = honest();
    let m = &manifests[0];
    assert_eq!(m.verify(&[public]), Ok(()));
    let with = |f: &dyn Fn(&mut LogManifest)| {
        let mut forged = m.clone();
        f(&mut forged);
        forged.verify(&[public])
    };
    assert_eq!(
        with(&|f| f.signature.scheme = 1),
        Err(ManifestError::Reserved)
    );
    assert_eq!(
        with(&|f| f.signature.scheme = 2),
        Err(ManifestError::UnknownScheme(2))
    );
    assert_eq!(
        with(&|f| {
            f.signature.public_key.pop();
        }),
        Err(ManifestError::KeyLength)
    );
    assert_eq!(
        with(&|f| f.signature.signature.push(0)),
        Err(ManifestError::SignatureLength)
    );
    assert_eq!(
        with(&|f| f.body.recorder_key_id[0] ^= 1),
        Err(ManifestError::KeyId)
    );
    assert_eq!(m.verify(&[[7; 32]]), Err(ManifestError::Unpinned));
    assert_eq!(m.verify(&[]), Err(ManifestError::Unpinned));
    // Every signed field is covered by the signature.
    for field in 0..8 {
        let r = with(&|f| match field {
            0 => f.body.log_id[3] ^= 1,
            1 => f.body.period_start_ns += 1,
            2 => f.body.period_end_ns += 1,
            3 => f.body.first_seq += 1,
            4 => f.body.entry_count += 1,
            5 => f.body.entries_root[31] ^= 1,
            6 => f.body.prev_manifest[0] ^= 1,
            _ => f.signature.signature[10] ^= 1,
        });
        assert_eq!(r, Err(ManifestError::Signature), "field {field}");
    }
    // A period that ends before it starts, honestly signed.
    let mut body = m.body.clone();
    body.period_end_ns = body.period_start_ns - 1;
    assert_eq!(
        body.sign(&seed).unwrap().verify(&[public]),
        Err(ManifestError::Period)
    );
    // A body whose key identifier is not the signer's.
    let mut body = m.body.clone();
    body.recorder_key_id = [3; 32];
    assert_eq!(body.sign(&seed), Err(ManifestError::KeyId));
}

/// Strict Ed25519 (RFC-0060 'Signatures'), each rule alone: `S + L`, a small-order `R` under the
/// honest key, and a small-order key. The last two satisfy the cofactorless equation, which a lax
/// verifier accepts; the oracle checked both with RFC 8032's arithmetic.
#[test]
fn strict_verification_refuses_each_case_alone() {
    let v = vectors();
    let (manifests, public, _) = honest();
    let m = &manifests[0];
    // S + L, little-endian, in the signature's last 32 bytes.
    let mut forged = m.clone();
    let l: [u8; 32] = [
        0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58, 0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9, 0xde,
        0x14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x10,
    ];
    let mut carry = 0u16;
    for (byte, &add) in forged.signature.signature[32..].iter_mut().zip(&l) {
        let sum = u16::from(*byte) + u16::from(add) + carry;
        *byte = sum as u8;
        carry = sum >> 8;
    }
    assert_eq!(carry, 0);
    assert_eq!(forged.verify(&[public]), Err(ManifestError::Signature));
    let lax = |key: &[u8; 32], message: &[u8], signature: &[u8]| {
        let key = ed25519_dalek::VerifyingKey::from_bytes(key).unwrap();
        key.verify(
            message,
            &ed25519_dalek::Signature::from_slice(signature).unwrap(),
        )
        .is_ok()
    };
    let strict = &v["strictness"];
    // R the identity, under the honest key.
    let r_rule = hex(&strict["r_rule"]);
    assert!(
        lax(&public, &m.body.signed_message(), &r_rule),
        "lax verification accepts it"
    );
    let mut forged = m.clone();
    forged.signature.signature = r_rule;
    assert_eq!(forged.verify(&[public]), Err(ManifestError::Signature));
    // The identity as the key, pinned and with its own identifier; R the base point, S = 1.
    let identity: [u8; 32] = array(&strict["key_rule_public_key"]);
    let key_rule = hex(&strict["key_rule"]);
    assert!(
        lax(&identity, &m.body.signed_message(), &key_rule),
        "lax verification accepts it"
    );
    let mut weak = m.clone();
    weak.signature.public_key = identity.to_vec();
    weak.signature.signature = key_rule;
    weak.body.recorder_key_id = key_id(&identity);
    assert_eq!(weak.verify(&[identity]), Err(ManifestError::Signature));
    // A pinned 32-byte key that is not a point: y = 2 has no x.
    let mut not_a_point = [0u8; 32];
    not_a_point[0] = 2;
    let mut bad = m.clone();
    bad.signature.public_key = not_a_point.to_vec();
    bad.body.recorder_key_id = key_id(&not_a_point);
    assert_eq!(bad.verify(&[not_a_point]), Err(ManifestError::KeyLength));
}

/// A manifest's other checks alone: its range of entries, and an empty period's root.
#[test]
fn a_manifest_refuses_an_overflowing_range_and_a_wrong_empty_root() {
    let (manifests, public, seed) = honest();
    let mut body = manifests[0].body.clone();
    body.first_seq = 2;
    body.entry_count = u64::MAX;
    assert_eq!(
        body.sign(&seed).unwrap().verify(&[public]),
        Err(ManifestError::Range)
    );
    let mut body = manifests[2].body.clone();
    assert_eq!(body.entry_count, 0);
    body.entries_root = [9; 32];
    assert_eq!(
        body.sign(&seed).unwrap().verify(&[public]),
        Err(ManifestError::EmptyRoot)
    );
}

/// Every entry verifies in its manifest at its own `seq`, and each refusal of `verify_entry`
/// holds alone: a `seq` before or after the range, a proof longer than any path, and a proof of
/// another entry.
#[test]
fn entries_verify_at_their_own_seq_in_their_manifest() {
    let v = vectors();
    let (manifests, _, _) = honest();
    for (m, json) in manifests.iter().zip(v["manifests"].as_array().unwrap()) {
        let entries: Vec<DigestLogEntry> = json["entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| decode(&hex(e), DecodeLimits::MESSAGE).unwrap())
            .collect();
        let proofs: Vec<Vec<[u8; 32]>> = json["proofs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p.as_array().unwrap().iter().map(array).collect())
            .collect();
        for (entry, proof) in entries.iter().zip(&proofs) {
            assert_eq!(m.verify_entry(entry, proof), Ok(()));
            // Its seq lies in the manifest's period and range.
            assert!((m.body.period_start_ns..m.body.period_end_ns).contains(&entry.t_ns()));
        }
        if entries.len() > 1 {
            assert!(matches!(
                m.verify_entry(&entries[0], &proofs[1]),
                Err(EntryError::Inclusion(_))
            ));
        }
    }
    let m = &manifests[0];
    let proof = inclusion_proof(
        &v["manifests"][0]["leaves"]
            .as_array()
            .unwrap()
            .iter()
            .map(array)
            .collect::<Vec<_>>(),
        0,
    )
    .unwrap();
    let outside = |seq: u64| DigestLogEntry::SessionClose {
        seq,
        t_ns: m.body.period_start_ns,
        session_id: [0; 16],
    };
    let after = m.body.first_seq + m.body.entry_count;
    assert_eq!(
        m.verify_entry(&outside(after), &proof),
        Err(EntryError::Range)
    );
    let later = &manifests[1];
    assert_eq!(
        later.verify_entry(&outside(later.body.first_seq - 1), &proof),
        Err(EntryError::Range)
    );
    let first: DigestLogEntry = decode(
        &hex(&v["manifests"][0]["entries"][0]),
        DecodeLimits::MESSAGE,
    )
    .unwrap();
    assert_eq!(
        m.verify_entry(&first, &vec![[0; 32]; 65]),
        Err(EntryError::TooLong)
    );
}

#[test]
fn each_break_of_the_chain_alone() {
    let (manifests, public, seed) = honest();
    let pinned = [public];
    assert_eq!(check_chain(&[], &pinned), Err(ChainError::Empty));
    // Not from the genesis: each of its two conditions, alone.
    assert_eq!(
        check_chain(&manifests[1..], &pinned),
        Err(ChainError::Genesis)
    );
    for broken in 0..2 {
        let mut body = manifests[0].body.clone();
        match broken {
            0 => body.prev_manifest[7] = 1,
            _ => body.first_seq = 1,
        }
        let first = body.sign(&seed).unwrap();
        assert_eq!(
            check_chain(&[first], &pinned),
            Err(ChainError::Genesis),
            "{broken}"
        );
    }
    // Reordered.
    let swapped = [
        manifests[0].clone(),
        manifests[2].clone(),
        manifests[1].clone(),
    ];
    assert_eq!(
        check_chain(&swapped, &pinned),
        Err(ChainError::Link {
            index: 1,
            error: LinkError::Previous
        })
    );
    // A manifest invalid on its own.
    let mut bad = manifests.clone();
    bad[1].signature.signature[0] ^= 1;
    assert_eq!(
        check_chain(&bad, &pinned),
        Err(ChainError::Manifest {
            index: 1,
            error: ManifestError::Signature
        })
    );
    // Each link check, alone: re-sign the second manifest with one field changed.
    let (a, b) = (&manifests[0], &manifests[1]);
    assert_eq!(check_link(a, b), Ok(()));
    let resigned = |f: &dyn Fn(&mut ManifestBody)| {
        let mut body = b.body.clone();
        f(&mut body);
        body.sign(&seed).unwrap()
    };
    assert_eq!(
        check_link(a, &resigned(&|x| x.log_id[0] ^= 1)),
        Err(LinkError::LogId)
    );
    assert_eq!(
        check_link(a, &resigned(&|x| x.prev_manifest[5] ^= 1)),
        Err(LinkError::Previous)
    );
    assert_eq!(
        check_link(a, &resigned(&|x| x.first_seq += 1)),
        Err(LinkError::Entries)
    );
    assert_eq!(
        check_link(a, &resigned(&|x| x.first_seq -= 1)),
        Err(LinkError::Entries)
    );
    assert_eq!(
        check_link(a, &resigned(&|x| x.period_start_ns += 1)),
        Err(LinkError::Period)
    );
    // An overflowing entry range, whose wrapped end would be the next manifest's first `seq`.
    let mut huge = a.body.clone();
    huge.first_seq = 2;
    huge.entry_count = u64::MAX;
    let huge = huge.sign(&seed).unwrap();
    let mut next = b.body.clone();
    next.prev_manifest = huge.identity();
    next.first_seq = 1;
    assert_eq!(2u64.wrapping_add(u64::MAX), 1);
    assert_eq!(
        check_link(&huge, &next.sign(&seed).unwrap()),
        Err(LinkError::Entries)
    );
}

/// An entry is proven at the index its `seq` gives: the proof of one index proves no other in
/// the same tree. The size must come from the manifest because a proof can also verify against
/// the same root at another size (RFC-0060 'The tree over a period's entries').
#[test]
fn an_inclusion_proof_holds_only_at_the_entrys_own_index() {
    let v = vectors();
    let m = &v["manifests"][0];
    let leaves: Vec<[u8; 32]> = m["leaves"].as_array().unwrap().iter().map(array).collect();
    let r: [u8; 32] = array(&m["entries_root"]);
    let n = leaves.len() as u64;
    for (index, leaf) in leaves.iter().enumerate() {
        let proof = inclusion_proof(&leaves, index).unwrap();
        for other in (0..n).filter(|&o| o != index as u64) {
            assert!(
                verify_inclusion(leaf, other, n, &proof, &r).is_err(),
                "{index} as {other}"
            );
        }
    }
    // Leaf 0 of four leaves, with proof [d1, H(d2, d3)], also verifies as leaf 0 of three.
    let four: Vec<[u8; 32]> = (0u8..4).map(|i| [i; 32]).collect();
    let proof = inclusion_proof(&four, 0).unwrap();
    let r4 = root(&four);
    assert_eq!(verify_inclusion(&four[0], 0, 4, &proof, &r4), Ok(()));
    assert_eq!(verify_inclusion(&four[0], 0, 3, &proof, &r4), Ok(()));
}
