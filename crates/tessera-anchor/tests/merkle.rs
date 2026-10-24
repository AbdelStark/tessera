// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC 9162's tree and inclusion proofs (RFC-0060): the 98 RFC 6962 inclusion vectors of
//! transparency-dev/merkle, and proofs generated at random positions of random trees, checked
//! by this crate and by the verifier's independent `tv-hash` (DEP-2).

use std::fs;
use std::path::Path;
use tessera_anchor::merkle::{
    Hash, InclusionError, inclusion_proof, node_hash, root, verify_inclusion,
};

fn base64(s: &str) -> Vec<u8> {
    let value = |c: u8| -> u32 {
        match c {
            b'A'..=b'Z' => u32::from(c - b'A'),
            b'a'..=b'z' => u32::from(c - b'a') + 26,
            b'0'..=b'9' => u32::from(c - b'0') + 52,
            b'+' => 62,
            b'/' => 63,
            _ => panic!("not base64: {c}"),
        }
    };
    let bytes: Vec<u8> = s.bytes().filter(|&c| c != b'=').collect();
    let mut out = Vec::new();
    for chunk in bytes.chunks(4) {
        let mut acc = 0u32;
        for &c in chunk {
            acc = (acc << 6) | value(c);
        }
        acc <<= 6 * (4 - chunk.len() as u32);
        for i in 0..chunk.len() - 1 {
            out.push((acc >> (16 - 8 * i)) as u8);
        }
    }
    out
}

fn digest(v: &serde_json::Value) -> Option<Hash> {
    v.as_str().and_then(|s| base64(s).try_into().ok())
}

/// The vectors' digests are 32 bytes where valid; a malformed one, empty or garbage, cannot be a
/// proof element and makes the case fail at parsing, as a decoder would refuse it.
#[test]
fn every_rfc_6962_inclusion_vector_verifies_or_fails_as_it_should() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/rfc6962/inclusion");
    let mut cases = 0;
    let mut valid = 0;
    let mut files = Vec::new();
    let mut pending = vec![dir];
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else {
                files.push(path);
            }
        }
    }
    for path in files {
        let case: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        let want_err = case["wantErr"].as_bool().unwrap();
        let proof: Option<Vec<Hash>> = match &case["proof"] {
            serde_json::Value::Null => Some(Vec::new()),
            list => list.as_array().unwrap().iter().map(digest).collect(),
        };
        let outcome = match (digest(&case["leafHash"]), digest(&case["root"]), proof) {
            (Some(leaf), Some(r), Some(proof)) => verify_inclusion(
                &leaf,
                case["leafIdx"].as_u64().unwrap(),
                case["treeSize"].as_u64().unwrap(),
                &proof,
                &r,
            )
            .is_ok(),
            _ => false,
        };
        assert_eq!(outcome, !want_err, "{}", path.display());
        cases += 1;
        valid += usize::from(outcome);
    }
    assert_eq!((cases, valid), (98, 6));
}

/// SplitMix64 (Steele, Lea, Flood 2014).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn hash(&mut self) -> Hash {
        let mut out = [0u8; 32];
        for chunk in out.chunks_mut(8) {
            chunk.copy_from_slice(&self.next().to_le_bytes());
        }
        out
    }
}

#[test]
fn generated_proofs_verify_here_and_in_the_verifier_at_random_positions() {
    let mut rng = Rng(0x0138);
    let sizes: Vec<usize> = (1..=70).chain([127, 128, 129, 257, 1000]).collect();
    for size in sizes {
        let leaves: Vec<Hash> = (0..size).map(|_| rng.hash()).collect();
        let r = root(&leaves);
        let positions: Vec<usize> = if size <= 70 {
            (0..size).collect()
        } else {
            (0..16)
                .map(|_| (rng.next() % size as u64) as usize)
                .collect()
        };
        for index in positions {
            let proof = inclusion_proof(&leaves, index).unwrap();
            let (i, n) = (index as u64, size as u64);
            assert_eq!(verify_inclusion(&leaves[index], i, n, &proof, &r), Ok(()));
            assert!(tv_hash::rfc6962::verify_inclusion(i, n, &leaves[index], &proof, &r).is_ok());
            // One flipped bit anywhere fails both.
            if !proof.is_empty() {
                let mut bad = proof.clone();
                let at = (rng.next() % bad.len() as u64) as usize;
                bad[at][(rng.next() % 32) as usize] ^= 1;
                assert_eq!(
                    verify_inclusion(&leaves[index], i, n, &bad, &r),
                    Err(InclusionError::Root)
                );
                assert!(
                    tv_hash::rfc6962::verify_inclusion(i, n, &leaves[index], &bad, &r).is_err()
                );
            }
            // The proof of one index proves no other leaf.
            let other = (index + 1) % size;
            if other != index {
                assert!(verify_inclusion(&leaves[other], i, n, &proof, &r).is_err());
            }
        }
        assert_eq!(inclusion_proof(&leaves, size), None);
    }
}

#[test]
fn the_tree_hashes_are_rfc_9162s() {
    // The empty tree is SHA-256 of nothing.
    assert_eq!(
        root(&[]),
        <[u8; 32]>::try_from(
            (0..32)
                .map(|i| u8::from_str_radix(
                    &"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                        [2 * i..2 * i + 2],
                    16
                )
                .unwrap())
                .collect::<Vec<u8>>()
        )
        .unwrap()
    );
    // Three leaves split at two: (l0 l1) l2.
    let l: Vec<Hash> = (0u8..3).map(|i| [i; 32]).collect();
    assert_eq!(root(&l), node_hash(&node_hash(&l[0], &l[1]), &l[2]));
    assert_eq!(root(&l[..1]), l[0]);
    // Each refusal of the verification, alone.
    let proof = inclusion_proof(&l, 0).unwrap();
    let r = root(&l);
    assert_eq!(
        verify_inclusion(&l[0], 3, 3, &proof, &r),
        Err(InclusionError::Index)
    );
    let mut long = proof.clone();
    long.push([9; 32]);
    assert_eq!(
        verify_inclusion(&l[0], 0, 3, &long, &r),
        Err(InclusionError::TooLong)
    );
    assert_eq!(
        verify_inclusion(&l[0], 0, 3, &proof[..1], &r),
        Err(InclusionError::TooShort)
    );
    assert_eq!(
        verify_inclusion(&l[1], 0, 3, &proof, &r),
        Err(InclusionError::Root)
    );
}
