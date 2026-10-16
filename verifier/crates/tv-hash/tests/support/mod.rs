// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Test support shared by the integration tests: readers for the vector formats that
//! `tv-testkit` does not cover (base64, NIST's `.rsp` files), and oracles that compute what
//! `tv-hash` must return without `tv-hash`.
//!
//! The oracles are written from the specifications alone and differ in structure from the
//! crate's code:
//!
//! - BLAKE3 Merkle trees are built level by level with the test kit's reference BLAKE3, and a
//!   canonical multiproof is read off the full tree level by level, as RFC-0004 describes it
//!   (the crate counts siblings first and then climbs leaf by leaf).
//! - RFC 6962 roots and inclusion paths follow the recursive definitions `MTH` and `PATH` of
//!   RFC 6962 §2.1 (the crate runs RFC 9162's iterative check).
//! - [`multiproof_outcome`] and [`inclusion_outcome`] predict the exact result of each check, in
//!   the order the crate documents, recomputing a proof's root in their own way: level by level
//!   for a multiproof, by `PATH` read backwards for an inclusion proof.

#![allow(dead_code)]

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use tv_hash::Digest;
use tv_hash::merkle::MultiproofError;
use tv_hash::rfc6962::InclusionError;
use tv_hash::sha256::sha256;
use tv_testkit::{Json, SplitMix64, blake3, vectors_dir};

// ----- Vector formats ---------------------------------------------------------------------------

/// The member `key` of a vector object, which must be a string.
pub fn text<'a>(object: &'a Json, key: &str) -> &'a str {
    object
        .get(key)
        .and_then(Json::as_str)
        .unwrap_or_else(|| panic!("`{key}` is not a string"))
}

/// The member `key` of a vector object, which must be an array.
pub fn list<'a>(object: &'a Json, key: &str) -> &'a [Json] {
    object
        .get(key)
        .and_then(Json::as_array)
        .unwrap_or_else(|| panic!("`{key}` is not an array"))
}

/// The member `key` of a vector object, which must be a `u64`, written as a number or as a
/// decimal string.
pub fn number(object: &Json, key: &str) -> u64 {
    let value = object
        .get(key)
        .unwrap_or_else(|| panic!("`{key}` is missing"));
    value
        .as_u64()
        .or_else(|| value.as_str().and_then(|s| s.parse().ok()))
        .unwrap_or_else(|| panic!("`{key}` is not a u64"))
}

/// A 32-byte digest from hexadecimal.
pub fn digest_hex(text: &str) -> Digest {
    tv_testkit::hex(text)
        .try_into()
        .unwrap_or_else(|_| panic!("{text} is not 32 bytes"))
}

/// The bytes of standard base64 (RFC 4648 §4) with padding.
pub fn base64(text: &str) -> Vec<u8> {
    let value = |c: u8| -> u32 {
        match c {
            b'A'..=b'Z' => u32::from(c - b'A'),
            b'a'..=b'z' => u32::from(c - b'a') + 26,
            b'0'..=b'9' => u32::from(c - b'0') + 52,
            b'+' => 62,
            b'/' => 63,
            _ => panic!("{c:?} is not base64"),
        }
    };
    assert!(text.len() % 4 == 0, "base64 of length {}", text.len());
    let mut out = Vec::new();
    for quad in text.as_bytes().chunks(4) {
        let padding = quad.iter().rev().take_while(|&&c| c == b'=').count();
        let mut acc = 0u32;
        for &c in &quad[..4 - padding] {
            acc = acc << 6 | value(c);
        }
        acc <<= 6 * padding as u32;
        let bytes = acc.to_be_bytes();
        out.extend_from_slice(&bytes[1..4 - padding]);
    }
    out
}

/// The `name = value` lines of a NIST CAVP response file, in order, comments and section
/// headers skipped.
pub fn rsp_fields(rel: &str) -> Vec<(String, String)> {
    let path = vectors_dir().join(rel);
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#') && !line.starts_with('['))
        .map(|line| {
            let (name, value) = line
                .split_once('=')
                .unwrap_or_else(|| panic!("{rel}: not a field: {line}"));
            (name.trim().to_owned(), value.trim().to_owned())
        })
        .collect()
}

/// Every file under `spec/vectors/<rel>`, sorted by path.
pub fn files_under(rel: &str) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
            let path = entry.expect("a directory entry").path();
            if path.is_dir() {
                walk(&path, out);
            } else {
                out.push(path);
            }
        }
    }
    let mut out = Vec::new();
    walk(&vectors_dir().join(rel), &mut out);
    out.sort();
    out
}

// ----- Seeded inputs ----------------------------------------------------------------------------

/// A seeded generator with the helpers the tests need.
pub struct Rng(pub SplitMix64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(SplitMix64::new(seed))
    }

    pub fn next(&mut self) -> u64 {
        self.0.next_u64()
    }

    /// Uniform below `bound`, which is not 0.
    pub fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }

    pub fn index(&mut self, bound: usize) -> usize {
        self.below(bound as u64) as usize
    }

    pub fn digest(&mut self) -> Digest {
        let mut out = [0u8; 32];
        for chunk in out.chunks_mut(8) {
            chunk.copy_from_slice(&self.next().to_le_bytes());
        }
        out
    }

    pub fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next() as u8).collect()
    }
}

// ----- BLAKE3 Merkle trees (RFC-0004 'Merkle trees') ----------------------------------------------

/// `H("merkle-node", left ‖ right)` by the test kit's reference BLAKE3.
pub fn reference_node(left: &Digest, right: &Digest) -> Digest {
    blake3::derive_key("tessera/v1/merkle-node", &[*left, *right].concat())
}

/// Every level of a tree of a power of two of leaves, the leaves first and the root last.
pub struct Tree {
    pub levels: Vec<Vec<Digest>>,
}

impl Tree {
    pub fn new(leaves: Vec<Digest>) -> Self {
        assert!(leaves.len().is_power_of_two());
        let mut levels = vec![leaves];
        while levels.last().unwrap().len() > 1 {
            let below = levels.last().unwrap();
            let above = below
                .chunks(2)
                .map(|pair| reference_node(&pair[0], &pair[1]))
                .collect();
            levels.push(above);
        }
        Self { levels }
    }

    pub fn random(log_n: u32, rng: &mut Rng) -> Self {
        Self::new((0..1u64 << log_n).map(|_| rng.digest()).collect())
    }

    pub fn n(&self) -> u64 {
        self.levels[0].len() as u64
    }

    pub fn root(&self) -> Digest {
        self.levels.last().unwrap()[0]
    }

    /// The opened leaves at `indices`, which are sorted and distinct.
    pub fn open(&self, indices: &[u64]) -> Vec<(u64, Digest)> {
        indices
            .iter()
            .map(|&i| (i, self.levels[0][i as usize]))
            .collect()
    }

    /// The canonical multiproof of `indices` (sorted and distinct, below `n`): level by level from
    /// the leaves upward, the siblings of the known nodes that are not known themselves, left to
    /// right.
    pub fn multiproof(&self, indices: &[u64]) -> Vec<Digest> {
        canonical_multiproof(&self.levels, indices)
    }
}

/// The canonical multiproof of `indices` in the tree whose levels, leaves first, are `levels`.
pub fn canonical_multiproof(levels: &[Vec<Digest>], indices: &[u64]) -> Vec<Digest> {
    let mut known: BTreeSet<u64> = indices.iter().copied().collect();
    let mut siblings = Vec::new();
    for level in &levels[..levels.len() - 1] {
        for &i in &known {
            if !known.contains(&(i ^ 1)) {
                siblings.push(level[(i ^ 1) as usize]);
            }
        }
        known = known.iter().map(|i| i >> 1).collect();
    }
    siblings
}

/// The number of digests in the canonical multiproof of `indices` in a tree of `n` leaves, from
/// the indices alone.
pub fn canonical_count(n: u64, indices: &[u64]) -> usize {
    let mut known: BTreeSet<u64> = indices.iter().copied().collect();
    let mut count = 0;
    for _ in 0..n.trailing_zeros() {
        count += known.iter().filter(|&&i| !known.contains(&(i ^ 1))).count();
        known = known.iter().map(|i| i >> 1).collect();
    }
    count
}

/// The root that `leaves` and `siblings` lead to in a tree of `n` leaves, recomputed level by
/// level from the leaves upward: at each level, every known node takes its sibling from the known
/// nodes or, failing that, from the next digest of `siblings`, left to right. `None` when the
/// digests run out or are left over.
pub fn root_from_multiproof(
    n: u64,
    leaves: &[(u64, Digest)],
    siblings: &[Digest],
) -> Option<Digest> {
    let mut known: std::collections::BTreeMap<u64, Digest> = leaves.iter().copied().collect();
    let mut proof = siblings.iter();
    for _ in 0..n.trailing_zeros() {
        let mut above = std::collections::BTreeMap::new();
        for (&i, digest) in &known {
            if i & 1 == 1 && known.contains_key(&(i ^ 1)) {
                continue;
            }
            let sibling = match known.get(&(i ^ 1)) {
                Some(sibling) => *sibling,
                None => *proof.next()?,
            };
            let (left, right) = if i & 1 == 0 {
                (*digest, sibling)
            } else {
                (sibling, *digest)
            };
            above.insert(i >> 1, reference_node(&left, &right));
        }
        known = above;
    }
    if proof.next().is_some() || known.len() != 1 {
        return None;
    }
    known.get(&0).copied()
}

/// What `verify_multiproof(root, n, leaves, siblings)` must return, by the order of checks the
/// crate documents: the structural checks, then the root recomputed by [`root_from_multiproof`].
/// A proof is accepted exactly when the leaves and siblings lead to `root`, which, for a proof
/// built against a tree, is the canonical multiproof of its leaves in that tree, or in the tree of
/// its nodes at some level, with the same root: `n` comes from public data, and this is why.
pub fn multiproof_outcome(
    root: &Digest,
    n: u64,
    leaves: &[(u64, Digest)],
    siblings: &[Digest],
) -> Result<(), MultiproofError> {
    if !n.is_power_of_two() {
        return Err(MultiproofError::TreeSize);
    }
    if leaves.is_empty() {
        return Err(MultiproofError::NoLeaves);
    }
    for (position, &(index, _)) in leaves.iter().enumerate() {
        if position > 0 && index <= leaves[position - 1].0 {
            return Err(MultiproofError::Unordered { position });
        }
        if index >= n {
            return Err(MultiproofError::OutOfRange { position });
        }
    }
    let indices: Vec<u64> = leaves.iter().map(|&(i, _)| i).collect();
    let expected = canonical_count(n, &indices);
    if siblings.len() < expected {
        return Err(MultiproofError::MissingSiblings {
            expected,
            found: siblings.len(),
        });
    }
    if siblings.len() > expected {
        return Err(MultiproofError::ExtraSiblings {
            expected,
            found: siblings.len(),
        });
    }
    if root_from_multiproof(n, leaves, siblings) == Some(*root) {
        Ok(())
    } else {
        Err(MultiproofError::RootMismatch)
    }
}

// ----- RFC 6962 trees (RFC 6962 §2.1) -------------------------------------------------------------

/// `MTH` of RFC 6962 §2.1: SHA-256 of the empty string for no leaf, `SHA-256(0x00 ‖ d)` for one,
/// and `SHA-256(0x01 ‖ MTH(D[0:k]) ‖ MTH(D[k:n]))` with `k` the largest power of two below `n`.
pub fn mth(leaves: &[Vec<u8>]) -> Digest {
    match leaves.len() {
        0 => sha256(&[]),
        1 => sha256(&[&[0x00][..], &leaves[0]].concat()),
        n => {
            let k = split(n as u64) as usize;
            sha256(&[&[0x01][..], &mth(&leaves[..k]), &mth(&leaves[k..])].concat())
        }
    }
}

/// `PATH(m, D[n])` of RFC 6962 §2.1.1: the inclusion proof of leaf `m`, from the leaf upward.
pub fn path(m: usize, leaves: &[Vec<u8>]) -> Vec<Digest> {
    let n = leaves.len();
    if n <= 1 {
        return Vec::new();
    }
    let k = split(n as u64) as usize;
    if m < k {
        let mut proof = path(m, &leaves[..k]);
        proof.push(mth(&leaves[k..]));
        proof
    } else {
        let mut proof = path(m - k, &leaves[k..]);
        proof.push(mth(&leaves[..k]));
        proof
    }
}

/// The length of `PATH(m, D[n])`, from `m` and `n` alone.
pub fn path_len(m: u64, n: u64) -> usize {
    if n <= 1 {
        return 0;
    }
    let k = split(n);
    1 + if m < k {
        path_len(m, k)
    } else {
        path_len(m - k, n - k)
    }
}

/// The largest power of two smaller than `n`, for `n > 1`.
fn split(n: u64) -> u64 {
    1 << (63 - (n - 1).leading_zeros())
}

/// The root that `proof` leads to for leaf `m` of a tree of `n` leaves, from RFC 6962's
/// recursive definition read backwards: `PATH(m, D[n])` ends with the hash of the subtree on the
/// other side of the top split. `None` when the proof's length does not fit.
pub fn root_from_path(m: u64, n: u64, leaf: &Digest, proof: &[Digest]) -> Option<Digest> {
    if n <= 1 {
        return proof.is_empty().then_some(*leaf);
    }
    let (sibling, below) = proof.split_last()?;
    let k = split(n);
    let (left, right) = if m < k {
        (root_from_path(m, k, leaf, below)?, *sibling)
    } else {
        (*sibling, root_from_path(m - k, n - k, leaf, below)?)
    };
    Some(sha256(&[&[0x01][..], &left, &right].concat()))
}

/// What `verify_inclusion(index, size, leaf_hash, proof, root)` must return, in RFC 9162's order
/// of failures: a proof of the right length is accepted when it leads to `root` by
/// [`root_from_path`].
///
/// A proof can lead to the same root for two tree sizes, such as leaf 0 of trees of 3 and 4
/// leaves, whose paths have the same shape: RFC 9162 binds a root to its size only through the
/// signed tree head, here the manifest's `entry_count`.
pub fn inclusion_outcome(
    index: u64,
    size: u64,
    leaf_hash: &Digest,
    proof: &[Digest],
    root: &Digest,
) -> Result<(), InclusionError> {
    if index >= size {
        return Err(InclusionError::IndexBeyondSize);
    }
    let needed = path_len(index, size);
    if proof.len() > needed {
        return Err(InclusionError::ProofTooLong);
    }
    if proof.len() < needed {
        return Err(InclusionError::ProofTooShort);
    }
    if root_from_path(index, size, leaf_hash, proof) == Some(*root) {
        Ok(())
    } else {
        Err(InclusionError::RootMismatch)
    }
}
