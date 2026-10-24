// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC 9162's SHA-256 Merkle tree over a period's entries (RFC-0060 'The tree over a period's
//! entries'): leaves `SHA-256(0x00 ‖ TCE(entry))`, nodes `SHA-256(0x01 ‖ left ‖ right)`, the
//! empty tree `SHA-256("")`, and inclusion proofs generated as RFC 9162 §2.1.3.1 and verified as
//! §2.1.3.2. All of it is public.
//!
//! The tree of `n > 1` leaves splits at `k`, the largest power of two below `n`: the left
//! subtree holds leaves `0..k` and the right one `k..n`.

use crate::log::DigestLogEntry;
use core::fmt;
use sha2::{Digest, Sha256};
use tessera_codec::encode;

/// A node of the tree: a SHA-256 digest.
pub type Hash = [u8; 32];

/// An inclusion proof on the wire (RFC-0060 'Inclusion proofs'): the audit path's digests from
/// the leaf upward, TCE-encoded as a `Vec<[u8; 32]>`. It carries no index and no size: those come
/// from the entry and the manifest.
pub type InclusionProof = Vec<Hash>;

/// The longest audit path, that of a tree of `2^64` leaves.
pub const MAX_PROOF_LEN: usize = 64;

/// `SHA-256(0x00 ‖ TCE(entry))`, the leaf of `entry`.
pub fn leaf_hash(entry: &DigestLogEntry) -> Hash {
    leaf_hash_bytes(&encode(entry))
}

/// `SHA-256(0x00 ‖ bytes)`, the leaf of an entry's TCE encoding.
pub fn leaf_hash_bytes(bytes: &[u8]) -> Hash {
    let mut hash = Sha256::new();
    hash.update([0x00]);
    hash.update(bytes);
    hash.finalize().into()
}

/// `SHA-256(0x01 ‖ left ‖ right)`, an interior node.
pub fn node_hash(left: &Hash, right: &Hash) -> Hash {
    let mut hash = Sha256::new();
    hash.update([0x01]);
    hash.update(left);
    hash.update(right);
    hash.finalize().into()
}

/// The largest power of two strictly below `n`, for `n > 1`.
fn split(n: usize) -> usize {
    debug_assert!(n > 1);
    1 << (usize::BITS - 1 - (n - 1).leading_zeros())
}

/// The Merkle tree hash of `leaves` (RFC 9162 §2.1.1): `SHA-256("")` for none.
pub fn root(leaves: &[Hash]) -> Hash {
    match leaves.len() {
        0 => Sha256::digest([]).into(),
        1 => leaves[0],
        n => {
            let k = split(n);
            node_hash(&root(&leaves[..k]), &root(&leaves[k..]))
        }
    }
}

/// The inclusion proof of leaf `index` among `leaves` (RFC 9162 §2.1.3.1, `PATH`), from the leaf
/// upward, or `None` when `index` is no leaf.
pub fn inclusion_proof(leaves: &[Hash], index: usize) -> Option<Vec<Hash>> {
    if index >= leaves.len() {
        return None;
    }
    let mut proof = Vec::new();
    path(leaves, index, &mut proof);
    Some(proof)
}

fn path(leaves: &[Hash], index: usize, out: &mut Vec<Hash>) {
    if leaves.len() <= 1 {
        return;
    }
    let k = split(leaves.len());
    if index < k {
        path(&leaves[..k], index, out);
        out.push(root(&leaves[k..]));
    } else {
        path(&leaves[k..], index - k, out);
        out.push(root(&leaves[..k]));
    }
}

/// Why an inclusion proof failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InclusionError {
    /// The leaf index is not below the tree size.
    Index,
    /// The proof holds more digests than the leaf's path.
    TooLong,
    /// The proof holds fewer digests than the leaf's path.
    TooShort,
    /// The path reaches another root.
    Root,
}

impl fmt::Display for InclusionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Index => "the leaf index is not below the tree size",
            Self::TooLong => "the proof is longer than the leaf's path",
            Self::TooShort => "the proof is shorter than the leaf's path",
            Self::Root => "the proof leads to another root",
        })
    }
}

impl std::error::Error for InclusionError {}

/// Verifies that `leaf` is leaf `index` of the tree of `size` leaves whose root is `root`, by
/// `proof` (RFC 9162 §2.1.3.2). The index comes from the entry and the size and the root from the
/// signed manifest, never from the proof (RFC-0060).
///
/// # Errors
/// [`InclusionError::Index`] first, then [`InclusionError::TooLong`] as soon as the proof outruns
/// the path, then [`InclusionError::TooShort`] or [`InclusionError::Root`].
pub fn verify_inclusion(
    leaf: &Hash,
    index: u64,
    size: u64,
    proof: &[Hash],
    root: &Hash,
) -> Result<(), InclusionError> {
    if index >= size {
        return Err(InclusionError::Index);
    }
    let (mut fnode, mut snode) = (index, size - 1);
    let mut r = *leaf;
    for p in proof {
        if snode == 0 {
            return Err(InclusionError::TooLong);
        }
        if fnode & 1 == 1 || fnode == snode {
            r = node_hash(p, &r);
            while fnode & 1 == 0 && fnode != 0 {
                fnode >>= 1;
                snode >>= 1;
            }
        } else {
            r = node_hash(&r, p);
        }
        fnode >>= 1;
        snode >>= 1;
    }
    if snode != 0 {
        return Err(InclusionError::TooShort);
    }
    if r != *root {
        return Err(InclusionError::Root);
    }
    Ok(())
}
