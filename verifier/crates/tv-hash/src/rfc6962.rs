// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The SHA-256 Merkle tree of RFC 6962 over the recorder's digest log (RFC-0015 'Log format'):
//! leaves `SHA-256(0x00 ‖ entry)`, nodes `SHA-256(0x01 ‖ left ‖ right)`, and inclusion proofs
//! checked by the algorithm of RFC 9162 §2.1.3.2, for trees of any size.
//!
//! In RFC 9162's tree (§2.1.1) the left subtree of `n > 1` leaves holds the largest power of two
//! smaller than `n`; an inclusion proof lists the siblings of the leaf's path from the leaf up.
//!
//! The tree size must come from the signed manifest (`entry_count`), with the root it signs, and
//! the leaf index from the entry (`seq − first_seq`), never from the proof: one proof can verify
//! against one root for several pairs of a leaf index and a tree size whose paths have the same
//! shape. Leaf 0 of a tree of 4 leaves, whose proof is `[d1, H(d2, d3)]`, also verifies as leaf 0
//! of a tree of 3 leaves; leaf 1 of a tree of 2 leaves, whose proof is `[d0]`, also verifies as
//! leaf 2 of a tree of 3 leaves.

use core::fmt;

use crate::Digest;
use crate::sha256::Sha256;

/// The hash of a leaf, `SHA-256(0x00 ‖ entry)`, for an entry's TCE encoding.
pub fn leaf_hash(entry: &[u8]) -> Digest {
    let mut hasher = Sha256::new();
    hasher.update(&[0x00]).update(entry);
    hasher.finalize()
}

/// The hash of an interior node, `SHA-256(0x01 ‖ left ‖ right)`.
pub fn node_hash(left: &Digest, right: &Digest) -> Digest {
    let mut hasher = Sha256::new();
    hasher.update(&[0x01]).update(left).update(right);
    hasher.finalize()
}

/// Why an inclusion proof failed, by the step of RFC 9162 §2.1.3.2 that rejected it. Every
/// failure is a rejection of the proof (`Reject(Anchor)` in RFC-0015 'Verifier checks').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InclusionError {
    /// Step 1: the leaf index is not below the tree size, which includes every proof against an
    /// empty tree.
    IndexBeyondSize,
    /// Step 4a: the path reached the root with proof digests left over.
    ProofTooLong,
    /// Step 5: the proof ended before the path reached the root.
    ProofTooShort,
    /// Step 5: the path reached the root, but its hash is not the root hash.
    RootMismatch,
}

impl fmt::Display for InclusionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            InclusionError::IndexBeyondSize => "the leaf index is not below the tree size",
            InclusionError::ProofTooLong => "the inclusion proof has digests past the root",
            InclusionError::ProofTooShort => "the inclusion proof ends below the root",
            InclusionError::RootMismatch => "the inclusion proof leads to another root",
        })
    }
}

impl std::error::Error for InclusionError {}

/// Checks that the leaf whose hash is `leaf_hash` sits at `leaf_index` in the tree of
/// `tree_size` leaves whose root is `root`, given the inclusion proof `proof`, by the algorithm
/// of RFC 9162 §2.1.3.2:
///
/// 1. `leaf_index ≥ tree_size` fails ([`IndexBeyondSize`](InclusionError::IndexBeyondSize)).
/// 2. `fn = leaf_index`, `sn = tree_size − 1`.
/// 3. `r = leaf_hash`.
/// 4. For each digest `p` of the proof, in order:
///    1. if `sn = 0`, fail ([`ProofTooLong`](InclusionError::ProofTooLong));
///    2. if `fn` is odd or `fn = sn`, `r = node(p, r)`, and if `fn` is even, shift `fn` and `sn`
///       right until `fn` is odd or zero; otherwise `r = node(r, p)`;
///    3. shift `fn` and `sn` right once.
/// 5. `sn ≠ 0` fails ([`ProofTooShort`](InclusionError::ProofTooShort)); `r ≠ root` fails
///    ([`RootMismatch`](InclusionError::RootMismatch)).
///
/// The work is one hash per proof digest, at most 64 of which can be used.
pub fn verify_inclusion(
    leaf_index: u64,
    tree_size: u64,
    leaf_hash: &Digest,
    proof: &[Digest],
    root: &Digest,
) -> Result<(), InclusionError> {
    // 1.
    if leaf_index >= tree_size {
        return Err(InclusionError::IndexBeyondSize);
    }
    // 2. `tree_size > leaf_index ≥ 0`, so the subtraction does not wrap.
    let mut f_n = leaf_index;
    let mut s_n = tree_size.wrapping_sub(1);
    // 3.
    let mut r = *leaf_hash;
    // 4.
    for p in proof {
        // 4a.
        if s_n == 0 {
            return Err(InclusionError::ProofTooLong);
        }
        // 4b.
        if f_n & 1 == 1 || f_n == s_n {
            r = node_hash(p, &r);
            // `f_n` halves at each step and the loop ends at zero: at most 64 steps.
            while f_n & 1 == 0 && f_n != 0 {
                f_n >>= 1;
                s_n >>= 1;
            }
        } else {
            r = node_hash(&r, p);
        }
        // 4c.
        f_n >>= 1;
        s_n >>= 1;
    }
    // 5.
    if s_n != 0 {
        return Err(InclusionError::ProofTooShort);
    }
    if r != *root {
        return Err(InclusionError::RootMismatch);
    }
    Ok(())
}
