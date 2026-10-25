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
use crate::sha256::{Sha256, sha256};

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

/// The RFC 9162 Merkle tree hash (§2.1.1) of a sequence of leaves, computed as the leaves arrive,
/// in constant memory: the root a manifest's `entries_root` signs over its period's entries
/// (RFC-0060 'The tree over a period's entries').
///
/// RFC 9162 defines the tree recursively: the empty tree's hash is `SHA-256("")`, one leaf's is
/// its leaf hash, and `n > 1` leaves split at the largest power of two `k < n` into
/// `node(MTH(D[0..k]), MTH(D[k..n]))`. This builder takes another route to the same value. After
/// `c` leaves it holds one perfect subtree per set bit of `c`, the largest first, as the binary
/// expansion of `c` cuts the leaves into consecutive power-of-two runs; pushing a leaf merges the
/// runs it completes, as adding 1 carries. [`root`](Self::root) then folds the runs from the
/// smallest, right to left: every left subtree of RFC 9162's split is one of the runs or a merge
/// of them, so the fold builds the recursive tree. It never hashes a leaf twice and needs no list
/// of leaves.
#[derive(Clone)]
pub struct TreeHash {
    /// The runs' roots, the largest first: `runs[i]` covers `2^b_i` leaves for the `i`-th set bit
    /// `b_i` of `count` from the top. At most 64, one per bit.
    runs: [Digest; 64],
    /// How many entries of `runs` are in use: the set bits of `count`.
    len: usize,
    /// The leaves pushed so far.
    count: u64,
}

impl TreeHash {
    /// The tree of no leaves.
    pub const fn new() -> Self {
        Self {
            runs: [[0; 32]; 64],
            len: 0,
            count: 0,
        }
    }

    /// The number of leaves pushed.
    pub const fn count(&self) -> u64 {
        self.count
    }

    /// Appends the leaf whose hash is `leaf`, [`leaf_hash`] of an entry.
    ///
    /// # Errors
    ///
    /// [`TreeFull`] when the tree already holds `2^64 − 1` leaves, the most a `u64` counts; the
    /// tree is then unchanged.
    pub fn push(&mut self, leaf: &Digest) -> Result<(), TreeFull> {
        let count = self.count.checked_add(1).ok_or(TreeFull)?;
        // Each trailing 1 bit of the old count is a run of the same size as the subtree being
        // built: merge it on the left, as the carry clears that bit.
        let mut node = *leaf;
        let mut bits = self.count;
        while bits & 1 == 1 {
            let Some(left_at) = self.len.checked_sub(1) else {
                break;
            };
            let Some(left) = self.runs.get(left_at) else {
                break;
            };
            node = node_hash(left, &node);
            self.len = left_at;
            bits >>= 1;
        }
        // At most one run per set bit of a u64, so the slot exists.
        let slot = self.runs.get_mut(self.len).ok_or(TreeFull)?;
        *slot = node;
        self.len = self.len.saturating_add(1);
        self.count = count;
        Ok(())
    }

    /// The tree hash of the leaves pushed so far: `SHA-256("")` for none.
    pub fn root(&self) -> Digest {
        let mut runs = self.runs.iter().take(self.len).rev();
        let Some(last) = runs.next() else {
            return sha256(&[]);
        };
        runs.fold(*last, |right, left| node_hash(left, &right))
    }
}

impl Default for TreeHash {
    fn default() -> Self {
        Self::new()
    }
}

/// Shows the count only.
impl fmt::Debug for TreeHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TreeHash")
            .field("count", &self.count)
            .finish_non_exhaustive()
    }
}

/// A [`TreeHash`] that already holds `2^64 − 1` leaves cannot take another.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TreeFull;

impl fmt::Display for TreeFull {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the tree already holds 2^64 - 1 leaves")
    }
}

impl std::error::Error for TreeFull {}

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
