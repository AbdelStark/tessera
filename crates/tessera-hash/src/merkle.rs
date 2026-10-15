// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Binary Merkle trees and canonical multiproofs (RFC-0004 'Merkle trees').

use core::fmt;

use crate::{Digest, Domain, Hasher};

/// `H("merkle-node", left ‖ right)`.
pub fn merkle_node(left: &Digest, right: &Digest) -> Digest {
    Hasher::new(Domain::MerkleNode)
        .update(left)
        .update(right)
        .finalize()
}

/// Why a tree cannot be built or opened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MerkleError {
    /// The number of leaves is zero or not a power of two.
    LeafCount {
        /// The number given.
        leaves: usize,
    },
    /// A queried index is not below the number of leaves.
    IndexOutOfRange {
        /// The index.
        index: u64,
        /// The number of leaves.
        leaves: usize,
    },
    /// No index was queried.
    EmptyQuery,
}

impl fmt::Display for MerkleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LeafCount { leaves } => {
                write!(
                    f,
                    "a Merkle tree needs a power-of-two number of leaves, not {leaves}"
                )
            }
            Self::IndexOutOfRange { index, leaves } => {
                write!(f, "leaf index {index} is outside a tree of {leaves} leaves")
            }
            Self::EmptyQuery => write!(f, "a multiproof needs at least one leaf index"),
        }
    }
}

impl std::error::Error for MerkleError {}

/// The siblings a verifier cannot compute for a set of opened leaves, in canonical order: level
/// by level from the leaves upward, left to right within a level.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MultiProof {
    /// The sibling digests, in canonical order.
    pub siblings: Vec<Digest>,
}

/// A binary Merkle tree over a power-of-two number of leaves, with the root at level `log2 n`.
#[derive(Clone, Debug)]
pub struct MerkleTree {
    /// Every level, leaves first: level `k` holds `n >> k` nodes.
    nodes: Vec<Digest>,
    leaves: usize,
}

impl MerkleTree {
    /// The tree over `leaves`, whose number must be a power of two.
    pub fn new(mut leaves: Vec<Digest>) -> Result<Self, MerkleError> {
        let n = leaves.len();
        if !n.is_power_of_two() {
            return Err(MerkleError::LeafCount { leaves: n });
        }
        leaves.reserve_exact(n - 1);
        let (mut start, mut width) = (0, n);
        while width > 1 {
            for i in (start..start + width).step_by(2) {
                let parent = merkle_node(&leaves[i], &leaves[i + 1]);
                leaves.push(parent);
            }
            start += width;
            width /= 2;
        }
        Ok(Self {
            nodes: leaves,
            leaves: n,
        })
    }

    /// The root.
    pub fn root(&self) -> Digest {
        self.nodes[self.nodes.len() - 1]
    }

    /// The number of leaves.
    pub fn leaves(&self) -> usize {
        self.leaves
    }

    /// The canonical multiproof for the leaves at `indices`, given in any order and possibly
    /// repeated.
    pub fn open(&self, indices: &[u64]) -> Result<MultiProof, MerkleError> {
        let mut known = Vec::with_capacity(indices.len());
        for &index in indices {
            match usize::try_from(index) {
                Ok(i) if i < self.leaves => known.push(i),
                _ => {
                    return Err(MerkleError::IndexOutOfRange {
                        index,
                        leaves: self.leaves,
                    });
                }
            }
        }
        if known.is_empty() {
            return Err(MerkleError::EmptyQuery);
        }
        known.sort_unstable();
        known.dedup();
        let mut siblings = Vec::new();
        let (mut start, mut width) = (0, self.leaves);
        while width > 1 {
            let mut parents = Vec::with_capacity(known.len());
            let mut i = 0;
            while i < known.len() {
                let node = known[i];
                if node % 2 == 0 && known.get(i + 1) == Some(&(node + 1)) {
                    // Both children are known: the parent needs no sibling.
                    i += 2;
                } else {
                    siblings.push(self.nodes[start + (node ^ 1)]);
                    i += 1;
                }
                parents.push(node / 2);
            }
            known = parents;
            start += width;
            width /= 2;
        }
        Ok(MultiProof { siblings })
    }
}

/// Whether `proof` shows that the `leaves`, as `(index, digest)` pairs, belong to the tree of `n`
/// leaves with root `root`.
///
/// The leaves must be non-empty, in strictly increasing index order and below `n`, a power of
/// two. The proof must hold exactly the siblings the canonical order calls for: a missing, extra
/// or reordered digest rejects.
///
/// `n` must come from public data, never from the prover: the nodes at level `k` of a tree are
/// the leaves of a tree of `n >> k` leaves with the same root, so a prover choosing `n` could
/// open internal nodes as leaves.
pub fn verify_multiproof(
    root: &Digest,
    n: u64,
    leaves: &[(u64, Digest)],
    proof: &MultiProof,
) -> bool {
    let increasing = leaves.windows(2).all(|pair| pair[0].0 < pair[1].0);
    let in_range = leaves.last().is_some_and(|&(last, _)| last < n);
    if !n.is_power_of_two() || !increasing || !in_range {
        return false;
    }
    let mut siblings = proof.siblings.iter();
    let mut known = leaves.to_vec();
    let mut width = n;
    while width > 1 {
        let mut parents = Vec::with_capacity(known.len());
        let mut i = 0;
        while i < known.len() {
            let (node, digest) = known[i];
            let pair = if node % 2 == 0 {
                match known.get(i + 1) {
                    Some(&(next, right)) if next == node + 1 => {
                        i += 1;
                        Some((digest, right))
                    }
                    _ => siblings.next().map(|right| (digest, *right)),
                }
            } else {
                siblings.next().map(|left| (*left, digest))
            };
            let Some((left, right)) = pair else {
                return false;
            };
            parents.push((node / 2, merkle_node(&left, &right)));
            i += 1;
        }
        known = parents;
        width /= 2;
    }
    siblings.next().is_none() && known == [(0, *root)]
}
