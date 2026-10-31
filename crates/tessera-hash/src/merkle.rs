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

// Each additional sorted leaf diverges from its predecessor at height d, adding
// d-1 internal union nodes. S = 1 + internal_union_nodes - opened_leaves, hence
// the sibling frontier changes by d-2. Every proper nonempty prefix has at least
// one sibling. Intermediate values fit the existing tree's 2*n-1 node capacity.
fn sibling_count(leaves: usize, sorted_distinct: &[usize]) -> usize {
    let first = sorted_distinct[0];
    let last = sorted_distinct[sorted_distinct.len() - 1];
    if last - first + 1 == sorted_distinct.len() {
        // A contiguous interval's missing prefix/suffix decompose into one maximal
        // dyadic subtree per set bit. This also avoids an extra scan for all leaves.
        return (first.count_ones() + (leaves - last - 1).count_ones()) as usize;
    }
    sorted_distinct
        .windows(2)
        .fold(leaves.ilog2() as usize, |count, pair| {
            let divergence = (usize::BITS - (pair[0] ^ pair[1]).leading_zeros()) as usize;
            count + divergence - 2
        })
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
        // Validated distinct in-range indices cover every leaf: the canonical proof
        // has no siblings. Keep this public full-set case independent of tree height.
        if known.len() == self.leaves {
            return Ok(MultiProof {
                siblings: Vec::new(),
            });
        }
        let mut siblings = Vec::with_capacity(sibling_count(self.leaves, &known));
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
#[cfg(test)]
mod allocation_tests {
    use super::*;
    use std::collections::BTreeSet;

    // Independent set frontier: membership, then sorted parents; the production
    // path uses paired Vec indices. Keep this scalar oracle separate from sizing.
    fn reference(tree: &MerkleTree, queries: &[u64]) -> Vec<Digest> {
        let mut current = queries.iter().map(|&q| q as usize).collect::<BTreeSet<_>>();
        let mut expected = Vec::new();
        let (mut offset, mut width) = (0, tree.leaves);
        while width > 1 {
            for &node in &current {
                if !current.contains(&(node ^ 1)) {
                    expected.push(tree.nodes[offset + (node ^ 1)]);
                }
            }
            current = current.iter().map(|node| node / 2).collect();
            offset += width;
            width /= 2;
        }
        expected
    }

    #[test]
    fn canonical_opening_matches_scalar_frontier_for_every_small_subset() {
        for bits in 0..=4 {
            let n = 1usize << bits;
            let tree = MerkleTree::new((0..n).map(|i| [i as u8; 32]).collect()).unwrap();
            for subset in 1u64..(1u64 << n) {
                let queries = (0..n)
                    .filter(|&i| subset & (1 << i) != 0)
                    .map(|i| i as u64)
                    .collect::<Vec<_>>();
                let expected = reference(&tree, &queries);
                let actual = tree.open(&queries).unwrap();
                assert_eq!(actual.siblings, expected);
                assert_eq!(actual.siblings.capacity(), actual.siblings.len());
                let mut repeated = queries.iter().rev().copied().collect::<Vec<_>>();
                repeated.extend(&queries);
                assert_eq!(tree.open(&repeated).unwrap().siblings, expected);
            }
        }
    }
    #[test]
    fn exact_capacity_matches_sparse_clustered_and_boundary_queries() {
        let n = 1024usize;
        let tree = MerkleTree::new(
            (0..n)
                .map(|i| {
                    let mut digest = [0xA5; 32];
                    digest[..8].copy_from_slice(&(i as u64).to_le_bytes());
                    digest
                })
                .collect(),
        )
        .unwrap();
        for count in [1usize, 2, 3, 4, 7, 8, 15, 16, 31, 33, 193, 1024] {
            for queries in [
                (0..count).map(|i| i as u64).collect::<Vec<_>>(),
                (0..count).map(|i| ((i * 337 + 17) % n) as u64).collect(),
                vec![n as u64 - 1; count],
            ] {
                let actual = tree.open(&queries).unwrap();
                assert_eq!(actual.siblings, reference(&tree, &queries));
                assert_eq!(actual.siblings.capacity(), actual.siblings.len());
            }
        }
        // No large tree allocation: independently count sparse public frontiers at
        // every usize-representable power-of-two height, including the PCS maximum32.
        for bits in 1..usize::BITS {
            let leaves = 1usize << bits;
            for queries in [
                vec![0],
                vec![leaves - 1],
                vec![0, leaves - 1],
                vec![1, leaves / 2],
            ] {
                let mut sorted = queries.clone();
                sorted.sort_unstable();
                sorted.dedup();
                let mut nodes = sorted.iter().copied().collect::<BTreeSet<_>>();
                let mut expected = 0;
                for _ in 0..bits {
                    expected += nodes
                        .iter()
                        .filter(|&&index| !nodes.contains(&(index ^ 1)))
                        .count();
                    nodes = nodes.iter().map(|index| index / 2).collect();
                }
                assert_eq!(sibling_count(leaves, &sorted), expected);
            }
        }
    }
}
