// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Canonical multiproofs (RFC-0004 'Merkle trees'): honest openings of random trees of 2 to 2^12
//! leaves, and of sparse trees of up to 2^63, verify, and each check of `verify_multiproof` has a
//! test that violates exactly that check and asserts its exact error. The trees and the canonical proofs come from the test
//! support's oracle, built level by level with the test kit's reference BLAKE3.

mod support;

use support::{Rng, Tree, canonical_count, multiproof_outcome};
use tv_hash::Digest;
use tv_hash::merkle::{MultiproofError, node, verify_multiproof, verify_multiproof_with_cancel};

/// `count` distinct sorted indices below `n`.
fn indices(rng: &mut Rng, n: u64, count: usize) -> Vec<u64> {
    let mut set = std::collections::BTreeSet::new();
    while set.len() < count.min(n as usize) {
        set.insert(rng.below(n));
    }
    set.into_iter().collect()
}

#[test]
fn the_node_hash_is_the_merkle_node_domain() {
    let (left, right) = ([7; 32], [9; 32]);
    assert_eq!(node(&left, &right), support::reference_node(&left, &right));
    assert_ne!(node(&left, &right), node(&right, &left));
}

#[test]
fn honest_openings_of_random_trees_verify() {
    let mut rng = Rng::new(0x3e4c1e);
    for log_n in 0..=12 {
        let tree = Tree::random(log_n, &mut rng);
        let n = tree.n();
        let mut sets: Vec<Vec<u64>> = vec![vec![0], vec![n - 1], (0..n).collect()];
        for count in [1, 2, 3, 5, 17, 64, 155, 193] {
            sets.push(indices(&mut rng, n, count));
        }
        for set in sets {
            let siblings = tree.multiproof(&set);
            assert_eq!(siblings.len(), canonical_count(n, &set));
            assert_eq!(
                verify_multiproof(&tree.root(), n, &tree.open(&set), &siblings),
                Ok(()),
                "n = {n}, {} leaves",
                set.len()
            );
        }
    }
}

#[test]
fn every_single_leaf_of_a_tree_opens_alone() {
    let mut rng = Rng::new(0x51e);
    let tree = Tree::random(6, &mut rng);
    for i in 0..tree.n() {
        let siblings = tree.multiproof(&[i]);
        assert_eq!(siblings.len(), 6);
        assert_eq!(
            verify_multiproof(&tree.root(), 64, &tree.open(&[i]), &siblings),
            Ok(())
        );
    }
}

/// `count` distinct sorted indices below `n`, for trees too large to build: both ends, a pair of
/// neighbours across a subtree boundary, then random indices.
fn sparse_indices(rng: &mut Rng, n: u64, count: usize) -> Vec<u64> {
    let mut set = std::collections::BTreeSet::from([0, n - 1]);
    if n > 4 {
        let middle = (n / 2) | 1;
        set.extend([middle, middle + 1]);
    }
    while (set.len() as u64) < (count as u64).min(n) {
        set.insert(rng.below(n));
    }
    set.into_iter().take(count).collect()
}

/// Trees of up to 2^63 leaves, the largest a `u64` size can count, and too large to build: the
/// opened leaves and the siblings their canonical multiproof names hold random digests, and the
/// oracle recomputes the root from those alone, level by level. The check accepts exactly that
/// root at every depth, and uses every digest: changing a leaf or a sibling is `RootMismatch`.
#[test]
fn sparse_openings_of_trees_up_to_2_63_leaves_verify() {
    let mut rng = Rng::new(0x5ba45e);
    for log_n in [1u32, 2, 3, 10, 31, 32, 33, 62, 63] {
        let n = 1u64 << log_n;
        for count in [1usize, 2, 3, 17, 193] {
            let set = sparse_indices(&mut rng, n, count);
            let leaves: Vec<(u64, Digest)> = set.iter().map(|&i| (i, rng.digest())).collect();
            let siblings: Vec<Digest> = (0..canonical_count(n, &set))
                .map(|_| rng.digest())
                .collect();
            let root = support::root_from_multiproof(n, &leaves, &siblings).unwrap();
            let at = format!("n = 2^{log_n}, {} leaves", set.len());
            assert_eq!(
                verify_multiproof(&root, n, &leaves, &siblings),
                Ok(()),
                "{at}"
            );
            if !siblings.is_empty() {
                let last = siblings.len() - 1;
                for position in [0, last / 2, last, rng.index(siblings.len())] {
                    let mut changed = siblings.clone();
                    changed[position][9] ^= 0x40;
                    assert_eq!(
                        verify_multiproof(&root, n, &leaves, &changed),
                        Err(MultiproofError::RootMismatch),
                        "{at}, sibling {position} changed"
                    );
                }
            }
            for position in [0, leaves.len() - 1] {
                let mut changed = leaves.clone();
                changed[position].1[30] ^= 1;
                assert_eq!(
                    verify_multiproof(&root, n, &changed, &siblings),
                    Err(MultiproofError::RootMismatch),
                    "{at}, leaf {position} changed"
                );
            }
        }
    }
}

/// Fixtures for the one-check violations: a tree, its root, an opening of several leaves and
/// the opening's canonical proof.
struct Case {
    tree: Tree,
    leaves: Vec<(u64, Digest)>,
    siblings: Vec<Digest>,
}

fn cases() -> Vec<Case> {
    let mut rng = Rng::new(0xadc5a);
    let mut out = Vec::new();
    for (log_n, count) in [(1, 1), (2, 2), (3, 3), (5, 4), (8, 9), (10, 40)] {
        let tree = Tree::random(log_n, &mut rng);
        let set = indices(&mut rng, tree.n(), count);
        out.push(Case {
            leaves: tree.open(&set),
            siblings: tree.multiproof(&set),
            tree,
        });
    }
    out
}

#[test]
fn check_1_a_tree_size_that_is_not_a_power_of_two() {
    for case in cases() {
        let n = case.tree.n();
        let root = case.tree.root();
        for bad in [0, 3, n + 1, 3 * n, u64::MAX] {
            assert_eq!(
                verify_multiproof(&root, bad, &case.leaves, &case.siblings),
                Err(MultiproofError::TreeSize),
                "n = {bad}"
            );
        }
    }
}

#[test]
fn check_2_no_opened_leaf() {
    for case in cases() {
        let root = case.tree.root();
        for siblings in [&case.siblings[..], &[]] {
            assert_eq!(
                verify_multiproof(&root, case.tree.n(), &[], siblings),
                Err(MultiproofError::NoLeaves)
            );
        }
    }
}

#[test]
fn check_3_indices_unsorted_or_repeated() {
    for case in cases() {
        let (n, root) = (case.tree.n(), case.tree.root());
        let len = case.leaves.len();
        // A repeated leaf, with its true digest, at every position.
        for at in 0..len {
            let mut leaves = case.leaves.clone();
            leaves.insert(at + 1, leaves[at]);
            assert_eq!(
                verify_multiproof(&root, n, &leaves, &case.siblings),
                Err(MultiproofError::Unordered { position: at + 1 })
            );
        }
        // Two neighbours swapped.
        for at in 1..len {
            let mut leaves = case.leaves.clone();
            leaves.swap(at - 1, at);
            assert_eq!(
                verify_multiproof(&root, n, &leaves, &case.siblings),
                Err(MultiproofError::Unordered { position: at })
            );
        }
    }
}

#[test]
fn check_3_an_index_outside_the_tree() {
    for case in cases() {
        let (n, root) = (case.tree.n(), case.tree.root());
        let last = case.leaves.len() - 1;
        for index in [n, n + 1, 2 * n, u64::MAX] {
            let mut leaves = case.leaves.clone();
            leaves[last].0 = index;
            assert_eq!(
                verify_multiproof(&root, n, &leaves, &case.siblings),
                Err(MultiproofError::OutOfRange { position: last })
            );
        }
        // The order is checked before the range, leaf by leaf: an index past `n` followed by a
        // smaller one is out of range at its own position.
        let mut leaves = case.leaves.clone();
        leaves[0].0 = n;
        assert_eq!(
            verify_multiproof(&root, n, &leaves, &case.siblings),
            Err(MultiproofError::OutOfRange { position: 0 })
        );
    }
}

#[test]
fn check_4_missing_or_extra_siblings() {
    let mut rng = Rng::new(0xe47a);
    for case in cases() {
        let (n, root) = (case.tree.n(), case.tree.root());
        let expected = case.siblings.len();
        for at in 0..expected {
            let mut fewer = case.siblings.clone();
            fewer.remove(at);
            assert_eq!(
                verify_multiproof(&root, n, &case.leaves, &fewer),
                Err(MultiproofError::MissingSiblings {
                    expected,
                    found: expected - 1
                })
            );
        }
        for at in 0..=expected {
            for extra in [rng.digest(), root, case.leaves[0].1] {
                let mut more = case.siblings.clone();
                more.insert(at, extra);
                assert_eq!(
                    verify_multiproof(&root, n, &case.leaves, &more),
                    Err(MultiproofError::ExtraSiblings {
                        expected,
                        found: expected + 1
                    })
                );
            }
        }
        assert_eq!(
            verify_multiproof(&root, n, &case.leaves, &[]),
            if expected == 0 {
                Ok(())
            } else {
                Err(MultiproofError::MissingSiblings { expected, found: 0 })
            }
        );
    }
}

#[test]
fn check_5_a_wrong_sibling_leaf_or_root_or_misordered_siblings() {
    for case in cases() {
        let (n, root) = (case.tree.n(), case.tree.root());
        for at in 0..case.siblings.len() {
            for bit in [0, 100, 255] {
                let mut changed = case.siblings.clone();
                changed[at][bit / 8] ^= 1 << (bit % 8);
                assert_eq!(
                    verify_multiproof(&root, n, &case.leaves, &changed),
                    Err(MultiproofError::RootMismatch)
                );
            }
        }
        for at in 1..case.siblings.len() {
            let mut swapped = case.siblings.clone();
            swapped.swap(at - 1, at);
            assert_eq!(
                verify_multiproof(&root, n, &case.leaves, &swapped),
                Err(MultiproofError::RootMismatch),
                "siblings {} and {at} swapped",
                at - 1
            );
        }
        if case.siblings.len() > 2 {
            let mut reversed = case.siblings.clone();
            reversed.reverse();
            assert_eq!(
                verify_multiproof(&root, n, &case.leaves, &reversed),
                Err(MultiproofError::RootMismatch)
            );
        }
        for at in 0..case.leaves.len() {
            let mut leaves = case.leaves.clone();
            leaves[at].1[17] ^= 0x20;
            assert_eq!(
                verify_multiproof(&root, n, &leaves, &case.siblings),
                Err(MultiproofError::RootMismatch)
            );
            // The digest of another leaf of the tree, at this index.
            let other = (leaves[at].0 + 1) % n;
            leaves[at].1 = case.tree.levels[0][other as usize];
            if other != leaves[at].0 {
                assert_eq!(
                    verify_multiproof(&root, n, &leaves, &case.siblings),
                    Err(MultiproofError::RootMismatch)
                );
            }
        }
        let mut other_root = root;
        other_root[0] ^= 1;
        assert_eq!(
            verify_multiproof(&other_root, n, &case.leaves, &case.siblings),
            Err(MultiproofError::RootMismatch)
        );
    }
}

/// Opened leaves with other indices of the same tree, the proof unchanged: the counts may fit,
/// but the root does not; or the counts do not. The oracle predicts which.
#[test]
fn shifted_indices_are_rejected() {
    for case in cases() {
        let (n, root) = (case.tree.n(), case.tree.root());
        for shift in 1..n.min(9) {
            let leaves: Vec<(u64, Digest)> = case
                .leaves
                .iter()
                .map(|&(i, d)| ((i + shift) % n, d))
                .collect();
            let mut sorted = leaves.clone();
            sorted.sort_by_key(|l| l.0);
            sorted.dedup_by_key(|l| l.0);
            let outcome = verify_multiproof(&root, n, &sorted, &case.siblings);
            assert!(outcome.is_err());
            assert_eq!(
                outcome,
                multiproof_outcome(&root, n, &sorted, &case.siblings)
            );
        }
    }
}

/// RFC-0004: "The nodes at level k are the leaves of a tree of n / 2^k leaves with the same root,
/// so a prover choosing n could open internal nodes as leaves." The check accepts exactly that
/// when given the smaller `n`, which is why `n` must come from public data; with the true `n` the
/// same opening is rejected.
#[test]
fn internal_nodes_open_as_leaves_only_under_a_smaller_tree_size() {
    let mut rng = Rng::new(0x1e7e1);
    let tree = Tree::random(6, &mut rng);
    for level in 1..=6usize {
        let small_n = 64u64 >> level;
        let set = indices(&mut rng, small_n, 3);
        let nodes: Vec<(u64, Digest)> = set
            .iter()
            .map(|&i| (i, tree.levels[level][i as usize]))
            .collect();
        let siblings = support::canonical_multiproof(&tree.levels[level..], &set);
        assert_eq!(
            verify_multiproof(&tree.root(), small_n, &nodes, &siblings),
            Ok(()),
            "level {level} nodes as the leaves of {small_n}"
        );
        let honest_n = verify_multiproof(&tree.root(), 64, &nodes, &siblings);
        assert!(honest_n.is_err());
        assert_eq!(
            honest_n,
            multiproof_outcome(&tree.root(), 64, &nodes, &siblings)
        );
    }
}

#[test]
fn a_cancellable_opening_of_an_independent_tree_preserves_its_result() {
    let mut rng = Rng::new(0x182_004);
    let tree = Tree::random(13, &mut rng);
    let set: Vec<_> = (0..tree.n()).collect();
    let leaves = tree.open(&set);
    let siblings = tree.multiproof(&set);
    let mut polls = 0;
    assert_eq!(
        verify_multiproof_with_cancel(&tree.root(), tree.n(), &leaves, &siblings, || {
            polls += 1;
            false
        }),
        Ok(())
    );
    assert!(
        polls > 4,
        "the traversal must check beyond its entry and exit"
    );
    for stop in 1..=polls {
        let mut seen = 0;
        assert_eq!(
            verify_multiproof_with_cancel(&tree.root(), tree.n(), &leaves, &siblings, || {
                seen += 1;
                seen == stop
            }),
            Err(MultiproofError::Cancelled)
        );
        assert_eq!(seen, stop);
    }
}
