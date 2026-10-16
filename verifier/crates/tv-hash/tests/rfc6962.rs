// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The RFC 6962 tree of the recorder log (RFC-0015 'Log format'): the 98 golden inclusion proofs
//! of `spec/vectors/rfc6962/inclusion/`, each accepted or rejected as its `wantErr` says and
//! rejected by the step the oracle predicts; generated trees of every size up to 140 leaves, each
//! leaf's proof built from RFC 6962's recursive definitions; and proofs that each violate exactly
//! one step of RFC 9162 §2.1.3.2.

mod support;

use support::{Rng, base64, files_under, inclusion_outcome, mth, number, path, path_len, text};
use tv_hash::Digest;
use tv_hash::rfc6962::{InclusionError, leaf_hash, node_hash, verify_inclusion};
use tv_hash::sha256::sha256;
use tv_testkit::{Json, hex};

/// The leaves of the tree the golden vectors use, the test data of RFC 6962 implementations.
fn golden_entries() -> Vec<Vec<u8>> {
    [
        "",
        "00",
        "10",
        "2021",
        "3031",
        "40414243",
        "5051525354555657",
        "606162636465666768696a6b6c6d6e6f",
    ]
    .iter()
    .map(|h| hex(h))
    .collect()
}

#[test]
fn leaf_and_node_hashes_carry_their_prefixes() {
    assert_eq!(leaf_hash(b"abc"), sha256(b"\x00abc"));
    let (left, right) = ([0x11; 32], [0x22; 32]);
    assert_eq!(
        node_hash(&left, &right),
        sha256(&[&[0x01][..], &left, &right].concat())
    );
    // An empty leaf is not the empty tree: SHA-256(0x00) against SHA-256 of nothing.
    assert_ne!(leaf_hash(b""), sha256(b""));
}

/// A golden case's components, or `None` when one of them is not 32 bytes long and so cannot be
/// a digest: a TCE `Digest` is exactly 32 bytes, so such a proof is rejected when it is decoded
/// and never reaches the check.
#[allow(clippy::type_complexity)]
fn digests_of(case: &Json) -> Option<(Digest, Digest, Vec<Digest>)> {
    let leaf = base64(text(case, "leafHash")).try_into().ok()?;
    let root = base64(text(case, "root")).try_into().ok()?;
    let proof = match case.get("proof").unwrap() {
        Json::Null => Vec::new(),
        Json::Array(items) => items
            .iter()
            .map(|item| base64(item.as_str().unwrap()).try_into().ok())
            .collect::<Option<Vec<Digest>>>()?,
        other => panic!("a proof is {other:?}"),
    };
    Some((leaf, root, proof))
}

#[test]
fn every_golden_inclusion_proof() {
    let files = files_under("rfc6962/inclusion");
    assert_eq!(files.len(), 98, "every golden case");
    let (mut accepted, mut rejected, mut not_digests) = (0, 0, 0);
    for file in &files {
        let name = file.display().to_string();
        let case = Json::parse(&std::fs::read_to_string(file).unwrap()).unwrap();
        let want_error = case.get("wantErr") == Some(&Json::Bool(true));
        let index = number(&case, "leafIdx");
        let size = number(&case, "treeSize");
        let Some((leaf, root, proof)) = digests_of(&case) else {
            assert!(want_error, "{name}: a component that is not a digest");
            not_digests += 1;
            continue;
        };
        let outcome = verify_inclusion(index, size, &leaf, &proof, &root);
        assert_eq!(outcome.is_err(), want_error, "{name}: {outcome:?}");
        // The oracle predicts the step that rejects each case.
        assert_eq!(
            outcome,
            inclusion_outcome(index, size, &leaf, &proof, &root),
            "{name}"
        );
        if want_error {
            rejected += 1;
        } else {
            accepted += 1;
        }
    }
    assert_eq!(
        (accepted, rejected + not_digests),
        (6, 92),
        "6 proofs verify and 92 are rejected"
    );
    assert_eq!(not_digests, 26, "cases rejected at the type boundary");
}

/// The roots of the golden happy paths are `MTH` of a prefix of the test entries, so the oracle's
/// trees are the golden trees.
#[test]
fn the_golden_roots_are_roots_of_the_test_entries() {
    let entries = golden_entries();
    for file in files_under("rfc6962/inclusion") {
        if !file.ends_with("happy-path.json") {
            continue;
        }
        let case = Json::parse(&std::fs::read_to_string(&file).unwrap()).unwrap();
        let size = number(&case, "treeSize") as usize;
        let index = number(&case, "leafIdx") as usize;
        let tree = &entries[..size];
        assert_eq!(base64(text(&case, "root")), mth(tree), "{}", file.display());
        assert_eq!(
            base64(text(&case, "leafHash")),
            leaf_hash(&tree[index]),
            "{}",
            file.display()
        );
        let proof: Vec<Vec<u8>> = match case.get("proof").unwrap() {
            Json::Array(items) => items.iter().map(|i| base64(i.as_str().unwrap())).collect(),
            _ => Vec::new(),
        };
        let path: Vec<Vec<u8>> = path(index, tree).iter().map(|d| d.to_vec()).collect();
        assert_eq!(proof, path, "{}", file.display());
    }
}

/// Every leaf of every tree of 1 to 140 entries, and some larger trees, proven with `PATH`.
#[test]
fn every_leaf_of_generated_trees_verifies() {
    let mut rng = Rng::new(0x6962);
    let entries: Vec<Vec<u8>> = (0..1100).map(|i| rng.bytes(i % 70)).collect();
    let mut sizes: Vec<usize> = (1..=140).collect();
    sizes.extend([255, 256, 257, 511, 1000, 1024, 1025, 1100]);
    for size in sizes {
        let tree = &entries[..size];
        let root = mth(tree);
        let indices: Vec<usize> = if size <= 140 {
            (0..size).collect()
        } else {
            vec![0, 1, size / 3, size / 2, size - 2, size - 1]
        };
        for index in indices {
            let proof = path(index, tree);
            assert_eq!(proof.len(), path_len(index as u64, size as u64));
            assert_eq!(
                verify_inclusion(
                    index as u64,
                    size as u64,
                    &leaf_hash(&tree[index]),
                    &proof,
                    &root
                ),
                Ok(()),
                "leaf {index} of {size}"
            );
        }
    }
}

/// Each change violates exactly one step of RFC 9162 §2.1.3.2 and is rejected by that step.
#[test]
fn each_violated_step_is_its_own_rejection() {
    let mut rng = Rng::new(0x9162);
    let entries: Vec<Vec<u8>> = (0..64).map(|_| rng.bytes(24)).collect();
    for (size, index) in [
        (1usize, 0usize),
        (2, 1),
        (3, 2),
        (7, 3),
        (13, 12),
        (32, 17),
        (63, 0),
    ] {
        let tree = &entries[..size];
        let root = mth(tree);
        let leaf = leaf_hash(&tree[index]);
        let proof = path(index, tree);
        let (i, n) = (index as u64, size as u64);
        assert_eq!(verify_inclusion(i, n, &leaf, &proof, &root), Ok(()));

        // Step 1: the index is not below the size.
        assert_eq!(
            verify_inclusion(n, n, &leaf, &proof, &root),
            Err(InclusionError::IndexBeyondSize)
        );
        assert_eq!(
            verify_inclusion(u64::MAX, n, &leaf, &proof, &root),
            Err(InclusionError::IndexBeyondSize)
        );
        assert_eq!(
            verify_inclusion(0, 0, &leaf, &[], &sha256(b"")),
            Err(InclusionError::IndexBeyondSize),
            "the empty tree has no leaf"
        );
        // Step 4a: a digest past the root.
        for at in 0..=proof.len() {
            let mut longer = proof.clone();
            longer.insert(at, rng.digest());
            assert_eq!(
                verify_inclusion(i, n, &leaf, &longer, &root),
                Err(InclusionError::ProofTooLong),
                "size {size}, index {index}, digest inserted at {at}"
            );
        }
        // Step 5: the proof ends below the root.
        for at in 0..proof.len() {
            let mut shorter = proof.clone();
            shorter.remove(at);
            assert_eq!(
                verify_inclusion(i, n, &leaf, &shorter, &root),
                Err(InclusionError::ProofTooShort),
                "size {size}, index {index}, digest {at} removed"
            );
        }
        // Step 5: another root, from a changed digest, leaf or root, or two digests swapped.
        for at in 0..proof.len() {
            for bit in [0, 77, 255] {
                let mut changed = proof.clone();
                changed[at][bit / 8] ^= 1 << (bit % 8);
                assert_eq!(
                    verify_inclusion(i, n, &leaf, &changed, &root),
                    Err(InclusionError::RootMismatch)
                );
            }
        }
        if proof.len() >= 2 {
            let mut swapped = proof.clone();
            swapped.swap(0, 1);
            assert_eq!(
                verify_inclusion(i, n, &leaf, &swapped, &root),
                Err(InclusionError::RootMismatch)
            );
        }
        let mut other_leaf = leaf;
        other_leaf[0] ^= 0x80;
        assert_eq!(
            verify_inclusion(i, n, &other_leaf, &proof, &root),
            Err(InclusionError::RootMismatch)
        );
        let mut other_root = root;
        other_root[31] ^= 1;
        assert_eq!(
            verify_inclusion(i, n, &leaf, &proof, &other_root),
            Err(InclusionError::RootMismatch)
        );
        // A leaf's data is not its hash: proving the entry bytes' plain SHA-256 fails.
        assert_eq!(
            verify_inclusion(i, n, &sha256(&tree[index]), &proof, &root),
            Err(InclusionError::RootMismatch)
        );
    }
}

/// The tree size must come from the signed manifest (`entry_count`): a proof checked against
/// another size is rejected by length or by root, except where the leaf's path has the same shape
/// in both trees, as for leaf 0 of trees of 3 and 4 leaves. The oracle's recursive reading of
/// `PATH` predicts every case.
#[test]
fn a_proof_checked_against_another_tree_size_depends_on_its_shape() {
    let mut rng = Rng::new(0x51e);
    let entries: Vec<Vec<u8>> = (0..48).map(|_| rng.bytes(8)).collect();
    let mut same_shape = 0;
    for size in 1..=40usize {
        for index in [0, size / 2, size - 1] {
            let tree = &entries[..size];
            let root = mth(tree);
            let leaf = leaf_hash(&tree[index]);
            let proof = path(index, tree);
            for other in 1..=48u64 {
                if other == size as u64 {
                    continue;
                }
                let outcome = verify_inclusion(index as u64, other, &leaf, &proof, &root);
                assert_eq!(
                    outcome,
                    inclusion_outcome(index as u64, other, &leaf, &proof, &root),
                    "leaf {index} of {size}, checked against {other}"
                );
                same_shape += usize::from(outcome.is_ok());
            }
        }
    }
    // Leaf 0 of 4 leaves verifies against size 3: both paths are [d1, H(d2, d3)].
    let tree = &entries[..4];
    assert_eq!(
        verify_inclusion(0, 3, &leaf_hash(&tree[0]), &path(0, tree), &mth(tree)),
        Ok(())
    );
    assert!(same_shape > 0);
}
