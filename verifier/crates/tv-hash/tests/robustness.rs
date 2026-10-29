// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Hostile proofs (RFC-0019 'Robustness'): seeded mutations of honest multiproofs and inclusion
//! proofs, and seeded random ones, never panic, and each yields exactly the result the test
//! support's oracle predicts, so that no mutated proof is accepted and every rejection names the
//! first check it fails. A panic is reported with the input that caused it.

mod support;

use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

use support::{Rng, Tree, inclusion_outcome, mth, multiproof_outcome, path};
use tv_hash::Digest;
use tv_hash::merkle::{MultiproofError, verify_multiproof};
use tv_hash::rfc6962::{InclusionError, leaf_hash, verify_inclusion};

/// The name of a multiproof outcome, without its fields.
fn multiproof_kind(outcome: &Result<(), MultiproofError>) -> &'static str {
    match outcome {
        Ok(()) => "Ok",
        Err(MultiproofError::Cancelled) => "Cancelled",
        Err(MultiproofError::TreeSize) => "TreeSize",
        Err(MultiproofError::NoLeaves) => "NoLeaves",
        Err(MultiproofError::Unordered { .. }) => "Unordered",
        Err(MultiproofError::OutOfRange { .. }) => "OutOfRange",
        Err(MultiproofError::MissingSiblings { .. }) => "MissingSiblings",
        Err(MultiproofError::ExtraSiblings { .. }) => "ExtraSiblings",
        Err(MultiproofError::RootMismatch) => "RootMismatch",
    }
}

/// The name of an inclusion outcome.
fn inclusion_kind(outcome: &Result<(), InclusionError>) -> &'static str {
    match outcome {
        Ok(()) => "Ok",
        Err(InclusionError::IndexBeyondSize) => "IndexBeyondSize",
        Err(InclusionError::ProofTooLong) => "ProofTooLong",
        Err(InclusionError::ProofTooShort) => "ProofTooShort",
        Err(InclusionError::RootMismatch) => "RootMismatch",
    }
}

/// Tree sizes that are interesting in a `u64`: small, around the honest tree, and huge.
fn tree_size(rng: &mut Rng, honest: u64) -> u64 {
    match rng.below(8) {
        0 => 0,
        1 => honest.wrapping_add(1),
        2 => honest.wrapping_mul(2),
        3 => honest / 2,
        4 => 1 << rng.below(64),
        5 => u64::MAX,
        6 => rng.next(),
        _ => rng.below(70),
    }
}

/// One mutation of a multiproof's inputs.
fn mutate_multiproof(
    rng: &mut Rng,
    tree: &Tree,
    n: &mut u64,
    root: &mut Digest,
    leaves: &mut Vec<(u64, Digest)>,
    siblings: &mut Vec<Digest>,
) {
    match rng.below(14) {
        0 if !siblings.is_empty() => {
            siblings.remove(rng.index(siblings.len()));
        }
        1 => siblings.insert(rng.index(siblings.len() + 1), rng.digest()),
        2 if siblings.len() >= 2 => {
            let (a, b) = (rng.index(siblings.len()), rng.index(siblings.len()));
            siblings.swap(a, b);
        }
        3 if !siblings.is_empty() => {
            let at = rng.index(siblings.len());
            siblings[at][rng.index(32)] ^= 1 << rng.below(8);
        }
        4 if !leaves.is_empty() => {
            let at = rng.index(leaves.len());
            leaves[at].1[rng.index(32)] ^= 1 << rng.below(8);
        }
        5 if !leaves.is_empty() => {
            let at = rng.index(leaves.len());
            leaves[at].0 = match rng.below(4) {
                0 => leaves[at].0.wrapping_add(1),
                1 => leaves[at].0.wrapping_sub(1),
                2 => tree.n(),
                _ => rng.next(),
            };
        }
        6 if !leaves.is_empty() => {
            let at = rng.index(leaves.len());
            let copy = leaves[at];
            leaves.insert(at, copy);
        }
        7 if !leaves.is_empty() => {
            leaves.remove(rng.index(leaves.len()));
        }
        8 if leaves.len() >= 2 => {
            let (a, b) = (rng.index(leaves.len()), rng.index(leaves.len()));
            leaves.swap(a, b);
        }
        9 => leaves.clear(),
        10 => *n = tree_size(rng, tree.n()),
        11 => root[rng.index(32)] ^= 1 << rng.below(8),
        12 => {
            // A leaf of the tree at another index, or a node from above the leaves.
            if let Some(at) = (!leaves.is_empty()).then(|| rng.index(leaves.len())) {
                let level = rng.index(tree.levels.len());
                let nodes = &tree.levels[level];
                leaves[at].1 = nodes[rng.index(nodes.len())];
            }
        }
        _ => {
            *siblings = (0..rng.below(80)).map(|_| rng.digest()).collect();
        }
    }
}

#[test]
fn mutated_multiproofs_never_panic_and_are_judged_exactly() {
    let mut rng = Rng::new(0x20b057);
    let trees: Vec<Tree> = (0..=10)
        .map(|log_n| Tree::random(log_n, &mut rng))
        .collect();
    let mut seen = BTreeMap::new();
    for round in 0..20_000 {
        let tree = &trees[rng.index(trees.len())];
        let count = rng.index(tree.n().min(40) as usize + 1);
        let mut indices: Vec<u64> = (0..count).map(|_| rng.below(tree.n())).collect();
        indices.sort_unstable();
        indices.dedup();
        let mut n = tree.n();
        let mut root = tree.root();
        let mut leaves = tree.open(&indices);
        let mut siblings = tree.multiproof(&indices);
        for _ in 0..rng.below(4) {
            mutate_multiproof(
                &mut rng,
                tree,
                &mut n,
                &mut root,
                &mut leaves,
                &mut siblings,
            );
        }
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            verify_multiproof(&root, n, &leaves, &siblings)
        }))
        .unwrap_or_else(|_| {
            panic!(
                "round {round}: panic on n = {n}, leaves {:?}, {} siblings",
                leaves.iter().map(|l| l.0).collect::<Vec<_>>(),
                siblings.len()
            )
        });
        assert_eq!(
            outcome,
            multiproof_outcome(&root, n, &leaves, &siblings),
            "round {round}: n = {n}, leaves {:?}",
            leaves.iter().map(|l| l.0).collect::<Vec<_>>()
        );
        *seen.entry(multiproof_kind(&outcome)).or_insert(0) += 1;
    }
    // Every outcome is reached; rounds without a mutation, or whose mutations cancel out, are
    // honest proofs.
    assert_eq!(seen.len(), 8, "{seen:?}");
    assert!(seen["Ok"] > 1_000, "{seen:?}");
}

#[test]
fn random_multiproofs_never_panic_and_never_verify() {
    let mut rng = Rng::new(0x2a1d0);
    let mut seen = BTreeMap::new();
    for round in 0..20_000 {
        let honest = 1 << rng.below(20);
        let n = tree_size(&mut rng, honest);
        let mut leaves: Vec<(u64, Digest)> = (0..rng.below(12))
            .map(|_| {
                let index = if n > 0 && rng.below(2) == 0 {
                    rng.below(n)
                } else {
                    rng.next()
                };
                (index, rng.digest())
            })
            .collect();
        if rng.below(2) == 0 {
            leaves.sort_unstable_by_key(|l| l.0);
            leaves.dedup_by_key(|l| l.0);
        }
        let siblings: Vec<Digest> = (0..rng.below(100)).map(|_| rng.digest()).collect();
        let root = rng.digest();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            verify_multiproof(&root, n, &leaves, &siblings)
        }))
        .unwrap_or_else(|_| panic!("round {round}: panic on n = {n}"));
        assert!(
            outcome.is_err(),
            "round {round}: a random multiproof verified"
        );
        assert_eq!(
            outcome,
            multiproof_outcome(&root, n, &leaves, &siblings),
            "round {round}"
        );
        *seen.entry(multiproof_kind(&outcome)).or_insert(0) += 1;
    }
    assert_eq!(seen.len(), 7, "every rejection: {seen:?}");
}

#[test]
fn mutated_inclusion_proofs_never_panic_and_are_judged_exactly() {
    let mut rng = Rng::new(0x1c1d);
    let entries: Vec<Vec<u8>> = (0..100)
        .map(|_| {
            let len = rng.index(40);
            rng.bytes(len)
        })
        .collect();
    let roots: Vec<Digest> = (0..=entries.len()).map(|n| mth(&entries[..n])).collect();
    let mut seen = BTreeMap::new();
    for round in 0..10_000 {
        let size = 1 + rng.index(entries.len());
        let index = rng.index(size);
        let mut proof = path(index, &entries[..size]);
        let mut leaf = leaf_hash(&entries[index]);
        let mut root = roots[size];
        let (mut index, mut size) = (index as u64, size as u64);
        for _ in 0..rng.below(4) {
            match rng.below(10) {
                0 if !proof.is_empty() => {
                    proof.remove(rng.index(proof.len()));
                }
                1 => proof.insert(rng.index(proof.len() + 1), rng.digest()),
                2 if proof.len() >= 2 => {
                    let (a, b) = (rng.index(proof.len()), rng.index(proof.len()));
                    proof.swap(a, b);
                }
                3 if !proof.is_empty() => {
                    let at = rng.index(proof.len());
                    proof[at][rng.index(32)] ^= 1 << rng.below(8);
                }
                4 => leaf[rng.index(32)] ^= 1 << rng.below(8),
                5 => root = roots[rng.index(roots.len())],
                6 => {
                    index = match rng.below(4) {
                        0 => index.wrapping_add(1),
                        1 => index.wrapping_sub(1),
                        2 => size,
                        _ => rng.next(),
                    }
                }
                7 => size = tree_size(&mut rng, size),
                8 => {
                    let at = rng.index(entries.len());
                    leaf = leaf_hash(&entries[at]);
                }
                _ => proof = (0..rng.below(70)).map(|_| rng.digest()).collect(),
            }
        }
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            verify_inclusion(index, size, &leaf, &proof, &root)
        }))
        .unwrap_or_else(|_| panic!("round {round}: panic on leaf {index} of {size}"));
        assert_eq!(
            outcome,
            inclusion_outcome(index, size, &leaf, &proof, &root),
            "round {round}: leaf {index} of {size}, {} digests",
            proof.len()
        );
        *seen.entry(inclusion_kind(&outcome)).or_insert(0) += 1;
    }
    assert_eq!(seen.len(), 5, "{seen:?}");
    assert!(seen["Ok"] > 1_000, "{seen:?}");
}

#[test]
fn random_inclusion_proofs_never_panic_and_never_verify() {
    let mut rng = Rng::new(0x7a2d);
    let mut seen = BTreeMap::new();
    for round in 0..20_000 {
        let honest = 1 + rng.below(1 << 20);
        let size = tree_size(&mut rng, honest);
        let index = match rng.below(3) {
            0 if size > 0 => rng.below(size),
            1 => size.wrapping_sub(1),
            _ => rng.next(),
        };
        let proof: Vec<Digest> = (0..rng.below(70)).map(|_| rng.digest()).collect();
        let (leaf, root) = (rng.digest(), rng.digest());
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            verify_inclusion(index, size, &leaf, &proof, &root)
        }))
        .unwrap_or_else(|_| panic!("round {round}: panic on leaf {index} of {size}"));
        assert!(outcome.is_err(), "round {round}: a random proof verified");
        assert_eq!(
            outcome,
            inclusion_outcome(index, size, &leaf, &proof, &root),
            "round {round}: leaf {index} of {size}"
        );
        *seen.entry(inclusion_kind(&outcome)).or_insert(0) += 1;
    }
    assert_eq!(seen.len(), 4, "every rejection: {seen:?}");
}
