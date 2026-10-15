// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Canonical multiproofs (RFC-0004 'Merkle trees') on trees of 2^1 to 2^20 leaves: random query
//! sets verify, and every way of breaking an opening rejects. RFC-0004 names the extra, missing
//! and reordered sibling; the rest are the other inputs a verifier must not trust.

mod common;

use common::SplitMix64;
use tessera_hash::{Digest, MerkleError, MerkleTree, MultiProof, merkle_node, verify_multiproof};

/// One valid opening.
struct Opening {
    root: Digest,
    n: u64,
    leaves: Vec<(u64, Digest)>,
    proof: MultiProof,
}

impl Opening {
    fn verifies(&self) -> bool {
        verify_multiproof(&self.root, self.n, &self.leaves, &self.proof)
    }
}

fn flip(digest: &mut Digest) {
    digest[31] ^= 1;
}

/// Every single change to `opening` that must reject, with its name.
fn tamperings(opening: &Opening, rng: &mut SplitMix64) -> Vec<(&'static str, Opening)> {
    let changed = |f: &dyn Fn(&mut Opening)| {
        let mut copy = Opening {
            root: opening.root,
            n: opening.n,
            leaves: opening.leaves.clone(),
            proof: opening.proof.clone(),
        };
        f(&mut copy);
        copy
    };
    let extra = rng.digest();
    let mut out = vec![
        (
            "extra sibling at the end",
            changed(&|o| o.proof.siblings.push(extra)),
        ),
        (
            "extra sibling at the start",
            changed(&|o| o.proof.siblings.insert(0, extra)),
        ),
        ("altered leaf", changed(&|o| flip(&mut o.leaves[0].1))),
        ("altered root", changed(&|o| flip(&mut o.root))),
        ("doubled leaf count", changed(&|o| o.n *= 2)),
        ("no leaves", changed(&|o| o.leaves.clear())),
        (
            "repeated leaf",
            changed(&|o| o.leaves.insert(0, o.leaves[0])),
        ),
        (
            "index out of range",
            changed(&|o| {
                let last = o.leaves.len() - 1;
                o.leaves[last].0 = o.n;
            }),
        ),
    ];
    let siblings = opening.proof.siblings.len();
    if siblings > 0 {
        out.push((
            "missing last sibling",
            changed(&|o| {
                o.proof.siblings.pop();
            }),
        ));
        out.push((
            "missing first sibling",
            changed(&|o| {
                o.proof.siblings.remove(0);
            }),
        ));
        let at = rng.below(siblings as u64) as usize;
        out.push((
            "altered sibling",
            changed(&|o| flip(&mut o.proof.siblings[at])),
        ));
    }
    if siblings > 1 {
        let (a, b) = (0, siblings - 1);
        out.push((
            "reordered siblings",
            changed(&|o| o.proof.siblings.swap(a, b)),
        ));
    }
    if opening.leaves.len() > 1 {
        out.push(("unsorted leaves", changed(&|o| o.leaves.reverse())));
    }
    if opening.n > 1 {
        out.push(("halved leaf count", changed(&|o| o.n /= 2)));
    }
    // An opened digest claimed at an index that was not opened.
    let opened: Vec<u64> = opening.leaves.iter().map(|&(i, _)| i).collect();
    if let Some(free) = (0..opening.n).find(|i| !opened.contains(i)) {
        out.push((
            "leaf moved to another index",
            changed(&|o| {
                o.leaves[0].0 = free;
                o.leaves.sort_by_key(|&(i, _)| i);
            }),
        ));
    }
    out
}

fn check(tree: &MerkleTree, leaves: &[Digest], indices: &[u64], rng: &mut SplitMix64) {
    let proof = tree.open(indices).expect("valid indices");
    let mut sorted = indices.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let opening = Opening {
        root: tree.root(),
        n: leaves.len() as u64,
        leaves: sorted.iter().map(|&i| (i, leaves[i as usize])).collect(),
        proof,
    };
    assert!(opening.verifies(), "{} leaves, {sorted:?}", opening.n);
    for (name, tampered) in tamperings(&opening, rng) {
        assert!(
            !tampered.verifies(),
            "{name} accepted: {} leaves, {sorted:?}",
            opening.n
        );
    }
}

#[test]
fn multiproofs_verify_and_every_tampering_rejects() {
    let mut rng = SplitMix64::new(0x6d65_726b_6c65_0001);
    for log_n in 1..=20u32 {
        let n = 1u64 << log_n;
        let leaves: Vec<Digest> = (0..n).map(|_| rng.digest()).collect();
        let tree = MerkleTree::new(leaves.clone()).expect("a power of two");
        let mut queries: Vec<Vec<u64>> = vec![vec![0], vec![n - 1], vec![0, n - 1]];
        for size in [1, 2, 3, 17, 64] {
            queries.push((0..size).map(|_| rng.below(n)).collect());
        }
        // Siblings that pair up inside the query set, and a query in no particular order.
        queries.push((0..n.min(8)).collect());
        queries.push((0..n.min(16)).rev().step_by(3).collect());
        if n <= 64 {
            queries.push((0..n).collect());
        }
        for indices in &queries {
            check(&tree, &leaves, indices, &mut rng);
        }
    }
}

/// Openings no honest prover produces, each with the siblings the verifier would consume. The
/// strict index order, the range check, the power-of-two check and the exact final state
/// overlap, so each forgery is also rejected by another rule. These pin the contract the
/// verifier relies on, whichever rule a refactor drops.
#[test]
fn forged_openings_reject() {
    let mut rng = SplitMix64::new(0x6d65_726b_6c65_0003);
    let l: Vec<Digest> = (0..4).map(|_| rng.digest()).collect();
    let tree = MerkleTree::new(l.clone()).expect("four leaves");
    let root = tree.root();
    let (h01, h23) = (merkle_node(&l[0], &l[1]), merkle_node(&l[2], &l[3]));
    let fake = rng.digest();
    let honest = tree.open(&[0]).expect("index 0");
    assert!(verify_multiproof(&root, 4, &[(0, l[0])], &honest));

    // The honest opening of leaf 0 against every size that is not 4.
    for n in [0, 1, 2, 3, 5, 6, 7, 8, u64::MAX] {
        assert!(
            !verify_multiproof(&root, n, &[(0, l[0])], &honest),
            "n = {n}"
        );
    }
    let proof = |siblings: Vec<Digest>| MultiProof { siblings };
    let forgeries = [
        // A second digest for an opened index: the first chain still reaches the root.
        (
            "repeated index",
            vec![(0, l[0]), (0, fake)],
            proof(vec![l[1], fake, h23, fake]),
        ),
        // The right leaves in the wrong order: both chains reach the root.
        (
            "unsorted indices",
            vec![(2, l[2]), (0, l[0])],
            proof(vec![l[3], l[1], h01, h23]),
        ),
        // A digest beyond the tree: the in-range chain still reaches the root.
        (
            "index beyond n",
            vec![(0, l[0]), (4, fake)],
            proof(vec![l[1], fake, h23, fake]),
        ),
    ];
    for (name, leaves, proof) in forgeries {
        assert!(
            !verify_multiproof(&root, 4, &leaves, &proof),
            "{name} accepted"
        );
    }
}

#[test]
fn single_leaf_trees_have_the_leaf_as_root() {
    let mut rng = SplitMix64::new(0x6d65_726b_6c65_0002);
    let leaf = rng.digest();
    let tree = MerkleTree::new(vec![leaf]).expect("one leaf");
    assert_eq!(tree.root(), leaf);
    let proof = tree.open(&[0]).expect("index 0");
    assert!(proof.siblings.is_empty());
    assert!(verify_multiproof(&leaf, 1, &[(0, leaf)], &proof));
}

#[test]
fn invalid_trees_and_queries_are_errors() {
    for n in [0, 3, 6, 100] {
        assert_eq!(
            MerkleTree::new(vec![[0; 32]; n]).map(|t| t.root()),
            Err(MerkleError::LeafCount { leaves: n })
        );
    }
    let tree = MerkleTree::new(vec![[7; 32]; 8]).expect("eight leaves");
    assert_eq!(tree.open(&[]), Err(MerkleError::EmptyQuery));
    assert_eq!(
        tree.open(&[1, 8]),
        Err(MerkleError::IndexOutOfRange {
            index: 8,
            leaves: 8
        })
    );
    // Duplicates are the same query as one index.
    assert_eq!(tree.open(&[3, 3, 3]), tree.open(&[3]));
}
