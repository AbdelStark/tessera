// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Cases and oracles for the `multiproof` and `inclusion` targets.
//!
//! A case is an honest proof, built here from the specifications, then changed by a script the
//! fuzz input spells out. The oracles predict exactly what `tv-hash` must return for the changed
//! proof, in the order of checks its documentation gives, without `tv-hash`'s verification code:
//!
//! - BLAKE3 Merkle trees (RFC-0004 'Merkle trees') are built level by level with the test kit's
//!   reference BLAKE3, and a canonical multiproof is read off the full tree level by level. A
//!   proof that passes the structural checks is accepted when its leaves and siblings lead to
//!   its root, recomputed level by level, where `tv-hash` climbs from leaf to leaf.
//! - RFC 6962 roots and inclusion paths follow the recursive definitions `MTH` and `PATH` of
//!   RFC 6962 §2.1, with `tv-hash`'s SHA-256, which NIST's vectors check. A proof of the right
//!   length is accepted if `PATH`, read backwards, leads from the leaf to the root.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::OnceLock;

use tv_hash::Digest;
use tv_hash::merkle::MultiproofError;
use tv_hash::rfc6962::InclusionError;
use tv_hash::sha256::sha256;
use tv_testkit::{SplitMix64, blake3};

/// The fuzz input, read front to back; past its end every read is zero.
pub struct Script<'a> {
    rest: &'a [u8],
}

impl<'a> Script<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { rest: data }
    }

    pub fn is_empty(&self) -> bool {
        self.rest.is_empty()
    }

    pub fn u8(&mut self) -> u8 {
        match self.rest.split_first() {
            Some((&byte, rest)) => {
                self.rest = rest;
                byte
            }
            None => 0,
        }
    }

    pub fn u16(&mut self) -> u16 {
        u16::from_le_bytes([self.u8(), self.u8()])
    }

    pub fn u64(&mut self) -> u64 {
        u64::from_le_bytes(std::array::from_fn(|_| self.u8()))
    }

    pub fn digest(&mut self) -> Digest {
        std::array::from_fn(|_| self.u8())
    }

    /// A position below `len`, which is not 0.
    pub fn below(&mut self, len: usize) -> usize {
        usize::from(self.u16()) % len
    }
}

/// A tree size: one of the values that sit on a check, or any `u64`.
fn tree_size(script: &mut Script<'_>, honest: u64) -> u64 {
    match script.u8() % 8 {
        0 => 0,
        1 => honest.wrapping_add(1),
        2 => honest.wrapping_sub(1),
        3 => honest.wrapping_mul(2),
        4 => honest / 2,
        5 => 1 << (script.u8() % 64),
        6 => u64::MAX,
        _ => script.u64(),
    }
}

// ----- BLAKE3 Merkle trees ------------------------------------------------------------------------

/// The largest tree a case opens: `2^MAX_LOG_N` leaves.
pub const MAX_LOG_N: u8 = 10;

/// Every level of a tree, the leaves first and the root last.
pub struct Tree {
    pub levels: Vec<Vec<Digest>>,
}

impl Tree {
    fn random(log_n: u8, rng: &mut SplitMix64) -> Self {
        let leaves: Vec<Digest> = (0..1u64 << log_n)
            .map(|_| std::array::from_fn(|i| (rng.next_u64() >> (8 * (i % 8))) as u8))
            .collect();
        let mut levels = vec![leaves];
        while levels.last().unwrap().len() > 1 {
            let above = levels
                .last()
                .unwrap()
                .chunks(2)
                .map(|pair| {
                    blake3::derive_key("tessera/v1/merkle-node", &[pair[0], pair[1]].concat())
                })
                .collect();
            levels.push(above);
        }
        Self { levels }
    }

    pub fn n(&self) -> u64 {
        self.levels[0].len() as u64
    }

    pub fn root(&self) -> Digest {
        self.levels.last().unwrap()[0]
    }
}

/// The trees of 2^0 to 2^MAX_LOG_N leaves, built once.
pub fn trees() -> &'static [Tree] {
    static TREES: OnceLock<Vec<Tree>> = OnceLock::new();
    TREES.get_or_init(|| {
        let mut rng = SplitMix64::new(0x7e55_e7a0_f022);
        (0..=MAX_LOG_N)
            .map(|log_n| Tree::random(log_n, &mut rng))
            .collect()
    })
}

/// The canonical multiproof of `indices`, sorted and distinct, in the tree whose levels are
/// `levels`: level by level from the leaves, the siblings of the known nodes that are not known
/// themselves, left to right.
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

/// The length of the canonical multiproof of `indices` in a tree of `n` leaves.
pub fn canonical_count(n: u64, indices: &[u64]) -> usize {
    let mut known: BTreeSet<u64> = indices.iter().copied().collect();
    let mut count = 0;
    for _ in 0..n.trailing_zeros() {
        count += known.iter().filter(|&&i| !known.contains(&(i ^ 1))).count();
        known = known.iter().map(|i| i >> 1).collect();
    }
    count
}

/// A multiproof to check.
pub struct MultiproofCase {
    pub tree: &'static Tree,
    pub root: Digest,
    pub n: u64,
    pub leaves: Vec<(u64, Digest)>,
    pub siblings: Vec<Digest>,
}

/// The case a fuzz input spells: a tree of `2^(byte 0 mod 11)` leaves; an opening of the distinct
/// indices among the next `byte 1 mod 65` 16-bit values, reduced modulo `n`, with its canonical
/// multiproof; then changes, an operation byte and its arguments each, until the input ends.
pub fn multiproof_case(data: &[u8]) -> MultiproofCase {
    let mut script = Script::new(data);
    let tree = &trees()[usize::from(script.u8() % (MAX_LOG_N + 1))];
    let n = tree.n();
    let count = script.u8() % 65;
    let indices: BTreeSet<u64> = (0..count).map(|_| u64::from(script.u16()) % n).collect();
    let indices: Vec<u64> = indices.into_iter().collect();
    let mut case = MultiproofCase {
        tree,
        root: tree.root(),
        n,
        leaves: indices
            .iter()
            .map(|&i| (i, tree.levels[0][i as usize]))
            .collect(),
        siblings: canonical_multiproof(&tree.levels, &indices),
    };
    while !script.is_empty() {
        let op = script.u8();
        let (leaves, siblings) = (&mut case.leaves, &mut case.siblings);
        match op % 16 {
            0 if !siblings.is_empty() => {
                siblings.remove(script.below(siblings.len()));
            }
            1 => {
                let at = script.below(siblings.len() + 1);
                siblings.insert(at, script.digest());
            }
            2 if !siblings.is_empty() => {
                let copy = siblings[script.below(siblings.len())];
                let at = script.below(siblings.len() + 1);
                siblings.insert(at, copy);
            }
            3 if !siblings.is_empty() => {
                let (a, b) = (script.below(siblings.len()), script.below(siblings.len()));
                siblings.swap(a, b);
            }
            4 if !siblings.is_empty() => {
                let at = script.below(siblings.len());
                let bit = script.u8();
                siblings[at][usize::from(bit / 8)] ^= 1 << (bit % 8);
            }
            5 if !leaves.is_empty() => {
                let at = script.below(leaves.len());
                let bit = script.u8();
                leaves[at].1[usize::from(bit / 8)] ^= 1 << (bit % 8);
            }
            6 if !leaves.is_empty() => {
                let at = script.below(leaves.len());
                leaves[at].0 = script.u64();
            }
            7 if !leaves.is_empty() => {
                let at = script.below(leaves.len());
                leaves[at].0 = if script.u8() & 1 == 0 {
                    leaves[at].0.wrapping_add(1)
                } else {
                    leaves[at].0.wrapping_sub(1)
                };
            }
            8 if !leaves.is_empty() => {
                let at = script.below(leaves.len());
                let copy = leaves[at];
                leaves.insert(at, copy);
            }
            9 if !leaves.is_empty() => {
                leaves.remove(script.below(leaves.len()));
            }
            10 if !leaves.is_empty() => {
                let (a, b) = (script.below(leaves.len()), script.below(leaves.len()));
                leaves.swap(a, b);
            }
            11 => case.n = tree_size(&mut script, n),
            12 => {
                let bit = script.u8();
                case.root[usize::from(bit / 8)] ^= 1 << (bit % 8);
            }
            13 if !leaves.is_empty() => {
                // A node of the tree, from any level, as the digest of a leaf.
                let at = script.below(leaves.len());
                let level = &tree.levels[script.below(tree.levels.len())];
                leaves[at].1 = level[script.below(level.len())];
            }
            14 => {
                let keep = script.below(siblings.len() + 1);
                siblings.truncate(keep);
            }
            _ => {
                // The nodes at a level above the leaves, opened as the leaves of the smaller tree
                // of that level, with their canonical proof: honest for that tree size.
                let level = script.below(tree.levels.len());
                let width = tree.levels[level].len() as u64;
                let picked: BTreeSet<u64> = (0..1 + script.u8() % 8)
                    .map(|_| u64::from(script.u16()) % width)
                    .collect();
                let picked: Vec<u64> = picked.into_iter().collect();
                *leaves = picked
                    .iter()
                    .map(|&i| (i, tree.levels[level][i as usize]))
                    .collect();
                *siblings = canonical_multiproof(&tree.levels[level..], &picked);
                case.n = width;
            }
        }
    }
    case
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
    let mut known: BTreeMap<u64, Digest> = leaves.iter().copied().collect();
    let mut proof = siblings.iter();
    for _ in 0..n.trailing_zeros() {
        let mut above = BTreeMap::new();
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
            let parent = blake3::derive_key("tessera/v1/merkle-node", &[left, right].concat());
            above.insert(i >> 1, parent);
        }
        known = above;
    }
    if proof.next().is_some() || known.len() != 1 {
        return None;
    }
    known.get(&0).copied()
}

/// What `verify_multiproof` must return for `case`: the structural checks in `tv-hash`'s order,
/// then the root [`root_from_multiproof`] recomputes. A changed case is accepted exactly when it
/// is still a true statement: the canonical multiproof of its leaves in a tree with its root.
pub fn multiproof_outcome(case: &MultiproofCase) -> Result<(), MultiproofError> {
    let MultiproofCase {
        root,
        n,
        leaves,
        siblings,
        ..
    } = case;
    let n = *n;
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
    let found = siblings.len();
    if found < expected {
        return Err(MultiproofError::MissingSiblings { expected, found });
    }
    if found > expected {
        return Err(MultiproofError::ExtraSiblings { expected, found });
    }
    if root_from_multiproof(n, leaves, siblings) == Some(*root) {
        Ok(())
    } else {
        Err(MultiproofError::RootMismatch)
    }
}

// ----- RFC 6962 trees -----------------------------------------------------------------------------

/// The most leaves of a case's log.
pub const MAX_ENTRIES: usize = 260;

/// The entries of the log, built once.
fn entries() -> &'static [Vec<u8>] {
    static ENTRIES: OnceLock<Vec<Vec<u8>>> = OnceLock::new();
    ENTRIES.get_or_init(|| {
        let mut rng = SplitMix64::new(0x6962_f022);
        (0..MAX_ENTRIES)
            .map(|_| {
                let len = (rng.next_u64() % 48) as usize;
                (0..len).map(|_| rng.next_u64() as u8).collect()
            })
            .collect()
    })
}

/// The largest power of two smaller than `n`, for `n > 1`.
fn split(n: u64) -> u64 {
    1 << (63 - (n - 1).leading_zeros())
}

fn node(left: &Digest, right: &Digest) -> Digest {
    sha256(&[&[0x01][..], left, right].concat())
}

fn leaf(entry: &[u8]) -> Digest {
    sha256(&[&[0x00][..], entry].concat())
}

/// `MTH(D[start..end])` of RFC 6962 §2.1 over the log's entries, remembered across cases.
fn mth(start: usize, end: usize) -> Digest {
    thread_local! {
        static KNOWN: RefCell<HashMap<(usize, usize), Digest>> = RefCell::new(HashMap::new());
    }
    if let Some(digest) = KNOWN.with(|known| known.borrow().get(&(start, end)).copied()) {
        return digest;
    }
    let digest = match end - start {
        0 => sha256(&[]),
        1 => leaf(&entries()[start]),
        n => {
            let k = split(n as u64) as usize;
            node(&mth(start, start + k), &mth(start + k, end))
        }
    };
    KNOWN.with(|known| known.borrow_mut().insert((start, end), digest));
    digest
}

/// `PATH(m, D[start..end])` of RFC 6962 §2.1.1, from the leaf upward.
fn path(m: usize, start: usize, end: usize) -> Vec<Digest> {
    let n = end - start;
    if n <= 1 {
        return Vec::new();
    }
    let k = split(n as u64) as usize;
    let (mut below, sibling) = if m < k {
        (path(m, start, start + k), mth(start + k, end))
    } else {
        (path(m - k, start + k, end), mth(start, start + k))
    };
    below.push(sibling);
    below
}

/// The length of `PATH(m, D[n])`.
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

/// The root `proof` leads to for leaf `m` of `n`, by `PATH` read backwards.
pub fn root_from_path(m: u64, n: u64, leaf: &Digest, proof: &[Digest]) -> Option<Digest> {
    if n <= 1 {
        return proof.is_empty().then_some(*leaf);
    }
    let (sibling, below) = proof.split_last()?;
    let k = split(n);
    Some(if m < k {
        node(&root_from_path(m, k, leaf, below)?, sibling)
    } else {
        node(sibling, &root_from_path(m - k, n - k, leaf, below)?)
    })
}

/// An inclusion proof to check.
pub struct InclusionCase {
    pub index: u64,
    pub size: u64,
    pub leaf: Digest,
    pub proof: Vec<Digest>,
    pub root: Digest,
}

/// The case a fuzz input spells: a log of `1 + (u16 mod MAX_ENTRIES)` entries, the leaf at
/// `u16 mod size` with its `PATH` and the log's root; then changes, an operation byte and its
/// arguments each, until the input ends.
pub fn inclusion_case(data: &[u8]) -> InclusionCase {
    let mut script = Script::new(data);
    let size = 1 + script.below(MAX_ENTRIES);
    let index = script.below(size);
    let mut case = InclusionCase {
        index: index as u64,
        size: size as u64,
        leaf: leaf(&entries()[index]),
        proof: path(index, 0, size),
        root: mth(0, size),
    };
    while !script.is_empty() {
        let proof = &mut case.proof;
        match script.u8() % 12 {
            0 if !proof.is_empty() => {
                proof.remove(script.below(proof.len()));
            }
            1 => {
                let at = script.below(proof.len() + 1);
                proof.insert(at, script.digest());
            }
            2 if !proof.is_empty() => {
                let copy = proof[script.below(proof.len())];
                let at = script.below(proof.len() + 1);
                proof.insert(at, copy);
            }
            3 if !proof.is_empty() => {
                let (a, b) = (script.below(proof.len()), script.below(proof.len()));
                proof.swap(a, b);
            }
            4 if !proof.is_empty() => {
                let at = script.below(proof.len());
                let bit = script.u8();
                proof[at][usize::from(bit / 8)] ^= 1 << (bit % 8);
            }
            5 => {
                let bit = script.u8();
                case.leaf[usize::from(bit / 8)] ^= 1 << (bit % 8);
            }
            6 => case.leaf = leaf(&entries()[script.below(MAX_ENTRIES)]),
            7 => case.root = mth(0, 1 + script.below(MAX_ENTRIES)),
            8 => {
                let bit = script.u8();
                case.root[usize::from(bit / 8)] ^= 1 << (bit % 8);
            }
            9 => {
                case.index = match script.u8() % 4 {
                    0 => case.index.wrapping_add(1),
                    1 => case.index.wrapping_sub(1),
                    2 => case.size,
                    _ => script.u64(),
                }
            }
            10 => case.size = tree_size(&mut script, case.size),
            _ => {
                // Another leaf and size of the log, honest: a proof the check must accept.
                let size = 1 + script.below(MAX_ENTRIES);
                let index = script.below(size);
                case.index = index as u64;
                case.size = size as u64;
                case.leaf = leaf(&entries()[index]);
                case.proof = path(index, 0, size);
                case.root = mth(0, size);
            }
        }
    }
    case
}

/// What `verify_inclusion` must return for `case`, in RFC 9162's order of failures.
pub fn inclusion_outcome(case: &InclusionCase) -> Result<(), InclusionError> {
    if case.index >= case.size {
        return Err(InclusionError::IndexBeyondSize);
    }
    let needed = path_len(case.index, case.size);
    if case.proof.len() > needed {
        return Err(InclusionError::ProofTooLong);
    }
    if case.proof.len() < needed {
        return Err(InclusionError::ProofTooShort);
    }
    if root_from_path(case.index, case.size, &case.leaf, &case.proof) == Some(case.root) {
        Ok(())
    } else {
        Err(InclusionError::RootMismatch)
    }
}
