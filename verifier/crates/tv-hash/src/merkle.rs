// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Binary Merkle trees over the `merkle-node` domain and their canonical multiproofs
//! (RFC-0004 'Merkle trees').
//!
//! A tree has `n` leaves, `n` a power of two; a node is `H("merkle-node", left ‖ right)` and the
//! root sits at level `log2 n`. A multiproof opens a set of leaves: given their indices, sorted
//! and without duplicates, it lists, level by level from the leaves upward and left to right
//! within a level, exactly the sibling digests the verifier cannot compute itself.
//!
//! # How the proof is checked
//!
//! The prover's natural algorithm builds the opened nodes level by level. This one, written for
//! the verifier, makes two passes instead (SP-6: the two should not share a structural bug):
//!
//! 1. From the indices alone, it counts the siblings each level needs: the opened nodes at that
//!    level whose sibling is not opened. That gives where each level's siblings start in the
//!    proof and how many digests the proof must hold.
//! 2. It climbs from each leaf in turn, left to right, as far as it can: a left child whose
//!    sibling sits above a later leaf waits on a stack, at most one node per level, for that
//!    leaf's climb to reach it; every other node takes its sibling from its level's next proof
//!    digest. The last leaf's climb ends at the root.
//!
//! Nothing is allocated: the stack and the per-level counts are arrays of 64 entries, one per
//! level of the deepest tree a `u64` can count.

use core::fmt;

use crate::Digest;
use crate::domain::{Domain, hash_pair};

/// Levels of the deepest tree: `n ≤ 2^63`.
const MAX_LEVELS: usize = 64;

/// A node of a commitment's tree: `H("merkle-node", left ‖ right)`.
pub fn node(left: &Digest, right: &Digest) -> Digest {
    hash_pair(Domain::MerkleNode, left, right)
}

/// Why a multiproof was rejected, reported as the first check that failed in the order of
/// [`verify_multiproof`].
///
/// `MissingSiblings`, `ExtraSiblings` and `RootMismatch` judge the proof: `Reject(MerklePath)` in
/// RFC-0004 'Failure modes'. Apart from `Cancelled`, the others judge the verifier's tree size from
/// public data and the indices from the transcript, sorted and deduplicated: from a correct
/// verifier they never come, and they mean a fault of the verifier (`ERR-INTERNAL`), not of the
/// proof. `Cancelled` is an operational deadline error (`ERR-RES-TIME`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MultiproofError {
    /// Cooperative deadline expired (`ERR-RES-TIME`), not a proof rejection.
    Cancelled,
    /// The number of leaves `n` is not a power of two; zero is not.
    TreeSize,
    /// No leaf is opened.
    NoLeaves,
    /// The leaf at `position` does not have a larger index than the one before it: the indices
    /// are unsorted or repeated.
    Unordered {
        /// The leaf's position in the opened leaves.
        position: usize,
    },
    /// The leaf at `position` has an index of `n` or more.
    OutOfRange {
        /// The leaf's position in the opened leaves.
        position: usize,
    },
    /// The proof holds fewer digests than the canonical multiproof of these indices.
    MissingSiblings {
        /// Digests the canonical multiproof holds.
        expected: usize,
        /// Digests the proof holds.
        found: usize,
    },
    /// The proof holds more digests than the canonical multiproof of these indices.
    ExtraSiblings {
        /// Digests the canonical multiproof holds.
        expected: usize,
        /// Digests the proof holds.
        found: usize,
    },
    /// The opened leaves and the proof lead to another root: a leaf or a sibling is wrong, or
    /// the siblings are out of order.
    RootMismatch,
}

impl fmt::Display for MultiproofError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MultiproofError::Cancelled => f.write_str("multiproof verification cancelled"),
            MultiproofError::TreeSize => f.write_str("the tree size is not a power of two"),
            MultiproofError::NoLeaves => f.write_str("the multiproof opens no leaf"),
            MultiproofError::Unordered { position } => write!(
                f,
                "opened leaf {position} does not follow the one before it in index order"
            ),
            MultiproofError::OutOfRange { position } => {
                write!(f, "opened leaf {position} is outside the tree")
            }
            MultiproofError::MissingSiblings { expected, found } => write!(
                f,
                "the multiproof holds {found} digests where {expected} are needed"
            ),
            MultiproofError::ExtraSiblings { expected, found } => write!(
                f,
                "the multiproof holds {found} digests where {expected} are needed"
            ),
            MultiproofError::RootMismatch => f.write_str("the multiproof leads to another root"),
        }
    }
}

impl std::error::Error for MultiproofError {}

/// Checks that `leaves`, pairs of a leaf index and the leaf's digest, are leaves of the tree of
/// `n` leaves whose root is `root`, given the canonical multiproof `siblings`. The checks run in
/// this order, and the first that fails is the error:
///
/// 1. `n` is a power of two ([`TreeSize`](MultiproofError::TreeSize)). It comes from public
///    data, never from the proof: the nodes at level `k` are the leaves of a tree of `n / 2^k`
///    leaves with the same root.
/// 2. At least one leaf is opened ([`NoLeaves`](MultiproofError::NoLeaves)).
/// 3. Leaf by leaf, in order: its index is larger than the one before it
///    ([`Unordered`](MultiproofError::Unordered)), then below `n`
///    ([`OutOfRange`](MultiproofError::OutOfRange)).
/// 4. `siblings` holds exactly as many digests as the canonical multiproof of these indices
///    ([`MissingSiblings`](MultiproofError::MissingSiblings),
///    [`ExtraSiblings`](MultiproofError::ExtraSiblings)).
/// 5. The root recomputed from the leaves and the siblings, each sibling used once at the place
///    the canonical order gives it, is `root` ([`RootMismatch`](MultiproofError::RootMismatch)).
///
/// The work is one hash per node above the opened leaves, at most `|leaves| · log2 n`, and
/// nothing is allocated.
///
/// Query indices squeezed by [`challenge_indices`](crate::transcript::Transcript::challenge_indices)
/// come in draw order and may repeat: the leaves opened are their sorted, deduplicated set
/// (RFC-0004 'Merkle trees').
pub fn verify_multiproof(
    root: &Digest,
    n: u64,
    leaves: &[(u64, Digest)],
    siblings: &[Digest],
) -> Result<(), MultiproofError> {
    verify_multiproof_with_cancel(root, n, leaves, siblings, || false)
}

/// [`verify_multiproof`] with a cooperative deadline (RFC-0019, SP-8).
///
/// Checks before work, at most every 4,096 validation/counting/climb steps, and
/// before returning success. No allocation or partial authentication result escapes.
///
/// # Errors
/// [`MultiproofError::Cancelled`] when `cancel` returns true; all other checks and
/// their order are those of [`verify_multiproof`] when cancellation is disabled.
pub fn verify_multiproof_with_cancel(
    root: &Digest,
    n: u64,
    leaves: &[(u64, Digest)],
    siblings: &[Digest],
    mut cancel: impl FnMut() -> bool,
) -> Result<(), MultiproofError> {
    let mut work = Work {
        remaining: 0,
        cancel: &mut cancel,
    };
    work.check()?;
    // 1.
    if !n.is_power_of_two() {
        return Err(MultiproofError::TreeSize);
    }
    let levels = n.trailing_zeros();
    // 2.
    if leaves.is_empty() {
        return Err(MultiproofError::NoLeaves);
    }
    // 3.
    let mut previous: Option<u64> = None;
    for (position, (index, _)) in leaves.iter().enumerate() {
        work.step()?;
        if previous.is_some_and(|previous| *index <= previous) {
            return Err(MultiproofError::Unordered { position });
        }
        if *index >= n {
            return Err(MultiproofError::OutOfRange { position });
        }
        previous = Some(*index);
    }
    // 4.
    let layout = Layout::of(leaves, levels, &mut work)?;
    let found = siblings.len();
    if found < layout.total {
        return Err(MultiproofError::MissingSiblings {
            expected: layout.total,
            found,
        });
    }
    if found > layout.total {
        return Err(MultiproofError::ExtraSiblings {
            expected: layout.total,
            found,
        });
    }
    // 5.
    let computed = climb(leaves, levels, siblings, &layout, &mut work)?;
    work.check()?;
    if computed == *root {
        Ok(())
    } else {
        Err(MultiproofError::RootMismatch)
    }
}

struct Work<'a> {
    remaining: u32,
    cancel: &'a mut dyn FnMut() -> bool,
}
impl Work<'_> {
    fn check(&mut self) -> Result<(), MultiproofError> {
        if (self.cancel)() {
            return Err(MultiproofError::Cancelled);
        }
        self.remaining = 4096;
        Ok(())
    }
    fn step(&mut self) -> Result<(), MultiproofError> {
        if self.remaining == 0 {
            self.check()?;
        }
        self.remaining = self.remaining.saturating_sub(1);
        Ok(())
    }
}

/// Where each level's siblings sit in a canonical multiproof.
struct Layout {
    /// The first sibling of each level, as an index into the proof.
    start: [usize; MAX_LEVELS],
    /// One past the last sibling of each level.
    end: [usize; MAX_LEVELS],
    /// The proof's length.
    total: usize,
}

impl Layout {
    /// The layout for `leaves`, whose indices are strictly increasing and below `2^levels`.
    fn of(
        leaves: &[(u64, Digest)],
        levels: u32,
        work: &mut Work<'_>,
    ) -> Result<Self, MultiproofError> {
        let mut layout = Self {
            start: [0; MAX_LEVELS],
            end: [0; MAX_LEVELS],
            total: 0,
        };
        let mut level: u32 = 0;
        for (start, end) in layout.start.iter_mut().zip(layout.end.iter_mut()) {
            if level >= levels {
                break;
            }
            *start = layout.total;
            layout.total = layout.total.saturating_add(unpaired(leaves, level, work)?);
            *end = layout.total;
            level = level.saturating_add(1);
        }
        Ok(layout)
    }
}

/// How many of the nodes at `level` above the leaves have their sibling above no leaf: the
/// siblings a canonical multiproof holds for that level.
fn unpaired(
    leaves: &[(u64, Digest)],
    level: u32,
    work: &mut Work<'_>,
) -> Result<usize, MultiproofError> {
    let mut count: usize = 0;
    let mut last: Option<u64> = None;
    // A left child seen last, whose right sibling may be the next node.
    let mut waiting: Option<u64> = None;
    // The leaves are sorted, so their ancestors at `level` come in order, repeated when several
    // leaves share one.
    for node in leaves.iter().map(|(index, _)| ancestor(*index, level)) {
        work.step()?;
        if last == Some(node) {
            continue;
        }
        last = Some(node);
        if let Some(left) = waiting.take() {
            if node == left | 1 {
                // `node` and the left child before it are siblings: neither needs a digest.
                continue;
            }
            count = count.saturating_add(1);
        }
        if node & 1 == 0 {
            waiting = Some(node);
        } else {
            // A right child whose left sibling would have been the node just before it.
            count = count.saturating_add(1);
        }
    }
    if waiting.is_some() {
        count = count.saturating_add(1);
    }
    Ok(count)
}

/// The ancestor of leaf `index` at `level` above the leaves.
fn ancestor(index: u64, level: u32) -> u64 {
    index.checked_shr(level).unwrap_or(0)
}

/// A node that waits for its right sibling.
#[derive(Clone, Copy)]
struct Waiting {
    level: u32,
    index: u64,
    digest: Digest,
}

/// The root recomputed from `leaves` and `siblings` laid out as `layout` says, or `None` when
/// the climb would use a digest outside its level's range or leave one unused. A proof of the
/// canonical length never makes it do either: those checks are defensive, and no test reaches
/// them.
fn climb(
    leaves: &[(u64, Digest)],
    levels: u32,
    siblings: &[Digest],
    layout: &Layout,
    work: &mut Work<'_>,
) -> Result<Digest, MultiproofError> {
    let mut waiting = [Waiting {
        level: 0,
        index: 0,
        digest: [0; 32],
    }; MAX_LEVELS];
    let mut waiting_len: usize = 0;
    // The next sibling of each level.
    let mut next = layout.start;
    let mut root = None;

    let mut later = leaves.iter().skip(1).map(|(index, _)| *index);
    for (leaf, digest) in leaves {
        let following = later.next();
        let mut level: u32 = 0;
        let mut index = *leaf;
        let mut digest = *digest;
        loop {
            work.step()?;
            if level == levels {
                root = Some(digest);
                break;
            }
            if index & 1 == 0 {
                let right = index | 1;
                if following.is_some_and(|after| ancestor(after, level) == right) {
                    // The right sibling is above the next leaf: its climb will come back here.
                    let slot = waiting
                        .get_mut(waiting_len)
                        .ok_or(MultiproofError::RootMismatch)?;
                    *slot = Waiting {
                        level,
                        index,
                        digest,
                    };
                    waiting_len = waiting_len
                        .checked_add(1)
                        .ok_or(MultiproofError::RootMismatch)?;
                    break;
                }
                let sibling = take(siblings, layout, &mut next, level)
                    .ok_or(MultiproofError::RootMismatch)?;
                digest = node(&digest, sibling);
            } else {
                let top = waiting_len
                    .checked_sub(1)
                    .and_then(|top| waiting.get(top))
                    .filter(|top| top.level == level && top.index == index ^ 1);
                if let Some(left) = top {
                    digest = node(&left.digest, &digest);
                    waiting_len = waiting_len
                        .checked_sub(1)
                        .ok_or(MultiproofError::RootMismatch)?;
                } else {
                    let sibling = take(siblings, layout, &mut next, level)
                        .ok_or(MultiproofError::RootMismatch)?;
                    digest = node(sibling, &digest);
                }
            }
            index >>= 1;
            level = level.checked_add(1).ok_or(MultiproofError::RootMismatch)?;
        }
    }
    // Every waiting node was met by its right sibling, and every digest of the proof was used.
    let used_all = next
        .iter()
        .zip(&layout.end)
        .take(usize::try_from(levels).map_err(|_| MultiproofError::RootMismatch)?)
        .all(|(next, end)| next == end);
    if waiting_len != 0 || !used_all {
        return Err(MultiproofError::RootMismatch);
    }
    root.ok_or(MultiproofError::RootMismatch)
}

/// The next sibling of `level`, if the level has one left.
fn take<'a>(
    siblings: &'a [Digest],
    layout: &Layout,
    next: &mut [usize; MAX_LEVELS],
    level: u32,
) -> Option<&'a Digest> {
    let level = usize::try_from(level).ok()?;
    let position = next.get_mut(level)?;
    if *position >= *layout.end.get(level)? {
        return None;
    }
    let sibling = siblings.get(*position)?;
    *position = position.checked_add(1)?;
    Some(sibling)
}

#[cfg(test)]
mod cancellation_tests {
    use super::*;

    #[test]
    fn entry_and_completion_cannot_release_a_single_leaf_after_expiry() {
        let root = [7; 32];
        assert_eq!(
            verify_multiproof_with_cancel(&root, 0, &[], &[], || true),
            Err(MultiproofError::Cancelled)
        );
        for stop in [1, 2] {
            let mut polls = 0;
            assert_eq!(
                verify_multiproof_with_cancel(&root, 1, &[(0, root)], &[], || {
                    polls += 1;
                    polls == stop
                }),
                Err(MultiproofError::Cancelled)
            );
            assert_eq!(polls, stop);
        }
    }

    #[test]
    fn validation_polls_before_a_late_duplicate() {
        let mut leaves: Vec<_> = (0..8192).map(|i| (i, [0; 32])).collect();
        leaves[4096].0 = 4095;
        assert_eq!(
            verify_multiproof(&[0; 32], 8192, &leaves, &[]),
            Err(MultiproofError::Unordered { position: 4096 })
        );
        let mut polls = 0;
        assert_eq!(
            verify_multiproof_with_cancel(&[0; 32], 8192, &leaves, &[], || {
                polls += 1;
                polls == 2
            }),
            Err(MultiproofError::Cancelled)
        );
        assert_eq!(polls, 2);
    }

    #[test]
    fn sibling_counting_and_climbing_poll_after_progress() {
        let leaves: Vec<_> = (0..8192).map(|i| (i, [0; 32])).collect();
        let mut active = || false;
        let layout = Layout::of(
            &leaves,
            13,
            &mut Work {
                remaining: 0,
                cancel: &mut active,
            },
        )
        .unwrap();
        assert_eq!(layout.total, 0);
        assert!(
            climb(
                &leaves,
                13,
                &[],
                &layout,
                &mut Work {
                    remaining: 0,
                    cancel: &mut active
                }
            )
            .is_ok()
        );
        for counting in [true, false] {
            let mut polls = 0;
            let mut cancel = || {
                polls += 1;
                polls == 2
            };
            let mut work = Work {
                remaining: 0,
                cancel: &mut cancel,
            };
            let error = if counting {
                Layout::of(&leaves, 13, &mut work).err()
            } else {
                climb(&leaves, 13, &[], &layout, &mut work).err()
            };
            assert_eq!(error, Some(MultiproofError::Cancelled));
            assert_eq!(polls, 2);
        }
    }
}
