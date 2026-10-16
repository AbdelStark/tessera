// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Honest forests through the independent verifier (DEP-2; RFC-0010 'Tree levels', 'Testing
//! Strategy'): every batch is proven by `prove_forests` and checked level by level by the driver
//! of `common::forests`, which squeezes the root challenges itself, forms the root claims from the
//! trees' true products and calls `tv_core::tree::verify_product_level` at each level. The
//! verifier must accept, its challenges and claims must be the prover's, the transcripts must end
//! in the same state, and each leaf claim must be the stacked leaves' multilinear extension at its
//! point.
//!
//! Forests have zero roots, a zero leaf in every tree as `SoftmaxRow`'s, or random ones.

mod common;

use common::forests::{TestForest, prove_and_verify};
use common::{Session, SplitMix64};

/// Every tree count from 1 to 2^10 and every height from 1 to 12: each shape with at most 2^13
/// leaves, alone in its batch, alternately with zero and random roots.
#[test]
fn forests_of_every_tree_count_and_height_verify() {
    let mut rng = SplitMix64::new(0x58_0001);
    let session = Session::random(&mut rng);
    let mut shapes = 0;
    for trees_log in 0..=10 {
        for height in 1..=12 {
            if trees_log + height > 13 {
                continue;
            }
            let forest = TestForest::random(&mut rng, trees_log, height, shapes % 2 == 0);
            let proven = prove_and_verify(&session, &[forest]);
            let first = &proven.levels()[0];
            assert_eq!(first.root_challenges().len(), trees_log);
            assert_eq!(first.point().len(), trees_log, "level 0 is over the roots");
            shapes += 1;
        }
    }
    assert_eq!(shapes, 87);
}

/// RFC-0010's largest case: 2^10 trees of 2^12 leaves, 2^22 leaves, each tree with a zero leaf.
#[test]
fn the_forest_of_2_10_trees_of_2_12_leaves_verifies() {
    let mut rng = SplitMix64::new(0x58_0002);
    let session = Session::random(&mut rng);
    let forest = TestForest::random(&mut rng, 10, 12, true);
    let proven = prove_and_verify(&session, &[forest]);
    assert_eq!(proven.levels().len(), 12);
    assert_eq!(proven.leaf_claims()[0].point.len(), 22);
}

/// Forests of one height join at one level and share its root challenges: with 2^0, 2^4, 2^2,
/// 2^4 and 2^1 trees, four are squeezed, the forests of fewer trees take the last of them, and
/// the single tree none.
#[test]
fn forests_joining_at_one_level_share_its_root_challenges() {
    let mut rng = SplitMix64::new(0x58_0003);
    let session = Session::random(&mut rng);
    let forests: Vec<TestForest> = [(0, 5), (4, 5), (2, 5), (4, 5), (1, 5)]
        .into_iter()
        .enumerate()
        .map(|(i, (s, h))| TestForest::random(&mut rng, s, h, i % 2 == 1))
        .collect();
    let proven = prove_and_verify(&session, &forests);
    let first = &proven.levels()[0];
    assert_eq!(first.root_challenges().len(), 4);
    assert_eq!(first.forests(), [0, 1, 2, 3, 4]);
    for level in &proven.levels()[1..] {
        assert_eq!(level.forests(), [0, 1, 2, 3, 4], "no forest joins late");
        assert!(level.root_challenges().is_empty());
    }
}

/// Forests of several heights join where their heights match, in the order given, front-padded:
/// heights 5, 9, 2, 9, 5, 1 and 3 join at levels 4, 0, 7, 0, 4, 8 and 6, with 2^3, 2^0, 2^6,
/// 2^2, 2^4, 2^0 and 2^5 trees. Two levels have joining forests of more than one tree, one only a
/// single tree.
#[test]
fn forests_of_several_heights_join_where_their_heights_match() {
    let mut rng = SplitMix64::new(0x58_0004);
    let session = Session::random(&mut rng);
    let shapes = [(3, 5), (0, 9), (6, 2), (2, 9), (4, 5), (0, 1), (5, 3)];
    let forests: Vec<TestForest> = shapes
        .into_iter()
        .enumerate()
        .map(|(i, (s, h))| TestForest::random(&mut rng, s, h, i % 3 != 1))
        .collect();
    let proven = prove_and_verify(&session, &forests);
    let roots: Vec<usize> = proven
        .levels()
        .iter()
        .map(|level| level.root_challenges().len())
        .collect();
    assert_eq!(roots, [2, 0, 0, 0, 4, 0, 5, 6, 0]);
    let present: Vec<&[usize]> = proven.levels().iter().map(|l| l.forests()).collect();
    assert_eq!(
        present,
        [
            &[1, 3][..],
            &[1, 3],
            &[1, 3],
            &[1, 3],
            &[0, 1, 3, 4],
            &[0, 1, 3, 4],
            &[0, 1, 3, 4, 6],
            &[0, 1, 2, 3, 4, 6],
            &[0, 1, 2, 3, 4, 5, 6],
        ]
    );
}

/// Seeded random batches: 1 to 5 forests of 2^0 to 2^6 trees of height 1 to 8, each with zero or
/// random roots, in random order.
#[test]
fn seeded_random_batches_verify() {
    let mut rng = SplitMix64::new(0x58_0005);
    let session = Session::random(&mut rng);
    for _ in 0..40 {
        let count = 1 + rng.below(5) as usize;
        let forests: Vec<TestForest> = (0..count)
            .map(|_| {
                let trees_log = rng.below(7) as usize;
                let height = 1 + rng.below(8) as usize;
                let zero_rows = rng.coin();
                TestForest::random(&mut rng, trees_log, height, zero_rows)
            })
            .collect();
        let _ = prove_and_verify(&session, &forests);
    }
}
