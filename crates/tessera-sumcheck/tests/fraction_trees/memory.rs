// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0010/0018/0054: count the actual resident constructors and complete proofs.

use super::super::*;
use tessera_sumcheck::{
    ResidentTreeMemory, ResidentTreeShape as Shape, SumcheckTranscript, TreeScope, WindowError,
    prove_trees_in,
};

struct NoAllocationTranscript;
impl SumcheckTranscript<Fp2> for NoAllocationTranscript {
    fn absorb_message(&mut self, _: &str, _: &[Fp2]) {}
    fn challenge(&mut self, _: &str) -> Fp2 {
        Fp2::from(Fp::from_u64_reduced(17))
    }
}

fn check(trees: &[TestTree]) {
    let shapes: Vec<_> = trees
        .iter()
        .map(|tree| match tree {
            TestTree::Fraction(f) => Shape::Fraction {
                height: f.height(),
                base_numerators: f.numerators.is_some(),
            },
            TestTree::Forest(f) => Shape::Forest {
                height: f.height,
                trees_log: f.trees_log,
            },
        })
        .collect();
    let mut memory = None;
    let planning = allocation_counter::measure(|| {
        memory = Some(ResidentTreeMemory::new::<Fp2>(&shapes).unwrap());
    });
    assert_eq!(planning.count_total, 0, "planning allocates");
    let memory = memory.unwrap();

    // Caller-owned input vector is separate; its capacity is reserved before the
    // counter. The tree constructors own every allocation the counter observes.
    let mut built = Vec::with_capacity(trees.len());
    let construction = allocation_counter::measure(|| {
        built.extend(trees.iter().map(TestTree::tree));
    });
    assert_eq!(
        construction.bytes_current as u64,
        memory.fraction_bytes(),
        "canonical fraction storage {shapes:?}: {construction:?}"
    );
    assert!(
        construction.bytes_max as u64 <= memory.fraction_bytes(),
        "fraction constructor {shapes:?}: {construction:?} > {memory:?}"
    );
    drop(built);

    // Construct again inside the same measurement as proving: released input
    // trees cannot make a negative counter hide any peak or returned allocation.
    let mut proven = None;
    let execution = allocation_counter::measure(|| {
        let mut built = Vec::with_capacity(trees.len());
        built.extend(trees.iter().map(TestTree::tree));
        proven = Some(
            prove_trees_in(
                &mut NoAllocationTranscript,
                TreeScope::Group(u32::MAX),
                built,
            )
            .unwrap(),
        );
    });
    let input = (trees.len() * size_of::<Tree<'_, Fp2>>()) as u64;
    assert!(
        execution.bytes_max as u64 <= memory.peak_bytes() + input,
        "complete resident proof {shapes:?}: {execution:?} > {memory:?} + {input}"
    );
    assert!(
        execution.bytes_current as u64 <= memory.output_bytes(),
        "returned levels/claims {shapes:?}: {execution:?} > {memory:?}"
    );
    drop(proven);

    // A real transcript and independent tv-core/tree driver check these same
    // synthetic sources separately from the allocation-free transcript stub.
    let mut rng = SplitMix64::new(0x6730);
    drop(prove_and_verify(&Session::random(&mut rng), trees));
}

#[test]
fn public_resident_tree_memory_covers_fraction_construction_and_zero_roots() {
    let mut rng = SplitMix64::new(0x6731);
    check(&[]);
    for height in 0..=12 {
        for unit in [true, false] {
            check(&[TestTree::Fraction(TestFraction::random(
                &mut rng, height, unit,
            ))]);
        }
    }
    let roots: Vec<_> = (0..19)
        .map(|i| TestTree::Fraction(TestFraction::random(&mut rng, 0, i % 2 == 0)))
        .collect();
    check(&roots);
}

#[test]
fn public_resident_tree_memory_covers_stacked_forests_and_mixed_join_orders() {
    let mut rng = SplitMix64::new(0x6732);
    // The whole forest is one level: its owned base leaves remain live beside
    // the large folded extension operands, without freed parents masking them.
    check(&[TestTree::Forest(TestForest::random(&mut rng, 14, 1, false))]);
    for height in [1, 2, 5, 8] {
        for trees_log in [0, 1, 4] {
            check(&[TestTree::Forest(TestForest::random(
                &mut rng, trees_log, height, false,
            ))]);
        }
    }
    let mut trees = vec![
        TestTree::Fraction(TestFraction::random(&mut rng, 2, false)),
        TestTree::Forest(TestForest::random(&mut rng, 3, 6, false)),
        TestTree::Fraction(TestFraction::random(&mut rng, 4, true)),
        TestTree::Forest(TestForest::random(&mut rng, 4, 2, true)),
        TestTree::Fraction(TestFraction::random(&mut rng, 0, false)),
        TestTree::Forest(TestForest::random(&mut rng, 0, 1, false)),
        TestTree::Fraction(TestFraction::random(&mut rng, 5, false)),
    ];
    check(&trees);
    trees.reverse();
    check(&trees);
    trees.rotate_left(3);
    check(&trees);
}

#[cfg(target_pointer_width = "64")]
#[test]
fn public_resident_tree_memory_refuses_aggregate_overflow_without_field_tables() {
    // Each individual 2^50-leaf canonical tree fits the type's address limits;
    // their aggregate requested heap exceeds u64. Only public shapes are stored.
    let shapes = vec![
        Shape::Fraction {
            height: 50,
            base_numerators: true
        };
        512
    ];
    let planning = allocation_counter::measure(|| {
        assert_eq!(
            ResidentTreeMemory::new::<Fp2>(&shapes),
            Err(WindowError::Allocation)
        );
    });
    assert_eq!(planning.count_total, 0);
}

#[test]
fn public_resident_tree_memory_keeps_all_large_fraction_and_forest_owners() {
    let mut rng = SplitMix64::new(0x6733);
    let mut trees = Vec::new();
    for i in 0..17 {
        trees.push(TestTree::Fraction(TestFraction::random(
            &mut rng,
            10 + i % 3,
            i % 2 == 0,
        )));
        trees.push(TestTree::Forest(TestForest::random(
            &mut rng,
            i % 4,
            8,
            false,
        )));
    }
    check(&trees);
}

#[test]
fn public_resident_tree_memory_refuses_invalid_and_unaddressable_geometry() {
    for shape in [
        Shape::Forest {
            height: 0,
            trees_log: 1,
        },
        Shape::Forest {
            height: 1,
            trees_log: usize::MAX,
        },
        Shape::Fraction {
            height: usize::BITS as usize,
            base_numerators: false,
        },
    ] {
        let planning = allocation_counter::measure(|| {
            assert!(matches!(
                ResidentTreeMemory::new::<Fp2>(&[shape]),
                Err(WindowError::Shape(_))
            ));
        });
        assert_eq!(planning.count_total, 0);
    }
    for shape in [
        Shape::Fraction {
            height: usize::BITS as usize - 1,
            base_numerators: false,
        },
        Shape::Forest {
            height: 1,
            trees_log: usize::BITS as usize - 2,
        },
    ] {
        let planning = allocation_counter::measure(|| {
            assert_eq!(
                ResidentTreeMemory::new::<Fp2>(&[shape]),
                Err(WindowError::Allocation)
            );
        });
        assert_eq!(planning.count_total, 0);
    }
}
