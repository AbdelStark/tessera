// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0008: compare every ordered node; root equality alone permits permutations.

use super::*;
use core::cell::RefCell;
use tessera_field::{Field, Goldilocks as B, GoldilocksExt2 as F, PrimeField, QuadraticExtension};

// The production scalar fallback is the unchanged pre-#663 traversal.

struct Source {
    height: usize,
    reads: RefCell<Vec<(usize, usize)>>,
    fail: Option<usize>,
    zero_denominators: bool,
}

impl FractionTreeSource<F> for Source {
    fn height(&self) -> usize {
        self.height
    }
    fn read(&self, first: usize, n: &mut [F], d: &mut [F]) -> Result<(), WindowError> {
        self.reads.borrow_mut().push((first, n.len()));
        if self.fail == Some(first) {
            return Err(WindowError::Source);
        }
        const VALUES: [u64; 8] = [
            0,
            1,
            0xffff_ffff_0000_0000,
            0xffff_ffff,
            0x1_0000_0000,
            0xffff_fffe_0000_0001,
            u64::MAX,
            0x1234_5678_abcd_ef01,
        ];
        for (i, (n, d)) in n.iter_mut().zip(d).enumerate() {
            let x = first + i;
            let b = |j: usize| B::from_u64_reduced(VALUES[j % VALUES.len()]);
            *n = F::new(b(x), b(x / 8 + 3));
            *d = if self.zero_denominators && x % 19 == 7 {
                F::ZERO
            } else {
                F::new(b(x * 3 + 1), b(x / 3 + 5))
            };
        }
        Ok(())
    }
}

fn source(height: usize) -> Source {
    Source {
        height,
        reads: RefCell::new(Vec::new()),
        fail: None,
        zero_denominators: false,
    }
}

#[test]
fn cached_intermediate_levels_preserve_ordered_nodes_and_original_descendants() {
    for height in 2..=7 {
        for cache_depth in 1..height {
            for chunk in [1, 3, 32, 65] {
                for zeros in [false, true] {
                    let mut source = source(height);
                    source.zero_denominators = zeros;
                    let reference =
                        FractionTreeRows::new(&source, NonZeroUsize::new(chunk).unwrap(), u64::MAX)
                            .unwrap();
                    let mut cached =
                        FractionTreeRows::new(&source, NonZeroUsize::new(chunk).unwrap(), u64::MAX)
                            .unwrap()
                            .with_cache_depth(cache_depth, u64::MAX)
                            .unwrap();
                    assert!(source.reads.borrow().is_empty());
                    cached.prepare_cache().unwrap();
                    let constructed = source.reads.replace(Vec::new());
                    let expected: Vec<_> = (0..1usize << height)
                        .step_by(chunk)
                        .map(|first| (first, chunk.min((1usize << height) - first)))
                        .collect();
                    assert_eq!(constructed, expected, "every original leaf is read once");
                    assert_eq!(cached.root().unwrap(), reference.root().unwrap());
                    for depth in 0..=height {
                        let nodes = 1usize << depth;
                        for first in [0, 1.min(nodes), nodes / 2, nodes.saturating_sub(3), nodes] {
                            for count in
                                [0, 1.min(nodes - first), 3.min(nodes - first), nodes - first]
                            {
                                let mut n = SecretBuf::filled(F::ZERO, count);
                                let mut d = SecretBuf::filled(F::ZERO, count);
                                let mut expected_n = SecretBuf::filled(F::ZERO, count);
                                let mut expected_d = SecretBuf::filled(F::ZERO, count);
                                source.reads.borrow_mut().clear();
                                cached.read_level(depth, first, &mut n, &mut d).unwrap();
                                let reads = source.reads.replace(Vec::new());
                                reference
                                    .read_level(depth, first, &mut expected_n, &mut expected_d)
                                    .unwrap();
                                assert!(n[..] == expected_n[..] && d[..] == expected_d[..]);
                                if depth <= cache_depth {
                                    assert!(reads.is_empty(), "ancestors reuse the prepared level");
                                } else {
                                    assert_eq!(reads, *source.reads.borrow());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn cache_geometry_and_allowance_refuse_without_reads_or_allocations() {
    let source = source(8);
    let rows = || FractionTreeRows::new(&source, NonZeroUsize::new(17).unwrap(), u64::MAX).unwrap();
    for depth in [0, 8, 9, usize::MAX] {
        assert!(matches!(
            rows().with_cache_depth(depth, u64::MAX),
            Err(WindowError::Shape(_))
        ));
    }
    let needed = rows().scratch_bytes() + (1 << 6) * 2 * size_of::<F>() as u64;
    let measured = allocation_counter::measure(|| {
        assert!(matches!(
            rows().with_cache_depth(6, needed - 1),
            Err(WindowError::Allocation)
        ));
    });
    assert_eq!(measured.count_total, 0);
    assert!(source.reads.borrow().is_empty());
    assert_eq!(
        rows().with_cache_depth(6, needed).unwrap().retained_bytes(),
        2048
    );
    let huge = self::source(usize::BITS as usize - 1);
    let measured = allocation_counter::measure(|| {
        let rows = FractionTreeRows::new(&huge, NonZeroUsize::new(1).unwrap(), u64::MAX).unwrap();
        assert!(matches!(
            rows.with_cache_depth(usize::BITS as usize - 2, u64::MAX),
            Err(WindowError::Allocation)
        ));
    });
    assert_eq!(measured.count_total, 0);
    assert!(huge.reads.borrow().is_empty());
}

#[test]
fn cached_levels_match_the_scalar_fold_over_every_supported_field() {
    struct Generic;
    impl<T: PolyField> FractionTreeSource<T> for Generic {
        fn height(&self) -> usize {
            7
        }
        fn read(&self, first: usize, n: &mut [T], d: &mut [T]) -> Result<(), WindowError> {
            for (i, (n, d)) in n.iter_mut().zip(d).enumerate() {
                let at = (first + i) as u64;
                *n = T::from(T::Base::from_u64_reduced(u64::MAX - at));
                *d = T::from(T::Base::from_u64_reduced(at + 1));
            }
            Ok(())
        }
    }
    fn check<T: PolyField>() {
        for chunk in [1, 3, 32, 65] {
            let mut cached =
                FractionTreeRows::<T>::new(&Generic, NonZeroUsize::new(chunk).unwrap(), u64::MAX)
                    .unwrap()
                    .with_cache_depth(4, u64::MAX)
                    .unwrap();
            cached.prepare_cache().unwrap();
            for depth in 0..=7 {
                let mut n = SecretBuf::filled(T::ZERO, 1 << depth);
                let mut d = SecretBuf::filled(T::ZERO, 1 << depth);
                cached.read_level(depth, 0, &mut n, &mut d).unwrap();
                let mut expected = Vec::new();
                Scratch::<T>::new(7 - depth, 3)
                    .unwrap()
                    .subtrees_scalar(&Generic, 0, 7 - depth, 1 << depth, |_, n, d| {
                        expected.push((n, d))
                    })
                    .unwrap();
                for (at, &(en, ed)) in expected.iter().enumerate() {
                    assert_eq!((n[at], d[at]), (en, ed));
                }
            }
        }
    }
    check::<B>();
    check::<F>();
    check::<tessera_field::F65537>();
    check::<tessera_field::F65537Ext2>();
}

#[test]
fn descendants_keep_original_high_leaf_offsets_with_an_installed_cache() {
    if usize::BITS < 64 {
        return;
    }
    let source = source(32);
    let mut rows = FractionTreeRows::new(&source, NonZeroUsize::new(3).unwrap(), u64::MAX)
        .unwrap()
        .with_cache_depth(1, u64::MAX)
        .unwrap();
    // No 2^32 scan: this white-box cache may only serve depth <= 1.
    rows.cache = Some(CachedLevel {
        depth: 1,
        numerators: SecretBuf::filled(F::ZERO, 2),
        denominators: SecretBuf::filled(F::ZERO, 2),
    });
    let first = usize::try_from((1u64 << 32) - 5).unwrap();
    let mut n = SecretBuf::filled(F::ZERO, 5);
    let mut d = SecretBuf::filled(F::ZERO, 5);
    rows.read_level(32, first, &mut n, &mut d).unwrap();
    assert_eq!(*source.reads.borrow(), [(first, 3), (first + 3, 2)]);
    let mut en = SecretBuf::filled(F::ZERO, 5);
    let mut ed = SecretBuf::filled(F::ZERO, 5);
    source.read(first, &mut en, &mut ed).unwrap();
    assert!(n[..] == en[..] && d[..] == ed[..]);
}

#[test]
fn cache_construction_and_error_drop_stay_in_clearing_allocation_bounds() {
    for fail in [None, Some(17)] {
        let mut source = source(10);
        source.fail = fail;
        source.reads.borrow_mut().reserve(1024);
        let mut rows = FractionTreeRows::new(&source, NonZeroUsize::new(17).unwrap(), u64::MAX)
            .unwrap()
            .with_cache_depth(6, u64::MAX)
            .unwrap();
        let bound = rows.scratch_bytes() + rows.retained_bytes();
        let measured = allocation_counter::measure(|| {
            let result = rows.prepare_cache();
            if fail.is_some() {
                assert_eq!(result, Err(WindowError::Source));
                assert!(rows.cache.is_none());
            } else {
                result.unwrap();
                assert!(rows.cache.is_some());
            }
            drop(rows);
        });
        assert_eq!(measured.bytes_current, 0);
        assert!(measured.bytes_max as u64 <= bound);
        assert_eq!(measured.count_total, 5);
    }
}

#[test]
fn every_subrange_matches_scalar_nodes_and_exact_source_reads() {
    for height in 0..=7 {
        let source = source(height);
        for chunk in [1, 2, 3, 5, 8, 17, 31, 64, 256] {
            let rows = FractionTreeRows::new(&source, NonZeroUsize::new(chunk).unwrap(), u64::MAX)
                .unwrap();
            for depth in 0..=height {
                let nodes = 1usize << depth;
                let below = height - depth;
                for first in 0..=nodes {
                    for count in 0..=nodes - first {
                        let mut n = SecretBuf::filled(F::ZERO, count);
                        let mut d = SecretBuf::filled(F::ZERO, count);
                        source.reads.borrow_mut().clear();
                        rows.read_level(depth, first, &mut n, &mut d).unwrap();
                        // Multiple zero leaves collapse upper nodes to (0, 0), hiding reorderings.
                        assert!(d.iter().all(|&value| value != F::ZERO));
                        let reads = source.reads.replace(Vec::new());
                        let mut expected = Vec::new();
                        if count != 0 {
                            let mut scalar =
                                Scratch::new(below, chunk.min(count << below)).unwrap();
                            scalar
                                .subtrees_scalar(
                                    &source,
                                    first << below,
                                    below,
                                    count,
                                    |_, n, d| expected.push((n, d)),
                                )
                                .unwrap();
                        }
                        assert_eq!(
                            reads,
                            *source.reads.borrow(),
                            "h={height} chunk={chunk} depth={depth} first={first} count={count}"
                        );
                        assert!(reads.iter().all(|&(_, n)| n <= chunk));
                        assert_eq!(
                            n.iter().copied().zip(d.iter().copied()).collect::<Vec<_>>(),
                            expected,
                            "h={height} chunk={chunk} depth={depth} first={first} count={count}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn source_failure_after_a_compacted_chunk_discards_the_pass() {
    for chunk in [3, 8, 17, 256] {
        let mut source = source(10);
        source.fail = Some(chunk);
        let rows =
            FractionTreeRows::new(&source, NonZeroUsize::new(chunk).unwrap(), u64::MAX).unwrap();
        assert_eq!(rows.root(), Err(WindowError::Source));
        assert_eq!(*source.reads.borrow(), vec![(0, chunk), (chunk, chunk)]);
        source.reads.borrow_mut().clear();
        let mut n = SecretBuf::filled(F::ZERO, 2);
        let mut d = SecretBuf::filled(F::ZERO, 2);
        assert_eq!(
            rows.read_level(1, 0, &mut n, &mut d),
            Err(WindowError::Source)
        );
        assert_eq!(*source.reads.borrow(), vec![(0, chunk), (chunk, chunk)]);
    }
}

#[test]
fn compaction_uses_only_the_existing_clearing_buffers() {
    struct NoAlloc;
    impl FractionTreeSource<F> for NoAlloc {
        fn height(&self) -> usize {
            16
        }
        fn read(&self, first: usize, n: &mut [F], d: &mut [F]) -> Result<(), WindowError> {
            for (i, (n, d)) in n.iter_mut().zip(d).enumerate() {
                *n = F::from(B::from_u64_reduced((first + i) as u64));
                *d = F::ONE + *n;
            }
            Ok(())
        }
    }
    for chunk in [1, 3, 17, 256, 4096] {
        let rows =
            FractionTreeRows::new(&NoAlloc, NonZeroUsize::new(chunk).unwrap(), u64::MAX).unwrap();
        let mut n = SecretBuf::filled(F::ZERO, 19);
        let mut d = SecretBuf::filled(F::ZERO, 19);
        let measured = allocation_counter::measure(|| {
            rows.read_level(8, 7, &mut n, &mut d).unwrap();
        });
        assert!(measured.bytes_max as u64 <= rows.scratch_bytes());
        assert_eq!(
            measured.count_total, 3,
            "only the original numerator/denominator/stack allocations"
        );
        assert_eq!(measured.bytes_current, 0);
    }
}

#[test]
fn zero_denominators_are_preserved_in_every_ordered_node() {
    let mut source = source(8);
    source.zero_denominators = true;
    for chunk in [1, 3, 17, 64, 256] {
        let rows =
            FractionTreeRows::new(&source, NonZeroUsize::new(chunk).unwrap(), u64::MAX).unwrap();
        for depth in 0..=8 {
            let count = 1usize << depth;
            let below = 8 - depth;
            let mut n = SecretBuf::filled(F::ZERO, count);
            let mut d = SecretBuf::filled(F::ZERO, count);
            rows.read_level(depth, 0, &mut n, &mut d).unwrap();
            let mut expected = Vec::new();
            Scratch::new(below, chunk.min(count << below))
                .unwrap()
                .subtrees_scalar(&source, 0, below, count, |_, n, d| expected.push((n, d)))
                .unwrap();
            assert_eq!(
                n.iter().copied().zip(d.iter().copied()).collect::<Vec<_>>(),
                expected
            );
            source.reads.borrow_mut().clear();
        }
    }
}

#[test]
fn maximum_chunks_preserve_deep_nodes_reads_and_allocation_bounds() {
    struct SeededSource {
        height: usize,
        reads: RefCell<Vec<(usize, usize)>>,
    }
    fn mix(mut x: u64) -> B {
        x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        B::from_u64_reduced(x ^ (x >> 31))
    }
    impl FractionTreeSource<F> for SeededSource {
        fn height(&self) -> usize {
            self.height
        }
        fn read(&self, first: usize, n: &mut [F], d: &mut [F]) -> Result<(), WindowError> {
            if n.len() != d.len()
                || first
                    .checked_add(n.len())
                    .is_none_or(|end| end > 1usize << self.height)
            {
                return Err(WindowError::Source);
            }
            self.reads.borrow_mut().push((first, n.len()));
            for (i, (n, d)) in n.iter_mut().zip(d).enumerate() {
                let x = (first + i) as u64;
                *n = F::new(mix(x), mix(x ^ 0x6630_6650));
                // The extension coefficient is one, so no leaf or ancestor denominator is zero.
                *d = F::new(mix(x ^ 0xa5a5_5a5a), B::ONE);
            }
            Ok(())
        }
    }
    const CHUNK: usize = 65_536;
    let mut cases = vec![
        (22, 0, 0, 1),       // Complete 2^22-leaf root and carries above the chunk.
        (22, 4, 3, 11),      // Several chunks per output node.
        (22, 8, 5, 19),      // Several output nodes per chunk, then a ragged tail.
        (22, 16, 123, 1027), // 1,024 nodes in the full chunk, then three more.
        (22, 22, (1usize << 22) - 19, 19), // Original direct-leaf path at the domain's end.
    ];
    if usize::BITS == 64 {
        // Only the final 65,538 leaves, never a full 2^32-leaf traversal or allocation.
        cases.push((32, 31, (1usize << 31) - 32_769, 32_769));
    }
    for (height, depth, first, count) in cases {
        let source = SeededSource {
            height,
            reads: RefCell::new(Vec::with_capacity(128)),
        };
        let rows =
            FractionTreeRows::new(&source, NonZeroUsize::new(CHUNK).unwrap(), u64::MAX).unwrap();
        let mut n = SecretBuf::filled(F::ZERO, count);
        let mut d = SecretBuf::filled(F::ZERO, count);
        // The public observer backing and caller outputs are preallocated outside this scope.
        let measured =
            allocation_counter::measure(|| rows.read_level(depth, first, &mut n, &mut d).unwrap());
        assert!(measured.bytes_max as u64 <= rows.scratch_bytes());
        assert_eq!(measured.bytes_current, 0);
        assert_eq!(measured.count_total, if depth == height { 0 } else { 3 });
        assert!(d.iter().all(|&x| x != F::ZERO));
        let reads = source.reads.borrow().clone();
        source.reads.borrow_mut().clear();
        let below = height - depth;
        let mut expected = Vec::new();
        Scratch::new(below, CHUNK.min(count << below))
            .unwrap()
            .subtrees_scalar(&source, first << below, below, count, |_, n, d| {
                expected.push((n, d))
            })
            .unwrap();
        // Pin the fixture's ability to expose an output permutation.
        assert!(expected.windows(2).all(|pair| pair[0] != pair[1]));
        assert_eq!(
            reads,
            *source.reads.borrow(),
            "height={height} depth={depth}"
        );
        assert!(reads.iter().all(|&(_, len)| len <= CHUNK));
        assert_eq!(
            n.iter().copied().zip(d.iter().copied()).collect::<Vec<_>>(),
            expected,
            "height={height} depth={depth}"
        );
        if depth == 0 {
            source.reads.borrow_mut().clear();
            assert_eq!(rows.root().unwrap(), expected[0]);
            assert_eq!(reads, *source.reads.borrow());
        }
    }
}
