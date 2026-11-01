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
