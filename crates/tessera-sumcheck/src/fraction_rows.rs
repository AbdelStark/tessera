// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Bounded regeneration of canonical fraction-tree nodes (RFC-0008/0054).

use core::{fmt, mem::size_of, num::NonZeroUsize};

use tessera_codec::SecretBuf;
use tessera_poly::PolyField;

use crate::WindowError;

/// Immutable canonical fraction leaves, including the table tree's row-0 padding.
///
/// Every pass must reproduce the same leaves. Secret-derived entries are written directly
/// into the caller's clearing storage; a source retains no borrowed destination. Source
/// backing storage and any public generator core are charged separately from regeneration.
pub trait FractionTreeSource<F: PolyField> {
    /// Height of the tree: exactly `2^height` leaves, lowest pair bit first.
    fn height(&self) -> usize;

    /// Additional temporary allocation bound for any source read of at most `rows` leaves.
    /// Sources that only copy/compute directly into the destinations need no extra storage.
    /// `None` means the public geometry cannot be represented. Backing storage is separate.
    fn read_scratch_bytes(&self, _rows: usize) -> Option<u64> {
        Some(0)
    }

    /// Fill equally sized consecutive numerator/denominator chunks starting at `first`.
    /// Control flow and scratch capacity depend on public geometry only.
    ///
    /// # Errors
    /// A public geometry, availability or allocation error. An error may leave a partial
    /// destination; the caller must discard that clearing buffer and the errored pass.
    fn read(
        &self,
        first: usize,
        numerators: &mut [F],
        denominators: &mut [F],
    ) -> Result<(), WindowError>;
}

/// A fraction tree's rows, regenerated from bounded leaf chunks rather than retained levels.
///
/// Construction validates public geometry/allowance only and reads no leaf. This descriptor
/// does not prove a tree or discharge a claim. The caller still owes every tree-level proof,
/// the original root/identity checks, and the multiplicity/original-opening obligations.
pub struct FractionTreeRows<'a, F: PolyField> {
    source: &'a dyn FractionTreeSource<F>,
    height: usize,
    leaves: usize,
    chunk: usize,
    scratch_bytes: u64,
    cache_depth: Option<usize>,
    cache_bytes: u64,
    cache: Option<CachedLevel<F>>,
}

struct CachedLevel<F: PolyField> {
    depth: usize,
    numerators: SecretBuf<F>,
    denominators: SecretBuf<F>,
}

impl<F: PolyField> FractionTreeSource<F> for CachedLevel<F> {
    fn height(&self) -> usize {
        self.depth
    }

    fn read(&self, first: usize, n: &mut [F], d: &mut [F]) -> Result<(), WindowError> {
        let end = first.checked_add(n.len()).ok_or(WindowError::Source)?;
        if n.len() != d.len() || end > self.numerators.len() {
            return Err(WindowError::Source);
        }
        n.copy_from_slice(&self.numerators[first..end]);
        d.copy_from_slice(&self.denominators[first..end]);
        Ok(())
    }
}

impl<F: PolyField> fmt::Debug for FractionTreeRows<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FractionTreeRows")
            .field("height", &self.height)
            .field("chunk", &self.chunk)
            .field("scratch_bytes", &self.scratch_bytes)
            .field("cache_depth", &self.cache_depth)
            .finish_non_exhaustive()
    }
}

struct Scratch<F: PolyField> {
    numerators: SecretBuf<F>,
    denominators: SecretBuf<F>,
    stack: SecretBuf<F>,
}

fn buffer<F: PolyField>(len: usize) -> Result<SecretBuf<F>, WindowError> {
    let mut out = SecretBuf::try_with_capacity(len).map_err(|_| WindowError::Allocation)?;
    out.resize(len, F::ZERO);
    Ok(out)
}

impl<F: PolyField> Scratch<F> {
    fn new(height: usize, chunk: usize) -> Result<Self, WindowError> {
        Ok(Self {
            numerators: buffer(chunk)?,
            denominators: buffer(chunk)?,
            stack: buffer(2 * (height + 1))?,
        })
    }

    fn subtree(
        &mut self,
        source: &dyn FractionTreeSource<F>,
        first: usize,
        height: usize,
    ) -> Result<(F, F), WindowError> {
        let mut root = (F::ZERO, F::ZERO);
        self.subtrees(source, first, height, 1, |_, n, d| root = (n, d))?;
        Ok(root)
    }

    fn subtrees(
        &mut self,
        source: &dyn FractionTreeSource<F>,
        first: usize,
        height: usize,
        nodes: usize,
        mut emit: impl FnMut(usize, F, F),
    ) -> Result<(), WindowError> {
        // Preserve the original traversal where chunk setup costs more than compaction.
        if self.numerators.len() < 32 || !self.numerators.len().is_power_of_two() {
            return self.subtrees_scalar(source, first, height, nodes, emit);
        }
        self.stack.fill(F::ZERO);
        let count = nodes << height;
        let mut consumed = 0;
        while consumed < count {
            let take = self.numerators.len().min(count - consumed);
            source.read(
                first + consumed,
                &mut self.numerators[..take],
                &mut self.denominators[..take],
            )?;
            // RFC-0008: compact only aligned whole subtrees inside this read.
            // Ragged reads retain the scalar traversal, and no fold crosses an output node.
            let compact = if take.is_power_of_two() && consumed.is_multiple_of(take) {
                height.min(take.ilog2() as usize)
            } else {
                0
            };
            let mut width = take;
            for _ in 0..compact {
                for pair in 0..width / 2 {
                    let left = 2 * pair;
                    let right = left + 1;
                    self.numerators[pair] = self.numerators[left] * self.denominators[right]
                        + self.numerators[right] * self.denominators[left];
                    self.denominators[pair] = self.denominators[left] * self.denominators[right];
                }
                width /= 2;
            }
            for row in 0..width {
                let index = consumed + (row << compact);
                let mut numerator = self.numerators[row];
                let mut denominator = self.denominators[row];
                let mut level = compact;
                // Public binary carries combine adjacent subcubes in canonical leaf order.
                while level < height && (index >> level) & 1 == 1 {
                    let left_n = self.stack[2 * level];
                    let left_d = self.stack[2 * level + 1];
                    numerator = left_n * denominator + numerator * left_d;
                    denominator *= left_d;
                    level += 1;
                }
                self.stack[2 * level] = numerator;
                self.stack[2 * level + 1] = denominator;
                if level == height {
                    emit(index >> height, numerator, denominator);
                }
            }
            consumed += take;
        }
        Ok(())
    }

    fn subtrees_scalar(
        &mut self,
        source: &dyn FractionTreeSource<F>,
        first: usize,
        height: usize,
        nodes: usize,
        mut emit: impl FnMut(usize, F, F),
    ) -> Result<(), WindowError> {
        self.stack.fill(F::ZERO);
        let count = nodes << height;
        let mut consumed = 0;
        while consumed < count {
            let take = self.numerators.len().min(count - consumed);
            source.read(
                first + consumed,
                &mut self.numerators[..take],
                &mut self.denominators[..take],
            )?;
            for row in 0..take {
                let index = consumed + row;
                let mut numerator = self.numerators[row];
                let mut denominator = self.denominators[row];
                let mut level = 0;
                // Public binary carries combine adjacent subcubes in canonical leaf order.
                while level < height && (index >> level) & 1 == 1 {
                    let left_n = self.stack[2 * level];
                    let left_d = self.stack[2 * level + 1];
                    numerator = left_n * denominator + numerator * left_d;
                    denominator *= left_d;
                    level += 1;
                }
                self.stack[2 * level] = numerator;
                self.stack[2 * level + 1] = denominator;
                if level == height {
                    emit(index >> height, numerator, denominator);
                }
            }
            consumed += take;
        }
        Ok(())
    }
}

impl<'a, F: PolyField> FractionTreeRows<'a, F> {
    /// Validate a source and the complete regeneration scratch allowance, without reading it.
    ///
    /// The allowance includes this descriptor, three clearing buffer descriptors, at most
    /// `2*min(chunk, leaves)` field elements and a `2*(height+1)`-element carry stack.
    /// Source backing storage and caller-owned output chunks are additional working sets.
    ///
    /// # Errors
    /// `Shape` when the leaf count does not fit `usize`; `Allocation` for an overflowing or
    /// insufficient scratch allowance. No source read or allocation occurs on these errors.
    pub fn new(
        source: &'a dyn FractionTreeSource<F>,
        chunk: NonZeroUsize,
        allowance: u64,
    ) -> Result<Self, WindowError> {
        let height = source.height();
        let leaves = u32::try_from(height)
            .ok()
            .and_then(|height| 1usize.checked_shl(height))
            .ok_or(WindowError::Shape("fraction leaves fit usize"))?;
        let chunk = chunk.get().min(leaves);
        let owned = chunk
            .checked_add(height + 1)
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| n.checked_mul(size_of::<F>()))
            .and_then(|n| n.checked_add(size_of::<Scratch<F>>()))
            .and_then(|n| n.checked_add(size_of::<Self>()))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WindowError::Allocation)?;
        let scratch_bytes = source
            .read_scratch_bytes(chunk)
            .and_then(|source| owned.checked_add(source))
            .filter(|&n| n <= allowance)
            .ok_or(WindowError::Allocation)?;
        Ok(Self {
            source,
            height,
            leaves,
            chunk,
            scratch_bytes,
            cache_depth: None,
            cache_bytes: 0,
            cache: None,
        })
    }

    /// Request one intermediate level, chosen only from public geometry and budget.
    ///
    /// This validates capacity without reading or allocating. [`crate::PreparedTreeBatch`]
    /// charges every requested cache together, then constructs them before proving.
    /// Ancestor reads reuse that level; descendant reads retain the original leaf source.
    /// Outside a prepared batch, reads remain correct through the uncached path.
    ///
    /// # Errors
    /// `Shape` unless `0 < depth < height`; `Allocation` for overflowing capacity or when
    /// the retained payload and original regeneration scratch exceed `allowance`.
    pub fn with_cache_depth(mut self, depth: usize, allowance: u64) -> Result<Self, WindowError> {
        if depth == 0 || depth >= self.height {
            return Err(WindowError::Shape(
                "fraction cache is an intermediate level",
            ));
        }
        let bytes = 1usize
            .checked_shl(u32::try_from(depth).map_err(|_| WindowError::Allocation)?)
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| n.checked_mul(size_of::<F>()))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WindowError::Allocation)?;
        self.scratch_bytes
            .checked_add(bytes)
            .filter(|&n| n <= allowance)
            .ok_or(WindowError::Allocation)?;
        self.cache_depth = Some(depth);
        self.cache_bytes = bytes;
        self.cache = None;
        Ok(self)
    }

    /// Retained cache payload charged for the complete batch, even before construction.
    /// Its descriptors are already included in [`Self::scratch_bytes`].
    pub fn retained_bytes(&self) -> u64 {
        self.cache_bytes
    }

    pub(crate) fn prepare_cache(&mut self) -> Result<(), WindowError> {
        let Some(depth) = self.cache_depth else {
            return Ok(());
        };
        if self.cache.is_some() {
            return Ok(());
        }
        let nodes = 1usize << depth;
        let mut numerators = buffer(nodes)?;
        let mut denominators = buffer(nodes)?;
        // RFC-0008: every original leaf contributes, including canonical padding/zero counts.
        // The cache is installed only after the complete original-source pass succeeds.
        self.read_level(depth, 0, &mut numerators, &mut denominators)?;
        self.cache = Some(CachedLevel {
            depth,
            numerators,
            denominators,
        });
        Ok(())
    }

    fn cached_rows<'s>(
        &self,
        cache: &'s CachedLevel<F>,
    ) -> Result<FractionTreeRows<'s, F>, WindowError> {
        FractionTreeRows::new(
            cache,
            NonZeroUsize::new(self.chunk).ok_or(WindowError::Allocation)?,
            self.scratch_bytes,
        )
    }

    /// Tree height, counted from the root to the leaves.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Logical leaf count; optional caching retains intermediate nodes, never these leaves.
    pub fn leaves(&self) -> usize {
        self.leaves
    }

    /// Bound on this descriptor and its maximum owned regeneration scratch, in bytes.
    /// Retained cache payload is additional and reported by [`Self::retained_bytes`].
    pub fn scratch_bytes(&self) -> u64 {
        self.scratch_bytes
    }

    /// Largest leaf chunk requested from the source; also bounds level-adapter chunks.
    pub fn chunk_rows(&self) -> usize {
        self.chunk
    }

    /// Compute the root in one leaf scan, with clearing chunk/stack scratch.
    /// Zero denominators are retained: the existing verifier must still check them.
    ///
    /// # Errors
    /// An allocation or source error. A failed pass is discarded.
    pub fn root(&self) -> Result<(F, F), WindowError> {
        if let Some(cache) = &self.cache {
            return self.cached_rows(cache)?.root();
        }
        Scratch::new(self.height, self.chunk)?.subtree(self.source, 0, self.height)
    }

    /// Regenerate consecutive nodes at `depth` from the root (`height` names the leaves).
    /// Every node combines its original contiguous leaf subcube, lowest pair bit first.
    /// Source requests never exceed the configured chunk size. The destinations are lent
    /// clearing buffers whose capacity the caller charges beside [`Self::scratch_bytes`].
    ///
    /// # Errors
    /// `Shape` before any write/read/allocation if depth, lengths or range disagree; otherwise
    /// an allocation/source error, after which the caller discards the destinations/pass.
    pub fn read_level(
        &self,
        depth: usize,
        first: usize,
        numerators: &mut [F],
        denominators: &mut [F],
    ) -> Result<(), WindowError> {
        let fits = depth <= self.height
            && numerators.len() == denominators.len()
            && first
                .checked_add(numerators.len())
                .is_some_and(|end| end <= 1usize << depth);
        if !fits {
            return Err(WindowError::Shape(
                "fraction level read fits its public domain",
            ));
        }
        if numerators.is_empty() {
            return Ok(());
        }
        if let Some(cache) = &self.cache
            && depth <= cache.depth
        {
            return self
                .cached_rows(cache)?
                .read_level(depth, first, numerators, denominators);
        }
        if depth == self.height {
            for (index, (n, d)) in numerators
                .chunks_mut(self.chunk)
                .zip(denominators.chunks_mut(self.chunk))
                .enumerate()
            {
                self.source.read(first + index * self.chunk, n, d)?;
            }
            return Ok(());
        }
        let below = self.height - depth;
        let count = numerators.len();
        // Public subtree boundaries do not shorten reads: carry across chunks and emit
        // each original subtree separately, with no retained level or skipped leaf.
        let mut scratch = Scratch::new(below, self.chunk.min(count << below))?;
        scratch.subtrees(self.source, first << below, below, count, |index, n, d| {
            numerators[index] = n;
            denominators[index] = d;
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::Cell;
    use tessera_field::{Field, Goldilocks as G, GoldilocksExt2 as F, PrimeField};

    struct Source {
        height: usize,
        calls: Cell<usize>,
        largest: Cell<usize>,
        fail: bool,
    }
    impl FractionTreeSource<F> for Source {
        fn height(&self) -> usize {
            self.height
        }
        fn read(&self, first: usize, n: &mut [F], d: &mut [F]) -> Result<(), WindowError> {
            self.calls.set(self.calls.get() + 1);
            self.largest.set(self.largest.get().max(n.len()));
            if self.fail {
                return Err(WindowError::Source);
            }
            for (j, (n, d)) in n.iter_mut().zip(d).enumerate() {
                *n = F::from(G::from_u64_reduced((first + j) as u64));
                *d = F::ONE + *n;
            }
            Ok(())
        }
    }
    fn source(height: usize) -> Source {
        Source {
            height,
            calls: Cell::new(0),
            largest: Cell::new(0),
            fail: false,
        }
    }

    #[test]
    fn adjacent_subtrees_fill_chunks_across_their_public_boundaries() {
        use crate::{FractionTree, LeafNumerators};
        for below in [1, 2, 4, 8] {
            let input = source(10);
            let chunk = 17;
            let depth = input.height - below;
            let first = 1;
            let count = 13.min((1usize << depth) - first);
            let rows =
                FractionTreeRows::new(&input, NonZeroUsize::new(chunk).unwrap(), u64::MAX).unwrap();
            let mut n = SecretBuf::filled(F::ZERO, count);
            let mut d = SecretBuf::filled(F::ZERO, count);
            let measured = allocation_counter::measure(|| {
                rows.read_level(depth, first, &mut n, &mut d).unwrap();
            });
            assert_eq!(input.calls.get(), (count << below).div_ceil(chunk));
            assert!(input.largest.get() <= chunk);
            assert!(measured.bytes_max as u64 <= rows.scratch_bytes());
            assert_eq!(measured.bytes_current, 0);
            for node in 0..count {
                let start = (first + node) << below;
                let numerators: Vec<_> = (start..start + (1 << below))
                    .map(|i| G::from_u64_reduced(i as u64))
                    .collect();
                let expected =
                    FractionTree::<F>::new(below, LeafNumerators::Base(&numerators), |i| {
                        F::from(G::from_u64_reduced((start + i + 1) as u64))
                    })
                    .unwrap();
                assert_eq!((n[node], d[node]), expected.root());
            }
        }
    }

    #[test]
    fn invalid_geometry_and_allowance_never_read_the_source() {
        let huge = source(usize::BITS as usize);
        assert!(matches!(
            FractionTreeRows::new(&huge, NonZeroUsize::MIN, u64::MAX),
            Err(WindowError::Shape("fraction leaves fit usize"))
        ));
        assert_eq!(huge.calls.get(), 0);
        let valid = source(3);
        let rows = FractionTreeRows::new(&valid, NonZeroUsize::new(3).unwrap(), u64::MAX).unwrap();
        assert!(matches!(
            FractionTreeRows::new(
                &valid,
                NonZeroUsize::new(3).unwrap(),
                rows.scratch_bytes() - 1
            ),
            Err(WindowError::Allocation)
        ));
        let mut n = SecretBuf::filled(F::ONE, 2);
        let mut d = SecretBuf::filled(F::ONE, 2);
        for (depth, first, len) in [(4, 0, 2), (3, 7, 2), (3, usize::MAX, 2), (3, 0, 1)] {
            assert_eq!(
                rows.read_level(depth, first, &mut n[..len], &mut d),
                Err(WindowError::Shape(
                    "fraction level read fits its public domain"
                ))
            );
            assert!(n.iter().chain(d.iter()).all(|v| *v == F::ONE));
        }
        assert_eq!(valid.calls.get(), 0);
    }

    #[test]
    fn source_failure_is_fatal_in_root_and_level_regeneration() {
        let mut s = source(3);
        s.fail = true;
        let rows = FractionTreeRows::new(&s, NonZeroUsize::new(3).unwrap(), u64::MAX).unwrap();
        assert_eq!(rows.root(), Err(WindowError::Source));
        let mut n = SecretBuf::filled(F::ZERO, 1);
        let mut d = SecretBuf::filled(F::ZERO, 1);
        assert_eq!(
            rows.read_level(1, 0, &mut n, &mut d),
            Err(WindowError::Source)
        );
        assert_eq!(
            rows.read_level(3, 0, &mut n, &mut d),
            Err(WindowError::Source)
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn final_rows_of_a_four_billion_leaf_domain_allocate_no_table() {
        let s = source(32);
        let mut n = SecretBuf::filled(F::ZERO, 19);
        let mut d = SecretBuf::filled(F::ZERO, 19);
        let measured = allocation_counter::measure(|| {
            let rows = FractionTreeRows::new(&s, NonZeroUsize::new(17).unwrap(), 4096).unwrap();
            rows.read_level(32, (1usize << 32) - 19, &mut n, &mut d)
                .unwrap();
        });
        assert_eq!(measured.bytes_total, 0);
        assert_eq!(s.calls.get(), 2);
        assert_eq!(s.largest.get(), 17);
        assert_eq!(n[18], F::from(G::from_u64_reduced((1u64 << 32) - 1)));
        assert_eq!(d[18], F::from(G::from_u64_reduced(1u64 << 32)));
    }
}

#[cfg(test)]
mod differential;
