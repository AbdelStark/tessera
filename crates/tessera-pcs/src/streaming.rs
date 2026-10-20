// SPDX-License-Identifier: Apache-2.0 OR MIT

//! One-block advice commitments (RFC-0011), over the canonical layout's public row runs.

use core::fmt;
use core::num::NonZeroUsize;
use core::ops::Range;
use std::iter::{Cloned, Flatten, Peekable};
use std::slice::Iter;

use tessera_codec::{Decode, Encode, SecretBuf, Zeroize};
use tessera_field::{Field, Goldilocks};
use tessera_hash::{ColumnChains, Digest, Domain, Hasher, merkle_node};

use crate::{AdviceLayout, CommitmentParams, MaskSource, PcsError, RowEncoder};

type Rows<'a> = Peekable<Flatten<Cloned<Iter<'a, Range<u64>>>>>;

/// The advice commitment (RFC-0011 'Column digests and root'). Its encoding is unchanged
/// when re-exported by the protocol's `Round1` message (RFC-0036).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub struct AdviceCommitment {
    /// The Merkle root over the merged column leaves.
    pub root: Digest,
    /// The advice descriptor's `commitment_id`, independently derived by the verifier (RFC-0031).
    pub layout_digest: Digest,
}

/// One segment's completed column chains, bound to its public layout and commitment parameters.
/// Only [`merge_segments`] turns the ordered complete set into an advice commitment.
pub struct SegmentDigests<'layout> {
    layout: &'layout AdviceLayout,
    params: CommitmentParams,
    segment: u32,
    digests: Vec<Digest>,
}

impl fmt::Debug for SegmentDigests<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SegmentDigests")
            .field("segment", &self.segment)
            .field("columns", &self.digests.len())
            .finish_non_exhaustive()
    }
}

impl SegmentDigests<'_> {
    /// The segment's index in the canonical layout.
    pub fn segment(&self) -> u32 {
        self.segment
    }

    /// Its last chain value per column, in column order; empty segments hold the initial digests.
    pub fn digests(&self) -> &[Digest] {
        &self.digests
    }
}

/// Commits one segment without retaining rows, masks or codewords beyond one block (RFC-0011).
///
/// The layout, encoder tables and mask source are borrowed. The source is the caller's per-proof
/// PRG stream (RFC-0018); segments consume their prescribed portion in layout order. Exactly `P`
/// coefficients are drawn per accepted row. No coefficient is drawn for a layout-skipped row or on error.
/// A caller lends a row from its secret storage; the committer keeps only a codeword block in a
/// [`SecretBuf`], clears it after each hash step, and clears its allocation on drop (OBS-6).
///
/// Heap storage is `8·min(B, segment_rows)·n` bytes for the block and `32·n` for chain state.
/// It is independent of the witness length. The borrowed public metadata and encoder tables,
/// the caller's input row and PRG state are outside that storage.
pub struct StreamingCommitter<'layout, 'work, M: MaskSource + ?Sized> {
    layout: &'layout AdviceLayout,
    params: CommitmentParams,
    segment: u32,
    rows: Rows<'layout>,
    encoder: &'work RowEncoder,
    mask: &'work mut M,
    chains: ColumnChains,
    buffer: SecretBuf<Goldilocks>,
    buffered: usize,
}

impl<M: MaskSource + ?Sized> fmt::Debug for StreamingCommitter<'_, '_, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamingCommitter")
            .field("segment", &self.segment)
            .field("columns", &self.encoder.codeword_len())
            .field("buffered_rows", &self.buffered)
            .finish_non_exhaustive()
    }
}

impl<'layout, 'work, M: MaskSource + ?Sized> StreamingCommitter<'layout, 'work, M> {
    /// Starts `segment`'s chains from `params.commitment_id` (RFC-0031). The caller derives that
    /// identifier from the statement, proof unit, layout parameters and total committed rows.
    ///
    /// # Errors
    /// The masked [`CommitmentParams`] checks run first, then [`PcsError::LayoutData`] for a
    /// layout width or encoder that disagrees with `params`, [`PcsError::SegmentIndex`] for an
    /// absent segment, or [`PcsError::LayoutData`] for a block exceeding the platform's capacity.
    /// All checks precede allocation and mask draws.
    pub fn new(
        params: CommitmentParams,
        layout: &'layout AdviceLayout,
        segment: u32,
        encoder: &'work RowEncoder,
        mask: &'work mut M,
    ) -> Result<Self, PcsError> {
        let shape = params.shape()?;
        if layout.params().log2_k != params.log2_k {
            return Err(PcsError::LayoutData {
                rule: "the streaming layout uses the commitment's row width",
            });
        }
        if encoder.message_len() != shape.k
            || encoder.codeword_len() != shape.n
            || encoder.mask_coeffs() != shape.mask_coeffs
        {
            return Err(PcsError::LayoutData {
                rule: "the streaming encoder uses the commitment's k, n and P",
            });
        }
        let part = layout
            .segments()
            .get(segment as usize)
            .ok_or(PcsError::SegmentIndex {
                segment,
                segments: layout.params().segments,
            })?;
        let count: u64 = part.nonzero.iter().map(|r| r.end - r.start).sum();
        let block = params
            .row_block
            .get()
            .min(usize::try_from(count).unwrap_or(usize::MAX));
        let symbols = block
            .checked_mul(shape.n)
            .filter(|&len| len <= isize::MAX as usize / size_of::<Goldilocks>())
            .ok_or(PcsError::LayoutData {
                rule: "one streaming codeword block fits the platform's capacity",
            })?;
        Ok(Self {
            layout,
            params,
            segment,
            rows: part.nonzero.iter().cloned().flatten().peekable(),
            encoder,
            mask,
            chains: ColumnChains::new(
                &params.commitment_id,
                NonZeroUsize::new(shape.n).expect("a checked codeword has at least two columns"),
                params.row_block,
            ),
            buffer: SecretBuf::filled(Goldilocks::ZERO, symbols),
            buffered: 0,
        })
    }

    /// Encodes the next committed row. Layout-skipped rows, repeats, omissions and out-of-order pushes are
    /// refused (INV-PCS-2); a refused push leaves the source and buffered contents untouched.
    ///
    /// # Errors
    /// [`PcsError::StreamRow`] unless `row` is the next row of this segment, then
    /// [`PcsError::RowLength`] unless `values` holds exactly `k` field elements.
    pub fn push_row(&mut self, row: u64, values: &[Goldilocks]) -> Result<(), PcsError> {
        let expected = self.rows.peek().copied();
        if expected != Some(row) {
            return Err(PcsError::StreamRow {
                segment: self.segment,
                expected,
                found: Some(row),
            });
        }
        let n = self.encoder.codeword_len();
        let start = self.buffered * n;
        self.encoder
            .encode_row(values, self.mask, &mut self.buffer[start..start + n])?;
        self.rows.next();
        self.buffered += 1;
        if self.buffered * n == self.buffer.len() {
            self.flush();
        }
        Ok(())
    }

    /// Hashes the short final block, if any, and releases the row buffer and mask-source borrow.
    /// An empty segment returns its initial chains. No Merkle tree or witness is retained.
    ///
    /// # Errors
    /// [`PcsError::StreamRow`] with `found: None` if a non-zero row is still missing.
    pub fn finish(mut self) -> Result<SegmentDigests<'layout>, PcsError> {
        if let Some(row) = self.rows.peek().copied() {
            return Err(PcsError::StreamRow {
                segment: self.segment,
                expected: Some(row),
                found: None,
            });
        }
        if self.buffered != 0 {
            self.flush();
        }
        Ok(SegmentDigests {
            layout: self.layout,
            params: self.params,
            segment: self.segment,
            digests: self.chains.finish(),
        })
    }

    fn flush(&mut self) {
        let used = self.buffered * self.encoder.codeword_len();
        self.chains
            .absorb_block_no_alloc(&self.buffer[..used])
            .expect("a full block, or the one short final block, of whole codewords");
        // A live buffer holds no previous block while the next row is regenerated (RFC-0011).
        self.buffer.iter_mut().for_each(Zeroize::zeroize);
        self.buffered = 0;
    }
}

/// Merges the complete ordered set of segment chains, including empty segments and `S = 1`.
/// Every leaf hashes all segments' digests, in layout order (INV-PCS-3). A public stack frontier
/// builds the Merkle root without a heap allocation or a retained tree.
///
/// # Errors
/// [`PcsError::NoSegments`] for no part, [`PcsError::SegmentCount`] for an incomplete set,
/// [`PcsError::SegmentOrder`] for an absent/repeated/reordered index, or
/// [`PcsError::SegmentContext`] for mixed layouts or commitment parameters. Nothing is mutated.
pub fn merge_segments(parts: &[SegmentDigests<'_>]) -> Result<AdviceCommitment, PcsError> {
    let first = parts.first().ok_or(PcsError::NoSegments)?;
    let expected = first.layout.params().segments as usize;
    if parts.len() != expected {
        return Err(PcsError::SegmentCount {
            expected,
            found: parts.len(),
        });
    }
    for (position, part) in parts.iter().enumerate() {
        if part.segment as usize != position {
            return Err(PcsError::SegmentOrder {
                position,
                segment: part.segment,
            });
        }
        if part.params != first.params || part.layout != first.layout {
            return Err(PcsError::SegmentContext {
                segment: part.segment,
            });
        }
    }
    let n = first.digests.len();
    let mut frontier = [None; crate::ntt::TWO_ADICITY as usize + 1];
    for column in 0..n {
        let mut merge = Hasher::new(Domain::ColumnMerge);
        for part in parts {
            merge.update(&part.digests[column]);
        }
        let mut digest = merge.finalize();
        let (mut index, mut level) = (column, 0);
        while index & 1 != 0 {
            digest = merkle_node(
                &frontier[level].take().expect("the preceding left subtree"),
                &digest,
            );
            index >>= 1;
            level += 1;
        }
        frontier[level] = Some(digest);
    }
    Ok(AdviceCommitment {
        root: frontier[n.trailing_zeros() as usize].expect("a checked power-of-two column count"),
        layout_digest: first.params.commitment_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Axis, DenseShape, LayoutParams, Padding, TensorShape, TileShape};

    #[test]
    fn a_flushed_block_is_cleared_before_the_next_row() {
        let layout = AdviceLayout::new(
            LayoutParams {
                log2_k: 0,
                segments: 1,
                layers: 1,
                group: 1,
                query_block: 1,
                max_context: 1,
            },
            1,
            &[TileShape {
                layer: Some(0),
                values: vec![TensorShape::Dense(DenseShape::new(
                    vec![Axis::new(5)],
                    Padding::Zero,
                ))],
            }],
        )
        .expect("layout");
        let params = CommitmentParams {
            log2_k: 0,
            log2_inv_rate: 1,
            mask_coeffs: 1,
            row_block: NonZeroUsize::new(3).expect("B"),
            commitment_id: [31; 32],
        };
        let encoder = RowEncoder::new(1, 1, 1).expect("encoder");
        let mut source = || Goldilocks::ONE;
        let mut stream =
            StreamingCommitter::new(params, &layout, 0, &encoder, &mut source).expect("stream");
        stream.push_row(0, &[Goldilocks::ONE]).expect("row 0");
        assert!(
            stream.buffer.iter().any(|v| *v != Goldilocks::ZERO),
            "the encoded block has material to clear"
        );
        stream.push_row(1, &[Goldilocks::ONE]).expect("row 1");
        stream.push_row(2, &[Goldilocks::ONE]).expect("flush");
        assert_eq!(stream.buffered, 0);
        assert!(
            stream.buffer.iter().all(|v| *v == Goldilocks::ZERO),
            "RFC-0011 clears the live block after hashing"
        );
        stream.push_row(3, &[Goldilocks::ONE]).expect("next block");
        assert!(stream.buffer[..2].iter().any(|v| *v != Goldilocks::ZERO));
        assert!(
            stream.buffer[2..].iter().all(|v| *v == Goldilocks::ZERO),
            "no previous block remains"
        );
    }
}
