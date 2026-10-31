// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Requested live heap storage owned by regenerated advice commitments and openings.

use super::*;

/// Checked pre-P1 storage bounds for advice grouping and its complete P4 pass (RFC-0018).
/// Counts must bound every original claim/image and all possible column groups. Numeric
/// coincidences and zero coefficients cannot reduce these public counts.
///
/// This covers the opening implementation's requested allocations, including both private
/// accumulators and public outputs. Caller-owned claims, layout/P1 tree, encoder, source row,
/// mask generator, allocator overhead and other proof phases remain separate. This is not
/// by itself a complete CPU memory bound or a host-budget admission decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdviceOpeningMemory {
    groups: u64,
    prepare_peak: u64,
    private_pass: u64,
    output: u64,
    peak: u64,
}

fn plus(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or_else(|| capacity(u64::MAX, 1))
}
fn times(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).ok_or_else(|| capacity(u64::MAX, 1))
}
fn allocation<T>(entries: u64) -> Result<u64> {
    let bytes = times(entries, size_of::<T>() as u64)?;
    usize::try_from(entries).map_err(|_| capacity(entries, size_of::<T>()))?;
    if bytes > isize::MAX as u64 {
        return Err(capacity(entries, size_of::<T>()));
    }
    Ok(bytes)
}
impl AdviceOpeningMemory {
    /// Plan without allocating any payload, before sampling a challenge or beginning P1.
    /// `claims` counts original nonempty combinations, `images` all their pieces, and
    /// `groups` an upper bound on distinct column prefixes, excluding proximity.
    ///
    /// # Errors
    /// Invalid commitment/layout/count geometry, or unaddressable/overflowing storage.
    pub fn new(
        layout: &AdviceLayout,
        params: crate::CommitmentParams,
        claims: u64,
        images: u64,
        groups: u64,
    ) -> Result<Self> {
        let shape = params.shape()?;
        if params.log2_inv_rate != 1 || params.log2_k != layout.params().log2_k {
            return Err(invalid("advice opening memory uses the P1 layout and rate"));
        }
        if images < claims
            || groups > claims
            || (claims == 0) != (images == 0)
            || (claims == 0) != (groups == 0)
        {
            return Err(invalid(
                "opening counts include every nonempty original combination",
            ));
        }
        let (k, n, p) = (shape.k as u64, shape.n as u64, shape.mask_coeffs as u64);
        let row_vars = u64::from(layout.row_vars());
        let column_vars = u64::from(params.log2_k);
        let rows = usize::try_from(layout.rows()).map_err(|_| capacity(layout.rows(), shape.k))?;
        let b = params.row_block.get().min(rows) as u64;
        let q = plus(groups, 1)?;
        let width = plus(k, p)?;
        // `prepare` reserves claims descriptors/powers for C, not merely G. Each image
        // retains its own RowClaim and row point even at coefficient zero.
        let points = allocation::<Fp2>(row_vars)?;
        let group_bytes = plus(
            plus(
                allocation::<ClaimGroup>(claims)?,
                times(groups, allocation::<Fp2>(column_vars)?)?,
            )?,
            plus(
                plus(allocation::<RowClaim>(images)?, times(images, points)?)?,
                points,
            )?,
        )?;
        // A repeated column group grows its RowClaim vector once per original. During
        // reserve_exact, old and new descriptor arrays can coexist. Charge a full
        // extra image descriptor array; pointed-to row coordinates are moved, not copied.
        let prepare_peak = plus(
            plus(group_bytes, allocation::<Fp2>(claims)?)?,
            allocation::<RowClaim>(images)?,
        )?;
        // The pass's aggregate addressability checks match OpeningPass::new.
        let accumulators = allocation::<Fp2>(times(q, width)?)?;
        allocation::<Fp2>(times(q, b)?)?;
        allocation::<u8>(times(q, COMBINE_SCRATCH as u64)?)?;
        let private_pass = plus(
            plus(accumulators, times(2, allocation::<SecretBuf<Fp2>>(q)?)?)?,
            plus(
                plus(
                    allocation::<Fp>(times(b, k)?)?,
                    allocation::<Fp>(times(b, p)?)?,
                )?,
                plus(
                    plus(
                        allocation::<Fp>(n)?,
                        plus(allocation::<Fp>(b)?, allocation::<u64>(b)?)?,
                    )?,
                    allocation::<Fp2>(q)?,
                )?,
            )?,
        )?;
        // During flush, the coefficients coexist with either eq's factors/partials,
        // or combine_into's raw coefficient words and clearing Wide accumulators.
        let eq = plus(
            allocation::<[Fp2; 2]>(row_vars)?,
            allocation::<Fp2>(plus(row_vars, 1)?)?,
        )?;
        let combine = plus(
            allocation::<[u64; 2]>(times(q, b)?)?,
            times(q, COMBINE_SCRATCH as u64)?,
        )?;
        let flush = plus(
            plus(allocation::<Vec<Fp2>>(q)?, allocation::<Fp2>(times(q, b)?)?)?,
            eq.max(combine),
        )?;
        let output = plus(allocation::<CombinedRow>(q)?, accumulators)?;
        // Montgomery inversion holds weights and prefix together. The prefix has dropped
        // before all public rows are allocated; weights remain beside those output copies.
        let inversion = times(2, allocation::<Fp>(k)?)?;
        let publish = plus(allocation::<Fp>(k)?, output)?;
        let peak = prepare_peak.max(plus(
            group_bytes,
            plus(private_pass, flush.max(inversion).max(publish))?,
        )?);
        Ok(Self {
            groups: group_bytes,
            prepare_peak,
            private_pass,
            output,
            peak,
        })
    }
    /// Retained group descriptors and row/column points, excluding caller-owned claims.
    pub fn groups_bytes(self) -> u64 {
        self.groups
    }
    /// Peak while grouping original claims, including the power vector and claim-vector growth overlap.
    pub fn prepare_peak_bytes(self) -> u64 {
        self.prepare_peak
    }
    /// Retained private P4 buffers; temporary flush/self-check/output copies are additional.
    pub fn private_pass_bytes(self) -> u64 {
        self.private_pass
    }
    /// All public combined rows and their descriptors, excluding private accumulators.
    pub fn output_bytes(self) -> u64 {
        self.output
    }
    /// Maximum requested live heap bytes across grouping, P4 flush, self-check and publication.
    pub fn peak_bytes(self) -> u64 {
        self.peak
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opening_memory_capacity_limits_reject_before_allocation() {
        let mut result = None;
        // Fits usize, but not one Rust allocation, on both 32- and 64-bit targets.
        let measured = allocation_counter::measure(|| {
            result = Some(allocation::<u8>((isize::MAX as u64) + 1));
        });
        assert_eq!(measured.count_total, 0);
        assert_eq!(result.unwrap().unwrap_err().code(), ErrorCode::ResMemory);
    }
}

/// Checked pre-P1 bounds for allocations owned by advice P5 (RFC-0011/0018).
/// The public draw count covers every query, including repetitions; no favorable
/// collision or adjacency is assumed. Caller P1/layout/encoder/source/mask state,
/// retained Round2 messages and other proof phases are separate allocations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdviceColumnMemory {
    private: u64,
    proof: u64,
    output: u64,
    peak: u64,
}
impl AdviceColumnMemory {
    /// Plan the worst requested live heap before any column indices are sampled.
    /// `draws` is the actual public query count from the named parameter set.
    ///
    /// # Errors
    /// Invalid commitment/layout, zero draws, or unaddressable/overflowing arrays.
    pub fn new(layout: &AdviceLayout, params: crate::CommitmentParams, draws: u64) -> Result<Self> {
        let shape = params.shape()?;
        if params.log2_inv_rate != 1 || params.log2_k != layout.params().log2_k {
            return Err(invalid("advice column memory uses the P1 layout and rate"));
        }
        if draws == 0 {
            return Err(PcsError::NoQueries.into());
        }
        let n = shape.n as u64;
        let rows = layout.rows();
        let distinct = draws.min(n);
        // Query deduplication retains the original draw-sized capacity. Each private
        // column reserves all R symbols even before the first row arrives.
        let values = allocation::<Fp>(times(rows, distinct)?)?;
        let private = plus(
            plus(
                allocation::<u64>(draws)?,
                allocation::<SecretBuf<Fp>>(distinct)?,
            )?,
            plus(values, allocation::<Fp>(n)?)?,
        )?;
        // For d distinct leaves a canonical proof has at most d*h siblings, and
        // at most n-1 overall. Do NOT use n-distinct: before sampling, repeated
        // draws may give fewer distinct leaves and a larger proof than that formula.
        let siblings = times(distinct, u64::from(shape.n.ilog2()))?.min(n - 1);
        let proof = allocation::<Digest>(siblings)?;
        // Merkle opening holds old and next known-index arrays together. These
        // are dropped before the output columns are copied from private buffers.
        let proving = plus(proof, times(2, allocation::<usize>(distinct)?)?)?;
        let output = plus(proof, plus(allocation::<Vec<Fp>>(distinct)?, values)?)?;
        let peak = plus(private, proving.max(output))?;
        Ok(Self {
            private,
            proof,
            output,
            peak,
        })
    }
    /// Private columns, their containers, query copy and one retained codeword.
    pub fn private_pass_bytes(self) -> u64 {
        self.private
    }
    /// Maximum reserved digest payload of the canonical Merkle multiproof.
    pub fn proof_bytes(self) -> u64 {
        self.proof
    }
    /// All returned columns, their containers and the multiproof, after P5 completes.
    pub fn output_bytes(self) -> u64 {
        self.output
    }
    /// Maximum requested live heap while constructing and publishing P5's answer.
    pub fn peak_bytes(self) -> u64 {
        self.peak
    }
}

/// Checked PCS storage bounds for sequential advice P1 and its owned opening state.
/// The caller collects all segments in an exactly reserved segment-descriptor vector.
/// Encoder construction, all retained segment chains, the active codeword block,
/// merged tree and owned layout clone are included. Caller original layout/source
/// row, masks, tensor/tile generation, allocator overhead and other phases are separate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdviceCommitmentMemory {
    encoder: u64,
    encoder_peak: u64,
    segments: u64,
    commit: u64,
    merge: u64,
    retained: u64,
    peak: u64,
}
impl AdviceCommitmentMemory {
    /// Plan requested heap before encoder construction or P1 without allocating.
    /// Layout construction has already occurred and remains separately charged.
    ///
    /// # Errors
    /// Inconsistent layout/commitment parameters or unaddressable/overflowing storage.
    pub fn new(layout: &AdviceLayout, params: crate::CommitmentParams) -> Result<Self> {
        let shape = params.shape()?;
        if params.log2_inv_rate != 1 || params.log2_k != layout.params().log2_k {
            return Err(invalid(
                "advice commitment memory uses the P1 layout and rate",
            ));
        }
        let k = shape.k as u64;
        let n = shape.n as u64;
        let count = layout.segments().len() as u64;
        let mut block = 0;
        for segment in layout.segments() {
            let rows = segment
                .nonzero
                .iter()
                .try_fold(0, |rows, run| plus(rows, run.end - run.start))?;
            block = block.max(rows.min(params.row_block.get() as u64));
        }
        let encoder = allocation::<Fp>(plus(plus(k / 2, n / 2)?, k)?)?;
        // bit_reversed_powers retains its squares beside its output. At rate1/2,
        // forward twiddles and spread need log2(k) powers; spread is built last.
        let encoder_peak = plus(encoder, allocation::<Fp>(u64::from(params.log2_k))?)?;
        let segments = plus(
            allocation::<SegmentDigests<'_>>(count)?,
            allocation::<Digest>(times(count, n)?)?,
        )?;
        let commit = plus(
            plus(encoder, segments)?,
            allocation::<Fp>(times(block, n)?)?,
        )?;
        // AdviceOpening::build reserves its final tree capacity before writing leaves;
        // MerkleTree::new needs no further allocation. Every segment chain is still live.
        let tree = allocation::<Digest>(times(2, n)? - 1)?;
        let layout = layout
            .heap_bytes()
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| capacity(u64::MAX, 1))?;
        let retained = plus(tree, layout)?;
        let merge = plus(plus(encoder, segments)?, retained)?;
        Ok(Self {
            encoder,
            encoder_peak,
            segments,
            commit,
            merge,
            retained,
            peak: encoder_peak.max(commit).max(merge),
        })
    }
    /// Retained public encoder tables; the constructor's square powers are additional.
    pub fn encoder_bytes(self) -> u64 {
        self.encoder
    }
    /// Encoder construction peak, including its temporary square powers.
    pub fn encoder_peak_bytes(self) -> u64 {
        self.encoder_peak
    }
    /// All segment chains and an exactly reserved segment-descriptor array.
    pub fn segments_bytes(self) -> u64 {
        self.segments
    }
    /// Encoder, all segments and the largest sequential committer's codeword block.
    pub fn commit_peak_bytes(self) -> u64 {
        self.commit
    }
    /// Encoder, all segments, merged tree and simultaneous owned-layout clone.
    pub fn merge_peak_bytes(self) -> u64 {
        self.merge
    }
    /// Final owned P1 tree/layout after caller encoder and segment vectors drop.
    pub fn retained_bytes(self) -> u64 {
        self.retained
    }
    /// Maximum requested heap across encoder construction, P1 and owned publication.
    pub fn peak_bytes(self) -> u64 {
        self.peak
    }
}
