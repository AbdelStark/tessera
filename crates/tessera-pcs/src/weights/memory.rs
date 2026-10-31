// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public requested-byte bounds for protocol/1 regenerated weight work (RFC-0012/0040/0052).
use super::*;
use crate::{CombinedRow, WeightMatrix};
use tessera_codec::SecretBuf;
use tessera_field::{Goldilocks as Fp, GoldilocksExt2 as Fp2};
use tessera_hash::{Digest, MerkleTree};

/// Component allocation bounds for regenerated protocol/1 weights. Borrowed model
/// payload/layout, input decoding, original opening targets, generic proof/messages,
/// stack/runtime and allocator overhead remain separate. These do not admit a route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeightWorkMemory {
    retained: u64,
    construction: u64,
    model_refs: u64,
    fold: u64,
    combined: u64,
    answer: u64,
    outputs: u64,
}
impl WeightWorkMemory {
    /// All ten retained trees, their container and the commitment's cloned layout.
    pub fn retained_bytes(self) -> u64 {
        self.retained
    }
    /// Commitment constructor peak, including retained trees and canonical model validation.
    pub fn construction_peak_bytes(self) -> u64 {
        self.construction
    }
    /// Additional canonical ModelWeights descriptors; underlying INT8/channel values are borrowed.
    pub fn model_refs_bytes(self) -> u64 {
        self.model_refs
    }
    /// Peak for any one canonical row reader/fold and its input or combined output.
    /// Add previously retained combined rows separately.
    pub fn fold_peak_bytes(self) -> u64 {
        self.fold
    }
    /// All matrices' public combined rows, with every ChannelParams claim counted.
    pub fn combined_output_bytes(self) -> u64 {
        self.combined
    }
    /// Additional peak of any one regenerated answer, including its public output.
    /// Add the commitment and previously retained answers separately.
    pub fn answer_peak_bytes(self) -> u64 {
        self.answer
    }
    /// All matrices' public answer columns/proofs, without assuming repeated queries.
    pub fn answer_output_bytes(self) -> u64 {
        self.outputs
    }
}
fn large() -> crate::PcsError {
    crate::PcsError::TooLarge {
        rows: u64::MAX,
        row_len: 1,
    }
}
fn add(a: u64, b: u64) -> Result<u64, crate::PcsError> {
    a.checked_add(b).ok_or_else(large)
}
fn mul(a: u64, b: u64) -> Result<u64, crate::PcsError> {
    a.checked_mul(b).ok_or_else(large)
}
fn array<T>(n: u64) -> Result<u64, crate::PcsError> {
    let b = mul(n, size_of::<T>() as u64)?;
    if usize::try_from(n).is_err() || b > isize::MAX as u64 {
        return Err(large());
    }
    Ok(b)
}
fn grow<T>(n: u64) -> Result<u64, crate::PcsError> {
    if n == 0 {
        return Ok(0);
    }
    let min = if size_of::<T>() == 1 {
        8
    } else if size_of::<T>() <= 1024 {
        4
    } else {
        1
    };
    array::<T>(mul(4, n.max(min))?)
}
fn power(n: u32) -> Result<u64, crate::PcsError> {
    1u64.checked_shl(n).ok_or_else(large)
}
fn eq(vars: u64) -> Result<u64, crate::PcsError> {
    add(array::<[Fp2; 2]>(vars)?, array::<Fp2>(add(vars, 1)?)?)
}

impl WeightLayout {
    /// Plan all ten protocol/1 regenerated trees, their constructor, canonical
    /// model-reference descriptors, fold/combined-row work and column answers,
    /// without allocating or reading weight values. `channel_claims` includes
    /// every original ChannelParams opening, even zero/repeated combinations.
    /// `draws` is the named parameter set's full query count.
    ///
    /// Canonical ModelWeights/WeightRows are assumed; arbitrary MatrixRows callbacks
    /// are excluded. Per-matrix folds/answers are sequential, but all their outputs
    /// remain retained and are reported separately. The commitment/input layout and
    /// caller-owned model payload must be charged separately where indicated.
    ///
    /// # Errors
    /// No queries, missing ChannelParams claims for committed rows, or an
    /// overflowing/unaddressable allocation returns the existing typed PCS error.
    pub fn regenerated_work_memory(
        &self,
        row_block: core::num::NonZeroU32,
        draws: u64,
        channel_claims: u64,
    ) -> Result<WeightWorkMemory, crate::PcsError> {
        if draws == 0 {
            return Err(crate::PcsError::NoQueries);
        }
        let tensors = add(2, mul(7, u64::from(self.shapes().layers))?)?;
        let channels = self.shapes().channels.len() as u64;
        // check() may retain the original tuple allocations after in-place collect.
        // Sum source and destination requests to cover either specialization.
        let validation = add(
            add(
                array::<(WeightMatrix, u32, &[i8])>(tensors)?,
                array::<&[i8]>(tensors)?,
            )?,
            add(
                array::<(WeightMatrix, u32, ChannelValues<'_>)>(channels)?,
                array::<ChannelValues<'_>>(channels)?,
            )?,
        )?;
        let model_refs = add(
            array::<WeightTensorRef<'_>>(tensors)?,
            grow::<ChannelParamsRef<'_>>(channels)?,
        )?;
        let layout_clone = mul(2, array::<(WeightMatrix, u32)>(channels)?)?;
        let mut retained = add(array::<MerkleTree>(10)?, layout_clone)?;
        let (mut construction_work, mut fold_peak, mut combined, mut answer_peak, mut outputs) =
            (0, 0, 0, 0, 0);
        for matrix in WeightMatrix::ALL {
            let s = self.matrix(matrix);
            let (k, r) = (s.message_len() as u64, s.rows());
            let log_n = s
                .log2_k()
                .checked_add(u32::from(self.params().log2_inv_rate))
                .ok_or_else(large)?;
            let n = power(log_n)?;
            let tree = array::<Digest>(mul(2, n)?.checked_sub(1).ok_or_else(large)?)?;
            retained = add(retained, tree)?;
            let encoder = add(
                array::<Fp>(add(add(k / 2, n / 2)?, k)?)?,
                array::<Fp>(u64::from(s.log2_k().max(log_n.saturating_sub(1))))?,
            )?;
            let reader = add(
                validation,
                if matrix == WeightMatrix::ChannelParams {
                    grow::<u64>(r)?
                } else {
                    0
                },
            )?;
            let encode = add(
                add(encoder, reader)?,
                add(
                    array::<Fp>(k)?,
                    add(
                        array::<Fp>(mul(u64::from(row_block.get()).min(r), n)?)?,
                        array::<Digest>(n)?,
                    )?,
                )?,
            )?;
            // The tree growth's old leaf allocation is also <= encode's chain allocation.
            construction_work = construction_work.max(encode);
            let d = draws.min(n);
            let proof = array::<Digest>(mul(d, u64::from(log_n))?.min(n - 1))?;
            let values = array::<Fp>(mul(d, r)?)?;
            let output = add(proof, add(array::<Vec<Fp>>(d)?, values)?)?;
            outputs = add(outputs, output)?;
            let private = add(
                array::<u64>(draws)?,
                add(grow::<SecretBuf<Fp>>(d)?, values)?,
            )?;
            // open: old/next usize indices. Self-check: leaves, known and next pairs.
            let checking = add(
                proof,
                add(
                    mul(2, array::<usize>(d)?)?,
                    mul(3, array::<(u64, Digest)>(d)?)?,
                )?,
            )?;
            let answer_encode = add(encode, array::<Fp>(d)?)?;
            answer_peak = answer_peak.max(add(
                private,
                add(tree, answer_encode.max(add(checking, output)?))?,
            )?);
            if r == 0 {
                continue;
            }
            let (claims, layers, column_vars, layer_vars) = match s.kind() {
                MatrixKind::Weight { layers, a, h, .. } => {
                    (1, u64::from(layers), a, s.row_vars() - h)
                }
                MatrixKind::ChannelParams { .. } => {
                    if channel_claims == 0 {
                        return Err(crate::PcsError::NoClaims);
                    }
                    (channel_claims, channel_claims, 0, 0)
                }
            };
            let folds = array::<Fp2>(mul(layers, k)?)?;
            let kept = add(
                folds,
                add(array::<Fp2>(k)?, array::<Fp2>(u64::from(column_vars))?)?,
            )?;
            let xi_vars = u64::from(u64::BITS - (r - 1).leading_zeros());
            let coefficient_vars = xi_vars.max(u64::from(s.row_vars()));
            let coefficients = add(
                array::<Fp2>(mul(add(claims, 1)?, r)?)?,
                mul(add(claims, 1)?, eq(coefficient_vars)?)?,
            )?;
            let build = add(
                reader,
                add(
                    kept,
                    add(grow::<u64>(r)?, add(coefficients, array::<Fp>(k)?)?)?,
                )?,
            )?;
            let rows_out = add(
                array::<CombinedRow>(claims)?,
                array::<Fp2>(mul(add(claims, 1)?, k)?)?,
            )?;
            combined = add(combined, rows_out)?;
            let combine = add(
                kept,
                add(
                    rows_out,
                    add(array::<Fp2>(layers)?, eq(u64::from(layer_vars))?)?,
                )?,
            )?;
            let input_cells = if matrix == WeightMatrix::ChannelParams {
                mul(claims, k)?
            } else {
                mul(k >> column_vars, power(layer_vars)?)?
            };
            let input = add(
                kept,
                add(
                    array::<Fp2>(input_cells)?,
                    add(
                        array::<Fp2>(power(column_vars)?)?,
                        eq(u64::from(column_vars))?,
                    )?,
                )?,
            )?;
            fold_peak = fold_peak.max(build).max(add(reader, combine.max(input))?);
        }
        Ok(WeightWorkMemory {
            retained,
            construction: add(retained, add(validation, construction_work)?)?,
            model_refs,
            fold: fold_peak,
            combined,
            answer: answer_peak,
            outputs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn weight_work_sizes_refuse_overflow_and_missing_counts_without_allocating() {
        let layout = WeightLayout::new(
            crate::WeightLayoutParams {
                log2_k: 3,
                log2_inv_rate: 2,
                mask_coeffs: 0,
                blinding_rows: 0,
            },
            WeightShapes {
                layers: 3,
                shapes: [WeightShape { out: 3, inputs: 2 }; 9],
                channels: vec![(WeightMatrix::Q, 0)],
            },
        )
        .unwrap();
        let block = core::num::NonZeroU32::new(3).unwrap();
        let measured = allocation_counter::measure(|| {
            assert_eq!(
                array::<Fp2>(isize::MAX as u64 / size_of::<Fp2>() as u64 + 1),
                Err(large())
            );
            assert_eq!(array::<Fp2>(u64::MAX), Err(large()));
            assert_eq!(
                array::<Fp2>(u64::MAX / size_of::<Fp2>() as u64 + 1),
                Err(large())
            );
            assert_eq!(grow::<Fp>(u64::MAX), Err(large()));
            assert_eq!(power(64), Err(large()));
            assert_eq!(
                layout.regenerated_work_memory(block, 0, 7),
                Err(crate::PcsError::NoQueries)
            );
            assert_eq!(
                layout.regenerated_work_memory(block, 5, 0),
                Err(crate::PcsError::NoClaims)
            );
            assert_eq!(
                layout.regenerated_work_memory(block, u64::MAX, 7),
                Err(large())
            );
            assert_eq!(
                layout.regenerated_work_memory(block, 5, u64::MAX),
                Err(large())
            );
        });
        assert_eq!(measured.count_total, 0);
    }
}
