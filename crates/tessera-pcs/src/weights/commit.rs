// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The weight commitment of `tessera-protocol/1` (RFC-0040 'Encoding, leaves and roots'): each
//! matrix's committed rows at rate `1/2^log2_inv_rate`, no mask and no blinding row, one chain per
//! column from the matrix's `commitment_id`, the Merkle root, and `R_W` over the ten roots.

use core::fmt;
use core::num::{NonZeroU32, NonZeroUsize};

use tessera_codec::Secret;
use tessera_field::{Field, Goldilocks};
use tessera_hash::Digest;

use crate::weights::layout::{ModelWeights, WeightLayout};
use crate::{
    CommitmentDescriptor, CommitmentParams, PcsError, ReferenceCommitment, RowLayout,
    WeightCommitmentPublic, WeightMatrix,
};

type Fp = Goldilocks;

/// The weight commitment of `D` (RFC-0040): every matrix's [`ReferenceCommitment`], and the
/// public artifact that `R_W` hashes.
///
/// It keeps each matrix's committed rows and codewords for its openings, in their `Secret` and
/// `SecretBuf`s, cleared when it drops (05 §3 OBS-6), as RFC-0012 'Migration / Rollout' plans for
/// `tessera-protocol/1`: codewords held in memory. Its `Debug` form shows the public part and the
/// matrices' shapes.
pub struct WeightCommitment {
    public: WeightCommitmentPublic,
    matrices: Vec<ReferenceCommitment>,
}

/// Shows the public part and each matrix's shape.
impl fmt::Debug for WeightCommitment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WeightCommitment")
            .field("public", &self.public)
            .field("matrices", &self.matrices)
            .finish()
    }
}

impl WeightCommitment {
    /// Commits `weights` on `layout`, with the parameter set's row block `B` (RFC-0031 'The
    /// weight matrices' row block'), at `epoch`: 0 for the first commitment.
    ///
    /// Each matrix `T` lays its committed rows out ([`WeightLayout::write_rows`]) and commits
    /// them with [`ReferenceCommitment::commit_unmasked`], from
    /// `commitment_id_T = H("commitment-id", TCE(CommitmentDescriptor::WeightMatrix { layout,
    /// matrix: T, log2_k: log2 k_T, row_block: B, rows: m_T, epoch }))`. The public part holds the
    /// layout, the ten roots in [`WeightMatrix`] order and the epoch.
    ///
    /// # Errors
    ///
    /// Those of the weights' checks ([`WeightLayout::write_rows`]); then, for the first matrix
    /// in [`WeightMatrix`] order that meets one, [`PcsError::TooLarge`] if its committed rows
    /// cannot be held, and those of [`ReferenceCommitment::commit_unmasked`], which a layout
    /// from [`WeightLayout::new`] does not meet.
    #[allow(clippy::disallowed_methods)]
    pub fn commit(
        layout: &WeightLayout,
        row_block: NonZeroU32,
        epoch: u32,
        weights: &ModelWeights<'_>,
    ) -> Result<Self, PcsError> {
        let checked = layout.check(weights)?;
        let block = usize::try_from(row_block.get())
            .ok()
            .and_then(NonZeroUsize::new)
            .unwrap_or(NonZeroUsize::MAX);
        let params = *layout.params();
        let mut roots = [Digest::default(); 10];
        let mut matrices = Vec::with_capacity(roots.len());
        for matrix in WeightMatrix::ALL {
            let shape = layout.matrix(matrix);
            let descriptor = CommitmentDescriptor::WeightMatrix {
                layout: params,
                matrix,
                // log2 k_T + log2_inv_rate ≤ 32 (WeightLayout::new).
                log2_k: shape.log2_k() as u8,
                row_block: row_block.get(),
                rows: shape.rows(),
                epoch,
            };
            let code = CommitmentParams {
                log2_k: shape.log2_k(),
                log2_inv_rate: u32::from(params.log2_inv_rate),
                mask_coeffs: 0,
                row_block: block,
                commitment_id: descriptor.commitment_id(),
            };
            let too_large = PcsError::TooLarge {
                rows: shape.rows(),
                row_len: shape.message_len(),
            };
            let len = usize::try_from(shape.rows())
                .map_err(|_| too_large)?
                .checked_mul(shape.message_len())
                .ok_or(too_large)?;
            // The row layout first, from public data alone: it fails, if it does, before any
            // weight is read.
            let rows = RowLayout::new(
                shape.row_vars(),
                vec![layout.committed_rows(matrix).collect()],
            )?;
            // The rows' values are allocated inside their Secret and written through it, so
            // that an error or a panic from here on drops them cleared (05 §3 OBS-6); the
            // commitment keeps them there.
            let mut values = Secret::new(vec![Fp::ZERO; len]);
            layout.fill(matrix, &checked, values.expose_secret_mut());
            let commitment = ReferenceCommitment::commit_unmasked(code, rows, values)?;
            roots[matrix.index()] = commitment.root();
            matrices.push(commitment);
        }
        Ok(Self {
            public: WeightCommitmentPublic {
                layout: params,
                roots,
                epoch,
            },
            matrices,
        })
    }

    /// The public part: the layout, the ten roots and the epoch, which `R_W` hashes.
    pub fn public(&self) -> &WeightCommitmentPublic {
        &self.public
    }

    /// `R_W = H("weights-root", TCE(WeightCommitmentPublic))` (RFC-0040).
    pub fn weights_root(&self) -> Digest {
        self.public.weights_root()
    }

    /// The commitment of `matrix`.
    pub fn matrix(&self, matrix: WeightMatrix) -> &ReferenceCommitment {
        &self.matrices[matrix.index()]
    }
}
