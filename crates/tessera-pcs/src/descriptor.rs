// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The commitments' layout parameters, which the claim list states (RFC-0026 'Schema'), and the
//! descriptor every column chain starts from (RFC-0031 'The descriptor').
//!
//! `AdviceLayoutParams` and `WeightLayoutParams` are the commitment crate's, so that it names the
//! layout it encodes (RFC-0040 'Migration / Rollout'); `tessera-ir` re-exports both, field for
//! field, so the claim list's bytes do not change.

use tessera_codec::{Decode, Encode, encode};
use tessera_hash::{Digest, Domain, hash};

/// The advice commitment's parameters (RFC-0011 'Column digests and root', RFC-0026 'Schema'):
/// the parameter set's values, which the claim list states.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub struct AdviceLayoutParams {
    /// `log2 k`: a row holds `k` values.
    pub log2_k: u8,
    /// `log2` of the inverse rate: 1, rate 1/2.
    pub log2_inv_rate: u8,
    /// `P_A`, each row's mask coefficients.
    pub mask_coeffs: u32,
    /// `B`, the rows a column chain absorbs per step.
    pub row_block: u32,
    /// `S`, the segments the rows stream in.
    pub segments: u32,
}

/// The weight commitment's parameters (RFC-0012 'Layout', RFC-0026 'Schema'): the parameter
/// set's values, which the claim list states and which `R_W` and every matrix's
/// `commitment_id` bind. Each matrix's shape and padding follow from the architecture
/// (RFC-0040 'The matrices').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub struct WeightLayoutParams {
    /// `log2 k_W`, the message width of the widest rows the layout aims for: a weight type's is
    /// `k_T = max(in_pad, min(k_W, out_pad · in_pad))` (RFC-0040).
    pub log2_k: u8,
    /// `log2` of the inverse rate: 2, rate 1/4.
    pub log2_inv_rate: u8,
    /// `P_W`, each row's mask coefficients. `tessera-protocol/1` masks no weight row, and binds
    /// the value as it is written (RFC-0040).
    pub mask_coeffs: u32,
    /// `B_W`, each matrix's blinding rows. `tessera-protocol/1` commits none, and binds the
    /// value as it is written (RFC-0040).
    pub blinding_rows: u32,
}

/// The ten matrices of the weight commitment, in RFC-0012's table order (RFC-0031): the nine
/// weight types of 03 §2 in their declaration order, then the channel parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Encode, Decode)]
pub enum WeightMatrix {
    /// The transposed embedding, `d_model × vocab`.
    Embed,
    /// The query projection.
    Q,
    /// The key projection.
    K,
    /// The value projection.
    V,
    /// The output projection.
    O,
    /// The MLP gate projection.
    Gate,
    /// The MLP up projection.
    Up,
    /// The MLP down projection.
    Down,
    /// The vocabulary head, `vocab × d_model`.
    LmHead,
    /// The channel parameters of every weight that has them (03 §2 `ChannelParams`).
    ChannelParams,
}

impl WeightMatrix {
    /// Every matrix, in declaration order: the order of `WeightCommitmentPublic.roots`.
    pub const ALL: [Self; 10] = [
        Self::Embed,
        Self::Q,
        Self::K,
        Self::V,
        Self::O,
        Self::Gate,
        Self::Up,
        Self::Down,
        Self::LmHead,
        Self::ChannelParams,
    ];

    /// The nine weight types, `Embed` to `LmHead`, in 03 §2's declaration order.
    pub const WEIGHTS: [Self; 9] = [
        Self::Embed,
        Self::Q,
        Self::K,
        Self::V,
        Self::O,
        Self::Gate,
        Self::Up,
        Self::Down,
        Self::LmHead,
    ];

    /// The declaration index, from 0: for a weight type, its index `t` in 03 §2 (RFC-0040
    /// 'The matrices').
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Whether a weight of this type exists once per layer: `Q` to `Down`. `Embed` and `LmHead`
    /// have one layer, and the channel parameters are no weight.
    pub const fn per_layer(self) -> bool {
        matches!(
            self,
            Self::Q | Self::K | Self::V | Self::O | Self::Gate | Self::Up | Self::Down
        )
    }
}

/// What a commitment's column chains start from (RFC-0031 'The descriptor'): its purpose, the
/// context it belongs to, its code, its row block and its row count, all public.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub enum CommitmentDescriptor {
    /// A proof unit's advice commitment (RFC-0011).
    Advice {
        /// `H("statement", TCE(PublicStatement))` (RFC-0007).
        statement_digest: Digest,
        /// The proof unit, from 0 (RFC-0018 'Proof units').
        unit: u32,
        /// The claim list's advice layout.
        layout: AdviceLayoutParams,
        /// The unit's non-zero rows, every segment's.
        rows: u64,
    },
    /// One matrix of the weight commitment (RFC-0012, RFC-0040).
    WeightMatrix {
        /// The claim list's weight layout.
        layout: WeightLayoutParams,
        /// The matrix.
        matrix: WeightMatrix,
        /// `log2 k_T`, the matrix's own message width.
        log2_k: u8,
        /// `B`, the parameter set's row block.
        row_block: u32,
        /// The committed data rows, `m_T`, and from the version that adds them, the `B_W`
        /// blinding rows (RFC-0040): `m_T` in `tessera-protocol/1`.
        rows: u64,
        /// The epoch: 0 at the first commitment, `e + 1` after a refresh.
        epoch: u32,
    },
}

impl CommitmentDescriptor {
    /// `commitment_id = H("commitment-id", TCE(descriptor))`, the 32 bytes every column chain of
    /// the commitment starts from (RFC-0031, RFC-0004 'Column digests').
    pub fn commitment_id(&self) -> Digest {
        hash(Domain::CommitmentId, &encode(self))
    }
}
