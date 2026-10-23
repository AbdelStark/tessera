// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `Round1` and the public statement it carries (RFC-0036 '`Round1`', 03 §4, RFC-0007 'Public
//! statement', RFC-0017 'Decoding and the sampler PRF').
//!
//! Every field is public (05 §3): digests, roots, ids, lengths and the predictor.

use tessera_codec::{Decode, DecodeError, Encode, Rational, Reader};
use tessera_hash::{Digest, Domain, Hasher};

pub use tessera_pcs::AdviceCommitment;

/// Step 1: the prover's commitments (schema `0x0510`).
///
/// `tessera-protocol/1` burns no blinding rows, so `Round1` holds no weight reservation: the
/// epoch is `statement.weights_epoch` (RFC-0036 '`Round1`').
#[derive(Clone, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub struct Round1 {
    /// The protocol's number, `tessera-protocol/{protocol_version}`:
    /// [`PROTOCOL_VERSION`](crate::PROTOCOL_VERSION), the claim list's. It is 0 until
    /// `tessera-protocol/1` is assigned (09 §3), then 1.
    pub protocol_version: u16,
    /// `SeedCommit`'s session in live mode, chosen fresh by the prover in non-interactive mode;
    /// `SessionContext` binds it either way.
    pub session_id: [u8; 16],
    /// What the proof is about (03 §4).
    pub statement: PublicStatement,
    /// The advice commitment (RFC-0011).
    pub advice: AdviceCommitment,
    /// How the leaf stage is grouped (RFC-0054 'Messages'), absorbed with `Round1` before any
    /// challenge: [`LeafGroups::NONE`] for a leaf stage proven in one batch.
    pub leaf_groups: LeafGroups,
}

/// The most group bits a leaf stage takes, `g_T + g_L ≤ 14`: at most `2^{14}` groups, the most
/// RFC-0054's union bound is evaluated at (RFC-0054 'Soundness').
pub const MAX_LEAF_GROUP_BITS: u8 = 14;

/// A leaf stage's token groups (RFC-0054 'Groups'): `2^{token_bits}` token subsets times
/// `2^{layer_bits}` layer subsets.
///
/// Its decoding refuses `token_bits + layer_bits > 14` with `ERR-CODEC-LIMIT` at its first
/// byte. The bounds the claim list and the statement set, `token_bits ≤ t` and
/// `layer_bits ≤ n_ℓ`, are the verifier's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Encode)]
pub struct LeafGroups {
    /// `g_T`: the low bits of the token index each group fixes.
    pub token_bits: u8,
    /// `g_L`: the low bits of the layer index each group fixes.
    pub layer_bits: u8,
}

impl LeafGroups {
    /// An ungrouped leaf stage, `g = 0`.
    pub const NONE: Self = Self {
        token_bits: 0,
        layer_bits: 0,
    };

    /// `g = g_T + g_L`.
    pub fn bits(self) -> u8 {
        self.token_bits.saturating_add(self.layer_bits)
    }
}

/// The fields in order, then the bound on their sum.
impl Decode for LeafGroups {
    const MIN_SIZE: usize = 2;

    fn decode_from(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        let at = reader.position();
        reader.enter()?;
        let value = Self {
            token_bits: Decode::decode_from(reader)?,
            layer_bits: Decode::decode_from(reader)?,
        };
        reader.leave();
        if u16::from(value.token_bits) + u16::from(value.layer_bits)
            > u16::from(MAX_LEAF_GROUP_BITS)
        {
            return Err(reader.limit(at));
        }
        Ok(value)
    }
}

/// The public statement (03 §4, RFC-0007 'Public statement'). The verifier compares it with the
/// statement its own inputs give, field by field in this order (RFC-0019 step 4).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub struct PublicStatement {
    /// `H("claim-list", TCE(ClaimList))` (RFC-0007).
    pub claim_list_digest: Digest,
    /// `R_W`, the weight commitment's root (RFC-0012).
    pub weights_root: Digest,
    /// The epoch of `R_W`.
    pub weights_epoch: u32,
    /// The `LogManifest` the batch is anchored to (RFC-0015).
    pub manifest_digest: Digest,
    /// The requests proven.
    pub batch: BatchDescriptor,
    /// The scoring predictor (RFC-0017).
    pub predictor: PredictorSpec,
}

impl PublicStatement {
    /// `H("statement", TCE(PublicStatement))`, the `statement_digest` that `SessionContext`
    /// binds (RFC-0007 'Public statement').
    pub fn digest(&self) -> Digest {
        let mut hasher = Hasher::new(Domain::Statement);
        self.encode_to(&mut hasher);
        hasher.finalize()
    }
}

/// The batch's requests (03 §4).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub struct BatchDescriptor {
    /// One entry per request, in batch order.
    pub requests: Vec<RequestRef>,
}

/// One request of the batch (03 §4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub struct RequestRef {
    /// The request's id in the digest log.
    pub request_id: u64,
    /// The serving session it was recorded in (RFC-0015).
    pub session_id: [u8; 16],
    /// Its prompt's tokens.
    pub prompt_len: u32,
    /// Its output's tokens.
    pub output_len: u32,
}

/// The scoring predictor (RFC-0017 'Decoding and the sampler PRF').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub enum PredictorSpec {
    /// `q_t` is a point mass on `D`'s decoded token: `Û = 0` iff the outputs are `D`'s.
    Exact {
        /// How `D` decodes.
        decoding: Decoding,
    },
    /// `q_t = (1 − ε)·δ_ŷ + ε·p̂`, for bounded mode.
    Mixture {
        /// How `D` decodes.
        decoding: Decoding,
        /// `k` of `ε = 2^{−k}`: 0 to 16, 6 by default.
        eps_log2: u8,
    },
}

/// How `D` decodes (RFC-0017).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub enum Decoding {
    /// The argmax token.
    Greedy,
    /// Inverse-CDF sampling at temperature `t` from the committed sampler PRF.
    Temperature {
        /// `0 < t ≤ 4`, an exact fraction.
        t: Rational,
    },
}
