// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The claim list (RFC-0007 'Schema', as RFC-0026 amends it): the public description of what a
//! proof certifies about `D`, which the verifier lowers itself, and the parameters it states.

use tessera_codec::{Decode, Encode, encode};
use tessera_hash::{Digest, Domain, ParamSetId, hash};
use tessera_tables::TableSpec;

use crate::{ArchSpec, Profile, PublicQuantSpec};

/// The protocol version a claim list states until `tessera-protocol/1` is assigned (09 §3): no
/// release verifier accepts it.
pub const PROTOCOL_VERSION: u16 = 0;

/// The claim list, schema `0x0201` (RFC-0007 'Schema', RFC-0026): everything the verifier needs to
/// build `D`'s constraints, from public data alone (INV-CL-1).
///
/// Scoring and the anchor join it with their circuits (RFC-0026).
#[derive(Clone, Debug, PartialEq, Eq, Encode, Decode)]
pub struct ClaimList {
    /// The protocol version: [`PROTOCOL_VERSION`] before the first release.
    pub protocol_version: u16,
    /// The architecture (03 §2).
    pub arch: ArchSpec,
    /// The profile.
    pub profile: Profile,
    /// The public quantization data (RFC-0007, RFC-0025).
    pub quant: PublicQuantSpec,
    /// Every table, its definition and digest, sorted by id (RFC-0008).
    pub tables: Vec<TableSpec>,
    /// How gadgets are sized.
    pub gadget_policy: GadgetPolicy,
    /// The commitments' layout: the parameter set's.
    pub layout: CommitmentLayout,
    /// The parameter set (RFC-0014).
    pub params: ParamSetId,
}

impl ClaimList {
    /// `claim_list_digest = H("claim-list", TCE(ClaimList))` (RFC-0007 'Schema').
    pub fn digest(&self) -> Digest {
        hash(Domain::ClaimList, &encode(self))
    }
}

/// The commitments' layout (RFC-0026): the values the parameter set prescribes, which the
/// verifier checks (RFC-0007 'Verifier validation', step 6).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub struct CommitmentLayout {
    /// The streaming advice commitment's (RFC-0011).
    pub advice: AdviceLayoutParams,
    /// The persistent weight commitment's (RFC-0012).
    pub weights: WeightLayoutParams,
}

/// The advice commitment's parameters (RFC-0011 'Column digests and root').
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

/// The weight commitment's parameters (RFC-0012 'Layout', RFC-0026). Each weight type's rows,
/// columns and padding follow from the architecture; `R_W` hashes these.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub struct WeightLayoutParams {
    /// `log2 k_W`, the largest message width: a type's is `min(2^{log2_k}, out_pad · in_pad)`.
    pub log2_k: u8,
    /// `log2` of the inverse rate: 2, rate 1/4.
    pub log2_inv_rate: u8,
    /// `P_W`, each row's mask coefficients.
    pub mask_coeffs: u32,
    /// `B_W`, each matrix's blinding rows.
    pub blinding_rows: u32,
}

/// How gadgets are sized: the claim list states it, and the verifier derives the same gadgets
/// from it (RFC-0007 'Schema', RFC-0006 'Gadget selection').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub struct GadgetPolicy {
    /// `B_max`: the largest table, `2^{B_max}` rows. A key wider than one table splits
    /// (RFC-0009 'RequantSplit'). 32 by default, the most a `Range` table holds.
    pub max_log2_rows: u8,
    /// Positions per attention tile (RFC-0006 'Tiling and the execution plan'): 512 by default.
    pub query_block: u32,
    /// `L`, the limb width of `NonNeg` and `BigIntCompare` (RFC-0009): 16 by default.
    pub limb_bits: u8,
}

impl Default for GadgetPolicy {
    fn default() -> Self {
        Self {
            max_log2_rows: 32,
            query_block: 512,
            limb_bits: 16,
        }
    }
}
