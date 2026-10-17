// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Claim-list parameters of the compiler's choices (RFC-0007 'Schema').

use tessera_codec::{Decode, Encode};

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
