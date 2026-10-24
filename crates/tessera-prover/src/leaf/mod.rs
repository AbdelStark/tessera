// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The leaf stage's lookup layout and its split (RFC-0045): how each LogUp instance lays out its
//! lookups, and how its leaf claims become one claim per tensor that a site's lookups read, at
//! one point per size.
//!
//! - **Blocks.** A block is one lookup of one gadget site at one layer, over its domain padded
//!   per dimension: `2^k` entries. [`LeafLayout::new`] derives every block from public data, the
//!   sites in node order ([`gadget_nodes`], [`Site::requant`]), and groups them by table, one
//!   instance per table in table order.
//! - **The column.** An instance's blocks come by site, layer and lookup, then a stable sort puts
//!   larger ones first. Offsets are prefix sums, so a block of `2^k` entries starts at a multiple
//!   of `2^k`. With `m` the total, the column has `2^h` entries, `h = ⌈log2 m⌉`, the last
//!   `2^h − m` holding row 0's tuple.
//! - **The leaves.** The lookup tree's leaf `i` holds column entry `σ_h(i)` ([`leaf_index`]): bit
//!   0 kept, bits 1 to `h − 1` reversed. The tree levels front-pad their instances, so a tree of
//!   height `h` ends at `(ρ_{H−1}, r_{n−h+1}, …, r_{n−1})`. Read in column order, its claims lie
//!   at `P[0..h]`, with `P = (ρ_{H−1}, r_{n−1}, …, r_0)` the last level's challenges in reverse
//!   ([`leaf_point`]), and a block of `2^k` entries at `P[0..k]` in every instance, whatever its
//!   height:
//!
//!   ```text
//!   column~(P[0..h]) = Σ_b eq(P[k_b..h], o_b / 2^{k_b})·block~_b(P[0..k_b]) + (1 − LT(P[0..h], m))·t_0
//!   ```
//! - **The split.** A lookup's key and value are affine forms of tensors, so a block's extension
//!   at `P[0..k]` is its forms at the tensors' values `X̃_t(π_t(P[0..k]))`, `π_t` the coordinates
//!   of the axes `X_t` is read over. The prover sends those values under `stage/0/split`, after
//!   `logup/values`: site by site in node order, layer by layer, tensor by tensor in the order the
//!   lookups first read them ([`split_values`]). The verifier checks each instance's key and value
//!   claims by the identity above, `Reject(Sumcheck { stage: 0, round: 0 })` on a mismatch, and
//!   each value is then RFC-0037's per-layer claim on its tensor.
//!
//! - **Token groups.** A leaf stage larger than memory is proven in groups (RFC-0054): group `h`
//!   holds the blocks of its layers, restricted to its tokens, with the token axis's low bits set
//!   to its own ([`SiteLookup::token`]). [`GroupedLayout::new`] derives the merge's polynomials,
//!   one per value of the ungrouped split, and [`GroupedLayout::group`] lays out one group at a
//!   time: its instances by RFC-0045 over its blocks, and its split as claims on the polynomials.
//!
//! [`prove_leaf`] runs it all on a transcript: the columns ([`LeafColumns`]), `tessera-logup`'s
//! [`prove_logup`](tessera_logup::prove_logup), the leaf point and the split.
//!
//! The layout is public: the verifier derives the same sites, blocks, offsets and heights from the
//! claim list. The tensors, the columns and the multiplicities are secret-class (05 §3): the caller
//! lends the tensors in [`SiteTensors`], and what is computed from them lives in `SecretBuf`s.

mod claims;
mod grouped;
mod groups;
mod layout;
mod model;
mod model_witness;
pub use model::ModelColumn;
pub use model_witness::ModelWitness;
mod prove;

use core::fmt;

use tessera_codec::error::ErrorCode;
use tessera_gadgets::GadgetError;
use tessera_logup::LogUpError;

pub use claims::{GroupedClaims, LeafClaims};
pub use grouped::{
    MergeMode, ProvenGroupedLeaf, StageForest, balanced_j_star, grouped_multiplicities,
    prove_grouped_leaf,
};
pub use groups::{GroupLayout, GroupRead, GroupedLayout};
pub use layout::{
    Block, InstanceLayout, LeafLayout, Read, Site, SiteGadget, SiteLookup, gadget_nodes, leaf_index,
};
pub(crate) use layout::{token, tokens};
pub use prove::{
    InstanceColumns, LeafColumns, PreparedLeaf, ProvenLeaf, SPLIT_LABEL, SiteTensors, leaf_point,
    prepare_leaf, prove_leaf, split_values,
};

/// Why the leaf stage's layout, columns or split could not be built.
///
/// The layout follows from public data, and an honest witness fits it, so every variant but
/// [`Memory`](Self::Memory) or a tile-source error is a bug upstream (`ERR-INTERNAL`).
/// Tile-source errors preserve their original code. The variants carry rules, never a
/// value (05 §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafError {
    /// The sites, the witness or the tables break a rule of the layout (`ERR-INTERNAL`).
    Internal {
        /// The rule broken.
        rule: &'static str,
    },
    /// A gadget refused its selection or its tables (`ERR-INTERNAL`).
    Gadget(GadgetError),
    /// The canonical witness tile source failed without exposing secret values.
    Witness(tessera_witness::WitnessError),
    /// The LogUp prover refused the instances (`ERR-INTERNAL`).
    LogUp(LogUpError),
    /// A column does not fit in memory (`ERR-RES-MEMORY`).
    Memory,
}

impl LeafError {
    /// The stable code of 04 §3.
    pub fn code(&self) -> ErrorCode {
        match self {
            Self::Internal { .. } | Self::Gadget(_) | Self::LogUp(_) => ErrorCode::Internal,
            Self::Witness(error) => error.code(),
            Self::Memory => ErrorCode::ResMemory,
        }
    }
}

impl fmt::Display for LeafError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Internal { rule } => write!(f, "the leaf layout breaks a rule: {rule}"),
            Self::Gadget(error) => write!(f, "a leaf site's gadget: {error}"),
            Self::Witness(error) => write!(f, "a model leaf witness tile: {error}"),
            Self::LogUp(error) => write!(f, "the leaf stage's LogUp instances: {error}"),
            Self::Memory => f.write_str("a leaf column does not fit in memory"),
        }
    }
}

impl std::error::Error for LeafError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Gadget(error) => Some(error),
            Self::Witness(error) => Some(error),
            Self::LogUp(error) => Some(error),
            Self::Internal { .. } | Self::Memory => None,
        }
    }
}

impl From<GadgetError> for LeafError {
    fn from(error: GadgetError) -> Self {
        Self::Gadget(error)
    }
}

impl From<LogUpError> for LeafError {
    fn from(error: LogUpError) -> Self {
        Self::LogUp(error)
    }
}

fn internal<T>(rule: &'static str) -> Result<T, LeafError> {
    Err(LeafError::Internal { rule })
}
