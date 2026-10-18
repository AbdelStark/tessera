// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The parameter sets of RFC-0014 'Parameter sets' that this verifier accepts, and the
//! commitment layout each prescribes (RFC-0026 'Schema').
//!
//! | Set | Accepted | Advice: `log2 k`, `log2` inverse rate, `P_A`, `B`, `S` | Weights: `log2 k_W`, `log2` inverse rate, `P_W`, `B_W` |
//! |---|---|---|---|
//! | `live-64/1` | always | 22, 1, 160, 32, 8 | 20, 2, 65,536, 4,096 |
//! | `ni-100/1` | always | 22, 1, 200, 32, 8 | 20, 2, 65,536, 4,096 |
//! | `test-small/1` | with the `test-params` feature only | 10, 1, 16, 4, 1 | 8, 2, 64, 32 |
//!
//! `test-small/1` certifies nothing. Its variant exists only in a build with `test-params`, which
//! a release build cannot enable (RFC-0001 'Cargo features'), so no other build can name it, let
//! alone accept it. A set is identified by its exact name and version: `live-64/2` is another set,
//! which this verifier does not know.

use crate::claims::{AdviceLayoutParams, CommitmentLayout, ParamSetId, WeightLayoutParams};

/// A parameter set this verifier accepts (RFC-0014 'Parameter sets').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ParamSet {
    /// `live-64/1`: live sessions, about 64 bits per query against a prover that cannot retry.
    Live64,
    /// `ni-100/1`: non-interactive proofs, with 20 bits of grinding on the seed and the queries.
    Ni100,
    /// `test-small/1`: small parameters for tests, which certify nothing.
    #[cfg(feature = "test-params")]
    TestSmall,
}

/// The weight layout `live-64/1` and `ni-100/1` share: `k_W = 2^20`, rate 1/4, `P_W = 2^16`,
/// `B_W = 4,096`.
const PRODUCTION_WEIGHTS: WeightLayoutParams = WeightLayoutParams {
    log2_k: 20,
    log2_inv_rate: 2,
    mask_coeffs: 65_536,
    blinding_rows: 4_096,
};

impl ParamSet {
    /// Every set this build accepts.
    #[cfg(not(feature = "test-params"))]
    pub const ALL: &'static [ParamSet] = &[ParamSet::Live64, ParamSet::Ni100];

    /// Every set this build accepts.
    #[cfg(feature = "test-params")]
    pub const ALL: &'static [ParamSet] = &[ParamSet::Live64, ParamSet::Ni100, ParamSet::TestSmall];

    /// The set a claim list names, if this verifier accepts it: RFC-0019 'Check order' step 2,
    /// whose failure is `Reject(Params { name })` (RFC-0043 'Parameter sets').
    pub fn accept(name: &ParamSetId) -> Option<ParamSet> {
        Self::ALL
            .iter()
            .copied()
            .find(|set| set.name() == name.as_str())
    }

    /// The set's identifier, as RFC-0014's table writes it.
    pub const fn name(self) -> &'static str {
        match self {
            ParamSet::Live64 => "live-64/1",
            ParamSet::Ni100 => "ni-100/1",
            #[cfg(feature = "test-params")]
            ParamSet::TestSmall => "test-small/1",
        }
    }

    /// The commitment layout the set prescribes, which a claim list naming it must state
    /// (RFC-0026, RFC-0007 'Verifier validation' step 6).
    pub const fn layout(self) -> CommitmentLayout {
        match self {
            ParamSet::Live64 => CommitmentLayout {
                advice: AdviceLayoutParams {
                    log2_k: 22,
                    log2_inv_rate: 1,
                    mask_coeffs: 160,
                    row_block: 32,
                    segments: 8,
                },
                weights: PRODUCTION_WEIGHTS,
            },
            ParamSet::Ni100 => CommitmentLayout {
                advice: AdviceLayoutParams {
                    log2_k: 22,
                    log2_inv_rate: 1,
                    mask_coeffs: 200,
                    row_block: 32,
                    segments: 8,
                },
                weights: PRODUCTION_WEIGHTS,
            },
            #[cfg(feature = "test-params")]
            ParamSet::TestSmall => CommitmentLayout {
                advice: AdviceLayoutParams {
                    log2_k: 10,
                    log2_inv_rate: 1,
                    mask_coeffs: 16,
                    row_block: 4,
                    segments: 1,
                },
                weights: WeightLayoutParams {
                    log2_k: 8,
                    log2_inv_rate: 2,
                    mask_coeffs: 64,
                    blinding_rows: 32,
                },
            },
        }
    }
}
