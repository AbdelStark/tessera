// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The parameter sets (RFC-0014 'Parameter sets', 02 §5): named, versioned bundles, the only
//! source of every parameter either side uses (INV-PROTO-5).
//!
//! | | `live-64/1` | `ni-100/1` | `test-small/1` |
//! |---|---|---|---|
//! | Mode | Live | Non-interactive | Either |
//! | Advice: `log2 k`, rate, mask `P_A` | 22, 1/2, 160 | 22, 1/2, 200 | 10, 1/2, 16 |
//! | Advice `τ_A` | 155 | 193 | 8 |
//! | Weights: `log2 k_W`, rate, mask `P_W`, blinding `B_W` | 20, 1/4, 2^16, 4096 | 20, 1/4, 2^16, 4096 | 8, 1/4, 64, 32 |
//! | Weights `τ_W` | 97 | 122 | 8 |
//! | `g_seed`, `g_query` | 0, 0 | 20, 20 | 0, 0 |
//! | Row block `B`, segments `S` | 32, 8 | 32, 8 | 4, 1 |
//! | Accepted by a release verifier | yes | yes | no |
//! | Targets: every row, the one-shot rows | 100, 64 | 100, 100 | none |
//!
//! The layouts are those RFC-0026 'Schema' writes into the claim list. The targets are SP-1's
//! (06 §4), as RFC-0014 'Soundness calculator' assigns them to the rows of the soundness table.

use tessera_codec::Ascii;
use tessera_hash::{Mode, ParamSetId};
use tessera_ir::{AdviceLayoutParams, CommitmentLayout, WeightLayoutParams};

use crate::ProtocolError;

/// The modes a parameter set's proofs may run in (RFC-0014 'Parameter sets').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ParamMode {
    /// A live session only: the verifier tosses the seed's coin and draws the queries.
    Live,
    /// A non-interactive proof only: the prover grinds for the seed and the queries.
    NonInteractive,
    /// Either mode: `test-small/1`.
    Either,
}

impl ParamMode {
    /// Whether a proof in `mode` may use the set.
    pub const fn allows(self, mode: Mode) -> bool {
        matches!(
            (self, mode),
            (Self::Either, _)
                | (Self::Live, Mode::Live)
                | (Self::NonInteractive, Mode::NonInteractive)
        )
    }
}

/// The bits every row of a parameter set's soundness report must reach (SP-1, RFC-0014
/// 'Soundness calculator').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Targets {
    /// The target of every row a prover can retry: 100 bits.
    pub bits: u32,
    /// The target of the rows a proof of work covers, the LogUp identity and each commitment's
    /// column queries. A live session draws their challenges once, from the coin-tossed seed and
    /// the verifier's draw: 64 bits at `live-64/1`, SP-1's per live audit. A non-interactive
    /// proof retries them like any other row, at the cost of its proofs of work: 100 bits at
    /// `ni-100/1`.
    pub one_shot: u32,
}

/// A parameter set (RFC-0014 'Parameter sets').
///
/// The three sets of RFC-0014 are [`LIVE_64`](Self::LIVE_64), [`NI_100`](Self::NI_100) and
/// [`TEST_SMALL`](Self::TEST_SMALL). [`accept`](Self::accept) is how a proof takes one from the
/// identifier its claim list names, and refuses `test-small/1` in every build without the
/// `test-params` feature. A value built by hand describes a hypothetical set for the soundness
/// calculator, and no proof can use it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ParamSet {
    /// The identifier, its name and version: `live-64/1` (RFC-0014's [`ParamSetId`]).
    pub id: &'static str,
    /// The modes its proofs may run in.
    pub mode: ParamMode,
    /// The commitments' layout the claim list states (RFC-0026 'Schema'). The advice layout
    /// holds the row block `B` and the segments `S`, which the weights' chains share.
    pub layout: CommitmentLayout,
    /// `τ_A`, the advice column queries.
    pub advice_queries: u32,
    /// `τ_W`, the column queries of each opened weight matrix.
    pub weight_queries: u32,
    /// `g_seed`, the leading zero bits of the seed's proof of work (RFC-0014 'Grinding').
    pub seed_pow: u32,
    /// `g_query`, the leading zero bits of the queries' proof of work.
    pub query_pow: u32,
    /// Whether a release verifier accepts it.
    pub release: bool,
    /// The bits its soundness report must reach, or `None` for a set held to none.
    pub targets: Option<Targets>,
}

impl ParamSet {
    /// `live-64/1`: live sessions, 64-bit proximity per audit and 100 bits wherever a prover can
    /// retry.
    pub const LIVE_64: Self = Self {
        id: "live-64/1",
        mode: ParamMode::Live,
        layout: CommitmentLayout {
            advice: AdviceLayoutParams {
                log2_k: 22,
                log2_inv_rate: 1,
                mask_coeffs: 160,
                row_block: 32,
                segments: 8,
            },
            weights: WeightLayoutParams {
                log2_k: 20,
                log2_inv_rate: 2,
                mask_coeffs: 1 << 16,
                blinding_rows: 4096,
            },
        },
        advice_queries: 155,
        weight_queries: 97,
        seed_pow: 0,
        query_pow: 0,
        release: true,
        targets: Some(Targets {
            bits: 100,
            one_shot: 64,
        }),
    };

    /// `ni-100/1`: non-interactive proofs, at least 100 bits throughout, with 20 bits of grinding
    /// before the seed and before the queries.
    pub const NI_100: Self = Self {
        id: "ni-100/1",
        mode: ParamMode::NonInteractive,
        layout: CommitmentLayout {
            advice: AdviceLayoutParams {
                log2_k: 22,
                log2_inv_rate: 1,
                mask_coeffs: 200,
                row_block: 32,
                segments: 8,
            },
            weights: WeightLayoutParams {
                log2_k: 20,
                log2_inv_rate: 2,
                mask_coeffs: 1 << 16,
                blinding_rows: 4096,
            },
        },
        advice_queries: 193,
        weight_queries: 122,
        seed_pow: 20,
        query_pow: 20,
        release: true,
        targets: Some(Targets {
            bits: 100,
            one_shot: 100,
        }),
    };

    /// `test-small/1`: tiny models in tests, in either mode. It certifies nothing, so it is held
    /// to no target, and only a build with the `test-params` feature accepts it.
    pub const TEST_SMALL: Self = Self {
        id: "test-small/1",
        mode: ParamMode::Either,
        layout: CommitmentLayout {
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
        advice_queries: 8,
        weight_queries: 8,
        seed_pow: 0,
        query_pow: 0,
        release: false,
        targets: None,
    };

    /// RFC-0014's parameter sets, in its table's order.
    pub const ALL: [&'static Self; 3] = [&Self::LIVE_64, &Self::NI_100, &Self::TEST_SMALL];

    /// The set's name, its identifier without the version: `live-64`.
    pub fn name(&self) -> &'static str {
        self.id.split_once('/').map_or(self.id, |(name, _)| name)
    }

    /// The identifier a claim list and a transcript bind (RFC-0004 'Transcript', RFC-0007).
    pub fn param_set_id(&self) -> ParamSetId {
        ParamSetId(Ascii::new(self.id).expect("RFC-0014's identifiers are printable ASCII"))
    }

    /// `B`, the rows a column chain absorbs per step, the advice's and every weight matrix's.
    pub fn row_block(&self) -> u32 {
        self.layout.advice.row_block
    }

    /// `S`, the segments the advice rows stream in.
    pub fn segments(&self) -> u32 {
        self.layout.advice.segments
    }

    /// The set of RFC-0014 that `id` names, `test-small/1` among them: what a report can describe,
    /// not what a proof may use.
    pub fn defined(id: &ParamSetId) -> Option<&'static Self> {
        Self::ALL.into_iter().find(|set| set.id == id.0.as_str())
    }

    /// The set a proof takes from the identifier its claim list names (INV-PROTO-5): `live-64/1`
    /// or `ni-100/1`, and `test-small/1` only in a build with the `test-params` feature, which no
    /// release build has (RFC-0001 'Cargo features').
    ///
    /// # Errors
    ///
    /// [`ProtocolError::UnknownParams`] for an identifier RFC-0014 does not define, and
    /// [`ProtocolError::TestParams`] for `test-small/1` without the feature.
    pub fn accept(id: &ParamSetId) -> Result<&'static Self, ProtocolError> {
        match Self::defined(id) {
            None => Err(ProtocolError::UnknownParams),
            Some(set) if set.release || cfg!(feature = "test-params") => Ok(set),
            Some(_) => Err(ProtocolError::TestParams),
        }
    }
}
