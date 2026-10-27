// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Independent soundness calculator (RFC-0014 'Soundness calculator', RFC-0049 'The calculator').
//!
//! Computes every row of RFC-0014's soundness table from a parameter set and statement sizes using
//! exact integer arithmetic only: no floating point enters any bit count or target decision.
//!
//! # Algorithm
//!
//! For `a >= b >= 1` and precision `W = 62`, `LOG2(a, b, F)` normalizes `a / b` into `[1, 2)` by
//! finding the largest `e` with `d = b · 2^e <= a`, then extracts the 62 fractional bits of
//! `(a - d) / d` by 62 steps of binary long division (shift-and-subtract) before running `F`
//! squaring steps on the `Q2.62` fixed-point value `y`. Every step rounds down, giving a guaranteed
//! lower bound `L <= 2^F · log2(a / b)`.
//!
//! Target pass/fail decisions use exact integer inequalities:
//! - field-form rows (`c / q` where `q = p^2`): `c == 0 || t <= g || c <= ⌊q / 2^{t - g}⌋`;
//! - proximity query rows (`((n + k + P) / (2n))^τ`): `t <= g` or `A^τ >= 2^{t - g} · B^τ`,
//!   decided first by `LOG2(A, B, 48)` when outside its 2-ulp rounding band and by exact
//!   multi-limb integer exponentiation inside the band;
//! - hash collision row: `t <= 128`.
//!
//! # Blinding rows
//!
//! Each matrix states its claims `e_T` in the pool ([`MatrixSize::claims`]). In
//! `tessera-protocol/2` its field term is `ϑ_T = ⌈log2(m_T + B_T)⌉`, with `B_T` derived from
//! `e_T` and never taken as an input ([`MatrixSize::blinding_rows`], RFC-0065 '`ChannelParams`'
//! blinding rows'): `B_W` for `Embed`..`LmHead`, and `B_W + 2·(e_CP − 1)·⌊max(B_W − 4, 0)/6⌋`
//! for `ChannelParams`. It refuses, with `ERR-USAGE` and in this order, a matrix with no claim,
//! which is not opened, `ChannelParams` included; a matrix other than `ChannelParams` with
//! `e_T ≠ 1`; and a matrix with `m_T + B_T = 0`. `tessera-protocol/1` reads no `e_T`: its field
//! term is `⌈log2 m_T⌉`, and a matrix of no committed row has no term.
//!
//! # The leaf stage in token groups
//!
//! [`StatementSizes::groups`] states a grouped leaf stage (RFC-0014 'Statement sizes' `groups`,
//! RFC-0054 'Soundness'): `G` groups, the leaf stage's level rounds by degree, the merge's
//! `n_max` rounds and its `P` claims. With `G > 1`:
//! - the rounds row's `D` also takes each degree of a nonzero count in `level_rounds`, the
//!   merge's degree 2 when `n_max > 0`, and `P + G`, the merge's batching;
//! - the union row's `U` adds `G·(Σ_{level_rounds} count·d + Σ_levels (m + 1)) + 2·n_max + P + G`:
//!   `G` more copies of the leaf stage's levels, each group's and the table batch's costing no
//!   more than the stage's, and the merge.
//!
//! A calculator refuses, with `ERR-USAGE`, a `G` that is not a power of two in `1..=2^14`, a
//! `G > 1` in `tessera-protocol/2`, and a `G = 1` with any grouped size. A statement's report
//! takes the largest `G` the verifier accepts ([`StatementSizes::default_405b`]); a proof's budget
//! its own ([`runtime`](crate::runtime)).
//!
//! # The layer point
//!
//! [`StatementSizes::layer_vars`] is `λ`, the coordinates of stage 1's layer point `ρ_1`
//! (RFC-0014 'Statement sizes' `layer_vars`, RFC-0037 'The layer point', #537). A false
//! per-layer value survives its stacking at `ρ_1` with probability at most `λ/q`, so the rounds
//! row's `D` takes `λ` as an item, counted only when `λ > 0`, and the union row's `U` adds it.
//! A calculator refuses, with `ERR-USAGE` and in this order, a `λ ≠ 0` in `tessera-protocol/2`,
//! whose leaf stage's layer sumcheck, a stage, replaces the layer point (RFC-0068), and a
//! `λ` above [`MAX_LAYER_VARS`].

use core::fmt::{self, Write as _};

use crate::claims::{AdviceLayoutParams, WeightLayoutParams};
use crate::params::ParamSet;
use crate::protocol::MAX_LEAF_GROUP_BITS;
use crate::weights::Matrix;

/// Goldilocks prime `p = 2^64 - 2^32 + 1` (RFC-0002).
pub const GOLDILOCKS_P: u128 = 0xFFFF_FFFF_0000_0001;

/// Quadratic extension order `q = |Fp2| = p^2` (RFC-0014).
pub const EXTENSION_Q: u128 = GOLDILOCKS_P.wrapping_mul(GOLDILOCKS_P);

/// Working fractional precision `W = 62` of `LOG2` (RFC-0014 'Soundness calculator').
pub const LOG2_PRECISION: u32 = 62;

/// Output fractional bits `F = 8` for field-form rows (`FixedBits`, units of `2^-8` bits).
pub const FIELD_FRACTION: u32 = 8;

/// Output fractional bits `F = 48` for proximity query rows before scaling by `τ`.
pub const QUERY_FRACTION: u32 = 48;

/// One bit in `FixedBits` (`2^8 = 256`).
pub const FIXED_BITS_UNIT: u64 = 256;

/// BLAKE3 collision resistance in bits (RFC-0004, RFC-0014).
pub const HASH_BITS: u32 = 128;

/// Maximum query count `τ <= 2^16` accepted by the calculator (RFC-0014).
pub const MAX_QUERIES: u32 = 65_536;

/// Maximum codeword length `n <= 2^32` accepted by the calculator (RFC-0014).
pub const MAX_CODEWORD: u64 = 1u64 << 32;

/// Default `Exp` table lookups at 405B 16K tokens, 16K context: `2^42` (RFC-0014, RFC-0030).
pub const DEFAULT_EXP_LOOKUPS: u64 = 1u64 << 42;

/// Default `Exp` table rows at 405B: `2^22` (RFC-0014).
pub const DEFAULT_EXP_ROWS: u64 = 1u64 << 22;

/// The most leaf groups a calculator accepts, `G = 2^{14}` (RFC-0014 'Statement sizes',
/// RFC-0054 'Messages': `g_T + g_L ≤ 14`).
pub const MAX_GROUPS: u64 = 1u64 << MAX_LEAF_GROUP_BITS;

/// The most coordinates of stage 1's layer point a calculator accepts, `λ ≤ 32` (RFC-0014
/// 'Statement sizes' `layer_vars`, #537).
pub const MAX_LAYER_VARS: u64 = 32;

/// A tree level's round degree, 3: `eq(r, x)` times two evaluations (RFC-0010 'Tree levels').
pub const LEVEL_DEGREE: u64 = 3;

/// The merge's round degree, 2: `W_i·P̃_i` (RFC-0054 'The merge', 'Rounds').
pub const MERGE_DEGREE: u64 = 2;

/// The model split values of the 405B lowering's ungrouped leaf stage, as RFC-0014's default
/// statement states them (RFC-0045 'What is sent'; RFC-0014 'The 405B 16K-token, 16K-context
/// statement', `groups`, before the generation request's greedy key column and the forest).
pub const DEFAULT_SPLIT_VALUES: u64 = 53_489;

/// `e_CP`, the claims of `ChannelParams` at 405B: one per weight type with channel parameters,
/// seven (RFC-0065 'The count', 'The data rows'; RFC-0014 'The 405B 16K-token, 16K-context
/// statement', `matrices`). Each other matrix has one.
pub const DEFAULT_CHANNEL_PARAMS_CLAIMS: u64 = 7;

/// Protocol modes a parameter set supports (RFC-0014 'Parameter sets').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SessionMode {
    /// Interactive coin-tossed session (`live-64/1`).
    Live,
    /// Non-interactive Fiat-Shamir proof with grinding (`ni-100/1`).
    NonInteractive,
    /// Either mode (`test-small/1`).
    Either,
}

impl SessionMode {
    /// The canonical JSON string in `tessera.soundness/1`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Live => "live",
            Self::NonInteractive => "non-interactive",
            Self::Either => "either",
        }
    }

    /// Parses a mode from its `tessera.soundness/1` string.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "live" => Some(Self::Live),
            "non-interactive" => Some(Self::NonInteractive),
            "either" => Some(Self::Either),
            _ => None,
        }
    }
}

/// Bit targets of a parameter set (SP-1, RFC-0014 'Targets').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Targets {
    /// Target in bits for every challenge a prover can retry by varying messages (`100`).
    pub bits: u32,
    /// Target in bits for challenges covered by a proof of work or drawn once in live mode (`64` or `100`).
    pub one_shot: u32,
}

/// Parameter set inputs to the soundness calculator (RFC-0014 'Parameter sets').
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SoundnessParams {
    /// Parameter-set identifier, e.g. `"live-64/1"`.
    pub id: String,
    /// Session mode.
    pub mode: SessionMode,
    /// Whether a release verifier accepts this parameter set.
    pub release: bool,
    /// Advice commitment layout parameters.
    pub advice: AdviceLayoutParams,
    /// Weight commitment layout parameters.
    pub weights: WeightLayoutParams,
    /// Number of opened advice columns `τ_A`.
    pub advice_queries: u32,
    /// Number of opened weight columns `τ_W`.
    pub weight_queries: u32,
    /// Seed proof-of-work grinding bits `g_seed`.
    pub seed_pow: u32,
    /// Query proof-of-work grinding bits `g_query`.
    pub query_pow: u32,
    /// Soundness targets, or `None` for test sets held to no target.
    pub targets: Option<Targets>,
}

const PRODUCTION_WEIGHTS: WeightLayoutParams = WeightLayoutParams {
    log2_k: 20,
    log2_inv_rate: 2,
    mask_coeffs: 65_536,
    blinding_rows: 4_096,
};

impl SoundnessParams {
    /// `live-64/1` (RFC-0014 'Parameter sets').
    pub fn live_64() -> Self {
        Self::live_64_named(String::from("live-64/1"))
    }
    fn live_64_named(id: String) -> Self {
        Self {
            id,
            mode: SessionMode::Live,
            release: true,
            advice: AdviceLayoutParams {
                log2_k: 22,
                log2_inv_rate: 1,
                mask_coeffs: 160,
                row_block: 32,
                segments: 8,
            },
            weights: PRODUCTION_WEIGHTS,
            advice_queries: 155,
            weight_queries: 97,
            seed_pow: 0,
            query_pow: 0,
            targets: Some(Targets {
                bits: 100,
                one_shot: 64,
            }),
        }
    }

    /// `ni-100/1` (RFC-0014 'Parameter sets').
    pub fn ni_100() -> Self {
        Self::ni_100_named(String::from("ni-100/1"))
    }
    fn ni_100_named(id: String) -> Self {
        Self {
            id,
            mode: SessionMode::NonInteractive,
            release: true,
            advice: AdviceLayoutParams {
                log2_k: 22,
                log2_inv_rate: 1,
                mask_coeffs: 200,
                row_block: 32,
                segments: 8,
            },
            weights: PRODUCTION_WEIGHTS,
            advice_queries: 193,
            weight_queries: 122,
            seed_pow: 20,
            query_pow: 20,
            targets: Some(Targets {
                bits: 100,
                one_shot: 100,
            }),
        }
    }

    /// `test-small/1` (RFC-0014 'Parameter sets').
    pub fn test_small() -> Self {
        Self::test_small_named(String::from("test-small/1"))
    }
    fn test_small_named(id: String) -> Self {
        Self {
            id,
            mode: SessionMode::Either,
            release: false,
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
            advice_queries: 8,
            weight_queries: 8,
            seed_pow: 0,
            query_pow: 0,
            targets: None,
        }
    }

    /// Converts a verifier [`ParamSet`] into its calculator parameters.
    pub fn from_param_set(set: ParamSet) -> Self {
        match set {
            ParamSet::Live64 => Self::live_64(),
            ParamSet::Ni100 => Self::ni_100(),
            #[cfg(feature = "test-params")]
            ParamSet::TestSmall => Self::test_small(),
        }
    }

    /// The accepted preset with a fallible name reservation for verifier sessions.
    ///
    /// # Errors
    /// `Memory` when its fixed public identifier cannot be reserved.
    pub fn try_from_param_set(set: ParamSet) -> Result<Self, SoundnessError> {
        let mut name = String::new();
        name.try_reserve_exact(set.name().len())
            .map_err(|_| SoundnessError::Memory)?;
        name.push_str(set.name());
        Ok(match set {
            ParamSet::Live64 => Self::live_64_named(name),
            ParamSet::Ni100 => Self::ni_100_named(name),
            #[cfg(feature = "test-params")]
            ParamSet::TestSmall => Self::test_small_named(name),
        })
    }

    /// Looks up a named parameter set by its CLI name (`"live-64"`, `"ni-100"`, `"test-small"`).
    pub fn by_cli_name(name: &str) -> Option<Self> {
        match name {
            "live-64" => Some(Self::live_64()),
            "ni-100" => Some(Self::ni_100()),
            "test-small" => Some(Self::test_small()),
            _ => None,
        }
    }

    /// Looks up a named parameter set by its versioned identifier (`"live-64/1"`, `"ni-100/1"`, `"test-small/1"`).
    pub fn by_id(id: &str) -> Option<Self> {
        match id {
            "live-64/1" => Some(Self::live_64()),
            "ni-100/1" => Some(Self::ni_100()),
            "test-small/1" => Some(Self::test_small()),
            _ => None,
        }
    }
}

/// A group of sumcheck or tree-level rounds sharing a polynomial degree `d` (RFC-0014).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RoundGroup {
    /// Number of rounds in this group.
    pub count: u64,
    /// Polynomial degree `d` of each round.
    pub degree: u64,
}

/// Lookup and table row counts of one LogUp instance (RFC-0008, RFC-0014).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LookupSize {
    /// Number of lookups `N`.
    pub lookups: u64,
    /// Number of table rows `T`.
    pub rows: u64,
}

/// Dimensions of one opened weight matrix (RFC-0012, RFC-0040, RFC-0014).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatrixSize {
    /// Weight matrix identifier.
    pub matrix: Matrix,
    /// Message width `k_T`.
    pub width: u64,
    /// Committed data rows `m_T`.
    pub rows: u64,
    /// Multilinear variables `ν_T`; the weight-linking row counts `ν_T + 1` (RFC-0014 'Rows').
    pub variables: u64,
    /// `e_T`, the matrix's claims in the pool (RFC-0065 'The count'): 1 for `Embed`..`LmHead`,
    /// `e_CP` for `ChannelParams`.
    ///
    /// In `tessera-protocol/2` the matrix's blinding rows `B_T` follow from it
    /// ([`MatrixSize::blinding_rows`]); a matrix with no claim is refused
    /// ([`SoundnessError::UnopenedMatrix`]), and so is one other than `ChannelParams` with
    /// `e_T ≠ 1` ([`SoundnessError::MatrixClaims`]). `tessera-protocol/1` does not read it.
    pub claims: u64,
}

impl MatrixSize {
    /// `B_T`, the matrix's blinding rows in `tessera-protocol/2`, for `single` the parameter
    /// set's `B_W` (`WeightLayoutParams::blinding_rows`), the blinding rows of a single-claim
    /// matrix (RFC-0065 '`ChannelParams`' blinding rows', RFC-0014 'Statement sizes'):
    ///
    /// - `B_W` for `Embed`..`LmHead`, whatever [`claims`](Self::claims) holds: the calculator
    ///   refuses `e_T ≠ 1` for them before it reads `B_T`;
    /// - for `ChannelParams`, `B_W` plus two rows for each claim beyond the first in each of the
    ///   `N_0 = ⌊max(B_W − 4, 0)/6⌋` proofs epoch 0 gives a single-claim matrix:
    ///   `B_W + 2·(e_CP − 1)·N_0`. A set of fewer than four blinding rows gives no proof,
    ///   `N_0 = 0`, and `B_CP = B_W`. With `e_CP = 0` it is 0: the matrix commits no row, a case
    ///   the calculator refuses in `tessera-protocol/2` before it reads `B_T`.
    ///
    /// The value is exact for every `u64` count and `u32` row count: computed in `u128`, it lies
    /// below `2^95`. The calculator derives it and never takes it as an input.
    pub fn blinding_rows(&self, single: u32) -> u128 {
        let b_w = u128::from(single);
        if self.matrix != Matrix::ChannelParams {
            return b_w;
        }
        let Some(extra) = self.claims.checked_sub(1) else {
            // RFC-0065: with `e_CP = 0` the matrix commits no row and has no blinding rows. The
            // calculator refuses a matrix of no claim in protocol/2 (`UnopenedMatrix`).
            return 0;
        };
        // Each proof of epoch 0 reserves `2·e_CP + 4` rows of `ChannelParams` where a
        // single-claim matrix reserves 6: `2·(e_CP − 1)` more, over the `N_0` proofs the link's
        // last four rows leave (RFC-0064 'The budget'): `⌊max(B_W − 4, 0)/6⌋`.
        let proofs = u128::from(single.saturating_sub(4) / 6);
        let per_proof = u128::from(extra).saturating_mul(2);
        // `per_proof < 2^65` and `proofs < 2^30`, so neither step saturates.
        per_proof.saturating_mul(proofs).saturating_add(b_w)
    }
}

/// Statement sizes passed to the soundness calculator (RFC-0014 'Statement sizes').
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StatementSizes {
    /// Protocol version `n` of `tessera-protocol/n`: `1` or `2`.
    pub protocol: u8,
    /// Sumcheck and tree rounds by count and degree; in `tessera-protocol/2`, also those of the
    /// LogUp identity sumcheck (RFC-0032), the leaf stage's layer sumcheck (RFC-0068), and the
    /// masks' batching of degree 2 (RFC-0070).
    pub rounds: Vec<RoundGroup>,
    /// Instances `m` batched in each reduction stage; in `tessera-protocol/2`, also the LogUp
    /// identity sumcheck (RFC-0032), the leaf stage's layer sumcheck (RFC-0068), and the masks'
    /// batching of `N` mask claims (RFC-0070).
    pub stages: Vec<u64>,
    /// Instances `m` batched in each tree level.
    pub tree_levels: Vec<u64>,
    /// Variables `n_i` of each `ZeroCheck`, including a recomposition's input and bit-axis
    /// variables (RFC-0035, RFC-0072).
    pub zero_checks: Vec<u64>,
    /// `λ`, the coordinates of stage 1's layer point (RFC-0014 'Statement sizes' `layer_vars`,
    /// RFC-0037 'The layer point', #537), 0 when the schedule has none. `tessera-protocol/2` has
    /// none: the leaf stage's layer sumcheck, a stage, replaces it (RFC-0068). At most
    /// [`MAX_LAYER_VARS`].
    pub layer_vars: u64,
    /// Logarithm `s` of the tree count `2^s` in each product forest.
    pub forests: Vec<u64>,
    /// Lookup and table row counts `(N, T)` of each LogUp instance.
    pub lookups: Vec<LookupSize>,
    /// Claims `T_g` in each advice opening group; in `tessera-protocol/2`, also the column
    /// masks' group (RFC-0068), each reduction stage's virtual-tensor fold group (RFC-0013
    /// 'Virtual tensors'), and the masks' batching's group of one (RFC-0070).
    pub claim_groups: Vec<u64>,
    /// Padded row-index domain size `2^{ν - log2 k}` of the advice commitment, including the
    /// masks' pad `S` in `tessera-protocol/2` (RFC-0070).
    pub advice_rows: u64,
    /// Opened weight matrices in strictly increasing [`Matrix`] order.
    pub matrices: Vec<MatrixSize>,
    /// The leaf stage in token groups; [`GroupSizes::NONE`] for an ungrouped stage.
    pub groups: GroupSizes,
}

/// The leaf stage in token groups (RFC-0014 'Statement sizes' `groups`, RFC-0054 'Soundness').
///
/// For a proof's budget, `count` is the proof's own `G = 2^{g_T + g_L}`, 1 for an ungrouped
/// proof; for a statement's report, the most groups the verifier accepts for the statement,
/// `2^{min(14, t + n_ℓ)}`, and 1 in `tessera-protocol/2`. With `count = 1` the other three are
/// empty and zero.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GroupSizes {
    /// `G`, the groups.
    pub count: u64,
    /// With `G > 1`, the leaf stage's level rounds by count and degree, which
    /// [`StatementSizes::rounds`] also counts once.
    pub level_rounds: Vec<RoundGroup>,
    /// With `G > 1`, `n_max`: the merge's rounds, the most variables of a merged polynomial.
    pub merge_rounds: u64,
    /// With `G > 1`, `P`: the merged polynomials' count, which bounds every group's claims.
    pub merge_claims: u64,
}

impl GroupSizes {
    /// An ungrouped leaf stage: `G = 1` and no grouped size.
    pub const NONE: Self = Self {
        count: 1,
        level_rounds: Vec::new(),
        merge_rounds: 0,
        merge_claims: 0,
    };

    /// Whether the stage runs in more than one group, `G > 1`: only then do its sizes count.
    pub const fn grouped(&self) -> bool {
        self.count > 1
    }
}

/// Per-token lookup counts and table row counts of the 26 table entries in the 405B servable
/// commitment report (`docs/perf/v0.2/commitment-report-405b-servable-c2-b32.json`, RFC-0014).
#[derive(Clone, Copy)]
enum TableFamily {
    Other,
    Exp,
    Range,
}

const REPORT_405B_TABLES: [(TableFamily, u64, u64); 26] = [
    (TableFamily::Other, 0, 8_388_608),
    (TableFamily::Exp, 0, DEFAULT_EXP_ROWS),
    (TableFamily::Other, 6_709_248, 1_073_741_824),
    (TableFamily::Range, 253, 4_294_967_296),
    (TableFamily::Range, 65_271, 65_536),
    (TableFamily::Range, 759, 128),
    (TableFamily::Range, 23_782, 256),
    (TableFamily::Range, 506, 16_777_216),
    (TableFamily::Range, 1_012, 33_554_432),
    (TableFamily::Range, 4_130_286, 67_108_864),
    (TableFamily::Range, 2_530, 134_217_728),
    (TableFamily::Range, 16_634, 131_072),
    (TableFamily::Range, 506, 32_768),
    (TableFamily::Range, 15_686, 2),
    (TableFamily::Range, 2_209_792, 2_097_152),
    (TableFamily::Other, 16_384, 1_048_576),
    (TableFamily::Other, 2_193_408, 16_777_216),
    (TableFamily::Other, 6_838_272, 1_073_741_824),
    (TableFamily::Range, 32_256, 4),
    (TableFamily::Range, 2_064_384, 1_048_576),
    (TableFamily::Other, 2_064_384, 1_048_576),
    (TableFamily::Other, 2_064_384, 1_073_741_824),
    (TableFamily::Other, 4_128_768, 8_388_608),
    (TableFamily::Other, 4_128_768, 67_108_864),
    (TableFamily::Other, 6_709_248, 65_536),
    (TableFamily::Other, 2_064_384, 4_294_967_296),
];

impl StatementSizes {
    /// The default 405B 16K-token, 16K-context statement under `tessera-protocol/1`
    /// (RFC-0014 'The 405B 16K-token, 16K-context statement', RFC-0040 'The matrices').
    ///
    /// Models `servable`'s composed schedule at 16,384 positions with a generation request
    /// (RFC-0010 'Stages', RFC-0050, RFC-0053, RFC-0056; #566):
    /// - `stages`: `[123, 98]` (90 of stage 1's instances are the three norms' root products and
    ///   90 of stage 2's their `SumSquares`);
    /// - `lookups`: 27 table entries — the commitment report's 26 model entries plus RFC-0053's
    ///   greedy `Range(29)` (`2^29` rows and `2^14 · 2^17 = 2^31` lookups), giving 42
    ///   `tree_levels` of `2 · 27 + 1 = 55` instances with the `SoftmaxRow` forest;
    /// - `zero_checks`: 14 instances of 42 variables (13 model checks plus the request's logit
    ///   check);
    /// - `groups`: `G = 2^14`, 861 level rounds of degree 3, `n_max = 42`, and `P = 53,491`
    ///   merged polynomials ([`DEFAULT_SPLIT_VALUES`] model split values, the greedy key column,
    ///   and the forest).
    ///
    /// Each matrix has one claim and `ChannelParams` [`DEFAULT_CHANNEL_PARAMS_CLAIMS`], so that
    /// under `tessera-protocol/2` its `B_CP` is 12,280 at `B_W = 4096` and its `ϑ` 14 (RFC-0065
    /// '`ChannelParams`' blinding rows'). Stage 1's layer point has `λ = ⌈log2 126⌉ = 7`
    /// coordinates ([`StatementSizes::layer_vars`]), which a statement under `tessera-protocol/2`
    /// sets to 0.
    pub fn default_405b() -> Self {
        const TOKENS: u64 = 16_384;
        const CONTEXT: u64 = 16_384;
        // The most variables of the lowering's split tensors, as RFC-0014's default statement
        // states them (`groups`: "above the split tensors' 35").
        const DEFAULT_SPLIT_VARIABLES: u64 = 35;
        const LAYERS: u64 = 126;
        const D_MODEL: u64 = 16_384;
        const HEADS: u64 = 128;
        const KV_HEADS: u64 = 8;
        const D_HEAD: u64 = 128;
        const D_FF: u64 = 53_248;
        const VOCAB: u64 = 128_256;
        const LOG2_KW: u32 = 20;
        // RFC-0053's runtime greedy certificate appends `Range(29)` (`2^29` rows, analyzed logit
        // span `< 2^29`) and contributes one greedy key column to `groups.merge_claims` (RFC-0014
        // `lookups`, `groups`; #566).
        const GREEDY_ROWS: u64 = 1u64 << 29;
        const GREEDY_SPLIT_VALUES: u64 = 1;

        let l_pad = LAYERS.next_power_of_two();
        let h_pad = HEADS.next_power_of_two();
        let padded_extra_layers = l_pad.saturating_sub(LAYERS);

        let mut lookups = Vec::with_capacity(REPORT_405B_TABLES.len().saturating_add(1));
        let mut tallest = 1u64;
        for (family, per_token, rows) in REPORT_405B_TABLES {
            let count = match family {
                TableFamily::Exp => l_pad
                    .saturating_mul(h_pad)
                    .saturating_mul(TOKENS)
                    .saturating_mul(TOKENS),
                TableFamily::Range => {
                    let softmax_per_row = match rows {
                        131_072 => 1u64,
                        65_536 => 4u64,
                        4 => 2u64,
                        _ => 0u64,
                    };
                    let extra = padded_extra_layers
                        .saturating_mul(HEADS)
                        .saturating_mul(TOKENS)
                        .saturating_mul(softmax_per_row);
                    per_token.saturating_mul(TOKENS).saturating_add(extra)
                }
                TableFamily::Other => per_token.saturating_mul(TOKENS),
            };
            tallest = tallest.max(count).max(rows);
            lookups.push(LookupSize {
                lookups: count,
                rows,
            });
        }
        let greedy_lookups = TOKENS.saturating_mul(VOCAB.next_power_of_two());
        tallest = tallest.max(greedy_lookups).max(GREEDY_ROWS);
        lookups.push(LookupSize {
            lookups: greedy_lookups,
            rows: GREEDY_ROWS,
        });

        let softmax_forest = u64::from(ceil_log2_u64(
            l_pad.saturating_mul(HEADS).saturating_mul(TOKENS),
        ));
        let forests = vec![softmax_forest];
        let levels = u64::from(ceil_log2_u64(tallest));
        let level_count = usize::try_from(levels).unwrap_or(0);
        let instances_per_level = u64::try_from(lookups.len())
            .unwrap_or(0)
            .saturating_mul(2)
            .saturating_add(u64::try_from(forests.len()).unwrap_or(0));

        // RFC-0054 'Soundness', 'A statement's report': the most groups the verifier accepts,
        // 2^{min(14, t + n_ℓ)}, at t = ⌈log2 16,384⌉ = 14 token bits and n_ℓ = ⌈log2 126⌉ = 7
        // layer bits (RFC-0054 'Messages').
        let group_bits = ceil_log2_u64(TOKENS)
            .saturating_add(ceil_log2_u64(LAYERS))
            .min(u32::from(MAX_LEAF_GROUP_BITS));
        // A SoftmaxRow tree runs over one query row's context positions: height ⌈log2 16,384⌉ =
        // 14. Its leaf table holds the forest's s variables and those 14, the most of any merged
        // polynomial, above the split tensors' 35 that RFC-0014 states.
        let row_height = u64::from(ceil_log2_u64(CONTEXT));
        let n_max = forests
            .iter()
            .map(|s| s.saturating_add(row_height))
            .fold(DEFAULT_SPLIT_VARIABLES, u64::max);
        let groups = GroupSizes {
            count: 1u64.checked_shl(group_bits).unwrap_or(0),
            level_rounds: vec![rounds_below(levels)],
            merge_rounds: n_max,
            // One merged polynomial per ungrouped model split value, the generation request's
            // greedy key column (RFC-0053), and one per forest leaf table (RFC-0054 'The merge').
            merge_claims: DEFAULT_SPLIT_VALUES
                .saturating_add(GREEDY_SPLIT_VALUES)
                .saturating_add(u64::try_from(forests.len()).unwrap_or(u64::MAX)),
        };

        Self {
            protocol: 1,
            rounds: vec![RoundGroup {
                count: 10_000,
                degree: 7,
            }],
            // `servable`'s two reduction stages on the composed schedule at 16K positions with a
            // generation request (RFC-0010 'Stages', RFC-0050, RFC-0053, RFC-0056; #566).
            stages: vec![123, 98],
            tree_levels: vec![instances_per_level; level_count],
            // 13 model zero-checks (`OneHot`'s four, first checkpoint equality, three norms' six
            // `RootPad`/`ResidualPad`, `SoftmaxRow`'s two) plus the generation request's logit
            // check, each bounded by the score tensor's 42 variables (RFC-0014 `zero_checks`; #566).
            zero_checks: vec![42; 14],
            // Stage 1's layer point under tessera-protocol/1: λ = ⌈log2 126⌉ = 7 coordinates
            // (RFC-0037 'Layers and kinds', 'The layer point'; #537).
            layer_vars: u64::from(ceil_log2_u64(LAYERS)),
            forests,
            lookups,
            claim_groups: vec![64; 16],
            advice_rows: 2_328_576_u64.next_power_of_two(),
            matrices: weight_matrices_405b(
                LOG2_KW, LAYERS, D_MODEL, HEADS, KV_HEADS, D_HEAD, D_FF, VOCAB,
            ),
            groups,
        }
    }

    /// The default statement under RFC-0014's `--lookups N` and `--table-rows T`: its LogUp
    /// instances replaced by `instance`, one of `N` lookups and `T` rows.
    ///
    /// The leaf stage's levels then follow the tallest tree. When `L = ⌈log2 max(N, T)⌉`
    /// exceeds the default's 42 levels, `tree_levels` holds `L` levels of the default's 55
    /// instances, `groups.level_rounds` `Σ_{ℓ<L} ℓ` rounds of degree 3, and `groups.merge_rounds`
    /// `L`, which bounds every merged polynomial's variables. Every other size is the default's.
    pub fn default_405b_with_instance(instance: LookupSize) -> Self {
        let mut statement = Self::default_405b();
        let levels = u64::from(ceil_log2_u64(instance.lookups.max(instance.rows)));
        let default_levels = u64::try_from(statement.tree_levels.len()).unwrap_or(u64::MAX);
        if levels > default_levels {
            let per_level = statement.tree_levels.iter().copied().max().unwrap_or(0);
            statement.tree_levels = vec![per_level; usize::try_from(levels).unwrap_or(0)];
            if statement.groups.grouped() {
                statement.groups.level_rounds = vec![rounds_below(levels)];
                statement.groups.merge_rounds = levels;
            }
        }
        statement.lookups = vec![instance];
        statement
    }
}

/// The rounds of a tallest tree's `levels` levels: level `ℓ` has `ℓ` rounds of degree 3, so
/// `Σ_{ℓ<levels} ℓ` of them (RFC-0010 'Tree levels'; RFC-0014's default statement, `groups`).
fn rounds_below(levels: u64) -> RoundGroup {
    RoundGroup {
        count: levels.saturating_mul(levels.saturating_sub(1)) / 2,
        degree: LEVEL_DEGREE,
    }
}

#[allow(clippy::too_many_arguments)]
fn weight_matrices_405b(
    log2_kw: u32,
    layers: u64,
    d_model: u64,
    heads: u64,
    kv_heads: u64,
    d_head: u64,
    d_ff: u64,
    vocab: u64,
) -> Vec<MatrixSize> {
    let kw = 1u64.checked_shl(log2_kw).unwrap_or(u64::MAX);
    let l_pad = layers.next_power_of_two();
    let l_bits = u64::from(ceil_log2_u64(l_pad));
    let q_dim = heads.saturating_mul(d_head);
    let kv_dim = kv_heads.saturating_mul(d_head);
    let specs = [
        (Matrix::Embed, d_model, vocab, 1u64),
        (Matrix::Q, q_dim, d_model, layers),
        (Matrix::K, kv_dim, d_model, layers),
        (Matrix::V, kv_dim, d_model, layers),
        (Matrix::O, d_model, q_dim, layers),
        (Matrix::Gate, d_ff, d_model, layers),
        (Matrix::Up, d_ff, d_model, layers),
        (Matrix::Down, d_model, d_ff, layers),
        (Matrix::LmHead, vocab, d_model, 1u64),
    ];
    let mut out = Vec::with_capacity(10);
    for (matrix, rows_dim, cols_dim, matrix_layers) in specs {
        let out_pad = rows_dim.next_power_of_two();
        let in_pad = cols_dim.next_power_of_two();
        let full = out_pad.saturating_mul(in_pad);
        let width = in_pad.max(kw.min(full));
        let group = (width.checked_div(in_pad).unwrap_or(1)).max(1);
        let rows_per_layer = rows_dim
            .saturating_add(group.saturating_sub(1))
            .checked_div(group)
            .unwrap_or(0);
        let committed_rows = matrix_layers.saturating_mul(rows_per_layer);
        let layer_vars = if matrix_layers > 1 { l_bits } else { 0 };
        let variables = layer_vars
            .saturating_add(u64::from(ceil_log2_u64(out_pad)))
            .saturating_add(u64::from(ceil_log2_u64(in_pad)));
        out.push(MatrixSize {
            matrix,
            width,
            rows: committed_rows,
            variables,
            claims: 1,
        });
    }
    let c = 1u64 << 16;
    let cp_full = 16u64.saturating_mul(l_pad).saturating_mul(c);
    let cp_width = kw.min(cp_full);
    let cp_vars = 4u64
        .saturating_add(l_bits)
        .saturating_add(u64::from(ceil_log2_u64(c)));
    out.push(MatrixSize {
        matrix: Matrix::ChannelParams,
        width: cp_width,
        rows: 882,
        variables: cp_vars,
        claims: DEFAULT_CHANNEL_PARAMS_CLAIMS,
    });
    out
}

/// Proof-of-work challenge that covers a one-shot soundness row (RFC-0014).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CoveredBy {
    /// Covered by the seed proof of work (`g_seed`).
    SeedPow,
    /// Covered by the column-query proof of work (`g_query`).
    QueryPow,
}

impl CoveredBy {
    /// The canonical JSON string in `tessera.soundness/1`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SeedPow => "seed-pow",
            Self::QueryPow => "query-pow",
        }
    }
}

/// Commitment named by a proximity queries or proximity field-term row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CommitmentName {
    /// The advice commitment (`"advice"`).
    Advice,
    /// One of the ten weight commitment matrices.
    Weight(Matrix),
}

impl CommitmentName {
    /// The canonical JSON string in `tessera.soundness/1`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Advice => "advice",
            Self::Weight(m) => m.name(),
        }
    }
}

/// Kind of row in RFC-0014's soundness table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RowKind {
    /// `"rounds"`: largest single-challenge error across sumcheck and tree rounds.
    Rounds,
    /// `"union"`: union bound across all sumcheck and tree rounds.
    Union,
    /// `"forests"`: product-forest root claim compression.
    Forests,
    /// `"logup"`: LogUp rational-function identity.
    LogUp,
    /// `"openings"`: opening evaluation checks.
    Openings,
    /// `"queries"`: Ligero column proximity queries for one commitment.
    Queries,
    /// `"field"`: Ligero proximity generator field term `ϑ · n / q` for one commitment.
    Field,
    /// `"linking"`: weight-commitment linking check, `max (ν_T + 1)/q` over the opened matrices
    /// (RFC-0014 'Rows', RFC-0064 'Soundness').
    Linking,
    /// `"hash"`: BLAKE3 hash collision bound.
    Hash,
}

impl RowKind {
    /// The canonical JSON identifier in `tessera.soundness/1`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Rounds => "rounds",
            Self::Union => "union",
            Self::Forests => "forests",
            Self::LogUp => "logup",
            Self::Openings => "openings",
            Self::Queries => "queries",
            Self::Field => "field",
            Self::Linking => "linking",
            Self::Hash => "hash",
        }
    }
}

/// One computed row of the soundness report (RFC-0014 'Rows', `tessera.soundness/1`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SoundnessRow {
    /// Row kind.
    pub row: RowKind,
    /// Commitment name for `Queries` and `Field` rows.
    pub commitment: Option<CommitmentName>,
    /// Numerator `c` of a field-form bound `c / q`, saturated at `2^128 - 1`.
    pub numerator: Option<u128>,
    /// Codeword length `n` for a `Queries` row.
    pub codeword: Option<u64>,
    /// Message length `k` for a `Queries` row.
    pub message: Option<u64>,
    /// Mask coefficient count `P` for a `Queries` row.
    pub mask: Option<u64>,
    /// Opened column count `τ` for a `Queries` row.
    pub queries: Option<u32>,
    /// Row-index bit width `ϑ` for a `Field` row.
    pub theta: Option<u32>,
    /// Per-query soundness lower bound in `FixedBits` for a `Queries` row.
    pub per_query_bits: Option<u64>,
    /// Total soundness lower bound `λ` in `FixedBits` (`2^-8` bits), or `None` when `ε = 0`.
    pub bits: Option<u64>,
    /// Grinding bits `g` added to this row.
    pub grinding: u32,
    /// Which proof-of-work challenge covers this row, if any.
    pub covered_by: Option<CoveredBy>,
    /// Target in whole bits, or `None` when the parameter set has no targets.
    pub target: Option<u32>,
    /// Margin `bits - 256 · target` in `FixedBits`, or `None` without bits or target.
    pub margin: Option<i64>,
    /// Exact target decision `λ >= target`, or `None` when the parameter set has no targets.
    pub pass: Option<bool>,
}

impl SoundnessRow {
    /// Human-readable label of this row in `tessera-verify params` output.
    pub fn label(&self) -> String {
        match (self.row, self.commitment) {
            (RowKind::Rounds, _) => String::from("sumcheck and tree rounds"),
            (RowKind::Union, _) => String::from("all rounds, union bound"),
            (RowKind::Forests, _) => String::from("product-forest root claims"),
            (RowKind::LogUp, _) => String::from("LogUp identity"),
            (RowKind::Openings, _) => String::from("opening evaluation checks"),
            (RowKind::Queries, Some(c)) => format!("proximity queries, {}", c.as_str()),
            (RowKind::Queries, None) => String::from("proximity queries"),
            (RowKind::Field, Some(c)) => format!("proximity field term, {}", c.as_str()),
            (RowKind::Field, None) => String::from("proximity field term"),
            (RowKind::Linking, _) => String::from("weight linking"),
            (RowKind::Hash, _) => String::from("hash collisions"),
        }
    }
}

/// Invalid calculator input (`ERR-USAGE`) or a report reservation failure (`ERR-RES-MEMORY`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SoundnessError {
    /// The report's bounded row reservation failed (`ERR-RES-MEMORY`).
    Memory,
    /// `protocol` is neither 1 nor 2.
    Protocol(u8),
    /// `advice_queries` or `weight_queries` lies outside `1..=2^16`.
    Queries,
    /// Advice codeword length `2^{log2_k + log2_inv_rate}` exceeds `2^32`.
    AdviceCodeword,
    /// `advice_rows` is 0.
    AdviceRows,
    /// `matrices` are not in strictly increasing [`Matrix`] order.
    MatrixOrder,
    /// Weight matrix codeword length `width · 2^{log2_inv_rate}` lies outside `1..=2^32`.
    MatrixCodeword {
        /// The offending matrix.
        matrix: Matrix,
    },
    /// In `tessera-protocol/2`, a matrix has no committed row and no blinding row,
    /// `m_T + B_T = 0` ([`MatrixSize::blinding_rows`]).
    EmptyMatrix {
        /// The offending matrix.
        matrix: Matrix,
    },
    /// In `tessera-protocol/2`, a matrix has no claim, `e_T = 0`: it is not opened, and
    /// `matrices` lists only opened ones (RFC-0014 'Statement sizes'). `ChannelParams` included.
    UnopenedMatrix {
        /// The offending matrix.
        matrix: Matrix,
    },
    /// In `tessera-protocol/2`, a matrix other than `ChannelParams` has `e_T ≠ 1` claims, with
    /// `e_T ≥ 2` (RFC-0065 'The count', RFC-0014 'Statement sizes').
    MatrixClaims {
        /// The offending matrix.
        matrix: Matrix,
        /// Its claims `e_T`.
        claims: u64,
    },
    /// `groups.count` is not a power of two in `1..=2^14`.
    GroupCount(u64),
    /// `groups.count` is above 1 in `tessera-protocol/2`, which accepts no grouping.
    GroupedProtocol,
    /// `layer_vars` is not 0 in `tessera-protocol/2`, whose leaf stage's layer sumcheck, a
    /// stage, replaces stage 1's layer point (RFC-0014 'Statement sizes', RFC-0068, #537).
    LayerPointProtocol,
    /// `layer_vars` is above [`MAX_LAYER_VARS`] (RFC-0014 'Statement sizes', #537).
    LayerVars(u64),
    /// `groups.count` is 1 with a `level_rounds` entry, or a nonzero `merge_rounds` or
    /// `merge_claims`.
    UngroupedSizes,
}

impl SoundnessError {
    /// Stable error code (`04 §3`).
    pub const fn code(self) -> &'static str {
        match self {
            Self::Memory => "ERR-RES-MEMORY",
            _ => "ERR-USAGE",
        }
    }
}

impl fmt::Display for SoundnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Memory => f.write_str("memory for the soundness report could not be reserved"),
            Self::Protocol(p) => write!(f, "protocol must be 1 or 2, got {p}"),
            Self::Queries => f.write_str("column query count τ must be in 1..=65536"),
            Self::AdviceCodeword => f.write_str("advice codeword length must be at most 2^32"),
            Self::AdviceRows => f.write_str("advice_rows must be at least 1"),
            Self::MatrixOrder => {
                f.write_str("weight matrices must be in strictly increasing order")
            }
            Self::MatrixCodeword { matrix } => write!(
                f,
                "matrix {} codeword length must be in 1..=2^32",
                matrix.name()
            ),
            Self::EmptyMatrix { matrix } => write!(
                f,
                "matrix {} has no committed or blinding rows in tessera-protocol/2",
                matrix.name()
            ),
            Self::UnopenedMatrix { matrix } => write!(
                f,
                "matrix {} has no claim, so it is not opened, in tessera-protocol/2",
                matrix.name()
            ),
            Self::MatrixClaims { matrix, claims } => write!(
                f,
                "matrix {} must have exactly 1 claim in tessera-protocol/2, got {claims}",
                matrix.name()
            ),
            Self::GroupCount(count) => write!(
                f,
                "groups.count must be a power of two in 1..={MAX_GROUPS}, got {count}"
            ),
            Self::GroupedProtocol => {
                f.write_str("tessera-protocol/2 accepts no grouping: groups.count must be 1")
            }
            Self::LayerPointProtocol => f.write_str(
                "tessera-protocol/2 has no layer point at stage 1: layer_vars must be 0",
            ),
            Self::LayerVars(vars) => {
                write!(f, "layer_vars must be at most {MAX_LAYER_VARS}, got {vars}")
            }
            Self::UngroupedSizes => f.write_str(
                "groups.count 1 takes no level_rounds and zero merge_rounds and merge_claims",
            ),
        }
    }
}

impl core::error::Error for SoundnessError {}

/// Complete `tessera.soundness/1` report (RFC-0014 'Soundness calculator').
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SoundnessReport {
    /// Parameter set evaluated.
    pub params: SoundnessParams,
    /// Statement sizes evaluated.
    pub statement: StatementSizes,
    /// Soundness table rows in RFC-0014 order.
    pub rows: Vec<SoundnessRow>,
    /// Index in `rows` of the row with the smallest margin (or fewest bits when untargeted).
    pub minimum: Option<usize>,
    /// Whether every row meets its target (`None` when the parameter set has no targets).
    pub meets_targets: Option<bool>,
}

impl SoundnessReport {
    /// Validates `params` and `statement` and computes every row of RFC-0014's soundness table.
    ///
    /// # Errors
    ///
    /// Returns [`SoundnessError`] (`ERR-USAGE`) for invalid RFC-0014 inputs, or `Memory`
    /// (`ERR-RES-MEMORY`) if its bounded report row reservation fails.
    pub fn evaluate(
        params: SoundnessParams,
        statement: StatementSizes,
    ) -> Result<Self, SoundnessError> {
        validate(&params, &statement)?;
        let v2 = statement.protocol == 2;
        let targets = params.targets;
        let capacity = statement
            .matrices
            .len()
            .checked_mul(2)
            .and_then(|n| n.checked_add(9))
            .ok_or(SoundnessError::Memory)?;
        let mut rows = Vec::new();
        rows.try_reserve_exact(capacity)
            .map_err(|_| SoundnessError::Memory)?;

        // 1. Sumcheck and tree rounds (RFC-0014 'Rows' row 1).
        let max_round_deg = statement
            .rounds
            .iter()
            .filter(|r| r.count > 0)
            .map(|r| u128::from(r.degree))
            .max()
            .unwrap_or(0);
        let max_stage = statement
            .stages
            .iter()
            .copied()
            .map(u128::from)
            .max()
            .unwrap_or(0);
        let max_level = statement
            .tree_levels
            .iter()
            .copied()
            .map(u128::from)
            .max()
            .unwrap_or(0);
        let max_zc = statement
            .zero_checks
            .iter()
            .copied()
            .map(u128::from)
            .max()
            .unwrap_or(0);
        // Stage 1's layer point, λ coordinates: an item only when λ > 0, as a maximum with 0
        // leaves `D` unchanged (RFC-0014 'Rows', RFC-0037 'The layer point', #537).
        let layer_term = u128::from(statement.layer_vars);
        let line_term = u128::from(!statement.tree_levels.is_empty());
        let mask_term =
            u128::from(v2 && (!statement.stages.is_empty() || !statement.tree_levels.is_empty()));
        // RFC-0054 'Soundness': with G > 1, the groups' and the table batch's levels are levels of
        // `level_rounds`' degrees, and the merge is a stage of degree 2 that batches with error
        // (P + G)/q.
        let groups = &statement.groups;
        let grouped_term = if groups.grouped() {
            let level_deg = groups
                .level_rounds
                .iter()
                .filter(|r| r.count > 0)
                .map(|r| u128::from(r.degree))
                .max()
                .unwrap_or(0);
            let merge_deg = if groups.merge_rounds > 0 {
                u128::from(MERGE_DEGREE)
            } else {
                0
            };
            let batching = u128::from(groups.merge_claims).saturating_add(u128::from(groups.count));
            level_deg.max(merge_deg).max(batching)
        } else {
            0
        };
        let rounds_num = max_round_deg
            .max(max_stage)
            .max(max_level)
            .max(max_zc)
            .max(layer_term)
            .max(line_term)
            .max(mask_term)
            .max(grouped_term);
        rows.push(build_field_row(
            RowKind::Rounds,
            None,
            None,
            rounds_num,
            0,
            None,
            targets,
        ));

        // 2. All rounds, union bound (RFC-0014 'Rows' row 2).
        let mut union_num = 0u128;
        for r in &statement.rounds {
            let prod = u128::from(r.count).saturating_mul(u128::from(r.degree));
            union_num = union_num.saturating_add(prod);
        }
        for &m in &statement.stages {
            union_num = union_num.saturating_add(u128::from(m));
        }
        for &m in &statement.tree_levels {
            union_num = union_num.saturating_add(u128::from(m).saturating_add(1));
        }
        for &n_i in &statement.zero_checks {
            union_num = union_num.saturating_add(u128::from(n_i));
        }
        union_num = union_num.saturating_add(layer_term);
        if v2 {
            let masks = u128::try_from(statement.stages.len())
                .unwrap_or(u128::MAX)
                .saturating_add(u128::try_from(statement.tree_levels.len()).unwrap_or(u128::MAX));
            union_num = union_num.saturating_add(masks);
        }
        if groups.grouped() {
            // G more copies of the leaf stage's level terms, no group's levels and not the table
            // batch's costing more than the stage's (RFC-0054 'Soundness'), and the merge's
            // n_max rounds of degree 2 and its batching, P + G.
            let mut levels = 0u128;
            for r in &groups.level_rounds {
                levels =
                    levels.saturating_add(u128::from(r.count).saturating_mul(u128::from(r.degree)));
            }
            for &m in &statement.tree_levels {
                levels = levels.saturating_add(u128::from(m).saturating_add(1));
            }
            let merge = u128::from(MERGE_DEGREE)
                .saturating_mul(u128::from(groups.merge_rounds))
                .saturating_add(u128::from(groups.merge_claims))
                .saturating_add(u128::from(groups.count));
            union_num = union_num
                .saturating_add(u128::from(groups.count).saturating_mul(levels))
                .saturating_add(merge);
        }
        rows.push(build_field_row(
            RowKind::Union,
            None,
            None,
            union_num,
            0,
            None,
            targets,
        ));

        // 3. Product-forest root claims (RFC-0014 'Rows' row 3).
        let max_forest = statement
            .forests
            .iter()
            .copied()
            .map(u128::from)
            .max()
            .unwrap_or(0);
        rows.push(build_field_row(
            RowKind::Forests,
            None,
            None,
            max_forest,
            0,
            None,
            targets,
        ));

        // 4. LogUp identity (RFC-0014 'Rows' row 4).
        let worst_lookup = statement
            .lookups
            .iter()
            .map(|l| u128::from(l.lookups).saturating_add(u128::from(l.rows)))
            .max()
            .unwrap_or(0);
        let logup_factor = if v2 { 2u128 } else { 1u128 };
        rows.push(build_field_row(
            RowKind::LogUp,
            None,
            None,
            worst_lookup.saturating_mul(logup_factor),
            params.seed_pow,
            Some(CoveredBy::SeedPow),
            targets,
        ));

        // 5. Opening evaluation checks (RFC-0014 'Rows' row 5).
        let max_group = statement
            .claim_groups
            .iter()
            .copied()
            .map(u128::from)
            .max()
            .unwrap_or(0);
        let matrix_opening = u128::from(!statement.matrices.is_empty());
        rows.push(build_field_row(
            RowKind::Openings,
            None,
            None,
            max_group.max(matrix_opening),
            0,
            None,
            targets,
        ));

        // 6 & 7. Proximity queries and proximity field term, per commitment (RFC-0014, RFC-0049).
        let adv_k = 1u64
            .checked_shl(u32::from(params.advice.log2_k))
            .unwrap_or(0);
        let adv_n = adv_k
            .checked_shl(u32::from(params.advice.log2_inv_rate))
            .unwrap_or(0);
        let adv_theta = ceil_log2_u64(statement.advice_rows);
        push_commitment_rows(
            &mut rows,
            CommitmentName::Advice,
            adv_n,
            adv_k,
            u64::from(params.advice.mask_coeffs),
            params.advice_queries,
            adv_theta,
            params.query_pow,
            targets,
        );

        for m in &statement.matrices {
            // ϑ_T = ⌈log2(m_T + B_T)⌉ in tessera-protocol/2, `B_T` from the matrix's claims
            // (RFC-0065 '`ChannelParams`' blinding rows'), and ⌈log2 m_T⌉ in protocol/1.
            // `m_T + B_T < 2^64 + 2^95`: the sum is exact.
            let blinding = if v2 {
                m.blinding_rows(params.weights.blinding_rows)
            } else {
                0
            };
            let total_rows = u128::from(m.rows).saturating_add(blinding);
            if total_rows == 0 {
                // RFC-0049: in tessera-protocol/1 a matrix with no committed row has no proximity rows.
                continue;
            }
            let n = m
                .width
                .checked_shl(u32::from(params.weights.log2_inv_rate))
                .unwrap_or(0);
            let mask = if v2 {
                u64::from(params.weights.mask_coeffs)
            } else {
                0
            };
            let theta = ceil_log2_u128(total_rows);
            push_commitment_rows(
                &mut rows,
                CommitmentName::Weight(m.matrix),
                n,
                m.width,
                mask,
                params.weight_queries,
                theta,
                params.query_pow,
                targets,
            );
        }

        // 8. Weight linking (RFC-0014 'Rows' row 8): `max (ν_T + 1)/q`. Per matrix, a false link
        // passes when two different weight polynomials agree at `ρ_T`, `ν_T/q`, or when the
        // combining challenge `θ_T`, drawn after the blinding claim, cancels their difference,
        // `1/q` (RFC-0014 'The link's combining challenge', RFC-0064 'Soundness'). A maximum
        // over no matrix is 0: the row then has no term.
        let max_link = statement
            .matrices
            .iter()
            .map(|m| u128::from(m.variables).saturating_add(1))
            .max()
            .unwrap_or(0);
        rows.push(build_field_row(
            RowKind::Linking,
            None,
            None,
            max_link,
            0,
            None,
            targets,
        ));

        // 9. Hash collisions (RFC-0014 'Rows' row 9).
        let hash_bits = u64::from(HASH_BITS).saturating_mul(FIXED_BITS_UNIT);
        let hash_target = targets.map(|t| t.bits);
        let hash_margin = hash_target.map(|t| {
            let scaled = i64::from(t).saturating_mul(256);
            i64::try_from(hash_bits)
                .unwrap_or(i64::MAX)
                .saturating_sub(scaled)
        });
        let hash_pass = hash_target.map(|t| HASH_BITS >= t);
        rows.push(SoundnessRow {
            row: RowKind::Hash,
            commitment: None,
            numerator: None,
            codeword: None,
            message: None,
            mask: None,
            queries: None,
            theta: None,
            per_query_bits: None,
            bits: Some(hash_bits),
            grinding: 0,
            covered_by: None,
            target: hash_target,
            margin: hash_margin,
            pass: hash_pass,
        });

        let minimum = find_minimum(&rows, targets.is_some());
        let meets_targets = targets.map(|_| rows.iter().all(|r| r.pass == Some(true)));

        Ok(Self {
            params,
            statement,
            rows,
            minimum,
            meets_targets,
        })
    }

    /// Formats the report as a `tessera.soundness/1` JSON document (`spec/schemas/soundness-v1.schema.json`).
    pub fn to_json(&self) -> String {
        let mut out = String::from("{\n");
        let _ = writeln!(
            out,
            "  \"meets_targets\": {},",
            opt_bool_json(self.meets_targets)
        );
        let _ = writeln!(out, "  \"minimum\": {},", opt_usize_json(self.minimum));
        write_params_json(&mut out, &self.params);
        out.push_str("  \"rows\": [\n");
        let row_count = self.rows.len();
        for (idx, row) in self.rows.iter().enumerate() {
            write_row_json(&mut out, row, idx.saturating_add(1) == row_count);
        }
        out.push_str("  ],\n");
        out.push_str("  \"schema\": \"tessera.soundness/1\",\n");
        write_statement_json(&mut out, &self.statement);
        out.push_str("}\n");
        out
    }

    /// Formats the report as a human-readable soundness table for `tessera-verify params`.
    pub fn to_table(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "parameter set: {} ({})",
            self.params.id,
            self.params.mode.as_str()
        );
        let _ = writeln!(
            out,
            "protocol: tessera-protocol/{}",
            self.statement.protocol
        );
        let _ = writeln!(out, "leaf groups: G = {}", self.statement.groups.count);
        out.push('\n');
        let _ = writeln!(
            out,
            "{:<38} {:>9} {:>8} {:>6}",
            "row", "bits", "target", "pass"
        );
        for row in &self.rows {
            let bits_col = match row.bits {
                Some(b) => format_fixed_bits(b),
                None => String::from("-"),
            };
            let target_col = match row.target {
                Some(t) => t.to_string(),
                None => String::from("-"),
            };
            let pass_col = match row.pass {
                Some(true) => "yes",
                Some(false) => "NO",
                None => "-",
            };
            let _ = writeln!(
                out,
                "{:<38} {:>9} {:>8} {:>6}",
                row.label(),
                bits_col,
                target_col,
                pass_col
            );
        }
        out.push('\n');
        if let Some(min_idx) = self.minimum
            && let Some(min_row) = self.rows.get(min_idx)
            && let Some(bits) = min_row.bits
        {
            match min_row.target {
                Some(t) => {
                    let _ = writeln!(
                        out,
                        "minimum: {}: {} bits, target {t}",
                        min_row.label(),
                        format_fixed_bits(bits)
                    );
                }
                None => {
                    let _ = writeln!(
                        out,
                        "minimum: {}: {} bits",
                        min_row.label(),
                        format_fixed_bits(bits)
                    );
                }
            }
        }
        match self.meets_targets {
            Some(true) => {
                let _ = writeln!(out, "every row meets its target");
            }
            Some(false) => {
                let missed = self.rows.iter().filter(|r| r.pass == Some(false)).count();
                if missed == 1 {
                    let _ = writeln!(out, "1 row misses its target");
                } else {
                    let _ = writeln!(out, "{missed} rows miss their targets");
                }
            }
            None => {
                let _ = writeln!(
                    out,
                    "{} has no targets: it certifies nothing",
                    self.params.id
                );
            }
        }
        out
    }
}

fn format_fixed_bits(bits: u64) -> String {
    let whole = bits / FIXED_BITS_UNIT;
    let rem = bits % FIXED_BITS_UNIT;
    let hundredths = rem.saturating_mul(100) / FIXED_BITS_UNIT;
    format!("{whole}.{hundredths:02}")
}

fn validate(params: &SoundnessParams, statement: &StatementSizes) -> Result<(), SoundnessError> {
    if statement.protocol != 1 && statement.protocol != 2 {
        return Err(SoundnessError::Protocol(statement.protocol));
    }
    if !(1..=MAX_QUERIES).contains(&params.advice_queries)
        || !(1..=MAX_QUERIES).contains(&params.weight_queries)
    {
        return Err(SoundnessError::Queries);
    }
    let adv_log2_n = params
        .advice
        .log2_k
        .checked_add(params.advice.log2_inv_rate)
        .ok_or(SoundnessError::AdviceCodeword)?;
    if adv_log2_n > 32 {
        return Err(SoundnessError::AdviceCodeword);
    }
    if statement.advice_rows == 0 {
        return Err(SoundnessError::AdviceRows);
    }
    let mut prev: Option<Matrix> = None;
    for m in &statement.matrices {
        if let Some(p) = prev
            && p >= m.matrix
        {
            return Err(SoundnessError::MatrixOrder);
        }
        prev = Some(m.matrix);
        if m.width == 0 || params.weights.log2_inv_rate > 32 {
            return Err(SoundnessError::MatrixCodeword { matrix: m.matrix });
        }
        let n_t = m
            .width
            .checked_shl(u32::from(params.weights.log2_inv_rate))
            .ok_or(SoundnessError::MatrixCodeword { matrix: m.matrix })?;
        if n_t == 0 || n_t > MAX_CODEWORD {
            return Err(SoundnessError::MatrixCodeword { matrix: m.matrix });
        }
        // RFC-0014 'Statement sizes': in tessera-protocol/2 each matrix's blinding rows follow
        // from its claims. In this order: a listed matrix is opened, so it has a claim; a
        // single-claim matrix has exactly one (RFC-0065 'The count'); and every matrix needs a
        // committed or a blinding row. Protocol/1 reads no `e_T`.
        if statement.protocol == 2 {
            if m.claims == 0 {
                return Err(SoundnessError::UnopenedMatrix { matrix: m.matrix });
            }
            if m.matrix != Matrix::ChannelParams && m.claims != 1 {
                return Err(SoundnessError::MatrixClaims {
                    matrix: m.matrix,
                    claims: m.claims,
                });
            }
            if m.rows == 0 && m.blinding_rows(params.weights.blinding_rows) == 0 {
                return Err(SoundnessError::EmptyMatrix { matrix: m.matrix });
            }
        }
    }
    // RFC-0014 'Statement sizes', `groups`: G = 2^g with g ≤ 14 (RFC-0054 'Messages'), none in
    // tessera-protocol/2 (OQ-0054-1), and no grouped size without groups.
    let groups = &statement.groups;
    if !groups.count.is_power_of_two() || groups.count > MAX_GROUPS {
        return Err(SoundnessError::GroupCount(groups.count));
    }
    if groups.grouped() && statement.protocol == 2 {
        return Err(SoundnessError::GroupedProtocol);
    }
    // RFC-0014 'Statement sizes', `layer_vars`, in this order: tessera-protocol/2 has no layer
    // point at stage 1 (RFC-0068), and λ ≤ 32 (#537).
    if statement.protocol == 2 && statement.layer_vars != 0 {
        return Err(SoundnessError::LayerPointProtocol);
    }
    if statement.layer_vars > MAX_LAYER_VARS {
        return Err(SoundnessError::LayerVars(statement.layer_vars));
    }
    if !groups.grouped()
        && (!groups.level_rounds.is_empty() || groups.merge_rounds != 0 || groups.merge_claims != 0)
    {
        return Err(SoundnessError::UngroupedSizes);
    }
    Ok(())
}

/// `LOG2(a, b, fraction)` of RFC-0014 'Soundness calculator': a guaranteed lower bound
/// `L <= 2^fraction · log2(a / b)` for `1 <= b <= a`.
///
/// Normalizes `a / b` by extracting the 62 fractional bits of `(a - b·2^e) / (b·2^e)` via binary
/// long division (shift-and-subtract), avoiding multi-word division while remaining exact across
/// the full `u128` domain.
pub fn log2_lower(a: u128, b: u128, fraction: u32) -> u128 {
    if b == 0 || a < b {
        return 0;
    }
    let a_bits = 128u32.saturating_sub(a.leading_zeros());
    let b_bits = 128u32.saturating_sub(b.leading_zeros());
    let diff = a_bits.saturating_sub(b_bits);
    let shifted = b.checked_shl(diff).unwrap_or(u128::MAX);
    let e = if shifted > a {
        diff.saturating_sub(1)
    } else {
        diff
    };
    let d = b.checked_shl(e).unwrap_or(b);

    // Since d <= a < 2d, ⌊(a · 2^62) / d⌋ = 2^62 + ⌊((a - d) · 2^62) / d⌋.
    let mut rem = a.saturating_sub(d);
    let mut frac = 0u64;
    for _ in 0..LOG2_PRECISION {
        let carry = (rem >> 127) != 0;
        let doubled = rem << 1;
        frac <<= 1;
        if carry || doubled >= d {
            rem = doubled.wrapping_sub(d);
            frac |= 1;
        } else {
            rem = doubled;
        }
    }

    let mut y = (1u128 << LOG2_PRECISION) | u128::from(frac);
    let mut result = u128::from(e);
    let two_w_plus_1 = 1u128 << (LOG2_PRECISION.saturating_add(1));
    for _ in 0..fraction {
        y = y.wrapping_mul(y) >> LOG2_PRECISION;
        result = result.saturating_mul(2);
        if y >= two_w_plus_1 {
            y >>= 1;
            result = result.saturating_add(1);
        }
    }
    result
}

/// FixedBits lower bound on `g + max(0, log2(q / c))`, or `None` when `c == 0` (RFC-0014).
pub fn field_bits(c: u128, g: u32) -> Option<u64> {
    if c == 0 {
        return None;
    }
    let g_bits = u64::from(g).saturating_mul(FIXED_BITS_UNIT);
    if c >= EXTENSION_Q {
        return Some(g_bits);
    }
    let lb = u64::try_from(log2_lower(EXTENSION_Q, c, FIELD_FRACTION)).unwrap_or(u64::MAX);
    Some(lb.saturating_add(g_bits))
}

/// Exact target decision for a field-form bound `c / q`: `c == 0 || t <= g || c · 2^{t - g} <= q`
/// (RFC-0014 'Decisions').
pub const fn field_pass(c: u128, g: u32, t: u32) -> bool {
    if c == 0 || t <= g {
        return true;
    }
    let shift = t.saturating_sub(g);
    if shift >= 128 {
        return false;
    }
    c <= (EXTENSION_Q >> shift)
}

/// Returns `(per_query_bits, total_bits)` in `FixedBits` for a proximity queries row
/// `((n + k + P) / (2n))^τ` with `g` bits of grinding (RFC-0014 'Bits').
pub fn query_bits(n: u64, k: u64, mask: u64, tau: u32, g: u32) -> (u64, u64) {
    let a = u128::from(n).saturating_mul(2);
    let b = u128::from(n)
        .saturating_add(u128::from(k))
        .saturating_add(u128::from(mask));
    let g_bits = u64::from(g).saturating_mul(FIXED_BITS_UNIT);
    if b >= a || a == 0 {
        return (0, g_bits);
    }
    let lb = log2_lower(a, b, QUERY_FRACTION);
    let shift = QUERY_FRACTION.saturating_sub(FIELD_FRACTION);
    let per_query = u64::try_from(lb >> shift).unwrap_or(u64::MAX);
    let query_total =
        u64::try_from(u128::from(tau).saturating_mul(lb) >> shift).unwrap_or(u64::MAX);
    (per_query, query_total.saturating_add(g_bits))
}

/// Exact target decision for a proximity queries row: `t <= g` or `A^τ >= 2^{t - g} · B^τ`
/// with `A = 2n` and `B = n + k + P` (RFC-0014 'Decisions').
pub fn query_pass(n: u64, k: u64, mask: u64, tau: u32, g: u32, t: u32) -> bool {
    if t <= g {
        return true;
    }
    let a = u128::from(n).saturating_mul(2);
    let b = u128::from(n)
        .saturating_add(u128::from(k))
        .saturating_add(u128::from(mask));
    if b >= a || a == 0 {
        return false;
    }
    let shift = t.saturating_sub(g);
    // RFC-0014 'Decisions': LOG2(A, B, 48) decides every case outside a 2-ulp band.
    let lb = log2_lower(a, b, QUERY_FRACTION);
    let tau_u128 = u128::from(tau);
    let goal = u128::from(shift) << QUERY_FRACTION;
    if tau_u128.saturating_mul(lb) >= goal {
        return true;
    }
    if tau_u128.saturating_mul(lb.saturating_add(2)) <= goal {
        return false;
    }
    query_pass_exact_powers(a, b, tau, shift)
}

/// Exact comparison `a^tau >= 2^shift · b^tau` via multi-limb base-`2^64` integer exponentiation.
fn query_pass_exact_powers(a: u128, b: u128, tau: u32, shift: u32) -> bool {
    let lhs = BigUint::pow(a, tau);
    let rhs = BigUint::pow(b, tau).shl_bits(shift);
    lhs.ge_value(&rhs)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BigUint {
    /// Little-endian base-`2^64` limbs with no trailing zero limb.
    limbs: Vec<u64>,
}

impl BigUint {
    fn from_u128(x: u128) -> Self {
        let lo = u64::try_from(x & u128::from(u64::MAX)).unwrap_or(0);
        let hi = u64::try_from(x >> 64).unwrap_or(0);
        if hi != 0 {
            Self {
                limbs: vec![lo, hi],
            }
        } else if lo != 0 {
            Self { limbs: vec![lo] }
        } else {
            Self { limbs: Vec::new() }
        }
    }

    fn trim(&mut self) {
        while self.limbs.last().copied() == Some(0) {
            self.limbs.pop();
        }
    }

    fn mul(&self, other: &Self) -> Self {
        if self.limbs.is_empty() || other.limbs.is_empty() {
            return Self { limbs: Vec::new() };
        }
        let mut out = vec![0u64; self.limbs.len().saturating_add(other.limbs.len())];
        for (i, &x) in self.limbs.iter().enumerate() {
            let mut carry = 0u128;
            for (j, &y) in other.limbs.iter().enumerate() {
                let pos = i.saturating_add(j);
                if let Some(slot) = out.get_mut(pos) {
                    let prod = u128::from(x)
                        .wrapping_mul(u128::from(y))
                        .wrapping_add(u128::from(*slot))
                        .wrapping_add(carry);
                    *slot = u64::try_from(prod & u128::from(u64::MAX)).unwrap_or(0);
                    carry = prod >> 64;
                }
            }
            let end = i.saturating_add(other.limbs.len());
            if let Some(slot) = out.get_mut(end) {
                let sum = u128::from(*slot).wrapping_add(carry);
                *slot = u64::try_from(sum & u128::from(u64::MAX)).unwrap_or(0);
            }
        }
        let mut res = Self { limbs: out };
        res.trim();
        res
    }

    fn pow(base: u128, mut exp: u32) -> Self {
        let mut result = Self::from_u128(1);
        let mut cur = Self::from_u128(base);
        while exp > 0 {
            if (exp & 1) != 0 {
                result = result.mul(&cur);
            }
            exp >>= 1;
            if exp > 0 {
                cur = cur.mul(&cur);
            }
        }
        result
    }

    fn shl_bits(&self, bits: u32) -> Self {
        if self.limbs.is_empty() || bits == 0 {
            return self.clone();
        }
        let word_shift = usize::try_from(bits / 64).unwrap_or(0);
        let bit_shift = bits % 64;
        let mut out = Vec::with_capacity(
            self.limbs
                .len()
                .saturating_add(word_shift)
                .saturating_add(1),
        );
        out.resize(word_shift, 0);
        if bit_shift == 0 {
            out.extend_from_slice(&self.limbs);
        } else {
            let inv_shift = 64u32.saturating_sub(bit_shift);
            let mut carry = 0u64;
            for &limb in &self.limbs {
                out.push((limb << bit_shift) | carry);
                carry = limb >> inv_shift;
            }
            if carry != 0 {
                out.push(carry);
            }
        }
        let mut res = Self { limbs: out };
        res.trim();
        res
    }

    fn ge_value(&self, other: &Self) -> bool {
        if self.limbs.len() != other.limbs.len() {
            return self.limbs.len() > other.limbs.len();
        }
        for (&a, &b) in self.limbs.iter().rev().zip(other.limbs.iter().rev()) {
            if a != b {
                return a > b;
            }
        }
        true
    }
}

const fn ceil_log2_u64(x: u64) -> u32 {
    if x <= 1 {
        0
    } else {
        64u32.saturating_sub(x.saturating_sub(1).leading_zeros())
    }
}

const fn ceil_log2_u128(x: u128) -> u32 {
    if x <= 1 {
        0
    } else {
        128u32.saturating_sub(x.saturating_sub(1).leading_zeros())
    }
}

fn build_field_row(
    row: RowKind,
    commitment: Option<CommitmentName>,
    theta: Option<u32>,
    numerator: u128,
    grinding: u32,
    covered_by: Option<CoveredBy>,
    targets: Option<Targets>,
) -> SoundnessRow {
    let target = targets.map(|t| {
        if covered_by.is_some() {
            t.one_shot
        } else {
            t.bits
        }
    });
    let bits = field_bits(numerator, grinding);
    let margin = match (bits, target) {
        (Some(b), Some(t)) => {
            let scaled = i64::from(t).saturating_mul(256);
            Some(i64::try_from(b).unwrap_or(i64::MAX).saturating_sub(scaled))
        }
        _ => None,
    };
    let pass = target.map(|t| field_pass(numerator, grinding, t));
    SoundnessRow {
        row,
        commitment,
        numerator: Some(numerator),
        codeword: None,
        message: None,
        mask: None,
        queries: None,
        theta,
        per_query_bits: None,
        bits,
        grinding,
        covered_by,
        target,
        margin,
        pass,
    }
}

#[allow(clippy::too_many_arguments)]
fn push_commitment_rows(
    rows: &mut Vec<SoundnessRow>,
    commitment: CommitmentName,
    n: u64,
    k: u64,
    mask: u64,
    tau: u32,
    theta: u32,
    query_pow: u32,
    targets: Option<Targets>,
) {
    let q_target = targets.map(|t| t.one_shot);
    let (per_query, bits) = query_bits(n, k, mask, tau, query_pow);
    let q_margin = q_target.map(|t| {
        let scaled = i64::from(t).saturating_mul(256);
        i64::try_from(bits)
            .unwrap_or(i64::MAX)
            .saturating_sub(scaled)
    });
    let q_pass = q_target.map(|t| query_pass(n, k, mask, tau, query_pow, t));
    rows.push(SoundnessRow {
        row: RowKind::Queries,
        commitment: Some(commitment),
        numerator: None,
        codeword: Some(n),
        message: Some(k),
        mask: Some(mask),
        queries: Some(tau),
        theta: None,
        per_query_bits: Some(per_query),
        bits: Some(bits),
        grinding: query_pow,
        covered_by: Some(CoveredBy::QueryPow),
        target: q_target,
        margin: q_margin,
        pass: q_pass,
    });

    let field_num = u128::from(theta).saturating_mul(u128::from(n));
    rows.push(build_field_row(
        RowKind::Field,
        Some(commitment),
        Some(theta),
        field_num,
        0,
        None,
        targets,
    ));
}

fn find_minimum(rows: &[SoundnessRow], has_targets: bool) -> Option<usize> {
    let mut best: Option<(i128, usize)> = None;
    for (idx, row) in rows.iter().enumerate() {
        let maybe_key = if has_targets {
            row.margin.map(i128::from)
        } else {
            row.bits.map(i128::from)
        };
        let Some(key) = maybe_key else {
            continue;
        };
        if best.is_none_or(|(b_key, _)| key < b_key) {
            best = Some((key, idx));
        }
    }
    best.map(|(_, idx)| idx)
}

// ----- JSON formatting (`tessera.soundness/1`) --------------------------------------------------

fn opt_bool_json(v: Option<bool>) -> &'static str {
    match v {
        Some(true) => "true",
        Some(false) => "false",
        None => "null",
    }
}

fn opt_usize_json(v: Option<usize>) -> String {
    match v {
        Some(x) => x.to_string(),
        None => String::from("null"),
    }
}

fn opt_u64_json(v: Option<u64>) -> String {
    match v {
        Some(x) => x.to_string(),
        None => String::from("null"),
    }
}

fn opt_u32_json(v: Option<u32>) -> String {
    match v {
        Some(x) => x.to_string(),
        None => String::from("null"),
    }
}

fn opt_i64_json(v: Option<i64>) -> String {
    match v {
        Some(x) => x.to_string(),
        None => String::from("null"),
    }
}

fn escape_json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len().saturating_add(2));
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn write_params_json(out: &mut String, p: &SoundnessParams) {
    out.push_str("  \"params\": {\n");
    out.push_str("    \"advice\": {\n");
    let _ = writeln!(out, "      \"log2_inv_rate\": {},", p.advice.log2_inv_rate);
    let _ = writeln!(out, "      \"log2_k\": {},", p.advice.log2_k);
    let _ = writeln!(out, "      \"mask_coeffs\": {},", p.advice.mask_coeffs);
    let _ = writeln!(out, "      \"row_block\": {},", p.advice.row_block);
    let _ = writeln!(out, "      \"segments\": {}", p.advice.segments);
    out.push_str("    },\n");
    let _ = writeln!(out, "    \"advice_queries\": {},", p.advice_queries);
    let _ = writeln!(out, "    \"id\": {},", escape_json_str(&p.id));
    let _ = writeln!(out, "    \"mode\": \"{}\",", p.mode.as_str());
    let _ = writeln!(out, "    \"query_pow\": {},", p.query_pow);
    let _ = writeln!(out, "    \"release\": {},", p.release);
    let _ = writeln!(out, "    \"seed_pow\": {},", p.seed_pow);
    match p.targets {
        Some(t) => {
            out.push_str("    \"targets\": {\n");
            let _ = writeln!(out, "      \"bits\": {},", t.bits);
            let _ = writeln!(out, "      \"one_shot\": {}", t.one_shot);
            out.push_str("    },\n");
        }
        None => {
            out.push_str("    \"targets\": null,\n");
        }
    }
    let _ = writeln!(out, "    \"weight_queries\": {},", p.weight_queries);
    out.push_str("    \"weights\": {\n");
    let _ = writeln!(out, "      \"blinding_rows\": {},", p.weights.blinding_rows);
    let _ = writeln!(out, "      \"log2_inv_rate\": {},", p.weights.log2_inv_rate);
    let _ = writeln!(out, "      \"log2_k\": {},", p.weights.log2_k);
    let _ = writeln!(out, "      \"mask_coeffs\": {}", p.weights.mask_coeffs);
    out.push_str("    }\n");
    out.push_str("  },\n");
}

fn write_row_json(out: &mut String, row: &SoundnessRow, is_last: bool) {
    out.push_str("    {\n");
    let _ = writeln!(out, "      \"bits\": {},", opt_u64_json(row.bits));
    if let Some(codeword) = row.codeword {
        let _ = writeln!(out, "      \"codeword\": {codeword},");
    }
    if let Some(commitment) = row.commitment {
        let _ = writeln!(out, "      \"commitment\": \"{}\",", commitment.as_str());
    }
    match row.covered_by {
        Some(c) => {
            let _ = writeln!(out, "      \"covered_by\": \"{}\",", c.as_str());
        }
        None => {
            out.push_str("      \"covered_by\": null,\n");
        }
    }
    let _ = writeln!(out, "      \"grinding\": {},", row.grinding);
    let _ = writeln!(out, "      \"margin\": {},", opt_i64_json(row.margin));
    if let Some(mask) = row.mask {
        let _ = writeln!(out, "      \"mask\": {mask},");
    }
    if let Some(message) = row.message {
        let _ = writeln!(out, "      \"message\": {message},");
    }
    if let Some(numerator) = row.numerator {
        let _ = writeln!(out, "      \"numerator\": \"{numerator}\",");
    }
    let _ = writeln!(out, "      \"pass\": {},", opt_bool_json(row.pass));
    if let Some(pq) = row.per_query_bits {
        let _ = writeln!(out, "      \"per_query_bits\": {pq},");
    }
    if let Some(q) = row.queries {
        let _ = writeln!(out, "      \"queries\": {q},");
    }
    let _ = writeln!(out, "      \"row\": \"{}\",", row.row.as_str());
    if let Some(theta) = row.theta {
        let _ = writeln!(out, "      \"target\": {},", opt_u32_json(row.target));
        let _ = writeln!(out, "      \"theta\": {theta}");
    } else {
        let _ = writeln!(out, "      \"target\": {}", opt_u32_json(row.target));
    }
    if is_last {
        out.push_str("    }\n");
    } else {
        out.push_str("    },\n");
    }
}

fn write_u64_array_json(out: &mut String, name: &str, items: &[u64], trailing_comma: bool) {
    if items.is_empty() {
        let suffix = if trailing_comma { "," } else { "" };
        let _ = writeln!(out, "    \"{name}\": []{suffix}");
        return;
    }
    let _ = writeln!(out, "    \"{name}\": [");
    let count = items.len();
    for (idx, &v) in items.iter().enumerate() {
        let comma = if idx.saturating_add(1) == count {
            ""
        } else {
            ","
        };
        let _ = writeln!(out, "      {v}{comma}");
    }
    let suffix = if trailing_comma { "," } else { "" };
    let _ = writeln!(out, "    ]{suffix}");
}

fn write_groups_json(out: &mut String, g: &GroupSizes) {
    out.push_str("    \"groups\": {\n");
    let _ = writeln!(out, "      \"count\": {},", g.count);
    if g.level_rounds.is_empty() {
        out.push_str("      \"level_rounds\": [],\n");
    } else {
        out.push_str("      \"level_rounds\": [\n");
        let count = g.level_rounds.len();
        for (idx, r) in g.level_rounds.iter().enumerate() {
            out.push_str("        {\n");
            let _ = writeln!(out, "          \"count\": {},", r.count);
            let _ = writeln!(out, "          \"degree\": {}", r.degree);
            if idx.saturating_add(1) == count {
                out.push_str("        }\n");
            } else {
                out.push_str("        },\n");
            }
        }
        out.push_str("      ],\n");
    }
    let _ = writeln!(out, "      \"merge_claims\": {},", g.merge_claims);
    let _ = writeln!(out, "      \"merge_rounds\": {}", g.merge_rounds);
    out.push_str("    },\n");
}

fn write_statement_json(out: &mut String, s: &StatementSizes) {
    out.push_str("  \"statement\": {\n");
    let _ = writeln!(out, "    \"advice_rows\": {},", s.advice_rows);
    write_u64_array_json(out, "claim_groups", &s.claim_groups, true);
    write_u64_array_json(out, "forests", &s.forests, true);
    write_groups_json(out, &s.groups);
    let _ = writeln!(out, "    \"layer_vars\": {},", s.layer_vars);

    if s.lookups.is_empty() {
        out.push_str("    \"lookups\": [],\n");
    } else {
        out.push_str("    \"lookups\": [\n");
        let l_count = s.lookups.len();
        for (idx, l) in s.lookups.iter().enumerate() {
            out.push_str("      {\n");
            let _ = writeln!(out, "        \"lookups\": {},", l.lookups);
            let _ = writeln!(out, "        \"rows\": {}", l.rows);
            if idx.saturating_add(1) == l_count {
                out.push_str("      }\n");
            } else {
                out.push_str("      },\n");
            }
        }
        out.push_str("    ],\n");
    }

    if s.matrices.is_empty() {
        out.push_str("    \"matrices\": [],\n");
    } else {
        out.push_str("    \"matrices\": [\n");
        let m_count = s.matrices.len();
        for (idx, m) in s.matrices.iter().enumerate() {
            out.push_str("      {\n");
            let _ = writeln!(out, "        \"claims\": {},", m.claims);
            let _ = writeln!(out, "        \"matrix\": \"{}\",", m.matrix.name());
            let _ = writeln!(out, "        \"rows\": {},", m.rows);
            let _ = writeln!(out, "        \"variables\": {},", m.variables);
            let _ = writeln!(out, "        \"width\": {}", m.width);
            if idx.saturating_add(1) == m_count {
                out.push_str("      }\n");
            } else {
                out.push_str("      },\n");
            }
        }
        out.push_str("    ],\n");
    }

    let _ = writeln!(out, "    \"protocol\": {},", s.protocol);

    if s.rounds.is_empty() {
        out.push_str("    \"rounds\": [],\n");
    } else {
        out.push_str("    \"rounds\": [\n");
        let r_count = s.rounds.len();
        for (idx, r) in s.rounds.iter().enumerate() {
            out.push_str("      {\n");
            let _ = writeln!(out, "        \"count\": {},", r.count);
            let _ = writeln!(out, "        \"degree\": {}", r.degree);
            if idx.saturating_add(1) == r_count {
                out.push_str("      }\n");
            } else {
                out.push_str("      },\n");
            }
        }
        out.push_str("    ],\n");
    }

    write_u64_array_json(out, "stages", &s.stages, true);
    write_u64_array_json(out, "tree_levels", &s.tree_levels, true);
    write_u64_array_json(out, "zero_checks", &s.zero_checks, false);
    out.push_str("  }\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    use tv_testkit::Json;

    fn canonical_json(j: &Json) -> Json {
        match j {
            Json::Array(items) => Json::Array(items.iter().map(canonical_json).collect()),
            Json::Object(members) => {
                let mut sorted: Vec<(String, Json)> = members
                    .iter()
                    .map(|(k, v)| (k.clone(), canonical_json(v)))
                    .collect();
                sorted.sort_by(|a, b| a.0.cmp(&b.0));
                Json::Object(sorted)
            }
            other => other.clone(),
        }
    }

    fn parse_params(j: &Json) -> SoundnessParams {
        let adv = j.get("advice").unwrap();
        let wts = j.get("weights").unwrap();
        let targets_j = j.get("targets").unwrap();
        let targets = if matches!(targets_j, Json::Null) {
            None
        } else {
            Some(Targets {
                bits: u32::try_from(targets_j.get("bits").unwrap().as_u64().unwrap()).unwrap(),
                one_shot: u32::try_from(targets_j.get("one_shot").unwrap().as_u64().unwrap())
                    .unwrap(),
            })
        };
        let release = match j.get("release").unwrap() {
            Json::Bool(b) => *b,
            other => panic!("expected bool for release, got {other:?}"),
        };
        SoundnessParams {
            id: String::from(j.get("id").unwrap().as_str().unwrap()),
            mode: SessionMode::parse(j.get("mode").unwrap().as_str().unwrap()).unwrap(),
            release,
            advice: AdviceLayoutParams {
                log2_k: u8::try_from(adv.get("log2_k").unwrap().as_u64().unwrap()).unwrap(),
                log2_inv_rate: u8::try_from(adv.get("log2_inv_rate").unwrap().as_u64().unwrap())
                    .unwrap(),
                mask_coeffs: u32::try_from(adv.get("mask_coeffs").unwrap().as_u64().unwrap())
                    .unwrap(),
                row_block: u32::try_from(adv.get("row_block").unwrap().as_u64().unwrap()).unwrap(),
                segments: u32::try_from(adv.get("segments").unwrap().as_u64().unwrap()).unwrap(),
            },
            weights: WeightLayoutParams {
                log2_k: u8::try_from(wts.get("log2_k").unwrap().as_u64().unwrap()).unwrap(),
                log2_inv_rate: u8::try_from(wts.get("log2_inv_rate").unwrap().as_u64().unwrap())
                    .unwrap(),
                mask_coeffs: u32::try_from(wts.get("mask_coeffs").unwrap().as_u64().unwrap())
                    .unwrap(),
                blinding_rows: u32::try_from(wts.get("blinding_rows").unwrap().as_u64().unwrap())
                    .unwrap(),
            },
            advice_queries: u32::try_from(j.get("advice_queries").unwrap().as_u64().unwrap())
                .unwrap(),
            weight_queries: u32::try_from(j.get("weight_queries").unwrap().as_u64().unwrap())
                .unwrap(),
            seed_pow: u32::try_from(j.get("seed_pow").unwrap().as_u64().unwrap()).unwrap(),
            query_pow: u32::try_from(j.get("query_pow").unwrap().as_u64().unwrap()).unwrap(),
            targets,
        }
    }

    fn parse_u64_vec(j: &Json, key: &str) -> Vec<u64> {
        j.get(key)
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap())
            .collect()
    }

    fn parse_statement(j: &Json) -> StatementSizes {
        let rounds = j
            .get("rounds")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|r| RoundGroup {
                count: r.get("count").unwrap().as_u64().unwrap(),
                degree: r.get("degree").unwrap().as_u64().unwrap(),
            })
            .collect();
        let lookups = j
            .get("lookups")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|l| LookupSize {
                lookups: l.get("lookups").unwrap().as_u64().unwrap(),
                rows: l.get("rows").unwrap().as_u64().unwrap(),
            })
            .collect();
        let matrices = j
            .get("matrices")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|m| MatrixSize {
                matrix: Matrix::by_name(m.get("matrix").unwrap().as_str().unwrap()).unwrap(),
                width: m.get("width").unwrap().as_u64().unwrap(),
                rows: m.get("rows").unwrap().as_u64().unwrap(),
                variables: m.get("variables").unwrap().as_u64().unwrap(),
                claims: m.get("claims").unwrap().as_u64().unwrap(),
            })
            .collect();
        let groups = j.get("groups").unwrap();
        let groups = GroupSizes {
            count: groups.get("count").unwrap().as_u64().unwrap(),
            level_rounds: groups
                .get("level_rounds")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|r| RoundGroup {
                    count: r.get("count").unwrap().as_u64().unwrap(),
                    degree: r.get("degree").unwrap().as_u64().unwrap(),
                })
                .collect(),
            merge_rounds: groups.get("merge_rounds").unwrap().as_u64().unwrap(),
            merge_claims: groups.get("merge_claims").unwrap().as_u64().unwrap(),
        };
        StatementSizes {
            groups,
            protocol: u8::try_from(j.get("protocol").unwrap().as_u64().unwrap()).unwrap(),
            rounds,
            stages: parse_u64_vec(j, "stages"),
            tree_levels: parse_u64_vec(j, "tree_levels"),
            zero_checks: parse_u64_vec(j, "zero_checks"),
            layer_vars: j.get("layer_vars").unwrap().as_u64().unwrap(),
            forests: parse_u64_vec(j, "forests"),
            lookups,
            claim_groups: parse_u64_vec(j, "claim_groups"),
            advice_rows: j.get("advice_rows").unwrap().as_u64().unwrap(),
            matrices,
        }
    }

    #[test]
    fn reproduces_every_oracle_log2_and_report_vector() {
        let vectors = tv_testkit::read_vectors("soundness/soundness.json");
        let field = vectors.get("field").unwrap();
        assert_eq!(
            field.get("p").unwrap().as_str().unwrap(),
            GOLDILOCKS_P.to_string()
        );
        assert_eq!(
            field.get("q").unwrap().as_str().unwrap(),
            EXTENSION_Q.to_string()
        );

        for case in vectors.get("log2_cases").unwrap().as_array().unwrap() {
            let a: u128 = case.get("a").unwrap().as_str().unwrap().parse().unwrap();
            let b: u128 = case.get("b").unwrap().as_str().unwrap().parse().unwrap();
            let f = u32::try_from(case.get("fraction").unwrap().as_u64().unwrap()).unwrap();
            let expected: u128 = case
                .get("result")
                .unwrap()
                .as_str()
                .unwrap()
                .parse()
                .unwrap();
            assert_eq!(log2_lower(a, b, f), expected, "LOG2({a}, {b}, {f})");
        }

        let cases = vectors.get("cases").unwrap().as_array().unwrap();
        assert_eq!(cases.len(), 93);
        // Stage 1's layer point alone, 7/q in the rounds and union rows (#537).
        assert!(
            cases
                .iter()
                .any(|c| c.get("name").unwrap().as_str() == Some("edge/layer-point"))
        );
        for case in cases {
            let name = case.get("name").unwrap().as_str().unwrap();
            let expected_report = case.get("report").unwrap();
            let params = parse_params(expected_report.get("params").unwrap());
            let statement = parse_statement(expected_report.get("statement").unwrap());
            let computed = SoundnessReport::evaluate(params, statement).expect(name);
            let parsed_computed =
                Json::parse(&computed.to_json()).expect("report JSON parses cleanly");
            assert_eq!(
                canonical_json(&parsed_computed),
                canonical_json(expected_report),
                "case {name}"
            );
        }
    }

    #[test]
    fn default_405b_matches_oracle_and_meets_targets() {
        let base = StatementSizes::default_405b();
        assert_eq!(base.stages, [123, 98]);
        assert_eq!(base.lookups.len(), 27);
        assert_eq!(
            base.lookups[26],
            LookupSize {
                lookups: 2_147_483_648,
                rows: 536_870_912,
            }
        );
        assert_eq!(base.tree_levels, vec![55; 42]);
        assert_eq!(base.zero_checks, vec![42; 14]);
        assert_eq!(base.groups.merge_claims, 53_491);
        let vectors = tv_testkit::read_vectors("soundness/soundness.json");
        let cases = vectors.get("cases").unwrap().as_array().unwrap();
        for (params, case_name) in [
            (SoundnessParams::live_64(), "405b/live-64/1"),
            (SoundnessParams::ni_100(), "405b/ni-100/1"),
            (SoundnessParams::test_small(), "405b/test-small/1"),
        ] {
            let oracle_report = cases
                .iter()
                .find(|c| c.get("name").unwrap().as_str() == Some(case_name))
                .unwrap()
                .get("report")
                .unwrap();
            let report = SoundnessReport::evaluate(params, StatementSizes::default_405b()).unwrap();
            let as_json = Json::parse(&report.to_json()).unwrap();
            assert_eq!(
                canonical_json(&as_json),
                canonical_json(oracle_report),
                "{case_name}"
            );
        }
    }

    #[test]
    fn proximity_field_term_scales_with_row_index_bits() {
        let base = StatementSizes::default_405b();
        let n_a = 1u128 << (22 + 1);
        for theta in 1u32..=32 {
            let mut s = base.clone();
            s.advice_rows = 1u64 << theta;
            let report = SoundnessReport::evaluate(SoundnessParams::live_64(), s).unwrap();
            let field_row = report
                .rows
                .iter()
                .find(|r| r.row == RowKind::Field && r.commitment == Some(CommitmentName::Advice))
                .unwrap();
            assert_eq!(field_row.theta, Some(theta));
            assert_eq!(field_row.numerator, Some(u128::from(theta) * n_a));
        }
    }

    /// The weight-linking row of a statement of nothing else but `matrices`, each `(T, ν_T)`.
    fn linking_row(protocol: u8, matrices: &[(Matrix, u64)]) -> SoundnessRow {
        let s = StatementSizes {
            protocol,
            rounds: Vec::new(),
            stages: Vec::new(),
            tree_levels: Vec::new(),
            zero_checks: Vec::new(),
            layer_vars: 0,
            forests: Vec::new(),
            lookups: Vec::new(),
            claim_groups: Vec::new(),
            advice_rows: 1,
            matrices: matrices
                .iter()
                .map(|&(matrix, variables)| MatrixSize {
                    matrix,
                    width: 1,
                    rows: 1,
                    variables,
                    claims: 1,
                })
                .collect(),
            groups: GroupSizes::NONE,
        };
        let report = SoundnessReport::evaluate(SoundnessParams::ni_100(), s).unwrap();
        let mut linking = report.rows.iter().filter(|r| r.row == RowKind::Linking);
        let row = linking.next().unwrap().clone();
        assert!(linking.next().is_none());
        row
    }

    /// RFC-0014 'Rows', weight linking: `max (ν_T + 1)/q`, `ν_T/q` from the evaluation at `ρ_T`
    /// and `1/q` from the combining challenge `θ_T` (RFC-0014 'The link's combining challenge',
    /// RFC-0064 'Soundness'); the largest matrix's, not their sum; and no term without a matrix.
    #[test]
    fn weight_linking_counts_each_matrix_variables_and_its_combining_challenge() {
        for protocol in [1, 2] {
            // A maximum over no matrix is 0: the row has no term, and passes.
            let none = linking_row(protocol, &[]);
            assert_eq!(none.numerator, Some(0));
            assert_eq!(none.bits, None);
            assert_eq!(none.pass, Some(true));

            for nu in [0, 1, 30, 37, 64, u64::MAX] {
                let row = linking_row(protocol, &[(Matrix::Gate, nu)]);
                assert_eq!(row.numerator, Some(u128::from(nu) + 1), "ν = {nu}");
                assert_eq!(row.bits, field_bits(u128::from(nu) + 1, 0), "ν = {nu}");
            }
            // A matrix of no variable still has its combining challenge: 1/q, log2 q = 127.99…
            // bits.
            let row = linking_row(protocol, &[(Matrix::ChannelParams, 0)]);
            let full = u64::try_from(log2_lower(EXTENSION_Q, 1, FIELD_FRACTION)).unwrap();
            assert_eq!(row.bits, Some(full));
            assert_eq!(format_fixed_bits(full), "127.99");

            // The largest ν_T + 1, wherever it sits.
            for nus in [[30, 37, 0], [37, 30, 0], [0, 30, 37]] {
                let matrices: Vec<(Matrix, u64)> = [Matrix::Q, Matrix::Gate, Matrix::ChannelParams]
                    .into_iter()
                    .zip(nus)
                    .collect();
                assert_eq!(
                    linking_row(protocol, &matrices).numerator,
                    Some(38),
                    "{matrices:?}"
                );
            }
        }

        // RFC-0064 'Soundness': at 405B the largest ν_T is 37 (Gate, Up and Down), 122.75 bits.
        let base = StatementSizes::default_405b();
        let gate = base.matrices.iter().find(|m| m.matrix == Matrix::Gate);
        assert_eq!(gate.map(|m| m.variables), Some(37));
        assert_eq!(base.matrices.iter().map(|m| m.variables).max(), Some(37));
        for params in [SoundnessParams::live_64(), SoundnessParams::ni_100()] {
            let report = SoundnessReport::evaluate(params, base.clone()).unwrap();
            let row = report
                .rows
                .iter()
                .find(|r| r.row == RowKind::Linking)
                .unwrap();
            assert_eq!(row.numerator, Some(38));
            assert_eq!(row.bits, Some(31_424));
            assert_eq!(format_fixed_bits(31_424), "122.75");
            assert_eq!(row.margin, Some(31_424 - 100 * 256));
            assert_eq!(row.pass, Some(true));
        }
    }

    #[test]
    fn exact_query_powers_decide_tie_band_boundaries() {
        // Exercise `query_pass_exact_powers` directly on synthetic ratios inside and around a
        // target boundary: A = 4, B = 2 gives A/B = 2 (1 bit per query), so τ = 64 passes t = 64
        // and fails t = 65.
        assert!(query_pass_exact_powers(4, 2, 64, 64));
        assert!(!query_pass_exact_powers(4, 2, 64, 65));
        // A = 6, B = 4 gives (3/2)^4 = 81/16 > 5 > 4 = 2^2, and < 8 = 2^3.
        assert!(query_pass_exact_powers(6, 4, 4, 2));
        assert!(!query_pass_exact_powers(6, 4, 4, 3));
    }

    #[test]
    fn rejects_every_invalid_usage_condition_with_err_usage() {
        let base_p = SoundnessParams::live_64();
        let base_s = StatementSizes::default_405b();

        let mut s = base_s.clone();
        s.protocol = 0;
        let err = SoundnessReport::evaluate(base_p.clone(), s).unwrap_err();
        assert_eq!(err, SoundnessError::Protocol(0));
        assert_eq!(err.code(), "ERR-USAGE");

        let mut s = base_s.clone();
        s.protocol = 3;
        assert_eq!(
            SoundnessReport::evaluate(base_p.clone(), s),
            Err(SoundnessError::Protocol(3))
        );

        let mut p = base_p.clone();
        p.advice_queries = 0;
        assert_eq!(
            SoundnessReport::evaluate(p, base_s.clone()),
            Err(SoundnessError::Queries)
        );

        let mut p = base_p.clone();
        p.weight_queries = MAX_QUERIES + 1;
        assert_eq!(
            SoundnessReport::evaluate(p, base_s.clone()),
            Err(SoundnessError::Queries)
        );

        let mut p = base_p.clone();
        p.advice.log2_k = 31;
        p.advice.log2_inv_rate = 2;
        assert_eq!(
            SoundnessReport::evaluate(p, base_s.clone()),
            Err(SoundnessError::AdviceCodeword)
        );

        let mut s = base_s.clone();
        s.advice_rows = 0;
        assert_eq!(
            SoundnessReport::evaluate(base_p.clone(), s),
            Err(SoundnessError::AdviceRows)
        );

        let mut s = base_s.clone();
        s.matrices.swap(0, 1);
        assert_eq!(
            SoundnessReport::evaluate(base_p.clone(), s),
            Err(SoundnessError::MatrixOrder)
        );

        let mut s = base_s.clone();
        s.matrices[0].width = 0;
        assert_eq!(
            SoundnessReport::evaluate(base_p.clone(), s),
            Err(SoundnessError::MatrixCodeword {
                matrix: Matrix::Embed
            })
        );

        let mut s = base_s.clone();
        s.matrices[0].width = (1u64 << 30) + 1;
        assert_eq!(
            SoundnessReport::evaluate(base_p.clone(), s),
            Err(SoundnessError::MatrixCodeword {
                matrix: Matrix::Embed
            })
        );

        let mut p = base_p;
        p.weights.blinding_rows = 0;
        let mut s = base_s;
        s.protocol = 2;
        s.groups = GroupSizes::NONE;
        s.layer_vars = 0;
        s.matrices[9].rows = 0;
        assert_eq!(
            SoundnessReport::evaluate(p, s),
            Err(SoundnessError::EmptyMatrix {
                matrix: Matrix::ChannelParams
            })
        );
    }

    /// The default statement under `protocol`, with `ChannelParams`' committed rows and claims
    /// replaced.
    fn with_channel_params(protocol: u8, rows: u64, claims: u64) -> StatementSizes {
        let mut s = StatementSizes::default_405b();
        s.protocol = protocol;
        if protocol == 2 {
            s.groups = GroupSizes::NONE;
            s.layer_vars = 0;
        }
        let cp = s.matrices.last_mut().unwrap();
        assert_eq!(cp.matrix, Matrix::ChannelParams);
        cp.rows = rows;
        cp.claims = claims;
        s
    }

    /// The field term of `matrix` in `report`, and its queries row's codeword `n_T`.
    fn field_term(report: &SoundnessReport, matrix: Matrix) -> (u32, u128, u64) {
        let rows: Vec<&SoundnessRow> = report
            .rows
            .iter()
            .filter(|r| r.commitment == Some(CommitmentName::Weight(matrix)))
            .collect();
        assert_eq!(
            rows.iter().map(|r| r.row).collect::<Vec<_>>(),
            [RowKind::Queries, RowKind::Field],
            "{matrix:?}"
        );
        (
            rows[1].theta.unwrap(),
            rows[1].numerator.unwrap(),
            rows[0].codeword.unwrap(),
        )
    }

    /// RFC-0065 '`ChannelParams`' blinding rows': `B_T = B_W` for `Embed`..`LmHead`, and
    /// `B_CP = B_W + 2·(e_CP − 1)·⌊max(B_W − 4, 0)/6⌋`, 0 at `e_CP = 0`. At 405B, `e_CP = 7`
    /// gives 12,280 at `B_W = 4096`; the test set's `B_W = 32` gives 80.
    #[test]
    fn channel_params_blinding_rows_follow_its_claims() {
        let cp = |claims| MatrixSize {
            matrix: Matrix::ChannelParams,
            width: 1,
            rows: 0,
            variables: 0,
            claims,
        };
        for (params, b_cp) in [
            (SoundnessParams::live_64(), [0u128, 4_096, 12_280]),
            (SoundnessParams::ni_100(), [0, 4_096, 12_280]),
            (SoundnessParams::test_small(), [0, 32, 80]),
        ] {
            let b_w = params.weights.blinding_rows;
            for (claims, expected) in [0, 1, 7].into_iter().zip(b_cp) {
                assert_eq!(
                    cp(claims).blinding_rows(b_w),
                    expected,
                    "{} e = {claims}",
                    params.id
                );
            }
            // Every other matrix keeps the layout's B_W.
            for matrix in Matrix::ALL {
                if matrix != Matrix::ChannelParams {
                    let m = MatrixSize { matrix, ..cp(1) };
                    assert_eq!(m.blinding_rows(b_w), u128::from(b_w), "{matrix:?}");
                }
            }
        }

        // RFC-0065: `ChannelParams` never ends an epoch first. Epoch 0 reserves from row 0 and
        // later epochs from row 4, the link keeping the last four rows, so with
        // `N_0 = ⌊max(B_W − 4, 0)/6⌋` and `N_1 = ⌊(B_W − 8)/6⌋` single-claim proofs, `ChannelParams`'
        // `2·e + 4` rows a proof fit as many times: and its rows are no more than that needs.
        for b_w in [8u32, 9, 10, 13, 14, 32, 100, 4_096, 4_097, 65_536] {
            let n_0 = u128::from((b_w - 4) / 6);
            let n_1 = u128::from((b_w - 8) / 6);
            for claims in [1u64, 2, 3, 7, 16, 1_000] {
                let b_cp = cp(claims).blinding_rows(b_w);
                let per_proof = 2 * u128::from(claims) + 4;
                assert!(b_cp - 4 >= per_proof * n_0, "B_W = {b_w}, e = {claims}");
                assert!(b_cp - 8 >= per_proof * n_1, "B_W = {b_w}, e = {claims}");
                // B_CP − B_W rows serve exactly the further claims' masks of N_0 proofs.
                assert_eq!(b_cp - u128::from(b_w), (per_proof - 6) * n_0);
            }
        }

        // `⌊max(B_W − 4, 0)/6⌋`: below ten blinding rows epoch 0 gives no single-claim proof,
        // N_0 = 0, so B_CP = B_W; below four, the `max` keeps the floor a count, never below
        // B_W (RFC-0065 '`max(B_W − 4, 0)` keeps `N_0` a count').
        for b_w in 0u32..10 {
            for claims in [1u64, 2, 7, u64::MAX] {
                assert_eq!(
                    cp(claims).blinding_rows(b_w),
                    u128::from(b_w),
                    "B_W = {b_w}"
                );
            }
        }

        // Exact at the extremes, e_CP = 2^64 − 1 and B_W = 2^32 − 1: no step saturates.
        assert_eq!(
            cp(u64::MAX).blinding_rows(u32::MAX),
            26_409_387_443_265_632_286_914_467_163
        );
        // In a report too: m_CP + B_CP = 2^64 − 1 + that, of 95 bits.
        let mut s = with_channel_params(2, u64::MAX, u64::MAX);
        s.matrices.drain(..9);
        let mut p = SoundnessParams::ni_100();
        p.weights.blinding_rows = u32::MAX;
        let report = SoundnessReport::evaluate(p, s).unwrap();
        assert_eq!(field_term(&report, Matrix::ChannelParams).0, 95);
    }

    /// RFC-0014 'Rows', the field term: `ϑ_CP = ⌈log2(m_CP + B_CP)⌉` in `tessera-protocol/2`,
    /// with `B_CP` from `e_CP`: 14 at 405B (RFC-0065 'Cost at 405B'), where `B_W` alone would
    /// give 13, and its numerator `ϑ·n_CP`.
    #[test]
    fn channel_params_field_term_counts_its_blinding_rows_in_protocol_2() {
        // (parameter set, m_CP, [ϑ at e_CP = 1, 2, 7]): the rows are chosen so that the counts of
        // blinding rows cross a power of two, or reach one exactly: m_CP = 944 and 945 at
        // test-small pin B_CP = 80, as the oracle's `edge/channel-params-*` cases do.
        for (params, rows, thetas) in [
            (SoundnessParams::live_64(), 882u64, [13u32, 13, 14]),
            (SoundnessParams::ni_100(), 882, [13, 13, 14]),
            (SoundnessParams::test_small(), 944, [10, 10, 10]),
            (SoundnessParams::test_small(), 945, [10, 10, 11]),
            (SoundnessParams::test_small(), 985, [10, 11, 11]),
        ] {
            // No claim: not opened, refused before any row.
            assert_eq!(
                SoundnessReport::evaluate(params.clone(), with_channel_params(2, rows, 0)),
                Err(SoundnessError::UnopenedMatrix {
                    matrix: Matrix::ChannelParams
                })
            );
            for (claims, theta) in [1u64, 2, 7].into_iter().zip(thetas) {
                let s = with_channel_params(2, rows, claims);
                let report = SoundnessReport::evaluate(params.clone(), s).unwrap();
                let (t, numerator, n) = field_term(&report, Matrix::ChannelParams);
                assert_eq!(t, theta, "{} m = {rows}, e = {claims}", params.id);
                assert_eq!(numerator, u128::from(theta) * u128::from(n));
                // Every other matrix's ϑ is the one of its own m_T + B_W.
                for m in &report.statement.matrices[..9] {
                    let total = u128::from(m.rows) + u128::from(params.weights.blinding_rows);
                    assert_eq!(field_term(&report, m.matrix).0, ceil_log2_u128(total));
                }
            }
        }

        // The default statement under protocol/2: e_CP = 7, ϑ_CP = 14, 102.19 bits at live-64
        // (RFC-0065 'Cost at 405B'), and every target met.
        for params in [SoundnessParams::live_64(), SoundnessParams::ni_100()] {
            let mut s = StatementSizes::default_405b();
            assert_eq!(s.matrices[9].claims, DEFAULT_CHANNEL_PARAMS_CLAIMS);
            assert!(s.matrices[..9].iter().all(|m| m.claims == 1));
            s.protocol = 2;
            s.groups = GroupSizes::NONE;
            s.layer_vars = 0;
            let report = SoundnessReport::evaluate(params, s).unwrap();
            assert_eq!(field_term(&report, Matrix::ChannelParams).0, 14);
            let field = report
                .rows
                .iter()
                .find(|r| {
                    r.row == RowKind::Field
                        && r.commitment == Some(CommitmentName::Weight(Matrix::ChannelParams))
                })
                .unwrap();
            assert_eq!(format_fixed_bits(field.bits.unwrap()), "102.19");
            assert_eq!(report.meets_targets, Some(true));
        }
    }

    /// The test parameter set the verifier accepts with `test-params` lays out `B_W = 32`, so
    /// its `B_CP` is 80 at `e_CP = 7`, 32 at 1 and 0 at none, which protocol/2 refuses.
    #[cfg(feature = "test-params")]
    #[test]
    fn the_accepted_test_set_gives_b_cp_80_at_seven_claims() {
        let params = SoundnessParams::from_param_set(ParamSet::TestSmall);
        assert_eq!(params.weights.blinding_rows, 32);
        assert_eq!(ParamSet::TestSmall.layout().weights.blinding_rows, 32);
        for (claims, b_cp) in [(0u64, 0u128), (1, 32), (7, 80)] {
            let m = MatrixSize {
                claims,
                ..StatementSizes::default_405b().matrices[9]
            };
            assert_eq!(m.blinding_rows(params.weights.blinding_rows), b_cp);
            if claims == 0 {
                assert_eq!(
                    SoundnessReport::evaluate(params.clone(), with_channel_params(2, 945, 0)),
                    Err(SoundnessError::UnopenedMatrix {
                        matrix: Matrix::ChannelParams
                    })
                );
                continue;
            }
            let report =
                SoundnessReport::evaluate(params.clone(), with_channel_params(2, 945, claims))
                    .unwrap();
            assert_eq!(
                field_term(&report, Matrix::ChannelParams).0,
                ceil_log2_u128(945 + b_cp)
            );
        }
    }

    /// RFC-0014 'Statement sizes': in `tessera-protocol/2` the calculator refuses, with
    /// `ERR-USAGE` and in this order, a matrix with no claim, which is not opened, `ChannelParams`
    /// included; a matrix other than `ChannelParams` whose `e_T` is not 1; and a matrix with
    /// `m_T + B_T = 0`. Each matrix alone, and each refusal against the next.
    #[test]
    fn protocol_2_refuses_unopened_and_multi_claim_matrices() {
        let base = with_channel_params(2, 882, 7);
        for index in 0..10 {
            let matrix = base.matrices[index].matrix;
            // No claim: every matrix, ChannelParams included.
            let mut s = base.clone();
            s.matrices[index].claims = 0;
            let err = SoundnessReport::evaluate(SoundnessParams::live_64(), s).unwrap_err();
            assert_eq!(err, SoundnessError::UnopenedMatrix { matrix });
            assert_eq!(err.code(), "ERR-USAGE");
            assert!(err.to_string().contains(matrix.name()), "{err}");
            if matrix == Matrix::ChannelParams {
                continue;
            }
            // More than one claim on a single-claim matrix.
            for claims in [2u64, 7, u64::MAX] {
                let mut s = base.clone();
                s.matrices[index].claims = claims;
                let err = SoundnessReport::evaluate(SoundnessParams::live_64(), s).unwrap_err();
                assert_eq!(err, SoundnessError::MatrixClaims { matrix, claims });
                assert_eq!(err.code(), "ERR-USAGE");
                assert!(err.to_string().contains(matrix.name()), "{err}");
            }
        }
        // ChannelParams takes any count of one or more.
        for claims in [1u64, 2, 7, 16, u64::MAX] {
            let s = with_channel_params(2, 882, claims);
            assert!(SoundnessReport::evaluate(SoundnessParams::live_64(), s).is_ok());
        }

        for params in [SoundnessParams::live_64(), SoundnessParams::test_small()] {
            // No claim is judged before the rows: a ChannelParams of neither is not opened.
            assert_eq!(
                SoundnessReport::evaluate(params.clone(), with_channel_params(2, 0, 0)),
                Err(SoundnessError::UnopenedMatrix {
                    matrix: Matrix::ChannelParams
                })
            );
            // A claim and no data row: B_CP blinding rows, ϑ = ⌈log2 B_CP⌉.
            for claims in [1u64, 7] {
                let report =
                    SoundnessReport::evaluate(params.clone(), with_channel_params(2, 0, claims))
                        .unwrap();
                let cp = report.statement.matrices[9];
                let b_cp = cp.blinding_rows(params.weights.blinding_rows);
                assert!(b_cp > 0);
                assert_eq!(
                    field_term(&report, Matrix::ChannelParams).0,
                    ceil_log2_u128(b_cp)
                );
            }
        }

        // m_T + B_T = 0 needs B_W = 0: then B_CP = 0 at any count, and B_T = 0.
        let mut p = SoundnessParams::live_64();
        p.weights.blinding_rows = 0;
        for claims in [1u64, 7] {
            let err = SoundnessReport::evaluate(p.clone(), with_channel_params(2, 0, claims))
                .unwrap_err();
            assert_eq!(
                err,
                SoundnessError::EmptyMatrix {
                    matrix: Matrix::ChannelParams
                }
            );
            assert_eq!(err.code(), "ERR-USAGE");
        }
        // A single-claim matrix of no data row has B_W rows, and is refused only without them.
        let mut s = base.clone();
        s.matrices[1].rows = 0;
        assert!(SoundnessReport::evaluate(SoundnessParams::live_64(), s.clone()).is_ok());
        assert_eq!(
            SoundnessReport::evaluate(p.clone(), s.clone()),
            Err(SoundnessError::EmptyMatrix { matrix: Matrix::Q })
        );
        // The order RFC-0014 lists: no claim, then e_T ≠ 1, then the rows.
        s.matrices[1].claims = 2;
        assert_eq!(
            SoundnessReport::evaluate(p.clone(), s.clone()),
            Err(SoundnessError::MatrixClaims {
                matrix: Matrix::Q,
                claims: 2
            })
        );
        s.matrices[1].claims = 0;
        assert_eq!(
            SoundnessReport::evaluate(p, s),
            Err(SoundnessError::UnopenedMatrix { matrix: Matrix::Q })
        );
    }

    /// RFC-0014 'Statement sizes': protocol/1 reads no `e_T`. Any count, on any matrix, leaves
    /// every row of the report as it is, and is not refused; the statement echoes it.
    #[test]
    fn protocol_1_does_not_read_the_claims() {
        for params in [
            SoundnessParams::live_64(),
            SoundnessParams::ni_100(),
            SoundnessParams::test_small(),
        ] {
            let base = StatementSizes::default_405b();
            assert_eq!(base.protocol, 1);
            let expected = SoundnessReport::evaluate(params.clone(), base.clone()).unwrap();
            for claims in [0u64, 2, 7, u64::MAX] {
                for index in 0..base.matrices.len() {
                    let mut s = base.clone();
                    s.matrices[index].claims = claims;
                    let report = SoundnessReport::evaluate(params.clone(), s.clone()).unwrap();
                    assert_eq!(
                        report.rows, expected.rows,
                        "{} {index} e = {claims}",
                        params.id
                    );
                    assert_eq!(report.minimum, expected.minimum);
                    assert_eq!(report.meets_targets, expected.meets_targets);
                    assert_eq!(report.statement, s);
                }
                // Every matrix at once, ChannelParams' empty data rows included.
                let mut s = base.clone();
                for m in &mut s.matrices {
                    m.claims = claims;
                }
                let report = SoundnessReport::evaluate(params.clone(), s).unwrap();
                assert_eq!(report.rows, expected.rows);
                let mut s = with_channel_params(1, 0, claims);
                s.matrices[0].claims = claims;
                let report = SoundnessReport::evaluate(params.clone(), s).unwrap();
                assert!(
                    report.rows.iter().all(
                        |r| r.commitment != Some(CommitmentName::Weight(Matrix::ChannelParams))
                    )
                );
            }
        }
    }

    /// The report emits each matrix's `claims` (`soundness-v1.schema.json`), its keys sorted as
    /// every object's of the report: `claims` first.
    #[test]
    fn the_report_states_each_matrix_claims() {
        let report =
            SoundnessReport::evaluate(SoundnessParams::live_64(), StatementSizes::default_405b())
                .unwrap();
        let json = report.to_json();
        assert!(json.contains(
            "      {\n        \"claims\": 7,\n        \"matrix\": \"ChannelParams\",\n        \"rows\": 882,\n        \"variables\": 27,\n        \"width\": 1048576\n      }\n"
        ));
        let parsed = Json::parse(&json).unwrap();
        let matrices = parsed
            .get("statement")
            .unwrap()
            .get("matrices")
            .unwrap()
            .as_array()
            .unwrap();
        let claims: Vec<u64> = matrices
            .iter()
            .map(|m| m.get("claims").unwrap().as_u64().unwrap())
            .collect();
        assert_eq!(claims, [1, 1, 1, 1, 1, 1, 1, 1, 1, 7]);
    }

    fn evaluate_groups(
        protocol: u8,
        groups: GroupSizes,
    ) -> Result<SoundnessReport, SoundnessError> {
        let mut s = StatementSizes::default_405b();
        s.protocol = protocol;
        s.groups = groups;
        if protocol == 2 {
            s.layer_vars = 0;
        }
        SoundnessReport::evaluate(SoundnessParams::live_64(), s)
    }

    /// RFC-0014 'Statement sizes': each grouping refusal, alone, with its exact error, and the
    /// groupings at their edges that are accepted.
    #[test]
    fn refuses_every_grouping_rfc_0014_names_with_err_usage() {
        let grouped = |count| GroupSizes {
            count,
            ..StatementSizes::default_405b().groups
        };
        for count in [
            0,
            3,
            6,
            12_288,
            MAX_GROUPS + 1,
            MAX_GROUPS << 1,
            1 << 63,
            u64::MAX,
        ] {
            let err = evaluate_groups(1, grouped(count)).unwrap_err();
            assert_eq!(err, SoundnessError::GroupCount(count), "{count}");
            assert_eq!(err.code(), "ERR-USAGE");
        }
        for count in [2, 4, MAX_GROUPS >> 1, MAX_GROUPS] {
            assert!(evaluate_groups(1, grouped(count)).is_ok(), "{count}");
        }
        assert!(evaluate_groups(1, GroupSizes::NONE).is_ok());

        // tessera-protocol/2 accepts no grouping: two groups already refuse, and none passes.
        for count in [2, MAX_GROUPS] {
            let err = evaluate_groups(2, grouped(count)).unwrap_err();
            assert_eq!(err, SoundnessError::GroupedProtocol);
            assert_eq!(err.code(), "ERR-USAGE");
        }
        assert!(evaluate_groups(2, GroupSizes::NONE).is_ok());
        // The count is judged before the protocol.
        assert_eq!(
            evaluate_groups(2, grouped(3)).unwrap_err(),
            SoundnessError::GroupCount(3)
        );

        // G = 1 with any grouped size, a level entry even of no round among them.
        for groups in [
            GroupSizes {
                level_rounds: vec![RoundGroup {
                    count: 0,
                    degree: 0,
                }],
                ..GroupSizes::NONE
            },
            GroupSizes {
                level_rounds: vec![RoundGroup {
                    count: 861,
                    degree: 3,
                }],
                ..GroupSizes::NONE
            },
            GroupSizes {
                merge_rounds: 1,
                ..GroupSizes::NONE
            },
            GroupSizes {
                merge_claims: 1,
                ..GroupSizes::NONE
            },
        ] {
            for protocol in [1, 2] {
                let err = evaluate_groups(protocol, groups.clone()).unwrap_err();
                assert_eq!(err, SoundnessError::UngroupedSizes, "{groups:?}");
                assert_eq!(err.code(), "ERR-USAGE");
            }
        }
    }

    /// RFC-0014 'The 405B 16K-token, 16K-context statement', `groups`: G = 2^{min(14, 14 + 7)},
    /// 861 level rounds of degree 3, n_max = 28 + 14 and P = 53,489 + 1 + 1 = 53,491; and the
    /// figures of 'At the default statement' with and without them.
    #[test]
    fn the_default_statement_is_grouped_at_the_largest_accepted_g() {
        let base = StatementSizes::default_405b();
        assert_eq!(
            base.groups,
            GroupSizes {
                count: 16_384,
                level_rounds: vec![RoundGroup {
                    count: 861,
                    degree: 3
                }],
                merge_rounds: 42,
                merge_claims: 53_491,
            }
        );
        assert_eq!(base.stages, [123, 98]);
        assert_eq!(base.tree_levels, vec![55; 42]);
        assert_eq!(base.zero_checks, vec![42; 14]);
        let rows = |groups: GroupSizes| {
            let mut s = base.clone();
            s.groups = groups;
            let report = SoundnessReport::evaluate(SoundnessParams::ni_100(), s).unwrap();
            assert_eq!(report.meets_targets, Some(true));
            let [rounds, union] = [0, 1].map(|i| {
                let row = &report.rows[i];
                (row.numerator.unwrap(), format_fixed_bits(row.bits.unwrap()))
            });
            (rounds, union)
        };
        let (rounds, union) = rows(base.groups.clone());
        let (plain_rounds, plain_union) = rows(GroupSizes::NONE);
        // The rounds row's largest error is the merge's batching, P + G = 69,875; without groups
        // it is stage 1's m = 123 (#566).
        assert_eq!(rounds, (69_875, String::from("111.90")));
        assert_eq!(plain_rounds, (123, String::from("121.05")));
        // The ungrouped union counts [123, 98] in stages, 42 levels of 55 + 1, 14 zero-checks of
        // 42 variables (#566), and stage 1's layer point, λ = 7 (#537): 73,161 + 7 = 73,168.
        assert_eq!(base.layer_vars, 7);
        assert_eq!(plain_union, (73_168, String::from("111.83")));
        // G more copies of the 42 levels, 861·3 + 42·(55 + 1), and the merge, 2·42 + P + G.
        assert_eq!(
            union.0,
            73_168 + 16_384 * (861 * 3 + 42 * 56) + 2 * 42 + 53_491 + 16_384
        );
        assert_eq!(union, (80_998_167, String::from("101.72")));
    }

    /// RFC-0014 'Under `tessera-protocol/2`': RFC-0053's runtime greedy certificate does not
    /// apply, so `405b-protocol-2` carries the 405B model sizes under `tessera-protocol/2` with
    /// stages `[122, 96, 26, 36, 2_668]` (model reduction stages, LogUp identity, leaf layer
    /// sumcheck, and `N = 2,668` mask claims), 26 model lookups, 42 `tree_levels` of 53 instances,
    /// 13 model `zero_checks` of 42 variables, 20 opening claim groups (`[64; 16]` plus
    /// `[36, 16, 96, 1]`), and `D = 2,668` (`116.61` bits); `edge/protocol-2` uses the compiled
    /// 405B model schedule's exact round counts (`Σ count·d = 7,146`) and zero-checks
    /// (`Σ n_i = 362`) (#565, #566).
    #[test]
    fn protocol_2_405b_and_compiled_schedule_vectors_match_rfc_0014() {
        let vectors = tv_testkit::read_vectors("soundness/soundness.json");
        let cases = vectors.get("cases").unwrap().as_array().unwrap();
        let find_report = |name: &str| {
            cases
                .iter()
                .find(|c| c.get("name").unwrap().as_str() == Some(name))
                .unwrap_or_else(|| panic!("missing oracle case {name}"))
                .get("report")
                .unwrap()
        };

        let mut s_405b_v2 = StatementSizes::default_405b();
        s_405b_v2.protocol = 2;
        s_405b_v2.rounds = vec![
            RoundGroup {
                count: 9_958,
                degree: 7,
            },
            RoundGroup {
                count: 42,
                degree: 10,
            },
            RoundGroup {
                count: 2,
                degree: 4,
            },
            RoundGroup {
                count: 22,
                degree: 4,
            },
            RoundGroup {
                count: 32,
                degree: 2,
            },
        ];
        s_405b_v2.stages = vec![122, 96, 26, 36, 2_668];
        s_405b_v2.lookups.truncate(26);
        s_405b_v2.tree_levels = vec![53; 42];
        s_405b_v2.zero_checks = vec![42; 13];
        s_405b_v2.layer_vars = 0;
        s_405b_v2.claim_groups.extend_from_slice(&[36, 16, 96, 1]);
        s_405b_v2.groups = GroupSizes::NONE;
        assert_eq!(s_405b_v2.lookups.len(), 26);
        assert_eq!(s_405b_v2.claim_groups.len(), 20);

        for (params, case_name, expected_meets) in [
            (
                SoundnessParams::live_64(),
                "405b-protocol-2/live-64/1",
                Some(true),
            ),
            (
                SoundnessParams::ni_100(),
                "405b-protocol-2/ni-100/1",
                Some(true),
            ),
            (
                SoundnessParams::test_small(),
                "405b-protocol-2/test-small/1",
                None,
            ),
        ] {
            let oracle_report = find_report(case_name);
            let oracle_statement = parse_statement(oracle_report.get("statement").unwrap());
            assert_eq!(oracle_statement, s_405b_v2, "{case_name}");

            let report = SoundnessReport::evaluate(params, s_405b_v2.clone()).unwrap();
            assert_eq!(report.meets_targets, expected_meets, "{case_name}");
            assert_eq!(report.rows[0].row, RowKind::Rounds);
            assert_eq!(report.rows[0].numerator, Some(2_668));
            assert_eq!(format_fixed_bits(report.rows[0].bits.unwrap()), "116.61");
            assert_eq!(report.rows[1].row, RowKind::Union);
            assert_eq!(report.rows[1].numerator, Some(76_095));
            assert_eq!(format_fixed_bits(report.rows[1].bits.unwrap()), "111.78");
            assert_eq!(report.rows[4].row, RowKind::Openings);
            assert_eq!(report.rows[4].numerator, Some(96));
            assert_eq!(format_fixed_bits(report.rows[4].bits.unwrap()), "121.41");

            let as_json = Json::parse(&report.to_json()).unwrap();
            assert_eq!(
                canonical_json(&as_json),
                canonical_json(oracle_report),
                "{case_name}"
            );
        }

        let edge_report = find_report("edge/protocol-2");
        let edge_params = parse_params(edge_report.get("params").unwrap());
        let edge_statement = parse_statement(edge_report.get("statement").unwrap());
        assert_eq!(edge_statement.protocol, 2);
        assert_eq!(edge_statement.stages, [122, 96, 26, 36, 2_668]);
        assert_eq!(edge_statement.lookups, s_405b_v2.lookups);
        assert_eq!(edge_statement.tree_levels, vec![53; 42]);
        assert_eq!(edge_statement.zero_checks.len(), 13);
        assert_eq!(edge_statement.zero_checks.iter().sum::<u64>(), 362);
        assert_eq!(edge_statement.claim_groups.len(), 20);
        assert_eq!(&edge_statement.claim_groups[..16], &[64; 16]);
        assert_eq!(&edge_statement.claim_groups[16..], &[36, 16, 96, 1]);
        assert_eq!(edge_statement.layer_vars, 0);
        assert_eq!(edge_statement.groups, GroupSizes::NONE);
        assert_eq!(
            edge_statement
                .rounds
                .iter()
                .map(|r| r.count * r.degree)
                .sum::<u64>(),
            7_146
        );

        let report = SoundnessReport::evaluate(edge_params, edge_statement).unwrap();
        assert_eq!(report.meets_targets, Some(true));
        assert_eq!(report.rows[0].row, RowKind::Rounds);
        assert_eq!(report.rows[0].numerator, Some(2_668));
        assert_eq!(format_fixed_bits(report.rows[0].bits.unwrap()), "116.61");
        assert_eq!(report.rows[1].row, RowKind::Union);
        assert_eq!(report.rows[1].numerator, Some(12_771));
        assert_eq!(format_fixed_bits(report.rows[1].bits.unwrap()), "114.35");
        assert_eq!(report.rows[4].row, RowKind::Openings);
        assert_eq!(report.rows[4].numerator, Some(96));
        assert_eq!(format_fixed_bits(report.rows[4].bits.unwrap()), "121.41");
    }

    fn only_groups(tree_levels: Vec<u64>, groups: GroupSizes) -> [u128; 2] {
        let s = StatementSizes {
            protocol: 1,
            rounds: Vec::new(),
            stages: Vec::new(),
            tree_levels,
            zero_checks: Vec::new(),
            layer_vars: 0,
            forests: Vec::new(),
            lookups: Vec::new(),
            claim_groups: Vec::new(),
            advice_rows: 1,
            matrices: Vec::new(),
            groups,
        };
        let report = SoundnessReport::evaluate(SoundnessParams::ni_100(), s).unwrap();
        [0, 1].map(|i| report.rows[i].numerator.unwrap())
    }

    /// RFC-0014 'Rows' with `G > 1`, term by term on statements of nothing else: each degree of a
    /// nonzero count in `level_rounds`, the merge's 2 when `n_max > 0` and `P + G` in `D`; `G`
    /// copies of the levels' rounds and of every tree level's `m + 1`, and `2·n_max + P + G`, in
    /// `U`.
    #[test]
    fn grouped_rows_count_each_term_rfc_0014_names() {
        let groups = |count, level_rounds: &[(u64, u64)], merge_rounds, merge_claims| GroupSizes {
            count,
            level_rounds: level_rounds
                .iter()
                .map(|&(count, degree)| RoundGroup { count, degree })
                .collect(),
            merge_rounds,
            merge_claims,
        };
        // A level degree above P + G; a zero count's degree is not a round.
        assert_eq!(
            only_groups(vec![], groups(2, &[(1, 9), (0, 99)], 0, 0)),
            [9, 2 * 9 + 2]
        );
        // The merge's degree, 2, when it has a round; here P + G = 2 too.
        assert_eq!(
            only_groups(vec![], groups(2, &[(1, 1)], 1, 0)),
            [2, 2 + 2 + 2]
        );
        // P + G: the merge's batching, above every degree.
        assert_eq!(
            only_groups(vec![], groups(4, &[(5, 3)], 3, 7)),
            [11, 4 * 15 + 6 + 7 + 4]
        );
        // G copies of each tree level's m + 1, beyond the statement's own.
        assert_eq!(
            only_groups(vec![3, 5], groups(4, &[(2, 3)], 3, 7)),
            [11, 4 + 6 + 4 * (6 + 4 + 6) + 6 + 7 + 4]
        );
        // The same statement without groups counts none of it.
        assert_eq!(only_groups(vec![3, 5], GroupSizes::NONE), [5, 10]);
        // A grouped numerator saturates at 2^128 − 1, and gives no bit.
        let mut s = StatementSizes::default_405b();
        s.groups.level_rounds = vec![RoundGroup {
            count: u64::MAX,
            degree: u64::MAX,
        }];
        let report = SoundnessReport::evaluate(SoundnessParams::ni_100(), s).unwrap();
        assert_eq!(report.rows[0].numerator, Some(u128::from(u64::MAX)));
        assert_eq!(report.rows[1].numerator, Some(u128::MAX));
        assert_eq!(report.rows[1].bits, Some(0));
        assert_eq!(report.rows[1].pass, Some(false));
    }

    /// RFC-0014: `--lookups N` and `--table-rows T` replace the instances; the leaf stage's levels
    /// follow the tallest tree once `⌈log2 max(N, T)⌉` exceeds the default's 42.
    #[test]
    fn one_instance_past_the_default_height_lengthens_the_leaf_levels() {
        let base = StatementSizes::default_405b();
        for (lookups, rows) in [(DEFAULT_EXP_LOOKUPS, 1), (1, DEFAULT_EXP_LOOKUPS), (1, 1)] {
            let instance = LookupSize { lookups, rows };
            let s = StatementSizes::default_405b_with_instance(instance);
            assert_eq!(
                s,
                StatementSizes {
                    lookups: vec![instance],
                    ..base.clone()
                }
            );
        }
        for (lookups, rows, levels) in [
            (DEFAULT_EXP_LOOKUPS + 1, DEFAULT_EXP_ROWS, 43u64),
            ((1 << 50) - (1 << 22), DEFAULT_EXP_ROWS, 50),
            (1, 1 << 50, 50),
            (u64::MAX, u64::MAX, 64),
        ] {
            let instance = LookupSize { lookups, rows };
            let s = StatementSizes::default_405b_with_instance(instance);
            let n = usize::try_from(levels).unwrap();
            assert_eq!(s.lookups, vec![instance]);
            assert_eq!(s.tree_levels, vec![55; n]);
            assert_eq!(
                s.groups,
                GroupSizes {
                    level_rounds: vec![RoundGroup {
                        count: levels * (levels - 1) / 2,
                        degree: 3
                    }],
                    merge_rounds: levels,
                    ..base.groups.clone()
                }
            );
            assert_eq!(
                StatementSizes {
                    lookups: base.lookups.clone(),
                    tree_levels: base.tree_levels.clone(),
                    groups: base.groups.clone(),
                    ..s.clone()
                },
                base
            );
            assert!(SoundnessReport::evaluate(SoundnessParams::live_64(), s).is_ok());
        }
    }

    /// The default statement with stage 1's layer point of `layer_vars` coordinates, under
    /// `protocol` and without groups, which `tessera-protocol/2` refuses.
    fn with_layer_point(protocol: u8, layer_vars: u64) -> StatementSizes {
        let mut s = StatementSizes::default_405b();
        s.protocol = protocol;
        s.groups = GroupSizes::NONE;
        s.layer_vars = layer_vars;
        s
    }

    /// RFC-0014 'Statement sizes': a calculator refuses, with `ERR-USAGE`, a `layer_vars` other
    /// than 0 in `tessera-protocol/2` and one above 32 (#537). Each refusal alone beside its
    /// passing neighbour, then the order RFC-0014 lists: after the grouping's count and protocol,
    /// the protocol before the bound, and both before the ungrouped sizes.
    #[test]
    fn refuses_every_layer_point_rfc_0014_names_with_err_usage() {
        let evaluate = |s| SoundnessReport::evaluate(SoundnessParams::live_64(), s);
        // tessera-protocol/2 has no layer point: one coordinate already refuses, none passes.
        for vars in [1, 7, MAX_LAYER_VARS] {
            let err = evaluate(with_layer_point(2, vars)).unwrap_err();
            assert_eq!(err, SoundnessError::LayerPointProtocol, "λ = {vars}");
            assert_eq!(err.code(), "ERR-USAGE");
            assert!(err.to_string().contains("layer_vars"), "{err}");
        }
        assert!(evaluate(with_layer_point(2, 0)).is_ok());
        // λ ≤ 32 in tessera-protocol/1: 32 passes, 33 refuses.
        for vars in [0, 1, 7, MAX_LAYER_VARS] {
            assert!(evaluate(with_layer_point(1, vars)).is_ok(), "λ = {vars}");
        }
        for vars in [MAX_LAYER_VARS + 1, 64, u64::MAX] {
            let err = evaluate(with_layer_point(1, vars)).unwrap_err();
            assert_eq!(err, SoundnessError::LayerVars(vars), "λ = {vars}");
            assert_eq!(err.code(), "ERR-USAGE");
            assert!(err.to_string().contains(&vars.to_string()), "{err}");
        }
        // The default statement's λ = 7 at the largest grouping passes under protocol/1.
        assert!(evaluate(StatementSizes::default_405b()).is_ok());

        // In RFC-0014's order: the protocol before the bound.
        assert_eq!(
            evaluate(with_layer_point(2, MAX_LAYER_VARS + 1)),
            Err(SoundnessError::LayerPointProtocol)
        );
        // The grouping's count and protocol before either.
        let mut s = with_layer_point(2, MAX_LAYER_VARS + 1);
        s.groups = StatementSizes::default_405b().groups;
        assert_eq!(evaluate(s.clone()), Err(SoundnessError::GroupedProtocol));
        s.groups.count = 3;
        assert_eq!(evaluate(s), Err(SoundnessError::GroupCount(3)));
        // Either before a grouped size without groups.
        for protocol in [1, 2] {
            let mut s = with_layer_point(protocol, MAX_LAYER_VARS + 1);
            s.groups.merge_rounds = 1;
            let expected = if protocol == 2 {
                SoundnessError::LayerPointProtocol
            } else {
                SoundnessError::LayerVars(MAX_LAYER_VARS + 1)
            };
            assert_eq!(evaluate(s.clone()), Err(expected));
            s.layer_vars = 0;
            assert_eq!(evaluate(s), Err(SoundnessError::UngroupedSizes));
        }
    }

    /// The rounds and union rows of a statement whose only item is stage 1's layer point.
    fn only_layer_point(protocol: u8, layer_vars: u64, zero_checks: Vec<u64>) -> [SoundnessRow; 2] {
        let s = StatementSizes {
            protocol,
            rounds: Vec::new(),
            stages: Vec::new(),
            tree_levels: Vec::new(),
            zero_checks,
            layer_vars,
            forests: Vec::new(),
            lookups: Vec::new(),
            claim_groups: Vec::new(),
            advice_rows: 1,
            matrices: Vec::new(),
            groups: GroupSizes::NONE,
        };
        let report = SoundnessReport::evaluate(SoundnessParams::ni_100(), s).unwrap();
        assert_eq!(report.rows[0].row, RowKind::Rounds);
        assert_eq!(report.rows[1].row, RowKind::Union);
        [report.rows[0].clone(), report.rows[1].clone()]
    }

    /// RFC-0014 'Rows': `λ`, the layer point's, is an item of the rounds row's `D` and a term of
    /// the union row's `U` (RFC-0037 'The layer point': `λ/|Fp2|`, #537). Alone, both rows are
    /// `λ/q`; with no layer point, neither has a term; beside a larger item `D` keeps that item
    /// and `U` still adds `λ`.
    #[test]
    fn the_layer_point_is_an_item_of_the_rounds_row_and_a_term_of_the_union() {
        for vars in [1u64, 7, 31, MAX_LAYER_VARS] {
            for row in only_layer_point(1, vars, Vec::new()) {
                assert_eq!(
                    row.numerator,
                    Some(u128::from(vars)),
                    "{:?} λ = {vars}",
                    row.row
                );
                assert_eq!(row.bits, field_bits(u128::from(vars), 0));
                assert_eq!(row.pass, Some(true));
            }
        }
        // `edge/layer-point`: 7/q, 125.19 bits, in both rows.
        let [rounds, union] = only_layer_point(1, 7, Vec::new());
        assert_eq!(rounds.bits, Some(32_049));
        assert_eq!(union.bits, Some(32_049));
        assert_eq!(format_fixed_bits(32_049), "125.19");
        // No layer point, no term: a maximum over no item is 0, in either protocol.
        for protocol in [1, 2] {
            for row in only_layer_point(protocol, 0, Vec::new()) {
                assert_eq!(row.numerator, Some(0), "{:?}", row.row);
                assert_eq!(row.bits, None);
                assert_eq!(row.pass, Some(true));
            }
        }
        // Beside a zero-check of 42 variables: D is the largest, U the sum.
        let [rounds, union] = only_layer_point(1, 7, vec![42]);
        assert_eq!((rounds.numerator, union.numerator), (Some(42), Some(49)));
        let [rounds, union] = only_layer_point(1, 7, vec![3]);
        assert_eq!((rounds.numerator, union.numerator), (Some(7), Some(10)));
        // The default statement counts its λ = 7 in the union, beside every other term.
        let base = StatementSizes::default_405b();
        for params in [SoundnessParams::live_64(), SoundnessParams::ni_100()] {
            let with = SoundnessReport::evaluate(params.clone(), base.clone()).unwrap();
            let without = SoundnessReport::evaluate(
                params,
                StatementSizes {
                    layer_vars: 0,
                    ..base.clone()
                },
            )
            .unwrap();
            assert_eq!(
                with.rows[1].numerator,
                without.rows[1].numerator.map(|u| u + 7)
            );
            assert_eq!(with.rows[0].numerator, without.rows[0].numerator);
        }
    }

    /// The report states `layer_vars` in its statement (`soundness-v1.schema.json`), its keys
    /// sorted as every object's of the report: after `groups`, before `lookups`.
    #[test]
    fn the_report_states_the_layer_point() {
        for (s, vars) in [
            (StatementSizes::default_405b(), 7u64),
            (with_layer_point(2, 0), 0),
        ] {
            let report = SoundnessReport::evaluate(SoundnessParams::live_64(), s).unwrap();
            let json = report.to_json();
            assert!(
                json.contains(&format!(
                    "\n    }},\n    \"layer_vars\": {vars},\n    \"lookups\": ["
                )),
                "{json}"
            );
            let parsed = Json::parse(&json).unwrap();
            let statement = parsed.get("statement").unwrap();
            assert_eq!(statement.get("layer_vars").unwrap().as_u64(), Some(vars));
        }
    }
}
