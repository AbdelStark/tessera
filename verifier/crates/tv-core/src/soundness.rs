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

use core::fmt::{self, Write as _};

use crate::claims::{AdviceLayoutParams, WeightLayoutParams};
use crate::params::ParamSet;
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
    /// Multilinear variables `ν_T`.
    pub variables: u64,
}

/// Statement sizes passed to the soundness calculator (RFC-0014 'Statement sizes').
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StatementSizes {
    /// Protocol version `n` of `tessera-protocol/n`: `1` or `2`.
    pub protocol: u8,
    /// Sumcheck and tree rounds by count and degree.
    pub rounds: Vec<RoundGroup>,
    /// Instances `m` batched in each reduction stage.
    pub stages: Vec<u64>,
    /// Instances `m` batched in each tree level.
    pub tree_levels: Vec<u64>,
    /// Variables `n_i` of each `ZeroCheck`.
    pub zero_checks: Vec<u64>,
    /// Logarithm `s` of the tree count `2^s` in each product forest.
    pub forests: Vec<u64>,
    /// Lookup and table row counts `(N, T)` of each LogUp instance.
    pub lookups: Vec<LookupSize>,
    /// Claims `T_g` in each advice opening group.
    pub claim_groups: Vec<u64>,
    /// Padded row-index domain size `2^{ν - log2 k}` of the advice commitment.
    pub advice_rows: u64,
    /// Opened weight matrices in strictly increasing [`Matrix`] order.
    pub matrices: Vec<MatrixSize>,
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
    pub fn default_405b() -> Self {
        const TOKENS: u64 = 16_384;
        const LAYERS: u64 = 126;
        const D_MODEL: u64 = 16_384;
        const HEADS: u64 = 128;
        const KV_HEADS: u64 = 8;
        const D_HEAD: u64 = 128;
        const D_FF: u64 = 53_248;
        const VOCAB: u64 = 128_256;
        const LOG2_KW: u32 = 20;

        let l_pad = LAYERS.next_power_of_two();
        let h_pad = HEADS.next_power_of_two();
        let padded_extra_layers = l_pad.saturating_sub(LAYERS);

        let mut lookups = Vec::with_capacity(REPORT_405B_TABLES.len());
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

        let softmax_forest = u64::from(ceil_log2_u64(
            l_pad.saturating_mul(HEADS).saturating_mul(TOKENS),
        ));
        let forests = vec![softmax_forest];
        let level_count = usize::try_from(ceil_log2_u64(tallest)).unwrap_or(0);
        let instances_per_level = u64::try_from(lookups.len())
            .unwrap_or(0)
            .saturating_mul(2)
            .saturating_add(u64::try_from(forests.len()).unwrap_or(0));

        Self {
            protocol: 1,
            rounds: vec![RoundGroup {
                count: 10_000,
                degree: 7,
            }],
            stages: vec![64, 64],
            tree_levels: vec![instances_per_level; level_count],
            zero_checks: vec![42; 8],
            forests,
            lookups,
            claim_groups: vec![64; 16],
            advice_rows: 2_328_576_u64.next_power_of_two(),
            matrices: weight_matrices_405b(
                LOG2_KW, LAYERS, D_MODEL, HEADS, KV_HEADS, D_HEAD, D_FF, VOCAB,
            ),
        }
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
    /// `"linking"`: weight-commitment linking check.
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
    /// In `tessera-protocol/2`, a matrix has `rows + blinding_rows == 0`.
    EmptyMatrix {
        /// The offending matrix.
        matrix: Matrix,
    },
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
        let line_term = u128::from(!statement.tree_levels.is_empty());
        let mask_term =
            u128::from(v2 && (!statement.stages.is_empty() || !statement.tree_levels.is_empty()));
        let rounds_num = max_round_deg
            .max(max_stage)
            .max(max_level)
            .max(max_zc)
            .max(line_term)
            .max(mask_term);
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
        if v2 {
            let masks = u128::try_from(statement.stages.len())
                .unwrap_or(u128::MAX)
                .saturating_add(u128::try_from(statement.tree_levels.len()).unwrap_or(u128::MAX));
            union_num = union_num.saturating_add(masks);
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
            let blinding = if v2 {
                u64::from(params.weights.blinding_rows)
            } else {
                0
            };
            let total_rows = u128::from(m.rows).saturating_add(u128::from(blinding));
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

        // 8. Weight linking (RFC-0014 'Rows' row 8).
        let max_vars = statement
            .matrices
            .iter()
            .map(|m| u128::from(m.variables))
            .max()
            .unwrap_or(0);
        rows.push(build_field_row(
            RowKind::Linking,
            None,
            None,
            max_vars,
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
        if statement.protocol == 2 && m.rows == 0 && params.weights.blinding_rows == 0 {
            return Err(SoundnessError::EmptyMatrix { matrix: m.matrix });
        }
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

fn write_statement_json(out: &mut String, s: &StatementSizes) {
    out.push_str("  \"statement\": {\n");
    let _ = writeln!(out, "    \"advice_rows\": {},", s.advice_rows);
    write_u64_array_json(out, "claim_groups", &s.claim_groups, true);
    write_u64_array_json(out, "forests", &s.forests, true);

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
            })
            .collect();
        StatementSizes {
            protocol: u8::try_from(j.get("protocol").unwrap().as_u64().unwrap()).unwrap(),
            rounds,
            stages: parse_u64_vec(j, "stages"),
            tree_levels: parse_u64_vec(j, "tree_levels"),
            zero_checks: parse_u64_vec(j, "zero_checks"),
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
        assert_eq!(cases.len(), 86);
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
        s.matrices[9].rows = 0;
        assert_eq!(
            SoundnessReport::evaluate(p, s),
            Err(SoundnessError::EmptyMatrix {
                matrix: Matrix::ChannelParams
            })
        );
    }
}
