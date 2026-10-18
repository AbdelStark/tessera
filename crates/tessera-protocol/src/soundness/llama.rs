// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The 405B 16K-token, 16K-context statement, the calculator's default (RFC-0014 'Soundness
//! calculator'): Llama-3.1-405B in the `servable` profile, one batch of 16,384 tokens at 16,384
//! positions of context, under `tessera-protocol/1`.

use crate::WeightMatrix;

use super::{LookupSizes, MatrixSizes, Rounds, StatementSizes};

/// The batch's tokens, and its context: 16K.
const TOKENS: u64 = 1 << 14;

/// Each LogUp instance of the compiler's 405B report, `commitment-report-405b-servable-c2-b32`
/// (`docs/perf/v0.2/`, two shift classes, `B_max = 32`), in table order: its lookups per token,
/// times the batch's tokens, and its table's rows.
const TABLES: [(u64, u64); 15] = [
    (0, 1 << 23),         // Rope
    (0, 1 << 22),         // Exp: looked up by the scores, below
    (6_709_248, 1 << 30), // Silu
    (2_209_792, 1 << 21), // Range
    (16_384, 1 << 20),    // Requant
    (2_193_408, 1 << 24), // Requant: RoPE's q and k
    (6_838_272, 1 << 30), // Requant
    (2_064_384, 1 << 20), // Range
    (2_064_384, 1 << 20), // Requant
    (2_064_384, 1 << 30), // Requant
    (4_128_768, 1 << 23), // Requant
    (4_128_768, 1 << 26), // Range
    (4_128_768, 1 << 26), // Requant
    (6_709_248, 1 << 16), // Requant
    (2_064_384, 1 << 32), // Requant
];

/// The `Exp` table, which every position of the padded score tensor looks up (RFC-0030 'Cost'):
/// `2^7` layers, `2^7` heads, `2^14` queries and `2^14` keys, `2^42` lookups.
const EXP: usize = 1;
const EXP_LOOKUPS: u64 = 1 << 42;

/// Llama-3.1-405B's weight matrices at `k_W = 2^20` (RFC-0040 'The matrices'): `k_T =
/// max(in_pad, min(2^20, out_pad·in_pad)) = 2^20` for each, `m_T = L·⌈out/2^{a_T}⌉` committed
/// rows with `a_T = log2(k_T/in_pad)`, and `ν_T = log2 L_pad + log2 out_pad + log2 in_pad`, with
/// 126 layers padded to 128. `ChannelParams` holds `[16, 128, 2^16]`, in at most 882 committed
/// rows (RFC-0012 'Column answers').
const MATRICES: [(WeightMatrix, u64, u32); 10] = [
    (WeightMatrix::Embed, 2_048, 31), // 16,384 × 128,256, transposed: a = 3
    (WeightMatrix::Q, 32_256, 35),    // 16,384 × 16,384: a = 6, 256 rows a layer
    (WeightMatrix::K, 2_016, 31),     // 1,024 × 16,384: 16 rows a layer
    (WeightMatrix::V, 2_016, 31),
    (WeightMatrix::O, 32_256, 35),
    (WeightMatrix::Gate, 104_832, 37), // 53,248 × 16,384: 832 rows a layer
    (WeightMatrix::Up, 104_832, 37),
    (WeightMatrix::Down, 129_024, 37), // 16,384 × 53,248: a = 4, 1,024 rows a layer
    (WeightMatrix::LmHead, 2_004, 31), // 128,256 × 16,384
    (WeightMatrix::ChannelParams, 882, 27),
];

impl StatementSizes {
    /// The 405B 16K-token, 16K-context statement under `tessera-protocol/1`, the sizes
    /// `tessera params check` reports at by default (RFC-0014 'Soundness calculator').
    ///
    /// Each size comes from RFC-0014's text, from the compiler's 405B commitment report, or from
    /// the architecture by the rule its RFC states; where none gives a count, an upper bound:
    ///
    /// - **Rounds:** 10,000, all of degree 7, RFC-0014's "about 10^4 rounds" of "degree `d ≤ 7`".
    /// - **Stages:** `servable`'s two reduction stages (RFC-0010 'Stages'), each batching at
    ///   most 64 instances; the compiler's schedule batches 18 and 4 before the gadgets' own
    ///   relations join it (#118).
    /// - **Tree levels:** 42, the height of the tallest tree, the `Exp` instance's lookup tree of
    ///   `2^42` leaves. Each batches at most 31 instances, the 15 instances' two trees and the
    ///   `SoftmaxRow` forest.
    /// - **Zero-checks:** 8 of at most 42 variables, the score tensor's: RFC-0035's four padding
    ///   zero-checks, `OneHot`'s booleanity and the first checkpoint's equality, with room.
    /// - **Forests:** the `SoftmaxRow` forest, `2^7 · 2^7 · 2^14 = 2^28` trees: `s = 28`.
    /// - **LogUp:** the report's 15 instances, `N` its lookups per token times 16,384 and `T`
    ///   its rows, except `Exp`'s `N`, the `2^42` positions of the padded score tensor (RFC-0030
    ///   'Cost'). RFC-0014's "`N + T ≤ 2^38` per instance" holds for the other 14, not for it.
    /// - **Claim groups:** 16, RFC-0011's target for `G`, of at most 64 claims each; the
    ///   compiler's pool holds 38 claims before the gadgets' join.
    /// - **Advice rows:** `2^21`, the 21-bit row index RFC-0014 counts for the 405B advice,
    ///   whose `5.8·10^5` non-zero rows span 20 or 21 bits (RFC-0011).
    /// - **Matrices:** the ten of RFC-0040 at `k_W = 2^20`, the width of `live-64/1` and
    ///   `ni-100/1`; under another width they would differ.
    pub fn llama_405b_16k() -> Self {
        let lookups = TABLES
            .iter()
            .enumerate()
            .map(|(i, &(per_token, rows))| LookupSizes {
                lookups: if i == EXP {
                    EXP_LOOKUPS
                } else {
                    per_token * TOKENS
                },
                rows,
            })
            .collect();
        Self {
            protocol: 1,
            rounds: vec![Rounds {
                count: 10_000,
                degree: 7,
            }],
            stages: vec![64, 64],
            tree_levels: vec![2 * TABLES.len() as u64 + 1; 42],
            zero_checks: vec![42; 8],
            forests: vec![28],
            lookups,
            claim_groups: vec![64; 16],
            advice_rows: 1 << 21,
            matrices: MATRICES
                .iter()
                .map(|&(matrix, rows, variables)| MatrixSizes {
                    matrix,
                    width: 1 << 20,
                    rows,
                    variables,
                })
                .collect(),
        }
    }
}
