// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Every input-dependent failure is a typed error, never a panic (04 §1): each test violates one
//! condition and asserts the exact error, and that a refused call leaves its values, its codeword
//! buffer and its mask source untouched. A refused commitment draws no mask coefficient.

mod common;

use std::num::NonZeroUsize;

use common::{Fp, Replay, SplitMix64, fp};
use tessera_codec::Secret;
use tessera_field::{Field, GoldilocksExt2};
use tessera_pcs::{
    ClaimGroup, CommitmentParams, NttPlan, PcsError, ReferenceCommitment, RowClaim, RowEncoder,
    RowLayout, WeightMatrix,
};

/// Every transform of `plan` on `values`, each on a fresh copy.
fn every_transform(plan: &NttPlan, values: &[Fp], shift: Fp) -> [(Result<(), PcsError>, bool); 4] {
    let run = |f: &dyn Fn(&mut [Fp]) -> Result<(), PcsError>| {
        let mut copy = values.to_vec();
        let result = f(&mut copy);
        (result, copy == values)
    };
    [
        run(&|v| plan.ntt(v)),
        run(&|v| plan.intt(v)),
        run(&|v| plan.coset_ntt(v, shift)),
        run(&|v| plan.coset_intt(v, shift)),
    ]
}

#[test]
fn a_length_that_is_not_a_power_of_two_is_refused() {
    for len in [0, 3, 5, 6, 7, 12, 1000, (1 << 20) + 1, usize::MAX] {
        let error = PcsError::NotPowerOfTwo { len };
        assert_eq!(NttPlan::new(len).map(|_| ()), Err(error));
        assert_eq!(RowEncoder::new(len, 1, 0).map(|_| ()), Err(error));
    }
}

#[test]
fn a_transform_above_two_to_the_thirty_two_points_is_refused() {
    // Refused before any table is allocated.
    for log_len in 33..usize::BITS {
        assert_eq!(
            NttPlan::new(1 << log_len).map(|_| ()),
            Err(PcsError::TooLong { log_len })
        );
    }
    // A row of 2^32 values would need a codeword of 2^33.
    for log_k in 32..usize::BITS {
        assert_eq!(
            RowEncoder::new(1 << log_k, 1, 0).map(|_| ()),
            Err(PcsError::TooLong { log_len: log_k + 1 })
        );
    }
}

#[test]
fn a_slice_of_another_length_than_the_plan_is_refused_and_left_unchanged() {
    let mut rng = SplitMix64::new(0x7E55_E7A0_0047_0301);
    for (size, len) in [(1, 0), (1, 2), (8, 7), (8, 9), (8, 16), (1 << 10, 1 << 9)] {
        let plan = NttPlan::new(size).expect("a supported size");
        let values = rng.fps(len);
        let error = PcsError::TransformLength {
            expected: size,
            found: len,
        };
        for (result, unchanged) in every_transform(&plan, &values, fp(7)) {
            assert_eq!(result, Err(error));
            assert!(unchanged, "a refused transform leaves the values unchanged");
        }
    }
}

#[test]
fn a_zero_shift_is_refused_and_the_values_left_unchanged() {
    let mut rng = SplitMix64::new(0x7E55_E7A0_0047_0302);
    for size in [1, 2, 64] {
        let plan = NttPlan::new(size).expect("a supported size");
        let values = rng.fps(size);
        let [_, _, coset, coset_inverse] = every_transform(&plan, &values, Fp::ZERO);
        for (result, unchanged) in [coset, coset_inverse] {
            assert_eq!(result, Err(PcsError::ZeroShift));
            assert!(unchanged);
        }
    }
    // The length is checked first.
    let plan = NttPlan::new(4).expect("a supported size");
    let [_, _, coset, coset_inverse] = every_transform(&plan, &[Fp::ONE; 3], Fp::ZERO);
    for (result, _) in [coset, coset_inverse] {
        assert_eq!(
            result,
            Err(PcsError::TransformLength {
                expected: 4,
                found: 3
            })
        );
    }
}

#[test]
fn more_mask_coefficients_than_row_values_are_refused() {
    for (k, p) in [
        (1, 2),
        (8, 9),
        (8, 1000),
        (1 << 20, (1 << 20) + 1),
        (2, usize::MAX),
    ] {
        assert_eq!(
            RowEncoder::new(k, 1, p).map(|_| ()),
            Err(PcsError::MaskTooLarge {
                mask_coeffs: p,
                message_len: k
            })
        );
    }
    // P = k is the largest mask, and the length is checked first.
    assert!(RowEncoder::new(8, 1, 8).is_ok());
    assert_eq!(
        RowEncoder::new(6, 1, 9).map(|_| ()),
        Err(PcsError::NotPowerOfTwo { len: 6 })
    );
}

/// Asserts that `encode_row` refuses `row` and `codeword` with `error`, drawing nothing and
/// leaving the buffer as it was.
fn refused_row(encoder: &RowEncoder, row: &[Fp], codeword_len: usize, error: PcsError) {
    let mut rng = SplitMix64::new(codeword_len as u64);
    let before = rng.fps(codeword_len);
    let mut codeword = before.clone();
    let mut source = Replay::new(vec![Fp::ONE; 64]);
    assert_eq!(
        encoder.encode_row(row, &mut source, &mut codeword),
        Err(error)
    );
    assert_eq!(source.drawn, 0, "nothing drawn on error");
    assert_eq!(codeword, before, "the buffer is untouched on error");
}

#[test]
fn a_row_of_the_wrong_length_is_refused() {
    let encoder = RowEncoder::new(8, 1, 3).expect("valid parameters");
    for found in [0, 7, 9, 16] {
        let error = PcsError::RowLength { expected: 8, found };
        refused_row(&encoder, &vec![Fp::ONE; found], 16, error);
        // The row is checked before the codeword buffer.
        refused_row(&encoder, &vec![Fp::ONE; found], 15, error);
    }
}

#[test]
fn a_codeword_buffer_of_the_wrong_length_is_refused() {
    let encoder = RowEncoder::new(8, 1, 3).expect("valid parameters");
    for found in [0, 8, 15, 17, 32] {
        let error = PcsError::CodewordLength {
            expected: 16,
            found,
        };
        refused_row(&encoder, &[Fp::ONE; 8], found, error);
    }
}

/// Asserts that `encode_rows` refuses `rows` and `codewords` with `error`, drawing nothing and
/// leaving the buffer as it was.
fn refused_rows(encoder: &RowEncoder, rows: usize, codewords: usize, error: PcsError) {
    let mut rng = SplitMix64::new((rows * 31 + codewords) as u64);
    let before = rng.fps(codewords);
    let mut buffer = before.clone();
    let mut source = Replay::new(vec![Fp::ONE; 64]);
    assert_eq!(
        encoder.encode_rows(&vec![Fp::ONE; rows], &mut source, &mut buffer),
        Err(error)
    );
    assert_eq!(source.drawn, 0, "nothing drawn on error");
    assert_eq!(buffer, before, "the buffer is untouched on error");
}

#[test]
fn a_batch_that_is_not_a_whole_number_of_rows_is_refused() {
    let encoder = RowEncoder::new(8, 1, 3).expect("valid parameters");
    for len in [1, 7, 9, 23] {
        let error = PcsError::PartialRow { len, row_len: 8 };
        refused_rows(&encoder, len, 2 * len, error);
        // The rows are checked before the codeword buffer.
        refused_rows(&encoder, len, 5, error);
    }
}

#[test]
fn a_batch_codeword_buffer_of_the_wrong_length_is_refused() {
    let encoder = RowEncoder::new(8, 1, 3).expect("valid parameters");
    for (rows, found) in [(0, 1), (8, 0), (8, 15), (8, 32), (24, 47), (24, 16)] {
        let error = PcsError::CodewordLength {
            expected: 2 * rows,
            found,
        };
        refused_rows(&encoder, rows, found, error);
    }
    // No rows and no symbols is an empty batch.
    let mut empty: [Fp; 0] = [];
    let mut never = || -> Fp { unreachable!("an empty batch draws nothing") };
    assert_eq!(encoder.encode_rows(&[], &mut never, &mut empty), Ok(()));
}

#[test]
fn every_error_says_what_was_wrong_in_numbers() {
    let cases = [
        (PcsError::NotPowerOfTwo { len: 12 }, "not 12"),
        (PcsError::TooLong { log_len: 33 }, "2^33 points"),
        (
            PcsError::TransformLength {
                expected: 8,
                found: 7,
            },
            "8 values, not 7",
        ),
        (
            PcsError::RowLength {
                expected: 8,
                found: 9,
            },
            "8 values, not 9",
        ),
        (
            PcsError::PartialRow { len: 9, row_len: 8 },
            "9 values do not fill rows of 8",
        ),
        (
            PcsError::CodewordLength {
                expected: 16,
                found: 15,
            },
            "16 symbols, not 15",
        ),
        (
            PcsError::MaskTooLarge {
                mask_coeffs: 9,
                message_len: 8,
            },
            "at most 8 mask coefficients, not 9",
        ),
        (PcsError::ZeroShift, "non-zero"),
        (PcsError::NoMask, "at least 1 mask coefficient, not 0"),
        (
            PcsError::RowVars { row_vars: 65 },
            "at most 64 bits, not 65",
        ),
        (PcsError::NoSegments, "at least 1 segment, not 0"),
        (
            PcsError::RowOrder {
                segment: 2,
                position: 0,
            },
            "row 0 of segment 2 is not above the row before it",
        ),
        (
            PcsError::RowRange {
                segment: 1,
                position: 3,
                row_vars: 4,
            },
            "row 3 of segment 1 is not below 2^4",
        ),
        (
            PcsError::MatrixLength {
                rows: 3,
                row_len: 4,
                found: 11,
            },
            "3 non-zero rows of 4 values each are not 11 values",
        ),
        (
            PcsError::ColumnPoint {
                group: 1,
                expected: 4,
                found: 5,
            },
            "column point of group 1 has 4 coordinates, not 5",
        ),
        (
            PcsError::RowPoint {
                group: 1,
                claim: 2,
                expected: 3,
                found: 0,
            },
            "row point of claim 2 of group 1 has 3 coordinates, not 0",
        ),
        (
            PcsError::XiPoint {
                expected: 3,
                found: 4,
            },
            "proximity point has 3 coordinates, not 4",
        ),
        (PcsError::NoQueries, "at least 1 column query, not 0"),
        (
            PcsError::QueryRange {
                position: 1,
                query: 16,
                columns: 16,
            },
            "query 1 asks for column 16 of a commitment of 16 columns",
        ),
        (
            PcsError::NoRows {
                matrix: WeightMatrix::ChannelParams,
            },
            "the ChannelParams matrix has no committed row",
        ),
        (
            PcsError::FoldPoint {
                matrix: WeightMatrix::Q,
                expected: 6,
                found: 7,
            },
            "the Q matrix takes a point of 6 coordinates, not 7",
        ),
        (
            PcsError::LayerPoint {
                matrix: WeightMatrix::Down,
                expected: 1,
                found: 0,
            },
            "layer point has 1 coordinates, not 0",
        ),
        (
            PcsError::OtherMatrix {
                expected: WeightMatrix::Q,
                found: WeightMatrix::O,
            },
            "the fold of the Q matrix cannot read the rows of the O matrix",
        ),
        (
            PcsError::RowIndex {
                matrix: WeightMatrix::K,
                index: 32,
                rows: 32,
            },
            "the K matrix has 32 committed rows, so no row 32",
        ),
        (
            PcsError::Layout {
                rule: "g·min(Q, N_max) ≤ k",
            },
            "no advice layout: g·min(Q, N_max) ≤ k",
        ),
        (
            PcsError::LayoutData {
                rule: "every layer has a tile",
            },
            "breaks a rule: every layer has a tile",
        ),
        (
            PcsError::Tile {
                tile: 7,
                rule: "a tile holds a value",
            },
            "tile 7 breaks a rule: a tile holds a value",
        ),
        (
            PcsError::ClaimPoint {
                expected: 5,
                found: 4,
            },
            "a tensor of 5 variables has 5 coordinates, not 4",
        ),
        (
            PcsError::ElementAxes {
                expected: 2,
                found: 3,
            },
            "a tensor of 2 axes has 2 coordinates, not 3",
        ),
        (
            PcsError::ElementRange { axis: 1 },
            "coordinate on axis 1 lies past the axis's padded length",
        ),
    ];
    for (error, shown) in cases {
        let text = error.to_string();
        assert!(text.contains(shown), "{text:?} should contain {shown:?}");
        let _: &dyn std::error::Error = &error;
    }
}

fn params(log2_k: u32, mask_coeffs: usize) -> CommitmentParams {
    CommitmentParams {
        log2_k,
        log2_inv_rate: 1,
        mask_coeffs,
        row_block: NonZeroUsize::new(32).expect("non-zero"),
        commitment_id: [1; 32],
    }
}

fn layout(row_vars: u32, segments: Vec<Vec<u64>>) -> RowLayout {
    RowLayout::new(row_vars, segments).expect("a valid layout")
}

/// Asserts that committing `values` values refuses with `error` and draws nothing.
fn refused_commit(params: CommitmentParams, layout: RowLayout, values: usize, error: PcsError) {
    let mut source = Replay::new(vec![Fp::ONE; 64]);
    let result = ReferenceCommitment::commit(
        params,
        layout,
        Secret::new(vec![Fp::ONE; values]),
        &mut source,
    );
    assert_eq!(result.map(|_| ()), Err(error));
    assert_eq!(source.drawn, 0, "nothing drawn on error");
}

#[test]
fn a_commitment_whose_codewords_exceed_two_to_the_thirty_two_symbols_is_refused() {
    for log2_k in [32, 33, 63, 64, u32::MAX] {
        let error = PcsError::TooLong {
            log_len: log2_k.saturating_add(1),
        };
        // Before the mask count and the rows.
        refused_commit(params(log2_k, 0), layout(1, vec![vec![0]]), 3, error);
    }
}

#[test]
fn a_commitment_without_mask_coefficients_is_refused() {
    for log2_k in [0, 3, 10] {
        // Before the rows.
        refused_commit(
            params(log2_k, 0),
            layout(1, vec![vec![0]]),
            3,
            PcsError::NoMask,
        );
    }
    // More than k coefficients is the encoder's refusal.
    for (log2_k, mask_coeffs) in [(0, 2), (3, 9), (3, usize::MAX)] {
        let error = PcsError::MaskTooLarge {
            mask_coeffs,
            message_len: 1 << log2_k,
        };
        refused_commit(
            params(log2_k, mask_coeffs),
            layout(1, vec![vec![0]]),
            3,
            error,
        );
    }
}

#[test]
fn a_row_index_of_more_than_sixty_four_bits_is_refused() {
    for row_vars in [65, 100, u32::MAX] {
        // Before the segments.
        assert_eq!(
            RowLayout::new(row_vars, Vec::new()),
            Err(PcsError::RowVars { row_vars })
        );
    }
    let widest = layout(64, vec![vec![0, u64::MAX]]);
    assert_eq!((widest.row_vars(), widest.rows()), (64, 2));
}

#[test]
fn a_layout_without_segments_is_refused() {
    for row_vars in [0, 3, 64] {
        assert_eq!(
            RowLayout::new(row_vars, Vec::new()),
            Err(PcsError::NoSegments)
        );
    }
    // Empty segments are segments.
    assert_eq!(layout(3, vec![Vec::new(), Vec::new()]).rows(), 0);
}

#[test]
fn rows_out_of_layout_order_are_refused() {
    let cases = [
        (vec![vec![1, 0]], 0, 1),
        (vec![vec![2, 2]], 0, 1),
        (vec![vec![0, 3], vec![1]], 1, 0),
        (vec![vec![3], vec![], vec![3, 4]], 2, 0),
        (vec![vec![0, 1, 2, 5, 4]], 0, 4),
    ];
    for (segments, segment, position) in cases {
        assert_eq!(
            RowLayout::new(3, segments),
            Err(PcsError::RowOrder { segment, position })
        );
    }
}

#[test]
fn a_row_at_or_above_two_to_the_row_vars_is_refused() {
    let cases = [
        (0, vec![vec![1]], 0, 0),
        (3, vec![vec![0, 8]], 0, 1),
        (3, vec![vec![2], vec![], vec![7, u64::MAX]], 2, 1),
        (63, vec![vec![1 << 63]], 0, 0),
    ];
    for (row_vars, segments, segment, position) in cases {
        assert_eq!(
            RowLayout::new(row_vars, segments),
            Err(PcsError::RowRange {
                segment,
                position,
                row_vars
            })
        );
    }
}

#[test]
fn rows_that_are_not_n_rows_of_k_values_are_refused() {
    // N = 3 rows of k = 4 values.
    for found in [0, 4, 11, 13, 16] {
        let error = PcsError::MatrixLength {
            rows: 3,
            row_len: 4,
            found,
        };
        refused_commit(
            params(2, 1),
            layout(3, vec![vec![0, 2], vec![5]]),
            found,
            error,
        );
    }
    // No non-zero row takes no value.
    let error = PcsError::MatrixLength {
        rows: 0,
        row_len: 4,
        found: 4,
    };
    refused_commit(params(2, 1), layout(3, vec![vec![]]), 4, error);
}

/// A commitment to rows 0, 2 and 5 of 8, of `k = 16` values, with one mask coefficient.
fn committed() -> ReferenceCommitment {
    let mut source = Replay::new(vec![Fp::ONE; 3]);
    let values = SplitMix64::new(0x7E55_E7A0_0054_0401).fps(3 * 16);
    ReferenceCommitment::commit(
        params(4, 1),
        layout(3, vec![vec![0, 2], vec![5]]),
        Secret::new(values),
        &mut source,
    )
    .expect("valid inputs")
}

fn group(column_vars: usize, row_vars: &[usize]) -> ClaimGroup {
    ClaimGroup {
        column_point: vec![GoldilocksExt2::ONE; column_vars],
        claims: row_vars
            .iter()
            .map(|&vars| RowClaim {
                row_point: vec![GoldilocksExt2::ONE; vars],
                gamma: GoldilocksExt2::ONE,
            })
            .collect(),
    }
}

#[test]
fn a_column_point_of_the_wrong_length_is_refused() {
    let commitment = committed();
    let xi = [GoldilocksExt2::ONE; 3];
    for found in [0, 3, 5] {
        let groups = [group(4, &[3]), group(found, &[3, 3])];
        assert_eq!(
            commitment.open(&groups, &xi),
            Err(PcsError::ColumnPoint {
                group: 1,
                expected: 4,
                found
            })
        );
    }
    // Every column point is checked before any row point.
    let groups = [group(4, &[2]), group(5, &[3])];
    assert_eq!(
        commitment.open(&groups, &xi),
        Err(PcsError::ColumnPoint {
            group: 1,
            expected: 4,
            found: 5
        })
    );
}

#[test]
fn a_row_point_of_the_wrong_length_is_refused() {
    let commitment = committed();
    let xi = [GoldilocksExt2::ONE; 3];
    for found in [0, 2, 4] {
        let groups = [group(4, &[3]), group(4, &[3, 3, found])];
        let error = PcsError::RowPoint {
            group: 1,
            claim: 2,
            expected: 3,
            found,
        };
        assert_eq!(commitment.open(&groups, &xi), Err(error));
        assert_eq!(
            commitment.layout().row_coefficients(&groups, &xi),
            Err(error)
        );
        // Before the proximity point.
        assert_eq!(commitment.open(&groups, &xi[..1]), Err(error));
    }
}

#[test]
fn a_proximity_point_of_the_wrong_length_is_refused() {
    let commitment = committed();
    let groups = [group(4, &[3])];
    for found in [0, 2, 4] {
        let xi = vec![GoldilocksExt2::ONE; found];
        let error = PcsError::XiPoint { expected: 3, found };
        assert_eq!(commitment.open(&groups, &xi), Err(error));
        assert_eq!(commitment.open(&[], &xi), Err(error));
        assert_eq!(
            commitment.layout().row_coefficients(&groups, &xi),
            Err(error)
        );
    }
}

#[test]
fn an_answer_to_no_query_is_refused() {
    assert_eq!(committed().answer(&[]), Err(PcsError::NoQueries));
}

#[test]
fn a_query_that_is_not_a_column_is_refused() {
    let commitment = committed();
    // n = 32 columns.
    let cases: [(&[u64], usize); 4] = [
        (&[32], 0),
        (&[0, 31, 32, 7], 2),
        (&[5, u64::MAX], 1),
        (&[1 << 32, 40], 0),
    ];
    for (queries, position) in cases {
        assert_eq!(
            commitment.answer(queries),
            Err(PcsError::QueryRange {
                position,
                query: queries[position],
                columns: 32
            })
        );
    }
}
