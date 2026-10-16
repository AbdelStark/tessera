// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Every input-dependent failure is a typed error, never a panic (04 §1): each test violates one
//! condition and asserts the exact error, and that a refused call leaves its values, its codeword
//! buffer and its mask source untouched.

mod common;

use common::{Fp, Replay, SplitMix64, fp};
use tessera_field::Field;
use tessera_pcs::{NttPlan, PcsError, RowEncoder};

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
        assert_eq!(RowEncoder::new(len, 0).map(|_| ()), Err(error));
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
            RowEncoder::new(1 << log_k, 0).map(|_| ()),
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
            RowEncoder::new(k, p).map(|_| ()),
            Err(PcsError::MaskTooLarge {
                mask_coeffs: p,
                message_len: k
            })
        );
    }
    // P = k is the largest mask, and the length is checked first.
    assert!(RowEncoder::new(8, 8).is_ok());
    assert_eq!(
        RowEncoder::new(6, 9).map(|_| ()),
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
    let encoder = RowEncoder::new(8, 3).expect("valid parameters");
    for found in [0, 7, 9, 16] {
        let error = PcsError::RowLength { expected: 8, found };
        refused_row(&encoder, &vec![Fp::ONE; found], 16, error);
        // The row is checked before the codeword buffer.
        refused_row(&encoder, &vec![Fp::ONE; found], 15, error);
    }
}

#[test]
fn a_codeword_buffer_of_the_wrong_length_is_refused() {
    let encoder = RowEncoder::new(8, 3).expect("valid parameters");
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
    let encoder = RowEncoder::new(8, 3).expect("valid parameters");
    for len in [1, 7, 9, 23] {
        let error = PcsError::PartialRow { len, row_len: 8 };
        refused_rows(&encoder, len, 2 * len, error);
        // The rows are checked before the codeword buffer.
        refused_rows(&encoder, len, 5, error);
    }
}

#[test]
fn a_batch_codeword_buffer_of_the_wrong_length_is_refused() {
    let encoder = RowEncoder::new(8, 3).expect("valid parameters");
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
    ];
    for (error, shown) in cases {
        let text = error.to_string();
        assert!(text.contains(shown), "{text:?} should contain {shown:?}");
        let _: &dyn std::error::Error = &error;
    }
}
