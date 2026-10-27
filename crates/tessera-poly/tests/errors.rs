// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Every input-dependent failure is a typed error, never a panic (04 §1): each test violates one
//! condition and asserts the exact error, and that a refused binding leaves its table unchanged.

mod common;

use common::{SplitMix64, small};
use tessera_codec::SecretBuf;
use tessera_field::{F65537, Field, Goldilocks, GoldilocksExt2};
use tessera_poly::{
    MAX_DEGREE, PolyError, RoundPoly, bind_lowest, bind_lowest_in_place, causal_selector, eq,
    eq_table, evaluate, evaluate_base, less_than, less_than_points, num_vars,
};

type Fp = Goldilocks;
type Fp2 = GoldilocksExt2;

fn fp2s(n: usize) -> Vec<Fp2> {
    let mut rng = SplitMix64::new(n as u64);
    rng.elements(n)
}

#[test]
fn a_length_that_is_not_a_power_of_two_is_refused() {
    for len in [0, 3, 5, 6, 7, 12, 1000, usize::MAX] {
        assert_eq!(num_vars(len), Err(PolyError::TableLength { len }));
    }
    for len in [0, 3, 6, 12] {
        let error = PolyError::TableLength { len };
        let table = fp2s(len);
        let base = vec![Fp::ONE; len];
        // The point length is checked after the table's, so any point will do.
        assert_eq!(evaluate(&table, &[]), Err(error));
        assert_eq!(evaluate_base(&base, &fp2s(2)), Err(error));
        assert_eq!(bind_lowest(&base, Fp2::ONE).err(), Some(error));
        let mut refused: SecretBuf<Fp2> = table.iter().copied().collect();
        assert_eq!(bind_lowest_in_place(&mut refused, Fp2::ONE), Err(error));
        assert_eq!(
            refused[..],
            table[..],
            "a refused binding leaves the table unchanged"
        );
    }
}

#[test]
fn a_point_of_the_wrong_length_is_refused() {
    for (vars, coordinates) in [(0, 1), (3, 2), (3, 4), (12, 0), (1, 12)] {
        let error = PolyError::PointLength {
            expected: vars,
            found: coordinates,
        };
        let point = fp2s(coordinates);
        assert_eq!(evaluate(&fp2s(1 << vars), &point), Err(error));
        assert_eq!(evaluate_base(&vec![Fp::ONE; 1 << vars], &point), Err(error));
    }
}

#[test]
fn points_compared_coordinate_by_coordinate_must_have_equal_lengths() {
    for (a, b) in [(0, 1), (1, 0), (3, 4), (10, 9)] {
        let error = PolyError::PointLength {
            expected: a,
            found: b,
        };
        let (x, y) = (fp2s(a), fp2s(b));
        assert_eq!(eq(&x, &y), Err(error));
        assert_eq!(less_than_points(&x, &y), Err(error));
        // `expected` is the length of q, the first argument.
        assert_eq!(causal_selector(&x, &y), Err(error));
    }
}

#[test]
fn a_table_of_one_entry_has_no_variable_to_bind() {
    assert_eq!(
        bind_lowest(&[Fp::ONE], Fp2::ONE).err(),
        Some(PolyError::NoVariable)
    );
    let mut table = SecretBuf::filled(Fp2::ONE, 1);
    assert_eq!(
        bind_lowest_in_place(&mut table, Fp2::ONE),
        Err(PolyError::NoVariable)
    );
    assert_eq!(table[..], [Fp2::ONE]);
    // A table of one entry is still evaluated, at the empty point.
    assert_eq!(evaluate(&[small::<Fp2>(5)], &[]), Ok(small(5)));
    assert_eq!(evaluate_base(&[small::<Fp>(5)], &[]), Ok(small::<Fp2>(5)));
}

#[test]
fn a_less_than_bound_above_the_cube_is_refused() {
    for vars in [0, 1, 5, 20, 62, 63] {
        let r = fp2s(vars);
        let cube = 1u64 << vars;
        assert!(less_than(&r, cube).is_ok(), "a = 2^n is the whole cube");
        for bound in [cube + 1, cube.saturating_mul(2), u64::MAX] {
            assert_eq!(
                less_than(&r, bound),
                Err(PolyError::BoundTooLarge { bound, vars })
            );
        }
    }
    // From 64 coordinates on, every u64 lies in the cube.
    for vars in [64, 65, 90] {
        assert!(less_than(&fp2s(vars), u64::MAX).is_ok());
    }
}

#[test]
fn an_eq_table_too_large_to_index_or_allocate_is_refused() {
    // 2^59 entries of 16 bytes overflow isize; 2^64 entries overflow usize. Neither reaches the
    // allocator.
    for vars in [59, 63, 64, 65, 200] {
        let point = vec![Fp2::ZERO; vars];
        assert_eq!(eq_table(&point), Err(PolyError::TooLarge { vars }));
    }
    assert_eq!(
        eq_table(&vec![F65537::ONE; 64]),
        Err(PolyError::TooLarge { vars: 64 })
    );
}

#[test]
fn a_round_polynomial_degree_outside_1_to_10_is_refused() {
    // RFC-0013 'Degree': a reduction stage of tessera-protocol/2 reaches 10.
    assert_eq!(MAX_DEGREE, 10);
    for degree in [0, MAX_DEGREE + 1, 20] {
        assert_eq!(
            RoundPoly::new(&fp2s(degree)),
            Err(PolyError::Degree { degree })
        );
    }
    for degree in 1..=MAX_DEGREE {
        assert!(RoundPoly::new(&fp2s(degree)).is_ok());
    }
}

#[test]
fn every_error_says_what_was_wrong_in_numbers() {
    let cases = [
        (PolyError::TableLength { len: 12 }, "12"),
        (PolyError::NoVariable, "no variable"),
        (
            PolyError::PointLength {
                expected: 3,
                found: 5,
            },
            "3 coordinates here, not 5",
        ),
        (
            PolyError::BoundTooLarge { bound: 9, vars: 3 },
            "2^3, not a = 9",
        ),
        (PolyError::TooLarge { vars: 70 }, "2^70"),
        (PolyError::Degree { degree: 11 }, "1 to 10, not 11"),
    ];
    for (error, shown) in cases {
        let text = error.to_string();
        assert!(text.contains(shown), "{text:?} should contain {shown:?}");
        let _: &dyn std::error::Error = &error;
    }
}
