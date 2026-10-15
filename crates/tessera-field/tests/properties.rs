// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Algebraic properties of every prime field (RFC-0002 'Testing Strategy'), run over Goldilocks
//! and the test field F65537 with the same generic code: ring axioms on 2^16 random cases,
//! edge-case pairs against 128-bit integer arithmetic, generators, roots of unity, batch
//! inversion and canonical decoding.

use proptest::prelude::*;
use tessera_field::{F65537, Goldilocks, PrimeField, ZeroElement, batch_inverse};

const GOLDILOCKS_P: u64 = 0xFFFF_FFFF_0000_0001;

/// The edge values of RFC-0002, reduced into the field.
fn edge_values<F: PrimeField>() -> Vec<F> {
    let p = F::MODULUS;
    [
        0,
        1,
        p - 1,
        (1 << 32) - 1,
        1 << 32,
        1 << 63,
        p.wrapping_sub(1 << 32),
    ]
    .into_iter()
    .map(F::from_u64_reduced)
    .collect()
}

/// Any element, with the edge values over-represented.
fn element<F: PrimeField>() -> impl Strategy<Value = F> {
    prop_oneof![
        1 => proptest::sample::select(edge_values::<F>()),
        4 => any::<u64>().prop_map(F::from_u64_reduced),
    ]
}

fn ring_axioms<F: PrimeField>(a: F, b: F, c: F) {
    assert_eq!((a + b) + c, a + (b + c));
    assert_eq!(a + b, b + a);
    assert_eq!(a + F::ZERO, a);
    assert_eq!(a + (-a), F::ZERO);
    assert_eq!((a - b) + b, a);
    assert_eq!((a * b) * c, a * (b * c));
    assert_eq!(a * b, b * a);
    assert_eq!(a * F::ONE, a);
    assert_eq!(a * (b + c), a * b + a * c);
    assert_eq!(a.square(), a * a);
    assert_eq!(a.double(), a + a);
    match a.inverse() {
        Some(inverse) => assert_eq!(a * inverse, F::ONE),
        None => assert_eq!(a, F::ZERO),
    }
    let mut assigned = a;
    assigned += b;
    assigned *= c;
    assigned -= b;
    assert_eq!(assigned, (a + b) * c - b);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1 << 16))]

    #[test]
    fn goldilocks_is_a_field(a in element::<Goldilocks>(), b in element::<Goldilocks>(), c in element::<Goldilocks>()) {
        ring_axioms(a, b, c);
    }

    #[test]
    fn f65537_is_a_field(a in element::<F65537>(), b in element::<F65537>(), c in element::<F65537>()) {
        ring_axioms(a, b, c);
    }

    #[test]
    fn goldilocks_matches_wide_integer_arithmetic(x in any::<u64>(), y in any::<u64>()) {
        let p = u128::from(GOLDILOCKS_P);
        let (a, b) = (Goldilocks::from_u64_reduced(x), Goldilocks::from_u64_reduced(y));
        let (x, y) = (u128::from(x) % p, u128::from(y) % p);
        prop_assert_eq!(u128::from((a + b).to_canonical_u64()), (x + y) % p);
        prop_assert_eq!(u128::from((a - b).to_canonical_u64()), (x + p - y) % p);
        prop_assert_eq!(u128::from((a * b).to_canonical_u64()), x * y % p);
    }

    #[test]
    fn signed_embedding_round_trips(x in any::<i64>()) {
        let a = Goldilocks::from_i64(x);
        prop_assert_eq!(a.to_centered_i128(), i128::from(x));
        prop_assert_eq!(F65537::from_i64(x).to_canonical_u64(), x.rem_euclid(65_537) as u64);
    }
}

/// Every pair of RFC-0002 edge values, in every binary operation, against 128-bit integers.
#[test]
fn goldilocks_edge_pairs_match_wide_integer_arithmetic() {
    let p = u128::from(GOLDILOCKS_P);
    for a in edge_values::<Goldilocks>() {
        for b in edge_values::<Goldilocks>() {
            let (x, y) = (
                u128::from(a.to_canonical_u64()),
                u128::from(b.to_canonical_u64()),
            );
            assert_eq!(
                u128::from((a + b).to_canonical_u64()),
                (x + y) % p,
                "{a:?} + {b:?}"
            );
            assert_eq!(
                u128::from((a - b).to_canonical_u64()),
                (x + p - y) % p,
                "{a:?} - {b:?}"
            );
            assert_eq!(
                u128::from((a * b).to_canonical_u64()),
                x * y % p,
                "{a:?} * {b:?}"
            );
        }
    }
}

/// The generator generates: `g^((P − 1)/q) ≠ 1` for every prime factor `q` of `P − 1`.
fn check_generator<F: PrimeField>(generator: F, prime_factors: &[u64]) {
    let mut cofactor = F::MODULUS - 1;
    for &q in prime_factors {
        assert_eq!(cofactor % q, 0, "{q} divides P - 1");
        while cofactor % q == 0 {
            cofactor /= q;
        }
        assert_ne!(
            generator.pow((F::MODULUS - 1) / q),
            F::ONE,
            "g^((P-1)/{q}) = 1"
        );
    }
    assert_eq!(cofactor, 1, "the prime factors of P - 1 are complete");
    assert_eq!(generator.pow(F::MODULUS - 1), F::ONE);
}

#[test]
fn generators_generate_the_multiplicative_group() {
    check_generator(
        Goldilocks::MULTIPLICATIVE_GENERATOR,
        &[2, 3, 5, 17, 257, 65_537],
    );
    check_generator(F65537::MULTIPLICATIVE_GENERATOR, &[2]);
}

fn check_roots<F: PrimeField>() {
    for k in 1..=F::TWO_ADICITY {
        let root = F::two_adic_generator(k);
        assert_eq!(root.pow(1 << k), F::ONE, "root^(2^{k}) = 1");
        assert_eq!(root.pow(1 << (k - 1)), -F::ONE, "root^(2^{}) = -1", k - 1);
    }
    assert_eq!(F::two_adic_generator(0), F::ONE);
}

#[test]
fn two_adic_generators_have_exact_order() {
    check_roots::<Goldilocks>();
    check_roots::<F65537>();
}

#[test]
#[should_panic(expected = "two-adicity is 32")]
fn goldilocks_has_no_root_beyond_the_two_adicity() {
    let _ = Goldilocks::two_adic_generator(33);
}

#[test]
fn canonical_decoding_rejects_every_value_at_or_above_the_modulus() {
    let p = GOLDILOCKS_P;
    for x in [p, p + 1, u64::MAX - 1, u64::MAX] {
        assert_eq!(Goldilocks::from_canonical_u64(x), None, "{x:#x}");
    }
    // 10,000 points spread over [P, 2^64), the endpoints above included in spirit.
    let span = u64::MAX - p;
    for i in 0..10_000u64 {
        let x = p + span / 10_000 * i;
        assert_eq!(Goldilocks::from_canonical_u64(x), None, "{x:#x}");
    }
    assert_eq!(
        Goldilocks::from_canonical_u64(p - 1).map(|a| a.to_canonical_u64()),
        Some(p - 1)
    );
    assert_eq!(F65537::from_canonical_u64(65_537), None);
    assert_eq!(
        F65537::from_canonical_u64(65_536).map(|a| a.to_canonical_u64()),
        Some(65_536)
    );
}

fn check_batch_inverse<F: PrimeField>() {
    for n in [1usize, 2, 3, 8, 33] {
        let values: Vec<F> = (1..=n as u64)
            .map(|i| F::from_u64_reduced(i * 7919 + 3))
            .collect();
        let mut inverted = values.clone();
        batch_inverse(&mut inverted).expect("no zero");
        for (x, inverse) in values.iter().zip(&inverted) {
            assert_eq!(*x * *inverse, F::ONE);
        }
        // A zero at each position is reported at that position, and the slice is untouched.
        for zero_at in 0..n {
            let mut with_zero = values.clone();
            with_zero[zero_at] = F::ZERO;
            if zero_at + 1 < n {
                with_zero[n - 1] = F::ZERO; // a later zero must not mask the first
            }
            let before = with_zero.clone();
            assert_eq!(
                batch_inverse(&mut with_zero),
                Err(ZeroElement { index: zero_at })
            );
            assert_eq!(with_zero, before);
        }
    }
    let mut empty: Vec<F> = Vec::new();
    assert_eq!(batch_inverse(&mut empty), Ok(()));
}

#[test]
fn batch_inverse_inverts_and_reports_the_first_zero() {
    check_batch_inverse::<Goldilocks>();
    check_batch_inverse::<F65537>();
}
