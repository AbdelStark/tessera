// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The NTT against its definition: evaluation at the powers of a root of unity, on the subgroup
//! `L` and on the coset `7·L` that re-encodes combined rows (RFC-0011 'Encoding').

use tv_field::{Fp, Fp2, NttError, coset_intt, coset_ntt, intt, ntt, root_of_unity};
use tv_testkit::SplitMix64;

fn random_values(rng: &mut SplitMix64, n: usize) -> Vec<Fp2> {
    (0..n)
        .map(|_| Fp2::new(Fp::reduce(rng.next_u64()), Fp::reduce(rng.next_u64())))
        .collect()
}

/// `Σ_j coefficients[j]·x^j` by Horner's rule.
fn evaluate(coefficients: &[Fp2], x: Fp) -> Fp2 {
    coefficients
        .iter()
        .rev()
        .fold(Fp2::ZERO, |acc, &c| acc * x + c)
}

#[test]
fn transforms_evaluate_on_the_subgroup_and_the_coset() {
    let mut rng = SplitMix64::new(0x7466_6965_6c64_0010);
    for log_n in 0..=10 {
        let n = 1 << log_n;
        let coefficients = random_values(&mut rng, n);
        let mut on_subgroup = coefficients.clone();
        ntt(&mut on_subgroup).expect("a valid length");
        let mut on_coset = coefficients.clone();
        coset_ntt(&mut on_coset, Fp::SEVEN).expect("a valid length");
        // The same transforms on Fp values, one coefficient lane at a time.
        let lane = |pick: fn(&Fp2) -> Fp, coset: bool| {
            let mut values: Vec<Fp> = coefficients.iter().map(pick).collect();
            let done = if coset {
                coset_ntt(&mut values, Fp::SEVEN)
            } else {
                ntt(&mut values)
            };
            done.expect("a valid length");
            values
        };
        let lanes = [
            (lane(|c| c.c0, false), lane(|c| c.c1, false)),
            (lane(|c| c.c0, true), lane(|c| c.c1, true)),
        ];

        let omega = root_of_unity(log_n).expect("log_n ≤ TWO_ADICITY");
        let mut x = Fp::ONE;
        for i in 0..n {
            assert_eq!(
                on_subgroup[i],
                evaluate(&coefficients, x),
                "n = 2^{log_n}, ω^{i}"
            );
            let shifted = Fp::SEVEN * x;
            assert_eq!(
                on_coset[i],
                evaluate(&coefficients, shifted),
                "n = 2^{log_n}, 7·ω^{i}"
            );
            for (evaluations, (c0, c1)) in [&on_subgroup, &on_coset].into_iter().zip(&lanes) {
                assert_eq!(
                    Fp2::new(c0[i], c1[i]),
                    evaluations[i],
                    "Fp lanes, n = 2^{log_n}"
                );
            }
            x *= omega;
        }
        assert_eq!(x, Fp::ONE, "ω has order n");
    }
}

#[test]
fn inverse_transforms_invert_on_sizes_up_to_2_20() {
    let mut rng = SplitMix64::new(0x7466_6965_6c64_0011);
    let shift = Fp::SEVEN;
    for log_n in 0..=20 {
        let original = random_values(&mut rng, 1 << log_n);
        let mut values = original.clone();
        ntt(&mut values).expect("a valid length");
        // A single value is its own transform; any longer input must move.
        assert_eq!(
            values == original,
            log_n == 0,
            "n = 2^{log_n}: only a single value is its own transform"
        );
        intt(&mut values).expect("a valid length");
        assert!(values == original, "intt ∘ ntt ≠ id for n = 2^{log_n}");
        coset_ntt(&mut values, shift).expect("a valid length");
        coset_intt(&mut values, shift).expect("a valid length");
        assert!(
            values == original,
            "coset_intt ∘ coset_ntt ≠ id for n = 2^{log_n}"
        );
    }
}

#[test]
fn invalid_arguments_are_rejected_and_leave_the_values_unchanged() {
    let mut rng = SplitMix64::new(0x7466_6965_6c64_0012);
    for len in [0, 3, 5, 6, 7, 12, 1000, (1 << 20) + 1] {
        let original = random_values(&mut rng, len);
        let mut values = original.clone();
        let expected = Err(NttError::NotPowerOfTwo { len });
        assert_eq!(ntt(&mut values), expected);
        assert_eq!(intt(&mut values), expected);
        assert_eq!(coset_ntt(&mut values, Fp::SEVEN), expected);
        assert_eq!(coset_intt(&mut values, Fp::SEVEN), expected);
        assert!(values == original, "length {len}: values changed");
    }
    let original = random_values(&mut rng, 8);
    let mut values = original.clone();
    assert_eq!(coset_ntt(&mut values, Fp::ZERO), Err(NttError::ZeroShift));
    assert_eq!(coset_intt(&mut values, Fp::ZERO), Err(NttError::ZeroShift));
    assert_eq!(values, original);
}
