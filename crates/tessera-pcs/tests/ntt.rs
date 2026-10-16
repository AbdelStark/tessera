// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The transforms against their definitions: the DFT as a sum with `ω_n = 7^((P − 1)/n)`, in
//! natural order, at every size from 2^0 to 2^10; round trips at every size from 2^1 to 2^20 and
//! at 2^23, where a few outputs are checked by direct evaluation.

mod common;

use common::{Fp, SplitMix64, dft, horner, inverse_dft, root, seven};
use tessera_field::Field;
use tessera_pcs::{COSET_SHIFT, NttPlan};

#[test]
fn the_coset_shift_is_seven() {
    assert_eq!(COSET_SHIFT, seven());
}

#[test]
fn every_transform_matches_the_dft_up_to_two_to_the_ten() {
    let mut rng = SplitMix64::new(0x7E55_E7A0_0047_0001);
    for log_n in 0..=10 {
        let n = 1 << log_n;
        let plan = NttPlan::new(n).expect("a supported size");
        assert_eq!(plan.size(), n);
        let coefficients = rng.fps(n);
        for shift in [Fp::ONE, seven(), rng.nonzero()] {
            let values = dft(&coefficients, shift);

            let mut forward = coefficients.clone();
            if shift == Fp::ONE {
                plan.ntt(&mut forward).expect("matching length");
            } else {
                plan.coset_ntt(&mut forward, shift)
                    .expect("matching length");
            }
            assert_eq!(forward, values, "forward, 2^{log_n}, shift {shift:?}");

            let mut backward = values.clone();
            if shift == Fp::ONE {
                plan.intt(&mut backward).expect("matching length");
            } else {
                plan.coset_intt(&mut backward, shift)
                    .expect("matching length");
            }
            assert_eq!(
                backward, coefficients,
                "inverse, 2^{log_n}, shift {shift:?}"
            );
            // The definitions are each other's inverse too.
            assert_eq!(inverse_dft(&values, shift), coefficients);
        }
        // The coset transforms at shift 1 are the subgroup transforms.
        let mut a = coefficients.clone();
        let mut b = coefficients.clone();
        plan.ntt(&mut a).expect("matching length");
        plan.coset_ntt(&mut b, Fp::ONE).expect("matching length");
        assert_eq!(a, b, "shift 1, 2^{log_n}");
        plan.intt(&mut a).expect("matching length");
        plan.coset_intt(&mut b, Fp::ONE).expect("matching length");
        assert_eq!(a, b, "inverse at shift 1, 2^{log_n}");
    }
}

#[test]
fn a_single_coefficient_is_constant_and_a_single_monomial_is_a_geometric_sequence() {
    // f = X^e: the values ω^{je}, in natural order. A bit-reversed or mis-rooted output fails
    // this at the first odd e.
    let mut rng = SplitMix64::new(0x7E55_E7A0_0047_0002);
    for log_n in [1, 2, 5, 12] {
        let n = 1usize << log_n;
        let plan = NttPlan::new(n).expect("a supported size");
        for e in [0, 1, n / 2, n - 1, rng.below(n)] {
            let mut values = vec![Fp::ZERO; n];
            values[e] = Fp::ONE;
            plan.ntt(&mut values).expect("matching length");
            let w = root(n).pow(e as u64);
            let mut expected = Fp::ONE;
            for (j, &y) in values.iter().enumerate() {
                assert_eq!(y, expected, "X^{e} at ω^{j}, 2^{log_n}");
                expected *= w;
            }
        }
    }
}

/// Round trips both ways at size `n`, and the forward transform checked at a few positions by
/// direct evaluation.
fn round_trips(rng: &mut SplitMix64, n: usize, probes: usize) {
    let plan = NttPlan::new(n).expect("a supported size");
    let coefficients = rng.fps(n);
    let w = root(n);
    for shift in [Fp::ONE, seven()] {
        let mut values = coefficients.clone();
        if shift == Fp::ONE {
            plan.ntt(&mut values).expect("matching length");
        } else {
            plan.coset_ntt(&mut values, shift).expect("matching length");
        }
        for _ in 0..probes {
            let j = rng.below(n);
            let x = shift * w.pow(j as u64);
            assert_eq!(values[j], horner(&coefficients, x), "position {j} of {n}");
        }
        let mut back = values.clone();
        if shift == Fp::ONE {
            plan.intt(&mut back).expect("matching length");
        } else {
            plan.coset_intt(&mut back, shift).expect("matching length");
        }
        assert!(back == coefficients, "round trip of {n}, shift {shift:?}");
        // And from values to values.
        if shift == Fp::ONE {
            plan.ntt(&mut back).expect("matching length");
        } else {
            plan.coset_ntt(&mut back, shift).expect("matching length");
        }
        assert!(back == values, "inverse round trip of {n}, shift {shift:?}");
    }
}

#[test]
fn transforms_round_trip_at_every_size_to_two_to_the_twenty() {
    let mut rng = SplitMix64::new(0x7E55_E7A0_0047_0003);
    for log_n in 1..=20 {
        round_trips(&mut rng, 1 << log_n, 2);
    }
}

#[test]
fn transforms_round_trip_at_two_to_the_twenty_three() {
    let mut rng = SplitMix64::new(0x7E55_E7A0_0047_0004);
    round_trips(&mut rng, 1 << 23, 3);
}
