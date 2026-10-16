// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The prover's transforms against the verifier's, `tv-field`, written separately from RFC-0002
//! and RFC-0011 (SP-6) with another algorithm and Montgomery arithmetic: `ntt`, `intt`,
//! `coset_ntt` and `coset_intt` agree on random inputs at every size from 2^1 to 2^18, at the
//! codeword shift 7 and at random shifts. The verifier re-encodes combined rows with its own
//! coset NTT (RFC-0011 'Opening protocol'), so a disagreement here would reject honest proofs or
//! accept wrong ones. Prover crates may use verifier crates as dev-dependencies only (DEP-2).

mod common;

use common::{Fp, SplitMix64};
use tessera_field::PrimeField;
use tessera_pcs::{COSET_SHIFT, NttPlan};

fn to_verifier(values: &[Fp]) -> Vec<tv_field::Fp> {
    values
        .iter()
        .map(|x| tv_field::Fp::from_canonical(x.to_canonical_u64()).expect("canonical"))
        .collect()
}

fn from_verifier(values: &[tv_field::Fp]) -> Vec<Fp> {
    values
        .iter()
        .map(|x| Fp::from_canonical_u64(x.to_canonical()).expect("canonical"))
        .collect()
}

#[test]
fn the_roots_agree() {
    for log_n in 0..=32 {
        let ours = common::root(1 << log_n);
        let theirs = tv_field::root_of_unity(log_n).expect("log_n ≤ 32");
        assert_eq!(ours.to_canonical_u64(), theirs.to_canonical(), "2^{log_n}");
    }
}

#[test]
fn every_transform_agrees_with_the_verifier_up_to_two_to_the_eighteen() {
    let mut rng = SplitMix64::new(0x7E55_E7A0_0047_0101);
    for log_n in 1..=18 {
        let n = 1 << log_n;
        let plan = NttPlan::new(n).expect("a supported size");
        let input = rng.fps(n);

        let mut ours = input.clone();
        let mut theirs = to_verifier(&input);
        plan.ntt(&mut ours).expect("matching length");
        tv_field::ntt(&mut theirs).expect("a power of two");
        assert!(ours == from_verifier(&theirs), "ntt, 2^{log_n}");

        let mut ours = input.clone();
        let mut theirs = to_verifier(&input);
        plan.intt(&mut ours).expect("matching length");
        tv_field::intt(&mut theirs).expect("a power of two");
        assert!(ours == from_verifier(&theirs), "intt, 2^{log_n}");

        for shift in [COSET_SHIFT, rng.nonzero(), rng.nonzero()] {
            let tv_shift = to_verifier(&[shift])[0];

            let mut ours = input.clone();
            let mut theirs = to_verifier(&input);
            plan.coset_ntt(&mut ours, shift).expect("matching length");
            tv_field::coset_ntt(&mut theirs, tv_shift).expect("a power of two");
            assert!(
                ours == from_verifier(&theirs),
                "coset_ntt, 2^{log_n}, shift {shift:?}"
            );

            let mut ours = input.clone();
            let mut theirs = to_verifier(&input);
            plan.coset_intt(&mut ours, shift).expect("matching length");
            tv_field::coset_intt(&mut theirs, tv_shift).expect("a power of two");
            assert!(
                ours == from_verifier(&theirs),
                "coset_intt, 2^{log_n}, shift {shift:?}"
            );
        }
    }
}
