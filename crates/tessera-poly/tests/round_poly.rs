// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Round polynomials in the evaluation form of RFC-0010 'Sumcheck messages': polynomials drawn by
//! their coefficients and sent as their values at `0, 2, …, d` come back exactly at any challenge,
//! from the round's claim, at every degree from 1 to 10; a wrong claim moves `p(1)` and with it
//! `p(r)`. Over Goldilocks and `F65537` and their extensions.

mod common;

use common::{SplitMix64, TestField, small};
use tessera_field::{F65537, F65537Ext2, Field, Goldilocks, GoldilocksExt2};
use tessera_poly::{MAX_DEGREE, RoundPoly};

macro_rules! over_every_field {
    ($check:ident) => {
        $check::<GoldilocksExt2>(31);
        $check::<F65537Ext2>(32);
        $check::<Goldilocks>(33);
        $check::<F65537>(34);
    };
}

/// `Σ_i c_i·x^i`.
fn horner<F: Field>(coefficients: &[F], x: F) -> F {
    coefficients
        .iter()
        .rev()
        .fold(F::ZERO, |acc, &c| acc * x + c)
}

/// A random polynomial of degree at most `d`, its round message `p(0), p(2), …, p(d)` and its
/// claim `p(0) + p(1)`.
fn draw<F: TestField>(rng: &mut SplitMix64, d: usize) -> (Vec<F>, Vec<F>, F) {
    let coefficients: Vec<F> = rng.elements(d + 1);
    let sent = core::iter::once(0)
        .chain(2..=d as u64)
        .map(|k| horner(&coefficients, small(k)))
        .collect();
    let claim = horner(&coefficients, F::ZERO) + horner(&coefficients, F::ONE);
    (coefficients, sent, claim)
}

/// A challenge off the nodes `0, …, d`.
fn off_the_nodes<F: TestField>(rng: &mut SplitMix64, d: usize) -> F {
    loop {
        let r: F = rng.element();
        if (0..=d as u64).all(|k| r != small(k)) {
            return r;
        }
    }
}

fn evaluation_recovers_the_polynomial<F: TestField>(seed: u64) {
    let mut rng = SplitMix64::new(seed);
    for d in 1..=MAX_DEGREE {
        for _ in 0..32 {
            let (coefficients, sent, claim) = draw::<F>(&mut rng, d);
            let poly = RoundPoly::new(&sent).expect("a supported degree");
            assert_eq!(poly.degree(), d);
            assert_eq!(poly.evaluations(), &sent[..]);
            let r: F = rng.element();
            assert_eq!(
                poly.evaluate(claim, r),
                horner(&coefficients, r),
                "degree {d}"
            );
            // Exact at the nodes too, where the interpolation basis degenerates, and past them.
            for k in 0..=d as u64 + 1 {
                let node = small(k);
                assert_eq!(poly.evaluate(claim, node), horner(&coefficients, node));
            }
        }
    }
}

#[test]
fn a_round_polynomial_round_trips_at_every_degree_from_1_to_10() {
    over_every_field!(evaluation_recovers_the_polynomial);
}

fn a_wrong_claim_moves_the_value_at_one<F: TestField>(seed: u64) {
    let mut rng = SplitMix64::new(seed);
    for d in 1..=MAX_DEGREE {
        let (coefficients, sent, claim) = draw::<F>(&mut rng, d);
        let poly = RoundPoly::new(&sent).expect("a supported degree");
        let delta = off_the_nodes::<F>(&mut rng, 0);
        let wrong = claim + delta;
        // The derived p(1) = claim − p(0) moves by delta; the sent values do not move.
        assert_eq!(
            poly.evaluate(wrong, F::ONE),
            horner(&coefficients, F::ONE) + delta
        );
        for k in core::iter::once(0).chain(2..=d as u64) {
            let node = small(k);
            assert_eq!(poly.evaluate(wrong, node), horner(&coefficients, node));
        }
        // So p(r) moves by delta·L_1(r), with L_1 the Lagrange basis polynomial of the node 1,
        // computed here with a field inverse: non-zero off the nodes, so p(r) changes.
        let r = off_the_nodes::<F>(&mut rng, d);
        let others = core::iter::once(0).chain(2..=d as u64);
        let numerator = others.clone().fold(F::ONE, |p, m| p * (r - small(m)));
        let denominator = others.fold(F::ONE, |p, m| p * (F::ONE - small(m)));
        let basis = numerator * denominator.inverse().expect("distinct nodes");
        assert_ne!(basis, F::ZERO);
        let moved = poly.evaluate(wrong, r) - poly.evaluate(claim, r);
        assert_eq!(moved, delta * basis, "degree {d}");
        assert_ne!(poly.evaluate(wrong, r), horner(&coefficients, r));
    }
}

#[test]
fn a_wrong_claim_changes_p_of_one_and_hence_p_of_r() {
    over_every_field!(a_wrong_claim_moves_the_value_at_one);
}
