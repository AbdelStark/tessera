// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The claim a round leaves (RFC-0010 'Sumcheck messages'): the round polynomial `p`, of degree
//! at most `d ≤ 10`, known at the nodes `0, 1, …, d`, evaluated at the round's challenge.
//!
//! The prover may evaluate it with Lagrange's basis; this module takes Newton's forward
//! differences instead (SP-6: the two should not share a structural bug). With
//! `a_k = Δ^k p(0)`, the `k`-th forward difference at 0, and `C(r, k) = r(r − 1)⋯(r − k + 1)/k!`,
//!
//! ```text
//! p(r) = Σ_k a_k · C(r, k) = a_0 + (r/1)·(a_1 + ((r − 1)/2)·(a_2 + ⋯ + ((r − d + 1)/d)·a_d))
//! ```
//!
//! evaluated from the inside out. Only the constants `1/1, …, 1/d` are inverted, so no value
//! that depends on the challenge is ever divided by, and a challenge equal to a node needs no
//! special case.

use tv_field::{Fp, Fp2};

use crate::sumcheck::MAX_DEGREE;

/// The nodes of a round polynomial of the largest degree: `0, 1, …, 10`.
const NODES: usize = MAX_DEGREE + 1;

/// The nodes `k = 0, 1, …, 10` in `Fp2`.
const NODE: [Fp2; NODES] = [
    Fp2::new(Fp::reduce(0), Fp::ZERO),
    Fp2::new(Fp::reduce(1), Fp::ZERO),
    Fp2::new(Fp::reduce(2), Fp::ZERO),
    Fp2::new(Fp::reduce(3), Fp::ZERO),
    Fp2::new(Fp::reduce(4), Fp::ZERO),
    Fp2::new(Fp::reduce(5), Fp::ZERO),
    Fp2::new(Fp::reduce(6), Fp::ZERO),
    Fp2::new(Fp::reduce(7), Fp::ZERO),
    Fp2::new(Fp::reduce(8), Fp::ZERO),
    Fp2::new(Fp::reduce(9), Fp::ZERO),
    Fp2::new(Fp::reduce(10), Fp::ZERO),
];

/// `1/(k + 1)` in `Fp` for `k = 0, 1, …, 10`, the unique `x < P` with `(k + 1)·x ≡ 1 (mod P)`;
/// a unit test checks each.
const INVERSE_OF_NEXT: [Fp; NODES] = [
    Fp::reduce(0x0000_0000_0000_0001),
    Fp::reduce(0x7FFF_FFFF_8000_0001),
    Fp::reduce(0xAAAA_AAAA_0000_0001),
    Fp::reduce(0xBFFF_FFFF_4000_0001),
    Fp::reduce(0xCCCC_CCCC_0000_0001),
    Fp::reduce(0xD555_5554_8000_0001),
    Fp::reduce(0x2492_4924_6DB6_DB6E),
    Fp::reduce(0xDFFF_FFFF_2000_0001),
    Fp::reduce(0x38E38E38AAAAAAAB),
    Fp::reduce(0xE666666580000001),
    Fp::reduce(0x745D17455D1745D2),
];

/// The claim after a round whose message is `p(0) = p0` and `p(2), …, p(d) = rest`, the claim
/// before it `claim` and its challenge `r`: `p(r)`, with `p(1) = claim − p(0)`.
///
/// `rest` has at most `MAX_DEGREE − 1` values, which the caller checks; any beyond are ignored.
pub(crate) fn next_claim(p0: Fp2, rest: &[Fp2], claim: Fp2, r: Fp2) -> Fp2 {
    let mut values = [Fp2::ZERO; NODES];
    let nodes = [p0, claim - p0].into_iter().chain(rest.iter().copied());
    let mut count: usize = 0;
    for (slot, value) in values.iter_mut().zip(nodes) {
        *slot = value;
        count = count.saturating_add(1);
    }
    evaluate(&values, count, r)
}

/// `p(r)` for the polynomial of degree below `count` whose values at `0, 1, …, count − 1` are
/// the first `count` of `values`.
fn evaluate(values: &[Fp2; NODES], count: usize, r: Fp2) -> Fp2 {
    // differences[k] = Δ^k p(0). Row k of the difference table starts as row k − 1 shifted by
    // one difference; differences[k] needs only values[0..=k], so the entries a row leaves
    // stale at its end are never read.
    let mut differences = [Fp2::ZERO; NODES];
    let mut row = *values;
    for difference in differences.iter_mut().take(count) {
        let [head, ..] = row;
        *difference = head;
        let previous = row;
        for (slot, pair) in row.iter_mut().zip(previous.windows(2)) {
            if let [low, high] = pair {
                *slot = *high - *low;
            }
        }
    }
    // a_0 + (r/1)·(a_1 + ((r − 1)/2)·(a_2 + …)), from the inside out: the innermost term is
    // a_{count−1}, reached with an accumulator of zero.
    differences
        .iter()
        .zip(NODE)
        .zip(INVERSE_OF_NEXT)
        .take(count)
        .rev()
        .fold(Fp2::ZERO, |inner, ((a, node), inverse)| {
            *a + inner * (r - node) * inverse
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tv_testkit::{Json, SplitMix64, read_vectors};

    fn fp(text: &Json) -> Fp {
        Fp::from_canonical(text.as_hex_u64().unwrap()).unwrap()
    }

    fn fp2(pair: &Json) -> Fp2 {
        let pair = pair.as_array().unwrap();
        Fp2::new(fp(&pair[0]), fp(&pair[1]))
    }

    #[test]
    fn the_constants_are_the_nodes_and_the_inverses_of_their_successors() {
        for k in 0..NODES {
            assert_eq!(NODE[k], Fp2::from(Fp::reduce(k as u64)));
            assert_eq!(
                Fp::reduce(k as u64 + 1) * INVERSE_OF_NEXT[k],
                Fp::ONE,
                "1/{}",
                k + 1
            );
        }
    }

    /// `spec/vectors/poly/poly.json` 'round_polys', written by an independent oracle: a round
    /// message `p(0), p(2), …, p(d)`, the claim `p(0) + p(1)`, a challenge and `p(r)`, for every
    /// degree from 1 to 7, the last case of each at a node.
    #[test]
    fn every_round_polynomial_of_the_poly_vectors() {
        let file = read_vectors("poly/poly.json");
        let cases = file.get("round_polys").and_then(Json::as_array).unwrap();
        assert_eq!(cases.len(), 28);
        for (n, case) in cases.iter().enumerate() {
            let message: Vec<Fp2> = case
                .get("evaluations")
                .and_then(Json::as_array)
                .unwrap()
                .iter()
                .map(fp2)
                .collect();
            let degree = case.get("degree").and_then(Json::as_u64).unwrap() as usize;
            assert_eq!(message.len(), degree, "case {n}");
            let claim = fp2(case.get("claim").unwrap());
            let r = fp2(case.get("r").unwrap());
            let value = fp2(case.get("value").unwrap());
            assert_eq!(
                next_claim(message[0], &message[1..], claim, r),
                value,
                "case {n}"
            );
        }
    }

    fn random(rng: &mut SplitMix64) -> Fp2 {
        Fp2::new(Fp::reduce(rng.next_u64()), Fp::reduce(rng.next_u64()))
    }

    /// `p(r)` by Horner's rule on coefficients, low degree first.
    fn horner(coefficients: &[Fp2], r: Fp2) -> Fp2 {
        coefficients
            .iter()
            .rev()
            .fold(Fp2::ZERO, |acc, c| acc * r + *c)
    }

    /// For seeded polynomials of every degree, the value at a random point and at every node is
    /// the polynomial's, computed from its coefficients.
    #[test]
    fn interpolates_every_degree_exactly() {
        let mut rng = SplitMix64::new(0x5eed_0059);
        for degree in 1..=MAX_DEGREE {
            for _ in 0..64 {
                let coefficients: Vec<Fp2> = (0..=degree).map(|_| random(&mut rng)).collect();
                let at = |k: u64| horner(&coefficients, Fp2::from(Fp::reduce(k)));
                let message: Vec<Fp2> = [0].into_iter().chain(2..=degree as u64).map(at).collect();
                let claim = at(0) + at(1);
                for r in (0..=degree as u64)
                    .map(|k| Fp2::from(Fp::reduce(k)))
                    .chain([random(&mut rng), Fp2::from(Fp::reduce(8))])
                {
                    assert_eq!(
                        next_claim(message[0], &message[1..], claim, r),
                        horner(&coefficients, r),
                        "degree {degree}"
                    );
                }
            }
        }
    }

    /// A message of degree `d` fits exactly the polynomials of degree at most `d`: one of degree
    /// `d + 1` through the same nodes is not what the message interpolates.
    #[test]
    fn a_polynomial_of_higher_degree_is_not_its_interpolant() {
        let mut rng = SplitMix64::new(0x5eed_0060);
        for degree in 1..MAX_DEGREE {
            let coefficients: Vec<Fp2> = (0..=degree + 1).map(|_| random(&mut rng)).collect();
            let at = |k: u64| horner(&coefficients, Fp2::from(Fp::reduce(k)));
            let message: Vec<Fp2> = [0].into_iter().chain(2..=degree as u64).map(at).collect();
            let r = random(&mut rng);
            assert_ne!(
                next_claim(message[0], &message[1..], at(0) + at(1), r),
                horner(&coefficients, r)
            );
        }
    }
}
