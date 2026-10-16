// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The closed forms of `LT(r, a)`, `LT2(x, y)` and the causal selector `C(q, i)` (RFC-0010
//! 'Multilinear extensions'), each against the sum that defines it, enumerated: every integer
//! below `a`, and every pair of positions up to 2^10 per side. Then the two facts that make a
//! closed form the multilinear extension: it equals the predicate on the hypercube, and it is
//! affine in every coordinate.

mod common;

use common::{SplitMix64, TestField, affine_in, eq_vertex, vertex};
use tessera_field::{F65537, F65537Ext2, Field, Goldilocks, GoldilocksExt2};
use tessera_poly::{causal_selector, eq, eq_table, less_than, less_than_points};

macro_rules! over_every_field {
    ($check:ident) => {
        $check::<GoldilocksExt2>(21);
        $check::<F65537Ext2>(22);
        $check::<Goldilocks>(23);
        $check::<F65537>(24);
    };
}

fn indicator<F: Field>(holds: bool) -> F {
    if holds { F::ONE } else { F::ZERO }
}

fn less_than_is_its_sum<F: TestField>(seed: u64) {
    let mut rng = SplitMix64::new(seed);
    for n in 0..=10 {
        let r: Vec<F> = rng.elements(n);
        // below[a] = Σ_{h < a} eq(r, h), for every a in 0..=2^n.
        let mut below = vec![F::ZERO];
        for h in 0..1 << n {
            below.push(below[h] + eq_vertex(&r, h));
        }
        for (a, &sum) in below.iter().enumerate() {
            assert_eq!(less_than(&r, a as u64), Ok(sum), "LT(r, {a}) over {n} bits");
        }
        assert_eq!(below[0], F::ZERO);
        assert_eq!(below[1 << n], F::ONE);
    }
}

#[test]
fn less_than_equals_its_enumerated_sum_for_every_bound() {
    over_every_field!(less_than_is_its_sum);
}

/// `LT(r, a)` from the dyadic blocks of `[0, a)`, highest bit first: the block of the integers
/// that agree with `a` above a set bit `j` and clear it weighs `(1 − r_j)` times the `eq` factors
/// of `a`'s higher bits.
fn dyadic<F: Field>(r: &[F], a: u64) -> F {
    let bit = |i: usize| i < 64 && (a >> i) & 1 == 1;
    (0..r.len()).filter(|&j| bit(j)).fold(F::ZERO, |sum, j| {
        let above = (j + 1..r.len()).fold(F::ONE, |product, i| {
            product * if bit(i) { r[i] } else { F::ONE - r[i] }
        });
        sum + (F::ONE - r[j]) * above
    })
}

fn less_than_past_64_bits<F: TestField>(seed: u64) {
    let mut rng = SplitMix64::new(seed);
    // 2^63 is the largest bound that is a whole cube, and a = 2^n is the whole cube.
    let r: Vec<F> = rng.elements(63);
    assert_eq!(less_than(&r, 1 << 63), Ok(F::ONE));
    assert_eq!(less_than(&r, (1 << 63) - 1), Ok(dyadic(&r, (1 << 63) - 1)));
    for n in [64, 65, 70, 100] {
        let r: Vec<F> = rng.elements(n);
        for a in [0, 1, u64::MAX, rng.next_u64(), rng.next_u64() >> 7] {
            assert_eq!(
                less_than(&r, a),
                Ok(dyadic(&r, a)),
                "LT(r, {a:#x}) over {n} bits"
            );
        }
        // Coordinates past the 64th see zero bits of a: factors 1 − r_i.
        let a = rng.next_u64();
        let tail = r[64..]
            .iter()
            .fold(F::ONE, |product, &ri| product * (F::ONE - ri));
        assert_eq!(
            less_than(&r, a),
            Ok(less_than(&r[..64], a).expect("a < 2^64") * tail)
        );
    }
}

#[test]
fn less_than_handles_bounds_at_and_past_64_bits() {
    over_every_field!(less_than_past_64_bits);
}

fn less_than_is_the_extension_of_its_predicate<F: TestField>(seed: u64) {
    let mut rng = SplitMix64::new(seed);
    for n in 0..=6 {
        for x in 0..1usize << n {
            for a in 0..=1u64 << n {
                let expected = indicator::<F>((x as u64) < a);
                assert_eq!(
                    less_than(&vertex::<F>(x, n), a),
                    Ok(expected),
                    "[{x} < {a}]"
                );
            }
        }
    }
    for n in 1..=10 {
        let r: Vec<F> = rng.elements(n);
        let a = rng.below((1 << n) + 1);
        for j in 0..n {
            let s = rng.element();
            assert!(affine_in(|p| less_than(p, a).expect("a ≤ 2^n"), &r, j, s));
        }
    }
}

#[test]
fn less_than_agrees_with_the_predicate_and_is_multilinear() {
    over_every_field!(less_than_is_the_extension_of_its_predicate);
}

fn pair_predicates_are_their_sums<F: TestField>(seed: u64) {
    let mut rng = SplitMix64::new(seed);
    for n in 0..=10 {
        let (q, i): (Vec<F>, Vec<F>) = (rng.elements(n), rng.elements(n));
        let eq_q: Vec<F> = (0..1 << n).map(|x| eq_vertex(&q, x)).collect();
        let eq_i: Vec<F> = (0..1 << n).map(|x| eq_vertex(&i, x)).collect();
        // Every pair (q', i'), 4^n of them: i' < q', q' < i', and the diagonal.
        let (mut i_below, mut q_below, mut equal) = (F::ZERO, F::ZERO, F::ZERO);
        for (qp, &wq) in eq_q.iter().enumerate() {
            for (ip, &wi) in eq_i.iter().enumerate() {
                let weight = wq * wi;
                match ip.cmp(&qp) {
                    core::cmp::Ordering::Less => i_below += weight,
                    core::cmp::Ordering::Greater => q_below += weight,
                    core::cmp::Ordering::Equal => equal += weight,
                }
            }
        }
        assert_eq!(less_than_points(&i, &q), Ok(i_below), "LT2(i, q), {n} bits");
        assert_eq!(less_than_points(&q, &i), Ok(q_below), "LT2(q, i), {n} bits");
        assert_eq!(
            causal_selector(&q, &i),
            Ok(i_below + equal),
            "C(q, i), {n} bits"
        );
        assert_eq!(eq(&q, &i), Ok(equal), "eq(q, i), {n} bits");
    }
}

#[test]
fn less_than_points_and_the_causal_selector_match_enumeration_up_to_2_10() {
    over_every_field!(pair_predicates_are_their_sums);
}

fn pair_predicates_are_extensions<F: TestField>(seed: u64) {
    let mut rng = SplitMix64::new(seed);
    for n in 0..=4 {
        for x in 0..1 << n {
            for y in 0..1 << n {
                let (vx, vy) = (vertex::<F>(x, n), vertex::<F>(y, n));
                assert_eq!(
                    less_than_points(&vx, &vy),
                    Ok(indicator(x < y)),
                    "[{x} < {y}]"
                );
                // C(q, i) with q = x and i = y: key y is visible from query x when y ≤ x.
                assert_eq!(
                    causal_selector(&vx, &vy),
                    Ok(indicator(y <= x)),
                    "[{y} ≤ {x}]"
                );
            }
        }
    }
    for n in 1..=8 {
        let (x, y): (Vec<F>, Vec<F>) = (rng.elements(n), rng.elements(n));
        for j in 0..n {
            let s = rng.element();
            let lt_x = |p: &[F]| less_than_points(p, &y).expect("lengths");
            let lt_y = |p: &[F]| less_than_points(&x, p).expect("lengths");
            let c_q = |p: &[F]| causal_selector(p, &y).expect("lengths");
            let c_i = |p: &[F]| causal_selector(&x, p).expect("lengths");
            assert!(affine_in(lt_x, &x, j, s), "LT2 in x_{j}");
            assert!(affine_in(lt_y, &y, j, s), "LT2 in y_{j}");
            assert!(affine_in(c_q, &x, j, s), "C in q_{j}");
            assert!(affine_in(c_i, &y, j, s), "C in i_{j}");
        }
    }
}

#[test]
fn less_than_points_and_the_causal_selector_agree_with_their_predicates_and_are_multilinear() {
    over_every_field!(pair_predicates_are_extensions);
}

fn pair_identities<F: TestField>(seed: u64) {
    let mut rng = SplitMix64::new(seed);
    for n in [0, 1, 5, 17, 40] {
        let (x, y): (Vec<F>, Vec<F>) = (rng.elements(n), rng.elements(n));
        let lt = |a: &[F], b: &[F]| less_than_points(a, b).expect("lengths");
        let e = eq(&x, &y).expect("lengths");
        // C(q, i) = LT2(i, q) + eq(i, q).
        assert_eq!(causal_selector(&y, &x), Ok(lt(&x, &y) + e));
        // Every pair is ordered one way, the other, or equal, and all eq weights sum to 1.
        assert_eq!(lt(&x, &y) + lt(&y, &x) + e, F::ONE, "{n} bits");
        // LT2 against a constant integer point is LT: x' < a exactly when a is the vertex y'.
        if n <= 16 {
            let a = rng.below(1 << n) as usize;
            let point = vertex::<F>(a, n);
            assert_eq!(Ok(lt(&x, &point)), less_than(&x, a as u64));
        }
    }
    // The eq table of q weighs the causal mask's rows: Σ_{q'} eq(q, q')·C(q', i) = C(q, i).
    let n = 6;
    let (q, i): (Vec<F>, Vec<F>) = (rng.elements(n), rng.elements(n));
    let rows = eq_table(&q).expect("a small table");
    let weighted = rows.iter().enumerate().fold(F::ZERO, |sum, (qp, &w)| {
        sum + w * causal_selector(&vertex::<F>(qp, n), &i).expect("lengths")
    });
    assert_eq!(causal_selector(&q, &i), Ok(weighted));
}

#[test]
fn the_causal_selector_is_less_than_plus_eq_and_the_orders_partition_the_pairs() {
    over_every_field!(pair_identities);
}
