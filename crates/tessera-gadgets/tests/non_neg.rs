// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `NonNeg`'s soundness, RFC-0009 Lemma 3: its lookups pass iff `0 ≤ v < 2^k`, and then only with
//! `v`'s digits as limbs.
//!
//! - **Exhaustion.** Over `F65537`, every field element `v`, and over Goldilocks every `v` in
//!   `[−2^k, 2^{k+1})`, against every limb vector that passes the limbs' `Range { L }`: exactly
//!   one passes, the digits, when `v ∈ [0, 2^k)`, and none otherwise.
//! - **Boundaries**, at small and production widths: `v = 0` and `2^k − 1` pass; `v = 2^k` and
//!   `v = −1` have no honest witness, and their natural limbs, the digits of `v` modulo `2^k`,
//!   are rejected at the top limb's lookup alone.
//! - **Alternative limbs.** A limb plus `2^L`, the next limb carrying it, keeps every value an
//!   integer and is rejected by its own `Range { L }` alone, which accepts it when removed; a limb
//!   off by one leaves a top limb that is no integer.

mod common;

use common::{Fp, SplitMix64, first_rejected, rejected};
use tessera_field::{F65537, PrimeField};
use tessera_gadgets::{Committed, Constraints, GadgetError, NonNeg, Term};
use tessera_poly::PolyField;
use tessera_tables::{GeneratedTable, Table, TableId, TableKind, TableSpec};

/// `NonNeg { k, L }` with its two `Range` tables.
fn non_neg(k: u8, l: u8) -> NonNeg {
    let n = u32::from(k).div_ceil(u32::from(l));
    let top = u8::try_from(u32::from(k) - (n - 1) * u32::from(l)).expect("small");
    let specs: Vec<TableSpec> = [l, top]
        .iter()
        .zip(0..)
        .map(|(&bits, id)| {
            GeneratedTable::new(TableId(id), TableKind::Range { bits })
                .expect("a Range table")
                .spec()
                .clone()
        })
        .collect();
    NonNeg::new(k, l, &specs).expect("NonNeg")
}

/// The assignment of `v` and its committed limbs.
fn assign<F: PrimeField>(v: F, limbs: &[F]) -> impl FnMut(Term) -> F + '_ {
    move |term| match term {
        Term::Input => v,
        Term::Committed(Committed::Limb(j)) => limbs[usize::from(j)],
        other => panic!("NonNeg reads no {other:?}"),
    }
}

/// The digits of `v ≥ 0` in radix `2^l`, `count` of them.
fn digits(v: i128, l: u32, count: u32) -> Vec<i128> {
    (0..count)
        .map(|j| (v >> (j * l)) & ((1 << l) - 1))
        .collect()
}

/// Lemma 3 by exhaustion over `F`: for each `v` of `values`, every limb vector in
/// `[0, 2^L)^{n−1}`.
fn exhaust<F: PrimeField + PolyField<Base = F>>(
    k: u8,
    l: u8,
    values: impl Iterator<Item = F>,
) -> usize {
    let gadget = non_neg(k, l);
    let constraints: Constraints<F> = gadget.constraints();
    let tables = gadget.tables();
    let committed = gadget.limbs() - 1;
    let l = u32::from(l);
    let mut passing = 0;
    for v in values {
        // `v` as the integer it stands for when it is one of `[0, 2^k)`.
        let canonical = v.to_canonical_u64();
        let in_range = canonical < 1 << k;
        let mut found = 0;
        for vector in 0..1u64 << (l * committed) {
            let limbs: Vec<F> = (0..committed)
                .map(|j| F::from_u64_reduced((vector >> (j * l)) & ((1 << l) - 1)))
                .collect();
            if first_rejected(&constraints, &tables, assign(v, &limbs)).is_none() {
                found += 1;
                let truth = digits(i128::from(canonical), l, committed);
                assert!(in_range, "k = {k}: {canonical} passes");
                assert_eq!(
                    limbs,
                    truth
                        .iter()
                        .map(|&d| F::from_i64(d as i64))
                        .collect::<Vec<_>>(),
                    "k = {k}, v = {canonical}"
                );
            }
        }
        assert_eq!(found, usize::from(in_range), "k = {k}, v = {canonical}");
        passing += found;
    }
    passing
}

#[test]
fn only_values_in_range_pass_with_their_digits_by_exhaustion() {
    // Every element of F65537, against three limbs of 3 bits and a top of 2.
    let all = (0..F65537::MODULUS).map(F65537::from_u64_reduced);
    assert_eq!(exhaust::<F65537>(8, 3, all), 1 << 8);
    // A top limb of L bits, around the range and at the field's top.
    let around = (-(1i64 << 12)..1 << 13).chain([(F65537::MODULUS - 1) as i64]);
    assert_eq!(
        exhaust::<F65537>(12, 4, around.map(F65537::from_i64)),
        1 << 12
    );
    // Goldilocks: one committed limb of 6 bits beside a top of 6, and two beside a top of 2.
    let around = |k: u32| (-(1i64 << k)..1 << (k + 1)).map(Fp::from_i64);
    assert_eq!(exhaust::<Fp>(12, 6, around(12)), 1 << 12);
    assert_eq!(exhaust::<Fp>(10, 4, around(10)), 1 << 10);
}

/// `NonNeg { k, L }` at `v`: whether the honest witness exists, and the lookups that the digits
/// of `v mod 2^k` fail.
fn boundary<F: PrimeField + PolyField<Base = F>>(k: u8, l: u8, v: i128) -> (bool, Vec<usize>) {
    let gadget = non_neg(k, l);
    let committed = gadget.limbs() - 1;
    let honest = match gadget.witness(&[i64::try_from(v).expect("an i64")]) {
        Ok(_) => true,
        Err(error) => {
            assert_eq!(
                error,
                GadgetError::Input {
                    rule: "every value lies in [0, 2^k) (RFC-0009 Lemma 3)"
                }
            );
            false
        }
    };
    let limbs: Vec<F> = digits(v.rem_euclid(1 << k), u32::from(l), committed)
        .iter()
        .map(|&d| F::from_i64(d as i64))
        .collect();
    let constraints: Constraints<F> = gadget.constraints();
    let v = F::from_i64(i64::try_from(v).expect("an i64"));
    (
        honest,
        rejected(&constraints, &gadget.tables(), assign(v, &limbs)),
    )
}

#[test]
fn the_boundaries_pass_or_fail_as_lemma_3_says() {
    for (k, l) in [
        (8u8, 3u8),
        (12, 4),
        (14, 7),
        (16, 16),
        (34, 16),
        (48, 16),
        (62, 16),
    ] {
        let n = u32::from(k).div_ceil(u32::from(l));
        let top = usize::try_from(n - 1).expect("small");
        let bound = 1i128 << k;
        for (v, passes) in [(0, true), (bound - 1, true), (bound, false), (-1, false)] {
            let expected = match passes {
                true => (true, vec![]),
                false => (false, vec![top]),
            };
            assert_eq!(
                boundary::<Fp>(k, l, v),
                expected,
                "k = {k}, L = {l}, v = {v}"
            );
            if k <= 14 {
                assert_eq!(
                    boundary::<F65537>(k, l, v),
                    expected,
                    "F65537, k = {k}, v = {v}"
                );
            }
        }
    }
}

/// The lookups that limbs `limbs` of `v` fail, by integer arithmetic: each committed limb
/// outside `[0, 2^L)`, and the top limb unless `v − Σ v_j·2^{jL}` is a multiple of
/// `2^{(n−1)L}` whose quotient lies in `[0, 2^{k − (n−1)L})`.
fn predicted(v: i128, limbs: &[i128], k: u32, l: u32) -> Vec<usize> {
    let mut fails: Vec<usize> = (0..limbs.len())
        .filter(|&j| !(0..1 << l).contains(&limbs[j]))
        .collect();
    let shift = l * limbs.len() as u32;
    let rest = v
        - (0..limbs.len())
            .map(|j| limbs[j] << (l * j as u32))
            .sum::<i128>();
    let quotient = rest >> shift;
    if rest != quotient << shift || !(0..1 << (k - shift)).contains(&quotient) {
        fails.push(limbs.len());
    }
    fails
}

#[test]
fn alternative_limbs_are_rejected_at_their_own_lookup() {
    let mut rng = SplitMix64::new(0x118_0301);
    for (k, l) in [(12u8, 4u8), (34, 16), (62, 16)] {
        let gadget = non_neg(k, l);
        let constraints: Constraints<Fp> = gadget.constraints();
        let tables = gadget.tables();
        let committed = (gadget.limbs() - 1) as usize;
        let (k32, l32) = (u32::from(k), u32::from(l));
        let mut alone = vec![0; committed];
        for _ in 0..2_000 {
            let v = i128::from(rng.next_u64() >> (64 - k32));
            let honest = digits(v, l32, committed as u32);
            for j in 0..committed {
                // A limb plus 2^L, the next limb (committed or the virtual top) carrying it; a
                // limb minus 2^L the same; a limb off by one, which nothing carries.
                let mut alternatives = Vec::new();
                for d in [1i128, -1] {
                    let mut limbs = honest.clone();
                    limbs[j] += d << l32;
                    if j + 1 < committed {
                        limbs[j + 1] -= d;
                    }
                    alternatives.push(limbs);
                    let mut limbs = honest.clone();
                    limbs[j] += d;
                    alternatives.push(limbs);
                }
                for limbs in alternatives {
                    let field: Vec<Fp> = limbs.iter().map(|&x| Fp::from_i64(x as i64)).collect();
                    let vf = Fp::from_i64(v as i64);
                    let found = rejected(&constraints, &tables, assign(vf, &field));
                    let expected = predicted(v, &limbs, k32, l32);
                    assert!(!expected.is_empty());
                    assert_eq!(found, expected, "k = {k}, v = {v}, limbs {limbs:?}");
                    if expected == [j] {
                        // Its own range check alone stands in the way.
                        let mut without = constraints.clone();
                        without.lookups.remove(j);
                        assert!(rejected(&without, &tables, assign(vf, &field)).is_empty());
                        alone[j] += 1;
                    }
                }
            }
        }
        assert!(alone.iter().all(|&count| count > 0), "k = {k}: {alone:?}");
    }
}
