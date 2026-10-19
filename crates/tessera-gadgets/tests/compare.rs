// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0009 Lemma 4: independent integer oracles, alternatives and small-field wraps.

mod common;

use common::SplitMix64;
use tessera_field::{F65537, Field, Goldilocks, PrimeField};
use tessera_gadgets::{BigIntCompare, CompareColumn as C, CompareConstraints, GadgetError, Side};
use tessera_poly::PolyField;
use tessera_tables::{GeneratedTable, Table, TableId, TableKind, TableSpec};

fn tables() -> Vec<TableSpec> {
    (1..=32)
        .map(|bits| {
            GeneratedTable::new(TableId(u32::from(bits)), TableKind::Range { bits })
                .unwrap()
                .spec()
                .clone()
        })
        .collect()
}

fn gadget(l: u8, bounds: &[u64]) -> BigIntCompare {
    BigIntCompare::new(l, bounds, bounds, &tables()).unwrap()
}

fn digits(x: u128, l: u8, n: usize) -> Vec<i64> {
    (0..n)
        .map(|j| ((x >> (j * usize::from(l))) & ((1u128 << l) - 1)) as i64)
        .collect()
}

/// A separate integer implementation, storing even a false comparison's modular difference.
fn assignment<F: PrimeField>(l: u8, left: &[i64], right: &[i64]) -> Vec<(C, F)> {
    let base = 1i128 << l;
    let mut values = Vec::new();
    let mut normalized = Vec::new();
    for (side, input) in [(Side::Left, left), (Side::Right, right)] {
        let mut carry = 0;
        let mut ds = Vec::new();
        for (j, &raw) in input.iter().enumerate() {
            let total = i128::from(raw) + carry;
            let digit = total.rem_euclid(base);
            carry = total.div_euclid(base);
            values.extend([
                (C::Input(side, j), F::from_i64(raw)),
                (C::Digit(side, j), F::from_i64(digit as i64)),
                (C::Carry(side, j), F::from_i64(carry as i64)),
            ]);
            ds.push(digit);
        }
        normalized.push(ds);
    }
    let mut incoming = 0;
    for (j, (&x, &y)) in normalized[0].iter().zip(&normalized[1]).enumerate() {
        let delta = y - x - incoming;
        let (digit, borrow) = if delta < 0 {
            (delta + base, 1)
        } else {
            (delta, 0)
        };
        values.push((C::Difference(j), F::from_i64(digit as i64)));
        values.push((C::Borrow(j), F::from_i64(borrow)));
        incoming = i128::from(borrow);
    }
    values
}

fn get<F: Copy>(values: &[(C, F)], column: C) -> F {
    values.iter().find(|(c, _)| *c == column).unwrap().1
}

fn set<F: Copy>(values: &mut [(C, F)], column: C, value: F) {
    values.iter_mut().find(|(c, _)| *c == column).unwrap().1 = value;
}

#[derive(Debug, PartialEq, Eq)]
enum Failure {
    Equality(usize),
    Range(C),
}

fn failures<F: PrimeField + PolyField<Base = F>>(
    g: &BigIntCompare,
    constraints: &CompareConstraints<F>,
    values: &[(C, F)],
) -> Vec<Failure> {
    let mut failed = Vec::new();
    for (i, eq) in constraints.equalities.iter().enumerate() {
        if eq.evaluate::<F>(|c| get(values, c)) != F::ZERO {
            failed.push(Failure::Equality(i));
        }
    }
    for lookup in &constraints.lookups {
        let table = g
            .tables()
            .iter()
            .find(|t| t.spec().id == lookup.table)
            .unwrap();
        let key = lookup.key.evaluate::<F>(|c| get(values, c));
        if key.to_canonical_u64() >= table.row_count() {
            failed.push(Failure::Range(lookup.key.terms[0].1));
        }
    }
    failed
}

#[test]
fn random_64_to_128_bit_comparisons_in_both_directions() {
    let mut rng = SplitMix64::new(121);
    for width in [64usize, 80, 96, 112, 128] {
        for l in [4u8, 8, 16, 32] {
            let n = width.div_ceil(usize::from(l));
            let g = gadget(l, &vec![(1u64 << l) - 1; n]);
            let constraints = g.constraints::<Goldilocks>().unwrap();
            let maximum = u128::MAX >> (128 - width);
            for trial in 0..120 {
                let random = |rng: &mut SplitMix64| {
                    ((u128::from(rng.next_u64()) << 64) | u128::from(rng.next_u64())) & maximum
                };
                let (x, y) = match trial {
                    0 => (0, maximum),
                    1 => (maximum, maximum),
                    2 => (maximum - 1, maximum),
                    3 => (0, 0),
                    _ => (random(&mut rng), random(&mut rng)),
                };
                for (x, y) in [(x, y), (y, x)] {
                    let left = digits(x, l, n);
                    let right = digits(y, l, n);
                    let oracle = assignment::<Goldilocks>(l, &left, &right);
                    let rejected = failures(&g, &constraints, &oracle);
                    let final_borrow = constraints.equalities.len() - 1;
                    assert_eq!(
                        rejected,
                        if x <= y {
                            vec![]
                        } else {
                            vec![Failure::Equality(final_borrow)]
                        }
                    );
                    let left_columns: Vec<_> = left.iter().map(core::slice::from_ref).collect();
                    let right_columns: Vec<_> = right.iter().map(core::slice::from_ref).collect();
                    match g.witness(&left_columns, &right_columns) {
                        Ok(witness) => {
                            assert!(x <= y);
                            for &column in &constraints.committed {
                                let value = witness.tensor(column).unwrap()[0];
                                assert_eq!(Goldilocks::from_i64(value), get(&oracle, column));
                            }
                        }
                        Err(error) => {
                            assert!(x > y);
                            assert!(matches!(error, GadgetError::Input { .. }));
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn raw_product_sums_are_normalized_without_truncating_high_carries() {
    let l = 16;
    let bounds = [((1u64 << 40) - 1), 1 << 32, 0, 0];
    let g = gadget(l, &bounds);
    let constraints = g.constraints::<Goldilocks>().unwrap();
    let mut rng = SplitMix64::new(0x0caa_a121);
    for _ in 0..500 {
        let left: Vec<_> = bounds
            .iter()
            .map(|&b| (rng.next_u64() % (b + 1)) as i64)
            .collect();
        let right: Vec<_> = bounds
            .iter()
            .map(|&b| (rng.next_u64() % (b + 1)) as i64)
            .collect();
        let integer = |v: &[i64]| {
            v.iter()
                .enumerate()
                .map(|(j, &v)| (v as u128) << (j * usize::from(l)))
                .sum::<u128>()
        };
        let pass = integer(&left) <= integer(&right);
        let lc: Vec<_> = left.iter().map(core::slice::from_ref).collect();
        let rc: Vec<_> = right.iter().map(core::slice::from_ref).collect();
        assert_eq!(g.witness(&lc, &rc).is_ok(), pass);
        assert_eq!(
            failures(&g, &constraints, &assignment(l, &left, &right)).is_empty(),
            pass
        );
    }
    // X = 16 is larger than Y = 1, although truncating its high carry would compare 0 ≤ 1.
    let g = gadget(4, &[16]);
    let constraints = g.constraints::<Goldilocks>().unwrap();
    assert_eq!(
        failures(&g, &constraints, &assignment(4, &[16], &[1])),
        vec![Failure::Equality(1)]
    );
    assert!(matches!(
        g.witness(&[&[16]], &[&[1]]),
        Err(GadgetError::Input { .. })
    ));
    // The right side may not truncate either, even when it would preserve the verdict.
    assert_eq!(
        failures(&g, &constraints, &assignment(4, &[0], &[16])),
        vec![Failure::Equality(3)]
    );
}

#[test]
fn root_square_times_a_55_bit_input_is_compared_as_a_119_bit_integer() {
    let left_bounds = [vec![(1 << 34) - 1; 7], vec![0]].concat();
    let right_bounds = vec![65535; 8];
    let g = BigIntCompare::new(16, &left_bounds, &right_bounds, &tables()).unwrap();
    let constraints = g.constraints::<Goldilocks>().unwrap();
    let mut rng = SplitMix64::new(0x1210_0126);
    for trial in 0..120 {
        let rho = u128::from(rng.next_u64() as u32);
        let u = u128::from(rng.next_u64() & ((1 << 55) - 1));
        let product = rho * rho * u;
        let square_digits = digits(rho * rho, 16, 4);
        let u_digits = digits(u, 16, 4);
        let mut raw = vec![0i64; 8];
        for (i, &a) in square_digits.iter().enumerate() {
            for (j, &b) in u_digits.iter().enumerate() {
                raw[i + j] += a * b;
            }
        }
        let target = if trial % 2 == 0 {
            product + 1
        } else {
            product.saturating_sub(1)
        };
        let y = digits(target, 16, 8);
        let left: Vec<_> = raw.iter().map(core::slice::from_ref).collect();
        let right: Vec<_> = y.iter().map(core::slice::from_ref).collect();
        assert_eq!(g.witness(&left, &right).is_ok(), product <= target);
        assert_eq!(
            failures(&g, &constraints, &assignment(16, &raw, &y)).is_empty(),
            product <= target
        );
    }
}

#[test]
fn every_committed_column_off_by_one_is_rejected() {
    let g = gadget(4, &[255, 255, 0]);
    let constraints = g.constraints::<Goldilocks>().unwrap();
    let honest = assignment::<Goldilocks>(4, &[33, 19, 0], &[47, 23, 0]);
    assert!(failures(&g, &constraints, &honest).is_empty());
    for &column in &constraints.committed {
        for offset in [Goldilocks::ONE, -Goldilocks::ONE] {
            let mut alternative = honest.clone();
            let value = get(&alternative, column) + offset;
            set(&mut alternative, column, value);
            assert!(
                !failures(&g, &constraints, &alternative).is_empty(),
                "{column:?}"
            );
        }
    }
}

#[test]
fn removing_a_difference_range_check_admits_a_field_wrap() {
    let g = gadget(4, &[15]);
    let mut constraints = g.constraints::<F65537>().unwrap();
    let mut cheat = assignment::<F65537>(4, &[1], &[0]);
    // Certify the false 1 ≤ 0 using d = P − 1, not a four-bit integer, and no borrow.
    set(&mut cheat, C::Difference(0), -F65537::ONE);
    set(&mut cheat, C::Borrow(0), F65537::ZERO);
    assert_eq!(
        failures(&g, &constraints, &cheat),
        vec![Failure::Range(C::Difference(0))]
    );
    constraints
        .lookups
        .retain(|l| l.key.terms[0].1 != C::Difference(0));
    assert!(failures(&g, &constraints, &cheat).is_empty());
}

#[test]
fn each_normalization_digit_carry_and_borrow_range_is_necessary() {
    let g = gadget(32, &[u32::MAX as u64; 2]);
    let constraints = g.constraints::<Goldilocks>().unwrap();
    let p_digits = [1, i64::from(u32::MAX)];
    let inv = Goldilocks::from_u64_reduced(1 << 32).inverse().unwrap();
    // X = P, Y = 0. Without one carry range check, normalize P to zero in the field.
    let mut carry_attack = assignment::<Goldilocks>(32, &p_digits, &[0, 0]);
    for j in 0..2 {
        set(&mut carry_attack, C::Digit(Side::Left, j), Goldilocks::ZERO);
        set(&mut carry_attack, C::Difference(j), Goldilocks::ZERO);
        set(&mut carry_attack, C::Borrow(j), Goldilocks::ZERO);
    }
    set(&mut carry_attack, C::Carry(Side::Left, 0), inv);
    // The same false comparison with correct normalization, but a non-Boolean borrow.
    let mut borrow_attack = assignment::<Goldilocks>(32, &p_digits, &[0, 0]);
    for j in 0..2 {
        set(&mut borrow_attack, C::Difference(j), Goldilocks::ZERO);
        set(&mut borrow_attack, C::Borrow(j), Goldilocks::ZERO);
    }
    set(&mut borrow_attack, C::Borrow(0), inv);
    // A second witness for B-1 <= B, using digit -1 with carry 1 rather than digit B-1.
    let mut digit_attack = assignment::<Goldilocks>(32, &[i64::from(u32::MAX), 0], &[0, 1]);
    set(&mut digit_attack, C::Digit(Side::Left, 0), -Goldilocks::ONE);
    set(&mut digit_attack, C::Carry(Side::Left, 0), Goldilocks::ONE);
    set(&mut digit_attack, C::Digit(Side::Left, 1), Goldilocks::ONE);
    set(&mut digit_attack, C::Borrow(0), Goldilocks::ZERO);
    let mut right_carry_attack = assignment::<Goldilocks>(32, &[0, 0], &p_digits);
    for j in 0..2 {
        set(
            &mut right_carry_attack,
            C::Digit(Side::Right, j),
            Goldilocks::ZERO,
        );
        set(&mut right_carry_attack, C::Difference(j), Goldilocks::ZERO);
        set(&mut right_carry_attack, C::Borrow(j), Goldilocks::ZERO);
    }
    set(&mut right_carry_attack, C::Carry(Side::Right, 0), inv);
    let mut right_digit_attack = assignment::<Goldilocks>(32, &[0, 0], &[i64::from(u32::MAX), 0]);
    set(
        &mut right_digit_attack,
        C::Digit(Side::Right, 0),
        -Goldilocks::ONE,
    );
    set(
        &mut right_digit_attack,
        C::Carry(Side::Right, 0),
        Goldilocks::ONE,
    );
    set(
        &mut right_digit_attack,
        C::Digit(Side::Right, 1),
        Goldilocks::ONE,
    );
    set(&mut right_digit_attack, C::Borrow(0), Goldilocks::ONE);
    for (column, attack) in [
        (C::Carry(Side::Left, 0), carry_attack),
        (C::Borrow(0), borrow_attack),
        (C::Digit(Side::Left, 0), digit_attack),
        (C::Carry(Side::Right, 0), right_carry_attack),
        (C::Digit(Side::Right, 0), right_digit_attack),
    ] {
        assert_eq!(
            failures(&g, &constraints, &attack),
            vec![Failure::Range(column)]
        );
        let mut missing = constraints.clone();
        missing.lookups.retain(|l| l.key.terms[0].1 != column);
        assert!(failures(&g, &missing, &attack).is_empty());
    }
}

#[test]
fn every_limb_equation_has_a_single_fault_rejection() {
    let n = 3;
    let g = gadget(4, &vec![15; n]);
    let constraints = g.constraints::<Goldilocks>().unwrap();
    let honest = assignment::<Goldilocks>(4, &vec![0; n], &vec![0; n]);
    for (side, start) in [(Side::Left, 0), (Side::Right, n + 1)] {
        for j in 0..n {
            let mut cheat = honest.clone();
            set(&mut cheat, C::Input(side, j), Goldilocks::ONE);
            assert_eq!(
                failures(&g, &constraints, &cheat),
                vec![Failure::Equality(start + j)]
            );
        }
    }
    for j in 0..n {
        let mut cheat = honest.clone();
        set(&mut cheat, C::Difference(j), Goldilocks::ONE);
        assert_eq!(
            failures(&g, &constraints, &cheat),
            vec![Failure::Equality(2 * (n + 1) + j)]
        );
    }
}

#[test]
fn exactly_one_witness_by_exhaustion_in_the_small_field() {
    let g = gadget(1, &[1, 1]);
    let constraints = g.constraints::<F65537>().unwrap();
    assert_eq!(constraints.committed.len(), 12);
    for x in 0..4 {
        for y in 0..4 {
            let mut values = assignment::<F65537>(1, &digits(x, 1, 2), &digits(y, 1, 2));
            let mut passing = 0;
            for vector in 0..1 << constraints.committed.len() {
                for (j, &column) in constraints.committed.iter().enumerate() {
                    set(
                        &mut values,
                        column,
                        F65537::from_u64_reduced((vector >> j) & 1),
                    );
                }
                passing += usize::from(failures(&g, &constraints, &values).is_empty());
            }
            assert_eq!(passing, usize::from(x <= y), "{x} <= {y}");
        }
    }
}

#[test]
fn shape_bounds_and_small_field_boxes_are_checked() {
    let specs = tables();
    for (l, x, y) in [
        (0, vec![0], vec![0]),
        (33, vec![0], vec![0]),
        (16, vec![], vec![]),
        (16, vec![0], vec![0, 0]),
        (16, vec![1 << 40], vec![0]),
        (1, vec![(1 << 40) - 1], vec![0]),
    ] {
        assert!(matches!(
            BigIntCompare::new(l, &x, &y, &specs),
            Err(GadgetError::Shape { .. })
        ));
    }
    assert!(matches!(
        BigIntCompare::new(16, &[0], &[0], &[]),
        Err(GadgetError::Shape { .. })
    ));
    let g = gadget(16, &[65535]);
    assert!(matches!(
        g.constraints::<F65537>(),
        Err(GadgetError::Shape { .. })
    ));
    let g = gadget(4, &[15, 15]);
    for (left, right) in [
        (vec![], vec![&[0][..], &[0][..]]),
        (vec![&[0, 1][..], &[0][..]], vec![&[0][..], &[0][..]]),
    ] {
        assert!(matches!(
            g.witness(&left, &right),
            Err(GadgetError::Shape { .. })
        ));
    }
    for x in [-1, 16, i64::MIN, i64::MAX] {
        assert!(matches!(
            g.witness(&[&[x], &[0]], &[&[15], &[15]]),
            Err(GadgetError::Input { .. })
        ));
    }
}

#[test]
fn batches_are_column_major_and_debug_is_redacted() {
    let g = gadget(16, &[65535, 65535]);
    let left = [&[12345, 65432, 11111][..], &[1, 0, 4][..]];
    let right = [&[12346, 65500, 22222][..], &[1, 0, 4][..]];
    let witness = g.witness(&left, &right).unwrap();
    let constraints = g.constraints::<Goldilocks>().unwrap();
    for row in 0..3 {
        let x: Vec<_> = left.iter().map(|v| v[row]).collect();
        let y: Vec<_> = right.iter().map(|v| v[row]).collect();
        let oracle = assignment::<Goldilocks>(16, &x, &y);
        for &column in &constraints.committed {
            assert_eq!(
                Goldilocks::from_i64(witness.tensor(column).unwrap()[row]),
                get(&oracle, column)
            );
        }
    }
    let debug = format!("{witness:?}");
    for secret in ["12345", "65432", "11111", "22222", "65500"] {
        assert!(!debug.contains(secret));
    }
    assert!(witness.tensor(C::Input(Side::Left, 0)).is_none());
    assert!(g.witness(&[&[], &[]], &[&[], &[]]).is_ok());
}
