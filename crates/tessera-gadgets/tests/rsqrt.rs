// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Integer witnesses and Goldilocks alternatives for RFC-0009 Lemma 7.
mod common;
use common::SplitMix64;
use tessera_field::{F65537, Field, Goldilocks as F, PrimeField};
use tessera_gadgets::{
    Bracket as B, CompareColumn as C, GadgetError, RsqrtColumn as Col, RsqrtRow, RsqrtTerm as T,
    Side,
};
use tessera_tables::{GeneratedTable, Table, TableId, TableKind, TableSpec};

fn specs() -> Vec<TableSpec> {
    (1..=32)
        .map(|bits| {
            GeneratedTable::new(TableId(u32::from(bits)), TableKind::Range { bits })
                .unwrap()
                .spec()
                .clone()
        })
        .collect()
}
fn gadget(d: u32, k: u8, e: u64) -> RsqrtRow {
    RsqrtRow::new(d, k, e, &specs()).unwrap()
}

#[test]
fn every_root_width_uses_the_independent_canonical_column_and_request_schema() {
    for line in include_str!("../../../spec/vectors/rsqrt/canonical.txt")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let expected: Vec<u64> = line
            .split_whitespace()
            .map(|part| part.parse().unwrap())
            .collect();
        let root = gadget(expected[1] as u32, expected[2] as u8, expected[3]);
        assert_eq!(u64::from(root.root_bits()), expected[0]);
        let constraints = root.constraints::<F>().unwrap();
        assert_eq!(
            constraints.committed.len() as u64,
            expected[4],
            "width={}",
            expected[0]
        );
        assert_eq!(constraints.lookups.len() as u64, expected[5]);
        let requests: Vec<_> = constraints
            .lookups
            .iter()
            .map(|lookup| u64::from(lookup.table.0))
            .collect();
        assert_eq!(
            requests,
            expected[6..],
            "width={}: canonical request order",
            expected[0]
        );
        let witness = root
            .witness(&[1, (1 << 55) - 1, 0], &[true, true, false])
            .unwrap();
        for column in &constraints.committed {
            assert_eq!(witness.tensor(*column).unwrap()[2], 0, "padded column");
        }
        let limbs = (2 * expected[0] + 55).div_ceil(8) as usize;
        assert!(
            witness
                .tensor(Col::Compare(B::Lower, C::Carry(Side::Left, limbs)))
                .is_none()
        );
        let columns: Vec<_> = witness
            .into_tensors()
            .into_iter()
            .map(|(column, _)| column)
            .collect();
        assert_eq!(
            columns, constraints.committed,
            "canonical owned-column order"
        );
    }
}
fn get(values: &[(Col, F)], c: Col) -> F {
    values.iter().find(|(k, _)| *k == c).unwrap().1
}
fn failures(g: &RsqrtRow, u: F, active: bool, values: &[(Col, F)]) -> (Vec<usize>, Vec<usize>) {
    let c = g.constraints::<F>().unwrap();
    let value = |t| match t {
        T::Input => u,
        T::RowSelector => F::from_u64_reduced(u64::from(active)),
        T::Column(c) => get(values, c),
    };
    let eq = c
        .equalities
        .iter()
        .enumerate()
        .filter(|(_, e)| e.evaluate(value) != F::ZERO)
        .map(|(i, _)| i)
        .collect();
    let look = c
        .lookups
        .iter()
        .enumerate()
        .filter(|(_, l)| {
            let table = g
                .tables()
                .into_iter()
                .find(|t| t.spec().id == l.table)
                .unwrap();
            let TableKind::Range { bits } = table.spec().kind else {
                panic!("range")
            };
            l.key.evaluate(value).to_canonical_u64() >= 1u64 << bits
        })
        .map(|(i, _)| i)
        .collect();
    (eq, look)
}

// A cheating prover's assignment from arbitrary integer root and decomposition. Compute
// normalization and modular subtraction even when false, leaving the final borrow to reject it.
fn assignment(g: &RsqrtRow, n: u128, u: u128, r: u64, active: bool) -> Vec<(Col, F)> {
    let mut v = vec![(Col::Root, F::from_u64_reduced(r))];
    for j in 0..3 {
        v.push((
            Col::InputLimb(j),
            F::from_u64_reduced(((u >> (16 * j)) & 65535) as u64),
        ));
    }
    let nr = usize::from(g.root_bits()).div_ceil(8);
    let limbs = (2 * usize::from(g.root_bits()) + 55).div_ceil(8);
    for j in 0..nr - 1 {
        v.push((
            Col::RootLimb(j as u8),
            F::from_u64_reduced((r >> (8 * j)) & 255),
        ));
    }
    for b in [B::Lower, B::Upper] {
        let mut p = vec![0u128; limbs];
        for i in 0..nr {
            for j in 0..nr {
                for k in 0..4 {
                    let a = u128::from((r >> (8 * i)) & if i + 1 == nr { u64::MAX } else { 255 })
                        + u128::from(b == B::Upper && i == 0);
                    let c = u128::from((r >> (8 * j)) & if j + 1 == nr { u64::MAX } else { 255 })
                        + u128::from(b == B::Upper && j == 0);
                    p[i + j + 2 * k] += a * c * ((u >> (16 * k)) & 65535);
                }
            }
        }
        let mut constant = vec![0u128; limbs];
        for (j, x) in constant.iter_mut().enumerate() {
            *x = if active {
                ((n + u128::from(b == B::Upper)) >> (8 * j)) & 255
            } else {
                0
            };
        }
        let full = p.iter().rev().fold(0u128, |s, x| s * 256 + x);
        assert_eq!(full, u128::from(r + u64::from(b == B::Upper)).pow(2) * u);
        let (x, y) = if b == B::Lower {
            (p, constant)
        } else {
            (constant, p)
        };
        let mut ds = [vec![0i128; limbs], vec![0i128; limbs]];
        for (index, (side, input)) in [(Side::Left, x), (Side::Right, y)].into_iter().enumerate() {
            let mut carry = 0;
            for (j, a) in input.into_iter().enumerate() {
                let s = a + carry;
                ds[index][j] = (s & 255) as i128;
                carry = s >> 8;
                v.push((
                    Col::Compare(b, C::Digit(side, j)),
                    F::from_u64_reduced((s & 255) as u64),
                ));
                v.push((
                    Col::Compare(b, C::Carry(side, j)),
                    F::from_u64_reduced(carry as u64),
                ));
            }
        }
        let mut borrow = 0;
        for (j, (&right, &left)) in ds[1].iter().zip(&ds[0]).enumerate() {
            let delta = right - left - borrow;
            borrow = i128::from(delta < 0);
            v.push((
                Col::Compare(b, C::Difference(j)),
                F::from_i64(delta.rem_euclid(256) as i64),
            ));
            v.push((Col::Compare(b, C::Borrow(j)), F::from_i64(borrow as i64)));
        }
    }
    v
}

#[test]
fn honest_witnesses_and_all_neighbouring_roots() {
    let mut rng = SplitMix64::new(126);
    for (d, k, e) in [
        (1, 0, 1),
        (4096, 30, 672),
        (16384, 30, 2685),
        (1, 31, 1),
        (u32::MAX, 16, 1),
    ] {
        let g = gadget(d, k, e);
        let limbs = (2 * usize::from(g.root_bits()) + 55).div_ceil(8);
        let n = u128::from(d) << (2 * k);
        let mut us = vec![e as i64, (1i64 << 55) - 1];
        us.extend((0..30).map(|_| (rng.below((1u64 << 55) - e) + e) as i64));
        let w = g.witness(&us, &vec![true; us.len()]).unwrap();
        let c = g.constraints::<F>().unwrap();
        for (row, &u) in us.iter().enumerate() {
            let r = w.tensor(Col::Root).unwrap()[row] as u64;
            assert_eq!(u128::from(r), (n / u as u128).isqrt());
            let values = c
                .committed
                .iter()
                .map(|&col| (col, F::from_i64(w.tensor(col).unwrap()[row])))
                .collect::<Vec<_>>();
            assert_eq!(
                failures(&g, F::from_i64(u), true, &values),
                (vec![], vec![])
            );
            for alt in [r.checked_sub(1), r.checked_add(1)].into_iter().flatten() {
                let a = assignment(&g, n, u as u128, alt, true);
                let (eq, look) = failures(&g, F::from_i64(u), true, &a);
                if alt >= 1u64 << g.root_bits() {
                    assert!(look.contains(&0));
                } else {
                    assert!(look.is_empty());
                    // Each comparison has 3m+3 equalities; its final borrow rejects.
                    assert_eq!(
                        eq,
                        if alt < r {
                            vec![6 * limbs + 5]
                        } else {
                            vec![3 * limbs + 2]
                        }
                    );
                }
            }
        }
        assert!(g.witness(&[], &[]).is_ok());
    }
}

#[test]
fn padding_requires_zero_root_even_when_both_comparisons_pass() {
    let g = gadget(4096, 30, 672);
    let w = g.witness(&[0], &[false]).unwrap();
    assert_eq!(w.tensor(Col::Root), Some(&[0][..]));
    for r in [0, 1, 255, 1 << 30] {
        let a = assignment(&g, 4096u128 << 60, 0, r, false);
        assert_eq!(
            failures(&g, F::ZERO, false, &a),
            (if r == 0 { vec![] } else { vec![96] }, vec![])
        );
    }
}

#[test]
fn goldilocks_u_plus_p_is_rejected_by_top_seven_bit_lookup() {
    let g = gadget(1, 20, 1);
    let n = 1u128 << 40;
    let u = u128::from(F::MODULUS) + 1;
    let a = assignment(&g, n, u, 0, true);
    assert_eq!(failures(&g, F::ONE, true, &a), (vec![], vec![4]));
    assert!(0u128.pow(2) * u <= n && u > n);
    assert_eq!(n.isqrt(), 1 << 20);
}

#[test]
fn each_committed_column_is_bound_and_each_table_needed() {
    let g = gadget(4096, 30, 672);
    let n = 4096u128 << 60;
    let u = 672;
    let r = (n / u).isqrt() as u64;
    let a = assignment(&g, n, u, r, true);
    assert_eq!(
        failures(&g, F::from_u64_reduced(u as u64), true, &a),
        (vec![], vec![])
    );
    for j in 0..a.len() {
        let mut corrupt = a.clone();
        corrupt[j].1 += F::ONE;
        assert_ne!(
            failures(&g, F::from_u64_reduced(u as u64), true, &corrupt),
            (vec![], vec![]),
            "{:?}",
            a[j].0
        );
    }
    for t in g.tables() {
        let mut s = specs();
        s.retain(|x| x.id != t.spec().id);
        assert!(RsqrtRow::new(4096, 30, 672, &s).is_err());
    }
}

#[test]
fn public_parameters_input_domains_and_errors_are_bounded_and_redacted() {
    for (d, k, e) in [
        (0, 16, 1),
        (1, 32, 1),
        (1, 32, 2),
        (1, 16, 0),
        (1, 16, 1 << 55),
        (u32::MAX, 31, 1),
    ] {
        assert!(matches!(
            RsqrtRow::new(d, k, e, &specs()),
            Err(GadgetError::Shape { .. })
        ));
    }
    let g = gadget(4096, 30, 672);
    for (u, real) in [
        (vec![-1], vec![true]),
        (vec![0], vec![true]),
        (vec![671], vec![true]),
        (vec![1 << 55], vec![true]),
        (vec![i64::MAX], vec![true]),
        (vec![1], vec![false]),
    ] {
        let error = g.witness(&u, &real).unwrap_err();
        assert!(matches!(error, GadgetError::Input { .. }));
        assert_eq!(error.code(), tessera_codec::error::ErrorCode::Internal);
    }
    assert!(matches!(
        g.witness(&[1], &[]),
        Err(GadgetError::Shape { .. })
    ));
    assert!(g.constraints::<F65537>().is_err());
    let w = g.witness(&[672], &[true]).unwrap();
    let debug = format!("{w:?}");
    let root = w.tensor(Col::Root).unwrap()[0].to_string();
    assert!(!debug.contains(&root));
}

#[test]
fn independent_python_integer_vectors() {
    let mut count = 0;
    for line in include_str!("../../../spec/vectors/rsqrt/roots.txt")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let v: Vec<u128> = line
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        assert_eq!(v.len(), 8);
        let g = gadget(v[0] as u32, v[1] as u8, v[2] as u64);
        let u = v[3];
        let active = v[4] == 1;
        let w = g.witness(&[u as i64], &[active]).unwrap();
        let r = w.tensor(Col::Root).unwrap()[0] as u128;
        assert_eq!(r, v[5]);
        assert_eq!(r * r * u, v[6]);
        assert_eq!((r + 1) * (r + 1) * u, v[7]);
        let c = g.constraints::<F>().unwrap();
        let values: Vec<_> = c
            .committed
            .iter()
            .map(|&col| (col, F::from_i64(w.tensor(col).unwrap()[0])))
            .collect();
        assert_eq!(
            failures(&g, F::from_u64_reduced(u as u64), active, &values),
            (vec![], vec![])
        );
        count += 1;
    }
    assert_eq!(count, 60);
}

#[test]
fn full_119_bit_products_and_root_plus_one_carry_do_not_truncate() {
    let g = gadget(4096, 30, 672);
    let n = 4096u128 << 60;
    let u = (1u128 << 55) - 1;
    let r = u64::from(u32::MAX);
    let a = assignment(&g, n, u, r, true);
    assert_eq!(
        failures(&g, F::from_u64_reduced(u as u64), true, &a),
        (vec![47], vec![])
    );
    assert!(u128::from(r).pow(2) * u > 1u128 << 118);
    assert_eq!(u128::from(r + 1).pow(2) * u, (1u128 << 64) * u);
}

#[test]
fn the_first_input_outside_55_bits_fails_exactly_its_top_range() {
    let g = gadget(4096, 30, 672);
    let n = 4096u128 << 60;
    let u = 1u128 << 55;
    let r = (n / u).isqrt() as u64;
    let a = assignment(&g, n, u, r, true);
    assert_eq!(
        failures(&g, F::from_u64_reduced(u as u64), true, &a),
        (vec![], vec![4])
    );
}

#[test]
fn every_root_width_and_partial_top_byte_has_an_exact_bracket() {
    for k in 0..32 {
        let g = gadget(1, k, 1);
        let n = 1u128 << (2 * k);
        assert_eq!(g.root_bits(), k + 1);
        let inputs = [1, 2, 3, 255, 256, 257, (1i64 << 55) - 1];
        let w = g.witness(&inputs, &[true; 7]).unwrap();
        let c = g.constraints::<F>().unwrap();
        for (row, &u) in inputs.iter().enumerate() {
            assert_eq!(
                w.tensor(Col::Root).unwrap()[row] as u128,
                (n / u as u128).isqrt()
            );
            let values = c
                .committed
                .iter()
                .map(|&col| (col, F::from_i64(w.tensor(col).unwrap()[row])))
                .collect::<Vec<_>>();
            assert_eq!(
                failures(&g, F::from_i64(u), true, &values),
                (vec![], vec![])
            );
        }
    }
}
