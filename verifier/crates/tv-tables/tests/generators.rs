// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The generators against their definitions, computed here another way, at parameters the
//! vectors do not cover: `Requant`'s `rq` of RFC-0005, `Bitwise8`, `Pow2` and `Gf256Mul` row by
//! row, `Exp`'s zero tail by a scan of RFC-0008's rule over every key, and `Silu`'s dependence on
//! `h = k >> r_c` alone.

mod support;

use support::{Rng, centered};
use tv_codec::Rational;
use tv_field::Fp;
use tv_tables::fixed::LN2;
use tv_tables::{BitOp, Table, TableId, TableKind};

/// `Requant` rows against `rq(k, r, lo, hi) = clamp(⌊(k + 2^{r−1}) / 2^r⌋, lo, hi)`, the floor
/// taken by Euclidean division, at the row `pack(c, k) = (k + 2^{b−1}) + 2^b·c` of RFC-0009.
#[test]
fn requant_rows_are_rq_of_their_signed_keys() {
    let rq = |k: i64, r: u32, lo: i32, hi: i32| -> i64 {
        let floor = (i128::from(k) + (1i128 << (r - 1))).div_euclid(1i128 << r);
        floor.clamp(i128::from(lo), i128::from(hi)) as i64
    };
    for (b, shifts, lo, hi) in [
        (5u8, vec![1u8, 2, 3], -2, 3),
        (7, vec![6, 7, 8, 63], -100, 100),
        (4, vec![1], i32::MIN, i32::MAX),
        (9, vec![3, 1], 7, 7),
    ] {
        let kind = TableKind::Requant {
            b,
            r_by_class: shifts.clone(),
            lo,
            hi,
        };
        let table = Table::generate(TableId(3), kind).unwrap();
        for (class, &r) in shifts.iter().enumerate() {
            for k in -(1i64 << (b - 1))..(1i64 << (b - 1)) {
                let row = (k + (1 << (b - 1))) as u64 + ((class as u64) << b);
                let value = rq(k, u32::from(r), lo, hi);
                assert_eq!(table.value(row), Some(value), "class {class}, k = {k}");
                assert_eq!(
                    table.row(row),
                    Some((Fp::reduce(row), centered(value.into())))
                );
            }
        }
    }
}

#[test]
fn bitwise_and_power_rows_are_their_definitions() {
    for op in [BitOp::Xor, BitOp::And] {
        let table = Table::generate(TableId(1), TableKind::Bitwise8 { op }).unwrap();
        for a in 0..256u64 {
            for b in 0..256u64 {
                let expected = match op {
                    BitOp::Xor => a ^ b,
                    BitOp::And => a & b,
                };
                assert_eq!(table.value(a + (b << 8)), Some(expected as i64));
            }
        }
    }
    let table = Table::generate(TableId(2), TableKind::Pow2 { max_exp: 63 }).unwrap();
    for j in 0..63 {
        assert_eq!(table.value(j), Some(1i64 << j));
    }
    assert_eq!(table.value(63), None);
}

/// `c • x` as a carry-less product of degree at most 14, reduced bit by bit from the top by
/// `x^8 + x^4 + x^3 + x + 1`: another algorithm than the generator's shift-and-add.
#[test]
fn gf256_rows_are_reduced_carry_less_products() {
    let product = |c: u16, x: u16| -> i64 {
        let mut wide = (0..8).fold(0u16, |acc, i| {
            if (x >> i) & 1 == 1 {
                acc ^ (c << i)
            } else {
                acc
            }
        });
        for bit in (8..15).rev() {
            if (wide >> bit) & 1 == 1 {
                wide ^= 0x11B << (bit - 8);
            }
        }
        i64::from(wide)
    };
    for c in [0u8, 1, 2, 3, 0x0E, 0x57, 0x83, 0xFF] {
        let table = Table::generate(TableId(4), TableKind::Gf256Mul { c }).unwrap();
        for x in 0..256u16 {
            assert_eq!(
                table.value(u64::from(x)),
                Some(product(u16::from(c), x)),
                "{c:#04x} • {x:#04x}"
            );
        }
    }
}

/// `z` against a scan of every key with RFC-0008's rule, `q ≥ 2^20` or `x ≥ (σ + 2)·LN2`,
/// computed here, at seeded scales that put the tail anywhere in the table or beyond it.
#[test]
fn exp_zero_tail_is_the_first_key_the_rule_puts_in_it() {
    let mut rng = Rng::new(0x7a11);
    let gcd = |mut a: u128, mut b: u128| {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    };
    for _ in 0..400 {
        let key_bits = 1 + rng.below(12) as u8;
        let sigma = rng.below(33) as u8;
        let rows = 1u64 << key_bits;
        // A scale near (σ + 2)·ln 2 / target, for a target from 1 to twice the rows.
        let target = 1 + u128::from(rng.below(2 * rows));
        let den = 1 + u128::from(rng.below(1 << 40));
        let threshold = (u128::from(sigma) + 2) * LN2;
        let num = ((threshold >> 24) * den / (target << 40)).max(1) + u128::from(rng.below(3));
        let common = gcd(num, den);
        let (num, den) = (num / common, den / common);
        let (Ok(num), Ok(den)) = (i64::try_from(num), u64::try_from(den)) else {
            continue;
        };
        let in_tail = |k: u64| {
            let product = u128::from(k) * num as u128;
            let (q, m) = (product / u128::from(den), product % u128::from(den));
            q >= 1 << 20 || (q << 64) + (m << 64) / u128::from(den) >= threshold
        };
        let expected = (0..rows).find(|&k| in_tail(k)).unwrap_or(rows);
        let kind = TableKind::Exp {
            key_bits,
            sigma,
            scale: Rational::new(num, den).unwrap(),
        };
        let table = Table::generate(TableId(5), kind).unwrap();
        assert_eq!(
            table.pieces(),
            vec![expected],
            "σ = {sigma}, scale = {num}/{den}, {rows} rows"
        );
        for k in expected..rows {
            assert_eq!(table.value(k), Some(0), "key {k} in the tail");
        }
        // Key 0 is exactly 2^σ, and never in the tail.
        assert_eq!(table.value(0), Some(1i64 << sigma));
    }
}

/// A `Silu` row's value depends on its class and `h = k >> r_c` only, and `h = 0` gives 0.
#[test]
fn silu_rows_depend_on_the_shifted_key_alone() {
    let kind = TableKind::Silu {
        b: 9,
        r_by_class: vec![0, 3, 8, 5],
        in_exp: -4,
        out_exp: -3,
    };
    let table = Table::generate(TableId(6), kind).unwrap();
    for (class, r) in [0u32, 3, 8, 5].into_iter().enumerate() {
        let base = (class as u64) << 9;
        for x in 0..512u64 {
            let h_start = x & !((1 << r) - 1);
            assert_eq!(table.value(base + x), table.value(base + h_start));
        }
        // k = 0 is the offset key 2^8, whose h is 0.
        assert_eq!(table.value(base + 256), Some(0));
    }
}
