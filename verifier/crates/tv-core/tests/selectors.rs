// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The public factors of the stage checks against their definitions, summed over the hypercube
//! term by term: `LT` (RFC-0010 'Multilinear extensions'), the causal selector `C̃` (RFC-0030
//! 'The causal selector'), the row selector `ψ̃` (RFC-0035 'The row selector'), a stacked
//! instance's layer weights `w̃` (RFC-0037 'Stages and their instances'), and the extension of
//! `Rotate`'s kernel `E_r` (RFC-0037 'Rotate'), whose product with `X̃` must also equal RoPE as
//! RFC-0005 states it, applied entry by entry.
//!
//! The sums here take no shortcut: each is the definition, `Σ_x eq(r, x)·f(x)` with `eq` a
//! product over the coordinates, so they share no algorithm with the closed forms they check.

use tv_core::selectors::{causal, layer_weights, lt, rotate_kernel, row_selector, suffix_kernel};
use tv_field::{Fp, Fp2};
use tv_tables::rope::Rope;
use tv_tables::{Table, TableId, TableKind};
use tv_testkit::SplitMix64;

fn random(rng: &mut SplitMix64) -> Fp2 {
    Fp2::new(Fp::reduce(rng.next_u64()), Fp::reduce(rng.next_u64()))
}

fn point(rng: &mut SplitMix64, len: usize) -> Vec<Fp2> {
    (0..len).map(|_| random(rng)).collect()
}

/// `eq(r, x)` for the index `x`, bit `i` of `x` read by coordinate `i`.
fn eq_at(r: &[Fp2], x: u64) -> Fp2 {
    r.iter().enumerate().fold(Fp2::ONE, |product, (i, &c)| {
        product * if (x >> i) & 1 == 1 { c } else { Fp2::ONE - c }
    })
}

/// A signed integer as a field element, `P − |v|` below zero.
fn signed(v: i64) -> Fp2 {
    let magnitude = Fp2::from(Fp::reduce(v.unsigned_abs()));
    if v < 0 { -magnitude } else { magnitude }
}

#[test]
fn the_suffix_kernel_sums_strict_pairs_and_keeps_the_two_points_ordered() {
    let mut rng = SplitMix64::new(0x124_5aff);
    for vars in 0..=6 {
        for _ in 0..5 {
            let (claim, at) = (point(&mut rng, vars), point(&mut rng, vars));
            let expected: Fp2 = (0..1u64 << vars)
                .map(|v| (0..v).fold(Fp2::ZERO, |sum, i| sum + eq_at(&claim, i)) * eq_at(&at, v))
                .fold(Fp2::ZERO, |sum, value| sum + value);
            assert_eq!(suffix_kernel(&claim, &at), Some(expected));
        }
    }
    let bits = |v: u64| -> Vec<_> {
        (0..4)
            .map(|i| {
                if (v >> i) & 1 == 0 {
                    Fp2::ZERO
                } else {
                    Fp2::ONE
                }
            })
            .collect()
    };
    for i in 0..16 {
        for v in 0..16 {
            assert_eq!(
                suffix_kernel(&bits(i), &bits(v)),
                Some(if i < v { Fp2::ONE } else { Fp2::ZERO }),
                "{i} < {v}"
            );
        }
    }
    assert_eq!(suffix_kernel(&[], &[]), Some(Fp2::ZERO));
    assert_eq!(suffix_kernel(&[Fp2::ZERO], &[]), None);
    assert_eq!(suffix_kernel(&[], &[Fp2::ONE]), None);
    // No cube-size shift or enumeration: the factor remains linear above 64 variables.
    let mut at = vec![Fp2::ZERO; 129];
    at[128] = Fp2::ONE;
    assert_eq!(suffix_kernel(&vec![Fp2::ZERO; 129], &at), Some(Fp2::ONE));
    assert_eq!(suffix_kernel(&at, &vec![Fp2::ZERO; 129]), Some(Fp2::ZERO));
}

#[test]
fn lt_is_the_sum_of_eq_below_its_bound() {
    let mut rng = SplitMix64::new(0x4c54);
    for n in 0..=6usize {
        for _ in 0..3 {
            let r = point(&mut rng, n);
            for a in 0..=(1u64 << n) {
                let expected = (0..a).fold(Fp2::ZERO, |sum, h| sum + eq_at(&r, h));
                assert_eq!(lt(&r, a), Some(expected), "n = {n}, a = {a}");
            }
            assert_eq!(lt(&r, (1u64 << n) + 1), None, "a bound past the cube");
        }
    }
    // At a Boolean point, LT is the comparison itself.
    for x in 0..16u64 {
        let r: Vec<Fp2> = (0..4)
            .map(|i| {
                if (x >> i) & 1 == 1 {
                    Fp2::ONE
                } else {
                    Fp2::ZERO
                }
            })
            .collect();
        for a in 0..=16 {
            let expected = if x < a { Fp2::ONE } else { Fp2::ZERO };
            assert_eq!(lt(&r, a), Some(expected), "x = {x}, a = {a}");
        }
    }
}

#[test]
fn the_causal_selector_is_the_sum_over_valid_pairs() {
    let mut rng = SplitMix64::new(0xc0_5a1);
    for m in 0..=4usize {
        for length in 0..=(1u64 << m) {
            let (q, i) = (point(&mut rng, m), point(&mut rng, m));
            let mut expected = Fp2::ZERO;
            for qq in 0..length {
                for ii in 0..=qq {
                    expected += eq_at(&q, qq) * eq_at(&i, ii);
                }
            }
            assert_eq!(
                causal(&q, &i, length),
                Some(expected),
                "m = {m}, n = {length}"
            );
        }
        let (q, i) = (point(&mut rng, m), point(&mut rng, m));
        assert_eq!(
            causal(&q, &i, (1u64 << m) + 1),
            None,
            "a length past the axis"
        );
    }
    // Points of different lengths name no pair of positions.
    let mut rng = SplitMix64::new(9);
    assert_eq!(causal(&point(&mut rng, 2), &point(&mut rng, 3), 3), None);
    // At Boolean points it is [i ≤ q < n].
    for n in 0..=8u64 {
        for q in 0..8u64 {
            for i in 0..8u64 {
                let bits = |x: u64| -> Vec<Fp2> {
                    (0..3)
                        .map(|b| {
                            if (x >> b) & 1 == 1 {
                                Fp2::ONE
                            } else {
                                Fp2::ZERO
                            }
                        })
                        .collect()
                };
                let expected = if i <= q && q < n { Fp2::ONE } else { Fp2::ZERO };
                assert_eq!(causal(&bits(q), &bits(i), n), Some(expected), "{q} {i} {n}");
            }
        }
    }
}

#[test]
fn the_row_selector_is_the_sum_over_real_rows() {
    let mut rng = SplitMix64::new(0x5e1);
    // A tensor of rows [layers, heads, tokens], the tokens lowest: 2, 1 and 3 variables.
    let (layer_vars, head_vars, row_vars) = (2usize, 1usize, 3usize);
    for (layers, heads, rows) in [(3u64, 2u64, 5u64), (4, 1, 8), (1, 2, 1), (2, 1, 0)] {
        let r = point(&mut rng, row_vars + head_vars + layer_vars);
        let (t, rest) = r.split_at(row_vars);
        let (h, l) = rest.split_at(head_vars);
        let mut expected = Fp2::ZERO;
        for x in 0..1u64 << r.len() {
            let token = x & 0b111;
            let head = (x >> 3) & 1;
            let layer = x >> 4;
            if token < rows && head < heads && layer < layers {
                expected += eq_at(&r, x);
            }
        }
        assert_eq!(
            row_selector(t, rows, Some((h, heads)), Some((l, layers))),
            Some(expected),
            "{layers} layers, {heads} heads, {rows} rows"
        );
        // Without heads and layers it is LT over the rows alone.
        assert_eq!(row_selector(t, rows, None, None), lt(t, rows));
    }
    let r = point(&mut rng, 2);
    assert_eq!(
        row_selector(&r, 5, None, None),
        None,
        "5 rows on 2 variables"
    );
}

#[test]
fn layer_weights_are_the_extension_of_the_weights() {
    let mut rng = SplitMix64::new(0x1a7e);
    for vars in 0..=5usize {
        let weights: Vec<Fp2> = (0..1usize << vars).map(|_| random(&mut rng)).collect();
        let at = point(&mut rng, vars);
        let expected = weights
            .iter()
            .enumerate()
            .fold(Fp2::ZERO, |sum, (l, &w)| sum + eq_at(&at, l as u64) * w);
        assert_eq!(layer_weights(&weights, &at), Some(expected), "λ = {vars}");
        // eq(ρ, ·) as weights over the whole cube gives eq(ρ, ρ').
        let rho = point(&mut rng, vars);
        let eq_weights: Vec<Fp2> = (0..1u64 << vars).map(|l| eq_at(&rho, l)).collect();
        let direct = rho.iter().zip(&at).fold(Fp2::ONE, |p, (&a, &b)| {
            p * (a * b + (Fp2::ONE - a) * (Fp2::ONE - b))
        });
        assert_eq!(layer_weights(&eq_weights, &at), Some(direct));
    }
    let mut rng = SplitMix64::new(3);
    assert_eq!(layer_weights(&[Fp2::ONE; 3], &point(&mut rng, 2)), None);
}

/// A small Llama-3 RoPE table: `θ = 500,000`, factor 8, bands 1 and 4 against 8,192, 14 bits.
fn rope(max_context: u32, d_head: u16) -> Rope {
    let kind = TableKind::Rope {
        theta: 500_000,
        factor: 8,
        low_freq_factor: 1,
        high_freq_factor: 4,
        original_max: 8192,
        bits: 14,
        max_context,
        d_head,
    };
    let table = Table::generate(TableId(0), kind).expect("a table RFC-0008 allows");
    table.rope().expect("a Rope table").clone()
}

/// `E_r(x)` over every entry `x` of a tensor `[positions, heads, d_head]` padded per dimension,
/// as RFC-0037 'Rotate' defines it: `eq(r_{−b}, x_{−b})·(C·eq(r_b, x_b) + S·(r_b − x_b))`, with
/// `C` and `S` RoPE's constants at the row's position and the pair's frequency, 0 on a padded
/// row.
fn kernel_table(
    rope: &Rope,
    positions: u64,
    (pos_vars, head_vars, dh_vars): (usize, usize, usize),
    r: &[Fp2],
) -> Vec<Fp2> {
    let b = dh_vars - 1;
    let total = pos_vars + head_vars + dh_vars;
    (0..1u64 << total)
        .map(|x| {
            let frequency = x & ((1 << b) - 1);
            let bit = (x >> b) & 1;
            let position = x >> (dh_vars + head_vars);
            let (c, s) = if position < positions {
                let (c, s) = rope.pair(position as u32, frequency as usize).unwrap();
                (signed(c), signed(s))
            } else {
                (Fp2::ZERO, Fp2::ZERO)
            };
            // eq over every coordinate but b.
            let mut rest = Fp2::ONE;
            for (i, &ri) in r.iter().enumerate() {
                if i != b {
                    rest *= if (x >> i) & 1 == 1 { ri } else { Fp2::ONE - ri };
                }
            }
            let rb = r[b];
            let xb = if bit == 1 { Fp2::ONE } else { Fp2::ZERO };
            let eq_b = rb * xb + (Fp2::ONE - rb) * (Fp2::ONE - xb);
            rest * (c * eq_b + s * (rb - xb))
        })
        .collect()
}

#[test]
fn the_rotate_kernel_is_the_extension_of_its_table() {
    let mut rng = SplitMix64::new(0x0b07);
    // d_head 8 (b = 2), heads on 0 and 1 variables, 3 to 6 positions on 3 variables.
    let table = rope(8, 8);
    for (positions, head_vars) in [(5u64, 1usize), (8, 0), (3, 1), (6, 2)] {
        let vars = (3usize, head_vars, 3usize);
        let total = vars.0 + vars.1 + vars.2;
        let r = point(&mut rng, total);
        let at = point(&mut rng, total);
        let kernel = kernel_table(&table, positions, vars, &r);
        let expected = kernel
            .iter()
            .enumerate()
            .fold(Fp2::ZERO, |sum, (x, &e)| sum + eq_at(&at, x as u64) * e);
        assert_eq!(
            rotate_kernel(&table, positions as u32, head_vars, &r, &at),
            Some(expected),
            "{positions} positions, {head_vars} head variables"
        );
    }
}

/// `Ỹ(r) = Σ_x E_r(x)·X̃(x)` is RoPE's output: on a random integer tensor, the kernel's table
/// against `X` equals the extension at `r` of RFC-0005's rotation applied pair by pair, with
/// `(a, b) ↦ (a·C − b·S, a·S + b·C)`, the pair `(x_i, x_{i + d_head/2})`.
#[test]
fn the_kernel_applies_rope_as_rfc_0005_states_it() {
    let mut rng = SplitMix64::new(0x40e);
    let table = rope(16, 16);
    let (positions, heads, d_head) = (5usize, 2usize, 16usize);
    let (pos_vars, head_vars, dh_vars) = (3usize, 1usize, 4usize);
    let half = d_head / 2;
    let mut x = vec![0i64; 1 << (pos_vars + head_vars + dh_vars)];
    for p in 0..positions {
        for h in 0..heads {
            for j in 0..d_head {
                x[(p << (head_vars + dh_vars)) | (h << dh_vars) | j] =
                    (rng.next_u64() % 255) as i64 - 127;
            }
        }
    }
    let mut y = vec![0i64; x.len()];
    for p in 0..positions {
        for h in 0..heads {
            let base = (p << (head_vars + dh_vars)) | (h << dh_vars);
            for i in 0..half {
                let (c, s) = table.pair(p as u32, i).unwrap();
                let (a, b) = (x[base + i], x[base + i + half]);
                y[base + i] = a * c - b * s;
                y[base + i + half] = a * s + b * c;
            }
        }
    }
    let r = point(&mut rng, pos_vars + head_vars + dh_vars);
    let kernel = kernel_table(&table, positions as u64, (pos_vars, head_vars, dh_vars), &r);
    let through_kernel = kernel
        .iter()
        .zip(&x)
        .fold(Fp2::ZERO, |sum, (&e, &v)| sum + e * signed(v));
    let direct = y.iter().enumerate().fold(Fp2::ZERO, |sum, (i, &v)| {
        sum + eq_at(&r, i as u64) * signed(v)
    });
    assert_eq!(through_kernel, direct);
}

#[test]
fn the_rotate_kernel_refuses_what_it_cannot_evaluate() {
    let mut rng = SplitMix64::new(4);
    let table = rope(8, 8);
    let r = point(&mut rng, 7);
    // More positions than the table has, or than the position variables hold.
    assert_eq!(rotate_kernel(&table, 9, 1, &r, &r), None);
    assert_eq!(rotate_kernel(&table, 9, 2, &r[..6], &r[..6]), None);
    // Points of different lengths, or too short for the head and pair coordinates.
    assert_eq!(rotate_kernel(&table, 5, 1, &r, &r[..6]), None);
    assert_eq!(rotate_kernel(&table, 1, 5, &r[..3], &r[..3]), None);
    // d_head / 2 must be a power of two for the pair to differ in one bit.
    let six = rope(4, 12);
    let r = point(&mut rng, 6);
    assert_eq!(rotate_kernel(&six, 2, 0, &r, &r), None);
}
