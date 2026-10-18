// SPDX-License-Identifier: Apache-2.0 OR MIT

//! [`exp_maps_to_zero`], the value test the lowering's `Exp` width search reads (RFC-0044 'The
//! tables', RFC-0030 'Relation 1'): whether an `Exp` table maps one key to 0, without generating
//! the table. It must agree with the generated tables of `spec/vectors/tables/` key for key,
//! with seeded tables of every `σ` RFC-0008 allows, and, past that, with RFC-0008's formula
//! evaluated directly here, since the lowering derives definitions that validation refuses only
//! later (RFC-0007 step 5.3).

mod support;

use support::{kind, list, member, text};
use tv_codec::Rational;
use tv_tables::{Table, TableError, TableId, TableKind, exp_maps_to_zero, fixed};
use tv_testkit::{SplitMix64, read_vectors};

/// The `Exp` parameters of a vector case's kind.
fn exp_parameters(kind: &TableKind) -> Option<(u8, u8, Rational)> {
    match kind {
        TableKind::Exp {
            key_bits,
            sigma,
            scale,
        } => Some((*key_bits, *sigma, *scale)),
        _ => None,
    }
}

/// RFC-0008's `Exp` value at `key`, evaluated as the RFC writes it, for `σ ≤ 63`, where every
/// intermediate stays below `2^128`: `k·num = q·den + m`; 0 when `q ≥ 2^20`; else
/// `x = q·2^64 + ⌊m·2^64 / den⌋`, `(y, n) = exp_neg(x)`, 0 when `n ≥ σ + 2`, and
/// `(y·2^σ + 2^{63+n}) >> (64 + n)` otherwise.
fn formula(key: u64, sigma: u8, scale: Rational) -> u128 {
    assert!(sigma <= 63, "the formula needs σ ≤ 63 to stay below 2^128");
    let num = u128::try_from(scale.num()).expect("a positive scale");
    let den = u128::from(scale.den());
    let product = u128::from(key) * num;
    let (q, m) = (product / den, product % den);
    if q >= 1 << 20 {
        return 0;
    }
    let x = (q << 64) + ((m << 64) / den);
    let (y, n) = fixed::exp_neg(x).expect("exp_neg is total below 2^84");
    if n >= u128::from(sigma) + 2 {
        return 0;
    }
    let n = u32::try_from(n).expect("n ≤ σ + 1");
    ((y << sigma) + (1u128 << (63 + n))) >> (64 + n)
}

/// A seeded positive scale `num / den` in lowest terms, of magnitude near `2^-shift`.
fn scale(rng: &mut SplitMix64, shift: u32) -> Rational {
    loop {
        let num = i64::try_from(rng.next_u64() % 64 + 1).expect("small");
        let den = (rng.next_u64() % 1024 + 1) << shift.min(50);
        if let Some(scale) = Rational::new(num, den) {
            return scale;
        }
    }
}

#[test]
fn every_vector_table_agrees_key_for_key() {
    let mut keys = 0u64;
    for case in list(&read_vectors("tables/tables.json"), "cases") {
        let kind = kind(member(case, "kind"));
        let Some((key_bits, sigma, scale)) = exp_parameters(&kind) else {
            continue;
        };
        let name = text(case, "name");
        let table = Table::generate(TableId(0), kind).unwrap_or_else(|e| panic!("{name}: {e}"));
        let rows = table.rows();
        // Every key of the small tables; around the zero tail and at both ends of the large ones.
        let z = table.pieces()[0];
        let mut probe: Vec<u64> = if rows <= 1 << 12 {
            (0..rows).collect()
        } else {
            let mut around = vec![0, 1, rows - 2, rows - 1];
            around.extend(z.saturating_sub(64)..(z + 64).min(rows));
            around
        };
        probe.sort_unstable();
        probe.dedup();
        for key in probe {
            let zero = exp_maps_to_zero(key, sigma, scale).unwrap();
            assert_eq!(
                zero,
                table.value(key) == Some(0),
                "{name} (key_bits {key_bits}): key {key}"
            );
            keys += 1;
        }
    }
    assert!(keys > 1_000, "{keys} keys");
}

#[test]
fn seeded_tables_of_every_allowed_sigma_agree_key_for_key() {
    let mut rng = SplitMix64::new(0x0107_e7a0);
    let mut zeros = 0u64;
    let mut values = 0u64;
    for round in 0..200u32 {
        let sigma = u8::try_from(round % 33).expect("σ ≤ 32");
        let key_bits = u8::try_from(rng.next_u64() % 12 + 1).expect("small");
        let scale = scale(&mut rng, round % 13);
        let kind = TableKind::Exp {
            key_bits,
            sigma,
            scale,
        };
        let table = Table::generate(TableId(0), kind).expect("an allowed table");
        for key in 0..table.rows() {
            let zero = exp_maps_to_zero(key, sigma, scale).unwrap();
            assert_eq!(
                zero,
                table.value(key) == Some(0),
                "σ {sigma}, scale {scale:?}, key {key}"
            );
            if zero {
                zeros += 1;
            } else {
                values += 1;
            }
        }
    }
    // Both answers are exercised, and so are keys that round to 0 before the zero tail.
    assert!(
        zeros > 10_000 && values > 10_000,
        "{zeros} zeros, {values} values"
    );
}

#[test]
fn a_value_that_rounds_to_zero_before_the_zero_tail_is_zero() {
    // The check reads the value, not the zero tail's first key (RFC-0030): find keys below `z`
    // whose value rounds to 0, which the generated table holds in its core.
    let mut found = 0;
    for den in 2..400u64 {
        let scale = Rational::new(1, den).expect("1/den is reduced");
        let kind = TableKind::Exp {
            key_bits: 14,
            sigma: 4,
            scale,
        };
        let table = Table::generate(TableId(0), kind).expect("an allowed table");
        let z = table.pieces()[0];
        for key in 0..z {
            if table.value(key) == Some(0) {
                assert!(
                    exp_maps_to_zero(key, 4, scale).unwrap(),
                    "1/{den}, key {key}"
                );
                found += 1;
            }
        }
    }
    assert!(found > 0, "some value rounds to 0 below its zero tail");
}

#[test]
fn sigmas_past_rfc_0008s_bound_follow_its_formula() {
    // The lowering evaluates the definitions it derives before validation refuses them, so the
    // test must be exact for every σ of a `u8`: up to 63, against the formula itself.
    let mut rng = SplitMix64::new(0x0107_5163);
    for round in 0..2_000u32 {
        let sigma = u8::try_from(33 + round % 31).expect("33 to 63");
        let scale = scale(&mut rng, round % 40);
        let key = match round % 4 {
            0 => rng.next_u64() % (1 << 12),
            1 => rng.next_u64() % (1 << 32),
            2 => (1u64 << (round % 33)) - 1,
            _ => rng.next_u64(),
        };
        assert_eq!(
            exp_maps_to_zero(key, sigma, scale).unwrap(),
            formula(key, sigma, scale) == 0,
            "σ {sigma}, scale {scale:?}, key {key}"
        );
    }
}

#[test]
fn the_widest_sigmas_are_exact_at_their_edges() {
    let one = Rational::new(1, 1).expect("1/1");
    let tiny = Rational::new(1, 1 << 40).expect("2^-40");
    for sigma in [64u8, 100, 200, 255] {
        // Key 0 maps to 2^σ, never 0.
        assert!(
            !exp_maps_to_zero(0, sigma, one).unwrap(),
            "σ {sigma}: key 0"
        );
        // A key whose `⌊k · scale⌋` reaches 2^20 lies in the zero tail by its first rule.
        assert!(exp_maps_to_zero(1 << 20, sigma, one).unwrap(), "σ {sigma}");
        assert!(exp_maps_to_zero(u64::MAX, sigma, one).unwrap(), "σ {sigma}");
        // A key whose exponent `n` stays below σ maps to a value of about 2^{σ − n}.
        assert!(
            !exp_maps_to_zero((1 << 32) - 1, sigma, tiny).unwrap(),
            "σ {sigma}"
        );
    }
    // The exponent rule: σ = 64 and n = 66 (`x` just past 66·LN2) is 0, n = 63 is not.
    let n_of = |key: u64| {
        let x = u128::from(key) << 64;
        fixed::exp_neg(x).expect("in range").1
    };
    assert_eq!(n_of(46), 66, "46 ≥ 66·ln 2 = 45.7");
    assert!(exp_maps_to_zero(46, 64, one).unwrap());
    assert_eq!(n_of(44), 63);
    assert!(!exp_maps_to_zero(44, 64, one).unwrap());
}

#[test]
fn a_scale_that_is_not_positive_is_refused() {
    for (num, den) in [(0, 1), (-1, 3), (i64::MIN, 1)] {
        let scale = Rational::new(num, den).expect("reduced");
        assert_eq!(
            exp_maps_to_zero(5, 16, scale),
            Err(TableError::ExpScale),
            "{num}/{den}"
        );
    }
}
