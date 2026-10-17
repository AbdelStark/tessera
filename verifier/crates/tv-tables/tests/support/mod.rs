// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Test support shared by the integration tests: readers for the table vectors, a seeded
//! generator of field elements, and a brute-force oracle of a table's extension.
//!
//! The oracle sums `eq(ρ, j)·(key_j + β·value_j)` over every padded row, each `eq` a product of
//! its factors, as RFC-0010 'Multilinear extensions' defines it: it shares no step with the
//! crate's structured evaluation, which never enumerates rows.

#![allow(dead_code)]

use tv_codec::Rational;
use tv_field::{Fp, Fp2};
use tv_tables::{BitOp, Table, TableKind};
use tv_testkit::{Json, SplitMix64, read_vectors};

// ----- Vector formats ---------------------------------------------------------------------------

/// Every case of `tables/tables.json`.
pub fn table_cases() -> Vec<Json> {
    list(&read_vectors("tables/tables.json"), "cases").to_vec()
}

/// The case of `tables/tables.json` named `name`.
pub fn table_case(name: &str) -> Json {
    table_cases()
        .into_iter()
        .find(|case| text(case, "name") == name)
        .unwrap_or_else(|| panic!("no case is named {name}"))
}

/// The member `key` of a vector object.
pub fn member<'a>(object: &'a Json, key: &str) -> &'a Json {
    object
        .get(key)
        .unwrap_or_else(|| panic!("`{key}` is missing"))
}

/// The member `key` of a vector object, which must be a string.
pub fn text<'a>(object: &'a Json, key: &str) -> &'a str {
    member(object, key)
        .as_str()
        .unwrap_or_else(|| panic!("`{key}` is not a string"))
}

/// The member `key` of a vector object, which must be an array.
pub fn list<'a>(object: &'a Json, key: &str) -> &'a [Json] {
    member(object, key)
        .as_array()
        .unwrap_or_else(|| panic!("`{key}` is not an array"))
}

/// A JSON integer of any sign that fits `i128`.
pub fn integer(value: &Json) -> i128 {
    match value {
        Json::Number(n) => n.parse().unwrap_or_else(|e| panic!("{n}: {e}")),
        other => panic!("{other:?} is not an integer"),
    }
}

/// The member `key` of a vector object, which must be an integer.
pub fn number(object: &Json, key: &str) -> i128 {
    integer(member(object, key))
}

/// A `"0x…"` string of any width up to 128 bits.
pub fn hex_u128(value: &Json) -> u128 {
    let text = value
        .as_str()
        .and_then(|s| s.strip_prefix("0x"))
        .unwrap_or_else(|| panic!("{value:?} is not a hex string"));
    u128::from_str_radix(text, 16).unwrap_or_else(|e| panic!("{text}: {e}"))
}

/// A 32-byte digest from hexadecimal.
pub fn digest_hex(text: &str) -> [u8; 32] {
    tv_testkit::hex(text)
        .try_into()
        .unwrap_or_else(|_| panic!("{text} is not 32 bytes"))
}

/// A vector's `kind` object, such as `{"Range": {"bits": 8}}`, as a [`TableKind`].
pub fn kind(json: &Json) -> TableKind {
    let Json::Object(members) = json else {
        panic!("{json:?} is not a kind");
    };
    let [(name, fields)] = members.as_slice() else {
        panic!("{json:?} names more than one kind");
    };
    let u8_of = |key: &str| u8::try_from(number(fields, key)).unwrap();
    let shifts = |key: &str| -> Vec<u8> {
        list(fields, key)
            .iter()
            .map(|s| u8::try_from(integer(s)).unwrap())
            .collect()
    };
    match name.as_str() {
        "Range" => TableKind::Range {
            bits: u8_of("bits"),
        },
        "Requant" => TableKind::Requant {
            b: u8_of("b"),
            r_by_class: shifts("r_by_class"),
            lo: i32::try_from(number(fields, "lo")).unwrap(),
            hi: i32::try_from(number(fields, "hi")).unwrap(),
        },
        "Pow2" => TableKind::Pow2 {
            max_exp: u8_of("max_exp"),
        },
        "Bitwise8" => TableKind::Bitwise8 {
            op: match text(fields, "op") {
                "Xor" => BitOp::Xor,
                "And" => BitOp::And,
                other => panic!("unknown operation {other}"),
            },
        },
        "AesSbox" => TableKind::AesSbox,
        "Gf256Mul" => TableKind::Gf256Mul { c: u8_of("c") },
        "Exp" => {
            let scale = list(fields, "scale");
            let [num, den] = scale else {
                panic!("a scale is [num, den]");
            };
            TableKind::Exp {
                key_bits: u8_of("key_bits"),
                sigma: u8_of("sigma"),
                scale: Rational::new(
                    i64::try_from(integer(num)).unwrap(),
                    u64::try_from(integer(den)).unwrap(),
                )
                .unwrap(),
            }
        }
        "Exp2Frac" => TableKind::Exp2Frac {
            frac_bits: u8_of("frac_bits"),
            out_bits: u8_of("out_bits"),
        },
        "Silu" => TableKind::Silu {
            b: u8_of("b"),
            r_by_class: shifts("r_by_class"),
            in_exp: i8::try_from(number(fields, "in_exp")).unwrap(),
            out_exp: i8::try_from(number(fields, "out_exp")).unwrap(),
        },
        "Rope" => TableKind::Rope {
            theta: u32::try_from(number(fields, "theta")).unwrap(),
            factor: u16::try_from(number(fields, "factor")).unwrap(),
            low_freq_factor: u16::try_from(number(fields, "low_freq_factor")).unwrap(),
            high_freq_factor: u16::try_from(number(fields, "high_freq_factor")).unwrap(),
            original_max: u32::try_from(number(fields, "original_max")).unwrap(),
            bits: u8_of("bits"),
            max_context: u32::try_from(number(fields, "max_context")).unwrap(),
            d_head: u16::try_from(number(fields, "d_head")).unwrap(),
        },
        other => panic!("unknown kind {other}"),
    }
}

// ----- Field elements ---------------------------------------------------------------------------

/// The true integer `v` in `Fp`, centered: `P − |v|` for negative `v` (RFC-0002), computed
/// with `u128` arithmetic rather than the field's negation.
pub fn centered(v: i128) -> Fp {
    let p = u128::from(Fp::MODULUS);
    let magnitude = v.unsigned_abs();
    assert!(magnitude < p / 2, "{v} is outside the centered range");
    let canonical = if v < 0 { p - magnitude } else { magnitude };
    Fp::from_canonical(u64::try_from(canonical).unwrap()).unwrap()
}

/// Seeded field elements and points.
pub struct Rng(SplitMix64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(SplitMix64::new(seed))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }

    /// A value below `n > 0`.
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    pub fn fp(&mut self) -> Fp {
        Fp::reduce(self.next_u64())
    }

    pub fn fp2(&mut self) -> Fp2 {
        Fp2::new(self.fp(), self.fp())
    }

    /// A coordinate: mostly random, sometimes 0 or 1, where an algorithm that divided by `ρ_i`
    /// or `1 − ρ_i` would fail.
    pub fn coordinate(&mut self) -> Fp2 {
        match self.below(8) {
            0 => Fp2::ZERO,
            1 => Fp2::ONE,
            _ => self.fp2(),
        }
    }

    /// A point of `n` coordinates from [`coordinate`](Self::coordinate).
    pub fn point(&mut self, n: usize) -> Vec<Fp2> {
        (0..n).map(|_| self.coordinate()).collect()
    }
}

/// The Boolean point of index `j` over `n` variables, the lowest first.
pub fn boolean_point(j: u64, n: usize) -> Vec<Fp2> {
    (0..n)
        .map(|i| {
            if (j >> i) & 1 == 1 {
                Fp2::ONE
            } else {
                Fp2::ZERO
            }
        })
        .collect()
}

/// `eq(ρ, j) = Π_i (ρ_i·j_i + (1 − ρ_i)·(1 − j_i))`.
pub fn eq(rho: &[Fp2], j: u64) -> Fp2 {
    rho.iter().enumerate().fold(Fp2::ONE, |product, (i, &r)| {
        let bit = Fp2::from(Fp::reduce((j >> i) & 1));
        product * (r * bit + (Fp2::ONE - r) * (Fp2::ONE - bit))
    })
}

/// `Σ_{j < 2^{log2_rows}} eq(ρ, j)·(key_j + β·value_j)` over the table's padded rows, read with
/// [`Table::row`].
pub fn brute_force(table: &Table, rho: &[Fp2], beta: Fp2) -> Fp2 {
    let padded = 1u64 << table.log2_rows();
    assert!(table.row(padded).is_none(), "no row past the padded rows");
    (0..padded).fold(Fp2::ZERO, |sum, j| {
        let (key, value) = table.row(j).unwrap();
        sum + eq(rho, j) * (Fp2::from(key) + beta * value)
    })
}
