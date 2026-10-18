// SPDX-License-Identifier: Apache-2.0 OR MIT

//! [`TableKind::check_parameters`], RFC-0008's 'Parameters allowed' without generating the
//! table: what claim-list validation reads to know that RFC-0008 allows the `Exp` table a score
//! scale needs (RFC-0007 step 2). It must give what [`Table::generate`] gives, `⌈log2 T⌉` or the
//! same first error, for every kind of the vectors, for kinds that break each rule, and for
//! seeded `Exp` kinds on both sides of the zero tail's bound.

mod support;

use support::{kind, list, member, text};
use tv_codec::Rational;
use tv_tables::{BitOp, Table, TableError, TableId, TableKind};
use tv_testkit::{SplitMix64, read_vectors};

/// What `Table::generate` decides: `⌈log2 T⌉`, or its error.
fn generated(kind: &TableKind) -> Result<u8, TableError> {
    Table::generate(TableId(0), kind.clone()).map(|table| table.log2_rows())
}

fn ratio(num: i64, den: u64) -> Rational {
    Rational::new(num, den).expect("in lowest terms")
}

#[test]
fn every_vector_kind_is_allowed_with_its_row_count() {
    let file = read_vectors("tables/tables.json");
    let cases = list(&file, "cases");
    assert!(!cases.is_empty());
    for case in cases {
        let kind = kind(member(case, "kind"));
        let log2_rows = member(case, "log2_rows").as_u64().expect("log2_rows");
        assert_eq!(
            kind.check_parameters(),
            Ok(u8::try_from(log2_rows).unwrap()),
            "{}",
            text(case, "name")
        );
    }
}

#[test]
fn a_kind_that_breaks_a_rule_gets_the_error_generation_gives() {
    let refused = [
        TableKind::Range { bits: 0 },
        TableKind::Range { bits: 33 },
        TableKind::Requant {
            b: 0,
            r_by_class: vec![1],
            lo: -1,
            hi: 1,
        },
        TableKind::Requant {
            b: 8,
            r_by_class: vec![1, 2, 3, 4, 5],
            lo: -1,
            hi: 1,
        },
        TableKind::Requant {
            b: 8,
            r_by_class: vec![64],
            lo: -1,
            hi: 1,
        },
        TableKind::Requant {
            b: 8,
            r_by_class: vec![3],
            lo: 2,
            hi: 1,
        },
        TableKind::Pow2 { max_exp: 64 },
        TableKind::Exp {
            key_bits: 33,
            sigma: 16,
            scale: ratio(1, 2),
        },
        TableKind::Exp {
            key_bits: 8,
            sigma: 33,
            scale: ratio(1, 2),
        },
        TableKind::Exp {
            key_bits: 8,
            sigma: 16,
            scale: ratio(0, 1),
        },
        // A zero tail that starts past key 2^24: the table widened to 25 bits at 2^-21.
        TableKind::Exp {
            key_bits: 25,
            sigma: 16,
            scale: ratio(1, 1 << 21),
        },
        TableKind::Exp2Frac {
            frac_bits: 17,
            out_bits: 8,
        },
        TableKind::Silu {
            b: 8,
            r_by_class: vec![8],
            in_exp: -4,
            out_exp: -3,
        },
        TableKind::Silu {
            b: 8,
            r_by_class: vec![2],
            in_exp: 1,
            out_exp: -3,
        },
        TableKind::Rope {
            theta: 1,
            factor: 8,
            low_freq_factor: 1,
            high_freq_factor: 4,
            original_max: 8192,
            bits: 14,
            max_context: 64,
            d_head: 16,
        },
        TableKind::Rope {
            theta: 500_000,
            factor: 8,
            low_freq_factor: 1,
            high_freq_factor: 4,
            original_max: 8192,
            bits: 14,
            max_context: 1 << 17,
            d_head: 512,
        },
    ];
    for kind in &refused {
        let checked = kind.check_parameters();
        assert!(checked.is_err(), "{kind:?} is refused");
        assert_eq!(checked, generated(kind), "{kind:?}");
    }
    // Kinds of no parameter are allowed.
    for kind in [
        TableKind::AesSbox,
        TableKind::Gf256Mul { c: 3 },
        TableKind::Bitwise8 { op: BitOp::Xor },
    ] {
        assert_eq!(kind.check_parameters(), generated(&kind), "{kind:?}");
    }
}

#[test]
fn seeded_exp_kinds_on_both_sides_of_the_zero_tails_bound_agree() {
    // Scales near 2^-k for key widths up to 26 put the zero tail on both sides of key 2^24;
    // generation is cheap for the refused ones, which stop before any core, and for the allowed
    // ones whose tail starts early.
    let mut rng = SplitMix64::new(0x0107_c4ec);
    let (mut allowed, mut refused) = (0, 0);
    for round in 0..400u32 {
        let key_bits = u8::try_from(22 + round % 5).unwrap();
        let sigma = u8::try_from(rng.next_u64() % 33).unwrap();
        // Half the scales put the zero tail early, half late: `1/den` is always in lowest terms.
        let shift = if round % 2 == 0 {
            rng.next_u64() % 8
        } else {
            16 + rng.next_u64() % 12
        };
        let den = (2 * (rng.next_u64() % 32) + 1) << shift;
        let scale = ratio(1, den);
        let kind = TableKind::Exp {
            key_bits,
            sigma,
            scale,
        };
        let checked = kind.check_parameters();
        // Generation stops at its checks when they refuse, and is cheap when the zero tail starts
        // before key 2^14, which the 14-bit table of the same scale shows: compare both there.
        let early_tail = Table::generate(
            TableId(0),
            TableKind::Exp {
                key_bits: 14,
                sigma,
                scale,
            },
        )
        .is_ok_and(|t| t.pieces()[0] < 1 << 14);
        if checked.is_err() {
            assert_eq!(checked, generated(&kind), "{kind:?}");
            refused += 1;
        } else if early_tail {
            assert_eq!(checked, generated(&kind), "{kind:?}");
            allowed += 1;
        }
    }
    // Both answers are compared with generation.
    assert!(
        allowed > 30 && refused > 30,
        "{allowed} allowed, {refused} refused"
    );
}
