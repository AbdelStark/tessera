// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Every rule of RFC-0008 'Table specifications' broken in turn, each with its exact error:
//! the parameters of every kind, in the order the crate documentation gives them, with the values
//! at each boundary accepted; a wrong `log2_rows`; another digest; and specifications that do
//! not decode, with their code and offset.

mod support;

use support::{digest_hex, table_case, table_cases, text};
use tv_codec::{DecodeError, DecodeLimits, ErrorCode, Rational, decode};
use tv_tables::fixed::LN2;
use tv_tables::{BitOp, Table, TableError, TableId, TableKind, TableSpec};
use tv_testkit::hex;

/// A specification of `kind` whose `log2_rows` no table has: when the parameters pass, the
/// check that follows them refuses it, so a parameter rule can be tested without generating.
fn with_impossible_rows(kind: TableKind) -> TableSpec {
    TableSpec {
        id: TableId(7),
        kind,
        log2_rows: 99,
        digest: [0; 32],
    }
}

/// `kind` breaks exactly the rule of `error`: both entry points refuse it with that error.
fn refused(kind: TableKind, error: TableError) {
    assert!(error.is_rejection(), "{error:?}");
    assert_eq!(
        Table::generate(TableId(7), kind.clone()).map(|t| t.spec().clone()),
        Err(error),
        "{kind:?}"
    );
    assert_eq!(
        Table::from_spec(&with_impossible_rows(kind.clone())).map(|t| t.spec().clone()),
        Err(error),
        "{kind:?}"
    );
}

/// `kind` passes every parameter rule, and has `2^log2_rows` padded rows.
fn accepted(kind: TableKind, log2_rows: u8) {
    assert_eq!(
        Table::from_spec(&with_impossible_rows(kind.clone())).map(|t| t.spec().clone()),
        Err(TableError::Log2Rows {
            expected: log2_rows,
            found: 99
        }),
        "{kind:?}"
    );
}

fn requant(b: u8, r_by_class: &[u8], lo: i32, hi: i32) -> TableKind {
    TableKind::Requant {
        b,
        r_by_class: r_by_class.to_vec(),
        lo,
        hi,
    }
}

fn silu(b: u8, r_by_class: &[u8], in_exp: i8, out_exp: i8) -> TableKind {
    TableKind::Silu {
        b,
        r_by_class: r_by_class.to_vec(),
        in_exp,
        out_exp,
    }
}

fn exp(key_bits: u8, sigma: u8, num: i64, den: u64) -> TableKind {
    TableKind::Exp {
        key_bits,
        sigma,
        scale: Rational::new(num, den).unwrap(),
    }
}

/// Llama-3.1's RoPE parameters with `max_context` and `d_head` given.
fn rope(max_context: u32, d_head: u16) -> TableKind {
    TableKind::Rope {
        theta: 500_000,
        factor: 8,
        low_freq_factor: 1,
        high_freq_factor: 4,
        original_max: 8192,
        bits: 14,
        max_context,
        d_head,
    }
}

/// `rope(16, 8)` with one parameter changed by `change`.
fn rope_with(change: impl FnOnce(&mut TableKind)) -> TableKind {
    let mut kind = rope(16, 8);
    change(&mut kind);
    kind
}

#[test]
fn range_bits() {
    for bits in [0, 33, 64, 255] {
        refused(TableKind::Range { bits }, TableError::RangeBits { bits });
    }
    accepted(TableKind::Range { bits: 1 }, 1);
    accepted(TableKind::Range { bits: 32 }, 32);
}

#[test]
fn requant_rules_in_order() {
    refused(requant(0, &[1], -1, 1), TableError::KeyBits);
    refused(requant(4, &[], -1, 1), TableError::Classes { classes: 0 });
    refused(
        requant(4, &[1, 1, 1, 1, 1], -1, 1),
        TableError::Classes { classes: 5 },
    );
    refused(
        requant(33, &[1], -1, 1),
        TableError::RowBits { b: 33, classes: 1 },
    );
    refused(
        requant(32, &[1, 1], -1, 1),
        TableError::RowBits { b: 32, classes: 2 },
    );
    refused(
        requant(31, &[1, 1, 1], -1, 1),
        TableError::RowBits { b: 31, classes: 3 },
    );
    refused(
        requant(4, &[1, 0], -1, 1),
        TableError::RequantShift { class: 1, shift: 0 },
    );
    refused(
        requant(4, &[1, 2, 64], -1, 1),
        TableError::RequantShift {
            class: 2,
            shift: 64,
        },
    );
    refused(
        requant(4, &[1], 2, 1),
        TableError::RequantClamp { lo: 2, hi: 1 },
    );
    // The first rule broken is the error.
    refused(requant(0, &[0, 0, 0, 0, 0], 2, 1), TableError::KeyBits);
    refused(
        requant(40, &[0, 0, 0, 0, 0], 2, 1),
        TableError::Classes { classes: 5 },
    );
    refused(
        requant(40, &[0], 2, 1),
        TableError::RowBits { b: 40, classes: 1 },
    );
    refused(
        requant(4, &[64, 0], 2, 1),
        TableError::RequantShift {
            class: 0,
            shift: 64,
        },
    );
    // Every boundary value is accepted.
    accepted(requant(1, &[1], -1, 1), 1);
    accepted(requant(32, &[63], i32::MIN, i32::MAX), 32);
    accepted(requant(31, &[1, 63], 0, 0), 32);
    accepted(requant(30, &[1, 2, 3, 4], 0, 0), 32);
    accepted(requant(5, &[1, 2, 3], -2, 3), 7);
}

#[test]
fn pow2_exponent() {
    for max_exp in [0, 64, 255] {
        refused(
            TableKind::Pow2 { max_exp },
            TableError::Pow2Exponent { max_exp },
        );
    }
    accepted(TableKind::Pow2 { max_exp: 1 }, 0);
    accepted(TableKind::Pow2 { max_exp: 63 }, 6);
}

#[test]
fn kinds_without_rules_accept_every_parameter() {
    accepted(TableKind::Bitwise8 { op: BitOp::Xor }, 16);
    accepted(TableKind::Bitwise8 { op: BitOp::And }, 16);
    accepted(TableKind::AesSbox, 8);
    for c in [0, 1, 0x57, 0xFF] {
        accepted(TableKind::Gf256Mul { c }, 8);
    }
}

/// A scale `num / 2^62` whose zero tail, at `σ = 0`, starts at key `z` exactly: then
/// `x = ⌊k · num · 2^64 / 2^62⌋ = 4·k·num`, and `z` is the least `k` with `x ≥ 2·LN2` by the
/// RFC's rule, so `num` is the least odd integer with `4·z·num ≥ 2·LN2`, checked here.
fn scale_with_tail_at(z: u64) -> (i64, u64) {
    let threshold = 2 * LN2;
    let x = |k: u64, num: u128| 4 * u128::from(k) * num;
    let num = threshold.div_ceil(4 * u128::from(z)) | 1;
    assert!(x(z - 1, num) < threshold && x(z, num) >= threshold);
    // ⌊k · scale⌋ stays far below 2^20 in a table of 2^26 keys: only the second rule applies.
    assert!(((1u128 << 26) * num) >> 62 < 1 << 20);
    (i64::try_from(num).unwrap(), 1 << 62)
}

#[test]
fn exp_rules_in_order() {
    for key_bits in [0, 33] {
        refused(exp(key_bits, 0, 1, 2), TableError::ExpKeyBits { key_bits });
    }
    refused(exp(8, 33, 1, 2), TableError::ExpSigma { sigma: 33 });
    refused(exp(8, 16, 0, 1), TableError::ExpScale);
    refused(exp(8, 16, -1, 4), TableError::ExpScale);
    // No tail before 2^25 keys at a scale of 2^-40: z = T = 2^25.
    refused(
        exp(25, 16, 1, 1 << 40),
        TableError::ExpTail { tail: 1 << 25 },
    );
    // The tail starts by key 2^24, or the table is refused.
    let (num, den) = scale_with_tail_at((1 << 24) + 1);
    refused(
        exp(26, 0, num, den),
        TableError::ExpTail {
            tail: (1 << 24) + 1,
        },
    );
    let (num, den) = scale_with_tail_at(1 << 24);
    accepted(exp(26, 0, num, den), 26);
    // No tail in a table of 2^24 keys: z = T = 2^24 is admitted.
    accepted(exp(24, 32, 1, 1 << 40), 24);
    // The first rule broken is the error.
    refused(exp(0, 40, -1, 1), TableError::ExpKeyBits { key_bits: 0 });
    refused(exp(8, 40, -1, 1), TableError::ExpSigma { sigma: 40 });
    refused(exp(32, 32, 0, 1), TableError::ExpScale);
    accepted(exp(1, 0, 1, 1), 1);
    accepted(exp(32, 32, i64::MAX, 1), 32);
}

#[test]
fn exp2frac_rules_in_order() {
    for frac_bits in [0, 17] {
        refused(
            TableKind::Exp2Frac {
                frac_bits,
                out_bits: 16,
            },
            TableError::Exp2FracBits { frac_bits },
        );
    }
    for out_bits in [0, 57] {
        refused(
            TableKind::Exp2Frac {
                frac_bits: 8,
                out_bits,
            },
            TableError::Exp2FracOutBits { out_bits },
        );
    }
    refused(
        TableKind::Exp2Frac {
            frac_bits: 0,
            out_bits: 0,
        },
        TableError::Exp2FracBits { frac_bits: 0 },
    );
    for (frac_bits, out_bits) in [(1, 1), (16, 56)] {
        accepted(
            TableKind::Exp2Frac {
                frac_bits,
                out_bits,
            },
            frac_bits,
        );
    }
}

#[test]
fn silu_rules_in_order() {
    refused(silu(0, &[0], -1, -1), TableError::KeyBits);
    refused(silu(4, &[], -1, -1), TableError::Classes { classes: 0 });
    refused(
        silu(4, &[0, 0, 0, 0, 0], -1, -1),
        TableError::Classes { classes: 5 },
    );
    refused(
        silu(31, &[20, 20, 20], -1, -1),
        TableError::RowBits { b: 31, classes: 3 },
    );
    refused(
        silu(4, &[0, 4], -1, -1),
        TableError::SiluShift { class: 1, shift: 4 },
    );
    refused(
        silu(26, &[1], -1, -1),
        TableError::SiluCore { values: 1 << 25 },
    );
    refused(
        silu(24, &[0, 1], -1, -1),
        TableError::SiluCore {
            values: (1 << 24) + (1 << 23),
        },
    );
    for in_exp in [1, 127, -65, -128] {
        refused(silu(4, &[1], in_exp, -1), TableError::SiluInExp { in_exp });
    }
    // s = 63 − in_exp + out_exp: 0 and 127 are out, 1 and 126 in.
    refused(silu(4, &[1], 0, -63), TableError::SiluOutShift { shift: 0 });
    refused(
        silu(4, &[1], -1, 63),
        TableError::SiluOutShift { shift: 127 },
    );
    refused(
        silu(4, &[1], -64, 127),
        TableError::SiluOutShift { shift: 254 },
    );
    // The first rule broken is the error.
    refused(
        silu(4, &[4, 9], 1, 127),
        TableError::SiluShift { class: 0, shift: 4 },
    );
    refused(
        silu(26, &[1], 1, 127),
        TableError::SiluCore { values: 1 << 25 },
    );
    refused(silu(4, &[1], 1, 127), TableError::SiluInExp { in_exp: 1 });
    // Every boundary value is accepted.
    accepted(silu(1, &[0], 0, -62), 1);
    accepted(silu(25, &[1], -64, -1), 25);
    accepted(silu(32, &[8], -1, 62), 32);
    accepted(silu(30, &[7, 8, 9, 10], 0, 63), 32);
}

#[test]
fn rope_rules_in_order() {
    let set = |field: &str, value: u32| {
        rope_with(|kind| {
            let TableKind::Rope {
                theta,
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max,
                bits,
                max_context,
                d_head,
            } = kind
            else {
                unreachable!()
            };
            match field {
                "theta" => *theta = value,
                "factor" => *factor = value as u16,
                "low" => *low_freq_factor = value as u16,
                "high" => *high_freq_factor = value as u16,
                "original_max" => *original_max = value,
                "bits" => *bits = value as u8,
                "max_context" => *max_context = value,
                "d_head" => *d_head = value as u16,
                _ => unreachable!(),
            }
        })
    };
    for theta in [0, 1] {
        refused(set("theta", theta), TableError::RopeTheta { theta });
    }
    refused(set("factor", 0), TableError::RopeFactor);
    refused(
        set("low", 0),
        TableError::RopeFrequencyFactors { low: 0, high: 4 },
    );
    refused(
        set("low", 4),
        TableError::RopeFrequencyFactors { low: 4, high: 4 },
    );
    refused(
        set("high", 1),
        TableError::RopeFrequencyFactors { low: 1, high: 1 },
    );
    refused(set("original_max", 0), TableError::RopeOriginalMax);
    for bits in [0, 31] {
        refused(set("bits", bits), TableError::RopeBits { bits: bits as u8 });
    }
    for d_head in [0, 1, 7] {
        refused(
            set("d_head", d_head),
            TableError::RopeHeadDim {
                d_head: d_head as u16,
            },
        );
    }
    refused(set("max_context", 0), TableError::RopeContext);
    refused(rope(1 << 23, 6), TableError::RopePairs { pairs: 3 << 23 });
    refused(
        rope(u32::MAX, u16::MAX - 1),
        TableError::RopePairs {
            pairs: u64::from(u32::MAX) * 32767,
        },
    );
    // Every boundary value is accepted: 2^24 pairs at most.
    accepted(set("theta", 2), 6);
    accepted(set("bits", 30), 6);
    accepted(rope(1 << 23, 4), 24);
    accepted(rope(1, 2), 0);
}

/// The first rule broken is the error, in the order of RFC-0008's 'Parameters allowed': from
/// parameters that break every rule, each step mends the first broken one, and the next rule is
/// the error. No `max_context` breaks both of its rules, since no position makes no pair, so the
/// steps run twice: without positions, and with `2^32 − 1` positions at the odd `d_head = 65,535`,
/// whose 32,767 frequencies make more than `2^24` pairs at every step.
#[test]
fn the_first_broken_rope_rule_is_the_error() {
    #[allow(clippy::too_many_arguments)]
    fn rope_of(
        theta: u32,
        factor: u16,
        low_freq_factor: u16,
        high_freq_factor: u16,
        original_max: u32,
        bits: u8,
        max_context: u32,
        d_head: u16,
    ) -> TableKind {
        TableKind::Rope {
            theta,
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max,
            bits,
            max_context,
            d_head,
        }
    }
    // Without positions.
    let steps = [
        (
            rope_of(1, 0, 0, 0, 0, 0, 0, 1),
            TableError::RopeTheta { theta: 1 },
        ),
        (
            rope_of(500_000, 0, 0, 0, 0, 0, 0, 1),
            TableError::RopeFactor,
        ),
        (
            rope_of(500_000, 8, 0, 0, 0, 0, 0, 1),
            TableError::RopeFrequencyFactors { low: 0, high: 0 },
        ),
        (
            rope_of(500_000, 8, 1, 4, 0, 0, 0, 1),
            TableError::RopeOriginalMax,
        ),
        (
            rope_of(500_000, 8, 1, 4, 8192, 0, 0, 1),
            TableError::RopeBits { bits: 0 },
        ),
        (
            rope_of(500_000, 8, 1, 4, 8192, 14, 0, 1),
            TableError::RopeHeadDim { d_head: 1 },
        ),
        (
            rope_of(500_000, 8, 1, 4, 8192, 14, 0, 8),
            TableError::RopeContext,
        ),
    ];
    for (kind, error) in steps {
        refused(kind, error);
    }
    accepted(rope_of(500_000, 8, 1, 4, 8192, 14, 16, 8), 6);
    // With more than 2^24 pairs.
    let many = u32::MAX;
    let steps = [
        (
            rope_of(1, 0, 4, 4, 0, 31, many, 65_535),
            TableError::RopeTheta { theta: 1 },
        ),
        (
            rope_of(2, 0, 4, 4, 0, 31, many, 65_535),
            TableError::RopeFactor,
        ),
        (
            rope_of(2, 1, 4, 4, 0, 31, many, 65_535),
            TableError::RopeFrequencyFactors { low: 4, high: 4 },
        ),
        (
            rope_of(2, 1, 4, 5, 0, 31, many, 65_535),
            TableError::RopeOriginalMax,
        ),
        (
            rope_of(2, 1, 4, 5, 1, 31, many, 65_535),
            TableError::RopeBits { bits: 31 },
        ),
        (
            rope_of(2, 1, 4, 5, 1, 30, many, 65_535),
            TableError::RopeHeadDim { d_head: 65_535 },
        ),
        (
            rope_of(2, 1, 4, 5, 1, 30, many, 65_534),
            TableError::RopePairs {
                pairs: u64::from(many) * 32_767,
            },
        ),
    ];
    for (kind, error) in steps {
        refused(kind, error);
    }
    // 512 · 32,767 = 16,776,704 pairs, just below 2^24.
    accepted(rope_of(2, 1, 4, 5, 1, 30, 512, 65_534), 24);
}

/// A vector specification with another `log2_rows` is refused, with the count it needs.
#[test]
fn a_wrong_log2_rows_is_refused() {
    for case in &table_cases() {
        let bytes = hex(text(case, "spec"));
        let spec = decode(&bytes, &DecodeLimits::MESSAGE, TableSpec::decode).unwrap();
        for found in [
            spec.log2_rows.wrapping_add(1),
            spec.log2_rows.wrapping_sub(1),
            255,
        ] {
            let wrong = TableSpec {
                log2_rows: found,
                ..spec.clone()
            };
            assert_eq!(
                Table::from_spec(&wrong).map(|t| t.spec().clone()),
                Err(TableError::Log2Rows {
                    expected: spec.log2_rows,
                    found
                }),
                "{}",
                text(case, "name")
            );
        }
    }
}

/// Another digest is refused: a changed bit of the digest, another id under the same kind, and
/// another valid parameter with the same row count.
#[test]
fn another_digest_is_refused() {
    for case in &table_cases() {
        let name = text(case, "name");
        let bytes = hex(text(case, "spec"));
        let spec = decode(&bytes, &DecodeLimits::MESSAGE, TableSpec::decode).unwrap();
        assert_eq!(spec.digest, digest_hex(text(case, "digest")));
        for bit in [0, 7, 100, 255] {
            let mut digest = spec.digest;
            digest[bit / 8] ^= 1 << (bit % 8);
            let wrong = TableSpec {
                digest,
                ..spec.clone()
            };
            assert_eq!(
                Table::from_spec(&wrong).map(|t| t.spec().clone()),
                Err(TableError::Digest),
                "{name}, bit {bit}"
            );
        }
        let moved = TableSpec {
            id: TableId(spec.id.0 + 1),
            ..spec.clone()
        };
        assert_eq!(
            Table::from_spec(&moved).map(|t| t.spec().clone()),
            Err(TableError::Digest),
            "{name}, another id"
        );
    }
    // Other parameters, same rows: the digest binds every field.
    let file_spec = |name: &str| {
        decode(
            &hex(text(&table_case(name), "spec")),
            &DecodeLimits::MESSAGE,
            TableSpec::decode,
        )
        .unwrap()
    };
    let changes: [(&str, TableKind); 6] = [
        ("requant-b4-one-class", requant(4, &[1], -128, 126)),
        ("requant-b4-one-class", requant(4, &[2], -128, 127)),
        ("bitwise8-xor", TableKind::Bitwise8 { op: BitOp::And }),
        ("gf256-mul-2", TableKind::Gf256Mul { c: 3 }),
        ("exp-k6-s8-quarter", exp(6, 8, 1, 5)),
        ("silu-b8-saturating", silu(8, &[1], 0, -3)),
    ];
    for (name, kind) in changes {
        let changed = TableSpec {
            kind,
            ..file_spec(name)
        };
        assert_eq!(
            Table::from_spec(&changed).map(|t| t.spec().clone()),
            Err(TableError::Digest),
            "{name}: {:?}",
            changed.kind
        );
    }
}

fn decode_spec(bytes: &[u8]) -> Result<TableSpec, DecodeError> {
    decode(bytes, &DecodeLimits::MESSAGE, TableSpec::decode)
}

fn decode_error(bytes: &[u8]) -> (ErrorCode, u64) {
    let error = decode_spec(bytes).unwrap_err();
    (error.code(), error.offset())
}

/// Tags and rationals that are not canonical, at their offsets.
#[test]
fn noncanonical_specifications_do_not_decode() {
    // range-1: id 0, tag 0, bits 1, log2_rows 1, digest.
    let mut range = vec![0, 0, 0, 0, 0, 0, 1, 1];
    range.extend([0xAB; 32]);
    assert!(decode_spec(&range).is_ok());
    for tag in [10u16, 11, 0xFFFF] {
        let mut bad = range.clone();
        bad[4..6].copy_from_slice(&tag.to_le_bytes());
        assert_eq!(
            decode_error(&bad),
            (ErrorCode::Noncanonical, 4),
            "tag {tag}"
        );
    }
    // Bitwise8: tag 3, then the operation's own tag at offset 6.
    let mut bitwise = vec![0, 0, 0, 0, 3, 0, 0, 0, 16];
    bitwise.extend([0; 32]);
    assert!(decode_spec(&bitwise).is_ok());
    for op in [2u16, 0x100] {
        let mut bad = bitwise.clone();
        bad[6..8].copy_from_slice(&op.to_le_bytes());
        assert_eq!(decode_error(&bad), (ErrorCode::Noncanonical, 6), "op {op}");
    }
    // Exp: tag 6, key_bits, sigma, then the scale at offset 8.
    let exp_bytes = |num: i64, den: u64| {
        let mut bytes = vec![0, 0, 0, 0, 6, 0, 6, 8];
        bytes.extend(num.to_le_bytes());
        bytes.extend(den.to_le_bytes());
        bytes.push(6);
        bytes.extend([0; 32]);
        bytes
    };
    assert!(decode_spec(&exp_bytes(1, 4)).is_ok());
    for (num, den) in [(2, 4), (1, 0), (0, 2), (-6, 9)] {
        assert_eq!(
            decode_error(&exp_bytes(num, den)),
            (ErrorCode::Noncanonical, 8),
            "{num}/{den}"
        );
    }
    // A negative scale in lowest terms decodes; validation refuses it.
    let negative = decode_spec(&exp_bytes(-1, 4)).unwrap();
    assert_eq!(
        Table::from_spec(&negative).map(|t| t.spec().clone()),
        Err(TableError::ExpScale)
    );
    // A byte past the specification.
    let mut trailing = range.clone();
    trailing.push(0);
    assert_eq!(decode_error(&trailing), (ErrorCode::Noncanonical, 40));
}

/// Every prefix of every vector specification is cut short, at the field the cut falls in; the
/// offsets of three layouts are spelled out.
#[test]
fn truncated_specifications_do_not_decode() {
    for case in &table_cases() {
        let bytes = hex(text(case, "spec"));
        for len in 0..bytes.len() {
            let (code, offset) = decode_error(&bytes[..len]);
            assert_eq!(code, ErrorCode::Truncated, "{}, {len}", text(case, "name"));
            assert!(offset <= len as u64, "{}, {len}", text(case, "name"));
        }
    }
    let spec_of = |name: &str| hex(text(&table_case(name), "spec"));
    // range-1: id @0, tag @4, bits @6, log2_rows @7, digest @8.
    let range = spec_of("range-1");
    for (cuts, offset) in [(0..4, 0), (4..6, 4), (6..7, 6), (7..8, 7), (8..40, 8)] {
        for len in cuts {
            assert_eq!(decode_error(&range[..len]), (ErrorCode::Truncated, offset));
        }
    }
    // requant-b5-three-classes-saturating: b @6, r_by_class's length @7 and its three bytes @15,
    // lo @18, hi @22, log2_rows @26, digest @27.
    let requant = spec_of("requant-b5-three-classes-saturating");
    for (cuts, offset) in [
        (6..7, 6),
        (7..18, 7),
        (18..22, 18),
        (22..26, 22),
        (26..27, 26),
        (27..59, 27),
    ] {
        for len in cuts {
            assert_eq!(
                decode_error(&requant[..len]),
                (ErrorCode::Truncated, offset),
                "{len}"
            );
        }
    }
    // exp-k6-s8-quarter: key_bits @6, sigma @7, num @8, den @16, log2_rows @24, digest @25.
    let exp = spec_of("exp-k6-s8-quarter");
    for (cuts, offset) in [(6..7, 6), (7..8, 7), (8..16, 8), (16..24, 16), (24..25, 24)] {
        for len in cuts {
            assert_eq!(
                decode_error(&exp[..len]),
                (ErrorCode::Truncated, offset),
                "{len}"
            );
        }
    }
}

/// The decoder runs under the caller's limits: the classes' length against `max_vec_len`, and
/// the nesting of the kind and its fields against `max_depth`.
#[test]
fn specifications_decode_under_the_limits() {
    let case = table_case("requant-b5-three-classes-saturating");
    let bytes = hex(text(&case, "spec"));
    let short = DecodeLimits {
        max_vec_len: 2,
        ..DecodeLimits::MESSAGE
    };
    let error = decode(&bytes, &short, TableSpec::decode).unwrap_err();
    assert_eq!((error.code(), error.offset()), (ErrorCode::Limit, 7));
    // The specification takes depth 1, its kind 2 and the classes 3.
    for (max_depth, offset) in [(0, 0), (1, 4), (2, 7)] {
        let shallow = DecodeLimits {
            max_depth,
            ..DecodeLimits::MESSAGE
        };
        let error = decode(&bytes, &shallow, TableSpec::decode).unwrap_err();
        assert_eq!(
            (error.code(), error.offset()),
            (ErrorCode::Limit, offset),
            "depth {max_depth}"
        );
    }
    let deep_enough = DecodeLimits {
        max_depth: 3,
        ..DecodeLimits::MESSAGE
    };
    assert!(decode(&bytes, &deep_enough, TableSpec::decode).is_ok());
}

/// Classes as many as a claim list's limits allow (RFC-0003 'Limits'), here `2^20 + 1`, each of
/// shift 0, which both kinds' shift rules refuse: they decode into an exact copy, and both entry
/// points refuse them by their number, `Classes`, read from their length before any class is
/// read. With `b = 0`, the key-width rule, which comes first, is the error. `from_spec` copies
/// and allocates nothing before these rules pass, so the decoder's copy is the only one.
#[test]
fn oversized_classes_are_refused_by_their_number() {
    const CLASSES: usize = (1 << 20) + 1;
    // id 7, the kind's tag, `b`, the classes, the kind's last fields, `log2_rows` and a digest.
    let spec_bytes = |tag: u16, b: u8, last: &[u8]| {
        let mut bytes = Vec::new();
        bytes.extend(7u32.to_le_bytes());
        bytes.extend(tag.to_le_bytes());
        bytes.push(b);
        bytes.extend((CLASSES as u64).to_le_bytes());
        bytes.resize(bytes.len() + CLASSES, 0);
        bytes.extend(last);
        bytes.push(4);
        bytes.extend([0; 32]);
        bytes
    };
    // Requant's `lo = −1` and `hi = 1`; Silu's `in_exp = out_exp = −1`.
    let requant_last = [(-1i32).to_le_bytes(), 1i32.to_le_bytes()].concat();
    let silu_last = [(-1i8).to_le_bytes(), (-1i8).to_le_bytes()].concat();
    for (tag, last) in [(1u16, &requant_last), (8, &silu_last)] {
        for (b, error) in [
            (4, TableError::Classes { classes: CLASSES }),
            (0, TableError::KeyBits),
        ] {
            let spec = decode_spec(&spec_bytes(tag, b, last)).unwrap();
            let (TableKind::Requant { r_by_class, .. } | TableKind::Silu { r_by_class, .. }) =
                &spec.kind
            else {
                panic!("tag {tag} decoded to {:?}", spec.kind.tag());
            };
            assert_eq!(r_by_class.len(), CLASSES);
            assert!(r_by_class.iter().all(|&shift| shift == 0));
            assert_eq!(
                Table::from_spec(&spec).map(|t| t.spec().clone()),
                Err(error),
                "tag {tag}, b = {b}"
            );
            assert_eq!(
                Table::generate(spec.id, spec.kind).map(|t| t.spec().clone()),
                Err(error),
                "tag {tag}, b = {b}"
            );
        }
    }
}

#[test]
fn only_the_claim_lists_faults_are_rejections() {
    for error in [
        TableError::NotLookupTable,
        TableError::PointLength {
            expected: 3,
            found: 2,
        },
        TableError::Memory,
        TableError::Internal,
    ] {
        assert!(!error.is_rejection(), "{error:?}");
        assert!(!error.to_string().is_empty());
    }
    for error in [
        TableError::Digest,
        TableError::Log2Rows {
            expected: 3,
            found: 2,
        },
        TableError::ExpTail { tail: 1 << 25 },
    ] {
        assert!(error.is_rejection(), "{error:?}");
        assert!(!error.to_string().is_empty());
    }
}
