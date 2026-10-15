// SPDX-License-Identifier: Apache-2.0 OR MIT

//! One test per check of the crate-level specification: each input violates exactly one check,
//! or exactly the pair whose order is being pinned, and the test asserts the code and the offset.
//! The vectors fix codes for most of these; offsets inside a payload are fixed here.

mod support;

use support::{Ty, decode_whole};
use tv_codec::{Cursor, DecodeError, DecodeLimits, Envelope, ErrorCode, Expected, decode, size};
use tv_testkit::blake3;

use ErrorCode::{Digest, Internal, Limit, Magic, Noncanonical, Truncated, Version};

const SMALL: DecodeLimits = DecodeLimits {
    max_total_bytes: 64,
    max_vec_len: 16,
    max_depth: 8,
};

fn le(value: u64) -> [u8; 8] {
    value.to_le_bytes()
}

/// Decodes `bytes` as the vector type `name` under `limits`, expecting `code` at `offset`.
#[track_caller]
fn rejects(name: &str, bytes: &[u8], limits: &DecodeLimits, code: ErrorCode, offset: u64) {
    match decode_whole(&Ty::parse(name), bytes, limits) {
        Ok(value) => panic!("{name} accepted as {value:?}"),
        Err(error) => assert_eq!(
            (error.code(), error.offset()),
            (code, offset),
            "{name}: {error}"
        ),
    }
}

#[track_caller]
fn accepts(name: &str, bytes: &[u8], limits: &DecodeLimits) {
    if let Err(error) = decode_whole(&Ty::parse(name), bytes, limits) {
        panic!("{name}: {error}");
    }
}

#[test]
fn a_cut_value_fails_at_its_first_byte() {
    // The third field of `(u8, u16, u64)` has 7 of its 8 bytes.
    let bytes = [1, 2, 0, 0, 0, 0, 0, 0, 0, 0];
    rejects("(u8, u16, u64)", &bytes, &SMALL, Truncated, 3);
    // An enum tag cut short, and `Some` without its payload.
    rejects("Option<u8>", &[1], &SMALL, Truncated, 0);
    rejects("Option<u64>", &[1, 0, 9], &SMALL, Truncated, 2);
}

#[test]
fn non_canonical_primitives_fail_at_the_value() {
    rejects("(u8, u8, u8, bool)", &[0, 0, 0, 2], &SMALL, Noncanonical, 3);
    let mut fps = vec![2, 0, 0, 0, 0, 0, 0, 0];
    fps.extend(le(5));
    fps.extend(le(tv_field::Fp::MODULUS));
    rejects("Vec<Fp>", &fps, &SMALL, Noncanonical, 16);
}

#[test]
fn fp2_is_read_coefficient_by_coefficient() {
    let p = tv_field::Fp::MODULUS;
    let bad_c1 = [le(1), le(p)].concat();
    rejects("Fp2", &bad_c1, &SMALL, Noncanonical, 8);
    // A bad c0 with c1 missing: the coefficient is wrong before the pair is short.
    rejects("Fp2", &le(p), &SMALL, Noncanonical, 0);
}

#[test]
fn unknown_tags_fail_at_the_tag() {
    rejects("(u8, Option<u8>)", &[7, 2, 0], &SMALL, Noncanonical, 1);
    rejects("Shape", &[4, 0], &SMALL, Noncanonical, 0);
    rejects("Shape", &[0xff, 0xff], &SMALL, Noncanonical, 0);
}

#[test]
fn ascii_fails_at_the_first_byte_outside_the_printable_range() {
    let bytes = [&le(4)[..], b"ab\x7fc"].concat();
    rejects("Ascii", &bytes, &SMALL, Noncanonical, 10);
    let bytes = [&le(3)[..], b"\x00\x7f!"].concat();
    rejects("Ascii", &bytes, &SMALL, Noncanonical, 8);
    accepts("Ascii", &[&le(2)[..], b" ~"].concat(), &SMALL);
}

#[test]
fn non_canonical_rationals_fail_at_their_first_byte() {
    let rational = |num: i64, den: u64| [&[9][..], &num.to_le_bytes(), &den.to_le_bytes()].concat();
    rejects("(u8, Rational)", &rational(1, 0), &SMALL, Noncanonical, 1);
    rejects("(u8, Rational)", &rational(0, 0), &SMALL, Noncanonical, 1);
    rejects("(u8, Rational)", &rational(0, 5), &SMALL, Noncanonical, 1);
    rejects("(u8, Rational)", &rational(-4, 6), &SMALL, Noncanonical, 1);
    rejects(
        "(u8, Rational)",
        &rational(i64::MIN, 4),
        &SMALL,
        Noncanonical,
        1,
    );
    accepts("(u8, Rational)", &rational(i64::MIN, 3), &SMALL);
    accepts("(u8, Rational)", &rational(i64::MAX, u64::MAX), &SMALL);
}

#[test]
fn map_keys_are_checked_at_the_key_before_the_value() {
    // Keys 1 then 1: the duplicate fails at the second key, offset 8 + 3.
    let duplicate = [&le(2)[..], &[1, 0, 10, 1, 0, 20]].concat();
    rejects("Map<u16, u8>", &duplicate, &SMALL, Noncanonical, 11);
    // A duplicate key whose value, a `Bytes` declaring 5 bytes, has 2: the map's own check passes
    // (2 entries of at least 10 bytes in 22), and the key is checked before the value is read.
    let cut = [&le(2)[..], &[1, 0], &le(0), &[1, 0], &le(5), &[0, 0]].concat();
    let error = decode(&cut, &SMALL, |c| {
        c.map(size::U16, size::SEQUENCE, Cursor::u16, Cursor::bytes)?;
        Ok(())
    })
    .expect_err("a duplicate key");
    assert_eq!((error.code(), error.offset()), (Noncanonical, 18));
    // The same entries with distinct keys fail on the value.
    let distinct = [&le(2)[..], &[1, 0], &le(0), &[2, 0], &le(5), &[0, 0]].concat();
    rejects("Map<u16, Bytes>", &distinct, &SMALL, Truncated, 20);
    // Byte order of the encodings, not numeric order: 0x0100 (1) sorts after 0x0001 (256).
    accepts(
        "Map<u16, u8>",
        &[&le(2)[..], &[0, 1, 1, 1, 0, 2]].concat(),
        &SMALL,
    );
    rejects(
        "Map<u16, u8>",
        &[&le(2)[..], &[1, 0, 1, 0, 1, 2]].concat(),
        &SMALL,
        Noncanonical,
        11,
    );
}

#[test]
fn bytes_left_over_fail_at_the_first_of_them() {
    rejects("u16", &[1, 0, 0], &SMALL, Noncanonical, 2);
    rejects(
        "Vec<u8>",
        &[&le(1)[..], &[5, 6, 7]].concat(),
        &SMALL,
        Noncanonical,
        9,
    );
}

#[test]
fn the_total_size_is_checked_before_decoding_starts() {
    // 65 bytes over a limit of 64, beginning with a bool that is not canonical: the size wins.
    let mut bytes = vec![0; 65];
    bytes[0] = 2;
    rejects("(bool, [u8; 64])", &bytes, &SMALL, Limit, 0);
    rejects("Bytes", &[0; 65], &SMALL, Limit, 0);
    accepts("[u8; 64]", &[0; 64], &SMALL);
}

#[test]
fn sequence_lengths_check_the_limit_then_the_bytes_left_at_the_length() {
    // Inside a struct, at the length's offset 1.
    let over = [&[3][..], &le(17), &[0; 17]].concat();
    rejects(
        "(u8, Bytes)",
        &over,
        &DecodeLimits {
            max_total_bytes: 1 << 10,
            ..SMALL
        },
        Limit,
        1,
    );
    let short = [&[3][..], &le(3), &[0; 5]].concat();
    rejects("(u8, Vec<u16>)", &short, &SMALL, Truncated, 1);
    // Above the limit and the bytes left: the limit comes first.
    rejects(
        "(u8, Vec<u64>)",
        &[&[3][..], &le(1 << 63)].concat(),
        &SMALL,
        Limit,
        1,
    );
    // A length at exactly the bytes left over the minimum passes the check.
    accepts("Vec<u16>", &[&le(2)[..], &[1, 0, 2, 0]].concat(), &SMALL);
}

#[test]
fn a_hostile_length_is_rejected_without_allocating() {
    // A 2^63 length with 16 bytes left and no length limit: had the decoder reserved 2^63
    // elements first, it would have failed with ERR-CODEC-LIMIT on the allocation, or aborted.
    let unlimited = DecodeLimits {
        max_total_bytes: u64::MAX,
        max_vec_len: u64::MAX,
        max_depth: 32,
    };
    let bytes = [&le(1 << 63)[..], &[0; 16]].concat();
    rejects("Vec<u64>", &bytes, &unlimited, Truncated, 0);
    rejects("Vec<Vec<u8>>", &bytes, &unlimited, Truncated, 0);
    rejects(
        "Map<u8, u8>",
        &[&le(u64::MAX)[..], &[0; 16]].concat(),
        &unlimited,
        Truncated,
        0,
    );
}

#[test]
fn element_minimums_bound_lengths_and_elements_must_meet_them() {
    // An element type of size 0 cannot occur in a schema (RFC-0003 'Limits'); were one passed,
    // the length is still bounded by the bytes left. Accepted, the value would be 2^34 units:
    // the closure reports its length, so that a failure message stays small.
    let unit = |c: &mut Cursor<'_>| c.vec(0, |_| Ok(())).map(|units| units.len());
    let error = decode(&le(1 << 34), &DecodeLimits::MESSAGE, unit).expect_err("bounded");
    assert_eq!((error.code(), error.offset()), (Truncated, 0));
    // An element that consumes nothing, or less than its declared minimum, is the schema's bug,
    // reported at the element and never blamed on the input.
    let three = [&le(3)[..], &[0; 3]].concat();
    let error = decode(&three, &DecodeLimits::MESSAGE, unit).expect_err("nothing consumed");
    assert_eq!((error.code(), error.offset()), (Internal, 8));
    assert!(!error.code().is_input_fault());
    let pairs = [&le(2)[..], &[1, 2, 3, 4]].concat();
    let error = decode(&pairs, &DecodeLimits::MESSAGE, |c| {
        c.vec(size::U16, Cursor::u8)
    })
    .expect_err("a u8 under a u16 minimum");
    assert_eq!((error.code(), error.offset()), (Internal, 8));
    let error = decode(&[0, 0], &DecodeLimits::MESSAGE, |c| {
        c.array(2, size::U8, |_| Ok(()))
    })
    .expect_err("an empty array element");
    assert_eq!((error.code(), error.offset()), (Internal, 0));
    // A minimum too large for a u64 saturates rather than wrapping to a small one, so no input
    // holds even one element: 2^63 · 2 and u64::MAX + 1 would both wrap to 0.
    let one = [&le(1)[..], &[0; 40]].concat();
    for element_min in [
        size::array(1 << 63, size::U16),
        size::sum(&[u64::MAX, size::U8]),
    ] {
        let error = decode(&one, &DecodeLimits::MESSAGE, |c| {
            c.vec(element_min, Cursor::u8)
        })
        .expect_err("no room for one element");
        assert_eq!((error.code(), error.offset()), (Truncated, 0));
    }
}

#[test]
fn size_constants_are_the_minimum_encodings() {
    // Schemas pass these to the bytes-left checks; the test support states each minimum
    // independently. A constant too small moves an error from a sequence's length to an element.
    for (name, constant) in [
        ("bool", size::BOOL),
        ("u8", size::U8),
        ("i8", size::I8),
        ("u16", size::U16),
        ("i16", size::I16),
        ("u32", size::U32),
        ("i32", size::I32),
        ("u64", size::U64),
        ("i64", size::I64),
        ("Fp", size::FP),
        ("Fp2", size::FP2),
        ("Digest", size::DIGEST),
        ("Rational", size::RATIONAL),
        ("Bytes", size::SEQUENCE),
        ("Map<u8, u8>", size::SEQUENCE),
        ("Shape", size::TAG),
        ("Option<u64>", size::OPTION),
        ("[u16; 3]", size::array(3, size::U16)),
        (
            "(u8, Bytes, Fp2)",
            size::sum(&[size::U8, size::SEQUENCE, size::FP2]),
        ),
    ] {
        assert_eq!(Ty::parse(name).min_size(), constant, "min_size({name})");
    }
}

#[test]
fn arrays_are_checked_as_a_whole() {
    rejects("[u16; 3]", &[1, 0, 2, 0, 3], &SMALL, Truncated, 0);
    let digest_short = [&[1, 7][..], &[0; 31]].concat();
    rejects("(bool, u8, Digest)", &digest_short, &SMALL, Truncated, 2);
}

/// `Option^n<u8>` holding 7.
fn options(n: usize) -> Vec<u8> {
    [[1u8, 0].repeat(n), vec![7]].concat()
}

fn nested_options(n: usize) -> String {
    format!("Option^{n}<u8>")
}

#[test]
fn depth_is_taken_before_a_value_is_read() {
    // Nine Options under a depth of 8, the ninth with no byte at all: too deep before cut short.
    rejects("Option^9<u8>", &options(8)[..16], &SMALL, Limit, 16);
    rejects("Option^9<u8>", &options(9), &SMALL, Limit, 16);
    // Each composite is one level: here a Vec, an Fp2, an array, and a map and its entry; the
    // vectors cover Ascii, Rational and tuples.
    let deep = |inner: &str| format!("Option^8<{inner}>");
    rejects(
        &deep("Vec<u8>"),
        &[[1u8, 0].repeat(8), le(0).to_vec()].concat(),
        &SMALL,
        Limit,
        16,
    );
    rejects(
        &deep("Fp2"),
        &[[1u8, 0].repeat(8), vec![0; 16]].concat(),
        &SMALL,
        Limit,
        16,
    );
    rejects(
        &deep("Digest"),
        &[[1u8, 0].repeat(8), vec![0; 32]].concat(),
        &SMALL,
        Limit,
        16,
    );
    // Option^7<Map<u8, u8>>: the map is at depth 8, its entry at 9.
    let map = [[1u8, 0].repeat(7), le(1).to_vec(), vec![1, 2]].concat();
    rejects("Option^7<Map<u8, u8>>", &map, &SMALL, Limit, 22);
    // Integers, bool and Fp take no level.
    accepts(
        "Option^8<Fp>",
        &[[1u8, 0].repeat(8), vec![0; 8]].concat(),
        &SMALL,
    );
}

#[test]
fn depth_limits_above_256_count_as_256() {
    for max_depth in [256, 257, 1000, u16::MAX] {
        let limits = DecodeLimits {
            max_total_bytes: 1 << 20,
            max_vec_len: 16,
            max_depth,
        };
        accepts(&nested_options(256), &options(256), &limits);
        rejects(&nested_options(257), &options(257), &limits, Limit, 512);
    }
}

#[test]
fn a_depth_limit_of_zero_admits_primitives_only() {
    let flat = DecodeLimits {
        max_depth: 0,
        ..SMALL
    };
    accepts("u64", &le(5), &flat);
    accepts("Fp", &le(5), &flat);
    rejects("Option<u8>", &[0, 0], &flat, Limit, 0);
    rejects("Bytes", &le(0), &flat, Limit, 0);
}

#[test]
fn siblings_release_their_level() {
    // Two-level siblings under a depth of 2.
    let two = DecodeLimits {
        max_depth: 2,
        ..SMALL
    };
    let vec_of_options = [&le(3)[..], &[1, 0, 1, 0, 0, 1, 0, 2]].concat();
    accepts("Vec<Option<u8>>", &vec_of_options, &two);
    let after = decode(&vec_of_options, &two, |c| {
        c.vec(2, |c| c.option(Cursor::u8))?;
        Ok(c.depth())
    })
    .expect("decodes");
    assert_eq!(after, 0);
}

// The envelope's own checks, beyond the vectors.

const PROBE: Expected = Expected {
    schema_id: 0xFFF0,
    version: 3,
};

fn header(schema_version: u16, payload_len: u64) -> Vec<u8> {
    [
        &b"TSRA"[..],
        &1u16.to_le_bytes(),
        &PROBE.schema_id.to_le_bytes(),
        &schema_version.to_le_bytes(),
        &0u16.to_le_bytes(),
        &payload_len.to_le_bytes(),
    ]
    .concat()
}

/// A sealed envelope of the probe schema.
fn seal(payload: &[u8]) -> Vec<u8> {
    let mut bytes = [header(3, payload.len() as u64), payload.to_vec()].concat();
    let digest = blake3::derive_key("tessera/v1/envelope", &bytes);
    bytes.extend(digest);
    bytes
}

fn open<'a>(bytes: &'a [u8], limits: &DecodeLimits) -> Result<Envelope<'a>, DecodeError> {
    Envelope::open(bytes, PROBE, limits, |signed| {
        blake3::derive_key("tessera/v1/envelope", signed)
    })
}

#[track_caller]
fn open_fails(bytes: &[u8], limits: &DecodeLimits, code: ErrorCode, offset: u64) {
    let error = open(bytes, limits).expect_err("rejected");
    assert_eq!((error.code(), error.offset()), (code, offset), "{error}");
}

#[test]
fn the_digest_is_computed_only_once_the_framing_is_sound() {
    let sealed = seal(&le(0));
    let refuse =
        |_: &[u8]| -> [u8; 32] { panic!("digest computed before the framing was checked") };
    for cut in 0..sealed.len() {
        let error = Envelope::open(&sealed[..cut], PROBE, &SMALL, refuse).expect_err("cut");
        assert_eq!(error.code(), Truncated);
    }
    let longer = [sealed.clone(), vec![0]].concat();
    let error = Envelope::open(&longer, PROBE, &SMALL, refuse).expect_err("longer");
    assert_eq!((error.code(), error.offset()), (Noncanonical, 60));
}

#[test]
fn an_envelope_counts_its_declared_size_not_its_input() {
    // A 60-byte envelope followed by 100 bytes, under a limit of 64: the size it declares is
    // within the limit, so the bytes after its digest are the fault.
    let bytes = [seal(&le(0)), vec![0; 100]].concat();
    open_fails(&bytes, &SMALL, Noncanonical, 60);
    // Declared: 13 + 52 = 65 bytes, one over.
    open_fails(&header(3, 13), &SMALL, Limit, 12);
}

#[test]
fn header_faults_are_reported_field_by_field() {
    let sealed = seal(&le(0));
    let with = |at: usize, bytes: &[u8]| {
        let mut copy = sealed.clone();
        copy[at..at + bytes.len()].copy_from_slice(bytes);
        copy
    };
    open_fails(&with(3, b"B"), &SMALL, Magic, 0);
    open_fails(&with(4, &2u16.to_le_bytes()), &SMALL, Version, 4);
    open_fails(&with(6, &0x0601u16.to_le_bytes()), &SMALL, Magic, 6);
    open_fails(&with(8, &1u16.to_le_bytes()), &SMALL, Version, 8);
    open_fails(&with(10, &1u16.to_le_bytes()), &SMALL, Noncanonical, 10);
    open_fails(&with(28, &[1]), &SMALL, Digest, 28);
    // The previous version, sealed under its own header, opens.
    let mut previous = [header(2, 8), le(0).to_vec()].concat();
    previous.extend(blake3::derive_key("tessera/v1/envelope", &previous));
    let envelope = open(&previous, &SMALL).expect("version 2 is accepted");
    assert_eq!(envelope.header().schema_version, 2);
    // The payload decoder is told which version's layout to read.
    let version = |envelope: Envelope<'_>| {
        envelope
            .decode_payload(|c, version| c.u64().map(|_| version))
            .expect("one u64")
    };
    assert_eq!(version(envelope), 2);
    assert_eq!(version(open(&sealed, &SMALL).expect("sound")), 3);
}

#[test]
fn payload_errors_count_from_the_start_of_the_envelope() {
    let limits = DecodeLimits {
        max_total_bytes: 256,
        ..SMALL
    };
    // A Vec<u8> declaring 17 elements, over max_vec_len 16: the length is at payload offset 0.
    let over = seal(&[&le(17)[..], &[0; 2]].concat());
    let envelope = open(&over, &limits).expect("sound framing");
    let error = envelope
        .decode_payload(|c, _| c.bytes())
        .expect_err("over the limit");
    assert_eq!((error.code(), error.offset()), (Limit, 20));
    // A value that does not fill the payload: the first byte left over.
    let short = seal(&[&le(1)[..], &[5, 6]].concat());
    let envelope = open(&short, &limits).expect("sound framing");
    let error = envelope
        .decode_payload(|c, _| c.bytes())
        .expect_err("left over");
    assert_eq!((error.code(), error.offset()), (Noncanonical, 29));
    // The payload decodes under the envelope's own limits, depth included.
    let deep = seal(&options(9));
    let envelope = open(&deep, &limits).expect("sound framing");
    let ty = Ty::parse("Option^9<u8>");
    let error = envelope
        .decode_payload(|c, _| support::decode_value(c, &ty))
        .expect_err("too deep");
    assert_eq!((error.code(), error.offset()), (Limit, 36));
    let fine = seal(&options(8));
    let envelope = open(&fine, &limits).expect("sound framing");
    assert_eq!(envelope.payload(), &options(8)[..]);
    assert_eq!(envelope.header().payload_len, 17);
}
