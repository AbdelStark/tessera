// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The invariants of RFC-0003 over every type the codec implements:
//! - INV-TCE-1: `decode(encode(v)) = v`.
//! - INV-TCE-2: whatever decodes, re-encodes to the same bytes. The inputs are valid encodings
//!   after up to four edits (a byte replaced, inserted or deleted), which explores the boundary
//!   between canonical and not. Uniform random bytes almost never decode as a structured type,
//!   so they would test nothing.
//! - INV-TCE-3: a length whose elements cannot fit in the bytes left fails before any element is
//!   read.
//! - INV-TCE-4: trailing bytes are rejected.
//! - Every encoding is at least `MIN_SIZE` bytes, the bound the length checks rely on: an
//!   overstated `MIN_SIZE` would reject valid input.

use std::fmt::Debug;

use proptest::prelude::*;
use tessera_codec::error::ErrorCode;
use tessera_codec::golden::{CORPUS_LIMITS, Sample, Shape};
use tessera_codec::{
    Ascii, DEPTH_CEILING, Decode, DecodeLimits, Encode, Map, Rational, decode, encode,
};

const LIMITS: DecodeLimits = DecodeLimits {
    max_total_bytes: 1 << 20,
    max_vec_len: 1 << 16,
    max_depth: 64,
};

/// Up to four edits of an encoding: `(kind, position, byte)`, where the kind replaces, inserts
/// or deletes.
type Edits = Vec<(u8, usize, u8)>;

fn edits() -> impl Strategy<Value = Edits> {
    prop::collection::vec((0..3u8, any::<usize>(), any::<u8>()), 1..=4)
}

fn apply(mut bytes: Vec<u8>, edits: &Edits) -> Vec<u8> {
    for &(kind, position, byte) in edits {
        let at = position % (bytes.len() + 1);
        match kind {
            0 if at < bytes.len() => bytes[at] = byte,
            1 => bytes.insert(at, byte),
            _ if at < bytes.len() => {
                bytes.remove(at);
            }
            _ => bytes.push(byte),
        }
    }
    bytes
}

/// Every invariant for `value`.
fn laws<T: Encode + Decode + PartialEq + Debug>(
    value: T,
    edits: &Edits,
) -> Result<(), TestCaseError> {
    let encoded = encode(&value);
    prop_assert!(
        encoded.len() >= T::MIN_SIZE,
        "{} < MIN_SIZE {}",
        encoded.len(),
        T::MIN_SIZE
    );
    prop_assert_eq!(decode::<T>(&encoded, LIMITS), Ok(value));
    let edited = apply(encoded.clone(), edits);
    if let Ok(other) = decode::<T>(&edited, LIMITS) {
        prop_assert_eq!(encode(&other), edited);
    }
    let mut longer = encoded;
    longer.push(0);
    prop_assert_eq!(
        decode::<T>(&longer, LIMITS).map(|_| ()).map_err(|e| e.code),
        Err(ErrorCode::CodecNoncanonical)
    );
    Ok(())
}

fn ascii() -> impl Strategy<Value = Ascii> {
    "[ -~]{0,24}".prop_map(|text| Ascii::new(text).expect("printable"))
}

fn rational() -> impl Strategy<Value = Rational> {
    (any::<i64>(), 1..=u64::MAX).prop_map(|(num, den)| Rational::new(num, den).expect("den > 0"))
}

fn map() -> impl Strategy<Value = Map<u16, Vec<u8>>> {
    prop::collection::vec(
        (any::<u16>(), prop::collection::vec(any::<u8>(), 0..4)),
        0..8,
    )
    .prop_map(|mut entries| {
        entries.sort_by_key(|e| e.0);
        entries.dedup_by_key(|e| e.0);
        Map::from_entries(entries).expect("distinct keys")
    })
}

fn sample() -> impl Strategy<Value = Sample> {
    (
        any::<bool>(),
        any::<u8>(),
        prop::collection::vec(any::<u16>(), 0..6),
        prop::option::of(any::<i32>()),
        ascii(),
        any::<[u8; 32]>(),
    )
        .prop_map(|(flag, small, list, maybe, name, digest)| Sample {
            flag,
            small,
            list,
            maybe,
            name,
            digest,
        })
}

fn shape() -> impl Strategy<Value = Shape> {
    let leaf = prop_oneof![
        Just(Shape::Empty),
        any::<u32>().prop_map(Shape::Point),
        (any::<u16>(), any::<u16>()).prop_map(|(width, height)| Shape::Box { width, height }),
    ];
    leaf.prop_recursive(4, 24, 4, |inner| {
        prop::collection::vec(inner, 0..4).prop_map(Shape::Nested)
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn integers(a in any::<u8>(), b in any::<u16>(), c in any::<u32>(), d in any::<u64>(),
                e in any::<i8>(), f in any::<i16>(), g in any::<i32>(), h in any::<i64>(),
                edits in edits()) {
        laws(a, &edits)?;
        laws(b, &edits)?;
        laws(c, &edits)?;
        laws(d, &edits)?;
        laws(e, &edits)?;
        laws(f, &edits)?;
        laws(g, &edits)?;
        laws(h, &edits)?;
    }

    #[test]
    fn bools_and_arrays(flag in any::<bool>(), digest in any::<[u8; 32]>(), triple in any::<[u16; 3]>(),
                        edits in edits()) {
        laws(flag, &edits)?;
        laws(digest, &edits)?;
        laws(triple, &edits)?;
    }

    #[test]
    fn sequences(words in prop::collection::vec(any::<u32>(), 0..20),
                 bytes in prop::collection::vec(any::<u8>(), 0..64),
                 signed in prop::collection::vec(any::<i8>(), 0..64),
                 nested in prop::collection::vec(prop::collection::vec(any::<i16>(), 0..5), 0..5),
                 pair in (any::<u8>(), prop::collection::vec(any::<u8>(), 0..8)),
                 maybe in prop::option::of(any::<u64>()),
                 edits in edits()) {
        laws(words, &edits)?;
        laws(bytes, &edits)?;
        laws(signed, &edits)?;
        laws(nested, &edits)?;
        laws(pair, &edits)?;
        laws(maybe, &edits)?;
    }

    #[test]
    fn named_types(text in ascii(), fraction in rational(), entries in map(), edits in edits()) {
        laws(text, &edits)?;
        laws(fraction, &edits)?;
        laws(entries, &edits)?;
    }

    #[test]
    fn derived_types(value in sample(), figure in shape(), edits in edits()) {
        laws(value, &edits)?;
        laws(figure, &edits)?;
    }

    /// Any fraction is stored in lowest terms, so equal fractions encode equally.
    #[test]
    fn fractions_are_reduced(num in any::<i64>(), den in 1..=u64::MAX, k in 1..=1000u64) {
        let reduced = Rational::new(num, den).expect("den > 0");
        if let (Some(scaled_num), Some(scaled_den)) = (num.checked_mul(k as i64), den.checked_mul(k)) {
            prop_assert_eq!(Rational::new(scaled_num, scaled_den), Some(reduced));
        }
        prop_assert!(reduced.den() > 0);
    }
}

/// Derived minimum sizes are the smallest encodings: a sample with every sequence empty and no
/// option, and the unit variant of `Shape`.
#[test]
fn derived_minimum_sizes_are_tight() {
    let smallest = Sample {
        flag: false,
        small: 0,
        list: vec![],
        maybe: None,
        name: Ascii::default(),
        digest: [0; 32],
    };
    assert_eq!(encode(&smallest).len(), Sample::MIN_SIZE);
    assert_eq!(encode(&Shape::Empty).len(), Shape::MIN_SIZE);
}

/// INV-TCE-3 at its exact bound: lengths that fit the bytes left one byte per element but not
/// at `MIN_SIZE` bytes each fail at the length itself, before any element is read and before
/// anything is reserved. A check against the element count alone, or none, would fail later,
/// at an element, after reserving memory for a length the input never backs.
#[test]
fn lengths_are_checked_against_min_size_before_reading() {
    let at_one = |result: Result<(), tessera_codec::DecodeError>| {
        let error = result.expect_err("the elements cannot fit");
        (error.code, error.offset)
    };
    let expected = (ErrorCode::CodecTruncated, 1);
    // Ten u64 against 16 bytes: 10 ≤ 16, but 80 > 16.
    let vec = [vec![0xAA], 10u64.to_le_bytes().to_vec(), vec![0; 16]].concat();
    assert_eq!(
        at_one(decode::<(u8, Vec<u64>)>(&vec, LIMITS).map(drop)),
        expected
    );
    // Ten (u16, u8) entries, three bytes each, against 12 bytes.
    let map = [vec![0xAA], 10u64.to_le_bytes().to_vec(), vec![0; 12]].concat();
    assert_eq!(
        at_one(decode::<(u8, Map<u16, u8>)>(&map, LIMITS).map(drop)),
        expected
    );
    // Four u64 against 16 bytes: an array has no length, and is checked as a whole.
    let array = [vec![0xAA], vec![0; 16]].concat();
    assert_eq!(
        at_one(decode::<(u8, [u64; 4])>(&array, LIMITS).map(drop)),
        expected
    );
}

/// The nesting limit counts every composite value: three nested shapes fit a depth of 8, four
/// do not.
#[test]
fn nesting_beyond_the_limit_is_rejected() {
    assert_eq!(
        decode::<Shape>(&encode(&Shape::nest(3)), CORPUS_LIMITS),
        Ok(Shape::nest(3))
    );
    let error = decode::<Shape>(&encode(&Shape::nest(4)), CORPUS_LIMITS).expect_err("too deep");
    assert_eq!(error.code, ErrorCode::CodecLimit);
}

/// Whatever `max_depth` says, no reader nests deeper than `DEPTH_CEILING`, and nesting right up
/// to it fits a 2 MiB thread stack, the smallest a decoding thread is given here.
#[test]
fn nesting_stops_at_the_ceiling_within_a_small_stack() {
    let unbounded = DecodeLimits {
        max_depth: u16::MAX,
        ..LIMITS
    };
    // `nest(k)` reaches depth 2k + 1.
    let deepest = (usize::from(DEPTH_CEILING) - 1) / 2;
    let fits = encode(&Shape::nest(deepest));
    let too_deep = encode(&Shape::nest(deepest + 1));
    let outcome = std::thread::Builder::new()
        .stack_size(2 << 20)
        .spawn(move || {
            (
                decode::<Shape>(&fits, unbounded).map(drop),
                decode::<Shape>(&too_deep, unbounded).map_err(|e| e.code),
            )
        })
        .expect("a thread")
        .join()
        .expect("no stack overflow");
    assert_eq!(outcome.0, Ok(()));
    assert_eq!(outcome.1.map(drop), Err(ErrorCode::CodecLimit));
}

/// Maps sort by key encoding and refuse duplicate keys.
#[test]
fn maps_are_canonical_on_construction() {
    let map = Map::from_entries(vec![(1u16, b'a'), (256, b'b'), (2, b'c')]).expect("distinct keys");
    let keys: Vec<u16> = map.entries().iter().map(|e| e.0).collect();
    assert_eq!(keys, [256, 1, 2]);
    assert_eq!(map.get(&2), Some(&b'c'));
    assert_eq!(map.get(&3), None);
    assert!(Map::from_entries(vec![(5u16, 1u8), (5, 2)]).is_err());
}

#[test]
fn ascii_rejects_unprintable_bytes() {
    assert_eq!(
        Ascii::new("ok\u{7}").map(|_| ()).map_err(|e| e.index),
        Err(2)
    );
    assert_eq!(Ascii::new("é").map(|_| ()).map_err(|e| e.index), Err(0));
    assert!(Ascii::new(" ~").is_ok());
}

/// A reader's debug form says where it is, never what it reads: decoders run on secret-class
/// artifacts (05 §3).
#[test]
fn readers_do_not_print_their_input() {
    let secret = [0xDE, 0xAD, 0xBE, 0xEF];
    let reader = tessera_codec::Reader::new(&secret, LIMITS).expect("a small input");
    let shown = format!("{reader:?}");
    for byte in secret {
        assert!(!shown.contains(&byte.to_string()), "{shown}");
    }
    assert!(shown.contains("len: 4"), "{shown}");
}
