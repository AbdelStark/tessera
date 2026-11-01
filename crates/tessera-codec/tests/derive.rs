// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `#[derive(Encode, Decode)]` on the shapes schemas take: generic structs and enums, unit and
//! tuple structs, and an enum without variants. Exact bytes pin RFC-0003 'Composite types':
//! fields in declaration order with no tags, and a `u16` tag equal to the declaration index.
//! The test module shadows `Result`, as schema modules do, to check that the generated code
//! names everything by absolute path.

#![allow(dead_code)]

use tessera_codec::error::ErrorCode;
use tessera_codec::{Decode, DecodeLimits, Encode, decode, encode};

/// A crate-local alias, as many modules define one.
type Result<T> = core::result::Result<T, ()>;

const LIMITS: DecodeLimits = DecodeLimits {
    max_total_bytes: 1 << 16,
    max_vec_len: 1 << 10,
    max_depth: 16,
};

#[derive(Debug, PartialEq, Encode, Decode)]
struct Pair<T> {
    first: T,
    rest: Vec<T>,
}

#[derive(Debug, PartialEq, Encode, Decode)]
struct Unit;

#[derive(Debug, PartialEq, Encode, Decode)]
struct Tuple(u8, i16);

#[derive(Debug, PartialEq, Encode, Decode)]
enum Never {}

#[derive(Debug, PartialEq, Encode, Decode)]
enum Either<T> {
    Left(T),
    Right { value: T, flag: bool },
}

fn round_trip<T: Encode + Decode + PartialEq + std::fmt::Debug>(value: T, bytes: &[u8]) {
    assert_eq!(encode(&value), bytes, "{value:?}");
    assert_eq!(decode::<T>(bytes, LIMITS), Ok(value));
}

#[test]
fn fields_follow_declaration_order() {
    round_trip(
        Pair {
            first: 0x0102u16,
            rest: vec![0x0304],
        },
        &[0x02, 0x01, 1, 0, 0, 0, 0, 0, 0, 0, 0x04, 0x03],
    );
    round_trip(Tuple(9, -2), &[9, 0xFE, 0xFF]);
    round_trip(Unit, &[]);
    assert_eq!(<Pair<u16> as Decode>::MIN_SIZE, 2 + 8);
    assert_eq!(<Unit as Decode>::MIN_SIZE, 0);
}

#[test]
fn variants_are_tagged_by_declaration_index() {
    round_trip(Either::Left(7u8), &[0, 0, 7]);
    round_trip(
        Either::Right {
            value: 7u8,
            flag: true,
        },
        &[1, 0, 7, 1],
    );
    let error = decode::<Either<u8>>(&[2, 0, 7], LIMITS).expect_err("tag 2");
    assert_eq!(
        (error.code, error.offset),
        (ErrorCode::CodecNoncanonical, 0)
    );
    assert_eq!(<Either<u8> as Decode>::MIN_SIZE, 2 + 1);
}

#[test]
fn an_enum_without_variants_decodes_nothing() {
    for tag in [[0u8, 0], [1, 0], [0xFF, 0xFF]] {
        let error = decode::<Never>(&tag, LIMITS).expect_err("no variant");
        assert_eq!(error.code, ErrorCode::CodecNoncanonical);
    }
    assert_eq!(<Never as Decode>::MIN_SIZE, 2);
    let _: Result<()> = Ok(());
}

#[test]
fn signed_byte_slices_stream_exact_bits_without_heap_scratch() {
    struct Check<'a> {
        expected: &'a [u8],
        offset: usize,
        largest: usize,
    }
    impl tessera_codec::Sink for Check<'_> {
        fn put(&mut self, bytes: &[u8]) {
            assert_eq!(
                bytes,
                &self.expected[self.offset..self.offset + bytes.len()]
            );
            self.offset += bytes.len();
            self.largest = self.largest.max(bytes.len());
        }
    }
    for len in [0, 1, 127, 128, 255, 256, 4095, 4096, 4097, (1 << 20) + 1] {
        let expected = (0..len).map(|i| (i % 256) as u8).collect::<Vec<_>>();
        let values = expected
            .iter()
            .map(|&byte| i8::from_ne_bytes([byte]))
            .collect::<Vec<_>>();
        let mut check = Check {
            expected: &expected,
            offset: 0,
            largest: 0,
        };
        let observed =
            allocation_counter::measure(|| <i8 as Encode>::encode_slice(&values, &mut check));
        assert_eq!(observed.count_total, 0, "{len}: {observed:?}");
        assert_eq!(check.offset, len);
        assert!(check.largest <= 4096);
        let encoded = encode(&values);
        assert_eq!(&encoded[..8], &(len as u64).to_le_bytes());
        assert_eq!(&encoded[8..], expected);
        assert_eq!(
            decode::<Vec<i8>>(&encoded, DecodeLimits::MESSAGE).unwrap(),
            values
        );
    }
}
