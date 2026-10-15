// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Hostile lengths cost nothing (RFC-0003 'Limits', INV-TCE-3), as a counting allocator sees it.
//! `allocation-counter` counts the allocations of the current thread only, so tests running in
//! parallel do not disturb each other.
//!
//! A decoder checks a declared length against `max_vec_len`, then against the bytes that
//! remain, before it allocates; a sequence that fails at its first element has reserved no
//! more memory than the input holds; and streaming an envelope through its digest holds one
//! 64 KiB chunk, whatever length the envelope declares.

use std::io::{Cursor, sink};

use allocation_counter::{AllocationInfo, measure};
use tessera_codec::envelope::{self, open, seal, verify};
use tessera_codec::error::ErrorCode;
use tessera_codec::schema;
use tessera_codec::{Decode, DecodeError, DecodeLimits, decode};

/// The outcome of decoding `input` as `T`, and what decoding allocated.
fn decoding<T: Decode>(
    input: &[u8],
    limits: DecodeLimits,
) -> (Result<T, DecodeError>, AllocationInfo) {
    let mut outcome = None;
    let info = measure(|| outcome = Some(decode::<T>(input, limits)));
    (outcome.expect("measured"), info)
}

fn failure(code: ErrorCode, offset: u64) -> DecodeError {
    DecodeError { code, offset }
}

/// A length prefix followed by `left` zero bytes.
fn declaring(len: u64, left: usize) -> Vec<u8> {
    [len.to_le_bytes().to_vec(), vec![0; left]].concat()
}

#[test]
fn a_hostile_length_allocates_nothing() {
    // 2^63 elements declared, 16 bytes left: over max_vec_len.
    let input = declaring(1 << 63, 16);
    let (outcome, info) = decoding::<Vec<u64>>(&input, DecodeLimits::MESSAGE);
    assert_eq!(outcome, Err(failure(ErrorCode::CodecLimit, 0)));
    assert_eq!(info.count_total, 0, "{info:?}");
    // The same bytes as the payload of an envelope, whose digest is checked first.
    let raw: [u8; 24] = input.try_into().expect("24 bytes");
    let sealed = seal(&schema::CLAIM_LIST, &raw);
    let opened = measure(|| {
        let outcome = open::<Vec<u64>>(&sealed, &schema::CLAIM_LIST, DecodeLimits::MESSAGE);
        let at = envelope::HEADER_LEN as u64;
        assert_eq!(outcome, Err(failure(ErrorCode::CodecLimit, at)));
    });
    assert_eq!(opened.count_total, 0, "{opened:?}");
}

#[test]
fn a_length_the_input_cannot_hold_allocates_nothing() {
    // Within max_vec_len, so it is the bytes left that refuse it.
    let unlimited = DecodeLimits {
        max_vec_len: u64::MAX,
        ..DecodeLimits::MESSAGE
    };
    let input = declaring(1 << 63, 16);
    let (outcome, info) = decoding::<Vec<u64>>(&input, unlimited);
    assert_eq!(outcome, Err(failure(ErrorCode::CodecTruncated, 0)));
    assert_eq!(info.count_total, 0, "{info:?}");
    // 2^16 pages of 4 KiB declared, 2^16 bytes left: room for 16.
    let input = declaring(1 << 16, 1 << 16);
    let (outcome, info) = decoding::<Vec<[u8; 4096]>>(&input, DecodeLimits::MESSAGE);
    assert_eq!(outcome, Err(failure(ErrorCode::CodecTruncated, 0)));
    assert_eq!(info.count_total, 0, "{info:?}");
}

#[test]
fn a_sequence_failing_at_its_first_element_holds_no_more_than_its_input() {
    // An `Option<[u64; 512]>` takes 4,104 bytes in memory and 2 bytes at least on the wire. With
    // 2^15 of them declared over 2^16 bytes, a reservation sized by the declared length would
    // take 2,000 times the input; the first tag, 0x0202, is invalid.
    let mut input = declaring(1 << 15, 1 << 16);
    input[8..].fill(2);
    let (outcome, info) = decoding::<Vec<Option<[u64; 512]>>>(&input, DecodeLimits::MESSAGE);
    assert_eq!(outcome, Err(failure(ErrorCode::CodecNoncanonical, 8)));
    assert!(info.bytes_max > 0, "nothing was reserved: {info:?}");
    assert!(info.bytes_max <= input.len() as u64, "{info:?}");
}

#[test]
fn streaming_holds_one_chunk_whatever_the_length() {
    let chunk = 64 << 10;
    // Room for the chunk, and for the reader's and writer's bookkeeping.
    let bound = chunk + 1024;
    let payload = vec![0xA5u8; 1 << 20];
    let sealed = seal(&schema::INTEGER_MODEL, &payload);
    let written = measure(|| {
        envelope::write(&mut sink(), &schema::INTEGER_MODEL, &payload).expect("a sink");
    });
    assert!(
        written.bytes_max <= bound,
        "writing in one piece: {written:?}"
    );
    // Written four bytes at a time, through the buffer.
    let words: Vec<u32> = (0..1 << 18).collect();
    let piecewise = measure(|| {
        envelope::write(&mut sink(), &schema::INTEGER_MODEL, &words).expect("a sink");
    });
    assert!(
        piecewise.bytes_max <= bound,
        "writing piecewise: {piecewise:?}"
    );
    let verified = measure(|| {
        verify(
            Cursor::new(&sealed),
            &schema::INTEGER_MODEL,
            DecodeLimits::INTEGER_MODEL,
        )
        .expect("a sealed envelope");
    });
    assert!(verified.bytes_max <= bound, "verifying: {verified:?}");
    // An envelope declaring 2^39 bytes of payload, of which 1 MiB is there.
    let mut declared = sealed.clone();
    declared[12..20].copy_from_slice(&(1u64 << 39).to_le_bytes());
    let truncated = measure(|| {
        let outcome = verify(
            Cursor::new(&declared),
            &schema::INTEGER_MODEL,
            DecodeLimits::INTEGER_MODEL,
        );
        assert!(matches!(
            outcome,
            Err(envelope::ReadError::Invalid(e)) if e == failure(ErrorCode::CodecTruncated, 12)
        ));
    });
    assert!(
        truncated.bytes_max <= bound,
        "a declared 2^39 bytes: {truncated:?}"
    );
}
