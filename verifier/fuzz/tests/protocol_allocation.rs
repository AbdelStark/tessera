// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Packet length checks and allocation failures at the actual RFC-0036 trust boundary (SP-8).
use cap::Cap;
use std::alloc::System;
use tv_codec::{DecodeError, DecodeLimits, ErrorCode, decode};
use tv_core::protocol::{ColumnQueries, Round2Response, Round3Response};

#[global_allocator]
static ALLOCATOR: Cap<System> = Cap::new(System, usize::MAX);

fn within<T>(budget: usize, f: impl FnOnce() -> T) -> (T, usize) {
    let before = ALLOCATOR.total_allocated();
    ALLOCATOR.set_limit(ALLOCATOR.allocated() + budget).unwrap();
    let result = f();
    ALLOCATOR.set_limit(usize::MAX).unwrap();
    (result, ALLOCATOR.total_allocated() - before)
}

#[test]
fn packets_allocate_from_decoded_bytes_and_report_memory_as_a_reader_failure() {
    let unlimited = DecodeLimits {
        max_total_bytes: u64::MAX,
        max_vec_len: u64::MAX,
        max_depth: 32,
    };
    let hostile = [&u64::MAX.to_le_bytes()[..], &[0; 16]].concat();
    for read in [
        |b: &[u8], l: &DecodeLimits| decode(b, l, Round2Response::decode).map(|_| ()),
        |b: &[u8], l: &DecodeLimits| decode(b, l, Round3Response::decode).map(|_| ()),
        |b: &[u8], l: &DecodeLimits| decode(b, l, ColumnQueries::decode).map(|_| ()),
    ] {
        let (result, bytes) = within(0, || read(&hostile, &unlimited));
        assert_eq!(result, Err(DecodeError::new(ErrorCode::Truncated, 0)));
        assert_eq!(bytes, 0);
    }

    // Empty LogUp (16 bytes) and trees (48: four empty lists and an empty merge, RFC-0054),
    // then 100,000 minimal stages and three empty vectors. The payload fits, but its decoded
    // stage structs require more space than their wire representation.
    let count = 100_000u64;
    let mut bytes = vec![0; 64];
    bytes.extend(count.to_le_bytes());
    bytes.extend([0u8; 26].repeat(count as usize));
    bytes.extend([0; 24]);
    let read = || decode(&bytes, &DecodeLimits::MESSAGE, Round2Response::decode);
    let (short, allocated) = within(256 << 10, read);
    assert_eq!(short, Err(DecodeError::new(ErrorCode::Memory, 64)));
    // Total includes freed reservations while the enforced cap bounds simultaneous memory.
    assert!(allocated <= 512 << 10);
    let (enough, allocated) = within(16 << 20, read);
    let decoded = enough.unwrap();
    assert_eq!(decoded.stages.len(), count as usize);
    assert!(allocated <= 32 << 20);
    let (encoded, allocated) = within(0, || decoded.to_tce());
    assert_eq!(encoded, Err(ErrorCode::Memory));
    assert_eq!(allocated, 0);
    let (encoded, _) = within(4 << 20, || decoded.to_tce());
    assert_eq!(encoded.unwrap(), bytes);

    // A nested column declares the entire remaining byte count as field elements.
    // The outer Vec may reserve a small chunk; the impossible inner count reserves nothing.
    let input = [&1u64.to_le_bytes()[..], &u64::MAX.to_le_bytes(), &[0; 24]].concat();
    let (result, allocated) = within(1024, || decode(&input, &unlimited, Round3Response::decode));
    assert_eq!(result, Err(DecodeError::new(ErrorCode::Truncated, 8)));
    assert!(allocated <= 1024);
}
