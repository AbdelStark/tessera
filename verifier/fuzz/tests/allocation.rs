// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The memory `tv-codec` reserves, measured and capped by a counting allocator (RFC-0003 'Testing
//! Strategy'). A global allocator cannot live in the verifier's workspace, which forbids unsafe
//! code; here it is the `cap` crate's, wrapping the system allocator.
//!
//! The allocator's counters and limit are global, so the cases run one after another in a single
//! test.

use std::alloc::System;

use cap::Cap;
use tv_codec::{Cursor, DecodeError, DecodeLimits, ErrorCode, decode, size};

#[global_allocator]
static ALLOCATOR: Cap<System> = Cap::new(System, usize::MAX);

/// Runs `f` with at most `budget` bytes allocatable beyond those allocated now, and returns its
/// result and the bytes it allocated in all.
fn within<T>(budget: usize, f: impl FnOnce() -> T) -> (T, usize) {
    let before = ALLOCATOR.total_allocated();
    ALLOCATOR
        .set_limit(ALLOCATOR.allocated() + budget)
        .expect("a limit above the memory in use");
    let result = f();
    ALLOCATOR.set_limit(usize::MAX).expect("no limit");
    (result, ALLOCATOR.total_allocated() - before)
}

fn le(value: u64) -> [u8; 8] {
    value.to_le_bytes()
}

/// No limit but the input's: only the bytes-left check stands between a hostile length and an
/// allocation.
const UNLIMITED: DecodeLimits = DecodeLimits {
    max_total_bytes: u64::MAX,
    max_vec_len: u64::MAX,
    max_depth: 32,
};

/// A value of nested sequences: `Vec<Vec<…<Bytes>>>`.
enum Node<'a> {
    List(#[allow(dead_code)] Vec<Node<'a>>),
    Leaf(#[allow(dead_code)] &'a [u8]),
}

fn nested<'a>(c: &mut Cursor<'a>, levels: usize) -> Result<Node<'a>, DecodeError> {
    match levels {
        0 => c.bytes().map(Node::Leaf),
        _ => c
            .vec(size::SEQUENCE, |c| nested(c, levels - 1))
            .map(Node::List),
    }
}

fn memory_follows_what_is_decoded_not_what_is_declared() {
    // A 2^63 length with 16 bytes left, and maps of u64::MAX entries: rejected before any
    // allocation (RFC-0003 'Testing Strategy', hostile lengths).
    let hostile = [&le(1 << 63)[..], &[0; 16]].concat();
    let (result, allocated) = within(0, || {
        decode(&hostile, &UNLIMITED, |c| c.vec(8, Cursor::u64))
    });
    assert_eq!(
        result.map(|v| v.len()),
        Err(DecodeError::new(ErrorCode::Truncated, 0))
    );
    assert_eq!(allocated, 0);
    let hostile_map = [&le(u64::MAX)[..], &[0; 16]].concat();
    let (result, allocated) = within(0, || {
        decode(&hostile_map, &UNLIMITED, |c| {
            c.map(1, 1, Cursor::u8, Cursor::u8).map(|m| m.len())
        })
    });
    assert_eq!(result, Err(DecodeError::new(ErrorCode::Truncated, 0)));
    assert_eq!(allocated, 0);

    // A sequence never has room for more values than its length (INV-TCE-3): three u64s take 24
    // bytes, in one allocation.
    let three = [le(3), le(1), le(2), le(3)].concat();
    let (result, allocated) = within(64, || {
        decode(&three, &DecodeLimits::MESSAGE, |c| {
            c.vec(size::U64, Cursor::u64)
        })
    });
    assert_eq!(result, Ok(vec![1, 2, 3]));
    assert_eq!(allocated, 24);

    // Thirty sequences open inside one another, each declaring as many elements as the bytes left
    // allow, around a `Bytes` that is cut short, in 16 MiB. Reserving each declared length up
    // front would take about 30 · 4 · 16 MiB; room that grows with the values decoded keeps
    // every open level near its first 64 KiB, and the cut is found.
    let levels = 30;
    let total = 16 << 20;
    let mut deep = Vec::with_capacity(total);
    for _ in 0..levels {
        let left = (total - deep.len() - 8) as u64;
        deep.extend(le(left / 8));
    }
    deep.extend(le(total as u64));
    deep.resize(total, 0);
    let (result, allocated) = within(4 << 20, || {
        decode(&deep, &DecodeLimits::MESSAGE, |c| nested(c, levels)).map(|_| ())
    });
    assert_eq!(
        result,
        Err(DecodeError::new(ErrorCode::Truncated, 8 * levels as u64))
    );
    assert!(allocated <= 4 << 20, "{allocated} bytes");

    // An honest list of 100,000 empty `Bytes`: 800 KB of input whose values take 1.6 MB. Short of
    // memory, the reader reports its own failure, ERR-RES-MEMORY, not a fault of the input; with
    // enough, the same input decodes.
    let count = 100_000;
    let list = [le(count).to_vec(), le(0).repeat(count as usize)].concat();
    let read = || {
        decode(&list, &DecodeLimits::MESSAGE, |c| {
            c.vec(size::SEQUENCE, Cursor::bytes).map(|v| v.len())
        })
    };
    let (short, _) = within(256 << 10, read);
    let error = short.expect_err("1.6 MB of values in 256 KiB");
    assert_eq!(error, DecodeError::new(ErrorCode::Memory, 0));
    assert!(!error.code().is_input_fault());
    let (enough, _) = within(8 << 20, read);
    assert_eq!(enough, Ok(count as usize));
}

/// Runs on the process's only thread (`harness = false`): libtest would keep a thread of its own,
/// whose allocations the process-wide cap counts against this test's budget (#501).
fn main() {
    memory_follows_what_is_decoded_not_what_is_declared();
    println!("memory_follows_what_is_decoded_not_what_is_declared: ok");
}
