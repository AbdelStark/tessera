// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Minimum encoded sizes, `min_size(T)` in RFC-0003 'Limits', for the bytes-left checks of
//! sequences and arrays.
//!
//! A schema states its element's minimum when it decodes a sequence, for example
//! `cursor.vec(size::FP2, Cursor::fp2)`. The minimum of a struct or tuple is the [`sum`] of its
//! fields'; of an enum, [`TAG`] plus its smallest variant's fields; of an array,
//! [`array`](fn@array). Saturating arithmetic keeps an oversized minimum oversized: no input can
//! satisfy it.
//!
//! A minimum must be exact. One too large rejects honest inputs: the length check fails, or an
//! element consumes less than its minimum, `ERR-INTERNAL`. One too small only weakens the length
//! check, so that a malformed input fails at an element rather than at its length; no check can
//! see it at run time, and the schema's conformance vectors must.

/// `bool`.
pub const BOOL: u64 = 1;
/// `u8`.
pub const U8: u64 = 1;
/// `i8`.
pub const I8: u64 = 1;
/// `u16`.
pub const U16: u64 = 2;
/// `i16`.
pub const I16: u64 = 2;
/// `u32`.
pub const U32: u64 = 4;
/// `i32`.
pub const I32: u64 = 4;
/// `u64`.
pub const U64: u64 = 8;
/// `i64`.
pub const I64: u64 = 8;
/// `Fp`.
pub const FP: u64 = 8;
/// `Fp2`: two `Fp`.
pub const FP2: u64 = 16;
/// `Digest`, `[u8; 32]`.
pub const DIGEST: u64 = 32;
/// `Rational`: an `i64` and a `u64`.
pub const RATIONAL: u64 = 16;
/// Any sequence, `Vec`, `Bytes`, `Ascii` or map: an empty one is its `u64` length.
pub const SEQUENCE: u64 = 8;
/// An enum's tag, its whole size when its smallest variant has no field.
pub const TAG: u64 = 2;
/// `Option<T>`: the tag of `None`.
pub const OPTION: u64 = TAG;

/// `[T; len]` with `element = min_size(T)`.
pub const fn array(len: u64, element: u64) -> u64 {
    len.saturating_mul(element)
}

/// A struct or tuple with fields of these minimum sizes.
pub const fn sum(mut fields: &[u64]) -> u64 {
    let mut total: u64 = 0;
    while let [first, rest @ ..] = fields {
        total = total.saturating_add(*first);
        fields = rest;
    }
    total
}
