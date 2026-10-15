// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Hand-written TCE decoders under explicit decode limits (RFC-0003), written from the
//! specification without reference to the prover's codec, so that the two agree only because
//! both follow the specification (SP-6, RFC-0001).
//!
//! # Design
//!
//! There is no `Decode` trait and no derive macro. A [`Cursor`] walks the input with one method
//! per primitive type, and checks every sequence length against its element type's minimum
//! encoded size before any element is read. Composite types are combinator methods that take the
//! decoders of their parts as closures; each takes the value's depth level and releases it when
//! the value ends. A schema is a function that calls them in field order:
//!
//! ```
//! use tv_codec::{DecodeLimits, decode, size};
//!
//! // The tuple `(u8, Vec<u16>)`.
//! let bytes = [9, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0];
//! let value = decode(&bytes, &DecodeLimits::MESSAGE, |c| {
//!     c.nested(|c| Ok((c.u8()?, c.vec(size::U16, |c| c.u16())?)))
//! })?;
//! assert_eq!(value, (9, vec![1, 2]));
//! # Ok::<(), tv_codec::DecodeError>(())
//! ```
//!
//! [`Envelope::open`] checks an artifact envelope and its digest. The caller supplies the digest
//! function, BLAKE3 `derive_key` with context [`DIGEST_CONTEXT`], so this crate carries no hash
//! implementation; the verifier's hashing crate supplies it.
//!
//! # Specification
//!
//! What this crate rejects, and in which order, written down from RFC-0003 ('Primitive types',
//! 'Composite types', 'Envelope', 'Limits', 'Invariants', 'Failure modes') before the code. The
//! tests check every item against `spec/vectors/codec/`.
//!
//! ## Types
//!
//! | Type | Encoding | Minimum size | Depth |
//! |---|---|---|---|
//! | `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64` | little-endian, two's complement for signed, fixed width | its width | 0 |
//! | `bool` | one byte, `0x00` or `0x01` | 1 | 0 |
//! | `Fp` | `u64` below `P = 2^64 − 2^32 + 1` (RFC-0002) | 8 | 0 |
//! | `Fp2` | `Fp` `c0`, then `Fp` `c1` | 16 | 1 |
//! | `[T; N]` | `N` values of `T`, no length; `Digest` is `[u8; 32]` | `N · min_size(T)` | 1 |
//! | `Vec<T>` | `u64` length `n`, then `n` values of `T`; `Bytes` is `Vec<u8>` | 8 | 1 |
//! | `Ascii` | `Vec<u8>` with every byte in `0x20..=0x7E` | 8 | 1 |
//! | `Rational` | `i64` numerator, `u64` denominator | 16 | 1 |
//! | struct, tuple | fields in declaration order | sum of the fields | 1 |
//! | enum, `Option` | `u16` tag (the variant's index), then the variant's fields | 2 plus the smallest variant | 1 |
//! | map | `Vec<(K, V)>`, keys strictly ascending by their encodings | 8 | 1, and 1 per entry |
//!
//! ## Error codes
//!
//! Every error carries one [`ErrorCode`] and the offset, from the start of the input, of the
//! check that failed. For an envelope, offsets count from the start of the envelope.
//!
//! - `ERR-CODEC-TRUNCATED` ([`ErrorCode::Truncated`]):
//!   - a fixed-width value (an integer, `bool`, `Fp`, an enum tag or a sequence length) runs past
//!     the end of the input, at the value's first byte;
//!   - a sequence declares `n` elements and `n · min_size(T)` exceeds the bytes left after its
//!     length, at the sequence's first byte, its length;
//!   - an array `[T; N]` needs `N · min_size(T)` bytes and fewer are left, at the array's first
//!     byte;
//!   - an envelope header field is cut short, at the field's offset: 0, 4, 6, 8, 10 or 12;
//!   - fewer than `payload_len + 32` bytes follow an envelope header, at 12, the length that
//!     promised them.
//! - `ERR-CODEC-NONCANONICAL` ([`ErrorCode::Noncanonical`]):
//!   - a `bool` byte other than `0x00` or `0x01`, at the byte;
//!   - an `Fp` value `≥ P`, alone or as a coefficient of `Fp2`, at the value's first byte;
//!   - an enum tag, `Option`'s included, that names no variant, at the tag;
//!   - an `Ascii` byte outside `0x20..=0x7E`, at the byte;
//!   - a `Rational` whose denominator is 0, or whose numerator and denominator have a common
//!     factor (so zero is `0/1` only), at the rational's first byte;
//!   - a map key that is not strictly above the previous key in the byte order of their
//!     encodings (unsorted or duplicate), at the key's first byte;
//!   - bytes left after a whole input, or after the value in an envelope's payload (INV-TCE-4),
//!     at the first byte left over;
//!   - an envelope `reserved` field other than 0, at 10;
//!   - bytes after an envelope's digest, at `payload_len + 52`, the first of them.
//! - `ERR-CODEC-LIMIT` ([`ErrorCode::Limit`]):
//!   - a whole input longer than `max_total_bytes`, at 0;
//!   - an envelope whose size `payload_len + 52` exceeds `max_total_bytes`, or overflows a `u64`,
//!     at 12;
//!   - a sequence length above `max_vec_len`, at the sequence's first byte;
//!   - a composite value deeper than `min(max_depth, 256)`, at the value's first byte.
//! - `ERR-CODEC-MAGIC` ([`ErrorCode::Magic`]): an envelope magic other than `b"TSRA"`, at 0; an
//!   envelope schema id other than the one the reader expects, at 6.
//! - `ERR-CODEC-VERSION` ([`ErrorCode::Version`]): an envelope version other than 1, at 4; a schema
//!   version that is neither the version `v` the reader's writers emit nor `v − 1`, or is 0, at 8.
//! - `ERR-CODEC-DIGEST` ([`ErrorCode::Digest`]): an envelope digest other than the digest of the
//!   header and payload, at the digest, `20 + payload_len`.
//!
//! Two codes are faults of the reader, not of the input ([`ErrorCode::is_input_fault`]); a
//! verifier reports them as `Inconclusive` (04 §5):
//!
//! - `ERR-RES-MEMORY` ([`ErrorCode::Memory`]): memory for the values of a sequence cannot be
//!   reserved, at the sequence's first byte. Values can take more memory than their encodings,
//!   so this depends on the reader, not on the input alone.
//! - `ERR-INTERNAL` ([`ErrorCode::Internal`]): a schema decoder broke its contract, decoding a
//!   sequence or array element from fewer bytes than the minimum it declared, or from none, at
//!   the element's first byte. No input causes it under a correct schema.
//!
//! ## Order of checks
//!
//! So that independent decoders return the same code at the same offset for every input, a
//! decoder reports the first check that fails in this order:
//!
//! 1. `max_total_bytes`, before decoding starts. An envelope checks its declared size instead.
//! 2. The value, front to back. Each check runs as soon as the bytes it needs have been read:
//!    - a composite value takes its depth level before any of its bytes is read;
//!    - a fixed-width value is read whole, then checked: a field is incomplete before it is wrong;
//!    - a sequence (`Vec`, `Bytes`, `Ascii`, map) reads its length, checks it against
//!      `max_vec_len`, then checks `length · min_size(T)` against the bytes left, then reads its
//!      elements in order;
//!    - an array checks `N · min_size(T)` against the bytes left, then reads its elements;
//!    - an enum reads and checks its tag, then reads the variant's fields;
//!    - a `Rational` reads both integers, then checks the denominator, then the common factor;
//!    - `Ascii` checks its bytes in order, after its length checks;
//!    - a map entry checks its key's order as soon as the key is read, before its value.
//! 3. Bytes left over (INV-TCE-4).
//!
//! ## Depth
//!
//! Each value whose encoding holds other values adds one level while it is decoded: structs and
//! enums (so `Option`; an enum's variant fields add no level of their own), tuples and map
//! entries, arrays (so `Digest`), `Vec` (so `Bytes`) and maps, `Ascii`, `Rational` and `Fp2`.
//! Integers, `bool` and `Fp` add none. The outermost composite value is at depth 1, and a value
//! deeper than the limit is rejected, so `max_depth = 0` admits primitives only. A limit above
//! 256 counts as 256 ([`DecodeLimits::DEPTH_CEILING`]): decoding recurses once per level. A level
//! is released when its value ends, so siblings share a depth: sixteen `Option`s in a `Vec` all sit
//! at depth 2.
//!
//! ## Empty elements and allocation
//!
//! A sequence's element type encodes to at least one byte, so a sequence of `n` elements needs
//! at least `n` bytes. The sequence combinators take the element's minimum size from the caller
//! ([`size`]); a minimum of 0, which no schema may declare, is treated as 1 in the length check.
//! Every element must then consume at least its minimum, and at least one byte, or decoding
//! stops with `ERR-INTERNAL`: decoding time stays linear in the input even under a wrong
//! schema.
//!
//! A sequence never has room for more than its checked length, at most
//! `min(declared length, bytes left / min_size)` elements (INV-TCE-3). Its room grows with the
//! elements decoded: at most 64 KiB before the first, then doubling. Sequences open inside one
//! another, each declaring as many elements as the bytes left allow, so hold memory in
//! proportion to what they have decoded, not to what they declare. `Bytes` and `Ascii` borrow
//! from the input and allocate nothing.
//!
//! ## Envelope
//!
//! An envelope is `magic: [u8; 4]`, `envelope_version: u16`, `schema_id: u16`,
//! `schema_version: u16`, `reserved: u16`, `payload_len: u64` (20 bytes), the payload, then a
//! 32-byte digest, BLAKE3 `derive_key("tessera/v1/envelope")` over every preceding byte.
//! [`Envelope::open`] checks, in this order:
//!
//! 1. The header, field by field as it is read; a field cut short is `ERR-CODEC-TRUNCATED` at its
//!    offset:
//!    - `magic` (0): `b"TSRA"`, else `ERR-CODEC-MAGIC`;
//!    - `envelope_version` (4): 1, else `ERR-CODEC-VERSION`;
//!    - `schema_id` (6): the reader's, else `ERR-CODEC-MAGIC`;
//!    - `schema_version` (8): the reader's `v` or `v − 1`, never 0, else `ERR-CODEC-VERSION`;
//!    - `reserved` (10): 0, else `ERR-CODEC-NONCANONICAL`;
//!    - `payload_len` (12): `payload_len + 52 ≤ max_total_bytes`, else `ERR-CODEC-LIMIT`.
//! 2. The rest of the input: fewer than `payload_len + 32` bytes is `ERR-CODEC-TRUNCATED` at 12;
//!    more is `ERR-CODEC-NONCANONICAL` at `payload_len + 52`.
//! 3. The digest: a mismatch is `ERR-CODEC-DIGEST` at `20 + payload_len`. Nothing is decoded
//!    before it verifies.
//! 4. The payload ([`Envelope::decode_payload`]), as the schema version found and under the same
//!    limits: its value must fill it, and its errors carry their offsets plus 20.
//!
//! ## Offsets
//!
//! RFC-0003 'Limits' fixes every offset listed under 'Error codes': an error carries the offset
//! of the first byte its check reads. Some consequences:
//!
//! - a whole input above `max_total_bytes` fails at offset 0;
//! - the depth check comes before a composite's first byte is read, so a value that is both too
//!   deep and cut short is `ERR-CODEC-LIMIT`;
//! - `Fp2` is read coefficient by coefficient, not checked as a whole against the bytes left, so
//!   a bad `c0` with `c1` missing is `ERR-CODEC-NONCANONICAL` at `c0`;
//! - a map key's order is checked before its value is read.
//!
//! `spec/vectors/codec/noncanonical.json` pins the code and offset of every case, and the tests
//! check the offset at which every prefix of every accepted input is cut short.

#![forbid(unsafe_code)]
// Decoders read hostile bytes and must reject them with typed errors, never by panicking (RFC-0019
// 'Robustness'). Size arithmetic is checked (RFC-0019 'Robustness'), and conversions are explicit.
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::as_conversions
)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable,
        clippy::indexing_slicing,
        clippy::arithmetic_side_effects,
        clippy::as_conversions
    )
)]

mod cursor;
mod envelope;
mod error;
mod limits;
mod rational;
pub mod size;

pub use cursor::{Cursor, decode};
pub use envelope::{
    DIGEST_CONTEXT, DIGEST_LEN, ENVELOPE_VERSION, Envelope, EnvelopeHeader, Expected, HEADER_LEN,
    MAGIC,
};
pub use error::{DecodeError, ErrorCode};
pub use limits::DecodeLimits;
pub use rational::Rational;
