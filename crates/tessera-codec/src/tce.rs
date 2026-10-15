// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The Tessera Canonical Encoding (RFC-0003 'Primitive types', 'Composite types'): each value
//! has exactly one encoding, and a decoder accepts nothing else.
//!
//! Decoding reads a byte slice through a [`Reader`], which enforces [`DecodeLimits`] and checks
//! every declared length against the bytes that remain before allocating (INV-TCE-3).

use core::fmt;
use std::error::Error;

use crate::error::{ByteOffset, ErrorCode, PublicContext, TesseraError};

/// Where encoded bytes go: a buffer, a file, a hasher.
pub trait Sink {
    /// Appends `bytes`.
    fn put(&mut self, bytes: &[u8]);
}

impl Sink for Vec<u8> {
    fn put(&mut self, bytes: &[u8]) {
        self.extend_from_slice(bytes);
    }
}

/// A value with a TCE encoding.
pub trait Encode {
    /// Appends the encoding of `self` to `out`.
    fn encode_to<S: Sink + ?Sized>(&self, out: &mut S);

    /// Appends the encodings of `items` in order, with no length. Byte types override this with
    /// a single copy.
    fn encode_slice<S: Sink + ?Sized>(items: &[Self], out: &mut S)
    where
        Self: Sized,
    {
        for item in items {
            item.encode_to(out);
        }
    }
}

/// The encoding of `value`.
pub fn encode<T: Encode + ?Sized>(value: &T) -> Vec<u8> {
    let mut out = Vec::new();
    value.encode_to(&mut out);
    out
}

/// Bounds a decoder enforces before it allocates (RFC-0003 'Limits').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DecodeLimits {
    /// The most bytes the whole input may hold.
    pub max_total_bytes: u64,
    /// The largest length a single `Vec` or map may declare.
    pub max_vec_len: u64,
    /// The deepest nesting of composite values: every struct, enum (with `Option`), tuple, map
    /// entry, array, `Vec` (and so `Bytes`), `Ascii`, `Rational` and `Fp2` is one level. The
    /// reader never goes deeper than [`DEPTH_CEILING`], whatever this says.
    pub max_depth: u16,
}

impl DecodeLimits {
    /// RFC-0003's defaults for verifier messages: 8 GiB in all, 2^34 elements in a sequence, 32
    /// levels of nesting.
    pub const MESSAGE: Self = Self {
        max_total_bytes: 8 << 30,
        max_vec_len: 1 << 34,
        max_depth: 32,
    };

    /// RFC-0003's defaults for `IntegerModel`: 1 TiB in all, the message defaults otherwise.
    pub const INTEGER_MODEL: Self = Self {
        max_total_bytes: 1 << 40,
        ..Self::MESSAGE
    };
}

/// The deepest nesting any reader accepts. Decoding recurses once per level, and a deeper limit
/// could exhaust a thread's stack before the input runs out.
pub const DEPTH_CEILING: u16 = 256;

/// How many elements of `T` to reserve for a sequence that declares `n`: never more memory than
/// the bytes left in the input. Reserving `n` outright would let an input that fails at its
/// first element claim `n · size_of::<T>()` bytes, which can be thousands of times the input
/// when `T` is much larger in memory than its smallest encoding.
pub(crate) fn reservation<T>(n: usize, remaining: usize) -> usize {
    n.min(remaining / size_of::<T>().max(1))
}

/// Why decoding failed, and where: `ERR-CODEC-TRUNCATED`, `ERR-CODEC-NONCANONICAL` or
/// `ERR-CODEC-LIMIT`, at the offset of the value at fault.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeError {
    /// The stable code.
    pub code: ErrorCode,
    /// The offset, from the start of the input, of the value that failed.
    pub offset: u64,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at byte {}", self.code, self.offset)
    }
}

impl Error for DecodeError {}

impl From<DecodeError> for TesseraError {
    fn from(error: DecodeError) -> Self {
        TesseraError::new(error.code).with_context(PublicContext {
            offset: Some(ByteOffset(error.offset)),
            ..PublicContext::default()
        })
    }
}

/// A cursor over an input being decoded.
pub struct Reader<'a> {
    input: &'a [u8],
    pos: usize,
    limits: DecodeLimits,
    depth: u16,
}

/// Shows where the reader is, never what it reads: the input may be secret-class data (05 §3).
impl fmt::Debug for Reader<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Reader")
            .field("len", &self.input.len())
            .field("pos", &self.pos)
            .field("depth", &self.depth)
            .field("limits", &self.limits)
            .finish()
    }
}

impl<'a> Reader<'a> {
    /// A reader at the start of `input`, which must fit `limits.max_total_bytes`.
    pub fn new(input: &'a [u8], limits: DecodeLimits) -> Result<Self, DecodeError> {
        let reader = Self {
            input,
            pos: 0,
            limits,
            depth: 0,
        };
        if input.len() as u64 > limits.max_total_bytes {
            return Err(reader.error(ErrorCode::CodecLimit, 0));
        }
        Ok(reader)
    }

    /// The offset of the next byte.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// The bytes not yet read.
    pub fn remaining(&self) -> usize {
        self.input.len() - self.pos
    }

    /// The limits in force.
    pub fn limits(&self) -> DecodeLimits {
        self.limits
    }

    fn error(&self, code: ErrorCode, at: usize) -> DecodeError {
        DecodeError {
            code,
            offset: at as u64,
        }
    }

    /// `ERR-CODEC-NONCANONICAL` for the value that starts at `at`.
    pub fn noncanonical(&self, at: usize) -> DecodeError {
        self.error(ErrorCode::CodecNoncanonical, at)
    }

    /// `ERR-CODEC-LIMIT` for the value that starts at `at`.
    pub fn limit(&self, at: usize) -> DecodeError {
        self.error(ErrorCode::CodecLimit, at)
    }

    /// `ERR-CODEC-TRUNCATED` for the value that starts at `at`.
    pub fn truncated(&self, at: usize) -> DecodeError {
        self.error(ErrorCode::CodecTruncated, at)
    }

    /// Fails with `ERR-CODEC-TRUNCATED`, for the value at `at`, unless `n` bytes remain.
    pub fn require(&self, n: usize, at: usize) -> Result<(), DecodeError> {
        if n > self.remaining() {
            return Err(self.truncated(at));
        }
        Ok(())
    }

    /// The next `n` bytes.
    pub fn take(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        self.require(n, self.pos)?;
        let bytes = &self.input[self.pos..self.pos + n];
        self.pos += n;
        Ok(bytes)
    }

    /// The next `N` bytes, as an array.
    pub fn take_array<const N: usize>(&mut self) -> Result<[u8; N], DecodeError> {
        let mut array = [0u8; N];
        array.copy_from_slice(self.take(N)?);
        Ok(array)
    }

    /// The bytes read since offset `start`.
    pub fn since(&self, start: usize) -> &'a [u8] {
        &self.input[start..self.pos]
    }

    /// Enters a composite value: `ERR-CODEC-LIMIT` beyond `max_depth` or [`DEPTH_CEILING`].
    pub fn enter(&mut self) -> Result<(), DecodeError> {
        if self.depth >= self.limits.max_depth.min(DEPTH_CEILING) {
            return Err(self.limit(self.pos));
        }
        self.depth += 1;
        Ok(())
    }

    /// Leaves the composite value entered last.
    pub fn leave(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    /// The length prefix of a sequence starting here. In order, as RFC-0003 'Limits' fixes it:
    /// `ERR-CODEC-LIMIT` above `max_vec_len`, then `ERR-CODEC-TRUNCATED` when the elements, at
    /// `min_size` bytes each, cannot fit in what remains. Both checks run before any element is
    /// read or anything is allocated (INV-TCE-3).
    pub fn sequence_len(&mut self, min_size: usize) -> Result<usize, DecodeError> {
        let at = self.pos;
        let declared = u64::decode_from(self)?;
        if declared > self.limits.max_vec_len {
            return Err(self.limit(at));
        }
        // A length beyond `usize`, possible within `max_vec_len` where `usize` has 32 bits,
        // cannot fit in what remains either.
        let n = usize::try_from(declared).map_err(|_| self.truncated(at))?;
        let needed = n.checked_mul(min_size).ok_or_else(|| self.truncated(at))?;
        self.require(needed, at)?;
        Ok(n)
    }

    /// Fails with `ERR-CODEC-NONCANONICAL` unless the whole input was read (INV-TCE-4).
    pub fn finish(self) -> Result<(), DecodeError> {
        if self.pos != self.input.len() {
            return Err(self.noncanonical(self.pos));
        }
        Ok(())
    }
}

/// A value with a TCE decoding.
pub trait Decode: Sized {
    /// The fewest bytes any value encodes to. Length checks use it before allocating.
    const MIN_SIZE: usize;

    /// Decodes one value.
    fn decode_from(reader: &mut Reader<'_>) -> Result<Self, DecodeError>;

    /// Decodes `n` values onto `out`. Byte types override this with a single copy.
    fn decode_many(
        reader: &mut Reader<'_>,
        n: usize,
        out: &mut Vec<Self>,
    ) -> Result<(), DecodeError> {
        for _ in 0..n {
            out.push(Self::decode_from(reader)?);
        }
        Ok(())
    }

    /// Decodes `N` values. Byte types override this with a single copy.
    fn decode_array<const N: usize>(reader: &mut Reader<'_>) -> Result<[Self; N], DecodeError> {
        let at = reader.position();
        let needed = N
            .checked_mul(Self::MIN_SIZE)
            .ok_or_else(|| reader.truncated(at))?;
        reader.require(needed, at)?;
        let mut items = Vec::with_capacity(reservation::<Self>(N, reader.remaining()));
        Self::decode_many(reader, N, &mut items)?;
        match items.try_into() {
            Ok(array) => Ok(array),
            Err(_) => unreachable!("decode_many returned {N} values"),
        }
    }
}

/// Decodes one value that fills `input` exactly.
pub fn decode<T: Decode>(input: &[u8], limits: DecodeLimits) -> Result<T, DecodeError> {
    let mut reader = Reader::new(input, limits)?;
    let value = T::decode_from(&mut reader)?;
    reader.finish()?;
    Ok(value)
}

// ----- Primitive types ------------------------------------------------------------------------

macro_rules! integers {
    ($($t:ty),*) => {$(
        /// Little-endian, fixed width.
        impl Encode for $t {
            fn encode_to<S: Sink + ?Sized>(&self, out: &mut S) {
                out.put(&self.to_le_bytes());
            }
        }

        impl Decode for $t {
            const MIN_SIZE: usize = size_of::<$t>();

            fn decode_from(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
                Ok(<$t>::from_le_bytes(reader.take_array()?))
            }
        }
    )*};
}

integers!(u16, u32, u64, i16, i32, i64);

/// One byte. Sequences of bytes encode and decode with a single copy.
impl Encode for u8 {
    fn encode_to<S: Sink + ?Sized>(&self, out: &mut S) {
        out.put(&[*self]);
    }

    fn encode_slice<S: Sink + ?Sized>(items: &[Self], out: &mut S) {
        out.put(items);
    }
}

impl Decode for u8 {
    const MIN_SIZE: usize = 1;

    fn decode_from(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        Ok(reader.take(1)?[0])
    }

    fn decode_many(
        reader: &mut Reader<'_>,
        n: usize,
        out: &mut Vec<Self>,
    ) -> Result<(), DecodeError> {
        out.extend_from_slice(reader.take(n)?);
        Ok(())
    }

    fn decode_array<const N: usize>(reader: &mut Reader<'_>) -> Result<[Self; N], DecodeError> {
        reader.take_array()
    }
}

/// One byte, two's complement.
impl Encode for i8 {
    fn encode_to<S: Sink + ?Sized>(&self, out: &mut S) {
        out.put(&self.to_le_bytes());
    }

    fn encode_slice<S: Sink + ?Sized>(items: &[Self], out: &mut S) {
        let bytes: Vec<u8> = items.iter().map(|x| x.to_le_bytes()[0]).collect();
        out.put(&bytes);
    }
}

impl Decode for i8 {
    const MIN_SIZE: usize = 1;

    fn decode_from(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        Ok(i8::from_le_bytes(reader.take_array()?))
    }

    fn decode_many(
        reader: &mut Reader<'_>,
        n: usize,
        out: &mut Vec<Self>,
    ) -> Result<(), DecodeError> {
        out.extend(reader.take(n)?.iter().map(|&b| i8::from_le_bytes([b])));
        Ok(())
    }
}

/// One byte, `0x00` or `0x01`.
impl Encode for bool {
    fn encode_to<S: Sink + ?Sized>(&self, out: &mut S) {
        out.put(&[u8::from(*self)]);
    }
}

impl Decode for bool {
    const MIN_SIZE: usize = 1;

    fn decode_from(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        let at = reader.position();
        match reader.take(1)?[0] {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(reader.noncanonical(at)),
        }
    }
}

// ----- Composite types ------------------------------------------------------------------------

/// `N` encodings, concatenated, with no length.
impl<T: Encode, const N: usize> Encode for [T; N] {
    fn encode_to<S: Sink + ?Sized>(&self, out: &mut S) {
        T::encode_slice(self, out);
    }
}

impl<T: Decode, const N: usize> Decode for [T; N] {
    const MIN_SIZE: usize = N * T::MIN_SIZE;

    fn decode_from(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        reader.enter()?;
        let array = T::decode_array(reader)?;
        reader.leave();
        Ok(array)
    }
}

/// A `u64` length `n`, then `n` encodings.
impl<T: Encode> Encode for Vec<T> {
    fn encode_to<S: Sink + ?Sized>(&self, out: &mut S) {
        (self.len() as u64).encode_to(out);
        T::encode_slice(self, out);
    }
}

impl<T: Decode> Decode for Vec<T> {
    const MIN_SIZE: usize = 8;

    fn decode_from(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        const {
            assert!(
                T::MIN_SIZE > 0,
                "a TCE sequence needs elements of at least one byte: empty elements carry only \
                 a count, and decoding them would cost time without consuming input"
            );
        }
        reader.enter()?;
        let n = reader.sequence_len(T::MIN_SIZE)?;
        let mut items = Vec::with_capacity(reservation::<T>(n, reader.remaining()));
        T::decode_many(reader, n, &mut items)?;
        reader.leave();
        Ok(items)
    }
}

/// The enum `{ None, Some(T) }`: a `u16` tag 0 or 1, then the payload. It means a value that is
/// semantically absent, never a field a later version may add (RFC-0003 'No optional fields').
impl<T: Encode> Encode for Option<T> {
    fn encode_to<S: Sink + ?Sized>(&self, out: &mut S) {
        match self {
            None => 0u16.encode_to(out),
            Some(value) => {
                1u16.encode_to(out);
                value.encode_to(out);
            }
        }
    }
}

impl<T: Decode> Decode for Option<T> {
    const MIN_SIZE: usize = 2;

    fn decode_from(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        reader.enter()?;
        let at = reader.position();
        let value = match u16::decode_from(reader)? {
            0 => None,
            1 => Some(T::decode_from(reader)?),
            _ => return Err(reader.noncanonical(at)),
        };
        reader.leave();
        Ok(value)
    }
}

/// The fields in order, like a struct.
impl<A: Encode, B: Encode> Encode for (A, B) {
    fn encode_to<S: Sink + ?Sized>(&self, out: &mut S) {
        self.0.encode_to(out);
        self.1.encode_to(out);
    }
}

impl<A: Decode, B: Decode> Decode for (A, B) {
    const MIN_SIZE: usize = A::MIN_SIZE + B::MIN_SIZE;

    fn decode_from(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        reader.enter()?;
        let value = (A::decode_from(reader)?, B::decode_from(reader)?);
        reader.leave();
        Ok(value)
    }
}
