// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The cursor: primitive reads, depth levels, the sequence check and the combinators that build
//! composite values from closures.

use tv_field::{Fp, Fp2};

use crate::{DecodeError, DecodeLimits, ErrorCode, Rational, size};

/// A position in an input being decoded, under the limits that bound it.
///
/// The cursor only moves forward. Each method reads one value at the current position and either
/// advances past it or returns the error of the first check that failed, with its offset; after
/// an error the cursor's state is unspecified and decoding stops. A schema decoder is a function
/// that calls these methods in field order:
///
/// ```
/// use tv_codec::{Cursor, DecodeError, DecodeLimits, decode, size};
///
/// // `struct Entry { id: u32, tags: Vec<u16>, parent: Option<u64> }`
/// fn entry(c: &mut Cursor<'_>) -> Result<(u32, Vec<u16>, Option<u64>), DecodeError> {
///     c.nested(|c| Ok((c.u32()?, c.vec(size::U16, Cursor::u16)?, c.option(Cursor::u64)?)))
/// }
///
/// let bytes = [7, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0];
/// assert_eq!(decode(&bytes, &DecodeLimits::MESSAGE, entry)?, (7, vec![5], None));
/// # Ok::<(), DecodeError>(())
/// ```
pub struct Cursor<'a> {
    /// The bytes not yet read.
    rest: &'a [u8],
    /// The offset of `rest`'s first byte from the start of the outermost input.
    offset: u64,
    max_vec_len: u64,
    /// `min(max_depth, 256)`.
    max_depth: u16,
    /// The levels taken by the composite values being decoded.
    depth: u16,
}

impl<'a> Cursor<'a> {
    /// A cursor at the start of a whole input, after the check that comes before decoding: an
    /// input longer than `max_total_bytes` is `ERR-CODEC-LIMIT` at offset 0.
    pub fn new(input: &'a [u8], limits: &DecodeLimits) -> Result<Self, DecodeError> {
        if wide(input.len()) > limits.max_total_bytes {
            return Err(DecodeError::new(ErrorCode::Limit, 0));
        }
        Ok(Self::at(input, 0, limits))
    }

    /// A cursor over `input`, which starts `offset` bytes into the outermost input. It skips the
    /// total-size check, which an envelope makes on its declared size instead.
    pub(crate) fn at(input: &'a [u8], offset: u64, limits: &DecodeLimits) -> Self {
        Self {
            rest: input,
            offset,
            max_vec_len: limits.max_vec_len,
            max_depth: limits.effective_max_depth(),
            depth: 0,
        }
    }

    /// The offset of the next byte, from the start of the outermost input.
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// The number of bytes not yet read.
    pub fn remaining(&self) -> usize {
        self.rest.len()
    }

    /// The levels taken by the composite values being decoded.
    pub fn depth(&self) -> u16 {
        self.depth
    }

    /// `code` at the current position.
    fn error(&self, code: ErrorCode) -> DecodeError {
        DecodeError::new(code, self.offset)
    }

    /// Consumes `len` bytes, or fails with `ERR-CODEC-TRUNCATED` here.
    fn take_slice(&mut self, len: usize) -> Result<&'a [u8], DecodeError> {
        let (head, tail) = self
            .rest
            .split_at_checked(len)
            .ok_or(self.error(ErrorCode::Truncated))?;
        self.rest = tail;
        self.offset = self.offset.saturating_add(wide(len));
        Ok(head)
    }

    /// Consumes `N` bytes, or fails with `ERR-CODEC-TRUNCATED` here.
    pub(crate) fn take<const N: usize>(&mut self) -> Result<[u8; N], DecodeError> {
        let (head, tail) = self
            .rest
            .split_first_chunk::<N>()
            .ok_or(self.error(ErrorCode::Truncated))?;
        self.rest = tail;
        self.offset = self.offset.saturating_add(wide(N));
        Ok(*head)
    }

    /// Whether `count` values of at least `each` bytes can fit in the bytes left.
    fn fits(&self, count: u64, each: u64) -> bool {
        count
            .checked_mul(each)
            .is_some_and(|need| need <= wide(self.rest.len()))
    }

    // Primitives: fixed width, no depth, cut short is `ERR-CODEC-TRUNCATED` at their first byte.

    /// `u8`.
    pub fn u8(&mut self) -> Result<u8, DecodeError> {
        self.take().map(u8::from_le_bytes)
    }

    /// `u16`, little-endian.
    pub fn u16(&mut self) -> Result<u16, DecodeError> {
        self.take().map(u16::from_le_bytes)
    }

    /// `u32`, little-endian.
    pub fn u32(&mut self) -> Result<u32, DecodeError> {
        self.take().map(u32::from_le_bytes)
    }

    /// `u64`, little-endian.
    pub fn u64(&mut self) -> Result<u64, DecodeError> {
        self.take().map(u64::from_le_bytes)
    }

    /// `i8`, two's complement.
    pub fn i8(&mut self) -> Result<i8, DecodeError> {
        self.take().map(i8::from_le_bytes)
    }

    /// `i16`, two's complement, little-endian.
    pub fn i16(&mut self) -> Result<i16, DecodeError> {
        self.take().map(i16::from_le_bytes)
    }

    /// `i32`, two's complement, little-endian.
    pub fn i32(&mut self) -> Result<i32, DecodeError> {
        self.take().map(i32::from_le_bytes)
    }

    /// `i64`, two's complement, little-endian.
    pub fn i64(&mut self) -> Result<i64, DecodeError> {
        self.take().map(i64::from_le_bytes)
    }

    /// `bool`: `0x00` or `0x01`; any other byte is `ERR-CODEC-NONCANONICAL` at that byte.
    pub fn bool(&mut self) -> Result<bool, DecodeError> {
        let at = self.offset;
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(DecodeError::new(ErrorCode::Noncanonical, at)),
        }
    }

    /// `Fp`: a `u64` below `P`; a value `≥ P` is `ERR-CODEC-NONCANONICAL` at its first byte
    /// (RFC-0002 'Canonical encoding').
    pub fn fp(&mut self) -> Result<Fp, DecodeError> {
        let at = self.offset;
        let raw = self.u64()?;
        Fp::from_canonical(raw).ok_or(DecodeError::new(ErrorCode::Noncanonical, at))
    }

    // Depth.

    /// Takes one level for a composite value that starts here: `ERR-CODEC-LIMIT` here when every
    /// level of `min(max_depth, 256)` is taken. Pair it with [`leave`](Self::leave) when the value
    /// ends, or use [`nested`](Self::nested), which pairs them.
    pub fn enter(&mut self) -> Result<(), DecodeError> {
        if self.depth >= self.max_depth {
            return Err(self.error(ErrorCode::Limit));
        }
        self.depth = self.depth.saturating_add(1);
        Ok(())
    }

    /// Releases the level of a composite value that has ended.
    pub fn leave(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    /// A struct or tuple: `fields` decodes its fields in declaration order, one level deeper.
    pub fn nested<T>(
        &mut self,
        fields: impl FnOnce(&mut Self) -> Result<T, DecodeError>,
    ) -> Result<T, DecodeError> {
        self.enter()?;
        let value = fields(self)?;
        self.leave();
        Ok(value)
    }

    // Sequences.

    /// Reads a sequence's `u64` length and checks it before any element is read: above
    /// `max_vec_len` is `ERR-CODEC-LIMIT`, then `length · element_min` above the bytes left is
    /// `ERR-CODEC-TRUNCATED`, both at the length's offset. `element_min` is `min_size(T)` of the
    /// element type ([`size`]); 0, which no schema may declare (RFC-0003 'Limits', empty
    /// elements), counts as 1.
    ///
    /// The caller has taken the sequence's level; [`vec`](Self::vec) and the other sequence
    /// combinators do both.
    pub fn sequence_len(&mut self, element_min: u64) -> Result<usize, DecodeError> {
        let at = self.offset;
        let declared = self.u64()?;
        if declared > self.max_vec_len {
            return Err(DecodeError::new(ErrorCode::Limit, at));
        }
        if !self.fits(declared, element_min.max(1)) {
            return Err(DecodeError::new(ErrorCode::Truncated, at));
        }
        // At most one element per byte left, so it fits a `usize`.
        usize::try_from(declared).map_err(|_| DecodeError::new(ErrorCode::Truncated, at))
    }

    /// `Vec<T>`: a checked length, then that many values decoded by `element`, one level deeper.
    /// It reserves exactly the checked length (INV-TCE-3).
    pub fn vec<T>(
        &mut self,
        element_min: u64,
        mut element: impl FnMut(&mut Self) -> Result<T, DecodeError>,
    ) -> Result<Vec<T>, DecodeError> {
        self.nested(|c| {
            let at = c.offset;
            let len = c.sequence_len(element_min)?;
            let mut values = reserve(len, at)?;
            for _ in 0..len {
                values.push(element(c)?);
            }
            Ok(values)
        })
    }

    /// `Bytes`, that is `Vec<u8>`, borrowed from the input.
    pub fn bytes(&mut self) -> Result<&'a [u8], DecodeError> {
        self.nested(|c| {
            let len = c.sequence_len(size::U8)?;
            c.take_slice(len)
        })
    }

    /// `Ascii`, borrowed from the input: `Bytes` whose bytes are all in `0x20..=0x7E`; the first
    /// byte outside is `ERR-CODEC-NONCANONICAL` at that byte.
    pub fn ascii(&mut self) -> Result<&'a str, DecodeError> {
        self.nested(|c| {
            let len = c.sequence_len(size::U8)?;
            let start = c.offset;
            let text = c.take_slice(len)?;
            if let Some(index) = text.iter().position(|byte| !(0x20..=0x7E).contains(byte)) {
                let at = start.saturating_add(wide(index));
                return Err(DecodeError::new(ErrorCode::Noncanonical, at));
            }
            // Printable ASCII is UTF-8, so this cannot fail.
            core::str::from_utf8(text).map_err(|_| DecodeError::new(ErrorCode::Noncanonical, start))
        })
    }

    /// A map, `Vec<(K, V)>` with keys strictly ascending by their encodings: a checked length,
    /// then entries of one level each. A key that is not above the previous one in byte order is
    /// `ERR-CODEC-NONCANONICAL` at the key, checked before its value is read.
    pub fn map<K, V>(
        &mut self,
        key_min: u64,
        value_min: u64,
        mut key: impl FnMut(&mut Self) -> Result<K, DecodeError>,
        mut value: impl FnMut(&mut Self) -> Result<V, DecodeError>,
    ) -> Result<Vec<(K, V)>, DecodeError> {
        self.nested(|c| {
            let at = c.offset;
            let len = c.sequence_len(key_min.saturating_add(value_min))?;
            let mut entries = reserve(len, at)?;
            let mut previous: Option<&'a [u8]> = None;
            for _ in 0..len {
                let entry = c.nested(|c| {
                    let key_at = c.offset;
                    let before = c.rest;
                    let k = key(c)?;
                    let used = before.len().saturating_sub(c.rest.len());
                    let encoding = before
                        .get(..used)
                        .ok_or(DecodeError::new(ErrorCode::Noncanonical, key_at))?;
                    if previous.is_some_and(|previous| encoding <= previous) {
                        return Err(DecodeError::new(ErrorCode::Noncanonical, key_at));
                    }
                    previous = Some(encoding);
                    Ok((k, value(c)?))
                })?;
                entries.push(entry);
            }
            Ok(entries)
        })
    }

    // Fixed-size composites.

    /// `[T; len]`, as a vector of exactly `len` values decoded by `element`, one level deeper.
    /// The array is checked as a whole first: `len · element_min` above the bytes left is
    /// `ERR-CODEC-TRUNCATED` at its first byte.
    pub fn array<T>(
        &mut self,
        len: usize,
        element_min: u64,
        mut element: impl FnMut(&mut Self) -> Result<T, DecodeError>,
    ) -> Result<Vec<T>, DecodeError> {
        self.nested(|c| {
            if !c.fits(wide(len), element_min) {
                return Err(c.error(ErrorCode::Truncated));
            }
            let mut values = reserve(len, c.offset)?;
            for _ in 0..len {
                values.push(element(c)?);
            }
            Ok(values)
        })
    }

    /// `[u8; N]`, one level deeper: fewer than `N` bytes left is `ERR-CODEC-TRUNCATED` at its first
    /// byte.
    pub fn byte_array<const N: usize>(&mut self) -> Result<[u8; N], DecodeError> {
        self.nested(Self::take::<N>)
    }

    /// `Digest`, that is `[u8; 32]`.
    pub fn digest(&mut self) -> Result<[u8; 32], DecodeError> {
        self.byte_array()
    }

    /// `Fp2`: `c0`, then `c1`, one level deeper; a non-canonical coefficient is
    /// `ERR-CODEC-NONCANONICAL` at that coefficient.
    pub fn fp2(&mut self) -> Result<Fp2, DecodeError> {
        self.nested(|c| {
            let c0 = c.fp()?;
            let c1 = c.fp()?;
            Ok(Fp2::new(c0, c1))
        })
    }

    /// `Rational`: an `i64` numerator, then a `u64` denominator, one level deeper. A zero
    /// denominator, or a common factor, is `ERR-CODEC-NONCANONICAL` at the rational's first byte.
    pub fn rational(&mut self) -> Result<Rational, DecodeError> {
        self.nested(|c| {
            let at = c.offset;
            let num = c.i64()?;
            let den = c.u64()?;
            Rational::new(num, den).ok_or(DecodeError::new(ErrorCode::Noncanonical, at))
        })
    }

    // Tagged unions.

    /// An enum of `variants` variants, one level deeper: a `u16` tag, then `variant` decodes the
    /// fields of the variant it names. A tag of `variants` or more is `ERR-CODEC-NONCANONICAL` at
    /// the tag.
    pub fn enumeration<T>(
        &mut self,
        variants: u16,
        variant: impl FnOnce(&mut Self, u16) -> Result<T, DecodeError>,
    ) -> Result<T, DecodeError> {
        self.nested(|c| {
            let at = c.offset;
            let tag = c.u16()?;
            if tag >= variants {
                return Err(DecodeError::new(ErrorCode::Noncanonical, at));
            }
            variant(c, tag)
        })
    }

    /// `Option<T>`, the enum `{ None, Some(T) }`: tag 0, or tag 1 then the value `some` decodes.
    pub fn option<T>(
        &mut self,
        some: impl FnOnce(&mut Self) -> Result<T, DecodeError>,
    ) -> Result<Option<T>, DecodeError> {
        self.enumeration(2, |c, tag| match tag {
            0 => Ok(None),
            _ => some(c).map(Some),
        })
    }

    // The end.

    /// Ends a whole input, or an envelope's payload: a byte left over is
    /// `ERR-CODEC-NONCANONICAL` at that byte (INV-TCE-4).
    pub fn finish(self) -> Result<(), DecodeError> {
        if self.rest.is_empty() {
            Ok(())
        } else {
            Err(self.error(ErrorCode::Noncanonical))
        }
    }
}

/// Shows the position, never the bytes: an input may be a secret-class artifact (05 §3).
impl core::fmt::Debug for Cursor<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Cursor")
            .field("offset", &self.offset)
            .field("remaining", &self.rest.len())
            .field("depth", &self.depth)
            .field("max_depth", &self.max_depth)
            .field("max_vec_len", &self.max_vec_len)
            .finish()
    }
}

/// Decodes a whole input with `value`, which reads exactly one value: the input is checked
/// against `max_total_bytes` before decoding starts, and a byte left over is an error.
pub fn decode<'a, T>(
    input: &'a [u8],
    limits: &DecodeLimits,
    value: impl FnOnce(&mut Cursor<'a>) -> Result<T, DecodeError>,
) -> Result<T, DecodeError> {
    let mut cursor = Cursor::new(input, limits)?;
    let decoded = value(&mut cursor)?;
    cursor.finish()?;
    Ok(decoded)
}

/// An empty vector with room for exactly `len` values, which the caller checked against the
/// bytes left. Memory that cannot be reserved is `ERR-CODEC-LIMIT` at `at` rather than an abort.
fn reserve<T>(len: usize, at: u64) -> Result<Vec<T>, DecodeError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| DecodeError::new(ErrorCode::Limit, at))?;
    Ok(values)
}

/// A length as `u64`. Every supported target has 64-bit or smaller pointers, so this is exact.
pub(crate) fn wide(len: usize) -> u64 {
    u64::try_from(len).unwrap_or(u64::MAX)
}
