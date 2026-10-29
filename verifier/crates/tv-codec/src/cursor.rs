// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The cursor: primitive reads, depth levels, the sequence check and the combinators that build
//! composite values from closures.

use std::time::Instant;
use tv_field::{Fp, Fp2};

use crate::{DecodeError, DecodeLimits, ErrorCode, Rational, size};

/// A position in an input being decoded, under the limits that bound it.
///
/// The cursor only moves forward. Each method reads one value at the current position and either
/// advances past it or returns the error of the first check that failed, with its offset; after
/// an error the cursor's state is unspecified and decoding stops. A schema decoder is a function
/// that calls these methods in field order, and decodes every composite value through a
/// combinator, which takes the value's depth level and releases it when the value ends.
///
/// A cursor exists only inside [`decode`] and [`Envelope::decode_payload`](crate::Envelope),
/// which check, once the value is decoded, that no byte is left over (INV-TCE-4):
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
    deadline: Option<Instant>,
    work_left: u16,
    // A private clock keeps deadline conformance tests deterministic.
    now: fn() -> Instant,
}

impl<'a> Cursor<'a> {
    /// A cursor at the start of a whole input, after the check that comes before decoding: an
    /// input longer than `max_total_bytes` is `ERR-CODEC-LIMIT` at offset 0.
    pub(crate) fn new(input: &'a [u8], limits: &DecodeLimits) -> Result<Self, DecodeError> {
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
            deadline: None,
            work_left: 0,
            now: Instant::now,
        }
    }

    /// Use the reader's absolute cooperative deadline (RFC-0019, SP-8).
    /// A nested reader may shorten an existing deadline, never extend it.
    /// Subsequent reads/sequence steps poll at most every 4,096 operations; finishing
    /// always checks again. Expiry is `ERR-RES-TIME`, never an input fault.
    ///
    /// # Errors
    /// `Time` at the current offset if the deadline has already expired.
    pub fn set_deadline(&mut self, deadline: Instant) -> Result<(), DecodeError> {
        self.deadline = Some(
            self.deadline
                .map_or(deadline, |current| current.min(deadline)),
        );
        self.work_left = 0;
        self.check_time()
    }

    fn check_time(&self) -> Result<(), DecodeError> {
        if self
            .deadline
            .is_some_and(|deadline| (self.now)() >= deadline)
        {
            return Err(self.error(ErrorCode::Time));
        }
        Ok(())
    }

    fn step(&mut self) -> Result<(), DecodeError> {
        if self.deadline.is_some() {
            if self.work_left == 0 {
                self.check_time()?;
                self.work_left = 4096;
            }
            self.work_left = self.work_left.saturating_sub(1);
        }
        Ok(())
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
        self.step()?;
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
        self.step()?;
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
    /// level of `min(max_depth, 256)` is taken. Only [`nested`](Self::nested) calls it, paired
    /// with [`leave`](Self::leave), so that no schema can leave a level taken or skip one.
    fn enter(&mut self) -> Result<(), DecodeError> {
        if self.depth >= self.max_depth {
            return Err(self.error(ErrorCode::Limit));
        }
        self.depth = self.depth.saturating_add(1);
        Ok(())
    }

    /// Releases the level of a composite value that has ended.
    fn leave(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    /// A struct or tuple: `fields` decodes its fields in declaration order, one level deeper.
    /// Every combinator for a composite value goes through it.
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
    /// Call it inside [`nested`](Self::nested), which takes the sequence's level; [`vec`](Self::vec)
    /// and the other sequence combinators do both, and also bound the memory the elements take.
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
    ///
    /// The vector's room grows with the values decoded, up to the checked length (INV-TCE-3);
    /// memory that cannot be reserved is `ERR-RES-MEMORY` at the sequence. Each value must
    /// consume at least `element_min` bytes, and at least one: a value that consumes less shows a
    /// wrong minimum or an empty element type in the schema, `ERR-INTERNAL` at the value.
    pub fn vec<T>(
        &mut self,
        element_min: u64,
        element: impl FnMut(&mut Self) -> Result<T, DecodeError>,
    ) -> Result<Vec<T>, DecodeError> {
        self.nested(|c| {
            let at = c.offset;
            let len = c.sequence_len(element_min)?;
            c.elements(len, element_min, at, element)
        })
    }

    /// Decodes the `len` values of a sequence or an array that starts at `at`, each by `element`.
    ///
    /// - **Memory.** The vector never has room for more than `len` values (INV-TCE-3), and its
    ///   room grows with the values decoded: at most [`FIRST_RESERVATION`] bytes at first, then
    ///   doubling. Sequences open inside one another thus hold memory in proportion to what they
    ///   have decoded, not to the lengths they declare. Memory that cannot be reserved is
    ///   `ERR-RES-MEMORY` at `at`: the reader ran out, the input is not at fault.
    /// - **Each value's size.** A value that consumes fewer than `element_min` bytes, or none,
    ///   shows that the schema declared a wrong minimum or an empty element type, which RFC-0003
    ///   forbids: `ERR-INTERNAL` at the value. So every value consumes input, and decoding time
    ///   stays linear in it whatever the schema.
    fn elements<T>(
        &mut self,
        len: usize,
        element_min: u64,
        at: u64,
        mut element: impl FnMut(&mut Self) -> Result<T, DecodeError>,
    ) -> Result<Vec<T>, DecodeError> {
        let mut values = Vec::new();
        for _ in 0..len {
            self.step()?;
            make_room(&mut values, len, at)?;
            let start = self.offset;
            values.push(element(self)?);
            if self.offset.saturating_sub(start) < element_min.max(1) {
                return Err(DecodeError::new(ErrorCode::Internal, start));
            }
        }
        Ok(values)
    }

    /// `Bytes`, that is `Vec<u8>`, borrowed from the input.
    pub fn bytes(&mut self) -> Result<&'a [u8], DecodeError> {
        self.nested(|c| {
            let len = c.sequence_len(size::U8)?;
            c.take_slice(len)
        })
    }

    /// `Ascii`, built into an owned string in bounded chunks: `Bytes` whose bytes
    /// are all in `0x20..=0x7E`; the first byte outside is noncanonical there.
    /// Reserving the string is fallible. Chunked UTF-8 conversion avoids a second
    /// whole-input scan that could run beyond a cooperative deadline (SP-8).
    pub fn ascii(&mut self) -> Result<String, DecodeError> {
        self.nested(|c| {
            let at = c.offset;
            let len = c.sequence_len(size::U8)?;
            let start = c.offset;
            let text = c.take_slice(len)?;
            for (block, chunk) in text.chunks(4096).enumerate() {
                c.check_time()?;
                if let Some(index) = chunk.iter().position(|byte| !(0x20..=0x7E).contains(byte)) {
                    let at = start
                        .saturating_add(wide(block).saturating_mul(4096))
                        .saturating_add(wide(index));
                    return Err(DecodeError::new(ErrorCode::Noncanonical, at));
                }
            }
            c.check_time()?;
            let mut value = String::new();
            value
                .try_reserve_exact(text.len())
                .map_err(|_| DecodeError::new(ErrorCode::Memory, at))?;
            for chunk in text.chunks(4096) {
                c.check_time()?;
                // Printable ASCII is UTF-8; each validation and copy is at most 4 KiB.
                let part = core::str::from_utf8(chunk)
                    .map_err(|_| DecodeError::new(ErrorCode::Noncanonical, start))?;
                value.push_str(part);
            }
            c.check_time()?;
            Ok(value)
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
        let entry_min = key_min.saturating_add(value_min);
        self.nested(|c| {
            let at = c.offset;
            let len = c.sequence_len(entry_min)?;
            let mut previous: Option<&'a [u8]> = None;
            c.elements(len, entry_min, at, |c| {
                c.nested(|c| {
                    let key_at = c.offset;
                    let before = c.rest;
                    let k = key(c)?;
                    let used = before.len().saturating_sub(c.rest.len());
                    let encoding = before
                        .get(..used)
                        .ok_or(DecodeError::new(ErrorCode::Internal, key_at))?;
                    if let Some(previous) = previous {
                        let mut order = core::cmp::Ordering::Equal;
                        for (left, right) in encoding.chunks(4096).zip(previous.chunks(4096)) {
                            c.check_time()?;
                            order = left.cmp(right);
                            if !order.is_eq() {
                                break;
                            }
                        }
                        if order.is_eq() {
                            order = encoding.len().cmp(&previous.len());
                        }
                        if !order.is_gt() {
                            return Err(DecodeError::new(ErrorCode::Noncanonical, key_at));
                        }
                    }
                    previous = Some(encoding);
                    Ok((k, value(c)?))
                })
            })
        })
    }

    // Fixed-size composites.

    /// `[T; len]`, as a vector of exactly `len` values decoded by `element`, one level deeper.
    /// The array is checked as a whole first: `len · element_min` above the bytes left is
    /// `ERR-CODEC-TRUNCATED` at its first byte. Memory and each value's size are bounded as for
    /// [`vec`](Self::vec).
    pub fn array<T>(
        &mut self,
        len: usize,
        element_min: u64,
        element: impl FnMut(&mut Self) -> Result<T, DecodeError>,
    ) -> Result<Vec<T>, DecodeError> {
        self.nested(|c| {
            let at = c.offset;
            if !c.fits(wide(len), element_min) {
                return Err(c.error(ErrorCode::Truncated));
            }
            c.elements(len, element_min, at, element)
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
    pub(crate) fn finish(self) -> Result<(), DecodeError> {
        self.check_time()?;
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

/// The most memory, in bytes, a sequence reserves before its first value is decoded.
const FIRST_RESERVATION: usize = 64 << 10;

/// Makes room in `values` for one more of the `len` values of a sequence that starts at `at`:
/// none when there is room already; otherwise as much again as the vector holds, at least
/// [`FIRST_RESERVATION`] bytes' worth, and never past `len` values in all.
fn make_room<T>(values: &mut Vec<T>, len: usize, at: u64) -> Result<(), DecodeError> {
    // A type of size 0 never allocates and its vectors always have room, so `T` has a non-zero
    // size past this point.
    if values.len() < values.capacity() {
        return Ok(());
    }
    let first = FIRST_RESERVATION
        .checked_div(size_of::<T>())
        .unwrap_or(len)
        .max(1);
    let additional = values
        .capacity()
        .max(first)
        .min(len.saturating_sub(values.len()));
    values
        .try_reserve_exact(additional)
        .map_err(|_| DecodeError::new(ErrorCode::Memory, at))
}

/// A length as `u64`. Every supported target has 64-bit or smaller pointers, so this is exact.
pub(crate) fn wide(len: usize) -> u64 {
    u64::try_from(len).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod deadline_tests {
    use super::*;
    use std::sync::{
        Mutex, OnceLock,
        atomic::{AtomicUsize, Ordering},
    };
    use std::time::Duration;
    static LOCK: Mutex<()> = Mutex::new(());
    static BASE: OnceLock<Instant> = OnceLock::new();
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    static STOP: AtomicUsize = AtomicUsize::new(usize::MAX);
    fn now() -> Instant {
        let base = *BASE.get_or_init(Instant::now);
        if CALLS.fetch_add(1, Ordering::SeqCst) + 1 >= STOP.load(Ordering::SeqCst) {
            base + Duration::from_secs(2)
        } else {
            base
        }
    }
    fn timed(input: &[u8], stop: usize) -> Cursor<'_> {
        let mut cursor = Cursor::new(input, &DecodeLimits::MESSAGE).unwrap();
        cursor.now = now;
        cursor.deadline = Some(*BASE.get_or_init(Instant::now) + Duration::from_secs(1));
        CALLS.store(0, Ordering::SeqCst);
        STOP.store(stop, Ordering::SeqCst);
        cursor
    }
    #[test]
    fn expired_entry_cannot_run_the_schema_and_time_is_not_an_input_fault() {
        let mut ran = false;
        let result = decode(&[], &DecodeLimits::MESSAGE, |c| {
            c.set_deadline(Instant::now())?;
            ran = true;
            Ok(())
        });
        assert_eq!(result, Err(DecodeError::new(ErrorCode::Time, 0)));
        assert!(!ran);
        assert!(!ErrorCode::Time.is_input_fault());
        assert_eq!(ErrorCode::Time.id(), "ERR-RES-TIME");
    }
    #[test]
    fn primitive_and_sequence_loops_stop_at_their_work_boundary() {
        let _lock = LOCK.lock().unwrap();
        let bytes = vec![1; 8192];
        let mut c = timed(&bytes, 2);
        for _ in 0..4096 {
            assert_eq!(c.u8(), Ok(1));
        }
        assert_eq!(c.u8(), Err(DecodeError::new(ErrorCode::Time, 4096)));
        let mut c = timed(&bytes, 2);
        assert_eq!(
            c.elements(8192, 1, 0, Cursor::u8),
            Err(DecodeError::new(ErrorCode::Time, 2048))
        );
        let mut c = timed(&bytes, 1);
        assert_eq!(
            c.take_slice(4096),
            Err(DecodeError::new(ErrorCode::Time, 0))
        );
    }
    #[test]
    fn ascii_and_map_comparison_check_between_chunks_before_late_input_faults() {
        let _lock = LOCK.lock().unwrap();
        let mut bytes = (8192u64).to_le_bytes().to_vec();
        bytes.extend([b'A'; 8192]);
        bytes[8 + 4096] = 0xff;
        assert_eq!(
            decode(&bytes, &DecodeLimits::MESSAGE, Cursor::ascii)
                .unwrap_err()
                .code(),
            ErrorCode::Noncanonical
        );
        let mut c = timed(&bytes, 3);
        assert_eq!(c.ascii().unwrap_err().code(), ErrorCode::Time);
        assert_eq!(CALLS.load(Ordering::SeqCst), 3);
        bytes[8 + 4096] = b'A';
        let plain = decode(&bytes, &DecodeLimits::MESSAGE, Cursor::ascii).unwrap();
        assert_eq!(plain.len(), 8192);
        assert!(plain.bytes().all(|byte| byte == b'A'));
        for stop in [5, 6, 7] {
            let mut c = timed(&bytes, stop);
            assert_eq!(c.ascii().unwrap_err().code(), ErrorCode::Time);
            assert_eq!(CALLS.load(Ordering::SeqCst), stop);
        }
        let mut bytes = (2u64).to_le_bytes().to_vec();
        for _ in 0..2 {
            bytes.extend(8192u64.to_le_bytes());
            bytes.extend([1; 8192]);
            bytes.push(0);
        }
        let map = |c: &mut Cursor<'_>| c.map(8, 1, |c| c.bytes().map(|_| ()), Cursor::u8);
        assert_eq!(
            decode(&bytes, &DecodeLimits::MESSAGE, map)
                .unwrap_err()
                .code(),
            ErrorCode::Noncanonical
        );
        let mut c = timed(&bytes, 3);
        assert_eq!(map(&mut c).unwrap_err().code(), ErrorCode::Time);
        assert_eq!(CALLS.load(Ordering::SeqCst), 3);
    }
    #[test]
    fn a_nested_reader_cannot_extend_the_original_deadline() {
        let _lock = LOCK.lock().unwrap();
        let mut c = timed(&[1], 2);
        let later = *BASE.get().unwrap() + Duration::from_secs(10);
        assert_eq!(c.set_deadline(later), Ok(()));
        assert_eq!(c.u8(), Err(DecodeError::new(ErrorCode::Time, 0)));
    }
    #[test]
    fn completion_checks_even_when_the_last_read_succeeded() {
        let _lock = LOCK.lock().unwrap();
        let mut c = timed(&[1], 2);
        assert_eq!(c.u8(), Ok(1));
        assert_eq!(c.finish(), Err(DecodeError::new(ErrorCode::Time, 1)));
    }
}
