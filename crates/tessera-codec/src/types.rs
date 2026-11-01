// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The named types of RFC-0003 'Primitive types' and 'Composite types' that Rust has no direct
//! counterpart for: printable ASCII text, exact fractions and canonically sorted maps.

use core::cmp::Ordering;
use core::fmt;

use crate::tce::{Decode, DecodeError, Encode, Reader, Sink, encode, reservation};

/// Raw bytes: a `Vec<u8>`, with a `u64` length.
pub type Bytes = Vec<u8>;

/// A byte that is not printable ASCII, at `index`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NotAscii {
    /// Where the byte is.
    pub index: usize,
}

impl fmt::Display for NotAscii {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "byte {} is not printable ASCII (0x20 to 0x7E)",
            self.index
        )
    }
}

impl std::error::Error for NotAscii {}

/// Printable ASCII text, every byte in `0x20..=0x7E`, encoded as `Vec<u8>`.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ascii(String);

impl Ascii {
    /// The text, or the first byte outside `0x20..=0x7E`.
    pub fn new(text: impl Into<String>) -> Result<Self, NotAscii> {
        let text = text.into();
        match text.bytes().position(|b| !is_printable(b)) {
            Some(index) => Err(NotAscii { index }),
            None => Ok(Self(text)),
        }
    }

    /// Retained text allocation, including spare capacity, in bytes (RFC-0018).
    /// Does not allocate or expose any additional text.
    pub fn heap_bytes(&self) -> usize {
        self.0.capacity()
    }

    /// The text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

fn is_printable(byte: u8) -> bool {
    (0x20..=0x7E).contains(&byte)
}

impl fmt::Display for Ascii {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Encode for Ascii {
    fn encode_to<S: Sink + ?Sized>(&self, out: &mut S) {
        (self.0.len() as u64).encode_to(out);
        out.put(self.0.as_bytes());
    }
}

impl Decode for Ascii {
    const MIN_SIZE: usize = 8;

    fn decode_from(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        reader.enter()?;
        let n = reader.sequence_len(1)?;
        let start = reader.position();
        let bytes = reader.take(n)?;
        if let Some(index) = bytes.iter().position(|&b| !is_printable(b)) {
            return Err(reader.noncanonical(start + index));
        }
        // Printable ASCII is valid UTF-8.
        let text = core::str::from_utf8(bytes).map_err(|_| reader.noncanonical(start))?;
        reader.leave();
        Ok(Self(text.to_owned()))
    }
}

/// An exact fraction `num / den` with `den > 0` and `gcd(|num|, den) = 1`: TCE's stand-in for
/// quantities that would be floats (03 INV-DM-6). Each value has one representation, so equal
/// fractions encode equally.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rational {
    num: i64,
    den: u64,
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

impl Rational {
    /// The multiplicative identity `1 / 1`.
    pub const ONE: Self = Self { num: 1, den: 1 };

    /// `num / den` in lowest terms, or `None` when `den` is zero.
    pub fn new(num: i64, den: u64) -> Option<Self> {
        if den == 0 {
            return None;
        }
        let g = gcd(num.unsigned_abs(), den);
        // |num| / g ≤ |num|, so the quotient fits i64 even for num = i64::MIN.
        let num = i64::try_from(i128::from(num) / i128::from(g)).ok()?;
        Some(Self { num, den: den / g })
    }

    /// The numerator.
    pub fn num(&self) -> i64 {
        self.num
    }

    /// The denominator, always positive.
    pub fn den(&self) -> u64 {
        self.den
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.num, self.den)
    }
}

impl Encode for Rational {
    fn encode_to<S: Sink + ?Sized>(&self, out: &mut S) {
        self.num.encode_to(out);
        self.den.encode_to(out);
    }
}

impl Decode for Rational {
    const MIN_SIZE: usize = 16;

    fn decode_from(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        reader.enter()?;
        let at = reader.position();
        let num = i64::decode_from(reader)?;
        let den = u64::decode_from(reader)?;
        if den == 0 || gcd(num.unsigned_abs(), den) != 1 {
            return Err(reader.noncanonical(at));
        }
        reader.leave();
        Ok(Self { num, den })
    }
}

/// Two entries with equal keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DuplicateKey;

impl fmt::Display for DuplicateKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("two map entries have the same key")
    }
}

impl std::error::Error for DuplicateKey {}

/// A map: entries sorted strictly ascending by the byte encoding of their keys, encoded as
/// `Vec<(K, V)>`. Sorting by encoding, not by value, keeps the order independent of language
/// and type: for little-endian integers it is not numeric order.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Map<K, V> {
    entries: Vec<(K, V)>,
}

impl<K: Encode, V> Map<K, V> {
    /// The map holding `entries`, in canonical order; [`DuplicateKey`] if two keys are equal.
    pub fn from_entries(entries: Vec<(K, V)>) -> Result<Self, DuplicateKey> {
        let mut keyed: Vec<(Vec<u8>, (K, V))> = entries
            .into_iter()
            .map(|entry| (encode(&entry.0), entry))
            .collect();
        keyed.sort_by(|a, b| a.0.cmp(&b.0));
        if keyed.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(DuplicateKey);
        }
        Ok(Self {
            entries: keyed.into_iter().map(|(_, entry)| entry).collect(),
        })
    }

    /// The value under `key`.
    pub fn get(&self, key: &K) -> Option<&V> {
        let wanted = encode(key);
        self.entries
            .binary_search_by(|(k, _)| encode(k).cmp(&wanted))
            .ok()
            .map(|i| &self.entries[i].1)
    }
}

impl<K, V> Map<K, V> {
    /// The entries, in canonical order.
    pub fn entries(&self) -> &[(K, V)] {
        &self.entries
    }

    /// The entries, in canonical order.
    pub fn into_entries(self) -> Vec<(K, V)> {
        self.entries
    }

    /// The number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl<K, V> Default for Map<K, V> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

impl<K: Encode, V: Encode> Encode for Map<K, V> {
    fn encode_to<S: Sink + ?Sized>(&self, out: &mut S) {
        self.entries.encode_to(out);
    }
}

impl<K: Decode, V: Decode> Decode for Map<K, V> {
    const MIN_SIZE: usize = 8;

    fn decode_from(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        const {
            assert!(
                K::MIN_SIZE + V::MIN_SIZE > 0,
                "a TCE map needs entries of at least one byte"
            );
        }
        reader.enter()?;
        let n = reader.sequence_len(K::MIN_SIZE + V::MIN_SIZE)?;
        let mut entries = Vec::with_capacity(reservation::<(K, V)>(n, reader.remaining()));
        let mut previous: Option<&[u8]> = None;
        for _ in 0..n {
            // Each entry is one level, as the pair it is in `Vec<(K, V)>`.
            reader.enter()?;
            let start = reader.position();
            let key = K::decode_from(reader)?;
            let key_bytes = reader.since(start);
            // Strictly ascending: an equal key is a duplicate, a smaller one is out of order.
            if previous.is_some_and(|p| p.cmp(key_bytes) != Ordering::Less) {
                return Err(reader.noncanonical(start));
            }
            previous = Some(key_bytes);
            entries.push((key, V::decode_from(reader)?));
            reader.leave();
        }
        reader.leave();
        Ok(Self { entries })
    }
}
