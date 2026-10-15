// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The artifact envelope (RFC-0003 'Envelope'), opened from memory.

use crate::cursor::wide;
use crate::{Cursor, DecodeError, DecodeLimits, ErrorCode};

/// The envelope's first four bytes.
pub const MAGIC: [u8; 4] = *b"TSRA";

/// The only envelope version.
pub const ENVELOPE_VERSION: u16 = 1;

/// The header's size: magic, four `u16` fields and the `u64` payload length.
pub const HEADER_LEN: u64 = 20;

/// The digest's size.
pub const DIGEST_LEN: u64 = 32;

/// The BLAKE3 `derive_key` context of the envelope digest, which covers every byte before it.
pub const DIGEST_CONTEXT: &str = "tessera/v1/envelope";

/// What an envelope's reader expects: a schema id, and the version `v` its writers emit. It
/// accepts `v` and `v − 1`, never 0, which no writer emits (03 §7, COMPAT-2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Expected {
    /// The schema id (03 §5).
    pub schema_id: u16,
    /// The version the schema's writers emit.
    pub version: u16,
}

impl Expected {
    /// Whether this reader decodes schema version `found`.
    pub const fn accepts(&self, found: u16) -> bool {
        if found == 0 {
            return false;
        }
        match self.version.checked_sub(1) {
            Some(previous) => found == self.version || found == previous,
            None => found == self.version,
        }
    }
}

/// The header fields that vary between envelopes the reader accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EnvelopeHeader {
    /// The schema id, the one the reader expects.
    pub schema_id: u16,
    /// The schema version found: the reader's version or the one before, which selects the
    /// payload's layout.
    pub schema_version: u16,
    /// The payload's size in bytes.
    pub payload_len: u64,
}

/// An envelope whose header, size and digest have been checked, so its payload may be decoded.
#[derive(Clone, Copy)]
pub struct Envelope<'a> {
    header: EnvelopeHeader,
    payload: &'a [u8],
    limits: DecodeLimits,
}

/// Shows the header, never the payload: an artifact may be secret-class (05 §3).
impl core::fmt::Debug for Envelope<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Envelope")
            .field("header", &self.header)
            .field("limits", &self.limits)
            .finish_non_exhaustive()
    }
}

impl<'a> Envelope<'a> {
    /// Checks `input` as an envelope of the `expected` schema, in the order RFC-0003 fixes: the
    /// header field by field, then the size of the rest, then the digest. `digest` computes the
    /// digest of the header and payload bytes it is given, BLAKE3 `derive_key` with context
    /// [`DIGEST_CONTEXT`]; it runs only once the framing is sound.
    ///
    /// Offsets count from the start of the envelope.
    pub fn open(
        input: &'a [u8],
        expected: Expected,
        limits: &DecodeLimits,
        digest: impl FnOnce(&[u8]) -> [u8; 32],
    ) -> Result<Self, DecodeError> {
        // 1. The header, field by field: a field cut short is ERR-CODEC-TRUNCATED at its offset,
        // and a complete field is checked before the next is read.
        let mut header = Cursor::at(input, 0, limits);
        let fail = |code, at| Err(DecodeError::new(code, at));

        if header.take::<4>()? != MAGIC {
            return fail(ErrorCode::Magic, 0);
        }
        let at = header.offset();
        if header.u16()? != ENVELOPE_VERSION {
            return fail(ErrorCode::Version, at);
        }
        let at = header.offset();
        let schema_id = header.u16()?;
        if schema_id != expected.schema_id {
            return fail(ErrorCode::Magic, at);
        }
        let at = header.offset();
        let schema_version = header.u16()?;
        if !expected.accepts(schema_version) {
            return fail(ErrorCode::Version, at);
        }
        let at = header.offset();
        if header.u16()? != 0 {
            return fail(ErrorCode::Noncanonical, at);
        }
        let length_at = header.offset();
        let payload_len = header.u64()?;
        let total = match payload_len.checked_add(HEADER_LEN.saturating_add(DIGEST_LEN)) {
            Some(total) if total <= limits.max_total_bytes => total,
            _ => return fail(ErrorCode::Limit, length_at),
        };

        // 2. The rest of the input: the payload and the digest, exactly.
        match wide(header.remaining()).cmp(&payload_len.saturating_add(DIGEST_LEN)) {
            core::cmp::Ordering::Less => return fail(ErrorCode::Truncated, length_at),
            core::cmp::Ordering::Greater => return fail(ErrorCode::Noncanonical, total),
            core::cmp::Ordering::Equal => {}
        }

        // 3. The digest, over every byte before it.
        let digest_at = HEADER_LEN.saturating_add(payload_len);
        let framing = DecodeError::new(ErrorCode::Truncated, length_at);
        let signed_len = usize::try_from(digest_at).map_err(|_| framing)?;
        let (signed, found) = input.split_at_checked(signed_len).ok_or(framing)?;
        if digest(signed).as_slice() != found {
            return fail(ErrorCode::Digest, digest_at);
        }
        let header_len = usize::try_from(HEADER_LEN).map_err(|_| framing)?;
        let payload = signed.get(header_len..).ok_or(framing)?;

        Ok(Self {
            header: EnvelopeHeader {
                schema_id,
                schema_version,
                payload_len,
            },
            payload,
            limits: *limits,
        })
    }

    /// The checked header.
    pub fn header(&self) -> EnvelopeHeader {
        self.header
    }

    /// The payload's bytes, covered by the verified digest.
    pub fn payload(&self) -> &'a [u8] {
        self.payload
    }

    /// Decodes the payload with `value`, as the schema version found, under the limits the
    /// envelope was opened with: the last step of RFC-0003's order. The value must fill the
    /// payload (INV-TCE-4), and errors carry offsets from the start of the envelope, their
    /// offsets in the payload plus 20.
    pub fn decode_payload<T>(
        &self,
        value: impl FnOnce(&mut Cursor<'a>) -> Result<T, DecodeError>,
    ) -> Result<T, DecodeError> {
        let mut cursor = Cursor::at(self.payload, HEADER_LEN, &self.limits);
        let decoded = value(&mut cursor)?;
        cursor.finish()?;
        Ok(decoded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_output_shows_no_input_byte() {
        // A slice would print as `[…]`; positions and lengths print as numbers.
        let input = [0x5a; 24];
        let cursor = Cursor::new(&input, &DecodeLimits::MESSAGE).unwrap();
        let shown = format!("{cursor:?}");
        assert!(
            shown.contains("remaining: 24") && !shown.contains('['),
            "{shown}"
        );
        let envelope = Envelope {
            header: EnvelopeHeader {
                schema_id: 1,
                schema_version: 1,
                payload_len: 24,
            },
            payload: &input,
            limits: DecodeLimits::MESSAGE,
        };
        let shown = format!("{envelope:?}");
        assert!(
            shown.contains("payload_len: 24") && !shown.contains('['),
            "{shown}"
        );
    }
}
