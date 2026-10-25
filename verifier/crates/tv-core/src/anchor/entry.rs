// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The recorder's digest log entries, `DigestLogEntry` (schema `0x0401`; RFC-0015 'Log format',
//! RFC-0059 'The log entry', RFC-0061 'Sessions', RFC-0062 'The carry entry', RFC-0063 'The
//! request's end'), and a period's log on disk, the concatenation of their encodings (RFC-0062
//! 'On disk').

use tv_codec::{Cursor, DecodeError, ErrorCode};

/// A frame's direction across the cluster's boundary: a TCE enum, `In` tag 0 and `Out` tag 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Direction {
    /// Inbound: the prompt.
    In,
    /// Outbound: the output.
    Out,
}

/// A `Frame` entry: the frame header's fields, which the recorder copies, with the entry's `seq`
/// and `t_ns` and the frame's digest (RFC-0059 'The log entry').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrameEntry {
    /// The entry's position in the log.
    pub seq: u64,
    /// The recorder's time, in nanoseconds.
    pub t_ns: u64,
    /// The frame's session.
    pub session_id: [u8; 16],
    /// The frame's request.
    pub request_id: u64,
    /// In or out.
    pub direction: Direction,
    /// The frame's position among its request's frames of its direction.
    pub frame_index: u32,
    /// The ciphertext's length in bytes, `|TCE(TokenFrame)|`; any value decodes, and RFC-0059's
    /// `22 + 4n` is a check of the request's frames.
    pub length: u32,
    /// The session counter (RFC-0059 'The session counter').
    pub frame_seq: u32,
    /// `SHA-256(header ‖ ciphertext)`.
    pub sha256: [u8; 32],
}

impl FrameEntry {
    /// The frame's token count `n`, when its `length` is `22 + 4n`, which every plaintext's is
    /// (RFC-0059 'The plaintext'); `None` for any other length.
    pub const fn tokens(&self) -> Option<u32> {
        match self.length.checked_sub(PLAINTEXT_PREFIX) {
            Some(rest) if rest % 4 == 0 => Some(rest / 4),
            _ => None,
        }
    }
}

/// The plaintext's prefix: `u64` request, `u16` direction tag, `u32` frame index, `u64` count.
const PLAINTEXT_PREFIX: u32 = 22;

/// One entry of the recorder's digest log (schema `0x0401`, version 1): a `u16` tag, then the
/// variant's fields in order (RFC-0003 'Composite types').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DigestLogEntry {
    /// Tag 0: a session's open, with its key commitment (RFC-0061 'Sessions').
    SessionOpen {
        /// The entry's position in the log.
        seq: u64,
        /// The recorder's time.
        t_ns: u64,
        /// The session.
        session_id: [u8; 16],
        /// `SHA-256("tessera-anchor-key/v1" ‖ k_s ‖ salt_k)`.
        key_commitment: [u8; 32],
    },
    /// Tag 1: a token frame's digest.
    Frame(FrameEntry),
    /// Tag 2: a session's close.
    SessionClose {
        /// The entry's position in the log.
        seq: u64,
        /// The recorder's time.
        t_ns: u64,
        /// The session.
        session_id: [u8; 16],
    },
    /// Tag 3: a session carried into the period with its key commitment (RFC-0062 'The carry
    /// entry').
    SessionCarry {
        /// The entry's position in the log.
        seq: u64,
        /// The recorder's time.
        t_ns: u64,
        /// The session.
        session_id: [u8; 16],
        /// The commitment of the session's open or earlier carry.
        key_commitment: [u8; 32],
    },
    /// Tag 4: a request's end (RFC-0063 'The request's end').
    RequestEnd {
        /// The entry's position in the log.
        seq: u64,
        /// The recorder's time.
        t_ns: u64,
        /// The session.
        session_id: [u8; 16],
        /// The request that ended.
        request_id: u64,
    },
}

/// The number of `DigestLogEntry` variants: a tag of 5 or more names none.
const VARIANTS: u16 = 5;

impl DigestLogEntry {
    /// The entry's position in the log.
    pub const fn seq(&self) -> u64 {
        match self {
            Self::SessionOpen { seq, .. }
            | Self::SessionClose { seq, .. }
            | Self::SessionCarry { seq, .. }
            | Self::RequestEnd { seq, .. } => *seq,
            Self::Frame(frame) => frame.seq,
        }
    }

    /// The recorder's time of the entry.
    pub const fn t_ns(&self) -> u64 {
        match self {
            Self::SessionOpen { t_ns, .. }
            | Self::SessionClose { t_ns, .. }
            | Self::SessionCarry { t_ns, .. }
            | Self::RequestEnd { t_ns, .. } => *t_ns,
            Self::Frame(frame) => frame.t_ns,
        }
    }

    /// The session the entry belongs to; every variant names one.
    pub const fn session_id(&self) -> &[u8; 16] {
        match self {
            Self::SessionOpen { session_id, .. }
            | Self::SessionClose { session_id, .. }
            | Self::SessionCarry { session_id, .. }
            | Self::RequestEnd { session_id, .. } => session_id,
            Self::Frame(frame) => &frame.session_id,
        }
    }

    /// The variant's TCE tag.
    pub const fn tag(&self) -> u16 {
        match self {
            Self::SessionOpen { .. } => 0,
            Self::Frame(_) => 1,
            Self::SessionClose { .. } => 2,
            Self::SessionCarry { .. } => 3,
            Self::RequestEnd { .. } => 4,
        }
    }

    /// The key commitment of an open or a carry, the entries that give a session its key in a
    /// period (RFC-0062 'The session rules', rule 1); `None` for the others.
    pub const fn key_commitment(&self) -> Option<&[u8; 32]> {
        match self {
            Self::SessionOpen { key_commitment, .. }
            | Self::SessionCarry { key_commitment, .. } => Some(key_commitment),
            _ => None,
        }
    }

    /// Decodes one entry at `cursor`: the tag, then the variant's fields.
    ///
    /// # Errors
    ///
    /// The cursor's errors, at their offsets: a field cut short is `ERR-CODEC-TRUNCATED` at its
    /// first byte; a tag of 5 or more, or a direction tag other than 0 or 1, is
    /// `ERR-CODEC-NONCANONICAL` at that tag; a depth beyond the limits is `ERR-CODEC-LIMIT`.
    pub fn decode(cursor: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        cursor.enumeration(VARIANTS, |c, tag| {
            let seq = c.u64()?;
            let t_ns = c.u64()?;
            let session_id = c.byte_array::<16>()?;
            Ok(match tag {
                0 => Self::SessionOpen {
                    seq,
                    t_ns,
                    session_id,
                    key_commitment: c.digest()?,
                },
                1 => Self::Frame(FrameEntry {
                    seq,
                    t_ns,
                    session_id,
                    request_id: c.u64()?,
                    direction: direction(c)?,
                    frame_index: c.u32()?,
                    length: c.u32()?,
                    frame_seq: c.u32()?,
                    sha256: c.digest()?,
                }),
                2 => Self::SessionClose {
                    seq,
                    t_ns,
                    session_id,
                },
                3 => Self::SessionCarry {
                    seq,
                    t_ns,
                    session_id,
                    key_commitment: c.digest()?,
                },
                4 => Self::RequestEnd {
                    seq,
                    t_ns,
                    session_id,
                    request_id: c.u64()?,
                },
                // `enumeration` refuses every tag of `VARIANTS` or more.
                _ => return Err(DecodeError::new(ErrorCode::Internal, c.offset())),
            })
        })
    }
}

/// `Direction`: a `u16` tag, 0 or 1.
fn direction(cursor: &mut Cursor<'_>) -> Result<Direction, DecodeError> {
    cursor.enumeration(2, |_, tag| {
        Ok(if tag == 0 {
            Direction::In
        } else {
            Direction::Out
        })
    })
}

/// Cuts a period's log, the concatenation of its entries' encodings with no length prefix
/// (RFC-0062 'On disk'), one entry at a time, handing `each` the entry and its encoding.
///
/// Every entry's length is fixed by its tag, so the stream cuts one way. The offsets of errors
/// count from the start of the stream. The caller checks the stream's total size first (`decode`
/// of `tv-codec` does), and `each` may stop the cut with an error of its own.
///
/// # Errors
///
/// The first decoding error, or the first error of `each`.
pub(crate) fn cut<'a>(
    cursor: &mut Cursor<'a>,
    stream: &'a [u8],
    mut each: impl FnMut(DigestLogEntry, &'a [u8], u64) -> Result<(), DecodeError>,
) -> Result<(), DecodeError> {
    while cursor.remaining() > 0 {
        let start = cursor.offset();
        let entry = DigestLogEntry::decode(cursor)?;
        let end = cursor.offset();
        let range = usize::try_from(start)
            .ok()
            .zip(usize::try_from(end).ok())
            .and_then(|(start, end)| stream.get(start..end))
            .ok_or(DecodeError::new(ErrorCode::Internal, start))?;
        each(entry, range, start)?;
    }
    Ok(())
}
