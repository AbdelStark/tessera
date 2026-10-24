// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The core of the software reference recorder (RFC-0015 'Reference recorder'), without its
//! network: the frames a connection carries, the entries the recorder logs for them, and the
//! signed manifest that closes each period. `tessera recorder ref` wraps it in a TCP proxy.
//!
//! - **Frames on the wire.** A connection's bytes, in each direction, are a sequence of frames,
//!   each told apart by its magic: a session open (`TSOP`, RFC-0059), a sealed frame (`TSFR`, a
//!   41-byte header and `length` bytes of ciphertext), or a request end (`TRQE`, RFC-0063).
//!   [`FrameParser`] cuts them out of the stream, so that the proxy forwards each frame only once
//!   the recorder has logged it: it stores and forwards (RFC-0063 'Inbound before outbound').
//! - **Entries.** [`Recorder::record`] gives each frame its log entry and its `seq`, and
//!   [`Recorder::close`] logs a session's close.
//! - **Periods.** [`Recorder::rotate`] signs the period's manifest over its entries (RFC-0060)
//!   and opens the next period with a `SessionCarry` for every session that kept RFC-0062's rule 1,
//!   is not closed, and had its open or a frame in the period or the one before.
//! - **The period's log on disk** is the concatenation of its entries' TCE encodings in `seq`
//!   order ([`encode_log`], [`decode_log`]): the `stream` of 03 §5, each entry's length fixed by
//!   its tag.
//!
//! Everything here is public: frames are ciphertext, and the signing seed stays in its
//! [`Secret`].

use crate::frame::{AnchorError, FrameHeader, RequestEndFrame, SealedFrame, SessionOpenFrame};
use crate::log::DigestLogEntry;
use crate::manifest::{LogManifest, ManifestBody, ManifestError, key_id, public_key};
use crate::merkle;
use std::collections::BTreeMap;
use tessera_codec::{DecodeLimits, HeapKey, Secret, decode, encode};

/// The largest ciphertext a [`FrameParser`] accepts in one sealed frame, 64 MiB: a bound on the
/// memory one frame can make the recorder hold. A larger `length` stops the parser.
pub const MAX_FRAME_LEN: u32 = 1 << 26;

/// One frame on the wire, as [`FrameParser`] cuts it out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WireFrame {
    /// A session's cleartext open (RFC-0059).
    Open(SessionOpenFrame),
    /// A sealed token frame (RFC-0059).
    Sealed(SealedFrame),
    /// A request's cleartext end (RFC-0063).
    End(RequestEndFrame),
}

impl WireFrame {
    /// The frame's bytes on the wire.
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            Self::Open(open) => open.to_bytes().to_vec(),
            Self::Sealed(sealed) => sealed.to_bytes(),
            Self::End(end) => end.to_bytes().to_vec(),
        }
    }

    /// The session the frame belongs to.
    pub fn session_id(&self) -> [u8; 16] {
        match self {
            Self::Open(open) => open.session_id,
            Self::Sealed(sealed) => sealed.header.session_id,
            Self::End(end) => end.session_id,
        }
    }
}

/// A streaming parser for one direction of a connection: bytes go in as they arrive, whole
/// frames come out.
#[derive(Debug, Default)]
pub struct FrameParser {
    buffer: Vec<u8>,
}

impl FrameParser {
    /// An empty parser.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends `bytes` that arrived.
    pub fn push(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }

    /// The bytes received and not yet cut into a frame.
    pub fn pending(&self) -> &[u8] {
        &self.buffer
    }

    /// The next whole frame, if the buffer holds one.
    ///
    /// # Errors
    /// [`AnchorError::Magic`] for bytes that start no known frame, [`AnchorError::TooLong`] for a
    /// sealed frame longer than [`MAX_FRAME_LEN`], or the frame's own decoding error. After an
    /// error the stream cannot be cut further, and the buffer is left as it was.
    pub fn next_frame(&mut self) -> Result<Option<WireFrame>, AnchorError> {
        if self.buffer.len() < 4 {
            return Ok(None);
        }
        let magic: [u8; 4] = [
            self.buffer[0],
            self.buffer[1],
            self.buffer[2],
            self.buffer[3],
        ];
        let (len, frame) = if magic == SessionOpenFrame::MAGIC {
            let len = SessionOpenFrame::LEN;
            if self.buffer.len() < len {
                return Ok(None);
            }
            (
                len,
                WireFrame::Open(SessionOpenFrame::from_bytes(&self.buffer[..len])?),
            )
        } else if magic == RequestEndFrame::MAGIC {
            let len = RequestEndFrame::LEN;
            if self.buffer.len() < len {
                return Ok(None);
            }
            (
                len,
                WireFrame::End(RequestEndFrame::from_bytes(&self.buffer[..len])?),
            )
        } else if magic == FrameHeader::MAGIC {
            if self.buffer.len() < FrameHeader::LEN {
                return Ok(None);
            }
            let header = FrameHeader::from_bytes(&self.buffer[..FrameHeader::LEN])?;
            if header.length > MAX_FRAME_LEN {
                return Err(AnchorError::TooLong);
            }
            let len = FrameHeader::LEN + header.length as usize;
            if self.buffer.len() < len {
                return Ok(None);
            }
            (
                len,
                WireFrame::Sealed(SealedFrame::from_bytes(&self.buffer[..len])?),
            )
        } else {
            return Err(AnchorError::Magic);
        };
        self.buffer.drain(..len);
        Ok(Some(frame))
    }
}

/// A session the recorder has seen in the current period (RFC-0062 'The carry entry').
#[derive(Clone, Debug)]
struct SessionState {
    /// The commitment of its first open or carry in the period.
    key_commitment: Option<[u8; 32]>,
    /// Its opens and carries in the period.
    keys: u32,
    /// Whether one of its entries came before its first open or carry.
    entry_before_key: bool,
    /// Whether it closed in the period.
    closed: bool,
    /// Whether it had its open or a frame in the period.
    active: bool,
    /// Whether it had its open or a frame in the previous period.
    active_before: bool,
}

impl SessionState {
    fn new(active_before: bool) -> Self {
        Self {
            key_commitment: None,
            keys: 0,
            entry_before_key: false,
            closed: false,
            active: false,
            active_before,
        }
    }

    /// Whether the next period carries it: rule 1 kept, not closed, and its open or a frame in
    /// this period or the one before.
    fn carried(&self) -> bool {
        self.keys == 1
            && !self.entry_before_key
            && !self.closed
            && (self.active || self.active_before)
    }
}

/// A period the recorder closed: its signed manifest and its entries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Period {
    /// The period's signed manifest.
    pub manifest: LogManifest,
    /// The period's entries, in `seq` order.
    pub entries: Vec<DigestLogEntry>,
}

/// The recorder's state: its key, its log, the current period's entries and the sessions open
/// in it.
pub struct Recorder {
    seed: Secret<HeapKey<32>>,
    recorder_key_id: [u8; 32],
    log_id: [u8; 16],
    period_start_ns: u64,
    first_seq: u64,
    entries: Vec<DigestLogEntry>,
    prev_manifest: [u8; 32],
    last_t_ns: u64,
    sessions: BTreeMap<[u8; 16], SessionState>,
}

impl Recorder {
    /// A recorder for the log `log_id`, signing with `seed`, whose first period starts at
    /// `start_ns`. Its first manifest links to 32 zero bytes, from entry 0 (RFC-0060 'The chain').
    pub fn new(seed: Secret<HeapKey<32>>, log_id: [u8; 16], start_ns: u64) -> Self {
        let recorder_key_id = key_id(&public_key(&seed));
        Self {
            seed,
            recorder_key_id,
            log_id,
            period_start_ns: start_ns,
            first_seq: 0,
            entries: Vec::new(),
            prev_manifest: [0; 32],
            last_t_ns: start_ns,
            sessions: BTreeMap::new(),
        }
    }

    /// The `seq` the next entry takes.
    pub fn next_seq(&self) -> u64 {
        self.first_seq + self.entries.len() as u64
    }

    /// The current period's start, in nanoseconds.
    pub fn period_start_ns(&self) -> u64 {
        self.period_start_ns
    }

    /// The current period's entries so far.
    pub fn entries(&self) -> &[DigestLogEntry] {
        &self.entries
    }

    /// The clock for an entry at `t_ns`: never before the period's start or a previous entry.
    fn clock(&mut self, t_ns: u64) -> u64 {
        self.last_t_ns = self.last_t_ns.max(t_ns).max(self.period_start_ns);
        self.last_t_ns
    }

    fn push(&mut self, entry: DigestLogEntry) -> DigestLogEntry {
        self.entries.push(entry);
        entry
    }

    fn session(&mut self, id: [u8; 16]) -> &mut SessionState {
        self.sessions
            .entry(id)
            .or_insert_with(|| SessionState::new(false))
    }

    /// Logs `frame`, at `t_ns` or later, and returns its entry.
    pub fn record(&mut self, frame: &WireFrame, t_ns: u64) -> DigestLogEntry {
        let t_ns = self.clock(t_ns);
        let seq = self.next_seq();
        let session = self.session(frame.session_id());
        let entry = match frame {
            WireFrame::Open(open) => {
                session.keys += 1;
                session.key_commitment.get_or_insert(open.key_commitment);
                session.active = true;
                DigestLogEntry::session_open(seq, t_ns, open)
            }
            WireFrame::Sealed(sealed) => {
                session.entry_before_key |= session.keys == 0;
                session.active = true;
                DigestLogEntry::frame(seq, t_ns, sealed)
            }
            WireFrame::End(end) => {
                session.entry_before_key |= session.keys == 0;
                DigestLogEntry::request_end(seq, t_ns, end)
            }
        };
        self.push(entry)
    }

    /// Logs the close of `session_id`, at `t_ns` or later, and returns its entry.
    pub fn close(&mut self, session_id: [u8; 16], t_ns: u64) -> DigestLogEntry {
        let t_ns = self.clock(t_ns);
        let seq = self.next_seq();
        let session = self.session(session_id);
        session.entry_before_key |= session.keys == 0;
        session.closed = true;
        self.push(DigestLogEntry::SessionClose {
            seq,
            t_ns,
            session_id,
        })
    }

    /// Closes the current period at `end_ns`, after its last entry, and opens the next.
    ///
    /// It signs the period's manifest over its entries. Then it starts the next period at
    /// `end_ns` with a `SessionCarry` for every session RFC-0062 'The carry entry' carries, in
    /// `session_id` order.
    ///
    /// # Errors
    /// [`ManifestError::Period`] when `end_ns` is not after the period's start and its last entry:
    /// the reference recorder writes no zero-length periods.
    pub fn rotate(&mut self, end_ns: u64) -> Result<Period, ManifestError> {
        if end_ns <= self.period_start_ns || end_ns <= self.last_t_ns {
            return Err(ManifestError::Period);
        }
        let leaves: Vec<merkle::Hash> = self.entries.iter().map(merkle::leaf_hash).collect();
        let body = ManifestBody {
            recorder_key_id: self.recorder_key_id,
            log_id: self.log_id,
            period_start_ns: self.period_start_ns,
            period_end_ns: end_ns,
            first_seq: self.first_seq,
            entry_count: self.entries.len() as u64,
            entries_root: merkle::root(&leaves),
            prev_manifest: self.prev_manifest,
        };
        let manifest = body.sign(&self.seed)?;
        let entries = core::mem::take(&mut self.entries);
        self.prev_manifest = manifest.identity();
        self.first_seq = manifest.body.first_seq + manifest.body.entry_count;
        self.period_start_ns = end_ns;
        self.last_t_ns = end_ns;
        let sessions = core::mem::take(&mut self.sessions);
        for (id, state) in sessions {
            if !state.carried() {
                continue;
            }
            let Some(key_commitment) = state.key_commitment else {
                continue;
            };
            let mut next = SessionState::new(state.active);
            next.keys = 1;
            next.key_commitment = Some(key_commitment);
            self.sessions.insert(id, next);
            let entry = DigestLogEntry::SessionCarry {
                seq: self.next_seq(),
                t_ns: end_ns,
                session_id: id,
                key_commitment,
            };
            self.push(entry);
        }
        Ok(Period { manifest, entries })
    }
}

/// Shows nothing of the seed (05 §3).
impl core::fmt::Debug for Recorder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Recorder")
            .field("log_id", &self.log_id)
            .field("first_seq", &self.first_seq)
            .field("entries", &self.entries.len())
            .finish_non_exhaustive()
    }
}

/// A fresh log identifier, 16 bytes from the operating system's generator: a new log for every
/// recorder that starts (RFC-0060 'The chain').
///
/// # Errors
/// [`AnchorError::Random`] if the generator fails.
pub fn random_log_id() -> Result<[u8; 16], AnchorError> {
    let mut id = [0u8; 16];
    getrandom::fill(&mut id).map_err(|_| AnchorError::Random)?;
    Ok(id)
}

/// A period's log on disk: its entries' TCE encodings, concatenated in `seq` order.
pub fn encode_log(entries: &[DigestLogEntry]) -> Vec<u8> {
    entries.iter().flat_map(encode).collect()
}

/// The length of the TCE encoding of an entry with tag `tag`: the tag, then the variant's fixed
/// fields.
fn entry_len(tag: u16) -> Option<usize> {
    match tag {
        0 => Some(2 + 8 + 8 + 16 + 32),
        1 => Some(2 + 8 + 8 + 16 + 8 + 2 + 4 + 4 + 4 + 32),
        2 => Some(2 + 8 + 8 + 16),
        3 => Some(2 + 8 + 8 + 16 + 32),
        4 => Some(2 + 8 + 8 + 16 + 8),
        _ => None,
    }
}

/// The entries of a period's log on disk ([`encode_log`]).
///
/// # Errors
/// [`AnchorError::Noncanonical`] for an unknown tag, [`AnchorError::Truncated`] for a log that
/// ends inside an entry, or an entry's own decoding failure, mapped to the same two.
pub fn decode_log(mut bytes: &[u8]) -> Result<Vec<DigestLogEntry>, AnchorError> {
    let mut entries = Vec::new();
    while !bytes.is_empty() {
        if bytes.len() < 2 {
            return Err(AnchorError::Truncated);
        }
        let tag = u16::from_le_bytes([bytes[0], bytes[1]]);
        let len = entry_len(tag).ok_or(AnchorError::Noncanonical)?;
        if bytes.len() < len {
            return Err(AnchorError::Truncated);
        }
        let entry = decode::<DigestLogEntry>(&bytes[..len], DecodeLimits::MESSAGE)
            .map_err(|_| AnchorError::Noncanonical)?;
        entries.push(entry);
        bytes = &bytes[len..];
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each tag's length is its entry's encoded length.
    #[test]
    fn entry_lengths_are_the_encodings() {
        let id = [7; 16];
        let entries = [
            DigestLogEntry::SessionOpen {
                seq: 1,
                t_ns: 2,
                session_id: id,
                key_commitment: [3; 32],
            },
            DigestLogEntry::Frame {
                seq: 1,
                t_ns: 2,
                session_id: id,
                request_id: 4,
                direction: crate::Direction::Out,
                frame_index: 5,
                length: 26,
                frame_seq: 6,
                sha256: [8; 32],
            },
            DigestLogEntry::SessionClose {
                seq: 1,
                t_ns: 2,
                session_id: id,
            },
            DigestLogEntry::SessionCarry {
                seq: 1,
                t_ns: 2,
                session_id: id,
                key_commitment: [3; 32],
            },
            DigestLogEntry::RequestEnd {
                seq: 1,
                t_ns: 2,
                session_id: id,
                request_id: 4,
            },
        ];
        for (tag, entry) in entries.iter().enumerate() {
            assert_eq!(
                Some(encode(entry).len()),
                entry_len(tag as u16),
                "tag {tag}"
            );
        }
        assert_eq!(entry_len(5), None);
        let bytes = encode_log(&entries);
        assert_eq!(decode_log(&bytes), Ok(entries.to_vec()));
        assert_eq!(
            decode_log(&bytes[..bytes.len() - 1]),
            Err(AnchorError::Truncated)
        );
        assert_eq!(decode_log(&[5, 0]), Err(AnchorError::Noncanonical));
        assert_eq!(decode_log(&[1]), Err(AnchorError::Truncated));
    }
}
