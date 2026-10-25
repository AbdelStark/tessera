// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A session's stored transcript: the secret-class artifact `SessionTranscript` (schema
//! `0x0404`, version 1, `.tst`; RFC-0015 'Prover pre-check'). The serving stack keeps one per
//! session, and the prover's pre-check replays it against the period's log before `Round1`.
//!
//! It holds the session's identifier, its secrets `k_s` and `salt_k` (RFC-0059 'Storing the
//! keys'), and every frame it sealed: the counter it took and its plaintext. Keys and tokens stay
//! in their [`Secret`]s; the encoding reads them only to write the artifact, which
//! `envelope::write_file` writes with mode 0600 through a clearing buffer (06 §7).
//!
//! TCE, in order:
//! - `session_id: [u8; 16]`, `key: [u8; 16]`, `key_salt: [u8; 16]`;
//! - `frames: Vec<StoredFrame>`, each `frame_seq: u32`, `request_id: u64`, the `u16` direction
//!   tag, `frame_index: u32`, then `tokens: Vec<u32>`; `frame_seq` strictly increasing.
//!
//! The decoder refuses a direction tag other than 0 or 1, and a `frame_seq` not above the one
//! before it, with `ERR-CODEC-NONCANONICAL` at the field's first byte.

use core::borrow::{Borrow, BorrowMut};
use core::fmt;

use tessera_codec::{Decode, DecodeError, Encode, HeapKey, Reader, Secret, Sink};

use crate::frame::{AnchorError, Direction, SealedFrame, TokenFrame};
use crate::session::SessionKeys;

/// One frame a session sealed: its counter and its plaintext.
pub struct StoredFrame {
    /// The counter the frame was sealed under (RFC-0059 'The keystream').
    pub frame_seq: u32,
    /// The plaintext, its tokens in their [`Secret`].
    pub frame: TokenFrame,
}

impl fmt::Debug for StoredFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StoredFrame")
            .field("frame_seq", &self.frame_seq)
            .field("frame", &self.frame)
            .finish()
    }
}

/// A session's stored transcript (schema `0x0404`).
pub struct SessionTranscript {
    session_id: [u8; 16],
    keys: SessionKeys,
    frames: Vec<StoredFrame>,
}

impl fmt::Debug for SessionTranscript {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SessionTranscript")
            .field("session_id", &self.session_id)
            .field("frames", &self.frames.len())
            .finish_non_exhaustive()
    }
}

impl SessionTranscript {
    /// The transcript of a session just opened, from its identifier and the secrets
    /// [`Session::open`](crate::Session::open) returned.
    pub fn new(session_id: [u8; 16], keys: SessionKeys) -> Self {
        Self {
            session_id,
            keys,
            frames: Vec::new(),
        }
    }

    /// Records `frame`, which the session sealed as `sealed`.
    ///
    /// # Errors
    /// [`AnchorError::Mismatch`] when `sealed` is of another session, or its header's request,
    /// direction, index or length is not `frame`'s, or its counter is not above the last one
    /// recorded.
    pub fn record(&mut self, sealed: &SealedFrame, frame: TokenFrame) -> Result<(), AnchorError> {
        let h = &sealed.header;
        let consistent = h.session_id == self.session_id
            && (h.request_id, h.direction, h.frame_index)
                == (frame.request_id, frame.direction, frame.frame_index)
            && u64::from(h.length) == frame.plaintext_len()
            && self
                .frames
                .last()
                .is_none_or(|last| h.frame_seq > last.frame_seq);
        if !consistent {
            return Err(AnchorError::Mismatch);
        }
        self.frames.push(StoredFrame {
            frame_seq: h.frame_seq,
            frame,
        });
        Ok(())
    }

    /// The session's identifier.
    pub fn session_id(&self) -> [u8; 16] {
        self.session_id
    }

    /// The session's secrets.
    pub fn keys(&self) -> &SessionKeys {
        &self.keys
    }

    /// The frames, in counter order.
    pub fn frames(&self) -> &[StoredFrame] {
        &self.frames
    }

    /// The frame sealed under `frame_seq`, if the transcript holds one.
    pub fn frame(&self, frame_seq: u32) -> Option<&StoredFrame> {
        self.frames
            .binary_search_by_key(&frame_seq, |f| f.frame_seq)
            .ok()
            .and_then(|i| self.frames.get(i))
    }
}

/// The frame's encoding, after its counter.
#[allow(clippy::disallowed_methods)]
fn encode_frame<S: Sink + ?Sized>(frame: &TokenFrame, out: &mut S) {
    frame.request_id.encode_to(out);
    u16::from(frame.direction.byte()).encode_to(out);
    frame.frame_index.encode_to(out);
    frame.tokens.expose_secret().encode_to(out);
}

impl Encode for SessionTranscript {
    #[allow(clippy::disallowed_methods)]
    fn encode_to<S: Sink + ?Sized>(&self, out: &mut S) {
        out.put(&self.session_id);
        out.put(Borrow::<[u8; 16]>::borrow(self.keys.key.expose_secret()));
        out.put(Borrow::<[u8; 16]>::borrow(
            self.keys.key_salt.expose_secret(),
        ));
        (self.frames.len() as u64).encode_to(out);
        for stored in &self.frames {
            stored.frame_seq.encode_to(out);
            encode_frame(&stored.frame, out);
        }
    }
}

/// A key read from the input into its own heap [`Secret`].
#[allow(clippy::disallowed_methods)]
fn read_key(reader: &mut Reader<'_>) -> Result<Secret<HeapKey>, DecodeError> {
    let bytes = reader.take(16)?;
    let mut key = Secret::new(HeapKey::zeroed());
    let slot: &mut [u8; 16] = key.expose_secret_mut().borrow_mut();
    slot.copy_from_slice(bytes);
    Ok(key)
}

/// The fewest bytes a stored frame encodes to: its counter, request, tag, index and an empty
/// token list.
const STORED_FRAME_MIN: usize = 4 + 8 + 2 + 4 + 8;

impl Decode for SessionTranscript {
    const MIN_SIZE: usize = 16 + 16 + 16 + 8;

    #[allow(clippy::disallowed_methods)]
    fn decode_from(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        reader.enter()?;
        let session_id = reader.take_array::<16>()?;
        let key = read_key(reader)?;
        let key_salt = read_key(reader)?;
        reader.enter()?;
        let n = reader.sequence_len(STORED_FRAME_MIN)?;
        let mut frames: Vec<StoredFrame> =
            Vec::with_capacity(n.min(reader.remaining() / STORED_FRAME_MIN));
        for _ in 0..n {
            reader.enter()?;
            let at = reader.position();
            let frame_seq = u32::decode_from(reader)?;
            if frames
                .last()
                .is_some_and(|last| frame_seq <= last.frame_seq)
            {
                return Err(reader.noncanonical(at));
            }
            let request_id = u64::decode_from(reader)?;
            let at = reader.position();
            let tag = u16::decode_from(reader)?;
            let direction = u8::try_from(tag)
                .ok()
                .and_then(|tag| Direction::from_byte(tag).ok())
                .ok_or_else(|| reader.noncanonical(at))?;
            let frame_index = u32::decode_from(reader)?;
            let tokens = Secret::new(Vec::<u32>::decode_from(reader)?);
            reader.leave();
            frames.push(StoredFrame {
                frame_seq,
                frame: TokenFrame::new(request_id, direction, frame_index, tokens),
            });
        }
        reader.leave();
        reader.leave();
        Ok(Self {
            session_id,
            keys: SessionKeys { key, key_salt },
            frames,
        })
    }
}
