// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A session's stored transcript: the secret-class artifact `SessionTranscript` (schema
//! `0x0404`, version 1, `.tst`; RFC-0015 'Prover pre-check'). The serving stack keeps one per
//! session, and the prover's pre-check replays it against the period's log before `Round1`.
//!
//! It holds the session's identifier, its secrets `k_s` and `salt_k` (RFC-0059 'Storing the
//! keys'), and every frame it sealed: the counter it took and its plaintext. Keys and tokens stay
//! in their [`Secret`]s; the encoding reads them only to write the artifact, which
//! `envelope::write_file` writes with mode 0600 through a clearing buffer (06 §7), and
//! [`SessionTranscript::request_tokens`] only to copy a request's tokens into one new [`Secret`]
//! for its proof.
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
use crate::log::DigestLogEntry;
use crate::period::VerifiedLog;
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

    /// The tokens of request `request_id` of this session, as its proof takes them: the stored
    /// plaintexts of the frames the window logs for it, its `In` frames' tokens and then its
    /// `Out` frames', each direction in `frame_index` order. The window is `current`'s log and
    /// `previous`'s, as [`precheck`](crate::precheck()) reads it, which the caller runs first: the
    /// tokens are then those whose re-sealed digests the log holds, and their count is the
    /// statement's `prompt_len + output_len`. A stored frame the log does not name is left out.
    ///
    /// `None` when the logged frames break RFC-0063's rule 5 (in each direction, the indices
    /// `0, …, n − 1`, each once), or when one has no stored frame at its counter with the
    /// entry's request, direction, index and length.
    ///
    /// The tokens are copied into one new vector, sized before the first copy, so no buffer of
    /// them is freed uncleared (05 §3 OBS-6).
    #[allow(clippy::disallowed_methods)]
    pub fn request_tokens(
        &self,
        previous: Option<&VerifiedLog>,
        current: &VerifiedLog,
        request_id: u64,
    ) -> Option<Secret<Vec<u32>>> {
        // The logged frames: (direction, index, counter, length), in proof order once sorted.
        let before = previous.map_or(&[][..], VerifiedLog::entries);
        let mut logged: Vec<(u8, u32, u32, u32)> = before
            .iter()
            .chain(current.entries())
            .filter_map(|entry| match *entry {
                DigestLogEntry::Frame {
                    session_id,
                    request_id: r,
                    direction,
                    frame_index,
                    length,
                    frame_seq,
                    ..
                } if session_id == self.session_id && r == request_id => {
                    Some((direction.byte(), frame_index, frame_seq, length))
                }
                _ => None,
            })
            .collect();
        logged.sort_unstable();
        let mut stored = Vec::with_capacity(logged.len());
        let mut total = 0usize;
        for (i, &(direction, frame_index, frame_seq, length)) in logged.iter().enumerate() {
            // Rule 5: each direction's indices run 0, 1, … with no gap or repeat.
            let expected = match i.checked_sub(1).and_then(|p| logged.get(p)) {
                Some(&(d, index, _, _)) if d == direction => index.checked_add(1)?,
                _ => 0,
            };
            if frame_index != expected {
                return None;
            }
            let frame = &self.frame(frame_seq)?.frame;
            if (frame.request_id, frame.direction.byte(), frame.frame_index)
                != (request_id, direction, frame_index)
                || frame.plaintext_len() != u64::from(length)
            {
                return None;
            }
            total = total.checked_add(frame.tokens().expose_secret().len())?;
            stored.push(frame);
        }
        let mut tokens = Vec::with_capacity(total);
        for frame in stored {
            tokens.extend_from_slice(frame.tokens().expose_secret());
        }
        Some(Secret::new(tokens))
    }

    /// The tokens of a batch of `requests` (`(transcript, request_id)` in slot order) for a proof
    /// unit (RFC-0071 'The slots').
    ///
    /// With one request, returns its [`Self::request_tokens`] when non-empty. With two or more
    /// requests, requires a power-of-two `slot_len` bounding every request length, and
    /// concatenates their real tokens in one pre-sized clearing buffer. The prover inserts
    /// the slot gaps (RFC-0071 'The tokens', 05 §3 OBS-6).
    #[allow(clippy::disallowed_methods)]
    pub fn batch_tokens(
        previous: Option<&VerifiedLog>,
        current: &VerifiedLog,
        slot_len: u32,
        requests: &[(&Self, u64)],
    ) -> Option<Secret<Vec<u32>>> {
        if let &[(transcript, request_id)] = requests {
            let single = transcript.request_tokens(previous, current, request_id)?;
            if single.expose_secret().is_empty() {
                return None;
            }
            return Some(single);
        }
        if requests.is_empty() || !slot_len.is_power_of_two() {
            return None;
        }
        let s = slot_len as usize;
        let total = requests
            .iter()
            .try_fold(0usize, |total, &(transcript, id)| {
                let seq = transcript.request_tokens(previous, current, id)?;
                let len = seq.expose_secret().len();
                if len == 0 || len > s {
                    return None;
                }
                total.checked_add(len)
            })?;
        let mut packed = Secret::new(vec![0u32; total]);
        let dst = packed.expose_secret_mut();
        let mut offset = 0;
        for &(transcript, request_id) in requests {
            let seq = transcript.request_tokens(previous, current, request_id)?;
            let src = seq.expose_secret();
            if src.is_empty() || src.len() > s {
                return None;
            }
            dst[offset..offset + src.len()].copy_from_slice(src);
            offset += src.len();
        }
        Some(packed)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{ManifestBody, key_id, public_key};
    use crate::{Session, merkle};

    /// The verified log of `entries`, numbered from `first_seq`.
    fn log(entries: Vec<DigestLogEntry>, first_seq: u64) -> VerifiedLog {
        let seed = Secret::new(HeapKey::<32>::from_box(Box::new([0x29; 32])));
        let leaves: Vec<_> = entries.iter().map(merkle::leaf_hash).collect();
        let manifest = ManifestBody {
            recorder_key_id: key_id(&public_key(&seed)),
            log_id: [6; 16],
            period_start_ns: 0,
            period_end_ns: 100,
            first_seq,
            entry_count: entries.len() as u64,
            entries_root: merkle::root(&leaves),
            prev_manifest: [0; 32],
        }
        .sign(&seed)
        .expect("a signature");
        VerifiedLog::check(&manifest, entries).expect("its log")
    }

    #[allow(clippy::disallowed_methods)]
    fn read(tokens: Option<Secret<Vec<u32>>>) -> Option<Vec<u32>> {
        tokens.map(|t| t.expose_secret().clone())
    }

    /// Request 5's `Out` frames sealed out of index order, request 6 between them, and a frame of
    /// request 5 stored but never logged: the tokens are the logged frames', `In` first, each
    /// direction by index, across the period boundary.
    #[test]
    fn a_requests_tokens_are_its_logged_frames_in_and_then_out_by_index() {
        let (mut session, _, keys) = Session::open().expect("a session");
        let mut transcript = SessionTranscript::new(session.session_id(), keys);
        let mut seal = |request: u64, direction: Direction, index: u32, tokens: &[u32]| {
            let frame =
                |t: &[u32]| TokenFrame::new(request, direction, index, Secret::new(t.to_vec()));
            let sealed = session.seal(&frame(tokens)).expect("sealed");
            transcript.record(&sealed, frame(tokens)).expect("recorded");
            sealed
        };
        let prompt = seal(5, Direction::In, 0, &[1, 2]);
        let other = seal(6, Direction::In, 0, &[9]);
        let second = seal(5, Direction::Out, 1, &[0]);
        let first = seal(5, Direction::Out, 0, &[2, 6]);
        let _unlogged = seal(5, Direction::Out, 2, &[7, 7]);
        let previous = log(
            vec![
                DigestLogEntry::frame(0, 1, &prompt),
                DigestLogEntry::frame(1, 1, &other),
            ],
            0,
        );
        let current = log(
            vec![
                DigestLogEntry::frame(2, 2, &second),
                DigestLogEntry::frame(3, 2, &first),
            ],
            2,
        );
        assert_eq!(
            read(transcript.request_tokens(Some(&previous), &current, 5)),
            Some(vec![1, 2, 2, 6, 0])
        );
        assert_eq!(
            read(transcript.request_tokens(Some(&previous), &current, 6)),
            Some(vec![9])
        );
        // Outside the window, the prompt is not the request's.
        assert_eq!(
            read(transcript.request_tokens(None, &current, 5)),
            Some(vec![2, 6, 0])
        );
        // A gap in the indices: request 5's `Out` frame 0 not logged.
        let gap = log(vec![DigestLogEntry::frame(2, 2, &second)], 2);
        assert_eq!(
            read(transcript.request_tokens(Some(&previous), &gap, 5)),
            None
        );
        // An index logged twice.
        let twice = log(
            vec![
                DigestLogEntry::frame(2, 2, &first),
                DigestLogEntry::frame(3, 2, &first),
                DigestLogEntry::frame(4, 2, &second),
            ],
            2,
        );
        assert_eq!(
            read(transcript.request_tokens(Some(&previous), &twice, 5)),
            None
        );
        // An entry the stored frame at its counter does not match: `Out` frame 0 pointing at the
        // prompt's counter, a frame of the same length in the other direction; or at its own,
        // with another length.
        assert_eq!(first.header.length, prompt.header.length);
        let mut elsewhere = DigestLogEntry::frame(3, 2, &first);
        let mut resized = elsewhere;
        if let DigestLogEntry::Frame { frame_seq, .. } = &mut elsewhere {
            *frame_seq = prompt.header.frame_seq;
        }
        if let DigestLogEntry::Frame { length, .. } = &mut resized {
            *length = second.header.length;
        }
        for moved in [elsewhere, resized] {
            let pointed = log(vec![DigestLogEntry::frame(2, 2, &second), moved], 2);
            assert_eq!(
                read(transcript.request_tokens(Some(&previous), &pointed, 5)),
                None,
                "{moved:?}"
            );
        }
        // A logged frame with no stored frame at its counter.
        let (_, _, keys) = Session::open().expect("a session");
        let empty = SessionTranscript::new(transcript.session_id(), keys);
        assert_eq!(
            read(empty.request_tokens(Some(&previous), &current, 5)),
            None
        );
        // No frame of the request in the window: no tokens, not a refusal.
        assert_eq!(
            read(transcript.request_tokens(Some(&previous), &current, 99)),
            Some(Vec::new())
        );

        // Multi-request input concatenates real sequences; the prover inserts slot padding.
        assert_eq!(
            read(SessionTranscript::batch_tokens(
                Some(&previous),
                &current,
                8,
                &[(&transcript, 5), (&transcript, 6)],
            )),
            Some(vec![1, 2, 2, 6, 0, 9])
        );
        // Single-request batch_tokens matches request_tokens without slot padding, and refuses 0 tokens.
        assert_eq!(
            read(SessionTranscript::batch_tokens(
                Some(&previous),
                &current,
                8,
                &[(&transcript, 5)],
            )),
            Some(vec![1, 2, 2, 6, 0])
        );
        assert_eq!(
            read(SessionTranscript::batch_tokens(
                Some(&previous),
                &current,
                8,
                &[(&transcript, 99)],
            )),
            None
        );
        // Refuses empty batches, non-power-of-two slot_len, or a sequence longer than slot_len.
        assert_eq!(
            read(SessionTranscript::batch_tokens(
                Some(&previous),
                &current,
                8,
                &[]
            )),
            None
        );
        assert_eq!(
            read(SessionTranscript::batch_tokens(
                Some(&previous),
                &current,
                6,
                &[(&transcript, 5), (&transcript, 6)],
            )),
            None
        );
        assert_eq!(
            read(SessionTranscript::batch_tokens(
                Some(&previous),
                &current,
                4,
                &[(&transcript, 5), (&transcript, 6)],
            )),
            None
        );
    }
}
