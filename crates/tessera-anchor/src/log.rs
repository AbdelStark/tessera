// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The recorder's digest-log entries, schema `0x0401` (RFC-0015 'Log format', with RFC-0059's
//! `frame_seq`). All of it is public.

use crate::frame::{Direction, SealedFrame, SessionOpenFrame};
use tessera_codec::{Decode, Encode};

/// One entry of the recorder's digest log (schema `0x0401`, version 1). TCE: a `u16` tag, then
/// the variant's fields in order (RFC-0003 'Composite types').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub enum DigestLogEntry {
    /// A session opened: its commitments, logged before any of its frames (INV-ANC-2).
    SessionOpen {
        /// The entry's position in the log.
        seq: u64,
        /// The recorder's clock, in nanoseconds.
        t_ns: u64,
        /// The session's identifier.
        session_id: [u8; 16],
        /// `SHA-256("tessera-anchor-key/v1" ‖ k_s ‖ salt_k)`.
        key_commitment: [u8; 32],
    },
    /// A token frame passed: its header's fields and its digest.
    Frame {
        /// The entry's position in the log.
        seq: u64,
        /// The recorder's clock, in nanoseconds.
        t_ns: u64,
        /// The session's identifier.
        session_id: [u8; 16],
        /// The request the frame belongs to.
        request_id: u64,
        /// The frame's direction.
        direction: Direction,
        /// The frame's position among its request's frames in its direction.
        frame_index: u32,
        /// The ciphertext's length in bytes.
        length: u32,
        /// The session counter, which fixes the frame's CTR counter blocks (RFC-0059).
        frame_seq: u32,
        /// `SHA-256(header ‖ ciphertext)`.
        sha256: [u8; 32],
    },
    /// A session closed.
    SessionClose {
        /// The entry's position in the log.
        seq: u64,
        /// The recorder's clock, in nanoseconds.
        t_ns: u64,
        /// The session's identifier.
        session_id: [u8; 16],
    },
}

impl DigestLogEntry {
    /// The entry's position in the log.
    pub fn seq(&self) -> u64 {
        match self {
            Self::SessionOpen { seq, .. }
            | Self::Frame { seq, .. }
            | Self::SessionClose { seq, .. } => *seq,
        }
    }

    /// The recorder's clock at the entry, in nanoseconds.
    pub fn t_ns(&self) -> u64 {
        match self {
            Self::SessionOpen { t_ns, .. }
            | Self::Frame { t_ns, .. }
            | Self::SessionClose { t_ns, .. } => *t_ns,
        }
    }

    /// The entry a recorder writes for `open` at position `seq` and time `t_ns`.
    pub fn session_open(seq: u64, t_ns: u64, open: &SessionOpenFrame) -> Self {
        Self::SessionOpen {
            seq,
            t_ns,
            session_id: open.session_id,
            key_commitment: open.key_commitment,
        }
    }

    /// The entry a recorder writes for `frame`: its header's fields, copied, and its digest.
    pub fn frame(seq: u64, t_ns: u64, frame: &SealedFrame) -> Self {
        let h = &frame.header;
        Self::Frame {
            seq,
            t_ns,
            session_id: h.session_id,
            request_id: h.request_id,
            direction: h.direction,
            frame_index: h.frame_index,
            length: h.length,
            frame_seq: h.frame_seq,
            sha256: frame.digest(),
        }
    }
}
