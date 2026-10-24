// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Digest-log formats, log manifests and the software reference recorder (RFC-0015).
//!
//! This version holds the frames and the log entries, with RFC-0059's header and counter:
//!
//! - [`FrameHeader`], the 41-byte cleartext header, which carries the session counter
//!   `frame_seq`; [`SealedFrame`], a header and its ciphertext, with the digest the log records;
//!   [`SessionOpenFrame`], the 84-byte frame that commits a session's keys before its frames.
//! - [`TokenFrame`], the plaintext, whose tokens are secret-class (05 §3).
//! - [`Session`], the reference cipher: it draws the session's identifier and secrets
//!   ([`SessionKeys`]), sends their commitments, and seals frames under AES-128-CTR with the
//!   counter it keeps. No operation takes a counter, a key or a session identifier from its
//!   caller, so a session never reuses a nonce (INV-ANC-3). [`key_commitment`] and
//!   [`sampler_commitment`] are the commitments.
//! - [`DigestLogEntry`], the recorder's entries (schema `0x0401`).
//!
//! Manifests and their signatures (#138) and the reference recorder (#139) come next.

#![forbid(unsafe_code)]

mod frame;
mod log;
mod session;
#[cfg(test)]
mod vectors;

pub use frame::{AnchorError, Direction, FrameHeader, SealedFrame, SessionOpenFrame, TokenFrame};
pub use log::DigestLogEntry;
pub use session::{HeapKey, Session, SessionKeys, key_commitment, sampler_commitment};
