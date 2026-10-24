// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Digest-log formats, log manifests and the software reference recorder (RFC-0015).
//!
//! This version holds the frames and the log entries, with RFC-0059's header and counter:
//!
//! - [`FrameHeader`], the 41-byte cleartext header, which carries the session counter
//!   `frame_seq`; [`SealedFrame`], a header and its ciphertext, with the digest the log records;
//!   [`SessionOpenFrame`], the 52-byte frame that commits a session's key before its frames.
//! - [`TokenFrame`], the plaintext, whose tokens are secret-class (05 §3).
//! - [`Session`], the reference cipher: it draws the session's identifier and secrets
//!   ([`SessionKeys`]), sends the key's commitment, and seals frames under AES-128-CTR with the
//!   counter it keeps. No operation takes a counter, a key or a session identifier from its
//!   caller, so a session never reuses a nonce (INV-ANC-3). [`key_commitment`] is a session's
//!   commitment, and [`sampler_commitment`] the sampler key's, which the verifier pins
//!   (RFC-0061).
//! - [`DigestLogEntry`], the recorder's entries (schema `0x0401`).
//! - [`LogManifest`] (schema `0x0402`), with RFC-0060's identity, strict Ed25519 signatures and
//!   chain ([`check_link`], [`check_chain`]), and [`merkle`], RFC 9162's tree over a period's
//!   entries and its inclusion proofs.
//!
//! The reference recorder (#139) comes next.

#![forbid(unsafe_code)]

mod frame;
mod log;
mod manifest;
pub mod merkle;
mod session;
#[cfg(test)]
mod vectors;

pub use frame::{AnchorError, Direction, FrameHeader, SealedFrame, SessionOpenFrame, TokenFrame};
pub use log::DigestLogEntry;
pub use manifest::{
    ChainError, EntryError, LinkError, LogManifest, ManifestBody, ManifestError, SignatureEnvelope,
    check_chain, check_link, key_id, public_key,
};
pub use session::{Session, SessionKeys, key_commitment, sampler_commitment};
/// The heap-held key type, from `tessera-codec`, re-exported for the anchor's callers.
pub use tessera_codec::HeapKey;
