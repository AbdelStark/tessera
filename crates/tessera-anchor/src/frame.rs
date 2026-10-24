// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The frames of RFC-0015 'Frames', with RFC-0059's header and counter: the cleartext header,
//! the session-open frame, a sealed frame and its digest, and the plaintext token frame.

use core::fmt;
use sha2::{Digest, Sha256};
use tessera_codec::error::ErrorCode;
use tessera_codec::{Decode, Encode, Secret};

/// A frame's direction: into the model (the prompt) or out of it (the sampled tokens).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub enum Direction {
    /// Inbound: header byte 0, TCE tag 0.
    In,
    /// Outbound: header byte 1, TCE tag 1.
    Out,
}

impl Direction {
    /// The header's byte: 0 for [`Direction::In`], 1 for [`Direction::Out`].
    pub fn byte(self) -> u8 {
        match self {
            Self::In => 0,
            Self::Out => 1,
        }
    }

    /// The direction of a header byte.
    ///
    /// # Errors
    /// [`AnchorError::Direction`] for a byte other than 0 or 1.
    pub fn from_byte(byte: u8) -> Result<Self, AnchorError> {
        match byte {
            0 => Ok(Self::In),
            1 => Ok(Self::Out),
            _ => Err(AnchorError::Direction),
        }
    }
}

/// Why a frame was refused or could not be sealed. Errors carry no frame content (05 §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnchorError {
    /// A header or session-open frame without its magic, `TSFR` or `TSOP`.
    Magic,
    /// A header direction other than 0 or 1, or a plaintext direction tag other than 0 or 1.
    Direction,
    /// Fewer bytes than the value needs: a header under 41 bytes, a session-open frame under 84,
    /// a ciphertext shorter than its header's `length`, or a plaintext shorter than its count's
    /// `22 + 4n`.
    Truncated,
    /// More bytes than the value holds, or a length no value has: a header over 41 bytes, a
    /// session-open frame over 84, a ciphertext longer than its header's `length`, a plaintext
    /// longer than `22 + 4n`, or a header `length` that is not `22 + 4n` for any `n`.
    Noncanonical,
    /// A plaintext of `2^32` bytes or more, which a header's `length` cannot state.
    TooLong,
    /// A plaintext whose request, direction or frame index differs from its header's, or a
    /// frame of another session than the cipher's.
    Mismatch,
    /// The session has sealed `2^32` frames, the most its counter numbers (RFC-0059).
    Exhausted,
    /// The operating system's random source failed.
    Random,
}

impl AnchorError {
    /// The stable error code (04 §3).
    pub fn code(self) -> ErrorCode {
        match self {
            Self::Magic => ErrorCode::CodecMagic,
            Self::Direction => ErrorCode::CodecNoncanonical,
            Self::Truncated => ErrorCode::CodecTruncated,
            Self::Noncanonical => ErrorCode::CodecNoncanonical,
            Self::TooLong => ErrorCode::CodecLimit,
            Self::Mismatch => ErrorCode::WitnessAnchor,
            Self::Exhausted => ErrorCode::CodecLimit,
            Self::Random => ErrorCode::ResIo,
        }
    }
}

impl fmt::Display for AnchorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let what = match self {
            Self::Magic => "a frame without its magic",
            Self::Direction => "a direction other than In (0) or Out (1)",
            Self::Truncated => "a frame shorter than its length",
            Self::Noncanonical => "a frame longer than its length, or a length no frame has",
            Self::TooLong => "a plaintext of 2^32 bytes or more",
            Self::Mismatch => "a plaintext that disagrees with its header",
            Self::Exhausted => "the session's 2^32 frame counters are spent",
            Self::Random => "the operating system's random source failed",
        };
        write!(f, "{}: {what}", self.code())
    }
}

impl std::error::Error for AnchorError {}

/// A frame's cleartext header (RFC-0059 'The header'): 41 bytes, its TCE encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrameHeader {
    /// The session's identifier.
    pub session_id: [u8; 16],
    /// The request the frame belongs to.
    pub request_id: u64,
    /// The frame's direction.
    pub direction: Direction,
    /// The frame's position among its request's frames in its direction, from 0.
    pub frame_index: u32,
    /// The ciphertext's length in bytes, `|TCE(TokenFrame)|`.
    pub length: u32,
    /// The session counter: the frame's position among every frame the session sealed, from 0.
    pub frame_seq: u32,
}

impl FrameHeader {
    /// The header's magic, `TSFR`.
    pub const MAGIC: [u8; 4] = *b"TSFR";
    /// The header's length in bytes.
    pub const LEN: usize = 41;

    /// The header's 41 bytes: magic, session, request, direction byte, index, length, counter,
    /// integers little-endian.
    pub fn to_bytes(&self) -> [u8; Self::LEN] {
        let mut out = [0u8; Self::LEN];
        out[..4].copy_from_slice(&Self::MAGIC);
        out[4..20].copy_from_slice(&self.session_id);
        out[20..28].copy_from_slice(&self.request_id.to_le_bytes());
        out[28] = self.direction.byte();
        out[29..33].copy_from_slice(&self.frame_index.to_le_bytes());
        out[33..37].copy_from_slice(&self.length.to_le_bytes());
        out[37..41].copy_from_slice(&self.frame_seq.to_le_bytes());
        out
    }

    /// The header encoded by `bytes`, exactly 41 of them.
    ///
    /// # Errors
    /// In order: [`AnchorError::Truncated`] or [`AnchorError::Noncanonical`] for fewer or more
    /// bytes, [`AnchorError::Magic`], [`AnchorError::Direction`], and
    /// [`AnchorError::Noncanonical`] for a `length` that is not `22 + 4n`, which no plaintext
    /// has (RFC-0059 'The header').
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, AnchorError> {
        let bytes: &[u8; Self::LEN] = bytes
            .try_into()
            .map_err(|_| sized(bytes.len(), Self::LEN))?;
        if bytes[..4] != Self::MAGIC {
            return Err(AnchorError::Magic);
        }
        let u32_at = |at: usize| {
            u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
        };
        let mut session_id = [0u8; 16];
        session_id.copy_from_slice(&bytes[4..20]);
        let mut request = [0u8; 8];
        request.copy_from_slice(&bytes[20..28]);
        let header = Self {
            session_id,
            request_id: u64::from_le_bytes(request),
            direction: Direction::from_byte(bytes[28])?,
            frame_index: u32_at(29),
            length: u32_at(33),
            frame_seq: u32_at(37),
        };
        match header.length.checked_sub(22) {
            Some(tokens) if tokens.is_multiple_of(4) => Ok(header),
            _ => Err(AnchorError::Noncanonical),
        }
    }
}

/// The error for `have` bytes where a value takes exactly `want`.
fn sized(have: usize, want: usize) -> AnchorError {
    if have < want {
        AnchorError::Truncated
    } else {
        AnchorError::Noncanonical
    }
}

/// A sealed frame on the wire: its header and its ciphertext (RFC-0015 'Frames'). Public.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SealedFrame {
    /// The cleartext header.
    pub header: FrameHeader,
    /// `TCE(TokenFrame)` under the session's keystream; `header.length` bytes.
    pub ciphertext: Vec<u8>,
}

impl SealedFrame {
    /// `header ‖ ciphertext`, the bytes on the wire.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(FrameHeader::LEN + self.ciphertext.len());
        out.extend_from_slice(&self.header.to_bytes());
        out.extend_from_slice(&self.ciphertext);
        out
    }

    /// The frame on the wire in `bytes`.
    ///
    /// # Errors
    /// [`AnchorError::Truncated`] under 41 bytes, [`FrameHeader::from_bytes`]'s on the first 41,
    /// then [`AnchorError::Truncated`] or [`AnchorError::Noncanonical`] unless exactly `length`
    /// bytes follow.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, AnchorError> {
        if bytes.len() < FrameHeader::LEN {
            return Err(AnchorError::Truncated);
        }
        let (head, ciphertext) = bytes.split_at(FrameHeader::LEN);
        let header = FrameHeader::from_bytes(head)?;
        if ciphertext.len() as u64 != u64::from(header.length) {
            return Err(if (ciphertext.len() as u64) < u64::from(header.length) {
                AnchorError::Truncated
            } else {
                AnchorError::Noncanonical
            });
        }
        Ok(Self {
            header,
            ciphertext: ciphertext.to_vec(),
        })
    }

    /// `SHA-256(header ‖ ciphertext)`, the digest the frame's log entry records (RFC-0015 'Log
    /// format').
    pub fn digest(&self) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(self.header.to_bytes());
        hash.update(&self.ciphertext);
        hash.finalize().into()
    }
}

/// The cleartext session-open frame (RFC-0015 'Frames'): 84 bytes, its TCE encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SessionOpenFrame {
    /// The session's identifier.
    pub session_id: [u8; 16],
    /// `SHA-256("tessera-anchor-key/v1" ‖ k_s ‖ salt_k)`.
    pub key_commitment: [u8; 32],
    /// `SHA-256("tessera-sampler-key/v1" ‖ K_samp ‖ salt_s)`.
    pub sampler_commitment: [u8; 32],
}

impl SessionOpenFrame {
    /// The frame's magic, `TSOP`.
    pub const MAGIC: [u8; 4] = *b"TSOP";
    /// The frame's length in bytes.
    pub const LEN: usize = 84;

    /// The frame's 84 bytes: magic, session, key commitment, sampler commitment.
    pub fn to_bytes(&self) -> [u8; Self::LEN] {
        let mut out = [0u8; Self::LEN];
        out[..4].copy_from_slice(&Self::MAGIC);
        out[4..20].copy_from_slice(&self.session_id);
        out[20..52].copy_from_slice(&self.key_commitment);
        out[52..84].copy_from_slice(&self.sampler_commitment);
        out
    }

    /// The frame encoded by `bytes`, exactly 84 of them.
    ///
    /// # Errors
    /// [`AnchorError::Truncated`] or [`AnchorError::Noncanonical`] for fewer or more bytes, then
    /// [`AnchorError::Magic`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, AnchorError> {
        let bytes: &[u8; Self::LEN] = bytes
            .try_into()
            .map_err(|_| sized(bytes.len(), Self::LEN))?;
        if bytes[..4] != Self::MAGIC {
            return Err(AnchorError::Magic);
        }
        let mut frame = Self {
            session_id: [0; 16],
            key_commitment: [0; 32],
            sampler_commitment: [0; 32],
        };
        frame.session_id.copy_from_slice(&bytes[4..20]);
        frame.key_commitment.copy_from_slice(&bytes[20..52]);
        frame.sampler_commitment.copy_from_slice(&bytes[52..84]);
        Ok(frame)
    }
}

/// A plaintext token frame (RFC-0015 'Frames'). Its tokens are secret-class (05 §3) and live in
/// a [`Secret`]; Debug shows the identifiers and the token count only.
pub struct TokenFrame {
    /// The request the frame belongs to.
    pub request_id: u64,
    /// The frame's direction.
    pub direction: Direction,
    /// The frame's position among its request's frames in its direction, from 0.
    pub frame_index: u32,
    pub(crate) tokens: Secret<Vec<u32>>,
}

impl TokenFrame {
    /// The frame of `tokens`.
    pub fn new(
        request_id: u64,
        direction: Direction,
        frame_index: u32,
        tokens: Secret<Vec<u32>>,
    ) -> Self {
        Self {
            request_id,
            direction,
            frame_index,
            tokens,
        }
    }

    /// The tokens, still in their [`Secret`].
    pub fn tokens(&self) -> &Secret<Vec<u32>> {
        &self.tokens
    }

    /// `|TCE(TokenFrame)| = 22 + 4n` for `n` tokens: the ciphertext's length (RFC-0059 'The
    /// plaintext').
    pub fn plaintext_len(&self) -> u64 {
        #[allow(clippy::disallowed_methods)]
        let n = self.tokens.expose_secret().len() as u64;
        22 + 4 * n
    }
}

impl fmt::Debug for TokenFrame {
    #[allow(clippy::disallowed_methods)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TokenFrame")
            .field("request_id", &self.request_id)
            .field("direction", &self.direction)
            .field("frame_index", &self.frame_index)
            .field("tokens", &self.tokens.expose_secret().len())
            .finish()
    }
}
