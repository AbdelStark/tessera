// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The reference session cipher of RFC-0059 'The reference cipher': the session's identifier and
//! secrets drawn at open, the commitments, and frames sealed under AES-128-CTR with the counter
//! the session keeps. No operation takes a counter, a key or a session identifier from its
//! caller, and [`Session`] is not `Clone`, so a session never reuses a nonce (INV-ANC-3).

#[cfg(test)]
use crate::frame::Direction;
use crate::frame::{AnchorError, FrameHeader, SealedFrame, SessionOpenFrame, TokenFrame};
use aes::Aes128;
use aes::cipher::{BlockEncrypt, KeyInit, generic_array::GenericArray};
use core::borrow::Borrow;
use core::fmt;
use tessera_codec::{Secret, SecretBuf, Zeroize};

/// The label of the key commitment (RFC-0015 'Frames').
const KEY_LABEL: &[u8] = b"tessera-anchor-key/v1";
/// The label of the sampler commitment (RFC-0015 'Frames').
const SAMPLER_LABEL: &[u8] = b"tessera-sampler-key/v1";

/// Sixteen secret bytes on the heap: moving the value moves a pointer and leaves no copy of the
/// bytes behind (OBS-2). Zeroized through its [`Secret`].
pub struct HeapKey(Box<[u8; 16]>);

impl HeapKey {
    fn zeroed() -> Self {
        Self(Box::new([0; 16]))
    }
}

impl Zeroize for HeapKey {
    fn zeroize(&mut self) {
        self.0.as_mut().zeroize();
    }
}

impl Borrow<[u8; 16]> for HeapKey {
    fn borrow(&self) -> &[u8; 16] {
        &self.0
    }
}

/// The secrets a session draws at open: `k_s`, `salt_k` and `salt_s` (RFC-0059 'The session-open
/// frame and the commitments'). The prover keeps them for the anchor circuits. Each lives on the
/// heap, so moving the keys moves pointers and leaves no copy behind; Debug shows nothing.
pub struct SessionKeys {
    /// `k_s`, the frames' AES-128 key.
    pub key: Secret<HeapKey>,
    /// `salt_k`, the key commitment's salt.
    pub key_salt: Secret<HeapKey>,
    /// `salt_s`, the sampler commitment's salt.
    pub sampler_salt: Secret<HeapKey>,
}

impl fmt::Debug for SessionKeys {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SessionKeys(..)")
    }
}

/// One session's cipher: the key schedule of `k_s`, on the heap and cleared on drop, its
/// identifier and the next counter. Boxing the schedule makes moving the session move a pointer.
/// The schedule is computed on the stack before it is boxed, and that copy, like every stack
/// copy, is beyond reach (OBS-6).
pub struct Session {
    cipher: Box<Aes128>,
    session_id: [u8; 16],
    /// The next frame's `frame_seq`; `2^32` once every counter is spent.
    next: u64,
}

impl fmt::Debug for Session {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Session")
            .field("session_id", &self.session_id)
            .field("next", &self.next)
            .finish_non_exhaustive()
    }
}

impl Session {
    /// Opens a session: draws its identifier, `k_s`, `salt_k` and `salt_s` from the operating
    /// system's generator (RFC-0059 'Session identity'), commits `k_s` and the sampler key
    /// `K_samp` (RFC-0017), and returns the cipher, the session-open frame to send before any
    /// token frame (INV-ANC-2), and the secrets.
    ///
    /// # Errors
    /// [`AnchorError::Random`] when the operating system's generator fails.
    #[allow(clippy::disallowed_methods)]
    pub fn open<K: Borrow<[u8; 16]> + Zeroize>(
        sampler_key: &Secret<K>,
    ) -> Result<(Self, SessionOpenFrame, SessionKeys), AnchorError> {
        let mut drawn = Secret::new(vec![0u8; 64]);
        getrandom::fill(drawn.expose_secret_mut()).map_err(|_| AnchorError::Random)?;
        Ok(Self::from_drawn(&drawn, sampler_key))
    }

    /// The session of 64 drawn bytes: its identifier, then `k_s`, `salt_k` and `salt_s`.
    #[allow(clippy::disallowed_methods)]
    fn from_drawn<K: Borrow<[u8; 16]> + Zeroize>(
        drawn: &Secret<Vec<u8>>,
        sampler_key: &Secret<K>,
    ) -> (Self, SessionOpenFrame, SessionKeys) {
        let bytes = drawn.expose_secret();
        let mut session_id = [0u8; 16];
        session_id.copy_from_slice(&bytes[..16]);
        let part = |i: usize| {
            let mut out = Secret::new(HeapKey::zeroed());
            out.expose_secret_mut()
                .0
                .copy_from_slice(&bytes[16 * i..16 * (i + 1)]);
            out
        };
        let keys = SessionKeys {
            key: part(1),
            key_salt: part(2),
            sampler_salt: part(3),
        };
        let open = SessionOpenFrame {
            session_id,
            key_commitment: key_commitment(&keys.key, &keys.key_salt),
            sampler_commitment: sampler_commitment(sampler_key, &keys.sampler_salt),
        };
        let session = Self {
            cipher: Box::new(Aes128::new(GenericArray::from_slice(
                &keys.key.expose_secret().0[..],
            ))),
            session_id,
            next: 0,
        };
        (session, open, keys)
    }

    /// The session of given secrets, for the conformance vectors: test builds only.
    #[cfg(test)]
    pub(crate) fn from_secrets<K: Borrow<[u8; 16]> + Zeroize>(
        drawn: [u8; 64],
        sampler_key: &Secret<K>,
    ) -> (Self, SessionOpenFrame, SessionKeys) {
        Self::from_drawn(&Secret::new(drawn.to_vec()), sampler_key)
    }

    /// Moves the counter, for tests at its edges: test builds only.
    #[cfg(test)]
    pub(crate) fn set_next(&mut self, next: u64) {
        self.next = next;
    }

    /// The session's identifier.
    pub fn session_id(&self) -> [u8; 16] {
        self.session_id
    }

    /// The counter the next sealed frame takes.
    pub fn next_seq(&self) -> u64 {
        self.next
    }

    /// Seals `frame` with the session's next counter, which it then spends: the header and
    /// `TCE(TokenFrame) ⊕ keystream` (RFC-0059 'The keystream').
    ///
    /// # Errors
    /// [`AnchorError::Exhausted`] once `2^32` frames are sealed, and [`AnchorError::TooLong`] for
    /// a plaintext of `2^32` bytes or more. Neither spends a counter.
    pub fn seal(&mut self, frame: &TokenFrame) -> Result<SealedFrame, AnchorError> {
        let frame_seq = u32::try_from(self.next).map_err(|_| AnchorError::Exhausted)?;
        let length = u32::try_from(frame.plaintext_len()).map_err(|_| AnchorError::TooLong)?;
        // Encrypted in place in clearing storage, so the plaintext never reaches a plain vector;
        // what is copied out is the public ciphertext.
        let mut sealed = plaintext(frame);
        keystream(&self.cipher, &self.session_id, frame_seq, &mut sealed);
        let ciphertext: Vec<u8> = sealed.iter().copied().collect();
        self.next += 1;
        Ok(SealedFrame {
            header: FrameHeader {
                session_id: self.session_id,
                request_id: frame.request_id,
                direction: frame.direction,
                frame_index: frame.frame_index,
                length,
                frame_seq,
            },
            ciphertext,
        })
    }

    /// The token frame `sealed` carries, decrypted with its header's counter: test builds only.
    ///
    /// The public API decrypts nothing (RFC-0059 'The reference cipher'). A decryption that
    /// returned the tokens of a forged frame, one whose prefix decrypts correctly, would return
    /// keystream of a used counter, from which a caller could build a ciphertext under that
    /// counter without sealing. The protocol needs no decryption in the clear: the circuits
    /// check the plaintext's prefix against the header.
    ///
    /// # Errors
    /// In order:
    /// - [`AnchorError::Truncated`] or [`AnchorError::Noncanonical`] for a ciphertext shorter or
    ///   longer than its header's `length`;
    /// - [`AnchorError::Mismatch`] for a frame of another session;
    /// - the plaintext's: [`AnchorError::Truncated`] below its 22-byte prefix,
    ///   [`AnchorError::Direction`] for a bad tag, then [`AnchorError::Truncated`] or
    ///   [`AnchorError::Noncanonical`] for fewer or more bytes than its count's `22 + 4n`;
    /// - [`AnchorError::Mismatch`] for a plaintext whose request, direction or index differs from
    ///   the header's.
    #[cfg(test)]
    pub(crate) fn open_frame(&self, sealed: &SealedFrame) -> Result<TokenFrame, AnchorError> {
        let header = &sealed.header;
        let (have, want) = (sealed.ciphertext.len() as u64, u64::from(header.length));
        if have != want {
            return Err(if have < want {
                AnchorError::Truncated
            } else {
                AnchorError::Noncanonical
            });
        }
        if header.session_id != self.session_id {
            return Err(AnchorError::Mismatch);
        }
        let mut bytes = SecretBuf::with_capacity(sealed.ciphertext.len());
        for &b in &sealed.ciphertext {
            bytes.push(b);
        }
        keystream(&self.cipher, &self.session_id, header.frame_seq, &mut bytes);
        let frame = parse(&bytes)?;
        if (frame.request_id, frame.direction, frame.frame_index)
            != (header.request_id, header.direction, header.frame_index)
        {
            return Err(AnchorError::Mismatch);
        }
        Ok(frame)
    }
}

/// `TCE(TokenFrame)` (RFC-0059 'The plaintext'), in clearing storage: `u64` request, `u16`
/// direction tag, `u32` index, `u64` count, then the `u32` tokens, little-endian.
#[allow(clippy::disallowed_methods)]
pub(crate) fn plaintext(frame: &TokenFrame) -> SecretBuf<u8> {
    let tokens = frame.tokens.expose_secret();
    let mut out = SecretBuf::with_capacity(22 + 4 * tokens.len());
    let mut put = |bytes: &[u8]| {
        for &b in bytes {
            out.push(b);
        }
    };
    put(&frame.request_id.to_le_bytes());
    put(&u16::from(frame.direction.byte()).to_le_bytes());
    put(&frame.frame_index.to_le_bytes());
    put(&(tokens.len() as u64).to_le_bytes());
    for token in tokens {
        put(&token.to_le_bytes());
    }
    out
}

/// The token frame in `bytes`, `TCE(TokenFrame)`: test builds only, as [`Session::open_frame`].
///
/// # Errors
/// In order: [`AnchorError::Truncated`] below the 22-byte prefix, [`AnchorError::Direction`] for
/// a tag other than 0 or 1, then [`AnchorError::Truncated`] or [`AnchorError::Noncanonical`] for
/// fewer or more bytes than the count's `22 + 4n`.
#[cfg(test)]
pub(crate) fn parse(bytes: &[u8]) -> Result<TokenFrame, AnchorError> {
    if bytes.len() < 22 {
        return Err(AnchorError::Truncated);
    }
    let le = |at: usize, n: usize| -> u64 {
        let mut value = 0u64;
        for (i, &b) in bytes[at..at + n].iter().enumerate() {
            value |= u64::from(b) << (8 * i);
        }
        value
    };
    let request_id = le(0, 8);
    let direction = match le(8, 2) {
        0 => Direction::In,
        1 => Direction::Out,
        _ => return Err(AnchorError::Direction),
    };
    let frame_index = le(10, 4) as u32;
    let n = le(14, 8);
    let expected = 22 + 4 * u128::from(n);
    let have = bytes.len() as u128;
    if have != expected {
        return Err(if have < expected {
            AnchorError::Truncated
        } else {
            AnchorError::Noncanonical
        });
    }
    let mut tokens = Secret::new(Vec::with_capacity(n as usize));
    #[allow(clippy::disallowed_methods)]
    let list = tokens.expose_secret_mut();
    for chunk in bytes[22..].chunks_exact(4) {
        list.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(TokenFrame {
        request_id,
        direction,
        frame_index,
        tokens,
    })
}

/// XORs the keystream of counter `frame_seq` into `data`: block `i` is AES-128 of
/// `session_id[0..8] ‖ u32_be(frame_seq) ‖ u32_be(i)` (RFC-0059 'The keystream').
fn keystream(cipher: &Aes128, session_id: &[u8; 16], frame_seq: u32, data: &mut [u8]) {
    let mut block = [0u8; 16];
    block[..8].copy_from_slice(&session_id[..8]);
    block[8..12].copy_from_slice(&frame_seq.to_be_bytes());
    ctr(cipher, block, data);
}

/// NIST SP 800-38A CTR mode from the initial counter block `initial`, with the standard
/// incrementing function on its last 32 bits (Appendix B.1, `m = 32`).
fn ctr(cipher: &Aes128, initial: [u8; 16], data: &mut [u8]) {
    let mut counter = initial;
    for chunk in data.chunks_mut(16) {
        let mut pad = GenericArray::from(counter);
        cipher.encrypt_block(&mut pad);
        for (byte, k) in chunk.iter_mut().zip(pad.iter()) {
            *byte ^= k;
        }
        pad.zeroize();
        let low = u32::from_be_bytes([counter[12], counter[13], counter[14], counter[15]]);
        counter[12..].copy_from_slice(&low.wrapping_add(1).to_be_bytes());
    }
}

/// `SHA-256("tessera-anchor-key/v1" ‖ k_s ‖ salt_k)` (RFC-0059).
pub fn key_commitment<K, S>(key: &Secret<K>, salt: &Secret<S>) -> [u8; 32]
where
    K: Borrow<[u8; 16]> + Zeroize,
    S: Borrow<[u8; 16]> + Zeroize,
{
    commitment(KEY_LABEL, key, salt)
}

/// `SHA-256("tessera-sampler-key/v1" ‖ K_samp ‖ salt_s)` (RFC-0059).
pub fn sampler_commitment<K, S>(key: &Secret<K>, salt: &Secret<S>) -> [u8; 32]
where
    K: Borrow<[u8; 16]> + Zeroize,
    S: Borrow<[u8; 16]> + Zeroize,
{
    commitment(SAMPLER_LABEL, key, salt)
}

/// SHA-256 of `label ‖ key ‖ salt`, at most 54 bytes: one padded block, built in a buffer that
/// is cleared on drop and passed to the compression function by reference, so no hasher's
/// buffer keeps a copy of the key. The compression function's own stack copies are beyond
/// reach, as OBS-6 says of every stack copy.
#[allow(clippy::disallowed_methods)]
fn commitment<K, S>(label: &[u8], key: &Secret<K>, salt: &Secret<S>) -> [u8; 32]
where
    K: Borrow<[u8; 16]> + Zeroize,
    S: Borrow<[u8; 16]> + Zeroize,
{
    let len = label.len() + 32;
    let mut block = Secret::new(vec![0u8; 64]);
    let bytes = &mut block.expose_secret_mut()[..];
    bytes[..label.len()].copy_from_slice(label);
    bytes[label.len()..label.len() + 16].copy_from_slice(key.expose_secret().borrow());
    bytes[label.len() + 16..len].copy_from_slice(salt.expose_secret().borrow());
    bytes[len] = 0x80;
    bytes[56..].copy_from_slice(&(8 * len as u64).to_be_bytes());
    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    sha2::compress256(
        &mut state,
        core::slice::from_ref(GenericArray::from_slice(bytes)),
    );
    let mut digest = [0u8; 32];
    for (out, word) in digest.chunks_exact_mut(4).zip(state) {
        out.copy_from_slice(&word.to_be_bytes());
    }
    digest
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len() / 2)
            .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
            .collect()
    }

    /// NIST SP 800-38A F.5.1, CTR-AES128.Encrypt, from its initial counter block.
    #[test]
    fn ctr_is_sp_800_38a_f_5_1() {
        let key = hex("2b7e151628aed2a6abf7158809cf4f3c");
        let cipher = Aes128::new(GenericArray::from_slice(&key));
        let initial: [u8; 16] = hex("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff").try_into().unwrap();
        let mut data = hex(concat!(
            "6bc1bee22e409f96e93d7e117393172a",
            "ae2d8a571e03ac9c9eb76fac45af8e51",
            "30c81c46a35ce411e5fbc1191a0a52ef",
            "f69f2445df4f9b17ad2b417be66c3710"
        ));
        ctr(&cipher, initial, &mut data);
        assert_eq!(
            data,
            hex(concat!(
                "874d6191b620e3261bef6864990db6ce",
                "9806f66b7970fdff8617187bb9fffdff",
                "5ae4df3edbd5d35e5b4f09020db03eab",
                "1e031dda2fbe03d1792170a0f3009cee"
            ))
        );
    }

    /// The one-block commitment is SHA-256, against `sha2`'s own hasher, at both labels, which
    /// keep a commitment in one block.
    #[test]
    fn the_commitment_block_is_sha_256() {
        use sha2::{Digest, Sha256};
        let key = Secret::new([7u8; 16]);
        let salt = Secret::new([9u8; 16]);
        for label in [KEY_LABEL, SAMPLER_LABEL] {
            let mut direct = Sha256::new();
            direct.update(label);
            direct.update([7u8; 16]);
            direct.update([9u8; 16]);
            let expected: [u8; 32] = direct.finalize().into();
            assert_eq!(commitment(label, &key, &salt), expected);
            assert!(label.len() + 32 <= 55, "one padded block");
        }
    }

    fn session() -> Session {
        Session::from_secrets([0x31; 64], &Secret::new([1u8; 16])).0
    }

    /// INV-ANC-3 at its edge: the last counter seals, and the frame after it is refused without
    /// spending anything.
    #[test]
    fn the_counter_never_wraps() {
        let mut session = session();
        session.set_next(u64::from(u32::MAX));
        let frame = TokenFrame::new(1, Direction::Out, 0, Secret::new(vec![5, 6]));
        assert_eq!(session.seal(&frame).unwrap().header.frame_seq, u32::MAX);
        assert_eq!(session.seal(&frame), Err(AnchorError::Exhausted));
        assert_eq!(session.seal(&frame), Err(AnchorError::Exhausted));
        assert_eq!(session.next_seq(), 1 << 32);
    }

    fn frame(direction: Direction, index: u32, tokens: Vec<u32>) -> TokenFrame {
        TokenFrame::new(77, direction, index, Secret::new(tokens))
    }

    /// A frame whose plaintext disagrees with its header, or whose ciphertext is tampered with, does
    /// not open as a token frame.
    #[test]
    fn a_plaintext_must_agree_with_its_header() {
        let mut s = session();
        let sealed = s.seal(&frame(Direction::In, 3, vec![10, 20, 30])).unwrap();
        for change in 0..4 {
            let mut forged = sealed.clone();
            match change {
                0 => forged.header.request_id ^= 1,
                1 => forged.header.direction = Direction::Out,
                2 => forged.header.frame_index += 1,
                _ => forged.header.session_id[15] ^= 1,
            }
            assert_eq!(
                s.open_frame(&forged).map(|_| ()),
                Err(AnchorError::Mismatch),
                "{change}"
            );
        }
        // The plaintext's direction tag at byte 8: a flip there gives tag 2.
        let mut forged = sealed.clone();
        forged.ciphertext[8] ^= 2;
        assert_eq!(
            s.open_frame(&forged).map(|_| ()),
            Err(AnchorError::Direction)
        );
        // Its token count at bytes 14..22: one more token than the bytes hold.
        let mut forged = sealed.clone();
        forged.ciphertext[14] ^= 7;
        assert_eq!(
            s.open_frame(&forged).map(|_| ()),
            Err(AnchorError::Truncated)
        );
        // A header counter other than the frame's own decrypts to noise.
        let mut forged = sealed;
        forged.header.frame_seq += 1;
        assert!(s.open_frame(&forged).is_err());
    }

    /// Each refusal of the plaintext decoder, alone.
    #[test]
    fn the_plaintext_decoder_refuses_each_malformed_shape() {
        let frame = TokenFrame::new(9, Direction::In, 4, Secret::new(vec![7, 8, 9]));
        let honest = plaintext(&frame);
        assert!(parse(&honest).is_ok());
        // Below the 22-byte prefix, down to nothing: no panic.
        for len in [0, 1, 13, 21] {
            assert_eq!(
                parse(&honest[..len]).map(|_| ()),
                Err(AnchorError::Truncated),
                "{len}"
            );
        }
        // One to three bytes past the count's tokens.
        for extra in 1..=3 {
            let mut longer = honest.to_vec();
            longer.extend(std::iter::repeat_n(0, extra));
            assert_eq!(
                parse(&longer).map(|_| ()),
                Err(AnchorError::Noncanonical),
                "{extra}"
            );
        }
        // A token short.
        assert_eq!(
            parse(&honest[..honest.len() - 4]).map(|_| ()),
            Err(AnchorError::Truncated)
        );
        // The tag's high byte: 0x0100 is no direction, though its low byte is In's.
        let mut tag = honest.to_vec();
        tag[9] = 1;
        assert_eq!(parse(&tag).map(|_| ()), Err(AnchorError::Direction));
        // A count of 2^64 − 1 neither overflows nor allocates.
        let mut huge = honest.to_vec();
        huge[14..22].copy_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(parse(&huge).map(|_| ()), Err(AnchorError::Truncated));
    }

    /// The ciphertext must be its header's length, whatever the plaintext's own count says.
    #[test]
    fn a_header_length_other_than_the_ciphertexts_is_refused() {
        let mut session = session();
        let sealed = session
            .seal(&TokenFrame::new(
                3,
                Direction::Out,
                0,
                Secret::new(vec![1, 2]),
            ))
            .unwrap();
        let mut forged = sealed.clone();
        forged.header.length += 4;
        assert_eq!(
            session.open_frame(&forged).map(|_| ()),
            Err(AnchorError::Truncated)
        );
        let mut forged = sealed;
        forged.header.length -= 4;
        assert_eq!(
            session.open_frame(&forged).map(|_| ()),
            Err(AnchorError::Noncanonical)
        );
    }
}
