// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The reference sampler of RFC-0017 'Sampler PRF', as RFC-0061 amends it, which serving engines
//! link or port so that they draw exactly the stream the circuit proves.
//!
//! - **The class.** A request whose prompt is the tokens `x_0, …, x_{n−1}` has, under the sampler
//!   key `K` with `b` class bits, the stream class `u64_le(h[0..8]) >> (64 − b)` (0 for `b = 0`),
//!   where `h = SHA-256("tessera-stream-class/v1" ‖ K ‖ 0^25 ‖ u64_le(n) ‖ u32_le(x_0) ‖ …)`
//!   ([`Sampler::class`]). However a serving stack varies a prompt, it reaches at most `2^b`
//!   streams (RFC-0061 'The stream').
//! - **The stream.** Output position `t` draws
//!   `u_t = u32_le(AES-128_K(u64_le(class) ‖ u32_le(t) ‖ 0^4)[0..4])` ([`Sampler::draw`]).
//! - **The selection.** Given the vocabulary's non-negative integer weights `e_v` and
//!   `Z = Σ_v e_v`, the sampled token is `min{v : C(v)·2^32 > u·Z}` with `C(v) = Σ_{w ≤ v} e_w`
//!   ([`inverse_cdf`], RFC-0009 'InverseCdf', Lemma 10).
//!
//! `K`, the prompt, the class, the stream and the sampled tokens are secret-class (05 §3). The key
//! lives in a [`Secret`] on the heap, the class's hash input in a buffer that clears itself, and
//! the cipher's key schedule is cleared on drop (OBS-2). A class, a draw and a token are single
//! values a caller must keep out of logs, errors and reports.

use aes::Aes128;
use aes::cipher::{BlockEncrypt, KeyInit, generic_array::GenericArray};
use core::borrow::{Borrow, BorrowMut};
use core::fmt;
use tessera_codec::{HeapKey, Secret, SecretBuf};
use zeroize::Zeroize;

/// The prover's base key `K_p` (RFC-0061 'The sampler key and its pins'), whose commitment the
/// verifier pins. The PRF never runs under it: each pin derives its own [`SamplerKey`]
/// ([`SamplerBaseKey::derive`]). Its bytes are on the heap ([`HeapKey`]), so moving it leaves no
/// copy behind (OBS-2).
pub struct SamplerBaseKey(Secret<HeapKey>);

/// The PRF's key `K` for one pin, derived from the base key ([`SamplerBaseKey::derive`]), with the
/// pin's class bits `b`, which the derivation hashed. It serves every request of its pin's
/// window. Its bytes are on the heap ([`HeapKey`]), so moving it leaves no copy behind (OBS-2).
pub struct SamplerKey {
    key: Secret<HeapKey>,
    bits: ClassBits,
}

/// The derivation's label, 25 bytes of ASCII without a terminator (RFC-0061 'The sampler key and
/// its pins').
const DERIVE_LABEL: &[u8] = b"tessera-sampler-derive/v1";

impl SamplerBaseKey {
    /// The base key `key`.
    pub fn new(key: Secret<HeapKey>) -> Self {
        Self(key)
    }

    /// The key, still in its [`Secret`]: for its commitment (RFC-0059, RFC-0061).
    pub fn secret(&self) -> &Secret<HeapKey> {
        &self.0
    }

    /// The PRF's key for the pin with verifier nonce `nonce`, `bits` class bits and start
    /// `from_ns`:
    /// `SHA-256("tessera-sampler-derive/v1" ‖ K_p ‖ 0^23 ‖ nonce ‖ u8(b) ‖ u64_le(from_ns))[0..16]`.
    /// The label and `K_p` fill the first block alone, as in the class hash.
    ///
    /// The 105-byte message and its padding are built in a [`HeapKey`] that clears itself, and
    /// the compression state is cleared after it is read. The compression function's own stack
    /// copies are beyond reach, as OBS-6 says of every stack copy.
    #[allow(clippy::disallowed_methods)]
    pub fn derive(&self, nonce: &[u8; 32], bits: ClassBits, from_ns: u64) -> SamplerKey {
        let base: &[u8; 16] = self.0.expose_secret().borrow();
        let len = 64 + 32 + 1 + 8;
        let mut message = Secret::new(HeapKey::<128>::zeroed());
        let bytes: &mut [u8; 128] = message.expose_secret_mut().borrow_mut();
        bytes[..25].copy_from_slice(DERIVE_LABEL);
        bytes[25..41].copy_from_slice(base);
        bytes[64..96].copy_from_slice(nonce);
        bytes[96] = bits.get();
        bytes[97..105].copy_from_slice(&from_ns.to_le_bytes());
        bytes[len] = 0x80;
        bytes[120..].copy_from_slice(&(8 * len as u64).to_be_bytes());
        let mut state = SHA256_IV;
        for block in bytes.chunks_exact(64) {
            sha2::compress256(
                &mut state,
                core::slice::from_ref(GenericArray::from_slice(block)),
            );
        }
        let mut key = Secret::new(HeapKey::zeroed());
        let out: &mut [u8; 16] = key.expose_secret_mut().borrow_mut();
        for (chunk, word) in out.chunks_exact_mut(4).zip(state) {
            chunk.copy_from_slice(&word.to_be_bytes());
        }
        state.zeroize();
        SamplerKey { key, bits }
    }
}

/// Shows nothing of the key (05 §3).
impl fmt::Debug for SamplerBaseKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SamplerBaseKey(..)")
    }
}

impl SamplerKey {
    /// The PRF's key `key` with `bits` class bits, for a key derived elsewhere, such as a test
    /// vector's.
    pub fn new(key: Secret<HeapKey>, bits: ClassBits) -> Self {
        Self { key, bits }
    }

    /// The class bits `b` the key was derived with.
    pub fn bits(&self) -> ClassBits {
        self.bits
    }
}

/// Shows nothing of the key (05 §3).
impl fmt::Debug for SamplerKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SamplerKey(..)")
    }
}

/// The number of class bits `b`, at most 64, which the verifier pins with the key's commitment
/// (RFC-0061 'The sampler key and its pins'). It is public.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClassBits(u8);

impl ClassBits {
    /// The largest `b`: a class fills RFC-0017's 64-bit `request_id` slot.
    pub const MAX: u8 = 64;

    /// `b`, or `None` above [`ClassBits::MAX`].
    pub fn new(bits: u8) -> Option<Self> {
        (bits <= Self::MAX).then_some(Self(bits))
    }

    /// `b`.
    pub fn get(self) -> u8 {
        self.0
    }
}

/// A request's stream class, below `2^b`. It depends on the key and the prompt, so it is
/// secret-class, and `Debug` shows nothing of it (05 §3).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct StreamClass(u64);

/// Shows nothing of the class (05 §3).
impl fmt::Debug for StreamClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("StreamClass(..)")
    }
}

/// The class hash's label, 23 bytes of ASCII without a terminator (RFC-0061 'The stream').
const CLASS_LABEL: &[u8] = b"tessera-stream-class/v1";

/// RFC-0017's PRF under one sampler key and class width, as RFC-0061 amends it. It keeps:
/// - AES-128's key schedule, whose first round key is the key itself;
/// - SHA-256's state after the class hash's key block, `"tessera-stream-class/v1" ‖ K ‖ 0^25`.
///
/// Both are on the heap, so moving the sampler moves pointers, and both are cleared on drop.
pub struct Sampler {
    cipher: Box<Aes128>,
    midstate: Secret<HeapKey<32>>,
    bits: ClassBits,
}

/// SHA-256's initial state (FIPS 180-4 §5.3.3).
const SHA256_IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

impl Sampler {
    /// The sampler of `key`, with the class bits it was derived with.
    ///
    /// `key` is the PRF's key `K` of one pin ([`SamplerBaseKey::derive`]). The key block is built
    /// in a [`HeapKey`] that clears itself, and its compression's state is kept, as big-endian
    /// words, in another; the local copy is cleared. The compression function's own stack copies
    /// are beyond reach, as OBS-6 says of every stack copy.
    #[allow(clippy::disallowed_methods)]
    pub fn new(key: &SamplerKey) -> Self {
        let bytes: &[u8; 16] = key.key.expose_secret().borrow();
        let mut block = Secret::new(HeapKey::<64>::zeroed());
        let block_bytes: &mut [u8; 64] = block.expose_secret_mut().borrow_mut();
        block_bytes[..CLASS_LABEL.len()].copy_from_slice(CLASS_LABEL);
        block_bytes[CLASS_LABEL.len()..CLASS_LABEL.len() + 16].copy_from_slice(bytes);
        let mut state = SHA256_IV;
        sha2::compress256(
            &mut state,
            core::slice::from_ref(GenericArray::from_slice(&block_bytes[..])),
        );
        let mut midstate = Secret::new(HeapKey::<32>::zeroed());
        let words: &mut [u8; 32] = midstate.expose_secret_mut().borrow_mut();
        for (out, word) in words.chunks_exact_mut(4).zip(state) {
            out.copy_from_slice(&word.to_be_bytes());
        }
        state.zeroize();
        Self {
            cipher: Box::new(Aes128::new(GenericArray::from_slice(bytes))),
            midstate,
            bits: key.bits,
        }
    }

    /// The stream class of the request whose prompt is `prompt`: its inbound tokens, its `In`
    /// frames' tokens in `frame_index` order (RFC-0015 'Frames'). It is 0 when `b = 0`, and
    /// otherwise the top `b` bits of `u64_le(h[0..8])` for
    /// `h = SHA-256("tessera-stream-class/v1" ‖ K ‖ 0^25 ‖ u64_le(n) ‖ u32_le(x_0) ‖ … ‖ u32_le(x_{n−1}))`.
    ///
    /// The hash resumes from the key block's state. The rest of the padded message, which holds
    /// the prompt, is built in a [`SecretBuf`] that clears it on release, and passed to the
    /// compression function by reference, so no hasher's buffer keeps a copy. A prompt held in
    /// memory is far below SHA-256's limit of `2^64` bits.
    #[allow(clippy::disallowed_methods)]
    pub fn class(&self, prompt: &Secret<Vec<u32>>) -> StreamClass {
        let bits = u32::from(self.bits.get());
        if bits == 0 {
            return StreamClass(0);
        }
        let tokens = prompt.expose_secret();
        let rest = 8 + 4 * tokens.len();
        // The key block, then the count, the tokens, the 0x80 byte and the 64-bit length.
        let len = 64 + rest;
        let padded = (rest + 9).div_ceil(64) * 64;
        let mut message: SecretBuf<u8> = SecretBuf::with_capacity(padded);
        message.extend((tokens.len() as u64).to_le_bytes());
        message.extend(tokens.iter().flat_map(|x| x.to_le_bytes()));
        message.push(0x80);
        message.resize(padded - 8, 0);
        message.extend((8 * len as u64).to_be_bytes());
        let words: &[u8; 32] = self.midstate.expose_secret().borrow();
        let mut state: [u32; 8] = core::array::from_fn(|i| {
            u32::from_be_bytes([
                words[4 * i],
                words[4 * i + 1],
                words[4 * i + 2],
                words[4 * i + 3],
            ])
        });
        for block in message.chunks_exact(64) {
            sha2::compress256(
                &mut state,
                core::slice::from_ref(GenericArray::from_slice(block)),
            );
        }
        // h[0..8] is the first two state words, big-endian; read as u64_le.
        let mut head = [0u8; 8];
        head[..4].copy_from_slice(&state[0].to_be_bytes());
        head[4..].copy_from_slice(&state[1].to_be_bytes());
        let class = StreamClass(u64::from_le_bytes(head) >> (64 - bits));
        // The state is `h`, derived from the key and the prompt: cleared where it is held (OBS-6).
        state.zeroize();
        head.zeroize();
        class
    }

    /// `u_t` for output position `t` of a request of class `class`: the low four bytes,
    /// little-endian, of AES-128 of the block `u64_le(class) ‖ u32_le(t) ‖ 0^4`, RFC-0017's block
    /// with the class in the slot of `request_id` (RFC-0061 'The stream').
    pub fn draw(&self, class: StreamClass, t: u32) -> u32 {
        let mut block = [0u8; 16];
        block[..8].copy_from_slice(&class.0.to_le_bytes());
        block[8..12].copy_from_slice(&t.to_le_bytes());
        let mut block = GenericArray::from(block);
        self.cipher.encrypt_block(&mut block);
        u32::from_le_bytes([block[0], block[1], block[2], block[3]])
    }
}

/// Shows nothing of the key schedule (05 §3).
impl fmt::Debug for Sampler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Sampler(..)")
    }
}

/// The largest vocabulary of the v1 protocol limits (RFC-0007 'Verifier validation' step 2).
const MAX_VOCAB: usize = 1 << 18;

/// Why [`inverse_cdf`] has no token to return.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplerError {
    /// The vocabulary is empty or larger than `2^18`, the v1 limit of RFC-0007 'Verifier
    /// validation' step 2; it holds this many weights.
    Vocabulary(usize),
    /// Every weight is zero: `Z = 0`, and no token has mass.
    NoMass,
}

impl fmt::Display for SamplerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vocabulary(len) => {
                write!(f, "a vocabulary of {len} weights is outside 1..=2^18")
            }
            Self::NoMass => f.write_str("the distribution has no mass: every weight is zero"),
        }
    }
}

impl std::error::Error for SamplerError {}

/// The inverse-CDF sample of RFC-0009 Lemma 10: the least `v` with `C(v)·2^32 > u·Z`, for the
/// prefix sums `C(v) = Σ_{w ≤ v} weights[w]` and `Z = C(last)`.
///
/// The sums are exact: at most `2^18` weights below `2^64` keep `Z·2^32` below `2^114`. A token
/// exists when `Z > 0`, since `u < 2^32` makes `u·Z < Z·2^32`. A zero weight is never selected,
/// since its `C(v)` equals its predecessor's.
///
/// # Errors
///
/// [`SamplerError::Vocabulary`] for fewer than one or more than `2^18` weights, and
/// [`SamplerError::NoMass`] when `Z = 0`.
pub fn inverse_cdf(weights: &[u64], u: u32) -> Result<usize, SamplerError> {
    if !(1..=MAX_VOCAB).contains(&weights.len()) {
        return Err(SamplerError::Vocabulary(weights.len()));
    }
    let total: u128 = weights.iter().map(|&w| u128::from(w)).sum();
    let target = u128::from(u) * total;
    let mut prefix = 0u128;
    for (v, &w) in weights.iter().enumerate() {
        prefix += u128::from(w);
        if prefix << 32 > target {
            return Ok(v);
        }
    }
    // Reached exactly when Z = 0: otherwise the last prefix is Z, and Z·2^32 > u·Z.
    Err(SamplerError::NoMass)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FIPS-197 Appendix C.1: AES-128 of `00112233…ff` under key `000102…0f`.
    #[test]
    fn the_cipher_is_fips_197_aes_128() {
        let key: [u8; 16] = core::array::from_fn(|i| i as u8);
        let mut block = GenericArray::from(core::array::from_fn::<u8, 16, _>(|i| (i * 0x11) as u8));
        Aes128::new(GenericArray::from_slice(&key)).encrypt_block(&mut block);
        assert_eq!(
            block.as_slice(),
            [
                0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30, 0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4,
                0xc5, 0x5a
            ]
        );
    }

    #[test]
    fn the_inverse_cdf_brackets_the_draw() {
        // Weights 1, 0, 3: Z = 4, prefix sums 1, 1, 4.
        let weights = [1, 0, 3];
        // u·Z < C(0)·2^32 = 2^32 exactly when u < 2^30.
        assert_eq!(inverse_cdf(&weights, 0), Ok(0));
        assert_eq!(inverse_cdf(&weights, (1 << 30) - 1), Ok(0));
        // At u = 2^30, u·Z = 2^32 = C(0)·2^32 is not below it: the zero weight is skipped.
        assert_eq!(inverse_cdf(&weights, 1 << 30), Ok(2));
        assert_eq!(inverse_cdf(&weights, u32::MAX), Ok(2));
        assert_eq!(inverse_cdf(&[0, 0], 7), Err(SamplerError::NoMass));
        // The largest weights and the largest draw: Z·2^32 stays in u128.
        assert_eq!(inverse_cdf(&[u64::MAX, u64::MAX], u32::MAX), Ok(1));
    }

    /// RFC-0007's v1 limits: `1 ≤ vocab ≤ 2^18`.
    #[test]
    fn the_vocabulary_is_within_the_v1_limits() {
        assert_eq!(inverse_cdf(&[], 7), Err(SamplerError::Vocabulary(0)));
        let mut weights = vec![u64::MAX; MAX_VOCAB];
        assert_eq!(inverse_cdf(&weights, u32::MAX), Ok(MAX_VOCAB - 1));
        weights.push(1);
        assert_eq!(
            inverse_cdf(&weights, 0),
            Err(SamplerError::Vocabulary(MAX_VOCAB + 1))
        );
    }

    #[test]
    fn the_key_never_shows() {
        let base = SamplerBaseKey::new(Secret::new(HeapKey::from_box(Box::new([0x5a; 16]))));
        let bits = ClassBits::new(8).unwrap();
        let key = base.derive(&[7; 32], bits, 0);
        let sampler = Sampler::new(&key);
        let class = sampler.class(&Secret::new(vec![7, 8, 9]));
        assert_eq!(format!("{base:?}"), "SamplerBaseKey(..)");
        assert_eq!(format!("{key:?}"), "SamplerKey(..)");
        assert_eq!(format!("{sampler:?}"), "Sampler(..)");
        assert_eq!(format!("{class:?}"), "StreamClass(..)");
    }

    /// Each row's class against the oracle's `hashlib`. The draws of `tests/sampler.rs` check
    /// the class only through AES; this checks the value itself, so that a class wrong in a way
    /// that AES hides from four columns of digits is caught here.
    #[test]
    fn every_class_is_the_oracles() {
        let rows = include_str!("../../../spec/vectors/scoring/prf.txt")
            .lines()
            .filter(|line| !line.starts_with('#'));
        let mut count = 0;
        for row in rows {
            let columns: Vec<&str> = row.split_whitespace().collect();
            let key: [u8; 16] = core::array::from_fn(|i| {
                u8::from_str_radix(&columns[0][2 * i..2 * i + 2], 16).unwrap()
            });
            let bits = ClassBits::new(columns[1].parse().unwrap()).unwrap();
            let key = SamplerKey::new(Secret::new(HeapKey::from_box(Box::new(key))), bits);
            let prompt = match columns[2] {
                "-" => Vec::new(),
                list => list.split(',').map(|x| x.parse().unwrap()).collect(),
            };
            let class = Sampler::new(&key).class(&Secret::new(prompt));
            assert_eq!(class.0, columns[4].parse::<u64>().unwrap(), "row {count}");
            count += 1;
        }
        assert_eq!(count, ROWS);
    }

    /// The rows of `prf.txt`.
    const ROWS: usize = 369;

    /// Each derived key's bytes against the oracle's `hashlib`.
    #[test]
    #[allow(clippy::disallowed_methods)]
    fn every_derived_key_is_the_oracles() {
        let rows = include_str!("../../../spec/vectors/scoring/derive.txt")
            .lines()
            .filter(|line| !line.starts_with('#'));
        let hex = |s: &str| -> Vec<u8> {
            (0..s.len() / 2)
                .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
                .collect()
        };
        let mut count = 0;
        for row in rows {
            let columns: Vec<&str> = row.split_whitespace().collect();
            let base: [u8; 16] = hex(columns[0]).try_into().unwrap();
            let nonce: [u8; 32] = hex(columns[1]).try_into().unwrap();
            let bits = ClassBits::new(columns[2].parse().unwrap()).unwrap();
            let base = SamplerBaseKey::new(Secret::new(HeapKey::from_box(Box::new(base))));
            let derived = base.derive(&nonce, bits, columns[3].parse().unwrap());
            let bytes: &[u8; 16] = derived.key.expose_secret().borrow();
            assert_eq!(derived.bits(), bits);
            assert_eq!(bytes.as_slice(), hex(columns[4]).as_slice(), "row {count}");
            count += 1;
        }
        assert_eq!(count, DERIVATIONS);
    }

    /// The rows of `derive.txt`.
    const DERIVATIONS: usize = 46;

    /// `b` is at most 64, the width of RFC-0017's `request_id` slot.
    #[test]
    fn class_bits_stop_at_64() {
        assert_eq!(ClassBits::new(64).map(ClassBits::get), Some(64));
        assert_eq!(ClassBits::new(0).map(ClassBits::get), Some(0));
        assert_eq!(ClassBits::new(65), None);
    }
}
