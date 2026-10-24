// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The reference sampler of RFC-0017 'Sampler PRF', which serving engines link or port so that
//! they draw exactly the stream the circuit proves.
//!
//! - **The stream.** Output position `t` of request `r` draws
//!   `u_t = u32_le(AES-128_{K_samp}(u64_le(r) ‖ u32_le(t) ‖ 0x00000000)[0..4])`
//!   ([`Sampler::draw`]).
//! - **The selection.** Given the vocabulary's non-negative integer weights `e_v` and
//!   `Z = Σ_v e_v`, the sampled token is `min{v : C(v)·2^32 > u·Z}` with `C(v) = Σ_{w ≤ v} e_w`
//!   ([`inverse_cdf`], RFC-0009 'InverseCdf', Lemma 10).
//!
//! `K_samp`, the stream and the sampled tokens are secret-class (05 §3). The key lives in a
//! [`Secret`] and the cipher's key schedule is cleared on drop (OBS-2). A draw and a token are
//! single values a caller must keep out of logs, errors and reports.

use aes::Aes128;
use aes::cipher::{BlockEncrypt, KeyInit, generic_array::GenericArray};
use core::borrow::Borrow;
use core::fmt;
use tessera_codec::{HeapKey, Secret};

/// A session's sampler key `K_samp`, committed in its session-open frame (RFC-0015 'Frames').
/// It lives as long as its session, so its bytes are on the heap ([`HeapKey`]): moving the key
/// moves a pointer and leaves no copy behind (OBS-2).
pub struct SamplerKey(Secret<HeapKey>);

impl SamplerKey {
    /// The key `key`.
    pub fn new(key: Secret<HeapKey>) -> Self {
        Self(key)
    }

    /// The key, still in its [`Secret`]: for the session-open frame's sampler commitment
    /// (RFC-0059).
    pub fn secret(&self) -> &Secret<HeapKey> {
        &self.0
    }
}

/// Shows nothing of the key (05 §3).
impl fmt::Debug for SamplerKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SamplerKey(..)")
    }
}

/// RFC-0017's PRF under one sampler key: AES-128, its key schedule on the heap, so moving the
/// sampler moves a pointer, and cleared on drop. AES-128's first round key is the key itself.
pub struct Sampler {
    cipher: Box<Aes128>,
}

impl Sampler {
    /// The sampler of `key`.
    #[allow(clippy::disallowed_methods)]
    pub fn new(key: &SamplerKey) -> Self {
        let bytes: &[u8; 16] = key.0.expose_secret().borrow();
        Self {
            cipher: Box::new(Aes128::new(GenericArray::from_slice(bytes))),
        }
    }

    /// `u_t` for output position `t` of request `request_id`: the low four bytes, little-endian, of
    /// AES-128 of the block `u64_le(request_id) ‖ u32_le(t) ‖ 0x00000000`.
    pub fn draw(&self, request_id: u64, t: u32) -> u32 {
        let mut block = [0u8; 16];
        block[..8].copy_from_slice(&request_id.to_le_bytes());
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
        let key = SamplerKey::new(Secret::new(HeapKey::from_box(Box::new([0x5a; 16]))));
        let sampler = Sampler::new(&key);
        assert_eq!(format!("{key:?}"), "SamplerKey(..)");
        assert_eq!(format!("{sampler:?}"), "Sampler(..)");
    }
}
