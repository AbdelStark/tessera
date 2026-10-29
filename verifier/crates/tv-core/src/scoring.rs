// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Sampler key derivation, keyed stream classes, FIPS-197 AES-128 draws, reference
//! inverse-CDF selection, public predictor validation, and RFC-0019 step-12 `Û`
//! bound checks (RFC-0017, RFC-0061, RFC-0043).
//!
//! Written from RFC-0017 ('Decoding and the sampler PRF', '`D`'s decoded token',
//! 'In-circuit scoring (mixture)', 'Verifier'), RFC-0061 ('The stream', 'The sampler
//! key and its pins'), RFC-0027, RFC-0043 and FIPS-197 without reference to prover
//! code (SP-6, RFC-0001). Where the specification admits several algorithms this
//! module chooses its own:
//!
//! - [`inverse_cdf`] reduces `C(v) · 2^32 > u · Z` to `C(v) > ⌊(u · Z) / 2^32⌋`
//!   via a split 64-bit limbs floor quotient `(u · hi) · 2^32 + ⌊(u · lo) / 2^32⌋`,
//!   which never overflows `u128` for any slice of `u64` weights.
//! - [`aes128_encrypt_block`] represents the four columns directly as `[u8; 4]` words
//!   and evaluates the FIPS-197 §5.1.1 S-box algebraically via `a^254` in
//!   `GF(2^8) / (x^8 + x^4 + x^3 + x + 1)` followed by the affine rotation map.

use core::fmt;

use tv_codec::Rational;
use tv_hash::sha256::Sha256;
use zeroize::Zeroize;

use crate::protocol::{Decoding, FixedBits, PredictorSpec, RequestRef};

/// Maximum `class_bits` (`b`) permitted by RFC-0061 ('The stream').
pub const MAX_CLASS_BITS: u8 = 64;

/// Exclusive upper bound `2^18` on vocabulary token IDs in prompt streams (RFC-0015).
pub const TOKEN_ID_LIMIT: u32 = 1 << 18;

/// Maximum `eps_log2` (`k` in `ε = 2^-k`) permitted by RFC-0017.
pub const MAX_EPS_LOG2: u8 = 16;

/// Maximum rational temperature `T` permitted by RFC-0017 (`0 < T ≤ 4`).
pub const MAX_TEMPERATURE: u64 = 4;

/// Maximum distinct effective temperatures permitted in a single batch (RFC-0017).
pub const MAX_BATCH_TEMPERATURES: usize = 4;

/// Exact `c_match(k) = ⌈−log2(1 − 2^{−k}) · 2^8⌉` for `k = 0..=16`, with `c_match(0) = 0`
/// (RFC-0017 'In-circuit scoring (mixture)' item 2).
const C_MATCH_TABLE: [u64; 17] = [0, 256, 107, 50, 24, 12, 6, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1];

/// FIPS-197 §5.2 round constants `Rcon[1..=10]` in `GF(2^8)`.
const AES128_RCON: [u8; 10] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36];

/// A 16-byte AES-128 sampler key `K` (RFC-0061 'The stream', 05 §3).
///
/// Key material is cleared on drop and redacted in [`fmt::Debug`].
pub struct SamplerKey {
    bytes: [u8; 16],
}

impl SamplerKey {
    /// Wrap a raw 16-byte sampler key `K`.
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self { bytes }
    }

    /// Borrow the underlying 16-byte key.
    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.bytes
    }
}

impl Drop for SamplerKey {
    fn drop(&mut self) {
        self.bytes.zeroize();
    }
}

impl fmt::Debug for SamplerKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SamplerKey([REDACTED])")
    }
}

/// Errors from sampler key derivation, stream classification, reference inverse-CDF
/// selection, or public predictor/temperature validation (RFC-0017, RFC-0061).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScoringError {
    /// `class_bits` exceeds [`MAX_CLASS_BITS`] (`64`).
    ClassBits {
        /// Supplied `class_bits`.
        bits: u8,
    },
    /// Prompt token at `index` is at least [`TOKEN_ID_LIMIT`] (`2^18`).
    PromptToken {
        /// Zero-based position in the prompt slice.
        index: usize,
        /// Out-of-range token identifier.
        token: u32,
    },
    /// Every weight in the distribution is zero (`Z = 0`), or the sum overflowed `u128`.
    NoMass,
    /// `Decoding::Temperature` lies outside `(0, 4]` (RFC-0017).
    Temperature {
        /// Invalid rational temperature.
        t: Rational,
    },
    /// `PredictorSpec::Mixture` has `eps_log2 > 16` (RFC-0017).
    EpsLog2 {
        /// Invalid `eps_log2`.
        eps_log2: u8,
    },
    /// `2^{logits_exponent} / T` cannot be represented as a canonical [`Rational`].
    LogitsExponent {
        /// Supplied `logits_exponent`.
        logits_exponent: i8,
    },
}

impl fmt::Display for ScoringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::ClassBits { bits } => write!(f, "sampler class_bits {bits} exceeds 64"),
            Self::PromptToken { index, token } => {
                write!(f, "prompt token {token} at index {index} reaches 2^18")
            }
            Self::NoMass => f.write_str("inverse-CDF distribution has zero total weight"),
            Self::Temperature { t } => {
                write!(f, "temperature {}/{} outside (0, 4]", t.num(), t.den())
            }
            Self::EpsLog2 { eps_log2 } => write!(f, "mixture eps_log2 {eps_log2} exceeds 16"),
            Self::LogitsExponent { logits_exponent } => write!(
                f,
                "logits_exponent {logits_exponent} overflows canonical Rational scale"
            ),
        }
    }
}

impl std::error::Error for ScoringError {}

/// Derive a pin's 16-byte sampler key `K` from the prover's base key `K_p`, the
/// verifier's 32-byte nonce `r`, `class_bits` (`b ≤ 64`), and `from_ns`
/// (RFC-0061 'The sampler key and its pins'):
///
/// `K = SHA-256("tessera-sampler-derive/v1" ‖ K_p ‖ 0^23 ‖ r ‖ u8(b) ‖ u64_le(from_ns))[0..16]`.
///
/// # Errors
/// [`ScoringError::ClassBits`] when `class_bits > 64`.
pub fn derive_sampler_key(
    base_key: &[u8; 16],
    nonce: &[u8; 32],
    class_bits: u8,
    from_ns: u64,
) -> Result<SamplerKey, ScoringError> {
    if class_bits > MAX_CLASS_BITS {
        return Err(ScoringError::ClassBits { bits: class_bits });
    }
    let mut hasher = Sha256::new();
    hasher
        .update(b"tessera-sampler-derive/v1")
        .update(base_key)
        .update(&[0u8; 23])
        .update(nonce)
        .update(&[class_bits])
        .update(&from_ns.to_le_bytes());
    let mut digest = hasher.finalize();
    let mut key = [0u8; 16];
    for (dst, &src) in key.iter_mut().zip(&digest) {
        *dst = src;
    }
    digest.zeroize();
    Ok(SamplerKey::from_bytes(key))
}

/// Compute a request's `b`-bit keyed stream class from its prompt tokens `x_0, …, x_{n−1}`
/// (RFC-0017 'Sampler PRF', RFC-0061 'The stream'):
///
/// - Rejects `class_bits > 64` and any prompt token `≥ 2^18` regardless of `class_bits`.
/// - Returns `0` when `class_bits == 0`.
/// - Otherwise hashes `"tessera-stream-class/v1" ‖ K ‖ 0^25 ‖ u64_le(n) ‖ u32_le(x_0) ‖ …`
///   with SHA-256 and returns `u64_le(h[0..8]) >> (64 − b)`.
///
/// # Errors
/// [`ScoringError::ClassBits`] when `class_bits > 64`, or [`ScoringError::PromptToken`]
/// at the first token `≥ 2^18`.
pub fn stream_class(key: &SamplerKey, class_bits: u8, prompt: &[u32]) -> Result<u64, ScoringError> {
    if class_bits > MAX_CLASS_BITS {
        return Err(ScoringError::ClassBits { bits: class_bits });
    }
    for (index, &token) in prompt.iter().enumerate() {
        if token >= TOKEN_ID_LIMIT {
            return Err(ScoringError::PromptToken { index, token });
        }
    }
    if class_bits == 0 {
        return Ok(0);
    }
    let count = u64::try_from(prompt.len()).unwrap_or(u64::MAX);
    let mut hasher = Sha256::new();
    hasher
        .update(b"tessera-stream-class/v1")
        .update(&key.bytes)
        .update(&[0u8; 25])
        .update(&count.to_le_bytes());
    for &token in prompt {
        hasher.update(&token.to_le_bytes());
    }
    let mut digest = hasher.finalize();
    let mut head = [0u8; 8];
    for (dst, &src) in head.iter_mut().zip(&digest) {
        *dst = src;
    }
    digest.zeroize();
    let shift = u32::from(MAX_CLASS_BITS.saturating_sub(class_bits));
    Ok(u64::from_le_bytes(head).checked_shr(shift).unwrap_or(0))
}

/// Encrypt one 16-byte block under `key` with FIPS-197 AES-128 (`Nk = 4, Nb = 4, Nr = 10`).
pub fn aes128_encrypt_block(key: &SamplerKey, block: &[u8; 16]) -> [u8; 16] {
    let [
        k0,
        k1,
        k2,
        k3,
        k4,
        k5,
        k6,
        k7,
        k8,
        k9,
        k10,
        k11,
        k12,
        k13,
        k14,
        k15,
    ] = key.bytes;
    let mut rk = [
        [k0, k1, k2, k3],
        [k4, k5, k6, k7],
        [k8, k9, k10, k11],
        [k12, k13, k14, k15],
    ];
    let [
        b0,
        b1,
        b2,
        b3,
        b4,
        b5,
        b6,
        b7,
        b8,
        b9,
        b10,
        b11,
        b12,
        b13,
        b14,
        b15,
    ] = *block;
    let mut state = add_round_key(
        [
            [b0, b1, b2, b3],
            [b4, b5, b6, b7],
            [b8, b9, b10, b11],
            [b12, b13, b14, b15],
        ],
        rk,
    );

    for (round, &rcon) in AES128_RCON.iter().enumerate() {
        rk = next_round_key(rk, rcon);
        state = shift_rows(sub_bytes(state));
        if round != 9 {
            state = mix_columns(state);
        }
        state = add_round_key(state, rk);
    }

    for word in &mut rk {
        word.zeroize();
    }
    let [
        [o0, o1, o2, o3],
        [o4, o5, o6, o7],
        [o8, o9, o10, o11],
        [o12, o13, o14, o15],
    ] = state;
    state.iter_mut().for_each(Zeroize::zeroize);
    [
        o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15,
    ]
}

/// Draw the 32-bit uniform integer `u_t` at output position `t` for `class` under `key`
/// (RFC-0017 'Sampler PRF', RFC-0061 'The stream'):
///
/// `u_t = u32_le(AES-128_K(u64_le(class) ‖ u32_le(t) ‖ 0^4)[0..4])`.
pub fn sampler_draw(key: &SamplerKey, class: u64, position: u32) -> u32 {
    let [c0, c1, c2, c3, c4, c5, c6, c7] = class.to_le_bytes();
    let [t0, t1, t2, t3] = position.to_le_bytes();
    let mut pt = [c0, c1, c2, c3, c4, c5, c6, c7, t0, t1, t2, t3, 0, 0, 0, 0];
    let mut ct = aes128_encrypt_block(key, &pt);
    let [u0, u1, u2, u3, ..] = ct;
    let draw = u32::from_le_bytes([u0, u1, u2, u3]);
    pt.zeroize();
    ct.zeroize();
    draw
}

/// Select the unique vocabulary index `v` whose cumulative weight crosses `u · Z / 2^32`
/// (RFC-0009 Lemma 10, RFC-0017):
///
/// `C(v − 1) · 2^32 ≤ u · Z < C(v) · 2^32`, where `C(v) = Σ_{i ≤ v} weights[i]`,
/// `C(−1) = 0`, and `Z = Σ_i weights[i]`.
///
/// Zero-weight tokens have `C(v) = C(v − 1)` and can never satisfy both inequalities.
///
/// # Errors
/// [`ScoringError::NoMass`] when `weights` is empty or every weight is `0`.
pub fn inverse_cdf(weights: &[u64], u: u32) -> Result<usize, ScoringError> {
    let mut total = 0u128;
    for &w in weights {
        total = total
            .checked_add(u128::from(w))
            .ok_or(ScoringError::NoMass)?;
    }
    if total == 0 {
        return Err(ScoringError::NoMass);
    }

    // C(v) * 2^32 > u * total  <=>  C(v) > floor((u * total) / 2^32).
    // Split total = hi * 2^64 + lo so u * hi and u * lo never exceed 2^96.
    let u_wide = u128::from(u);
    let lo = total & 0xFFFF_FFFF_FFFF_FFFF;
    let hi = total >> 64;
    let q_lo = u_wide.checked_mul(lo).ok_or(ScoringError::NoMass)? >> 32;
    let q_hi = u_wide
        .checked_mul(hi)
        .and_then(|v| v.checked_shl(32))
        .ok_or(ScoringError::NoMass)?;
    let threshold = q_hi.checked_add(q_lo).ok_or(ScoringError::NoMass)?;

    let mut prefix = 0u128;
    for (index, &w) in weights.iter().enumerate() {
        prefix = prefix
            .checked_add(u128::from(w))
            .ok_or(ScoringError::NoMass)?;
        if prefix > threshold {
            return Ok(index);
        }
    }
    Err(ScoringError::NoMass)
}

/// Validate a [`Decoding`] parameter's public domain (`0 < t ≤ 4` for temperature, RFC-0017).
///
/// # Errors
/// [`ScoringError::Temperature`] when `t ≤ 0` or `t > 4`.
pub fn validate_decoding(decoding: &Decoding) -> Result<(), ScoringError> {
    match *decoding {
        Decoding::Greedy => Ok(()),
        Decoding::Temperature { t } => {
            let Ok(num) = u128::try_from(t.num()) else {
                return Err(ScoringError::Temperature { t });
            };
            let max_num = u128::from(MAX_TEMPERATURE).saturating_mul(u128::from(t.den()));
            if num == 0 || num > max_num {
                return Err(ScoringError::Temperature { t });
            }
            Ok(())
        }
    }
}

/// Validate a [`PredictorSpec`]'s public domain (`0 < t ≤ 4` and `eps_log2 ≤ 16`, RFC-0017).
///
/// # Errors
/// [`ScoringError::Temperature`] or [`ScoringError::EpsLog2`] on an out-of-domain parameter.
pub fn validate_predictor(predictor: &PredictorSpec) -> Result<(), ScoringError> {
    match *predictor {
        PredictorSpec::Exact { ref decoding } => validate_decoding(decoding),
        PredictorSpec::Mixture {
            ref decoding,
            eps_log2,
        } => {
            validate_decoding(decoding)?;
            if eps_log2 > MAX_EPS_LOG2 {
                return Err(ScoringError::EpsLog2 { eps_log2 });
            }
            Ok(())
        }
    }
}

/// Compute the reduced canonical [`Rational`] scale `2^{logits_exponent} / T` folded into a
/// vocabulary exponential table `Exp_T` (RFC-0017 '`D`'s decoded token', RFC-0027).
///
/// # Errors
/// [`ScoringError::Temperature`] when `temperature` is outside `(0, 4]`, or
/// [`ScoringError::LogitsExponent`] when the reduced fraction overflows [`Rational`].
pub fn vocab_exp_scale(
    logits_exponent: i8,
    temperature: Rational,
) -> Result<Rational, ScoringError> {
    validate_decoding(&Decoding::Temperature { t: temperature })?;
    let num_t = u64::try_from(temperature.num())
        .map_err(|_| ScoringError::Temperature { t: temperature })?;
    let den_t = temperature.den();

    // Since gcd(num_t, den_t) == 1, shifting one side by 2^e only shares factors of 2
    // with the opposite side.
    let (raw_num, raw_den) = if logits_exponent >= 0 {
        let exp = u32::from(logits_exponent.unsigned_abs());
        let cancel = exp.min(num_t.trailing_zeros());
        let shift = exp.saturating_sub(cancel);
        let shifted = u128::from(den_t)
            .checked_shl(shift)
            .ok_or(ScoringError::LogitsExponent { logits_exponent })?;
        (shifted, u128::from(num_t >> cancel))
    } else {
        let exp = u32::from(logits_exponent.unsigned_abs());
        let cancel = exp.min(den_t.trailing_zeros());
        let shift = exp.saturating_sub(cancel);
        let shifted = u128::from(num_t)
            .checked_shl(shift)
            .ok_or(ScoringError::LogitsExponent { logits_exponent })?;
        (u128::from(den_t >> cancel), shifted)
    };

    let num_i64 =
        i64::try_from(raw_num).map_err(|_| ScoringError::LogitsExponent { logits_exponent })?;
    let den_u64 =
        u64::try_from(raw_den).map_err(|_| ScoringError::LogitsExponent { logits_exponent })?;
    Rational::new(num_i64, den_u64).ok_or(ScoringError::LogitsExponent { logits_exponent })
}

/// Return the public match cost `c_match(k) = ⌈−log2(1 − 2^{−k}) · 2^8⌉` in `FixedBits`
/// units (`0` for `k = 0`, RFC-0017 'In-circuit scoring (mixture)' item 2).
///
/// # Errors
/// [`ScoringError::EpsLog2`] when `eps_log2 > 16`.
pub fn c_match(eps_log2: u8) -> Result<u64, ScoringError> {
    C_MATCH_TABLE
        .get(usize::from(eps_log2))
        .copied()
        .ok_or(ScoringError::EpsLog2 { eps_log2 })
}

/// Failure of RFC-0019 step 12's `Û` acceptance check (RFC-0017, RFC-0043).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UhatError {
    /// `claimed.len() != requests.len()` or `Σ_r Û_r` overflows `u64` (`REJ-MALFORMED`).
    Malformed,
    /// A per-request `Û_r` or the batch sum `Σ_r Û_r` exceeded the applicable bound (`REJ-UHAT`).
    Exceeded {
        /// Offending request's `request_id`, or `None` when the batch total exceeded the bound.
        request_id: Option<u64>,
        /// Claimed fixed-point unexplained bits.
        uhat: FixedBits,
        /// Applicable fixed-point upper bound (`0` for `Exact`, `accept_bound` for `Mixture`).
        bound: FixedBits,
    },
}

/// Check per-request and batch-total `Û` claims in RFC-0019 step 12 without heap allocation,
/// returning `batch_uhat = Σ_r Û_r` on success (RFC-0017 'Verifier', RFC-0043).
///
/// - `PredictorSpec::Exact` enforces `bound = FixedBits(0)` regardless of `accept_bound`.
/// - `PredictorSpec::Mixture` enforces `bound = accept_bound` on every request in batch
///   order, then on `Σ_r Û_r`.
///
/// # Errors
/// [`UhatError::Malformed`] when `requests` is outside `1..=256`, `claimed.len() != requests.len()`,
/// or `Σ_r Û_r` overflows `u64`; [`UhatError::Exceeded`] on the first request exceeding `bound`,
/// or with `request_id: None` when `Σ_r Û_r > bound`.
pub fn check_uhat_bound(
    predictor: PredictorSpec,
    requests: &[RequestRef],
    claimed: &[FixedBits],
    accept_bound: FixedBits,
) -> Result<FixedBits, UhatError> {
    if !(1..=256).contains(&requests.len()) || claimed.len() != requests.len() {
        return Err(UhatError::Malformed);
    }
    let bound = match predictor {
        PredictorSpec::Exact { .. } => FixedBits(0),
        PredictorSpec::Mixture { .. } => accept_bound,
    };
    for (req, &uhat) in requests.iter().zip(claimed) {
        if uhat.0 > bound.0 {
            return Err(UhatError::Exceeded {
                request_id: Some(req.request_id),
                uhat,
                bound,
            });
        }
    }
    let mut total = 0u64;
    for &uhat in claimed {
        total = total.checked_add(uhat.0).ok_or(UhatError::Malformed)?;
    }
    if total > bound.0 {
        return Err(UhatError::Exceeded {
            request_id: None,
            uhat: FixedBits(total),
            bound,
        });
    }
    Ok(FixedBits(total))
}

/// Sums the per-token fixed-point bounds over request `ordinal`'s active window
/// `ordinal · slot_len + (prompt_len − 1) .. ordinal · slot_len + prompt_len + output_len − 1`
/// (RFC-0071 'The greedy certificate and scoring'). Returns `Some(FixedBits(0))` when
/// `output_len == 0`, or `None` on invalid geometry or `u64` overflow.
pub fn window_uhat_sum(
    row_bounds: &[u64],
    slot_len: u32,
    ordinal: u32,
    prompt_len: u32,
    output_len: u32,
) -> Option<FixedBits> {
    if prompt_len == 0 {
        return None;
    }
    let n_r = prompt_len.checked_add(output_len)?;
    if n_r > slot_len {
        return None;
    }
    let base = u64::from(ordinal).checked_mul(u64::from(slot_len))?;
    let slot_end = usize::try_from(base.checked_add(u64::from(n_r))?).ok()?;
    if slot_end > row_bounds.len() {
        return None;
    }
    if output_len == 0 {
        return Some(FixedBits(0));
    }
    let win_start =
        usize::try_from(base.checked_add(u64::from(prompt_len.checked_sub(1)?))?).ok()?;
    let win_end = win_start.checked_add(usize::try_from(output_len).ok()?)?;
    let mut sum = 0u64;
    for &b in row_bounds.get(win_start..win_end)? {
        sum = sum.checked_add(b)?;
    }
    Some(FixedBits(sum))
}

// --- FIPS-197 AES-128 internals ---

/// `xtime(a) = a • {02}` modulo `x^8 + x^4 + x^3 + x + 1` (FIPS-197 §4.2.1).
const fn gf_xtime(a: u8) -> u8 {
    let shifted = (a & 0x7F) << 1;
    if a & 0x80 == 0 {
        shifted
    } else {
        shifted ^ 0x1B
    }
}

/// `a • b` in `GF(2^8)` (FIPS-197 §4.2.1).
const fn gf_mul(a: u8, b: u8) -> u8 {
    let (mut power, mut rest, mut acc) = (a, b, 0u8);
    while rest != 0 {
        if rest & 1 != 0 {
            acc ^= power;
        }
        power = gf_xtime(power);
        rest >>= 1;
    }
    acc
}

/// FIPS-197 §5.1.1 S-box: `a^254` in `GF(2^8)` followed by the affine map over `GF(2)`.
const fn aes_sbox(a: u8) -> u8 {
    let (mut inv, mut sq, mut exp) = (1u8, a, 254u8);
    while exp != 0 {
        if exp & 1 != 0 {
            inv = gf_mul(inv, sq);
        }
        sq = gf_mul(sq, sq);
        exp >>= 1;
    }
    inv ^ inv.rotate_left(1) ^ inv.rotate_left(2) ^ inv.rotate_left(3) ^ inv.rotate_left(4) ^ 0x63
}

const fn xor4(a: [u8; 4], b: [u8; 4]) -> [u8; 4] {
    let [a0, a1, a2, a3] = a;
    let [b0, b1, b2, b3] = b;
    [a0 ^ b0, a1 ^ b1, a2 ^ b2, a3 ^ b3]
}

const fn sub_word(w: [u8; 4]) -> [u8; 4] {
    let [b0, b1, b2, b3] = w;
    [aes_sbox(b0), aes_sbox(b1), aes_sbox(b2), aes_sbox(b3)]
}

const fn next_round_key(prev: [[u8; 4]; 4], rcon: u8) -> [[u8; 4]; 4] {
    let [w0, w1, w2, w3] = prev;
    let [r0, r1, r2, r3] = w3;
    let [s0, s1, s2, s3] = sub_word([r1, r2, r3, r0]);
    let n0 = xor4(w0, [s0 ^ rcon, s1, s2, s3]);
    let n1 = xor4(w1, n0);
    let n2 = xor4(w2, n1);
    let n3 = xor4(w3, n2);
    [n0, n1, n2, n3]
}

const fn add_round_key(state: [[u8; 4]; 4], rk: [[u8; 4]; 4]) -> [[u8; 4]; 4] {
    let [s0, s1, s2, s3] = state;
    let [k0, k1, k2, k3] = rk;
    [xor4(s0, k0), xor4(s1, k1), xor4(s2, k2), xor4(s3, k3)]
}

const fn sub_bytes(state: [[u8; 4]; 4]) -> [[u8; 4]; 4] {
    let [c0, c1, c2, c3] = state;
    [sub_word(c0), sub_word(c1), sub_word(c2), sub_word(c3)]
}

const fn shift_rows(state: [[u8; 4]; 4]) -> [[u8; 4]; 4] {
    let [
        [s00, s10, s20, s30],
        [s01, s11, s21, s31],
        [s02, s12, s22, s32],
        [s03, s13, s23, s33],
    ] = state;
    [
        [s00, s11, s22, s33],
        [s01, s12, s23, s30],
        [s02, s13, s20, s31],
        [s03, s10, s21, s32],
    ]
}

const fn mix_column(col: [u8; 4]) -> [u8; 4] {
    let [s0, s1, s2, s3] = col;
    let x0 = gf_xtime(s0);
    let x1 = gf_xtime(s1);
    let x2 = gf_xtime(s2);
    let x3 = gf_xtime(s3);
    [
        x0 ^ x1 ^ s1 ^ s2 ^ s3,
        s0 ^ x1 ^ x2 ^ s2 ^ s3,
        s0 ^ s1 ^ x2 ^ x3 ^ s3,
        x0 ^ s0 ^ s1 ^ s2 ^ x3,
    ]
}

const fn mix_columns(state: [[u8; 4]; 4]) -> [[u8; 4]; 4] {
    let [c0, c1, c2, c3] = state;
    [
        mix_column(c0),
        mix_column(c1),
        mix_column(c2),
        mix_column(c3),
    ]
}
