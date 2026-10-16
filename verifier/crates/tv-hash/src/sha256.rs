// SPDX-License-Identifier: Apache-2.0 OR MIT

//! SHA-256, written from FIPS 180-4 ("Secure Hash Standard", 2015): the functions of §4.1.2, the
//! constants of §4.2.2, the padding of §5.1.1, the initial hash value of §5.3.3 and the
//! computation of §6.2. It hashes the recorder's digest log (RFC-0015), whose Merkle tree is in
//! [`rfc6962`](crate::rfc6962).
//!
//! ```
//! use tv_hash::sha256::{Sha256, sha256};
//!
//! let mut hasher = Sha256::new();
//! hasher.update(b"a").update(b"bc");
//! assert_eq!(hasher.finalize(), sha256(b"abc"));
//! ```

use crate::Digest;

/// The block size in bytes.
const BLOCK_LEN: usize = 64;

/// The initial hash value `H(0)`: the first 32 bits of the fractional parts of the square roots
/// of the first eight primes (§5.3.3).
const H0: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

/// The constants `K_0 … K_63`: the first 32 bits of the fractional parts of the cube roots of the
/// first sixty-four primes (§4.2.2), in four groups of sixteen rounds.
const K: [[u32; 16]; 4] = [
    [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
    ],
    [
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
    ],
    [
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
    ],
    [
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ],
];

// The functions of §4.1.2.

#[inline(always)]
const fn ch(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ (!x & z)
}

#[inline(always)]
const fn maj(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ (x & z) ^ (y & z)
}

/// `Σ0`.
#[inline(always)]
const fn big_sigma0(x: u32) -> u32 {
    x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
}

/// `Σ1`.
#[inline(always)]
const fn big_sigma1(x: u32) -> u32 {
    x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
}

/// `σ0`.
#[inline(always)]
const fn small_sigma0(x: u32) -> u32 {
    x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
}

/// `σ1`.
#[inline(always)]
const fn small_sigma1(x: u32) -> u32 {
    x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
}

/// The sixteen big-endian words of a block, `M_0 … M_15` (§5.2.1).
#[inline(always)]
const fn block_words(b: &[u8; BLOCK_LEN]) -> [u32; 16] {
    macro_rules! word {
        ($i:literal) => {
            u32::from_be_bytes([b[4 * $i], b[4 * $i + 1], b[4 * $i + 2], b[4 * $i + 3]])
        };
    }
    [
        word!(0),
        word!(1),
        word!(2),
        word!(3),
        word!(4),
        word!(5),
        word!(6),
        word!(7),
        word!(8),
        word!(9),
        word!(10),
        word!(11),
        word!(12),
        word!(13),
        word!(14),
        word!(15),
    ]
}

/// One block of the hash computation (§6.2.2). The message schedule is kept as its last sixteen
/// words: before round `t ≥ 16` uses `W_t`, it replaces `W_(t−16)` in the same slot, computed
/// from the slots of `W_(t−2)`, `W_(t−7)`, `W_(t−15)` and `W_(t−16)`.
fn compress(state: &mut [u32; 8], block: &[u8; BLOCK_LEN]) {
    let mut w = block_words(block);
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for (group, k) in K.iter().enumerate() {
        macro_rules! round {
            ($i:literal) => {
                if group > 0 {
                    w[$i] = small_sigma1(w[($i + 14) % 16])
                        .wrapping_add(w[($i + 9) % 16])
                        .wrapping_add(small_sigma0(w[($i + 1) % 16]))
                        .wrapping_add(w[$i]);
                }
                let t1 = h
                    .wrapping_add(big_sigma1(e))
                    .wrapping_add(ch(e, f, g))
                    .wrapping_add(k[$i])
                    .wrapping_add(w[$i]);
                let t2 = big_sigma0(a).wrapping_add(maj(a, b, c));
                h = g;
                g = f;
                f = e;
                e = d.wrapping_add(t1);
                d = c;
                c = b;
                b = a;
                a = t1.wrapping_add(t2);
            };
        }
        round!(0);
        round!(1);
        round!(2);
        round!(3);
        round!(4);
        round!(5);
        round!(6);
        round!(7);
        round!(8);
        round!(9);
        round!(10);
        round!(11);
        round!(12);
        round!(13);
        round!(14);
        round!(15);
    }
    for (word, add) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *word = word.wrapping_add(add);
    }
}

/// SHA-256 over an input given in pieces.
///
/// FIPS 180-4 defines SHA-256 for messages shorter than `2^64` bits, `2^61` bytes. The length is
/// counted modulo `2^64` bits, so a longer input, which no memory holds and no stream reaches in
/// practice, would be hashed with its length wrapped.
#[derive(Clone)]
pub struct Sha256 {
    state: [u32; 8],
    /// Input bytes not yet compressed: always fewer than a block.
    block: [u8; BLOCK_LEN],
    block_len: usize,
    /// Input bytes so far, modulo `2^64`.
    total: u64,
}

impl Sha256 {
    /// A hasher that has seen no input.
    pub const fn new() -> Self {
        Self {
            state: H0,
            block: [0; BLOCK_LEN],
            block_len: 0,
            total: 0,
        }
    }

    /// Appends `input` to the bytes hashed so far.
    pub fn update(&mut self, mut input: &[u8]) -> &mut Self {
        self.total = self
            .total
            .wrapping_add(u64::try_from(input.len()).unwrap_or(u64::MAX));
        while !input.is_empty() {
            if self.block_len == 0 {
                while let Some((block, tail)) = input.split_first_chunk::<BLOCK_LEN>() {
                    compress(&mut self.state, block);
                    input = tail;
                }
            }
            let mut taken: usize = 0;
            for (slot, byte) in self.block.iter_mut().skip(self.block_len).zip(input) {
                *slot = *byte;
                taken = taken.wrapping_add(1);
            }
            self.block_len = self.block_len.saturating_add(taken);
            input = input.get(taken..).unwrap_or_default();
            if self.block_len == BLOCK_LEN {
                compress(&mut self.state, &self.block);
                self.block_len = 0;
            }
        }
        self
    }

    /// The digest of everything appended so far (§5.1.1 padding, then §6.2.2). The hasher is
    /// unchanged and may take more input.
    pub fn finalize(&self) -> Digest {
        let mut state = self.state;
        let mut block = [0u8; BLOCK_LEN];
        for (slot, byte) in block.iter_mut().zip(self.block.iter().take(self.block_len)) {
            *slot = *byte;
        }
        // The bit `1` right after the message, then zeros, then the length in bits as a
        // big-endian 64-bit integer, which needs the last 8 bytes of a block.
        if let Some(slot) = block.get_mut(self.block_len) {
            *slot = 0x80;
        }
        if self.block_len >= BLOCK_LEN.saturating_sub(8) {
            compress(&mut state, &block);
            block = [0; BLOCK_LEN];
        }
        let bits = self.total.wrapping_mul(8).to_be_bytes();
        for (slot, byte) in block.iter_mut().skip(BLOCK_LEN.saturating_sub(8)).zip(bits) {
            *slot = byte;
        }
        compress(&mut state, &block);
        let mut digest = [0u8; 32];
        for (out, word) in digest.chunks_exact_mut(4).zip(state) {
            for (slot, byte) in out.iter_mut().zip(word.to_be_bytes()) {
                *slot = byte;
            }
        }
        digest
    }
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

/// Shows nothing of the input.
impl core::fmt::Debug for Sha256 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Sha256").finish_non_exhaustive()
    }
}

/// The SHA-256 digest of `input`.
pub fn sha256(input: &[u8]) -> Digest {
    let mut hasher = Sha256::new();
    hasher.update(input);
    hasher.finalize()
}
