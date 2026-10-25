// SPDX-License-Identifier: Apache-2.0 OR MIT

//! SHA-512, written from FIPS 180-4 ("Secure Hash Standard", 2015): the functions of §4.1.3, the
//! constants of §4.2.3, the padding of §5.1.2, the initial hash value of §5.3.5 and the
//! computation of §6.4. It is the hash of Ed25519 (RFC 8032), which
//! [`ed25519`](crate::ed25519) verifies for recorder manifests (RFC-0060 'Signatures').
//!
//! ```
//! use tv_hash::sha512::{Sha512, sha512};
//!
//! let mut hasher = Sha512::new();
//! hasher.update(b"a").update(b"bc");
//! assert_eq!(hasher.finalize(), sha512(b"abc"));
//! ```

/// The block size in bytes.
const BLOCK_LEN: usize = 128;

/// The length field at the end of the last block: the message length in bits, a big-endian
/// 128-bit integer (§5.1.2).
const LENGTH_LEN: usize = 16;

/// A SHA-512 digest.
pub type Digest512 = [u8; 64];

/// The initial hash value `H(0)`: the first 64 bits of the fractional parts of the square roots
/// of the first eight primes (§5.3.5).
const H0: [u64; 8] = [
    0x6a09_e667_f3bc_c908,
    0xbb67_ae85_84ca_a73b,
    0x3c6e_f372_fe94_f82b,
    0xa54f_f53a_5f1d_36f1,
    0x510e_527f_ade6_82d1,
    0x9b05_688c_2b3e_6c1f,
    0x1f83_d9ab_fb41_bd6b,
    0x5be0_cd19_137e_2179,
];

/// The constants `K_0 … K_79`: the first 64 bits of the fractional parts of the cube roots of the
/// first eighty primes (§4.2.3), in five groups of sixteen rounds.
const K: [[u64; 16]; 5] = [
    [
        0x428a_2f98_d728_ae22,
        0x7137_4491_23ef_65cd,
        0xb5c0_fbcf_ec4d_3b2f,
        0xe9b5_dba5_8189_dbbc,
        0x3956_c25b_f348_b538,
        0x59f1_11f1_b605_d019,
        0x923f_82a4_af19_4f9b,
        0xab1c_5ed5_da6d_8118,
        0xd807_aa98_a303_0242,
        0x1283_5b01_4570_6fbe,
        0x2431_85be_4ee4_b28c,
        0x550c_7dc3_d5ff_b4e2,
        0x72be_5d74_f27b_896f,
        0x80de_b1fe_3b16_96b1,
        0x9bdc_06a7_25c7_1235,
        0xc19b_f174_cf69_2694,
    ],
    [
        0xe49b_69c1_9ef1_4ad2,
        0xefbe_4786_384f_25e3,
        0x0fc1_9dc6_8b8c_d5b5,
        0x240c_a1cc_77ac_9c65,
        0x2de9_2c6f_592b_0275,
        0x4a74_84aa_6ea6_e483,
        0x5cb0_a9dc_bd41_fbd4,
        0x76f9_88da_8311_53b5,
        0x983e_5152_ee66_dfab,
        0xa831_c66d_2db4_3210,
        0xb003_27c8_98fb_213f,
        0xbf59_7fc7_beef_0ee4,
        0xc6e0_0bf3_3da8_8fc2,
        0xd5a7_9147_930a_a725,
        0x06ca_6351_e003_826f,
        0x1429_2967_0a0e_6e70,
    ],
    [
        0x27b7_0a85_46d2_2ffc,
        0x2e1b_2138_5c26_c926,
        0x4d2c_6dfc_5ac4_2aed,
        0x5338_0d13_9d95_b3df,
        0x650a_7354_8baf_63de,
        0x766a_0abb_3c77_b2a8,
        0x81c2_c92e_47ed_aee6,
        0x9272_2c85_1482_353b,
        0xa2bf_e8a1_4cf1_0364,
        0xa81a_664b_bc42_3001,
        0xc24b_8b70_d0f8_9791,
        0xc76c_51a3_0654_be30,
        0xd192_e819_d6ef_5218,
        0xd699_0624_5565_a910,
        0xf40e_3585_5771_202a,
        0x106a_a070_32bb_d1b8,
    ],
    [
        0x19a4_c116_b8d2_d0c8,
        0x1e37_6c08_5141_ab53,
        0x2748_774c_df8e_eb99,
        0x34b0_bcb5_e19b_48a8,
        0x391c_0cb3_c5c9_5a63,
        0x4ed8_aa4a_e341_8acb,
        0x5b9c_ca4f_7763_e373,
        0x682e_6ff3_d6b2_b8a3,
        0x748f_82ee_5def_b2fc,
        0x78a5_636f_4317_2f60,
        0x84c8_7814_a1f0_ab72,
        0x8cc7_0208_1a64_39ec,
        0x90be_fffa_2363_1e28,
        0xa450_6ceb_de82_bde9,
        0xbef9_a3f7_b2c6_7915,
        0xc671_78f2_e372_532b,
    ],
    [
        0xca27_3ece_ea26_619c,
        0xd186_b8c7_21c0_c207,
        0xeada_7dd6_cde0_eb1e,
        0xf57d_4f7f_ee6e_d178,
        0x06f0_67aa_7217_6fba,
        0x0a63_7dc5_a2c8_98a6,
        0x113f_9804_bef9_0dae,
        0x1b71_0b35_131c_471b,
        0x28db_77f5_2304_7d84,
        0x32ca_ab7b_40c7_2493,
        0x3c9e_be0a_15c9_bebc,
        0x431d_67c4_9c10_0d4c,
        0x4cc5_d4be_cb3e_42b6,
        0x597f_299c_fc65_7e2a,
        0x5fcb_6fab_3ad6_faec,
        0x6c44_198c_4a47_5817,
    ],
];

// The functions of §4.1.3.

#[inline(always)]
const fn ch(x: u64, y: u64, z: u64) -> u64 {
    (x & y) ^ (!x & z)
}

#[inline(always)]
const fn maj(x: u64, y: u64, z: u64) -> u64 {
    (x & y) ^ (x & z) ^ (y & z)
}

/// `Σ0`.
#[inline(always)]
const fn big_sigma0(x: u64) -> u64 {
    x.rotate_right(28) ^ x.rotate_right(34) ^ x.rotate_right(39)
}

/// `Σ1`.
#[inline(always)]
const fn big_sigma1(x: u64) -> u64 {
    x.rotate_right(14) ^ x.rotate_right(18) ^ x.rotate_right(41)
}

/// `σ0`.
#[inline(always)]
const fn small_sigma0(x: u64) -> u64 {
    x.rotate_right(1) ^ x.rotate_right(8) ^ (x >> 7)
}

/// `σ1`.
#[inline(always)]
const fn small_sigma1(x: u64) -> u64 {
    x.rotate_right(19) ^ x.rotate_right(61) ^ (x >> 6)
}

/// The sixteen big-endian words of a block, `M_0 … M_15` (§5.2.2).
fn block_words(block: &[u8; BLOCK_LEN]) -> [u64; 16] {
    let mut words = [0u64; 16];
    for (word, bytes) in words.iter_mut().zip(block.chunks_exact(8)) {
        let mut be = [0u8; 8];
        be.copy_from_slice(bytes);
        *word = u64::from_be_bytes(be);
    }
    words
}

/// One block of the hash computation (§6.4.2). The message schedule is kept as its last sixteen
/// words: before round `t ≥ 16` uses `W_t`, it replaces `W_(t−16)` in the same slot, computed
/// from the slots of `W_(t−2)`, `W_(t−7)`, `W_(t−15)` and `W_(t−16)`.
fn compress(state: &mut [u64; 8], block: &[u8; BLOCK_LEN]) {
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

/// SHA-512 over an input given in pieces.
///
/// FIPS 180-4 defines SHA-512 for messages shorter than `2^128` bits. The length is counted in
/// bytes modulo `2^64`, so an input of `2^64` bytes or more, which no memory holds and no stream
/// reaches in practice, would be hashed with its length wrapped.
#[derive(Clone)]
pub struct Sha512 {
    state: [u64; 8],
    /// Input bytes not yet compressed: always fewer than a block.
    block: [u8; BLOCK_LEN],
    block_len: usize,
    /// Input bytes so far, modulo `2^64`.
    total: u64,
}

impl Sha512 {
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

    /// The digest of everything appended so far (§5.1.2 padding, then §6.4.2). The hasher is
    /// unchanged and may take more input.
    pub fn finalize(&self) -> Digest512 {
        let mut state = self.state;
        let mut block = [0u8; BLOCK_LEN];
        for (slot, byte) in block.iter_mut().zip(self.block.iter().take(self.block_len)) {
            *slot = *byte;
        }
        // The bit `1` right after the message, then zeros, then the length in bits as a
        // big-endian 128-bit integer, which needs the last 16 bytes of a block.
        if let Some(slot) = block.get_mut(self.block_len) {
            *slot = 0x80;
        }
        if self.block_len >= BLOCK_LEN.saturating_sub(LENGTH_LEN) {
            compress(&mut state, &block);
            block = [0; BLOCK_LEN];
        }
        // Eight times a 64-bit count fits in 128 bits exactly.
        let bits = (u128::from(self.total) << 3).to_be_bytes();
        for (slot, byte) in block
            .iter_mut()
            .skip(BLOCK_LEN.saturating_sub(LENGTH_LEN))
            .zip(bits)
        {
            *slot = byte;
        }
        compress(&mut state, &block);
        let mut digest = [0u8; 64];
        for (out, word) in digest.chunks_exact_mut(8).zip(state) {
            out.copy_from_slice(&word.to_be_bytes());
        }
        digest
    }
}

impl Default for Sha512 {
    fn default() -> Self {
        Self::new()
    }
}

/// Shows nothing of the input.
impl core::fmt::Debug for Sha512 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Sha512").finish_non_exhaustive()
    }
}

/// The SHA-512 digest of `input`.
pub fn sha512(input: &[u8]) -> Digest512 {
    let mut hasher = Sha512::new();
    hasher.update(input);
    hasher.finalize()
}
