// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Arithmetic in GF(2^8) modulo `x^8 + x^4 + x^3 + x + 1` and the AES S-box (FIPS-197 §4.2,
//! §5.1.1). Every function runs a fixed sequence of operations with masks in place of branches:
//! the witness engine applies them to secret bytes (INV-ARCH-2).

/// `a • b` in GF(2^8): shift-and-add over the eight bits of `b`, reducing by `0x1b` whenever the
/// top bit of the running multiple of `a` falls out (FIPS-197 §4.2.1, `xtime`).
pub(crate) const fn mul(a: u8, b: u8) -> u8 {
    let mut multiple = a;
    let mut product = 0u8;
    let mut bit = 0;
    while bit < 8 {
        let take = 0u8.wrapping_sub((b >> bit) & 1);
        product ^= multiple & take;
        let reduce = 0u8.wrapping_sub(multiple >> 7);
        multiple = (multiple << 1) ^ (0x1b & reduce);
        bit += 1;
    }
    product
}

/// `a^254`: the multiplicative inverse of a non-zero `a`, and 0 for 0, as FIPS-197 §5.1.1 maps
/// it. `254 = 2 + 4 + … + 128`, so it is the product of `a`'s squarings `a^2` to `a^128`.
pub(crate) const fn inverse(a: u8) -> u8 {
    let mut square = mul(a, a);
    let mut result = square;
    let mut step = 2;
    while step < 8 {
        square = mul(square, square);
        result = mul(result, square);
        step += 1;
    }
    result
}

/// The AES S-box: the inverse, then the affine map
/// `b ⊕ (b ⋘ 1) ⊕ (b ⋘ 2) ⊕ (b ⋘ 3) ⊕ (b ⋘ 4) ⊕ 0x63` (FIPS-197 §5.1.1).
pub(crate) const fn sbox(x: u8) -> u8 {
    let b = inverse(x);
    b ^ b.rotate_left(1) ^ b.rotate_left(2) ^ b.rotate_left(3) ^ b.rotate_left(4) ^ 0x63
}

// FIPS-197's own examples: {57} • {83} = {c1} (§4.2), {57} • {13} = {fe} (§4.2.1), and the
// S-box entries for {00} and {53} (§5.1.1 and its Figure 7).
const _: () = assert!(mul(0x57, 0x83) == 0xc1);
const _: () = assert!(mul(0x57, 0x13) == 0xfe);
const _: () = assert!(sbox(0x00) == 0x63);
const _: () = assert!(sbox(0x53) == 0xed);
