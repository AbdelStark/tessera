// SPDX-License-Identifier: Apache-2.0 OR MIT

//! GF(2^8) modulo `x^8 + x^4 + x^3 + x + 1` (FIPS-197 §4.2) and the AES S-box (FIPS-197
//! §5.1.1), computed from their definitions rather than copied: the S-box is the multiplicative
//! inverse followed by the affine transformation.

/// The reduction of `x^8`: `x^4 + x^3 + x + 1`, the modulus without its leading term.
const REDUCTION: u8 = 0x1B;

/// The affine transformation's constant `c = {63}` (FIPS-197 §5.1.1).
const AFFINE_CONSTANT: u8 = 0x63;

/// `a • {02}`: `xtime` of FIPS-197 §4.2.1, a shift that reduces when `x^7` overflows.
fn xtime(a: u8) -> u8 {
    let shifted = (a & 0x7F) << 1;
    if a & 0x80 == 0 {
        shifted
    } else {
        shifted ^ REDUCTION
    }
}

/// `a • b`: the sum of `a • x^i` over the bits `i` of `b` (FIPS-197 §4.2.1).
pub(crate) fn multiply(a: u8, b: u8) -> u8 {
    let (mut power, mut rest, mut product) = (a, b, 0);
    while rest != 0 {
        if rest & 1 == 1 {
            product ^= power;
        }
        power = xtime(power);
        rest >>= 1;
    }
    product
}

/// `a^{-1}`, and `{00}` for `{00}`: `a^254`, since `a^255 = 1` for every `a ≠ 0`, by
/// square-and-multiply.
fn inverse(a: u8) -> u8 {
    let (mut result, mut square, mut exponent) = (1, a, 254u8);
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = multiply(result, square);
        }
        square = multiply(square, square);
        exponent >>= 1;
    }
    result
}

/// `S(a)`: the inverse `b = a^{-1}`, then the affine transformation
/// `b'_i = b_i ⊕ b_{i+4} ⊕ b_{i+5} ⊕ b_{i+6} ⊕ b_{i+7} ⊕ c_i`, indices mod 8. Bit `i + j` of `b`
/// lands on bit `i` when `b` is rotated left by `8 − j`.
pub(crate) fn sbox(a: u8) -> u8 {
    let b = inverse(a);
    b ^ b.rotate_left(4) ^ b.rotate_left(3) ^ b.rotate_left(2) ^ b.rotate_left(1) ^ AFFINE_CONSTANT
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn products_of_fips_197_section_4_2() {
        // {57} • {83} = {c1} (§4.2) and {57} • {13} = {fe} (§4.2.1), with its powers of x.
        assert_eq!(multiply(0x57, 0x83), 0xC1);
        assert_eq!(multiply(0x57, 0x13), 0xFE);
        for (power, expected) in [(0x02, 0xAE), (0x04, 0x47), (0x08, 0x8E), (0x10, 0x07)] {
            assert_eq!(multiply(0x57, power), expected);
        }
    }

    #[test]
    fn every_nonzero_element_has_its_inverse() {
        assert_eq!(inverse(0), 0);
        for a in 1..=255u8 {
            assert_eq!(multiply(a, inverse(a)), 1, "{a:#04x}");
        }
    }

    #[test]
    fn sbox_entries_of_fips_197() {
        // §5.1.1: S({53}) = {ed}; Figure 7's corners and a few more entries.
        for (input, output) in [
            (0x00, 0x63),
            (0x01, 0x7C),
            (0x53, 0xED),
            (0x10, 0xCA),
            (0x80, 0xCD),
            (0x9A, 0xB8),
            (0xFF, 0x16),
        ] {
            assert_eq!(sbox(input), output, "{input:#04x}");
        }
        // The S-box is a permutation without fixed points (FIPS-197 §5.1.1 design).
        let mut seen = [false; 256];
        for a in 0..=255u8 {
            let s = sbox(a);
            assert_ne!(s, a);
            seen[usize::from(s)] = true;
        }
        assert!(seen.iter().all(|&hit| hit));
    }
}
