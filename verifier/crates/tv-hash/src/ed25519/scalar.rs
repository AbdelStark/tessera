// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Integers modulo the group order `L = 2^252 + 27742317777372353535851937790883648493`
//! (RFC 8032 §5.1): the range check on `S` and the reduction of `SHA-512(R ‖ A ‖ M)`.
//!
//! The reduction is binary long division, one bit of the 512-bit dividend at a time from the
//! top: a slow and plain algorithm, chosen because each step is evidently right and one hash
//! is reduced per signature.

/// `L` as four little-endian 64-bit limbs.
const L: [u64; 4] = [
    0x5812_631a_5cf5_d3ed,
    0x14de_f9de_a2f7_9cd6,
    0x0000_0000_0000_0000,
    0x1000_0000_0000_0000,
];

/// The four little-endian limbs of 32 little-endian bytes.
fn limbs(bytes: &[u8; 32]) -> [u64; 4] {
    let mut out = [0u64; 4];
    for (limb, chunk) in out.iter_mut().zip(bytes.chunks_exact(8)) {
        let mut le = [0u8; 8];
        le.copy_from_slice(chunk);
        *limb = u64::from_le_bytes(le);
    }
    out
}

/// `v − L` when `v ≥ L`, else `None`.
fn minus_l(v: &[u64; 4]) -> Option<[u64; 4]> {
    let mut out = [0u64; 4];
    let mut borrow = false;
    for ((slot, a), b) in out.iter_mut().zip(v).zip(L) {
        let (d, b1) = a.overflowing_sub(b);
        let (d, b2) = d.overflowing_sub(u64::from(borrow));
        *slot = d;
        borrow = b1 || b2;
    }
    (!borrow).then_some(out)
}

/// Whether the 256-bit little-endian integer `s` is below `L`: the range RFC-0060 requires of a
/// signature's `S`, all 256 bits counted.
pub(super) fn is_below_l(s: &[u8; 32]) -> bool {
    minus_l(&limbs(s)).is_none()
}

/// The 512-bit little-endian integer `h` modulo `L`, as 32 little-endian bytes.
///
/// The remainder `r` starts at 0; for each bit of `h` from bit 511 down, `r = 2r + bit`, then
/// `r − L` replaces `r` if it is not negative. `r < L < 2^253` before each step, so `2r + 1`
/// fits in 254 bits.
pub(super) fn reduce_wide(h: &[u8; 64]) -> [u8; 32] {
    let mut r = [0u64; 4];
    for byte in h.iter().rev() {
        for bit in (0..8u32).rev() {
            let incoming = u64::from(byte.wrapping_shr(bit) & 1);
            r = [
                r[0] << 1 | incoming,
                r[1] << 1 | r[0] >> 63,
                r[2] << 1 | r[1] >> 63,
                r[3] << 1 | r[2] >> 63,
            ];
            if let Some(reduced) = minus_l(&r) {
                r = reduced;
            }
        }
    }
    let mut out = [0u8; 32];
    for (chunk, limb) in out.chunks_exact_mut(8).zip(r) {
        chunk.copy_from_slice(&limb.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(limbs: [u64; 4]) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (chunk, limb) in out.chunks_exact_mut(8).zip(limbs) {
            chunk.copy_from_slice(&limb.to_le_bytes());
        }
        out
    }

    #[test]
    fn the_range_check_is_exact_at_l() {
        let l = bytes(L);
        assert!(!is_below_l(&l));
        let mut l_minus_1 = l;
        l_minus_1[0] -= 1;
        assert!(is_below_l(&l_minus_1));
        assert!(is_below_l(&[0; 32]));
        assert!(!is_below_l(&[0xff; 32]));
        let mut top = [0u8; 32];
        top[31] = 0x80;
        assert!(!is_below_l(&top));
    }

    #[test]
    fn reduction_matches_small_cases() {
        // L itself, L + 5, 2L + 7 and 2^512 − 1 against values computed by hand.
        let wide = |v: [u8; 32], hi: [u8; 32]| -> [u8; 64] {
            let mut out = [0u8; 64];
            out[..32].copy_from_slice(&v);
            out[32..].copy_from_slice(&hi);
            out
        };
        assert_eq!(reduce_wide(&wide(bytes(L), [0; 32])), [0; 32]);
        let mut l_plus_5 = bytes(L);
        l_plus_5[0] += 5;
        let mut five = [0u8; 32];
        five[0] = 5;
        assert_eq!(reduce_wide(&wide(l_plus_5, [0; 32])), five);
        // 2L + 7 = (L << 1) + 7.
        let two_l = [
            L[0] << 1,
            L[1] << 1 | L[0] >> 63,
            L[2] << 1 | L[1] >> 63,
            L[3] << 1 | L[2] >> 63,
        ];
        let mut two_l_plus_7 = bytes(two_l);
        two_l_plus_7[0] += 7;
        let mut seven = [0u8; 32];
        seven[0] = 7;
        assert_eq!(reduce_wide(&wide(two_l_plus_7, [0; 32])), seven);
        // 2^512 − 1 mod L, from Python: (2**512 - 1) % L.
        let expected = bytes([
            0xa406_11e3_449c_0f00,
            0xd00e_1ba7_6885_9347,
            0xceec_73d2_17f5_be65,
            0x0399_411b_7c30_9a3d,
        ]);
        assert_eq!(reduce_wide(&[0xff; 64]), expected);
    }
}
