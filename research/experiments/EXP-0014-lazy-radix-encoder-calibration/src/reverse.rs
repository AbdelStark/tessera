// SPDX-License-Identifier: Apache-2.0 OR MIT
use super::{mul, rev};
// Synthetic u64 adapter of the production tiled permutation; preserve its memory traffic.
const TILE_BITS: u32 = 4;
const TILE: usize = 1 << TILE_BITS;
pub(super) fn reverse(values: &mut [u64], scale: Option<u64>) {
    let map = |x: u64| scale.map_or(x, |s| mul::<false>(x, s));
    let bits = values.len().trailing_zeros();
    if bits < 2 * TILE_BITS {
        for i in 0..values.len() {
            let j = rev(i, bits);
            if i < j {
                let (x, y) = (values[i], values[j]);
                values[i] = map(y);
                values[j] = map(x);
            } else if i == j {
                values[i] = map(values[i]);
            }
        }
        return;
    }
    let middle_bits = bits - 2 * TILE_BITS;
    // The distance between consecutive rows of a tile.
    let row = TILE << middle_bits;
    let reversed: [usize; TILE] = core::array::from_fn(|i| rev(i, TILE_BITS));
    let mut a = [0u64; TILE * TILE];
    let mut b = [0u64; TILE * TILE];
    for m in 0..1usize << middle_bits {
        let m_rev = rev(m, middle_bits);
        if m_rev < m {
            continue; // swapped with m_rev already
        }
        for h in 0..TILE {
            let rows = h * row;
            a[h * TILE..][..TILE].copy_from_slice(&values[rows + m * TILE..][..TILE]);
            b[h * TILE..][..TILE].copy_from_slice(&values[rows + m_rev * TILE..][..TILE]);
        }
        // Entry (h, l) of a tile lands in row rev(l), column rev(h) of the other.
        for r in 0..TILE {
            let l = reversed[r];
            let rows = r * row;
            for (c, out) in values[rows + m_rev * TILE..][..TILE].iter_mut().enumerate() {
                *out = map(a[reversed[c] * TILE + l]);
            }
            if m_rev != m {
                for (c, out) in values[rows + m * TILE..][..TILE].iter_mut().enumerate() {
                    *out = map(b[reversed[c] * TILE + l]);
                }
            }
        }
    }
}
