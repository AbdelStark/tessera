// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public source coordinates with fixed group bits (RFC-0077).

use super::{LeafError, own, variables};
use core::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Projection {
    vars: u32,
    mask: usize,
    bits: usize,
}

impl Projection {
    pub(super) fn full(vars: u32) -> Result<Self, LeafError> {
        Self::new(vars, &[], 0)
    }

    pub(super) fn new(vars: u32, positions: &[usize], bits: u64) -> Result<Self, LeafError> {
        1usize.checked_shl(vars).ok_or(LeafError::Memory)?;
        if positions.windows(2).any(|w| w[0] >= w[1])
            || positions.last().is_some_and(|&v| v >= vars as usize)
            || bits.checked_shr(positions.len() as u32).unwrap_or(0) != 0
        {
            return own();
        }
        let mut mask = 0;
        let mut selected = 0;
        for (j, &var) in positions.iter().enumerate() {
            mask |= 1usize << var;
            selected |= (((bits >> j) & 1) as usize) << var;
        }
        Ok(Self {
            vars,
            mask,
            bits: selected,
        })
    }

    pub(super) fn len(self) -> usize {
        1usize << (self.vars - self.mask.count_ones())
    }

    pub(super) fn selected(self, original: usize) -> Option<usize> {
        if original >= 1usize << self.vars || original & self.mask != self.bits {
            return None;
        }
        if self.mask == 0 {
            return Some(original);
        }
        let (mut index, mut next) = (0, 0);
        for var in 0..self.vars {
            if self.mask & (1usize << var) == 0 {
                index |= ((original >> var) & 1) << next;
                next += 1;
            }
        }
        Some(index)
    }

    pub(super) fn original(self, selected: usize) -> Option<usize> {
        if selected >= self.len() {
            return None;
        }
        let (mut index, mut next) = (self.bits, 0);
        for var in 0..self.vars {
            if self.mask & (1usize << var) == 0 {
                index |= ((selected >> next) & 1) << var;
                next += 1;
            }
        }
        Some(index)
    }

    /// Count selected real positions in an axis-aligned public tile, without scanning it.
    pub(super) fn count(self, shape: &[u32], ranges: &[Range<u32>]) -> Result<usize, LeafError> {
        if ranges.len() != shape.len() {
            return own();
        }
        self.count_axes(shape, shape.iter().copied().zip(ranges.iter().cloned()))
    }

    pub(super) fn real_len(self, shape: &[u32]) -> Result<usize, LeafError> {
        self.count_axes(shape, shape.iter().map(|&d| (d, 0..d)))
    }

    fn count_axes(
        self,
        shape: &[u32],
        axes: impl DoubleEndedIterator<Item = (u32, Range<u32>)>,
    ) -> Result<usize, LeafError> {
        if variables(shape) != Some(self.vars) {
            return own();
        }
        let (mut count, mut shift) = (1usize, 0);
        for (d, r) in axes.rev() {
            if d == 0 || r.start > r.end || r.end > d {
                return own();
            }
            let width = d
                .checked_next_power_of_two()
                .ok_or(LeafError::Memory)?
                .trailing_zeros();
            let axis_mask = (1usize << width) - 1;
            let mask = (self.mask >> shift) & axis_mask;
            let bits = (self.bits >> shift) & axis_mask;
            let n = prefix(r.end as usize, width, mask, bits)
                - prefix(r.start as usize, width, mask, bits);
            count = count.checked_mul(n).ok_or(LeafError::Memory)?;
            shift += width;
        }
        Ok(count)
    }
}

// Count x < end with x & mask == bits. Only public bounds/coordinates choose branches.
fn prefix(end: usize, width: u32, mask: usize, bits: usize) -> usize {
    if end == 1usize << width {
        return 1usize << (width - mask.count_ones());
    }
    let mut count = 0;
    for var in (0..width).rev() {
        let bit = 1usize << var;
        if end & bit != 0 {
            if bits & bit == 0 {
                let lower = bit - 1;
                count += 1usize << (var - (mask & lower).count_ones());
            }
            if mask & bit != 0 && bits & bit == 0 {
                break;
            }
        } else if bits & bit != 0 {
            break;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_small_cube_matches_direct_boolean_selection_and_inverse() {
        for vars in 0..=8 {
            for mask in 0usize..1 << vars {
                let positions: Vec<_> = (0..vars).filter(|v| mask & (1 << v) != 0).collect();
                for assignment in 0..1u64 << positions.len() {
                    let p = Projection::new(vars as u32, &positions, assignment).unwrap();
                    let expected: Vec<_> = (0..1usize << vars)
                        .filter(|&x| {
                            positions
                                .iter()
                                .enumerate()
                                .all(|(j, &v)| (x >> v) & 1 == ((assignment >> j) & 1) as usize)
                        })
                        .collect();
                    assert_eq!(p.len(), expected.len());
                    for (selected, &original) in expected.iter().enumerate() {
                        assert_eq!(p.original(selected), Some(original));
                        assert_eq!(p.selected(original), Some(selected));
                    }
                    assert_eq!(p.original(expected.len()), None);
                    assert_eq!(p.selected(1 << vars), None);
                    for x in 0..1usize << vars {
                        assert_eq!(p.selected(x).is_some(), expected.contains(&x));
                    }
                }
            }
        }
    }

    #[test]
    fn uneven_axes_and_request_gaps_match_integer_coordinate_enumeration() {
        let shape = [3, 5, 2];
        for mask in 0usize..1 << 6 {
            let positions: Vec<_> = (0..6).filter(|v| mask & (1 << v) != 0).collect();
            for assignment in 0..1u64 << positions.len() {
                let p = Projection::new(6, &positions, assignment).unwrap();
                for first in 0..=5 {
                    for end in first..=5 {
                        let expected = (0..3)
                            .flat_map(|h| {
                                (first..end)
                                    .flat_map(move |q| (0..2).map(move |k| (h << 4) | (q << 1) | k))
                            })
                            .filter(|&x| {
                                positions
                                    .iter()
                                    .enumerate()
                                    .all(|(j, &v)| (x >> v) & 1 == ((assignment >> j) & 1) as usize)
                            })
                            .count();
                        assert_eq!(
                            p.count(&shape, &[0..3, first as u32..end as u32, 0..2])
                                .unwrap(),
                            expected
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn invalid_public_maps_and_rectangles_refuse_before_allocation() {
        for (positions, bits) in [(vec![1, 1], 0), (vec![2, 1], 0), (vec![4], 0), (vec![1], 2)] {
            assert!(matches!(
                Projection::new(4, &positions, bits),
                Err(LeafError::Internal { .. })
            ));
        }
        assert!(matches!(
            Projection::full(usize::BITS),
            Err(LeafError::Memory)
        ));
        let p = Projection::full(3).unwrap();
        for (shape, ranges) in [
            (vec![2, 4], core::iter::once(0..2).collect()),
            (vec![2, 4], vec![0..3, 0..4]),
            (vec![4], core::iter::once(0..4).collect()),
            (vec![0, 8], vec![0..0, 0..8]),
        ] {
            assert!(matches!(
                p.count(&shape, &ranges),
                Err(LeafError::Internal { .. })
            ));
        }
    }
}
