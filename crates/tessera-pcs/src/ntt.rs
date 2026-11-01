// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The number-theoretic transform over the power-of-two subgroups of `Fp^*` (RFC-0002
//! 'Constants', RFC-0011 'Encoding').
//!
//! With `ω_n = 7^((P − 1)/n)`, the transform of size `n` evaluates `f(X) = Σ_i a_i·X^i` at
//! `ω_n^0, …, ω_n^{n−1}`, and the coset transform at `s·ω_n^0, …, s·ω_n^{n−1}`. Both sides are in
//! natural order.
//!
//! # Algorithm
//!
//! Two radix-2 kernels share one twiddle table, `Z[t] = ω_n^{rev(t)}` for `t < n/2`, where `rev`
//! reverses the `log2 n − 1` low bits of `t`:
//!
//! - [`cooley_tukey`] takes values in natural order and leaves them in bit-reversed order. It
//!   walks the remainder tree of `X^n − s^n`: a group of `2m` entries holds `f mod (X^{2m} − c^2)`,
//!   and the butterflies `(u, v) ↦ (u + c·v, u − c·v)` split it into `f mod (X^m − c)` and
//!   `f mod (X^m + c)`. Group `g` of a level uses `c = s^m·Z[g]`, one twiddle for the whole
//!   group, and its halves are groups `2g` and `2g + 1` of the next level, because
//!   `Z[2g]^2 = Z[g]` and `Z[2g + 1] = ω_4·Z[2g]` squares to `−Z[g]`. The leaf in position `p`
//!   ends holding `f(s·ω_n^{rev(p)})`. On the subgroup, `s = 1`.
//! - [`gentleman_sande`] runs the same tree upward: values in bit-reversed order, butterflies
//!   `(u, v) ↦ (u + v, c·(u − v))`, levels in reverse order, natural order out. With the inverse
//!   twiddles `c^{−1}` it undoes [`cooley_tukey`] up to a factor `n`: from the values of `f` at
//!   `s·ω_n^{rev(p)}` it rebuilds `n` times the coefficients of `f`. With the forward twiddles on
//!   the subgroup it is the transpose of [`cooley_tukey`]: the DFT matrix `F` is symmetric, so
//!   with `R` the bit-reversal permutation, `(R·F)^T = F·R`, and it evaluates in natural order
//!   from coefficients in bit-reversed order.
//!
//! `Z` does not depend on `n`, except through its length: bit `j` of `t` contributes the factor
//! `ω_{2^{j+2}}`, so every table is a prefix of the table of a larger size, and the first entry
//! is 1, which spares the multiplications of the first group of every level on the subgroup.
//!
//! A transform in natural order on both sides is one kernel and one bit-reversal permutation,
//! which works on tiles of 16 × 16 entries so that it reads and writes whole cache lines. The
//! encoding chains the two kernels and needs no permutation ([`RowEncoder`](crate::RowEncoder)).
//!
//! Each kernel recurses depth-first until a group fits in the first-level cache, then finishes
//! it level by level, so that a transform of `2^23` points streams the whole array from memory
//! only for its first levels. Nothing is allocated per transform. Control flow depends on
//! lengths and indices only, never on values (INV-ARCH-2).

use core::fmt;

use tessera_field::{Field, Goldilocks, PrimeField};

use crate::PcsError;

type Fp = Goldilocks;

/// `log2` of the largest power-of-two subgroup of `Fp^*`: `P − 1 = 2^32·(2^32 − 1)`.
pub(crate) const TWO_ADICITY: u32 = Goldilocks::TWO_ADICITY;

/// Groups of at most this many entries are finished level by level: 32 KiB of values, which stay
/// in the first-level cache while every level runs over them.
const LEAF: usize = 1 << 12;

/// `log2` of the side of a bit-reversal tile.
const TILE_BITS: u32 = 4;
/// The side of a bit-reversal tile: 16 entries, one 128-byte cache line.
const TILE: usize = 1 << TILE_BITS;

/// `log2 len` for a power of two `len` no larger than `2^TWO_ADICITY`.
pub(crate) fn log2_len(len: usize) -> Result<u32, PcsError> {
    if !len.is_power_of_two() {
        return Err(PcsError::NotPowerOfTwo { len });
    }
    let log_len = len.trailing_zeros();
    if log_len > TWO_ADICITY {
        return Err(PcsError::TooLong { log_len });
    }
    Ok(log_len)
}

/// `rev(i)` over `bits` bits: bit `j` of `i < 2^bits` moves to bit `bits − 1 − j`.
#[inline]
pub(crate) fn reverse_bits(i: usize, bits: u32) -> usize {
    // A shift by the full width overflows; zero bits reverse to zero.
    i.reverse_bits()
        .checked_shr(usize::BITS - bits)
        .unwrap_or(0)
}

/// `out[t] = scale·base^{rev(t)}` for `t < 2^bits`, with `rev` over `bits` bits.
///
/// Bit `j` of `t` lands on bit `bits − 1 − j` of `rev(t)`, so for `t' < 2^j`,
/// `out[2^j + t'] = out[t']·base^{2^{bits − 1 − j}}`: each doubling of the table is a copy of it
/// times one constant, `2^bits − 1` multiplications in all.
pub(crate) fn bit_reversed_powers(base: Fp, bits: u32, scale: Fp) -> Vec<Fp> {
    // squares[e] = base^{2^e}.
    let mut squares = Vec::with_capacity(bits as usize);
    let mut square = base;
    for _ in 0..bits {
        squares.push(square);
        square = square.square();
    }
    let mut out = Vec::with_capacity(1 << bits);
    out.push(scale);
    for factor in squares.iter().rev() {
        let half = out.len();
        out.extend_from_within(..);
        for x in &mut out[half..] {
            *x *= *factor;
        }
    }
    out
}

/// The twiddle table of the transforms of size `2^log_n`: `Z[t] = ω^{rev(t)}` with `ω = ω_n`
/// (forward) or `ω = ω_n^{−1}` (inverse), for `t < n/2`. `ω_n = 7^((P − 1)/n)` is
/// `two_adic_generator(log_n)` (RFC-0002 'Constants').
pub(crate) fn twiddles(log_n: u32, inverse: bool) -> Vec<Fp> {
    let Some(bits) = log_n.checked_sub(1) else {
        // A transform of one point has no butterflies.
        return Vec::new();
    };
    let root = Fp::two_adic_generator(log_n);
    // ω^{−1} = ω^{n−1}.
    let base = if inverse {
        root.pow((1u64 << log_n) - 1)
    } else {
        root
    };
    bit_reversed_powers(base, bits, Fp::ONE)
}

/// `1/n` for `n = 2^log_n`: `1/2 = (P + 1)/2`, a power of which is `1/n`.
pub(crate) fn inverse_of_power_of_two(log_n: u32) -> Fp {
    Fp::from_u64_reduced(Fp::MODULUS / 2 + 1).pow(u64::from(log_n))
}

/// The points a kernel works on: the subgroup `L`, or a coset `s·L`, whose shift multiplies the
/// twiddle of every group.
pub(crate) trait Shift {
    /// Whether the first group of every level has twiddle 1, so that its butterflies skip the
    /// multiplication.
    const UNIT_FIRST: bool;

    /// The twiddle of a group whose halves hold `2^log_half` entries, from its table entry `z`.
    fn twiddle(&self, z: Fp, log_half: u32) -> Fp;
}

/// The subgroup `L`: the table's twiddles as they are.
pub(crate) struct Subgroup;

impl Shift for Subgroup {
    const UNIT_FIRST: bool = true;

    #[inline(always)]
    fn twiddle(&self, z: Fp, _: u32) -> Fp {
        z
    }
}

/// The coset `s·L`: a group whose halves hold `m = 2^e` entries takes `s^m·z`.
pub(crate) struct Coset {
    /// `s^{2^e}` for `e < 32`.
    powers: [Fp; TWO_ADICITY as usize],
}

impl Coset {
    pub(crate) fn new(shift: Fp) -> Self {
        let mut powers = [shift; TWO_ADICITY as usize];
        for e in 1..powers.len() {
            powers[e] = powers[e - 1].square();
        }
        Self { powers }
    }
}

impl Shift for Coset {
    const UNIT_FIRST: bool = false;

    #[inline(always)]
    fn twiddle(&self, z: Fp, log_half: u32) -> Fp {
        z * self.powers[log_half as usize]
    }
}

/// One level of Cooley–Tukey butterflies between the halves of a group: `(u, v) ↦ (u + c·v,
/// u − c·v)`.
#[inline(always)]
fn ct_butterflies(lo: &mut [Fp], hi: &mut [Fp], c: Fp) {
    for (u, v) in lo.iter_mut().zip(hi.iter_mut()) {
        let t = *v * c;
        *v = *u - t;
        *u += t;
    }
}

/// [`ct_butterflies`] with `c = 1`.
#[inline(always)]
fn ct_unit_butterflies(lo: &mut [Fp], hi: &mut [Fp]) {
    for (u, v) in lo.iter_mut().zip(hi.iter_mut()) {
        let t = *v;
        *v = *u - t;
        *u += t;
    }
}

/// One level of Gentleman–Sande butterflies between the halves of a group: `(u, v) ↦ (u + v,
/// c·(u − v))`.
#[inline(always)]
fn gs_butterflies(lo: &mut [Fp], hi: &mut [Fp], c: Fp) {
    for (u, v) in lo.iter_mut().zip(hi.iter_mut()) {
        let (a, b) = (*u, *v);
        *u = a + b;
        *v = (a - b) * c;
    }
}

/// [`gs_butterflies`] with `c = 1`.
#[inline(always)]
fn gs_unit_butterflies(lo: &mut [Fp], hi: &mut [Fp]) {
    for (u, v) in lo.iter_mut().zip(hi.iter_mut()) {
        let (a, b) = (*u, *v);
        *u = a + b;
        *v = a - b;
    }
}

/// Natural order in, bit-reversed order out: position `p` ends holding
/// `Σ_i values[i]·(s·ω^{rev(p)})^i`, with `rev` over `log2 n` bits, `ω` the root of the table
/// and `s` the shift (1 on the subgroup).
///
/// `twiddles` holds at least `n/2` entries; a longer table works, since tables are prefixes of
/// one another.
pub(crate) fn cooley_tukey<S: Shift>(values: &mut [Fp], twiddles: &[Fp], shift: &S) {
    ct_group(values, twiddles, shift, 0);
}

/// The Cooley–Tukey levels of the group `values`, the `group`-th of its level.
fn ct_group<S: Shift>(values: &mut [Fp], twiddles: &[Fp], shift: &S, group: usize) {
    if values.len() <= LEAF {
        ct_leaf(values, twiddles, shift, group);
        return;
    }
    let half = values.len() / 2;
    let (lo, hi) = values.split_at_mut(half);
    if S::UNIT_FIRST && group == 0 {
        ct_unit_butterflies(lo, hi);
    } else {
        ct_butterflies(
            lo,
            hi,
            shift.twiddle(twiddles[group], half.trailing_zeros()),
        );
    }
    ct_group(lo, twiddles, shift, 2 * group);
    ct_group(hi, twiddles, shift, 2 * group + 1);
}

/// Every level of a group small enough for the first-level cache, one level at a time. Below a
/// group `g`, the groups of the `d`-th level are `g·2^d, …, g·2^d + 2^d − 1`.
fn ct_leaf<S: Shift>(values: &mut [Fp], twiddles: &[Fp], shift: &S, group: usize) {
    let mut half = values.len() / 2;
    let mut first = group;
    let mut count = 1;
    while half > 0 {
        let log_half = half.trailing_zeros();
        let mut groups = values
            .chunks_exact_mut(2 * half)
            .zip(&twiddles[first..first + count]);
        if S::UNIT_FIRST
            && first == 0
            && let Some((chunk, _)) = groups.next()
        {
            let (lo, hi) = chunk.split_at_mut(half);
            ct_unit_butterflies(lo, hi);
        }
        for (chunk, &z) in groups {
            let (lo, hi) = chunk.split_at_mut(half);
            ct_butterflies(lo, hi, shift.twiddle(z, log_half));
        }
        half /= 2;
        first *= 2;
        count *= 2;
    }
}

/// Bit-reversed order in, natural order out, the levels of [`cooley_tukey`] in reverse order.
///
/// With the forward table on the subgroup, position `j` ends holding
/// `Σ_p values[p]·ω^{rev(p)·j}`. With the inverse table of `ω` and the shift `s^{−1}`, it maps
/// the values of `f` at `s·ω^{rev(p)}`, in position `p`, to `n·a_j` in position `j`, where `a_j`
/// are the coefficients of `f`. `twiddles` holds at least `n/2` entries, as for
/// [`cooley_tukey`].
pub(crate) fn gentleman_sande<S: Shift>(values: &mut [Fp], twiddles: &[Fp], shift: &S) {
    gs_group(values, twiddles, shift, 0);
}

/// The Gentleman–Sande levels of the group `values`, the `group`-th of its level.
fn gs_group<S: Shift>(values: &mut [Fp], twiddles: &[Fp], shift: &S, group: usize) {
    if values.len() <= LEAF {
        gs_leaf(values, twiddles, shift, group);
        return;
    }
    let half = values.len() / 2;
    let (lo, hi) = values.split_at_mut(half);
    gs_group(lo, twiddles, shift, 2 * group);
    gs_group(hi, twiddles, shift, 2 * group + 1);
    if S::UNIT_FIRST && group == 0 {
        gs_unit_butterflies(lo, hi);
    } else {
        gs_butterflies(
            lo,
            hi,
            shift.twiddle(twiddles[group], half.trailing_zeros()),
        );
    }
}

/// [`ct_leaf`] upward: the deepest level first.
fn gs_leaf<S: Shift>(values: &mut [Fp], twiddles: &[Fp], shift: &S, group: usize) {
    let mut half = 1usize;
    let mut count = values.len() / 2;
    let mut first = group * count;
    while count > 0 {
        let log_half = half.trailing_zeros();
        let mut groups = values
            .chunks_exact_mut(2 * half)
            .zip(&twiddles[first..first + count]);
        if S::UNIT_FIRST
            && first == 0
            && let Some((chunk, _)) = groups.next()
        {
            let (lo, hi) = chunk.split_at_mut(half);
            gs_unit_butterflies(lo, hi);
        }
        for (chunk, &z) in groups {
            let (lo, hi) = chunk.split_at_mut(half);
            gs_butterflies(lo, hi, shift.twiddle(z, log_half));
        }
        half *= 2;
        first /= 2;
        count /= 2;
    }
}

/// The bit-reversal permutation, each value multiplied by `scale` on the way: position
/// `rev(i)` receives `scale·values[i]`, `rev` over `log2 n` bits.
///
/// An index splits into a high and a low part of [`TILE_BITS`] bits around a middle part `m`, and
/// `rev(h, m, l) = (rev(l), rev(m), rev(h))`: the tile of middle part `m`, 16 rows of 16
/// consecutive entries, moves transposed to the tile of `rev(m)`. Each pair of tiles goes through
/// two buffers on the stack.
fn bit_reverse_scaled(values: &mut [Fp], scale: Option<Fp>) {
    let map = |x: Fp| scale.map_or(x, |s| x * s);
    let bits = values.len().trailing_zeros();
    if bits < 2 * TILE_BITS {
        for i in 0..values.len() {
            let j = reverse_bits(i, bits);
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
    let reversed: [usize; TILE] = core::array::from_fn(|i| reverse_bits(i, TILE_BITS));
    let mut a = [Fp::ZERO; TILE * TILE];
    let mut b = [Fp::ZERO; TILE * TILE];
    for m in 0..1usize << middle_bits {
        let m_rev = reverse_bits(m, middle_bits);
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

/// Precomputed twiddles for the transforms of one size `n`, a power of two from `2^0` to `2^32`.
///
/// A plan holds two tables of `n/2` twiddles, 8 bytes each (64 MiB in all at `n = 2^23`), and
/// allocates nothing per transform. Every transform runs in place, in natural order on both
/// sides, and checks its arguments before it touches the values.
#[derive(Clone)]
pub struct NttPlan {
    log_size: u32,
    /// `ω_n^{rev(t)}`, `t < n/2`.
    forward: Vec<Fp>,
    /// `ω_n^{−rev(t)}`, `t < n/2`.
    inverse: Vec<Fp>,
    /// `1/n`.
    size_inv: Fp,
}

/// Shows the size; the tables are powers of `ω_n`, public and long.
impl fmt::Debug for NttPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NttPlan")
            .field("size", &self.size())
            .finish_non_exhaustive()
    }
}

impl NttPlan {
    /// Computes the twiddles of the transforms of `size` points, about `n` multiplications.
    ///
    /// # Errors
    ///
    /// [`PcsError::NotPowerOfTwo`] if `size` is not a power of two, and [`PcsError::TooLong`] if
    /// it exceeds `2^32`.
    pub fn new(size: usize) -> Result<Self, PcsError> {
        let log_size = log2_len(size)?;
        Ok(Self {
            log_size,
            forward: twiddles(log_size, false),
            inverse: twiddles(log_size, true),
            size_inv: inverse_of_power_of_two(log_size),
        })
    }

    /// The number of points `n` of every transform of this plan.
    pub fn size(&self) -> usize {
        1 << self.log_size
    }

    fn check(&self, values: &[Fp]) -> Result<(), PcsError> {
        if values.len() == self.size() {
            Ok(())
        } else {
            Err(PcsError::TransformLength {
                expected: self.size(),
                found: values.len(),
            })
        }
    }

    /// Evaluates `f(X) = Σ_i values[i]·X^i` at `ω_n^0, …, ω_n^{n−1}`, in place: afterwards
    /// `values[j] = f(ω_n^j)`.
    ///
    /// # Errors
    ///
    /// [`PcsError::TransformLength`] unless `values` holds exactly `n` values.
    pub fn ntt(&self, values: &mut [Fp]) -> Result<(), PcsError> {
        self.check(values)?;
        cooley_tukey(values, &self.forward, &Subgroup);
        bit_reverse_scaled(values, None);
        Ok(())
    }

    /// The inverse of [`ntt`](Self::ntt), in place: from the values of `f` at
    /// `ω_n^0, …, ω_n^{n−1}`, the coefficients of the polynomial of degree below `n` that takes
    /// them.
    ///
    /// # Errors
    ///
    /// [`PcsError::TransformLength`] unless `values` holds exactly `n` values.
    pub fn intt(&self, values: &mut [Fp]) -> Result<(), PcsError> {
        self.check(values)?;
        bit_reverse_scaled(values, Some(self.size_inv));
        gentleman_sande(values, &self.inverse, &Subgroup);
        Ok(())
    }

    /// Evaluates `f(X) = Σ_i values[i]·X^i` on the coset `shift·L`, in place: afterwards
    /// `values[j] = f(shift·ω_n^j)`.
    ///
    /// # Errors
    ///
    /// [`PcsError::TransformLength`] unless `values` holds exactly `n` values, then
    /// [`PcsError::ZeroShift`] if `shift` is zero.
    pub fn coset_ntt(&self, values: &mut [Fp], shift: Fp) -> Result<(), PcsError> {
        self.check(values)?;
        if shift == Fp::ZERO {
            return Err(PcsError::ZeroShift);
        }
        cooley_tukey(values, &self.forward, &Coset::new(shift));
        bit_reverse_scaled(values, None);
        Ok(())
    }

    /// The inverse of [`coset_ntt`](Self::coset_ntt), in place: from the values of `f` at
    /// `shift·ω_n^0, …, shift·ω_n^{n−1}`, the coefficients of the polynomial of degree below `n`
    /// that takes them.
    ///
    /// # Errors
    ///
    /// [`PcsError::TransformLength`] unless `values` holds exactly `n` values, then
    /// [`PcsError::ZeroShift`] if `shift` is zero.
    pub fn coset_intt(&self, values: &mut [Fp], shift: Fp) -> Result<(), PcsError> {
        self.check(values)?;
        // Every non-zero element is invertible.
        let Some(shift_inv) = shift.inverse() else {
            return Err(PcsError::ZeroShift);
        };
        bit_reverse_scaled(values, Some(self.size_inv));
        gentleman_sande(values, &self.inverse, &Coset::new(shift_inv));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The pre-optimization scalar traversal, independently spelling out each butterfly.
    // Preserve it as the INV-FIELD-5 / INV-TCE-1 oracle for fused leaf stages (#659).
    fn reference<S: Shift>(values: &mut [Fp], table: &[Fp], shift: &S, inverse: bool) {
        let n = values.len();
        for stage in 0..n.trailing_zeros() {
            let half = if inverse {
                1 << stage
            } else {
                n >> (stage + 1)
            };
            for (group, chunk) in values.chunks_exact_mut(2 * half).enumerate() {
                let c = shift.twiddle(table[group], half.trailing_zeros());
                let (lo, hi) = chunk.split_at_mut(half);
                for (u, v) in lo.iter_mut().zip(hi) {
                    let (a, b) = (*u, *v);
                    if inverse {
                        *u = a + b;
                        *v = (a - b) * c;
                    } else {
                        let t = b * c;
                        *u = a + t;
                        *v = a - t;
                    }
                }
            }
        }
    }

    fn compare<S: Shift>(input: &[Fp], table: &[Fp], shift: &S, offset: usize) {
        for inverse in [false, true] {
            let mut expected = input.to_vec();
            reference(&mut expected, table, shift, inverse);
            let mut backing = vec![Fp::ONE; input.len() + offset + 3];
            backing[offset..offset + input.len()].copy_from_slice(input);
            let actual = &mut backing[offset..offset + input.len()];
            if inverse {
                gentleman_sande(actual, table, shift);
            } else {
                cooley_tukey(actual, table, shift);
            }
            assert_eq!(
                actual,
                expected,
                "n={}, offset={offset}, gs={inverse}",
                input.len()
            );
            assert!(backing[..offset].iter().all(|x| *x == Fp::ONE));
            assert!(
                backing[offset + input.len()..]
                    .iter()
                    .all(|x| *x == Fp::ONE)
            );
        }
    }

    #[test]
    fn transforms_match_scalar_stages_at_boundaries_and_offsets() {
        let boundary = [
            0,
            1,
            Fp::MODULUS - 1,
            (1 << 32) - 1,
            1 << 32,
            Fp::MODULUS - (1 << 32),
            u64::MAX,
        ];
        let mut state = 0x7e55_e7a0_0659_0001u64;
        for log_n in 0..=14 {
            let n = 1 << log_n;
            for inverse_table in [false, true] {
                let table = twiddles(log_n, inverse_table);
                for offset in 0..4 {
                    let input: Vec<_> = (0..n)
                        .map(|i| {
                            state ^= state << 13;
                            state ^= state >> 7;
                            state ^= state << 17;
                            Fp::from_u64_reduced(if offset == 0 {
                                boundary[i % boundary.len()]
                            } else {
                                state
                            })
                        })
                        .collect();
                    compare(&input, &table, &Subgroup, offset);
                    compare(&input, &table, &Coset::new(Fp::from_u64_reduced(7)), offset);
                    compare(
                        &input,
                        &table,
                        &Coset::new(Fp::from_u64_reduced(Fp::MODULUS - 1)),
                        offset,
                    );
                }
            }
        }
    }
}
