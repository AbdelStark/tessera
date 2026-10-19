// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Openings at many points through combined rows and a proximity row (RFC-0011 'Opening
//! protocol' steps 2 to 5).
//!
//! The claims on the matrix's extension `Ã(column_point ‖ row_point) = v` come grouped by column
//! sub-point ([`ClaimGroup`]), each with its batching challenge `γ`. The prover needs neither the
//! values nor the column points:
//!
//! - Group `g`'s row coefficients are `c_g(row) = Σ_{claims of g} γ·eq(row_point, row)`, and the
//!   proximity row's are `ξ_row = eq(ξ*, row)`, at every non-zero row
//!   ([`RowLayout::row_coefficients`]).
//! - The combined rows are `u_g = Σ_row c_g(row)·A_row`, `k` values in `Fp2`, and
//!   `M_g = Σ_row c_g(row)·m_row`, `P_A` coefficients in `Fp2`, over the non-zero rows. Then
//!   `u_ξ` and `M_ξ` follow from `ξ_row`, last
//!   ([`ReferenceCommitment::open`](crate::ReferenceCommitment::open)).
//!
//! # Algorithm
//!
//! **Coefficients.** `eq(r, row) = Π_t f_t(bit t of row)`, with `f_t(1) = r_t` and
//! `f_t(0) = 1 − r_t`, coordinate `t` setting bit `t` (RFC-0010 'Multilinear extensions'). The
//! layout lists the non-zero rows in increasing order, so consecutive rows share their high
//! bits. The prover keeps the partial products `γ·Π_{s ≥ t} f_s` from the top coordinate down.
//! For each row it recomputes only those below the highest bit in which the row differs from the
//! one before. A run of consecutive rows costs about two products per row and claim. A single
//! row costs at most `row_vars` products, and no table of `2^row_vars` entries is built, so
//! layouts as sparse as the 64-bit row index allows cost no more.
//!
//! **Combination.** At production size this is the whole cost: `(G + 1)·N·(k + P_A)` products
//! of an `Fp2` coefficient by an `Fp` value. A product of canonical values is below `P^2 < 2^128`.
//! The sums are therefore kept unreduced, as 128 bits and a count of carries, and reduced once
//! per output with `2^64 ≡ 2^32 − 1` and `2^128 ≡ −2^32`. A product then costs two
//! `64 × 64 → 128` multiplications and two additions with carry, in place of two reduced
//! multiplications and two reduced additions. The columns are processed in tiles of 64, and the
//! `(G + 1)·64` accumulators of a tile, 48 bytes each, stay in the first-level cache for
//! `G + 1 ≤ 17`. Rows are taken four at a time, so that each accumulator is loaded and stored
//! once per four products. On one Apple M4 Max core, with `G + 1 = 17` rows, the product costs
//! about 0.5 ns this way and 1.7 ns with reduced field operations. The loop allocates nothing,
//! and its control flow depends on the shape only, never on a value (INV-ARCH-2).

use core::fmt;

use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};

use crate::{PcsError, RowLayout};

type Fp = Goldilocks;
type Fp2 = GoldilocksExt2;

/// The part of a claim `Ã(column_point ‖ row_point) = value` that the prover's opening uses: its
/// row point and its batching challenge. The value never reaches the prover's opening.
///
/// Both are public, from the claim list and the transcript.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowClaim {
    /// The row part of the claim's point, `row_vars` coordinates, lowest variable first:
    /// coordinate `t` sets bit `t` of the row index.
    pub row_point: Vec<Fp2>,
    /// `γ`, the claim's batching challenge.
    pub gamma: Fp2,
}

/// The claims that share a column sub-point (RFC-0011 step 1), in the order the caller batches
/// them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClaimGroup {
    /// The column part of the claims' points, `log2 k` coordinates, lowest variable first.
    pub column_point: Vec<Fp2>,
    /// The group's claims. A group may have none: it then weighs no row, and its combined row
    /// is zero.
    pub claims: Vec<RowClaim>,
}

/// A combined row (RFC-0011 step 5): `u_g` and `M_g` for a group, or `u_ξ` and `M_ξ` for the
/// proximity row.
///
/// Combined rows are protocol-class (05 §3), which includes combined rows after blinding: the
/// `Debug` form shows lengths only.
#[derive(Clone, PartialEq, Eq)]
pub struct CombinedRow {
    /// `u`: the combined row's `k` values on `H`.
    pub u: Vec<Fp2>,
    /// `M`: the combined mask's `P_A` coefficients, constant first.
    pub mask: Vec<Fp2>,
}

/// Shows the lengths: combined rows are protocol-class.
impl fmt::Debug for CombinedRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CombinedRow")
            .field("values", &self.u.len())
            .field("mask_coeffs", &self.mask.len())
            .finish()
    }
}

impl RowLayout {
    /// The row coefficients of RFC-0011 steps 2 and 3, at every non-zero row in layout order:
    /// one list per group in group order, `c_g(row) = Σ_{claims of g} γ·eq(row_point, row)`,
    /// then the proximity coefficients `ξ_row = eq(ξ*, row)`. That is `G + 1` lists of `N`
    /// values, computed from public data alone, as the verifier computes them.
    ///
    /// The column points do not enter the coefficients and are not checked here:
    /// [`ReferenceCommitment::open`](crate::ReferenceCommitment::open) checks them. Consecutive
    /// rows share their high bits, so each claim costs about two products of `Fp2` per row for
    /// a run of consecutive rows, and at most `row_vars` per row; no table of `2^row_vars`
    /// entries is built.
    ///
    /// # Errors
    ///
    /// In this order, before anything is computed: [`PcsError::RowPoint`] for the first claim,
    /// group by group and claim by claim, whose row point does not have `row_vars` coordinates,
    /// then [`PcsError::XiPoint`] if `ξ*` does not have `row_vars` coordinates.
    pub fn row_coefficients(
        &self,
        groups: &[ClaimGroup],
        xi_point: &[Fp2],
    ) -> Result<Vec<Vec<Fp2>>, PcsError> {
        // row_vars ≤ 64.
        let expected = self.row_vars() as usize;
        for (group, claims) in groups.iter().enumerate() {
            for (claim, row_claim) in claims.claims.iter().enumerate() {
                if row_claim.row_point.len() != expected {
                    return Err(PcsError::RowPoint {
                        group,
                        claim,
                        expected,
                        found: row_claim.row_point.len(),
                    });
                }
            }
        }
        if xi_point.len() != expected {
            return Err(PcsError::XiPoint {
                expected,
                found: xi_point.len(),
            });
        }
        let mut coefficients = Vec::with_capacity(groups.len() + 1);
        for group in groups {
            let mut column = vec![Fp2::ZERO; self.rows()];
            for claim in &group.claims {
                add_scaled_eq(
                    &mut column,
                    self.row_indices(),
                    &claim.row_point,
                    claim.gamma,
                );
            }
            coefficients.push(column);
        }
        let mut xi = vec![Fp2::ZERO; self.rows()];
        add_scaled_eq(&mut xi, self.row_indices(), xi_point, Fp2::ONE);
        coefficients.push(xi);
        Ok(coefficients)
    }
}

/// Adds `scale·eq(point, row)` to `out[i]` for `row` the `i`-th of `rows`. The rows lie below
/// `2^point.len()`, as a [`RowLayout`]'s do, and cost least when consecutive rows share their
/// high bits, as increasing rows do; any order gives the same values.
///
/// `partial[t] = scale·Π_{t ≤ s < vars} f_s(bit s of row)` for the current row, with
/// `partial[vars] = scale`. The partial products at and above the highest bit in which a row
/// differs from the one before stay as they were, since they depend on those bits alone.
pub(crate) fn add_scaled_eq(
    out: &mut [Fp2],
    rows: impl Iterator<Item = u64>,
    point: &[Fp2],
    scale: Fp2,
) {
    let vars = point.len();
    // factors[t] = [f_t(0), f_t(1)] = [1 − r_t, r_t].
    let factors: Vec<[Fp2; 2]> = point.iter().map(|&r| [Fp2::ONE - r, r]).collect();
    let mut partial = vec![scale; vars + 1];
    let mut previous: Option<u64> = None;
    for (slot, row) in out.iter_mut().zip(rows) {
        // The bits that may differ from the previous row's: all of them for the first row.
        let changed = previous.map_or(vars, |before| {
            (u64::BITS - (row ^ before).leading_zeros()) as usize
        });
        for t in (0..changed).rev() {
            partial[t] = partial[t + 1] * factors[t][(row >> t) as usize & 1];
        }
        *slot += partial[0];
        previous = Some(row);
    }
}

/// Columns per tile of [`combine`].
const TILE: usize = 64;

/// `hi·2^128 + lo` for each coordinate of an `Fp2` sum of products, unreduced.
#[derive(Clone, Copy, Default)]
struct Wide {
    lo0: u128,
    lo1: u128,
    hi0: u64,
    hi1: u64,
}

impl Wide {
    /// Adds `c·x`, the coordinates `c0·x` and `c1·x` of a product of canonical values, each below
    /// `P^2 < 2^128`, so that each addition carries at most once. A sum of `N` products carries
    /// at most `N < 2^64` times.
    #[inline(always)]
    fn add_product(&mut self, [c0, c1]: [u64; 2], x: Fp) {
        let x = u128::from(x.to_canonical_u64());
        let (lo0, carry0) = self.lo0.overflowing_add(u128::from(c0) * x);
        let (lo1, carry1) = self.lo1.overflowing_add(u128::from(c1) * x);
        self.lo0 = lo0;
        self.lo1 = lo1;
        self.hi0 += u64::from(carry0);
        self.hi1 += u64::from(carry1);
    }

    /// The sums modulo `P`.
    fn reduce(&self) -> Fp2 {
        Fp2::new(reduce(self.lo0, self.hi0), reduce(self.lo1, self.hi1))
    }
}

/// `hi·2^128 + lo` modulo `P`. With `lo = lo_1·2^64 + lo_0`, `2^64 ≡ 2^32 − 1` and
/// `2^128 ≡ −2^32 = −(2^32 − 1) − 1`, it is `lo_0 + (lo_1 − hi)·(2^32 − 1) − hi`.
fn reduce(lo: u128, hi: u64) -> Fp {
    let low = Fp::from_u64_reduced(lo as u64);
    let high = Fp::from_u64_reduced((lo >> 64) as u64);
    let hi = Fp::from_u64_reduced(hi);
    low + (high - hi).mul_small(0xFFFF_FFFF) - hi
}

/// `out[g][i] = Σ_r coefficients[g][r]·matrix[r·width + i]` for every list `g` and `i < width`,
/// over the `N = matrix.len()/width` rows of `matrix`: the combined rows of the lists of row
/// coefficients, each list holding `N` values. The [module documentation](self) gives the
/// algorithm.
pub(crate) fn combine(coefficients: &[Vec<Fp2>], matrix: &[Fp], width: usize) -> Vec<Vec<Fp2>> {
    let count = coefficients.len();
    if count == 0 || width == 0 {
        return vec![Vec::new(); count];
    }
    let rows = matrix.len() / width;
    // Row r's coefficients, as canonical words, side by side.
    let mut raw = vec![[0u64; 2]; rows * count];
    for (g, list) in coefficients.iter().enumerate() {
        for (r, c) in list.iter().enumerate() {
            let [c0, c1] = c.coeffs();
            raw[r * count + g] = [c0.to_canonical_u64(), c1.to_canonical_u64()];
        }
    }
    let mut out = vec![vec![Fp2::ZERO; width]; count];
    let mut acc = vec![Wide::default(); count * TILE];
    // Four rows; a width too large for a usize leaves them all to the single rows.
    let quad = width.saturating_mul(4);
    let mut start = 0;
    while start < width {
        let len = TILE.min(width - start);
        acc.fill(Wide::default());
        let quads = matrix.chunks_exact(quad);
        let singles = quads.remainder();
        let quad_coefficients = raw.chunks_exact(4 * count);
        let single_coefficients = quad_coefficients.remainder();
        for (block, c) in quads.zip(quad_coefficients) {
            let x0 = &block[start..][..len];
            let x1 = &block[width + start..][..len];
            let x2 = &block[2 * width + start..][..len];
            let x3 = &block[3 * width + start..][..len];
            for (g, sums) in acc.chunks_exact_mut(TILE).enumerate() {
                let (c0, c1, c2, c3) = (c[g], c[count + g], c[2 * count + g], c[3 * count + g]);
                for ((((sum, &a0), &a1), &a2), &a3) in
                    sums[..len].iter_mut().zip(x0).zip(x1).zip(x2).zip(x3)
                {
                    sum.add_product(c0, a0);
                    sum.add_product(c1, a1);
                    sum.add_product(c2, a2);
                    sum.add_product(c3, a3);
                }
            }
        }
        for (row, c) in singles
            .chunks_exact(width)
            .zip(single_coefficients.chunks_exact(count))
        {
            let x = &row[start..][..len];
            for (sums, &c) in acc.chunks_exact_mut(TILE).zip(c) {
                for (sum, &a) in sums[..len].iter_mut().zip(x) {
                    sum.add_product(c, a);
                }
            }
        }
        for (list, sums) in out.iter_mut().zip(acc.chunks_exact(TILE)) {
            for (value, sum) in list[start..start + len].iter_mut().zip(&sums[..len]) {
                *value = sum.reduce();
            }
        }
        start += len;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{Fp, reduce};
    use tessera_field::PrimeField;

    /// `hi·2^128 + lo` modulo `P` by `u128` remainders: `2^128 mod P = P − 2^32`, so the value
    /// is `(hi mod P)·(P − 2^32) + lo mod P`, a sum below `2^128`.
    fn by_remainders(lo: u128, hi: u64) -> u64 {
        let p = u128::from(Fp::MODULUS);
        let wide = (u128::from(hi) % p) * (p - (1 << 32)) + lo % p;
        (wide % p) as u64
    }

    #[test]
    fn the_reduction_of_unreduced_sums_matches_remainders() {
        let p = Fp::MODULUS;
        let words = [
            0,
            1,
            2,
            (1 << 32) - 1,
            1 << 32,
            p - 1,
            p,
            p + 1,
            u64::MAX - 1,
            u64::MAX,
        ];
        for &lo_lo in &words {
            for &lo_hi in &words {
                for &hi in &words {
                    let lo = (u128::from(lo_hi) << 64) | u128::from(lo_lo);
                    assert_eq!(
                        reduce(lo, hi).to_canonical_u64(),
                        by_remainders(lo, hi),
                        "lo = {lo:#x}, hi = {hi:#x}"
                    );
                }
            }
        }
        let mut state = 0x7E55_E7A0_0054_0001u64;
        let mut next = move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        for _ in 0..100_000 {
            let lo = (u128::from(next()) << 64) | u128::from(next());
            let hi = next();
            assert_eq!(reduce(lo, hi).to_canonical_u64(), by_remainders(lo, hi));
        }
    }
}
