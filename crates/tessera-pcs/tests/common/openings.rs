// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Openings checked by the independent verifier, `tv_core::opening` (DEP-2), through its public
//! API only, and what those checks need computed without the prover's algorithms: the claims'
//! values, as the matrix's multilinear extension from `eq`'s product formula, in `tv-field`'s
//! arithmetic.

use tessera_field::{Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_hash::Digest;
use tessera_pcs::{ClaimGroup, ColumnAnswer, CombinedRow, CommitmentParams, RowLayout};
use tv_core::opening as tv;

pub type TvFp = tv_field::Fp;
pub type TvFp2 = tv_field::Fp2;

pub fn tv_fp(x: Goldilocks) -> TvFp {
    TvFp::from_canonical(x.to_canonical_u64()).expect("canonical")
}

pub fn tv_fp2(x: GoldilocksExt2) -> TvFp2 {
    let [c0, c1] = x.coeffs();
    TvFp2::new(tv_fp(c0), tv_fp(c1))
}

pub fn tv_fp2s(xs: &[GoldilocksExt2]) -> Vec<TvFp2> {
    xs.iter().copied().map(tv_fp2).collect()
}

/// `eq(point, index) = Π_t (point_t·x_t + (1 − point_t)·(1 − x_t))`, `x_t` bit `t` of `index`.
pub fn eq_at(point: &[TvFp2], index: u64) -> TvFp2 {
    point
        .iter()
        .enumerate()
        .fold(TvFp2::ONE, |product, (t, &r)| {
            let bit = TvFp2::from(TvFp::reduce((index >> t) & 1));
            product * (r * bit + (TvFp2::ONE - r) * (TvFp2::ONE - bit))
        })
}

/// A matrix of `2^row_vars` rows of `k` values, of which `rows` are non-zero and hold `values`,
/// row after row; every other row is zero.
pub struct Matrix<'a> {
    pub log2_k: u32,
    pub rows: Vec<u64>,
    pub values: &'a [Goldilocks],
}

impl Matrix<'_> {
    /// `Σ_i A_row[i]·eq(column_point, i)` for every non-zero row, in layout order.
    pub fn at_column_point(&self, column_point: &[TvFp2]) -> Vec<TvFp2> {
        let k = 1usize << self.log2_k;
        let weights: Vec<TvFp2> = (0..k as u64).map(|i| eq_at(column_point, i)).collect();
        self.values
            .chunks_exact(k)
            .map(|row| {
                row.iter()
                    .zip(&weights)
                    .fold(TvFp2::ZERO, |sum, (&a, &w)| sum + w * tv_fp(a))
            })
            .collect()
    }

    /// The claimed values of every group, `Ã(column_point ‖ row_point)`: the sum over the
    /// non-zero rows of `eq(row_point, row)·Σ_i A_row[i]·eq(column_point, i)`.
    pub fn values(&self, groups: &[ClaimGroup]) -> Vec<Vec<TvFp2>> {
        groups
            .iter()
            .map(|group| {
                let by_row = self.at_column_point(&tv_fp2s(&group.column_point));
                group
                    .claims
                    .iter()
                    .map(|claim| {
                        let point = tv_fp2s(&claim.row_point);
                        self.rows
                            .iter()
                            .zip(&by_row)
                            .fold(TvFp2::ZERO, |sum, (&row, &v)| sum + eq_at(&point, row) * v)
                    })
                    .collect()
            })
            .collect()
    }
}

/// The verifier's groups: the prover's points and challenges with the claimed `values`.
pub fn tv_groups(groups: &[ClaimGroup], values: &[Vec<TvFp2>]) -> Vec<tv::Group> {
    groups
        .iter()
        .zip(values)
        .map(|(group, values)| tv::Group {
            column_point: tv_fp2s(&group.column_point),
            claims: group
                .claims
                .iter()
                .zip(values)
                .map(|(claim, &value)| tv::Claim {
                    row_point: tv_fp2s(&claim.row_point),
                    gamma: tv_fp2(claim.gamma),
                    value,
                })
                .collect(),
        })
        .collect()
}

pub fn tv_combined(combined: &[CombinedRow]) -> Vec<tv::CombinedRow> {
    combined
        .iter()
        .map(|row| tv::CombinedRow {
            u: tv_fp2s(&row.u),
            mask: tv_fp2s(&row.mask),
        })
        .collect()
}

pub fn tv_columns(answer: &ColumnAnswer) -> Vec<Vec<TvFp>> {
    answer
        .columns
        .iter()
        .map(|column| column.iter().copied().map(tv_fp).collect())
        .collect()
}

/// What the verifier knows before the combined rows arrive.
pub struct Verifier<'a> {
    pub params: &'a CommitmentParams,
    pub layout: &'a RowLayout,
    pub root: Digest,
    pub groups: &'a [tv::Group],
    pub xi_point: &'a [TvFp2],
}

impl Verifier<'_> {
    pub fn opening(&self) -> tv::Opening<'_> {
        tv::Opening {
            params: tv::Params {
                log2_k: self.params.log2_k,
                log2_inv_rate: 1,
                mask_coeffs: u32::try_from(self.params.mask_coeffs).expect("P_A ≤ k ≤ 2^31"),
                row_block: u32::try_from(self.params.row_block.get()).expect("a small B"),
                commitment_id: self.params.commitment_id,
            },
            layout: tv::Layout {
                row_vars: self.layout.row_vars(),
                segments: self.layout.segments(),
            },
            root: self.root,
            groups: self.groups,
            xi_point: self.xi_point,
        }
    }

    /// `check_evaluations`, then `check_columns` on what it returns.
    pub fn verify(
        &self,
        combined: &[tv::CombinedRow],
        queries: &[u64],
        columns: &[Vec<TvFp>],
        siblings: &[Digest],
    ) -> Result<(), tv::OpeningError> {
        tv::check_evaluations(self.opening(), combined)?.check_columns(queries, columns, siblings)
    }
}
