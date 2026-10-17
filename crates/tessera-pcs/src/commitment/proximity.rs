// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The proximity check against a cheating prover (RFC-0011 'Opening protocol', soundness, and
//! 'Testing Strategy'). The prover corrupts one row's codeword at exactly `⌈δ·n⌉` columns `D`,
//! commits to the corrupted codewords and sends the honest combined rows, from the row's
//! message.
//!
//! Steps 8.1 and 8.2 pass on every draw, since the columns are the committed ones. A draw that
//! hits `D` fails step 8.3 for the proximity row, whose coefficient `eq(ξ*, row)` of the
//! corrupted row is not zero. Each of `10^4` seeded draws of `τ` queries, with replacement,
//! must therefore be rejected exactly when it hits `D`, with `Reject(Proximity)` at the smallest
//! queried column in `D`. Over the draws, the rejection rate must reach `1 − (1 − δ')^τ`, `δ'`
//! the corrupted fraction, less four standard deviations of the binomial count.
//!
//! The test sits inside the crate because the cheating prover needs to commit to codewords that
//! no encoding produces: [`ReferenceCommitment::from_encoded`] is the commitment's own chaining,
//! merging and tree over given codewords. The independent verifier, `tv_core::opening`, judges
//! every opening (DEP-2).

use core::num::NonZeroUsize;

use tessera_codec::{Secret, SecretBuf};
use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tv_core::opening as tv;

use super::{CommitmentParams, ReferenceCommitment, RowLayout};
use crate::{ClaimGroup, RowClaim, RowEncoder};

type Fp = Goldilocks;
type Fp2 = GoldilocksExt2;
type TvFp = tv_field::Fp;
type TvFp2 = tv_field::Fp2;

/// SplitMix64 (Steele, Lea and Flood, 2014): reproducible inputs and draws, nothing secret.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform element: words at or above `P` are rejected.
    fn fp(&mut self) -> Fp {
        loop {
            if let Some(x) = Fp::from_canonical_u64(self.next_u64()) {
                return x;
            }
        }
    }

    fn fps(&mut self, n: usize) -> Vec<Fp> {
        (0..n).map(|_| self.fp()).collect()
    }

    fn fp2(&mut self) -> Fp2 {
        let c0 = self.fp();
        Fp2::new(c0, self.fp())
    }

    fn fp2s(&mut self, n: usize) -> Vec<Fp2> {
        (0..n).map(|_| self.fp2()).collect()
    }
}

fn tv_fp(x: Fp) -> TvFp {
    TvFp::from_canonical(x.to_canonical_u64()).expect("canonical")
}

fn tv_fp2(x: Fp2) -> TvFp2 {
    let [c0, c1] = x.coeffs();
    TvFp2::new(tv_fp(c0), tv_fp(c1))
}

fn tv_fp2s(xs: &[Fp2]) -> Vec<TvFp2> {
    xs.iter().copied().map(tv_fp2).collect()
}

/// `eq(point, index)` from its product formula, in the verifier's arithmetic.
fn eq_at(point: &[TvFp2], index: u64) -> TvFp2 {
    point
        .iter()
        .enumerate()
        .fold(TvFp2::ONE, |product, (t, &r)| {
            let bit = TvFp2::from(TvFp::reduce((index >> t) & 1));
            product * (r * bit + (TvFp2::ONE - r) * (TvFp2::ONE - bit))
        })
}

/// The verifier's groups, each claim's value the matrix's extension at its point:
/// `Σ_row eq(row_point, row)·Σ_i A_row[i]·eq(column_point, i)`.
fn claimed(groups: &[ClaimGroup], rows: &[u64], values: &[Fp], k: usize) -> Vec<tv::Group> {
    groups
        .iter()
        .map(|group| {
            let column_point = tv_fp2s(&group.column_point);
            let by_row: Vec<TvFp2> = values
                .chunks_exact(k)
                .map(|row| {
                    row.iter().enumerate().fold(TvFp2::ZERO, |sum, (i, &a)| {
                        sum + eq_at(&column_point, i as u64) * TvFp2::from(tv_fp(a))
                    })
                })
                .collect();
            tv::Group {
                column_point,
                claims: group
                    .claims
                    .iter()
                    .map(|claim| {
                        let row_point = tv_fp2s(&claim.row_point);
                        tv::Claim {
                            value: rows
                                .iter()
                                .zip(&by_row)
                                .fold(TvFp2::ZERO, |sum, (&r, &v)| sum + eq_at(&row_point, r) * v),
                            row_point,
                            gamma: tv_fp2(claim.gamma),
                        }
                    })
                    .collect(),
            }
        })
        .collect()
}

/// `check_evaluations` then `check_columns` on the commitment's opening and answer.
fn verify(
    commitment: &ReferenceCommitment,
    groups: &[tv::Group],
    xi_point: &[TvFp2],
    combined: &[tv::CombinedRow],
    queries: &[u64],
) -> Result<(), tv::OpeningError> {
    let params = commitment.params();
    let opening = tv::Opening {
        params: tv::Params {
            log2_k: params.log2_k,
            log2_inv_rate: 1,
            mask_coeffs: u32::try_from(params.mask_coeffs).expect("small"),
            row_block: u32::try_from(params.row_block.get()).expect("small"),
            commitment_id: params.commitment_id,
        },
        layout: tv::Layout {
            row_vars: commitment.layout().row_vars(),
            segments: commitment.layout().segments(),
        },
        root: commitment.root(),
        groups,
        xi_point,
    };
    let answer = commitment.answer(queries).expect("valid queries");
    let columns: Vec<Vec<TvFp>> = answer
        .columns
        .iter()
        .map(|column| column.iter().copied().map(tv_fp).collect())
        .collect();
    tv::check_evaluations(opening, combined)?.check_columns(
        queries,
        &columns,
        &answer.proof.siblings,
    )
}

#[test]
fn a_corrupted_row_is_rejected_at_the_rate_of_its_distance() {
    const TRIALS: usize = 10_000;
    let mut rng = SplitMix64(0x7E55_E7A0_0054_0301);
    // k = 32, n = 64, P_A = 4, B = 2; non-zero rows 0, 1, 3, 4, 6 of 8 in three segments, the
    // middle one empty.
    let params = CommitmentParams {
        log2_k: 5,
        mask_coeffs: 4,
        row_block: NonZeroUsize::new(2).expect("non-zero"),
        commitment_id: [0x54; 32],
    };
    let shape = params.shape().expect("valid parameters");
    let (k, n) = (shape.k, shape.n);
    let layout = RowLayout::new(3, vec![vec![0, 1, 3], vec![], vec![4, 6]]).expect("valid");
    let rows: Vec<u64> = layout.row_indices().collect();
    let values = rng.fps(rows.len() * k);
    let masks = rng.fps(rows.len() * shape.mask_coeffs);
    let mut codewords = vec![Fp::ZERO; rows.len() * n];
    let mut drawn = masks.iter().copied();
    RowEncoder::new(k, shape.mask_coeffs)
        .expect("valid parameters")
        .encode_rows(
            &values,
            &mut || drawn.next().expect("P_A per row"),
            &mut codewords,
        )
        .expect("matching lengths");

    // Group 0 weighs every row; group 1's Boolean row point selects row 0 alone, so it does not
    // see the corrupted row; the proximity row sees every row.
    let groups = vec![
        ClaimGroup {
            column_point: rng.fp2s(5),
            claims: (0..2)
                .map(|_| RowClaim {
                    row_point: rng.fp2s(3),
                    gamma: rng.fp2(),
                })
                .collect(),
        },
        ClaimGroup {
            column_point: rng.fp2s(5),
            claims: vec![RowClaim {
                row_point: vec![Fp2::ZERO; 3],
                gamma: rng.fp2(),
            }],
        },
    ];
    let xi = rng.fp2s(3);
    let tv_groups = claimed(&groups, &rows, &values, k);
    let tv_xi = tv_fp2s(&xi);
    let tv_combined = |commitment: &ReferenceCommitment| -> Vec<tv::CombinedRow> {
        commitment
            .open(&groups, &xi)
            .expect("valid points")
            .iter()
            .map(|row| tv::CombinedRow {
                u: tv_fp2s(&row.u),
                mask: tv_fp2s(&row.mask),
            })
            .collect()
    };

    // The honest prover: the same rows and masks, committed by `commit`, verify.
    let mut replay = masks.iter().copied();
    let honest = ReferenceCommitment::commit(
        params,
        layout.clone(),
        Secret::new(values.clone()),
        &mut || replay.next().expect("P_A per row"),
    )
    .expect("valid inputs");
    let encoded = ReferenceCommitment::from_encoded(
        params,
        shape,
        layout.clone(),
        Secret::new(values.clone()),
        masks.iter().copied().collect::<SecretBuf<Fp>>(),
        codewords.iter().copied().collect::<SecretBuf<Fp>>(),
    );
    assert_eq!(honest.root(), encoded.root());
    let combined = tv_combined(&honest);
    for _ in 0..100 {
        let queries: Vec<u64> = (0..8).map(|_| rng.next_u64() % n as u64).collect();
        assert_eq!(
            verify(&honest, &tv_groups, &tv_xi, &combined, &queries),
            Ok(())
        );
    }

    // Row 4, the fourth non-zero row, corrupted at ⌈δ·n⌉ random columns.
    let target = 3;
    for delta in [0.05, 0.25, 0.5] {
        let corrupted_count = (delta * n as f64).ceil() as usize;
        let mut corrupted_columns: Vec<u64> = Vec::new();
        while corrupted_columns.len() < corrupted_count {
            let j = rng.next_u64() % n as u64;
            if !corrupted_columns.contains(&j) {
                corrupted_columns.push(j);
            }
        }
        let mut corrupted = codewords.clone();
        for &j in &corrupted_columns {
            let shift = loop {
                let x = rng.fp();
                if x != Fp::ZERO {
                    break x;
                }
            };
            corrupted[target * n + j as usize] += shift;
        }
        let cheating = ReferenceCommitment::from_encoded(
            params,
            shape,
            layout.clone(),
            Secret::new(values.clone()),
            masks.iter().copied().collect::<SecretBuf<Fp>>(),
            corrupted.into_iter().collect::<SecretBuf<Fp>>(),
        );
        assert_ne!(cheating.root(), honest.root());
        // The honest combined rows, from the row's message.
        let combined = tv_combined(&cheating);
        assert!(combined == tv_combined(&honest));

        let fraction = corrupted_count as f64 / n as f64;
        for tau in [4u32, 8] {
            let mut rejections = 0;
            for _ in 0..TRIALS {
                let queries: Vec<u64> = (0..tau).map(|_| rng.next_u64() % n as u64).collect();
                let result = verify(&cheating, &tv_groups, &tv_xi, &combined, &queries);
                let hit = queries
                    .iter()
                    .copied()
                    .filter(|j| corrupted_columns.contains(j))
                    .min();
                match hit {
                    Some(j) => {
                        assert_eq!(
                            result,
                            Err(tv::OpeningError::Proximity {
                                column: u32::try_from(j).expect("j < n")
                            }),
                            "δ = {delta}, τ = {tau}, queries {queries:?}"
                        );
                        rejections += 1;
                    }
                    None => assert_eq!(result, Ok(()), "δ = {delta}, τ = {tau}"),
                }
            }
            let bound = 1.0 - (1.0 - fraction).powi(tau as i32);
            let expected = TRIALS as f64 * bound;
            let margin = 4.0 * (TRIALS as f64 * bound * (1.0 - bound)).sqrt();
            assert!(
                rejections as f64 >= expected - margin,
                "δ = {delta} (δ' = {fraction}), τ = {tau}: {rejections} rejections of {TRIALS}, \
                 below {expected:.1} − {margin:.1}"
            );
        }
    }
}
