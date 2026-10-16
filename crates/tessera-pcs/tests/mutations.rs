// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Honest openings changed in one place, each rejected by the independent verifier,
//! `tv_core::opening`, with the reason RFC-0011's check order gives (steps 6 and 8, 'Failure
//! modes'). `j_0` is the smallest queried column.
//!
//! - Every entry `u_g[i]` of a group's combined row, changed by 1 and by `X`, moves the group's
//!   opened sum by that change times `eq(column_point_g, i)`. It fails step 6,
//!   `Evaluation { group: g }`, unless that weight is zero. Otherwise, and for every entry of
//!   the proximity row, which step 6 does not read, `enc(u)` changes at every point outside
//!   `H`, so step 8.3 fails: `Proximity` at `j_0`. A Boolean coordinate in one group's column
//!   point makes some weights zero, so both outcomes occur.
//! - Every coefficient of every combined mask, changed by 1 and by `X`, changes
//!   `Z_H(x_j)·M(x_j)` at every queried `x_j`: `Proximity` at `j_0`.
//! - Every symbol of every opened column, and every sibling, changes a leaf or the root:
//!   `MerklePath` at `j_0`, since a multiproof fails as a whole.
//! - The answers RFC-0011 'Testing Strategy' sets against the honest one for a draw that
//!   repeats an index and is not increasing, and the other answers of the wrong shape, are
//!   `MerklePath` at `j_0`.

mod common;

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use common::SplitMix64;
use common::openings::{
    Matrix, TvFp, TvFp2, Verifier, eq_at, tv_columns, tv_combined, tv_fp2s, tv_groups,
};
use tessera_field::{Field, GoldilocksExt2, QuadraticExtension};
use tessera_hash::Digest;
use tessera_pcs::{ClaimGroup, CommitmentParams, ReferenceCommitment, RowClaim, RowLayout};
use tv_core::opening::{CombinedRow, Group, OpeningError};

type Fp2 = GoldilocksExt2;

/// An honest opening, as the verifier receives it.
struct Opened {
    commitment: ReferenceCommitment,
    groups: Vec<Group>,
    xi_point: Vec<TvFp2>,
    combined: Vec<CombinedRow>,
    queries: Vec<u64>,
    columns: Vec<Vec<TvFp>>,
    siblings: Vec<Digest>,
}

impl Opened {
    /// Commits to random rows on `layout` and opens `claims[g]` claims in group `g`; group 1's
    /// column point has the coordinate 1 first, so that `eq(column_point_1, i) = 0` at every
    /// even `i`.
    fn new(
        seed: u64,
        (log2_k, mask_coeffs, row_block): (u32, usize, usize),
        layout: RowLayout,
        claims: &[usize],
        queries: Vec<u64>,
    ) -> Self {
        let mut rng = SplitMix64::new(seed);
        let fp2 = |rng: &mut SplitMix64| {
            let c0 = rng.fp();
            Fp2::new(c0, rng.fp())
        };
        let row_vars = layout.row_vars() as usize;
        let values = rng.fps(layout.rows() << log2_k);
        let groups: Vec<ClaimGroup> = claims
            .iter()
            .enumerate()
            .map(|(g, &count)| {
                let mut column_point: Vec<Fp2> = (0..log2_k).map(|_| fp2(&mut rng)).collect();
                if g == 1 && log2_k > 0 {
                    column_point[0] = Fp2::ONE;
                }
                ClaimGroup {
                    column_point,
                    claims: (0..count)
                        .map(|_| RowClaim {
                            row_point: (0..row_vars).map(|_| fp2(&mut rng)).collect(),
                            gamma: fp2(&mut rng),
                        })
                        .collect(),
                }
            })
            .collect();
        let xi_point: Vec<Fp2> = (0..row_vars).map(|_| fp2(&mut rng)).collect();
        let params = CommitmentParams {
            log2_k,
            mask_coeffs,
            row_block: NonZeroUsize::new(row_block).expect("non-zero"),
            commitment_id: [seed as u8; 32],
        };
        let mut masks = SplitMix64::new(seed ^ 0xFFFF);
        let commitment =
            ReferenceCommitment::commit(params, layout, values.clone(), &mut || masks.fp())
                .expect("valid inputs");
        let combined = commitment.open(&groups, &xi_point).expect("valid points");
        let answer = commitment.answer(&queries).expect("valid queries");
        let matrix = Matrix {
            log2_k,
            rows: commitment.layout().row_indices().collect(),
            values: &values,
        };
        Self {
            groups: tv_groups(&groups, &matrix.values(&groups)),
            xi_point: tv_fp2s(&xi_point),
            combined: tv_combined(&combined),
            columns: tv_columns(&answer),
            siblings: answer.proof.siblings,
            commitment,
            queries,
        }
    }

    fn verify(
        &self,
        combined: &[CombinedRow],
        columns: &[Vec<TvFp>],
        siblings: &[Digest],
    ) -> Result<(), OpeningError> {
        let verifier = Verifier {
            params: self.commitment.params(),
            layout: self.commitment.layout(),
            root: self.commitment.root(),
            groups: &self.groups,
            xi_point: &self.xi_point,
        };
        verifier.verify(combined, &self.queries, columns, siblings)
    }

    /// `j_0`, the smallest queried column.
    fn smallest_query(&self) -> u32 {
        let smallest = self.queries.iter().copied().min().expect("queries");
        u32::try_from(smallest).expect("a column is below 2^32")
    }

    /// The distinct queried columns, in increasing order.
    fn distinct(&self) -> Vec<u64> {
        let set: BTreeSet<u64> = self.queries.iter().copied().collect();
        set.into_iter().collect()
    }
}

/// Three openings: the first has four segments, the second empty, zero rows inside, between and
/// after them, and three groups; the second `P_A = k` and one segment after a zero row; the
/// third a single row.
fn openings() -> Vec<Opened> {
    let layout = |row_vars, segments| RowLayout::new(row_vars, segments).expect("valid");
    vec![
        Opened::new(
            0x7E55_E7A0_0054_0201,
            (4, 5, 2),
            layout(4, vec![vec![0, 2, 3], vec![], vec![5, 6, 7, 9], vec![12]]),
            &[2, 1, 3],
            vec![9, 3, 9, 30, 3, 17],
        ),
        Opened::new(
            0x7E55_E7A0_0054_0202,
            (3, 8, 32),
            layout(2, vec![vec![1, 3]]),
            &[1],
            vec![15, 0],
        ),
        Opened::new(
            0x7E55_E7A0_0054_0203,
            (2, 1, 1),
            layout(0, vec![vec![0]]),
            &[1, 2],
            vec![7, 7],
        ),
    ]
}

/// The changes made to each element of the combined rows: 1 and `X`.
fn changes() -> [TvFp2; 2] {
    [TvFp2::ONE, TvFp2::new(TvFp::ZERO, TvFp::ONE)]
}

#[test]
fn every_changed_entry_of_a_combined_row_or_mask_fails_the_first_check_that_reads_it() {
    let (mut evaluation, mut proximity) = (0, 0);
    for opened in openings() {
        assert_eq!(
            opened.verify(&opened.combined, &opened.columns, &opened.siblings),
            Ok(())
        );
        let at_smallest = OpeningError::Proximity {
            column: opened.smallest_query(),
        };
        for (g, row) in opened.combined.iter().enumerate() {
            for i in 0..row.u.len() {
                // Group g's opened sum moves by the change times eq(column_point_g, i); the
                // proximity row is not in the evaluation check.
                let weight = opened
                    .groups
                    .get(g)
                    .map_or(TvFp2::ZERO, |group| eq_at(&group.column_point, i as u64));
                let expected = if weight == TvFp2::ZERO {
                    proximity += 1;
                    at_smallest
                } else {
                    evaluation += 1;
                    OpeningError::Evaluation { group: Some(g) }
                };
                for change in changes() {
                    let mut combined = opened.combined.clone();
                    combined[g].u[i] += change;
                    assert_eq!(
                        opened.verify(&combined, &opened.columns, &opened.siblings),
                        Err(expected),
                        "u[{i}] of combined row {g}"
                    );
                }
            }
            for i in 0..row.mask.len() {
                for change in changes() {
                    let mut combined = opened.combined.clone();
                    combined[g].mask[i] += change;
                    assert_eq!(
                        opened.verify(&combined, &opened.columns, &opened.siblings),
                        Err(at_smallest),
                        "mask coefficient {i} of combined row {g}"
                    );
                }
            }
        }
    }
    // Both outcomes of a changed group entry occur.
    assert!(evaluation > 0 && proximity > 0);
}

#[test]
fn every_changed_column_symbol_or_sibling_fails_the_multiproof() {
    for opened in openings() {
        let expected = Err(OpeningError::MerklePath {
            column: opened.smallest_query(),
        });
        for d in 0..opened.columns.len() {
            for r in 0..opened.columns[d].len() {
                let mut columns = opened.columns.clone();
                columns[d][r] += TvFp::ONE;
                assert_eq!(
                    opened.verify(&opened.combined, &columns, &opened.siblings),
                    expected,
                    "symbol {r} of column {d}"
                );
            }
        }
        assert!(!opened.siblings.is_empty());
        for s in 0..opened.siblings.len() {
            let mut siblings = opened.siblings.clone();
            siblings[s][s % 32] ^= 1 << (s % 8);
            assert_eq!(
                opened.verify(&opened.combined, &opened.columns, &siblings),
                expected,
                "sibling {s}"
            );
        }
    }
}

#[test]
fn other_answers_to_a_repeated_unordered_draw_fail_the_multiproof() {
    let opened = &openings()[0];
    // The draw 9, 3, 9, 30, 3, 17 is answered with columns 3, 9, 17 and 30, which verify.
    let distinct = opened.distinct();
    assert_eq!(distinct, [3, 9, 17, 30]);
    let column =
        |j: u64| opened.columns[distinct.iter().position(|&d| d == j).expect("queried")].clone();
    let expected = Err(OpeningError::MerklePath {
        column: opened.smallest_query(),
    });
    let verify = |columns: &[Vec<TvFp>], siblings: &[Digest]| {
        opened.verify(&opened.combined, columns, siblings)
    };
    // The distinct columns in draw order.
    let draw_order: Vec<Vec<TvFp>> = [9, 3, 30, 17].into_iter().map(column).collect();
    assert_eq!(verify(&draw_order, &opened.siblings), expected);
    // One column per draw, in draw order: the repeated ones sent twice.
    let per_draw: Vec<Vec<TvFp>> = opened.queries.iter().map(|&j| column(j)).collect();
    assert_eq!(verify(&per_draw, &opened.siblings), expected);
    // The repeated columns twice, next to themselves.
    let doubled: Vec<Vec<TvFp>> = [3, 3, 9, 9, 17, 30].into_iter().map(column).collect();
    assert_eq!(verify(&doubled, &opened.siblings), expected);
    // A column missing, or an extra one: committed column 2, unqueried, in increasing order.
    assert_eq!(verify(&opened.columns[..3], &opened.siblings), expected);
    let extra = opened
        .commitment
        .answer(&[2, 3, 9, 17, 30])
        .expect("valid queries");
    assert_eq!(verify(&tv_columns(&extra), &opened.siblings), expected);
    // A column one symbol short or long.
    let mut short = opened.columns.clone();
    short[1].pop();
    assert_eq!(verify(&short, &opened.siblings), expected);
    let mut long = opened.columns.clone();
    long[1].push(TvFp::ONE);
    assert_eq!(verify(&long, &opened.siblings), expected);
    // A sibling missing, an extra one, or two swapped.
    let siblings = &opened.siblings;
    assert_eq!(
        verify(&opened.columns, &siblings[..siblings.len() - 1]),
        expected
    );
    let mut extra_sibling = siblings.clone();
    extra_sibling.push([0x5A; 32]);
    assert_eq!(verify(&opened.columns, &extra_sibling), expected);
    let mut swapped = siblings.clone();
    swapped.swap(0, 1);
    assert_ne!(swapped, *siblings);
    assert_eq!(verify(&opened.columns, &swapped), expected);
    // The honest answer, for contrast.
    assert_eq!(verify(&opened.columns, siblings), Ok(()));
}
