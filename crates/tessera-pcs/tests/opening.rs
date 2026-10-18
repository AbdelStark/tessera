// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Openings of random matrices, checked by the independent verifier `tv_core::opening`
//! (`check_evaluations`, then `Evaluated::check_columns`), whose claims' values are computed here
//! from the matrix in `tv-field`'s arithmetic (RFC-0011 'Testing Strategy', DEP-2).
//!
//! The cases cover 1 to 40 claims in 1 to 16 groups, a group with no claim and openings with no
//! group; one to four segments, empty ones included; zero rows before, between, inside and
//! after the segments; `B` of 1, 2, 5 and 32; `P_A` of 1, of `k` and in between; draws that
//! repeat columns and are not increasing; and `k` from `2^0` to `2^10`.
//!
//! The combined rows and the row coefficients are also compared, value by value, with their
//! definitions computed here: `c_g(row) = Σ γ·eq(row_point, row)` from `eq`'s product formula,
//! and `u_g` and `M_g` as sums over the rows.

mod common;

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use common::openings::{
    Matrix, TvFp2, Verifier, eq_at, tv_columns, tv_combined, tv_fp, tv_fp2, tv_fp2s, tv_groups,
};
use common::{Fp, SplitMix64};
use tessera_codec::Secret;
use tessera_field::{GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_pcs::{ClaimGroup, CommitmentParams, ReferenceCommitment, RowClaim, RowLayout};

type Fp2 = GoldilocksExt2;

fn fp2(rng: &mut SplitMix64) -> Fp2 {
    let c0 = rng.fp();
    Fp2::new(c0, rng.fp())
}

/// A coordinate: uniform, or Boolean one time in `boolean_one_in`, so that some coefficients
/// vanish on whole sets of rows.
fn coordinate(rng: &mut SplitMix64, boolean_one_in: usize) -> Fp2 {
    if boolean_one_in > 0 && rng.below(boolean_one_in) == 0 {
        Fp2::from(Fp::from_u64_reduced(rng.below(2) as u64))
    } else {
        fp2(rng)
    }
}

fn point(rng: &mut SplitMix64, len: usize, boolean_one_in: usize) -> Vec<Fp2> {
    (0..len).map(|_| coordinate(rng, boolean_one_in)).collect()
}

/// Rows of `2^row_vars`, each non-zero with probability 2/3, cut into `segments` contiguous
/// segments at random places, so that segments may be empty and zero rows fall anywhere.
fn layout(rng: &mut SplitMix64, row_vars: u32, segments: usize) -> RowLayout {
    let rows: Vec<u64> = (0..1u64 << row_vars)
        .filter(|_| rng.below(3) != 0)
        .collect();
    let mut cuts: Vec<usize> = (1..segments).map(|_| rng.below(rows.len() + 1)).collect();
    cuts.sort_unstable();
    let mut parts = Vec::with_capacity(segments);
    let mut start = 0;
    for cut in cuts.into_iter().chain([rows.len()]) {
        parts.push(rows[start..cut].to_vec());
        start = cut;
    }
    RowLayout::new(row_vars, parts).expect("a valid layout")
}

/// `claims` claims in `groups` groups, every group with at least one claim when
/// `claims ≥ groups`.
fn groups(
    rng: &mut SplitMix64,
    log2_k: u32,
    row_vars: u32,
    claims: usize,
    groups: usize,
) -> Vec<ClaimGroup> {
    let mut sizes = vec![0; groups];
    for c in 0..claims {
        let g = if c < groups { c } else { rng.below(groups) };
        sizes[g] += 1;
    }
    sizes
        .into_iter()
        .map(|size| ClaimGroup {
            column_point: point(rng, log2_k as usize, 6),
            claims: (0..size)
                .map(|_| RowClaim {
                    row_point: point(rng, row_vars as usize, 3),
                    gamma: fp2(rng),
                })
                .collect(),
        })
        .collect()
}

/// `tau` draws with replacement, in draw order. Every other case repeats one draw out of
/// order, so that the draws repeat and are not increasing.
fn draws(rng: &mut SplitMix64, n: usize, tau: usize) -> Vec<u64> {
    let mut queries: Vec<u64> = (0..tau).map(|_| rng.below(n) as u64).collect();
    if tau >= 2 && rng.below(2) == 0 {
        let repeated = queries[tau - 1];
        queries.insert(0, repeated);
    }
    queries
}

struct Case {
    params: CommitmentParams,
    layout: RowLayout,
    values: Vec<Fp>,
    groups: Vec<ClaimGroup>,
    xi_point: Vec<Fp2>,
    queries: Vec<u64>,
}

/// Commits, opens and answers, checks the answer's shape, and has the verifier judge it.
fn prove_and_verify(case: Case, rng: &mut SplitMix64) {
    let Case {
        params,
        layout,
        values,
        groups,
        xi_point,
        queries,
    } = case;
    let mask_coeffs = params.mask_coeffs;
    let mut masks = || rng.fp();
    let commitment =
        ReferenceCommitment::commit(params, layout, Secret::new(values.clone()), &mut masks)
            .expect("valid inputs");
    let layout = commitment.layout();
    let combined = commitment.open(&groups, &xi_point).expect("valid points");
    assert_eq!(combined.len(), groups.len() + 1);
    for row in &combined {
        assert_eq!(
            (row.u.len(), row.mask.len()),
            (commitment.message_len(), mask_coeffs)
        );
    }
    let answer = commitment.answer(&queries).expect("valid queries");
    let distinct: BTreeSet<u64> = queries.iter().copied().collect();
    assert_eq!(answer.columns.len(), distinct.len());
    assert!(
        answer
            .columns
            .iter()
            .all(|column| column.len() == layout.rows())
    );

    let matrix = Matrix {
        log2_k: params.log2_k,
        rows: layout.row_indices().collect(),
        values: &values,
    };
    let claimed = tv_groups(&groups, &matrix.values(&groups));
    let xi = tv_fp2s(&xi_point);
    let verifier = Verifier {
        params: &params,
        layout,
        root: commitment.root(),
        groups: &claimed,
        xi_point: &xi,
    };
    assert_eq!(
        verifier.verify(
            &tv_combined(&combined),
            &queries,
            &tv_columns(&answer),
            &answer.proof.siblings
        ),
        Ok(()),
        "k = 2^{}, P_A = {}, B = {}, layout {:?}, {} groups, queries {queries:?}",
        params.log2_k,
        params.mask_coeffs,
        params.row_block,
        layout,
        groups.len(),
    );
}

/// What the cases of `random_claims_at_one_to_forty_points_verify` must include between them.
#[derive(Default)]
struct Coverage {
    empty_segment: bool,
    zero_row: bool,
    no_row: bool,
    repeated_draw: bool,
    unordered_draw: bool,
    sixteen_groups: bool,
    k_two_to_the_ten: bool,
    mask_of_one: bool,
    mask_of_k: bool,
    block_of_one: bool,
    block_of_thirty_two: bool,
    several_segments: bool,
}

impl Coverage {
    fn note(&mut self, case: &Case) {
        let k = 1usize << case.params.log2_k;
        let segments = case.layout.segments();
        self.empty_segment |= segments.iter().any(Vec::is_empty);
        self.zero_row |= case.layout.rows() < 1 << case.layout.row_vars();
        self.no_row |= case.layout.rows() == 0;
        let distinct: BTreeSet<u64> = case.queries.iter().copied().collect();
        self.repeated_draw |= distinct.len() < case.queries.len();
        self.unordered_draw |= case.queries.windows(2).any(|pair| pair[0] > pair[1]);
        self.sixteen_groups |= case.groups.len() == 16;
        self.k_two_to_the_ten |= k == 1 << 10;
        self.mask_of_one |= case.params.mask_coeffs == 1 && k > 1;
        self.mask_of_k |= case.params.mask_coeffs == k && k > 1;
        self.block_of_one |= case.params.row_block.get() == 1;
        self.block_of_thirty_two |= case.params.row_block.get() == 32;
        self.several_segments |= segments.len() > 1;
    }

    /// The names of what no case included.
    fn missing(&self) -> Vec<&'static str> {
        [
            (self.empty_segment, "an empty segment"),
            (self.zero_row, "a zero row"),
            (self.no_row, "no non-zero row"),
            (self.repeated_draw, "a repeated draw"),
            (self.unordered_draw, "draws out of order"),
            (self.sixteen_groups, "16 groups"),
            (self.k_two_to_the_ten, "k = 2^10"),
            (self.mask_of_one, "P_A = 1 < k"),
            (self.mask_of_k, "P_A = k > 1"),
            (self.block_of_one, "B = 1"),
            (self.block_of_thirty_two, "B = 32"),
            (self.several_segments, "several segments"),
        ]
        .into_iter()
        .filter(|&(seen, _)| !seen)
        .map(|(_, what)| what)
        .collect()
    }
}

#[test]
fn random_claims_at_one_to_forty_points_verify() {
    let mut rng = SplitMix64::new(0x7E55_E7A0_0054_0101);
    let mut coverage = Coverage::default();
    for claims in 1..=40 {
        let log2_k = (claims % 11) as u32;
        let k = 1usize << log2_k;
        let mask_coeffs = match claims % 3 {
            0 => 1,
            1 => k,
            _ => 1 + rng.below(k),
        };
        let row_block = [1, 32, 2, 5][claims % 4];
        let row_vars = (claims % 7) as u32;
        let segments = 1 + claims % 4;
        let group_count = 1 + (claims - 1) % 16;
        // One matrix has no non-zero row at all: every claim on it is 0.
        let layout = if claims == 14 {
            RowLayout::new(row_vars, vec![Vec::new(); segments]).expect("valid")
        } else {
            layout(&mut rng, row_vars, segments)
        };
        let values = rng.fps(layout.rows() * k);
        let case = Case {
            params: CommitmentParams {
                log2_k,
                log2_inv_rate: 1,
                mask_coeffs,
                row_block: NonZeroUsize::new(row_block).expect("non-zero"),
                commitment_id: core::array::from_fn(|i| (claims * 31 + i) as u8),
            },
            groups: groups(&mut rng, log2_k, row_vars, claims, group_count),
            xi_point: point(&mut rng, row_vars as usize, 0),
            queries: draws(&mut rng, 2 * k, 1 + claims % 12),
            layout,
            values,
        };
        assert_eq!(
            case.groups.iter().map(|g| g.claims.len()).sum::<usize>(),
            claims
        );
        coverage.note(&case);
        prove_and_verify(case, &mut rng);
    }
    assert_eq!(coverage.missing(), Vec::<&str>::new(), "the cases cover");
}

#[test]
fn a_group_with_no_claim_and_an_opening_with_no_group_verify() {
    let mut rng = SplitMix64::new(0x7E55_E7A0_0054_0102);
    for group_counts in [vec![], vec![0], vec![2, 0, 1]] {
        let (log2_k, row_vars) = (4, 3);
        let layout = layout(&mut rng, row_vars, 2);
        let values = rng.fps(layout.rows() * 16);
        let mut claim_groups = Vec::new();
        for &size in &group_counts {
            claim_groups.extend(groups(&mut rng, log2_k, row_vars, size, 1));
        }
        let case = Case {
            params: CommitmentParams {
                log2_k,
                log2_inv_rate: 1,
                mask_coeffs: 3,
                row_block: NonZeroUsize::new(2).expect("non-zero"),
                commitment_id: [9; 32],
            },
            groups: claim_groups,
            xi_point: point(&mut rng, row_vars as usize, 0),
            queries: draws(&mut rng, 32, 5),
            layout,
            values,
        };
        prove_and_verify(case, &mut rng);
    }
}

#[test]
fn combined_rows_and_row_coefficients_are_their_definitions() {
    let mut rng = SplitMix64::new(0x7E55_E7A0_0054_0103);
    // P_A = 300 spans several tiles of 64 columns and ends inside one; 1, 2, 5 and 7 rows
    // cover each count of rows left over by four.
    for (log2_k, mask_coeffs, row_vars) in [(10, 300, 3), (6, 64, 1), (9, 5, 2), (0, 1, 4)] {
        let k = 1usize << log2_k;
        for segments in [1, 3] {
            let layout = layout(&mut rng, row_vars, segments);
            let values = rng.fps(layout.rows() * k);
            let claim_groups = groups(&mut rng, log2_k, row_vars, 7, 3);
            let xi_point = point(&mut rng, row_vars as usize, 0);
            let mut drawn = Vec::new();
            let mut masks = || {
                let x = rng.fp();
                drawn.push(x);
                x
            };
            let params = CommitmentParams {
                log2_k,
                log2_inv_rate: 1,
                mask_coeffs,
                row_block: NonZeroUsize::new(32).expect("non-zero"),
                commitment_id: [3; 32],
            };
            let commitment = ReferenceCommitment::commit(
                params,
                layout.clone(),
                Secret::new(values.clone()),
                &mut masks,
            )
            .expect("valid inputs");
            let rows: Vec<u64> = layout.row_indices().collect();
            assert_eq!(drawn.len(), rows.len() * mask_coeffs, "P_A draws per row");

            // c_g(row) and ξ_row from the product formula of eq.
            let mut expected: Vec<Vec<TvFp2>> = claim_groups
                .iter()
                .map(|group| {
                    rows.iter()
                        .map(|&row| {
                            group.claims.iter().fold(TvFp2::ZERO, |sum, claim| {
                                sum + tv_fp2(claim.gamma) * eq_at(&tv_fp2s(&claim.row_point), row)
                            })
                        })
                        .collect()
                })
                .collect();
            expected.push(
                rows.iter()
                    .map(|&row| eq_at(&tv_fp2s(&xi_point), row))
                    .collect(),
            );
            let coefficients = layout
                .row_coefficients(&claim_groups, &xi_point)
                .expect("valid points");
            let ours: Vec<Vec<TvFp2>> = coefficients.iter().map(|c| tv_fp2s(c)).collect();
            assert!(ours == expected, "row coefficients, k = 2^{log2_k}");

            // u = Σ_row c(row)·A_row and M = Σ_row c(row)·m_row.
            let combined = commitment.open(&claim_groups, &xi_point).expect("valid");
            for (row, c) in combined.iter().zip(&expected) {
                let u: Vec<TvFp2> = (0..k)
                    .map(|i| {
                        c.iter().enumerate().fold(TvFp2::ZERO, |sum, (r, &c)| {
                            sum + c * TvFp2::from(tv_fp(values[r * k + i]))
                        })
                    })
                    .collect();
                let mask: Vec<TvFp2> = (0..mask_coeffs)
                    .map(|i| {
                        c.iter().enumerate().fold(TvFp2::ZERO, |sum, (r, &c)| {
                            sum + c * TvFp2::from(tv_fp(drawn[r * mask_coeffs + i]))
                        })
                    })
                    .collect();
                assert!(
                    tv_fp2s(&row.u) == u,
                    "u, k = 2^{log2_k}, N = {}",
                    rows.len()
                );
                assert!(tv_fp2s(&row.mask) == mask, "M, P_A = {mask_coeffs}");
            }
        }
    }
}

#[test]
fn debug_forms_show_shapes_and_counts_only() {
    // Every row value and mask coefficient is a canary, which no Debug form may show.
    let canary = Fp::from_u64_reduced(0x0123_4567_89AB_CDEF);
    let params = CommitmentParams {
        log2_k: 4,
        log2_inv_rate: 1,
        mask_coeffs: 3,
        row_block: NonZeroUsize::new(2).expect("non-zero"),
        commitment_id: [5; 32],
    };
    let layout = RowLayout::new(3, vec![vec![0, 2], vec![5]]).expect("valid");
    let commitment = ReferenceCommitment::commit(
        params,
        layout,
        Secret::new(vec![canary; 3 * 16]),
        &mut || canary,
    )
    .expect("valid inputs");
    assert_eq!(
        format!("{commitment:?}"),
        "ReferenceCommitment { message_len: 16, codeword_len: 32, mask_coeffs: 3, row_block: 2, \
         row_vars: 3, segments: 2, rows: 3, .. }"
    );
    assert_eq!(
        (commitment.message_len(), commitment.codeword_len()),
        (16, 32)
    );
    assert_eq!(commitment.leaves().len(), 32);
    assert_eq!(*commitment.params(), params);

    let group = ClaimGroup {
        column_point: vec![Fp2::from(canary); 4],
        claims: vec![RowClaim {
            row_point: vec![Fp2::from(canary); 3],
            gamma: Fp2::from(canary),
        }],
    };
    let combined = commitment
        .open(&[group], &[Fp2::from(canary); 3])
        .expect("valid points");
    assert_eq!(
        format!("{:?}", combined[0]),
        "CombinedRow { values: 16, mask_coeffs: 3 }"
    );
    let answer = commitment.answer(&[7, 3, 7]).expect("valid queries");
    assert_eq!(
        format!("{answer:?}"),
        format!(
            "ColumnAnswer {{ columns: 2, symbols: 3, siblings: {} }}",
            answer.proof.siblings.len()
        )
    );
}

#[test]
fn every_leaf_and_the_root_are_the_verifiers_hashes_of_the_committed_columns() {
    let mut rng = SplitMix64::new(0x7E55_E7A0_0054_0104);
    let mut empty_segment = false;
    // Several segments, empty ones and zero rows, and chains of B = 1, 2 and 32 rows with
    // partial last blocks.
    for (log2_k, row_vars, segments, row_block) in [(3, 4, 3, 2), (2, 3, 4, 1), (4, 5, 2, 32)] {
        let k = 1usize << log2_k;
        let layout = layout(&mut rng, row_vars, segments);
        empty_segment |= layout.segments().iter().any(Vec::is_empty);
        let params = CommitmentParams {
            log2_k,
            log2_inv_rate: 1,
            mask_coeffs: k / 2,
            row_block: NonZeroUsize::new(row_block).expect("non-zero"),
            commitment_id: core::array::from_fn(|i| i as u8),
        };
        let values = rng.fps(layout.rows() * k);
        let commitment =
            ReferenceCommitment::commit(params, layout.clone(), Secret::new(values), &mut || {
                rng.fp()
            })
            .expect("valid inputs");
        let n = 2 * k;
        let all: Vec<u64> = (0..n as u64).collect();
        let answer = commitment.answer(&all).expect("valid queries");
        // An answer to every column needs no sibling.
        assert!(answer.proof.siblings.is_empty());
        let block = core::num::NonZeroU32::new(row_block as u32).expect("non-zero");
        let leaves: Vec<[u8; 32]> = tv_columns(&answer)
            .iter()
            .enumerate()
            .map(|(j, column)| {
                // d_j^(s) over each segment's symbols, then column-merge.
                let mut merged = Vec::new();
                let mut rest = column.as_slice();
                for segment in layout.segments() {
                    let (here, after) = rest.split_at(segment.len());
                    merged.extend_from_slice(&tv_hash::column::column_digest(
                        &params.commitment_id,
                        j as u64,
                        block,
                        here,
                    ));
                    rest = after;
                }
                tv_hash::hash(tv_hash::Domain::ColumnMerge, &merged)
            })
            .collect();
        assert_eq!(commitment.leaves(), leaves.as_slice());
        let mut level = leaves;
        while level.len() > 1 {
            level = level
                .chunks_exact(2)
                .map(|pair| tv_hash::merkle::node(&pair[0], &pair[1]))
                .collect();
        }
        assert_eq!(commitment.root(), level[0]);
    }
    assert!(empty_segment, "a segment is empty");
}
