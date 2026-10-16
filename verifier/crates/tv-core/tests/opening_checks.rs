// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The opening checks on seeded openings of the tests' reference prover: shapes the vectors do not
//! hold (inverse rates 4 and 8, a 64-bit row index, no group, no non-zero row, larger rows), each
//! check violated on its own, provers that lie in the combined rows or commit changed codewords,
//! and every possible draw of the queries against a matrix whose codewords were changed.
//!
//! Every verdict is also the tests' oracle's ([`reference::predict`]), which the vector tests
//! check against the independent Python oracle.

mod openings;

use std::collections::BTreeSet;

use openings::Case;
use openings::reference::{
    self, ClaimSpec, GroupSpec, Matrix, eq_at, random_fe, random_groups, random_layout, to_fe,
    to_fp2,
};
use tv_core::opening::{OpeningError, Params, check_evaluations};
use tv_field::{Fp, Fp2};
use tv_testkit::SplitMix64;
use tv_testkit::sumcheck::Fe;

/// Parameters with a seeded `commitment_id`.
fn params(
    rng: &mut SplitMix64,
    log2_k: u32,
    log2_inv_rate: u32,
    mask_coeffs: u32,
    row_block: u32,
) -> Params {
    let mut commitment_id = [0u8; 32];
    for chunk in commitment_id.chunks_mut(8) {
        chunk.copy_from_slice(&rng.next_u64().to_le_bytes());
    }
    Params {
        log2_k,
        log2_inv_rate,
        mask_coeffs,
        row_block,
        commitment_id,
    }
}

/// `count` seeded draws below `n`.
fn draws(rng: &mut SplitMix64, n: u64, count: usize) -> Vec<u64> {
    (0..count).map(|_| rng.next_u64() % n).collect()
}

/// A shape of opening: `(log2_k, log2_inv_rate, P_A, B, row_vars, N, S, claims per group, τ)`.
type Shape = (
    u32,
    u32,
    u32,
    u32,
    u32,
    usize,
    usize,
    &'static [usize],
    usize,
);

/// An honest opening of a seeded matrix of `shape`.
fn honest(rng: &mut SplitMix64, shape: Shape) -> (Matrix, Vec<GroupSpec>, Vec<Fe>, Case) {
    let (log2_k, log2_inv_rate, mask, block, row_vars, rows, segments, claims, tau) = shape;
    let params = params(rng, log2_k, log2_inv_rate, mask, block);
    let layout = random_layout(rng, row_vars, rows, segments);
    let matrix = Matrix::random(params, row_vars, layout, rng);
    let groups = random_groups(rng, log2_k, row_vars, claims);
    let xi: Vec<Fe> = (0..row_vars).map(|_| random_fe(rng)).collect();
    let queries = draws(rng, matrix.n(), tau);
    let case = matrix.open(&groups, &xi, &queries);
    (matrix, groups, xi, case)
}

/// `case`'s verdict, which must be the oracle's too.
fn verdict(case: &Case) -> Result<(), OpeningError> {
    let outcome = case.check();
    assert_eq!(outcome, reference::predict(case), "the oracle disagrees");
    outcome
}

const SHAPES: [Shape; 12] = [
    (0, 1, 1, 1, 0, 1, 1, &[1], 1),
    (0, 2, 1, 3, 3, 5, 2, &[2, 1], 4),
    (1, 1, 2, 1, 2, 3, 3, &[1, 0, 2], 3),
    (2, 1, 1, 2, 5, 9, 2, &[3], 6),
    (2, 3, 3, 32, 4, 7, 1, &[1, 2, 1], 5),
    (3, 2, 8, 2, 6, 12, 4, &[2, 2], 8),
    (4, 1, 5, 3, 9, 20, 3, &[4, 1, 3], 9),
    (4, 2, 16, 1, 7, 6, 2, &[], 3),
    (2, 1, 2, 1, 64, 4, 2, &[2], 4),
    (3, 1, 1, 4, 3, 0, 2, &[1], 3),
    (5, 1, 7, 5, 4, 10, 2, &[2, 3], 12),
    (1, 3, 1, 2, 1, 2, 5, &[1, 1, 1, 1], 7),
];

/// Honest openings of every shape are accepted: inverse rates 2, 4 and 8, `k` from 1 to 32, `P_A`
/// from 1 to `k`, `B` from 1 to 32, row indices of 0 to 64 bits, empty segments, groups without
/// claims and no group at all, no non-zero row, repeated draws, and 30 seeded shapes more.
#[test]
fn honest_openings_of_every_shape_are_accepted() {
    let mut rng = SplitMix64::new(0x6f70_656e_0061_0001);
    for shape in SHAPES {
        let (_, _, _, case) = honest(&mut rng, shape);
        assert_eq!(verdict(&case), Ok(()), "{shape:?}");
    }
    for _ in 0..30 {
        let log2_k = (rng.next_u64() % 5) as u32;
        let log2_inv_rate = 1 + (rng.next_u64() % 3) as u32;
        let k = 1u64 << log2_k;
        let mask = 1 + (rng.next_u64() % k) as u32;
        let block = [1, 2, 3, 5, 32][(rng.next_u64() % 5) as usize];
        let row_vars = [0, 1, 3, 6, 10, 64][(rng.next_u64() % 6) as usize];
        let space = if row_vars >= 6 {
            16
        } else {
            1usize << row_vars
        };
        let rows = (rng.next_u64() % (space as u64 + 1)) as usize;
        let segments = 1 + (rng.next_u64() % 4) as usize;
        let claims: &'static [usize] =
            [&[][..], &[1], &[2, 0], &[1, 3, 2]][(rng.next_u64() % 4) as usize];
        let tau = 1 + (rng.next_u64() % 10) as usize;
        let shape = (
            log2_k,
            log2_inv_rate,
            mask,
            block,
            row_vars,
            rows,
            segments,
            claims,
            tau,
        );
        let (_, _, _, case) = honest(&mut rng, shape);
        assert_eq!(verdict(&case), Ok(()), "{shape:?}");
    }
}

/// The evaluation check reports the first group whose sums differ, and a shape that is wrong
/// before any group.
#[test]
fn the_first_failing_group_is_reported_and_the_shape_comes_first() {
    let mut rng = SplitMix64::new(0x6f70_656e_0061_0002);
    let (_, _, _, case) = honest(&mut rng, (3, 1, 2, 2, 4, 9, 2, &[2, 1, 3], 5));
    for (first, second) in [(0, 2), (1, 2), (0, 1)] {
        let mut changed = case.clone();
        changed.groups[first].claims[0].value += Fp2::ONE;
        changed.groups[second].claims[0].value += Fp2::ONE;
        assert_eq!(
            verdict(&changed),
            Err(OpeningError::Evaluation { group: Some(first) })
        );
        // The proximity row dropped as well: the shape is checked before any group.
        changed.combined.pop();
        assert_eq!(
            verdict(&changed),
            Err(OpeningError::Evaluation { group: None })
        );
    }
    // A group's row too short and another's mask too long, in either order.
    let mut changed = case.clone();
    changed.combined[2].u.pop();
    changed.combined[0].mask.push(Fp2::ZERO);
    assert_eq!(
        verdict(&changed),
        Err(OpeningError::Evaluation { group: None })
    );
}

/// The leaves and the multiproof are checked before consistency: a column whose symbol changed
/// is `MerklePath` even when a combined row also disagrees with every column.
#[test]
fn the_columns_are_authenticated_before_they_are_compared() {
    let mut rng = SplitMix64::new(0x6f70_656e_0061_0003);
    let (_, _, _, case) = honest(&mut rng, (2, 2, 3, 2, 3, 6, 2, &[1, 2], 6));
    let smallest = case.smallest();
    let mut changed = case.clone();
    let last = changed.combined.len() - 1;
    changed.combined[last].u[0] += Fp2::ONE;
    assert_eq!(
        verdict(&changed),
        Err(OpeningError::Proximity { column: smallest })
    );
    let column = changed.columns.len() - 1;
    changed.columns[column][0] += Fp::ONE;
    assert_eq!(
        verdict(&changed),
        Err(OpeningError::MerklePath { column: smallest })
    );
}

/// A prover that claims a false value and changes its group's combined row so that the
/// evaluation check passes is rejected by the columns: the changed row is no longer the
/// combination of the committed rows.
#[test]
fn a_false_claim_with_a_combined_row_to_match_fails_the_column_check() {
    let mut rng = SplitMix64::new(0x6f70_656e_0061_0004);
    for shape in SHAPES.iter().copied().filter(|shape| !shape.7.is_empty()) {
        let (_, _, _, case) = honest(&mut rng, shape);
        let Some((g, group)) = case
            .groups
            .iter()
            .enumerate()
            .find(|(_, g)| !g.claims.is_empty())
        else {
            continue;
        };
        let lie = to_fe(Fp2::new(
            Fp::reduce(rng.next_u64()),
            Fp::reduce(rng.next_u64()),
        ));
        let gamma = to_fe(group.claims[0].gamma);
        let point: Vec<Fe> = group.column_point.iter().copied().map(to_fe).collect();
        // Spread γ·lie over u_g: a random vector w, its last entry fixed so that
        // ⟨w, eq(column point)⟩ = γ·lie.
        let k = case.k();
        let mut w: Vec<Fe> = (0..k).map(|_| random_fe(&mut rng)).collect();
        let last_factor = eq_at(&point, (k - 1) as u64);
        let partial = w[..k - 1]
            .iter()
            .enumerate()
            .fold(Fe::ZERO, |s, (i, &x)| s + x * eq_at(&point, i as u64));
        w[k - 1] = (gamma * lie - partial) * last_factor.inverse().expect("a random point");

        let mut changed = case.clone();
        changed.groups[g].claims[0].value += to_fp2(lie);
        for (entry, &x) in changed.combined[g].u.iter_mut().zip(&w) {
            *entry += to_fp2(x);
        }
        // The evaluation check passes; the columns reject it.
        let evaluated = check_evaluations(changed.opening(), &changed.combined);
        assert!(evaluated.is_ok(), "{shape:?}");
        let outcome = verdict(&changed);
        assert!(
            matches!(outcome, Err(OpeningError::Proximity { .. })),
            "{shape:?}: {outcome:?}"
        );
    }
}

/// With a Boolean column point, `eq(column point, i)` is zero at every entry but one, so a change
/// there passes the evaluation check; the column check sees it at every column, `enc(e_i)`
/// vanishing only on `H`.
#[test]
fn a_change_the_evaluation_check_cannot_see_is_rejected_by_the_columns() {
    let mut rng = SplitMix64::new(0x6f70_656e_0061_0005);
    let params = params(&mut rng, 3, 1, 2, 2);
    let layout = random_layout(&mut rng, 3, 5, 2);
    let matrix = Matrix::random(params, 3, layout, &mut rng);
    let mut groups = random_groups(&mut rng, 3, 3, &[2, 1]);
    // Column point (1, 0, 1): only entry 5 counts.
    groups[1].column_point = vec![Fe::ONE, Fe::ZERO, Fe::ONE];
    let xi: Vec<Fe> = (0..3).map(|_| random_fe(&mut rng)).collect();
    let queries = draws(&mut rng, matrix.n(), 6);
    let case = matrix.open(&groups, &xi, &queries);
    assert_eq!(verdict(&case), Ok(()));
    for i in 0..case.k() {
        let mut changed = case.clone();
        changed.combined[1].u[i] += Fp2::ONE;
        let expected = if i == 5 {
            OpeningError::Evaluation { group: Some(1) }
        } else {
            OpeningError::Proximity {
                column: case.smallest(),
            }
        };
        assert_eq!(verdict(&changed), Err(expected), "entry {i}");
    }
}

/// Every draw of `τ` queries, with replacement, against a matrix two of whose rows' codewords
/// were changed at a few columns and committed: the opening is accepted exactly when no draw
/// lands on a changed column, and otherwise rejected with `Proximity` at the smallest changed
/// column drawn. The honest combined rows agree with the committed columns everywhere else.
#[test]
fn every_draw_is_judged_by_the_changed_columns_it_lands_on() {
    let mut rng = SplitMix64::new(0x6f70_656e_0061_0006);
    for (log2_k, log2_inv_rate, tau, changes) in [
        (2, 1, 3, &[(0, 5), (2, 1), (2, 6)][..]),
        (1, 2, 3, &[(1, 7), (3, 2)]),
        (3, 1, 2, &[(0, 0), (1, 15), (4, 9)]),
    ] {
        let params = params(
            &mut rng,
            log2_k,
            log2_inv_rate,
            1 << log2_k.saturating_sub(1),
            2,
        );
        let layout = random_layout(&mut rng, 3, 5, 2);
        let mut matrix = Matrix::random(params, 3, layout, &mut rng);
        let changed: BTreeSet<u64> = changes.iter().map(|&(_, j)| j).collect();
        for &(row, j) in changes {
            matrix.codewords[row][j as usize] = matrix.codewords[row][j as usize] + Fe::ONE;
        }
        let groups = random_groups(&mut rng, log2_k, 3, &[1, 2]);
        let xi: Vec<Fe> = (0..3).map(|_| random_fe(&mut rng)).collect();
        let base = matrix.open(&groups, &xi, &[0]);
        let levels = matrix.tree();
        let evaluated = check_evaluations(base.opening(), &base.combined).expect("honest rows");

        let n = matrix.n();
        let mut accepted = 0u64;
        let total = n.pow(tau);
        for draw in 0..total {
            let queries: Vec<u64> = (0..tau).map(|t| draw / n.pow(t) % n).collect();
            let distinct: BTreeSet<u64> = queries.iter().copied().collect();
            let distinct: Vec<u64> = distinct.into_iter().collect();
            let columns: Vec<Vec<Fp>> = distinct
                .iter()
                .map(|&j| matrix.column(j).into_iter().map(reference::to_fp).collect())
                .collect();
            let siblings = reference::multiproof(&levels, &distinct);
            let expected = match distinct.iter().find(|j| changed.contains(j)) {
                Some(&j) => Err(OpeningError::Proximity { column: j as u32 }),
                None => Ok(()),
            };
            let outcome = evaluated
                .clone()
                .check_columns(&queries, &columns, &siblings);
            assert_eq!(outcome, expected, "draw {queries:?}");
            accepted += u64::from(outcome.is_ok());
        }
        // (n − |changed|)^τ of the n^τ draws miss every changed column.
        assert_eq!(accepted, (n - changed.len() as u64).pow(tau));
    }
}

/// The reported column is the smallest failing one over all rows: a row that fails only at a
/// larger column does not hide another row's failure at a smaller one, whichever comes first.
/// Group 0 weighs rows 0 and 1 only, group 1 and `ξ` rows 2 and 3 only, through Boolean
/// coordinates of their points. The distinct draws are 1, 3 and 6. A failure at 1, the first, ends
/// the check at once, so the cases that tell the reported column apart fail first at 3: in group
/// 0's row, with group 1's and `ξ`'s failing only at 6 after it, or in theirs, after group 0's
/// failing only at 6.
#[test]
fn the_reported_column_is_the_smallest_failing_one_over_all_rows() {
    let mut rng = SplitMix64::new(0x6f70_656e_0061_0007);
    let params = params(&mut rng, 2, 1, 2, 1);
    let mut base = Matrix::random(params, 2, vec![vec![0, 1], vec![2, 3]], &mut rng);
    let high = |bit: Fe, rng: &mut SplitMix64| vec![random_fe(rng), bit];
    let groups = vec![
        GroupSpec {
            column_point: vec![random_fe(&mut rng), random_fe(&mut rng)],
            claims: vec![ClaimSpec {
                row_point: high(Fe::ZERO, &mut rng),
                gamma: random_fe(&mut rng),
            }],
        },
        GroupSpec {
            column_point: vec![random_fe(&mut rng), random_fe(&mut rng)],
            claims: vec![ClaimSpec {
                row_point: high(Fe::ONE, &mut rng),
                gamma: random_fe(&mut rng),
            }],
        },
    ];
    let xi = high(Fe::ONE, &mut rng);
    let queries = [6, 1, 3, 6];
    assert_eq!(verdict(&base.open(&groups, &xi, &queries)), Ok(()));
    // (row changed in group 0's support, its column; row in group 1's and ξ's, its column).
    for (small, large, expected) in [
        ((0, 1), (2, 6), 1),
        ((0, 3), (2, 6), 3),
        ((1, 3), (3, 6), 3),
        ((0, 6), (3, 3), 3),
        ((1, 3), (2, 3), 3),
    ] {
        let mut matrix = base.clone();
        for (row, j) in [small, large] {
            matrix.codewords[row][j] = matrix.codewords[row][j] + Fe::ONE;
        }
        let case = matrix.open(&groups, &xi, &queries);
        assert_eq!(
            verdict(&case),
            Err(OpeningError::Proximity { column: expected }),
            "{small:?}, {large:?}"
        );
    }
    // A change that only group 0 weighs: row 1 is outside group 1's and ξ's support, so group 0's
    // row alone disagrees, and the rejection is still at the column.
    base.codewords[1][3] = base.codewords[1][3] + Fe::ONE;
    let case = base.open(&groups, &xi, &queries);
    assert_eq!(verdict(&case), Err(OpeningError::Proximity { column: 3 }));
}

/// The answer to a draw that repeats an index holds each distinct column once, in increasing
/// order. Columns in draw order, the repeated one twice, a missing or extra column, and siblings
/// of the draw's order are `MerklePath` at the smallest queried index.
#[test]
fn an_answer_other_than_the_sorted_distinct_columns_is_rejected() {
    let mut rng = SplitMix64::new(0x6f70_656e_0061_0008);
    let (matrix, groups, xi, _) = honest(&mut rng, (2, 2, 2, 2, 3, 6, 2, &[2], 1));
    let queries = [9, 4, 9, 13, 4];
    let case = matrix.open(&groups, &xi, &queries);
    assert_eq!(verdict(&case), Ok(()));
    assert_eq!(case.distinct(), [4, 9, 13]);
    let column =
        |j: u64| -> Vec<Fp> { matrix.column(j).into_iter().map(reference::to_fp).collect() };
    let rejected = Err(OpeningError::MerklePath { column: 4 });
    let answers: [Vec<u64>; 5] = [
        vec![9, 4, 13],
        vec![9, 4, 9, 13, 4],
        vec![4, 9, 9, 13],
        vec![4, 9],
        vec![4, 5, 9, 13],
    ];
    for answer in answers {
        let mut changed = case.clone();
        changed.columns = answer.iter().map(|&j| column(j)).collect();
        assert_eq!(verdict(&changed), rejected, "{answer:?}");
    }
    // The siblings of another set of indices.
    let mut changed = case.clone();
    changed.siblings = reference::multiproof(&matrix.tree(), &[4, 13]);
    assert_eq!(verdict(&changed), rejected);
}

/// Openings with nothing committed (`N = 0`): every column is empty, every leaf merges the
/// segments' initial digests, and the combined rows must be zero at every queried column.
#[test]
fn an_opening_with_no_non_zero_row_checks_its_combined_rows_against_zero() {
    let mut rng = SplitMix64::new(0x6f70_656e_0061_0009);
    let (_, _, _, case) = honest(&mut rng, (2, 1, 2, 3, 4, 0, 3, &[1, 2], 4));
    assert_eq!(verdict(&case), Ok(()));
    assert!(case.columns.iter().all(Vec::is_empty));
    // A non-zero u that opens to the claimed 0 at the column point still disagrees with the
    // columns.
    let mut changed = case.clone();
    let point: Vec<Fe> = case.groups[0]
        .column_point
        .iter()
        .copied()
        .map(to_fe)
        .collect();
    changed.combined[0].u[0] += to_fp2(eq_at(&point, 1));
    changed.combined[0].u[1] -= to_fp2(eq_at(&point, 0));
    assert!(check_evaluations(changed.opening(), &changed.combined).is_ok());
    assert!(matches!(
        verdict(&changed),
        Err(OpeningError::Proximity { .. })
    ));
}
