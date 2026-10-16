// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The opening checks against `spec/vectors/pcs/opening.json`, written by an independent Python
//! oracle from RFC-0011 (#61): every honest opening is accepted, and every opening changed in one
//! place is rejected with exactly the oracle's reason, group and column.
//!
//! The tests' own reference ([`openings::reference`]) is checked against the same file: its
//! oracle gives every vector's verdict, and it recomputes every value the file records to locate
//! a divergence. The other opening tests rely on it for what the file cannot hold.

mod openings;

use std::collections::BTreeSet;

use openings::reference::{self, Interpolator, column_point, eq_at, to_fe};
use openings::{list, member, number, vectors};
use tv_testkit::sumcheck::{Fe, P};

#[test]
fn the_file_states_the_field_and_the_coset() {
    let json = vectors().json;
    assert_eq!(member(&json, "modulus").as_hex_u64(), Some(P));
    assert_eq!(number(&json, "non_residue"), 7);
    assert_eq!(number(&json, "coset_shift"), 7);
    assert_eq!(number(&json, "official_blake3_cases_checked"), 35);
    assert!(member(&json, "seed").as_hex_u64().is_some());
    assert_eq!(list(&json, "openings").len(), 10);
}

#[test]
fn every_honest_opening_is_accepted() {
    let vectors = vectors();
    for case in &vectors.openings {
        assert_eq!(case.check(), Ok(()), "{}", case.name);
    }
    // Shapes the cases cover: one row (row_vars = 0); empty segments, first and middle; P_A from
    // 1 to k; B = 1, 2 and 32; repeated draws; every column queried, with no sibling.
    let cases = &vectors.openings;
    assert!(cases.iter().any(|c| c.row_vars == 0));
    assert!(cases.iter().any(|c| c.segments.iter().any(Vec::is_empty)));
    assert!(cases.iter().any(|c| c.params.mask_coeffs == 1));
    assert!(cases.iter().any(|c| c.params.mask_coeffs as usize == c.k()));
    for block in [1, 2, 32] {
        assert!(cases.iter().any(|c| c.params.row_block == block));
    }
    assert!(cases.iter().any(|c| c.distinct().len() < c.queries.len()));
    assert!(
        cases
            .iter()
            .any(|c| c.siblings.is_empty() && c.distinct().len() as u64 == c.n())
    );
}

#[test]
fn every_rejection_has_the_oracles_reason_group_and_column() {
    let vectors = vectors();
    assert_eq!(vectors.rejections.len(), 28);
    let mut reasons = BTreeSet::new();
    for rejection in &vectors.rejections {
        let outcome = rejection.case.check();
        assert_eq!(outcome, Err(rejection.expected), "{}", rejection.name);
        assert!(rejection.expected.is_rejection(), "{}", rejection.name);
        reasons.insert(rejection.expected.code());
        // A rejection's opening carries the name of the honest opening it was changed from.
        assert!(
            vectors
                .openings
                .iter()
                .any(|case| case.name == rejection.case.name),
            "{}: no honest opening is named {}",
            rejection.name,
            rejection.case.name
        );
    }
    assert_eq!(
        reasons,
        BTreeSet::from(["REJ-PCS-EVAL", "REJ-PCS-MERKLE", "REJ-PCS-PROXIMITY"])
    );
}

/// The tests' oracle, written from the crate documentation, gives every vector's verdict: the
/// independent Python oracle and it agree on all of them.
#[test]
fn the_tests_oracle_gives_every_vectors_verdict() {
    let vectors = vectors();
    for case in &vectors.openings {
        assert_eq!(reference::predict(case), Ok(()), "{}", case.name);
    }
    for rejection in &vectors.rejections {
        assert_eq!(
            reference::predict(&rejection.case),
            Err(rejection.expected),
            "{}",
            rejection.name
        );
    }
}

/// The tests' reference recomputes every value the file records for an honest opening: the row
/// coefficients, the queried columns' points and leaves, and each combined row's value there.
#[test]
fn the_tests_reference_recomputes_every_recorded_value() {
    for case in vectors().openings {
        let what = &case.name;
        let recorded = case
            .intermediate
            .as_ref()
            .expect("an honest opening records its values");
        let rows = case.rows();
        let distinct = case.distinct();

        let coefficients: Vec<Vec<Fe>> = case
            .groups
            .iter()
            .map(|group| {
                rows.iter()
                    .map(|&row| {
                        group.claims.iter().fold(Fe::ZERO, |s, claim| {
                            let point: Vec<Fe> =
                                claim.row_point.iter().copied().map(to_fe).collect();
                            s + to_fe(claim.gamma) * eq_at(&point, row)
                        })
                    })
                    .collect()
            })
            .chain([{
                let xi: Vec<Fe> = case.xi_point.iter().copied().map(to_fe).collect();
                rows.iter().map(|&row| eq_at(&xi, row)).collect()
            }])
            .collect();
        let expected: Vec<Vec<Fe>> = recorded
            .row_coefficients
            .iter()
            .map(|c| c.iter().copied().map(to_fe).collect())
            .collect();
        assert_eq!(coefficients, expected, "{what}: the row coefficients");

        let points: Vec<Fe> = distinct
            .iter()
            .map(|&j| column_point(case.n(), j))
            .collect();
        let expected: Vec<Fe> = recorded
            .points
            .iter()
            .map(|x| Fe(x.to_canonical(), 0))
            .collect();
        assert_eq!(points, expected, "{what}: the points");

        let leaves: Vec<_> = distinct
            .iter()
            .zip(&case.columns)
            .map(|(&j, column)| {
                let column: Vec<Fe> = column.iter().map(|s| Fe(s.to_canonical(), 0)).collect();
                reference::leaf(
                    &case.params.commitment_id,
                    case.params.row_block as usize,
                    &case.segments,
                    j,
                    &column,
                )
            })
            .collect();
        assert_eq!(leaves, recorded.leaves, "{what}: the leaves");

        let interpolator = Interpolator::new(case.k() as u64);
        for (r, row) in case.combined.iter().enumerate() {
            let u: Vec<Fe> = row.u.iter().copied().map(to_fe).collect();
            let mask: Vec<Fe> = row.mask.iter().copied().map(to_fe).collect();
            let values: Vec<Fe> = points
                .iter()
                .map(|&x| interpolator.encode_at(&u, &mask, x))
                .collect();
            let expected: Vec<Fe> = recorded.combined_at_queries[r]
                .iter()
                .copied()
                .map(to_fe)
                .collect();
            assert_eq!(values, expected, "{what}: combined row {r} at the queries");
        }
    }
}
