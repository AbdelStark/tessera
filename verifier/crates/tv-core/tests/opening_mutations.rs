// SPDX-License-Identifier: Apache-2.0 OR MIT

//! #61's acceptance criterion: every single-element change of every honest opening of
//! `spec/vectors/pcs/opening.json` is rejected with the reason the order of checks gives. Each
//! coefficient of each entry of every `u` and every mask, and of every claim's value, is changed
//! by one in turn, every column symbol is changed by one, and one bit of every sibling is
//! flipped:
//!
//! - Entry `i` of group `g`'s `u` changes `⟨u_g, eq(column point)⟩` by `δ·eq(column point, i)`,
//!   so it is `Evaluation` at `g` unless that factor is zero: a column point with a Boolean
//!   coordinate gives `eq` zero at half the entries. Then the evaluation check cannot see the
//!   change, and the column check does: `enc(δ·e_i)` is `δ` times the Lagrange polynomial of
//!   `ω_k^i`, which vanishes only on `H`, so the row disagrees with every column, and the
//!   rejection is `Proximity` at the smallest queried index.
//! - An entry of the proximity row's `u`, which no evaluation check reads, and any mask
//!   coefficient, whose term `δ·Z_H(x_j)·x_j^c` vanishes nowhere on `7·L`: `Proximity` at the
//!   smallest queried index.
//! - A column symbol changes its segment's digest and the leaf, and a sibling the path through
//!   it: `MerklePath` at the smallest queried index, whichever column or sibling changed.
//! - A claim's value changes its group's sum by `γ·δ`: `Evaluation` at its group, unless `γ` is
//!   zero, when nothing reads the value and the opening is accepted.
//!
//! The tests' oracle ([`reference::predict`]) gives the same verdict for every change.

mod openings;

use openings::{Case, reference, vectors};
use reference::{eq_at, to_fe};
use tv_core::opening::OpeningError;
use tv_field::{Fp, Fp2};

/// The two coefficients' unit changes: `+1` and `+X`.
const DELTAS: [Fp2; 2] = [Fp2::ONE, Fp2::new(Fp::ZERO, Fp::ONE)];

/// Checks `changed` against `expected` and against the tests' oracle.
fn assert_verdict(changed: &Case, expected: Result<(), OpeningError>, what: &str) {
    assert_eq!(changed.check(), expected, "{}: {what}", changed.name);
    assert_eq!(
        reference::predict(changed),
        expected,
        "{}: the oracle, {what}",
        changed.name
    );
}

#[test]
fn every_single_element_change_of_every_honest_opening_is_rejected() {
    let mut changed = 0;
    let mut expected_total = 0;
    let mut unseen_by_evaluation = 0;
    for case in vectors().openings {
        let groups = case.groups.len();
        let proximity = Err(OpeningError::Proximity {
            column: case.smallest(),
        });
        let merkle_path = Err(OpeningError::MerklePath {
            column: case.smallest(),
        });
        let claims: usize = case.groups.iter().map(|g| g.claims.len()).sum();
        expected_total += 2 * case.combined.len() * (case.k() + case.params.mask_coeffs as usize)
            + case.columns.iter().map(Vec::len).sum::<usize>()
            + case.siblings.len()
            + 2 * claims;

        for (r, row) in case.combined.iter().enumerate() {
            for i in 0..row.u.len() {
                // eq(column point, i) for a group's row; none for the proximity row.
                let factor = case.groups.get(r).map(|group| {
                    let point: Vec<_> = group.column_point.iter().copied().map(to_fe).collect();
                    eq_at(&point, i as u64)
                });
                for delta in DELTAS {
                    let mut mutant = case.clone();
                    mutant.combined[r].u[i] += delta;
                    let expected = match factor {
                        Some(factor) if factor != tv_testkit::sumcheck::Fe::ZERO => {
                            Err(OpeningError::Evaluation { group: Some(r) })
                        }
                        Some(_) => {
                            unseen_by_evaluation += 1;
                            proximity
                        }
                        None => proximity,
                    };
                    assert_verdict(&mutant, expected, &format!("row {r}, u[{i}] + {delta:?}"));
                    changed += 1;
                }
            }
            for c in 0..row.mask.len() {
                for delta in DELTAS {
                    let mut mutant = case.clone();
                    mutant.combined[r].mask[c] += delta;
                    assert_verdict(&mutant, proximity, &format!("row {r}, M[{c}] + {delta:?}"));
                    changed += 1;
                }
            }
        }
        for (column, symbols) in case.columns.iter().enumerate() {
            for s in 0..symbols.len() {
                let mut mutant = case.clone();
                mutant.columns[column][s] += Fp::ONE;
                assert_verdict(
                    &mutant,
                    merkle_path,
                    &format!("column {column}, symbol {s}"),
                );
                changed += 1;
            }
        }
        for s in 0..case.siblings.len() {
            let mut mutant = case.clone();
            mutant.siblings[s][s % 32] ^= 1 << (s % 8);
            assert_verdict(&mutant, merkle_path, &format!("sibling {s}"));
            changed += 1;
        }
        for g in 0..groups {
            for c in 0..case.groups[g].claims.len() {
                let gamma = case.groups[g].claims[c].gamma;
                for delta in DELTAS {
                    let mut mutant = case.clone();
                    mutant.groups[g].claims[c].value += delta;
                    let expected = if gamma == Fp2::ZERO {
                        Ok(())
                    } else {
                        Err(OpeningError::Evaluation { group: Some(g) })
                    };
                    assert_verdict(&mutant, expected, &format!("group {g}, claim {c}'s value"));
                    changed += 1;
                }
            }
        }
    }
    assert_eq!(changed, expected_total);
    // 10 openings: 484 entries of u and masks changed in each coefficient, 167 column symbols,
    // 56 siblings and 60 claim values changed in each coefficient.
    assert_eq!(changed, 2 * 484 + 167 + 56 + 2 * 60);
    // Every column point of the file is random, so every change of a group's u is Evaluation.
    assert_eq!(unseen_by_evaluation, 0);
}
