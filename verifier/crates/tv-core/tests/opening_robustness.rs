// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Adversarial and malformed inputs to the opening checks (RFC-0019 'Robustness', SP-8): every
//! input yields a value or a typed error, without a panic.
//!
//! - Each of the verifier's own inputs out of range gives exactly its fault, in the order of the
//!   crate documentation, before anything of the proof is judged.
//! - Proofs with wrong counts and lengths of everything, some of them huge, get the rejection the
//!   order of checks gives, and a parameter set that declares a huge `k` is judged from the
//!   lengths alone.
//! - Seeded honest openings changed by one to four random edits, of the proof or of the
//!   verifier's inputs, get exactly the tests' oracle's verdict ([`reference::predict`]).

mod openings;

use std::time::{Duration, Instant};

use openings::Case;
use openings::reference::{self, Matrix, below, random_fe, random_groups, random_layout};
use tv_core::opening::{
    Claim, CombinedRow, Group, Layout, MAX_ROW_VARS, Opening, OpeningError, Params,
    check_evaluations,
};
use tv_field::{Fp, Fp2};
use tv_testkit::SplitMix64;
use tv_testkit::sumcheck::Fe;

fn random_fp2(rng: &mut SplitMix64) -> Fp2 {
    Fp2::new(Fp::reduce(rng.next_u64()), Fp::reduce(rng.next_u64()))
}

/// An honest opening of a seeded matrix: `k = 2^log2_k`, inverse rate 2, `P_A = 2`, `B = 2`, 6
/// non-zero rows of 3-bit indices in 2 segments, groups of 2, 1 and 3 claims, 6 draws.
fn honest(seed: u64, log2_k: u32) -> Case {
    let mut rng = SplitMix64::new(seed);
    let params = Params {
        log2_k,
        log2_inv_rate: 1,
        mask_coeffs: 2.min(1 << log2_k),
        row_block: 2,
        commitment_id: [0x5a; 32],
    };
    let layout = random_layout(&mut rng, 3, 6, 2);
    let matrix = Matrix::random(params, 3, layout, &mut rng);
    let groups = random_groups(&mut rng, log2_k, 3, &[2, 1, 3]);
    let xi: Vec<Fe> = (0..3).map(|_| random_fe(&mut rng)).collect();
    let queries: Vec<u64> = (0..6).map(|_| below(&mut rng, matrix.n())).collect();
    let case = matrix.open(&groups, &xi, &queries);
    assert_eq!(case.check(), Ok(()));
    case
}

/// A change made to a copy of an opening or of one of its parts.
type Change<T> = Box<dyn Fn(&mut T)>;

/// Each fault of the verifier's inputs, alone, gives exactly its error, and is reported before
/// the proof is read: here the proof also lacks its proximity row.
#[test]
fn each_fault_of_the_verifiers_inputs_is_reported_before_the_proof_is_read() {
    let case = honest(0x0b57_0061_0001, 2);
    let faults: Vec<(Change<Case>, OpeningError)> = vec![
        (
            Box::new(|c| c.params.log2_inv_rate = 0),
            OpeningError::CodeLength {
                log2_k: 2,
                log2_inv_rate: 0,
            },
        ),
        (
            Box::new(|c| c.params.log2_inv_rate = 31),
            OpeningError::CodeLength {
                log2_k: 2,
                log2_inv_rate: 31,
            },
        ),
        (
            Box::new(|c| c.params.log2_k = 40),
            OpeningError::CodeLength {
                log2_k: 40,
                log2_inv_rate: 1,
            },
        ),
        (
            Box::new(|c| c.params.log2_k = u32::MAX),
            OpeningError::CodeLength {
                log2_k: u32::MAX,
                log2_inv_rate: 1,
            },
        ),
        (
            Box::new(|c| c.params.log2_inv_rate = u32::MAX),
            OpeningError::CodeLength {
                log2_k: 2,
                log2_inv_rate: u32::MAX,
            },
        ),
        (
            Box::new(|c| c.params.mask_coeffs = 0),
            OpeningError::MaskCoefficients { mask_coeffs: 0 },
        ),
        (
            Box::new(|c| c.params.mask_coeffs = 5),
            OpeningError::MaskCoefficients { mask_coeffs: 5 },
        ),
        (
            Box::new(|c| c.params.mask_coeffs = u32::MAX),
            OpeningError::MaskCoefficients {
                mask_coeffs: u32::MAX,
            },
        ),
        (Box::new(|c| c.params.row_block = 0), OpeningError::RowBlock),
        (
            Box::new(|c| c.row_vars = MAX_ROW_VARS + 1),
            OpeningError::RowVars { row_vars: 65 },
        ),
        (
            Box::new(|c| c.row_vars = u32::MAX),
            OpeningError::RowVars { row_vars: u32::MAX },
        ),
        (Box::new(|c| c.segments.clear()), OpeningError::NoSegments),
        (
            Box::new(|c| {
                let last = *c.segments[0].last().unwrap();
                c.segments[1].insert(0, last);
            }),
            OpeningError::RowOrder {
                segment: 1,
                position: 0,
            },
        ),
        (
            Box::new(|c| {
                let first = c.segments[0][0];
                c.segments[0].insert(1, first);
            }),
            OpeningError::RowOrder {
                segment: 0,
                position: 1,
            },
        ),
        (
            Box::new(|c| c.segments[1].push(8)),
            OpeningError::RowRange {
                segment: 1,
                position: c_len(&case, 1),
            },
        ),
        (
            Box::new(|c| c.segments[1].push(u64::MAX)),
            OpeningError::RowRange {
                segment: 1,
                position: c_len(&case, 1),
            },
        ),
        (
            Box::new(|c| {
                c.groups[1].column_point.pop();
            }),
            OpeningError::ColumnPoint {
                group: 1,
                length: 1,
            },
        ),
        (
            Box::new(|c| c.groups[2].column_point.push(Fp2::ONE)),
            OpeningError::ColumnPoint {
                group: 2,
                length: 3,
            },
        ),
        (
            Box::new(|c| {
                c.groups[2].claims[1].row_point.pop();
            }),
            OpeningError::RowPoint {
                group: 2,
                claim: 1,
                length: 2,
            },
        ),
        (
            Box::new(|c| c.groups[0].claims[0].row_point.push(Fp2::ZERO)),
            OpeningError::RowPoint {
                group: 0,
                claim: 0,
                length: 4,
            },
        ),
        (
            Box::new(|c| c.xi_point.clear()),
            OpeningError::XiPoint { length: 0 },
        ),
    ];
    for (fault, expected) in faults {
        let mut changed = case.clone();
        fault(&mut changed);
        assert_eq!(changed.check(), Err(expected));
        assert_eq!(reference::predict(&changed), Err(expected));
        assert!(!expected.is_rejection());
        assert_eq!(expected.code(), "ERR-INTERNAL");
        // The proof's defect comes after: its proximity row missing.
        changed.combined.pop();
        assert_eq!(changed.check(), Err(expected));
    }
    // Two faults: the first in the documented order wins.
    let mut changed = case.clone();
    changed.params.row_block = 0;
    changed.xi_point.clear();
    changed.segments.clear();
    assert_eq!(changed.check(), Err(OpeningError::RowBlock));
    changed.params.mask_coeffs = 0;
    assert_eq!(
        changed.check(),
        Err(OpeningError::MaskCoefficients { mask_coeffs: 0 })
    );
}

fn c_len(case: &Case, segment: usize) -> usize {
    case.segments[segment].len()
}

/// The queries are the verifier's: none, or one at `n` or beyond, is its fault, at the first such
/// query in draw order, whatever the answer holds.
#[test]
fn queries_out_of_range_or_none_are_faults_of_the_verifier() {
    let case = honest(0x0b57_0061_0002, 2);
    let n = case.n();
    for (queries, expected) in [
        (vec![], OpeningError::NoQueries),
        (vec![n], OpeningError::QueryRange { position: 0 }),
        (
            vec![1, 2, n + 5, u64::MAX],
            OpeningError::QueryRange { position: 2 },
        ),
        (vec![u64::MAX], OpeningError::QueryRange { position: 0 }),
        (
            vec![0, n - 1, 1 << 40],
            OpeningError::QueryRange { position: 2 },
        ),
    ] {
        let mut changed = case.clone();
        changed.queries = queries;
        for columns in [vec![], case.columns.clone()] {
            changed.columns = columns;
            assert_eq!(changed.check(), Err(expected));
            assert_eq!(reference::predict(&changed), Err(expected));
            assert_eq!(expected.code(), "ERR-INTERNAL");
        }
    }
}

/// Wrong counts and lengths of every part of the proof, some huge, get the rejection the order of
/// checks gives: `Evaluation` without a group for the combined rows, `MerklePath` at the smallest
/// queried index for the answer. None of them is computed on: each is judged from its length.
#[test]
fn wrong_counts_and_lengths_of_everything_are_rejected_from_their_lengths() {
    const HUGE: usize = 1 << 20;
    let case = honest(0x0b57_0061_0003, 2);
    let k = case.k();
    let shape = Err(OpeningError::Evaluation { group: None });
    let merkle_path = Err(OpeningError::MerklePath {
        column: case.smallest(),
    });
    let row = |u: usize, mask: usize| CombinedRow {
        u: vec![Fp2::ONE; u],
        mask: vec![Fp2::ONE; mask],
    };
    let start = Instant::now();
    let combined_changes: Vec<Change<Vec<CombinedRow>>> = vec![
        Box::new(|c| c.clear()),
        Box::new(|c| {
            c.pop();
        }),
        Box::new(|c| c.push(c[0].clone())),
        Box::new(|c| {
            c.extend((0..100_000).map(|_| CombinedRow {
                u: vec![],
                mask: vec![],
            }))
        }),
        Box::new(|c| c[0].u.clear()),
        Box::new(|c| {
            c[1].u.pop();
        }),
        Box::new(|c| c[2].u.push(Fp2::ZERO)),
        Box::new(move |c| c[3] = row(HUGE, 2)),
        Box::new(|c| c[3].mask.clear()),
        Box::new(|c| {
            c[0].mask.pop();
        }),
        Box::new(|c| c[1].mask.push(Fp2::ZERO)),
        Box::new(move |c| c[2] = row(k, HUGE)),
    ];
    for change in combined_changes {
        let mut changed = case.clone();
        change(&mut changed.combined);
        assert_eq!(changed.check(), shape);
    }
    let rows = case.rows().len();
    let answer_changes: Vec<Change<Case>> = vec![
        Box::new(|c| c.columns.clear()),
        Box::new(|c| {
            c.columns.pop();
        }),
        Box::new(|c| c.columns.push(c.columns[0].clone())),
        Box::new(move |c| c.columns.extend((0..10_000).map(|_| vec![Fp::ONE; rows]))),
        Box::new(|c| c.columns[0].clear()),
        Box::new(|c| {
            c.columns[1].pop();
        }),
        Box::new(|c| c.columns[1].push(Fp::ZERO)),
        Box::new(|c| c.columns[0] = vec![Fp::ONE; HUGE]),
        Box::new(|c| c.siblings.clear()),
        Box::new(|c| {
            c.siblings.pop();
        }),
        Box::new(|c| c.siblings.push([7; 32])),
        Box::new(|c| c.siblings.extend(std::iter::repeat_n([1; 32], 1 << 16))),
        Box::new(|c| {
            c.columns.clear();
            c.siblings.clear();
        }),
    ];
    for change in answer_changes {
        let mut changed = case.clone();
        change(&mut changed);
        assert_eq!(changed.check(), merkle_path);
    }
    assert!(
        start.elapsed() < Duration::from_secs(20),
        "{:?}",
        start.elapsed()
    );
}

/// Parameters that declare a huge `k` are judged from lengths: rows of `2^31` values would be
/// 32 GiB, and the proof's rows of a few values are rejected before anything is computed. Beyond
/// `2^32` symbols there is no code at all.
#[test]
fn a_huge_declared_k_is_judged_from_the_lengths_alone() {
    let segments = vec![vec![0, 1, 2]];
    let groups = vec![Group {
        column_point: vec![Fp2::ONE; 31],
        claims: vec![Claim {
            row_point: vec![Fp2::ONE; 2],
            gamma: Fp2::ONE,
            value: Fp2::ONE,
        }],
    }];
    let xi = vec![Fp2::ONE; 2];
    let combined = vec![
        CombinedRow {
            u: vec![Fp2::ONE; 4],
            mask: vec![Fp2::ONE; 2],
        };
        2
    ];
    let opening = |log2_k, log2_inv_rate| Opening {
        params: Params {
            log2_k,
            log2_inv_rate,
            mask_coeffs: 2,
            row_block: 32,
            commitment_id: [0; 32],
        },
        layout: Layout {
            row_vars: 2,
            segments: &segments,
        },
        root: [0; 32],
        groups: &groups,
        xi_point: &xi,
    };
    let start = Instant::now();
    assert_eq!(
        check_evaluations(opening(31, 1), &combined).err(),
        Some(OpeningError::Evaluation { group: None })
    );
    for (log2_k, log2_inv_rate) in [(31, 2), (32, 1), (33, 1), (u32::MAX, 1)] {
        assert_eq!(
            check_evaluations(opening(log2_k, log2_inv_rate), &combined).err(),
            Some(OpeningError::CodeLength {
                log2_k,
                log2_inv_rate
            })
        );
    }
    assert!(start.elapsed() < Duration::from_secs(1));
}

/// One random edit: of the proof (a value, a length, an order), or of the verifier's inputs (a
/// parameter, the layout, a point, a challenge, the queries, the root).
fn edit(case: &mut Case, rng: &mut SplitMix64) {
    let pick = |rng: &mut SplitMix64, len: usize| below(rng, len.max(1) as u64) as usize;
    match below(rng, 24) {
        0..=2 => {
            let r = pick(rng, case.combined.len());
            if let Some(row) = case.combined.get_mut(r) {
                let i = pick(rng, row.u.len());
                if let Some(entry) = row.u.get_mut(i) {
                    *entry += random_fp2(rng);
                }
            }
        }
        3 => {
            let r = pick(rng, case.combined.len());
            if let Some(row) = case.combined.get_mut(r) {
                let i = pick(rng, row.mask.len());
                if let Some(entry) = row.mask.get_mut(i) {
                    *entry += random_fp2(rng);
                }
            }
        }
        4 => match below(rng, 6) {
            0 => {
                case.combined.pop();
            }
            1 if !case.combined.is_empty() => {
                let r = pick(rng, case.combined.len());
                case.combined.insert(r, case.combined[r].clone());
            }
            2 if !case.combined.is_empty() => {
                let r = pick(rng, case.combined.len());
                case.combined[r].u.pop();
            }
            3 if !case.combined.is_empty() => {
                let r = pick(rng, case.combined.len());
                case.combined[r].mask.push(Fp2::ZERO);
            }
            4 if case.combined.len() > 1 => {
                let (a, b) = (
                    pick(rng, case.combined.len()),
                    pick(rng, case.combined.len()),
                );
                case.combined.swap(a, b);
            }
            _ => {}
        },
        5 | 6 => {
            let g = pick(rng, case.groups.len());
            if let Some(group) = case.groups.get_mut(g) {
                let c = pick(rng, group.claims.len());
                if let Some(claim) = group.claims.get_mut(c) {
                    if below(rng, 2) == 0 {
                        claim.value += random_fp2(rng);
                    } else {
                        claim.gamma = random_fp2(rng);
                    }
                }
            }
        }
        7 | 8 => {
            let c = pick(rng, case.columns.len());
            if let Some(column) = case.columns.get_mut(c) {
                let s = pick(rng, column.len());
                if let Some(symbol) = column.get_mut(s) {
                    *symbol += Fp::reduce(1 + below(rng, 1 << 20));
                }
            }
        }
        9 => {
            let len = case.columns.len();
            match below(rng, 4) {
                0 if len > 1 => case.columns.swap(pick(rng, len), pick(rng, len)),
                1 if len > 0 => {
                    case.columns.remove(pick(rng, len));
                }
                2 if len > 0 => {
                    let c = pick(rng, len);
                    case.columns.insert(c, case.columns[c].clone());
                }
                _ if len > 0 => {
                    let c = pick(rng, len);
                    if below(rng, 2) == 0 {
                        case.columns[c].pop();
                    } else {
                        case.columns[c].push(Fp::ONE);
                    }
                }
                _ => {}
            }
        }
        10 | 11 => {
            let len = case.siblings.len();
            match below(rng, 4) {
                0 if len > 0 => case.siblings[pick(rng, len)][pick(rng, 32)] ^= 0x10,
                1 if len > 1 => case.siblings.swap(pick(rng, len), pick(rng, len)),
                2 if len > 0 => {
                    case.siblings.remove(pick(rng, len));
                }
                _ => case.siblings.insert(pick(rng, len + 1), [0x42; 32]),
            }
        }
        12 => case.root[pick(rng, 32)] ^= 1,
        13 | 14 => {
            let n = case.n();
            let len = case.queries.len();
            match below(rng, 5) {
                0 if len > 0 => case.queries[pick(rng, len)] = below(rng, n),
                1 => case.queries.push(below(rng, n)),
                2 if len > 1 => {
                    case.queries.remove(pick(rng, len));
                }
                3 if len > 0 => case.queries[pick(rng, len)] = n + below(rng, 3),
                _ => case.queries.reverse(),
            }
        }
        15 => match below(rng, 5) {
            0 => case.params.mask_coeffs = below(rng, 6) as u32,
            1 => case.params.row_block = below(rng, 4) as u32,
            2 => case.params.commitment_id[pick(rng, 32)] ^= 4,
            3 => case.params.log2_inv_rate = below(rng, 3) as u32,
            _ => case.params.log2_k = below(rng, 4) as u32,
        },
        16 | 17 => {
            // A row of the layout moved, removed or added, or two segments merged.
            let s = pick(rng, case.segments.len());
            match below(rng, 4) {
                0 if !case.segments[s].is_empty() => {
                    let p = pick(rng, case.segments[s].len());
                    case.segments[s][p] = below(rng, 9);
                }
                1 if !case.segments[s].is_empty() => {
                    let p = pick(rng, case.segments[s].len());
                    case.segments[s].remove(p);
                }
                2 if case.segments.len() > 1 => {
                    let tail = case.segments.remove(1);
                    case.segments[0].extend(tail);
                }
                _ => case.segments.push(vec![]),
            }
        }
        18 | 19 => {
            let g = pick(rng, case.groups.len());
            if let Some(group) = case.groups.get_mut(g) {
                if below(rng, 2) == 0 {
                    let t = pick(rng, group.column_point.len());
                    if let Some(x) = group.column_point.get_mut(t) {
                        *x = random_fp2(rng);
                    }
                } else if let Some(claim) = group.claims.first_mut() {
                    let t = pick(rng, claim.row_point.len());
                    if let Some(x) = claim.row_point.get_mut(t) {
                        *x = random_fp2(rng);
                    }
                }
            }
        }
        20 => {
            let t = pick(rng, case.xi_point.len());
            if let Some(x) = case.xi_point.get_mut(t) {
                *x = random_fp2(rng);
            }
        }
        21 => match below(rng, 3) {
            0 => {
                case.groups.pop();
            }
            1 => case.row_vars = below(rng, 5) as u32,
            _ => {
                case.xi_point.pop();
            }
        },
        _ => {
            // A group's claims reordered: the same sum, the same coefficients.
            if let Some(group) = case.groups.first_mut() {
                group.claims.reverse();
            }
        }
    }
}

/// Seeded honest openings changed by one to four random edits get exactly the oracle's verdict,
/// and an edit that changes nothing the checks read is accepted, as the oracle says.
#[test]
fn randomly_edited_openings_get_exactly_the_oracles_verdict() {
    let mut rng = SplitMix64::new(0x0b57_0061_0004);
    let bases: Vec<Case> = (0..21u64)
        .map(|i| honest(0x0b57_1000 + i, 1 + (i % 3) as u32))
        .collect();
    let mut tally = std::collections::BTreeMap::<&'static str, usize>::new();
    for trial in 0..3_000usize {
        let mut changed = bases[trial % bases.len()].clone();
        for _ in 0..1 + below(&mut rng, 4) {
            edit(&mut changed, &mut rng);
        }
        let outcome = changed.check();
        assert_eq!(outcome, reference::predict(&changed), "trial {trial}");
        *tally
            .entry(outcome.err().map_or("accepted", |e| e.code()))
            .or_default() += 1;
    }
    // Every kind of verdict is reached.
    for code in [
        "accepted",
        "REJ-PCS-EVAL",
        "REJ-PCS-MERKLE",
        "REJ-PCS-PROXIMITY",
        "ERR-INTERNAL",
    ] {
        assert!(
            tally.get(code).copied().unwrap_or(0) > 30,
            "{code}: {tally:?}"
        );
    }
}
