// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The memory `tv-core`'s opening checks allocate, counted and capped by the `cap` allocator
//! (RFC-0019 'Robustness'):
//!
//! - The evaluation check allocates nothing, whatever the proof holds.
//! - The column check allocates the sorted queries and the leaves, which the draws size; then one
//!   combined row's coefficients and one codeword, which the layout and the parameters size; the
//!   NTT's twiddles; and for each claim and `ξ*` its tables of `eq`, at most 256 entries for each
//!   8 bits of the row index. An answer of any shape, however many columns, symbols or siblings it
//!   holds, makes it allocate no more than the honest answer.
//! - Memory too short for the sorted queries or the codeword is `ERR-RES-MEMORY`, a fault of the
//!   verifier and never a verdict on the proof; with enough, the same answer verifies.
//!
//! The openings are the honest ones of `spec/vectors/pcs/opening.json`. The allocator's counters
//! and limit are global, so the cases run one after another in a single test.

use std::alloc::System;

use cap::Cap;
use tv_core::opening::{
    Claim, CombinedRow, Group, Layout, Opening, OpeningError, Params, check_evaluations,
};
use tv_field::{Fp, Fp2};
use tv_testkit::{Json, read_vectors};

#[global_allocator]
static ALLOCATOR: Cap<System> = Cap::new(System, usize::MAX);

/// Runs `f` with at most `budget` bytes allocatable beyond those allocated now, and returns its
/// result and the bytes it allocated in all.
fn within<T>(budget: usize, f: impl FnOnce() -> T) -> (T, usize) {
    let before = ALLOCATOR.total_allocated();
    ALLOCATOR
        .set_limit(ALLOCATOR.allocated().saturating_add(budget))
        .expect("a limit above the memory in use");
    let result = f();
    ALLOCATOR.set_limit(usize::MAX).expect("no limit");
    (result, ALLOCATOR.total_allocated() - before)
}

/// The bytes `f` allocates in all, and its result.
fn allocated_by<T>(f: impl FnOnce() -> T) -> (T, usize) {
    within(usize::MAX, f)
}

fn member<'a>(json: &'a Json, key: &str) -> &'a Json {
    json.get(key).unwrap_or_else(|| panic!("no `{key}`"))
}

fn list<'a>(json: &'a Json, key: &str) -> &'a [Json] {
    member(json, key).as_array().expect("an array")
}

fn number(json: &Json, key: &str) -> u32 {
    u32::try_from(member(json, key).as_u64().expect("a number")).expect("a u32")
}

fn fp(json: &Json) -> Fp {
    Fp::from_canonical(json.as_hex_u64().expect("hex")).expect("canonical")
}

fn fp2(json: &Json) -> Fp2 {
    match json.as_array() {
        Some([c0, c1]) => Fp2::new(fp(c0), fp(c1)),
        _ => panic!("an Fp2 element"),
    }
}

fn fp2s(json: &Json) -> Vec<Fp2> {
    json.as_array().expect("a list").iter().map(fp2).collect()
}

fn digest(json: &Json) -> [u8; 32] {
    tv_testkit::hex(json.as_str().expect("hex"))
        .try_into()
        .expect("32 bytes")
}

/// An honest opening of the vectors, owned.
struct Case {
    params: Params,
    row_vars: u32,
    segments: Vec<Vec<u64>>,
    root: [u8; 32],
    groups: Vec<Group>,
    xi_point: Vec<Fp2>,
    combined: Vec<CombinedRow>,
    queries: Vec<u64>,
    columns: Vec<Vec<Fp>>,
    siblings: Vec<[u8; 32]>,
}

impl Case {
    fn from_json(json: &Json) -> Self {
        Self {
            params: Params {
                log2_k: number(json, "log2_k"),
                log2_inv_rate: number(json, "log2_inv_rate"),
                mask_coeffs: number(json, "mask_coeffs"),
                row_block: number(json, "row_block"),
                commitment_id: digest(member(json, "commitment_id")),
            },
            row_vars: number(json, "row_vars"),
            segments: list(json, "segments")
                .iter()
                .map(|s| {
                    s.as_array()
                        .expect("rows")
                        .iter()
                        .map(|r| r.as_u64().expect("a row"))
                        .collect()
                })
                .collect(),
            root: digest(member(json, "root")),
            groups: list(json, "groups")
                .iter()
                .map(|g| Group {
                    column_point: fp2s(member(g, "column_point")),
                    claims: list(g, "claims")
                        .iter()
                        .map(|c| Claim {
                            row_point: fp2s(member(c, "row_point")),
                            gamma: fp2(member(c, "gamma")),
                            value: fp2(member(c, "value")),
                        })
                        .collect(),
                })
                .collect(),
            xi_point: fp2s(member(json, "xi_point")),
            combined: list(json, "combined")
                .iter()
                .map(|row| CombinedRow {
                    u: fp2s(member(row, "u")),
                    mask: fp2s(member(row, "mask")),
                })
                .collect(),
            queries: list(json, "queries")
                .iter()
                .map(|q| q.as_u64().expect("a query"))
                .collect(),
            columns: list(json, "columns")
                .iter()
                .map(|c| c.as_array().expect("a column").iter().map(fp).collect())
                .collect(),
            siblings: list(json, "siblings").iter().map(digest).collect(),
        }
    }

    fn opening(&self) -> Opening<'_> {
        Opening {
            params: self.params,
            layout: Layout {
                row_vars: self.row_vars,
                segments: &self.segments,
            },
            root: self.root,
            groups: &self.groups,
            xi_point: &self.xi_point,
        }
    }

    /// The column check on `columns` and `siblings` in place of the answer, within `budget`
    /// bytes, after an evaluation check that allocates nothing.
    fn columns_within(
        &self,
        budget: usize,
        columns: &[Vec<Fp>],
        siblings: &[[u8; 32]],
    ) -> (Result<(), OpeningError>, usize) {
        let (evaluated, bytes) = allocated_by(|| check_evaluations(self.opening(), &self.combined));
        assert_eq!(bytes, 0, "the evaluation check allocates nothing");
        let evaluated = evaluated.expect("honest combined rows");
        within(budget, || {
            evaluated.check_columns(&self.queries, columns, siblings)
        })
    }
}

fn opening_checks_allocate_only_what_the_parameters_and_the_draws_size() {
    let file = read_vectors("pcs/opening.json");
    let cases: Vec<Case> = list(&file, "openings")
        .iter()
        .map(Case::from_json)
        .collect();
    assert_eq!(cases.len(), 10);
    for (index, case) in cases.iter().enumerate() {
        let k = 1usize << case.params.log2_k;
        let n = k << case.params.log2_inv_rate;
        let rows: usize = case.segments.iter().map(Vec::len).sum();
        let tau = case.queries.len();
        let distinct = case.columns.len();
        let groups = case.groups.len();
        let claims: usize = case.groups.iter().map(|g| g.claims.len()).sum();
        let smallest = u32::try_from(*case.queries.iter().min().unwrap()).unwrap();

        // The evaluation check allocates nothing, honest or hostile.
        let mut hostile = case.combined.clone();
        hostile[0].u = vec![Fp2::ONE; 1 << 20];
        let (outcome, bytes) = allocated_by(|| check_evaluations(case.opening(), &hostile).err());
        assert_eq!(outcome, Some(OpeningError::Evaluation { group: None }));
        assert_eq!(bytes, 0, "opening {index}: a row of a million values");
        hostile = case.combined.clone();
        hostile.extend((0..100_000).map(|_| CombinedRow {
            u: vec![],
            mask: vec![],
        }));
        let (outcome, bytes) = allocated_by(|| check_evaluations(case.opening(), &hostile).err());
        assert_eq!(outcome, Some(OpeningError::Evaluation { group: None }));
        assert_eq!(bytes, 0, "opening {index}: 100,000 rows");

        // The honest answer: the sorted queries, the leaves, one row's coefficients, one
        // codeword, the twiddles of an inverse NTT of size k and a coset NTT of size n per
        // combined row, and the tables of eq of each claim and ξ*.
        let (outcome, honest) = case.columns_within(usize::MAX, &case.columns, &case.siblings);
        assert_eq!(outcome, Ok(()), "opening {index}");
        let buffers = 8 * tau + 40 * distinct + 16 * rows + 16 * n;
        let twiddles = (groups + 1) * 4 * (n + if k > 1 { k } else { 0 });
        let tables = case.row_vars.div_ceil(8) as usize;
        let eq_tables = (claims + 1) * (24 * tables + tables * (16 + 48 * 256));
        assert!(
            honest >= buffers + twiddles && honest <= buffers + twiddles + eq_tables,
            "opening {index}: {honest} bytes, buffers {buffers}, twiddles {twiddles}, \
             tables at most {eq_tables}"
        );

        // Hostile answers allocate no more than the honest one: each is rejected from its shape
        // before anything is sized by it.
        let rejected = Err(OpeningError::MerklePath { column: smallest });
        let many_columns: Vec<Vec<Fp>> = case
            .columns
            .iter()
            .cloned()
            .chain((0..10_000).map(|_| vec![Fp::ONE; rows]))
            .collect();
        let mut long_column = case.columns.clone();
        long_column[0] = vec![Fp::ONE; 1 << 20];
        let many_siblings: Vec<[u8; 32]> = case
            .siblings
            .iter()
            .copied()
            .chain(std::iter::repeat_n([3; 32], 1 << 16))
            .collect();
        for (columns, siblings, what) in [
            (
                &many_columns[..],
                &case.siblings[..],
                "10,000 extra columns",
            ),
            (
                &long_column[..],
                &case.siblings[..],
                "a column of a million symbols",
            ),
            (
                &case.columns[..],
                &many_siblings[..],
                "65,536 extra siblings",
            ),
            (&[][..], &[][..], "an empty answer"),
        ] {
            let (outcome, bytes) = case.columns_within(usize::MAX, columns, siblings);
            assert_eq!(outcome, rejected, "opening {index}: {what}");
            assert!(
                bytes <= 8 * tau + 40 * distinct,
                "opening {index}: {what}, {bytes} bytes"
            );
        }

        // Too little memory for the sorted queries, or for the codeword: the verifier's own
        // failure, ERR-RES-MEMORY, never a verdict. The codeword is reserved before any NTT runs.
        for budget in [8 * tau - 1, buffers - 1] {
            let (outcome, _) = case.columns_within(budget, &case.columns, &case.siblings);
            assert_eq!(
                outcome,
                Err(OpeningError::Memory),
                "opening {index}, {budget} bytes"
            );
            assert_eq!(OpeningError::Memory.code(), "ERR-RES-MEMORY");
        }
        let (outcome, _) = case.columns_within(honest, &case.columns, &case.siblings);
        assert_eq!(
            outcome,
            Ok(()),
            "opening {index}: its own allocation suffices"
        );
    }
}

/// Runs on the process's only thread (`harness = false`): libtest would keep a thread of its own,
/// whose allocations the process-wide cap counts against this test's budget (#501).
fn main() {
    opening_checks_allocate_only_what_the_parameters_and_the_draws_size();
    println!("opening_checks_allocate_only_what_the_parameters_and_the_draws_size: ok");
}
