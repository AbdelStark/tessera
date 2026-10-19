// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Independent RFC-0012/0040/0049 vectors and isolated proof faults; no prover crate.

mod claim_support;
mod openings;

use claim_support::{at, golden, integer};
use openings::{fp, fp2, list, member, number};
use tv_core::claims::{ClaimList, Site};
use tv_core::opening::CombinedRow;
use tv_core::weights::{self, Claim, Error, Layout, Matrix, Opening};
use tv_field::{Fp, Fp2};
use tv_hash::Digest;
use tv_testkit::{Json, hex, read_vectors};

fn digest(j: &Json) -> Digest {
    hex(j.as_str().unwrap()).try_into().unwrap()
}
fn pairs(j: &Json) -> Vec<Fp2> {
    j.as_array().unwrap().iter().map(fp2).collect()
}
fn fixture(j: &Json) -> ClaimList {
    let mut c = golden("tiny-2-servable");
    let a = member(j, "arch");
    let get = |key| u32::try_from(integer(member(a, key))).unwrap();
    c.arch.n_layers = get("n_layers");
    c.arch.d_model = get("d_model");
    c.arch.n_heads = get("n_heads");
    c.arch.n_kv_heads = get("n_kv_heads");
    c.arch.d_head = get("d_head");
    c.arch.d_ff = get("d_ff");
    c.arch.vocab = get("vocab");
    let l = member(j, "layout");
    c.layout.weights.log2_k = u8::try_from(number(l, "log2_k")).unwrap();
    c.layout.advice.row_block = u32::try_from(number(j, "row_block")).unwrap();
    c.quant.shift_class_sets.clear();
    c.quant.affine_sites.clear();
    for entry in list(j, "channels") {
        let site = match number(entry, "matrix") {
            1 => Site::RopeQ,
            2 => Site::RopeK,
            3 => Site::V,
            4 => Site::AttnResidual,
            5 => Site::Silu,
            6 => Site::Up,
            7 => Site::MlpResidual,
            _ => panic!("a channel type"),
        };
        let point = at(u32::try_from(number(entry, "layer")).unwrap(), site);
        if member(entry, "kind").as_str() == Some("class") {
            c.quant.shift_class_sets.push((point, vec![0, 1]));
        } else {
            c.quant.affine_sites.push((point, 2));
        }
    }
    c
}
#[derive(Clone)]
struct Case {
    name: String,
    layout: Layout,
    root: Digest,
    row: Vec<Fp2>,
    column: Vec<Fp2>,
    xi: Vec<Fp2>,
    gamma: Fp2,
    value: Fp2,
    combined: Vec<CombinedRow>,
    queries: Vec<u64>,
    columns: Vec<Vec<Fp>>,
    siblings: Vec<Digest>,
}
impl Case {
    fn read(j: &Json) -> Self {
        let layouts =
            weights::derive(&fixture(j), u32::try_from(number(j, "epoch")).unwrap()).unwrap();
        let layout = layouts[usize::try_from(number(j, "matrix")).unwrap()].clone();
        assert_eq!(layout.log2_k(), u32::try_from(number(j, "log2_k")).unwrap());
        assert_eq!(
            layout.row_vars(),
            u32::try_from(number(j, "row_vars")).unwrap()
        );
        assert_eq!(layout.commitment_id(), digest(member(j, "commitment_id")));
        let rows = list(j, "committed_rows");
        assert_eq!(layout.committed_rows(), rows.len() as u64);
        for (p, r) in rows.iter().enumerate() {
            assert_eq!(layout.row_at(p as u64), r.as_u64());
        }
        Self {
            name: member(j, "name").as_str().unwrap().into(),
            layout,
            root: digest(member(j, "root")),
            row: pairs(member(j, "row_point")),
            column: pairs(member(j, "column_point")),
            xi: pairs(member(j, "xi_point")),
            gamma: fp2(member(j, "gamma")),
            value: fp2(member(j, "value")),
            combined: list(j, "combined")
                .iter()
                .map(|u| CombinedRow {
                    u: pairs(u),
                    mask: vec![],
                })
                .collect(),
            queries: list(j, "queries")
                .iter()
                .map(|x| x.as_u64().unwrap())
                .collect(),
            columns: list(j, "columns")
                .iter()
                .map(|col| col.as_array().unwrap().iter().map(fp).collect())
                .collect(),
            siblings: list(j, "siblings").iter().map(digest).collect(),
        }
    }
    fn opening(&self) -> Opening<'_> {
        Opening {
            layout: &self.layout,
            root: self.root,
            xi_point: &self.xi,
            claim: Claim {
                row_point: &self.row,
                column_point: &self.column,
                gamma: self.gamma,
                value: self.value,
            },
        }
    }
    fn check(&self) -> Result<(), Error> {
        weights::check_evaluations(self.opening(), &self.combined)?.check_columns(
            &self.queries,
            &self.columns,
            &self.siblings,
        )
    }
    fn eval_error(&self) -> Error {
        Error::Evaluation {
            matrix: self.layout.matrix(),
        }
    }
    fn merkle_error(&self) -> Error {
        Error::MerklePath {
            matrix: self.layout.matrix(),
            column: 0,
        }
    }
    fn proximity_error(&self) -> Error {
        Error::Proximity {
            matrix: self.layout.matrix(),
            column: 0,
        }
    }
}
fn cases() -> Vec<Case> {
    list(&read_vectors("weights-opening/openings.json"), "cases")
        .iter()
        .map(Case::read)
        .collect()
}

#[test]
fn independent_weight_openings_verify_with_positions_distinct_from_structural_rows() {
    let cases = cases();
    assert_eq!(cases.len(), 7);
    for c in cases {
        assert_eq!(c.check(), Ok(()), "{}", c.name);
    }
}
#[test]
fn value_and_combined_row_shape_fail_at_evaluation() {
    for honest in cases().into_iter().filter(|c| !c.combined.is_empty()) {
        for fault in 0..9 {
            let mut c = honest.clone();
            match fault {
                0 => c.value += Fp2::ONE,
                1 => c.combined[0].u[0] += Fp2::ONE,
                2 => {
                    c.combined[0].u.pop();
                }
                3 => c.combined[1].mask.push(Fp2::ZERO),
                4 => {
                    c.combined.pop();
                }
                5 => c.combined.push(c.combined[0].clone()),
                6 => c.combined[0].mask.push(Fp2::ZERO),
                7 => {
                    c.combined[1].u.pop();
                }
                _ => c.combined[1].u.push(Fp2::ZERO),
            }
            assert_eq!(c.check(), Err(c.eval_error()), "{} fault{fault}", c.name);
        }
    }
}
#[test]
fn every_column_authentication_fault_has_the_exact_matrix_and_smallest_column() {
    for honest in cases().into_iter().filter(|c| !c.combined.is_empty()) {
        for fault in 0..7 {
            // Identical symbols are the same answer; an empty multiproof has no sibling to omit.
            if (fault == 3 && honest.columns[0] == honest.columns[1])
                || (fault == 4 && honest.siblings.is_empty())
            {
                continue;
            }
            let mut c = honest.clone();
            match fault {
                0 => c.columns[0][0] += Fp::ONE,
                1 => {
                    c.columns[0].pop();
                }
                2 => {
                    c.columns.pop();
                }
                3 => c.columns.swap(0, 1),
                4 => {
                    c.siblings.pop();
                }
                5 => c.siblings.push([0; 32]),
                _ => c.root[0] ^= 1,
            }
            assert_eq!(c.check(), Err(c.merkle_error()), "{} fault{fault}", c.name);
        }
    }
}
#[test]
fn both_combined_rows_are_bound_to_authenticated_columns() {
    for honest in cases().into_iter().filter(|c| !c.combined.is_empty()) {
        let mut c = honest.clone();
        c.combined[1].u[0] += Fp2::ONE;
        assert!(weights::check_evaluations(c.opening(), &c.combined).is_ok());
        assert_eq!(c.check(), Err(c.proximity_error()), "{} proximity", c.name);
        if !honest.column.is_empty() {
            let mut c = honest.clone();
            let r = c.column[0];
            c.combined[0].u[0] += r;
            c.combined[0].u[1] -= Fp2::ONE - r;
            assert!(weights::check_evaluations(c.opening(), &c.combined).is_ok());
            assert_eq!(
                c.check(),
                Err(c.proximity_error()),
                "{} evaluation row",
                c.name
            );
        }
    }
}
#[test]
fn empty_matrices_have_zero_claims_and_no_combined_rows_or_queries() {
    let honest = cases()
        .into_iter()
        .find(|c| c.layout.committed_rows() == 0)
        .unwrap();
    assert_eq!(honest.check(), Ok(()));
    let mut c = honest.clone();
    c.value = Fp2::ONE;
    assert_eq!(c.check(), Err(c.eval_error()));
    let mut c = honest.clone();
    c.combined.push(CombinedRow {
        u: vec![],
        mask: vec![],
    });
    assert_eq!(c.check(), Err(c.eval_error()));
    let mut c = honest.clone();
    c.queries.push(0);
    assert_eq!(c.check(), Err(Error::Internal));
    let mut c = honest.clone();
    c.columns.push(vec![]);
    assert_eq!(c.check(), Err(c.merkle_error()));
    let mut c = honest;
    c.siblings.push([0; 32]);
    assert_eq!(c.check(), Err(c.merkle_error()));
}
#[test]
fn incorrect_verifier_points_and_queries_are_inconclusive() {
    let honest = cases().remove(0);
    for part in 0..3 {
        let mut c = honest.clone();
        match part {
            0 => c.column.push(Fp2::ZERO),
            1 => c.row.push(Fp2::ZERO),
            _ => c.xi.push(Fp2::ZERO),
        }
        assert_eq!(c.check(), Err(Error::Internal));
        c.combined.clear();
        assert_eq!(
            c.check(),
            Err(Error::Internal),
            "own point errors precede proof shape errors"
        );
    }
    let mut c = honest.clone();
    c.queries.clear();
    assert_eq!(c.check(), Err(Error::Internal));
    let mut c = honest;
    c.queries.push(1u64 << (c.layout.log2_k() + 2));
    assert_eq!(c.check(), Err(Error::Internal));
    // Own-input validation precedes proof rejection, even if the answer is also malformed.
    c.columns.clear();
    c.siblings.clear();
    assert_eq!(c.check(), Err(Error::Internal));
    for e in [Error::Usage, Error::Internal, Error::Memory] {
        assert!(!e.is_rejection());
    }
    for e in [
        Error::Layout,
        Error::Evaluation { matrix: Matrix::Q },
        Error::MerklePath {
            matrix: Matrix::Q,
            column: 0,
        },
        Error::Proximity {
            matrix: Matrix::Q,
            column: 0,
        },
    ] {
        assert!(e.is_rejection());
    }
}
#[test]
fn public_artifacts_bind_root_epoch_and_every_layout_field_in_order() {
    use tv_codec::{WeightCommitmentPublic, WeightLayoutParams};
    assert_eq!(Error::Layout.code(), "REJ-STATEMENT");
    assert_eq!(Error::Usage.code(), "ERR-USAGE");
    let vectors = read_vectors("weights/weights.json");
    for m in list(&vectors, "models")
        .iter()
        .filter(|m| member(m, "name").as_str() != Some("synthetic"))
    {
        let claims = golden(member(m, "name").as_str().unwrap());
        let p = claims.layout.weights;
        let public = WeightCommitmentPublic {
            layout: WeightLayoutParams {
                log2_k: p.log2_k,
                log2_inv_rate: p.log2_inv_rate,
                mask_coeffs: p.mask_coeffs,
                blinding_rows: p.blinding_rows,
            },
            roots: list(m, "matrices")
                .iter()
                .map(|t| digest(member(t, "root")))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            epoch: 0,
        };
        let root = digest(member(m, "weights_root"));
        assert_eq!(weights::check_public(&claims, &public, &root, 0), Ok(()));
        assert_eq!(
            weights::check_public(&claims, &public, &root, 1),
            Err(Error::Usage)
        );
        let mut wrong = root;
        wrong[0] ^= 1;
        assert_eq!(
            weights::check_public(&claims, &public, &wrong, 0),
            Err(Error::Usage)
        );
        for part in 0..4 {
            let mut c = claims.clone();
            match part {
                0 => c.layout.weights.log2_k ^= 1,
                1 => c.layout.weights.log2_inv_rate ^= 1,
                2 => c.layout.weights.mask_coeffs ^= 1,
                _ => c.layout.weights.blinding_rows ^= 1,
            }
            assert_eq!(
                weights::check_public(&c, &public, &root, 0),
                Err(Error::Layout)
            );
            assert_eq!(
                weights::check_public(&c, &public, &wrong, 0),
                Err(Error::Usage)
            );
        }
    }
}

#[test]
fn the_smallest_proximity_failure_is_chosen_across_both_combined_rows() {
    let file = read_vectors("weights-opening/openings.json");
    let mut checked = 0;
    for j in list(&file, "cases") {
        for fault in list(j, "ordered_faults") {
            let mut c = Case::read(j);
            c.value = fp2(member(fault, "value"));
            c.combined = list(fault, "combined")
                .iter()
                .map(|row| CombinedRow {
                    u: pairs(row),
                    mask: vec![],
                })
                .collect();
            assert!(
                weights::check_evaluations(c.opening(), &c.combined).is_ok(),
                "{}",
                c.name
            );
            assert_eq!(
                c.check(),
                Err(Error::Proximity {
                    matrix: c.layout.matrix(),
                    column: u32::try_from(number(fault, "first_failing_column")).unwrap()
                }),
                "{}",
                c.name
            );
            checked += 1;
        }
    }
    assert_eq!(
        checked, 10,
        "two fault orders on every non-scalar nonempty case"
    );
}

#[test]
fn an_authenticated_proper_subset_does_not_answer_every_query() {
    let file = read_vectors("weights-opening/openings.json");
    let mut checked = 0;
    for j in list(&file, "cases") {
        if list(j, "combined").is_empty() {
            continue;
        }
        let mut c = Case::read(j);
        let omitted = member(j, "omitted_columns");
        c.columns = list(omitted, "columns")
            .iter()
            .map(|col| col.as_array().unwrap().iter().map(fp).collect())
            .collect();
        c.siblings = list(omitted, "siblings").iter().map(digest).collect();
        assert_eq!(c.check(), Err(c.merkle_error()), "{}", c.name);
        // The same proof is valid when the verifier really requested only the answered subset.
        let last = c.queries.iter().max().copied().unwrap();
        c.queries.retain(|&j| j != last);
        assert_eq!(
            c.check(),
            Ok(()),
            "{}: subset authentication control",
            c.name
        );
        checked += 1;
    }
    assert_eq!(checked, 6);
}

#[test]
fn authenticated_and_consistent_columns_still_have_the_derived_row_count() {
    let file = read_vectors("weights-opening/openings.json");
    let mut checked = 0;
    for j in list(&file, "cases") {
        if list(j, "combined").is_empty() {
            continue;
        }
        let mut c = Case::read(j);
        let short = member(j, "short_columns");
        c.root = digest(member(short, "root"));
        c.value = fp2(member(short, "value"));
        c.combined = list(short, "combined")
            .iter()
            .map(|row| CombinedRow {
                u: pairs(row),
                mask: vec![],
            })
            .collect();
        c.columns = list(short, "columns")
            .iter()
            .map(|col| col.as_array().unwrap().iter().map(fp).collect())
            .collect();
        c.siblings = list(short, "siblings").iter().map(digest).collect();
        assert!(weights::check_evaluations(c.opening(), &c.combined).is_ok());
        assert_eq!(c.check(), Err(c.merkle_error()), "{}", c.name);
        checked += 1;
    }
    assert_eq!(checked, 6);
}
