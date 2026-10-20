// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Independent RFC-0012/0040/0049 vectors and isolated proof faults; no prover crate.

mod claim_support;
mod openings;

use claim_support::{at, golden, integer};
use openings::{fp, fp2, list, member, number};
use tv_core::claims::{ClaimList, Site, WeightType};
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
fn weight_type(tag: u64) -> WeightType {
    match tag {
        0 => WeightType::Embed,
        1 => WeightType::Q,
        2 => WeightType::K,
        3 => WeightType::V,
        4 => WeightType::O,
        5 => WeightType::Gate,
        6 => WeightType::Up,
        7 => WeightType::Down,
        8 => WeightType::LmHead,
        _ => panic!("weight type tag {tag}"),
    }
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
struct OwnedClaim {
    row: Vec<Fp2>,
    column: Vec<Fp2>,
    gamma: Fp2,
    value: Fp2,
}
#[derive(Clone)]
struct Case {
    name: String,
    layout: Layout,
    root: Digest,
    claims: Vec<OwnedClaim>,
    xi: Vec<Fp2>,
    combined: Vec<CombinedRow>,
    queries: Vec<u64>,
    columns: Vec<Vec<Fp>>,
    siblings: Vec<Digest>,
}
impl Case {
    fn read(j: &Json) -> Self {
        let claim_list = fixture(j);
        let layouts =
            weights::derive(&claim_list, u32::try_from(number(j, "epoch")).unwrap()).unwrap();
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
        let gamma = fp2(member(j, "gamma"));
        let claims: Vec<OwnedClaim> = list(j, "claims")
            .iter()
            .map(|c| OwnedClaim {
                row: pairs(member(c, "row_point")),
                column: pairs(member(c, "column_point")),
                gamma,
                value: fp2(member(c, "value")),
            })
            .collect();
        for (site, claim) in list(j, "site_claims").iter().zip(&claims) {
            let ty = weight_type(number(site, "matrix"));
            let ch_pt = pairs(member(site, "channel_point"));
            let lay_pt = pairs(member(site, "layer_point"));
            let (col_pt, row_pt) = layout
                .channel_point(&claim_list, ty, &ch_pt, &lay_pt)
                .unwrap();
            assert_eq!(col_pt, claim.column);
            assert_eq!(row_pt, claim.row);
        }
        Self {
            name: member(j, "name").as_str().unwrap().into(),
            layout,
            root: digest(member(j, "root")),
            claims,
            xi: pairs(member(j, "xi_point")),
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
    fn borrowed_claims(&self) -> Vec<Claim<'_>> {
        self.claims
            .iter()
            .map(|c| Claim {
                row_point: &c.row,
                column_point: &c.column,
                gamma: c.gamma,
                value: c.value,
            })
            .collect()
    }
    fn opening<'a>(&'a self, claims: &'a [Claim<'a>]) -> Opening<'a> {
        Opening {
            layout: &self.layout,
            root: self.root,
            xi_point: &self.xi,
            claims,
        }
    }
    fn check(&self) -> Result<(), Error> {
        let claims = self.borrowed_claims();
        weights::check_evaluations(self.opening(&claims), &self.combined)?.check_columns(
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
    assert_eq!(cases.len(), 8);
    for c in cases {
        assert_eq!(c.check(), Ok(()), "{}", c.name);
    }
}
#[test]
fn value_and_combined_row_shape_fail_at_evaluation() {
    for honest in cases().into_iter().filter(|c| !c.combined.is_empty()) {
        for claim_idx in 0..honest.claims.len() {
            let mut c = honest.clone();
            c.claims[claim_idx].value += Fp2::ONE;
            assert_eq!(
                c.check(),
                Err(c.eval_error()),
                "{} value{claim_idx}",
                c.name
            );
            let mut c = honest.clone();
            c.combined[claim_idx].u[0] += Fp2::ONE;
            assert_eq!(c.check(), Err(c.eval_error()), "{} row{claim_idx}", c.name);
        }
        for fault in 0..7 {
            let mut c = honest.clone();
            let last = c.combined.len() - 1;
            match fault {
                0 => {
                    c.combined[0].u.pop();
                }
                1 => c.combined[last].mask.push(Fp2::ZERO),
                2 => {
                    c.combined.pop();
                }
                3 => c.combined.push(c.combined[0].clone()),
                4 => c.combined[0].mask.push(Fp2::ZERO),
                5 => {
                    c.combined[last].u.pop();
                }
                _ => c.combined[last].u.push(Fp2::ZERO),
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
        let last = c.combined.len() - 1;
        c.combined[last].u[0] += Fp2::ONE;
        let claims = c.borrowed_claims();
        assert!(weights::check_evaluations(c.opening(&claims), &c.combined).is_ok());
        assert_eq!(c.check(), Err(c.proximity_error()), "{} proximity", c.name);
        for claim_idx in 0..honest.claims.len() {
            if !honest.claims[claim_idx].column.is_empty() {
                let mut c = honest.clone();
                let r = c.claims[claim_idx].column[0];
                c.combined[claim_idx].u[0] += r;
                c.combined[claim_idx].u[1] -= Fp2::ONE - r;
                let claims = c.borrowed_claims();
                assert!(weights::check_evaluations(c.opening(&claims), &c.combined).is_ok());
                assert_eq!(
                    c.check(),
                    Err(c.proximity_error()),
                    "{} evaluation row {claim_idx}",
                    c.name
                );
            }
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
    c.claims[0].value = Fp2::ONE;
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
    for part in 0..5 {
        let mut c = honest.clone();
        match part {
            0 => c.claims[0].column.push(Fp2::ZERO),
            1 => c.claims[0].row.push(Fp2::ZERO),
            2 => c.xi.push(Fp2::ZERO),
            3 => c.claims.clear(),
            _ => c.claims.push(c.claims[0].clone()),
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
            for (claim, val) in c.claims.iter_mut().zip(list(fault, "values")) {
                claim.value = fp2(val);
            }
            c.combined = list(fault, "combined")
                .iter()
                .map(|row| CombinedRow {
                    u: pairs(row),
                    mask: vec![],
                })
                .collect();
            let claims = c.borrowed_claims();
            assert!(
                weights::check_evaluations(c.opening(&claims), &c.combined).is_ok(),
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
        checked, 12,
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
    assert_eq!(checked, 7);
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
        for (claim, val) in c.claims.iter_mut().zip(list(short, "values")) {
            claim.value = fp2(val);
        }
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
        let claims = c.borrowed_claims();
        assert!(weights::check_evaluations(c.opening(&claims), &c.combined).is_ok());
        assert_eq!(c.check(), Err(c.merkle_error()), "{}", c.name);
        checked += 1;
    }
    assert_eq!(checked, 7);
}

#[test]
fn channel_point_refuses_non_channel_layout_unlisted_type_and_bad_lengths() {
    let claims = golden("tiny-2-servable");
    let layouts = weights::derive(&claims, 0).unwrap();
    let q_layout = &layouts[Matrix::Q.tag() as usize];
    let cp_layout = &layouts[Matrix::ChannelParams.tag() as usize];
    assert_eq!(
        q_layout.channel_point(&claims, WeightType::Q, &[Fp2::ZERO; 6], &[Fp2::ZERO]),
        Err(Error::Internal)
    );
    assert_eq!(
        cp_layout.channel_point(&claims, WeightType::Embed, &[Fp2::ZERO; 6], &[]),
        Err(Error::Internal)
    );
    assert_eq!(
        cp_layout.channel_point(&claims, WeightType::Q, &[Fp2::ZERO; 5], &[Fp2::ZERO]),
        Err(Error::Internal)
    );
    assert_eq!(
        cp_layout.channel_point(&claims, WeightType::Q, &[Fp2::ZERO; 6], &[]),
        Err(Error::Internal)
    );
    let (col_pt, row_pt) = cp_layout
        .channel_point(&claims, WeightType::Q, &[Fp2::ZERO; 6], &[Fp2::ZERO])
        .unwrap();
    assert_eq!(col_pt.len() as u32, cp_layout.log2_k());
    assert_eq!(row_pt.len() as u32, cp_layout.row_vars());
}
