// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Conformance with `spec/vectors/pcs/opening.json`, written by an independent oracle of
//! RFC-0011 (`generate.py`, Python big integers, no fast transform). Every honest opening
//! carries its `witness`, the rows and masks it commits. From the witness and the opening's
//! public inputs, the prover must reproduce every other value of the opening, as the README's
//! 'Reproducing an opening' lists them:
//!
//! - the root, and the leaves of the queried columns (`intermediate.leaves`);
//! - the answer: the distinct queried columns in increasing order, and their multiproof;
//! - the row coefficients `c_g` and `ξ_row` (`intermediate.row_coefficients`) and the combined
//!   rows, groups in order and then the proximity row;
//! - the points `x_j = 7·ω_n^j` (`intermediate.points`), and each combined row's
//!   `enc(u)(x_j) + Z_H(x_j)·M(x_j)` (`intermediate.combined_at_queries`), from the combined row
//!   encoded as a row is, one `Fp2` coordinate at a time.
//!
//! Each claim's value must equal the matrix's multilinear extension at its point, computed here
//! in `tv-field`'s arithmetic, and the independent verifier must accept every opening as the
//! prover produced it.

mod common;

use std::collections::BTreeSet;
use std::fs;
use std::num::NonZeroUsize;
use std::path::Path;

use common::openings::{Matrix, Verifier, tv_columns, tv_combined, tv_fp2, tv_fp2s, tv_groups};
use common::{Fp, Replay};
use serde_json::Value;
use tessera_codec::Secret;
use tessera_field::{Field, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_hash::Digest;
use tessera_pcs::{
    COSET_SHIFT, ClaimGroup, CombinedRow, CommitmentParams, ReferenceCommitment, RowClaim,
    RowEncoder, RowLayout,
};

type Fp2 = GoldilocksExt2;

fn load() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/pcs/opening.json");
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).expect("the vector file is JSON")
}

fn list(value: &Value) -> &[Value] {
    value.as_array().expect("a list")
}

fn number(value: &Value) -> u64 {
    value.as_u64().expect("a number")
}

fn fp(value: &Value) -> Fp {
    let text = value.as_str().expect("a hex string");
    let word = u64::from_str_radix(text.strip_prefix("0x").expect("0x"), 16).expect("hex");
    Fp::from_canonical_u64(word).expect("canonical")
}

fn fps(value: &Value) -> Vec<Fp> {
    list(value).iter().map(fp).collect()
}

fn fp2(value: &Value) -> Fp2 {
    let pair = list(value);
    assert_eq!(pair.len(), 2, "an Fp2 element is [c0, c1]");
    Fp2::new(fp(&pair[0]), fp(&pair[1]))
}

fn fp2s(value: &Value) -> Vec<Fp2> {
    list(value).iter().map(fp2).collect()
}

fn digest(value: &Value) -> Digest {
    let text = value.as_str().expect("a hex digest");
    assert_eq!(text.len(), 64);
    core::array::from_fn(|i| u8::from_str_radix(&text[2 * i..2 * i + 2], 16).expect("hex"))
}

fn digests(value: &Value) -> Vec<Digest> {
    list(value).iter().map(digest).collect()
}

/// The groups as the prover takes them: points and `γ`, without the values.
fn groups(case: &Value) -> Vec<ClaimGroup> {
    list(&case["groups"])
        .iter()
        .map(|group| ClaimGroup {
            column_point: fp2s(&group["column_point"]),
            claims: list(&group["claims"])
                .iter()
                .map(|claim| RowClaim {
                    row_point: fp2s(&claim["row_point"]),
                    gamma: fp2(&claim["gamma"]),
                })
                .collect(),
        })
        .collect()
}

/// `enc(u) + Z_H·M` on `7·L`, each `Fp2` coordinate encoded as a row is, with its mask.
fn encode_combined(row: &CombinedRow) -> Vec<Fp2> {
    let encoder = RowEncoder::new(row.u.len(), 1, row.mask.len()).expect("valid shape");
    let coordinate = |values: &[Fp2], c: usize| -> Vec<Fp> {
        values.iter().map(|value| value.coeffs()[c]).collect()
    };
    let [c0, c1] = [0, 1].map(|c| {
        let mut codeword = vec![Fp::ZERO; encoder.codeword_len()];
        let mut mask = Replay::new(coordinate(&row.mask, c));
        encoder
            .encode_row(&coordinate(&row.u, c), &mut mask, &mut codeword)
            .expect("matching lengths");
        codeword
    });
    c0.into_iter()
        .zip(c1)
        .map(|(a, b)| Fp2::new(a, b))
        .collect()
}

#[test]
fn the_file_is_for_goldilocks_its_extension_and_the_coset_seven_l() {
    let file = load();
    let modulus = file["modulus"].as_str().expect("hex");
    assert_eq!(
        u64::from_str_radix(modulus.strip_prefix("0x").expect("0x"), 16).expect("hex"),
        Fp::MODULUS
    );
    assert_eq!(number(&file["non_residue"]), 7);
    assert_eq!(number(&file["coset_shift"]), COSET_SHIFT.to_canonical_u64());
}

#[test]
fn every_honest_opening_is_reproduced_value_by_value_and_verifies() {
    let file = load();
    let openings = list(&file["openings"]);
    assert_eq!(openings.len(), 10, "the README lists 10 honest openings");
    for case in openings {
        let name = case["name"].as_str().expect("a name");
        assert_eq!(number(&case["log2_inv_rate"]), 1, "{name}: rate 1/2");
        let params = CommitmentParams {
            log2_k: u32::try_from(number(&case["log2_k"])).expect("small"),
            log2_inv_rate: 1,
            mask_coeffs: number(&case["mask_coeffs"]) as usize,
            row_block: NonZeroUsize::new(number(&case["row_block"]) as usize).expect("B ≥ 1"),
            commitment_id: digest(&case["commitment_id"]),
        };
        let (k, p) = (1usize << params.log2_k, params.mask_coeffs);
        let segments: Vec<Vec<u64>> = list(&case["segments"])
            .iter()
            .map(|segment| list(segment).iter().map(number).collect())
            .collect();
        let layout = RowLayout::new(
            u32::try_from(number(&case["row_vars"])).expect("small"),
            segments,
        )
        .expect("a valid layout");
        let n_rows = layout.rows();

        // The witness: one row of k values and one mask of P_A coefficients per non-zero row.
        let witness = &case["witness"];
        let rows: Vec<Vec<Fp>> = list(&witness["rows"]).iter().map(fps).collect();
        let masks: Vec<Vec<Fp>> = list(&witness["masks"]).iter().map(fps).collect();
        assert_eq!((rows.len(), masks.len()), (n_rows, n_rows), "{name}");
        assert!(rows.iter().all(|row| row.len() == k), "{name}");
        assert!(masks.iter().all(|mask| mask.len() == p), "{name}");
        let values = rows.concat();
        let mut source = Replay::new(masks.concat());
        let commitment =
            ReferenceCommitment::commit(params, layout, Secret::new(values.clone()), &mut source)
                .expect("valid inputs");
        let layout = commitment.layout();
        assert_eq!(source.drawn, n_rows * p, "{name}: P_A draws per row");

        assert_eq!(commitment.root(), digest(&case["root"]), "{name}: root");

        let queries: Vec<u64> = list(&case["queries"]).iter().map(number).collect();
        let distinct: Vec<u64> = queries
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let answer = commitment.answer(&queries).expect("valid queries");
        let columns: Vec<Vec<Fp>> = list(&case["columns"]).iter().map(fps).collect();
        assert!(answer.columns == columns, "{name}: columns");
        assert_eq!(
            answer.proof.siblings,
            digests(&case["siblings"]),
            "{name}: siblings"
        );

        let intermediate = &case["intermediate"];
        let leaves: Vec<Digest> = distinct
            .iter()
            .map(|&j| commitment.leaves()[j as usize])
            .collect();
        assert_eq!(leaves, digests(&intermediate["leaves"]), "{name}: leaves");

        let groups = groups(case);
        let xi_point = fp2s(&case["xi_point"]);
        let coefficients = layout
            .row_coefficients(&groups, &xi_point)
            .expect("valid points");
        let expected: Vec<Vec<Fp2>> = list(&intermediate["row_coefficients"])
            .iter()
            .map(fp2s)
            .collect();
        assert!(coefficients == expected, "{name}: row coefficients");

        let combined = commitment.open(&groups, &xi_point).expect("valid points");
        let expected: Vec<CombinedRow> = list(&case["combined"])
            .iter()
            .map(|row| CombinedRow {
                u: fp2s(&row["u"]),
                mask: fp2s(&row["mask"]),
            })
            .collect();
        assert_eq!(combined.len(), groups.len() + 1, "{name}");
        assert!(combined == expected, "{name}: combined rows");

        // x_j = 7·ω_n^j, from the definition of ω_n.
        let n = 2 * k;
        let omega = Fp::from_u64_reduced(7).pow((Fp::MODULUS - 1) / n as u64);
        let points: Vec<Fp> = distinct
            .iter()
            .map(|&j| Fp::from_u64_reduced(7) * omega.pow(j))
            .collect();
        assert!(points == fps(&intermediate["points"]), "{name}: points");

        let at_queries: Vec<Vec<Fp2>> = combined
            .iter()
            .map(|row| {
                let codeword = encode_combined(row);
                distinct.iter().map(|&j| codeword[j as usize]).collect()
            })
            .collect();
        let expected: Vec<Vec<Fp2>> = list(&intermediate["combined_at_queries"])
            .iter()
            .map(fp2s)
            .collect();
        assert!(
            at_queries == expected,
            "{name}: combined rows at the queries"
        );

        // Each claim's value is the extension of the matrix at its point.
        let matrix = Matrix {
            log2_k: params.log2_k,
            rows: layout.row_indices().collect(),
            values: &values,
        };
        let claimed: Vec<Vec<_>> = list(&case["groups"])
            .iter()
            .map(|group| {
                list(&group["claims"])
                    .iter()
                    .map(|claim| tv_fp2(fp2(&claim["value"])))
                    .collect()
            })
            .collect();
        assert!(matrix.values(&groups) == claimed, "{name}: claim values");

        // The independent verifier accepts the opening as the prover produced it.
        let tv_groups = tv_groups(&groups, &claimed);
        let xi = tv_fp2s(&xi_point);
        let verifier = Verifier {
            params: &params,
            layout,
            root: commitment.root(),
            groups: &tv_groups,
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
            "{name}: the verifier accepts"
        );
    }
}
