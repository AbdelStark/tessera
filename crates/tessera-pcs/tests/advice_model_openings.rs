// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0011/0037 original-claim powers and weighted images, frozen before the model verifier.

mod common;

use std::{fs, num::NonZeroUsize, path::Path};

use common::{Fp, Replay};
use serde_json::Value;
use tessera_codec::Secret;
use tessera_field::{Field, GoldilocksExt2 as Fp2, PrimeField, QuadraticExtension};
use tessera_pcs::{ClaimGroup, CommitmentParams, ReferenceCommitment, RowClaim, RowLayout};

fn vectors() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../spec/vectors/advice-opening/opening.json");
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

fn list(value: &Value) -> &[Value] {
    value.as_array().unwrap()
}
fn number(value: &Value) -> u64 {
    value.as_u64().unwrap()
}
fn fp(value: &Value) -> Fp {
    let text = value.as_str().unwrap().strip_prefix("0x").unwrap();
    Fp::from_canonical_u64(u64::from_str_radix(text, 16).unwrap()).unwrap()
}
fn fp2(value: &Value) -> Fp2 {
    Fp2::new(fp(&value[0]), fp(&value[1]))
}
fn fp2s(value: &Value) -> Vec<Fp2> {
    list(value).iter().map(fp2).collect()
}
fn digest(value: &Value) -> [u8; 32] {
    let s = value.as_str().unwrap();
    std::array::from_fn(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
}

#[test]
fn the_reference_producer_matches_every_complete_model_opening_and_coefficient_digest() {
    let vectors = vectors();
    assert_eq!(list(&vectors["layouts"]).len(), 54);
    assert_eq!(list(&vectors["openings"]).len(), 3);
    for case in list(&vectors["openings"]) {
        let k = 1usize << number(&case["log2_k"]);
        let segments: Vec<Vec<u64>> = list(&case["segments"])
            .iter()
            .map(|part| list(part).iter().map(number).collect())
            .collect();
        let indices: Vec<_> = segments.iter().flatten().copied().collect();
        let layout = RowLayout::new(number(&case["row_vars"]) as u32, segments).unwrap();
        let omega = common::root(k);
        let mut values = Vec::new();
        let mut masks = Vec::new();
        for &row in &indices {
            let c = Fp::from_canonical_u64(17 * row + 5).unwrap();
            let d = Fp::from_canonical_u64(row * row + 3).unwrap();
            let mut power = Fp::ONE;
            for _ in 0..k {
                values.push(c + d * power);
                power *= omega;
            }
            for i in 0..number(&case["mask_coeffs"]) {
                masks.push(Fp::from_canonical_u64((row + 1) * (i + 1) + 7).unwrap());
            }
        }
        let groups: Vec<_> = list(&case["groups"])
            .iter()
            .map(|g| ClaimGroup {
                column_point: fp2s(&g["column_point"]),
                claims: list(&g["claims"])
                    .iter()
                    .map(|c| RowClaim {
                        row_point: fp2s(&c["row_point"]),
                        gamma: fp2(&c["gamma"]),
                    })
                    .collect(),
            })
            .collect();
        let xi = fp2s(&case["xi_point"]);
        let coefficients = layout.row_coefficients(&groups, &xi).unwrap();
        assert_eq!(coefficients.len(), list(&case["coefficient_sha256"]).len());
        for (values, expected) in coefficients.iter().zip(list(&case["coefficient_sha256"])) {
            let bytes: Vec<_> = values
                .iter()
                .flat_map(|v| v.coeffs())
                .flat_map(|c| c.to_canonical_u64().to_le_bytes())
                .collect();
            assert_eq!(tv_hash::sha256::sha256(&bytes), digest(expected));
        }
        let params = CommitmentParams {
            log2_k: number(&case["log2_k"]) as u32,
            log2_inv_rate: 1,
            mask_coeffs: number(&case["mask_coeffs"]) as usize,
            row_block: NonZeroUsize::new(number(&case["row_block"]) as usize).unwrap(),
            commitment_id: digest(&case["commitment_id"]),
        };
        let mut source = Replay::new(masks);
        let committed =
            ReferenceCommitment::commit(params, layout, Secret::new(values), &mut source).unwrap();
        assert_eq!(committed.root(), digest(&case["root"]), "{}", case["name"]);
        assert_eq!(source.drawn, indices.len() * params.mask_coeffs);
        let combined = committed.open(&groups, &xi).unwrap();
        assert_eq!(combined.len(), list(&case["combined"]).len());
        for (actual, expected) in combined.iter().zip(list(&case["combined"])) {
            assert_eq!(actual.u, fp2s(&expected["u"]));
            assert_eq!(actual.mask, fp2s(&expected["mask"]));
        }
        let queries: Vec<_> = list(&case["queries"]).iter().map(number).collect();
        let answer = committed.answer(&queries).unwrap();
        let expected: Vec<Vec<_>> = list(&case["columns"])
            .iter()
            .map(|column| list(column).iter().map(fp).collect())
            .collect();
        assert_eq!(answer.columns, expected);
        assert_eq!(
            answer.proof.siblings,
            list(&case["siblings"])
                .iter()
                .map(digest)
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn canonical_verifier_counts_and_lazy_coefficients_allocate_nothing() {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/claims/claim-lists.json");
    let golden: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    let case = list(&golden["cases"])
        .iter()
        .find(|case| case["model"] == "tiny-2-servable.d")
        .unwrap();
    let hex = case["hex"].as_str().unwrap();
    let bytes: Vec<_> = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect();
    let claims =
        tv_core::claims::ClaimList::open(&bytes, &tv_codec::DecodeLimits::MESSAGE).unwrap();
    let public = tv_core::validation::check_public(&claims).unwrap();
    let graph = tv_core::lowering::lower(&public, 5).unwrap();
    let selected = tv_core::selection::derive(&graph, claims.gadget_policy).unwrap();
    let mut config = tv_core::advice::Config::new(&claims.arch, claims.gadget_policy, 3, 8);
    config.query_block = 2;
    let layout = tv_core::advice::Layout::derive(&graph, &selected, config, 5).unwrap();
    assert!(
        layout.rows() > 1 << 20,
        "complete native table metadata, never a resident matrix"
    );
    let mut samples = Vec::new();
    for tile in layout.tiles() {
        for column in 0..tile.columns() {
            let location = tile.location(column).unwrap();
            let variables = if matches!(tile.key(), tv_core::advice::TileKey::Attention(..)) {
                claims.arch.n_heads.next_power_of_two().ilog2() + if column == 0 { 6 } else { 3 }
            } else {
                location
                    .domain
                    .shape()
                    .iter()
                    .map(|n| n.next_power_of_two().ilog2())
                    .sum()
            };
            let point: Vec<_> = (0..variables)
                .map(|i| {
                    tv_field::Fp2::new(tv_field::Fp::reduce(13 + u64::from(i)), tv_field::Fp::ONE)
                })
                .collect();
            let rows = [
                location.base,
                location.base + location.zero_at.saturating_sub(1),
                location.base + location.domain.rows() - 1,
                0,
                u64::MAX,
            ];
            samples.push((location.source, point, rows));
        }
    }
    // Prove that the current-thread counter observes allocations before requiring zero.
    let mut control = None;
    let observed = allocation_counter::measure(|| control = Some(vec![3u64; 128]));
    assert!(observed.bytes_max >= 1024);
    drop(control);
    let measured = allocation_counter::measure(|| {
        std::hint::black_box(layout.committed_rows().unwrap());
        for segment in 0..8 {
            std::hint::black_box(layout.committed_rows_in_segment(segment).unwrap());
        }
        for (source, point, rows) in &samples {
            let images = layout.claim(*source, point).unwrap();
            for &row in rows {
                std::hint::black_box(images.row_coefficient(row).unwrap());
            }
        }
    });
    assert_eq!(
        (
            measured.count_total,
            measured.bytes_total,
            measured.bytes_max
        ),
        (0, 0, 0)
    );
}
