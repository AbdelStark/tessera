// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Independent canonical layout and commitment vectors (RFC-0011/0046/0047).

mod common;

use std::fs;
use std::num::NonZeroUsize;
use std::path::Path;

use common::{Fp, Replay};
use serde_json::Value;
use tessera_codec::{Secret, encode};
use tessera_field::PrimeField;
use tessera_pcs::{
    AdviceLayout, AdviceLayoutParams, Axis, CommitmentDescriptor, CommitmentParams, DenseShape,
    LayoutParams, Padding, ReferenceCommitment, TensorShape, TileShape,
};

fn vectors() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../spec/vectors/pcs-streaming/streaming.json");
    serde_json::from_str(&fs::read_to_string(path).expect("vectors")).expect("JSON")
}

fn list(value: &Value) -> &[Value] {
    value.as_array().expect("array")
}

fn number(value: &Value) -> u32 {
    u32::try_from(value.as_u64().expect("integer")).expect("u32")
}

fn bytes(value: &Value) -> Vec<u8> {
    let text = value.as_str().expect("hex");
    assert!(text.len().is_multiple_of(2));
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).expect("hex byte"))
        .collect()
}

fn elements(value: &Value) -> Vec<Fp> {
    list(value)
        .iter()
        .map(|x| {
            let text = x.as_str().expect("hex field element");
            let x = u64::from_str_radix(text.strip_prefix("0x").expect("prefix"), 16).expect("u64");
            Fp::from_canonical_u64(x).expect("canonical")
        })
        .collect()
}

fn layout(case: &Value) -> AdviceLayout {
    let p = &case["layout"];
    let params = LayoutParams {
        log2_k: number(&p["log2_k"]),
        segments: number(&p["segments"]),
        layers: number(&p["layers"]),
        group: number(&p["group"]),
        query_block: number(&p["query_block"]),
        max_context: number(&p["max_context"]),
    };
    let tiles: Vec<_> = list(&case["tiles"])
        .iter()
        .map(|tile| TileShape {
            layer: tile["layer"]
                .as_u64()
                .map(|l| u32::try_from(l).expect("layer")),
            values: list(&tile["values"])
                .iter()
                .map(|value| {
                    if let Some(shape) = value.get("dense") {
                        TensorShape::Dense(DenseShape::new(
                            list(shape)
                                .iter()
                                .map(|axis| Axis::new(u64::from(number(axis))))
                                .collect(),
                            Padding::Zero,
                        ))
                    } else {
                        TensorShape::Scores {
                            block: number(&value["score"]),
                        }
                    }
                })
                .collect(),
        })
        .collect();
    AdviceLayout::new(params, number(&case["positions"]), &tiles).expect("canonical layout")
}

fn params(case: &Value, layout: &AdviceLayout) -> CommitmentParams {
    let descriptor = CommitmentDescriptor::Advice {
        statement_digest: bytes(&case["statement_digest"]).try_into().expect("digest"),
        unit: number(&case["unit"]),
        layout: AdviceLayoutParams {
            log2_k: u8::try_from(layout.params().log2_k).expect("log k"),
            log2_inv_rate: 1,
            mask_coeffs: number(&case["mask_coeffs"]),
            row_block: number(&case["row_block"]),
            segments: layout.params().segments,
        },
        rows: layout.rows(),
    };
    assert_eq!(encode(&descriptor), bytes(&case["descriptor_tce"]));
    assert_eq!(
        descriptor.commitment_id().as_slice(),
        bytes(&case["commitment_id"])
    );
    CommitmentParams {
        log2_k: layout.params().log2_k,
        log2_inv_rate: 1,
        mask_coeffs: number(&case["mask_coeffs"]) as usize,
        row_block: NonZeroUsize::new(number(&case["row_block"]) as usize).expect("B"),
        commitment_id: descriptor.commitment_id(),
    }
}

#[test]
fn the_independent_layouts_and_batch_roots_agree() {
    let file = vectors();
    let cases = list(&file["cases"]);
    assert_eq!(cases.len(), 16);
    for case in cases {
        let name = case["name"].as_str().expect("name");
        let layout = layout(case);
        assert_eq!(
            layout.allocated_rows(),
            case["allocated_rows"].as_u64().expect("allocated"),
            "{name}"
        );
        assert_eq!(layout.row_vars(), number(&case["row_vars"]), "{name}");
        let rows = layout.row_layout().expect("reference row indices");
        assert_eq!(
            serde_json::to_value(rows.segments()).expect("indices"),
            case["segments"],
            "{name}"
        );
        let values = list(&case["rows"]).iter().flat_map(elements).collect();
        let masks: Vec<_> = list(&case["masks"]).iter().flat_map(elements).collect();
        let count = masks.len();
        let mut source = Replay::new(masks);
        let reference = ReferenceCommitment::commit(
            params(case, &layout),
            rows,
            Secret::new(values),
            &mut source,
        )
        .expect("reference commitment");
        assert_eq!(source.drawn, count, "{name}: masks once, in layout order");
        assert_eq!(reference.root().as_slice(), bytes(&case["root"]), "{name}");
        let leaves: Vec<tessera_hash::Digest> = list(&case["leaves"])
            .iter()
            .map(|v| bytes(v).try_into().expect("leaf"))
            .collect();
        assert_eq!(reference.leaves(), leaves, "{name}");
    }
}
