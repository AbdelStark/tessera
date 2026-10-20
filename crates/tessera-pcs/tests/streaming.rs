// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Independent canonical layout and commitment vectors (RFC-0011/0046/0047).

mod common;

use std::fs;
use std::num::NonZeroUsize;
use std::path::Path;

use common::{Fp, Replay};
use serde_json::Value;
use tessera_codec::{Secret, encode};
use tessera_field::{Field, PrimeField};
use tessera_pcs::{
    AdviceLayout, AdviceLayoutParams, Axis, CommitmentDescriptor, CommitmentParams, DenseShape,
    LayoutParams, Padding, PcsError, ReferenceCommitment, RowEncoder, SegmentDigests,
    StreamingCommitter, TensorShape, TileShape, merge_segments,
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

#[test]
fn streaming_matches_every_independent_chain_and_root() {
    let file = vectors();
    for case in list(&file["cases"]) {
        let name = case["name"].as_str().expect("name");
        let layout = layout(case);
        let params = params(case, &layout);
        let encoder = RowEncoder::new(1 << params.log2_k, params.log2_inv_rate, params.mask_coeffs)
            .expect("encoder");
        let values: Vec<_> = list(&case["rows"]).iter().map(elements).collect();
        let masks: Vec<_> = list(&case["masks"]).iter().flat_map(elements).collect();
        let draws = masks.len();
        let mut source = Replay::new(masks);
        let mut at = 0;
        let mut parts = Vec::new();
        for (index, segment) in layout.segments().iter().enumerate() {
            let mut stream =
                StreamingCommitter::new(params, &layout, index as u32, &encoder, &mut source)
                    .expect("stream");
            for row in segment.nonzero.iter().cloned().flatten() {
                stream.push_row(row, &values[at]).expect("next row");
                at += 1;
            }
            let part = stream.finish().expect("complete segment");
            let expected: Vec<tessera_hash::Digest> = list(&case["segment_digests"][index])
                .iter()
                .map(|d| bytes(d).try_into().expect("digest"))
                .collect();
            assert_eq!(part.digests(), expected, "{name}: segment {index}");
            parts.push(part);
        }
        assert_eq!(at, values.len());
        assert_eq!(
            source.drawn, draws,
            "{name}: prescribed per-proof mask stream"
        );
        let commitment = merge_segments(&parts).expect("ordered complete parts");
        assert_eq!(commitment.root.as_slice(), bytes(&case["root"]), "{name}");
        assert_eq!(commitment.layout_digest, params.commitment_id, "{name}");
    }
}

fn mixed_case() -> Value {
    list(&vectors()["cases"])
        .iter()
        .find(|c| c["name"] == "mixed, S=2, B=3")
        .expect("mixed case")
        .clone()
}

fn parts<'a>(
    case: &Value,
    layout: &'a AdviceLayout,
    params: CommitmentParams,
) -> Vec<SegmentDigests<'a>> {
    let encoder = RowEncoder::new(1 << params.log2_k, params.log2_inv_rate, params.mask_coeffs)
        .expect("encoder");
    let values: Vec<_> = list(&case["rows"]).iter().map(elements).collect();
    let mut source = Replay::new(list(&case["masks"]).iter().flat_map(elements).collect());
    let mut at = 0;
    layout
        .segments()
        .iter()
        .enumerate()
        .map(|(index, segment)| {
            let mut stream =
                StreamingCommitter::new(params, layout, index as u32, &encoder, &mut source)
                    .expect("stream");
            for row in segment.nonzero.iter().cloned().flatten() {
                stream.push_row(row, &values[at]).expect("next row");
                at += 1;
            }
            stream.finish().expect("complete")
        })
        .collect()
}

#[test]
fn refused_rows_leave_the_buffer_and_mask_stream_untouched() {
    use std::cell::Cell;
    let case = mixed_case();
    let layout = layout(&case);
    let params = params(&case, &layout);
    let encoder = RowEncoder::new(1 << params.log2_k, 1, params.mask_coeffs).expect("encoder");
    let values: Vec<_> = list(&case["rows"]).iter().map(elements).collect();
    let masks: Vec<_> = list(&case["masks"]).iter().flat_map(elements).collect();
    let draws = Cell::new(0usize);
    let mut source = || {
        let at = draws.get();
        draws.set(at + 1);
        masks[at]
    };
    let indices: Vec<_> = layout.segments()[0]
        .nonzero
        .iter()
        .cloned()
        .flatten()
        .collect();
    let mut stream =
        StreamingCommitter::new(params, &layout, 0, &encoder, &mut source).expect("stream");
    // Index validation precedes length validation and any mask draw (INV-PCS-2).
    assert_eq!(
        stream.push_row(1, &[]),
        Err(PcsError::StreamRow {
            segment: 0,
            expected: Some(indices[0]),
            found: Some(1)
        })
    );
    assert_eq!(draws.get(), 0);
    assert_eq!(
        stream.push_row(indices[0], &values[0][..values[0].len() - 1]),
        Err(PcsError::RowLength {
            expected: values[0].len(),
            found: values[0].len() - 1
        })
    );
    assert_eq!(draws.get(), 0);
    stream.push_row(indices[0], &values[0]).expect("first row");
    for wrong in [indices[0], indices[2], u64::MAX] {
        assert_eq!(
            stream.push_row(wrong, &values[0]),
            Err(PcsError::StreamRow {
                segment: 0,
                expected: Some(indices[1]),
                found: Some(wrong),
            })
        );
        assert_eq!(draws.get(), params.mask_coeffs);
    }
    for (index, &row) in indices.iter().enumerate().skip(1) {
        stream.push_row(row, &values[index]).expect("correct retry");
    }
    assert_eq!(
        stream.push_row(u64::MAX, &values[0]),
        Err(PcsError::StreamRow {
            segment: 0,
            expected: None,
            found: Some(u64::MAX),
        })
    );
    assert_eq!(draws.get(), indices.len() * params.mask_coeffs);
    let part = stream.finish().expect("complete after retries");
    let expected: Vec<tessera_hash::Digest> = list(&case["segment_digests"][0])
        .iter()
        .map(|d| bytes(d).try_into().expect("digest"))
        .collect();
    assert_eq!(
        part.digests(),
        expected,
        "refused values never entered the chains"
    );
}

#[test]
fn a_premature_finish_identifies_the_first_missing_row_without_drawing() {
    let case = mixed_case();
    let layout = layout(&case);
    let params = params(&case, &layout);
    let encoder = RowEncoder::new(1 << params.log2_k, 1, params.mask_coeffs).expect("encoder");
    for supplied in [0, 1] {
        let mut source = Replay::new(list(&case["masks"]).iter().flat_map(elements).collect());
        let indices: Vec<_> = layout.segments()[0]
            .nonzero
            .iter()
            .cloned()
            .flatten()
            .collect();
        let mut stream =
            StreamingCommitter::new(params, &layout, 0, &encoder, &mut source).expect("stream");
        for (i, row) in indices.iter().enumerate().take(supplied) {
            stream
                .push_row(*row, &elements(&case["rows"][i]))
                .expect("row");
        }
        assert_eq!(
            stream.finish().map(|_| ()),
            Err(PcsError::StreamRow {
                segment: 0,
                expected: Some(indices[supplied]),
                found: None,
            })
        );
        assert_eq!(source.drawn, supplied * params.mask_coeffs);
    }
}

#[test]
fn constructor_checks_precede_mask_draws_and_large_allocations() {
    let case = mixed_case();
    let layout = layout(&case);
    let params = params(&case, &layout);
    let k = 1 << params.log2_k;
    let encoder = RowEncoder::new(k, 1, params.mask_coeffs).expect("encoder");
    let mut source = Replay::new(Vec::new());
    for (bad, expected) in [
        (
            CommitmentParams {
                log2_inv_rate: 0,
                ..params
            },
            PcsError::InverseRate { log2_inv_rate: 0 },
        ),
        (
            CommitmentParams {
                log2_k: 32,
                ..params
            },
            PcsError::TooLong { log_len: 33 },
        ),
        (
            CommitmentParams {
                mask_coeffs: 0,
                ..params
            },
            PcsError::NoMask,
        ),
        (
            CommitmentParams {
                mask_coeffs: k + 1,
                ..params
            },
            PcsError::MaskTooLarge {
                mask_coeffs: k + 1,
                message_len: k,
            },
        ),
        (
            CommitmentParams {
                log2_k: params.log2_k - 1,
                ..params
            },
            PcsError::LayoutData {
                rule: "the streaming layout uses the commitment's row width",
            },
        ),
    ] {
        assert_eq!(
            StreamingCommitter::new(bad, &layout, 0, &encoder, &mut source).map(|_| ()),
            Err(expected)
        );
    }
    for (message, rate, masks) in [
        (k / 2, 2, params.mask_coeffs),
        (k, 2, params.mask_coeffs),
        (k, 1, 1),
    ] {
        let wrong = RowEncoder::new(message, rate, masks).expect("other encoder");
        assert_eq!(
            StreamingCommitter::new(params, &layout, 0, &wrong, &mut source).map(|_| ()),
            Err(PcsError::LayoutData {
                rule: "the streaming encoder uses the commitment's k, n and P"
            })
        );
    }
    assert_eq!(
        StreamingCommitter::new(params, &layout, 2, &encoder, &mut source).map(|_| ()),
        Err(PcsError::SegmentIndex {
            segment: 2,
            segments: 2
        })
    );
    // A row block larger than the entire segment allocates only that segment's codewords.
    let huge = CommitmentParams {
        row_block: NonZeroUsize::new(usize::MAX).expect("B"),
        ..params
    };
    let stream =
        StreamingCommitter::new(huge, &layout, 0, &encoder, &mut source).expect("bounded buffer");
    assert_eq!(
        stream.finish().map(|_| ()),
        Err(PcsError::StreamRow {
            segment: 0,
            expected: Some(0),
            found: None
        })
    );
    assert_eq!(source.drawn, 0);
}

#[test]
fn merges_refuse_missing_reordered_duplicated_and_mixed_parts() {
    let case = mixed_case();
    let layout = layout(&case);
    let params = params(&case, &layout);
    let mut valid = parts(&case, &layout, params);
    assert_eq!(merge_segments(&[]), Err(PcsError::NoSegments));
    assert_eq!(
        merge_segments(&valid[..1]),
        Err(PcsError::SegmentCount {
            expected: 2,
            found: 1
        })
    );
    valid.swap(0, 1);
    assert_eq!(
        merge_segments(&valid),
        Err(PcsError::SegmentOrder {
            position: 0,
            segment: 1
        })
    );
    valid.swap(0, 1);
    valid[1] = parts(&case, &layout, params).remove(0);
    assert_eq!(
        merge_segments(&valid),
        Err(PcsError::SegmentOrder {
            position: 1,
            segment: 0
        })
    );
    for other in [
        CommitmentParams {
            commitment_id: [99; 32],
            ..params
        },
        CommitmentParams {
            row_block: NonZeroUsize::new(1).expect("B"),
            ..params
        },
        CommitmentParams {
            mask_coeffs: 1,
            ..params
        },
    ] {
        let mut valid = parts(&case, &layout, params);
        valid[1] = parts(&case, &layout, other).pop().expect("other segment 1");
        assert_eq!(
            merge_segments(&valid),
            Err(PcsError::SegmentContext { segment: 1 })
        );
    }
    let mut changed = case.clone();
    changed["layout"]["max_context"] = 8.into();
    let other_layout = self::layout(&changed);
    let mut valid = parts(&case, &layout, params);
    valid[1] = parts(&changed, &other_layout, params)
        .pop()
        .expect("other layout");
    assert_eq!(
        merge_segments(&valid),
        Err(PcsError::SegmentContext { segment: 1 })
    );
    // Equal canonical layouts need not occupy the same address.
    let equal = layout.clone();
    valid[1] = parts(&case, &equal, params).pop().expect("equal layout");
    assert!(merge_segments(&valid).is_ok());
}

#[test]
fn randomized_canonical_layouts_match_batch_for_one_two_and_eight_segments() {
    use common::SplitMix64;
    let mut rng = SplitMix64::new(0x7E55_E7A0_0115_0011);
    for segments in [1, 2, 8] {
        for iteration in 0..24 {
            let layers = 1 + rng.below(9) as u32;
            let n = 1 + rng.below(16) as u32;
            let p = LayoutParams {
                log2_k: 4,
                segments,
                layers,
                group: 1 << rng.below(2),
                query_block: 1 << rng.below(3),
                max_context: 16,
            };
            let attention = p.attention(n).expect("attention");
            let mut tiles = Vec::new();
            for layer in 0..layers {
                let padding = if rng.below(2) == 0 {
                    Padding::Zero
                } else {
                    Padding::Constants
                };
                tiles.push(TileShape {
                    layer: Some(layer),
                    values: vec![TensorShape::Dense(DenseShape::new(
                        vec![Axis::new(u64::from(n)), Axis::new(1 + rng.below(17) as u64)],
                        padding,
                    ))],
                });
                for block in [0, attention.blocks - 1] {
                    tiles.push(TileShape {
                        layer: Some(layer),
                        values: vec![
                            TensorShape::Scores { block },
                            TensorShape::Dense(attention.row_shape(block, Padding::Zero)),
                        ],
                    });
                }
            }
            let layout = AdviceLayout::new(p, n, &tiles).expect("layout");
            let params = CommitmentParams {
                log2_k: 4,
                log2_inv_rate: 1,
                mask_coeffs: 1 + rng.below(16),
                row_block: NonZeroUsize::new([1, 2, 3, 5, 32][rng.below(5)]).expect("B"),
                commitment_id: std::array::from_fn(|_| rng.next_u64() as u8),
            };
            // A structurally committed row is still encoded when its private values are all zero.
            let values = if iteration == 0 {
                vec![Fp::ZERO; layout.rows() as usize * 16]
            } else {
                rng.fps(layout.rows() as usize * 16)
            };
            let seed = rng.next_u64();
            let mut batch_rng = SplitMix64::new(seed);
            let batch = ReferenceCommitment::commit(
                params,
                layout.row_layout().expect("indices"),
                Secret::new(values.clone()),
                &mut || batch_rng.fp(),
            )
            .expect("batch");
            let encoder = RowEncoder::new(16, 1, params.mask_coeffs).expect("encoder");
            let mut stream_rng = SplitMix64::new(seed);
            let mut source = || stream_rng.fp();
            let mut parts = Vec::new();
            let mut at = 0;
            for (index, segment) in layout.segments().iter().enumerate() {
                let mut stream =
                    StreamingCommitter::new(params, &layout, index as u32, &encoder, &mut source)
                        .expect("stream");
                for row in segment.nonzero.iter().cloned().flatten() {
                    stream
                        .push_row(row, &values[at * 16..(at + 1) * 16])
                        .expect("next row");
                    at += 1;
                }
                parts.push(stream.finish().expect("complete"));
            }
            assert_eq!(merge_segments(&parts).expect("merge").root, batch.root());
            assert_eq!(
                batch_rng.next_u64(),
                stream_rng.next_u64(),
                "identical mask consumption"
            );
        }
    }
}
