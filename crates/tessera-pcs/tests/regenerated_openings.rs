// SPDX-License-Identifier: Apache-2.0 OR MIT

//! P4/P5 against the resident producer and independent verifier (RFC-0011, DEP-2).
mod common;

use common::openings::{Matrix, Verifier, tv_columns, tv_combined, tv_fp2s, tv_groups};
use common::{Fp, Replay, SplitMix64, fp};
use std::num::NonZeroUsize;
use tessera_codec::{Secret, error::ErrorCode};
use tessera_field::{Field, GoldilocksExt2 as Fp2, QuadraticExtension};
use tessera_pcs::*;
use tv_core::opening::OpeningError as Reject;

fn layout(segments: u32) -> AdviceLayout {
    let dense = |a, b| {
        TensorShape::Dense(DenseShape::new(
            vec![Axis::new(a), Axis::new(b)],
            Padding::Zero,
        ))
    };
    AdviceLayout::new(
        LayoutParams {
            log2_k: 3,
            segments,
            layers: 2,
            group: 1,
            query_block: 2,
            max_context: 8,
        },
        5,
        &[
            TileShape {
                layer: None,
                values: vec![dense(3, 5)],
            },
            TileShape {
                layer: Some(0),
                values: vec![dense(5, 3), dense(2, 2)],
            },
            TileShape {
                layer: Some(1),
                values: vec![dense(7, 3)],
            },
            TileShape {
                layer: None,
                values: vec![dense(3, 3)],
            },
        ],
    )
    .unwrap()
}
fn params(block: usize) -> CommitmentParams {
    CommitmentParams {
        log2_k: 3,
        log2_inv_rate: 1,
        mask_coeffs: 3,
        row_block: NonZeroUsize::new(block).unwrap(),
        commitment_id: [71; 32],
    }
}
fn pieces(
    g: usize,
    rv: usize,
    gamma: Fp2,
    rng: &mut SplitMix64,
) -> (Vec<AdviceClaim>, Vec<ClaimGroup>) {
    let point = |i: usize| {
        vec![
            Fp2::from(fp(i as u64 + 3)),
            Fp2::from(fp(11)),
            Fp2::from(fp(13)),
        ]
    };
    let mut original = Vec::new();
    let mut groups: Vec<_> = (0..g)
        .map(|i| ClaimGroup {
            column_point: point(i),
            claims: Vec::new(),
        })
        .collect();
    let mut powers = vec![Fp2::ONE; g];
    for i in 0..40 {
        let index = i % g;
        powers[index] *= gamma;
        let mut images = Vec::new();
        for j in 0..2 {
            let weight = Fp2::from(fp(j + 2));
            let row: Vec<_> = (0..rv).map(|_| Fp2::new(rng.fp(), rng.fp())).collect();
            let mut full = point(index);
            full.extend_from_slice(&row);
            images.push(AdvicePoint {
                point: full,
                weight,
            });
            groups[index].claims.push(RowClaim {
                row_point: row,
                gamma: powers[index] * weight,
            });
        }
        original.push(AdviceClaim { pieces: images });
    }
    (original, groups)
}
fn committed<'a>(
    layout: &'a AdviceLayout,
    p: CommitmentParams,
    enc: &RowEncoder,
    values: &[Fp],
    masks: &[Fp],
) -> Vec<SegmentDigests<'a>> {
    let mut source = Replay::new(masks.to_vec());
    let mut offset = 0;
    let mut parts = Vec::new();
    for (segment, part) in layout.segments().iter().enumerate() {
        let mut stream =
            StreamingCommitter::new(p, layout, segment as u32, enc, &mut source).unwrap();
        for row in part.nonzero.iter().cloned().flatten() {
            stream
                .push_row(row, &values[offset..offset + enc.message_len()])
                .unwrap();
            offset += enc.message_len();
        }
        parts.push(stream.finish().unwrap());
    }
    assert_eq!(offset, values.len());
    assert_eq!(source.drawn, masks.len());
    parts
}

#[test]
fn every_group_count_and_segment_shape_matches_batch_and_verifies() {
    let mut rng = SplitMix64::new(119);
    for segments in [1, 2, 8] {
        let layout = layout(segments);
        let row_layout = layout.row_layout().unwrap();
        let indices: Vec<_> = row_layout.row_indices().collect();
        let p = params(5);
        let encoder = RowEncoder::new(8, 1, 3).unwrap();
        let values = rng.fps(indices.len() * 8);
        let masks = rng.fps(indices.len() * 3);
        let parts = committed(&layout, p, &encoder, &values, &masks);
        let state = AdviceOpening::new(&parts).unwrap();
        assert_eq!(state.commitment(), merge_segments(&parts).unwrap());
        let reference = ReferenceCommitment::commit(
            p,
            row_layout.clone(),
            Secret::new(values.clone()),
            &mut Replay::new(masks.clone()),
        )
        .unwrap();
        assert_eq!(state.commitment().root, reference.root());
        let queries = [15, 1, 7, 1];
        for count in 1..=16 {
            let gamma = Fp2::from(fp(17));
            let (claims, groups) = pieces(count, layout.row_vars() as usize, gamma, &mut rng);
            let xi: Vec<_> = (0..layout.row_vars())
                .map(|_| Fp2::new(rng.fp(), rng.fp()))
                .collect();
            let prepared = state.prepare(&claims, gamma, &xi).unwrap();
            assert_eq!(prepared.group_count(), count);
            let mut source = Replay::new(masks.clone());
            let mut pass = prepared.begin(&encoder, &mut source).unwrap();
            for (&row, values) in indices.iter().zip(values.chunks_exact(8)) {
                pass.push_row(row, values).unwrap();
            }
            let checked = pass.finish().unwrap();
            assert_eq!(source.drawn, masks.len());
            assert_eq!(checked.combined(), reference.open(&groups, &xi).unwrap());
            let mut source = Replay::new(masks.clone());
            let mut pass = checked.answer(&encoder, &mut source, &queries).unwrap();
            for (&row, values) in indices.iter().zip(values.chunks_exact(8)) {
                pass.push_row(row, values).unwrap();
            }
            let answer = pass.finish().unwrap();
            assert_eq!(source.drawn, masks.len());
            assert_eq!(answer, reference.answer(&queries).unwrap());
            let matrix = Matrix {
                log2_k: 3,
                rows: indices.clone(),
                values: &values,
            };
            let claims = tv_groups(&groups, &matrix.values(&groups));
            let xi = tv_fp2s(&xi);
            let verifier = Verifier {
                params: &p,
                layout: &row_layout,
                root: state.commitment().root,
                groups: &claims,
                xi_point: &xi,
            };
            let combined = tv_combined(checked.combined());
            let columns = tv_columns(&answer);
            assert_eq!(
                verifier.verify(&combined, &queries, &columns, &answer.proof.siblings),
                Ok(())
            );
            let mut bad = combined.clone();
            bad[0].u[0] += tv_field::Fp2::ONE;
            assert_eq!(
                verifier.verify(&bad, &queries, &columns, &answer.proof.siblings),
                Err(Reject::Evaluation { group: Some(0) })
            );
            let mut bad = combined.clone();
            bad[0].mask[0] += tv_field::Fp2::ONE;
            assert_eq!(
                verifier.verify(&bad, &queries, &columns, &answer.proof.siblings),
                Err(Reject::Proximity { column: 1 })
            );
            let mut bad = columns.clone();
            bad[0][0] += tv_field::Fp::ONE;
            assert_eq!(
                verifier.verify(&combined, &queries, &bad, &answer.proof.siblings),
                Err(Reject::MerklePath { column: 1 })
            );
            let mut bad = answer.proof.siblings.clone();
            bad[0][0] ^= 1;
            assert_eq!(
                verifier.verify(&combined, &queries, &columns, &bad),
                Err(Reject::MerklePath { column: 1 })
            );
        }
    }
}

#[test]
fn changed_regeneration_or_mask_stream_is_internal_before_combined_rows_exist() {
    let layout = layout(8);
    let indices: Vec<_> = layout.row_layout().unwrap().row_indices().collect();
    let p = params(5);
    let encoder = RowEncoder::new(8, 1, 3).unwrap();
    let values = vec![fp(19); indices.len() * 8];
    let masks = vec![fp(23); indices.len() * 3];
    let parts = committed(&layout, p, &encoder, &values, &masks);
    let state = AdviceOpening::new(&parts).unwrap();
    let xi = vec![Fp2::from(fp(2)); layout.row_vars() as usize];
    let prepared = state.prepare(&[], Fp2::ONE, &xi).unwrap();
    for corrupt_mask in [false, true] {
        let mut changed = masks.clone();
        if corrupt_mask {
            changed[0] += Fp::ONE;
        }
        let mut source = Replay::new(changed);
        let mut pass = prepared.begin(&encoder, &mut source).unwrap();
        for (i, (&row, value)) in indices.iter().zip(values.chunks_exact(8)).enumerate() {
            let mut changed = value.to_vec();
            if !corrupt_mask && i == 0 {
                changed[0] += Fp::ONE;
            }
            pass.push_row(row, &changed).unwrap();
        }
        let error = pass.finish().err().unwrap();
        assert_eq!(error, AdviceOpeningError::SelfCheck);
        assert_eq!(error.code(), ErrorCode::Internal);
    }
}

#[test]
fn owned_p1_retains_its_p4_permit_and_rejects_another_commitment() {
    let layout = layout(2);
    let indices: Vec<_> = layout.row_layout().unwrap().row_indices().collect();
    let encoder = RowEncoder::new(8, 1, 3).unwrap();
    let values = vec![fp(19); indices.len() * 8];
    let masks = vec![fp(23); indices.len() * 3];
    let state = {
        let parts = committed(&layout, params(5), &encoder, &values, &masks);
        AdviceOpening::new(&parts).unwrap().into_owned()
    };
    let xi = vec![Fp2::from(fp(2)); layout.row_vars() as usize];
    let prepared = state.prepare(&[], Fp2::ONE, &xi).unwrap();
    let mut source = Replay::new(masks.clone());
    let mut pass = prepared.begin(&encoder, &mut source).unwrap();
    for (&row, value) in indices.iter().zip(values.chunks_exact(8)) {
        pass.push_row(row, value).unwrap();
    }
    let (_, permit) = pass.finish().unwrap().into_parts();
    let mut different = values.clone();
    different[0] += Fp::ONE;
    let other_parts = committed(&layout, params(5), &encoder, &different, &masks);
    let other = AdviceOpening::new(&other_parts).unwrap().into_owned();
    let mut source = Replay::new(masks.clone());
    let error = other
        .begin_columns(&permit, &encoder, &mut source, &[3])
        .err()
        .unwrap();
    assert_eq!(
        error,
        AdviceOpeningError::Invalid(PcsError::LayoutData {
            rule: "P5 uses the P1 whose P4 self-check passed"
        })
    );
    assert_eq!(source.drawn, 0);
    let mut source = Replay::new(masks.clone());
    let queries = [3, 1, 3];
    let mut pass = state
        .begin_columns(&permit, &encoder, &mut source, &queries)
        .unwrap();
    for (&row, value) in indices.iter().zip(values.chunks_exact(8)) {
        pass.push_row(row, value).unwrap();
    }
    let answer = pass.finish().unwrap();
    let reference = ReferenceCommitment::commit(
        params(5),
        layout.row_layout().unwrap(),
        Secret::new(values),
        &mut Replay::new(masks),
    )
    .unwrap();
    assert_eq!(answer, reference.answer(&queries).unwrap());
}

#[test]
fn shapes_are_checked_before_draws_and_refused_rows_can_be_retried() {
    let layout = layout(8);
    let indices: Vec<_> = layout.row_layout().unwrap().row_indices().collect();
    let p = params(5);
    let encoder = RowEncoder::new(8, 1, 3).unwrap();
    let values = vec![fp(19); indices.len() * 8];
    let masks = vec![fp(23); indices.len() * 3];
    let mut parts = committed(&layout, p, &encoder, &values, &masks);
    assert_eq!(
        AdviceOpening::new(&[]).err(),
        Some(AdviceOpeningError::Invalid(PcsError::NoSegments))
    );
    assert_eq!(
        AdviceOpening::new(&parts[..7]).err(),
        Some(AdviceOpeningError::Invalid(PcsError::SegmentCount {
            expected: 8,
            found: 7
        }))
    );
    parts.swap(0, 1);
    assert_eq!(
        AdviceOpening::new(&parts).err(),
        Some(AdviceOpeningError::Invalid(PcsError::SegmentOrder {
            position: 0,
            segment: 1
        }))
    );
    parts.swap(0, 1);
    let state = AdviceOpening::new(&parts).unwrap();
    let xi = vec![Fp2::ONE; layout.row_vars() as usize];
    let wrong = state.prepare(&[], Fp2::ONE, &[]).err().unwrap();
    assert_eq!(
        wrong,
        AdviceOpeningError::Invalid(PcsError::XiPoint {
            expected: xi.len(),
            found: 0
        })
    );
    let empty = AdviceClaim { pieces: Vec::new() };
    assert!(matches!(
        state.prepare(&[empty], Fp2::ONE, &xi).err(),
        Some(AdviceOpeningError::Invalid(PcsError::LayoutData { .. }))
    ));
    let point = AdvicePoint {
        point: vec![Fp2::ONE],
        weight: Fp2::ONE,
    };
    assert_eq!(
        state
            .prepare(
                &[AdviceClaim {
                    pieces: vec![point]
                }],
                Fp2::ONE,
                &xi
            )
            .err(),
        Some(AdviceOpeningError::Invalid(PcsError::ClaimPoint {
            expected: layout.variables() as usize,
            found: 1
        }))
    );
    let mut points = vec![
        AdvicePoint {
            point: vec![Fp2::ONE; layout.variables() as usize],
            weight: Fp2::ONE
        };
        2
    ];
    points[1].point[0] = Fp2::ZERO;
    assert!(matches!(
        state
            .prepare(&[AdviceClaim { pieces: points }], Fp2::ONE, &xi)
            .err(),
        Some(AdviceOpeningError::Invalid(PcsError::LayoutData { .. }))
    ));
    let prepared = state.prepare(&[], Fp2::ONE, &xi).unwrap();
    for bad in [
        RowEncoder::new(4, 2, 3).unwrap(),
        RowEncoder::new(8, 2, 3).unwrap(),
        RowEncoder::new(8, 1, 2).unwrap(),
    ] {
        let mut source = Replay::new(masks.clone());
        assert!(matches!(
            prepared.begin(&bad, &mut source).err(),
            Some(AdviceOpeningError::Invalid(PcsError::LayoutData { .. }))
        ));
        assert_eq!(source.drawn, 0);
    }
    let mut source = Replay::new(masks.clone());
    let mut pass = prepared.begin(&encoder, &mut source).unwrap();
    let first = indices[0];
    assert_eq!(
        pass.push_row(first + 1, &[]),
        Err(AdviceOpeningError::Invalid(PcsError::StreamRow {
            segment: 0,
            expected: Some(first),
            found: Some(first + 1)
        }))
    );
    assert_eq!(
        pass.push_row(first, &[]),
        Err(AdviceOpeningError::Invalid(PcsError::RowLength {
            expected: 8,
            found: 0
        }))
    );
    for (&row, v) in indices.iter().zip(values.chunks_exact(8)) {
        pass.push_row(row, v).unwrap();
    }
    let checked = pass.finish().unwrap();
    assert_eq!(source.drawn, masks.len());
    for queries in [vec![], vec![1, 16]] {
        let mut source = Replay::new(masks.clone());
        let expected = if queries.is_empty() {
            PcsError::NoQueries
        } else {
            PcsError::QueryRange {
                position: 1,
                query: 16,
                columns: 16,
            }
        };
        assert_eq!(
            checked.answer(&encoder, &mut source, &queries).err(),
            Some(AdviceOpeningError::Invalid(expected))
        );
        assert_eq!(source.drawn, 0);
    }
    let mut source = Replay::new(masks.clone());
    let mut pass = checked.answer(&encoder, &mut source, &[15, 1, 1]).unwrap();
    assert_eq!(
        pass.push_row(first, &[]),
        Err(AdviceOpeningError::Invalid(PcsError::RowLength {
            expected: 8,
            found: 0
        }))
    );
    for (&row, v) in indices.iter().zip(values.chunks_exact(8)) {
        pass.push_row(row, v).unwrap();
    }
    let expected = AdviceOpeningError::Invalid(PcsError::StreamRow {
        segment: 8,
        expected: None,
        found: Some(u64::MAX),
    });
    assert_eq!(pass.push_row(u64::MAX, &[]), Err(expected));
    assert_eq!(pass.push_row(u64::MAX, &[]), Err(expected));
    assert_eq!(pass.finish().unwrap().columns.len(), 2);
    assert_eq!(source.drawn, masks.len());
    let mut source = Replay::new(masks.clone());
    let pass = prepared.begin(&encoder, &mut source).unwrap();
    assert_eq!(
        pass.finish().err(),
        Some(AdviceOpeningError::Invalid(PcsError::StreamRow {
            segment: 0,
            expected: Some(first),
            found: None
        }))
    );
    assert_eq!(source.drawn, 0);
    let mut source = Replay::new(masks);
    let pass = checked.answer(&encoder, &mut source, &[1]).unwrap();
    assert_eq!(
        pass.finish().err(),
        Some(AdviceOpeningError::Invalid(PcsError::StreamRow {
            segment: 0,
            expected: Some(first),
            found: None
        }))
    );
    assert_eq!(source.drawn, 0);
}

#[test]
fn empty_segments_and_zero_values_are_not_filtered_by_witness_data() {
    for empty in [false, true] {
        let layout = if empty {
            AdviceLayout::new(
                LayoutParams {
                    log2_k: 3,
                    segments: 8,
                    layers: 2,
                    group: 1,
                    query_block: 2,
                    max_context: 8,
                },
                5,
                &[
                    TileShape {
                        layer: Some(0),
                        values: vec![TensorShape::Scores { block: 3 }],
                    },
                    TileShape {
                        layer: Some(1),
                        values: vec![TensorShape::Scores { block: 3 }],
                    },
                ],
            )
            .unwrap()
        } else {
            layout(8)
        };
        let rows = layout.row_layout().unwrap();
        let indices: Vec<_> = rows.row_indices().collect();
        assert_eq!(indices.is_empty(), empty);
        let p = params(3);
        let encoder = RowEncoder::new(8, 1, 3).unwrap();
        let values = vec![Fp::ZERO; indices.len() * 8];
        let masks = vec![fp(3); indices.len() * 3];
        let parts = committed(&layout, p, &encoder, &values, &masks);
        let state = AdviceOpening::new(&parts).unwrap();
        for gamma in [Fp2::ZERO, Fp2::ONE] {
            let mut rng = SplitMix64::new(17);
            let (claims, groups) = pieces(3, layout.row_vars() as usize, gamma, &mut rng);
            let xi = vec![Fp2::from(fp(7)); layout.row_vars() as usize];
            let prepared = state.prepare(&claims, gamma, &xi).unwrap();
            let mut source = Replay::new(masks.clone());
            let mut pass = prepared.begin(&encoder, &mut source).unwrap();
            for (&row, v) in indices.iter().zip(values.chunks_exact(8)) {
                pass.push_row(row, v).unwrap();
            }
            let checked = pass.finish().unwrap();
            let mut source = Replay::new(masks.clone());
            let mut pass = checked.answer(&encoder, &mut source, &[15, 0, 15]).unwrap();
            for (&row, v) in indices.iter().zip(values.chunks_exact(8)) {
                pass.push_row(row, v).unwrap();
            }
            let answer = pass.finish().unwrap();
            let reference = ReferenceCommitment::commit(
                p,
                rows.clone(),
                Secret::new(values.clone()),
                &mut Replay::new(masks.clone()),
            )
            .unwrap();
            assert_eq!(checked.combined(), reference.open(&groups, &xi).unwrap());
            assert_eq!(answer, reference.answer(&[15, 0, 15]).unwrap());
            assert_eq!(answer.columns[0].len(), indices.len());
        }
    }
}
