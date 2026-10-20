// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Actual synchronous heap allocations: encoder/layout/input/PRG are borrowed baseline state.
//! Committer storage is one initialized codeword block plus column chains, independent of rows.
//! Segment merging adds no heap allocation. Stack state is documented separately.

#![allow(clippy::print_stderr)]

use std::num::NonZeroUsize;

use allocation_counter::measure;
use tessera_codec::SecretBuf;
use tessera_field::{Field, Goldilocks as Fp};
use tessera_hash::Digest;
use tessera_pcs::{
    AdviceLayout, Axis, CommitmentParams, DenseShape, LayoutParams, Padding, RowEncoder,
    StreamingCommitter, TensorShape, TileShape, merge_segments,
};

fn layout(log2_k: u32, segments: u32, rows_per_layer: u64) -> AdviceLayout {
    let params = LayoutParams {
        log2_k,
        segments,
        layers: 8,
        group: 1,
        query_block: 1,
        max_context: 1,
    };
    let tiles: Vec<_> = (0..8)
        .map(|layer| TileShape {
            layer: Some(layer),
            values: vec![TensorShape::Dense(DenseShape::new(
                vec![Axis::new(rows_per_layer << log2_k)],
                Padding::Zero,
            ))],
        })
        .collect();
    AdviceLayout::new(params, 1, &tiles).expect("layout")
}

#[test]
fn the_allocator_sees_one_block_and_chains_and_no_per_row_allocation() {
    // No spawned workers: allocation-counter observes the current thread only.
    let _ = tessera_hash::hash(tessera_hash::Domain::ColumnInit, &[]);
    for log_k in [0, 3, 6] {
        let k = 1usize << log_k;
        let n = 2 * k;
        let encoder = RowEncoder::new(k, 1, 1).expect("shared public tables");
        let row = SecretBuf::filled(Fp::ONE, k);
        for segments in [1, 2, 8] {
            for block in [1, 3, 5, 32] {
                let mut peaks = Vec::new();
                for rows_per_layer in [64, 512] {
                    let layout = layout(log_k, segments, rows_per_layer);
                    let params = CommitmentParams {
                        log2_k: log_k,
                        log2_inv_rate: 1,
                        mask_coeffs: 1,
                        row_block: NonZeroUsize::new(block).expect("B"),
                        commitment_id: [41; 32],
                    };
                    // Caller-owned result slots do not allocate during a measured committer.
                    let mut parts = Vec::with_capacity(segments as usize);
                    let mut source = || Fp::ONE; // synthetic test masks, never production RNG
                    for (index, segment) in layout.segments().iter().enumerate() {
                        let mut finished = None;
                        let info = measure(|| {
                            let mut stream = StreamingCommitter::new(
                                params,
                                &layout,
                                index as u32,
                                &encoder,
                                &mut source,
                            )
                            .expect("stream");
                            for row_index in segment.nonzero.iter().cloned().flatten() {
                                stream.push_row(row_index, &row).expect("next row");
                            }
                            finished = Some(stream.finish().expect("complete"));
                        });
                        let chains = n * size_of::<Digest>();
                        let expected = block * n * size_of::<Fp>() + chains;
                        assert_eq!(info.bytes_max as usize, expected, "whole committer peak");
                        assert_eq!(
                            info.bytes_total as usize, expected,
                            "no allocation per row or block"
                        );
                        assert_eq!(
                            info.bytes_current as usize, chains,
                            "only digests survive finish"
                        );
                        assert_eq!(info.count_total, 2, "block and chain allocations only");
                        peaks.push(info.bytes_max);
                        parts.push(finished.expect("part"));
                    }
                    let mut merged = None;
                    let info = measure(|| merged = Some(merge_segments(&parts).expect("merge")));
                    assert_eq!(
                        (info.bytes_max, info.bytes_total, info.count_total),
                        (0, 0, 0)
                    );
                    assert_eq!(merged.expect("root").layout_digest, params.commitment_id);
                }
                assert!(
                    peaks.iter().all(|peak| *peak == peaks[0]),
                    "eightfold longer witness keeps the same peak"
                );
                eprintln!(
                    "streaming k={k} B={block} S={segments}: worker peak={} bytes, merge heap=0",
                    peaks[0]
                );
            }
        }
    }
    // S > L leaves empty segments: their only allocation is their n initial chain digests.
    let layout = AdviceLayout::new(
        LayoutParams {
            log2_k: 3,
            segments: 8,
            layers: 1,
            group: 1,
            query_block: 1,
            max_context: 1,
        },
        1,
        &[TileShape {
            layer: Some(0),
            values: vec![TensorShape::Dense(DenseShape::new(
                vec![Axis::new(1)],
                Padding::Zero,
            ))],
        }],
    )
    .expect("empty segments");
    let encoder = RowEncoder::new(8, 1, 1).expect("encoder");
    let params = CommitmentParams {
        log2_k: 3,
        log2_inv_rate: 1,
        mask_coeffs: 1,
        row_block: NonZeroUsize::new(32).expect("B"),
        commitment_id: [42; 32],
    };
    let mut source = || -> Fp { panic!("an empty segment draws no masking randomness") };
    let mut part = None;
    let info = measure(|| {
        part = Some(
            StreamingCommitter::new(params, &layout, 7, &encoder, &mut source)
                .expect("empty stream")
                .finish()
                .expect("initial chains"),
        );
    });
    assert_eq!(
        (info.bytes_max, info.bytes_total, info.bytes_current),
        (512, 512, 512)
    );
    assert_eq!(info.count_total, 1);
    assert_eq!(part.expect("empty part").digests().len(), 16);
}
