// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Component heap measurements, not RSS: borrowed layout/encoder/input and public P1 metadata
//! are baseline. P4's peak is bounded independently of witness length; P5 includes its output
//! columns and the final protocol copy. No spawned workers escape the current-thread counter.
#![allow(clippy::print_stderr)]
use allocation_counter::measure;
use std::num::NonZeroUsize;
use tessera_codec::SecretBuf;
use tessera_field::{Field, Goldilocks as Fp, GoldilocksExt2 as Fp2, PrimeField};
use tessera_pcs::*;

#[test]
fn heap_peaks_account_for_blocks_outputs_public_tree_and_declassification() {
    let _ = tessera_hash::hash(tessera_hash::Domain::ColumnInit, &[]);
    let mut control = None;
    let measured = measure(|| control = Some(vec![1u64; 256]));
    assert!(measured.bytes_max >= 2048);
    drop(control);
    let mut peaks = std::collections::BTreeMap::<_, Vec<u64>>::new();
    for log_k in [0, 3, 6] {
        let k = 1usize << log_k;
        let n = 2 * k;
        let p = 1;
        let b = 5;
        let encoder = RowEncoder::new(k, 1, p).unwrap();
        let input = SecretBuf::filled(Fp::ONE, k);
        for segments in [1, 8] {
            for groups in [1, 16] {
                for rows_per_layer in [64, 512] {
                    let tiles: Vec<_> = (0..2)
                        .map(|layer| TileShape {
                            layer: Some(layer),
                            values: vec![TensorShape::Dense(DenseShape::new(
                                vec![Axis::new((rows_per_layer * k) as u64)],
                                Padding::Zero,
                            ))],
                        })
                        .collect();
                    let layout = AdviceLayout::new(
                        LayoutParams {
                            log2_k: log_k,
                            segments,
                            layers: 2,
                            group: 1,
                            query_block: 1,
                            max_context: 1,
                        },
                        1,
                        &tiles,
                    )
                    .unwrap();
                    let params = CommitmentParams {
                        log2_k: log_k,
                        log2_inv_rate: 1,
                        mask_coeffs: p,
                        row_block: NonZeroUsize::new(b).unwrap(),
                        commitment_id: [83; 32],
                    };
                    let mut masks = || Fp::ONE;
                    let mut parts = Vec::new();
                    for (s, part) in layout.segments().iter().enumerate() {
                        let mut pass = StreamingCommitter::new(
                            params, &layout, s as u32, &encoder, &mut masks,
                        )
                        .unwrap();
                        for r in part.nonzero.iter().cloned().flatten() {
                            pass.push_row(r, &input).unwrap();
                        }
                        parts.push(pass.finish().unwrap());
                    }
                    let mut state = None;
                    let metadata = measure(|| state = Some(AdviceOpening::new(&parts).unwrap()));
                    let tree_bytes = (2 * n - 1) * 32;
                    assert_eq!(metadata.bytes_max as usize, tree_bytes);
                    assert_eq!(metadata.bytes_current as usize, tree_bytes);
                    let state = state.unwrap();
                    // At k=1 no column coordinate distinguishes groups: one original claim per group
                    // would collapse into a single actual group, so measure the derived count below.
                    let claims: Vec<_> = (0..groups)
                        .map(|i| {
                            let mut point = vec![
                                Fp2::from(Fp::from_u64_reduced(3));
                                layout.variables() as usize
                            ];
                            if log_k > 0 {
                                point[0] = Fp2::from(Fp::from_u64_reduced(3 + i as u64));
                            }
                            AdviceClaim {
                                pieces: vec![AdvicePoint {
                                    point,
                                    weight: Fp2::ONE,
                                }],
                            }
                        })
                        .collect();
                    let xi = vec![Fp2::from(Fp::from_u64_reduced(7)); layout.row_vars() as usize];
                    let mut prepared = None;
                    let group_metadata = measure(|| {
                        prepared = Some(
                            state
                                .prepare(&claims, Fp2::from(Fp::from_u64_reduced(5)), &xi)
                                .unwrap(),
                        )
                    });
                    let prepared = prepared.unwrap();
                    assert!(group_metadata.bytes_current > 0);
                    let g = prepared.group_count() + 1;
                    let mut checked = None;
                    let p4 = measure(|| {
                        let mut masks = || Fp::ONE;
                        let mut pass = prepared.begin(&encoder, &mut masks).unwrap();
                        for part in layout.segments() {
                            for r in part.nonzero.iter().cloned().flatten() {
                                pass.push_row(r, &input).unwrap();
                            }
                        }
                        checked = Some(pass.finish().unwrap());
                    });
                    let checked = checked.unwrap();
                    peaks
                        .entry((log_k, segments, groups))
                        .or_default()
                        .push(p4.bytes_max);
                    let private = 16 * g * (k + p + 1) + 8 * b * (k + p + 2) + 8 * n + 48 * g;
                    let transient = 32 * g * b
                        + 24 * g
                        + g * 64 * 48
                        + 48 * layout.row_vars() as usize
                        + 16
                        + 16 * k;
                    let message = 16 * g * (k + p) + 48 * g;
                    let bound = private + transient + message;
                    assert!(
                        p4.bytes_max as usize <= bound,
                        "P4 N={} k={k} g={g}: observed{} > bound{bound}",
                        layout.rows(),
                        p4.bytes_max
                    );
                    assert_eq!(
                        p4.bytes_current as usize, message,
                        "only combined-row messages survive P4"
                    );
                    let queries = [0, (n - 1) as u64, 0];
                    let q = 2;
                    let mut answer = None;
                    let p5 = measure(|| {
                        let mut masks = || Fp::ONE;
                        let mut pass = checked.answer(&encoder, &mut masks, &queries).unwrap();
                        for part in layout.segments() {
                            for r in part.nonzero.iter().cloned().flatten() {
                                pass.push_row(r, &input).unwrap();
                            }
                        }
                        answer = Some(pass.finish().unwrap());
                    });
                    let output = layout.rows() as usize * q * 8;
                    let bound = 2 * output + 8 * n + 128 * q * (n.ilog2() as usize + 1) + 96 * q;
                    assert!(
                        p5.bytes_max as usize <= bound,
                        "P5 observed{} > bound{bound}",
                        p5.bytes_max
                    );
                    assert!(
                        p5.bytes_max as usize >= 2 * output,
                        "final output copy is included"
                    );
                    let answer = answer.unwrap();
                    assert_eq!(answer.columns.len(), q);
                    assert_eq!(answer.columns[0].len(), layout.rows() as usize);
                    eprintln!(
                        "opening k={k} B={b} S={segments} G={} N={}: tree={tree_bytes} groups_heap={} groups_peak={} P4_peak={} P4_message={message} P5_peak={} P5_symbols={output}",
                        g - 1,
                        layout.rows(),
                        group_metadata.bytes_current,
                        group_metadata.bytes_max,
                        p4.bytes_max,
                        p5.bytes_max
                    );
                }
            }
        }
    }
    for peaks in peaks.values() {
        assert_eq!(peaks.len(), 2);
        assert_eq!(
            peaks[0], peaks[1],
            "an eightfold longer witness has the same P4 heap peak"
        );
    }
}
