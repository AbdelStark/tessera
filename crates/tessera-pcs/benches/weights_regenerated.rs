// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Reproducible protocol/1 resident/regenerated weight heap and time measurements.
//! Synthetic INT8 input is baseline; output columns and metadata count toward peak heap.
//! `cargo bench -p tessera-pcs --bench weights_regenerated` runs five samples; a test build
//! runs one small smoke case. All roots, columns and multiproofs must be identical.

#![forbid(unsafe_code)]
#![allow(clippy::print_stdout)]

use std::hint::black_box;
use std::num::NonZeroU32;
use std::time::Instant;

use allocation_counter::measure;
use tessera_pcs::{
    ModelWeights, RegeneratedWeightCommitment, WeightCommitment, WeightLayout, WeightLayoutParams,
    WeightMatrix, WeightShape, WeightShapes, WeightTensorRef,
};

fn main() {
    let bench = std::env::args().any(|arg| arg == "--bench");
    let layers: u32 = if bench { 16 } else { 2 };
    let width: u32 = if bench { 128 } else { 16 };
    let runs = if bench { 5 } else { 1 };
    let block = NonZeroU32::new(4).unwrap();
    let mut tensors = Vec::new();
    for matrix in WeightMatrix::WEIGHTS {
        let count = if matrix.per_layer() { layers } else { 1 };
        for layer in 0..count {
            let data: Vec<i8> = (0..width * width)
                .map(|i| (i.wrapping_mul(17).wrapping_add(layer) % 255) as i8)
                .collect();
            tensors.push((matrix, layer, data));
        }
    }
    let shapes = WeightShapes {
        layers,
        shapes: [WeightShape {
            out: width,
            inputs: width,
        }; 9],
        channels: Vec::new(),
    };
    let layout = WeightLayout::new(
        WeightLayoutParams {
            log2_k: width.trailing_zeros() as u8,
            log2_inv_rate: 2,
            mask_coeffs: 64,
            blinding_rows: 32,
        },
        shapes,
    )
    .unwrap();
    let source = ModelWeights {
        tensors: tensors
            .iter()
            .map(|(matrix, layer, data)| WeightTensorRef {
                matrix: *matrix,
                layer: *layer,
                data,
            })
            .collect(),
        channels: Vec::new(),
    };
    // Warm public hashing and metric initialization before counting component allocations.
    drop(RegeneratedWeightCommitment::commit(&layout, block, 0, &source).unwrap());
    let queries: Vec<u64> = (0..32).map(|i| i * 3 % u64::from(width * 4)).collect();
    for run in 0..runs {
        let mut resident = None;
        let start = Instant::now();
        let resident_memory = measure(|| {
            resident = Some(WeightCommitment::commit(&layout, block, 0, &source).unwrap())
        });
        let resident_time = start.elapsed();
        let mut streamed = None;
        let start = Instant::now();
        let streamed_memory = measure(|| {
            streamed =
                Some(RegeneratedWeightCommitment::commit(&layout, block, 0, &source).unwrap())
        });
        let streamed_time = start.elapsed();
        let resident = resident.unwrap();
        let streamed = streamed.unwrap();
        assert_eq!(resident.public(), streamed.public());
        let mut resident_answers = None;
        let start = Instant::now();
        let resident_answer_memory = measure(|| {
            resident_answers =
                Some(WeightMatrix::ALL.map(|m| resident.matrix(m).answer(&queries).unwrap()));
        });
        let resident_answer_time = start.elapsed();
        let mut streamed_answers = None;
        let start = Instant::now();
        let streamed_answer_memory = measure(|| {
            streamed_answers =
                Some(WeightMatrix::ALL.map(|m| streamed.answer(m, &source, &queries).unwrap()));
        });
        let streamed_answer_time = start.elapsed();
        assert_eq!(resident_answers, streamed_answers);
        black_box((&resident_answers, &streamed_answers));
        println!(
            "run={run} layers={layers} width={width} resident_commit_ns={} resident_peak={} resident_retained={} regenerated_commit_ns={} regenerated_peak={} regenerated_retained={} resident_answer_ns={} resident_answer_peak={} regenerated_answer_ns={} regenerated_answer_peak={}",
            resident_time.as_nanos(),
            resident_memory.bytes_max,
            resident_memory.bytes_current,
            streamed_time.as_nanos(),
            streamed_memory.bytes_max,
            streamed_memory.bytes_current,
            resident_answer_time.as_nanos(),
            resident_answer_memory.bytes_max,
            streamed_answer_time.as_nanos(),
            streamed_answer_memory.bytes_max
        );
    }
}
