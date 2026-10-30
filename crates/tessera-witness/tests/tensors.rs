// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Whole-tensor checkpoint replay for reduction consumers (RFC-0018 P3, OBS-6).

mod common;

use std::num::NonZeroUsize;

use allocation_counter::measure;
use common::{conformance, lowered};
use tessera_codec::Secret;
use tessera_ir::TensorId;
use tessera_witness::{Checkpoints, Engine, Regenerator, WitnessError};

#[test]
fn every_tensor_matches_the_resident_engine_and_only_the_result_survives() {
    // One thread: the allocation counter must see every replay allocation.
    for fixture in conformance() {
        for n in [1, 3, 17] {
            let (lowering, selection) = lowered(&fixture.model, n);
            let engine = Engine::new(&lowering)
                .unwrap()
                .threads(NonZeroUsize::new(1).unwrap());
            let tokens = Secret::new((0..n).map(|i| i % fixture.model.arch.vocab).collect());
            let first = engine.run(&fixture.model, &tokens).unwrap();
            let checkpoints = Checkpoints::new(&engine, &fixture.model, &tokens).unwrap();
            let source =
                Regenerator::new(&engine, &selection, &fixture.model, &tokens, &checkpoints)
                    .unwrap();
            // Reverse order exercises arbitrary reads, including a checkpoint, graph inputs,
            // native outputs and full score tensors, without a preceding tile pass.
            for spec in lowering.graph.tensors.iter().rev() {
                let mut result = None;
                let info = measure(|| result = Some(source.tensor(spec.id).unwrap()));
                let result = result.unwrap();
                assert_eq!(
                    &result[..],
                    &first.tensor(spec.id).unwrap()[..],
                    "{} n={n}, tensor {:?}",
                    fixture.name,
                    spec.id
                );
                assert_eq!(
                    info.bytes_current,
                    (result.capacity() * size_of::<i64>()) as i64,
                    "no replay cache or second tensor allocation survives"
                );
            }
        }
    }
}

#[test]
fn reading_one_weight_allocates_only_that_matrix_and_invalid_ids_allocate_nothing() {
    let fixture = &conformance()[0];
    let (lowering, selection) = lowered(&fixture.model, 3);
    let engine = Engine::new(&lowering)
        .unwrap()
        .threads(NonZeroUsize::new(1).unwrap());
    let tokens = Secret::new(vec![0, 1, 2]);
    let checkpoints = Checkpoints::new(&engine, &fixture.model, &tokens).unwrap();
    let source =
        Regenerator::new(&engine, &selection, &fixture.model, &tokens, &checkpoints).unwrap();
    for &(weight, id) in &lowering.graph.weights {
        let expected = fixture
            .model
            .weights
            .iter()
            .find(|w| w.ty == weight.ty && w.layer == weight.layer)
            .unwrap();
        let mut result = None;
        let info = measure(|| result = Some(source.tensor(id).unwrap()));
        let result = result.unwrap();
        assert!(
            result
                .iter()
                .zip(expected.data.iter())
                .all(|(&a, &b)| a == i64::from(b))
        );
        assert_eq!(result.len(), expected.data.len());
        assert_eq!(info.count_total, 1, "no other weights are widened");
        assert_eq!(
            info.bytes_max,
            (result.capacity() * size_of::<i64>()) as u64
        );
        let freed = measure(|| drop(result));
        assert_eq!(freed.bytes_current, -info.bytes_current);
    }
    for id in [
        TensorId(lowering.graph.tensors.len() as u32),
        TensorId(u32::MAX),
    ] {
        let info = measure(|| {
            assert_eq!(
                source.tensor(id).err().expect("invalid tensor refused"),
                WitnessError::Usage {
                    rule: "a regenerated tensor belongs to the source's lowering",
                }
            );
        });
        assert_eq!((info.count_total, info.bytes_max), (0, 0));
    }
}
