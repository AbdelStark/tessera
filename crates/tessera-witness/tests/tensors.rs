// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Whole-tensor checkpoint replay for reduction consumers (RFC-0018 P3, OBS-6).

mod common;

use std::num::NonZeroUsize;

use allocation_counter::measure;
use common::{conformance, lowered};
use tessera_codec::Secret;
use tessera_ir::TensorId;
use tessera_witness::{
    CheckpointTensors, Checkpoints, Engine, Regenerator, Resident, TensorSource, WitnessError,
};

#[test]
fn tile_sources_refuse_missing_or_misshaped_borrowed_weights_before_reading_tiles() {
    struct Malformed<'a> {
        source: &'a CheckpointTensors<'a>,
        target: TensorId,
        missing: bool,
        short: bool,
        wide: Option<&'a [i64]>,
    }
    impl TensorSource for Malformed<'_> {
        fn values(&self, tensor: TensorId) -> Option<&[i64]> {
            if tensor == self.target {
                self.wide
            } else {
                self.source.values(tensor)
            }
        }
        fn int8_values(&self, tensor: TensorId) -> Option<&[i8]> {
            let values = self.source.int8_values(tensor)?;
            if tensor != self.target {
                return Some(values);
            }
            if self.missing {
                None
            } else if self.short {
                Some(&values[..values.len() - 1])
            } else {
                Some(values)
            }
        }
    }
    let fixture = &conformance()[0];
    let (lowering, selection) = lowered(&fixture.model, 3);
    let engine = Engine::new(&lowering).unwrap();
    let tokens = Secret::new(vec![0, 1, 2]);
    let checkpoints = Checkpoints::new(&engine, &fixture.model, &tokens).unwrap();
    let source = CheckpointTensors::new(&engine, &fixture.model, &tokens, &checkpoints).unwrap();
    let target = lowering.graph.weights[0].1;
    Resident::from_tensors(&engine, &selection, &fixture.model, &source).unwrap();
    // An invalid wide representation must not fall back to the borrowed representation.
    for (missing, short, wide) in [
        (true, false, None),
        (false, true, None),
        (false, false, Some(&[0i64][..])),
    ] {
        let malformed = Malformed {
            source: &source,
            target,
            missing,
            short,
            wide,
        };
        assert_eq!(
            Resident::from_tensors(&engine, &selection, &fixture.model, &malformed).err(),
            Some(WitnessError::Usage {
                rule: "the witness is the first pass of the engine's lowering",
            })
        );
    }
}

#[test]
fn the_cache_owns_every_nonweight_and_borrows_original_int8_weights() {
    for fixture in conformance() {
        for n in [1, 3, 17] {
            let (lowering, _) = lowered(&fixture.model, n);
            let engine = Engine::new(&lowering)
                .unwrap()
                .threads(NonZeroUsize::new(1).unwrap());
            let tokens = Secret::new((0..n).map(|i| i % fixture.model.arch.vocab).collect());
            let first = engine.run(&fixture.model, &tokens).unwrap();
            let checkpoints = Checkpoints::new(&engine, &fixture.model, &tokens).unwrap();
            let mut cache = None;
            let built = measure(|| {
                cache = Some(
                    CheckpointTensors::new(&engine, &fixture.model, &tokens, &checkpoints).unwrap(),
                );
            });
            let cache = cache.unwrap();
            assert_eq!(
                built.bytes_current,
                (cache.storage_bytes().unwrap() - size_of::<CheckpointTensors<'_>>()) as i64,
                "retained cache allocation accounting"
            );
            // Network rounds can outlive all construction inputs except the borrowed model.
            drop(checkpoints);
            drop(tokens);
            drop(engine);
            for tensor in &lowering.graph.tensors {
                if let Some(&(weight, _)) = lowering
                    .graph
                    .weights
                    .iter()
                    .find(|(_, id)| *id == tensor.id)
                {
                    assert!(cache.values(tensor.id).is_none(), "no widened weight cache");
                    let bytes = cache.int8_values(tensor.id).unwrap();
                    let original = fixture
                        .model
                        .weights
                        .iter()
                        .find(|w| w.ty == weight.ty && w.layer == weight.layer)
                        .unwrap();
                    assert!(std::ptr::eq(bytes.as_ptr(), original.data.as_ptr()));
                    assert_eq!(bytes.len(), first.tensor(tensor.id).unwrap().len());
                    assert!(
                        bytes
                            .iter()
                            .zip(first.tensor(tensor.id).unwrap().iter())
                            .all(|(&byte, &wide)| i64::from(byte) == wide)
                    );
                } else {
                    assert!(cache.int8_values(tensor.id).is_none());
                    assert_eq!(
                        cache.values(tensor.id).unwrap(),
                        &first.tensor(tensor.id).unwrap()[..],
                        "{} n={n}, tensor {:?}",
                        fixture.name,
                        tensor.id
                    );
                }
            }
            assert!(cache.values(TensorId(u32::MAX)).is_none());
            assert!(cache.int8_values(TensorId(u32::MAX)).is_none());
        }
    }
}

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
