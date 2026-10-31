// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Independent allocation counters for checkpoint/cache constructor buffers (RFC-0018).

mod common;

use allocation_counter::measure;
use common::{conformance, lowered};
use tessera_codec::Secret;
use tessera_compiler::{Lowering, PublicModel, lower_batch};
use tessera_ir::IntegerModel;
use tessera_witness::{CheckpointTensors, Checkpoints, Engine};

#[test]
fn constructor_bounds_cover_tokens_checkpoints_and_every_nonweight_cache_buffer() {
    for fixture in conformance() {
        for n in [1, 3, 17, fixture.model.arch.max_context.min(65)] {
            let (lowering, _) = lowered(&fixture.model, n);
            let input = Secret::new((0..n).map(|i| i % fixture.model.arch.vocab).collect());
            check(&fixture.model, &lowering, &input);
        }
    }
}

#[test]
fn constructor_bounds_cover_three_slots_with_zero_gaps_and_request_tail_padding() {
    let fixture = &conformance()[0];
    for lengths in [[1, 3, 2], [3, 5, 2]] {
        let lowering = lower_batch(PublicModel::of(&fixture.model), &lengths).unwrap();
        let count = lengths.iter().sum::<u32>();
        let input = Secret::new((0..count).map(|i| i % fixture.model.arch.vocab).collect());
        check(&fixture.model, &lowering, &input);
    }
}

fn check(model: &IntegerModel, lowering: &Lowering, input: &Secret<Vec<u32>>) {
    let engine = Engine::new(lowering)
        .unwrap()
        .threads(std::num::NonZeroUsize::MIN);
    let mut plan = None;
    let planning = measure(|| plan = Some(engine.checkpoint_memory().unwrap()));
    assert_eq!(planning.count_total, 0);
    let plan = plan.unwrap();
    let mut tokens = None;
    let copying = measure(|| tokens = Some(engine.canonical_batch_tokens(model, input).unwrap()));
    let token_bytes = u64::try_from(copying.bytes_current).unwrap();
    assert_eq!(token_bytes, plan.tokens_bytes());
    assert!(copying.bytes_max <= plan.token_peak_bytes());
    let tokens = tokens.unwrap();
    let mut store = None;
    let building = measure(|| store = Some(Checkpoints::new(&engine, model, &tokens).unwrap()));
    let store_bytes = u64::try_from(building.bytes_current).unwrap();
    assert_eq!(store_bytes, plan.checkpoints_bytes());
    assert!(
        token_bytes + building.bytes_max <= plan.checkpoint_peak_bytes(),
        "checkpoints: {building:?} {plan:?}"
    );
    let store = store.unwrap();
    let mut cache = None;
    let caching =
        measure(|| cache = Some(CheckpointTensors::new(&engine, model, &tokens, &store).unwrap()));
    let cache_bytes = u64::try_from(caching.bytes_current).unwrap();
    assert_eq!(cache_bytes, plan.cache_bytes());
    assert!(
        token_bytes + store_bytes + caching.bytes_max <= plan.cache_peak_bytes(),
        "cache: {caching:?} {plan:?}"
    );
    assert_eq!(
        token_bytes + store_bytes + cache_bytes,
        plan.retained_bytes()
    );
    let observed_peak = copying
        .bytes_max
        .max(token_bytes + building.bytes_max)
        .max(token_bytes + store_bytes + caching.bytes_max);
    assert!(observed_peak <= plan.peak_bytes());
    drop(cache);
}

#[test]
fn checkpoint_planning_refuses_an_engine_without_its_model_lowering() {
    let fixture = &conformance()[0];
    let (lowering, _) = lowered(&fixture.model, 1);
    let engine = Engine::for_graph(&lowering.graph, &lowering.tables).unwrap();
    let mut result = None;
    let planning = measure(|| result = Some(engine.checkpoint_memory()));
    assert_eq!(planning.count_total, 0);
    assert_eq!(
        result.unwrap(),
        Err(tessera_witness::WitnessError::Usage {
            rule: "a model runs on the engine of its lowering",
        })
    );
}

#[test]
fn checkpoint_planning_refuses_missing_position_metadata_before_allocation() {
    let fixture = &conformance()[0];
    let (mut lowering, _) = lowered(&fixture.model, 1);
    lowering.positions = tessera_ir::TensorId(u32::MAX);
    let engine = Engine::new(&lowering).unwrap();
    let mut result = None;
    let planning = measure(|| result = Some(engine.checkpoint_memory()));
    assert_eq!(planning.count_total, 0);
    assert_eq!(
        result.unwrap(),
        Err(tessera_witness::WitnessError::Usage {
            rule: "a checkpoint memory tensor belongs to the lowering",
        })
    );
}
