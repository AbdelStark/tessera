// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Retained source metadata measured independently of the accounting (RFC-0018, #641).
mod common;

use allocation_counter::measure;
use tessera_codec::{Ascii, Secret};
use tessera_compiler::{PublicModel, RuntimePlan, analyze, claim_list, lower_batch, select};
use tessera_ir::ParamSetId;
use tessera_witness::{Checkpoints, Engine, Regenerator, Resident};

fn check<T>(make: impl FnOnce() -> T, bytes: impl Fn(&T) -> Option<usize>) {
    let mut value = None;
    let construction = measure(|| value = Some(make()));
    let value = value.unwrap();
    assert!(construction.bytes_current > 0);
    assert_eq!(
        usize::try_from(construction.bytes_current).unwrap(),
        bytes(&value).unwrap(),
        "every retained allocation, including spare capacity, is charged"
    );
    let querying = measure(|| assert!(bytes(&value).is_some()));
    assert_eq!(querying.count_total, 0);
}

#[test]
fn resident_and_replay_metadata_count_all_gadget_templates_and_group_capacity() {
    for fixture in common::conformance() {
        for n in [1, 3, 17] {
            let (lowering, selection) = common::lowered(&fixture.model, n);
            let engine = Engine::new(&lowering)
                .unwrap()
                .threads(std::num::NonZeroUsize::MIN);
            let tokens = Secret::new(vec![0; n as usize]);
            let witness = engine.run(&fixture.model, &tokens).unwrap();
            let checkpoints = Checkpoints::new(&engine, &fixture.model, &tokens).unwrap();
            for spare in [0, 17] {
                check(
                    || {
                        Resident::new(&engine, &selection, &fixture.model, &witness)
                            .unwrap()
                            .with_grouped_columns(Vec::with_capacity(spare))
                            .unwrap()
                    },
                    Resident::metadata_bytes,
                );
                check(
                    || {
                        Regenerator::new(&engine, &selection, &fixture.model, &tokens, &checkpoints)
                            .unwrap()
                            .with_grouped_columns(Vec::with_capacity(spare))
                            .unwrap()
                    },
                    Regenerator::metadata_bytes,
                );
            }
        }
    }
}

#[test]
fn runtime_metadata_counts_extra_reads_and_ragged_request_key_columns() {
    let fixture = &common::conformance()[0];
    let public = PublicModel::of(&fixture.model);
    for requests in [vec![(2, 1)], vec![(2, 1), (2, 2), (2, 0)]] {
        let lengths = requests
            .iter()
            .map(|(prompt, output)| prompt + output)
            .collect::<Vec<_>>();
        let lowering = lower_batch(public, &lengths).unwrap();
        let ranges = analyze(&lowering.graph, &lowering.tables).unwrap();
        let policy = common::policy(2);
        let selection = select(&lowering.graph, &ranges, &lowering.tables, &policy).unwrap();
        let params = ParamSetId(Ascii::new("test-small/1").unwrap());
        let claims = claim_list(public, &selection, &policy, common::TEST_SMALL, params).unwrap();
        let runtime = RuntimePlan::derive_batch(&lowering, &selection, &claims, &requests).unwrap();
        let engine = Engine::new(&lowering)
            .unwrap()
            .threads(std::num::NonZeroUsize::MIN);
        let concatenated = Secret::new(vec![0; lengths.iter().sum::<u32>() as usize]);
        let tokens = engine
            .canonical_batch_tokens(&fixture.model, &concatenated)
            .unwrap();
        let witness = engine.run(&fixture.model, &tokens).unwrap();
        let checkpoints = Checkpoints::new(&engine, &fixture.model, &tokens).unwrap();
        // Source construction validates public shapes. Exact greedy output is checked when
        // requesting its tile, outside this metadata-only allocation measurement.
        check(
            || Resident::runtime(&engine, &runtime, &fixture.model, &witness).unwrap(),
            Resident::metadata_bytes,
        );
        check(
            || {
                Regenerator::runtime(&engine, &runtime, &fixture.model, &tokens, &checkpoints)
                    .unwrap()
            },
            Regenerator::metadata_bytes,
        );
    }
}

#[test]
fn public_construction_bound_covers_resident_and_cached_source_metadata_requests() {
    for fixture in common::conformance() {
        for n in [1, 3, 17] {
            let (lowering, selection) = common::lowered(&fixture.model, n);
            let engine = Engine::new(&lowering)
                .unwrap()
                .threads(std::num::NonZeroUsize::MIN);
            let tokens = Secret::new(vec![0; n as usize]);
            let mut plan = None;
            let planning =
                measure(|| plan = Some(engine.source_metadata_memory(&selection).unwrap()));
            assert_eq!(planning.count_total, 0);
            let plan = plan.unwrap();
            let witness = engine.run(&fixture.model, &tokens).unwrap();
            let mut source = None;
            let construction = measure(|| {
                source = Some(Resident::new(&engine, &selection, &fixture.model, &witness).unwrap())
            });
            assert!(
                construction.bytes_total <= plan.bytes(),
                "{construction:?} {plan:?}"
            );
            let checkpoints = Checkpoints::new(&engine, &fixture.model, &tokens).unwrap();
            let cached = tessera_witness::CheckpointTensors::new(
                &engine,
                &fixture.model,
                &tokens,
                &checkpoints,
            )
            .unwrap();
            let mut cached_source = None;
            let construction = measure(|| {
                cached_source = Some(
                    Resident::from_tensors(&engine, &selection, &fixture.model, &cached).unwrap(),
                )
            });
            assert!(
                construction.bytes_total <= plan.bytes(),
                "{construction:?} {plan:?}"
            );
        }
    }
}

#[test]
fn wrong_requant_role_still_charges_a_large_generated_core_before_rejection() {
    use tessera_codec::Rational;
    use tessera_compiler::Gadget;
    use tessera_ir::{Op, TableId};
    use tessera_tables::{TableKind, TableSpec};
    let fixture = &common::conformance()[0];
    let (lowering, mut selection) = common::lowered(&fixture.model, 1);
    let node = lowering
        .graph
        .nodes
        .iter()
        .find(|node| matches!(node.op, Op::Requant { .. }))
        .unwrap()
        .id;
    let table = TableId(selection.tables.iter().map(|spec| spec.id.0).max().unwrap() + 1);
    selection.tables.push(TableSpec {
        id: table,
        log2_rows: 24,
        digest: [0; 32],
        kind: TableKind::Exp {
            key_bits: 24,
            sigma: 31,
            scale: Rational::new(1, 1 << 32).unwrap(),
        },
    });
    selection
        .gadgets
        .iter_mut()
        .find(|(id, _)| *id == node)
        .unwrap()
        .1 = Gadget::Requant1 { table };
    let engine = Engine::new(&lowering).unwrap();
    let mut plan = None;
    let planning = measure(|| plan = Some(engine.source_metadata_memory(&selection).unwrap()));
    assert_eq!(planning.count_total, 0);
    // Every key is far before the Exp zero tail. The malformed digest/role is
    // deliberate: from_spec constructs this entire core before rejecting it.
    assert!(plan.unwrap().bytes() >= 8 * (1 << 24));
}

#[test]
fn metadata_planning_requires_the_model_lowering_before_allocation() {
    let fixture = &common::conformance()[0];
    let (lowering, selection) = common::lowered(&fixture.model, 1);
    let engine = Engine::for_graph(&lowering.graph, &lowering.tables).unwrap();
    let measured = measure(|| {
        assert_eq!(
            engine.source_metadata_memory(&selection).err(),
            Some(tessera_witness::WitnessError::Usage {
                rule: "a model runs on the engine of its lowering",
            })
        );
    });
    assert_eq!(measured.count_total, 0);
}
