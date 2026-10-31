// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Independent allocation traffic for public source validation (RFC-0018, #641).
mod common;

use allocation_counter::measure;
use tessera_codec::{Ascii, Secret};
use tessera_compiler::{
    PublicModel, RuntimePlan, analysis_memory_bytes, analyze, claim_list, lower_batch, select,
};
use tessera_ir::{
    CommitmentClass, GraphBuilder, IntRange, IntType, NodeKind, Op, ParamSetId, TensorDecl,
};
use tessera_witness::{CheckpointTensors, Checkpoints, Engine, Regenerator, Resident};

fn decl(shape: &[u32], kind: NodeKind, lo: i128, hi: i128) -> TensorDecl {
    TensorDecl {
        shape: shape.to_vec(),
        ty: IntType::I32,
        range: IntRange { lo, hi },
        kind,
    }
}

#[test]
fn coverage_walks_charge_repeated_readers_and_all_move_scratch() {
    let advice = NodeKind::Committed {
        commitment: CommitmentClass::Advice,
    };
    for readers in [1, 3, 17] {
        let mut b = GraphBuilder::new();
        let x = b.input(decl(&[128, 64], advice, 0, 3)).unwrap();
        let mut t = x;
        let mut step = |b: &mut GraphBuilder, op, shape: &[u32]| {
            t = b
                .node(op, &[t], vec![decl(shape, NodeKind::Virtual, 0, 0)])
                .unwrap()[0];
        };
        step(&mut b, Op::Transpose { perm: vec![1, 0] }, &[64, 128]);
        step(&mut b, Op::Expand { axis: 1, len: 2 }, &[64, 2, 128]);
        step(
            &mut b,
            Op::Slice {
                axis: 1,
                start: 0,
                len: 1,
            },
            &[64, 1, 128],
        );
        let index = b.constant((0..64).rev().collect()).unwrap();
        step(&mut b, Op::GatherPublic { index }, &[64, 1, 128]);
        step(&mut b, Op::Reshape, &[64, 128]);
        for _ in 0..readers {
            b.node(
                Op::BitDecompose { low: 1, high: 1 },
                &[t],
                vec![decl(&[64, 128, 1], advice, 0, 1); 2],
            )
            .unwrap();
        }
        let graph = b.finish();
        let mut bounds = None;
        let planning = measure(|| {
            bounds = Some((
                graph.validation_memory_bytes().unwrap(),
                analysis_memory_bytes(&graph).unwrap(),
            ))
        });
        assert_eq!(planning.count_total, 0);
        let (validation, analysis) = bounds.unwrap();
        let checked = measure(|| graph.validate().unwrap());
        assert!(
            checked.bytes_total <= validation,
            "{checked:?} {validation}"
        );
        let mut ranges = None;
        let checked = measure(|| ranges = Some(analyze(&graph, &[]).unwrap()));
        assert!(checked.bytes_total <= analysis, "{checked:?} {analysis}");
        // Reverse Gather, Slice, Expand and Transpose each allocate their input
        // coverage, once per reader. Reshape allocates no coverage buffer.
        assert!(analysis >= readers * (8192 + 16384 + 8192 + 8192));
    }
}

#[test]
fn graph_validation_counts_exact_view_and_producer_arrays() {
    let mut b = GraphBuilder::new();
    let x = b.input(decl(&[1], NodeKind::Public, 0, 1)).unwrap();
    b.node(Op::Add, &[x, x], vec![decl(&[1], NodeKind::Virtual, 0, 0)])
        .unwrap();
    let graph = b.finish();
    let plan = graph.validation_memory_bytes().unwrap();
    let checked = measure(|| graph.validate().unwrap());
    assert_eq!(checked.bytes_total, plan);
}

#[test]
fn runtime_replay_and_group_constructors_fit_public_request_bounds() {
    for fixture in common::conformance() {
        for requests in [vec![(3, 0)], vec![(2, 1)], vec![(2, 1), (2, 2), (2, 0)]] {
            let lengths = requests.iter().map(|(p, o)| p + o).collect::<Vec<_>>();
            let public = PublicModel::of(&fixture.model);
            let lowering = lower_batch(public, &lengths).unwrap();
            let ranges = analyze(&lowering.graph, &lowering.tables).unwrap();
            let policy = common::policy(2);
            let selection = select(&lowering.graph, &ranges, &lowering.tables, &policy).unwrap();
            let claims = claim_list(
                public,
                &selection,
                &policy,
                common::TEST_SMALL,
                ParamSetId(Ascii::new("test-small/1").unwrap()),
            )
            .unwrap();
            let runtime =
                RuntimePlan::derive_batch(&lowering, &selection, &claims, &requests).unwrap();
            let engine = Engine::new(&lowering)
                .unwrap()
                .threads(std::num::NonZeroUsize::MIN);
            let mut bound = None;
            let planning = measure(|| {
                bound = Some((
                    engine.runtime_source_metadata_memory(&runtime).unwrap(),
                    engine.source_replay_validation_memory().unwrap(),
                ))
            });
            assert_eq!(planning.count_total, 0);
            let (bound, replay) = bound.unwrap();
            assert!(replay > 0);
            if runtime.certificate().is_none() {
                assert_eq!(
                    bound,
                    engine.source_metadata_memory(runtime.selection()).unwrap()
                );
            }
            let input = Secret::new(vec![0; lengths.iter().sum::<u32>() as usize]);
            let tokens = engine
                .canonical_batch_tokens(&fixture.model, &input)
                .unwrap();
            let witness = engine.run(&fixture.model, &tokens).unwrap();
            let checkpoints = Checkpoints::new(&engine, &fixture.model, &tokens).unwrap();
            let cached =
                CheckpointTensors::new(&engine, &fixture.model, &tokens, &checkpoints).unwrap();
            let mut source = None;
            let baseline = measure(|| {
                source = Some(
                    Resident::new(&engine, runtime.selection(), &fixture.model, &witness).unwrap(),
                )
            });
            let baseline_bound = engine.source_metadata_memory(runtime.selection()).unwrap();
            let mut source = None;
            let built = measure(|| {
                source =
                    Some(Resident::runtime(&engine, &runtime, &fixture.model, &witness).unwrap())
            });
            assert!(built.bytes_total <= bound.bytes(), "{built:?} {bound:?}");
            assert!(
                built.bytes_total - baseline.bytes_total <= bound.bytes() - baseline_bound.bytes(),
                "runtime validation has its own reservation"
            );
            let mut source = None;
            let built = measure(|| {
                source = Some(
                    Resident::runtime_from_tensors(&engine, &runtime, &fixture.model, &cached)
                        .unwrap(),
                )
            });
            assert!(built.bytes_total <= bound.bytes(), "{built:?} {bound:?}");
            let resident_requests = built.bytes_total;
            let mut source = None;
            let built = measure(|| {
                source = Some(
                    Regenerator::runtime(&engine, &runtime, &fixture.model, &tokens, &checkpoints)
                        .unwrap(),
                )
            });
            assert!(
                built.bytes_total <= bound.bytes() + replay,
                "{built:?} {bound:?} {replay}"
            );
            assert_eq!(
                built.bytes_total - resident_requests,
                replay,
                "replay validates each input once, including both position vectors"
            );
            drop(source);
            for (len, capacity) in [(0, 17), (3, 17), (1, 1)] {
                let grouped = bound.with_grouped_columns(len, capacity).unwrap();
                assert_eq!(
                    grouped.bytes() - bound.bytes(),
                    capacity * size_of::<(tessera_ir::TableId, u64)>() as u64
                        + len * size_of::<tessera_ir::TableId>() as u64
                );
                let mut source = None;
                let built = measure(|| {
                    let mut columns = Vec::with_capacity(capacity as usize);
                    columns.extend((0..len).map(|t| (tessera_ir::TableId(t as u32), 1)));
                    source = Some(
                        Resident::runtime_from_tensors(&engine, &runtime, &fixture.model, &cached)
                            .unwrap()
                            .with_grouped_columns(columns)
                            .unwrap(),
                    );
                });
                assert!(
                    built.bytes_total <= grouped.bytes(),
                    "{built:?} {grouped:?}"
                );
            }
            assert_eq!(bound.with_grouped_columns(2, 1), None);
            assert_eq!(bound.with_grouped_columns(0, u64::MAX), None);
        }
    }
}

#[test]
fn validation_planners_refuse_missing_references_without_allocation() {
    let fixture = &common::conformance()[0];
    let (mut lowering, _) = common::lowered(&fixture.model, 3);
    let engine = Engine::for_graph(&lowering.graph, &lowering.tables).unwrap();
    assert_eq!(
        engine.source_replay_validation_memory(),
        Err(tessera_witness::WitnessError::Usage {
            rule: "a model runs on the engine of its lowering"
        })
    );
    lowering.positions = tessera_ir::TensorId(u32::MAX);
    let engine = Engine::new(&lowering).unwrap();
    let querying = measure(|| {
        assert_eq!(
            engine.source_replay_validation_memory(),
            Err(tessera_witness::WitnessError::Memory {
                tensor: lowering.positions
            })
        );
    });
    assert_eq!(querying.count_total, 0);
    lowering.graph.nodes[0].inputs[0] = tessera_ir::TensorId(u32::MAX);
    assert_eq!(lowering.graph.validation_memory_bytes(), None);
    assert_eq!(analysis_memory_bytes(&lowering.graph), None);
}

#[test]
fn two_output_bit_decomposition_charges_every_shape_clone_and_growth() {
    let advice = NodeKind::Committed {
        commitment: CommitmentClass::Advice,
    };
    for rank in [1, 2, 3, 4, 5, 8, 17, 64] {
        for high in [0, 1] {
            let mut b = GraphBuilder::new();
            let shape = vec![1; rank];
            let x = b
                .input(decl(&shape, NodeKind::Public, 0, 1 + i128::from(high) * 2))
                .unwrap();
            let output_shape = vec![1; rank + 1];
            let count = if high == 0 { 1 } else { 2 };
            b.node(
                Op::BitDecompose { low: 1, high },
                &[x],
                vec![decl(&output_shape, advice, 0, 1); count],
            )
            .unwrap();
            let graph = b.finish();
            let bound = graph.validation_memory_bytes().unwrap();
            let checked = measure(|| graph.validate().unwrap());
            assert!(
                checked.bytes_total <= bound,
                "rank={rank} high={high} {checked:?} bound={bound}"
            );
            let bound = analysis_memory_bytes(&graph).unwrap();
            let checked = measure(|| {
                analyze(&graph, &[]).unwrap();
            });
            assert!(
                checked.bytes_total <= bound,
                "rank={rank} high={high} {checked:?} bound={bound}"
            );
        }
    }
}

#[test]
fn runtime_validation_reservation_scales_with_nonlookup_graph_work() {
    use tessera_ir::{Node, NodeId, TensorId};
    let fixture = &common::conformance()[0];
    let public = PublicModel::of(&fixture.model);
    let (mut lowering, selection) = common::lowered(&fixture.model, 3);
    // Valid virtual relations add validation work without adding lookup reads.
    // This prevents a generous read-growth charge from hiding omitted analysis.
    for _ in 0..4096 {
        let id = TensorId(lowering.graph.tensors.len() as u32);
        let mut spec = lowering.graph.tensors[lowering.logits.0 as usize].clone();
        spec.id = id;
        spec.kind = NodeKind::Virtual;
        lowering.graph.tensors.push(spec);
        lowering.graph.nodes.push(Node {
            id: NodeId(lowering.graph.nodes.len() as u32),
            op: Op::Add,
            inputs: vec![lowering.logits, lowering.logits],
            outputs: vec![id],
        });
    }
    let policy = common::policy(2);
    let claims = claim_list(
        public,
        &selection,
        &policy,
        common::TEST_SMALL,
        ParamSetId(Ascii::new("test-small/1").unwrap()),
    )
    .unwrap();
    let runtime = RuntimePlan::derive(&lowering, &selection, &claims, 2, 0).unwrap();
    let engine = Engine::new(&lowering).unwrap();
    let base = engine.source_metadata_memory(runtime.selection()).unwrap();
    let bound = engine.runtime_source_metadata_memory(&runtime).unwrap();
    let actual = measure(|| {
        analyze(&lowering.graph, &lowering.tables).unwrap();
    });
    assert!(
        bound.bytes() - base.bytes() >= actual.bytes_total,
        "range validation must fit its own reservation: {actual:?} base={base:?} bound={bound:?}"
    );
}
