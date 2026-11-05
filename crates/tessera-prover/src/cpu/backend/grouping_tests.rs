// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::*;
use tessera_codec::Ascii;
use tessera_hash::ParamSetId;

struct NoWitness;
impl crate::stages::WitnessSource for NoWitness {
    fn values(&self, _: tessera_ir::TensorId) -> Option<&[i64]> {
        None
    }
}

#[test]
fn public_group_candidate_order_preserves_token_floor_and_layer_exhaustion() {
    let groups = |t, l, g| group_candidate(t, l, g).map(|x| (x.token_bits, x.layer_bits));
    assert_eq!(groups(0, 0, 1), None);
    assert_eq!(groups(1, 0, 1), Some((1, 0)));
    assert_eq!(groups(8, 4, 4), Some((0, 4)));
    assert_eq!(groups(8, 4, 5), Some((1, 4)));
    assert_eq!(groups(11, 4, 1), Some((1, 0)));
    assert_eq!(groups(11, 4, 4), Some((3, 1)));
    assert_eq!(groups(11, 4, 7), Some((3, 4)));
    assert_eq!(groups(11, 4, 14), Some((10, 4)));
    assert_eq!(groups(1, 1, 3), None);
    assert_eq!(groups(30, 30, 15), None);
}

#[test]
fn actual_resident_group_search_fits_the_public_preparation_ceiling() {
    let model = crate::leaf::source_test_model();
    for lengths in [vec![3], vec![3, 1, 2]] {
        let lowering = compiler::lower_batch(PublicModel::of(&model), &lengths).unwrap();
        let ranges = compiler::analyze(&lowering.graph, &lowering.tables).unwrap();
        let policy = tessera_ir::GadgetPolicy {
            max_log2_rows: 21,
            query_block: 2,
            ..Default::default()
        };
        let selected =
            compiler::select_model(protocol::PROTOCOL_VERSION, &lowering, &ranges, &policy)
                .unwrap();
        let params = ParamSet::TEST_SMALL;
        let claims = compiler::claim_list(
            PublicModel::of(&model),
            &selected,
            &policy,
            params.layout,
            ParamSetId(Ascii::new(params.id).unwrap()),
        )
        .unwrap();
        let requests: Vec<_> = lengths.iter().map(|&n| (n, 0)).collect();
        let runtime = RuntimePlan::derive_batch(&lowering, &selected, &claims, &requests).unwrap();
        let advice = ModelAdviceLayout::runtime(&lowering, &runtime, &claims).unwrap();
        let layers = LayerMap::new(&lowering).unwrap();
        let leaf = LeafLayout::runtime(&lowering, &layers, &runtime, &claims).unwrap();
        let native = ModelNativePlan::derive_runtime(&lowering, &runtime, &claims, &leaf).unwrap();
        let weights = pcs::WeightLayout::new(
            claims.layout.weights,
            pcs::WeightShapes {
                layers: claims.arch.n_layers,
                shapes: claims.arch.weight_shapes().unwrap(),
                channels: lowering
                    .classes
                    .iter()
                    .chain(&lowering.multipliers)
                    .map(|(w, _)| (w.ty.matrix(), w.layer))
                    .collect(),
            },
        )
        .unwrap();
        let statement = protocol::PublicStatement {
            claim_list_digest: [0; 32],
            weights_root: [0; 32],
            weights_epoch: 0,
            manifest_digest: [0; 32],
            batch: protocol::BatchDescriptor {
                requests: requests
                    .iter()
                    .enumerate()
                    .map(|(r, &(prompt_len, output_len))| protocol::RequestRef {
                        request_id: r as u64,
                        session_id: [0; 16],
                        prompt_len,
                        output_len,
                    })
                    .collect(),
            },
            predictor: protocol::PredictorSpec::Exact {
                decoding: protocol::Decoding::Greedy,
            },
        };
        let mut cells = Cells::default();
        for tensor in &lowering.graph.tensors {
            cells.tensors += tensor
                .shape
                .iter()
                .map(|&n| u64::from(n.next_power_of_two()))
                .product::<u64>();
        }
        for instance in leaf.instances() {
            cells.lookups += power(instance.entries).unwrap();
            cells.table_rows += 1u64
                << runtime
                    .selection()
                    .tables
                    .iter()
                    .find(|t| t.id == instance.table)
                    .unwrap()
                    .log2_rows;
        }
        cells.advice = advice.layout().rows() * (1u64 << claims.layout.advice.log2_k);
        let metadata = crate::StageProver::new(&lowering, native.schedule(), &NoWitness);
        let shapes = (0..native.schedule().stages.len())
            .map(|k| {
                if k == 0 {
                    Vec::new()
                } else {
                    metadata
                        .window_shapes(k as u16, |tag| native.window_shape(tag))
                        .unwrap()
                }
            })
            .collect();
        let plan = PassPlan::new(native.schedule().stages.len(), cells, 0)
            .staged(shapes)
            .unwrap()
            .tables(TablePlan::new(&leaf, runtime.selection(), None, None).unwrap());
        let input = physical::Inputs {
            claims: &claims,
            statement: &statement,
            lowering: &lowering,
            layers: &layers,
            runtime: &runtime,
            advice: &advice,
            leaf: &leaf,
            native: &native,
            grouped: None,
            weights: &weights,
            session: None,
            caller_batch: None,
            planner_caller: plan.heap_bytes().unwrap(),
        };
        for host in [0, plan.resident_bytes().unwrap() - 1, 64 << 30] {
            let mut bound = None;
            let sized = allocation_counter::measure(|| {
                bound = Some(physical::group_search_memory(&input, host, 0).unwrap())
            });
            assert_eq!(sized.count_total, 0);
            let mut result = None;
            let actual = allocation_counter::measure(|| {
                result = Some(
                    plan_groups(
                        plan.clone(),
                        &leaf,
                        &layers,
                        lowering.graph.tensors[lowering.tokens.0 as usize].shape[0],
                        MemoryBudget { host },
                        runtime.selection(),
                    )
                    .unwrap(),
                );
            });
            assert!(
                actual.bytes_max <= bound.unwrap(),
                "{host}: {actual:?} > {bound:?}"
            );
            drop(result);
        }
    }
}
