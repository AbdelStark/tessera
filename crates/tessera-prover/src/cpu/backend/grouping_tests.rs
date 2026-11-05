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
    let mut model = crate::leaf::source_test_model();
    model.arch.max_context = 512;
    for lengths in [vec![3], vec![3, 1, 2], vec![512]] {
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
                bound = Some(physical::group_search_memory(&input, 0).unwrap())
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
                        protocol::PROTOCOL_VERSION,
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
        let (v2, grouped_v2) = plan_groups(
            plan.clone(),
            &leaf,
            &layers,
            lowering.graph.tensors[lowering.tokens.0 as usize].shape[0],
            MemoryBudget { host: 0 },
            2,
            runtime.selection(),
        )
        .unwrap();
        assert_eq!(v2.leaf_groups(), protocol::LeafGroups::NONE);
        assert!(grouped_v2.is_none());
        let mut unsupported_claims = claims.clone();
        unsupported_claims.protocol_version = 2;
        let unsupported_input = physical::Inputs {
            claims: &unsupported_claims,
            ..input
        };
        let refused = allocation_counter::measure(|| {
            assert!(matches!(
                physical::select(&unsupported_input, 64 << 30),
                Err(ProverError::Usage {
                    rule: "physical checkpoint candidates require protocol/1"
                })
            ));
        });
        assert_eq!(refused.count_total, 0);
        if lengths != [512] {
            continue;
        }
        let execution = compiler::plan(PublicModel::of(&model), &claims).unwrap();
        let source = model_weights(&model, &lowering).unwrap();
        let commitment = pcs::RegeneratedWeightCommitment::commit(
            &weights,
            NonZeroU32::new(claims.layout.advice.row_block).unwrap(),
            0,
            &source,
        )
        .unwrap();
        let mut state = pcs::WeightCommitmentState::new(*commitment.public()).unwrap();
        let mut statement = statement;
        statement.claim_list_digest = tessera_hash::hash(
            tessera_hash::Domain::ClaimList,
            &tessera_codec::encode(&claims),
        );
        statement.weights_root = commitment.weights_root();
        let batch = ProofBatch {
            statement,
            tokens: Secret::new(vec![0; lengths.iter().sum::<u32>() as usize]),
        };
        let mut cfg = ProverConfig {
            backend: Backend::Cpu { threads: 1 },
            params,
            mode: Mode::NonInteractive,
            memory_budget: MemoryBudget { host: 64 << 30 },
            mask_rng: MaskRng::OsRng,
            release: ReleaseConfig,
        };
        let (mut low, mut high) = (0, 64u64 << 30);
        {
            let prover = Prover::new(&execution, &model, &mut state, cfg.clone()).unwrap();
            let prepared = prover.prepared.as_ref().unwrap();
            let candidate_input = physical::Inputs {
                claims: &claims,
                statement: &batch.statement,
                lowering: &lowering,
                layers: &layers,
                runtime: &runtime,
                advice: &advice,
                leaf: &leaf,
                native: &native,
                grouped: None,
                weights: &prepared.layout,
                session: Some(prepared.memory),
                caller_batch: Some(preparation::caller_batch_bytes(&batch).unwrap()),
                planner_caller: plan.heap_bytes().unwrap(),
            };
            while high - low > 1 {
                let mid = low + (high - low) / 2;
                match physical::select(&candidate_input, mid) {
                    Ok(_) => high = mid,
                    Err(e) if e.code() == ErrorCode::ResMemory => low = mid,
                    Err(e) => panic!("unexpected direct candidate refusal: {e}"),
                }
            }
            assert_eq!(
                physical::select(&candidate_input, low).unwrap_err().code(),
                ErrorCode::ResMemory
            );
        }
        let first = GroupedLayout::new(&leaf, &layers, group_candidate(9, layers.vars, 1).unwrap())
            .unwrap();
        let mut held_trial = None;
        let trial_allocation = allocation_counter::measure(|| {
            held_trial = Some(
                plan.clone()
                    .grouped(first.groups(), largest_group_lookups(&first).unwrap()),
            );
        });
        assert!(trial_allocation.bytes_current > 0);
        drop(held_trial);
        struct Trace;
        impl Drop for Trace {
            fn drop(&mut self) {
                GROUP_SEARCH_TRACE.with(|trace| trace.set(None));
                GROUP_PLANNER_OWNER_MIN.with(|owner| owner.set(None));
            }
        }
        {
            let _trace = Trace;
            GROUP_SEARCH_TRACE.with(|trace| trace.set(Some(0)));
            GROUP_PLANNER_OWNER_MIN.with(|owner| owner.set(Some(u64::MAX)));
            cfg.memory_budget.host = low;
            let chosen = Prover::new(&execution, &model, &mut state, cfg.clone())
                .unwrap()
                .plan(&batch)
                .unwrap();
            assert_eq!(
                chosen.check(cfg.memory_budget).unwrap_err().code(),
                ErrorCode::ResMemory
            );
            assert!(!chosen.uses_physical_bound());
            let token_vars =
                lowering.graph.tensors[lowering.tokens.0 as usize].shape[0].trailing_zeros();
            let expected = (1..=token_vars + layers.vars)
                .fold(0u64, |trace, bits| (trace << 4) | u64::from(bits));
            assert_eq!(
                GROUP_SEARCH_TRACE.with(std::cell::Cell::get),
                Some(expected),
                "the public search tries every complete grouped candidate in canonical order"
            );
            assert!(
                GROUP_PLANNER_OWNER_MIN.with(std::cell::Cell::get).unwrap()
                    >= plan.heap_bytes().unwrap()
                        + u64::try_from(trial_allocation.bytes_current).unwrap(),
                "the held original and allocated trial are both charged"
            );
        }
        cfg.memory_budget.host = high + (64 << 10);
        let chosen = Prover::new(&execution, &model, &mut state, cfg.clone())
            .unwrap()
            .plan(&batch)
            .unwrap();
        chosen.check(cfg.memory_budget).unwrap();
        assert!(chosen.uses_physical_bound());
        assert_eq!(chosen.leaf_groups(), protocol::LeafGroups::NONE);
        cfg.memory_budget.host = 64 << 30;
        let chosen = Prover::new(&execution, &model, &mut state, cfg)
            .unwrap()
            .plan(&batch)
            .unwrap();
        assert!(!chosen.uses_physical_bound());
        assert_eq!(chosen.leaf_groups(), protocol::LeafGroups::NONE);
        assert!(
            chosen.resident_bytes().unwrap()
                >= chosen.cells().total().unwrap() * PassPlan::BYTES_PER_CELL
        );
    }
}
