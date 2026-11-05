// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::*;
use tessera_codec::{DecodeLimits, envelope, schema};

#[test]
fn canonical_group_work_planning_is_bounded_and_refuses_before_field_work() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../spec/vectors/tamper/claims.tcl");
    let claims: tessera_ir::ClaimList = envelope::open(
        &std::fs::read(path).unwrap(),
        &schema::CLAIM_LIST,
        DecodeLimits::MESSAGE,
    )
    .unwrap();
    let mut cases = 0;
    for (lengths, output) in [(vec![3], false), (vec![5], true), (vec![3, 1, 2], true)] {
        let requests: Vec<_> = lengths
            .iter()
            .map(|&n| if output { (1, n - 1) } else { (n, 0) })
            .collect();
        let lowering = compiler::lower_batch(PublicModel::of_claims(&claims), &lengths).unwrap();
        let ranges = compiler::analyze(&lowering.graph, &lowering.tables).unwrap();
        let selected = compiler::select_model(
            claims.protocol_version,
            &lowering,
            &ranges,
            &claims.gadget_policy,
        )
        .unwrap();
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
        for (token_bits, layer_bits) in [(0, 1), (1, 0), (1, 1)] {
            let grouping = GroupedLayout::new(
                &leaf,
                &layers,
                protocol::LeafGroups {
                    token_bits,
                    layer_bits,
                },
            )
            .unwrap();
            let input = Inputs {
                claims: &claims,
                statement: &statement,
                lowering: &lowering,
                layers: &layers,
                runtime: &runtime,
                advice: &advice,
                leaf: &leaf,
                native: &native,
                grouped: Some(&grouping),
                weights: &weights,
                session: None,
                caller_batch: None,
                planner_caller: 0,
            };
            let mut unrestricted = None;
            let sized = allocation_counter::measure(|| {
                unrestricted = Some(GroupedLayout::unrestricted_group_memory(&leaf).unwrap());
            });
            assert_eq!(sized.count_total, 0);
            let unrestricted = unrestricted.unwrap();
            for h in 0..grouping.count() {
                let restricted = grouping.group_memory(h).unwrap();
                assert!(restricted.peak_bytes() <= unrestricted.peak_bytes());
                assert!(restricted.output_bytes() <= unrestricted.output_bytes());
            }
            let table =
                table_plan::TablePlan::new(&leaf, runtime.selection(), Some(&grouping), None)
                    .unwrap();
            let mut ceiling = None;
            let sized = allocation_counter::measure(|| {
                ceiling = Some(
                    table_plan::TablePlan::search_metadata_ceiling(
                        &leaf,
                        runtime.selection(),
                        layers.vars,
                        grouping.groups().bits(),
                    )
                    .unwrap(),
                );
            });
            assert_eq!(sized.count_total, 0);
            assert!(
                table
                    .construction_peak_bytes(&leaf, Some(&grouping))
                    .unwrap()
                    <= ceiling.unwrap()
            );
            let mut bill = None;
            let measured = allocation_counter::measure(|| {
                bill = Some(GroupedWorkMemory::new(&input, 64 << 30).unwrap());
            });
            let bill = bill.unwrap();
            assert_eq!(measured.bytes_current, 0);
            assert!(
                measured.bytes_max <= bill.planning,
                "{measured:?} > {bill:?}"
            );
            assert!(bill.peak > bill.messages.driver_bytes());
            assert_eq!(
                GroupedWorkMemory::new(&input, bill.planning - 1)
                    .unwrap_err()
                    .code(),
                ErrorCode::ResMemory
            );
            let mut refused = None;
            let tiny =
                allocation_counter::measure(|| refused = Some(GroupedWorkMemory::new(&input, 0)));
            assert_eq!(tiny.count_total, 0);
            assert_eq!(refused.unwrap().unwrap_err().code(), ErrorCode::ResMemory);
            // Neither a component nor an unlimited allowance bypasses either production guard.
            assert!(matches!(
                leaf_messages(&input, 0),
                Err(ProverError::Usage { .. })
            ));
            cases += 1;
        }
    }
    assert_eq!(cases, 9);
}

#[test]
fn canonical_scoped_generated_proof_matches_resident_with_composed_work_bound() {
    use crate::leaf::table_source::GeneratedCounts;
    use crate::leaf::{
        MergeMode, ModelSourceFactory, ModelWitness, ScopedSourceMemory,
        prove_grouped_leaf_bounded, prove_grouped_leaf_sources, source_test_model,
    };
    use tessera_codec::{Ascii, Encode, Secret};
    use tessera_hash::{Mode, ParamSetId, SessionContext};
    use tessera_ir::GadgetPolicy;
    use tessera_tables::GeneratedTable;
    use tessera_witness::{Resident, WitnessSource};
    let model = source_test_model();
    for lengths in [vec![3], vec![3, 1, 2]] {
        let lowering = compiler::lower_batch(PublicModel::of(&model), &lengths).unwrap();
        let ranges = compiler::analyze(&lowering.graph, &lowering.tables).unwrap();
        let policy = GadgetPolicy {
            max_log2_rows: 21,
            query_block: 2,
            ..Default::default()
        };
        let selected =
            compiler::select_model(protocol::PROTOCOL_VERSION, &lowering, &ranges, &policy)
                .unwrap();
        let p = ParamSet::TEST_SMALL;
        let claims = compiler::claim_list(
            PublicModel::of(&model),
            &selected,
            &policy,
            p.layout,
            ParamSetId(Ascii::new(p.id).unwrap()),
        )
        .unwrap();
        let requests: Vec<_> = lengths.iter().map(|&n| (n, 0)).collect();
        let runtime = RuntimePlan::derive_batch(&lowering, &selected, &claims, &requests).unwrap();
        let advice = ModelAdviceLayout::runtime(&lowering, &runtime, &claims).unwrap();
        let layers = LayerMap::new(&lowering).unwrap();
        let leaf = LeafLayout::runtime(&lowering, &layers, &runtime, &claims).unwrap();
        let native = ModelNativePlan::derive_runtime(&lowering, &runtime, &claims, &leaf).unwrap();
        let grouped = GroupedLayout::new(
            &leaf,
            &layers,
            protocol::LeafGroups {
                token_bits: 1,
                layer_bits: 1,
            },
        )
        .unwrap();
        let engine = Engine::new(&lowering).unwrap();
        let tokens = Secret::new(
            (0..lengths.iter().sum::<u32>())
                .map(|i| (i + 1) % model.arch.vocab)
                .collect(),
        );
        let witness = engine.run_batch(&model, &tokens).unwrap();
        let source = Resident::runtime(&engine, &runtime, &model, &witness)
            .unwrap()
            .with_grouped_columns(grouped.columns().unwrap())
            .unwrap();
        let counts = engine
            .runtime_multiplicity_plans(&claims.arch, &runtime)
            .unwrap();
        // A real public table plan, with no guessed zero source backing or
        // unlimited allowance. Graph/model/context owners remain outside this
        // incremental source/kernel control, as in the component's public contract.
        let table_plan =
            table_plan::TablePlan::new(&leaf, runtime.selection(), Some(&grouped), Some(&counts))
                .unwrap()
                .select_cached(128 << 20)
                .unwrap();
        assert!(
            table_plan.cache_log_rows.is_some(),
            "fixture must exercise the chosen real cache"
        );
        let tables: Vec<_> = leaf
            .instances()
            .iter()
            .map(|i| {
                let spec = runtime
                    .selection()
                    .tables
                    .iter()
                    .find(|s| s.id == i.table)
                    .unwrap();
                let tile = advice
                    .tiles()
                    .iter()
                    .find(|t| matches!(t.tile,compiler::Tile::Table {table} if table==i.table.0))
                    .unwrap();
                GeneratedCounts::new(
                    GeneratedTable::from_spec(spec).unwrap(),
                    source.tile(tile).unwrap(),
                )
                .unwrap()
                .with_cache_depth(table_plan::cache_depth(
                    usize::from(spec.log2_rows),
                    table_plan.cache_log_rows,
                ))
            })
            .collect();
        let unused =
            (tables.capacity() - tables.len()) as u64 * size_of::<GeneratedCounts>() as u64;
        let source_bytes = tables
            .iter()
            .try_fold(table_plan.retained + table_plan.forests + unused, |n, t| {
                n.checked_add(t.bytes().unwrap())
            })
            .unwrap();
        let budget = tessera_logup::TableBatchBudget {
            allowance: table_plan.working_bytes,
            source_bytes,
            chunk_rows: NonZeroUsize::new(table_plan.chunk_rows).unwrap(),
        };
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
        let input = Inputs {
            claims: &claims,
            statement: &statement,
            lowering: &lowering,
            layers: &layers,
            runtime: &runtime,
            advice: &advice,
            leaf: &leaf,
            native: &native,
            grouped: Some(&grouped),
            weights: &weights,
            session: None,
            caller_batch: None,
            planner_caller: 0,
        };
        let work = GroupedWorkMemory::new(&input, 128 << 20).unwrap();
        let scoped = ScopedSourceMemory::new(&leaf, &lowering, &layers, &grouped, &advice).unwrap();
        let composed = sum([
            scoped.peak().unwrap(),
            work.peak,
            work.messages.retained_bytes(),
            work.messages.wire_bytes(),
            engine.runtime_source_tile_work_memory(&runtime).unwrap(),
            table_plan.working_bytes,
        ])
        .unwrap();
        let factory = ModelSourceFactory {
            layout: &leaf,
            grouped: &grouped,
            lowering: &lowering,
            layers: &layers,
            witness: &witness,
            source: &source,
            advice: &advice,
        };
        let context = SessionContext {
            protocol: Ascii::new("tessera-protocol/1").unwrap(),
            mode: Mode::NonInteractive,
            params: ParamSetId(Ascii::new(p.id).unwrap()),
            statement_digest: [91; 32],
            session_id: [27; 16],
        };
        let mut candidate_transcript = Transcript::new(&context);
        let mut candidate = None;
        let measured = allocation_counter::measure(|| {
            candidate = Some(
                prove_grouped_leaf_sources(
                    &mut candidate_transcript,
                    &factory,
                    &tables,
                    MergeMode::TwoPass { j_star: None },
                    budget,
                )
                .unwrap(),
            );
        });
        assert!(
            measured.bytes_max <= composed,
            "{measured:?} > {composed}, work {work:?}, source {scoped:?}"
        );
        let candidate = candidate.unwrap();
        // Construct the unrestricted oracle only after the scoped measurement.
        let full =
            ModelWitness::new(&leaf, &lowering, &layers, &witness, &source, &advice).unwrap();
        let mut resident_transcript = Transcript::new(&context);
        let resident = prove_grouped_leaf_bounded(
            &mut resident_transcript,
            &leaf,
            &grouped,
            &tables,
            &full.tensors(),
            &full.stage_forests(),
            MergeMode::TwoPass { j_star: None },
            budget,
        )
        .unwrap();
        assert_eq!(candidate_transcript.state(), resident_transcript.state());
        assert_eq!(
            candidate_transcript.counter(),
            resident_transcript.counter()
        );
        assert_eq!(candidate.merged(), resident.merged());
        let mut candidate_wire = Vec::new();
        let mut resident_wire = Vec::new();
        candidate.to_wire(&[]).encode_to(&mut candidate_wire);
        resident.to_wire(&[]).encode_to(&mut resident_wire);
        assert_eq!(candidate_wire, resident_wire);
    }
}
