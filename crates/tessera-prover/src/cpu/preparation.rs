// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Session preparation lifetimes before P1 (RFC-0018/0041/0047).
use super::*;

#[derive(Clone, Copy, Debug)]
pub(super) struct SessionMemory {
    pub(super) caller: u64,
    pub(super) peak: u64,
    #[cfg(test)]
    pub(super) weight_construction: u64,
}

impl SessionMemory {
    pub(super) fn with_batch(self, batch: &ProofBatch) -> Result<Self, ProverError> {
        let fail = || ProverError::Resource(ErrorCode::ResMemory);
        let bytes = caller_batch_bytes(batch).ok_or_else(fail)?;
        Ok(Self {
            caller: self.caller.checked_add(bytes).ok_or_else(fail)?,
            peak: self.peak.checked_add(bytes).ok_or_else(fail)?,
            #[cfg(test)]
            weight_construction: self
                .weight_construction
                .checked_add(bytes)
                .ok_or_else(fail)?,
        })
    }
}

pub(super) fn caller_batch_bytes(batch: &ProofBatch) -> Option<u64> {
    tessera_witness::Engine::token_storage_bytes(&batch.tokens)?.checked_add(
        (batch.statement.batch.requests.capacity() as u64)
            .checked_mul(size_of::<protocol::RequestRef>() as u64)?,
    )
}

pub(super) fn session(
    plan: &ExecutionPlan,
    model: PublicModel<'_>,
    state: &pcs::WeightCommitmentState,
    lower: &compiler::Lowering,
    layout: &pcs::WeightLayout,
    plan_metadata: Option<u64>,
) -> Result<SessionMemory, ProverError> {
    let fail = || ProverError::Resource(ErrorCode::ResMemory);
    let tables = tessera_witness::Engine::table_memory(&plan.claims.tables)?.peak_bytes();
    let params = ParamSet::defined(&plan.claims.params)
        .ok_or_else(|| usage("defined parameters required"))?;
    // Only construction/model-reference fields are used here. Their bounds are
    // independent of later opening counts; no claim is dropped or invented.
    let weight = layout.regenerated_work_memory(
        NonZeroU32::new(plan.claims.layout.advice.row_block).ok_or_else(fail)?,
        u64::from(params.weight_queries),
        1,
    )?;
    let bytes = (|| {
        let caller = plan.heap_upper_bound()?.checked_add(state.heap_bytes()?)?;
        let lowering = lower.construction_metadata_bytes(model)?;
        let layout_bytes = layout.heap_bytes()?;
        let commit = lower
            .heap_bytes()?
            .checked_add(layout_bytes)?
            .checked_add(weight.model_refs_bytes())?
            .checked_add(weight.construction_peak_bytes())?;
        // validate(model)'s public point/binding/weight-shape lists are included
        // by lowering's metadata bound. Its secret payload scan allocates nothing.
        let peak = lowering
            .checked_add(tables)?
            .max(plan_metadata?.checked_add(tables)?)
            .max(commit)
            .checked_add(caller)?;
        Some(SessionMemory {
            caller,
            peak,
            #[cfg(test)]
            weight_construction: commit.checked_add(caller)?,
        })
    })();
    bytes.ok_or_else(fail)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tessera_codec::{DecodeLimits, envelope, schema};

    #[test]
    fn session_constructor_peak_covers_canonical_checks_and_weight_commitment() {
        for name in [
            "tiny-2-servable",
            "tiny-2-minimal",
            "tiny-2-affine",
            "tiny-4-servable",
        ] {
            let root =
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/models");
            let model: IntegerModel = envelope::open(
                &std::fs::read(root.join(format!("{name}.d"))).unwrap(),
                &schema::INTEGER_MODEL,
                DecodeLimits::INTEGER_MODEL,
            )
            .unwrap();
            let lower = compiler::lower(PublicModel::of(&model), 1).unwrap();
            let ranges = compiler::analyze(&lower.graph, &lower.tables).unwrap();
            let policy = tessera_ir::GadgetPolicy::default();
            let selection =
                compiler::select_model(protocol::PROTOCOL_VERSION, &lower, &ranges, &policy)
                    .unwrap();
            let params = ParamSet::TEST_SMALL;
            let claims = compiler::claim_list(
                PublicModel::of(&model),
                &selection,
                &policy,
                params.layout,
                tessera_hash::ParamSetId(Ascii::new(params.id).unwrap()),
            )
            .unwrap();
            let mut plan = compiler::plan(PublicModel::of(&model), &claims).unwrap();
            // The borrowed plan's public string and tile vector can retain spare
            // capacity even when their canonical encodings are unchanged.
            let mut text = String::with_capacity(1 << 16);
            text.push_str(params.id);
            plan.claims.params = tessera_hash::ParamSetId(Ascii::new(text).unwrap());
            plan.tiles.reserve(1 << 12);
            let source = backend::model_weights(&model, &lower).unwrap();
            let layout = pcs::WeightLayout::new(
                claims.layout.weights,
                pcs::WeightShapes {
                    layers: claims.arch.n_layers,
                    shapes: claims.arch.weight_shapes().unwrap(),
                    channels: source
                        .channels
                        .iter()
                        .map(|c| (c.matrix, c.layer))
                        .collect(),
                },
            )
            .unwrap();
            let actual = pcs::RegeneratedWeightCommitment::commit(
                &layout,
                NonZeroU32::new(claims.layout.advice.row_block).unwrap(),
                0,
                &source,
            )
            .unwrap();
            let mut state = pcs::WeightCommitmentState::new(*actual.public()).unwrap();
            let config = ProverConfig {
                backend: Backend::Cpu { threads: 1 },
                params,
                mode: Mode::NonInteractive,
                memory_budget: MemoryBudget { host: 64 << 30 },
                mask_rng: MaskRng::OsRng,
                release: ReleaseConfig,
            };
            let mut bound = None;
            let mut weight_bound = None;
            let observed = allocation_counter::measure(|| {
                let prover = Prover::new(&plan, &model, &mut state, config.clone()).unwrap();
                let memory = prover.prepared.as_ref().unwrap().memory;
                bound = Some(memory.peak);
                weight_bound = Some(memory.weight_construction);
                std::hint::black_box(prover);
            });
            assert!(
                observed.bytes_max <= bound.unwrap(),
                "{name}: {observed:?} > {bound:?}"
            );
            // The caller's allocations are additional to the measured constructor.
            let caller = plan.heap_upper_bound().unwrap() + state.heap_bytes().unwrap();
            assert!(
                observed.bytes_max.checked_add(caller).unwrap() <= bound.unwrap(),
                "{name}: constructor + caller {observed:?} + {caller} > {bound:?}"
            );
            let committed = allocation_counter::measure(|| {
                std::hint::black_box(
                    pcs::RegeneratedWeightCommitment::commit(
                        &layout,
                        NonZeroU32::new(claims.layout.advice.row_block).unwrap(),
                        0,
                        &source,
                    )
                    .unwrap(),
                );
            });
            let retained = lower.heap_bytes().unwrap()
                + layout.heap_bytes().unwrap()
                + (source.tensors.capacity() * size_of::<pcs::WeightTensorRef<'_>>()
                    + source.channels.capacity() * size_of::<pcs::ChannelParamsRef<'_>>())
                    as u64
                + caller;
            assert!(
                committed.bytes_max + retained <= weight_bound.unwrap(),
                "{name}: weight constructor {committed:?} + {retained} > {weight_bound:?}"
            );
            let mut metadata = None;
            let checked = allocation_counter::measure(|| {
                metadata = Some(
                    compiler::check_plan_with_memory(&plan, PublicModel::of(&model), &claims)
                        .unwrap(),
                );
            });
            let metadata = metadata.unwrap().unwrap();
            let cores = tessera_witness::Engine::table_memory(&claims.tables)
                .unwrap()
                .peak_bytes();
            assert!(
                checked.bytes_max <= metadata + cores,
                "{name}: {checked:?} > {metadata} + {cores}"
            );
            if name == "tiny-2-servable" {
                let batch = ProofBatch {
                    statement: protocol::PublicStatement {
                        claim_list_digest: tessera_hash::hash(
                            tessera_hash::Domain::ClaimList,
                            &tessera_codec::encode(&claims),
                        ),
                        weights_root: actual.weights_root(),
                        weights_epoch: 0,
                        manifest_digest: [0; 32],
                        batch: protocol::BatchDescriptor {
                            requests: vec![protocol::RequestRef {
                                request_id: 0,
                                session_id: [0; 16],
                                prompt_len: 1,
                                output_len: 0,
                            }],
                        },
                        predictor: protocol::PredictorSpec::Exact {
                            decoding: protocol::Decoding::Greedy,
                        },
                    },
                    tokens: Secret::new(vec![0]),
                };
                let before = {
                    let prover = Prover::new(&plan, &model, &mut state, config.clone()).unwrap();
                    prover.plan(&batch).unwrap().resident_bytes().unwrap()
                };
                let mut spare_batch = ProofBatch {
                    statement: batch.statement.clone(),
                    tokens: Secret::new(Vec::with_capacity(4097)),
                };
                // An invalid token count is not read during public planning; its
                // actual allocation must still be reserved before witness checks.
                spare_batch.statement.batch.requests.reserve(4097);
                let spare = {
                    let prover = Prover::new(&plan, &model, &mut state, config.clone()).unwrap();
                    prover.plan(&spare_batch).unwrap().resident_bytes().unwrap()
                };
                assert_eq!(
                    spare - before,
                    caller_batch_bytes(&spare_batch).unwrap() - caller_batch_bytes(&batch).unwrap()
                );
                let before_heap = plan.heap_upper_bound().unwrap();
                let mut text = String::with_capacity(32 << 20);
                text.push_str(params.id);
                plan.claims.params.0 = Ascii::new(text).unwrap();
                let exact = {
                    let prover = Prover::new(&plan, &model, &mut state, config.clone()).unwrap();
                    let planned = prover.plan(&batch).unwrap();
                    assert_eq!(
                        planned.resident_bytes().unwrap() - before,
                        plan.heap_upper_bound().unwrap() - before_heap
                    );
                    prover.prepared.as_ref().unwrap().memory.peak
                };
                let mut limited = config.clone();
                limited.memory_budget.host = exact - 1;
                // Isolate the new session gate from the existing weight floor.
                assert!(backend::check_weight_memory(&claims, limited.memory_budget).is_ok());
                assert!(matches!(
                    Prover::new(&plan, &model, &mut state, limited.clone()),
                    Err(ProverError::Resource(ErrorCode::ResMemory))
                ));
                limited.memory_budget.host = exact;
                assert!(Prover::new(&plan, &model, &mut state, limited).is_ok());
                let missing = session(
                    &plan,
                    PublicModel::of(&model),
                    &state,
                    &lower,
                    &layout,
                    None,
                );
                assert!(matches!(
                    missing,
                    Err(ProverError::Resource(ErrorCode::ResMemory))
                ));
            }
        }
    }
}

#[cfg(test)]
mod batch_tests {
    use super::*;
    #[test]
    fn caller_batch_is_an_additive_owner_and_overflow_refuses() {
        let mut held = None;
        let observed = allocation_counter::measure(|| {
            held = Some(ProofBatch {
                statement: protocol::PublicStatement {
                    claim_list_digest: [0; 32],
                    weights_root: [0; 32],
                    weights_epoch: 0,
                    manifest_digest: [0; 32],
                    batch: protocol::BatchDescriptor {
                        requests: Vec::with_capacity(31),
                    },
                    predictor: protocol::PredictorSpec::Exact {
                        decoding: protocol::Decoding::Greedy,
                    },
                },
                tokens: Secret::new(Vec::with_capacity(129)),
            });
        });
        let batch = held.unwrap();
        let bytes = caller_batch_bytes(&batch).unwrap();
        assert_eq!(bytes, observed.bytes_current as u64);
        let memory = SessionMemory {
            caller: 7,
            peak: 11,
            weight_construction: 3,
        };
        let charged = memory.with_batch(&batch).unwrap();
        assert_eq!(charged.caller, 7 + bytes);
        assert_eq!(charged.peak, 11 + bytes);
        assert_eq!(charged.weight_construction, 3 + bytes);
        for memory in [
            SessionMemory {
                caller: u64::MAX,
                ..memory
            },
            SessionMemory {
                peak: u64::MAX,
                ..memory
            },
        ] {
            assert!(matches!(
                memory.with_batch(&batch),
                Err(ProverError::Resource(ErrorCode::ResMemory))
            ));
        }
    }
}
