// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public CPU API proofs, checked only by the independent verifier.
use super::*;
use tessera_pcs::{WeightCommitmentPublic, WeightCommitmentState};
use tessera_protocol as protocol;
use tessera_prover::{
    Backend, MaskRng, MemoryBudget, ProofBatch, Prover, ProverConfig, ProverError, ReleaseConfig,
};

mod cli;

fn cfg(mode: Mode) -> ProverConfig {
    ProverConfig {
        backend: Backend::Cpu { threads: 1 },
        params: ParamSet::TEST_SMALL,
        mode,
        memory_budget: MemoryBudget { host: 64 << 30 },
        mask_rng: MaskRng::OsRng,
        release: ReleaseConfig,
    }
}
// Optional table-read chunks may shrink before any reduction stage needs windows.
// Choose the actual ungrouped/unwindowed boundary, not an ample plan's size (#641).
fn window_budget(
    plan: &tessera_compiler::ExecutionPlan,
    model: &IntegerModel,
    state: &mut WeightCommitmentState,
    batch: &ProofBatch,
    config: &ProverConfig,
    high: u64,
) -> Option<u64> {
    let planned = |host, state: &mut WeightCommitmentState| {
        let mut cfg = config.clone();
        cfg.memory_budget.host = host;
        match Prover::new(plan, model, state, cfg).and_then(|p| p.plan(batch)) {
            Ok(plan) => Some(plan),
            Err(ProverError::Resource(tessera_codec::error::ErrorCode::ResMemory)) => None,
            Err(error) => panic!("unexpected fixture planning error: {error}"),
        }
    };
    let (mut low, mut high) = (0, high);
    while high - low > 1 {
        let middle = low + (high - low) / 2;
        let feasible = planned(middle, state).is_some_and(|p| {
            p.resident_bytes().is_some_and(|bytes| bytes <= middle)
                && p.leaf_groups() == protocol::LeafGroups::NONE
                && p.stage_windows().iter().all(|stage| stage.cut == 0)
        });
        if feasible {
            high = middle;
        } else {
            low = middle;
        }
    }
    let host = high.checked_sub(1)?;
    let candidate = planned(host, state)?;
    (candidate
        .resident_bytes()
        .is_some_and(|bytes| bytes <= host)
        && candidate.leaf_groups() == protocol::LeafGroups::NONE
        && candidate.stage_windows().iter().any(|stage| stage.cut > 0))
    .then_some(host)
}

fn producer<T: tessera_codec::Decode>(bytes: &[u8]) -> T {
    tessera_codec::decode(bytes, DecodeLimits::INTEGER_MODEL).unwrap()
}
fn fixture(
    model: &IntegerModel,
    n: u32,
    prompt: u32,
) -> (
    tessera_compiler::ExecutionPlan,
    ProofBatch,
    WeightCommitmentPublic,
) {
    let p = public(
        model,
        n,
        GadgetPolicy {
            max_log2_rows: 18,
            query_block: 2,
            ..Default::default()
        },
    );
    let engine = Engine::new(&p.lowering).unwrap();
    let mut input: Vec<_> = (0..n).map(|i| (i + 1) % model.arch.vocab).collect();
    for t in prompt..n {
        let w = engine.run(model, &Secret::new(input.clone())).unwrap();
        let v = model.arch.vocab as usize;
        let row = &w.tensor(p.lowering.logits).unwrap()[(t as usize - 1) * v..t as usize * v];
        let max = row.iter().max().unwrap();
        input[t as usize] = row.iter().position(|x| x == max).unwrap() as u32;
    }
    let tokens = Secret::new(input);
    let witness = engine.run(model, &tokens).unwrap();
    let runtime =
        tessera_compiler::RuntimePlan::derive(&p.lowering, &p.selection, &p.claims, prompt, 0)
            .unwrap();
    let source = Resident::runtime(&engine, &runtime, model, &witness).unwrap();
    let advice = ModelAdviceLayout::runtime(&p.lowering, &runtime, &p.claims).unwrap();
    let (_, independent) = verifier(&p, n);
    let committed = model_openings::Committed::new(
        &p,
        model,
        &advice,
        &source,
        &source,
        &independent,
        Some(prompt),
    );
    let statement = committed.statement();
    let weights = committed.public_weights();
    let plan = tessera_compiler::plan(PublicModel::of(model), &p.claims).unwrap();
    (plan, ProofBatch { statement, tokens }, weights)
}
fn independent(
    plan: &tessera_compiler::ExecutionPlan,
    batch: &ProofBatch,
    weights: &WeightCommitmentPublic,
) -> (
    tv_core::claims::ClaimList,
    tv_core::protocol::PublicStatement,
    tv_core::VerifierConfig,
) {
    let claims = tv_core::claims::ClaimList::open(
        &envelope::seal(&schema::CLAIM_LIST, &plan.claims),
        &tv_codec::DecodeLimits::MESSAGE,
    )
    .unwrap();
    let packet = protocol::Round1 {
        protocol_version: protocol::PROTOCOL_VERSION,
        session_id: [0; 16],
        statement: batch.statement.clone(),
        advice: protocol::AdviceCommitment {
            root: [0; 32],
            layout_digest: [0; 32],
        },
        leaf_groups: protocol::LeafGroups::NONE,
    };
    let statement = tv_codec::decode(
        &tessera_codec::encode(&packet),
        &tv_codec::DecodeLimits::MESSAGE,
        tv_core::protocol::Round1::decode,
    )
    .unwrap()
    .statement;
    let weights = tv_codec::decode(
        &tessera_codec::encode(weights),
        &tv_codec::DecodeLimits::MESSAGE,
        tv_codec::WeightCommitmentPublic::decode,
    )
    .unwrap();
    let params = tv_core::params::ParamSet::accept(&claims.params)
        .unwrap_or(tv_core::params::ParamSet::Live64);
    (
        claims,
        statement,
        tv_core::VerifierConfig {
            params,
            weights,
            limits: Default::default(),
            accept_bound: tv_core::protocol::FixedBits(0),
        },
    )
}
fn out_of_order<T: core::fmt::Debug>(
    answer: Result<T, ProverError>,
    message: protocol::MessageKind,
) {
    assert!(
        matches!(answer, Err(ProverError::Protocol(protocol::ProtocolError::OutOfOrder { message:m })) if m == message),
        "{answer:?}"
    );
}

fn check_api(model: &IntegerModel, n: u32, prompt: u32) {
    let (plan, batch, weights) = fixture(model, n, prompt);
    let (claims, statement, vcfg) = independent(&plan, &batch, &weights);
    let mut state = WeightCommitmentState::new(weights).unwrap();
    let mut prover = Prover::new(&plan, model, &mut state, cfg(Mode::NonInteractive)).unwrap();
    let dummy_commit = protocol::SeedCommit {
        session_id: [0; 16],
        commitment: [0; 32],
    };
    let dummy_reveal = protocol::SeedReveal {
        session_id: [0; 16],
        share: [0; 32],
    };
    let dummy_queries = protocol::ColumnQueries {
        advice: Vec::new(),
        weights: Vec::new(),
    };
    out_of_order(
        prover.start(&batch, &dummy_commit),
        protocol::MessageKind::SeedCommit,
    );
    out_of_order(
        prover.on_seed(&dummy_reveal),
        protocol::MessageKind::SeedReveal,
    );
    out_of_order(
        prover.on_queries(&dummy_queries),
        protocol::MessageKind::ColumnQueries,
    );
    let proof = prover.prove_non_interactive(&batch).unwrap();
    let geometry = public_opening_geometry(&plan, &batch, proof.round1.leaf_groups);
    assert!(proof.round2.advice_combined.len() <= geometry.groups() + 1);
    cli::check(&plan, &batch, &weights, &proof);
    out_of_order(
        prover.prove_non_interactive(&batch),
        protocol::MessageKind::SeedCommit,
    );
    let decoded = tv_codec::decode(
        &tessera_codec::encode(&proof),
        &tv_codec::DecodeLimits::MESSAGE,
        tv_core::protocol::NiProof::decode,
    )
    .unwrap();
    let verdict = tv_core::verify_non_interactive(&claims, &statement, &decoded, &vcfg);
    if tv_core::params::ParamSet::accept(&claims.params).is_none() {
        assert!(
            matches!(
                verdict,
                tv_core::Verdict::Reject(tv_core::RejectReason::Params { .. })
            ),
            "{verdict:?}"
        );
        return; // The explicit CI test-params step below exercises actual acceptance.
    }
    assert!(
        matches!(verdict, tv_core::Verdict::Accept(_)),
        "{verdict:?}"
    );
    drop(prover);
    assert_eq!(state.public, weights);
    let mut prover = Prover::new(&plan, model, &mut state, cfg(Mode::Live)).unwrap();
    out_of_order(
        prover.prove_non_interactive(&batch),
        protocol::MessageKind::SeedCommit,
    );
    out_of_order(
        prover.on_seed(&dummy_reveal),
        protocol::MessageKind::SeedReveal,
    );
    out_of_order(
        prover.on_queries(&dummy_queries),
        protocol::MessageKind::ColumnQueries,
    );
    let mut verifier = tv_core::LiveVerifier::new(claims, statement, vcfg);
    let commit = verifier.start().unwrap();
    let (round1, share) = prover
        .start(&batch, &producer(&commit.to_tce().unwrap()))
        .unwrap();
    out_of_order(
        prover.start(&batch, &producer(&commit.to_tce().unwrap())),
        protocol::MessageKind::SeedCommit,
    );
    out_of_order(
        prover.on_queries(&dummy_queries),
        protocol::MessageKind::ColumnQueries,
    );
    let round1 = tv_codec::decode(
        &tessera_codec::encode(&round1),
        &tv_codec::DecodeLimits::MESSAGE,
        tv_core::protocol::Round1::decode,
    )
    .unwrap();
    let share = tv_core::protocol::ProverSeedShare {
        session_id: share.session_id,
        share: share.share,
    };
    let reveal: protocol::SeedReveal = producer(
        &verifier
            .on_round1(&round1, &share)
            .unwrap()
            .to_tce()
            .unwrap(),
    );
    let mut other = reveal.clone();
    other.session_id[0] ^= 1;
    assert!(matches!(
        prover.on_seed(&other),
        Err(ProverError::Protocol(
            protocol::ProtocolError::OtherSession { .. }
        ))
    ));
    let round2 = prover.on_seed(&reveal).unwrap();
    assert!(matches!(
        prover.on_seed(&reveal),
        Err(ProverError::Protocol(protocol::ProtocolError::Reuse))
    ));
    let round2 = tv_codec::decode(
        &tessera_codec::encode(&round2),
        &tv_codec::DecodeLimits::MESSAGE,
        tv_core::protocol::Round2Response::decode,
    )
    .unwrap();
    let queries: protocol::ColumnQueries =
        producer(&verifier.on_round2(&round2).unwrap().to_tce().unwrap());
    let mut bad = queries.clone();
    bad.advice.pop();
    assert!(matches!(
        prover.on_queries(&bad),
        Err(ProverError::Protocol(protocol::ProtocolError::Queries(
            protocol::QueryFault::Count { matrix: None, .. }
        )))
    ));
    let round3 = prover.on_queries(&queries).unwrap();
    out_of_order(
        prover.on_queries(&queries),
        protocol::MessageKind::ColumnQueries,
    );
    let round3 = tv_codec::decode(
        &tessera_codec::encode(&round3),
        &tv_codec::DecodeLimits::MESSAGE,
        tv_core::protocol::Round3Response::decode,
    )
    .unwrap();
    let verdict = verifier.on_round3(&round3);
    assert!(
        matches!(verdict, tv_core::Verdict::Accept(_)),
        "{verdict:?}"
    );
}

#[test]
fn public_cpu_api_proofs_verify_in_ni_and_fresh_live_sessions() {
    for name in ["tiny-2-servable", "tiny-2-minimal", "tiny-2-affine"] {
        let model = bounded_model_named(name);
        for (n, prompt) in [(3, 1), (5, 4)] {
            check_api(&model, n, prompt);
        }
    }
}

#[test]
fn public_cpu_api_rejects_configuration_epoch_and_batch_mismatches() {
    use tessera_codec::error::ErrorCode;
    let model = bounded_model();
    let (plan, batch, weights) = fixture(&model, 3, 1);
    let mut state = WeightCommitmentState::new(weights).unwrap();
    for fault in 0..5 {
        let mut c = cfg(Mode::NonInteractive);
        match fault {
            0 => c.backend = Backend::Cpu { threads: 0 },
            1 => c.backend = Backend::Gpu { devices: vec![0] },
            2 => c.params.advice_queries += 1,
            3 => c.params.mode = protocol::ParamMode::Live,
            4 => c.memory_budget.host = 0,
            _ => unreachable!(),
        }
        let error = Prover::new(&plan, &model, &mut state, c).unwrap_err();
        assert_eq!(
            error.code(),
            if fault == 4 {
                ErrorCode::ResMemory
            } else {
                ErrorCode::Usage
            },
            "{error}"
        );
    }
    state.public.roots[0][0] ^= 1;
    assert_eq!(
        Prover::new(&plan, &model, &mut state, cfg(Mode::NonInteractive))
            .unwrap_err()
            .code(),
        ErrorCode::Usage
    );
    state.public = weights;
    state.budget[0].opened_columns = tessera_pcs::OpenedColumns::new(vec![1]).unwrap();
    assert_eq!(
        Prover::new(&plan, &model, &mut state, cfg(Mode::NonInteractive))
            .unwrap_err()
            .code(),
        ErrorCode::Usage
    );
    state.budget[0].opened_columns = tessera_pcs::OpenedColumns::new(Vec::new()).unwrap();
    state.budget[0].next_blinding_row = 1;
    assert_eq!(
        Prover::new(&plan, &model, &mut state, cfg(Mode::NonInteractive))
            .unwrap_err()
            .code(),
        ErrorCode::Usage
    );
    state.budget[0].next_blinding_row = 0;
    for fault in 0..2 {
        let mut bad_plan = plan.clone();
        let mut config = cfg(Mode::NonInteractive);
        if fault == 0 {
            bad_plan.claims.layout.advice.row_block *= 2;
        } else {
            config.params = ParamSet::LIVE_64;
            bad_plan.claims.params = config.params.param_set_id();
            bad_plan.claims.layout = config.params.layout;
        }
        let error = Prover::new(&bad_plan, &model, &mut state, config).unwrap_err();
        assert!(
            matches!(
                error,
                ProverError::Usage {
                    rule: "configuration must match the named claim-list parameters and mode"
                }
            ),
            "{error}"
        );
    }
    for fault in 0..9 {
        let mut bad = ProofBatch {
            statement: batch.statement.clone(),
            tokens: Secret::new(vec![1, 2, 0]),
        };
        match fault {
            0 => bad.statement.claim_list_digest[0] ^= 1,
            1 => bad.statement.weights_root[0] ^= 1,
            2 => bad.statement.weights_epoch += 1,
            3 => bad
                .statement
                .batch
                .requests
                .push(bad.statement.batch.requests[0]),
            4 => bad.statement.batch.requests[0].prompt_len = 0,
            5 => bad.statement.batch.requests[0].output_len = u32::MAX,
            6 => bad.statement.batch.requests[0].prompt_len = model.arch.max_context,
            7 => {
                bad.statement.predictor = protocol::PredictorSpec::Mixture {
                    decoding: protocol::Decoding::Greedy,
                    eps_log2: 6,
                }
            }
            8 => bad.statement.batch.requests.clear(),
            _ => unreachable!(),
        }
        let mut p = Prover::new(&plan, &model, &mut state, cfg(Mode::NonInteractive)).unwrap();
        let error = p.prove_non_interactive(&bad).unwrap_err();
        assert_eq!(error.code(), ErrorCode::Usage, "fault {fault}");
        let rule = match fault {
            0..=2 | 7 | 8 => "an exact-greedy batch must bind this claim list and weight epoch",
            3 => "a batch names each request key once",
            5 => "request length overflow",
            _ => "requests require a nonempty prompt within max_context",
        };
        assert!(
            matches!(error,ProverError::Usage { rule:r } if r == rule),
            "fault {fault}: {error}"
        );
        out_of_order(
            p.prove_non_interactive(&bad),
            protocol::MessageKind::SeedCommit,
        );
    }
    let mut c = cfg(Mode::NonInteractive);
    c.memory_budget.host = 1 << 20;
    let mut p = Prover::new(&plan, &model, &mut state, c).unwrap();
    assert_eq!(
        p.prove_non_interactive(&batch).unwrap_err().code(),
        ErrorCode::ResMemory
    );
}

#[test]
fn public_cpu_api_aborts_on_bad_coin_or_witness_and_on_explicit_abort() {
    use tessera_codec::error::ErrorCode;
    let mut model = bounded_model();
    for w in &mut model.weights {
        if w.ty.matrix() == tessera_pcs::WeightMatrix::LmHead {
            w.data.fill(0);
        }
    }
    let (plan, batch, weights) = fixture(&model, 3, 1);
    let commit = protocol::SeedCommit {
        session_id: [13; 16],
        commitment: protocol::coin_commitment(&[7; 32]),
    };
    let good = protocol::SeedReveal {
        session_id: [13; 16],
        share: [7; 32],
    };
    let bad = protocol::SeedReveal {
        session_id: [13; 16],
        share: [8; 32],
    };
    let mut state = WeightCommitmentState::new(weights).unwrap();
    let mut p = Prover::new(&plan, &model, &mut state, cfg(Mode::Live)).unwrap();
    p.start(&batch, &commit).unwrap();
    assert!(matches!(
        p.on_seed(&bad),
        Err(ProverError::Protocol(protocol::ProtocolError::CoinToss))
    ));
    out_of_order(p.on_seed(&good), protocol::MessageKind::SeedReveal);
    drop(p);
    let bad = ProofBatch {
        statement: batch.statement.clone(),
        tokens: Secret::new(vec![1, 7, 0]),
    };
    let mut p = Prover::new(&plan, &model, &mut state, cfg(Mode::Live)).unwrap();
    assert_eq!(
        p.start(&bad, &commit).unwrap_err().code(),
        ErrorCode::WitnessExactMismatch
    );
    out_of_order(p.start(&batch, &commit), protocol::MessageKind::SeedCommit);
    drop(p);
    let mut p = Prover::new(&plan, &model, &mut state, cfg(Mode::Live)).unwrap();
    {
        let scoped = ProofBatch {
            statement: batch.statement.clone(),
            tokens: Secret::new(vec![1, 0, 0]),
        };
        p.start(&scoped, &commit).unwrap();
    }
    p.on_seed(&good).unwrap();
    p.abort();
    out_of_order(p.on_seed(&good), protocol::MessageKind::SeedReveal);
    drop(p);
    assert_eq!(state.public, weights);
    assert!(
        state
            .budget
            .iter()
            .all(|b| b.next_blinding_row == 0 && b.opened_columns.columns().is_empty())
    );
}

#[cfg(feature = "test-rng")]
#[test]
fn public_cpu_api_test_seed_produces_identical_messages_across_thread_counts() {
    let model = bounded_model_named("tiny-2-affine");
    let (plan, batch, weights) = fixture(&model, 3, 1);
    let mut state = WeightCommitmentState::new(weights).unwrap();
    let mut proofs = Vec::new();
    for threads in [1, 2] {
        let mut config = cfg(Mode::NonInteractive);
        config.backend = Backend::Cpu { threads };
        config.mask_rng = MaskRng::Seeded(17);
        let mut p = Prover::new(&plan, &model, &mut state, config).unwrap();
        proofs.push(tessera_codec::encode(
            &p.prove_non_interactive(&batch).unwrap(),
        ));
    }
    assert_eq!(proofs[0], proofs[1]);
    if let Some(folder) = std::env::var_os("TESSERA_CAPTURE_SINGLE_FIXTURE") {
        let folder = PathBuf::from(folder);
        fs::create_dir_all(&folder).unwrap();
        for (name, bytes) in [
            ("proof.tce", proofs[0].clone()),
            ("claims.tce", tessera_codec::encode(&plan.claims)),
            ("plan.tce", tessera_codec::encode(&plan)),
        ] {
            let path = folder.join(name);
            assert!(
                !path.exists(),
                "capture refuses to overwrite existing artifacts"
            );
            fs::write(path, bytes).unwrap();
        }
    }
}

// RFC-0018 / SP-6: the same actual proof bytes must result from borrowed INT8
// and resident i64 operands, including nonzero time padding and affine sites.
#[cfg(feature = "test-rng")]
#[test]
fn public_cpu_api_checkpoint_and_resident_sources_produce_identical_proofs() {
    for name in ["tiny-2-servable", "tiny-2-minimal", "tiny-2-affine"] {
        let model = bounded_model_named(name);
        let (plan, batch, weights) = fixture(&model, 3, 1);
        let (claims, statement, vcfg) = independent(&plan, &batch, &weights);
        let mut state = WeightCommitmentState::new(weights).unwrap();
        let mut config = cfg(Mode::NonInteractive);
        config.mask_rng = MaskRng::Seeded(17);
        let cached = Prover::new(&plan, &model, &mut state, config.clone())
            .unwrap()
            .plan(&batch)
            .unwrap();
        assert!(cached.uses_checkpoints());
        let mut messages = Vec::new();
        for (host, checkpoint) in [
            (config.memory_budget.host, true),
            (cached.resident_bytes().unwrap() - 1, false),
        ] {
            config.memory_budget.host = host;
            let planned = Prover::new(&plan, &model, &mut state, config.clone())
                .unwrap()
                .plan(&batch)
                .unwrap();
            assert_eq!(planned.uses_checkpoints(), checkpoint);
            assert_eq!(planned.leaf_groups(), protocol::LeafGroups::NONE);
            assert!(planned.stage_windows().iter().all(|s| s.cut == 0));
            let mut proof = None;
            let allocations = allocation_counter::measure(|| {
                proof = Some(
                    Prover::new(&plan, &model, &mut state, config.clone())
                        .unwrap()
                        .prove_non_interactive(&batch)
                        .unwrap(),
                );
            });
            assert!(allocations.bytes_max <= planned.resident_bytes().unwrap());
            let bytes = tessera_codec::encode(&proof.unwrap());
            let decoded = tv_codec::decode(
                &bytes,
                &tv_codec::DecodeLimits::MESSAGE,
                tv_core::protocol::NiProof::decode,
            )
            .unwrap();
            let verdict = tv_core::verify_non_interactive(&claims, &statement, &decoded, &vcfg);
            if tv_core::params::ParamSet::accept(&claims.params).is_some() {
                assert!(
                    matches!(verdict, tv_core::Verdict::Accept(_)),
                    "{name}: {verdict:?}"
                );
            } else {
                assert!(
                    matches!(
                        verdict,
                        tv_core::Verdict::Reject(tv_core::RejectReason::Params { .. })
                    ),
                    "{name}: {verdict:?}"
                );
            }
            messages.push(bytes);
        }
        assert_eq!(messages[0], messages[1], "{name}");
    }
}

/// The pass plan bounds what a proof keeps resident (RFC-0018 'Memory scheduler', #132). A
/// counting allocator measures the whole of `Prover::new` and a non-interactive proof on one
/// thread, the only thread the CPU backend then uses: the peak fits the plan on every
/// conformance model and length. Bounded P2 storage can make the unchanged conservative
/// resident-cell floor exceed four times the measured peak; that is not permission to
/// lower the floor. The plan itself suffices; smaller budgets select a bounded route
/// or fail with `ERR-RES-MEMORY` before P1 commits anything.
#[test]
fn the_pass_plan_bounds_the_measured_peak_and_the_budget_refuses_before_p1() {
    use tessera_codec::error::ErrorCode;
    use tessera_prover::{Pass, PassPlan};
    for name in ["tiny-2-servable", "tiny-2-minimal", "tiny-2-affine"] {
        let model = bounded_model_named(name);
        for (n, prompt) in [(3, 1), (5, 4), (8, 4)] {
            let (plan, batch, weights) = fixture(&model, n, prompt);
            let mut state = WeightCommitmentState::new(weights).unwrap();
            let passes = Prover::new(&plan, &model, &mut state, cfg(Mode::NonInteractive))
                .unwrap()
                .plan(&batch)
                .unwrap();
            assert_eq!(passes.passes()[..2], [Pass::P1, Pass::P2]);
            assert_eq!(
                passes.passes()[passes.passes().len() - 2..],
                [Pass::P4, Pass::P5]
            );
            let bytes = passes.resident_bytes().unwrap();
            let info = allocation_counter::measure(|| {
                Prover::new(&plan, &model, &mut state, cfg(Mode::NonInteractive))
                    .unwrap()
                    .prove_non_interactive(&batch)
                    .unwrap();
            });
            let peak = info.bytes_max;
            assert!(
                peak <= bytes,
                "{name}, n = {n}: measured {peak} bytes against a plan of {bytes}"
            );
            let cells = passes.cells().total().unwrap();
            assert!(bytes >= cells * PassPlan::BYTES_PER_CELL);
            let budgeted = |host: u64| {
                let mut c = cfg(Mode::NonInteractive);
                c.memory_budget.host = host;
                c
            };
            // The plan's own budget proves the leaf stage in one batch.
            let own = Prover::new(&plan, &model, &mut state, budgeted(bytes))
                .unwrap()
                .plan(&batch)
                .unwrap();
            assert_eq!(own, passes);
            assert!(
                Prover::new(&plan, &model, &mut state, budgeted(bytes))
                    .unwrap()
                    .prove_non_interactive(&batch)
                    .is_ok(),
                "{name}, n = {n}: the plan's own budget"
            );
            // One byte short, reduce stage/table work, group the leaf or choose
            // the resident source. Every choice keeps the complete cell floor.
            let grouped = Prover::new(&plan, &model, &mut state, budgeted(bytes - 1))
                .unwrap()
                .plan(&batch)
                .unwrap();
            assert!(
                grouped.leaf_groups() != protocol::LeafGroups::NONE
                    || grouped.stage_windows().iter().any(|s| s.cut > 0)
                    || grouped.table_cuts().iter().any(|&cut| cut > 0)
                    || (passes.uses_checkpoints() && !grouped.uses_checkpoints())
            );
            let grouped_bytes = grouped.resident_bytes().unwrap();
            assert!(grouped_bytes < bytes);
            let info = allocation_counter::measure(|| {
                Prover::new(&plan, &model, &mut state, budgeted(bytes - 1))
                    .unwrap()
                    .prove_non_interactive(&batch)
                    .unwrap();
            });
            assert!(
                info.bytes_max <= grouped_bytes,
                "{name}, n = {n}: measured {} bytes against a grouped plan of {grouped_bytes}",
                info.bytes_max
            );
            // A budget no grouping fits refuses before P1.
            let result = Prover::new(&plan, &model, &mut state, budgeted(1 << 20))
                .unwrap()
                .prove_non_interactive(&batch);
            assert!(
                matches!(result, Err(ProverError::Resource(ErrorCode::ResMemory))),
                "{name}, n = {n}: no grouping fits"
            );
        }
    }
}

/// One live session between `prover` and the independent live verifier: its final verdict.
fn live_verdict(
    prover: &mut Prover<'_>,
    batch: &ProofBatch,
    mut verifier: tv_core::LiveVerifier,
    groups: protocol::LeafGroups,
) -> tv_core::Verdict {
    let commit = verifier.start().unwrap();
    let (round1, share) = prover
        .start(batch, &producer(&commit.to_tce().unwrap()))
        .unwrap();
    assert_eq!(round1.leaf_groups, groups);
    let limits = tv_codec::DecodeLimits::MESSAGE;
    let round1 = tessera_codec::encode(&round1);
    let round1 = tv_codec::decode(&round1, &limits, tv_core::protocol::Round1::decode).unwrap();
    let share = tv_core::protocol::ProverSeedShare {
        session_id: share.session_id,
        share: share.share,
    };
    let reveal: protocol::SeedReveal = producer(
        &verifier
            .on_round1(&round1, &share)
            .unwrap()
            .to_tce()
            .unwrap(),
    );
    let round2 = tessera_codec::encode(&prover.on_seed(&reveal).unwrap());
    let round2 =
        tv_codec::decode(&round2, &limits, tv_core::protocol::Round2Response::decode).unwrap();
    let queries: protocol::ColumnQueries =
        producer(&verifier.on_round2(&round2).unwrap().to_tce().unwrap());
    let round3 = tessera_codec::encode(&prover.on_queries(&queries).unwrap());
    let round3 =
        tv_codec::decode(&round3, &limits, tv_core::protocol::Round3Response::decode).unwrap();
    verifier.on_round3(&round3)
}

/// RFC-0054 'When groups apply': a budget the leaf stage in one batch exceeds takes the fewest
/// token groups that fit, and the proof carries them. The independent verifier accepts each
/// grouped proof, non-interactive and live, at its merged points ('Testing Strategy').
#[test]
fn public_cpu_api_groups_the_leaf_stage_under_a_forced_budget() {
    let model = layered(bounded_model());
    let (plan, batch, weights) = fixture(&model, 5, 4);
    let (claims, statement, vcfg) = independent(&plan, &batch, &weights);
    let certified = tv_core::params::ParamSet::accept(&claims.params).is_some();
    let mut state = WeightCommitmentState::new(weights).unwrap();
    let budgeted_in = |mode: Mode, host: u64| {
        let mut c = cfg(mode);
        c.memory_budget.host = host;
        c
    };
    let budgeted = |host: u64| budgeted_in(Mode::NonInteractive, host);
    let ungrouped = Prover::new(&plan, &model, &mut state, budgeted(64 << 30))
        .unwrap()
        .plan(&batch)
        .unwrap();
    assert_eq!(ungrouped.leaf_groups(), protocol::LeafGroups::NONE);
    for bits in 1..=3u8 {
        // Find the lowest feasible public budget with at most this many group bits. Stage
        // windows now share the memory budget, so one byte below the resident total may use
        // windows first. Physical fallback can create a lower-budget ungrouped island;
        // restrict this monotone search to the resident/grouped strategy (RFC-0018).
        // Its minimal grouping and layer-before-token order remain pinned.
        let (mut low, mut high) = (0, ungrouped.resident_bytes().unwrap());
        while high - low > 1 {
            let middle = low + (high - low) / 2;
            let candidate = Prover::new(&plan, &model, &mut state, budgeted(middle))
                .unwrap()
                .plan(&batch);
            let fits = candidate.is_ok_and(|p| {
                !p.uses_physical_bound()
                    && p.resident_bytes().is_some_and(|bytes| bytes <= middle)
                    && u16::from(p.leaf_groups().token_bits) + u16::from(p.leaf_groups().layer_bits)
                        <= u16::from(bits)
            });
            if fits {
                high = middle;
            } else {
                low = middle;
            }
        }
        let budget = high + 1;
        let grouped = Prover::new(&plan, &model, &mut state, budgeted(budget - 1))
            .unwrap()
            .plan(&batch)
            .unwrap();
        let groups = grouped.leaf_groups();
        // Layers first, the layer index's two bits, then tokens (RFC-0054 'Compute-bound groups').
        let split = [(0, 1), (0, 2), (1, 2)][usize::from(bits) - 1];
        assert_eq!((groups.token_bits, groups.layer_bits), split, "{groups:?}");
        assert!(grouped.passes().contains(&tessera_prover::Pass::P2b));
        let mut proof = None;
        let measured = allocation_counter::measure(|| {
            proof = Some(
                Prover::new(&plan, &model, &mut state, budgeted(budget - 1))
                    .unwrap()
                    .prove_non_interactive(&batch)
                    .unwrap(),
            );
        });
        assert!(
            measured.bytes_max <= grouped.resident_bytes().unwrap(),
            "grouped P2 source/merge coexistence: measured {measured:?}, plan={grouped:?}"
        );
        let proof = proof.unwrap();
        assert_eq!(proof.round1.leaf_groups, groups);
        let geometry = public_opening_geometry(&plan, &batch, groups);
        assert!(proof.round2.advice_combined.len() <= geometry.groups() + 1);
        let trees = &proof.round2.trees;
        assert_eq!(trees.groups.len(), 1 << bits);
        assert!(proof.round2.logup.roots.is_empty() && !trees.table_roots.is_empty());
        assert!(!trees.merge.rounds.is_empty());
        // RFC-0054 'Testing Strategy', 'Determinism': two runs with the same seed are
        // byte-identical, the merge's table rounds and its pass included. Seeded masks are
        // test-rng's, which CI's test-params step enables.
        #[cfg(feature = "test-rng")]
        {
            let seeded = || {
                let mut c = budgeted(budget - 1);
                c.mask_rng = MaskRng::Seeded(17);
                c
            };
            let twice: Vec<Vec<u8>> = (0..2)
                .map(|_| {
                    let mut p = Prover::new(&plan, &model, &mut state, seeded()).unwrap();
                    tessera_codec::encode(&p.prove_non_interactive(&batch).unwrap())
                })
                .collect();
            assert_eq!(twice[0], twice[1], "{groups:?}");
        }
        let decoded = tv_codec::decode(
            &tessera_codec::encode(&proof),
            &tv_codec::DecodeLimits::MESSAGE,
            tv_core::protocol::NiProof::decode,
        )
        .unwrap();
        let verdict = tv_core::verify_non_interactive(&claims, &statement, &decoded, &vcfg);
        if certified {
            assert!(
                matches!(verdict, tv_core::Verdict::Accept(_)),
                "{verdict:?}"
            );
            let mut live = Prover::new(
                &plan,
                &model,
                &mut state,
                budgeted_in(Mode::Live, budget - 1),
            )
            .unwrap();
            let verifier =
                tv_core::LiveVerifier::new(claims.clone(), statement.clone(), vcfg.clone());
            let verdict = live_verdict(&mut live, &batch, verifier, groups);
            assert!(
                matches!(verdict, tv_core::Verdict::Accept(_)),
                "{verdict:?}"
            );
        } else {
            // Without `tv-core/test-params`, `test-small` certifies nothing: CI's test-params
            // step runs this test for acceptance.
            assert!(
                matches!(
                    verdict,
                    tv_core::Verdict::Reject(tv_core::RejectReason::Params { .. })
                ),
                "{verdict:?}"
            );
        }
        // Recorded, on request, as the verifier's grouped driver fixture: public messages only.
        if let Some(folder) = std::env::var_os("TESSERA_CAPTURE_GROUPED_FIXTURE") {
            use tessera_codec::{envelope, schema};
            let folder = PathBuf::from(folder).join(format!("g{bits}"));
            fs::create_dir_all(&folder).unwrap();
            let write = |name: &str, bytes: Vec<u8>| {
                let path = folder.join(name);
                assert!(!path.exists(), "the capture refuses to overwrite {path:?}");
                fs::write(path, bytes).unwrap();
            };
            write(
                "claims.tcl",
                envelope::seal(&schema::CLAIM_LIST, &plan.claims),
            );
            write("weights.tce", tessera_codec::encode(&weights));
            write("ni.tnp", envelope::seal(&schema::NI_PROOF, &proof));
        }
    }
}

#[test]
fn public_cpu_api_selects_real_stage_windows_under_budget() {
    // Metrics are process-global; use a child running this exact test so other proof tests
    // cannot change the manifest counters between the reset and the snapshot.
    const CHILD: &str = "TESSERA_WINDOWED_CPU_METRICS_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "cpu_api::public_cpu_api_selects_real_stage_windows_under_budget",
            ])
            .env(CHILD, "1")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    for name in ["tiny-2-servable", "tiny-2-minimal", "tiny-2-affine"] {
        let model = bounded_model_named(name);
        for n in [3, 5, 8, 16] {
            if n > model.arch.max_context {
                continue;
            }
            let (plan, batch, weights) = fixture(&model, n, n - 1);
            let (claims, statement, verifier_config) = independent(&plan, &batch, &weights);
            let mut state = WeightCommitmentState::new(weights).unwrap();
            let resident = Prover::new(&plan, &model, &mut state, cfg(Mode::NonInteractive))
                .unwrap()
                .plan(&batch)
                .unwrap();
            let total = resident.resident_bytes().unwrap();
            if let Some(host) = window_budget(
                &plan,
                &model,
                &mut state,
                &batch,
                &cfg(Mode::NonInteractive),
                total,
            ) {
                let mut config = cfg(Mode::NonInteractive);
                config.memory_budget.host = host;
                #[cfg(feature = "test-rng")]
                {
                    config.mask_rng = MaskRng::Seeded(19);
                }
                let Ok(candidate) = Prover::new(&plan, &model, &mut state, config.clone())
                    .unwrap()
                    .plan(&batch)
                else {
                    continue;
                };
                if candidate.stage_windows().iter().any(|s| s.cut > 0) {
                    assert!(candidate.passes().contains(&tessera_prover::Pass::P3b));
                    tessera_codec::metrics::reset();
                    let mut proof = None;
                    let info = allocation_counter::measure(|| {
                        proof = Some(
                            Prover::new(&plan, &model, &mut state, config.clone())
                                .unwrap()
                                .prove_non_interactive(&batch)
                                .unwrap(),
                        );
                    });
                    assert!(
                        info.bytes_max <= host,
                        "windowed peak {} exceeds budget {host}",
                        info.bytes_max
                    );
                    let expected: usize = candidate
                        .stage_windows()
                        .iter()
                        .map(|s| s.fold_passes)
                        .sum();
                    let samples = tessera_codec::metrics::snapshot();
                    let recorded = samples
                        .iter()
                        .find(|s| {
                            s.name == "tessera_passes_total"
                                && s.labels.iter().any(|(k, v)| *k == "pass" && v == "p3b")
                        })
                        .unwrap();
                    assert_eq!(
                        recorded.value,
                        tessera_codec::metrics::Value::Counter(expected as u64)
                    );
                    assert_eq!(
                        candidate
                            .passes()
                            .iter()
                            .filter(|&&p| p == tessera_prover::Pass::P3b)
                            .count(),
                        expected
                    );
                    let proof = proof.unwrap();
                    // 05 §4: independently materialize the published parts after measuring
                    // the prover peak; these test-only buffers are outside that measurement.
                    for (part, bytes) in [
                        ("round1", tessera_codec::encode(&proof.round1).len()),
                        (
                            "sumcheck",
                            tessera_codec::encode(&proof.round2.logup).len()
                                + tessera_codec::encode(&proof.round2.trees).len()
                                + tessera_codec::encode(&proof.round2.stages).len(),
                        ),
                        (
                            "openings",
                            tessera_codec::encode(&proof.round2.advice_combined).len()
                                + tessera_codec::encode(&proof.round2.weight_combined).len(),
                        ),
                        ("columns", tessera_codec::encode(&proof.round3).len()),
                    ] {
                        let metric = samples
                            .iter()
                            .find(|sample| {
                                sample.name == "tessera_proof_bytes"
                                    && sample
                                        .labels
                                        .iter()
                                        .any(|(key, value)| *key == "part" && value == part)
                            })
                            .expect("every proof-byte part is recorded");
                        assert_eq!(
                            metric.value,
                            tessera_codec::metrics::Value::Gauge(bytes as f64),
                            "published {part} bytes equal the complete canonical encoding"
                        );
                    }
                    let decoded = tv_codec::decode(
                        &tessera_codec::encode(&proof),
                        &tv_codec::DecodeLimits::MESSAGE,
                        tv_core::protocol::NiProof::decode,
                    )
                    .unwrap();
                    let verdict = tv_core::verify_non_interactive(
                        &claims,
                        &statement,
                        &decoded,
                        &verifier_config,
                    );
                    if tv_core::params::ParamSet::accept(&claims.params).is_some() {
                        assert!(
                            matches!(verdict, tv_core::Verdict::Accept(_)),
                            "{verdict:?}"
                        );
                        let mut live_config = config.clone();
                        live_config.mode = Mode::Live;
                        let mut live = Prover::new(&plan, &model, &mut state, live_config).unwrap();
                        let verifier = tv_core::LiveVerifier::new(
                            claims.clone(),
                            statement.clone(),
                            verifier_config.clone(),
                        );
                        let verdict =
                            live_verdict(&mut live, &batch, verifier, candidate.leaf_groups());
                        assert!(
                            matches!(verdict, tv_core::Verdict::Accept(_)),
                            "{verdict:?}"
                        );
                    } else {
                        assert!(
                            matches!(
                                verdict,
                                tv_core::Verdict::Reject(tv_core::RejectReason::Params { .. })
                            ),
                            "{verdict:?}"
                        );
                    }
                    #[cfg(feature = "test-rng")]
                    if candidate.leaf_groups() == resident.leaf_groups() {
                        let mut literal_config = config.clone();
                        literal_config.memory_budget.host = total;
                        let literal = Prover::new(&plan, &model, &mut state, literal_config)
                            .unwrap()
                            .prove_non_interactive(&batch)
                            .unwrap();
                        assert_eq!(
                            tessera_codec::encode(&proof),
                            tessera_codec::encode(&literal)
                        );
                    }
                    if let Some(path) = std::env::var_os("TESSERA_WINDOWED_CPU_EVIDENCE") {
                        let path = PathBuf::from(path);
                        assert!(!path.exists(), "evidence capture refuses to overwrite");
                        let stages: Vec<_> = candidate.stage_windows().iter().enumerate().map(|(k,s)| serde_json::json!({"stage":k,"cut":s.cut,"fold_passes":s.fold_passes,"working_bytes":s.working_bytes})).collect();
                        let evidence = serde_json::json!({"schema_version":1,"model":name,"tokens":n,"resident_plan_bytes":total,"budget_bytes":host,"windowed_plan_bytes":candidate.resident_bytes().unwrap(),"measured_peak_bytes":info.bytes_max,"p3b":expected,"stages":stages,"independent_verifier_accepts":tv_core::params::ParamSet::accept(&claims.params).is_some()});
                        fs::write(path, serde_json::to_vec_pretty(&evidence).unwrap()).unwrap();
                    }
                    return;
                }
            }
        }
    }
    panic!("the conformance budgets must exercise the production window path");
}

#[test]
fn public_cpu_api_plans_and_proves_two_slotted_requests() {
    check_slotted_batch(&[(1, 1), (1, 2)]);
}

#[test]
fn public_cpu_api_proves_three_slots_with_an_outputless_request() {
    check_slotted_batch(&[(1, 0), (1, 2), (2, 1)]);
}

#[test]
fn public_cpu_api_proves_three_outputless_slots() {
    check_slotted_batch(&[(1, 0), (2, 0), (3, 0)]);
}

#[test]
fn public_cpu_api_proves_one_token_requests_without_a_key_forest() {
    check_slotted_batch(&[(1, 0), (1, 0)]);
    check_slotted_batch(&[(1, 0), (1, 0), (1, 0)]);
}

#[test]
fn public_cpu_api_proves_slotted_batches_through_budget_windows() {
    check_slotted_batch_with_windows(&[(1, 1), (1, 2)], true);
}

fn check_slotted_batch(requests: &[(u32, u32)]) {
    check_slotted_batch_with_windows(requests, false);
}

fn check_slotted_batch_with_windows(requests: &[(u32, u32)], windows: bool) {
    let model = bounded_model();
    let (plan, mut batch, weights) = fixture(&model, 3, 1);
    let template = batch.statement.batch.requests[0];
    batch.statement.batch.requests = requests
        .iter()
        .enumerate()
        .map(|(r, &(prompt_len, output_len))| protocol::RequestRef {
            request_id: template.request_id + r as u64,
            prompt_len,
            output_len,
            ..template
        })
        .collect();
    let lengths: Vec<_> = requests.iter().map(|&(p, o)| p + o).collect();
    let slot = *lengths.iter().max().unwrap();
    let slot = slot.next_power_of_two() as usize;
    let lowering = tessera_compiler::lower_batch(PublicModel::of(&model), &lengths).unwrap();
    let engine = Engine::new(&lowering).unwrap();
    let mut tokens = vec![0u32; requests.len() * slot];
    for (r, &(prompt, outputs)) in requests.iter().enumerate() {
        let base = r * slot;
        tokens[base..base + prompt as usize].fill(1);
        for t in base + prompt as usize..base + (prompt + outputs) as usize {
            let witness = engine.run(&model, &Secret::new(tokens.clone())).unwrap();
            let v = model.arch.vocab as usize;
            let row = &witness.tensor(lowering.logits).unwrap()[(t - 1) * v..t * v];
            let max = row.iter().max().unwrap();
            tokens[t] = row.iter().position(|x| x == max).unwrap() as u32;
        }
    }
    let gap_tokens = lengths
        .iter()
        .enumerate()
        .find(|(_, n)| **n < slot as u32)
        .map(|(r, &n)| {
            let mut bad = tokens.clone();
            bad[r * slot + n as usize] = 1;
            bad
        });
    let tokens: Vec<_> = lengths
        .iter()
        .enumerate()
        .flat_map(|(r, &n)| tokens[r * slot..r * slot + n as usize].iter().copied())
        .collect();
    let mut bad_tokens = tokens.clone();
    let changed = requests
        .iter()
        .enumerate()
        .rfind(|(_, (_, outputs))| *outputs > 0)
        .map(|(r, &(p, o))| {
            lengths[..r].iter().map(|&n| n as usize).sum::<usize>() + (p + o - 1) as usize
        });
    if let Some(at) = changed {
        bad_tokens[at] = (bad_tokens[at] + 1) % model.arch.vocab;
    }
    batch.tokens = Secret::new(tokens);
    let (claims, statement, vcfg) = independent(&plan, &batch, &weights);
    let mut state = WeightCommitmentState::new(weights).unwrap();
    let mut config = cfg(Mode::NonInteractive);
    let resident = Prover::new(&plan, &model, &mut state, config.clone())
        .unwrap()
        .plan(&batch)
        .unwrap();
    if windows {
        config.memory_budget.host = window_budget(
            &plan,
            &model,
            &mut state,
            &batch,
            &config,
            resident.resident_bytes().unwrap(),
        )
        .expect("the slotted fixture must exercise budget windows");
    }
    let mut prover = Prover::new(&plan, &model, &mut state, config.clone()).unwrap();
    let selected = prover.plan(&batch).unwrap();
    if windows {
        assert!(selected.stage_windows().iter().any(|s| s.cut > 0));
    }
    if let Some(tokens) = gap_tokens {
        let bad = ProofBatch {
            statement: batch.statement.clone(),
            tokens: Secret::new(tokens),
        };
        assert_eq!(
            prover.check_witness(&bad).unwrap_err().code(),
            tessera_codec::error::ErrorCode::Usage
        );
    }
    assert!(prover.plan(&batch).is_ok());
    prover.check_witness(&batch).unwrap();
    let mut proof = None;
    let measured = allocation_counter::measure(|| {
        proof = Some(prover.prove_non_interactive(&batch).unwrap());
    });
    if windows {
        assert!(
            measured.bytes_max <= config.memory_budget.host,
            "slotted window peak: {measured:?}"
        );
    }
    let proof = proof.unwrap();
    let geometry = public_opening_geometry(&plan, &batch, proof.round1.leaf_groups);
    assert!(proof.round2.advice_combined.len() <= geometry.groups() + 1);

    if tv_core::params::ParamSet::accept(&claims.params).is_some() {
        let decoded = tv_codec::decode(
            &tessera_codec::encode(&proof),
            &tv_codec::DecodeLimits::MESSAGE,
            tv_core::protocol::NiProof::decode,
        )
        .unwrap();
        let verdict = tv_core::verify_non_interactive(&claims, &statement, &decoded, &vcfg);
        assert!(
            matches!(verdict, tv_core::Verdict::Accept(_)),
            "{verdict:?}"
        );
    }
    assert_eq!(
        proof.round2.scoring.requests,
        vec![protocol::FixedBits(0); requests.len()]
    );
    drop(prover);
    if tv_core::params::ParamSet::accept(&claims.params).is_some() {
        let verifier = tv_core::LiveVerifier::new(claims, statement, vcfg);
        config.mode = Mode::Live;
        let mut live = Prover::new(&plan, &model, &mut state, config).unwrap();
        let verdict = live_verdict(&mut live, &batch, verifier, selected.leaf_groups());
        assert!(
            matches!(verdict, tv_core::Verdict::Accept(_)),
            "{verdict:?}"
        );
    }
    if changed.is_none() {
        return;
    }
    let bad = ProofBatch {
        statement: batch.statement.clone(),
        tokens: Secret::new(bad_tokens),
    };
    let mut p = Prover::new(&plan, &model, &mut state, cfg(Mode::NonInteractive)).unwrap();
    assert_eq!(
        p.prove_non_interactive(&bad).unwrap_err().code(),
        tessera_codec::error::ErrorCode::WitnessExactMismatch
    );
}

/// #598: layer ownership at lambda=0 must survive the complete CPU/PCS route.
#[cfg(feature = "test-params")]
#[test]
fn public_cpu_api_proves_one_layer_models_in_ni_and_live() {
    for name in ["tiny-2-minimal", "tiny-2-servable", "tiny-2-affine"] {
        let mut model = bounded_model_named(name);
        model.arch.n_layers = 1;
        model.weights.retain(|w| w.layer == 0);
        model.quant.channels.retain(|c| c.layer == 0);
        let keep = |p: tessera_ir::QuantPoint| {
            !matches!(p,
            tessera_ir::QuantPoint::Layer { layer, .. } if layer != 0)
        };
        model.quant.public.act_exponents.retain(|(p, _)| keep(*p));
        model
            .quant
            .public
            .shift_class_sets
            .retain(|(p, _)| keep(*p));
        model.quant.public.affine_sites.retain(|(p, _)| keep(*p));
        model.quant.public.score_scales.retain(|(l, _)| *l == 0);
        model.quant.silu_in_exponents.retain(|(l, _)| *l == 0);
        let (plan, batch, weights) = fixture(&model, 3, 1);
        let (claims, statement, config) = independent(&plan, &batch, &weights);
        let mut state = WeightCommitmentState::new(weights).unwrap();
        let proof = Prover::new(&plan, &model, &mut state, cfg(Mode::NonInteractive))
            .unwrap()
            .prove_non_interactive(&batch)
            .unwrap();
        let geometry = public_opening_geometry(&plan, &batch, proof.round1.leaf_groups);
        assert!(proof.round2.advice_combined.len() <= geometry.groups() + 1);
        let proof = tv_codec::decode(
            &tessera_codec::encode(&proof),
            &tv_codec::DecodeLimits::MESSAGE,
            tv_core::protocol::NiProof::decode,
        )
        .unwrap();
        let verdict = tv_core::verify_non_interactive(&claims, &statement, &proof, &config);
        assert!(
            matches!(verdict, tv_core::Verdict::Accept(_)),
            "{name}: {verdict:?}"
        );
        let verifier = tv_core::LiveVerifier::new(claims, statement, config);
        let mut prover = Prover::new(&plan, &model, &mut state, cfg(Mode::Live)).unwrap();
        let verdict = live_verdict(&mut prover, &batch, verifier, protocol::LeafGroups::NONE);
        assert!(
            matches!(verdict, tv_core::Verdict::Accept(_)),
            "{name}: {verdict:?}"
        );
    }
}

#[test]
fn public_cpu_api_proves_heterogeneous_requant_layers() {
    for (d_ff, first_shift, second_shift) in [(6, 2, 12), (24, 2, 12), (24, 1, 5)] {
        let mut model = bounded_model();
        model.arch.d_ff = d_ff;
        let shapes = model.arch.weight_shapes().unwrap();
        for w in &mut model.weights {
            let shape = shapes[w.ty as usize];
            w.rows = shape.out;
            w.cols = shape.inputs;
            w.data.resize((shape.out * shape.inputs) as usize, 0);
        }
        for channel in &mut model.quant.channels {
            let n = shapes[channel.ty as usize].out as usize;
            if !channel.class.is_empty() {
                channel.class.resize(n, 0);
            }
            if !channel.multiplier.is_empty() {
                channel.multiplier.resize(n, 1);
            }
        }
        for (point, shifts) in &mut model.quant.public.shift_class_sets {
            if let tessera_ir::QuantPoint::Layer {
                layer,
                site: tessera_ir::Site::MlpResidual,
            } = *point
            {
                *shifts = vec![if layer == 0 {
                    first_shift
                } else {
                    second_shift
                }];
            }
        }
        // These Down sites now have one pot class, so they carry no channel vector.
        model
            .quant
            .channels
            .retain(|c| c.ty != tessera_ir::WeightType::Down);
        tessera_compiler::validate(&model).unwrap();
        let policy = GadgetPolicy {
            max_log2_rows: 18,
            query_block: 2,
            ..Default::default()
        };
        let p = public(&model, 3, policy);
        let ranges = analyze(&p.lowering.graph, &p.lowering.tables).unwrap();
        let old = tessera_compiler::select(&p.lowering.graph, &ranges, &p.lowering.tables, &policy)
            .unwrap();
        let legacy = claim_list(
            PublicModel::of(&model),
            &old,
            &policy,
            ParamSet::TEST_SMALL.layout,
            ParamSetId(Ascii::new(ParamSet::TEST_SMALL.id).unwrap()),
        )
        .unwrap();
        let first_difference = old
            .tables
            .iter()
            .zip(&p.selection.tables)
            .position(|(a, b)| a != b)
            .unwrap_or(old.tables.len().min(p.selection.tables.len()));
        let mut stale = tv_core::claims::ClaimList::open(
            &envelope::seal(&schema::CLAIM_LIST, &legacy),
            &tv_codec::DecodeLimits::MESSAGE,
        )
        .unwrap();
        // The table check precedes layout checks; a production parameter name also
        // exercises this rejection when the test-small verifier feature is disabled.
        stale.params = tv_core::claims::ParamSetId::new("ni-100/1").unwrap();
        assert_eq!(
            tv_core::validation::validate(&stale).err(),
            Some(tv_core::validation::ClaimsError::StatementMismatch {
                field: tv_core::validation::StatementField::Table {
                    table: tv_tables::TableId(first_difference as u32)
                }
            })
        );
        assert!(
            matches!(LeafLayout::model(&p.lowering,&p.layers,&old,&legacy),
        Err(tessera_prover::leaf::LeafError::Internal{rule})
        if rule=="a site's real layers read the same tensors over the same axes (RFC-0045 'One claim per input')")
        );
        assert_ne!(p.selection.tables, old.tables);
        LeafLayout::model(&p.lowering, &p.layers, &p.selection, &p.claims).unwrap();
        let (graph, claims) = verifier(&p, 3);
        let selected = tv_core::selection::derive(&graph, claims.gadget_policy).unwrap();
        assert_eq!(selected.tables.len(), p.selection.tables.len());
        for (actual, expected) in selected.tables.iter().zip(&p.selection.tables) {
            assert_eq!(actual.id.0, expected.id.0);
            assert_eq!(actual.log2_rows, expected.log2_rows);
            // Whole canonical table bytes are also compared by independent claim validation
            // and CLI acceptance in check_api, including their digests and repeated reads.
        }
        split::Layout::derive(&graph, claims.gadget_policy).unwrap();
        check_api(&model, 3, 1);
    }
}

// Public metadata only: the helper never reads ProofBatch.tokens or a model weight.
fn public_opening_geometry(
    plan: &tessera_compiler::ExecutionPlan,
    batch: &ProofBatch,
    groups: protocol::LeafGroups,
) -> tessera_prover::OpeningGeometry {
    let c = &plan.claims;
    let requests: Vec<_> = batch
        .statement
        .batch
        .requests
        .iter()
        .map(|r| (r.prompt_len, r.output_len))
        .collect();
    let lengths: Vec<_> = requests
        .iter()
        .map(|&(p, o)| p.checked_add(o).unwrap())
        .collect();
    let lowering = tessera_compiler::lower_batch(PublicModel::of_claims(c), &lengths).unwrap();
    let ranges = analyze(&lowering.graph, &lowering.tables).unwrap();
    let selection = select_model(c.protocol_version, &lowering, &ranges, &c.gadget_policy).unwrap();
    let runtime =
        tessera_compiler::RuntimePlan::derive_batch(&lowering, &selection, c, &requests).unwrap();
    let layers = LayerMap::new(&lowering).unwrap();
    let leaf = LeafLayout::runtime(&lowering, &layers, &runtime, c).unwrap();
    let native =
        tessera_prover::native::ModelNativePlan::derive_runtime(&lowering, &runtime, c, &leaf)
            .unwrap();
    let advice = ModelAdviceLayout::runtime(&lowering, &runtime, c).unwrap();
    let weight_layout = tessera_pcs::WeightLayout::new(
        c.layout.weights,
        tessera_pcs::WeightShapes {
            layers: c.arch.n_layers,
            shapes: c.arch.weight_shapes().unwrap(),
            channels: lowering
                .classes
                .iter()
                .chain(&lowering.multipliers)
                .map(|(w, _)| (w.ty.matrix(), w.layer))
                .collect(),
        },
    )
    .unwrap();
    let grouped = (groups != protocol::LeafGroups::NONE)
        .then(|| tessera_prover::leaf::GroupedLayout::new(&leaf, &layers, groups).unwrap());
    tessera_prover::OpeningGeometry::derive(
        &lowering,
        &runtime.selection().tables,
        &leaf,
        grouped.as_ref(),
        &native,
        &advice,
        &weight_layout,
    )
    .unwrap()
}
