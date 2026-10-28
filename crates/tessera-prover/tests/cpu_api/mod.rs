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

/// The pass plan bounds what a proof keeps resident (RFC-0018 'Memory scheduler', #132). A
/// counting allocator measures the whole of `Prover::new` and a non-interactive proof on one
/// thread, the only thread the CPU backend then uses: the peak lies between a quarter of the plan
/// and the plan, on every conformance model and length. A budget one byte short of the plan fails
/// with `ERR-RES-MEMORY` before P1 commits anything, and the plan itself suffices.
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
                peak <= bytes && bytes <= 4 * peak,
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
            // One byte short, the leaf stage takes token groups (RFC-0054): the grouped plan
            // fits, and the grouped proof's measured peak stays within it.
            let grouped = Prover::new(&plan, &model, &mut state, budgeted(bytes - 1))
                .unwrap()
                .plan(&batch)
                .unwrap();
            assert_ne!(grouped.leaf_groups(), protocol::LeafGroups::NONE);
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
    let mut budget = ungrouped.resident_bytes().unwrap();
    for bits in 1..=3u8 {
        let grouped = Prover::new(&plan, &model, &mut state, budgeted(budget - 1))
            .unwrap()
            .plan(&batch)
            .unwrap();
        let groups = grouped.leaf_groups();
        // Layers first, the layer index's two bits, then tokens (RFC-0054 'Compute-bound groups').
        let split = [(0, 1), (0, 2), (1, 2)][usize::from(bits) - 1];
        assert_eq!((groups.token_bits, groups.layer_bits), split, "{groups:?}");
        assert!(grouped.passes().contains(&tessera_prover::Pass::P2b));
        let proof = Prover::new(&plan, &model, &mut state, budgeted(budget - 1))
            .unwrap()
            .prove_non_interactive(&batch)
            .unwrap();
        assert_eq!(proof.round1.leaf_groups, groups);
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
        budget = grouped.resident_bytes().unwrap();
    }
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

fn check_slotted_batch(requests: &[(u32, u32)]) {
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
    let mut prover = Prover::new(&plan, &model, &mut state, cfg(Mode::NonInteractive)).unwrap();
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
    let proof = prover.prove_non_interactive(&batch).unwrap();
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
        let mut live = Prover::new(&plan, &model, &mut state, cfg(Mode::Live)).unwrap();
        let verdict = live_verdict(&mut live, &batch, verifier, protocol::LeafGroups::NONE);
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
