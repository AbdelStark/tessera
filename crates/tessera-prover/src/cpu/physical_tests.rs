// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Whole-route controls use the production chooser and independent verifier.
use super::*;
use tessera_codec::{DecodeLimits, envelope, schema};

struct Route;
impl Route {
    fn enable() -> Self {
        physical::FORCE_ROUTE.with(|v| v.set(true));
        Self
    }
}
impl Drop for Route {
    fn drop(&mut self) {
        physical::FORCE_ROUTE.with(|v| v.set(false));
    }
}

#[test]
fn physical_route_matches_resident_and_independently_verifies() {
    // One test serializes the process-wide pass registry across all fixtures.
    for (name, requests) in [
        ("tiny-2-servable", &[(3, 0)][..]),
        ("tiny-2-affine", &[(1, 1)][..]),
        ("tiny-2-servable", &[(2, 1), (1, 0)][..]),
    ] {
        physical_case(name, requests);
    }
}

fn physical_case(name: &str, requests: &[(u32, u32)]) {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors");
    let mut model: IntegerModel = envelope::open(
        &std::fs::read(root.join(format!("models/{name}.d"))).unwrap(),
        &schema::INTEGER_MODEL,
        DecodeLimits::INTEGER_MODEL,
    )
    .unwrap();
    model.arch.d_model = 2;
    model.arch.n_heads = 1;
    model.arch.n_kv_heads = 1;
    model.arch.d_head = 2;
    model.arch.d_ff = 6;
    model.arch.vocab = 8;
    model.arch.max_context = 8;
    model.quant.public.rsqrt_kappa = 2;
    model.quant.public.exp_sigma = 3;
    model.quant.public.residual_bits = 8;
    model.quant.public.residual_exponent = model
        .quant
        .public
        .act_exponents
        .iter()
        .find(|(p, _)| *p == tessera_ir::QuantPoint::Embed)
        .unwrap()
        .1;
    let shifts = model.quant.public.shift_class_sets.clone();
    for (point, values) in &mut model.quant.public.shift_class_sets {
        if let tessera_ir::QuantPoint::Layer { site, .. } = *point {
            *values = shifts
                .iter()
                .find(|(p, _)| *p == tessera_ir::QuantPoint::Layer { layer: 0, site })
                .unwrap()
                .1
                .clone();
        }
    }
    let affine = model.quant.public.affine_sites.clone();
    for (point, value) in &mut model.quant.public.affine_sites {
        if let tessera_ir::QuantPoint::Layer { site, .. } = *point {
            *value = affine
                .iter()
                .find(|(p, _)| *p == tessera_ir::QuantPoint::Layer { layer: 0, site })
                .unwrap()
                .1;
        }
    }
    if name != "tiny-2-servable" {
        for (point, shifts) in &mut model.quant.public.shift_class_sets {
            let base = if matches!(
                point,
                tessera_ir::QuantPoint::Layer {
                    site: tessera_ir::Site::Silu,
                    ..
                }
            ) {
                12
            } else {
                24
            };
            for (class, shift) in shifts.iter_mut().enumerate() {
                *shift = base + class as i8;
            }
        }
        for (_, shift) in &mut model.quant.public.affine_sites {
            *shift = 24;
        }
    }
    let shapes = model.arch.weight_shapes().unwrap();
    for weight in &mut model.weights {
        let shape = shapes[weight.ty as usize];
        weight.rows = shape.out;
        weight.cols = shape.inputs;
        weight.data = (0..shape.out * shape.inputs)
            .map(|i| (i % 5) as i8 - 2)
            .collect();
    }
    for channel in &mut model.quant.channels {
        let n = shapes[channel.ty as usize].out as usize;
        if let Some(&value) = channel.class.first() {
            channel.class.resize(n, value);
            channel.class.fill(0);
        }
        if let Some(&value) = channel.multiplier.first() {
            channel.multiplier.resize(n, value);
        }
    }
    let lengths: Vec<_> = requests.iter().map(|(p, o)| p + o).collect();
    let lower = compiler::lower_batch(PublicModel::of(&model), &lengths).unwrap();
    let ranges = compiler::analyze(&lower.graph, &lower.tables).unwrap();
    let policy = tessera_ir::GadgetPolicy {
        max_log2_rows: 18,
        query_block: 2,
        ..Default::default()
    };
    let selection =
        compiler::select_model(protocol::PROTOCOL_VERSION, &lower, &ranges, &policy).unwrap();
    let params = ParamSet::TEST_SMALL;
    let claims = compiler::claim_list(
        PublicModel::of(&model),
        &selection,
        &policy,
        params.layout,
        tessera_hash::ParamSetId(Ascii::new(params.id).unwrap()),
    )
    .unwrap();
    let plan = compiler::plan(PublicModel::of(&model), &claims).unwrap();
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
    let weight = pcs::RegeneratedWeightCommitment::commit(
        &layout,
        NonZeroU32::new(claims.layout.advice.row_block).unwrap(),
        0,
        &source,
    )
    .unwrap();
    let mut state = pcs::WeightCommitmentState::new(*weight.public()).unwrap();
    let mut recorded = Vec::new();
    for &(prompt, output) in requests {
        let n = prompt + output;
        let one = compiler::lower(PublicModel::of(&model), n).unwrap();
        let engine = tessera_witness::Engine::new(&one).unwrap();
        let mut tokens: Vec<_> = (0..n).map(|i| i % model.arch.vocab).collect();
        for position in prompt..n {
            let witness = engine.run(&model, &Secret::new(tokens.clone())).unwrap();
            let vocab = model.arch.vocab as usize;
            let row = &witness.tensor(one.logits).unwrap()
                [(position as usize - 1) * vocab..position as usize * vocab];
            let maximum = row.iter().max().unwrap();
            tokens[position as usize] = row.iter().position(|v| v == maximum).unwrap() as u32;
        }
        recorded.extend(tokens);
    }
    let batch = ProofBatch {
        statement: protocol::PublicStatement {
            claim_list_digest: tessera_hash::hash(
                tessera_hash::Domain::ClaimList,
                &tessera_codec::encode(&claims),
            ),
            weights_root: weight.weights_root(),
            weights_epoch: 0,
            manifest_digest: [0; 32],
            batch: protocol::BatchDescriptor {
                requests: requests
                    .iter()
                    .enumerate()
                    .map(|(i, &(prompt_len, output_len))| protocol::RequestRef {
                        request_id: i as u64,
                        session_id: [i as u8; 16],
                        prompt_len,
                        output_len,
                    })
                    .collect(),
            },
            predictor: protocol::PredictorSpec::Exact {
                decoding: protocol::Decoding::Greedy,
            },
        },
        tokens: Secret::new(recorded),
    };
    let config = ProverConfig {
        backend: Backend::Cpu { threads: 1 },
        params,
        mode: Mode::NonInteractive,
        memory_budget: MemoryBudget { host: 64 << 30 },
        mask_rng: MaskRng::Seeded(17),
        release: ReleaseConfig,
    };
    let resident = Prover::new(&plan, &model, &mut state, config.clone())
        .unwrap()
        .prove_non_interactive(&batch)
        .unwrap();
    let _route = Route::enable();
    let planned = Prover::new(&plan, &model, &mut state, config.clone())
        .unwrap()
        .plan(&batch)
        .unwrap();
    assert!(planned.uses_physical_bound());
    assert!(planned.uses_checkpoints());
    assert!(planned.stage_windows().iter().skip(1).all(|w| w.cut > 0));
    metrics::reset();
    let mut proof = None;
    let measured = allocation_counter::measure(|| {
        proof = Some(
            Prover::new(&plan, &model, &mut state, config.clone())
                .unwrap()
                .prove_non_interactive(&batch)
                .unwrap(),
        );
    });
    assert!(
        measured.bytes_max <= planned.resident_bytes().unwrap(),
        "{measured:?} > {planned:?}"
    );
    let expected: usize = planned.stage_windows().iter().map(|w| w.fold_passes).sum();
    assert!(expected > 0);
    let samples = metrics::snapshot();
    let folds = samples
        .iter()
        .find(|s| {
            s.name == "tessera_passes_total"
                && s.labels.iter().any(|(k, v)| *k == "pass" && v == "p3b")
        })
        .unwrap();
    assert_eq!(folds.value, metrics::Value::Counter(expected as u64));
    let proof = proof.unwrap();
    assert_eq!(
        tessera_codec::encode(&proof),
        tessera_codec::encode(&resident)
    );
    let vc = tv_core::claims::ClaimList::open(
        &envelope::seal(&schema::CLAIM_LIST, &claims),
        &tv_codec::DecodeLimits::MESSAGE,
    )
    .unwrap();
    let decode = |proof: &protocol::NiProof| {
        tv_codec::decode(
            &tessera_codec::encode(proof),
            &tv_codec::DecodeLimits::MESSAGE,
            tv_core::protocol::NiProof::decode,
        )
        .unwrap()
    };
    let decoded = decode(&proof);
    let vcfg = tv_core::VerifierConfig {
        params: tv_core::params::ParamSet::accept(&vc.params).unwrap(),
        weights: tv_codec::decode(
            &tessera_codec::encode(weight.public()),
            &tv_codec::DecodeLimits::MESSAGE,
            tv_codec::WeightCommitmentPublic::decode,
        )
        .unwrap(),
        limits: Default::default(),
        accept_bound: tv_core::protocol::FixedBits(0),
    };
    // Caller pins come from the public input, never from the generated proof.
    let expected = tv_core::protocol::PublicStatement {
        claim_list_digest: vc.digest(),
        weights_root: batch.statement.weights_root,
        weights_epoch: batch.statement.weights_epoch,
        manifest_digest: batch.statement.manifest_digest,
        batch: tv_core::protocol::BatchDescriptor {
            requests: batch
                .statement
                .batch
                .requests
                .iter()
                .map(|r| tv_core::protocol::RequestRef {
                    request_id: r.request_id,
                    session_id: r.session_id,
                    prompt_len: r.prompt_len,
                    output_len: r.output_len,
                })
                .collect(),
        },
        predictor: tv_core::protocol::PredictorSpec::Exact {
            decoding: tv_core::protocol::Decoding::Greedy,
        },
    };
    let verdict = tv_core::verify_non_interactive(&vc, &expected, &decoded, &vcfg);
    assert!(
        matches!(verdict, tv_core::Verdict::Accept(_)),
        "{verdict:?}"
    );
    let mut tampered = proof.clone();
    let stage = tampered.round2.stages[0].stage.0;
    // Compressed round messages reconstruct p(1) from the claim (RFC-0010).
    // A changed evaluation fails at the final public relation, at round n.
    let final_round = tampered.round2.stages[0].sumcheck.rounds.len() as u32;
    tampered.round2.stages[0].sumcheck.rounds[0][0] +=
        <tessera_field::GoldilocksExt2 as tessera_field::Field>::ONE;
    let verdict = tv_core::verify_non_interactive(&vc, &expected, &decode(&tampered), &vcfg);
    assert!(
        matches!(verdict, tv_core::Verdict::Reject(tv_core::RejectReason::Sumcheck {
        stage: rejected, round,
    }) if rejected == stage && round == final_round),
        "{name}/{requests:?}: {verdict:?}"
    );

    if name == "tiny-2-servable" && requests == [(3, 0)] {
        let mut root = proof.clone();
        root.round1.advice.root[0] ^= 1;
        let verdict = tv_core::verify_non_interactive(&vc, &expected, &decode(&root), &vcfg);
        assert!(
            matches!(
                verdict,
                tv_core::Verdict::Reject(tv_core::RejectReason::TreeLevel {
                    group: 0,
                    level: 1,
                    round: 1,
                })
            ),
            "{verdict:?}"
        );
    }

    let mut wrong_statement = proof.clone();
    wrong_statement.round1.statement.manifest_digest[0] ^= 1;
    let verdict = tv_core::verify_non_interactive(&vc, &expected, &decode(&wrong_statement), &vcfg);
    assert!(
        matches!(
            verdict,
            tv_core::Verdict::Reject(tv_core::RejectReason::StatementMismatch {
                field: tv_core::validation::StatementField::ManifestDigest,
            })
        ),
        "{verdict:?}"
    );

    // The actual Live FSM checks the same selected tables/windows and owned replay.
    let mut live_cfg = config.clone();
    live_cfg.mode = Mode::Live;
    let mut live = Prover::new(&plan, &model, &mut state, live_cfg).unwrap();
    let mut verifier = tv_core::LiveVerifier::new(vc, expected, vcfg);
    let commit = tessera_codec::decode(
        &verifier.start().unwrap().to_tce().unwrap(),
        DecodeLimits::MESSAGE,
    )
    .unwrap();
    let (first, share) = live.start(&batch, &commit).unwrap();
    let first = tv_codec::decode(
        &tessera_codec::encode(&first),
        &tv_codec::DecodeLimits::MESSAGE,
        tv_core::protocol::Round1::decode,
    )
    .unwrap();
    let share = tv_core::protocol::ProverSeedShare {
        session_id: share.session_id,
        share: share.share,
    };
    let reveal = tessera_codec::decode(
        &verifier
            .on_round1(&first, &share)
            .unwrap()
            .to_tce()
            .unwrap(),
        DecodeLimits::MESSAGE,
    )
    .unwrap();
    let second = live.on_seed(&reveal).unwrap();
    let second = tv_codec::decode(
        &tessera_codec::encode(&second),
        &tv_codec::DecodeLimits::MESSAGE,
        tv_core::protocol::Round2Response::decode,
    )
    .unwrap();
    let queries = tessera_codec::decode(
        &verifier.on_round2(&second).unwrap().to_tce().unwrap(),
        DecodeLimits::MESSAGE,
    )
    .unwrap();
    let third = live.on_queries(&queries).unwrap();
    let third = tv_codec::decode(
        &tessera_codec::encode(&third),
        &tv_codec::DecodeLimits::MESSAGE,
        tv_core::protocol::Round3Response::decode,
    )
    .unwrap();
    let verdict = verifier.on_round3(&third);
    assert!(
        matches!(verdict, tv_core::Verdict::Accept(_)),
        "{verdict:?}"
    );
    drop(live);

    // Locate the public admission boundary; its lower neighbor cannot start P1.
    let (mut low, mut high) = (0, planned.resident_bytes().unwrap());
    while high - low > 1 {
        let mid = low + (high - low) / 2;
        let mut cfg = config.clone();
        cfg.memory_budget.host = mid;
        match Prover::new(&plan, &model, &mut state, cfg).and_then(|p| p.plan(&batch)) {
            Ok(p) if p.resident_bytes().is_some_and(|bytes| bytes <= mid) => high = mid,
            Ok(_) | Err(ProverError::Resource(ErrorCode::ResMemory)) => low = mid,
            Err(e) => panic!("unexpected public planning error: {e}"),
        }
    }
    let mut limited = config;
    limited.memory_budget.host = high - 1;
    let result = Prover::new(&plan, &model, &mut state, limited)
        .and_then(|mut p| p.prove_non_interactive(&batch));
    assert!(matches!(
        result,
        Err(ProverError::Resource(ErrorCode::ResMemory))
    ));
    assert_eq!(state.public, *weight.public());
}
