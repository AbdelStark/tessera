// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Recorded public messages exercise replay without making a Live session reusable.
use super::*;

pub(crate) fn fixture() -> (ClaimList, NiProof, VerifierConfig) {
    let limits = DecodeLimits::MESSAGE;
    let claims = ClaimList::open(
        include_bytes!("../../tests/fixtures/driver/claims.tcl"),
        &limits,
    )
    .unwrap();
    let weights = tv_codec::decode(
        include_bytes!("../../tests/fixtures/driver/weights.tce"),
        &limits,
        WeightCommitmentPublic::decode,
    )
    .unwrap();
    let cfg = VerifierConfig {
        params: ParamSet::TestSmall,
        limits: ResourceLimits::default(),
        accept_bound: FixedBits(0),
        weights,
    };
    let ni = NiProof::open(
        include_bytes!("../../tests/fixtures/driver/ni.tnp"),
        &limits,
    )
    .unwrap();
    (claims, ni, cfg)
}

#[test]
fn recorded_live_and_ni_reports_are_identical_on_100_complete_replays() {
    let (claims, ni, cfg) = fixture();
    let limits = DecodeLimits::MESSAGE;
    let live = NiProof::open(
        include_bytes!("../../tests/fixtures/driver/live-messages.tnp"),
        &limits,
    )
    .unwrap();
    let queries = tv_codec::decode(
        include_bytes!("../../tests/fixtures/driver/queries.tce"),
        &limits,
        protocol::ColumnQueries::decode,
    )
    .unwrap();
    let seed = include_bytes!("../../tests/fixtures/driver/live-seed.bin");
    // Pin the application input to the recorded fixture, never to a newly supplied proof.
    let expected = &live.round1.statement;
    assert_eq!(&ni.round1.statement, expected);
    let mut ni_report = None;
    let mut live_report = None;
    for _ in 0..100 {
        let result = verify_non_interactive(&claims, expected, &ni, &cfg);
        assert!(matches!(result, Verdict::Accept(_)), "{result:?}");
        let bytes = result.to_json().unwrap();
        assert_eq!(ni_report.get_or_insert_with(|| bytes.clone()), &bytes);
        let outcome = verify_unit(
            &claims,
            expected,
            Frame {
                round1: &live.round1,
                round2: &live.round2,
                round3: &live.round3,
            },
            &cfg,
            Session::Live {
                seed,
                queries: Some(&queries),
            },
        );
        let Ok(Outcome::Accepted(report)) = outcome else {
            panic!("recorded Live verification failed");
        };
        let bytes = Verdict::Accept(report).to_json().unwrap();
        assert_eq!(live_report.get_or_insert_with(|| bytes.clone()), &bytes);
    }
    let example = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../docs/perf/examples/verdict-test-small.json"
    );
    let json = ni_report.unwrap();
    if std::env::var_os("TESSERA_WRITE_EXAMPLES").is_some() {
        std::fs::write(example, &json).unwrap();
    }
    assert_eq!(std::fs::read_to_string(example).unwrap(), json);
}

#[test]
fn opening_row_scratch_is_budgeted_before_a_live_reveal() {
    for large_advice in [true, false] {
        let (mut claims, mut proof, mut cfg) = fixture();
        let frame_bytes = protocol::encoded_len(&proof.round1).unwrap()
            + protocol::encoded_len(&EMPTY_ROUND2).unwrap()
            + protocol::encoded_len(&EMPTY_ROUND3).unwrap();
        let allowance = (1u64 << (u32::from(claims.layout.advice.log2_k) + 1)) * 64;
        let coarse = frame_bytes * 8 + u64::from(claims.arch.n_layers) * (1 << 20) + allowance;
        cfg.limits.max_memory_bytes = coarse + 1;
        let share = |id| protocol::ProverSeedShare {
            session_id: id,
            share: [17; 32],
        };
        let mut control =
            LiveVerifier::new(claims.clone(), proof.round1.statement.clone(), cfg.clone());
        let commit = control.start().unwrap();
        proof.round1.session_id = commit.session_id;
        control
            .on_round1(&proof.round1, &share(commit.session_id))
            .unwrap();
        if large_advice {
            claims.arch.max_context = 4096;
            proof.round1.statement.batch.requests[0].prompt_len = 4096;
            proof.round1.statement.batch.requests[0].output_len = 0;
        } else {
            claims.arch.d_model *= 2;
            claims.arch.n_heads *= 2;
            claims.arch.vocab = 1 << 18;
            claims.quant.rsqrt_kappa -= 1;
        }
        // Regenerate the public table catalog for this model before claiming it is valid.
        let public = validation::check_public(&claims).unwrap();
        let maximal = crate::lowering::lower(&public, claims.arch.max_context).unwrap();
        let selected = crate::selection::derive(&maximal, claims.gadget_policy).unwrap();
        claims.tables = selected
            .tables
            .iter()
            .map(|t| {
                tv_tables::Table::generate(t.id, t.kind.clone())
                    .unwrap()
                    .spec()
                    .clone()
            })
            .collect();
        let validated = validation::validate(&claims).unwrap();
        proof.round1.statement.claim_list_digest = claims.digest();
        let expected = &proof.round1.statement;
        let request = &expected.batch.requests[0];
        let n = request.prompt_len + request.output_len;
        let graph = crate::lowering::lower(&validation::check_public(&claims).unwrap(), n).unwrap();
        let runtime = gadgets::runtime::Plan::derive(
            &graph,
            claims.gadget_policy,
            request.prompt_len,
            0,
            validated.tables(),
        )
        .unwrap();
        let layout = runtime
            .advice_layout(advice::Config::new(
                &claims.arch,
                claims.gadget_policy,
                claims.layout.advice.log2_k,
                claims.layout.advice.segments,
            ))
            .unwrap();
        let matrices = weights::derive(&claims, cfg.weights.epoch).unwrap();
        let a = claims.layout.advice;
        let advice = opening_scratch(
            layout.committed_rows().unwrap(),
            u32::from(a.log2_k),
            u32::from(a.log2_inv_rate),
        )
        .unwrap_or_else(|_| panic!("addressable advice scratch"));
        let weights = matrices
            .iter()
            .filter(|m| m.committed_rows() > 0)
            .map(|m| {
                opening_scratch(
                    m.committed_rows(),
                    m.log2_k(),
                    u32::from(claims.layout.weights.log2_inv_rate),
                )
                .unwrap_or_else(|_| panic!("addressable weight scratch"))
            })
            .max()
            .unwrap();
        assert_eq!(advice > weights, large_advice);
        let required = coarse + advice.max(weights).saturating_sub(allowance);
        assert!(required > cfg.limits.max_memory_bytes);
        // RFC-0031 descriptor from public geometry, never a supplied row count.
        let mut h = tv_hash::DomainHasher::new(tv_hash::Domain::CommitmentId);
        h.update(&0u16.to_le_bytes());
        h.update(&expected.digest());
        h.update(&0u32.to_le_bytes());
        h.update(&[a.log2_k, a.log2_inv_rate]);
        for x in [a.mask_coeffs, a.row_block, a.segments] {
            h.update(&x.to_le_bytes());
        }
        h.update(&layout.committed_rows().unwrap().to_le_bytes());
        proof.round1.advice.layout_digest = h.finalize();
        for (budget, accepts) in [(coarse + 1, false), (required - 1, false), (required, true)] {
            cfg.limits.max_memory_bytes = budget;
            let mut live =
                LiveVerifier::new(claims.clone(), proof.round1.statement.clone(), cfg.clone());
            let commit = live.start().unwrap();
            proof.round1.session_id = commit.session_id;
            let result = live.on_round1(&proof.round1, &share(commit.session_id));
            if accepts {
                result.unwrap();
            } else {
                assert!(matches!(
                    result.unwrap_err(),
                    Verdict::Inconclusive(VerifierError {
                        code: ErrorCode::Memory
                    })
                ));
                assert!(matches!(
                    live.on_round2(&proof.round2),
                    Err(Verdict::Inconclusive(VerifierError {
                        code: ErrorCode::ProtocolOrder
                    }))
                ));
            }
        }
    }
}

#[test]
fn opening_scratch_size_overflow_is_a_memory_failure() {
    assert!(matches!(opening_scratch(3, 3, 1), Ok(368)));
    for (rows, log_k, log_rate) in [
        (u64::MAX, 1, 1),
        (1 << 60, 1, 1),
        (u64::MAX / 16, 1, 0),
        (u64::MAX / 16 - 4, 2, 0),
        (0, 60, 0),
        (0, 63, 0),
        (0, 64, 0),
        (0, u32::MAX, 1),
    ] {
        assert!(matches!(
            opening_scratch(rows, log_k, log_rate),
            Err(Failure::Inconclusive(VerifierError {
                code: ErrorCode::Memory
            }))
        ));
    }
}

#[test]
fn pcs_cancellation_is_inconclusive_time_at_the_driver_boundary() {
    for failure in [
        opening_error(opening::OpeningError::Time),
        weight_error(weights::Error::Time),
    ] {
        assert!(matches!(
            Verdict::from(failure),
            Verdict::Inconclusive(VerifierError {
                code: ErrorCode::Time
            })
        ));
    }
}

#[test]
fn expired_work_cannot_release_an_accepted_report() {
    use crate::observe::CheckEvent;
    struct Delay(CheckKind);
    impl Observer for Delay {
        fn check_completed(&mut self, event: CheckEvent) {
            if event.kind == self.0 {
                std::thread::sleep(Duration::from_millis(550));
            }
        }
    }
    let (claims, proof, mut cfg) = fixture();
    cfg.limits.max_wall_time = Duration::from_millis(500);
    let expected = &proof.round1.statement;
    assert!(matches!(
        verify_non_interactive(&claims, expected, &proof, &cfg),
        Verdict::Accept(_)
    ));
    let bytes = include_bytes!("../../tests/fixtures/driver/ni.tnp");
    for kind in [CheckKind::ExactBound, CheckKind::Complete] {
        for verdict in [
            verify_non_interactive_observed(&claims, expected, &proof, &cfg, &mut Delay(kind)),
            verify_non_interactive_bytes_observed(&claims, expected, bytes, &cfg, &mut Delay(kind)),
        ] {
            assert!(matches!(
                verdict,
                Verdict::Inconclusive(VerifierError {
                    code: ErrorCode::Time
                })
            ));
        }
    }
}

#[test]
fn recorded_ni_preserves_exact_rejections_and_canonical_envelope_policy() {
    let (claims, good, cfg) = fixture();
    let expected = &good.round1.statement;
    let check =
        |proof: &NiProof, reason| match verify_non_interactive(&claims, expected, proof, &cfg) {
            Verdict::Reject(actual) => assert_eq!(actual, reason),
            other => panic!("expected exact rejection, got {other:?}"),
        };
    let mut bad = good.clone();
    bad.round2.logup.roots[1] = Fp2::ZERO;
    // Table 1 is the captured runtime's first used lookup; table 0 is unused.
    let table = tv_tables::TableId(1);
    check(&bad, RejectReason::Lookup { table });
    let mut bad = good.clone();
    bad.round2.stages.clear();
    check(
        &bad,
        RejectReason::Malformed {
            artifact: ArtifactKind(0x0520),
            code: ErrorCode::CodecNoncanonical,
        },
    );
    let mut bad = good.clone();
    bad.round2.advice_combined[0].u[0] += Fp2::ONE;
    check(
        &bad,
        RejectReason::Evaluation {
            commitment: CommitmentClass::Advice,
        },
    );
    let mut bad = good.clone();
    bad.round2.weight_combined[0].evaluations[0].u[0] += Fp2::ONE;
    check(
        &bad,
        RejectReason::Evaluation {
            commitment: CommitmentClass::Weight,
        },
    );
    let mut bad = good.clone();
    bad.round2.scoring.requests[0] = FixedBits(1);
    // Scoring is absorbed before query generation, so a changed claim invalidates columns first.
    assert!(!matches!(
        verify_non_interactive(&claims, expected, &bad, &cfg),
        Verdict::Accept(_)
    ));
    let mut bad = good.clone();
    bad.round3.advice_columns[0][0] += tv_field::Fp::ONE;
    assert!(matches!(
        verify_non_interactive(&claims, expected, &bad, &cfg),
        Verdict::Reject(RejectReason::MerklePath {
            commitment: CommitmentClass::Advice,
            ..
        })
    ));
    let bytes = include_bytes!("../../tests/fixtures/driver/ni.tnp");
    let mut limited = cfg.clone();
    limited.limits.max_artifact_bytes = bytes.len() as u64;
    assert!(matches!(
        verify_non_interactive_bytes(&claims, expected, bytes, &limited),
        Verdict::Accept(_)
    ));
    limited.limits.max_artifact_bytes = bytes.len() as u64 - 1;
    assert!(matches!(
        verify_non_interactive_bytes(&claims, expected, bytes, &limited),
        Verdict::Reject(RejectReason::Malformed {
            code: ErrorCode::CodecLimit,
            ..
        })
    ));
    limited = cfg.clone();
    limited.limits.max_memory_bytes = bytes.len() as u64;
    assert!(matches!(
        verify_non_interactive_bytes(&claims, expected, bytes, &limited),
        Verdict::Inconclusive(VerifierError {
            code: ErrorCode::Memory
        })
    ));
}

#[test]
fn both_pow_phases_reject_actually_invalid_twenty_bit_nonces() {
    let params = SoundnessParams::try_from_param_set(ParamSet::Ni100).unwrap();
    assert_eq!((params.seed_pow, params.query_pow), (20, 20));
    let context = SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::NonInteractive,
        params: tv_hash::transcript::ParamSetId(Ascii::new("ni-100/1").unwrap()),
        statement_digest: [7; 32],
        session_id: [9; 16],
    };
    for (phase, label) in [
        (PowPhase::Seed, "seed-pow"),
        (PowPhase::Queries, "query-pow"),
    ] {
        let invalid = (0..1024)
            .find(|&nonce| {
                Transcript::new(&context)
                    .check_grind(Ascii::new(label).unwrap(), 20, nonce)
                    .is_err()
            })
            .unwrap();
        assert!(
            matches!(proof_of_work(Transcript::new(&context), &params, phase, invalid),
            Err(Failure::Rejected(RejectReason::ProofOfWork { phase: actual })) if actual == phase)
        );
    }
}
