// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Recorded public messages exercise replay without making a Live session reusable.
use super::*;

pub(super) fn fixture() -> (ClaimList, NiProof, VerifierConfig) {
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
