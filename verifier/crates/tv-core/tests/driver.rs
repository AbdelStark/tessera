// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Trust boundaries and fixed first-error behavior before any honest producer integration.
mod claim_support;
use tv_codec::{WeightCommitmentPublic, WeightLayoutParams};
use tv_core::claims::{ClaimList, ParamSetId};
use tv_core::params::ParamSet;
use tv_core::protocol::*;
use tv_core::verdict::{ArtifactKind, VerifierError};
use tv_core::{
    ErrorCode, RejectReason as R, Verdict, VerifierConfig, verify_non_interactive as verify,
};
use tv_hash::{Domain, hash};
fn fixture() -> (ClaimList, PublicStatement, NiProof, VerifierConfig) {
    let golden = claim_support::goldens()
        .into_iter()
        .find(|g| g.model == "tiny-2-servable.d")
        .unwrap();
    let c = claim_support::twin(&golden.name);
    let p = c.layout.weights;
    let weights = WeightCommitmentPublic {
        layout: WeightLayoutParams {
            log2_k: p.log2_k,
            log2_inv_rate: p.log2_inv_rate,
            mask_coeffs: p.mask_coeffs,
            blinding_rows: p.blinding_rows,
        },
        roots: [[3; 32]; 10],
        epoch: 7,
    };
    let expected = PublicStatement {
        claim_list_digest: c.digest(),
        weights_root: hash(Domain::WeightsRoot, &weights.encode()),
        weights_epoch: 7,
        manifest_digest: [9; 32],
        batch: BatchDescriptor {
            requests: vec![RequestRef {
                request_id: 23,
                session_id: [4; 16],
                prompt_len: 1,
                output_len: 0,
            }],
        },
        predictor: PredictorSpec::Exact {
            decoding: Decoding::Greedy,
        },
    };
    let proof = NiProof {
        round1: Round1 {
            protocol_version: 0,
            session_id: [5; 16],
            statement: expected.clone(),
            advice: AdviceCommitment {
                root: [0; 32],
                layout_digest: [0; 32],
            },
            leaf_groups: LeafGroups::NONE,
        },
        seed_nonce: 0,
        round2: Round2Response {
            logup: LogUpProof {
                roots: vec![],
                values: vec![],
            },
            trees: TreeProof {
                levels: vec![],
                groups: vec![],
                table_roots: vec![],
                merge: tv_core::sumcheck::SumcheckProof {
                    rounds: vec![],
                    final_evals: vec![],
                },
                splits: vec![],
            },
            stages: vec![],
            advice_combined: vec![],
            weight_combined: vec![],
            scoring: ScoringClaims {
                requests: vec![FixedBits(0)],
            },
        },
        query_nonce: 0,
        round3: Round3Response {
            advice_columns: vec![],
            advice_proof: MultiProof { siblings: vec![] },
            weight_columns: vec![],
        },
    };
    let cfg = VerifierConfig {
        params: ParamSet::Ni100,
        limits: Default::default(),
        accept_bound: FixedBits(0),
        weights,
    };
    (c, expected, proof, cfg)
}
fn rejected(v: Verdict, expected: R) {
    match v {
        Verdict::Reject(r) => assert_eq!(r, expected),
        other => panic!("expected rejection, got {other:?}"),
    }
}
fn inconclusive(v: Verdict, code: ErrorCode) {
    match v {
        Verdict::Inconclusive(e) => assert_eq!(e, VerifierError { code }),
        other => panic!("expected inconclusive, got {other:?}"),
    }
}
#[test]
fn syntax_and_resource_failures_precede_parameter_acceptance() {
    let (mut c, s, mut p, mut cfg) = fixture();
    c.params = ParamSetId::new("unknown/1").unwrap();
    rejected(
        tv_core::verify_non_interactive_bytes(&c, &s, &[], &cfg),
        R::Malformed {
            artifact: ArtifactKind(0x0520),
            code: ErrorCode::CodecTruncated,
        },
    );
    let empty = tv_core::opening::CombinedRow {
        u: vec![],
        mask: vec![],
    };
    p.round2.weight_combined = vec![
        WeightCombinedRows {
            matrix: 2,
            evaluations: vec![],
            proximity: empty.clone(),
        },
        WeightCombinedRows {
            matrix: 1,
            evaluations: vec![],
            proximity: empty,
        },
    ];
    rejected(
        verify(&c, &s, &p, &cfg),
        R::Malformed {
            artifact: ArtifactKind(0x0520),
            code: ErrorCode::CodecNoncanonical,
        },
    );
    p.round2.weight_combined.truncate(1);
    p.round2.weight_combined[0].matrix = 10;
    rejected(
        verify(&c, &s, &p, &cfg),
        R::Malformed {
            artifact: ArtifactKind(0x0520),
            code: ErrorCode::CodecNoncanonical,
        },
    );
    p.round2.weight_combined.clear();
    cfg.limits.max_memory_bytes = 0;
    inconclusive(verify(&c, &s, &p, &cfg), ErrorCode::Memory);
    cfg.limits = Default::default();
    cfg.limits.max_wall_time = std::time::Duration::ZERO;
    inconclusive(verify(&c, &s, &p, &cfg), ErrorCode::Time);
}
#[test]
fn unknown_name_wrong_bundle_and_mode_reject_before_bad_claim_fields() {
    let (mut c, s, p, mut cfg) = fixture();
    c.arch.n_layers = 0;
    c.params = ParamSetId::new("unknown/1").unwrap();
    rejected(
        verify(&c, &s, &p, &cfg),
        R::Params {
            name: c.params.clone(),
        },
    );
    c.params = ParamSetId::new("ni-100/1").unwrap();
    cfg.params = ParamSet::Live64;
    rejected(
        verify(&c, &s, &p, &cfg),
        R::Params {
            name: c.params.clone(),
        },
    );
    c.params = ParamSetId::new("live-64/1").unwrap();
    rejected(
        verify(&c, &s, &p, &cfg),
        R::Params {
            name: c.params.clone(),
        },
    );
}
#[test]
fn statement_fields_reject_in_declaration_order_and_own_weight_disagreement_is_usage() {
    let (c, s, mut p, mut cfg) = fixture();
    p.round1.statement.claim_list_digest = [1; 32];
    p.round1.statement.weights_root = [2; 32];
    rejected(
        verify(&c, &s, &p, &cfg),
        R::StatementMismatch {
            field: tv_core::validation::StatementField::ClaimListDigest,
        },
    );
    p.round1.statement.claim_list_digest = s.claim_list_digest;
    rejected(
        verify(&c, &s, &p, &cfg),
        R::StatementMismatch {
            field: tv_core::validation::StatementField::WeightsRoot,
        },
    );
    cfg.weights.epoch += 1;
    inconclusive(verify(&c, &s, &p, &cfg), ErrorCode::Usage);
}
#[test]
fn layout_digest_and_batch_bounds_are_checked_before_seed_or_proof_rounds() {
    let (c, mut s, mut p, cfg) = fixture();
    rejected(
        verify(&c, &s, &p, &cfg),
        R::StatementMismatch {
            field: tv_core::validation::StatementField::LayoutDigest,
        },
    );
    s.batch.requests[0].output_len = c.arch.max_context;
    p.round1.statement = s.clone();
    rejected(
        verify(&c, &s, &p, &cfg),
        R::StatementMismatch {
            field: tv_core::validation::StatementField::Batch,
        },
    );
}

#[cfg(unix)]
fn live_fixture() -> (ClaimList, PublicStatement, NiProof, VerifierConfig) {
    let (mut c, mut s, mut p, mut cfg) = fixture();
    c.params = ParamSetId::new("live-64/1").unwrap();
    c.layout = ParamSet::Live64.layout();
    s.claim_list_digest = c.digest();
    p.round1.statement = s.clone();
    cfg.params = ParamSet::Live64;
    set_layout_digest(&c, &s, &mut p);
    (c, s, p, cfg)
}
fn set_layout_digest(c: &ClaimList, s: &PublicStatement, p: &mut NiProof) {
    let validated = tv_core::validation::validate(c).unwrap();
    let graph =
        tv_core::lowering::lower(&tv_core::validation::check_public(c).unwrap(), 1).unwrap();
    let runtime =
        tv_core::gadgets::runtime::Plan::derive(&graph, c.gadget_policy, 1, 0, validated.tables())
            .unwrap();
    let a = c.layout.advice;
    let layout = runtime
        .advice_layout(tv_core::advice::Config::new(
            &c.arch,
            c.gadget_policy,
            a.log2_k,
            a.segments,
        ))
        .unwrap();
    // Literal RFC-0031 descriptor, independently assembled without Commitment::bind.
    let mut h = tv_hash::DomainHasher::new(Domain::CommitmentId);
    h.update(&0u16.to_le_bytes());
    h.update(&s.digest());
    h.update(&0u32.to_le_bytes());
    h.update(&[a.log2_k, a.log2_inv_rate]);
    for n in [a.mask_coeffs, a.row_block, a.segments] {
        h.update(&n.to_le_bytes());
    }
    h.update(&layout.committed_rows().unwrap().to_le_bytes());
    p.round1.advice.layout_digest = h.finalize();
}

#[cfg(unix)]
#[test]
fn live_reveals_only_bound_round1_and_never_queries_invalid_round2() {
    let (c, s, mut p, cfg) = live_fixture();
    let mut live = tv_core::LiveVerifier::new(c, s, cfg);
    let commit = live.start().unwrap();
    assert!(matches!(
        live.start(),
        Err(Verdict::Inconclusive(VerifierError {
            code: ErrorCode::ProtocolOrder
        }))
    ));
    assert!(matches!(
        live.on_round2(&p.round2),
        Err(Verdict::Inconclusive(VerifierError {
            code: ErrorCode::ProtocolOrder
        }))
    ));
    p.round1.session_id = commit.session_id;
    let share = ProverSeedShare {
        session_id: commit.session_id,
        share: [17; 32],
    };
    let reveal = live.on_round1(&p.round1, &share).unwrap();
    assert_eq!(reveal.session_id, commit.session_id);
    assert_eq!(hash(Domain::CoinCommit, &reveal.share), commit.commitment);
    assert!(matches!(
        live.on_round1(&p.round1, &share),
        Err(Verdict::Inconclusive(VerifierError {
            code: ErrorCode::ProtocolOrder
        }))
    ));
    assert!(matches!(
        live.on_round3(&p.round3),
        Verdict::Inconclusive(VerifierError {
            code: ErrorCode::ProtocolOrder
        })
    ));
    rejected(
        live.on_round2(&p.round2).unwrap_err(),
        R::Malformed {
            artifact: ArtifactKind(0x0511),
            code: ErrorCode::CodecNoncanonical,
        },
    );
    assert!(matches!(
        live.on_round2(&p.round2),
        Err(Verdict::Inconclusive(VerifierError {
            code: ErrorCode::ProtocolOrder
        }))
    ));
}
#[cfg(unix)]
#[test]
fn live_invalid_round1_is_terminal_and_timeout_reveals_nothing() {
    let (c, s, mut p, cfg) = live_fixture();
    let mut live = tv_core::LiveVerifier::new(c.clone(), s.clone(), cfg.clone());
    let commit = live.start().unwrap();
    p.round1.session_id = commit.session_id;
    let share = ProverSeedShare {
        session_id: commit.session_id,
        share: [17; 32],
    };
    p.round1.advice.layout_digest[0] ^= 1;
    rejected(
        live.on_round1(&p.round1, &share).unwrap_err(),
        R::StatementMismatch {
            field: tv_core::validation::StatementField::LayoutDigest,
        },
    );
    assert!(matches!(
        live.on_round1(&p.round1, &share),
        Err(Verdict::Inconclusive(VerifierError {
            code: ErrorCode::ProtocolOrder
        }))
    ));
    let mut cfg = cfg;
    cfg.limits.max_wall_time = std::time::Duration::from_millis(1);
    let mut timed = tv_core::LiveVerifier::new(c, s, cfg);
    let commit = timed.start().unwrap();
    p.round1.session_id = commit.session_id;
    let share = ProverSeedShare {
        session_id: commit.session_id,
        share: [17; 32],
    };
    std::thread::sleep(std::time::Duration::from_millis(3));
    inconclusive(
        timed.on_round1(&p.round1, &share).unwrap_err(),
        ErrorCode::ProtocolTimeout,
    );
}

#[test]
fn live_start_failure_is_terminal_and_bad_session_ids_never_reveal_a_share() {
    let (c, s, _p, cfg) = fixture();
    let mut wrong_mode = tv_core::LiveVerifier::new(c, s, cfg);
    assert!(matches!(
        wrong_mode.start(),
        Err(Verdict::Reject(R::Params { .. }))
    ));
    assert!(matches!(
        wrong_mode.start(),
        Err(Verdict::Inconclusive(VerifierError {
            code: ErrorCode::ProtocolOrder
        }))
    ));
    #[cfg(unix)]
    {
        let (c, s, mut p, cfg) = live_fixture();
        let mut live = tv_core::LiveVerifier::new(c, s, cfg);
        let commit = live.start().unwrap();
        p.round1.session_id = commit.session_id;
        let mut share = ProverSeedShare {
            session_id: commit.session_id,
            share: [17; 32],
        };
        share.session_id[0] ^= 1;
        inconclusive(
            live.on_round1(&p.round1, &share).unwrap_err(),
            ErrorCode::ProtocolOrder,
        );
        share.session_id = commit.session_id;
        inconclusive(
            live.on_round1(&p.round1, &share).unwrap_err(),
            ErrorCode::ProtocolOrder,
        );
    }
}

#[test]
fn ni_invalid_seed_work_rejects_before_invalid_round2() {
    use tv_hash::transcript::{
        Ascii, Mode, ParamSetId as TranscriptParams, SessionContext, Transcript,
    };
    let (c, s, mut p, cfg) = fixture();
    set_layout_digest(&c, &s, &mut p);
    let context = SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::NonInteractive,
        params: TranscriptParams(Ascii::new("ni-100/1").unwrap()),
        statement_digest: s.digest(),
        session_id: p.round1.session_id,
    };
    // Independent hash/transcript conformance tests establish this 20-bit production predicate.
    // Select an actually invalid nonce; no probabilistic expectation enters this assertion.
    let invalid = (0..1024)
        .find(|nonce| {
            let mut transcript = Transcript::new(&context);
            transcript.absorb(Ascii::new("round1").unwrap(), &p.round1.to_tce().unwrap());
            transcript
                .check_grind(Ascii::new("seed-pow").unwrap(), 20, *nonce)
                .is_err()
        })
        .unwrap();
    p.seed_nonce = invalid;
    rejected(
        verify(&c, &s, &p, &cfg),
        R::ProofOfWork {
            phase: tv_core::verdict::PowPhase::Seed,
        },
    );
}

#[test]
fn each_statement_field_and_version_preserve_their_exact_first_error() {
    use tv_core::validation::StatementField as F;
    let (c, s, good, cfg) = fixture();
    for (i, field) in [
        F::ClaimListDigest,
        F::WeightsRoot,
        F::WeightsEpoch,
        F::ManifestDigest,
        F::Batch,
        F::Predictor,
    ]
    .into_iter()
    .enumerate()
    {
        let mut p = good.clone();
        match i {
            0 => p.round1.statement.claim_list_digest[0] ^= 1,
            1 => p.round1.statement.weights_root[0] ^= 1,
            2 => p.round1.statement.weights_epoch += 1,
            3 => p.round1.statement.manifest_digest[0] ^= 1,
            4 => p.round1.statement.batch.requests[0].request_id += 1,
            5 => {
                p.round1.statement.predictor = PredictorSpec::Mixture {
                    decoding: Decoding::Greedy,
                    eps_log2: 1,
                }
            }
            _ => unreachable!(),
        }
        rejected(verify(&c, &s, &p, &cfg), R::StatementMismatch { field });
    }
    let mut p = good;
    p.round1.protocol_version = 1;
    inconclusive(verify(&c, &s, &p, &cfg), ErrorCode::ProtocolVersion);
    let mut own = s;
    own.claim_list_digest[0] ^= 1;
    inconclusive(verify(&c, &own, &p, &cfg), ErrorCode::Usage);
}

// ----- Observation (05 §2 'The verifier's lines', §5 'The verifier's spans') ------------------

use tv_core::observe::{CheckEvent, CheckKind, CheckOutcome};

/// The outcome the last reported check carries for `verdict`.
fn outcome_of(verdict: &Verdict) -> CheckOutcome {
    match verdict {
        Verdict::Accept(_) => CheckOutcome::Pass,
        Verdict::Reject(_) => CheckOutcome::Reject,
        Verdict::Inconclusive(_) => CheckOutcome::Inconclusive,
    }
}
/// `events` are RFC-0019's steps 1 to `last` exactly, in order, each the check of its step;
/// step 5 is `not_applicable`, every other check before the last passed, and the last carries
/// `outcome`.
fn assert_steps(events: &[CheckEvent], last: u8, outcome: CheckOutcome) {
    let steps: Vec<u8> = events.iter().map(|e| e.kind.step()).collect();
    assert_eq!(steps, (1..=last).collect::<Vec<_>>(), "{events:?}");
    for (i, e) in events.iter().enumerate() {
        assert_eq!(e.kind, CheckKind::ALL[i]);
        let expected = if e.kind == CheckKind::WeightReservation {
            CheckOutcome::NotApplicable
        } else if i + 1 == events.len() {
            outcome
        } else {
            CheckOutcome::Pass
        };
        assert_eq!(e.outcome, expected, "{events:?}");
    }
}
/// The NI driver observed and unobserved: one verdict, byte for byte, and the observed checks.
fn observed(
    c: &ClaimList,
    s: &PublicStatement,
    p: &NiProof,
    cfg: &VerifierConfig,
) -> (Verdict, Vec<CheckEvent>) {
    let plain = verify(c, s, p, cfg);
    let mut events = Vec::new();
    let seen = tv_core::verify_non_interactive_observed(c, s, p, cfg, &mut events);
    assert_eq!(plain.to_json().unwrap(), seen.to_json().unwrap());
    let mut ignored = ();
    let quiet = tv_core::verify_non_interactive_observed(c, s, p, cfg, &mut ignored);
    assert_eq!(quiet.to_json().unwrap(), seen.to_json().unwrap());
    (seen, events)
}

#[test]
fn observed_ni_checks_end_at_the_check_that_decides() {
    use tv_core::validation::StatementField as F;
    let (c, s, good, cfg) = fixture();
    // (what is broken, the step that decides, the verdict).
    let mut cases: Vec<(
        &str,
        ClaimList,
        PublicStatement,
        NiProof,
        VerifierConfig,
        u8,
    )> = vec![];
    let mut p = good.clone();
    let empty = tv_core::opening::CombinedRow {
        u: vec![],
        mask: vec![],
    };
    p.round2.weight_combined = vec![
        WeightCombinedRows {
            matrix: 2,
            evaluations: vec![],
            proximity: empty.clone(),
        },
        WeightCombinedRows {
            matrix: 1,
            evaluations: vec![],
            proximity: empty,
        },
    ];
    cases.push(("matrix order", c.clone(), s.clone(), p, cfg.clone(), 1));
    let mut limited = cfg.clone();
    limited.limits.max_memory_bytes = 0;
    cases.push(("memory", c.clone(), s.clone(), good.clone(), limited, 1));
    let mut limited = cfg.clone();
    limited.limits.max_wall_time = std::time::Duration::ZERO;
    cases.push(("deadline", c.clone(), s.clone(), good.clone(), limited, 1));
    let mut unknown = c.clone();
    unknown.params = ParamSetId::new("unknown/1").unwrap();
    cases.push((
        "parameter set",
        unknown,
        s.clone(),
        good.clone(),
        cfg.clone(),
        2,
    ));
    let mut mode = cfg.clone();
    mode.params = ParamSet::Live64;
    cases.push(("mode", c.clone(), s.clone(), good.clone(), mode, 2));
    let mut invalid = c.clone();
    invalid.arch.n_layers = 0;
    cases.push((
        "claim list",
        invalid,
        s.clone(),
        good.clone(),
        cfg.clone(),
        3,
    ));
    let mut p = good.clone();
    p.round1.statement.claim_list_digest[0] ^= 1;
    cases.push(("statement field", c.clone(), s.clone(), p, cfg.clone(), 4));
    let mut own = s.clone();
    own.claim_list_digest[0] ^= 1;
    cases.push((
        "own statement",
        c.clone(),
        own,
        good.clone(),
        cfg.clone(),
        4,
    ));
    let mut p = good.clone();
    p.round1.protocol_version = 1;
    cases.push(("protocol version", c.clone(), s.clone(), p, cfg.clone(), 4));
    cases.push((
        "layout digest",
        c.clone(),
        s.clone(),
        good.clone(),
        cfg.clone(),
        4,
    ));
    let mut p = good.clone();
    set_layout_digest(&c, &s, &mut p);
    p.seed_nonce = (0..1024)
        .find(|&nonce| {
            let mut q = p.clone();
            q.seed_nonce = nonce;
            matches!(
                verify(&c, &s, &q, &cfg),
                Verdict::Reject(R::ProofOfWork { .. })
            )
        })
        .unwrap();
    cases.push((
        "seed proof of work",
        c.clone(),
        s.clone(),
        p,
        cfg.clone(),
        6,
    ));
    for (what, c, s, p, cfg, step) in &cases {
        let (verdict, events) = observed(c, s, p, cfg);
        assert!(!matches!(verdict, Verdict::Accept(_)), "{what}");
        assert_steps(&events, *step, outcome_of(&verdict));
    }
    // The verdicts themselves are the unobserved driver's, as the tests above pin them.
    let (verdict, _) = observed(&cases[6].1, &cases[6].2, &cases[6].3, &cases[6].4);
    rejected(
        verdict,
        R::StatementMismatch {
            field: F::ClaimListDigest,
        },
    );
}

#[test]
fn observed_ni_bytes_judge_the_envelope_in_step_one() {
    let (c, s, _, cfg) = fixture();
    for bytes in [&[][..], &[0u8; 64][..]] {
        let plain = tv_core::verify_non_interactive_bytes(&c, &s, bytes, &cfg);
        let mut events = Vec::new();
        let seen = tv_core::verify_non_interactive_bytes_observed(&c, &s, bytes, &cfg, &mut events);
        assert_eq!(plain.to_json().unwrap(), seen.to_json().unwrap());
        assert!(matches!(seen, Verdict::Reject(R::Malformed { .. })));
        assert_steps(&events, 1, CheckOutcome::Reject);
    }
}

#[test]
fn observed_live_messages_report_step_one_and_start_reports_step_two() {
    let vectors = tv_testkit::read_vectors("protocol/protocol.json");
    let round1 = vectors
        .get("messages")
        .and_then(|m| m.as_array())
        .unwrap()
        .iter()
        .map(|m| tv_testkit::hex(m.get("envelope").unwrap().as_str().unwrap()))
        .find(|e| e[6..8] == [0x10, 0x05])
        .unwrap();
    let limits = tv_codec::DecodeLimits::MESSAGE;
    let mut events = Vec::new();
    let opened = tv_core::open_live_observed::<Round1>(&round1, &limits, &mut events);
    assert!(opened.is_ok());
    assert_steps(&events, 1, CheckOutcome::Pass);
    let mut events = Vec::new();
    let refused = tv_core::open_live_observed::<Round1>(&round1[1..], &limits, &mut events);
    assert!(matches!(refused, Err(Verdict::Reject(R::Malformed { .. }))));
    assert_steps(&events, 1, CheckOutcome::Reject);
    assert_eq!(
        tv_core::open_live::<Round1>(&round1[1..], &limits)
            .unwrap_err()
            .to_json()
            .unwrap(),
        refused.unwrap_err().to_json().unwrap()
    );
    // The NI fixture's claim list is ni-100's: a live session refuses it at step 2.
    let (c, s, _p, cfg) = fixture();
    let mut live = tv_core::LiveVerifier::new(c, s, cfg);
    let mut events = Vec::new();
    assert!(matches!(
        live.start_observed(&mut events),
        Err(Verdict::Reject(R::Params { .. }))
    ));
    assert_eq!(
        events
            .iter()
            .map(|e| (e.kind, e.outcome))
            .collect::<Vec<_>>(),
        [(CheckKind::Params, CheckOutcome::Reject)]
    );
}

#[cfg(unix)]
#[test]
fn observed_live_rounds_run_their_checks_per_message() {
    let (c, s, mut p, cfg) = live_fixture();
    let mut live = tv_core::LiveVerifier::new(c, s, cfg);
    let mut events = Vec::new();
    let commit = live.start_observed(&mut events).unwrap();
    assert_eq!(
        events
            .iter()
            .map(|e| (e.kind, e.outcome))
            .collect::<Vec<_>>(),
        [(CheckKind::Params, CheckOutcome::Pass)]
    );
    // Out of order: the session's state is step 1's, and the call changes nothing.
    let mut events = Vec::new();
    inconclusive(
        live.on_round2_observed(&p.round2, &mut events).unwrap_err(),
        ErrorCode::ProtocolOrder,
    );
    assert_steps(&events, 1, CheckOutcome::Inconclusive);
    p.round1.session_id = commit.session_id;
    let share = ProverSeedShare {
        session_id: commit.session_id,
        share: [17; 32],
    };
    // Steps 1 to 6 complete before SeedReveal (RFC-0019).
    let mut events = Vec::new();
    live.on_round1_observed(&p.round1, &share, &mut events)
        .unwrap();
    assert_steps(&events, 6, CheckOutcome::Pass);
    let mut events = Vec::new();
    let verdict = live.on_round2_observed(&p.round2, &mut events).unwrap_err();
    rejected(
        verdict,
        R::Malformed {
            artifact: ArtifactKind(0x0511),
            code: ErrorCode::CodecNoncanonical,
        },
    );
    assert_steps(&events, 7, CheckOutcome::Reject);
}

#[cfg(unix)]
#[test]
fn observed_live_round1_rejection_reveals_nothing_and_ends_at_the_statement() {
    let (c, s, mut p, cfg) = live_fixture();
    let mut live = tv_core::LiveVerifier::new(c, s, cfg);
    let commit = live.start().unwrap();
    p.round1.session_id = commit.session_id;
    let share = ProverSeedShare {
        session_id: commit.session_id,
        share: [17; 32],
    };
    p.round1.advice.layout_digest[0] ^= 1;
    let mut events = Vec::new();
    rejected(
        live.on_round1_observed(&p.round1, &share, &mut events)
            .unwrap_err(),
        R::StatementMismatch {
            field: tv_core::validation::StatementField::LayoutDigest,
        },
    );
    assert_steps(&events, 4, CheckOutcome::Reject);
}

/// `TEST_PARAMS` says what the build accepts: test-small exactly when the feature is on
/// (RFC-0001 'Cargo features'), which a run manifest names (05 §6, OBS-5).
#[test]
fn test_params_names_whether_this_build_accepts_test_small() {
    let accepted = ParamSet::accept(&ParamSetId::new("test-small/1").unwrap()).is_some();
    assert_eq!(tv_core::TEST_PARAMS, accepted);
}
