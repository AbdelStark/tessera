// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The verifier's live session (RFC-0036 'The verifier's live session'): every state and
//! accepted message, and every way it ends with its exact outcome. Each adversarial test breaks
//! one rule: the order of the session (`Inconclusive(ERR-PROTO-ORDER)`), its session id, the
//! protocol version (`Inconclusive(ERR-PROTO-VERSION)`), a check that fails (its verdict), and
//! a query shape no queries can have (`Inconclusive(ERR-INTERNAL)`).

mod common;

use common::{
    CheckCall, Checks, MATRICES, SplitMix64, Step, Verdict, advice, round2, round3, statement,
};
use tessera_codec::error::ErrorCode;
use tessera_protocol::{
    Ending, LeafGroups, MatrixCodeword, MessageKind, PROTOCOL_VERSION, ProtocolError,
    ProverSeedShare, QueryShape, Round1, Round2Response, Round3Response, SeedCommit,
    VerifierSession, VerifierState, coin_commitment, coin_toss,
};

/// What a prover sends a verifier's session, made for its `SeedCommit`.
struct Prover {
    round1: Round1,
    share: ProverSeedShare,
    round2: Round2Response,
    round3: Round3Response,
}

impl Prover {
    fn new(commit: &SeedCommit, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        Self {
            round1: Round1 {
                protocol_version: PROTOCOL_VERSION,
                session_id: commit.session_id,
                statement: statement(&mut rng),
                advice: advice(&mut rng),
                leaf_groups: LeafGroups::NONE,
            },
            share: ProverSeedShare {
                session_id: commit.session_id,
                share: rng.digest(),
            },
            round2: round2(&mut rng, &MATRICES),
            round3: round3(&mut rng, &MATRICES),
        }
    }
}

/// A session over `checks` driven to `state`, and its prover.
fn at(state: VerifierState, checks: Checks, seed: u64) -> (VerifierSession<Checks>, Prover) {
    let mut session = VerifierSession::new(checks);
    let commit = match state {
        VerifierState::Start => SeedCommit {
            session_id: [0; 16],
            commitment: [0; 32],
        },
        _ => session.start().expect("SeedCommit"),
    };
    let prover = Prover::new(&commit, seed);
    if matches!(
        state,
        VerifierState::AwaitRound2 | VerifierState::AwaitRound3 | VerifierState::Done
    ) {
        session
            .on_round1(&prover.round1, &prover.share)
            .expect("SeedReveal");
    }
    if matches!(state, VerifierState::AwaitRound3 | VerifierState::Done) {
        session.on_round2(&prover.round2).expect("ColumnQueries");
    }
    if state == VerifierState::Done {
        assert_eq!(
            session.on_round3(&prover.round3),
            Ending::Verdict(Verdict::Accept)
        );
    }
    (session, prover)
}

/// The session's own refusal, from any of its calls.
fn inconclusive<T: std::fmt::Debug>(outcome: Result<T, Ending<Verdict>>) -> ProtocolError {
    match outcome {
        Err(Ending::Inconclusive(error)) => error,
        other => panic!("not inconclusive: {other:?}"),
    }
}

#[test]
fn a_session_runs_through_every_state_to_its_verdict() {
    let mut session = VerifierSession::new(Checks::new());
    assert_eq!(session.state(), VerifierState::Start);
    let commit = session.start().expect("SeedCommit");
    assert_eq!(session.state(), VerifierState::AwaitRound1);
    assert!(session.checks().calls.is_empty());

    let prover = Prover::new(&commit, 0x98_1101);
    let reveal = session
        .on_round1(&prover.round1, &prover.share)
        .expect("SeedReveal");
    assert_eq!(session.state(), VerifierState::AwaitRound2);
    // The reveal names the session and opens the commitment made before the prover's share
    // was seen; steps 1 to 6 ran on Round1 with the seed H("coin-toss", s_V ‖ s_P).
    assert_eq!(reveal.session_id, commit.session_id);
    assert_eq!(coin_commitment(&reveal.share), commit.commitment);
    let seed = coin_toss(&reveal.share, &prover.share.share);
    assert_eq!(
        session.checks().calls,
        vec![CheckCall::Round1(Box::new(prover.round1.clone()), seed)]
    );

    let queries = session.on_round2(&prover.round2).expect("ColumnQueries");
    assert_eq!(session.state(), VerifierState::AwaitRound3);
    // Queries of the shape the checks derived, absorbed before they were sent.
    assert_eq!(session.checks().shape.check(&queries), Ok(()));
    assert_eq!(
        session.checks().calls[1..],
        [
            CheckCall::Round2(Box::new(prover.round2.clone())),
            CheckCall::Queries(queries),
        ]
    );

    assert_eq!(
        session.on_round3(&prover.round3),
        Ending::Verdict(Verdict::Accept)
    );
    assert_eq!(session.state(), VerifierState::Done);
    assert_eq!(
        session.checks().calls.last(),
        Some(&CheckCall::Round3(Box::new(prover.round3)))
    );
}

/// Every call a state does not expect ends the session `Inconclusive(ERR-PROTO-ORDER)`, naming
/// the message, before any check runs.
#[test]
fn every_message_a_state_does_not_expect_ends_the_session() {
    let calls = [
        MessageKind::SeedCommit,
        MessageKind::Round1,
        MessageKind::Round2Response,
        MessageKind::Round3Response,
    ];
    let states = [
        VerifierState::Start,
        VerifierState::AwaitRound1,
        VerifierState::AwaitRound2,
        VerifierState::AwaitRound3,
        VerifierState::Done,
    ];
    let mut refusals = 0;
    for (s, &state) in states.iter().enumerate() {
        for (c, &message) in calls.iter().enumerate() {
            // The one call each state but Done expects.
            if s == c {
                continue;
            }
            let (mut session, prover) = at(state, Checks::new(), 0x98_1201);
            let checked = session.checks().calls.len();
            let error = match message {
                MessageKind::SeedCommit => session.start().expect_err("refused"),
                MessageKind::Round1 => {
                    inconclusive(session.on_round1(&prover.round1, &prover.share))
                }
                MessageKind::Round2Response => inconclusive(session.on_round2(&prover.round2)),
                _ => match session.on_round3(&prover.round3) {
                    Ending::Inconclusive(error) => error,
                    other => panic!("{message} in {state:?}: {other:?}"),
                },
            };
            assert_eq!(
                error,
                ProtocolError::OutOfOrder { message },
                "{message} in {state:?}"
            );
            assert_eq!(error.code(), ErrorCode::ProtoOrder);
            assert_eq!(
                session.state(),
                VerifierState::Done,
                "{message} in {state:?}"
            );
            assert_eq!(
                session.checks().calls.len(),
                checked,
                "{message} in {state:?}"
            );
            refusals += 1;
        }
    }
    // Four states expect one call each; Done expects none.
    assert_eq!(refusals, 4 * 3 + 4);
}

/// A Round1 or a share that names another session ends the session `ERR-PROTO-ORDER`, Round1's
/// checked first, and the verifier's share is never revealed.
#[test]
fn a_round1_or_a_share_of_another_session_ends_the_session() {
    let other = |id: &mut [u8; 16]| id[7] ^= 0x10;
    for (why, round1_other, share_other, message) in [
        ("Round1", true, false, MessageKind::Round1),
        ("the share", false, true, MessageKind::ProverSeedShare),
        ("both", true, true, MessageKind::Round1),
    ] {
        let (mut session, mut prover) = at(VerifierState::AwaitRound1, Checks::new(), 0x98_1301);
        if round1_other {
            other(&mut prover.round1.session_id);
        }
        if share_other {
            other(&mut prover.share.session_id);
        }
        let error = inconclusive(session.on_round1(&prover.round1, &prover.share));
        assert_eq!(error, ProtocolError::OtherSession { message }, "{why}");
        assert_eq!(error.code(), ErrorCode::ProtoOrder, "{why}");
        assert_eq!(session.state(), VerifierState::Done, "{why}");
        assert!(session.checks().calls.is_empty(), "{why}");
    }
}

/// A Round1 of another protocol version than the claim list's, 0 until `tessera-protocol/1` is
/// assigned (09 §3), ends the session `ERR-PROTO-VERSION` before any check; version 0 reaches
/// the checks; a Round1 of another session is refused for that first.
#[test]
fn another_protocol_version_ends_the_session() {
    assert_eq!(PROTOCOL_VERSION, 0);
    let (mut session, prover) = at(VerifierState::AwaitRound1, Checks::new(), 0x98_1403);
    assert_eq!(prover.round1.protocol_version, 0);
    session
        .on_round1(&prover.round1, &prover.share)
        .expect("version 0 is the session's");
    assert_eq!(session.checks().calls.len(), 1);
    for version in [1, 2, u16::MAX] {
        let (mut session, mut prover) = at(VerifierState::AwaitRound1, Checks::new(), 0x98_1401);
        prover.round1.protocol_version = version;
        let error = inconclusive(session.on_round1(&prover.round1, &prover.share));
        assert_eq!(error, ProtocolError::Version { found: version });
        assert_eq!(error.code(), ErrorCode::ProtoVersion);
        assert_eq!(session.state(), VerifierState::Done);
        assert!(session.checks().calls.is_empty());
    }
    let (mut session, mut prover) = at(VerifierState::AwaitRound1, Checks::new(), 0x98_1402);
    prover.round1.protocol_version = 2;
    prover.round1.session_id[0] ^= 1;
    assert_eq!(
        inconclusive(session.on_round1(&prover.round1, &prover.share)),
        ProtocolError::OtherSession {
            message: MessageKind::Round1
        }
    );
}

/// The first check that fails ends the session with its verdict: after Round1 the share stays
/// hidden, after Round2Response no query is drawn, and Round3Response's verdict is the session's.
#[test]
fn a_failed_check_ends_the_session_with_its_verdict() {
    let rejecting = |step| Checks {
        reject: Some(step),
        ..Checks::new()
    };

    let (mut session, prover) = at(
        VerifierState::AwaitRound1,
        rejecting(Step::Round1),
        0x98_1501,
    );
    assert_eq!(
        session.on_round1(&prover.round1, &prover.share),
        Err(Ending::Verdict(Verdict::Reject(Step::Round1)))
    );
    assert_eq!(session.state(), VerifierState::Done);
    assert_eq!(
        inconclusive(session.on_round2(&prover.round2)),
        ProtocolError::OutOfOrder {
            message: MessageKind::Round2Response
        }
    );

    let (mut session, prover) = at(
        VerifierState::AwaitRound2,
        rejecting(Step::Round2),
        0x98_1502,
    );
    assert_eq!(
        session.on_round2(&prover.round2),
        Err(Ending::Verdict(Verdict::Reject(Step::Round2)))
    );
    assert_eq!(session.state(), VerifierState::Done);
    assert!(
        !session
            .checks()
            .calls
            .iter()
            .any(|call| matches!(call, CheckCall::Queries(_)))
    );

    let (mut session, prover) = at(
        VerifierState::AwaitRound3,
        rejecting(Step::Round3),
        0x98_1503,
    );
    assert_eq!(
        session.on_round3(&prover.round3),
        Ending::Verdict(Verdict::Reject(Step::Round3))
    );
    assert_eq!(session.state(), VerifierState::Done);
}

/// A shape no queries can have is the verifier's own bug: `Inconclusive(ERR-INTERNAL)`, and no
/// query is drawn or absorbed, nor memory reserved for a count of 2^32 - 1.
#[test]
fn a_shape_no_queries_can_have_ends_the_session_internal() {
    let mut empty = Checks::new();
    empty.shape.matrices[0].len = 0;
    let mut unordered = Checks::new();
    unordered.shape.matrices.reverse();
    let mut unknown = Checks::new();
    unknown
        .shape
        .matrices
        .push(MatrixCodeword { matrix: 10, len: 8 });
    let mut no_query = Checks::new();
    no_query.shape.matrix_count = 0;
    let mut too_many = Checks::new();
    too_many.shape.advice_count = u32::MAX;
    for (why, checks) in [
        ("an empty codeword", empty),
        ("matrices out of order", unordered),
        ("an unknown matrix", unknown),
        ("no query per matrix", no_query),
        ("2^32 - 1 advice queries", too_many),
    ] {
        let (mut session, prover) = at(VerifierState::AwaitRound2, checks, 0x98_1601);
        let error = inconclusive(session.on_round2(&prover.round2));
        assert!(
            matches!(error, ProtocolError::Internal { .. }),
            "{why}: {error:?}"
        );
        assert_eq!(error.code(), ErrorCode::Internal, "{why}");
        assert_eq!(session.state(), VerifierState::Done, "{why}");
        assert!(
            !session
                .checks()
                .calls
                .iter()
                .any(|call| matches!(call, CheckCall::Queries(_))),
            "{why}"
        );
    }
}

/// Every session draws its own session id and share (INV-PROTO-4).
#[test]
fn every_session_draws_a_fresh_session_and_share() {
    let mut first = VerifierSession::new(Checks::new());
    let mut second = VerifierSession::new(Checks::new());
    let (a, b) = (first.start().expect("a"), second.start().expect("b"));
    assert_ne!(a.session_id, b.session_id);
    assert_ne!(a.commitment, b.commitment);
}

/// The draw is uniform below each codeword's length and independent across draws: with 4,096
/// indices per codeword, every index appears, about `4096/len` times. Fewer than half that is
/// more than ten standard deviations away. A length that is not a power of two takes the
/// rejection branch.
#[test]
fn the_queries_are_uniform_below_every_codeword() {
    let shape = QueryShape {
        advice_count: 4096,
        advice_len: 8,
        matrix_count: 4096,
        matrices: vec![
            MatrixCodeword { matrix: 2, len: 2 },
            MatrixCodeword { matrix: 5, len: 5 },
            MatrixCodeword { matrix: 9, len: 1 },
        ],
    };
    let queries = shape.draw().expect("a draw");
    assert_eq!(shape.check(&queries), Ok(()));
    let codewords = [
        (&queries.advice, 8usize),
        (&queries.weights[0].indices, 2),
        (&queries.weights[1].indices, 5),
        (&queries.weights[2].indices, 1),
    ];
    for (indices, len) in codewords {
        let mut counts = vec![0usize; len];
        for &index in indices {
            counts[index as usize] += 1;
        }
        assert!(
            counts.iter().all(|&c| c >= 4096 / len / 2),
            "{len}: {counts:?}"
        );
    }
}
