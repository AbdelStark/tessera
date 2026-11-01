// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The prover's live session (RFC-0036 'The prover's live session'): every state and accepted
//! message, and every refusal of the table with its exact code and the state it leaves. Each
//! adversarial test breaks one rule: the order of the session (`ERR-PROTO-ORDER`), its session
//! id, the coin toss (`ERR-PROTO-COINTOSS`, then abort), one seed per `Round1` (INV-PROTO-1,
//! `ERR-PROTO-REUSE`), the queries' matrices, counts and ranges (INV-PROTO-3,
//! `ERR-PROTO-QUERIES`), and a session given up before its end, dropped or with its engine
//! taken back, which must discard what it committed to (INV-PROTO-1). The share is the
//! engine's: a resumed prover sends the share it sent (RFC-0018 'Crash consistency').

mod common;

use std::fmt::Debug;

use common::{Engine, EngineCall, EngineError, MATRICES, SplitMix64, Step, queries, shape};
use tessera_codec::error::ErrorCode;
use tessera_protocol::{
    ColumnQueries, MatrixQueries, Message, MessageKind, PROTOCOL_VERSION, ProtocolError,
    ProverEngine, ProverSession, ProverState, QueryFault, SeedCommit, SeedReveal, SessionError,
    coin_commitment, coin_toss,
};

/// The verifier's side of a coin toss: its share and its commitment, for one session.
struct Coin {
    share: [u8; 32],
    commit: SeedCommit,
}

impl Coin {
    fn new(seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let share = rng.digest();
        Self {
            share,
            commit: SeedCommit {
                session_id: rng.session(),
                commitment: coin_commitment(&share),
            },
        }
    }

    fn reveal(&self) -> SeedReveal {
        SeedReveal {
            session_id: self.commit.session_id,
            share: self.share,
        }
    }
}

/// The refusal `outcome` must be.
fn refusal<T: Debug>(outcome: Result<T, SessionError<EngineError>>) -> ProtocolError {
    match outcome {
        Err(SessionError::Protocol(error)) => error,
        other => panic!("not a refusal: {other:?}"),
    }
}

/// A session driven to `state`, over an engine and a coin from `seed`.
fn at(state: ProverState, seed: u64) -> (ProverSession<Engine>, Coin) {
    let coin = Coin::new(seed);
    let mut session = ProverSession::new(Engine::new(seed ^ 0xE));
    drive(&mut session, state, &coin);
    (session, coin)
}

/// Drives `session` to `state` with `coin`, over an engine whose shape is the sample one.
fn drive<E: ProverEngine<Error = EngineError>>(
    session: &mut ProverSession<E>,
    state: ProverState,
    coin: &Coin,
) {
    if state != ProverState::AwaitCommit {
        session.start(&coin.commit).expect("Round1");
    }
    if matches!(state, ProverState::AwaitQueries | ProverState::Done) {
        session.on_seed(&coin.reveal()).expect("Round2Response");
    }
    if state == ProverState::Done {
        let queries = queries(&shape(&MATRICES));
        session.on_queries(&queries).expect("Round3Response");
    }
}

fn proves(call: &EngineCall) -> bool {
    matches!(call, EngineCall::Prove(..))
}

fn answers(call: &EngineCall) -> bool {
    matches!(call, EngineCall::Answer(..))
}

fn discards(call: &EngineCall) -> bool {
    matches!(call, EngineCall::Discard)
}

#[test]
fn a_session_runs_through_every_state_to_its_end() {
    let coin = Coin::new(0x98_0101);
    let mut session = ProverSession::new(Engine::new(0x98_0102));
    assert_eq!(session.state(), ProverState::AwaitCommit);

    let (round1, share) = session.start(&coin.commit).expect("Round1");
    assert_eq!(session.state(), ProverState::AwaitReveal);
    // Round1 speaks the claim list's protocol version, 0 until tessera-protocol/1 is assigned,
    // and names the commit's session; so does the share, which is the engine's.
    assert_eq!(round1.protocol_version, PROTOCOL_VERSION);
    assert_eq!(PROTOCOL_VERSION, 0);
    assert_eq!(round1.session_id, coin.commit.session_id);
    assert_eq!(share.session_id, coin.commit.session_id);
    assert_eq!(share.share, session.engine().share());
    assert_eq!(round1.statement, session.engine().statement);
    assert_eq!(round1.advice, session.engine().advice);

    let round2 = session.on_seed(&coin.reveal()).expect("Round2Response");
    assert_eq!(session.state(), ProverState::AwaitQueries);
    assert_eq!(round2, session.engine().round2);

    let queries = queries(&session.engine().shape);
    let round3 = session.on_queries(&queries).expect("Round3Response");
    assert_eq!(session.state(), ProverState::Done);
    assert!(!session.is_aborted());
    assert_eq!(round3, session.engine().round3);

    // The engine committed for the commit's session, gave the share, proved from
    // H("coin-toss", s_V ‖ s_P) on the Round1 the session sent, answered the queries as they
    // came, and discarded nothing.
    let seed = coin_toss(&coin.share, &share.share);
    assert_eq!(
        session.engine().calls,
        vec![
            EngineCall::Commit(coin.commit.session_id),
            EngineCall::Share,
            EngineCall::Prove(Box::new(round1), seed),
            EngineCall::Answer(queries),
        ]
    );
}

/// Every message a state does not expect: `ERR-PROTO-ORDER`, the state and the engine as they
/// were. In `AwaitQueries` a reveal is `ERR-PROTO-REUSE` instead, the table's own row.
#[test]
fn every_message_a_state_does_not_expect_is_refused_and_leaves_it() {
    let order = |message| ProtocolError::OutOfOrder { message };
    let finished = || at(ProverState::Done, 0x98_0201);
    let aborted = || {
        let (mut session, coin) = at(ProverState::AwaitQueries, 0x98_0202);
        session.abort();
        (session, coin)
    };
    let refused_reveal = || {
        let (mut session, coin) = at(ProverState::AwaitReveal, 0x98_0203);
        let mut wrong = coin.reveal();
        wrong.share[0] ^= 1;
        assert!(session.on_seed(&wrong).is_err());
        (session, coin)
    };
    type Make = Box<dyn Fn() -> (ProverSession<Engine>, Coin)>;
    let states: Vec<(&str, Make)> = vec![
        (
            "AwaitCommit",
            Box::new(|| at(ProverState::AwaitCommit, 0x98_0204)),
        ),
        (
            "AwaitReveal",
            Box::new(|| at(ProverState::AwaitReveal, 0x98_0205)),
        ),
        (
            "AwaitQueries",
            Box::new(|| at(ProverState::AwaitQueries, 0x98_0206)),
        ),
        ("Done", Box::new(finished)),
        ("Done after an abort", Box::new(aborted)),
        ("Done after a failed reveal", Box::new(refused_reveal)),
    ];
    let mut refusals = 0;
    for (name, make) in &states {
        let (probe, _) = make();
        let state = probe.state();
        let expected = |message: MessageKind| -> Option<ProtocolError> {
            match (state, message) {
                (ProverState::AwaitCommit, MessageKind::SeedCommit)
                | (ProverState::AwaitReveal, MessageKind::SeedReveal)
                | (ProverState::AwaitQueries, MessageKind::ColumnQueries) => None,
                (ProverState::AwaitQueries, MessageKind::SeedReveal) => Some(ProtocolError::Reuse),
                _ => Some(order(message)),
            }
        };
        for message in [
            MessageKind::SeedCommit,
            MessageKind::SeedReveal,
            MessageKind::ColumnQueries,
        ] {
            let Some(error) = expected(message) else {
                continue;
            };
            let (mut session, coin) = make();
            let calls = session.engine().calls.clone();
            let was_aborted = session.is_aborted();
            let queries = queries(&shape(&MATRICES));
            let found = match message {
                MessageKind::SeedCommit => refusal(session.start(&coin.commit)),
                MessageKind::SeedReveal => refusal(session.on_seed(&coin.reveal())),
                _ => refusal(session.on_queries(&queries)),
            };
            assert_eq!(found, error, "{message} in {name}");
            let code = match error {
                ProtocolError::Reuse => ErrorCode::ProtoReuse,
                _ => ErrorCode::ProtoOrder,
            };
            assert_eq!(found.code(), code, "{message} in {name}");
            assert_eq!(session.state(), state, "{message} in {name}");
            assert_eq!(session.is_aborted(), was_aborted, "{message} in {name}");
            assert_eq!(session.engine().calls, calls, "{message} in {name}");
            refusals += 1;
        }
    }
    // 2 in AwaitCommit, 2 in AwaitReveal, 2 in AwaitQueries, 3 in each Done.
    assert_eq!(refusals, 15);
}

/// The reveal's session is checked first (`ERR-PROTO-ORDER`), and its refusal is not an abort:
/// the session's own reveal is still answered.
#[test]
fn a_reveal_naming_another_session_is_refused_and_leaves_the_state() {
    let (mut session, coin) = at(ProverState::AwaitReveal, 0x98_0301);
    let mut other = coin.reveal();
    other.session_id[15] ^= 0x40;
    let mut other_and_wrong = other.clone();
    other_and_wrong.share[3] ^= 1;
    for reveal in [other, other_and_wrong] {
        let error = refusal(session.on_seed(&reveal));
        assert_eq!(
            error,
            ProtocolError::OtherSession {
                message: MessageKind::SeedReveal
            }
        );
        assert_eq!(error.code(), ErrorCode::ProtoOrder);
        assert_eq!(session.state(), ProverState::AwaitReveal);
        assert!(!session.is_aborted());
        assert_eq!(session.engine().count(proves), 0);
        assert_eq!(session.engine().count(discards), 0);
    }
    session.on_seed(&coin.reveal()).expect("Round2Response");
    assert_eq!(session.state(), ProverState::AwaitQueries);
}

/// A share that does not open `c_V` is `ERR-PROTO-COINTOSS` and an abort: the Round1 and its
/// randomness are discarded, nothing is proven, and the right share is not tried after it.
#[test]
fn a_reveal_that_does_not_open_the_commitment_aborts_the_session() {
    for bit in [0usize, 77, 255] {
        let (mut session, coin) = at(ProverState::AwaitReveal, 0x98_0401 + bit as u64);
        let mut wrong = coin.reveal();
        wrong.share[bit / 8] ^= 1 << (bit % 8);
        let error = refusal(session.on_seed(&wrong));
        assert_eq!(error, ProtocolError::CoinToss);
        assert_eq!(error.code(), ErrorCode::ProtoCointoss);
        assert_eq!(session.state(), ProverState::Done);
        assert!(session.is_aborted());
        assert_eq!(session.engine().calls.last(), Some(&EngineCall::Discard));
        assert_eq!(session.engine().count(proves), 0);

        // A verifier cannot try a second s_V against this Round1.
        let error = refusal(session.on_seed(&coin.reveal()));
        assert_eq!(
            error,
            ProtocolError::OutOfOrder {
                message: MessageKind::SeedReveal
            }
        );
        assert_eq!(session.engine().count(proves), 0);
        assert_eq!(session.engine().count(discards), 1);
    }
}

/// Once the seed of a Round1 is answered, any reveal is `ERR-PROTO-REUSE` (INV-PROTO-1), the
/// session's own included, and the session still waits for its queries.
#[test]
fn a_second_reveal_for_one_round1_is_reuse() {
    let (mut session, coin) = at(ProverState::AwaitQueries, 0x98_0501);
    let fresh = [7u8; 32];
    let mut other_share = coin.reveal();
    other_share.share = fresh;
    let mut other_session = coin.reveal();
    other_session.session_id[0] ^= 1;
    for reveal in [coin.reveal(), other_share, other_session] {
        let error = refusal(session.on_seed(&reveal));
        assert_eq!(error, ProtocolError::Reuse);
        assert_eq!(error.code(), ErrorCode::ProtoReuse);
        assert_eq!(session.state(), ProverState::AwaitQueries);
        assert!(!session.is_aborted());
        assert_eq!(session.engine().count(proves), 1);
    }
    let queries = queries(&session.engine().shape);
    session.on_queries(&queries).expect("Round3Response");
}

/// Each fault INV-PROTO-3 names, alone: `ERR-PROTO-QUERIES` with the fault, the session still
/// waiting and nothing answered; then the right queries are answered.
#[test]
fn queries_inv_proto_3_refuses_leave_the_session_waiting() {
    let (mut session, _) = at(ProverState::AwaitQueries, 0x98_0601);
    let good = queries(&session.engine().shape);
    let with = |change: &dyn Fn(&mut ColumnQueries)| {
        let mut queries = good.clone();
        change(&mut queries);
        queries
    };
    let count = |matrix, expected, found| QueryFault::Count {
        matrix,
        expected,
        found,
    };
    let range = |matrix, position, len| QueryFault::Range {
        matrix,
        position,
        len,
    };
    let cases: Vec<(&str, ColumnQueries, QueryFault)> = vec![
        (
            "a matrix missing",
            with(&|q| {
                q.weights.pop();
            }),
            QueryFault::Matrices,
        ),
        (
            "a matrix more",
            with(&|q| {
                q.weights.push(MatrixQueries {
                    matrix: 9,
                    indices: vec![0, 0, 0],
                })
            }),
            QueryFault::Matrices,
        ),
        (
            "another matrix",
            with(&|q| q.weights[1].matrix = 7),
            QueryFault::Matrices,
        ),
        (
            "the matrices swapped",
            with(&|q| q.weights.swap(0, 1)),
            QueryFault::Matrices,
        ),
        (
            "no matrix",
            with(&|q| q.weights.clear()),
            QueryFault::Matrices,
        ),
        (
            "an advice index short",
            with(&|q| {
                q.advice.pop();
            }),
            count(None, 4, 3),
        ),
        (
            "an advice index more",
            with(&|q| q.advice.push(0)),
            count(None, 4, 5),
        ),
        (
            "no advice index",
            with(&|q| q.advice.clear()),
            count(None, 4, 0),
        ),
        (
            "a Q index short",
            with(&|q| {
                q.weights[0].indices.pop();
            }),
            count(Some(1), 3, 2),
        ),
        (
            "an LmHead index more",
            with(&|q| q.weights[1].indices.push(1)),
            count(Some(8), 3, 4),
        ),
        (
            "an advice index at n_A",
            with(&|q| q.advice[2] = 16),
            range(None, 2, 16),
        ),
        (
            "an advice index at u32::MAX",
            with(&|q| q.advice[0] = u32::MAX),
            range(None, 0, 16),
        ),
        (
            "a Q index at n_T",
            with(&|q| q.weights[0].indices[1] = 8),
            range(Some(1), 1, 8),
        ),
        (
            "an LmHead index past n_T",
            with(&|q| q.weights[1].indices[2] = 1000),
            range(Some(8), 2, 8),
        ),
        // The order of the checks: the matrices, the counts, then the ranges.
        (
            "other matrices and a count short",
            with(&|q| {
                q.weights[0].matrix = 2;
                q.advice.pop();
            }),
            QueryFault::Matrices,
        ),
        (
            "a count short and an index out of range",
            with(&|q| {
                q.advice[0] = 99;
                q.weights[1].indices.pop();
            }),
            count(Some(8), 3, 2),
        ),
        (
            "two indices out of range",
            with(&|q| {
                q.weights[0].indices[0] = 9;
                q.advice[3] = 17;
            }),
            range(None, 3, 16),
        ),
    ];
    for (why, queries, fault) in cases {
        let error = refusal(session.on_queries(&queries));
        assert_eq!(error, ProtocolError::Queries(fault), "{why}");
        assert_eq!(error.code(), ErrorCode::ProtoQueries, "{why}");
        assert_eq!(session.state(), ProverState::AwaitQueries, "{why}");
        assert!(!session.is_aborted(), "{why}");
        assert_eq!(session.engine().count(answers), 0, "{why}");
    }
    session.on_queries(&good).expect("Round3Response");
    assert_eq!(session.state(), ProverState::Done);
}

/// Queries are drawn with replacement, so an index may repeat (INV-PROTO-3): the engine gets
/// them as they came.
#[test]
fn repeated_indices_are_answered() {
    let (mut session, _) = at(ProverState::AwaitQueries, 0x98_0701);
    let mut repeated = queries(&session.engine().shape);
    repeated.advice = vec![5; 4];
    for matrix in &mut repeated.weights {
        matrix.indices = vec![7; 3];
    }
    session.on_queries(&repeated).expect("Round3Response");
    assert_eq!(
        session.engine().calls.last(),
        Some(&EngineCall::Answer(repeated))
    );
}

/// An abort in any state before the end discards the session's Round1 and randomness once,
/// and nothing is answered after it (INV-PROTO-1). A finished session is not aborted later.
#[test]
fn an_abort_discards_the_round1_and_ends_the_session() {
    for state in [
        ProverState::AwaitCommit,
        ProverState::AwaitReveal,
        ProverState::AwaitQueries,
    ] {
        let (mut session, coin) = at(state, 0x98_0801);
        let proven = session.engine().count(proves);
        session.abort();
        assert_eq!(session.state(), ProverState::Done, "{state:?}");
        assert!(session.is_aborted(), "{state:?}");
        assert_eq!(session.engine().count(discards), 1, "{state:?}");
        let order = |message| ProtocolError::OutOfOrder { message };
        assert_eq!(
            refusal(session.start(&coin.commit)),
            order(MessageKind::SeedCommit)
        );
        assert_eq!(
            refusal(session.on_seed(&coin.reveal())),
            order(MessageKind::SeedReveal)
        );
        let queries = queries(&shape(&MATRICES));
        assert_eq!(
            refusal(session.on_queries(&queries)),
            order(MessageKind::ColumnQueries)
        );
        session.abort();
        assert_eq!(session.engine().count(discards), 1, "{state:?}");
        assert_eq!(session.engine().count(proves), proven, "{state:?}");
        assert_eq!(session.engine().count(answers), 0, "{state:?}");
    }
    let (mut session, _) = at(ProverState::Done, 0x98_0802);
    session.abort();
    assert!(!session.is_aborted());
    assert_eq!(session.engine().count(discards), 0);
}

/// An engine that fails at any step aborts the session: its error comes back, the session is
/// done and the engine discards what it committed.
#[test]
fn an_engine_failure_aborts_the_session() {
    for step in [Step::Commit, Step::Share, Step::Prove, Step::Answer] {
        let coin = Coin::new(0x98_0901);
        let mut engine = Engine::new(0x98_0902);
        engine.fail = Some(step);
        let mut session = ProverSession::new(engine);
        let outcome = match step {
            Step::Commit | Step::Share => session.start(&coin.commit).map(drop),
            Step::Prove => {
                session.start(&coin.commit).expect("Round1");
                session.on_seed(&coin.reveal()).map(drop)
            }
            _ => {
                session.start(&coin.commit).expect("Round1");
                session.on_seed(&coin.reveal()).expect("Round2Response");
                let queries = queries(&session.engine().shape);
                session.on_queries(&queries).map(drop)
            }
        };
        assert_eq!(outcome, Err(SessionError::Engine(EngineError(step))));
        assert_eq!(session.state(), ProverState::Done, "{step:?}");
        assert!(session.is_aborted(), "{step:?}");
        assert_eq!(
            session.engine().calls.last(),
            Some(&EngineCall::Discard),
            "{step:?}"
        );
    }
}

/// A query shape whose matrices are not `weight_combined`'s, whose codewords no query can
/// index, or whose counts are 0 or past 2^16, is the engine's bug: `ERR-INTERNAL`, an abort,
/// and no `Round2Response`.
#[test]
fn a_query_shape_off_weight_combined_is_internal_and_aborts() {
    let mut empty = shape(&MATRICES);
    empty.advice_len = 0;
    let mut wide = shape(&MATRICES);
    wide.matrices[1].len = (1 << 32) + 1;
    let mut no_advice_query = shape(&MATRICES);
    no_advice_query.advice_count = 0;
    let mut no_matrix_query = shape(&MATRICES);
    no_matrix_query.matrix_count = 0;
    let mut too_many = shape(&MATRICES);
    too_many.advice_count = (1 << 16) + 1;
    for (why, shape) in [
        ("a matrix missing", shape(&[1])),
        ("a matrix more", shape(&[1, 8, 9])),
        ("another matrix", shape(&[1, 7])),
        ("a codeword of no symbol", empty),
        ("a codeword past 2^32", wide),
        ("no advice query", no_advice_query),
        ("no query per matrix", no_matrix_query),
        ("2^16 + 1 advice queries", too_many),
    ] {
        let coin = Coin::new(0x98_0A01);
        let mut engine = Engine::new(0x98_0A02);
        engine.shape = shape;
        let mut session = ProverSession::new(engine);
        session.start(&coin.commit).expect("Round1");
        let error = refusal(session.on_seed(&coin.reveal()));
        assert!(
            matches!(error, ProtocolError::Internal { .. }),
            "{why}: {error:?}"
        );
        assert_eq!(error.code(), ErrorCode::Internal, "{why}");
        assert_eq!(session.state(), ProverState::Done, "{why}");
        assert!(session.is_aborted(), "{why}");
        assert_eq!(session.engine().calls.last(), Some(&EngineCall::Discard));
    }
}

/// A retry after an abort is a new session, which commits again and sends a fresh share: the
/// abort made the engine discard its per-proof seed (INV-PROTO-1). Freshness is the engine's;
/// the session sends the share the engine gives.
#[test]
fn every_round1_has_a_fresh_share_and_a_retry_is_a_new_session() {
    let coin = Coin::new(0x98_0B01);
    let mut session = ProverSession::new(Engine::new(0x98_0B02));
    let (_, first) = session.start(&coin.commit).expect("Round1");
    session.abort();
    let mut retry = ProverSession::new(session.into_engine());
    let (_, second) = retry.start(&coin.commit).expect("Round1");
    assert_ne!(first.share, second.share);
    assert_eq!(second.share, retry.engine().share());
    assert_eq!(
        retry
            .engine()
            .count(|call| matches!(call, EngineCall::Commit(_))),
        2
    );
    assert_eq!(retry.engine().count(discards), 1);
}

/// The session sends exactly the share the engine gives, and draws none of its own: engines of
/// different per-proof seeds give different shares, and an engine whose seed is set gives that
/// seed's share.
#[test]
fn the_session_sends_the_share_the_engine_gives() {
    let coin = Coin::new(0x98_0C01);
    let mut shares = Vec::new();
    for per_proof_seed in [1, 2, 0xDEAD_BEEF] {
        let mut engine = Engine::new(0x98_0C02);
        engine.per_proof_seed = per_proof_seed;
        let expected = engine.share();
        let mut session = ProverSession::new(engine);
        let (_, share) = session.start(&coin.commit).expect("Round1");
        assert_eq!(share.share, expected);
        assert_eq!(
            session
                .engine()
                .count(|call| matches!(call, EngineCall::Share)),
            1
        );
        shares.push(share.share);
    }
    assert_ne!(shares[0], shares[1]);
    assert_ne!(shares[1], shares[2]);
    assert_ne!(shares[0], shares[2]);
}

/// A prover that crashes after sending `Round1` and its share, and resumes from the same
/// engine state, the per-proof seed it persisted, sends the same two messages byte for byte
/// (RFC-0018 'Crash consistency'): never a second share with one `Round1`, so never a second
/// seed. A crash runs no drop, so nothing is discarded.
#[test]
fn a_resumed_prover_sends_the_same_round1_and_share() {
    let coin = Coin::new(0x98_0D01);
    let mut before = ProverSession::new(Engine::new(0x98_0D02));
    let (round1, share) = before.start(&coin.commit).expect("Round1");
    // The crash: the process dies, and no drop runs.
    std::mem::forget(before);
    let mut resumed = ProverSession::new(Engine::new(0x98_0D02));
    let (again, share_again) = resumed.start(&coin.commit).expect("Round1");
    assert_eq!(again.seal(), round1.seal());
    assert_eq!(share_again.seal(), share.seal());
    assert_eq!(resumed.engine().count(discards), 0);
}

/// A session dropped before its end, a timeout or a transport error giving it up, aborts: its
/// engine discards the Round1 and its randomness exactly once (INV-PROTO-1). A session dropped
/// once done discards nothing more.
#[test]
fn a_session_dropped_before_its_end_discards_once() {
    for (state, abort_first, discarded) in [
        (ProverState::AwaitCommit, false, 1),
        (ProverState::AwaitReveal, false, 1),
        (ProverState::AwaitQueries, false, 1),
        (ProverState::Done, false, 0),
        (ProverState::AwaitReveal, true, 1),
    ] {
        let coin = Coin::new(0x98_0E01);
        let mut engine = Engine::new(0x98_0E02);
        {
            let mut session = ProverSession::new(&mut engine);
            drive(&mut session, state, &coin);
            if abort_first {
                session.abort();
            }
        }
        assert_eq!(
            engine.count(discards),
            discarded,
            "dropped in {state:?}, aborted first: {abort_first}"
        );
    }
}

/// Taking the engine back from a session not done aborts it: the engine comes back having
/// discarded once. From a session done, it comes back untouched.
#[test]
fn taking_the_engine_back_aborts_a_session_not_done() {
    for (state, abort_first, discarded) in [
        (ProverState::AwaitCommit, false, 1),
        (ProverState::AwaitReveal, false, 1),
        (ProverState::AwaitQueries, false, 1),
        (ProverState::Done, false, 0),
        (ProverState::AwaitQueries, true, 1),
    ] {
        let (mut session, _) = at(state, 0x98_0F01);
        if abort_first {
            session.abort();
        }
        let engine = session.into_engine();
        assert_eq!(
            engine.count(discards),
            discarded,
            "taken back in {state:?}, aborted first: {abort_first}"
        );
    }
}

#[test]
fn live_phase_memory_covers_sent_state_and_query_shape_overlap() {
    let mut engine = Engine::new(73);
    let coin = Coin::new(91);
    let query_shape = shape(&MATRICES);
    engine.calls.reserve(4);
    for count in [0, 1, 2, 5, 17, 129] {
        engine.statement.batch.requests.resize(
            count,
            tessera_protocol::RequestRef {
                request_id: 1,
                session_id: [0; 16],
                prompt_len: 1,
                output_len: 0,
            },
        );
        let mut result = None;
        let mut query_owner = None;
        let mut session = ProverSession::new(&mut engine);
        let actual = allocation_counter::measure(|| {
            result = Some(session.start(&coin.commit).unwrap());
            query_owner = Some(query_shape.clone());
        });
        // Remove the returned Round1 clone. Engine call records use the
        // pre-existing capacity, leaving exactly the private owner allocation.
        let caller = count as u64 * size_of::<tessera_protocol::RequestRef>() as u64;
        let bound =
            tessera_protocol::prover_session_memory_bytes(count as u64, MATRICES.len() as u64)
                .unwrap();
        assert_eq!(actual.bytes_max, bound + caller, "{count}: {actual:?}");
        assert_eq!(u64::try_from(actual.bytes_current).unwrap() - caller, bound);
        drop(query_owner);
        drop(result);
        drop(session);
        engine.calls.clear();
    }
    let sizing = allocation_counter::measure(|| {
        assert_eq!(
            tessera_protocol::prover_session_memory_bytes(u64::MAX, 0),
            None
        );
        assert_eq!(
            tessera_protocol::prover_session_memory_bytes(0, u64::MAX),
            None
        );
        assert_eq!(
            tessera_protocol::prover_session_memory_bytes(1 << 61, 0),
            None
        );
    });
    assert_eq!(sizing.bytes_total, 0);
}
