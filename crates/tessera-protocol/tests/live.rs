// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A live session between the two sides, in RFC-0014's order, every message sealed in its
//! envelope by its sender and opened by its receiver (RFC-0036 'Versions and envelopes').
//!
//! Both sides derive the same seed from their shares, the verifier's queries are queries the
//! prover answers, and both sessions end done.

mod common;

use common::{CheckCall, Checks, Engine, EngineCall, Verdict};
use tessera_codec::DecodeLimits;
use tessera_protocol::{
    ColumnQueries, Ending, Message, ProverSeedShare, ProverSession, ProverState, Round1,
    Round2Response, Round3Response, SeedCommit, SeedReveal, VerifierSession, VerifierState,
};

/// `message` as its receiver gets it: sealed, then opened.
fn send<T: Message + PartialEq + std::fmt::Debug>(message: &T) -> T {
    let opened = T::open(&message.seal(), DecodeLimits::MESSAGE).expect("an envelope");
    assert_eq!(&opened, message);
    opened
}

#[test]
fn a_live_session_runs_between_the_two_sides() {
    let mut verifier = VerifierSession::new(Checks::new());
    let mut prover = ProverSession::new(Engine::new(0x98_2001));

    let commit: SeedCommit = send(&verifier.start().expect("SeedCommit"));
    let (round1, share) = prover.start(&commit).expect("Round1");
    let (round1, share): (Round1, ProverSeedShare) = (send(&round1), send(&share));
    let reveal: SeedReveal = send(&verifier.on_round1(&round1, &share).expect("SeedReveal"));
    let round2: Round2Response = send(&prover.on_seed(&reveal).expect("Round2Response"));
    let queries: ColumnQueries = send(&verifier.on_round2(&round2).expect("ColumnQueries"));
    let round3: Round3Response = send(&prover.on_queries(&queries).expect("Round3Response"));
    assert_eq!(
        verifier.on_round3(&round3),
        Ending::Verdict(Verdict::Accept)
    );

    assert_eq!(prover.state(), ProverState::Done);
    assert!(!prover.is_aborted());
    assert_eq!(verifier.state(), VerifierState::Done);
    // One seed, derived on both sides from the two shares.
    let proven = prover.engine().calls.iter().find_map(|call| match call {
        EngineCall::Prove(_, seed) => Some(*seed),
        _ => None,
    });
    let checked = verifier.checks().calls.iter().find_map(|call| match call {
        CheckCall::Round1(_, seed) => Some(*seed),
        _ => None,
    });
    assert!(proven.is_some());
    assert_eq!(proven, checked);
    // The prover answered the very queries the verifier drew and absorbed.
    assert!(
        verifier
            .checks()
            .calls
            .contains(&CheckCall::Queries(queries.clone()))
    );
    assert_eq!(
        prover.engine().calls.last(),
        Some(&EngineCall::Answer(queries))
    );
}
