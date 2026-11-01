// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The prover's live session (RFC-0036 'The prover's live session').
//!
//! | State | Message | Checks | Action | Next |
//! |---|---|---|---|---|
//! | `AwaitCommit` | `SeedCommit` | none | the engine runs the witness pass and commits; sends `Round1` with the commit's session and `ProverSeedShare` with the engine's `s_P` | `AwaitReveal` |
//! | `AwaitReveal` | `SeedReveal` | its session (`ERR-PROTO-ORDER`), then `H("coin-commit", s_V) = c_V` (`ERR-PROTO-COINTOSS`, then abort) | derives the seed; the engine proves the tree levels, stages and openings; sends `Round2Response` | `AwaitQueries` |
//! | `AwaitQueries` | `SeedReveal` | — | refused, `ERR-PROTO-REUSE`: it would answer a second seed for one `Round1` (INV-PROTO-1) | unchanged |
//! | `AwaitQueries` | `ColumnQueries` | the matrices, the counts and every index's range (`ERR-PROTO-QUERIES`) | the engine answers the distinct indices; sends `Round3Response` | `Done` |
//! | any | any other | — | refused, `ERR-PROTO-ORDER` | unchanged |
//!
//! A refused message leaves the state as it was. An abort is possible in any state: it
//! discards the session's `Round1`, its advice commitment and its randomness, and the session
//! is over, so a retry is a new session with a fresh `s_P` and fresh randomness (INV-PROTO-1).
//! A reveal that fails its commitment is an abort, so that a verifier cannot try a second `s_V`
//! against one `Round1`, and so is any failure of the engine: no seed is answered twice, and no
//! queries are answered twice. A session dropped, or whose engine is taken back, before it is
//! done aborts too, so that a session given up on a timeout or a transport error leaves no
//! `Round1` to answer a second seed.
//!
//! The session draws no randomness. The engine gives `s_P`, derived from its per-proof secret
//! seed (RFC-0018 'Crash consistency'): a prover that resumes after a crash sends the `Round1`
//! and the share it sent before, byte for byte, and never a second share with one `Round1`.

use core::fmt;

use tessera_hash::Digest;

use crate::{
    AdviceCommitment, ColumnQueries, LeafGroups, MessageKind, PROTOCOL_VERSION, ProtocolError,
    ProverSeedShare, PublicStatement, QueryShape, Round1, Round2Response, Round3Response,
    SeedCommit, SeedReveal, coin_commitment, coin_toss,
};

/// The proving work a prover's live session drives, one call per accepted message, in the
/// session order of RFC-0014. The prover implements it.
pub trait ProverEngine {
    /// Why the engine failed: a completeness failure, a resource, or a bug (04 §3).
    type Error;

    /// Runs the witness pass and commits for the session `session_id`: the statement and the
    /// advice commitment `Round1` carries. The session is `SeedCommit`'s, which the verifier
    /// chose; a non-interactive prover chooses its own, outside a live session.
    fn commit(
        &mut self,
        session_id: &[u8; 16],
    ) -> Result<(PublicStatement, AdviceCommitment), Self::Error>;

    /// `s_P`, the prover's share for the `Round1` just committed. It derives from the per-proof
    /// secret seed through a PRG under a label of its own (RFC-0018 'Crash consistency'), so that
    /// a prover that resumes after a crash gives the share it sent. Its freshness is the
    /// engine's: the per-proof seed is fresh for every proof and discarded on an abort, which
    /// gives every `Round1` a fresh share. The session draws nothing.
    fn prover_share(&mut self) -> Result<[u8; 32], Self::Error>;

    /// Proves the tree levels, the stages and the openings from `seed`, the live seed, on a
    /// transcript that absorbed `round1`, the `Round1` the session sent (RFC-0036 'What the
    /// transcript absorbs', rows 2 and 3). Returns the `Round2Response` to send, and the shape of
    /// the queries the engine answers: the parameter set's counts, and the codewords of the
    /// advice and of the matrices `weight_combined` opens, in its order.
    fn prove(
        &mut self,
        round1: &Round1,
        seed: &Digest,
    ) -> Result<(Round2Response, QueryShape), Self::Error>;

    /// Answers `queries`, which the session checked against the shape: each distinct index of
    /// each commitment once, in increasing order.
    fn answer(&mut self, queries: &ColumnQueries) -> Result<Round3Response, Self::Error>;

    /// The leaf stage's token groups for the `Round1` just committed, which `Round1` carries
    /// (RFC-0054 'Messages'): [`LeafGroups::NONE`] for a stage proven in one batch, the default.
    fn leaf_groups(&self) -> LeafGroups {
        LeafGroups::NONE
    }

    /// Discards everything the session committed to, after an abort: its `Round1`, its advice
    /// commitment and its randomness, the per-proof secret seed included (INV-PROTO-1). A later
    /// `commit` starts afresh, from a fresh seed.
    fn discard(&mut self);
}

/// A session can drive an engine it borrows, which then outlives the session.
impl<E: ProverEngine + ?Sized> ProverEngine for &mut E {
    type Error = E::Error;

    fn commit(
        &mut self,
        session_id: &[u8; 16],
    ) -> Result<(PublicStatement, AdviceCommitment), Self::Error> {
        (**self).commit(session_id)
    }

    fn prover_share(&mut self) -> Result<[u8; 32], Self::Error> {
        (**self).prover_share()
    }

    fn prove(
        &mut self,
        round1: &Round1,
        seed: &Digest,
    ) -> Result<(Round2Response, QueryShape), Self::Error> {
        (**self).prove(round1, seed)
    }

    fn answer(&mut self, queries: &ColumnQueries) -> Result<Round3Response, Self::Error> {
        (**self).answer(queries)
    }

    fn leaf_groups(&self) -> LeafGroups {
        (**self).leaf_groups()
    }

    fn discard(&mut self) {
        (**self).discard();
    }
}

/// The states of a prover's live session.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProverState {
    /// Waiting for the verifier's `SeedCommit`.
    AwaitCommit,
    /// `Round1` and the prover's share are sent; waiting for `SeedReveal`.
    AwaitReveal,
    /// `Round2Response` is sent; waiting for `ColumnQueries`.
    AwaitQueries,
    /// The session is over: `Round3Response` is sent, or the session aborted.
    Done,
}

/// Why a prover's session call failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SessionError<E> {
    /// The session refused the message, and is as it was, or aborted on it (RFC-0036): see
    /// [`ProtocolError`].
    Protocol(ProtocolError),
    /// The engine failed, and the session aborted.
    Engine(E),
}

impl<E: fmt::Display> fmt::Display for SessionError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Protocol(error) => error.fmt(f),
            Self::Engine(error) => write!(f, "the engine failed: {error}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for SessionError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Protocol(error) => Some(error),
            Self::Engine(error) => Some(error),
        }
    }
}

/// A prover's live session over an engine (RFC-0036 'The prover's live session'): `start`,
/// `on_seed`, `on_queries`, in the order of 02 §3.1.
///
/// Dropping a session that is not done aborts it, and so does taking its engine back with
/// [`into_engine`](Self::into_engine): the engine discards the session's `Round1` and its
/// randomness (INV-PROTO-1). A crash runs no drop, so the per-proof seed survives it, and a
/// resumed prover reproduces what it sent (RFC-0018 'Crash consistency').
pub struct ProverSession<E: ProverEngine> {
    /// The engine. Only `into_engine` takes it, as it consumes the session, so every other
    /// method finds it.
    engine: Option<E>,
    phase: Phase,
    aborted: bool,
}

/// A state, with what the session keeps in it.
enum Phase {
    AwaitCommit,
    AwaitReveal(Box<Sent>),
    AwaitQueries(QueryShape),
    Done,
}

/// What the session sent with `Round1`, and the commitment its reveal must open.
struct Sent {
    commit: SeedCommit,
    share: [u8; 32],
    round1: Round1,
}

/// Requested heap bound for the live prover session's private phase owners
/// (RFC-0036/0018): boxed sent state, retained Round1 request capacity and query
/// shape capacity. Sent and the returned query shape can overlap during on_seed.
/// Engine data, returned messages, packet encodings, inline session storage and
/// allocator overhead are separate. Does not allocate or validate. Returns
/// `None` on overflowing or unaddressable capacities.
pub fn prover_session_memory_bytes(request_capacity: u64, matrix_capacity: u64) -> Option<u64> {
    fn array<T>(n: u64) -> Option<u64> {
        let bytes = n.checked_mul(size_of::<T>() as u64)?;
        (usize::try_from(n).is_ok() && bytes <= isize::MAX as u64).then_some(bytes)
    }
    (size_of::<Sent>() as u64)
        .checked_add(array::<crate::RequestRef>(request_capacity)?)?
        .checked_add(array::<crate::MatrixCodeword>(matrix_capacity)?)
}

/// The engine a session holds, which is always there: only `into_engine` takes it, and it
/// consumes the session.
fn held<E>(engine: &mut Option<E>) -> &mut E {
    engine
        .as_mut()
        .expect("only into_engine takes the engine, and it consumes the session")
}

/// The state, never a message or a share.
impl<E: ProverEngine> fmt::Debug for ProverSession<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProverSession")
            .field("state", &self.state())
            .field("aborted", &self.aborted)
            .finish_non_exhaustive()
    }
}

/// A session given up before its end aborts (INV-PROTO-1).
impl<E: ProverEngine> Drop for ProverSession<E> {
    fn drop(&mut self) {
        self.abort();
    }
}

impl<E: ProverEngine> ProverSession<E> {
    /// A session in `AwaitCommit`, over `engine`.
    pub fn new(engine: E) -> Self {
        Self {
            engine: Some(engine),
            phase: Phase::AwaitCommit,
            aborted: false,
        }
    }

    /// The session's state.
    pub fn state(&self) -> ProverState {
        match self.phase {
            Phase::AwaitCommit => ProverState::AwaitCommit,
            Phase::AwaitReveal(_) => ProverState::AwaitReveal,
            Phase::AwaitQueries(_) => ProverState::AwaitQueries,
            Phase::Done => ProverState::Done,
        }
    }

    /// Whether the session ended by an abort rather than by sending `Round3Response`.
    pub fn is_aborted(&self) -> bool {
        self.aborted
    }

    /// The engine.
    pub fn engine(&self) -> &E {
        self.engine
            .as_ref()
            .expect("only into_engine takes the engine, and it consumes the session")
    }

    /// The engine, for the next session. A session that is not done aborts first, so that the
    /// engine discards its `Round1` and its randomness (INV-PROTO-1).
    pub fn into_engine(mut self) -> E {
        self.abort();
        self.engine
            .take()
            .expect("only into_engine takes the engine, and it consumes the session")
    }

    /// `SeedCommit` in `AwaitCommit`: the engine commits and gives `s_P`, and the session sends
    /// `Round1` with the commit's session and `ProverSeedShare` with that share, then awaits the
    /// reveal.
    ///
    /// # Errors
    ///
    /// `ERR-PROTO-ORDER` in any other state, which leaves it as it was. An engine that fails
    /// aborts the session.
    pub fn start(
        &mut self,
        commit: &SeedCommit,
    ) -> Result<(Round1, ProverSeedShare), SessionError<E::Error>> {
        if !matches!(self.phase, Phase::AwaitCommit) {
            return Err(SessionError::Protocol(ProtocolError::OutOfOrder {
                message: MessageKind::SeedCommit,
            }));
        }
        let (statement, advice) = match held(&mut self.engine).commit(&commit.session_id) {
            Ok(parts) => parts,
            Err(error) => {
                self.abort();
                return Err(SessionError::Engine(error));
            }
        };
        let share = match held(&mut self.engine).prover_share() {
            Ok(share) => share,
            Err(error) => {
                self.abort();
                return Err(SessionError::Engine(error));
            }
        };
        let round1 = Round1 {
            protocol_version: PROTOCOL_VERSION,
            session_id: commit.session_id,
            statement,
            advice,
            leaf_groups: held(&mut self.engine).leaf_groups(),
        };
        let share_message = ProverSeedShare {
            session_id: commit.session_id,
            share,
        };
        let reply = (round1.clone(), share_message);
        self.phase = Phase::AwaitReveal(Box::new(Sent {
            commit: commit.clone(),
            share,
            round1,
        }));
        Ok(reply)
    }

    /// `SeedReveal` in `AwaitReveal`: the session checks the reveal, derives the seed
    /// `H("coin-toss", s_V ‖ s_P)`, and the engine proves from it; the session sends
    /// `Round2Response` and awaits the queries.
    ///
    /// # Errors
    ///
    /// In this order, each leaving the session as it was unless it says otherwise:
    /// - `ERR-PROTO-REUSE` in `AwaitQueries`: the seed of this `Round1` is answered
    ///   (INV-PROTO-1);
    /// - `ERR-PROTO-ORDER` in `AwaitCommit` or `Done`;
    /// - `ERR-PROTO-ORDER` for a reveal of another session;
    /// - `ERR-PROTO-COINTOSS` for a share that does not open the commitment, which aborts the
    ///   session;
    /// - an engine failure, or a query shape that does not match `weight_combined` or that no
    ///   queries can have (`ERR-INTERNAL`), which abort it.
    pub fn on_seed(
        &mut self,
        reveal: &SeedReveal,
    ) -> Result<Round2Response, SessionError<E::Error>> {
        let sent = match &self.phase {
            Phase::AwaitReveal(sent) => sent,
            Phase::AwaitQueries(_) => return Err(SessionError::Protocol(ProtocolError::Reuse)),
            Phase::AwaitCommit | Phase::Done => {
                return Err(SessionError::Protocol(ProtocolError::OutOfOrder {
                    message: MessageKind::SeedReveal,
                }));
            }
        };
        if reveal.session_id != sent.commit.session_id {
            return Err(SessionError::Protocol(ProtocolError::OtherSession {
                message: MessageKind::SeedReveal,
            }));
        }
        if coin_commitment(&reveal.share) != sent.commit.commitment {
            self.abort();
            return Err(SessionError::Protocol(ProtocolError::CoinToss));
        }
        let seed = coin_toss(&reveal.share, &sent.share);
        let (round2, queries) = match held(&mut self.engine).prove(&sent.round1, &seed) {
            Ok(proven) => proven,
            Err(error) => {
                self.abort();
                return Err(SessionError::Engine(error));
            }
        };
        if let Err(error) = check_opened(&queries, &round2) {
            self.abort();
            return Err(SessionError::Protocol(error));
        }
        self.phase = Phase::AwaitQueries(queries);
        Ok(round2)
    }

    /// `ColumnQueries` in `AwaitQueries`: the session checks the queries (INV-PROTO-3), the
    /// engine answers them, and the session sends `Round3Response` and is done.
    ///
    /// # Errors
    ///
    /// `ERR-PROTO-ORDER` in any other state, queries that arrive before `Round2Response` is
    /// sent included, and `ERR-PROTO-QUERIES` for queries of other matrices, of another count
    /// or with an index past its codeword ([`QueryShape::check`]): both leave the session as it
    /// was. An engine failure aborts it.
    pub fn on_queries(
        &mut self,
        queries: &ColumnQueries,
    ) -> Result<Round3Response, SessionError<E::Error>> {
        let Phase::AwaitQueries(shape) = &self.phase else {
            return Err(SessionError::Protocol(ProtocolError::OutOfOrder {
                message: MessageKind::ColumnQueries,
            }));
        };
        if let Err(fault) = shape.check(queries) {
            return Err(SessionError::Protocol(ProtocolError::Queries(fault)));
        }
        match held(&mut self.engine).answer(queries) {
            Ok(round3) => {
                self.phase = Phase::Done;
                Ok(round3)
            }
            Err(error) => {
                self.abort();
                Err(SessionError::Engine(error))
            }
        }
    }

    /// Aborts the session: the engine discards its `Round1`, advice commitment and randomness
    /// (INV-PROTO-1), and every later message is refused. Nothing happens once the session is
    /// done.
    pub fn abort(&mut self) {
        if matches!(self.phase, Phase::Done) {
            return;
        }
        held(&mut self.engine).discard();
        self.phase = Phase::Done;
        self.aborted = true;
    }
}

/// The engine's query shape against the `Round2Response` it comes with: a shape queries can
/// have, for the matrices `weight_combined` opens, in its order (RFC-0036 '`ColumnQueries`').
/// Another shape is the engine's bug.
fn check_opened(queries: &QueryShape, round2: &Round2Response) -> Result<(), ProtocolError> {
    queries.validate()?;
    let same = queries.matrices.len() == round2.weight_combined.len()
        && queries
            .matrices
            .iter()
            .zip(&round2.weight_combined)
            .all(|(codeword, rows)| codeword.matrix == rows.matrix);
    if !same {
        return Err(ProtocolError::Internal {
            rule: "the queried matrices are those weight_combined opens, in its order",
        });
    }
    Ok(())
}
