// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The verifier's live session (RFC-0036 'The verifier's live session').
//!
//! | State | Message | Action | Next |
//! |---|---|---|---|
//! | `Start` | — | draws `session_id` and `s_V` from the CSPRNG, and sends `SeedCommit` | `AwaitRound1` |
//! | `AwaitRound1` | `Round1`, `ProverSeedShare` | in this order: `Round1` names the session, the share names it, `Round1`'s version is the session's; then the checks run RFC-0019 steps 1 to 6 with the seed; sends `SeedReveal` | `AwaitRound2` |
//! | `AwaitRound2` | `Round2Response` | the checks run steps 7 to 9; step 10 draws the column queries from the CSPRNG, which the checks absorb under `queries`; sends `ColumnQueries` | `AwaitRound3` |
//! | `AwaitRound3` | `Round3Response` | the checks run steps 11 to 13 and give the verdict | `Done` |
//!
//! The session ends at the first failed check, with the verdict that check gives. A message the
//! state does not expect, or one that names another session, ends it `Inconclusive` with
//! `ERR-PROTO-ORDER`, and a `Round1` of another protocol version with `ERR-PROTO-VERSION`. Only
//! acceptance certifies (04 §1), so a prover gains nothing by forcing either.
//!
//! The verifier's share stays in the session, out of every debug form, until `SeedReveal`
//! carries it; a session that ends before then, or that is dropped, clears it. The verifier owes
//! nothing more to a session it gives up, which certifies nothing. The share is committed
//! before the prover's share is seen, and revealed only after `Round1` and the prover's share
//! are in and checked, so the seed is fixed only after `Round1` is complete (INV-PROTO-2). The
//! column queries are drawn only after the checks have taken `Round2Response` in whole
//! (INV-PROTO-3).

use core::fmt;

use tessera_codec::Zeroize;
use tessera_hash::Digest;

use crate::{
    ColumnQueries, MessageKind, PROTOCOL_VERSION, ProtocolError, ProverSeedShare, QueryShape,
    Round1, Round2Response, Round3Response, SeedCommit, SeedReveal, coin_commitment, coin_toss,
};

/// The checks of RFC-0019 a verifier's live session runs, in its check order, with the
/// transcript they replay. The verifier implements them; each returns the verdict of the first
/// check that fails.
pub trait VerifierChecks {
    /// What a check that fails gives, and what the last checks give: RFC-0019's verdict.
    type Verdict;

    /// RFC-0019 steps 1 to 6 for `round1`, whose session and version the session has checked:
    /// decoding, parameters, claim list, statement, then the transcript, which absorbs `round1`
    /// under `round1` and the 32 bytes of `seed` under `seed`.
    fn round1(&mut self, round1: &Round1, seed: &Digest) -> Result<(), Self::Verdict>;

    /// Steps 7 to 9 for `round2`: the tree levels, the stages and the opening groups. Returns
    /// the queries' shape, which the verifier derives from public data: the parameter set's
    /// counts and the codewords of the advice and of the matrices with claims.
    fn round2(&mut self, round2: &Round2Response) -> Result<QueryShape, Self::Verdict>;

    /// Absorbs the queries the session drew at step 10 under `queries` (RFC-0036 'What the
    /// transcript absorbs', row 10), before they are sent, and keeps them for the column checks.
    fn queries(&mut self, queries: &ColumnQueries);

    /// Steps 11 to 13 for `round3`, the answers to the queries step 10 drew: the columns, the
    /// scoring claims and the ledger. Returns the verdict.
    fn round3(&mut self, round3: &Round3Response) -> Self::Verdict;
}

/// The states of a verifier's live session.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VerifierState {
    /// Nothing is sent yet.
    Start,
    /// `SeedCommit` is sent; waiting for `Round1` and the prover's share.
    AwaitRound1,
    /// `SeedReveal` is sent; waiting for `Round2Response`.
    AwaitRound2,
    /// `ColumnQueries` is sent; waiting for `Round3Response`.
    AwaitRound3,
    /// The session is over.
    Done,
}

/// How a verifier's live session ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ending<V> {
    /// The session refused a message: `Inconclusive` with the error's code, `ERR-PROTO-ORDER`
    /// or `ERR-PROTO-VERSION`, or it failed on its own side.
    Inconclusive(ProtocolError),
    /// The verdict of the check that failed, or of the last checks.
    Verdict(V),
}

/// A verifier's live session over its checks (RFC-0036 'The verifier's live session'):
/// `start`, `on_round1`, `on_round2`, `on_round3`, in the order of 02 §3.2.
pub struct VerifierSession<C> {
    checks: C,
    phase: Phase,
}

/// A state, with what the session keeps in it.
enum Phase {
    Start,
    AwaitRound1 {
        session_id: [u8; 16],
        share: Box<Share>,
    },
    AwaitRound2,
    AwaitRound3,
    Done,
}

/// The verifier's share `s_V` before `SeedReveal` carries it, cleared when it is dropped. The
/// session draws it on the heap and keeps it there, so that moving the session's state moves a
/// pointer, not the share (05 §3 OBS-2). The hashes that read it may keep copies in their own
/// state, which this does not reach.
struct Share([u8; 32]);

impl Drop for Share {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

/// The state, never the share.
impl<C> fmt::Debug for VerifierSession<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VerifierSession")
            .field("state", &self.state())
            .finish_non_exhaustive()
    }
}

impl<C> VerifierSession<C> {
    /// A session in `Start`, over `checks`.
    pub fn new(checks: C) -> Self {
        Self {
            checks,
            phase: Phase::Start,
        }
    }

    /// The session's state.
    pub fn state(&self) -> VerifierState {
        match self.phase {
            Phase::Start => VerifierState::Start,
            Phase::AwaitRound1 { .. } => VerifierState::AwaitRound1,
            Phase::AwaitRound2 => VerifierState::AwaitRound2,
            Phase::AwaitRound3 => VerifierState::AwaitRound3,
            Phase::Done => VerifierState::Done,
        }
    }

    /// The checks.
    pub fn checks(&self) -> &C {
        &self.checks
    }
}

impl<C: VerifierChecks> VerifierSession<C> {
    /// In `Start`: draws a fresh session id and share `s_V` (INV-PROTO-4), and sends
    /// `SeedCommit` with `c_V = H("coin-commit", s_V)`.
    ///
    /// # Errors
    ///
    /// `ERR-PROTO-ORDER` in any other state, and [`ProtocolError::Randomness`] if the random
    /// source fails. Either ends the session.
    pub fn start(&mut self) -> Result<SeedCommit, ProtocolError> {
        if !matches!(
            core::mem::replace(&mut self.phase, Phase::Done),
            Phase::Start
        ) {
            return Err(ProtocolError::OutOfOrder {
                message: MessageKind::SeedCommit,
            });
        }
        let mut session_id = [0u8; 16];
        let mut share = Box::new(Share([0u8; 32]));
        getrandom::fill(&mut session_id).map_err(|_| ProtocolError::Randomness)?;
        getrandom::fill(&mut share.0).map_err(|_| ProtocolError::Randomness)?;
        let commit = SeedCommit {
            session_id,
            commitment: coin_commitment(&share.0),
        };
        self.phase = Phase::AwaitRound1 { session_id, share };
        Ok(commit)
    }

    /// `Round1` and `ProverSeedShare` in `AwaitRound1`: checks that both name the session and
    /// that `Round1` speaks [`PROTOCOL_VERSION`], derives the seed `H("coin-toss", s_V ‖ s_P)`,
    /// runs RFC-0019 steps 1 to 6, and sends `SeedReveal`.
    ///
    /// # Errors
    ///
    /// In this order, each ending the session: `Inconclusive(ERR-PROTO-ORDER)` in any other
    /// state, for a `Round1` of another session, then for a share of another session;
    /// `Inconclusive(ERR-PROTO-VERSION)` for another protocol version; then the verdict of the
    /// first failed check. The share is not revealed.
    pub fn on_round1(
        &mut self,
        round1: &Round1,
        share: &ProverSeedShare,
    ) -> Result<SeedReveal, Ending<C::Verdict>> {
        let Phase::AwaitRound1 {
            session_id,
            share: verifier,
        } = core::mem::replace(&mut self.phase, Phase::Done)
        else {
            return Err(Ending::Inconclusive(ProtocolError::OutOfOrder {
                message: MessageKind::Round1,
            }));
        };
        if round1.session_id != session_id {
            return Err(Ending::Inconclusive(ProtocolError::OtherSession {
                message: MessageKind::Round1,
            }));
        }
        if share.session_id != session_id {
            return Err(Ending::Inconclusive(ProtocolError::OtherSession {
                message: MessageKind::ProverSeedShare,
            }));
        }
        if round1.protocol_version != PROTOCOL_VERSION {
            return Err(Ending::Inconclusive(ProtocolError::Version {
                found: round1.protocol_version,
            }));
        }
        let seed = coin_toss(&verifier.0, &share.share);
        self.checks.round1(round1, &seed).map_err(Ending::Verdict)?;
        self.phase = Phase::AwaitRound2;
        Ok(SeedReveal {
            session_id,
            share: verifier.0,
        })
    }

    /// `Round2Response` in `AwaitRound2`: runs steps 7 to 9, then step 10: draws the column
    /// queries of the shape the checks derive from the CSPRNG, has the checks absorb them, and
    /// sends them.
    ///
    /// # Errors
    ///
    /// Each ending the session: `Inconclusive(ERR-PROTO-ORDER)` in any other state; the verdict
    /// of the first failed check; `Inconclusive` for a shape no queries can have
    /// (`ERR-INTERNAL`) or a failed random source.
    pub fn on_round2(
        &mut self,
        round2: &Round2Response,
    ) -> Result<ColumnQueries, Ending<C::Verdict>> {
        if !matches!(
            core::mem::replace(&mut self.phase, Phase::Done),
            Phase::AwaitRound2
        ) {
            return Err(Ending::Inconclusive(ProtocolError::OutOfOrder {
                message: MessageKind::Round2Response,
            }));
        }
        let shape = self.checks.round2(round2).map_err(Ending::Verdict)?;
        let queries = shape.draw().map_err(Ending::Inconclusive)?;
        self.checks.queries(&queries);
        self.phase = Phase::AwaitRound3;
        Ok(queries)
    }

    /// `Round3Response` in `AwaitRound3`: runs steps 11 to 13 and ends the session with their
    /// verdict, or `Inconclusive(ERR-PROTO-ORDER)` in any other state.
    pub fn on_round3(&mut self, round3: &Round3Response) -> Ending<C::Verdict> {
        if !matches!(
            core::mem::replace(&mut self.phase, Phase::Done),
            Phase::AwaitRound3
        ) {
            return Ending::Inconclusive(ProtocolError::OutOfOrder {
                message: MessageKind::Round3Response,
            });
        }
        Ending::Verdict(self.checks.round3(round3))
    }
}
