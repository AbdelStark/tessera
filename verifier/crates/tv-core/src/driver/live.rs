// SPDX-License-Identifier: Apache-2.0 OR MIT

//! One-shot Live session; verifier entropy is drawn only on the supported Unix hosts.
use super::*;
use std::io::Read;
use tv_hash::{Domain, DomainHasher, hash};
use zeroize::Zeroizing;

// Private, non-Debug/non-Clone and heap-resident until it becomes a SeedReveal message.
struct SecretSeed(Zeroizing<Vec<u8>>);
enum State {
    Created,
    Committed {
        id: [u8; 16],
        share: SecretSeed,
    },
    Round2 {
        first: Vec<u8>,
        seed: Digest,
    },
    Round3 {
        first: Vec<u8>,
        second: Vec<u8>,
        queries: Vec<u8>,
        seed: Digest,
    },
    Done,
}
/// Verifier-owned Live session. Only on_round3 can return Accept; every failure is terminal
/// except an out-of-order call, which leaves the current state unchanged (RFC-0036).
/// No Debug/Clone implementation exposes the unrevealed coin-toss share.
#[must_use = "complete Round3; seed/query messages are not proof acceptance"]
pub struct LiveVerifier {
    claims: ClaimList,
    statement: PublicStatement,
    cfg: VerifierConfig,
    state: State,
    started: Option<Instant>,
}
fn entropy() -> Check<std::fs::File> {
    #[cfg(unix)]
    {
        std::fs::File::open("/dev/urandom").map_err(|_| fail(ErrorCode::Io))
    }
    #[cfg(not(unix))]
    {
        Err(fail(ErrorCode::Io))
    }
}
fn fill(file: &mut impl Read, bytes: &mut [u8]) -> Check<()> {
    file.read_exact(bytes).map_err(|_| fail(ErrorCode::Io))
}
fn own_decode<T>(
    bytes: &[u8],
    decode: impl FnOnce(&mut tv_codec::Cursor<'_>) -> Result<T, tv_codec::DecodeError>,
) -> Check<T> {
    tv_codec::decode(bytes, &DecodeLimits::MESSAGE, decode).map_err(|e| {
        if e.code() == tv_codec::ErrorCode::Memory {
            memory()
        } else {
            internal()
        }
    })
}
impl LiveVerifier {
    /// Take ownership of trusted inputs. Randomness and validation start in the session methods.
    pub fn new(claims: ClaimList, statement: PublicStatement, cfg: VerifierConfig) -> Self {
        Self {
            claims,
            statement,
            cfg,
            state: State::Created,
            started: None,
        }
    }
    /// Commit to a fresh OS-random share and a fresh session id.
    ///
    /// # Errors
    /// Inconclusive Order for a repeated call, Params for another mode/name, Memory for
    /// a policy below the 32-byte share or reservation failure, IO for unavailable entropy.
    /// Unsupported hosts fail closed.
    #[allow(
        clippy::result_large_err,
        reason = "the specified Live API returns Verdict; reports stay by value without an infallible allocation"
    )]
    pub fn start(&mut self) -> Result<protocol::SeedCommit, Verdict> {
        self.start_observed(&mut ())
    }
    /// [`start`](Self::start), reporting its work to `observer` as step 2, `params`: the
    /// parameter set and mode, then the session's limits and entropy (05 §2,
    /// [`observe`](crate::observe)).
    ///
    /// # Errors
    /// As [`start`](Self::start).
    #[allow(
        clippy::result_large_err,
        reason = "the specified Live API returns Verdict; reports stay by value without an infallible allocation"
    )]
    pub fn start_observed(
        &mut self,
        observer: &mut dyn Observer,
    ) -> Result<protocol::SeedCommit, Verdict> {
        let mut tracker = Tracker::new(observer);
        tracker.enter(CheckKind::Params);
        let result = self.start_inner();
        let result = conclude(&mut tracker, result);
        if result.as_ref().is_err_and(|error| {
            !matches!(error, Failure::Inconclusive(e) if e.code == ErrorCode::ProtocolOrder)
        }) {
            self.state = State::Done;
        }
        result.map_err(Into::into)
    }
    fn start_inner(&mut self) -> Check<protocol::SeedCommit> {
        if !matches!(self.state, State::Created) {
            return Err(fail(ErrorCode::ProtocolOrder));
        }
        if ParamSet::accept(&self.claims.params) != Some(self.cfg.params)
            || !mode_allowed(self.cfg.params, Mode::Live)
        {
            return Err(params_error(&self.claims));
        }
        if self.cfg.limits.max_memory_bytes < 32 {
            return Err(memory());
        }
        if self.cfg.limits.max_wall_time.is_zero() {
            return Err(fail(ErrorCode::Time));
        }
        let mut file = entropy()?;
        let mut id = [0; 16];
        fill(&mut file, &mut id)?;
        let mut bytes = reserve(32)?;
        bytes.resize(32, 0);
        let mut share = SecretSeed(Zeroizing::new(bytes));
        fill(&mut file, share.0.as_mut_slice())?;
        let commitment = hash(Domain::CoinCommit, share.0.as_slice());
        self.state = State::Committed { id, share };
        self.started = Some(Instant::now());
        Ok(protocol::SeedCommit {
            session_id: id,
            commitment,
        })
    }
    fn timed_out(&mut self) -> Check<()> {
        if self
            .started
            .is_some_and(|t| t.elapsed() >= self.cfg.limits.max_wall_time)
        {
            self.state = State::Done;
            return Err(fail(ErrorCode::ProtocolTimeout));
        }
        Ok(())
    }
    /// Bind Round1 and the prover's share before revealing the verifier's share.
    ///
    /// # Errors
    /// The first RFC-0019 predicate failure, or a session/order/resource error. A rejected
    /// Round1 never receives a seed reveal, and the unrevealed share is cleared.
    #[allow(
        clippy::result_large_err,
        reason = "the specified Live API returns Verdict; reports stay by value without an infallible allocation"
    )]
    pub fn on_round1(
        &mut self,
        message: &protocol::Round1,
        prover: &protocol::ProverSeedShare,
    ) -> Result<protocol::SeedReveal, Verdict> {
        self.on_round1_observed(message, prover, &mut ())
    }
    /// [`on_round1`](Self::on_round1), reporting each check to `observer` as it completes: steps
    /// 1 to 6 on success, all before the reveal (05 §2, [`observe`](crate::observe)).
    ///
    /// # Errors
    /// As [`on_round1`](Self::on_round1).
    #[allow(
        clippy::result_large_err,
        reason = "the specified Live API returns Verdict; reports stay by value without an infallible allocation"
    )]
    pub fn on_round1_observed(
        &mut self,
        message: &protocol::Round1,
        prover: &protocol::ProverSeedShare,
        observer: &mut dyn Observer,
    ) -> Result<protocol::SeedReveal, Verdict> {
        let mut tracker = Tracker::new(observer);
        tracker.enter(CheckKind::Input);
        let result = self.round1(message, prover, &mut tracker);
        conclude(&mut tracker, result).map_err(Into::into)
    }
    fn round1(
        &mut self,
        message: &protocol::Round1,
        prover: &protocol::ProverSeedShare,
        tracker: &mut Tracker<'_>,
    ) -> Check<protocol::SeedReveal> {
        if !matches!(self.state, State::Committed { .. }) {
            return Err(fail(ErrorCode::ProtocolOrder));
        }
        self.timed_out()?;
        let State::Committed { id, share } = std::mem::replace(&mut self.state, State::Done) else {
            return Err(internal());
        };
        if message.session_id != id || prover.session_id != id {
            return Err(fail(ErrorCode::ProtocolOrder));
        }
        match verify_unit_tracked(
            &self.claims,
            &self.statement,
            Frame {
                round1: message,
                round2: &EMPTY_ROUND2,
                round3: &EMPTY_ROUND3,
            },
            &self.cfg,
            Session::Initial,
            tracker,
            self.started.ok_or_else(internal)?,
        )? {
            Outcome::Ready => {}
            _ => return Err(internal()),
        }
        // Step 5 reserves nothing in `tessera-protocol/1`; step 6 binds Round1 and the seed.
        tracker.not_applicable(CheckKind::WeightReservation);
        tracker.enter(CheckKind::Transcript);
        let first = message.to_tce().map_err(codec)?;
        let context = SessionContext {
            protocol: ascii("tessera-protocol/1")?,
            mode: Mode::Live,
            params: tv_hash::transcript::ParamSetId(ascii(self.cfg.params.name())?),
            statement_digest: self.statement.digest(),
            session_id: id,
        };
        let round1_label = ascii("round1")?;
        let seed_label = ascii("seed")?;
        // All fallible/header work is complete before declassifying the committed share.
        let mut h = DomainHasher::new(Domain::CoinToss);
        h.update(share.0.as_slice());
        h.update(&prover.share);
        let seed = h.finalize();
        let mut t = Transcript::new(&context);
        t.absorb(round1_label, &first);
        t.absorb(seed_label, &seed);
        // The session budget also covers transcript work before the share is declassified.
        deadline(self.started.ok_or_else(internal)?, self.cfg.limits)?;
        let mut revealed = [0; 32];
        revealed.copy_from_slice(share.0.as_slice());
        self.state = State::Round2 { first, seed };
        Ok(protocol::SeedReveal {
            session_id: id,
            share: revealed,
        })
    }
    /// Check all Round2 equations, original evaluations and the runtime budget, then draw
    /// fresh OS-random column indices. A second call cannot draw a second query set.
    ///
    /// # Errors
    /// Exact proof rejections or operational/order errors. No queries follow an invalid Round2.
    #[allow(
        clippy::result_large_err,
        reason = "the specified Live API returns Verdict; reports stay by value without an infallible allocation"
    )]
    pub fn on_round2(
        &mut self,
        message: &protocol::Round2Response,
    ) -> Result<protocol::ColumnQueries, Verdict> {
        self.on_round2_observed(message, &mut ())
    }
    /// [`on_round2`](Self::on_round2), reporting each check to `observer` as it completes: steps
    /// 1 to 10 on success, all before the queries go out (05 §2, [`observe`](crate::observe)).
    ///
    /// # Errors
    /// As [`on_round2`](Self::on_round2).
    #[allow(
        clippy::result_large_err,
        reason = "the specified Live API returns Verdict; reports stay by value without an infallible allocation"
    )]
    pub fn on_round2_observed(
        &mut self,
        message: &protocol::Round2Response,
        observer: &mut dyn Observer,
    ) -> Result<protocol::ColumnQueries, Verdict> {
        let mut tracker = Tracker::new(observer);
        tracker.enter(CheckKind::Input);
        let result = self.round2(message, &mut tracker);
        conclude(&mut tracker, result).map_err(Into::into)
    }
    fn round2(
        &mut self,
        message: &protocol::Round2Response,
        tracker: &mut Tracker<'_>,
    ) -> Check<protocol::ColumnQueries> {
        if !matches!(self.state, State::Round2 { .. }) {
            return Err(fail(ErrorCode::ProtocolOrder));
        }
        self.timed_out()?;
        let State::Round2 { first, seed } = std::mem::replace(&mut self.state, State::Done) else {
            return Err(internal());
        };
        let round1 = own_decode(&first, protocol::Round1::decode)?;
        let result = verify_unit_tracked(
            &self.claims,
            &self.statement,
            Frame {
                round1: &round1,
                round2: message,
                round3: &EMPTY_ROUND3,
            },
            &self.cfg,
            Session::Live {
                seed: &seed,
                queries: None,
            },
            tracker,
            self.started.ok_or_else(internal)?,
        )?;
        let Outcome::Queries(queries) = result else {
            return Err(internal());
        };
        let second = message.to_tce().map_err(codec)?;
        let encoded_queries = queries.to_tce().map_err(codec)?;
        deadline(self.started.ok_or_else(internal)?, self.cfg.limits)?;
        self.state = State::Round3 {
            first,
            second,
            queries: encoded_queries,
            seed,
        };
        Ok(queries)
    }
    /// Authenticate every queried column and complete the exact proof. The response consumes
    /// the session on either acceptance or rejection.
    pub fn on_round3(&mut self, message: &protocol::Round3Response) -> Verdict {
        self.on_round3_observed(message, &mut ())
    }
    /// [`on_round3`](Self::on_round3), reporting each check to `observer` as it completes: steps
    /// 1 to 13 on acceptance (05 §2, [`observe`](crate::observe)). The verdict is the unobserved
    /// one.
    pub fn on_round3_observed(
        &mut self,
        message: &protocol::Round3Response,
        observer: &mut dyn Observer,
    ) -> Verdict {
        let mut tracker = Tracker::new(observer);
        tracker.enter(CheckKind::Input);
        let result = self.round3(message, &mut tracker);
        into_verdict(conclude(&mut tracker, result))
    }
    fn round3(
        &mut self,
        message: &protocol::Round3Response,
        tracker: &mut Tracker<'_>,
    ) -> Check<AcceptReport> {
        if !matches!(self.state, State::Round3 { .. }) {
            return Err(fail(ErrorCode::ProtocolOrder));
        }
        self.timed_out()?;
        let State::Round3 {
            first,
            second,
            queries,
            seed,
        } = std::mem::replace(&mut self.state, State::Done)
        else {
            return Err(internal());
        };
        let round1 = own_decode(&first, protocol::Round1::decode)?;
        let round2 = own_decode(&second, protocol::Round2Response::decode)?;
        let queries = own_decode(&queries, protocol::ColumnQueries::decode)?;
        // shortcut: replay immutable Round2 to reconstruct borrowed PCS state; retain an owned
        // verified column plan when optimizing Live verification latency after the MVP.
        match verify_unit_tracked(
            &self.claims,
            &self.statement,
            Frame {
                round1: &round1,
                round2: &round2,
                round3: message,
            },
            &self.cfg,
            Session::Live {
                seed: &seed,
                queries: Some(&queries),
            },
            tracker,
            self.started.ok_or_else(internal)?,
        )? {
            Outcome::Accepted(report) => Ok(report),
            _ => Err(internal()),
        }
    }
}
/// Where a recorded session resumes, for the tests that replay one through the observed calls.
#[cfg(all(test, feature = "test-params"))]
pub(super) enum Phase {
    /// Round1's turn, as `start` leaves it, the verifier's share all zeros.
    Round1 { session_id: [u8; 16] },
    /// Round2's turn, as `on_round1` leaves it.
    Round2 { first: Vec<u8>, seed: Digest },
    /// Round3's turn, as `on_round2` leaves it.
    Round3 {
        first: Vec<u8>,
        second: Vec<u8>,
        queries: Vec<u8>,
        seed: Digest,
    },
}
#[cfg(all(test, feature = "test-params"))]
impl LiveVerifier {
    pub(super) fn at(
        claims: ClaimList,
        statement: PublicStatement,
        cfg: VerifierConfig,
        phase: Phase,
    ) -> Self {
        let mut session = Self::new(claims, statement, cfg);
        session.state = match phase {
            Phase::Round1 { session_id } => State::Committed {
                id: session_id,
                share: SecretSeed(Zeroizing::new(vec![0; 32])),
            },
            Phase::Round2 { first, seed } => State::Round2 { first, seed },
            Phase::Round3 {
                first,
                second,
                queries,
                seed,
            } => State::Round3 {
                first,
                second,
                queries,
                seed,
            },
        };
        session
    }
}
fn indices(file: &mut impl Read, log_n: u32, count: u32) -> Check<Vec<u32>> {
    let len = usize::try_from(count).map_err(|_| internal())?;
    let mut out = reserve(len)?;
    let mask = 1u64
        .checked_shl(log_n)
        .and_then(|n| n.checked_sub(1))
        .ok_or_else(internal)?;
    let mut bytes = Zeroizing::new([0; 8]);
    for _ in 0..count {
        fill(file, bytes.as_mut())?;
        out.push(u32::try_from(u64::from_le_bytes(*bytes) & mask).map_err(|_| internal())?);
    }
    Ok(out)
}
pub(super) fn queries(
    layouts: &[weights::Layout; 10],
    required: &[usize],
    params: &SoundnessParams,
    advice_log_k: u8,
) -> Check<protocol::ColumnQueries> {
    let mut file = entropy()?;
    let advice = indices(
        &mut file,
        u32::from(advice_log_k) + 1,
        params.advice_queries,
    )?;
    let mut weights = reserve(required.len())?;
    for &i in required {
        let l = layouts.get(i).ok_or_else(internal)?;
        weights.push(protocol::MatrixQueries {
            matrix: u8::try_from(i).map_err(|_| internal())?,
            indices: indices(
                &mut file,
                l.log2_k().checked_add(2).ok_or_else(internal)?,
                params.weight_queries,
            )?,
        });
    }
    Ok(protocol::ColumnQueries { advice, weights })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn query_sampler_uses_all_low_bits_with_replacement_and_fails_closed_on_entropy_loss() {
        for bits in [0, 1, 4, 31] {
            let mut source = std::io::Cursor::new([255u8; 32]);
            let sampled = indices(&mut source, bits, 4).unwrap_or_else(|_| panic!("sample failed"));
            assert_eq!(sampled, vec![(1u32 << bits) - 1; 4]);
            assert_eq!(source.position(), 32);
        }
        assert!(matches!(
            indices(&mut std::io::empty(), 4, 1),
            Err(Failure::Inconclusive(VerifierError {
                code: ErrorCode::Io
            }))
        ));
    }

    #[cfg(feature = "test-params")]
    #[test]
    fn completed_recorded_session_is_consumed_and_nonzero_exact_bound_rejects() {
        let (claims, _, cfg) = super::super::tests::fixture();
        let proof = NiProof::open(
            include_bytes!("../../tests/fixtures/driver/live-messages.tnp"),
            &DecodeLimits::MESSAGE,
        )
        .unwrap();
        let mut session = LiveVerifier::new(claims, proof.round1.statement.clone(), cfg);
        let state = |second: &protocol::Round2Response| State::Round3 {
            first: proof.round1.to_tce().unwrap(),
            second: second.to_tce().unwrap(),
            queries: include_bytes!("../../tests/fixtures/driver/queries.tce").to_vec(),
            seed: *include_bytes!("../../tests/fixtures/driver/live-seed.bin"),
        };
        session.state = state(&proof.round2);
        assert!(matches!(
            session.on_round3(&proof.round3),
            Verdict::Accept(_)
        ));
        assert!(matches!(
            session.on_round3(&proof.round3),
            Verdict::Inconclusive(VerifierError {
                code: ErrorCode::ProtocolOrder
            })
        ));
        let mut bad = proof.round2.clone();
        bad.scoring.requests[0] = FixedBits(1);
        session.state = state(&bad);
        assert!(matches!(
            session.on_round3(&proof.round3),
            Verdict::Reject(RejectReason::UBoundExceeded {
                request_id: Some(_),
                uhat: FixedBits(1),
                bound: FixedBits(0)
            })
        ));
    }
}
