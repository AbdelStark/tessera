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
    deadline: Option<Instant>,
    decode: impl FnOnce(&mut tv_codec::Cursor<'_>) -> Result<T, tv_codec::DecodeError>,
) -> Check<T> {
    tv_codec::decode(bytes, &DecodeLimits::MESSAGE, |c| {
        if let Some(deadline) = deadline {
            c.set_deadline(deadline)?;
        }
        decode(c)
    })
    .map_err(|e| match e.code() {
        tv_codec::ErrorCode::Memory => memory(),
        tv_codec::ErrorCode::Time => fail(ErrorCode::Time),
        _ => internal(),
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
    /// Decode the next prover message under this session's original deadline.
    /// Reports the input check and consumes the session on malformed input or work expiry.
    /// An out-of-order message type leaves the current state unchanged.
    ///
    /// # Errors
    /// Protocol order/idle timeout, malformed input, memory failure, or operational Time.
    #[allow(
        clippy::result_large_err,
        reason = "the specified Live API returns Verdict by value"
    )]
    pub fn open_message_observed<T: protocol::ProverMessage>(
        &mut self,
        bytes: &[u8],
        limits: &DecodeLimits,
        observer: &mut dyn Observer,
    ) -> Result<T, Verdict> {
        let permitted = match &self.state {
            State::Committed { .. } => matches!(T::SCHEMA.schema_id, 0x0510 | 0x0502),
            State::Round2 { .. } => T::SCHEMA.schema_id == 0x0511,
            State::Round3 { .. } => T::SCHEMA.schema_id == 0x0513,
            _ => false,
        };
        let admission = if permitted {
            self.timed_out()
        } else {
            Err(fail(ErrorCode::ProtocolOrder))
        };
        let mut tracker = Tracker::new(observer);
        tracker.enter(CheckKind::Input);
        let result = admission.and_then(|()| {
            let len = u64::try_from(bytes.len()).map_err(|_| memory())?;
            let held = match &self.state {
                State::Committed { .. } => [32, 0, 0, 0],
                State::Round2 { first, .. } => [32, first.len(), 0, 0],
                State::Round3 {
                    first,
                    second,
                    queries,
                    ..
                } => [32, first.len(), second.len(), queries.len()],
                _ => [0; 4],
            };
            let planned = held.into_iter().try_fold(
                len.checked_mul(8).ok_or_else(memory)?,
                |total, bytes| {
                    total
                        .checked_add(u64::try_from(bytes).map_err(|_| memory())?)
                        .ok_or_else(memory)
                },
            )?;
            if planned > self.cfg.limits.max_memory_bytes {
                return Err(memory());
            }
            let limits = DecodeLimits {
                max_total_bytes: limits
                    .max_total_bytes
                    .min(self.cfg.limits.max_artifact_bytes),
                ..*limits
            };
            open_message::<T>(
                bytes,
                &limits,
                self.started
                    .and_then(|start| start.checked_add(self.cfg.limits.max_wall_time)),
            )
        });
        self.finish(&mut tracker, result).map_err(Into::into)
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
        self.finish(&mut tracker, result).map_err(Into::into)
    }
    fn finish<T>(&mut self, tracker: &mut Tracker<'_>, result: Check<T>) -> Check<T> {
        let result = conclude_with_deadline(tracker, result, self.started, self.cfg.limits);
        if result.as_ref().is_err_and(|error| {
            !matches!(error, Failure::Inconclusive(e) if e.code == ErrorCode::ProtocolOrder)
        }) {
            self.state = State::Done;
        }
        result
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
        self.finish(&mut tracker, result).map_err(Into::into)
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
        self.finish(&mut tracker, result).map_err(Into::into)
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
        let round1 = own_decode(
            &first,
            self.started
                .and_then(|start| start.checked_add(self.cfg.limits.max_wall_time)),
            protocol::Round1::decode,
        )?;
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
        into_verdict(self.finish(&mut tracker, result))
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
        let round1 = own_decode(
            &first,
            self.started
                .and_then(|start| start.checked_add(self.cfg.limits.max_wall_time)),
            protocol::Round1::decode,
        )?;
        let round2 = own_decode(
            &second,
            self.started
                .and_then(|start| start.checked_add(self.cfg.limits.max_wall_time)),
            protocol::Round2Response::decode,
        )?;
        let queries = own_decode(
            &queries,
            self.started
                .and_then(|start| start.checked_add(self.cfg.limits.max_wall_time)),
            protocol::ColumnQueries::decode,
        )?;
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
        session.started = Some(Instant::now());
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
    #[cfg(feature = "test-params")]
    #[derive(Debug)]
    struct TimeProbe<const SCHEMA: u16>;
    #[cfg(feature = "test-params")]
    impl<const SCHEMA: u16> protocol::ProverMessage for TimeProbe<SCHEMA> {
        const SCHEMA: tv_codec::Expected = tv_codec::Expected {
            schema_id: SCHEMA,
            version: 1,
        };
        fn open_sealed(_: &[u8], _: &DecodeLimits) -> Result<Self, tv_codec::DecodeError> {
            Err(tv_codec::DecodeError::new(tv_codec::ErrorCode::Internal, 0))
        }
        fn open_sealed_with_deadline(
            _: &[u8],
            _: &DecodeLimits,
            deadline: Option<Instant>,
        ) -> Result<Self, tv_codec::DecodeError> {
            assert!(
                deadline.is_some(),
                "the finite session deadline reaches decoding"
            );
            Err(tv_codec::DecodeError::new(tv_codec::ErrorCode::Time, 9))
        }
    }

    #[cfg(feature = "test-params")]
    #[test]
    fn live_message_decoding_uses_the_original_deadline_and_consumes_work_failure() {
        let (claims, proof, cfg) = super::super::tests::fixture();
        let mut session = LiveVerifier::at(
            claims,
            proof.round1.statement.clone(),
            cfg,
            Phase::Round1 {
                session_id: [0; 16],
            },
        );
        let wrong = session
            .open_message_observed::<TimeProbe<0x0513>>(&[], &DecodeLimits::MESSAGE, &mut ())
            .unwrap_err();
        assert!(matches!(
            wrong,
            Verdict::Inconclusive(VerifierError {
                code: ErrorCode::ProtocolOrder
            })
        ));
        assert!(matches!(session.state, State::Committed { .. }));
        let expired = session
            .open_message_observed::<TimeProbe<0x0510>>(&[], &DecodeLimits::MESSAGE, &mut ())
            .unwrap_err();
        assert!(matches!(
            expired,
            Verdict::Inconclusive(VerifierError {
                code: ErrorCode::Time
            })
        ));
        assert!(matches!(session.state, State::Done));
        let next = session
            .open_message_observed::<TimeProbe<0x0510>>(&[], &DecodeLimits::MESSAGE, &mut ())
            .unwrap_err();
        assert!(matches!(
            next,
            Verdict::Inconclusive(VerifierError {
                code: ErrorCode::ProtocolOrder
            })
        ));
        assert!(matches!(
            own_decode(&[], Some(Instant::now()), |_| Ok(())),
            Err(Failure::Inconclusive(VerifierError {
                code: ErrorCode::Time
            }))
        ));
        for artifact in [ArtifactKind(0x0520), ArtifactKind(0x0510)] {
            assert!(matches!(
                decode_error(
                    tv_codec::DecodeError::new(tv_codec::ErrorCode::Time, 7),
                    artifact
                ),
                Failure::Inconclusive(VerifierError {
                    code: ErrorCode::Time
                })
            ));
            assert!(
                matches!(decode_error(tv_codec::DecodeError::new(tv_codec::ErrorCode::Noncanonical, 7), artifact), Failure::Rejected(RejectReason::Malformed { artifact: found, code: ErrorCode::CodecNoncanonical }) if found == artifact)
            );
        }
    }

    #[cfg(feature = "test-params")]
    #[test]
    fn live_decode_accounts_for_retained_bytes_before_any_parser_runs() {
        let (claims, proof, mut cfg) = super::super::tests::fixture();
        cfg.limits.max_memory_bytes = 39;
        let mut session = LiveVerifier::at(
            claims,
            proof.round1.statement.clone(),
            cfg,
            Phase::Round1 {
                session_id: [0; 16],
            },
        );
        let refused = session
            .open_message_observed::<TimeProbe<0x0510>>(&[0], &DecodeLimits::MESSAGE, &mut ())
            .unwrap_err();
        assert!(matches!(
            refused,
            Verdict::Inconclusive(VerifierError {
                code: ErrorCode::Memory
            })
        ));
        assert!(matches!(session.state, State::Done));
    }

    #[cfg(feature = "test-params")]
    #[test]
    fn healthy_live_decode_and_idle_expiry_keep_their_distinct_outcomes() {
        let (claims, proof, cfg) = super::super::tests::fixture();
        let mut session = LiveVerifier::at(
            claims,
            proof.round1.statement.clone(),
            cfg,
            Phase::Round1 {
                session_id: [0; 16],
            },
        );
        let payload = proof.round1.to_tce().unwrap();
        let mut bytes = b"TSRA".to_vec();
        for field in [1u16, 0x0510, 1, 0] {
            bytes.extend(field.to_le_bytes());
        }
        bytes.extend((payload.len() as u64).to_le_bytes());
        bytes.extend(payload);
        let digest = hash(Domain::Envelope, &bytes);
        bytes.extend(digest);
        let decoded = session
            .open_message_observed::<protocol::Round1>(&bytes, &DecodeLimits::MESSAGE, &mut ())
            .unwrap_or_else(|_| panic!("valid recorded Round1 decodes"));
        assert!(decoded == proof.round1);
        assert!(matches!(session.state, State::Committed { .. }));
        session.cfg.limits.max_wall_time = Duration::ZERO;
        let idle = session
            .open_message_observed::<TimeProbe<0x0510>>(&[], &DecodeLimits::MESSAGE, &mut ())
            .unwrap_err();
        assert!(matches!(
            idle,
            Verdict::Inconclusive(VerifierError {
                code: ErrorCode::ProtocolTimeout
            })
        ));
        assert!(matches!(session.state, State::Done));
    }

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
    fn final_observer_work_cannot_release_live_queries_or_acceptance() {
        use crate::observe::CheckEvent;
        struct Delay(CheckKind);
        impl Observer for Delay {
            fn check_completed(&mut self, event: CheckEvent) {
                if event.kind == self.0 {
                    std::thread::sleep(Duration::from_millis(550));
                }
            }
        }
        let proof = NiProof::open(
            include_bytes!("../../tests/fixtures/driver/live-messages.tnp"),
            &DecodeLimits::MESSAGE,
        )
        .unwrap();
        let seed = *include_bytes!("../../tests/fixtures/driver/live-seed.bin");
        for round in [2, 3] {
            let (claims, _, mut cfg) = super::super::tests::fixture();
            cfg.limits.max_wall_time = Duration::from_millis(500);
            let phase = if round == 2 {
                Phase::Round2 {
                    first: proof.round1.to_tce().unwrap(),
                    seed,
                }
            } else {
                Phase::Round3 {
                    first: proof.round1.to_tce().unwrap(),
                    second: proof.round2.to_tce().unwrap(),
                    queries: include_bytes!("../../tests/fixtures/driver/queries.tce").to_vec(),
                    seed,
                }
            };
            let mut session = LiveVerifier::at(claims, proof.round1.statement.clone(), cfg, phase);
            let verdict = if round == 2 {
                session
                    .on_round2_observed(&proof.round2, &mut Delay(CheckKind::Queries))
                    .unwrap_err()
            } else {
                session.on_round3_observed(&proof.round3, &mut Delay(CheckKind::Complete))
            };
            assert!(matches!(
                verdict,
                Verdict::Inconclusive(VerifierError {
                    code: ErrorCode::Time
                })
            ));
            assert!(matches!(
                session.on_round3(&proof.round3),
                Verdict::Inconclusive(VerifierError {
                    code: ErrorCode::ProtocolOrder
                })
            ));
        }
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
        session.started = Some(Instant::now());
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
