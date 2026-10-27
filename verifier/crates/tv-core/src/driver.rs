// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Protocol/1 verification over trusted application inputs (RFC-0019).
//! The current runtime composes one exact-greedy request; unsupported predictors/batches
//! fail explicitly. Anchor signatures, scoring circuits and persistent ledgers are v0.4.
//!
//! A leaf stage in token groups (RFC-0054) runs in this order: `Round1.leaf_groups` within
//! `t = ⌈log2 n⌉` token bits and `n_ℓ = ⌈log2 L⌉` layer bits of the trusted statement and claim
//! list, else `Malformed { ERR-CODEC-LIMIT }` on Round1 (the `NiProof` in non-interactive mode)
//! as soon as it has decoded; the grouped layouts from public data; the counts the grouping fixes,
//! else `Malformed { ERR-CODEC-NONCANONICAL }` on the Round2Response (the `NiProof`), before any
//! transcript work; then [`leaf::grouped::Stage::verify`] and its atom splits in place of
//! RFC-0010's single batch and RFC-0045's split. Every later check reads the stage's claims
//! through [`leaf::split::LeafPoint`], so an ungrouped stage's are unchanged.

use crate::claims::ClaimList;
use crate::observe::{CheckKind, CheckOutcome, Observer, Tracker};
use crate::protocol::{Decoding, FixedBits, NiProof, PredictorSpec, PublicStatement};
use crate::verdict::{
    AcceptReport, ArtifactKind, CommitmentClass, ErrorCode, PowPhase, RejectReason, Verdict,
    VerifierError,
};
use crate::{
    advice, gadgets, leaf, model_openings, opening, params::ParamSet, protocol, runtime,
    soundness::SoundnessParams, stages, sumcheck, validation, weights,
};
use std::time::{Duration, Instant};
use tv_codec::{DecodeLimits, WeightCommitmentPublic};
use tv_field::Fp2;
use tv_hash::Digest;
use tv_hash::transcript::{Ascii, Mode, SessionContext, Transcript, TranscriptError};

/// Verifier resource policy; input limits apply before proof-dependent allocations.
#[derive(Clone, Copy, Debug)]
pub struct ResourceLimits {
    /// Maximum outer artifact bytes, including its envelope.
    pub max_artifact_bytes: u64,
    /// Budget for the conservative input/codeword/metadata work estimate.
    /// Full allocation accounting belongs to the v1 hardening work (#182).
    pub max_memory_bytes: u64,
    /// Cooperative deadline checked between phases, covering NI decoding or the whole Live session.
    pub max_wall_time: Duration,
}
impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_artifact_bytes: DecodeLimits::MESSAGE.max_total_bytes,
            max_memory_bytes: 64 << 30,
            max_wall_time: Duration::from_secs(300),
        }
    }
}
/// Application-owned policy and the trusted epoch's public weight artifact (RFC-0040).
#[derive(Clone, Debug)]
pub struct VerifierConfig {
    /// The exact accepted bundle, never supplied by the proof.
    pub params: ParamSet,
    /// Input/work/deadline limits.
    pub limits: ResourceLimits,
    /// Public unexplained-information threshold; exact proofs establish zero.
    pub accept_bound: FixedBits,
    /// Epoch artifact whose hash must equal the trusted statement's weight root.
    pub weights: WeightCommitmentPublic,
}

enum Failure {
    Rejected(RejectReason),
    Inconclusive(VerifierError),
}
impl From<Failure> for Verdict {
    fn from(error: Failure) -> Self {
        match error {
            Failure::Rejected(r) => Self::Reject(r),
            Failure::Inconclusive(e) => Self::Inconclusive(e),
        }
    }
}
type Check<T> = Result<T, Failure>;
fn fail(code: ErrorCode) -> Failure {
    Failure::Inconclusive(VerifierError { code })
}
fn memory() -> Failure {
    fail(ErrorCode::Memory)
}
fn internal() -> Failure {
    fail(ErrorCode::Internal)
}
fn reject(reason: RejectReason) -> Failure {
    Failure::Rejected(reason)
}
fn mismatch(field: validation::StatementField) -> Failure {
    reject(RejectReason::StatementMismatch { field })
}
fn malformed(artifact: ArtifactKind) -> Failure {
    reject(RejectReason::Malformed {
        artifact,
        code: ErrorCode::CodecNoncanonical,
    })
}
fn codec(code: tv_codec::ErrorCode) -> Failure {
    fail(code.into())
}
/// How the check running when `result` came back ended (05 §2, `check_completed`).
fn outcome<T>(result: &Check<T>) -> CheckOutcome {
    match result {
        Ok(_) => CheckOutcome::Pass,
        Err(Failure::Rejected(_)) => CheckOutcome::Reject,
        Err(Failure::Inconclusive(_)) => CheckOutcome::Inconclusive,
    }
}
/// Ends the running check with `result`'s outcome, then returns `result`.
fn conclude<T>(tracker: &mut Tracker<'_>, result: Check<T>) -> Check<T> {
    tracker.close(outcome(&result));
    result
}
/// Reporting can consume the remaining budget too; check before releasing a successful value.
fn conclude_with_deadline<T>(
    tracker: &mut Tracker<'_>,
    result: Check<T>,
    started: Option<Instant>,
    limits: ResourceLimits,
) -> Check<T> {
    let value = conclude(tracker, result)?;
    deadline(started.ok_or_else(internal)?, limits)?;
    Ok(value)
}
fn into_verdict(result: Check<AcceptReport>) -> Verdict {
    match result {
        Ok(report) => Verdict::Accept(report),
        Err(failure) => failure.into(),
    }
}
fn reserve<T>(n: usize) -> Check<Vec<T>> {
    let mut out = Vec::new();
    out.try_reserve_exact(n).map_err(|_| memory())?;
    Ok(out)
}
fn copy<T: Copy>(xs: &[T]) -> Check<Vec<T>> {
    let mut out = reserve(xs.len())?;
    out.extend_from_slice(xs);
    Ok(out)
}
fn ascii(s: &str) -> Check<Ascii<'_>> {
    Ascii::new(s).map_err(|_| internal())
}
fn weight_label(matrix: weights::Matrix, phase: &str) -> Check<String> {
    let parts = ["pcs/weights/", matrix.name(), "/", phase];
    let len = parts
        .iter()
        .try_fold(0usize, |n, s| n.checked_add(s.len()))
        .ok_or_else(memory)?;
    let mut out = String::new();
    out.try_reserve_exact(len).map_err(|_| memory())?;
    for part in parts {
        out.push_str(part);
    }
    Ok(out)
}
fn transcript(e: TranscriptError) -> Failure {
    if e == TranscriptError::Memory {
        memory()
    } else {
        internal()
    }
}
fn proof_of_work(
    t: Transcript,
    params: &SoundnessParams,
    phase: PowPhase,
    nonce: u64,
) -> Check<Transcript> {
    let (label, bits) = match phase {
        PowPhase::Seed => ("seed-pow", params.seed_pow),
        PowPhase::Queries => ("query-pow", params.query_pow),
    };
    t.check_grind(
        ascii(label)?,
        u8::try_from(bits).map_err(|_| internal())?,
        nonce,
    )
    .map_err(|_| reject(RejectReason::ProofOfWork { phase }))
}
fn point(t: &mut Transcript, label: &str, n: u32) -> Check<Vec<Fp2>> {
    let mut out = reserve(usize::try_from(n).map_err(|_| internal())?)?;
    for _ in 0..n {
        out.push(t.challenge_fp2(ascii(label)?).map_err(transcript)?);
    }
    Ok(out)
}
fn claims_error(e: validation::ClaimsError) -> Failure {
    match e {
        validation::ClaimsError::Decode(e) => codec(e.code()),
        validation::ClaimsError::Params { name } => reject(RejectReason::Params { name }),
        validation::ClaimsError::StatementMismatch { field } => mismatch(field),
        validation::ClaimsError::Memory => memory(),
        validation::ClaimsError::Internal => internal(),
    }
}
fn params_error(claims: &ClaimList) -> Failure {
    match claims.params.try_clone() {
        Some(name) => reject(RejectReason::Params { name }),
        None => memory(),
    }
}
fn runtime_error(e: runtime::Error, claims: &ClaimList) -> Failure {
    match e {
        runtime::Error::Memory => memory(),
        runtime::Error::Internal => internal(),
        runtime::Error::Params | runtime::Error::Targets { .. } => params_error(claims),
    }
}
fn gadget_error(e: gadgets::GadgetError) -> Failure {
    match e {
        gadgets::GadgetError::Rejected { gadget } => reject(RejectReason::Gadget { gadget }),
        gadgets::GadgetError::Memory => memory(),
        gadgets::GadgetError::Internal => internal(),
    }
}
fn model_error(e: model_openings::Error) -> Failure {
    if e == model_openings::Error::Memory {
        memory()
    } else {
        internal()
    }
}
fn opening_error(e: opening::OpeningError) -> Failure {
    use opening::OpeningError as E;
    let commitment = CommitmentClass::Advice;
    match e {
        E::Evaluation { .. } => reject(RejectReason::Evaluation { commitment }),
        E::MerklePath { column } => reject(RejectReason::MerklePath { commitment, column }),
        E::Proximity { column } => reject(RejectReason::Proximity { commitment, column }),
        E::Memory => memory(),
        _ => internal(),
    }
}
fn weight_error(e: weights::Error) -> Failure {
    use weights::Error as E;
    let commitment = CommitmentClass::Weight;
    match e {
        E::Usage => fail(ErrorCode::Usage),
        E::Layout => mismatch(validation::StatementField::Layout),
        E::Evaluation { .. } => reject(RejectReason::Evaluation { commitment }),
        E::MerklePath { column, .. } => reject(RejectReason::MerklePath { commitment, column }),
        E::Proximity { column, .. } => reject(RejectReason::Proximity { commitment, column }),
        E::Memory => memory(),
        E::Internal => internal(),
    }
}
fn stage_error(e: stages::StageError, artifact: ArtifactKind) -> Failure {
    if let Some(r) = e.rejection() {
        return reject(RejectReason::Sumcheck {
            stage: r.stage,
            round: r.round,
        });
    }
    if e.is_malformed() {
        return malformed(artifact);
    }
    match e {
        stages::StageError::Gadget { gadget } => reject(RejectReason::Gadget { gadget }),
        stages::StageError::Memory => memory(),
        stages::StageError::Sumcheck {
            error: sumcheck::SumcheckError::Transcript(e),
            ..
        } => transcript(e),
        _ => internal(),
    }
}
fn leaf_error(
    e: leaf::Error,
    tables: &[(tv_tables::TableId, u64)],
    artifact: ArtifactKind,
) -> Failure {
    match e {
        leaf::Error::LevelCount { .. } => malformed(artifact),
        leaf::Error::Lookup(e) => {
            if let Some(i) = e.rejected_instance() {
                return match tables.get(i) {
                    Some(&(table, _)) => reject(RejectReason::Lookup { table }),
                    None => internal(),
                };
            }
            if e.is_rejection() {
                return malformed(artifact);
            }
            if let crate::logup::LookupError::Transcript(e) = e {
                return transcript(e);
            }
            internal()
        }
        leaf::Error::Level { level, error } => match error.rejected_round() {
            Some(round) => reject(RejectReason::TreeLevel {
                group: 0,
                level,
                round,
            }),
            None => {
                if let sumcheck::SumcheckError::Transcript(e) = error {
                    transcript(e)
                } else {
                    internal()
                }
            }
        },
        leaf::Error::Memory => memory(),
        leaf::Error::Internal => internal(),
    }
}
fn split_error(e: leaf::split::Error, artifact: ArtifactKind) -> Failure {
    match e {
        leaf::split::Error::Count { .. } => malformed(artifact),
        leaf::split::Error::Mismatch => reject(RejectReason::Sumcheck { stage: 0, round: 0 }),
        leaf::split::Error::Memory => memory(),
        leaf::split::Error::Internal => internal(),
    }
}
/// A grouped stage's failure by its own mapping (RFC-0054 'Messages', 'Rejections'), `artifact`
/// being the one a `Malformed` count names.
fn grouped_error(e: leaf::grouped::Error, artifact: ArtifactKind) -> Failure {
    match e.verdict(artifact) {
        Verdict::Reject(reason) => reject(reason),
        Verdict::Inconclusive(error) => Failure::Inconclusive(error),
        Verdict::Accept(_) => internal(),
    }
}
/// A grouped stage's layout: a grouping beyond its bounds is the proof's, `Malformed {
/// ERR-CODEC-LIMIT }` on `artifact`, `Round1`'s (RFC-0054 'Messages'); anything else the
/// verifier's own.
fn layout_error(e: &leaf::groups::Error, artifact: ArtifactKind) -> Failure {
    if e.is_rejection() {
        return reject(RejectReason::Malformed {
            artifact,
            code: ErrorCode::CodecLimit,
        });
    }
    match e {
        leaf::groups::Error::Memory
        | leaf::groups::Error::Model(validation::ClaimsError::Memory) => memory(),
        _ => internal(),
    }
}
/// The extent `leaf_groups` must keep (RFC-0054 'Messages'), from the trusted inputs alone, so
/// that `Round1` is judged as soon as it has decoded: `t = ⌈log2 n⌉` for the statement's `n`
/// tokens (0 for none), and the claim list's layers (one at least: a claim list of none fails
/// its own validation, RFC-0007).
fn grouping_extent(claims: &ClaimList, statement: &PublicStatement) -> leaf::groups::Extent {
    let tokens = statement.batch.requests.iter().fold(0u64, |n, r| {
        n.saturating_add(u64::from(r.prompt_len))
            .saturating_add(u64::from(r.output_len))
    });
    leaf::groups::Extent {
        token_bits: u64::BITS.saturating_sub(tokens.saturating_sub(1).leading_zeros()),
        layers: claims.arch.n_layers.max(1),
    }
}
fn deadline(start: Instant, limits: ResourceLimits) -> Check<()> {
    if start.elapsed() >= limits.max_wall_time {
        Err(fail(ErrorCode::Time))
    } else {
        Ok(())
    }
}
fn matrix_order(ids: impl Iterator<Item = u8>) -> bool {
    let mut prev = None;
    for id in ids {
        if id >= 10 || prev.is_some_and(|p| p >= id) {
            return false;
        }
        prev = Some(id);
    }
    true
}
fn mode_allowed(params: ParamSet, mode: Mode) -> bool {
    match params {
        ParamSet::Live64 => mode == Mode::Live,
        ParamSet::Ni100 => mode == Mode::NonInteractive,
        #[cfg(feature = "test-params")]
        ParamSet::TestSmall => true,
    }
}
fn flat_rows(packet: &protocol::WeightCombinedRows) -> Check<Vec<opening::CombinedRow>> {
    let n = packet.evaluations.len().checked_add(1).ok_or_else(memory)?;
    let mut out = reserve(n)?;
    for row in packet.evaluations.iter().chain([&packet.proximity]) {
        out.push(opening::CombinedRow {
            u: copy(&row.u)?,
            mask: copy(&row.mask)?,
        });
    }
    Ok(out)
}

/// Open one of the prover's live messages (RFC-0043 'Malformed'): a decoding failure that is the
/// input's fault is `Reject(Malformed)` under the message's own schema, any other the
/// verifier's `Inconclusive`, as [`verify_non_interactive_bytes`] judges an `NiProof`.
///
/// # Errors
/// The verdict that ends the session.
#[allow(
    clippy::result_large_err,
    reason = "the specified Live API returns Verdict; reports stay by value without an infallible allocation"
)]
pub fn open_live<T: protocol::ProverMessage>(
    bytes: &[u8],
    limits: &DecodeLimits,
) -> Result<T, Verdict> {
    open_live_observed(bytes, limits, &mut ())
}
/// [`open_live`], reporting its decoding to `observer` as step 1, `input` (05 §2,
/// [`observe`](crate::observe)).
///
/// # Errors
/// The verdict that ends the session, as [`open_live`].
#[allow(
    clippy::result_large_err,
    reason = "the specified Live API returns Verdict; reports stay by value without an infallible allocation"
)]
pub fn open_live_observed<T: protocol::ProverMessage>(
    bytes: &[u8],
    limits: &DecodeLimits,
    observer: &mut dyn Observer,
) -> Result<T, Verdict> {
    let mut tracker = Tracker::new(observer);
    tracker.enter(CheckKind::Input);
    let opened = T::open_sealed(bytes, limits).map_err(|e| {
        if e.code().is_input_fault() {
            reject(RejectReason::Malformed {
                artifact: ArtifactKind(T::SCHEMA.schema_id),
                code: e.code().into(),
            })
        } else {
            codec(e.code())
        }
    });
    conclude(&mut tracker, opened).map_err(Into::into)
}
/// Verify a canonical NI envelope under the caller's limits, then every semantic predicate.
/// Decoding a prover artifact fails as Malformed; reader failures remain Inconclusive.
pub fn verify_non_interactive_bytes(
    claims: &ClaimList,
    statement: &PublicStatement,
    bytes: &[u8],
    cfg: &VerifierConfig,
) -> Verdict {
    verify_non_interactive_bytes_observed(claims, statement, bytes, cfg, &mut ())
}
/// [`verify_non_interactive_bytes`], reporting each check to `observer` as it completes (05 §2,
/// [`observe`](crate::observe)): the envelope and its limits are step 1's. The verdict is the
/// unobserved one.
pub fn verify_non_interactive_bytes_observed(
    claims: &ClaimList,
    statement: &PublicStatement,
    bytes: &[u8],
    cfg: &VerifierConfig,
    observer: &mut dyn Observer,
) -> Verdict {
    let mut tracker = Tracker::new(observer);
    tracker.enter(CheckKind::Input);
    let started = Instant::now();
    let result = verify_ni_bytes(claims, statement, bytes, cfg, &mut tracker, started);
    into_verdict(conclude_with_deadline(
        &mut tracker,
        result,
        Some(started),
        cfg.limits,
    ))
}
fn verify_ni_bytes(
    claims: &ClaimList,
    statement: &PublicStatement,
    bytes: &[u8],
    cfg: &VerifierConfig,
    tracker: &mut Tracker<'_>,
    started: Instant,
) -> Check<AcceptReport> {
    let limits = DecodeLimits {
        max_total_bytes: cfg.limits.max_artifact_bytes,
        ..DecodeLimits::MESSAGE
    };
    if cfg.limits.max_memory_bytes == 0 {
        return Err(memory());
    }
    let Ok(len) = u64::try_from(bytes.len()) else {
        return Err(memory());
    };
    if len > limits.max_total_bytes {
        return Err(reject(RejectReason::Malformed {
            artifact: ArtifactKind(0x0520),
            code: ErrorCode::CodecLimit,
        }));
    }
    // Reserve an input-sized planning allowance before decoding; full accounting is #182.
    if len
        .checked_mul(8)
        .is_none_or(|n| n > cfg.limits.max_memory_bytes)
    {
        return Err(memory());
    }
    deadline(started, cfg.limits)?;
    let proof = match NiProof::open(bytes, &limits) {
        Ok(proof) => proof,
        Err(e) if e.code().is_input_fault() => {
            return Err(reject(RejectReason::Malformed {
                artifact: ArtifactKind(0x0520),
                code: e.code().into(),
            }));
        }
        Err(e) => return Err(codec(e.code())),
    };
    deadline(started, cfg.limits)?;
    verify_ni(claims, statement, &proof, cfg, tracker, started)
}
/// Verify protocol/1 against application-owned claims, expected statement and weight artifact.
/// A test-small acceptance certifies nothing. No anchor, ledger or zero-knowledge claim is made.
pub fn verify_non_interactive(
    claims: &ClaimList,
    statement: &PublicStatement,
    proof: &NiProof,
    cfg: &VerifierConfig,
) -> Verdict {
    verify_non_interactive_observed(claims, statement, proof, cfg, &mut ())
}
/// [`verify_non_interactive`], reporting each check of RFC-0019 'Check order' to `observer` as
/// it completes, with its outcome and measured time (05 §2, [`observe`](crate::observe)). The
/// verdict is the unobserved one, and so is its report: per-check durations stay null there.
pub fn verify_non_interactive_observed(
    claims: &ClaimList,
    statement: &PublicStatement,
    proof: &NiProof,
    cfg: &VerifierConfig,
    observer: &mut dyn Observer,
) -> Verdict {
    let mut tracker = Tracker::new(observer);
    tracker.enter(CheckKind::Input);
    let started = Instant::now();
    let result = verify_ni(claims, statement, proof, cfg, &mut tracker, started);
    into_verdict(conclude_with_deadline(
        &mut tracker,
        result,
        Some(started),
        cfg.limits,
    ))
}
mod live;
pub use live::LiveVerifier;

#[cfg(all(test, feature = "test-params"))]
mod corpus;
#[cfg(all(test, feature = "test-params"))]
mod tests;

#[derive(Clone, Copy)]
enum Session<'a> {
    Initial,
    Ni {
        seed_nonce: u64,
        query_nonce: u64,
    },
    Live {
        seed: &'a Digest,
        queries: Option<&'a protocol::ColumnQueries>,
    },
}
impl Session<'_> {
    fn mode(self) -> Mode {
        if matches!(self, Self::Ni { .. }) {
            Mode::NonInteractive
        } else {
            Mode::Live
        }
    }
    fn artifact(self) -> ArtifactKind {
        ArtifactKind(match self {
            Self::Initial => 0x0510,
            Self::Ni { .. } => 0x0520,
            Self::Live { queries: None, .. } => 0x0511,
            Self::Live {
                queries: Some(_), ..
            } => 0x0513,
        })
    }
    /// The artifact a `Round1` fault names: `Round1` in live mode, `NiProof` otherwise (RFC-0054
    /// 'Messages').
    fn round1_artifact(self) -> ArtifactKind {
        ArtifactKind(if matches!(self, Self::Ni { .. }) {
            0x0520
        } else {
            0x0510
        })
    }
}
struct Frame<'a> {
    round1: &'a protocol::Round1,
    round2: &'a protocol::Round2Response,
    round3: &'a protocol::Round3Response,
}
#[allow(
    clippy::large_enum_variant,
    reason = "return reports by value without an infallible box allocation"
)]
enum Outcome {
    Ready,
    Queries(protocol::ColumnQueries),
    Accepted(AcceptReport),
}
const EMPTY_ROUND2: protocol::Round2Response = protocol::Round2Response {
    logup: protocol::LogUpProof {
        roots: Vec::new(),
        values: Vec::new(),
    },
    trees: protocol::TreeProof {
        levels: Vec::new(),
        groups: Vec::new(),
        table_roots: Vec::new(),
        merge: sumcheck::SumcheckProof {
            rounds: Vec::new(),
            final_evals: Vec::new(),
        },
        splits: Vec::new(),
    },
    stages: Vec::new(),
    advice_combined: Vec::new(),
    weight_combined: Vec::new(),
    scoring: protocol::ScoringClaims {
        requests: Vec::new(),
    },
};
const EMPTY_ROUND3: protocol::Round3Response = protocol::Round3Response {
    advice_columns: Vec::new(),
    advice_proof: protocol::MultiProof {
        siblings: Vec::new(),
    },
    weight_columns: Vec::new(),
};
fn verify_ni(
    claims: &ClaimList,
    expected: &PublicStatement,
    proof: &NiProof,
    cfg: &VerifierConfig,
    tracker: &mut Tracker<'_>,
    started: Instant,
) -> Check<AcceptReport> {
    match verify_unit_tracked(
        claims,
        expected,
        Frame {
            round1: &proof.round1,
            round2: &proof.round2,
            round3: &proof.round3,
        },
        cfg,
        Session::Ni {
            seed_nonce: proof.seed_nonce,
            query_nonce: proof.query_nonce,
        },
        tracker,
        started,
    )? {
        Outcome::Accepted(report) => Ok(report),
        _ => Err(internal()),
    }
}
/// [`verify_unit_tracked`] unobserved, as the recorded-session tests replay it.
#[cfg(all(test, feature = "test-params"))]
fn verify_unit(
    claims: &ClaimList,
    expected: &PublicStatement,
    proof: Frame<'_>,
    cfg: &VerifierConfig,
    session: Session<'_>,
) -> Check<Outcome> {
    let mut ignored = ();
    let mut tracker = Tracker::new(&mut ignored);
    verify_unit_tracked(
        claims,
        expected,
        proof,
        cfg,
        session,
        &mut tracker,
        Instant::now(),
    )
}
/// One proof unit's checks in RFC-0019's order, `tracker` entering each check where its work
/// starts (the table of [`observe`](crate::observe)). It leaves the last check it entered running:
/// the caller ends it with the call's outcome.
fn verify_unit_tracked(
    claims: &ClaimList,
    expected: &PublicStatement,
    proof: Frame<'_>,
    cfg: &VerifierConfig,
    session: Session<'_>,
    tracker: &mut Tracker<'_>,
    started: Instant,
) -> Check<Outcome> {
    tracker.enter(CheckKind::Input);
    let artifact = session.artifact();
    let mode = session.mode();
    let first_bytes = protocol::encoded_len(proof.round1).map_err(codec)?;
    let second_bytes = protocol::encoded_len(proof.round2).map_err(codec)?;
    let third_bytes = protocol::encoded_len(proof.round3).map_err(codec)?;
    let total = first_bytes
        .checked_add(second_bytes)
        .and_then(|n| n.checked_add(third_bytes))
        .ok_or_else(memory)?;
    let bytes = match session {
        Session::Initial => first_bytes,
        Session::Ni { .. } => total.checked_add(16).ok_or_else(memory)?,
        Session::Live { queries: None, .. } => second_bytes,
        Session::Live {
            queries: Some(_), ..
        } => third_bytes,
    };
    let envelope = tv_codec::HEADER_LEN + tv_codec::DIGEST_LEN;
    if bytes.checked_add(envelope).ok_or_else(memory)? > cfg.limits.max_artifact_bytes {
        return Err(reject(RejectReason::Malformed {
            artifact,
            code: ErrorCode::CodecLimit,
        }));
    }
    if !matrix_order(proof.round2.weight_combined.iter().map(|m| m.matrix))
        || !matrix_order(proof.round3.weight_columns.iter().map(|m| m.matrix))
    {
        return Err(malformed(artifact));
    }
    // RFC-0054 'Messages': a grouping beyond the statement's token axis or the template's layer
    // index is `Malformed { ERR-CODEC-LIMIT }` on Round1, judged as soon as it has decoded; the
    // decoder judges `g_T + g_L ≤ 14` itself. This driver is `tessera-protocol/1`'s, which admits
    // groups (`tessera-protocol/2` would refuse any). An ungrouped stage has no groups, no table
    // roots and an empty merge: lengths the grouping fixes, judged after it.
    let groups = proof.round1.leaf_groups;
    if groups != protocol::LeafGroups::NONE {
        leaf::groups::check_bounds(groups, grouping_extent(claims, expected))
            .map_err(|e| layout_error(&e, session.round1_artifact()))?;
    } else {
        let trees = &proof.round2.trees;
        if !trees.groups.is_empty()
            || !trees.table_roots.is_empty()
            || !trees.merge.rounds.is_empty()
            || !trees.merge.final_evals.is_empty()
        {
            return Err(malformed(artifact));
        }
    }
    // shortcut: conservative work planning; replace with complete allocation accounting in #182.
    let memory_estimate = total
        .checked_mul(8)
        .and_then(|n| n.checked_add(u64::from(claims.arch.n_layers).checked_mul(1 << 20)?))
        .and_then(|n| {
            n.checked_add(
                1u64.checked_shl(u32::from(claims.layout.advice.log2_k) + 1)?
                    .checked_mul(64)?,
            )
        })
        .ok_or_else(memory)?;
    if memory_estimate > cfg.limits.max_memory_bytes {
        return Err(memory());
    }
    deadline(started, cfg.limits)?;
    tracker.enter(CheckKind::Params);
    if ParamSet::accept(&claims.params) != Some(cfg.params) || !mode_allowed(cfg.params, mode) {
        return Err(params_error(claims));
    }
    tracker.enter(CheckKind::Claims);
    let validated = validation::validate(claims).map_err(claims_error)?;
    deadline(started, cfg.limits)?;
    tracker.enter(CheckKind::Statement);
    // RFC-0040: inconsistent own root/epoch is Usage before judging statement fields.
    let public = weights::check_public(
        claims,
        &cfg.weights,
        &expected.weights_root,
        expected.weights_epoch,
    );
    if public == Err(weights::Error::Usage) {
        return Err(fail(ErrorCode::Usage));
    }
    if expected.claim_list_digest != claims.digest() {
        return Err(fail(ErrorCode::Usage));
    }
    if proof.round1.protocol_version != claims.protocol_version {
        return Err(fail(ErrorCode::ProtocolVersion));
    }
    let actual = &proof.round1.statement;
    use validation::StatementField as F;
    for (different, field) in [
        (
            actual.claim_list_digest != expected.claim_list_digest,
            F::ClaimListDigest,
        ),
        (actual.weights_root != expected.weights_root, F::WeightsRoot),
        (
            actual.weights_epoch != expected.weights_epoch,
            F::WeightsEpoch,
        ),
        (
            actual.manifest_digest != expected.manifest_digest,
            F::ManifestDigest,
        ),
        (actual.batch != expected.batch, F::Batch),
        (actual.predictor != expected.predictor, F::Predictor),
    ] {
        if different {
            return Err(mismatch(field));
        }
    }
    public.map_err(weight_error)?;
    let [request] = expected.batch.requests.as_slice() else {
        return Err(fail(ErrorCode::Usage));
    };
    let n = request
        .prompt_len
        .checked_add(request.output_len)
        .ok_or_else(|| mismatch(F::Batch))?;
    if request.prompt_len == 0 || n > claims.arch.max_context {
        return Err(mismatch(F::Batch));
    }
    if expected.predictor
        != (PredictorSpec::Exact {
            decoding: Decoding::Greedy,
        })
    {
        return Err(fail(ErrorCode::Usage));
    }
    if !matches!(session, Session::Initial)
        && proof.round2.scoring.requests.len() != expected.batch.requests.len()
    {
        return Err(malformed(artifact));
    }
    let graph = crate::lowering::lower(&validation::check_public(claims).map_err(claims_error)?, n)
        .map_err(|e| {
            if e == crate::lowering::LowerError::Memory {
                memory()
            } else {
                internal()
            }
        })?;
    let runtime = gadgets::runtime::Plan::derive(
        &graph,
        claims.gadget_policy,
        request.prompt_len,
        0,
        validated.tables(),
    )
    .map_err(claims_error)?;
    let layout = runtime
        .advice_layout(advice::Config::new(
            &claims.arch,
            claims.gadget_policy,
            claims.layout.advice.log2_k,
            claims.layout.advice.segments,
        ))
        .map_err(claims_error)?;
    let statement_digest = expected.digest();
    let commitment = advice::opening::Commitment::bind(
        &layout,
        claims.layout.advice,
        &statement_digest,
        0,
        proof.round1.advice.root,
        proof.round1.advice.layout_digest,
    )
    .map_err(|e| match e {
        validation::ClaimsError::StatementMismatch { .. } => mismatch(F::LayoutDigest),
        other => claims_error(other),
    })?;
    let wl = weights::derive(claims, cfg.weights.epoch).map_err(|e| {
        if e == weights::LayoutError::Memory {
            memory()
        } else {
            internal()
        }
    })?;
    let leaf_plan = runtime.leaf_schedule().map_err(claims_error)?;
    let native = gadgets::model::Plan::derive_runtime(&runtime).map_err(gadget_error)?;
    let splits = leaf::split::Layout::new(&runtime.split_sites().map_err(claims_error)?)
        .map_err(|e| split_error(e, artifact))?;
    // RFC-0054: a grouped stage's layouts, from public data alone, over the same sites, forests
    // and table trees as the ungrouped stage's.
    let grouped = if groups == protocol::LeafGroups::NONE {
        None
    } else {
        Some(
            leaf::groups::Grouped::derive_runtime(&graph, &runtime, groups)
                .map_err(|e| layout_error(&e, session.round1_artifact()))?,
        )
    };
    // RFC-0054 'A proof's budget': the budget statement carries the proof's own grouping, its
    // sizes from these layouts alone.
    let meta = runtime::Geometry::derive(claims, &runtime, grouped.as_ref())
        .map_err(|e| runtime_error(e, claims))?;
    let mut tables = reserve(leaf_plan.lookups().len())?;
    for tree in leaf_plan.schedule().trees() {
        if let leaf::Tree::Table { instance, height } = *tree {
            let &(table, _) = leaf_plan.lookups().get(instance).ok_or_else(internal)?;
            tables.push(leaf::grouped::TableTree { table, height });
        }
    }
    let stage = grouped
        .as_ref()
        .map(|g| leaf::grouped::Stage::new(g, &tables))
        .transpose()
        .map_err(|e| grouped_error(e, artifact))?;
    deadline(started, cfg.limits)?;
    if matches!(session, Session::Initial) {
        return Ok(Outcome::Ready);
    }
    // Step 5 reserves nothing in `tessera-protocol/1`, whose report lists it as not applicable.
    tracker.not_applicable(CheckKind::WeightReservation);
    tracker.enter(CheckKind::Transcript);
    let grouped_proof = leaf::grouped::Proof {
        logup: &proof.round2.logup,
        trees: &proof.round2.trees,
    };
    // RFC-0054 'Counts': the grouping fixes these lengths, judged before any transcript work.
    if let Some(stage) = &stage {
        stage
            .check_counts(grouped_proof)
            .map_err(|e| grouped_error(e, artifact))?;
    }
    let parameters = SoundnessParams::try_from_param_set(cfg.params).map_err(|_| memory())?;
    let mut t = Transcript::new(&SessionContext {
        protocol: ascii("tessera-protocol/1")?,
        mode,
        params: tv_hash::transcript::ParamSetId(ascii(cfg.params.name())?),
        statement_digest,
        session_id: proof.round1.session_id,
    });
    t.absorb(ascii("round1")?, &proof.round1.to_tce().map_err(codec)?);
    match session {
        Session::Ni { seed_nonce, .. } => {
            t = proof_of_work(t, &parameters, PowPhase::Seed, seed_nonce)?;
            let _seed = t.squeeze(ascii("seed")?);
        }
        Session::Live { seed, .. } => t.absorb(ascii("seed")?, seed),
        Session::Initial => return Err(internal()),
    }
    let mut xi: [Vec<Fp2>; 10] = core::array::from_fn(|_| Vec::new());
    for (i, l) in wl.iter().enumerate() {
        if l.committed_rows() > 0 {
            let label = weight_label(l.matrix(), "xi")?;
            *xi.get_mut(i).ok_or_else(internal)? = point(&mut t, &label, l.proximity_vars())?;
        }
    }
    tracker.enter(CheckKind::Leaf);
    let mut table_error = None;
    let mut evaluate = |id: tv_tables::TableId, b: Fp2, p: &[Fp2]| {
        let table = runtime.table(id)?;
        match table.eval_mle(p, b) {
            Ok(v) => Some(v),
            Err(e) => {
                table_error = Some(e);
                None
            }
        }
    };
    let row_zero = |id| {
        let table = runtime.table(id)?;
        let (k, v) = table.row(0)?;
        Some((
            k,
            (!matches!(table.spec().kind, tv_tables::TableKind::Range { .. })).then_some(v),
        ))
    };
    // The leaf stage's claims, wherever it leaves them (`LeafPoint`): RFC-0010's single batch and
    // RFC-0045's split, or RFC-0054's groups, table batch, merge and atom splits.
    let bound = match &stage {
        None => {
            let checked = leaf_plan
                .schedule()
                .verify(
                    &mut t,
                    leaf::Proof {
                        roots: &proof.round2.logup.roots,
                        levels: &proof.round2.trees.levels,
                        values: &proof.round2.logup.values,
                    },
                    |i, p| leaf_plan.forest_root(i, p),
                    |i, b, p| evaluate(leaf_plan.lookups().get(i)?.0, b, p),
                )
                .map_err(|e| {
                    if table_error == Some(tv_tables::TableError::Memory) {
                        memory()
                    } else {
                        leaf_error(e, leaf_plan.lookups(), artifact)
                    }
                })?;
            splits
                .bind(
                    &mut t,
                    checked,
                    &proof.round2.trees.splits,
                    native.schedule().leaf_splits(),
                    row_zero,
                )
                .map_err(|e| split_error(e, artifact))?
        }
        Some(stage) => {
            let merged = stage
                .verify(
                    &mut t,
                    grouped_proof,
                    |i, p| leaf_plan.forest_root(i, p),
                    &mut evaluate,
                    row_zero,
                )
                .map_err(|e| {
                    if table_error == Some(tv_tables::TableError::Memory) {
                        memory()
                    } else {
                        grouped_error(e, artifact)
                    }
                })?;
            merged
                .bind(
                    &mut t,
                    &proof.round2.trees.splits,
                    native.schedule().leaf_splits(),
                )
                .map_err(|e| grouped_error(e, artifact))?
        }
    };
    tracker.enter(CheckKind::Stages);
    let prepared = native.bind_runtime(&bound).map_err(gadget_error)?;
    let rope = runtime
        .catalog()
        .iter()
        .find_map(|s| runtime.table(s.id)?.rope())
        .ok_or_else(internal)?;
    deadline(started, cfg.limits)?;
    let reduced = prepared
        .verify(&mut t, rope, &proof.round2.stages)
        .map_err(|e| stage_error(e, artifact))?;
    tracker.enter(CheckKind::Evaluations);
    let originals = model_openings::Claims::derive(&graph, claims, &layout, &wl, reduced)
        .map_err(model_error)?;
    if originals.external().is_some() {
        return Err(internal());
    }
    deadline(started, cfg.limits)?;
    let gamma = t
        .challenge_fp2(ascii("pcs/advice/gamma")?)
        .map_err(transcript)?;
    let adv_xi = point(
        &mut t,
        "pcs/advice/xi",
        layout
            .nu()
            .checked_sub(u32::from(claims.layout.advice.log2_k))
            .ok_or_else(internal)?,
    )?;
    let mut weight_claims: [Vec<weights::Claim<'_>>; 10] = core::array::from_fn(|_| Vec::new());
    let mut required = reserve(10)?;
    for matrix in originals.matrices() {
        let i = usize::from(matrix.tag());
        let l = wl.get(i).ok_or_else(internal)?;
        let gamma = if l.committed_rows() > 0 {
            required.push(i);
            let label = weight_label(matrix, "gamma")?;
            t.challenge_fp2(ascii(&label)?).map_err(transcript)?
        } else {
            Fp2::ONE
        };
        *weight_claims.get_mut(i).ok_or_else(internal)? = originals
            .weight_claims(matrix, gamma)
            .map_err(model_error)?;
    }
    if proof.round2.weight_combined.len() != required.len()
        || proof
            .round2
            .weight_combined
            .iter()
            .zip(&required)
            .any(|(p, &i)| usize::from(p.matrix) != i)
    {
        return Err(malformed(artifact));
    }
    let views = originals.advice_claims().map_err(model_error)?;
    let opening = commitment
        .prepare(&views, gamma, &adv_xi)
        .map_err(opening_error)?;
    let budget = meta
        .check_budget(cfg.params, mode, &opening, &originals, &wl)
        .map_err(|e| runtime_error(e, claims))?;
    let advice_evaluated = opening
        .check_evaluations(&proof.round2.advice_combined)
        .map_err(opening_error)?;
    t.absorb(
        ascii("pcs/advice/combined")?,
        &protocol::encode(proof.round2.advice_combined.as_slice()).map_err(codec)?,
    );
    let mut rows: [Vec<opening::CombinedRow>; 10] = core::array::from_fn(|_| Vec::new());
    for packet in &proof.round2.weight_combined {
        let i = usize::from(packet.matrix);
        *rows.get_mut(i).ok_or_else(internal)? = flat_rows(packet)?;
    }
    let mut evaluated = reserve(10)?;
    for (i, l) in wl.iter().enumerate() {
        let original = weight_claims.get(i).ok_or_else(internal)?;
        if original.is_empty() {
            continue;
        }
        let result = weights::check_evaluations(
            weights::Opening {
                layout: l,
                root: *cfg.weights.roots.get(i).ok_or_else(internal)?,
                claims: original,
                xi_point: xi.get(i).ok_or_else(internal)?,
            },
            rows.get(i).ok_or_else(internal)?,
        )
        .map_err(weight_error)?;
        evaluated.push((i, result));
        if l.committed_rows() > 0 {
            let packet = proof
                .round2
                .weight_combined
                .iter()
                .find(|p| usize::from(p.matrix) == i)
                .ok_or_else(internal)?;
            let label = weight_label(l.matrix(), "combined")?;
            t.absorb(ascii(&label)?, &protocol::encode(packet).map_err(codec)?);
        }
    }
    t.absorb(
        ascii("scoring")?,
        &protocol::encode(&proof.round2.scoring).map_err(codec)?,
    );
    deadline(started, cfg.limits)?;
    tracker.enter(CheckKind::Queries);
    let generated = match session {
        Session::Ni { query_nonce, .. } => {
            t = proof_of_work(t, &parameters, PowPhase::Queries, query_nonce)?;
            let aq = t
                .challenge_indices(
                    ascii("pcs/advice/queries")?,
                    u32::from(claims.layout.advice.log2_k) + 1,
                    parameters.advice_queries,
                )
                .map_err(transcript)?;
            let mut advice = reserve(aq.len())?;
            for q in aq {
                advice.push(u32::try_from(q).map_err(|_| internal())?);
            }
            let mut weights = reserve(required.len())?;
            for &i in &required {
                let l = wl.get(i).ok_or_else(internal)?;
                let label = weight_label(l.matrix(), "queries")?;
                let raw = t
                    .challenge_indices(
                        ascii(&label)?,
                        l.log2_k().checked_add(2).ok_or_else(internal)?,
                        parameters.weight_queries,
                    )
                    .map_err(transcript)?;
                let mut indices = reserve(raw.len())?;
                for q in raw {
                    indices.push(u32::try_from(q).map_err(|_| internal())?);
                }
                weights.push(protocol::MatrixQueries {
                    matrix: u8::try_from(i).map_err(|_| internal())?,
                    indices,
                });
            }
            Some(protocol::ColumnQueries { advice, weights })
        }
        Session::Live { queries: None, .. } => Some(live::queries(
            &wl,
            &required,
            &parameters,
            claims.layout.advice.log2_k,
        )?),
        Session::Live {
            queries: Some(_), ..
        } => None,
        Session::Initial => return Err(internal()),
    };
    let packet = match session {
        Session::Live {
            queries: Some(q), ..
        } => q,
        _ => generated.as_ref().ok_or_else(internal)?,
    };
    if mode == Mode::Live {
        t.absorb(ascii("queries")?, &protocol::encode(packet).map_err(codec)?);
        if matches!(session, Session::Live { queries: None, .. }) {
            return Ok(Outcome::Queries(generated.ok_or_else(internal)?));
        }
    }
    check_queries(
        packet,
        &wl,
        &required,
        &parameters,
        claims.layout.advice.log2_k,
    )?;
    tracker.enter(CheckKind::Columns);
    let mut advice_queries = reserve(packet.advice.len())?;
    advice_queries.extend(packet.advice.iter().copied().map(u64::from));
    let mut queries: [Vec<u64>; 10] = core::array::from_fn(|_| Vec::new());
    for p in &packet.weights {
        let mut values = reserve(p.indices.len())?;
        values.extend(p.indices.iter().copied().map(u64::from));
        *queries
            .get_mut(usize::from(p.matrix))
            .ok_or_else(internal)? = values;
    }
    if proof.round3.weight_columns.len() != required.len()
        || proof
            .round3
            .weight_columns
            .iter()
            .zip(&required)
            .any(|(p, &i)| usize::from(p.matrix) != i)
    {
        return Err(malformed(artifact));
    }
    advice_evaluated
        .check_columns(
            &advice_queries,
            &proof.round3.advice_columns,
            &proof.round3.advice_proof.siblings,
        )
        .map_err(opening_error)?;
    for (i, result) in evaluated {
        if let Some(packet) = proof
            .round3
            .weight_columns
            .iter()
            .find(|p| usize::from(p.matrix) == i)
        {
            result
                .check_columns(
                    queries.get(i).ok_or_else(internal)?,
                    &packet.columns,
                    &packet.proof.siblings,
                )
                .map_err(weight_error)?;
        } else {
            result.check_columns(&[], &[], &[]).map_err(weight_error)?;
        }
    }
    deadline(started, cfg.limits)?;
    tracker.enter(CheckKind::ExactBound);
    for (request, &uhat) in expected
        .batch
        .requests
        .iter()
        .zip(&proof.round2.scoring.requests)
    {
        if uhat.0 != 0 {
            return Err(reject(RejectReason::UBoundExceeded {
                request_id: Some(request.request_id),
                uhat,
                bound: FixedBits(0),
            }));
        }
    }
    tracker.enter(CheckKind::Complete);
    let mut requests = reserve(expected.batch.requests.len())?;
    requests.extend(
        expected
            .batch
            .requests
            .iter()
            .map(|r| (r.request_id, FixedBits(0))),
    );
    deadline(started, cfg.limits)?;
    Ok(Outcome::Accepted(AcceptReport {
        params: cfg.params,
        statement_digest,
        claim_list_digest: expected.claim_list_digest,
        weights_root: expected.weights_root,
        manifest_digest: expected.manifest_digest,
        requests,
        batch_uhat: FixedBits(0),
        budget,
    }))
}

fn check_queries(
    packet: &protocol::ColumnQueries,
    layouts: &[weights::Layout; 10],
    required: &[usize],
    params: &SoundnessParams,
    advice_log_k: u8,
) -> Check<()> {
    let advice_len = 1u64
        .checked_shl(u32::from(advice_log_k) + 1)
        .ok_or_else(internal)?;
    if packet.advice.len() != usize::try_from(params.advice_queries).map_err(|_| internal())?
        || packet.advice.iter().any(|&q| u64::from(q) >= advice_len)
        || packet.weights.len() != required.len()
    {
        return Err(internal());
    }
    for (p, &i) in packet.weights.iter().zip(required) {
        let l = layouts.get(i).ok_or_else(internal)?;
        let n = 1u64
            .checked_shl(l.log2_k().checked_add(2).ok_or_else(internal)?)
            .ok_or_else(internal)?;
        if usize::from(p.matrix) != i
            || p.indices.len() != usize::try_from(params.weight_queries).map_err(|_| internal())?
            || p.indices.iter().any(|&q| u64::from(q) >= n)
        {
            return Err(internal());
        }
    }
    Ok(())
}
