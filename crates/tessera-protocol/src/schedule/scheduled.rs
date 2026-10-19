// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A transcript that follows its schedule: every absorb, squeeze and proof of work is checked
//! against the next step, so that a party cannot use a label out of its place (RFC-0014
//! 'Transcript schedule', INV-FS-1, INV-FS-4), and the seed's and the queries' phases of both
//! modes (RFC-0014 'Session order' and 'Grinding').

use core::fmt;
use core::fmt::Write as _;

use tessera_codec::{Ascii, Encode};
use tessera_field::GoldilocksExt2;
use tessera_hash::{Blake3, Digest, SessionContext, Transcript, TranscriptHash};
use tessera_sumcheck::SumcheckTranscript;

use super::expand::{Op, TranscriptSchedule};
use super::label::Label;
use crate::{ColumnQueries, MatrixQueries, ProofOfWork, ProtocolError};

/// A proof-of-work nonce short of its target: the verifier's `Reject(ProofOfWork { phase })`
/// (04 §2; RFC-0043: `NiProof.seed_nonce`'s phase is the seed's, `query_nonce`'s the queries').
///
/// The nonce was not absorbed, and the verifier stops: it does not use the transcript again
/// (RFC-0004 `check_grind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PowRejected {
    /// The proof of work whose nonce failed.
    pub phase: ProofOfWork,
}

impl fmt::Display for PowRejected {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self.phase {
            ProofOfWork::Seed => Label::SeedPow,
            ProofOfWork::Query => Label::QueryPow,
        };
        write!(f, "the {label} nonce lacks its leading zero bits")
    }
}

impl std::error::Error for PowRejected {}

/// How a transcript left its schedule: a bug of the party that drives it (`ERR-INTERNAL`), or a
/// proof of work the verifier rejected.
///
/// Labels are public: the fault names the step and the label, never a message.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ScheduleFault {
    /// Step `step` is not what the caller did: another label or another operation, or any
    /// operation past the last step. `label` is the caller's.
    Diverged {
        /// The step the transcript expected, by position.
        step: usize,
        /// The label the caller used.
        label: String,
    },
    /// The schedule was left before step `step`, its first step not done.
    Unfinished {
        /// The first step not done.
        step: usize,
    },
    /// A proof of work was rejected: the verifier stops there (RFC-0004 `check_grind`).
    Rejected(PowRejected),
}

impl fmt::Display for ScheduleFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Diverged { step, label } => {
                write!(
                    f,
                    "the transcript left its schedule at step {step}, with {label}"
                )
            }
            Self::Unfinished { step } => {
                write!(
                    f,
                    "the transcript stopped before step {step} of its schedule"
                )
            }
            Self::Rejected(rejected) => write!(f, "{rejected}"),
        }
    }
}

impl std::error::Error for ScheduleFault {}

/// A transcript off its schedule is the party's own bug (`ERR-INTERNAL`). A rejected proof of
/// work is the proof's fault, which the verifier reports as `Reject(ProofOfWork)` from
/// [`PowRejected`], not as an error.
impl From<ScheduleFault> for ProtocolError {
    fn from(_: ScheduleFault) -> Self {
        Self::Internal {
            rule: "a transcript follows its schedule (RFC-0014)",
        }
    }
}

/// What a caller does on the transcript, which a step's [`Op`] must match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Action {
    Absorb,
    Digest,
    Fp2,
    Indices,
    Grind,
}

impl Op {
    fn action(self) -> Action {
        match self {
            Self::Absorb => Action::Absorb,
            Self::Digest => Action::Digest,
            Self::Fp2 { .. } => Action::Fp2,
            Self::Indices { .. } => Action::Indices,
            Self::Grind { .. } => Action::Grind,
        }
    }
}

/// RFC-0004's transcript, held to a [`TranscriptSchedule`].
///
/// Each operation must be the schedule's next step, under its label: [`absorb`](Self::absorb),
/// [`squeeze`](Self::squeeze), [`challenge_fp2`](Self::challenge_fp2) and
/// [`point`](Self::point), [`challenge_indices`](Self::challenge_indices), [`grind`](Self::grind)
/// and [`check_grind`](Self::check_grind), or the [`SumcheckTranscript`] calls of the sumcheck,
/// tree and LogUp provers. The phases of RFC-0014 that differ between the modes are whole
/// calls:
///
/// | | Live | Non-interactive, prover | Non-interactive, verifier |
/// |---|---|---|---|
/// | Seed, row 3 | [`seed_live`](Self::seed_live) | [`seed_grind`](Self::seed_grind) | [`seed_check`](Self::seed_check) |
/// | Queries, row 10 | [`queries_live`](Self::queries_live) | [`queries_grind`](Self::queries_grind) | [`queries_check`](Self::queries_check) |
///
/// An operation that is not the next step is a bug of the caller's: the transcript records the
/// first [`ScheduleFault`] and still performs the operation, so that the caller's code runs on,
/// and [`finish`](Self::finish) reports the fault. A party sends or accepts nothing whose
/// transcript did not finish. An out-of-place [`check_grind`](Self::check_grind) refuses its
/// nonce: a verifier never accepts a proof of work it did not check.
///
/// Like [`Transcript`], it is not `Clone`: one party, one transcript.
pub struct ScheduledTranscript<'s, H: TranscriptHash = Blake3> {
    transcript: Transcript<H>,
    schedule: &'s TranscriptSchedule,
    next: usize,
    drawn: u32,
    fault: Option<ScheduleFault>,
    text: String,
}

/// Shows the position in the schedule and the transcript's public state.
impl<H: TranscriptHash> fmt::Debug for ScheduledTranscript<'_, H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScheduledTranscript")
            .field("transcript", &self.transcript)
            .field("next", &self.next)
            .field("drawn", &self.drawn)
            .field("fault", &self.fault)
            .finish_non_exhaustive()
    }
}

impl<'s> ScheduledTranscript<'s> {
    /// A transcript over BLAKE3 for `context`, held to `schedule`.
    ///
    /// # Errors
    ///
    /// [`ProtocolError::Internal`] when `context` is not the schedule's: another protocol, mode
    /// or parameter set.
    pub fn new(
        context: &SessionContext,
        schedule: &'s TranscriptSchedule,
    ) -> Result<Self, ProtocolError> {
        Self::with_hash(context, Blake3, schedule)
    }
}

impl<'s, H: TranscriptHash> ScheduledTranscript<'s, H> {
    /// A transcript over `hash` for `context`, held to `schedule`.
    ///
    /// # Errors
    ///
    /// [`ProtocolError::Internal`] when `context` is not the schedule's: another protocol, mode
    /// or parameter set.
    pub fn with_hash(
        context: &SessionContext,
        hash: H,
        schedule: &'s TranscriptSchedule,
    ) -> Result<Self, ProtocolError> {
        let protocol = Ascii::new(schedule.protocol().name()).ok();
        if protocol.as_ref() != Some(&context.protocol)
            || context.mode != schedule.mode()
            || context.params != schedule.params().param_set_id()
        {
            return Err(ProtocolError::Internal {
                rule: "a transcript's session is its schedule's protocol, mode and parameter set",
            });
        }
        Ok(Self {
            transcript: Transcript::with_hash(context, hash),
            schedule,
            next: 0,
            drawn: 0,
            fault: None,
            text: String::with_capacity(32),
        })
    }

    /// The position of the next step in the schedule.
    pub fn position(&self) -> usize {
        self.next
    }

    /// The first fault, if the transcript has left its schedule.
    pub fn fault(&self) -> Option<&ScheduleFault> {
        self.fault.as_ref()
    }

    /// The transcript, whose state and counter are public.
    pub fn transcript(&self) -> &Transcript<H> {
        &self.transcript
    }

    /// The transcript once every step is done.
    ///
    /// # Errors
    ///
    /// The first [`ScheduleFault`]: an operation off the schedule, a rejected proof of work, or
    /// [`ScheduleFault::Unfinished`] when a step is left.
    pub fn finish(mut self) -> Result<Transcript<H>, ScheduleFault> {
        if let Some(fault) = self.fault.take() {
            return Err(fault);
        }
        // A point of no coordinate draws nothing.
        while self
            .schedule
            .steps()
            .get(self.next)
            .is_some_and(|step| step.op == Op::Fp2 { count: 0 })
        {
            self.next += 1;
        }
        if self.next < self.schedule.steps().len() {
            return Err(ScheduleFault::Unfinished { step: self.next });
        }
        Ok(self.transcript)
    }

    /// Checks that `action` under `label` is the next step, or the rest of a step being drawn,
    /// and returns the step's operation; records the first fault and returns `None` otherwise.
    /// The caller then advances with [`done`](Self::done).
    fn expect(&mut self, action: Action, label: &str) -> Option<Op> {
        if self.fault.is_some() {
            return None;
        }
        let steps = self.schedule.steps();
        while let Some(step) = steps.get(self.next) {
            self.text.clear();
            // Writing to a `String` cannot fail.
            let _ = write!(self.text, "{}", step.label);
            let same = self.text == label;
            if same && step.op.action() == action {
                return Some(step.op);
            }
            // A point of no coordinate draws nothing: the next step's operation goes on. A step
            // being drawn is never one: `drawn` is below its count.
            if !same && step.op == (Op::Fp2 { count: 0 }) {
                self.next += 1;
                continue;
            }
            break;
        }
        self.fault = Some(ScheduleFault::Diverged {
            step: self.next,
            label: label.to_owned(),
        });
        None
    }

    /// Advances past one draw of the current step, or past the step.
    fn done(&mut self, op: Op) {
        match op {
            Op::Fp2 { count } => {
                self.drawn += 1;
                if self.drawn >= count {
                    self.drawn = 0;
                    self.next += 1;
                }
            }
            _ => self.next += 1,
        }
    }

    /// Absorbs `value`'s TCE encoding under `label`, the next step.
    pub fn absorb<T: Encode + ?Sized>(&mut self, label: Label, value: &T) {
        let text = label.to_string();
        if let Some(op) = self.expect(Action::Absorb, &text) {
            self.done(op);
        }
        self.transcript.absorb(&text, value);
    }

    /// Squeezes 32 bytes under `label`, the next step: the non-interactive seed.
    pub fn squeeze(&mut self, label: Label) -> Digest {
        let text = label.to_string();
        if let Some(op) = self.expect(Action::Digest, &text) {
            self.done(op);
        }
        self.transcript.squeeze(&text)
    }

    /// One challenge in `Fp2` under `label`: the next step's, or its next coordinate.
    pub fn challenge_fp2(&mut self, label: Label) -> GoldilocksExt2 {
        let text = label.to_string();
        self.draw_fp2(&text)
    }

    fn draw_fp2(&mut self, label: &str) -> GoldilocksExt2 {
        match self.expect(Action::Fp2, label) {
            // A point of no coordinate draws nothing, so a draw under its label is one too many.
            Some(Op::Fp2 { count: 0 }) => {
                self.fault = Some(ScheduleFault::Diverged {
                    step: self.next,
                    label: label.to_owned(),
                });
            }
            Some(op) => self.done(op),
            None => {}
        }
        self.transcript.challenge_fp2(label)
    }

    /// The challenges the next step draws under `label` that are not drawn yet, in draw order:
    /// the whole of a proximity point, or a single challenge. A point of no coordinate is empty.
    /// Which coordinate each draw sets is the opening's (RFC-0011 step 3, RFC-0049).
    pub fn point(&mut self, label: Label) -> Vec<GoldilocksExt2> {
        let text = label.to_string();
        let Some(op @ Op::Fp2 { count }) = self.expect(Action::Fp2, &text) else {
            return vec![self.transcript.challenge_fp2(&text)];
        };
        if count == 0 {
            self.next += 1;
            return Vec::new();
        }
        let left = count - self.drawn;
        let mut point = Vec::with_capacity(left as usize);
        for _ in 0..left {
            self.done(op);
            point.push(self.transcript.challenge_fp2(&text));
        }
        point
    }

    /// The column indices the next step draws under `label`, each below `2^log_n`, in draw order.
    pub fn challenge_indices(&mut self, label: Label) -> Vec<u64> {
        let text = label.to_string();
        match self.expect(Action::Indices, &text) {
            Some(op @ Op::Indices { log_n, count }) => {
                self.done(op);
                self.transcript.challenge_indices(&text, log_n, count)
            }
            _ => self.transcript.challenge_indices(&text, 0, 1),
        }
    }

    /// The prover's proof of work under `label`, the next step: the smallest nonce whose hash has
    /// the step's leading zero bits, which it absorbs (RFC-0004 `grind`).
    pub fn grind(&mut self, label: Label) -> u64 {
        let text = label.to_string();
        let bits = match self.expect(Action::Grind, &text) {
            Some(op @ Op::Grind { bits, .. }) => {
                self.done(op);
                bits
            }
            _ => 0,
        };
        // A schedule's proofs of work ask at most 64 bits.
        self.transcript
            .grind(&text, u8::try_from(bits).unwrap_or(u8::MAX))
    }

    /// The verifier's proof of work under `label`, the next step: whether `nonce`'s hash has the
    /// step's leading zero bits, counted exactly (RFC-0004 `check_grind`). A valid nonce is
    /// absorbed, as [`grind`](Self::grind) absorbs it.
    ///
    /// # Errors
    ///
    /// [`PowRejected`] when the nonce falls short, which is not absorbed: the verifier rejects the
    /// proof and does not use the transcript again. A call out of the schedule's place is refused
    /// too, its fault recorded.
    pub fn check_grind(&mut self, label: Label, nonce: u64) -> Result<(), PowRejected> {
        let text = label.to_string();
        let Some(op @ Op::Grind { phase, bits }) = self.expect(Action::Grind, &text) else {
            let phase = match label {
                Label::QueryPow => ProofOfWork::Query,
                _ => ProofOfWork::Seed,
            };
            return Err(PowRejected { phase });
        };
        // A schedule's proofs of work ask at most 64 bits.
        let bits = u8::try_from(bits).unwrap_or(u8::MAX);
        if self.transcript.check_grind(&text, bits, nonce) {
            self.done(op);
            Ok(())
        } else {
            let rejected = PowRejected { phase };
            self.fault = Some(ScheduleFault::Rejected(rejected));
            Err(rejected)
        }
    }

    /// Row 3 in a live session: absorbs the coin-tossed seed's 32 bytes under `seed` (RFC-0036
    /// 'What the transcript absorbs').
    pub fn seed_live(&mut self, seed: &Digest) {
        self.absorb(Label::Seed, seed);
    }

    /// Row 3 for a non-interactive prover: grinds `seed-pow` for `g_seed` bits, then squeezes the
    /// seed under `seed`. Returns `NiProof.seed_nonce` and the seed.
    pub fn seed_grind(&mut self) -> (u64, Digest) {
        let nonce = self.grind(Label::SeedPow);
        (nonce, self.squeeze(Label::Seed))
    }

    /// Row 3 for a non-interactive verifier: checks `NiProof.seed_nonce` under `seed-pow`, then
    /// squeezes the seed under `seed`.
    ///
    /// # Errors
    ///
    /// [`PowRejected`] with the seed's phase when the nonce falls short (`Reject(ProofOfWork)`).
    pub fn seed_check(&mut self, nonce: u64) -> Result<Digest, PowRejected> {
        self.check_grind(Label::SeedPow, nonce)?;
        Ok(self.squeeze(Label::Seed))
    }

    /// Row 10 in a live session: absorbs the verifier's `ColumnQueries` under `queries`, after
    /// the whole `Round2Response` (INV-PROTO-3).
    pub fn queries_live(&mut self, queries: &ColumnQueries) {
        self.absorb(Label::Queries, queries);
    }

    /// Row 10 for a non-interactive prover: grinds `query-pow` for `g_query` bits, then squeezes
    /// the advice's `τ_A` indices and each opened matrix's `τ_W`, in id order (RFC-0004
    /// `challenge_indices`). Returns `NiProof.query_nonce` and the queries.
    pub fn queries_grind(&mut self) -> (u64, ColumnQueries) {
        let nonce = self.grind(Label::QueryPow);
        (nonce, self.draw_queries())
    }

    /// Row 10 for a non-interactive verifier: checks `NiProof.query_nonce` under `query-pow`,
    /// then squeezes the queries as [`queries_grind`](Self::queries_grind) does.
    ///
    /// # Errors
    ///
    /// [`PowRejected`] with the queries' phase when the nonce falls short
    /// (`Reject(ProofOfWork)`).
    pub fn queries_check(&mut self, nonce: u64) -> Result<ColumnQueries, PowRejected> {
        self.check_grind(Label::QueryPow, nonce)?;
        Ok(self.draw_queries())
    }

    /// The squeezed queries: `pcs/advice/queries`, then each `pcs/weights/{T}/queries` the
    /// schedule has next. Every index is below `2^log_n ≤ 2^32` (`ScheduleSizes`), so it is a
    /// `u32`.
    fn draw_queries(&mut self) -> ColumnQueries {
        let indices = |drawn: Vec<u64>| -> Vec<u32> {
            drawn
                .into_iter()
                .map(|i| u32::try_from(i).unwrap_or(u32::MAX))
                .collect()
        };
        let advice = indices(self.challenge_indices(Label::AdviceQueries));
        let mut weights = Vec::new();
        while let Some(&step) = self.schedule.steps().get(self.next) {
            let Label::WeightsQueries(matrix) = step.label else {
                break;
            };
            if self.fault.is_some() {
                break;
            }
            weights.push(MatrixQueries {
                matrix: matrix.id(),
                indices: indices(self.challenge_indices(step.label)),
            });
        }
        ColumnQueries { advice, weights }
    }
}

/// The sumcheck, tree and LogUp provers' operations, each held to the schedule: a message absorbs
/// as the TCE `Vec<Fp2>` it is, and a challenge is one draw in `Fp2`.
impl<H: TranscriptHash> SumcheckTranscript<GoldilocksExt2> for ScheduledTranscript<'_, H> {
    fn absorb_message(&mut self, label: &str, message: &[GoldilocksExt2]) {
        if let Some(op) = self.expect(Action::Absorb, label) {
            self.done(op);
        }
        self.transcript.absorb_message(label, message);
    }

    fn challenge(&mut self, label: &str) -> GoldilocksExt2 {
        self.draw_fp2(label)
    }
}
