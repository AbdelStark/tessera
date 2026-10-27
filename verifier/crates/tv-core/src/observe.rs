// SPDX-License-Identifier: Apache-2.0 OR MIT

//! What a caller sees of the checks while they run: each check of RFC-0019 'Check order' that
//! runs is reported once it completes, with its outcome and its measured time, so that a caller
//! can write 05 §2's `check_completed` line inside 05 §5's `check{kind}` span. This crate does no
//! I/O: it hands each [`CheckEvent`] to the caller's [`Observer`].
//!
//! Observation is a side channel. A report decides nothing: the verdict, the checks and their
//! order are those of the unobserved entry points, which are the observed ones with `()` as the
//! observer, and [`Verdict::to_json`](crate::Verdict::to_json) keeps its null per-check
//! durations (05 §5): the times reported here are diagnostics, not the audit trail.
//!
//! # What each call reports
//!
//! Each observed call reports the checks it runs, in RFC-0019's order, each once, and ends with
//! the check running when it returned: `pass` when the call succeeded, else `reject` for a
//! [`Verdict::Reject`](crate::Verdict::Reject) and `inconclusive` for a
//! [`Verdict::Inconclusive`](crate::Verdict::Inconclusive). Step 5, the weight reservation, is
//! `not_applicable` in `tessera-protocol/1` and takes no time. A check that does not run is not
//! reported.
//!
//! | Call | Checks it reports when it succeeds |
//! |---|---|
//! | [`verify_non_interactive_observed`](crate::verify_non_interactive_observed), [`verify_non_interactive_bytes_observed`](crate::verify_non_interactive_bytes_observed) | 1 to 13 |
//! | [`LiveVerifier::start_observed`](crate::LiveVerifier::start_observed) | 2 |
//! | [`open_live_observed`](crate::open_live_observed) | 1 |
//! | [`LiveVerifier::on_round1_observed`](crate::LiveVerifier::on_round1_observed) | 1 to 6, before `SeedReveal` |
//! | [`LiveVerifier::on_round2_observed`](crate::LiveVerifier::on_round2_observed) | 1 to 10, before `ColumnQueries` |
//! | [`LiveVerifier::on_round3_observed`](crate::LiveVerifier::on_round3_observed) | 1 to 13 |
//!
//! In live mode a check spans the work it does on each message (05 §5): the driver checks
//! every message afresh from the trusted inputs, so each `on_round` call runs, and reports,
//! steps 1 to 4 and 6 again before the steps it adds. `start` runs only the parameter set's
//! check and the session's own resources. A caller that opens a message with
//! [`open_live_observed`](crate::open_live_observed) and then hands it to an `on_round` call gets
//! step 1 twice in a row, the decoding and then the driver's own limits; joining two consecutive
//! reports of one check whose first passed gives one step 1 per message.
//!
//! Observer callback time consumes the verification budget. If the last callback exhausts it,
//! the call returns `ERR-RES-TIME` even when all checks it already reported passed.
//!
//! # Which work belongs to which check
//!
//! The driver's checks run in one sequence, and each step's report covers the code from its
//! start to the next step's. Where the driver judges a fault of RFC-0019's step `a` after it has
//! begun step `b > a`, as the order of the existing checks requires, the fault is reported under
//! `b`, with its own reason: the check order and the verdicts stay those of the unobserved
//! driver.
//!
//! | Step | `kind` | Driver work |
//! |---|---|---|
//! | 1 | `input` | NI bytes: the size limits and `NiProof::open`. Live: the session's state, deadline and session ids, and the replay's decoding of the earlier messages. Then the encoded sizes against `max_artifact_bytes`, the weight packets' matrix order, the leaf grouping's bounds or an ungrouped stage's empty grouped fields, and the memory estimate |
//! | 2 | `params` | the parameter set and the mode; `start`'s whole call in live mode |
//! | 3 | `claims` | claim-list validation (RFC-0007) |
//! | 4 | `statement` | the weight artifact and the trusted statement, the protocol version, the proof's statement field by field, the request's bounds and predictor, the scoring claims' count, the lowering, the runtime plan and advice layout, the `layout_digest`, the weight layouts, the leaf, split and grouped layouts, the runtime geometry and the grouped stage |
//! | 5 | `weight_reservation` | none in `tessera-protocol/1`: `not_applicable` |
//! | 6 | `transcript` | a grouped stage's counts, the transcript's context and `Round1`, the seed's proof of work or the coin-toss seed, the weight matrices' `xi` points |
//! | 7 | `leaf` | the leaf stage, ungrouped or grouped, and its splits |
//! | 8 | `stages` | the native gadget binding and the reduction stages |
//! | 9 | `evaluations` | the original opening claims, the advice and weight challenges, the combined rows' shapes, the runtime soundness budget, the advice and weight evaluations, and their absorption with the scoring claims |
//! | 10 | `queries` | the query proof of work and indices (NI), the verifier's own draw (live, Round2), or the recorded queries (live, Round3); their absorption and their check |
//! | 11 | `columns` | the weight columns' shape, the advice and weight column checks |
//! | 12 | `exact_bound` | each request's `Û` against the exact bound |
//! | 13 | `complete` | the accepted report |

use core::time::Duration;
use std::time::Instant;

/// A check of RFC-0019 'Check order', named as the verdict report names it (05 §7).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CheckKind {
    /// Step 1: every input decoded under the resource limits.
    Input,
    /// Step 2: the parameter set.
    Params,
    /// Step 3: the claim list's validation and lowering.
    Claims,
    /// Step 4: the statement's binding.
    Statement,
    /// Step 5: the weight reservation, not applicable in `tessera-protocol/1`.
    WeightReservation,
    /// Step 6: the transcript and the seed.
    Transcript,
    /// Step 7: the leaf stage's tree levels.
    Leaf,
    /// Step 8: the reduction stages.
    Stages,
    /// Step 9: the opening groups' evaluations.
    Evaluations,
    /// Step 10: the column queries.
    Queries,
    /// Step 11: the queried columns.
    Columns,
    /// Step 12: the exact bound on `Û`.
    ExactBound,
    /// Step 13: completion and the accepted report.
    Complete,
}

impl CheckKind {
    /// Every check, in RFC-0019's order.
    pub const ALL: [Self; 13] = [
        Self::Input,
        Self::Params,
        Self::Claims,
        Self::Statement,
        Self::WeightReservation,
        Self::Transcript,
        Self::Leaf,
        Self::Stages,
        Self::Evaluations,
        Self::Queries,
        Self::Columns,
        Self::ExactBound,
        Self::Complete,
    ];

    /// Its step in RFC-0019 'Check order', 1 to 13.
    pub const fn step(self) -> u8 {
        match self {
            Self::Input => 1,
            Self::Params => 2,
            Self::Claims => 3,
            Self::Statement => 4,
            Self::WeightReservation => 5,
            Self::Transcript => 6,
            Self::Leaf => 7,
            Self::Stages => 8,
            Self::Evaluations => 9,
            Self::Queries => 10,
            Self::Columns => 11,
            Self::ExactBound => 12,
            Self::Complete => 13,
        }
    }

    /// Its name in the verdict report's `checks` (05 §7) and in the `check{kind}` span (05 §5).
    pub const fn name(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Params => "params",
            Self::Claims => "claims",
            Self::Statement => "statement",
            Self::WeightReservation => "weight_reservation",
            Self::Transcript => "transcript",
            Self::Leaf => "leaf",
            Self::Stages => "stages",
            Self::Evaluations => "evaluations",
            Self::Queries => "queries",
            Self::Columns => "columns",
            Self::ExactBound => "exact_bound",
            Self::Complete => "complete",
        }
    }
}

/// How a check ended (05 §2, `check_completed`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CheckOutcome {
    /// The check passed.
    Pass,
    /// The check does not apply: step 5 in `tessera-protocol/1`.
    NotApplicable,
    /// The check rejected the proof: the verdict is `Reject`.
    Reject,
    /// The verifier could not complete the check: the verdict is `Inconclusive`.
    Inconclusive,
}

impl CheckOutcome {
    /// Its name in the log line and in the verdict report: `pass`, `not_applicable`, `reject`
    /// or `inconclusive`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::NotApplicable => "not_applicable",
            Self::Reject => "reject",
            Self::Inconclusive => "inconclusive",
        }
    }
}

/// One completed check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckEvent {
    /// Which check.
    pub kind: CheckKind,
    /// How it ended.
    pub outcome: CheckOutcome,
    /// Its measured time, from its start to its end within the call that reports it.
    pub duration: Duration,
}

/// Receives each check as it completes. `()` observes nothing; a `Vec<CheckEvent>` keeps every
/// event, in order.
pub trait Observer {
    /// Called once per check that ran, in the order the checks ran, when it completes.
    fn check_completed(&mut self, event: CheckEvent);
}

impl Observer for () {
    fn check_completed(&mut self, _: CheckEvent) {}
}

impl Observer for Vec<CheckEvent> {
    fn check_completed(&mut self, event: CheckEvent) {
        self.push(event);
    }
}

/// The driver's side: which check is running, since when, and where its report goes.
pub(crate) struct Tracker<'o> {
    observer: &'o mut dyn Observer,
    current: Option<(CheckKind, Instant)>,
}

impl<'o> Tracker<'o> {
    pub(crate) fn new(observer: &'o mut dyn Observer) -> Self {
        Self {
            observer,
            current: None,
        }
    }

    /// Starts `kind`, ending the running check as passed; nothing when `kind` is running.
    pub(crate) fn enter(&mut self, kind: CheckKind) {
        if self.current.is_some_and(|(running, _)| running == kind) {
            return;
        }
        self.close(CheckOutcome::Pass);
        self.current = Some((kind, Instant::now()));
    }

    /// Reports `kind` as not applicable, ending the running check as passed.
    pub(crate) fn not_applicable(&mut self, kind: CheckKind) {
        self.close(CheckOutcome::Pass);
        self.observer.check_completed(CheckEvent {
            kind,
            outcome: CheckOutcome::NotApplicable,
            duration: Duration::ZERO,
        });
    }

    /// Ends the running check with `outcome`; nothing when none runs.
    pub(crate) fn close(&mut self, outcome: CheckOutcome) {
        if let Some((kind, started)) = self.current.take() {
            self.observer.check_completed(CheckEvent {
                kind,
                outcome,
                duration: started.elapsed(),
            });
        }
    }
}
