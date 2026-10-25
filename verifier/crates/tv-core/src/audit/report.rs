// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The audit report at a sample's deadline (RFC-0016 'Audit report'): one outcome per selected
//! request, in the sample's order, their counts, and the `tessera.audit-report/1` rendering.

use core::fmt::Write;

use tv_hash::Digest;

use super::ledger::VerifierLedger;
use super::sample::{AuditSample, RequestKey};
use crate::anchor::{Failure, usage};
use crate::protocol::FixedBits;
use crate::verdict::{ErrorCode, Json, RejectReason, VerifierError};

/// Why a selected request is unproven.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnprovenReason {
    /// No accepted proof covered it by the deadline.
    DeadlineMissed,
    /// The prover returned `ERR-WITNESS-EXACT-MISMATCH` for it: it disclosed a deviation.
    ProverReportedMismatch,
    /// The prover's pre-check failed, `ERR-WITNESS-ANCHOR`: it disclosed that its transcripts do
    /// not match the log.
    ProverReportedAnchorMismatch,
}

impl UnprovenReason {
    /// The name `tessera.audit-report/1` gives it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::DeadlineMissed => "DeadlineMissed",
            Self::ProverReportedMismatch => "ProverReportedMismatch",
            Self::ProverReportedAnchorMismatch => "ProverReportedAnchorMismatch",
        }
    }
}

/// A selected request's end in the audit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RequestOutcome {
    /// An accepted proof covered it, with its `Û`.
    Accepted {
        /// The proof's `Û` for the request, in units of `2^-8` bits.
        uhat: FixedBits,
    },
    /// A proof that covered it was rejected.
    Rejected(RejectReason),
    /// It is unproven.
    Unproven(UnprovenReason),
    /// The verifier could not judge the proof that covered it.
    Inconclusive(VerifierError),
}

/// The outcomes' counts: `selected` is their number and the sum of the other four; `max_uhat`
/// is the largest accepted `Û`, `None` when no request is accepted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AuditSummary {
    /// The outcomes.
    pub selected: u64,
    /// The accepted ones.
    pub accepted: u64,
    /// The rejected ones.
    pub rejected: u64,
    /// The unproven ones.
    pub unproven: u64,
    /// The inconclusive ones.
    pub inconclusive: u64,
    /// The largest accepted `Û`.
    pub max_uhat: Option<FixedBits>,
}

/// The verifier's audit report at a sample's deadline (RFC-0016 'Audit report'): `sample`, the
/// sample seed; the period's `manifest_digest`; one outcome per selected request, in
/// `selected`'s order; and their summary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuditReport {
    sample: Digest,
    manifest_digest: Digest,
    outcomes: Vec<(RequestKey, RequestOutcome)>,
    summary: AuditSummary,
}

impl AuditReport {
    /// The report of `sample` with `outcomes`, one per selected request in `selected`'s order,
    /// for a verifier that keeps its verdicts (RFC-0067 'The report from the ledger'). The sample
    /// must be the verifier's own, in `ledger`'s `derived`; the seed is recomputed from its coin
    /// toss, which must open; and the outcomes must agree with the ledger: `Accepted` exactly for
    /// the requests the ledger records accepted, each with the `Û` it records (RFC-0067 'The
    /// verifier's own sample').
    ///
    /// # Errors
    ///
    /// `Inconclusive(ERR-USAGE)` for a sample the verifier did not derive, another number of
    /// outcomes than of selected requests, outcomes that disagree with the ledger, or a record
    /// that names an index past `selected`; and `Inconclusive(ERR-PROTO-COINTOSS)` for a coin
    /// toss that does not open.
    pub fn new(
        sample: &AuditSample,
        ledger: &VerifierLedger,
        outcomes: Vec<RequestOutcome>,
    ) -> Result<Self, Failure> {
        if !ledger.is_derived(sample) {
            return Err(usage());
        }
        sample.coin_toss().check().map_err(Failure::Inconclusive)?;
        if outcomes.len() != sample.selected().len() {
            return Err(usage());
        }
        let recorded = recorded(sample, ledger)?;
        let agrees =
            outcomes
                .iter()
                .zip(&recorded)
                .all(|(outcome, record)| match (outcome, record) {
                    (RequestOutcome::Accepted { uhat }, Some(r)) => uhat == r,
                    (RequestOutcome::Accepted { .. }, None) | (_, Some(_)) => false,
                    (_, None) => true,
                });
        if !agrees {
            return Err(usage());
        }
        let mut summary = AuditSummary::default();
        let mut paired = Vec::new();
        paired.try_reserve_exact(outcomes.len()).map_err(|_| {
            Failure::Inconclusive(VerifierError {
                code: ErrorCode::Memory,
            })
        })?;
        for (key, outcome) in sample.selected().iter().zip(outcomes) {
            summary.selected = summary.selected.saturating_add(1);
            match &outcome {
                RequestOutcome::Accepted { uhat } => {
                    summary.accepted = summary.accepted.saturating_add(1);
                    summary.max_uhat = Some(
                        summary
                            .max_uhat
                            .map_or(*uhat, |m| if uhat.0 > m.0 { *uhat } else { m }),
                    );
                }
                RequestOutcome::Rejected(_) => {
                    summary.rejected = summary.rejected.saturating_add(1);
                }
                RequestOutcome::Unproven(_) => {
                    summary.unproven = summary.unproven.saturating_add(1);
                }
                RequestOutcome::Inconclusive(_) => {
                    summary.inconclusive = summary.inconclusive.saturating_add(1);
                }
            }
            paired.push((*key, outcome));
        }
        Ok(Self {
            sample: sample.seed(),
            manifest_digest: *sample.manifest_digest(),
            outcomes: paired,
            summary,
        })
    }

    /// The report at the deadline from the verifier's ledger: each request the ledger records
    /// accepted for this sample is `Accepted` with its `Û`, and every other one
    /// `Unproven(DeadlineMissed)`. The ledger keeps acceptances only (RFC-0019 'Ledger'), so a
    /// request whose only proofs were rejected or inconclusive is unproven here too.
    ///
    /// # Errors
    ///
    /// `Inconclusive(ERR-USAGE)` for a record of the sample that names an index past its
    /// `selected`, and the errors of [`new`](Self::new), which refuses a sample the verifier did
    /// not derive (RFC-0067 'The verifier's own sample'). The ledger holds records of derived
    /// samples only, so such a sample has none, and is refused there.
    pub fn from_ledger(sample: &AuditSample, ledger: &VerifierLedger) -> Result<Self, Failure> {
        let mut outcomes = Vec::new();
        let recorded = recorded(sample, ledger)?;
        outcomes
            .try_reserve_exact(recorded.len())
            .map_err(|_| memory())?;
        outcomes.extend(recorded.into_iter().map(|record| match record {
            Some(uhat) => RequestOutcome::Accepted { uhat },
            None => RequestOutcome::Unproven(UnprovenReason::DeadlineMissed),
        }));
        Self::new(sample, ledger, outcomes)
    }

    /// The sample seed.
    pub fn sample(&self) -> &Digest {
        &self.sample
    }

    /// The period's manifest digest.
    pub fn manifest_digest(&self) -> &Digest {
        &self.manifest_digest
    }

    /// One outcome per selected request, in `selected`'s order.
    pub fn outcomes(&self) -> &[(RequestKey, RequestOutcome)] {
        &self.outcomes
    }

    /// The outcomes' counts.
    pub fn summary(&self) -> &AuditSummary {
        &self.summary
    }

    /// The `tessera.audit-report/1` rendering: digests lowercase hex, `Û` as `uhat_fixed`, a
    /// rejection as `tessera.verdict/1` renders a verdict's `reason`, an error as its `error`.
    ///
    /// # Errors
    ///
    /// `ERR-RES-MEMORY` when the text cannot be reserved.
    pub fn to_json(&self) -> Result<String, VerifierError> {
        let mut out = Json(String::new());
        self.write_json(&mut out).map_err(|_| VerifierError {
            code: ErrorCode::Memory,
        })?;
        Ok(out.0)
    }

    fn write_json(&self, out: &mut Json) -> core::fmt::Result {
        out.write_str("{\"schema\":\"tessera.audit-report/1\",\"sample\":")?;
        out.digest(&self.sample)?;
        out.write_str(",\"manifest\":")?;
        out.digest(&self.manifest_digest)?;
        out.write_str(",\"outcomes\":[")?;
        for (i, (key, outcome)) in self.outcomes.iter().enumerate() {
            if i != 0 {
                out.write_char(',')?;
            }
            out.write_str("{\"session_id\":\"")?;
            for b in key.session_id {
                write!(out, "{b:02x}")?;
            }
            write!(out, "\",\"request_id\":{},\"outcome\":", key.request_id)?;
            match outcome {
                RequestOutcome::Accepted { uhat } => {
                    write!(out, "\"accepted\",\"uhat_fixed\":{}", uhat.0)?;
                }
                RequestOutcome::Rejected(reason) => {
                    write!(
                        out,
                        "\"rejected\",\"reject\":{{\"code\":\"{}\",",
                        reason.code()
                    )?;
                    reason.payload(out)?;
                    out.write_char('}')?;
                }
                RequestOutcome::Unproven(reason) => {
                    write!(out, "\"unproven\",\"unproven\":\"{}\"", reason.name())?;
                }
                RequestOutcome::Inconclusive(error) => {
                    write!(
                        out,
                        "\"inconclusive\",\"error\":{{\"code\":\"{}\"}}",
                        error.code.id()
                    )?;
                }
            }
            out.write_char('}')?;
        }
        let s = &self.summary;
        write!(
            out,
            "],\"summary\":{{\"selected\":{},\"accepted\":{},\"rejected\":{},\"unproven\":{},\"inconclusive\":{},\"max_uhat_fixed\":",
            s.selected, s.accepted, s.rejected, s.unproven, s.inconclusive
        )?;
        match s.max_uhat {
            Some(u) => write!(out, "{}", u.0)?,
            None => out.write_str("null")?,
        }
        out.write_str("}}\n")
    }
}

/// The ledger's record of `sample`, one entry per selected request: its `Û` when the ledger
/// records it accepted. A record naming an index past `selected` is `Inconclusive(ERR-USAGE)`:
/// the ledger is not this sample's.
fn recorded(
    sample: &AuditSample,
    ledger: &VerifierLedger,
) -> Result<Vec<Option<FixedBits>>, Failure> {
    let count = sample.selected().len();
    let mut out = Vec::new();
    out.try_reserve_exact(count).map_err(|_| memory())?;
    out.resize(count, None);
    if let Some(record) = ledger.proven(sample.identity()) {
        for (index, uhat) in record.iter() {
            let slot = usize::try_from(index)
                .ok()
                .and_then(|i| out.get_mut(i))
                .ok_or(usage())?;
            *slot = Some(uhat);
        }
    }
    Ok(out)
}

const fn memory() -> Failure {
    Failure::Inconclusive(VerifierError {
        code: ErrorCode::Memory,
    })
}
