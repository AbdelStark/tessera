// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The verifier's audit plan (RFC-0016 'Calculators'): a policy's detection probability at a
//! traffic profile, from the policy and the traffic alone, and, given the prover's cost model's
//! `μ` at the traffic's mean lengths, the amortized overhead and the rates for a target overhead.
//!
//! It is written from RFC-0016 without reference to the prover's `AuditPlan`; CI checks that the
//! two agree within `10^{-12}` relative (RFC-0016 'Agreement'). The arithmetic is IEEE 754
//! binary64, with `ln_1p` and `exp_m1` so that small rates keep their precision. These are
//! planning figures: none enters a sample, whose thresholds are exact integers ([`crate::audit`]),
//! or any TCE artifact (INV-DM-6).
//!
//! # Specification
//!
//! - **Rates.** `s_b = threshold_b / 2^64`, the threshold rounded once to binary64; `s̲_b =
//!   min_{c ≥ b} s_c`, the lowest rate of bucket `b` and every later one.
//! - **Detection** of `v_b` violations per bucket: `1 − Π_b (1 − s̲_b)^{v_b}`, computed as
//!   `−exp_m1(Σ_b v_b · ln_1p(−s̲_b))`; per bucket `1 − (1 − s̲_b)^{v_b}`. A bucket with no
//!   violation contributes nothing, so a rate of 1 meets no `0 · ∞`.
//! - **Detection in place**: the same with `s_b`, the upper bound if no violation leaves its
//!   bucket.
//! - **Expected selections**: `n_b · s_b`, and their sum.
//! - **With `μ`**: the amortized overhead `Σ_b n_b·L_b·s_b·μ_b·η / Σ_b n_b·L_b`, and each
//!   bucket's target rate `min(1, O / (μ_b·η))`, with `η` and `O` the cost model's scenario's.
//!
//! A profile that is not one entry per bucket of the policy, whose mean length `L_b` is not above
//! the previous bucket's `max_tokens` and at most its own (and at least 1), whose `v_b` exceeds
//! `n_b`, or that has no request is refused ([`PlanError`], `ERR-USAGE`). So is a cost model
//! whose contexts are not the traffic's mean lengths, in order, or whose `μ_b`, `η` or `O` is not
//! a finite positive number.

use core::fmt::{self, Write};

use crate::audit::AuditPolicy;
use crate::verdict::{ErrorCode, VerifierError};

/// One bucket of a traffic profile, `tessera.audit-traffic/1`, in the policy's order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TrafficBucket {
    /// `n_b`: the requests whose length falls in the bucket over a period.
    pub requests: u64,
    /// `L_b`: their mean length in tokens.
    pub tokens: u32,
    /// `v_b`: the violating requests among them.
    pub violations: u64,
}

/// What the plan reads from a `tessera.cost-model/1` report (`--mu`): whether its counts are the
/// source's, its scenario's `η` and target overhead `O`, and each projection's context with its
/// `μ`'s total, in the report's order.
#[derive(Clone, Debug, PartialEq)]
pub struct CostModel {
    /// Whether `μ` comes from the source's counts and unit costs.
    pub source_defaults: bool,
    /// `η`, the ratio of ideal prefill time to serving time per token.
    pub eta: f64,
    /// `O`, the amortized overhead the target rates are set for.
    pub target_overhead: f64,
    /// Each projection's context and `μ`.
    pub mu: Vec<(u64, f64)>,
}

/// A profile or cost model the plan refuses, each `ERR-USAGE`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PlanError {
    /// The profile does not have one entry per bucket of the policy, or the policy has none.
    Buckets,
    /// Bucket `bucket`'s mean length is 0, at most the previous bucket's `max_tokens`, or above
    /// its own.
    Tokens {
        /// The bucket.
        bucket: usize,
    },
    /// Bucket `bucket` has more violations than requests.
    Violations {
        /// The bucket.
        bucket: usize,
    },
    /// The profile has no request.
    NoRequests,
    /// The cost model's contexts are not the traffic's mean lengths, in order.
    Contexts,
    /// Bucket `bucket`'s `μ` is not a finite positive number.
    Mu {
        /// The bucket.
        bucket: usize,
    },
    /// `η` is not a finite positive number.
    Eta,
    /// The target overhead is not a finite positive number.
    TargetOverhead,
}

impl PlanError {
    /// The error code: every refusal is the caller's input, `ERR-USAGE`.
    pub const fn code(self) -> ErrorCode {
        ErrorCode::Usage
    }
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Buckets => f.write_str("the traffic profile does not have one entry per bucket"),
            Self::Tokens { bucket } => {
                write!(f, "bucket {bucket}'s mean length is not within its bucket")
            }
            Self::Violations { bucket } => {
                write!(f, "bucket {bucket} has more violations than requests")
            }
            Self::NoRequests => f.write_str("the traffic profile has no request"),
            Self::Contexts => {
                f.write_str("the cost model's contexts are not the traffic's mean lengths")
            }
            Self::Mu { bucket } => write!(f, "bucket {bucket}'s mu is not a positive number"),
            Self::Eta => f.write_str("eta is not a positive number"),
            Self::TargetOverhead => f.write_str("the target overhead is not a positive number"),
        }
    }
}

impl std::error::Error for PlanError {}

/// One bucket's figures.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanBucket {
    /// The bucket's `max_tokens`.
    pub max_tokens: u32,
    /// `s_b`.
    pub rate: f64,
    /// `s̲_b = min_{c ≥ b} s_c`.
    pub lowest_later_rate: f64,
    /// `n_b`.
    pub requests: u64,
    /// `L_b`.
    pub tokens: u32,
    /// `v_b`.
    pub violations: u64,
    /// `μ_b`, with a cost model.
    pub mu: Option<f64>,
    /// `n_b · s_b`.
    pub expected_selected: f64,
    /// `min(1, O / (μ_b·η))`, with a cost model.
    pub target_rate: Option<f64>,
    /// `1 − (1 − s̲_b)^{v_b}`.
    pub detection: f64,
    /// `1 − (1 − s_b)^{v_b}`.
    pub detection_in_place: f64,
}

/// The figures that need the cost model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanCost {
    /// Whether `μ` comes from the source's counts.
    pub source_defaults: bool,
    /// `η`.
    pub eta: f64,
    /// `O`.
    pub target_overhead: f64,
    /// `Σ_b n_b·L_b·s_b·μ_b·η / Σ_b n_b·L_b`.
    pub overhead: f64,
}

/// The verifier's audit plan (`tessera.audit-plan/1`).
#[derive(Clone, Debug, PartialEq)]
pub struct AuditPlan {
    /// Each bucket's figures, in the policy's order.
    pub buckets: Vec<PlanBucket>,
    /// `1 − Π_b (1 − s̲_b)^{v_b}`.
    pub detection: f64,
    /// `1 − Π_b (1 − s_b)^{v_b}`.
    pub detection_in_place: f64,
    /// `Σ_b n_b · s_b`.
    pub expected_selected: f64,
    /// The overhead and the scenario, with a cost model; `None` reports detection only.
    pub cost: Option<PlanCost>,
}

/// `2^64`, exactly.
const TWO_64: f64 = 18_446_744_073_709_551_616.0;
/// `2^32`, exactly.
const TWO_32: f64 = 4_294_967_296.0;

/// `x` rounded once to binary64: the high half times `2^32` is exact, so the one addition
/// rounds the exact value.
fn wide(x: u64) -> f64 {
    let high = u32::try_from(x >> 32).unwrap_or(u32::MAX);
    let low = u32::try_from(x & 0xFFFF_FFFF).unwrap_or(u32::MAX);
    f64::from(high) * TWO_32 + f64::from(low)
}

/// `v · ln(1 − s)`, 0 for no violation.
fn log_miss(violations: u64, rate: f64) -> f64 {
    if violations == 0 {
        0.0
    } else {
        wide(violations) * (-rate).ln_1p()
    }
}

/// `1 − e^{x}` from `x = Σ v·ln(1 − s)`; `+ 0.0` turns `−0` into `0`.
fn detected(log_miss: f64) -> f64 {
    -log_miss.exp_m1() + 0.0
}

fn positive(x: f64) -> bool {
    x.is_finite() && x > 0.0
}

impl AuditPlan {
    /// The plan of `policy` at `traffic`, with `cost` for the overhead and the target rates.
    ///
    /// # Errors
    ///
    /// The first refusal, in the order of [`PlanError`]'s variants, bucket by bucket.
    pub fn new(
        policy: &AuditPolicy,
        traffic: &[TrafficBucket],
        cost: Option<&CostModel>,
    ) -> Result<Self, PlanError> {
        let buckets = policy.buckets();
        if traffic.is_empty() || traffic.len() != buckets.len() {
            return Err(PlanError::Buckets);
        }
        let mut floor: u32 = 0;
        for (bucket, (t, b)) in traffic.iter().zip(buckets).enumerate() {
            if t.tokens == 0 || t.tokens <= floor || t.tokens > b.max_tokens {
                return Err(PlanError::Tokens { bucket });
            }
            floor = b.max_tokens;
        }
        for (bucket, t) in traffic.iter().enumerate() {
            if t.violations > t.requests {
                return Err(PlanError::Violations { bucket });
            }
        }
        if traffic.iter().all(|t| t.requests == 0) {
            return Err(PlanError::NoRequests);
        }
        if let Some(cost) = cost {
            let contexts = cost.mu.iter().map(|&(context, _)| context);
            if !contexts.eq(traffic.iter().map(|t| u64::from(t.tokens))) {
                return Err(PlanError::Contexts);
            }
            if let Some(bucket) = cost.mu.iter().position(|&(_, mu)| !positive(mu)) {
                return Err(PlanError::Mu { bucket });
            }
            if !positive(cost.eta) {
                return Err(PlanError::Eta);
            }
            if !positive(cost.target_overhead) {
                return Err(PlanError::TargetOverhead);
            }
        }

        // The lowest rate of each bucket and every later one, from the last bucket back.
        let mut lowest = vec![u64::MAX; buckets.len()];
        let mut running = u64::MAX;
        for (slot, b) in lowest.iter_mut().zip(buckets).rev() {
            running = running.min(b.threshold);
            *slot = running;
        }

        let mut out = Vec::with_capacity(buckets.len());
        let (mut padded, mut in_place, mut expected) = (0.0f64, 0.0f64, 0.0f64);
        let (mut weighted, mut volume) = (0.0f64, 0.0f64);
        for (i, ((t, b), low)) in traffic.iter().zip(buckets).zip(&lowest).enumerate() {
            let rate = wide(b.threshold) / TWO_64;
            let lowest_later_rate = wide(*low) / TWO_64;
            let miss_padded = log_miss(t.violations, lowest_later_rate);
            let miss_in_place = log_miss(t.violations, rate);
            padded += miss_padded;
            in_place += miss_in_place;
            let n = wide(t.requests);
            let expected_selected = n * rate;
            expected += expected_selected;
            let mu = cost.and_then(|c| c.mu.get(i)).map(|&(_, mu)| mu);
            let target_rate = cost
                .zip(mu)
                .map(|(c, mu)| (c.target_overhead / (mu * c.eta)).min(1.0));
            if let Some(mu) = mu {
                let tokens = n * f64::from(t.tokens);
                weighted += tokens * rate * mu;
                volume += tokens;
            }
            out.push(PlanBucket {
                max_tokens: b.max_tokens,
                rate,
                lowest_later_rate,
                requests: t.requests,
                tokens: t.tokens,
                violations: t.violations,
                mu,
                expected_selected,
                target_rate,
                detection: detected(miss_padded),
                detection_in_place: detected(miss_in_place),
            });
        }
        let cost = cost.map(|c| PlanCost {
            source_defaults: c.source_defaults,
            eta: c.eta,
            target_overhead: c.target_overhead,
            overhead: weighted * c.eta / volume,
        });
        Ok(Self {
            buckets: out,
            detection: detected(padded),
            detection_in_place: detected(in_place),
            expected_selected: expected,
            cost,
        })
    }

    /// The `tessera.audit-plan/1` rendering. Without a cost model, the figures that need `μ`
    /// (`source_defaults`, `eta`, `target_overhead`, `overhead`, and each bucket's `mu` and
    /// `target_rate`) are `null`: the verifier reports detection only (RFC-0016 'Calculators').
    ///
    /// # Errors
    ///
    /// `ERR-RES-MEMORY` when the text cannot be reserved, and `ERR-INTERNAL` for a figure that is
    /// not finite, which the checks above make impossible.
    pub fn to_json(&self) -> Result<String, VerifierError> {
        let mut out = String::new();
        self.write_json(&mut out).map_err(|_| VerifierError {
            code: ErrorCode::Memory,
        })?;
        if self.figures().any(|x| !x.is_finite()) {
            return Err(VerifierError {
                code: ErrorCode::Internal,
            });
        }
        Ok(out)
    }

    fn figures(&self) -> impl Iterator<Item = f64> + '_ {
        let totals = [
            self.detection,
            self.detection_in_place,
            self.expected_selected,
        ];
        let cost = self
            .cost
            .iter()
            .flat_map(|c| [c.eta, c.target_overhead, c.overhead]);
        let buckets = self.buckets.iter().flat_map(|b| {
            [
                b.rate,
                b.lowest_later_rate,
                b.expected_selected,
                b.detection,
                b.detection_in_place,
            ]
            .into_iter()
            .chain(b.mu)
            .chain(b.target_rate)
        });
        totals.into_iter().chain(cost).chain(buckets)
    }

    fn write_json(&self, out: &mut String) -> fmt::Result {
        out.write_str("{\"schema\":\"tessera.audit-plan/1\",\"source_defaults\":")?;
        match &self.cost {
            Some(c) => write!(
                out,
                "{},\"eta\":{:?},\"target_overhead\":{:?}",
                c.source_defaults, c.eta, c.target_overhead
            )?,
            None => out.write_str("null,\"eta\":null,\"target_overhead\":null")?,
        }
        out.write_str(",\"buckets\":[")?;
        for (i, b) in self.buckets.iter().enumerate() {
            if i != 0 {
                out.write_char(',')?;
            }
            write!(
                out,
                "{{\"max_tokens\":{},\"rate\":{:?},\"lowest_later_rate\":{:?},\"requests\":{},\"tokens\":{},\"violations\":{},\"mu\":",
                b.max_tokens, b.rate, b.lowest_later_rate, b.requests, b.tokens, b.violations
            )?;
            optional(out, b.mu)?;
            write!(
                out,
                ",\"expected_selected\":{:?},\"target_rate\":",
                b.expected_selected
            )?;
            optional(out, b.target_rate)?;
            write!(
                out,
                ",\"detection\":{:?},\"detection_in_place\":{:?}}}",
                b.detection, b.detection_in_place
            )?;
        }
        write!(
            out,
            "],\"detection\":{:?},\"detection_in_place\":{:?},\"overhead\":",
            self.detection, self.detection_in_place
        )?;
        optional(out, self.cost.map(|c| c.overhead))?;
        writeln!(out, ",\"expected_selected\":{:?}}}", self.expected_selected)
    }

    /// A table for people: each bucket's rates, traffic and detections, then the totals.
    pub fn to_table(&self) -> String {
        let mut out = String::new();
        let _ = self.write_table(&mut out);
        out
    }

    fn write_table(&self, out: &mut String) -> fmt::Result {
        writeln!(
            out,
            "{:>10} {:>12} {:>12} {:>10} {:>8} {:>8} {:>12} {:>12} {:>12} {:>12}",
            "max_tokens",
            "rate",
            "lowest_rate",
            "requests",
            "tokens",
            "violate",
            "detection",
            "in_place",
            "mu",
            "target_rate"
        )?;
        for b in &self.buckets {
            writeln!(
                out,
                "{:>10} {:>12.6e} {:>12.6e} {:>10} {:>8} {:>8} {:>12.6} {:>12.6} {:>12} {:>12}",
                b.max_tokens,
                b.rate,
                b.lowest_later_rate,
                b.requests,
                b.tokens,
                b.violations,
                b.detection,
                b.detection_in_place,
                b.mu.map_or_else(|| "-".to_string(), |m| format!("{m:.6}")),
                b.target_rate
                    .map_or_else(|| "-".to_string(), |r| format!("{r:.6e}")),
            )?;
        }
        writeln!(
            out,
            "detection (padded into the lowest later rate): {:.6}",
            self.detection
        )?;
        writeln!(
            out,
            "detection in place (upper bound): {:.6}",
            self.detection_in_place
        )?;
        writeln!(out, "expected selections: {:.3}", self.expected_selected)?;
        match &self.cost {
            Some(c) => writeln!(
                out,
                "amortized overhead: {:.6} (eta {}, target overhead {})",
                c.overhead, c.eta, c.target_overhead
            ),
            None => writeln!(out, "amortized overhead: needs --mu"),
        }
    }
}

fn optional(out: &mut String, x: Option<f64>) -> fmt::Result {
    match x {
        Some(x) => write!(out, "{x:?}"),
        None => out.write_str("null"),
    }
}
