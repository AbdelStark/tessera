// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The audit calculators of RFC-0016 'Calculators': from a policy, a traffic profile and the
//! cost model's `μ` per bucket, the probability that violating requests are detected, the
//! amortized overhead, and the rates that reach a target overhead.
//!
//! For bucket `b` of the policy, with `n_b` requests of mean length `L_b`, `v_b` of them
//! violating, rate `s_b` and multiplier `μ_b`:
//!
//! - **Detection** is `1 − Π_b (1 − s̲_b)^{v_b}`, with `s̲_b = min_{c ≥ b} s_c` the lowest rate of
//!   bucket `b` and every later one. The serving stack can pad a request it cheats on into a
//!   later bucket, so a violation of bucket `b` is detected at `s̲_b` at best. A request padded
//!   past the last bucket is always selected, so the minimum stops at the last bucket.
//! - **Detection in place** is `1 − Π_b (1 − s_b)^{v_b}`: what the policy detects if no violating
//!   request leaves its bucket. It is an upper bound, labelled as such.
//! - **The amortized overhead** is `Σ_b n_b·L_b·s_b·μ_b·η / Σ_b n_b·L_b`, with `η` the ratio of
//!   ideal prefill time to serving time per token (00 §10).
//! - **The target rate** of a bucket is `min(1, O / (μ_b·η))`, the rate that alone gives the
//!   overhead `O` at the bucket's context, as the source's §8 table sets it.
//!
//! The arithmetic is IEEE 754 binary64, with `ln_1p` and `exp_m1` so that tiny rates keep their
//! precision. These are planning figures. None enters the sample, whose thresholds stay exact
//! integers ([`RateBucket::threshold_of`](crate::RateBucket::threshold_of)), or any TCE artifact
//! (INV-DM-6). Every value here is public (03 §5).

use core::fmt;

use crate::audit::AuditPolicy;

/// `2^64`, the scale of a bucket's threshold.
const TWO_64: f64 = 18_446_744_073_709_551_616.0;

/// The traffic of one bucket of the policy over a period (RFC-0016 'Calculators').
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BucketTraffic {
    /// The requests whose length falls in the bucket, `n_b`.
    pub requests: u64,
    /// Their mean length in tokens, `L_b`: above the previous bucket's `max_tokens` and at most
    /// the bucket's own.
    pub tokens: u64,
    /// The violating requests among them, `v_b ≤ n_b`.
    pub violations: u64,
}

/// The plan of one bucket.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BucketPlan {
    /// The bucket's `max_tokens`.
    pub max_tokens: u32,
    /// Its rate `s_b`, the threshold over `2^64`.
    pub rate: f64,
    /// `s̲_b = min_{c ≥ b} s_c`, the rate its violations are detected at against padding.
    pub lowest_later_rate: f64,
    /// Its traffic.
    pub traffic: BucketTraffic,
    /// The cost model's `μ_b` at the bucket's mean length.
    pub mu: f64,
    /// The requests the sample is expected to select, `n_b·s_b`.
    pub expected_selected: f64,
    /// `min(1, O / (μ_b·η))`.
    pub target_rate: f64,
    /// `1 − (1 − s̲_b)^{v_b}`: at least one of its violations selected, against padding.
    pub detection: f64,
    /// `1 − (1 − s_b)^{v_b}`: the same if none leaves the bucket.
    pub detection_in_place: f64,
}

/// What `tessera audit plan` reports (RFC-0016 'Calculators').
#[derive(Clone, Debug, PartialEq)]
pub struct AuditPlan {
    /// The ratio of ideal prefill time to serving time per token, `η`.
    pub eta: f64,
    /// The overhead the target rates are set for, `O`.
    pub target_overhead: f64,
    /// One plan per bucket of the policy, in its order.
    pub buckets: Vec<BucketPlan>,
    /// `1 − Π_b (1 − s̲_b)^{v_b}`.
    pub detection: f64,
    /// `1 − Π_b (1 − s_b)^{v_b}`.
    pub detection_in_place: f64,
    /// `Σ_b n_b·L_b·s_b·μ_b·η / Σ_b n_b·L_b`.
    pub overhead: f64,
    /// `Σ_b n_b·s_b`.
    pub expected_selected: f64,
}

/// An input the calculators refuse, `ERR-USAGE` at the command line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanError {
    /// The traffic profile, or the `μ` values, do not give one entry per bucket of the policy.
    Buckets {
        /// The policy's buckets.
        policy: usize,
        /// The entries given.
        given: usize,
    },
    /// Bucket `bucket`'s mean length is not above the previous bucket's `max_tokens` and at
    /// most its own.
    Tokens {
        /// The bucket, from 0.
        bucket: usize,
    },
    /// Bucket `bucket` has more violating requests than requests.
    Violations {
        /// The bucket, from 0.
        bucket: usize,
    },
    /// No bucket has a request: the overhead is undefined.
    NoTraffic,
    /// `η`, the target overhead or a bucket's `μ` is not finite and positive.
    Input {
        /// The input's name.
        name: &'static str,
    },
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Buckets { policy, given } => write!(
                f,
                "the policy has {policy} buckets, and {given} entries were given for them"
            ),
            Self::Tokens { bucket } => write!(
                f,
                "bucket {bucket}'s mean length does not lie in the bucket"
            ),
            Self::Violations { bucket } => write!(
                f,
                "bucket {bucket} has more violating requests than requests"
            ),
            Self::NoTraffic => f.write_str("the traffic has no request"),
            Self::Input { name } => write!(f, "{name} is not finite and positive"),
        }
    }
}

impl std::error::Error for PlanError {}

/// A finite, positive input, else [`PlanError::Input`].
fn positive(name: &'static str, value: f64) -> Result<f64, PlanError> {
    match value.is_finite() && value > 0.0 {
        true => Ok(value),
        false => Err(PlanError::Input { name }),
    }
}

/// `ln (1 − s)^v`, `v` violations at rate `s`; 0 for no violation, `−∞` for `s = 1`.
fn log_missed(rate: f64, violations: u64) -> f64 {
    match violations {
        0 => 0.0,
        v => v as f64 * (-rate).ln_1p(),
    }
}

/// `1 − e^x` for the log of the probability that every violation is missed.
fn detected(log_missed: f64) -> f64 {
    -log_missed.exp_m1()
}

/// Checks a traffic profile against `policy`, before any `μ` is computed for it: one entry per
/// bucket, each mean length in its bucket, no more violations than requests, and a request in
/// some bucket.
///
/// # Errors
/// [`PlanError::Buckets`], [`PlanError::Tokens`], [`PlanError::Violations`] or
/// [`PlanError::NoTraffic`], the first that applies, buckets in order.
pub fn check_traffic(policy: &AuditPolicy, traffic: &[BucketTraffic]) -> Result<(), PlanError> {
    let buckets = policy.buckets();
    if traffic.len() != buckets.len() {
        return Err(PlanError::Buckets {
            policy: buckets.len(),
            given: traffic.len(),
        });
    }
    let mut floor = 0u64;
    for (bucket, (t, b)) in traffic.iter().zip(buckets).enumerate() {
        if t.tokens <= floor || t.tokens > u64::from(b.max_tokens) {
            return Err(PlanError::Tokens { bucket });
        }
        if t.violations > t.requests {
            return Err(PlanError::Violations { bucket });
        }
        floor = u64::from(b.max_tokens);
    }
    // Every mean length is at least 1, so tokens are served exactly when a request is.
    match traffic.iter().all(|t| t.requests == 0) {
        true => Err(PlanError::NoTraffic),
        false => Ok(()),
    }
}

impl AuditPlan {
    /// The plan of `policy` at `traffic` and `mu`, one entry per bucket of the policy, with `η`
    /// `eta` and the target overhead `target_overhead`.
    ///
    /// # Errors
    /// [`check_traffic`]'s errors, then [`PlanError::Buckets`] for `μ` values that are not one
    /// per bucket, and [`PlanError::Input`] for `η`, the target overhead or a `μ` that is not
    /// finite and positive.
    pub fn new(
        policy: &AuditPolicy,
        traffic: &[BucketTraffic],
        mu: &[f64],
        eta: f64,
        target_overhead: f64,
    ) -> Result<Self, PlanError> {
        check_traffic(policy, traffic)?;
        let buckets = policy.buckets();
        if mu.len() != buckets.len() {
            return Err(PlanError::Buckets {
                policy: buckets.len(),
                given: mu.len(),
            });
        }
        let eta = positive("η", eta)?;
        let target_overhead = positive("the target overhead", target_overhead)?;
        let rates: Vec<f64> = buckets
            .iter()
            .map(|b| b.threshold as f64 / TWO_64)
            .collect();
        // s̲_b, from the last bucket back.
        let mut lowest = rates.clone();
        for b in (0..lowest.len().saturating_sub(1)).rev() {
            lowest[b] = lowest[b].min(lowest[b + 1]);
        }
        let mut plans = Vec::with_capacity(buckets.len());
        let (mut missed, mut missed_in_place) = (0.0, 0.0);
        let (mut cost, mut tokens, mut expected_selected) = (0.0, 0.0, 0.0);
        for (b, bucket) in buckets.iter().enumerate() {
            let (t, s, s_low) = (traffic[b], rates[b], lowest[b]);
            let mu_b = positive("μ", mu[b])?;
            let (log_low, log_place) =
                (log_missed(s_low, t.violations), log_missed(s, t.violations));
            missed += log_low;
            missed_in_place += log_place;
            let volume = t.requests as f64 * t.tokens as f64;
            cost += volume * s * mu_b * eta;
            tokens += volume;
            expected_selected += t.requests as f64 * s;
            plans.push(BucketPlan {
                max_tokens: bucket.max_tokens,
                rate: s,
                lowest_later_rate: s_low,
                traffic: t,
                mu: mu_b,
                expected_selected: t.requests as f64 * s,
                target_rate: (target_overhead / (mu_b * eta)).min(1.0),
                detection: detected(log_low),
                detection_in_place: detected(log_place),
            });
        }
        Ok(Self {
            eta,
            target_overhead,
            buckets: plans,
            detection: detected(missed),
            detection_in_place: detected(missed_in_place),
            overhead: cost / tokens,
            expected_selected,
        })
    }
}
