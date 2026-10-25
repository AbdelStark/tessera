// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The audit policy (RFC-0016 'Policy'): its rules, its TCE encoding, and a policy file's decimal
//! rates as exact thresholds over `2^64`.

use core::fmt;

use tv_codec::{Cursor, DecodeError, ErrorCode, size};

use crate::claims::ParamSetId;

/// A length bucket: the longest request it holds and its rate as a threshold over `2^64`
/// (RFC-0016 'Policy'). TCE: the `u32`, then the `u64`, 12 bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RateBucket {
    /// The longest request the bucket holds, in total tokens over both directions and both
    /// periods of its window (RFC-0063).
    pub max_tokens: u32,
    /// The rate: a request of the bucket is selected when its selection value is below it, with
    /// probability `threshold / 2^64`.
    pub threshold: u64,
}

impl RateBucket {
    /// The encoded size, the smallest an element of `buckets` takes.
    pub const ENCODED_LEN: u64 = 12;

    /// The threshold of a policy file's decimal rate (RFC-0016 'Policy files'): `0`, `1`, or
    /// `0.` or `1.` followed by one to 19 digits, at most 1; its threshold is `⌊rate · 2^64⌋`,
    /// capped at `2^64 − 1`, computed exactly. `None` for any other text.
    ///
    /// A fraction `0.d₁…d_k` is `D / 10^k` with `D < 10^k ≤ 10^19 < 2^64`, so `⌊D · 2^64 / 10^k⌋`
    /// is one division of a 128-bit integer, and below `2^64`. A rate of 1, written `1` or `1.`
    /// followed by zeros, is the cap itself.
    pub fn threshold_of(rate: &str) -> Option<u64> {
        match rate.as_bytes() {
            b"0" => Some(0),
            b"1" => Some(u64::MAX),
            [whole @ (b'0' | b'1'), b'.', digits @ ..]
                if (1..=19).contains(&digits.len()) && digits.iter().all(u8::is_ascii_digit) =>
            {
                if *whole == b'1' {
                    // At most 1: every digit of the fraction is 0.
                    return digits.iter().all(|&d| d == b'0').then_some(u64::MAX);
                }
                let mut numerator: u64 = 0;
                let mut denominator: u64 = 1;
                for &digit in digits {
                    numerator = numerator
                        .checked_mul(10)?
                        .checked_add(u64::from(digit.checked_sub(b'0')?))?;
                    denominator = denominator.checked_mul(10)?;
                }
                let scaled = u128::from(numerator).checked_shl(64)?;
                u64::try_from(scaled.checked_div(u128::from(denominator))?).ok()
            }
            _ => None,
        }
    }
}

/// A rule of RFC-0016 'Policy' that a policy breaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PolicyError {
    /// Bucket `index`'s `max_tokens` is not above the previous bucket's.
    BucketOrder {
        /// The first bucket, in order, that breaks the rule.
        index: usize,
    },
    /// `max_batch_tokens` is 0, or below `2 · min_batch_tokens`, which 'Batching' needs.
    BatchBounds,
}

impl fmt::Display for PolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BucketOrder { index } => write!(
                f,
                "bucket {index}'s max_tokens is not above the previous bucket's"
            ),
            Self::BatchBounds => {
                f.write_str("max_batch_tokens is 0 or below twice min_batch_tokens")
            }
        }
    }
}

impl std::error::Error for PolicyError {}

/// An audit policy (RFC-0016 'Policy'), whose rules hold: buckets strictly increasing by
/// `max_tokens`, and `max_batch_tokens` at least 1 and at least `2 · min_batch_tokens`.
/// [`new`](Self::new) and the decoder are its only constructors.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AuditPolicy {
    period_ns: u64,
    buckets: Vec<RateBucket>,
    min_batch_tokens: u32,
    max_batch_tokens: u32,
    proof_deadline_ns: u64,
    params: ParamSetId,
}

impl AuditPolicy {
    /// A policy from its fields, in RFC-0016's order.
    ///
    /// # Errors
    ///
    /// The first rule the fields break, in field order: a bucket whose `max_tokens` is not above
    /// the previous bucket's, then the batch bounds.
    pub fn new(
        period_ns: u64,
        buckets: Vec<RateBucket>,
        min_batch_tokens: u32,
        max_batch_tokens: u32,
        proof_deadline_ns: u64,
        params: ParamSetId,
    ) -> Result<Self, PolicyError> {
        for (index, pair) in buckets.windows(2).enumerate() {
            if let [before, after] = pair
                && after.max_tokens <= before.max_tokens
            {
                return Err(PolicyError::BucketOrder {
                    index: index.saturating_add(1),
                });
            }
        }
        if !batch_bounds_hold(min_batch_tokens, max_batch_tokens) {
            return Err(PolicyError::BatchBounds);
        }
        Ok(Self {
            period_ns,
            buckets,
            min_batch_tokens,
            max_batch_tokens,
            proof_deadline_ns,
            params,
        })
    }

    /// The recorder's period, in nanoseconds.
    pub fn period_ns(&self) -> u64 {
        self.period_ns
    }

    /// The length buckets, `max_tokens` strictly increasing.
    pub fn buckets(&self) -> &[RateBucket] {
        &self.buckets
    }

    /// The fewest tokens a batch proves, but for at most one batch of each group.
    pub fn min_batch_tokens(&self) -> u32 {
        self.min_batch_tokens
    }

    /// The most tokens a batch proves, but for a request longer than that alone.
    pub fn max_batch_tokens(&self) -> u32 {
        self.max_batch_tokens
    }

    /// The time the prover has to prove the sample once it is fixed, in nanoseconds.
    pub fn proof_deadline_ns(&self) -> u64 {
        self.proof_deadline_ns
    }

    /// The parameter set the sample's proofs use.
    pub fn params(&self) -> &ParamSetId {
        &self.params
    }

    /// The threshold of `bucket(tokens)`, the first bucket whose `max_tokens` is at least
    /// `tokens`; `None` when `tokens` is past every bucket, which selects the request whatever
    /// the seed (RFC-0016 'Sample derivation' step 4). The buckets increase strictly, so a
    /// binary search finds it.
    pub fn bucket_threshold(&self, tokens: u64) -> Option<u64> {
        let index = self
            .buckets
            .partition_point(|b| u64::from(b.max_tokens) < tokens);
        self.buckets.get(index).map(|b| b.threshold)
    }

    /// Whether a candidate of `tokens` tokens and selection value `value` is selected: `value`
    /// below its bucket's threshold, or past every bucket.
    pub fn selects(&self, value: u64, tokens: u64) -> bool {
        self.bucket_threshold(tokens).is_none_or(|t| value < t)
    }

    /// Decodes a policy, its fields in order (RFC-0003, RFC-0016 'Policy' 'Encoding'). Beyond
    /// RFC-0003's checks:
    ///
    /// - a bucket whose `max_tokens` is not above the previous bucket's is
    ///   `ERR-CODEC-NONCANONICAL` at the bucket's first byte, read before its threshold;
    /// - a `max_batch_tokens` of 0, or below `2 · min_batch_tokens`, is `ERR-CODEC-LIMIT` at
    ///   `max_batch_tokens`.
    pub(crate) fn decode(c: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        c.nested(|c| {
            let period_ns = c.u64()?;
            let mut previous: Option<u32> = None;
            let buckets = c.vec(RateBucket::ENCODED_LEN, |c| {
                c.nested(|c| {
                    let at = c.offset();
                    let max_tokens = c.u32()?;
                    if previous.is_some_and(|p| max_tokens <= p) {
                        return Err(DecodeError::new(ErrorCode::Noncanonical, at));
                    }
                    previous = Some(max_tokens);
                    Ok(RateBucket {
                        max_tokens,
                        threshold: c.u64()?,
                    })
                })
            })?;
            let min_batch_tokens = c.u32()?;
            let at = c.offset();
            let max_batch_tokens = c.u32()?;
            if !batch_bounds_hold(min_batch_tokens, max_batch_tokens) {
                return Err(DecodeError::new(ErrorCode::Limit, at));
            }
            let proof_deadline_ns = c.u64()?;
            let params = ParamSetId::decode(c)?;
            Ok(Self {
                period_ns,
                buckets,
                min_batch_tokens,
                max_batch_tokens,
                proof_deadline_ns,
                params,
            })
        })
    }

    /// The encoded size: 8 + 8 + 12 per bucket + 4 + 4 + 8 + 8 + the name's bytes. `None` past
    /// `u64`.
    pub(crate) fn encoded_len(&self) -> Option<u64> {
        let buckets = u64::try_from(self.buckets.len())
            .ok()?
            .checked_mul(RateBucket::ENCODED_LEN)?;
        let name = u64::try_from(self.params.as_str().len()).ok()?;
        size::sum(&[
            size::U64,
            size::SEQUENCE,
            size::U32,
            size::U32,
            size::U64,
            size::SEQUENCE,
        ])
        .checked_add(buckets)?
        .checked_add(name)
    }

    /// Appends the policy's encoding, its fields in order.
    pub(crate) fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.period_ns.to_le_bytes());
        put_len(out, self.buckets.len());
        for bucket in &self.buckets {
            out.extend_from_slice(&bucket.max_tokens.to_le_bytes());
            out.extend_from_slice(&bucket.threshold.to_le_bytes());
        }
        out.extend_from_slice(&self.min_batch_tokens.to_le_bytes());
        out.extend_from_slice(&self.max_batch_tokens.to_le_bytes());
        out.extend_from_slice(&self.proof_deadline_ns.to_le_bytes());
        let name = self.params.as_str().as_bytes();
        put_len(out, name.len());
        out.extend_from_slice(name);
    }
}

/// `max_batch_tokens ≥ 1` and `max_batch_tokens ≥ 2 · min_batch_tokens`, in 64 bits so that the
/// doubling cannot wrap.
fn batch_bounds_hold(min_batch_tokens: u32, max_batch_tokens: u32) -> bool {
    let twice = u64::from(min_batch_tokens).saturating_mul(2);
    max_batch_tokens != 0 && u64::from(max_batch_tokens) >= twice
}

/// A sequence's `u64` length.
pub(crate) fn put_len(out: &mut Vec<u8>, len: usize) {
    out.extend_from_slice(&u64::try_from(len).unwrap_or(u64::MAX).to_le_bytes());
}
