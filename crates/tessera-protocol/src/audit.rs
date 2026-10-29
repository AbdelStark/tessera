// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Audit sampling (RFC-0016): the policy, the coin toss over a period's manifest, the selection
//! rule, the [`AuditSample`] artifact (schema `0x0403`) and the batching of what it selects.
//!
//! - **The coin toss** ([`CoinTossTranscript`]). The verifier commits to a fresh share `s_V`,
//!   the prover answers with a fresh share `s_P`, and the verifier reveals `s_V` once the
//!   manifest's signature has passed and it holds the period's verified log and the previous
//!   one's. The prover answers one commitment per manifest ([`ShareRecord::answer`]), so a
//!   verifier that aborts after seeing `s_P` draws no second sample, and draws its own sample from
//!   its record and `s_V` alone ([`ShareRecord::draw`]), so a verifier cannot present the shares
//!   of another toss.
//! - **The seed.** `sample_seed = H("audit-sample", s_V ‖ s_P ‖ manifest_digest)`
//!   ([`sample_seed`]).
//! - **Selection.** A candidate of length `L` is selected iff its [`selection_value`] is below
//!   the threshold of `L`'s bucket ([`AuditPolicy::threshold`]). A candidate longer than every
//!   bucket is selected whatever the seed, so no length escapes the sample.
//! - **The candidates.** `tessera-anchor` reads them and their lengths from the period's verified
//!   log and the previous period's (RFC-0063). The anchor is a sibling of this crate (01 §3), so
//!   the functions here take them as `(RequestKey, L)` pairs.
//! - **Batching** ([`batches`]). First-fit decreasing on tokens, group by group, as RFC-0016
//!   'Batching' fixes it. A temperature request's group is the pinned sampler key that applies to
//!   it (RFC-0061 'Which pin applies').
//!
//! Every value here is public (03 §5): shares, seeds, policies and request identifiers.
//!
//! The decoders refuse what RFC-0016 'Policy' and 'Sample derivation' rule out, on top of
//! RFC-0003's checks:
//! - buckets whose `max_tokens` do not strictly increase: `ERR-CODEC-NONCANONICAL` at the
//!   bucket's first byte;
//! - `max_batch_tokens` of 0, or below twice `min_batch_tokens`: `ERR-CODEC-LIMIT` at
//!   `max_batch_tokens`;
//! - `selected` not strictly increasing: `ERR-CODEC-NONCANONICAL` at the key's first byte.

use core::fmt;

use tessera_codec::{Decode, DecodeError, Encode, Reader};
use tessera_hash::{Digest, Domain, Hasher, ParamSetId};

use crate::seed::coin_commitment;

/// A request of the log: its session, then its identifier within the session (RFC-0016). TCE:
/// the 16 bytes, then a `u64`. Requests order by `session_id`, then `request_id`, the order of
/// [`AuditSample::selected`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Encode, Decode)]
pub struct RequestKey {
    /// The session.
    pub session_id: [u8; 16],
    /// The request within its session.
    pub request_id: u64,
}

/// One length bucket of a policy: the requests of at most `max_tokens` tokens that no earlier
/// bucket holds, each selected with probability `threshold / 2^64`. TCE: `u32`, then `u64`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub struct RateBucket {
    /// The longest request the bucket holds, in total tokens over both directions and both
    /// periods (RFC-0063).
    pub max_tokens: u32,
    /// The rate as a threshold over `2^64`: a request is selected iff its selection value is
    /// below it. A policy file gives it as a decimal rate ([`RateBucket::threshold_of`]).
    pub threshold: u64,
}

impl RateBucket {
    /// The threshold of a rate a policy file writes as a decimal (RFC-0016 'Policy'):
    /// `⌊rate · 2^64⌋`, capped at `2^64 − 1`. The rate is `0`, `1`, or `0.` or `1.` followed by
    /// one to 19 digits, and at most 1. The arithmetic is exact: no floating point enters the
    /// threshold (INV-DM-6).
    ///
    /// # Errors
    /// [`RateError`] for any other text.
    pub fn threshold_of(rate: &str) -> Result<u64, RateError> {
        let (whole, fraction) = match rate.split_once('.') {
            Some((w, f)) => (w, Some(f)),
            None => (rate, None),
        };
        let whole: u128 = match whole {
            "0" => 0,
            "1" => 1,
            _ => return Err(RateError),
        };
        let (digits, scale) = match fraction {
            None => (0u128, 1u128),
            Some(f) => {
                if f.is_empty() || f.len() > 19 || !f.bytes().all(|b| b.is_ascii_digit()) {
                    return Err(RateError);
                }
                // At most 19 digits: below 10^19, and 10^19 · 2^64 < 2^128.
                let digits = f.bytes().fold(0u128, |n, b| n * 10 + u128::from(b - b'0'));
                (digits, 10u128.pow(f.len() as u32))
            }
        };
        if whole == 1 && digits != 0 {
            return Err(RateError);
        }
        if whole == 1 {
            return Ok(u64::MAX);
        }
        // ⌊digits / scale · 2^64⌋ < 2^64, since digits < scale.
        Ok(((digits << 64) / scale) as u64)
    }
}

/// A rate that is not a decimal in `[0, 1]` of the form [`RateBucket::threshold_of`] reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RateError;

impl fmt::Display for RateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a rate is 0, 1, or 0 or 1 with a point and one to 19 digits, at most 1")
    }
}

impl std::error::Error for RateError {}

/// What the parties agree to audit (RFC-0016 'Policy'). TCE: the fields in order.
///
/// A policy is valid by construction: [`AuditPolicy::new`] and the decoder both refuse buckets
/// whose `max_tokens` do not strictly increase, and batch bounds without `1 ≤ max` and
/// `2 · min ≤ max`, which 'Batching' needs.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Encode)]
pub struct AuditPolicy {
    period_ns: u64,
    buckets: Vec<RateBucket>,
    min_batch_tokens: u32,
    max_batch_tokens: u32,
    proof_deadline_ns: u64,
    params: ParamSetId,
}

/// The rule of RFC-0016 'Policy' that `buckets` and the batch bounds break, if any.
fn broken_rule(
    buckets: &[RateBucket],
    min_batch_tokens: u32,
    max_batch_tokens: u32,
) -> Option<&'static str> {
    if buckets
        .windows(2)
        .any(|w| w[1].max_tokens <= w[0].max_tokens)
    {
        return Some("the buckets' max_tokens strictly increase");
    }
    if max_batch_tokens == 0 || 2 * u64::from(min_batch_tokens) > u64::from(max_batch_tokens) {
        return Some("max_batch_tokens is at least 1 and twice min_batch_tokens");
    }
    None
}

impl AuditPolicy {
    /// A policy, from its fields in order.
    ///
    /// # Errors
    /// [`ProtocolError::Policy`](crate::ProtocolError::Policy) (`ERR-USAGE`) for buckets out of
    /// order or batch bounds without `1 ≤ max_batch_tokens` and `2 · min ≤ max`.
    pub fn new(
        period_ns: u64,
        buckets: Vec<RateBucket>,
        min_batch_tokens: u32,
        max_batch_tokens: u32,
        proof_deadline_ns: u64,
        params: ParamSetId,
    ) -> Result<Self, crate::ProtocolError> {
        if let Some(rule) = broken_rule(&buckets, min_batch_tokens, max_batch_tokens) {
            return Err(crate::ProtocolError::Policy { rule });
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

    /// The recorder's period, for example 24 hours.
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

    /// The most tokens a batch proves, but for a request longer than that, which is a batch on
    /// its own.
    pub fn max_batch_tokens(&self) -> u32 {
        self.max_batch_tokens
    }

    /// The time the prover has to prove the sample, after it is fixed.
    pub fn proof_deadline_ns(&self) -> u64 {
        self.proof_deadline_ns
    }

    /// The parameter set the proofs use.
    pub fn params(&self) -> &ParamSetId {
        &self.params
    }

    /// The threshold that applies to a candidate of `tokens` tokens: its bucket's, the first whose
    /// `max_tokens` is at least `tokens`. `None` when `tokens` is past every bucket: such a
    /// candidate is selected whatever the seed.
    pub fn threshold(&self, tokens: u64) -> Option<u64> {
        self.buckets
            .iter()
            .find(|b| tokens <= u64::from(b.max_tokens))
            .map(|b| b.threshold)
    }

    /// Whether a candidate is selected under `sample_seed` (RFC-0016 'Sample derivation',
    /// step 4).
    pub fn selects(&self, sample_seed: &Digest, key: &RequestKey, tokens: u64) -> bool {
        match self.threshold(tokens) {
            None => true,
            Some(threshold) => selection_value(sample_seed, key) < threshold,
        }
    }
}

impl Decode for AuditPolicy {
    // period_ns, an empty bucket list, the two batch bounds, the deadline and an empty name.
    const MIN_SIZE: usize = 8 + 8 + 4 + 4 + 8 + 8;

    fn decode_from(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        reader.enter()?;
        let period_ns = u64::decode_from(reader)?;
        reader.enter()?;
        let n = reader.sequence_len(RateBucket::MIN_SIZE)?;
        let mut buckets: Vec<RateBucket> =
            Vec::with_capacity(n.min(reader.remaining() / RateBucket::MIN_SIZE));
        for _ in 0..n {
            let at = reader.position();
            let bucket = RateBucket::decode_from(reader)?;
            if buckets
                .last()
                .is_some_and(|b| bucket.max_tokens <= b.max_tokens)
            {
                return Err(reader.noncanonical(at));
            }
            buckets.push(bucket);
        }
        reader.leave();
        let min_batch_tokens = u32::decode_from(reader)?;
        let at = reader.position();
        let max_batch_tokens = u32::decode_from(reader)?;
        if max_batch_tokens == 0 || 2 * u64::from(min_batch_tokens) > u64::from(max_batch_tokens) {
            return Err(reader.limit(at));
        }
        let proof_deadline_ns = u64::decode_from(reader)?;
        let params = ParamSetId::decode_from(reader)?;
        reader.leave();
        Ok(Self {
            period_ns,
            buckets,
            min_batch_tokens,
            max_batch_tokens,
            proof_deadline_ns,
            params,
        })
    }
}

/// The coin toss over a manifest, as both parties keep it (RFC-0016 'Sample derivation',
/// step 2). TCE: `c_V`, `s_P`, `s_V`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub struct CoinTossTranscript {
    /// `c_V = H("coin-commit", s_V)`, which the verifier sent first.
    pub verifier_commitment: Digest,
    /// `s_P`, the prover's answer to the commitment.
    pub prover_share: [u8; 32],
    /// `s_V`, revealed last.
    pub verifier_share: [u8; 32],
}

impl CoinTossTranscript {
    /// Checks that the revealed share opens the commitment.
    ///
    /// # Errors
    /// [`ProtocolError::CoinToss`](crate::ProtocolError::CoinToss) (`ERR-PROTO-COINTOSS`) when
    /// it does not.
    pub fn check(&self) -> Result<(), crate::ProtocolError> {
        if coin_commitment(&self.verifier_share) == self.verifier_commitment {
            Ok(())
        } else {
            Err(crate::ProtocolError::CoinToss)
        }
    }
}

/// What the prover keeps of its answer to the coin toss over a manifest (RFC-0016 'Sample
/// derivation', step 2): the manifest, the policy, the commitment it answered and its share. TCE:
/// the fields in order. Nothing in it is secret: the prover sends its share once the record is
/// kept.
///
/// The record binds the toss twice:
/// - **one commitment per manifest**: [`ShareRecord::answer`] answers no other commitment and no
///   other policy for the manifest;
/// - **one transcript per manifest**: [`ShareRecord::draw`] builds the transcript from the record
///   and the verifier's revealed share alone. A verifier holding shares from other tosses, of other
///   manifests or of none, cannot present one of them for this manifest.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub struct ShareRecord {
    /// The identity of the manifest the toss is over.
    pub manifest_digest: Digest,
    /// The policy, fixed before the toss.
    pub policy: AuditPolicy,
    /// The commitment `c_V` the prover answered.
    pub verifier_commitment: Digest,
    /// The share `s_P` it answered with.
    pub prover_share: [u8; 32],
}

impl ShareRecord {
    /// The prover's answer to the commitment `verifier_commitment` over `manifest_digest` under
    /// `policy`, given the record it kept of an earlier answer for that manifest, if any:
    /// - none: a fresh `s_P` from the operating system's CSPRNG, never a function of the
    ///   manifest. The caller keeps the record durably, where every responder for this prover
    ///   finds it, before it sends the share.
    /// - a record of the same commitment and policy: that record, so the same share again, for a
    ///   verifier that lost the first answer.
    ///
    /// # Errors
    /// - [`ProtocolError::Answered`](crate::ProtocolError::Answered) (`ERR-PROTO-ORDER`) for a
    ///   record of another commitment or another policy: a verifier that aborted after seeing
    ///   `s_P` draws no second sample.
    /// - [`ProtocolError::Internal`](crate::ProtocolError::Internal) for a record of another
    ///   manifest, which the caller should not have given.
    /// - [`ProtocolError::Randomness`](crate::ProtocolError::Randomness) when the generator fails.
    pub fn answer(
        kept: Option<&ShareRecord>,
        manifest_digest: Digest,
        policy: &AuditPolicy,
        verifier_commitment: Digest,
    ) -> Result<Self, crate::ProtocolError> {
        match kept {
            Some(record) if record.manifest_digest != manifest_digest => {
                Err(crate::ProtocolError::Internal {
                    rule: "a share record answers the manifest it is kept for",
                })
            }
            Some(record)
                if record.verifier_commitment == verifier_commitment
                    && record.policy == *policy =>
            {
                Ok(record.clone())
            }
            Some(_) => Err(crate::ProtocolError::Answered),
            None => {
                let mut prover_share = [0u8; 32];
                getrandom::fill(&mut prover_share).map_err(|_| crate::ProtocolError::Randomness)?;
                Ok(Self {
                    manifest_digest,
                    policy: policy.clone(),
                    verifier_commitment,
                    prover_share,
                })
            }
        }
    }

    /// The sample of the record's manifest under its policy, once the verifier has revealed
    /// `verifier_share`: the transcript is the record's commitment and share with it, and no
    /// other.
    ///
    /// # Errors
    /// [`ProtocolError::CoinToss`](crate::ProtocolError::CoinToss) (`ERR-PROTO-COINTOSS`) when
    /// `verifier_share` does not open the record's commitment.
    pub fn draw(
        &self,
        verifier_share: [u8; 32],
        candidates: impl IntoIterator<Item = (RequestKey, u64)>,
    ) -> Result<AuditSample, crate::ProtocolError> {
        let coin_toss = CoinTossTranscript {
            verifier_commitment: self.verifier_commitment,
            prover_share: self.prover_share,
            verifier_share,
        };
        AuditSample::draw(
            self.manifest_digest,
            self.policy.clone(),
            coin_toss,
            candidates,
        )
    }
}

/// `sample_seed = H("audit-sample", s_V ‖ s_P ‖ manifest_digest)` (RFC-0016 step 3).
pub fn sample_seed(coin_toss: &CoinTossTranscript, manifest_digest: &Digest) -> Digest {
    Hasher::new(Domain::AuditSample)
        .update(&coin_toss.verifier_share)
        .update(&coin_toss.prover_share)
        .update(manifest_digest)
        .finalize()
}

/// A request's selection value, `u64_le(H("audit-select", sample_seed ‖ session_id ‖
/// u64(request_id))[0..8])` (RFC-0016 step 4).
pub fn selection_value(sample_seed: &Digest, key: &RequestKey) -> u64 {
    let digest = Hasher::new(Domain::AuditSelect)
        .update(sample_seed)
        .update(&key.session_id)
        .update(&key.request_id.to_le_bytes())
        .finalize();
    let mut low = [0u8; 8];
    low.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(low)
}

/// The audit sample of a period (schema `0x0403`, version 1, `.tas`; RFC-0016 step 5). TCE: the
/// manifest's digest, the policy, the coin toss, then the selected requests.
///
/// A sample comes from [`AuditSample::draw`] or the decoder, so its selected requests strictly
/// increase and its policy is valid. It does not record the candidates, so whether its selection
/// is the policy's under its seed is for a reader with the period's logs to check.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Encode)]
pub struct AuditSample {
    manifest_digest: Digest,
    policy: AuditPolicy,
    coin_toss: CoinTossTranscript,
    selected: Vec<RequestKey>,
}

impl AuditSample {
    /// Draws the sample of a period: checks the coin toss, derives the seed and selects among
    /// `candidates`, the period's candidates with their lengths (RFC-0063). Their order does not
    /// matter; a key given twice is selected if either of its lengths selects it.
    ///
    /// # Errors
    /// [`ProtocolError::CoinToss`](crate::ProtocolError::CoinToss) when `s_V` does not open
    /// `c_V`.
    pub fn draw(
        manifest_digest: Digest,
        policy: AuditPolicy,
        coin_toss: CoinTossTranscript,
        candidates: impl IntoIterator<Item = (RequestKey, u64)>,
    ) -> Result<Self, crate::ProtocolError> {
        coin_toss.check()?;
        let seed = sample_seed(&coin_toss, &manifest_digest);
        let mut selected: Vec<RequestKey> = candidates
            .into_iter()
            .filter(|(key, tokens)| policy.selects(&seed, key, *tokens))
            .map(|(key, _)| key)
            .collect();
        selected.sort_unstable();
        selected.dedup();
        Ok(Self {
            manifest_digest,
            policy,
            coin_toss,
            selected,
        })
    }

    /// The sample's seed.
    pub fn seed(&self) -> Digest {
        sample_seed(&self.coin_toss, &self.manifest_digest)
    }

    /// The identity of the period's manifest (RFC-0060).
    pub fn manifest_digest(&self) -> &Digest {
        &self.manifest_digest
    }

    /// The policy the sample was drawn under.
    pub fn policy(&self) -> &AuditPolicy {
        &self.policy
    }

    /// The coin toss the seed came from.
    pub fn coin_toss(&self) -> &CoinTossTranscript {
        &self.coin_toss
    }

    /// The selected requests, strictly increasing.
    pub fn selected(&self) -> &[RequestKey] {
        &self.selected
    }
}

impl Decode for AuditSample {
    const MIN_SIZE: usize = 32 + AuditPolicy::MIN_SIZE + CoinTossTranscript::MIN_SIZE + 8;

    fn decode_from(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        reader.enter()?;
        let manifest_digest = <Digest>::decode_from(reader)?;
        let policy = AuditPolicy::decode_from(reader)?;
        let coin_toss = CoinTossTranscript::decode_from(reader)?;
        reader.enter()?;
        let n = reader.sequence_len(RequestKey::MIN_SIZE)?;
        let mut selected: Vec<RequestKey> =
            Vec::with_capacity(n.min(reader.remaining() / RequestKey::MIN_SIZE));
        for _ in 0..n {
            let at = reader.position();
            let key = RequestKey::decode_from(reader)?;
            if selected.last().is_some_and(|last| key <= *last) {
                return Err(reader.noncanonical(at));
            }
            selected.push(key);
        }
        reader.leave();
        reader.leave();
        Ok(Self {
            manifest_digest,
            policy,
            coin_toss,
            selected,
        })
    }
}

/// One batch: one proof's requests, in [`RequestKey`] order, which its `PublicStatement.batch`
/// lists (RFC-0016 'Batching').
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Batch<G> {
    /// The group the batch was packed in.
    pub group: G,
    /// Its requests, strictly increasing.
    pub requests: Vec<RequestKey>,
    /// Their total tokens.
    pub tokens: u64,
}

/// A request given to [`batches`] twice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RepeatedRequest(pub RequestKey);

impl fmt::Display for RepeatedRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "request {} of a session is given twice",
            self.0.request_id
        )
    }
}

impl std::error::Error for RepeatedRequest {}

/// The slot class `c = ⌈log2 max(L, 1)⌉` of a request with `tokens = L`
/// (RFC-0016 'Batching', RFC-0071 'The statement and the batching').
fn slot_class(tokens: u64) -> u32 {
    u64::BITS - (tokens.max(1) - 1).leading_zeros()
}

/// The counted tokens `2^{c + ⌈log2 R⌉}` of a batch of `count = R ≥ 1` requests in slot class `c`
/// (RFC-0016 'Batching'), or `None` when `c + ⌈log2 R⌉ ≥ 64`.
fn counted_tokens(class: u32, count: usize) -> Option<u64> {
    let req_bits = if count <= 1 {
        0
    } else {
        usize::BITS - (count - 1).leading_zeros()
    };
    let shift = class.checked_add(req_bits)?;
    (shift < u64::BITS).then(|| 1u64 << shift)
}

/// Packs `requests`, each `(key, group, tokens)`, into batches by slot class and first-fit
/// decreasing (RFC-0016 'Batching', RFC-0071 'The statement and the batching').
///
/// Groups are packed on their own, in increasing order. Within a group, classes
/// `c = ⌈log2 max(L, 1)⌉` are packed in decreasing order; within a class, requests go in
/// decreasing `L`, ties in [`RequestKey`] order, each into the first open batch whose counted
/// tokens `2^{c + ⌈log2 R⌉}` stay at most `policy.max_batch_tokens`, else into a new batch. A
/// request of class `c > ⌊log2 max_batch_tokens⌋` opens a batch of its own, which takes no other.
/// `Batch.tokens` remains the batch's total real `L`.
///
/// What follows, since a valid policy has `2 · min_batch_tokens ≤ max_batch_tokens`: every
/// batch's counted tokens stay within the maximum except a lone request of a larger class, and at
/// most one batch of each class in a group has fewer than `min_batch_tokens` counted tokens.
///
/// # Errors
/// [`RepeatedRequest`] for a key given twice.
pub fn batches<G: Ord + Clone>(
    policy: &AuditPolicy,
    requests: &[(RequestKey, G, u64)],
) -> Result<Vec<Batch<G>>, RepeatedRequest> {
    let mut order: Vec<&(RequestKey, G, u64)> = requests.iter().collect();
    order.sort_by_key(|r| r.0);
    if let Some(pair) = order.windows(2).find(|w| w[0].0 == w[1].0) {
        return Err(RepeatedRequest(pair[0].0));
    }
    // By group, then decreasing slot class, then decreasing tokens, then key.
    order.sort_by(|a, b| {
        a.1.cmp(&b.1)
            .then(slot_class(b.2).cmp(&slot_class(a.2)))
            .then(b.2.cmp(&a.2))
            .then(a.0.cmp(&b.0))
    });
    let max = u64::from(policy.max_batch_tokens);
    let max_class = policy.max_batch_tokens.max(1).ilog2();
    let mut out = Vec::new();
    let mut start = 0;
    while start < order.len() {
        let group = &order[start].1;
        let class = slot_class(order[start].2);
        let end = start
            + order[start..]
                .iter()
                .take_while(|r| r.1 == *group && slot_class(r.2) == class)
                .count();
        if class > max_class {
            for &&(key, _, tokens) in &order[start..end] {
                out.push(Batch {
                    group: group.clone(),
                    requests: vec![key],
                    tokens,
                });
            }
            start = end;
            continue;
        }
        let mut open: Vec<(u64, Vec<RequestKey>)> = Vec::new();
        for &&(key, _, tokens) in &order[start..end] {
            match open.iter_mut().find(|(_, keys)| {
                counted_tokens(class, keys.len() + 1).is_some_and(|counted| counted <= max)
            }) {
                Some((total, keys)) => {
                    *total = total.saturating_add(tokens);
                    keys.push(key);
                }
                None => open.push((tokens, vec![key])),
            }
        }
        for (tokens, mut keys) in open {
            keys.sort_unstable();
            out.push(Batch {
                group: group.clone(),
                requests: keys,
                tokens,
            });
        }
        start = end;
    }
    Ok(out)
}
