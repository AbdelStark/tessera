// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The sample (RFC-0016 'Sample derivation', steps 2 to 5): the coin toss's transcript, the
//! sample seed, each candidate's selection value and decision, and the public artifact
//! `AuditSample` (schema `0x0403`).

use tv_codec::{Cursor, DecodeError, DecodeLimits, Envelope, ErrorCode, Expected, decode, size};
use tv_hash::{Digest, Domain, hash};

use super::policy::{AuditPolicy, put_len};
use super::{memory, seal};
use crate::anchor::{Candidate, Failure, Window};
use crate::verdict::{ErrorCode as Code, VerifierError};

/// A request of a period: its session and its id within the session. The derived order is
/// `(session_id, request_id)`: the session's bytes, then the id as a number, the order of
/// `AuditSample.selected`. TCE: the 16 bytes, then the `u64`, 24 bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RequestKey {
    /// The request's session.
    pub session_id: [u8; 16],
    /// The request's id.
    pub request_id: u64,
}

impl RequestKey {
    /// The encoded size.
    pub const ENCODED_LEN: u64 = 24;
}

impl From<&Candidate> for RequestKey {
    fn from(candidate: &Candidate) -> Self {
        Self {
            session_id: candidate.session_id,
            request_id: candidate.request_id,
        }
    }
}

/// The coin toss over a manifest, as both parties keep it (RFC-0016 'Sample derivation' step 2):
/// `c_V`, `s_P`, `s_V`, TCE in that order, 96 bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CoinTossTranscript {
    /// `c_V = H("coin-commit", s_V)`, which the verifier sent first.
    pub verifier_commitment: Digest,
    /// `s_P`, the prover's share.
    pub prover_share: [u8; 32],
    /// `s_V`, the verifier's share, revealed last.
    pub verifier_share: [u8; 32],
}

impl CoinTossTranscript {
    /// The encoded size.
    pub const ENCODED_LEN: u64 = 96;

    /// The transcript's check: `c_V = H("coin-commit", s_V)`.
    ///
    /// # Errors
    ///
    /// A reveal that does not open the commitment is `ERR-PROTO-COINTOSS`.
    pub fn check(&self) -> Result<(), VerifierError> {
        if hash(Domain::CoinCommit, &self.verifier_share) == self.verifier_commitment {
            Ok(())
        } else {
            Err(VerifierError {
                code: Code::ProtocolCoinToss,
            })
        }
    }

    /// The sample seed over `manifest_digest`: `H("audit-sample", s_V ‖ s_P ‖ manifest_digest)`
    /// (step 3).
    pub fn sample_seed(&self, manifest_digest: &Digest) -> Digest {
        let mut input = [0; 96];
        let (v, rest) = input.split_at_mut(32);
        let (p, m) = rest.split_at_mut(32);
        v.copy_from_slice(&self.verifier_share);
        p.copy_from_slice(&self.prover_share);
        m.copy_from_slice(manifest_digest);
        hash(Domain::AuditSample, &input)
    }

    fn decode(c: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        c.nested(|c| {
            Ok(Self {
                verifier_commitment: c.digest()?,
                prover_share: c.digest()?,
                verifier_share: c.digest()?,
            })
        })
    }
}

/// A request's selection value under `seed`: `u64_le(H("audit-select", seed ‖ session_id ‖
/// u64(request_id))[0..8])` (step 4).
pub fn selection_value(seed: &Digest, key: &RequestKey) -> u64 {
    let mut input = [0; 56];
    let (s, rest) = input.split_at_mut(32);
    let (session, request) = rest.split_at_mut(16);
    s.copy_from_slice(seed);
    session.copy_from_slice(&key.session_id);
    request.copy_from_slice(&key.request_id.to_le_bytes());
    let digest = hash(Domain::AuditSelect, &input);
    let mut low = [0; 8];
    low.copy_from_slice(digest.get(..8).unwrap_or(&[0; 8]));
    u64::from_le_bytes(low)
}

/// The candidates `policy` selects under `seed`, sorted by `(session_id, request_id)` with no
/// repeat (step 4): a candidate of length `L` is selected iff its selection value is below
/// `bucket(L)`'s threshold, or `L` is past every bucket.
///
/// # Errors
///
/// `Inconclusive(ERR-RES-MEMORY)` when the selection cannot be reserved.
pub fn select(
    policy: &AuditPolicy,
    seed: &Digest,
    candidates: &[Candidate],
) -> Result<Vec<RequestKey>, Failure> {
    let mut selected: Vec<RequestKey> = Vec::new();
    for candidate in candidates {
        let key = RequestKey::from(candidate);
        // A candidate past every bucket is selected without its value.
        let chosen = match policy.bucket_threshold(candidate.tokens) {
            None => true,
            Some(threshold) => selection_value(seed, &key) < threshold,
        };
        if chosen {
            if selected.len() == selected.capacity() {
                selected
                    .try_reserve(selected.len().max(16))
                    .map_err(|_| memory())?;
            }
            selected.push(key);
        }
    }
    selected.sort_unstable();
    selected.dedup();
    Ok(selected)
}

/// The public artifact of a period's audit (RFC-0016 'Sample derivation' step 5, schema
/// `0x0403`, version 1): the manifest, the policy, the coin toss and the selected requests,
/// strictly increasing. Only [`derive`](Self::derive) and the decoders construct one, and each
/// computes its identity, the digest of its envelope (as a manifest's identity is, RFC-0060).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuditSample {
    manifest_digest: Digest,
    policy: AuditPolicy,
    coin_toss: CoinTossTranscript,
    selected: Vec<RequestKey>,
    identity: Digest,
}

impl AuditSample {
    /// The envelope's schema: `0x0403`, version 1 (03 §5).
    pub const SCHEMA: Expected = Expected {
        schema_id: 0x0403,
        version: 1,
    };

    /// Derives the sample of `window`'s period from the verifier's own inputs (RFC-0016 'Sample
    /// derivation', INV-AUD-2): the coin toss's check, then `sample_seed` over the period's
    /// manifest, then the selection over [`Window::candidates`]. The window holds both periods'
    /// verified logs, bound by the chain, so no sample comes before step 1.
    ///
    /// # Errors
    ///
    /// `Inconclusive(ERR-PROTO-COINTOSS)` for a reveal that does not open its commitment, and
    /// `Inconclusive(ERR-RES-MEMORY)` when memory cannot be reserved.
    pub fn derive(
        window: &Window<'_>,
        policy: AuditPolicy,
        coin_toss: CoinTossTranscript,
    ) -> Result<Self, Failure> {
        coin_toss.check().map_err(Failure::Inconclusive)?;
        let manifest_digest = *window.period().manifest_digest();
        let seed = coin_toss.sample_seed(&manifest_digest);
        let candidates = window.candidates()?;
        let selected = select(&policy, &seed, &candidates)?;
        Self::with_identity(manifest_digest, policy, coin_toss, selected)
            .map_err(Failure::Inconclusive)
    }

    /// Checks that this sample is the one `window`'s period draws: its manifest is the period's,
    /// its coin toss opens, and its selection is the verifier's own (INV-AUD-2). It judges the
    /// selection that the sample's own policy and coin toss give; whether those are the policy
    /// and the toss the verifier fixed is a comparison with its [`AuditToss`](super::AuditToss)
    /// ([`AuditToss::policy`](super::AuditToss::policy), its commitment), or with the sample
    /// the toss revealed.
    ///
    /// # Errors
    ///
    /// `Inconclusive(ERR-PROTO-COINTOSS)` for a coin toss that does not open, and
    /// `Inconclusive(ERR-USAGE)` for another manifest or another selection: the verifier's own
    /// inputs disagree.
    pub fn check_derivation(&self, window: &Window<'_>) -> Result<(), Failure> {
        if &self.manifest_digest != window.period().manifest_digest() {
            return Err(crate::anchor::usage());
        }
        let own = Self::derive(window, self.policy.clone(), self.coin_toss)?;
        if own.selected != self.selected {
            return Err(crate::anchor::usage());
        }
        Ok(())
    }

    /// Opens a `.tas` artifact: RFC-0003's envelope, schema `0x0403` version 1, then the
    /// payload ([`decode_payload`](Self::decode_payload)'s rules, offsets from the envelope's
    /// start). The identity is the envelope's digest.
    ///
    /// # Errors
    ///
    /// The first failure of RFC-0003's order, `ERR-CODEC-*` at its offset, or `ERR-RES-MEMORY`.
    pub fn open(tas: &[u8], limits: &DecodeLimits) -> Result<Self, DecodeError> {
        let envelope = Envelope::open(tas, Self::SCHEMA, limits, |b| hash(Domain::Envelope, b))?;
        let (manifest_digest, policy, coin_toss, selected) =
            envelope.decode_payload(|c, _| Self::fields(c))?;
        let at = u64::try_from(tas.len()).unwrap_or(u64::MAX);
        let identity: Digest = tas
            .len()
            .checked_sub(32)
            .and_then(|start| tas.get(start..))
            .and_then(|d| d.try_into().ok())
            .ok_or(DecodeError::new(ErrorCode::Internal, at))?;
        Ok(Self {
            manifest_digest,
            policy,
            coin_toss,
            selected,
            identity,
        })
    }

    /// Decodes a payload alone, its fields in order (RFC-0016 'Sample derivation' step 5): the
    /// manifest digest, the policy ([`AuditPolicy`]'s rules), the coin toss, then `selected`,
    /// whose keys strictly increase: a key not above the one before it is
    /// `ERR-CODEC-NONCANONICAL` at its first byte. Bytes past the value are
    /// `ERR-CODEC-NONCANONICAL` at the first of them.
    ///
    /// # Errors
    ///
    /// The first failed check, `ERR-CODEC-*` at its offset in the payload, or `ERR-RES-MEMORY`.
    pub fn decode_payload(payload: &[u8], limits: &DecodeLimits) -> Result<Self, DecodeError> {
        let (manifest_digest, policy, coin_toss, selected) = decode(payload, limits, Self::fields)?;
        // Sealing fails only for memory.
        Self::with_identity(manifest_digest, policy, coin_toss, selected)
            .map_err(|_| DecodeError::new(ErrorCode::Memory, 0))
    }

    fn fields(
        c: &mut Cursor<'_>,
    ) -> Result<(Digest, AuditPolicy, CoinTossTranscript, Vec<RequestKey>), DecodeError> {
        c.nested(|c| {
            let manifest_digest = c.digest()?;
            let policy = AuditPolicy::decode(c)?;
            let coin_toss = CoinTossTranscript::decode(c)?;
            let mut previous: Option<RequestKey> = None;
            let selected = c.vec(RequestKey::ENCODED_LEN, |c| {
                c.nested(|c| {
                    let at = c.offset();
                    let key = RequestKey {
                        session_id: c.byte_array()?,
                        request_id: c.u64()?,
                    };
                    if previous.is_some_and(|p| key <= p) {
                        return Err(DecodeError::new(ErrorCode::Noncanonical, at));
                    }
                    previous = Some(key);
                    Ok(key)
                })
            })?;
            Ok((manifest_digest, policy, coin_toss, selected))
        })
    }

    fn with_identity(
        manifest_digest: Digest,
        policy: AuditPolicy,
        coin_toss: CoinTossTranscript,
        selected: Vec<RequestKey>,
    ) -> Result<Self, VerifierError> {
        let mut sample = Self {
            manifest_digest,
            policy,
            coin_toss,
            selected,
            identity: [0; 32],
        };
        let sealed = sample.seal()?;
        sample.identity = super::envelope_digest(&sealed);
        Ok(sample)
    }

    /// The payload's TCE encoding.
    ///
    /// # Errors
    ///
    /// `ERR-RES-MEMORY` when it cannot be reserved.
    pub fn encode_payload(&self) -> Result<Vec<u8>, VerifierError> {
        let keys = u64::try_from(self.selected.len())
            .ok()
            .and_then(|n| n.checked_mul(RequestKey::ENCODED_LEN));
        let len = self
            .policy
            .encoded_len()
            .zip(keys)
            .and_then(|(policy, keys)| {
                size::sum(&[
                    size::DIGEST,
                    CoinTossTranscript::ENCODED_LEN,
                    size::SEQUENCE,
                ])
                .checked_add(policy)?
                .checked_add(keys)
            })
            .and_then(|n| usize::try_from(n).ok())
            .ok_or(VerifierError { code: Code::Memory })?;
        let mut out = Vec::new();
        out.try_reserve_exact(len)
            .map_err(|_| VerifierError { code: Code::Memory })?;
        out.extend_from_slice(&self.manifest_digest);
        self.policy.encode(&mut out);
        out.extend_from_slice(&self.coin_toss.verifier_commitment);
        out.extend_from_slice(&self.coin_toss.prover_share);
        out.extend_from_slice(&self.coin_toss.verifier_share);
        put_len(&mut out, self.selected.len());
        for key in &self.selected {
            out.extend_from_slice(&key.session_id);
            out.extend_from_slice(&key.request_id.to_le_bytes());
        }
        Ok(out)
    }

    /// The `.tas` artifact: the payload in RFC-0003's envelope, schema `0x0403` version 1.
    ///
    /// # Errors
    ///
    /// `ERR-RES-MEMORY` when it cannot be reserved.
    pub fn seal(&self) -> Result<Vec<u8>, VerifierError> {
        seal(Self::SCHEMA, &self.encode_payload()?)
    }

    /// The sample's identity: its envelope's digest, the `.tas` file's last 32 bytes. The ledger
    /// names a sample by it.
    pub fn identity(&self) -> &Digest {
        &self.identity
    }

    /// The period's manifest digest.
    pub fn manifest_digest(&self) -> &Digest {
        &self.manifest_digest
    }

    /// The policy the parties fixed before the toss.
    pub fn policy(&self) -> &AuditPolicy {
        &self.policy
    }

    /// The coin toss's transcript.
    pub fn coin_toss(&self) -> &CoinTossTranscript {
        &self.coin_toss
    }

    /// The sample seed, recomputed from the transcript and the manifest digest (step 3).
    pub fn seed(&self) -> Digest {
        self.coin_toss.sample_seed(&self.manifest_digest)
    }

    /// The selected requests, strictly increasing.
    pub fn selected(&self) -> &[RequestKey] {
        &self.selected
    }

    /// The index of `key` in `selected`, `None` when the sample did not select it.
    pub fn index_of(&self, key: &RequestKey) -> Option<usize> {
        self.selected.binary_search(key).ok()
    }
}
