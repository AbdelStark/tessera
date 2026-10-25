// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The recorder's signed manifest, `LogManifest` (schema `0x0402`, the `.tlm` artifact), its
//! identity, and the checks every manifest passes on its own (RFC-0015 'Log format' and
//! 'Signatures', RFC-0060 'The manifest's identity', 'Signatures' and 'The chain').

use core::fmt;

use tv_codec::{Cursor, DecodeError, DecodeLimits, Envelope, ErrorCode, Expected};
use tv_hash::ed25519::{Ed25519Error, verify_strict};
use tv_hash::sha256::sha256;
use tv_hash::{Digest, Domain, hash};

use crate::verdict::{AnchorFailure, RejectReason};

/// The signed message's label: ASCII, without a length or a terminator (RFC-0060 'Signatures').
pub const SIGNED_LABEL: &[u8; 19] = b"tessera-manifest/v1";

/// The signed message's length: the label, then the 144 bytes of every field but `signature`.
pub const SIGNED_LEN: usize = 163;

/// Scheme 0, Ed25519 (RFC 8032), the only scheme v1 verifies. Scheme 1, ML-DSA-65, is reserved
/// and refused (OQ-SEC-1), as is every other.
pub const ED25519: u16 = 0;

/// The manifest's signature envelope (RFC-0015 'Signatures'): a `u16` scheme, then the key and the
/// signature as `Bytes`. Decoding takes any lengths; the signature check fixes them.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SignatureEnvelope {
    /// The signature scheme.
    pub scheme: u16,
    /// The recorder's public key, 32 bytes under Ed25519.
    pub public_key: Vec<u8>,
    /// The signature, 64 bytes `R ‖ S` under Ed25519.
    pub signature: Vec<u8>,
}

/// A recorder's signed statement of one period of its log (RFC-0015 'Log format').
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LogManifest {
    /// `SHA-256(public_key)`.
    pub recorder_key_id: [u8; 32],
    /// The log's identifier, shared by every manifest of its chain.
    pub log_id: [u8; 16],
    /// The period's start, inclusive, on the recorder's clock.
    pub period_start_ns: u64,
    /// The period's end, exclusive.
    pub period_end_ns: u64,
    /// The `seq` of the period's first entry.
    pub first_seq: u64,
    /// The number of entries in the period.
    pub entry_count: u64,
    /// RFC 9162's tree hash over the entries' leaves, `SHA-256("")` for none.
    pub entries_root: [u8; 32],
    /// The previous manifest's identity, 32 zero bytes for a log's first.
    pub prev_manifest: [u8; 32],
    /// The recorder's signature.
    pub signature: SignatureEnvelope,
}

impl LogManifest {
    /// What its reader expects: schema `0x0402`, whose writers emit version 1 (03 §5).
    pub const EXPECTED: Expected = Expected {
        schema_id: 0x0402,
        version: 1,
    };

    /// Decodes one manifest at `cursor`: the fields in declaration order, the signature envelope
    /// a struct of its own.
    ///
    /// # Errors
    ///
    /// The cursor's, at their offsets; `ERR-RES-MEMORY` when the key or signature cannot be
    /// copied.
    pub fn decode(cursor: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        cursor.nested(|c| {
            Ok(Self {
                recorder_key_id: c.digest()?,
                log_id: c.byte_array::<16>()?,
                period_start_ns: c.u64()?,
                period_end_ns: c.u64()?,
                first_seq: c.u64()?,
                entry_count: c.u64()?,
                entries_root: c.digest()?,
                prev_manifest: c.digest()?,
                signature: c.nested(|c| {
                    Ok(SignatureEnvelope {
                        scheme: c.u16()?,
                        public_key: copy(c)?,
                        signature: copy(c)?,
                    })
                })?,
            })
        })
    }

    /// Opens a sealed manifest, a `.tlm` file: the envelope in RFC-0003's order, its digest
    /// BLAKE3 in the `envelope` domain, then the payload. The envelope's digest is the manifest's
    /// identity (RFC-0060 'The manifest's identity').
    ///
    /// # Errors
    ///
    /// The envelope's and the payload's decoding errors, at their offsets from the envelope's
    /// start. The manifest is the verifier's own input, so each is `Inconclusive(ERR-CODEC-*)`
    /// (RFC-0043 'Malformed').
    pub fn open(sealed: &[u8], limits: &DecodeLimits) -> Result<SealedManifest, DecodeError> {
        let envelope = Envelope::open(sealed, Self::EXPECTED, limits, |signed| {
            hash(Domain::Envelope, signed)
        })?;
        let manifest = envelope.decode_payload(|c, _version| Self::decode(c))?;
        // The digest is the envelope's last 32 bytes, which `open` has just recomputed.
        let identity = sealed
            .split_last_chunk::<32>()
            .map(|(_, digest)| *digest)
            .ok_or(DecodeError::new(ErrorCode::Internal, 0))?;
        Ok(SealedManifest { manifest, identity })
    }

    /// The signed message, `"tessera-manifest/v1" ‖ TCE(m without signature)`: the label, then
    /// every field but `signature` in order, 163 bytes (RFC-0060 'Signatures').
    pub fn signed_message(&self) -> [u8; SIGNED_LEN] {
        let parts: [&[u8]; 9] = [
            SIGNED_LABEL,
            &self.recorder_key_id,
            &self.log_id,
            &self.period_start_ns.to_le_bytes(),
            &self.period_end_ns.to_le_bytes(),
            &self.first_seq.to_le_bytes(),
            &self.entry_count.to_le_bytes(),
            &self.entries_root,
            &self.prev_manifest,
        ];
        let mut message = [0u8; SIGNED_LEN];
        for (slot, byte) in message.iter_mut().zip(parts.into_iter().flatten()) {
            *slot = *byte;
        }
        message
    }

    /// Whether the period has zero length, `period_start_ns = period_end_ns`: such a period
    /// holds no entries and has no candidates (RFC-0063 'The periods').
    pub const fn is_zero_length(&self) -> bool {
        self.period_start_ns == self.period_end_ns
    }
}

/// A `Bytes` field copied out of the input, with its memory reserved fallibly.
fn copy(cursor: &mut Cursor<'_>) -> Result<Vec<u8>, DecodeError> {
    let at = cursor.offset();
    let bytes = cursor.bytes()?;
    let mut owned = Vec::new();
    owned
        .try_reserve_exact(bytes.len())
        .map_err(|_| DecodeError::new(ErrorCode::Memory, at))?;
    owned.extend_from_slice(bytes);
    Ok(owned)
}

/// A manifest decoded from its sealed envelope, with its identity, before any of its checks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SealedManifest {
    manifest: LogManifest,
    identity: Digest,
}

impl SealedManifest {
    /// The decoded manifest.
    pub fn manifest(&self) -> &LogManifest {
        &self.manifest
    }

    /// `identity(m)`, the envelope's digest (RFC-0060).
    pub fn identity(&self) -> &Digest {
        &self.identity
    }

    /// Runs a manifest's own checks, in RFC-0060's order ('The chain'):
    ///
    /// 1. its signature under a pinned recorder key, in the order of 'Signatures': the scheme is
    ///    Ed25519; the key is 32 bytes and the signature 64; `recorder_key_id` is the key's
    ///    SHA-256; the key's bytes equal a pinned key's, compared before the key is decoded; and
    ///    the signature verifies strictly over
    ///    [`signed_message`](LogManifest::signed_message);
    /// 2. `period_start_ns ≤ period_end_ns`;
    /// 3. `first_seq + entry_count` without overflow;
    /// 4. for `entry_count = 0`, `entries_root = SHA-256("")`.
    ///
    /// Strict verification is [`tv_hash::ed25519::verify_strict`], RFC-0060's set: `S < L`; `R`
    /// and the key decode to points, neither of small order; and `R`'s 32 bytes encode
    /// `[S]B − [k]A`, `k = SHA-512(R ‖ A ‖ message) mod L`, the cofactorless equation. A
    /// malleated `S + L`, a small-order `R` or key, and a signature that only the cofactored
    /// equation accepts are all refused. It runs at most once, after every other check of step 1,
    /// on a key already compared with the pinned ones.
    ///
    /// # Errors
    ///
    /// The first check that fails ([`ManifestError`]): every one is `Reject(Anchor { request_id:
    /// None, kind: ManifestSignature })` (RFC-0015 'Verifier checks', RFC-0043 'Anchor', RFC-0060
    /// 'Signatures').
    pub fn check(self, keys: &RecorderKeys) -> Result<CheckedManifest, ManifestError> {
        let m = &self.manifest;
        let envelope = &m.signature;
        if envelope.scheme != ED25519 {
            return Err(ManifestError::Scheme);
        }
        let public_key: &[u8; 32] = envelope
            .public_key
            .as_slice()
            .try_into()
            .map_err(|_| ManifestError::KeyLength)?;
        let signature: &[u8; 64] = envelope
            .signature
            .as_slice()
            .try_into()
            .map_err(|_| ManifestError::SignatureLength)?;
        if sha256(public_key) != m.recorder_key_id {
            return Err(ManifestError::KeyId);
        }
        if !keys.pins(public_key) {
            return Err(ManifestError::Unpinned);
        }
        verify_strict(public_key, &m.signed_message(), signature)
            .map_err(ManifestError::Signature)?;
        if m.period_start_ns > m.period_end_ns {
            return Err(ManifestError::Period);
        }
        if m.first_seq.checked_add(m.entry_count).is_none() {
            return Err(ManifestError::Range);
        }
        if m.entry_count == 0 && m.entries_root != sha256(&[]) {
            return Err(ManifestError::EmptyRoot);
        }
        Ok(CheckedManifest {
            manifest: self.manifest,
            identity: self.identity,
        })
    }
}

/// The recorder public keys the verifier pins out of band (`--recorder-key`; RFC-0015
/// 'Signatures'). A key rotation is a pinning of both keys (RFC-0060 'The chain').
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecorderKeys {
    keys: Vec<[u8; 32]>,
}

impl RecorderKeys {
    /// The pinned keys, as the verifier's configuration lists them.
    pub fn new(keys: Vec<[u8; 32]>) -> Self {
        Self { keys }
    }

    /// Whether `key`'s bytes equal a pinned key's.
    pub fn pins(&self, key: &[u8; 32]) -> bool {
        self.keys.iter().any(|pinned| pinned == key)
    }
}

/// Why a manifest failed its own checks, in their order. Each is `Reject(Anchor { request_id:
/// None, kind: ManifestSignature })`: the variants name the check for tests and diagnostics, and
/// carry no input bytes. [`Signature`](Self::Signature) carries the first rule of strict
/// verification the signature fails, which is public.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ManifestError {
    /// The scheme is not 0: scheme 1, ML-DSA-65, is reserved, and every other is unknown.
    Scheme,
    /// The public key is not 32 bytes.
    KeyLength,
    /// The signature is not 64 bytes.
    SignatureLength,
    /// `recorder_key_id` is not the key's SHA-256.
    KeyId,
    /// The key is not one the verifier pins.
    Unpinned,
    /// Strict Ed25519 verification refused the signature, or the key, at this rule.
    Signature(Ed25519Error),
    /// `period_start_ns > period_end_ns`.
    Period,
    /// `first_seq + entry_count` overflows.
    Range,
    /// No entries, and `entries_root` is not `SHA-256("")`.
    EmptyRoot,
}

impl ManifestError {
    /// The rejection: `Anchor { request_id: None, kind: ManifestSignature }`.
    pub const fn reason(self) -> RejectReason {
        RejectReason::Anchor {
            request_id: None,
            kind: AnchorFailure::ManifestSignature,
        }
    }
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::Scheme => "the manifest's signature scheme is not Ed25519",
            Self::KeyLength => "the manifest's public key is not 32 bytes",
            Self::SignatureLength => "the manifest's signature is not 64 bytes",
            Self::KeyId => "the manifest's recorder_key_id is not its key's SHA-256",
            Self::Unpinned => "the manifest's key is not pinned",
            Self::Signature(rule) => {
                return write!(f, "the manifest's signature does not verify: {rule}");
            }
            Self::Period => "the manifest's period ends before it starts",
            Self::Range => "the manifest's entry range overflows",
            Self::EmptyRoot => "the manifest has no entries and another root than SHA-256(\"\")",
        };
        f.write_str(text)
    }
}

impl std::error::Error for ManifestError {}

/// A manifest that passed its own checks under the verifier's pinned keys: the only form in
/// which the period's log is checked against it and the chain is followed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckedManifest {
    manifest: LogManifest,
    identity: Digest,
}

impl CheckedManifest {
    /// The manifest.
    pub fn manifest(&self) -> &LogManifest {
        &self.manifest
    }

    /// `identity(m)`, the envelope's digest, which a statement's `manifest_digest` and the next
    /// manifest's `prev_manifest` hold (RFC-0060).
    pub fn identity(&self) -> &Digest {
        &self.identity
    }

    /// Whether `next` follows this manifest in one log (RFC-0060 'The chain'): the same
    /// `log_id`, `next.prev_manifest` this manifest's identity, `next.first_seq` this one's
    /// `first_seq + entry_count`, and `next.period_start_ns` this one's `period_end_ns`.
    pub fn precedes(&self, next: &CheckedManifest) -> bool {
        let (a, b) = (&self.manifest, &next.manifest);
        b.log_id == a.log_id
            && b.prev_manifest == self.identity
            && a.first_seq.checked_add(a.entry_count) == Some(b.first_seq)
            && b.period_start_ns == a.period_end_ns
    }
}
