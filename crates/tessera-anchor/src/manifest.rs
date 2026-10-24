// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The recorder's manifests (RFC-0015 'Log format', 'Signatures', with RFC-0060): their TCE
//! types (schema `0x0402`), Ed25519 signing and strict verification, their identity, and the
//! chain consecutive manifests of a log form. Everything here is public but the signing key.

use crate::log::DigestLogEntry;
use crate::merkle::{self, Hash, InclusionError, MAX_PROOF_LEN};
use core::borrow::{Borrow, BorrowMut};
use core::fmt;
use ed25519_dalek::{Signer, SigningKey, VerifyingKey};
use sha2::{Digest, Sha256};
use tessera_codec::HeapKey;
use tessera_codec::{Bytes, Decode, Encode, Secret, encode, envelope, schema};

/// The label that opens a manifest's signed message (RFC-0015 'Signatures').
const LABEL: &[u8] = b"tessera-manifest/v1";

/// A manifest's signature (RFC-0015 'Signatures'): scheme 0 is Ed25519, scheme 1 (ML-DSA-65) is
/// reserved.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub struct SignatureEnvelope {
    /// 0 for Ed25519; 1, ML-DSA-65, is reserved and refused (OQ-SEC-1).
    pub scheme: u16,
    /// The recorder's public key: for Ed25519, the 32-byte encoded point.
    pub public_key: Bytes,
    /// The signature: for Ed25519, the 64 bytes `R ‖ S`.
    pub signature: Bytes,
}

/// What a manifest signs: every field of [`LogManifest`] but its signature, in order.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub struct ManifestBody {
    /// `SHA-256` of the recorder's public key.
    pub recorder_key_id: [u8; 32],
    /// The log's identifier, the same across its manifests.
    pub log_id: [u8; 16],
    /// The period's start, in nanoseconds.
    pub period_start_ns: u64,
    /// The period's end, in nanoseconds: the next period's start.
    pub period_end_ns: u64,
    /// The `seq` of the period's first entry.
    pub first_seq: u64,
    /// The period's entries: `seq` from `first_seq` to `first_seq + entry_count − 1`.
    pub entry_count: u64,
    /// RFC 9162's root over the period's entries ([`crate::merkle::root`]).
    pub entries_root: [u8; 32],
    /// The previous manifest's identity ([`LogManifest::identity`]); 32 zero bytes for a log's
    /// first.
    pub prev_manifest: [u8; 32],
}

/// A signed manifest (schema `0x0402`, version 1). Its TCE is the body's fields, then the
/// signature: exactly RFC-0015's flat `LogManifest`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub struct LogManifest {
    /// The signed fields.
    pub body: ManifestBody,
    /// The recorder's signature over `"tessera-manifest/v1" ‖ TCE(body)`.
    pub signature: SignatureEnvelope,
}

/// Why a manifest is not valid on its own (RFC-0060 'Signatures', 'The chain').
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ManifestError {
    /// Scheme 1, ML-DSA-65, which v1 reserves and refuses (OQ-SEC-1).
    Reserved,
    /// A scheme no version defines.
    UnknownScheme(u16),
    /// An Ed25519 public key that is not 32 bytes, or not a point.
    KeyLength,
    /// An Ed25519 signature that is not 64 bytes.
    SignatureLength,
    /// `recorder_key_id` is not the SHA-256 of the envelope's public key.
    KeyId,
    /// The public key is not one the verifier pins.
    Unpinned,
    /// The signature does not verify under strict Ed25519.
    Signature,
    /// The period ends before it starts.
    Period,
    /// `first_seq + entry_count` overflows: the range of entries does not exist.
    Range,
    /// A period without entries whose root is not the empty tree's, `SHA-256("")`.
    EmptyRoot,
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserved => f.write_str("signature scheme 1 (ML-DSA-65) is reserved in v1"),
            Self::UnknownScheme(s) => write!(f, "signature scheme {s} is unknown"),
            Self::KeyLength => f.write_str("the Ed25519 public key is not a 32-byte point"),
            Self::SignatureLength => f.write_str("the Ed25519 signature is not 64 bytes"),
            Self::KeyId => f.write_str("recorder_key_id is not the SHA-256 of the public key"),
            Self::Unpinned => f.write_str("the recorder key is not pinned"),
            Self::Signature => f.write_str("the signature does not verify"),
            Self::Period => f.write_str("the period ends before it starts"),
            Self::Range => f.write_str("first_seq + entry_count overflows"),
            Self::EmptyRoot => f.write_str("an empty period's root is not SHA-256 of nothing"),
        }
    }
}

impl std::error::Error for ManifestError {}

/// Why two consecutive manifests do not chain (RFC-0060 'The chain'), in the order checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkError {
    /// The log identifiers differ.
    LogId,
    /// `prev_manifest` is not the previous manifest's identity.
    Previous,
    /// The entries do not continue the previous ones: a gap, an overlap or an overflow.
    Entries,
    /// The period does not start where the previous one ended.
    Period,
}

impl fmt::Display for LinkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::LogId => "the manifests belong to different logs",
            Self::Previous => "prev_manifest is not the previous manifest's identity",
            Self::Entries => "the entries do not continue the previous manifest's",
            Self::Period => "the period does not start where the previous one ended",
        })
    }
}

impl std::error::Error for LinkError {}

/// The first break of a chain of manifests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChainError {
    /// No manifest.
    Empty,
    /// The first manifest is not a log's first: `prev_manifest` is not zero or `first_seq` not 0.
    Genesis,
    /// Manifest `index` is not valid on its own.
    Manifest {
        /// Its position in the sequence.
        index: usize,
        /// Why.
        error: ManifestError,
    },
    /// Manifest `index` does not chain to the one before it.
    Link {
        /// Its position in the sequence.
        index: usize,
        /// Why.
        error: LinkError,
    },
}

impl fmt::Display for ChainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("the chain holds no manifest"),
            Self::Genesis => f.write_str("the chain does not start at the log's first manifest"),
            Self::Manifest { index, error } => write!(f, "manifest {index}: {error}"),
            Self::Link { index, error } => write!(f, "manifest {index}: {error}"),
        }
    }
}

impl std::error::Error for ChainError {}

/// `SHA-256` of a recorder's 32-byte Ed25519 public key: the manifest's `recorder_key_id`.
pub fn key_id(public_key: &[u8; 32]) -> [u8; 32] {
    Sha256::digest(public_key).into()
}

/// A recorder's 32-byte Ed25519 seed, read from `path`: the file holds the seed's bytes and
/// nothing else. The bytes go straight into a heap [`Secret`], and the buffer the file was read
/// into is cleared before it is dropped (05 §3 OBS-2).
///
/// # Errors
/// The read's I/O error, or [`std::io::ErrorKind::InvalidData`] for a file that is not 32 bytes.
#[allow(clippy::disallowed_methods)]
pub fn read_seed(path: &std::path::Path) -> std::io::Result<Secret<HeapKey<32>>> {
    let mut read = Secret::new(std::fs::read(path)?);
    let bytes = read.expose_secret_mut();
    if bytes.len() != 32 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("a recorder seed is 32 bytes, not {}", bytes.len()),
        ));
    }
    let mut seed = Secret::new(HeapKey::<32>::zeroed());
    let target: &mut [u8; 32] = seed.expose_secret_mut().borrow_mut();
    target.copy_from_slice(bytes);
    Ok(seed)
}

/// The Ed25519 public key of a recorder's 32-byte secret seed.
#[allow(clippy::disallowed_methods)]
pub fn public_key(seed: &Secret<HeapKey<32>>) -> [u8; 32] {
    SigningKey::from_bytes(seed.expose_secret().borrow())
        .verifying_key()
        .to_bytes()
}

impl ManifestBody {
    /// The message the recorder signs: `"tessera-manifest/v1" ‖ TCE(body)`, 163 bytes.
    pub fn signed_message(&self) -> Vec<u8> {
        let mut message = LABEL.to_vec();
        message.extend_from_slice(&encode(self));
        message
    }

    /// The manifest of this body, signed with Ed25519 under the recorder's secret seed. The
    /// signing key is cleared when it drops.
    ///
    /// # Errors
    /// [`ManifestError::KeyId`] when `recorder_key_id` is not the SHA-256 of the seed's public key.
    #[allow(clippy::disallowed_methods)]
    pub fn sign(self, seed: &Secret<HeapKey<32>>) -> Result<LogManifest, ManifestError> {
        let key = SigningKey::from_bytes(seed.expose_secret().borrow());
        let public = key.verifying_key().to_bytes();
        if key_id(&public) != self.recorder_key_id {
            return Err(ManifestError::KeyId);
        }
        let signature = key.sign(&self.signed_message()).to_bytes();
        Ok(LogManifest {
            body: self,
            signature: SignatureEnvelope {
                scheme: 0,
                public_key: public.to_vec(),
                signature: signature.to_vec(),
            },
        })
    }
}

impl LogManifest {
    /// The manifest's identity: its envelope digest, schema `0x0402` version 1 (RFC-0060 'The
    /// manifest's identity'), which names it in a statement and in the next manifest.
    pub fn identity(&self) -> [u8; 32] {
        let sealed = envelope::seal(&schema::LOG_MANIFEST, self);
        let mut out = [0u8; 32];
        out.copy_from_slice(&sealed[sealed.len() - 32..]);
        out
    }

    /// Checks the manifest on its own, in order (RFC-0060 'Signatures', 'The chain'): its
    /// scheme, its key's and signature's lengths, its key identifier, the key's pinning, the
    /// signature under strict Ed25519, its period, its range of entries, and an empty period's
    /// root.
    ///
    /// # Errors
    /// The first [`ManifestError`] in that order.
    pub fn verify(&self, pinned: &[[u8; 32]]) -> Result<(), ManifestError> {
        let s = &self.signature;
        match s.scheme {
            0 => {}
            1 => return Err(ManifestError::Reserved),
            other => return Err(ManifestError::UnknownScheme(other)),
        }
        let public: [u8; 32] = s
            .public_key
            .as_slice()
            .try_into()
            .map_err(|_| ManifestError::KeyLength)?;
        let signature: [u8; 64] = s
            .signature
            .as_slice()
            .try_into()
            .map_err(|_| ManifestError::SignatureLength)?;
        if key_id(&public) != self.body.recorder_key_id {
            return Err(ManifestError::KeyId);
        }
        if !pinned.contains(&public) {
            return Err(ManifestError::Unpinned);
        }
        let key = VerifyingKey::from_bytes(&public).map_err(|_| ManifestError::KeyLength)?;
        key.verify_strict(
            &self.body.signed_message(),
            &ed25519_dalek::Signature::from_bytes(&signature),
        )
        .map_err(|_| ManifestError::Signature)?;
        let b = &self.body;
        if b.period_end_ns < b.period_start_ns {
            return Err(ManifestError::Period);
        }
        if b.first_seq.checked_add(b.entry_count).is_none() {
            return Err(ManifestError::Range);
        }
        if b.entry_count == 0 && b.entries_root != merkle::root(&[]) {
            return Err(ManifestError::EmptyRoot);
        }
        Ok(())
    }

    /// Verifies that `entry` is in this manifest by `proof` (RFC-0060 'The tree over a period's
    /// entries'): its leaf at index `seq − first_seq`, in a tree of `entry_count` leaves under
    /// `entries_root`. The index and the size come from the entry and the manifest, never from the
    /// proof. Check the manifest itself with [`Self::verify`] first.
    ///
    /// # Errors
    /// [`EntryError::Range`] for an entry whose `seq` lies outside the manifest's range,
    /// [`EntryError::TooLong`] for a proof of more than 64 digests, then the proof's own
    /// [`InclusionError`].
    pub fn verify_entry(&self, entry: &DigestLogEntry, proof: &[Hash]) -> Result<(), EntryError> {
        let b = &self.body;
        let index = entry
            .seq()
            .checked_sub(b.first_seq)
            .filter(|&i| i < b.entry_count)
            .ok_or(EntryError::Range)?;
        if proof.len() > MAX_PROOF_LEN {
            return Err(EntryError::TooLong);
        }
        merkle::verify_inclusion(
            &merkle::leaf_hash(entry),
            index,
            b.entry_count,
            proof,
            &b.entries_root,
        )
        .map_err(EntryError::Inclusion)
    }
}

/// Why an entry is not shown to be in a manifest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntryError {
    /// The entry's `seq` lies outside the manifest's range.
    Range,
    /// The proof holds more than 64 digests, more than any path.
    TooLong,
    /// The proof does not show the entry's leaf at its index.
    Inclusion(InclusionError),
}

impl fmt::Display for EntryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Range => f.write_str("the entry's seq lies outside the manifest's range"),
            Self::TooLong => f.write_str("the proof holds more than 64 digests"),
            Self::Inclusion(e) => write!(f, "the inclusion proof fails: {e}"),
        }
    }
}

impl std::error::Error for EntryError {}

/// Checks that `next` follows `prev` in one log (RFC-0060 'The chain'), in order: the log
/// identifier, the link, the entries, the period.
///
/// # Errors
/// The first [`LinkError`] in that order.
pub fn check_link(prev: &LogManifest, next: &LogManifest) -> Result<(), LinkError> {
    let (a, b) = (&prev.body, &next.body);
    if b.log_id != a.log_id {
        return Err(LinkError::LogId);
    }
    if b.prev_manifest != prev.identity() {
        return Err(LinkError::Previous);
    }
    if a.first_seq.checked_add(a.entry_count) != Some(b.first_seq) {
        return Err(LinkError::Entries);
    }
    if b.period_start_ns != a.period_end_ns {
        return Err(LinkError::Period);
    }
    Ok(())
}

/// Checks a chain of manifests from a log's first one: the genesis, then each manifest on its own
/// and its link to the one before, in sequence order.
///
/// # Errors
/// The first [`ChainError`].
pub fn check_chain(manifests: &[LogManifest], pinned: &[[u8; 32]]) -> Result<(), ChainError> {
    let first = manifests.first().ok_or(ChainError::Empty)?;
    if first.body.prev_manifest != [0; 32] || first.body.first_seq != 0 {
        return Err(ChainError::Genesis);
    }
    for (index, manifest) in manifests.iter().enumerate() {
        manifest
            .verify(pinned)
            .map_err(|error| ChainError::Manifest { index, error })?;
        if index > 0 {
            check_link(&manifests[index - 1], manifest)
                .map_err(|error| ChainError::Link { index, error })?;
        }
    }
    Ok(())
}
