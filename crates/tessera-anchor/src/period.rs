// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A period's verified log and the requests it draws its audit sample from (RFC-0062 'The
//! period's log', RFC-0063 'Which requests a period draws from').
//!
//! - [`VerifiedLog::check`] holds a log against its manifest: the count, each entry's `seq`,
//!   then RFC 9162's root. It is the only way to build a [`VerifiedLog`], so whatever reads one
//!   reads entries the check covered.
//! - [`candidates`] lists a period's candidates and their lengths from its verified log and the
//!   previous period's: every request with an end in the period, and every one with a frame in
//!   the previous period and no end in either. A candidate's length counts all its frames in
//!   both periods, wherever they lie against its end.
//! - [`lengths`] gives the lengths a request's statement carries over the same frames: the
//!   tokens of its `In` frames, then of its `Out` frames (RFC-0015 check 2).
//!
//! Everything here is public: the entries are digests and identifiers (03 §5).

use std::collections::{BTreeMap, BTreeSet};

use tessera_codec::error::ErrorCode;

use crate::frame::Direction;
use crate::log::DigestLogEntry;
use crate::manifest::LogManifest;
use crate::merkle;

/// A period's log, checked against its manifest (RFC-0062 'The check').
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedLog {
    manifest: LogManifest,
    entries: Vec<DigestLogEntry>,
}

/// Why a log is not its manifest's, in the order RFC-0062 'The check' runs its checks. Each is
/// `ERR-CODEC-DIGEST` ([`LogMismatch::code`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogMismatch {
    /// Another number of entries than the manifest's `entry_count`.
    Count {
        /// The manifest's `entry_count`.
        expected: u64,
        /// The log's.
        found: u64,
    },
    /// The entry at `index` does not have `seq = first_seq + index`.
    Seq {
        /// Its position in the log.
        index: u64,
    },
    /// The root over the log's leaves is not the manifest's `entries_root`.
    Root,
}

impl LogMismatch {
    /// `ERR-CODEC-DIGEST`: the log is the verifier's own input, fetched before any proof
    /// (RFC-0062 'Its fault').
    pub fn code(self) -> ErrorCode {
        ErrorCode::CodecDigest
    }
}

impl core::fmt::Display for LogMismatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Count { expected, found } => {
                write!(f, "the log has {found} entries, its manifest {expected}")
            }
            Self::Seq { index } => write!(f, "entry {index} of the log is out of sequence"),
            Self::Root => f.write_str("the log's root is not its manifest's"),
        }
    }
}

impl std::error::Error for LogMismatch {}

impl VerifiedLog {
    /// Checks `entries` against `manifest`, in RFC-0062's order: exactly `entry_count` entries,
    /// the `i`-th with `seq = first_seq + i`, and RFC 9162's root over their leaves equal to
    /// `entries_root`. The manifest's signature is the caller's to check first.
    ///
    /// # Errors
    /// The first [`LogMismatch`] in that order.
    pub fn check(
        manifest: &LogManifest,
        entries: Vec<DigestLogEntry>,
    ) -> Result<Self, LogMismatch> {
        let body = &manifest.body;
        let found = entries.len() as u64;
        if found != body.entry_count {
            return Err(LogMismatch::Count {
                expected: body.entry_count,
                found,
            });
        }
        for (index, entry) in (0u64..).zip(&entries) {
            if body.first_seq.checked_add(index) != Some(entry.seq()) {
                return Err(LogMismatch::Seq { index });
            }
        }
        let leaves: Vec<merkle::Hash> = entries.iter().map(merkle::leaf_hash).collect();
        if merkle::root(&leaves) != body.entries_root {
            return Err(LogMismatch::Root);
        }
        Ok(Self {
            manifest: manifest.clone(),
            entries,
        })
    }

    /// The manifest the log was checked against.
    pub fn manifest(&self) -> &LogManifest {
        &self.manifest
    }

    /// The log's entries, in `seq` order.
    pub fn entries(&self) -> &[DigestLogEntry] {
        &self.entries
    }
}

/// The tokens of a frame whose ciphertext is `length` bytes: `n` for `22 + 4n` (RFC-0059 'The
/// plaintext'). `None` for any other length, which no frame has.
pub fn frame_tokens(length: u32) -> Option<u64> {
    let body = length.checked_sub(22)?;
    (body % 4 == 0).then_some(u64::from(body / 4))
}

/// A request a period draws from (RFC-0063 'Which requests a period draws from').
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Candidate {
    /// The request's session.
    pub session_id: [u8; 16],
    /// The request within its session.
    pub request_id: u64,
    /// Its length `L`: the tokens of all its frames in the period and the one before. A frame
    /// whose length has no token count ([`frame_tokens`]), or a total past `u64`, makes it
    /// `u64::MAX`, longer than every bucket, so the candidate is selected whatever the seed
    /// (RFC-0016 'Sample derivation').
    pub tokens: u64,
    /// Whether it has a `RequestEnd` in the period: an ended candidate, else an unterminated one.
    pub ended: bool,
}

/// The candidates of the period `current`, with `previous` the verified log of the last period
/// of non-zero length before it, or `None` when it has none (RFC-0063 'The previous period's
/// log'). The caller binds the two by the chain. They come out in `(session_id, request_id)`
/// order.
///
/// A period of zero length has none: it holds no entry, and the next period of non-zero length
/// draws the unterminated requests of the one before it (RFC-0016 'Sample derivation').
pub fn candidates(previous: Option<&VerifiedLog>, current: &VerifiedLog) -> Vec<Candidate> {
    let body = &current.manifest().body;
    if body.period_start_ns == body.period_end_ns {
        return Vec::new();
    }
    let before = previous.map_or(&[][..], VerifiedLog::entries);
    let now = current.entries();
    let ends = |log: &[DigestLogEntry]| -> BTreeSet<([u8; 16], u64)> {
        log.iter()
            .filter_map(|e| match *e {
                DigestLogEntry::RequestEnd {
                    session_id,
                    request_id,
                    ..
                } => Some((session_id, request_id)),
                _ => None,
            })
            .collect()
    };
    let ended = ends(now);
    let ended_before = ends(before);
    // Every request's length over its frames in both periods.
    let mut tokens: BTreeMap<([u8; 16], u64), u64> = BTreeMap::new();
    let mut framed_before = BTreeSet::new();
    for (log, is_before) in [(before, true), (now, false)] {
        for entry in log {
            if let DigestLogEntry::Frame {
                session_id,
                request_id,
                length,
                ..
            } = *entry
            {
                let key = (session_id, request_id);
                if is_before {
                    framed_before.insert(key);
                }
                let total = tokens.entry(key).or_insert(0);
                *total = frame_tokens(length)
                    .and_then(|n| total.checked_add(n))
                    .unwrap_or(u64::MAX);
            }
        }
    }
    let unterminated = framed_before
        .into_iter()
        .filter(|key| !ended_before.contains(key) && !ended.contains(key));
    let mut keys: Vec<([u8; 16], u64)> = ended.iter().copied().chain(unterminated).collect();
    keys.sort_unstable();
    keys.into_iter()
        .map(|key| Candidate {
            session_id: key.0,
            request_id: key.1,
            tokens: tokens.get(&key).copied().unwrap_or(0),
            ended: ended.contains(&key),
        })
        .collect()
}

/// The lengths a request's statement carries (03 §4 `RequestRef`), over its frames in the
/// window: all its `Frame` entries in the period and the previous one, as a candidate's length
/// counts them (RFC-0063 'Which requests a period draws from').
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestLengths {
    /// `prompt_len`: the tokens of its `In` frames.
    pub prompt_len: u32,
    /// `output_len`: the tokens of its `Out` frames.
    pub output_len: u32,
}

/// The lengths of `requests`, each a `(session_id, request_id)`, over the window of `current`,
/// `previous` as [`candidates`] takes it; in the order of `requests`. A request with no frame in
/// the window has lengths 0 and 0.
///
/// `None` for a request with a frame whose `length` has no token count ([`frame_tokens`]), or a
/// count past `u32::MAX`: no statement can carry its lengths. Such a candidate is selected
/// whatever the seed (RFC-0016 'Sample derivation', step 4), and the pre-check refuses it
/// ([`PrecheckKind::Lengths`](crate::PrecheckKind::Lengths)).
pub fn lengths(
    previous: Option<&VerifiedLog>,
    current: &VerifiedLog,
    requests: &[([u8; 16], u64)],
) -> Vec<Option<RequestLengths>> {
    // Each request's tokens in and out; `None` once a frame has no count or a sum overflows.
    let mut counts: BTreeMap<([u8; 16], u64), [Option<u64>; 2]> = requests
        .iter()
        .map(|key| (*key, [Some(0), Some(0)]))
        .collect();
    let before = previous.map_or(&[][..], VerifiedLog::entries);
    for entry in before.iter().chain(current.entries()) {
        if let DigestLogEntry::Frame {
            session_id,
            request_id,
            direction,
            length,
            ..
        } = *entry
            && let Some(count) = counts.get_mut(&(session_id, request_id))
        {
            let slot = &mut count[usize::from(direction == Direction::Out)];
            *slot = slot.and_then(|total| total.checked_add(frame_tokens(length)?));
        }
    }
    requests
        .iter()
        .map(|key| {
            let [prompt, output] = counts.get(key).copied()?;
            Some(RequestLengths {
                prompt_len: u32::try_from(prompt?).ok()?,
                output_len: u32::try_from(output?).ok()?,
            })
        })
        .collect()
}
