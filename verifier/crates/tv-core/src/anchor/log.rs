// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A period's verified log (RFC-0062 'The period's log'), the window of a period and the one
//! before it (RFC-0063 'The periods', 'The previous period's log'), and the period's candidates
//! (RFC-0063 'Which requests a period draws from', RFC-0016 'Sample derivation' step 4).

use tv_codec::{DecodeError, DecodeLimits, ErrorCode, decode};
use tv_hash::Digest;
use tv_hash::rfc6962::{TreeHash, leaf_hash};

use super::entry::{DigestLogEntry, cut};
use super::manifest::CheckedManifest;
use super::{Failure, usage};
use crate::verdict::{ErrorCode as Code, VerifierError};

/// A period's log, checked whole against its manifest: the only form in which the verifier
/// reads a period's entries (RFC-0062 'The verified log'). Only [`check`](Self::check)
/// constructs one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedLog {
    manifest: CheckedManifest,
    entries: Vec<DigestLogEntry>,
}

impl VerifiedLog {
    /// Checks a period's log, `<period_start_ns>.tdl`, against its manifest (RFC-0062 'The
    /// check'):
    ///
    /// 1. the stream decodes whole, entry after entry, under `limits` (its size first);
    /// 2. it holds exactly `entry_count` entries;
    /// 3. the `i`-th entry's `seq` is `first_seq + i`;
    /// 4. RFC 9162's tree hash of the leaves `SHA-256(0x00 ‖ TCE(entry))` is `entries_root`.
    ///
    /// The manifest has passed its own checks, the signature first ([`CheckedManifest`]), so the
    /// check runs after the signature, as RFC-0062 'When' orders. Each entry's leaf hashes the
    /// bytes it was decoded from, which are its canonical encoding (INV-TCE-2).
    ///
    /// # Memory
    ///
    /// Before decoding, the check reserves room for `min(entry_count, log.len() / 34)` entries,
    /// `size_of::<DigestLogEntry>()` bytes each: never more than the bytes can hold, 34 being the
    /// smallest entry's size, whatever count the manifest signs. A log of more entries than its
    /// manifest counts grows the vector by doubling, to about twice `log.len() / 34` entries at
    /// most. `log.len()` is checked against `limits.max_total_bytes` first, so both bounds
    /// follow the caller's limit. The tree hash runs in constant memory.
    ///
    /// # Errors
    ///
    /// A decoding error, at its offset in the stream, or `ERR-CODEC-DIGEST` for checks 2 to 4, in
    /// that order: at the stream's end for the count and the root, at the first wrong entry's
    /// start for a `seq` (04 §3). The log is the verifier's own input, so each is
    /// `Inconclusive` (RFC-0062 'Its fault'); [`log_failure`] gives the verdict.
    pub fn check(
        manifest: &CheckedManifest,
        log: &[u8],
        limits: &DecodeLimits,
    ) -> Result<Self, DecodeError> {
        let m = manifest.manifest();
        // RFC-0003 'Limits': the whole input against `max_total_bytes` before anything is
        // reserved for it (`decode` checks it again, the same way).
        if u64::try_from(log.len()).map_or(true, |len| len > limits.max_total_bytes) {
            return Err(DecodeError::new(ErrorCode::Limit, 0));
        }
        let mut entries: Vec<DigestLogEntry> = Vec::new();
        // The signed count, but never more entries than the bytes can hold: the smallest entry,
        // a close, is 34 bytes.
        let expected = usize::try_from(m.entry_count)
            .unwrap_or(usize::MAX)
            .min(log.len() / SMALLEST_ENTRY);
        entries
            .try_reserve_exact(expected)
            .map_err(|_| DecodeError::new(ErrorCode::Memory, 0))?;
        let mut tree = TreeHash::new();
        let mut wrong_seq: Option<u64> = None;
        decode(log, limits, |c| {
            cut(c, log, |entry, encoding, at| {
                let index = tree.count();
                if wrong_seq.is_none() && m.first_seq.checked_add(index) != Some(entry.seq()) {
                    wrong_seq = Some(at);
                }
                tree.push(&leaf_hash(encoding))
                    .map_err(|_| DecodeError::new(ErrorCode::Internal, at))?;
                if entries.len() == entries.capacity() {
                    entries
                        .try_reserve(entries.len().max(1))
                        .map_err(|_| DecodeError::new(ErrorCode::Memory, at))?;
                }
                entries.push(entry);
                Ok(())
            })
        })?;
        let end = u64::try_from(log.len()).unwrap_or(u64::MAX);
        if tree.count() != m.entry_count {
            return Err(DecodeError::new(ErrorCode::Digest, end));
        }
        if let Some(at) = wrong_seq {
            return Err(DecodeError::new(ErrorCode::Digest, at));
        }
        if tree.root() != m.entries_root {
            return Err(DecodeError::new(ErrorCode::Digest, end));
        }
        Ok(Self {
            manifest: manifest.clone(),
            entries,
        })
    }

    /// The manifest the log was checked against.
    pub fn manifest(&self) -> &CheckedManifest {
        &self.manifest
    }

    /// The manifest's identity, the digest a statement and the sample must name (RFC-0019
    /// step 4).
    pub fn manifest_digest(&self) -> &Digest {
        self.manifest.identity()
    }

    /// The entries, in `seq` order.
    pub fn entries(&self) -> &[DigestLogEntry] {
        &self.entries
    }
}

/// The smallest encoded entry, `SessionClose`: a tag, `seq`, `t_ns` and the session.
const SMALLEST_ENTRY: usize = 34;

/// The verdict of a log that fails to decode or to match its manifest: `Inconclusive` with the
/// error's code, `ERR-CODEC-*`, or `ERR-RES-MEMORY` and `ERR-INTERNAL` for the reader's own faults
/// (RFC-0062 'Its fault').
pub fn log_failure(error: DecodeError) -> Failure {
    Failure::Inconclusive(VerifierError {
        code: error.code().into(),
    })
}

/// A period and the last period of non-zero length before it, bound by the chain of manifests:
/// the two logs a request's frames are read from (RFC-0063 'The periods').
#[derive(Clone, Copy, Debug)]
pub struct Window<'a> {
    period: &'a VerifiedLog,
    previous: Option<&'a VerifiedLog>,
}

impl<'a> Window<'a> {
    /// Binds `period`, `P`, to `previous`, `P − 1`, through `between`, the manifests of the
    /// periods of zero length between them in chain order (RFC-0063 'The previous period's log',
    /// RFC-0019 step 1). Walking back from `P`:
    ///
    /// 1. each manifest of `between`, from the last, has zero length and no entries, and
    ///    precedes the manifest after it ([`CheckedManifest::precedes`]);
    /// 2. then, with `previous`, its manifest has non-zero length and precedes the earliest
    ///    manifest reached;
    /// 3. without it, that manifest is a log's first, with `prev_manifest` zero and `first_seq`
    ///    0, so no period of non-zero length comes before `P`.
    ///
    /// # Errors
    ///
    /// The verifier's own inputs disagree with the chain: a `previous` that is not `P − 1`'s log,
    /// one missing when `P − 1` exists, or `between` not the zero-length periods between them.
    /// Each is `Inconclusive(ERR-USAGE)`.
    pub fn bind(
        period: &'a VerifiedLog,
        between: &[CheckedManifest],
        previous: Option<&'a VerifiedLog>,
    ) -> Result<Self, Failure> {
        let mut later = period.manifest();
        for zero in between.iter().rev() {
            let m = zero.manifest();
            if !m.is_zero_length() || m.entry_count != 0 || !zero.precedes(later) {
                return Err(usage());
            }
            later = zero;
        }
        match previous {
            Some(log) => {
                if log.manifest().manifest().is_zero_length() || !log.manifest().precedes(later) {
                    return Err(usage());
                }
            }
            None => {
                let m = later.manifest();
                if m.prev_manifest != [0; 32] || m.first_seq != 0 {
                    return Err(usage());
                }
            }
        }
        Ok(Self { period, previous })
    }

    /// `P`'s verified log.
    pub fn period(&self) -> &'a VerifiedLog {
        self.period
    }

    /// `P − 1`'s verified log, `None` when no period of non-zero length comes before `P`.
    pub fn previous(&self) -> Option<&'a VerifiedLog> {
        self.previous
    }

    /// The candidates of `P`, sorted by `(session_id, request_id)` (RFC-0063 'Which requests a
    /// period draws from'):
    ///
    /// - **ended:** every request with a `RequestEnd` in `P`;
    /// - **unterminated:** every request with a frame in `P − 1` and no `RequestEnd` in `P − 1`
    ///   or `P`;
    ///
    /// each with its length `L`, the token count of all its frames in `P − 1` and `P`, or
    /// `2^64 − 1` when one of them has a `length` that is not `22 + 4n` (RFC-0016 'Sample
    /// derivation' step 4). A sum past `2^64 − 1` saturates there, past every bucket. A period
    /// of zero length has none.
    ///
    /// # Errors
    ///
    /// `Inconclusive(ERR-RES-MEMORY)` when memory for the sort cannot be reserved.
    pub fn candidates(&self) -> Result<Vec<Candidate>, Failure> {
        if self.period.manifest().manifest().is_zero_length() {
            return Ok(Vec::new());
        }
        let previous = self.previous.map(VerifiedLog::entries).unwrap_or_default();
        let current = self.period.entries();
        let mut events: Vec<([u8; 16], u64, Event)> = Vec::new();
        events
            .try_reserve_exact(previous.len().saturating_add(current.len()))
            .map_err(|_| memory())?;
        for (entries, in_previous) in [(previous, true), (current, false)] {
            for entry in entries {
                let event = match entry {
                    DigestLogEntry::Frame(frame) => (
                        frame.session_id,
                        frame.request_id,
                        Event::Frame {
                            in_previous,
                            tokens: frame.tokens(),
                        },
                    ),
                    DigestLogEntry::RequestEnd {
                        session_id,
                        request_id,
                        ..
                    } => (*session_id, *request_id, Event::End { in_previous }),
                    _ => continue,
                };
                events.push(event);
            }
        }
        events.sort_unstable_by_key(|&(session, request, _)| (session, request));
        let mut out = Vec::new();
        for group in events.chunk_by(|a, b| (a.0, a.1) == (b.0, b.1)) {
            let Some(&(session_id, request_id, _)) = group.first() else {
                continue;
            };
            let (mut end_here, mut end_before, mut frame_before) = (false, false, false);
            let mut tokens: u64 = 0;
            for (_, _, event) in group {
                match *event {
                    Event::End { in_previous: false } => end_here = true,
                    Event::End { in_previous: true } => end_before = true,
                    Event::Frame {
                        in_previous,
                        tokens: count,
                    } => {
                        frame_before |= in_previous;
                        tokens = match count {
                            Some(n) => tokens.saturating_add(u64::from(n)),
                            None => u64::MAX,
                        };
                    }
                }
            }
            if end_here || (frame_before && !end_before) {
                if out.len() == out.capacity() {
                    out.try_reserve(out.len().max(16)).map_err(|_| memory())?;
                }
                out.push(Candidate {
                    session_id,
                    request_id,
                    ended: end_here,
                    tokens,
                });
            }
        }
        Ok(out)
    }
}

/// What a frame or an end contributes to its request's candidacy.
#[derive(Clone, Copy)]
enum Event {
    Frame {
        in_previous: bool,
        tokens: Option<u32>,
    },
    End {
        in_previous: bool,
    },
}

fn memory() -> Failure {
    Failure::Inconclusive(VerifierError { code: Code::Memory })
}

/// A request a period draws from (RFC-0063 'Which requests a period draws from').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Candidate {
    /// The request's session.
    pub session_id: [u8; 16],
    /// The request.
    pub request_id: u64,
    /// Whether it has a `RequestEnd` in the period; an unterminated candidate has none.
    pub ended: bool,
    /// Its length `L`: the token count of its frames in both periods, `2^64 − 1` when a frame
    /// has an impossible `length`.
    pub tokens: u64,
}
