// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The prover's pre-check (RFC-0015 'Prover pre-check', with RFC-0062 'The pre-check' and
//! RFC-0063 'The request's rules'): before `Round1`, the prover reads the batch's requests from
//! the period's verified log and the previous period's, as the verifier will, and replays its
//! stored transcripts against them. A batch that fails is not proven: `ERR-WITNESS-ANCHOR`.
//!
//! For each request, in batch order, the checks run in the order the verifier's do:
//! 1. **Its session**: RFC-0062's rules in each period that holds a frame of the request, the
//!    previous period first, and one key commitment across both (RFC-0063 rule 4). One open or
//!    carry, before the session's other entries, and one key are [`PrecheckKind::Session`]; one
//!    frame per position and no counter used twice are [`PrecheckKind::Frames`] (RFC-0063 'The
//!    kinds'). A period's rules are checked in their order over the whole period.
//! 2. **Its frames** ([`PrecheckKind::Frames`]): RFC-0063 rules 1, 2, 3 and 5. One end, in the
//!    period; every frame before it, and at least one; every inbound frame before the first
//!    outbound one; each `(direction, frame_index)` once, the indices `0, …, n − 1`.
//! 3. **Its lengths** ([`PrecheckKind::Lengths`]): `prompt_len` and `output_len` are the token
//!    counts of all its `In` and `Out` frames in the window.
//! 4. **The window's counters**, at the session's first request in the batch: no two frames of the
//!    batch's requests in the session share a counter across both periods
//!    ([`PrecheckKind::Frames`], RFC-0063 'One counter across the window').
//! 5. **The replay**, against the session's stored transcript: one held
//!    ([`PrecheckKind::Transcript`]); its key commitment ([`PrecheckKind::KeyCommitment`]); and,
//!    per frame, the stored plaintext at the frame's counter, whose request, direction, index and
//!    length are the entry's and whose digest, re-sealed, is the entry's
//!    ([`PrecheckKind::Digest`]).
//!
//! The sampler key's commitment and containment (RFC-0061) are checked with the in-circuit PRF's
//! inputs (#152); a greedy batch needs no pin.
//!
//! Everything a failure reports is public: the request's identifier and the kind.

use std::collections::{BTreeMap, BTreeSet};

use tessera_codec::error::ErrorCode;

use crate::frame::Direction;
use crate::log::DigestLogEntry;
use crate::period::{VerifiedLog, frame_tokens};
use crate::session::{digest_matches, key_commitment};
use crate::transcript::SessionTranscript;

/// A request of the batch, as its statement lists it (03 §4 `RequestRef`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchRequest {
    /// The session.
    pub session_id: [u8; 16],
    /// The request within its session.
    pub request_id: u64,
    /// Its prompt's tokens: those of its `In` frames.
    pub prompt_len: u32,
    /// Its output's tokens: those of its `Out` frames.
    pub output_len: u32,
}

/// Which check a request failed, in the order they run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrecheckKind {
    /// RFC-0062's rule 1, or two periods' different key commitments (RFC-0063 rule 4).
    Session,
    /// RFC-0063 rules 1, 2, 3 or 5, RFC-0062's rules 2 or 3, or a counter used twice across the
    /// window.
    Frames,
    /// The statement's lengths are not its frames' token counts.
    Lengths,
    /// No stored transcript of the request's session.
    Transcript,
    /// The stored keys do not open the session's key commitment.
    KeyCommitment,
    /// A logged frame has no stored plaintext at its counter, or one that is not the entry's, or
    /// whose re-sealed digest is not the entry's.
    Digest,
}

/// The first failed check of a batch: the request and the kind (`ERR-WITNESS-ANCHOR`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrecheckFailure {
    /// The request that failed.
    pub request_id: u64,
    /// The check it failed.
    pub kind: PrecheckKind,
}

impl PrecheckFailure {
    /// `ERR-WITNESS-ANCHOR`: the prover's transcripts do not match the log, so it cannot prove
    /// the batch (04 §3).
    pub fn code(self) -> ErrorCode {
        ErrorCode::WitnessAnchor
    }
}

impl core::fmt::Display for PrecheckFailure {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let what = match self.kind {
            PrecheckKind::Session => {
                "its session has no single key before its other entries, or two keys"
            }
            PrecheckKind::Frames => "its frames break a rule of RFC-0062 or RFC-0063",
            PrecheckKind::Lengths => "its lengths are not its frames' token counts",
            PrecheckKind::Transcript => "no transcript of its session is stored",
            PrecheckKind::KeyCommitment => "the stored keys do not open its session's commitment",
            PrecheckKind::Digest => "a stored frame does not re-seal to its logged digest",
        };
        write!(f, "request {}: {what}", self.request_id)
    }
}

impl std::error::Error for PrecheckFailure {}

/// A frame entry of the request, with the period it lies in: 0 for the previous one.
#[derive(Clone, Copy)]
struct Frame {
    seq: u64,
    direction: Direction,
    frame_index: u32,
    length: u32,
    frame_seq: u32,
    sha256: [u8; 32],
}

/// RFC-0062's session rules for `session` over one period's entries, and its key commitment
/// there. The rules are checked in their order over the whole period, so a session that breaks
/// rules 1 and 3 fails under rule 1 (RFC-0062 'The session rules'):
/// - rule 1, one open or carry before the session's other entries, is [`PrecheckKind::Session`];
/// - rules 2 and 3, one frame per position and no counter twice, are [`PrecheckKind::Frames`]
///   (RFC-0063 'The kinds').
fn session_key(entries: &[DigestLogEntry], session: [u8; 16]) -> Result<[u8; 32], PrecheckKind> {
    let mut keys = Vec::new();
    let mut first_other = None;
    let mut positions = BTreeMap::new();
    let mut counters = BTreeMap::new();
    let mut repeated = false;
    for e in entries {
        match *e {
            DigestLogEntry::SessionOpen {
                seq,
                session_id,
                key_commitment,
                ..
            }
            | DigestLogEntry::SessionCarry {
                seq,
                session_id,
                key_commitment,
                ..
            } if session_id == session => keys.push((seq, key_commitment)),
            DigestLogEntry::Frame {
                seq,
                session_id,
                request_id,
                direction,
                frame_index,
                frame_seq,
                ..
            } if session_id == session => {
                first_other.get_or_insert(seq);
                repeated |= positions
                    .insert((request_id, direction.byte(), frame_index), ())
                    .is_some();
                repeated |= counters.insert(frame_seq, ()).is_some();
            }
            DigestLogEntry::SessionClose {
                seq, session_id, ..
            }
            | DigestLogEntry::RequestEnd {
                seq, session_id, ..
            } if session_id == session => {
                first_other.get_or_insert(seq);
            }
            _ => {}
        }
    }
    // Rule 1: exactly one open or carry, before every other entry of the session.
    let [(seq, key)] = keys[..] else {
        return Err(PrecheckKind::Session);
    };
    if first_other.is_some_and(|other| other < seq) {
        return Err(PrecheckKind::Session);
    }
    if repeated {
        return Err(PrecheckKind::Frames);
    }
    Ok(key)
}

/// Checks one request of `batch`; `Err` names the first kind it fails. `first_of_session` says
/// whether it is its session's first request in the batch, which answers for the window's
/// counters.
fn check_request(
    batch: &[BatchRequest],
    request: &BatchRequest,
    first_of_session: bool,
    periods: &[&[DigestLogEntry]],
    transcripts: &BTreeMap<[u8; 16], &SessionTranscript>,
) -> Result<(), PrecheckKind> {
    let key = (request.session_id, request.request_id);
    // The window: the request's frames and ends in both periods.
    let mut frames: Vec<(usize, Frame)> = Vec::new();
    let mut ends: Vec<(usize, u64)> = Vec::new();
    for (period, entries) in periods.iter().enumerate() {
        for e in entries.iter() {
            match *e {
                DigestLogEntry::Frame {
                    seq,
                    session_id,
                    request_id,
                    direction,
                    frame_index,
                    length,
                    frame_seq,
                    sha256,
                    ..
                } if (session_id, request_id) == key => frames.push((
                    period,
                    Frame {
                        seq,
                        direction,
                        frame_index,
                        length,
                        frame_seq,
                        sha256,
                    },
                )),
                DigestLogEntry::RequestEnd {
                    seq,
                    session_id,
                    request_id,
                    ..
                } if (session_id, request_id) == key => ends.push((period, seq)),
                _ => {}
            }
        }
    }
    let current = periods.len() - 1;
    // 1. The session's rules in each period that holds a frame, with one key across them.
    let mut commitment = None;
    for (period, entries) in periods.iter().enumerate() {
        if !frames.iter().any(|(p, _)| *p == period) {
            continue;
        }
        let k = session_key(entries, request.session_id)?;
        if commitment.is_some_and(|c| c != k) {
            return Err(PrecheckKind::Session);
        }
        commitment = Some(k);
    }
    // 2. Rule 1: one end, in the period. Rule 2: every frame before it, and at least one.
    let [(end_period, end_seq)] = ends[..] else {
        return Err(PrecheckKind::Frames);
    };
    if end_period != current || frames.is_empty() || frames.iter().any(|(_, f)| f.seq > end_seq) {
        return Err(PrecheckKind::Frames);
    }
    // Rule 3: every inbound frame before the first outbound one.
    let first_out = frames
        .iter()
        .filter(|(_, f)| f.direction == Direction::Out)
        .map(|(_, f)| f.seq)
        .min();
    if let Some(first_out) = first_out
        && frames
            .iter()
            .any(|(_, f)| f.direction == Direction::In && f.seq > first_out)
    {
        return Err(PrecheckKind::Frames);
    }
    // Rule 5: each (direction, index) once, the indices 0..n−1 in each direction.
    for direction in [Direction::In, Direction::Out] {
        let mut indices: Vec<u32> = frames
            .iter()
            .filter(|(_, f)| f.direction == direction)
            .map(|(_, f)| f.frame_index)
            .collect();
        indices.sort_unstable();
        if indices.iter().zip(0u32..).any(|(index, i)| *index != i) {
            return Err(PrecheckKind::Frames);
        }
    }
    // 3. The lengths, over every frame in the window.
    let count = |direction: Direction| -> Option<u64> {
        frames
            .iter()
            .filter(|(_, f)| f.direction == direction)
            .try_fold(0u64, |total, (_, f)| {
                total.checked_add(frame_tokens(f.length)?)
            })
    };
    if count(Direction::In) != Some(u64::from(request.prompt_len))
        || count(Direction::Out) != Some(u64::from(request.output_len))
    {
        return Err(PrecheckKind::Lengths);
    }
    // 4. The window's counters, at the session's first request in the batch.
    if first_of_session && !counters_distinct(batch, request.session_id, periods) {
        return Err(PrecheckKind::Frames);
    }
    // 5. The replay.
    let transcript = transcripts
        .get(&request.session_id)
        .ok_or(PrecheckKind::Transcript)?;
    let keys = transcript.keys();
    if commitment != Some(key_commitment(&keys.key, &keys.key_salt)) {
        return Err(PrecheckKind::KeyCommitment);
    }
    for (_, f) in &frames {
        // The digest covers the header, whose request, direction, index and length are the
        // stored plaintext's own, so a stored frame of other fields fails it too.
        let stored = transcript.frame(f.frame_seq).ok_or(PrecheckKind::Digest)?;
        if !digest_matches(
            keys,
            &request.session_id,
            f.frame_seq,
            &stored.frame,
            &f.sha256,
        ) {
            return Err(PrecheckKind::Digest);
        }
    }
    Ok(())
}

/// The pre-check of a batch, before `Round1`: `current` the batch's period's verified log, and
/// `previous` the last period of non-zero length before it, or `None` when there is none
/// (RFC-0063 'The previous period's log'). `transcripts` are the stored transcripts of the
/// batch's sessions.
///
/// # Errors
/// The first [`PrecheckFailure`], requests in batch order and each request's checks in the
/// module's order.
pub fn precheck(
    batch: &[BatchRequest],
    transcripts: &[SessionTranscript],
    current: &VerifiedLog,
    previous: Option<&VerifiedLog>,
) -> Result<(), PrecheckFailure> {
    let mut periods: Vec<&[DigestLogEntry]> = Vec::with_capacity(2);
    if let Some(previous) = previous {
        periods.push(previous.entries());
    }
    periods.push(current.entries());
    let by_session: BTreeMap<[u8; 16], &SessionTranscript> =
        transcripts.iter().map(|t| (t.session_id(), t)).collect();
    let mut sessions_seen = BTreeSet::new();
    for request in batch {
        let failure = |kind| PrecheckFailure {
            request_id: request.request_id,
            kind,
        };
        let first_of_session = sessions_seen.insert(request.session_id);
        check_request(batch, request, first_of_session, &periods, &by_session).map_err(failure)?;
    }
    Ok(())
}

/// RFC-0063 'One counter across the window': no two frames of the batch's requests in `session`
/// share a `frame_seq`, over both periods together.
fn counters_distinct(
    batch: &[BatchRequest],
    session: [u8; 16],
    periods: &[&[DigestLogEntry]],
) -> bool {
    let requests: BTreeSet<u64> = batch
        .iter()
        .filter(|r| r.session_id == session)
        .map(|r| r.request_id)
        .collect();
    let mut counters = BTreeSet::new();
    periods.iter().flat_map(|p| p.iter()).all(|e| match *e {
        DigestLogEntry::Frame {
            session_id,
            request_id,
            frame_seq,
            ..
        } if session_id == session && requests.contains(&request_id) => counters.insert(frame_seq),
        _ => true,
    })
}
