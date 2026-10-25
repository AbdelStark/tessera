// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The anchor checks of a batch's requests over their window: the session rules (RFC-0062 'The
//! session rules'), the request's rules (RFC-0063 'The request's rules'), RFC-0059's frame checks
//! and the statement's lengths (RFC-0015 check 2, RFC-0059 'Verifier check 2, amended'), and the
//! sampler key's containment (RFC-0061 'Which pin applies').

use super::entry::{DigestLogEntry, Direction, FrameEntry};
use super::log::Window;
use super::pins::SamplerPins;
use super::{Failure, usage};
use crate::protocol::{Decoding, PredictorSpec, PublicStatement, RequestRef};
use crate::verdict::{AnchorFailure, ErrorCode, RejectReason, VerifierError};

/// Runs the anchor checks of `statement`'s batch over `window`, under the loaded `pins`, in
/// `statement.batch` order; the first failure is the verdict (RFC-0043 'Anchor').
///
/// First, the window must be the statement's: `P`'s manifest identity is
/// `statement.manifest_digest`, else the verifier's inputs disagree, `Inconclusive(ERR-USAGE)`
/// (RFC-0019 step 4; the statement's own check, `StatementMismatch { field: ManifestDigest }`,
/// came before). Then, for each request `r = (session_id, request_id)`, its frames being every
/// `Frame` entry of `r` in `P − 1` and `P`:
///
/// 1. **Its session** (RFC-0063 rule 4, with RFC-0062's session rules), in `P − 1` then `P`, in
///    each period that holds a frame of `r`, the rules in their order: exactly one open or carry
///    of the session, before its other entries (`Session`); no two of its frames at one
///    `(request_id, direction, frame_index)` (`Frames`); no two with one `frame_seq`
///    (`Frames`). Where both periods hold a frame of `r`, their opens or carries carry one key
///    commitment (`Session`).
/// 2. **Its frames**, each `Frames`: exactly one `RequestEnd` of `r` in `P − 1` and `P`, and it
///    lies in `P` (rule 1); at least one frame, every one before the end (rule 2); every inbound
///    frame before the first outbound one (rule 3); in each direction the indices exactly
///    `0, …, n − 1`, each once (rule 5); every `length` `22 + 4n` (RFC-0059); at the first
///    request of its session in the batch, no counter used both in `P − 1` and in `P` by the
///    frames of the batch's requests of the session (RFC-0059's distinct counters of a session's
///    frames in the batch, across the boundary); and `prompt_len` and `output_len` the token
///    counts of its inbound and outbound frames (RFC-0062 'Its lengths').
/// 3. **Its sampler key**, when the predictor decodes with temperature, each `SamplerKey`: a pin
///    the policy admits has a window containing the earliest `t_ns` of `r`'s frames; every frame
///    of `r` has `t_ns < from_ns + W + G` for it; and it is the batch's first request's pin.
///
/// Each failure is `Reject(Anchor { request_id: Some(r), kind })`, naming the request being
/// checked: a session that breaks a rule is named by the first request of the batch that reads
/// it, which is the first of the session's requests with a frame in that period.
///
/// # Errors
///
/// The first failure; `Inconclusive(ERR-RES-MEMORY)` when memory for the indices cannot be
/// reserved.
pub fn check_batch(
    statement: &PublicStatement,
    window: &Window<'_>,
    pins: &SamplerPins,
) -> Result<(), Failure> {
    if statement.manifest_digest != *window.period().manifest_digest() {
        return Err(usage());
    }
    let requests = statement.batch.requests.as_slice();
    let index = Index::gather(requests, window)?;
    let temperature = decodes_with_temperature(&statement.predictor);
    let mut first_pin: Option<usize> = None;
    let mut counted = vec_with(index.sessions.len())?;
    counted.resize(index.sessions.len(), false);
    for (position, request) in requests.iter().enumerate() {
        let reject = |kind| {
            Failure::Rejected(RejectReason::Anchor {
                request_id: Some(request.request_id),
                kind,
            })
        };
        let at = index.find(&request.session_id).ok_or_else(internal)?;
        let session = index.sessions.get(at).ok_or_else(internal)?;
        let [before, here] = &session.periods;
        let (frames_before, ends_before) = before.of(request.request_id)?;
        let (frames_here, ends_here) = here.of(request.request_id)?;

        // 1. The session, in each period holding a frame of the request, then one key across them.
        let mut keys: [Option<[u8; 32]>; 2] = [None, None];
        for (slot, (period, frames)) in keys
            .iter_mut()
            .zip([(before, &frames_before), (here, &frames_here)])
        {
            if !frames.is_empty() {
                *slot = Some(period.rules.map_err(reject)?);
            }
        }
        if let [Some(a), Some(b)] = keys
            && a != b
        {
            return Err(reject(AnchorFailure::Session));
        }

        // 2. The request's frames.
        let frames: Vec<&FrameEntry> = joined(&frames_before, &frames_here)?;
        request_rules(request, &frames, ends_before.len(), &ends_here).map_err(|e| match e {
            Broken::Kind(kind) => reject(kind),
            Broken::Verifier(failure) => failure,
        })?;
        let seen = counted.get_mut(at).ok_or_else(internal)?;
        if !*seen {
            *seen = true;
            if session.counter_reused()? {
                return Err(reject(AnchorFailure::Frames));
            }
        }

        // 3. The sampler key.
        if temperature {
            let pin = pin_of(&frames, pins).ok_or(reject(AnchorFailure::SamplerKey))?;
            match first_pin {
                None if position == 0 => first_pin = Some(pin),
                Some(first) if first == pin => {}
                _ => return Err(reject(AnchorFailure::SamplerKey)),
            }
        }
    }
    Ok(())
}

/// Whether the predictor decodes with temperature, so that its requests need a pin (RFC-0061).
fn decodes_with_temperature(predictor: &PredictorSpec) -> bool {
    let (PredictorSpec::Exact { decoding } | PredictorSpec::Mixture { decoding, .. }) = predictor;
    matches!(decoding, Decoding::Temperature { .. })
}

/// A request's rules other than its session's, all `Frames` (RFC-0063 rules 1, 2, 3 and 5,
/// RFC-0059's lengths, RFC-0062 'Its lengths'). `frames` are the request's frames in `P − 1` then
/// `P`, in `seq` order; `ends_before` counts its ends in `P − 1`, and `ends_here` lists its ends'
/// `seq` in `P`.
fn request_rules(
    request: &RequestRef,
    frames: &[&FrameEntry],
    ends_before: usize,
    ends_here: &[u64],
) -> Result<(), Broken> {
    let frames_broken = Err(Broken::Kind(AnchorFailure::Frames));
    // Rule 1: one end, in P.
    let &[end] = ends_here else {
        return frames_broken;
    };
    if ends_before != 0 {
        return frames_broken;
    }
    // Rule 2: at least one frame, all before the end.
    if frames.is_empty() || frames.iter().any(|frame| frame.seq >= end) {
        return frames_broken;
    }
    // Rule 3: every inbound frame before the first outbound one; the frames are in seq order.
    let first_out = frames
        .iter()
        .position(|frame| frame.direction == Direction::Out);
    if let Some(first_out) = first_out
        && frames
            .iter()
            .skip(first_out)
            .any(|frame| frame.direction == Direction::In)
    {
        return frames_broken;
    }
    // Rule 5 and the lengths, direction by direction.
    for (direction, stated) in [
        (Direction::In, request.prompt_len),
        (Direction::Out, request.output_len),
    ] {
        let mut indices = vec_with(frames.len()).map_err(Broken::Verifier)?;
        let mut tokens: u64 = 0;
        for frame in frames.iter().filter(|frame| frame.direction == direction) {
            // RFC-0059: a length no plaintext has.
            let Some(count) = frame.tokens() else {
                return frames_broken;
            };
            tokens = tokens.saturating_add(u64::from(count));
            indices.push(frame.frame_index);
        }
        indices.sort_unstable();
        let contiguous = indices.iter().enumerate().all(|(expected, &index)| {
            u64::try_from(expected).is_ok_and(|expected| expected == u64::from(index))
        });
        if !contiguous {
            return frames_broken;
        }
        if tokens != u64::from(stated) {
            return frames_broken;
        }
    }
    Ok(())
}

/// The pin `frames` fall within (RFC-0061 'Which pin applies'): the admitted pin whose window
/// contains their earliest `t_ns`, when every frame has `t_ns < from_ns + W + G` for it.
fn pin_of(frames: &[&FrameEntry], pins: &SamplerPins) -> Option<usize> {
    let earliest = frames.iter().map(|frame| frame.t_ns).min()?;
    let (index, pin) = pins.containing(earliest)?;
    let limit = pin.limit()?;
    frames
        .iter()
        .all(|frame| frame.t_ns < limit)
        .then_some(index)
}

/// A request rule's failure: a rejection's kind, or the verifier's own fault.
enum Broken {
    Kind(AnchorFailure),
    Verifier(Failure),
}

/// The batch's sessions and their entries in the window, gathered in one pass over each log.
struct Index<'a> {
    /// Sorted by session, one per distinct session of the batch.
    sessions: Vec<Session<'a>>,
}

/// One session of the batch: its entries in `P − 1` and `P`, and its requests in the batch.
struct Session<'a> {
    id: [u8; 16],
    /// `P − 1`, then `P`.
    periods: [Period<'a>; 2],
    /// The distinct ids of its requests in the batch, sorted.
    requests: Vec<u64>,
}

/// A session's entries in one period.
struct Period<'a> {
    /// Every entry of the session, in `seq` order.
    entries: Vec<&'a DigestLogEntry>,
    /// Its frames and request ends, sorted by request and then by `seq`.
    by_request: Vec<&'a DigestLogEntry>,
    /// The session rules in this period: the key of its one open or carry, or the first rule it
    /// breaks.
    rules: Result<[u8; 32], AnchorFailure>,
}

impl<'a> Index<'a> {
    fn gather(requests: &[RequestRef], window: &Window<'a>) -> Result<Self, Failure> {
        let mut ids = vec_with(requests.len())?;
        ids.extend(requests.iter().map(|request| request.session_id));
        ids.sort_unstable();
        ids.dedup();
        let mut sessions = vec_with(ids.len())?;
        for id in ids {
            sessions.push(Session {
                id,
                periods: [Period::empty(), Period::empty()],
                requests: Vec::new(),
            });
        }
        let mut index = Self { sessions };
        for request in requests {
            let at = index.find(&request.session_id).ok_or_else(internal)?;
            let session = index.sessions.get_mut(at).ok_or_else(internal)?;
            push(&mut session.requests, request.request_id)?;
        }
        let logs = [
            window
                .previous()
                .map(|log| log.entries())
                .unwrap_or_default(),
            window.period().entries(),
        ];
        for (slot, entries) in logs.into_iter().enumerate() {
            for entry in entries {
                let Some(at) = index.find(entry.session_id()) else {
                    continue;
                };
                let session = index.sessions.get_mut(at).ok_or_else(internal)?;
                let period = session.periods.get_mut(slot).ok_or_else(internal)?;
                push(&mut period.entries, entry)?;
                if request_of(entry).is_some() {
                    push(&mut period.by_request, entry)?;
                }
            }
        }
        for session in &mut index.sessions {
            session.requests.sort_unstable();
            session.requests.dedup();
            for period in &mut session.periods {
                // A stable sort keeps each request's entries in seq order.
                period.by_request.sort_by_key(|entry| request_of(entry));
                period.rules = session_rules(&period.entries)?;
            }
        }
        Ok(index)
    }

    fn find(&self, id: &[u8; 16]) -> Option<usize> {
        self.sessions
            .binary_search_by(|session| session.id.cmp(id))
            .ok()
    }
}

impl<'a> Period<'a> {
    fn empty() -> Self {
        Self {
            entries: Vec::new(),
            by_request: Vec::new(),
            rules: Err(AnchorFailure::Session),
        }
    }

    /// The request's frames in this period, in `seq` order, and its ends' `seq`.
    #[allow(
        clippy::type_complexity,
        reason = "a pair of the two lists one search yields"
    )]
    fn of(&self, request_id: u64) -> Result<(Vec<&'a FrameEntry>, Vec<u64>), Failure> {
        let start = self
            .by_request
            .partition_point(|entry| request_of(entry) < Some(request_id));
        let end = self
            .by_request
            .partition_point(|entry| request_of(entry) <= Some(request_id));
        let range = self.by_request.get(start..end).ok_or_else(internal)?;
        let mut frames = vec_with(range.len())?;
        let mut ends = Vec::new();
        for &entry in range {
            match entry {
                DigestLogEntry::Frame(frame) => frames.push(frame),
                DigestLogEntry::RequestEnd { seq, .. } => push(&mut ends, *seq)?,
                _ => {}
            }
        }
        Ok((frames, ends))
    }
}

impl Session<'_> {
    /// Whether a counter is used both in `P − 1` and in `P` by frames of the session's requests
    /// in the batch. A counter repeated within one period is that period's rule 3.
    fn counter_reused(&self) -> Result<bool, Failure> {
        let [before, here] = &self.periods;
        let counters = |period: &Period<'_>| -> Result<Vec<u32>, Failure> {
            let mut counters = vec_with(period.by_request.len())?;
            for entry in &period.by_request {
                if let DigestLogEntry::Frame(frame) = entry
                    && self.requests.binary_search(&frame.request_id).is_ok()
                {
                    counters.push(frame.frame_seq);
                }
            }
            counters.sort_unstable();
            counters.dedup();
            Ok(counters)
        };
        let before = counters(before)?;
        let here = counters(here)?;
        Ok(here
            .iter()
            .any(|counter| before.binary_search(counter).is_ok()))
    }
}

/// RFC-0062's session rules over one session's entries in one period, in their order:
/// 1. exactly one `SessionOpen` or `SessionCarry`, with a lower `seq` than every other entry of
///    the session (`Session`);
/// 2. no two frames at one `(request_id, direction, frame_index)` (`Frames`);
/// 3. no two frames with one `frame_seq` (`Frames`).
///
/// The outer error is the verifier's memory; the inner result the rules'.
fn session_rules(entries: &[&DigestLogEntry]) -> Result<Result<[u8; 32], AnchorFailure>, Failure> {
    let key = match entries.split_first() {
        Some((first, rest)) => match first.key_commitment() {
            Some(key) if rest.iter().all(|entry| entry.key_commitment().is_none()) => *key,
            _ => return Ok(Err(AnchorFailure::Session)),
        },
        None => return Ok(Err(AnchorFailure::Session)),
    };
    let mut positions = vec_with(entries.len())?;
    let mut counters = vec_with(entries.len())?;
    for entry in entries {
        if let DigestLogEntry::Frame(frame) = entry {
            positions.push((frame.request_id, frame.direction, frame.frame_index));
            counters.push(frame.frame_seq);
        }
    }
    positions.sort_unstable();
    counters.sort_unstable();
    if positions.windows(2).any(|pair| pair.first() == pair.last())
        || counters.windows(2).any(|pair| pair.first() == pair.last())
    {
        return Ok(Err(AnchorFailure::Frames));
    }
    Ok(Ok(key))
}

/// The request a frame or an end belongs to; `None` for a session's own entries.
fn request_of(entry: &DigestLogEntry) -> Option<u64> {
    match entry {
        DigestLogEntry::Frame(frame) => Some(frame.request_id),
        DigestLogEntry::RequestEnd { request_id, .. } => Some(*request_id),
        _ => None,
    }
}

/// `a` then `b`, in a vector reserved fallibly.
fn joined<'a>(a: &[&'a FrameEntry], b: &[&'a FrameEntry]) -> Result<Vec<&'a FrameEntry>, Failure> {
    let mut out = vec_with(a.len().saturating_add(b.len()))?;
    out.extend_from_slice(a);
    out.extend_from_slice(b);
    Ok(out)
}

/// An empty vector with room for `capacity` values, reserved fallibly.
fn vec_with<T>(capacity: usize) -> Result<Vec<T>, Failure> {
    let mut out = Vec::new();
    out.try_reserve_exact(capacity).map_err(|_| memory())?;
    Ok(out)
}

/// Pushes `value`, growing the vector fallibly.
fn push<T>(values: &mut Vec<T>, value: T) -> Result<(), Failure> {
    if values.len() == values.capacity() {
        values
            .try_reserve(values.len().max(8))
            .map_err(|_| memory())?;
    }
    values.push(value);
    Ok(())
}

fn memory() -> Failure {
    Failure::Inconclusive(VerifierError {
        code: ErrorCode::Memory,
    })
}

fn internal() -> Failure {
    Failure::Inconclusive(VerifierError {
        code: ErrorCode::Internal,
    })
}
