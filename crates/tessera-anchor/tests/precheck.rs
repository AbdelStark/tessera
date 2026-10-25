// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The prover's pre-check (RFC-0015 'Prover pre-check', RFC-0062, RFC-0063) and the stored
//! transcript it replays (`SessionTranscript`, 0x0404), through the public API:
//! - an honest batch passes, a request read across a period boundary included;
//! - each check alone, broken: the session's rules, the request's rules, the lengths, a missing
//!   transcript, other keys, and a stored frame tampered with or missing, each with its kind and
//!   `ERR-WITNESS-ANCHOR`;
//! - the first failing request of the batch is the one named;
//! - the artifact round trips through its secret-class envelope, and its decoder refuses
//!   counters out of order and an unknown direction at their offsets.

use tessera_anchor::{
    BatchRequest, DigestLogEntry, Direction, HeapKey, LogManifest, ManifestBody, PrecheckFailure,
    PrecheckKind, RequestEndFrame, SealedFrame, Session, SessionOpenFrame, SessionTranscript,
    TokenFrame, VerifiedLog, key_id, merkle, precheck, public_key,
};
use tessera_codec::error::ErrorCode;
use tessera_codec::{DecodeLimits, Secret, decode, encode, envelope, schema};

fn seed() -> Secret<HeapKey<32>> {
    Secret::new(HeapKey::from_box(Box::new([0x47; 32])))
}

/// A log's events, in order.
enum Event {
    Open(SessionOpenFrame),
    Frame(SealedFrame),
    End([u8; 16], u64),
    Carry([u8; 16], [u8; 32]),
}

/// The verified log of `events`, its entries' `seq` from `first_seq`.
fn log(events: &[Event], first_seq: u64) -> VerifiedLog {
    let entries: Vec<DigestLogEntry> = events
        .iter()
        .zip(first_seq..)
        .map(|(e, seq)| match e {
            Event::Open(open) => DigestLogEntry::session_open(seq, 10, open),
            Event::Frame(sealed) => DigestLogEntry::frame(seq, 10, sealed),
            Event::End(session_id, request_id) => DigestLogEntry::request_end(
                seq,
                10,
                &RequestEndFrame {
                    session_id: *session_id,
                    request_id: *request_id,
                },
            ),
            Event::Carry(session_id, key_commitment) => DigestLogEntry::SessionCarry {
                seq,
                t_ns: 10,
                session_id: *session_id,
                key_commitment: *key_commitment,
            },
        })
        .collect();
    let leaves: Vec<_> = entries.iter().map(merkle::leaf_hash).collect();
    let manifest: LogManifest = ManifestBody {
        recorder_key_id: key_id(&public_key(&seed())),
        log_id: [3; 16],
        period_start_ns: 0,
        period_end_ns: 100,
        first_seq,
        entry_count: entries.len() as u64,
        entries_root: merkle::root(&leaves),
        prev_manifest: [0; 32],
    }
    .sign(&seed())
    .expect("a signature");
    VerifiedLog::check(&manifest, entries).expect("its log")
}

fn tokens(n: usize, base: u32) -> Secret<Vec<u32>> {
    Secret::new((0..n as u32).map(|i| base + i).collect())
}

/// A session, its open frame and transcript, and a way to seal and record frames.
struct Served {
    session: Session,
    open: SessionOpenFrame,
    transcript: SessionTranscript,
}

impl Served {
    fn new() -> Self {
        let (session, open, keys) = Session::open().expect("a session");
        let transcript = SessionTranscript::new(session.session_id(), keys);
        Self {
            session,
            open,
            transcript,
        }
    }

    fn id(&self) -> [u8; 16] {
        self.session.session_id()
    }

    /// Seals a frame of `n` tokens and records it.
    fn seal(&mut self, request: u64, direction: Direction, index: u32, n: usize) -> SealedFrame {
        let frame = TokenFrame::new(request, direction, index, tokens(n, 100 * index + n as u32));
        let sealed = self.session.seal(&frame).expect("sealed");
        let again = TokenFrame::new(request, direction, index, tokens(n, 100 * index + n as u32));
        self.transcript.record(&sealed, again).expect("recorded");
        sealed
    }
}

fn request(served: &Served, request_id: u64, prompt_len: u32, output_len: u32) -> BatchRequest {
    BatchRequest {
        session_id: served.id(),
        request_id,
        prompt_len,
        output_len,
    }
}

fn failure(request_id: u64, kind: PrecheckKind) -> Result<(), PrecheckFailure> {
    Err(PrecheckFailure { request_id, kind })
}

/// Request 1 within period 0; request 2 from period 0 into period 1.
fn two_periods() -> (Served, VerifiedLog, VerifiedLog) {
    let mut s = Served::new();
    let id = s.id();
    let p0 = vec![
        Event::Open(s.open),
        Event::Frame(s.seal(1, Direction::In, 0, 5)),
        Event::Frame(s.seal(1, Direction::Out, 0, 9)),
        Event::End(id, 1),
        Event::Frame(s.seal(2, Direction::In, 0, 40)),
    ];
    let p1 = vec![
        Event::Carry(id, s.open.key_commitment),
        Event::Frame(s.seal(2, Direction::Out, 0, 60)),
        Event::Frame(s.seal(2, Direction::Out, 1, 3)),
        Event::End(id, 2),
    ];
    let previous = log(&p0, 0);
    let current = log(&p1, 5);
    (s, previous, current)
}

#[test]
fn an_honest_batch_passes_across_a_period_boundary() {
    let (s, previous, current) = two_periods();
    let r1 = request(&s, 1, 5, 9);
    let r2 = request(&s, 2, 40, 63);
    let transcripts = [s.transcript];
    assert_eq!(precheck(&[r1], &transcripts, &previous, None), Ok(()));
    assert_eq!(
        precheck(&[r2], &transcripts, &current, Some(&previous)),
        Ok(())
    );
    // The window counts the prompt logged in the previous period: a statement without it fails.
    let short = BatchRequest {
        prompt_len: 0,
        ..r2
    };
    assert_eq!(
        precheck(&[short], &transcripts, &current, Some(&previous)),
        failure(2, PrecheckKind::Lengths)
    );
    assert_eq!(
        PrecheckFailure {
            request_id: 2,
            kind: PrecheckKind::Lengths
        }
        .code(),
        ErrorCode::WitnessAnchor
    );
}

#[test]
fn a_tampered_or_missing_transcript_is_refused_before_round1() {
    let (s, previous, current) = two_periods();
    let r2 = request(&s, 2, 40, 63);
    // One token byte of the stored transcript changed: its re-sealed digest is not the log's.
    let mut bytes = encode(&s.transcript);
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    let tampered: SessionTranscript =
        decode(&bytes, DecodeLimits::MESSAGE).expect("still a transcript");
    assert_eq!(
        precheck(&[r2], &[tampered], &current, Some(&previous)),
        failure(2, PrecheckKind::Digest)
    );
    // No transcript of the session.
    assert_eq!(
        precheck(&[r2], &[], &current, Some(&previous)),
        failure(2, PrecheckKind::Transcript)
    );
    // Another session's keys under this session's identifier.
    let (_, _, other_keys) = Session::open().expect("a session");
    let foreign = SessionTranscript::new(s.id(), other_keys);
    assert_eq!(
        precheck(&[r2], &[foreign], &current, Some(&previous)),
        failure(2, PrecheckKind::KeyCommitment)
    );
    // A transcript that lacks a logged frame: request 3's only frame was sealed but not kept.
    let mut t = Served::new();
    let id = t.id();
    let frame = TokenFrame::new(3, Direction::In, 0, tokens(4, 7));
    let unkept = t.session.seal(&frame).expect("sealed");
    let events = [Event::Open(t.open), Event::Frame(unkept), Event::End(id, 3)];
    assert_eq!(
        precheck(
            &[request(&t, 3, 4, 0)],
            &[t.transcript],
            &log(&events, 0),
            None
        ),
        failure(3, PrecheckKind::Digest)
    );
}

/// Request 7's frames and end in `events`, checked alone.
fn alone(s: Served, events: &[Event], prompt: u32, output: u32) -> Result<(), PrecheckFailure> {
    let r = request(&s, 7, prompt, output);
    precheck(&[r], &[s.transcript], &log(events, 0), None)
}

#[test]
fn each_request_rule_is_refused_alone() {
    // Rule 1: no end in the period.
    let mut s = Served::new();
    let events = [
        Event::Open(s.open),
        Event::Frame(s.seal(7, Direction::In, 0, 2)),
    ];
    assert_eq!(alone(s, &events, 2, 0), failure(7, PrecheckKind::Frames));
    // Rule 1: two ends.
    let mut s = Served::new();
    let id = s.id();
    let events = [
        Event::Open(s.open),
        Event::Frame(s.seal(7, Direction::In, 0, 2)),
        Event::End(id, 7),
        Event::End(id, 7),
    ];
    assert_eq!(alone(s, &events, 2, 0), failure(7, PrecheckKind::Frames));
    // Rule 2: a frame after the end.
    let mut s = Served::new();
    let id = s.id();
    let first = s.seal(7, Direction::In, 0, 2);
    let late = s.seal(7, Direction::Out, 0, 1);
    let events = [
        Event::Open(s.open),
        Event::Frame(first),
        Event::End(id, 7),
        Event::Frame(late),
    ];
    assert_eq!(alone(s, &events, 2, 1), failure(7, PrecheckKind::Frames));
    // Rule 2: an end with no frame before it.
    let s = Served::new();
    let id = s.id();
    let events = [Event::Open(s.open), Event::End(id, 7)];
    assert_eq!(alone(s, &events, 0, 0), failure(7, PrecheckKind::Frames));
    // Rule 3: an inbound frame after an outbound one.
    let mut s = Served::new();
    let id = s.id();
    let out = s.seal(7, Direction::Out, 0, 1);
    let inbound = s.seal(7, Direction::In, 0, 2);
    let events = [
        Event::Open(s.open),
        Event::Frame(out),
        Event::Frame(inbound),
        Event::End(id, 7),
    ];
    assert_eq!(alone(s, &events, 2, 1), failure(7, PrecheckKind::Frames));
    // Rule 5: an index missing in a direction.
    let mut s = Served::new();
    let id = s.id();
    let events = [
        Event::Open(s.open),
        Event::Frame(s.seal(7, Direction::In, 1, 2)),
        Event::End(id, 7),
    ];
    assert_eq!(alone(s, &events, 2, 0), failure(7, PrecheckKind::Frames));
    // Rule 1: the request's one end lies in the period before, not in the period proven.
    let (s, previous, current) = two_periods();
    assert_eq!(
        precheck(
            &[request(&s, 1, 5, 9)],
            &[s.transcript],
            &current,
            Some(&previous)
        ),
        failure(1, PrecheckKind::Frames)
    );
    // The control: the same request with its index 0 passes.
    let mut s = Served::new();
    let id = s.id();
    let events = [
        Event::Open(s.open),
        Event::Frame(s.seal(7, Direction::In, 0, 2)),
        Event::Frame(s.seal(7, Direction::Out, 0, 3)),
        Event::End(id, 7),
    ];
    assert_eq!(alone(s, &events, 2, 3), Ok(()));
}

#[test]
fn each_session_rule_is_refused_alone() {
    // Rule 1: a frame of the session before its open.
    let mut s = Served::new();
    let id = s.id();
    let early = s.seal(7, Direction::In, 0, 2);
    let events = [Event::Frame(early), Event::Open(s.open), Event::End(id, 7)];
    assert_eq!(alone(s, &events, 2, 0), failure(7, PrecheckKind::Session));
    // Rule 1: no open at all.
    let mut s = Served::new();
    let id = s.id();
    let events = [
        Event::Frame(s.seal(7, Direction::In, 0, 2)),
        Event::End(id, 7),
    ];
    assert_eq!(alone(s, &events, 2, 0), failure(7, PrecheckKind::Session));
    // Rule 1: an open and a carry.
    let mut s = Served::new();
    let id = s.id();
    let key = s.open.key_commitment;
    let events = [
        Event::Open(s.open),
        Event::Carry(id, key),
        Event::Frame(s.seal(7, Direction::In, 0, 2)),
        Event::End(id, 7),
    ];
    assert_eq!(alone(s, &events, 2, 0), failure(7, PrecheckKind::Session));
    // Rule 2: two frames of the session at one position, here another request's.
    let mut s = Served::new();
    let id = s.id();
    let a = s.seal(8, Direction::In, 0, 1);
    let b = s.seal(8, Direction::In, 0, 1);
    let events = [
        Event::Open(s.open),
        Event::Frame(a),
        Event::Frame(b),
        Event::Frame(s.seal(7, Direction::In, 0, 2)),
        Event::End(id, 7),
    ];
    assert_eq!(alone(s, &events, 2, 0), failure(7, PrecheckKind::Frames));
    // Rule 3: one counter logged twice.
    let mut s = Served::new();
    let id = s.id();
    let frame = s.seal(7, Direction::In, 0, 2);
    let events = [
        Event::Open(s.open),
        Event::Frame(frame.clone()),
        Event::Frame(frame),
        Event::End(id, 7),
    ];
    assert_eq!(alone(s, &events, 2, 0), failure(7, PrecheckKind::Frames));
    // Rule 3: two positions at one counter, one of them a request the batch leaves out, so only
    // the session's rule sees it.
    let mut s = Served::new();
    let id = s.id();
    let mine = s.seal(7, Direction::In, 0, 2);
    let mut other = s.seal(8, Direction::In, 0, 1);
    other.header.frame_seq = mine.header.frame_seq;
    let events = [
        Event::Open(s.open),
        Event::Frame(mine),
        Event::Frame(other),
        Event::End(id, 7),
    ];
    assert_eq!(alone(s, &events, 2, 0), failure(7, PrecheckKind::Frames));
    // Rules 1 and 3 both broken: rule 1 is named, wherever the repeated counter lies.
    let mut s = Served::new();
    let id = s.id();
    let frame = s.seal(7, Direction::In, 0, 2);
    let events = [
        Event::Frame(frame.clone()),
        Event::Frame(frame),
        Event::Open(s.open),
        Event::End(id, 7),
    ];
    assert_eq!(alone(s, &events, 2, 0), failure(7, PrecheckKind::Session));
    // Rule 4: each period keeps the session's rules, but the carry commits another key than the
    // open the request's prompt was logged under.
    let mut s = Served::new();
    let id = s.id();
    let p0 = [
        Event::Open(s.open),
        Event::Frame(s.seal(7, Direction::In, 0, 2)),
    ];
    let p1 = [
        Event::Carry(id, [0xee; 32]),
        Event::Frame(s.seal(7, Direction::Out, 0, 3)),
        Event::End(id, 7),
    ];
    let r = request(&s, 7, 2, 3);
    let transcripts = [s.transcript];
    assert_eq!(
        precheck(&[r], &transcripts, &log(&p1, 2), Some(&log(&p0, 0))),
        failure(7, PrecheckKind::Session)
    );
    // The control: the carry copies the open's key, and the window passes.
    let mut s = Served::new();
    let id = s.id();
    let key = s.open.key_commitment;
    let p0 = [
        Event::Open(s.open),
        Event::Frame(s.seal(7, Direction::In, 0, 2)),
    ];
    let p1 = [
        Event::Carry(id, key),
        Event::Frame(s.seal(7, Direction::Out, 0, 3)),
        Event::End(id, 7),
    ];
    let r = request(&s, 7, 2, 3);
    assert_eq!(
        precheck(&[r], &[s.transcript], &log(&p1, 2), Some(&log(&p0, 0))),
        Ok(())
    );
}

#[test]
fn the_first_failing_request_of_the_batch_is_named() {
    let (s, previous, current) = two_periods();
    let good = request(&s, 2, 40, 63);
    let bad = request(&s, 2, 41, 63);
    let missing = BatchRequest {
        request_id: 9,
        ..good
    };
    let transcripts = [s.transcript];
    assert_eq!(
        precheck(
            &[good, missing, bad],
            &transcripts,
            &current,
            Some(&previous)
        ),
        failure(9, PrecheckKind::Frames)
    );
    assert_eq!(
        precheck(
            &[good, bad, missing],
            &transcripts,
            &current,
            Some(&previous)
        ),
        failure(2, PrecheckKind::Lengths)
    );
}

#[test]
fn the_transcript_round_trips_and_its_decoder_refuses_what_it_must() {
    let (s, _, _) = two_periods();
    let bytes = encode(&s.transcript);
    let back: SessionTranscript = decode(&bytes, DecodeLimits::MESSAGE).expect("decoded");
    assert_eq!(encode(&back), bytes);
    assert_eq!(back.session_id(), s.id());
    assert_eq!(back.frames().len(), 5);
    assert_eq!(
        back.frames()
            .iter()
            .map(|f| f.frame_seq)
            .collect::<Vec<_>>(),
        [0, 1, 2, 3, 4]
    );
    // Its envelope is secret-class, written with mode 0600, and opens back to it.
    let dir = std::env::temp_dir().join(format!("tessera-precheck-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a directory");
    let path = dir.join("session.tst");
    envelope::write_file(&path, &schema::SESSION_TRANSCRIPT, &s.transcript).expect("written");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&path)
            .expect("metadata")
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600);
    }
    let opened: SessionTranscript = envelope::open(
        &std::fs::read(&path).expect("read"),
        &schema::SESSION_TRANSCRIPT,
        DecodeLimits::MESSAGE,
    )
    .expect("an envelope");
    assert_eq!(encode(&opened), bytes);
    let _ = std::fs::remove_dir_all(&dir);
    // The second frame's counter at the first's: refused at it. The header is 48 bytes and the
    // frame count 8; the first frame is 4 + 8 + 2 + 4 + 8 + 4·5 = 46 bytes.
    let second = 48 + 8 + 46;
    let mut repeated = bytes.clone();
    repeated[second..second + 4].copy_from_slice(&0u32.to_le_bytes());
    let error = decode::<SessionTranscript>(&repeated, DecodeLimits::MESSAGE).expect_err("refused");
    assert_eq!(
        (error.code, error.offset),
        (ErrorCode::CodecNoncanonical, second as u64)
    );
    // A direction tag of 2 in the first frame: refused at it.
    let tag = 48 + 8 + 4 + 8;
    let mut direction = bytes.clone();
    direction[tag] = 2;
    let error =
        decode::<SessionTranscript>(&direction, DecodeLimits::MESSAGE).expect_err("refused");
    assert_eq!(
        (error.code, error.offset),
        (ErrorCode::CodecNoncanonical, tag as u64)
    );
    // Debug shows no key and no token.
    let shown = format!("{:?}", s.transcript);
    assert!(shown.contains("frames: 5"), "{shown}");
}

#[test]
fn a_counter_used_in_both_periods_is_refused_at_the_sessions_first_request() {
    // Requests 1 (in period 1 alone) and 2 (across the boundary) of one session; request 2's
    // outbound frame in period 1 reuses the counter of its inbound frame in period 0. Each period
    // keeps RFC-0062's rules on its own.
    let mut s = Served::new();
    let id = s.id();
    let inbound = s.seal(2, Direction::In, 0, 4);
    let p0 = [Event::Open(s.open), Event::Frame(inbound.clone())];
    let first = s.seal(1, Direction::In, 0, 1);
    let mut reused = s.seal(2, Direction::Out, 0, 4);
    reused.header.frame_seq = inbound.header.frame_seq;
    let p1 = [
        Event::Carry(id, s.open.key_commitment),
        Event::Frame(first),
        Event::End(id, 1),
        Event::Frame(reused),
        Event::End(id, 2),
    ];
    let (previous, current) = (log(&p0, 0), log(&p1, 2));
    let batch = [request(&s, 1, 1, 0), request(&s, 2, 4, 4)];
    assert_eq!(
        precheck(&batch, &[s.transcript], &current, Some(&previous)),
        failure(1, PrecheckKind::Frames)
    );
}

#[test]
fn a_transcript_records_only_its_sessions_frames_in_counter_order() {
    let mut a = Served::new();
    let frame = TokenFrame::new(1, Direction::In, 0, tokens(2, 1));
    let sealed = a.session.seal(&frame).expect("sealed");
    // A plaintext that is not the header's.
    let other = TokenFrame::new(1, Direction::Out, 0, tokens(2, 1));
    assert_eq!(
        a.transcript.record(&sealed, other),
        Err(tessera_anchor::AnchorError::Mismatch)
    );
    let longer = TokenFrame::new(1, Direction::In, 0, tokens(3, 1));
    assert_eq!(
        a.transcript.record(&sealed, longer),
        Err(tessera_anchor::AnchorError::Mismatch)
    );
    // Another session's frame.
    let b = Served::new();
    let mut foreign = sealed.clone();
    foreign.header.session_id = b.id();
    let same = TokenFrame::new(1, Direction::In, 0, tokens(2, 1));
    assert_eq!(
        a.transcript.record(&foreign, same),
        Err(tessera_anchor::AnchorError::Mismatch)
    );
    // The frame itself, then a counter not above it.
    let same = TokenFrame::new(1, Direction::In, 0, tokens(2, 1));
    assert_eq!(a.transcript.record(&sealed, same), Ok(()));
    let again = TokenFrame::new(1, Direction::In, 0, tokens(2, 1));
    assert_eq!(
        a.transcript.record(&sealed, again),
        Err(tessera_anchor::AnchorError::Mismatch)
    );
    assert_eq!(a.transcript.frames().len(), 1);
}
