// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The reference recorder's core (RFC-0015 'Reference recorder'), through the public API:
//! - frames cut from a byte stream wherever it is split;
//! - entries that match an offline recomputation of their digests;
//! - periods whose signed manifests form a valid chain over their logs;
//! - RFC-0062's carries and lapses.

use sha2::{Digest, Sha256};
use tessera_anchor::recorder::{
    FrameParser, MAX_FRAME_LEN, Recorder, WireFrame, decode_log, encode_log,
};
use tessera_anchor::{
    AnchorError, DigestLogEntry, Direction, FrameHeader, HeapKey, RequestEndFrame, Session,
    SessionOpenFrame, TokenFrame, check_chain, merkle, public_key,
};
use tessera_codec::Secret;

fn seed() -> Secret<HeapKey<32>> {
    Secret::new(HeapKey::from_box(Box::new([0x5e; 32])))
}

fn token(request: u64, direction: Direction, index: u32, tokens: Vec<u32>) -> TokenFrame {
    TokenFrame::new(request, direction, index, Secret::new(tokens))
}

/// A session's frames: its open, request 1 (two inbound frames, one outbound) and its end.
fn conversation() -> (Session, Vec<WireFrame>) {
    let (mut session, open, _) = Session::open().unwrap();
    let id = session.session_id();
    let mut frames = vec![WireFrame::Open(open)];
    for f in [
        token(1, Direction::In, 0, vec![1, 2, 3]),
        token(1, Direction::In, 1, vec![]),
        token(1, Direction::Out, 0, vec![u32::MAX; 40]),
    ] {
        frames.push(WireFrame::Sealed(session.seal(&f).unwrap()));
    }
    frames.push(WireFrame::End(RequestEndFrame {
        session_id: id,
        request_id: 1,
    }));
    (session, frames)
}

#[test]
fn frames_come_out_whole_wherever_the_stream_is_split() {
    let (_, frames) = conversation();
    let wire: Vec<u8> = frames.iter().flat_map(WireFrame::to_bytes).collect();
    for split in 0..=wire.len() {
        let mut parser = FrameParser::new();
        let mut out = Vec::new();
        for chunk in [&wire[..split], &wire[split..]] {
            parser.push(chunk);
            while let Some(frame) = parser.next_frame().unwrap() {
                out.push(frame);
            }
        }
        assert_eq!(out, frames, "split at {split}");
        assert!(parser.pending().is_empty());
    }
    // Byte by byte.
    let mut parser = FrameParser::new();
    let mut out = Vec::new();
    for byte in &wire {
        parser.push(&[*byte]);
        while let Some(frame) = parser.next_frame().unwrap() {
            out.push(frame);
        }
    }
    assert_eq!(out, frames);
}

#[test]
fn the_parser_refuses_what_starts_no_frame() {
    let mut parser = FrameParser::new();
    parser.push(b"XXXX and more");
    assert_eq!(parser.next_frame(), Err(AnchorError::Magic));
    assert_eq!(parser.pending(), b"XXXX and more");
    // A sealed frame longer than the bound is refused at its header.
    let mut header = FrameHeader {
        session_id: [1; 16],
        request_id: 2,
        direction: Direction::Out,
        frame_index: 0,
        length: MAX_FRAME_LEN + 2,
        frame_seq: 0,
    };
    // The length must be one a plaintext has, 22 + 4n.
    header.length -= (header.length - 22) % 4;
    assert!(header.length > MAX_FRAME_LEN);
    let mut parser = FrameParser::new();
    parser.push(&header.to_bytes());
    assert_eq!(parser.next_frame(), Err(AnchorError::TooLong));
    // Fewer than four bytes, or a frame cut short, is not an error yet.
    let mut parser = FrameParser::new();
    parser.push(b"TS");
    assert_eq!(parser.next_frame(), Ok(None));
    parser.push(&SessionOpenFrame::MAGIC[2..]);
    assert_eq!(parser.next_frame(), Ok(None));
}

#[test]
fn entries_match_an_offline_recomputation() {
    let (_, frames) = conversation();
    let mut recorder = Recorder::new(seed(), [9; 16], 1_000);
    for (i, frame) in frames.iter().enumerate() {
        let entry = recorder.record(frame, 2_000 + i as u64);
        assert_eq!(entry.seq(), i as u64);
        assert_eq!(entry.t_ns(), 2_000 + i as u64);
        match (frame, entry) {
            (WireFrame::Sealed(sealed), DigestLogEntry::Frame { sha256, .. }) => {
                let mut hash = Sha256::new();
                hash.update(sealed.header.to_bytes());
                hash.update(&sealed.ciphertext);
                assert_eq!(sha256, <[u8; 32]>::from(hash.finalize()));
                assert_eq!(
                    entry,
                    DigestLogEntry::frame(i as u64, 2_000 + i as u64, sealed)
                );
            }
            (WireFrame::Open(open), DigestLogEntry::SessionOpen { key_commitment, .. }) => {
                assert_eq!(key_commitment, open.key_commitment);
            }
            (WireFrame::End(end), DigestLogEntry::RequestEnd { request_id, .. }) => {
                assert_eq!(request_id, end.request_id);
            }
            _ => panic!("frame {i} logged as another kind"),
        }
    }
    // The clock never goes back: an earlier time is logged at the last one.
    let late = recorder.record(&frames[0], 5);
    assert_eq!(late.t_ns(), 2_000 + frames.len() as u64 - 1);
}

/// Three periods: a chain from genesis, each root over its log, `seq` contiguous across them,
/// and RFC-0062's carries.
#[test]
fn periods_form_a_valid_chain_with_carries() {
    let (session_a, frames_a) = conversation();
    let (mut session_b, open_b, _) = Session::open().unwrap();
    let a = session_a.session_id();
    let b = session_b.session_id();
    let mut recorder = Recorder::new(seed(), [9; 16], 1_000);
    let mut t = 1_000;
    let mut tick = || {
        t += 10;
        t
    };
    // Period 0: A's whole conversation; B opens and sends request 2's prompt.
    for frame in &frames_a {
        recorder.record(frame, tick());
    }
    recorder.record(&WireFrame::Open(open_b), tick());
    let prompt = session_b
        .seal(&token(2, Direction::In, 0, vec![7, 8]))
        .unwrap();
    recorder.record(&WireFrame::Sealed(prompt), tick());
    // A session never opened, and one opened twice: neither is carried.
    let stray = [0xcc; 16];
    recorder.record(
        &WireFrame::End(RequestEndFrame {
            session_id: stray,
            request_id: 5,
        }),
        tick(),
    );
    let (_, twice, _) = Session::open().unwrap();
    recorder.record(&WireFrame::Open(twice), tick());
    recorder.record(&WireFrame::Open(twice), tick());
    // A session whose frame precedes its open broke rule 1: not carried either.
    let (mut late, late_open, _) = Session::open().unwrap();
    let early = late.seal(&token(6, Direction::In, 0, vec![1])).unwrap();
    recorder.record(&WireFrame::Sealed(early), tick());
    recorder.record(&WireFrame::Open(late_open), tick());
    let p0 = recorder.rotate(tick()).unwrap();
    // Period 1 opens with A's and B's carries, in session order; B answers and ends request 2,
    // then closes.
    let carried: Vec<[u8; 16]> = recorder
        .entries()
        .iter()
        .map(|e| match e {
            DigestLogEntry::SessionCarry { session_id, .. } => *session_id,
            other => panic!("{other:?} before the carries"),
        })
        .collect();
    let mut expected = vec![a, b];
    expected.sort();
    assert_eq!(carried, expected);
    let answer = session_b
        .seal(&token(2, Direction::Out, 0, vec![9]))
        .unwrap();
    recorder.record(&WireFrame::Sealed(answer), tick());
    recorder.record(
        &WireFrame::End(RequestEndFrame {
            session_id: b,
            request_id: 2,
        }),
        tick(),
    );
    recorder.close(b, tick());
    // A request end alone is no activity (RFC-0062 counts an open or a frame), so A, last active
    // in period 0, still lapses after period 2.
    recorder.record(
        &WireFrame::End(RequestEndFrame {
            session_id: a,
            request_id: 9,
        }),
        tick(),
    );
    let p1 = recorder.rotate(tick()).unwrap();
    // Period 2: A, idle in period 1 but active in period 0, is still carried; B, closed, is not.
    assert!(matches!(
        recorder.entries(),
        [DigestLogEntry::SessionCarry { session_id, .. }] if *session_id == a
    ));
    let p2 = recorder.rotate(tick()).unwrap();
    // Period 3: A has been idle for two periods and lapses.
    assert!(recorder.entries().is_empty());
    let p3 = recorder.rotate(tick()).unwrap();

    let periods = [p0, p1, p2, p3];
    let manifests: Vec<_> = periods.iter().map(|p| p.manifest.clone()).collect();
    assert_eq!(check_chain(&manifests, &[public_key(&seed())]), Ok(()));
    let mut seq = 0;
    for period in &periods {
        let body = &period.manifest.body;
        assert_eq!(body.first_seq, seq);
        assert_eq!(body.entry_count, period.entries.len() as u64);
        let leaves: Vec<_> = period.entries.iter().map(merkle::leaf_hash).collect();
        assert_eq!(body.entries_root, merkle::root(&leaves));
        for entry in &period.entries {
            assert_eq!(entry.seq(), seq);
            // A carry is stamped with its period's start (RFC-0062 'The carry entry').
            if matches!(entry, DigestLogEntry::SessionCarry { .. }) {
                assert_eq!(entry.t_ns(), body.period_start_ns);
            }
            assert!(entry.t_ns() >= body.period_start_ns && entry.t_ns() < body.period_end_ns);
            seq += 1;
        }
        assert_eq!(
            decode_log(&encode_log(&period.entries)),
            Ok(period.entries.clone())
        );
    }
}

#[test]
fn a_period_has_a_length() {
    let mut recorder = Recorder::new(seed(), [9; 16], 1_000);
    assert!(recorder.rotate(1_000).is_err());
    let (_, frames) = conversation();
    recorder.record(&frames[0], 5_000);
    assert!(recorder.rotate(5_000).is_err());
    assert!(recorder.rotate(5_001).is_ok());
}
