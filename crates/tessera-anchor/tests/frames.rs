// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0015 'Frames' with RFC-0059's header and counter, through the public API: the session
//! counter, fresh secrets per session, and each decoder's refusals. The conformance vectors and
//! decryption are checked inside the crate (`src/vectors.rs`, `src/session.rs`), where a test
//! may fix a session's secrets and decrypt.

use tessera_anchor::{
    AnchorError, DigestLogEntry, Direction, FrameHeader, SealedFrame, Session, SessionOpenFrame,
    TokenFrame,
};
use tessera_codec::error::ErrorCode;
use tessera_codec::{DecodeLimits, Secret, decode, encode};

fn session() -> Session {
    Session::open().unwrap().0
}

fn frame(direction: Direction, index: u32, tokens: Vec<u32>) -> TokenFrame {
    TokenFrame::new(77, direction, index, Secret::new(tokens))
}

/// INV-ANC-3: the session numbers its frames 0, 1, 2, … in sealing order, across both
/// directions, and no operation takes a counter from its caller.
#[test]
fn the_session_counts_its_frames_across_both_directions() {
    let mut s = session();
    let seqs: Vec<u32> = [
        frame(Direction::In, 0, vec![1]),
        frame(Direction::Out, 0, vec![2]),
        frame(Direction::In, 1, vec![3]),
        frame(Direction::Out, 1, vec![]),
    ]
    .iter()
    .map(|f| s.seal(f).unwrap().header.frame_seq)
    .collect();
    assert_eq!(seqs, [0, 1, 2, 3]);
    assert_eq!(s.next_seq(), 4);
    // The same frame sealed twice takes two counters, so two keystreams.
    let f = frame(Direction::Out, 2, vec![9; 8]);
    let (a, b) = (s.seal(&f).unwrap(), s.seal(&f).unwrap());
    assert_ne!(a.ciphertext, b.ciphertext);
}

/// Every session draws its own identifier and secrets (RFC-0059 'Session identity'): two sessions
/// share neither, and their commitments differ.
#[test]
fn every_session_draws_its_own_identity_and_keys() {
    let (mut a, open_a, _) = Session::open().unwrap();
    let (mut b, open_b, _) = Session::open().unwrap();
    assert_ne!(open_a.session_id, open_b.session_id);
    assert_eq!(a.session_id(), open_a.session_id);
    assert_ne!(open_a.key_commitment, open_b.key_commitment);
    let f = frame(Direction::In, 0, vec![1, 2, 3]);
    assert_ne!(
        a.seal(&f).unwrap().ciphertext,
        b.seal(&f).unwrap().ciphertext
    );
}

#[test]
fn malformed_headers_and_frames_are_refused() {
    let mut s = session();
    let sealed = s.seal(&frame(Direction::Out, 0, vec![1, 2])).unwrap();
    let bytes = sealed.header.to_bytes();
    assert_eq!(FrameHeader::from_bytes(&bytes), Ok(sealed.header));
    assert_eq!(
        FrameHeader::from_bytes(&bytes[..40]),
        Err(AnchorError::Truncated)
    );
    let mut long = bytes.to_vec();
    long.push(0);
    assert_eq!(
        FrameHeader::from_bytes(&long),
        Err(AnchorError::Noncanonical)
    );
    let mut magic = bytes;
    magic[0] = b'X';
    assert_eq!(FrameHeader::from_bytes(&magic), Err(AnchorError::Magic));
    for byte in [2u8, 0xFF] {
        let mut wrong = bytes;
        wrong[28] = byte;
        assert_eq!(FrameHeader::from_bytes(&wrong), Err(AnchorError::Direction));
    }
    // Lengths no plaintext has: below 22, whatever their residue, or not 22 + 4n.
    for length in [0u32, 2, 6, 10, 14, 18, 21, 23, 24, 25, 31, u32::MAX] {
        let mut wrong = bytes;
        wrong[33..37].copy_from_slice(&length.to_le_bytes());
        assert_eq!(
            FrameHeader::from_bytes(&wrong),
            Err(AnchorError::Noncanonical),
            "{length}"
        );
    }
    let mut empty = bytes;
    empty[33..37].copy_from_slice(&22u32.to_le_bytes());
    assert!(FrameHeader::from_bytes(&empty).is_ok());
    // A frame on the wire whose ciphertext is not its header's length.
    let mut wire = sealed.to_bytes();
    wire.pop();
    assert_eq!(SealedFrame::from_bytes(&wire), Err(AnchorError::Truncated));
    wire.extend_from_slice(&[0, 0]);
    assert_eq!(
        SealedFrame::from_bytes(&wire),
        Err(AnchorError::Noncanonical)
    );
    assert_eq!(
        SealedFrame::from_bytes(&wire[..20]),
        Err(AnchorError::Truncated)
    );
    assert_eq!(
        SessionOpenFrame::from_bytes(&[0u8; 52]),
        Err(AnchorError::Magic)
    );
    assert_eq!(
        SessionOpenFrame::from_bytes(&[0u8; 51]),
        Err(AnchorError::Truncated)
    );
    assert_eq!(
        SessionOpenFrame::from_bytes(&[0u8; 53]),
        Err(AnchorError::Noncanonical)
    );
    assert_eq!(AnchorError::Direction.code(), ErrorCode::CodecNoncanonical);
    assert_eq!(AnchorError::Truncated.code(), ErrorCode::CodecTruncated);
    assert_eq!(AnchorError::TooLong.code(), ErrorCode::CodecLimit);
    assert_eq!(
        AnchorError::Noncanonical.code(),
        ErrorCode::CodecNoncanonical
    );
    assert_eq!(AnchorError::Magic.code(), ErrorCode::CodecMagic);
    assert_eq!(AnchorError::Mismatch.code(), ErrorCode::WitnessAnchor);
    assert_eq!(AnchorError::Exhausted.code(), ErrorCode::CodecLimit);
    assert_eq!(AnchorError::Random.code(), ErrorCode::ResIo);
}

#[test]
fn log_entries_refuse_other_tags_and_directions() {
    let mut s = session();
    let sealed = s.seal(&frame(Direction::Out, 0, vec![5])).unwrap();
    let entry = DigestLogEntry::frame(3, 4, &sealed);
    let mut bytes = encode(&entry);
    assert_eq!(bytes.len(), 2 + 8 + 8 + 16 + 8 + 2 + 4 + 4 + 4 + 32);
    assert_eq!(
        decode::<DigestLogEntry>(&bytes, DecodeLimits::MESSAGE),
        Ok(entry)
    );
    bytes[0] = 3;
    assert_eq!(
        decode::<DigestLogEntry>(&bytes, DecodeLimits::MESSAGE).map_err(|e| e.code),
        Err(ErrorCode::CodecNoncanonical)
    );
    let mut bytes = encode(&entry);
    bytes[42] = 2; // the direction's tag, after tag, seq, t_ns, session and request
    assert_eq!(
        decode::<DigestLogEntry>(&bytes, DecodeLimits::MESSAGE).map_err(|e| e.code),
        Err(ErrorCode::CodecNoncanonical)
    );
}

#[test]
fn debug_shows_no_secret() {
    let (session, _, keys) = Session::open().unwrap();
    assert_eq!(format!("{keys:?}"), "SessionKeys(..)");
    assert!(!format!("{session:?}").contains("cipher"));
    let f = frame(Direction::In, 0, vec![123_456, 7]);
    let shown = format!("{f:?}");
    assert!(!shown.contains("123456"), "{shown}");
    assert!(shown.contains("tokens: 2"), "{shown}");
}
