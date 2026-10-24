// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The reference cipher, the encoders and the decoders against `spec/vectors/anchor/`, written by
//! an independent standard-library Python oracle (RFC-0015 'Frames', RFC-0059). The sessions
//! are built from the oracle's drawn bytes through a test-only constructor; a release build has
//! no way to choose a session's secrets.

use crate::{
    DigestLogEntry, Direction, FrameHeader, SealedFrame, Session, SessionOpenFrame, TokenFrame,
    key_commitment, sampler_commitment,
};
use serde_json::Value;
use tessera_codec::{DecodeLimits, Secret, decode, encode};

fn hex(s: &Value) -> Vec<u8> {
    let s = s.as_str().unwrap();
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

fn array<const N: usize>(v: &Value) -> [u8; N] {
    hex(v).try_into().unwrap()
}

fn number(v: &Value) -> u64 {
    v.as_u64().unwrap()
}

#[test]
#[allow(clippy::disallowed_methods)]
fn every_session_matches_the_independent_oracle() {
    let document: Value =
        serde_json::from_str(include_str!("../../../spec/vectors/anchor/frames.json")).unwrap();
    let mut frames = 0;
    for s in document["sessions"].as_array().unwrap() {
        let sampler = Secret::new(array::<16>(&s["sampler_key"]));
        let (mut session, open, keys) = Session::from_secrets(array(&s["drawn"]), &sampler);
        session.set_next(number(&s["first_seq"]));
        assert_eq!(session.session_id(), array(&s["session_id"]));
        // The commitments, from the oracle's own keys and from the session's.
        let key = Secret::new(array::<16>(&s["key"]));
        let key_salt = Secret::new(array::<16>(&s["key_salt"]));
        let sampler_salt = Secret::new(array::<16>(&s["sampler_salt"]));
        assert_eq!(key_commitment(&key, &key_salt), array(&s["key_commitment"]));
        assert_eq!(
            key_commitment(&keys.key, &keys.key_salt),
            array(&s["key_commitment"])
        );
        assert_eq!(
            sampler_commitment(&sampler, &sampler_salt),
            array(&s["sampler_commitment"])
        );
        assert_eq!(
            sampler_commitment(&sampler, &keys.sampler_salt),
            array(&s["sampler_commitment"])
        );
        assert_eq!(open.to_bytes().to_vec(), hex(&s["session_open"]));
        assert_eq!(SessionOpenFrame::from_bytes(&open.to_bytes()), Ok(open));
        let entry = &s["open_entry"];
        let logged =
            DigestLogEntry::session_open(number(&entry["seq"]), number(&entry["t_ns"]), &open);
        assert_eq!(encode(&logged), hex(&entry["tce"]));
        assert_eq!(
            decode::<DigestLogEntry>(&hex(&entry["tce"]), DecodeLimits::MESSAGE),
            Ok(logged)
        );
        for f in s["frames"].as_array().unwrap() {
            let tokens: Vec<u32> = f["tokens"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| number(t) as u32)
                .collect();
            let direction = match number(&f["direction"]) {
                0 => Direction::In,
                _ => Direction::Out,
            };
            let frame = TokenFrame::new(
                number(&f["request_id"]),
                direction,
                number(&f["frame_index"]) as u32,
                Secret::new(tokens.clone()),
            );
            let sealed = session.seal(&frame).unwrap();
            assert_eq!(u64::from(sealed.header.frame_seq), number(&f["frame_seq"]));
            assert_eq!(sealed.header.to_bytes().to_vec(), hex(&f["header"]));
            assert_eq!(
                FrameHeader::from_bytes(&hex(&f["header"])),
                Ok(sealed.header)
            );
            assert_eq!(
                crate::session::plaintext(&frame).to_vec(),
                hex(&f["plaintext"])
            );
            assert_eq!(sealed.ciphertext, hex(&f["ciphertext"]));
            assert_eq!(sealed.digest(), array(&f["sha256"]));
            assert_eq!(
                SealedFrame::from_bytes(&sealed.to_bytes()).as_ref(),
                Ok(&sealed)
            );
            let opened = session.open_frame(&sealed).unwrap();
            assert_eq!(opened.tokens().expose_secret(), &tokens);
            let entry = &f["entry"];
            let logged =
                DigestLogEntry::frame(number(&entry["seq"]), number(&entry["t_ns"]), &sealed);
            let bytes = hex(&entry["tce"]);
            assert_eq!(encode(&logged), bytes);
            assert_eq!(
                decode::<DigestLogEntry>(&bytes, DecodeLimits::MESSAGE),
                Ok(logged)
            );
            frames += 1;
        }
        let entry = &s["close_entry"];
        let closed = DigestLogEntry::SessionClose {
            seq: number(&entry["seq"]),
            t_ns: number(&entry["t_ns"]),
            session_id: session.session_id(),
        };
        assert_eq!(encode(&closed), hex(&entry["tce"]));
        assert_eq!(
            decode::<DigestLogEntry>(&hex(&entry["tce"]), DecodeLimits::MESSAGE),
            Ok(closed)
        );
    }
    assert_eq!(frames, 15);
}
