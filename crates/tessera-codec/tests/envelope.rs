// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The artifact envelope (RFC-0003 'Envelope'). The in-memory reader and the streaming reader
//! are held to `spec/vectors/codec/envelope.json`, case by case, and to each other: same code,
//! same offset, whatever the input. Writing streams to exactly the bytes sealing produces.

use std::cell::Cell;
use std::fs;
use std::io::{self, Cursor, Read, Write};
use std::path::PathBuf;

use proptest::prelude::*;
use serde_json::Value;
use tessera_codec::envelope::{
    self, DIGEST_LEN, HEADER_LEN, Header, ReadError, open, open_payload, seal, verify,
};
use tessera_codec::error::{ErrorCode, SchemaId, TesseraError};
use tessera_codec::golden::{PROBE_SCHEMA, decode_as};
use tessera_codec::schema::{self, Class, Schema};
use tessera_codec::{Decode, DecodeError, DecodeLimits, Encode, Sink};

fn bytes(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex"))
        .collect()
}

fn field(case: &Value, key: &str) -> u64 {
    case[key]
        .as_u64()
        .unwrap_or_else(|| panic!("{key} in {case}"))
}

/// What both readers report for `input`: the header and payload, or the error.
fn read_both<'a>(
    input: &'a [u8],
    schema: &Schema,
    limits: DecodeLimits,
) -> Result<(Header, &'a [u8]), DecodeError> {
    let in_memory = open_payload(input, schema, limits);
    let streamed = verify(Cursor::new(input), schema, limits).map_err(|e| match e {
        ReadError::Invalid(e) => e,
        ReadError::Io(e) => panic!("a cursor does not fail: {e}"),
    });
    assert_eq!(
        in_memory.map(|(header, _)| header),
        streamed,
        "the readers disagree on {input:02x?}"
    );
    in_memory
}

fn limits(value: &Value) -> DecodeLimits {
    DecodeLimits {
        max_total_bytes: field(value, "max_total_bytes"),
        max_vec_len: field(value, "max_vec_len"),
        max_depth: u16::try_from(field(value, "max_depth")).expect("a u16 depth"),
    }
}

#[test]
fn envelope_vectors_hold() {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/codec/envelope.json");
    let file: Value =
        serde_json::from_str(&fs::read_to_string(path).expect("vector file")).expect("JSON");
    // The RFC-0003 defaults by name: this crate's constants must be the values the vectors
    // state, which are the RFC's.
    let named = |name: &str| match name {
        "message" => DecodeLimits::MESSAGE,
        "integer-model" => DecodeLimits::INTEGER_MODEL,
        other => panic!("unknown limits {other}"),
    };
    for (name, value) in file["named_limits"].as_object().expect("named limits") {
        assert_eq!(named(name), limits(value), "{name}");
    }
    let cases = file["cases"].as_array().expect("cases");
    assert!(cases.len() >= 50, "only {} cases", cases.len());
    for case in cases {
        let limits = case["limits"]
            .as_str()
            .map_or_else(|| limits(&file["limits"]), named);
        let why = case["why"].as_str().expect("why");
        let expect = Schema {
            name: "expected",
            id: SchemaId(u16::try_from(field(case, "schema")).expect("a u16 schema id")),
            version: u16::try_from(field(case, "version")).expect("a u16 version"),
            extension: None,
            class: Class::Public,
        };
        let input = bytes(case["hex"].as_str().expect("hex"));
        let ty = case["type"].as_str().expect("type");
        let outcome =
            read_both(&input, &expect, limits).and_then(|(_, payload)| {
                match decode_as(ty, payload, limits).expect("a payload type this crate defines") {
                    Ok(_) => Ok(payload.to_vec()),
                    Err(e) => Err(DecodeError {
                        code: e.code,
                        offset: HEADER_LEN as u64 + e.offset,
                    }),
                }
            });
        match case["code"].as_str() {
            None => {
                let payload = bytes(case["payload"].as_str().expect("payload"));
                assert_eq!(outcome, Ok(payload), "{why}");
            }
            Some(code) => {
                let expected = DecodeError {
                    code: ErrorCode::from_identifier(code).expect("a known code"),
                    offset: field(case, "offset"),
                };
                assert_eq!(outcome, Err(expected), "{why}");
            }
        }
    }
}

/// A writer that accepts `room` bytes, then fails.
struct Full {
    room: usize,
}

impl Write for Full {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.room == 0 {
            return Err(io::Error::new(io::ErrorKind::StorageFull, "full"));
        }
        let n = buf.len().min(self.room);
        self.room -= n;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// A reader that is interrupted before every read, and fails once `fail_at` bytes are read.
struct Flaky<'a> {
    input: &'a [u8],
    fail_at: usize,
    interrupt: bool,
}

impl Read for Flaky<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.interrupt = !self.interrupt;
        if self.interrupt {
            return Err(io::ErrorKind::Interrupted.into());
        }
        if self.fail_at == 0 {
            return Err(io::Error::new(io::ErrorKind::ConnectionReset, "reset"));
        }
        let n = buf.len().min(self.input.len()).min(self.fail_at).min(7);
        buf[..n].copy_from_slice(&self.input[..n]);
        self.input = &self.input[n..];
        self.fail_at -= n;
        Ok(n)
    }
}

#[test]
fn streams_fail_cleanly() {
    let value: Vec<u32> = (0..10_000).collect();
    let sealed = seal(&schema::TOKEN_FILE, &value);
    // A writer that fills up, at every point up to the digest's last byte.
    for room in [0, 1, HEADER_LEN, sealed.len() / 2, sealed.len() - 1] {
        let error = envelope::write(&mut Full { room }, &schema::TOKEN_FILE, &value)
            .expect_err("the writer is full");
        assert_eq!(error.kind(), io::ErrorKind::StorageFull, "room {room}");
    }
    // Interrupted reads are retried, so a reader interrupted before every read still verifies.
    let interrupted = Flaky {
        input: &sealed,
        fail_at: usize::MAX,
        interrupt: false,
    };
    let limits = DecodeLimits::MESSAGE;
    assert!(verify(interrupted, &schema::TOKEN_FILE, limits).is_ok());
    // Any other read error is an I/O failure, `ERR-RES-IO`, not a verdict on the envelope.
    for fail_at in [0, 5, HEADER_LEN + 3, sealed.len() - 1] {
        let reset = Flaky {
            input: &sealed,
            fail_at,
            interrupt: false,
        };
        match verify(reset, &schema::TOKEN_FILE, limits) {
            Err(error @ ReadError::Io(_)) => {
                assert_eq!(TesseraError::from(error).code, ErrorCode::ResIo);
            }
            other => panic!("failing at {fail_at}: {other:?}"),
        }
    }
}

/// An `Encode` that writes one byte more each time it runs: broken, since its two passes in
/// `write` disagree.
struct Growing(Cell<usize>);

impl Encode for Growing {
    fn encode_to<S: Sink + ?Sized>(&self, out: &mut S) {
        let n = self.0.get();
        self.0.set(n + 1);
        out.put(&vec![0; n]);
    }
}

/// A reader that ends after `pause_at` bytes, then has more to give.
struct Resuming<'a> {
    input: &'a [u8],
    pause_at: usize,
}

impl Read for Resuming<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.pause_at == 0 {
            self.pause_at = usize::MAX;
            return Ok(0);
        }
        let n = buf.len().min(self.input.len()).min(self.pause_at);
        buf[..n].copy_from_slice(&self.input[..n]);
        self.input = &self.input[n..];
        self.pause_at -= n;
        Ok(n)
    }
}

/// A reader that claims to have read one byte more than it was given room for.
struct Boastful;

impl Read for Boastful {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        Ok(buf.len() + 1)
    }
}

#[test]
fn misbehaving_encoders_and_readers_are_errors() {
    let error = envelope::write(&mut Vec::new(), &schema::CLAIM_LIST, &Growing(Cell::new(0)))
        .expect_err("the passes disagree");
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    // The first end of the input is the end, even if the reader has more to give.
    let sealed = seal(&schema::CLAIM_LIST, &vec![1u64, 2, 3]);
    let resuming = Resuming {
        input: &sealed,
        pause_at: HEADER_LEN + 5,
    };
    match verify(resuming, &schema::CLAIM_LIST, DecodeLimits::MESSAGE) {
        Err(ReadError::Invalid(e)) => assert_eq!(
            e,
            DecodeError {
                code: ErrorCode::CodecTruncated,
                offset: 12,
            }
        ),
        other => panic!("{other:?}"),
    }
    match verify(Boastful, &schema::CLAIM_LIST, DecodeLimits::MESSAGE) {
        Err(ReadError::Io(e)) => assert_eq!(e.kind(), io::ErrorKind::InvalidData),
        other => panic!("{other:?}"),
    }
}

#[test]
fn open_reads_only_the_current_version() {
    let previous = Schema {
        version: PROBE_SCHEMA.version - 1,
        ..PROBE_SCHEMA
    };
    let sealed = seal(&previous, &vec![1u64, 2]);
    // The reader of the current version accepts the previous one; the typed `open` cannot,
    // since its type is one version's layout.
    let (header, _) = open_payload(&sealed, &PROBE_SCHEMA, DecodeLimits::MESSAGE)
        .expect("a reader accepts v - 1");
    assert_eq!(header.version, previous.version);
    assert_eq!(
        open::<Vec<u64>>(&sealed, &PROBE_SCHEMA, DecodeLimits::MESSAGE),
        Err(DecodeError {
            code: ErrorCode::CodecVersion,
            offset: 8,
        })
    );
}

/// The offset a reader reports for the first `len` bytes of an envelope: the header field cut
/// short, else the payload length, which promised more.
fn truncated_at(len: usize) -> u64 {
    match len {
        0..4 => 0,
        4..6 => 4,
        6..8 => 6,
        8..10 => 8,
        10..12 => 10,
        _ => 12,
    }
}

/// Sealing and streaming write the same bytes; both readers open them to `value`; every proper
/// prefix is `ERR-CODEC-TRUNCATED` at the field or length it cuts; and flipping a bit of the
/// payload or the digest, at `flip`, is `ERR-CODEC-DIGEST`.
fn round_trip<T>(value: &T, flip: prop::sample::Index) -> Result<(), TestCaseError>
where
    T: Encode + Decode + PartialEq + std::fmt::Debug,
{
    let limits = DecodeLimits::MESSAGE;
    let schema = schema::CLAIM_LIST;
    let sealed = seal(&schema, value);
    let (header, payload) = read_both(&sealed, &schema, limits).expect("a sealed envelope");
    prop_assert_eq!(header.payload_len as usize, payload.len());
    let opened = open::<T>(&sealed, &schema, limits);
    prop_assert_eq!(opened.as_ref(), Ok(value));
    let mut streamed = Vec::new();
    let digest = envelope::write(&mut streamed, &schema, value).expect("a vector does not fail");
    prop_assert_eq!(&streamed, &sealed);
    prop_assert_eq!(&digest[..], &sealed[sealed.len() - DIGEST_LEN..]);
    let step = 1 + sealed.len() / 64;
    for len in (0..sealed.len()).step_by(step).chain([sealed.len() - 1]) {
        let error = read_both(&sealed[..len], &schema, limits).expect_err("a proper prefix");
        prop_assert_eq!(error.code, ErrorCode::CodecTruncated);
        prop_assert_eq!(error.offset, truncated_at(len));
    }
    let at = HEADER_LEN + flip.index(sealed.len() - HEADER_LEN);
    let mut tampered = sealed;
    tampered[at] ^= 1 << (at % 8);
    let error = read_both(&tampered, &schema, limits).expect_err("a flipped bit");
    prop_assert_eq!(error.code, ErrorCode::CodecDigest);
    prop_assert_eq!(error.offset, HEADER_LEN as u64 + header.payload_len);
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Small values, written through the buffer of the streaming writer.
    #[test]
    fn small_envelopes_round_trip(
        value in prop::collection::vec(prop::collection::vec(any::<u16>(), 0..40), 0..12),
        flip in any::<prop::sample::Index>(),
    ) {
        round_trip(&value, flip)?;
    }

    /// Payloads larger than the streaming writer's 64 KiB buffer, written in one piece.
    #[test]
    fn large_envelopes_round_trip(
        seed in prop::collection::vec(any::<u8>(), 1..4),
        repeat in 66_000usize..100_000,
        flip in any::<prop::sample::Index>(),
    ) {
        round_trip(&seed.repeat(repeat), flip)?;
    }

    /// Payloads written four bytes at a time, crossing the buffer's 64 KiB boundary.
    #[test]
    fn piecewise_envelopes_round_trip(
        value in prop::collection::vec(any::<u32>(), 16_000..40_000),
        flip in any::<prop::sample::Index>(),
    ) {
        round_trip(&value, flip)?;
    }
}
