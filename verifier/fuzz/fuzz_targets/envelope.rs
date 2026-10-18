// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `Envelope::open`, then the payload decoders, on arbitrary bytes.
//!
//! The first byte configures the reader: bits 0–2 the schema it expects, among the five
//! `READERS` modulo five, bits 3–4 its limits, and bit 5 whether to reseal the input, that is,
//! to make its declared length match and its digest verify so that the payload decoder is
//! reached. The second byte selects the payload's type in `tv_fuzz::TYPES`; the rest is the
//! envelope.
//!
//! A rejection must be a fault of the input and point inside it; a payload error, inside the
//! payload. An opened envelope must be, byte for byte, the envelope a writer of the version found
//! emits for its payload: magic, envelope version 1, the expected schema, a version the reader
//! accepts (`v` or `v − 1`, never 0), `reserved` 0, the payload's length, and the digest of all
//! that, recomputed here. A decoded payload must be the canonical encoding of its value, within
//! the limits.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tv_codec::{DecodeLimits, Envelope, Expected};
use tv_fuzz::decode_as;
use tv_testkit::blake3;

const CONTEXT: &str = "tessera/v1/envelope";

/// The readers: the vectors' probe schema at version 3, and four schemas of the catalog (03 §5),
/// the last the weight commitment's public artifact (RFC-0040).
const READERS: [Expected; 5] = [
    Expected {
        schema_id: 0xFFF0,
        version: 3,
    },
    Expected {
        schema_id: 0x0101,
        version: 1,
    },
    Expected {
        schema_id: 0x0201,
        version: 1,
    },
    Expected {
        schema_id: 0x0601,
        version: 2,
    },
    Expected {
        schema_id: 0x0301,
        version: 1,
    },
];

/// The vectors' limits, the RFC-0003 defaults, and limits small enough to meet every check.
const LIMITS: [DecodeLimits; 4] = [
    DecodeLimits {
        max_total_bytes: 256,
        max_vec_len: 256,
        max_depth: 8,
    },
    DecodeLimits::MESSAGE,
    DecodeLimits::INTEGER_MODEL,
    DecodeLimits {
        max_total_bytes: 96,
        max_vec_len: 4,
        max_depth: 3,
    },
];

/// Sets the declared payload length to what the input holds and recomputes the digest.
fn reseal(bytes: &mut [u8]) {
    let Some(signed) = bytes.len().checked_sub(32) else {
        return;
    };
    let Some(payload_len) = bytes.len().checked_sub(52) else {
        return;
    };
    bytes[12..20].copy_from_slice(&(payload_len as u64).to_le_bytes());
    let digest = blake3::derive_key(CONTEXT, &bytes[..signed]);
    bytes[signed..].copy_from_slice(&digest);
}

/// The envelope a writer of `schema_id` at `version` emits for `payload` (RFC-0003 'Envelope').
fn sealed(schema_id: u16, version: u16, payload: &[u8]) -> Vec<u8> {
    let mut bytes = [
        &b"TSRA"[..],
        &1u16.to_le_bytes(),
        &schema_id.to_le_bytes(),
        &version.to_le_bytes(),
        &0u16.to_le_bytes(),
        &(payload.len() as u64).to_le_bytes(),
        payload,
    ]
    .concat();
    bytes.extend(blake3::derive_key(CONTEXT, &bytes));
    bytes
}

fuzz_target!(|data: &[u8]| {
    let [config, selector, envelope @ ..] = data else {
        return;
    };
    let reader = READERS[usize::from(config & 0b111) % READERS.len()];
    let limits = LIMITS[usize::from((config >> 3) & 0b11)];
    let mut bytes = envelope.to_vec();
    if config & 0b10_0000 != 0 {
        reseal(&mut bytes);
    }
    let opened = Envelope::open(&bytes, reader, &limits, |signed| {
        blake3::derive_key(CONTEXT, signed)
    });
    let envelope = match opened {
        Ok(envelope) => envelope,
        Err(error) => {
            assert!(
                error.code().is_input_fault() && error.offset() <= bytes.len() as u64,
                "{error} on a fuzz envelope"
            );
            return;
        }
    };
    let found = envelope.header().schema_version;
    assert!(
        found != 0 && (found == reader.version || found.checked_add(1) == Some(reader.version)),
        "version {found} opened by a reader of version {}",
        reader.version
    );
    assert_eq!(
        sealed(reader.schema_id, found, envelope.payload()),
        bytes,
        "an opened envelope is not the one a writer emits"
    );
    assert!(bytes.len() as u64 <= limits.max_total_bytes);
    match envelope.decode_payload(|c, _version| decode_as(usize::from(*selector), c)) {
        Ok(echo) => {
            assert_eq!(echo.bytes, envelope.payload(), "a payload is not canonical");
            assert!(echo.depth <= limits.max_depth.min(256));
            assert!(echo.longest <= limits.max_vec_len);
        }
        Err(error) => assert!(
            error.code().is_input_fault()
                && (20..=bytes.len() as u64 - 32).contains(&error.offset()),
            "{error} outside the payload"
        ),
    }
});
