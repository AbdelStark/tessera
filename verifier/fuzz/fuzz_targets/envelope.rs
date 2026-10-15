// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `Envelope::open`, then the payload decoders, on arbitrary bytes.
//!
//! The first byte configures the reader: bits 0–1 the schema it expects, bits 2–3 its limits,
//! and bit 4 whether to reseal the input, that is, to make its declared length match and its
//! digest verify so that the payload decoder is reached. The second byte selects the payload's
//! type in `tv_fuzz::TYPES`; the rest is the envelope.
//!
//! Any input must be accepted or rejected without a panic. An opened envelope's payload has its
//! declared length; a decoded payload is the canonical encoding of its value (INV-TCE-2); and an
//! error points inside the input, a payload error past the header.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tv_codec::{DecodeLimits, Envelope, Expected};
use tv_fuzz::decode_as;
use tv_testkit::blake3;

const CONTEXT: &str = "tessera/v1/envelope";

/// The readers: the vectors' probe schema at version 3, and three schemas of the catalog.
const READERS: [Expected; 4] = [
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

fuzz_target!(|data: &[u8]| {
    let [config, selector, envelope @ ..] = data else {
        return;
    };
    let reader = READERS[usize::from(config & 0b11)];
    let limits = LIMITS[usize::from((config >> 2) & 0b11)];
    let mut bytes = envelope.to_vec();
    if config & 0b1_0000 != 0 {
        reseal(&mut bytes);
    }
    let opened = Envelope::open(&bytes, reader, &limits, |signed| {
        blake3::derive_key(CONTEXT, signed)
    });
    let envelope = match opened {
        Ok(envelope) => envelope,
        Err(error) => {
            assert!(
                error.offset() <= bytes.len() as u64,
                "{error} past the input"
            );
            return;
        }
    };
    assert_eq!(
        envelope.payload().len() as u64,
        envelope.header().payload_len
    );
    match envelope.decode_payload(|c| decode_as(usize::from(*selector), c)) {
        Ok(canonical) => assert_eq!(canonical, envelope.payload(), "a payload is not canonical"),
        Err(error) => assert!(
            (20..=bytes.len() as u64 - 32).contains(&error.offset()),
            "{error} outside the payload"
        ),
    }
});
