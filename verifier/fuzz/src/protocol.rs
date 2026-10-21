// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Protocol packet decoding and canonical re-encoding, without judging proof validity.

use tv_codec::{DecodeError, DecodeLimits, decode};
use tv_core::protocol::*;

/// Selector order, shared with the seed generator and the deterministic replay test.
pub const TYPES: [&str; 8] = [
    "SeedCommit",
    "ProverSeedShare",
    "SeedReveal",
    "Round1",
    "Round2Response",
    "ColumnQueries",
    "Round3Response",
    "NiProof",
];

const LIMITS: [DecodeLimits; 4] = [
    DecodeLimits::MESSAGE,
    DecodeLimits {
        max_total_bytes: 4096,
        max_vec_len: 16,
        max_depth: 8,
    },
    DecodeLimits {
        max_total_bytes: 256,
        max_vec_len: 4,
        max_depth: 3,
    },
    DecodeLimits {
        max_total_bytes: 96,
        max_vec_len: 0,
        max_depth: 0,
    },
];

/// Two control bytes: packet selector; low two bits select limits, bit 2 selects an envelope.
/// Accepted values must encode to the input payload from their actual decoded fields.
/// Returns whether decoding succeeded, so replay tests also detect misrouted valid seeds.
pub fn check(data: &[u8]) -> bool {
    let [selector, config, bytes @ ..] = data else {
        return false;
    };
    let limits = &LIMITS[usize::from(config & 3)];
    let enveloped = config & 4 != 0;
    macro_rules! packet {
        ($ty:ty) => {{
            let value = if enveloped {
                <$ty>::open(bytes, limits)
            } else {
                decode(bytes, limits, <$ty>::decode)
            };
            value.and_then(|v| v.to_tce().map_err(|code| DecodeError::new(code, 0)))
        }};
    }
    let result = match usize::from(*selector) % TYPES.len() {
        0 => packet!(SeedCommit),
        1 => packet!(ProverSeedShare),
        2 => packet!(SeedReveal),
        3 => packet!(Round1),
        4 => packet!(Round2Response),
        5 => packet!(ColumnQueries),
        6 => packet!(Round3Response),
        _ => packet!(NiProof),
    };
    match result {
        Ok(encoded) => {
            let payload = if enveloped {
                &bytes[20..bytes.len() - 32]
            } else {
                bytes
            };
            assert_eq!(encoded, payload, "a protocol value has another encoding");
            assert!(bytes.len() as u64 <= limits.max_total_bytes);
            true
        }
        Err(e) => {
            assert!(
                e.code().is_input_fault() && e.offset() <= bytes.len() as u64,
                "{e} on a protocol packet"
            );
            false
        }
    }
}
