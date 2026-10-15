// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Every decoder of `tv_fuzz::TYPES` on arbitrary bytes. The first byte selects the type, the
//! second the limits; the rest is decoded as a whole input.
//!
//! A rejection must be a fault of the input (never `ERR-RES-MEMORY` or `ERR-INTERNAL`) and point
//! inside it. An accepted input must be the canonical encoding of its value (INV-TCE-2), the value
//! must be within the limits, and each proper prefix of the input must be cut short. The value's
//! encoding, depth and lengths come from `tv_fuzz`'s echo of the value, not from the decoder's
//! checks. Allocations are bounded by libFuzzer's malloc limit.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tv_codec::{DecodeLimits, ErrorCode, decode};
use tv_fuzz::decode_as;

/// The limits, by the second byte: small enough to reach every limit with short inputs, tighter
/// still, RFC-0003's defaults for messages, and a depth limit above the 256 ceiling.
const LIMITS: [DecodeLimits; 4] = [
    DecodeLimits {
        max_total_bytes: 512,
        max_vec_len: 32,
        max_depth: 6,
    },
    DecodeLimits {
        max_total_bytes: 96,
        max_vec_len: 4,
        max_depth: 2,
    },
    DecodeLimits::MESSAGE,
    DecodeLimits {
        max_total_bytes: 4096,
        max_vec_len: 64,
        max_depth: 300,
    },
];

fuzz_target!(|data: &[u8]| {
    let [selector, config, input @ ..] = data else {
        return;
    };
    let selector = usize::from(*selector);
    let limits = LIMITS[usize::from(config & 0b11)];
    match decode(input, &limits, |c| decode_as(selector, c)) {
        Ok(echo) => {
            assert_eq!(
                echo.bytes, input,
                "an accepted input is not the canonical encoding of its value"
            );
            assert!(
                echo.depth <= limits.max_depth.min(256),
                "depth {} accepted",
                echo.depth
            );
            assert!(
                echo.longest <= limits.max_vec_len,
                "a sequence of {} accepted",
                echo.longest
            );
            assert!(input.len() as u64 <= limits.max_total_bytes);
            for cut in 0..input.len() {
                let error = decode(&input[..cut], &limits, |c| decode_as(selector, c))
                    .err()
                    .expect("a proper prefix of an encoding decodes");
                assert_eq!(error.code(), ErrorCode::Truncated, "{error}");
            }
        }
        Err(error) => assert!(
            error.code().is_input_fault() && error.offset() <= input.len() as u64,
            "{error} on a fuzz input"
        ),
    }
});
