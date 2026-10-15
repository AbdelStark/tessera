// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Every decoder of `tv_fuzz::TYPES` on arbitrary bytes. The first byte selects the type; the
//! rest is decoded as a whole input under small limits, so that every limit is within reach.
//!
//! Any input must be accepted or rejected without a panic or an allocation beyond the input's
//! size (INV-TCE-3; libFuzzer's malloc limit catches a hostile length that allocates). An
//! accepted input must be the canonical encoding of its value (INV-TCE-2), and each of its proper
//! prefixes must be cut short. A rejection must point inside the input.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tv_codec::{DecodeLimits, ErrorCode, decode};
use tv_fuzz::decode_as;

/// 512 bytes in all, sequences of up to 32 elements, depth 6: below the depth of several types.
const LIMITS: DecodeLimits = DecodeLimits {
    max_total_bytes: 512,
    max_vec_len: 32,
    max_depth: 6,
};

fuzz_target!(|data: &[u8]| {
    let Some((&selector, input)) = data.split_first() else {
        return;
    };
    let selector = usize::from(selector);
    match decode(input, &LIMITS, |c| decode_as(selector, c)) {
        Ok(canonical) => {
            assert_eq!(canonical, input, "an accepted input is not canonical");
            for cut in 0..input.len() {
                let error = decode(&input[..cut], &LIMITS, |c| decode_as(selector, c))
                    .expect_err("a proper prefix of an encoding decodes");
                assert_eq!(error.code(), ErrorCode::Truncated, "{error}");
            }
        }
        Err(error) => assert!(
            error.offset() <= input.len() as u64,
            "{error} past the input"
        ),
    }
});
