// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Claim-list decoding and semantic validation, including explicit refusal controls.

use std::sync::OnceLock;
use tv_codec::{DecodeLimits, decode};
use tv_core::{
    claims::{ClaimList, ParamSetId},
    sumcheck::Protocol,
    validation::{self, ClaimsError, StatementField},
};

const LIMITS: DecodeLimits = DecodeLimits {
    max_total_bytes: 2 << 20,
    max_vec_len: 4096,
    max_depth: 32,
};

fn fixture() -> &'static ClaimList {
    static FIXTURE: OnceLock<ClaimList> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        ClaimList::open(&crate::driver::fixture_bytes(false, "claims.tcl"), &LIMITS).unwrap()
    })
}

/// Input header: low three bits select raw payload, raw envelope, or a refusal control;
/// bit three selects protocol/2 for raw validation. Returns true on semantic success.
/// Allocation and per-input CPU caps belong to the binary running this oracle.
pub fn check(data: &[u8]) -> bool {
    let Some((&header, bytes)) = data.split_first() else {
        return false;
    };
    let protocol = if header & 8 == 0 {
        Protocol::One
    } else {
        Protocol::Two
    };
    let mode = header & 7;
    let claims = if mode <= 1 {
        let decoded = if mode == 0 {
            decode(bytes, &LIMITS, ClaimList::decode)
        } else {
            ClaimList::open(bytes, &LIMITS)
        };
        match decoded {
            Ok(value) => {
                let canonical = value.to_tce();
                let payload = if mode == 0 {
                    bytes
                } else {
                    &bytes[20..bytes.len() - 32]
                };
                assert_eq!(canonical, payload, "claim list has another encoding");
                value
            }
            Err(error) => {
                assert!(error.offset() <= bytes.len() as u64);
                assert!(
                    error.code().is_input_fault() || error.code() == tv_codec::ErrorCode::Memory
                );
                return false;
            }
        }
    } else {
        let mut value = fixture().clone();
        match mode {
            2 => value.protocol_version = 65535,
            3 => value.arch.n_heads = 0,
            4 => value.layout.weights.log2_inv_rate = 0,
            5 => value.params = ParamSetId::new("unknown-fuzz/1").unwrap(),
            _ => {}
        }
        value
    };
    // Structured controls use the fixture's protocol, independent of raw selector bits.
    let result = validation::validate_in(&claims, if mode <= 1 { protocol } else { Protocol::One });
    if (2..=4).contains(&mode) {
        let field = match mode {
            2 => StatementField::ProtocolVersion,
            3 => StatementField::Architecture,
            _ => StatementField::Layout,
        };
        assert!(
            matches!(result, Err(ClaimsError::StatementMismatch { field: got }) if got == field)
        );
        return false;
    }
    if mode == 5 {
        assert!(matches!(result, Err(ClaimsError::Params { .. })));
        return false;
    }
    match result {
        Ok(value) => {
            std::hint::black_box(value);
            true
        }
        Err(error) => {
            assert!(!matches!(error, ClaimsError::Internal), "{error}");
            false
        }
    }
}
