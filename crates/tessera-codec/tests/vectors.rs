// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `spec/vectors/codec/`, as the codec's own types see it. Field types are checked in
//! `tessera-field`.
//! - Every golden case decodes to its `value` and encodes back to the same bytes.
//! - Truncating a golden case at any byte offset fails with `ERR-CODEC-TRUNCATED`.
//! - Every corpus entry fails with the code RFC-0003 'Failure modes' assigns it.
//! - Every input exactly at a limit decodes.

use std::fs;
use std::path::PathBuf;

use serde_json::Value;
use tessera_codec::DecodeLimits;
use tessera_codec::error::ErrorCode;
use tessera_codec::golden::{self, decode_as};

fn load(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../spec/vectors/codec")
        .join(name);
    serde_json::from_str(&fs::read_to_string(&path).expect("vector file")).expect("JSON")
}

fn limits(value: &Value) -> DecodeLimits {
    let field = |key: &str| value[key].as_u64().expect("a limit");
    DecodeLimits {
        max_total_bytes: field("max_total_bytes"),
        max_vec_len: field("max_vec_len"),
        max_depth: u16::try_from(field("max_depth")).expect("a u16 depth"),
    }
}

fn bytes(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex"))
        .collect()
}

/// Any limits: `decode_as` names its types whatever the input.
const PROBE: DecodeLimits = DecodeLimits {
    max_total_bytes: 0,
    max_vec_len: 0,
    max_depth: 0,
};

/// The entries of `list` whose type this crate defines, with their bytes.
fn own(list: &Value) -> Vec<(&Value, String, Vec<u8>)> {
    list.as_array()
        .expect("a list of cases")
        .iter()
        .map(|case| {
            let ty = case["type"].as_str().expect("type").to_owned();
            (case, ty, bytes(case["hex"].as_str().expect("hex")))
        })
        .filter(|(_, ty, _)| decode_as(ty, &[], PROBE).is_some())
        .collect()
}

/// Every case this crate generates names a type it can decode, so no case is skipped.
#[test]
fn every_case_has_a_decoder() {
    let types = golden::cases()
        .into_iter()
        .map(|c| c.ty)
        .chain(golden::corpus().into_iter().map(|r| r.ty))
        .chain(golden::boundaries().into_iter().map(|b| b.ty));
    for ty in types {
        assert!(decode_as(ty, &[], PROBE).is_some(), "no decoder for {ty}");
    }
}

#[test]
fn golden_cases_decode_to_their_values_and_back() {
    let file = load("golden.json");
    let limits = limits(&file["limits"]);
    let cases = own(&file["cases"]);
    assert!(cases.len() >= 40, "only {} codec cases", cases.len());
    for (case, ty, encoded) in cases {
        let (shown, again) = decode_as(&ty, &encoded, limits)
            .expect("a codec type")
            .unwrap_or_else(|e| panic!("{ty} {encoded:02x?}: {e}"));
        let value: Value = serde_json::from_str(&shown.json()).expect("JSON");
        assert_eq!(value, case["value"], "{ty} {encoded:02x?}");
        assert_eq!(again, encoded, "{ty}");
    }
}

#[test]
fn every_truncation_of_a_golden_case_is_truncated() {
    let file = load("golden.json");
    let limits = limits(&file["limits"]);
    for (_, ty, encoded) in own(&file["cases"]) {
        for len in 0..encoded.len() {
            let error = decode_as(&ty, &encoded[..len], limits)
                .expect("a codec type")
                .expect_err("a proper prefix decoded");
            assert_eq!(
                error.code,
                ErrorCode::CodecTruncated,
                "{ty}, first {len} bytes"
            );
        }
    }
}

#[test]
fn every_corpus_entry_is_rejected_with_its_code_at_its_offset() {
    let file = load("noncanonical.json");
    let limits = limits(&file["limits"]);
    let cases = own(&file["cases"]);
    assert!(
        cases.len() >= 25,
        "only {} codec corpus entries",
        cases.len()
    );
    for (case, ty, input) in cases {
        let code =
            ErrorCode::from_identifier(case["code"].as_str().expect("code")).expect("a known code");
        let offset = case["offset"].as_u64().expect("offset");
        let error = decode_as(&ty, &input, limits)
            .expect("a codec type")
            .expect_err("a corpus entry decoded");
        assert_eq!(
            (error.code, error.offset),
            (code, offset),
            "{ty}: {}",
            case["why"]
        );
    }
}

#[test]
fn inputs_exactly_at_the_limits_decode() {
    let file = load("noncanonical.json");
    let limits = limits(&file["limits"]);
    let accepted = own(&file["accepted"]);
    assert_eq!(accepted.len(), golden::boundaries().len());
    for (case, ty, input) in accepted {
        let outcome = decode_as(&ty, &input, limits).expect("a codec type");
        assert_eq!(
            outcome.map(|(_, again)| again),
            Ok(input),
            "{ty}: {}",
            case["why"]
        );
    }
}
