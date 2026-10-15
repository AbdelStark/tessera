// SPDX-License-Identifier: Apache-2.0 OR MIT

//! TCE for field elements (RFC-0003 'Primitive types'). An element is its canonical value as a
//! `u64`, and decoding rejects anything at or above the modulus. An extension element is `c0`
//! then `c1`. Checked against the field cases of `spec/vectors/codec/`, and with the invariants
//! of RFC-0003 on values drawn near the modulus, where canonical and not meet.

use std::fs;
use std::path::PathBuf;

use proptest::prelude::*;
use serde_json::Value;
use tessera_codec::error::ErrorCode;
use tessera_codec::{DecodeLimits, decode, encode};
use tessera_field::golden::{self, decode_as};
use tessera_field::{
    F65537, F65537Ext2, Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension,
};

const LIMITS: DecodeLimits = DecodeLimits {
    max_total_bytes: 1 << 20,
    max_vec_len: 1 << 16,
    max_depth: 32,
};

const P: u64 = Goldilocks::MODULUS;

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

/// The field entries of a list of cases: `(case, type, bytes)`.
fn field_cases(list: &Value) -> Vec<(&Value, String, Vec<u8>)> {
    let cases: Vec<_> = list
        .as_array()
        .expect("cases")
        .iter()
        .map(|case| {
            let ty = case["type"].as_str().expect("type").to_owned();
            (case, ty, bytes(case["hex"].as_str().expect("hex")))
        })
        .filter(|(_, ty, _)| decode_as(ty, &[], LIMITS).is_some())
        .collect();
    assert!(cases.len() >= 8, "only {} field cases", cases.len());
    cases
}

#[test]
fn every_field_case_has_a_decoder() {
    let types = golden::cases()
        .into_iter()
        .map(|c| c.ty)
        .chain(golden::corpus().into_iter().map(|r| r.ty))
        .chain(golden::boundaries().into_iter().map(|b| b.ty));
    for ty in types {
        assert!(decode_as(ty, &[], LIMITS).is_some(), "no decoder for {ty}");
    }
}

#[test]
fn field_vectors_hold() {
    let file = load("golden.json");
    let golden_limits = limits(&file["limits"]);
    for (case, ty, encoded) in field_cases(&file["cases"]) {
        let (shown, again) = decode_as(&ty, &encoded, golden_limits)
            .expect("a field type")
            .unwrap_or_else(|e| panic!("{ty}: {e}"));
        let value: Value = serde_json::from_str(&shown.json()).expect("JSON");
        assert_eq!(value, case["value"], "{ty}");
        assert_eq!(again, encoded, "{ty}");
        for len in 0..encoded.len() {
            let error = decode_as(&ty, &encoded[..len], golden_limits)
                .expect("a field type")
                .expect_err("a proper prefix decoded");
            assert_eq!(
                error.code,
                ErrorCode::CodecTruncated,
                "{ty}, first {len} bytes"
            );
        }
    }
    let file = load("noncanonical.json");
    let corpus_limits = limits(&file["limits"]);
    for (case, ty, input) in field_cases(&file["cases"]) {
        let code =
            ErrorCode::from_identifier(case["code"].as_str().expect("code")).expect("a known code");
        let error = decode_as(&ty, &input, corpus_limits)
            .expect("a field type")
            .expect_err("a corpus entry decoded");
        assert_eq!(error.code, code, "{ty}: {}", case["why"]);
    }
    let accepted: Vec<_> = file["accepted"]
        .as_array()
        .expect("accepted cases")
        .iter()
        .filter_map(|case| {
            let ty = case["type"].as_str().expect("type");
            let input = bytes(case["hex"].as_str().expect("hex"));
            Some((ty.to_owned(), decode_as(ty, &input, corpus_limits)?))
        })
        .collect();
    assert_eq!(accepted.len(), golden::boundaries().len());
    for (ty, outcome) in accepted {
        assert!(outcome.is_ok(), "{ty} at the depth limit: {outcome:?}");
    }
}

/// A `u64` within 2^20 of `P` or of 2^64 one time in two: the edge between canonical and not.
fn word() -> impl Strategy<Value = u64> {
    prop_oneof![
        any::<u64>(),
        (0..1u64 << 20).prop_map(|d| P.wrapping_sub(1 << 19).wrapping_add(d)),
        (0..1u64 << 20).prop_map(|d| u64::MAX - d),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4096))]

    /// A `u64` decodes as a Goldilocks element exactly when it is below `P`, and then encodes
    /// back to the same bytes (INV-TCE-1 and INV-TCE-2).
    #[test]
    fn goldilocks_encodings_are_canonical(x in word()) {
        let input = x.to_le_bytes();
        match decode::<Goldilocks>(&input, LIMITS) {
            Ok(value) => {
                prop_assert!(x < P);
                prop_assert_eq!(encode(&value), input.to_vec());
            }
            Err(error) => {
                prop_assert!(x >= P);
                prop_assert_eq!(error.code, ErrorCode::CodecNoncanonical);
            }
        }
    }

    /// The same bound for the test field.
    #[test]
    fn f65537_encodings_are_canonical(x in 0..140_000u64) {
        let input = x.to_le_bytes();
        match decode::<F65537>(&input, LIMITS) {
            Ok(value) => {
                prop_assert!(x < F65537::MODULUS);
                prop_assert_eq!(encode(&value), input.to_vec());
            }
            Err(error) => {
                prop_assert!(x >= F65537::MODULUS);
                prop_assert_eq!(error.code, ErrorCode::CodecNoncanonical);
            }
        }
    }

    /// An extension element decodes exactly when both coefficients are canonical.
    #[test]
    fn extension_encodings_are_canonical(c0 in word(), c1 in word()) {
        let mut input = c0.to_le_bytes().to_vec();
        input.extend_from_slice(&c1.to_le_bytes());
        match decode::<GoldilocksExt2>(&input, LIMITS) {
            Ok(value) => {
                prop_assert!(c0 < P && c1 < P);
                prop_assert_eq!(encode(&value), input.clone());
            }
            Err(error) => {
                prop_assert!(c0 >= P || c1 >= P);
                prop_assert_eq!(error.code, ErrorCode::CodecNoncanonical);
            }
        }
        let small = F65537Ext2::new(F65537::from_u64_reduced(c0), F65537::from_u64_reduced(c1));
        prop_assert_eq!(decode::<F65537Ext2>(&encode(&small), LIMITS), Ok(small));
    }
}
