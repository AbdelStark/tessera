// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The official BLAKE3 test vectors (`spec/vectors/blake3/test_vectors.json`, from the BLAKE3
//! repository) in key-derivation mode, the mode every Tessera hash uses. Two paths are checked:
//! `derive_key` itself, and the precomputed context key `tessera-hash` builds its hashers from.

mod common;

use std::fs;

use blake3::hazmat::{HasherExt, hash_derive_key_context};
use serde_json::Value;

#[test]
fn derive_key_matches_the_official_vectors() {
    let path = common::vectors().join("blake3/test_vectors.json");
    let file: Value =
        serde_json::from_str(&fs::read_to_string(path).expect("vector file")).expect("JSON");
    let context = file["context_string"].as_str().expect("context string");
    let context_key = hash_derive_key_context(context);
    let cases = file["cases"].as_array().expect("cases");
    assert_eq!(cases.len(), 35);
    for case in cases {
        let len = case["input_len"].as_u64().expect("input length");
        let input: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
        let expected = common::bytes(case["derive_key"].as_str().expect("derive_key"));
        // The full extended output, through both paths.
        let mut direct = vec![0u8; expected.len()];
        blake3::Hasher::new_derive_key(context)
            .update(&input)
            .finalize_xof()
            .fill(&mut direct);
        assert_eq!(direct, expected, "derive_key, input length {len}");
        let mut precomputed = vec![0u8; expected.len()];
        blake3::Hasher::new_from_context_key(&context_key)
            .update(&input)
            .finalize_xof()
            .fill(&mut precomputed);
        assert_eq!(precomputed, expected, "context key, input length {len}");
        assert_eq!(
            blake3::derive_key(context, &input)[..],
            expected[..32],
            "32-byte derive_key, input length {len}"
        );
    }
}
