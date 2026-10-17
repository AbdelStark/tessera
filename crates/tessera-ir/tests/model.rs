// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `IntegerModel` (03 §2): each committed vector decodes under the `IntegerModel` limits and
//! encodes back to the same bytes (INV-TCE-2), and the debug form shows no weight or channel
//! entry (05 §3).

use std::fs;
use std::path::Path;

use tessera_codec::{DecodeLimits, envelope, schema};
use tessera_ir::IntegerModel;

/// Each case's name and bytes.
fn vectors() -> Vec<(String, Vec<u8>)> {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/model/integer-model.json");
    let text = fs::read_to_string(path).expect("the vectors");
    let file: serde_json::Value = serde_json::from_str(&text).expect("JSON");
    let cases = file["cases"].as_array().expect("cases");
    assert_eq!(cases.len(), 2, "both models");
    cases
        .iter()
        .map(|case| {
            let hex = case["hex"].as_str().expect("hex");
            let bytes = (0..hex.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex digits"))
                .collect();
            (case["name"].as_str().expect("name").to_owned(), bytes)
        })
        .collect()
}

fn open(bytes: &[u8]) -> IntegerModel {
    envelope::open(bytes, &schema::INTEGER_MODEL, DecodeLimits::INTEGER_MODEL)
        .expect("a valid IntegerModel")
}

#[test]
fn the_vectors_decode_and_encode_back_to_their_bytes() {
    for (name, bytes) in vectors() {
        let model = open(&bytes);
        assert_eq!(
            envelope::seal(&schema::INTEGER_MODEL, &model),
            bytes,
            "{name}"
        );
        assert_eq!(model.weights.len(), 9, "{name}: every weight type");
        assert_eq!(model.quant.channels.len(), 2, "{name}");
    }
}

#[test]
fn the_debug_form_shows_no_weight_or_channel_entry() {
    for (name, bytes) in vectors() {
        let shown = format!("{:?}", open(&bytes));
        for hidden in ["data:", "class:", "multiplier:"] {
            assert!(!shown.contains(hidden), "{name}: {hidden} in {shown}");
        }
        assert!(
            shown.contains("rows: 6") && shown.contains("channels: 2"),
            "{name}: {shown}"
        );
    }
}
