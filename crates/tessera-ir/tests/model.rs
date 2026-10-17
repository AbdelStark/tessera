// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `IntegerModel` (03 §2): the committed vector decodes under the `IntegerModel` limits and
//! encodes back to the same bytes (INV-TCE-2), and the debug form shows no weight or channel
//! entry (05 §3).

use std::fs;
use std::path::Path;

use tessera_codec::{DecodeLimits, envelope, schema};
use tessera_ir::IntegerModel;

fn vector() -> Vec<u8> {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/model/integer-model.json");
    let text = fs::read_to_string(path).expect("the vector");
    let value: serde_json::Value = serde_json::from_str(&text).expect("JSON");
    let hex = value["hex"].as_str().expect("hex");
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex digits"))
        .collect()
}

#[test]
fn the_vector_decodes_and_encodes_back_to_its_bytes() {
    let bytes = vector();
    let model: IntegerModel =
        envelope::open(&bytes, &schema::INTEGER_MODEL, DecodeLimits::INTEGER_MODEL)
            .expect("a valid IntegerModel");
    assert_eq!(envelope::seal(&schema::INTEGER_MODEL, &model), bytes);
    assert_eq!(model.weights.len(), 2);
    assert_eq!(model.quant.channels.len(), 1);
}

#[test]
fn the_debug_form_shows_no_weight_or_channel_entry() {
    let model: IntegerModel = envelope::open(
        &vector(),
        &schema::INTEGER_MODEL,
        DecodeLimits::INTEGER_MODEL,
    )
    .expect("a valid IntegerModel");
    let shown = format!("{model:?}");
    for hidden in ["data:", "class:", "multiplier:"] {
        assert!(!shown.contains(hidden), "{hidden} in {shown}");
    }
    assert!(
        shown.contains("rows: 4") && shown.contains("channels: 2"),
        "{shown}"
    );
}
