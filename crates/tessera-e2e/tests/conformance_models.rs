// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The tiny conformance models of `spec/vectors/models/` (07 §5), as the prover loads them.
//!
//! - Each artifact opens under the `IntegerModel` limits, its envelope digest is the index's, and
//!   it encodes back to its bytes (INV-TCE-2).
//! - The compiler accepts it, and lowers, analyzes and selects gadgets for each of its sequences.
//! - The index holds, for each sequence, the residual at each layer boundary and a logit per token
//!   and vocabulary entry, in the model's shapes. `tessera-compiler`'s lowering tests compare the
//!   logits with the lowered graph's.

use std::fs;
use std::path::{Path, PathBuf};

use tessera_codec::{DecodeLimits, envelope, schema};
use tessera_compiler::{PublicModel, analyze, lower, select, validate};
use tessera_ir::{GadgetPolicy, IntegerModel};

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/models")
}

fn count(value: &serde_json::Value) -> usize {
    usize::try_from(value.as_u64().expect("a count")).expect("small")
}

fn rows(value: &serde_json::Value) -> Vec<usize> {
    value
        .as_array()
        .expect("rows")
        .iter()
        .map(|row| row.as_array().expect("a row").len())
        .collect()
}

#[test]
fn every_conformance_model_loads_validates_and_lowers() {
    let text = fs::read_to_string(dir().join("models.json")).expect("the index");
    let index: serde_json::Value = serde_json::from_str(&text).expect("JSON");
    assert_eq!(index["schema"], "0x0101");
    assert_eq!(index["version"], "1");
    let models = index["models"].as_array().expect("models");
    let names: Vec<&str> = models
        .iter()
        .map(|m| m["name"].as_str().expect("a name"))
        .collect();
    assert_eq!(
        names,
        ["tiny-2-servable", "tiny-2-minimal", "tiny-4-servable"]
    );
    for entry in models {
        let name = entry["name"].as_str().expect("a name");
        let bytes =
            fs::read(dir().join(entry["file"].as_str().expect("a file"))).expect("the file");
        assert_eq!(bytes.len(), count(&entry["bytes"]), "{name}");
        let digest: String = bytes[bytes.len() - 32..]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(
            digest,
            entry["digest"].as_str().expect("a digest"),
            "{name}"
        );
        let model: IntegerModel =
            envelope::open(&bytes, &schema::INTEGER_MODEL, DecodeLimits::INTEGER_MODEL)
                .expect("an IntegerModel");
        assert_eq!(
            envelope::seal(&schema::INTEGER_MODEL, &model),
            bytes,
            "{name} encodes back to its bytes"
        );
        assert_eq!(
            format!("{:?}", model.profile),
            entry["profile"].as_str().expect("a profile")
        );
        let shape = &entry["shape"];
        let arch = &model.arch;
        for (field, value) in [
            ("n_layers", arch.n_layers),
            ("d_model", arch.d_model),
            ("n_heads", arch.n_heads),
            ("n_kv_heads", arch.n_kv_heads),
            ("d_head", arch.d_head),
            ("d_ff", arch.d_ff),
            ("vocab", arch.vocab),
            ("max_context", arch.max_context),
        ] {
            assert_eq!(
                shape[field].as_u64(),
                Some(u64::from(value)),
                "{name}: {field}"
            );
        }
        validate(&model).expect("the compiler accepts the model");
        let (layers, d, vocab) = (
            arch.n_layers as usize,
            arch.d_model as usize,
            arch.vocab as usize,
        );
        let sequences = entry["sequences"].as_array().expect("sequences");
        assert_eq!(sequences.len(), 2, "{name}");
        for sequence in sequences {
            let tokens = sequence["tokens"].as_array().expect("tokens");
            let n = tokens.len();
            assert!(
                tokens
                    .iter()
                    .all(|t| t.as_u64().is_some_and(|t| t < u64::from(arch.vocab)))
            );
            let residuals = sequence["residuals"].as_array().expect("residuals");
            assert_eq!(
                residuals.len(),
                layers + 1,
                "{name}: a checkpoint per layer boundary"
            );
            for checkpoint in residuals {
                assert_eq!(rows(checkpoint), vec![d; n], "{name}");
            }
            assert_eq!(rows(&sequence["logits"]), vec![vocab; n], "{name}");
            let lowering = lower(PublicModel::of(&model), u32::try_from(n).expect("a length"))
                .expect("the model lowers");
            let ranges = analyze(&lowering.graph, &lowering.tables).expect("ranges");
            select(
                &lowering.graph,
                &ranges,
                &lowering.tables,
                &GadgetPolicy::default(),
            )
            .expect("every gadget fits");
        }
    }
}
