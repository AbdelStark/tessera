// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared public lowering-vector readers and variant edits; no compiler algorithm is reused.
#![allow(dead_code)]
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use tessera_codec::{DecodeLimits, Rational, envelope, schema};
use tessera_ir::{ArchSpec, IntegerModel, PublicQuantSpec, QuantPoint, Site};

pub fn vectors() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors")
}

pub fn read_json(path: &Path) -> Value {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Each conformance model's artifact, decoded through the public API, by file name.
pub fn models() -> BTreeMap<String, IntegerModel> {
    let dir = vectors().join("models");
    let index = read_json(&dir.join("models.json"));
    index["models"]
        .as_array()
        .expect("models")
        .iter()
        .map(|entry| {
            let file = entry["file"].as_str().expect("a file");
            let bytes = fs::read(dir.join(file)).expect("the artifact");
            let model = envelope::open(&bytes, &schema::INTEGER_MODEL, DecodeLimits::INTEGER_MODEL)
                .expect("an IntegerModel");
            (file.to_owned(), model)
        })
        .collect()
}

pub fn int(value: &Value) -> i64 {
    value
        .as_i64()
        .unwrap_or_else(|| panic!("an integer: {value}"))
}

pub fn uint(value: &Value) -> u32 {
    u32::try_from(int(value)).unwrap_or_else(|_| panic!("a u32: {value}"))
}

fn site(name: &str) -> Site {
    match name {
        "AttnNorm" => Site::AttnNorm,
        "RopeQ" => Site::RopeQ,
        "RopeK" => Site::RopeK,
        "V" => Site::V,
        "AttnOut" => Site::AttnOut,
        "AttnResidual" => Site::AttnResidual,
        "MlpNorm" => Site::MlpNorm,
        "Silu" => Site::Silu,
        "Up" => Site::Up,
        "Product" => Site::Product,
        "MlpResidual" => Site::MlpResidual,
        other => panic!("unknown site {other}"),
    }
}

fn shift(value: &Value) -> i8 {
    i8::try_from(int(value)).unwrap_or_else(|_| panic!("a shift: {value}"))
}

/// A variant's public part: the model's, with the edits the vectors' README describes, made in
/// each layer they name. The lowering reads the public part alone (RFC-0006 'Llama lowering').
pub fn edited(model: &IntegerModel, variant: &Value) -> (ArchSpec, PublicQuantSpec) {
    let mut arch = model.arch.clone();
    let mut quant = model.quant.public.clone();
    if let Some(heads) = variant.get("n_kv_heads") {
        arch.n_kv_heads = uint(heads);
    }
    for edit in variant
        .get("rescale")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let point = QuantPoint::Layer {
            layer: uint(&edit[0]),
            site: site(edit[1].as_str().expect("a site")),
        };
        quant.shift_class_sets.retain(|(p, _)| *p != point);
        quant.affine_sites.retain(|(p, _)| *p != point);
        match edit[2].as_str() {
            Some("pot") => {
                let shifts = edit[3].as_array().expect("shifts").iter().map(shift);
                quant.shift_class_sets.push((point, shifts.collect()));
            }
            Some("affine") => quant.affine_sites.push((point, shift(&edit[3]))),
            other => panic!("a rescale is pot or affine: {other:?}"),
        }
    }
    quant.shift_class_sets.sort_by_key(|(p, _)| *p);
    quant.affine_sites.sort_by_key(|(p, _)| *p);
    for edit in variant
        .get("score_scales")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let layer = uint(&edit[0]);
        let den = u64::try_from(int(&edit[2])).expect("a denominator");
        let scale = Rational::new(int(&edit[1]), den).expect("a score scale");
        let entry = quant
            .score_scales
            .iter_mut()
            .find(|(l, _)| *l == layer)
            .expect("the layer's score scale");
        entry.1 = scale;
    }
    (arch, quant)
}
