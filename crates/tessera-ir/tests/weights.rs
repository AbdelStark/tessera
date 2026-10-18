// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The weight commitment of each conformance model (RFC-0040), from its `IntegerModel` and its
//! claim list, against `spec/vectors/weights/weights.json`, written by an independent oracle:
//! the shapes the architecture gives, every matrix's layout and root, `R_W` and the `.twr`
//! artifact. The commitment takes the claim list's `WeightLayoutParams`, and chains its columns
//! over the claim list's row block (RFC-0031 'The weight matrices' row block').
//!
//! And the domains the adapter holds `D` to before the commitment reads it, each refused with
//! its own error at its boundary: weights in `[−127, 127]` (RFC-0005 'Pipeline', RFC-0006
//! 'Range analysis'), a class below its site's shift classes and a multiplier from 1 (03 §2
//! 'Channels', RFC-0006 'Llama lowering'), and at least one layer (RFC-0007 'Verifier
//! validation').

use std::fs;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use serde_json::Value;
use tessera_codec::{DecodeLimits, envelope, schema};
use tessera_ir::{ClaimList, IntegerModel, IrError, QuantPoint, Site, WeightRef, WeightType};
use tessera_pcs::{WeightCommitment, WeightLayout, WeightMatrix};

fn vectors() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors")
}

fn json(path: &str) -> Value {
    let path = vectors().join(path);
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).expect("JSON")
}

fn hex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).expect("hex digits"))
        .collect()
}

fn number(value: &Value) -> u64 {
    value.as_u64().expect("a number")
}

fn model(name: &str) -> IntegerModel {
    let bytes = fs::read(vectors().join(format!("models/{name}.d"))).expect("the model");
    envelope::open(&bytes, &schema::INTEGER_MODEL, DecodeLimits::INTEGER_MODEL)
        .expect("a valid IntegerModel")
}

fn claim_list(name: &str) -> ClaimList {
    let file = json("claims/claim-lists.json");
    let case = file["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .find(|c| c["name"] == name)
        .unwrap_or_else(|| panic!("no claim list for {name}"));
    let bytes = hex(case["hex"].as_str().expect("hex"));
    envelope::open(&bytes, &schema::CLAIM_LIST, DecodeLimits::MESSAGE).expect("a claim list")
}

#[test]
fn every_conformance_model_commits_to_the_oracles_roots() {
    let file = json("weights/weights.json");
    let conformance: Vec<&Value> = file["models"]
        .as_array()
        .expect("models")
        .iter()
        .filter(|m| m.get("model").is_some())
        .collect();
    assert_eq!(conformance.len(), 4, "every conformance model");
    for expected in conformance {
        let name = expected["name"].as_str().expect("a name");
        let model = model(name);
        let claims = claim_list(name);
        let shapes = model.weight_shapes().expect("the shapes");
        // The architecture's shapes are the oracle's, and the tensors D holds.
        for (matrix, entry) in WeightMatrix::WEIGHTS
            .iter()
            .zip(expected["matrices"].as_array().expect("matrices"))
        {
            let shape = shapes.shapes[matrix.index()];
            assert_eq!(
                (u64::from(shape.out), u64::from(shape.inputs)),
                (number(&entry["out"]), number(&entry["in"])),
                "{name} {matrix:?}"
            );
        }
        for tensor in &model.weights {
            let shape = shapes.shapes[tensor.ty.matrix().index()];
            assert_eq!(
                (tensor.rows, tensor.cols),
                (shape.out, shape.inputs),
                "{name}"
            );
        }
        let listed: Vec<(String, u64)> = expected["channels"]
            .as_array()
            .expect("channels")
            .iter()
            .map(|c| {
                (
                    c["matrix"].as_str().expect("a name").to_owned(),
                    number(&c["layer"]),
                )
            })
            .collect();
        let ours: Vec<(String, u64)> = shapes
            .channels
            .iter()
            .map(|&(m, l)| (format!("{m:?}"), u64::from(l)))
            .collect();
        assert_eq!(ours, listed, "{name}: the weights with a ChannelParams");

        let layout = WeightLayout::new(claims.layout.weights, shapes).expect("a layout");
        let row_block = NonZeroU32::new(claims.layout.advice.row_block).expect("B ≥ 1");
        assert_eq!(row_block.get(), 4, "test-small/1's B");
        let commitment =
            WeightCommitment::commit(&layout, row_block, 0, &model.model_weights().expect("ok"))
                .expect("a commitment");
        for (matrix, entry) in WeightMatrix::ALL
            .iter()
            .zip(expected["matrices"].as_array().expect("matrices"))
        {
            let found = layout.matrix(*matrix);
            assert_eq!(
                u64::from(found.log2_k()),
                number(&entry["log2_k"]),
                "{name} {matrix:?}"
            );
            assert_eq!(found.rows(), number(&entry["rows"]), "{name} {matrix:?}");
            assert_eq!(
                hex(entry["root"].as_str().expect("hex")),
                commitment.public().root(*matrix),
                "{name} {matrix:?}"
            );
        }
        assert_eq!(
            commitment.weights_root().to_vec(),
            hex(expected["weights_root"].as_str().expect("hex")),
            "{name}: R_W"
        );
        let twr = fs::read(
            vectors()
                .join("weights")
                .join(expected["file"].as_str().expect("a file")),
        )
        .expect("the .twr artifact");
        assert_eq!(
            envelope::seal(&schema::WEIGHT_COMMITMENT_PUBLIC, commitment.public()),
            twr,
            "{name}: the .twr artifact"
        );
    }
}

#[test]
fn a_model_whose_weights_do_not_fit_its_architecture_is_refused() {
    let honest = model("tiny-2-servable");
    assert!(honest.model_weights().is_ok());
    // A weight transposed: as many entries, its rows and columns swapped.
    let mut transposed = honest.clone();
    let gate = transposed
        .weights
        .iter_mut()
        .find(|w| w.ty == WeightType::Gate)
        .expect("a Gate weight");
    (gate.rows, gate.cols) = (gate.cols, gate.rows);
    assert_eq!(
        transposed.model_weights().map(|_| ()),
        Err(IrError::Declaration {
            rule: "a weight's rows and columns are its type's output and input widths",
        })
    );
    // A ChannelParams with both class indices and multipliers.
    let mut both = honest.clone();
    let params = both.quant.channels.first_mut().expect("a ChannelParams");
    params.multiplier = vec![1; params.class.len()];
    assert_eq!(
        both.model_weights().map(|_| ()),
        Err(IrError::Declaration {
            rule: "a ChannelParams holds class indices or multipliers, not both",
        })
    );
    // Head widths past 32 bits.
    let mut wide = honest.clone();
    wide.arch.n_heads = u32::MAX;
    let refused = Err(IrError::Declaration {
        rule: "n_heads · d_head and n_kv_heads · d_head fit in 32 bits",
    });
    assert_eq!(wide.weight_shapes().map(|_| ()), refused);
    assert_eq!(wide.model_weights().map(|_| ()), refused);
}

fn refusal(model: &IntegerModel) -> Result<(), IrError> {
    model.model_weights().map(|_| ())
}

#[test]
fn a_weight_entry_of_minus_128_is_refused() {
    // RFC-0005 'Pipeline' step 5 rounds weights into [−127, 127]; RFC-0006 'Range analysis'
    // bounds every matmul by max|W| = 127, and 'Llama lowering' checks the entries before
    // lowering. −128 fits an i8 and nothing else refuses it.
    let honest = model("tiny-2-servable");
    assert_eq!(refusal(&honest), Ok(()));
    let at = |m: &IntegerModel, ty: WeightType, layer: u32| -> usize {
        m.weights
            .iter()
            .position(|w| (w.ty, w.layer) == (ty, layer))
            .expect("the weight")
    };
    for (ty, layer) in [
        (WeightType::Embed, 0),
        (WeightType::Q, 0),
        (WeightType::Down, 1),
        (WeightType::LmHead, 0),
    ] {
        for entry in [0, usize::MAX] {
            let mut m = honest.clone();
            let i = at(&m, ty, layer);
            let entry = entry.min(m.weights[i].data.len() - 1);
            // The domain's least value is a weight like any other.
            m.weights[i].data[entry] = -127;
            assert_eq!(refusal(&m), Ok(()), "{ty:?} {layer}: −127");
            m.weights[i].data[entry] = i8::MIN;
            assert_eq!(
                refusal(&m),
                Err(IrError::WeightOutOfRange {
                    weight: WeightRef { ty, layer },
                }),
                "{ty:?} {layer}: −128 at entry {entry}"
            );
        }
    }
    // Of two weights out of range, the first in D's order is named.
    let mut two = honest.clone();
    for (ty, layer) in [(WeightType::Up, 1), (WeightType::K, 0)] {
        let i = at(&two, ty, layer);
        two.weights[i].data[3] = i8::MIN;
    }
    assert_eq!(
        refusal(&two),
        Err(IrError::WeightOutOfRange {
            weight: WeightRef {
                ty: WeightType::K,
                layer: 0,
            },
        })
    );
}

/// The site that rescales each weight's outputs, whose shift classes its channel classes index
/// (03 §2 'Quantization', the `Site` comments): none after `Embed` and `LmHead`.
fn site_after(ty: WeightType) -> Option<Site> {
    match ty {
        WeightType::Q => Some(Site::RopeQ),
        WeightType::K => Some(Site::RopeK),
        WeightType::V => Some(Site::V),
        WeightType::O => Some(Site::AttnResidual),
        WeightType::Gate => Some(Site::Silu),
        WeightType::Up => Some(Site::Up),
        WeightType::Down => Some(Site::MlpResidual),
        WeightType::Embed | WeightType::LmHead => None,
    }
}

#[test]
fn a_channel_class_at_or_above_its_sites_shift_classes_is_refused() {
    // 03 §2 'Channels': `class[c]` indexes the shifts of the site after the weight; RFC-0006
    // 'Llama lowering' checks each "inside the site's classes" before lowering.
    let servable = model("tiny-2-servable");
    let mut types = Vec::new();
    for name in ["tiny-2-servable", "tiny-2-minimal"] {
        let honest = model(name);
        assert_eq!(refusal(&honest), Ok(()), "{name}");
        for (i, params) in honest.quant.channels.iter().enumerate() {
            if params.class.is_empty() {
                continue;
            }
            let weight = WeightRef {
                ty: params.ty,
                layer: params.layer,
            };
            let site = site_after(params.ty).expect("a weight a site rescales");
            types.push(params.ty);
            // Every site of one class but this weight's, of four, and every other channel of
            // class 0: only reading this weight's own site lets class 3 through.
            let mut m = honest.clone();
            for (point, shifts) in &mut m.quant.public.shift_class_sets {
                *shifts = match *point
                    == (QuantPoint::Layer {
                        layer: params.layer,
                        site,
                    }) {
                    true => vec![1, 2, 3, 4],
                    false => vec![1],
                };
            }
            for (j, other) in m.quant.channels.iter_mut().enumerate() {
                if j != i {
                    other.class.fill(0);
                }
            }
            let last = m.quant.channels[i].class.len() - 1;
            m.quant.channels[i].class[last] = 3;
            assert_eq!(refusal(&m), Ok(()), "{name} {weight:?}: class 3 of 4");
            for class in [4, u8::MAX] {
                m.quant.channels[i].class[last] = class;
                assert_eq!(
                    refusal(&m),
                    Err(IrError::ClassOutOfRange { weight, classes: 4 }),
                    "{name} {weight:?}: class {class} of 4"
                );
            }
        }
    }
    types.sort();
    types.dedup();
    assert_eq!(types.len(), 7, "every weight a site rescales: {types:?}");

    // The model's own classes: n − 1 passes, n does not.
    let params = &servable.quant.channels[0];
    let weight = WeightRef {
        ty: params.ty,
        layer: params.layer,
    };
    let point = QuantPoint::Layer {
        layer: params.layer,
        site: site_after(params.ty).expect("a site"),
    };
    let n = servable
        .quant
        .public
        .shift_class_sets
        .iter()
        .find(|(p, _)| *p == point)
        .map(|(_, shifts)| shifts.len())
        .expect("a pot site");
    assert!(n >= 2, "a site of several classes");
    let mut m = servable.clone();
    m.quant.channels[0].class[0] = u8::try_from(n - 1).expect("at most four");
    assert_eq!(refusal(&m), Ok(()));
    m.quant.channels[0].class[0] = u8::try_from(n).expect("at most four");
    assert_eq!(
        refusal(&m),
        Err(IrError::ClassOutOfRange { weight, classes: n })
    );

    // No class at all is inside the shift classes of a site that has none: an affine site's,
    // and those after Embed and LmHead, which no rescale site follows.
    let affine = model("tiny-2-affine");
    let i = affine
        .quant
        .channels
        .iter()
        .position(|c| !c.multiplier.is_empty())
        .expect("an affine site's weight");
    let mut m = affine.clone();
    let rows = m.quant.channels[i].multiplier.len();
    m.quant.channels[i].multiplier.clear();
    m.quant.channels[i].class = vec![0; rows];
    let weight = WeightRef {
        ty: m.quant.channels[i].ty,
        layer: m.quant.channels[i].layer,
    };
    assert_eq!(
        refusal(&m),
        Err(IrError::ClassOutOfRange { weight, classes: 0 })
    );
    for ty in [WeightType::Embed, WeightType::LmHead] {
        let mut m = servable.clone();
        let mut params = m.quant.channels[0].clone();
        params.ty = ty;
        params.layer = 0;
        params.class.fill(0);
        m.quant.channels.insert(0, params);
        assert_eq!(
            refusal(&m),
            Err(IrError::ClassOutOfRange {
                weight: WeightRef { ty, layer: 0 },
                classes: 0,
            }),
            "{ty:?}"
        );
    }
}

#[test]
fn a_multiplier_of_0_is_refused() {
    // 03 §2 'Channels', RFC-0025: an affine site's multipliers lie from 1 to 2^16 − 1; RFC-0006
    // 'Llama lowering' checks them before lowering.
    let honest = model("tiny-2-affine");
    assert_eq!(refusal(&honest), Ok(()));
    let mut found = 0;
    for (i, params) in honest.quant.channels.iter().enumerate() {
        if params.multiplier.is_empty() {
            continue;
        }
        found += 1;
        let weight = WeightRef {
            ty: params.ty,
            layer: params.layer,
        };
        let last = params.multiplier.len() - 1;
        let mut m = honest.clone();
        for multiplier in [1, u16::MAX] {
            m.quant.channels[i].multiplier[last] = multiplier;
            assert_eq!(refusal(&m), Ok(()), "{weight:?}: {multiplier}");
        }
        m.quant.channels[i].multiplier[last] = 0;
        assert_eq!(
            refusal(&m),
            Err(IrError::MultiplierOutOfRange { weight }),
            "{weight:?}"
        );
    }
    assert!(found > 0, "tiny-2-affine has affine sites");
}

#[test]
fn a_model_of_no_layer_is_refused() {
    // RFC-0007 'Verifier validation', architecture consistency: `n_layers ≥ 1`, as the compiler
    // checks ("a model has at least one layer"); the per-layer weights' layer count pads to
    // L_pad (RFC-0040 'The matrices').
    let mut none = model("tiny-2-servable");
    none.arch.n_layers = 0;
    assert_eq!(none.weight_shapes().map(|_| ()), Err(IrError::NoLayers));
    assert_eq!(refusal(&none), Err(IrError::NoLayers));
    // Before any other rule of the shapes.
    none.arch.n_heads = u32::MAX;
    assert_eq!(none.weight_shapes().map(|_| ()), Err(IrError::NoLayers));
    // One layer is a model.
    let mut one = model("tiny-2-servable");
    one.arch.n_layers = 1;
    assert!(one.weight_shapes().is_ok());
}
