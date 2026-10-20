// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Complete public model geometry against an independent graph walk (RFC-0014).
//! Both transcript modes replay every resulting label in the protocol and verifier consumers.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;
use tessera_codec::{Ascii, DecodeLimits, envelope, schema};
use tessera_compiler::{PublicModel, analyze, claim_list, lower, select};
use tessera_hash::Mode;
use tessera_ir::{ClaimList, GadgetPolicy, IntegerModel, ParamSetId, QuantPoint};
use tessera_protocol::{
    LeafTree, ParamSet, Protocol, ScheduleSizes, TranscriptSchedule, WeightMatrix,
};
use tessera_prover::{ModelSchedule, derive_model_schedule};

const MODELS: [&str; 4] = [
    "tiny-2-servable",
    "tiny-2-minimal",
    "tiny-2-affine",
    "tiny-4-servable",
];

fn vectors() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors")
}

/// A conformance model with its first sequence's tokens.
fn conformance(name: &str) -> (IntegerModel, u32) {
    let dir = vectors().join("models");
    let index: Value =
        serde_json::from_str(&fs::read_to_string(dir.join("models.json")).expect("the index"))
            .expect("JSON");
    let entry = index["models"]
        .as_array()
        .expect("models")
        .iter()
        .find(|m| m["name"] == name)
        .expect("the model");
    let bytes = fs::read(dir.join(entry["file"].as_str().expect("a file"))).expect("the file");
    let model = envelope::open(&bytes, &schema::INTEGER_MODEL, DecodeLimits::INTEGER_MODEL)
        .expect("an IntegerModel");
    let tokens = entry["sequences"][0]["tokens"].as_array().expect("tokens");
    (model, u32::try_from(tokens.len()).expect("small"))
}

/// Build public claims without reading any weights or channel values.
fn claims_for(name: &str, n: u32, query_block: u32, padded: bool) -> ClaimList {
    let (model, _) = conformance(name);
    let mut arch = model.arch.clone();
    let mut quant = model.quant.public.clone();
    let mut silu = model.quant.silu_in_exponents.clone();
    if padded {
        arch.n_layers = 3;
        fn extend<T: Clone>(values: &mut Vec<(QuantPoint, T)>) {
            let added: Vec<_> = values
                .iter()
                .filter_map(|(p, v)| match p {
                    QuantPoint::Layer { layer: 0, site } => Some((
                        QuantPoint::Layer {
                            layer: 2,
                            site: *site,
                        },
                        v.clone(),
                    )),
                    _ => None,
                })
                .collect();
            values.extend(added);
            values.sort_by_key(|(p, _)| *p);
        }
        extend(&mut quant.act_exponents);
        extend(&mut quant.shift_class_sets);
        extend(&mut quant.affine_sites);
        quant.score_scales.push((2, quant.score_scales[0].1));
        silu.push((2, silu[0].1));
    }
    let public = PublicModel {
        arch: &arch,
        profile: model.profile,
        quant: &quant,
        silu_in_exponents: &silu,
    };
    let lowering = lower(public, n).expect("the public model lowers");
    let ranges = analyze(&lowering.graph, &lowering.tables).expect("ranges");
    let policy = GadgetPolicy {
        query_block,
        ..GadgetPolicy::default()
    };
    let selection = select(&lowering.graph, &ranges, &lowering.tables, &policy).expect("gadgets");
    let params = ParamSet::TEST_SMALL;
    claim_list(
        public,
        &selection,
        &policy,
        params.layout,
        ParamSetId(Ascii::new(params.id).expect("ASCII")),
    )
    .expect("claim list")
}

fn derived(name: &str) -> ModelSchedule {
    let (_, n) = conformance(name);
    derive_model_schedule(&claims_for(name, n, 512, false), n).expect("complete model schedule")
}

/// The sizes the vector file states for `name`, both modes', which must be equal.
fn stated(file: &Value, name: &str) -> Value {
    let cases: Vec<&Value> = file["models"]
        .as_array()
        .expect("models")
        .iter()
        .filter(|c| {
            c["name"]
                .as_str()
                .is_some_and(|n| n.starts_with(&format!("{name}, ")))
        })
        .collect();
    assert_eq!(cases.len(), 2, "{name} in both modes");
    assert_eq!(cases[0]["sizes"], cases[1]["sizes"], "{name}");
    assert!(cases[0]["sizes"]["advice_vars"].is_u64(), "{name}");
    for case in &cases {
        assert_eq!(
            case["replayed"].as_u64(),
            Some(case["steps"].as_array().expect("steps").len() as u64),
            "{name}: the complete schedule is replayed"
        );
    }
    cases[0]["sizes"].clone()
}

/// `sizes` as the vector file writes them.
fn written(sizes: &ScheduleSizes) -> Value {
    let trees: Vec<Value> = sizes
        .trees
        .iter()
        .map(|tree| match *tree {
            LeafTree::Fraction { height } => {
                serde_json::json!({"kind": "fraction", "height": height})
            }
            LeafTree::Forest { trees_log, height } => {
                serde_json::json!({"kind": "forest", "trees_log": trees_log, "height": height})
            }
        })
        .collect();
    let stages: Vec<Value> = sizes
        .stages
        .iter()
        .map(|s| serde_json::json!({"rounds": s.rounds, "zero_vars": s.zero_vars}))
        .collect();
    let proximity: Vec<Value> = sizes
        .proximity
        .iter()
        .map(|p| serde_json::json!({"matrix": p.matrix.name(), "vars": p.vars}))
        .collect();
    let opened: Vec<Value> = sizes
        .opened
        .iter()
        .map(|c| {
            let name = WeightMatrix::from_id(c.matrix).expect("a matrix").name();
            serde_json::json!({"matrix": name, "codeword": c.len})
        })
        .collect();
    serde_json::json!({
        "trees": trees,
        "layer_vars": sizes.layer_vars,
        "stages": stages,
        "advice_vars": sizes.advice_vars,
        "proximity": proximity,
        "opened": opened,
    })
}

#[test]
fn the_vectors_state_the_sizes_the_derivations_give() {
    let path = vectors().join("transcript-schedule/schedule.json");
    let file: Value =
        serde_json::from_str(&fs::read_to_string(path).expect("vector file")).expect("JSON");
    for name in MODELS {
        let complete = derived(name);
        let counts: Vec<_> = complete
            .lookup_entries
            .iter()
            .map(|(t, n)| [u64::from(t.0), *n])
            .collect();
        for case in file["models"]
            .as_array()
            .expect("models")
            .iter()
            .filter(|case| {
                case["name"]
                    .as_str()
                    .expect("name")
                    .starts_with(&format!("{name},"))
            })
        {
            assert_eq!(
                serde_json::to_value(&counts).expect("counts"),
                case["lookup_entries"],
                "{name}: exact canonical contributions"
            );
        }
        let sizes = complete.sizes;
        assert_eq!(
            written(&sizes),
            stated(&file, name),
            "{name}: regenerate spec/vectors/transcript-schedule/ with the derivations' sizes"
        );
        // Every claim on a weight is a matrix with committed rows, as the oracle assumes.
        for codeword in &sizes.opened {
            assert!(
                sizes
                    .proximity
                    .iter()
                    .any(|p| p.matrix.id() == codeword.matrix),
                "{name}"
            );
        }
        // The schedules expand, with no label twice (INV-FS-4).
        for mode in [Mode::Live, Mode::NonInteractive] {
            let schedule =
                TranscriptSchedule::expand(&sizes, &ParamSet::TEST_SMALL, mode, Protocol::One)
                    .expect("a schedule");
            let labels: Vec<String> = schedule.labels().map(|l| l.to_string()).collect();
            let distinct: BTreeSet<&String> = labels.iter().collect();
            assert_eq!(distinct.len(), labels.len(), "{name}, {mode:?}");
        }
    }
}

#[test]
fn complete_geometry_at_public_boundaries() {
    let file: Value = serde_json::from_str(
        &fs::read_to_string(vectors().join("transcript-schedule/schedule.json")).expect("file"),
    )
    .expect("JSON");
    let cases = file["boundaries"].as_array().expect("boundaries");
    assert_eq!(cases.len(), 16, "eight public geometries, both modes");
    for case in cases {
        let name = case["name"].as_str().expect("name");
        let n = case["sequence_length"].as_u64().expect("n") as u32;
        let claims = claims_for(
            case["model"].as_str().expect("model"),
            n,
            case["query_block"].as_u64().expect("Q") as u32,
            case["padded"].as_bool().expect("padded"),
        );
        let actual = derive_model_schedule(&claims, n).expect("complete geometry");
        assert_eq!(written(&actual.sizes), case["sizes"], "{name}");
        let counts: Vec<_> = actual
            .lookup_entries
            .iter()
            .map(|(t, n)| [u64::from(t.0), *n])
            .collect();
        assert_eq!(
            serde_json::to_value(counts).expect("counts"),
            case["lookup_entries"],
            "{name}"
        );
        for mode in [Mode::Live, Mode::NonInteractive] {
            TranscriptSchedule::expand(&actual.sizes, &ParamSet::TEST_SMALL, mode, Protocol::One)
                .expect("valid dimensions");
        }
    }
}

#[test]
fn refuses_unsupported_public_inputs() {
    use tessera_compiler::CompileError;
    let claims = claims_for(MODELS[0], 3, 512, false);
    for n in [0, claims.arch.max_context + 1] {
        assert_eq!(
            derive_model_schedule(&claims, n),
            Err(CompileError::Usage {
                rule: "a sequence has 1 to max_context positions"
            })
        );
    }
    for version in [1, 2, u16::MAX] {
        let mut bad = claims.clone();
        bad.protocol_version = version;
        assert_eq!(
            derive_model_schedule(&bad, 3),
            Err(CompileError::Usage {
                rule: "model transcript geometry requires the current claim-list protocol version"
            })
        );
    }
    for remove in [true, false] {
        let mut bad = claims.clone();
        if remove {
            bad.tables.pop();
        } else {
            bad.tables.swap(0, 1);
        }
        assert_eq!(
            derive_model_schedule(&bad, 3),
            Err(CompileError::Usage {
                rule: "the claim list contains the complete selected catalog"
            })
        );
    }
}

#[test]
fn six_and_three_heads_derive_complete_schedule_and_expand() {
    use tessera_compiler::derive_schedule;
    for n in [1, 3, 4] {
        let mut claims = claims_for(MODELS[0], n, 512, false);
        claims.arch.n_heads = 6;
        claims.arch.n_kv_heads = 3;
        claims.arch.d_model = 96;
        let lowered =
            lower(PublicModel::of_claims(&claims), n).expect("public lowering accepts six heads");
        let ranges = analyze(&lowered.graph, &lowered.tables).expect("ranges");
        let selection = select(
            &lowered.graph,
            &ranges,
            &lowered.tables,
            &claims.gadget_policy,
        )
        .expect("selection");
        claims.tables = selection.tables;
        derive_schedule(&lowered).expect("six/three heads with d_head=16 preserve coordinates");
        let complete =
            derive_model_schedule(&claims, n).expect("complete model schedule for six/three heads");
        for mode in [Mode::Live, Mode::NonInteractive] {
            let schedule = TranscriptSchedule::expand(
                &complete.sizes,
                &ParamSet::TEST_SMALL,
                mode,
                Protocol::One,
            )
            .expect("valid transcript schedule");
            let labels: Vec<String> = schedule.labels().map(|l| l.to_string()).collect();
            let distinct: BTreeSet<&String> = labels.iter().collect();
            assert_eq!(distinct.len(), labels.len(), "n={n}, {mode:?}");
        }
    }
}

#[test]
fn non_power_of_two_head_width_returns_the_compiler_rule() {
    use tessera_compiler::{CompileError, derive_schedule};
    use tessera_ir::NodeId;
    for (heads, kv_heads, d_head, expected) in [
        (
            2,
            1,
            6,
            CompileError::Schedule {
                node: Some(NodeId(12)),
                rule: "a Reshape merges or splits only axes whose lengths are powers of two, the leading axis aside",
            },
        ),
        (
            6,
            3,
            48,
            CompileError::Schedule {
                node: Some(NodeId(12)),
                rule: "a Reshape merges or splits only axes whose lengths are powers of two, the leading axis aside",
            },
        ),
        (
            1,
            1,
            6,
            CompileError::Schedule {
                node: Some(NodeId(11)),
                rule: "a Rotate pairs a last axis whose length is a power of two",
            },
        ),
    ] {
        let mut claims = claims_for(MODELS[0], 3, 512, false);
        claims.arch.n_heads = heads;
        claims.arch.n_kv_heads = kv_heads;
        claims.arch.d_head = d_head;
        claims.arch.d_model = heads * d_head;
        let lowered = lower(PublicModel::of_claims(&claims), 3).expect("public lowering");
        let ranges = analyze(&lowered.graph, &lowered.tables).expect("ranges");
        let selection = select(
            &lowered.graph,
            &ranges,
            &lowered.tables,
            &claims.gadget_policy,
        )
        .expect("selection");
        claims.tables = selection.tables;
        assert_eq!(derive_schedule(&lowered).map(|_| ()), Err(expected));
        assert_eq!(derive_model_schedule(&claims, 3).map(|_| ()), Err(expected));
    }
}

#[test]
fn a_single_committed_row_has_a_zero_coordinate_proximity_point() {
    let mut claims = claims_for(MODELS[0], 3, 512, false);
    claims.layout.weights.log2_k = 20;
    let geometry = derive_model_schedule(&claims, 3).expect("large public row width");
    // Embed and LmHead each have 64 * 512 symbols and one real layer, hence one row.
    for matrix in [WeightMatrix::Embed, WeightMatrix::LmHead] {
        assert!(
            geometry
                .sizes
                .proximity
                .iter()
                .any(|p| p.matrix == matrix && p.vars == 0)
        );
    }
}
