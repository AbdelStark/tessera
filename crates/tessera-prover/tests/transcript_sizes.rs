// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The conformance models' transcript schedules (RFC-0014 'Transcript schedule'): the sizes
//! `spec/vectors/transcript-schedule/schedule.json` states for each model are those the prover's
//! derivations give from the model's public part and its first sequence, so that the vectors move
//! with the schedule. The oracle derives the weights' part itself, from RFC-0040; here it is
//! derived again with `tessera-pcs`'s weight layout and the stage schedule's pool:
//!
//! - **the leaf trees**: each LogUp instance of the requantization sites' layout (RFC-0045, the
//!   only gadget leaves included in this fixture), its lookup tree then its table tree, in table order;
//! - **the stages and the layer point**: the stage schedule's (RFC-0037);
//! - **the proximity points**: each matrix with committed rows, `⌈log2 m_T⌉` coordinates
//!   (RFC-0040, RFC-0049);
//! - **the opened matrices**: those the pool has claims on, each with its codeword's length.
//!
//! The advice's row-index point comes from the complete canonical layout (RFC-0050).

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;
use tessera_codec::{Ascii, DecodeLimits, envelope, schema};
use tessera_compiler::schedule::{LayerMap, NodeRef};
use tessera_compiler::{
    Gadget, ModelAdviceLayout, PublicModel, analyze, claim_list, derive_schedule, lower, select,
};
use tessera_hash::Mode;
use tessera_ir::{CommitmentClass, GadgetPolicy, IntegerModel, ParamSetId};
use tessera_pcs::WeightLayout;
use tessera_protocol::{
    LeafTree, MatrixCodeword, MatrixPoint, ParamSet, Protocol, ScheduleSizes, StageSizes,
    TranscriptSchedule, WeightMatrix,
};
use tessera_prover::leaf::{LeafLayout, Site, gadget_nodes};

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

/// The sizes the prover's derivations give a model at `test-small/1`.
fn derived(name: &str) -> ScheduleSizes {
    let (model, n) = conformance(name);
    let lowering = lower(PublicModel::of(&model), n).expect("the model lowers");
    let schedule = derive_schedule(&lowering).expect("a schedule");
    let ranges = analyze(&lowering.graph, &lowering.tables).expect("ranges");
    let selection = select(
        &lowering.graph,
        &ranges,
        &lowering.tables,
        &GadgetPolicy::default(),
    )
    .expect("every gadget fits");
    let layers = LayerMap::new(&lowering).expect("one template");

    // The leaf stage: the requantization sites' instances (RFC-0045).
    let requant = gadget_nodes(&lowering, &layers)
        .into_iter()
        .filter(|&site| {
            let node = match site {
                NodeRef::Position(position) => layers.node(position, 0),
                NodeRef::Node(id) => id,
            };
            matches!(
                selection.of(node),
                Some((
                    _,
                    Gadget::Requant1 { .. }
                        | Gadget::RequantSplit { .. }
                        | Gadget::RequantWide { .. }
                ))
            )
        });
    let sites: Vec<Site> = requant
        .map(|site| Site::requant(&lowering, &layers, &selection, site).expect("a site"))
        .collect();
    let layout = LeafLayout::new(&layers, sites).expect("a layout");
    let trees = layout
        .instances()
        .iter()
        .flat_map(|instance| {
            let table = selection
                .tables
                .iter()
                .find(|spec| spec.id == instance.table)
                .expect("the instance's table");
            [
                LeafTree::Fraction {
                    height: instance.height,
                },
                LeafTree::Fraction {
                    height: u32::from(table.log2_rows),
                },
            ]
        })
        .collect();

    // The weights: the committed rows of each matrix, and the matrices the pool has claims on.
    let params = ParamSet::TEST_SMALL;
    let claims = claim_list(
        PublicModel::of(&model),
        &selection,
        &GadgetPolicy::default(),
        params.layout,
        ParamSetId(Ascii::new(params.id).expect("a parameter id")),
    )
    .expect("the complete layout fits at max_context");
    let advice = ModelAdviceLayout::new(&lowering, &selection, &claims).expect("the advice layout");
    let weights = WeightLayout::new(
        params.layout.weights,
        model.weight_shapes().expect("the shapes"),
    )
    .expect("a weight layout");
    let proximity = tessera_pcs::WeightMatrix::ALL
        .into_iter()
        .filter_map(|matrix| {
            let rows = weights.matrix(matrix).rows();
            (rows > 0).then(|| MatrixPoint {
                matrix: WeightMatrix::ALL[matrix.index()],
                vars: rows.next_power_of_two().trailing_zeros(),
            })
        })
        .collect();
    let mut claimed = BTreeSet::new();
    for claim in schedule
        .pool
        .iter()
        .filter(|claim| claim.commitment == CommitmentClass::Weight)
    {
        for atom in &claim.atoms {
            // A weight-committed parameter that is no weight is a channel's class or multiplier.
            let matrix = schedule
                .weight_type(&lowering.graph, atom.kind)
                .map_or(tessera_pcs::WeightMatrix::ChannelParams, |ty| ty.matrix());
            claimed.insert(matrix.index());
        }
    }
    let inverse_rate = u32::from(params.layout.weights.log2_inv_rate);
    let opened = claimed
        .into_iter()
        .map(|index| {
            let matrix = tessera_pcs::WeightMatrix::ALL[index];
            MatrixCodeword {
                matrix: WeightMatrix::ALL[index].id(),
                len: 1 << (weights.matrix(matrix).log2_k() + inverse_rate),
            }
        })
        .collect();

    ScheduleSizes {
        trees,
        layer_vars: schedule.layers.vars,
        stages: schedule.stages[1..]
            .iter()
            .map(|stage| StageSizes {
                rounds: stage.rounds,
                zero_vars: stage.zero_vars,
            })
            .collect(),
        advice_vars: advice.layout().row_vars(),
        proximity,
        opened,
    }
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
        let sizes = derived(name);
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
