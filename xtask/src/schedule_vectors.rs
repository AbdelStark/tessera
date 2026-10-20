// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The stage schedules (RFC-0010 'Stages', RFC-0037), `spec/vectors/schedule/schedule.json`: each
//! conformance model's schedule as the compiler derives it (`tessera_compiler::derive_schedule`),
//! at the lengths 1, 4, 5 and `max_context`. The independent verifier derives its own schedule
//! from the claim list and must give the same one (INV-SCH-1, #114); its tests read this file and
//! never the compiler's code.
//!
//! Each case states, for people and for the verifier's comparison:
//!
//! - the stage count, the layers `L` and the layer variables `λ`;
//! - the leaf stage's sites, in order: a gadget's input, by layer 0's node for a template
//!   position, or the logits; and the values its split claims send;
//! - each reduction stage's rounds, zero-check coordinates and split values, and its instances in
//!   order: the relation's kind, the node (layer 0's for a template position), the tensor the
//!   claim states (layer 0's), whether it is stacked over the layers, its variables, and its
//!   claim's point and layer point, coordinate by coordinate;
//! - the pool's claims in order: the stage that leaves each, its commitment, its atoms and its
//!   point.
//!
//! A coordinate is written `leaf:{s}:{i}` for coordinate `i` of leaf site `s`'s point,
//! `layer:{i}` for the layer point `ρ_1`'s, `zero:{i}` for stage 1's zero-check point's,
//! `round:{k}:{j}` for the challenge of round `j` of stage `k`, and `half` for `½`.

use std::path::Path;

use serde_json::{Value, json};
use tessera_compiler::schedule::{Coord, Kind, Layers, NodeRef, Relation, Site, StageSchedule};
use tessera_compiler::{Lowering, PublicModel, derive_schedule, lower};
use tessera_ir::CommitmentClass;

use crate::claim_vectors::models;

/// Where the schedules are written.
pub(crate) const PATH: &str = "spec/vectors/schedule/schedule.json";

fn coord(coord: &Coord) -> Value {
    Value::String(match *coord {
        Coord::Leaf { site, index } => format!("leaf:{site}:{index}"),
        Coord::Layer(i) => format!("layer:{i}"),
        Coord::Zero(i) => format!("zero:{i}"),
        Coord::Round { stage, round } => format!("round:{stage}:{round}"),
        Coord::Half => "half".to_owned(),
    })
}

fn coords(point: &[Coord]) -> Value {
    Value::Array(point.iter().map(coord).collect())
}

/// RFC-0010's name of a relation's kind.
fn relation(relation: Relation) -> &'static str {
    match relation {
        Relation::MatMulWeight => "MatMulWeight",
        Relation::MatMulAct { .. } => "MatMulAct",
        Relation::Mul(_) => "Elementwise",
        Relation::SumSquares => "SumSquares",
        Relation::Rotate => "Rotate",
        Relation::ScaleChannels => "ScaleChannels",
        Relation::Equality | Relation::Booleanity => "ZeroCheck",
    }
}

/// The node of `node`: layer 0's for a template position.
fn node(schedule: &StageSchedule, node: NodeRef) -> u32 {
    match node {
        NodeRef::Node(id) => id.0,
        NodeRef::Position(position) => schedule.layers.node(position, 0).0,
    }
}

/// The tensor `kind` names: layer 0's for a stacked kind, `null` for the layer input.
fn tensor(lowering: &Lowering, schedule: &StageSchedule, kind: Kind) -> Value {
    match kind {
        Kind::LayerInput => Value::Null,
        _ => schedule
            .layers
            .tensor(&lowering.graph, kind, 0)
            .map_or(Value::Null, |t| Value::from(t.0)),
    }
}

fn layers(layers: &Layers) -> Value {
    match layers {
        Layers::Eq(point) => coords(point),
        Layers::None | Layers::Each | Layers::One(_) => Value::Null,
    }
}

/// One case: a model's schedule at `n` tokens.
fn case(name: &str, n: u32, lowering: &Lowering, schedule: &StageSchedule) -> Value {
    let layer_count = schedule.layers.count;
    let sites: Vec<Value> = schedule
        .sites
        .iter()
        .map(|site| match *site {
            Site::Gadget { node: at, input } => {
                json!({ "node": node(schedule, at), "input": input })
            }
            Site::Logits => json!({ "logits": true }),
        })
        .collect();
    let stages: Vec<Value> = schedule
        .stages
        .iter()
        .map(|stage| {
            let splits: usize = stage
                .claims
                .iter()
                .enumerate()
                .filter(|(_, claim)| claim.split)
                .map(|(c, claim)| {
                    // A leaf claim of a site in the layers sends its values in every layer.
                    let in_layers = stage.index == 0
                        && matches!(
                            schedule.sites.get(c),
                            Some(Site::Gadget {
                                node: NodeRef::Position(_),
                                ..
                            })
                        );
                    claim.parts.len() * if in_layers { layer_count as usize } else { 1 }
                })
                .sum();
            let instances: Vec<Value> = stage
                .instances
                .iter()
                .map(|instance| {
                    json!({
                        "relation": relation(instance.relation),
                        "node": node(schedule, instance.node),
                        "tensor": tensor(lowering, schedule, instance.kind),
                        "stacked": matches!(instance.layers, Layers::Eq(_)),
                        "vars": instance.vars,
                        "point": coords(&instance.point),
                        "layers": layers(&instance.layers),
                    })
                })
                .collect();
            json!({
                "index": stage.index,
                "rounds": stage.rounds,
                "zero_vars": stage.zero_vars,
                "splits": splits,
                "instances": instances,
            })
        })
        .collect();
    let pool: Vec<Value> = schedule
        .pool
        .iter()
        .map(|claim| {
            json!({
                "stage": claim.stage,
                "commitment": match claim.commitment {
                    CommitmentClass::Advice => "advice",
                    CommitmentClass::Weight => "weight",
                },
                "atoms": claim.atoms.len(),
                "point": coords(&claim.point),
            })
        })
        .collect();
    json!({
        "model": name,
        "n": n,
        "layers": layer_count,
        "layer_vars": schedule.layers.vars,
        "stage_count": schedule.stage_count(),
        "sites": sites,
        "stages": stages,
        "pool": pool,
    })
}

/// The schedules of the conformance models in the tree at `repo`, as the file's text.
pub(crate) fn render(repo: &Path) -> Result<String, String> {
    let mut cases = Vec::new();
    for (name, _, model) in models(repo)? {
        let public = PublicModel::of(&model);
        for n in [1, 4, 5, public.arch.max_context] {
            let lowering = lower(public, n).map_err(|e| format!("{name} at n = {n}: {e}"))?;
            let schedule =
                derive_schedule(&lowering).map_err(|e| format!("{name} at n = {n}: {e}"))?;
            cases.push(case(&name, n, &lowering, &schedule));
        }
    }
    let file = json!({
        "description": "Each conformance model's stage schedule (RFC-0010 'Stages', RFC-0037) at the lengths 1, 4, 5 and max_context, written by `cargo xtask gen-vectors` from the compiler's derivation; spec/vectors/schedule/README.md gives each field's meaning. The verifier derives its own schedule from the claim list and must give the same (INV-SCH-1).",
        "cases": cases,
    });
    let mut text = serde_json::to_string_pretty(&file).map_err(|e| e.to_string())?;
    text.push('\n');
    Ok(text)
}
