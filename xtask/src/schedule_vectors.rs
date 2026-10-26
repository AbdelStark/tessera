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
//!
//! The stage degrees, `spec/vectors/schedule/degrees.json` ([`render_degrees`], RFC-0013 'Masked
//! extensions' 'Degree', #565), hold each reduction stage's `tessera-protocol/2` degree on the
//! composed schedules `tessera-protocol/2` proves: each conformance claim list's, at `n` of 1, 4,
//! 5 and `max_context`, for the requests `none`, `prefill` and, at `n ≥ 2`, `generate`.

use std::path::Path;

use serde_json::{Value, json};
use tessera_compiler::schedule::masks::protocol2_length;
use tessera_compiler::schedule::{
    Coord, Kind, Layers, NodeRef, Relation, Request, Site, StageDegree, StageSchedule,
    derive_composed_schedule, model_natives, protocol2_stages,
};
use tessera_compiler::{Lowering, PublicModel, derive_schedule, lower};
use tessera_ir::{ClaimList, CommitmentClass};

use crate::claim_vectors::{claims, models};

/// Where the schedules are written.
pub(crate) const PATH: &str = "spec/vectors/schedule/schedule.json";

/// Where the stage degrees are written.
pub(crate) const DEGREES_PATH: &str = "spec/vectors/schedule/degrees.json";

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
        Relation::Native { zero: true, .. } => "ZeroCheck",
        Relation::Native { zero: false, .. } => "Elementwise",
        Relation::MatMulWeight => "MatMulWeight",
        Relation::MatMulAct { .. } => "MatMulAct",
        Relation::Mul(_) | Relation::BitFn(_) => "Elementwise",
        Relation::SumSquares => "SumSquares",
        Relation::Rotate => "Rotate",
        Relation::ScaleChannels => "ScaleChannels",
        Relation::Selection => "Selection",
        Relation::Equality
        | Relation::Booleanity
        | Relation::OneHotVocabulary
        | Relation::OneHotCount
        | Relation::OneHotIndex
        | Relation::InputPadding
        | Relation::BitPadding
        | Relation::Recomposition
        | Relation::AssertEqual => "ZeroCheck",
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

/// The requests of a case, as the vector names them, in its order.
const REQUESTS: [(Request, &str); 3] = [
    (Request::None, "none"),
    (Request::Prefill, "prefill"),
    (Request::Generate, "generate"),
];

/// One stage's degrees, as `degrees.json` records them.
fn stage_degrees(stage: &StageDegree) -> Value {
    let instances: Vec<Value> = stage
        .instances
        .iter()
        .map(|i| {
            json!({
                "relation": i.key.relation.name(),
                "stacked": i.key.stacked,
                "public": i.key.public,
                "constant": i.key.constant.map(|c| c.name()),
                "d_1": i.row.d_1,
                "s": i.row.s,
                "d": i.row.d,
            })
        })
        .collect();
    json!({
        "index": stage.index,
        "rounds": stage.rounds,
        "d": stage.d,
        "folds": stage.folds,
        "instances": instances,
    })
}

/// The reduction stages of `claims`' composed schedule for a sequence of `n` positions and
/// `request`, as `tessera-protocol/2` lays it out and proves it: lowered at `max(n, 3)` positions,
/// with the request's native relations (`derive_composed_schedule` with `model_natives`).
fn degrees(claims: &ClaimList, length: u32, request: Request) -> Result<Vec<StageDegree>, String> {
    let lowering = lower(PublicModel::of_claims(claims), length).map_err(|e| e.to_string())?;
    let natives = model_natives(&lowering, request).map_err(|e| e.to_string())?;
    let schedule = derive_composed_schedule(&lowering, &natives.deferred, &natives.native)
        .map_err(|e| e.to_string())?;
    let log2_k = u32::from(claims.layout.advice.log2_k);
    protocol2_stages(&lowering.graph, &schedule, log2_k).map_err(|e| e.to_string())
}

/// The stage degrees of the conformance claim lists in the tree at `repo`, as the file's text:
/// RFC-0013 'Testing Strategy''s 44 cases, the claim lists in `claim-lists.json`'s order, then
/// `n` ascending, then `none`, `prefill` and `generate`.
///
/// It checks what RFC-0013 says the file records: 10 for stage 1 and 7 for the later stages,
/// `[10, 7]` in the 22 cases of the two `servable` lists and `[10, 7, 7]` in the 22 of the
/// `minimal` and `affine` lists.
pub(crate) fn render_degrees(repo: &Path) -> Result<String, String> {
    let mut cases = Vec::new();
    let mut checked = [0usize; 2];
    for (name, _, model) in models(repo)? {
        let list = claims(&name, &model)?;
        let max_context = PublicModel::of_claims(&list).arch.max_context;
        for n in [1, 4, 5, max_context] {
            let length = protocol2_length(n);
            for (request, label) in REQUESTS {
                // RFC-0053 needs a prompt and an output.
                if request == Request::Generate && n < 2 {
                    continue;
                }
                let at = |e: String| format!("{name} at n = {n}, {label}: {e}");
                let stages = degrees(&list, length, request).map_err(at)?;
                let d: Vec<usize> = stages.iter().map(|s| s.d).collect();
                let (expected, count): (&[usize], _) = if name.ends_with("-servable") {
                    (&[10, 7], &mut checked[0])
                } else {
                    (&[10, 7, 7], &mut checked[1])
                };
                if d != expected {
                    return Err(at(format!(
                        "the stage degrees are {d:?}, where RFC-0013 records {expected:?}"
                    )));
                }
                *count += 1;
                cases.push(json!({
                    "model": name,
                    "n": n,
                    "length": length,
                    "request": label,
                    "stages": stages.iter().map(stage_degrees).collect::<Vec<_>>(),
                }));
            }
        }
    }
    if checked != [22, 22] {
        return Err(format!(
            "RFC-0013 records 22 servable and 22 minimal and affine cases, not {checked:?}"
        ));
    }
    let file = json!({
        "description": "Each reduction stage's tessera-protocol/2 round degree (RFC-0013 'Masked extensions' 'Degree') on the composed schedule tessera-protocol/2 proves, for each conformance claim list at n of 1, 4, 5 and max_context and the requests none, prefill and, at n >= 2, generate: written by `cargo xtask gen-vectors` from the compiler's schedules and RFC-0013's table; spec/vectors/schedule/README.md gives each field's meaning. The verifier derives each schedule and each instance's row with code of its own and must give the same.",
        "cases": cases,
    });
    let mut text = serde_json::to_string_pretty(&file).map_err(|e| e.to_string())?;
    text.push('\n');
    Ok(text)
}
