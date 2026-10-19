// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The lowering against RFC-0044's independent oracle (RFC-0044 'Testing Strategy'):
//! `spec/vectors/lowering/lowering.json`, which `generate.py` writes from the RFC alone.
//!
//! For every case of the vectors, a conformance model of `spec/vectors/models/`, or a variant of
//! one that takes a form the models miss, lowers at the case's length, and its graph must equal
//! the oracle's, id for id:
//!
//! - the graph inputs, with their roles, shapes, kinds, types and ranges;
//! - the nodes, with their operations, parameters, inputs, and their outputs' ids, shapes and
//!   kinds;
//! - the constants, the shift-class sets, the weight list, the tables and the checkpoints.
//!
//! Inputs and nodes are compared in creation order, so that the first difference reported is the
//! one every later id follows from. Each case that differs reports its first difference: the
//! case, the input, node or entry, the field, and both values.

// The per-case summary goes to stderr, shown with `--nocapture`.
#![allow(clippy::print_stderr)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use tessera_codec::{DecodeLimits, Rational, envelope, schema};
use tessera_compiler::{Lowering, PublicModel, lower};
use tessera_ir::{
    ArchSpec, Broadcast, CommitmentClass, Graph, IntRange, IntType, IntegerModel, NodeKind, Op,
    PublicQuantSpec, QuantPoint, Site, TensorId, WeightRef,
};
use tessera_tables::{TableKind, TableSpec};

fn vectors() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors")
}

fn read_json(path: &Path) -> Value {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Each conformance model's artifact, decoded through the public API, by file name.
fn models() -> BTreeMap<String, IntegerModel> {
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

fn int(value: &Value) -> i64 {
    value
        .as_i64()
        .unwrap_or_else(|| panic!("an integer: {value}"))
}

fn uint(value: &Value) -> u32 {
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
fn edited(model: &IntegerModel, variant: &Value) -> (ArchSpec, PublicQuantSpec) {
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

// ----- The compiler's graph in the vectors' notation (README.md) ------------------------------------

fn kind(kind: NodeKind) -> String {
    match kind {
        NodeKind::Virtual => "v".to_owned(),
        NodeKind::Committed {
            commitment: CommitmentClass::Advice,
        } => "A".to_owned(),
        NodeKind::Committed {
            commitment: CommitmentClass::Weight,
        } => "W".to_owned(),
        NodeKind::Public => "P".to_owned(),
        NodeKind::LookupOutput { table } => format!("L({})", table.0),
    }
}

fn int_type(ty: IntType) -> String {
    format!("{}{}", if ty.signed { "i" } else { "u" }, ty.bits)
}

fn bound(x: i128) -> Value {
    i64::try_from(x).map_or_else(|_| Value::from(x.to_string()), Value::from)
}

fn weight_ref(weight: WeightRef) -> Value {
    json!([format!("{:?}", weight.ty), weight.layer])
}

fn operation(op: &Op) -> (&'static str, Value) {
    match op {
        Op::MatMulWeight { weight } => ("MatMulWeight", json!({ "weight": weight_ref(*weight) })),
        Op::Add => ("Add", json!({})),
        Op::Sub => ("Sub", json!({})),
        Op::ConstMul { c } => ("ConstMul", json!({ "c": c.0 })),
        Op::ConstAdd { c } => ("ConstAdd", json!({ "c": c.0 })),
        Op::Rotate { rope } => ("Rotate", json!({ "rope": rope.0 })),
        Op::Sum { axis } => ("Sum", json!({ "axis": axis })),
        Op::Reshape => ("Reshape", json!({})),
        Op::Slice { axis, start, len } => {
            ("Slice", json!({ "axis": axis, "start": start, "len": len }))
        }
        Op::GatherPublic { index } => ("GatherPublic", json!({ "index": index.0 })),
        Op::Transpose { perm } => ("Transpose", json!({ "perm": perm })),
        Op::Expand { axis, len } => ("Expand", json!({ "axis": axis, "len": len })),
        Op::MatMulAct { transpose_rhs } => ("MatMulAct", json!({ "transpose_rhs": transpose_rhs })),
        Op::Mul { broadcast } => {
            let broadcast = match broadcast {
                Broadcast::Trailing => "Trailing",
                Broadcast::Leading => "Leading",
            };
            ("Mul", json!({ "broadcast": broadcast }))
        }
        Op::SumSquares { axis } => ("SumSquares", json!({ "axis": axis })),
        Op::OneHot { vocab } => ("OneHot", json!({ "vocab": vocab })),
        Op::Requant { shift, lo, hi } => {
            ("Requant", json!({ "shift": shift.0, "lo": lo, "hi": hi }))
        }
        Op::TableMap { table } => ("TableMap", json!({ "table": table.0 })),
        Op::SoftmaxRow { exp_table, sigma } => (
            "SoftmaxRow",
            json!({ "exp_table": exp_table.0, "sigma": sigma }),
        ),
        Op::RsqrtRow {
            kappa,
            eps_int,
            dim,
        } => (
            "RsqrtRow",
            json!({ "kappa": kappa, "eps_int": eps_int, "dim": dim }),
        ),
        Op::Argmax => ("Argmax", json!({})),
        Op::InverseCdf { prf } => ("InverseCdf", json!({ "prf": prf.0 })),
        Op::Bitwise { table } => ("Bitwise", json!({ "table": table.0 })),
    }
}

fn table_kind(kind: &TableKind) -> Value {
    match kind {
        TableKind::Rope {
            theta,
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max,
            bits,
            max_context,
            d_head,
        } => json!({ "Rope": {
            "theta": theta, "factor": factor, "low_freq_factor": low_freq_factor,
            "high_freq_factor": high_freq_factor, "original_max": original_max, "bits": bits,
            "max_context": max_context, "d_head": d_head,
        }}),
        TableKind::Exp {
            key_bits,
            sigma,
            scale,
        } => json!({ "Exp": {
            "key_bits": key_bits, "sigma": sigma, "scale": [scale.num(), scale.den()],
        }}),
        TableKind::Silu {
            b,
            r_by_class,
            in_exp,
            out_exp,
        } => json!({ "Silu": {
            "b": b, "r_by_class": r_by_class, "in_exp": in_exp, "out_exp": out_exp,
        }}),
        other => json!({ "other": format!("{other:?}") }),
    }
}

/// Each graph input's role: the tokens, the positions, a weight, a class vector or a multiplier
/// vector, as `Lowering` names them.
fn roles(lowering: &Lowering) -> BTreeMap<TensorId, String> {
    let mut roles = BTreeMap::new();
    roles.insert(lowering.tokens, "tokens".to_owned());
    roles.insert(lowering.positions, "positions".to_owned());
    let named = [
        ("weight", &lowering.graph.weights),
        ("classes", &lowering.classes),
        ("multipliers", &lowering.multipliers),
    ];
    for (role, list) in named {
        for (weight, tensor) in list {
            roles.insert(*tensor, format!("{role} {:?} {}", weight.ty, weight.layer));
        }
    }
    roles
}

/// The graph's inputs and nodes as the vectors' records, in creation order: each input at its
/// tensor's id, each node at its first output's.
fn records(lowering: &Lowering) -> Vec<Value> {
    let graph: &Graph = &lowering.graph;
    let roles = roles(lowering);
    let mut produced = vec![false; graph.tensors.len()];
    let mut by_first_tensor: BTreeMap<u32, Value> = BTreeMap::new();
    for node in &graph.nodes {
        let (name, params) = operation(&node.op);
        let outputs: Vec<Value> = node
            .outputs
            .iter()
            .map(|t| {
                if let Some(slot) = produced.get_mut(t.0 as usize) {
                    *slot = true;
                }
                let spec = &graph.tensors[t.0 as usize];
                json!([t.0, spec.shape, kind(spec.kind)])
            })
            .collect();
        let inputs: Vec<u32> = node.inputs.iter().map(|t| t.0).collect();
        let first = node.outputs.first().map_or(u32::MAX, |t| t.0);
        by_first_tensor.insert(
            first,
            json!(["node", node.id.0, name, params, inputs, outputs]),
        );
    }
    for (i, spec) in graph.tensors.iter().enumerate() {
        if produced[i] {
            continue;
        }
        let IntRange { lo, hi } = spec.range;
        let role = roles
            .get(&spec.id)
            .cloned()
            .unwrap_or_else(|| "?".to_owned());
        by_first_tensor.insert(
            spec.id.0,
            json!([
                "input",
                spec.id.0,
                role,
                spec.shape,
                kind(spec.kind),
                int_type(spec.ty),
                bound(lo),
                bound(hi)
            ]),
        );
    }
    by_first_tensor.into_values().collect()
}

/// The oracle's inputs and nodes as the same records, in the same order, with each node's step.
fn expected_records(case: &Value) -> Vec<(Value, String)> {
    let mut by_first_tensor: BTreeMap<i64, (Value, String)> = BTreeMap::new();
    for input in case["inputs"].as_array().expect("inputs") {
        let mut record = vec![json!("input")];
        record.extend(input.as_array().expect("an input").iter().cloned());
        by_first_tensor.insert(int(&input[0]), (Value::from(record), String::new()));
    }
    for node in case["nodes"].as_array().expect("nodes") {
        let fields = node.as_array().expect("a node");
        let mut record = vec![json!("node")];
        record.extend(fields[..5].iter().cloned());
        let first = fields[4]
            .as_array()
            .and_then(|outputs| outputs.first())
            .map_or(i64::MAX, |output| int(&output[0]));
        let at = fields[5].as_str().unwrap_or_default().to_owned();
        by_first_tensor.insert(first, (Value::from(record), at));
    }
    by_first_tensor.into_values().collect()
}

const INPUT_FIELDS: [&str; 8] = ["", "id", "role", "shape", "kind", "type", "lo", "hi"];
const NODE_FIELDS: [&str; 6] = ["", "id", "operation", "parameters", "inputs", "outputs"];

/// What a record is, for a report: `input 22 (classes Q 0)` or `node 16 (layer 0 ...)`.
fn describe(record: &Value, at: &str) -> String {
    match record[0].as_str() {
        Some("input") => format!(
            "input {} ({})",
            record[1],
            record[2].as_str().unwrap_or("?")
        ),
        _ if at.is_empty() => format!("node {}", record[1]),
        _ => format!("node {} ({at})", record[1]),
    }
}

/// The first field in which two records differ, with both values.
fn field_difference(expected: &Value, got: &Value) -> String {
    if expected[0] != got[0] {
        return format!(
            "the oracle has an {} where the compiler has a {}: {got}",
            expected[0], got[0]
        );
    }
    let names: &[&str] = if expected[0] == "input" {
        &INPUT_FIELDS
    } else {
        &NODE_FIELDS
    };
    for (i, name) in names.iter().enumerate().skip(1) {
        let (e, g) = (&expected[i], &got[i]);
        if e == g {
            continue;
        }
        if *name == "outputs" {
            let (e, g) = (e.as_array(), g.as_array());
            if let (Some(e), Some(g)) = (e, g) {
                for (k, (eo, go)) in e.iter().zip(g).enumerate() {
                    for (j, part) in ["id", "shape", "kind"].iter().enumerate() {
                        if eo[j] != go[j] {
                            return format!(
                                "output {k}'s {part}: the oracle's {}, the compiler's {}",
                                eo[j], go[j]
                            );
                        }
                    }
                }
                return format!(
                    "outputs: the oracle's {} outputs, the compiler's {}",
                    e.len(),
                    g.len()
                );
            }
        }
        return format!("{name}: the oracle's {e}, the compiler's {g}");
    }
    "no field".to_owned()
}

/// The first difference between the oracle's case and the compiler's lowering, if any.
fn first_difference(case: &Value, lowering: &Lowering) -> Option<String> {
    let graph = &lowering.graph;
    for (i, spec) in graph.tensors.iter().enumerate() {
        if spec.id.0 as usize != i {
            return Some(format!("tensor {i} has the id {}", spec.id.0));
        }
    }
    // Inputs and nodes, in creation order.
    let expected = expected_records(case);
    let got = records(lowering);
    for (k, (e, at)) in expected.iter().enumerate() {
        let Some(g) = got.get(k) else {
            return Some(format!(
                "{}: the compiler's graph ends before it",
                describe(e, at)
            ));
        };
        if e != g {
            return Some(format!("{}: {}", describe(e, at), field_difference(e, g)));
        }
    }
    if let Some(g) = got.get(expected.len()) {
        return Some(format!(
            "the compiler has more after the oracle's last: {g}"
        ));
    }
    let count = int(&case["tensors"]);
    if i64::try_from(graph.tensors.len()) != Ok(count) {
        return Some(format!(
            "tensors: the oracle's {count}, the compiler's {}",
            graph.tensors.len()
        ));
    }
    // The lists by index: constants, shift-class sets, weights, tables, checkpoints.
    let lists: [(&str, Vec<Value>); 5] = [
        (
            "constant",
            graph.constants.iter().map(|c| json!(c)).collect(),
        ),
        (
            "shift-class set",
            graph.shift_classes.iter().map(|s| json!(s)).collect(),
        ),
        (
            "weight",
            graph
                .weights
                .iter()
                .map(|(w, t)| json!([format!("{:?}", w.ty), w.layer, t.0]))
                .collect(),
        ),
        (
            "table",
            lowering
                .tables
                .iter()
                .map(|t: &TableSpec| json!([t.id.0, table_kind(&t.kind), t.log2_rows]))
                .collect(),
        ),
        (
            "checkpoint",
            lowering.checkpoints.iter().map(|t| json!(t.0)).collect(),
        ),
    ];
    let fields = [
        "constants",
        "shift_classes",
        "weights",
        "tables",
        "checkpoints",
    ];
    for ((what, got), field) in lists.iter().zip(fields) {
        let expected = case[field].as_array().expect("a list");
        for (i, e) in expected.iter().enumerate() {
            match got.get(i) {
                Some(g) if g == e => {}
                Some(g) => {
                    return Some(format!("{what} {i}: the oracle's {e}, the compiler's {g}"));
                }
                None => {
                    return Some(format!(
                        "{what} {i}: the oracle's {e}, the compiler has none"
                    ));
                }
            }
        }
        if got.len() != expected.len() {
            return Some(format!(
                "{field}: the oracle's {}, the compiler's {}",
                expected.len(),
                got.len()
            ));
        }
    }
    None
}

/// RFC-0044 'Testing Strategy', 'The vectors': for each conformance model and `n` in {1, 4, 5,
/// `max_context`}, and for the oracle's variants, the compiler's graph equals the oracle's.
#[test]
fn the_lowering_is_the_oracles_node_for_node() {
    let models = models();
    let document = read_json(&vectors().join("lowering/lowering.json"));
    let range_document = read_json(&vectors().join("ranges/ranges.json"));
    let cases = document["cases"].as_array().expect("cases");
    // Every conformance model at every length the RFC names is a case.
    for (file, model) in &models {
        for n in [1, 4, 5, model.arch.max_context] {
            assert!(
                cases.iter().any(|c| c["file"] == file.as_str()
                    && c.get("variant").is_none()
                    && c["n"] == n),
                "the vectors miss {file} at n = {n}"
            );
        }
    }
    let mut failures = Vec::new();
    let mut nodes = 0;
    for case in cases {
        let name = case["model"].as_str().expect("a model");
        let n = uint(&case["n"]);
        let model = &models[case["file"].as_str().expect("a file")];
        let edits = case.get("variant").map(|variant| edited(model, variant));
        let public = match &edits {
            Some((arch, quant)) => PublicModel {
                arch,
                quant,
                ..PublicModel::of(model)
            },
            None => PublicModel::of(model),
        };
        let lowering = match lower(public, n) {
            Ok(lowering) => lowering,
            Err(err) => {
                failures.push(format!("{name}, n = {n}: the compiler refuses it: {err:?}"));
                continue;
            }
        };
        // The same pre-implementation Python intervals are checked by the independent verifier.
        // Both sides matching every tensor establishes equality without sharing range code (SP-6).
        let expected_ranges = range_document["cases"]
            .as_array()
            .expect("range cases")
            .iter()
            .find(|c| c["model"] == case["model"] && c["n"] == case["n"])
            .expect("range case for each lowering");
        let expected: Vec<Option<(i128, i128)>> = expected_ranges["ranges"]
            .as_array()
            .expect("ranges")
            .iter()
            .map(|r| {
                if r.is_null() {
                    None
                } else {
                    Some((i128::from(int(&r[0])), i128::from(int(&r[1]))))
                }
            })
            .collect();
        let analyzed =
            tessera_compiler::analyze(&lowering.graph, &lowering.tables).expect("compiler ranges");
        let found: Vec<_> = analyzed
            .all()
            .iter()
            .map(|r| r.bounded().map(|r| (r.lo, r.hi)))
            .collect();
        assert_eq!(
            found, expected,
            "{name}, n={n}: public ranges differ from the independent oracle"
        );
        match first_difference(case, &lowering) {
            Some(difference) => failures.push(format!("{name}, n = {n}: {difference}")),
            None => {
                nodes += lowering.graph.nodes.len();
                eprintln!(
                    "ok  {name}, n = {n}: {} tensors, {} nodes, {} tables",
                    lowering.graph.tensors.len(),
                    lowering.graph.nodes.len(),
                    lowering.tables.len()
                );
            }
        }
    }
    eprintln!(
        "{} of {} cases match, {nodes} nodes in all",
        cases.len() - failures.len(),
        cases.len()
    );
    assert!(
        failures.is_empty(),
        "{} of {} cases differ from RFC-0044's oracle, each at its first difference:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}
