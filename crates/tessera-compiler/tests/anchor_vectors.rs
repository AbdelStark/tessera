// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The anchor's SHA-256 circuit against the independent oracles of `spec/vectors/anchor-graphs/`
//! and `spec/vectors/ranges/anchor.json` (#150, slice S1), written from RFC-0066, RFC-0072 and
//! RFC-0006 by an author who read neither implementation.
//!
//! - **The graph.** For each case of `sha256-graphs.json`, `tessera_compiler::anchor`'s builder
//!   (`sha256_inputs`, `sha256`, `sha256_digest_input`, `sha256_bind`) at the case's layout gives
//!   the oracle's graph node for node: the inputs with their roles, shapes, kinds, types and
//!   ranges, in creation order; each node's operation, parameters, inputs and outputs' ids, shapes
//!   and kinds; the constants; the chaining words of every block; the public tensors
//!   (`schedule::public_tensors`); and the per-block counts of RFC-0072's sizes table, the
//!   zero-checks those of stage 1 of the compiler's schedule. A case that differs reports its
//!   first differing input and node, with the oracle's `at` label and both sides.
//! - **The inputs from bytes.** For each of `sha256-inputs.json`'s 81 sets, the compiler's
//!   `sha256_pad`, `Sha256Layout::public_bits` and `place`, `sha256_bits` of `H^{(0)}` and of the
//!   digests, and `SHA256_MASKS` give the oracle's inputs; the circuit, run on them by this test's
//!   own interpreter (`common::evaluate`), gives FIPS 180-4's chaining words after every block and
//!   passes its digest binding.
//! - **Ranges.** `analyze` gives the oracle's interval of every tensor of every circuit, and
//!   `settle` writes those intervals into the graph; each of the 18 small graphs of
//!   `ranges/anchor.json` gets the oracle's verdict, and its ranges where the analysis admits it.

mod common;

use std::collections::BTreeMap;

use common::evaluate;
use common::public_vectors::{int, read_json, uint, vectors};
use serde_json::{Value, json};
use tessera_compiler::anchor::{
    SHA256_IV, SHA256_MASKS, Sha256Layout, sha256, sha256_bind, sha256_bits, sha256_blocks,
    sha256_digest_input, sha256_inputs, sha256_pad,
};
use tessera_compiler::schedule::{
    NodeRef, Relation, Source, derive_graph_schedule, public_tensors,
};
use tessera_compiler::{CompileError, Ranges, analyze, analyze_fragment, settle};
use tessera_ir::{
    BoolFn, Broadcast, CommitmentClass, ConstRef, Graph, GraphBuilder, IntRange, IntType, NodeKind,
    Op, TensorDecl, TensorId,
};

// ----- The compiler's circuit ------------------------------------------------------------------

/// A SHA-256 circuit as the compiler builds it, its last block's chaining words bound to a digest
/// input declared after the circuit's nodes (RFC-0066 'The graph').
struct Circuit {
    graph: Graph,
    /// The inputs in `sha256_inputs`'s order, then the digest.
    inputs: Vec<TensorId>,
    /// The chaining words after every block.
    chaining: Vec<[TensorId; 8]>,
    layout: Sha256Layout,
}

fn circuit(blocks: u32, messages: u32, committed: Vec<u32>) -> Circuit {
    let layout = Sha256Layout {
        blocks,
        messages,
        committed,
    };
    let mut g = GraphBuilder::new();
    let mut inputs: Vec<TensorId> = sha256_inputs(&layout)
        .into_iter()
        .map(|d| g.input(d).expect("an input"))
        .collect();
    let sha = sha256(&mut g, &layout, &inputs).expect("the circuit");
    let digest = g.input(sha256_digest_input(messages)).expect("the digest");
    sha256_bind(
        &mut g,
        sha.chaining.last().expect("a block"),
        digest,
        messages,
    )
    .expect("the binding");
    inputs.push(digest);
    Circuit {
        graph: g.finish(),
        inputs,
        chaining: sha.chaining,
        layout,
    }
}

/// The circuit of a case of either file: its `N`, `M` and committed offsets.
fn circuit_of(case: &Value) -> Circuit {
    let offsets = case["committed_offsets"]
        .as_array()
        .expect("offsets")
        .iter()
        .map(uint)
        .collect();
    circuit(uint(&case["N"]), uint(&case["M"]), offsets)
}

/// Each input's role, RFC-0066's names: `sha256_inputs`'s order, then the digest.
fn roles(c: &Circuit) -> BTreeMap<TensorId, &'static str> {
    let names: &[&str] = if c.layout.committed.is_empty() {
        &["message", "start", "masks", "digest"]
    } else {
        &["message", "start", "masks", "committed", "place", "digest"]
    };
    assert_eq!(names.len(), c.inputs.len(), "the circuit's inputs");
    c.inputs
        .iter()
        .copied()
        .zip(names.iter().copied())
        .collect()
}

// ----- The compiler's graph in the vectors' notation (anchor-graphs/README.md) ----------------

fn kind(kind: NodeKind) -> &'static str {
    match kind {
        NodeKind::Virtual => "v",
        NodeKind::Committed {
            commitment: CommitmentClass::Advice,
        } => "A",
        NodeKind::Committed {
            commitment: CommitmentClass::Weight,
        } => "W",
        NodeKind::Public => "P",
        NodeKind::LookupOutput { .. } => "L",
    }
}

fn int_type(ty: IntType) -> String {
    format!("{}{}", if ty.signed { "i" } else { "u" }, ty.bits)
}

/// An operation's name and parameters, as RFC-0006 names them; operations outside the circuit's
/// vocabulary by their debug form, which no case holds.
fn operation(op: &Op) -> (String, Value) {
    let (name, params) = match op {
        Op::GatherPublic { index } => ("GatherPublic", json!({ "index": index.0 })),
        Op::Slice { axis, start, len } => {
            ("Slice", json!({ "axis": axis, "start": start, "len": len }))
        }
        Op::Reshape => ("Reshape", json!({})),
        Op::Transpose { perm } => ("Transpose", json!({ "perm": perm })),
        Op::Add => ("Add", json!({})),
        Op::Mul { broadcast } => (
            "Mul",
            json!({ "broadcast": match broadcast {
                Broadcast::Trailing => "Trailing",
                Broadcast::Leading => "Leading",
            } }),
        ),
        Op::ConstMul { c } => ("ConstMul", json!({ "c": c.0 })),
        Op::ConstAdd { c } => ("ConstAdd", json!({ "c": c.0 })),
        Op::BitDecompose { low, high } => ("BitDecompose", json!({ "low": low, "high": high })),
        Op::BitFn { f } => ("BitFn", json!({ "f": format!("{f:?}") })),
        Op::AssertEqual => ("AssertEqual", json!({})),
        other => return (format!("{other:?}"), json!({})),
    };
    (name.to_owned(), params)
}

/// The graph's inputs as the vectors' records, by id: `[id, role, shape, kind, type, lo, hi]`.
fn input_records(c: &Circuit) -> Vec<Value> {
    let roles = roles(c);
    let mut produced = vec![false; c.graph.tensors.len()];
    for node in &c.graph.nodes {
        for t in &node.outputs {
            produced[t.0 as usize] = true;
        }
    }
    c.graph
        .tensors
        .iter()
        .filter(|spec| !produced[spec.id.0 as usize])
        .map(|spec| {
            json!([
                spec.id.0,
                roles.get(&spec.id).copied().unwrap_or("?"),
                spec.shape,
                kind(spec.kind),
                int_type(spec.ty),
                i64::try_from(spec.range.lo).expect("a bit or a byte"),
                i64::try_from(spec.range.hi).expect("a bit or a byte"),
            ])
        })
        .collect()
}

/// The graph's nodes as the vectors' records, in node order: `[id, operation, parameters,
/// inputs, outputs]`, each output `[id, shape, kind]`.
fn node_records(graph: &Graph) -> Vec<Value> {
    graph
        .nodes
        .iter()
        .map(|node| {
            let (name, params) = operation(&node.op);
            let inputs: Vec<u32> = node.inputs.iter().map(|t| t.0).collect();
            let outputs: Vec<Value> = node
                .outputs
                .iter()
                .map(|t| {
                    let spec = &graph.tensors[t.0 as usize];
                    json!([t.0, spec.shape, kind(spec.kind)])
                })
                .collect();
            json!([node.id.0, name, params, inputs, outputs])
        })
        .collect()
}

const INPUT_FIELDS: [&str; 7] = ["id", "role", "shape", "kind", "type", "lo", "hi"];
const NODE_FIELDS: [&str; 5] = ["id", "operation", "parameters", "inputs", "outputs"];

/// The first field in which two records differ, with both values.
fn field_difference(names: &[&str], expected: &Value, got: &Value) -> String {
    for (i, name) in names.iter().enumerate() {
        if expected[i] != got[i] {
            return format!(
                "{name}: the oracle's {}, the compiler's {}",
                expected[i], got[i]
            );
        }
    }
    "no field".to_owned()
}

/// The first record of `expected` that `got` does not hold at its place, described by `what`.
fn first_record_difference(
    what: &str,
    names: &[&str],
    expected: &[Value],
    got: &[Value],
    label: impl Fn(usize) -> String,
) -> Option<String> {
    for (k, e) in expected.iter().enumerate() {
        let e = Value::from(e.as_array().expect("a record")[..names.len()].to_vec());
        match got.get(k) {
            None => {
                return Some(format!(
                    "{what} {k}{}: the compiler's graph has none",
                    label(k)
                ));
            }
            Some(g) if *g != e => {
                return Some(format!(
                    "{what} {k}{}: {}\n      oracle:   {e}\n      compiler: {g}",
                    label(k),
                    field_difference(names, &e, g)
                ));
            }
            Some(_) => {}
        }
    }
    (got.len() > expected.len()).then(|| {
        format!(
            "{what}s: the oracle's {}, the compiler's {}; its next: {}",
            expected.len(),
            got.len(),
            got[expected.len()]
        )
    })
}

// ----- RFC-0072's counts -----------------------------------------------------------------------

const COUNT_KEYS: [&str; 10] = [
    "zero_checks",
    "input_padding",
    "booleanity",
    "bit_padding",
    "recomposition",
    "assert_equal",
    "selections",
    "bitfn",
    "mul",
    "nodes",
];

/// The counts of `sha256-graphs.json`, from the compiler's graph and its schedule: per block, for
/// the graph's own nodes outside the blocks, and in all.
///
/// - **A block's nodes** are RFC-0066's step 4 of one block: those after the eight initial words'
///   `Slice`s of `start`, or after the previous block's, up to the `Transpose` that gives its
///   chaining word `h`.
/// - **Zero-checks** are stage 1's instances of the schedule that prove one, by relation, each at
///   the node it states; selections, `BitFn`s and `Mul`s count those whose output is not public.
fn counts(c: &Circuit, public: &[bool]) -> Value {
    let graph = &c.graph;
    let start = c.inputs[1];
    let initial = graph
        .nodes
        .iter()
        .filter(|n| n.inputs.contains(&start))
        .map(|n| n.id.0)
        .max()
        .expect("the initial words");
    let ends: Vec<u32> = c
        .chaining
        .iter()
        .map(|words| graph.producer(words[7]).expect("a chaining word's node").0)
        .collect();
    let block_of = |node: u32| -> Option<usize> {
        if node <= initial {
            return None;
        }
        ends.iter().position(|&end| node <= end)
    };
    let mut rows: BTreeMap<Option<usize>, [u64; 10]> = BTreeMap::new();
    let slot = |key: &str| COUNT_KEYS.iter().position(|k| *k == key).expect("a key");
    for node in &graph.nodes {
        let row = rows.entry(block_of(node.id.0)).or_insert([0; 10]);
        row[slot("nodes")] += 1;
        let hidden = !node.outputs.is_empty() && node.outputs.iter().all(|t| public[t.0 as usize]);
        if hidden {
            continue;
        }
        match node.op {
            Op::Slice { .. } | Op::GatherPublic { .. } => row[slot("selections")] += 1,
            Op::BitFn { .. } => row[slot("bitfn")] += 1,
            Op::Mul { .. } => row[slot("mul")] += 1,
            _ => {}
        }
    }
    let schedule = derive_graph_schedule(graph, &[], &[], None).expect("a schedule");
    for instance in &schedule.stages[1].instances {
        if instance.source != Source::ZeroCheck {
            continue;
        }
        let NodeRef::Node(node) = instance.node else {
            panic!("a graph without layers has no template position");
        };
        let key = match instance.relation {
            Relation::InputPadding => "input_padding",
            Relation::Booleanity => "booleanity",
            Relation::BitPadding => "bit_padding",
            Relation::Recomposition => "recomposition",
            Relation::AssertEqual => "assert_equal",
            other => panic!("node {}: a zero-check of {other:?}", node.0),
        };
        let row = rows.entry(block_of(node.0)).or_insert([0; 10]);
        row[slot(key)] += 1;
        row[slot("zero_checks")] += 1;
    }
    let as_json = |row: &[u64; 10]| -> Value {
        COUNT_KEYS
            .iter()
            .zip(row)
            .map(|(k, v)| ((*k).to_owned(), json!(v)))
            .collect::<serde_json::Map<_, _>>()
            .into()
    };
    let mut total = [0u64; 10];
    for row in rows.values() {
        for (t, v) in total.iter_mut().zip(row) {
            *t += v;
        }
    }
    let blocks: Vec<Value> = (0..c.chaining.len())
        .map(|b| as_json(rows.get(&Some(b)).unwrap_or(&[0; 10])))
        .collect();
    json!({
        "blocks": blocks,
        "graph": as_json(rows.get(&None).unwrap_or(&[0; 10])),
        "total": as_json(&total),
    })
}

// ----- The comparison --------------------------------------------------------------------------

/// Every difference between the oracle's case and the compiler's circuit, each at its first place.
fn differences(case: &Value, c: &Circuit) -> Vec<String> {
    let graph = &c.graph;
    let mut out = Vec::new();
    for (i, spec) in graph.tensors.iter().enumerate() {
        if spec.id.0 as usize != i {
            out.push(format!("tensor {i} has the id {}", spec.id.0));
            return out;
        }
    }
    let tensors = int(&case["tensors"]);
    if i64::try_from(graph.tensors.len()) != Ok(tensors) {
        out.push(format!(
            "tensors: the oracle's {tensors}, the compiler's {}",
            graph.tensors.len()
        ));
    }
    // The inputs, by id; then the nodes, in node order.
    let expected = case["inputs"].as_array().expect("inputs");
    let got = input_records(c);
    let label = |k: usize| {
        expected
            .get(k)
            .map(|e| format!(" (id {}, {})", e[0], e[1].as_str().unwrap_or("?")))
            .unwrap_or_default()
    };
    out.extend(first_record_difference(
        "input",
        &INPUT_FIELDS,
        expected,
        &got,
        label,
    ));
    let expected = case["nodes"].as_array().expect("nodes");
    let got = node_records(graph);
    let label = |k: usize| {
        expected
            .get(k)
            .and_then(|e| e[5].as_str())
            .map(|at| format!(" ({at})"))
            .unwrap_or_default()
    };
    out.extend(first_record_difference(
        "node",
        &NODE_FIELDS,
        expected,
        &got,
        label,
    ));
    // The constants, by index.
    let expected = case["constants"].as_array().expect("constants");
    let got: Vec<Value> = graph.constants.iter().map(|c| json!(c)).collect();
    if let Some(k) = (0..expected.len().max(got.len())).find(|&k| expected.get(k) != got.get(k)) {
        out.push(format!(
            "constant {k}: the oracle's {}, the compiler's {}",
            expected.get(k).map_or("none".to_owned(), Value::to_string),
            got.get(k).map_or("none".to_owned(), Value::to_string)
        ));
    }
    // The chaining words of every block.
    let got = json!(
        c.chaining
            .iter()
            .map(|words| words.iter().map(|t| t.0).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    );
    if case["chaining"] != got {
        out.push(format!(
            "chaining: the oracle's {}, the compiler's {got}",
            case["chaining"]
        ));
    }
    // The public tensors.
    let public = public_tensors(graph);
    let got: Vec<i64> = (0i64..)
        .zip(&public)
        .filter(|(_, p)| **p)
        .map(|(t, _)| t)
        .collect();
    let expected: Vec<i64> = case["public"]
        .as_array()
        .expect("public")
        .iter()
        .map(int)
        .collect();
    if got != expected {
        let only = |a: &[i64], b: &[i64]| -> Vec<i64> {
            a.iter()
                .filter(|t| !b.contains(t))
                .take(8)
                .copied()
                .collect()
        };
        out.push(format!(
            "public: {} tensors in the oracle's, {} in the compiler's; the oracle's alone {:?}, \
             the compiler's alone {:?}",
            expected.len(),
            got.len(),
            only(&expected, &got),
            only(&got, &expected)
        ));
    }
    // RFC-0072's counts.
    let got = counts(c, &public);
    for part in ["blocks", "graph", "total"] {
        if case["counts"][part] != got[part] {
            out.push(format!(
                "counts, {part}: the oracle's {}, the compiler's {}",
                case["counts"][part], got[part]
            ));
        }
    }
    out
}

fn graphs_document() -> Value {
    read_json(&vectors().join("anchor-graphs/sha256-graphs.json"))
}

/// RFC-0066 'The SHA-256 circuit': the compiler's graph is the independent oracle's, node for
/// node, for each parameter set of `sha256-graphs.json`, and so are its public tensors and
/// RFC-0072's counts.
#[test]
fn the_sha256_graph_is_the_oracles_node_for_node() {
    let document = graphs_document();
    let k: Vec<String> = document["round_constants"]
        .as_array()
        .expect("K")
        .iter()
        .map(|v| v.as_str().expect("hex").to_owned())
        .collect();
    let h0: Vec<String> = document["initial_value"]
        .as_array()
        .expect("H0")
        .iter()
        .map(|v| v.as_str().expect("hex").to_owned())
        .collect();
    assert_eq!(
        k,
        tessera_compiler::anchor::SHA256_K
            .iter()
            .map(|k| format!("{k:08x}"))
            .collect::<Vec<_>>(),
        "FIPS 180-4's K_t"
    );
    assert_eq!(
        h0,
        SHA256_IV
            .iter()
            .map(|h| format!("{h:08x}"))
            .collect::<Vec<_>>(),
        "FIPS 180-4's H^(0)"
    );
    let cases = document["cases"].as_array().expect("cases");
    assert_eq!(cases.len(), 5, "five parameter sets");
    let mut failures = Vec::new();
    for case in cases {
        let name = case["name"].as_str().expect("a name");
        let c = circuit_of(case);
        let found = differences(case, &c);
        if !found.is_empty() {
            failures.push(format!("{name}:\n  {}", found.join("\n  ")));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} cases differ from RFC-0066's oracle:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

// ----- The inputs from bytes -------------------------------------------------------------------

fn hex(s: &str) -> Vec<u8> {
    assert!(s.len().is_multiple_of(2), "hex of whole bytes");
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex"))
        .collect()
}

/// A tensor written as a string of its entries, `0` or `1` each, row-major.
fn bits_of(s: &str) -> Vec<i64> {
    s.bytes()
        .map(|b| match b {
            b'0' => 0,
            b'1' => 1,
            other => panic!("a bit, not {other}"),
        })
        .collect()
}

/// The first entry at which two flat tensors differ, for a report.
fn first_entry(expected: &[i64], got: &[i64]) -> String {
    match expected.iter().zip(got).position(|(e, g)| e != g) {
        Some(i) => format!(
            "entry {i}: the oracle's {}, the compiler's {}",
            expected[i], got[i]
        ),
        None => format!(
            "lengths: the oracle's {}, the compiler's {}",
            expected.len(),
            got.len()
        ),
    }
}

/// The word of column `column` of a `[32, M]` bit tensor, row `i` of weight `2^i`.
fn word(bits: &[i128], m: usize, column: usize) -> u32 {
    (0..32).fold(0u32, |w, i| {
        w | (u32::try_from(bits[i * m + column]).expect("a bit") << i)
    })
}

/// RFC-0066 'The message' and 'The output': the circuit's public inputs computed from bytes by the
/// compiler's functions are the oracle's, for every input set of `sha256-inputs.json`; the
/// committed bytes are the messages' at the layout's offsets; and the compiler's circuit, run on
/// them, gives FIPS 180-4's chaining words after every block and passes its binding.
#[test]
fn the_sha256_inputs_from_bytes_are_the_oracles() {
    let document = read_json(&vectors().join("anchor-graphs/sha256-inputs.json"));
    let graphs = graphs_document();
    let cases = document["cases"].as_array().expect("cases");
    assert_eq!(cases.len(), 81, "81 input sets");
    let iv: Vec<u8> = SHA256_IV.iter().flat_map(|w| w.to_be_bytes()).collect();
    let masks: Vec<i64> = SHA256_MASKS.iter().flatten().copied().collect();
    let mut circuits: BTreeMap<(u32, u32, Vec<u32>), Circuit> = BTreeMap::new();
    let mut failures = Vec::new();
    // Blocks of chaining words compared, one per message and block: a guard against a run that
    // compares nothing.
    let mut compared = 0usize;
    let blocks: usize = cases
        .iter()
        .map(|c| (int(&c["M"]) * int(&c["N"])) as usize)
        .sum();
    for case in cases {
        let name = case["name"].as_str().expect("a name");
        let (n, m) = (uint(&case["N"]), uint(&case["M"]));
        let offsets: Vec<u32> = case["committed_offsets"]
            .as_array()
            .expect("offsets")
            .iter()
            .map(uint)
            .collect();
        assert_eq!(uint(&case["K"]) as usize, offsets.len(), "{name}: K");
        // A named graph is the file's graph of these parameters.
        if let Some(graph) = case["graph"].as_str() {
            let g = graphs["cases"]
                .as_array()
                .expect("cases")
                .iter()
                .find(|g| g["name"] == graph)
                .expect("the named graph");
            assert!(
                g["N"] == case["N"]
                    && g["M"] == case["M"]
                    && g["committed_offsets"] == case["committed_offsets"],
                "{name}: the graph {graph} has its parameters"
            );
        }
        let messages: Vec<Vec<u8>> = case["messages"]
            .as_array()
            .expect("messages")
            .iter()
            .map(|v| hex(v.as_str().expect("hex")))
            .collect();
        let digests: Vec<Vec<u8>> = case["digests"]
            .as_array()
            .expect("digests")
            .iter()
            .map(|v| hex(v.as_str().expect("hex")))
            .collect();
        assert_eq!(messages.len(), m as usize, "{name}: M messages");
        for message in &messages {
            assert_eq!(sha256_blocks(message.len()), n as usize, "{name}: N blocks");
        }
        let layout = Sha256Layout {
            blocks: n,
            messages: m,
            committed: offsets.clone(),
        };
        // The public bytes alone, the committed ones zeroed, as `sha256_pad` takes them; the
        // public bits do not depend on the committed bytes.
        let public: Vec<Vec<u8>> = messages
            .iter()
            .map(|message| {
                let mut bytes = message.clone();
                for &o in &offsets {
                    if let Some(b) = bytes.get_mut(o as usize) {
                        *b = 0;
                    }
                }
                sha256_pad(&bytes)
            })
            .collect();
        let padded: Vec<Vec<u8>> = messages.iter().map(|msg| sha256_pad(msg)).collect();
        let public_refs: Vec<&[u8]> = public.iter().map(Vec::as_slice).collect();
        let padded_refs: Vec<&[u8]> = padded.iter().map(Vec::as_slice).collect();
        let message_bits = layout.public_bits(&public_refs);
        assert_eq!(
            message_bits,
            layout.public_bits(&padded_refs),
            "{name}: the public bits do not depend on the committed bytes"
        );
        let start = sha256_bits(&vec![iv.as_slice(); m as usize]);
        let digest_refs: Vec<&[u8]> = digests.iter().map(Vec::as_slice).collect();
        let digest = sha256_bits(&digest_refs);
        let place = layout.place();
        let committed: Vec<i64> = offsets
            .iter()
            .flat_map(|&o| padded.iter().map(move |p| i64::from(p[o as usize])))
            .collect();
        let expected_bits = |field: &str| bits_of(case[field].as_str().expect("bits"));
        let mut fields = vec![
            ("message", &message_bits[..], expected_bits("message")),
            ("start", &start[..], expected_bits("start")),
            ("masks", &masks[..], expected_bits("masks")),
            ("digest", &digest[..], expected_bits("digest")),
        ];
        if offsets.is_empty() {
            assert!(
                case["place"].is_null() && case["committed"].is_null(),
                "{name}: no place and no committed bytes without committed bytes"
            );
        } else {
            let expected: Vec<i64> = case["committed"]
                .as_array()
                .expect("committed")
                .iter()
                .flat_map(|row| row.as_array().expect("a row").iter().map(int))
                .collect();
            fields.push(("place", &place[..], expected_bits("place")));
            fields.push(("committed", &committed[..], expected));
        }
        let mut differs = false;
        for (field, got, expected) in &fields {
            if got != expected {
                differs = true;
                failures.push(format!("{name}: {field}: {}", first_entry(expected, got)));
            }
        }
        if differs {
            continue;
        }
        // The compiler's circuit on these inputs.
        let c = circuits
            .entry((n, m, offsets.clone()))
            .or_insert_with(|| circuit(n, m, offsets.clone()));
        let wide = |v: &[i64]| v.iter().map(|&x| i128::from(x)).collect::<Vec<_>>();
        let mut values = vec![
            (c.inputs[0], wide(&message_bits)),
            (c.inputs[1], wide(&start)),
            (c.inputs[2], wide(&masks)),
        ];
        if !offsets.is_empty() {
            values.push((c.inputs[3], wide(&committed)));
            values.push((c.inputs[4], wide(&place)));
        }
        values.push((*c.inputs.last().expect("the digest"), wide(&digest)));
        let run = evaluate(&c.graph, &[], values);
        let chaining = case["chaining"].as_array().expect("chaining");
        assert_eq!(
            chaining.len(),
            m as usize,
            "{name}: chaining words per message"
        );
        for (column, per_message) in chaining.iter().enumerate() {
            let per_block = per_message.as_array().expect("blocks");
            assert_eq!(
                per_block.len(),
                c.chaining.len(),
                "{name}: N blocks of words"
            );
            for (b, words) in per_block.iter().enumerate() {
                compared += 1;
                let got: Vec<String> = c.chaining[b]
                    .iter()
                    .map(|t| format!("{:08x}", word(&run[t.0 as usize], m as usize, column)))
                    .collect();
                let expected: Vec<String> = words
                    .as_array()
                    .expect("words")
                    .iter()
                    .map(|w| w.as_str().expect("hex").to_owned())
                    .collect();
                if got != expected {
                    failures.push(format!(
                        "{name}: message {column}, block {b}: the oracle's chaining words \
                         {expected:?}, the circuit's {got:?}"
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} differences from the oracle's inputs:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert_eq!(compared, blocks, "every block of every message run");
}

// ----- Ranges ----------------------------------------------------------------------------------

/// An interval endpoint of the vectors: a JSON integer.
fn endpoint(v: &Value) -> i128 {
    v.as_i64()
        .map(i128::from)
        .or_else(|| v.as_u64().map(i128::from))
        .unwrap_or_else(|| panic!("an endpoint within 64 bits: {v}"))
}

fn interval(v: &Value) -> Option<(i128, i128)> {
    (!v.is_null()).then(|| (endpoint(&v[0]), endpoint(&v[1])))
}

fn analyzed(ranges: &Ranges) -> Vec<Option<(i128, i128)>> {
    ranges
        .all()
        .iter()
        .map(|r| r.bounded().map(|r| (r.lo, r.hi)))
        .collect()
}

/// The first tensor whose interval differs.
fn first_range(expected: &[Option<(i128, i128)>], got: &[Option<(i128, i128)>]) -> Option<String> {
    if let Some(t) = expected.iter().zip(got).position(|(e, g)| e != g) {
        return Some(format!(
            "tensor {t}: the oracle's {:?}, the compiler's {:?}",
            expected[t], got[t]
        ));
    }
    (expected.len() != got.len()).then(|| {
        format!(
            "{} tensors in the oracle's, {} in the compiler's",
            expected.len(),
            got.len()
        )
    })
}

/// A small graph of `ranges/anchor.json`, built through the IR's builder in creation order: its
/// constants by index, then its inputs and nodes by id. A constant outside `i64` is an error: the
/// IR's constants are `i64` values (RFC-0006 'Operation set').
fn rule_graph(case: &Value) -> Result<Graph, String> {
    let mut g = GraphBuilder::new();
    for values in case["constants"].as_array().expect("constants") {
        let values = values
            .as_array()
            .expect("a constant")
            .iter()
            .map(|v| {
                v.as_i64()
                    .ok_or_else(|| format!("the constant {v} is not an i64 (RFC-0006)"))
            })
            .collect::<Result<Vec<i64>, String>>()?;
        g.constant(values).expect("a constant");
    }
    let decl_kind = |k: &Value| match k.as_str() {
        Some("A") => NodeKind::Committed {
            commitment: CommitmentClass::Advice,
        },
        Some("P") => NodeKind::Public,
        Some("v") => NodeKind::Virtual,
        other => panic!("a kind: {other:?}"),
    };
    let shape =
        |v: &Value| -> Vec<u32> { v.as_array().expect("a shape").iter().map(uint).collect() };
    let mut pending: Vec<&Value> = case["inputs"].as_array().expect("inputs").iter().collect();
    pending.sort_by_key(|i| int(&i[0]));
    let mut pending = pending.into_iter().peekable();
    let mut count = 0i64;
    let add_input = |g: &mut GraphBuilder, input: &Value, count: &mut i64| {
        let ty = input[4].as_str().expect("a type");
        let (signed, bits) = ty.split_at(1);
        let decl = TensorDecl {
            shape: shape(&input[2]),
            ty: IntType {
                bits: bits.parse().expect("a width"),
                signed: signed == "i",
            },
            range: IntRange {
                lo: endpoint(&input[5]),
                hi: endpoint(&input[6]),
            },
            kind: decl_kind(&input[3]),
        };
        let id = g.input(decl).expect("an input");
        assert_eq!(i64::from(id.0), int(&input[0]), "the input's id");
        *count += 1;
    };
    for node in case["nodes"].as_array().expect("nodes") {
        let outputs = node[4].as_array().expect("outputs");
        let reads = node[3].as_array().expect("inputs").iter().map(int).max();
        let before = outputs
            .first()
            .map_or_else(|| reads.unwrap_or(0) + 1, |o| int(&o[0]));
        while pending.peek().is_some_and(|i| int(&i[0]) < before) {
            let input = pending.next().expect("an input");
            add_input(&mut g, input, &mut count);
        }
        let p = &node[2];
        let op = match node[1].as_str().expect("an operation") {
            "GatherPublic" => Op::GatherPublic {
                index: ConstRef(uint(&p["index"])),
            },
            "Slice" => Op::Slice {
                axis: u8::try_from(int(&p["axis"])).expect("an axis"),
                start: uint(&p["start"]),
                len: uint(&p["len"]),
            },
            "Reshape" => Op::Reshape,
            "Transpose" => Op::Transpose {
                perm: p["perm"]
                    .as_array()
                    .expect("perm")
                    .iter()
                    .map(|a| u8::try_from(int(a)).expect("an axis"))
                    .collect(),
            },
            "Add" => Op::Add,
            "ConstMul" => Op::ConstMul {
                c: ConstRef(uint(&p["c"])),
            },
            "ConstAdd" => Op::ConstAdd {
                c: ConstRef(uint(&p["c"])),
            },
            "BitDecompose" => Op::BitDecompose {
                low: u8::try_from(int(&p["low"])).expect("low"),
                high: u8::try_from(int(&p["high"])).expect("high"),
            },
            "BitFn" => Op::BitFn {
                f: match p["f"].as_str() {
                    Some("Xor") => BoolFn::Xor,
                    Some("Xor3") => BoolFn::Xor3,
                    Some("And") => BoolFn::And,
                    Some("Ch") => BoolFn::Ch,
                    Some("Maj") => BoolFn::Maj,
                    other => panic!("a function: {other:?}"),
                },
            },
            "AssertEqual" => Op::AssertEqual,
            other => panic!("no rule graph holds {other}"),
        };
        let inputs: Vec<TensorId> = node[3]
            .as_array()
            .expect("inputs")
            .iter()
            .map(|t| TensorId(uint(t)))
            .collect();
        let decls = outputs
            .iter()
            .map(|o| TensorDecl {
                shape: shape(&o[1]),
                ty: IntType {
                    bits: 64,
                    signed: true,
                },
                range: IntRange { lo: 0, hi: 0 },
                kind: decl_kind(&o[2]),
            })
            .collect();
        let ids = g.node(op, &inputs, decls).expect("a node");
        for (id, o) in ids.iter().zip(outputs) {
            assert_eq!(i64::from(id.0), int(&o[0]), "an output's id");
        }
        count += i64::try_from(ids.len()).expect("a count");
    }
    for input in pending {
        add_input(&mut g, input, &mut count);
    }
    assert_eq!(count, int(&case["tensors"]), "the tensors");
    Ok(g.finish())
}

/// The analysis's verdict in the vectors' notation.
fn verdict(result: &Result<Ranges, CompileError>) -> Value {
    match result {
        Ok(_) => Value::Null,
        Err(e @ (CompileError::Range { node, .. } | CompileError::Wrap { node, .. })) => {
            json!({ "code": e.code().as_str(), "node": node.0 })
        }
        Err(e @ CompileError::Unproven { tensor, .. }) => {
            json!({ "code": e.code().as_str(), "input": tensor.0 })
        }
        Err(other) => json!({ "unexpected": other.to_string() }),
    }
}

/// RFC-0066 'Ranges' and RFC-0006 'Range analysis', 'Committed graph inputs': the compiler's
/// analysis gives every tensor of every circuit the oracle's interval, `settle` writes it into
/// the graph, and every small graph of `ranges/anchor.json` gets the oracle's verdict, with its
/// ranges wherever the analysis returns them.
#[test]
fn the_anchor_ranges_are_the_oracles() {
    let document = read_json(&vectors().join("ranges/anchor.json"));
    let graphs = graphs_document();
    let mut failures = Vec::new();
    let circuits = document["circuits"].as_array().expect("circuits");
    assert_eq!(circuits.len(), 5, "the five graphs");
    for expected in circuits {
        let name = expected["name"].as_str().expect("a name");
        let case = graphs["cases"]
            .as_array()
            .expect("cases")
            .iter()
            .find(|c| c["name"] == name)
            .expect("the graph");
        let mut c = circuit_of(case);
        let want: Vec<Option<(i128, i128)>> = expected["ranges"]
            .as_array()
            .expect("ranges")
            .iter()
            .map(interval)
            .collect();
        let ranges = match analyze(&c.graph, &[]) {
            Ok(ranges) => ranges,
            Err(e) => {
                failures.push(format!("{name}: the analysis refuses it: {e}"));
                continue;
            }
        };
        if let Some(d) = first_range(&want, &analyzed(&ranges)) {
            failures.push(format!("{name}: analyze: {d}"));
        }
        settle(&mut c.graph, &ranges).expect("settled");
        let settled: Vec<Option<(i128, i128)>> = c
            .graph
            .tensors
            .iter()
            .map(|t| Some((t.range.lo, t.range.hi)))
            .collect();
        if let Some(d) = first_range(&want, &settled) {
            failures.push(format!("{name}: settle: {d}"));
        }
        let checked = expected["checked"].clone();
        let count = |pick: fn(&Op) -> bool| c.graph.nodes.iter().filter(|n| pick(&n.op)).count();
        let got = json!({
            "BitDecompose": count(|op| matches!(op, Op::BitDecompose { .. })),
            "BitFn": count(|op| matches!(op, Op::BitFn { .. })),
            "AssertEqual": count(|op| matches!(op, Op::AssertEqual)),
        });
        if checked != got {
            failures.push(format!(
                "{name}: checked: the oracle's {checked}, the compiler's {got}"
            ));
        }
    }
    let rules = document["rules"].as_array().expect("rules");
    assert_eq!(rules.len(), 18, "eighteen small graphs");
    for rule in rules {
        let name = rule["name"].as_str().expect("a name");
        let graph = match rule_graph(rule) {
            Ok(graph) => graph,
            Err(e) => {
                failures.push(format!("{name}: the IR cannot hold it: {e}"));
                continue;
            }
        };
        let result = analyze(&graph, &[]);
        let got = verdict(&result);
        if got != rule["verdict"] {
            failures.push(format!(
                "{name}: the oracle's verdict {}, the compiler's {got}",
                rule["verdict"]
            ));
            continue;
        }
        // The ranges the analysis returns: all of them when it admits the graph, and the
        // fragment's when only a committed input is unproven.
        let ranges = match (&result, &rule["verdict"]["input"]) {
            (Ok(ranges), _) => Some(ranges.clone()),
            (Err(_), Value::Number(_)) => Some(analyze_fragment(&graph, &[]).expect("a fragment")),
            _ => None,
        };
        if let Some(ranges) = ranges {
            let want: Vec<Option<(i128, i128)>> = rule["ranges"]
                .as_array()
                .expect("ranges")
                .iter()
                .map(interval)
                .collect();
            if let Some(d) = first_range(&want, &analyzed(&ranges)) {
                failures.push(format!("{name}: {d}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} differences from the oracle's ranges:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
