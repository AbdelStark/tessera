// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The CPU engine against the independent oracle on every tensor `D` commits (SC-7, 07 §5).
//!
//! `spec/vectors/models/traces.json` holds, for each sequence of each conformance model, every
//! tensor `D` commits as the Python oracle computes it, with the residual stream before the first
//! layer and the logits, in the order of the forward pass. The PyTorch reference is held to the
//! same traces (`python/tests/test_reference.py`), so the engine and the reference agree bit for
//! bit wherever each agrees with the oracle.
//!
//! - **Bit for bit.** On every sequence, each traced tensor the engine computes is the oracle's.
//! - **Exhaustive.** Every committed tensor a node of the lowering produces has its name in the
//!   traces, and every traced tensor has its tensor in the lowering, except the max certificates'
//!   maxima: the `SoftmaxRow` gadget's advice, which its witness generation computes, not the
//!   engine.
//! - **Located.** A divergence is reported at its first tensor, in the trace's order, and its
//!   index. One seeded at a drawn index of each tensor in turn is found there, and a channel of `V`
//!   moved to another class shows up at layer 1's `V`, in that channel.

use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use tessera_codec::{DecodeLimits, Secret, envelope, schema};
use tessera_compiler::{Lowering, PublicModel, lower, validate};
use tessera_ir::{IntegerModel, NodeKind, Op, QuantPoint, TensorId, WeightType};
use tessera_witness::Engine;

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/models")
}

fn json(file: &str) -> serde_json::Value {
    let text = fs::read_to_string(dir().join(file)).expect("the vectors");
    serde_json::from_str(&text).expect("JSON")
}

fn ints(value: &serde_json::Value) -> Vec<i64> {
    value
        .as_array()
        .expect("values")
        .iter()
        .map(|x| x.as_i64().expect("an integer"))
        .collect()
}

/// A tensor of a trace: its values flat in row-major order.
#[derive(Clone, Debug)]
struct Traced {
    name: String,
    shape: Vec<usize>,
    values: Vec<i64>,
}

/// A conformance model, and each of its sequences with the oracle's trace of it.
struct Case {
    name: String,
    model: IntegerModel,
    sequences: Vec<(Vec<u32>, Vec<Traced>)>,
}

fn cases() -> Vec<Case> {
    let (index, traces) = (json("models.json"), json("traces.json"));
    let entries = index["models"].as_array().expect("models");
    let traced = traces["models"].as_array().expect("models");
    assert_eq!(entries.len(), traced.len(), "a trace per model");
    entries
        .iter()
        .zip(traced)
        .map(|(entry, trace)| {
            let name = entry["name"].as_str().expect("a name").to_owned();
            assert_eq!(trace["name"], entry["name"]);
            let bytes = fs::read(dir().join(entry["file"].as_str().expect("a file")))
                .expect("the artifact");
            let model = envelope::open(&bytes, &schema::INTEGER_MODEL, DecodeLimits::INTEGER_MODEL)
                .expect("an IntegerModel");
            let sequences = entry["sequences"]
                .as_array()
                .expect("sequences")
                .iter()
                .zip(trace["sequences"].as_array().expect("sequences"))
                .map(|(sequence, trace)| {
                    let tokens = ints(&sequence["tokens"])
                        .into_iter()
                        .map(|t| u32::try_from(t).expect("a token"))
                        .collect();
                    let tensors = trace["tensors"]
                        .as_array()
                        .expect("tensors")
                        .iter()
                        .map(|t| Traced {
                            name: t["name"].as_str().expect("a name").to_owned(),
                            shape: ints(&t["shape"])
                                .into_iter()
                                .map(|d| usize::try_from(d).expect("a dimension"))
                                .collect(),
                            values: ints(&t["values"]),
                        })
                        .collect();
                    (tokens, tensors)
                })
                .collect();
            Case {
                name,
                model,
                sequences,
            }
        })
        .collect()
}

/// A site's name in the traces: `layer 0 RopeQ`, or `FinalNorm`.
fn label(point: QuantPoint) -> String {
    match point {
        QuantPoint::Layer { layer, site } => format!("layer {layer} {site:?}"),
        QuantPoint::FinalNorm => "FinalNorm".to_owned(),
        QuantPoint::Embed => "Embed".to_owned(),
    }
}

/// The trace's name of each tensor of the lowering the traces hold: every committed tensor a
/// node produces, the residual stream before the first layer, and the logits.
///
/// A root, an update or an attention layer's outputs take their name from the site their values
/// flow into, through the one node that reads each tensor on the way: a norm's root `ρ` feeds its
/// output's `Requant`, a residual site's update `u` its sum's (RFC-0022), and the exponentials the
/// attention output of their layer.
fn names(lowering: &Lowering) -> HashMap<TensorId, String> {
    let graph = &lowering.graph;
    let sites: HashMap<TensorId, QuantPoint> =
        lowering.sites.iter().map(|&(p, t)| (t, p)).collect();
    let mut readers: HashMap<TensorId, Vec<usize>> = HashMap::new();
    for (k, node) in graph.nodes.iter().enumerate() {
        for &t in &node.inputs {
            readers.entry(t).or_default().push(k);
        }
    }
    let site_of = |mut t: TensorId| loop {
        if let Some(&point) = sites.get(&t) {
            return point;
        }
        let [k] = readers.get(&t).map_or(&[][..], Vec::as_slice) else {
            panic!("tensor {} flows into a site through one reader", t.0);
        };
        t = graph.nodes[*k].outputs[0];
    };
    let mut names: HashMap<TensorId, String> = sites.iter().map(|(&t, &p)| (t, label(p))).collect();
    names.insert(lowering.one_hot, "one-hot".to_owned());
    names.insert(lowering.checkpoints[0], "embedding".to_owned());
    names.insert(lowering.logits, "logits".to_owned());
    for node in &graph.nodes {
        let first = node.outputs[0];
        match node.op {
            Op::RsqrtRow { .. } => {
                names.insert(first, format!("{} root", label(site_of(first))));
            }
            Op::Requant { .. } if !sites.contains_key(&first) => {
                names.insert(first, format!("{} update", label(site_of(first))));
            }
            Op::SoftmaxRow { .. } => {
                let QuantPoint::Layer { layer, .. } = site_of(first) else {
                    panic!("the exponentials flow into their layer");
                };
                names.insert(first, format!("layer {layer} exponentials"));
                names.insert(node.outputs[1], format!("layer {layer} reciprocals"));
            }
            _ => {}
        }
    }
    for node in &graph.nodes {
        for &t in &node.outputs {
            let committed = !matches!(
                graph.tensors[t.0 as usize].kind,
                NodeKind::Virtual | NodeKind::Public
            );
            assert!(
                !committed || names.contains_key(&t),
                "the committed output {} of {} has a name in the traces",
                t.0,
                node.op.name()
            );
        }
    }
    names
}

/// The traced tensors the engine does not compute: the max certificates' maxima, the
/// `SoftmaxRow` gadget's advice.
fn engine_computes(name: &str) -> bool {
    !name.ends_with(" maxima")
}

/// Every tensor of `names` on `tokens`, as the engine computes it, by name.
fn run(case: &Case, model: &IntegerModel, tokens: &[u32]) -> HashMap<String, Vec<i64>> {
    validate(model).expect("a valid model");
    let n = u32::try_from(tokens.len()).expect("a length");
    let lowering = lower(PublicModel::of(model), n).expect("the model lowers");
    let engine = Engine::new(&lowering).expect("the tables regenerate");
    let witness = engine
        .run(model, &Secret::new(tokens.to_vec()))
        .unwrap_or_else(|error| panic!("{}: {error}", case.name));
    let names = names(&lowering);
    let found: HashMap<String, Vec<i64>> = names
        .iter()
        .map(|(&t, name)| {
            let values = witness.tensor(t).expect("the tensor").to_vec();
            (name.clone(), values)
        })
        .collect();
    assert_eq!(found.len(), names.len(), "{}: a name per tensor", case.name);
    found
}

/// The first place where the engine's values differ from the oracle's.
#[derive(Debug, PartialEq, Eq)]
struct Divergence {
    tensor: String,
    index: Vec<usize>,
    oracle: i64,
    engine: i64,
}

impl fmt::Display for Divergence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the first divergence is {} at {:?}: the oracle has {}, the engine {}",
            self.tensor, self.index, self.oracle, self.engine
        )
    }
}

/// Flat index `i` of a row-major tensor of `shape`, one coordinate per axis.
fn unravel(mut i: usize, shape: &[usize]) -> Vec<usize> {
    let mut index = vec![0; shape.len()];
    for (slot, &d) in index.iter_mut().zip(shape).rev() {
        *slot = i % d;
        i /= d;
    }
    index
}

/// The first divergence of `found` from the trace, tensor by tensor in the trace's order, after
/// checking that the engine computes exactly the traced tensors it should, each of its shape.
fn first_divergence(trace: &[Traced], found: &HashMap<String, Vec<i64>>) -> Option<Divergence> {
    let traced: BTreeSet<&str> = trace
        .iter()
        .map(|t| t.name.as_str())
        .filter(|name| engine_computes(name))
        .collect();
    let computed: BTreeSet<&str> = found.keys().map(String::as_str).collect();
    assert_eq!(traced, computed, "the engine computes every traced tensor");
    trace
        .iter()
        .filter(|t| engine_computes(&t.name))
        .find_map(|t| {
            let values = &found[&t.name];
            let elements: usize = t.shape.iter().product();
            assert_eq!(t.values.len(), elements, "{}: its shape", t.name);
            assert_eq!(values.len(), elements, "{}: the engine's size", t.name);
            let i = (0..elements).find(|&i| t.values[i] != values[i])?;
            Some(Divergence {
                tensor: t.name.clone(),
                index: unravel(i, &t.shape),
                oracle: t.values[i],
                engine: values[i],
            })
        })
}

#[test]
fn the_engine_computes_every_committed_tensor_of_the_oracle() {
    let mut checked = 0;
    for case in cases() {
        for (s, (tokens, trace)) in case.sequences.iter().enumerate() {
            let found = run(&case, &case.model, tokens);
            if let Some(divergence) = first_divergence(trace, &found) {
                panic!("{}, sequence {s}: {divergence}", case.name);
            }
            checked += trace.len();
        }
    }
    // 41 tensors on each sequence of tiny-2-servable, 35 of tiny-2-minimal, 39 of tiny-2-affine
    // and 73 of tiny-4-servable.
    assert_eq!(checked, 2 * (41 + 35 + 39 + 73));
}

/// SplitMix64, for drawing where to seed a divergence.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        usize::try_from((z ^ (z >> 31)) % n as u64).expect("small")
    }
}

#[test]
fn a_divergence_seeded_in_any_tensor_is_found_where_it_was_seeded() {
    let mut rng = Rng(0x0091_D1FF_0000_0001);
    let mut seeded = 0;
    for case in cases() {
        let (tokens, trace) = &case.sequences[1];
        let found = run(&case, &case.model, tokens);
        for k in 0..trace.len() {
            let name = trace[k].name.clone();
            if !engine_computes(&name) {
                continue;
            }
            let mut wrong = trace.clone();
            let i = rng.below(wrong[k].values.len());
            wrong[k].values[i] += 1;
            let divergence = first_divergence(&wrong, &found).expect("a divergence");
            assert_eq!(
                divergence,
                Divergence {
                    tensor: name,
                    index: unravel(i, &trace[k].shape),
                    oracle: trace[k].values[i] + 1,
                    engine: trace[k].values[i],
                },
                "{}",
                case.name
            );
            seeded += 1;
        }
    }
    // Every traced tensor but the maxima, two of tiny-2's layers and four of tiny-4's.
    assert_eq!(seeded, (41 - 2) + (35 - 2) + (39 - 2) + (73 - 4));
}

#[test]
fn a_channel_moved_to_another_class_diverges_at_its_site_and_channel() {
    let mut rng = Rng(0x0091_C1A5_5000_0002);
    let case = cases().remove(0);
    let (tokens, trace) = &case.sequences[1];
    let mut model = case.model.clone();
    let params = model
        .quant
        .channels
        .iter_mut()
        .find(|c| c.ty == WeightType::V && c.layer == 1)
        .expect("layer 1's V has several classes");
    let classes = params.class.iter().max().map_or(1, |&c| c + 1);
    assert!(classes > 1);
    let channel = rng.below(params.class.len());
    params.class[channel] = (params.class[channel] + 1) % classes;
    let found = run(&case, &model, tokens);
    let divergence = first_divergence(trace, &found).expect("a divergence");
    assert_eq!(divergence.tensor, "layer 1 V", "{divergence}");
    assert_eq!(divergence.index[1], channel, "{divergence}");
    assert_ne!(divergence.oracle, divergence.engine);
}
