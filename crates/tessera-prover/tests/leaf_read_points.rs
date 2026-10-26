// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Every tensor the leaf stage reads is read at one leaf point (RFC-0068 'Zero knowledge', #554).
//!
//! A committed tensor's mask takes its data claim's column sub-point and no layer or tile bit
//! (RFC-0013 'Committed tensors'), so two values of one tensor that differ only in row bits would
//! share one functional of its mask and reveal a combination of its data (R10). RFC-0068's leaf
//! values `v_{s,t}` sit at `π_t(r_s) ‖ ρ'_y`: one layer point for every site. They stay apart from
//! each other only if every site that reads a tensor reads it at one point `π_t(P[0..k])`.
//! `LeafLayout::new` checks that within one gadget; this test checks it across the sites of every
//! conformance claim list's leaf layout, at the lengths `cargo xtask check-mask-rank` checks.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use tessera_codec::{DecodeLimits, envelope, schema};
use tessera_compiler::schedule::masks::protocol2_length;
use tessera_compiler::schedule::{LayerMap, NodeRef};
use tessera_compiler::{PublicModel, analyze, lower, select};
use tessera_ir::{ClaimList, TensorId};
use tessera_prover::leaf::{LeafLayout, ModelColumn};

/// The conformance claim lists of `spec/vectors/claims/`, by name.
fn claim_lists() -> Vec<(String, ClaimList)> {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/claims/claim-lists.json");
    let file: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(path).expect("the claim lists")).expect("JSON");
    file["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(|case| {
            let text = case["hex"].as_str().expect("hex");
            let bytes: Vec<u8> = (0..text.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&text[i..i + 2], 16).expect("hex"))
                .collect();
            let claims = envelope::open(&bytes, &schema::CLAIM_LIST, DecodeLimits::MESSAGE)
                .expect("a claim list");
            (case["name"].as_str().expect("a name").to_owned(), claims)
        })
        .collect()
}

/// The reads of each graph tensor, by sites inside and outside the layers alike: each read's
/// site and its leaf coordinates.
type Reads = BTreeMap<TensorId, Vec<(usize, Vec<u32>)>>;

/// Every leaf read of a graph tensor, inside the layers or outside them, with the sites that
/// read it and the leaf coordinates each reads it at. A `RsqrtRow`'s root input carries its
/// input's mask, so it counts as a read of that input.
fn reads(claims: &ClaimList, n: u32) -> Reads {
    let lowering = lower(PublicModel::of_claims(claims), protocol2_length(n)).expect("lowering");
    let ranges = analyze(&lowering.graph, &lowering.tables).expect("ranges");
    let selection = select(
        &lowering.graph,
        &ranges,
        &lowering.tables,
        &claims.gadget_policy,
    )
    .expect("a selection");
    let layers = LayerMap::new(&lowering).expect("layers");
    let leaf = LeafLayout::model(&lowering, &layers, &selection, claims).expect("the leaf layout");
    let mut found = Reads::new();
    for (s, site) in leaf.sites().iter().enumerate() {
        let nodes: Vec<(usize, tessera_ir::NodeId)> = match site.node {
            NodeRef::Position(p) => (0..layers.count)
                .map(|l| (l as usize, layers.node(p, l)))
                .collect(),
            NodeRef::Node(id) => vec![(0, id)],
        };
        for (layer, id) in nodes {
            let node = &lowering.graph.nodes[id.0 as usize];
            for read in leaf.reads(s, layer).expect("the site's reads") {
                let tensor = match read.term {
                    ModelColumn::Input(i) => node.inputs[usize::from(i)],
                    ModelColumn::Output(i) => node.outputs[usize::from(i)],
                    ModelColumn::RootInput => node.inputs[0],
                    // A node's own auxiliary columns and virtual forms: only its site reads them.
                    _ => continue,
                };
                found
                    .entry(tensor)
                    .or_default()
                    .push((s, read.coords.clone()));
            }
        }
    }
    found
}

/// Across the sites of each conformance claim list's leaf layout, at `tessera-protocol/2`'s
/// lengths 1, 3, 4, 5 and `max_context`, every graph tensor is read by one site, at one leaf
/// point: no two sites read one tensor, whether in the layers or outside them, a site in the
/// layers and one outside among them. One site reads a tensor at one point, which
/// `LeafLayout::new` refuses otherwise ('One claim per input'). The test also counts what it saw,
/// so that it is not vacuous.
#[test]
fn every_leaf_tensor_is_read_by_one_site_at_one_point() {
    let mut tensors = 0usize;
    for (name, claims) in claim_lists() {
        for n in [1, 3, 4, 5, claims.arch.max_context] {
            for (tensor, list) in reads(&claims, n) {
                tensors += 1;
                let (site, coords) = &list[0];
                assert!(
                    list.iter().all(|(s, c)| s == site && c == coords),
                    "`{name}` at n = {n}: t{} is read by several sites or at several leaf points: \
                     {list:?}",
                    tensor.0
                );
            }
        }
    }
    assert!(tensors > 1000, "{tensors} tensors read");
}
