// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Every canonical root request against the independent pre-implementation oracle.
mod common;
use common::{ADVICE, decl, one, out};
use serde_json::Value;
use tessera_compiler::{analyze_fragment, select};
use tessera_ir::{GadgetPolicy, GraphBuilder, NodeId, Op};
use tessera_tables::TableKind;

fn root(kappa: u8, dim: u32, epsilon: u64) -> tessera_ir::Graph {
    let mut graph = GraphBuilder::new();
    let input = graph.input(decl(&[2], ADVICE, 0, 0)).unwrap();
    one(
        &mut graph,
        Op::RsqrtRow {
            kappa,
            eps_int: epsilon,
            dim,
        },
        &[input],
        out(&[2], ADVICE),
    );
    graph.finish()
}

#[test]
fn all_root_widths_retain_every_request_in_canonical_order() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../spec/vectors/advice/layout.json"
    );
    let vectors: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let roots = vectors["roots"].as_array().unwrap();
    assert_eq!(roots.len(), 32);
    for case in roots {
        let width = case["width"].as_u64().unwrap() as u8;
        let graph = root((width - 1).min(30), if width == 32 { 4 } else { 1 }, 1);
        let ranges = analyze_fragment(&graph, &[]).unwrap();
        let selection = select(&graph, &ranges, &[], &GadgetPolicy::default()).unwrap();
        let found: Vec<_> = selection
            .reads(NodeId(0))
            .unwrap()
            .iter()
            .map(
                |id| match selection.tables.iter().find(|t| t.id == *id).unwrap().kind {
                    TableKind::Range { bits } => u64::from(bits),
                    _ => panic!("root requests only ranges"),
                },
            )
            .collect();
        let expected: Vec<_> = case["requests"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        assert_eq!(found, expected, "root width {width}");
    }
}

#[test]
fn a_zero_maximum_root_uses_one_bit_and_requires_the_fixed_input_ranges() {
    let graph = root(0, 1, 2);
    let ranges = analyze_fragment(&graph, &[]).unwrap();
    let output = graph.nodes[0].outputs[0];
    assert_eq!(ranges.of(output).unwrap().bounded().unwrap().hi, 1);
    let selected = select(&graph, &ranges, &[], &GadgetPolicy::default()).unwrap();
    let requests = selected.reads(NodeId(0)).unwrap();
    assert_eq!(
        selected
            .tables
            .iter()
            .find(|t| t.id == requests[0])
            .unwrap()
            .kind,
        TableKind::Range { bits: 1 }
    );
    let narrow = GadgetPolicy {
        max_log2_rows: 15,
        limb_bits: 15,
        ..GadgetPolicy::default()
    };
    assert!(matches!(
        select(&graph, &ranges, &[], &narrow),
        Err(tessera_compiler::CompileError::Range {
            node: NodeId(0),
            ..
        })
    ));
    for limb_bits in [0, 33] {
        let invalid = GadgetPolicy {
            limb_bits,
            ..GadgetPolicy::default()
        };
        assert!(matches!(
            select(&graph, &ranges, &[], &invalid),
            Err(tessera_compiler::CompileError::Range {
                node: NodeId(0),
                ..
            })
        ));
    }
}

#[test]
fn constants_above_goldilocks_keep_their_whole_integer_root_bound() {
    // N = 4 * 2^62 = 2^64; reducing N modulo Goldilocks would shrink the root to 16 bits.
    let graph = root(31, 4, 2);
    let ranges = analyze_fragment(&graph, &[]).unwrap();
    let output = graph.nodes[0].outputs[0];
    assert_eq!(
        ranges.of(output).unwrap().bounded().unwrap().hi,
        (1i128 << 32) - 1
    );
    let selected = select(&graph, &ranges, &[], &GadgetPolicy::default()).unwrap();
    assert_eq!(selected.reads(NodeId(0)).unwrap().len(), 189);
}
