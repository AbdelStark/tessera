// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The public selection view preserves the verifier's canonical derivation (RFC-0007).
mod claim_support;
use claim_support::{golden, goldens};
use tv_core::{lowering, ranges, selection, validation};

#[test]
fn derived_catalog_matches_golden_table_definitions_and_independent_ranges() {
    let mut count = 0;
    for g in goldens() {
        let c = golden(&g.name);
        let public = validation::check_public(&c).unwrap();
        for n in [1, 4, 5, c.arch.max_context] {
            let graph = lowering::lower(&public, n).unwrap();
            let derived = selection::derive(&graph, c.gadget_policy).unwrap();
            assert_eq!(derived.ranges, ranges::analyze(&graph).unwrap());
            assert_eq!(derived.tables.len(), c.tables.len());
            for (found, expected) in derived.tables.iter().zip(&c.tables) {
                assert_eq!(found.id, expected.id);
                assert_eq!(found.kind, expected.kind);
                assert_eq!(found.log2_rows, expected.log2_rows);
            }
            let expected: Vec<_> = graph
                .nodes()
                .iter()
                .enumerate()
                .filter(|(_, node)| {
                    matches!(
                        node.op(),
                        lowering::Op::OneHot { .. }
                            | lowering::Op::Requant { .. }
                            | lowering::Op::TableMap { .. }
                            | lowering::Op::SoftmaxRow { .. }
                            | lowering::Op::RsqrtRow { .. }
                    )
                })
                .map(|(i, _)| lowering::NodeId(i as u32))
                .collect();
            assert_eq!(
                derived.choices.iter().map(|g| g.node).collect::<Vec<_>>(),
                expected
            );
            for choice in &derived.choices {
                for id in &choice.tables {
                    assert_eq!(derived.tables[id.0 as usize].id, *id);
                }
            }
            count += 1;
        }
    }
    assert_eq!(count, 16);
}
