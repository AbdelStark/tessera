// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0050's complete auxiliary requests against the pre-implementation oracle.
mod claim_support;
use claim_support::{apply_variant, claims_of, integer, items, list, member, number, text};
use tv_core::{lowering, selection, validation};
use tv_tables::TableKind;
use tv_testkit::{Json, read_vectors};

#[test]
fn zero_maximum_root_uses_one_bit_and_still_needs_the_fixed_input_limb_tables() {
    let mut claims = claims_of("tiny-2-servable.d");
    claims.quant.rsqrt_kappa = 0;
    claims.quant.rms_eps_int = 128;
    let public = validation::check_public(&claims).unwrap();
    let graph = lowering::lower(&public, 1).unwrap();
    let intervals = tv_core::ranges::analyze(&graph).unwrap();
    let root = graph
        .nodes()
        .iter()
        .find(|n| matches!(n.op(), lowering::Op::RsqrtRow { .. }))
        .unwrap();
    assert_eq!(
        intervals[root.outputs()[0].0 as usize].bounds(),
        Some((0, 1))
    );
    let found = selection::derive(&graph, claims.gadget_policy).unwrap();
    let root = found
        .choices
        .iter()
        .find(|c| c.pattern == selection::Pattern::RsqrtRow)
        .unwrap();
    assert_eq!(
        found.tables[root.tables[0].0 as usize].kind,
        TableKind::Range { bits: 1 }
    );
    let mut policy = claims.gadget_policy;
    policy.max_log2_rows = 15;
    policy.limb_bits = 15;
    assert!(matches!(
        selection::derive(&graph, policy),
        Err(validation::ClaimsError::StatementMismatch {
            field: validation::StatementField::Gadget {
                gadget: validation::GadgetId(1)
            }
        })
    ));
}

#[test]
fn complete_auxiliary_families_and_repeated_requests_match_the_independent_oracle() {
    let vectors = read_vectors("advice/layout.json");
    let mut count = 0;
    for case in list(&vectors, "cases") {
        // Catalogs do not depend on k, query blocks or segments; cover each graph once.
        if integer(member(case, "log2_k")) != 10 || integer(member(case, "query_block")) != 32 {
            continue;
        }
        let mut claims = claims_of(text(case, "file"));
        if !matches!(member(case, "variant"), Json::Null) {
            apply_variant(member(case, "variant"), &mut claims);
        }
        let public = validation::check_public(&claims).unwrap();
        let graph = lowering::lower(&public, number(member(case, "n"))).unwrap();
        let found = selection::derive(&graph, claims.gadget_policy).unwrap();
        let expected = list(case, "tables");
        assert_eq!(
            found.tables.len(),
            expected.len(),
            "{}: complete tables",
            text(case, "model")
        );
        for (table, row) in found.tables.iter().zip(expected) {
            let [id, kind, bits] = items(row) else {
                panic!("table descriptor")
            };
            assert_eq!(table.id.0, number(id));
            assert_eq!(table.log2_rows, number(bits));
            match &table.kind {
                TableKind::Range { bits } => {
                    assert_eq!(*bits, number(member(member(kind, "Range"), "bits")))
                }
                TableKind::Requant {
                    b,
                    r_by_class,
                    lo,
                    hi,
                } => {
                    let q = member(kind, "Requant");
                    assert_eq!(*b, number(member(q, "b")));
                    assert_eq!(
                        *r_by_class,
                        list(q, "r_by_class")
                            .iter()
                            .map(number)
                            .collect::<Vec<u8>>()
                    );
                    assert_eq!(*lo, number(member(q, "lo")));
                    assert_eq!(*hi, number(member(q, "hi")));
                }
                _ => assert_eq!(table.kind, graph.table(table.id).unwrap().kind),
            }
        }
        assert_eq!(found.choices.len(), list(case, "choices").len());
        for (choice, expected) in found.choices.iter().zip(list(case, "choices")) {
            assert_eq!(choice.node.0, number(member(expected, "node")));
            assert_eq!(
                choice.tables.iter().map(|t| t.0).collect::<Vec<_>>(),
                list(expected, "tables")
                    .iter()
                    .map(number)
                    .collect::<Vec<u32>>(),
                "{} node {}: every lookup, including repeated families",
                text(case, "model"),
                choice.node.0
            );
            let (name, parameters) = match choice.pattern {
                selection::Pattern::OneHot => ("OneHot", vec![]),
                selection::Pattern::Lookup => ("Lookup", vec![]),
                selection::Pattern::RsqrtRow => ("RsqrtRow", vec![]),
                selection::Pattern::SoftmaxRow => ("SoftmaxRow", vec![]),
                selection::Pattern::Requant1 => ("Requant1", vec![]),
                selection::Pattern::RequantSplit { split } => ("RequantSplit", vec![split]),
                selection::Pattern::RequantWide { split, middle } => {
                    ("RequantWide", vec![split, middle])
                }
            };
            let pattern = list(expected, "pattern");
            assert_eq!(pattern[0].as_str(), Some(name));
            assert_eq!(
                pattern[1..].iter().map(number).collect::<Vec<u8>>(),
                parameters
            );
        }
        count += 1;
    }
    assert_eq!(count, 24);
}
