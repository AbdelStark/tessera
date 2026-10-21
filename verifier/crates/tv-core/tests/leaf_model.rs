// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Independent public leaf geometry, checked against the Python transcript-schedule oracle.
mod claim_support;
use claim_support::{golden, integer, items, list, member, text, with_layers};
use tv_core::leaf::{ModelSchedule, Tree};
use tv_core::lowering::{Graph, lower};
use tv_core::validation::check_public;
use tv_field::{Fp, Fp2};
use tv_testkit::{Json, read_vectors};

fn graph(
    model: &str,
    n: u32,
    query_block: u32,
    padded: bool,
) -> (Graph, tv_core::claims::ClaimList) {
    let mut claims = golden(model);
    if padded {
        claims = with_layers(&claims, 3);
    }
    claims.gadget_policy.query_block = query_block;
    let public = check_public(&claims).unwrap();
    (lower(&public, n).unwrap(), claims)
}
fn u32_value(value: &Json) -> u32 {
    u32::try_from(integer(value)).unwrap()
}

fn compare(case: &Json, graph: &Graph, policy: tv_core::claims::GadgetPolicy) {
    let plan = ModelSchedule::derive(graph, policy).unwrap();
    let entries: Vec<_> = list(case, "lookup_entries")
        .iter()
        .map(|value| {
            let pair = items(value);
            (
                tv_tables::TableId(u32_value(&pair[0])),
                u64::try_from(integer(&pair[1])).unwrap(),
            )
        })
        .collect();
    assert_eq!(
        plan.lookups(),
        entries,
        "{}: every canonical range family contributes",
        text(case, "name")
    );
    let expected = list(member(case, "sizes"), "trees");
    assert_eq!(plan.schedule().trees().len(), expected.len());
    for (tree, value) in plan.schedule().trees().iter().zip(expected) {
        assert_eq!(
            match tree {
                Tree::Lookup { height, .. }
                | Tree::Table { height, .. }
                | Tree::Forest { height, .. } => *height,
            },
            u32_value(member(value, "height"))
        );
        match tree {
            Tree::Forest { trees_log, .. } => {
                assert_eq!(text(value, "kind"), "forest");
                assert_eq!(*trees_log, u32_value(member(value, "trees_log")));
            }
            _ => assert_eq!(text(value, "kind"), "fraction"),
        }
    }
}

#[test]
fn all_native_and_boundary_geometries_match_the_python_oracle() {
    let file = read_vectors("transcript-schedule/schedule.json");
    let models = read_vectors("models/models.json");
    let cases = list(&file, "models");
    assert_eq!(cases.len(), 8);
    for case in cases {
        let model = text(case, "name").split(',').next().unwrap();
        let fixture = list(&models, "models")
            .iter()
            .find(|m| text(m, "name") == model)
            .unwrap();
        let tokens = list(&list(fixture, "sequences")[0], "tokens");
        let (graph, claims) = graph(model, tokens.len() as u32, 512, false);
        compare(case, &graph, claims.gadget_policy);
    }
    let boundaries = list(&file, "boundaries");
    assert_eq!(boundaries.len(), 16);
    for case in boundaries {
        let padded = matches!(member(case, "padded"), Json::Bool(true));
        let (graph, claims) = graph(
            text(case, "model"),
            u32_value(member(case, "sequence_length")),
            u32_value(member(case, "query_block")),
            padded,
        );
        compare(case, &graph, claims.gadget_policy);
    }
}

fn eq(point: &[Fp2], vertex: usize) -> Fp2 {
    point
        .iter()
        .enumerate()
        .fold(Fp2::ONE, |weight, (bit, &r)| {
            weight
                * if vertex >> bit & 1 == 0 {
                    Fp2::ONE - r
                } else {
                    r
                }
        })
}

#[test]
fn softmax_roots_bind_real_queries_heads_and_layers_with_all_padding_equal_to_one() {
    let mut claims = with_layers(&golden("tiny-2-servable"), 3);
    claims.arch.n_heads = 6;
    claims.arch.n_kv_heads = 3;
    claims.arch.d_model = 96;
    for n in [1u32, 3, 5] {
        let public = check_public(&claims).unwrap();
        let graph = lower(&public, n).unwrap();
        let model = ModelSchedule::derive(&graph, claims.gadget_policy).unwrap();
        let forests: Vec<_> = model
            .schedule()
            .trees()
            .iter()
            .enumerate()
            .filter_map(|(index, tree)| match tree {
                Tree::Forest { trees_log, .. } => Some((index, *trees_log)),
                _ => None,
            })
            .collect();
        assert_eq!(
            forests.len(),
            usize::from(n > 1),
            "a single key needs no product-tree level"
        );
        for (tree, variables) in forests {
            let query_size = n.next_power_of_two() as usize;
            let heads_size = claims.arch.n_heads.next_power_of_two() as usize;
            let layer_size = claims.arch.n_layers.next_power_of_two() as usize;
            let count = query_size * heads_size * layer_size;
            assert_eq!(count, 1 << variables);
            let roots: Vec<_> = (0..layer_size)
                .flat_map(|layer| {
                    (0..heads_size).flat_map(move |head| {
                        (0..query_size).map(move |query| {
                            if layer < 3 && head < 6 && query < n as usize {
                                Fp2::ZERO
                            } else {
                                Fp2::ONE
                            }
                        })
                    })
                })
                .collect();
            for (vertex, &expected) in roots.iter().enumerate() {
                let point: Vec<_> = (0..variables)
                    .map(|bit| Fp2::from(Fp::reduce(((vertex >> bit) & 1) as u64)))
                    .collect();
                assert_eq!(
                    model.forest_root(tree, &point),
                    Some(expected),
                    "n={n}, row={vertex}"
                );
            }
            let point: Vec<_> = (0..variables)
                .map(|bit| {
                    Fp2::new(
                        Fp::reduce(u64::from(bit) + 2),
                        Fp::reduce(u64::from(bit) + 7),
                    )
                })
                .collect();
            let expected = roots
                .iter()
                .enumerate()
                .fold(Fp2::ZERO, |sum, (row, &root)| sum + root * eq(&point, row));
            assert_eq!(
                model.forest_root(tree, &point),
                Some(expected),
                "non-Boolean selector equality"
            );
            assert_eq!(model.forest_root(tree, &point[1..]), None);
            assert_eq!(model.forest_root(usize::MAX, &point), None);
        }
    }
}
