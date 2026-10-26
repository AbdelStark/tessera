// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Independent consumption of RFC-0013's composed-schedule degrees and refusal boundaries.
mod claim_support;

use claim_support::{golden, items, member, number, text};
use tv_core::gadgets::{self, model, runtime};
use tv_core::lowering::Graph;
use tv_core::schedule::ScheduleError;
use tv_core::schedule::degree::{self, Constant, Key, StageDegree};
use tv_core::sumcheck::Protocol;
use tv_core::{lowering, selection, validation};
use tv_tables::Table;
use tv_testkit::{Json, read_vectors};

fn derive(
    graph: &Graph,
    policy: tv_core::claims::GadgetPolicy,
    request: &str,
    log2_k: usize,
) -> Vec<StageDegree> {
    if request == "none" {
        let gadgets = gadgets::Plan::derive(graph, policy).unwrap();
        let model = model::Plan::derive(&gadgets).unwrap();
        degree::protocol2_stages(graph, model.schedule(), log2_k).unwrap()
    } else {
        let selected = selection::derive(graph, policy).unwrap();
        let tables: Vec<_> = selected
            .tables
            .iter()
            .map(|t| Table::generate(t.id, t.kind.clone()).unwrap())
            .collect();
        let prompt = match request {
            "prefill" => graph.length(),
            "generate" => graph.length() - 1,
            _ => panic!("request"),
        };
        let runtime = runtime::Plan::derive(graph, policy, prompt, 0, &tables).unwrap();
        let model = model::Plan::derive_runtime(&runtime).unwrap();
        degree::protocol2_stages(graph, model.schedule(), log2_k).unwrap()
    }
}

#[test]
fn every_field_of_all_forty_four_degree_cases_is_independently_derived() {
    let vector = read_vectors("schedule/degrees.json");
    let cases = items(member(&vector, "cases"));
    assert_eq!(cases.len(), 44);
    let mut stages_seen = 0;
    for case in cases {
        let name = text(case, "model");
        let claims = golden(name);
        let n: u32 = number(member(case, "n"));
        let length: u32 = number(member(case, "length"));
        assert_eq!(length, n.max(3));
        let graph = lowering::lower(&validation::check_public(&claims).unwrap(), length).unwrap();
        let stages = derive(
            &graph,
            claims.gadget_policy,
            text(case, "request"),
            usize::from(claims.layout.advice.log2_k),
        );
        let expected = items(member(case, "stages"));
        assert_eq!(stages.len(), expected.len(), "{name}, {n}");
        for (stage, expected) in stages.iter().zip(expected) {
            stages_seen += 1;
            assert_eq!(stage.index, number(member(expected, "index")));
            assert_eq!(
                stage.rounds,
                number(member(expected, "rounds")),
                "{name}, {n}, stage {} rounds",
                stage.index
            );
            assert_eq!(
                stage.folds,
                number(member(expected, "folds")),
                "{name}, {n}, stage {} folds",
                stage.index
            );
            assert_eq!(
                stage.degree,
                number(member(expected, "d")),
                "{name}, {n}, stage {} degree",
                stage.index
            );
            let instances = items(member(expected, "instances"));
            assert_eq!(stage.instances.len(), instances.len());
            for (found, expected) in stage.instances.iter().zip(instances) {
                assert_eq!(found.relation, text(expected, "relation"));
                assert_eq!(
                    found.stacked,
                    matches!(member(expected, "stacked"), Json::Bool(true)),
                    "{found:?}"
                );
                let public: Vec<bool> = items(member(expected, "public"))
                    .iter()
                    .map(|v| match v {
                        Json::Bool(b) => *b,
                        _ => panic!("public flag"),
                    })
                    .collect();
                assert_eq!(found.public, public);
                let constant = match member(expected, "constant") {
                    Json::Null => None,
                    Json::String(s) => Some(match s.as_str() {
                        "scalar" => Constant::Scalar,
                        "layer" => Constant::Layer,
                        "channel" => Constant::Channel,
                        _ => panic!("constant"),
                    }),
                    _ => panic!("constant"),
                };
                assert_eq!(found.constant, constant);
                assert_eq!(found.row.base, number(member(expected, "d_1")));
                assert_eq!(found.row.masked, number(member(expected, "s")));
                assert_eq!(found.row.degree, number(member(expected, "d")));
            }
        }
    }
    assert_eq!(stages_seen, 110);
}

#[test]
fn unknown_patterns_and_stages_over_the_bound_are_internal_errors() {
    for public in [[true, false], [false, true]] {
        for stacked in [false, true] {
            assert_eq!(
                degree::degree_row(Key {
                    relation: "MatMulAct",
                    public: &public,
                    stacked,
                    constant: None
                }),
                Err(ScheduleError::Internal)
            );
        }
    }
    assert_eq!(
        degree::degree_row(Key {
            relation: "Xor3",
            public: &[true; 3],
            stacked: false,
            constant: None
        }),
        Err(ScheduleError::Internal)
    );
    assert_eq!(
        degree::degree_row(Key {
            relation: "unknown",
            public: &[],
            stacked: false,
            constant: None
        }),
        Err(ScheduleError::Internal)
    );
    assert_eq!(
        degree::stage_degree(Protocol::Two, 1, &[7, 11]),
        Err(ScheduleError::Internal)
    );
    assert_eq!(
        degree::stage_degree(Protocol::One, 1, &[8]),
        Err(ScheduleError::Internal)
    );
    assert_eq!(
        degree::stage_degree(Protocol::Two, 1, &[]),
        Err(ScheduleError::Internal)
    );
    assert_eq!(degree::stage_degree(Protocol::Two, 1, &[7, 10]), Ok(10));
    assert_eq!(degree::stage_degree(Protocol::One, 1, &[7]), Ok(7));
}

#[test]
fn operand_counts_constant_classes_and_public_output_guards_fail_closed() {
    let cases = [
        Key {
            relation: "Mul",
            public: &[false, false],
            constant: Some(Constant::Scalar),
            stacked: false,
        },
        Key {
            relation: "Equality.ConstMul",
            public: &[false, true],
            constant: None,
            stacked: false,
        },
        Key {
            relation: "Equality.ConstMul",
            public: &[true, false],
            constant: Some(Constant::Layer),
            stacked: true,
        },
        Key {
            relation: "Maj",
            public: &[false, false],
            constant: None,
            stacked: false,
        },
        Key {
            relation: "Maj",
            public: &[true; 3],
            constant: None,
            stacked: false,
        },
        Key {
            relation: "Mul",
            public: &[true; 2],
            constant: None,
            stacked: false,
        },
        Key {
            relation: "OneHotIndex",
            public: &[true, false],
            constant: None,
            stacked: false,
        },
        Key {
            relation: "Recomposition",
            public: &[false, true, false],
            constant: None,
            stacked: false,
        },
        Key {
            relation: "Fold",
            public: &[false],
            constant: None,
            stacked: false,
        },
        Key {
            relation: "Product",
            public: &[false],
            constant: None,
            stacked: false,
        },
    ];
    for key in cases {
        assert_eq!(
            degree::degree_row(key),
            Err(ScheduleError::Internal),
            "{key:?}"
        );
    }
    assert_eq!(
        degree::stage_degree(Protocol::Two, 0, &[8]),
        Err(ScheduleError::Internal)
    );
    assert_eq!(
        degree::stage_degree(Protocol::Two, 1, &[0]),
        Err(ScheduleError::Internal)
    );
}
