// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Actual Softmax witnesses and CPU proofs, after independent verifier tests passed.
use serde_json::Value;
use tessera_codec::{Ascii, Rational, SecretBuf};
use tessera_field::{Field, Goldilocks as G, GoldilocksExt2 as F, PrimeField, QuadraticExtension};
use tessera_gadgets::{Slack, SoftmaxColumn as C, SoftmaxRow};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_sumcheck::{Instance, Prefix, Table, Term, prove_batch};
use tessera_tables::{GeneratedTable, Table as _, TableId, TableKind};

fn array(v: &Value) -> &[Value] {
    v.as_array().unwrap()
}
fn field(v: &Value) -> F {
    F::new(
        G::from_canonical_u64(v[0].as_u64().unwrap()).unwrap(),
        G::from_canonical_u64(v[1].as_u64().unwrap()).unwrap(),
    )
}
fn bytes(v: &Value) -> Vec<u8> {
    v.as_str()
        .unwrap()
        .as_bytes()
        .chunks_exact(2)
        .map(|s| u8::from_str_radix(std::str::from_utf8(s).unwrap(), 16).unwrap())
        .collect()
}
fn eq(point: &[F], row: usize) -> F {
    point.iter().enumerate().fold(F::ONE, |a, (i, &x)| {
        a * if row >> i & 1 == 0 { F::ONE - x } else { x }
    })
}
fn column(v: &Value) -> SecretBuf<G> {
    array(v)
        .iter()
        .map(|v| G::from_i64(v.as_i64().unwrap()))
        .collect()
}

#[test]
fn actual_softmax_witnesses_and_cpu_proofs_match_every_independent_component_batch() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../spec/vectors/gadget-finals/softmax.json"
    ))
    .unwrap();
    let (mut honest, mut alternatives) = (0, 0);
    for case in array(&fixture["cases"]) {
        let n = case["n"].as_u64().unwrap() as usize;
        let heads = case["heads"].as_u64().unwrap() as usize;
        let padded = n.next_power_of_two();
        let padded_heads = heads.next_power_of_two();
        let rows = padded * padded_heads;
        let total = rows * padded;
        let sigma = case["sigma"].as_u64().unwrap() as u8;
        let limb = case["limb"].as_u64().unwrap() as u8;
        let failed = case["failed"].as_u64().map(|n| n as usize);
        let record = if failed.is_some() {
            &case["omission"]
        } else {
            &case["proof"]
        };
        if failed.is_none() {
            let exp_id = TableId(case["tables"][0].as_u64().unwrap() as u32);
            let mut specs = vec![
                GeneratedTable::new(
                    exp_id,
                    TableKind::Exp {
                        key_bits: case["key_bits"].as_u64().unwrap() as u8,
                        sigma,
                        scale: Rational::new(1, 1).unwrap(),
                    },
                )
                .unwrap()
                .spec()
                .clone(),
            ];
            let widths: Vec<_> = std::iter::once(sigma + 1)
                .chain(
                    array(&case["widths"])
                        .iter()
                        .map(|v| v.as_u64().unwrap() as u8),
                )
                .chain(
                    array(&case["widths"])
                        .iter()
                        .map(|v| v.as_u64().unwrap() as u8),
                )
                .collect();
            for (id, bits) in array(&case["tables"])[1..].iter().zip(widths) {
                let id = TableId(id.as_u64().unwrap() as u32);
                if specs.iter().all(|s| s.id != id) {
                    specs.push(
                        GeneratedTable::new(id, TableKind::Range { bits })
                            .unwrap()
                            .spec()
                            .clone(),
                    );
                }
            }
            let producer = SoftmaxRow::new(exp_id, limb, 8, &specs).unwrap();
            for row in 0..rows {
                let valid = if row / padded < heads && row % padded < n {
                    row % padded + 1
                } else {
                    0
                };
                let scores: SecretBuf<_> = array(&case["score_column"])
                    [row * padded..(row + 1) * padded]
                    .iter()
                    .map(|v| v.as_i64().unwrap())
                    .collect();
                let witness = producer.witness(&scores, valid).unwrap();
                assert_eq!(
                    witness.tensor(C::Maximum).unwrap()[0],
                    case["maximum_column"][row].as_i64().unwrap()
                );
                assert_eq!(
                    witness.tensor(C::Reciprocal).unwrap()[0],
                    case["source_columns"][0][row].as_i64().unwrap()
                );
                assert_eq!(
                    witness.tensor(C::Exponentials).unwrap(),
                    array(&case["exp_column"])[row * padded..(row + 1) * padded]
                        .iter()
                        .map(|v| v.as_i64().unwrap())
                        .collect::<Vec<_>>()
                );
                let lows = array(&case["widths"]).len() - 1;
                for (side, slack) in [Slack::A, Slack::B].into_iter().enumerate() {
                    for j in 0..lows {
                        assert_eq!(
                            witness.tensor(C::Limb(slack, j as u8)).unwrap()[0],
                            case["source_columns"][2 + side * lows + j][row]
                                .as_i64()
                                .unwrap()
                        );
                    }
                }
                let evaluated = producer.evaluate::<G>(&scores, valid, &witness).unwrap();
                assert_eq!(evaluated.root, G::from_u64_reduced(u64::from(valid == 0)));
                assert!(evaluated.padding.iter().all(|v| *v == G::ZERO));
                for (i, &leaf) in evaluated.leaves.iter().enumerate() {
                    assert_eq!(
                        leaf,
                        G::from_i64(case["tree_column"][row * padded + i].as_i64().unwrap())
                    );
                }
                let constraints = producer.constraints::<G>();
                assert_eq!(constraints.lookups.len(), array(&case["tables"]).len());
                for group in evaluated.lookups.groups() {
                    let mut offset = 0;
                    for (&request, &len) in group.lookups().iter().zip(group.lengths()) {
                        assert_eq!(
                            group.table().0,
                            case["tables"][request].as_u64().unwrap() as u32
                        );
                        for i in 0..len {
                            let expected = if request == 0 {
                                let c = valid > i;
                                let key = if c {
                                    witness.tensor(C::Maximum).unwrap()[0] - scores[i]
                                } else {
                                    (1i64 << case["key_bits"].as_u64().unwrap()) - 1
                                };
                                G::from_i64(key)
                            } else {
                                G::from_i64(
                                    case["range_columns"][request - 1][row].as_i64().unwrap(),
                                )
                            };
                            assert_eq!(group.keys()[offset + i], expected);
                            if let Some(values) = group.values() {
                                assert_eq!(
                                    values[offset + i],
                                    G::from_i64(
                                        case["exp_column"][row * padded + i].as_i64().unwrap()
                                    )
                                );
                            }
                        }
                        offset += len;
                    }
                }
            }
            honest += 1;
        } else {
            alternatives += 1;
        }
        let context = &case["context"];
        let mut transcript = Transcript::new(&SessionContext {
            protocol: Ascii::new(context["protocol"].as_str().unwrap()).unwrap(),
            mode: Mode::NonInteractive,
            params: ParamSetId(Ascii::new(context["params"].as_str().unwrap()).unwrap()),
            statement_digest: bytes(&context["statement_digest"]).try_into().unwrap(),
            session_id: bytes(&context["session_id"]).try_into().unwrap(),
        });
        let initial: Vec<_> = array(&case["source_values"])
            .iter()
            .chain(array(&case["range_keys"]))
            .chain([
                &case["lookup_key"],
                &case["lookup_value"],
                &case["tree_leaf"],
            ])
            .map(field)
            .collect();
        transcript.absorb("stage/0/split", &initial);
        assert_eq!(
            transcript.challenge_fp2("stage/1/layer/0"),
            field(&record["layer"])
        );
        let zero: Vec<_> = array(&record["zero"])
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let x = transcript.challenge_fp2(&format!("stage/1/zero/{i}"));
                assert_eq!(x, field(v));
                x
            })
            .collect();
        let lookup: Vec<_> = array(&case["lookup_target"]).iter().map(field).collect();
        let tree: Vec<_> = array(&case["tree_target"]).iter().map(field).collect();
        let row: Vec<_> = array(&case["row_target"]).iter().map(field).collect();
        let claims: Vec<_> = array(&case["claims"]).iter().map(field).collect();
        let lookup_eq: Vec<_> = (0..total).map(|i| eq(&lookup, i)).collect();
        let tree_eq: Vec<_> = (0..total).map(|i| eq(&tree, i)).collect();
        let row_eq: Vec<_> = (0..rows).map(|i| eq(&row, i)).collect();
        let zero_eq: Vec<_> = (0..rows).map(|i| eq(&zero, i)).collect();
        let causal: Vec<_> = array(&case["causal"])
            .iter()
            .map(|v| G::from_u64_reduced(v.as_u64().unwrap()))
            .collect();
        let mask: Vec<_> = array(&case["psi"])
            .iter()
            .map(|v| G::ONE - G::from_u64_reduced(v.as_u64().unwrap()))
            .collect();
        let scores = column(&case["score_column"]);
        let maxima = column(&case["maximum_column"]);
        let broadcast: SecretBuf<_> = (0..total).map(|i| maxima[i / padded]).collect();
        let reciprocal = column(&case["source_columns"][0]);
        let inverse = G::from_u64_reduced(padded as u64).inverse().unwrap();
        let sums: SecretBuf<_> = array(&case["source_columns"][1])
            .iter()
            .map(|v| G::from_i64(v.as_i64().unwrap()) * inverse)
            .collect();
        let origin: Vec<_> = (0..total)
            .map(|i| G::from_u64_reduced(u64::from(i == 0)))
            .collect();
        let row_origin: Vec<_> = (0..rows)
            .map(|i| G::from_u64_reduced(u64::from(i == 0)))
            .collect();
        let instances: Vec<_> = (0..5)
            .map(|i| {
                if i < 2 {
                    let tables = vec![
                        Table::ext(if i == 0 { &lookup_eq } else { &tree_eq }).public(),
                        Table::base(&causal).public(),
                        Table::base(&broadcast),
                        Table::base(&scores),
                        Table::base(&origin).public(),
                    ];
                    Instance::sum_of_products(
                        tables,
                        if failed == Some(i) {
                            vec![Term::new(claims[i], vec![4])]
                        } else {
                            vec![
                                Term::new(F::ONE, vec![0, 1, 2]),
                                Term::new(-F::ONE, vec![0, 1, 3]),
                            ]
                        },
                    )
                } else if i == 2 {
                    Instance::sum_of_products(
                        vec![
                            Table::ext(&row_eq).public(),
                            Table::base(&reciprocal),
                            Table::base(&sums),
                            Table::base(&row_origin).public(),
                        ],
                        if failed == Some(i) {
                            vec![Term::new(claims[i], vec![3])]
                        } else {
                            vec![Term::new(
                                F::from(G::from_u64_reduced(padded as u64)),
                                vec![0, 1, 2],
                            )]
                        },
                    )
                } else {
                    Instance::sum_of_products(
                        vec![
                            Table::ext(&zero_eq).public(),
                            Table::base(&mask).public(),
                            Table::base(if i == 3 { &maxima } else { &reciprocal }),
                            Table::base(&row_origin).public(),
                        ],
                        if failed == Some(i) {
                            vec![Term::new(claims[i], vec![3])]
                        } else {
                            vec![Term::new(F::ONE, vec![0, 1, 2])]
                        },
                    )
                }
            })
            .collect();
        let proven = prove_batch::<F, _>(&mut transcript, Prefix::Stage(1), 3, &instances).unwrap();
        let proof = proven.proof();
        assert_eq!(*proven.gamma(), field(&record["gamma"]));
        assert_eq!(
            proven.point(),
            array(&record["point"])
                .iter()
                .map(field)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            proof.rounds,
            array(&record["rounds"])
                .iter()
                .map(|r| array(r).iter().map(field).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            proof.final_evals,
            array(&record["final_evals"])
                .iter()
                .map(field)
                .collect::<Vec<_>>()
        );
        assert_eq!(transcript.state().as_slice(), bytes(&record["state"]));
        assert_eq!(transcript.counter(), record["counter"].as_u64().unwrap());
    }
    assert_eq!((honest, alternatives), (8, 6));
}
