// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Actual root columns and CPU proofs against the frozen model-binding oracle.
use serde_json::Value;
use tessera_codec::{Ascii, SecretBuf};
use tessera_field::{Field, Goldilocks as G, GoldilocksExt2 as F, PrimeField, QuadraticExtension};
use tessera_gadgets::{Bracket, CompareColumn as C, RsqrtColumn as R, RsqrtRow, Side};
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
    point.iter().enumerate().fold(F::ONE, |a, (i, &r)| {
        a * if row >> i & 1 == 0 { F::ONE - r } else { r }
    })
}
fn column(v: &Value) -> SecretBuf<G> {
    array(v)
        .iter()
        .map(|x| G::from_u64_reduced(x.as_u64().unwrap()))
        .collect()
}
fn role(v: &Value) -> R {
    let v = array(v);
    let name = v[2].as_str().unwrap();
    let j = v[3].as_u64().unwrap() as usize;
    if name == "root.input" {
        return R::InputLimb(j as u8);
    }
    if name == "root.low" {
        return R::RootLimb(j as u8);
    }
    let (bracket, name) = if let Some(s) = name.strip_prefix("root.lower.") {
        (Bracket::Lower, s)
    } else {
        (Bracket::Upper, name.strip_prefix("root.upper.").unwrap())
    };
    R::Compare(
        bracket,
        match name {
            "left.digit" => C::Digit(Side::Left, j),
            "left.carry" => C::Carry(Side::Left, j),
            "right.digit" => C::Digit(Side::Right, j),
            "right.carry" => C::Carry(Side::Right, j),
            "difference" => C::Difference(j),
            "borrow" => C::Borrow(j),
            _ => panic!("role"),
        },
    )
}

#[test]
fn actual_root_columns_and_cpu_binding_proofs_match_every_frozen_model_variant() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../spec/vectors/gadget-finals/roots.json"
    ))
    .unwrap();
    let specs: Vec<_> = (1..=32)
        .map(|bits| {
            GeneratedTable::new(TableId(u32::from(bits)), TableKind::Range { bits })
                .unwrap()
                .spec()
                .clone()
        })
        .collect();
    let (mut sites, mut batches) = (0, 0);
    for case in array(&fixture["cases"]) {
        let bad = case["bad_padding"].as_bool().unwrap();
        if let Some(wrong) = case.get("wrong_root") {
            let root = RsqrtRow::new(
                wrong["dim"].as_u64().unwrap() as u32,
                wrong["kappa"].as_u64().unwrap() as u8,
                wrong["epsilon"].as_u64().unwrap(),
                &specs,
            )
            .unwrap();
            let u: Vec<_> = array(&wrong["columns"][0])
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    if i < 3 {
                        v.as_i64().unwrap() + wrong["epsilon"].as_i64().unwrap()
                    } else {
                        0
                    }
                })
                .collect();
            let honest = root.witness(&u, &[true, true, true, false]).unwrap();
            assert_ne!(
                honest.tensor(R::Root).unwrap()[0],
                wrong["columns"][1][0].as_i64().unwrap()
            );
            for (keys, bits) in array(&wrong["key_columns"])
                .iter()
                .zip(array(&wrong["range_bits"]))
            {
                let table = GeneratedTable::new(
                    TableId(0),
                    TableKind::Range {
                        bits: bits.as_u64().unwrap() as u8,
                    },
                )
                .unwrap();
                for key in array(keys) {
                    assert_eq!(table.lookup(key.as_i64().unwrap(), 0), Some(0));
                }
            }
        }
        for expected in array(&case["roots"]) {
            let root = RsqrtRow::new(
                expected["dim"].as_u64().unwrap() as u32,
                expected["kappa"].as_u64().unwrap() as u8,
                expected["epsilon"].as_u64().unwrap(),
                &specs,
            )
            .unwrap();
            assert_eq!(root.root_bits(), expected["width"].as_u64().unwrap() as u8);
            let u: Vec<_> = array(&expected["columns"][0])
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    if i < 3 {
                        x.as_i64().unwrap() + expected["epsilon"].as_i64().unwrap()
                    } else {
                        0
                    }
                })
                .collect();
            let witness = root.witness(&u, &[true, true, true, false]).unwrap();
            for (j, source) in array(&expected["sources"]).iter().enumerate().skip(1) {
                let actual = witness
                    .tensor(if j == 1 { R::Root } else { role(source) })
                    .unwrap();
                for (row, (&actual, wanted)) in actual
                    .iter()
                    .zip(array(&expected["columns"][j]))
                    .enumerate()
                {
                    if bad && row == 3 {
                        continue;
                    }
                    assert_eq!(actual, wanted.as_i64().unwrap());
                }
            }
            assert_eq!(witness.tensor(R::Root).unwrap()[3], 0);
            sites += 1;
        }
        let root = &case["roots"][0];
        let record = if bad {
            &case["omission"]
        } else {
            &case["proof"]
        };
        let context = &record["context"];
        let mut t = Transcript::new(&SessionContext {
            protocol: Ascii::new(context["protocol"].as_str().unwrap()).unwrap(),
            mode: Mode::NonInteractive,
            params: ParamSetId(Ascii::new(context["params"].as_str().unwrap()).unwrap()),
            statement_digest: bytes(&context["statement_digest"]).try_into().unwrap(),
            session_id: bytes(&context["session_id"]).try_into().unwrap(),
        });
        let source_values: Vec<_> = array(&root["values"]).iter().map(field).collect();
        t.absorb("stage/0/split", &source_values);
        assert_eq!(t.challenge_fp2("stage/1/layer/0"), field(&record["layer"]));
        let zero: Vec<_> = array(&record["zero"])
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let x = t.challenge_fp2(&format!("stage/1/zero/{i}"));
                assert_eq!(x, field(v));
                x
            })
            .collect();
        let target: Vec<_> = array(&root["point"]).iter().map(field).collect();
        let columns: Vec<_> = array(&root["root_digits"])
            .iter()
            .chain(array(&root["u_digits"]))
            .map(column)
            .collect();
        let count = array(&root["root_digits"]).len();
        let m = array(&root["product_targets"]).len() / 2;
        let equals: Vec<_> = (0..4).map(|i| eq(&target, i)).collect();
        let zero_eq: Vec<_> = (0..4).map(|i| eq(&zero, i)).collect();
        let not_real = [G::ZERO, G::ZERO, G::ZERO, G::ONE];
        let rho = column(&root["columns"][1]);
        // The additional low factor is derived, not a second sent evaluation.
        let plus: SecretBuf<_> = columns[0].iter().map(|&v| v + G::ONE).collect();
        let shifted = columns.len() + 1;
        let mut instances = Vec::new();
        for inc in [false, true] {
            for j in 0..m {
                let mut tables = vec![Table::ext(&equals).public()];
                tables.extend(columns.iter().map(|c| Table::base(c)));
                tables.push(Table::base(&plus).public());
                let mut terms = Vec::new();
                for a in 0..count {
                    for b in 0..count {
                        for k in 0..4 {
                            if a + b + 2 * k == j {
                                let at = |i| if inc && i == 0 { shifted } else { 1 + i };
                                terms.push(Term::new(F::ONE, vec![0, at(a), at(b), 1 + count + k]));
                            }
                        }
                    }
                }
                if terms.is_empty() {
                    terms.push(Term::new(F::ZERO, vec![0]));
                }
                instances.push(Instance::sum_of_products(tables, terms));
            }
        }
        let padding_tables = vec![
            Table::ext(&zero_eq).public(),
            Table::base(&not_real).public(),
            Table::base(&rho),
        ];
        instances.push(if bad {
            Instance::sum_of_products(padding_tables, vec![Term::new(F::ZERO, vec![0])])
        } else {
            Instance::new(padding_tables)
        });
        let batch = prove_batch::<F, _>(&mut t, Prefix::Stage(1), 4, &instances).unwrap();
        let proof = batch.proof();
        assert_eq!(*batch.gamma(), field(&record["gamma"]));
        assert_eq!(
            batch.point(),
            array(&record["point"])
                .iter()
                .map(field)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            proof.rounds,
            array(&record["rounds"])
                .iter()
                .map(|v| array(v).iter().map(field).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            proof.final_evals,
            array(&record["final_evals"])
                .iter()
                .map(field)
                .collect::<Vec<_>>()
        );
        assert_eq!(t.state().as_slice(), bytes(&record["state"]));
        // Check the canonical source extensions independently of the CPU's folds.
        for (i, col) in array(&root["columns"]).iter().enumerate() {
            let expected = field(&root["values"][i]);
            let actual = array(col).iter().enumerate().fold(F::ZERO, |a, (row, v)| {
                a + eq(&target, row) * F::from(G::from_u64_reduced(v.as_u64().unwrap()))
            });
            assert_eq!(actual, expected);
        }
        batches += 1;
    }
    assert_eq!((sites, batches), (35, 7));
}
