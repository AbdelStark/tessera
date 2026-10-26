// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Actual Argmax witnesses and two-stage CPU proofs after independent tests passed.
use serde_json::Value;
use tessera_codec::{Ascii, SecretBuf};
use tessera_field::{Field, Goldilocks as G, GoldilocksExt2 as F, PrimeField, QuadraticExtension};
use tessera_gadgets::{Argmax, ArgmaxWitness, OneHot};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_sumcheck::{Instance, Prefix, Protocol, Table, Term, prove_batch};
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
fn column(v: &Value) -> SecretBuf<G> {
    array(v)
        .iter()
        .map(|v| G::from_i64(v.as_i64().unwrap()))
        .collect()
}
fn eq(point: &[F], i: usize) -> F {
    point.iter().enumerate().fold(F::ONE, |a, (j, &x)| {
        a * if i >> j & 1 == 0 { F::ONE - x } else { x }
    })
}
fn check(batch: &tessera_sumcheck::ProvenBatch<F>, t: &Transcript, record: &Value) {
    assert_eq!(*batch.gamma(), field(&record["gamma"]));
    assert_eq!(
        batch.point(),
        array(&record["point"])
            .iter()
            .map(field)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        batch.proof().rounds,
        array(&record["rounds"])
            .iter()
            .map(|r| array(r).iter().map(field).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        batch.proof().final_evals,
        array(&record["final_evals"])
            .iter()
            .map(field)
            .collect::<Vec<_>>()
    );
    assert_eq!(t.state().as_slice(), bytes(&record["state"]));
    assert_eq!(t.counter(), record["counter"].as_u64().unwrap());
}
#[test]
fn actual_argmax_columns_and_cpu_key_selected_suffix_proofs_match_independent_vectors() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../spec/vectors/gadget-finals/argmax.json"
    ))
    .unwrap();
    let range = GeneratedTable::new(TableId(29), TableKind::Range { bits: 29 }).unwrap();
    let (mut honest, mut alternatives) = (0, 0);
    for case in array(&fixture["cases"]) {
        let vocab = case["vocab"].as_u64().unwrap() as usize;
        let rows = case["rows"].as_u64().unwrap() as usize;
        let padded = vocab.next_power_of_two();
        let total = padded * rows.next_power_of_two();
        let vbits = padded.trailing_zeros() as usize;
        let producer = Argmax::<G>::new(vocab, range.spec()).unwrap();
        let onehot = OneHot::<G>::new(vocab).unwrap();
        let failed = case["failed"].as_u64().map(|v| v as usize);
        let record = if failed.is_some() {
            &case["omission"]
        } else {
            &case["proof"]
        };
        let logits = column(&case["logits"]);
        let h = column(&case["h"]);
        let suffix = column(&case["suffix"]);
        let maxima = column(&case["maxima"]);
        for (row, ell) in logits.chunks_exact(padded).enumerate() {
            let next = onehot
                .witness(case["tokens"][row].as_i64().unwrap(), row < rows)
                .unwrap();
            assert_eq!(next.indicator(), &h[row * padded..(row + 1) * padded]);
            let honest = producer.witness(ell, row < rows).unwrap();
            if failed.is_none() {
                assert_eq!(honest.maximum(), &maxima[row..row + 1]);
            }
            let assignment = ArgmaxWitness::from_column(core::iter::once(maxima[row]).collect());
            let evaluated = producer.evaluate(ell, &next, &assignment).unwrap();
            for (i, &value) in evaluated.suffix.iter().enumerate() {
                assert_eq!(
                    value,
                    G::from_i64(case["actual_suffix"][row * padded + i].as_i64().unwrap())
                );
            }
            if failed.is_none() {
                assert_eq!(
                    &*evaluated.suffix,
                    &suffix[row * padded..(row + 1) * padded]
                );
                for (i, &key) in evaluated.keys.iter().enumerate() {
                    assert_eq!(
                        key,
                        G::from_i64(case["keys"][row * padded + i].as_i64().unwrap())
                    );
                }
            }
        }
        for key in array(&case["keys"]) {
            assert_eq!(range.lookup(key.as_i64().unwrap(), 0), Some(0));
        }
        let context = &case["context"];
        let mut t = Transcript::new(&SessionContext {
            protocol: Ascii::new(context["protocol"].as_str().unwrap()).unwrap(),
            mode: Mode::NonInteractive,
            params: ParamSetId(Ascii::new(context["params"].as_str().unwrap()).unwrap()),
            statement_digest: bytes(&context["statement_digest"]).try_into().unwrap(),
            session_id: bytes(&context["session_id"]).try_into().unwrap(),
        });
        let claims: Vec<_> = array(&case["claims"]).iter().map(field).collect();
        t.absorb("stage/0/split", &claims);
        let key_point: Vec<_> = array(&case["key_target"]).iter().map(field).collect();
        let row_point: Vec<_> = array(&case["row_target"]).iter().map(field).collect();
        let key_eq: Vec<_> = (0..total).map(|i| eq(&key_point, i)).collect();
        let row_eq: Vec<_> = (0..total).map(|i| eq(&row_point, i / padded)).collect();
        let valid: Vec<_> = (0..total)
            .map(|i| G::from_u64_reduced(u64::from(i % padded < vocab)))
            .collect();
        let origin: Vec<_> = (0..total)
            .map(|i| G::from_u64_reduced(u64::from(i == 0)))
            .collect();
        let broadcast: SecretBuf<_> = (0..total).map(|i| maxima[i / padded]).collect();
        let instances = [
            Instance::sum_of_products(
                vec![
                    Table::ext(&key_eq).public(),
                    Table::base(&valid).public(),
                    Table::base(&broadcast),
                    Table::base(&logits),
                    Table::base(&suffix),
                    Table::base(&origin).public(),
                ],
                if failed == Some(0) {
                    vec![Term::new(claims[0], vec![5])]
                } else {
                    vec![
                        Term::new(F::ONE, vec![0, 1, 2]),
                        Term::new(-F::ONE, vec![0, 1, 3]),
                        Term::new(-F::ONE, vec![0, 1, 4]),
                    ]
                },
            ),
            Instance::sum_of_products(
                vec![
                    Table::ext(&row_eq).public(),
                    Table::base(&h),
                    Table::base(&logits),
                    Table::base(&origin).public(),
                ],
                if failed == Some(1) {
                    vec![Term::new(claims[1], vec![3])]
                } else {
                    vec![Term::new(F::ONE, vec![0, 1, 2])]
                },
            ),
        ];
        let first =
            prove_batch::<F, _>(&mut t, Protocol::One, Prefix::Stage(1), 3, &instances).unwrap();
        check(&first, &t, &record["first"]);
        let target = first.point();
        let prefix = producer.suffix_weights::<F>(&target[..vbits]).unwrap();
        let weights: Vec<_> = (0..total)
            .map(|i| eq(&target[vbits..], i / padded) * prefix[i % padded])
            .collect();
        let value = first.proof().final_evals[2];
        let instance = Instance::sum_of_products(
            vec![
                Table::ext(&weights).public(),
                Table::base(&h),
                Table::base(&origin).public(),
            ],
            if failed == Some(2) {
                vec![Term::new(value, vec![2])]
            } else {
                vec![Term::new(F::ONE, vec![0, 1])]
            },
        );
        let second =
            prove_batch::<F, _>(&mut t, Protocol::One, Prefix::Stage(2), 2, &[instance]).unwrap();
        check(&second, &t, &record["second"]);
        if failed.is_some() {
            alternatives += 1
        } else {
            honest += 1
        }
    }
    assert_eq!((honest, alternatives), (6, 6));
}
