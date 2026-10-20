// SPDX-License-Identifier: Apache-2.0 OR MIT

//! CPU row-copy reductions after independent canonical-source tests passed.
use serde_json::Value;
use tessera_codec::{Ascii, SecretBuf};
use tessera_field::{Field, Goldilocks as G, GoldilocksExt2 as F, PrimeField, QuadraticExtension};
use tessera_gadgets::OneHot;
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_sumcheck::{Instance, Prefix, Table, prove_batch};
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
fn eq(point: &[F], i: usize) -> F {
    point.iter().enumerate().fold(F::ONE, |a, (j, &x)| {
        a * if i >> j & 1 == 0 { F::ONE - x } else { x }
    })
}
#[test]
fn cpu_teacher_forcing_row_slices_match_all_independent_source_view_proofs() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../spec/vectors/gadget-finals/argmax-views.json"
    ))
    .unwrap();
    let mut count = 0;
    for case in array(&fixture["cases"]) {
        let n = case["n"].as_u64().unwrap() as usize;
        let outputs = case["outputs"].as_u64().unwrap() as usize;
        let vocab = case["vocab"].as_u64().unwrap() as usize;
        let padded = vocab.next_power_of_two();
        let vb = padded.trailing_zeros() as usize;
        let context = &case["context"];
        let mut t = Transcript::new(&SessionContext {
            protocol: Ascii::new(context["protocol"].as_str().unwrap()).unwrap(),
            mode: Mode::NonInteractive,
            params: ParamSetId(Ascii::new(context["params"].as_str().unwrap()).unwrap()),
            statement_digest: bytes(&context["statement_digest"]).try_into().unwrap(),
            session_id: bytes(&context["session_id"]).try_into().unwrap(),
        });
        let incoming: Vec<_> = array(&case["views"])
            .iter()
            .map(|v| field(&v["value"]))
            .collect();
        t.absorb("stage/1/evals", &incoming);
        let mut row_values = Vec::new();
        let mut kernels = Vec::new();
        for view in array(&case["views"]) {
            let source: SecretBuf<_> = array(&view["column"])
                .iter()
                .map(|v| G::from_i64(v.as_i64().unwrap()))
                .collect();
            if view["role"] == "Indicator" {
                let onehot = OneHot::<G>::new(vocab).unwrap();
                for (row, values) in source.chunks_exact(padded).enumerate() {
                    let token = values.iter().position(|v| *v == G::ONE).unwrap_or(0);
                    let witness = onehot.witness(token as i64, row < n).unwrap();
                    assert_eq!(witness.indicator(), values);
                }
            }
            let target: Vec<_> = array(&view["target"]).iter().map(field).collect();
            let values: SecretBuf<F> = source
                .chunks_exact(padded)
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .fold(F::ZERO, |a, (v, &x)| a + eq(&target[..vb], v) * F::from(x))
                })
                .collect();
            let offset = view["offset"].as_u64().unwrap() as usize;
            let kernel: Vec<_> = (0..n.next_power_of_two())
                .map(|row| {
                    if offset <= row && row < offset + outputs {
                        eq(&target[vb..], row - offset)
                    } else {
                        F::ZERO
                    }
                })
                .collect();
            assert_eq!(
                values
                    .iter()
                    .zip(&kernel)
                    .fold(F::ZERO, |a, (&v, &k)| a + v * k),
                field(&view["value"])
            );
            row_values.push(values);
            kernels.push(kernel);
        }
        let instances: Vec<_> = row_values
            .iter()
            .zip(&kernels)
            .map(|(values, kernel)| {
                Instance::new(vec![Table::ext(kernel).public(), Table::ext(values)])
            })
            .collect();
        let proven = prove_batch::<F, _>(&mut t, Prefix::Stage(2), 2, &instances).unwrap();
        let record = &case["proof"];
        assert_eq!(*proven.gamma(), field(&record["gamma"]));
        assert_eq!(
            proven.point(),
            array(&record["point"])
                .iter()
                .map(field)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            proven.proof().rounds,
            array(&record["rounds"])
                .iter()
                .map(|r| array(r).iter().map(field).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            proven.proof().final_evals,
            array(&record["final_evals"])
                .iter()
                .map(field)
                .collect::<Vec<_>>()
        );
        assert_eq!(t.state().as_slice(), bytes(&record["state"]));
        assert_eq!(t.counter(), record["counter"].as_u64().unwrap());
        count += 1;
    }
    assert_eq!(count, 6);
}
