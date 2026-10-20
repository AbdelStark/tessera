// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Actual OneHot witnesses and CPU batches, after the independent verifier passed.
use serde_json::Value;
use tessera_codec::{Ascii, SecretBuf};
use tessera_field::{Field, Goldilocks as G, GoldilocksExt2 as F, PrimeField, QuadraticExtension};
use tessera_gadgets::{OneHot, OneHotWitness};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_sumcheck::{Instance, Prefix, Table, Term, prove_batch};
use tv_core::gadgets::{Plan, onehot::OneHot as TvOneHot};
use tv_core::sumcheck::{Prefix as TvPrefix, SumcheckError, SumcheckProof, verify_batch};
use tv_field::{Fp, Fp2};

fn array(v: &Value) -> &[Value] {
    v.as_array().unwrap()
}
fn bytes(v: &Value) -> Vec<u8> {
    v.as_str()
        .unwrap()
        .as_bytes()
        .chunks_exact(2)
        .map(|x| u8::from_str_radix(std::str::from_utf8(x).unwrap(), 16).unwrap())
        .collect()
}
fn field(v: &Value) -> F {
    F::new(
        G::from_canonical_u64(v[0].as_u64().unwrap()).unwrap(),
        G::from_canonical_u64(v[1].as_u64().unwrap()).unwrap(),
    )
}
fn tv(v: F) -> Fp2 {
    let [a, b] = v.coeffs();
    Fp2::new(
        Fp::from_canonical(a.to_canonical_u64()).unwrap(),
        Fp::from_canonical(b.to_canonical_u64()).unwrap(),
    )
}
fn eq(point: &[F], row: usize) -> F {
    point.iter().enumerate().fold(F::ONE, |out, (i, &x)| {
        out * if row >> i & 1 == 0 { F::ONE - x } else { x }
    })
}
fn site(case: &Value) -> TvOneHot {
    let claims: Value = serde_json::from_str(include_str!(
        "../../../spec/vectors/claims/claim-lists.json"
    ))
    .unwrap();
    let golden = array(&claims["cases"])
        .iter()
        .find(|c| c["model"] == "tiny-2-servable.d")
        .unwrap();
    let mut claims =
        tv_core::claims::ClaimList::open(&bytes(&golden["hex"]), &tv_codec::DecodeLimits::MESSAGE)
            .unwrap();
    claims.arch.vocab = case["vocab"].as_u64().unwrap() as u32;
    claims.arch.max_context = 8;
    let public = tv_core::validation::check_public(&claims).unwrap();
    let graph = tv_core::lowering::lower(&public, case["n"].as_u64().unwrap() as u32).unwrap();
    let plan = Plan::derive(&graph, claims.gadget_policy).unwrap();
    plan.one_hots().next().unwrap().unwrap()
}
fn transcripts(case: &Value) -> (Transcript, tv_hash::transcript::Transcript) {
    let c = &case["context"];
    let statement = bytes(&c["statement_digest"]).try_into().unwrap();
    let session = bytes(&c["session_id"]).try_into().unwrap();
    (
        Transcript::new(&SessionContext {
            protocol: Ascii::new(c["protocol"].as_str().unwrap()).unwrap(),
            mode: Mode::NonInteractive,
            params: ParamSetId(Ascii::new(c["params"].as_str().unwrap()).unwrap()),
            statement_digest: statement,
            session_id: session,
        }),
        tv_hash::transcript::Transcript::new(&tv_hash::transcript::SessionContext {
            protocol: tv_hash::transcript::Ascii::new(c["protocol"].as_str().unwrap()).unwrap(),
            mode: tv_hash::transcript::Mode::NonInteractive,
            params: tv_hash::transcript::ParamSetId(
                tv_hash::transcript::Ascii::new(c["params"].as_str().unwrap()).unwrap(),
            ),
            statement_digest: statement,
            session_id: session,
        }),
    )
}

#[test]
fn cpu_witnesses_and_component_proofs_match_independent_onehot_and_omission_vectors() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../spec/vectors/gadget-finals/onehot.json"
    ))
    .unwrap();
    let (mut honest, mut alternatives) = (0, 0);
    for case in array(&fixture["cases"]) {
        let n = case["n"].as_u64().unwrap() as usize;
        let vocab = case["vocab"].as_u64().unwrap() as usize;
        let producer = OneHot::<G>::new(vocab).unwrap();
        let padded = producer.padded_len();
        let vbits = padded.trailing_zeros() as usize;
        let failed = case["failed"].as_u64().map(|x| x as usize);
        let record = if failed.is_some() {
            &case["omission"]
        } else {
            case
        };
        let mut h = SecretBuf::new();
        for (row, token) in array(&case["tokens"]).iter().enumerate() {
            let witness = if failed.is_none() {
                producer.witness(token.as_i64().unwrap(), row < n).unwrap()
            } else {
                OneHotWitness::from_column(
                    array(&case["h"])[row * padded..(row + 1) * padded]
                        .iter()
                        .map(|x| G::from_i64(x.as_i64().unwrap()))
                        .collect(),
                )
            };
            h.extend(witness.indicator().iter().copied());
        }
        assert_eq!(
            h.iter().map(|v| v.to_canonical_u64()).collect::<Vec<_>>(),
            array(&case["h"])
                .iter()
                .map(|v| G::from_i64(v.as_i64().unwrap()).to_canonical_u64())
                .collect::<Vec<_>>()
        );
        let target: Vec<_> = array(&case["target"]).iter().map(field).collect();
        let claims: Vec<_> = array(&case["claims"]).iter().map(field).collect();
        let (mut ours, mut theirs) = transcripts(case);
        ours.absorb("stage/0/split", &vec![claims[3]]);
        let mut encoded = 1u64.to_le_bytes().to_vec();
        encoded.extend_from_slice(&tv(claims[3]).to_le_bytes());
        theirs.absorb(
            tv_hash::transcript::Ascii::new("stage/0/split").unwrap(),
            &encoded,
        );
        assert_eq!(
            ours.challenge_fp2("stage/1/layer/0"),
            field(&record["layer"])
        );
        assert_eq!(
            theirs
                .challenge_fp2(tv_hash::transcript::Ascii::new("stage/1/layer/0").unwrap())
                .unwrap(),
            tv(field(&record["layer"]))
        );
        let zero: Vec<_> = array(&record["zero"])
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let label = format!("stage/1/zero/{i}");
                let value = ours.challenge_fp2(&label);
                assert_eq!(value, field(v));
                assert_eq!(
                    theirs
                        .challenge_fp2(tv_hash::transcript::Ascii::new(&label).unwrap())
                        .unwrap(),
                    tv(value)
                );
                value
            })
            .collect();
        let full_eq: Vec<_> = (0..h.len()).map(|row| eq(&zero, row)).collect();
        let row_eq: Vec<_> = (0..h.len()).map(|row| eq(&target, row / padded)).collect();
        let valid: Vec<_> = (0..h.len())
            .map(|row| G::from_u64_reduced(u64::from(row % padded < vocab)))
            .collect();
        let indices: Vec<_> = (0..h.len())
            .map(|row| G::from_u64_reduced((row % padded) as u64))
            .collect();
        let origin: Vec<_> = (0..h.len())
            .map(|row| G::from_u64_reduced(u64::from(row == 0)))
            .collect();
        let instances: Vec<_> = (0..4)
            .map(|i| {
                if failed == Some(i) {
                    Instance::sum_of_products(
                        vec![Table::base(&origin).public(), Table::base(&h)],
                        vec![Term::new(claims[i], vec![0])],
                    )
                } else {
                    match i {
                        0 => Instance::sum_of_products(
                            vec![Table::ext(&full_eq).public(), Table::base(&h)],
                            vec![
                                Term::new(F::ONE, vec![0, 1]),
                                Term::new(-F::ONE, vec![0, 1, 1]),
                            ],
                        ),
                        1 => Instance::sum_of_products(
                            vec![
                                Table::ext(&full_eq).public(),
                                Table::base(&valid).public(),
                                Table::base(&h),
                            ],
                            vec![
                                Term::new(F::ONE, vec![0, 2]),
                                Term::new(-F::ONE, vec![0, 1, 2]),
                            ],
                        ),
                        2 => Instance::new(vec![Table::ext(&row_eq).public(), Table::base(&h)]),
                        _ => Instance::new(vec![
                            Table::ext(&row_eq).public(),
                            Table::base(&indices).public(),
                            Table::base(&h),
                        ]),
                    }
                }
            })
            .collect();
        let proven = prove_batch::<F, _>(&mut ours, Prefix::Stage(1), 3, &instances).unwrap();
        let proof = proven.proof();
        assert_eq!(*proven.gamma(), field(&record["gamma"]));
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
        assert_eq!(ours.state().as_slice(), bytes(&record["state"]));
        let proof = SumcheckProof {
            rounds: proof
                .rounds
                .iter()
                .map(|r| r.iter().copied().map(tv).collect())
                .collect(),
            final_evals: proof.final_evals.iter().copied().map(tv).collect(),
        };
        let site = site(case);
        let target: Vec<_> = target.into_iter().map(tv).collect();
        let bound = site.at(&target, tv(claims[3])).unwrap();
        assert_eq!(bound.input_claim().value, tv(claims[3]));
        let relations: Vec<_> = site.relations().collect();
        let expected: Vec<_> = relations.iter().map(|&r| bound.instance(r)).collect();
        let zero: Vec<_> = zero.into_iter().map(tv).collect();
        let checked = verify_batch(
            &mut theirs,
            TvPrefix::Stage(1),
            3,
            &expected,
            &proof,
            |i, p, e| {
                bound
                    .evaluate(relations[i], &zero, p, e)
                    .ok()
                    .map(|v| v.value())
            },
        );
        if failed.is_some() {
            assert_eq!(
                checked.err(),
                Some(SumcheckError::FinalMismatch {
                    round: zero.len() as u32
                })
            );
            alternatives += 1;
        } else {
            let checked = checked.unwrap();
            assert_eq!(
                checked.point,
                proven.point().iter().copied().map(tv).collect::<Vec<_>>()
            );
            assert_eq!(theirs.state().as_slice(), bytes(&record["state"]));
            for (i, &relation) in relations.iter().enumerate() {
                let evaluated = bound
                    .evaluate(
                        relation,
                        &zero,
                        &checked.point,
                        &proof.final_evals[i..i + 1],
                    )
                    .unwrap();
                let claim = evaluated.claims().next().unwrap();
                assert_eq!(
                    claim.input.source,
                    tv_core::advice::Source::Tensor(tv_core::lowering::TensorId(2))
                );
                assert_eq!(claim.input.point, checked.point);
                assert_eq!(claim.value, proof.final_evals[i]);
            }
            assert_eq!(zero.len(), vbits + target.len());
            honest += 1;
        }
    }
    assert_eq!((honest, alternatives), (6, 6));
}
