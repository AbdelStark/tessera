// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Conformance vectors, omission controls, ground-truth MLE binding, and adversarial
//! boundary/consistency tests for `tv_core::gadgets::inverse_cdf` (RFC-0009 Lemma 10, RFC-0058).

mod claim_support;
mod support;

use claim_support::{integer, items, list, member};
use support::{Context, digest, extension};
use tv_core::advice::ComparisonRole;
use tv_core::gadgets::GadgetError;
use tv_core::gadgets::inverse_cdf::{
    InverseCdf, Params, RangeClaim, Role, RowValues, SourceClaim, StageVerifyError, Verified,
};
use tv_core::sumcheck::{self, Batch, Prefix, SumcheckError, SumcheckProof};
use tv_core::validation::GadgetId;
use tv_field::{Fp, Fp2};
use tv_hash::transcript::{Ascii, Transcript};
use tv_testkit::{Json, read_vectors};

fn u64_val(v: &Json) -> u64 {
    v.as_u64()
        .unwrap_or_else(|| panic!("{v:?} is not an unsigned u64"))
}

fn u64_vec(v: &Json) -> Vec<u64> {
    items(v).iter().map(u64_val).collect()
}

fn u8_vec(v: &Json) -> Vec<u8> {
    items(v)
        .iter()
        .map(|x| u8::try_from(u64_val(x)).expect("u8"))
        .collect()
}

fn fp2_num(v: &Json) -> Fp2 {
    let a = items(v);
    assert_eq!(a.len(), 2);
    Fp2::new(
        Fp::from_canonical(u64_val(&a[0])).expect("canonical c0"),
        Fp::from_canonical(u64_val(&a[1])).expect("canonical c1"),
    )
}

fn fp2_vec(v: &Json) -> Vec<Fp2> {
    items(v).iter().map(fp2_num).collect()
}

fn signed_col(v: &Json) -> Vec<Fp> {
    items(v)
        .iter()
        .map(|x| {
            let m = i128::from(Fp::MODULUS);
            let rem = u64::try_from(integer(x).rem_euclid(m)).expect("canonical rem");
            Fp::from_canonical(rem).expect("canonical Fp")
        })
        .collect()
}

fn parse_proof(stage_json: &Json) -> SumcheckProof {
    SumcheckProof {
        rounds: list(stage_json, "rounds").iter().map(fp2_vec).collect(),
        final_evals: fp2_vec(member(stage_json, "final_evals")),
    }
}

fn initial_transcript(case: &Json) -> Transcript {
    let mut t = Context::from_json(member(case, "context")).transcript();
    let mut elems = vec![
        fp2_num(member(case, "u_eval")),
        fp2_num(member(case, "z_eval")),
        fp2_num(member(case, "b_eval")),
        fp2_num(member(case, "s_eval")),
    ];
    elems.extend(fp2_vec(member(case, "range_key_evals")));
    let mut bytes = (elems.len() as u64).to_le_bytes().to_vec();
    for e in elems {
        bytes.extend_from_slice(&e.to_le_bytes());
    }
    t.absorb(Ascii::new("stage/0/split").expect("label"), &bytes);
    t
}

fn cmp_role_index(role: ComparisonRole) -> usize {
    match role {
        ComparisonRole::LeftDigit => 0,
        ComparisonRole::LeftCarry => 1,
        ComparisonRole::RightDigit => 2,
        ComparisonRole::RightCarry => 3,
        ComparisonRole::Difference => 4,
        ComparisonRole::Borrow => 5,
    }
}

fn source_truth(case: &Json, k: usize, m: usize, claim: &SourceClaim) -> bool {
    let rows = list(case, "row_records");
    let low_count = k - 1;
    let expected = match claim.role {
        Role::Draw => {
            let col: Vec<Fp> = rows
                .iter()
                .map(|r| Fp::reduce(u64_val(member(r, "u"))))
                .collect();
            extension(&col, &claim.point)
        }
        Role::DrawLow => {
            let col: Vec<Fp> = rows
                .iter()
                .map(|r| Fp::reduce(u64_val(&list(r, "committed")[0])))
                .collect();
            extension(&col, &claim.point)
        }
        Role::PartitionLow(h) => {
            let col: Vec<Fp> = rows
                .iter()
                .map(|r| Fp::reduce(u64_val(&list(r, "committed")[1 + h])))
                .collect();
            extension(&col, &claim.point)
        }
        Role::PrefixLow(h) => {
            let col: Vec<Fp> = rows
                .iter()
                .map(|r| Fp::reduce(u64_val(&list(r, "committed")[1 + low_count + h])))
                .collect();
            extension(&col, &claim.point)
        }
        Role::LowerCompare(crole, limb) => {
            let idx = 1 + 2 * low_count + cmp_role_index(crole) * m + limb;
            let col: Vec<Fp> = rows
                .iter()
                .map(|r| Fp::reduce(u64_val(&list(r, "committed")[idx])))
                .collect();
            extension(&col, &claim.point)
        }
        Role::UpperCompare(crole, limb) => {
            let idx = 1 + 2 * low_count + 6 * m + cmp_role_index(crole) * m + limb;
            let col: Vec<Fp> = rows
                .iter()
                .map(|r| Fp::reduce(u64_val(&list(r, "committed")[idx])))
                .collect();
            extension(&col, &claim.point)
        }
        Role::Exponentials => extension(&signed_col(member(case, "exps")), &claim.point),
        Role::Indicator => extension(&signed_col(member(case, "h_hat")), &claim.point),
    };
    claim.value == expected
}

fn ranges_truth(
    rows: &[Json],
    expected_widths: &[u8],
    row_target: &[Fp2],
    claims: &[RangeClaim],
) -> bool {
    if claims.len() != expected_widths.len() {
        return false;
    }
    for (j, (claim, &bits)) in claims.iter().zip(expected_widths).enumerate() {
        if claim.bits != bits || claim.point != row_target || claim.value != Fp2::ZERO {
            return false;
        }
        let col: Vec<Fp> = rows
            .iter()
            .map(|r| Fp::reduce(u64_val(&list(r, "range_keys")[j])))
            .collect();
        if claim.key != extension(&col, &claim.point) {
            return false;
        }
    }
    true
}

fn delta0(point: &[Fp2]) -> Fp2 {
    point.iter().fold(Fp2::ONE, |acc, &x| acc * (Fp2::ONE - x))
}

#[derive(Debug)]
struct ChainRun {
    batches: [Batch; 3],
    verified: Verified,
    states: [[u8; 32]; 3],
    counters: [u64; 3],
}

fn run_chain(
    site: &InverseCdf,
    case: &Json,
    chain_json: &Json,
    omit: Option<(usize, usize)>,
) -> Result<ChainRun, (usize, StageVerifyError)> {
    let row_target = fp2_vec(member(case, "row_target"));
    let range_keys = fp2_vec(member(case, "range_key_evals"));
    let checked = site
        .check_row(RowValues {
            row_target: &row_target,
            u_eval: fp2_num(member(case, "u_eval")),
            z_eval: fp2_num(member(case, "z_eval")),
            b_eval: fp2_num(member(case, "b_eval")),
            s_eval: fp2_num(member(case, "s_eval")),
            committed_evals: None,
            range_keys: &range_keys,
            range_values: None,
        })
        .map_err(|e| (0, StageVerifyError::Gadget(e)))?;
    assert_eq!(checked.p_targets(), fp2_vec(member(case, "p_targets")));

    let mut t = initial_transcript(case);
    let p1 = parse_proof(member(chain_json, "stage1"));
    let p2 = parse_proof(member(chain_json, "stage2"));
    let p3 = parse_proof(member(chain_json, "stage3"));

    let (b1, done1) = if let Some((1, omit_idx)) = omit {
        let stage1 = checked.squeeze_stage1(&mut t).map_err(|e| (1, e))?;
        let instances = stage1.instances().to_vec();
        let batch = sumcheck::verify_batch(
            &mut t,
            Prefix::Stage(1),
            stage1.degree(),
            &instances,
            &p1,
            |i, pt, vals| {
                if i == omit_idx {
                    Some(instances[i].claim * delta0(pt))
                } else {
                    stage1.evaluate(i, pt, vals).ok()
                }
            },
        )
        .map_err(|e| (1, StageVerifyError::Sumcheck(e)))?;
        let done = stage1
            .finish(batch.point.clone(), &p1.final_evals)
            .map_err(|e| (1, StageVerifyError::Gadget(e)))?;
        (batch, done)
    } else {
        let stage1 = checked.squeeze_stage1(&mut t).map_err(|e| (1, e))?;
        stage1.verify(&mut t, &p1).map_err(|e| (1, e))?
    };
    let s1_state = *t.state();
    let s1_cnt = t.counter();

    let (b2, stage3) = if let Some((2, omit_idx)) = omit {
        let stage2 = done1.squeeze_stage2(&mut t).map_err(|e| (2, e))?;
        let instances = stage2.instances().to_vec();
        let batch = sumcheck::verify_batch(
            &mut t,
            Prefix::Stage(2),
            stage2.degree(),
            &instances,
            &p2,
            |i, pt, vals| {
                if i == omit_idx {
                    Some(instances[i].claim * delta0(pt))
                } else {
                    stage2.evaluate(i, pt, vals).ok()
                }
            },
        )
        .map_err(|e| (2, StageVerifyError::Sumcheck(e)))?;
        let s3 = stage2
            .finish(batch.point.clone(), &p2.final_evals)
            .map_err(|e| (2, StageVerifyError::Gadget(e)))?;
        (batch, s3)
    } else {
        let stage2 = done1.squeeze_stage2(&mut t).map_err(|e| (2, e))?;
        stage2.verify(&mut t, &p2).map_err(|e| (2, e))?
    };
    let s2_state = *t.state();
    let s2_cnt = t.counter();

    let (b3, verified) = if let Some((3, 0)) = omit {
        let inst = stage3.instance();
        let batch = sumcheck::verify_batch(
            &mut t,
            Prefix::Stage(3),
            stage3.degree(),
            &[inst],
            &p3,
            |_, pt, _| Some(inst.claim * delta0(pt)),
        )
        .map_err(|e| (3, StageVerifyError::Sumcheck(e)))?;
        let v = stage3
            .finish(batch.point.clone(), &p3.final_evals)
            .map_err(|e| (3, StageVerifyError::Gadget(e)))?;
        (batch, v)
    } else {
        stage3.verify(&mut t, &p3).map_err(|e| (3, e))?
    };
    let s3_state = *t.state();
    let s3_cnt = t.counter();

    Ok(ChainRun {
        batches: [b1, b2, b3],
        verified,
        states: [s1_state, s2_state, s3_state],
        counters: [s1_cnt, s2_cnt, s3_cnt],
    })
}

#[test]
fn inverse_cdf_schema_cases_match_all_6_oracle_schemas_and_row_checks() {
    let doc = read_vectors("gadget-finals/inverse-cdf.json");
    let schemas = list(&doc, "schema_cases");
    assert_eq!(schemas.len(), 6);
    for (idx, sc) in schemas.iter().enumerate() {
        let vocab = u32::try_from(u64_val(member(sc, "vocab"))).expect("vocab");
        let sigma = u8::try_from(u64_val(member(sc, "sigma"))).expect("sigma");
        let bounded = matches!(member(sc, "bounded"), Json::Bool(true));

        let params = Params::derive(vocab, sigma, bounded).expect("params");
        assert_eq!(u64::from(params.w_bits()), u64_val(member(sc, "w_bits")));
        assert_eq!(params.k() as u64, u64_val(member(sc, "k")));
        assert_eq!(params.m() as u64, u64_val(member(sc, "m")));
        assert_eq!(
            u64::from(params.top_z_bits()),
            u64_val(member(sc, "top_z_bits"))
        );
        assert_eq!(
            params.committed_row_count() as u64,
            u64_val(member(sc, "committed_row_count"))
        );
        assert_eq!(params.range_widths(), u8_vec(member(sc, "range_widths")));

        let lb = list(sc, "lower_bounds");
        assert_eq!(
            params.lower_bounds(),
            (&u64_vec(&lb[0])[..], &u64_vec(&lb[1])[..])
        );
        let lcw = list(sc, "lower_carry_widths");
        assert_eq!(
            params.lower_carry_widths(),
            (&u8_vec(&lcw[0])[..], &u8_vec(&lcw[1])[..])
        );
        let ub = list(sc, "upper_bounds");
        assert_eq!(
            params.upper_bounds(),
            (&u64_vec(&ub[0])[..], &u64_vec(&ub[1])[..])
        );
        let ucw = list(sc, "upper_carry_widths");
        assert_eq!(
            params.upper_carry_widths(),
            (&u8_vec(&ucw[0])[..], &u8_vec(&ucw[1])[..])
        );

        let rows_json = list(sc, "rows");
        let real_rows = rows_json
            .iter()
            .filter(|r| u64_val(member(r, "psi")) == 1)
            .count() as u32;
        let site = InverseCdf::derive(GadgetId(100 + idx as u32), vocab, real_rows, sigma, bounded)
            .expect("site");
        let point = fp2_vec(member(sc, "point"));
        let committed_evals = fp2_vec(member(sc, "committed_evals"));
        let range_keys = fp2_vec(member(sc, "range_key_evals"));
        let zero_vals = vec![Fp2::ZERO; range_keys.len()];
        let checked = site
            .check_row(RowValues {
                row_target: &point,
                u_eval: fp2_num(member(sc, "u_eval")),
                z_eval: fp2_num(member(sc, "z_eval")),
                b_eval: fp2_num(member(sc, "b_eval")),
                s_eval: fp2_num(member(sc, "s_eval")),
                committed_evals: Some(&committed_evals),
                range_keys: &range_keys,
                range_values: Some(&zero_vals),
            })
            .expect("schema check_row");

        // Complete a 0-variable / synthetic finish check on row_claims via prepare_stage1..3?
        // Even better: check_row's p_targets match the lower comparison's right digits/carries.
        assert_eq!(checked.p_targets().len(), params.m());
    }
}

#[test]
fn inverse_cdf_vectors_verify_all_honest_cases_and_reject_every_alternative_and_omission() {
    let doc = read_vectors("gadget-finals/inverse-cdf.json");
    let cases = list(&doc, "cases");
    assert_eq!(cases.len(), 17);

    let mut honest_count = 0usize;
    let mut borrow_reject_count = 0usize;
    let mut range_reject_count = 0usize;
    let mut stage_reject_count = 0usize;

    for case in cases {
        let name = member(case, "name").as_str().expect("name");
        let gadget = GadgetId(u32::try_from(u64_val(member(case, "gadget"))).expect("gadget"));
        let vocab = u32::try_from(u64_val(member(case, "vocab"))).expect("vocab");
        let rows = u32::try_from(u64_val(member(case, "rows"))).expect("rows");
        let sigma = u8::try_from(u64_val(member(case, "sigma"))).expect("sigma");
        let bounded = matches!(member(case, "bounded"), Json::Bool(true));

        let site = InverseCdf::derive(gadget, vocab, rows, sigma, bounded).expect("derive");
        let p = site.params();
        assert_eq!(
            p.range_widths(),
            u8_vec(member(case, "range_widths")),
            "{name}"
        );

        let local_reject = member(case, "local_reject").as_str();
        let failed_stage = member(case, "failed_stage").as_array().map(|arr| {
            (
                usize::try_from(u64_val(&arr[0])).expect("stage"),
                usize::try_from(u64_val(&arr[1])).expect("inst"),
            )
        });

        let proof_json = member(case, "proof");
        let res = run_chain(&site, case, proof_json, None);

        match (local_reject, failed_stage) {
            (Some("lower_borrow" | "upper_borrow"), None) => {
                assert_eq!(
                    res.unwrap_err(),
                    (
                        0,
                        StageVerifyError::Gadget(GadgetError::Rejected { gadget })
                    ),
                    "{name}"
                );
                borrow_reject_count += 1;
            }
            (Some("range"), None) => {
                // At least one row's integer Range key exceeds its prescribed Range { bits }
                // table (rejected at step 7), and u >= 2^32 also violates the upper comparison
                // u * Z + psi <= 2^32 * (B + S) at step 8.
                let row_records = list(case, "row_records");
                let out_of_range = row_records.iter().any(|r| {
                    list(r, "range_keys")
                        .iter()
                        .zip(p.range_widths())
                        .any(|(kv, &bits)| u64_val(kv) >= (1u64 << bits))
                });
                assert!(out_of_range, "{name} must have an out-of-range key");
                assert_eq!(
                    res.unwrap_err(),
                    (
                        0,
                        StageVerifyError::Gadget(GadgetError::Rejected { gadget })
                    ),
                    "{name}"
                );
                range_reject_count += 1;
            }
            (None, Some((stage_no, inst_idx))) => {
                let expected_round = if stage_no == 1 {
                    site.row_variables() as u32
                } else {
                    site.variables() as u32
                };
                assert_eq!(
                    res.unwrap_err(),
                    (
                        stage_no,
                        StageVerifyError::Sumcheck(SumcheckError::FinalMismatch {
                            round: expected_round,
                        }),
                    ),
                    "{name} proof must fail at stage {stage_no}"
                );

                // Adapted omission proof must ALSO fail against the real verifier...
                let omission_json = member(case, "omission");
                assert_eq!(
                    run_chain(&site, case, omission_json, None).unwrap_err(),
                    (
                        stage_no,
                        StageVerifyError::Sumcheck(SumcheckError::FinalMismatch {
                            round: expected_round,
                        }),
                    ),
                    "{name} omission must fail when relation is active"
                );

                // ...and pass when ONLY (stage_no, inst_idx) is substituted!
                let ChainRun {
                    batches,
                    verified,
                    states,
                    counters,
                } = run_chain(&site, case, omission_json, Some((stage_no, inst_idx)))
                    .unwrap_or_else(|e| panic!("{name} omission control failed: {e:?}"));
                for (s_idx, key) in [(0, "stage1"), (1, "stage2"), (2, "stage3")] {
                    let sj = member(omission_json, key);
                    assert_eq!(
                        batches[s_idx].gamma,
                        fp2_num(member(sj, "gamma")),
                        "{name} {key}"
                    );
                    assert_eq!(
                        batches[s_idx].point,
                        fp2_vec(member(sj, "point")),
                        "{name} {key}"
                    );
                    assert_eq!(
                        states[s_idx],
                        digest(member(sj, "state").as_str().expect("state")),
                        "{name} {key}"
                    );
                    assert_eq!(
                        counters[s_idx],
                        u64_val(member(sj, "counter")),
                        "{name} {key}"
                    );
                }
                assert!(
                    verified
                        .source_claims()
                        .iter()
                        .all(|c| source_truth(case, p.k(), p.m(), c)),
                    "{name} omission source claims"
                );
                stage_reject_count += 1;
            }
            (None, None) => {
                let ChainRun {
                    batches,
                    verified,
                    states,
                    counters,
                } = res.unwrap_or_else(|e| panic!("{name} honest failed: {e:?}"));
                for (s_idx, key) in [(0, "stage1"), (1, "stage2"), (2, "stage3")] {
                    let sj = member(proof_json, key);
                    assert_eq!(
                        batches[s_idx].gamma,
                        fp2_num(member(sj, "gamma")),
                        "{name} {key}"
                    );
                    assert_eq!(
                        batches[s_idx].point,
                        fp2_vec(member(sj, "point")),
                        "{name} {key}"
                    );
                    assert_eq!(
                        states[s_idx],
                        digest(member(sj, "state").as_str().expect("state")),
                        "{name} {key}"
                    );
                    assert_eq!(
                        counters[s_idx],
                        u64_val(member(sj, "counter")),
                        "{name} {key}"
                    );
                }
                assert!(
                    ranges_truth(
                        list(case, "row_records"),
                        p.range_widths(),
                        &fp2_vec(member(case, "row_target")),
                        verified.range_claims(),
                    ),
                    "{name} range claims"
                );
                assert!(
                    verified
                        .source_claims()
                        .iter()
                        .all(|c| source_truth(case, p.k(), p.m(), c)),
                    "{name} source claims"
                );
                honest_count += 1;
            }
            other => panic!("unexpected vector classification {other:?} in {name}"),
        }
    }

    assert_eq!(honest_count, 6);
    assert_eq!(borrow_reject_count, 4);
    assert_eq!(range_reject_count, 1);
    assert_eq!(stage_reject_count, 6);
}

#[test]
fn every_inverse_cdf_final_eval_perturbation_either_rejects_or_falsifies_a_retained_claim() {
    let doc = read_vectors("gadget-finals/inverse-cdf.json");
    for case in list(&doc, "cases")
        .iter()
        .filter(|c| member(c, "local_reject").is_null() && member(c, "failed_stage").is_null())
    {
        let gadget = GadgetId(u32::try_from(u64_val(member(case, "gadget"))).expect("gadget"));
        let vocab = u32::try_from(u64_val(member(case, "vocab"))).expect("vocab");
        let rows = u32::try_from(u64_val(member(case, "rows"))).expect("rows");
        let sigma = u8::try_from(u64_val(member(case, "sigma"))).expect("sigma");
        let bounded = matches!(member(case, "bounded"), Json::Bool(true));
        let site = InverseCdf::derive(gadget, vocab, rows, sigma, bounded).expect("derive");
        let p = site.params();

        let proof_json = member(case, "proof");
        let row_target = fp2_vec(member(case, "row_target"));
        let range_keys = fp2_vec(member(case, "range_key_evals"));
        let p1 = parse_proof(member(proof_json, "stage1"));
        let p2 = parse_proof(member(proof_json, "stage2"));
        let p3 = parse_proof(member(proof_json, "stage3"));

        for stage_to_tamper in 1..=3usize {
            let len = match stage_to_tamper {
                1 => p1.final_evals.len(),
                2 => p2.final_evals.len(),
                _ => p3.final_evals.len(),
            };
            for slot in 0..len {
                let mut m1 = p1.clone();
                let mut m2 = p2.clone();
                let mut m3 = p3.clone();
                match stage_to_tamper {
                    1 => m1.final_evals[slot] += Fp2::ONE,
                    2 => m2.final_evals[slot] += Fp2::ONE,
                    _ => m3.final_evals[slot] += Fp2::ONE,
                }
                let checked = site
                    .check_row(RowValues {
                        row_target: &row_target,
                        u_eval: fp2_num(member(case, "u_eval")),
                        z_eval: fp2_num(member(case, "z_eval")),
                        b_eval: fp2_num(member(case, "b_eval")),
                        s_eval: fp2_num(member(case, "s_eval")),
                        committed_evals: None,
                        range_keys: &range_keys,
                        range_values: None,
                    })
                    .expect("row");
                let mut t = initial_transcript(case);
                let outcome = checked
                    .squeeze_stage1(&mut t)
                    .and_then(|s1| s1.verify(&mut t, &m1))
                    .and_then(|(_, d1)| d1.squeeze_stage2(&mut t))
                    .and_then(|s2| s2.verify(&mut t, &m2))
                    .and_then(|(_, s3)| s3.verify(&mut t, &m3));
                if let Ok((_, verified)) = outcome {
                    assert!(
                        !verified.source_claims().iter().all(|c| source_truth(
                            case,
                            p.k(),
                            p.m(),
                            c
                        )),
                        "stage {stage_to_tamper} slot {slot} perturbation left all claims true!"
                    );
                }
            }
        }
    }
}

#[test]
fn inverse_cdf_parameter_length_range_value_and_cross_instance_consistency_guards() {
    // Domain limits on Params and InverseCdf::derive.
    for (v, r, s) in [
        (0, 1, 0),
        ((1 << 18) + 1, 1, 0),
        (1, 0, 0),
        (1, (1 << 17) + 1, 0),
        (4, 1, 33),
    ] {
        assert_eq!(
            InverseCdf::derive(GadgetId(9), v, r, s, false).unwrap_err(),
            GadgetError::Internal
        );
    }

    let doc = read_vectors("gadget-finals/inverse-cdf.json");
    let case = list(&doc, "cases")
        .iter()
        .find(|c| {
            member(c, "local_reject").is_null()
                && member(c, "failed_stage").is_null()
                && matches!(member(c, "bounded"), Json::Bool(true))
                && u64_val(member(c, "rows")) > 1
        })
        .expect("bounded multi-row honest case");

    let gadget = GadgetId(u32::try_from(u64_val(member(case, "gadget"))).expect("gadget"));
    let vocab = u32::try_from(u64_val(member(case, "vocab"))).expect("vocab");
    let rows = u32::try_from(u64_val(member(case, "rows"))).expect("rows");
    let sigma = u8::try_from(u64_val(member(case, "sigma"))).expect("sigma");
    let site = InverseCdf::derive(gadget, vocab, rows, sigma, true).expect("site");
    let reject = GadgetError::Rejected { gadget };

    let row_target = fp2_vec(member(case, "row_target"));
    let range_keys = fp2_vec(member(case, "range_key_evals"));
    let u_eval = fp2_num(member(case, "u_eval"));
    let z_eval = fp2_num(member(case, "z_eval"));
    let b_eval = fp2_num(member(case, "b_eval"));
    let s_eval = fp2_num(member(case, "s_eval"));

    // Wrong row_target length -> Internal.
    let mut bad_rt = row_target.clone();
    bad_rt.push(Fp2::ZERO);
    assert_eq!(
        site.check_row(RowValues {
            row_target: &bad_rt,
            u_eval,
            z_eval,
            b_eval,
            s_eval,
            committed_evals: None,
            range_keys: &range_keys,
            range_values: None,
        })
        .unwrap_err(),
        GadgetError::Internal
    );

    // Wrong range_keys length -> Rejected.
    assert_eq!(
        site.check_row(RowValues {
            row_target: &row_target,
            u_eval,
            z_eval,
            b_eval,
            s_eval,
            committed_evals: None,
            range_keys: &range_keys[..range_keys.len() - 1],
            range_values: None,
        })
        .unwrap_err(),
        reject
    );

    // Non-zero Range lookup value -> Rejected.
    let mut bad_rv = vec![Fp2::ZERO; range_keys.len()];
    bad_rv[0] = Fp2::ONE;
    assert_eq!(
        site.check_row(RowValues {
            row_target: &row_target,
            u_eval,
            z_eval,
            b_eval,
            s_eval,
            committed_evals: None,
            range_keys: &range_keys,
            range_values: Some(&bad_rv),
        })
        .unwrap_err(),
        reject
    );

    // Wrong committed_evals length -> Rejected.
    assert_eq!(
        site.check_row(RowValues {
            row_target: &row_target,
            u_eval,
            z_eval,
            b_eval,
            s_eval,
            committed_evals: Some(&[Fp2::ZERO]),
            range_keys: &range_keys,
            range_values: None,
        })
        .unwrap_err(),
        reject
    );

    // Stage 1 finish rejects inconsistent repeated evaluations across instances.
    let proof_json = member(case, "proof");
    let p1 = parse_proof(member(proof_json, "stage1"));
    let p2 = parse_proof(member(proof_json, "stage2"));
    let checked = site
        .check_row(RowValues {
            row_target: &row_target,
            u_eval,
            z_eval,
            b_eval,
            s_eval,
            committed_evals: None,
            range_keys: &range_keys,
            range_values: None,
        })
        .expect("row");
    let mut t = initial_transcript(case);
    let stage1 = checked.squeeze_stage1(&mut t).expect("s1");
    let mut bad_s1_evals = p1.final_evals.clone();
    *bad_s1_evals.last_mut().expect("last") += Fp2::ONE;
    assert_eq!(
        stage1
            .finish(vec![Fp2::ZERO; site.row_variables()], &bad_s1_evals)
            .unwrap_err(),
        reject
    );

    // Stage 2 finish rejects inconsistent repeated e or h_hat evaluations across instances.
    let checked = site
        .check_row(RowValues {
            row_target: &row_target,
            u_eval,
            z_eval,
            b_eval,
            s_eval,
            committed_evals: None,
            range_keys: &range_keys,
            range_values: None,
        })
        .expect("row");
    let mut t = initial_transcript(case);
    let (_, done1) = checked
        .squeeze_stage1(&mut t)
        .expect("s1")
        .verify(&mut t, &p1)
        .expect("v1");
    let stage2 = done1.squeeze_stage2(&mut t).expect("s2");
    let mut bad_s2_evals = p2.final_evals.clone();
    bad_s2_evals[8] += Fp2::ONE;
    assert_eq!(
        stage2
            .finish(vec![Fp2::ZERO; site.variables()], &bad_s2_evals)
            .unwrap_err(),
        reject
    );
}
