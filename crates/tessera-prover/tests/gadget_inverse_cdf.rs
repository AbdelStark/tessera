// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Differential adapter tests (`DEP-2`) joining `tessera-scoring`, `tessera-compiler`,
//! `tessera-gadgets` and `tessera-sumcheck` to `tv-core::scoring` and
//! `tv-core::gadgets::inverse_cdf` over `spec/vectors/scoring/*` and
//! `spec/vectors/gadget-finals/inverse-cdf.json`.

use serde_json::Value;
use tessera_codec::{Ascii, HeapKey, Secret, SecretBuf};
use tessera_compiler::{CdfAdviceColumn, cdf_columns_and_requests};
use tessera_field::{Field, Goldilocks as G, GoldilocksExt2 as F, PrimeField, QuadraticExtension};
use tessera_gadgets::{GadgetError, InverseCdf, OneHot, OneHotWitness};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_scoring::{ClassBits, Sampler, SamplerBaseKey, SamplerKey, inverse_cdf};
use tessera_sumcheck::{Instance, Prefix, Protocol, Table, Term, prove_batch};
use tessera_tables::{GeneratedTable, Table as _, TableId, TableKind};
use tv_core::gadgets::GadgetError as TvGadgetError;
use tv_core::gadgets::inverse_cdf::{
    InverseCdf as TvInverseCdf, Params as TvParams, RowValues as TvRowValues, StageVerifyError,
};
use tv_core::sumcheck::{SumcheckError, SumcheckProof as TvSumcheckProof};
use tv_core::validation::GadgetId as TvGadgetId;
use tv_field::{Fp, Fp2};

fn array(v: &Value) -> &[Value] {
    v.as_array().unwrap()
}

fn field(v: &Value) -> F {
    F::new(
        G::from_canonical_u64(v[0].as_u64().unwrap()).unwrap(),
        G::from_canonical_u64(v[1].as_u64().unwrap()).unwrap(),
    )
}

fn tv_fp2(v: F) -> Fp2 {
    let [a, b] = v.coeffs();
    Fp2::new(
        Fp::from_canonical(a.to_canonical_u64()).unwrap(),
        Fp::from_canonical(b.to_canonical_u64()).unwrap(),
    )
}

fn hex_bytes(s: &str) -> Vec<u8> {
    s.as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

fn eq_at(point: &[F], index: usize) -> F {
    point.iter().enumerate().fold(F::ONE, |acc, (bit, &r)| {
        acc * if (index >> bit) & 1 == 0 {
            F::ONE - r
        } else {
            r
        }
    })
}

fn transcripts(context: &Value) -> (Transcript, tv_hash::transcript::Transcript) {
    let statement: [u8; 32] = hex_bytes(context["statement_digest"].as_str().unwrap())
        .try_into()
        .unwrap();
    let session: [u8; 16] = hex_bytes(context["session_id"].as_str().unwrap())
        .try_into()
        .unwrap();
    let protocol = context["protocol"].as_str().unwrap();
    let params = context["params"].as_str().unwrap();
    (
        Transcript::new(&SessionContext {
            protocol: Ascii::new(protocol).unwrap(),
            mode: Mode::NonInteractive,
            params: ParamSetId(Ascii::new(params).unwrap()),
            statement_digest: statement,
            session_id: session,
        }),
        tv_hash::transcript::Transcript::new(&tv_hash::transcript::SessionContext {
            protocol: tv_hash::transcript::Ascii::new(protocol).unwrap(),
            mode: tv_hash::transcript::Mode::NonInteractive,
            params: tv_hash::transcript::ParamSetId(
                tv_hash::transcript::Ascii::new(params).unwrap(),
            ),
            statement_digest: statement,
            session_id: session,
        }),
    )
}

fn absorb_fp2_slice(t: &mut tv_hash::transcript::Transcript, label: &str, values: &[F]) {
    let mut encoded = Vec::with_capacity(8 + 16 * values.len());
    encoded.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for &v in values {
        encoded.extend_from_slice(&tv_fp2(v).to_le_bytes());
    }
    t.absorb(tv_hash::transcript::Ascii::new(label).unwrap(), &encoded);
}

fn to_tv_proof(record: &Value) -> TvSumcheckProof {
    TvSumcheckProof {
        rounds: array(&record["rounds"])
            .iter()
            .map(|r| array(r).iter().map(|v| tv_fp2(field(v))).collect())
            .collect(),
        final_evals: array(&record["final_evals"])
            .iter()
            .map(|v| tv_fp2(field(v)))
            .collect(),
    }
}

fn check_stage_record(batch: &tessera_sumcheck::ProvenBatch<F>, t: &Transcript, record: &Value) {
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
    assert_eq!(
        t.state().as_slice(),
        hex_bytes(record["state"].as_str().unwrap())
    );
    assert_eq!(t.counter(), record["counter"].as_u64().unwrap());
}

#[test]
fn prover_and_verifier_samplers_agree_across_all_shared_scoring_vectors() {
    let mut derive_rows = 0usize;
    for line in include_str!("../../../spec/vectors/scoring/derive.txt")
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
    {
        let cols: Vec<&str> = line.split_whitespace().collect();
        let base_bytes: [u8; 16] = hex_bytes(cols[0]).try_into().unwrap();
        let nonce: [u8; 32] = hex_bytes(cols[1]).try_into().unwrap();
        let bits: u8 = cols[2].parse().unwrap();
        let from_ns: u64 = cols[3].parse().unwrap();
        let expected_key: [u8; 16] = hex_bytes(cols[4]).try_into().unwrap();
        let expected_class: u64 = cols[5].parse().unwrap();
        let expected_draw: u32 = cols[6].parse().unwrap();

        let prover_base = SamplerBaseKey::new(Secret::new(HeapKey::from_box(Box::new(base_bytes))));
        let prover_key = prover_base.derive(&nonce, ClassBits::new(bits).unwrap(), from_ns);
        let prover_sampler = Sampler::new(&prover_key);
        let prover_class = prover_sampler.class(&Secret::new(vec![1, 2, 3]));
        let prover_draw = prover_sampler.draw(prover_class, 0);

        let tv_key = tv_core::scoring::derive_sampler_key(&base_bytes, &nonce, bits, from_ns)
            .expect("valid derive vector");
        assert_eq!(*tv_key.as_bytes(), expected_key);
        let tv_class =
            tv_core::scoring::stream_class(&tv_key, bits, &[1, 2, 3]).expect("valid class vector");
        assert_eq!(tv_class, expected_class);
        let tv_draw = tv_core::scoring::sampler_draw(&tv_key, tv_class, 0);
        assert_eq!(prover_draw, expected_draw);
        assert_eq!(tv_draw, prover_draw);
        derive_rows += 1;
    }
    assert_eq!(derive_rows, 46);

    let mut prf_rows = 0usize;
    for line in include_str!("../../../spec/vectors/scoring/prf.txt")
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
    {
        let cols: Vec<&str> = line.split_whitespace().collect();
        let key_bytes: [u8; 16] = hex_bytes(cols[0]).try_into().unwrap();
        let bits: u8 = cols[1].parse().unwrap();
        let prompt: Vec<u32> = if cols[2] == "-" {
            Vec::new()
        } else {
            cols[2].split(',').map(|x| x.parse().unwrap()).collect()
        };
        let pos: u32 = cols[3].parse().unwrap();
        let expected_class: u64 = cols[4].parse().unwrap();
        let expected_draw: u32 = cols[6].parse().unwrap();

        let prover_key = SamplerKey::new(
            Secret::new(HeapKey::from_box(Box::new(key_bytes))),
            ClassBits::new(bits).unwrap(),
        );
        let prover_sampler = Sampler::new(&prover_key);
        let prover_draw =
            prover_sampler.draw(prover_sampler.class(&Secret::new(prompt.clone())), pos);

        let tv_key = tv_core::scoring::SamplerKey::from_bytes(key_bytes);
        let tv_class = tv_core::scoring::stream_class(&tv_key, bits, &prompt).unwrap();
        assert_eq!(tv_class, expected_class);
        let tv_draw = tv_core::scoring::sampler_draw(&tv_key, tv_class, pos);
        assert_eq!(prover_draw, expected_draw);
        assert_eq!(tv_draw, prover_draw);
        prf_rows += 1;
    }
    assert_eq!(prf_rows, 369);

    let mut cdf_rows = 0usize;
    for line in include_str!("../../../spec/vectors/scoring/inverse_cdf.txt")
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
    {
        let cols: Vec<&str> = line.split_whitespace().collect();
        let u: u32 = cols[0].parse().unwrap();
        let weights: Vec<u64> = cols[2..].iter().map(|w| w.parse().unwrap()).collect();
        let prover_res = inverse_cdf(&weights, u);
        let tv_res = tv_core::scoring::inverse_cdf(&weights, u);
        if cols[1] == "none" {
            assert!(prover_res.is_err());
            assert_eq!(tv_res, Err(tv_core::scoring::ScoringError::NoMass));
        } else {
            let expected: usize = cols[1].parse().unwrap();
            assert_eq!(prover_res, Ok(expected));
            assert_eq!(tv_res, Ok(expected));
        }
        cdf_rows += 1;
    }
    assert_eq!(cdf_rows, 228);
}

#[test]
fn actual_inverse_cdf_columns_cpu_proofs_and_tv_core_checks_match_independent_vectors() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../spec/vectors/gadget-finals/inverse-cdf.json"
    ))
    .unwrap();
    let range_specs: Vec<_> = (1..=32u8)
        .map(|bits| {
            GeneratedTable::new(TableId(u32::from(bits)), TableKind::Range { bits })
                .unwrap()
                .spec()
                .clone()
        })
        .collect();

    // 1. Differential schema verification across compiler, prover gadget, verifier Params, and oracle.
    let mut schema_count = 0usize;
    for sc in array(&fixture["schema_cases"]) {
        let vocab = sc["vocab"].as_u64().unwrap() as u32;
        let sigma = sc["sigma"].as_u64().unwrap() as u8;
        let bounded = sc["bounded"].as_bool().unwrap();
        let (comp_cols, comp_widths) = cdf_columns_and_requests(vocab, sigma, bounded).unwrap();
        let producer = InverseCdf::<G>::new(vocab as usize, sigma, &range_specs).unwrap();
        let tv_params = TvParams::derive(vocab, sigma, bounded).unwrap();

        assert_eq!(producer.width(), tv_params.w_bits());
        assert_eq!(producer.limbs(), tv_params.k());
        assert_eq!(comp_widths.as_slice(), tv_params.range_widths());
        let expected_col_len = tv_params.committed_row_count() + usize::from(bounded);
        assert_eq!(comp_cols.len(), expected_col_len);
        if bounded {
            assert_eq!(comp_cols[0], CdfAdviceColumn::SampledIndicator);
        }

        let tv_site = TvInverseCdf::derive(TvGadgetId(90), vocab, 3, sigma, bounded).unwrap();
        let row_target: Vec<Fp2> = array(&sc["point"])
            .iter()
            .map(|v| tv_fp2(field(v)))
            .collect();
        let committed_evals: Vec<Fp2> = array(&sc["committed_evals"])
            .iter()
            .map(|v| tv_fp2(field(v)))
            .collect();
        let range_keys: Vec<Fp2> = array(&sc["range_key_evals"])
            .iter()
            .map(|v| tv_fp2(field(v)))
            .collect();
        let checked = tv_site
            .check_row(TvRowValues {
                row_target: &row_target,
                u_eval: tv_fp2(field(&sc["u_eval"])),
                z_eval: tv_fp2(field(&sc["z_eval"])),
                b_eval: tv_fp2(field(&sc["b_eval"])),
                s_eval: tv_fp2(field(&sc["s_eval"])),
                committed_evals: Some(&committed_evals),
                range_keys: &range_keys,
                range_values: None,
            })
            .expect("honest schema case passes check_row");
        assert_eq!(checked.p_targets().len(), tv_params.m());
        schema_count += 1;
    }
    assert_eq!(schema_count, 6);

    // 2. Full witness, CPU 3-stage sumcheck proof, and tv_core end-to-end verification across all 17 cases.
    let inv_pow16: [G; 4] =
        core::array::from_fn(|j| G::from_u64_reduced(1u64 << (16 * j)).inverse().unwrap());
    let (mut honest_count, mut alt_count) = (0usize, 0usize);

    for case in array(&fixture["cases"]) {
        let name = case["name"].as_str().unwrap();
        let gadget_id = TvGadgetId(case["gadget"].as_u64().unwrap() as u32);
        let rows = case["rows"].as_u64().unwrap() as usize;
        let vocab = case["vocab"].as_u64().unwrap() as usize;
        let sigma = case["sigma"].as_u64().unwrap() as u8;
        let bounded = case["bounded"].as_bool().unwrap();
        let k = case["k"].as_u64().unwrap() as usize;
        let m = case["m"].as_u64().unwrap() as usize;

        let v_pad = vocab.next_power_of_two();
        let t_pad = rows.next_power_of_two();
        let vb = v_pad.trailing_zeros() as usize;
        let rb = t_pad.trailing_zeros() as usize;
        let total_vt = v_pad * t_pad;

        let producer = InverseCdf::<G>::new(vocab, sigma, &range_specs).unwrap();
        let onehot = OneHot::<G>::new(vocab).unwrap();
        let constraints = producer.constraints::<G>().unwrap();
        let tv_site =
            TvInverseCdf::derive(gadget_id, vocab as u32, rows as u32, sigma, bounded).unwrap();

        let local_reject = case["local_reject"].as_str();
        let failed_stage: Option<(usize, usize)> = case["failed_stage"].as_array().map(|arr| {
            (
                arr[0].as_u64().unwrap() as usize,
                arr[1].as_u64().unwrap() as usize,
            )
        });
        let is_honest = local_reject.is_none() && failed_stage.is_none();

        let exps_f: SecretBuf<G> = array(&case["exps"])
            .iter()
            .map(|v| G::from_i64(v.as_i64().unwrap()))
            .collect();
        let draws_f: SecretBuf<G> = array(&case["draws"])
            .iter()
            .map(|v| G::from_u64_reduced(v.as_u64().unwrap()))
            .collect();
        let h_col: SecretBuf<G> = array(&case["h_hat"])
            .iter()
            .map(|v| G::from_i64(v.as_i64().unwrap()))
            .collect();
        let l_col: SecretBuf<G> = array(&case["suffix"])
            .iter()
            .map(|v| G::from_i64(v.as_i64().unwrap()))
            .collect();

        // Check actual InverseCdf producer witness/evaluate behavior row by row.
        for (row_idx, row_rec) in array(&case["row_records"]).iter().enumerate() {
            let real = row_idx < rows;
            let row_e = &exps_f[row_idx * v_pad..(row_idx + 1) * v_pad];
            let u_val = draws_f[row_idx];
            let row_h = OneHotWitness::from_column(
                h_col[row_idx * v_pad..(row_idx + 1) * v_pad]
                    .iter()
                    .copied()
                    .collect(),
            );
            let evaluated = producer.evaluate(row_e, &row_h).unwrap();
            assert_eq!(
                &*evaluated.suffix,
                &array(&case["actual_suffix"])[row_idx * v_pad..(row_idx + 1) * v_pad]
                    .iter()
                    .map(|v| G::from_i64(v.as_i64().unwrap()))
                    .collect::<Vec<_>>()[..]
            );
            if is_honest {
                let sampled = producer.sample(row_e, u_val, real).unwrap();
                assert_eq!(sampled.indicator(), row_h.indicator());
                let expected_tok = onehot
                    .witness(case["tokens"][row_idx].as_i64().unwrap(), real)
                    .unwrap();
                assert_eq!(sampled.indicator(), expected_tok.indicator());
                let w = producer.witness(row_e, u_val, &sampled, real).unwrap();
                let expected_committed: Vec<i64> = array(&row_rec["committed"])
                    .iter()
                    .map(|v| v.as_i64().unwrap())
                    .collect();
                assert_eq!(
                    w.values(),
                    expected_committed.as_slice(),
                    "{name} row {row_idx}"
                );
                assert_eq!(w.columns(), constraints.committed.as_slice());
            } else if row_idx == 0 && matches!(local_reject, Some("lower_borrow" | "upper_borrow"))
            {
                let err = producer.witness(row_e, u_val, &row_h, real).unwrap_err();
                if name == "real-row-zero-mass" {
                    assert!(matches!(err, GadgetError::Input { .. }));
                } else {
                    assert_eq!(err, InverseCdf::<G>::MISMATCH, "{name}");
                }
            } else if row_idx == 0 && local_reject == Some("range") {
                assert!(matches!(
                    producer.witness(row_e, u_val, &row_h, real),
                    Err(GadgetError::Input { .. })
                ));
            }
        }

        let row_target_f: Vec<F> = array(&case["row_target"]).iter().map(field).collect();
        let row_target_tv: Vec<Fp2> = row_target_f.iter().copied().map(tv_fp2).collect();
        let u_eval_f = field(&case["u_eval"]);
        let z_eval_f = field(&case["z_eval"]);
        let b_eval_f = field(&case["b_eval"]);
        let s_eval_f = field(&case["s_eval"]);
        let range_keys_f: Vec<F> = array(&case["range_key_evals"]).iter().map(field).collect();
        let range_keys_tv: Vec<Fp2> = range_keys_f.iter().copied().map(tv_fp2).collect();

        let row_check_res = tv_site.check_row(TvRowValues {
            row_target: &row_target_tv,
            u_eval: tv_fp2(u_eval_f),
            z_eval: tv_fp2(z_eval_f),
            b_eval: tv_fp2(b_eval_f),
            s_eval: tv_fp2(s_eval_f),
            committed_evals: None,
            range_keys: &range_keys_tv,
            range_values: None,
        });

        if let Some(reason) = local_reject {
            alt_count += 1;
            match reason {
                "lower_borrow" | "upper_borrow" => {
                    assert_eq!(
                        row_check_res.unwrap_err(),
                        TvGadgetError::Rejected { gadget: gadget_id },
                        "{name}"
                    );
                }
                "range" => {
                    // `draw-overflow` has u_1 = 65536 outside Range(16); row_records[0] fails Range(16).
                    let table16 =
                        GeneratedTable::new(TableId(16), TableKind::Range { bits: 16 }).unwrap();
                    let u1_key = case["row_records"][0]["range_keys"][1].as_i64().unwrap();
                    assert_eq!(table16.lookup(u1_key, 0), None);
                }
                other => panic!("unexpected local_reject {other}"),
            }
            continue;
        }

        let checked_for_omission = row_check_res.expect("row equations hold at row_target");
        let record = if failed_stage.is_some() {
            &case["omission"]
        } else {
            &case["proof"]
        };

        // Build row columns for CPU prove_batch Stage 1.
        let u_col: SecretBuf<G> = array(&case["row_records"])
            .iter()
            .map(|r| G::from_u64_reduced(r["u"].as_u64().unwrap()))
            .collect();
        let z_col: SecretBuf<G> = array(&case["row_records"])
            .iter()
            .map(|r| G::from_i64(r["z"].as_i64().unwrap()))
            .collect();
        let u0_col: SecretBuf<G> = array(&case["row_records"])
            .iter()
            .map(|r| G::from_i64(r["committed"][0].as_i64().unwrap()))
            .collect();
        let z_low_cols: Vec<SecretBuf<G>> = (0..k - 1)
            .map(|h| {
                array(&case["row_records"])
                    .iter()
                    .map(|r| G::from_i64(r["committed"][1 + h].as_i64().unwrap()))
                    .collect()
            })
            .collect();
        let not_psi: Vec<G> = (0..t_pad)
            .map(|r| G::from_u64_reduced(u64::from(r >= rows)))
            .collect();
        let origin_row: Vec<G> = (0..t_pad)
            .map(|r| G::from_u64_reduced(u64::from(r == 0)))
            .collect();
        let p_targets_f: Vec<F> = array(&case["p_targets"]).iter().map(field).collect();

        let (mut ours, mut theirs) = transcripts(&case["context"]);
        let mut split_msg = vec![u_eval_f, z_eval_f, b_eval_f, s_eval_f];
        split_msg.extend_from_slice(&range_keys_f);
        ours.absorb("stage/0/split", &split_msg);
        absorb_fp2_slice(&mut theirs, "stage/0/split", &split_msg);

        let zero_row: Vec<F> = (0..rb)
            .map(|j| ours.challenge_fp2(&format!("stage/1/zero/{j}")))
            .collect();
        let eq_rt: Vec<F> = (0..t_pad).map(|r| eq_at(&row_target_f, r)).collect();
        let eq_zr: Vec<F> = (0..t_pad).map(|r| eq_at(&zero_row, r)).collect();

        let s1_omit = failed_stage.and_then(|(st, idx)| (st == 1).then_some(idx));
        let mut s1_instances = Vec::with_capacity(m + 1);
        for (j, &p_target_j) in p_targets_f.iter().enumerate().take(m) {
            let mut tables = vec![
                Table::ext(&eq_rt).public(),
                Table::base(&u_col),
                Table::base(&u0_col),
                Table::base(&z_col),
            ];
            for col in &z_low_cols {
                tables.push(Table::base(col));
            }
            let origin_idx = tables.len();
            tables.push(Table::base(&origin_row).public());
            let terms = if s1_omit == Some(j) {
                vec![Term::new(p_target_j, vec![origin_idx])]
            } else {
                // Expand u_i * Z_h over table indices:
                // u_0 = [2], u_1 = ([1] - [2]) * inv_pow16[1]
                // Z_h (h < k-1) = [4 + h], Z_{k-1} = ([3] - sum_{l<k-1} 2^{16l} [4+l]) * inv_pow16[k-1]
                let u_exprs: [Vec<(F, usize)>; 2] = [
                    vec![(F::ONE, 2)],
                    vec![(F::from(inv_pow16[1]), 1), (-F::from(inv_pow16[1]), 2)],
                ];
                let z_expr = |h: usize| -> Vec<(F, usize)> {
                    if h + 1 < k {
                        vec![(F::ONE, 4 + h)]
                    } else {
                        let scale_top = F::from(inv_pow16[k - 1]);
                        let mut v = vec![(scale_top, 3)];
                        for l in 0..k - 1 {
                            let shift = F::from(G::from_u64_reduced(1u64 << (16 * l)));
                            v.push((-scale_top * shift, 4 + l));
                        }
                        v
                    }
                };
                let mut acc = Vec::new();
                for (i, u_expr) in u_exprs.iter().enumerate() {
                    for h in 0..k {
                        if i + h == j {
                            for &(cu, iu) in u_expr {
                                for &(cz, iz) in &z_expr(h) {
                                    acc.push(Term::new(cu * cz, vec![0, iu, iz]));
                                }
                            }
                        }
                    }
                }
                if acc.is_empty() {
                    acc.push(Term::new(F::ZERO, vec![origin_idx]));
                }
                acc
            };
            s1_instances.push(Instance::sum_of_products(tables, terms));
        }
        s1_instances.push(Instance::sum_of_products(
            vec![
                Table::ext(&eq_zr).public(),
                Table::base(&not_psi).public(),
                Table::base(&z_col),
                Table::base(&origin_row).public(),
            ],
            if s1_omit == Some(m) {
                vec![Term::new(F::ZERO, vec![3])]
            } else {
                vec![Term::new(F::ONE, vec![0, 1, 2])]
            },
        ));

        let s1_proven =
            prove_batch::<F, _>(&mut ours, Protocol::One, Prefix::Stage(1), 3, &s1_instances)
                .unwrap();
        check_stage_record(&s1_proven, &ours, &record["stage1"]);

        // Stage 2 over (vocab, row)
        let r1 = s1_proven.point().to_vec();
        let z_r1 = *s1_proven.proof().final_evals.last().unwrap();
        let zero_vt: Vec<F> = if bounded {
            (0..vb + rb)
                .map(|j| ours.challenge_fp2(&format!("stage/2/zero/{j}")))
                .collect()
        } else {
            Vec::new()
        };

        let eq_rt_vt: Vec<F> = (0..total_vt)
            .map(|i| eq_at(&row_target_f, i / v_pad))
            .collect();
        let eq_r1_vt: Vec<F> = (0..total_vt).map(|i| eq_at(&r1, i / v_pad)).collect();
        let valid_vt: Vec<G> = (0..total_vt)
            .map(|i| G::from_u64_reduced(u64::from(i % v_pad < vocab)))
            .collect();
        let origin_vt: Vec<G> = (0..total_vt)
            .map(|i| G::from_u64_reduced(u64::from(i == 0)))
            .collect();
        let eq_zvt: Vec<F> = if bounded {
            (0..total_vt).map(|i| eq_at(&zero_vt, i)).collect()
        } else {
            Vec::new()
        };
        let eq_zrow_vt: Vec<F> = if bounded {
            (0..total_vt)
                .map(|i| eq_at(&zero_vt[vb..], i / v_pad))
                .collect()
        } else {
            Vec::new()
        };
        let psi_zrow: F = if bounded {
            (0..t_pad)
                .filter(|&r| r < rows)
                .fold(F::ZERO, |acc, r| acc + eq_at(&zero_vt[vb..], r))
        } else {
            F::ZERO
        };

        let s2_omit = failed_stage.and_then(|(st, idx)| (st == 2).then_some(idx));
        let mut s2_instances = vec![
            Instance::sum_of_products(
                vec![
                    Table::ext(&eq_rt_vt).public(),
                    Table::base(&valid_vt).public(),
                    Table::base(&exps_f),
                    Table::base(&origin_vt).public(),
                ],
                if s2_omit == Some(0) {
                    vec![Term::new(z_eval_f, vec![3])]
                } else {
                    vec![Term::new(F::ONE, vec![0, 1, 2])]
                },
            ),
            Instance::sum_of_products(
                vec![
                    Table::ext(&eq_r1_vt).public(),
                    Table::base(&valid_vt).public(),
                    Table::base(&exps_f),
                    Table::base(&origin_vt).public(),
                ],
                if s2_omit == Some(1) {
                    vec![Term::new(z_r1, vec![3])]
                } else {
                    vec![Term::new(F::ONE, vec![0, 1, 2])]
                },
            ),
            Instance::sum_of_products(
                vec![
                    Table::ext(&eq_rt_vt).public(),
                    Table::base(&exps_f),
                    Table::base(&l_col),
                    Table::base(&origin_vt).public(),
                ],
                if s2_omit == Some(2) {
                    vec![Term::new(b_eval_f, vec![3])]
                } else {
                    vec![Term::new(F::ONE, vec![0, 1, 2])]
                },
            ),
            Instance::sum_of_products(
                vec![
                    Table::ext(&eq_rt_vt).public(),
                    Table::base(&h_col),
                    Table::base(&exps_f),
                    Table::base(&origin_vt).public(),
                ],
                if s2_omit == Some(3) {
                    vec![Term::new(s_eval_f, vec![3])]
                } else {
                    vec![Term::new(F::ONE, vec![0, 1, 2])]
                },
            ),
        ];
        if bounded {
            s2_instances.push(Instance::sum_of_products(
                vec![
                    Table::ext(&eq_zvt).public(),
                    Table::base(&h_col),
                    Table::base(&origin_vt).public(),
                ],
                if s2_omit == Some(4) {
                    vec![Term::new(F::ZERO, vec![2])]
                } else {
                    vec![
                        Term::new(F::ONE, vec![0, 1]),
                        Term::new(-F::ONE, vec![0, 1, 1]),
                    ]
                },
            ));
            s2_instances.push(Instance::sum_of_products(
                vec![
                    Table::ext(&eq_zvt).public(),
                    Table::base(&valid_vt).public(),
                    Table::base(&h_col),
                    Table::base(&origin_vt).public(),
                ],
                if s2_omit == Some(5) {
                    vec![Term::new(F::ZERO, vec![3])]
                } else {
                    vec![
                        Term::new(F::ONE, vec![0, 2]),
                        Term::new(-F::ONE, vec![0, 1, 2]),
                    ]
                },
            ));
            s2_instances.push(Instance::sum_of_products(
                vec![
                    Table::ext(&eq_zrow_vt).public(),
                    Table::base(&h_col),
                    Table::base(&origin_vt).public(),
                ],
                if s2_omit == Some(6) {
                    vec![Term::new(psi_zrow, vec![2])]
                } else {
                    vec![Term::new(F::ONE, vec![0, 1])]
                },
            ));
        }

        let s2_proven =
            prove_batch::<F, _>(&mut ours, Protocol::One, Prefix::Stage(2), 3, &s2_instances)
                .unwrap();
        check_stage_record(&s2_proven, &ours, &record["stage2"]);

        // Stage 3: strict-suffix kernel over (vocab, row)
        let r2 = s2_proven.point().to_vec();
        let l_r2 = s2_proven.proof().final_evals[3];
        let prefix_weights = producer.suffix_weights::<F>(&r2[..vb]).unwrap();
        let s3_weights: Vec<F> = (0..total_vt)
            .map(|i| eq_at(&r2[vb..], i / v_pad) * prefix_weights[i % v_pad])
            .collect();
        let s3_omit = failed_stage.and_then(|(st, idx)| (st == 3).then_some(idx));
        let s3_instance = Instance::sum_of_products(
            vec![
                Table::ext(&s3_weights).public(),
                Table::base(&h_col),
                Table::base(&origin_vt).public(),
            ],
            if s3_omit == Some(0) {
                vec![Term::new(l_r2, vec![2])]
            } else {
                vec![Term::new(F::ONE, vec![0, 1])]
            },
        );
        let s3_proven = prove_batch::<F, _>(
            &mut ours,
            Protocol::One,
            Prefix::Stage(3),
            2,
            &[s3_instance],
        )
        .unwrap();
        check_stage_record(&s3_proven, &ours, &record["stage3"]);

        // Now verify the CPU proofs directly with tv_core::gadgets::inverse_cdf::InverseCdf!
        let s1_tv = checked_for_omission.squeeze_stage1(&mut theirs).unwrap();
        let s1_tv_proof = to_tv_proof(&record["stage1"]);
        let s2_tv_proof = to_tv_proof(&record["stage2"]);
        let s3_tv_proof = to_tv_proof(&record["stage3"]);

        if let Some((fail_stage, _)) = failed_stage {
            alt_count += 1;
            // Omission proof passes every sumcheck round up to `fail_stage`'s final evaluation check!
            match fail_stage {
                1 => {
                    let err = s1_tv.verify(&mut theirs, &s1_tv_proof).unwrap_err();
                    assert_eq!(
                        err,
                        StageVerifyError::Sumcheck(SumcheckError::FinalMismatch {
                            round: rb as u32,
                        }),
                        "{name}"
                    );
                }
                2 => {
                    let (_, done1) = s1_tv.verify(&mut theirs, &s1_tv_proof).unwrap();
                    let s2_tv = done1.squeeze_stage2(&mut theirs).unwrap();
                    let err = s2_tv.verify(&mut theirs, &s2_tv_proof).unwrap_err();
                    assert_eq!(
                        err,
                        StageVerifyError::Sumcheck(SumcheckError::FinalMismatch {
                            round: (vb + rb) as u32,
                        }),
                        "{name}"
                    );
                }
                3 => {
                    let (_, done1) = s1_tv.verify(&mut theirs, &s1_tv_proof).unwrap();
                    let s2_tv = done1.squeeze_stage2(&mut theirs).unwrap();
                    let (_, s3_tv) = s2_tv.verify(&mut theirs, &s2_tv_proof).unwrap();
                    let err = s3_tv.verify(&mut theirs, &s3_tv_proof).unwrap_err();
                    assert_eq!(
                        err,
                        StageVerifyError::Sumcheck(SumcheckError::FinalMismatch {
                            round: (vb + rb) as u32,
                        }),
                        "{name}"
                    );
                }
                other => panic!("unexpected fail_stage {other}"),
            }
        } else {
            honest_count += 1;
            let (_, done1) = s1_tv.verify(&mut theirs, &s1_tv_proof).unwrap();
            let s2_tv = done1.squeeze_stage2(&mut theirs).unwrap();
            let (_, s3_tv) = s2_tv.verify(&mut theirs, &s2_tv_proof).unwrap();
            let (_, verified) = s3_tv.verify(&mut theirs, &s3_tv_proof).unwrap();
            assert_eq!(verified.range_claims().len(), range_keys_tv.len());
            assert_eq!(theirs.state(), &ours.state());
            assert_eq!(theirs.counter(), ours.counter());
        }
    }
    assert_eq!((honest_count, alt_count), (6, 11));
}
