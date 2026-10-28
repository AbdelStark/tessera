// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Differential adapter tests (`DEP-2`) joining `tessera-gadgets`, `tessera-compiler` and
//! `tessera-sumcheck` to `tv-core::scoring` and `tv-core::gadgets::scoring` over
//! `spec/vectors/gadget-finals/one-sided-log2.json` and `spec/vectors/models/tiny-2-servable.d`
//! (`#160`, RFC-0017, RFC-0019, RFC-0030, RFC-0035, RFC-0050, RFC-0058).

use std::fs;
use std::path::Path;

use serde_json::Value;
use tessera_codec::{Ascii, DecodeLimits, Rational as ProverRational, SecretBuf, envelope, schema};
use tessera_compiler::{
    MixtureDecoding, MixturePlan, MixtureRequestSpec, PublicModel, TemperaturePlan,
    TemperatureRequestSpec, analyze, claim_list, log2_columns_and_requests, lower, select,
    vocab_exp_scale,
};
use tessera_field::{Field, Goldilocks as G, GoldilocksExt2 as F, PrimeField, QuadraticExtension};
use tessera_gadgets::{
    GadgetError, Log2BatchColumns, OneHot, OneHotWitness, OneSidedLog2, VocabMax, VocabWitness,
    c_match,
};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_ir::{
    AdviceLayoutParams, CommitmentLayout, GadgetPolicy, IntegerModel, ParamSetId as IrParamSetId,
    WeightLayoutParams,
};
use tessera_sumcheck::{Instance, Prefix, Protocol, Table, Term, prove_batch};
use tessera_tables::{GeneratedTable, Table as _, TableId, TableKind};
use tv_codec::Rational as TvRational;
use tv_core::gadgets::GadgetError as TvGadgetError;
use tv_core::gadgets::inverse_cdf::StageVerifyError;
use tv_core::gadgets::scoring::{
    InternConfig, Log2Params as TvLog2Params, Log2RowValues as TvLog2RowValues,
    OneSidedLog2 as TvOneSidedLog2, ScoringMode, ScoringRequest, VocabMax as TvVocabMax,
    intern_scoring_tables,
};
use tv_core::protocol::{Decoding as TvDecoding, FixedBits, PredictorSpec, RequestRef};
use tv_core::scoring::{UhatError, check_uhat_bound};
use tv_core::sumcheck::{SumcheckError, SumcheckProof as TvSumcheckProof};
use tv_core::validation::GadgetId as TvGadgetId;
use tv_field::{Fp, Fp2};
use tv_tables::{TableId as TvTableId, TableKind as TvTableKind};

const TEST_SMALL: CommitmentLayout = CommitmentLayout {
    advice: AdviceLayoutParams {
        log2_k: 10,
        log2_inv_rate: 1,
        mask_coeffs: 16,
        row_block: 4,
        segments: 1,
    },
    weights: WeightLayoutParams {
        log2_k: 8,
        log2_inv_rate: 2,
        mask_coeffs: 64,
        blinding_rows: 32,
    },
};

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

fn prover_rational(v: &Value) -> ProverRational {
    ProverRational::new(v[0].as_i64().unwrap(), v[1].as_u64().unwrap()).unwrap()
}

fn tv_rational(v: &Value) -> TvRational {
    TvRational::new(v[0].as_i64().unwrap(), v[1].as_u64().unwrap()).unwrap()
}

fn to_tv_rational(r: ProverRational) -> TvRational {
    TvRational::new(r.num(), r.den()).unwrap()
}

fn to_tv_table_kind(k: &TableKind) -> TvTableKind {
    match k {
        TableKind::Range { bits } => TvTableKind::Range { bits: *bits },
        TableKind::Requant {
            b,
            r_by_class,
            lo,
            hi,
        } => TvTableKind::Requant {
            b: *b,
            r_by_class: r_by_class.clone(),
            lo: *lo,
            hi: *hi,
        },
        TableKind::Pow2 { max_exp } => TvTableKind::Pow2 { max_exp: *max_exp },
        TableKind::Bitwise8 { op } => TvTableKind::Bitwise8 {
            op: match op {
                tessera_tables::BitOp::Xor => tv_tables::BitOp::Xor,
                tessera_tables::BitOp::And => tv_tables::BitOp::And,
            },
        },
        TableKind::AesSbox => TvTableKind::AesSbox,
        TableKind::Gf256Mul { c } => TvTableKind::Gf256Mul { c: *c },
        TableKind::Exp {
            key_bits,
            sigma,
            scale,
        } => TvTableKind::Exp {
            key_bits: *key_bits,
            sigma: *sigma,
            scale: to_tv_rational(*scale),
        },
        TableKind::Exp2Frac {
            frac_bits,
            out_bits,
        } => TvTableKind::Exp2Frac {
            frac_bits: *frac_bits,
            out_bits: *out_bits,
        },
        TableKind::Silu {
            b,
            r_by_class,
            in_exp,
            out_exp,
        } => TvTableKind::Silu {
            b: *b,
            r_by_class: r_by_class.clone(),
            in_exp: *in_exp,
            out_exp: *out_exp,
        },
        TableKind::Rope {
            theta,
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max,
            bits,
            max_context,
            d_head,
        } => TvTableKind::Rope {
            theta: *theta,
            factor: *factor,
            low_freq_factor: *low_freq_factor,
            high_freq_factor: *high_freq_factor,
            original_max: *original_max,
            bits: *bits,
            max_context: *max_context,
            d_head: *d_head,
        },
    }
}

fn parse_prover_table_kind(v: &Value) -> TableKind {
    let obj = v.as_object().unwrap();
    assert_eq!(obj.len(), 1);
    let (tag, body) = obj.iter().next().unwrap();
    match tag.as_str() {
        "Range" => TableKind::Range {
            bits: body["bits"].as_u64().unwrap() as u8,
        },
        "Pow2" => TableKind::Pow2 {
            max_exp: body["max_exp"].as_u64().unwrap() as u8,
        },
        "Exp2Frac" => TableKind::Exp2Frac {
            frac_bits: body["frac_bits"].as_u64().unwrap() as u8,
            out_bits: body["out_bits"].as_u64().unwrap() as u8,
        },
        "Exp" => TableKind::Exp {
            key_bits: body["key_bits"].as_u64().unwrap() as u8,
            sigma: body["sigma"].as_u64().unwrap() as u8,
            scale: prover_rational(&body["scale"]),
        },
        other => panic!("unexpected TableKind {other}"),
    }
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
fn c_match_vocab_max_and_runtime_table_interning_match_across_prover_verifier_and_vectors() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../spec/vectors/gadget-finals/one-sided-log2.json"
    ))
    .unwrap();

    // 1. Differential c_match(0..=16) and out-of-domain k > 16.
    let expected_cmatch: Vec<u64> = array(&fixture["c_match"])
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    assert_eq!(expected_cmatch.len(), 17);
    for k in 0..=16u8 {
        let p_cm = c_match(k).unwrap();
        let v_cm = tv_core::scoring::c_match(k).unwrap();
        assert_eq!(p_cm, expected_cmatch[usize::from(k)]);
        assert_eq!(v_cm, p_cm);
    }
    assert!(c_match(17).is_err());
    assert!(tv_core::scoring::c_match(17).is_err());

    // 2. Differential VocabMax: witness, CPU sumcheck proof, and tv_core::gadgets::scoring::VocabMax.
    let (mut vmax_honest, mut vmax_alt) = (0usize, 0usize);
    for case in array(&fixture["vocab_max_cases"]) {
        let name = case["name"].as_str().unwrap();
        let gadget = TvGadgetId(case["gadget"].as_u64().unwrap() as u32);
        let rows = case["rows"].as_u64().unwrap() as usize;
        let vocab = case["vocab"].as_u64().unwrap() as usize;
        let logits_exp = case["logits_exponent"].as_i64().unwrap() as i8;
        let p_temp = prover_rational(&case["temperature"]);
        let v_temp = tv_rational(&case["temperature"]);
        let sigma = case["sigma"].as_u64().unwrap() as u8;
        let lo = case["logit_lo"].as_i64().unwrap() as i32;
        let hi = case["logit_hi"].as_i64().unwrap() as i32;
        let max_log2 = case["max_log2_rows"].as_u64().unwrap() as u8;
        let key_bits = case["key_bits"].as_u64().unwrap() as u8;
        let failed = case["failed"].as_u64().map(|x| x as usize);

        let p_scale = vocab_exp_scale(logits_exp, p_temp).unwrap();
        let v_scale = tv_core::scoring::vocab_exp_scale(logits_exp, v_temp).unwrap();
        assert_eq!(to_tv_rational(p_scale), v_scale, "{name}");

        let exp_gen = GeneratedTable::new(
            TableId(0),
            TableKind::Exp {
                key_bits,
                sigma,
                scale: p_scale,
            },
        )
        .unwrap();
        let producer = VocabMax::<G>::new(vocab, exp_gen.spec()).unwrap();
        assert_eq!(producer.masked_key(), case["k0"].as_u64().unwrap());

        let tv_site = TvVocabMax::derive(
            gadget,
            vocab as u32,
            rows as u32,
            logits_exp,
            v_temp,
            sigma,
            lo,
            hi,
            max_log2,
        )
        .unwrap();
        assert_eq!(tv_site.key_bits(), key_bits, "{name}");
        assert_eq!(tv_site.k0(), producer.masked_key(), "{name}");

        let vocab_vars = (usize::BITS - (vocab - 1).leading_zeros()) as usize;
        let row_vars = (usize::BITS - (rows - 1).leading_zeros()) as usize;
        let v_pad = 1usize << vocab_vars;
        let t_pad = 1usize << row_vars;
        let total = v_pad * t_pad;

        let logits: SecretBuf<G> = array(&case["logits"])
            .iter()
            .map(|v| G::from_i64(v.as_i64().unwrap()))
            .collect();
        let maxima: SecretBuf<G> = array(&case["maxima"])
            .iter()
            .map(|v| G::from_i64(v.as_i64().unwrap()))
            .collect();
        let exps: SecretBuf<G> = array(&case["exps"])
            .iter()
            .map(|v| G::from_i64(v.as_i64().unwrap()))
            .collect();

        if failed.is_none() {
            let mut honest_max = SecretBuf::with_capacity(t_pad);
            let mut honest_exp = SecretBuf::with_capacity(total);
            for r in 0..t_pad {
                let row_w = producer
                    .witness(&logits[r * v_pad..(r + 1) * v_pad], r < rows)
                    .unwrap();
                for &m in row_w.maximum() {
                    honest_max.push(m);
                }
                for &e in row_w.exponentials() {
                    honest_exp.push(e);
                }
            }
            assert_eq!(&honest_max[..], &maxima[..], "{name}");
            assert_eq!(&honest_exp[..], &exps[..], "{name}");
            let evaluated = producer
                .evaluate_batch(
                    &logits,
                    t_pad,
                    rows,
                    &VocabWitness::from_columns(honest_max, honest_exp),
                )
                .unwrap();
            let expected_keys: Vec<G> = array(&case["keys"])
                .iter()
                .map(|v| G::from_i64(v.as_i64().unwrap()))
                .collect();
            let expected_leaves: Vec<G> = array(&case["leaves"])
                .iter()
                .map(|v| G::from_i64(v.as_i64().unwrap()))
                .collect();
            assert_eq!(&evaluated.keys[..], expected_keys.as_slice(), "{name}");
            assert_eq!(&evaluated.leaves[..], expected_leaves.as_slice(), "{name}");
            assert!(evaluated.padding.iter().all(|&p| p == G::ZERO));
        }

        let lookup_target: Vec<F> = array(&case["lookup_target"]).iter().map(field).collect();
        let tree_target: Vec<F> = array(&case["tree_target"]).iter().map(field).collect();
        let forest_root_target: Vec<F> = array(&case["forest_root_target"])
            .iter()
            .map(field)
            .collect();
        let lookup_key = field(&case["lookup_key"]);
        let lookup_value = field(&case["lookup_value"]);
        let tree_leaf = field(&case["tree_leaf"]);
        let forest_root_claim = field(&case["forest_root_claim"]);
        let claims: Vec<F> = array(&case["claims"]).iter().map(field).collect();

        let record = if failed.is_some() {
            &case["omission"]
        } else {
            &case["proof"]
        };

        let (mut ours, mut theirs) = transcripts(&case["context"]);
        let initial = vec![lookup_key, lookup_value, tree_leaf];
        ours.absorb("stage/0/split", &initial);
        absorb_fp2_slice(&mut theirs, "stage/0/split", &initial);

        let zero_row: Vec<F> = (0..row_vars)
            .map(|j| ours.challenge_fp2(&format!("stage/1/zero/{j}")))
            .collect();
        assert_eq!(
            zero_row,
            array(&record["zero_row"])
                .iter()
                .map(field)
                .collect::<Vec<_>>()
        );

        let lookup_eq: Vec<F> = (0..total).map(|i| eq_at(&lookup_target, i)).collect();
        let tree_eq: Vec<F> = (0..total).map(|i| eq_at(&tree_target, i)).collect();
        let zero_eq: Vec<F> = (0..t_pad).map(|i| eq_at(&zero_row, i)).collect();
        let selector_vt: Vec<G> = array(&case["selector"])
            .iter()
            .map(|v| G::from_u64_reduced(v.as_u64().unwrap()))
            .collect();
        let not_psi: Vec<G> = array(&case["psi"])
            .iter()
            .map(|v| G::ONE - G::from_u64_reduced(v.as_u64().unwrap()))
            .collect();
        let broadcast_max: SecretBuf<G> = (0..total).map(|i| maxima[i / v_pad]).collect();
        let origin_vt: Vec<G> = (0..total)
            .map(|i| G::from_u64_reduced(u64::from(i == 0)))
            .collect();
        let origin_row: Vec<G> = (0..t_pad)
            .map(|i| G::from_u64_reduced(u64::from(i == 0)))
            .collect();

        let instances = [
            Instance::sum_of_products(
                vec![
                    Table::ext(&lookup_eq).public(),
                    Table::base(&selector_vt).public(),
                    Table::base(&broadcast_max),
                    Table::base(&logits),
                    Table::base(&origin_vt).public(),
                ],
                if failed == Some(0) {
                    vec![Term::new(claims[0], vec![4])]
                } else {
                    vec![
                        Term::new(F::ONE, vec![0, 1, 2]),
                        Term::new(-F::ONE, vec![0, 1, 3]),
                    ]
                },
            ),
            Instance::sum_of_products(
                vec![
                    Table::ext(&tree_eq).public(),
                    Table::base(&selector_vt).public(),
                    Table::base(&broadcast_max),
                    Table::base(&logits),
                    Table::base(&origin_vt).public(),
                ],
                if failed == Some(1) {
                    vec![Term::new(claims[1], vec![4])]
                } else {
                    vec![
                        Term::new(F::ONE, vec![0, 1, 2]),
                        Term::new(-F::ONE, vec![0, 1, 3]),
                    ]
                },
            ),
            Instance::sum_of_products(
                vec![
                    Table::ext(&zero_eq).public(),
                    Table::base(&not_psi).public(),
                    Table::base(&maxima),
                    Table::base(&origin_row).public(),
                ],
                if failed == Some(2) {
                    vec![Term::new(claims[2], vec![3])]
                } else {
                    vec![Term::new(F::ONE, vec![0, 1, 2])]
                },
            ),
        ];

        let proven =
            prove_batch::<F, _>(&mut ours, Protocol::One, Prefix::Stage(1), 3, &instances).unwrap();
        check_stage_record(&proven, &ours, record);

        let lookup_target_tv: Vec<Fp2> = lookup_target.iter().copied().map(tv_fp2).collect();
        let tree_target_tv: Vec<Fp2> = tree_target.iter().copied().map(tv_fp2).collect();
        let root_target_tv: Vec<Fp2> = forest_root_target.iter().copied().map(tv_fp2).collect();
        let checked_tv = tv_site
            .check_claims(
                &lookup_target_tv,
                tv_fp2(lookup_key),
                tv_fp2(lookup_value),
                &tree_target_tv,
                tv_fp2(tree_leaf),
                &root_target_tv,
                tv_fp2(forest_root_claim),
            )
            .unwrap();
        let s1_tv = checked_tv.squeeze_stage1(&mut theirs).unwrap();
        let tv_proof = to_tv_proof(record);
        if failed.is_some() {
            vmax_alt += 1;
            assert_eq!(
                s1_tv.verify(&mut theirs, &tv_proof).unwrap_err(),
                StageVerifyError::Sumcheck(SumcheckError::FinalMismatch {
                    round: (vocab_vars + row_vars) as u32,
                }),
                "{name}"
            );
        } else {
            vmax_honest += 1;
            let (_, verified) = s1_tv.verify(&mut theirs, &tv_proof).unwrap();
            assert_eq!(verified.lookup_claim().point, lookup_target_tv);
            assert_eq!(theirs.state(), &ours.state());
            assert_eq!(theirs.counter(), ours.counter());
        }
    }
    assert_eq!((vmax_honest, vmax_alt), (3, 3));

    // 3. Differential runtime table interning on tiny-2-servable.d across MixturePlan, TemperaturePlan,
    //    and tv_core::gadgets::scoring::intern_scoring_tables.
    let model_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/models/tiny-2-servable.d");
    let model: IntegerModel = envelope::open(
        &fs::read(model_path).unwrap(),
        &schema::INTEGER_MODEL,
        DecodeLimits::INTEGER_MODEL,
    )
    .unwrap();
    let public = PublicModel::of(&model);
    let lowering = lower(public, 6).unwrap();
    let ranges = analyze(&lowering.graph, &lowering.tables).unwrap();
    let logit_range = ranges.of(lowering.logits).unwrap().bounded().unwrap();
    let policy = GadgetPolicy::default();
    let selection = select(&lowering.graph, &ranges, &lowering.tables, &policy).unwrap();
    let claims = claim_list(
        public,
        &selection,
        &policy,
        TEST_SMALL,
        IrParamSetId(Ascii::new("test-small/1").unwrap()),
    )
    .unwrap();
    let base_tv_tables: Vec<TvTableKind> = claims
        .tables
        .iter()
        .map(|s| to_tv_table_kind(&s.kind))
        .collect();

    // 3a. MixturePlan vs intern_scoring_tables (Greedy + zero-output + 3 temperatures).
    let p_t07 = ProverRational::new(7, 10).unwrap();
    let p_t15 = ProverRational::new(3, 2).unwrap();
    let p_t40 = ProverRational::new(4, 1).unwrap();
    let mix_reqs = [
        MixtureRequestSpec {
            prompt: 2,
            positions: 5,
            decoding: MixtureDecoding::Greedy,
        },
        MixtureRequestSpec {
            prompt: 4,
            positions: 4,
            decoding: MixtureDecoding::Temperature(p_t07),
        },
        MixtureRequestSpec {
            prompt: 1,
            positions: 4,
            decoding: MixtureDecoding::Temperature(p_t07),
        },
        MixtureRequestSpec {
            prompt: 3,
            positions: 6,
            decoding: MixtureDecoding::Temperature(p_t15),
        },
        MixtureRequestSpec {
            prompt: 2,
            positions: 4,
            decoding: MixtureDecoding::Temperature(p_t40),
        },
    ];
    let mix_plan = MixturePlan::derive(&lowering, &selection, &claims, &mix_reqs, 6, 0).unwrap();
    let tv_mix_reqs: Vec<ScoringRequest> = mix_reqs
        .iter()
        .map(|r| ScoringRequest {
            outputs: r.positions - r.prompt,
            decoding: match r.decoding {
                MixtureDecoding::Greedy => TvDecoding::Greedy,
                MixtureDecoding::Temperature(t) => TvDecoding::Temperature {
                    t: to_tv_rational(t),
                },
            },
        })
        .collect();
    let tv_mix = intern_scoring_tables(InternConfig {
        base_tables: &base_tv_tables,
        vocab: claims.arch.vocab,
        sigma: claims.quant.exp_sigma,
        logits_exponent: claims.quant.logits_exponent,
        logit_lo: logit_range.lo as i32,
        logit_hi: logit_range.hi as i32,
        max_log2_rows: claims.gadget_policy.max_log2_rows,
        mode: ScoringMode::Mixture { eps_log2: 6 },
        requests: &tv_mix_reqs,
    })
    .unwrap();
    let mix_plan_kinds: Vec<TvTableKind> = mix_plan
        .selection()
        .tables
        .iter()
        .map(|s| to_tv_table_kind(&s.kind))
        .collect();
    assert_eq!(tv_mix.catalog, mix_plan_kinds);
    assert_eq!(tv_mix.exp_tables.len(), mix_plan.temperatures().len());
    for (tv_e, p_e) in tv_mix.exp_tables.iter().zip(mix_plan.temperatures()) {
        assert_eq!(tv_e.table_id, TvTableId(p_e.table.0));
        assert_eq!(tv_e.temperature, to_tv_rational(p_e.temperature));
    }
    for cert in mix_plan.certificates() {
        let rt = &tv_mix.request_tables[cert.request_index];
        let expected_dec: Vec<TvTableId> = match cert.decoding {
            MixtureDecoding::Greedy => vec![TvTableId(cert.argmax_range_table.unwrap().0)],
            MixtureDecoding::Temperature(_) => cert
                .cdf_range_tables
                .iter()
                .map(|id| TvTableId(id.0))
                .collect(),
        };
        let expected_l2: Vec<TvTableId> =
            cert.log2_tables.iter().map(|id| TvTableId(id.0)).collect();
        assert_eq!(rt.decoding_ids, expected_dec);
        assert_eq!(rt.log2_ids, expected_l2);
    }

    // 3b. TemperaturePlan (Exact mode) vs intern_scoring_tables.
    let temp_reqs = [
        TemperatureRequestSpec {
            prompt: 2,
            positions: 5,
            temperature: p_t07,
            bounded: false,
        },
        TemperatureRequestSpec {
            prompt: 3,
            positions: 6,
            temperature: p_t15,
            bounded: false,
        },
    ];
    let temp_plan = TemperaturePlan::derive(&lowering, &selection, &claims, &temp_reqs, 0).unwrap();
    let tv_temp_reqs: Vec<ScoringRequest> = temp_reqs
        .iter()
        .map(|r| ScoringRequest {
            outputs: r.positions - r.prompt,
            decoding: TvDecoding::Temperature {
                t: to_tv_rational(r.temperature),
            },
        })
        .collect();
    let tv_exact = intern_scoring_tables(InternConfig {
        base_tables: &base_tv_tables,
        vocab: claims.arch.vocab,
        sigma: claims.quant.exp_sigma,
        logits_exponent: claims.quant.logits_exponent,
        logit_lo: logit_range.lo as i32,
        logit_hi: logit_range.hi as i32,
        max_log2_rows: claims.gadget_policy.max_log2_rows,
        mode: ScoringMode::Exact,
        requests: &tv_temp_reqs,
    })
    .unwrap();
    let temp_plan_kinds: Vec<TvTableKind> = temp_plan
        .selection()
        .tables
        .iter()
        .map(|s| to_tv_table_kind(&s.kind))
        .collect();
    assert_eq!(tv_exact.catalog, temp_plan_kinds);
    for cert in temp_plan.certificates() {
        let rt = &tv_exact.request_tables[cert.request_index];
        let expected_dec: Vec<TvTableId> =
            cert.range_tables.iter().map(|id| TvTableId(id.0)).collect();
        assert_eq!(rt.decoding_ids, expected_dec);
        assert!(rt.log2_ids.is_empty());
    }

    // 3c. Zero-output request before swapped output-bearing temperatures + narrow max_log2_rows parity.
    let swapped_mix = [
        MixtureRequestSpec {
            prompt: 4,
            positions: 4,
            decoding: MixtureDecoding::Temperature(p_t07),
        },
        MixtureRequestSpec {
            prompt: 2,
            positions: 5,
            decoding: MixtureDecoding::Temperature(p_t15),
        },
        MixtureRequestSpec {
            prompt: 1,
            positions: 4,
            decoding: MixtureDecoding::Temperature(p_t07),
        },
    ];
    let swapped_plan =
        MixturePlan::derive(&lowering, &selection, &claims, &swapped_mix, 6, 0).unwrap();
    let tv_swapped_reqs: Vec<ScoringRequest> = swapped_mix
        .iter()
        .map(|r| ScoringRequest {
            outputs: r.positions - r.prompt,
            decoding: match r.decoding {
                MixtureDecoding::Greedy => TvDecoding::Greedy,
                MixtureDecoding::Temperature(t) => TvDecoding::Temperature {
                    t: to_tv_rational(t),
                },
            },
        })
        .collect();
    let tv_swapped = intern_scoring_tables(InternConfig {
        base_tables: &base_tv_tables,
        vocab: claims.arch.vocab,
        sigma: claims.quant.exp_sigma,
        logits_exponent: claims.quant.logits_exponent,
        logit_lo: logit_range.lo as i32,
        logit_hi: logit_range.hi as i32,
        max_log2_rows: claims.gadget_policy.max_log2_rows,
        mode: ScoringMode::Mixture { eps_log2: 6 },
        requests: &tv_swapped_reqs,
    })
    .unwrap();
    let swapped_kinds: Vec<TvTableKind> = swapped_plan
        .selection()
        .tables
        .iter()
        .map(|s| to_tv_table_kind(&s.kind))
        .collect();
    assert_eq!(tv_swapped.catalog, swapped_kinds);
    assert_eq!(tv_swapped.exp_tables[0].temperature, to_tv_rational(p_t15));
    assert_eq!(tv_swapped.exp_tables[1].temperature, to_tv_rational(p_t07));

    let mut narrow_claims = claims.clone();
    narrow_claims.gadget_policy.max_log2_rows = 15;
    assert!(
        MixturePlan::derive(&lowering, &selection, &narrow_claims, &mix_reqs[..1], 6, 0).is_err()
    );
    assert!(
        TemperaturePlan::derive(&lowering, &selection, &narrow_claims, &temp_reqs[..1], 0).is_err()
    );
    assert!(
        intern_scoring_tables(InternConfig {
            base_tables: &base_tv_tables,
            vocab: claims.arch.vocab,
            sigma: claims.quant.exp_sigma,
            logits_exponent: claims.quant.logits_exponent,
            logit_lo: logit_range.lo as i32,
            logit_hi: logit_range.hi as i32,
            max_log2_rows: 15,
            mode: ScoringMode::Mixture { eps_log2: 6 },
            requests: &tv_mix_reqs[..1],
        })
        .is_err()
    );
    assert!(
        intern_scoring_tables(InternConfig {
            base_tables: &base_tv_tables,
            vocab: claims.arch.vocab,
            sigma: claims.quant.exp_sigma,
            logits_exponent: claims.quant.logits_exponent,
            logit_lo: logit_range.lo as i32,
            logit_hi: logit_range.hi as i32,
            max_log2_rows: 15,
            mode: ScoringMode::Exact,
            requests: &tv_temp_reqs[..1],
        })
        .is_err()
    );
}

#[test]
fn actual_one_sided_log2_columns_cpu_proofs_and_tv_core_checks_match_independent_vectors() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../spec/vectors/gadget-finals/one-sided-log2.json"
    ))
    .unwrap();

    // Prebuild all tables needed by `OneSidedLog2::new` (`Pow2(63)`, `Exp2Frac(8, 16)`, `Range(1..=32)`).
    let mut table_specs = Vec::new();
    for (next_id, kind) in [
        TableKind::Pow2 { max_exp: 63 },
        TableKind::Exp2Frac {
            frac_bits: 8,
            out_bits: 16,
        },
    ]
    .into_iter()
    .chain((1..=32u8).map(|bits| TableKind::Range { bits }))
    .enumerate()
    {
        table_specs.push(
            GeneratedTable::new(TableId(next_id as u32), kind)
                .unwrap()
                .spec()
                .clone(),
        );
    }

    // 1. Differential schema verification across compiler, prover OneSidedLog2, verifier Log2Params, and oracle.
    let mut schema_count = 0usize;
    for sc in array(&fixture["schema_cases"]) {
        let name = sc["name"].as_str().unwrap();
        let vocab = sc["vocab"].as_u64().unwrap() as u32;
        let sigma = sc["sigma"].as_u64().unwrap() as u8;
        let eps_log2 = sc["eps_log2"].as_u64().unwrap() as u8;

        let (comp_cols, comp_reqs) = log2_columns_and_requests(vocab, sigma).unwrap();
        let producer =
            OneSidedLog2::<G>::new(vocab as usize, sigma, eps_log2, &table_specs).unwrap();
        let constraints = producer.constraints::<G>().unwrap();
        let tv_params = TvLog2Params::derive(vocab, sigma, eps_log2).unwrap();

        assert_eq!(comp_cols.len(), constraints.committed.len(), "{name}");
        assert_eq!(comp_cols.len(), tv_params.committed_row_count(), "{name}");
        assert_eq!(comp_reqs.len(), constraints.lookups.len(), "{name}");
        let comp_tv_reqs: Vec<TvTableKind> = comp_reqs.iter().map(to_tv_table_kind).collect();
        assert_eq!(comp_tv_reqs.as_slice(), tv_params.lookup_specs(), "{name}");

        let oracle_reqs: Vec<TableKind> = array(&sc["lookup_specs"])
            .iter()
            .map(parse_prover_table_kind)
            .collect();
        assert_eq!(comp_reqs, oracle_reqs, "{name}");

        let tv_site = TvOneSidedLog2::derive(TvGadgetId(99), vocab, 3, sigma, eps_log2).unwrap();
        let row_target: Vec<Fp2> = array(&sc["point"])
            .iter()
            .map(|v| tv_fp2(field(v)))
            .collect();
        let lookup_keys: Vec<Fp2> = array(&sc["lookup_key_evals"])
            .iter()
            .map(|v| tv_fp2(field(v)))
            .collect();
        let lookup_values: Vec<Fp2> = array(&sc["lookup_value_evals"])
            .iter()
            .map(|v| tv_fp2(field(v)))
            .collect();
        let _checked = tv_site
            .check_row(TvLog2RowValues {
                row_target: &row_target,
                z_eval: tv_fp2(field(&sc["z_eval"])),
                q_eval: tv_fp2(field(&sc["q_eval"])),
                committed_evals: None,
                lookup_keys: &lookup_keys,
                lookup_values: &lookup_values,
            })
            .expect("honest schema case passes check_row");
        schema_count += 1;
    }
    assert_eq!(schema_count, 7);

    // 2. Full witness, CPU 2-stage sumcheck proof, tv_core verification, and step-12 Û bound across all 15 cases.
    let inv_pow16: [G; 4] =
        core::array::from_fn(|j| G::from_u64_reduced(1u64 << (16 * j)).inverse().unwrap());
    let (mut honest_count, mut alt_count) = (0usize, 0usize);

    for case in array(&fixture["cases"]) {
        let name = case["name"].as_str().unwrap();
        let gadget_id = TvGadgetId(case["gadget"].as_u64().unwrap() as u32);
        let rows = case["rows"].as_u64().unwrap() as usize;
        let vocab = case["vocab"].as_u64().unwrap() as usize;
        let sigma = case["sigma"].as_u64().unwrap() as u8;
        let eps_log2 = case["eps_log2"].as_u64().unwrap() as u8;
        let actual_uhat = case["actual_uhat"].as_u64().unwrap();
        let claimed_uhat = case["claimed_uhat"].as_u64().unwrap();
        let local_reject = case["local_reject"].as_str();
        let failed_stage = case["failed_stage"].as_array().map(|a| {
            (
                a[0].as_u64().unwrap() as usize,
                a[1].as_u64().unwrap() as usize,
            )
        });

        let producer = OneSidedLog2::<G>::new(vocab, sigma, eps_log2, &table_specs).unwrap();
        let constraints = producer.constraints::<G>().unwrap();
        let onehot = OneHot::<G>::new(vocab).unwrap();
        let tv_site =
            TvOneSidedLog2::derive(gadget_id, vocab as u32, rows as u32, sigma, eps_log2).unwrap();
        let p = tv_site.params();
        let k_z = p.k_z();
        let k_q = p.k_q();
        let y_active = p.y_active();

        let vocab_vars = tv_site.vocab_variables();
        let row_vars = tv_site.row_variables();
        let v_pad = 1usize << vocab_vars;
        let t_pad = 1usize << row_vars;
        let total_vt = v_pad * t_pad;

        let e_vocab_col: SecretBuf<G> = array(&case["exps"])
            .iter()
            .map(|v| G::from_u64_reduced(v.as_u64().unwrap()))
            .collect();
        let hy_col: SecretBuf<G> = array(&case["h_y"])
            .iter()
            .map(|v| G::from_u64_reduced(v.as_u64().unwrap()))
            .collect();
        let hyhat_col: SecretBuf<G> = array(&case["h_yhat"])
            .iter()
            .map(|v| G::from_u64_reduced(v.as_u64().unwrap()))
            .collect();
        let row_records = array(&case["row_records"]);
        assert_eq!(row_records.len(), t_pad);

        // On tight honest cases (excluding `one-sided-slack`, which intentionally uses a looser
        // exponent and `m_0 = 0` to prove one-sided slack soundness), check `OneSidedLog2::witness`,
        // `evaluate_batch`, `evaluate_lookups`, and `check_uhat_bound`.
        if local_reject.is_none() && failed_stage.is_none() && name != "one-sided-slack" {
            let mut per_col: Vec<SecretBuf<G>> = (0..constraints.committed.len())
                .map(|_| SecretBuf::with_capacity(t_pad))
                .collect();
            let mut sum_units = 0u64;
            for r in 0..t_pad {
                let real = r < rows;
                let row_e = &e_vocab_col[r * v_pad..(r + 1) * v_pad];
                let rec_buf: SecretBuf<G> =
                    hy_col[r * v_pad..(r + 1) * v_pad].iter().copied().collect();
                let pred_buf: SecretBuf<G> = hyhat_col[r * v_pad..(r + 1) * v_pad]
                    .iter()
                    .copied()
                    .collect();
                let rec_oh = OneHotWitness::from_column(rec_buf);
                let pred_oh = OneHotWitness::from_column(pred_buf);
                let w = producer.witness(row_e, &rec_oh, &pred_oh, real).unwrap();
                let expected_committed: Vec<i64> = array(&row_records[r]["committed"])
                    .iter()
                    .map(|v| v.as_i64().unwrap())
                    .collect();
                assert_eq!(w.values(), expected_committed.as_slice(), "{name} row {r}");
                assert_eq!(
                    w.bound_units(),
                    row_records[r]["b_t"].as_u64().unwrap(),
                    "{name} row {r}"
                );
                sum_units += w.bound_units();
                for (col_idx, &val) in w.values().iter().enumerate() {
                    per_col[col_idx].push(G::from_i64(val));
                }
            }
            assert_eq!(sum_units, actual_uhat, "{name}");
            let selector = producer.selector(t_pad, rows).unwrap();
            let batch_cols = Log2BatchColumns::from_columns(
                constraints.committed.iter().copied().zip(per_col).collect(),
            );
            let evaluated = producer
                .evaluate_batch(&e_vocab_col, &hy_col, &batch_cols, &selector)
                .unwrap();
            assert_eq!(
                evaluated.uhat_field(),
                G::from_u64_reduced(actual_uhat),
                "{name}"
            );
            let _lookups = producer
                .evaluate_lookups(&selector, &evaluated, &batch_cols)
                .unwrap();

            // Verify step-12 check_uhat_bound on the real produced Û_r.
            let dummy_req = [RequestRef {
                request_id: 7,
                session_id: [0u8; 16],
                prompt_len: 1,
                output_len: rows as u32,
            }];
            let mix_pred = PredictorSpec::Mixture {
                decoding: TvDecoding::Greedy,
                eps_log2,
            };
            assert_eq!(
                check_uhat_bound(
                    mix_pred,
                    &dummy_req,
                    &[FixedBits(actual_uhat)],
                    FixedBits(actual_uhat),
                ),
                Ok(FixedBits(actual_uhat))
            );
            if actual_uhat > 0 {
                assert_eq!(
                    check_uhat_bound(
                        mix_pred,
                        &dummy_req,
                        &[FixedBits(actual_uhat)],
                        FixedBits(actual_uhat - 1),
                    ),
                    Err(UhatError::Exceeded {
                        request_id: Some(7),
                        uhat: FixedBits(actual_uhat),
                        bound: FixedBits(actual_uhat - 1),
                    })
                );
                assert_eq!(
                    check_uhat_bound(
                        PredictorSpec::Exact {
                            decoding: TvDecoding::Greedy
                        },
                        &dummy_req,
                        &[FixedBits(actual_uhat)],
                        FixedBits(u64::MAX),
                    ),
                    Err(UhatError::Exceeded {
                        request_id: Some(7),
                        uhat: FixedBits(actual_uhat),
                        bound: FixedBits(0),
                    })
                );
            }
            let _ = &onehot;
        } else if name == "padded-z-nonzero" {
            // Prover witness rejects nonzero weights on a padded row (`real = false`).
            let r = rows;
            let row_e = &e_vocab_col[r * v_pad..(r + 1) * v_pad];
            let z_oh_buf: SecretBuf<G> = core::iter::repeat_n(G::ZERO, v_pad).collect();
            let z_oh_buf2: SecretBuf<G> = core::iter::repeat_n(G::ZERO, v_pad).collect();
            let z_oh = OneHotWitness::from_column(z_oh_buf);
            let z_oh2 = OneHotWitness::from_column(z_oh_buf2);
            assert!(matches!(
                producer.witness(row_e, &z_oh, &z_oh2, false),
                Err(GadgetError::Input { .. })
            ));
        }

        let row_target_f: Vec<F> = array(&case["row_target"]).iter().map(field).collect();
        let row_target_tv: Vec<Fp2> = row_target_f.iter().copied().map(tv_fp2).collect();
        let z_eval_f = field(&case["z_eval"]);
        let q_eval_f = field(&case["q_eval"]);
        let lookup_keys_f: Vec<F> = array(&case["lookup_key_evals"]).iter().map(field).collect();
        let lookup_vals_f: Vec<F> = array(&case["lookup_value_evals"])
            .iter()
            .map(field)
            .collect();
        let lookup_keys_tv: Vec<Fp2> = lookup_keys_f.iter().copied().map(tv_fp2).collect();
        let lookup_vals_tv: Vec<Fp2> = lookup_vals_f.iter().copied().map(tv_fp2).collect();

        let row_vals = TvLog2RowValues {
            row_target: &row_target_tv,
            z_eval: tv_fp2(z_eval_f),
            q_eval: tv_fp2(q_eval_f),
            committed_evals: None,
            lookup_keys: &lookup_keys_tv,
            lookup_values: &lookup_vals_tv,
        };

        if local_reject.is_some() {
            alt_count += 1;
            assert_eq!(
                tv_site.check_row(row_vals).unwrap_err(),
                TvGadgetError::Rejected { gadget: gadget_id },
                "{name}"
            );
            continue;
        }

        let checked_tv = tv_site.check_row(row_vals).unwrap();
        let x_targets_f: Vec<F> = array(&case["x_targets"]).iter().map(field).collect();
        let y_targets_f: Vec<F> = array(&case["y_targets"]).iter().map(field).collect();

        let record = if failed_stage.is_some() {
            &case["omission"]
        } else {
            &case["proof"]
        };
        let s1_omit = failed_stage.and_then(|(s, idx)| (s == 1).then_some(idx));
        let s2_omit = failed_stage.and_then(|(s, idx)| (s == 2).then_some(idx));

        let (mut ours, mut theirs) = transcripts(&case["context"]);
        let mut initial = vec![z_eval_f, q_eval_f, lookup_vals_f[0], lookup_vals_f[1]];
        initial.extend_from_slice(&lookup_keys_f);
        ours.absorb("stage/0/split", &initial);
        absorb_fp2_slice(&mut theirs, "stage/0/split", &initial);

        let zero_row: Vec<F> = (0..row_vars)
            .map(|j| ours.challenge_fp2(&format!("stage/1/zero/{j}")))
            .collect();
        assert_eq!(
            zero_row,
            array(&record["stage1"]["zero_row"])
                .iter()
                .map(field)
                .collect::<Vec<_>>()
        );

        let psi_col: Vec<G> = row_records
            .iter()
            .map(|r| G::from_u64_reduced(r["psi"].as_u64().unwrap()))
            .collect();
        let psi_minus_one: Vec<G> = psi_col.iter().map(|&x| x - G::ONE).collect();
        let z_col: SecretBuf<G> = row_records
            .iter()
            .map(|r| G::from_u64_reduced(r["z"].as_u64().unwrap()))
            .collect();
        let y_col: SecretBuf<G> = row_records
            .iter()
            .map(|r| G::from_u64_reduced(r["y"].as_u64().unwrap()))
            .collect();
        let m_col: SecretBuf<G> = row_records
            .iter()
            .map(|r| G::from_u64_reduced(r["m_bit"].as_u64().unwrap()))
            .collect();
        let i_col: SecretBuf<G> = row_records
            .iter()
            .map(|r| G::from_u64_reduced(r["i_exp"].as_u64().unwrap()))
            .collect();
        let p_col: SecretBuf<G> = row_records
            .iter()
            .map(|r| G::from_u64_reduced(r["p_val"].as_u64().unwrap()))
            .collect();
        let f_col: SecretBuf<G> = row_records
            .iter()
            .map(|r| G::from_u64_reduced(r["f_frac"].as_u64().unwrap()))
            .collect();
        let e_val_col: SecretBuf<G> = row_records
            .iter()
            .map(|r| G::from_u64_reduced(r["e_val"].as_u64().unwrap()))
            .collect();
        let q_col: SecretBuf<G> = row_records
            .iter()
            .map(|r| G::from_u64_reduced(r["q_val"].as_u64().unwrap()))
            .collect();

        let z_low_cols: Vec<SecretBuf<G>> = (0..k_z - 1)
            .map(|h| {
                row_records
                    .iter()
                    .map(|r| G::from_u64_reduced(r["lookup_keys"][2 + h].as_u64().unwrap()))
                    .collect()
            })
            .collect();
        let q_low_cols: Vec<SecretBuf<G>> = (0..k_q - 1)
            .map(|a| {
                row_records
                    .iter()
                    .map(|r| G::from_u64_reduced(r["lookup_keys"][2 + k_z + a].as_u64().unwrap()))
                    .collect()
            })
            .collect();
        let p_low_cols: Vec<SecretBuf<G>> = (0..3)
            .map(|b| {
                row_records
                    .iter()
                    .map(|r| {
                        G::from_u64_reduced(r["lookup_keys"][2 + k_z + k_q + b].as_u64().unwrap())
                    })
                    .collect()
            })
            .collect();

        let eq_rt: Vec<F> = (0..t_pad).map(|i| eq_at(&row_target_f, i)).collect();
        let eq_zr: Vec<F> = (0..t_pad).map(|i| eq_at(&zero_row, i)).collect();
        let origin_row: Vec<G> = (0..t_pad)
            .map(|i| G::from_u64_reduced(u64::from(i == 0)))
            .collect();
        let v_limbs = [(vocab as u64) & 0xFFFF, (vocab as u64) >> 16];

        // Build Stage 1 instances on `row_vars`:
        let mut s1_instances = Vec::new();
        // 0: match booleanity
        s1_instances.push(Instance::sum_of_products(
            vec![
                Table::ext(&eq_zr).public(),
                Table::base(&m_col),
                Table::base(&origin_row).public(),
            ],
            if s1_omit == Some(0) {
                vec![Term::new(F::ZERO, vec![2])]
            } else if eps_log2 == 0 {
                vec![Term::new(F::ONE, vec![0, 1])]
            } else {
                vec![
                    Term::new(F::ONE, vec![0, 1]),
                    Term::new(-F::ONE, vec![0, 1, 1]),
                ]
            },
        ));
        // 1: q_t - (Y_t + psi_t) * E_t = 0
        s1_instances.push(Instance::sum_of_products(
            vec![
                Table::ext(&eq_zr).public(),
                Table::base(&psi_col).public(),
                Table::base(&q_col),
                Table::base(&y_col),
                Table::base(&e_val_col),
                Table::base(&origin_row).public(),
            ],
            if s1_omit == Some(1) {
                vec![Term::new(F::ZERO, vec![5])]
            } else {
                vec![
                    Term::new(F::ONE, vec![0, 2]),
                    Term::new(-F::ONE, vec![0, 3, 4]),
                    Term::new(-F::ONE, vec![0, 1, 4]),
                ]
            },
        ));
        // 2 .. 2 + k_z: X_1 .. X_{k_z}
        for j in 1..=k_z {
            let inst_idx = s1_instances.len();
            let vj = v_limbs.get(j - 1).copied().unwrap_or(0);
            if j - 1 < k_z - 1 {
                let tables = vec![
                    Table::ext(&eq_rt).public(),
                    Table::base(&psi_col).public(),
                    Table::base(&z_low_cols[j - 1]),
                    Table::base(&origin_row).public(),
                ];
                let terms = if s1_omit == Some(inst_idx) {
                    vec![Term::new(x_targets_f[j], vec![3])]
                } else {
                    let mut v = vec![Term::new(F::ONE, vec![0, 1, 2])];
                    if vj != 0 {
                        v.push(Term::new(F::from(G::from_u64_reduced(vj)), vec![0, 1]));
                    }
                    v
                };
                s1_instances.push(Instance::sum_of_products(tables, terms));
            } else {
                let mut tables = vec![
                    Table::ext(&eq_rt).public(),
                    Table::base(&psi_col).public(),
                    Table::base(&z_col),
                ];
                for col in &z_low_cols {
                    tables.push(Table::base(col));
                }
                let origin_idx = tables.len();
                tables.push(Table::base(&origin_row).public());
                let terms = if s1_omit == Some(inst_idx) {
                    vec![Term::new(x_targets_f[j], vec![origin_idx])]
                } else {
                    let scale_top = F::from(inv_pow16[k_z - 1]);
                    let mut v = vec![Term::new(scale_top, vec![0, 1, 2])];
                    for l in 0..k_z - 1 {
                        let shift = F::from(G::from_u64_reduced(1u64 << (16 * l)));
                        v.push(Term::new(-scale_top * shift, vec![0, 1, 3 + l]));
                    }
                    if vj != 0 {
                        v.push(Term::new(F::from(G::from_u64_reduced(vj)), vec![0, 1]));
                    }
                    v
                };
                s1_instances.push(Instance::sum_of_products(tables, terms));
            }
        }
        // 2 + k_z .. 2 + k_z + y_active: Y_0 .. Y_{y_active - 1}
        for (j, &y_target_j) in y_targets_f.iter().enumerate().take(y_active) {
            let inst_idx = s1_instances.len();
            let mut tables = vec![Table::ext(&eq_rt).public(), Table::base(&q_col)];
            for col in &q_low_cols {
                tables.push(Table::base(col));
            }
            let p_base = tables.len();
            tables.push(Table::base(&p_col));
            for col in &p_low_cols {
                tables.push(Table::base(col));
            }
            let origin_idx = tables.len();
            tables.push(Table::base(&origin_row).public());
            let terms = if s1_omit == Some(inst_idx) {
                vec![Term::new(y_target_j, vec![origin_idx])]
            } else {
                let q_expr = |a: usize| -> Vec<(F, usize)> {
                    if a + 1 < k_q {
                        vec![(F::ONE, 2 + a)]
                    } else {
                        let scale_top = F::from(inv_pow16[k_q - 1]);
                        let mut v = vec![(scale_top, 1)];
                        for l in 0..k_q - 1 {
                            let shift = F::from(G::from_u64_reduced(1u64 << (16 * l)));
                            v.push((-scale_top * shift, 2 + l));
                        }
                        v
                    }
                };
                let p_expr = |b: usize| -> Vec<(F, usize)> {
                    if b < 3 {
                        vec![(F::ONE, p_base + 1 + b)]
                    } else {
                        let scale_top = F::from(inv_pow16[3]);
                        let mut v = vec![(scale_top, p_base)];
                        for l in 0..3 {
                            let shift = F::from(G::from_u64_reduced(1u64 << (16 * l)));
                            v.push((-scale_top * shift, p_base + 1 + l));
                        }
                        v
                    }
                };
                let mut acc = Vec::new();
                for a in 0..k_q {
                    for b in 0..4 {
                        if a + b == j {
                            for &(cq, iq) in &q_expr(a) {
                                for &(cp, ip) in &p_expr(b) {
                                    acc.push(Term::new(cq * cp, vec![0, iq, ip]));
                                }
                            }
                        }
                    }
                }
                acc
            };
            s1_instances.push(Instance::sum_of_products(tables, terms));
        }
        // idx_pad_z: (psi_t - 1) * Z_t = 0
        let idx_pad_z = s1_instances.len();
        s1_instances.push(Instance::sum_of_products(
            vec![
                Table::ext(&eq_zr).public(),
                Table::base(&psi_minus_one).public(),
                Table::base(&z_col),
                Table::base(&origin_row).public(),
            ],
            if s1_omit == Some(idx_pad_z) {
                vec![Term::new(F::ZERO, vec![3])]
            } else {
                vec![Term::new(F::ONE, vec![0, 1, 2])]
            },
        ));
        // idx_uhat: sum_t B_t = claimed_uhat
        let idx_uhat = s1_instances.len();
        let uhat_terms = if s1_omit == Some(idx_uhat) {
            vec![Term::new(
                F::from(G::from_u64_reduced(claimed_uhat)),
                vec![4],
            )]
        } else {
            let k_scaled = 256 * i64::from(eps_log2);
            let c_minus_k = (p.c_match() as i64) - k_scaled;
            [
                (G::from_i64(k_scaled), vec![0]),
                (G::from_i64(c_minus_k), vec![0, 1]),
                (G::from_i64(256), vec![0, 2]),
                (G::from_i64(-256), vec![0, 1, 2]),
                (G::ONE, vec![0, 3]),
                (-G::ONE, vec![0, 1, 3]),
            ]
            .into_iter()
            .filter(|&(w, _)| w != G::ZERO)
            .map(|(w, factors)| Term::new(F::from(w), factors))
            .collect()
        };
        s1_instances.push(Instance::sum_of_products(
            vec![
                Table::base(&psi_col).public(),
                Table::base(&m_col),
                Table::base(&i_col),
                Table::base(&f_col),
                Table::base(&origin_row).public(),
            ],
            uhat_terms,
        ));

        let s1_proven =
            prove_batch::<F, _>(&mut ours, Protocol::One, Prefix::Stage(1), 3, &s1_instances)
                .unwrap();
        check_stage_record(&s1_proven, &ours, &record["stage1"]);

        // Stage 2 on `vocab_vars + row_vars`:
        let r1 = s1_proven.point().to_vec();
        let y_r1 = s1_proven.proof().final_evals[2];
        let s1_evals_len = s1_proven.proof().final_evals.len();
        let z_r1 = s1_proven.proof().final_evals[s1_evals_len - 4];
        let zero_vt: Vec<F> = (0..vocab_vars + row_vars)
            .map(|j| ours.challenge_fp2(&format!("stage/2/zero/{j}")))
            .collect();
        assert_eq!(
            zero_vt,
            array(&record["stage2"]["zero_vt"])
                .iter()
                .map(field)
                .collect::<Vec<_>>()
        );

        let eq_rt_vt: Vec<F> = (0..total_vt)
            .map(|i| eq_at(&row_target_f, i / v_pad))
            .collect();
        let eq_r1_vt: Vec<F> = (0..total_vt).map(|i| eq_at(&r1, i / v_pad)).collect();
        let eq_zvt: Vec<F> = (0..total_vt).map(|i| eq_at(&zero_vt, i)).collect();
        let valid_vt: Vec<G> = (0..total_vt)
            .map(|i| G::from_u64_reduced(u64::from(i % v_pad < vocab)))
            .collect();
        let m_broadcast_vt: SecretBuf<G> = (0..total_vt).map(|i| m_col[i / v_pad]).collect();
        let origin_vt: Vec<G> = (0..total_vt)
            .map(|i| G::from_u64_reduced(u64::from(i == 0)))
            .collect();

        let s2_instances = [
            Instance::sum_of_products(
                vec![
                    Table::ext(&eq_rt_vt).public(),
                    Table::base(&valid_vt).public(),
                    Table::base(&e_vocab_col),
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
                    Table::base(&e_vocab_col),
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
                    Table::ext(&eq_r1_vt).public(),
                    Table::base(&hy_col),
                    Table::base(&e_vocab_col),
                    Table::base(&origin_vt).public(),
                ],
                if s2_omit == Some(2) {
                    vec![Term::new(y_r1, vec![3])]
                } else {
                    vec![Term::new(F::ONE, vec![0, 1, 2])]
                },
            ),
            Instance::sum_of_products(
                vec![
                    Table::ext(&eq_zvt).public(),
                    Table::base(&m_broadcast_vt),
                    Table::base(&hy_col),
                    Table::base(&hyhat_col),
                    Table::base(&origin_vt).public(),
                ],
                if s2_omit == Some(3) {
                    vec![Term::new(F::ZERO, vec![4])]
                } else {
                    vec![
                        Term::new(F::ONE, vec![0, 1, 2]),
                        Term::new(-F::ONE, vec![0, 1, 3]),
                    ]
                },
            ),
        ];

        let s2_proven =
            prove_batch::<F, _>(&mut ours, Protocol::One, Prefix::Stage(2), 3, &s2_instances)
                .unwrap();
        check_stage_record(&s2_proven, &ours, &record["stage2"]);

        // Verify the CPU proofs directly with tv_core::gadgets::scoring::OneSidedLog2!
        let s1_tv = checked_tv
            .squeeze_stage1(&mut theirs, FixedBits(claimed_uhat))
            .unwrap();
        let s1_tv_proof = to_tv_proof(&record["stage1"]);
        let s2_tv_proof = to_tv_proof(&record["stage2"]);

        if let Some((fail_stage, _)) = failed_stage {
            alt_count += 1;
            match fail_stage {
                1 => {
                    assert_eq!(
                        s1_tv.verify(&mut theirs, &s1_tv_proof).unwrap_err(),
                        StageVerifyError::Sumcheck(SumcheckError::FinalMismatch {
                            round: row_vars as u32,
                        }),
                        "{name}"
                    );
                }
                2 => {
                    let (_, done1) = s1_tv.verify(&mut theirs, &s1_tv_proof).unwrap();
                    let s2_tv = done1.squeeze_stage2(&mut theirs).unwrap();
                    assert_eq!(
                        s2_tv.verify(&mut theirs, &s2_tv_proof).unwrap_err(),
                        StageVerifyError::Sumcheck(SumcheckError::FinalMismatch {
                            round: (vocab_vars + row_vars) as u32,
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
            let (_, verified) = s2_tv.verify(&mut theirs, &s2_tv_proof).unwrap();
            assert_eq!(verified.lookup_claims().len(), lookup_keys_tv.len());
            assert_eq!(theirs.state(), &ours.state());
            assert_eq!(theirs.counter(), ours.counter());
        }
    }
    assert_eq!((honest_count, alt_count), (5, 10));
}
