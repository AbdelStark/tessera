// SPDX-License-Identifier: Apache-2.0 OR MIT

//! End-to-end conformance, compiler-to-gadget parity, adversarial misuse, and `tv-core`-verified
//! tests for [`MixtureScoring`] (#155, RFC-0017, RFC-0026, RFC-0050, RFC-0053, RFC-0058).

use std::fs;
use std::path::Path;

use tessera_codec::error::RequestId;
use tessera_codec::{Ascii, DecodeLimits, HeapKey, Rational, Secret, SecretBuf, envelope, schema};
use tessera_compiler::{
    CompareColumn as CompilerCompareColumn, Log2AdviceColumn, MixtureDecoding, MixturePlan,
    MixtureRequestSpec, PublicModel, TemperaturePlan, TemperatureRequestSpec, analyze, claim_list,
    log2_columns_and_requests, lower, select, settle,
};
use tessera_field::{Field, Goldilocks as G, GoldilocksExt2 as F, PrimeField, QuadraticExtension};
use tessera_gadgets::{
    CompareColumn as GadgetCompareColumn, Log2Column, Log2Term, Lookups, OneHot, OneSidedLog2,
    Side, c_match,
};
use tessera_hash::{Mode, ParamSetId as HashParamSetId, SessionContext, Transcript};
use tessera_ir::{
    AdviceLayoutParams, CommitmentLayout, GadgetPolicy, IntegerModel, ParamSetId,
    WeightLayoutParams,
};
use tessera_scoring::{
    ClassBits, ExactGreedy, ExactTemperature, GreedyInputs, MixtureBatch, MixtureInputs,
    MixtureScoring, Sampler, SamplerBaseKey, ScoringError, TemperatureInputs,
};
use tessera_sumcheck::{Elementwise, Prefix, Protocol, prove_batch};
use tessera_tables::{GeneratedTable, Table, TableId, TableKind, TableSpec};
use tessera_witness::Engine;
use tv_core::sumcheck as tv;
use tv_field::{Fp as TvFp, Fp2 as TvFp2};
use tv_hash::transcript as vt;

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

fn tiny_model() -> IntegerModel {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/models/tiny-2-servable.d");
    let bytes = fs::read(path).expect("tiny-2-servable.d");
    envelope::open(&bytes, &schema::INTEGER_MODEL, DecodeLimits::INTEGER_MODEL).expect("model")
}

fn map_log2_col(col: Log2Column) -> Log2AdviceColumn {
    let map_cmp = |c: GadgetCompareColumn| match c {
        GadgetCompareColumn::Digit(Side::Left, limb) => {
            (CompilerCompareColumn::LeftByte, limb as u32)
        }
        GadgetCompareColumn::Carry(Side::Left, limb) => {
            (CompilerCompareColumn::LeftCarry, limb as u32)
        }
        GadgetCompareColumn::Digit(Side::Right, limb) => {
            (CompilerCompareColumn::RightByte, limb as u32)
        }
        GadgetCompareColumn::Carry(Side::Right, limb) => {
            (CompilerCompareColumn::RightCarry, limb as u32)
        }
        GadgetCompareColumn::Difference(limb) => (CompilerCompareColumn::Difference, limb as u32),
        GadgetCompareColumn::Borrow(limb) => (CompilerCompareColumn::Borrow, limb as u32),
        GadgetCompareColumn::Input(..) => panic!("Input is never committed"),
    };
    match col {
        Log2Column::MatchBit => Log2AdviceColumn::MatchBit,
        Log2Column::IntExp => Log2AdviceColumn::IntExp,
        Log2Column::Pow2Val => Log2AdviceColumn::Pow2Val,
        Log2Column::FracExp => Log2AdviceColumn::FracExp,
        Log2Column::Exp2Val => Log2AdviceColumn::Exp2Val,
        Log2Column::ScaledSelected => Log2AdviceColumn::ScaledSelected,
        Log2Column::TotalLimb(j) => Log2AdviceColumn::TotalLimb(j),
        Log2Column::ScaledLimb(j) => Log2AdviceColumn::ScaledLimb(j),
        Log2Column::Pow2Limb(j) => Log2AdviceColumn::Pow2Limb(j),
        Log2Column::Compare(cmp) => {
            let (column, limb) = map_cmp(cmp);
            Log2AdviceColumn::Comparison { column, limb }
        }
    }
}

fn tv_fp(x: G) -> TvFp {
    TvFp::from_canonical(x.to_canonical_u64()).unwrap()
}

fn tv_fp2(x: F) -> TvFp2 {
    let [c0, c1] = x.coeffs();
    TvFp2::new(tv_fp(c0), tv_fp(c1))
}

fn fp2(x: TvFp2) -> F {
    F::new(
        G::from_canonical_u64(x.c0.to_canonical()).unwrap(),
        G::from_canonical_u64(x.c1.to_canonical()).unwrap(),
    )
}

fn tv_fp2s(xs: &[F]) -> Vec<TvFp2> {
    xs.iter().copied().map(tv_fp2).collect()
}

fn eq_weights(point: &[TvFp2]) -> Vec<TvFp2> {
    let mut weights = vec![TvFp2::ONE];
    for &r in point {
        let high: Vec<TvFp2> = weights.iter().map(|&e| e * r).collect();
        for (e, h) in weights.iter_mut().zip(&high) {
            *e -= *h;
        }
        weights.extend(high);
    }
    weights
}

fn mle_fp(values: &[G], point: &[TvFp2]) -> TvFp2 {
    assert_eq!(values.len(), 1 << point.len());
    values
        .iter()
        .zip(eq_weights(point))
        .fold(TvFp2::ZERO, |sum, (&v, e)| sum + e * TvFp2::from(tv_fp(v)))
}

fn check_sumcheck_elementwise(
    relation: &Elementwise<'_, F>,
    point: &[F],
    target: F,
    operands: &[&[G]],
    degree: usize,
    mut verifier_eval: impl FnMut(&[TvFp2], &[TvFp2]) -> TvFp2,
) {
    let vars = relation.shape().vars;
    let tables = relation.tables(point).expect("tables");
    let mut pt = Transcript::new(&SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::NonInteractive,
        params: HashParamSetId(Ascii::new("test-small/1").unwrap()),
        statement_digest: [0x55; 32],
        session_id: [0x17; 16],
    });
    let mut vt = vt::Transcript::new(&vt::SessionContext {
        protocol: vt::Ascii::new("tessera-protocol/1").unwrap(),
        mode: vt::Mode::NonInteractive,
        params: vt::ParamSetId(vt::Ascii::new("test-small/1").unwrap()),
        statement_digest: [0x55; 32],
        session_id: [0x17; 16],
    });
    let proved = prove_batch(
        &mut pt,
        Protocol::One,
        Prefix::Stage(1),
        degree,
        &[tables.instance()],
    )
    .expect("prove_batch");
    let proof = tv::SumcheckProof {
        rounds: proved.proof().rounds.iter().map(|r| tv_fp2s(r)).collect(),
        final_evals: tv_fp2s(&proved.proof().final_evals),
    };
    let checked = tv::verify_batch(
        &mut vt,
        tv::Prefix::Stage(1),
        degree,
        &[tv::Instance {
            claim: tv_fp2(target),
            vars,
            evals: operands.len(),
        }],
        &proof,
        |_, r, e| Some(verifier_eval(r, e)),
    )
    .expect("verify_batch");
    for (&val, col) in proof.final_evals.iter().zip(operands) {
        let p = checked
            .point_of(col.len().trailing_zeros() as usize)
            .expect("point_of");
        assert_eq!(val, mle_fp(col, p));
    }
    assert_eq!(pt.state(), *vt.state());
}

fn assert_all_lookups_in_catalog(lookups: &Lookups<G>, catalog: &[TableSpec]) {
    for grp in lookups.groups() {
        let spec = catalog
            .iter()
            .find(|s| s.id == grp.table())
            .expect("table in catalog");
        let tbl = GeneratedTable::from_spec(spec).expect("regen");
        match grp.values() {
            Some(vals) => {
                for (&k, &v) in grp.keys().iter().zip(vals.iter()) {
                    let key_i64 = k.to_centered_i128() as i64;
                    let val_i64 = v.to_centered_i128() as i64;
                    assert_eq!(
                        tbl.lookup(key_i64, 0),
                        Some(val_i64),
                        "key {key_i64} value {val_i64} missing from {:?}",
                        spec.kind
                    );
                }
            }
            None => {
                for &k in grp.keys() {
                    let key_i64 = k.to_centered_i128() as i64;
                    assert!(
                        tbl.index(key_i64, 0).is_some(),
                        "key {key_i64} out of range for {:?}",
                        spec.kind
                    );
                }
            }
        }
    }
}

fn verify_mixture_batch(batch: &MixtureBatch<'_>, vocab: usize, catalog: &[TableSpec]) {
    let padded_rows = batch.padded_rows();
    let real_rows = batch.real_rows();
    let pv = vocab.next_power_of_two();
    let row_vars = padded_rows.trailing_zeros() as usize;
    let vocab_vars = pv.trailing_zeros() as usize;
    let total_vars = row_vars + vocab_vars;

    let pt_all: Vec<F> = (0..total_vars)
        .map(|i| {
            F::new(
                G::from_u64_reduced(5 + i as u64),
                G::from_u64_reduced(11 + 3 * i as u64),
            )
        })
        .collect();
    let pt_row = &pt_all[vocab_vars..];
    let tv_all = tv_fp2s(&pt_all);
    let tv_row = tv_fp2s(pt_row);

    // 1. VocabMax relations + product-forest roots + Exp_T lookups.
    let vrel = batch.vocab_relations::<F>().unwrap();
    let veval = batch.vocab_evaluated();
    let active_fp: Vec<G> = (0..padded_rows * pv)
        .map(|idx| G::from_u64_reduced(u64::from(idx / pv < real_rows && idx % pv < vocab)))
        .collect();
    let k0_fp = G::from_u64_reduced(batch.vocab_max().masked_key());
    let inactive_k0: Vec<G> = active_fp
        .iter()
        .map(|&a| if a == G::ONE { G::ZERO } else { k0_fp })
        .collect();
    check_sumcheck_elementwise(
        &vrel.key,
        &pt_all,
        fp2(mle_fp(&veval.keys, &tv_all)),
        &[batch.vocab_witness().maximum(), batch.logits()],
        3,
        |p, e| {
            tv::eq(&tv_all, p).unwrap()
                * (mle_fp(&active_fp, p) * (e[0] - e[1]) + mle_fp(&inactive_k0, p))
        },
    );
    for row in 0..padded_rows {
        let prod = veval.leaves[row * pv..(row + 1) * pv]
            .iter()
            .copied()
            .fold(G::ONE, |a, b| a * b);
        assert_eq!(prod, veval.roots[row]);
        if row < real_rows {
            assert_eq!(veval.roots[row], G::ZERO);
        } else {
            assert_eq!(veval.roots[row], G::ONE);
        }
    }
    assert_all_lookups_in_catalog(&veval.lookups, catalog);

    // 2. Bounded decoded indicator OneHot relations + suffix weights + Argmax/InverseCdf checks.
    let srel = batch.predicted_indicator_relations::<F>().unwrap();
    let psi_vec: Vec<G> = (0..padded_rows)
        .map(|r| G::from_u64_reduced(u64::from(r < real_rows)))
        .collect();
    let valid_mask: Vec<G> = (0..padded_rows * pv)
        .map(|idx| G::from_u64_reduced(u64::from(idx % pv < vocab)))
        .collect();
    let inv_valid: Vec<G> = valid_mask
        .iter()
        .map(|&v| if v == G::ONE { G::ZERO } else { G::ONE })
        .collect();
    check_sumcheck_elementwise(
        &srel.booleanity,
        &pt_all,
        F::ZERO,
        &[batch.predicted_indicators()],
        3,
        |p, e| tv::eq(&tv_all, p).unwrap() * e[0] * (TvFp2::ONE - e[0]),
    );
    check_sumcheck_elementwise(
        &srel.vocabulary,
        &pt_all,
        F::ZERO,
        &[batch.predicted_indicators()],
        3,
        |p, e| tv::eq(&tv_all, p).unwrap() * mle_fp(&inv_valid, p) * e[0],
    );
    check_sumcheck_elementwise(
        &srel.count,
        pt_row,
        fp2(mle_fp(&psi_vec, &tv_row)),
        &[batch.predicted_indicators()],
        2,
        |p, e| tv::eq(&tv_row, &p[vocab_vars..]).unwrap() * e[0],
    );

    if let Some(arel) = batch.argmax_relations::<F>().unwrap() {
        let aeval = batch.argmax_evaluated().unwrap();
        let amax = batch.argmax_maxima().unwrap();
        check_sumcheck_elementwise(
            &arel.selected,
            pt_row,
            fp2(mle_fp(amax, &tv_row)),
            &[batch.predicted_indicators(), batch.logits()],
            3,
            |p, e| tv::eq(&tv_row, &p[vocab_vars..]).unwrap() * e[0] * e[1],
        );
        check_sumcheck_elementwise(
            &arel.key,
            &pt_all,
            fp2(mle_fp(&aeval.keys, &tv_all)),
            &[amax, batch.logits(), &aeval.suffix],
            3,
            |p, e| tv::eq(&tv_all, p).unwrap() * mle_fp(&valid_mask, p) * (e[0] - e[1] - e[2]),
        );
        let sw = batch.predicted_suffix_weights(&pt_all).unwrap();
        let dot = sw
            .iter()
            .zip(batch.predicted_indicators())
            .fold(F::ZERO, |acc, (&w, &h)| acc + F::from(h) * w);
        assert_eq!(dot, fp2(mle_fp(&aeval.suffix, &tv_all)));
        assert_all_lookups_in_catalog(&aeval.lookups, catalog);
    }

    if let Some(crel) = batch.cdf_contractions::<F>().unwrap() {
        let ceval = batch.cdf_evaluated().unwrap();
        check_sumcheck_elementwise(
            &crel.total,
            pt_row,
            fp2(mle_fp(&ceval.totals, &tv_row)),
            &[batch.vocab_witness().exponentials()],
            3,
            |p, e| tv::eq(&tv_row, &p[vocab_vars..]).unwrap() * mle_fp(&valid_mask, p) * e[0],
        );
        check_sumcheck_elementwise(
            &crel.below,
            pt_row,
            fp2(mle_fp(&ceval.belows, &tv_row)),
            &[batch.vocab_witness().exponentials(), &ceval.suffix],
            3,
            |p, e| tv::eq(&tv_row, &p[vocab_vars..]).unwrap() * e[0] * e[1],
        );
        check_sumcheck_elementwise(
            &crel.selected,
            pt_row,
            fp2(mle_fp(&ceval.selecteds, &tv_row)),
            &[
                batch.predicted_indicators(),
                batch.vocab_witness().exponentials(),
            ],
            3,
            |p, e| tv::eq(&tv_row, &p[vocab_vars..]).unwrap() * e[0] * e[1],
        );
        let sw = batch.predicted_suffix_weights(&pt_all).unwrap();
        let dot = sw
            .iter()
            .zip(batch.predicted_indicators())
            .fold(F::ZERO, |acc, (&w, &h)| acc + F::from(h) * w);
        assert_eq!(dot, fp2(mle_fp(&ceval.suffix, &tv_all)));
        assert_all_lookups_in_catalog(batch.cdf_lookups().unwrap(), catalog);
    }

    // 3. OneSidedLog2 vocabulary relations, row equalities, bound relation, and lookups.
    let lvocab = batch.log2_vocab_relations::<F>().unwrap();
    let leval = batch.log2_evaluated();
    let lcols = batch.log2_columns();
    check_sumcheck_elementwise(
        &lvocab.total,
        pt_row,
        fp2(mle_fp(&leval.totals, &tv_row)),
        &[batch.vocab_witness().exponentials()],
        3,
        |p, e| tv::eq(&tv_row, &p[vocab_vars..]).unwrap() * mle_fp(&valid_mask, p) * e[0],
    );
    check_sumcheck_elementwise(
        &lvocab.selected,
        pt_row,
        fp2(mle_fp(&leval.selecteds, &tv_row)),
        &[
            batch.recorded_indicators(),
            batch.vocab_witness().exponentials(),
        ],
        3,
        |p, e| tv::eq(&tv_row, &p[vocab_vars..]).unwrap() * e[0] * e[1],
    );
    let m_col = lcols.column(Log2Column::MatchBit).unwrap();
    check_sumcheck_elementwise(
        &lvocab.match_indicator,
        &pt_all,
        F::ZERO,
        &[
            m_col,
            batch.recorded_indicators(),
            batch.predicted_indicators(),
        ],
        3,
        |p, e| tv::eq(&tv_all, p).unwrap() * e[0] * (e[1] - e[2]),
    );

    let constraints = batch.log2().constraints::<G>().unwrap();
    let mut eq_ops: Vec<&[G]> = vec![&leval.totals, &leval.selecteds];
    for &c in &constraints.committed {
        eq_ops.push(lcols.column(c).unwrap());
    }
    let eq_rels = batch.log2_equality_relations::<F>().unwrap();
    for (eq_idx, rel) in eq_rels.iter().enumerate() {
        check_sumcheck_elementwise(rel, pt_row, F::ZERO, &eq_ops, 3, |p, e| {
            let psi = fp2(mle_fp(&psi_vec, p));
            let poly = constraints.equalities[eq_idx].evaluate(|term| match term {
                Log2Term::RowSelector => psi,
                Log2Term::Total => fp2(e[0]),
                Log2Term::Selected => fp2(e[1]),
                Log2Term::Column(col) => {
                    let pos = constraints
                        .committed
                        .iter()
                        .position(|&c| c == col)
                        .unwrap();
                    fp2(e[2 + pos])
                }
            });
            tv::eq(&tv_row, p).unwrap() * tv_fp2(poly)
        });
    }

    let bound_rel = batch.log2_bound_relation::<F>().unwrap();
    let i_col = lcols.column(Log2Column::IntExp).unwrap();
    let f_col = lcols.column(Log2Column::FracExp).unwrap();
    let c_val = G::from_u64_reduced(batch.log2().c_match_units());
    let k_val = G::from_u64_reduced(256 * u64::from(batch.log2().eps_log2()));
    let s256 = G::from_u64_reduced(256);
    check_sumcheck_elementwise(
        &bound_rel,
        &[],
        F::from(batch.uhat_field()),
        &[m_col, i_col, f_col],
        3,
        |p, e| {
            let psi = mle_fp(&psi_vec, p);
            let c_tv = tv_fp2(F::from(c_val));
            let k_tv = tv_fp2(F::from(k_val));
            let s_tv = tv_fp2(F::from(s256));
            let m = e[0];
            psi * (m * c_tv + (TvFp2::ONE - m) * (k_tv + s_tv * e[1] + e[2]))
        },
    );
    assert_eq!(leval.uhat_field(), batch.uhat_field());
    assert_all_lookups_in_catalog(batch.log2_lookups(), catalog);
}

#[test]
fn compiler_log2_columns_and_requests_match_gadget_constraints_across_widths() {
    let mut probe = Vec::new();
    probe.push(
        GeneratedTable::new(TableId(0), TableKind::Pow2 { max_exp: 63 })
            .unwrap()
            .spec()
            .clone(),
    );
    probe.push(
        GeneratedTable::new(
            TableId(1),
            TableKind::Exp2Frac {
                frac_bits: 8,
                out_bits: 16,
            },
        )
        .unwrap()
        .spec()
        .clone(),
    );
    for bits in 1..=32u8 {
        probe.push(
            GeneratedTable::new(TableId(1 + u32::from(bits)), TableKind::Range { bits })
                .unwrap()
                .spec()
                .clone(),
        );
    }

    for (vocab, sigma) in [
        (1u32, 0u8),
        (3, 1),
        (16, 15),
        (16, 16),
        (1024, 24),
        (1 << 18, 32),
    ] {
        let (comp_cols, comp_kinds) = log2_columns_and_requests(vocab, sigma).unwrap();
        let gadget = OneSidedLog2::<G>::new(vocab as usize, sigma, 6, &probe).unwrap();
        let constraints = gadget.constraints::<G>().unwrap();
        let gadget_cols: Vec<Log2AdviceColumn> = constraints
            .committed
            .iter()
            .copied()
            .map(map_log2_col)
            .collect();
        let gadget_kinds: Vec<TableKind> = constraints
            .lookups
            .iter()
            .map(|l| probe[l.table.0 as usize].kind.clone())
            .collect();
        assert_eq!(
            comp_cols, gadget_cols,
            "Log2AdviceColumn mismatch at vocab={vocab}, sigma={sigma}"
        );
        assert_eq!(
            comp_kinds, gadget_kinds,
            "Log2 lookup TableKind sequence mismatch at vocab={vocab}, sigma={sigma}"
        );
    }
}

#[test]
fn mixture_greedy_and_temperature_score_deviated_outputs_and_verify_end_to_end() {
    let model = tiny_model();
    let public = PublicModel::of(&model);
    let n = 5u32;
    let prompt_len = 2u32;
    let output_len = 3u32;
    let mut lowering = lower(public, n).expect("lower");
    let ranges = analyze(&lowering.graph, &lowering.tables).expect("ranges");
    settle(&mut lowering.graph, &ranges).expect("settle");
    let policy = GadgetPolicy::default();
    let selection = select(&lowering.graph, &ranges, &lowering.tables, &policy).expect("select");
    let params = ParamSetId(Ascii::new("test-small/1").unwrap());
    let claims = claim_list(public, &selection, &policy, TEST_SMALL, params).expect("claims");

    let temp_08 = Rational::new(4, 5).unwrap();
    let plan = MixturePlan::derive(
        &lowering,
        &selection,
        &claims,
        &[
            MixtureRequestSpec {
                prompt: prompt_len,
                positions: n,
                decoding: MixtureDecoding::Greedy,
            },
            MixtureRequestSpec {
                prompt: prompt_len,
                positions: n,
                decoding: MixtureDecoding::Temperature(temp_08),
            },
        ],
        6,
        0,
    )
    .expect("mixture plan");

    let catalog = &plan.selection().tables;
    let exp_tables: Vec<(Rational, TableSpec)> = plan
        .temperatures()
        .iter()
        .map(|t| (t.temperature, catalog[t.table.0 as usize].clone()))
        .collect();
    let argmax_spec = &catalog[plan.certificates()[0].argmax_range_table.unwrap().0 as usize];

    let scorer = MixtureScoring::new(
        &lowering,
        claims.quant.logits_exponent,
        6,
        &exp_tables,
        Some(argmax_spec),
        catalog,
    )
    .expect("MixtureScoring::new");

    let vocab = claims.arch.vocab as usize;
    let pv = vocab.next_power_of_two();
    let one_hot_dom = OneHot::<G>::new(vocab).unwrap();

    // Evaluate the tiny model on a recorded sequence `[1, 2, 0, 1, 2]` whose output tokens `[0, 1, 2]`
    // intentionally deviate from greedy/temperature on at least one position.
    let engine = Engine::new(&lowering).expect("engine");
    let rec_seq = vec![1u32, 2, 0, 1, 2];
    let witness = engine
        .run(&model, &Secret::new(rec_seq.clone()))
        .expect("run");
    let raw_logits = witness.tensor(lowering.logits).expect("logits");

    let make_row_logits = |pos: usize| -> SecretBuf<G> {
        let mut buf = SecretBuf::with_capacity(pv);
        for v in 0..vocab {
            buf.push(G::from_i64(raw_logits[pos * vocab + v]));
        }
        for _ in vocab..pv {
            buf.push(G::ZERO);
        }
        buf
    };

    // --- Part A: Greedy mixture ---
    // First find D's true greedy sequence by teacher-forcing step by step, AND test the deviated
    // sequence `[1, 2, 0, 1, 2]`.
    let greedy_req = scorer
        .request(
            RequestId(0),
            MixtureDecoding::Greedy,
            prompt_len,
            output_len,
        )
        .unwrap();
    let mut greedy_inputs_deviated = Vec::new();
    let mut exact_greedy_inputs = Vec::new();
    for out_idx in 0..(output_len as usize) {
        let pos = (prompt_len as usize) + out_idx;
        let row_logits = make_row_logits(pos - 1);
        let rec_tok = rec_seq[pos] as i64;
        let rec_h = one_hot_dom.witness(rec_tok, true).unwrap();
        let mut rl_copy = SecretBuf::with_capacity(pv);
        rl_copy.extend(row_logits.iter().copied());
        exact_greedy_inputs.push(GreedyInputs {
            logits: rl_copy,
            next_token: one_hot_dom.witness(rec_tok, true).unwrap(),
        });
        greedy_inputs_deviated.push(MixtureInputs {
            logits: row_logits,
            next_token: rec_h,
            draw: None,
        });
    }

    // Confirm ExactGreedy rejects this deviated sequence with ExactMismatch!
    let range_29 = GeneratedTable::new(TableId(999), TableKind::Range { bits: 29 }).unwrap();
    let exact_greedy = ExactGreedy::new(&lowering, range_29.spec()).unwrap();
    let exact_err = exact_greedy
        .request(RequestId(0), prompt_len, output_len)
        .unwrap()
        .witness(exact_greedy_inputs)
        .unwrap_err();
    assert!(matches!(exact_err, ScoringError::ExactMismatch { .. }));

    // Confirm MixtureScoring succeeds on the EXACT SAME deviated sequence with positive Û_r!
    let replay_greedy: Vec<MixtureInputs> = (0..(output_len as usize))
        .map(|out_idx| {
            let pos = (prompt_len as usize) + out_idx;
            MixtureInputs {
                logits: make_row_logits(pos - 1),
                next_token: one_hot_dom.witness(rec_seq[pos] as i64, true).unwrap(),
                draw: None,
            }
        })
        .collect();
    let greedy_cert = greedy_req.witness(greedy_inputs_deviated).unwrap();
    let match_only_bound = u64::from(output_len) * c_match(6).unwrap();
    assert!(
        greedy_cert.uhat() > match_only_bound,
        "deviated greedy request must exceed pure-match bound ({} <= {})",
        greedy_cert.uhat(),
        match_only_bound
    );
    let greedy_batch = greedy_cert.prepare_batch(&replay_greedy).unwrap();
    verify_mixture_batch(&greedy_batch, vocab, catalog);

    // Also confirm that when every recorded output equals D's greedy choice, Û_r == output_len * c_match(6)!
    let mut matched_greedy_inputs = Vec::new();
    let argmax_gadget = greedy_req.argmax().unwrap();
    for out_idx in 0..(output_len as usize) {
        let pos = (prompt_len as usize) + out_idx;
        let row_logits = make_row_logits(pos - 1);
        let (pred_h, _) = argmax_gadget.sample(&row_logits, true).unwrap();
        matched_greedy_inputs.push(MixtureInputs {
            logits: row_logits,
            next_token: pred_h,
            draw: None,
        });
    }
    let matched_cert = scorer
        .request(
            RequestId(1),
            MixtureDecoding::Greedy,
            prompt_len,
            output_len,
        )
        .unwrap()
        .witness(matched_greedy_inputs)
        .unwrap();
    assert_eq!(matched_cert.uhat(), match_only_bound);

    // --- Part B: Temperature mixture ---
    let base_key = SamplerBaseKey::new(Secret::new(HeapKey::from_box(Box::new([0x42u8; 16]))));
    let sampler_key = base_key.derive(&[0x11; 32], ClassBits::new(12).unwrap(), 0);
    let sampler = Sampler::new(&sampler_key);
    let stream_class = sampler.class(&Secret::new(rec_seq[..prompt_len as usize].to_vec()));

    let temp_req = scorer
        .request(
            RequestId(2),
            MixtureDecoding::Temperature(temp_08),
            prompt_len,
            output_len,
        )
        .unwrap();
    // Pick recorded tokens guaranteed to differ from InverseCdf's sample on row 0 so ExactTemperature
    // rejects while MixtureScoring succeeds!
    let cdf_gadget = temp_req.cdf().unwrap();
    let vocab_max_t = temp_req.vocab_max();
    let mut temp_mix_inputs = Vec::new();
    let mut temp_replay_inputs = Vec::new();
    let mut exact_temp_inputs = Vec::new();
    for out_idx in 0..(output_len as usize) {
        let pos = (prompt_len as usize) + out_idx;
        let row_logits = make_row_logits(pos - 1);
        let draw_u32 = sampler.draw(stream_class, out_idx as u32);
        let draw_g = G::from_u64_reduced(u64::from(draw_u32));
        let vw = vocab_max_t.witness(&row_logits, true).unwrap();
        let sampled_h = cdf_gadget.sample(vw.exponentials(), draw_g, true).unwrap();
        let sampled_tok = sampled_h
            .indicator()
            .iter()
            .position(|&x| x == G::ONE)
            .unwrap();
        // Force a mismatch on out_idx == 0, match on remaining positions!
        let rec_tok = if out_idx == 0 {
            ((sampled_tok + 1) % vocab) as i64
        } else {
            sampled_tok as i64
        };
        let mut rl1 = SecretBuf::with_capacity(pv);
        rl1.extend(row_logits.iter().copied());
        let mut rl2 = SecretBuf::with_capacity(pv);
        rl2.extend(row_logits.iter().copied());
        temp_mix_inputs.push(MixtureInputs {
            logits: row_logits,
            next_token: one_hot_dom.witness(rec_tok, true).unwrap(),
            draw: Some(draw_g),
        });
        temp_replay_inputs.push(MixtureInputs {
            logits: rl1,
            next_token: one_hot_dom.witness(rec_tok, true).unwrap(),
            draw: Some(draw_g),
        });
        exact_temp_inputs.push(TemperatureInputs {
            logits: rl2,
            next_token: one_hot_dom.witness(rec_tok, true).unwrap(),
            draw: draw_g,
        });
    }

    let t_plan = TemperaturePlan::derive(
        &lowering,
        &selection,
        &claims,
        &[TemperatureRequestSpec {
            prompt: prompt_len,
            positions: n,
            temperature: temp_08,
            bounded: false,
        }],
        0,
    )
    .unwrap();
    let t_cat = &t_plan.selection().tables;
    let t_exp = [(
        temp_08,
        t_cat[t_plan.temperatures()[0].table.0 as usize].clone(),
    )];
    let exact_temp =
        ExactTemperature::new(&lowering, claims.quant.logits_exponent, &t_exp, t_cat).unwrap();
    let exact_t_err = exact_temp
        .request(RequestId(2), temp_08, prompt_len, output_len)
        .unwrap()
        .witness(exact_temp_inputs)
        .unwrap_err();
    assert!(matches!(exact_t_err, ScoringError::ExactMismatch { .. }));

    let temp_cert = temp_req.witness(temp_mix_inputs).unwrap();
    assert!(temp_cert.uhat() > match_only_bound);
    let temp_batch = temp_cert.prepare_batch(&temp_replay_inputs).unwrap();
    verify_mixture_batch(&temp_batch, vocab, catalog);
}

#[test]
fn mixture_scoring_rejects_mismatched_replay_draw_misuse_and_redacts_secrets() {
    let model = tiny_model();
    let public = PublicModel::of(&model);
    let n = 3u32;
    let mut lowering = lower(public, n).expect("lower");
    let ranges = analyze(&lowering.graph, &lowering.tables).expect("ranges");
    settle(&mut lowering.graph, &ranges).expect("settle");
    let policy = GadgetPolicy::default();
    let selection = select(&lowering.graph, &ranges, &lowering.tables, &policy).expect("select");
    let params = ParamSetId(Ascii::new("test-small/1").unwrap());
    let claims = claim_list(public, &selection, &policy, TEST_SMALL, params).expect("claims");

    let temp_1 = Rational::ONE;
    let plan = MixturePlan::derive(
        &lowering,
        &selection,
        &claims,
        &[MixtureRequestSpec {
            prompt: 2,
            positions: n,
            decoding: MixtureDecoding::Greedy,
        }],
        6,
        0,
    )
    .unwrap();
    let catalog = &plan.selection().tables;
    let exp_tables = [(
        temp_1,
        catalog[plan.temperatures()[0].table.0 as usize].clone(),
    )];
    let argmax_spec = &catalog[plan.certificates()[0].argmax_range_table.unwrap().0 as usize];

    // eps_log2 = 17 rejected; duplicate temperatures rejected.
    assert!(matches!(
        MixtureScoring::new(
            &lowering,
            claims.quant.logits_exponent,
            17,
            &exp_tables,
            Some(argmax_spec),
            catalog
        ),
        Err(ScoringError::Usage { .. })
    ));
    let dup = [exp_tables[0].clone(), exp_tables[0].clone()];
    assert!(matches!(
        MixtureScoring::new(
            &lowering,
            claims.quant.logits_exponent,
            6,
            &dup,
            Some(argmax_spec),
            catalog
        ),
        Err(ScoringError::Usage { .. })
    ));

    let scorer = MixtureScoring::new(
        &lowering,
        claims.quant.logits_exponent,
        6,
        &exp_tables,
        Some(argmax_spec),
        catalog,
    )
    .unwrap();
    let vocab = claims.arch.vocab as usize;
    let pv = vocab.next_power_of_two();
    let dom = OneHot::<G>::new(vocab).unwrap();

    let make_logits = |val0: i64| -> SecretBuf<G> {
        let mut b = SecretBuf::with_capacity(pv);
        b.push(G::from_i64(val0));
        for _ in 1..pv {
            b.push(G::ZERO);
        }
        b
    };

    // Greedy request rejects Some(draw).
    let bad_draw_input = MixtureInputs {
        logits: make_logits(3),
        next_token: dom.witness(0, true).unwrap(),
        draw: Some(G::from_u64_reduced(999_999)),
    };
    let inp_dbg = format!("{bad_draw_input:?}");
    assert!(!inp_dbg.contains("999999"));
    assert!(matches!(
        scorer
            .request(RequestId(0), MixtureDecoding::Greedy, 2, 1)
            .unwrap()
            .witness([bad_draw_input]),
        Err(ScoringError::Usage { .. })
    ));

    // Temperature request rejects None draw.
    let no_draw_input = MixtureInputs {
        logits: make_logits(3),
        next_token: dom.witness(0, true).unwrap(),
        draw: None,
    };
    assert!(matches!(
        scorer
            .request(RequestId(1), MixtureDecoding::Temperature(temp_1), 2, 1)
            .unwrap()
            .witness([no_draw_input]),
        Err(ScoringError::Usage { .. })
    ));

    // Tampered replay inputs in prepare_batch are refused!
    let cert = scorer
        .request(RequestId(2), MixtureDecoding::Greedy, 2, 1)
        .unwrap()
        .witness([MixtureInputs {
            logits: make_logits(3),
            next_token: dom.witness(1, true).unwrap(),
            draw: None,
        }])
        .unwrap();
    assert!(matches!(
        cert.prepare_batch(&[MixtureInputs {
            logits: make_logits(4),
            next_token: dom.witness(1, true).unwrap(),
            draw: None,
        }]),
        Err(ScoringError::Usage { .. })
    ));

    // Zero-output request gives uhat == 0 and refuses prepare_batch.
    let empty_cert = scorer
        .request(RequestId(3), MixtureDecoding::Greedy, 3, 0)
        .unwrap()
        .witness([])
        .unwrap();
    assert_eq!(empty_cert.uhat(), 0);
    assert!(matches!(
        empty_cert.prepare_batch(&[]),
        Err(ScoringError::Usage { .. })
    ));

    // Verify zero-round (`output_len = 1`, `padded_rows = 1`, `row_vars = 0`) batch end-to-end
    // via `tv-core`, and check Debug redaction on MixtureScoring, MixtureCertificate and MixtureBatch.
    let batch = cert
        .prepare_batch(&[MixtureInputs {
            logits: make_logits(3),
            next_token: dom.witness(1, true).unwrap(),
            draw: None,
        }])
        .unwrap();
    verify_mixture_batch(&batch, vocab, catalog);
    let scorer_dbg = format!("{scorer:?}");
    let cert_dbg = format!("{cert:?}");
    let batch_dbg = format!("{batch:?}");
    assert!(scorer_dbg.contains("MixtureScoring"));
    assert!(cert_dbg.contains("MixtureCertificate"));
    assert!(batch_dbg.contains("MixtureBatch"));
}
