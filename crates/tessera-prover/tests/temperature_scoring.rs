// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tiny-model teacher forcing with Sampler PRF draws, `TemperaturePlan`, `ExactTemperature`, CPU
//! sumcheck / product-forest / LogUp proofs, and independent `tv-*` verification (#152).

use tessera_codec::error::{ErrorCode, RequestId, TokenPosition};
use tessera_codec::{
    Ascii, DecodeLimits, HeapKey, Rational, Secret, SecretBuf, encode, envelope, schema,
};
use tessera_compiler::anchor::{AesTables, build_standalone_sampler_prf};
use tessera_compiler::schedule::derive_graph_schedule;
use tessera_compiler::{
    Lowering, PublicModel, TemperaturePlan, TemperatureRequestSpec, analyze, claim_list, lower,
    select, settle,
};
use tessera_field::{Field, Goldilocks as G, GoldilocksExt2 as F, PrimeField, QuadraticExtension};
use tessera_gadgets::{OneHot, OneHotWitness, TableColumns};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_ir::{
    AdviceLayoutParams, CommitmentLayout, GadgetPolicy, GraphBuilder, IntegerModel,
    ParamSetId as IrParamSetId, WeightLayoutParams,
};
use tessera_logup::{TreeId, prove_logup};
use tessera_scoring::{
    ClassBits, ExactTemperature, OutputRow, Sampler, SamplerKey, TemperatureInputs, inverse_cdf,
};
use tessera_sumcheck::{Elementwise, Forest, Prefix, Protocol, prove_batch};
use tessera_tables::{BitOp, GeneratedTable, Table as _, TableId, TableKind, TableSpec};
use tessera_witness::{Engine, Witness};
use tv_core::sumcheck as tv;
use tv_field::{Fp as B, Fp2 as E};
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

fn tvf(x: F) -> E {
    let [c0, c1] = x.coeffs();
    E::new(
        B::from_canonical(c0.to_canonical_u64()).unwrap(),
        B::from_canonical(c1.to_canonical_u64()).unwrap(),
    )
}

fn lift(x: E) -> F {
    F::new(
        G::from_canonical_u64(x.c0.to_canonical()).unwrap(),
        G::from_canonical_u64(x.c1.to_canonical()).unwrap(),
    )
}

fn vectors(xs: &[F]) -> Vec<E> {
    xs.iter().copied().map(tvf).collect()
}

fn eq_index(p: &[E], i: usize) -> E {
    p.iter().enumerate().fold(E::ONE, |a, (j, &r)| {
        a * if (i >> j) & 1 == 0 { E::ONE - r } else { r }
    })
}

fn mle(xs: &[G], p: &[E]) -> E {
    assert_eq!(xs.len(), 1usize << p.len());
    xs.iter().enumerate().fold(E::ZERO, |s, (i, &x)| {
        s + eq_index(p, i) * E::from(B::from_canonical(x.to_canonical_u64()).unwrap())
    })
}

fn sessions(marker: u8, target: F) -> (Transcript, vt::Transcript) {
    let mut id = [0x82; 16];
    id[15] = marker;
    let mut p = Transcript::new(&SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("test-small/1").unwrap()),
        statement_digest: [0x38; 32],
        session_id: id,
    });
    let mut v = vt::Transcript::new(&vt::SessionContext {
        protocol: vt::Ascii::new("tessera-protocol/1").unwrap(),
        mode: vt::Mode::NonInteractive,
        params: vt::ParamSetId(vt::Ascii::new("test-small/1").unwrap()),
        statement_digest: [0x38; 32],
        session_id: id,
    });
    p.absorb("stage/0/split", &vec![target]);
    v.absorb(
        vt::Ascii::new("stage/0/split").unwrap(),
        &encode(&vec![target]),
    );
    (p, v)
}

fn proof_of(p: &tessera_sumcheck::ProvenBatch<F>) -> tv::SumcheckProof {
    tv::SumcheckProof {
        rounds: p.proof().rounds.iter().map(|r| vectors(r)).collect(),
        final_evals: vectors(&p.proof().final_evals),
    }
}

fn verify_component(
    relation: &Elementwise<'_, F>,
    point: &[F],
    target: F,
    operands: &[&[G]],
    degree: usize,
    marker: u8,
    mut final_value: impl FnMut(&[E], &[E]) -> E,
) {
    let vars = operands
        .iter()
        .map(|x| x.len().trailing_zeros() as usize)
        .max()
        .unwrap();
    assert_eq!(relation.shape().vars, vars);
    assert_eq!(relation.shape().summed_vars, vars - point.len());
    assert!(relation.degree() <= degree);
    let tables = relation.tables(point).unwrap();
    let (mut p, mut v) = sessions(marker, target);
    let proved = prove_batch(
        &mut p,
        Protocol::One,
        Prefix::Stage(1),
        degree,
        &[tables.instance()],
    )
    .unwrap();
    let proof = proof_of(&proved);
    let checked = tv::verify_batch(
        &mut v,
        tv::Prefix::Stage(1),
        degree,
        &[tv::Instance {
            claim: tvf(target),
            vars,
            evals: operands.len(),
        }],
        &proof,
        |_, r, e| Some(final_value(r, e)),
    )
    .unwrap();
    for (&value, column) in proof.final_evals.iter().zip(operands) {
        let pt = checked
            .point_of(column.len().trailing_zeros() as usize)
            .unwrap();
        assert_eq!(value, mle(column, pt));
    }
    assert_eq!(p.state(), *v.state());
    assert_eq!(p.counter(), v.counter());
}

fn native(model: &IntegerModel, tokens: &SecretBuf<u32>) -> (Lowering, Witness) {
    let mut lowering = lower(PublicModel::of(model), tokens.len() as u32).unwrap();
    let ranges = analyze(&lowering.graph, &lowering.tables).unwrap();
    settle(&mut lowering.graph, &ranges).unwrap();
    let input = Secret::new(tokens.iter().copied().collect());
    let witness = Engine::new(&lowering).unwrap().run(model, &input).unwrap();
    (lowering, witness)
}

fn row_inputs(w: &Witness, row: OutputRow, vocab: usize, draw: u32) -> TemperatureInputs {
    let padded = vocab.next_power_of_two();
    let logits = w.tensor(row.logits).unwrap();
    let h = w.tensor(row.one_hot).unwrap();
    let logits = (0..padded)
        .map(|v| {
            if v < vocab {
                G::from_i64(logits[row.logit_position as usize * vocab + v])
            } else {
                G::ZERO
            }
        })
        .collect();
    let indicator = (0..padded)
        .map(|v| {
            if v < vocab {
                G::from_i64(h[row.token_position as usize * vocab + v])
            } else {
                G::ZERO
            }
        })
        .collect();
    TemperatureInputs {
        logits,
        next_token: OneHotWitness::from_column(indicator),
        draw: G::from_u64_reduced(u64::from(draw)),
    }
}

fn tv_table(spec: &TableSpec) -> tv_tables::Table {
    let bytes = encode(spec);
    let decoded = tv_codec::decode(
        &bytes,
        &tv_codec::DecodeLimits::MESSAGE,
        tv_tables::TableSpec::decode,
    )
    .unwrap();
    tv_tables::Table::from_spec(&decoded).unwrap()
}

#[test]
fn tiny_model_temperature_requests_prove_zero_and_independent_verifiers_accept() {
    let bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../spec/vectors/models/tiny-2-servable.d"
    ))
    .unwrap();
    let model: IntegerModel =
        envelope::open(&bytes, &schema::INTEGER_MODEL, DecodeLimits::INTEGER_MODEL).unwrap();
    let vocab = model.arch.vocab as usize;
    let pv = vocab.next_power_of_two();
    let vocab_vars = pv.trailing_zeros() as usize;

    // Derive runtime TemperaturePlan from the public ClaimList alone (no wire change).
    let public = PublicModel::of(&model);
    let base_lowering = lower(public, 5).unwrap();
    let base_ranges = analyze(&base_lowering.graph, &base_lowering.tables).unwrap();
    let policy = GadgetPolicy::default();
    let selection = select(
        &base_lowering.graph,
        &base_ranges,
        &base_lowering.tables,
        &policy,
    )
    .unwrap();
    let claims = claim_list(
        public,
        &selection,
        &policy,
        TEST_SMALL,
        IrParamSetId(Ascii::new("test-small/1").unwrap()),
    )
    .unwrap();
    let temperature = Rational::new(7, 5).unwrap();
    let temp_plan = TemperaturePlan::derive(
        &base_lowering,
        &selection,
        &claims,
        &[TemperatureRequestSpec {
            prompt: 2,
            positions: 5,
            temperature,
            bounded: false,
        }],
        0,
    )
    .unwrap();
    let exp_id = temp_plan.temperatures()[0].table;
    let exp_spec = temp_plan.selection().tables[exp_id.0 as usize].clone();
    let exp_gen = GeneratedTable::from_spec(&exp_spec).unwrap();
    let range_specs: Vec<TableSpec> = temp_plan
        .selection()
        .tables
        .iter()
        .filter(|t| matches!(t.kind, TableKind::Range { .. }))
        .cloned()
        .collect();

    // Sample 3 output tokens autoregressively using Sampler + Exp_T + inverse_cdf.
    let prompt_vec = vec![0u32, 1];
    let sampler_key = SamplerKey::new(
        Secret::new(HeapKey::from_box(Box::new([0x33u8; 16]))),
        ClassBits::new(16).unwrap(),
    );
    let sampler = Sampler::new(&sampler_key);
    let stream_class = sampler.class(&Secret::new(prompt_vec.clone()));
    let draws_u32: [u32; 3] = core::array::from_fn(|t| sampler.draw(stream_class, t as u32));

    // Verify the in-circuit StandaloneSamplerPrf produces the exact same 3 draws!
    let aes_specs: Vec<TableSpec> = [
        (0u32, TableKind::AesSbox),
        (1, TableKind::Gf256Mul { c: 2 }),
        (2, TableKind::Gf256Mul { c: 3 }),
        (3, TableKind::Bitwise8 { op: BitOp::Xor }),
    ]
    .into_iter()
    .map(|(id, kind)| {
        GeneratedTable::new(TableId(id), kind)
            .unwrap()
            .spec()
            .clone()
    })
    .collect();
    let aes_tables = AesTables {
        sbox: TableId(0),
        mul2: TableId(1),
        mul3: TableId(2),
        xor: TableId(3),
    };
    let mut prf_gb = GraphBuilder::new();
    let prf_circuit = build_standalone_sampler_prf(&mut prf_gb, &aes_tables, 16, 2, 3).unwrap();
    let mut prf_graph = prf_gb.finish();
    let prf_ranges = analyze(&prf_graph, &aes_specs).unwrap();
    settle(&mut prf_graph, &prf_ranges).unwrap();
    let _prf_schedule =
        derive_graph_schedule(&prf_graph, &aes_specs, &[], Some(prf_circuit.draws)).unwrap();
    let prf_witness = Engine::for_graph(&prf_graph, &aes_specs)
        .unwrap()
        .evaluate_secret(sampler_key.standalone_prf_inputs(
            &Secret::new(prompt_vec.clone()),
            &[0, 1, 2],
            &prf_circuit,
        ))
        .unwrap();
    let circuit_draws: Vec<u32> = prf_witness
        .tensor(prf_circuit.draws)
        .unwrap()
        .iter()
        .map(|&d| u32::try_from(d).unwrap())
        .collect();
    assert_eq!(circuit_draws.as_slice(), &draws_u32);

    let mut tokens: SecretBuf<u32> = prompt_vec.iter().copied().collect();
    for &u in &draws_u32 {
        let (step_lowering, step_w) = native(&model, &tokens);
        let all_logits = step_w.tensor(step_lowering.logits).unwrap();
        let row = &all_logits[(tokens.len() - 1) * vocab..tokens.len() * vocab];
        let max_logit = row.iter().copied().max().unwrap();
        let weights: Vec<u64> = row
            .iter()
            .map(|&ell| exp_gen.lookup(max_logit - ell, 0).unwrap() as u64)
            .collect();
        let sampled = inverse_cdf(&weights, u).unwrap();
        tokens.push(sampled as u32);
    }

    // Teacher-force the complete 5-token sequence and build the ExactTemperature certificate.
    let (lowering, w) = native(&model, &tokens);
    let scorer = ExactTemperature::new(
        &lowering,
        model.quant.public.logits_exponent,
        &[(temperature, exp_spec.clone())],
        &range_specs,
    )
    .unwrap();
    let request = scorer.request(RequestId(19), temperature, 2, 3).unwrap();
    let inputs: Vec<TemperatureInputs> = (0..3)
        .map(|i| row_inputs(&w, request.row(i).unwrap().unwrap(), vocab, draws_u32[i]))
        .collect();
    let cert = request
        .witness(
            (0..3).map(|i| row_inputs(&w, request.row(i).unwrap().unwrap(), vocab, draws_u32[i])),
        )
        .unwrap();
    assert_eq!(cert.uhat(), 0);

    let batch = cert.prepare_batch(&inputs).unwrap();
    assert_eq!(batch.padded_rows(), 4);
    assert_eq!(batch.real_rows(), 3);

    // 1. Prove VocabMax's Exp_T lookups, InverseCdf's Range lookups, AND VocabMax's stacked
    //    product forest (4 trees of height vocab_vars; public roots [0, 0, 0, 1]) together in one
    //    leaf-stage GKR run, and verify with the production tv_core::leaf::Schedule!
    let exp_table_refs = [batch.vocab_max().table()];
    let exp_tcols = vec![TableColumns::new(exp_table_refs[0]).unwrap()];
    let exp_mults = batch
        .vocab_evaluated()
        .lookups
        .multiplicities(&exp_tcols)
        .unwrap();
    let exp_insts = batch
        .vocab_evaluated()
        .lookups
        .instances(&exp_tcols, &exp_mults)
        .unwrap();

    let cdf_table_refs = batch.cdf().tables();
    let cdf_tcols: Vec<_> = cdf_table_refs
        .iter()
        .map(|t| TableColumns::new(t).unwrap())
        .collect();
    let cdf_mults = batch.cdf_lookups().multiplicities(&cdf_tcols).unwrap();
    let cdf_insts = batch
        .cdf_lookups()
        .instances(&cdf_tcols, &cdf_mults)
        .unwrap();

    let mut all_insts = Vec::new();
    all_insts.extend(exp_insts);
    all_insts.extend(cdf_insts);
    let mut all_tv_tables = vec![(tv_table(&exp_spec), true, 4 * pv)];
    for (k, group) in batch.cdf_lookups().groups().iter().enumerate() {
        let spec = cdf_table_refs
            .iter()
            .find(|t| t.spec().id == group.table())
            .unwrap()
            .spec();
        all_tv_tables.push((tv_table(spec), false, all_insts[1 + k].lookups.len()));
    }

    let forest = Forest::<F>::new(&batch.vocab_evaluated().leaves, vocab_vars);
    let mut tree_ids = TreeId::paired(all_insts.len());
    tree_ids.push(TreeId::Forest(0));

    let (mut lp_tr, mut lv_tr) = sessions(10, F::ZERO);
    let proven_logup = prove_logup(&mut lp_tr, &all_insts, &[forest], &tree_ids).unwrap();

    let mut tv_trees: Vec<tv_core::leaf::Tree> = all_tv_tables
        .iter()
        .enumerate()
        .flat_map(|(instance, (table, has_values, n))| {
            [
                tv_core::leaf::Tree::Lookup {
                    instance,
                    height: n.next_power_of_two().trailing_zeros(),
                    has_values: *has_values,
                },
                tv_core::leaf::Tree::Table {
                    instance,
                    height: u32::from(table.log2_rows()),
                },
            ]
        })
        .collect();
    let forest_tree_idx = tv_trees.len();
    tv_trees.push(tv_core::leaf::Tree::Forest {
        height: vocab_vars as u32,
        trees_log: 2,
    });
    let leaf_schedule = tv_core::leaf::Schedule::new(&tv_trees).unwrap();
    let roots_tv = vectors(proven_logup.roots());
    let levels_tv: Vec<tv::SumcheckProof> = proven_logup
        .levels()
        .iter()
        .map(|lp| tv::SumcheckProof {
            rounds: lp.proof().rounds.iter().map(|r| vectors(r)).collect(),
            final_evals: vectors(&lp.proof().final_evals),
        })
        .collect();
    let values_tv = vectors(proven_logup.values());
    let checked_leaf = leaf_schedule
        .verify(
            &mut lv_tr,
            tv_core::leaf::Proof {
                roots: &roots_tv,
                levels: &levels_tv,
                values: &values_tv,
            },
            |tree, point| {
                (tree == forest_tree_idx).then(|| mle(&batch.vocab_evaluated().roots, point))
            },
            |instance, beta, point| all_tv_tables.get(instance)?.0.eval_mle(point, beta).ok(),
        )
        .unwrap();
    assert_eq!(lp_tr.state(), *lv_tr.state());
    assert_eq!(lp_tr.counter(), lv_tr.counter());

    let leaf_claim = &checked_leaf.forests[0].claim;
    assert_eq!(
        leaf_claim.value,
        mle(&batch.vocab_evaluated().leaves, &leaf_claim.point)
    );
    let key_claim = &checked_leaf.lookups.instances[0].key;
    assert_eq!(
        key_claim.value,
        mle(&batch.vocab_evaluated().keys, &key_claim.point)
    );

    // 2. Discharge VocabMax's leaf, key, and padding sumcheck relations against tv_core::sumcheck!
    let vrel = batch.vocab_relations::<F>().unwrap();
    let leaf_pt_f: Vec<F> = leaf_claim.point.iter().copied().map(lift).collect();
    let key_pt_f: Vec<F> = key_claim.point.iter().copied().map(lift).collect();
    let active_col: Vec<G> = (0..4 * pv)
        .map(|idx| G::from_u64_reduced(u64::from(idx / pv < 3 && idx % pv < vocab)))
        .collect();
    let inactive_one: Vec<G> = active_col
        .iter()
        .map(|&a| if a == G::ONE { G::ZERO } else { G::ONE })
        .collect();
    let k0_fp = G::from_u64_reduced(batch.vocab_max().masked_key());
    let inactive_k0: Vec<G> = active_col
        .iter()
        .map(|&a| if a == G::ONE { G::ZERO } else { k0_fp })
        .collect();
    verify_component(
        &vrel.leaf,
        &leaf_pt_f,
        lift(leaf_claim.value),
        &[batch.vocab_witness().maximum(), batch.logits()],
        3,
        11,
        |p, e| {
            tv::eq(&leaf_claim.point, p).unwrap()
                * (mle(&active_col, p) * (e[0] - e[1]) + mle(&inactive_one, p))
        },
    );
    verify_component(
        &vrel.key,
        &key_pt_f,
        lift(key_claim.value),
        &[batch.vocab_witness().maximum(), batch.logits()],
        3,
        12,
        |p, e| {
            tv::eq(&key_claim.point, p).unwrap()
                * (mle(&active_col, p) * (e[0] - e[1]) + mle(&inactive_k0, p))
        },
    );

    // 3. Discharge VocabMax's row-padding zero-check on padded rows against tv_core!
    let row_pt = vec![
        F::new(G::from_u64_reduced(3), G::from_u64_reduced(11)),
        F::new(G::from_u64_reduced(5), G::from_u64_reduced(17)),
    ];
    let row_pt_tv = vectors(&row_pt);
    let inv_psi: Vec<G> = (0..4)
        .map(|r| G::from_u64_reduced(u64::from(r >= 3)))
        .collect();
    verify_component(
        &vrel.padding,
        &row_pt,
        F::ZERO,
        &[batch.vocab_witness().maximum()],
        3,
        16,
        |p, e| tv::eq(&row_pt_tv, p).unwrap() * mle(&inv_psi, p) * e[0],
    );

    // 4. Prove InverseCdf's batched contractions (Z, B, S), suffix weights, and equalities!
    let crel = batch.cdf_contractions::<F>().unwrap();
    let z_claim = lift(mle(&batch.cdf_evaluated().totals, &row_pt_tv));
    let b_claim = lift(mle(&batch.cdf_evaluated().belows, &row_pt_tv));
    let s_claim = lift(mle(&batch.cdf_evaluated().selecteds, &row_pt_tv));

    let valid_col: Vec<G> = (0..4 * pv)
        .map(|idx| G::from_u64_reduced(u64::from(idx % pv < vocab)))
        .collect();
    verify_component(
        &crel.total,
        &row_pt,
        z_claim,
        &[batch.vocab_witness().exponentials()],
        3,
        13,
        |p, e| tv::eq(&row_pt_tv, &p[vocab_vars..]).unwrap() * mle(&valid_col, p) * e[0],
    );
    verify_component(
        &crel.below,
        &row_pt,
        b_claim,
        &[
            batch.vocab_witness().exponentials(),
            &batch.cdf_evaluated().suffix,
        ],
        3,
        14,
        |p, e| tv::eq(&row_pt_tv, &p[vocab_vars..]).unwrap() * e[0] * e[1],
    );
    verify_component(
        &crel.selected,
        &row_pt,
        s_claim,
        &[batch.indicators(), batch.vocab_witness().exponentials()],
        3,
        15,
        |p, e| tv::eq(&row_pt_tv, &p[vocab_vars..]).unwrap() * e[0] * e[1],
    );

    // Verify suffix weights reduce L~(leaf_claim.point) directly to batch.indicators().
    let s_weights = batch.suffix_weights(&leaf_pt_f).unwrap();
    let suffix_lhs = lift(mle(&batch.cdf_evaluated().suffix, &leaf_claim.point));
    let suffix_rhs = s_weights
        .iter()
        .zip(batch.indicators())
        .fold(F::ZERO, |acc, (&w, &h)| acc + w * F::from(h));
    assert_eq!(suffix_lhs, suffix_rhs);

    // Verify every batched InverseCdf equality relation proves 0 against tv_core::sumcheck!
    let c_base = batch.cdf().constraints::<G>().unwrap();
    let psi_col: Vec<G> = (0..4)
        .map(|row| G::from_u64_reduced(u64::from(row < 3)))
        .collect();
    let eq_rels = batch.cdf_equality_relations::<F>().unwrap();
    assert_eq!(eq_rels.len(), c_base.equalities.len());
    for (eq_idx, rel) in eq_rels.iter().enumerate() {
        let tbls = rel.tables(&row_pt).unwrap();
        let (mut ep, mut ev) = sessions(50 + eq_idx as u8, F::ZERO);
        let proved_eq = prove_batch(
            &mut ep,
            Protocol::One,
            Prefix::Stage(2),
            3,
            &[tbls.instance()],
        )
        .unwrap();
        let proof_eq = proof_of(&proved_eq);
        let eval_count = proof_eq.final_evals.len();
        let _ = tv::verify_batch(
            &mut ev,
            tv::Prefix::Stage(2),
            3,
            &[tv::Instance {
                claim: E::ZERO,
                vars: 2,
                evals: eval_count,
            }],
            &proof_eq,
            |_, r, _| {
                let eval_col = |col: &[G]| -> F { lift(mle(col, r)) };
                let poly = c_base.equalities[eq_idx].evaluate(|term| match term {
                    tessera_gadgets::CdfTerm::Draw => eval_col(batch.draws()),
                    tessera_gadgets::CdfTerm::Total => eval_col(&batch.cdf_evaluated().totals),
                    tessera_gadgets::CdfTerm::Below => eval_col(&batch.cdf_evaluated().belows),
                    tessera_gadgets::CdfTerm::Selected => {
                        eval_col(&batch.cdf_evaluated().selecteds)
                    }
                    tessera_gadgets::CdfTerm::RowSelector => eval_col(&psi_col),
                    tessera_gadgets::CdfTerm::Column(col) => {
                        eval_col(batch.cdf_columns().column(col).unwrap())
                    }
                });
                Some(tv::eq(&row_pt_tv, r).unwrap() * tvf(poly))
            },
        )
        .unwrap();
        assert_eq!(ep.state(), *ev.state());
    }

    // Prove OneHot on each real row's shared indicator (same indicator bound to tokens[pos]).
    let one_hot = OneHot::<G>::new(vocab).unwrap();
    let v_pt: Vec<F> = (0..vocab_vars)
        .map(|i| {
            F::new(
                G::from_u64_reduced(2 + i as u64),
                G::from_u64_reduced(7 + i as u64),
            )
        })
        .collect();
    let v_pt_tv = vectors(&v_pt);
    for (idx, inp) in inputs.iter().enumerate() {
        let row = cert.prepare_row(idx, inp).unwrap();
        let h = row.next_token().indicator();
        let r = one_hot.relations::<F>(row.next_token()).unwrap();
        verify_component(
            &r.booleanity,
            &v_pt,
            F::ZERO,
            &[h],
            3,
            20 + idx as u8,
            |p, e| tv::eq(&v_pt_tv, p).unwrap() * e[0] * (E::ONE - e[0]),
        );
        verify_component(
            &r.vocabulary,
            &v_pt,
            F::ZERO,
            &[h],
            3,
            30 + idx as u8,
            |p, e| {
                tv::eq(&v_pt_tv, p).unwrap()
                    * (E::ONE - tv_core::selectors::lt(p, vocab as u64).unwrap())
                    * e[0]
            },
        );
        let tok_val = G::from_u64_reduced(u64::from(tokens[row.sources().token_position as usize]));
        verify_component(
            &r.index,
            &[],
            F::from(tok_val),
            &[h],
            2,
            40 + idx as u8,
            |p, e| {
                let index = p
                    .iter()
                    .enumerate()
                    .fold(E::ZERO, |s, (i, &x)| s + E::from(B::reduce(1 << i)) * x);
                index * e[0]
            },
        );
    }

    // 5. Alter one recorded output token at position 3: ExactTemperature refuses with
    //    ERR-WITNESS-EXACT-MISMATCH naming RequestId(19) and TokenPosition(3)!
    tokens[3] = (tokens[3] + 1) % vocab as u32;
    let (bad_lowering, bad_w) = native(&model, &tokens);
    let bad_scorer = ExactTemperature::new(
        &bad_lowering,
        model.quant.public.logits_exponent,
        &[(temperature, exp_spec)],
        &range_specs,
    )
    .unwrap();
    let bad_req = bad_scorer
        .request(RequestId(19), temperature, 2, 3)
        .unwrap();
    let err = bad_req
        .witness((0..3).map(|i| {
            row_inputs(
                &bad_w,
                bad_req.row(i).unwrap().unwrap(),
                vocab,
                draws_u32[i],
            )
        }))
        .unwrap_err();
    assert_eq!(err.code(), ErrorCode::WitnessExactMismatch);
    assert_eq!(err.context().request, Some(RequestId(19)));
    assert_eq!(err.context().position, Some(TokenPosition(3)));
}
