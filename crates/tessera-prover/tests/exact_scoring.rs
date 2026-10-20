// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Actual tiny-model teacher forcing, CPU scoring proofs and independent final checks.
//! These are scoring components; the model reductions/openings remain driver obligations.
use tessera_codec::error::{ErrorCode, RequestId, TokenPosition};
use tessera_codec::{Ascii, DecodeLimits, Secret, SecretBuf, encode, envelope, schema};
use tessera_compiler::{Lowering, PublicModel, lower};
use tessera_field::{Field, Goldilocks as G, GoldilocksExt2 as F, PrimeField, QuadraticExtension};
use tessera_gadgets::{OneHot, OneHotWitness};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_ir::IntegerModel;
use tessera_scoring::{ExactGreedy, GreedyInputs, OutputRow};
use tessera_sumcheck::{Elementwise, Instance, Prefix, Table, prove_batch};
use tessera_tables::{GeneratedTable, Table as _, TableId, TableKind};
use tessera_witness::{Engine, Witness};
use tv_core::{gadgets::argmax::Argmax as TvArgmax, sumcheck as tv, validation::GadgetId};
use tv_field::{Fp as B, Fp2 as E};
use tv_hash::transcript as vt;

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
        a * if i >> j & 1 == 0 { E::ONE - r } else { r }
    })
}
fn mle(xs: &[G], p: &[E]) -> E {
    assert_eq!(xs.len(), 1usize << p.len());
    xs.iter().enumerate().fold(E::ZERO, |s, (i, &x)| {
        s + eq_index(p, i) * E::from(B::from_canonical(x.to_canonical_u64()).unwrap())
    })
}
fn sessions(marker: u8, target: F) -> (Transcript, vt::Transcript) {
    let mut id = [0x71; 16];
    id[15] = marker;
    let mut p = Transcript::new(&SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("test-small/1").unwrap()),
        statement_digest: [0x27; 32],
        session_id: id,
    });
    let mut v = vt::Transcript::new(&vt::SessionContext {
        protocol: vt::Ascii::new("tessera-protocol/1").unwrap(),
        mode: vt::Mode::NonInteractive,
        params: vt::ParamSetId(vt::Ascii::new("test-small/1").unwrap()),
        statement_digest: [0x27; 32],
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
fn component(
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
    let proved = prove_batch(&mut p, Prefix::Stage(1), degree, &[tables.instance()]).unwrap();
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
        let point = checked
            .point_of(column.len().trailing_zeros() as usize)
            .unwrap();
        assert_eq!(value, mle(column, point)); // Discharge every final operand against its original input.
    }
    assert_eq!(p.state(), *v.state());
    assert_eq!(p.counter(), v.counter());
}
fn row_inputs(w: &Witness, row: OutputRow, vocab: usize) -> GreedyInputs {
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
    GreedyInputs {
        logits,
        next_token: OneHotWitness::from_column(indicator),
    }
}
fn native(model: &IntegerModel, tokens: &SecretBuf<u32>) -> (Lowering, Witness) {
    let lowering = lower(PublicModel::of(model), tokens.len() as u32).unwrap();
    let input = Secret::new(tokens.iter().copied().collect());
    let witness = Engine::new(&lowering).unwrap().run(model, &input).unwrap();
    (lowering, witness)
}
#[test]
fn actual_tiny_model_exact_requests_prove_zero_and_one_altered_output_refuses() {
    let bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../spec/vectors/models/tiny-2-servable.d"
    ))
    .unwrap();
    let model: IntegerModel =
        envelope::open(&bytes, &schema::INTEGER_MODEL, DecodeLimits::INTEGER_MODEL).unwrap();
    let vocab = model.arch.vocab as usize;
    let mut tokens: SecretBuf<u32> = [0, 1].into_iter().collect();
    for _ in 0..3 {
        let (lowering, w) = native(&model, &tokens);
        let values = w.tensor(lowering.logits).unwrap();
        let row = &values[(tokens.len() - 1) * vocab..tokens.len() * vocab];
        let mut chosen = 0;
        for i in 1..vocab {
            if row[i] > row[chosen] {
                chosen = i
            }
        }
        tokens.push(chosen as u32);
    }
    let (lowering, w) = native(&model, &tokens);
    let range = GeneratedTable::new(TableId(700), TableKind::Range { bits: 29 }).unwrap();
    let encoded = encode(range.spec());
    let spec = tv_codec::decode(
        &encoded,
        &tv_codec::DecodeLimits::MESSAGE,
        tv_tables::TableSpec::decode,
    )
    .unwrap();
    let range = tv_tables::Table::from_spec(&spec).unwrap();
    let scorer = ExactGreedy::new(
        &lowering,
        GeneratedTable::new(TableId(700), TableKind::Range { bits: 29 })
            .unwrap()
            .spec(),
    )
    .unwrap();
    let request = scorer.request(RequestId(19), 2, 3).unwrap();
    let certificate = request
        .witness((0..3).map(|i| row_inputs(&w, request.row(i).unwrap().unwrap(), vocab)))
        .unwrap();
    assert_eq!(certificate.uhat(), 0);
    let vars = vocab.next_power_of_two().trailing_zeros() as usize;
    let point: Vec<F> = (0..vars)
        .map(|i| {
            F::new(
                G::from_u64_reduced(2 + i as u64),
                G::from_u64_reduced(11 + 3 * i as u64),
            )
        })
        .collect();
    let vp = vectors(&point);
    let site = TvArgmax::derive(GadgetId(99), vocab as u32, 1).unwrap();
    for index in 0..3 {
        let input = row_inputs(&w, request.row(index).unwrap().unwrap(), vocab);
        let row = certificate.prepare_row(index, &input).unwrap();
        let h = row.next_token().indicator();
        let ell = &input.logits;
        let maximum = row.maximum().maximum();
        let evaluated = row.evaluated();
        for &key in evaluated.keys.iter() {
            assert_eq!(
                range.row(key.to_canonical_u64()),
                Some((B::from_canonical(key.to_canonical_u64()).unwrap(), B::ZERO))
            );
        }
        let relations = row.relations::<F>().unwrap();
        let selected = site.selected(&[], tvf(F::from(maximum[0]))).unwrap();
        component(
            &relations.selected,
            &[],
            F::from(maximum[0]),
            &[h, ell],
            2,
            1,
            |p, e| selected.evaluate(p, e).unwrap().value(),
        );
        let key = lift(mle(&evaluated.keys, &vp));
        let lookup = site.lookup(&vp, tvf(key), E::ZERO).unwrap();
        component(
            &relations.key,
            &point,
            key,
            &[maximum, ell, &evaluated.suffix],
            3,
            2,
            |p, e| lookup.evaluate(p, e).unwrap().value(),
        );
        let suffix = lift(mle(&evaluated.suffix, &vp));
        let weights = row.gadget().suffix_weights(&point).unwrap();
        let (mut p, mut v) = sessions(3, suffix);
        let proved = prove_batch(
            &mut p,
            Prefix::Stage(2),
            2,
            &[Instance::new(vec![
                Table::ext(&weights).public(),
                Table::base(h),
            ])],
        )
        .unwrap();
        let proof = proof_of(&proved);
        let bound = site.suffix(&vp, tvf(suffix)).unwrap();
        let checked = tv::verify_batch(
            &mut v,
            tv::Prefix::Stage(2),
            2,
            &[tv::Instance {
                claim: tvf(suffix),
                vars,
                evals: 1,
            }],
            &proof,
            |_, p, e| Some(bound.evaluate(p, e).unwrap().value()),
        )
        .unwrap();
        assert_eq!(proof.final_evals, [mle(h, &checked.point)]);
        assert_eq!(p.state(), *v.state());
        assert_eq!(p.counter(), v.counter());
        // The SAME original indicator is proved against this recorded token.
        let one = OneHot::<G>::new(vocab).unwrap();
        let r = one.relations::<F>(row.next_token()).unwrap();
        let z = vp.clone();
        component(&r.booleanity, &point, F::ZERO, &[h], 3, 4, |p, e| {
            tv::eq(&z, p).unwrap() * e[0] * (E::ONE - e[0])
        });
        component(&r.vocabulary, &point, F::ZERO, &[h], 3, 5, |p, e| {
            tv::eq(&z, p).unwrap()
                * (E::ONE - tv_core::selectors::lt(p, vocab as u64).unwrap())
                * e[0]
        });
        component(&r.count, &[], F::ONE, &[h], 2, 6, |_, e| e[0]);
        let token = G::from_u64_reduced(u64::from(tokens[row.sources().token_position as usize]));
        component(&r.index, &[], F::from(token), &[h], 2, 7, |p, e| {
            let index = p
                .iter()
                .enumerate()
                .fold(E::ZERO, |s, (i, &x)| s + E::from(B::reduce(1 << i)) * x);
            index * e[0]
        });
    }
    tokens[3] = (tokens[3] + 1) % vocab as u32;
    let (lowering, w) = native(&model, &tokens);
    let scorer = ExactGreedy::new(
        &lowering,
        GeneratedTable::new(TableId(700), TableKind::Range { bits: 29 })
            .unwrap()
            .spec(),
    )
    .unwrap();
    let request = scorer.request(RequestId(19), 2, 3).unwrap();
    let error = request
        .witness((0..3).map(|i| row_inputs(&w, request.row(i).unwrap().unwrap(), vocab)))
        .unwrap_err();
    assert_eq!(error.code(), ErrorCode::WitnessExactMismatch);
    assert_eq!(error.context().position, Some(TokenPosition(3)));
}
