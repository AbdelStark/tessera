// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0009 Lemmas 8–9: adversarial assignments and real sumcheck subproofs checked independently.
//! The driver must bind the resulting operand claims to commitments and draw zero-check points
//! after those commitments (RFC-0014); these tests supply fixed public points to the components.

mod common;
use common::{Fp, Fp2, Session, SplitMix64, TvFp2, mle_fp, tv_fp2, tv_fp2s};
use tessera_codec::SecretBuf;
use tessera_field::{F65537, Field, PrimeField, QuadraticExtension};
use tessera_gadgets::{Argmax, ArgmaxWitness, OneHot, OneHotWitness, suffix_kernel};
use tessera_sumcheck::{Elementwise, Prefix, prove_batch};
use tessera_tables::{GeneratedTable, Table, TableId, TableKind};
use tv_core::sumcheck as tv;

fn fp(x: i64) -> Fp {
    Fp::from_i64(x)
}
fn buf(xs: &[i64]) -> SecretBuf<Fp> {
    xs.iter().copied().map(fp).collect()
}
fn argmax(vocab: usize) -> Argmax<Fp> {
    let table = GeneratedTable::new(TableId(7), TableKind::Range { bits: 29 }).unwrap();
    Argmax::new(vocab, table.spec()).unwrap()
}
fn one_hot_passes<B: PrimeField>(h: &[B], t: i64, real: bool, vocab: usize) -> bool {
    let sum = h.iter().copied().fold(B::ZERO, |a, b| a + b);
    let index = h
        .iter()
        .enumerate()
        .fold(B::ZERO, |a, (v, &h)| a + B::from_u64_reduced(v as u64) * h);
    h.iter().all(|&h| h * (B::ONE - h) == B::ZERO)
        && sum == B::from_u64_reduced(u64::from(real))
        && index == B::from_i64(t)
        && h[vocab..].iter().all(|&x| x == B::ZERO)
}

fn subproof(
    relation: &Elementwise<'_, Fp2>,
    point: &[Fp2],
    target: Fp2,
    operands: &[&[Fp]],
    mut final_relation: impl FnMut(&[TvFp2], &[TvFp2]) -> TvFp2,
) -> Result<(), tv::SumcheckError> {
    let session = Session::random(&mut SplitMix64::new(0x123_0001));
    let tables = relation.tables(point).unwrap();
    let mut transcript = session.transcript();
    let degree = relation.degree();
    let proven = prove_batch(
        &mut transcript,
        Prefix::Stage(0),
        degree,
        &[tables.instance()],
    )
    .unwrap();
    let proof = tv::SumcheckProof {
        rounds: proven.proof().rounds.iter().map(|r| tv_fp2s(r)).collect(),
        final_evals: tv_fp2s(&proven.proof().final_evals),
    };
    let mut verifier = session.tv_transcript();
    let checked = tv::verify_batch(
        &mut verifier,
        tv::Prefix::Stage(0),
        degree,
        &[tv::Instance {
            claim: tv_fp2(target),
            vars: relation.shape().vars,
            evals: operands.len(),
        }],
        &proof,
        |_, r, e| Some(final_relation(r, e)),
    )?;
    for (column, &value) in operands.iter().zip(&proof.final_evals) {
        let r = checked
            .point_of(column.len().trailing_zeros() as usize)
            .unwrap();
        assert_eq!(value, mle_fp(column, r), "consume each operand claim");
    }
    Ok(())
}

#[test]
fn one_hot_has_one_assignment_in_both_fields_by_exhaustion() {
    fn exhaust<B: PrimeField>() {
        let g = OneHot::<B>::new(4).unwrap();
        for t in 0..4 {
            for real in [false, true] {
                if !real && t != 0 {
                    assert!(g.witness(t, real).is_err());
                    continue;
                }
                let honest = g.witness(t, real).unwrap();
                let mut found = 0;
                for word in 0..256 {
                    let h: Vec<_> = (0..4)
                        .map(|j| B::from_i64(((word >> (2 * j)) & 3) - 1))
                        .collect();
                    if one_hot_passes(&h, t, real, 4) {
                        found += 1;
                        assert_eq!(h, honest.indicator());
                    }
                }
                assert_eq!(found, 1);
            }
        }
    }
    exhaust::<Fp>();
    exhaust::<F65537>();
}

#[test]
fn one_hot_sumcheck_isolates_two_ones_wrong_index_and_non_boolean_entries() {
    let g = OneHot::<Fp>::new(4).unwrap();
    let z = [Fp2::from(fp(2)), Fp2::from(fp(3))];
    let tv_z = tv_fp2s(&z);
    for (values, token, real, bad) in [
        ([0, 1, 0, 0], 1, true, None),
        ([0, 0, 0, 0], 0, false, None),
        ([1, 0, 1, 0], 2, true, Some(1)),   // count alone
        ([0, 1, 0, 0], 2, true, Some(2)),   // index alone
        ([-1, 2, 0, 0], 2, true, Some(0)),  // booleanity alone
        ([1, -2, 1, 0], 0, false, Some(0)), // padded count/index0, but non-Boolean
    ] {
        let w = OneHotWitness::from_column(buf(&values));
        let r = g.relations::<Fp2>(&w).unwrap();
        let booleans = subproof(&r.booleanity, &z, Fp2::ZERO, &[w.indicator()], |p, e| {
            tv::eq(&tv_z, p).unwrap() * e[0] * (TvFp2::ONE - e[0])
        });
        let count = subproof(
            &r.count,
            &[],
            Fp2::from(fp(i64::from(real))),
            &[w.indicator()],
            |_, e| e[0],
        );
        let index = subproof(
            &r.index,
            &[],
            Fp2::from(fp(token)),
            &[w.indicator()],
            |p, e| {
                let public = p.iter().enumerate().fold(TvFp2::ZERO, |a, (j, &r)| {
                    a + r * TvFp2::from(common::tv_fp(Fp::from_u64_reduced(1 << j)))
                });
                e[0] * public
            },
        );
        for (i, result) in [booleans, count, index].into_iter().enumerate() {
            assert_eq!(result.is_err(), bad == Some(i));
            if let Err(error) = result {
                assert_eq!(error, tv::SumcheckError::FinalMismatch { round: 2 });
            }
        }
    }
}

#[test]
fn every_three_logit_vector_and_recorded_token_chooses_the_first_maximum() {
    let a = argmax(3);
    let one = OneHot::<Fp>::new(3).unwrap();
    for x in -3..=3 {
        for y in -3..=3 {
            for z in -3..=3 {
                let logits = buf(&[x, y, z, 0]);
                let maximum = x.max(y).max(z);
                let first = [x, y, z].iter().position(|&v| v == maximum).unwrap();
                for t in 0..3 {
                    let h = one.witness(t, true).unwrap();
                    for proposed in -4..=4 {
                        let m = ArgmaxWitness::from_column(buf(&[proposed]));
                        let e = a.evaluate(&logits, &h, &m).unwrap();
                        let range = e.keys.iter().all(|v| v.to_canonical_u64() < 1 << 29);
                        let selected = h
                            .indicator()
                            .iter()
                            .zip(logits.iter())
                            .fold(Fp::ZERO, |s, (&h, &l)| s + h * l)
                            == m.maximum()[0];
                        assert_eq!(
                            range && selected,
                            t as usize == first && proposed == maximum
                        );
                        assert_eq!(
                            e.keys[3],
                            Fp::ZERO,
                            "vocabulary padding is public Range row0"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn later_ties_and_wrong_maxima_fail_exactly_their_required_obligation() {
    let g = argmax(3);
    let one = OneHot::<Fp>::new(3).unwrap();
    let logits = buf(&[-3, -1, -1, 0]);
    let maximum = g.witness(&logits, true).unwrap();
    let later = one.witness(2, true).unwrap();
    let evaluated = g.evaluate(&logits, &later, &maximum).unwrap();
    assert_eq!(evaluated.keys[1], -Fp::ONE);
    assert_eq!(
        evaluated
            .keys
            .iter()
            .filter(|k| k.to_canonical_u64() >= 1 << 29)
            .count(),
        1
    );
    assert_eq!(
        logits[2],
        maximum.maximum()[0],
        "selected equality passes for a later tie"
    );
    let h = one.witness(1, true).unwrap();
    let bad = ArgmaxWitness::from_column(buf(&[0]));
    let evaluated = g.evaluate(&logits, &h, &bad).unwrap();
    assert!(
        evaluated
            .keys
            .iter()
            .all(|k| k.to_canonical_u64() < 1 << 29)
    );
    let relations = g.relations::<Fp2>(&logits, &h, &bad, &evaluated).unwrap();
    assert_eq!(
        subproof(
            &relations.selected,
            &[],
            Fp2::ZERO,
            &[h.indicator(), &logits],
            |_, e| e[0] * e[1]
        ),
        Err(tv::SumcheckError::FinalMismatch { round: 2 })
    );
}

#[test]
fn suffix_reduction_matches_the_linear_operator_at_boolean_and_extension_points() {
    let g = argmax(5);
    let h = OneHot::<Fp>::new(5).unwrap().witness(3, true).unwrap();
    let logits = buf(&[-7, -6, -5, -1, -2, 0, 0, 0]);
    let m = g.witness(&logits, true).unwrap();
    let e = g.evaluate(&logits, &h, &m).unwrap();
    for a in 0..8 {
        let rho: Vec<Fp2> = (0..3).map(|j| Fp2::from(fp((a >> j) & 1))).collect();
        let weights = g.suffix_weights(&rho).unwrap();
        assert_eq!(
            weights
                .iter()
                .zip(h.indicator())
                .fold(Fp2::ZERO, |s, (&w, &h)| s + w * Fp2::from(h)),
            Fp2::from(e.suffix[a as usize])
        );
    }
    let rho = vec![
        Fp2::new(fp(2), fp(17)),
        Fp2::new(fp(3), fp(19)),
        Fp2::new(fp(5), fp(23)),
    ];
    let point = vec![
        Fp2::new(fp(7), fp(29)),
        Fp2::new(fp(11), fp(31)),
        Fp2::new(fp(13), fp(37)),
    ];
    let weights = g.suffix_weights(&rho).unwrap();
    let expected = mle_fp(&e.suffix, &tv_fp2s(&rho));
    let dot = weights
        .iter()
        .zip(h.indicator())
        .fold(Fp2::ZERO, |s, (&w, &h)| s + w * Fp2::from(h));
    assert_eq!(tv_fp2(dot), expected);
    let eq = |p: &[Fp2], v: usize| {
        p.iter().enumerate().fold(Fp2::ONE, |r, (j, &x)| {
            r * if (v >> j) & 1 == 1 { x } else { Fp2::ONE - x }
        })
    };
    let direct = (0..8)
        .map(|v| (0..v).fold(Fp2::ZERO, |s, i| s + eq(&rho, i)) * eq(&point, v))
        .fold(Fp2::ZERO, |s, v| s + v);
    assert_eq!(suffix_kernel(&rho, &point).unwrap(), direct);
}

#[test]
fn key_reduction_uses_masked_pointwise_products_and_binds_all_operands() {
    let g = argmax(3);
    let h = OneHot::<Fp>::new(3).unwrap().witness(1, true).unwrap();
    let logits = buf(&[-3, -1, -1, 0]);
    let m = g.witness(&logits, true).unwrap();
    let e = g.evaluate(&logits, &h, &m).unwrap();
    let r = g.relations::<Fp2>(&logits, &h, &m, &e).unwrap();
    let rho = [Fp2::from(fp(2)), Fp2::from(fp(3))];
    let target = common::fp2(mle_fp(&e.keys, &tv_fp2s(&rho)));
    let tv_rho = tv_fp2s(&rho);
    subproof(
        &r.key,
        &rho,
        target,
        &[m.maximum(), &logits, &e.suffix],
        |point, evals| {
            let mask = mle_fp(&[Fp::ONE, Fp::ONE, Fp::ONE, Fp::ZERO], point);
            tv::eq(&tv_rho, point).unwrap() * mask * (evals[0] - evals[1] - evals[2])
        },
    )
    .unwrap();
    subproof(
        &r.selected,
        &[],
        Fp2::from(m.maximum()[0]),
        &[h.indicator(), &logits],
        |_, e| e[0] * e[1],
    )
    .unwrap();
}

#[test]
fn vocabulary_128256_at_llama_8b_shape_handles_ties_and_padding_without_dense_tables() {
    // These per-row certificates depend on the 8B output width, not its hidden-layer widths.
    let vocab = 128_256;
    let g = argmax(vocab);
    let one = OneHot::<Fp>::new(vocab).unwrap();
    assert_eq!(one.padded_len(), 131_072);
    let mut logits: SecretBuf<_> = (0..one.padded_len())
        .map(|i| fp(if i < vocab { -7 } else { 0 }))
        .collect();
    for v in [65_535, vocab - 1] {
        logits[v] = fp(-1);
    }
    let maximum = g.witness(&logits, true).unwrap();
    assert_eq!(maximum.maximum(), [fp(-1)]);
    for (token, passes) in [(65_535, true), (vocab - 1, false)] {
        let h = one.witness(token as i64, true).unwrap();
        let e = g.evaluate(&logits, &h, &maximum).unwrap();
        assert_eq!(
            e.keys.iter().all(|k| k.to_canonical_u64() < 1 << 29),
            passes
        );
        assert!(e.keys[vocab..].iter().all(|k| *k == Fp::ZERO));
        if passes {
            let vars = one.padded_len().trailing_zeros() as usize;
            let z: Vec<_> = (0..vars)
                .map(|j| Fp2::new(fp(j as i64 + 2), fp(j as i64 + 23)))
                .collect();
            let tv_z = tv_fp2s(&z);
            let relations = one.relations::<Fp2>(&h).unwrap();
            subproof(
                &relations.booleanity,
                &z,
                Fp2::ZERO,
                &[h.indicator()],
                |p, e| tv::eq(&tv_z, p).unwrap() * e[0] * (TvFp2::ONE - e[0]),
            )
            .unwrap();
            subproof(&relations.count, &[], Fp2::ONE, &[h.indicator()], |_, e| {
                e[0]
            })
            .unwrap();
            subproof(
                &relations.index,
                &[],
                Fp2::from(fp(token as i64)),
                &[h.indicator()],
                |p, e| {
                    let index = p.iter().enumerate().fold(TvFp2::ZERO, |s, (j, &r)| {
                        s + r * TvFp2::from(common::tv_fp(Fp::from_u64_reduced(1 << j)))
                    });
                    e[0] * index
                },
            )
            .unwrap();
            let valid: Vec<_> = (0..one.padded_len())
                .map(|v| Fp::from_u64_reduced(u64::from(v < vocab)))
                .collect();
            subproof(
                &relations.vocabulary,
                &z,
                Fp2::ZERO,
                &[h.indicator()],
                |p, e| tv::eq(&tv_z, p).unwrap() * (TvFp2::ONE - mle_fp(&valid, p)) * e[0],
            )
            .unwrap();
            let relations = g.relations::<Fp2>(&logits, &h, &maximum, &e).unwrap();
            subproof(
                &relations.selected,
                &[],
                Fp2::from(fp(-1)),
                &[h.indicator(), &logits],
                |_, e| e[0] * e[1],
            )
            .unwrap();
        }
    }
    assert_eq!(g.table().row_count(), 1 << 29);
    let zero = SecretBuf::filled(Fp::ZERO, one.padded_len());
    let h = one.witness(0, false).unwrap();
    let m = g.witness(&zero, false).unwrap();
    let e = g.evaluate(&zero, &h, &m).unwrap();
    assert!(e.keys.iter().all(|&x| x == Fp::ZERO));
}

#[test]
fn constructors_shapes_and_goldilocks_integer_boundaries_are_checked() {
    for vocab in [0, (1 << 18) + 1, usize::MAX] {
        assert!(OneHot::<Fp>::new(vocab).is_err());
    }
    assert!(OneHot::<F65537>::new(65_537).is_err());
    assert!(OneHot::<F65537>::new(65_536).is_ok());
    let one = OneHot::<Fp>::new(3).unwrap();
    for token in [-1, 3, i64::MIN, i64::MAX] {
        assert!(one.witness(token, true).is_err());
    }
    assert!(one.witness(1, false).is_err());
    let g = argmax(2);
    let end = (1 << 28) - 1;
    for logits in [[-end, end], [end, -end]] {
        let l = buf(&logits);
        let m = g.witness(&l, true).unwrap();
        let t = if logits[0] > logits[1] { 0 } else { 1 };
        let h = OneHot::<Fp>::new(2).unwrap().witness(t, true).unwrap();
        let e = g.evaluate(&l, &h, &m).unwrap();
        assert!(e.keys.iter().all(|k| k.to_canonical_u64() < 1 << 29));
    }
    for bad in [1 << 28, -(1 << 28), i64::MIN, i64::MAX] {
        assert!(g.witness(&buf(&[bad, 0]), true).is_err());
    }
    assert!(g.witness(&buf(&[1, 0]), false).is_err());
    assert!(g.witness(&buf(&[0]), true).is_err());
    let pad = one.witness(0, false).unwrap();
    assert!(!format!("{pad:?}").contains("indicator:"));
    assert!(suffix_kernel::<Fp2>(&[], &[Fp2::ZERO]).is_err());
    assert!(suffix_kernel::<Fp2>(&[Fp2::ZERO; 19], &[Fp2::ZERO; 19]).is_err());
    assert_eq!(
        suffix_kernel::<Fp2>(&[Fp2::ZERO; 18], &[Fp2::ZERO; 18]).unwrap(),
        Fp2::ZERO
    );
    let canary = fp(1_234_567_897);
    let secret_h = OneHotWitness::from_column([canary].into_iter().collect());
    let secret_m = ArgmaxWitness::from_column([canary].into_iter().collect());
    let secret_e = tessera_gadgets::ArgmaxEvaluation {
        suffix: [canary].into_iter().collect(),
        keys: [canary].into_iter().collect(),
    };
    let needle = format!("{canary:?}");
    for debug in [
        format!("{secret_h:?}"),
        format!("{secret_m:?}"),
        format!("{secret_e:?}"),
    ] {
        assert!(
            !debug.contains(&needle),
            "secret class values never enter Debug"
        );
    }
}

#[test]
fn vocabulary_padding_alone_rejects_a_padded_token_index() {
    let g = OneHot::<Fp>::new(3).unwrap();
    let z = [Fp2::from(fp(2)), Fp2::from(fp(3))];
    let tv_z = tv_fp2s(&z);
    for token in [0, 2, 3] {
        let mut column = SecretBuf::filled(Fp::ZERO, 4);
        column[token] = Fp::ONE;
        let w = OneHotWitness::from_column(column);
        let r = g.relations::<Fp2>(&w).unwrap();
        subproof(&r.booleanity, &z, Fp2::ZERO, &[w.indicator()], |p, e| {
            tv::eq(&tv_z, p).unwrap() * e[0] * (TvFp2::ONE - e[0])
        })
        .unwrap();
        subproof(&r.count, &[], Fp2::ONE, &[w.indicator()], |_, e| e[0]).unwrap();
        subproof(
            &r.index,
            &[],
            Fp2::from(fp(token as i64)),
            &[w.indicator()],
            |p, e| e[0] * (p[0] + p[1] + p[1]),
        )
        .unwrap();
        let result = subproof(&r.vocabulary, &z, Fp2::ZERO, &[w.indicator()], |p, e| {
            tv::eq(&tv_z, p).unwrap()
                * (TvFp2::ONE - mle_fp(&[Fp::ONE, Fp::ONE, Fp::ONE, Fp::ZERO], p))
                * e[0]
        });
        assert_eq!(result.is_ok(), token < 3);
        if token == 3 {
            assert_eq!(result, Err(tv::SumcheckError::FinalMismatch { round: 2 }));
        }
    }
}

#[test]
fn suffix_dot_product_is_proved_against_the_independent_triangular_kernel() {
    use tessera_sumcheck::{Instance, Table as Factor};
    let g = argmax(5);
    let h = OneHot::<Fp>::new(5).unwrap().witness(3, true).unwrap();
    let logits = buf(&[-7, -6, -5, -1, -2, 0, 0, 0]);
    let m = g.witness(&logits, true).unwrap();
    let e = g.evaluate(&logits, &h, &m).unwrap();
    let rho = [
        Fp2::new(fp(2), fp(17)),
        Fp2::new(fp(3), fp(19)),
        Fp2::new(fp(5), fp(23)),
    ];
    let weights = g.suffix_weights(&rho).unwrap();
    let session = Session::random(&mut SplitMix64::new(0x123_0002));
    let mut transcript = session.transcript();
    let p = prove_batch(
        &mut transcript,
        Prefix::Stage(0),
        2,
        &[Instance::new(vec![
            Factor::ext(&weights).public(),
            Factor::base(h.indicator()),
        ])],
    )
    .unwrap();
    let proof = tv::SumcheckProof {
        rounds: p.proof().rounds.iter().map(|r| tv_fp2s(r)).collect(),
        final_evals: tv_fp2s(&p.proof().final_evals),
    };
    let tv_rho = tv_fp2s(&rho);
    let prefix: Vec<_> = {
        let eq = common::eq_weights(&tv_rho);
        let mut sum = TvFp2::ZERO;
        eq.into_iter()
            .map(|e| {
                let v = sum;
                sum += e;
                v
            })
            .collect()
    };
    let mut verifier = session.tv_transcript();
    let checked = tv::verify_batch(
        &mut verifier,
        tv::Prefix::Stage(0),
        2,
        &[tv::Instance {
            claim: mle_fp(&e.suffix, &tv_rho),
            vars: 3,
            evals: 1,
        }],
        &proof,
        |_, r, eval| {
            let eq = common::eq_weights(r);
            let kernel = prefix
                .iter()
                .zip(&eq)
                .fold(TvFp2::ZERO, |s, (&a, &b)| s + a * b);
            Some(kernel * eval[0])
        },
    )
    .unwrap();
    assert_eq!(proof.final_evals[0], mle_fp(h.indicator(), &checked.point));
}

#[test]
fn malformed_columns_and_table_specs_return_public_typed_errors() {
    use tessera_gadgets::{ArgmaxEvaluation, GadgetError};
    let one = OneHot::<Fp>::new(3).unwrap();
    let g = argmax(3);
    let h = one.witness(0, true).unwrap();
    let l = buf(&[1, 0, 0, 0]);
    let m = g.witness(&l, true).unwrap();
    for len in [0, 3, 5] {
        let h_bad = OneHotWitness::from_column(SecretBuf::filled(Fp::ZERO, len));
        assert!(matches!(
            one.relations::<Fp2>(&h_bad),
            Err(GadgetError::Shape {
                rule: "a choice column has pad(vocab) entries"
            })
        ));
        assert!(matches!(
            g.evaluate(&l, &h_bad, &m),
            Err(GadgetError::Shape {
                rule: "a choice column has pad(vocab) entries"
            })
        ));
    }
    for len in [0, 2] {
        let m_bad = ArgmaxWitness::from_column(SecretBuf::filled(Fp::ZERO, len));
        assert!(matches!(
            g.evaluate(&l, &h, &m_bad),
            Err(GadgetError::Shape {
                rule: "Argmax commits one maximum per row"
            })
        ));
    }
    for (keys, suffix) in [(3, 4), (4, 3)] {
        let malformed = ArgmaxEvaluation {
            keys: SecretBuf::filled(Fp::ZERO, keys),
            suffix: SecretBuf::filled(Fp::ZERO, suffix),
        };
        assert!(matches!(
            g.relations::<Fp2>(&l, &h, &m, &malformed),
            Err(GadgetError::Shape {
                rule: "a choice column has pad(vocab) entries"
            })
        ));
    }
    let bad = GeneratedTable::new(TableId(7), TableKind::Range { bits: 28 }).unwrap();
    assert!(matches!(
        Argmax::<Fp>::new(3, bad.spec()),
        Err(GadgetError::Table {
            table: TableId(7),
            ..
        })
    ));
    let mut bad = GeneratedTable::new(TableId(7), TableKind::Range { bits: 29 })
        .unwrap()
        .spec()
        .clone();
    bad.digest[0] ^= 1;
    assert!(matches!(
        Argmax::<Fp>::new(3, &bad),
        Err(GadgetError::Spec {
            table: TableId(7),
            ..
        })
    ));
    let good = GeneratedTable::new(TableId(7), TableKind::Range { bits: 29 }).unwrap();
    assert!(matches!(
        Argmax::<F65537>::new(3, good.spec()),
        Err(GadgetError::Shape { .. })
    ));
    assert!(g.suffix_weights::<Fp2>(&[Fp2::ZERO]).is_err());
}

#[test]
fn teacher_forcing_borrows_the_next_recorded_tokens_existing_indicator() {
    let one = OneHot::<Fp>::new(3).unwrap();
    let a = argmax(3);
    let tokens = [2, 0, 1, 2];
    let recorded: Vec<_> = tokens
        .iter()
        .map(|&t| one.witness(t, true).unwrap())
        .collect();
    for p in 0..tokens.len() - 1 {
        let mut logits = SecretBuf::filled(fp(-3), 4);
        logits[3] = Fp::ZERO;
        logits[tokens[p + 1] as usize] = fp(1);
        let maximum = a.witness(&logits, true).unwrap();
        let valid = |h: &OneHotWitness<Fp>| {
            let e = a.evaluate(&logits, h, &maximum).unwrap();
            let selected = h
                .indicator()
                .iter()
                .zip(logits.iter())
                .fold(Fp::ZERO, |s, (&h, &l)| s + h * l);
            e.keys.iter().all(|k| k.to_canonical_u64() < 1 << 29)
                && selected == maximum.maximum()[0]
        };
        assert!(valid(&recorded[p + 1]));
        assert!(
            !valid(&recorded[p]),
            "using the previous token's h changes the certificate"
        );
    }
}

#[test]
fn scalar_vocabulary_uses_zero_variable_relations_without_panicking() {
    let one = OneHot::<Fp>::new(1).unwrap();
    let a = argmax(1);
    for real in [false, true] {
        let h = one.witness(0, real).unwrap();
        let r = one.relations::<Fp2>(&h).unwrap();
        subproof(&r.booleanity, &[], Fp2::ZERO, &[h.indicator()], |_, e| {
            e[0] * (TvFp2::ONE - e[0])
        })
        .unwrap();
        subproof(&r.vocabulary, &[], Fp2::ZERO, &[h.indicator()], |_, _| {
            TvFp2::ZERO
        })
        .unwrap();
        subproof(
            &r.count,
            &[],
            Fp2::from(fp(i64::from(real))),
            &[h.indicator()],
            |_, e| e[0],
        )
        .unwrap();
        subproof(&r.index, &[], Fp2::ZERO, &[h.indicator()], |_, _| {
            TvFp2::ZERO
        })
        .unwrap();
        let logits = buf(&[if real { -13 } else { 0 }]);
        let m = a.witness(&logits, real).unwrap();
        let e = a.evaluate(&logits, &h, &m).unwrap();
        assert_eq!(e.keys[0], Fp::ZERO);
        let r = a.relations::<Fp2>(&logits, &h, &m, &e).unwrap();
        subproof(
            &r.selected,
            &[],
            Fp2::from(m.maximum()[0]),
            &[h.indicator(), &logits],
            |_, e| e[0] * e[1],
        )
        .unwrap();
        subproof(
            &r.key,
            &[],
            Fp2::ZERO,
            &[m.maximum(), &logits, &e.suffix],
            |_, e| e[0] - e[1] - e[2],
        )
        .unwrap();
    }
    assert_eq!(a.suffix_weights::<Fp2>(&[]).unwrap(), [Fp2::ZERO]);
    assert_eq!(suffix_kernel::<Fp2>(&[], &[]).unwrap(), Fp2::ZERO);
}
