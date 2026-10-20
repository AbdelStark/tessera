// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Canonical logits/next-token views against independent shifted-cube proofs.
mod claim_support;
mod support;
use claim_support::{claims_of, integer, items, list, member, number, string};
use support::{Context, digest, extension};
use tv_core::advice::Source;
use tv_core::claims::{ClaimList, QuantPoint};
use tv_core::gadgets::{
    GadgetError, Plan,
    argmax::{Claim, Role},
    argmax_outputs::{Binding, Outputs, View},
};
use tv_core::lowering::{self, Graph, TensorId};
use tv_core::sumcheck::{self, Prefix, SumcheckError, SumcheckProof};
use tv_core::validation::{self, GadgetId};
use tv_field::{Fp, Fp2};
use tv_hash::transcript::{Ascii, Transcript};
use tv_testkit::{Json, read_vectors};
fn field(v: &Json) -> Fp2 {
    let p = items(v);
    Fp2::new(
        Fp::from_canonical(number(&p[0])).unwrap(),
        Fp::from_canonical(number(&p[1])).unwrap(),
    )
}
fn fields(v: &Json) -> Vec<Fp2> {
    items(v).iter().map(field).collect()
}
fn column(v: &Json) -> Vec<Fp> {
    items(v)
        .iter()
        .map(|v| Fp::from_canonical(integer(v).rem_euclid(i128::from(Fp::MODULUS)) as u64).unwrap())
        .collect()
}
fn model(n: u32, v: u32) -> (ClaimList, Graph) {
    let mut c = claims_of("tiny-2-servable.d");
    c.arch.d_model = 2;
    c.arch.n_heads = 1;
    c.arch.n_kv_heads = 1;
    c.arch.d_head = 2;
    c.arch.d_ff = 6;
    c.arch.vocab = v;
    c.arch.max_context = 8;
    c.quant.residual_bits = 8;
    c.quant.residual_exponent = c
        .quant
        .act_exponents
        .iter()
        .find(|(p, _)| *p == QuantPoint::Embed)
        .unwrap()
        .1;
    c.quant.rsqrt_kappa = 0;
    c.quant.exp_sigma = 3;
    for (_, s) in &mut c.quant.shift_class_sets {
        *s = vec![12];
    }
    for (_, s) in &mut c.quant.score_scales {
        *s = tv_codec::Rational::new(1, 1).unwrap();
    }
    let public = validation::check_public(&c).unwrap();
    let g = lowering::lower(&public, n).unwrap();
    (c, g)
}
fn role(v: &Json) -> Role {
    match string(member(v, "role")) {
        "Indicator" => Role::Indicator,
        "Logits" => Role::Logits,
        _ => panic!("role"),
    }
}
fn bind<'a>(context: Outputs, view: &Json, point: &'a [Fp2]) -> View<'a> {
    match context
        .bind(Claim {
            role: role(view),
            point,
            value: field(member(view, "value")),
        })
        .unwrap()
    {
        Binding::Graph(v) => v,
        Binding::Maximum(_) => panic!("not a graph role"),
    }
}
fn before(case: &Json) -> Transcript {
    let mut t = Context::from_json(member(case, "context")).transcript();
    let mut bytes = (list(case, "views").len() as u64).to_le_bytes().to_vec();
    for v in list(case, "views") {
        bytes.extend_from_slice(&field(member(v, "value")).to_le_bytes());
    }
    t.absorb(Ascii::new("stage/1/evals").unwrap(), &bytes);
    t
}
#[test]
fn canonical_teacher_forcing_views_check_every_proof_and_source_endpoint() {
    let fixture = read_vectors("gadget-finals/argmax-views.json");
    let mut count = 0;
    for case in list(&fixture, "cases") {
        let (c, g) = model(number(member(case, "n")), number(member(case, "vocab")));
        let plan = Plan::derive(&g, c.gadget_policy).unwrap();
        let context = plan
            .argmax_outputs(
                GadgetId(number(member(case, "gadget"))),
                number(member(case, "prompt")),
            )
            .unwrap();
        let site = context.argmax();
        assert_eq!(site.rows(), number(member(case, "outputs")));
        assert_eq!(site.vocab(), number(member(case, "vocab")));
        let targets: Vec<_> = list(case, "views")
            .iter()
            .map(|v| fields(member(v, "target")))
            .collect();
        let views: Vec<_> = list(case, "views")
            .iter()
            .zip(&targets)
            .map(|(v, p)| bind(context, v, p))
            .collect();
        let record = member(case, "proof");
        let proof = SumcheckProof {
            rounds: list(record, "rounds").iter().map(fields).collect(),
            final_evals: fields(member(record, "final_evals")),
        };
        let instances: Vec<_> = views.iter().map(|v| v.instance()).collect();
        let mut t = before(case);
        let batch = sumcheck::verify_batch(
            &mut t,
            Prefix::Stage(2),
            2,
            &instances,
            &proof,
            |i, p, e| views[i].evaluate(p, e).ok().map(|v| v.value()),
        )
        .unwrap();
        assert_eq!(batch.point, fields(member(record, "point")));
        assert_eq!(batch.gamma, field(member(record, "gamma")));
        assert_eq!(
            *t.state(),
            digest(member(record, "state").as_str().unwrap())
        );
        assert_eq!(t.counter(), number(member(record, "counter")));
        for (i, (view, expected)) in views.iter().zip(list(case, "views")).enumerate() {
            let final_value = view
                .evaluate(&batch.point, &proof.final_evals[i..i + 1])
                .unwrap();
            assert_eq!(
                final_value.value(),
                field(&list(record, "final_relations")[i])
            );
            let claim = final_value.claim();
            assert_eq!(
                claim.input.source,
                Source::Tensor(TensorId(number(member(expected, "source"))))
            );
            assert_eq!(claim.input.point, fields(member(expected, "final_point")));
            assert_eq!(claim.value, proof.final_evals[i]);
            assert_eq!(
                claim.value,
                extension(&column(member(expected, "column")), claim.input.point)
            );
            assert_eq!(
                field(member(expected, "value")),
                extension(&column(member(expected, "sliced")), &targets[i])
            );
        }
        for i in 0..proof.final_evals.len() {
            let mut changed = proof.clone();
            changed.final_evals[i] += Fp2::ONE;
            match sumcheck::verify_batch(
                &mut before(case),
                Prefix::Stage(2),
                2,
                &instances,
                &changed,
                |j, p, e| views[j].evaluate(p, e).ok().map(|v| v.value()),
            ) {
                Err(e) => assert_eq!(
                    e,
                    SumcheckError::FinalMismatch {
                        round: instances[i].vars as u32
                    }
                ),
                Ok(batch) => {
                    let v = views[i]
                        .evaluate(&batch.point, &changed.final_evals[i..i + 1])
                        .unwrap();
                    let claim = v.claim();
                    assert_ne!(
                        claim.value,
                        extension(
                            &column(member(&list(case, "views")[i], "column")),
                            claim.input.point
                        )
                    );
                }
            }
        }
        count += 1;
    }
    assert_eq!(count, 6);
}
#[test]
fn shifted_copy_kernel_matches_literal_rows_for_every_small_prompt_and_output_interval() {
    for n in 2..=8 {
        let (c, g) = model(n, 3);
        let plan = Plan::derive(&g, c.gadget_policy).unwrap();
        for prompt in 1..n {
            let context = plan.argmax_outputs(GadgetId(73), prompt).unwrap();
            let site = context.argmax();
            let vb = site.variables() - site.row_variables();
            let target: Vec<_> = (0..site.variables())
                .map(|i| Fp2::new(Fp::reduce(11 + 5 * i as u64), Fp::reduce(7 + 3 * i as u64)))
                .collect();
            for (role, offset) in [(Role::Indicator, prompt), (Role::Logits, prompt - 1)] {
                let view = match context
                    .bind(Claim {
                        role,
                        point: &target,
                        value: Fp2::ZERO,
                    })
                    .unwrap()
                {
                    Binding::Graph(v) => v,
                    Binding::Maximum(_) => panic!("graph"),
                };
                let vars = view.instance().vars;
                for source in 0..1usize << vars {
                    let point: Vec<_> = (0..vars)
                        .map(|j| {
                            if source >> j & 1 == 0 {
                                Fp2::ZERO
                            } else {
                                Fp2::ONE
                            }
                        })
                        .collect();
                    let actual = view.evaluate(&point, &[Fp2::ONE]).unwrap().value();
                    let expected =
                        if source >= offset as usize && source < (offset + site.rows()) as usize {
                            let output = source - offset as usize;
                            target[vb..]
                                .iter()
                                .enumerate()
                                .fold(Fp2::ONE, |a, (j, &x)| {
                                    a * if output >> j & 1 == 0 {
                                        Fp2::ONE - x
                                    } else {
                                        x
                                    }
                                })
                        } else {
                            Fp2::ZERO
                        };
                    assert_eq!(
                        actual, expected,
                        "n={n} prompt={prompt} role={role:?} source={source}"
                    );
                }
            }
        }
    }
}
#[test]
fn every_view_keeps_maximum_debts_and_rejects_wrong_own_geometry_before_proof_counts() {
    let fixture = read_vectors("gadget-finals/argmax-views.json");
    for case in list(&fixture, "cases") {
        let n = number(member(case, "n"));
        let (c, g) = model(n, number(member(case, "vocab")));
        let plan = Plan::derive(&g, c.gadget_policy).unwrap();
        let gid = GadgetId(number(member(case, "gadget")));
        let context = plan
            .argmax_outputs(gid, number(member(case, "prompt")))
            .unwrap();
        let site = context.argmax();
        for prompt in [0, n, n + 1] {
            assert!(matches!(
                plan.argmax_outputs(gid, prompt),
                Err(GadgetError::Internal)
            ));
        }
        let row = vec![Fp2::ONE; site.row_variables()];
        let maximum = match context
            .bind(Claim {
                role: Role::Maximum,
                point: &row,
                value: Fp2::ONE,
            })
            .unwrap()
        {
            Binding::Maximum(m) => m,
            Binding::Graph(_) => panic!("maximum"),
        };
        assert_eq!(maximum.gadget, gid);
        assert_eq!(maximum.point, row);
        assert_eq!(maximum.value, Fp2::ONE);
        let full = vec![Fp2::ONE; site.variables()];
        assert!(matches!(
            context.bind(Claim {
                role: Role::Suffix,
                point: &full,
                value: Fp2::ZERO
            }),
            Err(GadgetError::Internal)
        ));
        for (role, vars) in [
            (Role::Maximum, site.row_variables()),
            (Role::Logits, site.variables()),
            (Role::Indicator, site.variables()),
        ] {
            let wrong = vec![Fp2::ONE; vars + 1];
            assert!(matches!(
                context.bind(Claim {
                    role,
                    point: &wrong,
                    value: Fp2::ZERO
                }),
                Err(GadgetError::Internal)
            ));
        }
        for expected in list(case, "views") {
            let target = fields(member(expected, "target"));
            let view = bind(context, expected, &target);
            let point = fields(member(member(case, "proof"), "point"));
            let mut wrong = point.clone();
            wrong.push(Fp2::ZERO);
            assert!(matches!(
                view.evaluate(&wrong, &[]),
                Err(GadgetError::Internal)
            ));
            for values in [&[][..], &[Fp2::ZERO; 2][..]] {
                assert!(
                    matches!(view.evaluate(&point,values),Err(GadgetError::Rejected{gadget})if gadget==gid)
                );
            }
        }
    }
}
