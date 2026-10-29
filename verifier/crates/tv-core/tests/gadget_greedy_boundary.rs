// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Literal Python operators precede any producer comparison (RFC-0001, SP-6).
mod claim_support;

use tv_core::claims::{ClaimList, QuantPoint};
use tv_core::gadgets::{
    GadgetError, Plan,
    greedy::{Boundary, Kind, Source},
};
use tv_core::lowering::{self, Graph};
use tv_core::validation::{self, GadgetId};
use tv_field::{Fp, Fp2};
use tv_testkit::{Json, read_vectors};

fn member<'a>(j: &'a Json, name: &str) -> &'a Json {
    j.get(name).unwrap()
}
fn list(j: &Json) -> &[Json] {
    j.as_array().unwrap()
}
fn number(j: &Json) -> u64 {
    j.as_u64()
        .unwrap_or_else(|| j.as_str().unwrap().parse().unwrap())
}
fn field(j: &Json) -> Fp2 {
    let a = list(j);
    Fp2::new(
        Fp::from_canonical(number(&a[0])).unwrap(),
        Fp::from_canonical(number(&a[1])).unwrap(),
    )
}
fn fields(j: &Json) -> Vec<Fp2> {
    list(j).iter().map(field).collect()
}
fn model(n: u32, vocab: u32) -> (ClaimList, Graph) {
    let mut c = claim_support::claims_of("tiny-2-servable.d");
    c.arch.d_model = 2;
    c.arch.n_heads = 1;
    c.arch.n_kv_heads = 1;
    c.arch.d_head = 2;
    c.arch.d_ff = 6;
    c.arch.vocab = vocab;
    c.arch.max_context = n.max(8);
    c.quant.residual_bits = 8;
    c.quant.rsqrt_kappa = 0;
    c.quant.exp_sigma = 3;
    c.quant.residual_exponent = c
        .quant
        .act_exponents
        .iter()
        .find(|(p, _)| *p == QuantPoint::Embed)
        .unwrap()
        .1;
    for (_, s) in &mut c.quant.shift_class_sets {
        *s = vec![12];
    }
    for (_, s) in &mut c.quant.score_scales {
        *s = tv_codec::Rational::new(1, 1).unwrap();
    }
    let public = validation::check_public(&c).unwrap();
    let graph = lowering::lower(&public, n).unwrap();
    (c, graph)
}
fn boundary(case: &Json) -> (Graph, Boundary) {
    let (claims, graph) = model(
        number(member(case, "n")) as u32,
        number(member(case, "vocab")) as u32,
    );
    let plan = Plan::derive(&graph, claims.gadget_policy).unwrap();
    let boundary = plan
        .greedy_boundary(GadgetId(73), number(member(case, "prompt")) as u32)
        .unwrap();
    (graph, boundary)
}

#[test]
fn native_polynomials_and_every_original_endpoint_match_literal_independent_sums() {
    let f = read_vectors("greedy-boundary/greedy-boundary.json");
    assert_eq!(list(member(&f, "cases")).len(), 32);
    for case in list(member(&f, "cases")) {
        let (graph, b) = boundary(case);
        let first = graph.nodes().first().unwrap();
        let onehot = *first.outputs().first().unwrap();
        let tokens = *first.inputs().first().unwrap();
        let last = graph.nodes().last().unwrap();
        let logits = *last.outputs().first().unwrap();
        let input = last.inputs()[0];
        let clamp = graph
            .nodes()
            .iter()
            .find(|n| n.outputs().contains(&input))
            .unwrap();
        let lowering::Op::Requant { lo, hi, .. } = *clamp.op() else {
            panic!("final norm clamp")
        };
        let weight = graph
            .inputs()
            .iter()
            .find(|i| i.tensor == last.inputs()[1])
            .unwrap();
        let products = [
            i128::from(lo) * i128::from(weight.lo),
            i128::from(lo) * i128::from(weight.hi),
            i128::from(hi) * i128::from(weight.lo),
            i128::from(hi) * i128::from(weight.hi),
        ];
        let fan = i128::from(*graph.tensor(input).unwrap().shape.dims().last().unwrap());
        let span =
            u128::try_from((products.iter().max().unwrap() - products.iter().min().unwrap()) * fan)
                .unwrap();
        assert_eq!(u32::from(b.range_bits()), u128::BITS - span.leading_zeros());
        let point = fields(member(case, "point"));
        let z = fields(member(case, "zero"));
        let mut target = fields(member(case, "target_v"));
        target.extend(fields(member(case, "target_o")));
        let raw = fields(member(case, "raw"));
        let traw = fields(member(case, "token_raw"));
        for (kind, check, values, expected) in [
            (
                Kind::Key,
                b.key(&target, field(member(case, "key_claim"))).unwrap(),
                &raw,
                field(member(case, "key_final")),
            ),
            (
                Kind::Selected,
                b.zero(Kind::Selected, &z).unwrap(),
                &raw,
                field(member(case, "selected_final")),
            ),
            (
                Kind::Tokens,
                b.zero(Kind::Tokens, &z).unwrap(),
                &traw,
                field(member(case, "token_final")),
            ),
        ] {
            assert_eq!(check.instance().vars, point.len());
            assert_eq!(check.instance().evals, values.len());
            assert_eq!(
                check.instance().claim,
                if kind == Kind::Key {
                    field(member(case, "key_claim"))
                } else {
                    Fp2::ZERO
                }
            );
            assert_eq!(b.degree(kind), if kind == Kind::Selected { 3 } else { 2 });
            let result = check.evaluate(&point, values).unwrap();
            assert_eq!(result.value, expected, "{kind:?} on {case:?}");
            assert_eq!(result.claims.len(), values.len());
            if kind == Kind::Tokens {
                assert_eq!(result.claims[0].source, Source::OneHot(onehot));
                assert_eq!(result.claims[0].point, point);
                assert_eq!(result.claims[0].value, traw[0]);
                assert_eq!(result.claims[1].source, Source::Tokens(tokens));
                assert_eq!(result.claims[1].point, point[point.len() - z.len()..]);
                assert_eq!(result.claims[1].value, traw[1]);
            } else {
                assert_eq!(result.claims[0].source, Source::Logits(logits));
                assert_eq!(result.claims[0].point, point);
                assert_eq!(result.claims[0].value, raw[0]);
                assert_eq!(result.claims[1].source, Source::Maximum(GadgetId(73)));
                assert_eq!(result.claims[1].point, point[point.len() - z.len()..]);
                assert_eq!(result.claims[1].value, raw[1]);
                let original = list(member(case, "original_onehot"));
                for (got, want) in result.claims[2..].iter().zip(original) {
                    assert_eq!(got.source, Source::OneHot(onehot));
                    assert_eq!(got.point, fields(member(want, "point")));
                    assert_eq!(got.value, field(member(want, "value")));
                }
            }
            for count in [0, values.len() - 1, values.len() + 1] {
                let mut bad = values.clone();
                bad.resize(count, Fp2::ZERO);
                assert!(matches!(
                    check.evaluate(&point, &bad),
                    Err(GadgetError::Rejected {
                        gadget: GadgetId(73)
                    })
                ));
            }
            let mut wrong = point.clone();
            wrong.push(Fp2::ZERO);
            assert!(matches!(
                check.evaluate(&wrong, values),
                Err(GadgetError::Internal)
            ));
        }
    }
}

#[test]
fn compensating_carry_values_keep_the_native_value_but_leave_two_false_original_claims() {
    let f = read_vectors("greedy-boundary/greedy-boundary.json");
    for case in list(member(&f, "cases")) {
        let (_, b) = boundary(case);
        let point = fields(member(case, "point"));
        let zero = fields(member(case, "zero"));
        let raw = fields(member(case, "raw"));
        let originals = list(member(case, "original_onehot"));
        let a = field(member(&originals[0], "coefficient"));
        let c = field(member(&originals[1], "coefficient"));
        assert_ne!(a, Fp2::ZERO);
        assert_ne!(c, Fp2::ZERO);
        let check = b.zero(Kind::Selected, &zero).unwrap();
        let honest = check.evaluate(&point, &raw).unwrap();
        let mut bad = raw.clone();
        bad[2] += c;
        bad[3] -= a;
        let changed = check.evaluate(&point, &bad).unwrap();
        assert_eq!(changed.value, honest.value);
        assert_eq!(changed.claims.len(), honest.claims.len());
        assert_eq!(
            changed
                .claims
                .iter()
                .zip(&honest.claims)
                .filter(|(a, b)| a.value != b.value)
                .count(),
            2
        );
        assert_eq!(changed.claims[2].point, honest.claims[2].point);
        assert_eq!(changed.claims[3].point, honest.claims[3].point);
        // A local equality cannot reject this; BOTH changed values must reach original PCS checks.
        assert_ne!(
            changed.claims[2].value,
            field(member(&originals[0], "value"))
        );
        assert_ne!(
            changed.claims[3].value,
            field(member(&originals[1], "value"))
        );
    }
}

#[test]
fn zero_carry_coefficients_never_drop_original_source_obligations() {
    let fixtures = read_vectors("greedy-boundary/greedy-boundary.json");
    for case in list(member(&fixtures, "cases")) {
        let (_, b) = boundary(case);
        let target = fields(member(case, "zero"));
        let raw = fields(member(case, "raw"));
        let check = b.zero(Kind::Selected, &target).unwrap();
        for bit in [Fp2::ZERO, Fp2::ONE] {
            let final_value = check.evaluate(&vec![bit; b.variables()], &raw).unwrap();
            assert_eq!(final_value.claims.len(), raw.len());
            assert!(
                final_value
                    .claims
                    .iter()
                    .zip(&raw)
                    .all(|(c, &v)| c.value == v)
            );
        }
    }
}

#[test]
fn own_request_and_zero_point_geometry_cannot_create_unbound_instances() {
    let (claims, graph) = model(3, 5);
    let plan = Plan::derive(&graph, claims.gadget_policy).unwrap();
    for prompt in [0, 3, 4, u32::MAX] {
        assert!(matches!(
            plan.greedy_boundary(GadgetId(73), prompt),
            Err(GadgetError::Internal)
        ));
    }
    let b = plan.greedy_boundary(GadgetId(73), 1).unwrap();
    assert!(matches!(b.key(&[], Fp2::ZERO), Err(GadgetError::Internal)));
    assert!(matches!(
        b.zero(Kind::Key, &[Fp2::ZERO; 2]),
        Err(GadgetError::Internal)
    ));
    assert!(matches!(
        b.zero(Kind::Selected, &[]),
        Err(GadgetError::Internal)
    ));
}

fn vertex(index: usize, variables: usize) -> Vec<Fp2> {
    (0..variables)
        .map(|bit| {
            if (index >> bit) & 1 == 0 {
                Fp2::ZERO
            } else {
                Fp2::ONE
            }
        })
        .collect()
}

#[test]
fn batched_keys_cannot_read_the_same_position_in_another_request() {
    let (claims, _) = model(8, 3);
    let public = validation::check_public(&claims).unwrap();
    let graph = lowering::lower_batch(&public, &[3, 2]).unwrap();
    let plan = Plan::derive(&graph, claims.gadget_policy).unwrap();
    for (ordinal, prompt, outputs) in [(0u32, 1u32, 2u32), (1, 1, 1)] {
        let b = plan
            .greedy_boundary_in_slot(GadgetId(73), ordinal, prompt, outputs)
            .unwrap();
        let output_vars = if outputs == 2 { 1 } else { 0 };
        for output in 0..outputs as usize {
            let target = vertex(output, output_vars);
            for row in 0..8 {
                let expected = if row == ordinal as usize * 4 + output {
                    Fp2::ONE
                } else {
                    Fp2::ZERO
                };
                assert_eq!(
                    b.key_time_coeff(&target, &vertex(row, 3)),
                    Some(expected),
                    "request {ordinal}, output {output}, row {row}"
                );
            }
        }
    }
}

#[test]
fn a_shared_maximum_cannot_hide_compensating_changes_in_other_rows_or_windows() {
    use tv_core::gadgets::greedy::Window;
    let (claims, _) = model(8, 3);
    let public = validation::check_public(&claims).unwrap();
    let graph = lowering::lower_batch(&public, &[3, 2]).unwrap();
    let plan = Plan::derive(&graph, claims.gadget_policy).unwrap();
    let b = plan.greedy_boundary_in_slot(GadgetId(73), 0, 1, 2).unwrap();
    let windows = [
        Window {
            ordinal: 0,
            prompt: 1,
            outputs: 2,
        },
        Window {
            ordinal: 1,
            prompt: 1,
            outputs: 1,
        },
    ];
    let zero = [
        Fp2::from(Fp::reduce(2)),
        Fp2::from(Fp::reduce(3)),
        Fp2::from(Fp::reduce(5)),
    ];
    let check = b.selected_zero(&windows, &zero).unwrap();
    let eq_at = |row: usize| {
        zero.iter().enumerate().fold(Fp2::ONE, |a, (bit, &r)| {
            a * if (row >> bit) & 1 == 1 {
                r
            } else {
                Fp2::ONE - r
            }
        })
    };
    // A unit selected logit requires M=1 on both requests' windows and M=0 elsewhere.
    // This control catches dropping request 1 from the Selected union.
    for row in 0..8 {
        let mut point = vertex(0, 2);
        point.extend(vertex(row, 3));
        let active = [0, 1, 4].contains(&row);
        let mut raw = vec![Fp2::ONE; check.instance().evals];
        raw[1] = if active { Fp2::ONE } else { Fp2::ZERO };
        assert_eq!(check.evaluate(&point, &raw).unwrap().value, Fp2::ZERO);
    }
    // Row 4 belongs to request 1's active window; row 5 lies outside both windows.
    for compensation in [4usize, 5] {
        let mut maxima = [Fp2::ZERO; 8];
        maxima[0] = Fp2::ONE;
        maxima[compensation] = -Fp2::ONE;
        assert_eq!(
            maxima.iter().copied().fold(Fp2::ZERO, |a, b| a + b),
            Fp2::ZERO
        );
        let mut actual = Fp2::ZERO;
        for (row, &maximum) in maxima.iter().enumerate() {
            let mut point = vertex(0, 2); // vocabulary 0, the selected token everywhere
            point.extend(vertex(row, 3));
            let mut raw = vec![Fp2::ONE; check.instance().evals];
            raw[0] = Fp2::ZERO; // original selected logit is zero
            raw[1] = maximum;
            let result = check.evaluate(&point, &raw).unwrap();
            let expected = -eq_at(row) * maximum;
            assert_eq!(result.value, expected);
            // Both altered entries owe the same committed maximum, at their own row.
            assert_eq!(result.claims[1].source, Source::Maximum(GadgetId(73)));
            assert_eq!(result.claims[1].point, vertex(row, 3));
            assert_eq!(result.claims[1].value, maximum);
            actual += result.value;
        }
        assert_eq!(actual, -eq_at(0) + eq_at(compensation));
        assert_ne!(actual, Fp2::ZERO, "compensation row {compensation}");
    }
}

#[test]
fn selected_sumcheck_rejects_compensated_maximum_endpoints_after_honest_zero_control() {
    use tv_core::gadgets::greedy::Window;
    use tv_core::sumcheck::{Prefix, SumcheckError, SumcheckProof, verify_batch};
    use tv_hash::transcript::{Ascii, Mode, ParamSetId, SessionContext, Transcript};

    let (claims, _) = model(8, 3);
    let public = validation::check_public(&claims).unwrap();
    let graph = lowering::lower_batch(&public, &[3, 2]).unwrap();
    let plan = Plan::derive(&graph, claims.gadget_policy).unwrap();
    let boundary = plan.greedy_boundary_in_slot(GadgetId(73), 0, 1, 2).unwrap();
    let windows = [
        Window {
            ordinal: 0,
            prompt: 1,
            outputs: 2,
        },
        Window {
            ordinal: 1,
            prompt: 1,
            outputs: 1,
        },
    ];
    let zero = [
        Fp2::from(Fp::reduce(2)),
        Fp2::from(Fp::reduce(3)),
        Fp2::from(Fp::reduce(5)),
    ];
    let check = boundary.selected_zero(&windows, &zero).unwrap();
    let instance = check.instance();
    let context = SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("test-small/1").unwrap()),
        statement_digest: [0x51; 32],
        session_id: [0x71; 16],
    };
    // Literal independent proof of the identically-zero relation: L=M=H=0.
    // All round polynomials and every original endpoint are exactly zero.
    let honest = SumcheckProof {
        rounds: vec![vec![Fp2::ZERO; 3]; instance.vars],
        final_evals: vec![Fp2::ZERO; instance.evals],
    };
    let mut transcript = Transcript::new(&context);
    let accepted = verify_batch(
        &mut transcript,
        Prefix::Stage(1),
        3,
        &[instance],
        &honest,
        |_, point, values| Some(check.evaluate(point, values).ok()?.value),
    )
    .unwrap();
    assert_eq!(accepted.point.len(), 5);
    let time = &accepted.point[2..];
    let eq_at = |row: usize| {
        time.iter().enumerate().fold(Fp2::ONE, |a, (bit, &r)| {
            a * if (row >> bit) & 1 == 1 {
                r
            } else {
                Fp2::ONE - r
            }
        })
    };
    // An attacker retains zero round messages while replacing the shared maximum by
    // +1 at request 0's active row and -1 either in request 1's window or outside it.
    for compensation in [4usize, 5] {
        let mut bad = honest.clone();
        bad.final_evals[1] = eq_at(0) - eq_at(compensation);
        assert_ne!(bad.final_evals[1], Fp2::ZERO);
        let mut debts = None;
        let mut transcript = Transcript::new(&context);
        let result = verify_batch(
            &mut transcript,
            Prefix::Stage(1),
            3,
            &[instance],
            &bad,
            |_, point, values| {
                let final_value = check.evaluate(point, values).ok()?;
                let value = final_value.value;
                debts = Some(final_value.claims);
                Some(value)
            },
        );
        assert_eq!(result, Err(SumcheckError::FinalMismatch { round: 5 }));
        let debts = debts.unwrap();
        assert_eq!(debts.len(), instance.evals);
        assert_eq!(debts[1].source, Source::Maximum(GadgetId(73)));
        assert_eq!(debts[1].point, time);
        assert_eq!(debts[1].value, bad.final_evals[1]);
    }
    // This component test does not authenticate the endpoints through PCS or the driver.
}

fn batch_component_context() -> tv_hash::transcript::SessionContext<'static> {
    use tv_hash::transcript::{Ascii, Mode, ParamSetId, SessionContext};
    SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("test-small/1").unwrap()),
        statement_digest: [0x51; 32],
        session_id: [0x71; 16],
    }
}

fn reference_field(v: tv_testkit::sumcheck::Fe) -> Fp2 {
    Fp2::new(
        Fp::from_canonical(v.0).unwrap(),
        Fp::from_canonical(v.1).unwrap(),
    )
}

#[test]
fn a_key_proof_reading_the_other_request_slot_fails_the_native_sumcheck() {
    use tv_core::sumcheck::{Prefix, SumcheckError, SumcheckProof, verify_batch};
    use tv_hash::transcript::Transcript;
    use tv_testkit::sumcheck::{self as reference, Fe, Form, Product};
    let (claims, _) = model(8, 3);
    let public = validation::check_public(&claims).unwrap();
    let graph = lowering::lower_batch(&public, &[3, 2]).unwrap();
    let plan = Plan::derive(&graph, claims.gadget_policy).unwrap();
    let b = plan.greedy_boundary_in_slot(GadgetId(73), 0, 1, 2).unwrap();
    let target = [Fp2::ZERO; 3]; // vocabulary 0, output 0 of request 0
    let check = b.key(&target, Fp2::ONE).unwrap();
    for row in [0usize, 4] {
        // M is 1 at the claimed row and 0 elsewhere. The attacker proves a key read
        // at request 1's row 4 instead of request 0's prescribed row 0.
        let maximum: Vec<_> = (0..32)
            .map(|i| if i >> 2 == row { Fe::ONE } else { Fe::ZERO })
            .collect();
        let coefficient: Vec<_> = (0..32)
            .map(|i| if i == row * 4 { Fe::ONE } else { Fe::ZERO })
            .collect();
        let product = Product {
            tables: vec![maximum],
            public: vec![coefficient],
            lambda: None,
            form: Form::Product,
        };
        let mut prover = reference::Transcript::new(&batch_component_context().to_tce());
        let generated = reference::prove(&mut prover, "stage/1", 2, &[product], None);
        let mut endpoints = vec![Fp2::ZERO; check.instance().evals];
        endpoints[1] = reference_field(generated.evals[0]);
        let proof = SumcheckProof {
            rounds: generated
                .rounds
                .iter()
                .map(|r| r.iter().copied().map(reference_field).collect())
                .collect(),
            final_evals: endpoints,
        };
        let mut debts = None;
        let mut transcript = Transcript::new(&batch_component_context());
        let result = verify_batch(
            &mut transcript,
            Prefix::Stage(1),
            2,
            &[check.instance()],
            &proof,
            |_, point, values| {
                let final_value = check.evaluate(point, values).ok()?;
                let value = final_value.value;
                debts = Some(final_value.claims);
                Some(value)
            },
        );
        if row == 0 {
            assert!(result.is_ok());
        } else {
            assert_eq!(result, Err(SumcheckError::FinalMismatch { round: 5 }));
        }
        let debts = debts.unwrap();
        assert_eq!(debts[1].source, Source::Maximum(GadgetId(73)));
        assert_eq!(
            debts[1].point,
            generated.challenges[2..]
                .iter()
                .copied()
                .map(reference_field)
                .collect::<Vec<_>>()
        );
        assert_eq!(debts[1].value, proof.final_evals[1]);
    }
}

#[test]
fn a_nonzero_root_in_an_inter_slot_gap_fails_the_padding_sumcheck() {
    use tv_core::sumcheck::{Prefix, SumcheckError, SumcheckProof, verify_batch};
    use tv_hash::transcript::Transcript;
    use tv_testkit::sumcheck::{self as reference, Fe, Form, Product};
    let (claims, _) = model(8, 3);
    let public = validation::check_public(&claims).unwrap();
    let graph = lowering::lower_batch(&public, &[3, 2]).unwrap();
    let plan = Plan::derive(&graph, claims.gadget_policy).unwrap();
    let root = plan.roots().next().unwrap().unwrap();
    let root_tensor = graph.node(root.node()).unwrap().outputs()[0];
    let zero = [Fe::base(2), Fe::base(3), Fe::base(5)];
    let eqs: Vec<_> = (0..8)
        .map(|row| {
            zero.iter().enumerate().fold(Fe::ONE, |a, (bit, &r)| {
                a * if (row >> bit) & 1 == 1 {
                    r
                } else {
                    Fe::ONE - r
                }
            })
        })
        .collect();
    let padded = |row| ![0usize, 1, 2, 4, 5].contains(&row);
    let padding: Vec<_> = (0..8)
        .map(|row| if padded(row) { Fe::ONE } else { Fe::ZERO })
        .collect();
    for malicious in [false, true] {
        let values: Vec<_> = (0..8)
            .map(|row| {
                if !padded(row) || (malicious && row == 3) {
                    Fe::ONE
                } else {
                    Fe::ZERO
                }
            })
            .collect();
        let product = Product {
            tables: vec![values],
            public: vec![eqs.clone(), padding.clone()],
            lambda: None,
            form: Form::Product,
        };
        let mut prover = reference::Transcript::new(&batch_component_context().to_tce());
        let generated = reference::prove(&mut prover, "stage/1", 3, &[product], None);
        let proof = SumcheckProof {
            rounds: generated
                .rounds
                .iter()
                .map(|r| r.iter().copied().map(reference_field).collect())
                .collect(),
            final_evals: generated
                .evals
                .iter()
                .copied()
                .map(reference_field)
                .collect(),
        };
        let zero: Vec<_> = zero.iter().copied().map(reference_field).collect();
        let mut debt = None;
        let mut transcript = Transcript::new(&batch_component_context());
        let result = verify_batch(
            &mut transcript,
            Prefix::Stage(1),
            3,
            &[root.padding_instance()],
            &proof,
            |_, point, values| {
                let checked = root.padding(&zero, point, values).ok()?;
                let claim = checked.claim();
                debt = Some((claim.input.source, claim.input.point.to_vec(), claim.value));
                Some(checked.value())
            },
        );
        if malicious {
            assert_eq!(result, Err(SumcheckError::FinalMismatch { round: 3 }));
        } else {
            assert!(result.is_ok());
        }
        let (source, point, value) = debt.unwrap();
        assert_eq!(source, tv_core::advice::Source::Tensor(root_tensor));
        assert_eq!(
            point,
            generated
                .challenges
                .iter()
                .copied()
                .map(reference_field)
                .collect::<Vec<_>>()
        );
        assert_eq!(value, proof.final_evals[0]);
    }
    // Neither component proof authenticates its original endpoints through PCS.
}
