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
