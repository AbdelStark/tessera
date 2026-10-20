// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Frozen model-derived source ids and independently computed integer lookup rows.
mod claim_support;

use claim_support::{claims_of, integer, items, list, member, number, string, text};
use tv_core::advice::{Auxiliary, Source};
use tv_core::claims::{ClaimList, QuantPoint, Site};
use tv_core::gadgets::{GadgetError, Plan};
use tv_core::lowering::{self, Graph, NodeId, TensorId};
use tv_core::selection::Pattern;
use tv_core::validation::{self, GadgetId};
use tv_field::{Fp, Fp2};
use tv_tables::TableId;
use tv_testkit::{Json, read_vectors};

fn field(value: &Json) -> Fp2 {
    let pair = items(value);
    Fp2::new(
        Fp::from_canonical(number(&pair[0])).unwrap(),
        Fp::from_canonical(number(&pair[1])).unwrap(),
    )
}
fn fields(value: &Json) -> Vec<Fp2> {
    items(value).iter().map(field).collect()
}
fn lift(value: &Json) -> Fp2 {
    Fp2::from(
        Fp::from_canonical(integer(value).rem_euclid(i128::from(Fp::MODULUS)) as u64).unwrap(),
    )
}
fn pairs(case: &Json) -> Vec<(Fp2, Fp2)> {
    fields(member(case, "key_evals"))
        .into_iter()
        .zip(fields(member(case, "value_evals")))
        .collect()
}
fn model(case: &Json) -> (ClaimList, Graph) {
    let mut claims = claims_of("tiny-2-servable.d");
    let arch = &mut claims.arch;
    arch.d_model = 2;
    arch.n_heads = 1;
    arch.n_kv_heads = 1;
    arch.d_head = 2;
    arch.d_ff = 6;
    arch.vocab = 8;
    arch.max_context = 8;
    let quant = &mut claims.quant;
    quant.residual_bits = 8;
    quant.residual_exponent = quant
        .act_exponents
        .iter()
        .find(|(p, _)| *p == QuantPoint::Embed)
        .unwrap()
        .1;
    quant.rsqrt_kappa = 0;
    quant.exp_sigma = 3;
    let shifts: Vec<u8> = list(case, "shifts").iter().map(number).collect();
    for (point, values) in &mut quant.shift_class_sets {
        *values = if matches!(point, QuantPoint::Layer { site, .. }
            if matches!(site, Site::RopeQ | Site::RopeK | Site::V | Site::AttnResidual
                | Site::Silu | Site::Up | Site::MlpResidual))
        {
            shifts.iter().map(|&s| s as i8).collect()
        } else {
            vec![12]
        };
    }
    for (_, scale) in &mut quant.score_scales {
        *scale = tv_codec::Rational::new(1, 1).unwrap();
    }
    claims.gadget_policy.max_log2_rows = number(member(case, "maximum"));
    claims.gadget_policy.limb_bits = 8;
    let public = validation::check_public(&claims).unwrap();
    let graph = lowering::lower(&public, number(member(case, "n"))).unwrap();
    (claims, graph)
}
fn source(value: &Json) -> Source {
    let value = items(value);
    match string(&value[0]) {
        "tensor" => Source::Tensor(TensorId(number(&value[1]))),
        "auxiliary" => Source::Auxiliary(
            NodeId(number(&value[1])),
            match string(&value[2]) {
                "requant.low" => Auxiliary::RequantLow,
                "requant.middle" => Auxiliary::RequantMiddle,
                "requant.saturation" => Auxiliary::RequantSaturation,
                other => panic!("unknown auxiliary {other}"),
            },
        ),
        other => panic!("unknown source {other}"),
    }
}
fn pattern(value: &Json) -> Pattern {
    let value = items(value);
    match string(&value[0]) {
        "Requant1" => Pattern::Requant1,
        "RequantSplit" => Pattern::RequantSplit {
            split: number(&value[1]),
        },
        "RequantWide" => Pattern::RequantWide {
            split: number(&value[1]),
            middle: number(&value[2]),
        },
        "Lookup" => Pattern::Lookup,
        other => panic!("unknown pattern {other}"),
    }
}

#[test]
fn all_model_sources_points_and_lookup_requests_match_the_frozen_integer_oracle() {
    let fixture = read_vectors("gadget-finals/requant.json");
    let (mut sites, mut lookups) = (0, 0);
    for case in list(&fixture, "cases") {
        let (claims, graph) = model(case);
        let plan = Plan::derive(&graph, claims.gadget_policy).unwrap();
        let found: Vec<_> = plan.requantizations().map(Result::unwrap).collect();
        assert_eq!(
            found.len(),
            list(case, "sites").len(),
            "{}",
            text(case, "name")
        );
        for (site, expected) in found.into_iter().zip(list(case, "sites")) {
            assert_eq!(site.node(), NodeId(number(member(expected, "node"))));
            assert_eq!(site.gadget(), GadgetId(number(member(expected, "gadget"))));
            assert_eq!(site.pattern(), pattern(member(expected, "pattern")));
            assert_eq!(
                site.tables(),
                list(expected, "tables")
                    .iter()
                    .map(|v| TableId(number(v)))
                    .collect::<Vec<_>>()
            );
            let point = fields(member(expected, "point"));
            let values = fields(member(expected, "evaluations"));
            let tuples = pairs(expected);
            let checked = site.check(&point, &values, &tuples).unwrap();
            let sources: Vec<_> = checked.claims().collect();
            assert_eq!(sources.len(), values.len());
            for (i, (claim, expected_source)) in
                sources.iter().zip(list(expected, "sources")).enumerate()
            {
                let id = source(expected_source);
                assert_eq!(claim.input.source, id);
                assert_eq!(claim.value, values[i]);
                let dims = match id {
                    Source::Tensor(t) => graph.tensor(t).unwrap().shape.dims().to_vec(),
                    Source::Auxiliary(_, _) => list(expected, "shape").iter().map(number).collect(),
                    Source::Multiplicity(_) => panic!("not a source"),
                };
                let vars: usize = dims
                    .iter()
                    .map(|&d| (u32::BITS - (d - 1).leading_zeros()) as usize)
                    .sum();
                assert_eq!(claim.input.point, &point[..vars]);
            }
            let before: Vec<_> = site.inputs(&point).unwrap().collect();
            for (input, claim) in before.iter().zip(&sources) {
                assert_eq!(input.source, claim.input.source);
                assert_eq!(input.point, claim.input.point);
            }
            let lookup: Vec<_> = checked.lookups().collect();
            assert_eq!(lookup.len(), site.tables().len());
            for (i, claim) in lookup.iter().enumerate() {
                assert_eq!(claim.table, site.tables()[i]);
                assert_eq!(claim.point, point);
                assert_eq!((claim.key, claim.value), tuples[i]);
            }
            // Check every Boolean row as well, including zero time/channel padding.
            for row in 0..1 << point.len() {
                let boolean: Vec<_> = (0..point.len())
                    .map(|i| {
                        if row >> i & 1 == 0 {
                            Fp2::ZERO
                        } else {
                            Fp2::ONE
                        }
                    })
                    .collect();
                let values: Vec<_> = list(expected, "columns")
                    .iter()
                    .map(|c| lift(&items(c)[row]))
                    .collect();
                let tuples: Vec<_> = list(expected, "keys")
                    .iter()
                    .zip(list(expected, "values"))
                    .map(|(k, v)| (lift(&items(k)[row]), lift(&items(v)[row])))
                    .collect();
                let checked = site.check(&boolean, &values, &tuples).unwrap();
                assert_eq!(checked.claims().count(), values.len());
                assert_eq!(checked.lookups().count(), tuples.len());
            }
            sites += 1;
            lookups += tuples.len();
        }
    }
    assert_eq!((sites, lookups), (119, 183));
}

#[test]
fn every_changed_source_key_and_value_rejects_with_the_canonical_gadget_id() {
    let fixture = read_vectors("gadget-finals/requant.json");
    for case in list(&fixture, "cases") {
        let (claims, graph) = model(case);
        let plan = Plan::derive(&graph, claims.gadget_policy).unwrap();
        for (site, expected) in plan
            .requantizations()
            .map(Result::unwrap)
            .zip(list(case, "sites"))
        {
            let point = fields(member(expected, "point"));
            let values = fields(member(expected, "evaluations"));
            let tuples = pairs(expected);
            let reject = Some(GadgetError::Rejected {
                gadget: site.gadget(),
            });
            for i in 0..values.len() {
                let mut changed = values.clone();
                changed[i] += Fp2::ONE;
                assert_eq!(site.check(&point, &changed, &tuples).err(), reject);
            }
            for i in 0..tuples.len() {
                for key in [true, false] {
                    let mut changed = tuples.clone();
                    if key {
                        changed[i].0 += Fp2::ONE;
                    } else {
                        changed[i].1 += Fp2::ONE;
                    }
                    assert_eq!(site.check(&point, &values, &changed).err(), reject);
                }
            }
            let mut extra = values.clone();
            extra.push(Fp2::ZERO);
            assert_eq!(site.check(&point, &extra, &tuples).err(), reject);
            assert_eq!(
                site.check(&point, &values[..values.len() - 1], &tuples)
                    .err(),
                reject
            );
            let mut extra = tuples.clone();
            extra.push((Fp2::ZERO, Fp2::ZERO));
            assert_eq!(site.check(&point, &values, &extra).err(), reject);
            assert_eq!(
                site.check(&point, &values, &tuples[..tuples.len() - 1])
                    .err(),
                reject
            );
            assert_eq!(
                site.check(&point[..point.len() - 1], &values, &tuples)
                    .err(),
                Some(GadgetError::Internal)
            );
            let mut longer = point.clone();
            longer.push(Fp2::ONE);
            assert_eq!(site.inputs(&longer).err(), Some(GadgetError::Internal));
            assert_eq!(
                site.check(&longer, &values, &tuples).err(),
                Some(GadgetError::Internal)
            );
        }
    }
}
