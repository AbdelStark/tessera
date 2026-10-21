// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Independent causal-key, reciprocal and padding component proofs.
mod claim_support;
mod split_support;
mod support;

use claim_support::{claims_of, integer, items, list, member, number};
use support::{Context, digest, extension};
use tv_core::advice::{Auxiliary, Source};
use tv_core::claims::{ClaimList, QuantPoint, Site};
use tv_core::gadgets::{
    Claim, GadgetError, Plan,
    softmax::{Padding, Softmax},
};
use tv_core::lowering::{self, Graph, NodeId, TensorId};
use tv_core::sumcheck::{self, Batch, Prefix, SumcheckError, SumcheckProof};
use tv_core::validation::{self, GadgetId};
use tv_field::{Fp, Fp2};
use tv_hash::transcript::{Ascii, Transcript};
use tv_tables::TableId;
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
fn model(case: &Json) -> (ClaimList, Graph) {
    let mut c = claims_of("tiny-2-servable.d");
    let h = number(member(case, "heads"));
    c.arch.d_model = 2 * h;
    c.arch.n_heads = h;
    c.arch.n_kv_heads = h;
    c.arch.d_head = 2;
    c.arch.d_ff = 6;
    c.arch.vocab = 8;
    c.arch.max_context = 8;
    let q = &mut c.quant;
    q.residual_bits = 8;
    q.residual_exponent = q
        .act_exponents
        .iter()
        .find(|(p, _)| *p == QuantPoint::Embed)
        .unwrap()
        .1;
    q.rsqrt_kappa = 0;
    q.exp_sigma = number(member(case, "sigma"));
    for (point, shifts) in &mut q.shift_class_sets {
        *shifts = vec![
            if matches!(point,QuantPoint::Layer {site,..} if *site != Site::Silu) {
                20
            } else {
                12
            },
        ];
    }
    for (_, scale) in &mut q.score_scales {
        *scale = tv_codec::Rational::new(1, 1).unwrap();
    }
    c.gadget_policy.max_log2_rows = 32;
    c.gadget_policy.limb_bits = number(member(case, "limb"));
    let public = validation::check_public(&c).unwrap();
    let graph = lowering::lower(&public, number(member(case, "n"))).unwrap();
    (c, graph)
}
fn site<'a>(plan: &'a Plan<'_>, case: &Json) -> Softmax<'a> {
    let id = GadgetId(number(member(case, "gadget")));
    let sites: Vec<_> = plan.softmaxes().map(Result::unwrap).collect();
    assert_eq!(sites.len(), 2);
    *sites.iter().find(|s| s.gadget() == id).unwrap()
}
fn proof(record: &Json) -> SumcheckProof {
    SumcheckProof {
        rounds: list(record, "rounds").iter().map(fields).collect(),
        final_evals: fields(member(record, "final_evals")),
    }
}
fn before(case: &Json, record: &Json) -> (Transcript, Vec<Fp2>) {
    let mut t = Context::from_json(member(case, "context")).transcript();
    let mut values = fields(member(case, "source_values"));
    values.extend(fields(member(case, "range_keys")));
    values.extend([
        field(member(case, "lookup_key")),
        field(member(case, "lookup_value")),
        field(member(case, "tree_leaf")),
    ]);
    let mut bytes = (values.len() as u64).to_le_bytes().to_vec();
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    t.absorb(Ascii::new("stage/0/split").unwrap(), &bytes);
    assert_eq!(
        t.challenge_fp2(Ascii::new("stage/1/layer/0").unwrap())
            .unwrap(),
        field(member(record, "layer"))
    );
    let zero: Vec<_> = (0..list(record, "zero").len())
        .map(|i| {
            t.challenge_fp2(Ascii::new(&format!("stage/1/zero/{i}")).unwrap())
                .unwrap()
        })
        .collect();
    assert_eq!(zero, fields(member(record, "zero")));
    (t, zero)
}
fn run(
    site: Softmax<'_>,
    case: &Json,
    record: &Json,
    p: &SumcheckProof,
    omit: Option<usize>,
) -> Result<(Batch, Transcript), SumcheckError> {
    let row = fields(member(case, "row_target"));
    let lookup = fields(member(case, "lookup_target"));
    let tree = fields(member(case, "tree_target"));
    let values = fields(member(case, "source_values"));
    let ranges = fields(member(case, "range_keys"));
    let bound = site
        .check(&row, &values, &ranges, &vec![Fp2::ZERO; ranges.len()])
        .unwrap();
    let keys = [
        site.lookup(
            &lookup,
            field(member(case, "lookup_key")),
            field(member(case, "lookup_value")),
        )
        .unwrap(),
        site.tree(&tree, field(member(case, "tree_leaf"))).unwrap(),
    ];
    let instances = [
        keys[0].instance(),
        keys[1].instance(),
        bound.instance(),
        site.padding_instance(),
        site.padding_instance(),
    ];
    assert_eq!(
        instances.iter().map(|i| i.claim).collect::<Vec<_>>(),
        fields(member(case, "claims"))
    );
    let (mut t, zero) = before(case, record);
    let batch = sumcheck::verify_batch(
        &mut t,
        Prefix::Stage(1),
        3,
        &instances,
        p,
        |i, point, evals| {
            let own = point;
            if omit == Some(i) {
                Some(instances[i].claim * own.iter().fold(Fp2::ONE, |a, &x| a * (Fp2::ONE - x)))
            } else if i < 2 {
                keys[i].evaluate(own, evals).ok().map(|v| v.value())
            } else if i == 2 {
                bound.evaluate(own, evals).ok().map(|v| v.value())
            } else {
                site.padding(
                    if i == 3 {
                        Padding::Maximum
                    } else {
                        Padding::Reciprocal
                    },
                    &zero,
                    own,
                    evals,
                )
                .ok()
                .map(|v| v.value())
            }
        },
    )?;
    Ok((batch, t))
}
fn truth(case: &Json, claim: Claim<'_>) -> bool {
    let expected = match claim.input.source {
        Source::Tensor(id) if id == TensorId(number(member(case, "scores"))) => {
            member(case, "score_column")
        }
        Source::Tensor(id) if id == TensorId(number(member(case, "exponentials"))) => {
            member(case, "exp_column")
        }
        Source::Tensor(id) if id == TensorId(number(member(case, "reciprocal"))) => {
            &list(case, "source_columns")[0]
        }
        Source::Auxiliary(id, Auxiliary::SoftmaxMaximum) => {
            assert_eq!(id, NodeId(number(member(case, "node"))));
            member(case, "maximum_column")
        }
        _ => panic!("unexpected final source"),
    };
    claim.value == extension(&column(expected), claim.input.point)
}
fn final_claims(
    site: Softmax<'_>,
    case: &Json,
    record: &Json,
    p: &SumcheckProof,
    point: &[Fp2],
) -> bool {
    let row = fields(member(case, "row_target"));
    let lookup = fields(member(case, "lookup_target"));
    let tree = fields(member(case, "tree_target"));
    let values = fields(member(case, "source_values"));
    let ranges = fields(member(case, "range_keys"));
    let bound = site
        .check(&row, &values, &ranges, &vec![Fp2::ZERO; ranges.len()])
        .unwrap();
    let keys = [
        site.lookup(
            &lookup,
            field(member(case, "lookup_key")),
            field(member(case, "lookup_value")),
        )
        .unwrap(),
        site.tree(&tree, field(member(case, "tree_leaf"))).unwrap(),
    ];
    let zero = fields(member(record, "zero"));
    let suffix = &point[site.variables() - site.row_variables()..];
    let mut all_true = true;
    for (i, expected) in list(record, "final_relations").iter().enumerate() {
        let (value, claims) = if i < 2 {
            let v = keys[i]
                .evaluate(point, &p.final_evals[2 * i..2 * i + 2])
                .unwrap();
            (
                v.value(),
                v.claims()
                    .map(|c| (c.input.source, c.input.point.to_vec(), c.value))
                    .collect::<Vec<_>>(),
            )
        } else if i == 2 {
            let v = bound.evaluate(suffix, &p.final_evals[4..6]).unwrap();
            (
                v.value(),
                v.claims()
                    .map(|c| (c.input.source, c.input.point.to_vec(), c.value))
                    .collect(),
            )
        } else {
            let v = site
                .padding(
                    if i == 3 {
                        Padding::Maximum
                    } else {
                        Padding::Reciprocal
                    },
                    &zero,
                    suffix,
                    &p.final_evals[i + 3..i + 4],
                )
                .unwrap();
            let c = v.claim();
            (
                v.value(),
                vec![(c.input.source, c.input.point.to_vec(), c.value)],
            )
        };
        let start = if i < 2 {
            2 * i
        } else if i == 2 {
            4
        } else {
            i + 3
        };
        let end = if i < 3 { start + 2 } else { start + 1 };
        if p.final_evals[start..end] == fields(member(record, "final_evals"))[start..end] {
            assert_eq!(value, field(expected));
        }
        for (source, point, value) in claims {
            all_true &= truth(
                case,
                Claim {
                    input: tv_core::gadgets::Input {
                        source,
                        point: &point,
                    },
                    value,
                },
            );
        }
    }
    all_true
}
#[test]
fn all_independent_softmax_proofs_and_single_check_omission_controls() {
    let fixture = read_vectors("gadget-finals/softmax.json");
    let mut counts = [0; 3];
    for case in list(&fixture, "cases") {
        let (c, g) = model(case);
        let plan = Plan::derive(&g, c.gadget_policy).unwrap();
        let site = site(&plan, case);
        assert_eq!(site.node(), NodeId(number(member(case, "node"))));
        let record = member(case, "proof");
        let p = proof(record);
        let point = fields(member(record, "point"));
        assert!(final_claims(site, case, record, &p, &point));
        let good = member(case, "failed") == &Json::Null;
        let result = run(site, case, record, &p, None);
        if good {
            let (batch, t) = result.unwrap();
            assert_eq!(batch.point, point);
            assert_eq!(batch.gamma, field(member(record, "gamma")));
            assert_eq!(
                *t.state(),
                digest(member(record, "state").as_str().unwrap())
            );
            assert_eq!(t.counter(), number(member(record, "counter")));
            counts[0] += 1;
        } else {
            assert_eq!(
                result.unwrap_err(),
                SumcheckError::FinalMismatch {
                    round: u32::try_from(site.variables()).unwrap()
                }
            );
            counts[1] += 1;
            let adapted = member(case, "omission");
            let p = proof(adapted);
            let point = fields(member(adapted, "point"));
            assert!(final_claims(site, case, adapted, &p, &point));
            let (control, _) = run(
                site,
                case,
                adapted,
                &p,
                Some(number(member(case, "failed"))),
            )
            .unwrap();
            assert_eq!(control.point, point);
            assert_eq!(
                run(site, case, adapted, &p, None).unwrap_err(),
                SumcheckError::FinalMismatch {
                    round: u32::try_from(site.variables()).unwrap()
                }
            );
            counts[2] += 1;
        }
        for i in 0..p.final_evals.len() {
            let mut changed = p.clone();
            changed.final_evals[i] += Fp2::ONE;
            match run(site, case, record, &changed, None) {
                Err(error) => assert_eq!(
                    error,
                    SumcheckError::FinalMismatch {
                        round: u32::try_from(site.variables()).unwrap()
                    }
                ),
                Ok((batch, _)) => {
                    assert!(!final_claims(site, case, record, &changed, &batch.point))
                }
            }
        }
    }
    assert_eq!(counts, [8, 6, 6]);
}
#[test]
fn every_softmax_range_source_point_and_lookup_is_canonical() {
    let fixture = read_vectors("gadget-finals/softmax.json");
    for case in list(&fixture, "cases") {
        let (c, g) = model(case);
        let plan = Plan::derive(&g, c.gadget_policy).unwrap();
        let site = site(&plan, case);
        let point = fields(member(case, "row_target"));
        let values = fields(member(case, "source_values"));
        let keys = fields(member(case, "range_keys"));
        let zeros = vec![Fp2::ZERO; keys.len()];
        let checked = site.check(&point, &values, &keys, &zeros).unwrap();
        let expected_tables: Vec<_> = list(case, "tables")[1..]
            .iter()
            .map(|v| TableId(number(v)))
            .collect();
        assert_eq!(site.range_tables(), expected_tables);
        let claims: Vec<_> = checked.claims().collect();
        assert_eq!(claims.len(), values.len());
        assert!(site.source(values.len()).is_none());
        for (i, claim) in claims.iter().enumerate() {
            let low = list(case, "widths").len() - 1;
            let source = if i == 0 {
                Source::Tensor(TensorId(number(member(case, "reciprocal"))))
            } else if i == 1 {
                Source::Tensor(TensorId(number(member(case, "exponentials"))))
            } else if i < 2 + low {
                Source::Auxiliary(site.node(), Auxiliary::SoftmaxA((i - 2) as u32))
            } else {
                Source::Auxiliary(site.node(), Auxiliary::SoftmaxB((i - 2 - low) as u32))
            };
            assert_eq!(claim.input.source, source);
            assert_eq!(site.source(i), Some(source));
            assert_eq!(claim.value, values[i]);
            if i == 1 {
                assert_eq!(claim.input.point, site.row_sum_point(&point).unwrap());
                assert!(truth(case, *claim));
            } else {
                assert_eq!(claim.input.point, point);
                assert_eq!(
                    claim.value,
                    extension(&column(&list(case, "source_columns")[i]), &point)
                );
            }
        }
        assert_eq!(checked.lookups().count(), keys.len());
        for (i, lookup) in checked.lookups().enumerate() {
            assert_eq!(lookup.table, expected_tables[i]);
            assert_eq!(lookup.point, point);
            assert_eq!(lookup.key, keys[i]);
            assert_eq!(lookup.value, Fp2::ZERO);
        }
        let p = fields(member(case, "lookup_target"));
        let key = site
            .lookup(
                &p,
                field(member(case, "lookup_key")),
                field(member(case, "lookup_value")),
            )
            .unwrap();
        let lookup = key.lookup().unwrap();
        assert_eq!(lookup.table, TableId(number(&list(case, "tables")[0])));
        assert_eq!(lookup.point, p);
        assert_eq!(lookup.key, field(member(case, "lookup_key")));
        assert_eq!(lookup.value, field(member(case, "lookup_value")));
        assert!(truth(case, key.input_claim().unwrap()));
        let tree = site.tree(&p, Fp2::ZERO).unwrap();
        assert!(tree.lookup().is_none());
        assert!(tree.input_claim().is_none());
        let roots = column(member(case, "psi"));
        assert_eq!(roots.len(), 1usize << site.row_variables());
        for (i, &root) in roots.iter().enumerate() {
            let p: Vec<_> = (0..site.row_variables())
                .map(|j| if i >> j & 1 == 0 { Fp2::ZERO } else { Fp2::ONE })
                .collect();
            assert_eq!(site.forest_root(&p).unwrap(), Fp2::ONE - Fp2::from(root));
        }
    }
}
#[test]
fn every_softmax_affine_field_and_proof_length_is_checked_with_own_error_precedence() {
    let fixture = read_vectors("gadget-finals/softmax.json");
    for case in list(&fixture, "cases") {
        let (c, g) = model(case);
        let plan = Plan::derive(&g, c.gadget_policy).unwrap();
        let site = site(&plan, case);
        let point = fields(member(case, "row_target"));
        let sources = fields(member(case, "source_values"));
        let keys = fields(member(case, "range_keys"));
        let zeros = vec![Fp2::ZERO; keys.len()];
        let reject = GadgetError::Rejected {
            gadget: site.gadget(),
        };
        for i in 0..sources.len() {
            let mut v = sources.clone();
            v[i] += Fp2::ONE;
            assert!(matches!(site.check(&point,&v,&keys,&zeros),Err(e) if e==reject));
        }
        for i in 0..keys.len() {
            let mut v = keys.clone();
            v[i] += Fp2::ONE;
            assert!(matches!(site.check(&point,&sources,&v,&zeros),Err(e) if e==reject));
            let mut v = zeros.clone();
            v[i] = Fp2::ONE;
            assert!(matches!(site.check(&point,&sources,&keys,&v),Err(e) if e==reject));
        }
        for extra in [false, true] {
            let resize = |v: &[Fp2]| {
                let mut v = v.to_vec();
                if extra {
                    v.push(Fp2::ZERO);
                } else {
                    v.pop();
                }
                v
            };
            assert!(
                matches!(site.check(&point,&resize(&sources),&keys,&zeros),Err(e) if e==reject)
            );
            assert!(
                matches!(site.check(&point,&sources,&resize(&keys),&zeros),Err(e) if e==reject)
            );
            assert!(
                matches!(site.check(&point,&sources,&keys,&resize(&zeros)),Err(e) if e==reject)
            );
            assert!(
                matches!(site.check(&point,&sources,&resize(&keys),&resize(&zeros)),Err(e) if e==reject)
            );
        }
        for end in 0..keys.len() {
            assert!(
                matches!(site.check(&point, &sources, &keys[..end], &zeros[..end]), Err(e) if e == reject)
            );
        }
        let mut wrong = point.clone();
        wrong.push(Fp2::ZERO);
        assert!(matches!(
            site.check(&wrong, &[], &[], &[]),
            Err(GadgetError::Internal)
        ));
        assert_eq!(site.forest_root(&wrong), Err(GadgetError::Internal));
        assert_eq!(site.row_sum_point(&wrong), Err(GadgetError::Internal));
        let target = fields(member(case, "lookup_target"));
        let mut bad = target.clone();
        bad.push(Fp2::ZERO);
        assert!(matches!(
            site.lookup(&bad, Fp2::ZERO, Fp2::ZERO),
            Err(GadgetError::Internal)
        ));
        assert!(matches!(
            site.tree(&bad, Fp2::ZERO),
            Err(GadgetError::Internal)
        ));
        let key = site.lookup(&target, Fp2::ZERO, Fp2::ZERO).unwrap();
        assert!(matches!(
            key.evaluate(&bad, &[]),
            Err(GadgetError::Internal)
        ));
        for v in [&[][..], &[Fp2::ZERO][..], &[Fp2::ZERO; 3][..]] {
            assert!(matches!(key.evaluate(&target,v),Err(e) if e==reject));
        }
        let bound = site.check(&point, &sources, &keys, &zeros).unwrap();
        assert!(matches!(
            bound.evaluate(&wrong, &[]),
            Err(GadgetError::Internal)
        ));
        for v in [&[][..], &[Fp2::ZERO][..], &[Fp2::ZERO; 3][..]] {
            assert!(matches!(bound.evaluate(&point,v),Err(e) if e==reject));
        }
        for kind in [Padding::Maximum, Padding::Reciprocal] {
            assert!(matches!(
                site.padding(kind, &wrong, &point, &[]),
                Err(GadgetError::Internal)
            ));
            assert!(matches!(
                site.padding(kind, &point, &wrong, &[]),
                Err(GadgetError::Internal)
            ));
            for v in [&[][..], &[Fp2::ZERO; 2][..]] {
                assert!(matches!(site.padding(kind,&point,&point,v),Err(e) if e==reject));
            }
        }
    }
}

#[test]
fn canonical_split_forms_match_frozen_softmax_keys_with_both_virtual_slack_tops() {
    use tv_core::leaf::split::Column;
    for case in list(&read_vectors("gadget-finals/softmax.json"), "cases") {
        let (claims, graph) = model(case);
        let sites = tv_core::leaf::split::model_sites(&graph, claims.gadget_policy).unwrap();
        let id = NodeId(number(member(case, "node")));
        let layer = sites
            .iter()
            .flat_map(|s| &s.layers)
            .find(|l| l.node == id)
            .unwrap();
        let sources = fields(member(case, "source_values"));
        let ranges = fields(member(case, "range_keys"));
        let parts = (ranges.len() - 1) / 2;
        let limb: u32 = number(member(case, "limb"));
        let mut values = vec![
            (Column::SoftmaxKey, field(member(case, "lookup_key"))),
            (Column::Output(0), field(member(case, "lookup_value"))),
            (
                split_support::local(
                    &graph,
                    id,
                    Source::Tensor(TensorId(number(member(case, "reciprocal")))),
                ),
                sources[0],
            ),
        ];
        for (i, column) in [Column::SoftmaxA, Column::SoftmaxB].into_iter().enumerate() {
            let slack = ranges[1 + i * parts..1 + (i + 1) * parts]
                .iter()
                .enumerate()
                .fold(Fp2::ZERO, |sum, (j, &v)| {
                    sum + Fp2::from(Fp::reduce(1 << (limb * j as u32))) * v
                });
            values.push((column, slack));
            for j in 0..parts - 1 {
                values.push((
                    Column::Auxiliary(if i == 0 {
                        Auxiliary::SoftmaxA(j as u32)
                    } else {
                        Auxiliary::SoftmaxB(j as u32)
                    }),
                    sources[2 + i * (parts - 1) + j],
                ));
            }
        }
        let mut order = vec![Column::SoftmaxKey, Column::Output(0), Column::Output(1)];
        for (side, column) in [Column::SoftmaxA, Column::SoftmaxB].into_iter().enumerate() {
            order.extend((0..parts - 1).map(|j| {
                Column::Auxiliary(if side == 0 {
                    Auxiliary::SoftmaxA(j as u32)
                } else {
                    Auxiliary::SoftmaxB(j as u32)
                })
            }));
            order.push(column);
        }
        assert_eq!(split_support::reads(layer), order);
        assert_eq!(
            layer.lookups.iter().map(|l| l.table).collect::<Vec<_>>(),
            list(case, "tables")
                .iter()
                .map(|v| TableId(number(v)))
                .collect::<Vec<_>>()
        );
        let mut keys = vec![field(member(case, "lookup_key"))];
        keys.extend(ranges);
        let mut outputs = vec![Fp2::ZERO; keys.len()];
        outputs[0] = field(member(case, "lookup_value"));
        split_support::check(layer, &values, &keys, &outputs, |column| {
            if matches!(column, Column::SoftmaxKey | Column::Output(0)) {
                list(case, "lookup_target").len()
            } else {
                list(case, "row_target").len()
            }
        });
    }
}
