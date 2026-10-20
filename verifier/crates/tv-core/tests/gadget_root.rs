// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Model-derived root bindings against frozen integer witnesses and component proofs.
mod claim_support;
mod support;

use claim_support::{claims_of, items, list, member, number, string};
use support::{Context, digest, extension};
use tv_core::advice::{Auxiliary, Comparison, ComparisonRole, Source};
use tv_core::claims::{ClaimList, QuantPoint};
use tv_core::gadgets::{GadgetError, Plan, root::Root};
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
        .map(|v| Fp::from_canonical(number(v)).unwrap())
        .collect()
}
fn model(case: &Json) -> (ClaimList, Graph) {
    let mut c = claims_of("tiny-2-servable.d");
    let a = &mut c.arch;
    a.d_model = number(member(case, "dim"));
    a.n_heads = 1;
    a.n_kv_heads = 1;
    a.d_head = a.d_model;
    a.d_ff = 6;
    a.vocab = 8;
    a.max_context = 8;
    let q = &mut c.quant;
    q.residual_bits = 8;
    q.residual_exponent = q
        .act_exponents
        .iter()
        .find(|(p, _)| *p == QuantPoint::Embed)
        .unwrap()
        .1;
    q.rsqrt_kappa = number(member(case, "kappa"));
    q.rms_eps_int = number(member(case, "epsilon"));
    q.exp_sigma = 3;
    for (_, s) in &mut q.shift_class_sets {
        *s = vec![12];
    }
    for (_, s) in &mut q.score_scales {
        *s = tv_codec::Rational::new(1, 1).unwrap();
    }
    let public = validation::check_public(&c).unwrap();
    let graph = lowering::lower(&public, number(member(case, "n"))).unwrap();
    (c, graph)
}
fn source(v: &Json) -> Source {
    let v = items(v);
    if string(&v[0]) == "tensor" {
        return Source::Tensor(TensorId(number(&v[1])));
    }
    let role = string(&v[2]);
    let j = number(&v[3]);
    let auxiliary = if role == "root.input" {
        Auxiliary::RootInput(j)
    } else if role == "root.low" {
        Auxiliary::RootLow(j)
    } else {
        let (side, role) = if let Some(r) = role.strip_prefix("root.lower.") {
            (Comparison::Lower, r)
        } else {
            (Comparison::Upper, role.strip_prefix("root.upper.").unwrap())
        };
        let role = match role {
            "left.digit" => ComparisonRole::LeftDigit,
            "left.carry" => ComparisonRole::LeftCarry,
            "right.digit" => ComparisonRole::RightDigit,
            "right.carry" => ComparisonRole::RightCarry,
            "difference" => ComparisonRole::Difference,
            "borrow" => ComparisonRole::Borrow,
            _ => panic!("role"),
        };
        Auxiliary::RootComparison(side, role, j)
    };
    Source::Auxiliary(NodeId(number(&v[1])), auxiliary)
}
fn proof(record: &Json) -> SumcheckProof {
    SumcheckProof {
        rounds: list(record, "rounds").iter().map(fields).collect(),
        final_evals: fields(member(record, "final_evals")),
    }
}
fn before(root: &Json, record: &Json) -> (Transcript, Vec<Fp2>) {
    let mut t = Context::from_json(member(record, "context")).transcript();
    let values = fields(member(root, "values"));
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
    let expected = fields(member(record, "zero"));
    let zero: Vec<_> = (0..expected.len())
        .map(|i| {
            t.challenge_fp2(Ascii::new(&format!("stage/1/zero/{i}")).unwrap())
                .unwrap()
        })
        .collect();
    assert_eq!(zero, expected);
    (t, zero)
}
fn run(
    root: &Root<'_>,
    fixture: &Json,
    record: &Json,
    proof: &SumcheckProof,
    omit_padding: bool,
) -> Result<(Batch, Transcript), SumcheckError> {
    let point = fields(member(fixture, "point"));
    let values = fields(member(fixture, "values"));
    let keys = fields(member(fixture, "keys"));
    let zeros = vec![Fp2::ZERO; keys.len()];
    let checked = root.check(&point, &values, &keys, &zeros).unwrap();
    let products: Vec<_> = checked.products().collect();
    let mut instances: Vec<_> = products.iter().map(|p| p.instance()).collect();
    instances.push(root.padding_instance());
    let (mut t, zero) = before(fixture, record);
    let result =
        sumcheck::verify_batch(&mut t, Prefix::Stage(1), 4, &instances, proof, |i, p, v| {
            if let Some(product) = products.get(i) {
                product.evaluate(p, v).ok().map(|v| v.value())
            } else if omit_padding {
                Some(Fp2::ZERO)
            } else {
                root.padding(&zero, p, v).ok().map(|v| v.value())
            }
        })?;
    Ok((result, t))
}
fn true_final_claims(
    root: &Root<'_>,
    fixture: &Json,
    record: &Json,
    proof: &SumcheckProof,
    point: &[Fp2],
) -> bool {
    let target = fields(member(fixture, "point"));
    let values = fields(member(fixture, "values"));
    let keys = fields(member(fixture, "keys"));
    let zeros = vec![Fp2::ZERO; keys.len()];
    let checked = root.check(&target, &values, &keys, &zeros).unwrap();
    let mut offset = 0;
    let mut truth = true;
    for (i, product) in checked.products().enumerate() {
        let count = product.instance().evals;
        let v = &proof.final_evals[offset..offset + count];
        offset += count;
        let final_value = product.evaluate(point, v).unwrap();
        if v == &fields(member(record, "final_evals"))[offset - count..offset] {
            assert_eq!(
                final_value.value(),
                field(&list(record, "final_relations")[i])
            );
        }
        let claims: Vec<_> = final_value.claims().collect();
        assert_eq!(
            claims.len(),
            5 + (number::<usize>(member(fixture, "width")) - 1) / 8
        );
        for (j, claim) in claims.iter().enumerate() {
            assert_eq!(claim.input.source, source(&list(fixture, "sources")[j]));
            assert_eq!(claim.input.point, point);
            truth &= claim.value == extension(&column(&list(fixture, "columns")[j]), point);
        }
    }
    let zero = fields(member(record, "zero"));
    let pad = root
        .padding(&zero, point, &proof.final_evals[offset..])
        .unwrap();
    let claim = pad.claim();
    assert_eq!(claim.input.source, source(&list(fixture, "sources")[1]));
    assert_eq!(claim.input.point, point);
    truth &= claim.value == extension(&column(&list(fixture, "columns")[1]), point);
    truth
}

#[test]
fn every_root_leaf_binds_complete_sources_ranges_and_derived_product_targets() {
    let fixture = read_vectors("gadget-finals/roots.json");
    let (mut sites, mut lookups) = (0, 0);
    for case in list(&fixture, "cases") {
        let (claims, graph) = model(case);
        let plan = Plan::derive(&graph, claims.gadget_policy).unwrap();
        let roots: Vec<_> = plan.roots().map(Result::unwrap).collect();
        assert_eq!(roots.len(), list(case, "roots").len());
        for (root, expected) in roots.iter().zip(list(case, "roots")) {
            assert_eq!(root.node(), NodeId(number(member(expected, "node"))));
            assert_eq!(root.gadget(), GadgetId(number(member(expected, "gadget"))));
            assert_eq!(
                root.sources(),
                list(expected, "sources")
                    .iter()
                    .map(source)
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                root.tables(),
                list(expected, "tables")
                    .iter()
                    .map(|v| TableId(number(v)))
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                root.range_bits(),
                list(expected, "range_bits")
                    .iter()
                    .map(number)
                    .collect::<Vec<u8>>()
            );
            let point = fields(member(expected, "point"));
            let values = fields(member(expected, "values"));
            let keys = fields(member(expected, "keys"));
            let zeros = vec![Fp2::ZERO; keys.len()];
            let checked = root.check(&point, &values, &keys, &zeros).unwrap();
            for ((claim, expected_source), &value) in
                checked.claims().zip(root.sources()).zip(&values)
            {
                assert_eq!(claim.input.source, *expected_source);
                assert_eq!(claim.input.point, point);
                assert_eq!(claim.value, value);
            }
            assert_eq!(checked.claims().count(), values.len());
            assert_eq!(checked.lookups().count(), keys.len());
            for (i, claim) in checked.lookups().enumerate() {
                assert_eq!(claim.table, root.tables()[i]);
                assert_eq!(claim.point, point);
                assert_eq!(claim.key, keys[i]);
                assert_eq!(claim.value, Fp2::ZERO);
            }
            let targets = fields(member(expected, "product_targets"));
            let products: Vec<_> = checked.products().collect();
            assert_eq!(products.len(), targets.len());
            let m = targets.len() / 2;
            for (i, product) in products.iter().enumerate() {
                assert_eq!(product.instance().claim, targets[i]);
                assert_eq!(product.coefficient(), i % m);
                assert_eq!(product.increment(), i >= m);
                assert_eq!(product.instance().vars, point.len());
            }
            // The four Boolean rows include the padded-root alternative; only its separate padding obligation can reject it.
            for row in 0..4 {
                let point: Vec<_> = (0..2)
                    .map(|bit| {
                        if row >> bit & 1 == 0 {
                            Fp2::ZERO
                        } else {
                            Fp2::ONE
                        }
                    })
                    .collect();
                let values: Vec<_> = list(expected, "columns")
                    .iter()
                    .map(|v| Fp2::from(Fp::from_canonical(number(&items(v)[row])).unwrap()))
                    .collect();
                let keys: Vec<_> = list(expected, "key_columns")
                    .iter()
                    .map(|v| Fp2::from(Fp::from_canonical(number(&items(v)[row])).unwrap()))
                    .collect();
                let checked = root.check(&point, &values, &keys, &zeros).unwrap();
                assert_eq!(checked.claims().count(), values.len());
            }
            sites += 1;
            lookups += keys.len();
        }
    }
    assert_eq!((sites, lookups), (35, 4745));
}

#[test]
fn root_product_and_padding_batches_verify_and_retain_every_final_source() {
    let fixture = read_vectors("gadget-finals/roots.json");
    let (mut honest, mut bad) = (0, 0);
    for case in list(&fixture, "cases") {
        let (claims, graph) = model(case);
        let plan = Plan::derive(&graph, claims.gadget_policy).unwrap();
        let root = plan.roots().next().unwrap().unwrap();
        let expected = &list(case, "roots")[0];
        let record = member(case, "proof");
        let p = proof(record);
        if matches!(member(case, "bad_padding"), Json::Bool(true)) {
            for record in [record, member(case, "omission")] {
                assert_eq!(
                    run(&root, expected, record, &proof(record), false).err(),
                    Some(SumcheckError::FinalMismatch { round: 2 })
                );
            }
            let record = member(case, "omission");
            let p = proof(record);
            let (batch, _) = run(&root, expected, record, &p, true).unwrap();
            assert!(true_final_claims(&root, expected, record, &p, &batch.point));
            bad += 1;
        } else {
            let (batch, t) = run(&root, expected, record, &p, false).unwrap();
            assert_eq!(batch.point, fields(member(record, "point")));
            assert_eq!(batch.gamma, field(member(record, "gamma")));
            assert_eq!(
                *t.state(),
                digest(member(record, "state").as_str().unwrap())
            );
            assert_eq!(t.counter(), number(member(record, "counter")));
            assert!(true_final_claims(&root, expected, record, &p, &batch.point));
            for j in 0..p.final_evals.len() {
                let mut changed = p.clone();
                changed.final_evals[j] += Fp2::ONE;
                match run(&root, expected, record, &changed, false) {
                    Err(e) => assert_eq!(e, SumcheckError::FinalMismatch { round: 2 }),
                    Ok((batch, _)) => assert!(!true_final_claims(
                        &root,
                        expected,
                        record,
                        &changed,
                        &batch.point
                    )),
                }
            }
            honest += 1;
        }
    }
    assert_eq!((honest, bad), (6, 1));
}

#[test]
fn a_consistent_wrong_root_passes_all_ranges_but_fails_the_model_bracket_with_its_gadget_id() {
    let fixture = read_vectors("gadget-finals/roots.json");
    let case = list(&fixture, "cases")
        .iter()
        .find(|c| c.get("wrong_root").is_some())
        .unwrap();
    let (claims, graph) = model(case);
    let plan = Plan::derive(&graph, claims.gadget_policy).unwrap();
    let root = plan.roots().next().unwrap().unwrap();
    let expected = member(case, "wrong_root");
    let point = fields(member(expected, "point"));
    let sources = fields(member(expected, "values"));
    let keys = fields(member(expected, "keys"));
    let zeros = vec![Fp2::ZERO; keys.len()];
    for ((col, &bits), &id) in list(expected, "key_columns")
        .iter()
        .zip(root.range_bits())
        .zip(root.tables())
    {
        let table = tv_tables::Table::generate(id, tv_tables::TableKind::Range { bits }).unwrap();
        for value in items(col) {
            assert!(table.row(number(value)).is_some());
        }
    }
    let rejected = Some(GadgetError::Rejected {
        gadget: GadgetId(number(member(expected, "gadget"))),
    });
    assert_eq!(root.check(&point, &sources, &keys, &zeros).err(), rejected);
    for row in 0..4 {
        let point: Vec<_> = (0..2)
            .map(|i| {
                if row >> i & 1 == 0 {
                    Fp2::ZERO
                } else {
                    Fp2::ONE
                }
            })
            .collect();
        let sources: Vec<_> = list(expected, "columns")
            .iter()
            .map(|c| Fp2::from(Fp::from_canonical(number(&items(c)[row])).unwrap()))
            .collect();
        let keys: Vec<_> = list(expected, "key_columns")
            .iter()
            .map(|c| Fp2::from(Fp::from_canonical(number(&items(c)[row])).unwrap()))
            .collect();
        assert_eq!(
            root.check(&point, &sources, &keys, &zeros).err(),
            if row == 0 { rejected } else { None }
        );
    }
}

#[test]
fn each_altered_leaf_source_key_or_range_value_and_malformed_count_has_the_exact_gadget_id() {
    let fixture = read_vectors("gadget-finals/roots.json");
    for case in list(&fixture, "cases") {
        let (claims, graph) = model(case);
        let plan = Plan::derive(&graph, claims.gadget_policy).unwrap();
        for (root, expected) in plan.roots().map(Result::unwrap).zip(list(case, "roots")) {
            let point = fields(member(expected, "point"));
            let sources = fields(member(expected, "values"));
            let keys = fields(member(expected, "keys"));
            let zeros = vec![Fp2::ZERO; keys.len()];
            let rejected = Some(GadgetError::Rejected {
                gadget: root.gadget(),
            });
            for i in 0..sources.len() {
                let mut v = sources.clone();
                v[i] += Fp2::ONE;
                assert_eq!(root.check(&point, &v, &keys, &zeros).err(), rejected);
            }
            let low = (number::<usize>(member(expected, "width")) - 1) / 8;
            let m = fields(member(expected, "product_targets")).len() / 2;
            // Change a committed comparison and its Range key together, so the
            // affine source binding alone cannot reject the altered witness.
            for i in 5 + low..sources.len() {
                let mut changed_sources = sources.clone();
                let mut changed_keys = keys.clone();
                changed_sources[i] += Fp2::ONE;
                changed_keys[i + 1] += Fp2::ONE;
                let product_carry = matches!(source(&list(expected,"sources")[i]),Source::Auxiliary(_,Auxiliary::RootComparison(Comparison::Lower,ComparisonRole::LeftCarry,j)|Auxiliary::RootComparison(Comparison::Upper,ComparisonRole::RightCarry,j)) if j as usize+1<m);
                let result = root.check(&point, &changed_sources, &changed_keys, &zeros);
                if product_carry {
                    let changed = result.unwrap();
                    let targets: Vec<_> = changed.products().map(|p| p.instance().claim).collect();
                    assert_ne!(targets, fields(member(expected, "product_targets")));
                    assert!(changed.claims().zip(&sources).any(|(c, &v)| c.value != v));
                } else {
                    assert_eq!(result.err(), rejected);
                }
            }
            for i in 0..keys.len() {
                let mut v = keys.clone();
                v[i] += Fp2::ONE;
                assert_eq!(root.check(&point, &sources, &v, &zeros).err(), rejected);
                let mut v = zeros.clone();
                v[i] = Fp2::ONE;
                assert_eq!(root.check(&point, &sources, &keys, &v).err(), rejected);
            }
            let mut extra = sources.clone();
            extra.push(Fp2::ZERO);
            assert_eq!(root.check(&point, &extra, &keys, &zeros).err(), rejected);
            assert_eq!(
                root.check(&point, &sources[..sources.len() - 1], &keys, &zeros)
                    .err(),
                rejected
            );
            let mut extra = keys.clone();
            extra.push(Fp2::ZERO);
            assert_eq!(root.check(&point, &sources, &extra, &zeros).err(), rejected);
            assert_eq!(
                root.check(&point, &sources, &keys[..keys.len() - 1], &zeros)
                    .err(),
                rejected
            );
            assert_eq!(
                root.check(&point, &sources, &keys, &zeros[..zeros.len() - 1])
                    .err(),
                rejected
            );
            let mut extra = zeros.clone();
            extra.push(Fp2::ZERO);
            assert_eq!(root.check(&point, &sources, &keys, &extra).err(), rejected);
            assert_eq!(
                root.check(&point[..1], &sources, &keys, &zeros).err(),
                Some(GadgetError::Internal)
            );
            let mut extra = point.clone();
            extra.push(Fp2::ZERO);
            assert_eq!(
                root.check(&extra, &sources, &keys, &zeros).err(),
                Some(GadgetError::Internal)
            );
            let checked = root.check(&point, &sources, &keys, &zeros).unwrap();
            let product = checked.products().next().unwrap();
            let operands = vec![Fp2::ZERO; product.instance().evals];
            assert_eq!(
                product
                    .evaluate(&point, &operands[..operands.len() - 1])
                    .err()
                    .map(|e| e.code()),
                Some("REJ-GADGET")
            );
            let mut long = operands.clone();
            long.push(Fp2::ZERO);
            assert_eq!(
                product
                    .evaluate(&point, &long)
                    .err()
                    .map(|e| e.rejected_gadget()),
                Some(Some(root.gadget()))
            );
            assert_eq!(
                product.evaluate(&extra, &operands).err().map(|e| e.code()),
                Some("ERR-INTERNAL")
            );
            assert_eq!(
                root.padding(&point, &point, &[])
                    .err()
                    .map(|e| e.rejected_gadget()),
                Some(Some(root.gadget()))
            );
            assert_eq!(
                root.padding(&point, &point, &[Fp2::ZERO; 2])
                    .err()
                    .map(|e| e.code()),
                Some("REJ-GADGET")
            );
            assert_eq!(
                root.padding(&extra, &point, &[Fp2::ZERO])
                    .err()
                    .map(|e| e.code()),
                Some("ERR-INTERNAL")
            );
            assert_eq!(
                root.padding(&point, &extra, &[Fp2::ZERO])
                    .err()
                    .map(|e| e.code()),
                Some("ERR-INTERNAL")
            );
        }
    }
}
