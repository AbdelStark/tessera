// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Independent OneHot component proofs, including proofs adapted to a missing check.
mod claim_support;
mod support;

use claim_support::{claims_of, integer, items, list, member, number};
use support::{Context, digest, extension};
use tv_core::advice::Source;
use tv_core::gadgets::{
    GadgetError, Plan,
    onehot::{OneHot, Relation},
};
use tv_core::lowering::{self, NodeId, TensorId};
use tv_core::sumcheck::{self, Batch, Prefix, SumcheckError, SumcheckProof};
use tv_core::validation::{self, GadgetId};
use tv_field::{Fp, Fp2};
use tv_hash::transcript::{Ascii, Transcript};
use tv_testkit::{Json, read_vectors};

fn field(v: &Json) -> Fp2 {
    let pair = items(v);
    Fp2::new(
        Fp::from_canonical(number(&pair[0])).unwrap(),
        Fp::from_canonical(number(&pair[1])).unwrap(),
    )
}
fn fields(v: &Json) -> Vec<Fp2> {
    items(v).iter().map(field).collect()
}
fn base(v: &Json) -> Fp {
    Fp::from_canonical(integer(v).rem_euclid(i128::from(Fp::MODULUS)) as u64).unwrap()
}
fn site(case: &Json) -> OneHot {
    let mut claims = claims_of("tiny-2-servable.d");
    claims.arch.vocab = number(member(case, "vocab"));
    claims.arch.max_context = 8;
    let public = validation::check_public(&claims).unwrap();
    let graph = lowering::lower(&public, number(member(case, "n"))).unwrap();
    let plan = Plan::derive(&graph, claims.gadget_policy).unwrap();
    let sites: Vec<_> = plan.one_hots().map(Result::unwrap).collect();
    assert_eq!(sites.len(), 1);
    assert_eq!(sites[0].gadget(), GadgetId(0));
    assert_eq!(sites[0].node(), NodeId(0));
    sites[0]
}
fn proof(record: &Json) -> SumcheckProof {
    SumcheckProof {
        rounds: list(record, "rounds").iter().map(fields).collect(),
        final_evals: fields(member(record, "final_evals")),
    }
}
fn before(case: &Json, record: &Json) -> (Transcript, Vec<Fp2>) {
    let mut t = Context::from_json(member(case, "context")).transcript();
    let mut bytes = 1u64.to_le_bytes().to_vec();
    bytes.extend_from_slice(&field(&list(case, "claims")[3]).to_le_bytes());
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
    site: OneHot,
    case: &Json,
    record: &Json,
    proof: &SumcheckProof,
    omit: Option<usize>,
) -> Result<(Batch, Transcript), SumcheckError> {
    let (mut t, zero) = before(case, record);
    let target = fields(member(case, "target"));
    let bound = site.at(&target, field(&list(case, "claims")[3])).unwrap();
    let relations: Vec<_> = site.relations().collect();
    let instances: Vec<_> = relations.iter().map(|&r| bound.instance(r)).collect();
    let checked = sumcheck::verify_batch(
        &mut t,
        Prefix::Stage(1),
        3,
        &instances,
        proof,
        |i, point, values| {
            if omit == Some(i) {
                Some(instances[i].claim * point.iter().fold(Fp2::ONE, |a, &x| a * (Fp2::ONE - x)))
            } else {
                bound
                    .evaluate(relations[i], &zero, point, values)
                    .ok()
                    .map(|v| v.value())
            }
        },
    )?;
    Ok((checked, t))
}
fn claims_are_true(
    site: OneHot,
    case: &Json,
    record: &Json,
    proof: &SumcheckProof,
    point: &[Fp2],
) -> bool {
    let zero = fields(member(record, "zero"));
    let h: Vec<_> = list(case, "h").iter().map(base).collect();
    let tokens: Vec<_> = list(case, "tokens").iter().map(base).collect();
    let target = fields(member(case, "target"));
    let bound = site.at(&target, field(&list(case, "claims")[3])).unwrap();
    let token_claim = bound.input_claim();
    assert_eq!(token_claim.input.source, Source::Tensor(TensorId(0)));
    assert_eq!(token_claim.input.point, target);
    assert_eq!(token_claim.value, field(&list(case, "claims")[3]));
    let mut offset = 0;
    let mut truth = token_claim.value == extension(&tokens, &target);
    for (i, relation) in site.relations().enumerate() {
        let values = &proof.final_evals[offset..offset + relation.evaluations()];
        offset += values.len();
        let final_value = bound.evaluate(relation, &zero, point, values).unwrap();
        // For honest fixture proofs, this is also a completely independent oracle
        // of each final relation, not just their random linear combination.
        if values == &fields(member(record, "final_evals"))[offset - values.len()..offset] {
            assert_eq!(
                final_value.value(),
                field(&list(record, "final_relations")[i])
            );
        }
        let claims: Vec<_> = final_value.claims().collect();
        assert_eq!(claims.len(), values.len());
        for (j, claim) in claims.iter().enumerate() {
            assert_eq!(claim.input.source, Source::Tensor(TensorId(2)));
            assert_eq!(claim.input.point, point);
            assert_eq!(claim.value, values[j]);
            truth &= claim.value == extension(&h, point);
        }
    }
    assert_eq!(offset, proof.final_evals.len());
    truth
}

#[test]
fn honest_component_proofs_check_all_relations_transcripts_and_canonical_source_claims() {
    let fixture = read_vectors("gadget-finals/onehot.json");
    let mut count = 0;
    for case in list(&fixture, "cases") {
        if !matches!(member(case, "failed"), Json::Null) {
            continue;
        }
        let site = site(case);
        let relations: Vec<_> = site.relations().collect();
        assert_eq!(
            relations,
            [
                Relation::Booleanity,
                Relation::Vocabulary,
                Relation::RowSum,
                Relation::Index
            ]
        );
        let proof = proof(case);
        let point = fields(member(case, "point"));
        let target = fields(member(case, "target"));
        let bound = site.at(&target, field(&list(case, "claims")[3])).unwrap();
        for (i, relation) in relations.into_iter().enumerate() {
            let instance = bound.instance(relation);
            assert_eq!(instance.claim, field(&list(case, "claims")[i]));
            assert_eq!(instance.vars, point.len());
            assert_eq!(instance.evals, 1);
            assert_eq!(
                relation.degree(),
                if matches!(relation, Relation::Booleanity | Relation::Vocabulary) {
                    3
                } else {
                    2
                }
            );
        }
        let (batch, t) = run(site, case, case, &proof, None).unwrap();
        assert_eq!(batch.point, point);
        assert_eq!(batch.gamma, field(member(case, "gamma")));
        assert_eq!(*t.state(), digest(member(case, "state").as_str().unwrap()));
        assert_eq!(t.counter(), number(member(case, "counter")));
        assert!(claims_are_true(site, case, case, &proof, &batch.point));
        for j in 0..proof.final_evals.len() {
            let mut changed = proof.clone();
            changed.final_evals[j] += Fp2::ONE;
            match run(site, case, case, &changed, None) {
                Err(error) => {
                    assert_eq!(
                        error,
                        SumcheckError::FinalMismatch {
                            round: point.len() as u32
                        }
                    )
                }
                Ok((batch, _)) => {
                    assert!(!claims_are_true(site, case, case, &changed, &batch.point))
                }
            }
        }
        count += 1;
    }
    assert_eq!(count, 6);
}

#[test]
fn each_single_obligation_alternative_fails_and_its_omission_control_passes_with_true_sources() {
    let fixture = read_vectors("gadget-finals/onehot.json");
    let mut count = 0;
    for case in list(&fixture, "cases") {
        if matches!(member(case, "failed"), Json::Null) {
            continue;
        }
        let site = site(case);
        let failed: usize = number(member(case, "failed"));
        for (i, actual) in list(case, "actual_sums").iter().enumerate() {
            assert_eq!(
                field(actual) == field(&list(case, "claims")[i]),
                i != failed
            );
        }
        let omitted = member(case, "omission");
        for record in [case, omitted] {
            let proof = proof(record);
            assert_eq!(
                run(site, case, record, &proof, None).err(),
                Some(SumcheckError::FinalMismatch {
                    round: list(record, "point").len() as u32
                })
            );
        }
        let proof = proof(omitted);
        let (batch, _) = run(site, case, omitted, &proof, Some(failed)).unwrap();
        assert!(claims_are_true(site, case, omitted, &proof, &batch.point));
        count += 1;
    }
    assert_eq!(count, 6);
}

#[test]
fn wrong_evaluation_counts_have_exact_gadget_ids_and_own_points_are_internal_errors() {
    let fixture = read_vectors("gadget-finals/onehot.json");
    for case in list(&fixture, "cases") {
        let site = site(case);
        let zero = fields(member(case, "zero"));
        let point = fields(member(case, "point"));
        let target = fields(member(case, "target"));
        let bound = site.at(&target, field(&list(case, "claims")[3])).unwrap();
        let mut wrong_target = target.clone();
        wrong_target.push(Fp2::ONE);
        assert_eq!(
            site.at(&wrong_target, Fp2::ZERO).err().map(|e| e.code()),
            Some("ERR-INTERNAL")
        );
        if !target.is_empty() {
            assert_eq!(
                site.at(&target[..target.len() - 1], Fp2::ZERO)
                    .err()
                    .map(|e| e.code()),
                Some("ERR-INTERNAL")
            );
        }
        let mut offset = 0;
        let proof = proof(case);
        for relation in site.relations() {
            let evaluations = &proof.final_evals[offset..offset + relation.evaluations()];
            offset += evaluations.len();
            let rejected = Some(GadgetError::Rejected {
                gadget: GadgetId(0),
            });
            assert_eq!(
                bound
                    .evaluate(
                        relation,
                        &zero,
                        &point,
                        &evaluations[..evaluations.len() - 1]
                    )
                    .err()
                    .map(|e| e.code()),
                Some("REJ-GADGET")
            );
            assert_eq!(
                bound
                    .evaluate(
                        relation,
                        &zero,
                        &point,
                        &evaluations[..evaluations.len() - 1]
                    )
                    .err(),
                rejected
            );
            let mut extra = evaluations.to_vec();
            extra.push(Fp2::ONE);
            assert_eq!(
                bound.evaluate(relation, &zero, &point, &extra).err(),
                rejected
            );
            let mut extra = point.clone();
            extra.push(Fp2::ONE);
            assert_eq!(
                bound.evaluate(relation, &zero, &extra, evaluations).err(),
                Some(GadgetError::Internal)
            );
            assert_eq!(
                bound.evaluate(relation, &extra, &point, evaluations).err(),
                Some(GadgetError::Internal)
            );
            if !point.is_empty() {
                assert_eq!(
                    bound
                        .evaluate(relation, &zero, &point[..point.len() - 1], evaluations)
                        .err(),
                    Some(GadgetError::Internal)
                );
                assert_eq!(
                    bound
                        .evaluate(relation, &zero[..zero.len() - 1], &point, evaluations)
                        .err(),
                    Some(GadgetError::Internal)
                );
            }
        }
    }
}
