// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Independent two-stage certificates; canonical role/source binding is still owed.
mod claim_support;
mod support;
use claim_support::{integer, items, list, member, number};
use support::{Context, digest, extension};
use tv_core::gadgets::{
    GadgetError,
    argmax::{Argmax, Claim, Role},
};
use tv_core::sumcheck::{self, Batch, Prefix, SumcheckError, SumcheckProof};
use tv_core::validation::GadgetId;
use tv_field::{Fp, Fp2};
use tv_hash::transcript::{Ascii, Transcript};
use tv_testkit::{Json, read_vectors};

fn field(v: &Json) -> Fp2 {
    let a = items(v);
    Fp2::new(
        Fp::from_canonical(number(&a[0])).unwrap(),
        Fp::from_canonical(number(&a[1])).unwrap(),
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
fn site(case: &Json) -> Argmax {
    Argmax::derive(
        GadgetId(number(member(case, "gadget"))),
        number(member(case, "vocab")),
        number(member(case, "rows")),
    )
    .unwrap()
}
fn proof(record: &Json) -> SumcheckProof {
    SumcheckProof {
        rounds: list(record, "rounds").iter().map(fields).collect(),
        final_evals: fields(member(record, "final_evals")),
    }
}
fn before(case: &Json) -> Transcript {
    let mut t = Context::from_json(member(case, "context")).transcript();
    let claims = fields(member(case, "claims"));
    let mut bytes = (claims.len() as u64).to_le_bytes().to_vec();
    for c in claims {
        bytes.extend_from_slice(&c.to_le_bytes());
    }
    t.absorb(Ascii::new("stage/0/split").unwrap(), &bytes);
    t
}
fn first(
    site: Argmax,
    case: &Json,
    t: &mut Transcript,
    p: &SumcheckProof,
    omit: Option<usize>,
) -> Result<Batch, SumcheckError> {
    let keypoint = fields(member(case, "key_target"));
    let rowpoint = fields(member(case, "row_target"));
    let claims = fields(member(case, "claims"));
    let key = site.lookup(&keypoint, claims[0], Fp2::ZERO).unwrap();
    let selected = site.selected(&rowpoint, claims[1]).unwrap();
    assert!(truth(case, selected.input_claim()));
    let instances = [key.instance(), selected.instance()];
    sumcheck::verify_batch(t, Prefix::Stage(1), 3, &instances, p, |i, point, values| {
        if omit == Some(i) {
            Some(instances[i].claim * point.iter().fold(Fp2::ONE, |a, &x| a * (Fp2::ONE - x)))
        } else {
            (if i == 0 {
                key.evaluate(point, values)
            } else {
                selected.evaluate(point, values)
            })
            .ok()
            .map(|v| v.value())
        }
    })
}
fn run(
    site: Argmax,
    case: &Json,
    record: &Json,
    omit: Option<usize>,
) -> Result<([Batch; 2], Transcript), (u16, SumcheckError)> {
    let mut t = before(case);
    let p1 = proof(member(record, "first"));
    let p2 = proof(member(record, "second"));
    let first = first(site, case, &mut t, &p1, omit.filter(|&i| i < 2)).map_err(|e| (1, e))?;
    let suffix = site.suffix(&first.point, p1.final_evals[2]).unwrap();
    let instance = suffix.instance();
    let second = sumcheck::verify_batch(
        &mut t,
        Prefix::Stage(2),
        2,
        &[instance],
        &p2,
        |_, point, values| {
            if omit == Some(2) {
                Some(instance.claim * point.iter().fold(Fp2::ONE, |a, &x| a * (Fp2::ONE - x)))
            } else {
                suffix.evaluate(point, values).ok().map(|v| v.value())
            }
        },
    )
    .map_err(|e| (2, e))?;
    Ok(([first, second], t))
}
fn truth(case: &Json, claim: Claim<'_>) -> bool {
    let key = match claim.role {
        Role::Maximum => "maxima",
        Role::Logits => "logits",
        Role::Indicator => "h",
        Role::Suffix => "suffix",
    };
    claim.value == extension(&column(member(case, key)), claim.point)
}
fn first_claims(
    site: Argmax,
    case: &Json,
    record: &Json,
    p: &SumcheckProof,
    point: &[Fp2],
) -> bool {
    let keypoint = fields(member(case, "key_target"));
    let rowpoint = fields(member(case, "row_target"));
    let incoming = fields(member(case, "claims"));
    let key = site.lookup(&keypoint, incoming[0], Fp2::ZERO).unwrap();
    let selected = site.selected(&rowpoint, incoming[1]).unwrap();
    let results = [
        key.evaluate(point, &p.final_evals[..3]).unwrap(),
        selected.evaluate(point, &p.final_evals[3..]).unwrap(),
    ];
    let expected_roles = [
        vec![Role::Maximum, Role::Logits, Role::Suffix],
        vec![Role::Indicator, Role::Logits],
    ];
    let mut all = true;
    for (i, result) in results.iter().enumerate() {
        let claims: Vec<_> = result.claims().collect();
        assert_eq!(
            claims.iter().map(|c| c.role).collect::<Vec<_>>(),
            expected_roles[i]
        );
        let (start, end) = if i == 0 { (0, 3) } else { (3, 5) };
        if p.final_evals[start..end] == fields(member(record, "final_evals"))[start..end] {
            assert_eq!(result.value(), field(&list(record, "final_relations")[i]));
        }
        for (j, claim) in claims.iter().enumerate() {
            assert_eq!(claim.value, p.final_evals[start + j]);
            assert_eq!(
                claim.point,
                if claim.role == Role::Maximum {
                    &point[site.variables() - site.row_variables()..]
                } else {
                    point
                }
            );
            all &= truth(case, *claim);
        }
    }
    all
}
#[test]
fn two_stage_argmax_proofs_reject_every_isolated_alternative_and_adapted_omission() {
    let fixture = read_vectors("gadget-finals/argmax.json");
    let mut counts = [0; 3];
    let range = tv_tables::Table::generate(
        tv_tables::TableId(29),
        tv_tables::TableKind::Range { bits: 29 },
    )
    .unwrap();
    for case in list(&fixture, "cases") {
        let site = site(case);
        let record = member(case, "proof");
        let p1 = proof(member(record, "first"));
        let p2 = proof(member(record, "second"));
        let point1 = fields(member(member(record, "first"), "point"));
        let point2 = fields(member(member(record, "second"), "point"));
        assert!(first_claims(
            site,
            case,
            member(record, "first"),
            &p1,
            &point1
        ));
        let suffix = site.suffix(&point1, p1.final_evals[2]).unwrap();
        let evaluated = suffix.evaluate(&point2, &p2.final_evals).unwrap();
        assert_eq!(
            evaluated.value(),
            field(&list(member(record, "second"), "final_relations")[0])
        );
        let claims: Vec<_> = evaluated.claims().collect();
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].role, Role::Indicator);
        assert_eq!(claims[0].point, point2);
        assert!(truth(case, claims[0]));
        for key in list(case, "keys") {
            assert_eq!(
                range.row(number(key)),
                Some((Fp::from_canonical(number(key)).unwrap(), Fp::ZERO))
            );
        }
        let failed = member(case, "failed").as_u64().map(|v| v as usize);
        let full = run(site, case, record, None);
        if let Some(failed) = failed {
            let stage = if failed == 2 { 2 } else { 1 };
            assert_eq!(
                full.unwrap_err(),
                (
                    stage,
                    SumcheckError::FinalMismatch {
                        round: site.variables() as u32
                    }
                )
            );
            counts[1] += 1;
            let record = member(case, "omission");
            let ([first, second], t) = run(site, case, record, Some(failed)).unwrap();
            assert_eq!(
                first.point,
                fields(member(member(record, "first"), "point"))
            );
            assert_eq!(
                second.point,
                fields(member(member(record, "second"), "point"))
            );
            assert_eq!(
                *t.state(),
                digest(member(member(record, "second"), "state").as_str().unwrap())
            );
            assert!(first_claims(
                site,
                case,
                member(record, "first"),
                &proof(member(record, "first")),
                &first.point
            ));
            let suffix = site
                .suffix(
                    &first.point,
                    field(&list(member(record, "first"), "final_evals")[2]),
                )
                .unwrap();
            assert!(
                suffix
                    .evaluate(
                        &second.point,
                        &fields(member(member(record, "second"), "final_evals"))
                    )
                    .unwrap()
                    .claims()
                    .all(|c| truth(case, c))
            );
            assert_eq!(
                run(site, case, record, None).unwrap_err(),
                (
                    stage,
                    SumcheckError::FinalMismatch {
                        round: site.variables() as u32
                    }
                )
            );
            counts[2] += 1;
        } else {
            let ([first, second], t) = full.unwrap();
            assert_eq!(first.point, point1);
            assert_eq!(second.point, point2);
            assert_eq!(first.gamma, field(member(member(record, "first"), "gamma")));
            assert_eq!(
                second.gamma,
                field(member(member(record, "second"), "gamma"))
            );
            assert_eq!(
                *t.state(),
                digest(member(member(record, "second"), "state").as_str().unwrap())
            );
            assert_eq!(
                t.counter(),
                number(member(member(record, "second"), "counter"))
            );
            counts[0] += 1;
        }
    }
    assert_eq!(counts, [6, 6, 6]);
}
#[test]
fn every_argmax_final_evaluation_leaves_a_rejection_or_false_canonical_role_claim() {
    let fixture = read_vectors("gadget-finals/argmax.json");
    for case in list(&fixture, "cases")
        .iter()
        .filter(|c| member(c, "failed") == &Json::Null)
    {
        let site = site(case);
        let record = member(case, "proof");
        let p1 = proof(member(record, "first"));
        let p2 = proof(member(record, "second"));
        for (i, _) in p1.final_evals.iter().enumerate() {
            let mut changed = p1.clone();
            changed.final_evals[i] += Fp2::ONE;
            match first(site, case, &mut before(case), &changed, None) {
                Err(e) => assert_eq!(
                    e,
                    SumcheckError::FinalMismatch {
                        round: site.variables() as u32
                    }
                ),
                Ok(batch) => assert!(!first_claims(
                    site,
                    case,
                    member(record, "first"),
                    &changed,
                    &batch.point
                )),
            }
        }
        let mut t = before(case);
        let first = first(site, case, &mut t, &p1, None).unwrap();
        let suffix = site.suffix(&first.point, p1.final_evals[2]).unwrap();
        let mut changed = p2.clone();
        changed.final_evals[0] += Fp2::ONE;
        match sumcheck::verify_batch(
            &mut t,
            Prefix::Stage(2),
            2,
            &[suffix.instance()],
            &changed,
            |_, p, e| suffix.evaluate(p, e).ok().map(|v| v.value()),
        ) {
            Err(e) => assert_eq!(
                e,
                SumcheckError::FinalMismatch {
                    round: site.variables() as u32
                }
            ),
            Ok(batch) => assert!(
                !suffix
                    .evaluate(&batch.point, &changed.final_evals)
                    .unwrap()
                    .claims()
                    .all(|c| truth(case, c))
            ),
        }
    }
}
#[test]
fn argmax_metadata_lookup_and_exact_own_error_precedence_are_checked() {
    for (v, r) in [(0, 1), (1, 0), ((1 << 18) + 1, 1), (1, (1 << 17) + 1)] {
        assert!(matches!(
            Argmax::derive(GadgetId(73), v, r),
            Err(GadgetError::Internal)
        ));
    }
    let boundary = Argmax::derive(GadgetId(73), 1 << 18, 1 << 17).unwrap();
    assert_eq!(boundary.variables(), 35);
    assert_eq!(boundary.row_variables(), 17);
    let fixture = read_vectors("gadget-finals/argmax.json");
    for case in list(&fixture, "cases") {
        let site = site(case);
        assert_eq!(site.gadget(), GadgetId(number(member(case, "gadget"))));
        assert_eq!(site.vocab(), number(member(case, "vocab")));
        assert_eq!(site.rows(), number(member(case, "rows")));
        let point = fields(member(case, "key_target"));
        let row = fields(member(case, "row_target"));
        let incoming = fields(member(case, "claims"));
        let reject = GadgetError::Rejected {
            gadget: site.gadget(),
        };
        assert!(matches!(site.lookup(&point,incoming[0],Fp2::ONE),Err(e)if e==reject));
        let key = site.lookup(&point, incoming[0], Fp2::ZERO).unwrap();
        let lookup = key.lookup();
        assert_eq!(lookup.bits(), 29);
        assert_eq!(lookup.point, point);
        assert_eq!(lookup.key, incoming[0]);
        assert_eq!(lookup.value, Fp2::ZERO);
        let selected = site.selected(&row, incoming[1]).unwrap();
        let input = selected.input_claim();
        assert_eq!(input.role, Role::Maximum);
        assert_eq!(input.point, row);
        assert_eq!(input.value, incoming[1]);
        let suffix = site.suffix(&point, Fp2::ONE).unwrap();
        let mut wrong = point.clone();
        wrong.push(Fp2::ONE);
        let mut wrongrow = row.clone();
        wrongrow.push(Fp2::ONE);
        assert!(matches!(
            site.lookup(&wrong, Fp2::ZERO, Fp2::ONE),
            Err(GadgetError::Internal)
        ));
        assert!(matches!(
            site.selected(&wrongrow, Fp2::ZERO),
            Err(GadgetError::Internal)
        ));
        assert!(matches!(
            site.suffix(&wrong, Fp2::ZERO),
            Err(GadgetError::Internal)
        ));
        assert!(matches!(
            key.evaluate(&wrong, &[]),
            Err(GadgetError::Internal)
        ));
        assert!(matches!(
            selected.evaluate(&wrong, &[]),
            Err(GadgetError::Internal)
        ));
        assert!(matches!(
            suffix.evaluate(&wrong, &[]),
            Err(GadgetError::Internal)
        ));
        for n in 0..5 {
            let values = vec![Fp2::ZERO; n];
            if n != 3 {
                assert!(matches!(key.evaluate(&point,&values),Err(e)if e==reject));
            }
            if n != 2 {
                assert!(matches!(selected.evaluate(&point,&values),Err(e)if e==reject));
            }
            if n != 1 {
                assert!(matches!(suffix.evaluate(&point,&values),Err(e)if e==reject));
            }
        }
    }
}
