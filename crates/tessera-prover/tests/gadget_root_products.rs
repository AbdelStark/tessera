// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Actual CPU sumchecks and root witnesses against the frozen independent product oracle.
use serde_json::Value;
use tessera_codec::Ascii;
use tessera_field::{Field, Goldilocks as G, GoldilocksExt2 as F, PrimeField, QuadraticExtension};
use tessera_gadgets::{Bracket, CompareColumn as C, RsqrtColumn as R, RsqrtRow, Side};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_sumcheck::{Instance, Prefix, Table, Term, prove_batch};
use tessera_tables::{GeneratedTable, Table as _, TableId, TableKind};
use tv_core::gadgets::root_products::Products;
use tv_core::sumcheck::{Instance as TvInstance, Prefix as TvPrefix, SumcheckProof, verify_batch};
use tv_field::{Fp, Fp2};

fn hex(v: &Value) -> Vec<u8> {
    v.as_str()
        .unwrap()
        .as_bytes()
        .chunks_exact(2)
        .map(|s| u8::from_str_radix(std::str::from_utf8(s).unwrap(), 16).unwrap())
        .collect()
}
fn fp(v: u64) -> G {
    G::from_canonical_u64(v).unwrap()
}
fn field(v: &Value) -> F {
    F::new(fp(v[0].as_u64().unwrap()), fp(v[1].as_u64().unwrap()))
}
fn tv(v: F) -> Fp2 {
    let [a, b] = v.coeffs();
    Fp2::new(
        Fp::from_canonical(a.to_canonical_u64()).unwrap(),
        Fp::from_canonical(b.to_canonical_u64()).unwrap(),
    )
}
fn eq(point: &[F], index: usize) -> F {
    point.iter().enumerate().fold(F::ONE, |out, (bit, &v)| {
        out * if index >> bit & 1 == 0 { F::ONE - v } else { v }
    })
}
fn transcripts(c: &Value) -> (Transcript, tv_hash::transcript::Transcript) {
    assert_eq!(c["mode"], "NonInteractive");
    let statement = hex(&c["statement_digest"]).try_into().unwrap();
    let session = hex(&c["session_id"]).try_into().unwrap();
    let ours = Transcript::new(&SessionContext {
        protocol: Ascii::new(c["protocol"].as_str().unwrap()).unwrap(),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new(c["params"].as_str().unwrap()).unwrap()),
        statement_digest: statement,
        session_id: session,
    });
    let theirs = tv_hash::transcript::Transcript::new(&tv_hash::transcript::SessionContext {
        protocol: tv_hash::transcript::Ascii::new(c["protocol"].as_str().unwrap()).unwrap(),
        mode: tv_hash::transcript::Mode::NonInteractive,
        params: tv_hash::transcript::ParamSetId(
            tv_hash::transcript::Ascii::new(c["params"].as_str().unwrap()).unwrap(),
        ),
        statement_digest: statement,
        session_id: session,
    });
    (ours, theirs)
}

#[test]
fn cpu_root_product_proofs_and_canonical_witnesses_match_the_independent_oracle() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../spec/vectors/gadget-finals/root-products.json"
    ))
    .unwrap();
    let comparisons: Value = serde_json::from_str(include_str!(
        "../../../spec/vectors/gadget-finals/comparison.json"
    ))
    .unwrap();
    let specs: Vec<_> = (1..=32)
        .map(|bits| {
            GeneratedTable::new(TableId(u32::from(bits)), TableKind::Range { bits })
                .unwrap()
                .spec()
                .clone()
        })
        .collect();
    let mut count = 0;
    for case in fixture["cases"].as_array().unwrap() {
        let width = case["width"].as_u64().unwrap() as u8;
        let plan = Products::derive(width).unwrap();
        let columns: Vec<Vec<G>> = case["columns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                c.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| {
                        assert_eq!(v[1], 0);
                        fp(v[0].as_u64().unwrap())
                    })
                    .collect()
            })
            .collect();
        let input: Vec<_> = (0..8)
            .map(|row| {
                (0..4)
                    .map(|i| columns[plan.root_limbs() + i][row].to_canonical_u64() << (16 * i))
                    .sum::<u64>() as i64
            })
            .collect();
        let roots: Vec<_> = (0..8)
            .map(|row| {
                (0..plan.root_limbs())
                    .map(|i| columns[i][row].to_canonical_u64() << (8 * i))
                    .sum::<u64>() as i64
            })
            .collect();
        let root = RsqrtRow::new(
            case["dim"].as_u64().unwrap() as u32,
            case["kappa"].as_u64().unwrap() as u8,
            case["epsilon"].as_u64().unwrap(),
            &specs,
        )
        .unwrap();
        assert_eq!(root.root_bits(), width);
        let witness = root
            .witness(
                &input,
                &[true, true, true, true, false, false, false, false],
            )
            .unwrap();
        let alternative = case.get("alternative_comparisons").is_some();
        if !alternative {
            assert_eq!(witness.tensor(R::Root).unwrap(), &roots);
            for (i, column) in columns.iter().skip(plan.root_limbs()).take(3).enumerate() {
                let actual = witness.tensor(R::InputLimb(i as u8)).unwrap();
                assert_eq!(actual.len(), column.len());
                for (&actual, expected) in actual.iter().zip(column) {
                    assert_eq!(actual as u64, expected.to_canonical_u64());
                }
            }
            for (i, column) in columns.iter().take(plan.root_limbs() - 1).enumerate() {
                let actual = witness.tensor(R::RootLimb(i as u8)).unwrap();
                assert_eq!(actual.len(), column.len());
                for (&actual, expected) in actual.iter().zip(column) {
                    assert_eq!(actual as u64, expected.to_canonical_u64());
                }
            }
            let name = case["name"].as_str().unwrap();
            for (side, bracket) in [("lower", Bracket::Lower), ("upper", Bracket::Upper)] {
                let wanted = format!("{name}-{side}");
                let comparison = comparisons["cases"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|c| c["name"] == wanted)
                    .unwrap();
                for a in 0..6 {
                    for j in 0..plan.coefficients() {
                        let col = match a {
                            0 => C::Digit(Side::Left, j),
                            1 => C::Carry(Side::Left, j),
                            2 => C::Digit(Side::Right, j),
                            3 => C::Carry(Side::Right, j),
                            4 => C::Difference(j),
                            5 => C::Borrow(j),
                            _ => unreachable!(),
                        };
                        for (row, &actual) in comparison["rows"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .zip(witness.tensor(R::Compare(bracket, col)).unwrap())
                        {
                            assert_eq!(row["arrays"][a][j].as_i64(), Some(actual));
                        }
                    }
                }
            }
        } else {
            // All public root parameters and range boxes are admissible, but
            // this candidate root is wrong. A product proof must not certify it.
            assert_ne!(witness.tensor(R::Root).unwrap(), &roots);
            let bad = &case["alternative_comparisons"][0];
            let bounds = |i| {
                bad["bounds"][i]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap())
                    .collect::<Vec<_>>()
            };
            let comparison =
                tv_core::gadgets::comparison::Comparison::derive(8, &bounds(0), &bounds(1))
                    .unwrap();
            let row = &bad["rows"][0];
            let values = |v: &Value| {
                v.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| Fp2::from(Fp::from_canonical(v.as_u64().unwrap()).unwrap()))
                    .collect::<Vec<_>>()
            };
            let left = values(&row["left"]);
            let right = values(&row["right"]);
            let arrays: [Vec<_>; 6] = std::array::from_fn(|a| values(&row["arrays"][a]));
            let checked = comparison.check(
                tv_core::validation::GadgetId(99),
                tv_core::gadgets::comparison::Values {
                    point: &[],
                    left: &left,
                    right: &right,
                    arrays: arrays.each_ref().map(Vec::as_slice),
                },
            );
            assert_eq!(
                checked.err(),
                Some(tv_core::gadgets::GadgetError::Rejected {
                    gadget: tv_core::validation::GadgetId(99)
                })
            );
        }
        let target: Vec<_> = case["target"]
            .as_array()
            .unwrap()
            .iter()
            .map(field)
            .collect();
        let equals: Vec<_> = (0..8).map(|i| eq(&target, i)).collect();
        let plus: Vec<_> = columns[0].iter().map(|&v| v + G::ONE).collect();
        let mut instances = Vec::new();
        for increment in [false, true] {
            for j in 0..plan.coefficients() {
                let mut tables = vec![Table::ext(&equals).public()];
                tables.extend(columns.iter().map(|c| Table::base(c)));
                let shifted = tables.len();
                tables.push(Table::base(&plus).public());
                let mut terms = Vec::new();
                for i in 0..plan.root_limbs() {
                    for h in 0..plan.root_limbs() {
                        for k in 0..4 {
                            if i + h + 2 * k == j {
                                let r = |i| if increment && i == 0 { shifted } else { 1 + i };
                                terms.push(Term::new(
                                    F::ONE,
                                    vec![0, r(i), r(h), 1 + plan.root_limbs() + k],
                                ));
                            }
                        }
                    }
                }
                if terms.is_empty() {
                    terms.push(Term::new(F::ZERO, vec![0]));
                }
                instances.push(Instance::sum_of_products(tables, terms));
            }
        }
        let claims: Vec<_> = case["claims"]
            .as_array()
            .unwrap()
            .iter()
            .map(field)
            .collect();
        let (mut ours, mut theirs) = transcripts(&case["context"]);
        ours.absorb("stage/0/split", &claims);
        let mut bytes = (claims.len() as u64).to_le_bytes().to_vec();
        for &v in &claims {
            bytes.extend_from_slice(&tv(v).to_le_bytes());
        }
        theirs.absorb(
            tv_hash::transcript::Ascii::new("stage/0/split").unwrap(),
            &bytes,
        );
        let proven = prove_batch::<F, _>(&mut ours, Prefix::Stage(1), 4, &instances).unwrap();
        let p = proven.proof();
        assert_eq!(p.rounds.len(), 3);
        for (actual, expected) in p.rounds.iter().zip(case["rounds"].as_array().unwrap()) {
            assert_eq!(
                *actual,
                expected
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(field)
                    .collect::<Vec<_>>()
            );
        }
        assert_eq!(
            p.final_evals,
            case["final_evals"]
                .as_array()
                .unwrap()
                .iter()
                .map(field)
                .collect::<Vec<_>>()
        );
        let proof = SumcheckProof {
            rounds: p
                .rounds
                .iter()
                .map(|r| r.iter().copied().map(tv).collect())
                .collect(),
            final_evals: p.final_evals.iter().copied().map(tv).collect(),
        };
        let expected: Vec<_> = claims
            .iter()
            .copied()
            .map(|claim| TvInstance {
                claim: tv(claim),
                vars: 3,
                evals: plan.evaluations(),
            })
            .collect();
        let target: Vec<_> = target.into_iter().map(tv).collect();
        let checked = verify_batch(
            &mut theirs,
            TvPrefix::Stage(1),
            4,
            &expected,
            &proof,
            |i, point, evals| {
                Some(
                    tv_core::sumcheck::eq(&target, point)?
                        * plan.evaluate(
                            i % plan.coefficients(),
                            i >= plan.coefficients(),
                            evals,
                        )?,
                )
            },
        )
        .unwrap();
        assert_eq!(
            checked.point,
            proven.point().iter().copied().map(tv).collect::<Vec<_>>()
        );
        assert_eq!(ours.state().as_slice(), hex(&case["state"]));
        assert_eq!(theirs.state().as_slice(), hex(&case["state"]));
        count += expected.len();
    }
    assert_eq!(count, 826);
}
