// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Elementwise relations through the independent verifier (DEP-2; RFC-0010 'Relations',
//! 'Batching within a stage'; RFC-0035 'The padding zero-check'). Every stage is proven by the
//! relations' constructors, tables and claims around `prove_batch`, and checked by
//! `tv_core::sumcheck::verify_batch` with the relations written in `common::elementwise` from the
//! RFCs, from the true targets' extensions at the input points and the claim 0 for a zero-check.
//!
//! - Each kind verifies alone at every size from 1 to 12 variables and at 20, a product and a
//!   rotation at 24, and all of them in one stage, its zero-checks at the stage's zero-check
//!   point; every claim is at the point the verifier derives, with the operand's extension there.
//! - `rotate`'s relation is RoPE's rotation of each pair, as RFC-0005 states it.
//! - A prover whose operand or public table differs in one entry, and a stage with one round
//!   value or final evaluation changed, fail the final check, `FinalMismatch { round: n_max }`.
//! - A zero-check rejects one non-boolean entry, and one non-zero value on a padded row, and
//!   accepts any value on a real row.

mod common;

use common::elementwise::{
    At, Build, Case, CaseInput, Read, prove_and_verify_cases, prove_cases, verdict,
};
use common::{Fp, Fp2, Session, SplitMix64, TvFp2, tv_fp, tv_proof};
use tessera_field::{Field, PrimeField};
use tessera_sumcheck::{Elementwise, ElementwiseShape, Input, MAX_DEGREE, SumcheckError, Term};
use tv_core::sumcheck as tv;

fn one() -> Fp2 {
    Fp2::ONE
}

fn input(read: Read, values: Vec<Fp>) -> CaseInput {
    CaseInput::new(read, values)
}

fn elementwise(vars: usize) -> ElementwiseShape {
    ElementwiseShape::elementwise(vars)
}

/// `Z = A ⊙ B`, two operands.
fn product(rng: &mut SplitMix64, vars: usize) -> Case {
    Case {
        build: Build::Mul,
        shape: elementwise(vars),
        inputs: vec![
            input(Read::Operand, rng.fps(1 << vars)),
            input(Read::Operand, rng.fps(1 << vars)),
        ],
        terms: vec![(one(), vec![0, 1])],
        at: At::Point(rng.fp2s(vars)),
    }
}

/// `Z[y, x] = R[y]·P[y, x]`: a row's scalar against the `2^low` entries of its row, as `R_q`
/// against `PV`.
fn by_row(rng: &mut SplitMix64, vars: usize, low: usize) -> Case {
    Case {
        build: Build::Mul,
        shape: elementwise(vars),
        inputs: vec![
            input(Read::Broadcast(low), rng.fps(1 << (vars - low))),
            input(Read::Operand, rng.fps(1 << vars)),
        ],
        terms: vec![(one(), vec![0, 1])],
        at: At::Point(rng.fp2s(vars)),
    }
}

/// `Z[y, c] = A[y, c]·g[c]`: a channel's value against every row, `2^own` channels.
fn by_channel(rng: &mut SplitMix64, vars: usize, own: usize) -> Case {
    Case {
        build: Build::Mul,
        shape: elementwise(vars),
        inputs: vec![
            input(Read::Operand, rng.fps(1 << vars)),
            input(Read::Broadcast(0), rng.fps(1 << own)),
        ],
        terms: vec![(one(), vec![0, 1])],
        at: At::Point(rng.fp2s(vars)),
    }
}

/// `Z = A ⊙ K`, `K` public constants.
fn by_constants(rng: &mut SplitMix64, vars: usize) -> Case {
    Case {
        build: Build::Mul,
        shape: elementwise(vars),
        inputs: vec![
            input(Read::Operand, rng.fps(1 << vars)),
            input(Read::Public, rng.fps(1 << vars)),
        ],
        terms: vec![(one(), vec![0, 1])],
        at: At::Point(rng.fp2s(vars)),
    }
}

/// RoPE on the pairs that differ in bit `bit`: `Y = X ⊙ C + X_⊕ ⊙ S_±`, with `C` and `S_±` random
/// public tables of the shape RFC-0005's constants take: `C` equal on a pair's two members, `S_±`
/// opposite.
fn rotation(rng: &mut SplitMix64, vars: usize, bit: usize) -> Case {
    let x = rng.fps(1 << vars);
    let (cos, sin) = rope_constants(rng, vars, bit);
    Case {
        build: Build::Rotate,
        shape: elementwise(vars),
        inputs: vec![
            input(Read::Operand, x.clone()),
            input(Read::Partner(bit), x),
            input(Read::Public, cos),
            input(Read::Public, sin),
        ],
        terms: vec![(one(), vec![0, 2]), (one(), vec![1, 3])],
        at: At::Point(rng.fp2s(vars)),
    }
}

/// `C` at every entry, equal on each pair, and `S_±`: `−S` at a pair's low member, `S` at its
/// high one.
fn rope_constants(rng: &mut SplitMix64, vars: usize, bit: usize) -> (Vec<Fp>, Vec<Fp>) {
    let mut cos = rng.fps(1 << vars);
    let mut sin = rng.fps(1 << vars);
    for x in (0..1usize << vars).filter(|x| (x >> bit) & 1 == 1) {
        cos[x] = cos[x ^ (1 << bit)];
        sin[x] = -sin[x ^ (1 << bit)];
    }
    (cos, sin)
}

/// `U[y] = Σ_x X[y, x]²` over rows of `2^summed` entries.
fn squares(rng: &mut SplitMix64, vars: usize, summed: usize) -> Case {
    Case {
        build: Build::SumSquares,
        shape: ElementwiseShape {
            vars,
            summed_vars: summed,
        },
        inputs: vec![input(Read::Operand, rng.fps(1 << vars))],
        terms: vec![(one(), vec![0, 0])],
        at: At::Point(rng.fp2s(vars - summed)),
    }
}

/// `0 = Σ_x eq(z, x)·(h − h²)`, `h` boolean.
fn booleanity(rng: &mut SplitMix64, vars: usize) -> Case {
    let h = (0..1 << vars)
        .map(|_| if rng.coin() { Fp::ONE } else { Fp::ZERO })
        .collect();
    Case {
        build: Build::Booleanity,
        shape: elementwise(vars),
        inputs: vec![input(Read::Operand, h)],
        terms: vec![(one(), vec![0]), (-one(), vec![0, 0])],
        at: At::Zero,
    }
}

/// `0 = Σ_x eq(z, x)·(w − ψ·w)`: the first `real` of `2^vars` rows real (`ψ = 1`), with random
/// values, and the others padded (`ψ = 0`), with zeros.
fn padding(rng: &mut SplitMix64, vars: usize, real: usize) -> Case {
    let psi: Vec<Fp> = (0..1 << vars)
        .map(|x| if x < real { Fp::ONE } else { Fp::ZERO })
        .collect();
    let w = (0..1 << vars)
        .map(|x| if x < real { rng.fp() } else { Fp::ZERO })
        .collect();
    Case {
        build: Build::Padding,
        shape: elementwise(vars),
        inputs: vec![input(Read::Operand, w), input(Read::Public, psi)],
        terms: vec![(one(), vec![0]), (-one(), vec![1, 0])],
        at: At::Zero,
    }
}

/// SHA-256's majority on three operands, `ab + ac + bc − 2abc`, plus a constant 5 and a public
/// table times a broadcast operand: degree 4 with `eq`, and a term with no factor.
fn general(rng: &mut SplitMix64, vars: usize) -> Case {
    let five = Fp2::from(Fp::from_u64_reduced(5));
    let two = Fp2::from(Fp::from_u64_reduced(2));
    Case {
        build: Build::New,
        shape: elementwise(vars),
        inputs: vec![
            input(Read::Operand, rng.fps(1 << vars)),
            input(Read::Operand, rng.fps(1 << vars)),
            input(Read::Operand, rng.fps(1 << vars)),
            input(Read::Public, rng.fps(1 << vars)),
            input(Read::Broadcast(vars / 2), rng.fps(1 << (vars - vars / 2))),
        ],
        terms: vec![
            (one(), vec![0, 1]),
            (one(), vec![0, 2]),
            (one(), vec![1, 2]),
            (-two, vec![0, 1, 2]),
            (five, vec![]),
            (one(), vec![3, 4]),
        ],
        at: At::Point(rng.fp2s(vars)),
    }
}

/// Every kind at `vars` variables, those that need more taking at least what they need.
fn every_kind(rng: &mut SplitMix64, vars: usize) -> Vec<Case> {
    let vars = vars.max(1);
    vec![
        product(rng, vars),
        by_row(rng, vars, vars / 2),
        by_channel(rng, vars, vars - vars / 3),
        by_constants(rng, vars),
        rotation(rng, vars, vars - 1),
        rotation(rng, vars, 0),
        squares(rng, vars, vars / 2),
        squares(rng, vars, vars),
        booleanity(rng, vars),
        padding(rng, vars, (1 << vars) - (1 << vars) / 3),
        general(rng, vars),
    ]
}

fn degree(cases: &[Case]) -> usize {
    cases
        .iter()
        .map(|case| case.relation().degree())
        .max()
        .expect("a case")
}

/// Every kind alone at 1 to 12 variables and at 20, and a product and a rotation at 24, the
/// top of RFC-0010's range.
#[test]
fn each_relation_verifies_alone_from_one_to_twenty_four_variables() {
    let mut rng = SplitMix64::new(0x99_0001);
    let session = Session::random(&mut rng);
    for case in [product(&mut rng, 24), rotation(&mut rng, 24, 6)] {
        let proven = prove_and_verify_cases(&session, 1, 3, &[case]);
        assert_eq!(proven.proven.proof().rounds.len(), 24);
    }
    for vars in (1..=12).chain([20]) {
        for (k, case) in every_kind(&mut rng, vars).into_iter().enumerate() {
            let degree = case.relation().degree();
            let expected = match case.build {
                Build::New => 4,
                Build::SumSquares if case.shape.summed_vars == vars => 2,
                _ => 3,
            };
            assert_eq!(degree, expected, "{vars} variables, case {k}");
            let proven = prove_and_verify_cases(&session, 1, degree, &[case]);
            assert_eq!(proven.proven.proof().rounds.len(), vars);
        }
    }
}

/// All the kinds in one stage at 3 to 11 variables, front-padded to the largest, with their
/// zero-checks of 3, 6 and 9 variables at the last coordinates of one zero-check point.
#[test]
fn a_stage_of_every_relation_verifies_with_its_zero_checks_at_the_stage_point() {
    let mut rng = SplitMix64::new(0x99_0002);
    let session = Session::random(&mut rng);
    let cases: Vec<Case> = [3, 6, 9, 11]
        .into_iter()
        .flat_map(|vars| every_kind(&mut rng, vars))
        .collect();
    let degree = degree(&cases);
    assert_eq!(degree, 4);
    let proven = prove_and_verify_cases(&session, 1, degree, &cases);
    assert_eq!(proven.zero.len(), 11, "the largest zero-check's variables");
    assert_eq!(proven.proven.proof().rounds.len(), 11);
    // A stage of zero-checks alone, at another stage's labels.
    let zero: Vec<Case> = cases.into_iter().filter(Case::is_zero_check).collect();
    let proven = prove_and_verify_cases(&session, 4, 3, &zero);
    assert_eq!(proven.zero.len(), 11);
}

/// The relation `rotate` proves, written in `common::elementwise` from RFC-0010 as
/// `X ⊙ C + X_⊕ ⊙ S_±`, is RFC-0005's RoPE: each pair `(a, b) = (x_i, x_{i+m})` becomes
/// `(a·C − b·S, a·S + b·C)`, here on rows of `2m = 2^4` entries, 2^3 heads and 2^2 positions.
#[test]
fn rotate_is_ropes_rotation_of_each_pair() {
    let mut rng = SplitMix64::new(0x99_0003);
    let session = Session::random(&mut rng);
    let (vars, bit) = (9, 3);
    let case = rotation(&mut rng, vars, bit);
    let [x, _, cos, sin] = [0, 1, 2, 3].map(|k| case.inputs[k].values.clone());
    let m = 1 << bit;
    let rope: Vec<TvFp2> = (0..1usize << vars)
        .map(|index| {
            let low = index & !m;
            let (a, b) = (tv_fp(x[low]), tv_fp(x[low | m]));
            // C and S of the pair's frequency, at its low member: S_± holds −S there.
            let (c, s) = (tv_fp(cos[low]), -tv_fp(sin[low]));
            TvFp2::from(if index & m == 0 {
                a * c - b * s
            } else {
                a * s + b * c
            })
        })
        .collect();
    assert_eq!(case.target(), rope);
    let _ = prove_and_verify_cases(&session, 2, 3, &[case]);
}

fn final_mismatch(rounds: usize) -> Result<(), tv::SumcheckError> {
    Err(tv::SumcheckError::FinalMismatch {
        round: rounds as u32,
    })
}

/// A prover whose input `k` differs in one entry from the case's, operand or public table,
/// proving against the true claim, fails the final check: alone, and among the other kinds in
/// one stage. A rotation's partner is its operand, changed with it. The zero-checks' claim is 0
/// whatever the witness, so their own test changes them where the check must catch it.
#[test]
fn a_prover_with_one_input_entry_changed_fails_the_final_check() {
    let mut rng = SplitMix64::new(0x99_0004);
    let session = Session::random(&mut rng);
    let honest = every_kind(&mut rng, 7);
    let degree = degree(&honest);
    for (i, case) in honest.iter().enumerate() {
        if case.is_zero_check() {
            continue;
        }
        for k in 0..case.inputs.len() {
            if matches!(case.inputs[k].read, Read::Partner(_)) {
                continue;
            }
            for entry in [0, 1, 37] {
                let mut cheating = case.clone();
                let values = &mut cheating.inputs[k].values;
                let len = values.len();
                values[entry % len] += Fp::ONE;
                let proven = prove_cases(&session, 3, degree, std::slice::from_ref(&cheating));
                assert_eq!(
                    verdict(
                        &session,
                        3,
                        degree,
                        std::slice::from_ref(case),
                        &tv_proof(proven.proven.proof())
                    ),
                    final_mismatch(7),
                    "case {i}, input {k}, entry {entry}"
                );
                let mut stage = honest.clone();
                stage[i] = cheating;
                let proven = prove_cases(&session, 3, degree, &stage);
                assert_eq!(
                    verdict(
                        &session,
                        3,
                        degree,
                        &honest,
                        &tv_proof(proven.proven.proof())
                    ),
                    final_mismatch(7),
                    "case {i}, input {k}, entry {entry}, in the stage"
                );
            }
        }
    }
}

/// A non-zero change: one, or a random element.
fn delta(rng: &mut SplitMix64) -> Fp2 {
    loop {
        let x = if rng.coin() { Fp2::ONE } else { rng.fp2() };
        if x != Fp2::ZERO {
            return x;
        }
    }
}

#[test]
fn every_change_to_a_stage_of_elementwise_relations_is_rejected() {
    let mut rng = SplitMix64::new(0x99_0005);
    let session = Session::random(&mut rng);
    let mut cases = every_kind(&mut rng, 5);
    cases.extend(every_kind(&mut rng, 3));
    let degree = degree(&cases);
    let proven = prove_cases(&session, 2, degree, &cases);
    let proof = proven.proven.proof();
    let check = |proof: &tessera_sumcheck::SumcheckProof<Fp2>| {
        verdict(&session, 2, degree, &cases, &tv_proof(proof))
    };
    assert_eq!(check(proof), Ok(()), "the honest stage verifies");
    for j in 0..proof.rounds.len() {
        for m in 0..degree {
            let mut changed = proof.clone();
            changed.rounds[j][m] += delta(&mut rng);
            assert_eq!(check(&changed), final_mismatch(5), "round {j}, value {m}");
        }
    }
    for e in 0..proof.final_evals.len() {
        let mut changed = proof.clone();
        changed.final_evals[e] += delta(&mut rng);
        assert_eq!(check(&changed), final_mismatch(5), "final evaluation {e}");
    }
}

/// `OneHot`'s booleanity rejects a 2, and a padding zero-check rejects a value on a padded row
/// but accepts any value on a real row: the check pins exactly the padded rows (RFC-0035).
#[test]
fn zero_checks_reject_a_non_boolean_entry_and_a_value_on_a_padded_row() {
    let mut rng = SplitMix64::new(0x99_0006);
    let session = Session::random(&mut rng);
    for (vars, real) in [(1, 1), (4, 5), (10, 777)] {
        let honest = [booleanity(&mut rng, vars), padding(&mut rng, vars, real)];
        let mut not_boolean = honest.clone();
        not_boolean[0].inputs[0].values[(1 << vars) - 1] = Fp::from_u64_reduced(2);
        let mut padded_value = honest.clone();
        padded_value[1].inputs[0].values[(1 << vars) - 1] = rng.fp();
        let mut real_value = honest.clone();
        real_value[1].inputs[0].values[0] = rng.fp();
        let _ = prove_and_verify_cases(&session, 1, 3, &honest);
        for (what, cheating) in [
            ("a 2", &not_boolean),
            ("a padded row's value", &padded_value),
        ] {
            let proven = prove_cases(&session, 1, 3, cheating);
            assert_eq!(
                verdict(&session, 1, 3, cheating, &tv_proof(proven.proven.proof())),
                final_mismatch(vars),
                "{what}, {vars} variables"
            );
        }
        let _ = prove_and_verify_cases(&session, 1, 3, &real_value);
    }
}

/// `Elementwise::new` and its tables refuse malformed relations with the first failure, in the
/// order its documentation gives.
#[test]
fn malformed_relations_are_refused_in_order() {
    let a: Vec<Fp> = (1..=8).map(Fp::from_u64_reduced).collect();
    let short = &a[..4];
    let term = |factors: Vec<usize>| Term::new(Fp2::ONE, factors);
    let new = |shape, inputs, terms| Elementwise::<Fp2>::new(shape, inputs, terms).map(|_| ());
    let shape = elementwise(3);
    assert_eq!(
        new(
            ElementwiseShape {
                vars: 3,
                summed_vars: 4
            },
            vec![Input::Operand(short)],
            vec![]
        ),
        Err(SumcheckError::RelationShape {
            vars: 3,
            summed_vars: 4
        })
    );
    assert_eq!(
        new(elementwise(64), vec![], vec![term(vec![])]),
        Err(SumcheckError::RelationShape {
            vars: 64,
            summed_vars: 0
        })
    );
    assert_eq!(
        new(
            shape,
            vec![Input::Operand(&a), Input::Public(short)],
            vec![term(vec![9])]
        ),
        Err(SumcheckError::InputLength {
            input: 1,
            len: 4,
            vars: 3
        })
    );
    // A broadcast of 2^2 entries fits above 1 low variable, not above 2.
    assert_eq!(
        new(
            shape,
            vec![Input::Broadcast {
                operand: short,
                low_vars: 1
            }],
            vec![term(vec![0])]
        ),
        Ok(())
    );
    assert_eq!(
        new(
            shape,
            vec![Input::Broadcast {
                operand: short,
                low_vars: 2
            }],
            vec![term(vec![0])]
        ),
        Err(SumcheckError::InputLength {
            input: 0,
            len: 4,
            vars: 3
        })
    );
    assert_eq!(
        new(
            shape,
            vec![Input::Partner {
                operand: &a,
                bit: 3
            }],
            vec![term(vec![0])]
        ),
        Err(SumcheckError::PartnerBit {
            input: 0,
            bit: 3,
            vars: 3
        })
    );
    assert_eq!(
        new(
            shape,
            vec![Input::Operand(&a)],
            vec![term(vec![0]), term(vec![0, 1])]
        ),
        Err(SumcheckError::TermInput {
            term: 1,
            factor: 1,
            inputs: 1
        })
    );
    assert_eq!(
        new(
            shape,
            vec![Input::Operand(&a)],
            vec![term(vec![0; MAX_DEGREE])]
        ),
        Err(SumcheckError::Degree {
            degree: MAX_DEGREE + 1
        }),
        "the term's factors and eq"
    );
    assert_eq!(
        new(
            ElementwiseShape {
                vars: 3,
                summed_vars: 3
            },
            vec![Input::Operand(&a)],
            vec![term(vec![0; MAX_DEGREE]), term(vec![])]
        ),
        Err(SumcheckError::Degree { degree: 0 }),
        "a term of no factor without eq"
    );
    assert_eq!(
        new(shape, vec![Input::Operand(&a)], vec![]),
        Err(SumcheckError::Degree { degree: 0 }),
        "no term"
    );
    // The tables check the input point, and the claims the instance's point and evaluations.
    let relation =
        Elementwise::<Fp2>::mul(Input::Operand(&a), Input::Public(&a), 3).expect("a relation");
    assert_eq!(
        relation.tables(&[Fp2::ONE; 2]).map(|_| ()),
        Err(SumcheckError::PointLength {
            len: 2,
            expected: 3
        })
    );
    let tables = relation.tables(&[Fp2::ONE; 3]).expect("tables");
    assert_eq!(
        tables.claims(&[Fp2::ONE; 4], &[Fp2::ONE]).map(|_| ()),
        Err(SumcheckError::PointLength {
            len: 4,
            expected: 3
        })
    );
    assert_eq!(
        tables.claims(&[Fp2::ONE; 3], &[Fp2::ONE; 2]).map(|_| ()),
        Err(SumcheckError::EvaluationCount {
            len: 2,
            expected: 1
        }),
        "one evaluation: the public table sends none"
    );
}
