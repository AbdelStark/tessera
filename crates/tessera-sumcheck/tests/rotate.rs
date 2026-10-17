// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RoPE's `Rotate` through the independent verifier (DEP-2; RFC-0037 'Rotate', RFC-0005 'RoPE').
//! The verifier's side is written here from the RFCs, in `tv-field`:
//!
//! - the claim is the extension, at the input point, of RoPE's output computed pair by pair:
//!   `(lo, hi) ↦ (lo·C − hi·S, lo·S + hi·C)` for the entries whose indices differ in bit `b`;
//! - the relation at the instance's point `r'` is `Ẽ_r(r')·x`, with the kernel's closed form
//!   `eq(r_b, r'_b)·⟨C, w⟩ + (r_b − r'_b)·⟨S, w⟩` and `w = eq(r_{−b}, ·)·eq(r'_{−b}, ·)`, itself
//!   checked against the kernel's extension by enumeration.
//!
//! An honest `Rotate` verifies at every size from 1 to 12 variables, every pair bit, and at 20,
//! alone and in a stage with an elementwise product. Its one claim is `X`'s extension at `r'`. A
//! prover whose `X`, `C` or `S` differs in one entry fails the final check, and so does any change
//! to a round value or the evaluation.

mod common;

use common::elementwise::extension;
use common::forests::mle;
use common::{Fp, Fp2, Session, SplitMix64, TvFp2, tv_fp, tv_fp2, tv_fp2s, tv_prefix, tv_proof};
use tessera_field::{Field, PrimeField};
use tessera_sumcheck::{
    Elementwise, Input, Instance, Prefix, Rotate, SumcheckError, SumcheckProof, prove_batch,
};
use tv_core::sumcheck as tv;

/// A rotation on random data: `X` of `2^vars` entries, and its pairs' constants.
#[derive(Clone)]
struct Case {
    vars: usize,
    bit: usize,
    x: Vec<Fp>,
    cos: Vec<Fp>,
    sin: Vec<Fp>,
    point: Vec<Fp2>,
}

impl Case {
    fn random(rng: &mut SplitMix64, vars: usize, bit: usize) -> Self {
        Self {
            vars,
            bit,
            x: rng.fps(1 << vars),
            cos: rng.fps(1 << (vars - 1)),
            sin: rng.fps(1 << (vars - 1)),
            point: rng.fp2s(vars),
        }
    }

    fn relation(&self) -> Rotate<'_, Fp2> {
        Rotate::new(&self.x, &self.cos, &self.sin, self.bit, self.vars).expect("a rotation")
    }

    /// The index of `x`'s pair, bit `b` removed.
    fn pair(&self, x: usize) -> usize {
        ((x >> (self.bit + 1)) << self.bit) | (x & ((1 << self.bit) - 1))
    }

    /// RoPE's output, pair by pair, as RFC-0005 states it.
    fn rope(&self) -> Vec<TvFp2> {
        let m = 1usize << self.bit;
        (0..1usize << self.vars)
            .map(|x| {
                let (lo, hi) = (tv_fp(self.x[x & !m]), tv_fp(self.x[x | m]));
                let k = self.pair(x);
                let (c, s) = (tv_fp(self.cos[k]), tv_fp(self.sin[k]));
                TvFp2::from(if x & m == 0 {
                    lo * c - hi * s
                } else {
                    lo * s + hi * c
                })
            })
            .collect()
    }

    /// The verifier's claim: RoPE's output's extension at the input point.
    fn tv_instance(&self) -> tv::Instance {
        tv::Instance {
            claim: extension(&self.rope(), &tv_fp2s(&self.point)),
            vars: self.vars,
            evals: 1,
        }
    }

    /// `Ẽ_r(r')` by RFC-0037's closed form.
    fn kernel_at(&self, at: &[TvFp2]) -> TvFp2 {
        let r = tv_fp2s(&self.point);
        let (r_b, at_b) = (r[self.bit], at[self.bit]);
        let without = |p: &[TvFp2]| -> Vec<TvFp2> {
            p.iter()
                .enumerate()
                .filter(|&(k, _)| k != self.bit)
                .map(|(_, &v)| v)
                .collect()
        };
        let (left, right) = (eq_table(&without(&r)), eq_table(&without(at)));
        let (mut c, mut s) = (TvFp2::ZERO, TvFp2::ZERO);
        for k in 0..left.len() {
            let w = left[k] * right[k];
            c += w * TvFp2::from(tv_fp(self.cos[k]));
            s += w * TvFp2::from(tv_fp(self.sin[k]));
        }
        let eq_b = r_b * at_b + (TvFp2::ONE - r_b) * (TvFp2::ONE - at_b);
        eq_b * c + (r_b - at_b) * s
    }

    /// `Ẽ_r(r')` by enumeration of the kernel's definition.
    fn kernel_by_enumeration(&self, at: &[TvFp2]) -> TvFp2 {
        let r = tv_fp2s(&self.point);
        let r_b = r[self.bit];
        let kernel: Vec<TvFp2> = (0..1usize << self.vars)
            .map(|x| {
                let k = self.pair(x);
                let rest_eq = (0..self.vars)
                    .filter(|&j| j != self.bit)
                    .fold(TvFp2::ONE, |e, j| {
                        e * if (x >> j) & 1 == 1 {
                            r[j]
                        } else {
                            TvFp2::ONE - r[j]
                        }
                    });
                let x_b = (x >> self.bit) & 1;
                let (eq_b, minus) = if x_b == 1 {
                    (r_b, r_b - TvFp2::ONE)
                } else {
                    (TvFp2::ONE - r_b, r_b)
                };
                rest_eq
                    * (TvFp2::from(tv_fp(self.cos[k])) * eq_b
                        + TvFp2::from(tv_fp(self.sin[k])) * minus)
            })
            .collect();
        extension(&kernel, at)
    }
}

/// `eq(point, ·)` on the point's vertices, by doubling.
fn eq_table(point: &[TvFp2]) -> Vec<TvFp2> {
    let mut table = vec![TvFp2::ONE];
    for &r in point {
        let high: Vec<TvFp2> = table.iter().map(|&e| e * r).collect();
        for (e, h) in table.iter_mut().zip(&high) {
            *e -= *h;
        }
        table.extend(high);
    }
    table
}

/// The prover's stage of rotations, with elementwise products after them, on a fresh transcript.
fn prove(
    session: &Session,
    degree: usize,
    rotations: &[Case],
    products: &[(Vec<Fp>, Vec<Fp>, Vec<Fp2>)],
) -> SumcheckProof<Fp2> {
    let rotate: Vec<_> = rotations
        .iter()
        .map(|case| case.relation().tables(&case.point).expect("a point"))
        .collect();
    let mul: Vec<_> = products
        .iter()
        .map(|(a, b, point)| {
            Elementwise::<Fp2>::mul(Input::Operand(a), Input::Operand(b), point.len())
                .expect("a product")
                .tables(point)
                .expect("a point")
        })
        .collect();
    let mut instances: Vec<Instance<'_, Fp2>> = rotate.iter().map(|t| t.instance()).collect();
    instances.extend(mul.iter().map(|t| t.instance()));
    prove_batch(
        &mut session.transcript(),
        Prefix::Stage(2),
        degree,
        &instances,
    )
    .expect("a well-formed stage")
    .into_proof()
}

/// The verifier's verdict on a stage of rotations and products, from their true claims.
fn verdict(
    session: &Session,
    degree: usize,
    rotations: &[Case],
    products: &[(Vec<Fp>, Vec<Fp>, Vec<Fp2>)],
    proof: &SumcheckProof<Fp2>,
) -> Result<tv::Batch, tv::SumcheckError> {
    let mut claims: Vec<tv::Instance> = rotations.iter().map(Case::tv_instance).collect();
    for (a, b, point) in products {
        let product: Vec<TvFp2> = a
            .iter()
            .zip(b)
            .map(|(&a, &b)| TvFp2::from(tv_fp(a) * tv_fp(b)))
            .collect();
        claims.push(tv::Instance {
            claim: extension(&product, &tv_fp2s(point)),
            vars: point.len(),
            evals: 2,
        });
    }
    tv::verify_batch(
        &mut session.tv_transcript(),
        tv_prefix(Prefix::Stage(2)),
        degree,
        &claims,
        &tv_proof(proof),
        |i, at, evals| match (rotations.get(i), evals) {
            (Some(case), [x]) => Some(case.kernel_at(at) * *x),
            (None, [a, b]) => {
                let (_, _, point) = products.get(i - rotations.len())?;
                Some(tv::eq(&tv_fp2s(point), at)? * *a * *b)
            }
            _ => None,
        },
    )
}

#[test]
fn the_kernels_closed_form_is_its_extension() {
    let mut rng = SplitMix64::new(0x113_0001);
    for vars in 1..=8 {
        for bit in 0..vars {
            let case = Case::random(&mut rng, vars, bit);
            let at = tv_fp2s(&rng.fp2s(vars));
            assert_eq!(
                case.kernel_at(&at),
                case.kernel_by_enumeration(&at),
                "{vars} variables, bit {bit}"
            );
        }
    }
}

/// Every size from 1 to 12 variables at every pair bit, and 20 variables at RoPE's bit of
/// `d_head = 128`: the verifier accepts from RoPE's output, and the one claim is `X`'s extension
/// at the instance's point.
#[test]
fn rotate_verifies_as_ropes_rotation_of_each_pair() {
    let mut rng = SplitMix64::new(0x113_0002);
    let session = Session::random(&mut rng);
    let sizes = (1..=12)
        .flat_map(|vars| (0..vars).map(move |bit| (vars, bit)))
        .chain([(20, 6)]);
    for (vars, bit) in sizes {
        let case = Case::random(&mut rng, vars, bit);
        let relation = case.relation();
        assert_eq!(relation.degree(), 2);
        let tables = relation.tables(&case.point).expect("a point");
        let proven = prove_batch(
            &mut session.transcript(),
            Prefix::Stage(2),
            2,
            &[tables.instance()],
        )
        .expect("a stage");
        let batch = verdict(
            &session,
            2,
            std::slice::from_ref(&case),
            &[],
            proven.proof(),
        )
        .expect("the verifier accepts");
        let at = batch.point_of(vars).expect("the instance's point");
        assert_eq!(tv_fp2s(proven.point()), batch.point, "the point");
        let claim = tables
            .claim(
                proven.instance_point(0).expect("an instance"),
                proven.instance_evals(0).expect("an instance"),
            )
            .expect("the instance's result");
        assert_eq!(
            tv_fp2s(&claim.point),
            at,
            "{vars}, {bit}: at the instance's point"
        );
        assert_eq!(
            tv_fp2(claim.value),
            mle(&case.x, at),
            "{vars}, {bit}: X's extension"
        );
    }
}

fn final_mismatch(rounds: usize) -> Result<(), tv::SumcheckError> {
    Err(tv::SumcheckError::FinalMismatch {
        round: rounds as u32,
    })
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

/// In a stage of degree 3 with elementwise products of other sizes, the rotations verify; a
/// prover whose `X`, `C` or `S` differs in one entry, and every change to a round value or an
/// evaluation, fail the final check.
#[test]
fn single_entry_changes_and_message_changes_fail_the_final_check() {
    let mut rng = SplitMix64::new(0x113_0003);
    let session = Session::random(&mut rng);
    let rotations = [Case::random(&mut rng, 7, 6), Case::random(&mut rng, 4, 0)];
    let products = vec![
        (rng.fps(1 << 5), rng.fps(1 << 5), rng.fp2s(5)),
        (rng.fps(1 << 8), rng.fps(1 << 8), rng.fp2s(8)),
    ];
    let proof = prove(&session, 3, &rotations, &products);
    assert!(
        verdict(&session, 3, &rotations, &products, &proof).is_ok(),
        "the honest stage verifies"
    );
    for (i, case) in rotations.iter().enumerate() {
        for (what, entry) in [("x", 0), ("x", 5), ("cos", 3), ("sin", 1)] {
            let mut cheating = rotations.clone();
            let values = match what {
                "x" => &mut cheating[i].x,
                "cos" => &mut cheating[i].cos,
                _ => &mut cheating[i].sin,
            };
            let len = values.len();
            values[entry % len] += Fp::ONE;
            let proof = prove(&session, 3, &cheating, &products);
            assert_eq!(
                verdict(&session, 3, &rotations, &products, &proof).map(|_| ()),
                final_mismatch(8),
                "rotation {i} on {} variables, {what}[{entry}]",
                case.vars
            );
        }
    }
    for j in 0..proof.rounds.len() {
        for m in 0..3 {
            let mut changed = proof.clone();
            changed.rounds[j][m] += delta(&mut rng);
            assert_eq!(
                verdict(&session, 3, &rotations, &products, &changed).map(|_| ()),
                final_mismatch(8),
                "round {j}, value {m}"
            );
        }
    }
    for e in 0..proof.final_evals.len() {
        let mut changed = proof.clone();
        changed.final_evals[e] += delta(&mut rng);
        assert_eq!(
            verdict(&session, 3, &rotations, &products, &changed).map(|_| ()),
            final_mismatch(8),
            "evaluation {e}"
        );
    }
}

/// `Rotate::new` and its tables refuse malformed rotations with the first failure, in the order
/// their documentation gives.
#[test]
fn malformed_rotations_are_refused_in_order() {
    let x: Vec<Fp> = (1..=8).map(Fp::from_u64_reduced).collect();
    let half = &x[..4];
    let new = |x, cos, sin, bit, vars| Rotate::<Fp2>::new(x, cos, sin, bit, vars).map(|_| ());
    assert_eq!(
        new(&x, half, half, 0, 64),
        Err(SumcheckError::RelationShape {
            vars: 64,
            summed_vars: 0
        })
    );
    assert_eq!(
        new(&x, half, half, 3, 3),
        Err(SumcheckError::PartnerBit {
            input: 0,
            bit: 3,
            vars: 3
        })
    );
    assert_eq!(
        new(&x[..1], &[], &[], 0, 0),
        Err(SumcheckError::PartnerBit {
            input: 0,
            bit: 0,
            vars: 0
        }),
        "no variable to pair along"
    );
    assert_eq!(
        new(half, half, half, 0, 3),
        Err(SumcheckError::InputLength {
            input: 0,
            len: 4,
            vars: 3
        })
    );
    assert_eq!(
        new(&x, &x, half, 0, 3),
        Err(SumcheckError::InputLength {
            input: 1,
            len: 8,
            vars: 3
        })
    );
    assert_eq!(
        new(&x, half, &x[..2], 0, 3),
        Err(SumcheckError::InputLength {
            input: 2,
            len: 2,
            vars: 3
        })
    );
    let rotate = Rotate::<Fp2>::new(&x, half, half, 1, 3).expect("a rotation");
    assert_eq!(
        rotate.tables(&[Fp2::ONE; 2]).map(|_| ()),
        Err(SumcheckError::PointLength {
            len: 2,
            expected: 3
        })
    );
    let tables = rotate.tables(&[Fp2::ONE; 3]).expect("tables");
    assert_eq!(
        tables.claim(&[Fp2::ONE; 4], &[Fp2::ONE]).map(|_| ()),
        Err(SumcheckError::PointLength {
            len: 4,
            expected: 3
        })
    );
    assert_eq!(
        tables.claim(&[Fp2::ONE; 3], &[]).map(|_| ()),
        Err(SumcheckError::EvaluationCount {
            len: 0,
            expected: 1
        })
    );
}
