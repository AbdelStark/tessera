// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0072's recomposition (`Recomposition`) through the independent verifier (DEP-2; RFC-0072
//! 'The bit zero-checks', RFC-0066 '`BitDecompose`'). The verifier's side is written here from
//! the RFCs, in `tv-field`: the claim is 0, and the relation at the instance's point
//! `r' = r'_i ‖ r'_y` is
//!
//! ```text
//! eq(z_y, r'_y)·[2^{−k}·x̃(r'_y) − W̃_b(r'_i)·Z_b(r'_i)·b̃(r'_{i, ≥ k − k_b} ‖ r'_y)
//!                − 2^{low}·W̃_h(r'_i)·Z_h(r'_i)·h̃(r'_{i, ≥ k − k_h} ‖ r'_y)]
//! ```
//!
//! with `Z_o(r) = Π_{t < k − k_o}(1 − r_t)` and the weights' closed form
//! `W̃_o(r) = Z_o(r)·Σ_{j < n_o} 2^j·eq(s, j)`, `s` the high `k_o` bit coordinates.
//!
//! - An honest decomposition verifies on every width RFC-0066's sums and bytes take, `k_h > k_b`
//!   and a `low` that is not a power of two among them, with `x` an operand and public; its
//!   claims are `x`'s, `b`'s and `h`'s extensions at `r'_y` and at their high bit coordinates.
//! - A wrong bit, or a wrong `x`, fails the final check; a non-Boolean bit that recomposes
//!   passes it, booleanity's to catch.
//! - Its terms have three factors: a batch of degree 2 is refused.
//! - `new`, `tables` and `claims` refuse what is not a decomposition, its point or its
//!   evaluations.

mod common;

use common::elementwise::extension;
use common::{Fp, Fp2, Session, SplitMix64, TvFp, TvFp2, tv_fp, tv_fp2s, tv_prefix, tv_proof};
use tessera_field::{Field, PrimeField};
use tessera_sumcheck::{
    Input, Prefix, Protocol, Recomposition, SumcheckError, SumcheckProof, prove_batch,
};
use tv_core::sumcheck as tv;

/// `2^j` in the verifier's field.
fn two_to(j: u64) -> TvFp2 {
    TvFp2::from(TvFp::from_canonical(2).expect("2").pow(j))
}

/// `⌈log2 n⌉`.
fn bit_vars(bits: u8) -> usize {
    u32::from(bits).next_power_of_two().trailing_zeros() as usize
}

/// A decomposition of `2^rows` integers into `low` and `high` bits.
#[derive(Clone)]
struct Case {
    low: u8,
    high: u8,
    rows: usize,
    public: bool,
    x: Vec<Fp>,
    b: Vec<Fp>,
    h: Vec<Fp>,
    z: Vec<Fp2>,
}

impl Case {
    /// Integers below `2^{low + high}` and their bits, padded slots 0, and a random `z_y`.
    fn honest(rng: &mut SplitMix64, low: u8, high: u8, rows: usize, public: bool) -> Self {
        let width = u32::from(low) + u32::from(high);
        let values: Vec<u64> = (0..1usize << rows)
            .map(|_| rng.next_u64() & ((1u64 << width) - 1))
            .collect();
        let table = |count: u8, shift: u32| -> Vec<Fp> {
            let slots = 1usize << bit_vars(count);
            values
                .iter()
                .flat_map(|&v| {
                    (0..slots).map(move |j| {
                        if j < usize::from(count) {
                            Fp::from_u64_reduced((v >> (shift + j as u32)) & 1)
                        } else {
                            Fp::ZERO
                        }
                    })
                })
                .collect()
        };
        Self {
            low,
            high,
            rows,
            public,
            x: values.iter().map(|&v| Fp::from_u64_reduced(v)).collect(),
            b: table(low, 0),
            h: if high > 0 {
                table(high, u32::from(low))
            } else {
                Vec::new()
            },
            z: rng.fp2s(rows),
        }
    }

    fn relation(&self) -> Recomposition<'_, Fp2> {
        let x = if self.public {
            Input::Public(&self.x)
        } else {
            Input::Operand(&self.x)
        };
        let h = (self.high > 0).then_some(self.h.as_slice());
        Recomposition::new(x, &self.b, h, self.low, self.high, self.rows).expect("a decomposition")
    }

    fn k(&self) -> usize {
        bit_vars(self.low).max(bit_vars(self.high))
    }

    fn sent(&self) -> usize {
        usize::from(!self.public) + 1 + usize::from(self.high > 0)
    }

    /// The proof, and the prover's point.
    fn prove(
        &self,
        session: &Session,
        degree: usize,
    ) -> Result<(SumcheckProof<Fp2>, Vec<Fp2>), SumcheckError> {
        let tables = self.relation().tables(&self.z)?;
        let proven = prove_batch(
            &mut session.transcript(),
            Protocol::One,
            Prefix::Stage(1),
            degree,
            &[tables.instance()],
        )?;
        let point = proven.point().to_vec();
        Ok((proven.into_proof(), point))
    }

    /// The relation at `at` from its evaluations, every public factor by its closed form.
    fn at(&self, at: &[TvFp2], evals: &[TvFp2]) -> Option<TvFp2> {
        let k = self.k();
        let (r_i, r_y) = at.split_at(k);
        let mut next = 0;
        let x = if self.public {
            let x: Vec<TvFp2> = self.x.iter().map(|&v| TvFp2::from(tv_fp(v))).collect();
            extension(&x, r_y)
        } else {
            next += 1;
            *evals.first()?
        };
        let mut residual = two_to(k as u64).inverse()? * x;
        residual -=
            weights(r_i, self.low) * zero(r_i, k - bit_vars(self.low)) * *evals.get(next)?;
        if self.high > 0 {
            residual -= two_to(u64::from(self.low))
                * weights(r_i, self.high)
                * zero(r_i, k - bit_vars(self.high))
                * *evals.get(next + 1)?;
        }
        Some(tv::eq(&tv_fp2s(&self.z), r_y)? * residual)
    }

    fn verdict(
        &self,
        session: &Session,
        proof: &SumcheckProof<Fp2>,
    ) -> Result<tv::Batch, tv::SumcheckError> {
        let instance = tv::Instance {
            claim: TvFp2::ZERO,
            vars: self.rows + self.k(),
            evals: self.sent(),
        };
        tv::verify_batch(
            &mut session.tv_transcript(),
            tv_prefix(Prefix::Stage(1)),
            3,
            &[instance],
            &tv_proof(proof),
            |_, at, evals| self.at(at, evals),
        )
    }
}

/// `Π_{t < pad}(1 − r_t)`.
fn zero(r: &[TvFp2], pad: usize) -> TvFp2 {
    r[..pad]
        .iter()
        .fold(TvFp2::ONE, |p, &x| p * (TvFp2::ONE - x))
}

/// `W̃(r) = Z(r)·Σ_{j < bits} 2^j·eq(s, j)` on `k = r.len()` bit coordinates.
fn weights(r: &[TvFp2], bits: u8) -> TvFp2 {
    let k_o = bit_vars(bits);
    let pad = r.len() - k_o;
    let s = &r[pad..];
    let sum = (0..u64::from(bits)).fold(TvFp2::ZERO, |sum, j| {
        let e = s.iter().enumerate().fold(TvFp2::ONE, |e, (t, &x)| {
            e * if (j >> t) & 1 == 1 { x } else { TvFp2::ONE - x }
        });
        sum + two_to(j) * e
    });
    zero(r, pad) * sum
}

/// RFC-0066's widths: bytes, a word's sums with carries of 1 to 3 bits, a `low` that is not a
/// power of two, `k_h > k_b`, and the widest.
const WIDTHS: [(u8, u8); 9] = [
    (1, 0),
    (8, 0),
    (5, 0),
    (32, 1),
    (32, 2),
    (32, 3),
    (2, 8),
    (3, 5),
    (61, 1),
];

#[test]
fn an_honest_decomposition_verifies_and_claims_its_tensors() {
    let mut rng = SplitMix64::new(0x0526_5001);
    let session = Session::random(&mut rng);
    for (low, high) in WIDTHS {
        for rows in [0, 1, 3] {
            for public in [false, true] {
                let name = format!("({low}, {high}), {rows} rows, public x {public}");
                let case = Case::honest(&mut rng, low, high, rows, public);
                let relation = case.relation();
                assert_eq!(relation.degree(), 3, "{name}");
                assert_eq!(relation.vars(), rows + case.k(), "{name}");
                let (proof, point) = case.prove(&session, 3).expect("a stage");
                assert_eq!(proof.final_evals.len(), case.sent(), "{name}");
                let batch = case
                    .verdict(&session, &proof)
                    .unwrap_or_else(|e| panic!("{name}: {e:?}"));
                assert_eq!(batch.point, tv_fp2s(&point), "{name}: one point");
                // Its claims: x at r'_y, b and h at their high bit coordinates with r'_y.
                let tables = relation.tables(&case.z).expect("a point");
                let claims = tables
                    .claims(&point, &proof.final_evals)
                    .expect("its claims");
                let k = case.k();
                let tv = |values: &[Fp]| -> Vec<TvFp2> {
                    values.iter().map(|&v| TvFp2::from(tv_fp(v))).collect()
                };
                let at = tv_fp2s(&point);
                let mut expected = Vec::new();
                if !public {
                    expected.push(extension(&tv(&case.x), &at[k..]));
                }
                expected.push(extension(&tv(&case.b), &at[k - bit_vars(low)..]));
                if high > 0 {
                    expected.push(extension(&tv(&case.h), &at[k - bit_vars(high)..]));
                }
                let values: Vec<TvFp2> = claims.iter().map(|c| tv_fp2s(&[c.value])[0]).collect();
                assert_eq!(values, expected, "{name}: the claims' values");
            }
        }
    }
}

#[test]
fn a_wrong_bit_or_input_fails_the_final_check_and_a_non_boolean_one_does_not() {
    let mut rng = SplitMix64::new(0x0526_5002);
    let session = Session::random(&mut rng);
    for (low, high) in [(8, 0), (32, 3), (2, 8), (3, 5)] {
        let honest = Case::honest(&mut rng, low, high, 2, false);
        let k = honest.k();
        let final_check = Err(tv::SumcheckError::FinalMismatch {
            round: 2 + k as u32,
        });
        let mut wrong = honest.clone();
        wrong.b[1] = Fp::ONE - wrong.b[1];
        let (proof, _) = wrong.prove(&session, 3).expect("a stage");
        assert_eq!(
            wrong.verdict(&session, &proof).map(|_| ()),
            final_check,
            "({low}, {high}): b"
        );
        if high > 0 {
            let mut wrong = honest.clone();
            let slots = 1usize << bit_vars(high);
            wrong.h[slots] = Fp::ONE - wrong.h[slots];
            let (proof, _) = wrong.prove(&session, 3).expect("a stage");
            assert_eq!(
                wrong.verdict(&session, &proof).map(|_| ()),
                final_check,
                "h"
            );
        }
        let mut wrong = honest.clone();
        wrong.x[3] += Fp::ONE;
        let (proof, _) = wrong.prove(&session, 3).expect("a stage");
        assert_eq!(
            wrong.verdict(&session, &proof).map(|_| ()),
            final_check,
            "x"
        );
        // Bit 1 of 2·b_0 + b_1 traded for b_0: recomposition holds, booleanity's to catch.
        let mut traded = honest.clone();
        let slots = 1usize << bit_vars(low);
        let (b0, b1) = (traded.b[slots], traded.b[slots + 1]);
        traded.b[slots] = b0 + b1 + b1;
        traded.b[slots + 1] = Fp::ZERO;
        let (proof, _) = traded.prove(&session, 3).expect("a stage");
        assert!(
            traded.verdict(&session, &proof).is_ok(),
            "({low}, {high}): traded"
        );
    }
}

#[test]
fn its_terms_have_three_factors_so_a_batch_of_degree_2_is_refused() {
    let mut rng = SplitMix64::new(0x0526_5003);
    let session = Session::random(&mut rng);
    let case = Case::honest(&mut rng, 8, 0, 1, false);
    assert!(matches!(
        case.prove(&session, 2),
        Err(SumcheckError::TermFactors {
            factors: 3,
            degree: 2,
            ..
        })
    ));
}

#[test]
fn what_is_not_a_decomposition_is_refused() {
    let x = vec![Fp::ZERO; 2];
    let b = vec![Fp::ZERO; 2 * 8];
    let h = vec![Fp::ZERO; 2 * 4];
    let new = |x: Input<'_, Fp2>, b: &[Fp], h: Option<&[Fp]>, low: u8, high: u8| {
        Recomposition::<Fp2>::new(x, b, h, low, high, 1).map(|_| ())
    };
    let widths = |low, high| Err(SumcheckError::BitWidths { low, high });
    assert_eq!(new(Input::Operand(&x), &b, None, 0, 0), widths(0, 0));
    assert_eq!(
        new(Input::Operand(&x), &b, Some(&h), 32, 31),
        widths(32, 31)
    );
    assert_eq!(new(Input::Operand(&x), &b, Some(&h), 8, 0), widths(8, 0));
    assert_eq!(new(Input::Operand(&x), &b, None, 8, 3), widths(8, 3));
    let length = |input, len| {
        Err(SumcheckError::InputLength {
            input,
            len,
            vars: 4,
        })
    };
    assert_eq!(
        new(Input::Operand(&x[..1]), &b, Some(&h), 8, 3),
        length(0, 1)
    );
    assert_eq!(
        new(Input::Operand(&x), &b[..8], Some(&h), 8, 3),
        length(1, 8)
    );
    assert_eq!(
        new(Input::Operand(&x), &b, Some(&h[..4]), 8, 3),
        length(2, 4)
    );
    let spread = Input::Broadcast {
        operand: &x,
        low_vars: 0,
    };
    assert_eq!(new(spread, &b, Some(&h), 8, 3), length(0, 2));
    let relation = Recomposition::<Fp2>::new(Input::Operand(&x), &b, Some(&h), 8, 3, 1)
        .expect("a decomposition");
    assert!(matches!(
        relation.tables(&[Fp2::ONE, Fp2::ONE]),
        Err(SumcheckError::PointLength {
            len: 2,
            expected: 1
        })
    ));
    let tables = relation.tables(&[Fp2::ONE]).expect("a point");
    assert_eq!(
        tables.claims(&[Fp2::ONE; 3], &[Fp2::ONE; 3]).map(|_| ()),
        Err(SumcheckError::PointLength {
            len: 3,
            expected: 4
        })
    );
    assert_eq!(
        tables.claims(&[Fp2::ONE; 4], &[Fp2::ONE; 2]).map(|_| ()),
        Err(SumcheckError::EvaluationCount {
            len: 2,
            expected: 3
        })
    );
}
