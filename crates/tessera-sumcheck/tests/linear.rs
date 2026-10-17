// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `Linear` claim propagation (RFC-0010 'Relations'), against direct evaluation: every claim a
//! chain of linear relations propagates to is evaluated here from the tables, by enumeration of
//! the hypercube in `tv-field`, and must equal the claim's value.
//!
//! - A residual stream over 8 layers, rescaled by powers of two, `res_{ℓ+1} = 2^{t_ℓ}·res_ℓ + u_ℓ`,
//!   propagates a claim on `res_8` to `res_0` and the 8 updates at its point, each coefficient
//!   the product of the rescalings after it, and one entry changed in any of them breaks it.
//! - A row's maximum against its row (`k = c − x`), a row sum (`s = Σ_i e`) and an offset by
//!   public constants place their claims as RFC-0010 and RFC-0030 state, and a public input moves
//!   to the value.

mod common;

use common::forests::mle;
use common::{Fp, Fp2, SplitMix64, TvFp2, tv_fp2, tv_fp2s};
use tessera_field::{Field, PrimeField};
use tessera_sumcheck::{Claim, Linear, LinearClaim, LinearInput, Placement, SumcheckError};

/// The tensors of the tests, as the schedule would name them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Poly {
    Res(usize),
    Update(usize),
    Max,
    Scores,
    Keys,
    Exps,
    Sums,
    Offsets,
    Shifted,
}

fn input(coefficient: Fp2, poly: Poly, placement: Placement) -> LinearInput<Poly, Fp2> {
    LinearInput {
        coefficient,
        poly,
        placement,
    }
}

fn fp2(x: u64) -> Fp2 {
    Fp2::from(Fp::from_u64_reduced(x))
}

/// `Σ_t c_t·P̃_t(r_t)` with each table's extension computed by enumeration in `tv-field`.
fn direct(claim: &LinearClaim<Poly, Fp2>, table: impl Fn(Poly) -> Vec<Fp>) -> TvFp2 {
    claim.terms.iter().fold(TvFp2::ZERO, |sum, term| {
        sum + tv_fp2(term.coefficient) * mle(&table(term.poly), &tv_fp2s(&term.point))
    })
}

#[test]
fn a_residual_chain_over_eight_layers_propagates_to_its_inputs() {
    let mut rng = SplitMix64::new(0x99_0101);
    let vars = 10;
    let shifts: Vec<u64> = (0..8).map(|_| rng.below(3)).collect();
    assert!(
        shifts[1..].iter().any(|&t| t > 0),
        "a substituted term's coefficient is not 1"
    );
    let res_0 = rng.fps(1 << vars);
    let updates: Vec<Vec<Fp>> = (0..8).map(|_| rng.fps(1 << vars)).collect();
    // res_{ℓ+1} = 2^{t_ℓ}·res_ℓ + u_ℓ, entry by entry.
    let mut res = vec![res_0.clone()];
    for (layer, update) in updates.iter().enumerate() {
        let scale = Fp::from_u64_reduced(1 << shifts[layer]);
        let next = res[layer]
            .iter()
            .zip(update)
            .map(|(&r, &u)| scale * r + u)
            .collect();
        res.push(next);
    }
    // The claim on res_8: the prover's value, which the verifier's field agrees with.
    let point = rng.fp2s(vars);
    let value = mle(&res[8], &tv_fp2s(&point));
    let mut claim = LinearClaim::single(
        Poly::Res(8),
        Claim {
            point: point.clone(),
            value: tessera_poly::evaluate_base(&res[8], &point).expect("a table at its point"),
        },
    );
    assert_eq!(tv_fp2(claim.value), value);
    for layer in (0..8).rev() {
        let step = Linear::new(
            vars,
            vec![
                input(fp2(1 << shifts[layer]), Poly::Res(layer), Placement::Same),
                input(Fp2::ONE, Poly::Update(layer), Placement::Same),
            ],
        )
        .expect("a relation");
        claim
            .substitute(&Poly::Res(layer + 1), &step)
            .expect("points of the target's coordinates");
    }
    // Each substitution takes its target's place: the chain ends on res_0, then the updates in
    // layer order, all at the claim's point.
    let polys: Vec<Poly> = claim.terms.iter().map(|term| term.poly).collect();
    let expected: Vec<Poly> = [Poly::Res(0)]
        .into_iter()
        .chain((0..8).map(Poly::Update))
        .collect();
    assert_eq!(polys, expected);
    assert!(claim.terms.iter().all(|term| term.point == point));
    // res_0 carries every rescaling, u_ℓ those of the layers after it.
    let after = |layer: usize| fp2(1 << shifts[layer..].iter().sum::<u64>());
    let coefficients: Vec<Fp2> = claim.terms.iter().map(|term| term.coefficient).collect();
    let expected_coefficients: Vec<Fp2> = [after(0)]
        .into_iter()
        .chain((0..8).map(|layer| after(layer + 1)))
        .collect();
    assert_eq!(coefficients, expected_coefficients);
    let table = |poly: Poly| match poly {
        Poly::Res(0) => res_0.clone(),
        Poly::Update(layer) => updates[layer].clone(),
        _ => unreachable!("the chain's inputs"),
    };
    assert_eq!(direct(&claim, table), value, "direct evaluation");
    // The prover's own check of the combination agrees.
    let combination = claim.combination(|&poly, point| {
        tessera_poly::evaluate_base(&table(poly), point).expect("a table at its point")
    });
    assert_eq!(combination, claim.value);
    // One entry changed in any input breaks it.
    for changed in expected {
        for entry in [0, 1 << (vars - 1), (1 << vars) - 1] {
            let wrong = |poly: Poly| {
                let mut values = table(poly);
                if poly == changed {
                    values[entry] += Fp::ONE;
                }
                values
            };
            assert_ne!(direct(&claim, wrong), value, "{changed:?}, entry {entry}");
        }
    }
}

/// `k[q, i] = c[q] − x[q, i]` places `c`'s claim at the point's row coordinates; `s[q] =
/// Σ_i e[q, i]` places `e`'s at `(½)^{n_i} ‖ r_q` with coefficient `2^{n_i}`; and an offset by
/// public per-key constants, `y[q, i] = k[q, i] + b[i]`, leaves a public term that moves to the
/// value.
#[test]
fn broadcasts_sums_and_offsets_place_their_claims() {
    let mut rng = SplitMix64::new(0x99_0102);
    let (row_vars, key_vars) = (4, 5);
    let vars = row_vars + key_vars;
    let max = rng.fps(1 << row_vars);
    let scores = rng.fps(1 << vars);
    let exps = rng.fps(1 << vars);
    let offsets = rng.fps(1 << key_vars);
    let keys: Vec<Fp> = (0..1 << vars)
        .map(|x| max[x >> key_vars] - scores[x])
        .collect();
    let shifted: Vec<Fp> = (0..1 << vars)
        .map(|x| keys[x] + offsets[x % (1 << key_vars)])
        .collect();
    let sums: Vec<Fp> = exps
        .chunks_exact(1 << key_vars)
        .map(|row| row.iter().fold(Fp::ZERO, |sum, &e| sum + e))
        .collect();
    let table = |poly: Poly| match poly {
        Poly::Max => max.clone(),
        Poly::Scores => scores.clone(),
        Poly::Keys => keys.clone(),
        Poly::Exps => exps.clone(),
        Poly::Sums => sums.clone(),
        Poly::Offsets => offsets.clone(),
        Poly::Shifted => shifted.clone(),
        _ => unreachable!("the test's tensors"),
    };
    let keys_of = Linear::new(
        vars,
        vec![
            input(
                Fp2::ONE,
                Poly::Max,
                Placement::Broadcast {
                    low_vars: key_vars,
                    vars: row_vars,
                },
            ),
            input(-Fp2::ONE, Poly::Scores, Placement::Same),
        ],
    )
    .expect("a relation");
    let shifted_of = Linear::new(
        vars,
        vec![
            input(Fp2::ONE, Poly::Keys, Placement::Same),
            input(
                Fp2::ONE,
                Poly::Offsets,
                Placement::Broadcast {
                    low_vars: 0,
                    vars: key_vars,
                },
            ),
        ],
    )
    .expect("a relation");
    let sums_of = Linear::new(
        row_vars,
        vec![input(
            Fp2::ONE,
            Poly::Exps,
            Placement::Sum { vars: key_vars },
        )],
    )
    .expect("a relation");

    // The offset, then the keys: a claim on y ends on c, x and the public b.
    let point = rng.fp2s(vars);
    let value = tessera_poly::evaluate_base(&shifted, &point).expect("at its point");
    let mut claim = LinearClaim::single(
        Poly::Shifted,
        Claim {
            point: point.clone(),
            value,
        },
    );
    claim
        .substitute(&Poly::Shifted, &shifted_of)
        .expect("a point");
    claim.substitute(&Poly::Keys, &keys_of).expect("a point");
    let placed: Vec<(Poly, Vec<Fp2>)> = claim
        .terms
        .iter()
        .map(|term| (term.poly, term.point.clone()))
        .collect();
    assert_eq!(
        placed,
        [
            (Poly::Max, point[key_vars..].to_vec()),
            (Poly::Scores, point.clone()),
            (Poly::Offsets, point[..key_vars].to_vec()),
        ]
    );
    assert_eq!(direct(&claim, table), tv_fp2(value));
    // The public offsets' term moves to the value; what remains holds on c and x alone.
    claim.evaluate_public(|&poly, point| {
        (poly == Poly::Offsets)
            .then(|| tessera_poly::evaluate_base(&offsets, point).expect("at its point"))
    });
    assert_eq!(claim.terms.len(), 2);
    assert_eq!(direct(&claim, table), tv_fp2(claim.value));

    // The row sum: e at (½, …, ½) ‖ r_q, coefficient 2^{n_i}.
    let rows = rng.fp2s(row_vars);
    let value = tessera_poly::evaluate_base(&sums, &rows).expect("at its point");
    let mut claim = LinearClaim::single(
        Poly::Sums,
        Claim {
            point: rows.clone(),
            value,
        },
    );
    claim.substitute(&Poly::Sums, &sums_of).expect("a point");
    let half = Fp2::from(Fp::from_u64_reduced(2))
        .inverse()
        .expect("2 is invertible");
    assert_eq!(claim.terms.len(), 1);
    assert_eq!(claim.terms[0].coefficient, fp2(1 << key_vars));
    assert_eq!(
        claim.terms[0].point,
        [vec![half; key_vars], rows.clone()].concat()
    );
    assert_eq!(direct(&claim, table), tv_fp2(value));
    // One entry of e changed breaks it.
    let wrong = |poly: Poly| {
        let mut values = table(poly);
        values[17] += Fp::ONE;
        values
    };
    assert_ne!(direct(&claim, wrong), tv_fp2(value));
}

/// `Linear::new` refuses inputs outside the target's variables, `terms` a point of another
/// length, and `substitute` leaves the claim as it was when one of its terms on the target has
/// such a point.
#[test]
fn misplaced_inputs_and_points_are_refused() {
    let broadcast = |low_vars, vars| {
        Linear::<Poly, Fp2>::new(
            4,
            vec![
                input(Fp2::ONE, Poly::Max, Placement::Same),
                input(
                    Fp2::ONE,
                    Poly::Scores,
                    Placement::Broadcast { low_vars, vars },
                ),
            ],
        )
        .map(|_| ())
    };
    assert_eq!(broadcast(1, 3), Ok(()));
    assert_eq!(
        broadcast(2, 3),
        Err(SumcheckError::InputPlacement {
            input: 1,
            target_vars: 4
        })
    );
    assert_eq!(
        broadcast(usize::MAX, 1),
        Err(SumcheckError::InputPlacement {
            input: 1,
            target_vars: 4
        })
    );
    assert_eq!(
        Linear::<Poly, Fp2>::new(
            4,
            vec![input(Fp2::ONE, Poly::Exps, Placement::Sum { vars: 60 })]
        )
        .map(|_| ()),
        Err(SumcheckError::InputPlacement {
            input: 0,
            target_vars: 4
        })
    );
    let relation =
        Linear::new(4, vec![input(Fp2::ONE, Poly::Max, Placement::Same)]).expect("a relation");
    assert_eq!(
        relation.terms(&[Fp2::ONE; 3]).map(|_| ()),
        Err(SumcheckError::PointLength {
            len: 3,
            expected: 4
        })
    );
    let mut claim = LinearClaim {
        terms: vec![
            tessera_sumcheck::LinearTerm {
                coefficient: Fp2::ONE,
                poly: Poly::Keys,
                point: vec![Fp2::ONE; 4],
            },
            tessera_sumcheck::LinearTerm {
                coefficient: Fp2::ONE,
                poly: Poly::Keys,
                point: vec![Fp2::ONE; 5],
            },
        ],
        value: Fp2::ZERO,
    };
    let before = claim.clone();
    assert_eq!(
        claim.substitute(&Poly::Keys, &relation),
        Err(SumcheckError::PointLength {
            len: 5,
            expected: 4
        })
    );
    assert_eq!(claim, before, "unchanged");
}
