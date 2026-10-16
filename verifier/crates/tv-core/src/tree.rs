// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The levels of the leaf stage, where product forests and LogUp fraction trees run together
//! (RFC-0010 'Tree levels'). The crate documentation is their specification ('Tree levels').
//!
//! A tree of height `h` has levels `0` (the root, one entry) to `h` (the leaves, `2^h`
//! entries), and the pair `2x, 2x + 1` of level `ℓ + 1` combines into entry `x` of level `ℓ`:
//! `V_ℓ[x] = V_{ℓ+1}[2x]·V_{ℓ+1}[2x + 1]` in a product tree, and
//! `(n, d) = (n_L·d_R + n_R·d_L, d_L·d_R)` in a fraction tree (RFC-0008 'Fraction tree'). As
//! multilinear polynomials, with the pair bit the lowest variable of level `ℓ + 1`,
//!
//! ```text
//! Ṽ_ℓ(r)          = Σ_x eq(r, x)·Ṽ_{ℓ+1}(0 ‖ x)·Ṽ_{ℓ+1}(1 ‖ x)
//! Ñ_ℓ(r) + λ·D̃_ℓ(r) = Σ_x eq(r, x)·(N(0 ‖ x)·D(1 ‖ x) + N(1 ‖ x)·D(0 ‖ x) + λ·D(0 ‖ x)·D(1 ‖ x))
//! ```
//!
//! the second for every `λ`, `N` and `D` being level `ℓ + 1`'s numerators and denominators. One
//! batched sumcheck of degree 3 per level of the batch reduces every instance's claims on its
//! level to claims on the next, from the roots to the leaves.
//!
//! Product trees of one height are stacked into a forest of `2^s` trees, the tree index in the
//! high `s` variables; a single tree is a forest with `s = 0`. A fraction tree is always single.
//! Instances of different heights share the batch's levels: with `H` the tallest instance's
//! height, an instance of height `h` takes its root level at level `H − h` of the batch, and a
//! forest has `s + ℓ − (H − h)` variables at level `ℓ`. The caller, which holds the schedule,
//! passes the instances present at each level in schedule order:
//! - the claims the previous level returned;
//! - the root claims of the forests that join, each `Σ_y v_y·eq(r, y)` from its public root
//!   values `v_y`, at the root point `r` that the caller squeezes under `tree/{ℓ}/root/{c}`
//!   before the level's `γ`;
//! - the root claims of the fraction trees that join, from
//!   [`check_roots`](crate::logup::check_roots), on no variable.

use tv_field::Fp2;
use tv_hash::transcript::Transcript;

use crate::fiat_shamir::{self, FiatShamir, Step};
use crate::sumcheck::{Instance, Prefix, Shape, SumcheckError, SumcheckProof, eq, finish_batch};

/// The degree of every tree level's sumcheck: `eq(r, x)·Ṽ(0 ‖ x)·Ṽ(1 ‖ x)` and
/// `eq(r, x)·N(0 ‖ x)·D(1 ‖ x)` are of degree 3 in each variable (RFC-0010 'Relations',
/// `ProductTreeLevel` and `FractionTreeLevel`).
pub const LEVEL_DEGREE: usize = 3;

/// The degree of a product-tree level's sumcheck, [`LEVEL_DEGREE`]: `eq(r, x)·Ṽ(0 ‖ x)·Ṽ(1 ‖ x)`
/// is of degree 3 in each variable (RFC-0010 'Relations', `ProductTreeLevel`).
pub const PRODUCT_LEVEL_DEGREE: usize = LEVEL_DEGREE;

/// A claim on one level of a product tree or forest: `Ṽ(point) = value`, over `point.len()`
/// variables.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TreeClaim {
    /// The point, its low variables first.
    pub point: Vec<Fp2>,
    /// The claimed value of the level's multilinear polynomial there.
    pub value: Fp2,
}

/// The two claims on one level of a LogUp fraction tree, at one point, over `point.len()`
/// variables: `Ñ(point) = numerator` on the level's numerators and `D̃(point) = denominator` on
/// its denominators.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FractionClaim {
    /// The point, its low variables first.
    pub point: Vec<Fp2>,
    /// The claimed value of the multilinear polynomial of the level's numerators there.
    pub numerator: Fp2,
    /// The claimed value of the multilinear polynomial of the level's denominators there.
    pub denominator: Fp2,
}

/// The claims an instance of a tree level holds on its current level: a product forest's or a
/// fraction tree's.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LevelClaim {
    /// A product forest's claim; a single product tree is a forest of one tree.
    Product(TreeClaim),
    /// A LogUp fraction tree's claims.
    Fraction(FractionClaim),
}

/// A product-tree level that passed every check of [`verify_product_level`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[must_use = "a level that passed has proven only its next claims, which the next level or the stage consuming the leaf claims must check"]
pub struct ProductLevel {
    /// The batching challenge `γ`.
    pub gamma: Fp2,
    /// The level's challenges in round order: its point, the last `n_i` of which are instance
    /// `i`'s.
    pub point: Vec<Fp2>,
    /// The line challenge `ρ`.
    pub rho: Fp2,
    /// Each instance's claim on its next level, in schedule order: `Ṽ(ρ ‖ r'_i)`.
    pub claims: Vec<TreeClaim>,
}

/// A level that passed every check of [`verify_level`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[must_use = "a level that passed has proven only its next claims, which the next level, the stage consuming a forest's leaf claims or the LogUp leaf checks must check"]
pub struct TreeLevel {
    /// The batching challenge `γ`.
    pub gamma: Fp2,
    /// The challenge `λ` that combines each fraction tree's two claims, squeezed after `γ`, or
    /// `None` at a level of forests alone, which squeezes none.
    pub lambda: Option<Fp2>,
    /// The level's challenges in round order: its point, the last `n_i` of which are instance
    /// `i`'s.
    pub point: Vec<Fp2>,
    /// The line challenge `ρ`.
    pub rho: Fp2,
    /// Each instance's claims on its next level, in schedule order, at `ρ ‖ r'_i`.
    pub claims: Vec<LevelClaim>,
}

/// Checks level `level` of the leaf stage for the instances present there, each `claims[i]` the
/// claims of a product forest or of a LogUp fraction tree on its current level, in schedule
/// order, and returns their claims on the next level, in the same order, with the level's
/// challenges.
///
/// The checks, in order, are those of the crate documentation ('Tree levels'): a batched
/// sumcheck of degree 3 under `tree/{level}`, its `γ` followed by `λ` when a fraction tree is
/// present, the claims `v_i` and `p_i + λ·q_i` front-padded, and the final evaluations `u_i, w_i`
/// of a forest and `N_0, N_1, D_0, D_1` of a fraction tree, which must satisfy
/// `Σ_i γ^i · eq(r_i, r'_i) · g_i = claim` with `g_i = u_i·w_i` or
/// `N_0·D_1 + N_1·D_0 + λ·D_0·D_1`; then the line challenge `ρ`. With `r'_i` instance `i`'s
/// point in the batch, a forest's next claim is `Ṽ(ρ ‖ r'_i) = u_i + ρ·(w_i − u_i)`, and a
/// fraction tree's are `Ñ(ρ ‖ r'_i) = N_0 + ρ·(N_1 − N_0)` and
/// `D̃(ρ ‖ r'_i) = D_0 + ρ·(D_1 − D_0)`.
///
/// After an instance's last level, its claims are its leaf claims: a forest's go to the stage
/// that consumes them, a fraction tree's to [`check_leaves`](crate::logup::check_leaves).
pub fn verify_level(
    transcript: &mut Transcript,
    level: u32,
    claims: &[LevelClaim],
    proof: &SumcheckProof,
) -> Result<TreeLevel, SumcheckError> {
    verify_level_with(transcript, level, claims, proof)
}

/// [`verify_level`] on any transcript.
pub(crate) fn verify_level_with(
    transcript: &mut impl FiatShamir,
    level: u32,
    claims: &[LevelClaim],
    proof: &SumcheckProof,
) -> Result<TreeLevel, SumcheckError> {
    let passed = level_with(transcript, level, claims, proof)?;
    Ok(TreeLevel {
        gamma: passed.gamma,
        lambda: passed.lambda,
        point: passed.point,
        rho: passed.rho,
        claims: passed.claims,
    })
}

/// Checks level `level` of the leaf stage for the product-tree instances present there, each
/// `claims[i]` a claim on its current level, in schedule order, and returns their claims on the
/// next level, in the same order, with the level's challenges. The checks, in order, are those
/// of the crate documentation ('Tree levels') for a level of forests alone: a batched sumcheck of
/// degree 3 under `tree/{level}` with no `λ`, whose final evaluations are `u_i, w_i` for each
/// instance and must satisfy `Σ_i γ^i · eq(r_i, r'_i) · u_i · w_i = claim`, then the line
/// challenge `ρ`. Instance `i`'s next claim is `Ṽ(ρ ‖ r'_i) = u_i + ρ·(w_i − u_i)`, with `r'_i`
/// its point in the batch.
///
/// After a tree's last level, its claim is a leaf claim, which the next stage checks. On the same
/// claims, this check and [`verify_level`] are one: the same errors, the same transcript and the
/// same next claims.
pub fn verify_product_level(
    transcript: &mut Transcript,
    level: u32,
    claims: &[TreeClaim],
    proof: &SumcheckProof,
) -> Result<ProductLevel, SumcheckError> {
    verify_product_level_with(transcript, level, claims, proof)
}

/// [`verify_product_level`] on any transcript.
pub(crate) fn verify_product_level_with(
    transcript: &mut impl FiatShamir,
    level: u32,
    claims: &[TreeClaim],
    proof: &SumcheckProof,
) -> Result<ProductLevel, SumcheckError> {
    let passed = level_with(transcript, level, claims, proof)?;
    Ok(ProductLevel {
        gamma: passed.gamma,
        point: passed.point,
        rho: passed.rho,
        claims: passed.claims,
    })
}

/// What a level needs of an instance's claims, for each kind of instance.
trait Entering: Sized {
    /// Its point `r_i`.
    fn point(&self) -> &[Fp2];
    /// `k_i`, the final evaluations its relation lists: 2 for a forest, 4 for a fraction tree.
    fn evals(&self) -> usize;
    /// Whether it is a fraction tree's, which makes its level squeeze `λ`.
    fn is_fraction(&self) -> bool;
    /// `c_i`, the claim it enters the level with: `v_i`, or `p_i + λ·q_i`.
    fn entering(&self, lambda: Fp2) -> Fp2;
    /// `g_i` without its factor `eq(r_i, r'_i)`, from its `k_i` evaluations, or `None` for
    /// another number of them.
    fn relation(&self, lambda: Fp2, evals: &[Fp2]) -> Option<Fp2>;
    /// Its claims on the next level at `point`, `ρ ‖ r'_i`, from its evaluations and `ρ`.
    fn next(&self, point: Vec<Fp2>, rho: Fp2, evals: &[Fp2]) -> Option<Self>;
}

/// `low + ρ·(high − low)`: the line through a pair's values, `low` at 0 and `high` at 1, at `ρ`.
fn line(low: Fp2, high: Fp2, rho: Fp2) -> Fp2 {
    low + rho * (high - low)
}

impl Entering for TreeClaim {
    fn point(&self) -> &[Fp2] {
        &self.point
    }

    fn evals(&self) -> usize {
        2
    }

    fn is_fraction(&self) -> bool {
        false
    }

    fn entering(&self, _: Fp2) -> Fp2 {
        self.value
    }

    fn relation(&self, _: Fp2, evals: &[Fp2]) -> Option<Fp2> {
        match evals {
            [u, w] => Some(*u * *w),
            _ => None,
        }
    }

    fn next(&self, point: Vec<Fp2>, rho: Fp2, evals: &[Fp2]) -> Option<Self> {
        match evals {
            [u, w] => Some(TreeClaim {
                point,
                value: line(*u, *w, rho),
            }),
            _ => None,
        }
    }
}

impl Entering for FractionClaim {
    fn point(&self) -> &[Fp2] {
        &self.point
    }

    fn evals(&self) -> usize {
        4
    }

    fn is_fraction(&self) -> bool {
        true
    }

    fn entering(&self, lambda: Fp2) -> Fp2 {
        self.numerator + lambda * self.denominator
    }

    fn relation(&self, lambda: Fp2, evals: &[Fp2]) -> Option<Fp2> {
        match evals {
            [n0, n1, d0, d1] => Some(*n0 * *d1 + *n1 * *d0 + lambda * *d0 * *d1),
            _ => None,
        }
    }

    fn next(&self, point: Vec<Fp2>, rho: Fp2, evals: &[Fp2]) -> Option<Self> {
        match evals {
            [n0, n1, d0, d1] => Some(FractionClaim {
                point,
                numerator: line(*n0, *n1, rho),
                denominator: line(*d0, *d1, rho),
            }),
            _ => None,
        }
    }
}

impl Entering for LevelClaim {
    fn point(&self) -> &[Fp2] {
        match self {
            LevelClaim::Product(claim) => claim.point(),
            LevelClaim::Fraction(claim) => claim.point(),
        }
    }

    fn evals(&self) -> usize {
        match self {
            LevelClaim::Product(claim) => claim.evals(),
            LevelClaim::Fraction(claim) => claim.evals(),
        }
    }

    fn is_fraction(&self) -> bool {
        match self {
            LevelClaim::Product(claim) => claim.is_fraction(),
            LevelClaim::Fraction(claim) => claim.is_fraction(),
        }
    }

    fn entering(&self, lambda: Fp2) -> Fp2 {
        match self {
            LevelClaim::Product(claim) => claim.entering(lambda),
            LevelClaim::Fraction(claim) => claim.entering(lambda),
        }
    }

    fn relation(&self, lambda: Fp2, evals: &[Fp2]) -> Option<Fp2> {
        match self {
            LevelClaim::Product(claim) => claim.relation(lambda, evals),
            LevelClaim::Fraction(claim) => claim.relation(lambda, evals),
        }
    }

    fn next(&self, point: Vec<Fp2>, rho: Fp2, evals: &[Fp2]) -> Option<Self> {
        match self {
            LevelClaim::Product(claim) => claim.next(point, rho, evals).map(LevelClaim::Product),
            LevelClaim::Fraction(claim) => claim.next(point, rho, evals).map(LevelClaim::Fraction),
        }
    }
}

/// What a level that passed returns, for either kind of claims.
struct Passed<C> {
    gamma: Fp2,
    lambda: Option<Fp2>,
    point: Vec<Fp2>,
    rho: Fp2,
    claims: Vec<C>,
}

/// The checks of 'Tree levels', steps 1 to 9, and the next claims, on instances of any kind.
fn level_with<C: Entering>(
    transcript: &mut impl FiatShamir,
    level: u32,
    claims: &[C],
    proof: &SumcheckProof,
) -> Result<Passed<C>, SumcheckError> {
    let prefix = Prefix::Tree(level);
    // 1. The caller's instances, before the transcript moves.
    let shape = Shape::of(
        LEVEL_DEGREE,
        claims
            .iter()
            .map(|claim| (claim.point().len(), claim.evals())),
    )?;
    // 2. and 3. γ, then λ when a fraction tree is present.
    let gamma = fiat_shamir::challenge(transcript, prefix, Step::Gamma)?;
    let lambda = if claims.iter().any(C::is_fraction) {
        Some(fiat_shamir::challenge(transcript, prefix, Step::Lambda)?)
    } else {
        None
    };
    // λ enters only the claims and relations of fraction trees, which are present exactly when
    // it was squeezed.
    let lambda_or_zero = lambda.unwrap_or(Fp2::ZERO);
    // 4. to 8. The batch, each instance entering with c_i.
    let instances: Vec<Instance> = claims
        .iter()
        .map(|claim| Instance {
            claim: claim.entering(lambda_or_zero),
            vars: claim.point().len(),
            evals: claim.evals(),
        })
        .collect();
    let batch = finish_batch(
        transcript,
        prefix,
        shape,
        gamma,
        &instances,
        proof,
        |index, point, evals| {
            let claim = claims.get(index)?;
            Some(eq(claim.point(), point)? * claim.relation(lambda_or_zero, evals)?)
        },
    )?;
    // 9. ρ, then each instance's claims at ρ ‖ r'_i.
    let rho = fiat_shamir::challenge(transcript, prefix, Step::Line)?;
    let mut next = Vec::with_capacity(claims.len());
    let mut rest: &[Fp2] = &proof.final_evals;
    for (index, claim) in claims.iter().enumerate() {
        // The batch checked that the proof holds each instance's k_i evaluations, and its point
        // has n_max ≥ n_i challenges.
        let fault = SumcheckError::Relation { instance: index };
        let (evals, tail) = rest.split_at_checked(claim.evals()).ok_or(fault)?;
        rest = tail;
        let own = batch.point_of(claim.point().len()).ok_or(fault)?;
        let mut point = Vec::with_capacity(own.len().saturating_add(1));
        point.push(rho);
        point.extend_from_slice(own);
        next.push(claim.next(point, rho, evals).ok_or(fault)?);
    }
    Ok(Passed {
        gamma,
        lambda,
        point: batch.point,
        rho,
        claims: next,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fiat_shamir::stub::{Op, Stub};
    use crate::fiat_shamir::tce;
    use tv_field::Fp;
    use tv_hash::transcript::TranscriptError;

    fn fp2(c0: u64, c1: u64) -> Fp2 {
        Fp2::new(Fp::reduce(c0), Fp::reduce(c1))
    }

    /// The stub's challenge number `n`.
    fn stub_challenge(n: u64) -> Fp2 {
        fp2(1000 + n, 77 * n + 5)
    }

    /// RFC-0014 'Transcript schedule', row 5: `γ_ℓ`, the rounds, the evaluations, then `ρ`.
    #[test]
    fn a_level_runs_the_transcript_schedule_in_order() {
        // One instance on one variable, one on none: n_max = 1, one round of degree 3.
        let claims = [
            TreeClaim {
                point: vec![fp2(4, 1)],
                value: fp2(6, 0),
            },
            TreeClaim {
                point: vec![],
                value: fp2(2, 0),
            },
        ];
        let proof = SumcheckProof {
            rounds: vec![vec![fp2(1, 0), fp2(2, 0), fp2(3, 0)]],
            final_evals: vec![fp2(1, 0), fp2(2, 0), fp2(3, 0), fp2(4, 0)],
        };
        let mut stub = Stub::default();
        let _ = verify_product_level_with(&mut stub, 3, &claims, &proof);
        assert_eq!(
            stub.script(),
            [
                "?tree/3/gamma",
                "+tree/3/round/0",
                "?tree/3/r/0",
                "+tree/3/evals"
            ]
        );
    }

    /// The line challenge is squeezed only after the level's checks pass, and its failure is
    /// the transcript's error.
    #[test]
    fn an_exhausted_line_challenge_is_a_fault_of_the_verifier() {
        // A root on no variable: the level's claim is u·w directly, 3·5 = 15.
        let claims = [TreeClaim {
            point: vec![],
            value: fp2(15, 0),
        }];
        let proof = SumcheckProof {
            rounds: vec![],
            final_evals: vec![fp2(3, 0), fp2(5, 0)],
        };
        let mut stub = Stub::default();
        let next = verify_product_level_with(&mut stub, 0, &claims, &proof).unwrap();
        assert_eq!(
            stub.script(),
            ["?tree/0/gamma", "+tree/0/evals", "?tree/0/line"]
        );
        // The stub's second challenge is ρ.
        let rho = fp2(1001, 82);
        assert_eq!(next.gamma, fp2(1000, 5));
        assert!(next.point.is_empty());
        assert_eq!(next.rho, rho);
        assert_eq!(
            next.claims,
            [TreeClaim {
                point: vec![rho],
                value: fp2(3, 0) + rho * (fp2(5, 0) - fp2(3, 0)),
            }]
        );
        let mut stub = Stub::failing_at(1);
        assert_eq!(
            verify_product_level_with(&mut stub, 0, &claims, &proof),
            Err(SumcheckError::Transcript(TranscriptError::Exhausted))
        );
    }

    /// A fraction tree's root `(n, d)` of two leaves `(N_0, D_0)` and `(N_1, D_1)`: the level's
    /// evaluations and its claims on no variable.
    fn two_leaves() -> (Vec<Fp2>, FractionClaim) {
        let (n0, n1, d0, d1) = (fp2(3, 1), fp2(5, 0), fp2(7, 2), fp2(11, 9));
        let root = FractionClaim {
            point: vec![],
            numerator: n0 * d1 + n1 * d0,
            denominator: d0 * d1,
        };
        (vec![n0, n1, d0, d1], root)
    }

    /// RFC-0014 row 5 and RFC-0010 'Tree levels' step 1: at a level with a fraction tree, `λ`
    /// follows `γ`, before the rounds, wherever the fraction tree stands in schedule order; the
    /// level's checks hold for the root of two leaves; and the next claims are the lines through
    /// the children at `ρ`.
    #[test]
    fn a_level_with_a_fraction_tree_squeezes_lambda_after_gamma() {
        let (evals, root) = two_leaves();
        let product = TreeClaim {
            point: vec![],
            value: fp2(6, 0),
        };
        for fraction_first in [true, false] {
            let (claims, final_evals) = if fraction_first {
                (
                    [
                        LevelClaim::Fraction(root.clone()),
                        LevelClaim::Product(product.clone()),
                    ],
                    [&evals[..], &[fp2(2, 0), fp2(3, 0)]].concat(),
                )
            } else {
                (
                    [
                        LevelClaim::Product(product.clone()),
                        LevelClaim::Fraction(root.clone()),
                    ],
                    [&[fp2(2, 0), fp2(3, 0)], &evals[..]].concat(),
                )
            };
            let proof = SumcheckProof {
                rounds: vec![],
                final_evals: final_evals.clone(),
            };
            let mut stub = Stub::default();
            let level = verify_level_with(&mut stub, 4, &claims, &proof).unwrap();
            assert_eq!(
                stub.script(),
                [
                    "?tree/4/gamma",
                    "?tree/4/lambda",
                    "+tree/4/evals",
                    "?tree/4/line"
                ]
            );
            assert_eq!(
                stub.ops[2],
                Op::Absorb("tree/4/evals".into(), tce(&final_evals))
            );
            assert_eq!(level.gamma, stub_challenge(0));
            assert_eq!(level.lambda, Some(stub_challenge(1)));
            let rho = stub_challenge(2);
            assert_eq!(level.rho, rho);
            let fraction = LevelClaim::Fraction(FractionClaim {
                point: vec![rho],
                numerator: evals[0] + rho * (evals[1] - evals[0]),
                denominator: evals[2] + rho * (evals[3] - evals[2]),
            });
            let forest = LevelClaim::Product(TreeClaim {
                point: vec![rho],
                value: fp2(2, 0) + rho * fp2(1, 0),
            });
            let expected = if fraction_first {
                [fraction, forest]
            } else {
                [forest, fraction]
            };
            assert_eq!(level.claims, expected);
        }
    }

    /// A level of forests alone squeezes no `λ` through [`verify_level`] either, and gives what
    /// [`verify_product_level`] gives.
    #[test]
    fn a_level_of_forests_alone_squeezes_no_lambda() {
        let claims = [TreeClaim {
            point: vec![],
            value: fp2(15, 0),
        }];
        let proof = SumcheckProof {
            rounds: vec![],
            final_evals: vec![fp2(3, 0), fp2(5, 0)],
        };
        let mut stub = Stub::default();
        let product = verify_product_level_with(&mut stub, 0, &claims, &proof).unwrap();
        let wrapped: Vec<LevelClaim> = claims.iter().cloned().map(LevelClaim::Product).collect();
        let mut other = Stub::default();
        let level = verify_level_with(&mut other, 0, &wrapped, &proof).unwrap();
        assert_eq!(other.ops, stub.ops);
        assert_eq!(level.lambda, None);
        assert_eq!(level.gamma, product.gamma);
        assert_eq!(level.rho, product.rho);
        let unwrapped: Vec<LevelClaim> = product
            .claims
            .into_iter()
            .map(LevelClaim::Product)
            .collect();
        assert_eq!(level.claims, unwrapped);
    }

    /// The fraction relation, term by term: each of `N_0·D_1`, `N_1·D_0` and `λ·D_0·D_1` is
    /// needed for the root of two leaves to pass, and an evaluation changed by one is caught by
    /// its cofactor, `D_1`, `D_0`, `N_1 + λ·D_1` and `N_0 + λ·D_0`.
    #[test]
    fn every_term_of_the_fraction_relation_is_checked() {
        let (evals, root) = two_leaves();
        let claims = [LevelClaim::Fraction(root.clone())];
        // Roots that miss one term each.
        let (n0, n1, d0, d1) = (evals[0], evals[1], evals[2], evals[3]);
        let wrong_roots = [
            (n1 * d0, d0 * d1),
            (n0 * d1, d0 * d1),
            (n0 * d1 + n1 * d0, Fp2::ZERO),
        ];
        for (numerator, denominator) in wrong_roots {
            let claims = [LevelClaim::Fraction(FractionClaim {
                point: vec![],
                numerator,
                denominator,
            })];
            let proof = SumcheckProof {
                rounds: vec![],
                final_evals: evals.clone(),
            };
            assert_eq!(
                verify_level_with(&mut Stub::default(), 0, &claims, &proof),
                Err(SumcheckError::FinalMismatch { round: 0 })
            );
        }
        for k in 0..4 {
            let mut changed = evals.clone();
            changed[k] += Fp2::ONE;
            let proof = SumcheckProof {
                rounds: vec![],
                final_evals: changed,
            };
            assert_eq!(
                verify_level_with(&mut Stub::default(), 0, &claims, &proof),
                Err(SumcheckError::FinalMismatch { round: 0 }),
                "evaluation {k}"
            );
        }
        // The children exchanged pass the level, which is symmetric in them, and the next claims
        // are the lines' values at 1 − ρ.
        let swapped = vec![n1, n0, d1, d0];
        let proof = SumcheckProof {
            rounds: vec![],
            final_evals: swapped,
        };
        let level = verify_level_with(&mut Stub::default(), 0, &claims, &proof).unwrap();
        let rho = stub_challenge(2);
        let mirror = Fp2::ONE - rho;
        assert_eq!(
            level.claims,
            [LevelClaim::Fraction(FractionClaim {
                point: vec![rho],
                numerator: n0 + mirror * (n1 - n0),
                denominator: d0 + mirror * (d1 - d0),
            })]
        );
    }

    /// Each challenge of a level with a fraction tree can fail, `γ`, `λ`, a round's and `ρ`, and
    /// nothing follows the failed one.
    #[test]
    fn an_exhausted_challenge_stops_a_level_with_a_fraction_tree() {
        let (_, root) = two_leaves();
        let claims = [LevelClaim::Fraction(FractionClaim {
            point: vec![fp2(9, 4)],
            ..root
        })];
        let proof = SumcheckProof {
            rounds: vec![vec![fp2(1, 0), fp2(2, 0), fp2(3, 0)]],
            final_evals: vec![fp2(1, 0); 4],
        };
        // γ, λ and the round's challenge come before the final check, which then fails.
        for fail_at in 0..3 {
            let mut stub = Stub::failing_at(fail_at);
            assert_eq!(
                verify_level_with(&mut stub, 1, &claims, &proof),
                Err(SumcheckError::Transcript(TranscriptError::Exhausted)),
                "challenge {fail_at}"
            );
            assert!(matches!(stub.ops.last(), Some(Op::Challenge(_))));
        }
    }
}
