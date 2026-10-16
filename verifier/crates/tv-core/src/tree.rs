// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The levels of the leaf stage's product trees (RFC-0010 'Tree levels'). The crate
//! documentation is their specification.
//!
//! A product tree of height `h` has levels `0` (the root, one entry) to `h` (the leaves,
//! `2^h` entries), and the pair `2x, 2x + 1` of level `ℓ + 1` combines into entry `x` of level
//! `ℓ`: `V_ℓ[x] = V_{ℓ+1}[2x]·V_{ℓ+1}[2x + 1]`. As multilinear polynomials,
//! `Ṽ_ℓ(r) = Σ_x eq(r, x)·Ṽ_{ℓ+1}(0 ‖ x)·Ṽ_{ℓ+1}(1 ‖ x)`: the pair bit is the lowest variable
//! of level `ℓ + 1`. One batched sumcheck of degree 3 per level of the batch reduces every
//! instance's claim on its level to a claim on the next, from the roots to the leaves.
//!
//! Trees of different heights share the batch's levels: with `H` the tallest tree's height, a
//! tree of height `h` takes its root level at level `H − h` of the batch and has `ℓ − (H − h)`
//! variables at level `ℓ`. The caller, which holds the schedule, passes the instances present at
//! each level in schedule order: the claims the previous level returned, and the public root
//! claims of the trees that join, on no variable.

use tv_field::Fp2;
use tv_hash::transcript::Transcript;

use crate::fiat_shamir::{self, FiatShamir, Step};
use crate::sumcheck::{Instance, Prefix, SumcheckError, SumcheckProof, eq, verify_batch_with};

/// The degree of a product-tree level's sumcheck: `eq(r, x)·Ṽ(0 ‖ x)·Ṽ(1 ‖ x)` is of degree 3
/// in each variable (RFC-0010 'Relations', `ProductTreeLevel`).
pub const PRODUCT_LEVEL_DEGREE: usize = 3;

/// A claim on one level of a product tree: `Ṽ(point) = value`, over `point.len()` variables.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TreeClaim {
    /// The point, its low variables first.
    pub point: Vec<Fp2>,
    /// The claimed value of the level's multilinear polynomial there.
    pub value: Fp2,
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

/// Checks level `level` of the leaf stage for the product-tree instances present there, each
/// `claims[i]` a claim on its current level, in schedule order, and returns their claims on the
/// next level, in the same order, with the level's challenges. The checks, in order, are those
/// of the crate documentation ('Product-tree levels'): a batched sumcheck of degree 3 under
/// `tree/{level}`, whose final evaluations are `u_i, w_i` for each instance and must satisfy
/// `Σ_i γ^i · eq(r_i, r'_i) · u_i · w_i = claim`, then the line challenge `ρ`. Instance `i`'s
/// next claim is `Ṽ(ρ ‖ r'_i) = u_i + ρ·(w_i − u_i)`, with `r'_i` its point in the batch.
///
/// After a tree's last level, its claim is a leaf claim, which the next stage checks.
///
/// Every instance is a product tree's. A level that also holds LogUp fraction-tree instances
/// (RFC-0008) is not checked here: their layout within a level is RFC-0010 OQ-0010-1.
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
    let prefix = Prefix::Tree(level);
    let instances: Vec<Instance> = claims
        .iter()
        .map(|claim| Instance {
            claim: claim.value,
            vars: claim.point.len(),
            evals: 2,
        })
        .collect();
    let batch = verify_batch_with(
        transcript,
        prefix,
        PRODUCT_LEVEL_DEGREE,
        &instances,
        proof,
        |index, point, evals| {
            let claim = claims.get(index)?;
            match evals {
                [u, w] => Some(eq(&claim.point, point)? * *u * *w),
                _ => None,
            }
        },
    )?;
    let rho = fiat_shamir::challenge(transcript, prefix, Step::Line)?;
    // verify_batch checked that the proof holds exactly two evaluations per instance.
    let us = proof.final_evals.iter().step_by(2);
    let ws = proof.final_evals.iter().skip(1).step_by(2);
    let next = claims
        .iter()
        .zip(us.zip(ws))
        .enumerate()
        .map(|(index, (claim, (&u, &w)))| {
            let own = batch
                .point_of(claim.point.len())
                .ok_or(SumcheckError::Relation { instance: index })?;
            let mut point = Vec::with_capacity(own.len().saturating_add(1));
            point.push(rho);
            point.extend_from_slice(own);
            Ok(TreeClaim {
                point,
                value: u + rho * (w - u),
            })
        })
        .collect::<Result<Vec<TreeClaim>, SumcheckError>>()?;
    Ok(ProductLevel {
        gamma: batch.gamma,
        point: batch.point,
        rho,
        claims: next,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fiat_shamir::stub::Stub;
    use tv_field::Fp;
    use tv_hash::transcript::TranscriptError;

    fn fp2(c0: u64, c1: u64) -> Fp2 {
        Fp2::new(Fp::reduce(c0), Fp::reduce(c1))
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
}
