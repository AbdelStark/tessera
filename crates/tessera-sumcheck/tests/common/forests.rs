// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Forests of product trees for the tests, and a verifier driver written from RFC-0010 'Tree
//! levels' around the verifier's independent level check, `tv_core::tree::verify_product_level`
//! (DEP-2).
//!
//! The driver holds the schedule as a verifier does: from the forests' shapes alone it knows
//! where each forest joins and which forests each level holds. It squeezes the root challenges
//! itself on a `tv-hash` transcript, forms each root claim `Σ_y v_y·eq(r, y)` from public root
//! values in `tv-field` arithmetic, and carries each forest's next claim to the next level. The
//! leaf claims are checked against the leaves' multilinear extension, computed here from its
//! definition. None of it uses the prover's code.

use tessera_field::Field;
use tessera_sumcheck::{Forest, ProvenForests, prove_forests};
use tv_core::sumcheck as tv;
use tv_core::tree::{ProductLevel, TreeClaim, verify_product_level};
use tv_hash::transcript::Ascii;

use super::{Fp, Fp2, Session, SplitMix64, TvFp2, tv_fp, tv_fp2, tv_fp2s, tv_proof};

/// `2^s` trees of height `h`, their leaves stacked: entry `y·2^h + x` is leaf `x` of tree `y`.
#[derive(Clone)]
pub struct TestForest {
    pub trees_log: usize,
    pub height: usize,
    pub leaves: Vec<Fp>,
}

impl TestForest {
    /// Random leaves. With `zero_rows`, every tree has one zero leaf at a random position, so
    /// that its product is zero, as a `SoftmaxRow` tree's.
    pub fn random(rng: &mut SplitMix64, trees_log: usize, height: usize, zero_rows: bool) -> Self {
        let mut leaves = rng.fps(1 << (trees_log + height));
        if zero_rows {
            for y in 0..1 << trees_log {
                let x = rng.below(1 << height) as usize;
                leaves[(y << height) + x] = Fp::ZERO;
            }
        }
        Self {
            trees_log,
            height,
            leaves,
        }
    }

    pub fn forest(&self) -> Forest<'_, Fp2> {
        Forest::new(&self.leaves, self.height)
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.trees_log, self.height)
    }

    /// The trees' products `v_y`, in `tv-field`: the values of the forest's root level.
    pub fn roots(&self) -> Vec<TvFp2> {
        self.leaves
            .chunks_exact(1 << self.height)
            .map(|tree| {
                TvFp2::from(
                    tree.iter()
                        .fold(super::TvFp::ONE, |product, &leaf| product * tv_fp(leaf)),
                )
            })
            .collect()
    }
}

/// `eq(r, y)` at the vertex `y`, from the definition: `Π_i (r_i·y_i + (1 − r_i)·(1 − y_i))`,
/// `y_i` bit `i − 1` of `y`.
pub fn eq_at_vertex(r: &[TvFp2], y: usize) -> TvFp2 {
    r.iter().enumerate().fold(TvFp2::ONE, |product, (i, &ri)| {
        product
            * if (y >> i) & 1 == 1 {
                ri
            } else {
                TvFp2::ONE - ri
            }
    })
}

/// A root claim's value, `Σ_y v_y·eq(r, y)`, from the public root values `v_y` (RFC-0010 'Tree
/// levels', 'Root claim').
pub fn root_claim(roots: &[TvFp2], r: &[TvFp2]) -> TvFp2 {
    assert_eq!(
        roots.len(),
        1 << r.len(),
        "one root per vertex of the point"
    );
    roots
        .iter()
        .enumerate()
        .fold(TvFp2::ZERO, |sum, (y, &v)| sum + v * eq_at_vertex(r, y))
}

/// `f̃(point) = Σ_x f[x]·eq(point, x)` for a table over `Fp`, in `tv-field`, the `eq` weights
/// built by doubling: after `r_1, …, r_j` the table holds the `2^j` products of the first `j`
/// factors, and `r_{j+1}` splits each entry into `e·(1 − r_{j+1})` and `e·r_{j+1}`.
pub fn mle(table: &[Fp], point: &[TvFp2]) -> TvFp2 {
    assert_eq!(
        table.len(),
        1 << point.len(),
        "2^n entries at n coordinates"
    );
    let mut weights = vec![TvFp2::ONE];
    for &r in point {
        let high: Vec<TvFp2> = weights.iter().map(|&e| e * r).collect();
        for (e, h) in weights.iter_mut().zip(&high) {
            *e -= *h;
        }
        weights.extend(high);
    }
    table
        .iter()
        .zip(&weights)
        .fold(TvFp2::ZERO, |sum, (&f, &e)| sum + e * tv_fp(f))
}

/// One level as the verifier checked it.
pub struct CheckedLevel {
    /// The root challenges it squeezed, `tree/{ℓ}/root/{c}`.
    pub root_challenges: Vec<TvFp2>,
    /// The forests present, in schedule order.
    pub present: Vec<usize>,
    /// The claims entering the level, in schedule order.
    pub claims: Vec<TreeClaim>,
    pub checked: ProductLevel,
}

/// The first check that failed: its level and the verifier's error.
#[derive(Debug, PartialEq, Eq)]
pub struct Rejected {
    pub level: usize,
    pub error: tv::SumcheckError,
}

/// What a verifier that accepted every level holds.
pub struct Verified {
    pub levels: Vec<CheckedLevel>,
    /// One per forest, in the order of the shapes.
    pub leaf_claims: Vec<TreeClaim>,
    pub state: [u8; 32],
}

/// Checks the levels' `proofs` as a verifier holding the schedule does, on a fresh `tv-hash`
/// transcript of `session` (RFC-0010 'Tree levels'). Forest `i` has `2^{s_i}` trees of height
/// `h_i`, `shapes[i] = (s_i, h_i)`, with the public root values `public_roots[i]`.
///
/// With `H` the tallest height, level `ℓ` runs from 0 to `H − 1`. The forests of height
/// `H − ℓ` join there: if any has more than one tree, the challenges `c = 0, …, s_max − 1` are
/// squeezed under `tree/{ℓ}/root/{c}`, and each joining forest's root point is the last `s_i` of
/// them, in the order squeezed, its root claim `Σ_y v_y·eq(r, y)`. The forests that have joined,
/// in the order of the shapes, are the level's instances. `root_change = Some((ℓ, c))` adds 1 to
/// challenge `c` of level `ℓ` before the root points are formed, as a prover would that used
/// another root challenge than the transcript's.
pub fn verify_forests(
    session: &Session,
    shapes: &[(usize, usize)],
    public_roots: &[Vec<TvFp2>],
    proofs: &[tv::SumcheckProof],
    root_change: Option<(usize, usize)>,
) -> Result<Verified, Rejected> {
    let tallest = shapes.iter().map(|&(_, h)| h).max().expect("a forest");
    assert_eq!(proofs.len(), tallest, "one proof per level");
    let mut transcript = session.tv_transcript();
    let mut claims: Vec<Option<TreeClaim>> = vec![None; shapes.len()];
    let mut levels = Vec::with_capacity(tallest);
    for (level, proof) in proofs.iter().enumerate() {
        let joining: Vec<usize> = (0..shapes.len())
            .filter(|&i| tallest - shapes[i].1 == level)
            .collect();
        let s_max = joining.iter().map(|&i| shapes[i].0).max().unwrap_or(0);
        let root_challenges: Vec<TvFp2> = (0..s_max)
            .map(|c| {
                let label = format!("tree/{level}/root/{c}");
                transcript
                    .challenge_fp2(Ascii::new(&label).expect("ASCII"))
                    .expect("a challenge")
            })
            .collect();
        let mut root = root_challenges.clone();
        if let Some((changed, c)) = root_change
            && changed == level
        {
            root[c] += TvFp2::ONE;
        }
        for &i in &joining {
            let r = root[s_max - shapes[i].0..].to_vec();
            let value = root_claim(&public_roots[i], &r);
            claims[i] = Some(TreeClaim { point: r, value });
        }
        let present: Vec<usize> = (0..shapes.len())
            .filter(|&i| tallest - shapes[i].1 <= level)
            .collect();
        let entering: Vec<TreeClaim> = present
            .iter()
            .map(|&i| claims[i].clone().expect("a forest that has joined"))
            .collect();
        let checked = verify_product_level(&mut transcript, level as u32, &entering, proof)
            .map_err(|error| Rejected { level, error })?;
        assert_eq!(
            checked.claims.len(),
            present.len(),
            "a next claim per instance"
        );
        for (&i, next) in present.iter().zip(&checked.claims) {
            claims[i] = Some(next.clone());
        }
        levels.push(CheckedLevel {
            root_challenges,
            present,
            claims: entering,
            checked,
        });
    }
    Ok(Verified {
        levels,
        leaf_claims: claims
            .into_iter()
            .map(|claim| claim.expect("every forest joins"))
            .collect(),
        state: *transcript.state(),
    })
}

pub fn shapes(forests: &[TestForest]) -> Vec<(usize, usize)> {
    forests.iter().map(TestForest::shape).collect()
}

pub fn true_roots(forests: &[TestForest]) -> Vec<Vec<TvFp2>> {
    forests.iter().map(TestForest::roots).collect()
}

/// The levels' messages in the verifier's types.
pub fn tv_proofs(proven: &ProvenForests<Fp2>) -> Vec<tv::SumcheckProof> {
    proven
        .levels()
        .iter()
        .map(|level| tv_proof(level.proof()))
        .collect()
}

/// Proves the forests on a fresh transcript of `session`, and returns the result and the
/// transcript's final state.
pub fn prove(session: &Session, forests: &[TestForest]) -> (ProvenForests<Fp2>, [u8; 32]) {
    let borrowed: Vec<Forest<'_, Fp2>> = forests.iter().map(TestForest::forest).collect();
    let mut transcript = session.transcript();
    let proven = prove_forests(&mut transcript, &borrowed).expect("well-formed forests");
    (proven, transcript.state())
}

/// Proves the forests and checks the proof with the verifier driver, from the trees' true
/// products: it accepts; its root challenges, instances, `γ`, points, `ρ` and next claims are
/// the prover's at every level; its transcript ends in the prover's state; and every leaf claim
/// is the verifier's and the stacked leaves' multilinear extension at its point.
///
/// The verifier checks the messages that `into_parts` hands over, from a second proof on a
/// fresh transcript, which must be the first's, and its leaf claims are the ones checked.
pub fn prove_and_verify(session: &Session, forests: &[TestForest]) -> ProvenForests<Fp2> {
    let (proven, state) = prove(session, forests);
    let (proofs, leaf_claims) = prove(session, forests).0.into_parts();
    let tallest = forests.iter().map(|f| f.height).max().expect("a forest");
    assert_eq!(proven.levels().len(), tallest, "one level per height");
    assert_eq!(proofs.len(), tallest, "one proof per level");
    for (level, (proof, mine)) in proofs.iter().zip(proven.levels()).enumerate() {
        assert!(
            proof == mine.proof(),
            "level {level}: the messages handed over"
        );
    }
    assert!(
        leaf_claims == proven.leaf_claims(),
        "the leaf claims handed over"
    );
    let verified = verify_forests(
        session,
        &shapes(forests),
        &true_roots(forests),
        &proofs.iter().map(tv_proof).collect::<Vec<_>>(),
        None,
    )
    .unwrap_or_else(|rejected| panic!("the verifier rejects an honest proof: {rejected:?}"));
    assert_eq!(
        verified.state, state,
        "the transcripts end in the same state"
    );
    for (level, (mine, theirs)) in proven.levels().iter().zip(&verified.levels).enumerate() {
        assert_eq!(
            tv_fp2s(mine.root_challenges()),
            theirs.root_challenges,
            "level {level}: the root challenges"
        );
        assert_eq!(
            mine.forests(),
            theirs.present,
            "level {level}: the instances"
        );
        assert_eq!(
            tv_fp2(*mine.gamma()),
            theirs.checked.gamma,
            "level {level}: γ"
        );
        assert_eq!(
            tv_fp2s(mine.point()),
            theirs.checked.point,
            "level {level}: the point"
        );
        assert_eq!(tv_fp2(*mine.rho()), theirs.checked.rho, "level {level}: ρ");
        assert_eq!(mine.next_claims().len(), theirs.checked.claims.len());
        for (k, (claim, next)) in mine
            .next_claims()
            .iter()
            .zip(&theirs.checked.claims)
            .enumerate()
        {
            assert_eq!(
                tv_fp2s(&claim.point),
                next.point,
                "level {level}, instance {k}: the next point"
            );
            assert_eq!(
                tv_fp2(claim.value),
                next.value,
                "level {level}, instance {k}: the next value"
            );
        }
    }
    assert_eq!(leaf_claims.len(), forests.len(), "a leaf claim per forest");
    for (i, ((claim, theirs), forest)) in leaf_claims
        .iter()
        .zip(&verified.leaf_claims)
        .zip(forests)
        .enumerate()
    {
        let point = tv_fp2s(&claim.point);
        assert_eq!(
            point.len(),
            forest.trees_log + forest.height,
            "forest {i}: a leaf claim on the stacked leaves"
        );
        assert_eq!(point, theirs.point, "forest {i}: the leaf point");
        assert_eq!(
            tv_fp2(claim.value),
            theirs.value,
            "forest {i}: the leaf value"
        );
        assert_eq!(
            tv_fp2(claim.value),
            mle(&forest.leaves, &point),
            "forest {i}: the leaves' extension at the leaf point"
        );
    }
    proven
}
