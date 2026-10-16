// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Fraction trees beside product forests through the independent verifier (DEP-2; RFC-0010
//! 'Tree levels', RFC-0008 'Fraction tree'): every schedule is proven by `prove_trees` and checked
//! level by level by a verifier driver written here from RFC-0010 around
//! `tv_core::tree::verify_level`.
//!
//! The driver holds the schedule as a verifier does: from the trees' shapes alone it knows where
//! each joins. It squeezes the forests' root challenges itself and forms their root claims from
//! the trees' products; a fraction tree enters with its root, computed here bottom-up in
//! `tv-field` from the leaves by the rule `(n_L·d_R + n_R·d_L, d_L·d_R)`, as the verifier would
//! receive it under `logup/roots`. The verifier must accept and draw the prover's challenges, `λ`
//! exactly at the levels where a fraction tree is present; every next claim must be the
//! verifier's; and every leaf claim must be the multilinear extension of the leaves' values,
//! numerators and denominators, at its point.
//!
//! A level of forests alone squeezes no `λ`, so `prove_forests` and `prove_trees` agree on a batch
//! of forests; a forest beside fraction trees keeps its messages' form, `u, w`.

mod common;

use common::forests::{TestForest, eq_at_vertex, mle, root_claim};
use common::{Fp, Fp2, Session, SplitMix64, TvFp2, tv_fp, tv_fp2, tv_fp2s, tv_proof};
use tessera_field::PrimeField;
use tessera_sumcheck::{
    FractionTree, LeafNumerators, LevelClaim, ProvenTrees, Tree, prove_forests, prove_trees,
};
use tv_core::sumcheck as tv;
use tv_core::tree as tvt;
use tv_hash::transcript::Ascii;

/// A fraction tree's leaves: numerators over `Fp`, or `None` for numerators of 1, and
/// denominators over `Fp2`.
#[derive(Clone)]
struct TestFraction {
    numerators: Option<Vec<Fp>>,
    denominators: Vec<Fp2>,
}

impl TestFraction {
    fn random(rng: &mut SplitMix64, height: usize, unit: bool) -> Self {
        Self {
            numerators: (!unit).then(|| rng.fps(1 << height)),
            denominators: rng.fp2s(1 << height),
        }
    }

    fn height(&self) -> usize {
        self.denominators.len().trailing_zeros() as usize
    }

    fn tree(&self) -> FractionTree<Fp2> {
        let numerators = match &self.numerators {
            Some(n) => LeafNumerators::Base(n),
            None => LeafNumerators::Unit,
        };
        FractionTree::new(self.height(), numerators, |x| self.denominators[x]).expect("a tree")
    }

    /// The leaves' numerators in the verifier's field.
    fn tv_numerators(&self) -> Vec<TvFp2> {
        match &self.numerators {
            Some(n) => n.iter().map(|&x| TvFp2::from(tv_fp(x))).collect(),
            None => vec![TvFp2::ONE; self.denominators.len()],
        }
    }

    /// The root, bottom-up from the leaves: `(n_L·d_R + n_R·d_L, d_L·d_R)` of entries `2x` and
    /// `2x + 1`.
    fn tv_root(&self) -> (TvFp2, TvFp2) {
        let mut level: Vec<(TvFp2, TvFp2)> = self
            .tv_numerators()
            .into_iter()
            .zip(tv_fp2s(&self.denominators))
            .collect();
        while level.len() > 1 {
            level = level
                .chunks_exact(2)
                .map(|pair| {
                    let ((nl, dl), (nr, dr)) = (pair[0], pair[1]);
                    (nl * dr + nr * dl, dl * dr)
                })
                .collect();
        }
        level[0]
    }
}

#[derive(Clone)]
enum TestTree {
    Forest(TestForest),
    Fraction(TestFraction),
}

impl TestTree {
    fn height(&self) -> usize {
        match self {
            Self::Forest(forest) => forest.height,
            Self::Fraction(fraction) => fraction.height(),
        }
    }

    fn trees_log(&self) -> usize {
        match self {
            Self::Forest(forest) => forest.trees_log,
            Self::Fraction(_) => 0,
        }
    }

    fn tree(&self) -> Tree<'_, Fp2> {
        match self {
            Self::Forest(forest) => Tree::Forest(forest.forest()),
            Self::Fraction(fraction) => Tree::Fraction(fraction.tree()),
        }
    }
}

/// `f̃(point)` of a table over `Fp2`, from the definition `Σ_x f[x]·eq(point, x)`.
fn mle_fp2(table: &[TvFp2], point: &[TvFp2]) -> TvFp2 {
    assert_eq!(table.len(), 1 << point.len());
    table
        .iter()
        .enumerate()
        .fold(TvFp2::ZERO, |sum, (x, &f)| sum + f * eq_at_vertex(point, x))
}

/// What the verifier driver holds after the last level.
struct Verified {
    levels: Vec<tvt::TreeLevel>,
    root_challenges: Vec<Vec<TvFp2>>,
    present: Vec<Vec<usize>>,
    leaf_claims: Vec<tvt::LevelClaim>,
    state: [u8; 32],
}

/// Checks the levels' `proofs` as a verifier holding the schedule does, on a fresh `tv-hash`
/// transcript of `session`, or returns the first level that fails and its error.
/// `scale = Some((i, c))` multiplies fraction tree `i`'s root claims by `c`.
fn verify_trees(
    session: &Session,
    trees: &[TestTree],
    proofs: &[tv::SumcheckProof],
    scale: Option<(usize, TvFp2)>,
) -> Result<Verified, (usize, tv::SumcheckError)> {
    let tallest = trees.iter().map(TestTree::height).max().unwrap_or(0);
    assert_eq!(proofs.len(), tallest, "one proof per level");
    let mut transcript = session.tv_transcript();
    // Each tree's claims on its current level; a fraction tree's root from the start.
    let mut claims: Vec<Option<tvt::LevelClaim>> = trees
        .iter()
        .enumerate()
        .map(|(i, tree)| match tree {
            TestTree::Fraction(fraction) => {
                let (numerator, denominator) = fraction.tv_root();
                let c = match scale {
                    Some((j, c)) if j == i => c,
                    _ => TvFp2::ONE,
                };
                Some(tvt::LevelClaim::Fraction(tvt::FractionClaim {
                    point: Vec::new(),
                    numerator: c * numerator,
                    denominator: c * denominator,
                }))
            }
            TestTree::Forest(_) => None,
        })
        .collect();
    let mut verified = Verified {
        levels: Vec::new(),
        root_challenges: Vec::new(),
        present: Vec::new(),
        leaf_claims: Vec::new(),
        state: [0; 32],
    };
    for (level, proof) in proofs.iter().enumerate() {
        let joins = |i: usize| tallest - trees[i].height() == level;
        let s_max = (0..trees.len())
            .filter(|&i| joins(i))
            .map(|i| trees[i].trees_log())
            .max()
            .unwrap_or(0);
        let root: Vec<TvFp2> = (0..s_max)
            .map(|c| {
                let label = format!("tree/{level}/root/{c}");
                transcript
                    .challenge_fp2(Ascii::new(&label).expect("ASCII"))
                    .expect("a challenge")
            })
            .collect();
        for i in (0..trees.len()).filter(|&i| joins(i)) {
            if let TestTree::Forest(forest) = &trees[i] {
                let r = root[s_max - forest.trees_log..].to_vec();
                let value = root_claim(&forest.roots(), &r);
                claims[i] = Some(tvt::LevelClaim::Product(tvt::TreeClaim { point: r, value }));
            }
        }
        let present: Vec<usize> = (0..trees.len())
            .filter(|&i| tallest - trees[i].height() <= level)
            .collect();
        let entering: Vec<tvt::LevelClaim> = present
            .iter()
            .map(|&i| claims[i].clone().expect("a tree that has joined"))
            .collect();
        let checked = tvt::verify_level(&mut transcript, level as u32, &entering, proof)
            .map_err(|error| (level, error))?;
        for (&i, next) in present.iter().zip(&checked.claims) {
            claims[i] = Some(next.clone());
        }
        verified.root_challenges.push(root);
        verified.present.push(present);
        verified.levels.push(checked);
    }
    verified.leaf_claims = claims
        .into_iter()
        .map(|claim| claim.expect("every tree has claims"))
        .collect();
    verified.state = *transcript.state();
    Ok(verified)
}

fn tv_claim(claim: &LevelClaim<Fp2>) -> tvt::LevelClaim {
    match claim {
        LevelClaim::Product(c) => tvt::LevelClaim::Product(tvt::TreeClaim {
            point: tv_fp2s(&c.point),
            value: tv_fp2(c.value),
        }),
        LevelClaim::Fraction(c) => tvt::LevelClaim::Fraction(tvt::FractionClaim {
            point: tv_fp2s(&c.point),
            numerator: tv_fp2(c.numerator),
            denominator: tv_fp2(c.denominator),
        }),
    }
}

fn prove(session: &Session, trees: &[TestTree]) -> (ProvenTrees<Fp2>, [u8; 32]) {
    let mut transcript = session.transcript();
    let proven = prove_trees(&mut transcript, trees.iter().map(TestTree::tree).collect())
        .expect("well-formed trees");
    (proven, transcript.state())
}

/// Proves the schedule and checks it with the verifier driver: it accepts, every challenge and
/// claim is the prover's, `λ` is squeezed exactly where a fraction tree is present, and every
/// leaf claim is its leaves' extension at its point.
fn prove_and_verify(session: &Session, trees: &[TestTree]) -> ProvenTrees<Fp2> {
    for tree in trees {
        if let TestTree::Fraction(fraction) = tree {
            let (n, d) = fraction.tree().root();
            assert_eq!((tv_fp2(n), tv_fp2(d)), fraction.tv_root(), "the root");
        }
    }
    let (proven, state) = prove(session, trees);
    let proofs: Vec<tv::SumcheckProof> = proven
        .levels()
        .iter()
        .map(|level| tv_proof(level.proof()))
        .collect();
    let verified = verify_trees(session, trees, &proofs, None)
        .unwrap_or_else(|(level, e)| panic!("level {level} rejects an honest proof: {e}"));
    assert_eq!(
        verified.state, state,
        "the transcripts end in the same state"
    );
    for (level, ((mine, theirs), present)) in proven
        .levels()
        .iter()
        .zip(&verified.levels)
        .zip(&verified.present)
        .enumerate()
    {
        assert_eq!(
            tv_fp2s(mine.root_challenges()),
            verified.root_challenges[level],
            "level {level}: the root challenges"
        );
        assert_eq!(mine.trees(), present, "level {level}: the instances");
        let fractions = present
            .iter()
            .any(|&i| matches!(trees[i], TestTree::Fraction(_)));
        assert_eq!(mine.lambda().is_some(), fractions, "level {level}: λ");
        assert_eq!(
            mine.lambda().map(|l| tv_fp2(*l)),
            theirs.lambda,
            "level {level}"
        );
        assert_eq!(tv_fp2(*mine.gamma()), theirs.gamma, "level {level}: γ");
        assert_eq!(
            tv_fp2s(mine.point()),
            theirs.point,
            "level {level}: the point"
        );
        assert_eq!(tv_fp2(*mine.rho()), theirs.rho, "level {level}: ρ");
        let next: Vec<tvt::LevelClaim> = mine.next_claims().iter().map(tv_claim).collect();
        assert_eq!(next, theirs.claims, "level {level}: the next claims");
    }
    assert_eq!(
        proven.leaf_claims().len(),
        trees.len(),
        "a leaf claim per tree"
    );
    for (i, ((claim, theirs), tree)) in proven
        .leaf_claims()
        .iter()
        .zip(&verified.leaf_claims)
        .zip(trees)
        .enumerate()
    {
        let claim = tv_claim(claim);
        assert_eq!(&claim, theirs, "tree {i}: the leaf claim");
        match (tree, claim) {
            (TestTree::Forest(forest), tvt::LevelClaim::Product(c)) => {
                assert_eq!(
                    c.value,
                    mle(&forest.leaves, &c.point),
                    "tree {i}: the leaves"
                );
            }
            (TestTree::Fraction(fraction), tvt::LevelClaim::Fraction(c)) => {
                assert_eq!(c.point.len(), fraction.height(), "tree {i}: on the leaves");
                assert_eq!(
                    c.numerator,
                    mle_fp2(&fraction.tv_numerators(), &c.point),
                    "tree {i}: the numerators"
                );
                assert_eq!(
                    c.denominator,
                    mle_fp2(&tv_fp2s(&fraction.denominators), &c.point),
                    "tree {i}: the denominators"
                );
            }
            _ => panic!("tree {i}: a claim of the other kind"),
        }
    }
    proven
}

/// A fraction tree of every height from 0 to 10, alone, with numerators of 1 and over `Fp`.
#[test]
fn single_fraction_trees_of_every_height_verify() {
    let mut rng = SplitMix64::new(0x60_0201);
    let session = Session::random(&mut rng);
    for height in 0..=10 {
        for unit in [true, false] {
            let tree = TestTree::Fraction(TestFraction::random(&mut rng, height, unit));
            let proven = prove_and_verify(&session, &[tree]);
            assert_eq!(proven.levels().len(), height);
            assert!(proven.levels().iter().all(|l| l.lambda().is_some()));
        }
    }
}

/// The two trees of LogUp instances of different heights, joining where their heights match, in
/// the schedule orders the vectors use: paired, all lookup trees first, a table tree first.
#[test]
fn fraction_trees_of_several_heights_join_where_their_heights_match() {
    let mut rng = SplitMix64::new(0x60_0202);
    let session = Session::random(&mut rng);
    let shapes = [
        (5, true),
        (3, false),
        (0, true),
        (7, false),
        (2, true),
        (7, true),
    ];
    let trees: Vec<TestTree> = shapes
        .iter()
        .map(|&(h, unit)| TestTree::Fraction(TestFraction::random(&mut rng, h, unit)))
        .collect();
    let proven = prove_and_verify(&session, &trees);
    let present: Vec<&[usize]> = proven.levels().iter().map(|l| l.trees()).collect();
    assert_eq!(
        present,
        [
            &[3, 5][..],
            &[3, 5],
            &[0, 3, 5],
            &[0, 3, 5],
            &[0, 1, 3, 5],
            &[0, 1, 3, 4, 5],
            &[0, 1, 3, 4, 5],
        ]
    );
    let mut reversed = trees.clone();
    reversed.reverse();
    let _ = prove_and_verify(&session, &reversed);
}

/// Forests and fraction trees in one schedule, interleaved: a level of forests alone squeezes
/// no `λ`; forests of several trees join under root challenges where fraction trees are present;
/// a forest's evaluations stay `u, w` beside a fraction tree's `N_0, N_1, D_0, D_1`.
#[test]
fn fraction_trees_beside_forests_verify() {
    let mut rng = SplitMix64::new(0x60_0203);
    let session = Session::random(&mut rng);
    let trees = vec![
        TestTree::Fraction(TestFraction::random(&mut rng, 2, false)),
        TestTree::Forest(TestForest::random(&mut rng, 1, 5, false)),
        TestTree::Fraction(TestFraction::random(&mut rng, 3, true)),
        TestTree::Forest(TestForest::random(&mut rng, 2, 2, true)),
        TestTree::Fraction(TestFraction::random(&mut rng, 0, true)),
        TestTree::Forest(TestForest::random(&mut rng, 0, 3, false)),
        TestTree::Fraction(TestFraction::random(&mut rng, 4, false)),
    ];
    let proven = prove_and_verify(&session, &trees);
    let lambdas: Vec<bool> = proven
        .levels()
        .iter()
        .map(|l| l.lambda().is_some())
        .collect();
    assert_eq!(lambdas, [false, true, true, true, true]);
    let roots: Vec<usize> = proven
        .levels()
        .iter()
        .map(|l| l.root_challenges().len())
        .collect();
    assert_eq!(roots, [1, 0, 0, 2, 0]);
    // The forest of 2^1 trees alone at level 0: two evaluations, u and w.
    assert_eq!(proven.levels()[0].proof().final_evals.len(), 2);
    // Level 4: trees 0, 1, 2, 3, 5 and 6, forests 2 evaluations, fraction trees 4.
    assert_eq!(proven.levels()[4].trees(), [0, 1, 2, 3, 5, 6]);
    assert_eq!(
        proven.levels()[4].proof().final_evals.len(),
        4 + 2 + 4 + 2 + 2 + 4
    );
}

/// Fraction trees of height 0 alone, or no tree at all: no level, the roots are the leaf claims,
/// and the transcript does not move.
#[test]
fn trees_of_height_0_join_no_level() {
    let mut rng = SplitMix64::new(0x60_0204);
    let session = Session::random(&mut rng);
    let trees = vec![
        TestTree::Fraction(TestFraction::random(&mut rng, 0, true)),
        TestTree::Fraction(TestFraction::random(&mut rng, 0, false)),
    ];
    let proven = prove_and_verify(&session, &trees);
    assert!(proven.levels().is_empty());
    let (proven, state) = prove(&session, &[]);
    assert!(proven.levels().is_empty() && proven.leaf_claims().is_empty());
    assert_eq!(
        state,
        session.transcript().state(),
        "no transcript operation"
    );
}

/// Seeded random schedules: 1 to 6 trees, forests of 2^0 to 2^4 trees of height 1 to 6 and
/// fraction trees of height 0 to 8 with either numerators, in random order.
#[test]
fn seeded_random_schedules_verify() {
    let mut rng = SplitMix64::new(0x60_0205);
    let session = Session::random(&mut rng);
    for _ in 0..40 {
        let count = 1 + rng.below(6) as usize;
        let trees: Vec<TestTree> = (0..count)
            .map(|_| match rng.coin() {
                true => {
                    let (s, h, zero) = (rng.below(5), 1 + rng.below(6), rng.coin());
                    TestTree::Forest(TestForest::random(&mut rng, s as usize, h as usize, zero))
                }
                false => {
                    let (h, unit) = (rng.below(9) as usize, rng.coin());
                    TestTree::Fraction(TestFraction::random(&mut rng, h, unit))
                }
            })
            .collect();
        let _ = prove_and_verify(&session, &trees);
    }
}

/// On forests alone, `prove_trees` is `prove_forests`: the same messages, challenges and claims.
#[test]
fn forests_alone_prove_as_prove_forests_does() {
    let mut rng = SplitMix64::new(0x60_0206);
    let session = Session::random(&mut rng);
    let forests = [
        TestForest::random(&mut rng, 3, 4, true),
        TestForest::random(&mut rng, 0, 6, false),
        TestForest::random(&mut rng, 2, 2, false),
    ];
    let trees: Vec<TestTree> = forests.iter().cloned().map(TestTree::Forest).collect();
    let (general, state) = prove(&session, &trees);
    let borrowed: Vec<_> = forests.iter().map(TestForest::forest).collect();
    let mut transcript = session.transcript();
    let specific = prove_forests(&mut transcript, &borrowed).expect("well-formed forests");
    assert_eq!(transcript.state(), state);
    for (a, b) in general.levels().iter().zip(specific.levels()) {
        assert!(a.lambda().is_none());
        assert!(a.proof() == b.proof());
        assert_eq!(a.root_challenges(), b.root_challenges());
        assert_eq!(a.trees(), b.forests());
        let products: Vec<_> = a
            .next_claims()
            .iter()
            .map(|c| match c {
                LevelClaim::Product(c) => c.clone(),
                LevelClaim::Fraction(_) => panic!("a forest's claim"),
            })
            .collect();
        assert!(products == b.next_claims());
    }
}

/// A fraction tree entering on `(c·n, c·d)` for its root `(n, d)`, the same fraction, as a
/// roots message that scales a root would make it, is rejected at the final check of the level
/// where it joins: its claims are not its level's.
#[test]
fn a_scaled_root_is_rejected_where_its_tree_joins() {
    let mut rng = SplitMix64::new(0x60_0207);
    let session = Session::random(&mut rng);
    let trees = vec![
        TestTree::Fraction(TestFraction::random(&mut rng, 4, false)),
        TestTree::Fraction(TestFraction::random(&mut rng, 2, true)),
    ];
    let (proven, _) = prove(&session, &trees);
    let proofs: Vec<tv::SumcheckProof> = proven
        .levels()
        .iter()
        .map(|level| tv_proof(level.proof()))
        .collect();
    assert!(verify_trees(&session, &trees, &proofs, None).is_ok());
    let c = TvFp2::from(tv_fp(Fp::from_canonical_u64(3).expect("canonical")));
    // Tree 1, of height 2, joins at level 2 on no variable, beside tree 0 on 2 variables.
    assert_eq!(
        verify_trees(&session, &trees, &proofs, Some((1, c))).map(|_| ()),
        Err((2, tv::SumcheckError::FinalMismatch { round: 2 }))
    );
    // Tree 0, the tallest, at level 0 alone: no round.
    assert_eq!(
        verify_trees(&session, &trees, &proofs, Some((0, c))).map(|_| ()),
        Err((0, tv::SumcheckError::FinalMismatch { round: 0 }))
    );
}
