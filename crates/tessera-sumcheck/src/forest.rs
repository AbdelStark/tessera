// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The product-tree levels of the leaf stage, over forests of stacked trees (RFC-0010 'Tree
//! levels').

use core::fmt;

use tessera_field::Field;
use tessera_poly::PolyField;

use crate::trees::{LevelClaim, ProvenTreeLevel, Tree, prove_trees};
use crate::{Claim, Prefix, SumcheckError, SumcheckProof, SumcheckTranscript};

/// `2^s` product trees of one height `h`, stacked into one table of `2^{s+h}` leaves over the
/// prime subfield: entry `y·2^h + x` is leaf `x` of tree `y` (RFC-0010 'Tree levels', 'Layout').
/// The tree index takes the high `s` variables of the table and the position within a tree the
/// low `h`. A single tree is a forest with `s = 0`, and a schedule whose trees are not a power of
/// two pads them with trees whose root values are public as well.
///
/// Level `t` of the forest, from its roots (`t = 0`, one entry per tree) to its leaves (`t = h`),
/// stacks its trees' levels the same way, `2^{s+t}` entries with
/// `V_t[z] = V_{t+1}[2z]·V_{t+1}[2z + 1]`, so the pair bit is the lowest variable at every level.
///
/// The constructor takes any slice and height; [`prove_forests`] checks the shape: a leaf count
/// that is a power of two, and `1 ≤ h ≤ log2` of it.
///
/// The leaves are secret-class values (05 §3): the debug form shows the leaf count and the
/// height, never a leaf.
pub struct Forest<'a, F: PolyField> {
    pub(crate) leaves: &'a [F::Base],
    pub(crate) height: usize,
}

impl<F: PolyField> Clone for Forest<'_, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: PolyField> Copy for Forest<'_, F> {}

impl<'a, F: PolyField> Forest<'a, F> {
    /// The forest whose stacked leaves are `leaves`, of trees of height `height`: `2^s` trees
    /// with `2^s = leaves.len() / 2^height`.
    pub fn new(leaves: &'a [F::Base], height: usize) -> Self {
        Self { leaves, height }
    }

    /// The height `h` of its trees: the number of levels below the roots.
    pub fn height(&self) -> usize {
        self.height
    }

    /// The number of leaves, `2^{s+h}` in a valid forest.
    pub fn len(&self) -> usize {
        self.leaves.len()
    }

    /// Whether the forest has no leaf, which [`prove_forests`] refuses.
    pub fn is_empty(&self) -> bool {
        self.leaves.is_empty()
    }
}

/// Shows the leaf count and the height, never a leaf (05 §3).
impl<F: PolyField> fmt::Debug for Forest<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Forest")
            .field("leaves", &self.len())
            .field("height", &self.height)
            .finish()
    }
}

/// One level `ℓ` of a proven batch of forests: its challenges, its messages and the claims it
/// leaves on the next level, in the order of RFC-0010 'Tree levels'.
///
/// Its instances are the forests that have joined by level `ℓ`, in the order given to
/// [`prove_forests`]; [`forests`](Self::forests) names them. Instance `i` on `n_i` variables
/// ends the sumcheck at its point `r'_i`, the last `n_i` challenges of
/// [`point`](Self::point), with the evaluations `u_i, w_i` in the proof's final evaluations, and
/// its next claim is `Ṽ(ρ ‖ r'_i) = u_i + ρ·(w_i − u_i)`.
pub struct ProvenLevel<F> {
    root_challenges: Vec<F>,
    gamma: F,
    proof: SumcheckProof<F>,
    point: Vec<F>,
    rho: F,
    forests: Vec<usize>,
    next_claims: Vec<Claim<F>>,
}

impl<F> ProvenLevel<F> {
    /// The root challenges squeezed under `tree/{ℓ}/root/{c}` before `γ`, `c` from 0: as many as
    /// the largest `s` among the forests that join at this level, none when they are single
    /// trees or none joins. A joining forest of `2^s` trees has the last `s` as its root point.
    pub fn root_challenges(&self) -> &[F] {
        &self.root_challenges
    }

    /// The batching challenge `γ`, squeezed under `tree/{ℓ}/gamma`.
    pub fn gamma(&self) -> &F {
        &self.gamma
    }

    /// The messages: the rounds of degree 3, then `u_0, w_0, u_1, w_1, …` (RFC-0010 'Sumcheck
    /// messages', 'Tree levels').
    pub fn proof(&self) -> &SumcheckProof<F> {
        &self.proof
    }

    /// The messages, by value.
    pub fn into_proof(self) -> SumcheckProof<F> {
        self.proof
    }

    /// The challenges in round order, `r_1, …, r_{n_max}`: the level's point, its lowest variable
    /// first.
    pub fn point(&self) -> &[F] {
        &self.point
    }

    /// The line challenge `ρ`, squeezed under `tree/{ℓ}/line`: the lowest coordinate of every
    /// next claim.
    pub fn rho(&self) -> &F {
        &self.rho
    }

    /// The forests the level's instances prove, in instance order: each forest's index in the
    /// slice given to [`prove_forests`].
    pub fn forests(&self) -> &[usize] {
        &self.forests
    }

    /// Each instance's claim on its next level, in instance order: `Ṽ(ρ ‖ r'_i) = u_i + ρ·(w_i −
    /// u_i)`, on one variable more than the instance. After the last level these are the leaf
    /// claims.
    ///
    /// A claim on level `t` of a forest of `2^s` trees has `s + t` coordinates, the lowest first:
    /// the first `t` bind the position within a tree and the last `s` the tree index (RFC-0010
    /// 'Tree levels', 'Layout').
    pub fn next_claims(&self) -> &[Claim<F>] {
        &self.next_claims
    }
}

impl<F> ProvenLevel<F> {
    /// A level of [`prove_trees`] whose instances are all forests, whose next claims are all
    /// claims on products.
    fn of_forests(level: ProvenTreeLevel<F>) -> Self {
        let (root_challenges, gamma, proof, point, rho, trees, next_claims) = level.into_fields();
        Self {
            root_challenges,
            gamma,
            proof,
            point,
            rho,
            forests: trees,
            next_claims: next_claims
                .into_iter()
                .filter_map(|claim| match claim {
                    LevelClaim::Product(claim) => Some(claim),
                    // Only forests were given, so every claim is on a product.
                    LevelClaim::Fraction(_) => None,
                })
                .collect(),
        }
    }
}

/// Shows the shapes: the messages, challenges and claims are protocol-class data (05 §3).
impl<F> fmt::Debug for ProvenLevel<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProvenLevel")
            .field("root_challenges", &self.root_challenges.len())
            .field("proof", &self.proof)
            .field("forests", &self.forests)
            .finish_non_exhaustive()
    }
}

/// A proven batch of forests: each level's messages and challenges, and each forest's leaf claim,
/// which the next stage owes a check (RFC-0010 'Tree levels').
///
/// The levels prove nothing on their own: each reduces its claims to claims on the level below,
/// and the leaf claims, one per forest on its stacked leaves, are where the reduction stops. A
/// caller that drops them accepts what nothing checked.
#[must_use]
pub struct ProvenForests<F> {
    levels: Vec<ProvenLevel<F>>,
}

impl<F> ProvenForests<F> {
    /// The levels, from batch level 0 to `H − 1`, `H` the tallest forest's height.
    pub fn levels(&self) -> &[ProvenLevel<F>] {
        &self.levels
    }

    /// The leaf claims, one per forest in the order given: forest `i`'s claim on its leaf table
    /// of `2^{s_i + h_i}` entries, the next claims of the last level, where every forest is an
    /// instance.
    pub fn leaf_claims(&self) -> &[Claim<F>] {
        self.levels
            .last()
            .map_or(&[], |level| level.next_claims.as_slice())
    }

    /// Each level's messages, from level 0, and the leaf claims, by value: what a proof carries
    /// and what the next stage must check.
    #[must_use]
    pub fn into_parts(self) -> (Vec<SumcheckProof<F>>, Vec<Claim<F>>) {
        let mut leaf_claims = Vec::new();
        let mut proofs = Vec::with_capacity(self.levels.len());
        for level in self.levels {
            leaf_claims = level.next_claims;
            proofs.push(level.proof);
        }
        (proofs, leaf_claims)
    }
}

/// Shows the shapes: the messages, challenges and claims are protocol-class data (05 §3).
impl<F> fmt::Debug for ProvenForests<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProvenForests")
            .field("levels", &self.levels)
            .finish()
    }
}

/// A level of a tree as its two halves, the tables of the sumcheck one level up: its even
/// entries `V[2z]`, the table of `Ṽ(0 ‖ x)`, and its odd entries `V[2z + 1]`, that of `Ṽ(1 ‖ x)`.
pub(crate) struct Halves<B> {
    pub(crate) even: Vec<B>,
    pub(crate) odd: Vec<B>,
}

impl<B> Halves<B> {
    pub(crate) fn empty() -> Self {
        Self {
            even: Vec::new(),
            odd: Vec::new(),
        }
    }
}

/// The tree index's number of variables `s` of forest `index`, or the first shape check it
/// fails, in the order of [`SumcheckError`].
pub(crate) fn shape<F: PolyField>(
    forest: &Forest<'_, F>,
    index: usize,
) -> Result<usize, SumcheckError> {
    let len = forest.len();
    if !len.is_power_of_two() {
        return Err(SumcheckError::LeafCount { forest: index, len });
    }
    let vars = len.trailing_zeros() as usize;
    if forest.height == 0 || forest.height > vars {
        return Err(SumcheckError::Height {
            forest: index,
            height: forest.height,
            vars,
        });
    }
    Ok(vars - forest.height)
}

/// Checks the shapes of `forests` as [`prove_forests`] and [`prove_trees`] do, forest by forest:
/// a leaf count that is a power of two, then a height between 1 and `log2` of it. A caller that
/// moves the transcript before it hands the forests to [`prove_trees`], as the LogUp prover does
/// with its challenges and roots, checks them first.
///
/// # Errors
///
/// [`SumcheckError::LeafCount`] or [`SumcheckError::Height`] for the first forest that fails,
/// `forest` its position in `forests`.
pub fn check_forests<F: PolyField>(forests: &[Forest<'_, F>]) -> Result<(), SumcheckError> {
    for (i, forest) in forests.iter().enumerate() {
        shape(forest, i)?;
    }
    Ok(())
}

/// A forest's levels `1, …, h` from its leaves, at entries `1..=h`; entry 0, the roots, stays
/// empty.
pub(crate) fn build_levels<B: Field>(leaves: &[B], height: usize) -> Vec<Halves<B>> {
    let mut levels = Vec::with_capacity(height + 1);
    levels.resize_with(height + 1, Halves::empty);
    levels[height] = leaf_level(leaves);
    for t in (1..height).rev() {
        levels[t] = level_above(&levels[t + 1]);
    }
    levels
}

/// The halves of a forest's leaf level, `h`: the leaves de-interleaved, in one pass.
pub(crate) fn leaf_level<B: Copy>(leaves: &[B]) -> Halves<B> {
    let (even, odd) = leaves
        .chunks_exact(2)
        .map(|pair| (pair[0], pair[1]))
        .unzip();
    Halves { even, odd }
}

/// The halves of level `t` from those of level `t + 1`, in one pass: entry `z` of level `t` is
/// `V_t[z] = even_{t+1}[z]·odd_{t+1}[z]`, so `even_t[z] = even_{t+1}[2z]·odd_{t+1}[2z]` and
/// `odd_t[z] = even_{t+1}[2z + 1]·odd_{t+1}[2z + 1]`.
fn level_above<B: Field>(below: &Halves<B>) -> Halves<B> {
    let (even, odd) = below
        .even
        .chunks_exact(2)
        .zip(below.odd.chunks_exact(2))
        .map(|(e, o)| (e[0] * o[0], e[1] * o[1]))
        .unzip();
    Halves { even, odd }
}

/// `tree/{ℓ}/{suffix}`: `root/{c}`, `gamma`, `lambda` and `line`, the labels of a level that its
/// rounds do not squeeze (RFC-0014 'Transcript schedule', row 5).
pub(crate) fn label(level: u32, suffix: fmt::Arguments<'_>) -> String {
    format!("{}/{suffix}", Prefix::Tree(level))
}

/// Proves the levels of a batch of product forests on `transcript`, from the roots to the
/// leaves, and returns each level's messages and challenges and each forest's leaf claim
/// (RFC-0010 'Tree levels'; RFC-0014 'Transcript schedule', row 5).
///
/// Forest `i` has `2^{s_i}` trees of height `h_i`. With `H` the tallest height, the batch has the
/// levels `ℓ = 0, …, H − 1`, and forest `i` joins at level `H − h_i` with its root claim: from
/// level `ℓ` on it is an instance on `s_i + ℓ − (H − h_i)` variables. At level `ℓ`, on the
/// transcript, in this order:
///
/// 1. If forests of more than one tree join at `ℓ`, the root challenges `c = 0, …, s_max − 1` are
///    squeezed under `tree/{ℓ}/root/{c}`, `s_max` the largest `s_i` among the joining forests. A
///    joining forest's root point is the last `s_i` of them, in the order squeezed, and a single
///    tree's has no coordinate.
/// 2. The forests that have joined, in the order given, are the instances of a
///    [`prove_batch`](crate::prove_batch) of degree 3 under `tree/{ℓ}`: forest `i`, at its own
///    level `t`, proves `Σ_x eq(r_i, x)·Ṽ_{t+1}(0 ‖ x)·Ṽ_{t+1}(1 ‖ x)`, `r_i` its claim's point and
///    `eq(r_i, ·)` a public factor. `γ` is squeezed under `tree/{ℓ}/gamma`, smaller instances are
///    front-padded, and `u_0, w_0, u_1, w_1, …`, the two halves' values at each instance's point
///    `r'_i`, the last `n_i` challenges, are absorbed under `tree/{ℓ}/evals`.
/// 3. `ρ` is squeezed under `tree/{ℓ}/line`, and instance `i`'s next claim is
///    `Ṽ_{t+1}(ρ ‖ r'_i) = u_i + ρ·(w_i − u_i)`, `ρ` its lowest coordinate.
///
/// After level `H − 1` every forest stands at its leaves: the last level's next claims are the
/// leaf claims, one per forest in the order given, each on its stacked leaves.
///
/// The prover never needs a claim's value, and takes none. A root claim is the verifier's:
/// `Σ_y v_y·eq(r, y)` from the public root values `v_y`, zero for `SoftmaxRow` (RFC-0009), and a
/// forest whose trees' products are not the public values yields a proof the verifier rejects.
///
/// These are the levels of [`prove_trees`] for a batch of forests alone, which squeezes no `λ`.
///
/// # Algorithm
///
/// - **Levels.** Each forest's levels are built once, bottom-up, in the prime subfield, each as
///   its two halves, the even entries and the odd ones, which are the tables of the sumcheck one
///   level up. The leaves are de-interleaved once; above them,
///   `even_t[z] = even_{t+1}[2z]·odd_{t+1}[2z]` and `odd_t[z] = even_{t+1}[2z + 1]·odd_{t+1}[2z + 1]`,
///   both in one pass, one product per entry, and the roots are never built. That measures 1.5
///   times faster than building each level whole and de-interleaving it. Each half is allocated
///   once, at its size; together the levels take about twice the leaves' size, and each is
///   dropped once its sumcheck has run.
/// - **`eq`.** Each instance's `eq(r_i, ·)` is built by [`tessera_poly::eq_table`] for its level
///   and dropped after it.
///
/// Control flow depends on shapes only, never on a leaf (INV-ARCH-2). The leaves and every level
/// are secret-class values (05 §3): no debug form or error shows one.
///
/// # Errors
///
/// A [`SumcheckError`] when there is no forest, a forest's leaf count is not a power of two, or
/// its height is 0 or more than `log2` of its leaf count, in that order and forest by forest.
/// The inputs are checked before the transcript moves.
pub fn prove_forests<F, T>(
    transcript: &mut T,
    forests: &[Forest<'_, F>],
) -> Result<ProvenForests<F>, SumcheckError>
where
    F: PolyField,
    T: SumcheckTranscript<F> + ?Sized,
{
    if forests.is_empty() {
        return Err(SumcheckError::NoForests);
    }
    check_forests(forests)?;
    let trees = forests.iter().map(|&forest| Tree::Forest(forest)).collect();
    let proven = prove_trees(transcript, trees)?;
    let (levels, _) = proven.into_levels();
    Ok(ProvenForests {
        levels: levels.into_iter().map(ProvenLevel::of_forests).collect(),
    })
}
