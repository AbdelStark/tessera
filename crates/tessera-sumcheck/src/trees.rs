// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The levels of the leaf stage, where product forests and LogUp fraction trees run together
//! (RFC-0010 'Tree levels'; RFC-0008 'Fraction tree').

use core::alloc::Layout;
use core::fmt;

use tessera_codec::{Secret, Zeroize};
use tessera_poly::PolyField;

use crate::forest::{Halves, build_levels, label, leaf_level, shape};
use crate::level::{LevelInstance, LevelNumerators, LevelTables, product_claim, prove_level};
use crate::{Claim, Forest, Prefix, SumcheckError, SumcheckProof, SumcheckTranscript};

/// The numerators of a fraction tree's leaves.
pub enum LeafNumerators<'a, B> {
    /// Every numerator is 1, as in a LogUp lookup tree, whose leaves are `(1, α − a_i)`. They are
    /// never stored: the level above adds denominators where it would multiply by them, and the
    /// last level's sumcheck reads no numerator and sends `N_0 = N_1 = 1`.
    Unit,
    /// One numerator per leaf in the prime subfield, as a LogUp table tree's `−m_j`.
    Base(&'a [B]),
}

impl<'a, B: Zeroize> LeafNumerators<'a, B> {
    /// Secret-class numerators in their [`Secret`], as a LogUp table tree's `−m_j`, which count
    /// how often each row is looked up: [`LeafNumerators::Base`] on the values, which the tree
    /// reads to build its levels, and its levels clear themselves when they drop (05 §3 OBS-1,
    /// OBS-6).
    #[allow(clippy::disallowed_methods)]
    pub fn secret(values: &'a Secret<Vec<B>>) -> Self {
        Self::Base(values.expose_secret())
    }
}

/// Shows the kind and the length, never a numerator (05 §3).
impl<B> fmt::Debug for LeafNumerators<'_, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unit => f.write_str("Unit"),
            Self::Base(values) => f.debug_tuple("Base").field(&values.len()).finish(),
        }
    }
}

/// A level's numerators, as its two halves: of 1, in the prime subfield, or in `F`.
enum Numerators<F: PolyField> {
    Unit,
    Base(Halves<F::Base>),
    Ext(Halves<F>),
}

/// A level of a fraction tree: its numerators and its denominators, each as its two halves.
struct FractionLevel<F: PolyField> {
    numerators: Numerators<F>,
    denominators: Halves<F>,
}

impl<F: PolyField> FractionLevel<F> {
    /// A level not built, or dropped once its sumcheck has run.
    fn empty() -> Self {
        Self {
            numerators: Numerators::Ext(Halves::empty()),
            denominators: Halves::empty(),
        }
    }
}

/// `n_L·d_R + n_R·d_L`, the numerator of a node whose children are `(n_L, d_L)` and `(n_R, d_R)`,
/// for each kind of numerator: `d_R + d_L` for numerators of 1, two products by a base element
/// for numerators in the prime subfield, two products in `F` otherwise.
trait Numerator<F: PolyField>: Copy {
    fn cross(left: Self, d_left: F, right: Self, d_right: F) -> F;
}

/// A numerator of 1.
#[derive(Clone, Copy)]
struct One;

impl<F: PolyField> Numerator<F> for One {
    #[inline(always)]
    fn cross(_: Self, d_left: F, _: Self, d_right: F) -> F {
        d_right + d_left
    }
}

/// A numerator in the prime subfield.
#[derive(Clone, Copy)]
struct InBase<B>(B);

impl<F: PolyField> Numerator<F> for InBase<F::Base> {
    #[inline(always)]
    fn cross(left: Self, d_left: F, right: Self, d_right: F) -> F {
        d_right.mul_by_base(left.0) + d_left.mul_by_base(right.0)
    }
}

/// A numerator in `F`.
#[derive(Clone, Copy)]
struct InExt<F>(F);

impl<F: PolyField> Numerator<F> for InExt<F> {
    #[inline(always)]
    fn cross(left: Self, d_left: F, right: Self, d_right: F) -> F {
        left.0 * d_right + right.0 * d_left
    }
}

/// The level above the halves `(n_even, n_odd)` and `(d_even, d_odd)` of level `t + 1`, in one
/// pass. Entry `z` of level `t` combines entries `2z` (left) and `2z + 1` (right) of level
/// `t + 1`, which are `(n_even[z], d_even[z])` and `(n_odd[z], d_odd[z])`; its halves hold its
/// entries `2y` and `2y + 1`.
fn combine<F: PolyField, N: Numerator<F>>(
    n_even: impl Iterator<Item = N>,
    n_odd: impl Iterator<Item = N>,
    denominators: &Halves<F>,
) -> FractionLevel<F> {
    let half = denominators.even.len() / 2;
    let mut numerators = Halves::zeroed(half);
    let mut products = Halves::zeroed(half);
    let mut n_even = n_even;
    let mut n_odd = n_odd;
    let outputs = (numerators.even.iter_mut().zip(numerators.odd.iter_mut()))
        .zip(products.even.iter_mut().zip(products.odd.iter_mut()));
    let inputs = denominators
        .even
        .chunks_exact(2)
        .zip(denominators.odd.chunks_exact(2));
    for (((n0, n1), (p0, p1)), (de, d_o)) in outputs.zip(inputs) {
        // Entries 2y and 2y + 1 of the level: left children de, right children d_o.
        let (Some(l0), Some(r0), Some(l1), Some(r1)) =
            (n_even.next(), n_odd.next(), n_even.next(), n_odd.next())
        else {
            break;
        };
        *n0 = N::cross(l0, de[0], r0, d_o[0]);
        *p0 = de[0] * d_o[0];
        *n1 = N::cross(l1, de[1], r1, d_o[1]);
        *p1 = de[1] * d_o[1];
    }
    FractionLevel {
        numerators: Numerators::Ext(numerators),
        denominators: products,
    }
}

/// The level above `below`, whatever its numerators: three products per entry, one when its
/// numerators are 1, of which two by base elements when they are in the prime subfield.
fn level_above_fractions<F: PolyField>(below: &FractionLevel<F>) -> FractionLevel<F> {
    let d = &below.denominators;
    match &below.numerators {
        Numerators::Unit => {
            let ones = || core::iter::repeat_n(One, d.even.len());
            combine::<F, One>(ones(), ones(), d)
        }
        Numerators::Base(n) => combine::<F, InBase<F::Base>>(
            n.even.iter().map(|&b| InBase(b)),
            n.odd.iter().map(|&b| InBase(b)),
            d,
        ),
        Numerators::Ext(n) => combine::<F, InExt<F>>(
            n.even.iter().map(|&x| InExt(x)),
            n.odd.iter().map(|&x| InExt(x)),
            d,
        ),
    }
}

/// The root `(n, d)` from level 1, whose halves hold one entry each.
fn root_of<F: PolyField>(level: &FractionLevel<F>) -> (F, F) {
    let d = &level.denominators;
    let (Some(&d_left), Some(&d_right)) = (d.even.first(), d.odd.first()) else {
        return (F::ZERO, F::ZERO);
    };
    let n = match &level.numerators {
        Numerators::Unit => d_right + d_left,
        Numerators::Base(n) => match (n.even.first(), n.odd.first()) {
            (Some(&l), Some(&r)) => {
                <InBase<F::Base> as Numerator<F>>::cross(InBase(l), d_left, InBase(r), d_right)
            }
            _ => F::ZERO,
        },
        Numerators::Ext(n) => match (n.even.first(), n.odd.first()) {
            (Some(&l), Some(&r)) => {
                <InExt<F> as Numerator<F>>::cross(InExt(l), d_left, InExt(r), d_right)
            }
            _ => F::ZERO,
        },
    };
    (n, d_left * d_right)
}

/// A LogUp fraction tree of height `h` (RFC-0008 'Fraction tree'), built from its `2^h` leaves
/// `(n_x, d_x)`, with all its levels: level `h` holds the leaves, and node `x` of level `ℓ`
/// combines entries `2x` (left) and `2x + 1` (right) of level `ℓ + 1`,
/// `(n, d) = (n_L·d_R + n_R·d_L, d_L·d_R)`, so the pair bit is the lowest variable as in a product
/// tree. The root `(n, d)` has `d` the product of the leaves' denominators and, when `d ≠ 0`,
/// `n/d = Σ_x n_x/d_x`.
///
/// [`prove_trees`] proves it level by level beside other fraction trees and product forests.
///
/// # Layout
///
/// Each level `ℓ ≥ 1` is stored as its two halves, the even and the odd entries, which are the
/// tables `N(0 ‖ x)`, `N(1 ‖ x)`, `D(0 ‖ x)` and `D(1 ‖ x)` of the sumcheck one level up. The leaf
/// numerators stay in the prime subfield, or are not stored at all when they are 1; every other
/// numerator and every denominator is in `F`. The level above the leaves costs one product per
/// node with numerators of 1, a product and two products by base elements with base numerators,
/// and every level above three products per node. The root is computed once and level 0 is never
/// stored.
///
/// The leaves and every level are secret-class values (05 §3): the debug form shows the height,
/// never a node, and the root is the message the protocol sends.
pub struct FractionTree<F: PolyField> {
    height: usize,
    root: (F, F),
    /// Level `t` at entry `t`, for `t = 1, …, h`; entry 0 stays empty, and each level is dropped
    /// once its sumcheck has run.
    levels: Vec<FractionLevel<F>>,
}

impl<F: PolyField> FractionTree<F> {
    /// The tree of height `height` whose leaf `x < 2^height` is `(n_x, denominators(x))`, `n_x`
    /// from `numerators`, building every level bottom-up. `denominators` is called once per
    /// leaf, in order.
    ///
    /// # Errors
    ///
    /// [`SumcheckError::FractionHeight`] when `2^height` leaves cannot be counted or allocated,
    /// and [`SumcheckError::FractionNumerators`] when numerators in the prime subfield are not
    /// `2^height`, in that order; before `denominators` is called.
    pub fn new(
        height: usize,
        numerators: LeafNumerators<'_, F::Base>,
        mut denominators: impl FnMut(usize) -> F,
    ) -> Result<Self, SumcheckError> {
        let leaves = u32::try_from(height)
            .ok()
            .and_then(|h| 1usize.checked_shl(h))
            .filter(|&leaves| Layout::array::<F>(leaves).is_ok())
            .ok_or(SumcheckError::FractionHeight { height })?;
        if let LeafNumerators::Base(values) = numerators
            && values.len() != leaves
        {
            return Err(SumcheckError::FractionNumerators {
                len: values.len(),
                expected: leaves,
            });
        }
        if height == 0 {
            let numerator = match numerators {
                LeafNumerators::Unit => F::ONE,
                LeafNumerators::Base(values) => values.first().map_or(F::ZERO, |&b| F::from(b)),
            };
            return Ok(Self {
                height,
                root: (numerator, denominators(0)),
                levels: Vec::new(),
            });
        }
        let mut halves = Halves::zeroed(leaves / 2);
        let outputs = halves.even.iter_mut().zip(halves.odd.iter_mut());
        for (x, (even, odd)) in outputs.enumerate() {
            *even = denominators(2 * x);
            *odd = denominators(2 * x + 1);
        }
        let leaf_numerators = match numerators {
            LeafNumerators::Unit => Numerators::Unit,
            LeafNumerators::Base(values) => Numerators::Base(leaf_level(values)),
        };
        let mut levels = Vec::with_capacity(height + 1);
        levels.resize_with(height + 1, FractionLevel::empty);
        levels[height] = FractionLevel {
            numerators: leaf_numerators,
            denominators: halves,
        };
        for t in (1..height).rev() {
            levels[t] = level_above_fractions(&levels[t + 1]);
        }
        let root = root_of(&levels[1]);
        Ok(Self {
            height,
            root,
            levels,
        })
    }

    /// The height `h`: the number of levels below the root, `log2` of the leaf count.
    pub fn height(&self) -> usize {
        self.height
    }

    /// The root `(n, d)`: the claims the tree enters its first level with, on no variable, and
    /// its leaf claims when its height is 0.
    pub fn root(&self) -> (F, F) {
        self.root
    }
}

/// Shows the height, never a node (05 §3).
impl<F: PolyField> fmt::Debug for FractionTree<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FractionTree")
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

/// One instance of the leaf stage's levels (RFC-0010 'Tree levels'): a product forest, or a LogUp
/// fraction tree.
pub enum Tree<'a, F: PolyField> {
    /// `2^s` product trees of one height, stacked; a single tree has `s = 0`.
    Forest(Forest<'a, F>),
    /// A fraction tree, always single.
    Fraction(FractionTree<F>),
}

/// Shows the kind and the shape, never a leaf (05 §3).
impl<F: PolyField> fmt::Debug for Tree<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Forest(forest) => f.debug_tuple("Forest").field(forest).finish(),
            Self::Fraction(tree) => f.debug_tuple("Fraction").field(tree).finish(),
        }
    }
}

/// The two claims on one level of a fraction tree, at one point: `Ñ(point) = numerator` on the
/// level's numerators and `D̃(point) = denominator` on its denominators. The point lists its
/// coordinates lowest variable first, the pair bit of the level below first.
///
/// The values are protocol-class data, functions of secret-class nodes (05 §3): the debug form
/// shows the number of coordinates only.
#[derive(Clone, PartialEq, Eq)]
pub struct FractionClaim<F> {
    /// The point, its lowest variable first.
    pub point: Vec<F>,
    /// The claimed value of the level's numerators' multilinear extension at the point.
    pub numerator: F,
    /// The claimed value of the level's denominators' multilinear extension at the point.
    pub denominator: F,
}

/// Shows the number of coordinates, never a value (05 §3).
impl<F> fmt::Debug for FractionClaim<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FractionClaim")
            .field("vars", &self.point.len())
            .finish_non_exhaustive()
    }
}

/// The claims an instance of a tree level holds on a level: a forest's or a fraction tree's.
#[derive(Clone, PartialEq, Eq)]
pub enum LevelClaim<F> {
    /// A product forest's claim on its stacked level.
    Product(Claim<F>),
    /// A fraction tree's claims on its numerators and denominators.
    Fraction(FractionClaim<F>),
}

impl<F> LevelClaim<F> {
    /// The point, its lowest variable first.
    pub fn point(&self) -> &[F] {
        match self {
            Self::Product(claim) => &claim.point,
            Self::Fraction(claim) => &claim.point,
        }
    }
}

/// Shows the kind and the number of coordinates, never a value (05 §3).
impl<F> fmt::Debug for LevelClaim<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Product(claim) => f.debug_tuple("Product").field(claim).finish(),
            Self::Fraction(claim) => f.debug_tuple("Fraction").field(claim).finish(),
        }
    }
}

/// One level `ℓ` of [`prove_trees`]: its challenges, its messages and the claims it leaves on
/// the next level, in the order of RFC-0010 'Tree levels'.
///
/// Its instances are the trees that have joined by level `ℓ`, in schedule order;
/// [`trees`](Self::trees) names them. Instance `i` on `n_i` variables ends the sumcheck at its
/// point `r'_i`, the last `n_i` challenges of [`point`](Self::point).
pub struct ProvenTreeLevel<F> {
    root_challenges: Vec<F>,
    gamma: F,
    lambda: Option<F>,
    proof: SumcheckProof<F>,
    point: Vec<F>,
    rho: F,
    trees: Vec<usize>,
    next_claims: Vec<LevelClaim<F>>,
}

/// A level's fields, in declaration order but `λ`, for the forests' view of it.
type LevelFields<F> = (
    Vec<F>,
    F,
    SumcheckProof<F>,
    Vec<F>,
    F,
    Vec<usize>,
    Vec<LevelClaim<F>>,
);

impl<F> ProvenTreeLevel<F> {
    /// The root challenges squeezed under `tree/{ℓ}/root/{c}` before `γ`, `c` from 0: as many as
    /// the largest `s` among the forests that join at this level, none when only single trees
    /// and fraction trees join. A joining forest of `2^s` trees has the last `s` as its root
    /// point.
    pub fn root_challenges(&self) -> &[F] {
        &self.root_challenges
    }

    /// The batching challenge `γ`, squeezed under `tree/{ℓ}/gamma`.
    pub fn gamma(&self) -> &F {
        &self.gamma
    }

    /// The challenge `λ`, squeezed under `tree/{ℓ}/lambda` right after `γ` at a level where a
    /// fraction tree is present; `None` at a level of forests alone.
    pub fn lambda(&self) -> Option<&F> {
        self.lambda.as_ref()
    }

    /// The messages: the rounds of degree 3, then the evaluations, instance by instance: `u, w`
    /// for a forest, `N_0, N_1, D_0, D_1` for a fraction tree.
    pub fn proof(&self) -> &SumcheckProof<F> {
        &self.proof
    }

    /// The messages, by value.
    pub fn into_proof(self) -> SumcheckProof<F> {
        self.proof
    }

    /// The challenges in round order: the level's point, its lowest variable first.
    pub fn point(&self) -> &[F] {
        &self.point
    }

    /// The line challenge `ρ`, squeezed under `tree/{ℓ}/line`: the lowest coordinate of every
    /// next claim.
    pub fn rho(&self) -> &F {
        &self.rho
    }

    /// The trees the level's instances prove, in instance order: each tree's position in the
    /// schedule given to [`prove_trees`].
    pub fn trees(&self) -> &[usize] {
        &self.trees
    }

    /// Each instance's claims on its next level, in instance order, at `ρ ‖ r'_i`:
    /// `u_i + ρ·(w_i − u_i)` for a forest; `N_0 + ρ·(N_1 − N_0)` and `D_0 + ρ·(D_1 − D_0)` for a
    /// fraction tree.
    pub fn next_claims(&self) -> &[LevelClaim<F>] {
        &self.next_claims
    }

    /// The fields, `λ` left out: a level of forests alone has none.
    pub(crate) fn into_fields(self) -> LevelFields<F> {
        (
            self.root_challenges,
            self.gamma,
            self.proof,
            self.point,
            self.rho,
            self.trees,
            self.next_claims,
        )
    }
}

/// Shows the shapes: the messages, challenges and claims are protocol-class data (05 §3).
impl<F> fmt::Debug for ProvenTreeLevel<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProvenTreeLevel")
            .field("root_challenges", &self.root_challenges.len())
            .field("lambda", &self.lambda.is_some())
            .field("proof", &self.proof)
            .field("trees", &self.trees)
            .finish_non_exhaustive()
    }
}

/// The proven levels of a leaf stage's trees, and each tree's leaf claims, which later checks
/// owe: a forest's go to the stage that consumes them, a fraction tree's to LogUp's leaf checks
/// (RFC-0010 'Tree levels', RFC-0008 'Fraction tree').
///
/// The levels prove nothing on their own: each reduces its claims to claims on the level below,
/// and a caller that drops the leaf claims accepts what nothing checked.
#[must_use]
pub struct ProvenTrees<F> {
    levels: Vec<ProvenTreeLevel<F>>,
    leaf_claims: Vec<LevelClaim<F>>,
}

impl<F> ProvenTrees<F> {
    /// The levels, from batch level 0 to `H − 1`, `H` the tallest tree's height.
    pub fn levels(&self) -> &[ProvenTreeLevel<F>] {
        &self.levels
    }

    /// The leaf claims, one per tree in schedule order: the next claims of the last level, where
    /// every tree of height at least 1 is an instance, and for a fraction tree of height 0, which
    /// joins no level, its root on no variable.
    pub fn leaf_claims(&self) -> &[LevelClaim<F>] {
        &self.leaf_claims
    }

    /// The levels and the leaf claims, by value.
    #[must_use]
    pub fn into_levels(self) -> (Vec<ProvenTreeLevel<F>>, Vec<LevelClaim<F>>) {
        (self.levels, self.leaf_claims)
    }

    /// Each level's messages, from level 0, and the leaf claims, by value: what a proof carries
    /// and what later checks owe.
    #[must_use]
    pub fn into_parts(self) -> (Vec<SumcheckProof<F>>, Vec<LevelClaim<F>>) {
        let proofs = self.levels.into_iter().map(|level| level.proof).collect();
        (proofs, self.leaf_claims)
    }
}

/// Shows the shapes: the messages, challenges and claims are protocol-class data (05 §3).
impl<F> fmt::Debug for ProvenTrees<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProvenTrees")
            .field("levels", &self.levels)
            .field("leaf_claims", &self.leaf_claims)
            .finish()
    }
}

/// What a tree holds while its levels are proven.
enum Kind<F: PolyField> {
    /// A forest of `2^s` trees, `s = trees_log`, its levels `1, …, h` at entries `1..=h`.
    Forest {
        trees_log: usize,
        levels: Vec<Halves<F::Base>>,
    },
    Fraction(FractionTree<F>),
}

/// A tree while its levels are proven.
struct State<F: PolyField> {
    kind: Kind<F>,
    /// `H − h`: the batch level at which it joins.
    join: usize,
    /// The point of its current claims, once it has joined.
    point: Vec<F>,
    /// Its current claims, once it has run a level; a fraction tree of height 0 starts with its
    /// root.
    claim: Option<LevelClaim<F>>,
}

impl<F: PolyField> State<F> {
    /// The tree index's variables `s`: 0 for a fraction tree.
    fn trees_log(&self) -> usize {
        match &self.kind {
            Kind::Forest { trees_log, .. } => *trees_log,
            Kind::Fraction(_) => 0,
        }
    }

    /// The sum the level proves for this tree at its own level `t`: at the level it joins, a
    /// forest's root claim at its root point and a fraction tree's root `p + λ·q`; after, the
    /// claims the level above left, `p + λ·q` for a fraction tree with the level's `λ`.
    fn level_claim(&self, t: usize, lambda: F) -> F {
        match (&self.claim, &self.kind) {
            (Some(LevelClaim::Product(claim)), _) => claim.value,
            (Some(LevelClaim::Fraction(claim)), _) => claim.numerator + lambda * claim.denominator,
            (None, Kind::Forest { levels, .. }) => levels.get(t + 1).map_or(F::ZERO, |below| {
                product_claim(&self.point, &below.even, &below.odd)
            }),
            (None, Kind::Fraction(tree)) => tree.root.0 + lambda * tree.root.1,
        }
    }

    /// The tree's level `t` as an instance of the level's sumcheck, proving `claim`, with the
    /// level's `λ` for a fraction tree.
    fn level_instance(&self, t: usize, claim: F, lambda: F) -> LevelInstance<'_, F> {
        let tables = match &self.kind {
            Kind::Forest { levels, .. } => {
                let (even, odd) = levels
                    .get(t + 1)
                    .map_or((&[][..], &[][..]), |b| (&b.even[..], &b.odd[..]));
                LevelTables::Product { even, odd }
            }
            Kind::Fraction(tree) => {
                let below = tree.levels.get(t + 1);
                let numerators = match below.map(|b| &b.numerators) {
                    Some(Numerators::Base(n)) => LevelNumerators::Base {
                        even: &n.even,
                        odd: &n.odd,
                    },
                    Some(Numerators::Ext(n)) => LevelNumerators::Ext {
                        even: &n.even,
                        odd: &n.odd,
                    },
                    Some(Numerators::Unit) | None => LevelNumerators::Unit,
                };
                let (d_even, d_odd) = below.map_or((&[][..], &[][..]), |b| {
                    (&b.denominators.even[..], &b.denominators.odd[..])
                });
                LevelTables::Fraction {
                    numerators,
                    d_even,
                    d_odd,
                    lambda,
                }
            }
        };
        LevelInstance {
            point: &self.point,
            claim,
            tables,
        }
    }

    /// Drops its level `t + 1` once the sumcheck of its level `t` has run.
    fn drop_level(&mut self, t: usize) {
        match &mut self.kind {
            Kind::Forest { levels, .. } => {
                if let Some(level) = levels.get_mut(t + 1) {
                    *level = Halves::empty();
                }
            }
            Kind::Fraction(tree) => {
                if let Some(level) = tree.levels.get_mut(t + 1) {
                    *level = FractionLevel::empty();
                }
            }
        }
    }
}

/// Proves the levels of a leaf stage's trees on `transcript`, product forests and LogUp fraction
/// trees in one schedule, from the roots to the leaves, and returns each level's messages and
/// challenges and each tree's leaf claims (RFC-0010 'Tree levels'; RFC-0014 'Transcript
/// schedule', row 5).
///
/// `trees` is the schedule: instances of both kinds are numbered from 0 in its order. A forest
/// has `2^s` trees of height `h ≥ 1`, a fraction tree is single, of height `h ≥ 0`. With `H` the
/// tallest height, the batch has the levels `ℓ = 0, …, H − 1`, and a tree of height `h` joins at
/// level `H − h`: from level `ℓ` on it is an instance on `s + ℓ − (H − h)` variables. A fraction
/// tree of height 0 joins no level. At level `ℓ`, on the transcript, in this order:
///
/// 1. If forests of more than one tree join at `ℓ`, the root challenges `c = 0, …, s_max − 1` are
///    squeezed under `tree/{ℓ}/root/{c}`, and a joining forest's root point is the last `s` of
///    them, in the order squeezed. Single trees and fraction trees join on no variable.
/// 2. `γ` is squeezed under `tree/{ℓ}/gamma`, then, when a fraction tree is present, `λ` under
///    `tree/{ℓ}/lambda`.
/// 3. The trees that have joined, in schedule order, are the instances of a sumcheck of degree 3
///    under `tree/{ℓ}`, front-padded, weighted by `γ^i`, each with `eq(r_i, ·)` a public factor:
///    a forest at its own level `t` proves `Σ_x eq(r_i, x)·Ṽ_{t+1}(0 ‖ x)·Ṽ_{t+1}(1 ‖ x)`; a
///    fraction tree proves `Σ_x eq(r_i, x)·(N(0 ‖ x)·D(1 ‖ x) + N(1 ‖ x)·D(0 ‖ x) +
///    λ·D(0 ‖ x)·D(1 ‖ x))`, `N` and `D` its level `t + 1`'s numerators and denominators. Its
///    evaluations at `r'_i`, the last `n_i` challenges, are `u, w` for a forest and `N_0, N_1,
///    D_0, D_1` for a fraction tree, absorbed under `tree/{ℓ}/evals`.
/// 4. `ρ` is squeezed under `tree/{ℓ}/line`, and the next claims are at `ρ ‖ r'_i`:
///    `u + ρ·(w − u)` for a forest; `N_0 + ρ·(N_1 − N_0)` and `D_0 + ρ·(D_1 − D_0)` for a fraction
///    tree.
///
/// A level of forests alone squeezes no `λ`, so a batch of forests alone runs exactly as
/// [`prove_forests`](crate::prove_forests). After level `H − 1` every tree stands at its leaves.
///
/// The prover takes no claim. A forest's root claim is the verifier's, from public root values,
/// and a fraction tree's is its root, which the caller sends before the first level
/// (`logup/roots`). The prover computes both itself, for the level kernel, which needs each
/// instance's claim.
///
/// # Algorithm
///
/// - **Forests** build their levels as [`prove_forests`](crate::prove_forests) does, in the prime
///   subfield; **fraction trees** come built, by [`FractionTree::new`].
/// - **Each level** runs the split-`eq` kernel of `level.rs`:
///   - `eq(r_i, ·)` is never built or bound: it is a scalar, the current variable's line, and two
///     tables of about `√N` entries rebuilt each round;
///   - each round sums `v(0)` and `v(2)` over the pairs and recovers `v(1)` from the instance's
///     claim (Gruen);
///   - the messages are the batch prover's over `[eq, …]` instances.
/// - **A fraction tree's level** reads `N(0)·D(1) + N(1)·D(0) + λ·D(0)·D(1)` from its four
///   tables on each pair. When level `t + 1` holds leaves with numerators of 1, it reads no
///   numerator, and its evaluations `N_0 = N_1 = 1` come from no table.
/// - Each tree level is dropped once its sumcheck has run.
///
/// Control flow depends on shapes only, never on a leaf (INV-ARCH-2). Leaves, levels and nodes
/// are secret-class values (05 §3): no debug form or error shows one.
///
/// # Errors
///
/// A [`SumcheckError`] when a forest's leaf count is not a power of two or its height is 0 or
/// more than `log2` of its leaf count, in that order and forest by forest, `forest` its position
/// in the schedule; before the transcript moves.
pub fn prove_trees<F, T>(
    transcript: &mut T,
    trees: Vec<Tree<'_, F>>,
) -> Result<ProvenTrees<F>, SumcheckError>
where
    F: PolyField,
    T: SumcheckTranscript<F> + ?Sized,
{
    let mut trees_logs = Vec::with_capacity(trees.len());
    for (i, tree) in trees.iter().enumerate() {
        trees_logs.push(match tree {
            Tree::Forest(forest) => shape(forest, i)?,
            Tree::Fraction(_) => 0,
        });
    }
    let height = |tree: &Tree<'_, F>| match tree {
        Tree::Forest(forest) => forest.height,
        Tree::Fraction(fraction) => fraction.height,
    };
    let tallest = trees.iter().map(height).max().unwrap_or(0);
    let mut states: Vec<State<F>> = trees
        .into_iter()
        .zip(trees_logs)
        .map(|(tree, trees_log)| {
            let join = tallest - height(&tree);
            match tree {
                Tree::Forest(forest) => State {
                    kind: Kind::Forest {
                        trees_log,
                        levels: build_levels(forest.leaves, forest.height),
                    },
                    join,
                    point: Vec::new(),
                    claim: None,
                },
                Tree::Fraction(fraction) => {
                    let (numerator, denominator) = fraction.root;
                    let claim = (fraction.height == 0).then(|| {
                        LevelClaim::Fraction(FractionClaim {
                            point: Vec::new(),
                            numerator,
                            denominator,
                        })
                    });
                    State {
                        kind: Kind::Fraction(fraction),
                        join,
                        point: Vec::new(),
                        claim,
                    }
                }
            }
        })
        .collect();

    let mut levels = Vec::with_capacity(tallest);
    for level in 0..tallest {
        let tag = level as u32;
        let s_max = states
            .iter()
            .filter(|state| state.join == level)
            .map(State::trees_log)
            .max()
            .unwrap_or(0);
        let root_challenges: Vec<F> = (0..s_max)
            .map(|c| transcript.challenge(&label(tag, format_args!("root/{c}"))))
            .collect();
        for state in states.iter_mut().filter(|state| state.join == level) {
            state.point = root_challenges[s_max - state.trees_log()..].to_vec();
        }

        let active: Vec<usize> = (0..states.len())
            .filter(|&i| states[i].join <= level)
            .collect();
        let gamma = transcript.challenge(&label(tag, format_args!("gamma")));
        let fractions = active
            .iter()
            .any(|&i| matches!(states[i].kind, Kind::Fraction(_)));
        let lambda = fractions.then(|| transcript.challenge(&label(tag, format_args!("lambda"))));

        let lambda_value = lambda.unwrap_or(F::ZERO);
        let instances: Vec<LevelInstance<'_, F>> = active
            .iter()
            .map(|&i| {
                let state = &states[i];
                let t = level - state.join;
                state.level_instance(t, state.level_claim(t, lambda_value), lambda_value)
            })
            .collect();
        let proven = prove_level(transcript, Prefix::Tree(tag), &instances, gamma);
        drop(instances);
        let rho = transcript.challenge(&label(tag, format_args!("line")));

        let mut next_claims = Vec::with_capacity(active.len());
        for (k, &i) in active.iter().enumerate() {
            let own = proven.instance_point(k).unwrap_or_default();
            let evals = proven.instance_evals(k).unwrap_or_default();
            let mut point = Vec::with_capacity(own.len() + 1);
            point.push(rho);
            point.extend_from_slice(own);
            let on_line = |at: usize| {
                let (zero, one) = (
                    evals.get(at).copied().unwrap_or(F::ZERO),
                    evals.get(at + 1).copied().unwrap_or(F::ZERO),
                );
                zero + rho * (one - zero)
            };
            let state = &mut states[i];
            let claim = match state.kind {
                Kind::Forest { .. } => LevelClaim::Product(Claim {
                    point: point.clone(),
                    value: on_line(0),
                }),
                Kind::Fraction(_) => LevelClaim::Fraction(FractionClaim {
                    point: point.clone(),
                    numerator: on_line(0),
                    denominator: on_line(2),
                }),
            };
            state.drop_level(level - state.join);
            state.point = point;
            state.claim = Some(claim.clone());
            next_claims.push(claim);
        }
        let point = proven.point().to_vec();
        levels.push(ProvenTreeLevel {
            root_challenges,
            gamma,
            lambda,
            proof: proven.into_proof(),
            point,
            rho,
            trees: active,
            next_claims,
        });
    }
    // Every tree of height at least 1 is an instance of the last level, and a fraction tree of
    // height 0 holds its root: every state has its claims.
    let leaf_claims = states.into_iter().filter_map(|state| state.claim).collect();
    Ok(ProvenTrees {
        levels,
        leaf_claims,
    })
}

#[cfg(test)]
mod tests {
    use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};

    use super::{FractionTree, LeafNumerators};

    type Fp = Goldilocks;
    type Fp2 = GoldilocksExt2;

    fn fp2(a: u64, b: u64) -> Fp2 {
        Fp2::new(Fp::from_u64_reduced(a), Fp::from_u64_reduced(b))
    }

    /// The root from its definition: `d = Π d_x` and `n = Σ_x n_x·Π_{y ≠ x} d_y`.
    fn root(numerators: &[Fp2], denominators: &[Fp2]) -> (Fp2, Fp2) {
        let d = denominators.iter().fold(Fp2::ONE, |p, &x| p * x);
        let n = (0..numerators.len()).fold(Fp2::ZERO, |sum, x| {
            let others = denominators
                .iter()
                .enumerate()
                .filter(|&(y, _)| y != x)
                .fold(Fp2::ONE, |p, (_, &d)| p * d);
            sum + numerators[x] * others
        });
        (n, d)
    }

    #[test]
    fn roots_are_the_sums_of_the_leaves_fractions() {
        for height in 0..=6 {
            let leaves = 1usize << height;
            let denominators: Vec<Fp2> = (0..leaves as u64)
                .map(|x| fp2(3 * x + 7, x * x + 1))
                .collect();
            let numerators: Vec<Fp> = (0..leaves as u64)
                .map(|x| Fp::from_u64_reduced(5 * x + 2))
                .collect();
            let unit = FractionTree::<Fp2>::new(height, LeafNumerators::Unit, |x| denominators[x])
                .expect("a tree");
            assert_eq!(unit.height(), height);
            assert_eq!(
                unit.root(),
                root(&vec![Fp2::ONE; leaves], &denominators),
                "height {height}, numerators of 1"
            );
            let base = FractionTree::<Fp2>::new(height, LeafNumerators::Base(&numerators), |x| {
                denominators[x]
            })
            .expect("a tree");
            let lifted: Vec<Fp2> = numerators.iter().map(|&n| Fp2::from(n)).collect();
            assert_eq!(
                base.root(),
                root(&lifted, &denominators),
                "height {height}, base numerators"
            );
        }
    }
}
