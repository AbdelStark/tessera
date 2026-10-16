// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The LogUp part of the leaf stage: challenges, trees, roots, levels, values and leaf claims
//! (RFC-0008 'Fraction tree', RFC-0010 'Tree levels', RFC-0014 'Transcript schedule' rows 4 and
//! 5).

use core::fmt;

use tessera_field::Field;
use tessera_poly::{PolyField, evaluate_base};
use tessera_sumcheck::{
    Claim, Forest, FractionClaim, FractionTree, LeafNumerators, LevelClaim, ProvenTreeLevel,
    SumcheckTranscript, Tree, check_forests, prove_trees,
};

use crate::{LogUpError, LogUpInstance, Tuples};

/// One tree of the leaf stage, as a schedule names it: instance `k`'s lookup tree or table tree,
/// or forest `k`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TreeId {
    /// The lookup tree of instance `k`, counted from 0 in the order of the instances.
    Lookup(usize),
    /// The table tree of instance `k`.
    Table(usize),
    /// Product forest `k`, counted from 0 in the order of the forests.
    Forest(usize),
}

impl TreeId {
    /// The schedule of `instances` LogUp instances alone: instance by instance, its lookup tree,
    /// then its table tree.
    pub fn paired(instances: usize) -> Vec<TreeId> {
        (0..instances)
            .flat_map(|k| [Self::Lookup(k), Self::Table(k)])
            .collect()
    }
}

/// The claims one instance's leaf checks leave for later stages (RFC-0008 'Fraction tree', 'Leaf
/// claims'), each on a polynomial of the instance at its tree's leaf point:
///
/// - `kẽy(ρ_a) = α − q_a − β·v` on the lookup-side key polynomial, virtual, of `2^{h_a}`
///   entries whose padding slots hold row 0's key: it goes to the stage that produces the keys;
/// - `valũe(ρ_a) = v` on the lookup-side value polynomial, committed advice of `2^{h_a}` entries
///   whose padding slots hold row 0's value, for a table with values only;
/// - `m̃(ρ_t) = −p_t` on the multiplicity polynomial, committed advice of `2^{h_t}` entries.
///
/// `q_a` is the lookup tree's leaf denominator claim, `p_t` the table tree's leaf numerator claim
/// and `v` the instance's entry of the values message; a verifier derives the same claims from
/// them.
#[derive(Clone, PartialEq, Eq)]
pub struct InstanceClaims<F> {
    /// The claim on the lookups' keys.
    pub key: Claim<F>,
    /// The claim on the lookups' values, `None` for a table without values.
    pub value: Option<Claim<F>>,
    /// The claim on the multiplicities.
    pub multiplicity: Claim<F>,
}

/// Shows the shapes, never a value (05 §3).
impl<F> fmt::Debug for InstanceClaims<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InstanceClaims")
            .field("key", &self.key)
            .field("value", &self.value)
            .field("multiplicity", &self.multiplicity)
            .finish()
    }
}

/// The LogUp part of a proven leaf stage: its challenges, its messages and the claims it leaves.
///
/// The messages, in transcript order: [`roots`](Self::roots) under `logup/roots`, each level's
/// [`proof`](ProvenTreeLevel::proof) under `tree/{ℓ}`, and [`values`](Self::values) under
/// `logup/values`. A verifier checks the roots, the levels and the leaf checks; what they cannot
/// check, the key, value and multiplicity claims of each instance and the forests' leaf claims,
/// later stages and openings owe, and a caller that drops them accepts what nothing checked.
#[must_use]
pub struct ProvenLogUp<F> {
    alpha: F,
    beta: F,
    roots: Vec<F>,
    levels: Vec<ProvenTreeLevel<F>>,
    values: Vec<F>,
    leaf_claims: Vec<LevelClaim<F>>,
    claims: Vec<InstanceClaims<F>>,
    forest_claims: Vec<Claim<F>>,
}

impl<F> ProvenLogUp<F> {
    /// `α`, squeezed under `logup/alpha`: the point of every leaf denominator `α − a`.
    pub fn alpha(&self) -> &F {
        &self.alpha
    }

    /// `β`, squeezed under `logup/beta`: the challenge that compresses a pair to
    /// `key + β·value`.
    pub fn beta(&self) -> &F {
        &self.beta
    }

    /// The `logup/roots` message: `n_a, d_a, n_t, d_t` for each instance in order, the roots of
    /// its lookup tree and of its table tree.
    pub fn roots(&self) -> &[F] {
        &self.roots
    }

    /// The tree levels, from batch level 0: the instances' trees beside the forests.
    pub fn levels(&self) -> &[ProvenTreeLevel<F>] {
        &self.levels
    }

    /// The `logup/values` message: `v = valũe(ρ_a)` for each instance whose table has values, in
    /// order.
    pub fn values(&self) -> &[F] {
        &self.values
    }

    /// Each tree's leaf claims, in schedule order: the next claims of the last level, or the root
    /// of a fraction tree of height 0.
    pub fn leaf_claims(&self) -> &[LevelClaim<F>] {
        &self.leaf_claims
    }

    /// Each instance's claims on its keys, values and multiplicities, in order.
    pub fn claims(&self) -> &[InstanceClaims<F>] {
        &self.claims
    }

    /// Each forest's leaf claim, in the order of the forests: on its stacked leaves, for the stage
    /// that consumes them.
    pub fn forest_claims(&self) -> &[Claim<F>] {
        &self.forest_claims
    }

    /// The instances' claims and the forests' leaf claims, by value.
    #[must_use]
    pub fn into_claims(self) -> (Vec<InstanceClaims<F>>, Vec<Claim<F>>) {
        (self.claims, self.forest_claims)
    }
}

/// Shows the shapes: the messages, challenges and claims are protocol-class data (05 §3).
impl<F> fmt::Debug for ProvenLogUp<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProvenLogUp")
            .field("roots", &self.roots.len())
            .field("levels", &self.levels)
            .field("values", &self.values.len())
            .field("claims", &self.claims)
            .field("forest_claims", &self.forest_claims)
            .finish_non_exhaustive()
    }
}

/// Each tree's position in a valid schedule.
struct Positions {
    lookup: Vec<usize>,
    table: Vec<usize>,
    forest: Vec<usize>,
}

/// The positions of the trees in `schedule`, or the first check it fails: one entry per tree of
/// `instances` instances and `forests` forests, each naming a tree no earlier entry names.
fn check_schedule(
    schedule: &[TreeId],
    instances: usize,
    forests: usize,
) -> Result<Positions, LogUpError> {
    let expected = 2 * instances + forests;
    if schedule.len() != expected {
        return Err(LogUpError::ScheduleLength {
            len: schedule.len(),
            expected,
        });
    }
    let mut positions = Positions {
        lookup: vec![usize::MAX; instances],
        table: vec![usize::MAX; instances],
        forest: vec![usize::MAX; forests],
    };
    for (position, &id) in schedule.iter().enumerate() {
        let slot = match id {
            TreeId::Lookup(k) => positions.lookup.get_mut(k),
            TreeId::Table(k) => positions.table.get_mut(k),
            TreeId::Forest(k) => positions.forest.get_mut(k),
        };
        match slot {
            Some(slot) if *slot == usize::MAX => *slot = position,
            _ => return Err(LogUpError::ScheduleTree { position }),
        }
    }
    Ok(positions)
}

/// `key + β·value`, the pair compressed, or the key alone without values.
#[inline(always)]
fn compress<F: PolyField>(beta: F, key: F::Base, value: Option<F::Base>) -> F {
    match value {
        Some(value) => beta.mul_by_base(value).add_base(key),
        None => F::from(key),
    }
}

/// The fraction tree over `count` leaves `(·, α − compress(pair x))` of `pairs` padded to
/// `2^height` leaves with `α − t_0`, numerators `numerators`.
fn tree<F: PolyField>(
    pairs: Tuples<'_, F::Base>,
    height: usize,
    numerators: LeafNumerators<'_, F::Base>,
    alpha: F,
    beta: F,
    padding: F,
) -> Result<FractionTree<F>, LogUpError> {
    let count = pairs.len();
    let keys = pairs.keys;
    // One closure per kind of pairs, so that no leaf asks whether the pairs have values.
    let tree = match pairs.values {
        Some(values) => FractionTree::new(height, numerators, |x| match x < count {
            true => alpha - compress(beta, keys[x], Some(values[x])),
            false => padding,
        }),
        None => FractionTree::new(height, numerators, |x| match x < count {
            true => alpha - compress(beta, keys[x], None),
            false => padding,
        }),
    };
    Ok(tree?)
}

/// An instance's lookup tree and table tree, both padded with row 0: the lookup tree's leaves
/// are `(1, α − a_i)`, then `(1, α − t_0)`; the table tree's `(−m_j, α − t_j)`, then
/// `(−m_j, α − t_0)`.
fn trees<F: PolyField>(
    instance: &LogUpInstance<'_, F::Base>,
    alpha: F,
    beta: F,
) -> Result<(FractionTree<F>, FractionTree<F>), LogUpError> {
    let table = instance.table;
    // The instance's checks give the table a row, and values to each row when it has them.
    let row_0 = table.keys.first().map_or(F::ZERO, |&key| {
        compress(beta, key, table.values.and_then(|v| v.first().copied()))
    });
    let padding = alpha - row_0;
    let lookup = tree(
        instance.lookups,
        instance.lookup_height(),
        LeafNumerators::Unit,
        alpha,
        beta,
        padding,
    )?;
    let negated: Vec<F::Base> = instance.multiplicities.iter().map(|&m| -m).collect();
    let rows = tree(
        table,
        instance.table_height(),
        LeafNumerators::Base(&negated),
        alpha,
        beta,
        padding,
    )?;
    Ok((lookup, rows))
}

/// `valũe(point)` of the lookup-side value polynomial: `values`, then row 0's value in the padding
/// slots, `2^{|point|}` entries.
fn value_at<F: PolyField>(values: &[F::Base], padding: F::Base, point: &[F]) -> F {
    let leaves = 1usize << point.len();
    let value = match values.len() == leaves {
        true => evaluate_base(values, point),
        false => {
            let mut padded = Vec::with_capacity(leaves);
            padded.extend_from_slice(values);
            padded.resize(leaves, padding);
            evaluate_base(&padded, point)
        }
    };
    // `2^{h_a}` entries at `h_a` coordinates: the evaluation cannot fail.
    value.unwrap_or(F::ZERO)
}

/// Proves the LogUp part of a leaf stage on `transcript`: the instances' lookup trees and table
/// trees, level by level beside the product forests `forests`, in the order `schedule` gives
/// (RFC-0008 'Fraction tree', 'Batching'; RFC-0010 'Tree levels'; RFC-0014 'Transcript
/// schedule', rows 4 and 5). It returns the challenges, the messages and the claims the leaf
/// checks leave.
///
/// On the transcript, in this order:
///
/// 1. `α` is squeezed under `logup/alpha`, then `β` under `logup/beta`. The values and the
///    multiplicities must already be committed: they are inputs, and nothing here depends on a
///    challenge but the trees.
/// 2. Each instance's two trees are built bottom-up, padded with row 0: the lookup tree's
///    `2^{h_a}` leaves are `(1, α − a_i)` for its lookups, then `(1, α − t_0)`; the table tree's
///    `2^{h_t}` leaves are `(−m_j, α − t_j)` for its rows, then `(−m_j, α − t_0)`, with
///    `a = key + β·value` and `t_j = key_j + β·value_j`, or the keys alone without values.
/// 3. The roots `n_a, d_a, n_t, d_t` of every instance, in order, are absorbed as one TCE
///    `Vec<F>` under `logup/roots`, even when empty.
/// 4. The trees join the levels of [`prove_trees`] where their heights match, each at `H − h`
///    with its root as its claims on no variable, in schedule order: `λ` is squeezed after `γ` at
///    each level where a fraction tree is present.
/// 5. `v = valũe(ρ_a)`, the lookup-side value polynomial at the lookup tree's leaf point, of each
///    instance whose table has values, is absorbed as one TCE `Vec<F>` under `logup/values`,
///    after the last level, or right after the roots when no tree has a level; even when empty.
/// 6. Each instance's claims follow from its trees' leaf claims, `p_a, q_a` at `ρ_a` and
///    `p_t, q_t` at `ρ_t` ([`InstanceClaims`]): `kẽy(ρ_a) = α − q_a − β·v`, `valũe(ρ_a) = v` and
///    `m̃(ρ_t) = −p_t`.
///
/// The prover never checks the identity: with lookups outside the table, wrong values or wrong
/// multiplicities, it builds honest trees over those leaves, and the verifier's root check
/// rejects them unless `α` is one of at most `N + T − 1` roots of their difference polynomial
/// (RFC-0008 'Relation').
///
/// # Algorithm
///
/// - **Leaves** are computed once, in order, into the trees' leaf levels: a lookup's denominator
///   costs one product of `β` by a base element; its numerator, 1, is never stored. The table
///   tree's numerators stay in the prime subfield.
/// - **Trees** are built by [`FractionTree::new`]: about 3 products per node, 1 on the lookup
///   tree's level above its leaves.
/// - **Levels** are [`prove_trees`]'s: each fraction tree's level is one sum-of-products
///   instance whose tables are each bound once per round, 5 products per pair and node, 3 on the
///   lookup tree's last level, whose numerators are 1.
/// - **Values** are one evaluation of the value column at `ρ_a`, about one product per lookup.
///
/// Control flow depends on shapes only, never on a lookup, a value or a multiplicity
/// (INV-ARCH-2). Every column and every tree node is a secret-class value (05 §3): no debug form
/// or error shows one, and the trees are freed level by level without being cleared, as the
/// rest of the prover does until #229 settles zeroization.
///
/// # Errors
///
/// A [`LogUpError`] for the first shape check that fails, in the order it documents: the
/// instances' columns, the forests' shapes, the schedule; before the transcript moves.
pub fn prove_logup<F, T>(
    transcript: &mut T,
    instances: &[LogUpInstance<'_, F::Base>],
    forests: &[Forest<'_, F>],
    schedule: &[TreeId],
) -> Result<ProvenLogUp<F>, LogUpError>
where
    F: PolyField,
    T: SumcheckTranscript<F> + ?Sized,
{
    for (k, instance) in instances.iter().enumerate() {
        instance.check(k)?;
    }
    check_forests(forests)?;
    let positions = check_schedule(schedule, instances.len(), forests.len())?;

    let alpha = transcript.challenge("logup/alpha");
    let beta = transcript.challenge("logup/beta");
    let mut lookup_trees = Vec::with_capacity(instances.len());
    let mut table_trees = Vec::with_capacity(instances.len());
    let mut roots = Vec::with_capacity(4 * instances.len());
    for instance in instances {
        let (lookup, table) = trees(instance, alpha, beta)?;
        let (n_a, d_a) = lookup.root();
        let (n_t, d_t) = table.root();
        roots.extend([n_a, d_a, n_t, d_t]);
        lookup_trees.push(Some(lookup));
        table_trees.push(Some(table));
    }
    transcript.absorb_message("logup/roots", &roots);

    let scheduled = schedule
        .iter()
        .enumerate()
        .map(|(position, &id)| {
            let tree = match id {
                TreeId::Lookup(k) => lookup_trees.get_mut(k).and_then(Option::take),
                TreeId::Table(k) => table_trees.get_mut(k).and_then(Option::take),
                TreeId::Forest(k) => {
                    return forests
                        .get(k)
                        .map(|&forest| Tree::Forest(forest))
                        .ok_or(LogUpError::ScheduleTree { position });
                }
            };
            tree.map(Tree::Fraction)
                .ok_or(LogUpError::ScheduleTree { position })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (levels, leaf_claims) = prove_trees(transcript, scheduled)?.into_levels();

    // A valid schedule names each tree once, so every position holds a claim of its tree's kind.
    let fraction = |position: usize| match leaf_claims.get(position) {
        Some(LevelClaim::Fraction(claim)) => Ok(claim.clone()),
        _ => Err(LogUpError::ScheduleTree { position }),
    };
    let mut values = Vec::new();
    let mut claims = Vec::with_capacity(instances.len());
    for (k, instance) in instances.iter().enumerate() {
        let lookup: FractionClaim<F> = fraction(positions.lookup[k])?;
        let table: FractionClaim<F> = fraction(positions.table[k])?;
        let value = instance.lookups.values.map(|column| {
            let padding = instance
                .table
                .values
                .and_then(|v| v.first().copied())
                .unwrap_or(F::Base::ZERO);
            value_at(column, padding, &lookup.point)
        });
        if let Some(v) = value {
            values.push(v);
        }
        let v = value.unwrap_or(F::ZERO);
        claims.push(InstanceClaims {
            key: Claim {
                point: lookup.point.clone(),
                value: alpha - lookup.denominator - beta * v,
            },
            value: value.map(|v| Claim {
                point: lookup.point.clone(),
                value: v,
            }),
            multiplicity: Claim {
                point: table.point,
                value: -table.numerator,
            },
        });
    }
    transcript.absorb_message("logup/values", &values);

    let forest_claims = positions
        .forest
        .iter()
        .map(|&position| match leaf_claims.get(position) {
            Some(LevelClaim::Product(claim)) => Ok(claim.clone()),
            _ => Err(LogUpError::ScheduleTree { position }),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ProvenLogUp {
        alpha,
        beta,
        roots,
        levels,
        values,
        leaf_claims,
        claims,
        forest_claims,
    })
}
