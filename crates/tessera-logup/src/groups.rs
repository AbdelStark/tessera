// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A leaf stage proven in token groups (RFC-0054 'The leaf stage with groups', 'The table
//! batch'; RFC-0014 'Transcript schedule', rows 4 and 5 with groups).
//!
//! On the transcript, in this order:
//!
//! 1. [`begin_groups`]: `α` under `logup/alpha`, `β` under `logup/beta`, then `logup/roots` and
//!    `logup/values` absorbed empty.
//! 2. [`prove_group`], for each group `h` in order: the roots `(n, d)` of its lookup trees under
//!    `group/{h}/logup/roots`, its levels beside its forests under `group/{h}/tree/{ℓ}/…`, and its
//!    lookup trees' value evaluations under `group/{h}/logup/values`. The group's split values
//!    and `γ_h` follow, from the caller.
//! 3. [`prove_table_batch`]: every instance's table tree, its roots `(n_t, d_t)` under
//!    `tables/logup/roots` and its levels under `tables/tree/{ℓ}/…`.
//!
//! An instance's lookups, `a = key + β·value`, are spread over the groups: each group proves a
//! lookup tree over its own share, padded with row 0, and the table tree's `m_0` counts every
//! group's padding. The verifier sums the groups' fractions and the table root of each instance
//! to RFC-0008's identity (RFC-0054 'The table batch', 'The identity').

use core::fmt;

use tessera_field::Field;
use tessera_poly::PolyField;
use tessera_sumcheck::{
    Claim, Forest, FractionClaim, FractionTree, LeafNumerators, LevelClaim, ProvenTreeLevel,
    SumcheckTranscript, Tree, TreeScope, check_forests, prove_trees_in,
};

use crate::prover::{compress, lookup_tree, table_tree, tree, value_at};
use crate::{Column, LogUpError, LogUpInstance, TreeId, Tuples};

/// The grouped stage's challenges `α` and `β`, under `logup/alpha` and `logup/beta`, with
/// `logup/roots` and `logup/values` then absorbed empty (RFC-0014 'Transcript schedule', row 4
/// with groups). The values and the multiplicities must already be committed.
pub fn begin_groups<F, T>(transcript: &mut T) -> (F, F)
where
    F: PolyField,
    T: SumcheckTranscript<F> + ?Sized,
{
    let alpha = transcript.challenge("logup/alpha");
    let beta = transcript.challenge("logup/beta");
    transcript.absorb_message("logup/roots", &[]);
    transcript.absorb_message("logup/values", &[]);
    (alpha, beta)
}

/// A lookup tree's claims at its leaf point: on the group's keys and, for a table with values, on
/// its values.
#[derive(Clone, PartialEq, Eq)]
pub struct LookupClaims<F> {
    /// The claim on the lookups' keys, `kẽy(ρ_a) = α − q_a − β·v`.
    pub key: Claim<F>,
    /// The claim on the lookups' values, `valũe(ρ_a) = v`; `None` for a table without values.
    pub value: Option<Claim<F>>,
}

/// The claims, whose points and values are protocol-class, as the claim's own form shows them.
impl<F> fmt::Debug for LookupClaims<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LookupClaims")
            .field("key", &self.key)
            .field("value", &self.value)
            .finish()
    }
}

/// One group's proven trees: its messages and the claims its leaves leave.
#[must_use]
pub struct ProvenGroup<F> {
    roots: Vec<F>,
    levels: Vec<ProvenTreeLevel<F>>,
    values: Vec<F>,
    claims: Vec<LookupClaims<F>>,
    forest_claims: Vec<Claim<F>>,
}

impl<F> ProvenGroup<F> {
    /// The `group/{h}/logup/roots` message: `n_a, d_a` of each lookup tree, in schedule order.
    pub fn roots(&self) -> &[F] {
        &self.roots
    }

    /// The group's levels, from its level 0.
    pub fn levels(&self) -> &[ProvenTreeLevel<F>] {
        &self.levels
    }

    /// The `group/{h}/logup/values` message: `valũe(ρ_a)` of each lookup tree whose table has
    /// values, in schedule order.
    pub fn values(&self) -> &[F] {
        &self.values
    }

    /// Each instance's lookup claims, in the order of the instances.
    pub fn claims(&self) -> &[LookupClaims<F>] {
        &self.claims
    }

    /// Each forest's leaf claim, in the order of the forests.
    pub fn forest_claims(&self) -> &[Claim<F>] {
        &self.forest_claims
    }
}

/// Counts and the levels' own forms: the messages are protocol-class.
impl<F> fmt::Debug for ProvenGroup<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProvenGroup")
            .field("roots", &self.roots.len())
            .field("levels", &self.levels)
            .field("values", &self.values.len())
            .field("claims", &self.claims)
            .field("forest_claims", &self.forest_claims)
            .finish()
    }
}

/// The positions of a group's lookup trees and forests in `schedule`, or the first check it
/// fails: one entry per lookup tree of `instances` instances and per forest of `forests`, no
/// table tree, each naming a tree no earlier entry names.
fn group_positions(
    schedule: &[TreeId],
    instances: usize,
    forests: usize,
) -> Result<(Vec<usize>, Vec<usize>), LogUpError> {
    let expected = instances + forests;
    if schedule.len() != expected {
        return Err(LogUpError::ScheduleLength {
            len: schedule.len(),
            expected,
        });
    }
    let mut lookup = vec![usize::MAX; instances];
    let mut forest = vec![usize::MAX; forests];
    for (position, &id) in schedule.iter().enumerate() {
        let slot = match id {
            TreeId::Lookup(k) => lookup.get_mut(k),
            TreeId::Forest(k) => forest.get_mut(k),
            TreeId::Table(_) => None,
        };
        match slot {
            Some(slot) if *slot == usize::MAX => *slot = position,
            _ => return Err(LogUpError::ScheduleTree { position }),
        }
    }
    Ok((lookup, forest))
}

/// Group `group`'s part of a grouped leaf stage (RFC-0054 'The leaf stage with groups', step 2):
/// the lookup trees of `instances`, each holding the group's lookups into its table, beside the
/// product forests `forests`, in the order `schedule` gives.
///
/// On the transcript, in this order:
/// 1. The roots `n_a, d_a` of each lookup tree, in schedule order, as one TCE `Vec<F>` under
///    `group/{h}/logup/roots`.
/// 2. The levels of [`prove_trees_in`] under `group/{h}/tree/{ℓ}/…`.
/// 3. `valũe(ρ_a)` of each lookup tree whose table has values, in schedule order, under
///    `group/{h}/logup/values`.
///
/// An instance's multiplicities enter no group: they are checked here as the ungrouped stage
/// checks them, and proven in [`prove_table_batch`].
///
/// # Errors
///
/// A [`LogUpError`] for the first shape check that fails: the instances' columns, the forests'
/// shapes, then the schedule, which must name each lookup tree and each forest once and no table
/// tree; before the transcript moves.
pub fn prove_group<F, T>(
    transcript: &mut T,
    group: u32,
    alpha: F,
    beta: F,
    instances: &[LogUpInstance<'_, F::Base>],
    forests: &[Forest<'_, F>],
    schedule: &[TreeId],
) -> Result<ProvenGroup<F>, LogUpError>
where
    F: PolyField,
    T: SumcheckTranscript<F> + ?Sized,
{
    for (k, instance) in instances.iter().enumerate() {
        instance.check(k)?;
    }
    check_forests(forests)?;
    let (lookup_positions, forest_positions) =
        group_positions(schedule, instances.len(), forests.len())?;
    let lookup_instances: Vec<_> = instances
        .iter()
        .map(|instance| LookupInstance {
            lookups: instance.lookups,
            padding_key: instance.table.keys[0],
            padding_value: instance.table.values.map(|v| v[0]),
        })
        .collect();
    let built = instances
        .iter()
        .map(|instance| lookup_tree(instance, alpha, beta).map(Some))
        .collect::<Result<Vec<_>, _>>()?;
    prove_group_with_trees(
        transcript,
        group,
        alpha,
        beta,
        &lookup_instances,
        forests,
        schedule,
        built,
        lookup_positions,
        forest_positions,
    )
}

/// One group's lookup-only input. The public padding tuple is table row 0; there is
/// no dummy table or multiplicity column. The actual table/count batch is proven later.
pub struct LookupInstance<'a, B> {
    /// Lookup tuples in canonical leaf order.
    pub lookups: Tuples<'a, B>,
    /// The original table's public row-0 key.
    pub padding_key: B,
    /// The original table's public row-0 value; absent exactly for a table without values.
    pub padding_value: Option<B>,
}

/// Prove a group's resident lookup trees/forests without materializing public table
/// columns or any multiplicity conversion (RFC-0054). The caller still owes the full
/// generated table/count batch, grouped identity, leaf merge and commitment openings.
///
/// # Errors
/// Invalid lookup columns, forests or schedule before transcript movement; tree errors
/// preserve the resident engine's error type.
pub fn prove_lookup_group<F, T>(
    transcript: &mut T,
    group: u32,
    alpha: F,
    beta: F,
    instances: &[LookupInstance<'_, F::Base>],
    forests: &[Forest<'_, F>],
    schedule: &[TreeId],
) -> Result<ProvenGroup<F>, LogUpError>
where
    F: PolyField,
    T: SumcheckTranscript<F> + ?Sized,
{
    for (k, instance) in instances.iter().enumerate() {
        if instance.lookups.is_empty() {
            return Err(LogUpError::NoLookups { instance: k });
        }
        if instance.lookups.has_values() != instance.padding_value.is_some() {
            return Err(LogUpError::ValuesMismatch { instance: k });
        }
        if let Some(values) = instance.lookups.values
            && values.len() != instance.lookups.len()
        {
            return Err(LogUpError::ColumnLength {
                instance: k,
                column: Column::LookupValues,
                len: values.len(),
                expected: instance.lookups.len(),
            });
        }
    }
    check_forests(forests)?;
    let (lookup_positions, forest_positions) =
        group_positions(schedule, instances.len(), forests.len())?;
    let built = instances
        .iter()
        .map(|instance| {
            let padding = alpha - compress(beta, instance.padding_key, instance.padding_value);
            tree(
                instance.lookups,
                crate::instance::ceil_log2(instance.lookups.len()),
                LeafNumerators::Unit,
                alpha,
                beta,
                padding,
            )
            .map(Some)
        })
        .collect::<Result<Vec<_>, _>>()?;
    prove_group_with_trees(
        transcript,
        group,
        alpha,
        beta,
        instances,
        forests,
        schedule,
        built,
        lookup_positions,
        forest_positions,
    )
}

#[allow(clippy::too_many_arguments)]
fn prove_group_with_trees<F, T>(
    transcript: &mut T,
    group: u32,
    alpha: F,
    beta: F,
    instances: &[LookupInstance<'_, F::Base>],
    forests: &[Forest<'_, F>],
    schedule: &[TreeId],
    mut built: Vec<Option<FractionTree<F>>>,
    lookup_positions: Vec<usize>,
    forest_positions: Vec<usize>,
) -> Result<ProvenGroup<F>, LogUpError>
where
    F: PolyField,
    T: SumcheckTranscript<F> + ?Sized,
{
    let mut roots = Vec::with_capacity(2 * instances.len());
    let mut scheduled = Vec::with_capacity(schedule.len());
    for (position, &id) in schedule.iter().enumerate() {
        scheduled.push(match id {
            TreeId::Lookup(k) => {
                let tree = built[k]
                    .take()
                    .ok_or(LogUpError::ScheduleTree { position })?;
                let (n, d) = tree.root();
                roots.extend([n, d]);
                Tree::Fraction(tree)
            }
            TreeId::Forest(k) => Tree::Forest(forests[k]),
            TreeId::Table(_) => return Err(LogUpError::ScheduleTree { position }),
        });
    }
    transcript.absorb_message(&format!("group/{group}/logup/roots"), &roots);
    let (levels, leaf_claims) =
        prove_trees_in(transcript, TreeScope::Group(group), scheduled)?.into_levels();

    let fraction = |position: usize| match leaf_claims.get(position) {
        Some(LevelClaim::Fraction(claim)) => Ok(claim.clone()),
        _ => Err(LogUpError::ScheduleTree { position }),
    };
    // Values in schedule order; claims in the order of the instances.
    let mut by_position: Vec<(usize, usize)> = lookup_positions
        .iter()
        .enumerate()
        .map(|(k, &position)| (position, k))
        .collect();
    by_position.sort_unstable();
    let mut values = Vec::new();
    let mut found: Vec<Option<F>> = vec![None; instances.len()];
    for &(position, k) in &by_position {
        let instance = &instances[k];
        let lookup: FractionClaim<F> = fraction(position)?;
        let value = instance.lookups.values.map(|column| {
            let padding = instance.padding_value.unwrap_or(F::Base::ZERO);
            value_at(column, padding, &lookup.point)
        });
        if let Some(v) = value {
            values.push(v);
        }
        found[k] = value;
    }
    let mut claims = Vec::with_capacity(instances.len());
    for (k, &position) in lookup_positions.iter().enumerate() {
        let lookup: FractionClaim<F> = fraction(position)?;
        let v = found[k];
        claims.push(LookupClaims {
            key: Claim {
                point: lookup.point.clone(),
                value: alpha - lookup.denominator - beta * v.unwrap_or(F::ZERO),
            },
            value: v.map(|v| Claim {
                point: lookup.point.clone(),
                value: v,
            }),
        });
    }
    transcript.absorb_message(&format!("group/{group}/logup/values"), &values);
    let forest_claims = forest_positions
        .iter()
        .map(|&position| match leaf_claims.get(position) {
            Some(LevelClaim::Product(claim)) => Ok(claim.clone()),
            _ => Err(LogUpError::ScheduleTree { position }),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ProvenGroup {
        roots,
        levels,
        values,
        claims,
        forest_claims,
    })
}

/// The table batch's proven trees: its roots, its levels and each instance's multiplicity claim.
#[must_use]
pub struct ProvenTables<F> {
    pub(crate) roots: Vec<F>,
    pub(crate) levels: Vec<ProvenTreeLevel<F>>,
    pub(crate) multiplicities: Vec<Claim<F>>,
}

impl<F> ProvenTables<F> {
    /// The `tables/logup/roots` message: `n_t, d_t` of each instance's table tree, in order.
    pub fn roots(&self) -> &[F] {
        &self.roots
    }

    /// The table batch's levels, from its level 0.
    pub fn levels(&self) -> &[ProvenTreeLevel<F>] {
        &self.levels
    }

    /// Each instance's claim on its multiplicities, `m̃(ρ_t) = −p_t`, in order.
    pub fn multiplicities(&self) -> &[Claim<F>] {
        &self.multiplicities
    }
}

/// Counts and the levels' own forms: the messages are protocol-class.
impl<F> fmt::Debug for ProvenTables<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProvenTables")
            .field("roots", &self.roots.len())
            .field("levels", &self.levels)
            .field("multiplicities", &self.multiplicities)
            .finish()
    }
}

/// The table batch of a grouped leaf stage (RFC-0054 'The table batch'): every instance's table
/// tree, its multiplicities counting every group's lookups and padding.
///
/// On the transcript, in this order: the roots `n_t, d_t` of each table tree, in the order of
/// the instances, under `tables/logup/roots`; then the levels of [`prove_trees_in`] under
/// `tables/tree/{ℓ}/…`. The lookups of `instances` enter no tree here.
///
/// # Errors
///
/// A [`LogUpError`] for the first instance whose columns fail their checks, before the
/// transcript moves.
pub fn prove_table_batch<F, T>(
    transcript: &mut T,
    alpha: F,
    beta: F,
    instances: &[LogUpInstance<'_, F::Base>],
) -> Result<ProvenTables<F>, LogUpError>
where
    F: PolyField,
    T: SumcheckTranscript<F> + ?Sized,
{
    for (k, instance) in instances.iter().enumerate() {
        instance.check(k)?;
    }
    let mut roots = Vec::with_capacity(2 * instances.len());
    let mut trees = Vec::with_capacity(instances.len());
    for instance in instances {
        let tree = table_tree(instance, alpha, beta)?;
        let (n, d) = tree.root();
        roots.extend([n, d]);
        trees.push(Tree::Fraction(tree));
    }
    transcript.absorb_message("tables/logup/roots", &roots);
    let (levels, leaf_claims) = prove_trees_in(transcript, TreeScope::Tables, trees)?.into_levels();
    let multiplicities = leaf_claims
        .into_iter()
        .enumerate()
        .map(|(position, claim)| match claim {
            LevelClaim::Fraction(table) => Ok(Claim {
                point: table.point,
                value: -table.numerator,
            }),
            LevelClaim::Product(_) => Err(LogUpError::ScheduleTree { position }),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ProvenTables {
        roots,
        levels,
        multiplicities,
    })
}
