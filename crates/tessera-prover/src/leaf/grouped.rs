// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A grouped leaf stage's proof up to its atom splits (RFC-0054 'The leaf stage with groups',
//! steps 1 to 4), over the layouts of [`GroupedLayout`].
//!
//! 1. **Challenges.** `logup/alpha` and `logup/beta`, then `logup/roots` and `logup/values`
//!    absorbed empty ([`begin_groups`]).
//! 2. **Each group `h`, from 0.** Its lookup columns, each block's share read through
//!    [`GroupedLayout::source`]; its share of each product forest; its trees under
//!    `group/{h}/…`, its lookup trees in table order and then its forests ([`prove_group`]); its
//!    split values under `group/{h}/split`, each a tensor's extension at the group's leaf point
//!    `P_h` and at the group's bits on `S`; then `γ_h` under `group/{h}/merge/gamma`.
//! 3. **The table batch.** Every instance's table tree, its multiplicities counting every group's
//!    lookups and padding ([`prove_table_batch`]).
//! 4. **The merge.** `β` under `merge/beta`, then the merge of every group's claims: the split
//!    values on the tensors, in the ungrouped split's order, and the forests' leaf claims on their
//!    stacked leaf tables, in forest order. Group `h`'s claim `k` counts its split values first,
//!    then its forests' leaf claims. [`MergeMode`] chooses its prover; both send the same
//!    messages:
//!    - **in memory** ([`prove_merge`]), every polynomial held at once;
//!    - **in two passes** (RFC-0054 'The prover in two passes', [`prove_merge_two_pass`]): each
//!      group's pass keeps its merge tables while its blocks are resident, the first `j*` rounds
//!      read them, and one pass, P2b, regenerates every polynomial for the rest.
//!
//! What the merge leaves, one claim per polynomial at its own point, RFC-0037's atom splits and
//! the later stages owe; the caller absorbs the atom splits under `stage/0/split` next.
//!
//! The tensors, the columns, the forests' leaves and the multiplicities are secret-class (05 §3):
//! the caller lends them, and what is computed from them lives in `SecretBuf`s and `Secret`s.

use core::fmt;

use tessera_codec::{Secret, SecretBuf};
use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField};
use tessera_gadgets::TableColumns;
use tessera_logup::{
    LogUpInstance, ProvenGroup, ProvenTables, TreeId, Tuples, begin_groups, prove_group,
    prove_table_batch,
};
use tessera_poly::evaluate_base;
use tessera_sumcheck::{
    Claim, Forest, Merge, MergeClaim, MergePolynomial, MergeTables, ProvenBatch, ProvenTreeLevel,
    SumcheckTranscript, largest_table_rounds, prove_merge, prove_merge_two_pass,
};

use super::layout::LeafLayout;
use super::prove::{SiteTensors, check_witness, columns, count, lookup_columns};
use super::{Block, GroupLayout, GroupedLayout, LeafError, internal};

/// A product forest of the leaf stage, stacked over the layers, as the groups split it (RFC-0054
/// 'The forest', 'Group variables'): its leaf table, its trees' height, and the lowest variables
/// of the leaf table that the query index and the layer index take.
///
/// Group `h` holds the trees of its layer subset's layers and its token subset's queries: the
/// table with the query index's low `g_T` bits set to its token bits and the layer index's low
/// `g_L` bits to its layer bits, in the table's order. Those bits are the forest's group
/// variables.
///
/// The leaves are secret-class (05 §3), lent for the call. The debug form shows the shape.
pub struct StageForest<'a> {
    leaves: &'a [Goldilocks],
    height: u32,
    queries: u32,
    layers: u32,
}

impl<'a> StageForest<'a> {
    /// The forest of `leaves`, trees of `height` levels, the query index from variable `queries`
    /// and the layer index from variable `layers`.
    pub fn new(leaves: &'a [Goldilocks], height: u32, queries: u32, layers: u32) -> Self {
        Self {
            leaves,
            height,
            queries,
            layers,
        }
    }

    /// The same forest's shape, its trees' height and its query and layer indices, over other
    /// `leaves`: a test plants a cheating prover's leaves with it (RFC-0054 'Testing Strategy').
    pub fn with_leaves<'b>(&self, leaves: &'b [Goldilocks]) -> StageForest<'b> {
        StageForest::new(leaves, self.height, self.queries, self.layers)
    }

    /// The leaf table's variables.
    fn vars(&self) -> Option<u32> {
        self.leaves
            .len()
            .is_power_of_two()
            .then(|| self.leaves.len().trailing_zeros())
    }
}

/// Shows the shape, never a leaf (05 §3).
impl fmt::Debug for StageForest<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StageForest")
            .field("leaves", &self.leaves.len())
            .field("height", &self.height)
            .field("queries", &self.queries)
            .field("layers", &self.layers)
            .finish()
    }
}

/// A grouped leaf stage, proven up to its atom splits (RFC-0054 steps 1 to 4).
///
/// The messages, in transcript order: each group's roots, levels, values and split values, the
/// table batch's roots and levels, and the merge's rounds and evaluations. What the merge leaves,
/// [`merged`](Self::merged), later stages owe; a caller that drops it accepts what nothing
/// checked.
#[must_use]
pub struct ProvenGroupedLeaf {
    groups: Vec<ProvenGroup<GoldilocksExt2>>,
    splits: Vec<Vec<GoldilocksExt2>>,
    points: Vec<Vec<GoldilocksExt2>>,
    tables: ProvenTables<GoldilocksExt2>,
    merge: ProvenBatch<GoldilocksExt2>,
    merged: Vec<Claim<GoldilocksExt2>>,
}

impl ProvenGroupedLeaf {
    /// Each group's trees, in group order.
    pub fn groups(&self) -> &[ProvenGroup<GoldilocksExt2>] {
        &self.groups
    }

    /// Each group's `group/{h}/split` message, in group order.
    pub fn splits(&self) -> &[Vec<GoldilocksExt2>] {
        &self.splits
    }

    /// Each group's leaf point `P_h`, empty for a group with no tree.
    pub fn points(&self) -> &[Vec<GoldilocksExt2>] {
        &self.points
    }

    /// The table batch.
    pub fn tables(&self) -> &ProvenTables<GoldilocksExt2> {
        &self.tables
    }

    /// The merge.
    pub fn merge(&self) -> &ProvenBatch<GoldilocksExt2> {
        &self.merge
    }

    /// One claim per merged polynomial, in the merge's order, at its own point with its lowest
    /// variable first: the tensors in the ungrouped split's order, then the forests' leaf tables.
    pub fn merged(&self) -> &[Claim<GoldilocksExt2>] {
        &self.merged
    }

    /// The merged point `r(n, S)` of `vars` variables with group variables `group_vars`
    /// (RFC-0054 'The stage's outputs'), where the stage leaves every claim on a tensor of that
    /// shape. `None` past the merge's rounds, or for group variables that are not ascending
    /// positions below `vars`.
    pub fn merged_point(&self, vars: usize, group_vars: &[usize]) -> Option<Vec<GoldilocksExt2>> {
        merged_point(self.merge.point(), vars, group_vars)
    }
}

/// `r(n, S)` from the merge's challenges in round order: the last `vars` of them, assigned to
/// the variables outside `group_vars` lowest first, then to those in it, front-padding giving a
/// polynomial of `vars` variables the last rounds (RFC-0054 'Rounds').
fn merged_point(
    challenges: &[GoldilocksExt2],
    vars: usize,
    group_vars: &[usize],
) -> Option<Vec<GoldilocksExt2>> {
    let ascending = group_vars.windows(2).all(|w| w[0] < w[1]);
    if !ascending || group_vars.last().is_some_and(|&v| v >= vars) {
        return None;
    }
    let bound = challenges.get(challenges.len().checked_sub(vars)?..)?;
    let order = (0..vars)
        .filter(|v| group_vars.binary_search(v).is_err())
        .chain(group_vars.iter().copied());
    let mut point = vec![GoldilocksExt2::ZERO; vars];
    for (var, &r) in order.zip(bound) {
        point[var] = r;
    }
    Some(point)
}

/// Shows the shapes: the messages and challenges are protocol-class data (05 §3).
impl fmt::Debug for ProvenGroupedLeaf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProvenGroupedLeaf")
            .field("groups", &self.groups.len())
            .field("tables", &self.tables)
            .field("merged", &self.merged.len())
            .finish_non_exhaustive()
    }
}

/// The leaf point of a group's levels, `(ρ_{H−1}, r_{n−1}, …, r_0)` as RFC-0045 'The leaf point'
/// states it for the stage; empty for a group with no level.
pub(super) fn level_point<F: Copy>(levels: &[ProvenTreeLevel<F>]) -> Vec<F> {
    match levels.last() {
        Some(level) => core::iter::once(*level.rho())
            .chain(level.point().iter().rev().copied())
            .collect(),
        None => Vec::new(),
    }
}

/// The multiplicities of a grouped stage, one column per instance of `stage` in table order
/// (RFC-0054 'When groups apply'): each instance's lookups in every group, and every group's
/// padding at row 0. They are committed before `logup/alpha`.
///
/// # Errors
///
/// As [`LeafColumns::new`](super::LeafColumns::new), group by group.
pub fn grouped_multiplicities<S: Copy + Eq>(
    stage: &LeafLayout<S>,
    grouped: &GroupedLayout,
    tables: &[TableColumns<Goldilocks>],
    witness: &[Vec<SiteTensors<'_, S>>],
) -> Result<Vec<SecretBuf<Goldilocks>>, LeafError> {
    check_witness(stage, witness)?;
    let mut totals: Vec<Option<SecretBuf<Goldilocks>>> =
        (0..stage.instances().len()).map(|_| None).collect();
    for h in 0..grouped.count() {
        let layout = grouped.group(h)?;
        for (instance, keys, _) in group_columns(stage, grouped, h, &layout, tables, witness)? {
            let rows = columns(tables, instance)?.tuples();
            let counted = count(rows, &keys)?;
            let k = stage
                .instances()
                .iter()
                .position(|i| i.table == instance)
                .ok_or(LeafError::Internal {
                    rule: "a group's instances are the stage's",
                })?;
            match &mut totals[k] {
                Some(total) => {
                    for (sum, &m) in total.iter_mut().zip(counted.iter()) {
                        *sum += m;
                    }
                }
                slot => *slot = Some(counted),
            }
        }
    }
    totals
        .into_iter()
        .map(|total| {
            total.ok_or(LeafError::Internal {
                rule: "every instance of the stage is in a group",
            })
        })
        .collect()
}

/// Group `h`'s lookup columns, instance by instance in table order: each table with its keys and
/// values in leaf order, row 0's tuple in the padding.
fn group_columns<S: Copy + Eq>(
    stage: &LeafLayout<S>,
    grouped: &GroupedLayout,
    h: usize,
    layout: &GroupLayout,
    tables: &[TableColumns<Goldilocks>],
    witness: &[Vec<SiteTensors<'_, S>>],
) -> Result<GroupColumns, LeafError> {
    let mut out = Vec::with_capacity(layout.instances().len());
    for instance in layout.instances() {
        let rows = columns(tables, instance.table)?.tuples();
        let source = |block: &Block, x: u64| grouped.source(h, block, x);
        let (keys, values) = lookup_columns(stage, instance, rows, witness, source)?;
        out.push((instance.table, keys, values));
    }
    Ok(out)
}

/// A group's columns: each instance's table, keys and values.
type GroupColumns = Vec<(
    tessera_tables::TableId,
    SecretBuf<Goldilocks>,
    Option<SecretBuf<Goldilocks>>,
)>;

/// Group `h`'s share of `forest`: its leaf table with the query index's low `g_T` bits set to
/// `tokens` and the layer index's low `g_L` bits to `layers`, in the table's order.
fn forest_share(
    forest: &StageForest<'_>,
    g_t: u32,
    g_l: u32,
    tokens: u64,
    layers: u64,
) -> Result<SecretBuf<Goldilocks>, LeafError> {
    let vars = forest.vars().ok_or(LeafError::Internal {
        rule: "a forest's leaf table holds 2^n leaves",
    })?;
    let len = 1usize
        .checked_shl(vars - g_t - g_l)
        .ok_or(LeafError::Memory)?;
    let mut share = SecretBuf::try_with_capacity(len).map_err(|_| LeafError::Memory)?;
    let (q, l) = (forest.queries, forest.layers);
    for j in 0..len as u64 {
        // The variables below the query index, between it and the layer index, and above.
        let low = j & ((1u64 << q) - 1);
        let middle = (j >> q) & ((1u64 << (l - q - g_t)) - 1);
        let high = j >> (l - g_t);
        let x = low | (tokens << q) | (middle << (q + g_t)) | (layers << l) | (high << (l + g_l));
        share.push(forest.leaves[x as usize]);
    }
    Ok(share)
}

/// How [`prove_grouped_leaf`] proves the merge. Both modes send the same messages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MergeMode {
    /// Every merged polynomial in memory at once ([`prove_merge`]).
    InMemory,
    /// RFC-0054 'The prover in two passes': the first `j_star` rounds from tables each group's
    /// pass keeps, then one pass, P2b, that regenerates every polynomial and folds the variables
    /// those rounds bound ([`prove_merge_two_pass`]).
    TwoPass {
        /// `j*`, the rounds the tables answer: at most [`largest_table_rounds`] of the merge's
        /// polynomials, or `None` for [`balanced_j_star`]'s.
        j_star: Option<usize>,
    },
}

/// What the witness lends for a merged polynomial (05 §3: secret-class, read where it is lent).
#[derive(Clone, Copy)]
enum Lent<'a> {
    /// A stage tensor's entries, true integers.
    Tensor(&'a [i64]),
    /// A forest's stacked leaves.
    Forest(&'a [Goldilocks]),
}

/// The `j*` that balances the merge's two memories (RFC-0054 'The prover in two passes',
/// 'Memory'): the tables take `2^{j*+1}` entries per class and group, and the folded state
/// `N_m/2^{j*}` for the `N_m` merged entries. With at most `G·P` classes, `P` the polynomials,
/// the balance is `2^{2j*} = N_m/(G·P)`; `j*` rounds it up and stays within
/// [`largest_table_rounds`].
pub fn balanced_j_star(polynomials: &[MergePolynomial], groups: usize) -> usize {
    let entries: u128 = polynomials.iter().map(|p| 1u128 << p.vars).sum();
    let classes = (groups as u128)
        .saturating_mul(polynomials.len() as u128)
        .max(1);
    let ratio = entries.div_ceil(classes);
    // ⌈log2 ratio⌉, then halved, rounding up.
    let bits = (u128::BITS - ratio.saturating_sub(1).leading_zeros()) as usize;
    bits.div_ceil(2).min(largest_table_rounds(polynomials))
}

/// Polynomial `shape`'s block at group bits `bits`: its `2^{m}` entries over its free variables
/// in increasing position, `entry(x)` giving its entry `x` in natural order.
fn block(
    entry: impl Fn(usize) -> Option<Goldilocks>,
    shape: &MergePolynomial,
    bits: u64,
) -> Option<Vec<Goldilocks>> {
    let free: Vec<usize> = (0..shape.vars)
        .filter(|v| shape.group_vars.binary_search(v).is_err())
        .collect();
    let fixed = shape
        .group_vars
        .iter()
        .enumerate()
        .fold(0usize, |x, (u, &v)| x | ((((bits >> u) & 1) as usize) << v));
    (0..1usize << free.len())
        .map(|y| {
            let x = free
                .iter()
                .enumerate()
                .fold(fixed, |x, (t, &v)| x | (((y >> t) & 1) << v));
            entry(x)
        })
        .collect()
}

/// Proves the grouped leaf stage of `stage` under `grouped` on `transcript`, up to its atom
/// splits (RFC-0054 'The leaf stage with groups', steps 1 to 4; the module documentation lists
/// the messages).
///
/// - `witness[s][l]` lends site `s`'s tensors at layer `l`, as for the ungrouped stage.
/// - `multiplicities` holds each instance's committed column, in table order:
///   [`grouped_multiplicities`]' for an honest prover.
/// - `forests` are the stage's product forests, in schedule order, each with its public root
///   values the verifier derives.
/// - `mode` chooses the merge's prover ([`MergeMode`]).
///
/// # Errors
///
/// [`LeafError::Internal`] (`ERR-INTERNAL`), before the transcript moves, for a witness that does
/// not fit the stage, an ungrouped stage (`g = 0`, which keeps RFC-0010's messages:
/// [`prepare_leaf`](super::prepare_leaf)), a multiplicity count other than the instances', or a forest whose leaf
/// table is not `2^n` leaves or whose query and layer indices do not hold the group's bits
/// beneath `n`, or a `j*` past [`largest_table_rounds`]. Then, group by group, the errors of the
/// columns ([`LeafError::Internal`], [`LeafError::Memory`]) and of the trees
/// ([`LeafError::LogUp`]); the table batch's; and the merge's ([`LeafError::Internal`]).
#[allow(clippy::too_many_arguments)]
pub fn prove_grouped_leaf<S, T>(
    transcript: &mut T,
    stage: &LeafLayout<S>,
    grouped: &GroupedLayout,
    tables: &[TableColumns<Goldilocks>],
    witness: &[Vec<SiteTensors<'_, S>>],
    multiplicities: &[&[Goldilocks]],
    forests: &[StageForest<'_>],
    mode: MergeMode,
) -> Result<ProvenGroupedLeaf, LeafError>
where
    S: Copy + Eq,
    T: SumcheckTranscript<GoldilocksExt2> + ?Sized,
{
    check_witness(stage, witness)?;
    if grouped.groups().bits() == 0 {
        return internal("an ungrouped stage keeps RFC-0010's messages (prepare_leaf)");
    }
    if multiplicities.len() != stage.instances().len() {
        return internal("one multiplicity column per instance of the stage");
    }
    let g_t = u32::from(grouped.groups().token_bits);
    let g_l = u32::from(grouped.groups().layer_bits);
    for forest in forests {
        let vars = forest.vars().ok_or(LeafError::Internal {
            rule: "a forest's leaf table holds 2^n leaves",
        })?;
        let fits = forest.height >= 1
            && forest.height <= forest.queries
            && forest.queries + g_t <= forest.layers
            && forest.layers + g_l <= vars;
        if !fits {
            return internal(
                "a forest's query and layer indices hold the groups' bits above its trees' leaves",
            );
        }
    }
    // The merge's polynomials: every stage tensor, then every forest's leaves. `table(i)` reads
    // polynomial `i` from what the witness lends, in natural order.
    let mut polynomials: Vec<MergePolynomial> = grouped.polynomials().to_vec();
    let mut tensor_reads: Vec<(usize, usize, usize)> = Vec::with_capacity(polynomials.len());
    for (s, layers) in witness.iter().enumerate() {
        for l in 0..layers.len() {
            let reads = stage.reads(s, l).ok_or(LeafError::Internal {
                rule: "the layout reads every site's tensors",
            })?;
            tensor_reads.extend((0..reads.len()).map(|r| (s, l, r)));
        }
    }
    if tensor_reads.len() != polynomials.len() {
        return internal("one merged polynomial per value of the stage's split");
    }
    for forest in forests {
        let vars = forest.vars().unwrap_or(0) as usize;
        let (q, l) = (forest.queries as usize, forest.layers as usize);
        polynomials.push(MergePolynomial {
            vars,
            group_vars: (q..q + g_t as usize).chain(l..l + g_l as usize).collect(),
        });
    }
    // What the witness lends for polynomial `i`: a tensor's entries or a forest's leaves.
    let lent = |i: usize| -> Result<Lent<'_>, LeafError> {
        match tensor_reads.get(i) {
            Some(&(s, l, r)) => {
                let read = stage.reads(s, l).and_then(|reads| reads.get(r)).ok_or(
                    LeafError::Internal {
                        rule: "the layout reads every site's tensors",
                    },
                )?;
                Ok(Lent::Tensor(witness[s][l].of(read)?))
            }
            None => forests
                .get(i - tensor_reads.len())
                .map(|forest| Lent::Forest(forest.leaves))
                .ok_or(LeafError::Internal {
                    rule: "a merged polynomial is a tensor or a forest",
                }),
        }
    };
    let table = |i: usize| -> Result<Secret<Vec<Goldilocks>>, LeafError> {
        Ok(Secret::new(match lent(i)? {
            Lent::Tensor(values) => values.iter().map(|&v| Goldilocks::from_i64(v)).collect(),
            Lent::Forest(leaves) => leaves.to_vec(),
        }))
    };
    let mut kept = match mode {
        MergeMode::InMemory => None,
        MergeMode::TwoPass { j_star } => {
            let j_star = j_star.unwrap_or_else(|| balanced_j_star(&polynomials, grouped.count()));
            Some(
                MergeTables::new(&polynomials, grouped.count(), j_star).map_err(|_| {
                    LeafError::Internal {
                        rule: "j* is at most the merge's table rounds",
                    }
                })?,
            )
        }
    };

    let (alpha, beta) = begin_groups(transcript);
    let mut groups = Vec::with_capacity(grouped.count());
    let mut splits = Vec::with_capacity(grouped.count());
    let mut points = Vec::with_capacity(grouped.count());
    let mut claims = Vec::with_capacity(grouped.count());
    let mut gammas = Vec::with_capacity(grouped.count());
    let tensors = grouped.polynomials().len();
    for h in 0..grouped.count() {
        let layout = grouped.group(h)?;
        let (tokens, layers) = (layout.tokens(), layout.layers());
        let columns_h = group_columns(stage, grouped, h, &layout, tables, witness)?;
        let instances = columns_h
            .iter()
            .map(|(table, keys, values)| {
                let k = stage
                    .instances()
                    .iter()
                    .position(|i| i.table == *table)
                    .ok_or(LeafError::Internal {
                        rule: "a group's instances are the stage's",
                    })?;
                Ok(LogUpInstance::new(
                    columns(tables, *table)?.tuples(),
                    Tuples::new(keys, values.as_deref()),
                    multiplicities[k],
                ))
            })
            .collect::<Result<Vec<_>, LeafError>>()?;
        let shares = forests
            .iter()
            .map(|forest| forest_share(forest, g_t, g_l, tokens, layers))
            .collect::<Result<Vec<_>, _>>()?;
        let group_forests: Vec<Forest<'_, GoldilocksExt2>> = shares
            .iter()
            .zip(forests)
            .map(|(share, forest)| Forest::new(share, forest.height as usize))
            .collect();
        let mut schedule: Vec<TreeId> = (0..instances.len()).map(TreeId::Lookup).collect();
        schedule.extend((0..group_forests.len()).map(TreeId::Forest));
        let index = u32::try_from(h).map_err(|_| LeafError::Internal {
            rule: "a group index fits 32 bits",
        })?;
        let proven = prove_group(
            transcript,
            index,
            alpha,
            beta,
            &instances,
            &group_forests,
            &schedule,
        )?;
        let point = level_point(proven.levels());

        // The split: each tensor at `P_h` outside `S` and at the group's bits on `S`.
        let mut values = Vec::with_capacity(layout.reads().len());
        let mut group_claims = Vec::with_capacity(layout.reads().len() + forests.len());
        for read in layout.reads() {
            let shape = &grouped.polynomials()[read.polynomial];
            let free = read
                .coords
                .iter()
                .map(|&c| point.get(c as usize).copied())
                .collect::<Option<Vec<_>>>()
                .ok_or(LeafError::Internal {
                    rule: "the group's leaf point holds every tensor's coordinates",
                })?;
            let mut at = Vec::with_capacity(shape.vars);
            let (mut next_free, mut next_bit) = (free.iter(), 0u32);
            for var in 0..shape.vars {
                at.push(match shape.group_vars.binary_search(&var) {
                    Ok(_) => {
                        let bit = (read.group_bits >> next_bit) & 1;
                        next_bit += 1;
                        if bit == 1 {
                            GoldilocksExt2::ONE
                        } else {
                            GoldilocksExt2::ZERO
                        }
                    }
                    Err(_) => *next_free.next().ok_or(LeafError::Internal {
                        rule: "a tensor's claim has a coordinate per free variable",
                    })?,
                });
            }
            let entries = stage
                .reads(read.site, read.layer)
                .and_then(|reads| reads.get(read.read))
                .ok_or(LeafError::Internal {
                    rule: "a group's split reads the stage's tensors",
                })
                .and_then(|r| witness[read.site][read.layer].of(r))?;
            let mut table =
                SecretBuf::try_with_capacity(entries.len()).map_err(|_| LeafError::Memory)?;
            table.extend(entries.iter().map(|&v| Goldilocks::from_i64(v)));
            let value =
                evaluate_base::<GoldilocksExt2>(&table, &at).map_err(|_| LeafError::Internal {
                    rule: "a tensor's point has its variables",
                })?;
            values.push(value);
            group_claims.push(MergeClaim {
                polynomial: read.polynomial,
                point: free,
                group_bits: read.group_bits,
                value,
            });
        }
        transcript.absorb_message(&format!("group/{h}/split"), &values);
        for (f, claim) in proven.forest_claims().iter().enumerate() {
            group_claims.push(MergeClaim {
                polynomial: tensors + f,
                point: claim.point.clone(),
                group_bits: h as u64,
                value: claim.value,
            });
        }
        let gamma = transcript.challenge(&format!("group/{h}/merge/gamma"));
        // The group's merge tables, while its blocks are resident (RFC-0054 'Group pass').
        if let Some(kept) = kept.as_mut() {
            let mut blocks: Vec<(usize, u64, Secret<Vec<Goldilocks>>)> =
                Vec::with_capacity(group_claims.len());
            for claim in &group_claims {
                let source = lent(claim.polynomial)?;
                let entry = |x: usize| match source {
                    Lent::Tensor(values) => values.get(x).map(|&v| Goldilocks::from_i64(v)),
                    Lent::Forest(leaves) => leaves.get(x).copied(),
                };
                let values = block(entry, &polynomials[claim.polynomial], claim.group_bits).ok_or(
                    LeafError::Internal {
                        rule: "a polynomial's table holds its blocks",
                    },
                )?;
                blocks.push((claim.polynomial, claim.group_bits, Secret::new(values)));
            }
            kept.add_group(h, &group_claims, gamma, |i, bits| {
                blocks
                    .iter()
                    .find(|(p, b, _)| (*p, *b) == (i, bits))
                    .map(|(_, _, t)| t)
            })
            .map_err(|_| LeafError::Internal {
                rule: "a group's merge tables take its claims and blocks",
            })?;
        }
        gammas.push(gamma);
        claims.push(group_claims);
        splits.push(values);
        points.push(point);
        groups.push(proven);
    }

    // The table batch: every table tree, its multiplicities counting every group. Its lookups
    // enter no tree; row 0's tuple stands for them.
    let table_instances = stage
        .instances()
        .iter()
        .zip(multiplicities)
        .map(|(instance, &m)| {
            let rows = columns(tables, instance.table)?.tuples();
            let row_0 = Tuples::new(&rows.keys[..1], rows.values.map(|v| &v[..1]));
            Ok(LogUpInstance::new(rows, row_0, m))
        })
        .collect::<Result<Vec<_>, LeafError>>()?;
    let table_batch = prove_table_batch(transcript, alpha, beta, &table_instances)?;

    // The merge, the pass P2b counts (RFC-0054 'Pass structure').
    let merge_started = std::time::Instant::now();
    let merge_beta = transcript.challenge("merge/beta");
    let merge =
        Merge::new(polynomials, claims, gammas, merge_beta).map_err(|_| LeafError::Internal {
            rule: "the groups' claims fit the merge's polynomials",
        })?;
    let proven_merge = match kept {
        None => {
            let tables = (0..merge.polynomials().len())
                .map(table)
                .collect::<Result<Vec<_>, _>>()?;
            let table_refs: Vec<&Secret<Vec<Goldilocks>>> = tables.iter().collect();
            prove_merge(transcript, &merge, &table_refs)
        }
        Some(kept) => {
            // P2b: every polynomial regenerated once, after the tables' rounds.
            let mut failed = None;
            let proven = prove_merge_two_pass(transcript, &merge, &kept, |i| {
                table(i).unwrap_or_else(|e| {
                    failed = Some(e);
                    Secret::new(Vec::new())
                })
            });
            if let Some(e) = failed {
                return Err(e);
            }
            proven
        }
    }
    .map_err(|_| LeafError::Internal {
        rule: "the merge takes one table per polynomial",
    })?;
    let mut merged = Vec::with_capacity(merge.polynomials().len());
    for (i, polynomial) in merge.polynomials().iter().enumerate() {
        let point = merged_point(
            proven_merge.point(),
            polynomial.vars,
            &polynomial.group_vars,
        )
        .ok_or(LeafError::Internal {
            rule: "the merge proves every polynomial",
        })?;
        let value = *proven_merge
            .proof()
            .final_evals
            .get(i)
            .ok_or(LeafError::Internal {
                rule: "the merge evaluates every polynomial",
            })?;
        merged.push(Claim { point, value });
    }
    tessera_codec::metrics::pass(tessera_codec::metrics::Pass::P2b, merge_started.elapsed());
    Ok(ProvenGroupedLeaf {
        groups,
        splits,
        points,
        tables: table_batch,
        merge: proven_merge,
        merged,
    })
}
