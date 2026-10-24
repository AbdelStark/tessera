// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A grouped leaf stage's layouts, from public data alone (RFC-0054 'Groups', 'What a group
//! holds'): the blocks each token group holds, its instances laid out by RFC-0045 over them, and
//! its split, as claims on the merge's polynomials.
//!
//! - **Groups.** `G = 2^{g_T + g_L}` of them. Group `h` sets the token index's low `g_T` bits to
//!   `s = h mod 2^{g_T}`, its token subset, and the layer index's low `g_L` bits to
//!   `o = ⌊h / 2^{g_T}⌋`, its layer subset.
//! - **Blocks.** Group `h` holds the stage's blocks at its layer subset's layers; the prologue's
//!   and the epilogue's are in layer subset 0. A block with a token axis is restricted to token
//!   subset `s`. Its entries are the stage block's whose token index has low bits `s`, in the
//!   same order, so its variables are the block's but the token axis's lowest `g_T`: `2^{k − g_T}`
//!   entries. A block without a token axis belongs to token subset 0, whole.
//! - **Instances.** Each table the group holds a block of is one of its instances, laid out by
//!   RFC-0045 'The column' over the group's blocks. A group holding no block of a table has no
//!   tree for it.
//! - **The split.** For each site in node order and each of its layers in the group, one value
//!   per tensor the group's blocks read, in the ungrouped split's order (RFC-0045 'What is
//!   sent'), so every group follows one order. Each is a claim on a merged
//!   polynomial: the tensor at that site and layer, one per value of the ungrouped split, in its
//!   order ('The merge').
//!   - **Group variables.** `S` is the token axis's low `g_T` bits among the tensor's variables:
//!     all of them when the tensor is read over its block's token axis, none otherwise.
//!   - **The claim.** At the group's leaf point `P_h` on the other variables, through the
//!     coordinates RFC-0045 gives them in the group's block, and at `s` on `S`.
//!
//! With `g = 0`, the one group is the stage: its instances, its split and its points.

use std::collections::BTreeMap;

use tessera_compiler::schedule::{LayerMap, NodeRef};
use tessera_protocol::{LeafGroups, MAX_LEAF_GROUP_BITS};
use tessera_sumcheck::MergePolynomial;
use tessera_tables::TableId;

use super::layout::{axis_vars, instance_layouts};
use super::{Block, InstanceLayout, LeafError, LeafLayout, SiteLookup, internal};

/// One value of a group's split: a claim on a merged polynomial (RFC-0054 'The merge').
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GroupRead {
    /// The site, in the stage layout's sites.
    pub site: usize,
    /// The layer, in the site's layers.
    pub layer: usize,
    /// The tensor, in the stage layout's reads at that site and layer.
    pub read: usize,
    /// The merged polynomial: the index of the ungrouped split's value for that site, layer and
    /// tensor.
    pub polynomial: usize,
    /// The claim's point on the polynomial's free variables, the ones outside `S`: indices into
    /// the group's leaf point `P_h`, lowest variable first.
    pub coords: Vec<u32>,
    /// The claim's bits on `S`: the group's token subset `s`, or 0 when `S` is empty.
    pub group_bits: u64,
}

/// One group's layout (RFC-0054 'What a group holds').
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupLayout {
    tokens: u64,
    layers: u64,
    instances: Vec<InstanceLayout>,
    reads: Vec<GroupRead>,
}

impl GroupLayout {
    /// `s`, the group's token subset: the token index's low `g_T` bits.
    pub fn tokens(&self) -> u64 {
        self.tokens
    }

    /// `o`, the group's layer subset: the layer index's low `g_L` bits.
    pub fn layers(&self) -> u64 {
        self.layers
    }

    /// The group's instances, one per table it holds a block of, in table order. Each block names
    /// the stage's site, layer and lookup, and its `vars` are the group's share's
    /// ([`GroupedLayout::source`]).
    pub fn instances(&self) -> &[InstanceLayout] {
        &self.instances
    }

    /// The group's split, in the order it is sent.
    pub fn reads(&self) -> &[GroupRead] {
        &self.reads
    }
}

/// A grouped leaf stage's layouts (RFC-0054 'Groups'): the merge's polynomials, and each site and
/// layer's share of every group, from which [`group`](Self::group) lays out one group at a time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupedLayout {
    groups: LeafGroups,
    polynomials: Vec<MergePolynomial>,
    /// Each polynomial's coordinates in its groups' blocks, outside its group variables.
    coords: Vec<Vec<u32>>,
    /// For each site, layer and lookup, the lowest of its block's variables the token axis takes,
    /// whose lowest `g_T` the groups fix; `None` for a block without a token axis.
    fixed: Vec<Vec<Vec<Option<u32>>>>,
    shares: Vec<Share>,
}

/// One site and layer's share of the groups of its layer subset.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Share {
    site: usize,
    layer: usize,
    /// Its layer subset `o`.
    subset: u64,
    /// The polynomial of its first tensor.
    first: usize,
    /// Its blocks in lookup order, each with its table and whether it has a token axis: restricted
    /// in every token subset if so, whole in token subset 0 if not.
    blocks: Vec<(TableId, Block, bool)>,
    /// The tensors the restricted blocks read, as indices into its reads.
    reads: Vec<usize>,
    /// The tensors every block reads, for token subset 0.
    reads_zero: Vec<usize>,
}

impl GroupedLayout {
    /// The layouts of `stage`'s groups under `groups`, its layers `layers` (RFC-0054 'Groups',
    /// 'What a group holds'):
    ///
    /// 1. **Polynomials.** One per value of the ungrouped split, in its order, each with its
    ///    variables, its group variables `S`, and its coordinates in its groups' blocks outside
    ///    them.
    /// 2. **Shares.** Each site and layer's blocks, restricted to a token subset or whole, and the
    ///    tensors they read, for the groups of its layer subset.
    ///
    /// # Errors
    ///
    /// [`LeafError::Internal`] (`ERR-INTERNAL`) for the first rule broken, in this order: more than
    /// 14 group bits, or more layer bits than the layer index has; then site by site, layer by
    /// layer and lookup by lookup, a token axis shorter than `2^{g_T}` padded; then tensor by
    /// tensor, lookups that read one tensor with different group variables or, outside them,
    /// different coordinates in their groups' blocks.
    pub fn new<T: Copy + Eq>(
        stage: &LeafLayout<T>,
        layers: &LayerMap,
        groups: LeafGroups,
    ) -> Result<Self, LeafError> {
        if groups.bits() > MAX_LEAF_GROUP_BITS {
            return internal("a grouping fixes at most 14 bits (RFC-0054 'Messages')");
        }
        if u32::from(groups.layer_bits) > layers.vars {
            return internal("a grouping fixes at most the layer index's bits");
        }
        let g_t = u32::from(groups.token_bits);
        let mut fixed = Vec::with_capacity(stage.sites().len());
        for site in stage.sites() {
            let mut per_layer = Vec::with_capacity(site.layers.len());
            for gadget in &site.layers {
                let mut per_lookup = Vec::with_capacity(gadget.lookups.len());
                for lookup in &gadget.lookups {
                    per_lookup.push(match lookup.token {
                        Some(a) => {
                            if axis_vars(lookup.domain[a]) < g_t {
                                return internal(
                                    "a token axis holds the groups' token bits (RFC-0054 'Messages')",
                                );
                            }
                            // The axes after the token axis take the block's lower variables.
                            Some(lookup.domain[a + 1..].iter().map(|&len| axis_vars(len)).sum())
                        }
                        None => None,
                    });
                }
                per_layer.push(per_lookup);
            }
            fixed.push(per_layer);
        }

        let layer_mask = (1u64 << groups.layer_bits) - 1;
        let mut polynomials = Vec::with_capacity(stage.split_len());
        let mut coords = Vec::with_capacity(stage.split_len());
        let mut shares = Vec::new();
        for (s, site) in stage.sites().iter().enumerate() {
            for (l, gadget) in site.layers.iter().enumerate() {
                let reads = stage.reads(s, l).ok_or(LeafError::Internal {
                    rule: "the layout reads every site's tensors",
                })?;
                // Each tensor's group variables and its coordinates in its groups' blocks.
                for read in reads {
                    let mut shape = None;
                    for (j, lookup) in gadget.lookups.iter().enumerate() {
                        if !reads_term(lookup, read.term) {
                            continue;
                        }
                        let own = group_coords(&read.coords, fixed[s][l][j], g_t);
                        match &shape {
                            None => shape = Some(own),
                            Some(first) if *first == own => {}
                            Some(_) => {
                                return internal(
                                    "the lookups that read a tensor give it one point in their \
                                     groups' blocks (RFC-0054 'Groups')",
                                );
                            }
                        }
                    }
                    let (group_vars, free) = shape.ok_or(LeafError::Internal {
                        rule: "a tensor the split sends is read by a lookup",
                    })?;
                    polynomials.push(MergePolynomial {
                        vars: free.len() + group_vars.len(),
                        group_vars,
                    });
                    coords.push(free);
                }
                let mut share = Share {
                    site: s,
                    layer: l,
                    subset: match site.node {
                        NodeRef::Position(_) => l as u64 & layer_mask,
                        NodeRef::Node(_) => 0,
                    },
                    first: coords.len() - reads.len(),
                    blocks: Vec::with_capacity(gadget.lookups.len()),
                    reads: Vec::new(),
                    reads_zero: Vec::new(),
                };
                let mut held = vec![(false, false); reads.len()];
                for (j, lookup) in gadget.lookups.iter().enumerate() {
                    let restricted = fixed[s][l][j].is_some();
                    let block = Block {
                        site: s,
                        layer: l,
                        lookup: j,
                        vars: lookup.vars() - if restricted { g_t } else { 0 },
                        offset: 0,
                    };
                    share.blocks.push((lookup.lookup.table, block, restricted));
                    for (r, read) in reads.iter().enumerate() {
                        let reads_it = reads_term(lookup, read.term);
                        held[r].0 |= reads_it && restricted;
                        held[r].1 |= reads_it;
                    }
                }
                for (r, &(every, zero)) in held.iter().enumerate() {
                    if every {
                        share.reads.push(r);
                    }
                    if zero {
                        share.reads_zero.push(r);
                    }
                }
                shares.push(share);
            }
        }
        Ok(Self {
            groups,
            polynomials,
            coords,
            fixed,
            shares,
        })
    }

    /// The grouping.
    pub fn groups(&self) -> LeafGroups {
        self.groups
    }

    /// `G = 2^{g_T + g_L}`, the groups.
    pub fn count(&self) -> usize {
        1 << self.groups.bits()
    }

    /// The merge's polynomials: one per value of the ungrouped split, in its order, each with
    /// its group variables (RFC-0054 'The merge').
    pub fn polynomials(&self) -> &[MergePolynomial] {
        &self.polynomials
    }

    /// Group `group`'s layout (RFC-0054 'What a group holds'): the blocks of its layer subset's
    /// shares, restricted to its token subset, and those without a token axis in token subset 0;
    /// its instances laid out over them by RFC-0045 'The column'; and its split, each value with
    /// its polynomial, its point's coordinates in `P_h` and its bits on `S`.
    ///
    /// # Errors
    ///
    /// [`LeafError::Internal`] for a group past [`count`](Self::count), or entries that overflow
    /// 64 bits, which a stage layout that holds them rules out.
    pub fn group(&self, group: usize) -> Result<GroupLayout, LeafError> {
        if group >= self.count() {
            return internal("a group of the grouping");
        }
        let h = group as u64;
        let g_t = self.groups.token_bits;
        let (tokens, subset) = (h & ((1u64 << g_t) - 1), h >> g_t);
        let mut tables: BTreeMap<TableId, Vec<Block>> = BTreeMap::new();
        let mut reads = Vec::new();
        for share in self.shares.iter().filter(|share| share.subset == subset) {
            let held = match tokens {
                0 => &share.reads_zero,
                _ => &share.reads,
            };
            // In site, layer and lookup order, which the instances' stable sort keeps.
            for &(table, block, restricted) in &share.blocks {
                if restricted || tokens == 0 {
                    tables.entry(table).or_default().push(block);
                }
            }
            for &r in held {
                let polynomial = share.first + r;
                reads.push(GroupRead {
                    site: share.site,
                    layer: share.layer,
                    read: r,
                    polynomial,
                    coords: self.coords[polynomial].clone(),
                    group_bits: match self.polynomials[polynomial].group_vars.is_empty() {
                        true => 0,
                        false => tokens,
                    },
                });
            }
        }
        Ok(GroupLayout {
            tokens,
            layers: subset,
            instances: instance_layouts(tables)?,
            reads,
        })
    }

    /// The entry of the stage's block that entry `index` of group `group`'s share of `block`
    /// holds: `index` with the group's token bits inserted at the token axis's lowest variables,
    /// or `index` itself for a block without a token axis. `None` for a group or a block outside
    /// the layout, or an index past the share's `2^{vars}` entries.
    pub fn source(&self, group: usize, block: &Block, index: u64) -> Option<u64> {
        if group >= self.count() {
            return None;
        }
        let tokens = group as u64 & ((1u64 << self.groups.token_bits) - 1);
        let fixed = *self
            .fixed
            .get(block.site)?
            .get(block.layer)?
            .get(block.lookup)?;
        if index.checked_shr(block.vars)? != 0 {
            return None;
        }
        Some(match fixed {
            Some(p) => {
                let low = index & ((1u64 << p) - 1);
                ((index >> p) << (p + u32::from(self.groups.token_bits))) | (tokens << p) | low
            }
            None => index,
        })
    }
}

/// Whether `lookup`'s key or value form names `term`.
fn reads_term<T: Copy + Eq>(lookup: &SiteLookup<T>, term: T) -> bool {
    core::iter::once(&lookup.lookup.key)
        .chain(&lookup.lookup.value)
        .any(|form| form.terms.iter().any(|&(_, t)| t == term))
}

/// A tensor's group variables and its coordinates in a group's block, from its `coords` in the
/// stage's block, the token axis's low `g_t` bits taking the block's variables from `fixed`: the
/// positions among its variables that those bits take, and its other coordinates, lowered past
/// them.
fn group_coords(coords: &[u32], fixed: Option<u32>, g_t: u32) -> (Vec<usize>, Vec<u32>) {
    let Some(p) = fixed else {
        return (Vec::new(), coords.to_vec());
    };
    let mut group_vars = Vec::new();
    let mut free = Vec::with_capacity(coords.len());
    for (v, &c) in coords.iter().enumerate() {
        match c {
            c if c < p => free.push(c),
            c if c < p + g_t => group_vars.push(v),
            c => free.push(c - g_t),
        }
    }
    (group_vars, free)
}
