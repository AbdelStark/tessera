// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0054 'What a group holds': each group's blocks, instances and split, from public data.
//!
//! - **Token axes.** Every model lookup runs over a token-indexed tensor, at its token axis.
//! - **`g = 0`.** The one group is the stage.
//! - **Partition.** Under every grouping, each entry of each stage block is in exactly one
//!   group's block: the group of its layer's subset and its token's low bits, decoded here from
//!   the block's index order.
//! - **The split.** A group block's extension at the group's point is its forms at the group's
//!   split values, each a tensor's extension at its claim: the coordinates on its free variables,
//!   the group's bits on `S`.
//! - **Shapes.** Hand-derived group variables and coordinates; a block without a token axis in
//!   token subset 0 alone; unlike and empty groups.
//! - **Refusals.** Each rule of [`GroupedLayout::new`] broken, at its rule.

use std::collections::BTreeMap;

use super::*;
use tessera_gadgets::{Affine, Lookup};
use tessera_protocol::LeafGroups;
use tessera_prover::leaf::{
    Block, GroupLayout, GroupedLayout, InstanceLayout, LeafError, Site, SiteGadget, SiteLookup,
};
use tessera_sumcheck::{Merge, MergeClaim};
use tessera_tables::TableId;

fn policy() -> GadgetPolicy {
    GadgetPolicy {
        max_log2_rows: 21,
        query_block: 2,
        ..GadgetPolicy::default()
    }
}

/// The model's public data and leaf layout at `n` tokens.
fn stage(model: &IntegerModel, n: u32) -> (Public, LeafLayout<ModelColumn>) {
    let p = public(model, n, policy());
    let layout = LeafLayout::model(&p.lowering, &p.layers, &p.selection, &p.claims).unwrap();
    (p, layout)
}

fn groups(token_bits: u8, layer_bits: u8) -> LeafGroups {
    LeafGroups {
        token_bits,
        layer_bits,
    }
}

/// `⌈log2 len⌉`.
fn vars(len: u32) -> u32 {
    len.next_power_of_two().trailing_zeros()
}

/// The index along axis `axis` of a block's entry `x`: the domain padded per axis, the last axis
/// taking the lowest bits.
fn index_along(domain: &[u32], axis: usize, x: u64) -> u64 {
    let shift: u32 = domain[axis + 1..].iter().map(|&len| vars(len)).sum();
    (x >> shift) & ((1 << vars(domain[axis])) - 1)
}

/// `eq(point, x)` over `point`'s variables, the lowest first.
fn eq(point: &[E], x: u64) -> E {
    point.iter().enumerate().fold(E::ONE, |acc, (bit, &r)| {
        acc * if x >> bit & 1 == 1 { r } else { E::ONE - r }
    })
}

fn form(affine: &Affine<G, ModelColumn>, at: impl Fn(ModelColumn) -> E) -> E {
    affine
        .terms
        .iter()
        .fold(E::from(v(affine.constant)), |sum, &(a, term)| {
            sum + E::from(v(a)) * at(term)
        })
}

/// A seeded generator.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn e(&mut self) -> E {
        let mut fp = || V::from_canonical(self.next() % 0xffff_ffff_0000_0001).unwrap();
        E::new(fp(), fp())
    }
}

/// Every group's layout, in order.
fn all(grouped: &GroupedLayout) -> Vec<GroupLayout> {
    (0..grouped.count())
        .map(|h| grouped.group(h).unwrap())
        .collect()
}

/// An instance's column is RFC-0045's: its blocks larger first, stably by site, layer and lookup,
/// each at a multiple of its size, and a height of `⌈log2 m⌉`.
fn assert_column(instance: &InstanceLayout) {
    let mut sorted = instance.blocks.clone();
    sorted.sort_by_key(|b| (core::cmp::Reverse(b.vars), b.site, b.layer, b.lookup));
    assert_eq!(sorted, instance.blocks, "larger first, then in site order");
    let mut offset = 0;
    for block in &instance.blocks {
        assert_eq!(block.offset, offset);
        assert_eq!(block.offset % block.entries(), 0);
        offset += block.entries();
    }
    assert_eq!(instance.entries, offset);
    assert!(
        1u64 << instance.height >= offset
            && (instance.height == 0 || 1u64 << (instance.height - 1) < offset)
    );
}

#[test]
fn every_model_lookup_runs_over_its_tensors_token_axis() {
    for (model, n) in [(bounded_model(), 3), (layered(bounded_model()), 5)] {
        let (_, layout) = stage(&model, n);
        let mut scores = 0;
        for site in layout.sites() {
            for lookup in site.layers.iter().flat_map(|g| &g.lookups) {
                let a = lookup.token.expect("a token-indexed domain");
                assert_eq!(
                    lookup.domain[a], n,
                    "the token axis is as long as the batch"
                );
                if lookup.domain.len() == 3 {
                    // The scores `[heads, queries, keys]`: the queries.
                    assert_eq!(a, 1);
                    scores += 1;
                }
            }
        }
        assert!(scores > 0);
    }
}

#[test]
fn ungrouped_the_one_group_is_the_stage() {
    let (p, layout) = stage(&layered(bounded_model()), 5);
    let grouped = GroupedLayout::new(&layout, &p.layers, LeafGroups::NONE).unwrap();
    assert_eq!(grouped.count(), 1);
    let group = &grouped.group(0).unwrap();
    assert_eq!((group.tokens(), group.layers()), (0, 0));
    assert_eq!(group.instances(), layout.instances());
    let mut polynomial = 0;
    for (s, site) in layout.sites().iter().enumerate() {
        for l in 0..site.layers.len() {
            for (r, read) in layout.reads(s, l).unwrap().iter().enumerate() {
                let own = &group.reads()[polynomial];
                assert_eq!((own.site, own.layer, own.read), (s, l, r));
                assert_eq!((own.polynomial, own.group_bits), (polynomial, 0));
                assert_eq!(own.coords, read.coords);
                let shape = &grouped.polynomials()[polynomial];
                assert_eq!((shape.vars, shape.group_vars.len()), (read.coords.len(), 0));
                polynomial += 1;
            }
        }
    }
    assert_eq!(
        (group.reads().len(), polynomial),
        (layout.split_len(), layout.split_len())
    );
}

#[test]
fn the_groups_partition_every_block_by_its_tokens_and_layers() {
    // Three layers, padded to four, and 5 tokens, padded to 8.
    let (p, layout) = stage(&layered(bounded_model()), 5);
    for (g_t, g_l) in [(1, 0), (0, 1), (1, 1), (3, 0), (0, 2), (2, 2), (3, 2)] {
        let grouped = GroupedLayout::new(&layout, &p.layers, groups(g_t, g_l)).unwrap();
        assert_eq!(grouped.count(), 1 << (g_t + g_l));
        let mut seen: BTreeMap<(usize, usize, usize), Vec<u32>> = BTreeMap::new();
        for (h, group) in all(&grouped).iter().enumerate() {
            let (s, o) = (h as u64 % (1 << g_t), h as u64 >> g_t);
            assert_eq!((group.tokens(), group.layers()), (s, o));
            for instance in group.instances() {
                assert_column(instance);
                for block in &instance.blocks {
                    let site = &layout.sites()[block.site];
                    let lookup = &site.layers[block.layer].lookups[block.lookup];
                    assert_eq!(lookup.lookup.table, instance.table);
                    let subset = match site.node {
                        NodeRef::Position(_) => block.layer as u64 % (1 << g_l),
                        NodeRef::Node(_) => 0,
                    };
                    assert_eq!(subset, o, "the block's layer is in the group's subset");
                    assert_eq!(block.vars + u32::from(g_t), lookup.vars());
                    let a = lookup.token.unwrap();
                    let counts = seen
                        .entry((block.site, block.layer, block.lookup))
                        .or_insert_with(|| vec![0; 1 << lookup.vars()]);
                    let mut previous = None;
                    for j in 0..block.entries() {
                        let x = grouped.source(h, block, j).expect("an entry of the share");
                        let token = index_along(&lookup.domain, a, x);
                        assert_eq!(
                            token % (1 << g_t),
                            s,
                            "its tokens' low bits are the group's"
                        );
                        assert!(previous < Some(x), "the share keeps the block's order");
                        previous = Some(x);
                        counts[x as usize] += 1;
                    }
                    assert_eq!(grouped.source(h, block, block.entries()), None);
                }
            }
        }
        let blocks: usize = layout.instances().iter().map(|i| i.blocks.len()).sum();
        assert_eq!(seen.len(), blocks, "every block of the stage is in a group");
        assert!(
            seen.values().flatten().all(|&c| c == 1),
            "every entry once, {g_t} token bits and {g_l} layer bits"
        );
    }
}

#[test]
fn a_groups_blocks_evaluate_to_its_split_at_its_claims() {
    let (p, layout) = stage(&layered(bounded_model()), 5);
    let mut rng = Rng(0x0054);
    // One tensor per value of the ungrouped split, and where each site and layer's start.
    let mut first = BTreeMap::new();
    let mut tensors: Vec<Vec<i64>> = Vec::new();
    for (s, site) in layout.sites().iter().enumerate() {
        for l in 0..site.layers.len() {
            first.insert((s, l), tensors.len());
            for read in layout.reads(s, l).unwrap() {
                let entries = 1usize << read.coords.len();
                tensors.push(
                    (0..entries)
                        .map(|_| (rng.next() % 512) as i64 - 256)
                        .collect(),
                );
            }
        }
    }
    for (g_t, g_l) in [(0, 0), (1, 0), (0, 1), (1, 1), (3, 2)] {
        let grouped = GroupedLayout::new(&layout, &p.layers, groups(g_t, g_l)).unwrap();
        let mut claims = Vec::new();
        for (h, group) in all(&grouped).iter().enumerate() {
            let height = group
                .instances()
                .iter()
                .map(|i| i.height)
                .max()
                .unwrap_or(0);
            let point: Vec<E> = (0..height).map(|_| rng.e()).collect();
            // Each split value, the tensor's extension at its claim: the coordinates on the free
            // variables, the group's bits on `S`.
            let mut values = BTreeMap::new();
            for read in group.reads() {
                let shape = &grouped.polynomials()[read.polynomial];
                let mut free = read.coords.iter();
                let mut bits = 0;
                let at: Vec<E> = (0..shape.vars)
                    .map(|var| match shape.group_vars.contains(&var) {
                        true => {
                            bits += 1;
                            match read.group_bits >> (bits - 1) & 1 {
                                1 => E::ONE,
                                _ => E::ZERO,
                            }
                        }
                        false => point[*free.next().unwrap() as usize],
                    })
                    .collect();
                assert!(free.next().is_none());
                assert_eq!(bits, shape.group_vars.len());
                let value = mle(&tensors[read.polynomial], &at);
                assert!(
                    values
                        .insert((read.site, read.layer, read.read), value)
                        .is_none()
                );
            }
            for instance in group.instances() {
                for block in &instance.blocks {
                    let gadget = &layout.sites()[block.site].layers[block.layer];
                    let lookup = &gadget.lookups[block.lookup].lookup;
                    let reads = layout.reads(block.site, block.layer).unwrap();
                    let base = first[&(block.site, block.layer)];
                    let index = |term| reads.iter().position(|r| r.term == term).unwrap();
                    // The share's extension, entry by entry through the stage block's order.
                    let entry = |affine: &Affine<G, ModelColumn>, x: u64| {
                        form(affine, |term| {
                            let r = index(term);
                            let at = reads[r]
                                .coords
                                .iter()
                                .enumerate()
                                .fold(0, |i, (j, &c)| i | ((x >> c & 1) << j));
                            E::from(v(G::from_i64(tensors[base + r][at as usize])))
                        })
                    };
                    let share = &point[..block.vars as usize];
                    for affine in std::iter::once(&lookup.key).chain(&lookup.value) {
                        let direct = (0..block.entries()).fold(E::ZERO, |sum, j| {
                            let x = grouped.source(h, block, j).unwrap();
                            sum + eq(share, j) * entry(affine, x)
                        });
                        let split = form(affine, |term| {
                            values[&(block.site, block.layer, index(term))]
                        });
                        assert_eq!(direct, split, "group {h}, {g_t} and {g_l} bits");
                    }
                }
            }
            claims.push(
                group
                    .reads()
                    .iter()
                    .map(|read| MergeClaim {
                        polynomial: read.polynomial,
                        point: vec![F::ZERO; read.coords.len()],
                        group_bits: read.group_bits,
                        value: F::ZERO,
                    })
                    .collect(),
            );
        }
        // The claims fit the merge's polynomials.
        let gammas = vec![F::ONE; claims.len()];
        Merge::new(grouped.polynomials().to_vec(), claims, gammas, F::ONE).unwrap();
    }
}

/// A key `x + 4·c` over `[4, 8]`, `c` read over `class_axes`, the domain's token axis `token`.
fn synthetic(table: u32, token: Option<usize>, class_axes: Vec<usize>) -> SiteLookup<ModelColumn> {
    SiteLookup {
        domain: vec![4, 8],
        token,
        axes: vec![(ModelColumn::Input(1), class_axes)],
        lookup: Lookup {
            table: TableId(table),
            key: Affine {
                constant: G::ZERO,
                terms: vec![
                    (G::ONE, ModelColumn::Input(0)),
                    (G::from_u64_reduced(4), ModelColumn::Input(1)),
                ],
            },
            value: None,
        },
    }
}

fn site(
    layers: &LayerMap,
    position: u32,
    lookups: Vec<SiteLookup<ModelColumn>>,
) -> Site<ModelColumn> {
    Site {
        node: NodeRef::Position(position),
        layers: vec![SiteGadget { lookups }; layers.count as usize],
    }
}

#[test]
fn group_variables_and_coordinates_follow_the_token_axis() {
    let p = public(&bounded_model(), 3, policy());
    let layers = &p.layers;
    // `[4, 8]`: the channels take variables 0 to 2, the tokens 3 and 4.
    let sites = vec![
        // The classes over the channels: no group variable.
        site(layers, 1, vec![synthetic(0, Some(0), vec![1])]),
        // The classes over the tokens: the token axis's low bit.
        site(layers, 2, vec![synthetic(0, Some(0), vec![0])]),
        // No token axis: token subset 0, whole.
        site(layers, 3, vec![synthetic(1, None, vec![1])]),
    ];
    let layout = LeafLayout::new(layers, sites).unwrap();
    let grouped = GroupedLayout::new(&layout, layers, groups(1, 0)).unwrap();
    let shapes: Vec<_> = grouped
        .polynomials()
        .iter()
        .map(|p| (p.vars, p.group_vars.clone()))
        .collect();
    let per_layer = [
        (5, vec![3]), // the input: its token bit, variable 3
        (3, vec![]),  // the classes over the channels
        (5, vec![3]),
        (2, vec![0]), // the classes over the tokens: variables 3 and 4 of the block
        (5, vec![]),  // no token axis
        (3, vec![]),
    ];
    let expected: Vec<_> = (0..3)
        .flat_map(|s| {
            let pair = per_layer[2 * s..2 * s + 2].to_vec();
            core::iter::repeat_n(pair, layers.count as usize).flatten()
        })
        .collect();
    assert_eq!(shapes, expected);
    for (h, group) in all(&grouped).iter().enumerate() {
        let reads: Vec<_> = group
            .reads()
            .iter()
            .filter(|r| r.layer == 0)
            .map(|r| (r.site, r.read, r.coords.clone(), r.group_bits))
            .collect();
        let h = h as u64;
        let mut expected = vec![
            // The input, its variable 4 lowered past the group's bit.
            (0, 0, vec![0, 1, 2, 3], h),
            (0, 1, vec![0, 1, 2], 0),
            (1, 0, vec![0, 1, 2, 3], h),
            (1, 1, vec![3], h),
        ];
        if h == 0 {
            expected.extend([(2, 0, vec![0, 1, 2, 3, 4], 0), (2, 1, vec![0, 1, 2], 0)]);
        }
        assert_eq!(reads, expected, "group {h}");
        let tables: Vec<_> = group.instances().iter().map(|i| i.table).collect();
        assert_eq!(
            tables,
            if h == 0 {
                vec![TableId(0), TableId(1)]
            } else {
                vec![TableId(0)]
            }
        );
        let block = |table: usize| &group.instances()[table].blocks[0];
        assert_eq!(block(0).vars, 4, "a token-indexed block's share");
        if h == 0 {
            assert_eq!(block(1).vars, 5, "a block without a token axis, whole");
        }
    }
}

#[test]
fn groups_of_unlike_layers_and_empty_groups() {
    // Three layers, padded to four: layer subset 3 holds the padding alone.
    let (p, layout) = stage(&layered(bounded_model()), 5);
    let grouped = GroupedLayout::new(&layout, &p.layers, groups(0, 2)).unwrap();
    let blocks = |h: usize| -> Vec<Block> {
        grouped
            .group(h)
            .unwrap()
            .instances()
            .iter()
            .flat_map(|i| i.blocks.iter().copied())
            .collect()
    };
    let sizes = |h: usize| -> Vec<u32> {
        let mut v: Vec<u32> = blocks(h).iter().map(|b| b.vars).collect();
        v.sort_unstable();
        v
    };
    let empty = &grouped.group(3).unwrap();
    assert_eq!(
        grouped.group(4),
        Err(LeafError::Internal {
            rule: "a group of the grouping"
        })
    );
    assert!(empty.instances().is_empty() && empty.reads().is_empty());
    // Subset 0 holds layer 0 and the prologue's and epilogue's sites; 1 and 2 one layer each.
    assert!(
        blocks(0)
            .iter()
            .any(|b| matches!(layout.sites()[b.site].node, NodeRef::Node(_)))
    );
    assert!(blocks(0).len() > blocks(1).len());
    assert_eq!(sizes(1), sizes(2));
    for h in 1..3 {
        assert!(blocks(h).iter().all(|b| b.layer == h));
    }
}

#[test]
fn a_grouping_the_layout_cannot_take_is_refused_at_its_rule() {
    let (p, layout) = stage(&bounded_model(), 3);
    let refused = |layout: &LeafLayout<ModelColumn>, g: LeafGroups, rule: &'static str| {
        assert_eq!(
            GroupedLayout::new(layout, &p.layers, g),
            Err(LeafError::Internal { rule })
        );
    };
    refused(
        &layout,
        groups(8, 7),
        "a grouping fixes at most 14 bits (RFC-0054 'Messages')",
    );
    // Two layers: one layer bit.
    assert_eq!(p.layers.vars, 1);
    refused(
        &layout,
        groups(0, 2),
        "a grouping fixes at most the layer index's bits",
    );
    GroupedLayout::new(&layout, &p.layers, groups(2, 1)).unwrap();
    // Three tokens: two token bits.
    refused(
        &layout,
        groups(3, 0),
        "a token axis holds the groups' token bits (RFC-0054 'Messages')",
    );
    // One tensor, the input, read with its token axis by one lookup and without by another.
    let mixed = site(
        &p.layers,
        1,
        vec![synthetic(0, Some(0), vec![1]), synthetic(1, None, vec![1])],
    );
    let mixed = LeafLayout::new(&p.layers, vec![mixed]).unwrap();
    GroupedLayout::new(&mixed, &p.layers, groups(0, 1)).unwrap();
    refused(
        &mixed,
        groups(1, 0),
        "the lookups that read a tensor give it one point in their groups' blocks (RFC-0054 \
         'Groups')",
    );
    // A token axis outside the domain.
    let outside = site(&p.layers, 1, vec![synthetic(0, Some(2), vec![1])]);
    assert_eq!(
        LeafLayout::new(&p.layers, vec![outside]),
        Err(LeafError::Internal {
            rule: "a lookup's token axis is an axis of its domain"
        })
    );
}
