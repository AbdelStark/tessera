// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The leaf layout from public data alone (RFC-0045 'Blocks', 'The column', 'The leaves', 'The
//! split'): the sites, each instance's blocks with their offsets and its height, the leaf order
//! `σ_h`, and the tensors each site's split sends.

use std::collections::BTreeMap;

use tessera_compiler::Lowering;
use tessera_compiler::Selection;
use tessera_compiler::schedule::{LayerMap, NodeRef};
use tessera_field::Goldilocks;
use tessera_gadgets::{Lookup, Requant, Term};
use tessera_ir::{NodeId, OpClass};
use tessera_logup::TreeId;
use tessera_tables::TableId;

use super::{LeafError, internal};

/// The most variables a block may have: its `2^k` entries, and every offset and total of an
/// instance, then fit a `u64` with room to spare.
const MAX_BLOCK_VARS: u32 = 48;

/// One lookup of a site's gadget in one layer: its table and its forms (RFC-0009), with the axes
/// of the gadget's domain over which each term's tensor is read (RFC-0045 'A block's claim').
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SiteLookup<T = Term> {
    /// The table, the key form and the value form.
    pub lookup: Lookup<Goldilocks, T>,
    /// The lookup domain, outermost axis first, before per-axis padding.
    pub domain: Vec<u32>,
    /// The terms read over some of the domain's axes, each with those axes as indices into the
    /// domain's shape, ascending. A term not listed is read over every axis.
    pub axes: Vec<(T, Vec<usize>)>,
}

/// A site's gadget in one layer, its per-request domains and RFC-0009 lookup order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SiteGadget<T = Term> {
    /// The lookups, in RFC-0009's order.
    pub lookups: Vec<SiteLookup<T>>,
}

impl SiteGadget<Term> {
    /// A requantization gadget, `Requant1`, `RequantSplit` or `RequantWide`, over an input of
    /// `shape`: every lookup runs over the input's elements, the classes are read over the last
    /// axis, the output channels, and every other term over every axis (RFC-0009, RFC-0045).
    pub fn requant(gadget: &Requant, shape: &[u32]) -> Self {
        let channels = vec![shape.len().saturating_sub(1)];
        let lookups = gadget
            .constraints::<Goldilocks>()
            .lookups
            .into_iter()
            .map(|lookup| {
                let reads_classes = core::iter::once(&lookup.key)
                    .chain(&lookup.value)
                    .any(|form| form.terms.iter().any(|&(_, t)| t == Term::Class));
                let axes = match reads_classes {
                    true => vec![(Term::Class, channels.clone())],
                    false => Vec::new(),
                };
                SiteLookup {
                    lookup,
                    axes,
                    domain: shape.to_vec(),
                }
            })
            .collect();
        Self { lookups }
    }
}

impl<T> SiteLookup<T> {
    /// Sum of the padded domain axes' variable counts.
    pub fn vars(&self) -> u32 {
        self.domain
            .iter()
            .fold(0u32, |sum, &len| sum.saturating_add(axis_vars(len)))
    }
}

/// One site of the leaf stage: a gadget node, with its gadget in each of its real layers
/// (RFC-0045 'Sites').
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Site<T = Term> {
    /// The node: a template position, one site for all its layers, or a node of the prologue or
    /// the epilogue.
    pub node: NodeRef,
    /// Its gadget in each real layer, in order; one for a site outside the layers. A padded layer
    /// has none.
    pub layers: Vec<SiteGadget<T>>,
}

impl Site<Term> {
    /// The requantization site of `node`: in each real layer, or once outside the layers, the
    /// `Requant1`, `RequantSplit` or `RequantWide` gadget that `selection` gives the node, over
    /// its input's shape (RFC-0006 'Gadget selection', RFC-0009). Public data alone: the graph's
    /// shapes, the selection and its tables.
    ///
    /// # Errors
    ///
    /// [`LeafError::Internal`] for a position past the template, a node inside the layers named
    /// as a node, a node without a gadget or input, or classes that are not one per output
    /// channel; [`LeafError::Gadget`] for a gadget that is no requantization or whose tables do
    /// not regenerate.
    pub fn requant(
        lowering: &Lowering,
        layers: &LayerMap,
        selection: &Selection,
        node: NodeRef,
    ) -> Result<Self, LeafError> {
        let nodes: Vec<NodeId> = match node {
            NodeRef::Position(position) if position < layers.width => (0..layers.count)
                .map(|layer| layers.node(position, layer))
                .collect(),
            NodeRef::Node(id) if layers.position(id).is_none() => vec![id],
            _ => return internal("a site is a template position or a node outside the layers"),
        };
        let mut gadgets = Vec::with_capacity(nodes.len());
        for id in nodes {
            let (_, gadget) = selection.of(id).ok_or(LeafError::Internal {
                rule: "the selection gives every site its gadget",
            })?;
            let requant = Requant::new(&gadget, &selection.tables)?;
            let graph = &lowering.graph;
            let shape = graph
                .nodes
                .get(id.0 as usize)
                .and_then(|n| n.inputs.first())
                .and_then(|t| graph.tensors.get(t.0 as usize))
                .map(|t| t.shape.as_slice())
                .ok_or(LeafError::Internal {
                    rule: "a requantization reads its key",
                })?;
            if requant.classes() > 1 {
                let classes = graph
                    .nodes
                    .get(id.0 as usize)
                    .and_then(|n| n.inputs.get(1))
                    .and_then(|t| graph.tensors.get(t.0 as usize))
                    .map(|t| t.shape.as_slice());
                if classes != shape.last().map(core::slice::from_ref) {
                    return internal("a requantization's classes are one per output channel");
                }
            }
            gadgets.push(SiteGadget::requant(&requant, shape));
        }
        Ok(Self {
            node,
            layers: gadgets,
        })
    }
}

/// Every gadget node of `lowering` in node order, a template position once: the prologue's, the
/// template's positions, then the epilogue's (RFC-0045 'Sites').
pub fn gadget_nodes(lowering: &Lowering, layers: &LayerMap) -> Vec<NodeRef> {
    let mut nodes = Vec::new();
    for node in &lowering.graph.nodes {
        if node.op.class() != OpClass::Gadget {
            continue;
        }
        match layers.position(node.id) {
            Some((0, position)) => nodes.push(NodeRef::Position(position)),
            Some(_) => {}
            None => nodes.push(NodeRef::Node(node.id)),
        }
    }
    nodes
}

/// A tensor a site's lookups read in one layer, which the split sends (RFC-0045 'What is sent').
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Read<T = Term> {
    /// The tensor, as the gadget's term.
    pub term: T,
    /// The axes of its first lookup domain, ascending.
    pub axes: Vec<usize>,
    /// Its point `π_t(P[0..k])`, as indices into the leaf point `P`, its lowest variable first:
    /// the coordinates of its axes, the domain's last axis taking the lowest. Its tensor, padded
    /// per dimension, holds `2^{coords.len()}` entries.
    pub coords: Vec<u32>,
}

/// One block of an instance's column: one lookup of one site at one layer (RFC-0045 'Blocks').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Block {
    /// The site, in the layout's sites.
    pub site: usize,
    /// The layer, in the site's layers.
    pub layer: usize,
    /// The lookup, in the gadget's lookups.
    pub lookup: usize,
    /// `k`: the block holds `2^k` entries.
    pub vars: u32,
    /// `o_b`: the column entry it starts at, a multiple of `2^k`.
    pub offset: u64,
}

impl Block {
    /// `2^k`, the block's entries.
    pub fn entries(&self) -> u64 {
        1 << self.vars
    }
}

/// One LogUp instance's lookup column: the blocks of every lookup into its table (RFC-0045 'The
/// column').
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct InstanceLayout {
    /// The table.
    pub table: TableId,
    /// The blocks in column order: by site, layer and lookup, then stably sorted larger first, so
    /// that each starts at a multiple of its size.
    pub blocks: Vec<Block>,
    /// `m`, the blocks' entries.
    pub entries: u64,
    /// `h = ⌈log2 m⌉`: the column has `2^h` entries, and entries `m` to `2^h − 1` hold row 0's
    /// tuple.
    pub height: u32,
}

/// The leaf stage's layout (RFC-0045), derived from public data alone: the sites, one LogUp
/// instance per table their lookups read with its blocks, offsets and height, and the tensors
/// each site's split sends in each layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeafLayout<T = Term> {
    sites: Vec<Site<T>>,
    instances: Vec<InstanceLayout>,
    reads: Vec<Vec<Vec<Read<T>>>>,
}

impl<T: Copy + Eq> LeafLayout<T> {
    /// The layout of `sites`, in the layers `layers` (RFC-0045 'The column', 'The split'):
    ///
    /// 1. **Sites** come in node order, each once: a template position at its layer-0 node, with
    ///    one gadget per real layer, and a node outside the layers with one.
    /// 2. **Reads.** In each layer, the tensors the gadget's lookups read, in RFC-0045's order:
    ///    lookup by lookup, the key's tensors before the value's, within a form in the order of its
    ///    terms, each once, at its first read. Each is read over one set of axes, every real layer
    ///    reads the same tensors over the same axes ('One claim per input'), and a tensor's point
    ///    is `π_t(P[0..k])`.
    /// 3. **Blocks.** One per site, real layer and lookup, in that order, grouped by table. Each
    ///    table with a block is one instance, in table order (`TableId`).
    /// 4. **Columns.** Within an instance, a stable sort puts larger blocks first. Offsets are
    ///    prefix sums, so a block of `2^k` entries starts at a multiple of `2^k`. With `m` the
    ///    total, the height is `h = ⌈log2 m⌉`.
    ///
    /// # Errors
    ///
    /// [`LeafError::Internal`] (`ERR-INTERNAL`) for the first rule broken, in this order. Site by
    /// site: a node that is neither a template position nor outside the layers, a site out of
    /// node order or repeated, another number of gadgets than its real layers. Then site by site,
    /// layer by layer and lookup by lookup: a domain with an axis of length 0 or blocks of more
    /// than `2^48` entries; axes that are not ascending axes of the domain, or that name a term
    /// the lookup does not read, or twice; a gadget whose lookups read one tensor over two sets
    /// of coordinates; a layer whose ordered sources or projections differ from the first's.
    /// Then instance by instance, entries that overflow 64 bits.
    pub fn new(layers: &LayerMap, sites: Vec<Site<T>>) -> Result<Self, LeafError> {
        let mut previous: Option<u32> = None;
        for site in &sites {
            let (first, gadgets) = match site.node {
                NodeRef::Position(position) if position < layers.width => {
                    (layers.node(position, 0).0, layers.count as usize)
                }
                NodeRef::Node(id) if layers.position(id).is_none() => (id.0, 1),
                _ => {
                    return internal("a site is a template position or a node outside the layers");
                }
            };
            if previous.is_some_and(|p| first <= p) {
                return internal("the sites come in node order, each once");
            }
            previous = Some(first);
            if site.layers.len() != gadgets {
                return internal("a site has one gadget per real layer, or one outside the layers");
            }
        }
        let mut reads = Vec::with_capacity(sites.len());
        let mut tables: BTreeMap<TableId, Vec<Block>> = BTreeMap::new();
        for (s, site) in sites.iter().enumerate() {
            let mut per_layer: Vec<Vec<Read<T>>> = Vec::with_capacity(site.layers.len());
            for (l, gadget) in site.layers.iter().enumerate() {
                for lookup in &gadget.lookups {
                    if lookup.domain.contains(&0) {
                        return internal("a domain's axes each hold a position");
                    }
                    if lookup.vars() > MAX_BLOCK_VARS {
                        return internal("a block holds at most 2^48 entries");
                    }
                }
                let read = reads_of(gadget)?;
                // Every real layer reads the first's tensors over its axes, so that stage 1 stacks
                // each of them over the layers.
                if let Some(first_read) = per_layer.first()
                    && !same_reads(&read, first_read)
                {
                    return internal(
                        "a site's real layers read the same tensors over the same axes (RFC-0045 'One claim per input')",
                    );
                }
                per_layer.push(read);
                for (j, lookup) in gadget.lookups.iter().enumerate() {
                    tables.entry(lookup.lookup.table).or_default().push(Block {
                        site: s,
                        layer: l,
                        lookup: j,
                        vars: lookup.vars(),
                        offset: 0,
                    });
                }
            }
            reads.push(per_layer);
        }
        let mut instances = Vec::with_capacity(tables.len());
        for (table, mut blocks) in tables {
            // Stable: blocks of one size keep their site, layer and lookup order.
            blocks.sort_by_key(|block| core::cmp::Reverse(block.vars));
            let mut offset = 0u64;
            for block in &mut blocks {
                block.offset = offset;
                offset = offset
                    .checked_add(block.entries())
                    .ok_or(LeafError::Internal {
                        rule: "an instance's entries fit 64 bits",
                    })?;
            }
            instances.push(InstanceLayout {
                table,
                blocks,
                entries: offset,
                height: ceil_log2(offset),
            });
        }
        Ok(Self {
            sites,
            instances,
            reads,
        })
    }

    /// The sites, in node order.
    pub fn sites(&self) -> &[Site<T>] {
        &self.sites
    }

    /// The instances, one per table with a block, in table order.
    pub fn instances(&self) -> &[InstanceLayout] {
        &self.instances
    }

    /// The tensors site `site`'s lookups read in its layer `layer`, in the order the split sends
    /// them; `None` past the sites or the site's layers.
    pub fn reads(&self, site: usize, layer: usize) -> Option<&[Read<T>]> {
        self.reads.get(site)?.get(layer).map(Vec::as_slice)
    }

    /// The number of values the split sends: one per tensor, site and real layer.
    pub fn split_len(&self) -> usize {
        self.reads.iter().flatten().map(Vec::len).sum()
    }

    /// The tree schedule of the instances alone, in table order: each instance's lookup tree, then
    /// its table tree.
    pub fn trees(&self) -> Vec<TreeId> {
        TreeId::paired(self.instances.len())
    }
}

/// `σ_h(i) = (i mod 2) + 2·rev_{h−1}(⌊i / 2⌋)` (RFC-0045 'The leaves'): bit 0 kept, bits 1 to
/// `h − 1` reversed. The lookup tree's leaf `i` holds column entry `σ_h(i)`, and `σ_h` is its own
/// inverse, so column entry `q` is leaf `σ_h(q)`. For `index < 2^height`; `σ_0` and `σ_1` are the
/// identity.
pub fn leaf_index(height: u32, index: u64) -> u64 {
    match height {
        0 | 1 => index,
        _ => {
            let bits = (height - 1).min(63);
            (index & 1) | (((index >> 1).reverse_bits() >> (64 - bits)) << 1)
        }
    }
}

/// Stage 1 stacks the same ordered sources and projections from every real layer.
fn same_reads<T: Eq>(a: &[Read<T>], b: &[Read<T>]) -> bool {
    a == b
}

/// The tensors `gadget`'s lookups read, in the order they first read them, each with its axes and
/// point.
fn reads_of<T: Copy + Eq>(gadget: &SiteGadget<T>) -> Result<Vec<Read<T>>, LeafError> {
    let mut reads: Vec<Read<T>> = Vec::new();
    for site_lookup in &gadget.lookups {
        let rank = site_lookup.domain.len();
        let lookup = &site_lookup.lookup;
        let forms = || core::iter::once(&lookup.key).chain(&lookup.value);
        for (term, axes) in &site_lookup.axes {
            let ascending = axes.windows(2).all(|pair| pair[0] < pair[1]);
            if !ascending || axes.last().is_some_and(|&a| a >= rank) {
                return internal("a term's axes are ascending axes of its domain");
            }
            let read = forms().any(|form| form.terms.iter().any(|&(_, t)| t == *term));
            let listed = site_lookup.axes.iter().filter(|(t, _)| t == term).count();
            if !read || listed != 1 {
                return internal("a lookup lists the axes of terms it reads, each once");
            }
        }
        for &(_, term) in forms().flat_map(|form| &form.terms) {
            let axes: Vec<usize> = site_lookup
                .axes
                .iter()
                .find(|(t, _)| *t == term)
                .map_or_else(|| (0..rank).collect(), |(_, axes)| axes.clone());
            let point = coords(&site_lookup.domain, &axes);
            match reads.iter().find(|read| read.term == term) {
                Some(read) if read.coords == point => {}
                Some(_) => {
                    return internal(
                        "a gadget's lookups read each tensor over one set of axes (RFC-0045 'One \
                         claim per input')",
                    );
                }
                None => reads.push(Read {
                    term,
                    coords: point,
                    axes,
                }),
            }
        }
    }
    Ok(reads)
}

/// `π`: the coordinates of `P[0..k]` that a tensor read over `axes` of `domain` takes, its lowest
/// variable first. The domain's last axis takes the lowest coordinates, and each axis its own
/// block of `⌈log2 len⌉` (RFC-0011 rule 1, RFC-0010 'Multilinear extensions').
fn coords(domain: &[u32], axes: &[usize]) -> Vec<u32> {
    let mut start = vec![0u32; domain.len()];
    let mut next = 0u32;
    for a in (0..domain.len()).rev() {
        start[a] = next;
        next += axis_vars(domain[a]);
    }
    axes.iter()
        .rev()
        .flat_map(|&a| start[a]..start[a] + axis_vars(domain[a]))
        .collect()
}

/// `⌈log2 len⌉`: an axis's variables, padded to a power of two.
fn axis_vars(len: u32) -> u32 {
    len.checked_next_power_of_two()
        .map_or(u32::BITS, u32::trailing_zeros)
}

/// `⌈log2 n⌉`, 0 for `n ≤ 1`.
fn ceil_log2(n: u64) -> u32 {
    match n {
        0 | 1 => 0,
        _ => u64::BITS - (n - 1).leading_zeros(),
    }
}
