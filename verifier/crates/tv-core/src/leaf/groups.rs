// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A grouped leaf stage's layouts (RFC-0054 'Groups', 'The merge'), from public data alone: no
//! proof value enters them, and they discharge no claim.
//!
//! # Specification
//!
//! [`Grouped::new`] takes the stage's sites ([`Site`], each request with its domain's
//! [`token`](Lookup::token) axis), its Softmax forests ([`ForestShape`], in the stage's forest
//! order), the template's `L` real layers and the token axis's `t` variables ([`Extent`]), and
//! `Round1.leaf_groups`, `(g_T, g_L)`. With `n_ℓ = ⌈log2 L⌉` and `G = 2^{g_T + g_L}`, it checks, in
//! order, before anything is allocated:
//!
//! 1. `g_T + g_L ≤ 14`, else [`Error::Bits`];
//! 2. `g_T ≤ t`, else [`Error::TokenBits`];
//! 3. `g_L ≤ n_ℓ`, else [`Error::LayerBits`].
//!
//! Each is `Reject(Malformed { artifact, code: ERR-CODEC-LIMIT })` (RFC-0054 'Messages'). It then
//! derives the stage's shared metadata, and refuses:
//!
//! 4. a term that reads part of its block's token axis, or a tensor that two of a site's requests
//!    at one layer would give different group variables or coordinates: [`Error::Agreement`] at
//!    that layer's node, `Inconclusive(ERR-INTERNAL)` (RFC-0054 'Agreement'; 'The token axis': a
//!    term read over some of a block's axes "has the token axis among them or none");
//! 5. own metadata that RFC-0045 or RFC-0054 cannot lay out, [`Error::Internal`]: a site of
//!    neither one nor `L` entries; a domain of more than 63 variables; a token axis outside its
//!    domain or not of `t` variables; a read whose coordinates leave the domain or repeat; one
//!    tensor read over two sets of coordinates at one layer (RFC-0045 'One claim per input'); a
//!    table read with and without values; a forest of height 0, whose queries are not `t`
//!    variables, or whose layers are neither 1 nor `L`.
//!
//! [`Grouped::group`]`(h)` then lays out group `h` alone, `s = h mod 2^{g_T}` its token subset and
//! `o = ⌊h / 2^{g_T}⌋` its layer subset, so that a verifier holds one group's coordinates at a
//! time:
//!
//! - **Layers.** A site's `ℓ`-th entry is layer `ℓ`; a site of one entry, outside the template, is
//!   layer 0. Layer subset `o` holds the entries with `ℓ mod 2^{g_L} = o`: prologue and epilogue
//!   sites belong to subset 0, and a subset of padded layers alone holds no site.
//! - **Blocks.** The group holds each request of its entries whose domain has a token axis,
//!   restricted to the entries whose token index has low bits `s`: its variables are the block's
//!   others, in order, `k − g_T` of them. A request without one it holds whole when `s = 0`, and
//!   not otherwise ('No token axis').
//! - **Reads.** A term read over block variables `c_0, …, c_{m−1}` (its variable `i` takes `c_i`)
//!   of a block whose token axis is `low .. low + t`: its variable `i` is a group variable when
//!   `low ≤ c_i < low + g_T`, and takes bit `c_i − low` of `s`; otherwise it takes the restricted
//!   block's variable `c_i` when `c_i < low`, `c_i − g_T` when above. That is coordinate `c'` of
//!   the group's leaf point `P_h`, for the restricted variable `c'` (RFC-0045 'The leaf point'). A
//!   block without a token axis keeps its coordinates and fixes nothing.
//! - **The split** ([`Group::split`]). For each held entry in site, then layer order, one value per
//!   tensor its held requests read, in the ungrouped stage's split order (RFC-0045 'What is
//!   sent'): the stage's values at that site and layer that the group reads, in the stage's
//!   order, so that every group shares one order. RFC-0045's first-read rule is not applied again
//!   to the group's own blocks. Each value names its merged polynomial, its coordinates in `P_h`
//!   for the variables outside `S`, lowest first, and `h`'s bits on `S`.
//! - **Instances** ([`Group::instances`]). One per table the group holds a block of, in ascending
//!   `TableId`: its blocks in site, layer and request order, then larger first, stable, each at
//!   the running offset; a column of `2^{⌈log2 m⌉}` entries for the `m` its blocks hold (RFC-0045
//!   'The column'). The offsets come from counting each size's blocks, not from a sort.
//! - **Forests** ([`Group::forests`]). A forest stacked over the template's layers is shared by
//!   every group: group `h` holds its trees whose query index has low bits `s` and whose layer
//!   index has low bits `o`, `2^{trees_log − g_T − g_L}` of them. A forest outside the template is
//!   held by layer subset 0 alone, split by its queries.
//!
//! **The merge's polynomials** ([`Grouped::polynomials`], RFC-0054 'The merge'): one per value of
//! the ungrouped split, in its order, then one per forest leaf table, in forest order. A tensor's
//! group variables `S` are the variables its reads fix; a forest's, its query index's low `g_T`
//! bits, then its layer index's low `g_L` bits when it is stacked over the template, so that group
//! `h`'s bits on them are `h` itself. Group `h`'s claim `k` is its `k`-th split value, then its
//! forest shares in order.
//!
//! With `g = 0`, group 0 is RFC-0045's stage layout: the same instances, blocks, offsets, split
//! order and coordinates.

mod axes;
pub use axes::{AxisError, TokenAxes};

use core::fmt;
use core::ops::Range;
use tv_field::{Fp, Fp2};
use tv_tables::TableId;

use super::ForestShape;
use super::split::{self, Affine, Column, Lookup, Read, Site, TokenAxis};
use crate::claims::GadgetPolicy;
use crate::lowering::{Graph, NodeId};
use crate::protocol::{LeafGroups, MAX_LEAF_GROUP_BITS};
use crate::validation::ClaimsError;

/// The stage's public extent: the bounds `leaf_groups` must keep (RFC-0054 'Messages').
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Extent {
    /// `t`: the token axis's variables, `⌈log2 n⌉` for `n` tokens.
    pub token_bits: u32,
    /// `L`: the template's real layers, padded to `2^{n_ℓ}`.
    pub layers: u32,
}

/// Why a grouped stage could not be laid out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// `Reject(Malformed { ERR-CODEC-LIMIT })`: `g_T + g_L > 14`.
    Bits {
        /// `g_T`.
        token_bits: u8,
        /// `g_L`.
        layer_bits: u8,
    },
    /// `Reject(Malformed { ERR-CODEC-LIMIT })`: `g_T > t`.
    TokenBits {
        /// `g_T`.
        token_bits: u8,
        /// `t`.
        axis_bits: u32,
    },
    /// `Reject(Malformed { ERR-CODEC-LIMIT })`: `g_L > n_ℓ`.
    LayerBits {
        /// `g_L`.
        layer_bits: u8,
        /// `n_ℓ`.
        axis_bits: u32,
    },
    /// `Inconclusive(ERR-INTERNAL)`: the graph breaks a token-axis rule at `node`.
    Axis {
        /// The node refused.
        node: NodeId,
    },
    /// `Inconclusive(ERR-INTERNAL)`: the lookups of `node` would give one tensor different group
    /// variables or coordinates, or read part of a block's token axis (RFC-0054 'Agreement').
    Agreement {
        /// The site's node at that layer.
        node: NodeId,
    },
    /// The model derivation's own error: its public selection errors, memory or faults.
    Model(ClaimsError),
    /// `Inconclusive(ERR-RES-MEMORY)`.
    Memory,
    /// `Inconclusive(ERR-INTERNAL)`: inconsistent own metadata, or a group outside `0..G`.
    Internal,
}

impl Error {
    /// Whether the proof's `leaf_groups` is at fault: `Reject(Malformed)` with `ERR-CODEC-LIMIT`.
    pub const fn is_rejection(&self) -> bool {
        matches!(
            self,
            Self::Bits { .. } | Self::TokenBits { .. } | Self::LayerBits { .. }
        )
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bits {
                token_bits,
                layer_bits,
            } => write!(
                f,
                "{token_bits} token and {layer_bits} layer group bits exceed {MAX_LEAF_GROUP_BITS}"
            ),
            Self::TokenBits {
                token_bits,
                axis_bits,
            } => write!(
                f,
                "{token_bits} token group bits exceed the token axis's {axis_bits}"
            ),
            Self::LayerBits {
                layer_bits,
                axis_bits,
            } => write!(
                f,
                "{layer_bits} layer group bits exceed the layer index's {axis_bits}"
            ),
            Self::Axis { node } => write!(f, "node {} breaks a token-axis rule", node.0),
            Self::Agreement { node } => write!(
                f,
                "the lookups of node {} give one tensor two group layouts",
                node.0
            ),
            Self::Model(error) => write!(f, "{error}"),
            Self::Memory => f.write_str("memory for the group layouts could not be reserved"),
            Self::Internal => f.write_str("inconsistent verifier-owned group metadata"),
        }
    }
}
impl std::error::Error for Error {}

/// An affine form over a group's split values: `constant + Σ coefficient · split[index]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Form {
    /// The public constant.
    pub constant: Fp,
    /// Coefficients and indices into [`Group::split`], in the form's term order.
    pub terms: Vec<(Fp, usize)>,
}

/// A restricted block of a group's column (RFC-0054 'A restricted block').
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    /// The site's node at its layer.
    pub node: NodeId,
    /// The site, by its position in the stage's sites.
    pub site: usize,
    /// Its layer, the site's entry.
    pub layer: u32,
    /// The request, by its position in RFC-0009's order at that layer.
    pub lookup: usize,
    /// `k − g_T` with a token axis, `k` without: it lies at `P_h[0..variables]`.
    pub variables: u32,
    /// Its offset in the group's column, a multiple of `2^variables`.
    pub offset: u64,
    /// Its key form.
    pub key: Form,
    /// Its value form; `None` exactly for a Range table.
    pub value: Option<Form>,
}

/// A group's LogUp instance: its lookup tree's column (RFC-0045 'The column').
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Instance {
    /// The table.
    pub table: TableId,
    /// Whether the table has values.
    pub has_values: bool,
    /// `m`, the entries its blocks hold before the column's row-0 tail.
    pub entries: u64,
    /// `⌈log2 m⌉`, its lookup tree's height.
    pub height: u32,
    /// Its blocks in column order.
    pub blocks: Vec<Block>,
}

/// One of a group's split values (RFC-0054 'The group's claim').
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Value {
    /// The site's node at its layer.
    pub node: NodeId,
    /// The tensor, local to that node.
    pub column: Column,
    /// Its merged polynomial, in [`Grouped::polynomials`]: its index in the ungrouped split.
    pub polynomial: usize,
    /// For the polynomial's variables outside `S`, lowest first, their coordinates in `P_h`.
    pub coordinates: Vec<u32>,
    /// `h`'s bits on `S`: bit `j` is the `j`-th group variable's, lowest variable first.
    pub bits: u32,
}

/// A group's share of a Softmax forest (RFC-0054 'The forest').
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Share {
    /// The forest, by its position in the stage's forest order.
    pub forest: usize,
    /// Its position in the stage's schedule.
    pub tree: usize,
    /// Each product tree's height.
    pub height: u32,
    /// The share's root table variables: the forest's but its group variables.
    pub trees_log: u32,
    /// Its merged polynomial, in [`Grouped::polynomials`].
    pub polynomial: usize,
    /// `h`'s bits on the forest's group variables: query bits first, then layer bits.
    pub bits: u32,
    /// The forest's query, head and layer variables, in root-point order.
    pub axes: [u32; 3],
    /// The low query and layer variables the share fixes: `g_T`, and `g_L` or 0.
    pub fixed: [u32; 2],
}

impl Share {
    /// The forest's root point (queries, heads, layers) at this share's root point `point`: the
    /// fixed low bits inserted, as [`ModelSchedule::forest_root`](super::ModelSchedule::forest_root)
    /// reads it. `None` when `point` is not `trees_log` long.
    pub fn root_point(&self, point: &[Fp2]) -> Option<Vec<Fp2>> {
        if point.len() != usize::try_from(self.trees_log).ok()? {
            return None;
        }
        let [queries, heads, layers] = self.axes;
        let [fixed_queries, fixed_layers] = self.fixed;
        let total = queries.checked_add(heads)?.checked_add(layers)?;
        let mut out = Vec::new();
        out.try_reserve_exact(usize::try_from(total).ok()?).ok()?;
        let mut rest = point.iter().copied();
        let bit = |j: u32| {
            if self.bits.checked_shr(j)? & 1 == 1 {
                Some(Fp2::ONE)
            } else {
                Some(Fp2::ZERO)
            }
        };
        for j in 0..queries {
            out.push(if j < fixed_queries {
                bit(j)?
            } else {
                rest.next()?
            });
        }
        for _ in 0..heads {
            out.push(rest.next()?);
        }
        for j in 0..layers {
            out.push(if j < fixed_layers {
                bit(fixed_queries.checked_add(j)?)?
            } else {
                rest.next()?
            });
        }
        Some(out)
    }

    /// The share's leaf table variables, `height + trees_log`: its leaf claim's point length.
    pub fn variables(&self) -> Option<u32> {
        self.height.checked_add(self.trees_log)
    }
}

/// One group's layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Group {
    /// `h`.
    pub index: u32,
    /// `s`: its token subset, the token index's low `g_T` bits.
    pub tokens: u32,
    /// `o`: its layer subset, the layer index's low `g_L` bits.
    pub layers: u32,
    /// Its instances, one per table it holds a block of, in table order. A table it holds no
    /// block of has no tree in it and sends no root.
    pub instances: Vec<Instance>,
    /// Its split values, in the order it sends them, the stage split's order restricted to what
    /// it reads: its claims `0..split.len()`, polynomials ascending.
    pub split: Vec<Value>,
    /// Its share of each forest it holds, in forest order: its claims after the split's.
    pub forests: Vec<Share>,
}

impl Group {
    /// The group's level count, its tallest tree's height: its lookup trees and its forests'.
    pub fn levels(&self) -> u32 {
        self.instances
            .iter()
            .map(|i| i.height)
            .chain(self.forests.iter().map(|f| f.height))
            .max()
            .unwrap_or(0)
    }
}

/// Where a merged polynomial comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// A tensor of a site at one layer: a value of the ungrouped split.
    Split {
        /// The site's node at that layer.
        node: NodeId,
        /// The tensor, local to that node.
        column: Column,
    },
    /// A forest's leaf table.
    Forest {
        /// The forest, in the stage's forest order.
        forest: usize,
        /// Its position in the stage's schedule.
        tree: usize,
    },
}

/// One of the merge's polynomials (RFC-0054 'The merge').
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Polynomial {
    /// `n_i`.
    pub variables: u32,
    /// `S_i`, ascending: the variables a group fixes to its bits.
    pub group_variables: Vec<u32>,
    /// The groups that claim it, `h` ascending.
    pub readers: Range<u32>,
    /// What it is.
    pub source: Source,
}

#[derive(Debug)]
struct Placed {
    column: Column,
    variables: u32,
    /// `(variable, token bit)` of each group variable, ascending.
    fixed: Vec<(u32, u32)>,
    /// Restricted block coordinates of the other variables, lowest first.
    coordinates: Vec<u32>,
    /// Whether a request with a token axis reads it.
    tokened: bool,
}
#[derive(Debug)]
struct Terms {
    constant: Fp,
    terms: Vec<(Fp, usize)>,
}
#[derive(Debug)]
struct Request {
    table: TableId,
    variables: u32,
    token: bool,
    key: Terms,
    value: Option<Terms>,
}
/// A site at one layer.
#[derive(Debug)]
struct Part {
    site: usize,
    layer: u32,
    node: NodeId,
    start: usize,
    reads: Vec<Placed>,
    requests: Vec<Request>,
}
#[derive(Debug)]
struct Stacked {
    tree: usize,
    height: u32,
    axes: [u32; 3],
    layered: bool,
}

/// A grouped leaf stage's shared metadata, from which each group's layout is derived on demand.
#[derive(Debug)]
pub struct Grouped {
    groups: LeafGroups,
    token_bits: u32,
    layer_bits: u32,
    parts: Vec<Part>,
    forests: Vec<Stacked>,
    polynomials: Vec<Polynomial>,
    values: usize,
}

fn vector<T>(n: usize) -> Result<Vec<T>, Error> {
    let mut out = Vec::new();
    out.try_reserve_exact(n).map_err(|_| Error::Memory)?;
    Ok(out)
}
fn push<T>(values: &mut Vec<T>, value: T) -> Result<(), Error> {
    values.try_reserve(1).map_err(|_| Error::Memory)?;
    values.push(value);
    Ok(())
}
/// `⌈log2 n⌉` for `n ≥ 1`.
fn ceil_log2(n: u32) -> Result<u32, Error> {
    Ok(u32::BITS.saturating_sub(n.checked_sub(1).ok_or(Error::Internal)?.leading_zeros()))
}
fn power(bits: u32) -> Result<u32, Error> {
    1u32.checked_shl(bits).ok_or(Error::Internal)
}
fn index(n: usize) -> Result<u32, Error> {
    u32::try_from(n).map_err(|_| Error::Internal)
}

/// Check 1: the grouping's own bound, which decoding also enforces.
fn total(groups: LeafGroups) -> Result<(), Error> {
    let total = u16::from(groups.token_bits).saturating_add(u16::from(groups.layer_bits));
    if total > u16::from(MAX_LEAF_GROUP_BITS) {
        return Err(Error::Bits {
            token_bits: groups.token_bits,
            layer_bits: groups.layer_bits,
        });
    }
    Ok(())
}

/// Checks 1 to 3 of the module documentation alone, against an extent the caller derives from
/// its trusted inputs: a driver judges them as soon as `Round1` has decoded, before anything is
/// derived from the claim list (RFC-0054 'Messages'). [`Grouped::new`] runs them first too.
///
/// # Errors
/// [`Error::Bits`], [`Error::TokenBits`], then [`Error::LayerBits`]; [`Error::Internal`] for an
/// extent of no layers.
pub fn check_bounds(groups: LeafGroups, extent: Extent) -> Result<(), Error> {
    bounds(groups, extent)
}

/// Checks 1 to 3: the grouping's own bound, then the statement's.
fn bounds(groups: LeafGroups, extent: Extent) -> Result<(), Error> {
    total(groups)?;
    if u32::from(groups.token_bits) > extent.token_bits {
        return Err(Error::TokenBits {
            token_bits: groups.token_bits,
            axis_bits: extent.token_bits,
        });
    }
    let layer_bits = ceil_log2(extent.layers)?;
    if u32::from(groups.layer_bits) > layer_bits {
        return Err(Error::LayerBits {
            layer_bits: groups.layer_bits,
            axis_bits: layer_bits,
        });
    }
    Ok(())
}

/// A term's read placed in the groups' blocks (module documentation, 'Reads').
fn place(
    read: &Read,
    axis: Option<TokenAxis>,
    token_bits: u32,
    node: NodeId,
) -> Result<Placed, Error> {
    let variables = index(read.coordinates.len())?;
    let mut fixed = Vec::new();
    let mut coordinates = vector(read.coordinates.len())?;
    match axis {
        None => coordinates.extend_from_slice(&read.coordinates),
        Some(axis) => {
            let end = axis.low.checked_add(axis.bits).ok_or(Error::Internal)?;
            let cut = axis.low.checked_add(token_bits).ok_or(Error::Internal)?;
            let inside = read
                .coordinates
                .iter()
                .filter(|&&c| c >= axis.low && c < end)
                .count();
            if inside != 0 && inside != usize::try_from(axis.bits).map_err(|_| Error::Internal)? {
                return Err(Error::Agreement { node });
            }
            fixed.try_reserve_exact(inside).map_err(|_| Error::Memory)?;
            for (variable, &c) in read.coordinates.iter().enumerate() {
                if c < axis.low {
                    coordinates.push(c);
                } else if c < cut {
                    let bit = c.checked_sub(axis.low).ok_or(Error::Internal)?;
                    fixed.push((index(variable)?, bit));
                } else {
                    coordinates.push(c.checked_sub(token_bits).ok_or(Error::Internal)?);
                }
            }
        }
    }
    Ok(Placed {
        column: read.column,
        variables,
        fixed,
        coordinates,
        tokened: axis.is_some(),
    })
}

/// Offsets by counting each size's blocks: the larger first, each size in arrival order.
fn arrange(instance: &mut Instance) -> Result<(), Error> {
    let mut sizes = [0u64; 64];
    for block in &instance.blocks {
        let count = sizes
            .get_mut(usize::try_from(block.variables).map_err(|_| Error::Internal)?)
            .ok_or(Error::Internal)?;
        *count = count.checked_add(1).ok_or(Error::Internal)?;
    }
    let mut base = [0u64; 64];
    let mut total = 0u64;
    for (k, (slot, &count)) in base.iter_mut().zip(&sizes).enumerate().rev() {
        *slot = total;
        let entries = 1u64
            .checked_shl(index(k)?)
            .and_then(|size| size.checked_mul(count))
            .ok_or(Error::Internal)?;
        total = total.checked_add(entries).ok_or(Error::Internal)?;
    }
    let mut seen = [0u64; 64];
    for block in &mut instance.blocks {
        let k = usize::try_from(block.variables).map_err(|_| Error::Internal)?;
        let earlier = seen.get_mut(k).ok_or(Error::Internal)?;
        block.offset = earlier
            .checked_shl(block.variables)
            .filter(|&o| o.checked_shr(block.variables) == Some(*earlier))
            .and_then(|o| o.checked_add(*base.get(k)?))
            .ok_or(Error::Internal)?;
        *earlier = earlier.checked_add(1).ok_or(Error::Internal)?;
    }
    instance.blocks.sort_unstable_by_key(|b| b.offset);
    instance.entries = total;
    instance.height =
        u64::BITS.saturating_sub(total.checked_sub(1).ok_or(Error::Internal)?.leading_zeros());
    if instance.height > 63 {
        return Err(Error::Internal);
    }
    Ok(())
}

impl Grouped {
    /// Validate the grouping and derive the stage's shared metadata and the merge's polynomials,
    /// by the module documentation's checks, in order. Its inputs are verifier-derived public
    /// descriptors, never decoded proof metadata, except `groups`, which only checks 1 to 3 judge.
    ///
    /// # Errors
    /// [`Error::Bits`], [`Error::TokenBits`] and [`Error::LayerBits`] first; then
    /// [`Error::Agreement`] or [`Error::Internal`] at the first site that breaks a rule;
    /// `Memory` on failed reservation.
    pub fn new(
        sites: &[Site],
        forests: &[ForestShape],
        extent: Extent,
        groups: LeafGroups,
    ) -> Result<Self, Error> {
        bounds(groups, extent)?;
        let g_t = u32::from(groups.token_bits);
        let g_l = u32::from(groups.layer_bits);
        let tokens = power(g_t)?;
        let count = power(g_t.checked_add(g_l).ok_or(Error::Internal)?)?;
        let layers = usize::try_from(extent.layers).map_err(|_| Error::Internal)?;
        let entries = sites
            .iter()
            .try_fold(0usize, |sum, site| sum.checked_add(site.layers.len()))
            .ok_or(Error::Internal)?;
        let mut parts = vector(entries)?;
        let mut kinds: Vec<(TableId, bool)> = Vec::new();
        let mut polynomials = Vec::new();
        let mut values = 0usize;
        for (site_index, site) in sites.iter().enumerate() {
            let len = site.layers.len();
            if len == 0 || (len != 1 && len != layers) {
                return Err(Error::Internal);
            }
            for (layer, entry) in site.layers.iter().enumerate() {
                let layer = index(layer)?;
                let node = entry.node;
                let mut reads: Vec<(&Read, Placed)> = Vec::new();
                let mut requests = vector(entry.lookups.len())?;
                for lookup in &entry.lookups {
                    requests.push(Self::request(lookup, extent, g_t, node, &mut reads)?);
                    let kind = (lookup.table, lookup.value.is_some());
                    match kinds.iter().find(|(table, _)| *table == lookup.table) {
                        Some(&known) if known != kind => return Err(Error::Internal),
                        Some(_) => {}
                        None => push(&mut kinds, kind)?,
                    }
                }
                let first = layer
                    .checked_rem(power(g_l)?)
                    .and_then(|o| o.checked_mul(tokens))
                    .ok_or(Error::Internal)?;
                polynomials
                    .try_reserve(reads.len())
                    .map_err(|_| Error::Memory)?;
                for (_, placed) in &reads {
                    let mut group_variables = vector(placed.fixed.len())?;
                    group_variables.extend(placed.fixed.iter().map(|&(variable, _)| variable));
                    let width = if placed.tokened { tokens } else { 1 };
                    polynomials.push(Polynomial {
                        variables: placed.variables,
                        group_variables,
                        readers: first..first.checked_add(width).ok_or(Error::Internal)?,
                        source: Source::Split {
                            node,
                            column: placed.column,
                        },
                    });
                }
                let start = values;
                values = values.checked_add(reads.len()).ok_or(Error::Internal)?;
                let mut placed = vector(reads.len())?;
                placed.extend(reads.into_iter().map(|(_, placed)| placed));
                parts.push(Part {
                    site: site_index,
                    layer,
                    node,
                    start,
                    reads: placed,
                    requests,
                });
            }
        }
        let mut stacked = vector(forests.len())?;
        polynomials
            .try_reserve(forests.len())
            .map_err(|_| Error::Memory)?;
        for (position, forest) in forests.iter().enumerate() {
            let queries = ceil_log2(forest.queries)?;
            let heads = ceil_log2(forest.heads)?;
            let layered = forest.layers != 1;
            if forest.height == 0
                || queries != extent.token_bits
                || (layered && forest.layers != extent.layers)
            {
                return Err(Error::Internal);
            }
            let layer_axis = ceil_log2(forest.layers)?;
            let variables = forest
                .height
                .checked_add(queries)
                .and_then(|v| v.checked_add(heads))
                .and_then(|v| v.checked_add(layer_axis))
                .filter(|&v| v <= 63)
                .ok_or(Error::Internal)?;
            let fixed_layers = if layered { g_l } else { 0 };
            let mut group_variables = vector(
                usize::try_from(g_t.checked_add(fixed_layers).ok_or(Error::Internal)?)
                    .map_err(|_| Error::Internal)?,
            )?;
            let low = forest.height;
            group_variables.extend(low..low.checked_add(g_t).ok_or(Error::Internal)?);
            let layers_low = low
                .checked_add(queries)
                .and_then(|v| v.checked_add(heads))
                .ok_or(Error::Internal)?;
            group_variables.extend(
                layers_low
                    ..layers_low
                        .checked_add(fixed_layers)
                        .ok_or(Error::Internal)?,
            );
            polynomials.push(Polynomial {
                variables,
                group_variables,
                readers: 0..if layered { count } else { tokens },
                source: Source::Forest {
                    forest: position,
                    tree: forest.tree,
                },
            });
            stacked.push(Stacked {
                tree: forest.tree,
                height: forest.height,
                axes: [queries, heads, layer_axis],
                layered,
            });
        }
        Ok(Self {
            groups,
            token_bits: g_t,
            layer_bits: g_l,
            parts,
            forests: stacked,
            polynomials,
            values,
        })
    }

    /// One request's validated metadata, its terms' reads placed among the layer's `reads`.
    fn request<'s>(
        lookup: &'s Lookup,
        extent: Extent,
        g_t: u32,
        node: NodeId,
        reads: &mut Vec<(&'s Read, Placed)>,
    ) -> Result<Request, Error> {
        let k = lookup.variables;
        if k > 63 {
            return Err(Error::Internal);
        }
        if let Some(axis) = lookup.token
            && (axis.bits != extent.token_bits
                || axis.low.checked_add(axis.bits).is_none_or(|end| end > k))
        {
            return Err(Error::Internal);
        }
        let mut terms = |affine: &'s Affine| -> Result<Terms, Error> {
            let mut out = vector(affine.terms.len())?;
            for (coefficient, read) in &affine.terms {
                if read.coordinates.iter().enumerate().any(|(i, &c)| {
                    c >= k
                        || read
                            .coordinates
                            .get(..i)
                            .is_some_and(|earlier| earlier.contains(&c))
                }) {
                    return Err(Error::Internal);
                }
                let placed = place(read, lookup.token, g_t, node)?;
                let local = match reads.iter().position(|(r, _)| r.column == read.column) {
                    Some(local) => {
                        let (first, known) = reads.get_mut(local).ok_or(Error::Internal)?;
                        if first.coordinates != read.coordinates {
                            return Err(Error::Internal);
                        }
                        if known.fixed != placed.fixed || known.coordinates != placed.coordinates {
                            return Err(Error::Agreement { node });
                        }
                        known.tokened |= placed.tokened;
                        local
                    }
                    None => {
                        push(reads, (read, placed))?;
                        reads.len().checked_sub(1).ok_or(Error::Internal)?
                    }
                };
                out.push((*coefficient, local));
            }
            Ok(Terms {
                constant: affine.constant,
                terms: out,
            })
        };
        let key = terms(&lookup.key)?;
        let value = lookup.value.as_ref().map(&mut terms).transpose()?;
        Ok(Request {
            table: lookup.table,
            variables: k,
            token: lookup.token.is_some(),
            key,
            value,
        })
    }

    /// [`Grouped::new`] over a lowered model: its token axes, canonical sites
    /// ([`split::model_sites`]), forests ([`ModelSchedule`](super::ModelSchedule)), the
    /// template's layers and `t = ⌈log2 n⌉`. Checks 1 to 3 run before the sites are derived.
    ///
    /// # Errors
    /// The bounds' errors first; then [`Error::Axis`] for a graph the token-axis rules refuse,
    /// [`Error::Model`] for the model derivation's, and [`Grouped::new`]'s.
    pub fn derive(graph: &Graph, policy: GadgetPolicy, groups: LeafGroups) -> Result<Self, Error> {
        let extent = Self::extent(graph, groups)?;
        let axes = TokenAxes::derive(graph).map_err(|e| match e {
            AxisError::Refused { node } => Error::Axis { node },
            AxisError::Memory => Error::Memory,
            AxisError::Internal => Error::Internal,
        })?;
        let sites = split::model_sites_with(graph, policy, &axes).map_err(Error::Model)?;
        let schedule = super::ModelSchedule::derive(graph, policy).map_err(Error::Model)?;
        let mut forests = Vec::new();
        for forest in schedule.forests() {
            push(&mut forests, forest)?;
        }
        Self::new(&sites, &forests, extent, groups)
    }

    /// [`Grouped::derive`] over a verifier's runtime plan of `graph`: the stage the driver
    /// verifies, whose sites are the model's and then the greedy key's
    /// ([`split_sites`](crate::gadgets::runtime::Plan::split_sites)), and whose forests and
    /// lookups are its [`leaf_schedule`](crate::gadgets::runtime::Plan::leaf_schedule)'s. With
    /// `g = 0` its one group is the layout of [`split::Layout`] over the same sites.
    ///
    /// # Errors
    /// The bounds' errors first; then [`Error::Model`] for the runtime derivation's, and
    /// [`Grouped::new`]'s.
    pub fn derive_runtime(
        graph: &Graph,
        runtime: &crate::gadgets::runtime::Plan<'_>,
        groups: LeafGroups,
    ) -> Result<Self, Error> {
        let extent = Self::extent(graph, groups)?;
        let sites = runtime.split_sites().map_err(Error::Model)?;
        let schedule = runtime.leaf_schedule().map_err(Error::Model)?;
        let mut forests = Vec::new();
        for forest in schedule.forests() {
            push(&mut forests, forest)?;
        }
        Self::new(&sites, &forests, extent, groups)
    }

    /// The graph's extent, `t = ⌈log2 n⌉` and the template's layers, after check 1 and before
    /// checks 2 and 3, which it then runs.
    fn extent(graph: &Graph, groups: LeafGroups) -> Result<Extent, Error> {
        total(groups)?;
        let layers = crate::schedule::template_layers(graph).map_err(|e| match e {
            crate::schedule::ScheduleError::Memory => Error::Memory,
            _ => Error::Internal,
        })?;
        let extent = Extent {
            token_bits: ceil_log2(graph.length())?,
            layers,
        };
        bounds(groups, extent)?;
        Ok(extent)
    }

    /// The grouping, `(g_T, g_L)`.
    pub fn groups(&self) -> LeafGroups {
        self.groups
    }

    /// `G = 2^{g_T + g_L}`.
    pub fn count(&self) -> u32 {
        1u32.checked_shl(self.token_bits.saturating_add(self.layer_bits))
            .unwrap_or(0)
    }

    /// The ungrouped split's value count: the merge's tensor polynomials.
    pub fn values(&self) -> usize {
        self.values
    }

    /// The merge's polynomials, in order: the ungrouped split's tensors, then the forests' leaf
    /// tables.
    pub fn polynomials(&self) -> &[Polynomial] {
        &self.polynomials
    }

    /// Group `h`'s layout, by the module documentation; nothing of another group is derived.
    ///
    /// # Errors
    /// `Internal` for `h ≥ G` or inconsistent own metadata; `Memory` on failed reservation.
    pub fn group(&self, h: u32) -> Result<Group, Error> {
        if h >= self.count() {
            return Err(Error::Internal);
        }
        let tokens = power(self.token_bits)?;
        let s = h.checked_rem(tokens).ok_or(Error::Internal)?;
        let o = h.checked_div(tokens).ok_or(Error::Internal)?;
        let layer_subsets = power(self.layer_bits)?;
        let mut split: Vec<Value> = Vec::new();
        let mut instances: Vec<Instance> = Vec::new();
        for part in &self.parts {
            if part.layer.checked_rem(layer_subsets) != Some(o) {
                continue;
            }
            let held = |request: &Request| request.token || s == 0;
            if !part.requests.iter().any(held) {
                continue;
            }
            // The tensors the held requests read, sent in the stage's split order (its reads'
            // order at this site and layer), not by a first-read pass over the group's own blocks.
            let mut slots: Vec<Option<usize>> = vector(part.reads.len())?;
            slots.resize(part.reads.len(), None);
            for request in part.requests.iter().filter(|r| held(r)) {
                let value = request.value.iter().flat_map(|v| &v.terms);
                for &(_, local) in request.key.terms.iter().chain(value) {
                    // Marked as read; numbered in the stage's order below.
                    *slots.get_mut(local).ok_or(Error::Internal)? = Some(0);
                }
            }
            for (local, (slot, placed)) in slots.iter_mut().zip(&part.reads).enumerate() {
                if slot.is_none() {
                    continue;
                }
                let mut bits = 0u32;
                for (j, &(_, bit)) in placed.fixed.iter().enumerate() {
                    let on = s.checked_shr(bit).ok_or(Error::Internal)? & 1;
                    bits |= on.checked_shl(index(j)?).ok_or(Error::Internal)?;
                }
                let mut coordinates = vector(placed.coordinates.len())?;
                coordinates.extend_from_slice(&placed.coordinates);
                *slot = Some(split.len());
                push(
                    &mut split,
                    Value {
                        node: part.node,
                        column: placed.column,
                        polynomial: part.start.checked_add(local).ok_or(Error::Internal)?,
                        coordinates,
                        bits,
                    },
                )?;
            }
            for (lookup, request) in part.requests.iter().enumerate() {
                if !held(request) {
                    continue;
                }
                let form = |terms: &Terms| -> Result<Form, Error> {
                    let mut out = vector(terms.terms.len())?;
                    for &(coefficient, local) in &terms.terms {
                        let at = slots.get(local).copied().flatten().ok_or(Error::Internal)?;
                        out.push((coefficient, at));
                    }
                    Ok(Form {
                        constant: terms.constant,
                        terms: out,
                    })
                };
                let key = form(&request.key)?;
                let value = request.value.as_ref().map(form).transpose()?;
                let block = Block {
                    node: part.node,
                    site: part.site,
                    layer: part.layer,
                    lookup,
                    variables: if request.token {
                        request
                            .variables
                            .checked_sub(self.token_bits)
                            .ok_or(Error::Internal)?
                    } else {
                        request.variables
                    },
                    offset: 0,
                    key,
                    value,
                };
                let at = match instances.binary_search_by_key(&request.table, |i| i.table) {
                    Ok(at) => at,
                    Err(at) => {
                        instances.try_reserve(1).map_err(|_| Error::Memory)?;
                        instances.insert(
                            at,
                            Instance {
                                table: request.table,
                                has_values: request.value.is_some(),
                                entries: 0,
                                height: 0,
                                blocks: Vec::new(),
                            },
                        );
                        at
                    }
                };
                let instance = instances.get_mut(at).ok_or(Error::Internal)?;
                if instance.has_values != request.value.is_some() {
                    return Err(Error::Internal);
                }
                push(&mut instance.blocks, block)?;
            }
        }
        for instance in &mut instances {
            arrange(instance)?;
        }
        let mut forests = vector(self.forests.len())?;
        for (position, forest) in self.forests.iter().enumerate() {
            if !forest.layered && o != 0 {
                continue;
            }
            let fixed_layers = if forest.layered { self.layer_bits } else { 0 };
            let [queries, heads, layers] = forest.axes;
            let trees_log = queries
                .checked_add(heads)
                .and_then(|v| v.checked_add(layers))
                .and_then(|v| v.checked_sub(self.token_bits))
                .and_then(|v| v.checked_sub(fixed_layers))
                .ok_or(Error::Internal)?;
            forests.push(Share {
                forest: position,
                tree: forest.tree,
                height: forest.height,
                trees_log,
                polynomial: self.values.checked_add(position).ok_or(Error::Internal)?,
                bits: if forest.layered { h } else { s },
                axes: forest.axes,
                fixed: [self.token_bits, fixed_layers],
            });
        }
        Ok(Group {
            index: h,
            tokens: s,
            layers: o,
            instances,
            split,
            forests,
        })
    }
}
