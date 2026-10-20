// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public model-to-advice geometry (RFC-0011/0034/0046/0047/0050).
//!
//! No values enter this derivation (INV-PCS-1). Repeated attention tiles remain a
//! count and stride, including during the maximum-context check: a hostile model
//! cannot make the verifier expand its quadratic score domain before rejecting it.

pub mod opening;

use crate::claims::{ArchSpec, GadgetPolicy};
use crate::lowering::{Graph, Kind, NodeId, Op, Role, TensorId};
use crate::selection::{Choice, Pattern, Selection};
use crate::validation::{ClaimsError, StatementField};
use tv_field::Fp2;
use tv_tables::{TableId, TableKind};

/// RFC-0047's maximum allocated row count, including skipped rows and gaps.
pub const MAX_ROWS: u64 = 1 << 32;

fn invalid() -> ClaimsError {
    ClaimsError::StatementMismatch {
        field: StatementField::Layout,
    }
}
fn add(a: u64, b: u64) -> Result<u64, ClaimsError> {
    a.checked_add(b).ok_or_else(invalid)
}
fn mul(a: u64, b: u64) -> Result<u64, ClaimsError> {
    a.checked_mul(b).ok_or_else(invalid)
}
fn power(bits: u32) -> Result<u64, ClaimsError> {
    1u64.checked_shl(bits).ok_or_else(invalid)
}
fn log(value: u64) -> u32 {
    u64::BITS.saturating_sub(value.saturating_sub(1).leading_zeros())
}
fn aligned(row: u64, alignment: u64) -> Result<u64, ClaimsError> {
    mul(row.div_ceil(alignment), alignment)
}

/// An element domain and its natural-to-layout variable permutation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Domain {
    dims: [u64; 4],
    rank: u8,
    permutation: [u8; 128],
    variables: u32,
    rows: u64,
    zero_at: u64,
}

impl Domain {
    /// Padded axes first, stable in natural axis order (RFC-0047).
    ///
    /// # Errors
    /// `Layout` for an empty/invalid domain or unrepresentable size.
    pub fn new(shape: &[u64], log_k: u8) -> Result<Self, ClaimsError> {
        if shape.is_empty() || shape.len() > 4 || shape.iter().any(|&d| d == 0 || d > MAX_ROWS) {
            return Err(invalid());
        }
        let mut out = Self {
            dims: [0; 4],
            rank: u8::try_from(shape.len()).map_err(|_| invalid())?,
            permutation: [0; 128],
            variables: 0,
            rows: 0,
            zero_at: 0,
        };
        out.dims
            .get_mut(..shape.len())
            .ok_or_else(invalid)?
            .copy_from_slice(shape);
        for &d in shape {
            out.variables = out.variables.checked_add(log(d)).ok_or_else(invalid)?;
        }
        out.rows = power(out.variables.saturating_sub(u32::from(log_k)))?;
        out.zero_at = out.rows;
        let mut next = 0usize;
        // LSB-first: reverse of padded-then-unpadded axis order.
        for padded in [false, true] {
            let mut start = 0u32;
            for &d in shape.iter().rev() {
                let width = log(d);
                if d.is_power_of_two() != padded {
                    for bit in start..start.checked_add(width).ok_or_else(invalid)? {
                        *out.permutation.get_mut(next).ok_or_else(invalid)? =
                            u8::try_from(bit).map_err(|_| invalid())?;
                        next = next.checked_add(1).ok_or_else(invalid)?;
                    }
                }
                start = start.checked_add(width).ok_or_else(invalid)?;
            }
        }
        if let Some(&d) = shape.iter().find(|d| !d.is_power_of_two()) {
            // Only the outermost padded axis gives entirely zero suffix rows.
            let inner_bits = out.variables.checked_sub(log(d)).ok_or_else(invalid)?;
            out.zero_at = if inner_bits >= u32::from(log_k) {
                mul(d, power(inner_bits.saturating_sub(u32::from(log_k)))?)?
            } else {
                d.div_ceil(power(u32::from(log_k).saturating_sub(inner_bits))?)
            };
        }
        Ok(out)
    }

    /// Unpadded axis lengths, outermost first.
    pub fn shape(&self) -> &[u64] {
        self.dims.get(..usize::from(self.rank)).unwrap_or(&[])
    }
    /// Layout coordinate `j` takes natural LSB-first coordinate `permutation[j]`.
    pub fn permutation(&self) -> &[u8] {
        self.permutation
            .get(..usize::try_from(self.variables).unwrap_or(0))
            .unwrap_or(&[])
    }
    /// Allocated rows, including padding.
    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// First entirely zero row, relative to this domain's base.
    pub fn zero_at(&self) -> u64 {
        self.zero_at
    }
}

/// The two comparisons of RFC-0050's root bracket, in allocation order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Comparison {
    /// `rho² u <= N`.
    Lower,
    /// `N+1 <= (rho+1)² u`.
    Upper,
}
/// The arrays within one complete root comparison, in allocation order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ComparisonRole {
    /// Normalized left bytes.
    LeftDigit,
    /// Left normalization carries, including the final zero carry.
    LeftCarry,
    /// Normalized right bytes.
    RightDigit,
    /// Right normalization carries.
    RightCarry,
    /// Subtraction bytes.
    Difference,
    /// Borrow bits, including the final zero borrow.
    Borrow,
}
/// A gadget's committed auxiliary role; virtual top limbs have no such role.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Auxiliary {
    /// Requantization's low digit.
    RequantLow,
    /// Wide requantization's middle digit.
    RequantMiddle,
    /// Wide requantization's saturation value.
    RequantSaturation,
    /// Softmax maximum.
    SoftmaxMaximum,
    /// A low limb of the first reciprocal slack.
    SoftmaxA(u32),
    /// A low limb of the second reciprocal slack.
    SoftmaxB(u32),
    /// One of the three low 16-bit limbs of the full root input.
    RootInput(u32),
    /// A low byte of the root.
    RootLow(u32),
    /// A member of a root comparison array.
    RootComparison(Comparison, ComparisonRole, u32),
}

pub(crate) fn root_auxiliary(r: u32, m: u32, index: u32) -> Option<Auxiliary> {
    if index < 3 {
        return Some(Auxiliary::RootInput(index));
    }
    let index = index.checked_sub(3)?;
    if index < r {
        return Some(Auxiliary::RootLow(index));
    }
    let index = index.checked_sub(r)?;
    let array = index.checked_div(m)?;
    let comparison = if array < 6 {
        Comparison::Lower
    } else if array < 12 {
        Comparison::Upper
    } else {
        return None;
    };
    let role = match array.checked_rem(6)? {
        0 => ComparisonRole::LeftDigit,
        1 => ComparisonRole::LeftCarry,
        2 => ComparisonRole::RightDigit,
        3 => ComparisonRole::RightCarry,
        4 => ComparisonRole::Difference,
        _ => ComparisonRole::Borrow,
    };
    Some(Auxiliary::RootComparison(
        comparison,
        role,
        index.checked_rem(m)?,
    ))
}
/// The public identity of one column; attention columns also name their tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Source {
    /// A graph tensor. The advice layout locates committed tensors once per share;
    /// virtual tensor claims must first pass through their graph reductions.
    Tensor(TensorId),
    /// An auxiliary, identified by node and role.
    Auxiliary(NodeId, Auxiliary),
    /// A used lookup table's multiplicities.
    Multiplicity(TableId),
}
/// A tile's identity in production order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TileKey {
    /// Token input.
    Tokens,
    /// An ordinary node's committed outputs and auxiliaries.
    Node(NodeId),
    /// One attention share, `(node, KV head, query block)`.
    Attention(NodeId, u32, u32),
    /// A used table, in ascending id order.
    Table(TableId),
}

/// Public layout controls, copied from the validated statement.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Advice row width exponent.
    pub log_k: u8,
    /// Layer segment count.
    pub segments: u32,
    /// Public query block size.
    pub query_block: u32,
    /// Public head count.
    pub heads: u32,
    /// Public KV head count.
    pub kv_heads: u32,
    /// NonNeg limb width.
    pub limb_bits: u8,
}
impl Config {
    /// The controls relevant to advice, without cloning the public statement.
    pub fn new(arch: &ArchSpec, policy: GadgetPolicy, log_k: u8, segments: u32) -> Self {
        Self {
            log_k,
            segments,
            query_block: policy.query_block,
            heads: arch.n_heads,
            kv_heads: arch.n_kv_heads,
            limb_bits: policy.limb_bits,
        }
    }
}

#[derive(Debug)]
struct Group {
    key: TileKey,
    segment: u32,
    base: u64,
    end: u64,
    stride: u64,
    repeats: u64,
    columns: u32,
    domain: Domain,
    query: Option<Domain>,
}

/// Complete model geometry, borrowing only verifier-derived public metadata.
#[derive(Debug)]
pub struct Layout<'a> {
    graph: &'a Graph,
    selection: &'a Selection,
    config: Config,
    n: u32,
    query_block: u64,
    blocks: u64,
    key_bits_in_row: u32,
    groups: Vec<Group>,
    rows: u64,
}

impl<'a> Layout<'a> {
    /// Derive all locations and the maximum-row check without expanding attention tiles.
    /// `graph` may be lowered at `max_context`; non-attention time axes are specialized to `n`.
    ///
    /// # Errors
    /// `Layout` for invalid controls/length, a row too short for one key or more than
    /// `2^32` allocated rows; `Memory` for a failed reservation; `Internal` for
    /// inconsistent verifier-derived graph/selection metadata.
    pub fn derive(
        graph: &'a Graph,
        selection: &'a Selection,
        config: Config,
        n: u32,
    ) -> Result<Self, ClaimsError> {
        if n == 0
            || n > graph.length()
            || config.log_k > 31 // RFC-0011: a 2k-symbol codeword needs 2k <= 2^32.
            || config.segments == 0
            || !config.query_block.is_power_of_two()
            || config.kv_heads == 0
            || config.heads == 0
            || config.heads.checked_rem(config.kv_heads) != Some(0)
            || config.limb_bits == 0
        {
            return Err(invalid());
        }
        let group = u64::from(
            config
                .heads
                .checked_div(config.kv_heads)
                .ok_or_else(invalid)?,
        );
        if !group.is_power_of_two() {
            return Err(invalid());
        }
        let padded = power(log(u64::from(n)))?;
        let query_block = padded.min(u64::from(config.query_block));
        let available = u32::from(config.log_k)
            .checked_sub(log(group))
            .and_then(|x| x.checked_sub(log(query_block)))
            .ok_or_else(invalid)?;
        let key_bits_in_row = log(padded).min(available);
        let mut out = Self {
            graph,
            selection,
            config,
            n,
            query_block,
            blocks: padded.checked_div(query_block).ok_or_else(invalid)?,
            key_bits_in_row,
            groups: Vec::new(),
            rows: 0,
        };
        let capacity = graph
            .nodes()
            .len()
            .checked_add(selection.tables.len())
            .and_then(|x| x.checked_add(1))
            .ok_or(ClaimsError::Memory)?;
        out.groups
            .try_reserve_exact(capacity)
            .map_err(|_| ClaimsError::Memory)?;
        graph
            .inputs()
            .iter()
            .find(|i| i.role == Role::Tokens)
            .ok_or(ClaimsError::Internal)?;
        out.push(
            TileKey::Tokens,
            0,
            Domain::new(&[u64::from(n)], config.log_k)?,
            None,
            1,
            1,
        )?;
        for (i, node) in graph.nodes().iter().enumerate() {
            let id = NodeId(u32::try_from(i).map_err(|_| ClaimsError::Internal)?);
            let outputs = node
                .outputs()
                .iter()
                .filter(|&&t| graph.tensor(t).is_some_and(|t| committed(t.kind)))
                .count();
            let auxiliary = out.auxiliary_count(id)?;
            let columns = u32::try_from(outputs)
                .map_err(|_| ClaimsError::Internal)?
                .checked_add(auxiliary)
                .ok_or_else(invalid)?;
            if columns == 0 {
                continue;
            }
            let segment = out.node_segment(id)?;
            if matches!(node.op(), Op::SoftmaxRow { .. }) {
                let mut score = Domain::new(&[group, query_block, padded], config.log_k)?;
                let mut next = 0usize;
                let high = log(padded);
                let head_query_end = high
                    .checked_add(log(query_block))
                    .and_then(|x| x.checked_add(log(group)))
                    .ok_or_else(invalid)?;
                for bit in (0..key_bits_in_row)
                    .chain(high..head_query_end)
                    .chain(key_bits_in_row..high)
                {
                    *score
                        .permutation
                        .get_mut(next)
                        .ok_or(ClaimsError::Internal)? =
                        u8::try_from(bit).map_err(|_| ClaimsError::Internal)?;
                    next = next.checked_add(1).ok_or(ClaimsError::Internal)?;
                }
                let query = Domain::new(&[group, query_block], config.log_k)?;
                out.push(
                    TileKey::Attention(id, 0, 0),
                    segment,
                    score,
                    Some(query),
                    columns,
                    mul(u64::from(config.kv_heads), out.blocks)?,
                )?;
            } else {
                let shape = node
                    .outputs()
                    .first()
                    .and_then(|&t| graph.tensor(t))
                    .ok_or(ClaimsError::Internal)?
                    .shape;
                let mut dims = [0u64; 4];
                for (slot, &dim) in dims.iter_mut().zip(shape.dims()) {
                    *slot = u64::from(dim);
                }
                *dims.first_mut().ok_or(ClaimsError::Internal)? = u64::from(n);
                let domain = Domain::new(
                    dims.get(..shape.dims().len())
                        .ok_or(ClaimsError::Internal)?,
                    config.log_k,
                )?;
                out.push(TileKey::Node(id), segment, domain, None, columns, 1)?;
            }
        }
        let mut used = Vec::new();
        used.try_reserve_exact(selection.tables.len())
            .map_err(|_| ClaimsError::Memory)?;
        used.resize(selection.tables.len(), false);
        for choice in &selection.choices {
            for table in &choice.tables {
                *used
                    .get_mut(usize::try_from(table.0).map_err(|_| ClaimsError::Internal)?)
                    .ok_or(ClaimsError::Internal)? = true;
            }
        }
        for (table, &used) in selection.tables.iter().zip(&used) {
            if !used {
                continue;
            }
            let rows = match &table.kind {
                TableKind::Requant { b, r_by_class, .. }
                | TableKind::Silu { b, r_by_class, .. } => mul(
                    u64::try_from(r_by_class.len()).map_err(|_| ClaimsError::Internal)?,
                    power(u32::from(*b))?,
                )?,
                _ => power(u32::from(table.log2_rows))?,
            };
            out.push(
                TileKey::Table(table.id),
                config.segments.saturating_sub(1),
                Domain::new(&[rows], config.log_k)?,
                None,
                1,
                1,
            )?;
        }
        Ok(out)
    }

    fn push(
        &mut self,
        key: TileKey,
        segment: u32,
        domain: Domain,
        query: Option<Domain>,
        columns: u32,
        repeats: u64,
    ) -> Result<(), ClaimsError> {
        let base = aligned(self.rows, domain.rows)?;
        let size = if query.is_some() {
            add(domain.rows, u64::from(columns.saturating_sub(1)))?
        } else {
            mul(domain.rows, u64::from(columns))?
        };
        let stride = aligned(size, domain.rows)?;
        let end = add(add(base, mul(repeats.saturating_sub(1), stride)?)?, size)?;
        if end > MAX_ROWS {
            return Err(invalid());
        }
        self.groups.push(Group {
            key,
            segment,
            base,
            end,
            stride,
            repeats,
            columns,
            domain,
            query,
        });
        self.rows = end;
        Ok(())
    }

    fn choice(&self, id: NodeId) -> Option<&Choice> {
        self.selection
            .choices
            .binary_search_by_key(&id, |c| c.node)
            .ok()
            .and_then(|i| self.selection.choices.get(i))
    }

    fn root_counts(&self, id: NodeId) -> Result<(u32, u32), ClaimsError> {
        let output = self
            .graph
            .node(id)
            .and_then(|n| n.outputs().first())
            .ok_or(ClaimsError::Internal)?;
        let (_, high) = self
            .selection
            .ranges
            .get(usize::try_from(output.0).map_err(|_| ClaimsError::Internal)?)
            .and_then(crate::ranges::Interval::bounds)
            .ok_or(ClaimsError::Internal)?;
        let width = u128::BITS
            .saturating_sub(
                u128::try_from(high)
                    .map_err(|_| ClaimsError::Internal)?
                    .leading_zeros(),
            )
            .max(1);
        if width > 32 {
            return Err(ClaimsError::Internal);
        }
        Ok((
            width
                .saturating_sub(1)
                .checked_div(8)
                .ok_or(ClaimsError::Internal)?,
            width
                .checked_mul(2)
                .and_then(|x| x.checked_add(55))
                .ok_or(ClaimsError::Internal)?
                .div_ceil(8),
        ))
    }

    fn auxiliary_count(&self, id: NodeId) -> Result<u32, ClaimsError> {
        let Some(choice) = self.choice(id) else {
            return Ok(0);
        };
        Ok(match choice.pattern {
            Pattern::RequantSplit { .. } => 1,
            Pattern::RequantWide { split, .. } => {
                if split == 0 {
                    2
                } else {
                    3
                }
            }
            Pattern::RsqrtRow => {
                let (r, m) = self.root_counts(id)?;
                3u32.checked_add(r)
                    .and_then(|x| m.checked_mul(12).and_then(|m| x.checked_add(m)))
                    .ok_or_else(invalid)?
            }
            Pattern::SoftmaxRow => {
                let lows = self.softmax_lows(id)?;
                1u32.checked_add(lows.checked_mul(2).ok_or_else(invalid)?)
                    .ok_or_else(invalid)?
            }
            _ => 0,
        })
    }

    fn softmax_lows(&self, id: NodeId) -> Result<u32, ClaimsError> {
        let Op::SoftmaxRow { sigma, .. } = self.graph.node(id).ok_or(ClaimsError::Internal)?.op()
        else {
            return Err(ClaimsError::Internal);
        };
        Ok(u32::from(*sigma)
            .checked_add(18)
            .ok_or(ClaimsError::Internal)?
            .div_ceil(u32::from(self.config.limb_bits))
            .saturating_sub(1))
    }

    fn auxiliary(&self, id: NodeId, index: u32) -> Option<Auxiliary> {
        match self.choice(id)?.pattern {
            Pattern::RequantSplit { .. } => (index == 0).then_some(Auxiliary::RequantLow),
            Pattern::RequantWide { split, .. } => {
                let offset = u32::from(split != 0);
                if offset == 1 && index == 0 {
                    Some(Auxiliary::RequantLow)
                } else if index == offset {
                    Some(Auxiliary::RequantMiddle)
                } else if index == offset.checked_add(1)? {
                    Some(Auxiliary::RequantSaturation)
                } else {
                    None
                }
            }
            Pattern::SoftmaxRow => {
                let lows = self.softmax_lows(id).ok()?;
                if index == 0 {
                    Some(Auxiliary::SoftmaxMaximum)
                } else if index <= lows {
                    Some(Auxiliary::SoftmaxA(index.checked_sub(1)?))
                } else if index <= lows.checked_mul(2)? {
                    Some(Auxiliary::SoftmaxB(
                        index.checked_sub(lows)?.checked_sub(1)?,
                    ))
                } else {
                    None
                }
            }
            Pattern::RsqrtRow => {
                let (r, m) = self.root_counts(id).ok()?;
                root_auxiliary(r, m, index)
            }
            _ => None,
        }
    }

    fn node_segment(&self, id: NodeId) -> Result<u32, ClaimsError> {
        let checkpoints = self.graph.checkpoints();
        let layers = u32::try_from(checkpoints.len().saturating_sub(1))
            .map_err(|_| ClaimsError::Internal)?;
        if layers == 0 {
            return Err(ClaimsError::Internal);
        }
        let passed = checkpoints.partition_point(|&t| {
            self.graph
                .tensor(t)
                .and_then(|t| t.producer)
                .is_some_and(|p| p < id)
        });
        if passed == 0 {
            return Ok(0);
        }
        if passed >= checkpoints.len() {
            return Ok(self.config.segments.saturating_sub(1));
        }
        let layer = u32::try_from(passed.saturating_sub(1)).map_err(|_| ClaimsError::Internal)?;
        let short = layers
            .checked_div(self.config.segments)
            .ok_or_else(invalid)?;
        let extra = layers
            .checked_rem(self.config.segments)
            .ok_or_else(invalid)?;
        let long = short.checked_add(1).ok_or_else(invalid)?;
        let boundary = extra.checked_mul(long).ok_or_else(invalid)?;
        if layer < boundary {
            layer.checked_div(long).ok_or_else(invalid)
        } else {
            extra
                .checked_add(
                    layer
                        .checked_sub(boundary)
                        .and_then(|x| x.checked_div(short))
                        .ok_or_else(invalid)?,
                )
                .ok_or_else(invalid)
        }
    }

    /// Allocated rows, including alignment gaps and public zero rows.
    pub fn rows(&self) -> u64 {
        self.rows
    }
    /// The committed rows of the complete advice descriptor (RFC-0031), excluding public zeros.
    /// Counts compact groups; even a maximum-size layout is never expanded.
    ///
    /// # Errors
    /// `Internal` if already-derived geometry has inconsistent sizes.
    pub fn committed_rows(&self) -> Result<u64, ClaimsError> {
        self.count_rows(None)
    }

    /// Committed rows of one canonical segment, including zero for an empty segment.
    ///
    /// # Errors
    /// `Internal` for an absent segment or inconsistent derived geometry.
    pub fn committed_rows_in_segment(&self, segment: u32) -> Result<u64, ClaimsError> {
        if segment >= self.config.segments {
            return Err(ClaimsError::Internal);
        }
        self.count_rows(Some(segment))
    }

    fn count_rows(&self, segment: Option<u32>) -> Result<u64, ClaimsError> {
        let mut total = 0u64;
        for group in &self.groups {
            if segment.is_some_and(|index| group.segment != index) {
                continue;
            }
            let count = self
                .group_committed_rows(group)
                .ok_or(ClaimsError::Internal)?;
            total = total.checked_add(count).ok_or(ClaimsError::Internal)?;
        }
        if total > self.rows {
            return Err(ClaimsError::Internal);
        }
        Ok(total)
    }

    fn group_committed_rows(&self, group: &Group) -> Option<u64> {
        let Some(query) = &group.query else {
            return group
                .domain
                .zero_at
                .checked_mul(u64::from(group.columns))?
                .checked_mul(group.repeats);
        };
        let valid = u64::from(self.n)
            .checked_div(self.query_block)?
            .checked_add(u64::from(
                u64::from(self.n).checked_rem(self.query_block)? != 0,
            ))?;
        let prior = valid.checked_sub(1)?;
        let keys = 1u64.checked_shl(self.key_bits_in_row)?;
        let triangular = |n: u64| n.checked_mul(n.checked_add(1)?).map(|n| n / 2);
        // RFC-0034: sum ceil(t*Q'/keys), then the final block's actual n-key prefix.
        let prefix = if self.query_block >= keys {
            triangular(prior)?.checked_mul(self.query_block.checked_div(keys)?)?
        } else {
            let blocks = keys.checked_div(self.query_block)?;
            let whole = prior.checked_div(blocks)?;
            blocks.checked_mul(triangular(whole)?)?.checked_add(
                prior
                    .checked_rem(blocks)?
                    .checked_mul(whole.checked_add(1)?)?,
            )?
        };
        let score = prefix.checked_add(u64::from(self.n).div_ceil(keys))?;
        let auxiliary = valid
            .checked_mul(query.zero_at)?
            .checked_mul(u64::from(group.columns.checked_sub(1)?))?;
        score
            .checked_add(auxiliary)?
            .checked_mul(group.repeats.checked_div(self.blocks)?)
    }
    /// Variables in the full advice multilinear extension (RFC-0047).
    pub fn nu(&self) -> u32 {
        u32::from(self.config.log_k).saturating_add(log(self.rows))
    }
    /// Iterate tiles in production order; callers choose whether to expand repetitions.
    pub fn tiles(&self) -> impl Iterator<Item = Tile<'_, 'a>> {
        self.groups.iter().flat_map(move |group| {
            (0..group.repeats).map(move |repeat| Tile {
                layout: self,
                group,
                repeat,
            })
        })
    }

    /// Look up a tile without expanding the attention repetition domain.
    pub fn tile(&self, key: TileKey) -> Option<Tile<'_, 'a>> {
        let (group_key, repeat) = match key {
            TileKey::Attention(id, kv, qb) => {
                if kv >= self.config.kv_heads || u64::from(qb) >= self.blocks {
                    return None;
                }
                (
                    TileKey::Attention(id, 0, 0),
                    u64::from(kv)
                        .checked_mul(self.blocks)?
                        .checked_add(u64::from(qb))?,
                )
            }
            other => (other, 0),
        };
        let group = self.groups.iter().find(|g| g.key == group_key)?;
        Some(Tile {
            layout: self,
            group,
            repeat,
        })
    }

    /// Reduce a whole tensor/auxiliary claim to weighted advice openings.
    /// Natural coordinates are LSB-first, as in RFC-0010. An ordinary claim has
    /// one image. An attention claim sums every actual KV/query share, weighted
    /// by equality on the fixed high head/query coordinates (RFC-0034/0047).
    /// Padded KV shares are zero and have no opening.
    /// Each image keeps the original claim's batching coefficient, multiplied
    /// by its equality weight; their sum owes the one original claimed value.
    ///
    /// # Errors
    /// An unknown/uncommitted source or wrong point arity. Allocation failures
    /// occur lazily as `PointError::Memory` in the returned iterator.
    pub fn claim<'l, 'p>(
        &'l self,
        source: Source,
        point: &'p [Fp2],
    ) -> Result<ClaimImages<'l, 'a, 'p>, PointError> {
        let key = match source {
            Source::Multiplicity(id) => TileKey::Table(id),
            Source::Tensor(id) => {
                let tensor = self.graph.tensor(id).ok_or(PointError::Source)?;
                if let Some(node) = tensor.producer {
                    if !committed(tensor.kind) {
                        return Err(PointError::Source);
                    }
                    self.node_key(node)?
                } else if self
                    .graph
                    .inputs()
                    .iter()
                    .any(|i| i.role == Role::Tokens && i.tensor == id)
                {
                    TileKey::Tokens
                } else {
                    return Err(PointError::Source);
                }
            }
            Source::Auxiliary(node, _) => self.node_key(node)?,
        };
        let tile = self.tile(key).ok_or(PointError::Source)?;
        let column = (0..tile.columns())
            .find(|&i| tile.location(i).is_some_and(|l| l.source == source))
            .ok_or(PointError::Source)?;
        let location = tile.location(column).ok_or(PointError::Source)?;
        let arity = if tile.group.query.is_some() {
            let sequence = log(u64::from(self.n));
            log(u64::from(self.config.heads))
                .checked_add(sequence)
                .and_then(|x| x.checked_add(if column == 0 { sequence } else { 0 }))
                .ok_or(PointError::Arity)?
        } else {
            location.domain.variables
        };
        if usize::try_from(arity).ok() != Some(point.len()) {
            return Err(PointError::Arity);
        }
        Ok(ClaimImages {
            layout: self,
            group: tile.group,
            column,
            source,
            point,
            next: 0,
        })
    }

    fn node_key(&self, id: NodeId) -> Result<TileKey, PointError> {
        Ok(
            if matches!(
                self.graph.node(id).ok_or(PointError::Source)?.op(),
                Op::SoftmaxRow { .. }
            ) {
                TileKey::Attention(id, 0, 0)
            } else {
                TileKey::Node(id)
            },
        )
    }
    /// The half-open row range of a segment, including its trailing alignment gap.
    pub fn segment(&self, index: u32) -> Option<(u64, u64)> {
        if index >= self.config.segments {
            return None;
        }
        let boundary = |i| {
            self.groups
                .iter()
                .find(|g| g.segment >= i)
                .map_or(self.rows, |g| g.base)
        };
        Some((
            if index == 0 { 0 } else { boundary(index) },
            boundary(index.checked_add(1)?),
        ))
    }
    /// Whether an allocated row is public zero, including alignment and causal gaps.
    /// Returns `None` outside the allocated row domain.
    pub fn is_zero_row(&self, row: u64) -> Option<bool> {
        if row >= self.rows {
            return None;
        }
        let group = self.groups.iter().find(|g| row < g.end)?;
        if row < group.base {
            return Some(true);
        }
        let relative = row.checked_sub(group.base)?;
        let repeat = relative.checked_div(group.stride)?;
        let tile = Tile {
            layout: self,
            group,
            repeat,
        };
        let within = relative.checked_rem(group.stride)?;
        if group.query.is_some() {
            if within < group.domain.rows {
                Some(within >= tile.score_zero_at()?)
            } else if within
                < group
                    .domain
                    .rows
                    .checked_add(u64::from(group.columns.checked_sub(1)?))?
            {
                Some(tile.query_zero())
            } else {
                Some(true)
            }
        } else {
            Some(within.checked_rem(group.domain.rows)? >= group.domain.zero_at)
        }
    }
}

fn committed(kind: Kind) -> bool {
    matches!(kind, Kind::Advice | Kind::Lookup { .. })
}

/// One tile in a compact layout; creating it does not allocate.
#[derive(Clone, Copy, Debug)]
pub struct Tile<'l, 'g> {
    layout: &'l Layout<'g>,
    group: &'l Group,
    repeat: u64,
}
impl Tile<'_, '_> {
    /// Public tile identity, with the actual KV head/query block for attention.
    pub fn key(&self) -> TileKey {
        match self.group.key {
            TileKey::Attention(id, _, _) => TileKey::Attention(
                id,
                u32::try_from(self.repeat.checked_div(self.layout.blocks).unwrap_or(0))
                    .unwrap_or(0),
                u32::try_from(self.repeat.checked_rem(self.layout.blocks).unwrap_or(0))
                    .unwrap_or(0),
            ),
            other => other,
        }
    }
    /// Layer segment owning this tile.
    pub fn segment(&self) -> u32 {
        self.group.segment
    }
    /// Half-open allocated row range, without the alignment gap before it.
    pub fn rows(&self) -> (u64, u64) {
        let base = self
            .group
            .base
            .saturating_add(self.repeat.saturating_mul(self.group.stride));
        let size = if self.group.query.is_some() {
            self.group
                .domain
                .rows
                .saturating_add(u64::from(self.group.columns.saturating_sub(1)))
        } else {
            self.group
                .domain
                .rows
                .saturating_mul(u64::from(self.group.columns))
        };
        (base, base.saturating_add(size))
    }
    /// Number of graph outputs and real auxiliaries, with no virtual-top columns.
    pub fn columns(&self) -> u32 {
        self.group.columns
    }
    fn query_zero(&self) -> bool {
        self.repeat
            .checked_rem(self.layout.blocks)
            .unwrap_or(0)
            .saturating_mul(self.layout.query_block)
            >= u64::from(self.layout.n)
    }
    fn score_zero_at(&self) -> Option<u64> {
        if self.query_zero() {
            return Some(0);
        }
        let block = self.repeat.checked_rem(self.layout.blocks)?;
        let keys = block
            .checked_add(1)?
            .checked_mul(self.layout.query_block)?
            .min(u64::from(self.layout.n));
        Some(
            self.group
                .domain
                .rows
                .min(keys.div_ceil(1u64.checked_shl(self.layout.key_bits_in_row)?)),
        )
    }
    /// Locate one output/auxiliary by allocation ordinal.
    pub fn location(&self, index: u32) -> Option<Location<'_>> {
        if index >= self.group.columns {
            return None;
        }
        let source = match self.group.key {
            TileKey::Tokens => Source::Tensor(
                self.layout
                    .graph
                    .inputs()
                    .iter()
                    .find(|i| i.role == Role::Tokens)?
                    .tensor,
            ),
            TileKey::Table(id) => Source::Multiplicity(id),
            TileKey::Node(id) | TileKey::Attention(id, _, _) => {
                let mut outputs = self.layout.graph.node(id)?.outputs().iter().filter(|&&t| {
                    self.layout
                        .graph
                        .tensor(t)
                        .is_some_and(|t| committed(t.kind))
                });
                let count = u32::try_from(outputs.clone().count()).ok()?;
                if index < count {
                    Source::Tensor(*outputs.nth(usize::try_from(index).ok()?)?)
                } else {
                    Source::Auxiliary(id, self.layout.auxiliary(id, index.checked_sub(count)?)?)
                }
            }
        };
        let (domain, offset, zero_at) = match &self.group.query {
            Some(query) if index > 0 => (
                query,
                self.group
                    .domain
                    .rows
                    .checked_add(u64::from(index.checked_sub(1)?))?,
                if self.query_zero() { 0 } else { query.zero_at },
            ),
            Some(_) => (&self.group.domain, 0, self.score_zero_at()?),
            None => (
                &self.group.domain,
                u64::from(index).checked_mul(self.group.domain.rows)?,
                self.group.domain.zero_at,
            ),
        };
        Some(Location {
            source,
            domain,
            base: self.rows().0.checked_add(offset)?,
            zero_at,
        })
    }
}

/// One tensor/auxiliary's aligned location inside its tile.
#[derive(Clone, Copy, Debug)]
pub struct Location<'a> {
    /// Public source identity.
    pub source: Source,
    /// Local element domain and variable permutation.
    pub domain: &'a Domain,
    /// First allocated row.
    pub base: u64,
    /// First public-zero row relative to `base`; equal to domain rows if none.
    pub zero_at: u64,
}

/// Failure to map a public claim; no secret values are attached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointError {
    /// The source is absent or not committed.
    Source,
    /// The number of natural point coordinates is incorrect.
    Arity,
    /// The mapped point could not be reserved.
    Memory,
}

/// One term of a whole claim's weighted opening sum.
#[derive(Debug)]
#[must_use = "each weighted image is an opening owed by the original claim"]
pub struct ClaimImage {
    /// The tile holding this source/share.
    pub tile: TileKey,
    /// The same tensor/auxiliary identity as the original claim.
    pub source: Source,
    /// Multiplier on this opening; one for ordinary tensors.
    pub weight: Fp2,
    /// The point in the full advice multilinear extension, LSB-first.
    pub point: Vec<Fp2>,
}

/// Lazy images: even a maximum-size attention domain is never allocated at once.
#[must_use = "all weighted images must be included in the original claim's opening sum"]
pub struct ClaimImages<'l, 'g, 'p> {
    layout: &'l Layout<'g>,
    group: &'l Group,
    column: u32,
    source: Source,
    point: &'p [Fp2],
    next: u64,
}

impl ClaimImages<'_, '_, '_> {
    /// The mapped source's row coefficient, summed over its images without expanding them.
    /// Returns zero on public-zero rows and outside this source. The iterator's current
    /// position does not affect the result; every attention tile share is included (RFC-0037).
    ///
    /// # Errors
    /// `Source` or `Arity` for inconsistent already-derived metadata. This method allocates nothing.
    pub fn row_coefficient(&self, row: u64) -> Result<Fp2, PointError> {
        let Some(offset) = row.checked_sub(self.group.base) else {
            return Ok(Fp2::ZERO);
        };
        let repeat = offset
            .checked_div(self.group.stride)
            .ok_or(PointError::Source)?;
        if repeat >= self.group.repeats {
            return Ok(Fp2::ZERO);
        }
        let tile = Tile {
            layout: self.layout,
            group: self.group,
            repeat,
        };
        let location = tile.location(self.column).ok_or(PointError::Source)?;
        let Some(local) = row.checked_sub(location.base) else {
            return Ok(Fp2::ZERO);
        };
        if local >= location.zero_at {
            return Ok(Fp2::ZERO);
        }
        let sequence = log(u64::from(self.layout.n));
        let queries = log(self.layout.query_block);
        let keys = if self.column == 0 { sequence } else { 0 };
        let attention = self.group.query.is_some();
        let mut coefficient = Fp2::ONE;
        let factor = |coordinate: usize, bit: u64| -> Result<Fp2, PointError> {
            let value = *self.point.get(coordinate).ok_or(PointError::Arity)?;
            Ok(if bit & 1 == 0 {
                Fp2::ONE - value
            } else {
                value
            })
        };
        // Invert the local layout permutation at the row's bits. Attention's
        // local head coordinates follow Q' query bits, rather than N query bits.
        for (position, &natural) in location
            .domain
            .permutation()
            .iter()
            .enumerate()
            .skip(usize::from(self.layout.config.log_k))
        {
            let mut natural = u32::from(natural);
            if attention && natural >= keys.checked_add(queries).ok_or(PointError::Arity)? {
                natural = natural
                    .checked_add(sequence.checked_sub(queries).ok_or(PointError::Arity)?)
                    .ok_or(PointError::Arity)?;
            }
            let shift = u32::try_from(position)
                .ok()
                .and_then(|p| p.checked_sub(u32::from(self.layout.config.log_k)))
                .ok_or(PointError::Arity)?;
            coefficient *= factor(
                usize::try_from(natural).map_err(|_| PointError::Arity)?,
                local.checked_shr(shift).ok_or(PointError::Arity)?,
            )?;
        }
        if attention {
            let heads = log(u64::from(self.layout.config.heads));
            let per_kv = self
                .layout
                .config
                .heads
                .checked_div(self.layout.config.kv_heads)
                .ok_or(PointError::Source)?;
            let low_heads = log(u64::from(per_kv));
            for (first, width, fixed) in [
                (
                    keys.checked_add(queries).ok_or(PointError::Arity)?,
                    sequence.checked_sub(queries).ok_or(PointError::Arity)?,
                    repeat
                        .checked_rem(self.layout.blocks)
                        .ok_or(PointError::Source)?,
                ),
                (
                    keys.checked_add(sequence)
                        .and_then(|p| p.checked_add(low_heads))
                        .ok_or(PointError::Arity)?,
                    heads.checked_sub(low_heads).ok_or(PointError::Arity)?,
                    repeat
                        .checked_div(self.layout.blocks)
                        .ok_or(PointError::Source)?,
                ),
            ] {
                for bit in 0..width {
                    let coordinate = first
                        .checked_add(bit)
                        .and_then(|p| usize::try_from(p).ok())
                        .ok_or(PointError::Arity)?;
                    coefficient *=
                        factor(coordinate, fixed.checked_shr(bit).ok_or(PointError::Arity)?)?;
                }
            }
        }
        Ok(coefficient)
    }

    fn image(&self, repeat: u64) -> Result<ClaimImage, PointError> {
        let tile = Tile {
            layout: self.layout,
            group: self.group,
            repeat,
        };
        let location = tile.location(self.column).ok_or(PointError::Source)?;
        let mut natural = [Fp2::ZERO; 128];
        let mut count = 0usize;
        let mut weight = Fp2::ONE;
        if self.group.query.is_some() {
            let sequence = log(u64::from(self.layout.n));
            let queries = log(self.layout.query_block);
            let heads = log(u64::from(self.layout.config.heads));
            let group = self
                .layout
                .config
                .heads
                .checked_div(self.layout.config.kv_heads)
                .ok_or(PointError::Source)?;
            let low_heads = log(u64::from(group));
            let query_start = if self.column == 0 { sequence } else { 0 };
            let head_start = query_start.checked_add(sequence).ok_or(PointError::Arity)?;
            let fixed_query = repeat
                .checked_rem(self.layout.blocks)
                .ok_or(PointError::Source)?;
            let fixed_head = repeat
                .checked_div(self.layout.blocks)
                .ok_or(PointError::Source)?;
            let key_coords = if self.column == 0 { sequence } else { 0 };
            for index in (0..key_coords)
                .chain(query_start..query_start.checked_add(queries).ok_or(PointError::Arity)?)
                .chain(head_start..head_start.checked_add(low_heads).ok_or(PointError::Arity)?)
            {
                *natural.get_mut(count).ok_or(PointError::Arity)? = *self
                    .point
                    .get(usize::try_from(index).map_err(|_| PointError::Arity)?)
                    .ok_or(PointError::Arity)?;
                count = count.checked_add(1).ok_or(PointError::Arity)?;
            }
            for (start, width, fixed) in [
                (
                    query_start.checked_add(queries).ok_or(PointError::Arity)?,
                    sequence.checked_sub(queries).ok_or(PointError::Arity)?,
                    fixed_query,
                ),
                (
                    head_start.checked_add(low_heads).ok_or(PointError::Arity)?,
                    heads.checked_sub(low_heads).ok_or(PointError::Arity)?,
                    fixed_head,
                ),
            ] {
                for bit in 0..width {
                    let index = start
                        .checked_add(bit)
                        .and_then(|x| usize::try_from(x).ok())
                        .ok_or(PointError::Arity)?;
                    let coordinate = *self.point.get(index).ok_or(PointError::Arity)?;
                    weight *= if fixed.checked_shr(bit).ok_or(PointError::Arity)? & 1 == 0 {
                        Fp2::ONE - coordinate
                    } else {
                        coordinate
                    };
                }
            }
        } else {
            natural
                .get_mut(..self.point.len())
                .ok_or(PointError::Arity)?
                .copy_from_slice(self.point);
            count = self.point.len();
        }
        if count != usize::try_from(location.domain.variables).map_err(|_| PointError::Arity)? {
            return Err(PointError::Arity);
        }
        let mut point = Vec::new();
        point
            .try_reserve_exact(usize::try_from(self.layout.nu()).map_err(|_| PointError::Arity)?)
            .map_err(|_| PointError::Memory)?;
        for &index in location.domain.permutation() {
            point.push(*natural.get(usize::from(index)).ok_or(PointError::Arity)?);
        }
        // Below-row padding is fixed at zero; above the local domain, alignment
        // fixes the base row. Local row variables were already permuted above.
        for bit in location.domain.variables..self.layout.nu() {
            let fixed = bit
                .checked_sub(u32::from(self.layout.config.log_k))
                .and_then(|shift| location.base.checked_shr(shift))
                .unwrap_or(0)
                & 1;
            point.push(if fixed == 0 { Fp2::ZERO } else { Fp2::ONE });
        }
        Ok(ClaimImage {
            tile: tile.key(),
            source: self.source,
            weight,
            point,
        })
    }
}

impl Iterator for ClaimImages<'_, '_, '_> {
    type Item = Result<ClaimImage, PointError>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.group.repeats {
            return None;
        }
        let repeat = self.next;
        self.next = self.next.saturating_add(1);
        Some(self.image(repeat))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocator_accepts_exactly_two_to_32_rows_and_counts_padding_and_auxiliaries() {
        let vectors = tv_testkit::read_vectors("claims/claim-lists.json");
        let case = &vectors.get("cases").unwrap().as_array().unwrap()[0];
        let claims = crate::claims::ClaimList::open(
            &tv_testkit::hex(case.get("hex").unwrap().as_str().unwrap()),
            &tv_codec::DecodeLimits::MESSAGE,
        )
        .unwrap();
        let public = crate::validation::check_public(&claims).unwrap();
        let graph = crate::lowering::lower(&public, 1).unwrap();
        let selection = crate::selection::derive(&graph, claims.gadget_policy).unwrap();
        let config = Config::new(&claims.arch, claims.gadget_policy, 10, 1);
        let mut layout = Layout::derive(&graph, &selection, config, 1).unwrap();
        layout.groups.clear();
        layout.rows = 0;
        let domain = Domain::new(&[1 << 21, 3 << 19], 10).unwrap();
        assert_eq!(domain.rows(), MAX_ROWS);
        assert!(domain.zero_at() < MAX_ROWS);
        layout
            .push(TileKey::Tokens, 0, domain.clone(), None, 1, 1)
            .unwrap();
        assert_eq!(layout.rows(), MAX_ROWS);
        assert_eq!(layout.nu(), 42);
        assert_eq!(layout.committed_rows().unwrap(), 3 << 30);
        assert_eq!(layout.committed_rows_in_segment(0).unwrap(), 3 << 30);
        assert_eq!(
            layout.push(
                TileKey::Tokens,
                0,
                Domain::new(&[1], 10).unwrap(),
                None,
                1,
                1
            ),
            Err(invalid())
        );
        layout.groups.clear();
        layout.rows = 0;
        // The score alone fits; one real scalar auxiliary crosses the bound.
        assert_eq!(
            layout.push(
                TileKey::Tokens,
                0,
                domain,
                Some(Domain::new(&[1], 10).unwrap()),
                2,
                1
            ),
            Err(invalid())
        );
        assert_eq!(layout.rows(), 0);
    }
}
