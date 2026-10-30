// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Adapt the canonical public model to the PCS allocator (RFC-0011/0041/0050).
//! The compiler follows PCS in the dependency DAG, so graph/gadget identities
//! belong here; the allocator knows only public tensor shapes (INV-PCS-1).

use crate::schedule::LayerMap;
use crate::{
    CompileError, ExecutionPlan, Gadget, Lowering, PublicModel, RuntimePlan, Selection,
    SequenceTile, Tile, tiles,
};
use core::ops::Range;
use std::collections::{BTreeMap, BTreeSet};
use tessera_field::Field;
use tessera_ir::{ClaimList, CommitmentClass, GadgetId, NodeId, NodeKind, Op, TableId, TensorId};
use tessera_pcs::{
    AdviceLayout, AdviceLayoutCounter, Axis, DenseShape, LayoutParams, Padding, PcsError,
    Placement, TensorShape, TileShape,
};
use tessera_tables::{TableKind, TableSpec};

const METADATA: &str = "advice metadata comes from the canonical lowering and gadget selection";
const SOURCE: &str = "the advice claim names a committed tensor or gadget auxiliary";
const POINT: &str = "the advice claim point has its source's natural variable count";

fn usage(rule: &'static str) -> CompileError {
    CompileError::Usage { rule }
}
fn pcs(error: PcsError) -> CompileError {
    match error {
        PcsError::Layout { rule } => CompileError::Model { rule },
        PcsError::LayoutData { rule } | PcsError::Tile { rule, .. } => usage(rule),
        PcsError::ClaimPoint { .. } => usage(POINT),
        _ => CompileError::Schedule {
            node: None,
            rule: METADATA,
        },
    }
}

/// One normalized comparison array, in RFC-0050 allocation order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CompareColumn {
    /// Left normalized bytes.
    LeftByte,
    /// Left normalization carries.
    LeftCarry,
    /// Right normalized bytes.
    RightByte,
    /// Right normalization carries.
    RightCarry,
    /// Subtraction bytes.
    Difference,
    /// Borrows, including the final zero borrow.
    Borrow,
}

/// A real auxiliary column; virtual top limbs have no identity here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AuxColumn {
    /// Requantization's low digit.
    Low,
    /// Wide requantization's middle digit.
    Middle,
    /// Wide requantization's saturation value.
    Saturation,
    /// Softmax maximum.
    Maximum,
    /// A committed low limb of slack `a` (`second=false`) or `b`.
    Slack {
        /// Which slack.
        second: bool,
        /// Low-to-high limb index.
        limb: u32,
    },
    /// One of the root input's three low 16-bit limbs.
    InputLimb(u32),
    /// A low byte of the root.
    RootByte(u32),
    /// A complete bracket comparison array; `upper` selects `(rho+1)^2 u`.
    Comparison {
        /// Lower or upper bracket.
        upper: bool,
        /// Normalization/subtraction array.
        column: CompareColumn,
        /// Low-to-high array index.
        limb: u32,
    },
}

/// A graph value, gadget role or used table's multiplicities.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AdviceSource {
    /// A committed graph tensor.
    Tensor(TensorId),
    /// A node's real auxiliary column.
    Auxiliary(NodeId, AuxColumn),
    /// One counter per actual table row, before power-of-two padding.
    Multiplicity(TableId),
    /// A request maximum, identified separately from every model node.
    GreedyMaximum(GadgetId),
}

fn committed(kind: NodeKind) -> bool {
    matches!(
        kind,
        NodeKind::Committed {
            commitment: CommitmentClass::Advice
        } | NodeKind::LookupOutput { .. }
    )
}

fn columns(
    lowering: &Lowering,
    selection: &Selection,
    node: NodeId,
) -> Result<Vec<AdviceSource>, CompileError> {
    let graph = &lowering.graph;
    let definition = graph
        .nodes
        .get(node.0 as usize)
        .ok_or_else(|| usage(METADATA))?;
    let mut values = definition
        .outputs
        .iter()
        .filter(|t| {
            graph
                .tensors
                .get(t.0 as usize)
                .is_some_and(|t| committed(t.kind))
        })
        .copied()
        .map(AdviceSource::Tensor)
        .collect::<Vec<_>>();
    let selected = selection
        .gadgets
        .binary_search_by_key(&node, |(n, _)| *n)
        .ok()
        .map(|i| selection.gadgets[i].1);
    let mut push = |column| values.push(AdviceSource::Auxiliary(node, column));
    match selected {
        Some(Gadget::RequantSplit { .. }) => push(AuxColumn::Low),
        Some(Gadget::RequantWide { split, .. }) => {
            if split > 0 {
                push(AuxColumn::Low);
            }
            push(AuxColumn::Middle);
            push(AuxColumn::Saturation);
        }
        Some(Gadget::RsqrtRow) => {
            let requests = selection.reads(node).ok_or_else(|| usage(METADATA))?;
            let width = match requests
                .first()
                .and_then(|id| selection.tables.iter().find(|t| t.id == *id))
                .map(|t| &t.kind)
            {
                Some(TableKind::Range { bits }) if (1..=32).contains(bits) => u32::from(*bits),
                _ => return Err(usage(METADATA)),
            };
            let low = (width - 1) / 8;
            // Two virtual tops distinguish lookup requests from real columns.
            // Derive the array count from the complete ordered request schema.
            let arrays = requests
                .len()
                .checked_sub(6 + low as usize)
                .ok_or_else(|| usage(METADATA))?;
            if arrays == 0 || arrays % 12 != 0 || arrays / 12 > 15 {
                return Err(usage(METADATA));
            }
            let limbs = (arrays / 12) as u32;
            for limb in 0..3 {
                push(AuxColumn::InputLimb(limb));
            }
            for byte in 0..low {
                push(AuxColumn::RootByte(byte));
            }
            for upper in [false, true] {
                for column in [
                    CompareColumn::LeftByte,
                    CompareColumn::LeftCarry,
                    CompareColumn::RightByte,
                    CompareColumn::RightCarry,
                    CompareColumn::Difference,
                    CompareColumn::Borrow,
                ] {
                    for limb in 0..limbs {
                        push(AuxColumn::Comparison {
                            upper,
                            column,
                            limb,
                        });
                    }
                }
            }
        }
        Some(Gadget::SoftmaxRow { .. }) => {
            let requests = selection.reads(node).ok_or_else(|| usage(METADATA))?;
            let slacks = requests
                .len()
                .checked_sub(2)
                .filter(|n| *n >= 2 && n % 2 == 0)
                .ok_or_else(|| usage(METADATA))?;
            let lows = (slacks / 2 - 1) as u32;
            push(AuxColumn::Maximum);
            for second in [false, true] {
                for limb in 0..lows {
                    push(AuxColumn::Slack { second, limb });
                }
            }
        }
        Some(Gadget::Argmax { .. } | Gadget::InverseCdf) => {
            return Err(usage(
                "scoring certificate advice is not part of the canonical Llama model layout",
            ));
        }
        Some(_) => {}
        None if matches!(
            definition.op,
            Op::Requant { .. }
                | Op::RsqrtRow { .. }
                | Op::SoftmaxRow { .. }
                | Op::TableMap { .. }
                | Op::OneHot { .. }
        ) =>
        {
            return Err(usage(METADATA));
        }
        None => {}
    }
    Ok(values)
}

pub(crate) fn table_rows(spec: &TableSpec) -> Result<u64, CompileError> {
    let (bits, classes) = match &spec.kind {
        TableKind::Requant { b, r_by_class, .. } | TableKind::Silu { b, r_by_class, .. } => {
            (u32::from(*b), r_by_class.len() as u64)
        }
        _ => (u32::from(spec.log2_rows), 1),
    };
    1u64.checked_shl(bits)
        .and_then(|n| n.checked_mul(classes))
        .ok_or_else(|| usage(METADATA))
}

fn params(claims: &ClaimList) -> Result<LayoutParams, CompileError> {
    let group = claims
        .arch
        .n_heads
        .checked_div(claims.arch.n_kv_heads)
        .ok_or_else(|| usage(METADATA))?;
    Ok(LayoutParams {
        log2_k: u32::from(claims.layout.advice.log2_k),
        segments: claims.layout.advice.segments,
        layers: claims.arch.n_layers,
        group,
        query_block: claims.gadget_policy.query_block,
        max_context: claims.arch.max_context,
    })
}
fn positions(lowering: &Lowering) -> Result<u32, CompileError> {
    lowering
        .graph
        .tensors
        .get(lowering.tokens.0 as usize)
        .and_then(|t| t.shape.first())
        .copied()
        .ok_or_else(|| usage(METADATA))
}
fn dense(shape: impl IntoIterator<Item = Axis>) -> TensorShape {
    TensorShape::Dense(DenseShape::new(shape.into_iter().collect(), Padding::Zero))
}
fn used_tables(selection: &Selection) -> Result<BTreeSet<TableId>, CompileError> {
    let mut used = BTreeSet::new();
    for &(node, _) in &selection.gadgets {
        used.extend(
            selection
                .reads(node)
                .ok_or_else(|| usage(METADATA))?
                .iter()
                .copied(),
        );
    }
    Ok(used)
}

fn shapes(
    lowering: &Lowering,
    selection: &Selection,
    token_axes: &[Option<u8>],
    nonzero: &[bool],
    sources: &[AdviceSource],
    attention: &tessera_pcs::Attention,
    block: Option<u32>,
) -> Result<Vec<TensorShape>, CompileError> {
    sources
        .iter()
        .enumerate()
        .map(|(column, &source)| {
            if let Some(block) = block {
                if column == 0 {
                    return Ok(TensorShape::Scores { block });
                }
                return Ok(TensorShape::Dense(
                    attention.row_shape(block, Padding::Zero),
                ));
            }
            let tensor_id = match source {
                AdviceSource::Tensor(id) => Some(id),
                AdviceSource::Auxiliary(node, _) => lowering
                    .graph
                    .nodes
                    .get(node.0 as usize)
                    .and_then(|n| n.outputs.first())
                    .copied(),
                AdviceSource::GreedyMaximum(_) => {
                    let n = u64::from(positions(lowering)?);
                    return Ok(match &lowering.graph.slots {
                        Some(slots) => TensorShape::Dense(DenseShape::with_slots(
                            vec![Axis::new(n)],
                            Padding::Zero,
                            0,
                            slots.slot_len,
                            slots.lengths.clone(),
                        )),
                        None => dense([Axis::new(n)]),
                    });
                }
                AdviceSource::Multiplicity(table) => {
                    let spec = selection
                        .tables
                        .iter()
                        .find(|t| t.id == table)
                        .ok_or_else(|| usage(METADATA))?;
                    return Ok(dense([Axis::new(table_rows(spec)?)]));
                }
            }
            .ok_or_else(|| usage(METADATA))?;
            let tensor = lowering
                .graph
                .tensors
                .get(tensor_id.0 as usize)
                .ok_or_else(|| usage(METADATA))?;
            let padding = match source {
                AdviceSource::Tensor(id)
                    if nonzero.get(id.0 as usize).copied().unwrap_or(false) =>
                {
                    Padding::Constants
                }
                _ => Padding::Zero,
            };
            let axes: Vec<Axis> = tensor
                .shape
                .iter()
                .map(|&n| Axis::new(u64::from(n)))
                .collect();
            let tok_axis = token_axes
                .get(tensor_id.0 as usize)
                .copied()
                .flatten()
                .map(usize::from);
            Ok(match (&lowering.graph.slots, tok_axis) {
                (Some(slots), Some(axis)) => TensorShape::Dense(DenseShape::with_slots(
                    axes,
                    padding,
                    axis,
                    slots.slot_len,
                    slots.lengths.clone(),
                )),
                _ => TensorShape::Dense(DenseShape::new(axes, padding)),
            })
        })
        .collect()
}

/// Checks and returns the allocated advice rows of `requests` (`(prompt_len, output_len)` in batch
/// order) under `claims`, including RFC-0053's runtime maximum and `Range` table (RFC-0071 'The
/// maximum layout', RFC-0047).
///
/// # Errors
/// `CompileError::Model` when the batch layout exceeds `2^32` rows or violates layout rules;
/// `CompileError::Usage` for malformed request lengths or dimension overflow.
pub fn check_batch_advice_rows(
    claims: &ClaimList,
    requests: &[(u32, u32)],
) -> Result<u64, CompileError> {
    if requests.is_empty() {
        return Err(usage("a batch has at least one request"));
    }
    let lengths: Vec<u32> = requests
        .iter()
        .map(|&(prompt, output)| {
            if prompt == 0 {
                return Err(usage("a sequence has 1 to max_context positions"));
            }
            prompt
                .checked_add(output)
                .ok_or_else(|| usage("a sequence has 1 to max_context positions"))
        })
        .collect::<Result<_, _>>()?;
    let public = PublicModel::of_claims(claims);
    let lowering = crate::lower_batch_for_protocol(claims.protocol_version, public, &lengths)?;
    let ranges = crate::analyze(&lowering.graph, &lowering.tables)?;
    let selection = crate::select_model(
        claims.protocol_version,
        &lowering,
        &ranges,
        &claims.gadget_policy,
    )?;
    let runtime = RuntimePlan::derive_batch(&lowering, &selection, claims, requests)?;
    allocated_runtime_rows(
        &lowering,
        runtime.selection(),
        claims,
        runtime.certificate().map(|c| (c.gadget, c.table)),
    )
}

/// Count before expanding attention shares, including all skipped rows and gaps.
pub(crate) fn allocated_rows(
    lowering: &Lowering,
    selection: &Selection,
    claims: &ClaimList,
) -> Result<u64, CompileError> {
    allocated_runtime_rows(lowering, selection, claims, None)
}

fn allocated_runtime_rows(
    lowering: &Lowering,
    selection: &Selection,
    claims: &ClaimList,
    greedy: Option<(GadgetId, TableId)>,
) -> Result<u64, CompileError> {
    let params = params(claims)?;
    let n = positions(lowering)?;
    let token_axes = crate::schedule::token_axes(lowering)?;
    let nonzero = crate::schedule::tensors_nonzero_at_padding(&lowering.graph, &lowering.tables)?;
    let (attention, mut counter) = match &lowering.graph.slots {
        Some(slots) => (
            params
                .batch_attention(slots.slot_len, &slots.lengths)
                .map_err(pcs)?,
            AdviceLayoutCounter::new_batch(params, slots.slot_len, &slots.lengths).map_err(pcs)?,
        ),
        None => (
            params.attention(n).map_err(pcs)?,
            AdviceLayoutCounter::new(params, n).map_err(pcs)?,
        ),
    };
    counter
        .append(
            &shapes(
                lowering,
                selection,
                &token_axes,
                &nonzero,
                &[AdviceSource::Tensor(lowering.tokens)],
                &attention,
                None,
            )?,
            1,
        )
        .map_err(pcs)?;
    for node in &lowering.graph.nodes {
        let sources = columns(lowering, selection, node.id)?;
        if sources.is_empty() {
            continue;
        }
        let block = matches!(node.op, Op::SoftmaxRow { .. }).then_some(0);
        let repeats = if block.is_some() {
            u64::from(claims.arch.n_kv_heads) * u64::from(attention.blocks)
        } else {
            1
        };
        counter
            .append(
                &shapes(
                    lowering,
                    selection,
                    &token_axes,
                    &nonzero,
                    &sources,
                    &attention,
                    block,
                )?,
                repeats,
            )
            .map_err(pcs)?;
    }
    let mut used = used_tables(selection)?;
    if let Some((gadget, table)) = greedy {
        counter
            .append(
                &shapes(
                    lowering,
                    selection,
                    &token_axes,
                    &nonzero,
                    &[AdviceSource::GreedyMaximum(gadget)],
                    &attention,
                    None,
                )?,
                1,
            )
            .map_err(pcs)?;
        used.insert(table);
    }
    for table in used {
        counter
            .append(
                &shapes(
                    lowering,
                    selection,
                    &token_axes,
                    &nonzero,
                    &[AdviceSource::Multiplicity(table)],
                    &attention,
                    None,
                )?,
                1,
            )
            .map_err(pcs)?;
    }
    Ok(counter.allocated_rows())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GroupKey {
    Tokens,
    Node(NodeId),
    Table(TableId),
    GreedyMaximum(GadgetId),
}
#[derive(Debug)]
struct Columns {
    sources: Vec<AdviceSource>,
    tiles: Range<usize>,
}

/// Canonical public graph sources bound to the PCS layout and whole-claim images.
#[derive(Debug)]
pub struct ModelAdviceLayout<'a> {
    lowering: &'a Lowering,
    geometry: AdviceLayout,
    tiles: Vec<SequenceTile>,
    columns: Vec<Columns>,
    bindings: BTreeMap<AdviceSource, (usize, usize)>,
}

impl<'a> ModelAdviceLayout<'a> {
    /// Build a layout for `lowering`'s sequence from its checked public selection.
    /// No weight, token or activation value is read (INV-PCS-1).
    ///
    /// # Errors
    /// `Model` when layout controls or allocated rows violate RFC-0011/0047;
    /// `Usage` for inconsistent metadata. The full row count is checked before
    /// expanding repeated attention tiles. `Schedule` reports a lowering whose
    /// layers do not form the canonical template.
    pub fn new(
        lowering: &'a Lowering,
        selection: &Selection,
        claims: &ClaimList,
    ) -> Result<Self, CompileError> {
        if claims.tables != selection.tables {
            return Err(usage(METADATA));
        }
        Self::derive(lowering, selection, claims, None)
    }

    /// Complete per-request advice, adding the maximum and every used runtime table.
    /// The model claim-list bytes and all original table ids stay fixed (draft RFC-0053).
    ///
    /// # Errors
    /// The same allocation/metadata errors as [`Self::new`], over the complete runtime layout.
    pub fn runtime(
        lowering: &'a Lowering,
        runtime: &RuntimePlan,
        claims: &ClaimList,
    ) -> Result<Self, CompileError> {
        let selection = runtime.selection();
        if !selection.tables.starts_with(&claims.tables) {
            return Err(usage(METADATA));
        }
        Self::derive(
            lowering,
            selection,
            claims,
            runtime.certificate().map(|c| (c.gadget, c.table)),
        )
    }

    fn derive(
        lowering: &'a Lowering,
        selection: &Selection,
        claims: &ClaimList,
        greedy: Option<(GadgetId, TableId)>,
    ) -> Result<Self, CompileError> {
        let expected_rows = allocated_runtime_rows(lowering, selection, claims, greedy)?;
        let params = params(claims)?;
        let n = positions(lowering)?;
        let token_axes = crate::schedule::token_axes(lowering)?;
        let nonzero =
            crate::schedule::tensors_nonzero_at_padding(&lowering.graph, &lowering.tables)?;
        let attention = match &lowering.graph.slots {
            Some(slots) => params
                .batch_attention(slots.slot_len, &slots.lengths)
                .map_err(pcs)?,
            None => params.attention(n).map_err(pcs)?,
        };
        let layers = LayerMap::new(lowering)?;
        let mut order = tiles(PublicModel::of_claims(claims), lowering, selection)?;
        if let Some((gadget, table)) = greedy {
            let first_table = order.partition_point(|t| !matches!(t, Tile::Table { .. }));
            order.insert(first_table, Tile::GreedyMaximum { gadget: gadget.0 });
            let mut used = used_tables(selection)?;
            used.insert(table);
            order.truncate(first_table + 1);
            order.extend(used.into_iter().map(|t| Tile::Table { table: t.0 }));
        }
        let plan = ExecutionPlan {
            claims: claims.clone(),
            tiles: order,
        };
        let tiles = plan.expand(lowering)?;
        let mut groups = Vec::<Columns>::new();
        let mut prior = None;
        let mut descriptions = Vec::with_capacity(tiles.len());
        let mut bindings = BTreeMap::new();
        for (index, tile) in tiles.iter().enumerate() {
            let (key, layer) = match tile.tile {
                Tile::Tokens => (GroupKey::Tokens, None),
                Tile::Node { node } | Tile::Attention { node, .. } => (
                    GroupKey::Node(NodeId(node)),
                    layers.position(NodeId(node)).map(|(layer, _)| layer),
                ),
                Tile::Table { table } => (GroupKey::Table(TableId(table)), None),
                Tile::GreedyMaximum { gadget } => (GroupKey::GreedyMaximum(GadgetId(gadget)), None),
            };
            if prior != Some(key) {
                let sources = match key {
                    GroupKey::Tokens => vec![AdviceSource::Tensor(lowering.tokens)],
                    GroupKey::Node(node) => columns(lowering, selection, node)?,
                    GroupKey::Table(table) => vec![AdviceSource::Multiplicity(table)],
                    GroupKey::GreedyMaximum(gadget) => vec![AdviceSource::GreedyMaximum(gadget)],
                };
                for (column, &source) in sources.iter().enumerate() {
                    if bindings.insert(source, (groups.len(), column)).is_some() {
                        return Err(usage(METADATA));
                    }
                }
                groups.push(Columns {
                    sources,
                    tiles: index..index + 1,
                });
                prior = Some(key);
            } else {
                groups.last_mut().ok_or_else(|| usage(METADATA))?.tiles.end = index + 1;
            }
            descriptions.push(TileShape {
                layer,
                values: shapes(
                    lowering,
                    selection,
                    &token_axes,
                    &nonzero,
                    &groups.last().ok_or_else(|| usage(METADATA))?.sources,
                    &attention,
                    tile.block.as_ref().map(|b| b.index),
                )?,
            });
        }
        let geometry = match &lowering.graph.slots {
            Some(slots) => {
                AdviceLayout::new_batch(params, slots.slot_len, &slots.lengths, &descriptions)
                    .map_err(pcs)?
            }
            None => AdviceLayout::new(params, n, &descriptions).map_err(pcs)?,
        };
        if geometry.allocated_rows() != expected_rows {
            return Err(usage(METADATA));
        }
        Ok(Self {
            lowering,
            geometry,
            tiles,
            columns: groups,
            bindings,
        })
    }

    /// Row geometry, zero rows, segments and individual PCS placements.
    pub fn layout(&self) -> &AdviceLayout {
        &self.geometry
    }
    /// Actual tile identities and share domains, in production order.
    pub fn tiles(&self) -> &[SequenceTile] {
        &self.tiles
    }
    /// Graph outputs followed by real auxiliaries, in this tile's column order.
    pub fn sources(&self, tile: usize) -> Option<&[AdviceSource]> {
        let group = self.columns.partition_point(|g| g.tiles.end <= tile);
        let group = self.columns.get(group)?;
        group
            .tiles
            .contains(&tile)
            .then_some(group.sources.as_slice())
    }
    /// Resolve a public source within one actual tile.
    pub fn location(&self, tile: usize, source: AdviceSource) -> Option<&Placement> {
        let &(group, column) = self.bindings.get(&source)?;
        if !self.columns.get(group)?.tiles.contains(&tile) {
            return None;
        }
        self.geometry.tensor(tile, column)
    }

    /// Map a natural tensor point to every weighted advice opening it owes.
    /// Attention shares fix the high coordinates of aligned head/query boxes;
    /// their equality weights sum local openings into the whole tensor claim.
    /// Multiply each image weight by the original claim's batching coefficient;
    /// the weighted sum still owes that one original claimed value.
    ///
    /// # Errors
    /// `Usage` for an absent source or wrong natural point arity.
    pub fn claim<'p, F: Field>(
        &'p self,
        source: AdviceSource,
        point: &'p [F],
    ) -> Result<impl Iterator<Item = Result<AdviceClaimImage<F>, CompileError>> + 'p, CompileError>
    {
        let &(group, column) = self.bindings.get(&source).ok_or_else(|| usage(SOURCE))?;
        let range = self.columns[group].tiles.clone();
        let first = self.tiles.get(range.start).ok_or_else(|| usage(METADATA))?;
        let arity = if matches!(first.tile, Tile::Attention { .. }) {
            self.attention_full_shape(column, source)?
                .iter()
                .map(|&n| bits(n))
                .sum::<u32>() as usize
        } else {
            self.geometry
                .tensor(range.start, column)
                .ok_or_else(|| usage(METADATA))?
                .vars() as usize
        };
        if point.len() != arity {
            return Err(usage(POINT));
        }
        Ok(range.map(move |tile| self.image(tile, column, source, point)))
    }

    fn full_shape(&self, source: AdviceSource) -> Result<&[u32], CompileError> {
        let tensor = match source {
            AdviceSource::Tensor(tensor) => tensor,
            AdviceSource::Auxiliary(node, _) => *self
                .lowering
                .graph
                .nodes
                .get(node.0 as usize)
                .and_then(|n| n.outputs.get(1))
                .ok_or_else(|| usage(METADATA))?,
            AdviceSource::Multiplicity(_) | AdviceSource::GreedyMaximum(_) => {
                return Err(usage(METADATA));
            }
        };
        self.lowering
            .graph
            .tensors
            .get(tensor.0 as usize)
            .map(|t| t.shape.as_slice())
            .ok_or_else(|| usage(METADATA))
    }

    fn attention_full_shape(
        &self,
        column: usize,
        source: AdviceSource,
    ) -> Result<Vec<u32>, CompileError> {
        let mut shape = self.full_shape(source)?.to_vec();
        if column == 0
            && let Some(last) = shape.last_mut()
        {
            *last = (*last).max(self.geometry.attention().padded);
        }
        Ok(shape)
    }

    fn image<F: Field>(
        &self,
        tile: usize,
        column: usize,
        source: AdviceSource,
        point: &[F],
    ) -> Result<AdviceClaimImage<F>, CompileError> {
        let placement = self
            .geometry
            .tensor(tile, column)
            .ok_or_else(|| usage(METADATA))?;
        let sequence = self.tiles.get(tile).ok_or_else(|| usage(METADATA))?;
        let (natural, weight) = if let Tile::Attention { kv_head, .. } = sequence.tile {
            let attention = self.geometry.attention();
            let block = sequence.block.as_ref().ok_or_else(|| usage(METADATA))?;
            let mut local = vec![attention.heads, attention.queries];
            let mut origin = vec![kv_head * attention.heads, block.queries.start];
            if column == 0 {
                local.push(attention.padded);
                origin.push(0);
            }
            let full = self.attention_full_shape(column, source)?;
            project(&full, &local, &origin, point)?
        } else {
            (point.to_vec(), F::ONE)
        };
        Ok(AdviceClaimImage {
            tile,
            source,
            weight,
            point: placement.point(&natural).map_err(pcs)?,
        })
    }
}

fn bits(length: u32) -> u32 {
    u32::BITS - length.saturating_sub(1).leading_zeros()
}

fn project<F: Field>(
    full: &[u32],
    local: &[u32],
    origin: &[u32],
    point: &[F],
) -> Result<(Vec<F>, F), CompileError> {
    if full.len() != local.len() || full.len() != origin.len() {
        return Err(usage(METADATA));
    }
    let mut projected = Vec::new();
    let mut weight = F::ONE;
    let mut coordinate = 0;
    for ((&full, &local), &origin) in full.iter().zip(local).zip(origin).rev() {
        let (full_bits, local_bits) = (bits(full), bits(local));
        if !local.is_power_of_two()
            || local_bits > full_bits
            || !origin.is_multiple_of(local)
            || u64::from(origin) + u64::from(local) > 1u64 << full_bits
        {
            return Err(usage(METADATA));
        }
        let fixed = origin >> local_bits;
        for bit in 0..full_bits {
            let &value = point.get(coordinate).ok_or_else(|| usage(POINT))?;
            if bit < local_bits {
                projected.push(value);
            } else {
                weight *= if fixed >> (bit - local_bits) & 1 == 0 {
                    F::ONE - value
                } else {
                    value
                };
            }
            coordinate += 1;
        }
    }
    if coordinate != point.len() {
        return Err(usage(POINT));
    }
    Ok((projected, weight))
}

/// One term of the weighted PCS opening sum for a public source claim.
#[derive(Debug)]
#[must_use = "every weighted image is an opening owed by the original claim"]
pub struct AdviceClaimImage<F: Field> {
    /// Index in [`ModelAdviceLayout::tiles`].
    pub tile: usize,
    /// Bound graph/auxiliary/table identity.
    pub source: AdviceSource,
    /// Equality weight of the fixed share coordinates, or one for ordinary tensors.
    pub weight: F,
    /// Point in the full advice extension, LSB-first.
    pub point: Vec<F>,
}
