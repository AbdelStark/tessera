// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Current-model public leaf geometry, independently derived from RFC-0008/0010/0045/0050.
use super::{Error, Schedule, Tree, reserved};
use crate::claims::GadgetPolicy;
use crate::lowering::{Graph, Op, TensorId};
use crate::selection;
use crate::validation::ClaimsError;
use tv_field::Fp2;
use tv_tables::TableId;

/// Public lookup counts, tree order and Softmax forest roots of one lowered model sequence.
/// No proof values or secret data enter this derivation. This metadata discharges no claims.
#[derive(Debug)]
pub struct ModelSchedule {
    schedule: Schedule,
    lookups: Vec<(TableId, u64)>,
    forests: Vec<Root>,
}

#[derive(Debug)]
struct Root {
    tree: usize,
    n: u32,
    heads: u32,
    layers: u32,
}

fn vars(length: u32) -> Result<u32, ClaimsError> {
    Ok(u32::BITS.saturating_sub(
        length
            .checked_sub(1)
            .ok_or(ClaimsError::Internal)?
            .leading_zeros(),
    ))
}
fn own(error: Error) -> ClaimsError {
    match error {
        Error::Memory => ClaimsError::Memory,
        _ => ClaimsError::Internal,
    }
}
fn producer(graph: &Graph, tensor: TensorId) -> Result<usize, ClaimsError> {
    graph
        .nodes()
        .iter()
        .position(|node| node.outputs().contains(&tensor))
        .ok_or(ClaimsError::Internal)
}

impl ModelSchedule {
    /// Derive one instance per used table in ascending `TableId`, counting every canonical
    /// request (including repeated families and real layers) over its per-axis padded domain.
    /// Paired lookup/table trees precede the Softmax forests in template-node order.
    ///
    /// # Errors
    /// The independent selection's public-domain errors; `Memory` on failed reservation;
    /// `Internal` for invalid own graph/template geometry or sizes outside 63 variables.
    pub fn derive(graph: &Graph, policy: GadgetPolicy) -> Result<Self, ClaimsError> {
        let layers = crate::schedule::template_layers(graph).map_err(|error| match error {
            crate::schedule::ScheduleError::Memory => ClaimsError::Memory,
            _ => ClaimsError::Internal,
        })?;
        let selected = selection::derive(graph, policy)?;
        let mut totals = reserved(selected.tables.len()).map_err(own)?;
        totals.resize(selected.tables.len(), 0u64);
        for choice in &selected.choices {
            let node = graph.node(choice.node).ok_or(ClaimsError::Internal)?;
            for (request, &table) in choice.tables.iter().enumerate() {
                let domain = if matches!(node.op(), Op::SoftmaxRow { .. }) && request > 0 {
                    node.outputs().get(1)
                } else {
                    node.inputs().first()
                }
                .ok_or(ClaimsError::Internal)?;
                let dimensions = graph.tensor(*domain).ok_or(ClaimsError::Internal)?.shape;
                let bits = dimensions.dims().iter().try_fold(0u32, |sum, &d| {
                    sum.checked_add(vars(d)?).ok_or(ClaimsError::Internal)
                })?;
                let entries = 1u64.checked_shl(bits).ok_or(ClaimsError::Internal)?;
                let total = totals
                    .get_mut(usize::try_from(table.0).map_err(|_| ClaimsError::Internal)?)
                    .ok_or(ClaimsError::Internal)?;
                *total = total.checked_add(entries).ok_or(ClaimsError::Internal)?;
            }
        }
        let mut trees = reserved(
            selected
                .tables
                .len()
                .checked_mul(2)
                .and_then(|n| n.checked_add(selected.choices.len()))
                .ok_or(ClaimsError::Internal)?,
        )
        .map_err(own)?;
        let mut lookups = reserved(selected.tables.len()).map_err(own)?;
        for (index, (table, count)) in selected.tables.iter().zip(totals).enumerate() {
            if usize::try_from(table.id.0).ok() != Some(index) {
                return Err(ClaimsError::Internal);
            }
            if count == 0 {
                continue;
            }
            let height = u64::BITS.saturating_sub(count.saturating_sub(1).leading_zeros());
            let instance = lookups.len();
            trees.push(Tree::Lookup {
                instance,
                height,
                has_values: !matches!(table.kind, tv_tables::TableKind::Range { .. }),
            });
            trees.push(Tree::Table {
                instance,
                height: u32::from(table.log2_rows),
            });
            lookups.push((table.id, count));
        }
        let checkpoints = graph.checkpoints();
        let first = producer(graph, *checkpoints.first().ok_or(ClaimsError::Internal)?)?
            .checked_add(1)
            .ok_or(ClaimsError::Internal)?;
        let end = producer(graph, *checkpoints.last().ok_or(ClaimsError::Internal)?)?
            .checked_add(1)
            .ok_or(ClaimsError::Internal)?;
        let layer_count = usize::try_from(layers).map_err(|_| ClaimsError::Internal)?;
        let span = end.checked_sub(first).ok_or(ClaimsError::Internal)?;
        if span == 0 || span.checked_rem(layer_count) != Some(0) {
            return Err(ClaimsError::Internal);
        }
        let width = span.checked_div(layer_count).ok_or(ClaimsError::Internal)?;
        let first_end = first.checked_add(width).ok_or(ClaimsError::Internal)?;
        let mut forests = reserved(selected.choices.len()).map_err(own)?;
        for choice in &selected.choices {
            let node = graph.node(choice.node).ok_or(ClaimsError::Internal)?;
            if !matches!(node.op(), Op::SoftmaxRow { .. }) {
                continue;
            }
            let index = usize::try_from(choice.node.0).map_err(|_| ClaimsError::Internal)?;
            if index >= first_end && index < end {
                continue;
            }
            let source = *node.inputs().first().ok_or(ClaimsError::Internal)?;
            let shape = graph.tensor(source).ok_or(ClaimsError::Internal)?.shape;
            let &[heads, queries, keys] = shape.dims() else {
                return Err(ClaimsError::Internal);
            };
            if queries != graph.length() || keys != queries {
                return Err(ClaimsError::Internal);
            }
            let height = vars(keys)?;
            // A one-key maximum has no product-tree reduction level.
            if height == 0 {
                continue;
            }
            let layers = if index >= first && index < end {
                layers
            } else {
                1
            };
            let trees_log = vars(heads)?
                .checked_add(vars(queries)?)
                .and_then(|n| n.checked_add(vars(layers).ok()?))
                .ok_or(ClaimsError::Internal)?;
            forests.push(Root {
                tree: trees.len(),
                n: queries,
                heads,
                layers,
            });
            trees.push(Tree::Forest { height, trees_log });
        }
        Ok(Self {
            schedule: Schedule::new(&trees).map_err(own)?,
            lookups,
            forests,
        })
    }

    /// The validated mixed-tree schedule used by the production leaf verifier.
    pub fn schedule(&self) -> &Schedule {
        &self.schedule
    }

    /// `(table, entries)` in instance order, before the combined column's row-0 tail.
    /// Counts include every request's per-axis padding and only real layers (RFC-0045).
    pub fn lookups(&self) -> &[(TableId, u64)] {
        &self.lookups
    }

    /// `1 - psi` at a Softmax forest's root point, in query/head/layer coordinate order.
    /// Padded queries, heads and layers have root one. `None` means a wrong tree or point
    /// supplied by the caller, which is a verifier fault rather than a proof rejection.
    pub fn forest_root(&self, tree: usize, point: &[Fp2]) -> Option<Fp2> {
        let root = self.forests.iter().find(|root| root.tree == tree)?;
        let (queries, rest) = point.split_at_checked(usize::try_from(vars(root.n).ok()?).ok()?)?;
        let (heads, layers) =
            rest.split_at_checked(usize::try_from(vars(root.heads).ok()?).ok()?)?;
        if layers.len() != usize::try_from(vars(root.layers).ok()?).ok()? {
            return None;
        }
        Some(
            Fp2::ONE
                - crate::selectors::row_selector(
                    queries,
                    u64::from(root.n),
                    Some((heads, u64::from(root.heads))),
                    Some((layers, u64::from(root.layers))),
                )?,
        )
    }
}
