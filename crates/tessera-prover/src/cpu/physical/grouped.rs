// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Group/tree/merge ownership beside sources and the complete CPU phase bill.

#[cfg(test)]
mod tests;

use super::*;
use crate::leaf::{
    GroupedMessageMemory, GroupedMessageShape, balanced_j_star, model_forest_shapes,
};
use tessera_field::Goldilocks as G;
use tessera_logup::{LookupForestShape, LookupGroupMemory, LookupGroupShape, TreeId};
use tessera_sumcheck::{Forest, MergeMemory, MergePolynomial};

#[derive(Clone, Copy, Debug)]
pub(super) struct GroupedWorkMemory {
    pub(super) messages: GroupedMessageMemory,
    pub(super) peak: u64,
    pub(super) planning: u64,
}

fn power(vars: usize) -> Result<u64, ProverError> {
    1u64.checked_shl(u32::try_from(vars).map_err(|_| memory())?)
        .ok_or_else(memory)
}
fn window(error: tessera_sumcheck::WindowError) -> ProverError {
    match error {
        tessera_sumcheck::WindowError::Allocation => memory(),
        tessera_sumcheck::WindowError::Shape(rule) => usage(rule),
        tessera_sumcheck::WindowError::Source => {
            internal("public grouped memory needs no source callback")
        }
    }
}
fn split_evaluation(vars: usize) -> Result<u64, ProverError> {
    // tessera-poly::evaluate_base keeps eq_low and a clearing high partial table.
    // BLOCK_VARS=10, independently pinned by the existing LogUp value evaluator bill.
    let low = vars.min(10);
    sum([array::<F>(power(low)?)?, array::<F>(power(vars - low)?)?])
}

impl GroupedWorkMemory {
    /// Bounds known canonical geometry only. Both admission guards remain until
    /// every phase and whole-route control has qualified the complete candidate.
    pub(super) fn new(input: &Inputs<'_>, allowance: u64) -> Result<Self, ProverError> {
        let grouped = input
            .grouped
            .ok_or_else(|| usage("grouped ownership requires a grouping"))?;
        // Check public descriptor requests before constructing even metadata.
        let forest_ceiling = sum([
            array::<crate::leaf::ModelForestShape>(input.leaf.sites().len() as u64)?,
            array::<Option<u32>>(3)?,
        ])?;
        if forest_ceiling > allowance {
            return Err(memory());
        }
        let forests = model_forest_shapes(input.leaf)?;
        let polynomial_count = grouped
            .polynomials()
            .len()
            .checked_add(forests.len())
            .ok_or_else(memory)?;
        let mut polynomial_metadata = growing::<MergePolynomial>(polynomial_count as u64)?;
        for shape in grouped.polynomials() {
            polynomial_metadata = sum([
                polynomial_metadata,
                array::<usize>(shape.group_vars.len() as u64)?,
            ])?;
        }
        polynomial_metadata = sum([
            polynomial_metadata,
            times(
                forests.len() as u64,
                array::<usize>(u64::from(grouped.groups().bits()))?,
            )?,
        ])?;
        let forest_metadata = array::<crate::leaf::ModelForestShape>(forests.capacity() as u64)?;
        let mut registry_bytes = array::<Vec<usize>>(grouped.count() as u64)?;
        if sum([polynomial_metadata, forest_ceiling, registry_bytes])? > allowance {
            return Err(memory());
        }
        let gt = u32::from(grouped.groups().token_bits);
        let gl = u32::from(grouped.groups().layer_bits);
        let mut polynomials = grouped.polynomials().to_vec();
        for forest in &forests {
            if forest.height == 0
                || forest.height > forest.queries
                || forest
                    .queries
                    .checked_add(gt)
                    .is_none_or(|v| v > forest.layers)
                || forest
                    .layers
                    .checked_add(gl)
                    .is_none_or(|v| v > forest.vars)
            {
                return Err(usage(
                    "canonical forest indices hold the query and layer group bits",
                ));
            }
            polynomials.push(MergePolynomial {
                vars: forest.vars as usize,
                group_vars: (forest.queries..forest.queries + gt)
                    .chain(forest.layers..forest.layers + gl)
                    .map(|v| v as usize)
                    .collect(),
            });
        }
        let vars = polynomials
            .iter()
            .map(|p| p.vars)
            .chain(input.leaf.instances().iter().map(|i| i.height as usize))
            .chain(
                input
                    .runtime
                    .selection()
                    .tables
                    .iter()
                    .map(|t| usize::from(t.log2_rows)),
            )
            .max()
            .ok_or_else(|| usage("a grouped model has source polynomials"))?;
        let atom_splits = input
            .native
            .schedule()
            .stages
            .first()
            .ok_or_else(memory)?
            .claims
            .iter()
            .filter(|c| c.split)
            .try_fold(0u64, |n, c| {
                let layers = if c.layers == compiler::schedule::Layers::Each {
                    u64::from(input.layers.count)
                } else {
                    1
                };
                sum([n, times(c.parts.len() as u64, layers)?])
            })?;
        let messages = GroupedMessageMemory::new(GroupedMessageShape {
            groups: grouped.count() as u64,
            lookups: input.leaf.instances().len() as u64,
            forests: forests.len() as u64,
            tensors: grouped.polynomials().len() as u64,
            max_vars: u32::try_from(vars).map_err(|_| memory())?,
            atom_splits,
        })?;
        let mut registry = Vec::with_capacity(grouped.count());
        let mut kernel = 0;
        let mut split = 0;
        let mut descriptors = 0;
        let mut planning = sum([polynomial_metadata, forest_metadata, registry_bytes])?;
        for h in 0..grouped.count() {
            let layout_bill = grouped.group_memory(h)?;
            let held = sum([polynomial_metadata, forest_metadata, registry_bytes])?;
            if sum([held, layout_bill.peak_bytes()])? > allowance {
                return Err(memory());
            }
            let layout = grouped.group(h)?;
            let claims = layout
                .reads()
                .len()
                .checked_add(forests.len())
                .ok_or_else(memory)?;
            let next_registry = sum([registry_bytes, array::<usize>(claims as u64)?])?;
            let trees = sum([layout.instances().len() as u64, forests.len() as u64])?;
            let shapes_ceiling = sum([
                array::<LookupGroupShape>(layout.instances().len() as u64)?,
                array::<LookupForestShape>(forests.len() as u64)?,
                array::<TreeId>(trees)?,
                array::<usize>(trees)?,
                array::<tessera_sumcheck::ResidentTreeShape>(trees)?,
            ])?;
            let public_peak = sum([
                polynomial_metadata,
                forest_metadata,
                next_registry,
                layout_bill.peak_bytes(),
                shapes_ceiling,
            ])?;
            if public_peak > allowance {
                return Err(memory());
            }
            let mut indices = Vec::with_capacity(claims);
            indices.extend(layout.reads().iter().map(|r| r.polynomial));
            indices.extend(grouped.polynomials().len()..polynomials.len());
            registry_bytes = next_registry;
            registry.push(indices);
            let mut lookup_shapes = Vec::with_capacity(layout.instances().len());
            for instance in layout.instances() {
                let spec = input
                    .runtime
                    .selection()
                    .tables
                    .iter()
                    .find(|t| t.id == instance.table)
                    .ok_or_else(|| internal("grouped memory has each canonical table"))?;
                lookup_shapes.push(LookupGroupShape {
                    rows: usize::try_from(power(instance.height as usize)?)
                        .map_err(|_| memory())?,
                    values: !matches!(spec.kind, tessera_tables::TableKind::Range { .. }),
                });
            }
            let mut forest_shapes = Vec::with_capacity(forests.len());
            for f in &forests {
                let trees_log = f
                    .vars
                    .checked_sub(gt)
                    .and_then(|n| n.checked_sub(gl))
                    .and_then(|n| n.checked_sub(f.height))
                    .ok_or_else(memory)?;
                forest_shapes.push(LookupForestShape {
                    height: f.height as usize,
                    trees_log: trees_log as usize,
                });
            }
            let mut schedule = Vec::with_capacity(lookup_shapes.len() + forest_shapes.len());
            schedule.extend((0..lookup_shapes.len()).map(TreeId::Lookup));
            schedule.extend((0..forest_shapes.len()).map(TreeId::Forest));
            let work = LookupGroupMemory::new::<F>(&lookup_shapes, &forest_shapes, &schedule)
                .map_err(crate::LeafError::from)?;
            kernel = kernel.max(work.peak_bytes());
            descriptors = descriptors.max(sum([
                growing::<tessera_logup::LookupInstance<'_, G>>(lookup_shapes.len() as u64)?,
                growing::<Forest<'_, F>>(forest_shapes.len() as u64)?,
                growing::<TreeId>(schedule.len() as u64)?,
            ])?);
            let planning_shapes = sum([
                array::<LookupGroupShape>(lookup_shapes.capacity() as u64)?,
                array::<LookupForestShape>(forest_shapes.capacity() as u64)?,
                array::<TreeId>(schedule.capacity() as u64)?,
                work.planning_bytes(),
            ])?;
            planning = planning.max(sum([
                polynomial_metadata,
                forest_metadata,
                registry_bytes,
                layout_bill.peak_bytes(),
                planning_shapes,
            ])?);
            for read in layout.reads() {
                let shape = polynomials.get(read.polynomial).ok_or_else(memory)?;
                split = split.max(split_evaluation(
                    shape
                        .vars
                        .checked_sub(shape.group_vars.len())
                        .ok_or_else(memory)?,
                )?);
            }
        }
        let cut = balanced_j_star(&polynomials, grouped.count());
        let merge = MergeMemory::for_two_pass::<F>(&polynomials, &registry, cut).map_err(window)?;
        let payload = sum([merge.table_bytes(), kernel.max(split)])?
            .max(merge.construction_peak_bytes())
            .max(sum([merge.table_bytes(), merge.proving_peak_bytes()])?);
        // Group/table/merge proof and transcript work have distinct owners.
        // GroupedMessageMemory covers the complete driver registry/point clones.
        let peak = sum([
            payload,
            descriptors,
            messages.driver_bytes(),
            growing::<&dyn tessera_logup::TableSource<G>>(input.leaf.instances().len() as u64)?,
            times(3, growing::<u8>(64)?)?,
        ])?;
        planning = planning.max(forest_ceiling);
        if planning > allowance {
            return Err(memory());
        }
        Ok(Self {
            messages,
            peak,
            planning,
        })
    }
}
