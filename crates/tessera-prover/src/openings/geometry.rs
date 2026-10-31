// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public original-opening geometry, with opaque coordinate origins rather than sampled fields.

use std::collections::HashSet;

use super::*;
use crate::leaf::{GroupedLayout, token, tokens};
use tessera_compiler::schedule::{Site, StageSchedule};
use tessera_tables::TableSpec;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Origin {
    Zero,
    One,
    Half,
    Stage(Coord),
    Leaf(u32),
    // Distance from the end of the one global merge point (RFC-0054).
    Merge(usize),
    // Equal-height tables share their global batch point; other heights may overcount.
    Table { height: u8, index: u8 },
}

/// Public upper bounds for the current Protocol One CPU producer's original openings,
/// before any challenge is sampled.
/// Symbolically distinct column points may collide in the field, so [`Self::groups`] is an
/// upper bound. No coefficient or zero image removes an original obligation (RFC-0011).
/// This describes opening geometry only, not a complete CPU memory or soundness bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpeningGeometry {
    groups: usize,
    claims: u64,
    original_pieces: u64,
    images: u64,
    advice_coordinates: u64,
    weight_claims: [u64; 10],
    weight_coordinates: [u64; 10],
}
impl OpeningGeometry {
    /// Derive every original target through the same traversal as [`ModelOpenings::derive`].
    /// `tables` is the complete runtime selection, including its greedy-range table.
    /// All inputs must be the canonical public metadata for one proof unit. No witness,
    /// completed prover pool or fabricated proof is used. Numeric coefficients are irrelevant
    /// to capacity: every original source and image is counted, including zero-weight ones.
    ///
    /// # Errors
    /// `Internal` for inconsistent public geometry, `Memory` for checked reservation/count overflow.
    #[allow(clippy::too_many_arguments)]
    pub fn derive(
        lowering: &Lowering,
        tables: &[TableSpec],
        leaf_layout: &LeafLayout<ModelColumn>,
        grouped: Option<&GroupedLayout>,
        native: &ModelNativePlan,
        advice: &ModelAdviceLayout<'_>,
        weights: &WeightLayout,
    ) -> Result<Self, StageError> {
        let leaf = PublicLeaf {
            layout: leaf_layout,
            tables,
            grouped,
        };
        let stages = PublicStage::new(lowering, native.schedule(), grouped)?;
        let mut sink = GeometryTargets {
            layout: advice,
            groups: HashSet::new(),
            counts: Self {
                groups: 0,
                claims: 0,
                original_pieces: 0,
                images: 0,
                advice_coordinates: 0,
                weight_claims: [0; 10],
                weight_coordinates: [0; 10],
            },
        };
        walk_model_targets(
            lowering,
            leaf_layout,
            &leaf,
            native,
            &stages,
            weights,
            &mut sink,
        )?;
        sink.counts.groups = sink.groups.len();
        Ok(sink.counts)
    }
    /// Conservative requested heap peak of this geometry's canonical derivation
    /// (RFC-0018), without allocating or reading witness values. Inputs must be
    /// the same public metadata used by `derive`. Borrowed inputs, inline storage,
    /// allocator/runtime overhead and other proof phases remain separate.
    ///
    /// # Errors
    /// Inconsistent geometry or checked storage overflow.
    pub fn construction_peak_bytes(
        &self,
        lowering: &Lowering,
        leaf: &LeafLayout<ModelColumn>,
        grouped: Option<&GroupedLayout>,
        native: &ModelNativePlan,
        advice: &ModelAdviceLayout<'_>,
    ) -> Result<u64, StageError> {
        let plus = |a: u64, b: u64| a.checked_add(b).ok_or(StageError::Memory);
        let times = |a: u64, b: u64| a.checked_mul(b).ok_or(StageError::Memory);
        // walk_model_targets is the same generic traversal used by the numeric
        // target producer. All coefficient/source metadata types are identical;
        // only point elements change from F to Origin. Scale its existing bound
        // by the larger element-size ratio, then add geometry-only ownership.
        let ratio = (size_of::<Origin>() as u64)
            .div_ceil(size_of::<F>() as u64)
            .max(1);
        let target = self.target_memory(lowering, leaf, native)?;
        let mut bytes = times(ratio, target.construction_peak_bytes())?;
        let claims = &native.schedule().stages.first().ok_or_else(bad)?.claims;
        bytes = plus(bytes, geometry_array::<Vec<Origin>>(claims.len() as u64)?)?;
        for claim in claims {
            bytes = plus(bytes, geometry_growing::<Origin>(claim.point.len() as u64)?)?;
        }
        if grouped.is_some() {
            // PublicStage's token-axis analysis, including intermediate output
            // arrays; each merged leaf also creates one temporary group-index list.
            bytes = plus(
                bytes,
                geometry_array::<Option<u8>>(lowering.graph.tensors.len() as u64)?,
            )?;
            for node in &lowering.graph.nodes {
                bytes = plus(
                    bytes,
                    geometry_growing::<Option<u8>>(node.outputs.len().max(1) as u64)?,
                )?;
            }
            bytes = plus(
                bytes,
                times(
                    claims.len() as u64,
                    geometry_growing::<usize>(u64::from(tessera_protocol::MAX_LEAF_GROUP_BITS))?,
                )?,
            )?;
        }
        bytes = plus(
            bytes,
            geometry_group_memory(
                self.groups as u64,
                u64::from(advice.layout().params().log2_k),
            )?,
        )?;
        Ok(bytes)
    }

    /// Maximum distinct advice column points, excluding proximity.
    pub fn groups(&self) -> usize {
        self.groups
    }
    /// Original advice combinations, before expanding source images.
    pub fn advice_claims(&self) -> u64 {
        self.claims
    }
    /// Original advice pieces, before layout expansion.
    pub fn original_pieces(&self) -> u64 {
        self.original_pieces
    }
    /// All mapped advice pieces, retaining repetitions and zero coefficients.
    pub fn advice_images(&self) -> u64 {
        self.images
    }
    /// Coordinates retained across all mapped advice pieces.
    pub fn advice_coordinates(&self) -> u64 {
        self.advice_coordinates
    }
    /// Original claims on `matrix`, retaining every ChannelParams reading site.
    pub fn weight_claims(&self, matrix: WeightMatrix) -> u64 {
        self.weight_claims[matrix.index()]
    }
    /// Coordinates retained across all original claims on `matrix`.
    pub fn weight_coordinates(&self, matrix: WeightMatrix) -> u64 {
        self.weight_coordinates[matrix.index()]
    }
}
fn add(count: &mut u64, entries: usize) -> Result<(), StageError> {
    *count = count
        .checked_add(u64::try_from(entries).map_err(|_| StageError::Memory)?)
        .ok_or(StageError::Memory)?;
    Ok(())
}
struct GeometryTargets<'layout, 'model> {
    layout: &'layout ModelAdviceLayout<'model>,
    groups: HashSet<Vec<Origin>>,
    counts: OpeningGeometry,
}
impl OpeningTargets<Origin> for GeometryTargets<'_, '_> {
    fn advice(&mut self, original: OriginalAdvice<Origin>) -> Result<(), StageError> {
        add(&mut self.counts.claims, 1)?;
        add(&mut self.counts.original_pieces, original.pieces.len())?;
        let k = self.layout.layout().params().log2_k as usize;
        let mut column = None;
        for (source, _) in original.pieces {
            for point in self
                .layout
                .claim_points(source, &original.point, Origin::Zero, Origin::One)
                .map_err(|_| bad())?
            {
                let point = point.map_err(|_| bad())?;
                let prefix = point.get(..k).ok_or_else(bad)?;
                match &column {
                    Some(first) if first != prefix => return Err(bad()),
                    None => column = Some(prefix.to_vec()),
                    _ => {}
                }
                add(&mut self.counts.images, 1)?;
                add(&mut self.counts.advice_coordinates, point.len())?;
            }
        }
        self.groups.try_reserve(1).map_err(|_| StageError::Memory)?;
        self.groups.insert(column.ok_or_else(bad)?);
        Ok(())
    }
    fn weight(&mut self, matrix: WeightMatrix, point: Vec<Origin>) -> Result<(), StageError> {
        add(&mut self.counts.weight_claims[matrix.index()], 1)?;
        add(
            &mut self.counts.weight_coordinates[matrix.index()],
            point.len(),
        )
    }
}
struct PublicLeaf<'a> {
    layout: &'a LeafLayout<ModelColumn>,
    tables: &'a [TableSpec],
    grouped: Option<&'a GroupedLayout>,
}
impl OpeningLeaf<Origin> for PublicLeaf<'_> {
    fn read_point(
        &self,
        layout: &LeafLayout<ModelColumn>,
        site: usize,
        layer: usize,
        read: usize,
    ) -> Option<Vec<Origin>> {
        if let Some(grouped) = self.grouped {
            let mut polynomial = 0usize;
            for (s, entry) in layout.sites().iter().enumerate() {
                for l in 0..entry.layers.len() {
                    let reads = layout.reads(s, l)?.len();
                    if (s, l) == (site, layer) {
                        if read >= reads {
                            return None;
                        }
                        let p = grouped.polynomials().get(polynomial.checked_add(read)?)?;
                        return merged_point(p.vars, &p.group_vars);
                    }
                    polynomial = polynomial.checked_add(reads)?;
                }
            }
            None
        } else {
            Some(
                layout
                    .reads(site, layer)?
                    .get(read)?
                    .coords
                    .iter()
                    .map(|&c| Origin::Leaf(c))
                    .collect(),
            )
        }
    }
    fn multiplicity_point(&self, instance: usize) -> Option<Vec<Origin>> {
        let id = self.layout.instances().get(instance)?.table;
        let height = self.tables.iter().find(|t| t.id == id)?.log2_rows;
        Some(
            (0..height)
                .map(|index| Origin::Table { height, index })
                .collect(),
        )
    }
    fn instances(&self) -> usize {
        self.layout.instances().len()
    }
}
fn merged_point(vars: usize, group_vars: &[usize]) -> Option<Vec<Origin>> {
    if group_vars.windows(2).any(|v| v[0] >= v[1]) || group_vars.last().is_some_and(|&v| v >= vars)
    {
        return None;
    }
    let order = (0..vars)
        .filter(|v| group_vars.binary_search(v).is_err())
        .chain(group_vars.iter().copied());
    let mut point = vec![Origin::Zero; vars];
    for (j, var) in order.enumerate() {
        point[var] = Origin::Merge(vars - 1 - j);
    }
    Some(point)
}
struct PublicStage {
    leaves: Vec<Vec<Origin>>,
    layers: u32,
}
impl PublicStage {
    fn new(
        lowering: &Lowering,
        schedule: &StageSchedule,
        grouped: Option<&GroupedLayout>,
    ) -> Result<Self, StageError> {
        let claims = &schedule.stages.first().ok_or_else(bad)?.claims;
        if schedule.sites.len() != claims.len() {
            return Err(bad());
        }
        let axes = grouped
            .map(|_| tokens(lowering).map_err(|_| bad()))
            .transpose()?;
        let mut leaves = Vec::new();
        leaves
            .try_reserve_exact(claims.len())
            .map_err(|_| StageError::Memory)?;
        for (site, claim) in schedule.sites.iter().zip(claims) {
            let point = if let Some(grouped) = grouped {
                let tensor = match *site {
                    Site::Gadget { node, input } => {
                        let node = match node {
                            NodeRef::Node(n) => n,
                            NodeRef::Position(p) => schedule.layers.node(p, 0),
                        };
                        *lowering
                            .graph
                            .nodes
                            .get(node.0 as usize)
                            .and_then(|n| n.inputs.get(input as usize))
                            .ok_or_else(bad)?
                    }
                    Site::Logits => lowering.logits,
                };
                let shape = &lowering
                    .graph
                    .tensors
                    .get(tensor.0 as usize)
                    .ok_or_else(bad)?
                    .shape;
                let vars = shape
                    .iter()
                    .try_fold(0usize, |n, &d| {
                        n.checked_add(d.checked_next_power_of_two()?.trailing_zeros() as usize)
                    })
                    .ok_or(StageError::Memory)?;
                let group_vars = if let Some(axis) = token(axes.as_deref().ok_or_else(bad)?, tensor)
                {
                    let low = shape
                        .get(axis + 1..)
                        .ok_or_else(bad)?
                        .iter()
                        .try_fold(0usize, |n, &d| {
                            n.checked_add(d.checked_next_power_of_two()?.trailing_zeros() as usize)
                        })
                        .ok_or(StageError::Memory)?;
                    let end = low
                        .checked_add(usize::from(grouped.groups().token_bits))
                        .ok_or(StageError::Memory)?;
                    (low..end).collect()
                } else {
                    Vec::new()
                };
                merged_point(vars, &group_vars).ok_or_else(bad)?
            } else {
                (0..claim.point.len())
                    .map(|j| {
                        u32::try_from(j)
                            .map(Origin::Leaf)
                            .map_err(|_| StageError::Memory)
                    })
                    .collect::<Result<Vec<_>, _>>()?
            };
            if point.len() != claim.point.len() {
                return Err(bad());
            }
            leaves.push(point);
        }
        Ok(Self {
            leaves,
            layers: schedule.layers.count,
        })
    }
}
impl OpeningStage<Origin> for PublicStage {
    fn point(&self, coords: &[Coord]) -> Result<Vec<Origin>, StageError> {
        coords
            .iter()
            .map(|&c| self.challenge(c).ok_or_else(bad))
            .collect()
    }
    fn challenge(&self, coord: Coord) -> Option<Origin> {
        match coord {
            Coord::Half => Some(Origin::Half),
            Coord::Leaf { site, index } => {
                self.leaves.get(site as usize)?.get(index as usize).copied()
            }
            _ => Some(Origin::Stage(coord)),
        }
    }
    fn weights(&self, layers: &Layers, each: Option<u32>) -> Result<Vec<(u32, F)>, StageError> {
        Ok(match layers {
            Layers::None => vec![(0, F::ONE)],
            Layers::One(layer) => vec![(*layer, F::ONE)],
            Layers::Each => vec![(each.ok_or_else(bad)?, F::ONE)],
            Layers::Eq(_) => (0..self.layers).map(|l| (l, F::ONE)).collect(),
        })
    }
    fn coefficient(&self, _: &[Vec<Factor>], _: u32) -> Result<F, StageError> {
        Ok(F::ONE)
    }
    fn layer_weight(&self, _: &[Origin], _: usize) -> F {
        F::ONE
    }
    fn constant(&self, constant: PointConstant) -> Result<Origin, StageError> {
        Ok(match constant {
            PointConstant::Zero => Origin::Zero,
            PointConstant::One => Origin::One,
            PointConstant::Half => Origin::Half,
        })
    }
}

fn geometry_array<T>(n: u64) -> Result<u64, StageError> {
    let bytes = n
        .checked_mul(size_of::<T>() as u64)
        .ok_or(StageError::Memory)?;
    if usize::try_from(n).is_err() || bytes > isize::MAX as u64 {
        return Err(StageError::Memory);
    }
    Ok(bytes)
}
fn geometry_growing<T>(n: u64) -> Result<u64, StageError> {
    if n == 0 {
        return Ok(0);
    }
    let minimum = if size_of::<T>() == 1 {
        8
    } else if size_of::<T>() <= 1024 {
        4
    } else {
        1
    };
    geometry_array::<T>(n.max(minimum).checked_mul(4).ok_or(StageError::Memory)?)
}

fn geometry_group_memory(groups: u64, columns: u64) -> Result<u64, StageError> {
    if groups == 0 {
        return Ok(0);
    }
    let plus = |a: u64, b: u64| a.checked_add(b).ok_or(StageError::Memory);
    let times = |a: u64, b: u64| a.checked_mul(b).ok_or(StageError::Memory);
    let mut bytes = 0;
    // Pinned std/hashbrown table, with both old/new bucket allocations
    // and controls/alignment during growth. Prefix vectors are exact
    // clones; a candidate exists beside all retained distinct prefixes.
    let n = groups;
    let buckets = plus(times(n, 8)?, 6)?
        .checked_div(7)
        .ok_or(StageError::Memory)?
        .max(4)
        .checked_next_power_of_two()
        .ok_or(StageError::Memory)?;
    let entries = times(2, buckets)?;
    bytes = plus(bytes, geometry_array::<Vec<Origin>>(entries)?)?;
    bytes = plus(bytes, geometry_array::<u8>(entries)?)?;
    bytes = plus(bytes, times(31, u64::from(buckets.trailing_zeros()))?)?;
    let points = times(plus(n, 1)?, columns)?;
    bytes = plus(bytes, geometry_array::<Origin>(points)?)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_counts_fail_closed_and_prefix_maps_fit_independently() {
        let checked = allocation_counter::measure(|| {
            assert_eq!(geometry_array::<Origin>(u64::MAX), Err(StageError::Memory));
            assert_eq!(
                geometry_array::<u8>(isize::MAX as u64 + 1),
                Err(StageError::Memory)
            );
            assert_eq!(
                geometry_growing::<Origin>(u64::MAX / 4 + 1),
                Err(StageError::Memory)
            );
            assert_eq!(geometry_group_memory(u64::MAX, 22), Err(StageError::Memory));
            assert_eq!(geometry_group_memory(1, u64::MAX), Err(StageError::Memory));
        });
        assert_eq!(checked.count_total, 0);
        for n in [0usize, 1, 2, 5, 6, 11, 12, 31, 64, 129, 513, 4097] {
            for k in [1usize, 22] {
                let mut groups = HashSet::new();
                let observed = allocation_counter::measure(|| {
                    for j in 0..n {
                        let mut point = vec![Origin::Zero; k];
                        point[0] = Origin::Leaf(j as u32);
                        groups.try_reserve(1).unwrap();
                        groups.insert(point);
                    }
                    if n != 0 {
                        let mut point = vec![Origin::Zero; k];
                        point[0] = Origin::Leaf(0);
                        groups.insert(point);
                    }
                });
                assert!(
                    observed.bytes_max <= geometry_group_memory(n as u64, k as u64).unwrap(),
                    "{n}/{k}: {observed:?}"
                );
            }
        }
    }

    #[test]
    fn grouped_origins_use_the_global_suffix_and_original_variable_positions() {
        // RFC-0054 r(n,S): four variables, free order 0,2 then group order 1,3.
        // A two-variable polynomial uses the last two challenges of the SAME merge.
        assert_eq!(
            merged_point(4, &[1, 3]),
            Some(vec![
                Origin::Merge(3),
                Origin::Merge(1),
                Origin::Merge(2),
                Origin::Merge(0)
            ])
        );
        assert_eq!(
            merged_point(2, &[]),
            Some(vec![Origin::Merge(1), Origin::Merge(0)])
        );
        assert_eq!(merged_point(0, &[]), Some(Vec::new()));
        for group_vars in [vec![1, 1], vec![3, 1], vec![4]] {
            assert_eq!(merged_point(4, &group_vars), None);
        }
    }

    #[test]
    fn opening_count_overflow_refuses_without_wrapping() {
        let mut count = u64::MAX - 1;
        add(&mut count, 1).unwrap();
        assert_eq!(count, u64::MAX);
        assert!(matches!(add(&mut count, 1), Err(StageError::Memory)));
        assert_eq!(count, u64::MAX);
    }
}
