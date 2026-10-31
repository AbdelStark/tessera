// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Original target storage and canonical construction work (RFC-0011/0045/0052).
use super::*;
use tessera_ir::NodeId;

/// Requested-byte bounds for original ModelOpenings targets and their constructor.
/// Borrowed public lowering/layout/leaf/native/stage metadata, OpeningGeometry's
/// own derivation, PCS grouping/accumulators, generic proof/messages, stack/runtime
/// and allocator overhead remain separate. This neither authenticates inputs nor
/// admits a CPU route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpeningTargetMemory {
    retained: u64,
    construction: u64,
}
impl OpeningTargetMemory {
    /// All original advice/weight targets, pieces and point capacities.
    pub fn retained_bytes(self) -> u64 {
        self.retained
    }
    /// Target storage plus canonical traversal/mapping/sorting temporaries.
    pub fn construction_peak_bytes(self) -> u64 {
        self.construction
    }
}
fn add(a: u64, b: u64) -> Result<u64, StageError> {
    a.checked_add(b).ok_or(StageError::Memory)
}
fn mul(a: u64, b: u64) -> Result<u64, StageError> {
    a.checked_mul(b).ok_or(StageError::Memory)
}
fn array<T>(n: u64) -> Result<u64, StageError> {
    let bytes = mul(n, size_of::<T>() as u64)?;
    if usize::try_from(n).is_err() || bytes > isize::MAX as u64 {
        return Err(StageError::Memory);
    }
    Ok(bytes)
}
fn grow<T>(n: u64) -> Result<u64, StageError> {
    if n == 0 {
        return Ok(0);
    }
    array::<T>(mul(4, n.max(4))?)
}
fn separate<T>(entries: u64, vectors: u64) -> Result<u64, StageError> {
    // All types used here have pinned Vec minimum capacity four. Sum growth
    // traffic, including each independently built vector's minimum-capacity slack.
    array::<T>(mul(4, add(entries, mul(4, vectors)?)?)?)
}
impl OpeningGeometry {
    /// Plan the original targets from matching public geometry without allocating
    /// or reading completed proof values. All original combinations and mapped
    /// images remain counted at repeated points and zero coefficients.
    /// Canonical leaf/stage callbacks are assumed; arbitrary callback work is excluded.
    ///
    /// # Errors
    /// `Internal` for invalid public shape metadata; `Memory` for overflow or
    /// unaddressable allocation requests. The geometry must include the full
    /// runtime table selection, including its greedy-range table when present.
    pub fn target_memory(
        &self,
        lowering: &Lowering,
        leaf: &LeafLayout<ModelColumn>,
        native: &ModelNativePlan,
    ) -> Result<OpeningTargetMemory, StageError> {
        let (c, images) = (self.advice_claims(), self.advice_images());
        let mut retained = add(grow::<AdviceClaim>(c)?, separate::<AdvicePoint>(images, c)?)?;
        retained = add(retained, separate::<F>(self.advice_coordinates(), images)?)?;
        let mut max_vars = 0u64;
        let mut rank = 3u64; // attention's head/query/key geometry
        for tensor in &lowering.graph.tensors {
            rank = rank.max(tensor.shape.len() as u64);
            let mut vars = 0;
            for &d in &tensor.shape {
                if d == 0 {
                    return Err(bad());
                }
                vars = add(
                    vars,
                    u64::from(
                        d.checked_next_power_of_two()
                            .ok_or(StageError::Memory)?
                            .trailing_zeros(),
                    ),
                )?;
            }
            max_vars = max_vars.max(vars);
        }
        for matrix in WeightMatrix::ALL {
            let claims = self.weight_claims(matrix);
            let coords = self.weight_coordinates(matrix);
            retained = add(
                retained,
                add(grow::<Vec<F>>(claims)?, separate::<F>(coords, claims)?)?,
            )?;
            max_vars = max_vars.max(coords); // sum is also an upper for each original point
        }
        let schedule = native.schedule();
        for claim in &schedule.pool {
            max_vars = max_vars.max(add(
                claim.point.len() as u64,
                u64::from(schedule.layers.vars),
            )?);
            for atom in &claim.atoms {
                max_vars = max_vars.max(atom.point.len() as u64);
                if let Layers::Eq(p) = &atom.layers {
                    max_vars = max_vars.max(p.len() as u64);
                }
                for product in &atom.coefficient {
                    for factor in product {
                        if let Factor::Lt { coords, .. } = factor {
                            max_vars = max_vars.max(coords.len() as u64);
                        }
                    }
                }
            }
        }
        let (mut sites, mut reads, mut rounds) = (0, 0, 0);
        for site in leaf.sites() {
            sites = add(sites, site.layers.len() as u64)?;
        }
        for (s, site) in leaf.sites().iter().enumerate() {
            for l in 0..site.layers.len() {
                let columns = leaf.reads(s, l).ok_or_else(bad)?;
                reads = add(reads, columns.len() as u64)?;
                for read in columns {
                    max_vars = max_vars.max(read.coords.len() as u64);
                }
            }
        }
        for stage in schedule.stages.iter().skip(1) {
            max_vars = max_vars.max(u64::from(stage.rounds));
            rounds = add(rounds, grow::<F>(u64::from(stage.rounds))?)?;
        }
        // source_openings constructs its complete native-original list before mapping.
        // Charge all originals, a conservative superset of that list.
        let originals = add(
            grow::<OriginalAdvice>(c)?,
            add(
                separate::<F>(mul(c, max_vars)?, c)?,
                separate::<(AdviceSource, F)>(self.original_pieces(), c)?,
            )?,
        )?;
        type Fixed = ((usize, usize, usize), NodeId, usize, usize);
        type Read = (AdviceSource, Vec<F>);
        // Stable sort may retain an element buffer as large as its input.
        let fixed = add(grow::<Fixed>(sites)?, array::<Fixed>(sites)?)?;
        let read_lists = add(
            add(separate::<Read>(reads, sites)?, array::<Read>(reads)?)?,
            separate::<F>(mul(reads, max_vars)?, reads)?,
        )?;
        let rows = mul(mul(2, sites)?, grow::<F>(max_vars)?)?; // Softmax row clone / half-prefix growth
        // Pool point, atom/row point and coefficient point; one returned layer-weight list.
        let pool = add(
            mul(3, grow::<F>(max_vars)?)?,
            array::<(u32, F)>(u64::from(schedule.layers.count).max(1))?,
        )?;
        // One natural point, dense natural-axis starts, or attention local/origin/full shapes.
        let image = add(
            grow::<F>(max_vars)?,
            add(
                array::<usize>(rank)?,
                add(mul(2, grow::<u32>(rank)?)?, array::<u32>(rank)?)?,
            )?,
        )?;
        let construction = add(
            retained,
            add(
                originals,
                add(
                    fixed,
                    add(read_lists, add(rows, add(rounds, add(pool, image)?)?)?)?,
                )?,
            )?,
        )?;
        Ok(OpeningTargetMemory {
            retained,
            construction,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn target_sizes_refuse_address_product_and_independent_vector_overflow() {
        let measured = allocation_counter::measure(|| {
            assert_eq!(
                array::<F>(isize::MAX as u64 / size_of::<F>() as u64 + 1),
                Err(StageError::Memory)
            );
            assert_eq!(
                array::<F>(u64::MAX / size_of::<F>() as u64 + 1),
                Err(StageError::Memory)
            );
            assert_eq!(grow::<F>(u64::MAX), Err(StageError::Memory));
            assert_eq!(
                separate::<AdvicePoint>(0, u64::MAX),
                Err(StageError::Memory)
            );
        });
        assert_eq!(measured.count_total, 0);
    }
}
