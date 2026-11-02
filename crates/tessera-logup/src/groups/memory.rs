// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Complete lookup-group owners around RFC-0054's resident tree kernel.

use tessera_poly::PolyField;
use tessera_sumcheck::{
    Claim, FractionTree, ResidentTreeMemory, ResidentTreeShape, Tree, WindowError,
};

use super::{LookupClaims, group_positions};
use crate::{BoundedLogUpError, LogUpError, TreeId};

/// Public geometry of one group's borrowed lookup columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LookupGroupShape {
    /// Positive unpadded lookup count; the tree pads it to a power of two.
    pub rows: usize,
    /// Whether the lookup also carries a value column and value claim.
    pub values: bool,
}

/// Public geometry of a borrowed product forest in one group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LookupForestShape {
    /// Positive height of each tree.
    pub height: usize,
    /// Variables selecting the trees in the stacked leaf table.
    pub trees_log: usize,
}

/// Requested heap bounds for complete `prove_lookup_group` and its output.
///
/// Borrowed columns/forest leaves, input descriptors and their spare capacity,
/// transcript/encoding callbacks, stack, allocator and process overhead are extra.
/// Surrounding grouped layouts, columns/shares, splits, merge and table batch are
/// also extra. This model admits no CPU route and authenticates no caller shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LookupGroupMemory {
    planning: u64,
    peak: u64,
    output: u64,
}

impl LookupGroupMemory {
    /// Public metadata scratch allocated during this calculation; no field tables.
    pub fn planning_bytes(self) -> u64 {
        self.planning
    }

    /// Complete owned group peak from before tree construction through output.
    pub fn peak_bytes(self) -> u64 {
        self.peak
    }

    /// Returned levels, roots, values and all key/value/forest claim points.
    pub fn output_bytes(self) -> u64 {
        self.output
    }

    /// Price each scheduled tree exactly once, with no witness or challenge input.
    ///
    /// # Errors
    /// Original `LogUpError` for empty lookups or an invalid schedule; a typed
    /// `WindowError` for invalid forest/index geometry or allocation overflow.
    pub fn new<F: PolyField>(
        lookups: &[LookupGroupShape],
        forests: &[LookupForestShape],
        schedule: &[TreeId],
    ) -> Result<Self, BoundedLogUpError> {
        for (instance, shape) in lookups.iter().enumerate() {
            if shape.rows == 0 {
                return Err(LogUpError::NoLookups { instance }.into());
            }
        }
        let n = (lookups.len() as u64)
            .checked_add(forests.len() as u64)
            .ok_or(WindowError::Allocation)?;
        let positions = array::<usize>(n)?;
        let planning = add(positions, array::<ResidentTreeShape>(n)?)?;
        // Reuse the runtime's exact schedule refusal order. Only public index
        // vectors are allocated; retain them while constructing the shape list.
        let (_lookup, _forest) = group_positions(schedule, lookups.len(), forests.len())?;
        let mut shapes = Vec::new();
        shapes
            .try_reserve_exact(schedule.len())
            .map_err(|_| WindowError::Allocation)?;
        for (position, &id) in schedule.iter().enumerate() {
            shapes.push(match id {
                TreeId::Lookup(i) => ResidentTreeShape::Fraction {
                    height: height(lookups[i].rows),
                    base_numerators: false,
                },
                TreeId::Forest(i) => ResidentTreeShape::Forest {
                    height: forests[i].height,
                    trees_log: forests[i].trees_log,
                },
                TreeId::Table(_) => return Err(LogUpError::ScheduleTree { position }.into()),
            });
        }
        let trees = ResidentTreeMemory::new::<F>(&shapes)?;
        let count = lookups.len() as u64;
        let forest_count = forests.len() as u64;
        let roots = array::<F>(mul(2, count)?)?;
        let mut claim_points = 0;
        let mut value_scratch = 0;
        let mut clone_point = 0;
        let mut values = 0;
        for shape in lookups {
            let h = height(shape.rows);
            let point = array::<F>(h as u64)?;
            clone_point = clone_point.max(point);
            claim_points = add(claim_points, mul(if shape.values { 2 } else { 1 }, point)?)?;
            if shape.values {
                values += 1;
                value_scratch = value_scratch.max(value_memory::<F>(shape.rows)?);
            }
        }
        for forest in forests {
            claim_points = add(
                claim_points,
                array::<F>((forest.height + forest.trees_log) as u64)?,
            )?;
        }
        let claims = add(
            array::<LookupClaims<F>>(count)?,
            grow::<Claim<F>>(forest_count)?,
        )?;
        let wrapper_output = add(
            roots,
            add(claims, add(claim_points, capacity::<F>(values)?)?)?,
        )?;
        let output = add(trees.level_bytes(), wrapper_output)?;
        // The original Option tree vector lives until return, and the consumed
        // scheduled Tree vector can lend capacity to the kernel's leaf output.
        let containers = add(
            grow::<Option<FractionTree<F>>>(count)?,
            array::<Tree<'_, F>>(n)?,
        )?;
        let wrapper_scratch = add(
            add(positions, containers)?,
            add(
                array::<(usize, usize)>(count)?,
                add(array::<Option<F>>(count)?, add(clone_point, value_scratch)?)?,
            )?,
        )?;
        // Conservatively keep every wrapper output beside the complete tree
        // peak, leaf claims and post-processing scratch, including Vec growth.
        let peak = add(
            trees.peak_bytes(),
            add(
                add(wrapper_output, grow::<F>(values)?)?,
                add(wrapper_scratch, 64)?, // largest group/{u32}/logup/{roots,values} label
            )?,
        )?;
        Ok(Self {
            planning,
            peak,
            output,
        })
    }
}

// value_at pads only non-power-of-two columns. evaluate_base keeps its low
// equality table and high partial table simultaneously (poly BLOCK_VARS=10).
fn value_memory<F: PolyField>(rows: usize) -> Result<u64, WindowError> {
    let h = height(rows);
    let low = h.min(10);
    let scratch = add(array::<F>(power(low)?)?, array::<F>(power(h - low)?)?)?;
    if rows.is_power_of_two() {
        Ok(scratch)
    } else {
        add(scratch, array::<F::Base>(power(h)?)?)
    }
}

fn height(rows: usize) -> usize {
    (usize::BITS - (rows - 1).leading_zeros()) as usize
}
fn add(a: u64, b: u64) -> Result<u64, WindowError> {
    a.checked_add(b).ok_or(WindowError::Allocation)
}
fn mul(a: u64, b: u64) -> Result<u64, WindowError> {
    a.checked_mul(b).ok_or(WindowError::Allocation)
}
fn power(n: usize) -> Result<u64, WindowError> {
    u32::try_from(n)
        .ok()
        .and_then(|n| 1u64.checked_shl(n))
        .ok_or(WindowError::Allocation)
}
fn array<T>(n: u64) -> Result<u64, WindowError> {
    let bytes = mul(n, size_of::<T>() as u64)?;
    if usize::try_from(n).is_err() || bytes > isize::MAX as u64 {
        return Err(WindowError::Allocation);
    }
    Ok(bytes)
}
fn capacity<T>(n: u64) -> Result<u64, WindowError> {
    if n == 0 {
        return Ok(0);
    }
    let min = if size_of::<T>() == 1 {
        8
    } else if size_of::<T>() <= 1024 {
        4
    } else {
        1
    };
    array::<T>(
        n.max(min)
            .checked_next_power_of_two()
            .ok_or(WindowError::Allocation)?,
    )
}
fn grow<T>(n: u64) -> Result<u64, WindowError> {
    mul(2, capacity::<T>(n)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tessera_field::{Field, Goldilocks, GoldilocksExt2};

    struct NoAllocationTranscript;
    impl tessera_sumcheck::SumcheckTranscript<GoldilocksExt2> for NoAllocationTranscript {
        fn absorb_message(&mut self, _: &str, _: &[GoldilocksExt2]) {}
        fn challenge(&mut self, _: &str) -> GoldilocksExt2 {
            GoldilocksExt2::ONE
        }
    }

    #[test]
    fn public_lookup_group_memory_covers_wrapper_output_without_level_headroom() {
        // Drop only the levels before reading retained bytes: the deliberately
        // conservative tree bound must not hide a missing wrapper point/claim.
        for rows in [1, 2, 15, 16] {
            for values in [false, true] {
                let keys = vec![Goldilocks::ONE; rows];
                let instances: Vec<_> = (0..8)
                    .map(|_| super::super::LookupInstance {
                        lookups: crate::Tuples::new(&keys, values.then_some(keys.as_slice())),
                        padding_key: Goldilocks::ONE,
                        padding_value: values.then_some(Goldilocks::ONE),
                    })
                    .collect();
                let shapes = [LookupGroupShape { rows, values }; 8];
                let schedule: Vec<_> = (0..8).map(TreeId::Lookup).collect();
                let trees = [ResidentTreeShape::Fraction {
                    height: height(rows),
                    base_numerators: false,
                }; 8];
                let levels = ResidentTreeMemory::new::<GoldilocksExt2>(&trees)
                    .unwrap()
                    .level_bytes();
                let bound = LookupGroupMemory::new::<GoldilocksExt2>(&shapes, &[], &schedule)
                    .unwrap()
                    .output_bytes()
                    - levels;
                let mut output = None;
                let run = allocation_counter::measure(|| {
                    let mut proven = super::super::prove_lookup_group(
                        &mut NoAllocationTranscript,
                        u32::MAX,
                        GoldilocksExt2::ONE,
                        GoldilocksExt2::ONE,
                        &instances,
                        &[],
                        &schedule,
                    )
                    .unwrap();
                    drop(core::mem::take(&mut proven.levels));
                    output = Some(proven);
                });
                assert!(
                    run.bytes_current as u64 <= bound,
                    "wrapper ({rows},{values}): {run:?} > {bound}"
                );
                drop(output);
            }
        }
    }

    #[test]
    fn public_lookup_group_memory_covers_each_value_evaluation_owner() {
        for rows in [1, 2, 3, 1023, 1024, 1025, 4095, 4096, 4097, 65535] {
            let values = vec![Goldilocks::ONE; rows];
            let point = vec![GoldilocksExt2::ONE; height(rows)];
            let bound = value_memory::<GoldilocksExt2>(rows).unwrap();
            let run = allocation_counter::measure(|| {
                core::hint::black_box(crate::prover::value_at::<GoldilocksExt2>(
                    &values,
                    Goldilocks::ONE,
                    &point,
                ));
            });
            assert!(
                run.bytes_max <= bound,
                "value_at({rows}): {run:?} > {bound}"
            );
            assert_eq!(run.bytes_current, 0);
        }
    }
}
