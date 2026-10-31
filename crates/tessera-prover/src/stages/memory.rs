// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Persistent stage-driver state from a compiled public schedule (RFC-0018/0037).
use super::{Coord, GoldilocksExt2 as F, Graph, Layers, NodeId, StageError, StageSchedule};

/// Canonical stage-driver state only. Borrowed graph/schedule/witness, pending
/// window preparations, relation arenas, temporary evaluations/claims, returned
/// messages, callbacks, stack/runtime and allocator overhead are additional.
/// This neither validates a schedule nor admits a CPU route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageStateMemory {
    construction: u64,
    retained: u64,
}
impl StageStateMemory {
    /// Peak requests during StageProver::new/for_graph, before leaf evaluation.
    pub fn construction_peak_bytes(self) -> u64 {
        self.construction
    }
    /// Persistent state through the complete schedule, excluding pending windows.
    pub fn retained_bytes(self) -> u64 {
        self.retained
    }
    /// Bound canonical constructor/state growth without allocation or field reads.
    /// The graph and schedule must already have passed public compilation checks.
    /// Every claim/part/layer is counted before any algebraic cancellation.
    ///
    /// # Errors
    /// Empty schedule, or overflowing/unaddressable public allocation geometry.
    pub fn new(graph: &Graph, schedule: &StageSchedule) -> Result<Self, StageError> {
        if schedule.stages.is_empty() {
            return Err(StageError::Internal {
                rule: "stage state has a compiled leaf stage",
            });
        }
        let tensors = graph.tensors.len() as u64;
        let producers = array::<Option<(NodeId, u32)>>(tensors)?;
        let base = add(
            array::<bool>(tensors)?,
            add(producers, array::<Option<F>>(schedule.pool.len() as u64)?)?,
        )?;
        // PublicTensors::of first derives its own producer vector before the
        // driver's producer vector is built. Charge both, conservatively.
        let construction = add(base, producers)?;
        let stages = schedule.stages.len() as u64;
        let mut retained = add(
            base,
            add(
                growing::<Vec<Vec<F>>>(stages)?,
                growing::<Vec<Vec<Vec<F>>>>(stages)?,
            )?,
        )?;
        let mut coordinates = u64::from(schedule.layers.vars);
        for (index, stage) in schedule.stages.iter().enumerate() {
            if index == 0 {
                for claim in &stage.claims {
                    coordinates = add(coordinates, claim.point.len() as u64)?;
                }
            } else {
                coordinates = add(coordinates, u64::from(stage.rounds))?;
            }
            // Only stage one currently inserts Zero coordinates; summing all
            // stage declarations is conservative for any valid compiled schedule.
            coordinates = add(coordinates, u64::from(stage.zero_vars))?;
            let claims = stage.claims.len() as u64;
            retained = add(
                retained,
                add(array::<Vec<F>>(claims)?, array::<Vec<Vec<F>>>(claims)?)?,
            )?;
            for claim in &stage.claims {
                let layers = if claim.layers == Layers::Each {
                    u64::from(schedule.layers.count).max(1)
                } else {
                    1
                };
                retained = add(retained, growing::<F>(layers)?)?;
                let parts = claim.parts.len() as u64;
                retained = add(
                    retained,
                    add(array::<Vec<F>>(parts)?, mul(parts, growing::<F>(layers)?)?)?,
                )?;
            }
        }
        retained = add(retained, coordinate_map(coordinates)?)?;
        Ok(Self {
            construction,
            retained,
        })
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
fn growing<T>(n: u64) -> Result<u64, StageError> {
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
    array::<T>(mul(4, n.max(minimum))?)
}
fn coordinate_map(n: u64) -> Result<u64, StageError> {
    if n == 0 {
        return Ok(0);
    }
    // Pinned std/hashbrown: <=7/8 load, power-of-two buckets, minimum four.
    // Twice the largest bucket array bounds all growth allocations together;
    // each allocation adds at most 16 control bytes and 15 alignment bytes.
    let buckets = (add(mul(n, 8)?, 6)? / 7)
        .max(4)
        .checked_next_power_of_two()
        .ok_or(StageError::Memory)?;
    let all = mul(2, buckets)?;
    add(
        add(array::<(Coord, F)>(all)?, array::<u8>(all)?)?,
        mul(31, u64::from(buckets.trailing_zeros()))?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tessera_field::Field;
    #[test]
    fn coordinate_map_bound_covers_real_growth_requests() {
        for n in [0usize, 1, 3, 4, 7, 8, 31, 512, 513, 4097] {
            let bound = coordinate_map(n as u64).unwrap();
            let measured = allocation_counter::measure(|| {
                let mut map = HashMap::new();
                for index in 0..n {
                    map.insert(
                        Coord::Leaf {
                            site: 0,
                            index: index as u32,
                        },
                        F::ZERO,
                    );
                }
                assert_eq!(map.len(), n);
            });
            assert!(measured.bytes_total <= bound, "{n}: {measured:?}, {bound}");
        }
    }
    #[test]
    fn state_allocation_limits_fail_closed_without_allocating() {
        let measured = allocation_counter::measure(|| {
            assert_eq!(
                array::<u64>(isize::MAX as u64 / 8 + 1),
                Err(StageError::Memory)
            );
            assert_eq!(add(u64::MAX, 1), Err(StageError::Memory));
            assert_eq!(mul(u64::MAX, 2), Err(StageError::Memory));
            assert_eq!(growing::<F>(u64::MAX), Err(StageError::Memory));
            assert_eq!(coordinate_map(u64::MAX), Err(StageError::Memory));
        });
        assert_eq!(measured.count_total, 0);
    }
}
