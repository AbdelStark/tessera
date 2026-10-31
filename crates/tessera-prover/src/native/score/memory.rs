// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Additional computed-score round state, excluding retained inputs (RFC-0056).
use super::super::{
    Kind, Lowering, ModelNativePlan, StageError,
    memory::{add, array, growing, mul},
};
use super::{F, State, inputs::Shape};

fn peak(shape: &Shape, layer_vars: u32) -> Result<u64, StageError> {
    let positions = shape
        .positions
        .checked_next_power_of_two()
        .ok_or(StageError::Memory)?;
    let heads = shape
        .heads
        .checked_next_power_of_two()
        .ok_or(StageError::Memory)?;
    let key_vars = positions.trailing_zeros();
    let vars = key_vars
        .checked_mul(2)
        .and_then(|n| n.checked_add(heads.trailing_zeros()))
        .and_then(|n| n.checked_add(layer_vars))
        .ok_or(StageError::Memory)?;
    if vars > 62 {
        return Err(StageError::Memory);
    }
    let weights = array::<F>(positions as u64)?;
    let dimension = array::<F>(shape.dim as u64)?;
    // key_round retains a cloned/extended public point, weights, and two folds.
    let key = add(
        growing::<F>(u64::from(key_vars))?,
        add(weights, mul(2, dimension)?)?,
    )?;
    // make_rows retains four full padded row arrays, weights and one fold.
    let rows = add(
        mul(4, array::<F>(shape.maxima as u64)?)?,
        add(weights, dimension)?,
    )?;
    add(
        add(array::<State<'_>>(1)?, growing::<F>(u64::from(vars))?)?,
        key.max(rows),
    )
}

impl ModelNativePlan {
    /// Additional requested-byte peak for all computed score working states,
    /// including their boxes, growing bound points, key-round temporaries and
    /// four clearing row arrays. Both key and forest states are charged at once.
    /// Computes from matching public lowering metadata without allocating.
    /// Slotted batches use literal scores and return zero.
    ///
    /// Retained native inputs, generic sumcheck wrappers/messages, final-output
    /// vectors, windows, stack/runtime and allocator overhead are separate. This
    /// neither authenticates metadata nor admits a CPU route.
    ///
    /// # Errors
    /// `Internal` for inconsistent score geometry; `Memory` for overflowing or
    /// unaddressable arrays, or more than 62 computed variables.
    pub fn score_round_memory_bytes(&self, lowering: &Lowering) -> Result<u64, StageError> {
        if lowering.graph.slots.is_some() {
            return Ok(0);
        }
        let mut bytes = 0;
        for family in &self.families {
            if matches!(family.kind, Kind::Softmax { .. }) {
                bytes = add(
                    bytes,
                    mul(2, peak(&Shape::derive(family, lowering)?, family.lambda)?)?,
                )?;
            }
        }
        Ok(bytes)
    }
}

#[cfg(test)]
pub(super) fn check(score: &super::Score) {
    use super::{Computed, Field};
    let a = &score.inputs;
    let shape = Shape {
        heads: a.heads,
        key_heads: a.key_heads,
        positions: a.positions,
        dim: a.dim,
        raw_queries: 0,
        raw_keys: 0,
        padded_queries: a.queries.len(),
        padded_keys: a.keys.len(),
        maxima: a.maxima.len(),
    };
    let mut budget = None;
    let measured =
        allocation_counter::measure(|| budget = Some(peak(&shape, a.layer_vars as u32).unwrap()));
    assert_eq!(measured.count_total, 0);
    let budget = budget.unwrap();
    let mut state = None;
    let measured = allocation_counter::measure(|| state = Some(score.start()));
    assert!(measured.bytes_max <= budget, "start {measured:?} {budget}");
    let mut retained = measured.bytes_current;
    let mut state = state.unwrap();
    let nodes = [F::ZERO, F::ONE];
    let mut values = [F::ZERO; 2];
    for &challenge in &score.point {
        let measured = allocation_counter::measure(|| state.round(&nodes, &mut values));
        assert!(
            u64::try_from(retained).unwrap() + measured.bytes_max <= budget,
            "round {measured:?} {retained} {budget}"
        );
        let measured = allocation_counter::measure(|| state.bind(challenge));
        assert!(
            u64::try_from(retained).unwrap() + measured.bytes_max <= budget,
            "bind {measured:?} {retained} {budget}"
        );
        retained += measured.bytes_current;
    }
    // evaluations appends into the caller-owned output vector, charged separately.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_sizes_refuse_unaddressable_buffers_and_excess_variables() {
        let mut shape = Shape {
            heads: 1,
            key_heads: 1,
            positions: 1,
            dim: 1,
            raw_queries: 1,
            raw_keys: 1,
            padded_queries: 1,
            padded_keys: 1,
            maxima: 1,
        };
        let measured = allocation_counter::measure(|| {
            assert_eq!(peak(&shape, 63), Err(StageError::Memory));
            shape.positions = usize::MAX;
            assert_eq!(peak(&shape, 0), Err(StageError::Memory));
            shape.positions = 1;
            shape.dim = usize::MAX;
            assert_eq!(peak(&shape, 0), Err(StageError::Memory));
            shape.dim = 1;
            shape.maxima = usize::MAX;
            assert_eq!(peak(&shape, 0), Err(StageError::Memory));
        });
        assert_eq!(measured.count_total, 0);
    }
}
