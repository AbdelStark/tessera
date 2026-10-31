// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Allocation requests made by graph validation, from public geometry (RFC-0018).
use super::*;

fn array<T>(count: u64) -> Option<u64> {
    let bytes = count.checked_mul(size_of::<T>() as u64)?;
    (usize::try_from(count).is_ok() && bytes <= isize::MAX as u64).then_some(bytes)
}

// Rust 1.95's doubling/minimum-capacity growth: total requests are below four
// times the largest length/minimum capacity, including replaced allocations.
fn shape_growth(count: u64) -> Option<u64> {
    if count == 0 {
        return Some(0);
    }
    array::<u32>(count.max(4).checked_mul(4)?)
}

impl Graph {
    /// Requested heap storage currently owned by this graph, including spare
    /// capacity in every nested vector (RFC-0006, RFC-0018). Reads lengths and
    /// capacities only; does not allocate, validate, or inspect constant values.
    /// Excludes the inline `Graph`, construction scratch and allocator overhead.
    /// Returns `None` if an allocation is unaddressable or a byte sum overflows.
    pub fn heap_bytes(&self) -> Option<u64> {
        let Self {
            nodes,
            tensors,
            constants,
            shift_classes,
            weights,
            slots,
        } = self;
        let mut bytes = array::<Node>(nodes.capacity() as u64)?
            .checked_add(array::<TensorSpec>(tensors.capacity() as u64)?)?
            .checked_add(array::<Vec<i64>>(constants.capacity() as u64)?)?
            .checked_add(array::<Vec<u8>>(shift_classes.capacity() as u64)?)?
            .checked_add(array::<(WeightRef, TensorId)>(weights.capacity() as u64)?)?;
        for Node {
            id: _,
            op,
            inputs,
            outputs,
        } in nodes
        {
            bytes = bytes
                .checked_add(array::<TensorId>(inputs.capacity() as u64)?)?
                .checked_add(array::<TensorId>(outputs.capacity() as u64)?)?
                .checked_add(op_heap(op)?)?;
        }
        for TensorSpec {
            id: _,
            shape,
            ty: _,
            range: _,
            kind: _,
        } in tensors
        {
            bytes = bytes.checked_add(array::<u32>(shape.capacity() as u64)?)?;
        }
        for values in constants {
            bytes = bytes.checked_add(array::<i64>(values.capacity() as u64)?)?;
        }
        for values in shift_classes {
            bytes = bytes.checked_add(array::<u8>(values.capacity() as u64)?)?;
        }
        if let Some(BatchSlots {
            slot_len: _,
            lengths,
        }) = slots
        {
            bytes = bytes.checked_add(array::<u32>(lengths.capacity() as u64)?)?;
        }
        Some(bytes)
    }

    /// Upper bound on all requested heap bytes of [`Self::validate`] for a valid
    /// graph, including replaced shape buffers. Derives sizes without allocating
    /// or validating the graph. Borrowed graph storage, stack and allocator/runtime
    /// overhead are separate; this is not an admission or authentication check.
    /// Returns `None` when sizing encounters a missing reference or an
    /// overflowing/unaddressable count. Other malformed graphs may yield a bound.
    pub fn validation_memory_bytes(&self) -> Option<u64> {
        let mut bytes = array::<Option<NodeId>>(self.tensors.len() as u64)?;
        for node in &self.nodes {
            bytes = bytes.checked_add(array::<View<'_>>(node.inputs.len() as u64)?)?;
            bytes = bytes.checked_add(array::<View<'_>>(node.outputs.len() as u64)?)?;
            let rank = if let Some(id) = node.inputs.first() {
                self.tensors.get(id.0 as usize)?.shape.len() as u64
            } else {
                0
            };
            let scratch = match node.op {
                Op::Slice { .. } => array::<u32>(rank)?,
                Op::Transpose { .. } => array::<bool>(rank)?.checked_add(array::<u32>(rank)?)?,
                Op::GatherPublic { .. }
                | Op::Expand { .. }
                | Op::OneHot { .. }
                | Op::MatMulAct { .. } => shape_growth(rank.checked_add(1)?)?,
                // Each output clones and grows a separate expected shape.
                Op::BitDecompose { .. } => {
                    shape_growth(rank.checked_add(1)?)?.checked_mul(node.outputs.len() as u64)?
                }
                Op::Sum { .. } | Op::SumSquares { .. } | Op::SoftmaxRow { .. } => {
                    // The filtered collection may be empty, then without_axis owns [1].
                    shape_growth(rank)?.checked_add(array::<u32>(1)?)?
                }
                _ => 0,
            };
            bytes = bytes.checked_add(scratch)?;
        }
        Some(bytes)
    }
}

// Exhaustive fields/variants keep a new heap-owning operation from silently
// inheriting a zero charge. Transpose is the only current owning variant.
fn op_heap(op: &Op) -> Option<u64> {
    match op {
        Op::Transpose { perm } => array::<u8>(perm.capacity() as u64),
        Op::MatMulWeight { weight: _ }
        | Op::ConstMul { c: _ }
        | Op::ConstAdd { c: _ }
        | Op::Rotate { rope: _ }
        | Op::Sum { axis: _ }
        | Op::Slice {
            axis: _,
            start: _,
            len: _,
        }
        | Op::GatherPublic { index: _ }
        | Op::Expand { axis: _, len: _ }
        | Op::MatMulAct { transpose_rhs: _ }
        | Op::Mul { broadcast: _ }
        | Op::SumSquares { axis: _ }
        | Op::OneHot { vocab: _ }
        | Op::Requant {
            shift: _,
            lo: _,
            hi: _,
        }
        | Op::TableMap { table: _ }
        | Op::SoftmaxRow {
            exp_table: _,
            sigma: _,
        }
        | Op::RsqrtRow {
            kappa: _,
            eps_int: _,
            dim: _,
        }
        | Op::InverseCdf { prf: _ }
        | Op::Bitwise { table: _ }
        | Op::BitDecompose { low: _, high: _ }
        | Op::BitFn { f: _ }
        | Op::Add
        | Op::Sub
        | Op::Reshape
        | Op::Argmax
        | Op::AssertEqual => Some(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_refuse_address_and_product_overflow() {
        assert_eq!(array::<u64>((isize::MAX as u64 / 8) + 1), None);
        assert_eq!(array::<u64>((u64::MAX / 8) + 1), None);
        assert_eq!(shape_growth(u64::MAX), None);
        assert_eq!(shape_growth(0), Some(0));
    }
}
