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
