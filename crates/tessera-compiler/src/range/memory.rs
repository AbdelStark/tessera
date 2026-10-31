// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public allocation traffic for range analysis and committed-input coverage.
use super::*;

fn array<T>(count: u64) -> Option<u64> {
    let bytes = count.checked_mul(size_of::<T>() as u64)?;
    (usize::try_from(count).is_ok() && bytes <= isize::MAX as u64).then_some(bytes)
}

/// Upper bound on every heap allocation request of [`analyze`] for a valid graph,
/// including graph validation, returned ranges and temporary coverage bitsets.
/// Computes from public geometry without allocating or analyzing ranges/tables.
/// Borrowed graph/table storage, stack and allocator/runtime overhead are separate.
/// This does not authenticate the graph or replace any analysis check.
/// Returns `None` when sizing encounters a missing reference, a move-walk cycle
/// or an overflowing/unaddressable size. Other malformed graphs may yield a bound.
pub fn analysis_memory_bytes(graph: &Graph) -> Option<u64> {
    let nt = graph.tensors.len() as u64;
    let mut bytes = graph.validation_memory_bytes()?;
    for value in [
        array::<Range>(nt)?,
        array::<Option<NodeId>>(nt)?,
        array::<Option<Reached>>(nt)?,
    ] {
        bytes = bytes.checked_add(value)?;
    }
    for node in &graph.nodes {
        bytes = bytes.checked_add(array::<Range>(node.outputs.len() as u64)?)?;
        if !matches!(node.op, Op::BitDecompose { .. }) {
            continue;
        }
        // Each BitDecompose may walk its own chain. Shared chains are not
        // deduplicated: repeated readers create fresh coverage allocations.
        let mut tensor = *node.inputs.first()?;
        let mut all = true;
        let mut steps = 0usize;
        while let Some(id) = graph.producer(tensor) {
            steps = steps.checked_add(1)?;
            if steps > graph.nodes.len() {
                return None;
            }
            let producer = graph.nodes.get(id.0 as usize)?;
            let input = *producer.inputs.first()?;
            match producer.op {
                Op::Reshape => {}
                Op::Transpose { .. } | Op::Expand { .. } if all => {}
                Op::Transpose { .. }
                | Op::Expand { .. }
                | Op::Slice { .. }
                | Op::GatherPublic { .. } => {
                    let from = &graph.tensors.get(input.0 as usize)?.shape;
                    let to = &graph.tensors.get(tensor.0 as usize)?.shape;
                    let count = from
                        .iter()
                        .try_fold(1u64, |n, &d| n.checked_mul(u64::from(d)))?;
                    bytes = bytes.checked_add(array::<bool>(count)?)?;
                    bytes = bytes.checked_add(array::<usize>(from.len() as u64)?)?;
                    bytes = bytes.checked_add(array::<usize>(to.len() as u64)?)?;
                    all = false;
                }
                _ => break,
            }
            tensor = input;
        }
    }
    Some(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_arrays_refuse_address_and_product_overflow() {
        assert_eq!(
            array::<usize>((isize::MAX as u64 / size_of::<usize>() as u64) + 1),
            None
        );
        assert_eq!(
            array::<Range>((u64::MAX / size_of::<Range>() as u64) + 1),
            None
        );
    }
}
