// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public geometry of the computed score operands (RFC-0056, #641).
use super::super::*;

/// Requested clearing buffers of the computed score inputs, including raw/padded
/// conversion overlap. Shared inputs are counted once per Softmax family. This
/// excludes their Arc/descriptor, borrowed sources, other native tables, rounds,
/// runtime/allocator overhead and all other proof phases; it does not admit a route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScoreInputMemory {
    pub(super) retained: u64,
    pub(super) peak: u64,
}
impl ScoreInputMemory {
    /// Sum of retained padded INT8 queries/keys and base-field maximum copies.
    pub fn retained_bytes(self) -> u64 {
        self.retained
    }
    /// Largest requested input-buffer footprint during sequential construction,
    /// including all earlier families' retained inputs.
    pub fn peak_bytes(self) -> u64 {
        self.peak
    }
}

fn product(values: &[usize]) -> Result<usize, StageError> {
    values
        .iter()
        .try_fold(1usize, |n, &v| n.checked_mul(v).ok_or(StageError::Memory))
}
fn array<T>(n: usize) -> Result<u64, StageError> {
    let bytes = n.checked_mul(size_of::<T>()).ok_or(StageError::Memory)?;
    if bytes > isize::MAX as usize {
        return Err(StageError::Memory);
    }
    u64::try_from(bytes).map_err(|_| StageError::Memory)
}
fn add(a: u64, b: u64) -> Result<u64, StageError> {
    a.checked_add(b).ok_or(StageError::Memory)
}

pub(in crate::native) fn ids(
    lowering: &Lowering,
    id: NodeId,
) -> Result<(TensorId, TensorId), StageError> {
    let node = lowering.graph.nodes.get(id.0 as usize).ok_or_else(bad)?;
    if !matches!(node.op, Op::SoftmaxRow { .. }) {
        return Err(bad());
    }
    let input = *node.inputs.first().ok_or_else(bad)?;
    let product = lowering.graph.producer(input).ok_or_else(bad)?;
    let product = lowering
        .graph
        .nodes
        .get(product.0 as usize)
        .ok_or_else(bad)?;
    if !matches!(
        product.op,
        Op::MatMulAct {
            transpose_rhs: true
        }
    ) {
        return Err(bad());
    }
    let &[q, mut k] = product.inputs.as_slice() else {
        return Err(bad());
    };
    // RFC-0037/0056: read K before GQA repetition.
    if let Some(r) = lowering.graph.producer(k) {
        let r = lowering.graph.nodes.get(r.0 as usize).ok_or_else(bad)?;
        if matches!(r.op, Op::Reshape)
            && let Some(e) = r
                .inputs
                .first()
                .and_then(|&input| lowering.graph.producer(input))
        {
            let e = lowering.graph.nodes.get(e.0 as usize).ok_or_else(bad)?;
            if matches!(e.op, Op::Expand { axis: 1, .. }) {
                k = *e.inputs.first().ok_or_else(bad)?;
            }
        }
    }
    Ok((q, k))
}

pub(in crate::native) struct Shape {
    pub heads: usize,
    pub key_heads: usize,
    pub positions: usize,
    pub dim: usize,
    pub raw_queries: usize,
    pub raw_keys: usize,
    pub padded_queries: usize,
    pub padded_keys: usize,
    pub maxima: usize,
}
impl Shape {
    pub(in crate::native) fn derive(f: &Family, lowering: &Lowering) -> Result<Self, StageError> {
        let mut dimensions = None;
        for &id in &f.nodes {
            let (q, k) = ids(lowering, id)?;
            let qt = lowering.graph.tensors.get(q.0 as usize).ok_or_else(bad)?;
            let kt = lowering.graph.tensors.get(k.0 as usize).ok_or_else(bad)?;
            let (&[h, n, d], &[kh, kn, kd]) = (qt.shape.as_slice(), kt.shape.as_slice()) else {
                return Err(bad());
            };
            if kh == 0
                || h == 0
                || n == 0
                || d == 0
                || h % kh != 0
                || n != kn
                || d != kd
                || qt.range.lo < -128
                || qt.range.hi > 127
                || kt.range.lo < -128
                || kt.range.hi > 127
            {
                return Err(bad());
            }
            let current = (h as usize, kh as usize, n as usize, d as usize);
            if dimensions.is_some_and(|g| g != current) {
                return Err(bad());
            }
            dimensions = Some(current);
        }
        let (heads, key_heads, positions, dim) = dimensions.ok_or_else(bad)?;
        Self::new(f.nodes.len(), f.lambda, heads, key_heads, positions, dim)
    }
    fn new(
        layers: usize,
        lambda: u32,
        heads: usize,
        key_heads: usize,
        positions: usize,
        dim: usize,
    ) -> Result<Self, StageError> {
        let padded_layers = 1usize.checked_shl(lambda).ok_or(StageError::Memory)?;
        if layers == 0
            || layers > padded_layers
            || heads == 0
            || key_heads == 0
            || positions == 0
            || dim == 0
            || !heads.is_multiple_of(key_heads)
        {
            return Err(bad());
        }
        let hp = heads
            .checked_next_power_of_two()
            .ok_or(StageError::Memory)?;
        let np = positions
            .checked_next_power_of_two()
            .ok_or(StageError::Memory)?;
        let out = Self {
            heads,
            key_heads,
            positions,
            dim,
            raw_queries: product(&[layers, heads, positions, dim])?,
            raw_keys: product(&[layers, key_heads, positions, dim])?,
            padded_queries: product(&[padded_layers, hp, np, dim])?,
            padded_keys: product(&[padded_layers, key_heads, np, dim])?,
            maxima: product(&[padded_layers, hp, np])?,
        };
        out.memory()?;
        Ok(out)
    }
    pub(in crate::native) fn memory(&self) -> Result<ScoreInputMemory, StageError> {
        let rq = array::<i8>(self.raw_queries)?;
        let rk = array::<i8>(self.raw_keys)?;
        let pq = array::<i8>(self.padded_queries)?;
        let pk = array::<i8>(self.padded_keys)?;
        let m = array::<G>(self.maxima)?;
        // The old input is dropped immediately after its padded copy is built.
        let peak = add(pq, add(rq, rk)?.max(add(rk, pk)?).max(add(pk, m)?))?;
        Ok(ScoreInputMemory {
            retained: add(add(pq, pk)?, m)?,
            peak,
        })
    }
}

impl ModelNativePlan {
    /// Plan the computed RFC-0056 query/key/maximum buffer families from public
    /// lowering metadata without reading witness values or allocating. Slotted batches
    /// use the existing literal path and contribute no computed input buffers.
    /// This covers `ModelNativeProver::new`; the explicit literal oracle is separate.
    ///
    /// # Errors
    /// `Internal` for inconsistent graph/shape metadata, `Memory` for overflow or
    /// requested arrays outside the target address space. Other native work is separate.
    pub fn score_input_memory(&self, lowering: &Lowering) -> Result<ScoreInputMemory, StageError> {
        let mut out = ScoreInputMemory {
            retained: 0,
            peak: 0,
        };
        if lowering.graph.slots.is_some() {
            return Ok(out);
        }
        for family in &self.families {
            if matches!(family.kind, Kind::Softmax { .. }) {
                let memory = Shape::derive(family, lowering)?.memory()?;
                out.peak = out.peak.max(add(out.retained, memory.peak)?);
                out.retained = add(out.retained, memory.retained)?;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn score_input_memory_covers_short_contexts_and_is_allocation_free() {
        let mut plan = None;
        let measured = allocation_counter::measure(|| {
            let shape = Shape::new(16, 4, 32, 8, 2, 64).unwrap();
            assert!(shape.padded_queries > 16 * 32 * 2 * 2);
            plan = Some(shape.memory().unwrap());
        });
        assert_eq!(measured.count_total, 0);
        let plan = plan.unwrap();
        assert_eq!(
            plan.retained_bytes(),
            16 * 32 * 2 * 64 + 16 * 8 * 2 * 64 + 8 * 16 * 32 * 2
        );
        assert_eq!(plan.peak_bytes(), 2 * 16 * 32 * 2 * 64 + 16 * 8 * 2 * 64);
    }
    #[test]
    fn score_input_memory_refuses_each_invalid_dimension_without_allocating() {
        let measured = allocation_counter::measure(|| {
            for (layers, lambda, heads, keys, positions, dim) in [
                (0, 0, 1, 1, 1, 1),
                (2, 0, 1, 1, 1, 1),
                (1, 0, 0, 1, 1, 1),
                (1, 0, 1, 0, 1, 1),
                (1, 0, 1, 1, 0, 1),
                (1, 0, 1, 1, 1, 0),
                (1, 0, 3, 2, 1, 1),
            ] {
                assert_eq!(
                    Shape::new(layers, lambda, heads, keys, positions, dim).err(),
                    Some(bad())
                );
            }
        });
        assert_eq!(measured.count_total, 0);
    }
    #[test]
    fn score_input_memory_refuses_address_and_product_overflow_without_allocating() {
        let measured = allocation_counter::measure(|| {
            assert_eq!(
                array::<G>(isize::MAX as usize / 8 + 1),
                Err(StageError::Memory)
            );
            assert_eq!(array::<G>(usize::MAX / 8 + 1), Err(StageError::Memory));
            assert_eq!(product(&[2, usize::MAX / 2 + 1]), Err(StageError::Memory));
            assert!(matches!(
                Shape::new(1, 0, 1, 1, usize::MAX, 1),
                Err(StageError::Memory)
            ));
            assert!(matches!(
                Shape::new(1, 0, 1, 1, 1, usize::MAX),
                Err(StageError::Memory)
            ));
        });
        assert_eq!(measured.count_total, 0);
    }
}
