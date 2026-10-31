// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public buffer bounds for table loading and token/checkpoint/cache construction.

use tessera_codec::SecretBuf;
use tessera_compiler::Lowering;
use tessera_ir::{Op, TensorId};

use crate::WitnessError;

/// Checked requested-buffer storage for canonical token/checkpoint/cache construction.
/// Graph-input INT8 model weights stay borrowed; every nonweight, including full score
/// tensors, is charged. Borrowed engine tables/graph/model, thread-runtime/stack storage,
/// tile/gadget/native work, allocator overhead and other proof phases remain separate.
/// This helper does not admit a CPU route or promise a whole-process RSS bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointMemory {
    tokens: u64,
    store: u64,
    cache: u64,
    token_peak: u64,
    checkpoint_peak: u64,
    cache_peak: u64,
    retained: u64,
}

type Result<T> = core::result::Result<T, WitnessError>;
fn memory(tensor: TensorId) -> WitnessError {
    WitnessError::Memory { tensor }
}
pub(crate) fn add(a: u64, b: u64, tensor: TensorId) -> Result<u64> {
    a.checked_add(b).ok_or_else(|| memory(tensor))
}
pub(crate) fn mul(a: u64, b: u64, tensor: TensorId) -> Result<u64> {
    a.checked_mul(b).ok_or_else(|| memory(tensor))
}
pub(crate) fn array<T>(count: u64, tensor: TensorId) -> Result<u64> {
    let bytes = mul(count, size_of::<T>() as u64, tensor)?;
    if usize::try_from(count).is_err() || bytes > isize::MAX as u64 {
        return Err(memory(tensor));
    }
    Ok(bytes)
}
fn cells(lowering: &Lowering, tensor: TensorId) -> Result<u64> {
    let spec = lowering
        .graph
        .tensors
        .get(tensor.0 as usize)
        .ok_or(WitnessError::Usage {
            rule: "a checkpoint memory tensor belongs to the lowering",
        })?;
    spec.shape
        .iter()
        .try_fold(1, |count, &length| mul(count, u64::from(length), tensor))
}
impl CheckpointMemory {
    /// Plan from the public lowering without reading a model/token or allocating.
    /// All tensor shapes describe real integer domains before PCS padding.
    ///
    /// # Errors
    /// Missing referenced tensors or overflowing/unaddressable buffer counts.
    pub fn new(lowering: &Lowering) -> Result<Self> {
        let graph = &lowering.graph;
        let context = lowering.tokens;
        let (nt, nn, nc) = (
            graph.tensors.len() as u64,
            graph.nodes.len() as u64,
            lowering.checkpoints.len() as u64,
        );
        // Only a graph input which is not also a checkpoint can stay borrowed in replay.
        let borrowed = |id| {
            graph.weights.iter().any(|&(_, tensor)| tensor == id)
                && !graph.nodes.iter().any(|node| node.outputs.contains(&id))
                && !lowering.checkpoints.contains(&id)
        };
        let values = graph.tensors.iter().try_fold(0, |sum, tensor| {
            if borrowed(tensor.id) {
                Ok(sum)
            } else {
                add(
                    sum,
                    array::<i64>(cells(lowering, tensor.id)?, tensor.id)?,
                    context,
                )
            }
        })?;
        let mut edges = 0;
        let mut arity = 3u64; // fused attention reads q, k and v even when graph nodes are binary.
        let mut rank = graph
            .tensors
            .iter()
            .map(|tensor| tensor.shape.len() as u64)
            .max()
            .unwrap_or(0);
        let mut extra_inputs = 0;
        for node in &graph.nodes {
            edges = add(edges, node.inputs.len() as u64, context)?;
            arity = arity.max(node.inputs.len() as u64);
            if let Op::Transpose { perm } = &node.op {
                rank = rank.max(perm.len() as u64);
            }
            let inputs = node.inputs.iter().try_fold(0, |sum, &tensor| {
                // Repeated leaf inputs each own a fresh copy; do not deduplicate them.
                if borrowed(tensor) {
                    Ok(sum)
                } else {
                    add(
                        sum,
                        array::<i64>(cells(lowering, tensor)?, tensor)?,
                        context,
                    )
                }
            })?;
            extra_inputs = extra_inputs.max(inputs);
        }
        let store_values = lowering.checkpoints.iter().try_fold(0, |sum, &tensor| {
            add(
                sum,
                array::<i32>(cells(lowering, tensor)?, tensor)?,
                context,
            )
        })?;
        let store_headers = add(
            array::<TensorId>(nc, context)?,
            array::<SecretBuf<i32>>(nc, context)?,
            context,
        )?;
        let store = add(store_values, store_headers, context)?;
        let token_count = cells(lowering, lowering.positions)?;
        let tokens = array::<u32>(token_count, context)?;
        let token_peak = add(tokens, array::<i64>(token_count, context)?, context)?;
        let cache_headers = add(
            array::<Option<SecretBuf<i64>>>(nt, context)?,
            array::<Option<&[i8]>>(nt, context)?,
            context,
        )?;
        let cache = add(cache_headers, values, context)?;

        // Public structure construction, including reader count/edge arrays and at
        // most one exact-reserved attention Core per graph node. Sum is conservative
        // across constructor stages; no std Vec growth factor is assumed.
        let mut scratch = add(
            array::<Option<u32>>(nt, context)?,
            array::<Option<usize>>(nt, context)?,
            context,
        )?;
        for bytes in [
            array::<crate::structure::Core>(nn, context)?,
            array::<Vec<u32>>(nt, context)?,
            array::<u32>(edges, context)?,
            array::<usize>(nt, context)?,
        ] {
            scratch = add(scratch, bytes, context)?;
        }
        // Live buffers and reachability/remaining-reader metadata; reverse reachability has no stack.
        for bytes in [
            array::<Option<SecretBuf<i64>>>(nt, context)?,
            array::<bool>(mul(2, nn, context)?, context)?,
            array::<Option<&crate::structure::Core>>(nn, context)?,
            array::<u32>(nt, context)?,
            array::<TensorId>(arity, context)?,
        ] {
            scratch = add(scratch, bytes, context)?;
        }
        scratch = add(
            scratch,
            crate::replay::operand_metadata_bytes(arity).ok_or_else(|| memory(context))?,
            context,
        )?;
        // Transpose holds dimensions, strides, axes and its index; BitFn has at
        // most three slice references, and every node returns at most two buffers.
        for bytes in [
            array::<usize>(mul(5, rank, context)?, context)?,
            array::<&[i64]>(3, context)?,
            array::<SecretBuf<i64>>(2, context)?,
        ] {
            scratch = add(scratch, bytes, context)?;
        }
        // graph_input(positions) holds the public positions_values Vec beside its
        // clearing copy. It is not a retained tensor-cache allocation.
        scratch = add(scratch, array::<i64>(token_count, context)?, context)?;
        let work = add(add(values, extra_inputs, context)?, scratch, context)?;
        let checkpoint_peak = add(add(tokens, store, context)?, work, context)?;
        // Cached outputs transfer from Live without copying. Charge that union
        // once, plus cache containers, wanted ids and Checkpoints::fits' empty store.
        let cache_peak = add(
            add(checkpoint_peak, cache_headers, context)?,
            add(array::<TensorId>(nt, context)?, store_headers, context)?,
            context,
        )?;
        let retained = add(add(tokens, store, context)?, cache, context)?;
        Ok(Self {
            tokens,
            store,
            cache,
            token_peak,
            checkpoint_peak,
            cache_peak,
            retained,
        })
    }
    /// Retained owned canonical u32 tokens, excluding the caller's input sequence.
    pub fn tokens_bytes(self) -> u64 {
        self.tokens
    }
    /// All INT32 checkpoints and their owned descriptor/id arrays.
    pub fn checkpoints_bytes(self) -> u64 {
        self.store
    }
    /// Full nonweight i64 cache and its value/borrowed-weight metadata arrays.
    pub fn cache_bytes(self) -> u64 {
        self.cache
    }
    /// Token conversion's simultaneous i64 input and owned u32 copy.
    pub fn token_peak_bytes(self) -> u64 {
        self.token_peak
    }
    /// Checkpoint construction beside the owned canonical tokens.
    pub fn checkpoint_peak_bytes(self) -> u64 {
        self.checkpoint_peak
    }
    /// Cache construction beside tokens and the complete checkpoint store.
    pub fn cache_peak_bytes(self) -> u64 {
        self.cache_peak
    }
    /// Retained tokens, checkpoints and complete nonweight cache together.
    pub fn retained_bytes(self) -> u64 {
        self.retained
    }
    /// Maximum requested owned buffers across all three construction phases.
    pub fn peak_bytes(self) -> u64 {
        self.token_peak
            .max(self.checkpoint_peak)
            .max(self.cache_peak)
    }
}

// Total Vec allocation requests under the pinned Rust growth policy. Count the
// maximum length reached before any deduplication, and include explicit capacity.
pub(crate) fn growing<T>(count: u64, tensor: TensorId) -> Result<u64> {
    if count == 0 || size_of::<T>() == 0 {
        return Ok(0);
    }
    let minimum = if size_of::<T>() == 1 {
        8
    } else if size_of::<T>() <= 1024 {
        4
    } else {
        1
    };
    array::<T>(mul(4, count.max(minimum), tensor)?, tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_array_guards_fail_before_allocation() {
        let tensor = TensorId(0);
        // Wrapping the last product to0 would pass the address guard on64bit.
        for (count, wide) in [
            ((isize::MAX as u64) + 1, false),
            (u64::MAX, true),
            (1u64 << 61, true),
        ] {
            let mut result = None;
            let allocations = allocation_counter::measure(|| {
                result = Some(if wide {
                    array::<u64>(count, tensor)
                } else {
                    array::<u8>(count, tensor)
                });
            });
            assert_eq!(allocations.count_total, 0);
            assert_eq!(result.unwrap(), Err(WitnessError::Memory { tensor }));
        }
    }
}

/// Requested heap buffers owned by the engine's sequential table loader.
/// Input specifications are borrowed; loaded tables own their cloned class vectors,
/// dense lookup cores or RoPE frequencies. Registry gaps and duplicate specifications
/// are charged. Graph validation, thread-runtime/stack storage and allocator overhead
/// remain separate. This is an upper bound, not table/digest authentication or CPU admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableMemory {
    retained: u64,
    peak: u64,
}

impl TableMemory {
    pub(crate) fn new(tables: &[tessera_tables::TableSpec], registry_entry: u64) -> Result<Self> {
        use tessera_tables::{GeneratedTable, TableKind};
        let context = tables
            .iter()
            .max_by_key(|t| t.id.0)
            .map_or(tessera_ir::TableId(0), |t| t.id);
        let fail = || WitnessError::MemoryTable { table: context };
        let slots = tables
            .iter()
            .map(|t| u64::from(t.id.0) + 1)
            .max()
            .unwrap_or(0);
        let registry = slots.checked_mul(registry_entry).ok_or_else(fail)?;
        if usize::try_from(slots).is_err() || registry > isize::MAX as u64 {
            return Err(fail());
        }
        let mut retained = registry;
        let mut digest_scratch = 0;
        for table in tables {
            let (values, width, classes) = match &table.kind {
                // RopeConstants keeps one u128 frequency per pair of head dimensions,
                // not max_context*d_head/2 cosine/sine pairs (RFC-0008).
                TableKind::Rope { d_head, .. } => (u64::from(d_head / 2), 16, 0),
                kind => {
                    let values = GeneratedTable::core_storage_len(kind).map_err(|error| {
                        WitnessError::Table {
                            table: table.id,
                            error,
                        }
                    })?;
                    let classes = match kind {
                        TableKind::Requant { r_by_class, .. }
                        | TableKind::Silu { r_by_class, .. } => r_by_class.len() as u64,
                        _ => 0,
                    };
                    if matches!(kind, TableKind::Exp { .. }) {
                        digest_scratch = 8; // one u64 zero-tail piece beside all completed cores.
                    }
                    (values, 8, classes)
                }
            };
            let bytes = values.checked_mul(width).ok_or_else(fail)?;
            if usize::try_from(values).is_err() || bytes > isize::MAX as u64 {
                return Err(fail());
            }
            retained = retained
                .checked_add(bytes)
                .and_then(|n| n.checked_add(classes))
                .ok_or_else(fail)?;
        }
        // Sum every supplied payload, including duplicates: replacement constructs
        // the new table before dropping the previous one. Digest work allocates no
        // other buffers; the incremental hasher and fixed-point generators are inline.
        let peak = retained.checked_add(digest_scratch).ok_or_else(fail)?;
        Ok(Self { retained, peak })
    }

    /// Registry descriptors plus all supplied owned table payloads, including duplicates.
    pub fn retained_bytes(self) -> u64 {
        self.retained
    }

    /// Conservative simultaneous requested buffers during sequential table loading.
    pub fn peak_bytes(self) -> u64 {
        self.peak
    }
}

#[cfg(test)]
mod table_memory_tests {
    use super::*;
    use tessera_tables::{TableId, TableKind, TableSpec};

    #[test]
    fn registry_address_and_product_overflow_refuse_without_allocating() {
        for (id, entry) in [(0, isize::MAX as u64 + 1), (3, 1u64 << 62)] {
            let tables = [TableSpec {
                id: TableId(id),
                kind: TableKind::Range { bits: 1 },
                log2_rows: 1,
                digest: [0; 32],
            }];
            let measured = allocation_counter::measure(|| {
                assert_eq!(
                    TableMemory::new(&tables, entry),
                    Err(WitnessError::MemoryTable { table: TableId(id) })
                )
            });
            assert_eq!(measured.count_total, 0);
        }
    }
}
