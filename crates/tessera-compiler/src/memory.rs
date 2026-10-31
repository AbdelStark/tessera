// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Retained public compiler containers (RFC-0018).
use crate::Lowering;
use tessera_tables::{TableKind, TableSpec};

pub(crate) fn vector<T>(v: &Vec<T>) -> Option<u64> {
    (v.capacity() as u64).checked_mul(size_of::<T>() as u64)
}
pub(crate) fn tables(v: &Vec<TableSpec>) -> Option<u64> {
    let mut bytes = vector(v)?;
    for TableSpec {
        id: _,
        kind,
        log2_rows: _,
        digest: _,
    } in v
    {
        bytes = bytes.checked_add(match kind {
            TableKind::Requant {
                b: _,
                r_by_class,
                lo: _,
                hi: _,
            }
            | TableKind::Silu {
                b: _,
                r_by_class,
                in_exp: _,
                out_exp: _,
            } => vector(r_by_class)?,
            TableKind::Range { bits: _ }
            | TableKind::Pow2 { max_exp: _ }
            | TableKind::Bitwise8 { op: _ }
            | TableKind::AesSbox
            | TableKind::Gf256Mul { c: _ }
            | TableKind::Exp {
                key_bits: _,
                sigma: _,
                scale: _,
            }
            | TableKind::Exp2Frac {
                frac_bits: _,
                out_bits: _,
            }
            | TableKind::Rope {
                theta: _,
                factor: _,
                low_freq_factor: _,
                high_freq_factor: _,
                original_max: _,
                bits: _,
                max_context: _,
                d_head: _,
            } => 0,
        })?;
    }
    Some(bytes)
}
pub(crate) fn btree<K, V>(n: u64) -> Option<u64> {
    // Rust1.95 alloc/btree/node.rs: B=6, eleven keys/values, twelve edges,
    // parent pointer + two u16s. Seven alignment gaps cover both repr(C) levels.
    // Each occupied node has a key; include transient splits and an empty root.
    let align = align_of::<K>()
        .max(align_of::<V>())
        .max(align_of::<usize>()) as u64;
    let node = (size_of::<usize>() as u64)
        .checked_add(4)?
        .checked_add(
            11u64.checked_mul((size_of::<K>() as u64).checked_add(size_of::<V>() as u64)?)?,
        )?
        .checked_add(12u64.checked_mul(size_of::<usize>() as u64)?)?
        .checked_add(7u64.checked_mul(align.checked_sub(1)?)?)?;
    2u64.checked_mul(n.checked_add(1)?)?.checked_mul(node)
}
impl Lowering {
    /// Requested retained heap bytes of the graph, table metadata and bindings
    /// (RFC-0006/0018), including nested spare capacities. Does not allocate or
    /// validate. Inline storage, construction/analysis scratch, payload values and
    /// allocator overhead are separate. Returns `None` on checked overflow.
    pub fn heap_bytes(&self) -> Option<u64> {
        let Self {
            graph,
            tables: specs,
            tokens: _,
            positions: _,
            one_hot: _,
            logits: _,
            checkpoints,
            sites,
            classes,
            multipliers,
        } = self;
        graph
            .heap_bytes()?
            .checked_add(tables(specs)?)?
            .checked_add(vector(checkpoints)?)?
            .checked_add(vector(sites)?)?
            .checked_add(vector(classes)?)?
            .checked_add(vector(multipliers)?)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn typed_selection_and_advice_maps_fit_pinned_node_request_bounds() {
        use crate::AdviceSource;
        use std::collections::BTreeMap;
        use tessera_ir::{NodeId, TensorId};
        use tessera_tables::TableId;
        for n in [0usize, 1, 2, 5, 6, 11, 12, 13, 31, 64, 129, 513, 4097] {
            let mut reads = BTreeMap::<NodeId, Vec<TableId>>::new();
            let observed = allocation_counter::measure(|| {
                for i in 0..n {
                    reads.insert(NodeId(i as u32), Vec::new());
                }
            });
            assert!(
                observed.bytes_total <= btree::<NodeId, Vec<TableId>>(n as u64).unwrap(),
                "{n}: {observed:?}"
            );
            let mut bindings = BTreeMap::<AdviceSource, (usize, usize)>::new();
            let observed = allocation_counter::measure(|| {
                for i in 0..n {
                    bindings.insert(AdviceSource::Tensor(TensorId(i as u32)), (i, i));
                }
            });
            assert!(
                observed.bytes_total <= btree::<AdviceSource, (usize, usize)>(n as u64).unwrap(),
                "{n}: {observed:?}"
            );
        }
    }

    #[test]
    fn unaddressable_tree_node_sums_fail_closed() {
        assert_eq!(btree::<u64, u64>(u64::MAX), None);
        assert_eq!(btree::<u64, Vec<u64>>(u64::MAX / 2), None);
    }
}
