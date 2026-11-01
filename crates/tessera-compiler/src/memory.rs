// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Retained public compiler containers (RFC-0018).
use crate::Lowering;
use tessera_tables::{TableKind, TableSpec};

pub(crate) fn array<T>(n: u64) -> Option<u64> {
    let bytes = n.checked_mul(size_of::<T>() as u64)?;
    (usize::try_from(n).is_ok() && bytes <= isize::MAX as u64).then_some(bytes)
}
// Pinned Rust1.95 geometric growth, including old/new allocation overlap.
pub(crate) fn grow<T>(n: u64) -> Option<u64> {
    if n == 0 {
        return Some(0);
    }
    let minimum = if size_of::<T>() == 1 {
        8
    } else if size_of::<T>() <= 1024 {
        4
    } else {
        1
    };
    array::<T>(4u64.checked_mul(n.max(minimum))?)
}
pub(crate) fn sum(values: impl IntoIterator<Item = u64>) -> Option<u64> {
    values.into_iter().try_fold(0u64, u64::checked_add)
}

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
pub(crate) fn map<K, V>(n: u64) -> Option<u64> {
    if n == 0 {
        return Some(0);
    }
    // Rust1.95 hashbrown: <=7/8 load; sum geometric bucket growth and control bytes.
    let buckets = (n.checked_mul(8)?.checked_add(6)? / 7)
        .max(4)
        .checked_next_power_of_two()?;
    let all = buckets.checked_mul(2)?;
    sum([
        array::<(K, V)>(all)?,
        array::<u8>(all)?,
        31u64.checked_mul(u64::from(buckets.trailing_zeros()))?,
    ])
}

pub(crate) fn claim_clone_bytes(claims: &tessera_ir::ClaimList) -> Option<u64> {
    let tessera_ir::ClaimList {
        protocol_version: _,
        arch: _,
        profile: _,
        quant,
        silu_in_exponents,
        tables: specs,
        gadget_policy: _,
        layout: _,
        params,
    } = claims;
    let tessera_ir::PublicQuantSpec {
        act_exponents,
        residual_exponent: _,
        residual_bits: _,
        rsqrt_kappa: _,
        rms_eps_int: _,
        exp_sigma: _,
        score_scales,
        shift_class_sets,
        affine_sites,
        rope: _,
        logits_bits: _,
        logits_exponent: _,
    } = quant;
    let mut bytes = sum([
        vector(act_exponents)?,
        vector(score_scales)?,
        vector(shift_class_sets)?,
        vector(affine_sites)?,
        vector(silu_in_exponents)?,
        tables(specs)?,
        params.0.as_str().len() as u64,
    ])?;
    for (_, shifts) in shift_class_sets {
        bytes = bytes.checked_add(vector(shifts)?)?;
    }
    Some(bytes)
}

impl Lowering {
    /// Conservative metadata peak while lowering the given public model
    /// (RFC-0006/0018), computed from this canonical lowering's geometry.
    /// Includes graph-vector growth, validation, range analysis, settlement and
    /// channel-binding preparation. Generated table cores, borrowed inputs and
    /// allocator overhead must be reserved separately. Does not authenticate the
    /// pairing of model and lowering or allocate. Returns `None` on overflow.
    pub fn construction_metadata_bytes(&self, model: crate::PublicModel<'_>) -> Option<u64> {
        use tessera_ir::{QuantPoint, TensorDecl, TensorId, WeightRef};
        let nw = u64::from(model.arch.n_layers)
            .checked_mul(7)?
            .checked_add(2)?;
        let sites = (model.quant.shift_class_sets.len() as u64)
            .checked_add(model.quant.affine_sites.len() as u64)?;
        let classed = u64::from(model.arch.n_layers)
            .checked_mul(crate::lower::sites(model.profile).len() as u64)?;
        let outputs = self
            .graph
            .nodes
            .iter()
            .try_fold(0u64, |n, node| n.checked_add(node.outputs.len() as u64))?;
        let arity = self
            .graph
            .nodes
            .iter()
            .map(|n| n.outputs.len() as u64)
            .max()
            .unwrap_or(0);
        let rank = self
            .graph
            .tensors
            .iter()
            .map(|t| t.shape.len() as u64)
            .max()
            .unwrap_or(0);
        sum([
            // One growing container may retain its old allocation until moved.
            self.heap_bytes()?.checked_mul(2)?,
            crate::analysis_memory_bytes(&self.graph)?,
            grow::<TensorId>(outputs)?,
            array::<QuantPoint>(sites)?.checked_mul(2)?,
            grow::<(WeightRef, u32, u32)>(nw)?,
            // The stable channel-binding sort may allocate a second tuple array.
            grow::<(WeightRef, u32, Option<usize>)>(classed)?.checked_mul(2)?,
            grow::<TensorDecl>(arity)?,
            array::<u32>(rank)?.checked_mul(arity)?,
            grow::<TensorId>(arity)?.checked_mul(2)?,
            // shifts() has at most four entries; caller and table clone can overlap.
            grow::<u8>(4)?.checked_mul(3)?,
        ])
    }

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
    fn cloned_claims_fit_the_public_container_bound() {
        use tessera_codec::{DecodeLimits, envelope, schema};
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../spec/vectors/tamper/claims.tcl");
        let claims: tessera_ir::ClaimList = envelope::open(
            &std::fs::read(path).unwrap(),
            &schema::CLAIM_LIST,
            DecodeLimits::MESSAGE,
        )
        .unwrap();
        let mut owned = None;
        let observed = allocation_counter::measure(|| owned = Some(claims.clone()));
        let mut bound = None;
        let sizing = allocation_counter::measure(|| bound = claim_clone_bytes(&claims));
        assert_eq!(sizing.bytes_total, 0);
        assert!(
            observed.bytes_max <= bound.unwrap(),
            "{observed:?} > {bound:?}"
        );
        drop(owned);
    }

    #[test]
    fn padding_table_map_requests_fit_the_pinned_bucket_bound() {
        use std::collections::HashMap;
        use tessera_ir::TableId;
        for n in [0u64, 1, 2, 4, 5, 13, 129, 513] {
            let actual = allocation_counter::measure(|| {
                let mut values = HashMap::new();
                for i in 0..n {
                    values.insert(TableId(i as u32), i % 2 == 0);
                }
                std::hint::black_box(&values);
            });
            assert!(
                actual.bytes_total <= map::<TableId, bool>(n).unwrap(),
                "{n}: {actual:?}"
            );
        }
        assert_eq!(map::<TableId, bool>(u64::MAX), None);
        assert_eq!(map::<TableId, bool>(1 << 61), None);
    }

    #[test]
    fn unaddressable_preparation_arrays_fail_closed() {
        assert_eq!(array::<u64>(u64::MAX), None);
        assert_eq!(array::<u64>(1 << 61), None);
        assert_eq!(grow::<u64>(1 << 62), None);
        assert_eq!(sum([u64::MAX, 1]), None);
        assert_eq!(grow::<u8>(0), Some(0));
    }

    #[test]
    fn unaddressable_tree_node_sums_fail_closed() {
        assert_eq!(btree::<u64, u64>(u64::MAX), None);
        assert_eq!(btree::<u64, Vec<u64>>(u64::MAX / 2), None);
    }
}
