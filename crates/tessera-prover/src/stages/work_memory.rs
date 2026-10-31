// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Stage-driver bookkeeping, evaluation and output work (RFC-0018/0037).
use super::{
    Coord, Factor, Goldilocks, GoldilocksExt2 as F, Layers, NodeId, NodeRef, Op, StageError,
    StageProver, WitnessSource,
};
use tessera_compiler::schedule::{Kind, Relation};
use tessera_sumcheck::{Claim as EvaluationClaim, Instance};
type SharingKey = (NodeRef, Kind, Vec<Coord>, Layers, u32, Vec<(Kind, Layers)>);

/// Canonical driver work beside state/source/relation/engine reservations.
/// Borrowed inputs, pending windows, transcript/codec callbacks and runtime/
/// allocator overhead remain separate. No route or input is authenticated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageWorkMemory {
    bookkeeping: u64,
    evaluation: u64,
    outputs: u64,
    padding: u64,
}
impl StageWorkMemory {
    /// Ordinary driver containers and worst-case public sharing-map ownership.
    pub fn bookkeeping_bytes(self) -> u64 {
        self.bookkeeping
    }
    /// Transient canonical evaluation/claim/native-callback storage.
    pub fn evaluation_bytes(self) -> u64 {
        self.evaluation
    }
    /// Current-stage splits, incoming claims and leaf value/flattening output.
    pub fn output_bytes(self) -> u64 {
        self.outputs
    }
    /// Leaf padding-reachability and conservative canonical table work.
    pub fn padding_bytes(self) -> u64 {
        self.padding
    }
    /// Conservative overlap of these driver-owned categories only.
    ///
    /// # Errors
    /// Byte-sum overflow, checked again defensively.
    pub fn peak_bytes(self) -> Result<u64, StageError> {
        add(
            add(self.bookkeeping, self.evaluation)?,
            add(self.outputs, self.padding)?,
        )
    }
}
impl<W: WitnessSource + ?Sized> StageProver<'_, W> {
    /// Plan driver work for a compiled stage without allocating or reading field
    /// values. Native claim callbacks are the canonical ModelNativeProver's;
    /// arbitrary callbacks and their heap require an additional reservation.
    /// Persistent state, source arenas/plans, relation tables/instances, engine
    /// work, pending next-stage windows and transcript encodings remain separate.
    ///
    /// # Errors
    /// Invalid public references or overflowing/unaddressable geometry.
    pub fn work_memory(&self, k: u16) -> Result<StageWorkMemory, StageError> {
        let stage = self
            .schedule
            .stages
            .get(usize::from(k))
            .ok_or(StageError::Internal {
                rule: "driver memory names a compiled stage",
            })?;
        let mut bits = 0u64;
        let mut public_cells = 0u64;
        for tensor in &self.graph.tensors {
            let vars = tensor
                .shape
                .iter()
                .try_fold(0u64, |n, &d| add(n, axis(d)?))?;
            bits = bits.max(vars);
            if self.public.contains(tensor.id) {
                public_cells = public_cells.max(power(vars)?);
            }
        }
        // Public ConstAdd per-channel evaluation also materializes a padded table.
        for node in &self.graph.nodes {
            if matches!(node.op, Op::ConstAdd { .. })
                && let Some(tensor) = node
                    .outputs
                    .first()
                    .and_then(|id| self.graph.tensors.get(id.0 as usize))
                && let Some(&n) = tensor.shape.last()
            {
                public_cells = public_cells.max(power(axis(n)?)?);
            }
        }
        let mut point = bits;
        for s in &self.schedule.stages {
            point = point.max(u64::from(s.rounds)).max(u64::from(s.zero_vars));
            for i in &s.instances {
                point = point.max(add(i.point.len() as u64, layer_points(&i.layers))?);
            }
            for c in &s.claims {
                point = point.max(add(c.point.len() as u64, layer_points(&c.layers))?);
                for a in &c.atoms {
                    point = point
                        .max(add(a.point.len() as u64, layer_points(&a.layers))?)
                        .max(coefficient_points(&a.coefficient));
                    for member in &a.combined {
                        point = point.max(coefficient_points(&member.coefficient));
                    }
                }
                for term in &c.public {
                    point = point
                        .max(term.point.len() as u64)
                        .max(coefficient_points(&term.coefficient));
                }
            }
        }
        point = add(point, u64::from(self.schedule.layers.vars))?;
        let (mut graphs, mut keys, mut max_claims) = (0u64, 0u64, 0u64);
        for i in &stage.instances {
            max_claims = max_claims.max(i.claims.len() as u64);
            if matches!(i.relation, Relation::Native { .. }) {
                continue;
            }
            graphs = add(graphs, 1)?;
            let claims = stage
                .claims
                .get(i.claims.clone())
                .ok_or(StageError::Internal {
                    rule: "driver sharing key has compiled claims",
                })?;
            keys = add(
                keys,
                add(
                    array::<Coord>(i.point.len() as u64)?,
                    layer_heap(&i.layers)?,
                )?,
            )?;
            keys = add(keys, array::<(Kind, Layers)>(claims.len() as u64)?)?;
            for c in claims {
                keys = add(keys, layer_heap(&c.layers)?)?;
            }
        }
        let instances = stage.instances.len() as u64;
        let bookkeeping = add(
            add(keys, map::<SharingKey, usize>(graphs)?)?,
            add(
                array::<super::relations::Built<'static>>(graphs)?,
                add(
                    array::<Instance<'static, F>>(instances)?,
                    growing::<Option<usize>>(instances)?,
                )?,
            )?,
        )?;
        let evaluation_tables = evaluation_arrays(bits, public_cells)?;
        // Returned claim points coexist with their outer point/Claim arrays, then
        // one expected point and one layer/coefficient/stack point temporary.
        let claims = add(
            add(
                growing::<EvaluationClaim<F>>(max_claims)?,
                growing::<Vec<F>>(max_claims)?,
            )?,
            add(
                mul(max_claims, growing::<F>(point)?)?,
                growing::<usize>(max_claims)?,
            )?,
        )?;
        let points = mul(2, growing::<F>(point)?)?;
        let layers = u64::from(self.schedule.layers.count).max(1);
        let layer_work = add(
            add(
                growing::<(u32, F)>(layers)?,
                growing::<Option<u32>>(layers)?,
            )?,
            growing::<Coord>(point)?,
        )?;
        let native = self
            .graph
            .slots
            .as_ref()
            .map_or(Ok(0), |s| growing::<u64>(s.lengths.len() as u64))?;
        let label = growing::<u8>(("stage/".len() + 5 + "/layer/".len() + 20) as u64)?;
        let evaluation = add(
            evaluation_tables,
            add(claims, add(points, add(layer_work, add(native, label)?)?)?)?,
        )?;
        let (mut split, mut leaf_values, mut leaf_nested) = (0u64, 0u64, 0u64);
        for c in &stage.claims {
            let count = if c.layers == Layers::Each {
                u64::from(self.schedule.layers.count)
            } else {
                1
            };
            if c.split {
                split = add(split, mul(c.parts.len() as u64, count)?)?;
            }
            if k == 0 {
                leaf_values = add(leaf_values, count)?;
                leaf_nested = add(leaf_nested, growing::<F>(count)?)?;
            }
        }
        let mut outputs = add(growing::<F>(split)?, growing::<F>(instances)?)?;
        if k == 0 {
            outputs = add(
                outputs,
                add(
                    array::<Vec<F>>(stage.claims.len() as u64)?,
                    add(leaf_nested, growing::<F>(leaf_values)?)?,
                )?,
            )?;
        }
        let padding = if k == 0 {
            let n = self.graph.tensors.len() as u64;
            let arrays = add(
                array::<Option<(NodeId, u32)>>(n)?,
                mul(2, array::<bool>(n)?)?,
            )?;
            let reach = add(
                arrays,
                map::<tessera_tables::TableId, bool>(self.graph.nodes.len() as u64)?,
            )?;
            // reach's only generated zero-value tables are a subset of this
            // established complete table-constructor bound, including class clones.
            let tables = tessera_witness::Engine::table_memory(self.tables)
                .map_err(|_| StageError::Memory)?
                .peak_bytes();
            add(reach, tables)?
        } else {
            0
        };
        let out = StageWorkMemory {
            bookkeeping,
            evaluation,
            outputs,
            padding,
        };
        out.peak_bytes()?;
        Ok(out)
    }
}
fn evaluation_arrays(bits: u64, public_cells: u64) -> Result<u64, StageError> {
    let split_eq = add(power(bits / 2)?, power(bits - bits / 2)?)?;
    let low = bits.min(10);
    let blocked = add(power(low)?, power(bits - low)?)?;
    add(
        array::<F>(split_eq.max(blocked))?,
        array::<Goldilocks>(public_cells)?,
    )
}
fn layer_points(layers: &Layers) -> u64 {
    match layers {
        Layers::Eq(p) => p.len() as u64,
        _ => 0,
    }
}
fn layer_heap(layers: &Layers) -> Result<u64, StageError> {
    array::<Coord>(layer_points(layers))
}
fn coefficient_points(products: &[Vec<Factor>]) -> u64 {
    products
        .iter()
        .flatten()
        .filter_map(|f| match f {
            Factor::Lt { coords, .. } => Some(coords.len() as u64),
            _ => None,
        })
        .max()
        .unwrap_or(0)
}
fn add(a: u64, b: u64) -> Result<u64, StageError> {
    a.checked_add(b).ok_or(StageError::Memory)
}
fn mul(a: u64, b: u64) -> Result<u64, StageError> {
    a.checked_mul(b).ok_or(StageError::Memory)
}
fn axis(n: u32) -> Result<u64, StageError> {
    n.checked_next_power_of_two()
        .map(|v| u64::from(v.trailing_zeros()))
        .ok_or(StageError::Memory)
}
fn power(bits: u64) -> Result<u64, StageError> {
    if bits >= u64::from(usize::BITS) {
        return Err(StageError::Memory);
    }
    1u64.checked_shl(bits as u32).ok_or(StageError::Memory)
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
fn map<K, V>(n: u64) -> Result<u64, StageError> {
    if n == 0 {
        return Ok(0);
    }
    let buckets = (add(mul(n, 8)?, 6)? / 7)
        .max(4)
        .checked_next_power_of_two()
        .ok_or(StageError::Memory)?;
    let all = mul(2, buckets)?;
    add(
        add(array::<(K, V)>(all)?, array::<u8>(all)?)?,
        mul(31, u64::from(buckets.trailing_zeros()))?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tessera_ir::TensorId;
    #[test]
    fn sharing_and_padding_maps_fit_the_pinned_bucket_request_bound() {
        for n in [0usize, 1, 3, 4, 7, 8, 512, 513, 4097] {
            let keys = (0..n)
                .map(|i| {
                    let kind = Kind::Tensor(TensorId(i as u32));
                    (
                        NodeRef::Node(NodeId(i as u32)),
                        kind,
                        vec![Coord::Half; 3],
                        Layers::Eq(vec![Coord::Half; 2]),
                        1,
                        vec![(kind, Layers::None)],
                    )
                })
                .collect::<Vec<SharingKey>>();
            let measured = allocation_counter::measure(|| {
                let mut map = HashMap::new();
                for key in keys {
                    map.insert(key, 0usize);
                }
                assert_eq!(map.len(), n);
            });
            assert!(measured.bytes_total <= map::<SharingKey, usize>(n as u64).unwrap());
            let measured = allocation_counter::measure(|| {
                let mut map = HashMap::new();
                for i in 0..n {
                    map.insert(tessera_tables::TableId(i as u32), false);
                }
                assert_eq!(map.len(), n);
            });
            assert!(
                measured.bytes_total <= map::<tessera_tables::TableId, bool>(n as u64).unwrap()
            );
        }
    }
    #[test]
    fn driver_byte_and_axis_limits_fail_without_allocating() {
        let measured = allocation_counter::measure(|| {
            assert_eq!(
                array::<u64>(isize::MAX as u64 / 8 + 1),
                Err(StageError::Memory)
            );
            assert_eq!(add(u64::MAX, 1), Err(StageError::Memory));
            assert_eq!(mul(u64::MAX, 2), Err(StageError::Memory));
            assert_eq!(power(u64::from(usize::BITS)), Err(StageError::Memory));
            assert_eq!(power(1u64 << 32), Err(StageError::Memory));
            assert_eq!(growing::<u8>(u64::MAX / 4 + 1), Err(StageError::Memory));
            assert_eq!(axis(u32::MAX), Err(StageError::Memory));
        });
        assert_eq!(measured.count_total, 0);
    }
    #[test]
    fn tensor_split_and_public_blocked_evaluation_requests_fit_independently() {
        use tessera_field::Field;
        struct Source(Vec<i64>);
        impl WitnessSource for Source {
            fn values(&self, _: TensorId) -> Option<&[i64]> {
                Some(&self.0)
            }
        }
        for vars in [0u32, 1, 9, 10, 11, 20] {
            let n = 1usize << vars;
            let source = Source(vec![1; n]);
            let point = vec![F::ONE; vars as usize];
            let shape = [n as u32];
            let reader =
                super::super::tables::TensorReader::new(&source, TensorId(0), &shape).unwrap();
            let measured = allocation_counter::measure(|| {
                reader.extension(&point).unwrap();
            });
            assert!(measured.bytes_max <= evaluation_arrays(u64::from(vars), 0).unwrap());
            let measured = allocation_counter::measure(|| {
                let table = vec![Goldilocks::ONE; n];
                tessera_poly::evaluate_base::<F>(&table, &point).unwrap();
            });
            assert!(measured.bytes_max <= evaluation_arrays(u64::from(vars), n as u64).unwrap());
        }
    }
}
