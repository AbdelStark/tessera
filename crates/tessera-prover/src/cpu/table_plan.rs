// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public P2 table-batch reservation, additional to the unchanged resident-cell floor.

use super::ProverError;
use crate::leaf::{GroupedLayout, LeafLayout, ModelColumn, balanced_j_star};
use tessera_compiler::Selection;
use tessera_logup::TABLE_BATCH_CONSTRUCTION_BYTES_PER_TREE;
use tessera_sumcheck::{MergePolynomial, WindowMemory, WindowShape};

pub(super) const CHUNK_ROWS: usize = 256;
const MAX_CHUNK_ROWS: usize = 65_536;
const MAX_CACHE_LOG_ROWS: usize = 20;

pub(super) fn cache_depth(height: usize, log_rows: Option<usize>) -> Option<usize> {
    // Retain at most the selected public cap and keep at least four original leaf bits.
    height
        .checked_sub(4)
        .map(|depth| depth.min(log_rows.unwrap_or(0)))
        .filter(|&d| d > 0)
}
const NATIVE_DESCRIPTOR_BYTES_PER_TREE: u64 = 4096;
// A collected Result<Vec<_>> may retain four slots for one source. Its descriptors
// and public class/container metadata coexist with the generic batch constructor.
const _: () = assert!(
    4 * size_of::<crate::leaf::table_source::GeneratedCounts>()
        + size_of::<tessera_tables::GeneratedTable>()
        + 4
        + 6 * size_of::<Vec<()>>()
        <= NATIVE_DESCRIPTOR_BYTES_PER_TREE as usize
);

fn memory() -> ProverError {
    ProverError::Resource(tessera_codec::error::ErrorCode::ResMemory)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stress(generated: bool) -> TablePlan {
        TablePlan {
            trees: vec![Shape {
                height: 20,
                trees_log: 0,
                fraction: true,
                generated,
            }],
            fixed: 80 << 20,
            scratch: 4096,
            retained: 0,
            forests: 0,
            working_bytes: 0,
            cuts: Vec::new(),
            chunk_rows: CHUNK_ROWS,
            cache_log_rows: None,
        }
    }

    #[test]
    fn caches_use_only_spare_budget_and_preserve_the_uncached_plan() {
        for generated in [false, true] {
            for allowance in [0, 80 << 20, 84 << 20, 96 << 20, 128 << 20, u64::MAX] {
                let reference = stress(generated).select(allowance);
                let cached = stress(generated).select_cached(allowance);
                assert_eq!(reference.is_ok(), cached.is_ok());
                if let (Ok(reference), Ok(cached)) = (reference, cached) {
                    assert_eq!(cached.cuts, reference.cuts);
                    assert_eq!(cached.chunk_rows, reference.chunk_rows);
                    assert!(cached.working_bytes <= allowance);
                    assert_eq!(
                        cached.working_bytes - reference.working_bytes,
                        cached.cache_bytes().unwrap()
                    );
                    if !generated {
                        assert_eq!(cached.cache_log_rows, None);
                    }
                    assert_eq!(cached.clone().select_cached(allowance).unwrap(), cached);
                    assert_eq!(cached.select(allowance).unwrap(), reference);
                }
            }
        }
        let ample = stress(true).select_cached(u64::MAX).unwrap();
        assert_eq!(ample.cache_log_rows, Some(MAX_CACHE_LOG_ROWS));
        assert_eq!(cache_depth(20, ample.cache_log_rows), Some(16));
        let base = stress(true).select(84 << 20).unwrap();
        let tight = ample.select_cached(base.working_bytes).unwrap();
        assert_eq!(tight.cache_log_rows, None);
        assert_eq!(tight.working_bytes, base.working_bytes);
        assert_eq!(tight.cuts, base.cuts);
        for height in 0..=4 {
            assert_eq!(cache_depth(height, Some(MAX_CACHE_LOG_ROWS)), None);
        }
        let mut multiple = stress(true);
        multiple.trees.push(Shape {
            height: 18,
            trees_log: 0,
            fraction: true,
            generated: true,
        });
        let baseline = multiple.clone().select(u64::MAX).unwrap();
        let cached = multiple.select_cached(u64::MAX).unwrap();
        assert_eq!(cached.cache_bytes().unwrap(), 32 * ((1 << 16) + (1 << 14)));
        assert_eq!(
            cached.working_bytes - baseline.working_bytes,
            cached.cache_bytes().unwrap()
        );
    }

    #[test]
    fn only_generated_tables_stream_to_fit_and_fixed_storage_never_disappears() {
        let allowance = 84 << 20;
        let fitted = stress(true).select(allowance).unwrap();
        assert!(fitted.cuts.iter().any(|&c| c > 0));
        assert!(fitted.working_bytes <= allowance);
        // A resident lookup/forest source is never reclassified as a generated table
        // merely to reduce its sumcheck charge (RFC-0054, g=0).
        assert!(matches!(
            stress(false).select(allowance),
            Err(ProverError::Resource(
                tessera_codec::error::ErrorCode::ResMemory
            ))
        ));
        assert!(matches!(
            stress(true).select(80 << 20),
            Err(ProverError::Resource(
                tessera_codec::error::ErrorCode::ResMemory
            ))
        ));
    }

    #[test]
    fn overflowing_or_insufficient_fixed_reservations_are_typed_memory_refusals() {
        for fixed in [u64::MAX, 80 << 20] {
            let mut plan = stress(true);
            plan.fixed = fixed;
            assert!(matches!(
                plan.select(0),
                Err(ProverError::Resource(
                    tessera_codec::error::ErrorCode::ResMemory
                ))
            ));
        }
        let mut plan = stress(true);
        plan.fixed = u64::MAX;
        assert!(matches!(
            plan.select(u64::MAX),
            Err(ProverError::Resource(
                tessera_codec::error::ErrorCode::ResMemory
            ))
        ));
    }
    #[test]
    fn larger_public_chunks_preserve_reference_cuts_and_feasibility() {
        for generated in [false, true] {
            for allowance in [
                0,
                80 << 20,
                (80 << 20) + 4096,
                84 << 20,
                96 << 20,
                128 << 20,
                u64::MAX,
            ] {
                let reference = stress(generated).select_with_chunk_limit(allowance, CHUNK_ROWS);
                let batched = stress(generated).select(allowance);
                assert_eq!(reference.is_ok(), batched.is_ok());
                if let (Ok(reference), Ok(batched)) = (reference, batched) {
                    assert_eq!(batched.cuts, reference.cuts);
                    assert!(batched.working_bytes <= allowance);
                    assert!(batched.working_bytes >= reference.working_bytes);
                    assert!((CHUNK_ROWS..=MAX_CHUNK_ROWS).contains(&batched.chunk_rows));
                    assert!(batched.chunk_rows.is_power_of_two());
                    if !generated {
                        assert_eq!(batched.chunk_rows, CHUNK_ROWS);
                    }
                }
            }
        }
        let ample = stress(true).select(u64::MAX).unwrap();
        assert_eq!(ample.chunk_rows, MAX_CHUNK_ROWS);
        let reference = stress(true)
            .select_with_chunk_limit(84 << 20, CHUNK_ROWS)
            .unwrap();
        let tight = ample.clone().select(reference.working_bytes).unwrap();
        assert_eq!(tight.chunk_rows, CHUNK_ROWS);
        assert_eq!(tight.cuts, reference.cuts);
        assert_eq!(tight.working_bytes, reference.working_bytes);
        assert_eq!(tight.select(u64::MAX).unwrap(), ample);
        for limit in [0, CHUNK_ROWS - 1, CHUNK_ROWS + 1] {
            assert!(matches!(
                stress(true).select_with_chunk_limit(u64::MAX, limit),
                Err(ProverError::Resource(
                    tessera_codec::error::ErrorCode::ResMemory
                ))
            ));
        }
    }

    #[test]
    fn enlarged_read_scratch_is_charged_and_bounds_actual_allocations() {
        use std::cell::Cell;
        use tessera_codec::{Ascii, SecretBuf};
        use tessera_field::{
            Field, Goldilocks as B, GoldilocksExt2 as F, PrimeField, QuadraticExtension,
        };
        use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
        use tessera_logup::{TableBatchBudget, TableSource, prove_table_batch_bounded};
        use tessera_sumcheck::WindowError;
        struct AllocatingSource {
            rows: usize,
            maximum_read: Cell<usize>,
        }
        impl TableSource<B> for AllocatingSource {
            fn rows(&self) -> usize {
                self.rows
            }
            fn has_values(&self) -> bool {
                false
            }
            fn read_scratch_bytes(&self, rows: usize) -> Option<u64> {
                (rows as u64).checked_mul(8)
            }
            fn read(
                &self,
                first: usize,
                keys: &mut [B],
                values: &mut [B],
                counts: &mut [B],
            ) -> Result<(), WindowError> {
                if keys.len() != values.len()
                    || keys.len() != counts.len()
                    || first
                        .checked_add(keys.len())
                        .is_none_or(|end| end > self.rows)
                {
                    return Err(WindowError::Source);
                }
                self.maximum_read
                    .set(self.maximum_read.get().max(keys.len()));
                let mut scratch = SecretBuf::try_with_capacity(keys.len())
                    .map_err(|_| WindowError::Allocation)?;
                scratch.resize(keys.len(), 0u64);
                for i in 0..keys.len() {
                    keys[i] = B::from_u64_reduced((first + i) as u64);
                    values[i] = B::ZERO;
                    counts[i] = B::from_u64_reduced(scratch[i]);
                }
                Ok(())
            }
        }
        for height in [12usize, 16] {
            let rows = 1usize << height;
            let source = AllocatingSource {
                rows,
                maximum_read: Cell::new(0),
            };
            let owned = 16 * rows as u64 + NATIVE_DESCRIPTOR_BYTES_PER_TREE;
            let plan = TablePlan {
                trees: vec![Shape {
                    height,
                    trees_log: 0,
                    fraction: true,
                    generated: true,
                }],
                fixed: owned + 65_536 + ((height + 1) * (height + 8) * 384) as u64,
                scratch: 128 * CHUNK_ROWS as u64 + 32 * (height + 1) as u64 + 512,
                retained: 0,
                forests: 0,
                working_bytes: 0,
                cuts: Vec::new(),
                chunk_rows: CHUNK_ROWS,
                cache_log_rows: None,
            }
            .select(u64::MAX)
            .unwrap();
            assert_eq!(plan.chunk_rows, rows.min(MAX_CHUNK_ROWS));
            let mut transcript = Transcript::new(&SessionContext {
                protocol: Ascii::new("tessera-protocol/1").unwrap(),
                mode: Mode::NonInteractive,
                params: ParamSetId(Ascii::new("test-small/1").unwrap()),
                statement_digest: [59; 32],
                session_id: [71; 16],
            });
            let mut proof = None;
            let measured = allocation_counter::measure(|| {
                proof = Some(
                    prove_table_batch_bounded(
                        &mut transcript,
                        F::new(B::ONE, B::ONE),
                        F::ZERO,
                        &[&source as &dyn TableSource<B>],
                        TableBatchBudget {
                            allowance: plan.working_bytes,
                            source_bytes: owned,
                            chunk_rows: core::num::NonZeroUsize::new(plan.chunk_rows).unwrap(),
                        },
                    )
                    .unwrap(),
                );
            });
            assert_eq!(source.maximum_read.get(), plan.chunk_rows);
            assert!(
                measured.bytes_max <= plan.working_bytes,
                "actual allocated peak {} exceeds charged {}",
                measured.bytes_max,
                plan.working_bytes
            );
            assert_eq!(proof.unwrap().multiplicities().len(), 1);
        }
    }

    #[test]
    fn constructor_metadata_is_reserved_beside_native_source_storage() {
        use core::mem::size_of;
        use tessera_codec::Ascii;
        use tessera_field::{Field, Goldilocks as B, GoldilocksExt2 as F};
        use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
        use tessera_logup::{TableBatchBudget, TableSource, prove_table_batch_bounded};
        use tessera_sumcheck::WindowError;
        struct Singleton;
        impl TableSource<B> for Singleton {
            fn rows(&self) -> usize {
                1
            }
            fn has_values(&self) -> bool {
                true
            }
            fn read_scratch_bytes(&self, rows: usize) -> Option<u64> {
                (rows as u64).checked_mul(8)
            }
            fn read(
                &self,
                first: usize,
                k: &mut [B],
                v: &mut [B],
                m: &mut [B],
            ) -> Result<(), WindowError> {
                if k.len() != v.len()
                    || k.len() != m.len()
                    || first.checked_add(k.len()).is_none_or(|end| end > 1)
                {
                    return Err(WindowError::Source);
                }
                k.fill(B::ZERO);
                v.fill(B::ZERO);
                m.fill(B::ONE);
                Ok(())
            }
        }
        // Source descriptor growth and wrappers coexist with the generic constructor.
        // Singleton/repeated sources exercise the case with no level working-set slack.
        let native = (4 * size_of::<crate::leaf::table_source::GeneratedCounts>()
            + size_of::<tessera_tables::GeneratedTable>()
            + 4
            + 6 * size_of::<Vec<()>>()) as u64;
        for count in [1usize, 64] {
            let n = count as u64;
            let plan = TablePlan {
                trees: vec![
                    Shape {
                        height: 0,
                        trees_log: 0,
                        fraction: true,
                        generated: true
                    };
                    count
                ],
                fixed: 16 * n + 65_536 + n * 8 * 384 + n * 4096,
                scratch: 128 + 32 + 512,
                retained: 0,
                forests: 0,
                working_bytes: 0,
                cuts: Vec::new(),
                chunk_rows: CHUNK_ROWS,
                cache_log_rows: None,
            }
            .select(u64::MAX)
            .unwrap();
            assert_eq!(plan.clone().select(plan.working_bytes).unwrap(), plan);
            let source = Singleton;
            let tables: Vec<&dyn TableSource<B>> = vec![&source; count];
            let mut transcript = Transcript::new(&SessionContext {
                protocol: Ascii::new("tessera-protocol/1").unwrap(),
                mode: Mode::NonInteractive,
                params: ParamSetId(Ascii::new("test-small/1").unwrap()),
                statement_digest: [51; 32],
                session_id: [63; 16],
            });
            let proof = prove_table_batch_bounded(
                &mut transcript,
                F::ONE,
                F::ONE,
                &tables,
                TableBatchBudget {
                    allowance: plan.working_bytes,
                    source_bytes: n * (16 + native),
                    chunk_rows: core::num::NonZeroUsize::new(plan.chunk_rows).unwrap(),
                },
            )
            .unwrap();
            assert_eq!(proof.multiplicities().len(), count);
        }
    }

    #[test]
    fn physical_count_sources_replace_only_the_logical_table_reservation() {
        use crate::leaf::{Site, SiteGadget, SiteLookup};
        use tessera_compiler::schedule::{LayerMap, NodeRef};
        use tessera_field::{Field, Goldilocks as B};
        use tessera_gadgets::{Affine, Lookup};
        use tessera_ir::{GadgetPolicy, Graph, NodeId};
        use tessera_tables::{GeneratedTable, Table, TableId, TableKind};
        use tessera_witness::MultiplicityPlan;
        let table = GeneratedTable::new(TableId(0), TableKind::Range { bits: 29 }).unwrap();
        let graph = Graph::default();
        let specs = vec![table.spec().clone()];
        let ranges = tessera_compiler::analyze(&graph, &specs).unwrap();
        let selection =
            tessera_compiler::select(&graph, &ranges, &specs, &GadgetPolicy::default()).unwrap();
        let leaf = LeafLayout::new(
            &LayerMap::none(),
            vec![Site {
                node: NodeRef::Node(NodeId(0)),
                layers: vec![SiteGadget {
                    lookups: vec![SiteLookup {
                        lookup: Lookup {
                            table: TableId(0),
                            key: Affine {
                                constant: B::ZERO,
                                terms: vec![(B::ONE, ModelColumn::Input(0))],
                            },
                            value: None,
                        },
                        domain: vec![4],
                        axes: Vec::new(),
                        token: Some(0),
                    }],
                }],
            }],
        )
        .unwrap();
        // Four public contributions plus the canonical weighted row-zero tail.
        // Neither the backing choice nor its size observes a lookup value.
        let plans = [MultiplicityPlan::new(TableId(0), 1 << 29, 5).unwrap()];
        assert!(!plans[0].is_dense());
        let dense = TablePlan::new(&leaf, &selection, None, None).unwrap();
        let mut physical = None;
        let measured = allocation_counter::measure(|| {
            physical = Some(TablePlan::new(&leaf, &selection, None, Some(&plans)).unwrap());
        });
        let physical = physical.unwrap();
        let mut metadata = None;
        let sizing = allocation_counter::measure(|| {
            metadata = Some(physical.construction_peak_bytes(&leaf, None).unwrap())
        });
        assert_eq!(sizing.bytes_total, 0);
        assert!(
            measured.bytes_max <= metadata.unwrap(),
            "{measured:?} > {metadata:?}"
        );
        let mut selected = None;
        let selection_allocations = allocation_counter::measure(|| {
            selected = Some(physical.clone().select(64 << 20).unwrap())
        });
        assert!(selection_allocations.bytes_max <= metadata.unwrap());
        let selected = selected.unwrap();
        assert_eq!(
            u64::try_from(selection_allocations.bytes_current).unwrap(),
            selected.metadata_bytes().unwrap()
        );
        drop(selected);
        assert_eq!(dense.trees, physical.trees);
        assert_eq!(dense.scratch, physical.scratch);
        assert_eq!(dense.forests, physical.forests);
        assert_eq!(dense.retained, physical.retained);
        let source = tessera_witness::Engine::table_memory(&selection.tables)
            .unwrap()
            .retained_bytes()
            + plans[0].payload_bytes();
        assert_eq!(dense.fixed - physical.fixed, 16 * (1u64 << 29) - source);
        let allowance = 64 << 20;
        assert!(matches!(
            dense.select(allowance),
            Err(ProverError::Resource(
                tessera_codec::error::ErrorCode::ResMemory
            ))
        ));
        let fixed = physical.fixed;
        let fitted = physical.clone().select(allowance).unwrap();
        assert!(fitted.working_bytes <= allowance);
        assert!(fitted.cuts.iter().any(|&c| c > 0));
        assert!(matches!(
            physical.select(fixed - 1),
            Err(ProverError::Resource(
                tessera_codec::error::ErrorCode::ResMemory
            ))
        ));
        for invalid in [
            vec![],
            vec![MultiplicityPlan::new(TableId(1), 1 << 29, 5).unwrap()],
            vec![MultiplicityPlan::new(TableId(0), 1 << 28, 5).unwrap()],
            vec![MultiplicityPlan::new(TableId(0), 1 << 29, 0).unwrap()],
        ] {
            assert!(matches!(
                TablePlan::new(&leaf, &selection, None, Some(&invalid)),
                Err(ProverError::Resource(
                    tessera_codec::error::ErrorCode::ResMemory
                ))
            ));
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Shape {
    height: usize,
    trees_log: usize,
    fraction: bool,
    generated: bool,
}

/// Geometry and conservative physical-capacity bounds, with no witness access.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct TablePlan {
    trees: Vec<Shape>,
    fixed: u64,
    scratch: u64,
    /// Group proofs/merge tables retained beside the table batch.
    pub(super) retained: u64,
    /// Original forest-leaf storage, lent to the P2 batch.
    pub(super) forests: u64,
    pub(super) working_bytes: u64,
    pub(super) cuts: Vec<usize>,
    pub(super) chunk_rows: usize,
    pub(super) cache_log_rows: Option<usize>,
}

fn add(a: u64, b: u64) -> Result<u64, ProverError> {
    a.checked_add(b).ok_or_else(memory)
}
fn mul(a: u64, b: u64) -> Result<u64, ProverError> {
    a.checked_mul(b).ok_or_else(memory)
}
fn power(n: usize) -> Result<u64, ProverError> {
    u32::try_from(n)
        .ok()
        .and_then(|n| 1u64.checked_shl(n))
        .ok_or_else(memory)
}
fn vars(dims: &[u32]) -> Result<usize, ProverError> {
    dims.iter().try_fold(0usize, |n, &d| {
        let v = d
            .checked_next_power_of_two()
            .ok_or_else(memory)?
            .trailing_zeros() as usize;
        n.checked_add(v).ok_or_else(memory)
    })
}

fn array<T>(n: u64) -> Result<u64, ProverError> {
    let bytes = mul(n, size_of::<T>() as u64)?;
    if usize::try_from(n).is_err() || bytes > isize::MAX as u64 {
        return Err(memory());
    }
    Ok(bytes)
}
fn grow<T>(n: u64) -> Result<u64, ProverError> {
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

impl TablePlan {
    pub(super) fn metadata_bytes(&self) -> Result<u64, ProverError> {
        add(
            array::<Shape>(self.trees.capacity() as u64)?,
            array::<usize>(self.cuts.capacity() as u64)?,
        )
    }

    // Public planning allocations only. P2 payloads, cores and source backing are
    // already separate fields and must never be replaced by this metadata bill.
    pub(super) fn construction_peak_bytes(
        &self,
        leaf: &LeafLayout<ModelColumn>,
        grouped: Option<&GroupedLayout>,
    ) -> Result<u64, ProverError> {
        let trees = self.trees.len() as u64;
        let rank = leaf
            .sites()
            .iter()
            .flat_map(|s| &s.layers)
            .flat_map(|g| &g.lookups)
            .map(|l| l.domain.len() as u64)
            .max()
            .unwrap_or(0);
        let mut bytes = add(
            mul(2, self.metadata_bytes()?)?,
            add(grow::<WindowShape>(trees)?, array::<u64>(trees)?)?,
        )?;
        bytes = add(bytes, grow::<u32>(rank)?)?;
        if let Some(grouped) = grouped {
            let forests = leaf.sites().len() as u64;
            let polynomials = add(grouped.polynomials().len() as u64, forests)?;
            bytes = add(bytes, mul(2, grow::<MergePolynomial>(polynomials)?)?)?;
            let groups = grouped.groups();
            let bits = add(u64::from(groups.token_bits), u64::from(groups.layer_bits))?;
            bytes = add(bytes, mul(forests, grow::<usize>(bits)?)?)?;
            for polynomial in grouped.polynomials() {
                bytes = add(bytes, array::<usize>(polynomial.group_vars.len() as u64)?)?;
            }
        }
        Ok(bytes)
    }

    pub(super) fn new(
        leaf: &LeafLayout<ModelColumn>,
        selection: &Selection,
        grouped: Option<&GroupedLayout>,
        physical_counts: Option<&[tessera_witness::MultiplicityPlan]>,
    ) -> Result<Self, ProverError> {
        let mut trees = Vec::new();
        let mut table_rows = 0;
        let mut lookup_rows = 0;
        let mut owned = 0;
        let mut scratch = 0;
        let mut full_vars = 0;
        for instance in leaf.instances() {
            let height = instance.height as usize;
            full_vars = full_vars.max(height);
            if grouped.is_none() {
                let rows = power(height)?;
                lookup_rows = add(lookup_rows, rows)?;
                // Unit-numerator leaves plus all extension parents and container capacity.
                owned = add(owned, add(mul(56, rows)?, mul(512, (height + 1) as u64)?)?)?;
                trees.push(Shape {
                    height,
                    trees_log: 0,
                    fraction: true,
                    generated: false,
                });
            }
            let spec = selection
                .tables
                .iter()
                .find(|t| t.id == instance.table)
                .ok_or_else(memory)?;
            let height = usize::from(spec.log2_rows);
            let rows = power(height)?;
            table_rows = add(table_rows, rows)?;
            let chunk = rows.min(CHUNK_ROWS as u64);
            // Paired extension chunks/stack, three base chunks, u64 count chunk,
            // partner outputs and descriptors. No complete table level is allocated.
            scratch = scratch.max(add(
                add(mul(128, chunk)?, mul(32, (height + 1) as u64)?)?,
                512,
            )?);
            trees.push(Shape {
                height,
                trees_log: 0,
                fraction: true,
                generated: true,
            });
        }
        let mut forest_rows = 0;
        let mut forest_polynomials = Vec::new();
        for (s, site) in leaf.sites().iter().enumerate() {
            let Some(read) = leaf
                .reads(s, 0)
                .and_then(|rs| rs.iter().find(|r| r.term == ModelColumn::ExpKey))
            else {
                continue;
            };
            let gadget = site.layers.first().ok_or_else(memory)?;
            let request = gadget
                .lookups
                .iter()
                .find(|r| {
                    core::iter::once(&r.lookup.key)
                        .chain(&r.lookup.value)
                        .any(|f| f.terms.iter().any(|(_, c)| *c == read.term))
                })
                .ok_or_else(memory)?;
            let domain = read
                .axes
                .iter()
                .map(|&a| request.domain.get(a).copied().ok_or_else(memory))
                .collect::<Result<Vec<_>, _>>()?;
            let [_, _, keys] = domain.as_slice() else {
                return Err(memory());
            };
            let height = vars(&[*keys])?;
            if height == 0 {
                continue;
            }
            let block = vars(&domain)?;
            let layers = site
                .layers
                .len()
                .checked_next_power_of_two()
                .ok_or_else(memory)?
                .trailing_zeros() as usize;
            let all = block.checked_add(layers).ok_or_else(memory)?;
            full_vars = full_vars.max(all);
            let rows = power(all)?;
            forest_rows = add(forest_rows, rows)?;
            if grouped.is_none() {
                owned = add(owned, add(mul(16, rows)?, mul(128, (height + 1) as u64)?)?)?;
                trees.push(Shape {
                    height,
                    trees_log: all - height,
                    fraction: false,
                    generated: false,
                });
            } else if let Some(grouped) = grouped {
                let g = grouped.groups();
                forest_polynomials.push(MergePolynomial {
                    vars: all,
                    group_vars: (height..height + usize::from(g.token_bits))
                        .chain(block..block + usize::from(g.layer_bits))
                        .collect(),
                });
            }
        }
        let retained = if let Some(grouped) = grouped {
            let forest_count = forest_polynomials.len() as u64;
            let mut polynomials = grouped.polynomials().to_vec();
            polynomials.extend(forest_polynomials);
            full_vars = full_vars.max(polynomials.iter().map(|p| p.vars).max().unwrap_or(0));
            let j = balanced_j_star(&polynomials, grouped.count());
            let classes = mul(grouped.count() as u64, polynomials.len() as u64)?;
            let buffers = mul(mul(classes, power(j + 2)?)?, 16)?;
            // All retained group levels, claims, split points, class/container metadata
            // and merge tables. Every group/polynomial remains in the bound.
            let v = full_vars as u64;
            let tree_count = add(leaf.instances().len() as u64, forest_count)?;
            // Per level: round/evaluation vectors and each tree's carried point/claim.
            let level = add(
                add(512, mul(160, v)?)?,
                mul(tree_count, add(256, mul(32, v)?)?)?,
            )?;
            let tree_metadata = add(mul(v, level)?, mul(tree_count, add(256, mul(32, v)?)?)?)?;
            // Per polynomial: split/merge claims, a class point/table descriptors and
            // vector growth. The table payload itself is charged in `buffers` above.
            let class_metadata = mul(polynomials.len() as u64, add(512, mul(128, v)?)?)?;
            let metadata = mul(
                grouped.count() as u64,
                add(512, add(tree_metadata, class_metadata)?)?,
            )?;
            add(buffers, metadata)?
        } else {
            0
        };
        let forests = mul(8, forest_rows)?;
        // The resident oracle keeps its established dense source reservation.
        // A cached-route caller can instead supply the canonical public backing
        // plans derived by Engine::runtime_multiplicity_plans. No secret sparsity
        // or arbitrary host allowance enters this selection (RFC-0018).
        let table_sources = if let Some(plans) = physical_counts {
            if plans.len() != leaf.instances().len() {
                return Err(memory());
            }
            let mut bytes =
                tessera_witness::Engine::table_memory(&selection.tables)?.retained_bytes();
            for (plan, instance) in plans.iter().zip(leaf.instances()) {
                let spec = selection
                    .tables
                    .iter()
                    .find(|s| s.id == instance.table)
                    .ok_or_else(memory)?;
                if plan.table() != instance.table
                    || plan.rows() != power(usize::from(spec.log2_rows))?
                    || plan.max_updates() == 0
                {
                    return Err(memory());
                }
                bytes = add(bytes, plan.payload_bytes())?;
            }
            bytes
        } else {
            mul(16, table_rows)?
        };
        // Actual lookup columns, forest inputs and retained group state remain
        // charged in full. Generic table descriptors/read scratch are unchanged.
        let source = add(
            add(table_sources, mul(16, lookup_rows)?)?,
            add(forests, retained)?,
        )?;
        let max_vars = trees
            .iter()
            .map(|s| s.height + s.trees_log)
            .max()
            .unwrap_or(0);
        if max_vars >= usize::BITS as usize {
            return Err(memory());
        }
        let metadata = add(
            65_536,
            mul(
                trees.len() as u64,
                mul(mul((max_vars + 1) as u64, (max_vars + 8) as u64)?, 384)?,
            )?,
        )?;
        // Native source descriptors and container capacities, separate from construction.
        let fixed = add(
            add(add(source, owned)?, metadata)?,
            mul(trees.len() as u64, NATIVE_DESCRIPTOR_BYTES_PER_TREE)?,
        )?;
        Self {
            trees,
            fixed,
            scratch,
            retained,
            forests,
            working_bytes: 0,
            cuts: Vec::new(),
            chunk_rows: CHUNK_ROWS,
            cache_log_rows: None,
        }
        .select(u64::MAX)
    }

    fn fixed_bytes(&self, chunk_rows: usize) -> Result<u64, ProverError> {
        let mut scratch = self.scratch;
        if chunk_rows > CHUNK_ROWS {
            for tree in self.trees.iter().filter(|tree| tree.generated) {
                let chunk = power(tree.height)?.min(chunk_rows as u64);
                let height = u64::try_from(tree.height).map_err(|_| memory())?;
                scratch = scratch.max(add(add(mul(128, chunk)?, mul(32, add(height, 1)?)?)?, 512)?);
            }
        }
        add(
            add(add(self.fixed, scratch)?, self.cache_bytes()?)?,
            mul(
                self.trees.len() as u64,
                TABLE_BATCH_CONSTRUCTION_BYTES_PER_TREE,
            )?,
        )
    }

    pub(super) fn select(self, allowance: u64) -> Result<Self, ProverError> {
        self.select_with_chunk_limit(allowance, MAX_CHUNK_ROWS)
    }

    fn cache_bytes(&self) -> Result<u64, ProverError> {
        self.trees
            .iter()
            .filter(|tree| tree.generated)
            .try_fold(0, |bytes, tree| {
                let Some(depth) = cache_depth(tree.height, self.cache_log_rows) else {
                    return Ok(bytes);
                };
                add(
                    bytes,
                    mul(
                        power(depth)?,
                        2 * size_of::<tessera_field::GoldilocksExt2>() as u64,
                    )?,
                )
            })
    }

    pub(super) fn select_cached(self, allowance: u64) -> Result<Self, ProverError> {
        let mut plan = self.select(allowance)?;
        let window_peak = plan
            .working_bytes
            .checked_sub(plan.fixed_bytes(plan.chunk_rows)?)
            .ok_or_else(memory)?;
        for log_rows in (1..=MAX_CACHE_LOG_ROWS).rev() {
            plan.cache_log_rows = Some(log_rows);
            if plan.cache_bytes().ok().is_some_and(|bytes| bytes > 0)
                && let Some(bytes) = plan
                    .fixed_bytes(plan.chunk_rows)
                    .ok()
                    .and_then(|fixed| fixed.checked_add(window_peak))
                    .filter(|&bytes| bytes <= allowance)
            {
                plan.working_bytes = bytes;
                return Ok(plan);
            }
        }
        plan.cache_log_rows = None;
        Ok(plan)
    }

    fn select_with_chunk_limit(
        mut self,
        allowance: u64,
        limit: usize,
    ) -> Result<Self, ProverError> {
        if limit < CHUNK_ROWS || !limit.is_power_of_two() {
            return Err(memory());
        }
        // First retain the existing feasible plan and its minimal window cuts. A larger read
        // chunk may consume spare public budget, never add a streaming round (RFC-0018).
        // `scratch` remains the original charge, so repeated selection can shrink again.
        self.chunk_rows = CHUNK_ROWS;
        self.cache_log_rows = None;
        let fixed = self.fixed_bytes(CHUNK_ROWS)?;
        if fixed > allowance {
            return Err(memory());
        }
        self.cuts.clear();
        self.working_bytes = fixed;
        let height = self.trees.iter().map(|s| s.height).max().unwrap_or(0);
        for level in 0..height {
            let shapes = self
                .trees
                .iter()
                .filter_map(|s| {
                    let join = height - s.height;
                    (join <= level).then(|| WindowShape {
                        vars: s.trees_log + level - join,
                        tables: if s.fraction { 4 } else { 2 },
                        delta: 2,
                        windowable: s.generated,
                    })
                })
                .collect::<Vec<_>>();
            let rounds = shapes.iter().map(|s| s.vars).max().unwrap_or(0);
            let choice = (0..rounds.max(1))
                .find_map(|cut| {
                    let memory = WindowMemory::with_window_vars(&shapes, cut, 16, 1).ok()?;
                    let bytes = fixed.checked_add(memory.working_bytes)?;
                    (bytes <= allowance).then_some((cut, bytes))
                })
                .ok_or_else(memory)?;
            self.cuts.push(choice.0);
            self.working_bytes = self.working_bytes.max(choice.1);
        }
        let window_peak = self.working_bytes.checked_sub(fixed).ok_or_else(memory)?;
        let largest = self
            .trees
            .iter()
            .filter(|tree| tree.generated)
            .map(|tree| tree.height)
            .max()
            .map(power)
            .transpose()?
            .unwrap_or(0);
        let mut chunk = usize::try_from(largest).map_err(|_| memory())?.min(limit);
        while chunk > CHUNK_ROWS {
            if let Some(bytes) = self
                .fixed_bytes(chunk)
                .ok()
                .and_then(|fixed| fixed.checked_add(window_peak))
                .filter(|&bytes| bytes <= allowance)
            {
                self.chunk_rows = chunk;
                self.working_bytes = bytes;
                break;
            }
            chunk /= 2;
        }
        Ok(self)
    }
}
