// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public P2 table-batch reservation, additional to the unchanged resident-cell floor.

use super::ProverError;
use crate::leaf::{GroupedLayout, LeafLayout, ModelColumn, balanced_j_star};
use tessera_compiler::Selection;
use tessera_logup::TABLE_BATCH_CONSTRUCTION_BYTES_PER_TREE;
use tessera_sumcheck::{MergePolynomial, WindowMemory, WindowShape};

pub(super) const CHUNK_ROWS: usize = 256;
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
        }
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
                    chunk_rows: core::num::NonZeroUsize::new(256).unwrap(),
                },
            )
            .unwrap();
            assert_eq!(proof.multiplicities().len(), count);
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

impl TablePlan {
    pub(super) fn new(
        leaf: &LeafLayout<ModelColumn>,
        selection: &Selection,
        grouped: Option<&GroupedLayout>,
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
        // Generated core + canonical counts each use at most 8*T physical bytes.
        // Smaller weighted count storage is still covered; the legacy floor is unchanged.
        let source = add(
            add(mul(16, table_rows)?, mul(16, lookup_rows)?)?,
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
        }
        .select(u64::MAX)
    }

    pub(super) fn select(mut self, allowance: u64) -> Result<Self, ProverError> {
        // `select` can run repeatedly; construction is a local charge, never accumulated.
        let fixed = add(
            add(self.fixed, self.scratch)?,
            mul(
                self.trees.len() as u64,
                TABLE_BATCH_CONSTRUCTION_BYTES_PER_TREE,
            )?,
        )?;
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
        Ok(self)
    }
}
