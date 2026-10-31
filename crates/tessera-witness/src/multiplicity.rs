// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Exact canonical table counts with storage selected from public bounds only.

use core::fmt;

use tessera_codec::SecretBuf;
use tessera_field::{Goldilocks, PrimeField};
use tessera_ir::TableId;

use crate::WitnessError;

const ROW: &str = "every lookup's key names a row of its table (INV-IR-2)";
const TOTAL: &str = "lookup multiplicities sum to less than P (RFC-0008)";
const BOUND: &str = "the public multiplicity update bound covers every contribution";

#[derive(Clone, Copy)]
#[repr(C)]
struct Update {
    // Two full words avoid uninitialized struct padding in clearing/allocator checks.
    row: u64,
    count: u64,
}

enum Storage {
    Dense(SecretBuf<u64>),
    Weighted(SecretBuf<Update>),
}

/// Allocation geometry for one canonical multiplicity column (RFC-0008, RFC-0018).
/// The public update bound includes every real/weighted padding contribution and the final
/// row-0 tail. This plan does not derive or authenticate that bound: its caller must do so
/// from the canonical public graph/runtime metadata, never from secret sparsity.
///
/// Counts describe requested backing payload only. Descriptors, constructor overlap,
/// allocator bookkeeping and all other proof phases remain separate charges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MultiplicityPlan {
    table: TableId,
    actual: u64,
    rows: u64,
    max_updates: u64,
    dense: bool,
    capacity: usize,
}

impl MultiplicityPlan {
    /// Choose the smaller backing from public row/update bounds; a byte tie uses dense storage.
    /// No allocation or witness read occurs. A zero update bound is preserved, but permits no
    /// later contribution, including the final row-0 tail.
    ///
    /// # Errors
    /// `Internal` for `actual` outside `1..=2^32`; `MemoryTable` if the chosen buffer cannot
    /// fit the target address space. Host-budget feasibility remains the caller's responsibility.
    pub fn new(table: TableId, actual: u64, max_updates: u64) -> Result<Self, WitnessError> {
        if actual == 0 {
            return Err(WitnessError::Internal {
                rule: "a multiplicity table has a row",
            });
        }
        if actual > 1 << 32 {
            return Err(WitnessError::Internal {
                rule: "a multiplicity table has at most 2^32 rows",
            });
        }
        let rows = actual.next_power_of_two();
        let dense_bytes = u128::from(rows) * size_of::<u64>() as u128;
        let weighted_bytes = u128::from(max_updates) * size_of::<Update>() as u128;
        let dense = dense_bytes <= weighted_bytes;
        let entries = if dense { rows } else { max_updates };
        let element = if dense {
            size_of::<u64>()
        } else {
            size_of::<Update>()
        };
        let memory = WitnessError::MemoryTable { table };
        let capacity = usize::try_from(entries).map_err(|_| memory)?;
        capacity
            .checked_mul(element)
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or(memory)?;
        Ok(Self {
            table,
            actual,
            rows,
            max_updates,
            dense,
            capacity,
        })
    }

    /// Canonical public table identity.
    pub fn table(&self) -> TableId {
        self.table
    }
    /// Padded logical row count, independent of the chosen representation.
    pub fn rows(&self) -> u64 {
        self.rows
    }
    /// Fixed public bound on all weighted contributions, including the row-0 tail.
    pub fn max_updates(&self) -> u64 {
        self.max_updates
    }
    /// Whether this plan retains a dense count array rather than fixed weighted updates.
    pub fn is_dense(&self) -> bool {
        self.dense
    }
    /// Requested backing payload in bytes, excluding descriptors and allocator overhead.
    pub fn payload_bytes(&self) -> u64 {
        // The chosen payload never exceeds 8*2^32 bytes, so neither cast nor product overflows.
        let element = if self.dense {
            size_of::<u64>()
        } else {
            size_of::<Update>()
        };
        self.capacity as u64 * element as u64
    }
}

/// One canonical multiplicity column, including zero table-tree padding (RFC-0045).
///
/// Physical storage may be dense or fixed-capacity weighted updates. Public table/update
/// bounds choose its capacity; secret sparsity never changes its logical domain
/// or permits skipping commitment rows. Counts remain secret-class (05 §3).
pub struct MultiplicityColumn {
    table: TableId,
    rows: u64,
    storage: Storage,
}

impl MultiplicityColumn {
    /// The canonical table identity whose committed counts this column contains.
    pub fn table(&self) -> TableId {
        self.table
    }

    /// Logical row count, including public power-of-two table padding.
    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// Fill `output` with consecutive canonical counts starting at `start`.
    /// Its length fixes the public chunk size. Values stay in clearing storage;
    /// the caller follows [`crate::TileWitness`]'s borrowing/redaction rules.
    ///
    /// # Errors
    /// [`WitnessError::Usage`] if the chunk exceeds the public logical domain.
    /// A refused chunk leaves `output` unchanged. No allocation occurs here.
    pub fn read_rows(&self, start: u64, output: &mut SecretBuf<u64>) -> Result<(), WitnessError> {
        let fits = u64::try_from(output.len())
            .ok()
            .and_then(|len| start.checked_add(len))
            .is_some_and(|end| end <= self.rows);
        if !fits {
            return Err(WitnessError::Usage {
                rule: "the multiplicity chunk fits its public table domain",
            });
        }
        match &self.storage {
            Storage::Dense(values) => {
                let start = start as usize;
                let end = start + output.len();
                output.copy_from_slice(&values[start..end]);
            }
            Storage::Weighted(updates) => {
                output.fill(0);
                if output.is_empty() {
                    return Ok(());
                }
                for update in updates.iter() {
                    let offset = update.row.wrapping_sub(start);
                    let inside = offset < output.len() as u64;
                    let index = core::hint::select_unpredictable(inside, offset as usize, 0);
                    let count = core::hint::select_unpredictable(inside, update.count, 0);
                    // INV-ARCH-2 permits keyed addresses for counting, never a keyed loop bound.
                    output[index] += count;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn bytes(&self) -> usize {
        match &self.storage {
            Storage::Dense(values) => values.capacity() * size_of::<u64>(),
            Storage::Weighted(updates) => updates.capacity() * size_of::<Update>(),
        }
    }
}

impl fmt::Debug for MultiplicityColumn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MultiplicityColumn")
            .field("table", &self.table)
            .field("rows", &self.rows)
            .field("public_storage_bytes", &self.bytes())
            .finish()
    }
}

pub(crate) struct Multiplicities {
    pub(crate) table: TableId,
    actual: u64,
    max_updates: u64,
    updates: u64,
    total: u64,
    column: MultiplicityColumn,
}

impl Multiplicities {
    /// `max_updates` includes weighted padding contributions and the final row-0 tail.
    pub(crate) fn new(table: TableId, actual: u64, max_updates: u64) -> Result<Self, WitnessError> {
        let plan = MultiplicityPlan::new(table, actual, max_updates)?;
        let memory = WitnessError::MemoryTable { table };
        let storage = if plan.dense {
            let size = plan.capacity;
            let mut counts = SecretBuf::try_with_capacity(size).map_err(|_| memory)?;
            counts.extend(core::iter::repeat_n(0, size));
            Storage::Dense(counts)
        } else {
            let capacity = plan.capacity;
            let mut updates = SecretBuf::try_with_capacity(capacity).map_err(|_| memory)?;
            // Fixed initialized dummy entries keep traversal length/capacity public (OBS-6).
            updates.extend(core::iter::repeat_n(Update { row: 0, count: 0 }, capacity));
            Storage::Weighted(updates)
        };
        Ok(Self {
            table: plan.table,
            actual: plan.actual,
            max_updates: plan.max_updates,
            updates: 0,
            total: 0,
            column: MultiplicityColumn {
                table: plan.table,
                rows: plan.rows,
                storage,
            },
        })
    }

    pub(crate) fn add(&mut self, row: u64, count: u64) -> Result<(), WitnessError> {
        let total = self
            .total
            .checked_add(count)
            .filter(|&n| n < Goldilocks::MODULUS)
            .ok_or(WitnessError::Internal { rule: TOTAL })?;
        if row >= self.actual {
            return Err(WitnessError::Internal { rule: ROW });
        }
        if self.updates == self.max_updates {
            return Err(WitnessError::Internal { rule: BOUND });
        }
        match &mut self.column.storage {
            Storage::Dense(values) => values[row as usize] += count,
            Storage::Weighted(updates) => updates[self.updates as usize] = Update { row, count },
        }
        self.total = total;
        self.updates += 1;
        Ok(())
    }

    /// The column, row 0 counting the lookups' padding (RFC-0045 'The column'): up to the next
    /// power of two of the entries counted, or, for a grouped stage, up to `grouped`, every
    /// group's column summed (RFC-0054 'When groups apply').
    pub(crate) fn finish(
        mut self,
        grouped: Option<u64>,
    ) -> Result<MultiplicityColumn, WitnessError> {
        let padded = match grouped {
            Some(entries) => Some(entries)
                .filter(|&entries| self.total > 0 && entries >= self.total)
                .ok_or(WitnessError::Internal {
                    rule: "a grouped stage's columns hold every lookup of the table",
                })?,
            None => self
                .total
                .checked_next_power_of_two()
                .filter(|_| self.total > 0)
                .ok_or(WitnessError::Internal {
                    rule: "a lookup column has at least one entry and fits u64",
                })?,
        };
        // RFC-0045 pads the lookup column with row 0, including its table value.
        self.add(0, padded - self.total)?;
        Ok(self.column)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_backing_plan_matches_allocations_at_dense_weighted_boundaries() {
        let boundaries = [
            0u64, 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128, 129,
        ];
        for actual in boundaries.into_iter().filter(|&n| n != 0) {
            let mut padded = 1u64;
            while padded < actual {
                padded *= 2;
            }
            for max_updates in boundaries {
                let mut plan = None;
                let planning = allocation_counter::measure(|| {
                    plan = Some(MultiplicityPlan::new(TableId(7), actual, max_updates).unwrap());
                });
                assert_eq!(planning.bytes_total, 0);
                let plan = plan.unwrap();
                let dense_bytes = 8 * padded;
                let weighted_bytes = 16 * max_updates;
                assert_eq!(plan.rows(), padded);
                assert_eq!(plan.is_dense(), dense_bytes <= weighted_bytes);
                assert_eq!(plan.payload_bytes(), dense_bytes.min(weighted_bytes));
                let mut counts = None;
                let allocated = allocation_counter::measure(|| {
                    counts = Some(Multiplicities::new(TableId(7), actual, max_updates).unwrap());
                });
                let counts = counts.unwrap();
                assert_eq!(
                    u64::try_from(allocated.bytes_current).unwrap(),
                    plan.payload_bytes()
                );
                assert_eq!(allocated.bytes_max, plan.payload_bytes());
                assert_eq!(counts.column.bytes() as u64, plan.payload_bytes());
                assert_eq!(counts.column.rows(), padded);
                match &counts.column.storage {
                    Storage::Dense(values) => {
                        assert!(plan.is_dense());
                        assert_eq!(values.len() as u64, padded);
                        assert!(values.iter().all(|&value| value == 0));
                    }
                    Storage::Weighted(updates) => {
                        assert!(!plan.is_dense());
                        assert_eq!(updates.len() as u64, max_updates);
                        assert!(
                            updates
                                .iter()
                                .all(|update| update.row == 0 && update.count == 0)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn public_backing_plan_preserves_exact_domain_and_zero_bound_refusals() {
        for (actual, rule) in [
            (0, "a multiplicity table has a row"),
            (
                (1u64 << 32) + 1,
                "a multiplicity table has at most 2^32 rows",
            ),
        ] {
            let measured = allocation_counter::measure(|| {
                assert_eq!(
                    MultiplicityPlan::new(TableId(7), actual, u64::MAX),
                    Err(WitnessError::Internal { rule })
                );
            });
            assert_eq!(measured.bytes_total, 0);
        }
        let mut counts = Multiplicities::new(TableId(7), 1, 0).unwrap();
        assert_eq!(
            counts.add(0, 0),
            Err(WitnessError::Internal { rule: BOUND })
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn huge_logical_backing_plans_allocate_nothing_and_never_wrap_update_bytes() {
        let measured = allocation_counter::measure(|| {
            let sparse = MultiplicityPlan::new(TableId(7), 1 << 32, 3).unwrap();
            assert!(!sparse.is_dense());
            assert_eq!(sparse.rows(), 1 << 32);
            assert_eq!(sparse.payload_bytes(), 48);
            let dense = MultiplicityPlan::new(TableId(7), 1 << 32, u64::MAX).unwrap();
            assert!(dense.is_dense());
            assert_eq!(dense.payload_bytes(), 8u64 << 32);
        });
        assert_eq!(measured.bytes_total, 0);
        assert_eq!(measured.bytes_max, 0);
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn backing_plan_refuses_an_unaddressable_dense_column_before_allocation() {
        assert_eq!(
            MultiplicityPlan::new(TableId(7), 1 << 32, u64::MAX),
            Err(WitnessError::MemoryTable { table: TableId(7) })
        );
    }

    #[test]
    fn grouped_tail_weight_preserves_the_public_backing_plan() {
        let plan = MultiplicityPlan::new(TableId(7), 1 << 32, 2).unwrap();
        for entries in [4u64, 16, 32] {
            let mut counts = Multiplicities::new(TableId(7), 1 << 32, 2).unwrap();
            counts.add(7, 3).unwrap();
            let column = counts.finish(Some(entries)).unwrap();
            assert_eq!(column.bytes() as u64, plan.payload_bytes());
            let mut first = SecretBuf::filled(0, 8);
            column.read_rows(0, &mut first).unwrap();
            assert_eq!(first[0], entries - 3);
            assert_eq!(first[7], 3);
            assert!(first[1..7].iter().all(|&count| count == 0));
        }
    }

    #[test]
    fn canonical_padding_chunks_and_exact_refusals() {
        let mut counts = Multiplicities::new(TableId(1), 3, 4).unwrap();
        counts.add(1, 2).unwrap();
        counts.add(2, 1).unwrap();
        assert_eq!(counts.add(3, 1), Err(WitnessError::Internal { rule: ROW }));
        let column = counts.finish(None).unwrap();
        assert_eq!(column.rows(), 4);
        let mut values = SecretBuf::filled(0, 4);
        column.read_rows(0, &mut values).unwrap();
        assert_eq!(&values[..], &[1, 2, 1, 0]);
        let mut suffix = SecretBuf::filled(99, 2);
        assert_eq!(
            column.read_rows(3, &mut suffix),
            Err(WitnessError::Usage {
                rule: "the multiplicity chunk fits its public table domain",
            })
        );
        assert_eq!(&suffix[..], &[99, 99]);
        assert_eq!(
            column.read_rows(u64::MAX, &mut suffix),
            Err(WitnessError::Usage {
                rule: "the multiplicity chunk fits its public table domain",
            })
        );
        column.read_rows(4, &mut SecretBuf::new()).unwrap();

        let mut bounded = Multiplicities::new(TableId(2), 1, 1).unwrap();
        bounded.add(0, 1).unwrap();
        assert!(matches!(
            bounded.finish(None),
            Err(WitnessError::Internal { rule: BOUND })
        ));
        let mut large = Multiplicities::new(TableId(2), 1, 3).unwrap();
        large.add(0, Goldilocks::MODULUS - 1).unwrap();
        assert_eq!(large.add(0, 1), Err(WitnessError::Internal { rule: TOTAL }));
        assert_eq!(
            large.add(0, u64::MAX),
            Err(WitnessError::Internal { rule: TOTAL })
        );
        assert!(matches!(
            Multiplicities::new(TableId(1), 0, 1),
            Err(WitnessError::Internal {
                rule: "a multiplicity table has a row"
            })
        ));
        assert!(matches!(
            Multiplicities::new(TableId(1), 3, 1).unwrap().finish(None),
            Err(WitnessError::Internal {
                rule: "a lookup column has at least one entry and fits u64"
            })
        ));
        assert_eq!(
            WitnessError::MemoryTable { table: TableId(1) }.code(),
            tessera_codec::error::ErrorCode::ResMemory
        );
    }

    #[test]
    fn wide_domain_uses_public_bounded_storage_and_reads_all_positions() {
        for distribution in [false, true] {
            let mut counts = Multiplicities::new(TableId(0), 1 << 32, 4).unwrap();
            counts
                .add(if distribution { u64::from(u32::MAX) } else { 0 }, 2)
                .unwrap();
            counts.add(7, 1).unwrap();
            let column = counts.finish(None).unwrap();
            assert_eq!(column.bytes(), 4 * size_of::<Update>());
            let mut first = SecretBuf::filled(0, 9);
            column.read_rows(0, &mut first).unwrap();
            assert_eq!(first[0], if distribution { 1 } else { 3 });
            assert_eq!(first[7], 1);
            assert!(first[1..7].iter().all(|&n| n == 0));
            let mut last = SecretBuf::filled(0, 1);
            column.read_rows(u64::from(u32::MAX), &mut last).unwrap();
            assert_eq!(last[0], if distribution { 2 } else { 0 });
            assert_eq!(column.rows(), 1 << 32);
        }
    }
}
