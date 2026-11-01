// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A CPU proof's pass plan (RFC-0018 'Pass structure', 'Memory scheduler').
//!
//! P1 keeps a canonical witness for commitments and openings. A reduction stage that exceeds
//! its working-set budget uses RFC-0055 grids over regenerated tensor recipes, then P3b folds;
//! RFC-0056 caps each window at eight own variables. Resident matrix/native operands remain
//! charged in full. The next stage's grid is prepared before the current stage's splits.
//!
//! The plan must fit before P1, or the proof returns `ERR-RES-MEMORY` before any commitment.
//! RFC-0054 leaf groups reduce the resident lookup-tree cells when stages alone cannot fit.
//! The cell floor, [`PassPlan::BYTES_PER_CELL`], reserves the cached witness, regenerated witness
//! scratch and the other proof phases beside the weight commitment; stage grids, construction
//! scratch and folded tables are charged separately. Counting-allocator regressions check the
//! whole proof peak under both resident and forced-window budgets. This CPU path retains a
//! resident witness floor; it does not claim to stream the entire model on arbitrary hardware.

pub use tessera_codec::metrics::Pass;
use tessera_protocol::LeafGroups;
use tessera_sumcheck::{WindowMemory, WindowShape};

/// The cells a CPU proof keeps resident, each counted at its padded size.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cells {
    /// Every graph tensor's entries: the witness, and the tables the passes derive from it.
    pub tensors: u64,
    /// Every LogUp instance's lookups: its lookup tree's leaves.
    pub lookups: u64,
    /// Every LogUp instance's table rows: its table tree's leaves.
    pub table_rows: u64,
    /// The advice matrix's entries.
    pub advice: u64,
}

impl Cells {
    /// Their sum, or `None` on overflow.
    pub fn total(&self) -> Option<u64> {
        self.tensors
            .checked_add(self.lookups)?
            .checked_add(self.table_rows)?
            .checked_add(self.advice)
    }
}

/// The passes a CPU proof runs and what it keeps resident (RFC-0018 'Pass structure').
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PassPlan {
    passes: Vec<Pass>,
    cells: Cells,
    weight_bytes: u64,
    leaf_groups: LeafGroups,
    stage_shapes: Vec<Vec<WindowShape>>,
    stage_windows: Vec<WindowMemory>,
    table_plan: Option<super::table_plan::TablePlan>,
    checkpoint_bytes: u64,
    caller_bytes: u64,
    preparation_bytes: u64,
}

impl PassPlan {
    /// The bytes charged per resident cell: the measured 32 to 34, up to 16 for a regenerated
    /// witness and its scratch, and margin. Stage grids/folded tables are charged separately.
    /// This floor is unchanged by window selection.
    pub const BYTES_PER_CELL: u64 = 64;

    /// The plan of a proof of `stages` stages, the leaf stage included, over `cells`, beside a
    /// weight commitment of `weight_bytes`. Its passes are P1, P2, one P3 per reduction stage, P4
    /// and P5, in that order.
    pub(crate) fn new(stages: usize, cells: Cells, weight_bytes: u64) -> Self {
        let mut passes = vec![Pass::P1, Pass::P2];
        passes.extend((1..stages).map(|_| Pass::P3));
        passes.extend([Pass::P4, Pass::P5]);
        Self {
            passes,
            cells,
            weight_bytes,
            leaf_groups: LeafGroups::NONE,
            stage_shapes: Vec::new(),
            stage_windows: Vec::new(),
            table_plan: None,
            checkpoint_bytes: 0,
            caller_bytes: 0,
            preparation_bytes: 0,
        }
    }

    pub(super) fn preparation(mut self, memory: super::preparation::SessionMemory) -> Self {
        self.caller_bytes = memory.caller;
        self.preparation_bytes = memory.peak;
        self
    }

    /// Whether this plan borrows INT8 weights and retains checkpoint-derived nonweight data.
    pub fn uses_checkpoints(&self) -> bool {
        self.checkpoint_bytes != 0
    }

    // Preserve the full resident floor and add every newly owned replay buffer.
    pub(super) fn checkpoints(mut self, bytes: u64) -> Self {
        self.checkpoint_bytes = bytes;
        self
    }

    pub(crate) fn staged(
        mut self,
        shapes: Vec<Vec<WindowShape>>,
    ) -> Result<Self, super::ProverError> {
        self.stage_windows = shapes
            .iter()
            .map(|s| {
                if s.is_empty() {
                    Ok(WindowMemory {
                        cut: 0,
                        fold_passes: 0,
                        working_bytes: 0,
                    })
                } else {
                    WindowMemory::for_cut(s, 0, 16).map_err(|_| Self::memory())
                }
            })
            .collect::<Result<_, _>>()?;
        self.stage_shapes = shapes;
        Ok(self)
    }

    fn memory() -> super::ProverError {
        super::ProverError::Resource(tessera_codec::error::ErrorCode::ResMemory)
    }

    pub(super) fn tables(mut self, plan: super::table_plan::TablePlan) -> Self {
        self.table_plan = Some(plan);
        self
    }

    fn table_reservation(&self) -> Option<u64> {
        self.cells
            .table_rows
            .checked_add(self.cells.lookups)?
            .checked_mul(Self::BYTES_PER_CELL)
    }

    pub(super) fn table_plan(&self) -> Option<&super::table_plan::TablePlan> {
        self.table_plan.as_ref()
    }

    /// Planned upper bounds on streamed global table rounds per level. The table engine
    /// may need fewer rounds when its actual capacities are below the public reservation.
    /// Table streaming regenerates no model witness.
    pub fn table_cuts(&self) -> &[usize] {
        self.table_plan.as_ref().map_or(&[], |p| &p.cuts)
    }

    fn base_bytes(&self) -> Option<u64> {
        self.cells
            .total()?
            .checked_mul(Self::BYTES_PER_CELL)?
            .checked_add(self.weight_bytes)?
            .checked_add(self.checkpoint_bytes)?
            .checked_add(self.caller_bytes)
    }

    /// Fit reduction stages and the complete table batch before P1. The cell floor remains;
    /// P2 uses its named table/lookup reservation and charges any excess separately.
    pub(crate) fn for_budget(
        mut self,
        budget: super::MemoryBudget,
    ) -> Result<Self, super::ProverError> {
        if self.check(budget).is_ok() {
            return Ok(self);
        }
        let available = budget
            .host
            .checked_sub(self.base_bytes().ok_or_else(Self::memory)?)
            .ok_or_else(Self::memory)?;
        if let Some(plan) = self.table_plan.take() {
            self.table_plan = Some(
                plan.select(
                    available
                        .checked_add(self.table_reservation().ok_or_else(Self::memory)?)
                        .ok_or_else(Self::memory)?,
                )?,
            );
        }
        self.stage_windows = self
            .stage_shapes
            .iter()
            .map(|s| {
                if s.is_empty() {
                    Ok(WindowMemory {
                        cut: 0,
                        fold_passes: 0,
                        working_bytes: 0,
                    })
                } else {
                    WindowMemory::select(s, available, 16).map_err(|_| Self::memory())
                }
            })
            .collect::<Result<_, _>>()?;
        let mut passes = vec![Pass::P1, Pass::P2];
        if self.leaf_groups != LeafGroups::NONE {
            passes.push(Pass::P2b);
        }
        for stage in self.stage_windows.iter().skip(1) {
            passes.push(Pass::P3);
            passes.extend(core::iter::repeat_n(Pass::P3b, stage.fold_passes));
        }
        passes.extend([Pass::P4, Pass::P5]);
        self.passes = passes;
        self.check(budget)?;
        Ok(self)
    }

    /// Per-stage global round cuts. Zero uses the resident stage prover.
    pub fn stage_windows(&self) -> &[WindowMemory] {
        &self.stage_windows
    }

    /// The plan with its leaf stage in the token groups `groups` (RFC-0054): `lookups`, its
    /// largest group's lookup trees' leaves, in place of the stage's, and the merge's pass, P2b,
    /// after the group pass.
    pub(crate) fn grouped(mut self, groups: LeafGroups, lookups: u64) -> Self {
        self.cells.lookups = lookups;
        if let Some(at) = self.passes.iter().position(|&p| p == Pass::P2) {
            self.passes.insert(at + 1, Pass::P2b);
        }
        self.leaf_groups = groups;
        self
    }

    /// The leaf stage's token groups: [`LeafGroups::NONE`] for a stage proven in one batch.
    pub fn leaf_groups(&self) -> LeafGroups {
        self.leaf_groups
    }

    /// The passes, in the order the proof runs them.
    pub fn passes(&self) -> &[Pass] {
        &self.passes
    }

    /// The resident cells.
    pub fn cells(&self) -> Cells {
        self.cells
    }

    /// The charged peak: the unchanged cell floor and weight commitment, plus the larger
    /// reduction working set or table-batch excess over its named floor reservation.
    /// A selected checkpoint source additionally reserves its retained inputs and
    /// constructor peak; it never reduces the resident-cell floor. Borrowed plan
    /// and weight-state storage is charged beside the proof; the returned peak
    /// also covers checked session preparation. The caller's model is excluded.
    /// Returns `None` on overflow.
    pub fn resident_bytes(&self) -> Option<u64> {
        let table_extra = self.table_plan.as_ref().map_or(0, |p| {
            p.working_bytes
                .saturating_sub(self.table_reservation().unwrap_or(0))
        });
        let proof_peak = self.base_bytes()?.checked_add(
            self.stage_windows
                .iter()
                .map(|s| s.working_bytes)
                .max()
                .unwrap_or(0)
                .max(table_extra),
        )?;
        Some(proof_peak.max(self.preparation_bytes))
    }

    /// Whether the plan fits `budget`; `ERR-RES-MEMORY` if not, or if its size overflows.
    pub fn check(&self, budget: super::MemoryBudget) -> Result<(), super::ProverError> {
        match self.resident_bytes() {
            Some(bytes) if bytes <= budget.host => Ok(()),
            _ => Err(super::ProverError::Resource(
                tessera_codec::error::ErrorCode::ResMemory,
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preparation_and_caller_owners_are_charged_without_lowering_the_cell_floor() {
        let plan = PassPlan::new(
            1,
            Cells {
                tensors: 10,
                ..Cells::default()
            },
            100,
        )
        .preparation(super::super::preparation::SessionMemory {
            caller: 17,
            peak: 1000,
            weight_construction: 0,
        });
        assert_eq!(plan.base_bytes(), Some(10 * 64 + 100 + 17));
        assert_eq!(plan.resident_bytes(), Some(1000));
        assert!(
            plan.check(super::super::MemoryBudget { host: 1000 })
                .is_ok()
        );
        assert!(matches!(
            plan.check(super::super::MemoryBudget { host: 999 }),
            Err(super::super::ProverError::Resource(
                tessera_codec::error::ErrorCode::ResMemory
            ))
        ));
        let plan = plan.preparation(super::super::preparation::SessionMemory {
            caller: 1000,
            peak: 0,
            weight_construction: 0,
        });
        assert_eq!(plan.resident_bytes(), Some(10 * 64 + 100 + 1000));
    }

    #[test]
    fn stage_working_sets_select_windows_and_count_each_capped_fold() {
        let cells = Cells {
            tensors: 1 << 24,
            ..Cells::default()
        };
        let shapes = vec![
            Vec::new(),
            vec![
                WindowShape {
                    vars: 24,
                    tables: 2,
                    delta: 2,
                    windowable: true,
                },
                WindowShape {
                    vars: 10,
                    tables: 4,
                    delta: 3,
                    windowable: false,
                },
            ],
        ];
        let resident = PassPlan::new(2, cells, 100).staged(shapes).unwrap();
        let base = (1u64 << 24) * 64 + 100;
        assert_eq!(
            resident.resident_bytes(),
            Some(base + resident.stage_windows()[1].working_bytes)
        );
        let budget = super::super::MemoryBudget {
            host: base + (2 << 20),
        };
        assert!(resident.resident_bytes().unwrap() > budget.host);
        let window = resident.clone().for_budget(budget).unwrap();
        assert_eq!(window.stage_windows()[1].cut, 9);
        assert_eq!(window.stage_windows()[1].fold_passes, 2);
        assert_eq!(
            window.passes(),
            [
                Pass::P1,
                Pass::P2,
                Pass::P3,
                Pass::P3b,
                Pass::P3b,
                Pass::P4,
                Pass::P5
            ]
        );
        assert_eq!(
            window.resident_bytes(),
            Some(base + window.stage_windows()[1].working_bytes)
        );
        assert!(window.resident_bytes().unwrap() <= budget.host);
        assert!(
            resident
                .for_budget(super::super::MemoryBudget { host: base - 1 })
                .is_err()
        );
    }

    #[test]
    fn checkpoint_storage_is_additional_to_the_unchanged_resident_floor() {
        let plan = PassPlan::new(
            2,
            Cells {
                tensors: 7,
                ..Cells::default()
            },
            19,
        );
        let baseline = plan.resident_bytes().unwrap();
        assert!(!plan.uses_checkpoints());
        let replay = plan.checkpoints(131);
        assert!(replay.uses_checkpoints());
        assert_eq!(replay.resident_bytes(), Some(baseline + 131));
        assert!(
            replay
                .check(super::super::MemoryBudget {
                    host: baseline + 131
                })
                .is_ok()
        );
        assert!(matches!(
            replay.check(super::super::MemoryBudget {
                host: baseline + 130
            }),
            Err(super::super::ProverError::Resource(
                tessera_codec::error::ErrorCode::ResMemory
            ))
        ));
        assert_eq!(replay.checkpoints(u64::MAX).resident_bytes(), None);
    }

    #[test]
    fn a_proof_runs_each_pass_once_and_p3_once_per_reduction_stage() {
        let plan = PassPlan::new(4, Cells::default(), 0);
        assert_eq!(
            plan.passes(),
            [
                Pass::P1,
                Pass::P2,
                Pass::P3,
                Pass::P3,
                Pass::P3,
                Pass::P4,
                Pass::P5
            ]
        );
        assert_eq!(
            PassPlan::new(1, Cells::default(), 0).passes(),
            [Pass::P1, Pass::P2, Pass::P4, Pass::P5]
        );
    }

    #[test]
    fn the_budget_holds_the_charged_cells_and_the_weights_exactly() {
        let cells = Cells {
            tensors: 3,
            lookups: 5,
            table_rows: 7,
            advice: 11,
        };
        let plan = PassPlan::new(2, cells, 100);
        assert_eq!(plan.resident_bytes(), Some(26 * 64 + 100));
        let budget = |host| super::super::MemoryBudget { host };
        assert!(plan.check(budget(26 * 64 + 100)).is_ok());
        assert!(matches!(
            plan.check(budget(26 * 64 + 99)),
            Err(super::super::ProverError::Resource(
                tessera_codec::error::ErrorCode::ResMemory
            ))
        ));
        let huge = PassPlan::new(
            2,
            Cells {
                tensors: u64::MAX,
                ..cells
            },
            0,
        );
        assert_eq!(huge.resident_bytes(), None);
        assert!(
            huge.check(budget(u64::MAX)).is_err(),
            "an overflow never fits"
        );
    }
}
