// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A CPU proof's pass plan (RFC-0018 'Pass structure', 'Memory scheduler').
//!
//! The CPU backend computes the witness once, in P1, and keeps it resident: P2 to P5 read it
//! back, and nothing streams. Its plan is therefore the passes it runs and the bytes it keeps
//! resident, which must fit the budget before P1 starts, or the proof fails with
//! `ERR-RES-MEMORY` before any commitment. A leaf stage larger than the budget needs RFC-0054's
//! token groups, which this backend does not prove yet (#134), so its plan refuses such a proof.
//!
//! The resident bytes are charged per cell, [`PassPlan::BYTES_PER_CELL`] each, beside the weight
//! commitment's. A counting allocator measures 32 to 34 bytes per cell over every conformance
//! proof, and `tests/cpu_api` holds every measured peak between the plan and a quarter of it.

pub use tessera_codec::metrics::Pass;

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
}

impl PassPlan {
    /// The bytes charged per resident cell: about twice the measured 32 to 34.
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
        }
    }

    /// The passes, in the order the proof runs them.
    pub fn passes(&self) -> &[Pass] {
        &self.passes
    }

    /// The resident cells.
    pub fn cells(&self) -> Cells {
        self.cells
    }

    /// The bytes the proof keeps resident: [`Self::BYTES_PER_CELL`] per cell and the weight
    /// commitment's, or `None` on overflow.
    pub fn resident_bytes(&self) -> Option<u64> {
        self.cells
            .total()?
            .checked_mul(Self::BYTES_PER_CELL)?
            .checked_add(self.weight_bytes)
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
