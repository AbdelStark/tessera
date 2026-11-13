// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Device offload of tree-level rounds (RFC-0018, #733, #734). A backend registers a
//! [`LevelOffload`]; the window prover hands it each resident fraction or product instance's
//! `Fp2` tables once they are folded, and runs the instance's remaining rounds through it. The
//! offload computes the same field elements as the host, so proofs do not depend on it; an
//! instance it declines stays on the host.

use std::sync::{Arc, RwLock};

use tessera_field::GoldilocksExt2 as Fp2;

use crate::WindowError;

/// The polynomial of an offloaded level instance over its tables.
#[derive(Clone, Copy, Debug)]
pub enum LevelShape {
    /// `N0·D1 + N1·D0 + λ·D0·D1` over `[N0, N1, D0, D1]`.
    Fraction {
        /// The level's `λ`.
        lambda: Fp2,
    },
    /// `V0·V1` over `[V0, V1]`.
    Product,
}

/// An instance's tables on a device, `2^m` entries each until bound.
pub trait DeviceLevel: Send + Sync {
    /// `Σ_p w(p)·P(line_x(p))` for `x = 0, 2, 3` over the pairs `p` of the current tables, with
    /// `w(p) = low[p mod 2^low_bits]·high[p >> low_bits]` and `line_x(p) = t[2p] + x·(t[2p+1] −
    /// t[2p])` per table.
    ///
    /// # Errors
    /// A device failure, which ends the proof.
    fn sums(&self, low: &[Fp2], high: &[Fp2], low_bits: usize) -> Result<[Fp2; 3], WindowError>;

    /// Binds every table's lowest variable at `r`: `t'[i] = t[2i] + r·(t[2i+1] − t[2i])`.
    ///
    /// # Errors
    /// A device failure.
    fn bind(&mut self, r: Fp2) -> Result<(), WindowError>;

    /// Each table's first entry, in table order: the final evaluations once every variable is
    /// bound.
    ///
    /// # Errors
    /// A device failure.
    fn firsts(&self) -> Result<Vec<Fp2>, WindowError>;
}

/// A device that takes level instances.
pub trait LevelOffload: Send + Sync {
    /// Takes `tables` (each `2^m` entries, `m ≥ 1`) for `shape`, or declines with `None` (no
    /// room, or too small to pay for the transfer).
    ///
    /// # Errors
    /// A device failure.
    fn load(
        &self,
        shape: LevelShape,
        tables: &[&[Fp2]],
    ) -> Result<Option<Box<dyn DeviceLevel>>, WindowError>;
}

static OFFLOAD: RwLock<Option<Arc<dyn LevelOffload>>> = RwLock::new(None);

/// Registers the offload every later level uses, or none.
pub fn set_level_offload(offload: Option<Arc<dyn LevelOffload>>) {
    *OFFLOAD
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = offload;
}

pub(crate) fn level_offload() -> Option<Arc<dyn LevelOffload>> {
    OFFLOAD
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}
