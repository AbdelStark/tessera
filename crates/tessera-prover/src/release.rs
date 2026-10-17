// SPDX-License-Identifier: Apache-2.0 OR MIT

//! When the prover releases a message that a verifier can time (RFC-0018 'Release schedule',
//! INV-ARCH-2).
//!
//! The work behind a message takes a time that depends on secrets. Multiplicity counting and
//! table reads touch memory at addresses the keys choose, and clocks move with the data. In
//! `tessera-protocol/2` a message therefore leaves at a slot that public data fixes:
//!
//! - at its budget `D` if its work is done by then;
//! - otherwise at the first of `2D, 4D, …, 2^L·D` that the work meets.
//!
//! Every duration that meets the same slot gives the same release time, so the time reveals the
//! slot and nothing else. Work past the last slot aborts the session with `ERR-RES-TIME`.
//!
//! The budget comes from public data alone: the caller computes it from the message's public
//! work counts and coefficients calibrated on public inputs, never from the timing of a session
//! that proved secret data.

use core::fmt;
use core::time::Duration;

/// The release slots of one message: its budget `D`, then `2D`, `4D`, …, `2^L·D`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReleaseSchedule {
    budget: Duration,
    doublings: u32,
}

/// When a message leaves, as a time after its trigger, and which slot that is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Release {
    /// The time after the trigger at which the message leaves, `2^slot · D`.
    pub after: Duration,
    /// The slot: 0 for the budget, `j` for `2^j · D`.
    pub slot: u32,
}

/// A release schedule that cannot be built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScheduleError {
    /// A budget of zero, which would release every message the moment it is ready.
    ZeroBudget,
    /// The last slot, `2^L · D`, does not fit a [`Duration`].
    LastSlotOverflows,
}

impl fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::ZeroBudget => "a release budget must be positive",
            Self::LastSlotOverflows => "the last release slot does not fit a duration",
        })
    }
}

impl std::error::Error for ScheduleError {}

impl ReleaseSchedule {
    /// The schedule of a message whose budget is `budget`, with `doublings` slots after it: `L`
    /// of RFC-0018, `ProverConfig.release.doublings`.
    ///
    /// # Errors
    ///
    /// [`ScheduleError::ZeroBudget`] for a zero budget, then
    /// [`ScheduleError::LastSlotOverflows`] if `2^doublings · budget` does not fit a [`Duration`].
    pub fn new(budget: Duration, doublings: u32) -> Result<Self, ScheduleError> {
        if budget.is_zero() {
            return Err(ScheduleError::ZeroBudget);
        }
        1u32.checked_shl(doublings)
            .and_then(|factor| budget.checked_mul(factor))
            .ok_or(ScheduleError::LastSlotOverflows)?;
        Ok(Self { budget, doublings })
    }

    /// The budget `D`.
    pub fn budget(&self) -> Duration {
        self.budget
    }

    /// The last slot, `2^L · D`: a verifier's session deadline must exceed it.
    pub fn last_slot(&self) -> Duration {
        // `new` checked that it fits.
        self.budget * (1 << self.doublings)
    }

    /// The release of a message whose work was done `elapsed` after its trigger: the first slot
    /// at or after `elapsed`. `None` past the last slot, where the session aborts.
    ///
    /// The result depends on `elapsed` only through the slot it falls in.
    pub fn release(&self, elapsed: Duration) -> Option<Release> {
        let mut after = self.budget;
        for slot in 0..=self.doublings {
            if elapsed <= after {
                return Some(Release { after, slot });
            }
            // At most the last slot, which `new` checked fits.
            after = after.saturating_mul(2);
        }
        None
    }
}
