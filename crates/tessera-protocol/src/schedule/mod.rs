// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The transcript schedule (RFC-0014 'Transcript schedule' and 'Grinding'): the label patterns,
//! their expansion for one proof unit, and a transcript held to it in both modes.
//!
//! - [`LabelPattern`] is the registry of `tessera-protocol/1`'s tier-A patterns, and a [`Label`]
//!   one of them with its indices. Both sides build every label from it (RFC-0004 'Transcript').
//!   [`WEIGHTS_XI`] is the one place that puts each weight matrix's proximity point
//!   `pcs/weights/{T}/xi`: right after the seed, as RFC-0049 moves it.
//! - [`TranscriptSchedule::expand`] lists every step of a proof unit, a label and what the
//!   transcript does under it, from the [`ScheduleSizes`] that public data fixes: the leaf stage's
//!   trees, the reduction stages, the layer point, the proximity points and the opened matrices,
//!   in a mode at a parameter set. Each label appears once (INV-FS-4), and every prover message is
//!   absorbed before the challenges that depend on it (INV-FS-1).
//! - [`ScheduledTranscript`] is RFC-0004's transcript held to a schedule: each operation must be
//!   the next step. In a live session it absorbs the coin-tossed seed and the verifier's queries
//!   at their places; in a non-interactive proof it grinds for the seed and the queries, and the
//!   verifier counts each nonce's leading zero bits exactly, rejecting a short one
//!   ([`PowRejected`], `Reject(ProofOfWork)`).
//!
//! `tessera-protocol/2`'s labels, its masked sums, LogUp identity sumcheck and blinding claims,
//! slot in at [`Protocol::Two`], which no schedule expands yet (#148).

mod expand;
mod label;
mod scheduled;
mod sizes;

pub use expand::{Op, Step, TranscriptSchedule};
pub use label::{Label, LabelPattern, Protocol, WEIGHTS_XI, WeightsXi};
pub use scheduled::{PowRejected, ScheduleFault, ScheduledTranscript};
pub use sizes::{LeafTree, MatrixPoint, ScheduleSizes, StageSizes};
