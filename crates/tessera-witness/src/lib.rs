// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The CPU reference integer engine for `D`, which is the bit-exact oracle, and the witness
//! streaming interface (RFC-0018).
//!
//! [`Engine`] runs a [`Lowering`](tessera_compiler::Lowering) of a model on one sequence of tokens
//! and returns its [`Witness`]: every tensor's values, exactly as RFC-0005 'Integer semantics'
//! defines them through RFC-0006's operations. The GPU engine and the PyTorch reference must give
//! the same values bit for bit (07 §5).
//!
//! - **Exact.** Values are `i64`, which range analysis shows holds each of them; every gadget
//!   computes through its regenerated table or exact integer arithmetic.
//! - **Checked.** Debug builds check every value against its tensor's analyzed range (INV-DM-2).
//! - **Deterministic.** The matrix products split rows among threads, and each value is one
//!   thread's exact sum, so the thread count changes nothing.
//! - **Secret.** The tokens arrive in a `Secret`, and every tensor lives in a `SecretBuf` that
//!   clears its memory (05 §3). The logits leave as a [`LogitsFile`](tessera_ir::LogitsFile), a
//!   secret-class artifact.
//!
//! # Checkpoints and regeneration
//!
//! [`Checkpoints`] keeps the layer-boundary INT32 residuals. [`Regenerator`] replays a requested
//! tile from its layer's checkpoint; [`Resident`] reads the first full pass. Both implement
//! [`WitnessSource`]. Shares, gadget values, lookup keys and table multiplicities live in
//! zeroizing buffers, and debug forms reveal shapes only. A table tile's counts include the
//! per-axis padding and row-0 column tail of RFC-0045, with zero counts on padded table rows.
//!
//! Memory includes the checkpoints, the returned tile and the replay's working tensors; no
//! regenerated value is retained between calls. The allocator test reports these quantities
//! separately. SoftmaxRow, RsqrtRow and Argmax auxiliary witness integration remains outstanding; this
//! interface supplies component tiles, not a complete proof witness. A table tile whose counts
//! would omit a SoftmaxRow or Argmax auxiliary lookup returns a public usage error, including when
//! a supported requantization shares that table. Such tables are excluded from memory figures.
//!
//! # Features
//!
//! - `unsafe-debug-dump` dumps secret tensors for debugging (OBS-4). Tests only: a release build
//!   with it fails to compile.

#![forbid(unsafe_code)]

// Test-only features cannot reach a release build (RFC-0001 'Cargo features').
#[cfg(all(feature = "unsafe-debug-dump", not(debug_assertions)))]
compile_error!(
    "the `unsafe-debug-dump` feature dumps secret tensors; it must not be enabled in a release build (OBS-4, RFC-0001 'Cargo features')"
);

mod checkpoint;
mod engine;
mod error;
mod replay;
mod source;
mod structure;
mod tile;

pub use checkpoint::Checkpoints;
pub use engine::{Engine, Witness};
pub use error::WitnessError;
pub use source::{Regenerator, Resident, WitnessSource};
pub use tile::{TileLookup, TileWitness};
