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

mod engine;
mod error;

pub use engine::{Engine, Witness};
pub use error::WitnessError;
