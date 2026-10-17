// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Orchestration of the witness pass, commitments, stages and openings over the CPU and GPU
//! backends (RFC-0018).
//!
//! [`ReleaseSchedule`] fixes when a message that a verifier can time leaves the prover in
//! `tessera-protocol/2`: at a slot that public data determines, whatever its secret-dependent
//! work took (RFC-0018 'Release schedule', INV-ARCH-2).
//!
//! # Features
//!
//! - `test-rng` seeds masking randomness so that tests are reproducible. Tests only: a release
//!   build with it fails to compile.
//! - `unsafe-debug-dump` dumps secret tensors for debugging (OBS-4). Tests only: a release build
//!   with it fails to compile.

#![forbid(unsafe_code)]

mod release;

pub use release::{Release, ReleaseSchedule, ScheduleError};

// Test-only features cannot reach a release build (RFC-0001 'Cargo features').
#[cfg(all(feature = "test-rng", not(debug_assertions)))]
compile_error!(
    "the `test-rng` feature pins masking randomness to a seed; it must not be enabled in a release build (RFC-0001 'Cargo features')"
);
#[cfg(all(feature = "unsafe-debug-dump", not(debug_assertions)))]
compile_error!(
    "the `unsafe-debug-dump` feature dumps secret tensors; it must not be enabled in a release build (OBS-4, RFC-0001 'Cargo features')"
);
