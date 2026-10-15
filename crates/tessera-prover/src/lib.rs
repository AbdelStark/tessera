// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Orchestration of the witness pass, commitments, stages and openings over the CPU and GPU
//! backends (RFC-0018).
//!
//! # Features
//!
//! - `test-rng` seeds masking randomness so that tests are reproducible. Tests only: a release
//!   build with it fails to compile.
//! - `unsafe-debug-dump` dumps secret tensors for debugging (OBS-4). Tests only: a release build
//!   with it fails to compile.

#![forbid(unsafe_code)]

// Test-only features cannot reach a release build (RFC-0001 'Cargo features').
#[cfg(all(feature = "test-rng", not(debug_assertions)))]
compile_error!(
    "the `test-rng` feature pins masking randomness to a seed; it must not be enabled in a release build (RFC-0001 'Cargo features')"
);
#[cfg(all(feature = "unsafe-debug-dump", not(debug_assertions)))]
compile_error!(
    "the `unsafe-debug-dump` feature dumps secret tensors; it must not be enabled in a release build (OBS-4, RFC-0001 'Cargo features')"
);
