// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The sumcheck engine, GKR layer protocols and stage batching (RFC-0010, RFC-0013).
//!
//! # Features
//!
//! - `test-rng` seeds masking randomness so that tests are reproducible. Tests only: a release
//!   build with it fails to compile.

#![forbid(unsafe_code)]

// Test-only features cannot reach a release build (RFC-0001 'Cargo features').
#[cfg(all(feature = "test-rng", not(debug_assertions)))]
compile_error!(
    "the `test-rng` feature pins masking randomness to a seed; it must not be enabled in a release build (RFC-0001 'Cargo features')"
);
