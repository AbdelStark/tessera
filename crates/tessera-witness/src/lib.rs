// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The CPU reference integer engine for `D`, which is the bit-exact oracle, and the witness
//! streaming interface (RFC-0018).
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
