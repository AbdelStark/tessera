// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Hand-written TCE decoders under explicit decode limits, and the few encoders needed to recompute
//! digests (RFC-0003).

#![forbid(unsafe_code)]
// Decoders read hostile bytes and must reject them with typed errors, never by panicking (RFC-0019
// 'Robustness').
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]
