// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The verifier's own `Fp` and `Fp2` arithmetic, roots of unity and the NTT used to re-encode
//! combined rows, written from RFC-0002 without reference to the prover's field crate.

#![forbid(unsafe_code)]
