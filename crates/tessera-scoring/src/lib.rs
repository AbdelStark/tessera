// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Exact greedy request scoring and `Û = 0` accounting (RFC-0017, RFC-0026).
//!
//! [`ExactGreedy`] attaches to the compiler's original token, OneHot and LM-head
//! logits sources. [`ExactRequest::witness`] consumes one regenerated output row
//! at a time, retains only clearing maximum advice, and reports any deviation
//! after all real output rows. Each [`GreedyRow`] reuses the SAME original
//! indicator in the existing Argmax key, selected-logit and suffix relations.
//!
//! The driver supplies the public selected Range(29) specification and assigns
//! global certificate/gadget identities. It commits the maximum column, joins
//! lookups and graph/source claims, and batches over the output-row axis before
//! releasing transcript messages. Row access is for table assembly; per-row
//! maxima, logits and match information must not be serialized or logged. These
//! components do not return a verifier verdict or define new protocol messages.
//!
//! `PredictorSpec` remains the protocol crate's sibling-owned wire type. Its
//! driver chooses this exact-greedy implementation; no scoring-to-protocol
//! dependency or extra claim-list field is needed. Sampler PRF, temperature and
//! mixture accounting remain with the v0.4 issues.

#![forbid(unsafe_code)]

mod exact;

pub use exact::{
    ExactGreedy, ExactRequest, GreedyCertificate, GreedyInputs, GreedyRow, OutputRow, ScoringError,
};
