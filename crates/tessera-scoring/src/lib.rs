// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Exact greedy and exact temperature request scoring, the reference sampler PRF, and `Û = 0`
//! accounting (RFC-0017, RFC-0026, RFC-0058, RFC-0061).
//!
//! [`ExactGreedy`] and [`ExactTemperature`] attach to the compiler's original token, OneHot and
//! LM-head logits sources. [`ExactRequest::witness`] and [`TemperatureRequest::witness`] consume
//! one regenerated output row at a time, retain only clearing certificate advice, and report any
//! deviation after scanning all real output rows. Each [`GreedyRow`] and [`TemperatureRow`] reuses
//! the SAME original indicator in the certificate relations.
//!
//! The driver supplies the runtime table specifications (`Range` and `Exp_T`) and assigns global
//! certificate/gadget identities. It commits the certificate columns, joins lookups, PRF claims
//! and graph/source claims, and batches over the output-row axis before releasing transcript
//! messages. Row access is for table assembly; per-row maxima, exponentials, logits, draws and
//! match information must not be serialized or logged. These components do not return a verifier
//! verdict or define new protocol messages.
//!
//! `PredictorSpec` remains the protocol crate's sibling-owned wire type. Its driver chooses the
//! exact-greedy or exact-temperature implementation; no scoring-to-protocol dependency or extra
//! claim-list field is needed.
//!
//! [`Sampler`] and [`inverse_cdf`] are RFC-0017's reference sampler, which serving engines link
//! or port. Mixture accounting with the one-sided `log2` bound remains with `#155`.

#![forbid(unsafe_code)]

mod exact;
mod sampler;
mod temperature;

pub use exact::{
    ExactGreedy, ExactRequest, GreedyCertificate, GreedyInputs, GreedyRow, OutputRow, ScoringError,
};
pub use sampler::{
    ClassBits, Sampler, SamplerBaseKey, SamplerError, SamplerKey, StreamClass, inverse_cdf,
};
pub use temperature::{
    ExactTemperature, TemperatureBatch, TemperatureCertificate, TemperatureInputs,
    TemperatureRequest, TemperatureRow,
};
