// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Exact greedy, exact temperature, and mixture request scoring, the reference sampler PRF, and
//! `Û` accounting (RFC-0017, RFC-0026, RFC-0058, RFC-0061).
//!
//! [`ExactGreedy`], [`ExactTemperature`] and [`MixtureScoring`] attach to the compiler's original
//! token, OneHot and LM-head logits sources. [`ExactRequest::witness`],
//! [`TemperatureRequest::witness`] and [`MixtureRequest::witness`] consume one regenerated output
//! row at a time and retain only clearing certificate advice. Exact predictors reuse the recorded
//! indicator `h_{y_t}` (`Û = 0` or `ERR-WITNESS-EXACT-MISMATCH`), whereas [`MixtureScoring`]
//! commits `D`'s decoded indicator `h_{ŷ_t}` alongside `OneSidedLog2` and exposes only the summed
//! per-request bound `Û_r` (`INV-SC-1..3`).
//!
//! The driver supplies the runtime table specifications (`Range`, `Exp_T`, `Pow2`, `Exp2Frac`) and
//! assigns global certificate/gadget identities. It commits the certificate columns, joins lookups,
//! PRF claims and graph/source claims, and batches over the output-row axis before releasing
//! transcript messages. Row access is for table assembly; per-row maxima, exponentials, logits,
//! draws, match bits and token bounds must not be serialized or logged. These components do not
//! return a verifier verdict or define new protocol messages.
//!
//! `PredictorSpec` remains the protocol crate's sibling-owned wire type. Its driver chooses the
//! exact or mixture implementation; no scoring-to-protocol dependency or extra claim-list field is
//! needed.
//!
//! [`Sampler`] and [`inverse_cdf`] are RFC-0017's reference sampler, which serving engines link
//! or port.

#![forbid(unsafe_code)]

mod exact;
mod mixture;
mod sampler;
mod temperature;

pub use exact::{
    ExactGreedy, ExactRequest, GreedyCertificate, GreedyInputs, GreedyRow, OutputRow, ScoringError,
};
pub use mixture::{
    MixtureBatch, MixtureCertificate, MixtureInputs, MixtureRequest, MixtureScoring,
};
pub use sampler::{
    ClassBits, Sampler, SamplerBaseKey, SamplerError, SamplerKey, StreamClass, inverse_cdf,
};
pub use temperature::{
    ExactTemperature, TemperatureBatch, TemperatureCertificate, TemperatureInputs,
    TemperatureRequest, TemperatureRow,
};
pub use tessera_compiler::MixtureDecoding;
