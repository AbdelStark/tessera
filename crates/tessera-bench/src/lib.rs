// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The executable cost model, unit-cost microbenchmarks and their reports (RFC-0020).
//!
//! This version holds the cost model and the CPU microbenchmarks' harness. Every projection of the
//! proof multiplier `μ` becomes code that reproduces the source's tables and takes measured
//! inputs (RFC-0020 'Executable cost model'), and every microbenchmark is measured the way RFC-0020
//! measures unit costs. The `tessera.unit-costs/1` report follows with the H100 measurements.
//!
//! - [`mu`] evaluates 08 §1 term by term: the linear layers' commitments and lookups, the
//!   attention scores' commitments, lookups and product-tree leaves, the shared weight pass,
//!   the witness regenerations of RISK-9, the leaf stage's memory plan of RISK-10 and RFC-0054,
//!   the column answers of RFC-0018 P5, the opening pass's row combination (P4) and the table
//!   side of LogUp, each over the native time of a token. [`leaf_plan`] gives the leaf stage's
//!   plan: its token groups and the table trees' streamed rounds. [`stage_plan`] gives each
//!   reduction stage's working set and window (RFC-0055). [`shared_streaming_passes`] counts the
//!   passes of the design RFC-0054 replaces.
//! - [`proof_time_s`], [`sampling_rate`] and [`detection_probability`] turn `μ` into the
//!   columns of the source's §8: the time to prove a request, the sampling rate for an amortized
//!   overhead, and the daily detection probability of violating requests (RFC-0016
//!   'Calculators').
//! - [`ModelReport`] gathers them at several contexts and writes the `tessera.cost-model/1`
//!   report of `tessera bench model`.
//! - [`measure`] runs a benchmark with RFC-0020's protocol, warm-up runs then measured runs, and
//!   returns their [`Statistics`]: median, extremes and coefficient of variation, flagged above
//!   [`NOISY_CV`]. [`MicrobenchmarkReport`] stores a run of the `primitives` benchmark as the
//!   `tessera.microbenchmarks/1` baseline of 08 §6's regression gate (`docs/perf/`).
//!
//! [`Counts::SOURCE_405B`], [`UnitCosts::SOURCE`] and [`Scenario::SOURCE`] are the source's
//! inputs. With them, which regenerate nothing, merge, stream and fold at no cost and price
//! neither the column answers nor the row combination, the
//! model reproduces the source's chart (source §5) within ±0.05, and its calculators reproduce
//! §8 within 0.1 percentage point from that table's `μ`: `spec/vectors/cost-model/` holds both,
//! each figure with its line.
//!
//! ```
//! use tessera_bench::{ModelReport, SOURCE_CONTEXTS};
//!
//! let report = ModelReport::source(&SOURCE_CONTEXTS)?;
//! let at_1k = &report.projections[0];
//! // The source projects μ = 3.2 at 1K, and a sampling rate of 10.4% (source §8).
//! assert!((at_1k.mu.total - 3.21).abs() < 0.05);
//! assert!((100.0 * at_1k.sampling_rate - 10.4).abs() < 0.1);
//! assert!(report.to_json()?.contains("\"schema\": \"tessera.cost-model/1\""));
//! # Ok::<(), tessera_bench::ModelError>(())
//! ```
//!
//! Floating point is acceptable here: the model plans, and nothing it computes enters the
//! protocol. Every input is checked to be finite and in range, and every output to be finite:
//! an input the model cannot price is an error, never a NaN in a report.

#![forbid(unsafe_code)]

mod error;
mod measure;
mod model;
mod report;

pub use error::{ModelError, Problem};
pub use measure::{
    MICROBENCHMARKS_SCHEMA, Measurement, MicrobenchmarkReport, NOISY_CV, Protocol, Statistics,
    measure,
};
pub use model::{
    Counts, LeafPlan, MAX_LEAF_GROUPS, MemoryModel, MuBreakdown, ReductionStage,
    STAGE_BYTES_PER_ENTRY, StageTable, StageWindow, TENSOR_BYTES_PER_LEAF, UnitCosts,
    detection_probability, leaf_plan, mu, proof_time_s, sampling_rate, shared_streaming_passes,
    stage_plan,
};
pub use report::{ModelReport, Projection, SCHEMA, SOURCE_CONTEXTS, Scenario};
