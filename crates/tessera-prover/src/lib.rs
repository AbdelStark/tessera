// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Orchestration of the witness pass, commitments, stages and openings over the CPU and GPU
//! backends (RFC-0018).
//!
//! # Non-interactive CPU API
//!
//! The caller supplies a checked plan, its integer model and committed weight state,
//! a matching configuration, and recorded tokens in a [`ProofBatch`]. A prover is
//! one-shot: after success or failure, start a fresh session rather than retrying
//! with the same commitment and randomness (INV-PROTO-1). The current protocol/1
//! CPU path is exact-greedy and provides no zero knowledge.
//!
//! ```no_run
//! use tessera_compiler::ExecutionPlan;
//! use tessera_ir::IntegerModel;
//! use tessera_pcs::WeightCommitmentState;
//! use tessera_protocol::Message;
//! use tessera_prover::{ProofBatch, Prover, ProverConfig, ProverError};
//!
//! fn prove(
//!     plan: &ExecutionPlan,
//!     model: &IntegerModel,
//!     weights: &mut WeightCommitmentState,
//!     config: ProverConfig,
//!     batch: &ProofBatch,
//! ) -> Result<Vec<u8>, ProverError> {
//!     let mut prover = Prover::new(plan, model, weights, config)?;
//!     let proof = prover.prove_non_interactive(batch)?;
//!     Ok(proof.seal())
//! }
//! ```
//!
//! [`StageProver`] proves a stage schedule's reduction stages over a [`WitnessSource`]: each
//! stage's batched sumcheck, its split values and the claim pool's values (RFC-0010 'Stages',
//! RFC-0037). [`PaddedWitness`] gives a source the padded positions of the tensors a lookup
//! non-zero at padding reaches, such as AES's after its S-box (RFC-0072 'Padded positions').
//!
//! [`leaf`] lays out the leaf stage's lookups (RFC-0045): each LogUp instance's column of blocks,
//! its lookup tree's leaf order, the leaf point `P`, and the split values that turn its leaf
//! claims into one claim per tensor a site's lookups read. [`prove_leaf`] proves the LogUp part
//! and the split on a transcript. [`prepare_leaf`] batches product forests beside the lookup
//! trees and binds lookup values and graph atom splits in one message via [`PreparedLeaf`].
//!
//! [`ReleaseSchedule`] fixes when a message that a verifier can time leaves the prover in
//! `tessera-protocol/2`: at a slot that public data determines, whatever its secret-dependent
//! work took (RFC-0018 'Release schedule', INV-ARCH-2).
//!
//! [`derive_model_schedule`] derives complete protocol/1 transcript dimensions from public
//! claims, including every native gadget lookup, forest and reduction path.
//! [`derive_runtime_schedule`] uses the complete exact-greedy runtime layout and composed
//! graph/native plan, including every shared table and original weight obligation.
//! [`leaf::LeafLayout::model`] and [`leaf::ModelWitness`] supply the canonical CPU lookup
//! sites, per-request domains, clearing source buffers and stacked Softmax forests. Their
//! leaf proofs leave the native/graph reductions and commitment openings to the driver.
//! [`native::ModelNativePlan`] and [`native::ModelNativeProver`] supply every current-model
//! native relation to the same graph stage batch, retaining the canonical opening debts.
//! [`advice::visit_rows`] lends the canonical witness rows to the existing P1, P4 and P5
//! commitment passes, including public padding and bounded multiplicity chunks.
//!
//! # Features
//!
//! - `test-params` enables non-certifying parameters in the independent verifier dev-dependency.
//!   Tests only: a release build with it fails to compile.
//! - `test-rng` seeds masking randomness so that tests are reproducible. Tests only: a release
//!   build with it fails to compile.
//! - `unsafe-debug-dump` dumps secret tensors for debugging (OBS-4). Tests only: a release build
//!   with it fails to compile.

#![forbid(unsafe_code)]

pub mod advice;
pub mod cpu;
pub mod leaf;
mod model;
pub mod native;
pub mod openings;
mod release;
pub mod stages;

pub use cpu::{
    Backend, Cells, MaskRng, MemoryBudget, Mode, ParamSet, Pass, PassPlan, ProofBatch, Prover,
    ProverBackend, ProverConfig, ProverError, ReleaseConfig,
};
pub use leaf::{LeafError, LeafLayout, PreparedLeaf, ProvenLeaf, prepare_leaf, prove_leaf};
pub use model::{ModelSchedule, derive_model_schedule, derive_runtime_schedule};
pub use openings::{
    GraphOpenings, ModelOpenings, OpeningGeometry, OpeningTarget, OpeningTargetMemory,
};
pub use release::{Release, ReleaseSchedule, ScheduleError};
pub use stages::{
    PaddedWitness, PublicTensors, StageError, StageProof, StageProver, StageSourceMemory,
    StageStateMemory, StageWorkMemory, WitnessSource, public_value,
};

// Test-only features cannot reach a release build (RFC-0001 'Cargo features').
#[cfg(all(feature = "test-params", not(debug_assertions)))]
compile_error!(
    "the `test-params` feature enables non-certifying verifier test parameters; it must not be enabled in a release build (RFC-0001 'Cargo features')"
);
#[cfg(all(feature = "test-rng", not(debug_assertions)))]
compile_error!(
    "the `test-rng` feature pins masking randomness to a seed; it must not be enabled in a release build (RFC-0001 'Cargo features')"
);
#[cfg(all(feature = "unsafe-debug-dump", not(debug_assertions)))]
compile_error!(
    "the `unsafe-debug-dump` feature dumps secret tensors; it must not be enabled in a release build (OBS-4, RFC-0001 'Cargo features')"
);
