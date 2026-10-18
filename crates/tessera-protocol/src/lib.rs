// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Message types, session state machines, parameter sets, the soundness calculator and audit
//! sampling (RFC-0014, RFC-0016).
//!
//! This version holds the messages of `tessera-protocol/1` and the two live sessions, as
//! RFC-0036 defines them.
//!
//! # Messages
//!
//! Schema version 1 of every message is `tessera-protocol/1` (RFC-0036 'Versions and
//! envelopes'). Each type below is its TCE schema (RFC-0003), field for field in RFC-0036's
//! order, and travels in an RFC-0003 envelope through [`Message`]:
//!
//! | Step | Message | Schema | Holds |
//! |---|---|---|---|
//! | 0 | [`SeedCommit`] | `0x0501` | the session and `c_V = H("coin-commit", s_V)` |
//! | 1 | [`Round1`] and [`ProverSeedShare`] | `0x0510`, `0x0502` | the statement and the advice commitment; `s_P` |
//! | 2 | [`SeedReveal`] | `0x0503` | `s_V` |
//! | 3 | [`Round2Response`] | `0x0511` | the leaf stage, the reduction stages, the combined rows and the scoring claims |
//! | 4 | [`ColumnQueries`] | `0x0512` | `τ_A` advice indices and `τ_W` per opened matrix, in draw order |
//! | 5 | [`Round3Response`] | `0x0513` | the distinct queried columns and their multiproofs |
//! | — | [`NiProof`] | `0x0520` | rounds 1 to 3 and the two proof-of-work nonces, without inner envelopes |
//!
//! The live seed is [`coin_toss`]`(s_V, s_P) = H("coin-toss", s_V ‖ s_P)`, and the verifier's
//! commitment [`coin_commitment`]`(s_V)`.
//!
//! Every decoder checks each length against the limits and the bytes left before it allocates,
//! and refuses any input that is not the one encoding of a value, with the code and offset
//! RFC-0003 'Limits' fixes (INV-TCE-1 to 4). Beyond RFC-0003's rules, the entries of
//! `weight_combined`, of the queries' `weights` and of the answer's `weight_columns` name their
//! matrices by [`WeightMatrix`] id, strictly increasing (RFC-0036 'Matrix ids'): an unknown id,
//! a repeated one or one out of order is `ERR-CODEC-NONCANONICAL` at the entry's first byte,
//! its id. The entry types have no `Decode` of their own, so they decode only inside their
//! message, which checks the order.
//!
//! The types reuse the field elements of `tessera-field` and [`SumcheckProof`] of
//! `tessera-sumcheck`, and take [`StageId`] from `tessera-ir`. `tessera-hash`'s `MultiProof`
//! and `tessera-pcs`'s `CombinedRow` have no TCE encoding, which the orphan rule keeps this
//! crate from giving them, so [`MultiProof`] and [`CombinedRow`] are this crate's wire types.
//! [`PredictorSpec`] is defined here too: `tessera-scoring` is a sibling of this crate in the
//! dependency order (01 §3), not a dependency.
//!
//! # Sessions
//!
//! The live sessions are state machines over typed messages, in the session order of RFC-0014
//! and with the calls of 02 §3:
//!
//! - [`ProverSession`] drives a [`ProverEngine`] through `AwaitCommit`, `AwaitReveal`,
//!   `AwaitQueries` and `Done`. A refused message leaves its state as it was. A reveal that does
//!   not open the commitment aborts it, and so does any failure of the engine, a drop before the
//!   end, or taking the engine back: an abort discards the `Round1`, its advice commitment and
//!   its randomness, so that a prover answers at most one seed per `Round1` (INV-PROTO-1). The
//!   session draws nothing: `s_P` is the engine's, derived from its per-proof secret seed, so
//!   that a prover resuming after a crash sends what it sent (RFC-0018 'Crash consistency').
//! - [`VerifierSession`] runs a [`VerifierChecks`] through `Start`, `AwaitRound1`,
//!   `AwaitRound2`, `AwaitRound3` and `Done`, drawing the session, its share and the column
//!   queries from the operating system's CSPRNG. Its first failed check ends it.
//!
//! The sessions hold no secret-class value (05 §3). The verifier's share stays out of every
//! debug form and error until `SeedReveal` carries it, on the heap, and the session clears it
//! when it lets it go, revealed or not. Errors ([`ProtocolError`]) name messages, matrices,
//! counts, positions and lengths only, never a query's index.
//!
//! # Features
//!
//! - `test-params` accepts the `test-small` parameter set, which certifies nothing. Tests only: a
//!   release build with it fails to compile.

#![forbid(unsafe_code)]

// Test-only features cannot reach a release build (RFC-0001 'Cargo features').
#[cfg(all(feature = "test-params", not(debug_assertions)))]
compile_error!(
    "the `test-params` feature accepts parameter sets that certify nothing; it must not be enabled in a release build (RFC-0001 'Cargo features')"
);

mod error;
mod matrix;
mod message;
mod prover;
mod queries;
mod round1;
mod round2;
mod round3;
mod seed;
mod verifier;

pub use error::{ProtocolError, QueryFault};
pub use matrix::WeightMatrix;
pub use message::{Message, MessageKind};
pub use prover::{ProverEngine, ProverSession, ProverState, SessionError};
pub use queries::{MatrixCodeword, QueryShape};
pub use round1::{
    AdviceCommitment, BatchDescriptor, Decoding, PredictorSpec, PublicStatement, RequestRef, Round1,
};
pub use round2::{
    CombinedRow, FixedBits, LogUpProof, Round2Response, ScoringClaims, StageProof, TreeProof,
    WeightCombinedRows,
};
pub use round3::{
    ColumnQueries, MatrixColumns, MatrixQueries, MultiProof, NiProof, Round3Response,
};
pub use seed::{ProverSeedShare, SeedCommit, SeedReveal, coin_commitment, coin_toss};
/// The protocol version `Round1` carries and both sessions check, the claim list's: 0 until
/// `tessera-protocol/1` is assigned, when the first end-to-end proof verifies, then 1
/// (09 §3, RFC-0036 '`Round1`'). A reader of another version answers `ERR-PROTO-VERSION`.
pub use tessera_ir::PROTOCOL_VERSION;
pub use tessera_ir::StageId;
pub use tessera_sumcheck::SumcheckProof;
pub use verifier::{Ending, VerifierChecks, VerifierSession, VerifierState};
