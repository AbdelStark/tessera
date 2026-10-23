// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Message types, session state machines, parameter sets, the soundness calculator and audit
//! sampling (RFC-0014, RFC-0016).
//!
//! This version holds the messages of `tessera-protocol/1` and the two live sessions, as
//! RFC-0036 defines them, the parameter sets, the soundness calculator and the transcript
//! schedule, as RFC-0014 does.
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
//! # Parameter sets
//!
//! [`ParamSet`] holds RFC-0014's three sets, `live-64/1`, `ni-100/1` and `test-small/1`, each
//! with the commitments' layout its claim list states (RFC-0026 'Schema'), its query counts, its
//! proofs of work, whether a release verifier accepts it and the targets its soundness must reach.
//! [`ParamSet::accept`] takes a set from its identifier, as a proof must (INV-PROTO-5), and
//! refuses `test-small/1` without the `test-params` feature.
//!
//! # The soundness calculator
//!
//! [`SoundnessReport`] gives every row of RFC-0014's soundness table for a parameter set at a
//! statement's [`StatementSizes`]: its exact error bound, its bits `λ = g + max(0, −log2 ε)` as
//! a [`FixedBits`] lower bound `r` with `r ≤ 256·λ < r + 1 + 2^{-20}`, its target and whether it
//! reaches it, then the minimum. The arithmetic is integers only, as RFC-0014 'Soundness
//! calculator' specifies it, so that the verifier's independent calculator agrees bit for bit,
//! and every decision compares integers, never rounded bits. [`StatementSizes::llama_405b_16k`]
//! is the 405B 16K-token, 16K-context statement, each of its sizes documented with its source.
//!
//! # The transcript schedule
//!
//! [`LabelPattern`] is the registry of RFC-0014's tier-A label patterns of `tessera-protocol/1`,
//! and a [`Label`] one of them with its indices: both sides build every label from it.
//! [`TranscriptSchedule::expand`] lists every label one proof unit absorbs or squeezes under, in
//! order, from the [`ScheduleSizes`] that public data fixes, in a mode at a parameter set: no label
//! twice (INV-FS-4), and every prover message absorbed before the challenges that depend on it
//! (INV-FS-1). [`WEIGHTS_XI`] is the one place that puts each weight matrix's proximity point,
//! right after the seed as RFC-0049 moves it.
//!
//! [`ScheduledTranscript`] holds RFC-0004's transcript to a schedule, each operation the next
//! step. A live session absorbs the coin-tossed seed and the verifier's queries at their places; a
//! non-interactive prover grinds for the seed and the queries under `seed-pow` and `query-pow`,
//! for the parameter set's `g_seed` and `g_query` bits, and a verifier counts each nonce's leading
//! zero bits exactly, refusing a short one with [`PowRejected`], `Reject(ProofOfWork)`.
//! `tessera-protocol/2`'s labels slot in at [`Protocol::Two`] (#148).
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
mod params;
mod prover;
mod queries;
mod round1;
mod round2;
mod round3;
mod schedule;
mod seed;
mod soundness;
mod verifier;

pub use error::{ProtocolError, QueryFault};
pub use matrix::WeightMatrix;
pub use message::{Message, MessageKind};
pub use params::{ParamMode, ParamSet, Targets};
pub use prover::{ProverEngine, ProverSession, ProverState, SessionError};
pub use queries::{MatrixCodeword, QueryShape};
pub use round1::{
    AdviceCommitment, BatchDescriptor, Decoding, LeafGroups, MAX_LEAF_GROUP_BITS, PredictorSpec,
    PublicStatement, RequestRef, Round1,
};
pub use round2::{
    CombinedRow, FixedBits, GroupProof, LogUpProof, Round2Response, ScoringClaims, StageProof,
    TreeProof, WeightCombinedRows,
};
pub use round3::{
    ColumnQueries, MatrixColumns, MatrixQueries, MultiProof, NiProof, Round3Response,
};
pub use schedule::{
    Label, LabelPattern, LeafTree, MatrixPoint, Op, PowRejected, Protocol, ScheduleFault,
    ScheduleSizes, ScheduledTranscript, StageSizes, Step, TranscriptSchedule, WEIGHTS_XI,
    WeightsXi,
};
pub use seed::{ProverSeedShare, SeedCommit, SeedReveal, coin_commitment, coin_toss};
pub use soundness::{
    Commitment, ErrorBound, LookupSizes, MatrixSizes, ProofOfWork, Rounds, RowKind,
    SoundnessReport, SoundnessRow, StatementSizes,
};
/// The protocol version `Round1` carries and both sessions check, the claim list's: 0 until
/// `tessera-protocol/1` is assigned, when the first end-to-end proof verifies, then 1
/// (09 §3, RFC-0036 '`Round1`'). A reader of another version answers `ERR-PROTO-VERSION`.
pub use tessera_ir::PROTOCOL_VERSION;
pub use tessera_ir::StageId;
pub use tessera_sumcheck::SumcheckProof;
pub use verifier::{Ending, VerifierChecks, VerifierSession, VerifierState};
