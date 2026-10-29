// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Protocol/1 CPU driver for exact-greedy request-slot batches (RFC-0071, 02 section 3.1).

use std::num::{NonZeroU32, NonZeroUsize};
use std::time::Instant;
use tessera_codec::{Ascii, Public, Secret, error::ErrorCode, event, metrics, span};
use tessera_compiler::{self as compiler, ExecutionPlan, PublicModel};
use tessera_hash::{SessionContext, Transcript};
use tessera_ir::IntegerModel;
use tessera_pcs as pcs;
use tessera_protocol::{self as protocol, ProverEngine};

mod backend;
mod error;
mod plan;
mod random;
pub use error::ProverError;
pub use plan::{Cells, Pass, PassPlan};
pub use tessera_hash::Mode;
pub use tessera_protocol::ParamSet;

/// The existing session/backend boundary, shared with the live protocol FSM (RFC-0036).
pub use tessera_protocol::ProverEngine as ProverBackend;

/// Execution target. This reference implementation accepts CPU only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Backend {
    /// The exact CPU witness engine; zero threads is a usage error.
    Cpu {
        /// Worker count.
        threads: usize,
    },
    /// Reserved GPU target; rejected explicitly by this implementation.
    Gpu {
        /// Device ordinals.
        devices: Vec<u32>,
    },
}

/// Host allocation allowance, in bytes. CPU proving has no device allocations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryBudget {
    /// Allowance for resident CPU proof data, excluding the caller's model.
    pub host: u64,
}

/// Per-proof masking seed source. Production always uses the operating system.
#[derive(Clone, Copy)]
pub enum MaskRng {
    /// Fresh operating-system entropy.
    OsRng,
    /// Reproducible test seed; cannot compile in a release build.
    #[cfg(feature = "test-rng")]
    Seeded(u64),
}
impl core::fmt::Debug for MaskRng {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("MaskRng([redacted])")
    }
}

/// Protocol/1 releases completed messages immediately and provides no timing hiding.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReleaseConfig;

/// Configuration checked against the complete named parameter bundle and execution plan.
#[derive(Clone, Debug)]
pub struct ProverConfig {
    /// Execution target.
    pub backend: Backend,
    /// Named RFC-0014 parameter bundle; hypothetical calculator parameters are refused.
    pub params: ParamSet,
    /// Live or non-interactive transcript.
    pub mode: Mode,
    /// Host memory allowance, checked before P1.
    pub memory_budget: MemoryBudget,
    /// Private per-proof entropy source.
    pub mask_rng: MaskRng,
    /// Protocol/1 release behavior.
    pub release: ReleaseConfig,
}

/// One caller-owned exact-greedy batch. Tokens never enter Debug or errors.
///
/// The tokens concatenate each prompt and recorded continuation in request order.
/// The backend inserts RFC-0071's zero slot gaps; callers supply only real tokens.
pub struct ProofBatch {
    /// Public expected statement, including the committed epoch and request lengths.
    pub statement: protocol::PublicStatement,
    /// Each prompt followed by its recorded continuation, concatenated in request order.
    pub tokens: Secret<Vec<u32>>,
}
impl core::fmt::Debug for ProofBatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ProofBatch")
            .field("statement", &self.statement)
            .finish_non_exhaustive()
    }
}

struct Prepared<'a> {
    plan: &'a ExecutionPlan,
    model: &'a IntegerModel,
    _state: &'a mut pcs::WeightCommitmentState,
    cfg: ProverConfig,
    threads: NonZeroUsize,
    layout: pcs::WeightLayout,
    weights: pcs::WeightCommitment,
}

/// A one-shot CPU proof session. Live refusals use the existing RFC-0036 state machine.
/// Dropping or aborting an unfinished session discards its witness and private randomness.
pub struct Prover<'a> {
    prepared: Option<Prepared<'a>>,
    session: Option<protocol::ProverSession<backend::CpuBackend<'a>>>,
    mode: Mode,
    /// The batch's statement digest, the `prove` span's public field (05 §5), once started.
    batch: Option<String>,
}
impl core::fmt::Debug for Prover<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Prover")
            .field("mode", &self.mode)
            .field("state", &self.session.as_ref().map(|s| s.state()))
            .finish_non_exhaustive()
    }
}

/// The `prove{batch}` span of 05 §5: the batch's statement digest, public.
fn prove_span(batch: Option<&str>) -> tessera_codec::telemetry::Span {
    span!(INFO, "prove", batch = Public(batch.unwrap_or("")))
}

fn statement_hex(batch: &ProofBatch) -> String {
    batch
        .statement
        .digest()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn usage(rule: &'static str) -> ProverError {
    ProverError::Usage { rule }
}
fn order(message: protocol::MessageKind) -> ProverError {
    ProverError::Protocol(protocol::ProtocolError::OutOfOrder { message })
}
fn internal(rule: &'static str) -> ProverError {
    ProverError::Protocol(protocol::ProtocolError::Internal { rule })
}

impl<'a> Prover<'a> {
    /// Validate the plan, named parameters and actual weight commitment before any session.
    /// The mutable state borrow reserves this epoch for the session; protocol/1 burns no
    /// hiding budget. The caller's state is unchanged on every path.
    pub fn new(
        plan: &'a ExecutionPlan,
        model: &'a IntegerModel,
        weights: &'a mut pcs::WeightCommitmentState,
        cfg: ProverConfig,
    ) -> Result<Self, ProverError> {
        let threads = match cfg.backend {
            Backend::Cpu { threads } => {
                NonZeroUsize::new(threads).ok_or_else(|| usage("CPU threads must be nonzero"))?
            }
            Backend::Gpu { .. } => return Err(usage("the CPU driver cannot use a GPU backend")),
        };
        if ParamSet::defined(&plan.claims.params) != Some(&cfg.params)
            || plan.claims.layout != cfg.params.layout
            || !cfg.params.mode.allows(cfg.mode)
        {
            return Err(usage(
                "configuration must match the named claim-list parameters and mode",
            ));
        }
        compiler::validate(model)?;
        compiler::check_plan(plan, PublicModel::of(model), &plan.claims)?;
        backend::check_weight_memory(&plan.claims, cfg.memory_budget)?;
        let lowering = compiler::lower(PublicModel::of(model), 1)?;
        let source = backend::model_weights(model, &lowering)?;
        let layout = pcs::WeightLayout::new(
            plan.claims.layout.weights,
            pcs::WeightShapes {
                layers: plan.claims.arch.n_layers,
                shapes: plan
                    .claims
                    .arch
                    .weight_shapes()
                    .map_err(|_| usage("invalid weight shapes"))?,
                channels: source
                    .channels
                    .iter()
                    .map(|c| (c.matrix, c.layer))
                    .collect(),
            },
        )?;
        let actual = pcs::WeightCommitment::commit(
            &layout,
            NonZeroU32::new(plan.claims.layout.advice.row_block)
                .ok_or_else(|| usage("zero commitment row block"))?,
            weights.public.epoch,
            &source,
        )?;
        if actual.public() != &weights.public
            || weights
                .budget
                .iter()
                .any(|b| !b.opened_columns.columns().is_empty() || b.next_blinding_row != 0)
        {
            return Err(usage(
                "weight state must match this model's protocol/1 commitment and empty budget",
            ));
        }
        Ok(Self {
            mode: cfg.mode,
            batch: None,
            prepared: Some(Prepared {
                plan,
                model,
                _state: weights,
                cfg,
                threads,
                layout,
                weights: actual,
            }),
            session: None,
        })
    }

    /// The pass plan of a proof of `batch`, from public data alone: its passes, its resident
    /// cells and bytes (RFC-0018 'Pass structure', 'Memory scheduler'). Proving fails with
    /// `ERR-RES-MEMORY` before P1 when the bytes exceed `ProverConfig.memory_budget`. The plan is
    /// available until the session starts.
    pub fn plan(&self, batch: &ProofBatch) -> Result<PassPlan, ProverError> {
        let prepared = self
            .prepared
            .as_ref()
            .ok_or_else(|| order(protocol::MessageKind::SeedCommit))?;
        backend::CpuBackend::unit(prepared, batch).map(|(_, plan)| plan)
    }

    /// Replay the exact-greedy witness within the configured budget, without a commitment,
    /// randomness, weight-opening budget changes or session transition (RFC-0071).
    ///
    /// # Errors
    /// The same public metadata, memory and witness errors as proving, including
    /// `ERR-WITNESS-EXACT-MISMATCH` for a recorded continuation that differs from D.
    pub fn check_witness(&self, batch: &ProofBatch) -> Result<(), ProverError> {
        let prepared = self
            .prepared
            .as_ref()
            .ok_or_else(|| order(protocol::MessageKind::SeedCommit))?;
        backend::CpuBackend::check_witness(prepared, batch)
    }

    /// Commit P1 and return the public first round and the per-proof coin-toss share.
    /// The batch is borrowed only while P1 computes the clearing witness; it may then be dropped.
    pub fn start(
        &mut self,
        batch: &ProofBatch,
        commit: &protocol::SeedCommit,
    ) -> Result<(protocol::Round1, protocol::ProverSeedShare), ProverError> {
        if self.mode != Mode::Live || self.prepared.is_none() {
            return Err(order(protocol::MessageKind::SeedCommit));
        }
        self.batch = Some(statement_hex(batch));
        let _prove = prove_span(self.batch.as_deref()).entered();
        let prepared = self
            .prepared
            .take()
            .ok_or_else(|| internal("unstarted CPU backend"))?;
        let mut session = protocol::ProverSession::new(backend::CpuBackend::new(prepared, batch)?);
        let answer = session.start(commit).map_err(Into::into);
        if let Ok((round1, _)) = &answer {
            metrics::proof_bytes(
                metrics::ProofPart::Round1,
                tessera_codec::encode(round1).len() as u64,
            );
        }
        self.session = Some(session);
        answer
    }

    /// Answer the one valid seed reveal; invalid-session and phase refusals preserve state.
    pub fn on_seed(
        &mut self,
        reveal: &protocol::SeedReveal,
    ) -> Result<protocol::Round2Response, ProverError> {
        let _prove = prove_span(self.batch.as_deref()).entered();
        self.session
            .as_mut()
            .ok_or_else(|| order(protocol::MessageKind::SeedReveal))?
            .on_seed(reveal)
            .map_err(Into::into)
    }

    /// Answer checked column indices once, then clear the per-proof witness and randomness.
    pub fn on_queries(
        &mut self,
        queries: &protocol::ColumnQueries,
    ) -> Result<protocol::Round3Response, ProverError> {
        let _prove = prove_span(self.batch.as_deref()).entered();
        self.session
            .as_mut()
            .ok_or_else(|| order(protocol::MessageKind::ColumnQueries))?
            .on_queries(queries)
            .map_err(Into::into)
    }

    /// Run the same P1–P5 computations with the non-interactive grinding/query schedule.
    /// A session is consumed on success or failure; retry with a new prover and fresh seed.
    pub fn prove_non_interactive(
        &mut self,
        batch: &ProofBatch,
    ) -> Result<protocol::NiProof, ProverError> {
        if self.mode != Mode::NonInteractive || self.prepared.is_none() {
            return Err(order(protocol::MessageKind::SeedCommit));
        }
        self.batch = Some(statement_hex(batch));
        let _prove = prove_span(self.batch.as_deref()).entered();
        let prepared = self
            .prepared
            .take()
            .ok_or_else(|| internal("unstarted CPU backend"))?;
        let params = prepared.cfg.params;
        let mut backend = backend::CpuBackend::new(prepared, batch)?;
        let session_id = backend.session()?;
        let (statement, advice) = backend.commit(&session_id)?;
        let round1 = protocol::Round1 {
            protocol_version: protocol::PROTOCOL_VERSION,
            session_id,
            statement,
            advice,
            leaf_groups: backend.leaf_groups(),
        };
        metrics::proof_bytes(
            metrics::ProofPart::Round1,
            tessera_codec::encode(&round1).len() as u64,
        );
        let mut tr = transcript(Mode::NonInteractive, params, &round1)?;
        let seed_nonce = tr.grind("seed-pow", params.seed_pow as u8);
        let _seed = tr.squeeze("seed");
        let (round2, shape) = backend.round2(&mut tr)?;
        let query_nonce = tr.grind("query-pow", params.query_pow as u8);
        let queries = protocol::ColumnQueries {
            advice: tr
                .challenge_indices(
                    "pcs/advice/queries",
                    shape.advice_len.trailing_zeros(),
                    shape.advice_count,
                )
                .into_iter()
                .map(|i| i as u32)
                .collect(),
            weights: shape
                .matrices
                .iter()
                .map(|m| protocol::MatrixQueries {
                    matrix: m.matrix,
                    indices: tr
                        .challenge_indices(
                            &format!(
                                "pcs/weights/{:?}/queries",
                                pcs::WeightMatrix::ALL[m.matrix as usize]
                            ),
                            m.len.trailing_zeros(),
                            shape.matrix_count,
                        )
                        .into_iter()
                        .map(|i| i as u32)
                        .collect(),
                })
                .collect(),
        };
        shape
            .check(&queries)
            .map_err(|e| ProverError::Protocol(protocol::ProtocolError::Queries(e)))?;
        let round3 = backend.answer(&queries)?;
        event!(DEBUG, "prover.proof_completed", mode = "non-interactive"; "proof completed");
        Ok(protocol::NiProof {
            round1,
            seed_nonce,
            round2,
            query_nonce,
            round3,
        })
    }

    /// End an unfinished session, clearing per-proof data. A later call cannot reuse P1.
    pub fn abort(&mut self) {
        self.prepared = None;
        if let Some(session) = &mut self.session {
            session.abort();
        }
    }
}

fn transcript(
    mode: Mode,
    params: ParamSet,
    round1: &protocol::Round1,
) -> Result<Transcript, ProverError> {
    let mut tr = Transcript::new(&SessionContext {
        protocol: Ascii::new("tessera-protocol/1").map_err(|_| internal("protocol ASCII"))?,
        mode,
        params: params.param_set_id(),
        statement_digest: round1.statement.digest(),
        session_id: round1.session_id,
    });
    tr.absorb("round1", round1);
    Ok(tr)
}
