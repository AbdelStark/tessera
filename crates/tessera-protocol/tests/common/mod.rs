// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared test support: a seeded generator, sample messages, and an engine and checks that
//! record what the sessions ask of them.

#![allow(dead_code)] // each test binary uses part of it

use std::path::PathBuf;

use tessera_codec::Rational;
use tessera_field::{Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_hash::Digest;
use tessera_protocol::{
    AdviceCommitment, BatchDescriptor, ColumnQueries, CombinedRow, Decoding, FixedBits, LogUpProof,
    MatrixCodeword, MatrixColumns, MatrixQueries, MultiProof, PredictorSpec, ProverEngine,
    PublicStatement, QueryShape, RequestRef, Round1, Round2Response, Round3Response, ScoringClaims,
    StageId, StageProof, SumcheckProof, TreeProof, VerifierChecks, WeightCombinedRows,
};

/// SplitMix64 (Steele, Lea and Flood, 2014): reproducible inputs, nothing secret.
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn fp(&mut self) -> Goldilocks {
        loop {
            if let Some(x) = Goldilocks::from_canonical_u64(self.next_u64()) {
                return x;
            }
        }
    }

    pub fn fp2(&mut self) -> GoldilocksExt2 {
        GoldilocksExt2::new(self.fp(), self.fp())
    }

    pub fn fps(&mut self, n: usize) -> Vec<Goldilocks> {
        (0..n).map(|_| self.fp()).collect()
    }

    pub fn fp2s(&mut self, n: usize) -> Vec<GoldilocksExt2> {
        (0..n).map(|_| self.fp2()).collect()
    }

    pub fn digest(&mut self) -> Digest {
        let mut out = [0u8; 32];
        for chunk in out.chunks_exact_mut(8) {
            chunk.copy_from_slice(&self.next_u64().to_le_bytes());
        }
        out
    }

    pub fn session(&mut self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out.copy_from_slice(&self.digest()[..16]);
        out
    }
}

/// The repository's `spec/vectors/`.
pub fn vectors() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors")
}

pub fn bytes(hex: &str) -> Vec<u8> {
    assert!(hex.len().is_multiple_of(2), "odd-length hex {hex:?}");
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex digit"))
        .collect()
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ----- Sample messages ---------------------------------------------------------------------------

/// The matrices the sample proofs open: `Q` and `LmHead`.
pub const MATRICES: [u8; 2] = [1, 8];

/// `τ_A` and `n_A` of the sample proofs.
pub const ADVICE: (u32, u64) = (4, 16);

/// `τ_W` and `n_T` of the sample proofs.
pub const WEIGHTS: (u32, u64) = (3, 8);

pub fn statement(rng: &mut SplitMix64) -> PublicStatement {
    PublicStatement {
        claim_list_digest: rng.digest(),
        weights_root: rng.digest(),
        weights_epoch: 3,
        manifest_digest: rng.digest(),
        batch: BatchDescriptor {
            requests: vec![
                RequestRef {
                    request_id: 11,
                    session_id: rng.session(),
                    prompt_len: 7,
                    output_len: 2,
                },
                RequestRef {
                    request_id: 12,
                    session_id: rng.session(),
                    prompt_len: 5,
                    output_len: 4,
                },
            ],
        },
        predictor: PredictorSpec::Mixture {
            decoding: Decoding::Temperature {
                t: Rational::new(3, 4).expect("a fraction"),
            },
            eps_log2: 6,
        },
    }
}

pub fn advice(rng: &mut SplitMix64) -> AdviceCommitment {
    AdviceCommitment {
        root: rng.digest(),
        layout_digest: rng.digest(),
    }
}

fn row(rng: &mut SplitMix64, k: usize, mask: usize) -> CombinedRow {
    CombinedRow {
        u: rng.fp2s(k),
        mask: rng.fp2s(mask),
    }
}

/// A `Round2Response` opening `matrices`, every list filled.
pub fn round2(rng: &mut SplitMix64, matrices: &[u8]) -> Round2Response {
    Round2Response {
        logup: LogUpProof {
            roots: rng.fp2s(4),
            values: rng.fp2s(1),
        },
        trees: TreeProof {
            levels: vec![
                SumcheckProof {
                    rounds: vec![],
                    final_evals: rng.fp2s(4),
                },
                SumcheckProof {
                    rounds: vec![rng.fp2s(3)],
                    final_evals: rng.fp2s(4),
                },
            ],
            splits: rng.fp2s(2),
        },
        stages: vec![StageProof {
            stage: StageId(1),
            sumcheck: SumcheckProof {
                rounds: vec![rng.fp2s(2), rng.fp2s(2)],
                final_evals: rng.fp2s(3),
            },
            splits: rng.fp2s(1),
        }],
        advice_combined: (0..2).map(|_| row(rng, 4, 2)).collect(),
        weight_combined: matrices
            .iter()
            .map(|&matrix| WeightCombinedRows {
                matrix,
                evaluation: row(rng, 2, 0),
                proximity: row(rng, 2, 0),
            })
            .collect(),
        scoring: ScoringClaims {
            requests: vec![FixedBits(0), FixedBits(700)],
        },
    }
}

/// A `Round3Response` for `matrices`.
pub fn round3(rng: &mut SplitMix64, matrices: &[u8]) -> Round3Response {
    Round3Response {
        advice_columns: (0..3).map(|_| rng.fps(2)).collect(),
        advice_proof: MultiProof {
            siblings: vec![rng.digest(), rng.digest()],
        },
        weight_columns: matrices
            .iter()
            .map(|&matrix| MatrixColumns {
                matrix,
                columns: (0..2).map(|_| rng.fps(2)).collect(),
                proof: MultiProof {
                    siblings: vec![rng.digest()],
                },
            })
            .collect(),
    }
}

/// The sample proofs' query shape for `matrices`.
pub fn shape(matrices: &[u8]) -> QueryShape {
    QueryShape {
        advice_count: ADVICE.0,
        advice_len: ADVICE.1,
        matrix_count: WEIGHTS.0,
        matrices: matrices
            .iter()
            .map(|&matrix| MatrixCodeword {
                matrix,
                len: WEIGHTS.1,
            })
            .collect(),
    }
}

/// Queries of `shape`: in every codeword its last index first, then 0 twice and on, so that
/// an index repeats once there are three.
pub fn queries(shape: &QueryShape) -> ColumnQueries {
    let indices = |count: u32, len: u64| -> Vec<u32> {
        (0..count)
            .map(|i| match i {
                0 => (len - 1) as u32,
                _ => ((u64::from(i) - 1) / 2 % len) as u32,
            })
            .collect()
    };
    ColumnQueries {
        advice: indices(shape.advice_count, shape.advice_len),
        weights: shape
            .matrices
            .iter()
            .map(|m| MatrixQueries {
                matrix: m.matrix,
                indices: indices(shape.matrix_count, m.len),
            })
            .collect(),
    }
}

// ----- An engine and checks that record their calls ----------------------------------------------

/// The step of an engine or of the checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Commit,
    Share,
    Prove,
    Answer,
    Round1,
    Round2,
    Round3,
}

/// What the session asked the engine, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EngineCall {
    Commit([u8; 16]),
    Share,
    Prove(Box<Round1>, Digest),
    Answer(ColumnQueries),
    Discard,
}

/// An engine with canned messages, failing at `fail` if it is set.
///
/// Its share derives from `per_proof_seed`, as RFC-0018 'Crash consistency' derives it from the
/// per-proof secret seed: two engines with the same seed give the same share, as a prover that
/// resumes after a crash does. `discard` draws a new seed, as an abort discards the old one
/// (INV-PROTO-1).
pub struct Engine {
    pub statement: PublicStatement,
    pub advice: AdviceCommitment,
    pub round2: Round2Response,
    pub shape: QueryShape,
    pub round3: Round3Response,
    pub per_proof_seed: u64,
    pub fail: Option<Step>,
    pub calls: Vec<EngineCall>,
}

/// The engine's failure, at its step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EngineError(pub Step);

impl Engine {
    pub fn new(seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        Self {
            statement: statement(&mut rng),
            advice: advice(&mut rng),
            round2: round2(&mut rng, &MATRICES),
            shape: shape(&MATRICES),
            round3: round3(&mut rng, &MATRICES),
            per_proof_seed: rng.next_u64(),
            fail: None,
            calls: Vec::new(),
        }
    }

    /// The share this engine gives for its current per-proof seed.
    pub fn share(&self) -> [u8; 32] {
        SplitMix64::new(self.per_proof_seed ^ 0x5350_5f73_6861_7265).digest()
    }

    fn step(&self, step: Step) -> Result<(), EngineError> {
        match self.fail {
            Some(fail) if fail == step => Err(EngineError(step)),
            _ => Ok(()),
        }
    }

    /// The calls of a kind.
    pub fn count(&self, kind: fn(&EngineCall) -> bool) -> usize {
        self.calls.iter().filter(|call| kind(call)).count()
    }
}

impl ProverEngine for Engine {
    type Error = EngineError;

    fn commit(
        &mut self,
        session_id: &[u8; 16],
    ) -> Result<(PublicStatement, AdviceCommitment), EngineError> {
        self.calls.push(EngineCall::Commit(*session_id));
        self.step(Step::Commit)?;
        Ok((self.statement.clone(), self.advice))
    }

    fn prover_share(&mut self) -> Result<[u8; 32], EngineError> {
        self.calls.push(EngineCall::Share);
        self.step(Step::Share)?;
        Ok(self.share())
    }

    fn prove(
        &mut self,
        round1: &Round1,
        seed: &Digest,
    ) -> Result<(Round2Response, QueryShape), EngineError> {
        self.calls
            .push(EngineCall::Prove(Box::new(round1.clone()), *seed));
        self.step(Step::Prove)?;
        Ok((self.round2.clone(), self.shape.clone()))
    }

    fn answer(&mut self, queries: &ColumnQueries) -> Result<Round3Response, EngineError> {
        self.calls.push(EngineCall::Answer(queries.clone()));
        self.step(Step::Answer)?;
        Ok(self.round3.clone())
    }

    fn discard(&mut self) {
        self.calls.push(EngineCall::Discard);
        self.per_proof_seed = SplitMix64::new(self.per_proof_seed).next_u64();
    }
}

/// What the session asked the checks, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckCall {
    Round1(Box<Round1>, Digest),
    Round2(Box<Round2Response>),
    Queries(ColumnQueries),
    Round3(Box<Round3Response>),
}

/// A verdict of the test checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Accept,
    Reject(Step),
}

/// Checks that accept everything but the step `reject` names, and derive `shape`.
pub struct Checks {
    pub shape: QueryShape,
    pub reject: Option<Step>,
    pub calls: Vec<CheckCall>,
}

impl Checks {
    pub fn new() -> Self {
        Self {
            shape: shape(&MATRICES),
            reject: None,
            calls: Vec::new(),
        }
    }

    fn verdict(&self, step: Step) -> Result<(), Verdict> {
        match self.reject {
            Some(reject) if reject == step => Err(Verdict::Reject(step)),
            _ => Ok(()),
        }
    }
}

impl VerifierChecks for Checks {
    type Verdict = Verdict;

    fn round1(&mut self, round1: &Round1, seed: &Digest) -> Result<(), Verdict> {
        self.calls
            .push(CheckCall::Round1(Box::new(round1.clone()), *seed));
        self.verdict(Step::Round1)
    }

    fn round2(&mut self, round2: &Round2Response) -> Result<QueryShape, Verdict> {
        self.calls.push(CheckCall::Round2(Box::new(round2.clone())));
        self.verdict(Step::Round2)?;
        Ok(self.shape.clone())
    }

    fn queries(&mut self, queries: &ColumnQueries) {
        self.calls.push(CheckCall::Queries(queries.clone()));
    }

    fn round3(&mut self, round3: &Round3Response) -> Verdict {
        self.calls.push(CheckCall::Round3(Box::new(round3.clone())));
        match self.verdict(Step::Round3) {
            Ok(()) => Verdict::Accept,
            Err(verdict) => verdict,
        }
    }
}
