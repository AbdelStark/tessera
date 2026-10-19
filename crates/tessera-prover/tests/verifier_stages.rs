// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The conformance models' reduction stages, proven by `tessera-prover` over the CPU engine's
//! witness as #113's tests prove them (`tests/stages.rs`), and checked by the independent
//! verifier (DEP-2): `tv-core` lowers each model's golden claim list from its fields, derives the
//! stage schedule itself, and checks the leaf stage's splits and every reduction stage
//! (`tv_core::stages::verify`, #114).
//!
//! - **Honest proofs verify.** The verifier's pool equals the prover's, claim by claim, and every
//!   pool claim the verifier returns, a combination of committed tensors at one point, holds of
//!   the witness.
//! - **Changed proofs are rejected with their exact reason.** A changed round value or final
//!   evaluation of stage `k` is `Reject(Sumcheck { stage: k, round: n_k })`, at the stage's final
//!   check (RFC-0010 'Sumcheck messages'); a round one value short is rejected at its round; a
//!   changed split value of the leaf stage is `Reject(Sumcheck { stage: 0, round: 0 })` (RFC-0037
//!   'Splits'); and a false leaf value is rejected at stage 1's final check.
//!
//! The leaf stage is the gadgets' (#118, #130): its claims are true values at random points,
//! bound under a test label before the leaf stage's splits on both sides, as #113's tests bind
//! them.

use std::fs;
use std::path::{Path, PathBuf};

use tessera_codec::{Ascii, DecodeLimits, Secret, envelope, schema};
use tessera_compiler::schedule::StageSchedule;
use tessera_compiler::{Lowering, PublicModel, derive_schedule, lower};
use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_ir::{IntegerModel, TensorId};
use tessera_poly::evaluate_base;
use tessera_prover::{StageProof, StageProver};
use tessera_sumcheck::SumcheckTranscript;
use tessera_witness::{Engine, Witness};
use tv_core::lowering::Role;
use tv_core::schedule::Schedule;
use tv_core::stages::{
    LeafValues, PoolClaim, PoolTensor, StageError, SumcheckRejection, Verified, verify,
};

type F = GoldilocksExt2;
type TvFp = tv_field::Fp;
type TvFp2 = tv_field::Fp2;

const MODELS: [&str; 4] = [
    "tiny-2-servable",
    "tiny-2-minimal",
    "tiny-2-affine",
    "tiny-4-servable",
];

fn vectors() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors")
}

fn json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&fs::read_to_string(path).expect("the vectors")).expect("JSON")
}

// ----- Field conversions ----------------------------------------------------------------------

fn tv_fp2(x: F) -> TvFp2 {
    let [c0, c1] = x.coeffs();
    TvFp2::new(
        TvFp::from_canonical(c0.to_canonical_u64()).expect("canonical"),
        TvFp::from_canonical(c1.to_canonical_u64()).expect("canonical"),
    )
}

fn tv_fp2s(xs: &[F]) -> Vec<TvFp2> {
    xs.iter().copied().map(tv_fp2).collect()
}

fn from_tv(x: TvFp2) -> F {
    let bytes = x.to_le_bytes();
    let word = |i: usize| u64::from_le_bytes(bytes[8 * i..8 * i + 8].try_into().expect("8 bytes"));
    F::new(
        Goldilocks::from_canonical_u64(word(0)).expect("canonical"),
        Goldilocks::from_canonical_u64(word(1)).expect("canonical"),
    )
}

/// TCE `Vec<Fp2>`: the `u64` length, then `c0 ‖ c1` of each element.
fn tce(values: &[F]) -> Vec<u8> {
    let mut out = (values.len() as u64).to_le_bytes().to_vec();
    for value in values {
        for c in value.coeffs() {
            out.extend_from_slice(&c.to_canonical_u64().to_le_bytes());
        }
    }
    out
}

// ----- The prover's side, as #113's tests run it ----------------------------------------------

/// A conformance model and its first sequence's tokens.
fn conformance(name: &str) -> (IntegerModel, Vec<u32>) {
    let dir = vectors().join("models");
    let index = json(&dir.join("models.json"));
    let entry = index["models"]
        .as_array()
        .expect("models")
        .iter()
        .find(|m| m["name"] == name)
        .expect("the model");
    let bytes = fs::read(dir.join(entry["file"].as_str().expect("a file"))).expect("the file");
    let model = envelope::open(&bytes, &schema::INTEGER_MODEL, DecodeLimits::INTEGER_MODEL)
        .expect("an IntegerModel");
    let tokens = entry["sequences"][0]["tokens"]
        .as_array()
        .expect("tokens")
        .iter()
        .map(|t| u32::try_from(t.as_u64().expect("a token")).expect("small"))
        .collect();
    (model, tokens)
}

const STATEMENT: [u8; 32] = [0x51; 32];
const SESSION: [u8; 16] = [0x14; 16];

fn transcript() -> Transcript {
    Transcript::new(&SessionContext {
        protocol: Ascii::new("tessera-protocol/1").expect("ASCII"),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("test-small/1").expect("ASCII")),
        statement_digest: STATEMENT,
        session_id: SESSION,
    })
}

/// Random leaf points, one per leaf claim, of its coordinates.
fn leaf_points(schedule: &StageSchedule, seed: u64) -> Vec<Vec<F>> {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Goldilocks::from_u64_reduced(z ^ (z >> 31))
    };
    schedule.stages[0]
        .claims
        .iter()
        .map(|claim| {
            (0..claim.point.len())
                .map(|_| {
                    let c0 = next();
                    F::new(c0, next())
                })
                .collect()
        })
        .collect()
}

/// A model proven: its lowering and witness, the leaf stage's points, claims and splits, each
/// reduction stage's messages, and the prover's pool.
struct Proven {
    lowering: Lowering,
    witness: Witness,
    points: Vec<Vec<F>>,
    leaf: Vec<Vec<F>>,
    leaf_splits: Vec<F>,
    stages: Vec<StageProof>,
    pool: Vec<Option<F>>,
}

fn prove(name: &str) -> Proven {
    let (model, tokens) = conformance(name);
    let n = u32::try_from(tokens.len()).expect("small");
    let lowering = lower(PublicModel::of(&model), n).expect("the model lowers");
    let schedule = derive_schedule(&lowering).expect("a schedule");
    let witness = Engine::new(&lowering)
        .expect("an engine")
        .run(&model, &Secret::new(tokens))
        .expect("a witness");
    let points = leaf_points(&schedule, u64::from(n) * 7919 + name.len() as u64);
    let mut t = transcript();
    let mut prover = StageProver::new(&lowering, &schedule, &witness);
    let leaf = prover.leaf(&points).expect("the leaf stage's claims");
    let flat: Vec<F> = leaf.iter().flatten().copied().collect();
    t.absorb_message("test/leaf", &flat);
    let leaf_splits = prover.leaf_splits(&mut t).expect("the leaf stage's splits");
    let stages = (1..schedule.stages.len())
        .map(|k| {
            let k = u16::try_from(k).expect("small");
            prover.prove_stage(&mut t, k).expect("a stage")
        })
        .collect();
    let pool = prover.pool().to_vec();
    Proven {
        lowering,
        witness,
        points,
        leaf,
        leaf_splits,
        stages,
        pool,
    }
}

// ----- The verifier's side ----------------------------------------------------------------------

/// The verifier's own lowering and schedule of a model's golden claim list, at `n` tokens, and
/// its RoPE constants, every one derived from the claim list's fields.
struct VerifierSide {
    graph: tv_core::lowering::Graph,
    schedule: Schedule,
    rope: tv_tables::rope::Rope,
}

fn verifier_side(name: &str, n: u32) -> VerifierSide {
    let lists = json(&vectors().join("claims/claim-lists.json"));
    let case = lists["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .find(|c| c["name"] == name)
        .expect("the golden claim list");
    let hex = case["hex"].as_str().expect("hex");
    let bytes: Vec<u8> = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex"))
        .collect();
    let claims = tv_core::claims::ClaimList::open(&bytes, &tv_codec::DecodeLimits::MESSAGE)
        .expect("the claim list opens");
    let public = tv_core::validation::check_public(&claims).expect("its public part");
    let graph = tv_core::lowering::lower(&public, n).expect("the verifier's lowering");
    let schedule = tv_core::schedule::derive(&graph).expect("the verifier's schedule");
    let kind = graph.tables()[0].kind.clone();
    let rope = tv_tables::Table::generate(tv_tables::TableId(0), kind)
        .expect("RoPE's table")
        .rope()
        .expect("a Rope table")
        .clone();
    VerifierSide {
        graph,
        schedule,
        rope,
    }
}

/// The prover's stage proof in the verifier's types.
fn tv_stage(proof: &StageProof) -> tv_core::stages::StageProof {
    tv_core::stages::StageProof {
        stage: proof.stage,
        sumcheck: tv_core::sumcheck::SumcheckProof {
            rounds: proof.sumcheck.rounds.iter().map(|r| tv_fp2s(r)).collect(),
            final_evals: tv_fp2s(&proof.sumcheck.final_evals),
        },
        splits: tv_fp2s(&proof.splits),
    }
}

/// What the verifier reads, in its types.
#[derive(Clone)]
struct Messages {
    leaf: Vec<LeafValues>,
    leaf_splits: Vec<TvFp2>,
    stages: Vec<tv_core::stages::StageProof>,
}

fn messages(proven: &Proven) -> Messages {
    Messages {
        leaf: proven
            .points
            .iter()
            .zip(&proven.leaf)
            .map(|(point, values)| LeafValues {
                point: tv_fp2s(point),
                values: tv_fp2s(values),
            })
            .collect(),
        leaf_splits: tv_fp2s(&proven.leaf_splits),
        stages: proven.stages.iter().map(tv_stage).collect(),
    }
}

/// Runs the verifier's stage checks from the transcript as the prover's stood before its leaf
/// stage's splits.
fn check(side: &VerifierSide, proven: &Proven, m: &Messages) -> Result<Verified, StageError> {
    let mut t = tv_hash::transcript::Transcript::new(&tv_hash::transcript::SessionContext {
        protocol: tv_hash::transcript::Ascii::new("tessera-protocol/1").expect("ASCII"),
        mode: tv_hash::transcript::Mode::NonInteractive,
        params: tv_hash::transcript::ParamSetId(
            tv_hash::transcript::Ascii::new("test-small/1").expect("ASCII"),
        ),
        statement_digest: STATEMENT,
        session_id: SESSION,
    });
    let flat: Vec<F> = proven.leaf.iter().flatten().copied().collect();
    t.absorb(
        tv_hash::transcript::Ascii::new("test/leaf").expect("ASCII"),
        &tce(&flat),
    );
    verify(
        &mut t,
        &side.schedule,
        &side.rope,
        &m.leaf,
        &m.leaf_splits,
        &m.stages,
    )
}

// ----- The pool against the witness -------------------------------------------------------------

/// A tensor's per-dimension padded table of the witness's true integers.
fn padded(lowering: &Lowering, witness: &Witness, tensor: TensorId) -> Vec<Goldilocks> {
    let shape = &lowering.graph.tensors[tensor.0 as usize].shape;
    let vars: Vec<u32> = shape
        .iter()
        .map(|&d| u32::BITS - (d - 1).leading_zeros())
        .collect();
    let mut table = vec![Goldilocks::ZERO; 1 << vars.iter().sum::<u32>()];
    for (flat, &v) in witness.tensor(tensor).expect("a tensor").iter().enumerate() {
        let (mut rest, mut index, mut shift) = (flat, 0usize, 0);
        for (axis, &len) in shape.iter().enumerate().rev() {
            index |= (rest % len as usize) << shift;
            rest /= len as usize;
            shift += vars[axis];
        }
        table[index] = Goldilocks::from_i64(v);
    }
    table
}

/// Layer `layer`'s tensor of the stacked kind layer 0's `rep` names, from the verifier's graph,
/// whose ids are the compiler's (RFC-0044).
fn layer_tensor(side: &VerifierSide, rep: tv_core::lowering::TensorId, layer: u32) -> TensorId {
    let graph = &side.graph;
    if let Some(producer) = graph.tensor(rep).expect("a tensor").producer {
        let producers: Vec<u32> = graph
            .checkpoints()
            .iter()
            .map(|&c| {
                graph
                    .tensor(c)
                    .expect("a checkpoint")
                    .producer
                    .expect("a node")
                    .0
            })
            .collect();
        let width = producers[1] - producers[0];
        let node = graph.node(producer).expect("a node");
        let output = node
            .outputs()
            .iter()
            .position(|&t| t == rep)
            .expect("an output");
        let at = tv_core::lowering::NodeId(producer.0 + layer * width);
        return TensorId(graph.node(at).expect("a node").outputs()[output].0);
    }
    let role = graph
        .inputs()
        .iter()
        .find(|i| i.tensor == rep)
        .expect("a graph input")
        .role;
    let same = |r: Role| match (role, r) {
        (Role::Weight(a), Role::Weight(b))
        | (Role::Classes(a), Role::Classes(b))
        | (Role::Multipliers(a), Role::Multipliers(b)) => a.ty == b.ty && b.layer == layer,
        _ => false,
    };
    let input = graph
        .inputs()
        .iter()
        .find(|i| same(i.role))
        .expect("its layer's");
    TensorId(input.tensor.0)
}

/// A pool claim's combination `Σ_t Σ_ℓ w_t(ℓ)·P̃_{t,ℓ}(point)`, from the witness.
fn from_witness(side: &VerifierSide, proven: &Proven, claim: &PoolClaim) -> F {
    let point: Vec<F> = claim.point.iter().copied().map(from_tv).collect();
    let value = |t: TensorId| -> F {
        evaluate_base::<F>(&padded(&proven.lowering, &proven.witness, t), &point).expect("a point")
    };
    let mut sum = F::ZERO;
    for term in &claim.terms {
        match term.tensor {
            PoolTensor::Single(t) => sum += from_tv(term.weights[0]) * value(TensorId(t.0)),
            PoolTensor::Stacked(rep) => {
                for (layer, &w) in term.weights.iter().enumerate() {
                    let t = layer_tensor(side, rep, u32::try_from(layer).expect("small"));
                    sum += from_tv(w) * value(t);
                }
            }
        }
    }
    sum
}

// ----- The tests --------------------------------------------------------------------------------

/// Every conformance model's stages, proven by the prover, verify with the verifier's own
/// schedule; the verifier's pool is the prover's, and every one of its claims holds of the
/// witness.
#[test]
fn the_tiny_models_stage_proofs_verify() {
    for name in MODELS {
        let proven = prove(name);
        let n = proven.lowering.graph.tensors[proven.lowering.positions.0 as usize].shape[0];
        let side = verifier_side(name, n);
        assert_eq!(
            side.graph.tensors().len(),
            proven.lowering.graph.tensors.len(),
            "{name}: the two lowerings name the same tensors"
        );
        let verified = check(&side, &proven, &messages(&proven))
            .unwrap_or_else(|e| panic!("{name}: the verifier rejects the prover's stages: {e}"));
        assert_eq!(
            verified.pool.len(),
            proven.pool.len(),
            "{name}: the pool's claims"
        );
        for (i, (claim, prover)) in verified.pool.iter().zip(&proven.pool).enumerate() {
            let prover = prover.unwrap_or_else(|| panic!("{name}: pool claim {i} has a value"));
            assert_eq!(claim.value, tv_fp2(prover), "{name}: pool claim {i}");
            assert_eq!(
                from_witness(&side, &proven, claim),
                prover,
                "{name}: pool claim {i} holds of the witness"
            );
        }
    }
}

fn changed(x: TvFp2) -> TvFp2 {
    x + TvFp2::ONE
}

/// RFC-0010 'Sumcheck messages': a changed round value of stage `k`, in its first, middle or last
/// round, is rejected at the stage's final check, `Reject(Sumcheck { stage: k, round: n_k })`; a
/// round one value short is rejected at its round.
#[test]
fn a_tampered_round_message_is_rejected_with_its_stage_and_round() {
    for name in MODELS {
        let proven = prove(name);
        let n = proven.lowering.graph.tensors[proven.lowering.positions.0 as usize].shape[0];
        let side = verifier_side(name, n);
        let honest = messages(&proven);
        for (k, stage) in side.schedule.stages().iter().enumerate() {
            let final_round = u32::try_from(stage.rounds).expect("small");
            for round in [0, stage.rounds / 2, stage.rounds - 1] {
                let mut m = honest.clone();
                let message = &mut m.stages[k].sumcheck.rounds[round];
                message[0] = changed(message[0]);
                let error = check(&side, &proven, &m).expect_err("a changed round");
                assert_eq!(
                    error.rejection(),
                    Some(SumcheckRejection {
                        stage: stage.index,
                        round: final_round
                    }),
                    "{name}: stage {} round {round}: {error}",
                    stage.index
                );
            }
            let mut m = honest.clone();
            m.stages[k].sumcheck.rounds[2].pop();
            let error = check(&side, &proven, &m).expect_err("a short round");
            assert_eq!(
                error.rejection(),
                Some(SumcheckRejection {
                    stage: stage.index,
                    round: 2
                }),
                "{name}: stage {}: {error}",
                stage.index
            );
        }
    }
}

/// A changed final evaluation of stage `k` fails its final check:
/// `Reject(Sumcheck { stage: k, round: n_k })`.
#[test]
fn a_tampered_final_evaluation_is_rejected_with_its_stage_and_round() {
    for name in MODELS {
        let proven = prove(name);
        let n = proven.lowering.graph.tensors[proven.lowering.positions.0 as usize].shape[0];
        let side = verifier_side(name, n);
        let honest = messages(&proven);
        for (k, stage) in side.schedule.stages().iter().enumerate() {
            let count = honest.stages[k].sumcheck.final_evals.len();
            for e in [0, count / 2, count - 1] {
                let mut m = honest.clone();
                let evals = &mut m.stages[k].sumcheck.final_evals;
                evals[e] = changed(evals[e]);
                let error = check(&side, &proven, &m).expect_err("a changed evaluation");
                assert_eq!(
                    error.rejection(),
                    Some(SumcheckRejection {
                        stage: stage.index,
                        round: u32::try_from(stage.rounds).expect("small"),
                    }),
                    "{name}: stage {} evaluation {e}: {error}",
                    stage.index
                );
            }
        }
    }
}

/// RFC-0037 'Splits': each value of the leaf stage's split claims, the residual's fused key in
/// every layer of the models that take it, changed, is `Reject(Sumcheck { stage: 0, round: 0 })`.
#[test]
fn a_tampered_split_value_is_rejected_at_the_leaf_stage() {
    for name in ["tiny-2-affine", "tiny-4-servable"] {
        let proven = prove(name);
        let n = proven.lowering.graph.tensors[proven.lowering.positions.0 as usize].shape[0];
        let side = verifier_side(name, n);
        let honest = messages(&proven);
        assert!(
            !honest.leaf_splits.is_empty(),
            "{name} splits its fused key"
        );
        for i in 0..honest.leaf_splits.len() {
            let mut m = honest.clone();
            m.leaf_splits[i] = changed(m.leaf_splits[i]);
            let error = check(&side, &proven, &m).expect_err("a changed split value");
            assert_eq!(
                error,
                StageError::Split { stage: 0, round: 0 },
                "{name}: value {i}"
            );
            assert_eq!(
                error.rejection(),
                Some(SumcheckRejection { stage: 0, round: 0 })
            );
        }
    }
}

/// A false leaf value, which the trees' leaf checks will bind (#124), reaches stage 1's claim
/// and fails its final check.
#[test]
fn a_false_leaf_value_is_rejected_at_stage_1() {
    let proven = prove("tiny-2-servable");
    let n = proven.lowering.graph.tensors[proven.lowering.positions.0 as usize].shape[0];
    let side = verifier_side("tiny-2-servable", n);
    let mut m = messages(&proven);
    m.leaf[1].values[1] = changed(m.leaf[1].values[1]);
    let error = check(&side, &proven, &m).expect_err("a false leaf value");
    let stage1 = &side.schedule.stages()[0];
    assert_eq!(
        error.rejection(),
        Some(SumcheckRejection {
            stage: 1,
            round: u32::try_from(stage1.rounds).expect("small"),
        })
    );
}
