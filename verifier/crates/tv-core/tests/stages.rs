// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The stage checks (RFC-0010 'Sumcheck messages', 'Batching within a stage'; RFC-0037 'Splits',
//! 'The layer point', 'The claim pool') on honest proofs of every conformance model's schedule,
//! over synthetic witnesses, from the reference prover of `stage_support`: they verify, every
//! pool claim equals its tensors' combination evaluated from the witness, and a proof changed in
//! one value is rejected with the exact payload of `Reject(Sumcheck { stage, round })`.

mod claim_support;
mod stage_support;

use claim_support::{goldens, with_layers};
use stage_support::{Prover, Sources, Witness, to_fp2};
use tv_codec::DecodeLimits;
use tv_core::claims::{ClaimList, WeightType};
use tv_core::lowering::{Graph, lower};
use tv_core::schedule::{Commitment, Schedule, derive};
use tv_core::stages::{LeafValues, StageError, StageProof, SumcheckRejection, Verified, verify};
use tv_core::sumcheck::SumcheckError;
use tv_core::validation::check_public;
use tv_field::Fp2;
use tv_hash::transcript::{Ascii, Mode, ParamSetId, SessionContext, Transcript};
use tv_tables::rope::Rope;
use tv_tables::{Table, TableId};
use tv_testkit::SplitMix64;
use tv_testkit::sumcheck::{self as reference, Fe, Lie};

fn context() -> SessionContext<'static> {
    SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("test-small/1").unwrap()),
        statement_digest: [7; 32],
        session_id: [9; 16],
    }
}

/// A model's graph at `n` tokens, its schedule and its RoPE constants; `layers` repeats its
/// layers' rescaling over another number of layers.
fn model(name: &str, n: u32, layers: Option<u32>) -> (Graph, Schedule, Rope) {
    model_vocabulary(name, n, layers, None)
}

fn model_vocabulary(
    name: &str,
    n: u32,
    layers: Option<u32>,
    vocab: Option<u32>,
) -> (Graph, Schedule, Rope) {
    let golden = goldens().into_iter().find(|g| g.name == name).unwrap();
    let mut claims = ClaimList::open(&golden.envelope, &DecodeLimits::MESSAGE).unwrap();
    if let Some(layers) = layers {
        claims = with_layers(&claims, layers);
    }
    if let Some(v) = vocab {
        claims.arch.vocab = v;
    }
    let public = check_public(&claims).unwrap();
    let graph = lower(&public, n).unwrap();
    let schedule = derive(&graph).unwrap();
    let kind = graph.tables()[0].kind.clone();
    let rope = Table::generate(TableId(0), kind)
        .unwrap()
        .rope()
        .unwrap()
        .clone();
    (graph, schedule, rope)
}

/// An honest proof of a model's stages, or one whose prover lies in one round of stage `k`.
struct Run {
    graph: Graph,
    schedule: Schedule,
    rope: Rope,
    witness: Witness,
    leaf: Vec<LeafValues>,
    leaf_splits: Vec<Fp2>,
    stages: Vec<StageProof>,
}

impl Run {
    fn new(name: &str, n: u32, seed: u64, lie: Option<(usize, Lie)>) -> Self {
        Self::with(name, n, None, seed, lie)
    }

    fn with(name: &str, n: u32, layers: Option<u32>, seed: u64, lie: Option<(usize, Lie)>) -> Self {
        Self::with_indicator(name, n, layers, seed, lie, None)
    }

    fn with_indicator(
        name: &str,
        n: u32,
        layers: Option<u32>,
        seed: u64,
        lie: Option<(usize, Lie)>,
        entry: Option<Fe>,
    ) -> Self {
        let (graph, schedule, rope) = model(name, n, layers);
        let witness = Witness::with_indicator_entry(&graph, &rope, seed, entry);
        Self::from_witness(graph, schedule, rope, witness, seed, lie)
    }

    fn from_witness(
        graph: Graph,
        schedule: Schedule,
        rope: Rope,
        witness: Witness,
        seed: u64,
        lie: Option<(usize, Lie)>,
    ) -> Self {
        Self::from_witness_omitting(graph, schedule, rope, witness, seed, lie, None)
    }

    fn from_witness_omitting(
        graph: Graph,
        schedule: Schedule,
        rope: Rope,
        witness: Witness,
        seed: u64,
        lie: Option<(usize, Lie)>,
        omit: Option<usize>,
    ) -> Self {
        let mut transcript = reference::Transcript::new(&context().to_tce());
        let (leaf, leaf_splits, stages) = {
            let mut prover = prover(&graph, &witness, &schedule, &rope);
            prover.omit_onehot = omit;
            let mut rng = SplitMix64::new(seed ^ 0x1eaf);
            let (leaf, splits) = prover.leaf(&mut rng);
            transcript.absorb("stage/0/split", &reference::tce(&splits));
            let stages = prover.stages(&mut transcript, lie);
            (leaf, splits.iter().map(|&v| to_fp2(v)).collect(), stages)
        };
        Self {
            graph,
            schedule,
            rope,
            witness,
            leaf,
            leaf_splits,
            stages,
        }
    }

    fn verify(&self) -> Result<Verified, StageError> {
        self.verify_with(&self.leaf, &self.leaf_splits, &self.stages)
    }

    fn verify_with(
        &self,
        leaf: &[LeafValues],
        leaf_splits: &[Fp2],
        stages: &[StageProof],
    ) -> Result<Verified, StageError> {
        let mut transcript = Transcript::new(&context());
        verify(
            &mut transcript,
            &self.schedule,
            &self.rope,
            leaf,
            leaf_splits,
            stages,
        )
    }

    /// The reference prover's view, to evaluate pool claims from the witness.
    fn prover(&self) -> Prover<'_> {
        let mut prover = prover(&self.graph, &self.witness, &self.schedule, &self.rope);
        prover.sources = Sources::default();
        prover
    }
}

fn prover<'a>(
    graph: &'a Graph,
    witness: &'a Witness,
    schedule: &'a Schedule,
    rope: &'a Rope,
) -> Prover<'a> {
    Prover {
        layout: stage_support::Layout::new(graph),
        witness,
        schedule,
        rope,
        sources: Sources::default(),
        omit_onehot: None,
    }
}

const MODELS: [&str; 4] = [
    "tiny-2-servable",
    "tiny-2-minimal",
    "tiny-2-affine",
    "tiny-4-servable",
];

/// Every conformance model's stages verify on honest proofs at lengths 1, 4 and 5, and every pool
/// claim equals the witness's combination of its tensors; each weight type has one pool claim
/// (INV-SCH-5), and the leaf stage's pool claims come first.
#[test]
fn honest_stages_verify_and_leave_true_pool_claims() {
    for name in MODELS {
        for n in [1, 4, 5] {
            let run = Run::new(name, n, u64::from(n) * 31 + name.len() as u64, None);
            let verified = run
                .verify()
                .unwrap_or_else(|e| panic!("{name} at n = {n}: {e}"));
            let prover = run.prover();
            for (i, claim) in verified.pool.iter().enumerate() {
                assert_eq!(
                    to_fp2(prover.pool_value(claim)),
                    claim.value,
                    "{name} at n = {n}: pool claim {i}, stage {}, {:?}",
                    claim.stage,
                    claim.commitment
                );
            }
            for ty in [
                WeightType::Embed,
                WeightType::Q,
                WeightType::K,
                WeightType::V,
                WeightType::O,
                WeightType::Gate,
                WeightType::Up,
                WeightType::Down,
                WeightType::LmHead,
            ] {
                let claims = verified
                    .pool
                    .iter()
                    .filter(|c| c.commitment == Commitment::Weight(ty))
                    .count();
                assert_eq!(claims, 1, "{name} at n = {n}: {ty:?}");
            }
            let stages: Vec<u16> = verified.pool.iter().map(|c| c.stage).collect();
            assert!(stages.is_sorted(), "{name}: {stages:?}");
        }
    }
}

/// One layer, three (the stack padded with a layer of zeros, whose weight `eq(ρ, 3)` is not 0)
/// and eight: honest stages verify and every pool claim holds (RFC-0037 'Layer variables',
/// 'Stages and their instances', as amended).
#[test]
fn stacks_of_one_three_and_eight_layers_verify() {
    for name in ["tiny-2-servable", "tiny-2-affine", "tiny-4-servable"] {
        for layers in [1, 3, 8] {
            let run = Run::with(name, 5, Some(layers), 41 + u64::from(layers), None);
            let verified = run
                .verify()
                .unwrap_or_else(|e| panic!("{name} at {layers} layers: {e}"));
            let prover = run.prover();
            for (i, claim) in verified.pool.iter().enumerate() {
                assert_eq!(
                    to_fp2(prover.pool_value(claim)),
                    claim.value,
                    "{name} at {layers} layers: pool claim {i}"
                );
            }
        }
    }
}

/// The longest length, `max_context`, verifies too.
#[test]
fn the_longest_sequence_verifies() {
    let run = Run::new("tiny-2-servable", 64, 5, None);
    let verified = run.verify().unwrap();
    let prover = run.prover();
    for claim in &verified.pool {
        assert_eq!(to_fp2(prover.pool_value(claim)), claim.value);
    }
}

fn offset(value: Fp2) -> Fp2 {
    value + Fp2::ONE
}

/// RFC-0010 'Sumcheck messages': a changed round value of stage `k` changes the claims that
/// follow, and the stage's final check fails: `Reject(Sumcheck { stage: k, round: n_k })`. A
/// round of another length is rejected at its round.
#[test]
fn a_tampered_round_message_is_rejected_at_its_stage() {
    for name in MODELS {
        let run = Run::new(name, 5, 11, None);
        for (k, stage) in run.schedule.stages().iter().enumerate() {
            for round in [0, stage.rounds / 2, stage.rounds - 1] {
                for value in 0..stage.degree {
                    let mut stages = run.stages.clone();
                    let message = &mut stages[k].sumcheck.rounds[round];
                    message[value] = offset(message[value]);
                    let error = run
                        .verify_with(&run.leaf, &run.leaf_splits, &stages)
                        .unwrap_err();
                    assert_eq!(
                        error.rejection(),
                        Some(SumcheckRejection {
                            stage: stage.index,
                            round: stage.rounds as u32,
                        }),
                        "{name}: stage {} round {round} value {value}: {error}",
                        stage.index
                    );
                }
            }
            // A round one value short.
            let mut stages = run.stages.clone();
            stages[k].sumcheck.rounds[1].pop();
            let error = run
                .verify_with(&run.leaf, &run.leaf_splits, &stages)
                .unwrap_err();
            assert_eq!(
                error,
                StageError::Sumcheck {
                    stage: stage.index,
                    error: SumcheckError::RoundLength {
                        round: 1,
                        expected: stage.degree,
                        found: stage.degree - 1,
                    },
                }
            );
            assert_eq!(
                error.rejection(),
                Some(SumcheckRejection {
                    stage: stage.index,
                    round: 1
                })
            );
        }
    }
}

/// A prover that lies in one round and forges the final evaluations to pass the final check of
/// its stage leaves false claims, which the next stage or the pool catches: the reference
/// prover's forged first evaluation is a claim on its first instance's first input.
#[test]
fn a_forged_round_passes_its_stage_and_leaves_a_false_claim() {
    let lie = Lie {
        round: 2,
        offset: Fe::base(5),
        value: Some(0),
        forge: true,
    };
    let run = Run::new("tiny-2-minimal", 5, 23, Some((1, lie)));
    // Stage 2's first instance is a stacked MatMulWeight whose first input is its committed
    // weight: the forged value reaches the pool, false.
    let verified = run.verify().unwrap();
    let prover = run.prover();
    let false_claims = verified
        .pool
        .iter()
        .filter(|c| to_fp2(prover.pool_value(c)) != c.value)
        .count();
    assert_eq!(false_claims, 1);
}

/// A changed final evaluation of stage `k` fails its final check:
/// `Reject(Sumcheck { stage: k, round: n_k })`; one too many or too few is rejected there too.
#[test]
fn a_tampered_final_evaluation_is_rejected_at_its_stage() {
    for name in MODELS {
        let run = Run::new(name, 4, 13, None);
        for (k, stage) in run.schedule.stages().iter().enumerate() {
            let expected = Some(SumcheckRejection {
                stage: stage.index,
                round: stage.rounds as u32,
            });
            let count = run.stages[k].sumcheck.final_evals.len();
            for e in [0, count / 2, count - 1] {
                let mut stages = run.stages.clone();
                let evals = &mut stages[k].sumcheck.final_evals;
                evals[e] = offset(evals[e]);
                let error = run
                    .verify_with(&run.leaf, &run.leaf_splits, &stages)
                    .unwrap_err();
                assert_eq!(error.rejection(), expected, "{name}: stage {k} eval {e}");
            }
            let mut stages = run.stages.clone();
            stages[k].sumcheck.final_evals.push(Fp2::ONE);
            let error = run
                .verify_with(&run.leaf, &run.leaf_splits, &stages)
                .unwrap_err();
            assert!(matches!(
                error,
                StageError::Sumcheck {
                    error: SumcheckError::EvaluationCount { .. },
                    ..
                }
            ));
            assert_eq!(error.rejection(), expected);
        }
    }
}

/// RFC-0037 'Splits': a changed split value of the leaf stage is
/// `Reject(Sumcheck { stage: 0, round: 0 })`, each value of the fused key's split in every layer:
/// its virtual atom and its committed group.
#[test]
fn a_tampered_split_value_is_rejected_at_the_leaf_stage() {
    for name in ["tiny-2-affine", "tiny-4-servable"] {
        let run = Run::new(name, 5, 17, None);
        assert!(!run.leaf_splits.is_empty(), "{name} splits its fused key");
        for i in 0..run.leaf_splits.len() {
            let mut splits = run.leaf_splits.clone();
            splits[i] = offset(splits[i]);
            let error = run
                .verify_with(&run.leaf, &splits, &run.stages)
                .unwrap_err();
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
        // Two values moved by opposite amounts, the virtual atom's coefficient being 1, pass the
        // split and leave false claims, which stage 1 rejects.
        let mut splits = run.leaf_splits.clone();
        splits[0] = offset(splits[0]);
        splits[1] -= Fp2::ONE;
        let error = run
            .verify_with(&run.leaf, &splits, &run.stages)
            .unwrap_err();
        let stage1 = &run.schedule.stages()[0];
        assert_eq!(
            error.rejection(),
            Some(SumcheckRejection {
                stage: 1,
                round: stage1.rounds as u32
            }),
            "{name}"
        );
    }
}

/// A false leaf claim value reaches stage 1's claim, whose sumcheck rejects it at its final
/// check.
#[test]
fn a_false_leaf_value_is_rejected_at_stage_1() {
    for name in MODELS {
        let run = Run::new(name, 5, 19, None);
        let stage1 = &run.schedule.stages()[0];
        for c in [1, run.leaf.len() - 1] {
            let mut leaf = run.leaf.clone();
            leaf[c].values[0] = offset(leaf[c].values[0]);
            let error = run
                .verify_with(&leaf, &run.leaf_splits, &run.stages)
                .unwrap_err();
            assert_eq!(
                error.rejection(),
                Some(SumcheckRejection {
                    stage: 1,
                    round: stage1.rounds as u32
                }),
                "{name}: leaf claim {c}"
            );
        }
    }
}

/// RFC-0036 'Shapes': the stage proofs, their stage ids and every stage's split values are
/// counted against the schedule before any check, `Reject(Malformed)`; leaf claims that do not
/// fit the schedule are the verifier's own fault.
#[test]
fn shapes_are_checked_before_any_stage() {
    let run = Run::new("tiny-4-servable", 4, 29, None);
    let count = run.stages.len();
    let error = run
        .verify_with(&run.leaf, &run.leaf_splits, &run.stages[..count - 1])
        .unwrap_err();
    assert_eq!(
        error,
        StageError::StageCount {
            expected: count,
            found: count - 1
        }
    );
    assert!(error.is_malformed() && error.rejection().is_none());

    let mut stages = run.stages.clone();
    stages[1].stage = 7;
    let error = run
        .verify_with(&run.leaf, &run.leaf_splits, &stages)
        .unwrap_err();
    assert_eq!(error, StageError::StageId { index: 1, found: 7 });
    assert!(error.is_malformed());

    let splits = &run.leaf_splits[..run.leaf_splits.len() - 1];
    let error = run.verify_with(&run.leaf, splits, &run.stages).unwrap_err();
    assert_eq!(
        error,
        StageError::SplitCount {
            stage: 0,
            expected: run.leaf_splits.len(),
            found: run.leaf_splits.len() - 1
        }
    );
    assert!(error.is_malformed());

    // A stage's extra split value is malformed even though an earlier stage is also wrong: the
    // shapes come first.
    let mut stages = run.stages.clone();
    stages[0].sumcheck.final_evals[0] = offset(stages[0].sumcheck.final_evals[0]);
    stages[1].splits.push(Fp2::ONE);
    let error = run
        .verify_with(&run.leaf, &run.leaf_splits, &stages)
        .unwrap_err();
    assert_eq!(
        error,
        StageError::SplitCount {
            stage: 2,
            expected: 0,
            found: 1
        }
    );

    let mut leaf = run.leaf.clone();
    leaf[3].values.push(Fp2::ONE);
    assert_eq!(
        run.verify_with(&leaf, &run.leaf_splits, &run.stages)
            .unwrap_err(),
        StageError::Leaf { claim: 3 }
    );
    let mut leaf = run.leaf.clone();
    leaf[0].point.pop();
    let error = run
        .verify_with(&leaf, &run.leaf_splits, &run.stages)
        .unwrap_err();
    assert_eq!(error, StageError::Leaf { claim: 0 });
    assert!(error.rejection().is_none() && !error.is_malformed());
    assert_eq!(
        run.verify_with(&run.leaf[1..], &run.leaf_splits, &run.stages)
            .unwrap_err(),
        StageError::Leaf {
            claim: run.leaf.len() - 1
        }
    );
}

/// RFC-0019 'Robustness': 600 seeded changes of an honest proof, of a value, a length or a stage
/// id anywhere in it, each yield a typed result and never panic; every change of a value or a
/// length the proof sends is rejected.
#[test]
fn seeded_changes_never_panic_and_are_rejected() {
    let run = Run::new("tiny-2-affine", 5, 37, None);
    let mut rng = SplitMix64::new(0x5eed);
    let pick = |rng: &mut SplitMix64, n: usize| (rng.next_u64() % n as u64) as usize;
    let random = |rng: &mut SplitMix64| Fp2::from(tv_field::Fp::reduce(rng.next_u64())) + Fp2::ONE;
    for trial in 0..600 {
        let mut stages = run.stages.clone();
        let mut splits = run.leaf_splits.clone();
        let k = pick(&mut rng, stages.len());
        let proof = &mut stages[k];
        match pick(&mut rng, 8) {
            0 => {
                let r = pick(&mut rng, proof.sumcheck.rounds.len());
                let v = pick(&mut rng, proof.sumcheck.rounds[r].len());
                proof.sumcheck.rounds[r][v] += random(&mut rng);
            }
            1 => {
                let r = pick(&mut rng, proof.sumcheck.rounds.len());
                let extra = random(&mut rng);
                proof.sumcheck.rounds[r].push(extra);
            }
            2 => {
                proof.sumcheck.rounds.pop();
            }
            3 => {
                let extra = proof.sumcheck.rounds[0].clone();
                proof.sumcheck.rounds.push(extra);
            }
            4 => {
                let e = pick(&mut rng, proof.sumcheck.final_evals.len());
                proof.sumcheck.final_evals[e] += random(&mut rng);
            }
            5 => {
                let len = proof.sumcheck.final_evals.len();
                proof.sumcheck.final_evals.truncate(pick(&mut rng, len));
            }
            6 => {
                let s = pick(&mut rng, splits.len());
                splits[s] += random(&mut rng);
            }
            _ => {
                proof.stage = (rng.next_u64() % 5) as u16;
                if usize::from(proof.stage) == k + 1 {
                    proof.splits.push(Fp2::ONE);
                }
            }
        }
        let outcome = run.verify_with(&run.leaf, &splits, &stages);
        assert!(
            outcome.is_err(),
            "trial {trial}: a changed proof was accepted"
        );
    }
}

#[test]
fn native_booleanity_rejects_a_non_boolean_indicator_with_consistent_downstream_values() {
    for name in MODELS {
        for n in [1, 3, 5] {
            let honest = Run::with_indicator(name, n, Some(3), 71, None, None);
            let accepted = honest.verify().unwrap();
            let h = honest
                .graph
                .nodes()
                .iter()
                .find(|n| matches!(n.op(), tv_core::lowering::Op::OneHot { .. }))
                .unwrap()
                .outputs()[0];
            assert!(accepted.pool.iter().any(|c| {
                c.stage == 1
                    && c.terms
                        .iter()
                        .any(|t| t.tensor == tv_core::stages::PoolTensor::Single(h))
            }));
            let changed = Run::with_indicator(name, n, Some(3), 71, None, Some(Fe::base(2)));
            assert_eq!(
                changed.verify().unwrap_err(),
                StageError::Sumcheck {
                    stage: 1,
                    error: SumcheckError::FinalMismatch {
                        round: changed.schedule.stages()[0].rounds as u32
                    },
                }
            );
        }
    }
}

#[test]
fn native_onehot_rejects_every_isolated_premise_and_retains_original_sources() {
    use stage_support::IndicatorFault;
    for n in [1, 3, 5] {
        for fault in [
            None,
            Some(IndicatorFault::NonBoolean),
            Some(IndicatorFault::Missing),
            Some(IndicatorFault::Extra),
            Some(IndicatorFault::Index),
            Some(IndicatorFault::Padding),
        ] {
            let (graph, schedule, rope) = model_vocabulary("tiny-2-servable", n, Some(3), Some(5));
            let witness = Witness::with_onehot_fault(&graph, &rope, 91, fault);
            let one = graph
                .nodes()
                .iter()
                .find(|node| matches!(node.op(), tv_core::lowering::Op::OneHot { .. }))
                .unwrap();
            let h_id = one.outputs()[0];
            let t_id = one.inputs()[0];
            let h = witness.table(h_id, graph.tensor(h_id).unwrap().shape.dims());
            let token = &witness.values[t_id.0 as usize];
            let mut holds = [true; 4];
            for (t, row) in h.chunks(8).enumerate() {
                holds[0] &= row.iter().all(|&v| v == Fe::ZERO || v == Fe::ONE);
                holds[1] &= row[5..].iter().all(|&v| v == Fe::ZERO);
                holds[2] &= row.iter().copied().fold(Fe::ZERO, |s, x| s + x)
                    == if t < n as usize { Fe::ONE } else { Fe::ZERO };
                holds[3] &= row
                    .iter()
                    .enumerate()
                    .fold(Fe::ZERO, |s, (i, &x)| s + Fe::base(i as u64) * x)
                    == token.get(t).copied().unwrap_or(Fe::ZERO);
            }
            let failed = match fault {
                None => None,
                Some(IndicatorFault::NonBoolean) => Some(0),
                Some(IndicatorFault::Padding) => Some(1),
                Some(IndicatorFault::Missing | IndicatorFault::Extra) => Some(2),
                Some(IndicatorFault::Index) => Some(3),
                _ => unreachable!(),
            };
            for (i, &valid) in holds.iter().enumerate() {
                assert_eq!(valid, Some(i) != failed, "{fault:?}, premise{i}");
            }
            let run = Run::from_witness(graph, schedule, rope, witness, 91, None);
            let verdict = run.verify();
            if let Ok(checked) = &verdict {
                let p = run.prover();
                for claim in &checked.pool {
                    assert_eq!(to_fp2(p.pool_value(claim)), claim.value);
                }
            }
            if failed.is_some() {
                assert_eq!(
                    verdict.unwrap_err(),
                    StageError::Sumcheck {
                        stage: 1,
                        error: SumcheckError::FinalMismatch {
                            round: run.schedule.stages()[0].rounds as u32
                        }
                    }
                );
            } else {
                assert!(verdict.is_ok());
            }
        }
    }
}

#[test]
fn native_onehot_scalar_and_ragged_domains_leave_true_source_claims() {
    for vocab in [1, 5, 512] {
        for n in [1, 3, 5] {
            let (graph, schedule, rope) =
                model_vocabulary("tiny-2-servable", n, Some(3), Some(vocab));
            let witness = Witness::new(&graph, &rope, 137);
            let run = Run::from_witness(graph, schedule, rope, witness, 137, None);
            let checked = run.verify().unwrap();
            let p = run.prover();
            for claim in checked.pool {
                assert_eq!(to_fp2(p.pool_value(&claim)), claim.value);
            }
        }
    }
}

#[test]
fn native_onehot_scalar_token_is_checked() {
    for n in [1, 3] {
        let (graph, schedule, rope) = model_vocabulary("tiny-2-servable", n, Some(3), Some(1));
        let witness = Witness::with_onehot_fault(
            &graph,
            &rope,
            157,
            Some(stage_support::IndicatorFault::ScalarToken),
        );
        let one = graph
            .nodes()
            .iter()
            .find(|node| matches!(node.op(), tv_core::lowering::Op::OneHot { .. }))
            .unwrap();
        assert_eq!(
            witness.values[one.outputs()[0].0 as usize],
            vec![Fe::ONE; n as usize]
        );
        assert_eq!(witness.values[one.inputs()[0].0 as usize][0], Fe::ONE);
        let run = Run::from_witness_omitting(graph, schedule, rope, witness, 157, None, Some(3));
        let result = run.verify();
        if let Ok(checked) = &result {
            let p = run.prover();
            for claim in &checked.pool {
                assert_eq!(to_fp2(p.pool_value(claim)), claim.value);
            }
        }
        assert_eq!(
            result.unwrap_err(),
            StageError::Sumcheck {
                stage: 1,
                error: SumcheckError::FinalMismatch {
                    round: run.schedule.stages()[0].rounds as u32
                }
            }
        );
    }
}

fn assert_onehot_omission(fault: stage_support::IndicatorFault, obligation: usize) {
    for n in [1, 3, 5] {
        let (graph, schedule, rope) = model_vocabulary("tiny-2-servable", n, Some(3), Some(5));
        let witness = Witness::with_onehot_fault(&graph, &rope, 91, Some(fault));
        let run =
            Run::from_witness_omitting(graph, schedule, rope, witness, 91, None, Some(obligation));
        let result = run.verify();
        if let Ok(checked) = &result {
            let p = run.prover();
            for claim in &checked.pool {
                assert_eq!(to_fp2(p.pool_value(claim)), claim.value);
            }
        }
        assert_eq!(
            result.unwrap_err(),
            StageError::Sumcheck {
                stage: 1,
                error: SumcheckError::FinalMismatch {
                    round: run.schedule.stages()[0].rounds as u32
                }
            }
        );
    }
}

#[test]
fn native_onehot_omission_booleanity() {
    assert_onehot_omission(stage_support::IndicatorFault::NonBoolean, 0);
}
#[test]
fn native_onehot_omission_vocabulary() {
    assert_onehot_omission(stage_support::IndicatorFault::Padding, 1);
}
#[test]
fn native_onehot_omission_count() {
    assert_onehot_omission(stage_support::IndicatorFault::Missing, 2);
    assert_onehot_omission(stage_support::IndicatorFault::Extra, 2);
}
#[test]
fn native_onehot_omission_index() {
    assert_onehot_omission(stage_support::IndicatorFault::Index, 3);
}
