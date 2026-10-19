// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `spec/vectors/transcript-schedule/schedule.json`, written by an independent oracle
//! (`generate.py`) from RFC-0014 'Transcript schedule' and the documents that amend it:
//!
//! - every case's labels and operations are this crate's expansion of its sizes, at its parameter
//!   set and in its mode, the conformance models' in both modes;
//! - no label appears twice in any of them (INV-FS-4);
//! - a prover held to the schedule, absorbing the case's messages, draws every challenge, seed,
//!   column index and proof-of-work nonce the oracle drew, and ends in its state; a verifier
//!   accepts the oracle's nonces and draws the same.

mod common;

use std::collections::HashSet;
use std::fs;

use serde_json::Value;
use tessera_codec::{Ascii, Encode, Sink, encode};
use tessera_field::{GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_hash::{Mode, ParamSetId, SessionContext};
use tessera_protocol::{
    Label, LeafTree, MatrixCodeword, MatrixPoint, Op, ParamSet, ProofOfWork, Protocol,
    ScheduleSizes, ScheduledTranscript, StageSizes, TranscriptSchedule, WeightMatrix, coin_toss,
};

fn load() -> Value {
    let path = common::vectors().join("transcript-schedule/schedule.json");
    serde_json::from_str(&fs::read_to_string(path).expect("vector file")).expect("JSON")
}

fn text(value: &Value) -> &str {
    value.as_str().expect("a string")
}

fn number<T: TryFrom<u64>>(value: &Value) -> T {
    T::try_from(value.as_u64().expect("a number")).unwrap_or_else(|_| panic!("in range"))
}

fn decimal(value: &Value) -> u64 {
    text(value).parse().expect("a decimal")
}

fn list(value: &Value) -> &Vec<Value> {
    value.as_array().expect("a list")
}

fn matrix(name: &Value) -> WeightMatrix {
    WeightMatrix::ALL
        .into_iter()
        .find(|m| m.name() == text(name))
        .expect("a matrix")
}

/// The case's sizes. A conformance model's advice point has no derivation yet (`null`): its
/// labels do not depend on it, and the oracle stops the transcript before it.
fn sizes(value: &Value) -> ScheduleSizes {
    ScheduleSizes {
        trees: list(&value["trees"])
            .iter()
            .map(|t| match text(&t["kind"]) {
                "fraction" => LeafTree::Fraction {
                    height: number(&t["height"]),
                },
                "forest" => LeafTree::Forest {
                    trees_log: number(&t["trees_log"]),
                    height: number(&t["height"]),
                },
                other => panic!("a tree of kind {other}"),
            })
            .collect(),
        layer_vars: number(&value["layer_vars"]),
        stages: list(&value["stages"])
            .iter()
            .map(|s| StageSizes {
                rounds: number(&s["rounds"]),
                zero_vars: number(&s["zero_vars"]),
            })
            .collect(),
        advice_vars: value["advice_vars"]
            .as_u64()
            .map_or(0, |v| u32::try_from(v).expect("a count")),
        proximity: list(&value["proximity"])
            .iter()
            .map(|p| MatrixPoint {
                matrix: matrix(&p["matrix"]),
                vars: number(&p["vars"]),
            })
            .collect(),
        opened: list(&value["opened"])
            .iter()
            .map(|c| MatrixCodeword {
                matrix: matrix(&c["matrix"]).id(),
                len: number(&c["codeword"]),
            })
            .collect(),
    }
}

fn mode(value: &Value) -> Mode {
    match text(value) {
        "Live" => Mode::Live,
        "NonInteractive" => Mode::NonInteractive,
        other => panic!("a mode {other}"),
    }
}

/// The case's parameter set, with the proofs of work it states: the set's own, or a hypothetical
/// set's, as the soundness calculator takes sets built by hand.
fn params(case: &Value) -> ParamSet {
    let id = ParamSetId(Ascii::new(text(&case["params"])).expect("ASCII"));
    ParamSet {
        seed_pow: number(&case["seed_pow"]),
        query_pow: number(&case["query_pow"]),
        ..*ParamSet::defined(&id).expect("a set of RFC-0014")
    }
}

fn context(case: &Value) -> SessionContext {
    let c = &case["context"];
    let context = SessionContext {
        protocol: Ascii::new(text(&c["protocol"])).expect("ASCII"),
        mode: mode(&c["mode"]),
        params: ParamSetId(Ascii::new(text(&c["params"])).expect("ASCII")),
        statement_digest: common::bytes(text(&c["statement_digest"]))
            .try_into()
            .expect("32 bytes"),
        session_id: common::bytes(text(&c["session_id"]))
            .try_into()
            .expect("16 bytes"),
    };
    assert_eq!(common::hex(&encode(&context)), text(&case["context_tce"]));
    context
}

/// The operation a step of the file states; `None` for a point whose size the case does not
/// know.
fn op(step: &Value) -> Option<Op> {
    Some(match text(&step["op"]) {
        "absorb" => Op::Absorb,
        "digest" => Op::Digest,
        "fp2" => Op::Fp2 {
            count: u32::try_from(step["count"].as_u64()?).expect("a count"),
        },
        "indices" => Op::Indices {
            log_n: number(&step["log_n"]),
            count: number(&step["count"]),
        },
        "grind" => Op::Grind {
            phase: match text(&step["phase"]) {
                "seed" => ProofOfWork::Seed,
                "query" => ProofOfWork::Query,
                other => panic!("a phase {other}"),
            },
            bits: number(&step["bits"]),
        },
        other => panic!("an op {other}"),
    })
}

/// Bytes absorbed as they are: the case's messages are TCE already.
struct Raw(Vec<u8>);

impl Encode for Raw {
    fn encode_to<S: Sink + ?Sized>(&self, out: &mut S) {
        out.put(&self.0);
    }
}

fn fp2s(value: &Value) -> Vec<GoldilocksExt2> {
    list(value)
        .iter()
        .map(|pair| {
            let [c0, c1] = &list(pair)[..] else {
                panic!("a pair");
            };
            let fp = |v: &Value| {
                tessera_field::Goldilocks::from_canonical_u64(decimal(v)).expect("canonical")
            };
            GoldilocksExt2::new(fp(c0), fp(c1))
        })
        .collect()
}

/// Every case of the file, the conformance models' with the others.
fn cases(file: &Value) -> Vec<&Value> {
    list(&file["cases"])
        .iter()
        .chain(list(&file["models"]))
        .collect()
}

/// The case's schedule, checked against the file's labels and operations.
fn schedule(case: &Value) -> TranscriptSchedule {
    let name = text(&case["name"]);
    let schedule = TranscriptSchedule::expand(
        &sizes(&case["sizes"]),
        &params(case),
        mode(&case["mode"]),
        Protocol::One,
    )
    .unwrap_or_else(|e| panic!("{name}: {e}"));
    let steps = list(&case["steps"]);
    assert_eq!(schedule.steps().len(), steps.len(), "{name}");
    for (i, (ours, theirs)) in schedule.steps().iter().zip(steps).enumerate() {
        assert_eq!(
            ours.label.to_string(),
            text(&theirs["label"]),
            "{name}: step {i}"
        );
        match op(theirs) {
            Some(op) => assert_eq!(ours.op, op, "{name}: step {i}"),
            // The advice's point of no known size.
            None => assert_eq!(ours.label, Label::AdviceXi, "{name}: step {i}"),
        }
    }
    schedule
}

#[test]
fn every_schedule_is_the_oracles() {
    let file = load();
    assert_eq!(file["official_blake3_cases_checked"], 35);
    let all = cases(&file);
    assert!(all.len() >= 16, "the cases and both modes of four models");
    let models: HashSet<&str> = list(&file["models"])
        .iter()
        .map(|c| text(&c["name"]))
        .collect();
    assert_eq!(models.len(), 8, "four conformance models in two modes");
    for case in all {
        let schedule = schedule(case);
        // INV-FS-4.
        let labels: Vec<String> = schedule.labels().map(|l| l.to_string()).collect();
        let distinct: HashSet<&String> = labels.iter().collect();
        assert_eq!(distinct.len(), labels.len(), "{}", text(&case["name"]));
    }
}

/// Plays the first `replayed` steps of a case as a prover, or as a verifier given the file's
/// nonces, checking every output against the file's.
fn replay(case: &Value, schedule: &TranscriptSchedule, verify: bool) {
    let name = text(&case["name"]);
    let session = context(case);
    let mut t = ScheduledTranscript::new(&session, schedule).expect("a transcript");
    let replayed: usize = number(&case["replayed"]);
    let steps = list(&case["steps"]);
    for (i, (step, theirs)) in schedule
        .steps()
        .iter()
        .zip(steps)
        .take(replayed)
        .enumerate()
    {
        let at = format!("{name}: step {i}, {}", step.label);
        match step.op {
            Op::Absorb => {
                let bytes = common::bytes(text(&theirs["tce"]));
                if step.label == Label::Seed {
                    // A live session's seed is the coin toss of the case's shares.
                    let share = |key: &str| -> [u8; 32] {
                        common::bytes(text(&case["coin_toss"][key]))
                            .try_into()
                            .expect("32 bytes")
                    };
                    let seed = coin_toss(&share("verifier_share"), &share("prover_share"));
                    assert_eq!(bytes, seed, "{at}");
                    t.seed_live(&seed);
                } else {
                    t.absorb(step.label, &Raw(bytes));
                }
            }
            Op::Digest => {
                assert_eq!(
                    common::hex(&t.squeeze(step.label)),
                    text(&theirs["out"]),
                    "{at}"
                );
            }
            Op::Fp2 { .. } => assert_eq!(t.point(step.label), fp2s(&theirs["values"]), "{at}"),
            Op::Indices { .. } => {
                let theirs: Vec<u64> = list(&theirs["values"]).iter().map(decimal).collect();
                assert_eq!(t.challenge_indices(step.label), theirs, "{at}");
            }
            Op::Grind { .. } => {
                let nonce = decimal(&theirs["nonce"]);
                if verify {
                    assert_eq!(t.check_grind(step.label, nonce), Ok(()), "{at}");
                } else {
                    // The honest prover's nonce is the smallest that meets the target.
                    assert_eq!(t.grind(step.label), nonce, "{at}");
                }
            }
        }
    }
    assert_eq!(t.fault(), None, "{name}");
    assert_eq!(
        common::hex(&t.transcript().state()),
        text(&case["state"]),
        "{name}"
    );
    assert_eq!(
        t.transcript().counter(),
        number::<u64>(&case["counter"]),
        "{name}"
    );
    if replayed == steps.len() {
        t.finish().unwrap_or_else(|e| panic!("{name}: {e}"));
    } else {
        // A conformance model stops at the advice's point, whose size is not derived yet.
        assert_eq!(schedule.steps()[replayed].label, Label::AdviceXi, "{name}");
    }
}

#[test]
fn a_prover_held_to_the_schedule_draws_the_oracles_challenges() {
    let file = load();
    for case in cases(&file) {
        replay(case, &schedule(case), false);
    }
}

#[test]
fn a_verifier_accepts_the_oracles_nonces_and_draws_the_same() {
    let file = load();
    for case in cases(&file) {
        replay(case, &schedule(case), true);
    }
}

/// The cases cover what the issue asks: both modes, proofs of work of more than 0 bits, every
/// placement of RFC-0049's points, every conformance model.
#[test]
fn the_vectors_cover_both_modes_and_the_proofs_of_work() {
    let file = load();
    let all = cases(&file);
    let grinding = all.iter().any(|c| {
        mode(&c["mode"]) == Mode::NonInteractive
            && number::<u32>(&c["seed_pow"]) > 0
            && number::<u32>(&c["query_pow"]) > 0
    });
    assert!(grinding, "a case grinds for both phases");
    for name in [
        "tiny-2-servable",
        "tiny-2-minimal",
        "tiny-2-affine",
        "tiny-4-servable",
    ] {
        for m in ["Live", "NonInteractive"] {
            assert!(
                all.iter()
                    .any(|c| text(&c["name"]) == format!("{name}, {m}")),
                "{name} in {m}"
            );
        }
    }
}
