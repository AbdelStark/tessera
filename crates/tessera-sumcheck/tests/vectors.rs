// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Conformance with the independent Python oracle, `spec/vectors/sumcheck/generate.py`, which
//! proves every sumcheck by padding tables literally and computes every value from its
//! definition (RFC-0010 'Batching within a stage', 'Sumcheck messages', 'Tree levels').
//!
//! - Every stage that records its tables is proven again from them, over `Fp2` as the oracle's
//!   tables are: `γ`, every round message, every challenge, the final evaluations, each
//!   instance's point and the transcript's state and counter must be the oracle's. The stages
//!   that do not record their tables are replayed through the transcript from their recorded
//!   messages, so that the next stage starts where the oracle's did.
//! - Every product-tree level is proven again from the trees' leaves, as the leaf stage will:
//!   instance `i` is `eq(r_i, ·)`, a public factor, times the two halves `V_{ℓ+1}(0 ‖ x)` and
//!   `V_{ℓ+1}(1 ‖ x)`, over the prime field, so these check the instances that mix the fields.
//!
//! The independent verifier checks every proof on its own transcript of the same session,
//! `tv_core::sumcheck::verify_batch` for the stages and `tv_core::tree::verify_product_level` for
//! the levels, from the vectors' claims: it accepts, draws the same challenges and ends each step
//! in the prover's transcript state.

mod common;

use std::fs;
use std::path::PathBuf;

use serde_json::Value;
use tessera_codec::Ascii;
use tessera_field::{Field, PrimeField, QuadraticExtension};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_poly::eq_table;
use tessera_sumcheck::{Instance, Prefix, Table, prove_batch};

use common::{Fp, Fp2, TvFp2, tv_fp2, tv_fp2s, tv_proof};
use tv_core::sumcheck as tv;
use tv_core::tree::{TreeClaim, verify_product_level};

fn vectors() -> Value {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/sumcheck/sumcheck.json");
    serde_json::from_str(&fs::read_to_string(path).expect("the vectors")).expect("JSON")
}

fn text(value: &Value) -> &str {
    value.as_str().expect("a string")
}

fn bytes(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex"))
        .collect()
}

fn fp(value: &Value) -> Fp {
    let hex = text(value).strip_prefix("0x").expect("0x");
    Fp::from_canonical_u64(u64::from_str_radix(hex, 16).expect("hex")).expect("canonical")
}

fn fp2(value: &Value) -> Fp2 {
    let pair = value.as_array().expect("a pair");
    Fp2::new(fp(&pair[0]), fp(&pair[1]))
}

fn list(value: &Value) -> &Vec<Value> {
    value.as_array().expect("a list")
}

fn fp2s(value: &Value) -> Vec<Fp2> {
    list(value).iter().map(fp2).collect()
}

fn number(value: &Value) -> usize {
    value.as_u64().expect("a number") as usize
}

fn prefix(value: &Value) -> Prefix {
    let label = text(value);
    let (kind, index) = label.split_once('/').expect("kind/index");
    let index = index.parse().expect("an index");
    match kind {
        "stage" => Prefix::Stage(index),
        "tree" => Prefix::Tree(index),
        other => panic!("prefix {other}"),
    }
}

fn transcript(context: &Value) -> Transcript {
    Transcript::new(&SessionContext {
        protocol: Ascii::new(text(&context["protocol"])).expect("ASCII"),
        mode: match text(&context["mode"]) {
            "Live" => Mode::Live,
            "NonInteractive" => Mode::NonInteractive,
            other => panic!("mode {other}"),
        },
        params: ParamSetId(Ascii::new(text(&context["params"])).expect("ASCII")),
        statement_digest: bytes(text(&context["statement_digest"]))
            .try_into()
            .expect("32 bytes"),
        session_id: bytes(text(&context["session_id"]))
            .try_into()
            .expect("16 bytes"),
    })
}

/// The verifier's transcript of the same session.
fn tv_transcript(context: &Value) -> tv_hash::transcript::Transcript {
    use tv_hash::transcript as t;
    let digest = bytes(text(&context["statement_digest"]));
    t::Transcript::new(&t::SessionContext {
        protocol: t::Ascii::new(text(&context["protocol"])).expect("ASCII"),
        mode: match text(&context["mode"]) {
            "Live" => t::Mode::Live,
            "NonInteractive" => t::Mode::NonInteractive,
            other => panic!("mode {other}"),
        },
        params: t::ParamSetId(t::Ascii::new(text(&context["params"])).expect("ASCII")),
        statement_digest: digest.try_into().expect("32 bytes"),
        session_id: bytes(text(&context["session_id"]))
            .try_into()
            .expect("16 bytes"),
    })
}

fn tv_prefix(prefix: Prefix) -> tv::Prefix {
    match prefix {
        Prefix::Stage(k) => tv::Prefix::Stage(k),
        Prefix::Tree(level) => tv::Prefix::Tree(level),
    }
}

/// Checks a stage's proof with the verifier, from the instances' recorded claims, and that it
/// draws the vectors' challenges.
fn verify_stage(
    vt: &mut tv_hash::transcript::Transcript,
    stage: &Value,
    proof: &tv::SumcheckProof,
    at: &str,
) {
    let instances: Vec<tv::Instance> = list(&stage["instances"])
        .iter()
        .map(|r| tv::Instance {
            claim: tv_fp2(fp2(&r["claim"])),
            vars: number(&r["vars"]),
            evals: number(&r["factors"]),
        })
        .collect();
    let product =
        |_: usize, _: &[TvFp2], evals: &[TvFp2]| Some(evals.iter().fold(TvFp2::ONE, |p, e| p * *e));
    let batch = tv::verify_batch(
        vt,
        tv_prefix(prefix(&stage["prefix"])),
        number(&stage["degree"]),
        &instances,
        proof,
        product,
    )
    .unwrap_or_else(|e| panic!("{at}: the verifier rejects the proof: {e}"));
    assert_eq!(
        batch.gamma,
        tv_fp2(fp2(&stage["gamma"])),
        "{at}: the verifier's γ"
    );
    assert_eq!(
        batch.point,
        tv_fp2s(&fp2s(&stage["challenges"])),
        "{at}: the verifier's point"
    );
}

fn assert_state(t: &Transcript, record: &Value, at: &str) {
    assert_eq!(
        t.state().to_vec(),
        bytes(text(&record["state"])),
        "{at}: state"
    );
    assert_eq!(
        t.counter(),
        record["counter"].as_u64().expect("a counter"),
        "{at}: counter"
    );
}

/// Replays a recorded proof through the transcript with `tessera-hash` alone: `γ`, each round
/// message and challenge, and the final evaluations.
fn replay(t: &mut Transcript, stage: &Value, at: &str) {
    let p = text(&stage["prefix"]);
    assert_eq!(
        t.challenge_fp2(&format!("{p}/gamma")),
        fp2(&stage["gamma"]),
        "{at}: γ"
    );
    for (j, (round, r)) in list(&stage["rounds"])
        .iter()
        .zip(list(&stage["challenges"]))
        .enumerate()
    {
        t.absorb(&format!("{p}/round/{j}"), &fp2s(round));
        assert_eq!(
            t.challenge_fp2(&format!("{p}/r/{j}")),
            fp2(r),
            "{at}: r_{j}"
        );
    }
    t.absorb(&format!("{p}/evals"), &fp2s(&stage["final_evals"]));
}

#[test]
fn stages_with_tables_are_reproduced_message_by_message() {
    let file = vectors();
    let mut reproduced = Vec::new();
    for session in list(&file["sessions"]) {
        let name = text(&session["name"]);
        let mut t = transcript(&session["context"]);
        let mut vt = tv_transcript(&session["context"]);
        for stage in list(&session["stages"]) {
            let at = format!("{name}, {}", text(&stage["prefix"]));
            let records = list(&stage["instances"]);
            if records[0].get("tables").is_none() {
                assert!(records.iter().all(|r| r.get("tables").is_none()));
                replay(&mut t, stage, &at);
                let recorded = tv::SumcheckProof {
                    rounds: list(&stage["rounds"])
                        .iter()
                        .map(|r| tv_fp2s(&fp2s(r)))
                        .collect(),
                    final_evals: tv_fp2s(&fp2s(&stage["final_evals"])),
                };
                verify_stage(&mut vt, stage, &recorded, &at);
                assert_state(&t, stage, &at);
                assert_eq!(vt.state(), &t.state(), "{at}: the verifier's state");
                continue;
            }
            let tables: Vec<Vec<Vec<Fp2>>> = records
                .iter()
                .map(|r| list(&r["tables"]).iter().map(fp2s).collect())
                .collect();
            let instances: Vec<Instance<'_, Fp2>> = tables
                .iter()
                .map(|ts| Instance::new(ts.iter().map(|t| Table::ext(t)).collect()))
                .collect();
            let degree = number(&stage["degree"]);
            let proven = prove_batch(&mut t, prefix(&stage["prefix"]), degree, &instances)
                .expect("a well-formed stage");
            assert_eq!(*proven.gamma(), fp2(&stage["gamma"]), "{at}: γ");
            let rounds: Vec<Vec<Fp2>> = list(&stage["rounds"]).iter().map(fp2s).collect();
            assert!(proven.proof().rounds == rounds, "{at}: the round messages");
            assert!(
                proven.point() == fp2s(&stage["challenges"]),
                "{at}: the challenges"
            );
            assert!(
                proven.proof().final_evals == fp2s(&stage["final_evals"]),
                "{at}: the final evaluations"
            );
            for (i, (point, value)) in list(&stage["points"])
                .iter()
                .zip(list(&stage["values"]))
                .enumerate()
            {
                assert!(
                    proven.instance_point(i) == Some(&fp2s(point)[..]),
                    "{at}: point {i}"
                );
                let evals = proven.instance_evals(i).expect("an instance");
                let product = evals.iter().fold(Fp2::ONE, |p, &e| p * e);
                assert!(product == fp2(value), "{at}: the value of instance {i}");
            }
            verify_stage(&mut vt, stage, &tv_proof(proven.proof()), &at);
            assert_state(&t, stage, &at);
            assert_eq!(vt.state(), &t.state(), "{at}: the verifier's state");
            reproduced.push((degree, records.len(), at));
        }
    }
    // Every degree, from 1 to 7, and the padded batch of degree 7.
    assert_eq!(reproduced.len(), 31);
    for degree in 1..=7 {
        assert!(
            reproduced.iter().any(|(d, _, _)| *d == degree),
            "degree {degree}"
        );
    }
    assert!(reproduced.iter().any(|(d, m, _)| *d == 7 && *m == 3));
}

/// The levels of a product tree from its leaves, the root first:
/// `V_ℓ[x] = V_{ℓ+1}[2x]·V_{ℓ+1}[2x + 1]`.
fn tree_levels(leaves: Vec<Fp>) -> Vec<Vec<Fp>> {
    let mut levels = vec![leaves];
    while levels[0].len() > 1 {
        let up = levels[0].chunks_exact(2).map(|p| p[0] * p[1]).collect();
        levels.insert(0, up);
    }
    levels
}

#[test]
fn tree_levels_are_reproduced_from_the_leaves() {
    let file = vectors();
    let mut levels_proven = 0;
    for batch in list(&file["tree_batches"]) {
        let name = text(&batch["name"]);
        let mut t = transcript(&batch["context"]);
        let mut vt = tv_transcript(&batch["context"]);
        let trees: Vec<Vec<Vec<Fp>>> = list(&batch["trees"])
            .iter()
            .map(|tree| tree_levels(list(&tree["leaves"]).iter().map(fp).collect()))
            .collect();
        let tallest = trees
            .iter()
            .map(|levels| levels.len() - 1)
            .max()
            .expect("a tree");
        // Each tree's current claim's point, once it has joined.
        let mut points: Vec<Option<Vec<Fp2>>> = vec![None; trees.len()];
        for (level, record) in list(&batch["levels"]).iter().enumerate() {
            let at = format!("{name}, level {level}");
            assert_eq!(prefix(&record["prefix"]), Prefix::Tree(level as u32));
            let entering = list(&record["instances"]);
            let mut tables = Vec::new();
            for claim in entering {
                let tree = number(&claim["tree"]);
                let height = trees[tree].len() - 1;
                if tallest - height == level {
                    points[tree] = Some(Vec::new()); // the root claim, on no variable
                }
                let point = points[tree].clone().expect("a tree that has joined");
                assert!(
                    point == fp2s(&claim["point"]),
                    "{at}: the point of tree {tree}"
                );
                let below = &trees[tree][point.len() + 1];
                let left: Vec<Fp> = below.iter().step_by(2).copied().collect();
                let right: Vec<Fp> = below.iter().skip(1).step_by(2).copied().collect();
                tables.push((eq_table(&point).expect("small"), left, right));
            }
            let instances: Vec<Instance<'_, Fp2>> = tables
                .iter()
                .map(|(eq, left, right)| {
                    Instance::new(vec![
                        Table::ext(eq).public(),
                        Table::base(left),
                        Table::base(right),
                    ])
                })
                .collect();
            let proven = prove_batch(&mut t, Prefix::Tree(level as u32), 3, &instances)
                .expect("a well-formed level");
            assert_eq!(*proven.gamma(), fp2(&record["gamma"]), "{at}: γ");
            let rounds: Vec<Vec<Fp2>> = list(&record["rounds"]).iter().map(fp2s).collect();
            assert!(proven.proof().rounds == rounds, "{at}: the round messages");
            assert!(
                proven.point() == fp2s(&record["challenges"]),
                "{at}: the challenges"
            );
            assert!(
                proven.proof().final_evals == fp2s(&record["final_evals"]),
                "{at}: u_0, w_0, u_1, w_1, …"
            );
            // The caller's part of a level: ρ, and each next claim on the line through u and w.
            let rho = t.challenge_fp2(&format!("tree/{level}/line"));
            assert_eq!(rho, fp2(&record["rho"]), "{at}: ρ");
            let claims: Vec<TreeClaim> = entering
                .iter()
                .map(|c| TreeClaim {
                    point: tv_fp2s(&fp2s(&c["point"])),
                    value: tv_fp2(fp2(&c["value"])),
                })
                .collect();
            let checked =
                verify_product_level(&mut vt, level as u32, &claims, &tv_proof(proven.proof()))
                    .unwrap_or_else(|e| panic!("{at}: the verifier rejects the level: {e}"));
            assert_eq!(
                checked.gamma,
                tv_fp2(*proven.gamma()),
                "{at}: the verifier's γ"
            );
            assert_eq!(
                checked.point,
                tv_fp2s(proven.point()),
                "{at}: the verifier's point"
            );
            assert_eq!(checked.rho, tv_fp2(rho), "{at}: the verifier's ρ");
            assert_eq!(vt.state(), &t.state(), "{at}: the verifier's state");
            for (i, (claim, next)) in entering.iter().zip(list(&record["next"])).enumerate() {
                let tree = number(&claim["tree"]);
                let [u, w] = proven.instance_evals(i).expect("an instance") else {
                    panic!("{at}: two evaluations per instance");
                };
                let mut point = vec![rho];
                point.extend_from_slice(proven.instance_point(i).expect("an instance"));
                assert!(
                    point == fp2s(&next["point"]),
                    "{at}: the next point of tree {tree}"
                );
                assert_eq!(
                    *u + rho * (*w - *u),
                    fp2(&next["value"]),
                    "{at}: the next value"
                );
                let verified = &checked.claims[i];
                assert_eq!(
                    verified.point,
                    tv_fp2s(&point),
                    "{at}: the verifier's next point"
                );
                assert_eq!(
                    verified.value,
                    tv_fp2(*u + rho * (*w - *u)),
                    "{at}: the verifier's next value"
                );
                points[tree] = Some(point);
            }
            assert_state(&t, record, &at);
            levels_proven += 1;
        }
    }
    assert_eq!(levels_proven, 39);
}
