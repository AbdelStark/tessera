// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Conformance with the independent Python oracle, `spec/vectors/sumcheck/generate.py`, through
//! the `logup_batches` of `spec/vectors/sumcheck/sumcheck.json` (RFC-0008 'Fraction tree',
//! RFC-0010 'Tree levels', RFC-0014 'Transcript schedule' rows 4 and 5).
//!
//! Every honest batch is proven again by `prove_logup` from its instances (the rows, the lookups'
//! rows and the recorded multiplicities), its forests and its schedule, the `trees` list, on a
//! transcript of its context. The prover must reproduce every recorded value: `α`, `β`, the roots
//! message; at every level the root challenges, `γ`, `λ` or its absence, the rounds, the
//! challenges, the claims after each round (derived as a verifier does, with
//! `tessera_poly::RoundPoly`), the final evaluations, `ρ`, the next claims and the transcript's
//! state and counter; the values message; every leaf check's claims, values and key; and the
//! final state and counter. The honest multiplicities are also those `multiplicities` counts from
//! the lookups' rows, except where a batch counts the lookup padding on a padding row.
//!
//! The independent verifier checks every reproduced proof with the driver of `common`, which
//! must accept and pass through every recorded state; the table evaluation it computes must be
//! the recorded `table_eval`.

mod common;

use std::fs;
use std::path::PathBuf;

use common::{
    Fp, Fp2, Messages, Session, TestForest, TestInstance, TestTable, tv_fp2, tv_fp2s, verify,
};
use serde_json::Value;
use tessera_codec::{Ascii, Secret};
use tessera_field::{Field, PrimeField, QuadraticExtension};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_logup::{LogUpInstance, TreeId, multiplicities, prove_logup};
use tessera_poly::RoundPoly;
use tessera_sumcheck::{Forest, LevelClaim};

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

fn mode(context: &Value) -> Mode {
    match text(&context["mode"]) {
        "Live" => Mode::Live,
        "NonInteractive" => Mode::NonInteractive,
        other => panic!("mode {other}"),
    }
}

fn session(context: &Value) -> Session {
    assert_eq!(text(&context["protocol"]), common::PROTOCOL);
    assert_eq!(text(&context["params"]), "test-small/1");
    Session {
        mode: mode(context),
        statement_digest: bytes(text(&context["statement_digest"]))
            .try_into()
            .expect("32 bytes"),
        session_id: bytes(text(&context["session_id"]))
            .try_into()
            .expect("16 bytes"),
    }
}

fn transcript(context: &Value) -> Transcript {
    Transcript::new(&SessionContext {
        protocol: Ascii::new(text(&context["protocol"])).expect("ASCII"),
        mode: mode(context),
        params: ParamSetId(Ascii::new(text(&context["params"])).expect("ASCII")),
        statement_digest: bytes(text(&context["statement_digest"]))
            .try_into()
            .expect("32 bytes"),
        session_id: bytes(text(&context["session_id"]))
            .try_into()
            .expect("16 bytes"),
    })
}

/// An honest batch's instance: its rows, its lookups' rows and its recorded multiplicities.
fn instance(record: &Value) -> (TestInstance, Vec<usize>) {
    let rows = list(&record["rows"]);
    let has_values = record["has_values"].as_bool().expect("a flag");
    let table = TestTable {
        keys: rows.iter().map(|row| fp(&row[0])).collect(),
        values: has_values.then(|| rows.iter().map(|row| fp(&row[1])).collect()),
    };
    if !has_values {
        assert!(
            rows.iter().all(|row| fp(&row[1]) == Fp::ZERO),
            "no values: 0"
        );
    }
    let lookups: Vec<usize> = list(&record["lookups"]).iter().map(number).collect();
    let mut instance = TestInstance::of_rows(table, &lookups);
    instance.multiplicities = list(&record["multiplicities"]).iter().map(fp).collect();
    assert_eq!(number(&record["h_a"]), instance.lookup_height());
    assert_eq!(number(&record["h_t"]), instance.table_height());
    (instance, lookups)
}

fn forest(record: &Value) -> TestForest {
    TestForest {
        trees_log: number(&record["rows_log"]),
        height: number(&record["height"]),
        leaves: list(&record["leaves"]).iter().map(fp).collect(),
    }
}

fn schedule(trees: &Value) -> Vec<TreeId> {
    list(trees)
        .iter()
        .map(|tree| {
            let index = number(&tree["index"]);
            match text(&tree["kind"]) {
                "lookup" => TreeId::Lookup(index),
                "table" => TreeId::Table(index),
                "forest" => TreeId::Forest(index),
                other => panic!("kind {other}"),
            }
        })
        .collect()
}

/// A recorded claim of a level, `(point, values)`: `[numerator, denominator]` for a fraction
/// tree, `[value]` for a forest.
fn recorded(claim: &Value) -> (Vec<Fp2>, Vec<Fp2>) {
    let point = fp2s(&claim["point"]);
    match claim.get("value") {
        Some(value) => (point, vec![fp2(value)]),
        None => (
            point,
            vec![fp2(&claim["numerator"]), fp2(&claim["denominator"])],
        ),
    }
}

fn mine(claim: &LevelClaim<Fp2>) -> (Vec<Fp2>, Vec<Fp2>) {
    match claim {
        LevelClaim::Product(c) => (c.point.clone(), vec![c.value]),
        LevelClaim::Fraction(c) => (c.point.clone(), vec![c.numerator, c.denominator]),
    }
}

/// `eq(a, b) = Π_i (a_i·b_i + (1 − a_i)·(1 − b_i))`.
fn eq_points(a: &[Fp2], b: &[Fp2]) -> Fp2 {
    a.iter().zip(b).fold(Fp2::ONE, |product, (&a, &b)| {
        product * (a * b + (Fp2::ONE - a) * (Fp2::ONE - b))
    })
}

#[test]
fn honest_logup_batches_are_reproduced_value_by_value() {
    let file = vectors();
    let mut reproduced = Vec::new();
    let mut skipped = Vec::new();
    for batch in list(&file["logup_batches"]) {
        let name = text(&batch["name"]);
        if batch["expected"] != "accept" {
            skipped.push(name);
            continue;
        }
        let parsed: Vec<(TestInstance, Vec<usize>)> =
            list(&batch["instances"]).iter().map(instance).collect();
        let instances: Vec<TestInstance> = parsed.iter().map(|(i, _)| i.clone()).collect();
        let forests: Vec<TestForest> = list(&batch["forests"]).iter().map(forest).collect();
        let schedule = schedule(&batch["trees"]);

        // The recorded multiplicities are those an honest prover counts, but where the batch
        // counts the lookup tree's padding on a padding row instead of row 0.
        for (k, (instance, rows)) in parsed.iter().enumerate() {
            let counted = multiplicities::<Fp>(instance.table.tuples(), &Secret::new(u32s(rows)))
                .expect("rows");
            let canonical = counted[..] == instance.multiplicities[..];
            assert_eq!(
                canonical,
                !name.contains("padding row"),
                "{name}: instance {k}'s multiplicities"
            );
        }

        let borrowed: Vec<LogUpInstance<'_, Fp>> =
            instances.iter().map(TestInstance::instance).collect();
        let borrowed_forests: Vec<Forest<'_, Fp2>> =
            forests.iter().map(TestForest::forest).collect();
        let mut t = transcript(&batch["context"]);
        let proven = prove_logup(&mut t, &borrowed, &borrowed_forests, &schedule)
            .unwrap_or_else(|e| panic!("{name}: the prover refuses the batch: {e}"));

        assert_eq!(*proven.alpha(), fp2(&batch["alpha"]), "{name}: α");
        assert_eq!(*proven.beta(), fp2(&batch["beta"]), "{name}: β");
        assert!(proven.roots() == fp2s(&batch["roots"]), "{name}: the roots");

        let levels = list(&batch["levels"]);
        assert_eq!(proven.levels().len(), levels.len(), "{name}: the levels");
        for (level, (record, mine_level)) in levels.iter().zip(proven.levels()).enumerate() {
            let at = format!("{name}, level {level}");
            assert_eq!(text(&record["prefix"]), format!("tree/{level}"), "{at}");
            assert!(
                mine_level.root_challenges() == fp2s(&record["root_challenges"]),
                "{at}: the root challenges"
            );
            assert_eq!(*mine_level.gamma(), fp2(&record["gamma"]), "{at}: γ");
            match &record["lambda"] {
                Value::Null => assert!(mine_level.lambda().is_none(), "{at}: no λ"),
                lambda => assert_eq!(mine_level.lambda(), Some(&fp2(lambda)), "{at}: λ"),
            }
            let entering = list(&record["instances"]);
            let present: Vec<usize> = entering.iter().map(|c| number(&c["tree"])).collect();
            assert_eq!(mine_level.trees(), present, "{at}: the instances");
            let rounds: Vec<Vec<Fp2>> = list(&record["rounds"]).iter().map(fp2s).collect();
            assert!(mine_level.proof().rounds == rounds, "{at}: the rounds");
            assert!(
                mine_level.point() == fp2s(&record["challenges"]),
                "{at}: the challenges"
            );
            assert!(
                mine_level.proof().final_evals == fp2s(&record["final_evals"]),
                "{at}: the final evaluations"
            );
            assert_eq!(*mine_level.rho(), fp2(&record["rho"]), "{at}: ρ");

            // The claims a verifier derives: the batched claim, then p_j(r_j) round by round.
            let gamma = *mine_level.gamma();
            let lambda = mine_level.lambda().copied().unwrap_or(Fp2::ZERO);
            let n_max = mine_level.point().len();
            let mut claim = Fp2::ZERO;
            let mut weight = Fp2::ONE;
            for c in entering {
                let (point, values) = recorded(c);
                let scale = Fp2::from(Fp::from_u64_reduced(1 << (n_max - point.len())));
                let value = match values[..] {
                    [v] => v,
                    [p, q] => p + lambda * q,
                    _ => unreachable!("one or two values"),
                };
                claim += weight * scale * value;
                weight *= gamma;
            }
            let mut claims = vec![claim];
            for (round, &r) in rounds.iter().zip(mine_level.point()) {
                claim = RoundPoly::new(round).expect("degree 3").evaluate(claim, r);
                claims.push(claim);
            }
            assert!(claims == fp2s(&record["claims"]), "{at}: the claims");
            // The final check, from the final evaluations: Σ_i γ^i·eq(r_i, r'_i)·g_i.
            let mut batched = Fp2::ZERO;
            let mut weight = Fp2::ONE;
            let mut evals = mine_level.proof().final_evals.iter().copied();
            for c in entering {
                let (point, values) = recorded(c);
                let own = &mine_level.point()[n_max - point.len()..];
                let g = match values.len() {
                    1 => {
                        let (u, w) = (evals.next().expect("u"), evals.next().expect("w"));
                        u * w
                    }
                    _ => {
                        let mut next = || evals.next().expect("an evaluation");
                        let (n0, n1, d0, d1) = (next(), next(), next(), next());
                        n0 * d1 + n1 * d0 + lambda * d0 * d1
                    }
                };
                batched += weight * eq_points(&point, own) * g;
                weight *= gamma;
            }
            assert_eq!(batched, claim, "{at}: the final check");

            let next = list(&record["next"]);
            assert_eq!(
                next.len(),
                mine_level.next_claims().len(),
                "{at}: next claims"
            );
            for (k, (n, claim)) in next.iter().zip(mine_level.next_claims()).enumerate() {
                assert_eq!(
                    number(&n["tree"]),
                    present[k],
                    "{at}: next claim {k}'s tree"
                );
                let (point, values) = mine(claim);
                assert!((point, values) == recorded(n), "{at}: next claim {k}");
            }
        }

        assert!(
            proven.values() == fp2s(&batch["values"]),
            "{name}: the values"
        );
        let checks = list(&batch["leaf_checks"]);
        assert_eq!(
            checks.len(),
            instances.len(),
            "{name}: a leaf check per instance"
        );
        for (k, ((check, claims), instance)) in checks
            .iter()
            .zip(proven.claims())
            .zip(&instances)
            .enumerate()
        {
            let at = format!("{name}, instance {k}");
            let lookup = &check["lookup"];
            let table = &check["table"];
            let position = |id: TreeId| schedule.iter().position(|&x| x == id).expect("named");
            let leaf = |id: TreeId| mine(&proven.leaf_claims()[position(id)]);
            assert!(
                leaf(TreeId::Lookup(k))
                    == (
                        fp2s(&lookup["point"]),
                        vec![fp2(&lookup["numerator"]), fp2(&lookup["denominator"])]
                    ),
                "{at}: the lookup tree's leaf claims"
            );
            assert!(
                leaf(TreeId::Table(k))
                    == (
                        fp2s(&table["point"]),
                        vec![fp2(&table["numerator"]), fp2(&table["denominator"])]
                    ),
                "{at}: the table tree's leaf claims"
            );
            assert!(
                claims.key.point == fp2s(&lookup["point"]),
                "{at}: kẽy's point"
            );
            assert_eq!(claims.key.value, fp2(&lookup["key"]), "{at}: kẽy(ρ_a)");
            match &lookup["value"] {
                Value::Null => assert!(claims.value.is_none(), "{at}: no value claim"),
                value => {
                    let claim = claims.value.as_ref().expect("a value claim");
                    assert!(claim.point == fp2s(&lookup["point"]), "{at}: valũe's point");
                    assert_eq!(claim.value, fp2(value), "{at}: valũe(ρ_a)");
                }
            }
            assert!(
                claims.multiplicity.point == fp2s(&table["point"]),
                "{at}: m̃'s point"
            );
            assert_eq!(
                claims.multiplicity.value,
                fp2(&table["multiplicity"]),
                "{at}: m̃(ρ_t)"
            );
            let rho_t = tv_fp2s(&fp2s(&table["point"]));
            let beta = tv_fp2(*proven.beta());
            assert_eq!(
                instance.table.tv_eval(&rho_t, beta),
                tv_fp2(fp2(&table["table_eval"])),
                "{at}: t̃(ρ_t), its padding rows copies of row 0"
            );
        }
        assert_eq!(
            t.state().to_vec(),
            bytes(text(&batch["state"])),
            "{name}: state"
        );
        assert_eq!(
            t.counter(),
            batch["counter"].as_u64().expect("a counter"),
            "{name}: counter"
        );

        // The verifier accepts the reproduced proof, through every recorded state.
        let session = session(&batch["context"]);
        let verified = verify(
            &session,
            &instances,
            &forests,
            &schedule,
            &Messages::of(&proven),
        )
        .unwrap_or_else(|rejected| panic!("{name}: the verifier rejects: {rejected:?}"));
        assert_eq!(
            verified.state.to_vec(),
            bytes(text(&batch["state"])),
            "{name}"
        );
        reproduced.push(name);
    }
    assert_eq!(reproduced.len(), 10, "the honest batches: {reproduced:?}");
    assert_eq!(skipped.len(), 11, "the cheating batches: {skipped:?}");
}

/// The state after each level, replayed from the reproduced messages through a `tessera-hash`
/// transcript: the recorded `state` and `counter` of every level.
#[test]
fn every_level_ends_in_the_recorded_state() {
    let file = vectors();
    let mut levels_checked = 0;
    for batch in list(&file["logup_batches"]) {
        if batch["expected"] != "accept" {
            continue;
        }
        let name = text(&batch["name"]);
        let mut t = transcript(&batch["context"]);
        assert_eq!(
            t.challenge_fp2("logup/alpha"),
            fp2(&batch["alpha"]),
            "{name}"
        );
        assert_eq!(t.challenge_fp2("logup/beta"), fp2(&batch["beta"]), "{name}");
        t.absorb("logup/roots", &fp2s(&batch["roots"]));
        for (level, record) in list(&batch["levels"]).iter().enumerate() {
            let p = format!("tree/{level}");
            for (c, challenge) in list(&record["root_challenges"]).iter().enumerate() {
                assert_eq!(t.challenge_fp2(&format!("{p}/root/{c}")), fp2(challenge));
            }
            assert_eq!(
                t.challenge_fp2(&format!("{p}/gamma")),
                fp2(&record["gamma"])
            );
            if !record["lambda"].is_null() {
                assert_eq!(
                    t.challenge_fp2(&format!("{p}/lambda")),
                    fp2(&record["lambda"])
                );
            }
            for (j, (round, r)) in list(&record["rounds"])
                .iter()
                .zip(list(&record["challenges"]))
                .enumerate()
            {
                t.absorb(&format!("{p}/round/{j}"), &fp2s(round));
                assert_eq!(t.challenge_fp2(&format!("{p}/r/{j}")), fp2(r));
            }
            t.absorb(&format!("{p}/evals"), &fp2s(&record["final_evals"]));
            assert_eq!(t.challenge_fp2(&format!("{p}/line")), fp2(&record["rho"]));
            assert_eq!(
                t.state().to_vec(),
                bytes(text(&record["state"])),
                "{name}, {p}"
            );
            assert_eq!(t.counter(), record["counter"].as_u64().expect("a counter"));
            levels_checked += 1;
        }
        t.absorb("logup/values", &fp2s(&batch["values"]));
        assert_eq!(t.state().to_vec(), bytes(text(&batch["state"])), "{name}");
    }
    assert_eq!(levels_checked, 29);
}

fn u32s(rows: &[usize]) -> Vec<u32> {
    rows.iter().map(|&r| r as u32).collect()
}
