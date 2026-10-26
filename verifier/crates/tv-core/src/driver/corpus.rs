// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The SC-4 tamper corpus (07 §4) against this verifier: every mutation in
//! `spec/vectors/tamper/protocol-1.json` must yield its predeclared verdict, exactly.
//! The corpus and its base messages are data; nothing of the prover enters this workspace.
use super::*;
use protocol::{
    BatchDescriptor, ColumnQueries, RequestRef, Round1, Round2Response, Round3Response,
};
use std::{collections::BTreeMap, fs};
use tv_codec::Envelope;
use tv_hash::{Domain, hash};
use tv_testkit::{Json, hex};

fn field<'a>(value: &'a Json, key: &str) -> &'a Json {
    value
        .get(key)
        .unwrap_or_else(|| panic!("corpus field {key}"))
}
fn text<'a>(value: &'a Json, key: &str) -> &'a str {
    field(value, key).as_str().unwrap()
}
fn number(value: &Json, key: &str) -> u64 {
    field(value, key).as_u64().unwrap()
}
fn base(name: &str) -> Vec<u8> {
    fs::read(tv_testkit::vectors_dir().join("tamper").join(name)).unwrap()
}

/// RFC-0003 'Envelope': the header's first 12 bytes, the payload's length, the payload, and
/// the digest of everything before it.
fn reseal(envelope: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut out = envelope[..12].to_vec();
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(payload);
    out.extend_from_slice(&hash(Domain::Envelope, &out));
    out
}
/// Replaces `before` at `at` by `after`; `None` when `before` is not there.
fn splice(bytes: &[u8], at: usize, before: &[u8], after: &[u8]) -> Option<Vec<u8>> {
    let end = at.checked_add(before.len())?;
    if bytes.get(at..end)? != before {
        return None;
    }
    let mut out = bytes[..at].to_vec();
    out.extend_from_slice(after);
    out.extend_from_slice(&bytes[end..]);
    Some(out)
}
/// RFC-0002's modulus.
const GOLDILOCKS: u64 = 0xFFFF_FFFF_0000_0001;
/// Adds `delta` to `count` canonical Fp2 values from `at`, coordinate-wise modulo P.
fn shift_fp2(bytes: &[u8], at: usize, shift: &Json) -> Option<Vec<u8>> {
    let count = usize::try_from(shift.get("count")?.as_u64()?).ok()?;
    let delta = shift.get("delta")?.as_array()?;
    let delta = [delta.first()?.as_u64()?, delta.get(1)?.as_u64()?];
    let end = at.checked_add(count.checked_mul(16)?)?;
    let mut out = bytes.to_vec();
    // Word 2i is coordinate c0 of value i, word 2i + 1 its c1.
    for (i, word) in out.get_mut(at..end)?.chunks_exact_mut(8).enumerate() {
        let d = delta[i % 2];
        let x = u64::from_le_bytes(word.try_into().ok()?);
        if x >= GOLDILOCKS || d >= GOLDILOCKS {
            return None;
        }
        let sum = (u128::from(x) + u128::from(d)) % u128::from(GOLDILOCKS);
        word.copy_from_slice(&u64::try_from(sum).ok()?.to_le_bytes());
    }
    Some(out)
}
/// The case's version of every base it edits. Edits come in decreasing offset per target, so
/// each applies to bytes the previous ones left in place.
fn mutate(case: &Json) -> BTreeMap<String, Vec<u8>> {
    let reseal_payload = match field(case, "reseal") {
        Json::Bool(b) => *b,
        _ => panic!("reseal is a boolean"),
    };
    let mut out: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for edit in field(case, "edits").as_array().unwrap() {
        let target = text(edit, "target");
        let original = base(target);
        let current = out.entry(target.into()).or_insert_with(|| {
            if reseal_payload {
                original[20..original.len() - 32].to_vec()
            } else {
                original.clone()
            }
        });
        let at = usize::try_from(number(edit, "offset")).unwrap();
        let edited = match edit.get("shift_fp2") {
            Some(shift) => shift_fp2(current, at, shift),
            None => splice(
                current,
                at,
                &hex(text(edit, "before")),
                &hex(text(edit, "after")),
            ),
        };
        *current =
            edited.unwrap_or_else(|| panic!("{}: edit does not match its base", text(case, "id")));
    }
    if reseal_payload {
        for (target, payload) in &mut out {
            *payload = reseal(&base(target), payload);
        }
    }
    out
}

struct Fixture {
    corpus: Json,
    claims: ClaimList,
    expected: PublicStatement,
    config: VerifierConfig,
    seed: Digest,
    queries: ColumnQueries,
}
impl Fixture {
    /// The caller's trusted inputs come from the claim list, the epoch's public weight artifact
    /// and the request metadata, never from a proof; both base sessions must accept first.
    fn new() -> Self {
        let corpus = tv_testkit::read_vectors("tamper/protocol-1.json");
        assert_eq!(text(&corpus, "schema"), "tessera.tamper-corpus/1");
        assert_eq!(text(&corpus, "protocol"), "tessera-protocol/1");
        let meta = field(&corpus, "base");
        for (name, digest) in match field(&corpus, "base_sha256") {
            Json::Object(entries) => entries.clone(),
            _ => panic!("base_sha256 is an object"),
        } {
            let actual: String = tv_hash::sha256::sha256(&base(&name))
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            assert_eq!(Some(actual.as_str()), digest.as_str(), "{name} changed");
        }
        let limits = DecodeLimits::MESSAGE;
        let claims = ClaimList::open(&base("claims.tcl"), &limits).unwrap();
        let weights = Envelope::open(
            &base("weights.twr"),
            WeightCommitmentPublic::EXPECTED,
            &limits,
            |b| hash(Domain::Envelope, b),
        )
        .unwrap()
        .decode_payload(|c, _| WeightCommitmentPublic::decode(c))
        .unwrap();
        let request = field(meta, "request");
        let expected = PublicStatement {
            claim_list_digest: claims.digest(),
            weights_root: weights.weights_root(|b| hash(Domain::WeightsRoot, b)),
            weights_epoch: weights.epoch,
            manifest_digest: hex(text(request, "manifest_digest")).try_into().unwrap(),
            batch: BatchDescriptor {
                requests: vec![RequestRef {
                    request_id: number(request, "request_id"),
                    session_id: hex(text(request, "session_id")).try_into().unwrap(),
                    prompt_len: u32::try_from(number(request, "prompt_len")).unwrap(),
                    output_len: u32::try_from(number(request, "output_len")).unwrap(),
                }],
            },
            predictor: PredictorSpec::Exact {
                decoding: Decoding::Greedy,
            },
        };
        assert_eq!(
            expected.digest().as_slice(),
            hex(text(meta, "statement_digest"))
        );
        // The producer's lookup metadata, which the corpus reads, against this verifier's own
        // derivation from the claim list (RFC-0045 'The verifier's derivation').
        let validated = validation::validate(&claims).unwrap();
        let n = expected.batch.requests[0].prompt_len + expected.batch.requests[0].output_len;
        let graph = crate::lowering::lower(&validation::check_public(&claims).unwrap(), n).unwrap();
        let runtime = gadgets::runtime::Plan::derive(
            &graph,
            claims.gadget_policy,
            expected.batch.requests[0].prompt_len,
            0,
            validated.tables(),
        )
        .unwrap();
        let derived: Vec<u64> = runtime
            .leaf_schedule()
            .unwrap()
            .lookups()
            .iter()
            .map(|(id, _)| u64::from(id.0))
            .collect();
        let declared: Vec<u64> = field(meta, "lookup_tables")
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        assert_eq!(derived, declared, "lookup instances");
        let fixture = Self {
            claims,
            expected,
            config: VerifierConfig {
                params: ParamSet::TestSmall,
                limits: ResourceLimits::default(),
                accept_bound: FixedBits(0),
                weights,
            },
            seed: base("live-seed.bin").try_into().unwrap(),
            queries: ColumnQueries::open(&base("live-queries.tce"), &limits).unwrap(),
            corpus,
        };
        for session in ["ni.tnp", "live-round2.tce"] {
            let verdict = fixture.run(session, &BTreeMap::new());
            assert!(
                matches!(verdict, Verdict::Accept(_)),
                "{session}: {verdict:?}"
            );
        }
        fixture
    }
    /// One session over the base messages, with `edited` in place of their originals. A case
    /// that edits `ni.tnp` judges the non-interactive proof, any other the recorded live one.
    fn run(&self, session: &str, edited: &BTreeMap<String, Vec<u8>>) -> Verdict {
        let bytes = |name: &str| edited.get(name).cloned().unwrap_or_else(|| base(name));
        if session == "ni.tnp" {
            return verify_non_interactive_bytes(
                &self.claims,
                &self.expected,
                &bytes("ni.tnp"),
                &self.config,
            );
        }
        let limits = DecodeLimits::MESSAGE;
        // Replays a recorded session as the live verifier runs it: each message is opened as the
        // CLI opens it (`open_live`, RFC-0043), only once the phase before it has passed, and
        // judged in its own phase, with the recorded queries in the last; a live verifier draws
        // its own, and the queries the second phase draws are dropped.
        let round1 = match open_live::<Round1>(&bytes("live-round1.tce"), &limits) {
            Ok(m) => m,
            Err(verdict) => return verdict,
        };
        let phase = |round2: &Round2Response, session| {
            let frame = Frame {
                round1: &round1,
                round2,
                round3: &EMPTY_ROUND3,
            };
            match verify_unit(&self.claims, &self.expected, frame, &self.config, session) {
                Ok(Outcome::Ready | Outcome::Queries(_)) => None,
                Ok(Outcome::Accepted(_)) => panic!("an early phase accepts nothing"),
                Err(failure) => Some(Verdict::from(failure)),
            }
        };
        if let Some(verdict) = phase(&EMPTY_ROUND2, Session::Initial) {
            return verdict;
        }
        let round2 = match open_live::<Round2Response>(&bytes("live-round2.tce"), &limits) {
            Ok(m) => m,
            Err(verdict) => return verdict,
        };
        let live = Session::Live {
            seed: &self.seed,
            queries: None,
        };
        if let Some(verdict) = phase(&round2, live) {
            return verdict;
        }
        let round3 = match open_live::<Round3Response>(&bytes("live-round3.tce"), &limits) {
            Ok(m) => m,
            Err(verdict) => return verdict,
        };
        match verify_unit(
            &self.claims,
            &self.expected,
            Frame {
                round1: &round1,
                round2: &round2,
                round3: &round3,
            },
            &self.config,
            Session::Live {
                seed: &self.seed,
                queries: Some(&self.queries),
            },
        ) {
            Ok(Outcome::Accepted(report)) => Verdict::Accept(report),
            Ok(_) => panic!("a recorded live session ends with a verdict"),
            Err(failure) => failure.into(),
        }
    }
}

#[test]
fn every_tamper_case_yields_its_predeclared_verdict() {
    let fixture = Fixture::new();
    let cases = field(&fixture.corpus, "mutations").as_array().unwrap();
    assert_eq!(
        cases.len() as u64,
        number(field(&fixture.corpus, "coverage"), "cases")
    );
    let mut wrong = Vec::new();
    for case in cases {
        let edited = mutate(case);
        assert!(
            !edited.contains_key("ni.tnp") || edited.len() == 1,
            "{}: one session per case",
            text(case, "id")
        );
        let session = if edited.contains_key("ni.tnp") {
            "ni.tnp"
        } else {
            "live-round2.tce"
        };
        let verdict = Json::parse(&fixture.run(session, &edited).to_json().unwrap()).unwrap();
        let actual = field(&verdict, "verdict");
        if actual != field(case, "expected") {
            wrong.push(format!("{}: {actual:?}", text(case, "id")));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong verdicts:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn proof_of_work_rejects_at_the_real_20_bit_boundary() {
    let corpus = tv_testkit::read_vectors("tamper/protocol-1.json");
    let pow = field(field(&corpus, "base"), "pow");
    let context = SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::NonInteractive,
        params: tv_hash::transcript::ParamSetId(Ascii::new(text(pow, "params")).unwrap()),
        statement_digest: hex(text(pow, "statement_digest")).try_into().unwrap(),
        session_id: hex(text(pow, "session_id")).try_into().unwrap(),
    };
    let parameters = SoundnessParams::try_from_param_set(ParamSet::Ni100).unwrap();
    assert_eq!((parameters.seed_pow, parameters.query_pow), (20, 20));
    let cases = field(pow, "cases").as_array().unwrap();
    assert_eq!(cases.len(), 2);
    for case in cases {
        let phase = match text(case, "phase") {
            "seed" => PowPhase::Seed,
            "queries" => PowPhase::Queries,
            other => panic!("phase {other}"),
        };
        let valid = number(case, "valid");
        assert!(proof_of_work(Transcript::new(&context), &parameters, phase, valid).is_ok());
        let invalid = number(case, "invalid");
        assert!(matches!(
            proof_of_work(Transcript::new(&context), &parameters, phase, invalid),
            Err(Failure::Rejected(RejectReason::ProofOfWork { phase: p })) if p == phase
        ));
    }
}

#[test]
fn corpus_edits_and_reseal_are_exact() {
    assert_eq!(splice(b"abcdef", 2, b"cd", b"X"), Some(b"abXef".to_vec()));
    assert_eq!(splice(b"abc", 3, b"", b"X"), Some(b"abcX".to_vec()));
    assert_eq!(splice(b"abc", 2, b"xy", b"X"), None);
    assert_eq!(splice(b"abc", usize::MAX, b"xy", b"X"), None);
    // The protocol vectors' envelopes, written by an independent oracle, reseal to themselves.
    let vectors = tv_testkit::read_vectors("protocol/protocol.json");
    for message in field(&vectors, "messages").as_array().unwrap() {
        let envelope = hex(text(message, "envelope"));
        assert_eq!(reseal(&envelope, &hex(text(message, "payload"))), envelope);
    }
}

/// Not a predeclared corpus: every single value of the recorded live session, changed in turn,
/// must be rejected. About 15 minutes, optimized:
/// `CARGO_PROFILE_TEST_OPT_LEVEL=3 cargo test -p tv-core --features test-params --lib -- --ignored every_value`
#[test]
#[ignore = "exhaustive sweep of every message value; run explicitly"]
fn every_value_of_the_recorded_live_session_is_bound() {
    let f = Fixture::new();
    let limits = DecodeLimits::MESSAGE;
    let mut r1 = Round1::open(&base("live-round1.tce"), &limits).unwrap();
    let mut r2 = Round2Response::open(&base("live-round2.tce"), &limits).unwrap();
    let mut r3 = Round3Response::open(&base("live-round3.tce"), &limits).unwrap();
    let judge = |r1: &Round1, r2: &Round2Response, r3: &Round3Response| {
        verify_unit(
            &f.claims,
            &f.expected,
            Frame {
                round1: r1,
                round2: r2,
                round3: r3,
            },
            &f.config,
            Session::Live {
                seed: &f.seed,
                queries: Some(&f.queries),
            },
        )
    };
    let mut accepted = Vec::new();
    let mut codes: BTreeMap<String, usize> = BTreeMap::new();
    macro_rules! sweep {
        ($label:expr, $place:expr, $bump:expr) => {{
            let old = $place;
            $place = $bump(old);
            match judge(&r1, &r2, &r3) {
                Ok(_) => accepted.push($label),
                Err(failure) => {
                    let verdict = Verdict::from(failure);
                    let json = Json::parse(&verdict.to_json().unwrap()).unwrap();
                    let reason = field(field(&json, "verdict"), "reason");
                    *codes.entry(text(reason, "code").into()).or_default() += 1;
                }
            }
            $place = old;
        }};
    }
    let fp2 = |x: Fp2| x + Fp2::ONE;
    let fp = |x: tv_field::Fp| x + tv_field::Fp::ONE;
    let digest = |mut d: Digest| {
        d[0] ^= 1;
        d
    };
    sweep!(
        "round1.session_id".to_string(),
        r1.session_id,
        |mut s: [u8; 16]| {
            s[0] ^= 1;
            s
        }
    );
    sweep!("round1.advice.root".to_string(), r1.advice.root, digest);
    for i in 0..r2.logup.roots.len() {
        sweep!(format!("roots[{i}]"), r2.logup.roots[i], fp2);
    }
    for i in 0..r2.logup.values.len() {
        sweep!(format!("values[{i}]"), r2.logup.values[i], fp2);
    }
    for l in 0..r2.trees.levels.len() {
        for j in 0..r2.trees.levels[l].rounds.len() {
            for k in 0..r2.trees.levels[l].rounds[j].len() {
                sweep!(
                    format!("level{l}.round{j}[{k}]"),
                    r2.trees.levels[l].rounds[j][k],
                    fp2
                );
            }
        }
        for k in 0..r2.trees.levels[l].final_evals.len() {
            sweep!(
                format!("level{l}.evals[{k}]"),
                r2.trees.levels[l].final_evals[k],
                fp2
            );
        }
    }
    for i in 0..r2.trees.splits.len() {
        sweep!(format!("splits[{i}]"), r2.trees.splits[i], fp2);
    }
    for s in 0..r2.stages.len() {
        for j in 0..r2.stages[s].sumcheck.rounds.len() {
            for k in 0..r2.stages[s].sumcheck.rounds[j].len() {
                sweep!(
                    format!("stage{s}.round{j}[{k}]"),
                    r2.stages[s].sumcheck.rounds[j][k],
                    fp2
                );
            }
        }
        for k in 0..r2.stages[s].sumcheck.final_evals.len() {
            sweep!(
                format!("stage{s}.evals[{k}]"),
                r2.stages[s].sumcheck.final_evals[k],
                fp2
            );
        }
        for k in 0..r2.stages[s].splits.len() {
            sweep!(format!("stage{s}.splits[{k}]"), r2.stages[s].splits[k], fp2);
        }
    }
    for g in 0..r2.advice_combined.len() {
        for i in 0..r2.advice_combined[g].u.len() {
            sweep!(format!("advice{g}.u[{i}]"), r2.advice_combined[g].u[i], fp2);
        }
        for i in 0..r2.advice_combined[g].mask.len() {
            sweep!(
                format!("advice{g}.mask[{i}]"),
                r2.advice_combined[g].mask[i],
                fp2
            );
        }
    }
    for m in 0..r2.weight_combined.len() {
        for g in 0..r2.weight_combined[m].evaluations.len() {
            for i in 0..r2.weight_combined[m].evaluations[g].u.len() {
                sweep!(
                    format!("weight{m}.{g}.u[{i}]"),
                    r2.weight_combined[m].evaluations[g].u[i],
                    fp2
                );
            }
            for i in 0..r2.weight_combined[m].evaluations[g].mask.len() {
                sweep!(
                    format!("weight{m}.{g}.mask[{i}]"),
                    r2.weight_combined[m].evaluations[g].mask[i],
                    fp2
                );
            }
        }
        for i in 0..r2.weight_combined[m].proximity.u.len() {
            sweep!(
                format!("weight{m}.proximity.u[{i}]"),
                r2.weight_combined[m].proximity.u[i],
                fp2
            );
        }
        for i in 0..r2.weight_combined[m].proximity.mask.len() {
            sweep!(
                format!("weight{m}.proximity.mask[{i}]"),
                r2.weight_combined[m].proximity.mask[i],
                fp2
            );
        }
    }
    for r in 0..r2.scoring.requests.len() {
        sweep!(
            format!("uhat[{r}]"),
            r2.scoring.requests[r],
            |b: FixedBits| FixedBits(b.0 + 1)
        );
    }
    for c in 0..r3.advice_columns.len() {
        for s in 0..r3.advice_columns[c].len() {
            sweep!(
                format!("advice.column{c}[{s}]"),
                r3.advice_columns[c][s],
                fp
            );
        }
    }
    for s in 0..r3.advice_proof.siblings.len() {
        sweep!(
            format!("advice.sibling{s}"),
            r3.advice_proof.siblings[s],
            digest
        );
    }
    for m in 0..r3.weight_columns.len() {
        for c in 0..r3.weight_columns[m].columns.len() {
            for s in 0..r3.weight_columns[m].columns[c].len() {
                sweep!(
                    format!("weight{m}.column{c}[{s}]"),
                    r3.weight_columns[m].columns[c][s],
                    fp
                );
            }
        }
        for s in 0..r3.weight_columns[m].proof.siblings.len() {
            sweep!(
                format!("weight{m}.sibling{s}"),
                r3.weight_columns[m].proof.siblings[s],
                digest
            );
        }
    }
    assert!(
        accepted.is_empty(),
        "accepted after a change: {accepted:?}; rejections by reason: {codes:?}"
    );
}

// ----- Observation (05 §2, §5): the observed entry points decide exactly what these do --------

use crate::observe::{CheckEvent, CheckKind, CheckOutcome};

impl Fixture {
    /// [`run`](Self::run) through the observed entry points the CLI calls, each call's reports
    /// kept apart under its name. A live session's state is set as each phase leaves it, with
    /// the recorded seed and queries, since `start` and `on_round2` draw fresh ones.
    fn run_observed(
        &self,
        session: &str,
        edited: &BTreeMap<String, Vec<u8>>,
    ) -> (Verdict, Vec<(&'static str, Vec<CheckEvent>)>) {
        let bytes = |name: &str| edited.get(name).cloned().unwrap_or_else(|| base(name));
        let mut calls = Vec::new();
        let mut events = Vec::new();
        if session == "ni.tnp" {
            let verdict = verify_non_interactive_bytes_observed(
                &self.claims,
                &self.expected,
                &bytes("ni.tnp"),
                &self.config,
                &mut events,
            );
            calls.push(("ni", events));
            return (verdict, calls);
        }
        let limits = DecodeLimits::MESSAGE;
        let session = |state| {
            LiveVerifier::at(
                self.claims.clone(),
                self.expected.clone(),
                self.config.clone(),
                state,
            )
        };
        let round1 = open_live_observed::<Round1>(&bytes("live-round1.tce"), &limits, &mut events);
        calls.push(("open Round1", std::mem::take(&mut events)));
        let round1 = match round1 {
            Ok(m) => m,
            Err(verdict) => return (verdict, calls),
        };
        let share = protocol::ProverSeedShare {
            session_id: round1.session_id,
            share: [0; 32],
        };
        let mut live = session(live::Phase::Round1 {
            session_id: round1.session_id,
        });
        let revealed = live.on_round1_observed(&round1, &share, &mut events);
        calls.push(("Round1", std::mem::take(&mut events)));
        if let Err(verdict) = revealed {
            return (verdict, calls);
        }
        let first = round1.to_tce().unwrap();
        let round2 =
            open_live_observed::<Round2Response>(&bytes("live-round2.tce"), &limits, &mut events);
        calls.push(("open Round2Response", std::mem::take(&mut events)));
        let round2 = match round2 {
            Ok(m) => m,
            Err(verdict) => return (verdict, calls),
        };
        let mut live = session(live::Phase::Round2 {
            first: first.clone(),
            seed: self.seed,
        });
        let queried = live.on_round2_observed(&round2, &mut events);
        calls.push(("Round2Response", std::mem::take(&mut events)));
        if let Err(verdict) = queried {
            return (verdict, calls);
        }
        let round3 =
            open_live_observed::<Round3Response>(&bytes("live-round3.tce"), &limits, &mut events);
        calls.push(("open Round3Response", std::mem::take(&mut events)));
        let round3 = match round3 {
            Ok(m) => m,
            Err(verdict) => return (verdict, calls),
        };
        let mut live = session(live::Phase::Round3 {
            first,
            second: round2.to_tce().unwrap(),
            queries: self.queries.to_tce().unwrap(),
            seed: self.seed,
        });
        let verdict = live.on_round3_observed(&round3, &mut events);
        calls.push(("Round3Response", events));
        (verdict, calls)
    }
}

/// The outcome the last reported check carries for `verdict`.
fn outcome_of(verdict: &Verdict) -> CheckOutcome {
    match verdict {
        Verdict::Accept(_) => CheckOutcome::Pass,
        Verdict::Reject(_) => CheckOutcome::Reject,
        Verdict::Inconclusive(_) => CheckOutcome::Inconclusive,
    }
}

/// One call's reports: steps 1 to `k` exactly, in order, each the check of its step, step 5
/// `not_applicable`, every other passed but the last, which carries `last`. Returns `k`.
fn contiguous(what: &str, events: &[CheckEvent], last: CheckOutcome) -> u8 {
    assert!(!events.is_empty(), "{what}: no check reported");
    for (i, e) in events.iter().enumerate() {
        assert_eq!(e.kind, CheckKind::ALL[i], "{what}: {events:?}");
        assert_eq!(usize::from(e.kind.step()), i + 1, "{what}");
        let expected = if e.kind == CheckKind::WeightReservation {
            CheckOutcome::NotApplicable
        } else if i + 1 == events.len() {
            last
        } else {
            CheckOutcome::Pass
        };
        assert_eq!(e.outcome, expected, "{what}: {events:?}");
    }
    events.last().map_or(0, |e| e.kind.step())
}

/// Every call of a session but the last succeeded, so it reports the steps its phase completes;
/// the last carries the verdict.
fn check_calls(id: &str, verdict: &Verdict, calls: &[(&'static str, Vec<CheckEvent>)]) {
    for (i, (call, events)) in calls.iter().enumerate() {
        let what = format!("{id}, {call}");
        let last = i + 1 == calls.len();
        let outcome = if last {
            outcome_of(verdict)
        } else {
            CheckOutcome::Pass
        };
        let k = contiguous(&what, events, outcome);
        let complete = match *call {
            "ni" | "Round3Response" => 13,
            "Round1" => 6,
            "Round2Response" => 10,
            _ => 1,
        };
        if !last || matches!(verdict, Verdict::Accept(_)) {
            assert_eq!(k, complete, "{what}");
        } else {
            assert!(k <= complete, "{what}");
        }
        if call.starts_with("open") {
            assert_eq!(k, 1, "{what}");
        }
    }
}

#[cfg(unix)]
#[test]
fn observed_sessions_keep_every_verdict_and_end_at_the_deciding_check() {
    let fixture = Fixture::new();
    let cases = field(&fixture.corpus, "mutations").as_array().unwrap();
    let mut runs = vec![
        ("base NI".to_string(), "ni.tnp", BTreeMap::new()),
        ("base live".to_string(), "live-round2.tce", BTreeMap::new()),
    ];
    for case in cases {
        let edited = mutate(case);
        let session = if edited.contains_key("ni.tnp") {
            "ni.tnp"
        } else {
            "live-round2.tce"
        };
        runs.push((text(case, "id").to_string(), session, edited));
    }
    let mut steps: BTreeMap<u8, usize> = BTreeMap::new();
    for (id, session, edited) in &runs {
        let plain = fixture.run(session, edited).to_json().unwrap();
        let (verdict, calls) = fixture.run_observed(session, edited);
        assert_eq!(
            verdict.to_json().unwrap(),
            plain,
            "{id}: the verdict changed"
        );
        check_calls(id, &verdict, &calls);
        let last = calls.last().and_then(|(_, e)| e.last()).unwrap();
        *steps.entry(last.kind.step()).or_default() += 1;
    }
    // The corpus decides at many steps, not one: the check above is not vacuous.
    assert!(steps.len() >= 5, "{steps:?}");
}

#[test]
fn an_observed_acceptance_reports_the_checks_of_its_report() {
    let fixture = Fixture::new();
    let (verdict, calls) = fixture.run_observed("ni.tnp", &BTreeMap::new());
    check_calls("base NI", &verdict, &calls);
    let report = Json::parse(&verdict.to_json().unwrap()).unwrap();
    let listed = field(&report, "checks").as_array().unwrap();
    let (_, events) = &calls[0];
    assert_eq!(listed.len(), events.len());
    for (check, event) in listed.iter().zip(events) {
        assert_eq!(number(check, "step"), u64::from(event.kind.step()));
        assert_eq!(text(check, "check"), event.kind.name());
        assert_eq!(text(check, "outcome"), event.outcome.name());
        // The deterministic report keeps its null durations (05 §5).
        assert_eq!(field(check, "duration_ns"), &Json::Null);
    }
}
