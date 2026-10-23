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
