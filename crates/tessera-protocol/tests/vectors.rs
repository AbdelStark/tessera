// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `spec/vectors/protocol/protocol.json`, written by an independent oracle (`generate.py`):
//!
//! - every message decodes to the value the oracle states, which this test builds from the JSON
//!   field by field, encodes back byte for byte, and every proper prefix of it is
//!   `ERR-CODEC-TRUNCATED`; its envelope is the oracle's, and opens to the value;
//! - the coin toss: `c_V = H("coin-commit", s_V)` and `seed = H("coin-toss", s_V ‖ s_P)`;
//! - every refused input fails with the oracle's code at its offset, every input at a limit
//!   decodes, and every refused envelope fails as the oracle says.

mod common;

use std::collections::BTreeSet;
use std::fmt::Debug;
use std::fs;
use std::str::FromStr;

use serde_json::Value;
use tessera_codec::error::ErrorCode;
use tessera_codec::{Decode, DecodeError, DecodeLimits, Encode, Rational, decode, encode};
use tessera_field::{Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_protocol::{
    AdviceCommitment, BatchDescriptor, ColumnQueries, CombinedRow, Decoding, FixedBits, LogUpProof,
    MatrixColumns, MatrixQueries, Message, MessageKind, MultiProof, NiProof, PROTOCOL_VERSION,
    PredictorSpec, ProverSeedShare, PublicStatement, RequestRef, Round1, Round2Response,
    Round3Response, ScoringClaims, SeedCommit, SeedReveal, StageId, StageProof, SumcheckProof,
    TreeProof, WeightCombinedRows, WeightMatrix, coin_commitment, coin_toss,
};

fn load() -> Value {
    let path = common::vectors().join("protocol/protocol.json");
    serde_json::from_str(&fs::read_to_string(path).expect("vector file")).expect("JSON")
}

fn limits(value: &Value) -> DecodeLimits {
    let field = |key: &str| value[key].as_u64().expect("a limit");
    DecodeLimits {
        max_total_bytes: field("max_total_bytes"),
        max_vec_len: field("max_vec_len"),
        max_depth: u16::try_from(field("max_depth")).expect("a u16 depth"),
    }
}

fn text(value: &Value) -> &str {
    value.as_str().expect("a string")
}

fn hex(value: &Value) -> Vec<u8> {
    common::bytes(text(value))
}

fn list(value: &Value) -> &Vec<Value> {
    value.as_array().expect("a list")
}

/// The value the oracle states, built from its JSON (README.md 'Values').
mod json {
    use super::*;

    pub fn int<T: FromStr>(value: &Value) -> T
    where
        T::Err: Debug,
    {
        text(value).parse().expect("a decimal integer")
    }

    pub fn array<const N: usize>(value: &Value) -> [u8; N] {
        hex(value).try_into().expect("an array of its length")
    }

    pub fn fp(value: &Value) -> Goldilocks {
        Goldilocks::from_canonical_u64(int(value)).expect("a canonical element")
    }

    pub fn fp2(value: &Value) -> GoldilocksExt2 {
        let pair = list(value);
        assert_eq!(pair.len(), 2);
        GoldilocksExt2::new(fp(&pair[0]), fp(&pair[1]))
    }

    pub fn many<T>(value: &Value, item: impl Fn(&Value) -> T) -> Vec<T> {
        list(value).iter().map(item).collect()
    }

    /// A struct's fields, which must be `names` and no other. JSON objects are unordered: the
    /// order of the fields is the encoding's, which the payload pins.
    fn fields<'a>(value: &'a Value, names: &[&str]) -> &'a serde_json::Map<String, Value> {
        let object = value.as_object().expect("a struct");
        let keys: BTreeSet<&str> = object.keys().map(String::as_str).collect();
        assert_eq!(keys, names.iter().copied().collect(), "the fields");
        object
    }

    /// The variant and its fields of an enum value.
    fn variant(value: &Value) -> (&str, &Value) {
        let object = value.as_object().expect("an enum value");
        assert_eq!(object.len(), 1);
        let (name, fields) = object.iter().next().expect("a variant");
        (name.as_str(), fields)
    }

    pub fn seed_commit(v: &Value) -> SeedCommit {
        let f = fields(v, &["session_id", "commitment"]);
        SeedCommit {
            session_id: array(&f["session_id"]),
            commitment: array(&f["commitment"]),
        }
    }

    pub fn prover_seed_share(v: &Value) -> ProverSeedShare {
        let f = fields(v, &["session_id", "share"]);
        ProverSeedShare {
            session_id: array(&f["session_id"]),
            share: array(&f["share"]),
        }
    }

    pub fn seed_reveal(v: &Value) -> SeedReveal {
        let f = fields(v, &["session_id", "share"]);
        SeedReveal {
            session_id: array(&f["session_id"]),
            share: array(&f["share"]),
        }
    }

    fn decoding(v: &Value) -> Decoding {
        match variant(v) {
            ("Greedy", body) => {
                assert!(body.is_null());
                Decoding::Greedy
            }
            ("Temperature", body) => {
                let f = fields(body, &["t"]);
                let t = f["t"].as_object().expect("a rational");
                let (num, den) = (int::<i64>(&t["num"]), int::<u64>(&t["den"]));
                let t = Rational::new(num, den).expect("a fraction");
                assert_eq!((t.num(), t.den()), (num, den), "in lowest terms");
                Decoding::Temperature { t }
            }
            (other, _) => panic!("no Decoding variant {other}"),
        }
    }

    fn predictor(v: &Value) -> PredictorSpec {
        match variant(v) {
            ("Exact", body) => {
                let f = fields(body, &["decoding"]);
                PredictorSpec::Exact {
                    decoding: decoding(&f["decoding"]),
                }
            }
            ("Mixture", body) => {
                let f = fields(body, &["decoding", "eps_log2"]);
                PredictorSpec::Mixture {
                    decoding: decoding(&f["decoding"]),
                    eps_log2: int(&f["eps_log2"]),
                }
            }
            (other, _) => panic!("no PredictorSpec variant {other}"),
        }
    }

    fn request(v: &Value) -> RequestRef {
        let f = fields(v, &["request_id", "session_id", "prompt_len", "output_len"]);
        RequestRef {
            request_id: int(&f["request_id"]),
            session_id: array(&f["session_id"]),
            prompt_len: int(&f["prompt_len"]),
            output_len: int(&f["output_len"]),
        }
    }

    pub fn statement(v: &Value) -> PublicStatement {
        let f = fields(
            v,
            &[
                "claim_list_digest",
                "weights_root",
                "weights_epoch",
                "manifest_digest",
                "batch",
                "predictor",
            ],
        );
        let batch = fields(&f["batch"], &["requests"]);
        PublicStatement {
            claim_list_digest: array(&f["claim_list_digest"]),
            weights_root: array(&f["weights_root"]),
            weights_epoch: int(&f["weights_epoch"]),
            manifest_digest: array(&f["manifest_digest"]),
            batch: BatchDescriptor {
                requests: many(&batch["requests"], request),
            },
            predictor: predictor(&f["predictor"]),
        }
    }

    pub fn round1(v: &Value) -> Round1 {
        let f = fields(
            v,
            &["protocol_version", "session_id", "statement", "advice"],
        );
        let advice = fields(&f["advice"], &["root", "layout_digest"]);
        Round1 {
            protocol_version: int(&f["protocol_version"]),
            session_id: array(&f["session_id"]),
            statement: statement(&f["statement"]),
            advice: AdviceCommitment {
                root: array(&advice["root"]),
                layout_digest: array(&advice["layout_digest"]),
            },
        }
    }

    fn sumcheck(v: &Value) -> SumcheckProof<GoldilocksExt2> {
        let f = fields(v, &["rounds", "final_evals"]);
        SumcheckProof {
            rounds: many(&f["rounds"], |round| many(round, fp2)),
            final_evals: many(&f["final_evals"], fp2),
        }
    }

    fn combined_row(v: &Value) -> CombinedRow {
        let f = fields(v, &["u", "mask"]);
        CombinedRow {
            u: many(&f["u"], fp2),
            mask: many(&f["mask"], fp2),
        }
    }

    pub fn round2(v: &Value) -> Round2Response {
        let f = fields(
            v,
            &[
                "logup",
                "trees",
                "stages",
                "advice_combined",
                "weight_combined",
                "scoring",
            ],
        );
        let logup = fields(&f["logup"], &["roots", "values"]);
        let trees = fields(&f["trees"], &["levels", "splits"]);
        let scoring = fields(&f["scoring"], &["requests"]);
        Round2Response {
            logup: LogUpProof {
                roots: many(&logup["roots"], fp2),
                values: many(&logup["values"], fp2),
            },
            trees: TreeProof {
                levels: many(&trees["levels"], sumcheck),
                splits: many(&trees["splits"], fp2),
            },
            stages: many(&f["stages"], |stage| {
                let s = fields(stage, &["stage", "sumcheck", "splits"]);
                StageProof {
                    stage: StageId(int(&s["stage"])),
                    sumcheck: sumcheck(&s["sumcheck"]),
                    splits: many(&s["splits"], fp2),
                }
            }),
            advice_combined: many(&f["advice_combined"], combined_row),
            weight_combined: many(&f["weight_combined"], |entry| {
                let e = fields(entry, &["matrix", "evaluations", "proximity"]);
                WeightCombinedRows {
                    matrix: int(&e["matrix"]),
                    evaluations: many(&e["evaluations"], combined_row),
                    proximity: combined_row(&e["proximity"]),
                }
            }),
            scoring: ScoringClaims {
                requests: many(&scoring["requests"], |bits| FixedBits(int(bits))),
            },
        }
    }

    pub fn queries(v: &Value) -> ColumnQueries {
        let f = fields(v, &["advice", "weights"]);
        ColumnQueries {
            advice: many(&f["advice"], int),
            weights: many(&f["weights"], |entry| {
                let e = fields(entry, &["matrix", "indices"]);
                MatrixQueries {
                    matrix: int(&e["matrix"]),
                    indices: many(&e["indices"], int),
                }
            }),
        }
    }

    fn multiproof(v: &Value) -> MultiProof {
        let f = fields(v, &["siblings"]);
        MultiProof {
            siblings: many(&f["siblings"], array),
        }
    }

    pub fn round3(v: &Value) -> Round3Response {
        let f = fields(v, &["advice_columns", "advice_proof", "weight_columns"]);
        Round3Response {
            advice_columns: many(&f["advice_columns"], |column| many(column, fp)),
            advice_proof: multiproof(&f["advice_proof"]),
            weight_columns: many(&f["weight_columns"], |entry| {
                let e = fields(entry, &["matrix", "columns", "proof"]);
                MatrixColumns {
                    matrix: int(&e["matrix"]),
                    columns: many(&e["columns"], |column| many(column, fp)),
                    proof: multiproof(&e["proof"]),
                }
            }),
        }
    }

    pub fn ni_proof(v: &Value) -> NiProof {
        let f = fields(
            v,
            &["round1", "seed_nonce", "round2", "query_nonce", "round3"],
        );
        NiProof {
            round1: round1(&f["round1"]),
            seed_nonce: int(&f["seed_nonce"]),
            round2: round2(&f["round2"]),
            query_nonce: int(&f["query_nonce"]),
            round3: round3(&f["round3"]),
        }
    }
}

/// One message case of the file.
struct Case {
    name: String,
    payload: Vec<u8>,
    /// Runs `(length, offset)`: every prefix from `length` to the next run's length is
    /// `ERR-CODEC-TRUNCATED` at `offset`.
    truncated: Vec<(usize, u64)>,
    envelope: Vec<u8>,
    limits: DecodeLimits,
}

impl Case {
    fn new(case: &Value, limits: DecodeLimits) -> Self {
        let truncated: Vec<(usize, u64)> = list(&case["truncated"])
            .iter()
            .map(|run| {
                let run = list(run);
                assert_eq!(run.len(), 2, "a run is [length, offset]");
                let length = run[0].as_u64().expect("a length");
                let offset = run[1].as_u64().expect("an offset");
                (usize::try_from(length).expect("a length"), offset)
            })
            .collect();
        let name = text(&case["name"]).to_owned();
        assert_eq!(
            truncated.first().map(|run| run.0),
            Some(0),
            "{name}: runs start at 0"
        );
        assert!(
            truncated.windows(2).all(|pair| pair[0].0 < pair[1].0),
            "{name}: runs in increasing order"
        );
        Self {
            name,
            payload: hex(&case["payload"]),
            truncated,
            envelope: hex(&case["envelope"]),
            limits,
        }
    }

    /// The offset of the prefix of `cut` bytes' `ERR-CODEC-TRUNCATED`.
    fn truncated_at(&self, cut: usize) -> u64 {
        self.truncated
            .iter()
            .rev()
            .find(|run| run.0 <= cut)
            .map(|run| run.1)
            .expect("a run from length 0")
    }

    /// The message checks: the encoding, the decoding, every prefix's code and offset, and the
    /// envelope both ways.
    fn check<T: Message + PartialEq + Debug>(&self, value: &T) {
        let (name, payload, limits) = (&self.name, &self.payload, self.limits);
        assert_eq!(&encode(value), payload, "{name}: the encoding");
        assert_eq!(
            decode::<T>(payload, limits).as_ref(),
            Ok(value),
            "{name}: the decoding"
        );
        for cut in 0..payload.len() {
            assert_eq!(
                decode::<T>(&payload[..cut], limits).map(drop),
                Err(DecodeError {
                    code: ErrorCode::CodecTruncated,
                    offset: self.truncated_at(cut),
                }),
                "{name}: the prefix of {cut} bytes"
            );
        }
        assert_eq!(value.seal(), self.envelope, "{name}: the envelope");
        assert_eq!(
            T::open(&self.envelope, limits).as_ref(),
            Ok(value),
            "{name}: the opened envelope"
        );
    }
}

#[test]
fn every_message_decodes_to_its_value_and_encodes_back() {
    let file = load();
    let limits = limits(&file["limits"]);
    assert_eq!(limits, DecodeLimits::MESSAGE, "RFC-0003's message defaults");
    let cases = list(&file["messages"]);
    let mut kinds = BTreeSet::new();
    for case in cases {
        let ty = text(&case["type"]);
        let value = &case["value"];
        let checked = Case::new(case, limits);
        match ty {
            "SeedCommit" => checked.check(&json::seed_commit(value)),
            "ProverSeedShare" => checked.check(&json::prover_seed_share(value)),
            "SeedReveal" => checked.check(&json::seed_reveal(value)),
            "Round1" => {
                let round1 = json::round1(value);
                checked.check(&round1);
                // The claim list's protocol version, 0 until tessera-protocol/1 (09 §3).
                assert_eq!(
                    round1.protocol_version, PROTOCOL_VERSION,
                    "{}",
                    checked.name
                );
                // SessionContext's statement digest (RFC-0007 'Public statement').
                assert_eq!(
                    round1.statement.digest().to_vec(),
                    hex(&case["statement_digest"]),
                    "{}",
                    checked.name
                );
            }
            "Round2Response" => checked.check(&json::round2(value)),
            "ColumnQueries" => checked.check(&json::queries(value)),
            "Round3Response" => checked.check(&json::round3(value)),
            "NiProof" => {
                let proof = json::ni_proof(value);
                checked.check(&proof);
                assert_eq!(proof.round1.protocol_version, PROTOCOL_VERSION);
            }
            other => panic!("no message {other}"),
        }
        kinds.insert(ty.to_owned());
    }
    let every: BTreeSet<String> = MessageKind::ALL.iter().map(|k| k.to_string()).collect();
    assert_eq!(kinds, every, "a case for every message");
}

/// The file's schemas and matrices are the catalog's (03 §5) and RFC-0031's.
#[test]
fn the_schemas_and_matrices_are_the_specifications() {
    let file = load();
    let schemas = file["schemas"].as_object().expect("schemas");
    assert_eq!(schemas.len(), MessageKind::ALL.len());
    for kind in MessageKind::ALL {
        let entry = &schemas[&kind.to_string()];
        assert_eq!(
            entry["id"].as_u64(),
            Some(u64::from(kind.schema().id.0)),
            "{kind}"
        );
        assert_eq!(
            entry["version"].as_u64(),
            Some(u64::from(kind.schema().version))
        );
    }
    let names: Vec<&str> = list(&file["matrices"]).iter().map(text).collect();
    let ours: Vec<&str> = WeightMatrix::ALL.iter().map(|m| m.name()).collect();
    assert_eq!(names, ours);
    for (id, matrix) in WeightMatrix::ALL.into_iter().enumerate() {
        assert_eq!(usize::from(matrix.id()), id);
        assert_eq!(WeightMatrix::from_id(matrix.id()), Some(matrix));
    }
    assert_eq!(WeightMatrix::from_id(10), None);
}

#[test]
fn the_coin_toss_is_the_oracles() {
    let file = load();
    let cases = list(&file["coin_toss"]);
    assert!(cases.len() >= 3);
    for case in cases {
        let s_v: [u8; 32] = json::array(&case["verifier_share"]);
        let s_p: [u8; 32] = json::array(&case["prover_share"]);
        assert_eq!(coin_commitment(&s_v).to_vec(), hex(&case["commitment"]));
        assert_eq!(coin_toss(&s_v, &s_p).to_vec(), hex(&case["seed"]));
    }
    // The messages' shares are the first case's: the commitment opens with the reveal.
    let messages = list(&file["messages"]);
    let value = |ty: &str| {
        &messages
            .iter()
            .find(|m| text(&m["type"]) == ty)
            .expect("a message")["value"]
    };
    let commit = json::seed_commit(value("SeedCommit"));
    let reveal = json::seed_reveal(value("SeedReveal"));
    let share = json::prover_seed_share(value("ProverSeedShare"));
    assert_eq!(coin_commitment(&reveal.share), commit.commitment);
    assert_eq!(reveal.share.to_vec(), hex(&cases[0]["verifier_share"]));
    assert_eq!(share.share.to_vec(), hex(&cases[0]["prover_share"]));
}

/// Decodes `bytes` as the message `ty`, and re-encodes what it accepts.
fn decode_as(ty: &str, bytes: &[u8], limits: DecodeLimits) -> Result<Vec<u8>, DecodeError> {
    fn run<T: Encode + Decode>(bytes: &[u8], limits: DecodeLimits) -> Result<Vec<u8>, DecodeError> {
        decode::<T>(bytes, limits).map(|value| encode(&value))
    }
    match ty {
        "SeedCommit" => run::<SeedCommit>(bytes, limits),
        "ProverSeedShare" => run::<ProverSeedShare>(bytes, limits),
        "SeedReveal" => run::<SeedReveal>(bytes, limits),
        "Round1" => run::<Round1>(bytes, limits),
        "Round2Response" => run::<Round2Response>(bytes, limits),
        "ColumnQueries" => run::<ColumnQueries>(bytes, limits),
        "Round3Response" => run::<Round3Response>(bytes, limits),
        "NiProof" => run::<NiProof>(bytes, limits),
        other => panic!("no message {other}"),
    }
}

fn case_limits(file: &Value, case: &Value) -> DecodeLimits {
    match case.get("limits") {
        Some(own) => limits(own),
        None => limits(&file["limits"]),
    }
}

#[test]
fn every_refused_input_fails_with_its_code_at_its_offset() {
    let file = load();
    let cases = list(&file["refused"]);
    let mut codes = BTreeSet::new();
    let mut kinds = BTreeSet::new();
    for case in cases {
        let ty = text(&case["type"]);
        let code = ErrorCode::from_identifier(text(&case["code"])).expect("a code");
        let offset = case["offset"].as_u64().expect("an offset");
        let outcome = decode_as(ty, &hex(&case["hex"]), case_limits(&file, case));
        assert_eq!(
            outcome.map(drop),
            Err(DecodeError { code, offset }),
            "{ty}: {}",
            text(&case["why"])
        );
        codes.insert(code.as_str());
        kinds.insert(ty);
    }
    assert_eq!(
        kinds.len(),
        MessageKind::ALL.len(),
        "every message has refused inputs"
    );
    for code in [
        "ERR-CODEC-TRUNCATED",
        "ERR-CODEC-NONCANONICAL",
        "ERR-CODEC-LIMIT",
    ] {
        assert!(codes.contains(code), "{code}");
    }
}

#[test]
fn every_input_at_its_limits_decodes_and_encodes_back() {
    let file = load();
    let cases = list(&file["accepted"]);
    assert!(cases.len() >= MessageKind::ALL.len());
    for case in cases {
        let bytes = hex(&case["hex"]);
        let outcome = decode_as(text(&case["type"]), &bytes, case_limits(&file, case));
        assert_eq!(outcome, Ok(bytes), "{}", text(&case["why"]));
    }
}

#[test]
fn every_refused_envelope_fails_with_its_code_at_its_offset() {
    let file = load();
    let limits = limits(&file["limits"]);
    let cases = list(&file["envelopes"]);
    assert!(!cases.is_empty());
    for case in cases {
        let bytes = hex(&case["hex"]);
        let outcome = match text(&case["schema"]) {
            "SeedCommit" => SeedCommit::open(&bytes, limits).map(drop),
            "Round1" => Round1::open(&bytes, limits).map(drop),
            "Round2Response" => Round2Response::open(&bytes, limits).map(drop),
            "ColumnQueries" => ColumnQueries::open(&bytes, limits).map(drop),
            other => panic!("no envelope case for {other}"),
        };
        let code = ErrorCode::from_identifier(text(&case["code"])).expect("a code");
        let offset = case["offset"].as_u64().expect("an offset");
        assert_eq!(
            outcome,
            Err(DecodeError { code, offset }),
            "{}",
            text(&case["why"])
        );
    }
}
