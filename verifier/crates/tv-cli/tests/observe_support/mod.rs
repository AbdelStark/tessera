// SPDX-License-Identifier: Apache-2.0 OR MIT

//! What the log and run-manifest tests share: the tamper corpus's base NI proof
//! (`spec/vectors/tamper/`), test-small, which a plain build refuses at step 2 and a build with
//! `tv-core/test-params` accepts; and a reader of `tessera.log-event/1` lines that checks each
//! against `spec/schemas/log-event-v1.schema.json`, written from the schema alone.

#![allow(dead_code)]

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use tv_codec::{DecodeLimits, Envelope, WeightCommitmentPublic};
use tv_hash::{Domain, hash};
use tv_testkit::Json;

/// The corpus's base session, copied into a directory of its own.
pub struct Fixture {
    pub dir: PathBuf,
    pub claims: PathBuf,
    pub weights: PathBuf,
    pub proof: PathBuf,
    pub root: String,
    pub statement: String,
}

impl Fixture {
    pub fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "tessera-cli-observe-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        let base = tv_testkit::vectors_dir().join("tamper");
        for name in ["claims.tcl", "weights.twr", "ni.tnp"] {
            fs::copy(base.join(name), dir.join(name)).unwrap();
        }
        let corpus = tv_testkit::read_vectors("tamper/protocol-1.json");
        let statement = corpus
            .get("base")
            .and_then(|b| b.get("statement_digest"))
            .and_then(Json::as_str)
            .unwrap()
            .to_string();
        let weights = fs::read(dir.join("weights.twr")).unwrap();
        let weights = Envelope::open(
            &weights,
            WeightCommitmentPublic::EXPECTED,
            &DecodeLimits::MESSAGE,
            |b| hash(Domain::Envelope, b),
        )
        .unwrap()
        .decode_payload(|c, _| WeightCommitmentPublic::decode(c))
        .unwrap();
        let root = hex(&weights.weights_root(|b| hash(Domain::WeightsRoot, b)));
        Self {
            claims: dir.join("claims.tcl"),
            weights: dir.join("weights.twr"),
            proof: dir.join("ni.tnp"),
            dir,
            root,
            statement,
        }
    }

    /// `verify` over the fixture, without `--json` or `--out`.
    pub fn args(&self) -> Vec<String> {
        vec![
            "verify".into(),
            "--claims".into(),
            self.claims.display().to_string(),
            "--weights".into(),
            self.weights.display().to_string(),
            "--proof".into(),
            self.proof.display().to_string(),
            "--weights-root".into(),
            self.root.clone(),
            "--statement-digest".into(),
            self.statement.clone(),
            "--bound".into(),
            "0".into(),
        ]
    }

    /// `args` with `extra` appended.
    pub fn with(&self, extra: &[&str]) -> Vec<String> {
        let mut args = self.args();
        args.extend(extra.iter().map(|s| s.to_string()));
        args
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

/// `payload` in an envelope of `schema` (RFC-0003 'Envelope'): the header, the payload and the
/// digest of both.
pub fn seal(schema: tv_codec::Expected, payload: &[u8]) -> Vec<u8> {
    let mut out = tv_codec::MAGIC.to_vec();
    out.extend_from_slice(&tv_codec::ENVELOPE_VERSION.to_le_bytes());
    out.extend_from_slice(&schema.schema_id.to_le_bytes());
    out.extend_from_slice(&schema.version.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(payload);
    let digest = hash(Domain::Envelope, &out);
    out.extend_from_slice(&digest);
    out
}

impl Fixture {
    /// The claim list resealed under `ni-100/1`, a production set every build accepts, so that
    /// tv-core runs its own checks in a plain build too. With `layout`, also ni-100's layout:
    /// the claim list then validates (step 3) and the statement, which names the original
    /// list, disagrees with it at step 4.
    pub fn as_ni_100(&self, layout: bool) {
        let bytes = fs::read(&self.claims).unwrap();
        let mut claims = tv_core::claims::ClaimList::open(&bytes, &DecodeLimits::MESSAGE).unwrap();
        claims.params = tv_core::claims::ParamSetId::new("ni-100/1").unwrap();
        if layout {
            claims.layout = tv_core::params::ParamSet::Ni100.layout();
        }
        fs::write(
            &self.claims,
            seal(tv_core::claims::SCHEMA, &claims.to_tce()),
        )
        .unwrap();
    }
}

/// Whether this build accepts test-small: a plain build refuses it as `REJ-PARAMS` at step 2.
pub fn accepts_test_small() -> bool {
    tv_core::params::ParamSet::accept(&tv_core::claims::ParamSetId::new("test-small/1").unwrap())
        .is_some()
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// An artifact's envelope digest: its last 32 bytes (RFC-0003 'Envelope').
pub fn envelope_digest(path: &std::path::Path) -> String {
    let bytes = fs::read(path).unwrap();
    hex(&bytes[bytes.len() - 32..])
}

/// A field's value: the schema admits integers, booleans and strings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Field {
    Int(u64),
    Text(String),
}

impl Field {
    pub fn text(&self) -> &str {
        match self {
            Self::Text(s) => s,
            Self::Int(_) => panic!("{self:?} is not text"),
        }
    }
    pub fn int(&self) -> u64 {
        match self {
            Self::Int(n) => *n,
            Self::Text(_) => panic!("{self:?} is not an integer"),
        }
    }
    fn shown(&self) -> String {
        match self {
            Self::Int(n) => n.to_string(),
            Self::Text(s) => s.clone(),
        }
    }
}

/// One span of a line's `span_path`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Span {
    pub name: String,
    pub fields: Vec<(String, Field)>,
}

/// One `tessera.log-event/1` line.
#[derive(Clone, Debug)]
pub struct Line {
    pub ts: String,
    pub level: String,
    pub target: String,
    pub event: String,
    pub spans: Vec<Span>,
    pub fields: Vec<(String, Field)>,
}

impl Line {
    pub fn field(&self, key: &str) -> Option<&Field> {
        self.fields.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    pub fn keys(&self) -> Vec<&str> {
        let mut keys: Vec<&str> = self.fields.iter().map(|(k, _)| k.as_str()).collect();
        keys.sort_unstable();
        keys
    }
}

fn members<'a>(value: &'a Json, what: &str) -> &'a [(String, Json)] {
    match value {
        Json::Object(m) => m,
        other => panic!("{what} is not an object: {other:?}"),
    }
}

fn fields(value: &Json, what: &str) -> Vec<(String, Field)> {
    members(value, what)
        .iter()
        .map(|(k, v)| {
            let field = match v {
                Json::Number(n) => {
                    assert!(
                        !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()),
                        "{what}.{k}: {n} is not a non-negative integer"
                    );
                    Field::Int(n.parse().unwrap())
                }
                Json::String(s) => Field::Text(s.clone()),
                other => panic!("{what}.{k}: {other:?} is no integer or string"),
            };
            (k.clone(), field)
        })
        .collect()
}

/// `YYYY-MM-DDTHH:MM:SS.ffffffZ`: UTC, RFC 3339, to the microsecond.
pub fn assert_timestamp(ts: &str) {
    assert_eq!(ts.len(), 27, "{ts}");
    for (i, b) in ts.bytes().enumerate() {
        let expected = match i {
            4 | 7 => b == b'-',
            10 => b == b'T',
            13 | 16 => b == b':',
            19 => b == b'.',
            26 => b == b'Z',
            _ => b.is_ascii_digit(),
        };
        assert!(expected, "{ts}: byte {i}");
    }
    let n = |r: std::ops::Range<usize>| ts[r].parse::<u32>().unwrap();
    assert!(
        (1..=12).contains(&n(5..7)) && (1..=31).contains(&n(8..10)),
        "{ts}"
    );
    assert!(n(11..13) < 24 && n(14..16) < 60 && n(17..19) < 61, "{ts}");
}

/// Parses `text` as one line of the schema, checking every rule it states.
pub fn parse_line(text: &str) -> Line {
    let value = Json::parse(text).unwrap_or_else(|e| panic!("{text}: {e}"));
    let top = members(&value, "line");
    let mut keys: Vec<&str> = top.iter().map(|(k, _)| k.as_str()).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        ["event", "fields", "level", "span_path", "target", "ts"],
        "{text}"
    );
    let ts = value.get("ts").and_then(Json::as_str).unwrap().to_string();
    assert_timestamp(&ts);
    let level = value
        .get("level")
        .and_then(Json::as_str)
        .unwrap()
        .to_string();
    assert!(
        ["error", "warn", "info", "debug", "trace"].contains(&level.as_str()),
        "{text}"
    );
    let target = value
        .get("target")
        .and_then(Json::as_str)
        .unwrap()
        .to_string();
    assert!(!target.is_empty());
    let event = value
        .get("event")
        .and_then(Json::as_str)
        .unwrap()
        .to_string();
    assert!(event.chars().count() >= 11, "{text}");
    let mut spans = Vec::new();
    for span in value.get("span_path").and_then(Json::as_array).unwrap() {
        let m = members(span, "span");
        let mut keys: Vec<&str> = m.iter().map(|(k, _)| k.as_str()).collect();
        keys.sort_unstable();
        assert_eq!(keys, ["fields", "name"], "{text}");
        let name = span.get("name").and_then(Json::as_str).unwrap().to_string();
        assert!(
            [
                "compile",
                "commit_weights",
                "prove",
                "witness_pass",
                "round1",
                "stage",
                "opening",
                "round3",
                "verify",
                "check"
            ]
            .contains(&name.as_str()),
            "{text}"
        );
        spans.push(Span {
            fields: fields(span.get("fields").unwrap(), "span fields"),
            name,
        });
    }
    Line {
        ts,
        level,
        target,
        event,
        spans,
        fields: fields(value.get("fields").unwrap(), "fields"),
    }
}

/// Every line of `stderr`, each checked against the schema.
pub fn lines(stderr: &str) -> Vec<Line> {
    assert!(stderr.is_empty() || stderr.ends_with('\n'), "{stderr:?}");
    stderr.lines().map(parse_line).collect()
}

/// The template with each `{name}` replaced by the field `name` (05 §2).
pub fn substituted(template: &str, line: &Line) -> String {
    let mut out = template.to_string();
    for (key, value) in &line.fields {
        out = out.replace(&format!("{{{key}}}"), &value.shown());
    }
    out
}

/// The three events of 05 §2 'The verifier's lines': level, field names, message template.
pub fn spec(event: &str) -> (&'static str, &'static [&'static str], &'static str) {
    match event {
        "tessera.tv_cli.command_failed" => (
            "error",
            &["code", "command", "detail", "message"],
            "error ({code}): {detail}",
        ),
        "tessera.tv_cli.check_completed" => (
            "debug",
            &["duration_ns", "message", "outcome", "step"],
            "step {step} {outcome} in {duration_ns} ns",
        ),
        "tessera.tv_cli.verified" => (
            "info",
            &["code", "duration_ns", "exit_code", "message", "status"],
            "verify: completed in {duration_ns} ns; protocol/1, anchor/ledger/ZK unchecked",
        ),
        other => panic!("unknown event {other}"),
    }
}

/// `line` is one of the verifier's events with its level, exactly its fields and its message.
pub fn assert_event(line: &Line) {
    let (level, keys, template) = spec(&line.event);
    assert_eq!(line.level, level, "{line:?}");
    assert_eq!(line.target, "tv_cli::log", "{line:?}");
    let mut expected: Vec<&str> = keys.to_vec();
    if line.event == "tessera.tv_cli.verified" && line.field("status").unwrap().text() == "accept" {
        // `code` is absent on accept.
        expected.retain(|k| *k != "code");
    }
    assert_eq!(line.keys(), expected, "{line:?}");
    assert_eq!(
        line.field("message").unwrap().text(),
        substituted(template, line),
        "{line:?}"
    );
}
