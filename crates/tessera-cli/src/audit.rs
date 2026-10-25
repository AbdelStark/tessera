// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tessera audit share`, the prover's side of the coin toss over a manifest ([`share`]), and
//! `tessera audit sample`, a period's audit sample from its verified log, the previous period's,
//! the policy and the coin toss (RFC-0016 'Sample derivation', RFC-0063). `audit sample` runs in
//! five steps:
//!
//! 1. **The logs.** Each `.tlm` is opened as a `LogManifest` envelope and each `.tdl` decoded as
//!    a period's log, then checked against its manifest (RFC-0062 'The check'): a log that is not
//!    its manifest's is `ERR-CODEC-DIGEST`.
//! 2. **The chain.** `--previous-manifest`, the `--between` manifests in order and `--manifest`
//!    must link (RFC-0060 'The chain'). The period and the previous one have non-zero length,
//!    and every `--between` period zero length and no entries (RFC-0063 'The previous period's
//!    log'). Without a previous period, the chain must start at the log's genesis: no previous
//!    manifest, `first_seq` 0. Anything else is `ERR-USAGE`: the previous log given is not the
//!    period's, or the period has no candidates.
//! 3. **The record.** `--state` holds the answer `tessera audit share` kept for the manifest: the
//!    policy, the verifier's commitment and the prover's share. The sample is drawn from that
//!    record and `--verifier-share` alone, so no other toss's shares, and no other policy, can be
//!    presented for the manifest. A manifest with no record is `ERR-USAGE`.
//! 4. **The reveal.** A `--verifier-share` that does not open the recorded commitment is
//!    `ERR-PROTO-COINTOSS`.
//! 5. **The sample**, drawn among the period's candidates (`tessera_anchor::candidates`), is
//!    written to `--out` as an `AuditSample` envelope, through a temporary file renamed into
//!    place.
//!
//! `tessera audit batch` packs a sample's selected requests into proof batches, read from the
//! same logs ([`batch`], RFC-0016 'Batching'). [`sampled`] gives the batches, and `tessera prove
//! --sample` calls it too, so the batch a proof's `--batch` names is the one the report lists.
//!
//! `tessera audit plan` reports a policy's detection probability and amortized overhead at a
//! traffic profile, a `tessera.audit-traffic/1` file, with the cost model's `μ` at each bucket's
//! mean length ([`plan`], RFC-0016 'Calculators').
//!
//! No sampling or batching command checks a recorder signature: the verifier does, before it
//! derives the same sample (RFC-0062 'The check'). Everything they read and write is public
//! (03 §5).

use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value};
use tessera_anchor::recorder::decode_log;
use tessera_anchor::{LogManifest, RequestLengths, VerifiedLog, candidates, lengths};
use tessera_bench::{Counts, MemoryModel, ModelReport, Scenario, UnitCosts};
use tessera_codec::error::{ErrorCode, TesseraError};
use tessera_codec::{Ascii, DecodeLimits, Public, decode, encode, envelope, schema};
use tessera_hash::ParamSetId;
use tessera_protocol::{
    AuditPlan, AuditPolicy, AuditSample, BucketTraffic, PlanError, RateBucket, RequestKey,
    ShareRecord, batches, check_traffic,
};

use crate::{Input, io_error, write_atomically, write_bytes_atomically};

/// The policy file's identifier (RFC-0016 'Policy', `spec/schemas/audit-policy-v1.schema.json`).
pub(crate) const POLICY_SCHEMA: &str = "tessera.audit-policy/1";

/// The log flags of `tessera audit sample`, `audit batch` and `prove --sample`: the period's
/// manifest and log, and the chain before it.
pub(crate) struct Logs<'a> {
    pub(crate) manifest: &'a Path,
    pub(crate) log: &'a Path,
    pub(crate) previous_manifest: Option<&'a Path>,
    pub(crate) previous_log: Option<&'a Path>,
    pub(crate) between: &'a [PathBuf],
}

impl Logs<'_> {
    /// Every file the flags name.
    pub(crate) fn paths(&self) -> Vec<&Path> {
        [Some(self.manifest), Some(self.log)]
            .into_iter()
            .chain([self.previous_manifest, self.previous_log])
            .flatten()
            .chain(self.between.iter().map(PathBuf::as_path))
            .collect()
    }
}

/// A period's verified log and the previous period's, bound by the chain (RFC-0063 'The
/// previous period's log'): the window a sample is drawn from and its batches are read in.
pub(crate) struct Window {
    pub(crate) current: VerifiedLog,
    pub(crate) previous: Option<VerifiedLog>,
}

/// The flags of `tessera audit sample`.
pub(crate) struct SampleOptions<'a> {
    pub(crate) logs: Logs<'a>,
    pub(crate) state: &'a Path,
    pub(crate) verifier_share: [u8; 32],
    pub(crate) out: &'a Path,
}

fn usage(message: String) -> TesseraError {
    TesseraError::new(ErrorCode::Usage).with_source(message)
}

/// A manifest, from its envelope.
fn manifest(path: &Path) -> Result<LogManifest, TesseraError> {
    Input::new(path, &schema::LOG_MANIFEST).open()
}

/// A period's log, checked against its manifest (RFC-0062 'The check').
fn verified(manifest: &LogManifest, path: &Path) -> Result<VerifiedLog, TesseraError> {
    let shown = path.display().to_string();
    let bytes = fs::read(path).map_err(io_error(format!("cannot read {shown}")))?;
    let entries = decode_log(&bytes).map_err(|e| {
        TesseraError::new(e.code()).with_source(format!("{shown} is not a period's log: {e}"))
    })?;
    VerifiedLog::check(manifest, entries).map_err(|e| {
        TesseraError::new(e.code()).with_source(format!("{shown} is not its manifest's log: {e}"))
    })
}

/// Checks that `previous`, then `between`, then `current` follow one another in one log, as
/// RFC-0063 'The previous period's log' binds a period to the one before it.
fn check_previous(
    previous: Option<&LogManifest>,
    between: &[LogManifest],
    current: &LogManifest,
) -> Result<(), TesseraError> {
    if current.body.period_start_ns == current.body.period_end_ns {
        return Err(usage(
            "--manifest is a period of zero length, which has no candidates: sample the next one"
                .to_owned(),
        ));
    }
    for (i, m) in between.iter().enumerate() {
        // RFC-0060: every entry lies in its period, so a period of zero length holds none.
        if m.body.period_start_ns != m.body.period_end_ns || m.body.entry_count != 0 {
            return Err(usage(format!(
                "--between manifest {} is not an empty period of zero length",
                i + 1
            )));
        }
    }
    let chain: Vec<&LogManifest> = previous
        .into_iter()
        .chain(between)
        .chain(core::iter::once(current))
        .collect();
    match previous {
        Some(p) if p.body.period_start_ns == p.body.period_end_ns => {
            return Err(usage(
                "--previous-manifest is a period of zero length, which has no log to draw from"
                    .to_owned(),
            ));
        }
        Some(_) => {}
        None if chain[0].body.prev_manifest != [0; 32] || chain[0].body.first_seq != 0 => {
            return Err(usage(
                "the period has a period before it: give its --previous-manifest and --previous-log"
                    .to_owned(),
            ));
        }
        None => {}
    }
    for pair in chain.windows(2) {
        tessera_anchor::check_link(pair[0], pair[1]).map_err(|e| {
            usage(format!(
                "the manifests given do not follow one another in one log: {e}"
            ))
        })?;
    }
    Ok(())
}

/// An unsigned integer field of the policy file, at most `max`.
fn integer(object: &Map<String, Value>, key: &str, max: u64) -> Result<u64, TesseraError> {
    object
        .get(key)
        .and_then(Value::as_u64)
        .filter(|n| *n <= max)
        .ok_or_else(|| {
            usage(format!(
                "the policy's {key} is not an integer in [0, {max}]"
            ))
        })
}

/// Reads a `tessera.audit-policy/1` file strictly: no key missing or unknown, each rate a
/// decimal (RFC-0016 'Policy'), and the bounds the TCE decoder enforces.
pub(crate) fn read_policy(path: &Path) -> Result<AuditPolicy, TesseraError> {
    let shown = path.display().to_string();
    let text = fs::read_to_string(path).map_err(io_error(format!("cannot read {shown}")))?;
    let value: Value =
        serde_json::from_str(&text).map_err(|e| usage(format!("{shown} is not JSON: {e}")))?;
    let object = value
        .as_object()
        .ok_or_else(|| usage(format!("{shown} is not a JSON object")))?;
    const KEYS: [&str; 7] = [
        "schema",
        "period_ns",
        "buckets",
        "min_batch_tokens",
        "max_batch_tokens",
        "proof_deadline_ns",
        "params",
    ];
    if let Some(key) = object.keys().find(|k| !KEYS.contains(&k.as_str())) {
        return Err(usage(format!("{shown} has an unknown key {key}")));
    }
    if object.get("schema").and_then(Value::as_str) != Some(POLICY_SCHEMA) {
        return Err(usage(format!("{shown} is not a {POLICY_SCHEMA} file")));
    }
    let buckets = object
        .get("buckets")
        .and_then(Value::as_array)
        .ok_or_else(|| usage("the policy's buckets is not an array".to_owned()))?
        .iter()
        .map(|b| {
            let b = b
                .as_object()
                .filter(|b| b.len() == 2)
                .ok_or_else(|| usage("a bucket is not {max_tokens, rate}".to_owned()))?;
            let max_tokens = integer(b, "max_tokens", u64::from(u32::MAX))? as u32;
            let rate = b
                .get("rate")
                .and_then(Value::as_str)
                .ok_or_else(|| usage("a bucket's rate is not a string".to_owned()))?;
            let threshold = RateBucket::threshold_of(rate)
                .map_err(|e| usage(format!("the rate {rate:?}: {e}")))?;
            Ok(RateBucket {
                max_tokens,
                threshold,
            })
        })
        .collect::<Result<Vec<_>, TesseraError>>()?;
    let params = object
        .get("params")
        .and_then(Value::as_str)
        .filter(|p| !p.is_empty())
        .and_then(|p| Ascii::new(p).ok())
        .ok_or_else(|| usage("the policy's params is not printable ASCII".to_owned()))?;
    AuditPolicy::new(
        integer(object, "period_ns", u64::MAX)?,
        buckets,
        integer(object, "min_batch_tokens", u64::from(u32::MAX))? as u32,
        integer(object, "max_batch_tokens", u64::from(u32::MAX))? as u32,
        integer(object, "proof_deadline_ns", u64::MAX)?,
        ParamSetId(params),
    )
    .map_err(|e| TesseraError::new(e.code()).with_source(format!("{shown}: {e}")))
}

/// Reads the window the log flags give, as `audit sample`'s steps 1 and 2 check it.
pub(crate) fn window(logs: &Logs<'_>) -> Result<Window, TesseraError> {
    let current = manifest(logs.manifest)?;
    let previous = logs.previous_manifest.map(manifest).transpose()?;
    let between = logs
        .between
        .iter()
        .map(|p| manifest(p))
        .collect::<Result<Vec<_>, _>>()?;
    check_previous(previous.as_ref(), &between, &current)?;
    let previous = match (&previous, logs.previous_log) {
        (Some(m), Some(path)) => Some(verified(m, path)?),
        (None, None) => None,
        _ => {
            return Err(usage(
                "--previous-manifest and --previous-log go together".to_owned(),
            ));
        }
    };
    let current = verified(&current, logs.log)?;
    Ok(Window { current, previous })
}

/// Runs `tessera audit sample`.
pub(crate) fn sample(options: &SampleOptions<'_>) -> Result<(), TesseraError> {
    let Window {
        current: log,
        previous: previous_log,
    } = window(&options.logs)?;
    let current = log.manifest();
    let digest = current.identity();
    let path = record_path(options.state, &digest);
    let record = kept(&path)?
        .filter(|r| r.manifest_digest == digest)
        .ok_or_else(|| {
            usage(format!(
                "{} holds no answer for this manifest: run tessera audit share first",
                options.state.display()
            ))
        })?;
    let found = candidates(previous_log.as_ref(), &log);
    let sample = record
        .draw(
            options.verifier_share,
            found.iter().map(|c| {
                (
                    RequestKey {
                        session_id: c.session_id,
                        request_id: c.request_id,
                    },
                    c.tokens,
                )
            }),
        )
        .map_err(|e| {
            TesseraError::new(e.code()).with_source(
                "the verifier's share does not open the commitment the prover answered",
            )
        })?;
    write_bytes_atomically(options.out, &envelope::seal(&schema::AUDIT_SAMPLE, &sample))?;
    tessera_codec::event!(
        INFO,
        "cli.audit_sampled",
        candidates = found.len(),
        selected = sample.selected().len(),
        sample = Public(options.out.display());
        "tessera audit sample: selected {selected} of {candidates} candidates into {sample}"
    );
    Ok(())
}

/// The report of `tessera audit batch` (RFC-0016 'Batching',
/// `spec/schemas/audit-batches-v1.schema.json`).
pub(crate) const BATCHES_SCHEMA: &str = "tessera.audit-batches/1";

/// The file `tessera audit batch` writes its report to, in `--out-dir`.
pub(crate) const BATCHES_FILE: &str = "batches.json";

/// A request of a batch: its key, its length `L` (RFC-0063), and the lengths its statement
/// carries, or `None` when its frames give none (`tessera_anchor::lengths`).
pub(crate) struct Batched {
    pub(crate) key: RequestKey,
    pub(crate) tokens: u64,
    pub(crate) lengths: Option<RequestLengths>,
}

/// One batch: its requests in `RequestKey` order, and their total `L`.
pub(crate) struct SampledBatch {
    pub(crate) tokens: u64,
    pub(crate) requests: Vec<Batched>,
}

/// A sample's batches, with the sample and the window they were read in.
pub(crate) struct Sampled {
    pub(crate) sample: AuditSample,
    pub(crate) window: Window,
    pub(crate) batches: Vec<SampledBatch>,
}

fn internal(what: &str) -> TesseraError {
    TesseraError::new(ErrorCode::Internal).with_source(what.to_owned())
}

/// The batches of the sample at `path`, from its period's logs (RFC-0016 'Batching'): what
/// `tessera audit batch` reports and `tessera prove --sample` proves, which both call this.
///
/// 1. **The window**, read and checked as `tessera audit sample` reads it ([`window`]).
/// 2. **The sample** must be the window's. Its `manifest_digest` is `--manifest`'s identity,
///    and its selection is the one its policy and coin toss give over the window's candidates,
///    redrawn here: a sample of another period, or one that selects a request the draw does not
///    or leaves one out, is `ERR-USAGE`. A coin toss that does not open is
///    `ERR-PROTO-COINTOSS`.
/// 3. **The lengths.** A selected request's `L` is its candidate's length, and its statement's
///    `prompt_len` and `output_len` are its `In` and `Out` frames' tokens in the window
///    (`tessera_anchor::lengths`).
/// 4. **The batches**: first-fit decreasing on `L`, in one group (`tessera_protocol::batches`).
///    Greedy only: a temperature request's group is the pinned sampler key that applies to it
///    (RFC-0061 'Which pin applies'), and no crate implements the pins yet.
pub(crate) fn sampled(path: &Path, logs: &Logs<'_>) -> Result<Sampled, TesseraError> {
    let sample: AuditSample = Input::new(path, &schema::AUDIT_SAMPLE).open()?;
    let window = window(logs)?;
    let shown = path.display();
    if *sample.manifest_digest() != window.current.manifest().identity() {
        return Err(usage(format!(
            "{shown} is not a sample of --manifest's period"
        )));
    }
    let found = candidates(window.previous.as_ref(), &window.current);
    let key = |c: &tessera_anchor::Candidate| RequestKey {
        session_id: c.session_id,
        request_id: c.request_id,
    };
    let redrawn = AuditSample::draw(
        *sample.manifest_digest(),
        sample.policy().clone(),
        sample.coin_toss().clone(),
        found.iter().map(|c| (key(c), c.tokens)),
    )
    .map_err(|e| {
        TesseraError::new(e.code()).with_source(format!(
            "{shown}: the verifier's share does not open its commitment"
        ))
    })?;
    if redrawn != sample {
        return Err(usage(format!(
            "{shown} does not select what its period's logs and coin toss select"
        )));
    }
    let tokens: BTreeMap<RequestKey, u64> = found.iter().map(|c| (key(c), c.tokens)).collect();
    let keys: Vec<([u8; 16], u64)> = sample
        .selected()
        .iter()
        .map(|k| (k.session_id, k.request_id))
        .collect();
    let lengths: BTreeMap<RequestKey, Option<RequestLengths>> = sample
        .selected()
        .iter()
        .copied()
        .zip(lengths(window.previous.as_ref(), &window.current, &keys))
        .collect();
    let requests = sample
        .selected()
        .iter()
        .map(|k| {
            let l = tokens
                .get(k)
                .ok_or_else(|| internal("a selected request is not a candidate"))?;
            Ok((*k, (), *l))
        })
        .collect::<Result<Vec<_>, TesseraError>>()?;
    let batches = batches(sample.policy(), &requests)
        .map_err(|e| internal(&e.to_string()))?
        .into_iter()
        .map(|b| SampledBatch {
            tokens: b.tokens,
            requests: b
                .requests
                .iter()
                .map(|k| Batched {
                    key: *k,
                    tokens: tokens.get(k).copied().unwrap_or(u64::MAX),
                    lengths: lengths.get(k).copied().flatten(),
                })
                .collect(),
        })
        .collect();
    Ok(Sampled {
        sample,
        window,
        batches,
    })
}

/// The flags of `tessera audit batch`.
pub(crate) struct BatchOptions<'a> {
    pub(crate) sample: &'a Path,
    pub(crate) logs: Logs<'a>,
    pub(crate) out_dir: &'a Path,
}

/// The batches as `tessera.audit-batches/1`.
fn batches_json(sampled: &Sampled) -> Value {
    let sample = &sampled.sample;
    let batches: Vec<Value> = (0u64..)
        .zip(&sampled.batches)
        .map(|(index, b)| {
            let requests: Vec<Value> = b
                .requests
                .iter()
                .map(|r| {
                    let mut request = serde_json::json!({
                        "session_id": hex(&r.key.session_id),
                        "request_id": r.key.request_id,
                        "tokens": r.tokens,
                    });
                    if let (Some(l), Some(object)) = (r.lengths, request.as_object_mut()) {
                        object.insert("prompt_len".to_owned(), l.prompt_len.into());
                        object.insert("output_len".to_owned(), l.output_len.into());
                    }
                    request
                })
                .collect();
            serde_json::json!({
                "index": index,
                "group": "greedy",
                "tokens": b.tokens,
                "requests": requests,
            })
        })
        .collect();
    serde_json::json!({
        "schema": BATCHES_SCHEMA,
        "manifest_digest": hex(sample.manifest_digest()),
        "sample": hex(&sample.seed()),
        "min_batch_tokens": sample.policy().min_batch_tokens(),
        "max_batch_tokens": sample.policy().max_batch_tokens(),
        "selected": sample.selected().len(),
        "batches": batches,
    })
}

/// Runs `tessera audit batch`: the sample's batches ([`sampled`]), written to
/// `<out-dir>/batches.json` as `tessera.audit-batches/1` through a temporary file renamed into
/// place. With `json` the report is printed too; otherwise one line per batch. The report is
/// informational: `tessera prove --sample` reads the batches from the sample and the logs again,
/// never from it.
pub(crate) fn batch(options: &BatchOptions<'_>, json: bool) -> Result<(), TesseraError> {
    let sampled = sampled(options.sample, &options.logs)?;
    let text = serde_json::to_string_pretty(&batches_json(&sampled))
        .map(|text| format!("{text}\n"))
        .map_err(|e| internal(&e.to_string()))?;
    let dir = options.out_dir;
    fs::create_dir_all(dir).map_err(io_error(format!("cannot create {}", dir.display())))?;
    let path = dir.join(BATCHES_FILE);
    write_atomically(&path, &text)?;
    tessera_codec::event!(
        INFO,
        "cli.audit_batched",
        selected = sampled.sample.selected().len(),
        batches = sampled.batches.len(),
        report = Public(path.display());
        "tessera audit batch: {selected} selected requests in {batches} batches, reported in {report}"
    );
    let shown = match json {
        true => text,
        false => (0..)
            .zip(&sampled.batches)
            .map(|(index, b): (u64, _)| {
                format!(
                    "batch {index}: {} requests, {} tokens\n",
                    b.requests.len(),
                    b.tokens
                )
            })
            .collect(),
    };
    io::stdout()
        .lock()
        .write_all(shown.as_bytes())
        .map_err(io_error("cannot write to stdout".to_owned()))
}

/// The flags of `tessera audit share`.
pub(crate) struct ShareOptions<'a> {
    pub(crate) manifest: &'a Path,
    pub(crate) policy: &'a Path,
    pub(crate) commitment: [u8; 32],
    pub(crate) state: &'a Path,
}

/// Where the prover keeps its answer for a manifest: `<state>/<manifest digest>.share`.
fn record_path(state: &Path, manifest_digest: &[u8; 32]) -> PathBuf {
    state.join(format!("{}.share", hex(manifest_digest)))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The record kept at `path`, if there is one.
fn kept(path: &Path) -> Result<Option<ShareRecord>, TesseraError> {
    let shown = path.display().to_string();
    match fs::read(path) {
        Ok(bytes) => decode::<ShareRecord>(&bytes, DecodeLimits::MESSAGE)
            .map(Some)
            .map_err(|e| {
                TesseraError::from(e).with_source(format!("{shown} is not a share record"))
            }),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(io_error(format!("cannot read {shown}"))(e)),
    }
}

/// Runs `tessera audit share`: the prover's answer to a verifier's commitment over a manifest
/// (RFC-0016 'Sample derivation', step 2), one commitment and one policy per manifest.
///
/// - **The record.** The answer is kept in `<state>/<manifest digest>.share`, `TCE(ShareRecord)`,
///   before the share is printed: the manifest, the policy (`--policy`, a
///   `tessera.audit-policy/1` file), the commitment and the share. Every responder for one prover
///   must share the directory.
/// - **One answer.** A first commitment gets a fresh share. The same commitment and policy again
///   get the same share, for a verifier that lost it. Any other commitment or policy is refused
///   with `ERR-PROTO-ORDER`.
/// - **One path.** Every call writes a fresh record to a temporary file of its own
///   (`create_new`), links it into place, which fails if a record is there, and syncs the
///   directory. It then reads the record kept and answers from it, so a repeated call and a
///   responder that lost a race take the same path.
///
/// It prints the share as 64 hexadecimal digits, or with `json` a JSON object. The share is
/// public once sent.
pub(crate) fn share(options: &ShareOptions<'_>, json: bool) -> Result<(), TesseraError> {
    let manifest_digest = manifest(options.manifest)?.identity();
    let policy = read_policy(options.policy)?;
    let state = options.state;
    fs::create_dir_all(state).map_err(io_error(format!("cannot create {}", state.display())))?;
    // A directory just made outlives a crash only once its parent is synced.
    if let Some(parent) = state.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::File::open(parent)
            .and_then(|dir| dir.sync_all())
            .map_err(io_error(format!("cannot sync {}", parent.display())))?;
    }
    let path = record_path(state, &manifest_digest);
    let shown = path.display().to_string();
    let protocol =
        |e: tessera_protocol::ProtocolError| TesseraError::new(e.code()).with_source(e.to_string());
    // A fresh record, under a temporary name no other responder uses: `create_new` refuses a
    // file that is there, and the name carries the process, the time and part of the share.
    let fresh = ShareRecord::answer(None, manifest_digest, &policy, options.commitment)
        .map_err(protocol)?;
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(format!(
        ".tmp-{}-{nanos}-{}",
        std::process::id(),
        hex(&fresh.prover_share[..8])
    ));
    let tmp = PathBuf::from(tmp);
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)
        .map_err(io_error(format!("cannot write {shown}")))?;
    file.write_all(&encode(&fresh))
        .and_then(|()| file.sync_all())
        .map_err(io_error(format!("cannot write {shown}")))?;
    drop(file);
    // Linked into place, which fails if a record is there; then the directory is synced, so the
    // link outlives a crash before the share is printed.
    let linked = fs::hard_link(&tmp, &path);
    let _ = fs::remove_file(&tmp);
    match linked {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(io_error(format!("cannot write {shown}"))(e)),
    }
    fs::File::open(state)
        .and_then(|dir| dir.sync_all())
        .map_err(io_error(format!("cannot sync {}", state.display())))?;
    // The answer comes from the record as kept, whichever responder kept it.
    let kept = kept(&path)?.ok_or_else(|| {
        TesseraError::new(ErrorCode::ResIo).with_source(format!("{shown} vanished"))
    })?;
    let record = ShareRecord::answer(Some(&kept), manifest_digest, &policy, options.commitment)
        .map_err(protocol)?;
    let share = hex(&record.prover_share);
    let shown = match json {
        true => {
            let object = serde_json::json!({
                "command": "audit share",
                "manifest_digest": hex(&manifest_digest),
                "verifier_commitment": hex(&record.verifier_commitment),
                "prover_share": share,
            });
            format!("{object}\n")
        }
        false => format!("{share}\n"),
    };
    io::stdout()
        .lock()
        .write_all(shown.as_bytes())
        .map_err(io_error("cannot write to stdout".to_owned()))
}

/// The traffic profile's identifier (RFC-0016 'Calculators',
/// `spec/schemas/audit-traffic-v1.schema.json`).
pub(crate) const TRAFFIC_SCHEMA: &str = "tessera.audit-traffic/1";

/// The plan's identifier (RFC-0016 'Calculators', `spec/schemas/audit-plan-v1.schema.json`).
pub(crate) const PLAN_SCHEMA: &str = "tessera.audit-plan/1";

/// The flags of `tessera audit plan`.
pub(crate) struct PlanOptions<'a> {
    pub(crate) policy: &'a Path,
    pub(crate) traffic: &'a Path,
    /// The execution plan whose counts give `μ`, or `None` for the source's counts.
    pub(crate) plan: Option<&'a Path>,
    pub(crate) gpus: u32,
    pub(crate) batch_tokens: u32,
    pub(crate) eta: f64,
    pub(crate) target_overhead: f64,
    pub(crate) out: Option<&'a Path>,
}

/// An unsigned integer field of the traffic profile.
fn count(object: &Map<String, Value>, key: &str) -> Result<u64, TesseraError> {
    object
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| usage(format!("a bucket's {key} is not an unsigned integer")))
}

/// Reads a `tessera.audit-traffic/1` file strictly: no key missing or unknown, and each bucket
/// its requests, mean length and violating requests as unsigned integers.
pub(crate) fn read_traffic(path: &Path) -> Result<Vec<BucketTraffic>, TesseraError> {
    let shown = path.display().to_string();
    let text = fs::read_to_string(path).map_err(io_error(format!("cannot read {shown}")))?;
    let value: Value =
        serde_json::from_str(&text).map_err(|e| usage(format!("{shown} is not JSON: {e}")))?;
    let object = value
        .as_object()
        .ok_or_else(|| usage(format!("{shown} is not a JSON object")))?;
    if let Some(key) = object
        .keys()
        .find(|k| !["schema", "buckets"].contains(&k.as_str()))
    {
        return Err(usage(format!("{shown} has an unknown key {key}")));
    }
    if object.get("schema").and_then(Value::as_str) != Some(TRAFFIC_SCHEMA) {
        return Err(usage(format!("{shown} is not a {TRAFFIC_SCHEMA} file")));
    }
    object
        .get("buckets")
        .and_then(Value::as_array)
        .ok_or_else(|| usage("the traffic's buckets is not an array".to_owned()))?
        .iter()
        .map(|b| {
            let b = b.as_object().filter(|b| b.len() == 3).ok_or_else(|| {
                usage("a bucket is not {requests, tokens, violations}".to_owned())
            })?;
            Ok(BucketTraffic {
                requests: count(b, "requests")?,
                tokens: count(b, "tokens")?,
                violations: count(b, "violations")?,
            })
        })
        .collect()
}

/// The plan as `tessera.audit-plan/1`.
fn plan_json(plan: &AuditPlan, source_defaults: bool) -> Value {
    let buckets: Vec<Value> = plan
        .buckets
        .iter()
        .map(|b| {
            serde_json::json!({
                "max_tokens": b.max_tokens,
                "rate": b.rate,
                "lowest_later_rate": b.lowest_later_rate,
                "requests": b.traffic.requests,
                "tokens": b.traffic.tokens,
                "violations": b.traffic.violations,
                "mu": b.mu,
                "expected_selected": b.expected_selected,
                "target_rate": b.target_rate,
                "detection": b.detection,
                "detection_in_place": b.detection_in_place,
            })
        })
        .collect();
    serde_json::json!({
        "schema": PLAN_SCHEMA,
        "source_defaults": source_defaults,
        "eta": plan.eta,
        "target_overhead": plan.target_overhead,
        "buckets": buckets,
        "detection": plan.detection,
        "detection_in_place": plan.detection_in_place,
        "overhead": plan.overhead,
        "expected_selected": plan.expected_selected,
    })
}

/// The plan as a table, one row per bucket, then the totals.
fn plan_table(plan: &AuditPlan) -> String {
    let mut text = format!(
        "{:>10} {:>9} {:>9} {:>12} {:>9} {:>10} {:>8} {:>12} {:>9} {:>10} {:>9}\n",
        "max tokens",
        "rate",
        "lowest",
        "requests",
        "tokens",
        "violating",
        "μ",
        "selected",
        "target",
        "detection",
        "in place",
    );
    for b in &plan.buckets {
        text.push_str(&format!(
            "{:>10} {:>8.4}% {:>8.4}% {:>12} {:>9} {:>10} {:>8.3} {:>12.1} {:>8.4}% {:>9.4}% {:>8.4}%\n",
            b.max_tokens,
            100.0 * b.rate,
            100.0 * b.lowest_later_rate,
            b.traffic.requests,
            b.traffic.tokens,
            b.traffic.violations,
            b.mu,
            b.expected_selected,
            100.0 * b.target_rate,
            100.0 * b.detection,
            100.0 * b.detection_in_place,
        ));
    }
    text.push_str(&format!(
        "detection {:.4}% against padding, {:.4}% in place; amortized overhead {:.4}%; {:.1} requests selected\n",
        100.0 * plan.detection,
        100.0 * plan.detection_in_place,
        100.0 * plan.overhead,
        plan.expected_selected,
    ));
    text
}

/// Runs `tessera audit plan`: the policy's detection probability and amortized overhead at a
/// traffic profile, with the cost model's `μ` at each bucket's mean length (RFC-0016
/// 'Calculators'). With `json` the plan is printed as `tessera.audit-plan/1`, and otherwise as a
/// table; `--out` writes the JSON through a temporary file renamed into place.
pub(crate) fn plan(options: &PlanOptions<'_>, json: bool) -> Result<(), TesseraError> {
    let policy = read_policy(options.policy)?;
    let traffic = read_traffic(options.traffic)?;
    let refused = |e: PlanError| usage(format!("{}: {e}", options.traffic.display()));
    // The profile is checked first, so that the cost model runs only at lengths in the buckets.
    check_traffic(&policy, &traffic).map_err(refused)?;
    let contexts = traffic
        .iter()
        .map(|t| u32::try_from(t.tokens))
        .collect::<Result<Vec<u32>, _>>()
        .map_err(|_| usage("a bucket's mean length is past 2^32 − 1".to_owned()))?;
    let (source_defaults, counts) = match options.plan {
        Some(plan) => (false, crate::compile::counts_from_plan(plan)?),
        None => (true, Counts::SOURCE_405B),
    };
    let scenario = Scenario {
        gpus: options.gpus,
        batch_tokens: options.batch_tokens,
        eta: options.eta,
        target_overhead: options.target_overhead,
        violations: 0,
    };
    let model = ModelReport::new(
        source_defaults,
        counts,
        UnitCosts::SOURCE,
        MemoryModel::NODE_405B,
        scenario,
        &contexts,
    )
    .map_err(crate::bench::model_error)?;
    let mu: Vec<f64> = model.projections.iter().map(|p| p.mu.total).collect();
    let plan = AuditPlan::new(&policy, &traffic, &mu, options.eta, options.target_overhead)
        .map_err(refused)?;
    let text = serde_json::to_string_pretty(&plan_json(&plan, source_defaults))
        .map(|text| format!("{text}\n"))
        .map_err(|e| TesseraError::new(ErrorCode::Internal).with_source(e.to_string()))?;
    if let Some(path) = options.out {
        write_atomically(path, &text)?;
    }
    let shown = match json {
        true => text,
        false => plan_table(&plan),
    };
    io::stdout()
        .lock()
        .write_all(shown.as_bytes())
        .map_err(io_error("cannot write to stdout".to_owned()))
}
