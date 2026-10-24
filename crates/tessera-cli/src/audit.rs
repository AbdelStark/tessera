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
//! 3. **The policy** is a `tessera.audit-policy/1` file, read strictly, each rate a decimal
//!    (`tessera_protocol::RateBucket::threshold_of`).
//! 4. **The coin toss.** `--verifier-commitment`, `--prover-share` and `--verifier-share` are its
//!    three values, which the exchange of RFC-0016 step 2 gave both parties. A share that does
//!    not open the commitment is `ERR-PROTO-COINTOSS`.
//! 5. **The sample**, drawn among the period's candidates (`tessera_anchor::candidates`), is
//!    written to `--out` as an `AuditSample` envelope, through a temporary file renamed into
//!    place.
//!
//! The command checks no recorder signature: the verifier does, before it derives the same
//! sample (RFC-0062 'The check'). Everything it reads and writes is public (03 §5).

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};
use tessera_anchor::recorder::decode_log;
use tessera_anchor::{LogManifest, VerifiedLog, candidates};
use tessera_codec::error::{ErrorCode, TesseraError};
use tessera_codec::{Ascii, DecodeLimits, Public, decode, encode, envelope, schema};
use tessera_hash::ParamSetId;
use tessera_protocol::{
    AuditPolicy, AuditSample, CoinTossTranscript, RateBucket, RequestKey, ShareRecord,
};

use crate::{Input, io_error, write_bytes_atomically};

/// The policy file's identifier (RFC-0016 'Policy', `spec/schemas/audit-policy-v1.schema.json`).
pub(crate) const POLICY_SCHEMA: &str = "tessera.audit-policy/1";

/// The flags of `tessera audit sample`.
pub(crate) struct SampleOptions<'a> {
    pub(crate) manifest: &'a Path,
    pub(crate) log: &'a Path,
    pub(crate) previous_manifest: Option<&'a Path>,
    pub(crate) previous_log: Option<&'a Path>,
    pub(crate) between: &'a [PathBuf],
    pub(crate) policy: &'a Path,
    pub(crate) verifier_commitment: [u8; 32],
    pub(crate) prover_share: [u8; 32],
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

/// Runs `tessera audit sample`.
pub(crate) fn sample(options: &SampleOptions<'_>) -> Result<(), TesseraError> {
    let current = manifest(options.manifest)?;
    let previous = options.previous_manifest.map(manifest).transpose()?;
    let between = options
        .between
        .iter()
        .map(|p| manifest(p))
        .collect::<Result<Vec<_>, _>>()?;
    check_previous(previous.as_ref(), &between, &current)?;
    let previous_log = match (&previous, options.previous_log) {
        (Some(m), Some(path)) => Some(verified(m, path)?),
        (None, None) => None,
        _ => {
            return Err(usage(
                "--previous-manifest and --previous-log go together".to_owned(),
            ));
        }
    };
    let log = verified(&current, options.log)?;
    let policy = read_policy(options.policy)?;
    let coin_toss = CoinTossTranscript {
        verifier_commitment: options.verifier_commitment,
        prover_share: options.prover_share,
        verifier_share: options.verifier_share,
    };
    let found = candidates(previous_log.as_ref(), &log);
    let sample = AuditSample::draw(
        current.identity(),
        policy,
        coin_toss,
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
        TesseraError::new(e.code()).with_source("the verifier's share does not open its commitment")
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

/// The flags of `tessera audit share`.
pub(crate) struct ShareOptions<'a> {
    pub(crate) manifest: &'a Path,
    pub(crate) commitment: [u8; 32],
    pub(crate) state: &'a Path,
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
/// (RFC-0016 'Sample derivation', step 2), one commitment per manifest.
///
/// - **The record.** The answer is kept in `<state>/<manifest digest>.share`, the 96 bytes of
///   `TCE(ShareRecord)`, before the share is printed. Every responder for one prover must share
///   the directory.
/// - **One answer.** A first commitment gets a fresh share. The same commitment again gets the
///   same share, for a verifier that lost it. Any other commitment is refused with
///   `ERR-PROTO-ORDER`.
/// - **Racing responders.** The record is written under a temporary name and linked into place,
///   which fails if a record is already there. A responder that loses the race answers from the
///   record that won.
///
/// It prints the share as 64 hexadecimal digits, or with `json` a JSON object. The share is
/// public once sent.
pub(crate) fn share(options: &ShareOptions<'_>, json: bool) -> Result<(), TesseraError> {
    let manifest_digest = manifest(options.manifest)?.identity();
    let state = options.state;
    fs::create_dir_all(state).map_err(io_error(format!("cannot create {}", state.display())))?;
    let path = state.join(format!("{}.share", hex(&manifest_digest)));
    let protocol =
        |e: tessera_protocol::ProtocolError| TesseraError::new(e.code()).with_source(e.to_string());
    let record = match kept(&path)? {
        Some(record) => ShareRecord::answer(Some(&record), manifest_digest, options.commitment)
            .map_err(protocol)?,
        None => {
            let record =
                ShareRecord::answer(None, manifest_digest, options.commitment).map_err(protocol)?;
            let mut tmp = path.as_os_str().to_owned();
            tmp.push(format!(".tmp-{}", std::process::id()));
            let tmp = PathBuf::from(tmp);
            let shown = path.display().to_string();
            let mut file =
                fs::File::create(&tmp).map_err(io_error(format!("cannot write {shown}")))?;
            file.write_all(&encode(&record))
                .and_then(|()| file.sync_all())
                .map_err(io_error(format!("cannot write {shown}")))?;
            drop(file);
            let linked = fs::hard_link(&tmp, &path);
            let _ = fs::remove_file(&tmp);
            match linked {
                Ok(()) => record,
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                    let won = kept(&path)?.ok_or_else(|| {
                        TesseraError::new(ErrorCode::ResIo)
                            .with_source(format!("{shown} vanished as it was written"))
                    })?;
                    ShareRecord::answer(Some(&won), manifest_digest, options.commitment)
                        .map_err(protocol)?
                }
                Err(e) => return Err(io_error(format!("cannot write {shown}"))(e)),
            }
        }
    };
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
