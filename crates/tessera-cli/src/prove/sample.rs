// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tessera prove --sample`: one batch of a sampled audit (RFC-0016 'Batching'), whose request is
//! read from the period's logs and whose tokens come from the stored transcripts. [`request`]
//! runs in four steps, all before the prover commits anything:
//!
//! 1. **The batch.** The sample's batches come from `crate::audit::sampled`, the function
//!    `tessera audit batch` reports them with, so `--batch K` names the report's batch `K`. A `K`
//!    past the last batch is `ERR-USAGE`.
//! 2. **The transcripts.** `--transcripts DIR` holds one `SessionTranscript` (0x0404) per session,
//!    named `<session_id>.tst`, the identifier in 32 lowercase hexadecimal digits. A session with
//!    no file there has no stored transcript, which the pre-check refuses.
//! 3. **The pre-check** (`tessera_anchor::precheck`, RFC-0015 'Prover pre-check'): the request's
//!    lengths come from its frames, and its stored frames re-seal to the logged digests. A request
//!    whose frames give no lengths, or any failed check, is `ERR-WITNESS-ANCHOR`, and nothing is
//!    proven.
//! 4. **The tokens**: its `In` frames' tokens, then its `Out` frames', from the stored frames at
//!    the counters the log names (`SessionTranscript::request_tokens`).
//!
//! The sample's policy also names the parameter set every proof of the audit is made under,
//! which the claim list must name too (RFC-0016 'Verifier checks', #530). The caller checks it
//! against the plan, also before anything is committed.
//!
//! Errors name requests and sessions by their identifiers and files by their paths, all public;
//! the keys and tokens stay in their `Secret`s (05 §3).

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use tessera_anchor::{BatchRequest, PrecheckFailure, PrecheckKind, SessionTranscript, precheck};
use tessera_codec::error::{ErrorCode, TesseraError};
use tessera_codec::{Secret, schema};
use tessera_protocol as protocol;

use super::{ProveArgs, usage};
use crate::audit::{self, Logs};
use crate::{Input, io_error};

/// The sampled requests and their slotted tokens, ready to prove.
pub(super) struct Sampled {
    /// Requests in statement order.
    pub(super) requests: Vec<protocol::RequestRef>,
    /// Public receipt entries for requests whose anchor pre-check failed.
    pub(super) unproven: Vec<serde_json::Value>,
    /// The sample's `manifest_digest`, the period's manifest identity.
    pub(super) manifest_digest: [u8; 32],
    /// The parameter set the sample's policy names.
    pub(super) params: tessera_hash::ParamSetId,
    /// Per-request prompt/output tokens in RFC-0071 slots.
    pub(super) tokens: Secret<Vec<u32>>,
    /// Every file read, which the proof must not overwrite.
    pub(super) inputs: Vec<PathBuf>,
    transcripts: Vec<SessionTranscript>,
    window: audit::Window,
}

/// The file of `session_id`'s transcript in `dir`: `<session_id>.tst`, in lowercase hex.
fn transcript_path(dir: &Path, session_id: &[u8; 16]) -> PathBuf {
    let name: String = session_id.iter().map(|b| format!("{b:02x}")).collect();
    dir.join(format!("{name}.tst"))
}

fn omitted(session: [u8; 16], request_id: u64, code: ErrorCode) -> serde_json::Value {
    let session_id: String = session.iter().map(|b| format!("{b:02x}")).collect();
    serde_json::json!({"session_id":session_id,"request_id":request_id,"code":code.to_string()})
}

fn anchor(failure: PrecheckFailure) -> TesseraError {
    TesseraError::new(failure.code()).with_source(failure.to_string())
}

fn internal(what: &'static str) -> TesseraError {
    TesseraError::new(ErrorCode::Internal).with_source(what)
}

/// Reads, checks and pre-checks batch `--batch` of the sample at `path`.
pub(super) fn request(a: &ProveArgs, path: &Path) -> Result<Sampled, TesseraError> {
    let (Some(manifest), Some(log), Some(dir), Some(k)) = (
        a.manifest.as_deref(),
        a.log.as_deref(),
        a.transcripts.as_deref(),
        a.batch,
    ) else {
        return Err(usage(
            "--sample needs --manifest, --log, --transcripts and --batch",
        ));
    };
    let logs = Logs {
        manifest,
        log,
        previous_manifest: a.previous_manifest.as_deref(),
        previous_log: a.previous_log.as_deref(),
        between: &a.between,
    };
    let sampled = audit::sampled(path, &logs)?;
    let mut inputs: Vec<PathBuf> = logs.paths().into_iter().map(Path::to_path_buf).collect();
    let count = sampled.batches.len();
    let batch = usize::try_from(k)
        .ok()
        .and_then(|k| sampled.batches.get(k))
        .ok_or_else(|| {
            usage(format!(
                "the sample has {count} batches, numbered from 0: there is no batch {k}"
            ))
        })?;
    if batch.requests.is_empty() {
        return Err(usage("a sampled batch contains at least one request"));
    }
    let mut unproven = Vec::new();
    let mut first_failure = None;
    let mut requests = Vec::new();
    for request in &batch.requests {
        let key = request.key;
        let Some(lengths) = request.lengths else {
            let failure = PrecheckFailure {
                request_id: key.request_id,
                kind: PrecheckKind::Lengths,
            };
            unproven.push(omitted(key.session_id, key.request_id, failure.code()));
            first_failure.get_or_insert_with(|| anchor(failure));
            continue;
        };
        requests.push(BatchRequest {
            session_id: key.session_id,
            request_id: key.request_id,
            prompt_len: lengths.prompt_len,
            output_len: lengths.output_len,
        });
    }
    if !fs::metadata(dir)
        .map_err(io_error(format!("cannot open {}", dir.display())))?
        .is_dir()
    {
        return Err(TesseraError::new(ErrorCode::ResIo)
            .with_source(format!("{} is not a directory", dir.display())));
    }
    // One transcript per session of the batch; a missing one is the pre-check's to name.
    let mut transcripts: Vec<SessionTranscript> = Vec::new();
    for request in &requests {
        if transcripts
            .iter()
            .any(|t| t.session_id() == request.session_id)
        {
            continue;
        }
        let file = transcript_path(dir, &request.session_id);
        match fs::symlink_metadata(&file) {
            Ok(_) => {
                transcripts.push(Input::new(&file, &schema::SESSION_TRANSCRIPT).open()?);
                inputs.push(file);
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(io_error(format!("cannot read {}", file.display()))(e)),
        }
    }
    let window = &sampled.window;
    let requests: Vec<_> = requests
        .into_iter()
        .filter(|request| {
            match precheck(
                core::slice::from_ref(request),
                &transcripts,
                &window.current,
                window.previous.as_ref(),
            ) {
                Ok(()) => true,
                Err(failure) => {
                    unproven.push(omitted(
                        request.session_id,
                        request.request_id,
                        failure.code(),
                    ));
                    first_failure.get_or_insert_with(|| anchor(failure));
                    false
                }
            }
        })
        .collect();
    if requests.is_empty() {
        return Err(
            first_failure.unwrap_or_else(|| internal("a sampled batch has no pre-checked request"))
        );
    }
    precheck(
        &requests,
        &transcripts,
        &window.current,
        window.previous.as_ref(),
    )
    .map_err(anchor)?;
    // The pre-check found the session's transcript and replayed every logged frame.
    let stored = requests
        .iter()
        .map(|request| {
            transcripts
                .iter()
                .find(|t| t.session_id() == request.session_id)
                .map(|t| (t, request.request_id))
                .ok_or_else(|| internal("a pre-checked request has its session transcript"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let slot_len = requests
        .iter()
        .map(|r| r.prompt_len.checked_add(r.output_len))
        .collect::<Option<Vec<_>>>()
        .and_then(|lengths| lengths.into_iter().max())
        .and_then(u32::checked_next_power_of_two)
        .ok_or_else(|| usage("batch slot length overflow"))?;
    let tokens = SessionTranscript::batch_tokens(
        window.previous.as_ref(),
        &window.current,
        slot_len,
        &stored,
    )
    .ok_or_else(|| internal("a pre-checked batch's tokens cannot be rebuilt"))?;
    Ok(Sampled {
        transcripts,
        window: sampled.window,
        unproven,
        requests: requests
            .iter()
            .map(|r| protocol::RequestRef {
                request_id: r.request_id,
                session_id: r.session_id,
                prompt_len: r.prompt_len,
                output_len: r.output_len,
            })
            .collect(),
        manifest_digest: *sampled.sample.manifest_digest(),
        params: sampled.sample.policy().params().clone(),
        tokens,
        inputs,
    })
}

impl Sampled {
    pub(super) fn replay(
        &mut self,
        prover: &tessera_prover::Prover<'_>,
        batch: &mut tessera_prover::ProofBatch,
    ) -> Result<(), TesseraError> {
        let mut survivors = Vec::new();
        let mut first = None;
        for &request in &batch.statement.batch.requests {
            let transcript = self
                .transcripts
                .iter()
                .find(|t| t.session_id() == request.session_id)
                .ok_or_else(|| internal("a checked request has its transcript"))?;
            let tokens = transcript
                .request_tokens(
                    self.window.previous.as_ref(),
                    &self.window.current,
                    request.request_id,
                )
                .ok_or_else(|| internal("a checked request has its tokens"))?;
            let mut statement = batch.statement.clone();
            statement.batch.requests = vec![request];
            match prover.check_witness(&tessera_prover::ProofBatch { statement, tokens }) {
                Ok(()) => survivors.push(request),
                Err(error) if error.code() == ErrorCode::WitnessExactMismatch => {
                    self.unproven.push(omitted(
                        request.session_id,
                        request.request_id,
                        error.code(),
                    ));
                    first.get_or_insert_with(|| super::prover_error(error));
                }
                Err(error) => return Err(super::prover_error(error)),
            }
        }
        if survivors.is_empty() {
            return Err(first.unwrap_or_else(|| internal("no sampled request survived replay")));
        }
        let slot = survivors
            .iter()
            .filter_map(|r| r.prompt_len.checked_add(r.output_len))
            .max()
            .and_then(u32::checked_next_power_of_two)
            .ok_or_else(|| usage("batch slot length overflow"))?;
        let stored = survivors
            .iter()
            .map(|r| {
                self.transcripts
                    .iter()
                    .find(|t| t.session_id() == r.session_id)
                    .map(|t| (t, r.request_id))
                    .ok_or_else(|| internal("a checked request has its transcript"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        batch.tokens = SessionTranscript::batch_tokens(
            self.window.previous.as_ref(),
            &self.window.current,
            slot,
            &stored,
        )
        .ok_or_else(|| internal("checked batch tokens fit their slots"))?;
        batch.statement.batch.requests = survivors;
        Ok(())
    }
}
