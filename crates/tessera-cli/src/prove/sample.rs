// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tessera prove --sample`: one batch of a sampled audit (RFC-0016 'Batching'), whose request is
//! read from the period's logs and whose tokens come from the stored transcripts. [`request`]
//! runs in four steps, all before the prover commits anything:
//!
//! 1. **The batch.** The sample's batches come from `crate::audit::sampled`, the function
//!    `tessera audit batch` reports them with, so `--batch K` names the report's batch `K`. A `K`
//!    past the last batch is `ERR-USAGE`, and so is a batch of several requests until
//!    multi-request proof units exist (#357).
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

/// The request of a batch of one, ready to prove.
pub(super) struct Sampled {
    /// The request as its statement lists it.
    pub(super) request: protocol::RequestRef,
    /// The sample's `manifest_digest`, the period's manifest identity.
    pub(super) manifest_digest: [u8; 32],
    /// Its prompt, then its output.
    pub(super) tokens: Secret<Vec<u32>>,
    /// Every file read, which the proof must not overwrite.
    pub(super) inputs: Vec<PathBuf>,
}

/// The file of `session_id`'s transcript in `dir`: `<session_id>.tst`, in lowercase hex.
fn transcript_path(dir: &Path, session_id: &[u8; 16]) -> PathBuf {
    let name: String = session_id.iter().map(|b| format!("{b:02x}")).collect();
    dir.join(format!("{name}.tst"))
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
    let [only] = &batch.requests[..] else {
        return Err(usage(format!(
            "batch {k} holds {} requests, and a proof of several waits on multi-request proof units (#357)",
            batch.requests.len()
        )));
    };
    let key = only.key;
    let lengths = only.lengths.ok_or_else(|| {
        anchor(PrecheckFailure {
            request_id: key.request_id,
            kind: PrecheckKind::Lengths,
        })
    })?;
    if !fs::metadata(dir)
        .map_err(io_error(format!("cannot open {}", dir.display())))?
        .is_dir()
    {
        return Err(TesseraError::new(ErrorCode::ResIo)
            .with_source(format!("{} is not a directory", dir.display())));
    }
    // One transcript per session of the batch; a missing one is the pre-check's to name.
    let mut transcripts: Vec<SessionTranscript> = Vec::new();
    let file = transcript_path(dir, &key.session_id);
    match fs::symlink_metadata(&file) {
        Ok(_) => {
            transcripts.push(Input::new(&file, &schema::SESSION_TRANSCRIPT).open()?);
            inputs.push(file);
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(io_error(format!("cannot read {}", file.display()))(e)),
    }
    let requests = [BatchRequest {
        session_id: key.session_id,
        request_id: key.request_id,
        prompt_len: lengths.prompt_len,
        output_len: lengths.output_len,
    }];
    let window = &sampled.window;
    precheck(
        &requests,
        &transcripts,
        &window.current,
        window.previous.as_ref(),
    )
    .map_err(anchor)?;
    // The pre-check found the session's transcript and replayed every logged frame.
    let tokens = transcripts
        .iter()
        .find(|t| t.session_id() == key.session_id)
        .and_then(|t| t.request_tokens(window.previous.as_ref(), &window.current, key.request_id))
        .ok_or_else(|| internal("a pre-checked request's tokens cannot be rebuilt"))?;
    Ok(Sampled {
        request: protocol::RequestRef {
            request_id: key.request_id,
            session_id: key.session_id,
            prompt_len: lengths.prompt_len,
            output_len: lengths.output_len,
        },
        manifest_digest: *sampled.sample.manifest_digest(),
        tokens,
        inputs,
    })
}
