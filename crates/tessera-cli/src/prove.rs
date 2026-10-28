// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Protocol/1 proving of one exact-greedy request, on the CPU, non-interactively (`--out`) or
//! live over an RFC-0014 directory (`--live`). The request comes from one of two sources:
//!
//! - **`--reference-tokens`**: one recorded TokenFile sequence, prompt then continuation, with
//!   the request's identifiers and manifest digest given as flags;
//! - **`--sample`**: one batch of a sampled audit (`sample`), its request read from the period's
//!   logs and its tokens from the stored transcripts, pre-checked before anything is committed.
//!
//! Both build the `PublicStatement` the same way, in [`produce`].
use crate::{Input, io_error};
use clap::{ArgGroup, Args};
use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    time::Duration,
};
use tessera_codec::{
    Secret, envelope,
    error::{ErrorCode, TesseraError},
    schema,
};
use tessera_compiler::ExecutionPlan;
use tessera_ir::IntegerModel;
use tessera_pcs::WeightCommitmentState;
use tessera_protocol::{self as protocol, ParamSet};
use tessera_prover::{
    Backend, MaskRng, MemoryBudget, Mode, ProofBatch, Prover, ProverConfig, ReleaseConfig,
};
mod directory;
mod sample;
mod tokens;

#[derive(Debug, Args)]
#[command(group(ArgGroup::new("mode").required(true).args(["live","out"])))]
#[command(group(ArgGroup::new("input").required(true).args(["sample","reference_tokens"])))]
pub(crate) struct ProveArgs {
    #[arg(long)]
    plan: PathBuf,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    weights: PathBuf,
    /// One batch of a sampled audit: the sample its requests were drawn in (RFC-0016).
    #[arg(long, value_name = "SAMPLE.tas", requires_all = ["transcripts", "batch", "manifest", "log"])]
    sample: Option<PathBuf>,
    /// The stored transcripts, one per session: `<session_id>.tst`, the identifier in 32
    /// lowercase hexadecimal digits.
    #[arg(long, value_name = "DIR", requires = "sample")]
    transcripts: Option<PathBuf>,
    /// The batch to prove: its index in `tessera audit batch`'s report, from 0.
    #[arg(long, value_name = "K", requires = "sample")]
    batch: Option<u32>,
    /// The sample's period's manifest.
    #[arg(long, value_name = "MANIFEST.tlm", requires = "sample")]
    manifest: Option<PathBuf>,
    /// The period's log.
    #[arg(long, value_name = "LOG.tdl", requires = "sample")]
    log: Option<PathBuf>,
    /// The manifest of the last period of non-zero length before it, unless it has none.
    #[arg(long, value_name = "MANIFEST.tlm", requires_all = ["sample", "previous_log"])]
    previous_manifest: Option<PathBuf>,
    /// That period's log.
    #[arg(long, value_name = "LOG.tdl", requires_all = ["sample", "previous_manifest"])]
    previous_log: Option<PathBuf>,
    /// A period of zero length between the two, in order; repeated for each.
    #[arg(long, value_name = "MANIFEST.tlm", requires = "sample")]
    between: Vec<PathBuf>,
    /// One existing TokenFile sequence: prompt followed by recorded continuation.
    #[arg(long,requires_all=["prompt_len","request_id","request_session","manifest_digest"])]
    reference_tokens: Option<PathBuf>,
    #[arg(long, requires = "reference_tokens")]
    prompt_len: Option<u32>,
    #[arg(long, requires = "reference_tokens")]
    request_id: Option<u64>,
    #[arg(long, requires = "reference_tokens")]
    request_session: Option<String>,
    #[arg(long, requires = "reference_tokens")]
    manifest_digest: Option<String>,
    /// Existing, fresh RFC-0014 directory; authenticated network transport is later work.
    #[arg(long)]
    live: Option<PathBuf>,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long,value_parser=["cpu","gpu"],default_value="cpu")]
    backend: String,
    #[arg(long, value_delimiter = ',')]
    devices: Vec<u32>,
    #[arg(long,value_parser=clap::value_parser!(u64).range(1..))]
    memory_budget: Option<u64>,
    #[arg(long, requires = "live")]
    timeout_ms: Option<u64>,
    /// Pins the masks for reproducibility tests (OBS-5); release builds refuse `test-rng`.
    #[cfg(feature = "test-rng")]
    #[arg(long, hide = true)]
    test_mask_seed: Option<u64>,
}
#[cfg(not(feature = "test-rng"))]
fn mask_rng(_: &ProveArgs) -> MaskRng {
    MaskRng::OsRng
}
#[cfg(feature = "test-rng")]
fn mask_rng(a: &ProveArgs) -> MaskRng {
    a.test_mask_seed.map_or(MaskRng::OsRng, MaskRng::Seeded)
}

fn usage(message: impl Into<String>) -> TesseraError {
    TesseraError::new(ErrorCode::Usage).with_source(message.into())
}
fn prover_error(e: tessera_prover::ProverError) -> TesseraError {
    TesseraError::new(e.code()).with_source(e)
}
fn hex<const N: usize>(value: &str) -> Result<[u8; N], TesseraError> {
    if value.len() != N * 2 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(usage("invalid request digest/session hex"));
    }
    let mut out = [0; N];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(
            value
                .get(2 * i..2 * i + 2)
                .ok_or_else(|| usage("invalid request hex"))?,
            16,
        )
        .map_err(|_| usage("invalid request hex"))?;
    }
    Ok(out)
}
fn digest(value: [u8; 32]) -> String {
    value.iter().map(|b| format!("{b:02x}")).collect()
}
fn distinct(path: &Path, inputs: &[&Path]) -> Result<(), TesseraError> {
    let destination = fs::canonicalize(path)
        .or_else(|_| {
            let parent = path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new("."));
            Ok::<_, io::Error>(
                fs::canonicalize(parent)?
                    .join(path.file_name().ok_or(io::ErrorKind::InvalidInput)?),
            )
        })
        .map_err(io_error("cannot resolve output path".into()))?;
    if inputs
        .iter()
        .any(|p| fs::canonicalize(p).is_ok_and(|p| p == destination))
    {
        return Err(usage("proof output aliases an input"));
    }
    Ok(())
}
pub(crate) fn prove(a: &ProveArgs, json: bool) -> Result<(), TesseraError> {
    // Beside the secret weight state: the operator's own, never the exchange directory (05 §6).
    crate::manifest::beside(&a.weights);
    if a.backend != "cpu" || !a.devices.is_empty() {
        return Err(usage("the reference prover supports CPU only"));
    }
    if let Some(path) = &a.sample {
        // Everything the sample route reads is checked, and the batch pre-checked, here.
        let mut sampled = sample::request(a, path)?;
        let requests = std::mem::take(&mut sampled.requests);
        let tokens = std::mem::replace(&mut sampled.tokens, Secret::new(Vec::new()));
        let sample_inputs = sampled.inputs.clone();
        let params = sampled.params.clone();
        let mut inputs: Vec<&Path> = vec![path];
        inputs.extend(sample_inputs.iter().map(PathBuf::as_path));
        return produce(
            a,
            json,
            &inputs,
            sampled.manifest_digest,
            Some(&params),
            Some(&mut sampled),
            || Ok((requests, tokens)),
        );
    }
    let input = a
        .reference_tokens
        .as_deref()
        .ok_or_else(|| usage("reference tokens required"))?;
    let prompt = a
        .prompt_len
        .ok_or_else(|| usage("prompt length required"))?;
    if prompt == 0 {
        return Err(usage("the prompt is nonempty"));
    }
    let request_id = a.request_id.ok_or_else(|| usage("request id required"))?;
    let session_id = hex::<16>(
        a.request_session
            .as_deref()
            .ok_or_else(|| usage("request session required"))?,
    )?;
    let manifest_digest = hex::<32>(
        a.manifest_digest
            .as_deref()
            .ok_or_else(|| usage("manifest digest required"))?,
    )?;
    produce(a, json, &[input], manifest_digest, None, None, || {
        let data: tokens::Tokens = Input::new(input, &schema::TOKEN_FILE).open()?;
        let output_len = data
            .len
            .checked_sub(prompt)
            .filter(|n| *n > 0)
            .ok_or_else(|| usage("recorded continuation is nonempty"))?;
        let request = protocol::RequestRef {
            request_id,
            session_id,
            prompt_len: prompt,
            output_len,
        };
        Ok((vec![request], data.values))
    })
}

/// Proves a batch: reads the model, plan and weight state, then its request references
/// and slotted tokens, and builds their `PublicStatement` with
/// `manifest_digest`, proves it in the mode the flags give, and prints the `tessera.proof-run/1`
/// receipt. `inputs` are the files besides those three that the output must not alias.
///
/// `policy_params` is a sample's policy's parameter set: the plan's claim list must name it, else
/// `ERR-USAGE` before anything is committed, since the verifier would refuse the proof as
/// `Reject(Params)` (RFC-0016 'Verifier checks').
fn produce(
    a: &ProveArgs,
    json: bool,
    inputs: &[&Path],
    manifest_digest: [u8; 32],
    policy_params: Option<&tessera_hash::ParamSetId>,
    mut sampled: Option<&mut sample::Sampled>,
    request: impl FnOnce() -> Result<(Vec<protocol::RequestRef>, Secret<Vec<u32>>), TesseraError>,
) -> Result<(), TesseraError> {
    if let Some(out) = &a.out {
        let mut all: Vec<&Path> = vec![&a.model, &a.plan, &a.weights];
        all.extend_from_slice(inputs);
        distinct(out, &all)?;
    }
    let model: IntegerModel = Input::new(&a.model, &schema::INTEGER_MODEL).open()?;
    let plan: ExecutionPlan = Input::new(&a.plan, &schema::EXECUTION_PLAN).open()?;
    if let Some(params) = policy_params.filter(|&p| *p != plan.claims.params) {
        return Err(usage(format!(
            "the claim list's parameter set {} is not the sample policy's {} (RFC-0016)",
            plan.claims.params.0.as_str(),
            params.0.as_str()
        )));
    }
    if !fs::metadata(&a.weights)
        .map_err(io_error("cannot read weight state".into()))?
        .is_file()
    {
        return Err(TesseraError::new(ErrorCode::ResIo));
    }
    let mut weights = WeightCommitmentState::read(&a.weights).map_err(|e| match e {
        envelope::ReadError::Invalid(e) => TesseraError::from(e),
        envelope::ReadError::Io(_) => TesseraError::new(ErrorCode::ResIo),
    })?;
    crate::manifest::input(&a.weights, &schema::WEIGHT_COMMITMENT_STATE);
    let (requests, tokens) = request()?;
    let params = ParamSet::defined(&plan.claims.params)
        .ok_or_else(|| usage("undefined claim-list parameter set"))?;
    crate::manifest::params(params.id, Some("tessera-protocol/1"));
    let mode = if a.live.is_some() {
        Mode::Live
    } else {
        Mode::NonInteractive
    };
    let budget = a
        .memory_budget
        .unwrap_or(64)
        .checked_mul(1 << 30)
        .ok_or_else(|| usage("memory budget overflows"))?;
    let statement = protocol::PublicStatement {
        claim_list_digest: tessera_hash::hash(
            tessera_hash::Domain::ClaimList,
            &tessera_codec::encode(&plan.claims),
        ),
        weights_root: weights.public.weights_root(),
        weights_epoch: weights.public.epoch,
        manifest_digest,
        batch: protocol::BatchDescriptor { requests },
        predictor: protocol::PredictorSpec::Exact {
            decoding: protocol::Decoding::Greedy,
        },
    };
    let mut batch = ProofBatch { statement, tokens };
    let cfg = ProverConfig {
        backend: Backend::Cpu { threads: 1 },
        params: *params,
        mode,
        memory_budget: MemoryBudget { host: budget },
        mask_rng: mask_rng(a),
        release: ReleaseConfig,
    };
    let mut prover = Prover::new(&plan, &model, &mut weights, cfg).map_err(prover_error)?;
    if let Some(sampled) = sampled.as_deref_mut() {
        sampled.replay(&prover, &mut batch)?;
    }
    let unproven = sampled.as_ref().map_or(&[][..], |s| s.unproven.as_slice());
    let statement_digest = digest(batch.statement.digest());
    let root = digest(batch.statement.weights_root);
    if let Some(dir) = &a.live {
        directory::run(
            &mut prover,
            &batch,
            dir,
            Duration::from_millis(a.timeout_ms.unwrap_or(600_000)),
        )?;
    } else {
        let proof = prover.prove_non_interactive(&batch).map_err(prover_error)?;
        directory::publish(
            a.out
                .as_deref()
                .ok_or_else(|| usage("proof output required"))?,
            &envelope::seal(&schema::NI_PROOF, &proof),
        )?;
    }
    let mut report = serde_json::json!({"schema":"tessera.proof-run/1","protocol":"tessera-protocol/1","params":params.id,"mode":if mode==Mode::Live {"Live"}else{"NonInteractive"},"statement_digest":statement_digest,"weights_root":root,"test_parameters":params.id=="test-small/1","zero_knowledge":false,"verifier_acceptance":false});
    if !unproven.is_empty() {
        report["unproven"] = serde_json::json!(unproven);
    }
    let text = if json {
        format!("{report}\n")
    } else {
        let mut text = format!(
            "Proof produced ({statement_digest}); obtain the independent verifier verdict\n"
        );
        for request in unproven {
            text.push_str(&format!(
                "Unproven request {} in session {}: {}\n",
                request["request_id"],
                request["session_id"].as_str().unwrap_or(""),
                request["code"].as_str().unwrap_or("")
            ));
        }
        text
    };
    io::stdout()
        .lock()
        .write_all(text.as_bytes())
        .map_err(io_error("cannot write to stdout".into()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn metadata_hex_is_exact_and_rejects_non_ascii() {
        assert_eq!(hex::<16>(&"aF".repeat(16)).unwrap(), [0xaf; 16]);
        for value in [
            "00".repeat(15),
            "00".repeat(17),
            "gg".repeat(16),
            "+1".repeat(16),
            "é".repeat(16),
        ] {
            assert_eq!(hex::<16>(&value).unwrap_err().code, ErrorCode::Usage);
        }
    }
}
