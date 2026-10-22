// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Protocol/1 reference proving for one recorded exact-greedy TokenFile sequence.
use crate::{Input, io_error};
use clap::{ArgGroup, Args};
use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    time::Duration,
};
use tessera_codec::{
    envelope,
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
    /// Future sampled-audit path (#157).
    #[arg(long, requires = "transcripts")]
    sample: Option<PathBuf>,
    #[arg(long, requires = "sample")]
    transcripts: Option<PathBuf>,
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
}
fn usage(message: &'static str) -> TesseraError {
    TesseraError::new(ErrorCode::Usage).with_source(message)
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
    if let Some(sample) = &a.sample {
        return crate::pending(
            "prove",
            &[
                Input::new(&a.plan, &schema::EXECUTION_PLAN),
                Input::new(&a.model, &schema::INTEGER_MODEL),
                Input::new(&a.weights, &schema::WEIGHT_COMMITMENT_STATE),
                Input::new(sample, &schema::AUDIT_SAMPLE),
            ],
            157,
        );
    }
    let input = a
        .reference_tokens
        .as_deref()
        .ok_or_else(|| usage("reference tokens required"))?;
    if a.backend != "cpu" || !a.devices.is_empty() {
        return Err(usage("the reference prover supports CPU only"));
    }
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
    if let Some(out) = &a.out {
        distinct(out, &[&a.model, &a.plan, &a.weights, input])?;
    }
    let model: IntegerModel = Input::new(&a.model, &schema::INTEGER_MODEL).open()?;
    let plan: ExecutionPlan = Input::new(&a.plan, &schema::EXECUTION_PLAN).open()?;
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
    let data: tokens::Tokens = Input::new(input, &schema::TOKEN_FILE).open()?;
    let output_len = data
        .len
        .checked_sub(prompt)
        .filter(|n| *n > 0)
        .ok_or_else(|| usage("recorded continuation is nonempty"))?;
    let params = ParamSet::defined(&plan.claims.params)
        .ok_or_else(|| usage("undefined claim-list parameter set"))?;
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
        batch: protocol::BatchDescriptor {
            requests: vec![protocol::RequestRef {
                request_id,
                session_id,
                prompt_len: prompt,
                output_len,
            }],
        },
        predictor: protocol::PredictorSpec::Exact {
            decoding: protocol::Decoding::Greedy,
        },
    };
    let statement_digest = digest(statement.digest());
    let root = digest(statement.weights_root);
    let batch = ProofBatch {
        statement,
        tokens: data.values,
    };
    let cfg = ProverConfig {
        backend: Backend::Cpu { threads: 1 },
        params: *params,
        mode,
        memory_budget: MemoryBudget { host: budget },
        mask_rng: MaskRng::OsRng,
        release: ReleaseConfig,
    };
    let mut prover = Prover::new(&plan, &model, &mut weights, cfg).map_err(prover_error)?;
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
    let report = serde_json::json!({"schema":"tessera.proof-run/1","protocol":"tessera-protocol/1","params":params.id,"mode":if mode==Mode::Live {"Live"}else{"NonInteractive"},"statement_digest":statement_digest,"weights_root":root,"test_parameters":params.id=="test-small/1","zero_knowledge":false,"verifier_acceptance":false});
    let text = if json {
        format!("{report}\n")
    } else {
        format!("Proof produced ({statement_digest}); obtain the independent verifier verdict\n")
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
