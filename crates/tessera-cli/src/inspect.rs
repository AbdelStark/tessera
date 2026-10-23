// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tessera inspect`: decodes any catalog TCE artifact and renders a structured text view
//! (02 §2.2, RFC-0003 'Non-Goals', 03 §5).
//!
//! Secret-class artifacts (`IntegerModel`, `WeightCommitmentState`, `TokenFile`, `LogitsFile`)
//! print shapes and BLAKE3 digests only, never weight entries, channel classes or multipliers,
//! mask seeds, token ids or logit values (05 §3).

use std::fs::File;
use std::io::{self, Read, Write as _};
use std::path::{Path, PathBuf};

use clap::Args;
use serde_json::{Value, json};
use tessera_codec::envelope::{self, ReadError};
use tessera_codec::error::{ErrorCode, SchemaId, TesseraError};
use tessera_codec::schema::{self, Class, Schema};
use tessera_codec::{Decode, DecodeLimits, SecretBuf, Zeroize};
use tessera_compiler::ExecutionPlan;
use tessera_ir::{ClaimList, IntegerModel, LogitsFile, TokenFile};
use tessera_pcs::{WeightCommitmentPublic, WeightCommitmentState, WeightMatrix};
use tessera_protocol::{
    ColumnQueries, NiProof, ProverSeedShare, Round1, Round2Response, Round3Response, SeedCommit,
    SeedReveal,
};

use crate::io_error;

/// `tessera inspect`.
#[derive(Debug, Args)]
pub(crate) struct InspectArgs {
    /// The TCE artifact to inspect.
    #[arg(value_name = "FILE")]
    pub(crate) file: PathBuf,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn class_name(class: Class) -> &'static str {
    match class {
        Class::Public => "Public",
        Class::Protocol => "Protocol",
        Class::Private => "Private",
        Class::Secret => "Secret",
    }
}

fn limits_for(schema: &Schema) -> DecodeLimits {
    match schema.id == schema::INTEGER_MODEL.id {
        true => DecodeLimits::INTEGER_MODEL,
        false => DecodeLimits::MESSAGE,
    }
}

/// Peeks the schema id at bytes 6..8 of `path`. When the header is shorter than 8 bytes or its
/// schema id is not in [`schema::CATALOG`], returns `None`, and [`verify_and_read`] hands
/// [`envelope::verify`] [`schema::INTEGER_MODEL`] so RFC-0003's header checks run in offset order.
fn peek_schema(path: &Path) -> Result<Option<&'static Schema>, TesseraError> {
    let shown = path.display().to_string();
    let mut file = File::open(path).map_err(io_error(format!("cannot read {shown}")))?;
    let mut head = [0u8; 8];
    let mut filled = 0;
    while filled < head.len() {
        match file.read(&mut head[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(io_error(format!("cannot read {shown}"))(e)),
        }
    }
    if filled < head.len() {
        return Ok(None);
    }
    let id = SchemaId(u16::from_le_bytes([head[6], head[7]]));
    Ok(schema::by_id(id))
}

/// Verifies the envelope of `path` in constant memory and reads its bytes into a zeroizing buffer.
fn verify_and_read(
    path: &Path,
) -> Result<(&'static Schema, envelope::Header, SecretBuf<u8>), TesseraError> {
    let shown = path.display().to_string();
    let found = peek_schema(path)?;
    let expected = found.unwrap_or(&schema::INTEGER_MODEL);
    let label = found.map_or("catalog", |s| s.name);
    let limits = limits_for(expected);
    let file = File::open(path).map_err(io_error(format!("cannot read {shown}")))?;
    // OBS-6: the envelope reader clears its buffer; read-ahead would add an uncleared copy.
    let header = match envelope::verify(file, expected, limits) {
        Ok(header) => header,
        Err(ReadError::Invalid(error)) => {
            return Err(TesseraError::from(error)
                .with_source(format!("{shown} is not a valid {label} envelope")));
        }
        Err(ReadError::Io(error)) => return Err(io_error(format!("cannot read {shown}"))(error)),
    };
    let Some(schema) = found else {
        return Err(TesseraError::new(ErrorCode::CodecMagic)
            .with_source(format!("{shown} is not a valid catalog envelope")));
    };
    let total = (envelope::HEADER_LEN + envelope::DIGEST_LEN) as u64 + header.payload_len;
    let len = usize::try_from(total).map_err(|_| {
        TesseraError::new(ErrorCode::CodecLimit).with_source(format!("{shown} exceeds memory"))
    })?;
    let mut bytes = SecretBuf::filled(0u8, len);
    let mut file = File::open(path).map_err(io_error(format!("cannot read {shown}")))?;
    file.read_exact(&mut bytes)
        .map_err(io_error(format!("cannot read {shown}")))?;
    if file
        .read(&mut [0u8; 1])
        .map_err(io_error(format!("cannot read {shown}")))?
        != 0
    {
        return Err(TesseraError::new(ErrorCode::ResIo)
            .with_source(format!("cannot read {shown}: file grew while it was read")));
    }
    Ok((schema, header, bytes))
}

fn open<T: Decode>(schema: &Schema, bytes: &[u8], shown: &str) -> Result<T, TesseraError> {
    envelope::open(bytes, schema, limits_for(schema)).map_err(|error| {
        TesseraError::from(error)
            .with_source(format!("{shown} is not a valid {} envelope", schema.name))
    })
}

/// Runs `tessera inspect FILE`; prints a structured text view (or a JSON object with `--json`) to
/// stdout.
pub(crate) fn inspect(args: &InspectArgs, json: bool) -> Result<(), TesseraError> {
    let shown = args.file.display().to_string();
    let (schema, header, bytes) = verify_and_read(&args.file)?;
    let digest = hex(&bytes[bytes.len() - envelope::DIGEST_LEN..]);
    let (body_lines, artifact_json) = render_payload(schema, &bytes, &shown)?;

    eprintln!(
        "tessera inspect: {} (0x{:04x}) v{}",
        schema.name, schema.id.0, header.version
    );
    let output = match json {
        true => {
            let object = json!({
                "command": "inspect",
                "schema": schema.name,
                "schema_id": schema.id.0,
                "version": header.version,
                "class": class_name(schema.class),
                "payload_len": header.payload_len,
                "digest": digest,
                "artifact": artifact_json,
            });
            let pretty = serde_json::to_string_pretty(&object)
                .map_err(|error| TesseraError::new(ErrorCode::Internal).with_source(error))?;
            format!("{pretty}\n")
        }
        false => {
            let mut text = format!(
                "schema: {} (0x{:04x})\nversion: {}\nclass: {}\npayload_len: {}\ndigest: {digest}\n",
                schema.name,
                schema.id.0,
                header.version,
                class_name(schema.class),
                header.payload_len,
            );
            for line in body_lines {
                text.push_str(&line);
                text.push('\n');
            }
            text
        }
    };
    io::stdout()
        .lock()
        .write_all(output.as_bytes())
        .map_err(io_error("cannot write to stdout".to_owned()))?;
    Ok(())
}

fn render_payload(
    schema: &Schema,
    bytes: &[u8],
    shown: &str,
) -> Result<(Vec<String>, Value), TesseraError> {
    match schema.id {
        id if id == schema::INTEGER_MODEL.id => {
            let mut model: IntegerModel = open(schema, bytes, shown)?;
            let out = render_integer_model(&model);
            for c in &mut model.quant.channels {
                c.class.zeroize();
                c.multiplier.zeroize();
            }
            for w in &mut model.weights {
                w.data.zeroize();
            }
            Ok(out)
        }
        id if id == schema::CLAIM_LIST.id => {
            let claims: ClaimList = open(schema, bytes, shown)?;
            Ok(render_claim_list(&claims))
        }
        id if id == schema::EXECUTION_PLAN.id => {
            let plan: ExecutionPlan = open(schema, bytes, shown)?;
            Ok(render_execution_plan(&plan))
        }
        id if id == schema::WEIGHT_COMMITMENT_PUBLIC.id => {
            let public: WeightCommitmentPublic = open(schema, bytes, shown)?;
            Ok(render_weight_public(&public))
        }
        id if id == schema::WEIGHT_COMMITMENT_STATE.id => {
            let state = WeightCommitmentState::open(bytes).map_err(|error| {
                TesseraError::from(error)
                    .with_source(format!("{shown} is not a valid {} envelope", schema.name))
            })?;
            Ok(render_weight_state(&state))
        }
        id if id == schema::SEED_COMMIT.id => {
            let msg: SeedCommit = open(schema, bytes, shown)?;
            let session = hex(&msg.session_id);
            let commitment = hex(&msg.commitment);
            Ok((
                vec![
                    format!("session_id: {session}"),
                    format!("commitment: {commitment}"),
                ],
                json!({"session_id": session, "commitment": commitment}),
            ))
        }
        id if id == schema::PROVER_SEED_SHARE.id => {
            let msg: ProverSeedShare = open(schema, bytes, shown)?;
            let session = hex(&msg.session_id);
            let share = hex(&msg.share);
            Ok((
                vec![format!("session_id: {session}"), format!("share: {share}")],
                json!({"session_id": session, "share": share}),
            ))
        }
        id if id == schema::SEED_REVEAL.id => {
            let msg: SeedReveal = open(schema, bytes, shown)?;
            let session = hex(&msg.session_id);
            let share = hex(&msg.share);
            Ok((
                vec![format!("session_id: {session}"), format!("share: {share}")],
                json!({"session_id": session, "share": share}),
            ))
        }
        id if id == schema::ROUND1.id => {
            let msg: Round1 = open(schema, bytes, shown)?;
            Ok(render_round1(&msg))
        }
        id if id == schema::ROUND2_RESPONSE.id => {
            let msg: Round2Response = open(schema, bytes, shown)?;
            Ok(render_round2(&msg))
        }
        id if id == schema::COLUMN_QUERIES.id => {
            let msg: ColumnQueries = open(schema, bytes, shown)?;
            Ok(render_column_queries(&msg))
        }
        id if id == schema::ROUND3_RESPONSE.id => {
            let msg: Round3Response = open(schema, bytes, shown)?;
            Ok(render_round3(&msg))
        }
        id if id == schema::NI_PROOF.id => {
            let msg: NiProof = open(schema, bytes, shown)?;
            Ok(render_ni_proof(&msg))
        }
        id if id == schema::TOKEN_FILE.id => {
            let mut tokens: TokenFile = open(schema, bytes, shown)?;
            let out = render_token_file(&tokens);
            for seq in &mut tokens.sequences {
                seq.zeroize();
            }
            Ok(out)
        }
        id if id == schema::LOGITS_FILE.id => {
            let mut logits: LogitsFile = open(schema, bytes, shown)?;
            let out = render_logits_file(&logits);
            for row in &mut logits.positions {
                row.zeroize();
            }
            Ok(out)
        }
        _ => Ok((Vec::new(), json!({}))),
    }
}

/// Renders `IntegerModel` (`0x0101`, `Secret`): shapes and digests only, never weight entries or
/// channel parameters (05 §3).
fn render_integer_model(model: &IntegerModel) -> (Vec<String>, Value) {
    let a = &model.arch;
    let profile = format!("{:?}", model.profile);
    let family = format!("{:?}", a.family);
    let mut lines = vec![
        format!("profile: {profile}"),
        format!(
            "arch: {family} (n_layers={}, d_model={}, n_heads={}, n_kv_heads={}, d_head={}, d_ff={}, vocab={}, max_context={})",
            a.n_layers,
            a.d_model,
            a.n_heads,
            a.n_kv_heads,
            a.d_head,
            a.d_ff,
            a.vocab,
            a.max_context
        ),
        format!("channels: {}", model.quant.channels.len()),
    ];
    let mut channels_json = Vec::with_capacity(model.quant.channels.len());
    for c in &model.quant.channels {
        let ty = format!("{:?}", c.ty);
        let len = c.class.len().max(c.multiplier.len());
        let affine = !c.multiplier.is_empty();
        lines.push(format!(
            "  {ty} layer {}: channels={len} affine={affine}",
            c.layer
        ));
        channels_json.push(json!({
            "ty": ty,
            "layer": c.layer,
            "channels": len,
            "affine": affine,
        }));
    }
    lines.push(format!("weights: {}", model.weights.len()));
    let mut weights_json = Vec::with_capacity(model.weights.len());
    for w in &model.weights {
        let ty = format!("{:?}", w.ty);
        lines.push(format!("  {ty} layer {}: {}x{}", w.layer, w.rows, w.cols));
        weights_json.push(json!({
            "ty": ty,
            "layer": w.layer,
            "rows": w.rows,
            "cols": w.cols,
        }));
    }
    let selection = hex(&model.provenance.calibration.selection);
    let statistics = hex(&model.provenance.statistics);
    lines.push("provenance:".to_owned());
    lines.push(format!("  selection_digest: {selection}"));
    lines.push(format!("  statistics_digest: {statistics}"));
    let mut checkpoints_json = Vec::with_capacity(model.provenance.checkpoint.len());
    for (name, digest) in &model.provenance.checkpoint {
        let digest_hex = hex(digest);
        lines.push(format!("  checkpoint {}: {digest_hex}", name.as_str()));
        checkpoints_json.push(json!({
            "name": name.as_str(),
            "digest": digest_hex,
        }));
    }
    let object = json!({
        "profile": profile,
        "arch": {
            "family": family,
            "n_layers": a.n_layers,
            "d_model": a.d_model,
            "n_heads": a.n_heads,
            "n_kv_heads": a.n_kv_heads,
            "d_head": a.d_head,
            "d_ff": a.d_ff,
            "vocab": a.vocab,
            "max_context": a.max_context,
        },
        "channels": channels_json,
        "weights": weights_json,
        "provenance": {
            "checkpoint": checkpoints_json,
            "selection_digest": selection,
            "statistics_digest": statistics,
        },
    });
    (lines, object)
}

fn render_claim_list(claims: &ClaimList) -> (Vec<String>, Value) {
    let claim_digest = hex(&claims.digest());
    let a = &claims.arch;
    let profile = format!("{:?}", claims.profile);
    let family = format!("{:?}", a.family);
    let mut lines = vec![
        format!("claim_list_digest: {claim_digest}"),
        format!("protocol_version: {}", claims.protocol_version),
        format!("params: {}", claims.params.0.as_str()),
        format!("profile: {profile}"),
        format!(
            "arch: {family} (n_layers={}, d_model={}, n_heads={}, n_kv_heads={}, d_head={}, d_ff={}, vocab={}, max_context={})",
            a.n_layers,
            a.d_model,
            a.n_heads,
            a.n_kv_heads,
            a.d_head,
            a.d_ff,
            a.vocab,
            a.max_context
        ),
        format!("tables: {}", claims.tables.len()),
    ];
    let mut tables_json = Vec::with_capacity(claims.tables.len());
    for t in &claims.tables {
        let td = hex(&t.digest);
        lines.push(format!(
            "  table {}: log2_rows={} digest={td} kind={:?}",
            t.id.0, t.log2_rows, t.kind
        ));
        tables_json.push(json!({
            "id": t.id.0,
            "log2_rows": t.log2_rows,
            "digest": td,
            "kind": format!("{:?}", t.kind),
        }));
    }
    let object = json!({
        "claim_list_digest": claim_digest,
        "protocol_version": claims.protocol_version,
        "params": claims.params.0.as_str(),
        "profile": profile,
        "arch": {
            "family": family,
            "n_layers": a.n_layers,
            "d_model": a.d_model,
            "n_heads": a.n_heads,
            "n_kv_heads": a.n_kv_heads,
            "d_head": a.d_head,
            "d_ff": a.d_ff,
            "vocab": a.vocab,
            "max_context": a.max_context,
        },
        "tables": tables_json,
    });
    (lines, object)
}

fn render_execution_plan(plan: &ExecutionPlan) -> (Vec<String>, Value) {
    let (mut lines, mut claims_json) = render_claim_list(&plan.claims);
    lines.push(format!("tiles: {}", plan.tiles.len()));
    let mut tiles_json = Vec::with_capacity(plan.tiles.len());
    for (index, tile) in plan.tiles.iter().enumerate() {
        let shown = format!("{tile:?}");
        lines.push(format!("  tile {index}: {shown}"));
        tiles_json.push(shown);
    }
    if let Some(obj) = claims_json.as_object_mut() {
        obj.insert("tiles".to_owned(), json!(tiles_json));
    }
    (lines, claims_json)
}

fn render_weight_public(public: &WeightCommitmentPublic) -> (Vec<String>, Value) {
    let weights_root = hex(&public.weights_root());
    let l = public.layout;
    let mut lines = vec![
        format!("weights_root: {weights_root}"),
        format!("epoch: {}", public.epoch),
        format!(
            "layout: log2_k={} log2_inv_rate={} mask_coeffs={} blinding_rows={}",
            l.log2_k, l.log2_inv_rate, l.mask_coeffs, l.blinding_rows
        ),
        format!("matrices: {}", WeightMatrix::ALL.len()),
    ];
    let mut roots_json = Vec::with_capacity(WeightMatrix::ALL.len());
    for matrix in WeightMatrix::ALL {
        let name = format!("{matrix:?}");
        let root = hex(&public.root(matrix));
        lines.push(format!("  {name}: {root}"));
        roots_json.push(json!({"matrix": name, "root": root}));
    }
    let object = json!({
        "weights_root": weights_root,
        "epoch": public.epoch,
        "layout": {
            "log2_k": l.log2_k,
            "log2_inv_rate": l.log2_inv_rate,
            "mask_coeffs": l.mask_coeffs,
            "blinding_rows": l.blinding_rows,
        },
        "matrices": roots_json,
    });
    (lines, object)
}

/// Renders `WeightCommitmentState` (`0x0302`, `Secret`): public roots, layout shapes and budget
/// counts only, never `masks_seed` (05 §3).
fn render_weight_state(state: &WeightCommitmentState) -> (Vec<String>, Value) {
    let weights_root = hex(&state.public.weights_root());
    let l = state.public.layout;
    let mut lines = vec![
        format!("weights_root: {weights_root}"),
        format!("epoch: {}", state.public.epoch),
        format!(
            "layout: log2_k={} log2_inv_rate={} mask_coeffs={} blinding_rows={}",
            l.log2_k, l.log2_inv_rate, l.mask_coeffs, l.blinding_rows
        ),
        format!("matrices: {}", WeightMatrix::ALL.len()),
    ];
    let mut matrices_json = Vec::with_capacity(WeightMatrix::ALL.len());
    for matrix in WeightMatrix::ALL {
        let name = format!("{matrix:?}");
        let root = hex(&state.public.root(matrix));
        let budget = &state.budget[matrix.index()];
        let opened = budget.opened_columns.columns().len();
        lines.push(format!(
            "  {name}: root={root} opened_columns={opened} next_blinding_row={}",
            budget.next_blinding_row
        ));
        matrices_json.push(json!({
            "matrix": name,
            "root": root,
            "opened_columns": opened,
            "next_blinding_row": budget.next_blinding_row,
        }));
    }
    let object = json!({
        "weights_root": weights_root,
        "epoch": state.public.epoch,
        "layout": {
            "log2_k": l.log2_k,
            "log2_inv_rate": l.log2_inv_rate,
            "mask_coeffs": l.mask_coeffs,
            "blinding_rows": l.blinding_rows,
        },
        "matrices": matrices_json,
    });
    (lines, object)
}

fn render_round1(msg: &Round1) -> (Vec<String>, Value) {
    let s = &msg.statement;
    let session = hex(&msg.session_id);
    let claim_list_digest = hex(&s.claim_list_digest);
    let weights_root = hex(&s.weights_root);
    let manifest_digest = hex(&s.manifest_digest);
    let advice_root = hex(&msg.advice.root);
    let layout_digest = hex(&msg.advice.layout_digest);
    let mut lines = vec![
        format!("protocol_version: {}", msg.protocol_version),
        format!("session_id: {session}"),
        format!("claim_list_digest: {claim_list_digest}"),
        format!("weights_root: {weights_root}"),
        format!("weights_epoch: {}", s.weights_epoch),
        format!("manifest_digest: {manifest_digest}"),
        format!("requests: {}", s.batch.requests.len()),
    ];
    let mut requests_json = Vec::with_capacity(s.batch.requests.len());
    for req in &s.batch.requests {
        let req_session = hex(&req.session_id);
        lines.push(format!(
            "  request {}: session={req_session} prompt_len={} output_len={}",
            req.request_id, req.prompt_len, req.output_len
        ));
        requests_json.push(json!({
            "request_id": req.request_id,
            "session_id": req_session,
            "prompt_len": req.prompt_len,
            "output_len": req.output_len,
        }));
    }
    lines.push(format!("predictor: {:?}", s.predictor));
    lines.push(format!(
        "advice: root={advice_root} layout_digest={layout_digest}"
    ));
    let object = json!({
        "protocol_version": msg.protocol_version,
        "session_id": session,
        "claim_list_digest": claim_list_digest,
        "weights_root": weights_root,
        "weights_epoch": s.weights_epoch,
        "manifest_digest": manifest_digest,
        "requests": requests_json,
        "predictor": format!("{:?}", s.predictor),
        "advice": {
            "root": advice_root,
            "layout_digest": layout_digest,
        },
    });
    (lines, object)
}

fn render_round2(msg: &Round2Response) -> (Vec<String>, Value) {
    let weights: Vec<u8> = msg.weight_combined.iter().map(|w| w.matrix).collect();
    let lines = vec![
        format!("logup_roots: {}", msg.logup.roots.len()),
        format!("logup_values: {}", msg.logup.values.len()),
        format!("tree_levels: {}", msg.trees.levels.len()),
        format!("stages: {}", msg.stages.len()),
        format!("advice_combined: {}", msg.advice_combined.len()),
        format!(
            "weight_combined: {} ({:?})",
            msg.weight_combined.len(),
            weights
        ),
        format!("scoring_requests: {}", msg.scoring.requests.len()),
    ];
    let object = json!({
        "logup_roots": msg.logup.roots.len(),
        "logup_values": msg.logup.values.len(),
        "tree_levels": msg.trees.levels.len(),
        "stages": msg.stages.len(),
        "advice_combined": msg.advice_combined.len(),
        "weight_combined": weights,
        "scoring_requests": msg.scoring.requests.len(),
    });
    (lines, object)
}

fn render_column_queries(msg: &ColumnQueries) -> (Vec<String>, Value) {
    let mut lines = vec![
        format!("advice_queries: {}", msg.advice.len()),
        format!("weight_queries: {}", msg.weights.len()),
    ];
    let mut weights_json = Vec::with_capacity(msg.weights.len());
    for w in &msg.weights {
        lines.push(format!("  matrix {}: {}", w.matrix, w.indices.len()));
        weights_json.push(json!({
            "matrix": w.matrix,
            "indices": w.indices.len(),
        }));
    }
    let object = json!({
        "advice": msg.advice.len(),
        "weights": weights_json,
    });
    (lines, object)
}

fn render_round3(msg: &Round3Response) -> (Vec<String>, Value) {
    let weights: Vec<u8> = msg.weight_columns.iter().map(|w| w.matrix).collect();
    let lines = vec![
        format!("advice_columns: {}", msg.advice_columns.len()),
        format!("advice_proof_siblings: {}", msg.advice_proof.siblings.len()),
        format!(
            "weight_columns: {} ({:?})",
            msg.weight_columns.len(),
            weights
        ),
    ];
    let object = json!({
        "advice_columns": msg.advice_columns.len(),
        "advice_proof_siblings": msg.advice_proof.siblings.len(),
        "weight_columns": weights,
    });
    (lines, object)
}

fn render_ni_proof(msg: &NiProof) -> (Vec<String>, Value) {
    let (r1_lines, r1_json) = render_round1(&msg.round1);
    let (r2_lines, r2_json) = render_round2(&msg.round2);
    let (r3_lines, r3_json) = render_round3(&msg.round3);
    let mut lines = vec![
        format!("seed_nonce: {}", msg.seed_nonce),
        format!("query_nonce: {}", msg.query_nonce),
        "round1:".to_owned(),
    ];
    for l in r1_lines {
        lines.push(format!("  {l}"));
    }
    lines.push("round2:".to_owned());
    for l in r2_lines {
        lines.push(format!("  {l}"));
    }
    lines.push("round3:".to_owned());
    for l in r3_lines {
        lines.push(format!("  {l}"));
    }
    let object = json!({
        "seed_nonce": msg.seed_nonce,
        "query_nonce": msg.query_nonce,
        "round1": r1_json,
        "round2": r2_json,
        "round3": r3_json,
    });
    (lines, object)
}

/// Renders `TokenFile` (`0x0601`, `Secret`): sequence count and sequence lengths only, never
/// token ids (05 §3).
fn render_token_file(tokens: &TokenFile) -> (Vec<String>, Value) {
    let lengths: Vec<usize> = tokens.sequences.iter().map(Vec::len).collect();
    let total: usize = lengths.iter().sum();
    let mut lines = vec![
        format!("sequences: {}", lengths.len()),
        format!("total_tokens: {total}"),
    ];
    for (index, len) in lengths.iter().enumerate() {
        lines.push(format!("  sequence {index}: len={len}"));
    }
    let object = json!({
        "sequences": lengths.len(),
        "total_tokens": total,
        "lengths": lengths,
    });
    (lines, object)
}

/// Renders `LogitsFile` (`0x0602`, `Secret`): vocabulary size and position counts only, never
/// logit values (05 §3).
fn render_logits_file(logits: &LogitsFile) -> (Vec<String>, Value) {
    let row_lengths: Vec<usize> = logits.positions.iter().map(Vec::len).collect();
    let lines = vec![
        format!("vocab: {}", logits.vocab),
        format!("positions: {}", logits.positions.len()),
    ];
    let object = json!({
        "vocab": logits.vocab,
        "positions": logits.positions.len(),
        "row_lengths": row_lengths,
    });
    (lines, object)
}
