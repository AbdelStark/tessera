// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The golden execution plans (RFC-0041 'The plan', schema `0x0202`), the codec vectors'
//! `spec/vectors/codec/execution-plan.json`: each conformance model's plan for `test-small/1` under
//! the default gadget policy, sealed in its envelope.
//!
//! Each case shows the plan's value as the codec vectors do (`spec/vectors/codec/README.md`,
//! 'Values'): its tiles, and its claim list by digest, since the claim list is the golden one that
//! `spec/vectors/claims/` holds for the same model. The compiler derives the plans from the models
//! `spec/vectors/models/generate.py` writes, so they are written after that oracle, with the claim
//! lists. The verifier never reads a plan; the compiler's tests decode them.

use std::path::Path;

use serde_json::{Value, json};
use tessera_codec::{envelope, schema};
use tessera_compiler::{PublicModel, Tile, plan};

use crate::claim_vectors::{PARAMS, claims, hex, models};

/// Where the plans are written.
pub(crate) const PATH: &str = "spec/vectors/codec/execution-plan.json";

/// A tile as the codec vectors write an enum value: `{"Variant": fields}`, integers in decimal.
fn tile(tile: &Tile) -> Value {
    let number = |n: u32| Value::String(n.to_string());
    match *tile {
        Tile::Tokens => json!({ "Tokens": null }),
        Tile::Node { node } => json!({ "Node": { "node": number(node) } }),
        Tile::Attention { node, kv_head } => {
            json!({ "Attention": { "node": number(node), "kv_head": number(kv_head) } })
        }
        Tile::Table { table } => json!({ "Table": { "table": number(table) } }),
    }
}

/// The plans of the conformance models in the tree at `repo`, as the file's text.
pub(crate) fn render(repo: &Path) -> Result<String, String> {
    let mut cases = Vec::new();
    for (name, file, model) in models(repo)? {
        let claims = claims(&name, &model)?;
        let plan = plan(PublicModel::of(&model), &claims).map_err(|e| format!("{name}: {e}"))?;
        let sealed = envelope::seal(&schema::EXECUTION_PLAN, &plan);
        cases.push(json!({
            "name": name,
            "model": file,
            "claim_list_digest": hex(&plan.claims.digest()),
            "tiles": plan.tiles.iter().map(tile).collect::<Vec<_>>(),
            "hex": hex(&sealed),
        }));
    }
    let file = json!({
        "description": "Each conformance model's execution plan (RFC-0041, schema 0x0202) for the parameter set test-small/1 under the default gadget policy, written by `cargo xtask gen-vectors` from the compiler's derivation. `hex` is the sealed ExecutionPlan envelope, read under the message limits of RFC-0003. Its payload decodes to a plan whose `claims` is the same model's claim list in spec/vectors/claims/, with the digest `claim_list_digest` = H(\"claim-list\", TCE(ClaimList)), and whose `tiles` are `tiles`, shown as the codec vectors show values. Every proper prefix of the payload fails with ERR-CODEC-TRUNCATED.",
        "schema": schema::EXECUTION_PLAN.id.0,
        "version": schema::EXECUTION_PLAN.version,
        "type": "ExecutionPlan",
        "limits": "message",
        "params": PARAMS,
        "cases": cases,
    });
    let mut text = serde_json::to_string_pretty(&file).map_err(|e| e.to_string())?;
    text.push('\n');
    Ok(text)
}
