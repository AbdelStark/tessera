// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The golden claim lists (RFC-0007 'Testing Strategy', RFC-0026): each conformance model's claim
//! list for `test-small/1`, sealed in its envelope, with its digest. The compiler emits them from
//! the models `spec/vectors/models/generate.py` writes, so they are written after that oracle
//! runs. The verifier decodes them with its claim-list validation (#107).

use std::fs;
use std::path::Path;

use serde_json::{Value, json};
use tessera_codec::{Ascii, DecodeLimits, envelope, schema};
use tessera_compiler::{PublicModel, analyze, claim_list, lower, select};
use tessera_ir::{
    AdviceLayoutParams, CommitmentLayout, GadgetPolicy, IntegerModel, ParamSetId,
    WeightLayoutParams,
};

/// Where the claim lists are written.
pub(crate) const PATH: &str = "spec/vectors/claims/claim-lists.json";

/// The conformance models' index, which the claim lists follow.
pub(crate) const MODELS: &str = "spec/vectors/models/models.json";

/// `test-small/1`, the parameter set of the conformance models' claim lists.
const PARAMS: &str = "test-small/1";

/// `test-small/1`'s layout (RFC-0014 'Parameter sets').
const LAYOUT: CommitmentLayout = CommitmentLayout {
    advice: AdviceLayoutParams {
        log2_k: 10,
        log2_inv_rate: 1,
        mask_coeffs: 16,
        row_block: 4,
        segments: 1,
    },
    weights: WeightLayoutParams {
        log2_k: 8,
        log2_inv_rate: 2,
        mask_coeffs: 64,
        blinding_rows: 32,
    },
};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The claim lists of the conformance models in the tree at `repo`, as the file's text.
pub(crate) fn render(repo: &Path) -> Result<String, String> {
    let dir = repo.join("spec/vectors/models");
    let index = fs::read_to_string(repo.join(MODELS)).map_err(|e| format!("{MODELS}: {e}"))?;
    let index: Value = serde_json::from_str(&index).map_err(|e| format!("models.json: {e}"))?;
    let entries = index["models"]
        .as_array()
        .ok_or("models.json lists no models")?;
    let mut cases = Vec::new();
    for entry in entries {
        let name = entry["name"].as_str().ok_or("a model without a name")?;
        let file = entry["file"].as_str().ok_or("a model without a file")?;
        let bytes = fs::read(dir.join(file)).map_err(|e| format!("{file}: {e}"))?;
        let model: IntegerModel =
            envelope::open(&bytes, &schema::INTEGER_MODEL, DecodeLimits::INTEGER_MODEL)
                .map_err(|e| format!("{file}: {e}"))?;
        let public = PublicModel::of(&model);
        // One token: every length gives the same tables, as the compiler's tests check.
        let lowering = lower(public, 1).map_err(|e| format!("{name}: {e}"))?;
        let ranges =
            analyze(&lowering.graph, &lowering.tables).map_err(|e| format!("{name}: {e}"))?;
        let policy = GadgetPolicy::default();
        let selection = select(&lowering.graph, &ranges, &lowering.tables, &policy)
            .map_err(|e| format!("{name}: {e}"))?;
        let params = ParamSetId(Ascii::new(PARAMS).map_err(|_| "an ASCII parameter set")?);
        let claims = claim_list(public, &selection, &policy, LAYOUT, params);
        let sealed = envelope::seal(&schema::CLAIM_LIST, &claims);
        cases.push(json!({
            "name": name,
            "model": file,
            "tables": claims.tables.len(),
            "hex": hex(&sealed),
            "digest": hex(&claims.digest()),
        }));
    }
    let file = json!({
        "description": "Each conformance model's claim list (RFC-0007, RFC-0026) for the parameter set test-small/1, written by `cargo xtask gen-vectors` from the compiler's emission: `hex` is the sealed ClaimList envelope, `digest` its claim_list_digest = H(\"claim-list\", TCE(ClaimList)).",
        "schema": schema::CLAIM_LIST.id.0,
        "version": schema::CLAIM_LIST.version,
        "params": PARAMS,
        "cases": cases,
    });
    let mut text = serde_json::to_string_pretty(&file).map_err(|e| e.to_string())?;
    text.push('\n');
    Ok(text)
}
