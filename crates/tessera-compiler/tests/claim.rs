// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The claim list's emission (RFC-0007 'Schema', RFC-0026) on the conformance models.
//!
//! - **INV-CL-1.** Weights and channel classes changed while the public part stays leave the
//!   claim list's bytes unchanged: no field depends on a hidden value, and no class assignment
//!   appears.
//! - **One claim list per model.** Lowerings of one token and of the whole context give the same
//!   claim list: the tables do not depend on the sequence's length.
//! - **Canonical.** It decodes from its encoding and from its envelope to itself.

use std::fs;
use std::path::Path;

use tessera_codec::{Ascii, DecodeLimits, decode, encode, envelope, schema};
use tessera_compiler::{PublicModel, analyze, claim_list, lower, select};
use tessera_ir::{
    AdviceLayoutParams, ClaimList, CommitmentLayout, GadgetPolicy, IntegerModel, ParamSetId,
    WeightLayoutParams,
};

/// `test-small/1`'s layout (RFC-0014 'Parameter sets').
const TEST_SMALL: CommitmentLayout = CommitmentLayout {
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

fn models() -> Vec<(String, IntegerModel)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/models");
    let text = fs::read_to_string(dir.join("models.json")).expect("the index");
    let index: serde_json::Value = serde_json::from_str(&text).expect("JSON");
    index["models"]
        .as_array()
        .expect("models")
        .iter()
        .map(|entry| {
            let file = entry["file"].as_str().expect("a file");
            let bytes = fs::read(dir.join(file)).expect("the artifact");
            let model = envelope::open(&bytes, &schema::INTEGER_MODEL, DecodeLimits::INTEGER_MODEL)
                .expect("an IntegerModel");
            (entry["name"].as_str().expect("a name").to_owned(), model)
        })
        .collect()
}

/// The claim list of `model`, lowered over `positions` tokens, for `test-small/1`.
fn emitted(model: &IntegerModel, positions: u32) -> ClaimList {
    let public = PublicModel::of(model);
    let lowering = lower(public, positions).expect("the model lowers");
    let ranges = analyze(&lowering.graph, &lowering.tables).expect("ranges");
    let policy = GadgetPolicy::default();
    let selection =
        select(&lowering.graph, &ranges, &lowering.tables, &policy).expect("every gadget fits");
    let params = ParamSetId(Ascii::new("test-small/1").expect("ASCII"));
    claim_list(public, &selection, &policy, TEST_SMALL, params)
}

#[test]
fn the_claim_list_reads_no_weight_and_no_channel_class() {
    for (name, model) in models() {
        let claims = encode(&emitted(&model, 4));
        let mut hidden = model.clone();
        for weight in &mut hidden.weights {
            for w in &mut weight.data {
                *w = w.saturating_neg().clamp(-127, 127) / 2;
            }
        }
        assert!(
            !hidden.quant.channels.is_empty(),
            "{name} has classed sites"
        );
        for params in &mut hidden.quant.channels {
            params.class.reverse();
        }
        assert_eq!(encode(&emitted(&hidden, 4)), claims, "{name}");
    }
}

#[test]
fn one_claim_list_serves_every_sequence_length() {
    for (name, model) in models() {
        let whole = emitted(&model, model.arch.max_context);
        assert_eq!(emitted(&model, 1), whole, "{name}");
    }
}

#[test]
fn the_claim_list_is_canonical() {
    for (name, model) in models() {
        let claims = emitted(&model, 1);
        let bytes = encode(&claims);
        let decoded: ClaimList = decode(&bytes, DecodeLimits::MESSAGE).expect("a claim list");
        assert_eq!(decoded, claims, "{name}");
        let sealed = envelope::seal(&schema::CLAIM_LIST, &claims);
        let opened: ClaimList = envelope::open(&sealed, &schema::CLAIM_LIST, DecodeLimits::MESSAGE)
            .expect("its envelope");
        assert_eq!(opened, claims, "{name}");
        // The tables are the selection's, in id order.
        let ids: Vec<u32> = claims.tables.iter().map(|t| t.id.0).collect();
        assert_eq!(
            ids,
            (0..u32::try_from(ids.len()).expect("few")).collect::<Vec<_>>()
        );
    }
}
