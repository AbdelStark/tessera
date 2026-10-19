// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The claim list's emission (RFC-0007 'Schema', RFC-0026) on the conformance models.
//!
//! - **INV-CL-1.** Weights and channel classes changed while the public part stays leave the
//!   claim list's bytes unchanged: no field depends on a hidden value, and no class assignment
//!   appears.
//! - **One claim list per model.** Lowerings of one token and of the whole context give the same
//!   claim list: the tables do not depend on the sequence's length.
//! - **Canonical.** It decodes from its encoding and from its envelope to itself.
//! - **Each layer's `Silu` table** (RFC-0042). Two models whose layers share their `Silu` tables
//!   differently list the same tables. Their claim lists still differ, and each lowers, from its
//!   own fields alone, to its model's graph, as every conformance model's does.

mod common;

use std::fs;
use std::path::Path;

use common::llama;
use tessera_codec::{Ascii, DecodeLimits, decode, encode, envelope, schema};
use tessera_compiler::{PublicModel, analyze, claim_list, lower, select};
use tessera_ir::{
    AdviceLayoutParams, ClaimList, CommitmentLayout, GadgetPolicy, IntegerModel, ParamSetId,
    Profile, WeightLayoutParams,
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
    of_public(PublicModel::of(model), positions)
}

/// The claim list of the public part `public`, lowered over `positions` tokens.
fn of_public(public: PublicModel<'_>, positions: u32) -> ClaimList {
    let lowering = lower(public, positions).expect("the model lowers");
    let ranges = analyze(&lowering.graph, &lowering.tables).expect("ranges");
    let policy = GadgetPolicy::default();
    let selection =
        select(&lowering.graph, &ranges, &lowering.tables, &policy).expect("every gadget fits");
    let params = ParamSetId(Ascii::new("test-small/1").expect("ASCII"));
    claim_list(public, &selection, &policy, TEST_SMALL, params)
        .expect("the maximum advice layout fits")
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
fn a_claim_list_lowers_to_its_models_graph() {
    for (name, model) in models() {
        let claims = emitted(&model, 1);
        for n in [1, 5, model.arch.max_context] {
            let from_claims = lower(PublicModel::of_claims(&claims), n).expect("it lowers");
            let from_model = lower(PublicModel::of(&model), n).expect("the model lowers");
            assert_eq!(from_claims.graph, from_model.graph, "{name}, n = {n}");
            assert_eq!(from_claims.tables, from_model.tables, "{name}, n = {n}");
        }
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

#[test]
fn the_claim_list_names_each_layers_silu_exponent() {
    // Three layers alike but for their Silu input exponents: (a, b, a) shares layer 0's table at
    // layer 2, and (a, b, b) layer 1's.
    let (mut arch, quant, silu) = llama(Profile::Servable, 3, 256, 2, 1, 512);
    // A short context keeps the RoPE table small.
    arch.vocab = 300;
    arch.max_context = 16;
    let (a, b) = (silu[0].1, silu[0].1 - 1);
    let claims = |exponents: [i8; 3]| {
        let silu: Vec<(u32, i8)> = (0..3).zip(exponents).collect();
        let public = PublicModel {
            arch: &arch,
            profile: Profile::Servable,
            quant: &quant,
            silu_in_exponents: &silu,
        };
        (of_public(public, 4), silu)
    };
    let (shared_first, first) = claims([a, b, a]);
    let (shared_second, second) = claims([a, b, b]);
    // The tables cannot tell the two apart (#358), so the claim list names the exponents.
    assert_eq!(shared_first.tables, shared_second.tables);
    assert_ne!(encode(&shared_first), encode(&shared_second));
    for (claims, silu) in [(&shared_first, &first), (&shared_second, &second)] {
        assert_eq!(&claims.silu_in_exponents, silu);
        // From the claim list alone, as the verifier lowers it, the model's graph and tables.
        let from_claims = lower(PublicModel::of_claims(claims), 4).expect("the claim list lowers");
        let public = PublicModel {
            arch: &arch,
            profile: Profile::Servable,
            quant: &quant,
            silu_in_exponents: silu,
        };
        let from_model = lower(public, 4).expect("the model lowers");
        assert_eq!(from_claims.graph, from_model.graph);
        assert_eq!(from_claims.tables, from_model.tables);
    }
}
