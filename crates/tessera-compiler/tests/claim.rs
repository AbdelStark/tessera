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
use tessera_codec::error::ErrorCode;
use tessera_codec::{Ascii, DecodeLimits, decode, encode, envelope, schema};
use tessera_compiler::{
    CompileError, PublicModel, analyze, claim_list, claim_list_for_protocol, compile_for_protocol,
    lower, plan, select,
};
use tessera_ir::{
    AdviceLayoutParams, ClaimList, CommitmentLayout, GadgetPolicy, IntegerModel, PROTOCOL_VERSION,
    PROTOCOL_VERSION_2, ParamSetId, Profile, QuantPoint, Site, WeightLayoutParams,
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

/// RFC-0012 'Layout' ('Masked rate and minimum width in `tessera-protocol/2`', #564):
/// `claim_list_for_protocol`, `compile_for_protocol` and `plan` admit all four conformance models
/// in `tessera-protocol/2`, refuse narrow committed weight matrices (`ERR-QUANT-ARCH`) while
/// `tessera-protocol/1` still admits them, exempt uncommitted `ChannelParams` (`m_CP == 0`), and
/// refuse malformed protocol versions or weight layouts (`ERR-USAGE`).
#[test]
fn protocol_2_refuses_narrow_committed_weight_matrices() {
    let policy = GadgetPolicy::default();
    let test_small_id = ParamSetId(Ascii::new("test-small/1").expect("ASCII"));

    // All four conformance models satisfy k_T >= k_W = 256 on every committed matrix at test-small/1.
    for (name, model) in models() {
        let out = compile_for_protocol(
            PROTOCOL_VERSION_2,
            &model,
            &policy,
            TEST_SMALL,
            test_small_id.clone(),
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            out.claim_list.protocol_version, PROTOCOL_VERSION_2,
            "{name}"
        );
        assert_eq!(out.plan.claims, out.claim_list, "{name}");
    }

    // #564's exact forgery-regime architecture: d_model = 128, 2 heads of 64, d_ff = 512 at
    // live-64/1 (`k_W = 2^20`, `P_W = 2^16`), where Q has k_Q = 2^14 and n_Q = 2^16 = P_W.
    let live_64 = CommitmentLayout {
        advice: AdviceLayoutParams {
            log2_k: 22,
            log2_inv_rate: 1,
            mask_coeffs: 160,
            row_block: 32,
            segments: 8,
        },
        weights: WeightLayoutParams {
            log2_k: 20,
            log2_inv_rate: 2,
            mask_coeffs: 65_536,
            blinding_rows: 4096,
        },
    };
    let live_64_id = ParamSetId(Ascii::new("live-64/1").expect("ASCII"));
    let (mut narrow_arch, narrow_quant, narrow_silu) = llama(Profile::Servable, 1, 128, 2, 2, 512);
    narrow_arch.d_head = 64;
    narrow_arch.vocab = 1024;
    narrow_arch.max_context = 16;
    let narrow_public = PublicModel {
        arch: &narrow_arch,
        profile: Profile::Servable,
        quant: &narrow_quant,
        silu_in_exponents: &narrow_silu,
    };
    let narrow_lowering = lower(narrow_public, 4).expect("lowers");
    let narrow_ranges = analyze(&narrow_lowering.graph, &narrow_lowering.tables).expect("ranges");
    let narrow_selection = select(
        &narrow_lowering.graph,
        &narrow_ranges,
        &narrow_lowering.tables,
        &policy,
    )
    .expect("gadgets");

    // tessera-protocol/1 (`PROTOCOL_VERSION`) has P = 0 and continues to admit narrow matrices.
    let v1_claims = claim_list(
        narrow_public,
        &narrow_selection,
        &policy,
        live_64,
        live_64_id.clone(),
    )
    .expect("protocol/1 admits narrow matrices");
    assert_eq!(v1_claims.protocol_version, PROTOCOL_VERSION);

    // tessera-protocol/2 refuses it (`ERR-QUANT-ARCH`), and `plan` also refuses a forged claim list
    // whose protocol_version was flipped to PROTOCOL_VERSION_2.
    let err = claim_list_for_protocol(
        PROTOCOL_VERSION_2,
        narrow_public,
        &narrow_selection,
        &policy,
        live_64,
        live_64_id,
    )
    .expect_err("protocol/2 refuses k_Q = 2^14 < k_W = 2^20");
    assert!(matches!(err, CompileError::Model { .. }), "{err:?}");
    assert_eq!(err.code(), ErrorCode::QuantArch);

    let mut forged = v1_claims;
    forged.protocol_version = PROTOCOL_VERSION_2;
    let plan_err =
        plan(narrow_public, &forged).expect_err("plan refuses forged narrow protocol/2 claims");
    assert!(
        matches!(plan_err, CompileError::Model { .. }),
        "{plan_err:?}"
    );
    assert_eq!(plan_err.code(), ErrorCode::QuantArch);

    // Narrow K/V alone at `k_W = 4096` (`log2_k = 12`): `d_model = 128`, 8 query heads, 1 KV head
    // of `d_head = 16` -> `K, V` are `16 × 128 = 2048 < 4096`, while `Embed`, `Q`, `O`, `Gate`,
    // `Up`, `Down`, `LmHead` are ≥ 16,384.
    let k_4096 = CommitmentLayout {
        weights: WeightLayoutParams {
            log2_k: 12,
            mask_coeffs: 1024,
            ..TEST_SMALL.weights
        },
        ..TEST_SMALL
    };
    let (mut kv_arch, kv_quant, kv_silu) = llama(Profile::Servable, 1, 128, 8, 1, 256);
    kv_arch.d_head = 16;
    kv_arch.vocab = 256;
    kv_arch.max_context = 16;
    let kv_public = PublicModel {
        arch: &kv_arch,
        profile: Profile::Servable,
        quant: &kv_quant,
        silu_in_exponents: &kv_silu,
    };
    let kv_low = lower(kv_public, 4).expect("lowers");
    let kv_sel = select(
        &kv_low.graph,
        &analyze(&kv_low.graph, &kv_low.tables).expect("ranges"),
        &kv_low.tables,
        &policy,
    )
    .expect("gadgets");
    let kv_err = claim_list_for_protocol(
        PROTOCOL_VERSION_2,
        kv_public,
        &kv_sel,
        &policy,
        k_4096,
        test_small_id.clone(),
    )
    .expect_err("narrow K/V refused");
    assert_eq!(kv_err.code(), ErrorCode::QuantArch);

    // ChannelParams isolation at `k_W = 16,384` (`log2_k = 14`): 1 layer, `d_model = d_ff =
    // vocab = 128` (`2` heads of `64`), so all nine weight matrices have `128 × 128 = 16,384 =
    // k_W`.
    // - Without classed/affine sites (`m_CP == 0`, `k_CP = 16 < 16,384`), `ChannelParams` is
    //   exempt.
    // - With 2 classes on `RopeQ` (`m_CP > 0`, `C = 128`, `k_CP = 16 · 1 · 128 = 2048 < 16,384`),
    //   `ChannelParams` is refused (`ERR-QUANT-ARCH`).
    let k_16384 = CommitmentLayout {
        weights: WeightLayoutParams {
            log2_k: 14,
            mask_coeffs: 1024,
            ..TEST_SMALL.weights
        },
        ..TEST_SMALL
    };
    let (mut cp_arch, cp_unclassed_quant, cp_silu) = llama(Profile::Servable, 1, 128, 2, 2, 128);
    cp_arch.d_head = 64;
    cp_arch.vocab = 128;
    cp_arch.max_context = 16;
    let unclassed_public = PublicModel {
        arch: &cp_arch,
        profile: Profile::Servable,
        quant: &cp_unclassed_quant,
        silu_in_exponents: &cp_silu,
    };
    let unclassed_low = lower(unclassed_public, 4).expect("lowers");
    let unclassed_sel = select(
        &unclassed_low.graph,
        &analyze(&unclassed_low.graph, &unclassed_low.tables).expect("ranges"),
        &unclassed_low.tables,
        &policy,
    )
    .expect("gadgets");
    let unclassed_claims = claim_list_for_protocol(
        PROTOCOL_VERSION_2,
        unclassed_public,
        &unclassed_sel,
        &policy,
        k_16384,
        test_small_id.clone(),
    )
    .expect("m_CP == 0 is exempt even when 16 · L_pad · C < k_W");
    assert!(plan(unclassed_public, &unclassed_claims).is_ok());

    let mut cp_classed_quant = cp_unclassed_quant.clone();
    for (point, shifts) in &mut cp_classed_quant.shift_class_sets {
        if *point
            == (QuantPoint::Layer {
                layer: 0,
                site: Site::RopeQ,
            })
        {
            let first = shifts[0];
            shifts.push(first + 1);
        }
    }
    let classed_public = PublicModel {
        arch: &cp_arch,
        profile: Profile::Servable,
        quant: &cp_classed_quant,
        silu_in_exponents: &cp_silu,
    };
    let classed_low = lower(classed_public, 4).expect("lowers");
    let classed_sel = select(
        &classed_low.graph,
        &analyze(&classed_low.graph, &classed_low.tables).expect("ranges"),
        &classed_low.tables,
        &policy,
    )
    .expect("gadgets");
    let cp_err = claim_list_for_protocol(
        PROTOCOL_VERSION_2,
        classed_public,
        &classed_sel,
        &policy,
        k_16384,
        test_small_id.clone(),
    )
    .expect_err("committed ChannelParams with k_CP = 2048 < k_W = 16,384 refused");
    assert_eq!(cp_err.code(), ErrorCode::QuantArch);

    // An affine site (`maximum.multipliers`, `m_CP > 0`) on the same 128 × 128 shape is refused by
    // the exact same ChannelParams width rule.
    let mut cp_affine_quant = cp_unclassed_quant.clone();
    let rope_q = QuantPoint::Layer {
        layer: 0,
        site: Site::RopeQ,
    };
    let shift = cp_affine_quant
        .shift_class_sets
        .iter()
        .find(|&&(p, _)| p == rope_q)
        .expect("RopeQ")
        .1[0];
    cp_affine_quant
        .shift_class_sets
        .retain(|&(p, _)| p != rope_q);
    cp_affine_quant.affine_sites.push((rope_q, shift));
    let affine_public = PublicModel {
        arch: &cp_arch,
        profile: Profile::Servable,
        quant: &cp_affine_quant,
        silu_in_exponents: &cp_silu,
    };
    let affine_low = lower(affine_public, 4).expect("lowers");
    let affine_sel = select(
        &affine_low.graph,
        &analyze(&affine_low.graph, &affine_low.tables).expect("ranges"),
        &affine_low.tables,
        &policy,
    )
    .expect("gadgets");
    let affine_cp_err = claim_list_for_protocol(
        PROTOCOL_VERSION_2,
        affine_public,
        &affine_sel,
        &policy,
        k_16384,
        test_small_id.clone(),
    )
    .expect_err("affine ChannelParams with k_CP = 2048 < k_W = 16,384 refused");
    assert_eq!(affine_cp_err.code(), ErrorCode::QuantArch);

    // An oversized input channel (`d_ff = 2^30 + 1` in `Profile::Minimal`, `in_pad = 2^31`,
    // `n_Down = 2^33 > 2^32`) under a valid parameter-set `layout.weights` (`live-64/1`) is a
    // model architecture error (`ERR-QUANT-ARCH`), not `ERR-USAGE`, in both protocol versions.
    let (mut huge_ff_arch, mut huge_ff_quant, huge_ff_silu) =
        llama(Profile::Minimal, 1, 2, 1, 1, (1 << 30) + 1);
    huge_ff_arch.d_head = 2;
    huge_ff_arch.vocab = 2;
    huge_ff_arch.max_context = 1;
    for (point, shifts) in &mut huge_ff_quant.shift_class_sets {
        match *point {
            QuantPoint::Layer {
                layer: 0,
                site: Site::Silu,
            } => *shifts = vec![10],
            QuantPoint::Layer {
                layer: 0,
                site: Site::MlpResidual,
            } => *shifts = vec![41],
            _ => {}
        }
    }
    let huge_ff_public = PublicModel {
        arch: &huge_ff_arch,
        profile: Profile::Minimal,
        quant: &huge_ff_quant,
        silu_in_exponents: &huge_ff_silu,
    };
    let huge_ff_low = lower(huge_ff_public, 1).expect("minimal profile lowers wide d_ff");
    let huge_ff_sel = select(
        &huge_ff_low.graph,
        &analyze(&huge_ff_low.graph, &huge_ff_low.tables).expect("ranges"),
        &huge_ff_low.tables,
        &policy,
    )
    .expect("split gadget fits");
    let live_64_id_2 = ParamSetId(Ascii::new("live-64/1").expect("ASCII"));
    for version in [PROTOCOL_VERSION, PROTOCOL_VERSION_2] {
        let err = claim_list_for_protocol(
            version,
            huge_ff_public,
            &huge_ff_sel,
            &policy,
            live_64,
            live_64_id_2.clone(),
        )
        .expect_err("n_Down = 2^33 > 2^32 refused");
        assert!(
            matches!(err, CompileError::Model { .. }),
            "{version}: {err:?}"
        );
        assert_eq!(err.code(), ErrorCode::QuantArch, "version {version}");
    }

    // Malformed protocol version or malformed masked weight layout parameters yield ERR-USAGE.
    for bad_version in [1u16, 3, u16::MAX] {
        let err = claim_list_for_protocol(
            bad_version,
            unclassed_public,
            &unclassed_sel,
            &policy,
            TEST_SMALL,
            test_small_id.clone(),
        )
        .expect_err("unknown protocol_version refused");
        assert_eq!(err.code(), ErrorCode::Usage);
    }
    for version in [PROTOCOL_VERSION, PROTOCOL_VERSION_2] {
        for bad_weights in [
            WeightLayoutParams {
                log2_inv_rate: 0,
                ..TEST_SMALL.weights
            },
            WeightLayoutParams {
                log2_k: 31,
                log2_inv_rate: 2,
                ..TEST_SMALL.weights
            },
        ] {
            let err = claim_list_for_protocol(
                version,
                unclassed_public,
                &unclassed_sel,
                &policy,
                CommitmentLayout {
                    weights: bad_weights,
                    ..TEST_SMALL
                },
                test_small_id.clone(),
            )
            .expect_err("bad nominal WeightLayoutParams refused");
            assert_eq!(err.code(), ErrorCode::Usage, "version {version}");
        }
    }
    for bad_weights in [
        WeightLayoutParams {
            mask_coeffs: 0,
            ..TEST_SMALL.weights
        },
        WeightLayoutParams {
            mask_coeffs: 768,
            ..TEST_SMALL.weights
        },
    ] {
        let err = claim_list_for_protocol(
            PROTOCOL_VERSION_2,
            unclassed_public,
            &unclassed_sel,
            &policy,
            CommitmentLayout {
                weights: bad_weights,
                ..TEST_SMALL
            },
            test_small_id.clone(),
        )
        .expect_err("bad masked WeightLayoutParams refused");
        assert_eq!(err.code(), ErrorCode::Usage);
    }
}
