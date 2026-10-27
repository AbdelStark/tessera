// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Claim-list validation, RFC-0007 'Verifier validation' in RFC-0039's order: every golden list
//! of `spec/vectors/claims/` validates, and each claim list below breaks exactly one rule and is
//! rejected with exactly the payload RFC-0039 and RFC-0043 name for its check. Where two checks
//! are adjacent, a list that breaks both is rejected with the earlier one's field (RFC-0039
//! 'Testing Strategy').
//!
//! The golden lists name `test-small/1`, which only a build with `test-params` accepts. Every
//! other test starts from a golden list's production twin, which names `ni-100/1` with that
//! set's layout (`claim_support`), so that it runs in every build.

mod claim_support;

use claim_support::{
    affine_index, at, golden, goldens, id, live_64_layout, make_affine, make_pot, mismatch,
    ni_100_layout, pot_index, seal, test_small_layout, twin, validated, with_layers,
};
use tv_codec::{DecodeLimits, Rational};
use tv_core::claims::{ClaimList, Profile, QuantPoint, Site};
use tv_core::params::ParamSet;
use tv_core::validation::{
    ClaimsError, PROTOCOL_VERSION, PROTOCOL_VERSION_2, Protocol, StatementField, check_public,
    load, load_in, validate, validate_in,
};
use tv_tables::{TableId, TableKind};

use StatementField::{
    Architecture, GadgetPolicy, Layout, Profile as ProfileField, ProtocolVersion, Quantization,
};

const SERVABLE: &str = "tiny-2-servable";
const MINIMAL: &str = "tiny-2-minimal";
const AFFINE: &str = "tiny-2-affine";
const DEEP: &str = "tiny-4-servable";

/// `validate`'s rejection of `claims` for its parameter set.
fn params_rejection(claims: &ClaimList) -> String {
    let error = match validate(claims) {
        Err(error) => error,
        Ok(_) => panic!("expected Reject(Params), the claim list validated"),
    };
    assert!(error.is_rejection(), "{error:?}");
    assert_eq!(error.code(), "REJ-PARAMS", "{error:?}");
    match error {
        ClaimsError::Params { name } => name.as_str().to_owned(),
        other => panic!("expected Reject(Params), got {other:?}"),
    }
}

/// A rational that must be in lowest terms.
fn ratio(num: i64, den: u64) -> Rational {
    Rational::new(num, den).expect("in lowest terms")
}

/// Asserts that `claims` passes every check up to and including `check`'s, which here means
/// that `validate` either accepts it or rejects it at a later check than `check`.
fn passes(claims: &ClaimList, check: StatementField) {
    let order = |field: &StatementField| match field {
        ProtocolVersion => 1,
        Architecture => 2,
        Quantization => 3,
        StatementField::Rescaling { .. } => 4,
        ProfileField => 5,
        StatementField::Table { .. } => 6,
        GadgetPolicy => 7,
        StatementField::Gadget { .. } => 8,
        Layout => 9,
        _ => 10,
    };
    match validate(claims) {
        Ok(_) => {}
        Err(ClaimsError::StatementMismatch { field }) => assert!(
            order(&field) > order(&check),
            "rejected with {field:?}, at or before {check:?}"
        ),
        Err(other) => panic!("rejected with {other:?}"),
    }
}

// ----- The golden lists ----------------------------------------------------------------------

#[cfg(feature = "test-params")]
#[test]
fn every_golden_claim_list_validates() {
    use tv_core::lowering::lower;
    for golden in goldens() {
        let (claims, validated) = load(&golden.envelope, &DecodeLimits::MESSAGE)
            .unwrap_or_else(|e| panic!("{}: {e:?}", golden.name));
        assert_eq!(validated.params(), ParamSet::TestSmall, "{}", golden.name);
        assert_eq!(claims.digest(), golden.digest, "{}", golden.name);
        // Step 4 regenerated every table of the list, in `TableId` order, to its specification.
        let specs: Vec<_> = validated
            .tables()
            .iter()
            .map(|t| t.spec().clone())
            .collect();
        assert_eq!(specs, claims.tables, "{}", golden.name);
        // Step 3's graph is the lowering of the longest sequence the list admits.
        let public = check_public(&claims).unwrap();
        let graph = lower(&public, claims.arch.max_context).unwrap();
        assert_eq!(validated.graph(), &graph, "{}", golden.name);
    }
}

#[cfg(feature = "test-params")]
#[test]
fn test_small_needs_its_own_version() {
    let mut claims = golden(SERVABLE);
    claims.params = id("test-small/2");
    assert_eq!(params_rejection(&claims), "test-small/2");
}

#[cfg(not(feature = "test-params"))]
#[test]
fn test_small_without_the_feature_is_refused_as_params() {
    // RFC-0019 step 2, before every other check: each golden list is refused by its name alone.
    for golden in goldens() {
        let claims = ClaimList::open(&golden.envelope, &DecodeLimits::MESSAGE).unwrap();
        assert_eq!(params_rejection(&claims), "test-small/1", "{}", golden.name);
        match load(&golden.envelope, &DecodeLimits::MESSAGE) {
            Err(ClaimsError::Params { name }) => assert_eq!(name.as_str(), "test-small/1"),
            other => panic!("{}: {other:?}", golden.name),
        }
    }
    assert_eq!(ParamSet::accept(&id("test-small/1")), None);
}

#[test]
fn every_golden_claim_list_validates_as_its_production_twin() {
    for golden in goldens() {
        let validated = validated(&twin(&golden.name));
        assert_eq!(validated.params(), ParamSet::Ni100, "{}", golden.name);
    }
    // `live-64/1` with its own layout too.
    let mut claims = golden(SERVABLE);
    claims.params = id("live-64/1");
    claims.layout = live_64_layout();
    assert_eq!(validated(&claims).params(), ParamSet::Live64);
}

#[test]
fn the_parameter_sets_are_rfc_0014s() {
    assert_eq!(ParamSet::accept(&id("live-64/1")), Some(ParamSet::Live64));
    assert_eq!(ParamSet::accept(&id("ni-100/1")), Some(ParamSet::Ni100));
    assert_eq!(ParamSet::Live64.name(), "live-64/1");
    assert_eq!(ParamSet::Ni100.name(), "ni-100/1");
    assert_eq!(ParamSet::Live64.layout(), live_64_layout());
    assert_eq!(ParamSet::Ni100.layout(), ni_100_layout());
    #[cfg(feature = "test-params")]
    {
        assert_eq!(
            ParamSet::accept(&id("test-small/1")),
            Some(ParamSet::TestSmall)
        );
        assert_eq!(ParamSet::TestSmall.layout(), test_small_layout());
    }
    let _ = test_small_layout;
}

// ----- RFC-0019 step 2: the parameter set ----------------------------------------------------

#[test]
fn an_unsupported_parameter_set_is_refused_with_its_name() {
    for name in [
        "live-65/1",
        "ni-100/2",
        "live-64/0",
        "",
        "LIVE-64/1",
        "ni-100/1 ",
    ] {
        let mut claims = twin(SERVABLE);
        claims.params = id(name);
        assert_eq!(params_rejection(&claims), name);
    }
}

// ----- Step 1: the protocol version ----------------------------------------------------------

#[test]
fn a_protocol_version_the_verifier_does_not_implement_is_refused() {
    assert_eq!(
        PROTOCOL_VERSION, 0,
        "tessera-protocol/1 is not assigned yet (09 §3)"
    );
    for version in [1, 2, u16::MAX] {
        let mut claims = twin(SERVABLE);
        claims.protocol_version = version;
        assert_eq!(mismatch(&claims), ProtocolVersion, "version {version}");
    }
}

// ----- Step 2.1: the architecture ------------------------------------------------------------

#[test]
fn each_architecture_rule_is_checked() {
    type Edit = fn(&mut ClaimList);
    let cases: [(&str, Edit); 15] = [
        ("no layer", |c| c.arch.n_layers = 0),
        // `n_kv_heads ≥ 1` divides `n_heads ≥ 1` (RFC-0007 step 2.1, as amended): 2 divides 0, and
        // `d_model = 0 · 16` holds, so only `n_heads ≥ 1` breaks.
        ("no head", |c| {
            c.arch.n_heads = 0;
            c.arch.d_model = 0;
        }),
        ("no KV head", |c| c.arch.n_kv_heads = 0),
        ("KV heads that do not divide the heads", |c| {
            c.arch.n_kv_heads = 3
        }),
        ("three query heads per KV head (RFC-0047)", |c| {
            c.arch.n_heads = 24;
            c.arch.n_kv_heads = 8;
            c.arch.d_model = 24 * c.arch.d_head;
        }),
        ("heads whose dimensions are not d_model", |c| {
            c.arch.n_heads = 8
        }),
        ("an odd d_head", |c| {
            c.arch.d_head = 15;
            c.arch.d_model = 60;
        }),
        ("a d_head of 0", |c| {
            c.arch.d_head = 0;
            c.arch.d_model = 0;
        }),
        ("a d_head of 2^16", |c| {
            c.arch.d_head = 1 << 16;
            c.arch.d_model = 4 << 16;
        }),
        ("d_model other than n_heads · d_head", |c| {
            c.arch.d_model = 65
        }),
        ("no MLP width", |c| c.arch.d_ff = 0),
        ("an empty vocabulary", |c| c.arch.vocab = 0),
        ("a vocabulary over 2^18", |c| c.arch.vocab = (1 << 18) + 1),
        ("no context", |c| c.arch.max_context = 0),
        ("a context over 2^17", |c| {
            c.arch.max_context = (1 << 17) + 1
        }),
    ];
    for (what, edit) in cases {
        let mut claims = twin(SERVABLE);
        edit(&mut claims);
        assert_eq!(mismatch(&claims), Architecture, "{what}");
    }
}

#[test]
fn the_architecture_admits_its_boundaries() {
    // The group, rather than either head count, must be a power of two (RFC-0047).
    let mut grouped = twin(SERVABLE);
    grouped.arch.n_heads = 24;
    grouped.arch.n_kv_heads = 3;
    grouped.arch.d_model = 24 * grouped.arch.d_head;
    passes(&grouped, Architecture);

    // In `minimal`, which has no accumulator bound (S2), so that only the rule is tested.
    let mut claims = twin(MINIMAL);
    claims.arch.d_head = (1 << 16) - 2;
    claims.arch.d_model = 4 * ((1 << 16) - 2);
    passes(&claims, Architecture);
    let mut claims = twin(MINIMAL);
    claims.arch.d_head = 2;
    claims.arch.d_model = 8;
    passes(&claims, Architecture);
    let mut claims = twin(SERVABLE);
    claims.arch.vocab = 1 << 18;
    passes(&claims, Architecture);
    let mut claims = twin(SERVABLE);
    claims.arch.vocab = 1;
    passes(&claims, Architecture);
    let mut claims = twin(SERVABLE);
    claims.arch.max_context = 1 << 17;
    passes(&claims, Architecture);
    let mut claims = twin(SERVABLE);
    claims.arch.max_context = 1;
    passes(&claims, Architecture);
    let mut claims = twin(SERVABLE);
    claims.arch.n_kv_heads = 4;
    passes(&claims, Architecture);
    let mut claims = twin(SERVABLE);
    claims.arch.n_kv_heads = 1;
    passes(&claims, Architecture);
    // One head, of one KV head: `n_heads ≥ 1` at its bound.
    let mut claims = twin(SERVABLE);
    claims.arch.n_heads = 1;
    claims.arch.n_kv_heads = 1;
    claims.arch.d_model = 16;
    passes(&claims, Architecture);
}

// ----- Step 2.2: the public quantization data ------------------------------------------------

#[test]
fn quant_rope_must_be_the_architectures() {
    let mut claims = twin(SERVABLE);
    claims.quant.rope.theta += 1;
    assert_eq!(mismatch(&claims), Quantization);
    let mut claims = twin(SERVABLE);
    claims.quant.rope.bits = 13;
    assert_eq!(mismatch(&claims), Quantization);
}

#[test]
fn act_exponents_hold_every_int8_point_once_in_order() {
    type Edit = fn(&mut ClaimList);
    let cases: [(&str, Edit); 7] = [
        ("the final norm missing", |c| {
            c.quant.act_exponents.pop();
        }),
        ("the embedding missing", |c| {
            c.quant.act_exponents.remove(0);
        }),
        ("a point repeated", |c| {
            let entry = c.quant.act_exponents[3];
            c.quant.act_exponents.insert(3, entry);
        }),
        ("two points out of order", |c| {
            c.quant.act_exponents.swap(2, 3)
        }),
        ("a residual site, which is not INT8", |c| {
            let point = at(0, Site::AttnResidual);
            let index = c.quant.act_exponents.partition_point(|(p, _)| *p < point);
            c.quant.act_exponents.insert(index, (point, -3));
        }),
        ("a point of a layer past the last", |c| {
            let index = c.quant.act_exponents.len() - 1;
            c.quant
                .act_exponents
                .insert(index, (at(2, Site::AttnNorm), -3));
        }),
        ("an INT8 site of the profile missing", |c| {
            c.quant
                .act_exponents
                .retain(|(p, _)| *p != at(1, Site::Product));
        }),
    ];
    for (what, edit) in cases {
        let mut claims = twin(SERVABLE);
        edit(&mut claims);
        assert_eq!(mismatch(&claims), Quantization, "{what}");
    }
    // A site of `servable` alone in a `minimal` list.
    let mut claims = twin(MINIMAL);
    let point = at(0, Site::AttnOut);
    let index = claims
        .quant
        .act_exponents
        .partition_point(|(p, _)| *p < point);
    claims.quant.act_exponents.insert(index, (point, -3));
    assert_eq!(mismatch(&claims), Quantization);
}

#[test]
fn each_rescale_site_is_in_exactly_one_list_both_sorted() {
    // In both lists.
    let mut claims = twin(SERVABLE);
    claims.quant.affine_sites.push((at(0, Site::V), 20));
    assert_eq!(mismatch(&claims), Quantization, "in both lists");
    // In neither.
    let mut claims = twin(SERVABLE);
    let index = pot_index(&claims, at(0, Site::MlpNorm));
    claims.quant.shift_class_sets.remove(index);
    assert_eq!(mismatch(&claims), Quantization, "in neither list");
    let mut claims = twin(SERVABLE);
    claims.quant.shift_class_sets.pop();
    assert_eq!(
        mismatch(&claims),
        Quantization,
        "the final norm in neither list"
    );
    // Repeated in one list.
    let mut claims = twin(SERVABLE);
    let entry = claims.quant.shift_class_sets[4].clone();
    claims.quant.shift_class_sets.insert(4, entry);
    assert_eq!(mismatch(&claims), Quantization, "repeated");
    // Out of order.
    let mut claims = twin(SERVABLE);
    claims.quant.shift_class_sets.swap(1, 2);
    assert_eq!(mismatch(&claims), Quantization, "pot sites out of order");
    let mut claims = twin(AFFINE);
    claims.quant.affine_sites.swap(0, 1);
    assert_eq!(mismatch(&claims), Quantization, "affine sites out of order");
    // A point that is no rescale site: the embedding, a layer past the last, `minimal`'s `Up`.
    let mut claims = twin(SERVABLE);
    claims
        .quant
        .shift_class_sets
        .insert(0, (QuantPoint::Embed, vec![3]));
    assert_eq!(mismatch(&claims), Quantization, "the embedding");
    let mut claims = twin(SERVABLE);
    let last = claims.quant.shift_class_sets.len() - 1;
    claims
        .quant
        .shift_class_sets
        .insert(last, (at(2, Site::AttnNorm), vec![26]));
    assert_eq!(mismatch(&claims), Quantization, "a layer past the last");
    let mut claims = twin(MINIMAL);
    make_pot(&mut claims, at(0, Site::Up), &[8]);
    assert_eq!(mismatch(&claims), Quantization, "`Up` in minimal");
}

#[test]
fn an_affine_site_follows_an_allowed_projection_with_a_shift_from_1_to_63() {
    // Sites that follow no Q, K, V, O, up or down projection, affine in every layer so that
    // only this rule breaks.
    for site in [
        Site::AttnNorm,
        Site::AttnOut,
        Site::MlpNorm,
        Site::Silu,
        Site::Product,
    ] {
        let mut claims = twin(SERVABLE);
        for layer in 0..2 {
            make_affine(&mut claims, at(layer, site), 20);
        }
        assert_eq!(mismatch(&claims), Quantization, "affine {site:?}");
    }
    let mut claims = twin(SERVABLE);
    claims
        .quant
        .shift_class_sets
        .retain(|(p, _)| *p != QuantPoint::FinalNorm);
    claims.quant.affine_sites.push((QuantPoint::FinalNorm, 26));
    assert_eq!(mismatch(&claims), Quantization, "affine final norm");
    // Every allowed site, in every layer, with shifts at both ends.
    for site in [
        Site::RopeQ,
        Site::RopeK,
        Site::V,
        Site::AttnResidual,
        Site::Up,
        Site::MlpResidual,
    ] {
        for r in [1, 63] {
            let mut claims = twin(SERVABLE);
            for layer in 0..2 {
                make_affine(&mut claims, at(layer, site), r);
            }
            passes(&claims, StatementField::Rescaling { site });
        }
    }
    // A shift of 0 or 64.
    for r in [0, 64, -1, i8::MIN] {
        let mut claims = twin(AFFINE);
        let index = affine_index(&claims, at(0, Site::RopeQ));
        claims.quant.affine_sites[index].1 = r;
        assert_eq!(mismatch(&claims), Quantization, "affine shift {r}");
    }
}

#[test]
fn a_pot_site_has_one_to_four_classes_from_1_to_63_and_one_after_no_weight() {
    let shifts: [(&str, Vec<i8>); 6] = [
        ("no class", vec![]),
        ("five classes", vec![8, 9, 10, 11, 12]),
        ("a class of 0", vec![0, 9]),
        ("a class of 64", vec![9, 64]),
        ("a negative class", vec![-9, 9]),
        ("a class of i8::MIN", vec![i8::MIN]),
    ];
    for (what, classes) in shifts {
        let mut claims = twin(SERVABLE);
        let index = pot_index(&claims, at(0, Site::V));
        claims.quant.shift_class_sets[index].1 = classes;
        assert_eq!(mismatch(&claims), Quantization, "{what}");
    }
    // Several classes at a site that follows no weight, in every layer.
    for site in [Site::AttnNorm, Site::AttnOut, Site::MlpNorm, Site::Product] {
        let mut claims = twin(SERVABLE);
        for layer in 0..2 {
            make_pot(&mut claims, at(layer, site), &[26, 27]);
        }
        assert_eq!(mismatch(&claims), Quantization, "{site:?} of two classes");
    }
    let mut claims = twin(SERVABLE);
    make_pot(&mut claims, QuantPoint::FinalNorm, &[26, 27]);
    assert_eq!(
        mismatch(&claims),
        Quantization,
        "the final norm of two classes"
    );
    // Four classes at both ends of the range pass.
    let mut claims = twin(SERVABLE);
    for layer in 0..2 {
        make_pot(&mut claims, at(layer, Site::V), &[1, 2, 62, 63]);
    }
    passes(&claims, Quantization);
}

#[test]
fn score_scales_hold_one_positive_scale_per_layer_in_order() {
    type Edit = fn(&mut ClaimList);
    let cases: [(&str, Edit); 6] = [
        ("a layer missing", |c| {
            c.quant.score_scales.pop();
        }),
        ("a layer past the last", |c| {
            c.quant.score_scales.push((2, ratio(1, 2)));
        }),
        ("out of order", |c| {
            c.quant.score_scales.swap(0, 1);
        }),
        ("a layer repeated", |c| {
            c.quant.score_scales[1].0 = 0;
        }),
        ("a scale of 0", |c| c.quant.score_scales[1].1 = ratio(0, 1)),
        ("a negative scale", |c| {
            c.quant.score_scales[0].1 = ratio(-4, 6827)
        }),
    ];
    for (what, edit) in cases {
        let mut claims = twin(SERVABLE);
        edit(&mut claims);
        assert_eq!(mismatch(&claims), Quantization, "{what}");
    }
}

#[test]
fn silu_in_exponents_hold_one_exponent_per_layer_in_order() {
    type Edit = fn(&mut ClaimList);
    let cases: [(&str, Edit); 4] = [
        ("a layer missing", |c| {
            c.silu_in_exponents.pop();
        }),
        ("a layer past the last", |c| {
            c.silu_in_exponents.push((2, -7))
        }),
        ("out of order", |c| {
            c.silu_in_exponents.swap(0, 1);
        }),
        ("none", |c| c.silu_in_exponents.clear()),
    ];
    for (what, edit) in cases {
        let mut claims = twin(SERVABLE);
        edit(&mut claims);
        assert_eq!(mismatch(&claims), Quantization, "{what}");
    }
}

#[test]
fn residual_bits_lie_from_2_to_31() {
    for bits in [0, 1, 32, u8::MAX] {
        let mut claims = twin(SERVABLE);
        claims.quant.residual_bits = bits;
        assert_eq!(mismatch(&claims), Quantization, "{bits} bits");
    }
    // 31 bits pass; 2 bits leave no room for the embedding's rows, which the next rule refuses.
    let mut claims = twin(SERVABLE);
    claims.quant.residual_bits = 31;
    passes(&claims, Quantization);
}

#[test]
fn the_embedding_shift_keeps_int8_rows_inside_the_residual() {
    // `s_e = e_Embed − residual_exponent`, here −3 − (−10) = 7, and RFC-0007 needs
    // `127 · 2^{s_e} < 2^{residual_bits − 1}`, with 20 bits `2^19`: `s_e = 12` is the last.
    let with_shift = |s_e: i8| {
        let mut claims = twin(SERVABLE);
        claims.quant.residual_exponent = -3 - s_e;
        claims
    };
    passes(&with_shift(12), Quantization);
    passes(&with_shift(0), Quantization);
    assert_eq!(mismatch(&with_shift(13)), Quantization, "s_e = 13");
    assert_eq!(mismatch(&with_shift(100)), Quantization, "s_e = 100");
    // Fewer residual bits: 16 bits hold `127 · 2^{s_e}` up to `s_e = 8`.
    let mut claims = with_shift(8);
    claims.quant.residual_bits = 16;
    passes(&claims, Quantization);
    claims.quant.residual_bits = 15;
    assert_eq!(mismatch(&claims), Quantization, "15 bits and s_e = 8");
    // RFC-0007 step 2.2, as amended: `s_e ≥ 0` (RFC-0005 'Scale policy'), beside the bound above.
    // A negative shift would make the embedding's constant `2^{s_e}` a fraction.
    assert_eq!(mismatch(&with_shift(-1)), Quantization, "s_e = −1");
    assert_eq!(mismatch(&with_shift(-100)), Quantization, "s_e = −100");
}

/// The narrowest `key_bits` from `from` up to 32 whose `Exp { key_bits, sigma, scale }` maps
/// its last key to 0, by `tv-tables`' value test: what RFC-0007 step 2.2 bounds by 25.
fn exp_width(from: u8, sigma: u8, scale: Rational) -> Option<u8> {
    (from..=32).find(|&key_bits| {
        tv_tables::exp_maps_to_zero((1u64 << key_bits) - 1, sigma, scale).expect("a positive scale")
    })
}

#[test]
fn a_score_scale_no_exp_table_serves_is_refused_as_quantization() {
    // At `σ = 16` the scale `2^-40` maps no last key to 0 up to 32 bits: no `Exp` table serves
    // layer 0, so step 2.2 refuses the claim list before any lowering.
    let tiny = ratio(1, 1 << 40);
    assert_eq!(exp_width(19, 16, tiny), None);
    let mut claims = twin(SERVABLE);
    assert_eq!(claims.quant.exp_sigma, 16);
    claims.quant.score_scales[0].1 = tiny;
    assert_eq!(mismatch(&claims), Quantization);
    let mut claims = twin(DEEP);
    claims.quant.score_scales[3].1 = tiny;
    assert_eq!(mismatch(&claims), Quantization, "the last of four layers");
}

/// Whether RFC-0008 allows `Exp { key_bits, sigma, scale }`, by `tv-tables`' parameter checks.
fn allowed(key_bits: u8, sigma: u8, scale: Rational) -> bool {
    TableKind::Exp {
        key_bits,
        sigma,
        scale,
    }
    .check_parameters()
    .is_ok()
}

#[test]
fn an_exp_table_widens_only_while_rfc_0008_allows_it() {
    // The tiny models' heads of 16 give a spread of `bitlen(128 · 255 · 16) = 19` bits. At
    // `σ = 16`, `2^-20` widens their table to 24 bits, which RFC-0008 allows. `2^-21` widens it
    // to 25 bits, whose zero tail starts past key 2^24, and `2^-22` to 26: RFC-0008 refuses both.
    let at = |den_bits: u32| ratio(1, 1 << den_bits);
    assert_eq!(exp_width(19, 16, at(20)), Some(24));
    assert!(allowed(24, 16, at(20)));
    assert_eq!(exp_width(19, 16, at(21)), Some(25));
    assert!(!allowed(25, 16, at(21)));
    assert_eq!(exp_width(19, 16, at(22)), Some(26));
    assert!(!allowed(26, 16, at(22)));
    for (den_bits, admitted) in [(20, true), (21, false), (22, false)] {
        let mut claims = twin(SERVABLE);
        claims.quant.score_scales[1].1 = at(den_bits);
        if admitted {
            passes(&claims, Quantization);
        } else {
            assert_eq!(mismatch(&claims), Quantization, "scale 2^-{den_bits}");
        }
    }
}

#[test]
fn a_wide_head_keeps_its_spreads_width_when_rfc_0008_allows_it() {
    // Heads of 2,048 give a spread of 26 bits. At the scale 1/2896 the zero tail starts at key
    // 36,133, so the 26-bit table, the first width, is one RFC-0008 allows.
    let scale = ratio(1, 2896);
    assert_eq!(exp_width(26, 16, scale), Some(26));
    assert!(allowed(26, 16, scale));
    let mut claims = twin(SERVABLE);
    claims.arch.d_head = 2_048;
    claims.arch.d_model = 4 * 2_048;
    for entry in &mut claims.quant.score_scales {
        entry.1 = scale;
    }
    passes(&claims, Quantization);
}

/// RFC-0009 Lemma 6's precondition in the form RFC-0007 step 2.2 states first,
/// `(2·max_context − 1)·2^{2σ} < P`, in 128-bit integers.
fn reciprocal_unique(sigma: u8, max_context: u32) -> bool {
    const P: u128 = 0xFFFF_FFFF_0000_0001;
    let rows = 2 * u128::from(max_context) - 1;
    1u128
        .checked_shl(2 * u32::from(sigma))
        .and_then(|power| rows.checked_mul(power))
        .is_some_and(|product| product < P)
}

#[test]
fn exp_sigma_keeps_every_softmax_reciprocal_unique() {
    // RFC-0007 step 2.2, RFC-0009 Lemma 6 (#376): `(2·max_context − 1)·2^{2σ} < P`. Past it, a
    // row of `max_context` positions at its maximal score admits a second reciprocal. At each
    // context the largest σ passes step 2 and the next is `Quantization`.
    let at = |context: u32, sigma: u8| {
        let mut claims = twin(SERVABLE);
        claims.arch.max_context = context;
        claims.quant.exp_sigma = sigma;
        claims
    };
    for (context, largest) in [
        (1, 31),
        (2, 31),
        (3, 30),
        (16, 29),
        (64, 28),
        (1 << 15, 24),
        ((1 << 15) + 1, 23),
        (1 << 17, 23),
    ] {
        assert!(reciprocal_unique(largest, context) && !reciprocal_unique(largest + 1, context));
        passes(&at(context, largest), Quantization);
        let past = largest + 1;
        assert_eq!(
            mismatch(&at(context, past)),
            Quantization,
            "max_context = {context}, σ = {past}"
        );
    }
    // Every context that is a power of two or next to one, at every σ RFC-0008 allows and the
    // next: step 2 decides as the defining form does.
    let refused = Err(ClaimsError::StatementMismatch {
        field: Quantization,
    });
    for log in 0..=17 {
        for context in [(1u32 << log) - 1, 1 << log, (1 << log) + 1] {
            if !(1..=1 << 17).contains(&context) {
                continue;
            }
            for sigma in 0..=33 {
                let claims = at(context, sigma);
                let verdict = check_public(&claims).map(|_| ());
                let expected = match reciprocal_unique(sigma, context) {
                    true => Ok(()),
                    false => refused.clone(),
                };
                assert_eq!(verdict, expected, "max_context = {context}, σ = {sigma}");
            }
        }
    }
    // RFC-0008 'Table specifications' allows `Exp` tables of `σ ≤ 32`, which the rule above
    // makes `σ ≤ 31` at every context: past both, refused at the smallest context.
    for sigma in [32, 33, 64, u8::MAX] {
        assert_eq!(mismatch(&at(1, sigma)), Quantization, "σ = {sigma}");
    }
    // The golden lists' σ = 16 at their context of 64 is far inside the bound.
    let claims = twin(SERVABLE);
    assert_eq!((claims.quant.exp_sigma, claims.arch.max_context), (16, 64));
    assert!(reciprocal_unique(16, 64));
}

#[test]
fn the_norms_constants_are_bounded() {
    let mut claims = twin(SERVABLE);
    claims.quant.rms_eps_int = 0;
    assert_eq!(mismatch(&claims), Quantization, "rms_eps_int = 0");
    for kappa in [32, u8::MAX] {
        let mut claims = twin(SERVABLE);
        claims.quant.rsqrt_kappa = kappa;
        assert_eq!(mismatch(&claims), Quantization, "κ = {kappa}");
    }
    let mut claims = twin(SERVABLE);
    claims.quant.rms_eps_int = 1;
    claims.quant.rsqrt_kappa = 0;
    passes(&claims, Quantization);
}

// ----- Step 2.3: every layer rescaled alike --------------------------------------------------

#[test]
fn a_site_affine_in_some_layers_only_is_refused_with_its_site() {
    let mut claims = twin(SERVABLE);
    make_affine(&mut claims, at(0, Site::V), 20);
    assert_eq!(
        mismatch(&claims),
        StatementField::Rescaling { site: Site::V }
    );
    let mut claims = twin(AFFINE);
    make_pot(&mut claims, at(1, Site::Up), &[8, 9]);
    assert_eq!(
        mismatch(&claims),
        StatementField::Rescaling { site: Site::Up }
    );
}

#[test]
fn a_pot_site_of_one_class_in_some_layers_and_several_in_others_is_refused() {
    let mut claims = twin(SERVABLE);
    make_pot(&mut claims, at(0, Site::AttnResidual), &[3]);
    assert_eq!(
        mismatch(&claims),
        StatementField::Rescaling {
            site: Site::AttnResidual
        }
    );
    let mut claims = twin(DEEP);
    make_pot(&mut claims, at(2, Site::AttnResidual), &[3, 4]);
    assert_eq!(
        mismatch(&claims),
        StatementField::Rescaling {
            site: Site::AttnResidual
        }
    );
    // Several classes in every layer, two in one and three in another, are alike.
    let mut claims = twin(SERVABLE);
    make_pot(&mut claims, at(1, Site::RopeK), &[22, 23, 24]);
    passes(&claims, StatementField::Rescaling { site: Site::RopeK });
}

#[test]
fn the_first_site_in_declaration_order_is_named() {
    // `RopeK` breaks the rule in layer 1 and `AttnResidual` in layer 0: the list meets
    // `AttnResidual` first, and 03 §2 declares `RopeK` first.
    let mut claims = twin(SERVABLE);
    make_affine(&mut claims, at(1, Site::RopeK), 22);
    make_pot(&mut claims, at(0, Site::AttnResidual), &[3]);
    assert_eq!(
        mismatch(&claims),
        StatementField::Rescaling { site: Site::RopeK }
    );
}

// ----- Step 3: the profile -------------------------------------------------------------------

#[test]
fn a_servable_accumulator_past_int32_breaks_the_profile() {
    // The down projection's accumulator is bounded by `127 · 128 · d_ff`: INT32 holds it up to
    // `d_ff = 132,104` (S2, RFC-0005 'Profiles').
    let mut claims = twin(SERVABLE);
    claims.arch.d_ff = 132_105;
    assert_eq!(mismatch(&claims), ProfileField, "the down projection");
    let mut claims = twin(SERVABLE);
    claims.arch.d_ff = 132_104;
    passes(&claims, ProfileField);
    // The Q, K and V projections read `d_model` channels: 4 heads of 33,026, `d_model = 132,104`,
    // keep the accumulators within INT32, and 4 heads of 33,028 do not. Their spread of 31 bits
    // takes the scales' first widths, which RFC-0008 allows.
    let with_heads = |d_head: u32| {
        let mut claims = twin(SERVABLE);
        claims.arch.d_head = d_head;
        claims.arch.d_model = 4 * d_head;
        claims
    };
    assert_eq!(
        mismatch(&with_heads(33_028)),
        ProfileField,
        "the attention projections"
    );
    passes(&with_heads(33_026), ProfileField);
}

#[test]
fn minimal_has_no_accumulator_bound() {
    let mut claims = twin(MINIMAL);
    claims.arch.d_ff = 1 << 20;
    passes(&claims, ProfileField);
    assert_eq!(claims.profile, Profile::Minimal);
}

// ----- Step 4: the tables --------------------------------------------------------------------

#[test]
fn a_tampered_table_digest_is_refused_with_its_table() {
    for index in [0, 1, 2, 5, 22] {
        let mut claims = twin(SERVABLE);
        claims.tables[index].digest[31] ^= 0x80;
        let table = claims.tables[index].id;
        assert_eq!(
            mismatch(&claims),
            StatementField::Table { table },
            "table {index}"
        );
    }
}

#[test]
fn a_table_whose_regenerated_digest_differs_is_refused() {
    // Table 1 is layer 0's `Exp`: another scale under the same digest.
    let mut claims = twin(SERVABLE);
    let TableKind::Exp { scale, .. } = &mut claims.tables[1].kind else {
        panic!("table 1 is an Exp table");
    };
    *scale = ratio(5, 6827);
    assert_eq!(
        mismatch(&claims),
        StatementField::Table { table: TableId(1) }
    );
    // Change a Requant clamp under its original digest; auxiliary families precede it.
    let mut claims = twin(SERVABLE);
    let (id, lo) = claims
        .tables
        .iter_mut()
        .find_map(|t| match &mut t.kind {
            TableKind::Requant { lo, .. } => Some((t.id, lo)),
            _ => None,
        })
        .expect("a Requant table");
    *lo += 1;
    assert_eq!(mismatch(&claims), StatementField::Table { table: id });
}

#[test]
fn a_wrong_row_count_or_refused_parameters_are_refused_with_the_table() {
    let mut claims = twin(SERVABLE);
    claims.tables[2].log2_rows += 1;
    assert_eq!(
        mismatch(&claims),
        StatementField::Table { table: TableId(2) }
    );
    let mut claims = twin(SERVABLE);
    claims.tables[5].kind = TableKind::Range { bits: 0 };
    assert_eq!(
        mismatch(&claims),
        StatementField::Table { table: TableId(5) }
    );
}

#[test]
fn tables_are_checked_in_table_id_order() {
    // Two broken tables, listed out of order: the lower id is named.
    let mut claims = twin(SERVABLE);
    claims.tables.swap(3, 7);
    claims.tables[3].digest[0] ^= 1; // id 7
    claims.tables[7].digest[0] ^= 1; // id 3
    assert_eq!(claims.tables[3].id, TableId(7));
    assert_eq!(
        mismatch(&claims),
        StatementField::Table { table: TableId(3) }
    );
}

// ----- Step 5.1: the gadget policy -----------------------------------------------------------

#[test]
fn max_log2_rows_lies_from_1_to_32() {
    for rows in [0, 33, 40, u8::MAX] {
        let mut claims = twin(SERVABLE);
        claims.gadget_policy.max_log2_rows = rows;
        assert_eq!(mismatch(&claims), GadgetPolicy, "max_log2_rows = {rows}");
    }
    for rows in [1, 32] {
        let mut claims = twin(SERVABLE);
        claims.gadget_policy.max_log2_rows = rows;
        passes(&claims, GadgetPolicy);
    }
}

// ----- Step 6: the layout --------------------------------------------------------------------

#[test]
fn the_commitment_layout_is_the_parameter_sets() {
    // The golden list's `test-small/1` layout under `ni-100/1`.
    let mut claims = golden(SERVABLE);
    claims.params = id("ni-100/1");
    assert_eq!(mismatch(&claims), Layout, "test-small's layout");
    // `live-64/1`'s layout under `ni-100/1`: only `P_A` differs.
    let mut claims = twin(SERVABLE);
    claims.layout = live_64_layout();
    assert_eq!(mismatch(&claims), Layout, "live-64's layout");
    type Edit = fn(&mut ClaimList);
    let cases: [(&str, Edit); 9] = [
        ("advice log2 k", |c| c.layout.advice.log2_k = 21),
        ("advice rate", |c| c.layout.advice.log2_inv_rate = 2),
        ("advice mask", |c| c.layout.advice.mask_coeffs = 199),
        ("advice row block", |c| c.layout.advice.row_block = 16),
        ("advice segments", |c| c.layout.advice.segments = 9),
        ("weights log2 k", |c| c.layout.weights.log2_k = 19),
        ("weights rate", |c| c.layout.weights.log2_inv_rate = 1),
        ("weights mask", |c| c.layout.weights.mask_coeffs = 65_535),
        ("weights blinding rows", |c| {
            c.layout.weights.blinding_rows = 4_097
        }),
    ];
    for (what, edit) in cases {
        let mut claims = twin(SERVABLE);
        edit(&mut claims);
        assert_eq!(mismatch(&claims), Layout, "{what}");
    }
}

#[test]
fn query_block_is_a_power_of_two() {
    for block in [0, 3, 513, 768, u32::MAX] {
        let mut claims = twin(SERVABLE);
        claims.gadget_policy.query_block = block;
        assert_eq!(mismatch(&claims), Layout, "query_block = {block}");
    }
    for block in [1, 64, 512, 1 << 31] {
        let mut claims = twin(SERVABLE);
        claims.gadget_policy.query_block = block;
        validated(&claims);
    }
}

#[test]
fn limb_bits_are_not_judged_before_the_gadgets() {
    // RFC-0039: a bad `limb_bits` fails where the lowering uses it, at step 5.2 (#117).
    let mut claims = twin(SERVABLE);
    claims.gadget_policy.limb_bits = 0;
    passes(&claims, GadgetPolicy);
}

// ----- The order of the checks ---------------------------------------------------------------

#[test]
fn each_pair_of_adjacent_checks_gives_the_earlier_ones_field() {
    // Params and the protocol version.
    let mut claims = twin(SERVABLE);
    claims.params = id("ni-101/1");
    claims.protocol_version = 9;
    assert_eq!(params_rejection(&claims), "ni-101/1");
    // The protocol version and the architecture.
    let mut claims = twin(SERVABLE);
    claims.protocol_version = 9;
    claims.arch.n_kv_heads = 3;
    assert_eq!(mismatch(&claims), ProtocolVersion);
    // The architecture and the quantization data.
    let mut claims = twin(SERVABLE);
    claims.arch.n_kv_heads = 3;
    claims.quant.rms_eps_int = 0;
    assert_eq!(mismatch(&claims), Architecture);
    // The quantization data and the rescaling.
    let mut claims = twin(SERVABLE);
    claims.quant.rms_eps_int = 0;
    make_affine(&mut claims, at(0, Site::V), 20);
    assert_eq!(mismatch(&claims), Quantization);
    // The rescaling and the profile.
    let mut claims = twin(SERVABLE);
    make_affine(&mut claims, at(0, Site::V), 20);
    claims.arch.d_ff = 132_105;
    assert_eq!(
        mismatch(&claims),
        StatementField::Rescaling { site: Site::V }
    );
    // The profile and the tables.
    let mut claims = twin(SERVABLE);
    claims.arch.d_ff = 132_105;
    claims.tables[3].digest[0] ^= 1;
    assert_eq!(mismatch(&claims), ProfileField);
    // The tables and the gadget policy.
    let mut claims = twin(SERVABLE);
    claims.tables[3].digest[0] ^= 1;
    claims.gadget_policy.max_log2_rows = 40;
    assert_eq!(
        mismatch(&claims),
        StatementField::Table { table: TableId(3) }
    );
    // The gadget policy and the layout.
    let mut claims = twin(SERVABLE);
    claims.gadget_policy.max_log2_rows = 40;
    claims.gadget_policy.query_block = 500;
    assert_eq!(mismatch(&claims), GadgetPolicy);
}

#[test]
fn the_verdict_classes_are_04s() {
    let mut claims = twin(SERVABLE);
    claims.arch.n_layers = 0;
    let error = validate(&claims).unwrap_err();
    assert!(error.is_rejection());
    assert_eq!(error.code(), "REJ-STATEMENT");
    assert_eq!(
        error,
        ClaimsError::StatementMismatch {
            field: Architecture
        }
    );
    assert!(!ClaimsError::Memory.is_rejection());
    assert_eq!(ClaimsError::Memory.code(), "ERR-RES-MEMORY");
    assert!(!ClaimsError::Internal.is_rejection());
    assert_eq!(ClaimsError::Internal.code(), "ERR-INTERNAL");
}

#[test]
fn statement_fields_carry_rfc_0039s_tags() {
    // RFC-0039 'The type': a `u16` tag, the variant's declaration index.
    let fields = [
        ProtocolVersion,
        Architecture,
        Quantization,
        StatementField::Rescaling { site: Site::V },
        ProfileField,
        StatementField::Table { table: TableId(0) },
        GadgetPolicy,
        StatementField::Gadget {
            gadget: tv_core::validation::GadgetId(0),
        },
        Layout,
        StatementField::Template,
        StatementField::ClaimListDigest,
        StatementField::WeightsRoot,
        StatementField::WeightsEpoch,
        StatementField::ManifestDigest,
        StatementField::Batch,
        StatementField::Predictor,
        StatementField::LayoutDigest,
    ];
    for (index, field) in fields.iter().enumerate() {
        assert_eq!(usize::from(field.tag()), index, "{field:?}");
    }
}

#[test]
fn a_narrow_requant_table_cannot_certify_a_wide_derived_key() {
    let mut claims = twin(SERVABLE);
    let spec = claims
        .tables
        .iter_mut()
        .find(|s| matches!(s.kind, TableKind::Requant { b, .. } if b > 1))
        .unwrap();
    let id = spec.id;
    let mut kind = spec.kind.clone();
    if let TableKind::Requant { b, .. } = &mut kind {
        *b = 1;
    }
    *spec = tv_tables::Table::generate(id, kind).unwrap().spec().clone();
    assert_eq!(mismatch(&claims), StatementField::Table { table: id });
}

#[test]
fn an_oversized_split_remainder_cannot_replace_the_derived_split() {
    let mut claims = twin(SERVABLE);
    let spec = claims
        .tables
        .iter_mut()
        .find(|s| matches!(s.kind, TableKind::Range { bits } if bits < 32))
        .unwrap();
    let id = spec.id;
    *spec = tv_tables::Table::generate(id, TableKind::Range { bits: 32 })
        .unwrap()
        .spec()
        .clone();
    assert_eq!(mismatch(&claims), StatementField::Table { table: id });
}

#[test]
fn canonical_table_ids_reject_missing_extra_duplicate_and_reordered_entries() {
    let base = twin(SERVABLE);
    let mut missing = base.clone();
    let removed = missing.tables.remove(2);
    assert_eq!(
        mismatch(&missing),
        StatementField::Table { table: removed.id }
    );
    let mut extra = base.clone();
    let orphan = TableId(10_000);
    extra.tables.push(
        tv_tables::Table::generate(orphan, TableKind::Range { bits: 1 })
            .unwrap()
            .spec()
            .clone(),
    );
    assert_eq!(mismatch(&extra), StatementField::Table { table: orphan });
    let mut duplicate = base.clone();
    duplicate.tables.push(base.tables[0].clone());
    assert_eq!(
        mismatch(&duplicate),
        StatementField::Table { table: TableId(0) }
    );
    let mut reordered = base;
    reordered.tables.swap(1, 2);
    assert_eq!(
        mismatch(&reordered),
        StatementField::Table { table: TableId(1) }
    );
}

#[test]
fn the_softmax_reciprocal_range_must_fit_the_table_policy() {
    let mut claims = twin(SERVABLE);
    claims.arch.max_context = 1;
    claims.quant.rsqrt_kappa = 2;
    claims.quant.exp_sigma = 31;
    claims.gadget_policy.max_log2_rows = 20;
    // Exp needs 19 key bits and fits. R needs Range(32), which does not; softmax is gadget 6.
    check_public(&claims).expect("valid public parameters");
    assert_eq!(
        mismatch(&claims),
        StatementField::Gadget {
            gadget: tv_core::validation::GadgetId(6)
        }
    );
}

#[test]
fn norm_integer_boxes_and_limb_policy_fail_at_the_first_norm() {
    for edit in 0..4 {
        let mut claims = twin(SERVABLE);
        match edit {
            0 => claims.quant.rms_eps_int = 1 << 55,
            1 => {
                claims.quant.rms_eps_int = 1;
                claims.quant.rsqrt_kappa = 31;
            }
            2 => claims.gadget_policy.limb_bits = 0,
            _ => claims.gadget_policy.limb_bits = 33,
        }
        check_public(&claims).expect("public parameters pass before range checks");
        assert_eq!(
            mismatch(&claims),
            StatementField::Gadget {
                gadget: tv_core::validation::GadgetId(1)
            }
        );
    }
}

#[test]
fn a_validly_regenerated_rope_table_must_match_the_public_model() {
    let mut claims = twin(SERVABLE);
    let spec = claims.tables.first_mut().unwrap();
    let mut kind = spec.kind.clone();
    let TableKind::Rope { theta, .. } = &mut kind else {
        panic!("table zero is RoPE")
    };
    *theta += 1;
    let id = spec.id;
    *spec = tv_tables::Table::generate(id, kind).unwrap().spec().clone();
    assert_eq!(mismatch(&claims), StatementField::Table { table: id });
}

#[test]
fn canonical_tables_pin_ids_and_requantization_functions_not_only_sizes() {
    let mut claims = twin(SERVABLE);
    let spec = claims.tables.last_mut().unwrap();
    let original_id = spec.id;
    *spec = tv_tables::Table::generate(TableId(10_000), spec.kind.clone())
        .unwrap()
        .spec()
        .clone();
    assert_eq!(
        mismatch(&claims),
        StatementField::Table { table: original_id }
    );
    let mut claims = twin(SERVABLE);
    let spec = claims
        .tables
        .iter_mut()
        .find(|s| matches!(&s.kind, TableKind::Requant { r_by_class, .. } if r_by_class[0] < 63))
        .unwrap();
    let original_id = spec.id;
    let mut kind = spec.kind.clone();
    if let TableKind::Requant { r_by_class, .. } = &mut kind {
        r_by_class[0] += 1;
    }
    *spec = tv_tables::Table::generate(original_id, kind)
        .unwrap()
        .spec()
        .clone();
    assert_eq!(
        mismatch(&claims),
        StatementField::Table { table: original_id }
    );
}

#[test]
fn softmax_auxiliary_ranges_are_canonical_even_with_valid_table_digests() {
    let claims = twin(SERVABLE);
    validated(&claims);
    for bits in [17, 16, 2] {
        let mut forged = claims.clone();
        let index = forged
            .tables
            .iter()
            .position(|t| t.kind == TableKind::Range { bits })
            .expect("SoftmaxRow's range");
        let tid = forged.tables[index].id;
        // The first-use position and id stay canonical, but the lookup box changes by one bit.
        forged.tables[index] = tv_tables::Table::generate(tid, TableKind::Range { bits: bits + 1 })
            .unwrap()
            .spec()
            .clone();
        assert_eq!(mismatch(&forged), StatementField::Table { table: tid });
    }
    let mut missing = claims.clone();
    let first = missing
        .tables
        .iter()
        .position(|t| t.kind == TableKind::Range { bits: 17 })
        .unwrap();
    let tid = missing.tables[first].id;
    missing.tables.remove(first);
    assert_eq!(mismatch(&missing), StatementField::Table { table: tid });
}

// ----- Protocol/2 version tag and step-6 masked weight rate (RFC-0012, #564) -----------------

struct CustomArch<'a> {
    params: &'a str,
    profile: Profile,
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    d_head: u32,
    d_ff: u32,
    vocab: u32,
}

/// Builds a claim list at `PROTOCOL_VERSION_2` whose public part, profile, tables, gadget
/// selection, and advice layout all pass steps 2..6, with every rescale site initialized to a
/// single `pot` class `[18]` (`m_CP == 0`) before `edit_public` customizes it.
fn custom_protocol_2_claims(
    spec: CustomArch<'_>,
    edit_public: impl FnOnce(&mut ClaimList),
) -> ClaimList {
    let template = match spec.profile {
        Profile::Servable => SERVABLE,
        Profile::Minimal => MINIMAL,
    };
    let mut claims = with_layers(&golden(template), spec.n_layers);
    claims.protocol_version = PROTOCOL_VERSION_2;
    claims.params = id(spec.params);
    claims.layout = ParamSet::accept(&claims.params)
        .expect("accepted parameter set")
        .layout();
    claims.arch.n_heads = spec.n_heads;
    claims.arch.n_kv_heads = spec.n_kv_heads;
    claims.arch.d_head = spec.d_head;
    claims.arch.d_model = spec.n_heads * spec.d_head;
    claims.arch.d_ff = spec.d_ff;
    claims.arch.vocab = spec.vocab;
    claims.arch.max_context = 4;

    for (_, exp) in &mut claims.quant.act_exponents {
        *exp = -6;
    }
    claims.quant.residual_exponent = -6;
    claims.quant.residual_bits = 16;
    claims.quant.rsqrt_kappa = 8;
    claims.quant.rms_eps_int = 1;
    claims.quant.exp_sigma = 12;
    for (_, scale) in &mut claims.quant.score_scales {
        *scale = ratio(1, 1);
    }
    for (_, exp) in &mut claims.silu_in_exponents {
        *exp = -6;
    }
    let points: Vec<QuantPoint> = claims
        .quant
        .shift_class_sets
        .iter()
        .map(|(p, _)| *p)
        .chain(claims.quant.affine_sites.iter().map(|(p, _)| *p))
        .collect();
    for point in points {
        make_pot(&mut claims, point, &[18]);
    }

    claims.gadget_policy.max_log2_rows = 32;
    claims.gadget_policy.limb_bits = 16;
    claims.gadget_policy.query_block = 4;

    edit_public(&mut claims);

    let public = check_public(&claims).expect("valid public part");
    let graph = tv_core::lowering::lower(&public, claims.arch.max_context).expect("lowers");
    let selected = tv_core::selection::derive(&graph, claims.gadget_policy).expect("selects");
    claims.tables = selected
        .tables
        .iter()
        .map(|t| {
            tv_tables::Table::generate(t.id, t.kind.clone())
                .expect("generates canonical table")
                .spec()
                .clone()
        })
        .collect();
    claims
}

fn mismatch_in(claims: &ClaimList, protocol: Protocol) -> StatementField {
    let error = match validate_in(claims, protocol) {
        Err(error) => error,
        Ok(_) => panic!("expected a statement mismatch in {protocol:?}, the claim list validated"),
    };
    assert!(error.is_rejection(), "{error:?}");
    assert_eq!(error.code(), "REJ-STATEMENT", "{error:?}");
    match error {
        ClaimsError::StatementMismatch { field } => field,
        other => panic!("expected a statement mismatch, got {other:?}"),
    }
}

fn assert_refused_in_protocol_2_only(claims: &ClaimList, what: &str) {
    assert_eq!(claims.protocol_version, PROTOCOL_VERSION_2, "{what}");
    assert_eq!(
        mismatch_in(claims, Protocol::Two),
        Layout,
        "{what} (validate_in)"
    );
    let envelope = seal(&claims.to_tce());
    match load_in(&envelope, &DecodeLimits::MESSAGE, Protocol::Two) {
        Err(ClaimsError::StatementMismatch { field: Layout }) => {}
        other => panic!("{what} (load_in): expected Layout mismatch, got {other:?}"),
    }

    let mut p1 = claims.clone();
    p1.protocol_version = PROTOCOL_VERSION;
    let v_default = validated(&p1);
    assert_eq!(v_default.protocol(), Protocol::One, "{what}");
    let v_explicit = validate_in(&p1, Protocol::One)
        .unwrap_or_else(|e| panic!("{what}: expected Protocol::One to accept: {e:?}"));
    assert_eq!(v_explicit.protocol(), Protocol::One, "{what}");
}

#[test]
fn protocol_aware_entry_points_check_each_protocols_version_tag() {
    assert_eq!(PROTOCOL_VERSION, 0);
    assert_eq!(PROTOCOL_VERSION_2, 2);

    // Protocol::One entry points (`validate`, `load`, and explicit `Protocol::One`) accept 0 and
    // keep rejecting `protocol_version = 2` at step 1 until #159 wires protocol/2 into the driver.
    let p1 = twin(SERVABLE);
    assert_eq!(validated(&p1).protocol(), Protocol::One);
    assert_eq!(
        validate_in(&p1, Protocol::One).unwrap().protocol(),
        Protocol::One
    );
    let p1_env = seal(&p1.to_tce());
    assert_eq!(
        load(&p1_env, &DecodeLimits::MESSAGE).unwrap().1.protocol(),
        Protocol::One
    );
    assert_eq!(
        load_in(&p1_env, &DecodeLimits::MESSAGE, Protocol::One)
            .unwrap()
            .1
            .protocol(),
        Protocol::One
    );
    for wrong in [1u16, PROTOCOL_VERSION_2, 3, u16::MAX] {
        let mut bad = p1.clone();
        bad.protocol_version = wrong;
        assert_eq!(mismatch(&bad), ProtocolVersion, "validate v={wrong}");
        assert_eq!(
            mismatch_in(&bad, Protocol::One),
            ProtocolVersion,
            "validate_in(One) v={wrong}"
        );
        let bad_env = seal(&bad.to_tce());
        assert!(matches!(
            load(&bad_env, &DecodeLimits::MESSAGE),
            Err(ClaimsError::StatementMismatch {
                field: ProtocolVersion
            })
        ));
        assert!(matches!(
            load_in(&bad_env, &DecodeLimits::MESSAGE, Protocol::One),
            Err(ClaimsError::StatementMismatch {
                field: ProtocolVersion
            })
        ));
    }

    // Protocol::Two entry points accept `PROTOCOL_VERSION_2` (2) when every committed matrix meets
    // `k_T >= k_W`, and refuse 0, 1, 3, and `u16::MAX` at step 1.
    let p2 = custom_protocol_2_claims(
        CustomArch {
            params: "live-64/1",
            profile: Profile::Servable,
            n_layers: 1,
            n_heads: 64,
            n_kv_heads: 64,
            d_head: 16,
            d_ff: 1_024,
            vocab: 1_024,
        },
        |_| {},
    );
    let v2 = validate_in(&p2, Protocol::Two).expect("wide protocol/2 claim list validates");
    assert_eq!(v2.protocol(), Protocol::Two);
    assert_eq!(v2.params(), ParamSet::Live64);
    let p2_env = seal(&p2.to_tce());
    let (opened, loaded_v2) =
        load_in(&p2_env, &DecodeLimits::MESSAGE, Protocol::Two).expect("sealed protocol/2 loads");
    assert_eq!(opened, p2);
    assert_eq!(loaded_v2.protocol(), Protocol::Two);

    for wrong in [PROTOCOL_VERSION, 1u16, 3, u16::MAX] {
        let mut bad = p2.clone();
        bad.protocol_version = wrong;
        assert_eq!(
            mismatch_in(&bad, Protocol::Two),
            ProtocolVersion,
            "validate_in(Two) v={wrong}"
        );
        let bad_env = seal(&bad.to_tce());
        assert!(matches!(
            load_in(&bad_env, &DecodeLimits::MESSAGE, Protocol::Two),
            Err(ClaimsError::StatementMismatch {
                field: ProtocolVersion
            })
        ));
    }
}

#[test]
fn protocol_2_admits_exact_minimum_width_architecture_matrices_and_exempts_uncommitted_channel_params()
 {
    // Exact boundary `k_T == k_W = 2^20` (`ρ'_T == ρ'_W`) across all nine architecture matrices at
    // both production parameter sets, with `m_CP == 0` (`16 · 1 · 1 = 16 < 2^20` exempt).
    for (params, expected_set) in [
        ("live-64/1", ParamSet::Live64),
        ("ni-100/1", ParamSet::Ni100),
    ] {
        let exact = custom_protocol_2_claims(
            CustomArch {
                params,
                profile: Profile::Servable,
                n_layers: 1,
                n_heads: 64,
                n_kv_heads: 64,
                d_head: 16,
                d_ff: 1_024,
                vocab: 1_024,
            },
            |_| {},
        );
        let v = validate_in(&exact, Protocol::Two).unwrap_or_else(|e| panic!("{params}: {e:?}"));
        assert_eq!(v.params(), expected_set);
        assert_eq!(v.protocol(), Protocol::Two);
    }

    // Non-power-of-two `d_ff = 513` and `vocab = 513` round up to `1024`, so `out_pad · in_pad =
    // 1024 · 1024 = 2^20 == k_W` even though unpadded `513 · 1024 < 2^20`.
    let padded_boundary = custom_protocol_2_claims(
        CustomArch {
            params: "live-64/1",
            profile: Profile::Servable,
            n_layers: 1,
            n_heads: 64,
            n_kv_heads: 64,
            d_head: 16,
            d_ff: 513,
            vocab: 513,
        },
        |_| {},
    );
    validate_in(&padded_boundary, Protocol::Two).expect("padded dimensions meet k_W");

    // Wide-input matrix (`in_pad = 2^21 > k_W = 2^20` on `Down` in `minimal`): `k_Down = 2^21`,
    // `n_Down = 2^23`, so `ρ'_Down = (2^21 + 2^16) / 2^23 < ρ'_W = (2^20 + 2^16) / 2^22`.
    let wide_input = custom_protocol_2_claims(
        CustomArch {
            params: "live-64/1",
            profile: Profile::Minimal,
            n_layers: 1,
            n_heads: 64,
            n_kv_heads: 64,
            d_head: 16,
            d_ff: 1 << 21,
            vocab: 1_024,
        },
        |_| {},
    );
    validate_in(&wide_input, Protocol::Two).expect("k_Down > k_W has strictly lower masked rate");
}

#[test]
fn protocol_2_refuses_narrow_architecture_matrices_with_layout_while_protocol_1_admits_them() {
    // RFC-0012's full-code forgery shape: `d_model = 128` at `live-64/1` gives `k_Q = 2^14` and
    // `n_Q = P_W = 2^16` (`k_Q + P_W > n_Q`, `ρ'_Q > 1`), while `Embed`, `Gate`, `Up`, `Down`,
    // and `LmHead` all have `128 · 8192 = 2^20 == k_W`.
    let forgery_d128 = custom_protocol_2_claims(
        CustomArch {
            params: "live-64/1",
            profile: Profile::Servable,
            n_layers: 1,
            n_heads: 8,
            n_kv_heads: 8,
            d_head: 16,
            d_ff: 8_192,
            vocab: 8_192,
        },
        |_| {},
    );
    assert_refused_in_protocol_2_only(&forgery_d128, "d_model = 128 full-code forgery shape");

    // `d_model = 512`, `d_ff = 2048`, `vocab = 2048`: `Q`, `K`, `V`, `O` have `k_T = 2^18`
    // (`n_T = 2^20`, `k_T + P_W < n_T` proper subcode, yet `ρ'_T > ρ'_W`).
    let narrow_attn = custom_protocol_2_claims(
        CustomArch {
            params: "ni-100/1",
            profile: Profile::Servable,
            n_layers: 1,
            n_heads: 32,
            n_kv_heads: 32,
            d_head: 16,
            d_ff: 2_048,
            vocab: 2_048,
        },
        |_| {},
    );
    assert_refused_in_protocol_2_only(&narrow_attn, "narrow attention projections at k_T = 2^18");

    // Only `K` and `V` narrow (`n_kv_heads = 32`, `kv = 512`, `k_K = k_V = 2^19 < 2^20`, while
    // `Q`, `O`, `Embed`, `Gate`, `Up`, `Down`, `LmHead` all have `1024 · 1024 = 2^20`).
    let narrow_kv = custom_protocol_2_claims(
        CustomArch {
            params: "live-64/1",
            profile: Profile::Servable,
            n_layers: 1,
            n_heads: 64,
            n_kv_heads: 32,
            d_head: 16,
            d_ff: 1_024,
            vocab: 1_024,
        },
        |_| {},
    );
    assert_refused_in_protocol_2_only(&narrow_kv, "narrow K and V projections alone");

    // Only `Embed` and `LmHead` narrow (`vocab = 512`, `k_Embed = k_LmHead = 2^19 < 2^20`, while
    // every layer projection has `1024 · 1024 = 2^20`).
    let narrow_vocab = custom_protocol_2_claims(
        CustomArch {
            params: "live-64/1",
            profile: Profile::Servable,
            n_layers: 1,
            n_heads: 64,
            n_kv_heads: 64,
            d_head: 16,
            d_ff: 1_024,
            vocab: 512,
        },
        |_| {},
    );
    assert_refused_in_protocol_2_only(&narrow_vocab, "narrow Embed and LmHead alone");

    // Only `Gate`, `Up`, and `Down` narrow (`d_ff = 512`, `k_Gate = k_Up = k_Down = 2^19 < 2^20`,
    // while `Embed`, `Q`, `K`, `V`, `O`, `LmHead` all have `1024 · 1024 = 2^20`).
    let narrow_mlp = custom_protocol_2_claims(
        CustomArch {
            params: "live-64/1",
            profile: Profile::Servable,
            n_layers: 1,
            n_heads: 64,
            n_kv_heads: 64,
            d_head: 16,
            d_ff: 512,
            vocab: 1_024,
        },
        |_| {},
    );
    assert_refused_in_protocol_2_only(&narrow_mlp, "narrow MLP projections alone");

    // Every production twin of the tiny golden models (`d_model = 64`) is refused in protocol/2.
    for name in [SERVABLE, MINIMAL, AFFINE, DEEP] {
        let mut claims = twin(name);
        claims.protocol_version = PROTOCOL_VERSION_2;
        assert_refused_in_protocol_2_only(&claims, name);
    }
    let mut live_tiny = golden(SERVABLE);
    live_tiny.params = id("live-64/1");
    live_tiny.layout = live_64_layout();
    live_tiny.protocol_version = PROTOCOL_VERSION_2;
    assert_refused_in_protocol_2_only(&live_tiny, "tiny-2-servable at live-64/1");
}

#[test]
fn protocol_2_checks_committed_channel_params_width_and_admits_its_boundary() {
    // All nine architecture matrices meet `k_T = 2^20`, while a single 2-class `pot` site commits
    // `ChannelParams` (`m_CP > 0`) at `16 · 1 · 1024 = 2^14 < 2^20`.
    let narrow_pot_cp = custom_protocol_2_claims(
        CustomArch {
            params: "live-64/1",
            profile: Profile::Servable,
            n_layers: 1,
            n_heads: 64,
            n_kv_heads: 64,
            d_head: 16,
            d_ff: 1_024,
            vocab: 1_024,
        },
        |c| make_pot(c, at(0, Site::RopeQ), &[18, 19]),
    );
    assert_refused_in_protocol_2_only(&narrow_pot_cp, "narrow 2-class pot ChannelParams");

    // An affine site with every `pot` site 1-class also commits `ChannelParams` (`m_CP > 0`).
    let narrow_affine_cp = custom_protocol_2_claims(
        CustomArch {
            params: "ni-100/1",
            profile: Profile::Servable,
            n_layers: 1,
            n_heads: 64,
            n_kv_heads: 64,
            d_head: 16,
            d_ff: 1_024,
            vocab: 1_024,
        },
        |c| make_affine(c, at(0, Site::V), 20),
    );
    assert_refused_in_protocol_2_only(&narrow_affine_cp, "narrow affine ChannelParams");

    // `C` is the maximum padded output width over weights that *have* `ChannelParams`, not over
    // 1-class `pot` weights: `d_ff = 32_768` on 1-class `Gate`/`Up` does not rescue `RopeK` (`K`
    // has `out = 1024`, so `C = 1024` and `16 · 2 · 1024 = 32_768 < 2^20`).
    let uncounted_wide_mlp = custom_protocol_2_claims(
        CustomArch {
            params: "live-64/1",
            profile: Profile::Servable,
            n_layers: 2,
            n_heads: 64,
            n_kv_heads: 64,
            d_head: 16,
            d_ff: 32_768,
            vocab: 1_024,
        },
        |c| {
            for layer in 0..2 {
                make_pot(c, at(layer, Site::RopeK), &[18, 19]);
            }
        },
    );
    assert_refused_in_protocol_2_only(
        &uncounted_wide_mlp,
        "1-class Gate/Up must not inflate ChannelParams C",
    );

    // Exact `L_pad` boundary on committed `ChannelParams` (`C = 1024`):
    // - `n_layers = 32` gives `L_pad = 32` and `16 · 32 · 1024 = 2^19 < 2^20` (refused);
    // - `n_layers = 33` gives `L_pad = 64` and `16 · 64 · 1024 = 2^20 == k_W` (admitted, even
    //   though unpadded `16 · 33 · 1024 = 540_672 < 2^20`).
    let cp_32_layers = custom_protocol_2_claims(
        CustomArch {
            params: "live-64/1",
            profile: Profile::Servable,
            n_layers: 32,
            n_heads: 64,
            n_kv_heads: 64,
            d_head: 16,
            d_ff: 1_024,
            vocab: 1_024,
        },
        |c| {
            for layer in 0..32 {
                make_pot(c, at(layer, Site::RopeQ), &[18, 19]);
            }
        },
    );
    assert_refused_in_protocol_2_only(&cp_32_layers, "n_layers = 32 gives k_CP = 2^19 < k_W");

    let cp_33_layers = custom_protocol_2_claims(
        CustomArch {
            params: "live-64/1",
            profile: Profile::Servable,
            n_layers: 33,
            n_heads: 64,
            n_kv_heads: 64,
            d_head: 16,
            d_ff: 1_024,
            vocab: 1_024,
        },
        |c| {
            for layer in 0..33 {
                make_pot(c, at(layer, Site::RopeQ), &[18, 19]);
            }
        },
    );
    validate_in(&cp_33_layers, Protocol::Two)
        .expect("n_layers = 33 pads L_pad to 64 and meets k_CP == k_W");

    // Multi-site `C = max` across `ChannelParams`-bearing weights, including a narrower site
    // (`MlpResidual`, `Down` `out = 1024`) after a wider non-power-of-two site (`Up` `out = 32_769`
    // -> `padded = 65_536`, giving `16 · 1 · 65_536 = 2^20 == k_W`).
    let cp_max_across_sites = custom_protocol_2_claims(
        CustomArch {
            params: "live-64/1",
            profile: Profile::Servable,
            n_layers: 1,
            n_heads: 64,
            n_kv_heads: 64,
            d_head: 16,
            d_ff: 32_769,
            vocab: 1_024,
        },
        |c| {
            make_pot(c, at(0, Site::RopeK), &[18, 19]);
            make_affine(c, at(0, Site::Up), 24);
            make_affine(c, at(0, Site::MlpResidual), 24);
        },
    );
    validate_in(&cp_max_across_sites, Protocol::Two)
        .expect("C takes max padded out across all ChannelParams-bearing sites");
}

#[cfg(feature = "test-params")]
#[test]
fn every_golden_claim_list_validates_under_protocol_2_at_test_small() {
    // RFC-0012: all four conformance models at `test-small/1` (`k_W = 256, P_W = 64`) satisfy
    // `k_T >= k_W` on every committed matrix (`Embed`..`ChannelParams`).
    for g in goldens() {
        let mut claims = ClaimList::open(&g.envelope, &DecodeLimits::MESSAGE).unwrap();
        claims.protocol_version = PROTOCOL_VERSION_2;
        let v = validate_in(&claims, Protocol::Two).unwrap_or_else(|e| panic!("{}: {e:?}", g.name));
        assert_eq!(v.protocol(), Protocol::Two, "{}", g.name);
        assert_eq!(v.params(), ParamSet::TestSmall, "{}", g.name);

        let resealed = seal(&claims.to_tce());
        let (_, loaded) = load_in(&resealed, &DecodeLimits::MESSAGE, Protocol::Two)
            .unwrap_or_else(|e| panic!("{} load_in: {e:?}", g.name));
        assert_eq!(loaded.protocol(), Protocol::Two, "{}", g.name);
    }
}

#[cfg(feature = "test-params")]
#[test]
fn test_small_protocol_2_refuses_narrow_architecture_and_channel_params_matrices() {
    // At `test-small/1` (`k_W = 256, P_W = 64`): `d_model = 16, d_ff = 16, vocab = 16` meets
    // `k_T = 256 == k_W` when `m_CP == 0`, while `vocab = 8` drops `Embed` and `LmHead` to `128`.
    let at_bound = custom_protocol_2_claims(
        CustomArch {
            params: "test-small/1",
            profile: Profile::Servable,
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            d_head: 16,
            d_ff: 16,
            vocab: 16,
        },
        |_| {},
    );
    validate_in(&at_bound, Protocol::Two).expect("exact k_T == 256 at test-small/1");

    let narrow_arch = custom_protocol_2_claims(
        CustomArch {
            params: "test-small/1",
            profile: Profile::Servable,
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            d_head: 16,
            d_ff: 16,
            vocab: 8,
        },
        |_| {},
    );
    assert_refused_in_protocol_2_only(&narrow_arch, "test-small narrow Embed/LmHead");

    // `d_head = 2, n_heads = 64, d_model = 128, n_kv_heads = 1` (`kv = 2`, `k_K = k_V = 256 == k_W`),
    // `d_ff = 128, vocab = 128`, with `RopeK` 2-class `pot`: every architecture matrix meets `k_W`,
    // while `ChannelParams` has `C = 2` and `k_CP = 16 · 1 · 2 = 32 < P_W = 64 < k_W = 256`.
    let narrow_cp = custom_protocol_2_claims(
        CustomArch {
            params: "test-small/1",
            profile: Profile::Servable,
            n_layers: 1,
            n_heads: 64,
            n_kv_heads: 1,
            d_head: 2,
            d_ff: 128,
            vocab: 128,
        },
        |c| make_pot(c, at(0, Site::RopeK), &[18, 19]),
    );
    assert_refused_in_protocol_2_only(&narrow_cp, "test-small narrow ChannelParams (k_CP < P_W)");
}

#[test]
fn protocol_2_checks_precede_step_6s_masked_weight_rate_in_rfc_0039_order() {
    // Base narrow protocol/2 claim list: fails at step 6 (`Layout`) alone.
    let mut base = twin(SERVABLE);
    base.protocol_version = PROTOCOL_VERSION_2;
    assert_eq!(mismatch_in(&base, Protocol::Two), Layout);

    // RFC-0019 step 2 (`Params`) precedes step 6.
    let mut bad_params = base.clone();
    bad_params.params = id("ni-101/1");
    assert!(matches!(
        validate_in(&bad_params, Protocol::Two),
        Err(ClaimsError::Params { name }) if name.as_str() == "ni-101/1"
    ));

    // Step 1 (`ProtocolVersion`) precedes step 6.
    let mut bad_version = base.clone();
    bad_version.protocol_version = PROTOCOL_VERSION;
    assert_eq!(mismatch_in(&bad_version, Protocol::Two), ProtocolVersion);

    // Step 2.1 (`Architecture`) precedes step 6.
    let mut bad_arch = base.clone();
    bad_arch.arch.n_kv_heads = 3;
    assert_eq!(mismatch_in(&bad_arch, Protocol::Two), Architecture);

    // Step 2.2 (`Quantization`) precedes step 6.
    let mut bad_quant = base.clone();
    bad_quant.quant.rms_eps_int = 0;
    assert_eq!(mismatch_in(&bad_quant, Protocol::Two), Quantization);

    // Step 2.3 (`Rescaling`) precedes step 6.
    let mut bad_rescale = base.clone();
    make_affine(&mut bad_rescale, at(0, Site::V), 20);
    assert_eq!(
        mismatch_in(&bad_rescale, Protocol::Two),
        StatementField::Rescaling { site: Site::V }
    );

    // Step 3 (`Profile`) precedes step 6.
    let mut bad_profile = base.clone();
    bad_profile.arch.d_ff = 132_105;
    assert_eq!(mismatch_in(&bad_profile, Protocol::Two), ProfileField);

    // Step 4 (`Table`) precedes step 6.
    let mut bad_table = base.clone();
    bad_table.tables[0].digest[0] ^= 1;
    assert_eq!(
        mismatch_in(&bad_table, Protocol::Two),
        StatementField::Table { table: TableId(0) }
    );

    // Step 5.1 (`GadgetPolicy`) precedes step 6.
    let mut bad_policy = base.clone();
    bad_policy.gadget_policy.max_log2_rows = 40;
    assert_eq!(mismatch_in(&bad_policy, Protocol::Two), GadgetPolicy);

    // Step 5.2 (`Gadget`) precedes step 6.
    let mut bad_gadget = base;
    bad_gadget.gadget_policy.limb_bits = 0;
    assert_eq!(
        mismatch_in(&bad_gadget, Protocol::Two),
        StatementField::Gadget {
            gadget: tv_core::validation::GadgetId(1)
        }
    );
}

#[test]
fn overwide_weight_matrix_codeword_past_two_to_32_is_refused_as_layout_in_both_protocols() {
    use tv_core::weights::{self, Matrix};

    // 1. Narrow `Profile::Minimal` reproducer (`d_model = 2, vocab = 2, max_context = 1`):
    //    `d_ff = 2^30` (`in_pad = 2^30`, `n_Down = 2^32` at rate 1/4) is admitted in `Protocol::One`
    //    and derives all ten weight layouts cleanly; `d_ff = 2^30 + 1` (`in_pad = 2^31`,
    //    `n_Down = 2^33 > 2^32`) exceeds Goldilocks's 2-adic root of unity and must be rejected at
    //    step 6 with `StatementField::Layout` rather than escaping to `weights::derive` (`SP-8`).
    let build_narrow_minimal = |d_ff: u32| {
        custom_protocol_2_claims(
            CustomArch {
                params: "live-64/1",
                profile: Profile::Minimal,
                n_layers: 1,
                n_heads: 1,
                n_kv_heads: 1,
                d_head: 2,
                d_ff,
                vocab: 2,
            },
            |c| {
                c.arch.max_context = 1;
                make_pot(c, at(0, Site::Silu), &[10]);
                make_pot(c, at(0, Site::MlpResidual), &[41]);
            },
        )
    };

    let mut narrow_at_ceiling_p1 = build_narrow_minimal(1 << 30);
    narrow_at_ceiling_p1.protocol_version = PROTOCOL_VERSION;
    assert_eq!(validated(&narrow_at_ceiling_p1).protocol(), Protocol::One);
    let derived_narrow =
        weights::derive(&narrow_at_ceiling_p1, 0).expect("n_Down == 2^32 derives weight layouts");
    assert_eq!(derived_narrow[usize::from(Matrix::Down.tag())].log2_k(), 30);

    let narrow_past_ceiling_p2 = build_narrow_minimal((1 << 30) + 1);
    let mut narrow_past_ceiling_p1 = narrow_past_ceiling_p2.clone();
    narrow_past_ceiling_p1.protocol_version = PROTOCOL_VERSION;
    assert!(
        weights::derive(&narrow_past_ceiling_p1, 0).is_err(),
        "n_Down = 2^33 has no 2-adic coset in Goldilocks"
    );
    assert_eq!(mismatch(&narrow_past_ceiling_p1), Layout);
    assert_eq!(mismatch_in(&narrow_past_ceiling_p1, Protocol::One), Layout);
    assert_eq!(mismatch_in(&narrow_past_ceiling_p2, Protocol::Two), Layout);
    let narrow_p1_env = seal(&narrow_past_ceiling_p1.to_tce());
    assert!(matches!(
        load(&narrow_p1_env, &DecodeLimits::MESSAGE),
        Err(ClaimsError::StatementMismatch { field: Layout })
    ));
    assert!(matches!(
        load_in(&narrow_p1_env, &DecodeLimits::MESSAGE, Protocol::One),
        Err(ClaimsError::StatementMismatch { field: Layout })
    ));
    let narrow_p2_env = seal(&narrow_past_ceiling_p2.to_tce());
    assert!(matches!(
        load_in(&narrow_p2_env, &DecodeLimits::MESSAGE, Protocol::Two),
        Err(ClaimsError::StatementMismatch { field: Layout })
    ));

    // Earlier RFC-0039 steps (e.g. step 5.1 `GadgetPolicy`) still precede step 6's `2^32` codeword
    // ceiling in `Protocol::One`.
    let mut bad_policy_p1 = narrow_past_ceiling_p1;
    bad_policy_p1.gadget_policy.max_log2_rows = 40;
    assert_eq!(mismatch(&bad_policy_p1), GadgetPolicy);

    // 2. Across both production parameter sets (`live-64/1` and `ni-100/1`) at `d_model = 8` (the
    //    widest `d_model` whose `Profile::Minimal` `Down` accumulator stays inside RFC-0006's
    //    62-bit signed integer gadget domain at `d_ff = 2^30 + 1`) and `vocab = 2^18`: `d_ff = 2^30`
    //    (`n_Down = 2^32`) is admitted by `Protocol::One` and `weights::derive`, whereas
    //    `d_ff = 2^30 + 1` (`n_Down = 2^33 > 2^32`) is refused with `StatementField::Layout` in
    //    both `Protocol::One` and `Protocol::Two` (never `Ok` and never `ClaimsError::Internal`).
    for params in ["live-64/1", "ni-100/1"] {
        let build_minimal = |d_ff: u32| {
            custom_protocol_2_claims(
                CustomArch {
                    params,
                    profile: Profile::Minimal,
                    n_layers: 1,
                    n_heads: 4,
                    n_kv_heads: 4,
                    d_head: 2,
                    d_ff,
                    vocab: 1 << 18,
                },
                |c| {
                    c.arch.max_context = 1;
                    c.quant.residual_bits = 8;
                    make_pot(c, at(0, Site::Silu), &[12]);
                    make_pot(c, at(0, Site::MlpResidual), &[43]);
                },
            )
        };

        let mut at_ceiling_p1 = build_minimal(1 << 30);
        at_ceiling_p1.protocol_version = PROTOCOL_VERSION;
        assert_eq!(validated(&at_ceiling_p1).protocol(), Protocol::One);
        let derived = weights::derive(&at_ceiling_p1, 0)
            .unwrap_or_else(|e| panic!("{params} n_Down == 2^32 derives weight layouts: {e:?}"));
        assert_eq!(derived[usize::from(Matrix::Down.tag())].log2_k(), 30);

        let past_ceiling_p2 = build_minimal((1 << 30) + 1);
        assert_eq!(
            mismatch_in(&past_ceiling_p2, Protocol::Two),
            Layout,
            "{params} p2 past 2^32"
        );
        let p2_env = seal(&past_ceiling_p2.to_tce());
        assert!(matches!(
            load_in(&p2_env, &DecodeLimits::MESSAGE, Protocol::Two),
            Err(ClaimsError::StatementMismatch { field: Layout })
        ));

        let mut past_ceiling_p1 = past_ceiling_p2;
        past_ceiling_p1.protocol_version = PROTOCOL_VERSION;
        assert!(weights::derive(&past_ceiling_p1, 0).is_err());
        assert_eq!(mismatch(&past_ceiling_p1), Layout, "{params} p1");
        assert_eq!(
            mismatch_in(&past_ceiling_p1, Protocol::One),
            Layout,
            "{params} p1 explicit"
        );
        let p1_env = seal(&past_ceiling_p1.to_tce());
        assert!(matches!(
            load(&p1_env, &DecodeLimits::MESSAGE),
            Err(ClaimsError::StatementMismatch { field: Layout })
        ));
        assert!(matches!(
            load_in(&p1_env, &DecodeLimits::MESSAGE, Protocol::One),
            Err(ClaimsError::StatementMismatch { field: Layout })
        ));
    }
}
