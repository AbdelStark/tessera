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
    ni_100_layout, pot_index, test_small_layout, twin, validated,
};
use tv_codec::{DecodeLimits, Rational};
use tv_core::claims::{ClaimList, Profile, QuantPoint, Site};
use tv_core::params::ParamSet;
use tv_core::validation::{
    ClaimsError, PROTOCOL_VERSION, StatementField, check_public, load, validate,
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
    // Table 6 is a `Requant` table: another clamp.
    let mut claims = twin(SERVABLE);
    let TableKind::Requant { lo, .. } = &mut claims.tables[6].kind else {
        panic!("table 6 is a Requant table");
    };
    *lo = -127;
    assert_eq!(
        mismatch(&claims),
        StatementField::Table { table: TableId(6) }
    );
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
