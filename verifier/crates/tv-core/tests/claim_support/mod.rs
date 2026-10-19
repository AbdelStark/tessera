// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Test support for the claim-list tests: the golden claim lists of `spec/vectors/claims/`,
//! their production twins, resealed payloads, and readers for the vectors' JSON.
//!
//! The golden lists name `test-small/1`, which only a build with `test-params` accepts
//! (RFC-0001 'Cargo features'). A list's **production twin** is the same list naming
//! `ni-100/1`, with the commitment layout RFC-0026 gives that set: it differs from the golden
//! list only where the parameter set is named, so a check that breaks one rule of the twin is
//! seen in every build, the default one included.

#![allow(dead_code)]

use tv_codec::{DIGEST_CONTEXT, DecodeLimits, HEADER_LEN, MAGIC, Rational};
use tv_core::claims::{
    AdviceLayoutParams, ClaimList, CommitmentLayout, ParamSetId, QuantPoint, Site,
    WeightLayoutParams,
};
use tv_core::validation::{ClaimsError, StatementField, Validated, validate};
use tv_testkit::{Json, blake3, hex, read_vectors};

/// A golden claim list: its name, its model's artifact, its sealed envelope and its digest.
pub struct Golden {
    pub name: String,
    pub model: String,
    pub envelope: Vec<u8>,
    pub digest: [u8; 32],
}

/// The member `key` of a vector object.
pub fn member<'a>(object: &'a Json, key: &str) -> &'a Json {
    object
        .get(key)
        .unwrap_or_else(|| panic!("`{key}` is missing"))
}

/// The member `key` of a vector object, which must be a string.
pub fn text<'a>(object: &'a Json, key: &str) -> &'a str {
    member(object, key)
        .as_str()
        .unwrap_or_else(|| panic!("`{key}` is not a string"))
}

/// The member `key` of a vector object, which must be an array.
pub fn list<'a>(object: &'a Json, key: &str) -> &'a [Json] {
    member(object, key)
        .as_array()
        .unwrap_or_else(|| panic!("`{key}` is not an array"))
}

/// The elements of a JSON array.
pub fn items(value: &Json) -> &[Json] {
    value
        .as_array()
        .unwrap_or_else(|| panic!("{value:?} is not an array"))
}

/// A JSON integer of any sign that fits `i128`.
pub fn integer(value: &Json) -> i128 {
    match value {
        Json::Number(n) => n.parse().unwrap_or_else(|e| panic!("{n}: {e}")),
        other => panic!("{other:?} is not an integer"),
    }
}

/// A JSON string.
pub fn string(value: &Json) -> &str {
    value
        .as_str()
        .unwrap_or_else(|| panic!("{value:?} is not a string"))
}

/// Every golden claim list, in file order.
pub fn goldens() -> Vec<Golden> {
    let file = read_vectors("claims/claim-lists.json");
    assert_eq!(integer(member(&file, "schema")), 0x0201);
    assert_eq!(integer(member(&file, "version")), 1);
    assert_eq!(text(&file, "params"), "test-small/1");
    list(&file, "cases")
        .iter()
        .map(|case| Golden {
            name: text(case, "name").to_owned(),
            model: text(case, "model").to_owned(),
            envelope: hex(text(case, "hex")),
            digest: hex(text(case, "digest"))
                .try_into()
                .expect("a 32-byte digest"),
        })
        .collect()
}

/// The golden claim list `name`, decoded.
pub fn golden(name: &str) -> ClaimList {
    let golden = goldens()
        .into_iter()
        .find(|g| g.name == name)
        .unwrap_or_else(|| panic!("no golden claim list is named {name}"));
    ClaimList::open(&golden.envelope, &DecodeLimits::MESSAGE)
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// The layout RFC-0026 gives `ni-100/1`: advice 22, 1, 200, 32, 8; weights 20, 2, 65,536, 4,096.
pub fn ni_100_layout() -> CommitmentLayout {
    CommitmentLayout {
        advice: AdviceLayoutParams {
            log2_k: 22,
            log2_inv_rate: 1,
            mask_coeffs: 200,
            row_block: 32,
            segments: 8,
        },
        weights: WeightLayoutParams {
            log2_k: 20,
            log2_inv_rate: 2,
            mask_coeffs: 65_536,
            blinding_rows: 4_096,
        },
    }
}

/// The layout RFC-0026 gives `live-64/1`: advice 22, 1, 160, 32, 8; weights as `ni-100/1`'s.
pub fn live_64_layout() -> CommitmentLayout {
    let mut layout = ni_100_layout();
    layout.advice.mask_coeffs = 160;
    layout
}

/// The layout RFC-0026 gives `test-small/1`: advice 10, 1, 16, 4, 1; weights 8, 2, 64, 32.
pub fn test_small_layout() -> CommitmentLayout {
    CommitmentLayout {
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
    }
}

/// A parameter-set identifier from printable ASCII.
pub fn id(name: &str) -> ParamSetId {
    ParamSetId::new(name).unwrap_or_else(|| panic!("{name} is not printable ASCII"))
}

/// The golden claim list `name` naming `ni-100/1`, with that set's layout.
pub fn twin(name: &str) -> ClaimList {
    let mut claims = golden(name);
    claims.params = id("ni-100/1");
    claims.layout = ni_100_layout();
    claims
}

/// `validate`'s rejection of `claims` as a statement mismatch, or a panic naming what it gave.
pub fn mismatch(claims: &ClaimList) -> StatementField {
    match validate(claims) {
        Err(ClaimsError::StatementMismatch { field }) => field,
        Err(other) => panic!("expected a statement mismatch, got {other:?}"),
        Ok(_) => panic!("expected a statement mismatch, the claim list validated"),
    }
}

/// `validate`'s result, which must be a success.
pub fn validated(claims: &ClaimList) -> Validated {
    validate(claims).unwrap_or_else(|e| panic!("expected the claim list to validate: {e:?}"))
}

/// The layer point `(layer, site)`.
pub fn at(layer: u32, site: Site) -> QuantPoint {
    QuantPoint::Layer { layer, site }
}

/// The position of `point` in `shift_class_sets`.
pub fn pot_index(claims: &ClaimList, point: QuantPoint) -> usize {
    claims
        .quant
        .shift_class_sets
        .iter()
        .position(|(p, _)| *p == point)
        .unwrap_or_else(|| panic!("{point:?} is not a pot site"))
}

/// The position of `point` in `affine_sites`.
pub fn affine_index(claims: &ClaimList, point: QuantPoint) -> usize {
    claims
        .quant
        .affine_sites
        .iter()
        .position(|(p, _)| *p == point)
        .unwrap_or_else(|| panic!("{point:?} is not an affine site"))
}

/// Makes `point` a `pot` site of `shifts`, keeping both lists sorted.
pub fn make_pot(claims: &mut ClaimList, point: QuantPoint, shifts: &[i8]) {
    claims.quant.affine_sites.retain(|(p, _)| *p != point);
    claims.quant.shift_class_sets.retain(|(p, _)| *p != point);
    let at = claims
        .quant
        .shift_class_sets
        .partition_point(|(p, _)| *p < point);
    claims
        .quant
        .shift_class_sets
        .insert(at, (point, shifts.to_vec()));
}

/// Makes `point` an affine site of shift `r`, keeping both lists sorted.
pub fn make_affine(claims: &mut ClaimList, point: QuantPoint, r: i8) {
    claims.quant.affine_sites.retain(|(p, _)| *p != point);
    claims.quant.shift_class_sets.retain(|(p, _)| *p != point);
    let at = claims
        .quant
        .affine_sites
        .partition_point(|(p, _)| *p < point);
    claims.quant.affine_sites.insert(at, (point, r));
}

/// The claim list of `layers` layers whose layer `ℓ` rescales as layer `ℓ mod L` of `claims`
/// does: every per-layer entry repeated, each list kept sorted. Its layers stay one template
/// (RFC-0038), so the lowering accepts it; its tables are not the lowering's, and only its
/// public part is meant to be read.
pub fn with_layers(claims: &ClaimList, layers: u32) -> ClaimList {
    let mut out = claims.clone();
    let source = claims.arch.n_layers;
    out.arch.n_layers = layers;
    let remap = |point: QuantPoint, to: u32| match point {
        QuantPoint::Layer { site, .. } => QuantPoint::Layer { layer: to, site },
        other => other,
    };
    let repeat = |entries: &[(QuantPoint, i8)]| -> Vec<(QuantPoint, i8)> {
        let mut out: Vec<(QuantPoint, i8)> = entries
            .iter()
            .filter(|(p, _)| !matches!(p, QuantPoint::Layer { .. }))
            .copied()
            .collect();
        for to in 0..layers {
            for &(p, v) in entries {
                if let QuantPoint::Layer { layer, .. } = p
                    && layer == to % source
                {
                    out.push((remap(p, to), v));
                }
            }
        }
        out.sort_by_key(|(p, _)| *p);
        out
    };
    out.quant.act_exponents = repeat(&claims.quant.act_exponents);
    out.quant.affine_sites = repeat(&claims.quant.affine_sites);
    let mut sets = Vec::new();
    for to in 0..layers {
        for (p, shifts) in &claims.quant.shift_class_sets {
            if let QuantPoint::Layer { layer, .. } = *p
                && layer == to % source
            {
                sets.push((remap(*p, to), shifts.clone()));
            }
        }
    }
    sets.extend(
        claims
            .quant
            .shift_class_sets
            .iter()
            .filter(|(p, _)| !matches!(p, QuantPoint::Layer { .. }))
            .cloned(),
    );
    sets.sort_by_key(|(p, _)| *p);
    out.quant.shift_class_sets = sets;
    out.quant.score_scales = per_layer(&claims.quant.score_scales, layers, source);
    out.silu_in_exponents = per_layer(&claims.silu_in_exponents, layers, source);
    out
}

/// Layer `ℓ`'s entry for each of `layers` layers, layer `ℓ mod source`'s of `entries`.
fn per_layer<T: Copy>(entries: &[(u32, T)], layers: u32, source: u32) -> Vec<(u32, T)> {
    (0..layers)
        .map(|to| {
            let (_, v) = entries.iter().find(|(l, _)| *l == to % source).unwrap();
            (to, *v)
        })
        .collect()
}

/// An envelope of schema `0x0201`, version 1, around `payload`, its digest BLAKE3
/// `derive_key("tessera/v1/envelope")` over the header and payload, from the test kit's
/// reference BLAKE3 (RFC-0003 'Envelope').
pub fn seal(payload: &[u8]) -> Vec<u8> {
    seal_as(0x0201, 1, payload)
}

/// An envelope of `schema_id` and `version` around `payload`.
pub fn seal_as(schema_id: u16, version: u16, payload: &[u8]) -> Vec<u8> {
    let mut envelope = Vec::with_capacity(payload.len() + 52);
    envelope.extend_from_slice(&MAGIC);
    envelope.extend_from_slice(&1u16.to_le_bytes());
    envelope.extend_from_slice(&schema_id.to_le_bytes());
    envelope.extend_from_slice(&version.to_le_bytes());
    envelope.extend_from_slice(&0u16.to_le_bytes());
    let len = u64::try_from(payload.len()).expect("a length");
    envelope.extend_from_slice(&len.to_le_bytes());
    envelope.extend_from_slice(payload);
    let digest = blake3::derive_key(DIGEST_CONTEXT, &envelope);
    envelope.extend_from_slice(&digest);
    envelope
}

/// The payload of a sealed envelope: the bytes between its header and its digest.
pub fn payload(envelope: &[u8]) -> &[u8] {
    let header = usize::try_from(HEADER_LEN).expect("20");
    &envelope[header..envelope.len() - 32]
}

/// An integer vector field checked against its destination type.
pub fn number<T: TryFrom<i128>>(value: &Json) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(integer(value)).expect("in range")
}

fn site_named(name: &str) -> Site {
    match name {
        "AttnNorm" => Site::AttnNorm,
        "RopeQ" => Site::RopeQ,
        "RopeK" => Site::RopeK,
        "V" => Site::V,
        "AttnOut" => Site::AttnOut,
        "AttnResidual" => Site::AttnResidual,
        "MlpNorm" => Site::MlpNorm,
        "Silu" => Site::Silu,
        "Up" => Site::Up,
        "Product" => Site::Product,
        "MlpResidual" => Site::MlpResidual,
        other => panic!("unknown site {other}"),
    }
}

/// The golden claim list associated with an oracle model artifact.
pub fn claims_of(file: &str) -> ClaimList {
    let golden = goldens()
        .into_iter()
        .find(|g| g.model == file)
        .unwrap_or_else(|| panic!("no golden claim list for {file}"));
    ClaimList::open(&golden.envelope, &DecodeLimits::MESSAGE).expect("a golden claim list")
}

/// Makes a variant's edits to the claim list's public part, each in every layer it names.
pub fn apply_variant(variant: &Json, claims: &mut ClaimList) {
    if let Some(heads) = variant.get("n_kv_heads") {
        claims.arch.n_kv_heads = number(heads);
    }
    if let Some(rescales) = variant.get("rescale") {
        for edit in items(rescales) {
            let [layer, site, mode, value] = items(edit) else {
                panic!("a rescale is [layer, site, mode, value]");
            };
            let point = at(number(layer), site_named(string(site)));
            match string(mode) {
                "pot" => {
                    let shifts: Vec<i8> = items(value).iter().map(number).collect();
                    make_pot(claims, point, &shifts);
                }
                "affine" => make_affine(claims, point, number(value)),
                other => panic!("unknown mode {other}"),
            }
        }
    }
    if let Some(scales) = variant.get("score_scales") {
        for edit in items(scales) {
            let [layer, num, den] = items(edit) else {
                panic!("a score scale is [layer, num, den]");
            };
            let layer: u32 = number(layer);
            let entry = claims
                .quant
                .score_scales
                .iter_mut()
                .find(|(l, _)| *l == layer)
                .expect("the layer's scale");
            entry.1 = Rational::new(number(num), number(den)).expect("in lowest terms");
        }
    }
}
