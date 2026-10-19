// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public metadata inputs for the differential harness. Production-size cases contain no
//! weight matrices or activations; they describe shapes and quantization only.
use serde_json::Value;
use tessera_codec::{DecodeLimits, Rational, envelope, schema};
use tessera_ir::{ClaimList, Profile, QuantPoint, Site};

pub struct Model {
    pub name: String,
    pub envelope: Vec<u8>,
    pub contexts: Vec<u32>,
}
pub fn unhex(s: &str) -> Vec<u8> {
    s.as_bytes()
        .chunks_exact(2)
        .map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
        .collect()
}
fn site(name: &str) -> Site {
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
        _ => panic!("unknown public site"),
    }
}
fn variant(c: &mut ClaimList, v: &Value) {
    if let Some(kv) = v.get("n_kv_heads") {
        c.arch.n_kv_heads = kv.as_u64().unwrap() as u32;
    }
    if let Some(edits) = v.get("rescale") {
        for edit in edits.as_array().unwrap() {
            let point = QuantPoint::Layer {
                layer: edit[0].as_u64().unwrap() as u32,
                site: site(edit[1].as_str().unwrap()),
            };
            c.quant.shift_class_sets.retain(|(p, _)| *p != point);
            c.quant.affine_sites.retain(|(p, _)| *p != point);
            if edit[2] == "affine" {
                c.quant
                    .affine_sites
                    .push((point, edit[3].as_i64().unwrap() as i8));
            } else {
                assert_eq!(edit[2], "pot");
                c.quant.shift_class_sets.push((
                    point,
                    edit[3]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_i64().unwrap() as i8)
                        .collect(),
                ));
            }
        }
    }
    c.quant.affine_sites.sort_by_key(|(p, _)| *p);
    c.quant.shift_class_sets.sort_by_key(|(p, _)| *p);
    if let Some(edits) = v.get("score_scales") {
        for row in edits.as_array().unwrap() {
            let layer = row[0].as_u64().unwrap() as u32;
            c.quant
                .score_scales
                .iter_mut()
                .find(|(l, _)| *l == layer)
                .unwrap()
                .1 = Rational::new(row[1].as_i64().unwrap(), row[2].as_u64().unwrap()).unwrap();
        }
    }
}
fn shape(base: &ClaimList, large: bool, profile: Profile) -> ClaimList {
    let mut c = base.clone();
    let (layers, d, heads, ff) = if large {
        (126, 16_384, 128, 53_248)
    } else {
        (32, 4096, 32, 14_336)
    };
    c.profile = profile;
    c.arch.n_layers = layers;
    c.arch.d_model = d;
    c.arch.n_heads = heads;
    c.arch.n_kv_heads = 8;
    c.arch.d_head = 128;
    c.arch.d_ff = ff;
    c.arch.vocab = 128_256;
    c.arch.max_context = 131_072;
    c.arch.rms_eps = Rational::new(1, 100_000).unwrap();
    let q = &mut c.quant;
    q.act_exponents = vec![(QuantPoint::Embed, -6)];
    q.shift_class_sets.clear();
    q.affine_sites.clear();
    q.residual_exponent = -7;
    q.residual_bits = 20;
    q.rsqrt_kappa = 30;
    q.rms_eps_int = (u64::from(d) * (1 << 14)).div_ceil(100_000);
    q.exp_sigma = 16;
    q.score_scales.clear();
    q.logits_bits = 28;
    q.logits_exponent = -11;
    for l in 0..layers {
        for s in [
            Site::AttnNorm,
            Site::RopeQ,
            Site::RopeK,
            Site::V,
            Site::AttnOut,
            Site::AttnResidual,
            Site::MlpNorm,
            Site::Silu,
            Site::Up,
            Site::Product,
            Site::MlpResidual,
        ] {
            if profile == Profile::Minimal && matches!(s, Site::AttnOut | Site::Up | Site::Product)
            {
                continue;
            }
            let point = QuantPoint::Layer { layer: l, site: s };
            if !matches!(s, Site::AttnResidual | Site::MlpResidual) {
                q.act_exponents.push((point, -3));
            }
            let r = match s {
                Site::AttnNorm | Site::MlpNorm => 27,
                Site::RopeQ | Site::RopeK => 22,
                Site::V | Site::Up => 8,
                Site::AttnOut => 32,
                Site::AttnResidual if profile == Profile::Minimal => 37,
                Site::MlpResidual if profile == Profile::Minimal => 17,
                Site::AttnResidual | Site::MlpResidual => 4,
                Site::Silu => 20,
                Site::Product => 5,
            };
            q.shift_class_sets.push((point, vec![r]));
        }
        q.score_scales.push((l, Rational::new(1, 1 << 14).unwrap()));
    }
    q.act_exponents.push((QuantPoint::FinalNorm, -3));
    q.shift_class_sets.push((QuantPoint::FinalNorm, vec![27]));
    q.rope = c.arch.rope;
    c.silu_in_exponents = (0..layers).map(|l| (l, -6)).collect();
    c.gadget_policy.max_log2_rows = 32;
    c.gadget_policy.limb_bits = 16;
    c.gadget_policy.query_block = 512;
    // This is a metadata input to the lowerers, not a complete validated production claim list.
    // Its lookup-table digests would require constructing the production model's artifacts.
    c.tables.clear();
    c
}

pub fn models() -> Vec<Model> {
    let golden: Value = serde_json::from_str(include_str!(
        "../../../../spec/vectors/claims/claim-lists.json"
    ))
    .unwrap();
    let oracle: Value = serde_json::from_str(include_str!(
        "../../../../spec/vectors/lowering/lowering.json"
    ))
    .unwrap();
    let mut models = Vec::new();
    let open = |row: &Value| {
        envelope::open::<ClaimList>(
            &unhex(row["hex"].as_str().unwrap()),
            &schema::CLAIM_LIST,
            DecodeLimits::MESSAGE,
        )
        .unwrap()
    };
    for case in oracle["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["n"] == 1)
    {
        let source = golden["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|g| g["model"] == case["file"])
            .unwrap();
        let mut c = open(source);
        if let Some(v) = case.get("variant") {
            variant(&mut c, v);
        }
        models.push(Model {
            name: case["model"].as_str().unwrap().to_owned(),
            contexts: vec![1, 4, 5, c.arch.max_context],
            envelope: envelope::seal(&schema::CLAIM_LIST, &c),
        });
    }
    let base = open(&golden["cases"][0]);
    for large in [false, true] {
        for profile in [Profile::Servable, Profile::Minimal] {
            let c = shape(&base, large, profile);
            let name = format!(
                "llama-3.1-{}-{}",
                if large { "405b" } else { "8b" },
                if profile == Profile::Servable {
                    "servable"
                } else {
                    "minimal"
                }
            );
            models.push(Model {
                name,
                contexts: vec![1, 1024, 16_384, c.arch.max_context],
                envelope: envelope::seal(&schema::CLAIM_LIST, &c),
            });
        }
    }
    models
}
