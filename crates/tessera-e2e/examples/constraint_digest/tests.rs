// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Diagnostic collector conformance before comparison with the compiler (RFC-0007, DEP-2).
use super::{compiler, fixtures, report, verifier};
use serde_json::{Value, json};
use tv_codec::DecodeLimits;
use tv_core::claims::ClaimList;

fn hex(s: &str) -> Vec<u8> {
    s.as_bytes()
        .chunks_exact(2)
        .map(|c| u8::from_str_radix(std::str::from_utf8(c).unwrap(), 16).unwrap())
        .collect()
}
#[test]
fn verifier_collector_matches_independent_graph_oracle_before_compiler_comparison() {
    let gold: Value = serde_json::from_str(include_str!(
        "../../../../spec/vectors/claims/claim-lists.json"
    ))
    .unwrap();
    let oracle: Value = serde_json::from_str(include_str!(
        "../../../../spec/vectors/lowering/lowering.json"
    ))
    .unwrap();
    let mut cases = 0;
    for c in oracle["cases"].as_array().unwrap() {
        let name = c["model"].as_str().unwrap();
        let source = gold["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["model"] == c["file"])
            .unwrap();
        let mut claim = ClaimList::open(
            &hex(source["hex"].as_str().unwrap()),
            &DecodeLimits::MESSAGE,
        )
        .unwrap();
        apply_verifier_variant(&mut claim, c);
        let n = c["n"].as_u64().unwrap() as u32;
        let got = verifier::snapshot(&claim, n);
        let g = &got["graph"];
        let expected = oracle_graph(c);
        assert_eq!(
            report::difference(&expected, g),
            None,
            "oracle graph for {name}, n={n}"
        );
        let range_vectors: Value =
            serde_json::from_str(include_str!("../../../../spec/vectors/ranges/ranges.json"))
                .unwrap();
        let expected_ranges = range_vectors["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["model"] == c["model"] && r["n"] == c["n"])
            .unwrap();
        let intervals: Vec<Value> = expected_ranges["ranges"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| {
                if r.is_null() {
                    Value::Null
                } else {
                    json!([r[0].to_string(), r[1].to_string()])
                }
            })
            .collect();
        assert_eq!(
            report::difference(&json!(intervals), &got["ranges"]),
            None,
            "oracle ranges for {name}, n={n}"
        );
        let schedules: Value = serde_json::from_str(include_str!(
            "../../../../spec/vectors/schedule/schedule.json"
        ))
        .unwrap();
        if let Some(reference) = schedules["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["model"] == c["model"] && r["n"] == c["n"])
        {
            let actual = &got["schedule"];
            assert_eq!(actual["layers"], reference["layers"]);
            assert_eq!(actual["layer_vars"], reference["layer_vars"]);
            assert_eq!(
                actual["stages"].as_array().unwrap().len() + 1,
                reference["stage_count"].as_u64().unwrap() as usize
            );
            assert_eq!(actual["leaf_splits"], reference["stages"][0]["splits"]);
            for (found, expected) in actual["stages"]
                .as_array()
                .unwrap()
                .iter()
                .zip(reference["stages"].as_array().unwrap().iter().skip(1))
            {
                for field in ["index", "rounds", "splits", "zero_vars"] {
                    assert_eq!(found[field], expected[field]);
                }
                assert_eq!(
                    found["instances"].as_array().unwrap().len(),
                    expected["instances"].as_array().unwrap().len()
                );
                for (instance, e) in found["instances"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .zip(expected["instances"].as_array().unwrap())
                {
                    for field in ["node", "stacked", "vars"] {
                        assert_eq!(instance[field], e[field]);
                    }
                    assert_eq!(instance["tensor"][1], e["tensor"]);
                    let relation = instance["relation"]
                        .as_object()
                        .unwrap()
                        .keys()
                        .next()
                        .unwrap();
                    assert_eq!(
                        if relation == "Equality" {
                            "ZeroCheck"
                        } else {
                            relation
                        },
                        e["relation"].as_str().unwrap()
                    );
                    let point: Vec<_> = e["point"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|p| {
                            let s = p.as_str().unwrap();
                            if let Some(i) = s.strip_prefix("zero:") {
                                format!("zero:1:{i}")
                            } else {
                                s.to_owned()
                            }
                        })
                        .collect();
                    assert_eq!(
                        instance[if relation == "Equality" {
                            "zero"
                        } else {
                            "point"
                        }],
                        json!(point)
                    );
                    assert_eq!(
                        if e["layers"].is_null() {
                            json!(["none"])
                        } else {
                            json!(["point", e["layers"]])
                        },
                        instance["layers"]
                    );
                }
            }
        }
        cases += 1;
    }
    assert_eq!(cases, 24);
}

#[test]
fn both_encoders_match_independent_python_frames_and_hashes() {
    let vectors: Value = serde_json::from_str(include_str!(
        "../../../../spec/vectors/constraint-digest/framing.json"
    ))
    .unwrap();
    for case in vectors["cases"].as_array().unwrap() {
        assert_eq!(
            compiler::frame(&case["tree"]),
            hex(case["frame"].as_str().unwrap())
        );
        assert_eq!(
            compiler::digest(&case["tree"]).as_slice(),
            hex(case["digest"].as_str().unwrap())
        );
        assert_eq!(
            verifier::frame(&case["tree"]),
            hex(case["frame"].as_str().unwrap())
        );
        assert_eq!(
            verifier::digest(&case["tree"]).as_slice(),
            hex(case["digest"].as_str().unwrap())
        );
    }
}

#[test]
fn diagnostic_reports_first_changed_missing_extra_and_reordered_constraint() {
    let good = json!({"graph":{"constants":["1","2"],"nodes":[{"op":{"Add":[]}}, {"op":{"Mul":["leading"]}}]}});
    let mut bad = good.clone();
    bad["graph"]["constants"][1] = json!("3");
    let d = report::difference(&good, &bad).unwrap();
    assert_eq!(d.path, "/graph/constants/1");
    assert_eq!(d.compiler, Some(json!("2")));
    assert_eq!(d.verifier, Some(json!("3")));
    bad["graph"]["nodes"][0]["op"] = json!({"Sub":[]});
    assert_eq!(
        report::difference(&good, &bad).unwrap().path,
        "/graph/constants/1"
    );
    let mut missing = good.clone();
    missing["graph"]["nodes"].as_array_mut().unwrap().pop();
    assert_eq!(
        report::difference(&good, &missing).unwrap().path,
        "/graph/nodes/1"
    );
    let mut extra = good.clone();
    extra["graph"]["nodes"]
        .as_array_mut()
        .unwrap()
        .push(json!({"op":{"Mul":[]}}));
    assert_eq!(
        report::difference(&good, &extra).unwrap().path,
        "/graph/nodes/2"
    );
    let mut reversed = good.clone();
    reversed["graph"]["nodes"].as_array_mut().unwrap().reverse();
    assert_eq!(
        report::difference(&good, &reversed).unwrap().path,
        "/graph/nodes/0/op/Add"
    );
    assert_eq!(
        report::difference(&json!({"~/":1}), &json!({"~/":2}))
            .unwrap()
            .path,
        "/~0~1"
    );
    assert!(report::difference(&good, &good).is_none());
}

fn oracle_kind(value: &Value) -> Value {
    match value.as_str().unwrap() {
        "A" => json!(["advice"]),
        "W" => json!(["weight"]),
        "P" => json!(["public"]),
        "v" => json!(["virtual"]),
        v => json!([
            "lookup",
            v.strip_prefix("L(")
                .unwrap()
                .strip_suffix(')')
                .unwrap()
                .parse::<u32>()
                .unwrap()
        ]),
    }
}
fn oracle_graph(c: &Value) -> Value {
    let mut tensors = vec![Value::Null; c["tensors"].as_u64().unwrap() as usize];
    let inputs:Vec<_>=c["inputs"].as_array().unwrap().iter().map(|row|{
        tensors[row[0].as_u64().unwrap() as usize]=json!({"shape":row[2],"kind":oracle_kind(&row[3]),"producer":null});
        let role:Vec<_>=row[1].as_str().unwrap().split_whitespace().collect();
        let role=if role.len()==1 {json!(role)}else {assert_eq!(role.len(),3);json!([role[0],[role[1],role[2].parse::<u32>().unwrap()]])};
        let ty=row[4].as_str().unwrap();json!({"tensor":row[0],"role":role,"type":[ty[1..].parse::<u8>().unwrap(),u8::from(ty.starts_with('i'))],"range":[row[5].to_string(),row[6].to_string()]})
    }).collect();
    let nodes:Vec<_>=c["nodes"].as_array().unwrap().iter().enumerate().map(|(index,row)|{
        assert_eq!(row[0].as_u64().unwrap() as usize,index);
        for output in row[4].as_array().unwrap() {let id=output[0].as_u64().unwrap() as usize;assert!(tensors[id].is_null());tensors[id]=json!({"shape":output[1],"kind":oracle_kind(&output[2]),"producer":index});}
        let tag=row[1].as_str().unwrap();let p=&row[2];let params=match tag {
            "OneHot"=>json!([p["vocab"]]),"MatMulWeight"=>p["weight"].clone(),"ConstMul"=>json!([p["c"]]),"Add"|"Reshape"=>json!([]),"SumSquares"=>json!([p["axis"]]),"RsqrtRow"=>json!([p["kappa"],p["eps_int"],p["dim"]]),"Mul"=>json!([p["broadcast"].as_str().unwrap().to_ascii_lowercase()]),"Requant"=>json!([p["shift"].to_string(),p["lo"].to_string(),p["hi"].to_string()]),"Rotate"=>json!([p["rope"]]),"Transpose"=>p["perm"].clone(),"Expand"=>json!([p["axis"],p["len"]]),"MatMulAct"=>json!([p["transpose_rhs"]]),"SoftmaxRow"=>json!([p["exp_table"],p["sigma"]]),"TableMap"=>json!([p["table"]]),other=>panic!("unrecognized oracle op {other}")};
        let mut op=serde_json::Map::new();op.insert(tag.to_owned(),params);
        json!({"op":op,"inputs":row[3],"outputs":row[4].as_array().unwrap().iter().map(|o|o[0].as_u64().unwrap()).collect::<Vec<_>>()})
    }).collect();
    assert!(tensors.iter().all(|v| !v.is_null()));
    let tables: Vec<_> = c["tables"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            let (name, params) = row[1].as_object().unwrap().iter().next().unwrap();
            let fields = match name.as_str() {
                "Rope" => vec![
                    "theta",
                    "factor",
                    "low_freq_factor",
                    "high_freq_factor",
                    "original_max",
                    "bits",
                    "max_context",
                    "d_head",
                ],
                "Silu" => vec!["b", "r_by_class", "in_exp", "out_exp"],
                "Exp" => vec![],
                _ => panic!("not a lowering table"),
            };
            let values = if name == "Exp" {
                json!([
                    params["key_bits"],
                    params["sigma"],
                    params["scale"][0].to_string(),
                    params["scale"][1].to_string()
                ])
            } else {
                json!(
                    fields
                        .into_iter()
                        .map(|f| params[f].clone())
                        .collect::<Vec<_>>()
                )
            };
            let mut kind = serde_json::Map::new();
            kind.insert(name.clone(), values);
            json!({"id":row[0],"kind":kind,"log2_rows":row[2]})
        })
        .collect();
    let route = |site| match c["routes"][site].as_str().unwrap() {
        "fused key" => "fused",
        "two steps" => "two-step",
        other => panic!("bad route {other}"),
    };
    json!({"length":c["n"],"tensors":tensors,"inputs":inputs,"nodes":nodes,"constants":c["constants"].as_array().unwrap().iter().map(|v|{assert_eq!(v.as_array().unwrap().len(),1);v[0].to_string()}).collect::<Vec<_>>(),"shifts":c["shift_classes"],"tables":tables,"weights":c["weights"].as_array().unwrap().iter().map(|w|json!([[w[0],w[1]],w[2]])).collect::<Vec<_>>(),"checkpoints":c["checkpoints"],"routes":[route("AttnResidual"),route("MlpResidual")]})
}

fn apply_verifier_variant(c: &mut ClaimList, case: &Value) {
    use tv_core::claims::{QuantPoint, Site};
    let Some(v) = case.get("variant") else {
        return;
    };
    if let Some(kv) = v.get("n_kv_heads") {
        c.arch.n_kv_heads = kv.as_u64().unwrap() as u32;
    }
    if let Some(edits) = v.get("rescale") {
        for edit in edits.as_array().unwrap() {
            let site = match edit[1].as_str().unwrap() {
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
            };
            let point = QuantPoint::Layer {
                layer: edit[0].as_u64().unwrap() as u32,
                site,
            };
            c.quant.shift_class_sets.retain(|(p, _)| *p != point);
            c.quant.affine_sites.retain(|(p, _)| *p != point);
            match edit[2].as_str().unwrap() {
                "pot" => c.quant.shift_class_sets.push((
                    point,
                    edit[3]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_i64().unwrap() as i8)
                        .collect(),
                )),
                "affine" => c
                    .quant
                    .affine_sites
                    .push((point, edit[3].as_i64().unwrap() as i8)),
                _ => panic!("unknown scale policy"),
            }
        }
    }
    c.quant.shift_class_sets.sort_by_key(|(p, _)| *p);
    c.quant.affine_sites.sort_by_key(|(p, _)| *p);
    if let Some(edits) = v.get("score_scales") {
        for e in edits.as_array().unwrap() {
            let layer = e[0].as_u64().unwrap() as u32;
            let scale =
                tv_codec::Rational::new(e[1].as_i64().unwrap(), e[2].as_u64().unwrap()).unwrap();
            c.quant
                .score_scales
                .iter_mut()
                .find(|(l, _)| *l == layer)
                .unwrap()
                .1 = scale;
        }
    }
}

#[test]
fn compiler_and_verifier_derive_identical_constraint_systems() {
    let mut count = 0;
    for model in fixtures::models() {
        let native: tessera_ir::ClaimList = tessera_codec::envelope::open(
            &model.envelope,
            &tessera_codec::schema::CLAIM_LIST,
            tessera_codec::DecodeLimits::MESSAGE,
        )
        .unwrap();
        let tv = ClaimList::open(&model.envelope, &DecodeLimits::MESSAGE).unwrap();
        for n in model.contexts {
            let expected = compiler::snapshot(&native, n);
            let found = verifier::snapshot(&tv, n);
            assert_eq!(
                report::difference(&expected, &found),
                None,
                "{} n={n}",
                model.name
            );
            assert_eq!(compiler::frame(&expected), verifier::frame(&found));
            assert_eq!(compiler::digest(&expected), verifier::digest(&found));
            count += 1;
        }
    }
    assert_eq!(count, 40);
}

#[test]
fn diagnostics_distinguish_missing_null_and_large_integer_boundaries() {
    let d = report::difference(&json!({"x":null}), &json!({})).unwrap();
    assert_eq!(d.path, "/x");
    assert_eq!(d.compiler, Some(Value::Null));
    assert_eq!(d.verifier, None);
    let d = report::difference(&json!({}), &json!({"x":null})).unwrap();
    assert_eq!(d.path, "/x");
    assert_eq!(d.compiler, None);
    assert_eq!(d.verifier, Some(Value::Null));
    let a = json!({"range":["-170141183460469231731687303715884105728","170141183460469231731687303715884105727"]});
    let b = json!({"range":["-170141183460469231731687303715884105728","170141183460469231731687303715884105726"]});
    assert_eq!(report::difference(&a, &b).unwrap().path, "/range/1");
    assert_ne!(compiler::digest(&a), compiler::digest(&b));
    assert_ne!(verifier::digest(&a), verifier::digest(&b));
}

#[test]
fn diagnostic_hashes_reject_floating_point() {
    assert!(std::panic::catch_unwind(|| compiler::digest(&json!(1.5))).is_err());
    assert!(std::panic::catch_unwind(|| verifier::digest(&json!(1.5))).is_err());
}

#[test]
fn softmax_selection_expands_reused_limb_families_instead_of_deduplicating_roles() {
    let model = fixtures::models()
        .into_iter()
        .find(|m| m.name == "tiny-2-servable")
        .unwrap();
    let mut native: tessera_ir::ClaimList = tessera_codec::envelope::open(
        &model.envelope,
        &tessera_codec::schema::CLAIM_LIST,
        tessera_codec::DecodeLimits::MESSAGE,
    )
    .unwrap();
    let mut tv = ClaimList::open(&model.envelope, &DecodeLimits::MESSAGE).unwrap();
    for bits in [1, 16, 17, 32] {
        native.gadget_policy.limb_bits = bits;
        tv.gadget_policy.limb_bits = bits;
        let a = compiler::snapshot(&native, 1);
        let b = verifier::snapshot(&tv, 1);
        assert_eq!(report::difference(&a, &b), None, "limb_bits={bits}");
        let choice = a["selection"]["choices"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["pattern"] == json!(["SoftmaxRow"]))
            .unwrap();
        let refs = choice["tables"].as_array().unwrap();
        assert_eq!(refs.len(), 2 + 34usize.div_ceil(usize::from(bits)));
        if bits == 17 {
            assert_eq!(refs[1], refs[2]);
            assert_eq!(refs[2], refs[3]);
        }
    }
}
