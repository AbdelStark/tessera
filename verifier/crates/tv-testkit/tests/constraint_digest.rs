// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Independent native consumption of the CI-only framing vectors (RFC-0007, DEP-2).
use tv_testkit::{Json, blake3, hex, read_vectors};

fn framed(tree: &Json) -> Vec<u8> {
    fn visit(v: &Json, path: &str, out: &mut Vec<u8>) {
        let (tag, body) = match v {
            Json::Null => ("null", String::new()),
            Json::Bool(b) => ("bool", if *b { "1" } else { "0" }.to_owned()),
            Json::Number(n) => {
                assert!(n.parse::<i64>().is_ok() || n.parse::<u64>().is_ok());
                ("integer", n.clone())
            }
            Json::String(s) => ("string", s.clone()),
            Json::Array(a) => ("array", a.len().to_string()),
            Json::Object(o) => ("object", o.len().to_string()),
        };
        for field in [path, tag, &body] {
            out.extend_from_slice(&(field.len() as u64).to_le_bytes());
            out.extend_from_slice(field.as_bytes());
        }
        match v {
            Json::Array(a) => {
                for (i, child) in a.iter().enumerate() {
                    visit(child, &format!("{path}/{i}"), out);
                }
            }
            Json::Object(o) => {
                let mut fields: Vec<_> = o.iter().collect();
                fields.sort_by_key(|(name, _)| name);
                for (name, child) in fields {
                    let mut escaped = String::new();
                    for c in name.chars() {
                        match c {
                            '~' => escaped.push_str("~0"),
                            '/' => escaped.push_str("~1"),
                            c => escaped.push(c),
                        }
                    }
                    visit(child, &format!("{path}/{escaped}"), out);
                }
            }
            _ => {}
        }
    }
    let mut bytes = Vec::new();
    visit(tree, "", &mut bytes);
    bytes
}

#[test]
fn independent_constraint_tree_frames_and_hashes() {
    let v = read_vectors("constraint-digest/framing.json");
    let context = v.get("context").unwrap().as_str().unwrap();
    assert_eq!(context, "tessera/ci/constraint-system-digest/1");
    let cases = v.get("cases").unwrap().as_array().unwrap();
    assert_eq!(cases.len(), 12);
    for case in cases {
        let bytes = framed(case.get("tree").unwrap());
        assert_eq!(bytes, hex(case.get("frame").unwrap().as_str().unwrap()));
        assert_eq!(
            blake3::derive_key(context, &bytes).as_slice(),
            hex(case.get("digest").unwrap().as_str().unwrap())
        );
    }
}
