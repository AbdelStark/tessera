// SPDX-License-Identifier: Apache-2.0 OR MIT

// Test-only validator for the keywords used by the committed observability schemas.
use tv_testkit::Json;

pub fn validate(schema: &Json, root: &Json, value: &Json) {
    if let Json::Bool(allowed) = schema {
        assert!(allowed, "additional property: {value:?}");
        return;
    }
    let Json::Object(keywords) = schema else {
        panic!("invalid schema")
    };
    for (key, rule) in keywords {
        match key.as_str() {
            "$schema" | "$id" | "title" | "description" | "$defs" => (),
            "$ref" => {
                let name = rule.as_str().unwrap().strip_prefix("#/$defs/").unwrap();
                validate(root.get("$defs").unwrap().get(name).unwrap(), root, value);
            }
            "const" => assert_eq!(value, rule),
            "enum" => assert!(rule.as_array().unwrap().contains(value), "{value:?}"),
            "type" => {
                let matches = |t: &str| match t {
                    "object" => matches!(value, Json::Object(_)),
                    "array" => matches!(value, Json::Array(_)),
                    "string" => matches!(value, Json::String(_)),
                    "boolean" => matches!(value, Json::Bool(_)),
                    "null" => value.is_null(),
                    "integer" => matches!(value, Json::Number(n) if n.parse::<i128>().is_ok()),
                    "number" => {
                        matches!(value, Json::Number(n) if n.parse::<f64>().is_ok_and(f64::is_finite))
                    }
                    _ => panic!("unsupported type {t}"),
                };
                assert!(
                    match rule {
                        Json::String(t) => matches(t),
                        Json::Array(ts) => ts.iter().any(|t| matches(t.as_str().unwrap())),
                        _ => panic!("invalid type"),
                    },
                    "wrong type: {value:?}"
                );
            }
            "required" => {
                if matches!(value, Json::Object(_)) {
                    for name in rule.as_array().unwrap() {
                        assert!(
                            value.get(name.as_str().unwrap()).is_some(),
                            "missing {name:?}"
                        );
                    }
                }
            }
            "properties" => (), // Applied together with additionalProperties below.
            "additionalProperties" => (),
            "items" => {
                if let Json::Array(items) = value {
                    for item in items {
                        validate(rule, root, item);
                    }
                }
            }
            "minLength" | "maxLength" => {
                if let Json::String(s) = value {
                    let len = s.chars().count() as u64;
                    let bound = rule.as_u64().unwrap();
                    assert!(if key == "minLength" {
                        len >= bound
                    } else {
                        len <= bound
                    });
                }
            }
            "minimum" | "maximum" => {
                if let Json::Number(n) = value {
                    let n: f64 = n.parse().unwrap();
                    let Json::Number(bound) = rule else {
                        panic!("invalid bound")
                    };
                    let bound: f64 = bound.parse().unwrap();
                    assert!(if key == "minimum" {
                        n >= bound
                    } else {
                        n <= bound
                    });
                }
            }
            _ => panic!("unsupported schema keyword {key}"),
        }
    }
    if let Json::Object(members) = value {
        for (key, member) in members {
            if let Some(rule) = schema.get("properties").and_then(|p| p.get(key)) {
                validate(rule, root, member);
            } else if let Some(rule) = schema.get("additionalProperties") {
                validate(rule, root, member);
            }
        }
    }
}

pub fn check(name: &str, text: &str) -> Json {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../spec/schemas")
        .join(format!("{name}-v1.schema.json"));
    let schema = Json::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
    let value = Json::parse(text).unwrap();
    validate(&schema, &schema, &value);
    value
}
