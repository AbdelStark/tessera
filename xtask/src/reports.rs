// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Committed JSON reports validate against their schemas (03 §5, RFC-0020 'Testing Strategy').
//!
//! Every JSON report carries a `"schema"` identifier, `tessera.<name>/<version>`, whose JSON Schema
//! is `spec/schemas/<name>-v<version>.schema.json`. This check reads every committed report under
//! `docs/perf/`, validates it against its schema, and checks that every schema file uses only the
//! keywords the validator implements. That subset of JSON Schema 2020-12 is small on purpose: an
//! unsupported keyword is an error, never silently ignored. It covers the annotations (`$schema`,
//! `$id`, `title`, `description`, `$comment`), `$defs` and local `$ref`, `type`, `properties`,
//! `required`, `additionalProperties`, `items`, `minItems`, `maxItems`, `const`, `enum`,
//! `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `minLength` and `maxLength`.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

/// Keywords that annotate and constrain nothing.
const ANNOTATIONS: [&str; 5] = ["$schema", "$id", "title", "description", "$comment"];

/// Keywords the validator implements.
const KEYWORDS: [&str; 17] = [
    "$defs",
    "$ref",
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "minItems",
    "maxItems",
    "const",
    "enum",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
];

/// The schema file of a report's `"schema"` identifier, `tessera.<name>/<version>`.
pub(crate) fn schema_path(identifier: &str) -> Option<PathBuf> {
    let (name, version) = identifier.strip_prefix("tessera.")?.split_once('/')?;
    let valid_name = !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-');
    let valid_version = !version.is_empty()
        && version.bytes().all(|b| b.is_ascii_digit())
        && !version.starts_with('0');
    (valid_name && valid_version)
        .then(|| Path::new("spec/schemas").join(format!("{name}-v{version}.schema.json")))
}

/// Problems with `schema` itself: a keyword the validator does not implement, a `$ref` that is
/// not a local `#/$defs/…` or names no definition, or a malformed keyword value.
pub(crate) fn schema_problems(schema: &Value) -> Vec<String> {
    let mut problems = Vec::new();
    let defs = schema.get("$defs").and_then(Value::as_object);
    walk_schema(schema, "", defs, &mut problems);
    problems
}

fn walk_schema(schema: &Value, at: &str, defs: Option<&Map<String, Value>>, out: &mut Vec<String>) {
    let Some(object) = schema.as_object() else {
        if !schema.is_boolean() {
            out.push(format!("{at}: a schema is an object or a boolean"));
        }
        return;
    };
    for (key, value) in object {
        let here = format!("{at}/{key}");
        if ANNOTATIONS.contains(&key.as_str()) {
            continue;
        }
        if !KEYWORDS.contains(&key.as_str()) {
            out.push(format!("{here}: unsupported keyword"));
            continue;
        }
        match key.as_str() {
            "$defs" | "properties" => match value.as_object() {
                Some(children) => {
                    for (name, child) in children {
                        walk_schema(child, &format!("{here}/{name}"), defs, out);
                    }
                }
                None => out.push(format!("{here}: expected an object of schemas")),
            },
            "items" | "additionalProperties" => walk_schema(value, &here, defs, out),
            "$ref" => {
                let target = value.as_str().and_then(|r| r.strip_prefix("#/$defs/"));
                if !target.is_some_and(|t| defs.is_some_and(|d| d.contains_key(t))) {
                    out.push(format!(
                        "{here}: not a local #/$defs/ reference to a definition"
                    ));
                }
            }
            "type" => {
                let names: Vec<&Value> = match value {
                    Value::Array(names) => names.iter().collect(),
                    other => vec![other],
                };
                for name in names {
                    let known = [
                        "object", "array", "string", "number", "integer", "boolean", "null",
                    ];
                    if !name.as_str().is_some_and(|n| known.contains(&n)) {
                        out.push(format!("{here}: unknown type {name}"));
                    }
                }
            }
            "required"
                if !value
                    .as_array()
                    .is_some_and(|a| a.iter().all(Value::is_string)) =>
            {
                out.push(format!("{here}: expected an array of names"));
            }
            "enum" if !value.is_array() => {
                out.push(format!("{here}: expected an array"));
            }
            "minItems" | "maxItems" | "minLength" | "maxLength" if !value.is_u64() => {
                out.push(format!("{here}: expected a non-negative integer"));
            }
            "minimum" | "maximum" | "exclusiveMinimum" | "exclusiveMaximum"
                if !value.is_number() =>
            {
                out.push(format!("{here}: expected a number"));
            }
            _ => {}
        }
    }
}

/// The ways `instance` fails `schema`, each with its JSON pointer; `root` holds the `$defs`.
pub(crate) fn validate(schema: &Value, root: &Value, instance: &Value, at: &str) -> Vec<String> {
    let mut errors = Vec::new();
    check(schema, root, instance, at, &mut errors);
    errors
}

fn check(schema: &Value, root: &Value, instance: &Value, at: &str, out: &mut Vec<String>) {
    let at_or_root = if at.is_empty() { "/" } else { at };
    let object = match schema {
        Value::Bool(true) => return,
        Value::Bool(false) => {
            out.push(format!("{at_or_root}: no value is allowed here"));
            return;
        }
        Value::Object(object) => object,
        _ => return,
    };
    if let Some(target) = object.get("$ref").and_then(Value::as_str) {
        let name = target.strip_prefix("#/$defs/").unwrap_or_default();
        match root.get("$defs").and_then(|d| d.get(name)) {
            Some(definition) => check(definition, root, instance, at, out),
            None => out.push(format!("{at_or_root}: unresolved {target}")),
        }
    }
    if let Some(expected) = object.get("type") {
        let names: Vec<&str> = match expected {
            Value::Array(names) => names.iter().filter_map(Value::as_str).collect(),
            other => other.as_str().into_iter().collect(),
        };
        if !names.iter().any(|name| has_type(instance, name)) {
            out.push(format!("{at_or_root}: expected {}", names.join(" or ")));
            return;
        }
    }
    if let Some(expected) = object.get("const")
        && !same(instance, expected)
    {
        out.push(format!("{at_or_root}: expected {expected}"));
    }
    if let Some(options) = object.get("enum").and_then(Value::as_array)
        && !options.iter().any(|option| same(instance, option))
    {
        out.push(format!(
            "{at_or_root}: not one of {}",
            Value::Array(options.clone())
        ));
    }
    if let Some(x) = instance.as_f64() {
        let bound = |key: &str| object.get(key).and_then(Value::as_f64);
        let checks: [(&str, Bound); 4] = [
            ("minimum", |x, b| x >= b),
            ("maximum", |x, b| x <= b),
            ("exclusiveMinimum", |x, b| x > b),
            ("exclusiveMaximum", |x, b| x < b),
        ];
        for (key, holds) in checks {
            if let Some(b) = bound(key)
                && !holds(x, b)
            {
                out.push(format!("{at_or_root}: {x} fails {key} {b}"));
            }
        }
    }
    if let Some(text) = instance.as_str() {
        let length = text.chars().count() as u64;
        if object
            .get("minLength")
            .and_then(Value::as_u64)
            .is_some_and(|m| length < m)
        {
            out.push(format!("{at_or_root}: shorter than minLength"));
        }
        if object
            .get("maxLength")
            .and_then(Value::as_u64)
            .is_some_and(|m| length > m)
        {
            out.push(format!("{at_or_root}: longer than maxLength"));
        }
    }
    if let Some(items) = instance.as_array() {
        let count = items.len() as u64;
        if object
            .get("minItems")
            .and_then(Value::as_u64)
            .is_some_and(|m| count < m)
        {
            out.push(format!("{at_or_root}: fewer than minItems"));
        }
        if object
            .get("maxItems")
            .and_then(Value::as_u64)
            .is_some_and(|m| count > m)
        {
            out.push(format!("{at_or_root}: more than maxItems"));
        }
        if let Some(item_schema) = object.get("items") {
            for (i, item) in items.iter().enumerate() {
                check(item_schema, root, item, &format!("{at}/{i}"), out);
            }
        }
    }
    if let Some(fields) = instance.as_object() {
        let properties = object.get("properties").and_then(Value::as_object);
        for name in object
            .get("required")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let Some(name) = name.as_str()
                && !fields.contains_key(name)
            {
                out.push(format!("{at_or_root}: missing required {name}"));
            }
        }
        for (name, value) in fields {
            let here = format!("{at}/{name}");
            match properties.and_then(|p| p.get(name)) {
                Some(property) => check(property, root, value, &here, out),
                None => {
                    if let Some(additional) = object.get("additionalProperties") {
                        check(additional, root, value, &here, out);
                    }
                }
            }
        }
    }
}

/// A numeric bound's test: whether the value satisfies the bound.
type Bound = fn(f64, f64) -> bool;

/// Whether `value` is of the JSON Schema type `name`; an integer is a number without a fraction.
fn has_type(value: &Value, name: &str) -> bool {
    match name {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        "number" => value.is_number(),
        "integer" => value.as_f64().is_some_and(|x| x.fract() == 0.0),
        _ => false,
    }
}

/// JSON Schema equality: numbers by value, so that `1` and `1.0` are the same.
fn same(a: &Value, b: &Value) -> bool {
    match (a.as_f64(), b.as_f64()) {
        (Some(x), Some(y)) if a.is_number() && b.is_number() => x == y,
        _ => a == b,
    }
}

fn read_json(path: &Path) -> Result<Value, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: not JSON: {e}", path.display()))
}

/// The committed reports under `docs/perf/` that fail their schemas, and the schema files under
/// `spec/schemas/` that use a keyword the validator does not implement.
pub(crate) fn check_reports(repo: &Path, files: &[PathBuf]) -> io::Result<Vec<String>> {
    let mut problems = Vec::new();
    for file in files {
        let schema_file =
            file.starts_with("spec/schemas") && file.to_string_lossy().ends_with(".schema.json");
        if schema_file {
            match read_json(&repo.join(file)) {
                Ok(schema) => {
                    for problem in schema_problems(&schema) {
                        problems.push(format!("{}: {problem}", file.display()));
                    }
                }
                Err(message) => problems.push(message),
            }
        }
    }
    for file in files {
        let report = file.starts_with("docs/perf") && file.extension().is_some_and(|e| e == "json");
        if !report {
            continue;
        }
        let instance = match read_json(&repo.join(file)) {
            Ok(instance) => instance,
            Err(message) => {
                problems.push(message);
                continue;
            }
        };
        let Some(identifier) = instance.get("schema").and_then(Value::as_str) else {
            problems.push(format!("{}: no \"schema\" identifier", file.display()));
            continue;
        };
        let Some(schema_file) = schema_path(identifier) else {
            problems.push(format!(
                "{}: malformed schema identifier {identifier:?}",
                file.display()
            ));
            continue;
        };
        let schema = match read_json(&repo.join(&schema_file)) {
            Ok(schema) => schema,
            Err(_) => {
                problems.push(format!(
                    "{}: {identifier} has no schema file {}",
                    file.display(),
                    schema_file.display()
                ));
                continue;
            }
        };
        for error in validate(&schema, &schema, &instance, "") {
            problems.push(format!("{}: {error}", file.display()));
        }
    }
    Ok(problems)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn schema() -> Value {
        json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "tessera.example/1",
            "type": "object",
            "required": ["schema", "runs"],
            "additionalProperties": false,
            "properties": {
                "schema": {"const": "tessera.example/1"},
                "runs": {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/run"}}
            },
            "$defs": {
                "run": {
                    "type": "object",
                    "required": ["name", "ns"],
                    "additionalProperties": false,
                    "properties": {
                        "name": {"type": "string", "minLength": 1, "maxLength": 8},
                        "ns": {"type": "number", "minimum": 0},
                        "count": {"type": "integer", "minimum": 1},
                        "kind": {"enum": ["cpu", "gpu"]},
                        "rate": {"type": "number", "exclusiveMinimum": 0, "maximum": 1}
                    }
                }
            }
        })
    }

    fn errors(instance: Value) -> Vec<String> {
        let schema = schema();
        validate(&schema, &schema, &instance, "")
    }

    #[test]
    fn a_conforming_report_passes() {
        assert_eq!(schema_problems(&schema()), Vec::<String>::new());
        let report = json!({"schema": "tessera.example/1", "runs": [
            {"name": "fp", "ns": 0.73, "count": 3.0, "kind": "cpu", "rate": 1},
            {"name": "ntt", "ns": 0}
        ]});
        assert_eq!(errors(report), Vec::<String>::new());
    }

    /// Each constraint the subset implements, violated alone, with the pointer it names.
    #[test]
    fn each_constraint_fails_alone() {
        let run = |field: Value| json!({"schema": "tessera.example/1", "runs": [field]});
        let cases = [
            (json!({"runs": []}), "/: missing required schema"),
            (
                json!({"schema": "tessera.example/2", "runs": [{"name": "a", "ns": 1}]}),
                "/schema: expected \"tessera.example/1\"",
            ),
            (
                json!({"schema": "tessera.example/1", "runs": []}),
                "/runs: fewer than minItems",
            ),
            (
                json!({"schema": "tessera.example/1", "runs": [], "extra": 1}),
                "/extra: no value is allowed here",
            ),
            (run(json!({"name": "a"})), "/runs/0: missing required ns"),
            (
                run(json!({"name": "a", "ns": -1})),
                "/runs/0/ns: -1 fails minimum 0",
            ),
            (
                run(json!({"name": "", "ns": 1})),
                "/runs/0/name: shorter than minLength",
            ),
            (
                run(json!({"name": "toolongname", "ns": 1})),
                "/runs/0/name: longer than maxLength",
            ),
            (
                run(json!({"name": "a", "ns": "1"})),
                "/runs/0/ns: expected number",
            ),
            (
                run(json!({"name": "a", "ns": 1, "count": 1.5})),
                "/runs/0/count: expected integer",
            ),
            (
                run(json!({"name": "a", "ns": 1, "kind": "tpu"})),
                "/runs/0/kind: not one of [\"cpu\",\"gpu\"]",
            ),
            (
                run(json!({"name": "a", "ns": 1, "rate": 0})),
                "/runs/0/rate: 0 fails exclusiveMinimum 0",
            ),
            (
                run(json!({"name": "a", "ns": 1, "rate": 2})),
                "/runs/0/rate: 2 fails maximum 1",
            ),
        ];
        for (instance, expected) in cases {
            let found = errors(instance.clone());
            assert!(found.iter().any(|e| e == expected), "{instance}: {found:?}");
        }
    }

    /// A schema the validator cannot apply in full is a problem, never a pass.
    #[test]
    fn unsupported_or_dangling_schemas_are_refused() {
        let mut schema = schema();
        schema["properties"]["runs"]["pattern"] = json!("^x$");
        schema["$defs"]["run"]["properties"]["ns"]["$ref"] = json!("#/$defs/missing");
        schema["properties"]["schema"]["type"] = json!("decimal");
        let problems = schema_problems(&schema);
        assert!(
            problems.contains(&"/properties/runs/pattern: unsupported keyword".to_owned()),
            "{problems:?}"
        );
        assert!(
            problems
                .iter()
                .any(|p| p.starts_with("/$defs/run/properties/ns/$ref")),
            "{problems:?}"
        );
        assert!(
            problems.iter().any(|p| p.contains("unknown type")),
            "{problems:?}"
        );
    }

    #[test]
    fn schema_identifiers_name_their_files() {
        assert_eq!(
            schema_path("tessera.microbenchmarks/1").as_deref(),
            Some(Path::new("spec/schemas/microbenchmarks-v1.schema.json"))
        );
        for malformed in [
            "microbenchmarks/1",
            "tessera.Micro/1",
            "tessera.micro/01",
            "tessera.micro",
            "tessera.micro/x",
            "tessera./1",
        ] {
            assert_eq!(schema_path(malformed), None, "{malformed}");
        }
    }
}
