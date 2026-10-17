// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Conformance of the error taxonomy: the codes, classes and exit codes of `ErrorCode` equal the
//! table of docs/spec/04-error-model.md §3 and the shared golden file
//! `spec/vectors/errors/codes.json`, row for row. A change to any of the three fails here until
//! the other two follow.

use std::fs;
use std::path::Path;

use tessera_codec::error::{ErrorCode, PublicContext};

fn repo_file(rel: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// `(code, class, exit)` for every code, in declaration order.
fn implemented() -> Vec<(String, String, u8)> {
    ErrorCode::ALL
        .iter()
        .map(|c| (c.to_string(), c.class().as_str().to_owned(), c.exit_code()))
        .collect()
}

#[test]
fn codes_equal_the_specification_table() {
    let spec = repo_file("docs/spec/04-error-model.md");
    let section = spec
        .split("## 3. Error codes")
        .nth(1)
        .and_then(|rest| rest.split("\n## ").next())
        .expect("04 §3 exists");
    let rows: Vec<(String, String, u8)> = section
        .lines()
        .filter(|line| line.starts_with("| `ERR-"))
        .map(|line| {
            let cells: Vec<&str> = line.trim_matches('|').split('|').map(str::trim).collect();
            let code = cells[0].trim_matches('`').to_owned();
            let exit = cells
                .last()
                .expect("exit column")
                .parse()
                .expect("numeric exit code");
            (code, cells[1].to_owned(), exit)
        })
        .collect();
    assert_eq!(rows.len(), 26, "04 §3 lists 26 codes");
    assert_eq!(implemented(), rows);
}

#[test]
fn codes_equal_the_golden_file() {
    let golden: serde_json::Value =
        serde_json::from_str(&repo_file("spec/vectors/errors/codes.json")).expect("valid JSON");
    let rows: Vec<(String, String, u8)> = golden["codes"]
        .as_array()
        .expect("a codes array")
        .iter()
        .map(|row| {
            let exit = row["exit"].as_u64().expect("numeric exit");
            (
                row["code"].as_str().expect("code").to_owned(),
                row["class"].as_str().expect("class").to_owned(),
                u8::try_from(exit).expect("exit fits u8"),
            )
        })
        .collect();
    assert_eq!(implemented(), rows);
}

#[test]
fn identifiers_are_unique_and_round_trip() {
    for code in ErrorCode::ALL {
        assert_eq!(ErrorCode::from_identifier(code.as_str()), Some(*code));
    }
    let mut names: Vec<&str> = ErrorCode::ALL.iter().map(|c| c.as_str()).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), ErrorCode::ALL.len());
    assert_eq!(ErrorCode::from_identifier("ERR-NOPE"), None);
}

/// Compile-time check of 04 §4: every field of `PublicContext` is an identifier type. Adding a
/// field, or changing a field's type, breaks this destructuring until the new field is pinned
/// here, which is the moment to check it cannot carry field elements, tokens or tensors.
#[test]
fn public_context_fields_are_identifiers_only() {
    use tessera_codec::error::{
        ByteOffset, GadgetId, LayerIndex, RequestId, SchemaId, StageId, TableId, TensorId,
        TokenPosition,
    };
    let PublicContext {
        schema,
        offset,
        stage,
        layer,
        tensor,
        table,
        gadget,
        request,
        position,
    } = PublicContext::default();
    let _: Option<SchemaId> = schema;
    let _: Option<ByteOffset> = offset;
    let _: Option<StageId> = stage;
    let _: Option<LayerIndex> = layer;
    let _: Option<TensorId> = tensor;
    let _: Option<TableId> = table;
    let _: Option<GadgetId> = gadget;
    let _: Option<RequestId> = request;
    let _: Option<TokenPosition> = position;
}

#[test]
fn display_prints_the_code_and_the_public_context() {
    use tessera_codec::error::{LayerIndex, TensorId, TesseraError};
    let bare = TesseraError::new(ErrorCode::CompileRange);
    assert_eq!(bare.to_string(), "ERR-COMPILE-RANGE");
    let located = TesseraError::new(ErrorCode::CompileRange).with_context(PublicContext {
        layer: Some(LayerIndex(3)),
        tensor: Some(TensorId(17)),
        ..PublicContext::default()
    });
    assert_eq!(
        located.to_string(),
        "ERR-COMPILE-RANGE (layer 3, tensor 17)"
    );
    assert_eq!(located.exit_code(), 20);
}
