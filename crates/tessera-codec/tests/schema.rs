// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The schema registry is the artifact catalog of 03 §5: the same schemas, in the same order,
//! with the same ids, versions, extensions and classes, read from the specification itself.

use std::fs;
use std::path::PathBuf;

use tessera_codec::schema::{self, CATALOG, Class};

/// The rows of the catalog table in `docs/spec/03-data-model.md`, one schema per name. A row
/// naming several artifacts gives an id range, `0x0501`–`0x0503`.
fn spec_catalog() -> Vec<(String, u16, u16, Option<String>, Class)> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/spec/03-data-model.md");
    let text = fs::read_to_string(path).expect("03-data-model.md");
    let section = text
        .split("## 5. Artifact catalog")
        .nth(1)
        .and_then(|rest| rest.split("\n## ").next())
        .expect("03 §5");
    let mut rows = Vec::new();
    for line in section.lines().filter(|l| l.starts_with("| `")) {
        let cells: Vec<&str> = line.trim_matches('|').split('|').map(str::trim).collect();
        let [names, ids, version, extension, class, _defined_in] = cells[..] else {
            panic!("a catalog row with six cells: {line}");
        };
        let names: Vec<String> = names
            .split(", ")
            .map(|n| n.trim_matches('`').to_owned())
            .collect();
        let id = |text: &str| {
            u16::from_str_radix(text.trim_matches('`').trim_start_matches("0x"), 16)
                .expect("hex id")
        };
        let ids: Vec<u16> = match ids.split_once('–') {
            Some((first, last)) => (id(first)..=id(last)).collect(),
            None => vec![id(ids)],
        };
        assert_eq!(names.len(), ids.len(), "{line}");
        let extension = extension
            .strip_prefix("`.")
            .map(|e| e.trim_end_matches('`').to_owned());
        let class = match class.split_whitespace().next() {
            Some("Public") => Class::Public,
            Some("Protocol") => Class::Protocol,
            Some("Private") => Class::Private,
            Some("Secret") => Class::Secret,
            other => panic!("class {other:?} in {line}"),
        };
        let version: u16 = version.parse().expect("a version");
        for (name, id) in names.into_iter().zip(ids) {
            rows.push((name, id, version, extension.clone(), class));
        }
    }
    rows
}

#[test]
fn the_registry_is_the_catalog_of_03_5() {
    let spec = spec_catalog();
    let code: Vec<_> = CATALOG
        .iter()
        .map(|s| {
            (
                s.name.to_owned(),
                s.id.0,
                s.version,
                s.extension.map(str::to_owned),
                s.class,
            )
        })
        .collect();
    assert_eq!(code, spec);
    for s in &CATALOG {
        assert_eq!(schema::by_id(s.id), Some(s));
    }
}
