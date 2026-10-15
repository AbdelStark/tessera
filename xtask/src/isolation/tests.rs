// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests of the isolation gate on a fixture repository (`xtask/fixtures/isolation`): the clean
//! fixture passes, and each violation class is reported on its own.

use super::*;
use crate::repo;
use crate::testing::Fixture;

/// A temporary copy of the fixture repository.
struct FixtureRepo {
    dir: Fixture,
}

fn copy_dir(from: &Path, to: &Path) {
    repo::copy_dir(from, to).expect("copy fixture");
}

impl FixtureRepo {
    fn clean() -> Self {
        let dir = Fixture::new("isolation");
        copy_dir(&repo::root().join("xtask/fixtures/isolation"), dir.path());
        let pin = fs::read_to_string(repo::root().join("rust-toolchain.toml")).expect("root pin");
        dir.write("rust-toolchain.toml", &pin);
        dir.write("verifier/rust-toolchain.toml", &pin);
        Self { dir }
    }

    fn path(&self) -> &Path {
        self.dir.path()
    }

    /// Replaces the contents of `rel`.
    fn write(&self, rel: &str, text: &str) {
        self.dir.write(rel, text);
    }

    /// Appends `text` to `rel`.
    fn append(&self, rel: &str, text: &str) {
        let path = self.path().join(rel);
        let old = fs::read_to_string(&path).expect("fixture file exists");
        fs::write(path, old + text).expect("append to fixture file");
    }

    /// Edits `rel` by replacing `from` (which must occur) with `to`.
    fn replace(&self, rel: &str, from: &str, to: &str) {
        let path = self.path().join(rel);
        let old = fs::read_to_string(&path).expect("fixture file exists");
        assert!(old.contains(from), "{rel} does not contain {from:?}");
        fs::write(path, old.replace(from, to)).expect("edit fixture file");
    }

    /// Cargo serving crates.io from the fixture's `vendor/` directory, offline.
    fn cargo(&self) -> Cargo {
        let vendor = self
            .path()
            .canonicalize()
            .expect("fixture path")
            .join("vendor");
        let mut cargo = Cargo::from_env();
        cargo.config = vec![
            r#"source.crates-io.replace-with="fixture-vendor""#.to_owned(),
            format!(r#"source.fixture-vendor.directory="{}""#, vendor.display()),
            "net.offline=true".to_owned(),
        ];
        cargo
    }

    /// Every violation as `"<check>: <violation>"`, after regenerating both lockfiles.
    fn violations(&self) -> Vec<String> {
        let cargo = self.cargo();
        let pin = repo::pinned_channel(&self.path().join("rust-toolchain.toml"));
        for dir in [self.path().to_path_buf(), self.path().join("verifier")] {
            let output = cargo
                .command(&dir, pin.as_deref())
                .arg("generate-lockfile")
                .output()
                .expect("run cargo generate-lockfile");
            assert!(
                output.status.success(),
                "generate-lockfile in {}: {}",
                dir.display(),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        checks(self.path(), &cargo)
            .expect("fixture is readable")
            .into_iter()
            .flat_map(|(check, violations)| {
                violations.into_iter().map(move |v| format!("{check}: {v}"))
            })
            .collect()
    }

    /// Asserts that the violations are exactly one per needle, each containing its needle.
    fn assert_violations(&self, needles: &[&str]) {
        let violations = self.violations();
        assert_eq!(
            violations.len(),
            needles.len(),
            "violations: {violations:#?}"
        );
        for needle in needles {
            assert!(
                violations.iter().any(|v| v.contains(needle)),
                "no violation mentions {needle:?}: {violations:#?}"
            );
        }
    }
}

const TV_A: &str = "verifier/crates/tv-a/Cargo.toml";
const TV_A_LIB: &str = "verifier/crates/tv-a/src/lib.rs";

#[test]
fn repository_is_isolated() {
    let failures: Vec<String> = checks(&repo::root(), &Cargo::from_env())
        .expect("repository is readable")
        .into_iter()
        .flat_map(|(check, violations)| {
            violations.into_iter().map(move |v| format!("{check}: {v}"))
        })
        .collect();
    assert!(
        failures.is_empty(),
        "isolation failures:\n{}",
        failures.join("\n")
    );
}

#[test]
fn clean_fixture_passes() {
    FixtureRepo::clean().assert_violations(&[]);
}

#[test]
fn path_dependency_outside_verifier_fails() {
    let repo = FixtureRepo::clean();
    repo.append(
        TV_A,
        "shared-util = { path = \"../../../crates/shared-util\" }\n",
    );
    repo.assert_violations(&["shared-util is a path package outside verifier/"]);
}

#[test]
fn prover_crate_name_fails_even_from_a_registry() {
    let repo = FixtureRepo::clean();
    repo.append(TV_A, "tessera-sneaky = \"1\"\n");
    repo.assert_violations(&[
        "tessera-sneaky is a prover crate name",
        "tessera-sneaky 1.0.0 is not in verifier/allowlist.toml",
    ]);
}

#[test]
fn non_allowlisted_registry_crate_fails() {
    let repo = FixtureRepo::clean();
    repo.append(TV_A, "notallowed = \"1\"\n");
    repo.assert_violations(&["notallowed 1.0.0 is not in verifier/allowlist.toml"]);
}

#[test]
fn allowlisted_crate_at_another_major_version_fails() {
    let repo = FixtureRepo::clean();
    repo.replace(TV_A, "blake3 = \"1\"", "blake3 = \"0.9\"");
    repo.assert_violations(&["blake3 0.9.0 is outside the allowlisted version 1"]);
}

#[test]
fn scoped_crate_used_by_another_verifier_crate_fails() {
    let repo = FixtureRepo::clean();
    repo.append(TV_A, "clap = \"4\"\n");
    repo.assert_violations(&["tv-a depends on clap, which only tv-cli may use"]);
}

#[test]
fn scoped_crate_as_dev_dependency_is_allowed() {
    let repo = FixtureRepo::clean();
    repo.append(TV_A, "\n[dev-dependencies]\nclap = \"4\"\n");
    repo.assert_violations(&[]);
}

#[test]
fn malformed_allowlist_admits_nothing() {
    let repo = FixtureRepo::clean();
    repo.replace(
        "verifier/allowlist.toml",
        "version = \"1\"",
        "version = \"1.5\"",
    );
    repo.append(
        "verifier/allowlist.toml",
        "\n[[package]]\nname = \"x\"\nversion = \"1\"\nreason = \"r\"\nnote = \"?\"\n",
    );
    repo.assert_violations(&["blake3 has version `1.5`", "entry 2 has unknown key `note`"]);
}

#[test]
fn include_str_reaching_prover_sources_fails() {
    let repo = FixtureRepo::clean();
    repo.append(
        TV_A_LIB,
        "pub const STOLEN: &str = include_str!(\"../../../../crates/shared-util/src/lib.rs\");\n",
    );
    repo.assert_violations(&[
        "tv-a/src/lib.rs:4: include_str! `../../../../crates/shared-util/src/lib.rs`",
    ]);
}

#[test]
fn path_attribute_reaching_prover_sources_fails() {
    let repo = FixtureRepo::clean();
    repo.append(
        TV_A_LIB,
        "#[path = \"../../../../crates/shared-util/src/lib.rs\"]\nmod stolen;\n",
    );
    repo.assert_violations(&[
        "tv-a/src/lib.rs:4: #[path] `../../../../crates/shared-util/src/lib.rs`",
    ]);
}

#[test]
fn indirect_file_references_fail() {
    let repo = FixtureRepo::clean();
    repo.append(
        TV_A_LIB,
        concat!(
            "#[cfg_attr(all(), path = \"../../../../crates/shared-util/src/lib.rs\")]\n", // line 4
            "mod a;\n",
            "#[doc = include_str!(\"../../../../README.md\")]\n", // line 6
            "pub struct Documented;\n",
            "use core::include_bytes as smuggle;\n", // line 8
            "macro_rules! m { ($p:literal) => { include_bytes!($p) }; }\n", // line 9
            "pub const C: &str = include_str!(concat!(\"../\", \"x\"));\n", // line 10
            "pub const D: &[u8] = include_bytes!(\"\\x2e\\x2e/x\");\n", // line 11
            "pub const E: &str = include_str!(\"/etc/hosts\");\n", // line 12
        ),
    );
    repo.assert_violations(&[
        "lib.rs:4: #[path] `../../../../crates/shared-util/src/lib.rs` resolves outside",
        "lib.rs:6: include_str! `../../../../README.md` resolves outside",
        "lib.rs:8: renaming include_bytes",
        "lib.rs:9: include_bytes! needs a plain string literal",
        "lib.rs:10: include_str! needs a plain string literal",
        "lib.rs:11: include_bytes! needs a plain string literal",
        "lib.rs:12: include_str! uses an absolute path",
    ]);
}

#[test]
fn references_inside_verifier_and_mentions_in_comments_pass() {
    let repo = FixtureRepo::clean();
    repo.write("verifier/crates/tv-a/data.txt", "data\n");
    repo.append(
        TV_A_LIB,
        concat!(
            "// include_str!(\"../../../../crates/shared-util/src/lib.rs\") in a comment\n",
            "pub const NOTE: &str = \"include_str!(\\\"../../../../x\\\")\";\n",
            "pub const DATA: &str = include_str!(\"../data.txt\");\n",
            "pub const RAW: &str = include_str!(r#\"../data.txt\"#);\n",
        ),
    );
    repo.assert_violations(&[]);
}

#[cfg(unix)]
#[test]
fn symlink_under_verifier_fails() {
    let repo = FixtureRepo::clean();
    std::os::unix::fs::symlink(
        repo.path().join("crates/shared-util/src/lib.rs"),
        repo.path().join("verifier/crates/tv-a/src/linked.rs"),
    )
    .expect("create symlink");
    repo.assert_violations(&["verifier/crates/tv-a/src/linked.rs is a symlink"]);
}

#[test]
fn build_script_fails() {
    let repo = FixtureRepo::clean();
    repo.write(
        "verifier/crates/tv-a/build.rs",
        "// SPDX-License-Identifier: Apache-2.0 OR MIT\nfn main() {}\n",
    );
    repo.assert_violations(&["tv-a has a build script"]);
}

#[test]
fn toolchain_pin_mismatch_fails() {
    let repo = FixtureRepo::clean();
    let pin = repo::pinned_channel(&repo.path().join("rust-toolchain.toml"))
        .expect("root pin has a channel");
    repo.replace(
        "verifier/rust-toolchain.toml",
        &format!("channel = \"{pin}\""),
        "channel = \"1.85.0\"",
    );
    repo.assert_violations(&["verifier/rust-toolchain.toml pins"]);
}

#[test]
fn prover_normal_dependency_on_a_verifier_crate_fails() {
    let repo = FixtureRepo::clean();
    repo.replace(
        "crates/tessera-probe/Cargo.toml",
        "[dev-dependencies]",
        "[dependencies]",
    );
    repo.assert_violations(&["tessera-probe has a normal dependency on verifier crate tv-a"]);
}

#[test]
fn directory_named_target_inside_sources_is_scanned() {
    // Only verifier/target is skipped; a force-added src/target/ is ordinary source.
    let repo = FixtureRepo::clean();
    repo.append(TV_A_LIB, "mod target;\n");
    repo.write(
        "verifier/crates/tv-a/src/target/mod.rs",
        "pub const S: &str = include_str!(\"../../../../../crates/shared-util/src/lib.rs\");\n",
    );
    repo.assert_violations(&["tv-a/src/target/mod.rs:1: include_str!"]);
}

#[test]
fn included_code_must_be_a_rust_file() {
    // Code imported from a non-.rs file would escape the scan.
    let repo = FixtureRepo::clean();
    repo.write("verifier/crates/tv-a/code.txt", "pub fn f() {}\n");
    repo.append(
        TV_A_LIB,
        "include!(\"../code.txt\");\n#[path = \"../code.txt\"]\nmod code;\n",
    );
    repo.assert_violations(&[
        "lib.rs:4: include! `../code.txt` must name a .rs file",
        "lib.rs:5: #[path] `../code.txt` must name a .rs file",
    ]);
}

#[test]
fn raw_identifiers_are_recognized() {
    let repo = FixtureRepo::clean();
    repo.append(
        TV_A_LIB,
        concat!(
            "pub const S: &str = r#include_str!(\"../../../../crates/shared-util/src/lib.rs\");\n",
            "#[r#path = \"../../../../crates/shared-util/src/lib.rs\"]\nmod stolen;\n",
        ),
    );
    repo.assert_violations(&["lib.rs:4: include_str!", "lib.rs:5: #[path]"]);
}

#[test]
fn procedural_macro_crate_fails() {
    let repo = FixtureRepo::clean();
    repo.write(
        "verifier/crates/tv-macros/Cargo.toml",
        concat!(
            "[package]\nname = \"tv-macros\"\nversion.workspace = true\nedition.workspace = true\n",
            "license.workspace = true\npublish.workspace = true\n\n[lib]\nproc-macro = true\n",
        ),
    );
    repo.write(
        "verifier/crates/tv-macros/src/lib.rs",
        "// SPDX-License-Identifier: Apache-2.0 OR MIT\n",
    );
    repo.assert_violations(&["tv-macros is a procedural macro"]);
}

#[test]
fn patch_that_makes_a_verifier_crate_a_normal_dependency_fails() {
    // The declared dependency names a registry crate; only the resolved graph shows that the
    // root workspace's [patch] substitutes a verifier crate for it.
    let repo = FixtureRepo::clean();
    repo.append(
        "crates/shared-util/Cargo.toml",
        "\n[dependencies]\ntv-a = \"0.0.0\"\n",
    );
    repo.append(
        "Cargo.toml",
        "\n[patch.crates-io]\ntv-a = { path = \"verifier/crates/tv-a\" }\n",
    );
    repo.assert_violations(&["shared-util has a normal dependency on verifier crate tv-a"]);
}

#[test]
fn target_sources_outside_verifier_fail() {
    // The manifests stay inside verifier/; only their target paths point at prover sources.
    let repo = FixtureRepo::clean();
    repo.write(
        "crates/shared-util/src/main.rs",
        "// SPDX-License-Identifier: Apache-2.0 OR MIT\nfn main() {}\n",
    );
    repo.append(
        TV_A,
        "\n[lib]\npath = \"../../../crates/shared-util/src/lib.rs\"\n",
    );
    repo.append(
        "verifier/crates/tv-cli/Cargo.toml",
        "\n[[bin]]\nname = \"tv-verify\"\npath = \"../../../crates/shared-util/src/main.rs\"\n",
    );
    repo.assert_violations(&[
        "tv-a target `tv_a` compiles a source outside verifier/",
        "tv-cli target `tv-verify` compiles a source outside verifier/",
    ]);
}

#[test]
fn rewritten_or_extra_aliases_fail() {
    // A rewritten `xtask` alias could make `cargo xtask check-isolation` print help and exit 0.
    let repo = FixtureRepo::clean();
    repo.write(
        ".cargo/config.toml",
        "[alias]\nxtask = \"run --quiet --package xtask -- help\"\nb = \"build\"\n",
    );
    repo.assert_violations(&[
        "alias `xtask = \"run --quiet --package xtask -- help\"` is not allowed",
        "alias `b = \"build\"` is not allowed",
    ]);
}

#[test]
fn canonical_alias_passes() {
    let repo = FixtureRepo::clean();
    repo.write(
        ".cargo/config.toml",
        "[alias]\nxtask = \"run --quiet --package xtask --\"\n",
    );
    repo.assert_violations(&[]);
}

#[test]
fn cargo_configuration_that_could_redirect_dependencies_fails() {
    let repo = FixtureRepo::clean();
    repo.write(
        "verifier/.cargo/config.toml",
        "[net]\ngit-fetch-with-cli = true\n",
    );
    repo.write(
        ".cargo/config.toml",
        "[alias]\nxtask = \"run --quiet --package xtask --\"\n[build]\njobs = 1\n",
    );
    // A patch to a copy of the crate inside verifier/ passes DEP-1 and DEP-3; only check 6
    // sees the substitution.
    copy_dir(
        &repo.path().join("vendor/blake3-1.5.0"),
        &repo.path().join("verifier/patched/blake3"),
    );
    repo.append(
        "verifier/Cargo.toml",
        "\n[patch.crates-io]\nblake3 = { path = \"patched/blake3\" }\n",
    );
    repo.assert_violations(&[
        "verifier/.cargo: cargo configuration under verifier/ is not allowed",
        ".cargo/config.toml: only [alias] is allowed, found `build`",
        "verifier/Cargo.toml: [patch] can substitute any dependency",
    ]);
}

#[test]
fn git_and_foreign_registry_sources_fail() {
    let graph = serde_json::json!({
        "packages": [
            {"name": "blake3", "version": "1.5.0", "id": "a", "source": "git+https://example.com/blake3#abc"},
            {"name": "blake3", "version": "1.5.1", "id": "b", "source": "registry+https://example.com/index"},
        ],
        "workspace_members": [],
    });
    let allowlist = vec![AllowEntry {
        name: "blake3".to_owned(),
        spec: "1".to_owned(),
        compat: Compat::Major(1),
        direct_dependents: None,
    }];
    let violations = dep3_violations(&graph, &allowlist);
    assert_eq!(violations.len(), 2, "{violations:#?}");
    assert!(
        violations
            .iter()
            .all(|v| v.contains("only crates.io is allowed"))
    );
}

#[test]
fn semver_compatibility_classes() {
    let cases = [
        ("1.2.3", Some(Compat::Major(1))),
        ("2.0.0-rc.1+build", Some(Compat::Major(2))),
        ("0.10.7", Some(Compat::Minor(10))),
        ("0.0.3", Some(Compat::Patch(3))),
        ("1.2", None),
        ("x.y.z", None),
    ];
    for (version, class) in cases {
        assert_eq!(Compat::of_version(version), class, "{version}");
    }
    let specs = [
        ("1", Some(Compat::Major(1))),
        ("0.10", Some(Compat::Minor(10))),
        ("0.0.3", Some(Compat::Patch(3))),
        ("0", None),
        ("1.2", None),
        ("0.0", None),
        ("^1", None),
    ];
    for (spec, class) in specs {
        assert_eq!(Compat::of_spec(spec), class, "{spec}");
    }
}
