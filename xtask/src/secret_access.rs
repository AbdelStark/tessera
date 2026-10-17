// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The secret-access policy of 05 §3 OBS-1: a secret-class value leaves `Secret<T>` only through
//! `expose_secret` or `expose_secret_mut`, and only in the files `secret-access.toml` lists, each
//! with a reason. The policy lists two kinds of file:
//!
//! - a **reader** takes values out of their `Secret`: it names an accessor, or its entry is stale;
//! - a **borrower** (`borrows = true`) computes on values a reader lends it, such as a proving
//!   engine's kernel: it names no accessor and switches no rule off, and its entry names a prover
//!   source.
//!
//! Two layers hold the line.
//!
//! - **Names.** A prover source outside the policy may not name either accessor. The scan reads
//!   tokens, so it sees code under every `cfg` and feature, which clippy never compiles, and no
//!   lint level or configuration changes what it sees. Every compiled prover source is one it
//!   reads: a `.rs` file reached through `mod` at its default path. No source may pull in code
//!   another way (`include!`, a `path` attribute) and no manifest may point a target outside its
//!   package or at a file that is not `.rs`.
//! - **Clippy.** `clippy.toml` disallows both methods, so clippy, which CI runs with
//!   `-D warnings`, rejects every call it compiles and resolves, macro expansions included. A
//!   listed file opts back in with `#[allow(clippy::disallowed_methods)]`. This check fails
//!   whatever else would switch the rule off:
//!   - in a source outside the policy, allowing or expecting the rule, its former name,
//!     `clippy::style`, `clippy::all` or `warnings`, or lowering `warnings` to `warn`, which
//!     overrides `-D warnings`. Lint levels are matched wherever they appear, so an allowance
//!     inside `cfg_attr` or handed to a macro counts, and raw identifiers count as plain ones;
//!   - a manifest allowing any of them;
//!   - a clippy configuration other than the root `clippy.toml`, or a root `clippy.toml`
//!     without both rules. Clippy reads the configuration nearest each crate, so another file
//!     switches the rules off for the crates below it.
//!
//!   Cargo configuration could also rewrite lint levels or clippy's arguments (`rustflags`,
//!   `[env]`, wrappers); `check-isolation` (check 6) already holds it to the `xtask` alias.
//!
//! Code built to evade both layers at once is left to review and to the canary scan of OBS-3:
//! names and attributes assembled from pieces (token pasting, procedural macros, attributes named
//! by macro metavariables), a build script that rewrites clippy's arguments through
//! `cargo::rustc-env`, a listed file that hands the value out, and unsafe code that reads it
//! through a pointer. The verifier's workspace is out of scope: it cannot depend on the prover's
//! codec (RFC-0001).

use std::collections::BTreeSet;
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::str::FromStr;

use proc_macro2::{Delimiter, Ident, TokenStream, TokenTree};
use toml::{Table, Value as Toml};

use crate::isolation;

/// The policy file, at the repository root.
pub(crate) const POLICY: &str = "secret-access.toml";

/// The methods that hand out a secret-class value.
const ACCESSORS: [&str; 2] = ["expose_secret", "expose_secret_mut"];

/// The paths the root `clippy.toml` must disallow.
const RULES: [&str; 2] = [
    "tessera_codec::Secret::expose_secret",
    "tessera_codec::Secret::expose_secret_mut",
];

/// The rule, its former name and the lint groups that contain it.
const SILENCERS: [&str; 5] = [
    "clippy::disallowed_methods",
    "clippy::disallowed_method",
    "clippy::style",
    "clippy::all",
    "warnings",
];

/// The lint-level attributes.
const LEVELS: [&str; 5] = ["allow", "expect", "warn", "deny", "forbid"];

/// Whether setting `lint` to `level` switches the rule off under `-D warnings`: allowing or
/// expecting it or a group that contains it, or lowering `warnings` itself to `warn`.
fn silences(level: &str, lint: &str) -> bool {
    match level {
        "allow" | "expect" => SILENCERS.contains(&lint),
        "warn" => lint == "warnings",
        _ => false,
    }
}

/// Checks the policy, the root clippy configuration, and every prover source and manifest.
pub(crate) fn check(repo: &Path, files: &[PathBuf]) -> io::Result<Vec<String>> {
    let mut violations = Vec::new();
    let Policy { readers, borrowers } = match load_policy(&repo.join(POLICY)) {
        Ok(policy) => policy,
        Err(problems) => {
            violations.extend(problems);
            Policy::default()
        }
    };
    violations.extend(clippy_rules(repo));
    let mut naming = BTreeSet::new();
    for rel in files.iter().filter(|f| !f.starts_with("verifier")) {
        let shown = rel.display();
        let name = rel.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        if rel.extension().is_some_and(|e| e == "rs") {
            let text = fs::read_to_string(repo.join(rel))?;
            let Ok(stream) = TokenStream::from_str(&text) else {
                violations.push(format!(
                    "{shown}: does not tokenize, so this check cannot vouch for it"
                ));
                continue;
            };
            for (line, what) in isolation::code_inclusions(stream.clone()) {
                violations.push(format!(
                    "{shown}:{line}: {what}, which this check cannot follow; prover code \
                     enters a crate only through `mod` at its default path"
                ));
            }
            let mut found = Scan::default();
            scan(stream, false, &mut found);
            if !found.names.is_empty() {
                naming.insert(rel.clone());
            }
            if readers.contains(rel) {
                continue;
            }
            for line in found.names {
                violations.push(if borrowers.contains(rel) {
                    format!(
                        "{shown}:{line}: names a secret accessor, which a borrower computing on \
                         lent values never does; list the file in {POLICY} as a reader, without \
                         `borrows`"
                    )
                } else {
                    format!(
                        "{shown}:{line}: names a secret accessor; list the file in {POLICY} with \
                         a reason, or keep the value in its `Secret`"
                    )
                });
            }
            for (line, what) in found.silencing {
                violations.push(format!(
                    "{shown}:{line}: {what} switches off clippy's rule against `expose_secret`, \
                     which only the readers {POLICY} lists may do"
                ));
            }
        } else if name == "Cargo.toml" {
            let text = fs::read_to_string(repo.join(rel))?;
            let manifest = match text.parse::<Table>() {
                Ok(manifest) => manifest,
                Err(e) => {
                    violations.push(format!("{shown}: not TOML: {e}"));
                    continue;
                }
            };
            for lint in manifest_silencers(&manifest) {
                violations.push(format!(
                    "{shown}: `{lint}` switches off clippy's rule against `expose_secret` in \
                     every crate it covers; allow it only in the files {POLICY} lists"
                ));
            }
            for target in hidden_targets(&manifest) {
                violations.push(format!(
                    "{shown}: target path `{target}` is not a `.rs` file inside the package, \
                     where this check reads it"
                ));
            }
        } else if matches!(name, "clippy.toml" | ".clippy.toml") && rel != Path::new("clippy.toml")
        {
            violations.push(format!(
                "{shown}: shadows the root clippy.toml, and its rules against the secret \
                 accessors, for every crate below it"
            ));
        }
    }
    for rel in &readers {
        if !naming.contains(rel) {
            violations.push(format!(
                "{POLICY}: {} names no secret accessor; remove the entry, or mark a file that \
                 computes on lent values `borrows = true`",
                rel.display()
            ));
        }
    }
    let sources: BTreeSet<&PathBuf> = files
        .iter()
        .filter(|f| !f.starts_with("verifier") && f.extension().is_some_and(|e| e == "rs"))
        .collect();
    for rel in &borrowers {
        if !sources.contains(rel) {
            violations.push(format!(
                "{POLICY}: {} is not a prover source; remove the entry",
                rel.display()
            ));
        }
    }
    Ok(violations)
}

/// The files the policy lists, by kind.
#[derive(Default)]
struct Policy {
    /// Files that take values out of `Secret`.
    readers: Vec<PathBuf>,
    /// Files that compute on values a reader lends them (`borrows = true`).
    borrowers: Vec<PathBuf>,
}

/// The files the policy lists, or what is wrong with it.
fn load_policy(path: &Path) -> Result<Policy, Vec<String>> {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Policy::default()),
        Err(e) => return Err(vec![format!("{POLICY}: {e}")]),
    };
    let table: Table = text
        .parse()
        .map_err(|e| vec![format!("{POLICY}: not TOML: {e}")])?;
    let mut problems = Vec::new();
    let mut policy = Policy::default();
    for key in table.keys().filter(|k| *k != "module") {
        problems.push(format!("{POLICY}: unknown key `{key}`"));
    }
    let modules = table.get("module").and_then(Toml::as_array);
    for (i, entry) in modules.into_iter().flatten().enumerate() {
        let n = i + 1;
        let Some(entry) = entry.as_table() else {
            problems.push(format!("{POLICY}: module {n} is not a table"));
            continue;
        };
        for key in entry
            .keys()
            .filter(|k| !matches!(k.as_str(), "path" | "reason" | "borrows"))
        {
            problems.push(format!("{POLICY}: module {n} has unknown key `{key}`"));
        }
        let borrows = match entry.get("borrows") {
            None => false,
            Some(Toml::Boolean(true)) => true,
            Some(_) => {
                problems.push(format!(
                    "{POLICY}: module {n}: `borrows` is `true` or absent"
                ));
                false
            }
        };
        if entry
            .get("reason")
            .and_then(Toml::as_str)
            .is_none_or(|r| r.trim().is_empty())
        {
            problems.push(format!("{POLICY}: module {n} needs a reason"));
        }
        let listed = |p: &str| {
            let p = Path::new(p);
            policy
                .readers
                .iter()
                .chain(&policy.borrowers)
                .any(|l| l == p)
        };
        match entry.get("path").and_then(Toml::as_str) {
            Some(p) if listed(p) => {
                problems.push(format!("{POLICY}: module {n} repeats `{p}`"));
            }
            Some(p) if p.ends_with(".rs") && borrows => policy.borrowers.push(PathBuf::from(p)),
            Some(p) if p.ends_with(".rs") => policy.readers.push(PathBuf::from(p)),
            _ => problems.push(format!("{POLICY}: module {n} needs a `path` to a .rs file")),
        }
    }
    if problems.is_empty() {
        Ok(policy)
    } else {
        Err(problems)
    }
}

/// What is missing from the root `clippy.toml`'s rules against the accessors.
fn clippy_rules(repo: &Path) -> Vec<String> {
    let table = match fs::read_to_string(repo.join("clippy.toml")).map(|t| t.parse::<Table>()) {
        Ok(Ok(table)) => table,
        Ok(Err(e)) => return vec![format!("clippy.toml: not TOML: {e}")],
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return vec!["clippy.toml: missing; it holds the rules against the accessors".into()];
        }
        Err(e) => return vec![format!("clippy.toml: {e}")],
    };
    let paths: Vec<&str> = table
        .get("disallowed-methods")
        .and_then(Toml::as_array)
        .into_iter()
        .flatten()
        .filter_map(|rule| {
            rule.as_str()
                .or_else(|| rule.get("path").and_then(Toml::as_str))
        })
        .collect();
    RULES
        .iter()
        .filter(|rule| !paths.contains(rule))
        .map(|rule| format!("clippy.toml: `disallowed-methods` does not list `{rule}`"))
        .collect()
}

/// The lints a manifest's `[lints]` or `[workspace.lints]` sets so that they switch the rule off.
fn manifest_silencers(manifest: &Table) -> Vec<String> {
    let tables = [
        manifest.get("lints"),
        manifest.get("workspace").and_then(|w| w.get("lints")),
    ];
    let mut found = Vec::new();
    for lints in tables.into_iter().flatten() {
        for tool in ["clippy", "rust"] {
            let Some(group) = lints.get(tool).and_then(Toml::as_table) else {
                continue;
            };
            for (lint, level) in group {
                let level = level
                    .as_str()
                    .or_else(|| level.get("level").and_then(Toml::as_str))
                    .unwrap_or_default();
                // Cargo hands `disallowed-methods` to rustc, which reads it as `disallowed_methods`.
                let lint = lint.replace('-', "_");
                let lint = if tool == "clippy" {
                    format!("clippy::{lint}")
                } else {
                    lint
                };
                if silences(level, &lint) {
                    found.push(format!("{lint} = \"{level}\""));
                }
            }
        }
    }
    found
}

/// The target paths of a manifest that this check would not read: crate roots and build scripts
/// that are not `.rs` files, or that lie outside the package (absolute, or through `..`).
fn hidden_targets(manifest: &Table) -> Vec<String> {
    let lib = manifest.get("lib").and_then(|lib| lib.get("path"));
    let listed = ["bin", "test", "example", "bench"]
        .into_iter()
        .filter_map(|kind| manifest.get(kind).and_then(Toml::as_array))
        .flatten()
        .filter_map(|target| target.get("path"));
    let build = manifest.get("package").and_then(|p| p.get("build"));
    lib.into_iter()
        .chain(listed)
        .chain(build)
        .filter_map(Toml::as_str)
        .filter(|path| {
            let path = Path::new(path);
            path.extension().is_none_or(|e| e != "rs")
                || !path
                    .components()
                    .all(|c| matches!(c, Component::Normal(_) | Component::CurDir))
        })
        .map(str::to_owned)
        .collect()
}

/// What a source does that this check governs, by line.
#[derive(Default)]
struct Scan {
    /// Where it names an accessor.
    names: Vec<usize>,
    /// Where it switches the clippy rule off, and how.
    silencing: Vec<(usize, String)>,
}

/// An identifier as rustc resolves it: `r#allow` is `allow`.
fn unraw(ident: &Ident) -> String {
    let name = ident.to_string();
    match name.strip_prefix("r#") {
        Some(bare) => bare.to_owned(),
        None => name,
    }
}

/// Scans a token stream. Comments are not tokens and literals are opaque, so prose about the
/// accessors counts for nothing. Inside a lint level's arguments (`in_level`), a lint's path is
/// accounted for by the level itself.
fn scan(stream: TokenStream, in_level: bool, found: &mut Scan) {
    let tokens: Vec<TokenTree> = stream.into_iter().collect();
    for (i, token) in tokens.iter().enumerate() {
        match token {
            TokenTree::Ident(ident) => {
                let name = unraw(ident);
                let line = ident.span().start().line;
                if ACCESSORS.contains(&name.as_str()) {
                    found.names.push(line);
                }
                if let Some(args) = level_arguments(&tokens, i) {
                    lint_levels(&name, line, args, found);
                } else if name == "clippy" && !in_level {
                    // A silencing lint's path outside a lint level: an argument for a macro
                    // that writes the attribute.
                    if let Some(lint) = clippy_lint(&tokens[i + 1..]) {
                        found
                            .silencing
                            .push((line, format!("`{lint}`, outside a lint attribute,")));
                    }
                }
            }
            TokenTree::Group(group) => {
                let arguments = i
                    .checked_sub(1)
                    .is_some_and(|level| level_arguments(&tokens, level).is_some());
                scan(group.stream(), in_level || arguments, found);
            }
            _ => {}
        }
    }
}

/// The arguments of the lint level at `tokens[i]`, if it is one: `allow(...)` and its kin, but
/// not a call such as `.expect(...)` or `Option::expect(...)`.
fn level_arguments(tokens: &[TokenTree], i: usize) -> Option<TokenStream> {
    let Some(TokenTree::Ident(ident)) = tokens.get(i) else {
        return None;
    };
    let called = i
        .checked_sub(1)
        .and_then(|before| tokens.get(before))
        .is_some_and(|t| matches!(t, TokenTree::Punct(p) if matches!(p.as_char(), '.' | ':')));
    match tokens.get(i + 1) {
        Some(TokenTree::Group(args))
            if args.delimiter() == Delimiter::Parenthesis
                && !called
                && LEVELS.contains(&unraw(ident).as_str()) =>
        {
            Some(args.stream())
        }
        _ => None,
    }
}

/// The silencing lint `clippy::<lint>` that the tokens after `clippy` name, if they name one.
fn clippy_lint(rest: &[TokenTree]) -> Option<String> {
    match rest {
        [
            TokenTree::Punct(a),
            TokenTree::Punct(b),
            TokenTree::Ident(lint),
            ..,
        ] if a.as_char() == ':' && b.as_char() == ':' => {
            let lint = format!("clippy::{}", unraw(lint));
            SILENCERS.contains(&lint.as_str()).then_some(lint)
        }
        _ => None,
    }
}

/// The arguments of a lint level: lint names, and perhaps a `reason`.
fn lint_levels(level: &str, line: usize, args: TokenStream, found: &mut Scan) {
    let args: Vec<TokenTree> = args.into_iter().collect();
    for item in args.split(|t| matches!(t, TokenTree::Punct(p) if p.as_char() == ',')) {
        let from_macro_input = item
            .iter()
            .any(|t| matches!(t, TokenTree::Punct(p) if p.as_char() == '$'));
        if from_macro_input && matches!(level, "allow" | "expect" | "warn") {
            found
                .silencing
                .push((line, format!("`{level}` over a macro's input")));
            continue;
        }
        let lint: String = item
            .iter()
            .map(|t| match t {
                TokenTree::Ident(ident) => unraw(ident),
                other => other.to_string(),
            })
            .collect();
        if silences(level, &lint) {
            found.silencing.push((line, format!("`{level}({lint})`")));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::process::Command;

    use super::*;
    use crate::repo;
    use crate::testing::Fixture;

    /// The clippy layer: under the repository's `clippy.toml`, clippy rejects `expose_secret` in a
    /// module that has not opted in and accepts it in one that has.
    #[test]
    fn clippy_rejects_expose_secret_outside_an_opted_in_module() {
        let root = repo::root();
        let fixture = Fixture::new("secret-clippy");
        let codec = root.join("crates/tessera-codec");
        fixture.write(
            "Cargo.toml",
            &format!(
                "[package]\nname = \"probe\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\
                 publish = false\n\n[dependencies]\ntessera-codec = {{ path = {:?} }}\n\n\
                 [workspace]\n",
                codec.display().to_string()
            ),
        );
        for file in ["clippy.toml", "Cargo.lock"] {
            let text = fs::read_to_string(root.join(file)).expect("a repository file");
            fixture.write(file, &text);
        }
        let pin = repo::pinned_channel(&root.join("rust-toolchain.toml"));
        let clippy = |source: &str| {
            fixture.write("src/lib.rs", source);
            let mut cmd = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
            if let Some(channel) = &pin {
                cmd.env("RUSTUP_TOOLCHAIN", channel);
            }
            cmd.args(["clippy", "--quiet", "--target-dir"])
                .arg(root.join("target/xtask-fixtures/secret-clippy"))
                .args(["--", "-D", "warnings"])
                .current_dir(fixture.path())
                .output()
                .expect("run cargo clippy")
        };
        let leak = "//! A module that reads a secret.\n\n\
                    pub fn first(key: &tessera_codec::Secret<[u8; 16]>) -> u8 {\n    \
                    key.expose_secret()[0]\n}\n";
        let rejected = clippy(leak);
        let stderr = String::from_utf8_lossy(&rejected.stderr);
        assert!(!rejected.status.success(), "clippy accepted the call");
        assert!(
            stderr.contains("use of a disallowed method `tessera_codec::Secret::expose_secret`"),
            "{stderr}"
        );
        let opted_in = clippy(&format!("#![allow(clippy::disallowed_methods)]\n{leak}"));
        assert!(
            opted_in.status.success(),
            "{}",
            String::from_utf8_lossy(&opted_in.stderr)
        );
    }

    /// A root `clippy.toml` with both rules, in the two forms clippy accepts.
    const CLIPPY_TOML: &str = "disallowed-methods = [\n    \
                               { path = \"tessera_codec::Secret::expose_secret\" },\n    \
                               \"tessera_codec::Secret::expose_secret_mut\",\n]\n";

    const POLICY_WITH_A: &str =
        "[[module]]\npath = \"crates/a/src/lib.rs\"\nreason = \"witness engine\"\n";

    /// The check's findings over `files`, plus `CLIPPY_TOML` unless `files` has a root one.
    fn violations(files: &[(&str, &str)]) -> Vec<String> {
        let fixture = Fixture::new("secret-access");
        let mut rels = Vec::new();
        if !files.iter().any(|(rel, _)| *rel == "clippy.toml") {
            fixture.write("clippy.toml", CLIPPY_TOML);
            rels.push(PathBuf::from("clippy.toml"));
        }
        for (rel, text) in files {
            fixture.write(rel, text);
            rels.push(PathBuf::from(rel));
        }
        check(fixture.path(), &rels).expect("fixture is readable")
    }

    /// `found` holds one violation per needle, and nothing else.
    fn assert_exactly(found: &[String], needles: &[&str]) {
        assert_eq!(found.len(), needles.len(), "{found:#?}");
        for needle in needles {
            assert!(
                found.iter().any(|v| v.contains(needle)),
                "{needle}: {found:#?}"
            );
        }
    }

    #[test]
    fn only_listed_sources_name_the_accessors() {
        let found = violations(&[
            (
                "crates/a/src/lib.rs",
                "#![allow(clippy::disallowed_methods)]\n\
                 fn f(k: &Secret<u8>) -> u8 { *k.expose_secret() }\n",
            ),
            (
                "crates/b/src/lib.rs",
                concat!(
                    "fn f(k: &Secret<u8>) -> u8 {\n",
                    "    *k.expose_secret()\n",
                    "}\n",
                    "fn g() { let _ = Secret::<u8>::r#expose_secret_mut; }\n",
                    "fn h(k: &Secret<u8>) { Some(1).expect(&k.expose_secret().to_string()); }\n",
                    // Code clippy never compiles in CI.
                    "#[cfg(feature = \"unsafe-debug-dump\")]\n",
                    "fn dump(k: &Secret<u8>) -> u8 { *k.expose_secret() }\n",
                    "// k.expose_secret() in a comment\n",
                    "/// `expose_secret` in documentation\n",
                    "const S: &str = \"expose_secret\";\n",
                ),
            ),
            (
                "xtask/src/peek.rs",
                "fn f(k: &Secret<u8>) { k.expose_secret(); }\n",
            ),
            ("crates/c/src/lib.rs", "fn f() { \"unterminated }\n"),
            (
                "verifier/crates/tv-a/src/lib.rs",
                "fn f() { x.expose_secret() }\n",
            ),
            (POLICY, POLICY_WITH_A),
        ]);
        assert_exactly(
            &found,
            &[
                "crates/b/src/lib.rs:2: names a secret accessor",
                "crates/b/src/lib.rs:4: names a secret accessor",
                "crates/b/src/lib.rs:5: names a secret accessor",
                "crates/b/src/lib.rs:7: names a secret accessor",
                "xtask/src/peek.rs:1: names a secret accessor",
                "crates/c/src/lib.rs: does not tokenize",
            ],
        );
    }

    /// Every compiled source is one the scan reads.
    #[test]
    fn code_enters_a_crate_only_where_the_scan_reads_it() {
        let found = violations(&[
            (
                "crates/a/src/lib.rs",
                concat!(
                    "#[path = \"hidden.inc\"]\n",
                    "mod hidden;\n",
                    "include!(concat!(env!(\"OUT_DIR\"), \"/generated.rs\"));\n",
                    "#[cfg_attr(unix, path = \"unix.rs\")]\n",
                    "mod platform;\n",
                    "use std::include as pull;\n",
                    // Data, not code.
                    "const KERNEL: &[u8] = include_bytes!(\"kernel.bin\");\n",
                ),
            ),
            (
                "crates/a/Cargo.toml",
                "[package]\nname = \"a\"\nbuild = \"build.inc\"\n\n[lib]\npath = \"src/lib.rs\"\n\n\
                 [[bin]]\nname = \"x\"\npath = \"../../verifier/x.rs\"\n\n\
                 [[test]]\nname = \"t\"\npath = \"tests/t.rs\"\n",
            ),
            (
                "crates/b/Cargo.toml",
                "[package]\nname = \"b\"\n\n[lib]\npath = \"lib.txt\"\n",
            ),
        ]);
        assert_exactly(
            &found,
            &[
                "crates/a/src/lib.rs:1: #[path]",
                "crates/a/src/lib.rs:3: include!",
                "crates/a/src/lib.rs:4: #[path]",
                "crates/a/src/lib.rs:6: `include`",
                "crates/a/Cargo.toml: target path `build.inc`",
                "crates/a/Cargo.toml: target path `../../verifier/x.rs`",
                "crates/b/Cargo.toml: target path `lib.txt`",
            ],
        );
    }

    #[test]
    fn only_listed_sources_switch_the_rule_off() {
        let found = violations(&[
            (
                "crates/a/src/lib.rs",
                "#![allow(warnings)]\nfn f(k: &Secret<u8>) -> u8 { *k.expose_secret() }\n",
            ),
            (
                "crates/b/src/lib.rs",
                concat!(
                    "#[allow(clippy::disallowed_methods)]\n",
                    "fn f() {}\n",
                    "#[cfg_attr(test, expect(clippy::style))]\n",
                    "fn g() {}\n",
                    "mod m {\n",
                    "    #![allow(dead_code, warnings)]\n",
                    "    #![warn(warnings)]\n",
                    "}\n",
                    "#[allow(clippy::all, reason = \"x\")]\n",
                    "fn h() {}\n",
                    "#[r#allow(clippy::r#disallowed_methods)]\n",
                    "fn i() {}\n",
                    "quiet!(clippy::disallowed_method);\n",
                    "macro_rules! quiet { ($l:tt) => { #[allow($l)] fn j() {} } }\n",
                    // Lint levels that leave the rule on, and calls that are no lint level.
                    "#[deny(warnings, clippy::all)]\n",
                    "#[warn(clippy::style)]\n",
                    "#[forbid(clippy::disallowed_methods)]\n",
                    "fn k() {}\n",
                    "const S: &str = \"#[allow(warnings)]\";\n",
                    "macro_rules! m { ($e:expr, $why:expr) => { $e.expect($why) } }\n",
                ),
            ),
            (POLICY, POLICY_WITH_A),
        ]);
        assert_exactly(
            &found,
            &[
                "crates/b/src/lib.rs:1: `allow(clippy::disallowed_methods)`",
                "crates/b/src/lib.rs:3: `expect(clippy::style)`",
                "crates/b/src/lib.rs:6: `allow(warnings)`",
                "crates/b/src/lib.rs:7: `warn(warnings)`",
                "crates/b/src/lib.rs:9: `allow(clippy::all)`",
                "crates/b/src/lib.rs:11: `allow(clippy::disallowed_methods)`",
                "crates/b/src/lib.rs:13: `clippy::disallowed_method`, outside",
                "crates/b/src/lib.rs:14: `allow` over a macro's input",
            ],
        );
    }

    #[test]
    fn no_manifest_or_clippy_configuration_switches_the_rule_off() {
        let found = violations(&[
            (
                "clippy.toml",
                "disallowed-methods = [\"tessera_codec::Secret::expose_secret\"]\n",
            ),
            (
                "Cargo.toml",
                "[workspace]\n\n[workspace.lints.clippy]\n\
                 disallowed-methods = { level = \"allow\", priority = -1 }\n",
            ),
            (
                "crates/b/Cargo.toml",
                "[package]\nname = \"b\"\n\n[lints.clippy]\nstyle = \"allow\"\n\
                 all = \"warn\"\n\n[lints.rust]\nwarnings = { level = \"allow\" }\n\
                 unused = \"allow\"\n",
            ),
            ("crates/b/clippy.toml", ""),
            (".clippy.toml", ""),
            ("verifier/clippy.toml", ""),
        ]);
        assert_exactly(
            &found,
            &[
                "clippy.toml: `disallowed-methods` does not list \
                 `tessera_codec::Secret::expose_secret_mut`",
                "Cargo.toml: `clippy::disallowed_methods = \"allow\"`",
                "crates/b/Cargo.toml: `clippy::style = \"allow\"`",
                "crates/b/Cargo.toml: `warnings = \"allow\"`",
                "crates/b/clippy.toml: shadows the root clippy.toml",
                ".clippy.toml: shadows the root clippy.toml",
            ],
        );
    }

    #[test]
    fn the_policy_is_well_formed_and_current() {
        let found = violations(&[
            ("crates/a/src/lib.rs", "fn f() {}\n"),
            (
                POLICY,
                concat!(
                    "[[module]]\npath = \"crates/a/src/lib.rs\"\nreason = \"fine\"\n",
                    "[[module]]\npath = \"crates/c/src/lib.rs\"\n",
                    "[[module]]\nreason = \"no path\"\nnote = \"?\"\n",
                    "[[module]]\npath = \"crates/a/src/lib.rs\"\nreason = \"again\"\n",
                ),
            ),
        ]);
        assert_exactly(
            &found,
            &[
                "module 2 needs a reason",
                "module 3 has unknown key `note`",
                "module 3 needs a `path`",
                "module 4 repeats `crates/a/src/lib.rs`",
            ],
        );
        // An entry for a file that silences the rule but names no accessor is stale.
        let stale = violations(&[
            (
                "crates/a/src/lib.rs",
                "#![allow(clippy::disallowed_methods)]\n",
            ),
            (POLICY, POLICY_WITH_A),
        ]);
        assert_exactly(&stale, &["crates/a/src/lib.rs names no secret accessor"]);
        // `borrows` is `true` or absent, and a path is listed once whatever its kind.
        let malformed = violations(&[
            ("crates/a/src/lib.rs", "fn f() {}\n"),
            (
                POLICY,
                concat!(
                    "[[module]]\npath = \"crates/a/src/lib.rs\"\nborrows = true\nreason = \"x\"\n",
                    "[[module]]\npath = \"crates/a/src/lib.rs\"\nreason = \"again\"\n",
                    "[[module]]\npath = \"crates/b/src/lib.rs\"\nborrows = false\nreason = \"x\"\n",
                    "[[module]]\npath = \"crates/c/src/lib.rs\"\nborrows = \"yes\"\nreason = \"x\"\n",
                ),
            ),
        ]);
        assert_exactly(
            &malformed,
            &[
                "module 2 repeats `crates/a/src/lib.rs`",
                "module 3: `borrows` is `true` or absent",
                "module 4: `borrows` is `true` or absent",
            ],
        );
    }

    /// A borrower computes on values a reader lends it: it names no accessor, switches no rule
    /// off, and its entry names a prover source.
    #[test]
    fn a_borrower_reads_nothing_out_of_a_secret() {
        let borrower = |path: &str| {
            format!("[[module]]\npath = \"{path}\"\nborrows = true\nreason = \"kernel\"\n")
        };
        let policy: String = [
            POLICY_WITH_A.to_owned(),
            borrower("crates/a/src/kernel.rs"),
            borrower("crates/a/src/peek.rs"),
            borrower("crates/a/src/quiet.rs"),
            borrower("crates/a/src/gone.rs"),
            borrower("verifier/crates/tv-a/src/lib.rs"),
        ]
        .concat();
        let found = violations(&[
            (
                "crates/a/src/lib.rs",
                "#![allow(clippy::disallowed_methods)]\n\
                 fn f(k: &Secret<Vec<u8>>) -> u8 { kernel::sum(k.expose_secret()) }\n",
            ),
            (
                "crates/a/src/kernel.rs",
                "pub fn sum(values: &[u8]) -> u8 { values.iter().sum() }\n",
            ),
            (
                "crates/a/src/peek.rs",
                "fn f() {}\nfn g(k: &Secret<u8>) -> u8 { *k.expose_secret() }\n",
            ),
            (
                "crates/a/src/quiet.rs",
                "#![allow(clippy::disallowed_methods)]\nfn f() {}\n",
            ),
            ("verifier/crates/tv-a/src/lib.rs", "fn f() {}\n"),
            (POLICY, &policy),
        ]);
        assert_exactly(
            &found,
            &[
                "crates/a/src/peek.rs:2: names a secret accessor, which a borrower",
                "crates/a/src/quiet.rs:1: `allow(clippy::disallowed_methods)`",
                "crates/a/src/gone.rs is not a prover source",
                "verifier/crates/tv-a/src/lib.rs is not a prover source",
            ],
        );
    }
}
