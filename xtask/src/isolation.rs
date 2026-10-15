// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `cargo xtask check-isolation`: the verifier-independence gate of RFC-0001 'Enforcement'
//! (DEP-1 to DEP-3 in 01 §3; security property SP-6).
//!
//! The five checks of the RFC, plus a sixth that closes ways around them:
//!
//! 1. **DEP-1.** The verifier's resolved dependency graph contains no package named `tessera-*`
//!    and no path package outside `verifier/`, and every target of every path package compiles
//!    a source inside `verifier/` (`[lib] path` and its siblings can point anywhere).
//! 2. **DEP-3.** Every registry package in that graph comes from crates.io and is listed in
//!    `verifier/allowlist.toml` with a compatible version. A package with `direct-dependents`
//!    is a normal or build dependency of those verifier crates only.
//! 3. **Sources.** `include!`, `include_str!`, `include_bytes!` and `path` attributes in verifier
//!    sources take plain string literals that resolve inside `verifier/` (to `.rs` files when they
//!    import code); nothing renames those macros; there are no symlinks, and no build scripts or
//!    procedural macros, which run at build time and can read any file.
//! 4. **Toolchain.** `verifier/rust-toolchain.toml` pins the same toolchain as the root.
//! 5. **DEP-2.** In the prover's resolved graph, verifier crates appear only behind
//!    dev-dependency edges, and no prover workspace member lives under `verifier/`.
//! 6. **Cargo configuration.** No `[patch]` or `[replace]` in verifier manifests, no cargo
//!    configuration under `verifier/`, and nothing in the repository's `.cargo/config.toml` but
//!    the canonical `xtask` alias. Source replacement is invisible in `cargo metadata`, so
//!    without this check a vendored directory could ship arbitrary code under an allowlisted
//!    crate name; and a rewritten alias could make `cargo xtask check-isolation` a no-op. CI
//!    runs the gate as `cargo run -p xtask -- check-isolation`, not through the alias.

use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::str::FromStr;

use proc_macro2::{Delimiter, TokenStream, TokenTree};
use serde_json::Value as Json;
use toml::{Table, Value as Toml};

use crate::repo;

/// The source id cargo reports for crates.io packages, whatever the index protocol.
const CRATES_IO: &str = "registry+https://github.com/rust-lang/crates.io-index";

/// The only alias the repository's cargo configuration may define (check 6).
const XTASK_ALIAS: (&str, &str) = ("xtask", "run --quiet --package xtask --");

/// Built-in macros that read files at compile time.
const INCLUDE_MACROS: [&str; 3] = ["include", "include_str", "include_bytes"];

/// How the gate runs cargo.
pub(crate) struct Cargo {
    /// The cargo binary.
    pub(crate) program: OsString,
    /// Extra `--config KEY=VALUE` arguments; tests use them to serve crates.io from a vendored
    /// directory, which keeps them offline.
    pub(crate) config: Vec<String>,
}

impl Cargo {
    /// The cargo that runs this tool, without extra configuration.
    pub(crate) fn from_env() -> Self {
        Self {
            program: std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()),
            config: Vec::new(),
        }
    }

    fn command(&self, dir: &Path, toolchain: Option<&str>) -> Command {
        let mut cmd = Command::new(&self.program);
        cmd.current_dir(dir);
        for config in &self.config {
            cmd.arg("--config").arg(config);
        }
        // Run every workspace with the root pin, so a mismatched verifier pin is reported by
        // check 4 instead of making rustup fetch another toolchain.
        if let Some(channel) = toolchain {
            cmd.env("RUSTUP_TOOLCHAIN", channel);
        }
        cmd
    }

    /// `cargo metadata` for the workspace in `dir` with its resolved graph, as JSON.
    fn metadata(&self, dir: &Path, toolchain: Option<&str>) -> Result<Json, String> {
        let mut cmd = self.command(dir, toolchain);
        cmd.args(["metadata", "--format-version", "1", "--locked"]);
        let output = cmd.output().map_err(|e| format!("cannot run cargo: {e}"))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("cargo metadata failed: {}", stderr.trim()));
        }
        serde_json::from_slice(&output.stdout).map_err(|e| format!("cargo metadata output: {e}"))
    }
}

/// Runs every check on the repository at `repo` and returns `(check, violations)` pairs.
pub(crate) fn checks(repo: &Path, cargo: &Cargo) -> io::Result<Vec<(&'static str, Vec<String>)>> {
    let repo = repo.canonicalize()?;
    let verifier = repo.join("verifier");
    if !verifier.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "no verifier/ directory",
        ));
    }
    let toolchain = repo::pinned_channel(&repo.join("rust-toolchain.toml"));
    let graph = cargo.metadata(&verifier, toolchain.as_deref());
    let prover = cargo.metadata(&repo, toolchain.as_deref());

    let allowlist = load_allowlist(&verifier.join("allowlist.toml"))?;
    let mut dep1 = Vec::new();
    let mut dep3 = allowlist.as_ref().err().cloned().unwrap_or_default();
    let mut sources = source_violations(&verifier)?;
    match &graph {
        Ok(graph) => {
            dep1.extend(dep1_violations(graph, &verifier));
            if let Ok(entries) = &allowlist {
                dep3.extend(dep3_violations(graph, entries));
            }
            sources.extend(build_time_code_violations(graph));
        }
        Err(e) => {
            dep1.push(format!("verifier/: {e}"));
            dep3.push("not checked: no dependency graph for verifier/".to_owned());
        }
    }
    let dep2 = match &prover {
        Ok(prover) => dep2_violations(prover, &verifier),
        Err(e) => vec![format!("repository root: {e}")],
    };
    Ok(vec![
        (
            "DEP-1: no prover crate in the verifier's dependency graph",
            dep1,
        ),
        ("DEP-3: third-party packages are allowlisted", dep3),
        ("verifier sources reach nothing outside verifier/", sources),
        (
            "toolchain pins are equal",
            toolchain_violations(&repo, &verifier)?,
        ),
        (
            "DEP-2: prover crates use verifier crates only as dev-dependencies",
            dep2,
        ),
        (
            "cargo configuration cannot redirect verifier dependencies",
            config_violations(&repo, &verifier)?,
        ),
    ])
}

/// Runs every check, reports on stdout and returns whether the repository is isolated.
pub(crate) fn run(repo: &Path) -> io::Result<bool> {
    let mut clean = true;
    for (name, violations) in checks(repo, &Cargo::from_env())? {
        if violations.is_empty() {
            println!("ok    {name}");
        } else {
            clean = false;
            println!("FAIL  {name}");
            for violation in violations {
                println!("      {violation}");
            }
        }
    }
    Ok(clean)
}

// ----- Check 1: DEP-1 -----------------------------------------------------------------------

fn packages(graph: &Json) -> impl Iterator<Item = &Json> {
    graph["packages"].as_array().into_iter().flatten()
}

fn str_field<'a>(value: &'a Json, key: &str) -> &'a str {
    value[key].as_str().unwrap_or_default()
}

/// Where a path really is: the canonical path when it exists, its lexical normalization otherwise.
fn resolve(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| normalize(path))
}

/// DEP-1. A path package is judged by where its code lives, not only by where its manifest
/// sits: target keys such as `[lib] path` let a manifest inside `verifier/` compile sources from
/// anywhere, and `targets[].src_path` is cargo's own record of what it compiles.
fn dep1_violations(graph: &Json, verifier: &Path) -> Vec<String> {
    let mut violations = Vec::new();
    for package in packages(graph) {
        let name = str_field(package, "name");
        if name.starts_with("tessera-") {
            violations.push(format!("{name} is a prover crate name"));
        }
        if !package["source"].is_null() {
            continue;
        }
        let manifest = Path::new(str_field(package, "manifest_path"));
        let dir = resolve(manifest.parent().unwrap_or(manifest));
        if !dir.starts_with(verifier) {
            violations.push(format!(
                "{name} is a path package outside verifier/: {}",
                dir.display()
            ));
            continue;
        }
        for target in package["targets"].as_array().into_iter().flatten() {
            let src = resolve(Path::new(str_field(target, "src_path")));
            if !src.starts_with(verifier) {
                violations.push(format!(
                    "{name} target `{}` compiles a source outside verifier/: {}",
                    str_field(target, "name"),
                    src.display()
                ));
            }
        }
    }
    violations
}

// ----- Check 2: DEP-3 -----------------------------------------------------------------------

/// SemVer compatibility class: Cargo treats versions in one class as compatible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Compat {
    /// `MAJOR` with `MAJOR >= 1`.
    Major(u64),
    /// `0.MINOR` with `MINOR >= 1`.
    Minor(u64),
    /// `0.0.PATCH`.
    Patch(u64),
}

impl Compat {
    /// The class of a full package version such as `1.2.3` or `0.10.0-rc.1+meta`.
    fn of_version(version: &str) -> Option<Self> {
        let core = version.split(['-', '+']).next()?;
        let parts: Vec<u64> = core
            .split('.')
            .map(str::parse)
            .collect::<Result<_, _>>()
            .ok()?;
        match parts[..] {
            [major, _, _] if major > 0 => Some(Self::Major(major)),
            [0, minor, _] if minor > 0 => Some(Self::Minor(minor)),
            [0, 0, patch] => Some(Self::Patch(patch)),
            _ => None,
        }
    }

    /// The class named by an allowlist entry: `1`, `0.10` or `0.0.3`, nothing else.
    fn of_spec(spec: &str) -> Option<Self> {
        let parts: Vec<u64> = spec
            .split('.')
            .map(str::parse)
            .collect::<Result<_, _>>()
            .ok()?;
        match parts[..] {
            [major] if major > 0 => Some(Self::Major(major)),
            [0, minor] if minor > 0 => Some(Self::Minor(minor)),
            [0, 0, patch] => Some(Self::Patch(patch)),
            _ => None,
        }
    }
}

#[derive(Debug)]
struct AllowEntry {
    name: String,
    spec: String,
    compat: Compat,
    direct_dependents: Option<Vec<String>>,
}

/// Parses `allowlist.toml`. A malformed file is a list of violations, not an I/O error: the
/// file is the policy, and a policy that does not parse admits nothing.
fn load_allowlist(path: &Path) -> io::Result<Result<Vec<AllowEntry>, Vec<String>>> {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return Ok(Err(vec!["verifier/allowlist.toml is missing".to_owned()]));
        }
        Err(e) => return Err(e),
    };
    let table = match text.parse::<Table>() {
        Ok(table) => table,
        Err(e) => {
            return Ok(Err(vec![format!(
                "verifier/allowlist.toml does not parse: {e}"
            )]));
        }
    };
    let mut errors = Vec::new();
    if let Some(key) = table.keys().find(|k| *k != "package") {
        errors.push(format!("allowlist: unknown top-level key `{key}`"));
    }
    let mut entries: Vec<AllowEntry> = Vec::new();
    let empty = Vec::new();
    for (i, item) in table
        .get("package")
        .and_then(Toml::as_array)
        .unwrap_or(&empty)
        .iter()
        .enumerate()
    {
        let Some(item) = item.as_table() else {
            errors.push(format!("allowlist: entry {i} is not a table"));
            continue;
        };
        if let Some(key) = item
            .keys()
            .find(|k| !["name", "version", "reason", "direct-dependents"].contains(&k.as_str()))
        {
            errors.push(format!("allowlist: entry {i} has unknown key `{key}`"));
        }
        let text = |key: &str| {
            item.get(key)
                .and_then(Toml::as_str)
                .unwrap_or_default()
                .trim()
                .to_owned()
        };
        let (name, spec, reason) = (text("name"), text("version"), text("reason"));
        if name.is_empty() || reason.is_empty() {
            errors.push(format!(
                "allowlist: entry {i} needs a non-empty name and reason"
            ));
            continue;
        }
        let Some(compat) = Compat::of_spec(&spec) else {
            errors.push(format!(
                "allowlist: {name} has version `{spec}`; expected a major version (`1`), `0.MINOR` or `0.0.PATCH`"
            ));
            continue;
        };
        let direct_dependents = match item.get("direct-dependents") {
            None => None,
            Some(list) => match list
                .as_array()
                .map(|a| a.iter().map(Toml::as_str).collect::<Option<Vec<_>>>())
            {
                Some(Some(names)) => Some(names.into_iter().map(str::to_owned).collect()),
                _ => {
                    errors.push(format!(
                        "allowlist: {name} direct-dependents must be a list of crate names"
                    ));
                    continue;
                }
            },
        };
        if entries.iter().any(|e| e.name == name) {
            errors.push(format!("allowlist: {name} is listed twice"));
            continue;
        }
        entries.push(AllowEntry {
            name,
            spec,
            compat,
            direct_dependents,
        });
    }
    Ok(if errors.is_empty() {
        Ok(entries)
    } else {
        Err(errors)
    })
}

fn dep3_violations(graph: &Json, allowlist: &[AllowEntry]) -> Vec<String> {
    let mut violations = Vec::new();
    let entry = |name: &str| allowlist.iter().find(|e| e.name == name);
    for package in packages(graph) {
        let Some(source) = package["source"].as_str() else {
            continue;
        };
        let (name, version) = (str_field(package, "name"), str_field(package, "version"));
        if source != CRATES_IO {
            violations.push(format!(
                "{name} {version} comes from {source}; only crates.io is allowed"
            ));
            continue;
        }
        match entry(name) {
            None => violations.push(format!(
                "{name} {version} is not in verifier/allowlist.toml"
            )),
            Some(e) if Compat::of_version(version) != Some(e.compat) => violations.push(format!(
                "{name} {version} is outside the allowlisted version {}",
                e.spec
            )),
            Some(_) => {}
        }
    }
    let members: Vec<&str> = graph["workspace_members"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Json::as_str)
        .collect();
    for member in packages(graph).filter(|p| members.contains(&str_field(p, "id"))) {
        let member_name = str_field(member, "name");
        for dep in member["dependencies"].as_array().into_iter().flatten() {
            // Dev-dependencies are not part of the verifier binary; they are still allowlisted.
            if dep["kind"].as_str() == Some("dev") || dep["source"].as_str() != Some(CRATES_IO) {
                continue;
            }
            let dep_name = str_field(dep, "name");
            if let Some(allowed) = entry(dep_name).and_then(|e| e.direct_dependents.as_ref()) {
                if !allowed.iter().any(|a| a == member_name) {
                    violations.push(format!(
                        "{member_name} depends on {dep_name}, which only {} may use",
                        allowed.join(", ")
                    ));
                }
            }
        }
    }
    violations
}

// ----- Check 3: sources ---------------------------------------------------------------------

/// Every file and directory under `root`, recursively and sorted, except the workspace build
/// directory `<root>/target`. Only that exact directory is skipped: a directory named `target`
/// anywhere else could hold force-added sources. Symlinks are listed, never followed.
fn walk(root: &Path) -> io::Result<Vec<PathBuf>> {
    fn visit(dir: &Path, build_dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
        let mut entries: Vec<PathBuf> = fs::read_dir(dir)?
            .map(|e| e.map(|e| e.path()))
            .collect::<io::Result<_>>()?;
        entries.sort();
        for path in entries {
            let is_dir = fs::symlink_metadata(&path)?.is_dir();
            if is_dir && path == build_dir {
                continue;
            }
            out.push(path.clone());
            if is_dir {
                visit(&path, build_dir, out)?;
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    visit(root, &root.join("target"), &mut out)?;
    Ok(out)
}

/// Lexical normalization: `.` dropped, `..` pops a component. Exact here because the check
/// rejects symlinks under `verifier/`.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other),
        }
    }
    out
}

/// A file reference found in Rust source: what referenced it, where, its literal value (`None`
/// when the argument is not a plain string literal), and whether it imports code (`include!`,
/// `path`) rather than data (`include_str!`, `include_bytes!`).
struct Reference {
    what: String,
    line: usize,
    value: Option<String>,
    imports_code: bool,
}

/// An identifier without its raw prefix: `r#path` names the same item as `path`.
fn ident_name(ident: &proc_macro2::Ident) -> String {
    let name = ident.to_string();
    name.strip_prefix("r#").map(str::to_owned).unwrap_or(name)
}

/// The value of a plain string literal (`"..."` without escapes, or a raw string), else `None`.
fn plain_string(token: &TokenTree) -> Option<String> {
    let TokenTree::Literal(lit) = token else {
        return None;
    };
    let text = lit.to_string();
    if let Some(inner) = text.strip_prefix('"').and_then(|t| t.strip_suffix('"')) {
        return (!inner.contains('\\')).then(|| inner.to_owned());
    }
    let raw = text.strip_prefix('r')?;
    let hashes = raw.len() - raw.trim_start_matches('#').len();
    let fence = "#".repeat(hashes);
    raw.strip_prefix(&format!("{fence}\""))?
        .strip_suffix(&format!("\"{fence}"))
        .map(str::to_owned)
}

fn line_of(token: &TokenTree) -> usize {
    token.span().start().line
}

/// Finds include macros, renames of them, and `path` attributes in a token stream, recursively.
fn scan(stream: TokenStream, out: &mut Vec<Reference>) {
    let tokens: Vec<TokenTree> = stream.into_iter().collect();
    for (i, token) in tokens.iter().enumerate() {
        match token {
            TokenTree::Ident(ident) if INCLUDE_MACROS.contains(&ident_name(ident).as_str()) => {
                let name = ident_name(ident);
                match (tokens.get(i + 1), tokens.get(i + 2)) {
                    (Some(TokenTree::Punct(bang)), Some(TokenTree::Group(group)))
                        if bang.as_char() == '!' =>
                    {
                        let args: Vec<TokenTree> = group.stream().into_iter().collect();
                        let value = match &args[..] {
                            [single] => plain_string(single),
                            _ => None,
                        };
                        out.push(Reference {
                            what: format!("{name}!"),
                            line: line_of(token),
                            value,
                            imports_code: name == "include",
                        });
                    }
                    (Some(TokenTree::Ident(next)), _) if next == "as" => out.push(Reference {
                        what: format!("renaming {name}"),
                        line: line_of(token),
                        value: None,
                        imports_code: false,
                    }),
                    _ => {}
                }
            }
            TokenTree::Punct(hash) if hash.as_char() == '#' => {
                let mut j = i + 1;
                if matches!(tokens.get(j), Some(TokenTree::Punct(p)) if p.as_char() == '!') {
                    j += 1;
                }
                if let Some(TokenTree::Group(group)) = tokens.get(j) {
                    if group.delimiter() == Delimiter::Bracket {
                        scan_attribute(group.stream(), out);
                    }
                }
            }
            TokenTree::Group(group) => scan(group.stream(), out),
            _ => {}
        }
    }
}

/// Records `path = ...` in an attribute body, looking inside `cfg_attr(predicate, ...)`.
fn scan_attribute(stream: TokenStream, out: &mut Vec<Reference>) {
    let tokens: Vec<TokenTree> = stream.into_iter().collect();
    match &tokens[..] {
        [TokenTree::Ident(path), rest @ ..] if ident_name(path) == "path" => {
            let value = match rest {
                [TokenTree::Punct(eq), lit] if eq.as_char() == '=' => plain_string(lit),
                _ => None,
            };
            out.push(Reference {
                what: "#[path]".to_owned(),
                line: line_of(&tokens[0]),
                value,
                imports_code: true,
            });
        }
        [TokenTree::Ident(cfg_attr), TokenTree::Group(args)]
            if ident_name(cfg_attr) == "cfg_attr" =>
        {
            // Skip the predicate, then check each attribute separated by top-level commas.
            let args: Vec<TokenTree> = args.stream().into_iter().collect();
            for attr in args
                .split(|t| matches!(t, TokenTree::Punct(p) if p.as_char() == ','))
                .skip(1)
            {
                scan_attribute(attr.iter().cloned().collect(), out);
            }
        }
        _ => {}
    }
}

fn source_violations(verifier: &Path) -> io::Result<Vec<String>> {
    let mut violations = Vec::new();
    for path in walk(verifier)? {
        let rel = path
            .strip_prefix(verifier)
            .unwrap_or(&path)
            .display()
            .to_string();
        if fs::symlink_metadata(&path)?.file_type().is_symlink() {
            violations.push(format!("verifier/{rel} is a symlink"));
            continue;
        }
        if path.extension().is_none_or(|ext| ext != "rs") {
            continue;
        }
        let text = fs::read_to_string(&path)?;
        let stream = match TokenStream::from_str(&text) {
            Ok(stream) => stream,
            Err(e) => {
                violations.push(format!("verifier/{rel} does not tokenize as Rust: {e}"));
                continue;
            }
        };
        let mut references = Vec::new();
        scan(stream, &mut references);
        let dir = path.parent().unwrap_or(verifier);
        for r in references {
            let at = format!("verifier/{rel}:{}", r.line);
            match r.value {
                None => violations.push(format!(
                    "{at}: {} needs a plain string literal argument",
                    r.what
                )),
                Some(value) if Path::new(&value).is_absolute() => {
                    violations.push(format!("{at}: {} uses an absolute path `{value}`", r.what));
                }
                Some(value) if !normalize(&dir.join(&value)).starts_with(verifier) => {
                    violations.push(format!(
                        "{at}: {} `{value}` resolves outside verifier/",
                        r.what
                    ));
                }
                // Only `.rs` files are scanned, so imported code must live in one.
                Some(value) if r.imports_code && !value.ends_with(".rs") => {
                    violations.push(format!("{at}: {} `{value}` must name a .rs file", r.what));
                }
                Some(_) => {}
            }
        }
    }
    Ok(violations)
}

/// Verifier crates that run code at build time: build scripts and procedural macros can read
/// any file, so neither is allowed.
fn build_time_code_violations(graph: &Json) -> Vec<String> {
    let members: Vec<&str> = graph["workspace_members"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Json::as_str)
        .collect();
    let mut violations = Vec::new();
    for package in packages(graph).filter(|p| members.contains(&str_field(p, "id"))) {
        let kinds: Vec<&str> = package["targets"]
            .as_array()
            .into_iter()
            .flatten()
            .flat_map(|t| {
                t["kind"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(Json::as_str)
            })
            .collect();
        let name = str_field(package, "name");
        if kinds.contains(&"custom-build") {
            violations.push(format!(
                "{name} has a build script; build-time code can read any file"
            ));
        }
        if kinds.contains(&"proc-macro") {
            violations.push(format!(
                "{name} is a procedural macro; build-time code can read any file"
            ));
        }
    }
    violations
}

// ----- Check 4: toolchain -------------------------------------------------------------------

fn toolchain_violations(repo: &Path, verifier: &Path) -> io::Result<Vec<String>> {
    let root = repo::toolchain_table(&repo.join("rust-toolchain.toml"))?;
    let ours = repo::toolchain_table(&verifier.join("rust-toolchain.toml"))?;
    Ok(match (root, ours) {
        (Some(root), Some(ours)) if root == ours => Vec::new(),
        (Some(root), Some(ours)) => {
            let channel = |t: &Toml| {
                t.get("channel")
                    .and_then(Toml::as_str)
                    .unwrap_or("?")
                    .to_owned()
            };
            let (ours_channel, root_channel) = (channel(&ours), channel(&root));
            if ours_channel == root_channel {
                vec![format!(
                    "[toolchain] tables differ: verifier {ours:?}, root {root:?}"
                )]
            } else {
                vec![format!(
                    "verifier/rust-toolchain.toml pins {ours_channel}, the root rust-toolchain.toml pins {root_channel}"
                )]
            }
        }
        (None, _) => vec!["rust-toolchain.toml has no [toolchain] table".to_owned()],
        (_, None) => vec!["verifier/rust-toolchain.toml has no [toolchain] table".to_owned()],
    })
}

// ----- Check 5: DEP-2 -----------------------------------------------------------------------

/// DEP-2 on the prover's resolved graph, so a dependency that `[patch]` turns into a verifier
/// crate is caught as well as a declared path dependency.
fn dep2_violations(prover: &Json, verifier: &Path) -> Vec<String> {
    let inside = |manifest: &str| {
        let path = Path::new(manifest);
        path.canonicalize()
            .unwrap_or_else(|_| path.to_path_buf())
            .starts_with(verifier)
    };
    let by_id: Vec<(&str, &str, bool)> = packages(prover)
        .map(|p| {
            (
                str_field(p, "id"),
                str_field(p, "name"),
                inside(str_field(p, "manifest_path")),
            )
        })
        .collect();
    let lookup = |id: &str| {
        by_id
            .iter()
            .find(|(pid, ..)| *pid == id)
            .map(|&(_, name, inside)| (name, inside))
    };
    let members: Vec<&str> = prover["workspace_members"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Json::as_str)
        .collect();

    let mut violations = Vec::new();
    for member in &members {
        if let Some((name, true)) = lookup(member) {
            violations.push(format!(
                "prover workspace member {name} lives under verifier/"
            ));
        }
    }
    for node in prover["resolve"]["nodes"].as_array().into_iter().flatten() {
        let Some((from, false)) = lookup(str_field(node, "id")) else {
            continue;
        };
        for dep in node["deps"].as_array().into_iter().flatten() {
            let Some((to, true)) = lookup(str_field(dep, "pkg")) else {
                continue;
            };
            for kind in dep["dep_kinds"].as_array().into_iter().flatten() {
                let kind = kind["kind"].as_str().unwrap_or("normal");
                if kind != "dev" {
                    violations.push(format!(
                        "{from} has a {kind} dependency on verifier crate {to}; only dev-dependencies are allowed"
                    ));
                }
            }
        }
    }
    violations.sort();
    violations.dedup();
    violations
}

// ----- Check 6: cargo configuration ---------------------------------------------------------

fn config_violations(repo: &Path, verifier: &Path) -> io::Result<Vec<String>> {
    let mut violations = Vec::new();
    for path in &walk(verifier)? {
        let rel = path
            .strip_prefix(repo)
            .unwrap_or(path)
            .display()
            .to_string();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        if name == ".cargo" {
            violations.push(format!(
                "{rel}: cargo configuration under verifier/ is not allowed"
            ));
        }
        if name == "Cargo.toml" {
            if let Ok(manifest) = fs::read_to_string(path)?.parse::<Table>() {
                for key in ["patch", "replace"] {
                    if manifest.contains_key(key) {
                        violations.push(format!("{rel}: [{key}] can substitute any dependency"));
                    }
                }
            }
        }
    }
    for name in ["config.toml", "config"] {
        let path = repo.join(".cargo").join(name);
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let table = match text.parse::<Table>() {
            Ok(table) => table,
            Err(e) => {
                violations.push(format!(".cargo/{name} does not parse: {e}"));
                continue;
            }
        };
        for key in table.keys().filter(|k| *k != "alias") {
            violations.push(format!(
                ".cargo/{name}: only [alias] is allowed, found `{key}`"
            ));
        }
        // An alias body decides what `cargo xtask` runs, so a rewritten alias could turn this
        // very gate into a no-op. Only the canonical alias is accepted.
        for (alias, body) in table
            .get("alias")
            .and_then(Toml::as_table)
            .into_iter()
            .flatten()
        {
            let body = body.as_str().unwrap_or_default();
            if (alias.as_str(), body) != XTASK_ALIAS {
                violations.push(format!(
                    ".cargo/{name}: alias `{alias} = {body:?}` is not allowed; the only alias is \
                     `{} = {:?}`",
                    XTASK_ALIAS.0, XTASK_ALIAS.1
                ));
            }
        }
    }
    Ok(violations)
}

#[cfg(test)]
mod tests;
