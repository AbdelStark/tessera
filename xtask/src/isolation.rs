// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `cargo xtask check-isolation`: the verifier-independence gate of RFC-0001 'Enforcement'
//! (DEP-1 to DEP-3 in 01 §3; security property SP-6).
//!
//! The eight checks the RFC lists:
//!
//! 1. **DEP-1.** The verifier's resolved dependency graph contains no package named `tessera-*`.
//!    Every path package is a member of the verifier workspace, inside `verifier/` but outside
//!    its build directory `verifier/target`. Every target of a path package has a `.rs` crate
//!    root inside that package's directory: `[lib] path` and its siblings can point anywhere.
//! 2. **DEP-3.** Every registry package in that graph comes from crates.io and is listed in
//!    `verifier/allowlist.toml` with a compatible version. A package with `direct-dependents`
//!    is a normal or build dependency of those verifier crates only.
//! 3. **Sources.** A token scan of every `.rs` file under `verifier/`, which sees the code of
//!    every `cfg`:
//!    - There is no `include!`, no `path` attribute and no `debugger_visualizer` attribute, so
//!      code enters a crate only through `mod` declarations at their default paths and through
//!      dependencies, which the graph checks see.
//!    - `include_str!` and `include_bytes!` are direct calls with plain string literals naming
//!      files of their own package.
//!    - Macro bodies hold no include macro, since its path would resolve at each call site.
//!      They call no macro named by a metavariable, and no attribute name in them holds one,
//!      since the caller could make it `path`. A metavariable in an attribute's value, as in
//!      `#[doc = $doc]`, is allowed: the caller's tokens are scanned where they are written.
//!    - There are no symlinks and no names with control characters.
//!    - There are no build scripts or procedural macros, which run at build time and can read any
//!      file.
//! 4. **Toolchain.** `verifier/rust-toolchain.toml` pins the same toolchain as the root.
//! 5. **DEP-2.** In the prover's resolved graph, verifier crates appear only behind
//!    dev-dependency edges, and no prover workspace member lives under `verifier/`.
//! 6. **Cargo configuration.** No `[patch]` or `[replace]` in verifier manifests, no cargo
//!    configuration under `verifier/`, and nothing in the repository's `.cargo/config.toml` but
//!    the canonical `xtask` alias. Source replacement is invisible in `cargo metadata`, so
//!    without this check a vendored directory could ship arbitrary code under an allowlisted
//!    crate name; and a rewritten alias could make `cargo xtask check-isolation` a no-op. CI
//!    runs the gate as `cargo run -p xtask -- check-isolation`, not through the alias. The
//!    secret-access check of `cargo xtask lint` (05 §3 OBS-1) relies on this check too:
//!    `rustflags`, `[env]` and compiler wrappers could switch off clippy's rule against
//!    `expose_secret`.
//! 7. **Test support.** In the verifier's resolved graph, every edge into a crate that exists for
//!    tests (`tv-testkit`, whose vector reader panics on malformed input) is a dev edge
//!    (robustness, RFC-0019).
//! 8. **Compiled files.** The compiler's own record. `cargo check` builds every target twice: in
//!    the dev profile with every feature, and in the release profile with every feature except
//!    the test-only ones. Every file in the rustc dep-info of each path package's units must be a
//!    file inside that package's directory and outside the build directory. The dep-info is read
//!    strictly, so a malformed record fails. This catches what no token scan can, such as a
//!    dependency's macro that expands to `include!`. It runs once checks 1, 2, 3 and 6 pass, so
//!    the gate never compiles code they reject.
//!
//! Both graphs are resolved with every feature enabled. An optional dependency counts as present,
//! since any build may enable it; a default resolve would leave it out of `cargo metadata`.

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

/// Verifier crates that exist only to support tests (check 7).
const TEST_SUPPORT: [&str; 1] = ["tv-testkit"];

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
        cmd.args([
            "metadata",
            "--format-version",
            "1",
            "--locked",
            "--all-features",
        ]);
        let output = cmd.output().map_err(|e| format!("cannot run cargo: {e}"))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("cargo metadata failed: {}", stderr.trim()));
        }
        serde_json::from_slice(&output.stdout).map_err(|e| format!("cargo metadata output: {e}"))
    }
}

/// The result of one check.
pub(crate) struct Report {
    /// The property the check asserts.
    pub(crate) name: &'static str,
    /// Each way the repository breaks it; empty when it holds.
    pub(crate) violations: Vec<String>,
    /// Why the check did not run. A check is skipped only when another check fails, so a
    /// skipped check never lets the gate pass.
    pub(crate) skipped: Option<&'static str>,
}

impl Report {
    fn new(name: &'static str, violations: Vec<String>) -> Self {
        Self {
            name,
            violations,
            skipped: None,
        }
    }

    fn skipped(name: &'static str, reason: &'static str) -> Self {
        Self {
            name,
            violations: Vec::new(),
            skipped: Some(reason),
        }
    }
}

const NO_GRAPH: &str = "no dependency graph for verifier/";

/// Runs every check on the repository at `repo`.
pub(crate) fn checks(repo: &Path, cargo: &Cargo) -> io::Result<Vec<Report>> {
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

    const DEP1: &str = "DEP-1: no prover crate in the verifier's dependency graph";
    const DEP3: &str = "DEP-3: third-party packages are allowlisted";
    const SOURCES: &str = "verifier sources reference no file outside their crate";
    const TEST_SUPPORT: &str = "verifier test support is used by tests only";
    const COMPILED: &str = "every file a verifier crate compiles lies inside that crate";

    let allowlist = load_allowlist(&verifier.join("allowlist.toml"))?;
    let mut sources = source_violations(&verifier)?;
    let config = config_violations(&repo, &verifier)?;
    let (dep1, dep3, test_support) = match &graph {
        Ok(graph) => {
            sources.extend(build_time_code_violations(graph));
            let mut dep3 = allowlist.as_ref().err().cloned().unwrap_or_default();
            if let Ok(entries) = &allowlist {
                dep3.extend(dep3_violations(graph, entries));
            }
            (
                Report::new(DEP1, dep1_violations(graph, &verifier)),
                Report::new(DEP3, dep3),
                Report::new(TEST_SUPPORT, test_support_violations(graph)),
            )
        }
        Err(e) => (
            Report::new(DEP1, vec![format!("verifier/: {e}")]),
            Report::skipped(DEP3, NO_GRAPH),
            Report::skipped(TEST_SUPPORT, NO_GRAPH),
        ),
    };
    // Compiling runs the build scripts of dependencies, so check 8 waits until the checks that
    // vet what gets compiled pass.
    let compiled = match &graph {
        Err(_) => Report::skipped(COMPILED, NO_GRAPH),
        Ok(_)
            if !(dep1.violations.is_empty()
                && dep3.violations.is_empty()
                && sources.is_empty()
                && config.is_empty()) =>
        {
            Report::skipped(
                COMPILED,
                "it runs once DEP-1, DEP-3, the source scan and the configuration check pass",
            )
        }
        Ok(graph) => Report::new(
            COMPILED,
            compiled_file_violations(cargo, &verifier, graph, toolchain.as_deref()),
        ),
    };
    let dep2 = match &prover {
        Ok(prover) => dep2_violations(prover, &verifier),
        Err(e) => vec![format!("repository root: {e}")],
    };
    Ok(vec![
        dep1,
        dep3,
        Report::new(SOURCES, sources),
        Report::new(
            "toolchain pins are equal",
            toolchain_violations(&repo, &verifier)?,
        ),
        Report::new(
            "DEP-2: prover crates use verifier crates only as dev-dependencies",
            dep2,
        ),
        Report::new(
            "cargo configuration cannot redirect verifier dependencies",
            config,
        ),
        test_support,
        compiled,
    ])
}

/// Runs every check, reports on stdout and returns whether the repository is isolated.
pub(crate) fn run(repo: &Path) -> io::Result<bool> {
    let mut clean = true;
    for report in checks(repo, &Cargo::from_env())? {
        let status = match (&report.skipped, report.violations.is_empty()) {
            (_, false) => "FAIL",
            (Some(_), true) => "skip",
            (None, true) => "ok",
        };
        clean &= status == "ok";
        println!("{status:<6}{}", report.name);
        if let Some(reason) = report.skipped {
            println!("      not run: {reason}");
        }
        for violation in &report.violations {
            println!("      {violation}");
        }
    }
    Ok(clean)
}

// ----- Check 1: DEP-1 -----------------------------------------------------------------------

fn packages(graph: &Json) -> impl Iterator<Item = &Json> {
    graph["packages"].as_array().into_iter().flatten()
}

fn workspace_members(graph: &Json) -> Vec<&str> {
    graph["workspace_members"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Json::as_str)
        .collect()
}

fn str_field<'a>(value: &'a Json, key: &str) -> &'a str {
    value[key].as_str().unwrap_or_default()
}

/// Where a path really is: the canonical path when it exists, its lexical normalization otherwise.
fn resolve(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| normalize(path))
}

/// DEP-1. A path package is judged by where its code lives, not only by where its manifest
/// sits: target keys such as `[lib] path` let a manifest compile sources from anywhere, and
/// `targets[].src_path` is cargo's own record of what it compiles. Each crate root must be a
/// `.rs` file of its own package, which the source scan reads; the scan skips the build
/// directory, so no package may live there.
fn dep1_violations(graph: &Json, verifier: &Path) -> Vec<String> {
    let build_dir = verifier.join("target");
    let members = workspace_members(graph);
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
        if dir.starts_with(&build_dir) {
            violations.push(format!(
                "{name} lives in the build directory verifier/target, which the source scan skips"
            ));
            continue;
        }
        if !members.contains(&str_field(package, "id")) {
            violations.push(format!(
                "{name} is a path package but not a member of the verifier workspace, which \
                 `--all-features` and check 8 cover"
            ));
            continue;
        }
        for target in package["targets"].as_array().into_iter().flatten() {
            let target_name = str_field(target, "name");
            let src = resolve(Path::new(str_field(target, "src_path")));
            if !src.starts_with(&dir) {
                violations.push(format!(
                    "{name} target `{target_name}` compiles a source outside its package: {}",
                    src.display()
                ));
            } else if src.extension().is_none_or(|ext| ext != "rs") {
                violations.push(format!(
                    "{name} target `{target_name}` has a crate root that is not a .rs file, which \
                     the source scan skips: {}",
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
            if let Some(allowed) = entry(dep_name).and_then(|e| e.direct_dependents.as_ref())
                && !allowed.iter().any(|a| a == member_name)
            {
                violations.push(format!(
                    "{member_name} depends on {dep_name}, which only {} may use",
                    allowed.join(", ")
                ));
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

/// What a file reference in Rust source does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// `include!` or a `path` attribute: imports code.
    Code,
    /// `include_str!` or `include_bytes!`: reads a data file.
    Data,
    /// An include macro's name used other than in a direct call (`use … as`, an argument to
    /// another macro), which can carry it out of sight of this scan.
    Indirect,
    /// In a macro body, a macro call or attribute whose name is a metavariable: the caller
    /// chooses it and can choose an include macro or `path`.
    Metavariable,
}

/// A file reference found in Rust source: what it is, where, its literal value (`None` when
/// the argument is not a plain string literal), and whether it sits in a `macro_rules!` body.
struct Reference {
    what: String,
    line: usize,
    value: Option<String>,
    kind: Kind,
    in_macro: bool,
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

fn is_punct(token: Option<&TokenTree>, c: char) -> bool {
    matches!(token, Some(TokenTree::Punct(p)) if p.as_char() == c)
}

/// Whether `tokens[i]` is the body of `macro_rules! name { ... }`.
fn is_macro_body(tokens: &[TokenTree], i: usize) -> bool {
    i >= 3
        && matches!(&tokens[i - 3], TokenTree::Ident(m) if ident_name(m) == "macro_rules")
        && is_punct(tokens.get(i - 2), '!')
        && matches!(&tokens[i - 1], TokenTree::Ident(_))
}

/// Finds file references in a token stream, recursively: include macros and any other use of
/// their names, `path` attributes, and, in macro bodies, macro calls and attributes named by a
/// metavariable.
fn scan(stream: TokenStream, in_macro: bool, out: &mut Vec<Reference>) {
    let tokens: Vec<TokenTree> = stream.into_iter().collect();
    for (i, token) in tokens.iter().enumerate() {
        let mut found = |what: String, value: Option<String>, kind: Kind| {
            out.push(Reference {
                what,
                line: line_of(token),
                value,
                kind,
                in_macro,
            });
        };
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
                        let kind = if name == "include" {
                            Kind::Code
                        } else {
                            Kind::Data
                        };
                        found(format!("{name}!"), value, kind);
                    }
                    _ => found(format!("`{name}`"), None, Kind::Indirect),
                }
            }
            TokenTree::Punct(dollar) if in_macro && dollar.as_char() == '$' => {
                if let Some(TokenTree::Ident(var)) = tokens.get(i + 1)
                    && is_punct(tokens.get(i + 2), '!')
                {
                    found(format!("`${var}!`"), None, Kind::Metavariable);
                }
            }
            TokenTree::Punct(hash) if hash.as_char() == '#' => {
                let j = if is_punct(tokens.get(i + 1), '!') {
                    i + 2
                } else {
                    i + 1
                };
                if let Some(TokenTree::Group(group)) = tokens.get(j)
                    && group.delimiter() == Delimiter::Bracket
                {
                    scan_attribute(group.stream(), in_macro, out);
                }
            }
            TokenTree::Group(group) => {
                scan(group.stream(), in_macro || is_macro_body(&tokens, i), out);
            }
            _ => {}
        }
    }
}

/// Records the attributes in an attribute body that read files, looking inside
/// `cfg_attr(predicate, ...)`. In a macro body it also records an attribute whose name holds a
/// metavariable, since the caller can make that name `path`. A metavariable in a value, as in
/// `#[doc = $doc]`, is harmless: what the caller passes there is scanned where it is written.
fn scan_attribute(stream: TokenStream, in_macro: bool, out: &mut Vec<Reference>) {
    let tokens: Vec<TokenTree> = stream.into_iter().collect();
    let Some(first) = tokens.first() else {
        return;
    };
    // The name is the path before `=` or the arguments.
    let name = tokens
        .iter()
        .take_while(|t| !matches!(t, TokenTree::Group(_)) && !is_punct(Some(t), '='));
    if in_macro && name.clone().any(|t| is_punct(Some(t), '$')) {
        out.push(Reference {
            what: "an attribute named by a metavariable".to_owned(),
            line: line_of(first),
            value: None,
            kind: Kind::Metavariable,
            in_macro,
        });
        return;
    }
    match &tokens[..] {
        [TokenTree::Ident(attr), ..] if ident_name(attr) == "debugger_visualizer" => {
            out.push(Reference {
                what: "#[debugger_visualizer]".to_owned(),
                line: line_of(first),
                value: None,
                kind: Kind::Code,
                in_macro,
            });
        }
        [TokenTree::Ident(path), rest @ ..] if ident_name(path) == "path" => {
            let value = match rest {
                [TokenTree::Punct(eq), lit] if eq.as_char() == '=' => plain_string(lit),
                _ => None,
            };
            out.push(Reference {
                what: "#[path]".to_owned(),
                line: line_of(&tokens[0]),
                value,
                kind: Kind::Code,
                in_macro,
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
                scan_attribute(attr.iter().cloned().collect(), in_macro, out);
            }
        }
        _ => {}
    }
}

/// Where a token stream pulls in code from another file (`include!`, a `path` or
/// `debugger_visualizer` attribute), or names an include macro other than in a direct call,
/// which can hide one. Direct `include_str!` and `include_bytes!` calls read data and are left
/// out. The secret-access check (05 §3 OBS-1) holds prover sources to this, since it reads only
/// the `.rs` files that `mod` declarations reach.
pub(crate) fn code_inclusions(stream: TokenStream) -> Vec<(usize, String)> {
    let mut references = Vec::new();
    scan(stream, false, &mut references);
    let mut found: Vec<(usize, String)> = references
        .into_iter()
        .filter_map(|r| match r.kind {
            Kind::Code => Some((
                r.line,
                format!("{} pulls in code from another file", r.what),
            )),
            Kind::Indirect => Some((
                r.line,
                format!(
                    "{} used other than in a direct call can pull in code",
                    r.what
                ),
            )),
            Kind::Data | Kind::Metavariable => None,
        })
        .collect();
    // A macro's matcher and transcriber can repeat one finding on one line.
    found.dedup();
    found
}

/// The directory of the package that `file` belongs to: its nearest ancestor holding a
/// `Cargo.toml`, or `verifier/` itself.
fn package_dir<'a>(file: &'a Path, verifier: &'a Path) -> &'a Path {
    file.ancestors()
        .skip(1)
        .take_while(|dir| dir.starts_with(verifier))
        .find(|dir| dir.join("Cargo.toml").is_file())
        .unwrap_or(verifier)
}

/// What is wrong with one reference found in `file`, if anything.
fn reference_violation(r: &Reference, file: &Path, verifier: &Path) -> Option<String> {
    let value = match (r.kind, &r.value) {
        (Kind::Code, _) => {
            return Some(format!(
                "{} is not allowed: code enters a verifier crate only through `mod` \
                 declarations at their default paths and through dependencies",
                r.what
            ));
        }
        (Kind::Indirect, _) => {
            return Some(format!(
                "{} used other than in a direct call can hide a file reference",
                r.what
            ));
        }
        (Kind::Metavariable, _) => {
            return Some(format!(
                "a macro body uses {}, which its caller can turn into a file reference",
                r.what
            ));
        }
        (Kind::Data, _) if r.in_macro => {
            return Some(format!(
                "{} in a macro body resolves at each call site",
                r.what
            ));
        }
        (Kind::Data, None) => {
            return Some(format!("{} needs a plain string literal argument", r.what));
        }
        (Kind::Data, Some(value)) => value,
    };
    if Path::new(value).is_absolute() {
        return Some(format!("{} uses an absolute path `{value}`", r.what));
    }
    let target = normalize(&file.parent().unwrap_or(verifier).join(value));
    let package = package_dir(file, verifier);
    if !target.starts_with(verifier) {
        Some(format!("{} `{value}` resolves outside verifier/", r.what))
    } else if package_dir(&target, verifier) != package {
        Some(format!(
            "{} `{value}` resolves outside its package (verifier/{})",
            r.what,
            package.strip_prefix(verifier).unwrap_or(package).display()
        ))
    } else {
        None
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
        // A newline in a path would split rustc's dep-info records, which check 8 parses.
        match path.file_name().and_then(|name| name.to_str()) {
            None => violations.push(format!("verifier/{rel:?} has a name that is not UTF-8")),
            Some(name) if name.chars().any(char::is_control) => {
                violations.push(format!(
                    "verifier/{rel:?} has a control character in its name"
                ));
            }
            Some(_) => {}
        }
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
        scan(stream, false, &mut references);
        for r in &references {
            if let Some(problem) = reference_violation(r, &path, verifier) {
                violations.push(format!("verifier/{rel}:{}: {problem}", r.line));
            }
        }
    }
    // A macro's matcher and transcriber can repeat one finding on one line.
    violations.dedup();
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
        if name == "Cargo.toml"
            && let Ok(manifest) = fs::read_to_string(path)?.parse::<Table>()
        {
            for key in ["patch", "replace"] {
                if manifest.contains_key(key) {
                    violations.push(format!("{rel}: [{key}] can substitute any dependency"));
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

// ----- Check 7: test support ----------------------------------------------------------------

/// Every edge into a [`TEST_SUPPORT`] crate in the verifier's resolved graph is a dev edge.
fn test_support_violations(graph: &Json) -> Vec<String> {
    let name = |id: &str| {
        packages(graph)
            .find(|p| str_field(p, "id") == id)
            .map_or("?", |p| str_field(p, "name"))
    };
    let mut violations = Vec::new();
    for node in graph["resolve"]["nodes"].as_array().into_iter().flatten() {
        for dep in node["deps"].as_array().into_iter().flatten() {
            let to = name(str_field(dep, "pkg"));
            if !TEST_SUPPORT.contains(&to) {
                continue;
            }
            for kind in dep["dep_kinds"].as_array().into_iter().flatten() {
                let kind = kind["kind"].as_str().unwrap_or("normal");
                if kind != "dev" {
                    violations.push(format!(
                        "{} has a {kind} dependency on test-support crate {to}; only dev-dependencies are allowed",
                        name(str_field(node, "id"))
                    ));
                }
            }
        }
    }
    violations.sort();
    violations.dedup();
    violations
}

// ----- Check 8: compiled files --------------------------------------------------------------

/// Check 8. The token scan of check 3 cannot follow macro expansion: a dependency's macro can
/// expand to `include!` at a call site that shows no include. So this check asks the compiler.
/// It runs `cargo check` on every target and reads the dep-info rustc writes for each unit of a
/// verifier crate, which lists every file that compilation read: the crate root, its modules and
/// every included file. Each must be a file inside that crate's directory and outside the build
/// directory.
///
/// It checks two configurations: the dev profile with every feature, and the release profile,
/// which is what ships and compiles `cfg(not(debug_assertions))` code, with every feature except
/// the test-only ones, whose guards fail release builds (RFC-0001 'Cargo features'). Code under
/// other `cfg`s, such as other targets, is left to the token scan, which reads every `cfg`.
fn compiled_file_violations(
    cargo: &Cargo,
    verifier: &Path,
    graph: &Json,
    toolchain: Option<&str>,
) -> Vec<String> {
    let members = workspace_members(graph);
    let release_features: Vec<String> = packages(graph)
        .filter(|p| members.contains(&str_field(p, "id")))
        .flat_map(|p| {
            let name = str_field(p, "name");
            p["features"]
                .as_object()
                .into_iter()
                .flat_map(|features| features.keys())
                .filter(|feature| !crate::test_features::TEST_ONLY.contains(&feature.as_str()))
                .map(move |feature| format!("{name}/{feature}"))
        })
        .collect();
    let mut release = vec!["--release".to_owned()];
    if !release_features.is_empty() {
        release.extend(["--features".to_owned(), release_features.join(",")]);
    }
    let mut violations = Vec::new();
    for args in [vec!["--all-features".to_owned()], release] {
        violations.extend(compiled_files_of_one_build(
            cargo, verifier, graph, toolchain, &args,
        ));
    }
    violations.sort();
    violations.dedup();
    violations
}

fn compiled_files_of_one_build(
    cargo: &Cargo,
    verifier: &Path,
    graph: &Json,
    toolchain: Option<&str>,
    args: &[String],
) -> Vec<String> {
    let build_dir = verifier.join("target");
    let mut cmd = cargo.command(verifier, toolchain);
    cmd.args([
        "check",
        "--workspace",
        "--all-targets",
        "--locked",
        "--message-format=json",
    ])
    .args(args)
    .arg("--target-dir")
    .arg(build_dir.join("check-isolation"));
    let build = format!("cargo check {}", args.first().map_or("", String::as_str));
    let output = match cmd.output() {
        Ok(output) => output,
        Err(e) => return vec![format!("cannot run cargo: {e}")],
    };
    let messages: Vec<Json> = output
        .stdout
        .split(|&b| b == b'\n')
        .filter_map(|line| serde_json::from_slice(line).ok())
        .collect();
    if !output.status.success() {
        let error = messages
            .iter()
            .find(|m| m["reason"] == "compiler-message" && m["message"]["level"] == "error")
            .and_then(|m| m["message"]["rendered"].as_str())
            .map_or_else(
                || String::from_utf8_lossy(&output.stderr).into_owned(),
                str::to_owned,
            );
        return vec![format!("{build} failed: {}", error.trim())];
    }

    let repo = verifier.parent().unwrap_or(verifier);
    let mut violations = Vec::new();
    for message in messages
        .iter()
        .filter(|m| m["reason"] == "compiler-artifact")
    {
        let id = str_field(message, "package_id");
        // Every path package, not only members: DEP-1 rejects non-members, and this check does
        // not rely on it.
        let Some(package) =
            packages(graph).find(|p| p["source"].is_null() && str_field(p, "id") == id)
        else {
            continue;
        };
        let name = str_field(package, "name");
        let manifest = Path::new(str_field(package, "manifest_path"));
        let dir = resolve(manifest.parent().unwrap_or(manifest));
        let target = str_field(&message["target"], "name");
        let unit = format!("{name} target `{target}`");
        for artifact in message["filenames"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Json::as_str)
        {
            let artifact = Path::new(artifact);
            let dep_info = dep_info_path(artifact);
            let files = fs::read_to_string(&dep_info)
                .map_err(|e| format!("cannot read {}: {e}", dep_info.display()))
                .and_then(|text| dep_info_files(&text, &[artifact, &dep_info]));
            let files = match files {
                Ok(files) => files,
                Err(e) => {
                    violations.push(format!("{build}: {unit}: dep-info: {e}"));
                    continue;
                }
            };
            for file in files {
                // Cargo runs rustc for path packages from the workspace root.
                let path = normalize(&verifier.join(&file));
                let shown = path.strip_prefix(repo).unwrap_or(&path).display();
                if !path.starts_with(&dir) || path.starts_with(&build_dir) {
                    violations.push(format!(
                        "compiling {unit} reads {shown}, outside verifier/{}",
                        dir.strip_prefix(verifier).unwrap_or(&dir).display()
                    ));
                } else if !path.is_file() {
                    violations.push(format!(
                        "{build}: {unit}: dep-info lists {shown}, which is not a file"
                    ));
                }
            }
        }
    }
    violations
}

/// The dep-info file rustc writes next to an artifact: `lib<name>-<hash>.rmeta` has
/// `<name>-<hash>.d`.
fn dep_info_path(artifact: &Path) -> PathBuf {
    let stem = artifact
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or_default();
    artifact.with_file_name(format!("{}.d", stem.strip_prefix("lib").unwrap_or(stem)))
}

/// The words of a dep-info line: separated by spaces, with a backslash escaping the next
/// character.
fn dep_info_words(line: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut word = String::new();
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => word.extend(chars.next()),
            ' ' if !word.is_empty() => words.push(std::mem::take(&mut word)),
            ' ' => {}
            c => word.push(c),
        }
    }
    if !word.is_empty() {
        words.push(word);
    }
    words
}

/// The files a rustc dep-info file lists, read strictly. rustc writes one rule
/// `output: file file …` per output, then an empty rule `file:` per file, and `#` comments for
/// environment and checksum records. Any other line, such as the remainder of a path that
/// contained a newline, makes the record untrustworthy, so it is an error.
fn dep_info_files(text: &str, outputs: &[&Path]) -> Result<Vec<String>, String> {
    let mut files: Vec<String> = Vec::new();
    for line in text.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let words = dep_info_words(line);
        let Some((target, rest)) = words.split_first() else {
            continue;
        };
        let Some(target) = target.strip_suffix(':') else {
            return Err(format!("unexpected line {line:?}"));
        };
        if outputs.iter().any(|output| Path::new(target) == *output) {
            files.extend(rest.iter().cloned());
        } else if !(rest.is_empty() && files.iter().any(|file| file == target)) {
            return Err(format!("unexpected line {line:?}"));
        }
    }
    if files.is_empty() {
        return Err("no rule for the artifact".to_owned());
    }
    Ok(files)
}

#[cfg(test)]
mod tests;
