// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Telemetry carries public data only (05 §2, §3 OBS-1, §5). `tessera_codec::event!` and
//! `span!` take fields of public types and literal message templates, so the policy is that no
//! other path reaches `tracing`:
//!
//! - only `tessera-codec`, which owns the helpers, and `tessera-cli`, which installs the
//!   subscriber, depend on `tracing`, `tracing-core` or `tracing-subscriber`; tests may;
//! - no production source but `tessera-codec`'s `telemetry.rs` invokes a macro through a
//!   `tracing` path or names the helpers' private re-export, and `tessera-cli` imports no macro
//!   from `tracing`;
//! - every event's level is a literal, and its name is `<crate>.<event>` in snake case, `<crate>`
//!   the crate's directory without `tessera-`; every span's name is one of 05 §5's;
//! - `tessera-codec` enables `tracing`'s `release_max_level_debug`, so that `trace` events, per
//!   tile, vanish from release builds (05 §2).
//!
//! The sources are lexed, not searched: comments and strings are not code.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use proc_macro2::{Delimiter, TokenStream, TokenTree};

/// The crates that may depend on the telemetry crates outside their tests.
const TELEMETRY_DEPENDENTS: [&str; 2] = ["tessera-codec", "tessera-cli"];
/// The telemetry crates.
const TELEMETRY_CRATES: [&str; 3] = ["tracing", "tracing-core", "tracing-subscriber"];
/// The one file that invokes `tracing`'s macros.
const HELPERS: &str = "crates/tessera-codec/src/telemetry.rs";
/// `tracing`'s event and span macros.
const TRACING_MACROS: [&str; 14] = [
    "event",
    "span",
    "trace",
    "debug",
    "info",
    "warn",
    "error",
    "trace_span",
    "debug_span",
    "info_span",
    "warn_span",
    "error_span",
    "instrument",
    "enabled",
];
/// 05 §5's spans.
pub(crate) const SPANS: [&str; 10] = [
    "compile",
    "commit_weights",
    "prove",
    "witness_pass",
    "round1",
    "stage",
    "opening",
    "round3",
    "verify",
    "check",
];
const LEVELS: [&str; 5] = ["ERROR", "WARN", "INFO", "DEBUG", "TRACE"];

/// Checks the prover workspace's manifests and production sources among `files`.
pub(crate) fn check(repo: &Path, files: &[PathBuf]) -> io::Result<Vec<String>> {
    let mut violations = Vec::new();
    for rel in files {
        let Some(krate) = crate_of(rel) else {
            continue;
        };
        let dir = Path::new("crates").join(krate);
        if rel.file_name().is_some_and(|n| n == "Cargo.toml") && rel.parent() == Some(dir.as_path())
        {
            let text = fs::read_to_string(repo.join(rel))?;
            violations.extend(
                manifest(krate, &text)
                    .into_iter()
                    .map(|v| format!("{}: {v}", rel.display())),
            );
        }
        let in_src = rel.starts_with(dir.join("src"));
        if in_src && rel.extension().is_some_and(|e| e == "rs") {
            let text = fs::read_to_string(repo.join(rel))?;
            let short = krate
                .strip_prefix("tessera-")
                .unwrap_or(krate)
                .replace('-', "_");
            let helpers = rel == Path::new(HELPERS);
            violations.extend(
                source(&text, &short, helpers, krate == "tessera-cli")
                    .into_iter()
                    .map(|(line, v)| format!("{}:{line}: {v}", rel.display())),
            );
        }
    }
    Ok(violations)
}

/// The crate directory a path lies in, `crates/<name>/...`.
fn crate_of(rel: &Path) -> Option<&str> {
    let mut parts = rel.components();
    (parts.next()?.as_os_str() == "crates").then_some(())?;
    parts.next()?.as_os_str().to_str()
}

/// A manifest's telemetry dependencies outside its tests.
fn manifest(krate: &str, text: &str) -> Vec<String> {
    let Ok(table) = text.parse::<toml::Table>() else {
        return vec!["cannot be parsed".to_owned()];
    };
    let mut out = Vec::new();
    let mut sections: Vec<(&str, &toml::Table)> = Vec::new();
    for key in ["dependencies", "build-dependencies"] {
        if let Some(t) = table.get(key).and_then(toml::Value::as_table) {
            sections.push((key, t));
        }
    }
    if let Some(targets) = table.get("target").and_then(toml::Value::as_table) {
        for target in targets.values().filter_map(toml::Value::as_table) {
            for key in ["dependencies", "build-dependencies"] {
                if let Some(t) = target.get(key).and_then(toml::Value::as_table) {
                    sections.push((key, t));
                }
            }
        }
    }
    // 05 §2: `trace` events vanish from release builds, by `tracing`'s static level, which the
    // helpers' crate sets for every build that links it.
    if krate == "tessera-codec" {
        let features = table
            .get("dependencies")
            .and_then(|d| d.get("tracing"))
            .and_then(|t| t.get("features"))
            .and_then(toml::Value::as_array);
        if !features.is_some_and(|f| {
            f.iter()
                .any(|x| x.as_str() == Some("release_max_level_debug"))
        }) {
            out.push("`tracing` lacks the feature `release_max_level_debug` (05 §2)".to_owned());
        }
    }
    for (section, deps) in sections {
        for name in TELEMETRY_CRATES {
            let renamed = deps
                .values()
                .any(|d| d.get("package").and_then(toml::Value::as_str) == Some(name));
            if (deps.contains_key(name) || renamed) && !TELEMETRY_DEPENDENTS.contains(&krate) {
                out.push(format!(
                    "{section} names `{name}`: emit telemetry through tessera_codec::event! and \
                     span! (05 §3 OBS-1)"
                ));
            }
        }
    }
    out
}

/// A source's violations, each with its line. `short` is its crate's event prefix.
fn source(text: &str, short: &str, helpers: bool, cli: bool) -> Vec<(usize, String)> {
    match TokenStream::from_str(text) {
        Ok(tokens) => {
            let mut found = Vec::new();
            walk(
                &tokens.into_iter().collect::<Vec<_>>(),
                short,
                helpers,
                cli,
                &mut found,
            );
            found
        }
        Err(_) => vec![(0, "cannot be lexed".to_owned())],
    }
}

fn ident(token: Option<&TokenTree>) -> Option<String> {
    match token? {
        TokenTree::Ident(i) => Some(i.to_string()),
        _ => None,
    }
}
fn punct(token: Option<&TokenTree>, c: char) -> bool {
    matches!(token, Some(TokenTree::Punct(p)) if p.as_char() == c)
}
fn path_sep(tokens: &[TokenTree], at: usize) -> bool {
    punct(tokens.get(at), ':') && punct(tokens.get(at + 1), ':')
}

fn walk(
    tokens: &[TokenTree],
    short: &str,
    helpers: bool,
    cli: bool,
    found: &mut Vec<(usize, String)>,
) {
    for (i, token) in tokens.iter().enumerate() {
        let line = token.span().start().line;
        match token {
            TokenTree::Group(group) => {
                walk(
                    &group.stream().into_iter().collect::<Vec<_>>(),
                    short,
                    helpers,
                    cli,
                    found,
                );
            }
            TokenTree::Ident(name) if !helpers => {
                let name = name.to_string();
                if name == "__private" {
                    found.push((
                        line,
                        "names the telemetry helpers' private re-export".to_owned(),
                    ));
                }
                if (name == "tracing" || name == "tracing_core") && path_sep(tokens, i + 1) {
                    if let Some(m) = ident(tokens.get(i + 3))
                        && punct(tokens.get(i + 4), '!')
                    {
                        found.push((
                            line,
                            format!("invokes `{name}::{m}!`: use tessera_codec::event! or span!"),
                        ));
                    }
                    let in_use = i > 0 && ident(tokens.get(i - 1)).as_deref() == Some("use");
                    if cli && in_use && imports_macro(&tokens[i..]) {
                        found.push((
                            line,
                            "imports a macro from `tracing`: use tessera_codec::event! or span!"
                                .to_owned(),
                        ));
                    }
                }
                if (name == "event" || name == "span") && punct(tokens.get(i + 1), '!') {
                    let qualified = i >= 2 && path_sep(tokens, i - 2);
                    let through_codec = !qualified
                        || (i >= 3 && ident(tokens.get(i - 3)).as_deref() == Some("tessera_codec"));
                    if through_codec && let Some(TokenTree::Group(args)) = tokens.get(i + 2) {
                        found.extend(invocation(&name, args.stream(), short).map(|v| (line, v)));
                    }
                }
            }
            _ => {}
        }
    }
}

/// Whether a `use tracing::…;` tree imports one of `tracing`'s macros.
fn imports_macro(tokens: &[TokenTree]) -> bool {
    // An identifier followed by `::` is a module on the way, such as `tracing::span::Id`.
    fn tree(tokens: &[TokenTree]) -> bool {
        tokens.iter().enumerate().any(|(i, t)| match t {
            TokenTree::Ident(name) => {
                TRACING_MACROS.contains(&name.to_string().as_str()) && !path_sep(tokens, i + 1)
            }
            TokenTree::Group(g) if g.delimiter() == Delimiter::Brace => {
                tree(&g.stream().into_iter().collect::<Vec<_>>())
            }
            _ => false,
        })
    }
    let end = tokens
        .iter()
        .position(|t| punct(Some(t), ';'))
        .unwrap_or(tokens.len());
    // `use tracing::*` and `use tracing::{self, …}` reach the macros too.
    tree(&tokens[1..end]) || tokens[..end].iter().any(|t| punct(Some(t), '*'))
}

/// An `event!` or `span!` invocation's level and name, if they break the policy.
fn invocation(kind: &str, args: TokenStream, short: &str) -> Option<String> {
    let args: Vec<TokenTree> = args.into_iter().collect();
    let level = ident(args.first());
    if !level.as_deref().is_some_and(|l| LEVELS.contains(&l)) || !punct(args.get(1), ',') {
        return Some(format!(
            "{kind}! takes a literal level first: {}",
            LEVELS.join(", ")
        ));
    }
    let Some(TokenTree::Literal(name)) = args.get(2) else {
        return Some(format!(
            "{kind}! takes a string literal name after its level"
        ));
    };
    let name = name.to_string();
    let Some(name) = name.strip_prefix('"').and_then(|n| n.strip_suffix('"')) else {
        return Some(format!(
            "{kind}! takes a string literal name after its level"
        ));
    };
    match kind {
        "span" if !SPANS.contains(&name) => Some(format!(
            "span `{name}` is not one of 05 §5's: {}",
            SPANS.join(", ")
        )),
        "event" => {
            let (krate, event) = name.split_once('.').unwrap_or(("", ""));
            if krate != short || !snake(event) {
                Some(format!(
                    "event `{name}` is not `{short}.<event>` in snake case (05 §2)"
                ))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn snake(name: &str) -> bool {
    !name.is_empty()
        && name.split('_').all(|part| {
            !part.is_empty()
                && part
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        })
        && name.starts_with(|c: char| c.is_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(text: &str, short: &str, cli: bool) -> Vec<String> {
        source(text, short, false, cli)
            .into_iter()
            .map(|(_, v)| v)
            .collect()
    }

    #[test]
    fn telemetry_goes_through_the_helpers() {
        for code in [
            "fn f() { tracing::info!(\"x\"); }",
            "fn f() { ::tracing::event!(tracing::Level::INFO, x = 1); }",
            "fn f() { tracing_core::event!(x); }",
            "fn f() { tessera_codec::telemetry::__private::tracing::info!(\"x\"); }",
        ] {
            assert!(!lines(code, "pcs", false).is_empty(), "{code}");
        }
        for code in [
            "use tracing::info;",
            "use tracing::{field, info_span};",
            "use tracing::*;",
        ] {
            assert!(!lines(code, "cli", true).is_empty(), "{code}");
        }
        for code in [
            "use tracing::{Event, Subscriber};",
            "use tracing::field::{Field, Visit};",
            "use tracing::span::{Attributes, Id, Record};",
            "fn f() { let s = tracing::Span::current(); }",
            "// tracing::info!(\"in a comment\")",
            "fn f() { let s = \"tracing::info!\"; }",
        ] {
            assert!(lines(code, "cli", true).is_empty(), "{code}");
        }
        // The helpers' own file is the one exception.
        assert!(source("fn f() { tracing::event!(x); }", "codec", true, false).is_empty());
    }

    #[test]
    fn event_and_span_names_follow_05() {
        for code in [
            "fn f() { event!(INFO, \"pcs.row_block_committed\"; \"m\"); }",
            "fn f() { tessera_codec::event!(DEBUG, \"pcs.weights_committed\", n = 1u8; \"m\"); }",
            "fn f() { let _s = span!(DEBUG, \"stage\", id = 1u16); }",
            "fn f() { tessera_codec::span!(INFO, \"prove\", batch = b); }",
            "fn f() { other::event!(anything); }",
        ] {
            assert!(lines(code, "pcs", false).is_empty(), "{code}");
        }
        for code in [
            "fn f() { event!(INFO, \"prover.round1\"; \"m\"); }",
            "fn f() { event!(INFO, \"pcs.RowBlock\"; \"m\"); }",
            "fn f() { event!(INFO, \"pcs.\"; \"m\"); }",
            "fn f() { event!(INFO, \"row_block\"; \"m\"); }",
            "fn f() { event!(level, \"pcs.x\"; \"m\"); }",
            "fn f() { event!(INFO, NAME; \"m\"); }",
            "fn f() { span!(DEBUG, \"tile\"); }",
            "fn f() { tessera_codec::span!(Level::DEBUG, \"stage\"); }",
        ] {
            assert!(!lines(code, "pcs", false).is_empty(), "{code}");
        }
    }

    #[test]
    fn manifests_keep_tracing_in_the_codec_and_the_cli() {
        let deps = "[dependencies]\ntracing = \"0.1\"\n";
        let codec = "[dependencies]\ntracing = { version = \"0.1\", \
                     features = [\"std\", \"release_max_level_debug\"] }\n";
        assert!(manifest("tessera-codec", codec).is_empty());
        assert!(
            !manifest("tessera-codec", deps).is_empty(),
            "trace events must leave release builds"
        );
        assert!(
            manifest(
                "tessera-cli",
                "[dependencies]\ntracing-subscriber = \"0.3\"\n"
            )
            .is_empty()
        );
        assert!(!manifest("tessera-pcs", deps).is_empty());
        assert!(
            !manifest(
                "tessera-pcs",
                "[dependencies]\nlog = { package = \"tracing\", version = \"0.1\" }\n"
            )
            .is_empty()
        );
        assert!(
            !manifest(
                "tessera-pcs",
                "[target.'cfg(unix)'.dependencies]\ntracing-core = \"0.1\"\n"
            )
            .is_empty()
        );
        assert!(manifest("tessera-pcs", "[dev-dependencies]\ntracing = \"0.1\"\n").is_empty());
    }

    #[test]
    fn the_repository_follows_the_policy() {
        let repo = crate::repo::root();
        let files = crate::repo::files(&repo).expect("repository is readable");
        assert_eq!(
            check(&repo, &files).expect("sources are readable"),
            Vec::<String>::new()
        );
    }
}
