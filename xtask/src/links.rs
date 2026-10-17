// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `cargo xtask check-links`: every relative link in the repository's Markdown resolves to an
//! existing file or directory, and every `#anchor` into a Markdown file names one of its
//! headings, slugged as GitHub renders them. External links (`http`, `https`, `mailto`) are not
//! fetched: the result must not depend on the network.
//!
//! The same pass checks tables: GitHub's table parser splits a row into cells at every unescaped
//! `|`, inside code spans too, so a row such as ``| `max|a|` |`` renders with its cells shifted.
//! A pipe in a code span of a table row (a line that starts with `|`, outside fenced blocks) is
//! written `\|`, which GitHub renders as `|`.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// GitHub's heading slug: lowercase; keep letters, digits, `-`, `_` and spaces; spaces become
/// hyphens. Repeated headings get `-1`, `-2`, ... in document order.
fn slug(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | ' '))
        .map(|c| if c == ' ' { '-' } else { c })
        .collect()
}

/// The rendered text of a heading: link text without its target, code spans and emphasis
/// without their markers.
fn rendered(heading: &str) -> String {
    let chars: Vec<char> = heading.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '`' | '*' | '[' => {}
            ']' if chars.get(i + 1) == Some(&'(') => i = closing_paren(&chars, i + 1),
            ']' => {}
            c => out.push(c),
        }
        i += 1;
    }
    out.trim().to_owned()
}

/// The index of the `)` matching the `(` at `open`, or the last index if unbalanced.
fn closing_paren(chars: &[char], open: usize) -> usize {
    let mut depth = 0usize;
    for (j, &c) in chars.iter().enumerate().skip(open) {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return j;
                }
            }
            _ => {}
        }
    }
    chars.len().saturating_sub(1)
}

/// Lines outside fenced code blocks, with their 1-based numbers.
fn prose_lines(text: &str) -> impl Iterator<Item = (usize, &str)> {
    let mut fence: Option<&str> = None;
    text.lines().enumerate().filter_map(move |(n, line)| {
        let trimmed = line.trim_start();
        if let Some(marker) = fence {
            if trimmed.starts_with(marker) {
                fence = None;
            }
            return None;
        }
        for marker in ["```", "~~~"] {
            if trimmed.starts_with(marker) {
                fence = Some(marker);
                return None;
            }
        }
        Some((n + 1, line))
    })
}

/// The anchors GitHub generates for the headings of a Markdown document.
fn anchors(text: &str) -> HashSet<String> {
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut out = HashSet::new();
    for (_, line) in prose_lines(text) {
        let trimmed = line.trim_start();
        let level = trimmed.chars().take_while(|&c| c == '#').count();
        if !(1..=6).contains(&level) || !trimmed[level..].starts_with(' ') {
            continue;
        }
        let base = slug(&rendered(
            trimmed[level..].trim().trim_end_matches('#').trim(),
        ));
        let count = seen.entry(base.clone()).or_insert(0);
        out.insert(if *count == 0 {
            base.clone()
        } else {
            format!("{base}-{count}")
        });
        *count += 1;
    }
    out
}

/// Inline link targets outside code spans and fences, as `(line, target)`.
fn link_targets(text: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    for (n, line) in prose_lines(text) {
        let chars: Vec<char> = line.chars().collect();
        let mut in_code = false;
        let mut i = 0;
        while i < chars.len() {
            if chars[i] == '`' {
                in_code = !in_code;
            } else if !in_code && chars[i] == ']' && chars.get(i + 1) == Some(&'(') {
                let close = closing_paren(&chars, i + 1);
                let inner: String = chars[i + 2..close].iter().collect();
                // `[text](target "title")` and `[text](<target>)`.
                let target = inner.split_whitespace().next().unwrap_or_default();
                out.push((
                    n,
                    target
                        .trim_start_matches('<')
                        .trim_end_matches('>')
                        .to_owned(),
                ));
                i = close;
            }
            i += 1;
        }
    }
    out
}

/// The code spans of `line`, as CommonMark delimits them: a run of `k` backticks opens one, and
/// the next run of exactly `k` closes it; an unclosed run is literal text.
fn code_spans(line: &str) -> Vec<&str> {
    let bytes = line.as_bytes();
    let run = |from: usize| bytes[from..].iter().take_while(|&&b| b == b'`').count();
    let mut spans = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'`' {
            i += 1;
            continue;
        }
        let open = run(i);
        let start = i + open;
        let mut j = start;
        let mut close = None;
        while j < bytes.len() {
            if bytes[j] == b'`' {
                let length = run(j);
                if length == open {
                    close = Some(j);
                    break;
                }
                j += length;
            } else {
                j += 1;
            }
        }
        match close {
            Some(end) => {
                spans.push(&line[start..end]);
                i = end + open;
            }
            None => i = start,
        }
    }
    spans
}

/// The lines of the table rows whose code spans hold a `|` with no `\` before it.
fn split_cells(text: &str) -> Vec<usize> {
    prose_lines(text)
        .filter(|(_, line)| line.trim_start().starts_with('|'))
        .filter(|(_, line)| {
            code_spans(line).iter().any(|span| {
                span.char_indices()
                    .any(|(k, c)| c == '|' && !span[..k].ends_with('\\'))
            })
        })
        .map(|(n, _)| n)
        .collect()
}

/// Checks the Markdown files among `files` (relative to `repo`).
pub(crate) fn check(repo: &Path, files: &[PathBuf]) -> io::Result<Vec<String>> {
    let mut anchor_cache: HashMap<PathBuf, HashSet<String>> = HashMap::new();
    let mut violations = Vec::new();
    for rel in files
        .iter()
        .filter(|f| f.extension().is_some_and(|e| e == "md"))
    {
        let text = fs::read_to_string(repo.join(rel))?;
        for line in split_cells(&text) {
            violations.push(format!(
                "{}:{line}: a `|` inside a code span splits this table row's cells; write it `\\|`",
                rel.display()
            ));
        }
        for (line, target) in link_targets(&text) {
            if target.is_empty()
                || ["http://", "https://", "mailto:"]
                    .iter()
                    .any(|s| target.starts_with(s))
            {
                continue;
            }
            let at = format!("{}:{line}", rel.display());
            let (path, anchor) = match target.split_once('#') {
                Some((path, anchor)) => (path, Some(anchor)),
                None => (target.as_str(), None),
            };
            let dest = if path.is_empty() {
                repo.join(rel)
            } else {
                repo.join(rel.parent().unwrap_or(Path::new(""))).join(path)
            };
            if !dest.exists() {
                violations.push(format!("{at}: `{target}`: {} does not exist", path));
                continue;
            }
            let Some(anchor) = anchor else { continue };
            if dest.extension().is_none_or(|e| e != "md") {
                continue;
            }
            let known = match anchor_cache.get(&dest) {
                Some(known) => known,
                None => {
                    let parsed = anchors(&fs::read_to_string(&dest)?);
                    anchor_cache.entry(dest.clone()).or_insert(parsed)
                }
            };
            if !known.contains(anchor) {
                violations.push(format!(
                    "{at}: `{target}`: no heading with anchor #{anchor}"
                ));
            }
        }
    }
    Ok(violations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::Fixture;

    #[test]
    fn headings_are_slugged_as_github_renders_them() {
        let doc = concat!(
            "# 07 \u{2014} Testing Strategy\n",
            "## 3. End-to-end targets (Llama-3.1-405B, one 8\u{d7}H100 node)\n",
            "## Intro\n",
            "## Intro\n",
            "### `Requant1` and [the table](x.md#y)\n",
            "## RFC-0017: Scoring and the Bound \u{db}\n",
            "```text\n## Not a heading\n```\n",
        );
        let found = anchors(doc);
        for expected in [
            "07--testing-strategy",
            "3-end-to-end-targets-llama-31-405b-one-8h100-node",
            "intro",
            "intro-1",
            "requant1-and-the-table",
            "rfc-0017-scoring-and-the-bound-\u{fb}",
        ] {
            assert!(found.contains(expected), "missing {expected}: {found:?}");
        }
        assert!(!found.contains("not-a-heading"));
    }

    #[test]
    fn broken_relative_links_and_anchors_are_reported() {
        let fixture = Fixture::new("links");
        fixture.write(
            "a.md",
            "# Title \u{2014} Sub\n## 1. Intro\n## 1. Intro\n```\n## Fenced\n```\n",
        );
        fixture.write("sub/c.md", "# C\n");
        fixture.write(
            "b.md",
            concat!(
                "## Local heading\n",
                "[ok](a.md#title--sub) [ok](a.md#1-intro) [ok](a.md#1-intro-1) [ok](#local-heading)\n",
                "[dir](sub/) [up](sub/../a.md) [title](a.md \"A title\") [angle](<a.md>)\n",
                "[web](https://example.com/missing.md#x) [mail](mailto:x@example.com)\n",
                "`[code](missing.md)` is code, and so is the fence below\n",
                "```\n[fenced](missing.md)\n```\n",
                "[bad anchor](a.md#fenced) [missing](missing.md) [local](#nope)\n",
            ),
        );
        let files: Vec<PathBuf> = ["a.md", "b.md", "sub/c.md"]
            .iter()
            .map(PathBuf::from)
            .collect();
        let violations = check(fixture.path(), &files).expect("fixture is readable");
        assert_eq!(violations.len(), 3, "{violations:#?}");
        assert!(violations[0].starts_with("b.md:9: `a.md#fenced`: no heading with anchor #fenced"));
        assert!(violations[1].starts_with("b.md:9: `missing.md`"));
        assert!(violations[2].starts_with("b.md:9: `#nope`: no heading with anchor #nope"));
    }

    #[test]
    fn pipes_inside_code_spans_of_table_rows_are_reported() {
        let fixture = Fixture::new("table-pipes");
        fixture.write(
            "t.md",
            concat!(
                "| Quantity | Range |\n",
                "|---|---|\n",
                "| Activations | `max|a| / 127` |\n",
                "| Scores | `\\|x\\| < 2^21` and `a || b` |\n",
                "| Escaped | `\\|x\\| < 2^21` |\n",
                "| Fences | ``a ` | b`` |\n",
                "| Unclosed | `a | b |\n",
                "Prose with `x | y` is not a table row.\n",
                "```\n| fenced | `a|b` |\n```\n",
            ),
        );
        let violations = check(fixture.path(), &[PathBuf::from("t.md")]).expect("readable");
        assert_eq!(violations.len(), 3, "{violations:#?}");
        for (violation, line) in violations.iter().zip([3, 4, 6]) {
            assert!(
                violation.starts_with(&format!("t.md:{line}: a `|` inside a code span")),
                "{violation}"
            );
        }
    }
}
