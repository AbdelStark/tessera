// SPDX-License-Identifier: Apache-2.0 OR MIT

//! No floating point in the table generators (RFC-0008 'Generators and digests') or the soundness
//! calculator (RFC-0014 'Soundness calculator'): a table's values are the exact integers its
//! generator defines, and the calculator's bits and decisions are integers that both calculators
//! must agree on bit for bit, so their sources name no `f32` or `f64` and write no
//! floating-point literal. Comments and strings are not code: the sources are lexed, not
//! searched.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use proc_macro2::{TokenStream, TokenTree};

/// The directories and files that compute with integers only: the table generators, prover and
/// verifier side, and both soundness calculators.
pub(crate) const INTEGER_ONLY: [&str; 4] = [
    "crates/tessera-tables/src",
    "verifier/crates/tv-tables/src",
    "crates/tessera-protocol/src/soundness",
    "verifier/crates/tv-core/src/soundness.rs",
];

/// The integer suffixes, which a float test must not mistake for exponents (`usize` holds an
/// `e`).
const INTEGER_SUFFIXES: [&str; 12] = [
    "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128", "isize",
];

/// Checks the Rust files of `files` (relative to `repo`) inside an integer-only directory.
pub(crate) fn check(repo: &Path, files: &[PathBuf]) -> io::Result<Vec<String>> {
    let mut violations = Vec::new();
    for rel in files {
        let integer_only = INTEGER_ONLY.iter().any(|dir| rel.starts_with(dir));
        if !integer_only || rel.extension().is_none_or(|ext| ext != "rs") {
            continue;
        }
        let text = fs::read_to_string(repo.join(rel))?;
        violations.extend(
            floats(&text)
                .into_iter()
                .map(|(line, what)| format!("{}:{line}: {what}", rel.display())),
        );
    }
    Ok(violations)
}

/// Each floating-point type or literal of `source`, with its line.
fn floats(source: &str) -> Vec<(usize, String)> {
    match TokenStream::from_str(source) {
        Ok(tokens) => {
            let mut found = Vec::new();
            walk(tokens, &mut found);
            found
        }
        Err(_) => vec![(0, "cannot be lexed".to_string())],
    }
}

fn walk(tokens: TokenStream, found: &mut Vec<(usize, String)>) {
    for token in tokens {
        let line = token.span().start().line;
        match &token {
            TokenTree::Group(group) => walk(group.stream(), found),
            TokenTree::Ident(ident) => {
                let name = ident.to_string();
                if name == "f32" || name == "f64" {
                    found.push((line, format!("the type `{name}`")));
                }
            }
            TokenTree::Literal(literal) => {
                let text = literal.to_string();
                if is_float_literal(&text) {
                    found.push((line, format!("the floating-point literal `{text}`")));
                }
            }
            TokenTree::Punct(_) => {}
        }
    }
}

/// A numeric literal with a fraction, an exponent or a float suffix; never a string, a
/// character, or a hexadecimal, octal or binary integer.
fn is_float_literal(text: &str) -> bool {
    if !text.starts_with(|c: char| c.is_ascii_digit()) {
        return false;
    }
    if ["0x", "0o", "0b"]
        .iter()
        .any(|prefix| text.starts_with(prefix))
    {
        return false;
    }
    if text.ends_with("f32") || text.ends_with("f64") {
        return true;
    }
    let digits = INTEGER_SUFFIXES
        .iter()
        .find_map(|suffix| text.strip_suffix(suffix))
        .unwrap_or(text);
    digits.contains(['.', 'e', 'E'])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_are_found_in_code_and_nowhere_else() {
        for code in [
            "let x = 1.5;",
            "let x = 2e10;",
            "let x = 1E3;",
            "let x = 1f32;",
            "let x = 1.0_f64;",
            "fn f(x: f32) {}",
            "let y = x as f64;",
            "let v: Vec<f64> = vec![];",
            "mod m { const C: u8 = { let z = 0.5; 1 }; }",
        ] {
            assert_eq!(floats(code).len(), 1, "{code}");
        }
        for code in [
            "let x = 1u64 + 0xE5 + 0b1 + 0o7 + 1_000usize + 3isize;",
            "let y = t.0 + t.1;",
            "for i in 0..10 {}",
            "let s = \"1.5 f64\";",
            "// 1.5 and f64 in a comment\nlet z = 1;",
            "/* 2e10 */ let c = b'e';",
        ] {
            assert_eq!(floats(code), Vec::new(), "{code}");
        }
    }

    #[test]
    fn only_integer_only_directories_are_checked() {
        let fixture = crate::testing::Fixture::new("no-float");
        fixture.write("crates/tessera-tables/src/a.rs", "let x = 1.5;\n");
        fixture.write("verifier/crates/tv-tables/src/b.rs", "fn f(x: f64) {}\n");
        fixture.write("crates/tessera-bench/src/c.rs", "let x = 1.5;\n");
        fixture.write("crates/tessera-tables/src/d.txt", "1.5\n");
        fixture.write(
            "crates/tessera-protocol/src/soundness/e.rs",
            "let bits = (x as f64).log2();\n",
        );
        fixture.write("crates/tessera-protocol/src/f.rs", "let x = 0.5;\n");
        let files = [
            "crates/tessera-tables/src/a.rs",
            "verifier/crates/tv-tables/src/b.rs",
            "crates/tessera-bench/src/c.rs",
            "crates/tessera-tables/src/d.txt",
            "crates/tessera-protocol/src/soundness/e.rs",
            "crates/tessera-protocol/src/f.rs",
        ]
        .map(PathBuf::from);
        let found = check(fixture.path(), &files).expect("the fixture is readable");
        assert_eq!(
            found,
            [
                "crates/tessera-tables/src/a.rs:1: the floating-point literal `1.5`",
                "verifier/crates/tv-tables/src/b.rs:1: the type `f64`",
                "crates/tessera-protocol/src/soundness/e.rs:1: the type `f64`",
            ]
        );
    }
}
