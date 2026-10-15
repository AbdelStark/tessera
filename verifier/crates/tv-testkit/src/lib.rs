// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Test support for the verifier workspace, used only as a dev-dependency.
//!
//! It reads the shared conformance vectors under `spec/vectors/` with a small JSON reader, so
//! that verifier tests add no third-party crate to the verifier's dependency graph, which
//! `allowlist.toml` and `cargo vet` must cover (RFC-0001, DEP-3). It also provides a seeded
//! generator for reproducible randomized tests.

#![forbid(unsafe_code)]
// Test support: a malformed vector file is a test failure, reported by panicking.
#![allow(clippy::panic)]

use std::fs;
use std::path::PathBuf;

/// A JSON value. Numbers keep their source text, so integers of any width survive exactly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Json {
    /// `null`.
    Null,
    /// `true` or `false`.
    Bool(bool),
    /// A number, as written.
    Number(String),
    /// A string, unescaped.
    String(String),
    /// An array.
    Array(Vec<Json>),
    /// An object, in source order.
    Object(Vec<(String, Json)>),
}

impl Json {
    /// Parses one JSON document; anything after it other than whitespace is an error.
    pub fn parse(text: &str) -> Result<Json, String> {
        let mut parser = Parser {
            bytes: text.as_bytes(),
            pos: 0,
            depth: 0,
        };
        let value = parser.value()?;
        parser.whitespace();
        if parser.pos != parser.bytes.len() {
            return Err(format!("trailing data at byte {}", parser.pos));
        }
        Ok(value)
    }

    /// The member `key` of an object.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(members) => members.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// The elements of an array.
    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(items) => Some(items),
            _ => None,
        }
    }

    /// The text of a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(s) => Some(s),
            _ => None,
        }
    }

    /// A non-negative integer that fits `u64`.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Number(n) => n.parse().ok(),
            _ => None,
        }
    }

    /// A `"0x…"` hexadecimal string as `u64`.
    pub fn as_hex_u64(&self) -> Option<u64> {
        u64::from_str_radix(self.as_str()?.strip_prefix("0x")?, 16).ok()
    }

    /// Whether this is `null`.
    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
    depth: usize,
}

const MAX_DEPTH: usize = 64;

impl Parser<'_> {
    fn whitespace(&mut self) {
        while self
            .bytes
            .get(self.pos)
            .is_some_and(|b| b.is_ascii_whitespace())
        {
            self.pos += 1;
        }
    }

    fn expect(&mut self, byte: u8) -> Result<(), String> {
        if self.bytes.get(self.pos) == Some(&byte) {
            self.pos += 1;
            Ok(())
        } else {
            Err(format!("expected `{}` at byte {}", byte as char, self.pos))
        }
    }

    fn literal(&mut self, word: &str, value: Json) -> Result<Json, String> {
        if self.bytes[self.pos..].starts_with(word.as_bytes()) {
            self.pos += word.len();
            Ok(value)
        } else {
            Err(format!("invalid literal at byte {}", self.pos))
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        self.whitespace();
        match self.bytes.get(self.pos) {
            Some(b'{') => self.nested(Self::object),
            Some(b'[') => self.nested(Self::array),
            Some(b'"') => self.string().map(Json::String),
            Some(b'n') => self.literal("null", Json::Null),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'-' | b'0'..=b'9') => Ok(self.number()),
            _ => Err(format!("unexpected input at byte {}", self.pos)),
        }
    }

    fn nested(&mut self, parse: fn(&mut Self) -> Result<Json, String>) -> Result<Json, String> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err("nesting too deep".to_owned());
        }
        let value = parse(self);
        self.depth -= 1;
        value
    }

    fn object(&mut self) -> Result<Json, String> {
        self.expect(b'{')?;
        let mut members = Vec::new();
        self.whitespace();
        if self.bytes.get(self.pos) == Some(&b'}') {
            self.pos += 1;
            return Ok(Json::Object(members));
        }
        loop {
            self.whitespace();
            let key = self.string()?;
            self.whitespace();
            self.expect(b':')?;
            members.push((key, self.value()?));
            self.whitespace();
            match self.bytes.get(self.pos) {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Json::Object(members));
                }
                _ => return Err(format!("expected `,` or `}}` at byte {}", self.pos)),
            }
        }
    }

    fn array(&mut self) -> Result<Json, String> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.whitespace();
        if self.bytes.get(self.pos) == Some(&b']') {
            self.pos += 1;
            return Ok(Json::Array(items));
        }
        loop {
            items.push(self.value()?);
            self.whitespace();
            match self.bytes.get(self.pos) {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Json::Array(items));
                }
                _ => return Err(format!("expected `,` or `]` at byte {}", self.pos)),
            }
        }
    }

    fn number(&mut self) -> Json {
        let start = self.pos;
        while self
            .bytes
            .get(self.pos)
            .is_some_and(|b| matches!(b, b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9'))
        {
            self.pos += 1;
        }
        Json::Number(String::from_utf8_lossy(&self.bytes[start..self.pos]).into_owned())
    }

    fn string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut out = Vec::new();
        loop {
            let byte = *self.bytes.get(self.pos).ok_or("unterminated string")?;
            self.pos += 1;
            match byte {
                b'"' => return String::from_utf8(out).map_err(|e| e.to_string()),
                b'\\' => {
                    let escape = *self.bytes.get(self.pos).ok_or("unterminated escape")?;
                    self.pos += 1;
                    let simple = match escape {
                        b'"' => b'"',
                        b'\\' => b'\\',
                        b'/' => b'/',
                        b'b' => 0x08,
                        b'f' => 0x0c,
                        b'n' => b'\n',
                        b'r' => b'\r',
                        b't' => b'\t',
                        b'u' => {
                            let hex = self
                                .bytes
                                .get(self.pos..self.pos + 4)
                                .ok_or("short \\u escape")?;
                            let code = u32::from_str_radix(&String::from_utf8_lossy(hex), 16)
                                .map_err(|e| e.to_string())?;
                            self.pos += 4;
                            let c =
                                char::from_u32(code).ok_or("\\u escape is not a scalar value")?;
                            let mut buf = [0u8; 4];
                            out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                            continue;
                        }
                        other => return Err(format!("invalid escape `\\{}`", other as char)),
                    };
                    out.push(simple);
                }
                other => out.push(other),
            }
        }
    }
}

/// The repository's `spec/vectors/` directory.
pub fn vectors_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../spec/vectors")
}

/// Reads and parses `spec/vectors/<rel>`, panicking with the path on any failure.
pub fn read_vectors(rel: &str) -> Json {
    let path = vectors_dir().join(rel);
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    Json::parse(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

/// SplitMix64 (Steele, Lea and Flood, 2014): a small seeded generator for reproducible
/// randomized tests. Not for anything secret.
#[derive(Clone, Debug)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    /// A generator starting from `seed`.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_shapes_the_vector_files_use() {
        let doc =
            Json::parse(r#" {"a": [1, "0x10", null, true], "b": {"c": "\u00a7 \"q\""}, "d": []} "#)
                .expect("valid");
        let a = doc.get("a").and_then(Json::as_array).expect("array");
        assert_eq!(a[0].as_u64(), Some(1));
        assert_eq!(a[1].as_hex_u64(), Some(16));
        assert!(a[2].is_null());
        assert_eq!(a[3], Json::Bool(true));
        assert_eq!(
            doc.get("b").and_then(|b| b.get("c")).and_then(Json::as_str),
            Some("§ \"q\"")
        );
        assert_eq!(
            doc.get("d").and_then(Json::as_array).map(<[Json]>::len),
            Some(0)
        );
    }

    #[test]
    fn rejects_malformed_documents() {
        for bad in [
            "",
            "{",
            "[1,",
            "{\"a\" 1}",
            "[1] x",
            "\"open",
            "nul",
            "[\"\\q\"]",
        ] {
            assert!(Json::parse(bad).is_err(), "{bad:?} was accepted");
        }
        let deep = "[".repeat(MAX_DEPTH + 1) + &"]".repeat(MAX_DEPTH + 1);
        assert!(Json::parse(&deep).is_err());
    }

    #[test]
    fn splitmix64_matches_its_reference_outputs() {
        // The published first outputs for seed 0; spec/vectors/field/generate.py uses the same
        // generator with its own seed.
        let mut rng = SplitMix64::new(0);
        assert_eq!(rng.next_u64(), 0xe220_a839_7b1d_cdaf);
        assert_eq!(rng.next_u64(), 0x6e78_9e6a_a1b9_65f4);
    }
}
