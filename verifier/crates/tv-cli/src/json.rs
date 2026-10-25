// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A strict reader of the JSON documents `tessera-verify` takes as input: the audit policy, the
//! traffic profile and the cost model (RFC-0016 'Calculators'). It is the CLI's own, so that no
//! parsing crate joins the trusted base: RFC 8259's grammar, one document per input, at most 64
//! levels deep, and no object with a key twice, found by sorting the keys. Numbers keep their text, so integers are read
//! exactly and binary64 values are rounded once, by the standard library.

/// A JSON value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Value {
    Null,
    Bool(bool),
    /// A number, as written.
    Number(String),
    String(String),
    Array(Vec<Value>),
    /// An object's members, in order, each key once.
    Object(Vec<(String, Value)>),
}

const MAX_DEPTH: usize = 64;

impl Value {
    /// Parses one document; anything after it but whitespace is an error.
    pub(crate) fn parse(text: &str) -> Option<Self> {
        let mut parser = Parser {
            bytes: text.as_bytes(),
            pos: 0,
            depth: 0,
        };
        let value = parser.value()?;
        parser.whitespace();
        (parser.pos == parser.bytes.len()).then_some(value)
    }

    /// An object whose keys are exactly `keys`, in any order: its members by `keys`' order.
    pub(crate) fn exactly<const N: usize>(&self, keys: [&str; N]) -> Option<[&Value; N]> {
        let Self::Object(members) = self else {
            return None;
        };
        if members.len() != N {
            return None;
        }
        let mut out = [self; N];
        for (slot, key) in out.iter_mut().zip(keys) {
            *slot = members.iter().find(|(k, _)| k == key).map(|(_, v)| v)?;
        }
        Some(out)
    }

    /// The member `key` of an object.
    pub(crate) fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Self::Object(members) => members.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub(crate) fn as_array(&self) -> Option<&[Value]> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }

    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    pub(crate) fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// A non-negative integer written as digits alone, no sign, fraction or exponent, that fits
    /// `u64`.
    pub(crate) fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Number(n) if n.bytes().all(|b| b.is_ascii_digit()) => n.parse().ok(),
            _ => None,
        }
    }

    /// A number as binary64, rounded once; `None` for one that overflows to infinity.
    pub(crate) fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(n) => n.parse::<f64>().ok().filter(|x| x.is_finite()),
            _ => None,
        }
    }
}

/// Whether no key repeats among `members`: the keys sorted, then neighbours compared, so an
/// object of `n` keys costs `O(n log n)` comparisons, not the `n²` of a scan per key.
fn unique(members: &[(String, Value)]) -> bool {
    let mut keys: Vec<&str> = members.iter().map(|(k, _)| k.as_str()).collect();
    keys.sort_unstable();
    !keys.windows(2).any(|w| matches!(w, [a, b] if a == b))
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
    depth: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.pos += 1;
        Some(b)
    }

    fn whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn literal(&mut self, word: &[u8]) -> Option<()> {
        let end = self.pos.checked_add(word.len())?;
        (self.bytes.get(self.pos..end)? == word).then(|| self.pos = end)
    }

    fn value(&mut self) -> Option<Value> {
        self.whitespace();
        match self.peek()? {
            b'n' => self.literal(b"null").map(|()| Value::Null),
            b't' => self.literal(b"true").map(|()| Value::Bool(true)),
            b'f' => self.literal(b"false").map(|()| Value::Bool(false)),
            b'"' => self.string().map(Value::String),
            b'[' => self.nested(Self::array),
            b'{' => self.nested(Self::object),
            b'-' | b'0'..=b'9' => self.number().map(Value::Number),
            _ => None,
        }
    }

    fn nested(&mut self, f: impl FnOnce(&mut Self) -> Option<Value>) -> Option<Value> {
        if self.depth >= MAX_DEPTH {
            return None;
        }
        self.depth += 1;
        let value = f(self);
        self.depth -= 1;
        value
    }

    fn array(&mut self) -> Option<Value> {
        self.bump()?;
        let mut items = Vec::new();
        self.whitespace();
        if self.peek()? == b']' {
            self.bump();
            return Some(Value::Array(items));
        }
        loop {
            items.push(self.value()?);
            self.whitespace();
            match self.bump()? {
                b',' => continue,
                b']' => return Some(Value::Array(items)),
                _ => return None,
            }
        }
    }

    fn object(&mut self) -> Option<Value> {
        self.bump()?;
        let mut members: Vec<(String, Value)> = Vec::new();
        self.whitespace();
        if self.peek()? == b'}' {
            self.bump();
            return Some(Value::Object(members));
        }
        loop {
            self.whitespace();
            if self.peek()? != b'"' {
                return None;
            }
            let key = self.string()?;
            self.whitespace();
            if self.bump()? != b':' {
                return None;
            }
            let value = self.value()?;
            members.push((key, value));
            self.whitespace();
            match self.bump()? {
                b',' => continue,
                b'}' => return unique(&members).then_some(Value::Object(members)),
                _ => return None,
            }
        }
    }

    /// RFC 8259 §6: `-? (0 | [1-9][0-9]*) (. [0-9]+)? ([eE] [+-]? [0-9]+)?`.
    fn number(&mut self) -> Option<String> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.bump()? {
            b'0' => {}
            b'1'..=b'9' => self.digits(),
            _ => return None,
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            self.some_digits()?;
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            self.some_digits()?;
        }
        let text = core::str::from_utf8(self.bytes.get(start..self.pos)?).ok()?;
        Some(text.to_owned())
    }

    fn digits(&mut self) {
        while self.peek().is_some_and(|b| b.is_ascii_digit()) {
            self.pos += 1;
        }
    }

    fn some_digits(&mut self) -> Option<()> {
        let start = self.pos;
        self.digits();
        (self.pos > start).then_some(())
    }

    /// A string: unescaped UTF-8, no control character, RFC 8259's escapes, and `\u` escapes
    /// that pair surrogates.
    fn string(&mut self) -> Option<String> {
        self.bump()?;
        let mut out = String::new();
        loop {
            let start = self.pos;
            while self
                .peek()
                .is_some_and(|b| b != b'"' && b != b'\\' && b >= 0x20)
            {
                self.pos += 1;
            }
            out.push_str(core::str::from_utf8(self.bytes.get(start..self.pos)?).ok()?);
            match self.bump()? {
                b'"' => return Some(out),
                b'\\' => {
                    let c = match self.bump()? {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => self.unicode()?,
                        _ => return None,
                    };
                    out.push(c);
                }
                _ => return None,
            }
        }
    }

    fn hex4(&mut self) -> Option<u32> {
        let end = self.pos.checked_add(4)?;
        let text = core::str::from_utf8(self.bytes.get(self.pos..end)?).ok()?;
        if !text.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        self.pos = end;
        u32::from_str_radix(text, 16).ok()
    }

    fn unicode(&mut self) -> Option<char> {
        let first = self.hex4()?;
        if (0xD800..0xDC00).contains(&first) {
            self.literal(b"\\u")?;
            let second = self.hex4()?;
            if !(0xDC00..0xE000).contains(&second) {
                return None;
            }
            char::from_u32(0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00))
        } else {
            char::from_u32(first)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_grammar_and_nothing_else() {
        let doc = Value::parse(
            r#" {"a": [0, -1.5e3, 12], "b": {"c": "\u00e9\ud83d\ude00\n"}, "d": [], "e": null, "f": true} "#,
        )
        .unwrap();
        assert!(
            doc.exactly(["a", "b"]).is_none(),
            "exactly() needs every key and no other"
        );
        let a = doc.get("a").and_then(Value::as_array).unwrap();
        assert_eq!(a[0].as_u64(), Some(0));
        assert_eq!(a[1].as_u64(), None);
        assert_eq!(a[1].as_f64(), Some(-1500.0));
        assert_eq!(a[2].as_u64(), Some(12));
        assert_eq!(
            doc.get("b")
                .and_then(|b| b.get("c"))
                .and_then(Value::as_str),
            Some("é😀\n")
        );
        assert_eq!(doc.get("e"), Some(&Value::Null));
        assert_eq!(doc.get("f").and_then(Value::as_bool), Some(true));
        for bad in [
            "",
            "{",
            "[1,]",
            "{\"a\":1,}",
            "{\"a\" 1}",
            "[1] x",
            "\"open",
            "nul",
            "01",
            "1.",
            ".5",
            "1e",
            "+1",
            "[\"\\q\"]",
            "[\"\\ud800\"]",
            "[\"\\udc00\"]",
            "\"\u{1}\"",
            "{\"a\":1,\"a\":2}",
            "NaN",
            "Infinity",
        ] {
            assert_eq!(Value::parse(bad), None, "{bad:?} was accepted");
        }
        let deep = "[".repeat(MAX_DEPTH + 1) + &"]".repeat(MAX_DEPTH + 1);
        assert_eq!(Value::parse(&deep), None);
        let ok = "[".repeat(MAX_DEPTH) + &"]".repeat(MAX_DEPTH);
        assert!(Value::parse(&ok).is_some());
        assert_eq!(
            Value::parse("18446744073709551616").and_then(|v| v.as_u64()),
            None
        );
        assert_eq!(Value::parse("1e400").and_then(|v| v.as_f64()), None);
        let object = Value::parse(r#"{"y": 2, "x": 1}"#).unwrap();
        let [x, y] = object.exactly(["x", "y"]).unwrap();
        assert_eq!((x.as_u64(), y.as_u64()), (Some(1), Some(2)));
        assert!(object.exactly(["x"]).is_none());
        assert!(object.exactly(["x", "z"]).is_none());
    }

    #[test]
    fn an_object_of_many_keys_is_read_in_quasi_linear_time() {
        // 500,000 distinct keys, then the same with the first repeated at the end. A scan per
        // key would make about 1.25·10^11 string comparisons; the sort takes a fraction of a
        // second, and the bound leaves a wide margin for a loaded host.
        let n = 500_000;
        let mut text = String::from("{");
        for i in 0..n {
            if i != 0 {
                text.push(',');
            }
            text.push_str(&format!("\"k{i}\":{i}"));
        }
        let repeated = format!("{text},\"k0\":1}}");
        text.push('}');
        let start = std::time::Instant::now();
        let parsed = Value::parse(&text).unwrap();
        let elapsed = start.elapsed();
        assert!(matches!(&parsed, Value::Object(m) if m.len() == n));
        assert_eq!(parsed.get("k499999").and_then(Value::as_u64), Some(499_999));
        assert_eq!(Value::parse(&repeated), None);
        assert!(elapsed < std::time::Duration::from_secs(30), "{elapsed:?}");
    }
}
