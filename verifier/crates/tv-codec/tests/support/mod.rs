// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Test support shared by the integration tests: the type names of `spec/vectors/codec/README.md`,
//! a decoder for each built from `tv-codec`'s combinators that renders values as the vectors'
//! JSON, and an independent encoder from that JSON back to bytes.
//!
//! The encoder exists only here: it lets the tests check `encode(decode(b)) = b` (INV-TCE-2) and
//! `decode(encode(v)) = v` (INV-TCE-1) without a second decoder sharing the first one's reading.

#![allow(dead_code)]

use tv_codec::{Cursor, DecodeError, DecodeLimits, ErrorCode, decode};
use tv_testkit::Json;

/// A type of the vectors, parsed from its name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ty {
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    Bool,
    Fp,
    Fp2,
    Bytes,
    Digest,
    Ascii,
    Rational,
    Array(Box<Ty>, usize),
    Vec(Box<Ty>),
    Option(Box<Ty>),
    Tuple(Vec<Ty>),
    Map(Box<Ty>, Box<Ty>),
    /// `struct { flag: bool, small: u8, list: Vec<u16>, maybe: Option<i32>, name: Ascii, digest: [u8; 32] }`
    Sample,
    /// `enum { Empty, Point(u32), Box { width: u16, height: u16 }, Nested(Vec<Shape>) }`
    Shape,
}

impl Ty {
    /// Parses a type name of the vectors, including the nesting shorthands `Option^n<T>`,
    /// `[T; N]^n` and `pair^n`.
    pub fn parse(name: &str) -> Ty {
        let mut parser = TyParser { text: name, pos: 0 };
        let ty = parser.ty();
        assert_eq!(parser.pos, name.len(), "trailing text in type {name:?}");
        ty
    }

    /// `min_size(T)`, stated here independently of `tv_codec::size`.
    pub fn min_size(&self) -> u64 {
        match self {
            Ty::U8 | Ty::I8 | Ty::Bool => 1,
            Ty::U16 | Ty::I16 => 2,
            Ty::U32 | Ty::I32 => 4,
            Ty::U64 | Ty::I64 | Ty::Fp => 8,
            Ty::Fp2 | Ty::Rational => 16,
            Ty::Digest => 32,
            Ty::Bytes | Ty::Ascii | Ty::Vec(_) | Ty::Map(..) => 8,
            Ty::Option(_) | Ty::Shape => 2,
            Ty::Array(element, len) => (*len as u64).saturating_mul(element.min_size()),
            Ty::Tuple(fields) => fields
                .iter()
                .fold(0, |sum: u64, field| sum.saturating_add(field.min_size())),
            Ty::Sample => 1 + 1 + 8 + 2 + 8 + 32,
        }
    }
}

struct TyParser<'a> {
    text: &'a str,
    pos: usize,
}

impl TyParser<'_> {
    fn rest(&self) -> &str {
        &self.text[self.pos..]
    }

    fn skip_spaces(&mut self) {
        while self.rest().starts_with(' ') {
            self.pos += 1;
        }
    }

    fn eat(&mut self, token: &str) -> bool {
        self.skip_spaces();
        if self.rest().starts_with(token) {
            self.pos += token.len();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, token: &str) {
        assert!(self.eat(token), "expected {token:?} in {:?}", self.text);
    }

    fn word(&mut self) -> &str {
        self.skip_spaces();
        let start = self.pos;
        while self
            .rest()
            .starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_')
        {
            self.pos += 1;
        }
        &self.text[start..self.pos]
    }

    fn number(&mut self) -> usize {
        let word = self.word();
        word.parse()
            .unwrap_or_else(|_| panic!("expected a number, found {word:?}"))
    }

    fn ty(&mut self) -> Ty {
        if self.eat("(") {
            let mut fields = vec![self.ty()];
            while self.eat(",") {
                fields.push(self.ty());
            }
            self.expect(")");
            return Ty::Tuple(fields);
        }
        if self.eat("[") {
            let element = self.ty();
            self.expect(";");
            let len = self.number();
            self.expect("]");
            let mut ty = Ty::Array(Box::new(element), len);
            if self.eat("^") {
                for _ in 1..self.number() {
                    ty = Ty::Array(Box::new(ty), len);
                }
            }
            return ty;
        }
        let word = self.word().to_owned();
        match word.as_str() {
            "u8" => Ty::U8,
            "u16" => Ty::U16,
            "u32" => Ty::U32,
            "u64" => Ty::U64,
            "i8" => Ty::I8,
            "i16" => Ty::I16,
            "i32" => Ty::I32,
            "i64" => Ty::I64,
            "bool" => Ty::Bool,
            "Fp" => Ty::Fp,
            "Fp2" => Ty::Fp2,
            "Bytes" => Ty::Bytes,
            "Digest" => Ty::Digest,
            "Ascii" => Ty::Ascii,
            "Rational" => Ty::Rational,
            "Sample" => Ty::Sample,
            "Shape" => Ty::Shape,
            "Vec" => {
                self.expect("<");
                let element = self.ty();
                self.expect(">");
                Ty::Vec(Box::new(element))
            }
            "Option" => {
                let levels = if self.eat("^") { self.number() } else { 1 };
                self.expect("<");
                let mut ty = self.ty();
                self.expect(">");
                for _ in 0..levels {
                    ty = Ty::Option(Box::new(ty));
                }
                ty
            }
            "Map" => {
                self.expect("<");
                let key = self.ty();
                self.expect(",");
                let value = self.ty();
                self.expect(">");
                Ty::Map(Box::new(key), Box::new(value))
            }
            "pair" => {
                self.expect("^");
                let mut ty = Ty::U8;
                for _ in 0..self.number() {
                    ty = Ty::Tuple(vec![ty, Ty::U8]);
                }
                ty
            }
            other => panic!("unknown type {other:?} in {:?}", self.text),
        }
    }
}

/// An integer or field element, as the vectors write it: a decimal string.
fn int(value: impl ToString) -> Json {
    Json::String(value.to_string())
}

fn object<const N: usize>(members: [(&str, Json); N]) -> Json {
    Json::Object(
        members
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value))
            .collect(),
    )
}

fn some(value: Option<Json>) -> Json {
    match value {
        None => Json::Null,
        Some(value) => object([("some", value)]),
    }
}

/// Decodes one value of `ty` at the cursor, rendered as the vectors' JSON.
pub fn decode_value(c: &mut Cursor<'_>, ty: &Ty) -> Result<Json, DecodeError> {
    Ok(match ty {
        Ty::U8 => int(c.u8()?),
        Ty::U16 => int(c.u16()?),
        Ty::U32 => int(c.u32()?),
        Ty::U64 => int(c.u64()?),
        Ty::I8 => int(c.i8()?),
        Ty::I16 => int(c.i16()?),
        Ty::I32 => int(c.i32()?),
        Ty::I64 => int(c.i64()?),
        Ty::Bool => Json::Bool(c.bool()?),
        Ty::Fp => int(c.fp()?.to_canonical()),
        Ty::Fp2 => {
            let x = c.fp2()?;
            Json::Array(vec![int(x.c0.to_canonical()), int(x.c1.to_canonical())])
        }
        Ty::Bytes => Json::Array(c.bytes()?.iter().map(int).collect()),
        Ty::Digest => Json::Array(c.digest()?.iter().map(int).collect()),
        Ty::Ascii => Json::String(c.ascii()?.to_owned()),
        Ty::Rational => {
            let r = c.rational()?;
            object([("num", int(r.num())), ("den", int(r.den()))])
        }
        Ty::Array(element, len) => {
            Json::Array(c.array(*len, element.min_size(), |c| decode_value(c, element))?)
        }
        Ty::Vec(element) => Json::Array(c.vec(element.min_size(), |c| decode_value(c, element))?),
        Ty::Option(inner) => some(c.option(|c| decode_value(c, inner))?),
        Ty::Tuple(fields) => Json::Array(c.nested(|c| {
            fields
                .iter()
                .map(|field| decode_value(c, field))
                .collect::<Result<_, _>>()
        })?),
        Ty::Map(key, value) => Json::Array(
            c.map(
                key.min_size(),
                value.min_size(),
                |c| decode_value(c, key),
                |c| decode_value(c, value),
            )?
            .into_iter()
            .map(|(k, v)| Json::Array(vec![k, v]))
            .collect(),
        ),
        Ty::Sample => sample(c)?,
        Ty::Shape => shape(c)?,
    })
}

/// The `Sample` test schema, written as a schema decoder would be: its fields in order.
fn sample(c: &mut Cursor<'_>) -> Result<Json, DecodeError> {
    c.nested(|c| {
        let flag = Json::Bool(c.bool()?);
        let small = int(c.u8()?);
        let list = Json::Array(c.vec(2, |c| c.u16().map(int))?);
        let maybe = some(c.option(|c| c.i32().map(int))?);
        let name = Json::String(c.ascii()?.to_owned());
        let digest = Json::Array(c.digest()?.iter().map(int).collect());
        Ok(object([
            ("flag", flag),
            ("small", small),
            ("list", list),
            ("maybe", maybe),
            ("name", name),
            ("digest", digest),
        ]))
    })
}

/// The `Shape` test schema: four variants, the last one recursive.
fn shape(c: &mut Cursor<'_>) -> Result<Json, DecodeError> {
    c.enumeration(4, |c, tag| {
        Ok(match tag {
            0 => object([("Empty", Json::Null)]),
            1 => object([("Point", Json::Array(vec![int(c.u32()?)]))]),
            2 => {
                let width = int(c.u16()?);
                let height = int(c.u16()?);
                object([("Box", object([("width", width), ("height", height)]))])
            }
            _ => object([("Nested", Json::Array(vec![Json::Array(c.vec(2, shape)?)]))]),
        })
    })
}

/// Decodes a whole input as `ty`.
pub fn decode_whole(ty: &Ty, input: &[u8], limits: &DecodeLimits) -> Result<Json, DecodeError> {
    decode(input, limits, |c| decode_value(c, ty))
}

fn number<T: std::str::FromStr>(value: &Json) -> T {
    let text = value
        .as_str()
        .unwrap_or_else(|| panic!("expected a decimal string, found {value:?}"));
    text.parse()
        .unwrap_or_else(|_| panic!("{text:?} out of range"))
}

fn items(value: &Json) -> &[Json] {
    value
        .as_array()
        .unwrap_or_else(|| panic!("expected an array, found {value:?}"))
}

fn member<'j>(value: &'j Json, key: &str) -> &'j Json {
    value
        .get(key)
        .unwrap_or_else(|| panic!("missing {key:?} in {value:?}"))
}

fn length(len: usize, out: &mut Vec<u8>) {
    out.extend((len as u64).to_le_bytes());
}

fn tag(tag: u16, out: &mut Vec<u8>) {
    out.extend(tag.to_le_bytes());
}

/// Encodes the JSON `value` as `ty`, from RFC-0003's encoding table. Map entries are written in
/// the order given, so a non-canonical order shows up as a byte difference.
pub fn encode(ty: &Ty, value: &Json, out: &mut Vec<u8>) {
    match ty {
        Ty::U8 => out.push(number(value)),
        Ty::U16 => out.extend(number::<u16>(value).to_le_bytes()),
        Ty::U32 => out.extend(number::<u32>(value).to_le_bytes()),
        Ty::U64 | Ty::Fp => out.extend(number::<u64>(value).to_le_bytes()),
        Ty::I8 => out.extend(number::<i8>(value).to_le_bytes()),
        Ty::I16 => out.extend(number::<i16>(value).to_le_bytes()),
        Ty::I32 => out.extend(number::<i32>(value).to_le_bytes()),
        Ty::I64 => out.extend(number::<i64>(value).to_le_bytes()),
        Ty::Bool => match value {
            Json::Bool(flag) => out.push(u8::from(*flag)),
            other => panic!("expected a bool, found {other:?}"),
        },
        Ty::Fp2 => {
            let coefficients = items(value);
            assert_eq!(coefficients.len(), 2, "an Fp2 has two coefficients");
            for coefficient in coefficients {
                encode(&Ty::Fp, coefficient, out);
            }
        }
        Ty::Bytes => encode(&Ty::Vec(Box::new(Ty::U8)), value, out),
        Ty::Digest => encode(&Ty::Array(Box::new(Ty::U8), 32), value, out),
        Ty::Ascii => {
            let text = value
                .as_str()
                .unwrap_or_else(|| panic!("expected a string, found {value:?}"));
            length(text.len(), out);
            out.extend(text.as_bytes());
        }
        Ty::Rational => {
            encode(&Ty::I64, member(value, "num"), out);
            encode(&Ty::U64, member(value, "den"), out);
        }
        Ty::Array(element, len) => {
            let values = items(value);
            assert_eq!(values.len(), *len, "array length");
            for v in values {
                encode(element, v, out);
            }
        }
        Ty::Vec(element) => {
            let values = items(value);
            length(values.len(), out);
            for v in values {
                encode(element, v, out);
            }
        }
        Ty::Option(inner) => match value {
            Json::Null => tag(0, out),
            _ => {
                tag(1, out);
                encode(inner, member(value, "some"), out);
            }
        },
        Ty::Tuple(fields) => {
            let values = items(value);
            assert_eq!(values.len(), fields.len(), "tuple arity");
            for (field, v) in fields.iter().zip(values) {
                encode(field, v, out);
            }
        }
        Ty::Map(key, val) => {
            let entries = items(value);
            length(entries.len(), out);
            for entry in entries {
                let pair = items(entry);
                assert_eq!(pair.len(), 2, "a map entry is a [key, value] pair");
                encode(key, &pair[0], out);
                encode(val, &pair[1], out);
            }
        }
        Ty::Sample => {
            encode(&Ty::Bool, member(value, "flag"), out);
            encode(&Ty::U8, member(value, "small"), out);
            encode(&Ty::Vec(Box::new(Ty::U16)), member(value, "list"), out);
            encode(&Ty::Option(Box::new(Ty::I32)), member(value, "maybe"), out);
            encode(&Ty::Ascii, member(value, "name"), out);
            encode(&Ty::Digest, member(value, "digest"), out);
        }
        Ty::Shape => {
            let Json::Object(members) = value else {
                panic!("expected a Shape object, found {value:?}")
            };
            assert_eq!(members.len(), 1, "an enum value names one variant");
            let (variant, fields) = &members[0];
            match variant.as_str() {
                "Empty" => tag(0, out),
                "Point" => {
                    tag(1, out);
                    encode(&Ty::Tuple(vec![Ty::U32]), fields, out);
                }
                "Box" => {
                    tag(2, out);
                    encode(&Ty::U16, member(fields, "width"), out);
                    encode(&Ty::U16, member(fields, "height"), out);
                }
                "Nested" => {
                    tag(3, out);
                    encode(&Ty::Tuple(vec![Ty::Vec(Box::new(Ty::Shape))]), fields, out);
                }
                other => panic!("unknown Shape variant {other:?}"),
            }
        }
    }
}

/// `encode` into a new vector.
pub fn encoded(ty: &Ty, value: &Json) -> Vec<u8> {
    let mut out = Vec::new();
    encode(ty, value, &mut out);
    out
}

/// JSON equality with objects compared as maps: the vectors sort members by name, while a
/// struct renders in declaration order.
pub fn same(a: &Json, b: &Json) -> bool {
    match (a, b) {
        (Json::Object(x), Json::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| b.get(k).is_some_and(|w| same(v, w)))
        }
        (Json::Array(x), Json::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(v, w)| same(v, w))
        }
        _ => a == b,
    }
}

/// A `limits` object of the vector files.
pub fn limits_of(json: &Json) -> DecodeLimits {
    let field = |key: &str| {
        member(json, key)
            .as_u64()
            .unwrap_or_else(|| panic!("{key} is not a u64"))
    };
    DecodeLimits {
        max_total_bytes: field("max_total_bytes"),
        max_vec_len: field("max_vec_len"),
        max_depth: u16::try_from(field("max_depth")).expect("max_depth fits u16"),
    }
}

/// The error code a vector names.
pub fn code_named(id: &str) -> ErrorCode {
    ErrorCode::ALL
        .into_iter()
        .find(|code| code.id() == id)
        .unwrap_or_else(|| panic!("no error code {id:?}"))
}

/// A string member of a vector case.
pub fn text<'j>(case: &'j Json, key: &str) -> &'j str {
    member(case, key)
        .as_str()
        .unwrap_or_else(|| panic!("{key} is not a string in {case:?}"))
}

/// The array member `key` of a vector file.
pub fn list<'j>(file: &'j Json, key: &str) -> &'j [Json] {
    items(member(file, key))
}
