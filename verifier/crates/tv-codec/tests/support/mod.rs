// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Test support shared by the integration tests: the type names of `spec/vectors/codec/README.md`,
//! a decoder for each built from `tv-codec`'s combinators that renders values as the vectors'
//! JSON, and oracles that judge a decoded value without `tv-codec`.
//!
//! The oracles are written from RFC-0003 alone: [`canonical`] encodes a value in its one
//! canonical form, and [`depth`] and [`longest`] measure it against the limits. They let the
//! tests check `canonical(decode(b)) = b` (INV-TCE-2), `decode(canonical(v)) = v` (INV-TCE-1)
//! and the limits on every accepted input, without a second decoder sharing the first one's
//! reading.

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

fn text_of(value: &Json) -> &str {
    value
        .as_str()
        .unwrap_or_else(|| panic!("expected a string, found {value:?}"))
}

/// The variant name and fields of a `Shape` value, `{"Variant": fields}`.
fn variant(value: &Json) -> (&str, &Json) {
    let Json::Object(members) = value else {
        panic!("expected a Shape object, found {value:?}")
    };
    assert_eq!(members.len(), 1, "an enum value names one variant");
    let (name, fields) = &members[0];
    (name.as_str(), fields)
}

/// The Goldilocks modulus `P = 2^64 − 2^32 + 1`, stated here independently of `tv-field`.
const P: u64 = 0xFFFF_FFFF_0000_0001;

/// The canonical encoding of the value `value` of type `ty`, written from RFC-0003's encoding
/// table: a rational in lowest terms, map entries sorted by their keys' encodings. It fails,
/// saying why, when `value` is not a value of `ty`: a zero denominator, two equal keys, a
/// character outside `Ascii`, an `Fp` not below `P`.
///
/// The encoding is computed from the value, never copied from the bytes it was decoded from, so
/// `canonical(decode(b)) = b` holds only if the decoder accepts no second encoding of a value
/// (INV-TCE-2).
pub fn canonical(ty: &Ty, value: &Json) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    write(ty, value, &mut out)?;
    Ok(out)
}

fn length(len: usize, out: &mut Vec<u8>) {
    out.extend((len as u64).to_le_bytes());
}

fn tag(tag: u16, out: &mut Vec<u8>) {
    out.extend(tag.to_le_bytes());
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn write(ty: &Ty, value: &Json, out: &mut Vec<u8>) -> Result<(), String> {
    match ty {
        Ty::U8 => out.push(number(value)),
        Ty::U16 => out.extend(number::<u16>(value).to_le_bytes()),
        Ty::U32 => out.extend(number::<u32>(value).to_le_bytes()),
        Ty::U64 => out.extend(number::<u64>(value).to_le_bytes()),
        Ty::I8 => out.extend(number::<i8>(value).to_le_bytes()),
        Ty::I16 => out.extend(number::<i16>(value).to_le_bytes()),
        Ty::I32 => out.extend(number::<i32>(value).to_le_bytes()),
        Ty::I64 => out.extend(number::<i64>(value).to_le_bytes()),
        Ty::Fp => {
            let x = number::<u64>(value);
            if x >= P {
                return Err(format!("the field element {x} is not below P"));
            }
            out.extend(x.to_le_bytes());
        }
        Ty::Bool => match value {
            Json::Bool(flag) => out.push(u8::from(*flag)),
            other => panic!("expected a bool, found {other:?}"),
        },
        Ty::Fp2 => {
            let coefficients = items(value);
            assert_eq!(coefficients.len(), 2, "an Fp2 has two coefficients");
            for coefficient in coefficients {
                write(&Ty::Fp, coefficient, out)?;
            }
        }
        Ty::Bytes => write(&Ty::Vec(Box::new(Ty::U8)), value, out)?,
        Ty::Digest => write(&Ty::Array(Box::new(Ty::U8), 32), value, out)?,
        Ty::Ascii => {
            let text = text_of(value);
            if let Some(c) = text.chars().find(|c| !(' '..='~').contains(c)) {
                return Err(format!("{c:?} is not printable ASCII"));
            }
            length(text.len(), out);
            out.extend(text.as_bytes());
        }
        Ty::Rational => {
            let num = number::<i64>(member(value, "num"));
            let den = number::<u64>(member(value, "den"));
            if den == 0 {
                return Err(format!("{num}/0 is not a rational"));
            }
            // Lowest terms; |num| / g fits an i64 except for i64::MIN / 1, kept as it is.
            let g = gcd(num.unsigned_abs(), den);
            let reduced = i64::try_from(i128::from(num) / i128::from(g))
                .map_err(|_| format!("{num}/{den} reduces out of range"))?;
            out.extend(reduced.to_le_bytes());
            out.extend((den / g).to_le_bytes());
        }
        Ty::Array(element, len) => {
            let values = items(value);
            assert_eq!(values.len(), *len, "array length");
            for v in values {
                write(element, v, out)?;
            }
        }
        Ty::Vec(element) => {
            let values = items(value);
            length(values.len(), out);
            for v in values {
                write(element, v, out)?;
            }
        }
        Ty::Option(inner) => match value {
            Json::Null => tag(0, out),
            _ => {
                tag(1, out);
                write(inner, member(value, "some"), out)?;
            }
        },
        Ty::Tuple(fields) => {
            let values = items(value);
            assert_eq!(values.len(), fields.len(), "tuple arity");
            for (field, v) in fields.iter().zip(values) {
                write(field, v, out)?;
            }
        }
        Ty::Map(key, val) => {
            let mut entries = items(value)
                .iter()
                .map(|entry| {
                    let pair = items(entry);
                    assert_eq!(pair.len(), 2, "a map entry is a [key, value] pair");
                    Ok((canonical(key, &pair[0])?, canonical(val, &pair[1])?))
                })
                .collect::<Result<Vec<_>, String>>()?;
            entries.sort();
            if let Some(pair) = entries.windows(2).find(|pair| pair[0].0 == pair[1].0) {
                return Err(format!("two entries have the key {:02x?}", pair[0].0));
            }
            length(entries.len(), out);
            for (k, v) in entries {
                out.extend(k);
                out.extend(v);
            }
        }
        Ty::Sample => {
            write(&Ty::Bool, member(value, "flag"), out)?;
            write(&Ty::U8, member(value, "small"), out)?;
            write(&Ty::Vec(Box::new(Ty::U16)), member(value, "list"), out)?;
            write(&Ty::Option(Box::new(Ty::I32)), member(value, "maybe"), out)?;
            write(&Ty::Ascii, member(value, "name"), out)?;
            write(&Ty::Digest, member(value, "digest"), out)?;
        }
        Ty::Shape => match variant(value) {
            ("Empty", _) => tag(0, out),
            ("Point", fields) => {
                tag(1, out);
                write(&Ty::Tuple(vec![Ty::U32]), fields, out)?;
            }
            ("Box", fields) => {
                tag(2, out);
                write(&Ty::U16, member(fields, "width"), out)?;
                write(&Ty::U16, member(fields, "height"), out)?;
            }
            ("Nested", fields) => {
                tag(3, out);
                write(&Ty::Tuple(vec![Ty::Vec(Box::new(Ty::Shape))]), fields, out)?;
            }
            (other, _) => panic!("unknown Shape variant {other:?}"),
        },
    }
    Ok(())
}

/// The nesting depth of `value` as RFC-0003 'Limits' counts it: one level for each value whose
/// encoding holds other values (structs, enums and `Option`, tuples and map entries, arrays,
/// `Vec`, maps, `Ascii`, `Rational` and `Fp2`), none for integers, `bool` and `Fp`.
pub fn depth(ty: &Ty, value: &Json) -> u64 {
    let deepest = |depths: &mut dyn Iterator<Item = u64>| depths.max().unwrap_or(0);
    match ty {
        Ty::U8
        | Ty::U16
        | Ty::U32
        | Ty::U64
        | Ty::I8
        | Ty::I16
        | Ty::I32
        | Ty::I64
        | Ty::Bool
        | Ty::Fp => 0,
        Ty::Fp2 | Ty::Bytes | Ty::Digest | Ty::Ascii | Ty::Rational => 1,
        Ty::Array(element, _) | Ty::Vec(element) => {
            1 + deepest(&mut items(value).iter().map(|v| depth(element, v)))
        }
        Ty::Option(inner) => match value {
            Json::Null => 1,
            _ => 1 + depth(inner, member(value, "some")),
        },
        Ty::Tuple(fields) => {
            1 + deepest(&mut fields.iter().zip(items(value)).map(|(f, v)| depth(f, v)))
        }
        Ty::Map(key, val) => {
            1 + deepest(&mut items(value).iter().map(|entry| {
                let pair = items(entry);
                1 + depth(key, &pair[0]).max(depth(val, &pair[1]))
            }))
        }
        Ty::Sample => {
            let list = depth(&Ty::Vec(Box::new(Ty::U16)), member(value, "list"));
            let maybe = depth(&Ty::Option(Box::new(Ty::I32)), member(value, "maybe"));
            1 + list.max(maybe).max(1)
        }
        // An enum's variant fields take no level of their own.
        Ty::Shape => match variant(value) {
            ("Nested", fields) => 1 + depth(&Ty::Vec(Box::new(Ty::Shape)), &items(fields)[0]),
            _ => 1,
        },
    }
}

/// The length of the longest sequence (`Vec`, `Bytes`, `Ascii` or map) in `value`, which
/// `max_vec_len` bounds.
pub fn longest(ty: &Ty, value: &Json) -> u64 {
    let most = |lengths: &mut dyn Iterator<Item = u64>| lengths.max().unwrap_or(0);
    match ty {
        Ty::Bytes => items(value).len() as u64,
        Ty::Ascii => text_of(value).len() as u64,
        Ty::Vec(element) => {
            let values = items(value);
            most(&mut values.iter().map(|v| longest(element, v))).max(values.len() as u64)
        }
        Ty::Array(element, _) => most(&mut items(value).iter().map(|v| longest(element, v))),
        Ty::Option(inner) => match value {
            Json::Null => 0,
            _ => longest(inner, member(value, "some")),
        },
        Ty::Tuple(fields) => most(&mut fields.iter().zip(items(value)).map(|(f, v)| longest(f, v))),
        Ty::Map(key, val) => {
            let entries = items(value);
            most(&mut entries.iter().map(|entry| {
                let pair = items(entry);
                longest(key, &pair[0]).max(longest(val, &pair[1]))
            }))
            .max(entries.len() as u64)
        }
        Ty::Sample => {
            let list = items(member(value, "list")).len() as u64;
            list.max(text_of(member(value, "name")).len() as u64)
        }
        Ty::Shape => match variant(value) {
            ("Nested", fields) => longest(&Ty::Vec(Box::new(Ty::Shape)), &items(fields)[0]),
            _ => 0,
        },
        _ => 0,
    }
}

/// Where decoding the first `cut` bytes of `value`'s canonical encoding fails, for `cut` below
/// its length: at the first check, in decoding order, that needs more than `cut` bytes, where the
/// bytes it reads begin (RFC-0003 'Limits', order of checks). Each fixed-width value is one
/// check; a sequence is two at its length, the length itself and `length · min_size(T)` bytes
/// after it; an array is one at its first byte for `N · min_size(T)` bytes. Composite values
/// then check their parts in order.
pub fn truncated_at(ty: &Ty, value: &Json, cut: u64) -> u64 {
    let mut checks = Vec::new();
    layout(ty, value, &mut 0, &mut checks);
    checks
        .into_iter()
        .find(|&(_, end)| end > cut)
        .map(|(start, _)| start)
        .unwrap_or_else(|| panic!("no check fails on {cut} bytes of {value:?}"))
}

/// A check that reads `width` bytes at `at`, which then moves past them.
fn fixed(width: u64, at: &mut u64, checks: &mut Vec<(u64, u64)>) {
    checks.push((*at, *at + width));
    *at += width;
}

/// A sequence of `len` values of minimum size `min` at `at`: its length, then the bytes-left
/// check at the length's offset.
fn sequence(len: usize, min: u64, at: &mut u64, checks: &mut Vec<(u64, u64)>) {
    let start = *at;
    fixed(8, at, checks);
    checks.push((start, start + 8 + len as u64 * min.max(1)));
}

/// The checks decoding `value`'s canonical encoding makes, from `at`, as (first byte, end).
fn layout(ty: &Ty, value: &Json, at: &mut u64, checks: &mut Vec<(u64, u64)>) {
    match ty {
        Ty::U8 | Ty::I8 | Ty::Bool => fixed(1, at, checks),
        Ty::U16 | Ty::I16 => fixed(2, at, checks),
        Ty::U32 | Ty::I32 => fixed(4, at, checks),
        Ty::U64 | Ty::I64 | Ty::Fp => fixed(8, at, checks),
        Ty::Fp2 | Ty::Rational => {
            fixed(8, at, checks);
            fixed(8, at, checks);
        }
        Ty::Digest => fixed(32, at, checks),
        Ty::Bytes | Ty::Ascii => {
            let len = match ty {
                Ty::Ascii => text_of(value).len(),
                _ => items(value).len(),
            };
            sequence(len, 1, at, checks);
            *at += len as u64;
        }
        Ty::Array(element, len) => {
            checks.push((*at, *at + *len as u64 * element.min_size()));
            for v in items(value) {
                layout(element, v, at, checks);
            }
        }
        Ty::Vec(element) => {
            let values = items(value);
            sequence(values.len(), element.min_size(), at, checks);
            for v in values {
                layout(element, v, at, checks);
            }
        }
        Ty::Option(inner) => {
            fixed(2, at, checks);
            if *value != Json::Null {
                layout(inner, member(value, "some"), at, checks);
            }
        }
        Ty::Tuple(fields) => {
            for (field, v) in fields.iter().zip(items(value)) {
                layout(field, v, at, checks);
            }
        }
        Ty::Map(key, val) => {
            let entries = items(value);
            sequence(entries.len(), key.min_size() + val.min_size(), at, checks);
            for entry in entries {
                let pair = items(entry);
                layout(key, &pair[0], at, checks);
                layout(val, &pair[1], at, checks);
            }
        }
        Ty::Sample => {
            fixed(1, at, checks);
            fixed(1, at, checks);
            layout(
                &Ty::Vec(Box::new(Ty::U16)),
                member(value, "list"),
                at,
                checks,
            );
            layout(
                &Ty::Option(Box::new(Ty::I32)),
                member(value, "maybe"),
                at,
                checks,
            );
            layout(&Ty::Ascii, member(value, "name"), at, checks);
            fixed(32, at, checks);
        }
        Ty::Shape => {
            fixed(2, at, checks);
            match variant(value) {
                ("Point", _) => fixed(4, at, checks),
                ("Box", _) => {
                    fixed(2, at, checks);
                    fixed(2, at, checks);
                }
                ("Nested", fields) => {
                    layout(&Ty::Vec(Box::new(Ty::Shape)), &items(fields)[0], at, checks)
                }
                _ => {}
            }
        }
    }
}

/// The checks an accepted whole input `input` must pass, computed from the value it decoded to
/// and never from the decoder: it is the canonical encoding of that value (INV-TCE-2), and the
/// value is within `limits` (RFC-0003 'Limits'). Returns what failed.
pub fn accepted_soundly(
    ty: &Ty,
    value: &Json,
    input: &[u8],
    limits: &DecodeLimits,
) -> Result<(), String> {
    match canonical(ty, value) {
        Ok(bytes) if bytes == input => {}
        Ok(bytes) => {
            return Err(format!(
                "not the canonical encoding, {}",
                bytes.iter().map(|b| format!("{b:02x}")).collect::<String>()
            ));
        }
        Err(why) => return Err(format!("not a value of the type: {why}")),
    }
    let max_depth = u64::from(limits.max_depth.min(256));
    if depth(ty, value) > max_depth {
        return Err(format!("depth {} over {max_depth}", depth(ty, value)));
    }
    if longest(ty, value) > limits.max_vec_len {
        return Err(format!(
            "a sequence of {} over {}",
            longest(ty, value),
            limits.max_vec_len
        ));
    }
    if input.len() as u64 > limits.max_total_bytes {
        return Err(format!(
            "{} bytes over {}",
            input.len(),
            limits.max_total_bytes
        ));
    }
    Ok(())
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
